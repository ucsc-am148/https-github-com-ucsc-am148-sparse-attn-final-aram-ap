"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

  dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
  sparse_flash_forward   (A2) block-sparse flash attention forward
  sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dsd_matmul_kernel(
    values_ptr,          # (nnz, block, block)  fp32 - A's live blocks, row-major
    row_offsets_ptr,     # (M//block + 1,)      int32 per block-row prefix sum of nnz
    column_indices_ptr,  # (nnz,)               int32 K-block of each live block
    B_ptr,               # (K, N)               fp32  dense right operand
    C_ptr,               # (M, N)               fp32  output
    N,
    stride_vb, stride_vr, stride_vc,   # values: block, within-block row, within-block col
    stride_bk, stride_bn,              # B: row (K), col (N)
    stride_cm, stride_cn,              # C: row (M), col (N)
    BLOCK: tl.constexpr,               # BCSR block size
    BLOCK_M: tl.constexpr,             # output row-tile (divides BLOCK)
    BLOCK_N: tl.constexpr,             # N-tile width
    BLOCK_K: tl.constexpr,             # divides BLOCK
):
    pid_m = tl.program_id(0)   # output row-tile
    pid_n = tl.program_id(1)   # N-column tile

    rows_per_block = BLOCK // BLOCK_M
    block_row = pid_m // rows_per_block                 # which BCSR block-row
    row0 = (pid_m % rows_per_block) * BLOCK_M           # first within-block row

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # (BLOCK_M,) global C rows
    offs_vr = row0 + tl.arange(0, BLOCK_M)              # (BLOCK_M,) rows inside the block
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # (BLOCK_N,) cols of C/B
    offs_kk = tl.arange(0, BLOCK_K)                     # (BLOCK_K,) within a contraction chunk
    n_mask = offs_n < N

    # column_indices[start:end]
    start = tl.load(row_offsets_ptr + block_row)
    end = tl.load(row_offsets_ptr + block_row + 1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for idx in tl.range(start, end):
        k_block = tl.load(column_indices_ptr + idx)
        base_k = k_block * BLOCK # global K row where this block starts

        for kk in tl.range(0, BLOCK, BLOCK_K): # Note: Tried tl.static_range; it blew up -- don't do it
            ck = kk + offs_kk # within-block K columns

            # A sub-tile: values[idx, offs_vr, ck] -> (BLOCK_M, BLOCK_K)
            a_ptrs = (values_ptr
                      + idx * stride_vb
                      + offs_vr[:, None] * stride_vr
                      + ck[None, :] * stride_vc)
            a = tl.load(a_ptrs)

            # B sub-tile: B[base_k + ck, offs_n] -> (BLOCK_K, BLOCK_N)
            gk = base_k + ck # global K rows (always < K)
            b_ptrs = (B_ptr
                      + gk[:, None] * stride_bk
                      + offs_n[None, :] * stride_bn)
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)

            acc += tl.dot(a, b, allow_tf32=False)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=n_mask[None, :])


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.

    Inputs:
      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32   dense right operand
      M, K, N, block                      ints   dims and block size
    Returns:
      C              (M, N)               fp32

    fp32 throughout, allow_tf32=False.
    """
    BLOCK_M = min(64, block)
    BLOCK_N = 64 # Tried 16, 32, 64, 128; 64 ended up having the best average time on my 3090
    BLOCK_K = min(32, block)

    C = torch.empty((M, N), device=B.device, dtype=torch.float32)
    grid = (M // BLOCK_M, triton.cdiv(N, BLOCK_N))
    _dsd_matmul_kernel[grid](
        values, row_offsets, column_indices, B, C,
        N,
        values.stride(0), values.stride(1), values.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK=block, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=3, num_warps=4, # Got these values just trying different combinations till I got something I liked
    )
    return C

@triton.jit
def _sparse_flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    q_row_offsets_ptr,   # (n+1,)  int32  query-block view prefix sums
    q_col_indices_ptr,   # (nnz,)  int32  live key blocks per query block
    sm_scale,
    T,
    stride_z, stride_t, stride_d,   # shared (B*H, T, d) layout for Q/K/V/O
    stride_lz, stride_lt,           # (B*H, T) layout for L
    BLOCK_Q: tl.constexpr,          # query rows per program (== block)
    BLOCK_K: tl.constexpr,          # key rows per live block (== block)
    D: tl.constexpr,                # head dim d
):
    pid_q = tl.program_id(0)   # query block i
    pid_z = tl.program_id(1)   # flattened (batch, head)
    LOG2E = 1.4426950458889634

    qkv_base = pid_z * stride_z          # start of this (b, h) plane in Q/K/V/O

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)   # (BLOCK_Q,) global query rows
    offs_d = tl.arange(0, D)                           # (D,) head-dim lane
    offs_k = tl.arange(0, BLOCK_K)                     # (BLOCK_K,) within-key-block lane
    q_mask = offs_q < T

    # Q_i -> (BLOCK_Q, D)
    q_ptrs = Q_ptr + qkv_base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)   # running row max
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)                 # running denominator
    acc = tl.zeros((BLOCK_Q, D), dtype=tl.float32)               # running output

    start = tl.load(q_row_offsets_ptr + pid_q)
    end = tl.load(q_row_offsets_ptr + pid_q + 1)

    for idx in range(start, end):
        j = tl.load(q_col_indices_ptr + idx)
        offs_kj = j * BLOCK_K + offs_k          # (BLOCK_K,) global key rows
        kj_mask = offs_kj < T

        # Load K_j, V_j -> (BLOCK_K, D)
        k_ptrs = K_ptr + qkv_base + offs_kj[:, None] * stride_t + offs_d[None, :] * stride_d
        v_ptrs = V_ptr + qkv_base + offs_kj[:, None] * stride_t + offs_d[None, :] * stride_d
        k = tl.load(k_ptrs, mask=kj_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=kj_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k), allow_tf32=False) * (sm_scale * LOG2E)

        qk = tl.where(kj_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)                 # rescale factor for old state
        p = tl.exp2(qk - m_new[:, None])             # (BLOCK_Q, BLOCK_K)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, allow_tf32=False)
        m_i = m_new

    o_i = acc / l_i[:, None]
    L_i = m_i + tl.log2(l_i)

    o_ptrs = O_ptr + qkv_base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    tl.store(o_ptrs, o_i.to(O_ptr.dtype.element_ty), mask=q_mask[:, None])

    l_ptrs = L_ptr + pid_z * stride_lz + offs_q * stride_lt
    tl.store(l_ptrs, L_i, mask=q_mask)


def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.

    Inputs:
      Q, K, V        (B, H, T, d)         fp16
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query
      q_col_indices  (nnz,)               int32  block i, its live key blocks j
      sm_scale       float                       1/sqrt(d)
      BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
      O              (B, H, T, d)         fp16
      L              (B, H, T)            fp32   log2 of the softmax denominator (sec 3)

    See ALGORITHMS.md sec 3 for O and L.
    """
    B, H, T, d = Q.shape

    Qf = Q.reshape(B * H, T, d)
    Kf = K.reshape(B * H, T, d)
    Vf = V.reshape(B * H, T, d)

    O = torch.empty_like(Q)
    L = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)
    Of = O.reshape(B * H, T, d)
    Lf = L.reshape(B * H, T)

    # Bigger tiles need more warps to hide their longer matmul/load latency; the
    # 64x64 tile is fastest with 4
    num_warps = 8 if BLOCK_Q >= 128 else 4

    # One program per (query block, head-plane)
    grid = (T // BLOCK_Q, B * H)
    _sparse_flash_fwd_kernel[grid](
        Qf, Kf, Vf, Of, Lf,
        q_row_offsets, q_col_indices,
        sm_scale,
        T,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Lf.stride(0), Lf.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, D=d,
        num_warps=num_warps, num_stages=2,
    )
    return O, L


@triton.jit
def _sparse_flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, Delta_ptr, L_ptr, dO_ptr, dK_ptr, dV_ptr,
    k_row_offsets_ptr,   # (n+1,)  int32  key-block view prefix sums
    k_col_indices_ptr,   # (nnz,)  int32  query blocks i that attend each key block j
    sm_scale,
    T,
    stride_z, stride_t, stride_d,   # shared (B*H, T, d) layout for Q/K/V/dO/dK/dV
    stride_lz, stride_lt,           # (B*H, T) layout for L and Delta
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
):
    pid_k = tl.program_id(0)   # key block j
    pid_z = tl.program_id(1)   # flattened (batch, head)
    LOG2E = 1.4426950408889634

    qkv_base = pid_z * stride_z

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)   # (BLOCK_K,) global key rows
    offs_d = tl.arange(0, D)
    offs_q = tl.arange(0, BLOCK_Q)                     # within-query-block lane
    k_mask = offs_k < T

    # K_j, V_j -> (BLOCK_K, D); held across the query loop
    kj_ptrs = K_ptr + qkv_base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
    vj_ptrs = V_ptr + qkv_base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
    k_j = tl.load(kj_ptrs, mask=k_mask[:, None], other=0.0)
    v_j = tl.load(vj_ptrs, mask=k_mask[:, None], other=0.0)

    dk = tl.zeros((BLOCK_K, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_K, D), dtype=tl.float32)

    start = tl.load(k_row_offsets_ptr + pid_k)
    end = tl.load(k_row_offsets_ptr + pid_k + 1)

    for idx in range(start, end):
        i = tl.load(k_col_indices_ptr + idx)
        offs_qi = i * BLOCK_Q + offs_q
        qi_mask = offs_qi < T

        q_ptrs = Q_ptr + qkv_base + offs_qi[:, None] * stride_t + offs_d[None, :] * stride_d
        do_ptrs = dO_ptr + qkv_base + offs_qi[:, None] * stride_t + offs_d[None, :] * stride_d
        q_i = tl.load(q_ptrs, mask=qi_mask[:, None], other=0.0)
        do_i = tl.load(do_ptrs, mask=qi_mask[:, None], other=0.0)
        l_i = tl.load(L_ptr + pid_z * stride_lz + offs_qi * stride_lt,
                      mask=qi_mask, other=0.0)
        D_i = tl.load(Delta_ptr + pid_z * stride_lz + offs_qi * stride_lt,
                      mask=qi_mask, other=0.0)

        # P_ij = exp2(LOG2E * sm_scale * Q_i.K_j^T - L_i)   (BLOCK_Q, BLOCK_K)
        s = tl.dot(q_i, tl.trans(k_j), allow_tf32=False) * (sm_scale * LOG2E)
        p = tl.exp2(s - l_i[:, None])
        p = tl.where(qi_mask[:, None] & k_mask[None, :], p, 0.0)

        # dV_j += P^T @ dO_i
        dv += tl.dot(tl.trans(p).to(do_i.dtype), do_i, allow_tf32=False)

        # dP_ij = dO_i @ V_j^T                (BLOCK_Q, BLOCK_K)
        dp = tl.dot(do_i, tl.trans(v_j), allow_tf32=False)
        # dS_ij = P_ij * (dP_ij - D_i)
        ds = p * (dp - D_i[:, None])
        # dK_j += dS^T @ Q_i
        dk += tl.dot(tl.trans(ds).to(q_i.dtype), q_i, allow_tf32=False)

    dk = dk * sm_scale

    dk_ptrs = dK_ptr + qkv_base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
    dv_ptrs = dV_ptr + qkv_base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d
    tl.store(dk_ptrs, dk.to(dK_ptr.dtype.element_ty), mask=k_mask[:, None])
    tl.store(dv_ptrs, dv.to(dV_ptr.dtype.element_ty), mask=k_mask[:, None])


@triton.jit
def _sparse_flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, Delta_ptr, L_ptr, dO_ptr, dQ_ptr,
    q_row_offsets_ptr,   # (n+1,)  int32  query-block view prefix sums
    q_col_indices_ptr,   # (nnz,)  int32  live key blocks j per query block i
    sm_scale,
    T,
    stride_z, stride_t, stride_d,
    stride_lz, stride_lt,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
):
    pid_q = tl.program_id(0)   # query block i
    pid_z = tl.program_id(1)   # flattened (batch, head)
    LOG2E = 1.4426950408889634

    qkv_base = pid_z * stride_z

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)   # (BLOCK_Q,) global query rows
    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_K)                      # within-key-block lane
    q_mask = offs_q < T

    q_ptrs = Q_ptr + qkv_base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    do_ptrs = dO_ptr + qkv_base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    q_i = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)
    l_i = tl.load(L_ptr + pid_z * stride_lz + offs_q * stride_lt,
                  mask=q_mask, other=0.0)
    D_i = tl.load(Delta_ptr + pid_z * stride_lz + offs_q * stride_lt,
                  mask=q_mask, other=0.0)

    dq = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    start = tl.load(q_row_offsets_ptr + pid_q)
    end = tl.load(q_row_offsets_ptr + pid_q + 1)

    for idx in range(start, end):
        j = tl.load(q_col_indices_ptr + idx)
        offs_kj = j * BLOCK_K + offs_k
        kj_mask = offs_kj < T

        k_ptrs = K_ptr + qkv_base + offs_kj[:, None] * stride_t + offs_d[None, :] * stride_d
        v_ptrs = V_ptr + qkv_base + offs_kj[:, None] * stride_t + offs_d[None, :] * stride_d
        k_j = tl.load(k_ptrs, mask=kj_mask[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=kj_mask[:, None], other=0.0)

        s = tl.dot(q_i, tl.trans(k_j), allow_tf32=False) * (sm_scale * LOG2E)
        p = tl.exp2(s - l_i[:, None])
        p = tl.where(q_mask[:, None] & kj_mask[None, :], p, 0.0)

        dp = tl.dot(do_i, tl.trans(v_j), allow_tf32=False)
        ds = p * (dp - D_i[:, None])
        # dQ_i += dS @ K_j
        dq += tl.dot(ds.to(k_j.dtype), k_j, allow_tf32=False)

    dq = dq * sm_scale

    dq_ptrs = dQ_ptr + qkv_base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    tl.store(dq_ptrs, dq.to(dQ_ptr.dtype.element_ty), mask=q_mask[:, None])


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.

    Inputs:
      Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
      L              (B, H, T)            fp32   the forward residual
      k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
      k_col_indices  (nnz,)               int32  the query blocks i that attend it
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
      q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
      sm_scale       float
      BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
      dQ, dK, dV     (B, H, T, d)         fp16

    See ALGORITHMS.md sec 4 for the five gradient equations.
    """
    B, H, T, d = Q.shape

    Qf = Q.reshape(B * H, T, d)
    Kf = K.reshape(B * H, T, d)
    Vf = V.reshape(B * H, T, d)
    dOf = dO.reshape(B * H, T, d)
    Lf = L.reshape(B * H, T)

    Delta = (dO.to(torch.float32) * O.to(torch.float32)).sum(-1)   # (B, H, T) fp32
    Deltaf = Delta.reshape(B * H, T)

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    dQf = dQ.reshape(B * H, T, d)
    dKf = dK.reshape(B * H, T, d)
    dVf = dV.reshape(B * H, T, d)

    num_warps = 8 if BLOCK_K >= 128 else 4

    # dK, dV
    grid_kv = (T // BLOCK_K, B * H)
    _sparse_flash_bwd_dkdv_kernel[grid_kv](
        Qf, Kf, Vf, Deltaf, Lf, dOf, dKf, dVf,
        k_row_offsets, k_col_indices,
        sm_scale,
        T,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Lf.stride(0), Lf.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, D=d,
        num_warps=num_warps, num_stages=2,
    )

    # dQ
    grid_q = (T // BLOCK_Q, B * H)
    _sparse_flash_bwd_dq_kernel[grid_q](
        Qf, Kf, Vf, Deltaf, Lf, dOf, dQf,
        q_row_offsets, q_col_indices,
        sm_scale,
        T,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Lf.stride(0), Lf.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, D=d,
        num_warps=num_warps, num_stages=2,
    )

    return dQ, dK, dV
