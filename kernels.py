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

    for idx in range(start, end):
        k_block = tl.load(column_indices_ptr + idx)
        base_k = k_block * BLOCK # global K row where this block starts

        for kk in range(0, BLOCK, BLOCK_K):
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
    raise NotImplementedError("TODO: implement sparse_flash_forward (A2)")


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

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement sparse_flash_backward (A3)")
