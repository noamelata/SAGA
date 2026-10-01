import torch
import triton
import triton.language as tl


@triton.jit
def threshold_select_fast_kernel(
    x_ptr,
    out_vals_ptr,
    out_idx_ptr,
    counts_ptr,
    threshold_ptr,  # Pass threshold as tensor ptr instead of scalar
    threshold_stride,  # 0 for scalar (broadcast), 1 for per-row
    D,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast version: 2D grid, read threshold from tensor.
    Supports per-row thresholds (threshold_stride=1) or scalar broadcast (threshold_stride=0).
    """
    row = tl.program_id(0)
    block_id = tl.program_id(1)

    # Load threshold: per-row or scalar depending on stride
    threshold = tl.load(threshold_ptr + row * threshold_stride)

    x_row_ptr = x_ptr + row * D
    out_vals_row_ptr = out_vals_ptr + row * K
    out_idx_row_ptr = out_idx_ptr + row * K
    count_ptr = counts_ptr + row

    block_start = block_id * BLOCK_SIZE
    if block_start >= D:
        return

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    x_vals = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))

    passes = (x_vals > threshold) & mask
    passes_int = passes.to(tl.int32)
    block_count = tl.sum(passes_int)

    if block_count > 0:
        base_pos = tl.atomic_add(count_ptr, block_count)
        local_cumsum = tl.cumsum(passes_int, axis=0)
        out_pos = base_pos + local_cumsum - 1

        write_mask = passes & (out_pos < K) & (out_pos >= 0)
        tl.store(out_vals_row_ptr + out_pos, x_vals, mask=write_mask)
        tl.store(out_idx_row_ptr + out_pos, offsets, mask=write_mask)


_BUFFER_CACHE = {}


def threshold_topk_fast(
    x: torch.Tensor,
    k: int,
    threshold: torch.Tensor,
    block_size: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast Triton threshold selection with buffer cache for CUDA graph compatibility."""
    original_shape = x.shape
    D = original_shape[-1]
    x_flat = x.reshape(-1, D).contiguous()
    N = x_flat.shape[0]
    device = x.device
    dtype = x.dtype

    # Use cached buffers (or allocate on first call for this shape)
    cache_key = (N, k, device, dtype)
    if cache_key not in _BUFFER_CACHE:
        _BUFFER_CACHE[cache_key] = (
            torch.empty(N, k, device=device, dtype=dtype),
            torch.empty(N, k, device=device, dtype=torch.int64),
            torch.empty(N, device=device, dtype=torch.int32),
        )
    out_values, out_indices, counts = _BUFFER_CACHE[cache_key]
    out_values.fill_(float('-inf'))  # GPU kernel, captured by graph
    out_indices.zero_()              # GPU kernel, captured by graph — prevents OOB in gather
    counts.zero_()                   # GPU kernel, captured by graph

    # Support per-row thresholds: shape (N,) -> stride 1, otherwise broadcast to (N,) or scalar
    if threshold.numel() == N:
        thresh_tensor = threshold.contiguous()
        threshold_stride = 1
    elif threshold.numel() == 1:
        thresh_tensor = threshold.view(1).contiguous()
        threshold_stride = 0
    else:
        # Intermediate shape (e.g. per-head): repeat to match N rows
        thresh_tensor = threshold.repeat((N + threshold.numel() - 1) // threshold.numel())[:N].contiguous()
        threshold_stride = 1

    blocks_per_row = (D + block_size - 1) // block_size

    threshold_select_fast_kernel[(N, blocks_per_row)](
        x_flat,
        out_values,
        out_indices,
        counts,
        thresh_tensor,
        threshold_stride,
        D,
        k,
        BLOCK_SIZE=block_size,
    )

    new_shape = original_shape[:-1] + (k,)
    return out_values.reshape(new_shape), out_indices.reshape(new_shape)
