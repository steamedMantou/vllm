# SPDX-License-Identifier: Apache-2.0
"""DSv4 sparse prefill on aiter's dsv4_mla_prefill assembly kernel.

The Triton path dequantizes the paged fp8 K cache into a dense bf16 window and
attends over that copy. This one hands the kernel the paged cache as written,
because vLLM's block layout is already the one it reads:

    block := [ block_size * 576 B token data | block_size * 8 B scales ]
             token data := 448 B fp8 NoPE + 128 B bf16 RoPE, padded to 576

so the dequantize-and-gather step goes away with the Triton kernel. The kernel
takes two KV sources, which is what the two DSv4 caches need: compressed
history as the "prefix" and the sliding window as the "extend".

Both GEMMs run in fp8, against bf16 for the second one in the Triton path, so
this is roughly a factor of ten more attention error. Opt in with
VLLM_DSV4_PREFILL_ASM=1.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

ROW_BYTES = 576
NOPE_DIM, ROPE_DIM, HEAD_DIM = 448, 64, 512

# One geometry line per cache, so a layout surprise shows up in the log rather
# than as an out-of-bounds view.
_LOGGED: dict[int, bool] = {}


def cache_geometry(k_cache: torch.Tensor, block_size: int) -> tuple[int, int, int]:
    """(page_shift, rows_per_page, scale_off) for a paged DSv4 K cache.

    A page holds ``block_size`` tokens of 576 B and their 8 B scales, padded up
    to a multiple of a row, so it spans more rows than it holds tokens; the
    kernel needs both numbers to turn a token index into a row.
    """
    if block_size & (block_size - 1):
        raise ValueError(f"page size must be a power of two, got {block_size}")
    # A layer's cache is a strided view of one shared pool, so consecutive pages
    # are a whole block stripe apart rather than one layer's worth of bytes. The
    # kernel walks pages in rows, so what it needs is that stride in rows -- the
    # padding every DSv4 spec gets is what keeps it a whole number of them.
    page_stride = k_cache.stride(0) * k_cache.element_size()
    if not _LOGGED.get(id(k_cache)):
        _LOGGED[id(k_cache)] = True
        logger.info(
            "DSv4 ASM prefill cache: shape=%s stride=%s dtype=%s tokens/page=%d "
            "page_stride=%d rows_per_page=%.4f",
            tuple(k_cache.shape), k_cache.stride(), k_cache.dtype, block_size,
            page_stride, page_stride / ROW_BYTES,
        )
    if page_stride % ROW_BYTES:
        raise ValueError(
            f"page stride of {page_stride} B is not a whole number of "
            f"{ROW_BYTES} B rows (shape {tuple(k_cache.shape)}, "
            f"stride {k_cache.stride()}, dtype {k_cache.dtype})"
        )
    return block_size.bit_length() - 1, page_stride // ROW_BYTES, block_size * ROW_BYTES


def paged_views(k_cache: torch.Tensor, rows_per_page: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The NoPE and RoPE planes of a K cache, addressed by cache row.

    Built over the whole pool the layer's view sits in, anchored at that layer's
    own offset, because a row index counts rows of the pool -- one page stride
    apart -- not rows of this layer's slice.
    """
    storage = k_cache.untyped_storage()
    base = k_cache.storage_offset() * k_cache.element_size()
    if base % 2:
        raise ValueError(f"layer base offset {base} B must be even for a bf16 view")
    flat = torch.empty(0, dtype=torch.uint8, device=k_cache.device)
    flat.set_(storage, 0, (storage.nbytes(),), (1,))
    grid = (storage.nbytes() - base) // ROW_BYTES
    nope = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (grid, HEAD_DIM),
        (ROW_BYTES, 1),
        storage_offset=base,
    )
    rope = torch.as_strided(
        flat.view(torch.bfloat16),
        (grid, ROPE_DIM),
        (ROW_BYTES // 2, 1),
        storage_offset=(base + NOPE_DIM) // 2,
    )
    return nope, rope


@triton.jit
def _swa_ragged_kernel(
    out_ptr,
    indptr_ptr,
    pos_ptr,
    token_to_req_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    chunk = tl.program_id(1)
    start = tl.load(indptr_ptr + token_idx)
    length = tl.load(indptr_ptr + token_idx + 1) - start
    offset = chunk * BLOCK + tl.arange(0, BLOCK)
    if chunk * BLOCK >= length:
        return
    mask = offset < length
    req = tl.load(token_to_req_ptr + token_idx)
    pos = tl.load(pos_ptr + token_idx)
    # The window is the `length` positions ending at this token, inclusive.
    seq_pos = pos - length + 1 + offset
    block_numbers = tl.load(
        block_table_ptr + req * block_table_stride + seq_pos // block_size,
        mask=mask,
        other=0,
    )
    rows = block_numbers * block_size + seq_pos % block_size
    tl.store(out_ptr + start + offset, rows, mask=mask)


def build_swa_ragged_indices(
    positions: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache rows of each token's sliding window, as a CSR pair.

    Lengths are analytic -- ``min(pos + 1, window)`` -- so the offsets come from
    a cumsum rather than a counting pass. The buffer is sized at the bound
    rather than the sum, which keeps the whole build off the host: reading the
    sum would sync, and this runs once per layer per chunk.
    """
    num_tokens = positions.numel()
    lens = torch.clamp(positions + 1, max=window_size).to(torch.int32)
    indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=lens.device)
    torch.cumsum(lens, 0, out=indptr[1:])
    out = torch.empty(
        max(num_tokens * window_size, 1), dtype=torch.int32, device=lens.device
    )
    if num_tokens:
        block = 128
        _swa_ragged_kernel[(num_tokens, triton.cdiv(window_size, block))](
            out,
            indptr,
            positions,
            token_to_req,
            block_table,
            block_table.stride(0),
            block_size,
            BLOCK=block,
        )
    return out, indptr


@triton.jit
def _token_meta_kernel(
    pos_ptr,
    req_ptr,
    qsl_ptr,
    seq_lens_ptr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    worker = tl.program_id(1)
    num_workers = tl.num_programs(1)
    base = tl.load(qsl_ptr)
    start = tl.load(qsl_ptr + req) - base
    query_len = tl.load(qsl_ptr + req + 1) - base - start
    # The chunk's last `query_len` positions belong to this request.
    first = tl.load(seq_lens_ptr + req) - query_len
    for i in range(worker * BLOCK, query_len, num_workers * BLOCK):
        offset = i + tl.arange(0, BLOCK)
        mask = offset < query_len
        tl.store(pos_ptr + start + offset, first + offset, mask=mask)
        tl.store(req_ptr + start + offset, req, mask=mask)


def token_positions(
    query_start_loc: torch.Tensor, seq_lens: torch.Tensor, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute position of every query token, and which request it belongs to.

    Done on device: the obvious repeat_interleave spelling needs the repeat
    counts on the host, and this is on the per-layer path.
    """
    dev = query_start_loc.device
    positions = torch.empty(num_tokens, dtype=torch.int32, device=dev)
    token_to_req = torch.empty(num_tokens, dtype=torch.int32, device=dev)
    num_reqs = seq_lens.numel()
    if num_tokens:
        _token_meta_kernel[(num_reqs, 8)](
            positions, token_to_req, query_start_loc, seq_lens, BLOCK=256
        )
    return positions, token_to_req


def asm_sparse_attn_prefill(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    prefix_indices: torch.Tensor | None,
    prefix_indptr: torch.Tensor | None,
    extend_indices: torch.Tensor,
    extend_indptr: torch.Tensor,
    compressed_block_size: int,
    swa_block_size: int,
    attn_sink: torch.Tensor | None,
    scale: float,
    output: torch.Tensor,
) -> None:
    from aiter.ops.dsv4_mla_prefill import dsv4_mla_prefill

    num_q, heads = q.shape[0], q.shape[1]
    dev = q.device
    swa_shift, swa_rows, swa_scale_off = cache_geometry(swa_k_cache, swa_block_size)
    swa_nope, swa_rope = paged_views(swa_k_cache, swa_rows)

    if compressed_k_cache is not None and prefix_indices is not None:
        c_shift, c_rows, c_scale_off = cache_geometry(
            compressed_k_cache, compressed_block_size
        )
        c_nope, c_rope = paged_views(compressed_k_cache, c_rows)
        assert prefix_indptr is not None
    else:
        # Window-only layers still have to pass a prefix; an all-empty CSR is
        # the documented way to say there is none.
        c_nope, c_rope = swa_nope, swa_rope
        c_shift, c_rows, c_scale_off = swa_shift, swa_rows, swa_scale_off
        prefix_indices = torch.zeros(0, dtype=torch.int32, device=dev)
        prefix_indptr = torch.zeros(num_q + 1, dtype=torch.int32, device=dev)

    # The kernel wants a dense bf16 [num_q, heads, HEAD_DIM]. The caller's
    # buffer already is one whenever the padded head count matches, so hand it
    # over directly instead of blitting into it afterwards -- that copy is
    # 48 us per layer at M=1024, the whole attention output moved for nothing.
    # q is read while out is written, so overlapping storage rules it out.
    write_into_output = (
        output.dtype == torch.bfloat16
        and output.is_contiguous()
        and tuple(output.shape) == (num_q, heads, HEAD_DIM)
        and output.data_ptr() != q.data_ptr()
    )
    out = (
        output
        if write_into_output
        else torch.empty(num_q, heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    )
    sink = (
        torch.zeros(heads, dtype=torch.float32, device=dev)
        if attn_sink is None
        else attn_sink[:heads].contiguous()
    )
    dsv4_mla_prefill(
        q_nope=q[..., :NOPE_DIM],
        q_rope=q[..., NOPE_DIM:HEAD_DIM],
        unified_kv_nope=c_nope,
        unified_kv_rope=c_rope,
        kv_indices_prefix=prefix_indices,
        kv_indptr_prefix=prefix_indptr,
        kv_nope=swa_nope,
        kv_rope=swa_rope,
        kv_indices_extend=extend_indices,
        kv_indptr_extend=extend_indptr,
        attn_sink=sink,
        kv_max_e=torch.zeros(1, dtype=torch.int32, device=dev),
        softmax_scale=scale,
        out=out,
        page_shift_prefix=c_shift,
        rows_per_page_prefix=c_rows,
        scale_off_prefix=c_scale_off,
        page_shift_extend=swa_shift,
        rows_per_page_extend=swa_rows,
        scale_off_extend=swa_scale_off,
    )
    if not write_into_output:
        output.copy_(out.to(output.dtype))
