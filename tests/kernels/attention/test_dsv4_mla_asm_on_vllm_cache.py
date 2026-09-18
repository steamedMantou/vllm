# SPDX-License-Identifier: Apache-2.0
"""Drive aiter's dsv4_mla_prefill straight off vLLM's paged DSv4 K cache.

The production ROCm prefill path dequantizes the paged fp8 cache into a dense
bf16 buffer and runs a Triton kernel over it. The assembly kernel reads the
paged cache in place, so the question this answers is whether vLLM's cache
layout is already the one the kernel expects:

  block := [ block_size * 576 bytes token data | block_size * 8 bytes scales ]
           token data := 448 bytes fp8 NoPE + 128 bytes bf16 RoPE
           padded up to a multiple of 576

which is exactly what quantize_and_insert_k_cache writes. If so, the dequant +
gather step disappears along with the Triton kernel.

Run directly (not under pytest) for the timing table:
    python tests/kernels/attention/test_dsv4_mla_asm_on_vllm_cache.py
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_prefill_ragged_triton,
)

ROW_BYTES = 576
NOPE_DIM, ROPE_DIM, HEAD_DIM = 448, 64, 512
SCALE_SLOT = 8


def _cache_geometry(block_size: int) -> tuple[int, int, int, int]:
    """(block_bytes, rows_per_page, scale_off, page_shift) for a vLLM K cache."""
    assert block_size & (block_size - 1) == 0, "page size must be a power of two"
    raw = block_size * ROW_BYTES + block_size * SCALE_SLOT
    block_bytes = -(-raw // ROW_BYTES) * ROW_BYTES
    return block_bytes, block_bytes // ROW_BYTES, block_size * ROW_BYTES, block_size.bit_length() - 1


def _paged_views(k_cache: torch.Tensor, rows_per_page: int):
    """The (NoPE, RoPE) strided views dsv4_mla_prefill indexes by cache row."""
    grid = k_cache.shape[0] * rows_per_page
    flat = k_cache.reshape(-1)
    nope = torch.as_strided(
        flat.view(torch.float8_e4m3fn), (grid, HEAD_DIM), (ROW_BYTES, 1)
    )
    rope = torch.as_strided(
        flat.view(torch.bfloat16),
        (grid, ROPE_DIM),
        (ROW_BYTES // 2, 1),
        storage_offset=NOPE_DIM // 2,
    )
    return nope, rope


def _build(seq_len: int, num_q: int, heads: int, block_size: int, nnz: int, seed: int = 0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    block_bytes, rows_per_page, scale_off, page_shift = _cache_geometry(block_size)
    num_blocks = (seq_len + block_size - 1) // block_size

    # A shuffled block table is what proves the row mapping, not just the layout.
    perm = torch.randperm(num_blocks, device=dev, generator=g).to(torch.int32)
    block_table = perm.reshape(1, num_blocks)

    k = (torch.randn(seq_len, HEAD_DIM, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    k_cache = torch.zeros(num_blocks, block_bytes, dtype=torch.uint8, device=dev)
    pos = torch.arange(seq_len, device=dev)
    slot_mapping = (
        perm[pos // block_size].to(torch.int64) * block_size + pos % block_size
    )
    quantize_and_insert_k_cache(k, k_cache, slot_mapping, block_size=block_size)

    q_nope = (torch.randn(num_q, heads, NOPE_DIM, device=dev, generator=g) * 0.125).to(
        torch.bfloat16
    )
    q_rope = (torch.randn(num_q, heads, ROPE_DIM, device=dev, generator=g) * 0.125).to(
        torch.bfloat16
    )
    q = torch.cat([q_nope, q_rope], dim=-1)
    sink = torch.randn(heads, dtype=torch.float32, device=dev, generator=g) * 0.25

    # Logical positions each query attends to, ragged but constant width here.
    local = torch.randint(0, seq_len, (num_q * nnz,), dtype=torch.int32, device=dev, generator=g)
    indptr = torch.arange(0, (num_q + 1) * nnz, nnz, dtype=torch.int32, device=dev)
    # Same positions, expressed as cache rows for the assembly kernel.
    global_idx = (
        perm[local.long() // block_size].to(torch.int32) * block_size
        + (local % block_size)
    )
    return dict(
        k=k, k_cache=k_cache, block_table=block_table, seq_len=seq_len,
        block_size=block_size, rows_per_page=rows_per_page, scale_off=scale_off,
        page_shift=page_shift, q=q, q_nope=q_nope, q_rope=q_rope, sink=sink,
        local=local, indptr=indptr, global_idx=global_idx, heads=heads, num_q=num_q,
    )


def _triton_path(b: dict) -> torch.Tensor:
    """Today's path: dequantize the whole window, then attend over the dense copy."""
    dense = torch.zeros(1, b["seq_len"], HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    dequantize_and_gather_k_cache(
        dense,
        b["k_cache"],
        seq_lens=torch.tensor([b["seq_len"]], dtype=torch.int32, device="cuda"),
        gather_lens=None,
        block_table=b["block_table"],
        block_size=b["block_size"],
        offset=0,
        use_fnuz=False,
    )
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=b["q"], kv=dense.view(-1, HEAD_DIM), indices=b["local"], indptr=b["indptr"],
        scale=1.0 / math.sqrt(HEAD_DIM), attn_sink=b["sink"],
        nope_head_dim=NOPE_DIM, rope_head_dim=ROPE_DIM,
    )


def _asm_path(b: dict, out: torch.Tensor | None = None) -> torch.Tensor:
    from aiter.ops.dsv4_mla_prefill import dsv4_mla_prefill

    nope, rope = _paged_views(b["k_cache"], b["rows_per_page"])
    empty_idx = torch.zeros(0, dtype=torch.int32, device="cuda")
    empty_ptr = torch.zeros(b["num_q"] + 1, dtype=torch.int32, device="cuda")
    if out is None:
        out = torch.empty(b["num_q"], b["heads"], HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    dsv4_mla_prefill(
        q_nope=b["q_nope"], q_rope=b["q_rope"],
        unified_kv_nope=nope, unified_kv_rope=rope,
        kv_indices_prefix=b["global_idx"], kv_indptr_prefix=b["indptr"],
        kv_nope=nope, kv_rope=rope,
        kv_indices_extend=empty_idx, kv_indptr_extend=empty_ptr,
        attn_sink=b["sink"],
        kv_max_e=torch.zeros(1, dtype=torch.int32, device="cuda"),
        softmax_scale=1.0 / math.sqrt(HEAD_DIM), out=out,
        page_shift_prefix=b["page_shift"], rows_per_page_prefix=b["rows_per_page"],
        scale_off_prefix=b["scale_off"],
        page_shift_extend=b["page_shift"], rows_per_page_extend=b["rows_per_page"],
        scale_off_extend=b["scale_off"],
    )
    return out


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="DSv4 assembly prefill is gfx950-only"
)
@pytest.mark.parametrize("block_size", [64, 256])
@pytest.mark.parametrize("heads", [128])
def test_asm_matches_triton_on_vllm_cache(block_size: int, heads: int) -> None:
    b = _build(seq_len=8192, num_q=256, heads=heads, block_size=block_size, nnz=1152)
    ref = _triton_path(b)
    got = _asm_path(b)
    torch.cuda.synchronize()
    # Both read the same fp8 rows; the reference rounds through bf16 first, so
    # they agree to fp8 resolution rather than exactly.
    err = (got.float() - ref.float()).abs().mean().item()
    rel = err / ref.float().abs().mean().item()
    # The reference keeps the second GEMM in bf16 and this kernel does it in
    # fp8, which against an fp32 reference is 2.5% of error versus the Triton
    # path's 0.2%. The bound is set to catch a wrong row mapping, which lands
    # far above it, not to pin the precision difference.
    assert rel < 0.05, f"relative error {rel:.4f} (mean abs {err:.5f})"


def _bench(fn, warmup: int = 5, reps: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def main() -> None:
    print(torch.cuda.get_device_name(0))
    print(f"{'shape':>34} {'Triton path':>12} {'ASM path':>10} {'speedup':>8} {'rel err':>9}")
    for block_size in (64, 256):
        for num_q, nnz in ((1024, 1152), (1024, 512), (8192, 1152)):
            b = _build(seq_len=8192, num_q=num_q, heads=128, block_size=block_size, nnz=nnz)
            ref = _triton_path(b)
            got = _asm_path(b)
            torch.cuda.synchronize()
            rel = ((got.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
            ms_t = _bench(lambda: _triton_path(b))
            ms_a = _bench(lambda: _asm_path(b))
            label = f"page={block_size} T={num_q} nnz={nnz}"
            print(f"{label:>34} {ms_t:10.3f}ms {ms_a:8.3f}ms {ms_t / ms_a:7.2f}x {rel:9.4f}")


if __name__ == "__main__":
    main()
