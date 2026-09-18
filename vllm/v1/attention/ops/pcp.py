# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from contextlib import contextmanager
from functools import cache

import torch

from vllm import envs
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
    pcp_comm_ablation_enabled,
    pcp_meta_gather_ablation_enabled,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class _PCPCommGPUTimer:
    """Accumulate per-collective CUDA-event timings for one model forward."""

    def __init__(self) -> None:
        self._active = False
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @property
    def active(self) -> bool:
        return self._active

    def begin(self) -> None:
        if not envs.VLLM_PCP_COMM_GPU_TIMING or not torch.cuda.is_available():
            return
        if self._active:
            self.finish()
        self._events = []
        self._active = True

    @contextmanager
    def scope(self, label: str):
        if not self._active:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events.append((label, start, end))

    def finish(self) -> None:
        if not self._active:
            return
        self._active = False
        if not self._events:
            return
        try:
            # Do this only in the opt-in diagnostic path. It makes event
            # elapsed_time exact even when a ProcessGroup uses another stream.
            torch.cuda.synchronize()
            totals: dict[str, float] = {}
            counts: dict[str, int] = {}
            for label, start, end in self._events:
                totals[label] = totals.get(label, 0.0) + start.elapsed_time(end)
                counts[label] = counts.get(label, 0) + 1
            summary = ", ".join(
                f"{label}={total:.2f}ms/{counts[label]} calls"
                for label, total in sorted(
                    totals.items(), key=lambda item: -item[1]
                )
            )
            logger.info(
                "PCP comm GPU timing: %.2fms across %d calls (%s)",
                sum(totals.values()),
                len(self._events),
                summary,
            )
        except RuntimeError as error:
            logger.warning("PCP comm GPU timing failed: %s", error)
        finally:
            self._events = []


_PCP_COMM_GPU_TIMER = _PCPCommGPUTimer()


def begin_pcp_comm_gpu_timing() -> None:
    _PCP_COMM_GPU_TIMER.begin()


def end_pcp_comm_gpu_timing() -> None:
    _PCP_COMM_GPU_TIMER.finish()


@contextmanager
def pcp_comm_trace(label: str):
    """Emit a ROCTx range around one PCP collective when profiling."""
    emit_nvtx = envs.VLLM_PCP_COMM_NVTX and torch.cuda.is_available()
    if not emit_nvtx and not _PCP_COMM_GPU_TIMER.active:
        yield
        return
    if emit_nvtx:
        torch.cuda.nvtx.range_push(f"PCP_COMM:{label}")
    try:
        with _PCP_COMM_GPU_TIMER.scope(label):
            yield
    finally:
        if emit_nvtx:
            torch.cuda.nvtx.range_pop()


def _gather_dim0(
    group,
    tensor: torch.Tensor,
    is_meta: bool = False,
    trace_label: str | None = None,
) -> torch.Tensor:
    """all_gather along dim 0, or a same-shaped local stand-in when ablating.

    ``is_meta`` marks the descriptors that ride along with a payload — the
    positions, slot mapping and token-to-request map — so the metadata-only
    ablation can drop them while leaving the payload gathers intact.
    """
    if pcp_comm_ablation_enabled() or (is_meta and pcp_meta_gather_ablation_enabled()):
        return tensor.repeat(group.world_size, *([1] * (tensor.dim() - 1)))
    if trace_label is None:
        return group.all_gather(tensor, dim=0)
    with pcp_comm_trace(trace_label):
        return group.all_gather(tensor, dim=0)


def coalesced_pcp_all_gather_payloads(
    tensors: tuple[torch.Tensor, ...],
    trace_label: str,
) -> tuple[torch.Tensor, ...] | None:
    """Batch several independent PCP dim-0 payload all-gathers.

    The individual output tensors remain contiguous and rank-major, matching
    ``GroupCoordinator.all_gather`` exactly. ``None`` means that this runtime
    cannot safely use ProcessGroup coalescing, so callers must retain their
    existing individual gather path.
    """
    if (
        len(tensors) < 2
        or not envs.VLLM_DSV4_C4_COALESCED_ALLGATHER
        or not current_platform.is_rocm()
        or pcp_comm_ablation_enabled()
        # ProcessGroup coalescing is an opaque Python object to Dynamo in the
        # installed ROCm build. Keep this staged operation eager until it has a
        # dedicated registered custom op.
        or torch.compiler.is_compiling()
        or not hasattr(torch.distributed, "_coalescing_manager")
        or not all(
            tensor.is_contiguous()
            and tensor.device == tensors[0].device
            for tensor in tensors
        )
    ):
        return None

    pcp_group = get_pcp_group()
    if pcp_group.world_size <= 1 or not hasattr(pcp_group, "device_group"):
        return None

    outputs = tuple(
        torch.empty(
            (tensor.shape[0] * pcp_group.world_size, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for tensor in tensors
    )
    with pcp_comm_trace(trace_label):
        with torch.distributed._coalescing_manager(  # type: ignore[attr-defined]
            group=pcp_group.device_group,
            device=tensors[0].device,
        ):
            for output, tensor in zip(outputs, tensors):
                torch.distributed.all_gather_into_tensor(
                    output, tensor, group=pcp_group.device_group
                )
    return outputs


@cache
def _pack_descriptors_enabled() -> bool:
    """Escape hatch for descriptor packing (VLLM_PCP_PACK_DESCRIPTORS=0)."""
    return os.getenv("VLLM_PCP_PACK_DESCRIPTORS", "1") != "0"


@cache
def _descriptor_cache_mode() -> str:
    """VLLM_PCP_DESC_CACHE: 0 to gather every layer, verify to check the reuse.

    ``verify`` keeps the collectives -- it costs more than gathering every layer
    -- and asserts each hit is bit-identical to what the wire would have
    returned. It is how the reuse was checked against the real workload rather
    than against the argument for why it holds.
    """
    return os.getenv("VLLM_PCP_DESC_CACHE", "1").lower()


@cache
def _rejoin_copy_enabled() -> bool:
    """Escape hatch for the pure-prefill rejoin skip (VLLM_DSV4_PCP_REJOIN_COPY=1)."""
    return os.getenv("VLLM_DSV4_PCP_REJOIN_COPY", "0") != "0"


def _is_descriptor(tensor: torch.Tensor) -> bool:
    """A descriptor is a 1-D integer map riding along with a payload.

    Everything else in ``tensors`` is payload. The split used to be positional
    -- first entry payload, rest descriptors -- which silently mislabelled
    k_pe, the second cache input on the MLA latent path. Descriptors are a few
    KiB and derivable locally; k_pe is 448 KiB per layer of data only its owning
    rank has. The packing path asserted this contract but the single-descriptor
    path returns before the assert, so the mislabel only showed up as the
    metadata-only ablation pricing k_pe as if it were removable.
    """
    return tensor.ndim == 1 and not tensor.is_floating_point()


def _descriptor_cache(key) -> dict | None:
    """The current forward pass's descriptor store, or None if there is none.

    Keyed on the attention metadata the descriptors came from, which is one
    object shared by every layer in the group, so a hit means the same tensors
    would have been put on the wire again. Outside a forward pass -- unit tests,
    profiling runs -- there is nothing to scope the cache to, so don't.
    """
    if key is None or _descriptor_cache_mode() in ("0", "off"):
        return None
    from vllm.forward_context import is_forward_context_available

    if not is_forward_context_available():
        return None
    from vllm.forward_context import get_forward_context

    return get_forward_context().pcp_descriptor_cache


def _gather_descriptors(
    group, descriptors: list[torch.Tensor], cache_key=None, trace_label: str | None = None
) -> tuple[torch.Tensor, ...]:
    """One collective for the integer descriptors riding with a payload.

    Positions, slot mappings and token-to-request maps are a few KB each, so a
    collective per tensor buys a launch and a round of latency for almost no
    bytes. Widening them to a common dtype and stacking lets one all_gather
    carry the lot.

    They also describe the chunk rather than the layer, and every layer in an
    attention group is handed the same metadata object, so all 61 layers were
    gathering byte-identical inputs. ``cache_key`` identifies that object; the
    result is reused for the rest of the forward pass.
    """
    if not descriptors:
        return ()

    cache = _descriptor_cache(cache_key)
    entry = None
    if cache is not None:
        # Shapes and dtypes go in the key so two call sites sharing a metadata
        # object cannot collide, and the entry holds the key object itself so
        # its id cannot be recycled onto a different metadata while it lives.
        entry = (
            id(cache_key),
            tuple((tuple(d.shape), d.dtype) for d in descriptors),
        )
        hit = cache.get(entry)
        if hit is not None:
            if _descriptor_cache_mode() != "verify":
                return hit[1]
            _assert_reuse_holds(
                hit[1],
                _gather_descriptors_uncached(group, descriptors, trace_label),
            )
            return hit[1]

    result = _gather_descriptors_uncached(group, descriptors, trace_label)
    if cache is not None:
        cache[entry] = (cache_key, result)
    return result


def _gather_descriptors_uncached(
    group, descriptors: list[torch.Tensor], trace_label: str | None = None
) -> tuple[torch.Tensor, ...]:
    if len(descriptors) == 1 or not _pack_descriptors_enabled():
        return tuple(
            _gather_dim0(
                group,
                d,
                is_meta=True,
                trace_label=(
                    None if trace_label is None else f"{trace_label}:descriptor_{i}"
                ),
            )
            for i, d in enumerate(descriptors)
        )
    assert all(d.ndim == 1 and not d.is_floating_point() for d in descriptors), (
        "gather_prefill_cache_inputs takes the payload first and 1-D integer "
        "descriptors after it"
    )
    packed = torch.stack([d.to(torch.int64) for d in descriptors], dim=1)
    gathered = _gather_dim0(
        group,
        packed,
        is_meta=True,
        trace_label=(
            None if trace_label is None else f"{trace_label}:descriptors_packed"
        ),
    )
    return tuple(
        gathered[:, i].to(d.dtype).contiguous() for i, d in enumerate(descriptors)
    )


def _assert_reuse_holds(
    cached: tuple[torch.Tensor, ...], fresh: tuple[torch.Tensor, ...]
) -> None:
    for i, (a, b) in enumerate(zip(cached, fresh)):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            raise AssertionError(
                f"PCP descriptor {i} changed within a forward pass: cached "
                f"{tuple(a.shape)}/{a.dtype} vs gathered {tuple(b.shape)}/{b.dtype}, "
                f"{int((a != b).sum()) if a.shape == b.shape else 'shape'} rows differ"
            )


def gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    descriptor_key=None,
    trace_label: str | None = None,
    gathered_payload_0: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Keep replicated decode writes local and gather partitioned prefills.

    ``descriptor_key`` is the attention metadata the descriptors were read from.
    Pass it and the descriptor collective runs once per forward pass instead of
    once per layer; the payload gather is per-layer either way.

    ``gathered_payload_0`` is an already-collected rank-major prefill suffix
    for the first payload. It lets independent cache writers batch their large
    payload all-gathers while retaining this function's descriptor handling and
    decode-prefix rejoin semantics.
    """
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= local_num_tokens

    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    pcp_group = get_pcp_group()
    pcp_size = pcp_group.world_size
    # ROCm DSV4 fused insertion consumes a rank-local map, so its prefill suffix
    # has to be gathered alongside the tensors; replicated decode rows remain
    # local and therefore writable in every process's private KV cache. Every
    # other caller already hands in a gathered map.
    slots_are_local = slot_mapping.shape[0] == local_num_tokens

    # Classify by the documented contract rather than by position, so a second
    # payload (k_pe) is gathered as one. Issue order is unchanged: payloads in
    # argument order, then the one packed descriptor collective.
    suffixes = [t[num_decode_tokens:].contiguous() for t in tensors[1:]]
    payload_at = [i for i, t in enumerate(suffixes) if not _is_descriptor(t)]
    descriptor_at = [i for i, t in enumerate(suffixes) if _is_descriptor(t)]

    descriptors = [suffixes[i] for i in descriptor_at]
    if slots_are_local:
        descriptors.append(slot_mapping[num_decode_tokens:].contiguous())

    def trace(suffix: str) -> str | None:
        return None if trace_label is None else f"{trace_label}:{suffix}"

    primary_suffix = tensors[0][num_decode_tokens:].contiguous()
    if gathered_payload_0 is None:
        head = _gather_dim0(
            pcp_group,
            primary_suffix,
            trace_label=trace("payload_0"),
        )
    else:
        expected_shape = (
            primary_suffix.shape[0] * pcp_size,
            *primary_suffix.shape[1:],
        )
        assert (
            gathered_payload_0.shape == expected_shape
            and gathered_payload_0.dtype == primary_suffix.dtype
            and gathered_payload_0.device == primary_suffix.device
            and gathered_payload_0.is_contiguous()
        ), (
            "pre-gathered PCP cache payload mismatch: "
            f"got {tuple(gathered_payload_0.shape)}/{gathered_payload_0.dtype}, "
            f"expected {expected_shape}/{primary_suffix.dtype}"
        )
        head = gathered_payload_0
    extra_payloads = [
        _gather_dim0(
            pcp_group,
            suffixes[i],
            trace_label=trace(f"payload_{i + 1}"),
        )
        for i in payload_at
    ]
    gathered_descriptors = _gather_descriptors(
        pcp_group,
        descriptors,
        cache_key=descriptor_key,
        trace_label=trace_label,
    )

    rest: list[torch.Tensor] = [None] * len(suffixes)  # type: ignore[list-item]
    for i, g in zip(payload_at, extra_payloads):
        rest[i] = g
    for i, g in zip(descriptor_at, gathered_descriptors):
        rest[i] = g
    gathered = (head, *rest) + (
        (gathered_descriptors[-1],) if slots_are_local else ()
    )

    def rejoin(local: torch.Tensor, gathered_prefill: torch.Tensor) -> torch.Tensor:
        # A pure prefill step has no replicated decode rows to put back in
        # front, and torch.cat onto an empty leading slice still copies the
        # whole gather. At 100k ISL that copy runs once per gathered tensor
        # per layer.
        if num_decode_tokens == 0 and not _rejoin_copy_enabled():
            return gathered_prefill
        return torch.cat((local[:num_decode_tokens], gathered_prefill), dim=0)

    if slots_are_local:
        cache_inputs = tuple(
            rejoin(t, g) for t, g in zip(tensors, gathered[: len(tensors)])
        )
        return cache_inputs, rejoin(slot_mapping, gathered[-1])

    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if num_decode_tokens == 0:
        return gathered[: len(tensors)], gathered_slot_mapping

    cache_inputs = tuple(
        rejoin(t, g) for t, g in zip(tensors, gathered[: len(tensors)])
    )
    rank_slot_mappings = gathered_slot_mapping.view(pcp_size, local_num_tokens)
    cache_slot_mapping = torch.cat(
        (
            rank_slot_mappings[0, :num_decode_tokens],
            rank_slot_mappings[:, num_decode_tokens:].flatten(),
        )
    )
    return cache_inputs, cache_slot_mapping


def maybe_gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = gather_prefill_cache_inputs(
        (kv_c_normed, k_pe_flat),
        slot_mapping,
        num_decode_tokens,
        trace_label="mla_latent_cache",
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    # No descriptor_key: the indexer is handed an already-gathered slot map, so
    # it has no descriptors to gather in the first place.
    (cache_k,), cache_slot_mapping = gather_prefill_cache_inputs(
        (k,),
        slot_mapping,
        num_decode_tokens,
        trace_label="indexer_cache",
    )
    return cache_k, cache_slot_mapping


def finalize_mla_pcp_decode(
    output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if output.shape[1] < num_heads:
        output = get_pcp_group().all_gather(output, dim=1)
    elif output.shape[1] > num_heads:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output
