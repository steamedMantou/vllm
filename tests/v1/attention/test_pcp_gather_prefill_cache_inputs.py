# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP cache gathers preserve rank-major payload and packed descriptors.

The positions / slot mapping / token-to-request maps that describe a payload
are a few KB each, so they are widened to a common dtype and stacked into a
single collective. C4 can additionally pre-gather two independent payloads in
one ProcessGroup batch; each compressor still receives its own contiguous,
rank-major output.
"""

import types

import pytest
import torch

from vllm.v1.attention.ops import pcp


class FakePCPGroup:
    """Stands in for the PCP group, counting the collectives it is asked for.

    A real all_gather would return each rank's own slice; here only one process
    exists, so rank r's contribution is synthesized by offsetting the local
    tensor. That keeps rank-major ordering bugs visible in the output.
    """

    def __init__(self, world_size: int = 4):
        self.world_size = world_size
        self.device_group = object()
        self.calls: list[tuple[torch.Size, torch.dtype]] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0, "gather_prefill_cache_inputs only gathers along dim 0"
        self.calls.append((tensor.shape, tensor.dtype))
        parts = []
        for rank in range(self.world_size):
            part = tensor.clone()
            if not part.is_floating_point():
                part += rank * 1000
            else:
                part += rank
            parts.append(part)
        return torch.cat(parts, dim=0)


def reference_impl(group, tensors, slot_mapping, num_decode_tokens):
    """The prior behaviour: one collective per tensor, none of them packed."""
    local_num_tokens = tensors[0].shape[0]
    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    gathered_prefills = tuple(
        group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
        for tensor in tensors
    )
    pcp_size = group.world_size
    if slot_mapping.shape[0] == local_num_tokens:
        gathered_prefill_slots = group.all_gather(
            slot_mapping[num_decode_tokens:].contiguous(), dim=0
        )
        cache_slot_mapping = torch.cat(
            (slot_mapping[:num_decode_tokens], gathered_prefill_slots), dim=0
        )
        cache_inputs = tuple(
            torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
            for tensor, gathered_prefill in zip(tensors, gathered_prefills)
        )
        return cache_inputs, cache_slot_mapping

    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

    cache_inputs = tuple(
        torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
        for tensor, gathered_prefill in zip(tensors, gathered_prefills)
    )
    rank_slot_mappings = gathered_slot_mapping.view(pcp_size, local_num_tokens)
    cache_slot_mapping = torch.cat(
        (
            rank_slot_mappings[0, :num_decode_tokens],
            rank_slot_mappings[:, num_decode_tokens:].flatten(),
        )
    )
    return cache_inputs, cache_slot_mapping


def build_case(num_tokens: int, num_descriptors: int, hidden: int = 8):
    """A payload plus `num_descriptors` integer descriptors, dtypes mixed."""
    torch.manual_seed(num_tokens * 31 + num_descriptors)
    payload = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    dtypes = [torch.int64, torch.int32, torch.int64]
    descriptors = [
        torch.arange(num_tokens, dtype=dtypes[i % len(dtypes)]) + i * 17
        for i in range(num_descriptors)
    ]
    return (payload, *descriptors)


@pytest.fixture
def group(monkeypatch):
    g = FakePCPGroup()
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: g)
    return g


@pytest.mark.parametrize("num_descriptors", [0, 1, 2])
@pytest.mark.parametrize("num_decode_tokens", [0, 3])
@pytest.mark.parametrize("local_slots", [True, False])
def test_matches_unpacked_reference(
    group, num_descriptors, num_decode_tokens, local_slots
):
    num_tokens = 12
    tensors = build_case(num_tokens, num_descriptors)
    slots = torch.arange(
        num_tokens if local_slots else num_tokens * group.world_size,
        dtype=torch.int64,
    )

    got_tensors, got_slots = pcp.gather_prefill_cache_inputs(
        tensors, slots, num_decode_tokens
    )
    want_tensors, want_slots = reference_impl(
        group, tensors, slots, num_decode_tokens
    )

    assert len(got_tensors) == len(want_tensors)
    for got, want in zip(got_tensors, want_tensors):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert got_slots.dtype == want_slots.dtype
    torch.testing.assert_close(got_slots, want_slots, rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_descriptors,local_slots,want_collectives",
    [
        # payload + descriptors + (slot map when rank-local), packed into
        # payload + one descriptor collective.
        (2, True, 2),  # compressor: was 4
        (1, True, 2),  # attention:  was 3
        (0, True, 2),  # indexer:    was 2, nothing to pack with
        (2, False, 2),  # slot map already gathered: was 3
        (0, False, 1),  # payload only
    ],
)
def test_collective_count(group, num_descriptors, local_slots, want_collectives):
    num_tokens = 12
    tensors = build_case(num_tokens, num_descriptors)
    slots = torch.arange(
        num_tokens if local_slots else num_tokens * group.world_size,
        dtype=torch.int64,
    )

    pcp.gather_prefill_cache_inputs(tensors, slots, 0)
    assert len(group.calls) == want_collectives


def test_coalesces_independent_cache_payloads(group, monkeypatch):
    """C4's 512- and 128-wide compressor payloads keep their own layout."""
    submitted: list[tuple[torch.Size, torch.dtype]] = []
    manager_calls: list[tuple[object, torch.device]] = []

    class FakeCoalescingManager:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_manager(*, group, device):
        manager_calls.append((group, device))
        return FakeCoalescingManager()

    def fake_all_gather_into_tensor(output, tensor, *, group):
        assert group is group_fixture.device_group
        submitted.append((tensor.shape, tensor.dtype))
        output.copy_(group_fixture.all_gather(tensor, dim=0))

    group_fixture = group
    monkeypatch.setattr(pcp.envs, "VLLM_DSV4_C4_COALESCED_ALLGATHER", True)
    monkeypatch.setattr(pcp.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(pcp, "pcp_comm_ablation_enabled", lambda: False)
    monkeypatch.setattr(torch.distributed, "_coalescing_manager", fake_manager)
    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor
    )

    main = torch.randn(12, 16)
    indexer = torch.randn(12, 4)
    got = pcp.coalesced_pcp_all_gather_payloads(
        (main, indexer), trace_label="compressor_c4:coalesced"
    )

    assert got is not None
    assert len(manager_calls) == 1
    assert manager_calls[0][0] is group.device_group
    assert submitted == [(main.shape, main.dtype), (indexer.shape, indexer.dtype)]
    assert torch.equal(got[0], group.all_gather(main, dim=0))
    assert torch.equal(got[1], group.all_gather(indexer, dim=0))


def test_coalesced_payload_gather_stays_eager(group, monkeypatch):
    """Dynamo cannot trace ProcessGroup's private coalescing context."""
    monkeypatch.setattr(pcp.envs, "VLLM_DSV4_C4_COALESCED_ALLGATHER", True)
    monkeypatch.setattr(pcp.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(pcp, "pcp_comm_ablation_enabled", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    assert (
        pcp.coalesced_pcp_all_gather_payloads(
            (torch.randn(4, 8), torch.randn(4, 2)),
            trace_label="compressor_c4:coalesced",
        )
        is None
    )


@pytest.mark.parametrize("num_decode_tokens", [0, 3])
def test_uses_pre_gathered_primary_payload(group, num_decode_tokens):
    """A pre-gathered C4 payload preserves descriptor and rejoin semantics."""
    tensors = build_case(12, num_descriptors=2)
    slots = torch.arange(12 * group.world_size, dtype=torch.int64)
    pre_gathered = group.all_gather(
        tensors[0][num_decode_tokens:].contiguous(), dim=0
    )
    group.calls.clear()

    got_tensors, got_slots = pcp.gather_prefill_cache_inputs(
        tensors,
        slots,
        num_decode_tokens,
        gathered_payload_0=pre_gathered,
    )
    # Only the two integer descriptors are sent; the primary payload was
    # collected by the outer C4 ProcessGroup batch.
    assert len(group.calls) == 1

    want_tensors, want_slots = reference_impl(
        group, tensors, slots, num_decode_tokens
    )
    for got, want in zip(got_tensors, want_tensors):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(got_slots, want_slots, rtol=0, atol=0)


def test_pcp_comm_nvtx_tags_named_collectives(group, monkeypatch):
    """ROCTx labels must identify cache payload and descriptor gathers."""
    labels: list[str] = []
    monkeypatch.setattr(pcp.envs, "VLLM_PCP_COMM_NVTX", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", labels.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    tensors = build_case(12, num_descriptors=2)
    slots = torch.arange(12 * group.world_size, dtype=torch.int64)
    pcp.gather_prefill_cache_inputs(
        tensors,
        slots,
        0,
        trace_label="swa_cache",
    )

    assert labels == [
        "PCP_COMM:swa_cache:payload_0",
        "PCP_COMM:swa_cache:descriptors_packed",
    ]


def test_second_payload_is_not_a_descriptor(group):
    """A float cache input after the first one is payload, not a descriptor.

    The MLA latent path hands in kv_c and k_pe together. Splitting the tuple
    positionally made k_pe a descriptor, which the packing assert would have
    caught except that a lone descriptor returns before it.
    """
    num_tokens = 12
    tensors = (
        torch.randn(num_tokens, 8, dtype=torch.bfloat16),
        torch.randn(num_tokens, 4, dtype=torch.bfloat16),
        torch.arange(num_tokens, dtype=torch.int64),
    )
    slots = torch.arange(num_tokens * group.world_size, dtype=torch.int64)

    got_tensors, _ = pcp.gather_prefill_cache_inputs(tensors, slots, 0)

    # Two payload collectives plus one for the descriptor, which stays 1-D
    # because a lone descriptor has nothing to be stacked with.
    assert [shape for shape, _ in group.calls] == [
        torch.Size([num_tokens, 8]),
        torch.Size([num_tokens, 4]),
        torch.Size([num_tokens]),
    ]
    for got, want in zip(got_tensors, tensors):
        assert got.dtype == want.dtype
        assert got.shape[1:] == want.shape[1:]


def test_sparse_mla_compressed_slots_match_pcp_payload_order(group, monkeypatch):
    """C128 compressor targets must follow gathered rank-major prefill rows."""
    from vllm.models.deepseek_v4 import sparse_mla

    monkeypatch.setattr(sparse_mla, "get_pcp_group", lambda: group)
    local_slots = torch.tensor([7, -1, 9, -1], dtype=torch.int64)

    got = sparse_mla._gather_pcp_prefill_slot_mapping(
        local_slots, num_decode_tokens=1
    )

    assert torch.equal(
        got,
        torch.tensor(
            [7, -1, 9, -1, 999, 1009, 999, 1999, 2009, 1999, 2999, 3009, 2999],
            dtype=torch.int64,
        ),
    )


def test_sparse_mla_builder_gathers_padded_rocm_pcp_slots(group, monkeypatch):
    """C128 cache slots must match the compressor's gathered payload rows."""
    from vllm.models.deepseek_v4 import sparse_mla

    monkeypatch.setattr(sparse_mla, "get_pcp_group", lambda: group)
    monkeypatch.setattr(sparse_mla.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        sparse_mla,
        "split_decodes_and_prefills",
        lambda *_args, **_kwargs: (1, 1, 1, 3),
    )

    def fake_compressed_slots(num_tokens, *_args, out, **_kwargs):
        assert num_tokens == 4
        out.fill_(-1)
        out[:4] = torch.tensor([7, -1, 9, -1], dtype=torch.int64)
        return out[:4]

    monkeypatch.setattr(sparse_mla, "get_compressed_slot_mapping", fake_compressed_slots)
    builder = types.SimpleNamespace(
        compress_ratio=128,
        pcp_world_size=group.world_size,
        compressed_slot_mapping_buffer=torch.empty(6, dtype=torch.int64),
        req_id_per_token_buffer=torch.empty(4, dtype=torch.int32),
        kv_cache_spec=types.SimpleNamespace(num_states=2, block_size=256),
        reorder_batch_threshold=1,
        topk_tokens=1024,
        _build_c128a_metadata=lambda *_args: {},
    )
    common = types.SimpleNamespace(
        num_actual_tokens=4,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.empty(6, dtype=torch.int64),
        num_reqs=1,
        max_query_len=4,
        max_seq_len=4,
        token_to_req_indices=lambda _out: torch.zeros(4, dtype=torch.int32),
    )

    metadata = sparse_mla.DeepseekV4SparseMLAMetadataBuilder.build(
        builder, 0, common
    )

    assert torch.equal(
        metadata.slot_mapping,
        torch.tensor(
            [
                7,
                -1,
                9,
                -1,
                -1,
                -1,
                999,
                1009,
                999,
                999,
                999,
                1999,
                2009,
                1999,
                1999,
                1999,
                2999,
                3009,
                2999,
                2999,
                2999,
            ],
            dtype=torch.int64,
        ),
    )


def test_metadata_ablation_leaves_payload_on_the_wire(group, monkeypatch):
    """VLLM_PCP_META_GATHER_OFF prices descriptors, so it may only drop those.

    k_pe is 448 KiB per layer that only its owning rank has, against 28-84 KiB
    for every real descriptor combined. Dropping it under the metadata tag made
    the ablation quote a ceiling for a fix that cannot include it.
    """
    monkeypatch.setattr(pcp, "pcp_meta_gather_ablation_enabled", lambda: True)
    num_tokens = 12
    tensors = (
        torch.randn(num_tokens, 8, dtype=torch.bfloat16),
        torch.randn(num_tokens, 4, dtype=torch.bfloat16),
        torch.arange(num_tokens, dtype=torch.int64),
    )
    slots = torch.arange(num_tokens * group.world_size, dtype=torch.int64)

    pcp.gather_prefill_cache_inputs(tensors, slots, 0)

    assert [shape for shape, _ in group.calls] == [
        torch.Size([num_tokens, 8]),
        torch.Size([num_tokens, 4]),
    ]


@pytest.fixture
def forward_context(monkeypatch):
    """A stand-in forward pass to scope the descriptor cache to."""
    import vllm.forward_context as fc

    ctx = types.SimpleNamespace(pcp_descriptor_cache={})
    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(fc, "get_forward_context", lambda: ctx)
    return ctx


def test_descriptors_gather_once_per_forward_pass(group, forward_context):
    """All 61 layers in a group are handed the same metadata object.

    So they gather byte-identical descriptors. The payload differs per layer and
    still rides its own collective; the descriptors should not.
    """
    metadata = object()
    slots = torch.arange(12 * group.world_size, dtype=torch.int64)

    first = None
    for _ in range(3):
        got, _ = pcp.gather_prefill_cache_inputs(
            build_case(12, 2), slots, 0, descriptor_key=metadata
        )
        if first is None:
            first = got
        else:
            for a, b in zip(got[1:], first[1:]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    # Three payload gathers, one descriptor gather.
    assert len(group.calls) == 4


def test_descriptor_cache_is_keyed_on_the_metadata(group, forward_context):
    """Two groups gather different descriptors and must not share an entry."""
    slots = torch.arange(12 * group.world_size, dtype=torch.int64)
    for metadata in (object(), object()):
        pcp.gather_prefill_cache_inputs(
            build_case(12, 2), slots, 0, descriptor_key=metadata
        )
    assert len(group.calls) == 4


def test_descriptors_are_not_cached_without_a_key(group, forward_context):
    """No key means no claim that the descriptors repeat, so gather every time."""
    slots = torch.arange(12 * group.world_size, dtype=torch.int64)
    for _ in range(3):
        pcp.gather_prefill_cache_inputs(build_case(12, 2), slots, 0)
    assert len(group.calls) == 6


def test_descriptors_stay_contiguous(group):
    """Unpacking a column of the stacked buffer must not leak a strided view."""
    tensors = build_case(12, 2)
    slots = torch.arange(12, dtype=torch.int64)
    got_tensors, got_slots = pcp.gather_prefill_cache_inputs(tensors, slots, 0)
    for got in (*got_tensors, got_slots):
        assert got.is_contiguous()


def test_all_decode_skips_communication(group):
    tensors = build_case(6, 2)
    slots = torch.arange(6, dtype=torch.int64)
    got_tensors, got_slots = pcp.gather_prefill_cache_inputs(tensors, slots, 6)
    assert group.calls == []
    for got, want in zip(got_tensors, tensors):
        assert got is want
    torch.testing.assert_close(got_slots, slots, rtol=0, atol=0)
