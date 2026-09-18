# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def test_replicated_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [3, 3]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 1, 2], [0, 1, 2]]
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 1, 2]))
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 1, 2, 0, 0, 1, 2, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, True, False, False, False, False, False]),
    )


def test_input_buffers_are_exposed_for_cudagraph_capture():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
    )

    assert manager.input_buffers is manager._input_buffers
    assert manager.input_buffers.input_ids.shape == (8,)
    assert manager.input_buffers.positions.shape == (8,)
    assert manager.input_buffers.is_padding.shape == (8,)


def test_gathered_padding_mask_is_derived_from_rank_layout():
    manager = PCPManager(
        pcp_world_size=3,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=5,
    )

    gathered = manager._get_gathered_is_padding([5, 3, 1], padded_num_tokens=5)
    assert torch.equal(
        gathered,
        torch.tensor(
            [
                False, False, False, False, False,
                False, False, False, True, True,
                False, True, True, True, True,
            ]
        ),
    )

    # The next forward reuses the persistent buffer rather than allocating a
    # mask for every MoE layer or every model step.
    reused = manager._get_gathered_is_padding([2, 5, 4], padded_num_tokens=5)
    assert reused.data_ptr() == gathered.data_ptr()
    assert torch.equal(
        reused,
        torch.tensor(
            [
                False, False, True, True, True,
                False, False, False, False, False,
                False, False, False, False, True,
            ]
        ),
    )


def test_c128_boundary_indices_follow_rank_major_zigzag_layout(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    segments_by_rank = [
        [
            pcp_manager_module.RankSegment(0, slice(0, 130), slice(0, 130)),
            pcp_manager_module.RankSegment(0, slice(500, 630), slice(130, 260)),
        ],
        [
            pcp_manager_module.RankSegment(0, slice(300, 430), slice(0, 130)),
        ],
    ]
    indices = manager._get_c128_boundary_indices(
        segments_by_rank,
        num_computed_tokens=np.asarray([0], dtype=np.int32),
        query_start_loc_np=np.asarray([0, 630], dtype=np.int32),
        padded_num_tokens=260,
    )

    # Segment starts are 0, 500 and 300. Their first positions satisfying
    # (position + 1) % 128 == 0 are offsets 127, 11 and 83 respectively.
    assert torch.equal(indices, torch.tensor([127, 141, 343]))


def test_short_prefill_is_replicated_on_every_rank():
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )

    chunks = [
        list(
            manager._iter_rank_chunks(
                rank,
                np.asarray([3], dtype=np.int32),
                np.asarray([True], dtype=np.bool_),
            )
        )
        for rank in range(4)
    ]

    assert chunks == [[(0, 0, 3)]] * 4


@pytest.mark.parametrize(
    ("pcp_world_size", "num_scheduled_tokens", "is_prefilling", "expected"),
    [
        (2, [8], [True], 4),
        (2, [7], [True], 4),
        (2, [3], [False], 3),
        (2, [3, 8], [False, True], 7),
        (4, [2, 9], [False, True], 5),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
    )

    actual = manager.get_num_tokens_for_dispatch(
        np.asarray(num_scheduled_tokens, dtype=np.int32),
        np.asarray(is_prefilling, dtype=np.bool_),
    )

    assert actual == expected


def test_graph_padding_cannot_be_smaller_than_largest_pcp_rank(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=2,
        )
