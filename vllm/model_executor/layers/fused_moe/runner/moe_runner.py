# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import pcp_comm_ablation_enabled
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm import envs
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.pcp import pcp_comm_trace
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    direct_register_custom_op,
)

logger = init_logger(__name__)


def register_layer_for_moe_forward_op(
    vllm_config: VllmConfig,
    layer: "MoERunner",
):
    # For smuggling this layer into the fused moe custom op
    prefix = layer.layer_name
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = layer
    compilation_config.static_all_moe_layers.append(prefix)


def get_layer_from_name(layer_name: str) -> MoERunnerInterface:
    forward_context: ForwardContext = get_forward_context()
    if not _USE_LAYERNAME and layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    layer = forward_context.no_compile_layers[layer_name]
    assert isinstance(layer, MoERunnerInterface)
    return layer


# On torch >= 2.11, layer_name is a hoisted LayerName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's '_forward_impl' method.
# These functions should never be called directly since they do not
# include all the functionality of the MoE layer.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return cast(
        torch.Tensor,
        layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        ),
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    # `hidden_dim_unpadded > 0` only on the TRT-LLM MXFP4 path, where the
    # real kernel writes narrower than `hidden_states.shape[-1]`. Plumbed
    # as an op arg (not peeked from the layer registry) to keep the fake
    # a pure shape function of its inputs and preserve subgraph dedup.
    if hidden_dim_unpadded > 0:
        return hidden_states.new_empty((*hidden_states.shape[:-1], hidden_dim_unpadded))
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        layer._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        ),
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `fused_out`: see `_moe_forward_fake` for hidden_dim_unpadded semantics.
    # `shared_out`: matches `shared_experts_input` if provided (latent MoE),
    # else `hidden_states`.
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded)
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


# NOTE: `moe_forward` and `moe_forward_shared` being opaque custom ops is a
# load-bearing assumption for the MoE-LoRA dual-stream path.
direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _unpack(
    result: torch.Tensor
    | UnfinalizedMoEOutput
    | tuple[torch.Tensor, torch.Tensor | UnfinalizedMoEOutput],
) -> tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]:
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


def _all_reduce(states: torch.Tensor) -> torch.Tensor:
    """TP all-reduce, elided under the PCP comm ablation."""
    if pcp_comm_ablation_enabled():
        return states
    return tensor_model_parallel_all_reduce(states)


class MoERunner(MoERunnerInterface):
    """
    Standard MoE runner implementation for executing Mixture of Experts layers.

    This is the primary concrete implementation of MoE execution logic, providing
    comprehensive support for standard MoE operations. It handles:
    - Expert routing and token dispatching using various routing strategies
    - Shared experts computation with optional parallel execution using CUDA streams
    - Tensor model parallel and expert parallel operations
    - Multiple quantization methods and optimized kernel selection
    - Both monolithic and decomposed expert execution paths
    - Integration with various parallel execution modes (TP, EP, DP)

    The runner orchestrates the complete MoE forward pass including routing tokens
    to experts, executing expert computations in parallel, and combining results.
    It supports advanced features like overlapped execution of shared experts,
    optimized kernels for different parallel configurations, and seamless
    integration with vLLM's distributed execution framework.

    Eventually, this class may be split into more specialized implementations
    for different configurations (e.g., with/without shared experts, gates, etc.).
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = gate
        self.shared_expert_gate = shared_expert_gate
        self.routed_experts = routed_experts
        self.enable_dbo = enable_dbo

        # When both gates are present and FSE is enabled, fuse their
        # weight matrices into [num_experts + num_shared, hidden] so one
        # F.linear produces combined logits. The topk kernel can then
        # apply routing softmax and shared expert activation (sigmoid)
        # in a single launch.
        self._fse_fuse_gate = gate is not None and shared_expert_gate is not None
        self._combined_gate_weight: torch.Tensor | None = None

        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            can_overlap = lambda: self._quant_method.mk_can_overlap_shared_experts
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
            )

        # Needed for string -> MoERunner layer lookup in custom ops.
        self.layer_name = layer_name

        self._forward_entry = self._select_forward()

        # For smuggling this layer into the fused moe custom op
        register_layer_for_moe_forward_op(get_current_vllm_config(), self)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        return self.routed_experts.load_weights(weights)

    def _select_forward(self) -> Callable:
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped _forward_impl.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    @property
    def shared_experts(self) -> SharedExperts | None:
        return self._shared_experts

    # TODO(bnell): Temporary hack. Get rid of this.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        self.routed_experts._replace_quant_method(quant_method)

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config
        self.routed_experts._set_moe_config(new_moe_config)
        if self._shared_experts is not None:
            self._shared_experts._set_moe_config(new_moe_config)

    def _maybe_fuse_gate_weights(self):
        """Fuse router and shared expert gate weights on first call.

        Cannot be done at __init__ because gate weights are loaded after
        module construction (via weight_loader). Called once from
        _forward_impl before the first forward pass.
        """
        if self._combined_gate_weight is None:
            assert self.gate is not None and self.shared_expert_gate is not None
            self._combined_gate_weight = torch.cat(
                [self.gate.weight, self.shared_expert_gate.weight],
                dim=0,
            )

    @property
    def _quant_method(self) -> FusedMoEMethodBase:
        return self.routed_experts.quant_method

    def apply_routed_input_transform(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by MoERunner.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self._shared_experts is not None else None,
        )

    def apply_routed_output_transform(
        self,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transform to routed expert output (e.g., latent to full dim).

        Used by latent MoE models (e.g., NemotronH) where routed experts
        operate in a compressed latent space and need projection back to
        the full hidden dimension before combining with shared expert output.
        """
        if self.routed_output_transform is not None:
            r = self.routed_output_transform(fused_output)
            fused_output = r[0] if isinstance(r, tuple) else r
        return fused_output

    def _maybe_apply_routed_scale_to_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply routed_scaling_factor to the output with FP16 overflow
        protection.

        Scale the fused expert output by routed_scaling_factor. For FP16,
        avoid overflow by dividing shared_output by the scale instead
        (the decoder layer compensates with matching divisions).
        """
        if self.routed_scaling_factor != 1.0:
            if fused_output.dtype != torch.float16 or shared_output is None:
                fused_output *= self.routed_scaling_factor
            elif shared_output is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
        return shared_output, fused_output

    @property
    def _fused_output_is_reduced(self) -> bool:
        return (
            self._quant_method.moe_kernel is not None
            and self._quant_method.moe_kernel.output_is_reduced()
        )

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        """All-reduce shared expert output when the combine kernel already
        reduced fused output.

        * If the combine kernel does the reduction for fused_output, reduce
          shared_output separately. O.w, reduce fused_output+shared_output later.
        * If we have SP (TP=N, DP=M, EP), there is a separate AG step handled
          in the model.
        """
        if fused_output_is_reduced is None:
            fused_output_is_reduced = self._fused_output_is_reduced

        if (
            shared_output is not None
            and not self.moe_config.is_sequence_parallel
            and fused_output_is_reduced
        ):
            shared_output = _all_reduce(shared_output)
        return shared_output

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        """All-reduce latent routed output before its output transform.

        Latent MoE output transforms may contain non-linear ops, e.g. RMSNorm.
        TP partial routed outputs must be summed in latent space before such
        transforms are applied.
        """
        if (
            self.routed_output_transform is not None
            and not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not fused_output_is_reduced
        ):
            fused_output = _all_reduce(fused_output)
            fused_output_is_reduced = True
        return fused_output, fused_output_is_reduced

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        """All-reduce the combined output if needed.

        This is the "late" all-reduce path. When neither fused nor shared
        output was individually reduced, the combined sum is all-reduced
        here. Skipped when sequence-parallel is active (SP handles its
        own reduction) or when the early path already reduced both outputs.
        """
        # skip_final_all_reduce must not coexist with a pre-reduced fused
        # output. This should be enforced by MoE config initialization.
        if self.moe_config.skip_final_all_reduce:
            assert not self._fused_output_is_reduced, (
                "skip_final_all_reduce requires an un-reduced fused output"
            )

        # We don't need to reduce the final output if:
        # - We are not running with TP or DP
        # - The MK already reduced the fused output itself.
        if output_is_reduced is None:
            output_is_reduced = self._fused_output_is_reduced

        if (
            not self.moe_config.is_sequence_parallel
            and not self.moe_config.skip_final_all_reduce
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not output_is_reduced
        ):
            states = _all_reduce(states)

        return states[..., :trunc_size] if trunc_size is not None else states

    def _encode_layer_name(self) -> str | LayerName:
        if _USE_LAYERNAME:
            return LayerName(self.layer_name)
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int | None, int | None]:
        """Pad hidden_states to moe_config.hidden_dim and compute the
        original dimension for later truncation.

        For latent MoE, the routed hidden_states may be smaller than
        hidden_dim. Padding ensures uniform tensor sizes through the
        fused MoE kernel. The returned trunc_size is used by
        _maybe_reduce_final_output to strip the padding from the result.
        """
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim: int | None = hidden_states.shape[-1]
        if (
            not self._quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            assert transformed_hidden_dim is not None
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # Truncation sizes for stripping kernel padding from the output.
        # None means no truncation needed (no padding was applied).
        #
        # Two truncation points exist in forward():
        #   pre_xform:  applied to fused_output BEFORE routed_output_transform
        #   post_xform: applied to the final result AFTER all-reduce
        #
        # MoE with routed output transform or shared experts:
        #   - pre_xform applies if the transform needs unpadded routed output
        #     or shared+routed add needs matching hidden dims. For Nemotron-3
        #     Nano, TRTLLM NVFP4 pads routed MoE hidden dim 2688->2816, while
        #     shared output stays 2688.
        #   - post_xform uses shared_experts_hidden_dim when transform and shared
        #     experts make the final output full hidden dim.
        #
        # Standard MoE / MoE without transforms (GPT-OSS, Mixtral):
        #   - pre_xform is None (no early truncation)
        #   - post_xform strips padding after all-reduce (or None if unpadded)
        if transformed_hidden_dim == hidden_states.shape[-1]:
            transformed_hidden_dim = None

        pre_xform_trunc_size = None
        if self.routed_output_transform is not None or shared_experts_hidden_dim > 0:
            pre_xform_trunc_size = transformed_hidden_dim
        post_xform_trunc_size = transformed_hidden_dim
        if self.routed_output_transform is not None and shared_experts_hidden_dim > 0:
            post_xform_trunc_size = shared_experts_hidden_dim

        return hidden_states, pre_xform_trunc_size, post_xform_trunc_size

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts(shared_experts_input, order)

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        topk: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).
        """
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )

        if self.routed_experts.quant_method.is_monolithic:
            # Monolithic kernels: pass router_logits to routed_experts
            fused_out = self.routed_experts.forward_monolithic(
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            # Modular kernels: select experts first, then call routed_experts.
            # Unless routing already ran on the local shard ahead of the gather
            # and rode across on it; see _pcp_wants_local_routing.
            if topk is not None:
                topk_weights, topk_ids = topk
            else:
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    topk_indices_dtype=self._quant_method.topk_indices_dtype,
                    input_ids=input_ids,
                )

            fused_out = self.routed_experts.forward_modular(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=self._shared_experts,
                shared_experts_input=shared_experts_input,
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        """Return a context manager for sequence-parallel token
        redistribution.

        When sequence parallelism is active, returns a context that handles
        local size tracking for proper token scatter/gather. Otherwise
        returns a no-op context.
        """
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor | None,
    ):
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the RoutedExperts via
        #        separate cuda stream)
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.maybe_sync_shared_experts_stream(shared_experts_input)

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Add the zero expert's contribution to the final result.

        When a ZeroExpertRouter is used, it computes a bias-like output
        from the "zero expert" that is added to the combined routed+shared
        expert output.
        """
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.

        Calling sequence
        - forward
          - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
            - _forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reason:
        1. pytorch cannot handle union types in custom op signatures so
           _moe_forward and _moe_forward_shared must be split.
        """

        # Apply transform for routed experts (e.g., latent projection for
        # latent MoE). When the caller pre-applies the routed input transform
        # outside the runner (e.g. to overlap it on a separate stream), it
        # passes the already-transformed routed input as ``hidden_states`` and
        # the original hidden states as ``shared_experts_input``; skip the
        # transform in that case so shared experts still see the original input.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.

        # Extract outputs from result
        shared_output, fused_output = _unpack(result)
        fused_output = cast(torch.Tensor, fused_output)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        fused_output_is_reduced = self._fused_output_is_reduced

        # Latent routed output has to be reduced before output transform,
        # because the transform may include non-linear normalization.
        fused_output, fused_output_is_reduced = (
            self._maybe_reduce_routed_output_before_transform(
                fused_output,
                fused_output_is_reduced,
            )
        )

        # If routed output is already reduced, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(
            shared_output, fused_output_is_reduced
        )

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output
        )

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(
            result, og_hidden_dim_post_xform, fused_output_is_reduced
        )

        return self._maybe_add_zero_expert_output(result)

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 or self.moe_config.is_sequence_parallel
        ) and not self._quant_method.supports_internal_mk

    def _pcp_flattens_to_tp(self) -> bool:
        # EP is off: PCP ranks are flattened into TP for expert GEMM sharding.
        # Tokens must be replicated (all-gather) so that TP all-reduce is valid.
        # This is MoE tensor-parallel, not attention PCP.
        return (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        )

    # Below this many gathered tokens aiter serves the A8W4 experts from its
    # flydsl kernels instead of the opus ones, and on that path only opus knows
    # how to take an already-MXFP8 input -- it has an explicit MXFP8 dispatch
    # passthrough, while flydsl's afp8 route assumes it is handed bf16 and hangs
    # on FP8. The opus kernels are selected from a few hundred tokens up, so this
    # bound sits well inside their range. It costs nothing: the gather is
    # proportional to the token count, so the small shapes this excludes had
    # nothing to save.
    #
    # A4W4 is on flydsl at every M by construction, and its fp4x2 entry does
    # accept a pre-quantized input (fused_moe.py takes the
    # `hidden_states.dtype == fp4x2 and a1_scale is not None` branch and only
    # sorts the scale). The floor still applies to it unchanged, because what it
    # is really buying is headroom against the ksplit hazard in
    # _pcp_gather_quant_dtype's docstring, which small M is likelier to trip.
    _PCP_MXFP8_MIN_GATHERED_TOKENS = 2048

    def _pcp_gather_quant_dtype(self, hidden_states: torch.Tensor) -> str | None:
        """How the expert input should cross the gather: "fp4", "fp8", or None.

        Only the AITER MXFP4-weight experts qualify: they quantize their input
        the moment it arrives, so pre-quantizing hands them the operand they were
        going to build anyway. Any other backend either wants bf16 or wants a
        different scheme, and pre-quantizing for those would be a real loss of
        precision rather than a reordering.

        Which of the two it is follows VLLM_DSV4_MOE_A4W4, because the gather has
        to produce whatever the experts are about to consume -- gathering MXFP8
        into an A4W4 expert call, or MXFP4 into an A8W4 one, is not a slower
        variant of the right thing, it is the wrong operand.

        Known hazard on the fp4 side, and the first thing to check if A4W4 comes
        back numerically wrong rather than merely slow: in aiter's fused_moe the
        pre-quantized fp4x2 branch is a *fallthrough*, and an earlier branch
        claims `q_dtype_a is fp4x2 and metadata.ksplit > 1 and is_shuffled` and
        does `a1 = hidden_states.to(dtype)`. Reached with a packed fp4x2 input
        that is a reinterpret, not a conversion, and the experts get garbage.
        Whether it is reached depends on the ksplit in the tuned entry aiter
        picks for the shape, which is a property of the tuning table rather than
        of anything vLLM can see from here -- so it cannot be asserted on, only
        checked against a run.
        """
        if not envs.VLLM_DSV4_MOE_ALLGATHER_FP8:
            return None
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            return None
        # MX scales a 32-wide group; a ragged tail has no defined scale.
        if hidden_states.shape[-1] % 32 != 0:
            return None
        gathered_tokens = hidden_states.shape[0] * self.moe_config.pcp_size
        if gathered_tokens < self._PCP_MXFP8_MIN_GATHERED_TOKENS:
            return None
        if not current_platform.is_rocm():
            return None

        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

        backend = getattr(self._quant_method, "mxfp4_backend", None)
        if backend != Mxfp4MoeBackend.AITER_MXFP4_BF16:
            return None
        return "fp4" if envs.VLLM_DSV4_MOE_A4W4 else "fp8"

    def _pcp_wants_local_routing(self) -> bool:
        """Whether routing can run on the local tokens, ahead of the gather.

        Top-k is per-token -- a token's experts come out of its own logits and
        the fixed correction bias -- so routing each rank's shard and
        concatenating is the same selection as routing the concatenation. What
        has to hold is that nothing downstream wants the routing as a view of
        the whole gathered batch.
        """
        if not envs.VLLM_DSV4_MOE_LOCAL_ROUTING:
            return False
        if not self._pcp_flattens_to_tp():
            return False
        # Naive DP/EP dispatch redistributes tokens ahead of the PCP gather, so
        # routing before both would have to chase the payload through two
        # reorderings rather than one concatenation.
        if self.do_naive_dispatch_combine:
            return False
        # Monolithic kernels route inside the expert call off router_logits;
        # there is no separate selection step to hoist out of the gather.
        if self.router is None or self.routed_experts.quant_method.is_monolithic:
            return False
        # EPLB tallies expert load off topk_ids, and routing locally would have
        # each rank tally its own eighth of the batch.
        if getattr(self.router, "eplb_state", None) is not None:
            return False
        # The capture/replay hooks record ids for the gathered batch.
        if getattr(self.router, "capture_fn", None) is not None:
            return False
        if getattr(self.router, "_routing_replay_out", None) is not None:
            return False
        return True

    @staticmethod
    def _pack_routing(topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> tuple:
        """Lay the top-k out as one row of opaque bytes per token.

        The collectives have no dtype for e8m0 and none of these payloads care
        what the wire calls them, so everything crosses as uint8 and gets its
        type put back on the far side. Returns the panel plus what is needed to
        undo it.
        """
        w = topk_weights.contiguous().view(torch.uint8)
        i = topk_ids.contiguous().view(torch.uint8)
        meta = (w.shape[1], topk_weights.dtype, topk_ids.dtype)
        return torch.cat((w, i), dim=1), meta

    @staticmethod
    def _unpack_routing(panel: torch.Tensor, meta: tuple) -> tuple:
        """Split a gathered routing panel back into (weights, ids)."""
        split, w_dtype, i_dtype = meta
        # A column slice of the gathered panel is strided, and viewing a dtype
        # back onto it needs the last dim packed.
        w = panel[:, :split].contiguous().view(w_dtype)
        i = panel[:, split:].contiguous().view(i_dtype)
        return w, i

    @staticmethod
    def _pcp_all_gather_pair(
        pcp, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather two dim-0 PCP tensors through one ProcessGroup batch.

        MXFP8 expert input has a large FP8 payload and a small E8M0 scale
        panel. They must remain separate, contiguous tensors: packing them
        column-wise makes the recovered activation rows strided, which is not
        numerically equivalent for the current Opus MoE kernels. NCCL/RCCL's
        coalesced API batches their launches without changing either layout.
        """
        can_coalesce = (
            envs.VLLM_DSV4_MOE_COALESCED_ALLGATHER
            and current_platform.is_rocm()
            and not pcp_comm_ablation_enabled()
            and hasattr(torch.distributed, "_coalescing_manager")
            and first.is_contiguous()
            and second.is_contiguous()
        )
        if not can_coalesce:
            with pcp_comm_trace("moe_input:fp8_payload"):
                first_out = pcp.all_gather(first, dim=0)
            with pcp_comm_trace("moe_input:scale_and_routing"):
                second_out = pcp.all_gather(second, dim=0)
            return first_out, second_out

        first_out = torch.empty(
            (first.shape[0] * pcp.world_size, *first.shape[1:]),
            dtype=first.dtype,
            device=first.device,
        )
        second_out = torch.empty(
            (second.shape[0] * pcp.world_size, *second.shape[1:]),
            dtype=second.dtype,
            device=second.device,
        )
        # On ROCm GroupCoordinator.all_gather also resolves to
        # all_gather_into_tensor. Going direct here is equivalent but lets the
        # ProcessGroup issue both collectives as one batch.
        with pcp_comm_trace("moe_input:coalesced_payload_and_scale"):
            with torch.distributed._coalescing_manager(  # type: ignore[attr-defined]
                group=pcp.device_group, device=first.device
            ):
                torch.distributed.all_gather_into_tensor(
                    first_out, first, group=pcp.device_group
                )
                torch.distributed.all_gather_into_tensor(
                    second_out, second, group=pcp.device_group
                )
        return first_out, second_out

    def _pcp_all_gather_expert_input(
        self,
        pcp,
        hidden_states: torch.Tensor,
        routing_panel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """All-gather the expert input, pre-quantized to whatever the experts want.

        This gather is the layer's most expensive collective -- every rank needs
        every token because the flattened MoE TP group shards the intermediate
        dim, not the batch -- and it moves bf16 that the experts immediately
        quantize. Quantizing first cuts the payload per token from 14336 bytes
        to 7392 under A8W4 (7168 payload + 224 scale), and to 3808 under A4W4
        (3584 + 224). A4W4 is where most of that switch's win is expected to
        come from: the weights are MXFP4 in both cases, so the expert GEMMs get
        only the activation side and the MFMA rate, while this collective halves
        outright.

        The result is bitwise what gathering bf16 would have produced. Two
        things have to hold for that, and only one of them is about the gather:
        MX groups 32 values along the hidden dim and takes each group's scale
        from that group alone, so a row quantizes the same whether or not the
        other ranks' rows are present and the gather only concatenates rows.
        That argument is about the group geometry and is the same for fp8 and
        fp4.

        The other is that we quantize with the same kernel aiter would have.
        MX leaves the scale rounding open and aiter carries several; its Triton
        `dynamic_mxfp8_quant` picks a scale one octave coarser than
        `per_1x32_mx_quant_hip` on about 0.3% of groups, which is invisible in
        any norm and still moved greedy decode off the baseline within a few
        dozen characters. The HIP entry below is the one aiter's own bf16 path
        reaches through fused_dynamic_mxfp8_quant_moe_sort, so the experts get
        byte-identical operands either way (check_mxfp8_gather_paths.py). The
        fp4 case rests on the same identity for the same reason: fp4 and fp8
        moe-sort wrappers are both thin forwarders onto
        fused_dynamic_mx_quant_moe_sort, which quantizes through this same
        per_1x32_mx_quant_hip before sorting the scale.

        A routing panel, when local routing produced one, rides along on the
        scale gather instead of taking a collective of its own: both are a row
        of bytes per token in the same token order, so concatenating the columns
        and splitting them on the far side costs two small copies and saves a
        launch. The scale panel is 224 bytes per token in both schemes -- one
        e8m0 byte per 32-wide group, and the grouping is over the unpacked
        hidden dim -- so the piggyback is unaffected by the switch.
        """
        num_tokens = hidden_states.shape[0]

        quant = self._pcp_gather_quant_dtype(hidden_states)
        if quant is None:
            with pcp_comm_trace("moe_input:bf16_payload"):
                gathered = pcp.all_gather(hidden_states, dim=0)
            if routing_panel is None:
                return gathered, None
            with pcp_comm_trace("moe_input:routing_panel"):
                gathered_routing = pcp.all_gather(routing_panel, dim=0)
            return gathered, gathered_routing

        from aiter import dtypes
        from aiter.ops.quant import per_1x32_mx_quant_hip

        # fp4x2 packs two values per byte, so a1q comes back half as wide as
        # hidden_states. Everything downstream of here works in bytes until the
        # experts, which take the logical width from w1 rather than from the
        # activation, so the narrower payload needs no special handling on the
        # wire -- only AiterExperts has to know not to read shape[1] as the
        # hidden dim. See the hidden_pad computation there.
        quant_dtype = dtypes.fp4x2 if quant == "fp4" else dtypes.fp8
        a1q, a1q_scale = per_1x32_mx_quant_hip(
            hidden_states,
            scale=None,
            quant_dtype=quant_dtype,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )
        # RCCL has no FP4, FP8 or E8M0 datatype; the payload is opaque bytes to
        # the gather either way, so hand both over as uint8 and put the types
        # back.
        payload_dtype, e8m0_dtype = a1q.dtype, a1q_scale.dtype
        a1q_bytes = a1q.view(torch.uint8)
        scale_bytes = a1q_scale.view(torch.uint8)
        piggyback = (
            routing_panel is not None
            and scale_bytes.dim() == 2
            and scale_bytes.shape[0] == num_tokens
        )
        if piggyback:
            split = scale_bytes.shape[1]
            scale_bytes = torch.cat((scale_bytes, routing_panel), dim=1)

        gathered_a1q, gathered = self._pcp_all_gather_pair(
            pcp, a1q_bytes, scale_bytes.contiguous()
        )
        a1q = gathered_a1q.view(payload_dtype)

        gathered_routing = None
        if piggyback:
            gathered_routing = gathered[:, split:].contiguous()
            gathered = gathered[:, :split].contiguous()
        elif routing_panel is not None:
            with pcp_comm_trace("moe_input:routing_panel"):
                gathered_routing = pcp.all_gather(routing_panel, dim=0)

        # The experts kernel is several call layers down and takes its scales
        # from the forward context; see AiterExperts.apply.
        get_forward_context().pcp_moe_a1q_scale = gathered.view(e8m0_dtype)
        return a1q, gathered_routing

    def _clear_pcp_moe_a1q_scale(self) -> None:
        # Scoped to the one expert call it was gathered for; leaving it set
        # would hand a stale, wrongly sized scale to whichever layer runs next.
        if is_forward_context_available():
            get_forward_context().pcp_moe_a1q_scale = None

    def _restore_pcp_is_padding(self) -> None:
        saved = getattr(self, "_pcp_saved_is_padding", None)
        if saved is False:
            return
        if is_forward_context_available():
            get_forward_context().is_padding = saved
        self._pcp_saved_is_padding = False

    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        topk: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if self.do_naive_dispatch_combine:
            result = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

        if self._pcp_flattens_to_tp():
            pcp = get_pcp_group()
            routing_panel, routing_meta = (
                self._pack_routing(*topk) if topk is not None else (None, None)
            )
            hidden_states, gathered_panel = self._pcp_all_gather_expert_input(
                pcp, hidden_states, routing_panel
            )
            if topk is not None:
                assert gathered_panel is not None
                topk = self._unpack_routing(gathered_panel, routing_meta)
                # The logits were only ever gathered to be routed, and the token
                # ids only to be hashed into that routing. Both already happened
                # on the local shard, so neither has to cross. Dropping the ids
                # rather than passing the local ones on keeps a later consumer
                # from quietly reading a tensor an eighth of the batch long.
                input_ids = None
            else:
                with pcp_comm_trace("moe_routing:logits"):
                    router_logits = pcp.all_gather(router_logits, dim=0)
                if input_ids is not None:
                    with pcp_comm_trace("moe_routing:input_ids"):
                        input_ids = pcp.all_gather(input_ids, dim=0)
            if is_forward_context_available():
                ctx = get_forward_context()
                self._pcp_saved_is_padding = ctx.is_padding
                if ctx.is_padding is not None:
                    gathered_is_padding = ctx.pcp_gathered_is_padding
                    if gathered_is_padding is not None:
                        assert gathered_is_padding.numel() == (
                            pcp.world_size * ctx.is_padding.numel()
                        )
                        ctx.is_padding = gathered_is_padding
                    else:
                        with pcp_comm_trace("moe_routing:padding"):
                            ctx.is_padding = pcp.all_gather(ctx.is_padding, dim=0)

        return hidden_states, router_logits, input_ids, topk

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor | UnfinalizedMoEOutput,
    ) -> (
        torch.Tensor
        | UnfinalizedMoEOutput
        | tuple[torch.Tensor | None, torch.Tensor | UnfinalizedMoEOutput]
    ):
        if self.do_naive_dispatch_combine:
            if isinstance(hidden_states, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "Naive expert-parallel combine cannot consume a deferred "
                    "MoE output."
                )
            hidden_states = get_ep_group().combine(
                hidden_states,
                self.moe_config.is_sequence_parallel,
            )

        if self._pcp_flattens_to_tp():
            if isinstance(hidden_states, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "PCP reduce-scatter cannot consume a deferred MoE output."
                )
            with pcp_comm_trace("moe_output:reduce_scatter"):
                hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
            self._restore_pcp_is_padding()

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | UnfinalizedMoEOutput
        | tuple[torch.Tensor, torch.Tensor | UnfinalizedMoEOutput]
    ):
        """Entry point called by the custom op to run the MoE computation.

        Handles pre-dispatch setup (gate application, external shared expert
        triggering, quant config init) then performs the following steps
        within the sequence-parallel context.

        - Performs expert routing
        - fused MoE kernel execution
        - shared expert computation.

        Returns routed output, optionally paired with shared-expert output. A
        fused consumer may request the routed output in deferred-finalize form.
        """
        # TODO(bnell): this can be removed after MK migration is complete.
        self.routed_experts._ensure_moe_quant_config_init()

        # Sync aux and main stream for shared expert multi-stream overlap.
        self._maybe_sync_shared_experts_stream(shared_experts_input)

        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        # Route before the gather when the gather is only replicating tokens.
        # Everything routing reads is still the local shard here -- the logits
        # the gate just produced, this rank's token ids, and the local padding
        # mask that _maybe_dispatch is about to replace with the gathered one.
        topk = None
        if self._pcp_wants_local_routing():
            topk = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )

        with self._sequence_parallel_context():
            # TODO(bnell): parts of the dispatch/combine steps will go away once
            # #32567 lands and the remaining kernels are made MKs.  The PCP
            # code will probably remain
            try:
                hidden_states, router_logits, input_ids, topk = self._maybe_dispatch(
                    hidden_states,
                    router_logits,
                    input_ids,
                    topk,
                )

                try:
                    shared_output, hidden_states = self._apply_quant_method(
                        hidden_states=hidden_states,
                        router_logits=router_logits,
                        shared_experts_input=shared_experts_input,
                        input_ids=input_ids,
                        topk=topk,
                    )
                finally:
                    self._clear_pcp_moe_a1q_scale()

                return self._maybe_combine(
                    shared_output,
                    hidden_states,
                )
            except Exception:
                self._restore_pcp_is_padding()
                raise

    #########################################################
    #
    # Old methods from FusedMoE layer. Remove when possible.
    #
    #########################################################

    #
    # Properties
    #

    @property
    def layer_id(self):
        # Delayed import to avoid circular dependency
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    #
    # Attributes still needed by models
    #

    @property
    def is_monolithic(self) -> bool:
        return self.routed_experts.quant_method.is_monolithic

    @property
    def activation(self) -> MoEActivation:
        return self.routed_experts.activation

    #
    # Expert maps
    #

    @property
    def expert_map_manager(self):
        """Forward to routed_experts.expert_map_manager for backward compatibility."""
        return self.routed_experts.expert_map_manager

    @property
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        return self.expert_map_manager.placement_strategy

    @property
    def expert_global_to_physical(self) -> torch.Tensor | None:
        tables = self.expert_map_manager.routing_tables
        return tables[0] if tables else None

    @property
    def expert_physical_to_global(self) -> torch.Tensor | None:
        """Routing table: physical expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[1] if tables else None

    @property
    def expert_local_to_global(self) -> torch.Tensor | None:
        """Routing table: local expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[2] if tables else None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self.routed_experts.expert_map

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self.routed_experts._expert_routing_tables()

    def update_expert_map(self):
        self.routed_experts.update_expert_map()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self.routed_experts._map_global_expert_id_to_local_expert_id(expert_id)

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        return self.routed_experts.get_expert_weights()

    #
    # EPLB
    #

    @property
    def eplb_state(self) -> EplbLayerState | None:
        return self.router.eplb_state

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        if self.router.eplb_state is not None:
            self.router.eplb_state.set_layer_state(
                moe_layer_idx,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )
