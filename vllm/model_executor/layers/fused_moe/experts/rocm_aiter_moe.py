# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum
from functools import lru_cache

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
    kMxfp4Static,
)


class QuantMethod(IntEnum):
    # This allows interfacing with AITER QuantType Enum
    # without importing the QuantType from AITER globally.

    # Note that these quantization methods are
    # supported in AITER package. However,
    # not all are used in this module.

    NO = 0  # a16w16
    PER_TENSOR = 1  # w8a8 (pre_Tensor)
    PER_TOKEN = 2  # w8a8/w8a4 (per_Token)
    BLOCK_1X32 = 3  # fp4x2
    BLOCK_1X128 = 4  # block quantized w8a8 (per_1x128)
    BLOCK_128x128 = 5  # block quantized w8a8 (per_128x128)


class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1


aiter_topK_meta_data: tuple[torch.Tensor, torch.Tensor] | None = None


@lru_cache(maxsize=1)
def init_aiter_topK_meta_data(
    n_routed_experts: int,
    n_shared_experts: int,
    top_k: int,
    tp_rank: int,
    tp_size: int,
    shared_experts_score: float = 1.0,
    max_num_tokens: int = 32768,
    is_EP: bool = False,
):
    global aiter_topK_meta_data
    fake_expertid = n_routed_experts + n_shared_experts

    # all layers reuse same buffer
    # This extra element when EP is enabled is used as a sentinel
    # to mask out shared expert processing for tokens not owned by
    # the current EP rank. This is necessary to avoid double-processing
    # of shared experts.
    total_topk_ids = torch.empty(
        (max_num_tokens, top_k + n_shared_experts + is_EP),
        dtype=torch.int32,
        device="cuda",
    )
    ns_topk_ids, s_topk_ids = total_topk_ids.split(
        [top_k, n_shared_experts + is_EP], dim=1
    )
    shared_expert_ids = [n_routed_experts + i for i in range(n_shared_experts + is_EP)]
    if is_EP:
        s_topk_ids_list = [
            [fake_expertid] * (n_shared_experts + is_EP)
        ] * max_num_tokens
        for i in range(tp_rank, max_num_tokens, tp_size):
            s_topk_ids_list[i] = shared_expert_ids
    else:
        s_topk_ids_list = [
            list(range(n_routed_experts, fake_expertid))
        ] * max_num_tokens
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=torch.int32, device="cuda")

    total_topk_weights = torch.empty(
        (max_num_tokens, top_k + n_shared_experts + is_EP),
        dtype=torch.float32,
        device="cuda",
    )
    ns_topk_weights, s_topk_weights = total_topk_weights.split(
        [top_k, n_shared_experts + is_EP], dim=1
    )
    s_topk_weights.fill_(shared_experts_score)
    assert aiter_topK_meta_data is None, "AITER topK meta data is already initialized"
    aiter_topK_meta_data = (total_topk_weights, total_topk_ids)


def inject_shared_expert_weights(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_fused_shared_experts: int,
    shared_expert_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge routed topk results with the shared expert buffer and inject
    dynamic per-token shared expert gate values for AITER fusion.

    For routers that already return the combined buffer (e.g. GroupedTopKRouter
    via rocm_aiter_grouped_topk), only the dynamic weight injection is needed.
    For routers that return only routed slots (e.g. FusedTopKRouter), this also
    copies the routed results into the pre-allocated combined buffer.
    """
    if num_fused_shared_experts == 0:
        return topk_weights, topk_ids

    assert aiter_topK_meta_data is not None, (
        "aiter_topK_meta_data is not initialized but "
        "num_fused_shared_experts > 0. Ensure init_aiter_topK_meta_data "
        "is called before routing."
    )

    total_topk_weights, total_topk_ids = aiter_topK_meta_data
    token = topk_weights.shape[0]

    assert total_topk_weights.shape[0] >= token, (
        f"AITER topK meta data supports {total_topk_weights.shape[0]} "
        f"tokens, but got {token} tokens."
    )

    total_topk_weights_slice = total_topk_weights[:token]
    total_topk_ids_slice = total_topk_ids[:token]

    if topk_weights.shape[1] == topk:
        total_topk_weights_slice[:, :topk] = topk_weights
        total_topk_ids_slice[:, :topk] = topk_ids
        topk_weights = total_topk_weights_slice
        topk_ids = total_topk_ids_slice

    if shared_expert_weights is not None:
        topk_weights[:, topk : topk + num_fused_shared_experts] = shared_expert_weights[
            :token
        ]

    return topk_weights, topk_ids


def rocm_aiter_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = hidden_states.shape[0]
    device = hidden_states.device
    if (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        and num_fused_shared_experts > 0
    ):
        assert aiter_topK_meta_data is not None, (
            "AITER topK meta data is not initialized. "
            "Please ensure that init_aiter_topK_meta_data "
            "is called before this function."
        )
        total_topk_weights, total_topk_ids = aiter_topK_meta_data
        assert total_topk_weights.shape[0] >= token, (
            f"AITER topK meta data support {total_topk_weights.shape[0]} "
            f"tokens which is determined by max_num_batched_tokens, "
            f"but got {token} tokens now."
        )
        total_topk_weights = total_topk_weights[:token]
        total_topk_ids = total_topk_ids[:token]
        topk_weights, _ = total_topk_weights.split(
            [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = total_topk_ids.split(
            [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    if e_score_correction_bias is not None:
        rocm_aiter_ops.biased_grouped_topk(
            gating_output,
            e_score_correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )
    else:
        assert scoring_func == "softmax" or scoring_func == "sigmoid"
        rocm_aiter_ops.grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    if (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        and num_fused_shared_experts > 0
    ):
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


logger = init_logger(__name__)

# Shapes already reported under VLLM_DSV4_MOE_LOG_SHAPES, so a 61-layer forward
# logs once rather than once per layer per step.
_LOGGED_MOE_SHAPES: set[tuple] = set()


def _mxfp4_packed_dtype() -> torch.dtype:
    """The dtype MXFP4 activations arrive in, two values to the byte.

    Older torch builds have no float4 dtype and aiter falls back to uint8, so
    resolve it the same way aiter.dtypes does rather than naming it directly.
    """
    return getattr(torch, "float4_e2m1fn_x2", torch.uint8)


def _is_prequantized_moe_input(hidden_states: torch.Tensor) -> bool:
    """Whether the expert input already crossed the PCP gather quantized.

    Both schemes are recognised: MXFP8 leaves an fp8 tensor of the full hidden
    width, MXFP4 an fp4x2 tensor of half of it. Neither is a dtype the experts
    would see from an ordinary bf16 forward, so the dtype alone is the signal.
    """
    return hidden_states.dtype in (
        current_platform.fp8_dtype(),
        _mxfp4_packed_dtype(),
    )


def _logical_hidden_dim(hidden_states: torch.Tensor) -> int:
    """hidden_states' hidden dim in values, undoing MXFP4's 2-per-byte packing.

    Only the fp4x2 case is packed. fp8 and bf16 both store one value per
    element, so their trailing extent is already the logical width.

    Reads shape[-1] rather than shape[1] because moe_problem_size is also
    reached with a 3D (E, M, K) activation.
    """
    width = hidden_states.shape[-1]
    if hidden_states.dtype == _mxfp4_packed_dtype():
        return width * 2
    return width


def rocm_aiter_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_config: FusedMoEConfig,
    activation: MoEActivation = MoEActivation.SILU,
    apply_router_weight_on_input: bool = False,
    expert_mask: torch.Tensor | None = None,
    quant_config: FusedMoEQuantConfig | None = None,
    a1q_scale: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """ROCm AITER fused MoE expert computation."""
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    # Gate/up interleave hint; only the SWIGLUOAI activations override it.
    activation_interleave = None
    if activation == MoEActivation.SILU:
        activation_method = ActivationMethod.SILU
    elif activation == MoEActivation.GELU:
        activation_method = ActivationMethod.GELU
    elif activation == MoEActivation.SWIGLUOAI:
        activation_method = rocm_aiter_ops.get_aiter_activation_type("swiglu")
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        activation_method = rocm_aiter_ops.get_aiter_activation_type("swiglu")
        activation_interleave = False
    elif activation == MoEActivation.SITU:
        activation_method = rocm_aiter_ops.get_aiter_activation_type("situ")
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # w8a8 per-channel quantization
    if (
        quant_config.per_act_token_quant
        and apply_router_weight_on_input
        and quant_config.use_fp8_w8a8
    ):
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`
        # This applies topk_weights on the GEMM output of the first FC layer
        #  rather than the second FC.
        assert topk_weights.dim() == 2, (
            "`topk_weights` should be in shape (num_tokens, topk)"
        )
        assert topk_weights.shape[-1] == 1, (
            "Only support topk=1 when `apply_router_weight_on_input` is True"
        )
        assert num_local_tokens is None, (
            "AITER tkw1 kernel does not support `num_local_tokens`"
        )

        return rocm_aiter_ops.asm_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale=quant_config.w1_scale,
            fc2_scale=quant_config.w2_scale,
            fc1_smooth_scale=None,
            fc2_smooth_scale=None,
            a16=False,
            per_tensor_quant_scale=None,
            expert_mask=expert_mask,
            activation_method=activation_method,
        )

    else:
        quant_method = QuantMethod.NO.value
        # mxfp4 i.e. w4a4, w4a16 uses BLOCK_1X32
        # mxfp6 and mxfp8 are unsupported in AITER currently and use emulation instead
        if quant_config.use_mxfp4_w4a4 or quant_config.use_mxfp4_w4a16:
            quant_method = QuantMethod.BLOCK_1X32.value
        # w8a8 block-scaled
        if quant_config.block_shape is not None and quant_config.use_fp8_w8a8:
            assert not apply_router_weight_on_input, (
                "apply_router_weight_on_input is not supported for block scaled moe"
            )
            assert quant_config.w1_scale is not None
            assert quant_config.w2_scale is not None
            quant_method = QuantMethod.BLOCK_128x128.value
        elif quant_config.use_fp8_w8a8 and quant_config.per_out_ch_quant:
            quant_method = QuantMethod.PER_TOKEN.value
        elif quant_config.use_fp8_w8a8:
            # Currently only per tensor quantization method is enabled.
            quant_method = QuantMethod.PER_TENSOR.value

        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, (
                "`topk_weights` should be in shape (num_tokens, topk)"
            )
            _, topk = topk_weights.shape
            assert topk == 1, (
                "Only support topk=1 when `apply_router_weight_on_input` is True"
            )

        # Compute padding on-the-fly for CK MXFP4 kernels
        hidden_pad = 0
        intermediate_pad = 0
        assert moe_config.hidden_dim_unpadded is not None
        assert moe_config.intermediate_size_per_partition_unpadded is not None
        # Not shape[1]: under A4W4 the activation is fp4x2 and shape[1] is half
        # the hidden dim, which would make this pad negative and drive the
        # kernel's padding logic off a cliff. The pad is a property of the
        # hidden dim, not of how many bytes it currently takes.
        hidden_pad = _logical_hidden_dim(hidden_states) - moe_config.hidden_dim_unpadded
        intermediate_pad = (
            (
                moe_config.intermediate_size_per_partition
                - moe_config.intermediate_size_per_partition_unpadded
            )
            if moe_config.intermediate_pad is None
            else moe_config.intermediate_pad
        )

        # Round hidden_pad/intermediate_pad to match AITER's CK/FlyDSL MoE
        # dispatch (currently pinned to v0.1.13.post1):
        # https://github.com/ROCm/aiter/blob/v0.1.13.post1/aiter/fused_moe.py#L1073
        # https://github.com/ROCm/aiter/blob/v0.1.13.post1/aiter/fused_moe.py#L1099
        # TODO: Revisit this once we bump AITER to 0.1.15 with padding fixes
        # for CK/FlyDSL MoE GEMM e.g. https://github.com/ROCm/aiter/pull/3401
        # SITU's A16W4 FlyDSL kernel pads per gate/up half; pass through unrounded.
        if activation != MoEActivation.SITU:
            hidden_pad = hidden_pad // 128 * 128
            intermediate_pad = (
                intermediate_pad // 64 * 64 * (2 if moe_config.tp_size == 1 else 1)
            )

        # https://github.com/ROCm/aiter/pull/3123 specialized the AITER stage1 GEMMs
        # for interleaved vs separated gate and up weights.
        # For gpt-oss i.e. use_mxfp4_w4a16=True, the weights are shuffled by
        # `rocm_aiter_ops.shuffle_weight_a16w4` in `oracle/mxfp4.py`,
        # which always sets `is_guinterleave=True`.
        # Hence, we pass in GateMode.INTERLEAVE to match the weight shuffling.
        from aiter.ops.flydsl.moe_common import GateMode

        # One line per distinct expert-call shape, for matching a run against
        # aiter's tuning tables. Which entry aiter picks is keyed on M, the
        # per-rank inter_dim and the activation dtype, and none of those are
        # obvious from the launch config -- EP-off flattens PCP into MoE TP and
        # shards inter_dim, so the shape the kernels see is not the model's.
        # Off unless VLLM_DSV4_MOE_LOG_SHAPES=1; deduplicated, so it cannot
        # turn into per-layer spam.
        if envs.VLLM_DSV4_MOE_LOG_SHAPES:
            key = (
                hidden_states.shape[0],
                _logical_hidden_dim(hidden_states),
                w1.shape[1] // 2,
                str(hidden_states.dtype),
            )
            if key not in _LOGGED_MOE_SHAPES:
                _LOGGED_MOE_SHAPES.add(key)
                logger.info(
                    "AITER MoE shape: M=%d model_dim=%d inter_dim(per rank)=%d "
                    "E=%d topk=%d act_dtype=%s hidden_pad=%d intermediate_pad=%d",
                    key[0], key[1], key[2], w1.shape[0], topk_ids.shape[1],
                    key[3], hidden_pad, intermediate_pad,
                )

        gate_mode = ""
        if activation == MoEActivation.SITU:
            # a8w4 (VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1) uses the gate/up-
            # interleaved (_gui_) fp8 flydsl kernels; default a16w4 SiTU stays
            # separated.
            gate_mode = (
                GateMode.INTERLEAVE.value
                if rocm_aiter_ops.is_fused_moe_situv2_a8w4_enabled()
                else GateMode.SEPARATED.value
            )
        elif quant_config.use_mxfp4_w4a16:
            # A4W4 has to ask for SEPARATED, and not as a preference: for SiLU
            # aiter picks the activation dtype from gate_mode, taking fp8 on
            # INTERLEAVE and reaching fp4x2 only on the SEPARATED fallthrough.
            # There is no interleaved A4W4 kernel to ask for instead -- every
            # flydsl_moe1_afp4_* entry in the tuned and untuned tables is
            # separated-layout, none carry the _gui_ tag the afp8 ones do.
            # oracle/mxfp4.py shuffles w13 to match off the same env var.
            gate_mode = (
                GateMode.SEPARATED.value
                if envs.VLLM_DSV4_MOE_A4W4
                else GateMode.INTERLEAVE.value
            )
        elif activation_interleave is not None:
            gate_mode = (
                GateMode.INTERLEAVE.value
                if activation_interleave
                else GateMode.SEPARATED.value
            )

        return rocm_aiter_ops.fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_mask=expert_mask,
            quant_method=quant_method,
            activation_method=activation_method,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            a1_scale=quant_config.a1_scale if a1q_scale is None else a1q_scale,
            a2_scale=quant_config.a2_scale,
            doweight_stage1=apply_router_weight_on_input,
            num_local_tokens=num_local_tokens,
            output_dtype=output_dtype,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            gate_mode=gate_mode,
            bias1=quant_config.w1_bias if quant_config.use_mxfp4_w4a16 else None,
            bias2=quant_config.w2_bias if quant_config.use_mxfp4_w4a16 else None,
            moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
            beta=moe_config.activation_situ_beta,
            linear_beta=moe_config.activation_situ_linear_beta,
        )


class AiterExperts(mk.FusedMoEExpertsModular):
    consumes_expert_mask = True

    @property
    def expects_unquantized_inputs(self) -> bool:
        # When paired with MoRI, the prepare/finalize handles FP8
        # quantization during dispatch to reduce network traffic,
        # so we should not defer input quantization.
        # Otherwise, AITER fused MoE kernels handle input quantization
        # internally via a single fused kernel.
        return not self.moe_config.use_mori_kernels

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not is_supported and not rocm_aiter_ops.is_fused_moe_enabled():
            reason = (
                f"{reason}. AITER MoE is not enabled — "
                "set VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 "
                "to enable it"
            )
        return is_supported, reason

    @staticmethod
    def _supports_current_device() -> bool:
        return rocm_aiter_ops.is_fused_moe_enabled()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        if (weight_key, activation_key) not in SUPPORTED_W_A:
            return False
        if weight_key == kMxfp4Static:
            from vllm.platforms.rocm import on_gfx950, on_gfx1250

            if not on_gfx950() or on_gfx1250():
                return False
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SITU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """K from the hidden dim, not from the activation's trailing extent.

        The base implementation reads K off ``a1.size(-1)``, which is the same
        thing for bf16 and for MXFP8 but not for MXFP4: fp4x2 packs two values
        per byte, so a pre-quantized A4W4 input arrives 3584 wide for a hidden
        dim of 7168. K feeds workspace_shapes, which sizes the *output* buffer
        -- and the output is bf16 at the full hidden dim regardless of how the
        input was quantized, so leaving it packed makes the experts' result and
        the buffer it is copied into disagree by exactly 2x.

        The base docstring anticipates this case ("the int4 kernels divide the
        trailing dimension by two, so it's not 'correct' to extract N or K from
        the trailing dimension") and invites the override.
        """
        E, M, N, K, topk = super().moe_problem_size(a1, w1, w2, topk_ids)
        return E, M, N, _logical_hidden_dim(a1), topk

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Workspaces are managed internally by AITER.
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # TODO(rob): rocm_aiter_fused_experts uses self.quant_config's
        # a_scales for static quantization. Update this to fit better
        # with the interface once all quant integrations are complete.

        if expert_tokens_meta is not None:
            num_local_tokens = expert_tokens_meta.expert_num_tokens
        else:
            num_local_tokens = None

        if a1q_scale is None and _is_prequantized_moe_input(hidden_states):
            # The PCP all-gather carried the expert input already quantized to
            # save bytes on the wire, so the scales come from the forward
            # context rather than from prepare(). aiter recognises the
            # pre-quantized pair and skips the quantization it usually fuses
            # into the sort -- it has one such branch per scheme, keyed on the
            # activation dtype being fp8 or fp4x2 with a1_scale set. See
            # MoERunner._pcp_all_gather_expert_input.
            a1q_scale = get_forward_context().pcp_moe_a1q_scale
            assert a1q_scale is not None, (
                "AITER experts got pre-quantized activations with no MX scales; "
                "the gather quantized them but the scales did not survive"
            )

        result = rocm_aiter_fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_mask=expert_map,
            quant_config=self.quant_config,
            moe_config=self.moe_config,
            a1q_scale=a1q_scale,
            num_local_tokens=num_local_tokens,
            output_dtype=output.dtype,
            moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
        )
        # avoid redundant copy when output is a view of the result
        if (
            output.shape == result.shape
            and output.dtype == result.dtype
            and output.device == result.device
            and output.is_contiguous()
            and result.is_contiguous()
            and output._base is None
        ):
            output.set_(result)
        else:
            output.copy_(result)
