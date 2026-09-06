"""vLLM fused-MoE kernel integration for nano-vLLM."""

from __future__ import annotations

import torch
import torch.distributed as dist


class VllmA100MoePlugin:
    """Run Qwen3 routed experts with vLLM's Triton fused-MoE kernel."""

    operation = "qwen3_sparse_moe"

    def __init__(self) -> None:
        self._weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._expert_maps: dict[tuple[int, torch.device], torch.Tensor] = {}

    @staticmethod
    def _fused_experts():
        try:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                fused_experts,
            )
        except ImportError as exc:
            raise RuntimeError(
                "vllm_a100_moe requires an installed vLLM checkout with "
                "the Triton fused-MoE implementation"
            ) from exc
        return fused_experts

    def _stack_local_weights(self, layer) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = id(layer)
        cached = self._weights.get(cache_key)
        if cached is not None:
            return cached

        if layer.enable_expert_parallel:
            expert_ids = range(layer.expert_start, layer.expert_end)
        else:
            expert_ids = range(layer.num_experts)
        experts = [layer.experts[expert_id] for expert_id in expert_ids]
        w1 = torch.stack(
            [expert.gate_up_proj.weight for expert in experts]
        ).contiguous()
        w2 = torch.stack(
            [expert.down_proj.weight for expert in experts]
        ).contiguous()
        self._weights[cache_key] = (w1, w2)
        return w1, w2

    def _expert_map(self, layer, device: torch.device) -> torch.Tensor | None:
        if not layer.enable_expert_parallel:
            return None
        cache_key = (id(layer), device)
        cached = self._expert_maps.get(cache_key)
        if cached is not None:
            return cached
        expert_map = torch.full(
            (layer.num_experts,), -1, dtype=torch.int32, device=device
        )
        expert_map[layer.expert_start : layer.expert_end] = torch.arange(
            layer.num_local_experts, dtype=torch.int32, device=device
        )
        self._expert_maps[cache_key] = expert_map
        return expert_map

    def forward(self, layer, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            raise RuntimeError("vllm_a100_moe requires a CUDA device")
        major, _ = torch.cuda.get_device_capability(hidden_states.device)
        if major != 8:
            raise RuntimeError(
                "vllm_a100_moe targets Ampere SM80/SM86 GPUs; "
                f"got compute capability {major}"
            )
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "vllm_a100_moe supports float16 and bfloat16 activations"
            )
        if layer.enable_ordered_expert_sum:
            raise RuntimeError(
                "vllm_a100_moe does not support enable_ordered_expert_sum"
            )

        moe_start = layer._profile_mark()
        if layer.enable_ep_profiling:
            layer._profile_calls += 1
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        router_logits = layer.gate(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, layer.top_k, dim=-1
        )
        if layer.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        layer._profile_routing(selected_experts, hidden_states.size(0))

        expert_start = layer._profile_mark()
        w1, w2 = self._stack_local_weights(layer)
        output = self._fused_experts()(
            hidden_states,
            w1,
            w2,
            routing_weights,
            selected_experts,
            global_num_experts=layer.num_experts,
            expert_map=self._expert_map(layer, hidden_states.device),
        )
        expert_end = layer._profile_mark()
        layer._profile_interval("expert_compute", expert_start, expert_end)
        if layer.enable_expert_parallel:
            combine_start = layer._profile_mark()
            dist.all_reduce(output)
            combine_end = layer._profile_mark()
            layer._profile_interval("combine", combine_start, combine_end)
        output = output.reshape(original_shape)
        moe_end = layer._profile_mark()
        layer._profile_interval("moe_total", moe_start, moe_end)
        return output


__all__ = ["VllmA100MoePlugin"]
