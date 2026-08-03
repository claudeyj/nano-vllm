import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.parallel_state import get_ep_rank, get_ep_size, get_tp_size


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3ExpertMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedReplicatedLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = ReplicatedLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3SparseMoeBlock(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.enable_expert_parallel = getattr(config, "enable_expert_parallel", False)
        self.enable_ordered_expert_sum = getattr(
            config, "enable_ordered_expert_sum", False
        )
        self.enable_ep_profiling = getattr(
            config, "enable_ep_profiling", False
        )
        self.ep_size = get_ep_size() if self.enable_expert_parallel else 1
        self.ep_rank = get_ep_rank() if self.enable_expert_parallel else 0
        assert self.num_experts % self.ep_size == 0, (
            f"num_experts ({self.num_experts}) must be divisible by expert "
            f"parallel size ({self.ep_size})"
        )
        self.num_local_experts = self.num_experts // self.ep_size
        self.expert_start = self.ep_rank * self.num_local_experts
        self.expert_end = self.expert_start + self.num_local_experts
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
        )
        self.experts = nn.ModuleList([
            Qwen3ExpertMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            )
            if self.enable_expert_parallel
            and self.expert_start <= expert_idx < self.expert_end
            else Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            )
            if not self.enable_expert_parallel
            else nn.Identity()
            for expert_idx in range(self.num_experts)
        ])
        self.reset_profile_stats()

    def reset_profile_stats(self) -> None:
        self._profile_calls = 0
        self._profile_token_layers = 0
        self._profile_assignments = 0
        self._profile_dispatch_bytes = 0
        self._profile_return_bytes = 0
        self._profile_expert_counts = None
        self._profile_events = {}

    def _profile_mark(self):
        if not self.enable_ep_profiling:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _profile_interval(self, name: str, start, end) -> None:
        if start is not None:
            self._profile_events.setdefault(name, []).append((start, end))

    def _profile_routing(self, experts: torch.Tensor, num_tokens: int) -> None:
        if not self.enable_ep_profiling:
            return
        counts = torch.bincount(
            experts.reshape(-1), minlength=self.num_experts
        ).to(torch.int64)
        if self._profile_expert_counts is None:
            self._profile_expert_counts = torch.zeros_like(counts)
        self._profile_expert_counts += counts
        self._profile_token_layers += num_tokens
        self._profile_assignments += experts.numel()

    def get_profile_stats(self) -> dict:
        timings = {
            f"{name}_gpu_ms": sum(start.elapsed_time(end) for start, end in events)
            for name, events in self._profile_events.items()
        }
        expert_counts = (
            self._profile_expert_counts.cpu().tolist()
            if self._profile_expert_counts is not None
            else [0] * self.num_experts
        )
        return {
            "layer": self.layer_idx,
            "calls": self._profile_calls,
            "token_layers": self._profile_token_layers,
            "assignments": self._profile_assignments,
            "expert_counts": expert_counts,
            "dispatch_bytes": self._profile_dispatch_bytes,
            "return_bytes": self._profile_return_bytes,
            "timings": timings,
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        moe_start = self._profile_mark()
        if self.enable_ep_profiling:
            self._profile_calls += 1
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1])
        router_logits = self.gate(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self.enable_expert_parallel:
            output = self._forward_expert_parallel(
                hidden_states, routing_weights, selected_experts
            )
        else:
            self._profile_routing(selected_experts, hidden_states.size(0))
            expert_start = self._profile_mark()
            output_dtype = (
                torch.float32
                if self.enable_ordered_expert_sum
                else hidden_states.dtype
            )
            output = torch.zeros_like(hidden_states, dtype=output_dtype)
            for expert_idx, expert in enumerate(self.experts):
                token_idx, top_k_idx = torch.where(
                    selected_experts == expert_idx
                )
                if token_idx.numel() == 0:
                    continue
                expert_output = expert(hidden_states[token_idx])
                if self.enable_ordered_expert_sum:
                    expert_output = expert_output.float()
                    expert_output *= routing_weights[
                        token_idx, top_k_idx, None
                    ].float()
                else:
                    expert_output *= routing_weights[token_idx, top_k_idx, None]
                output.index_add_(0, token_idx, expert_output)
            if self.enable_ordered_expert_sum:
                output = output.to(hidden_states.dtype)
            expert_end = self._profile_mark()
            self._profile_interval(
                "expert_compute_and_combine", expert_start, expert_end
            )
        output = output.reshape(original_shape)
        moe_end = self._profile_mark()
        self._profile_interval("moe_total", moe_start, moe_end)
        return output

    def _forward_expert_parallel(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        dispatch_setup_start = self._profile_mark()
        num_tokens, hidden_size = hidden_states.shape
        # The fast path stripes replicated inputs across ranks. Correctness
        # mode instead uses rank 0 as the canonical source, matching EP=1 even
        # if preceding dense kernels differ slightly between GPUs.
        if self.enable_ordered_expert_sum and self.ep_rank == 0:
            source_tokens = torch.arange(num_tokens, device=hidden_states.device)
        elif self.enable_ordered_expert_sum:
            source_tokens = torch.empty(
                0, dtype=torch.long, device=hidden_states.device
            )
        elif self.ep_rank < num_tokens:
            source_tokens = torch.arange(
                self.ep_rank,
                num_tokens,
                self.ep_size,
                device=hidden_states.device,
            )
        else:
            source_tokens = torch.empty(
                0, dtype=torch.long, device=hidden_states.device
            )
        source_experts = selected_experts[source_tokens]
        send_tokens = source_tokens[:, None].expand(-1, self.top_k).reshape(-1)
        send_experts = source_experts.reshape(-1)
        send_weights = routing_weights[source_tokens].reshape(-1)
        self._profile_routing(source_experts, source_tokens.numel())
        if self.enable_ep_profiling:
            num_assignments = send_experts.numel()
            self._profile_dispatch_bytes += (
                num_assignments * hidden_size * hidden_states.element_size()
                + num_assignments * send_experts.element_size()
                + self.ep_size * 8
            )
            self._profile_return_bytes += (
                num_assignments * hidden_size * hidden_states.element_size()
            )
        destinations = send_experts // self.num_local_experts
        order = torch.argsort(destinations)
        send_tokens = send_tokens[order]
        send_experts = send_experts[order]
        send_weights = send_weights[order]
        destinations = destinations[order]

        send_counts_tensor = torch.bincount(
            destinations, minlength=self.ep_size
        ).to(torch.int64)
        recv_counts_tensor = torch.empty_like(send_counts_tensor)
        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
        send_counts = send_counts_tensor.cpu().tolist()
        recv_counts = recv_counts_tensor.cpu().tolist()
        num_recv = sum(recv_counts)
        dispatch_setup_end = self._profile_mark()
        self._profile_interval(
            "dispatch_setup", dispatch_setup_start, dispatch_setup_end
        )

        dispatch_collective_start = self._profile_mark()
        dispatched_states = hidden_states[send_tokens].contiguous()
        received_states = hidden_states.new_empty((num_recv, hidden_size))
        received_experts = send_experts.new_empty(num_recv)
        received_tokens = (
            send_tokens.new_empty(num_recv)
            if self.enable_ordered_expert_sum
            else None
        )
        dist.all_to_all_single(
            received_states,
            dispatched_states,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        dist.all_to_all_single(
            received_experts,
            send_experts,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        if received_tokens is not None:
            dist.all_to_all_single(
                received_tokens,
                send_tokens,
                output_split_sizes=recv_counts,
                input_split_sizes=send_counts,
            )
        dispatch_collective_end = self._profile_mark()
        self._profile_interval(
            "dispatch_collective",
            dispatch_collective_start,
            dispatch_collective_end,
        )

        expert_start = self._profile_mark()
        received_outputs = torch.zeros_like(received_states)
        for expert_idx in range(self.expert_start, self.expert_end):
            assignment_idx = torch.where(received_experts == expert_idx)[0]
            if assignment_idx.numel() == 0:
                continue
            if received_tokens is not None:
                # EP=1 obtains assignments from torch.where in global token
                # order. all-to-all groups them by source rank, so restore the
                # EP=1 row order before the expert GEMMs in correctness mode.
                assignment_idx = assignment_idx[
                    torch.argsort(received_tokens[assignment_idx])
                ]
            received_outputs[assignment_idx] = self.experts[expert_idx](
                received_states[assignment_idx]
            )
        expert_end = self._profile_mark()
        self._profile_interval("expert_compute", expert_start, expert_end)

        return_start = self._profile_mark()
        returned_outputs = hidden_states.new_empty(
            (send_tokens.numel(), hidden_size)
        )
        dist.all_to_all_single(
            returned_outputs,
            received_outputs,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
        )
        return_end = self._profile_mark()
        self._profile_interval("return_collective", return_start, return_end)
        combine_start = self._profile_mark()
        if self.enable_ordered_expert_sum:
            returned_outputs = returned_outputs.float()
            returned_outputs *= send_weights[:, None].float()
            output = torch.zeros_like(hidden_states, dtype=torch.float32)
            for expert_idx in range(self.num_experts):
                assignment_idx = torch.where(send_experts == expert_idx)[0]
                if assignment_idx.numel() == 0:
                    continue
                output.index_add_(
                    0,
                    send_tokens[assignment_idx],
                    returned_outputs[assignment_idx],
                )
        else:
            returned_outputs *= send_weights[:, None]
            output = torch.zeros_like(hidden_states)
            output.index_add_(0, send_tokens, returned_outputs)
        if self.enable_ordered_expert_sum:
            # Only rank 0 owns source tokens in correctness mode. Replicate its
            # canonical EP=1-ordered result without another reduction.
            dist.broadcast(output, src=0)
            output = output.to(hidden_states.dtype)
        else:
            # Restore the replicated layout expected by the surrounding dense
            # and attention layers.
            dist.all_reduce(output)
        combine_end = self._profile_mark()
        self._profile_interval("combine_reduce", combine_start, combine_end)
        return output


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        is_sparse = (
            getattr(config, "num_experts", 0) > 0
            and (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
            and layer_idx not in getattr(config, "mlp_only_layers", [])
        )

        if is_sparse:
            self.mlp = Qwen3SparseMoeBlock(config, layer_idx)
        else:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def should_skip_weight(self, weight_name: str) -> bool:
        marker = ".mlp.experts."
        if marker not in weight_name:
            return False
        expert_idx = int(weight_name.split(marker, 1)[1].split(".", 1)[0])
        layer_idx = int(weight_name.split("model.layers.", 1)[1].split(".", 1)[0])
        mlp = self.model.layers[layer_idx].mlp
        if not getattr(mlp, "enable_expert_parallel", False):
            return False
        return not (mlp.expert_start <= expert_idx < mlp.expert_end)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
