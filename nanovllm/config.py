import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    enable_expert_parallel: bool = False
    enable_ordered_expert_sum: bool = False
    enable_ep_profiling: bool = False
    moe_kernel_plugin: str = ""
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.expert_parallel_size <= 8
        if self.enable_expert_parallel and self.expert_parallel_size == 1:
            self.expert_parallel_size = self.tensor_parallel_size
            self.tensor_parallel_size = 1
        assert self.tensor_parallel_size == 1 or self.expert_parallel_size == 1, (
            "Combining tensor parallelism and expert parallelism is not supported"
        )
        self.enable_expert_parallel = self.expert_parallel_size > 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.enable_expert_parallel = self.enable_expert_parallel
        self.hf_config.enable_ordered_expert_sum = self.enable_ordered_expert_sum
        self.hf_config.enable_ep_profiling = self.enable_ep_profiling
        self.hf_config.moe_kernel_plugin = self.moe_kernel_plugin
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
