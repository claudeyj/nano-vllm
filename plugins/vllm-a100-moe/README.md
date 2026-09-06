# vLLM A100 MoE plugin

This optional nano-vLLM plugin executes the routed experts in
`Qwen/Qwen3-30B-A3B` with vLLM's unquantized Triton fused-MoE kernel. It
supports FP16/BF16 on Ampere GPUs, including A100, and works with nano-vLLM's
single-rank and expert-parallel Qwen3 layouts.

The plugin imports the kernel from the installed `vllm` package. This vLLM
checkout uses newer Torch and Triton releases than nano-vLLM's package metadata.
Create the environment from vLLM, then install nano-vLLM and this plugin without
letting the resolver downgrade those packages:

```bash
uv pip install -e /path/to/vllm
uv pip install --no-deps -e /path/to/nano-vllm
uv pip install --no-deps -e /path/to/nano-vllm/plugins/vllm-a100-moe
```

The current local pair is developed against vLLM's Torch 2.13/Triton 3.7
environment. Running nano-vLLM on that newer stack is part of this plugin's
compatibility contract; nano-vLLM's standalone pinned environment remains
unchanged.

Select it when constructing the engine:

```python
llm = LLM(
    "/path/to/Qwen3-30B-A3B",
    expert_parallel_size=8,
    moe_kernel_plugin="vllm_a100_moe",
    enforce_eager=True,
)
```

The plugin uses vLLM's `expert_map` support so each EP rank evaluates only its
local experts. The per-rank results are combined with an all-reduce. It does
not currently support nano-vLLM's ordered expert-sum correctness mode.

On first use, each layer packs its local per-expert parameters into vLLM's
stacked weight layout. This temporarily requires enough memory for both layouts.
