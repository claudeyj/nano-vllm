# Run nano-vLLM on custom prompts

This workflow runs nano-vLLM directly, without `neval` or a benchmark
dataset.

## Setup

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm
uv sync
```

## Run configurations

Replace `/path/to/model` with a local model directory. Add `--chat` for
instruction/chat models such as Qwen3.

```bash
# One GPU: TP1 / EP1 baseline
uv run python run_prompts.py --model /path/to/model --tp 1 --ep 1 \
  --chat --prompt "Explain expert parallelism in three sentences."

# Four GPUs: tensor parallelism
uv run python run_prompts.py --model /path/to/model --tp 4 --ep 1 \
  --chat --prompt "Explain expert parallelism in three sentences."

# Four GPUs: expert parallelism
uv run python run_prompts.py --model /path/to/model --tp 1 --ep 4 \
  --chat --prompt "Explain expert parallelism in three sentences."

# Eight GPUs: expert parallelism
uv run python run_prompts.py --model /path/to/model --tp 1 --ep 8 \
  --chat --prompt "Explain expert parallelism in three sentences."
```

Pass `--prompt` more than once, or use `--prompt-file prompts.json` with a
JSON array of strings:

```bash
uv run python run_prompts.py --model /path/to/model --ep 8 --chat \
  --prompt-file prompts.json --temperature 0 --max-tokens 128
```

Run `uv run python run_prompts.py --help` for all options. Temperature zero
uses greedy argmax; use a positive temperature for sampling.

## Current limits

- TP and EP cannot be enabled together.
- The model's number of experts must be divisible by EP.
- TP must divide both the attention-head and KV-head counts. For
  Qwen3-30B-A3B, supported TP sizes are 1, 2, and 4.
