# Evaluate nano-vLLM with neval

[`neval`](https://github.com/claudeyj/neval) compares nano-vLLM
configurations for correctness and performance. The commands below assume
`nano-vllm` and `neval` are sibling directories.

## Setup

```bash
git clone https://github.com/claudeyj/nano-vllm.git
git clone https://github.com/claudeyj/neval.git
uv sync --project nano-vllm

mkdir -p neval/benchmark_data
wget -P neval/benchmark_data \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

The dataset is downloaded separately because `neval/benchmark_data` is
gitignored.

## Performance matrix

| Name | GPUs | Engine arguments |
| --- | ---: | --- |
| EP1 baseline | 1 | `TP=1, EP=1` |
| TP4 | 4 | `TP=4, EP=1` |
| EP4 | 4 | `TP=1, EP=4` |
| EP8 | 8 | `TP=1, EP=8` |

Set the model path, then run every configuration with the same workload:

```bash
MODEL_PATH=/path/to/Qwen3-30B-A3B
DATASET_PATH=neval/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json

declare -A ENGINE_CONFIGS=(
  [ep1]='{"tensor_parallel_size":1,"expert_parallel_size":1}'
  [tp4]='{"tensor_parallel_size":4,"expert_parallel_size":1}'
  [ep4]='{"tensor_parallel_size":1,"expert_parallel_size":4}'
  [ep8]='{"tensor_parallel_size":1,"expert_parallel_size":8}'
)

for config_name in ep1 tp4 ep4 ep8; do
  uv run --project nano-vllm --with-editable ./neval neval profile \
    --backend nanovllm --name "nano-${config_name}-perf" \
    --model "${MODEL_PATH}" \
    --benchmark sharegpt --dataset "${DATASET_PATH}" \
    --num-prompts 10 --output "nano-${config_name}-perf.json" \
    --engine-args "${ENGINE_CONFIGS[$config_name]}" \
    --sampling-args '{"temperature":0,"max_tokens":128,"ignore_eos":true}'
done
```

Compare each configuration with the EP1 baseline:

```bash
for config_name in tp4 ep4 ep8; do
  uv run --project neval neval perf-compare \
    nano-ep1-perf.json "nano-${config_name}-perf.json" \
    --report "nano-ep1-vs-${config_name}-perf.json"
done
```

For an EP timing and routing breakdown, add `--profile-ep` to both profile
runs being compared. This records collective and expert-compute time, logical
communication volume, expert load balance, and peak memory. CUDA events add
overhead, so omit this flag when measuring headline throughput.

## Correctness

Replace `profile` with `capture`, use a smaller output length, and add ordered
expert accumulation to every configuration:

```bash
uv run --project nano-vllm --with-editable ./neval neval capture \
  --backend nanovllm --name nano-ep8 --model "${MODEL_PATH}" \
  --benchmark sharegpt --dataset "${DATASET_PATH}" --num-prompts 10 \
  --output nano-ep8.json \
  --engine-args '{"tensor_parallel_size":1,"expert_parallel_size":8,"enable_ordered_expert_sum":true}' \
  --sampling-args '{"temperature":0,"max_tokens":32,"ignore_eos":true}'

uv run --project neval neval compare nano-ep1.json nano-ep8.json \
  --report nano-ep1-vs-ep8-correctness.json
```

Capture EP1 with the same arguments except `expert_parallel_size=1` before
running the comparison. Do not enable `enable_ordered_expert_sum` for
performance tests.

## Limits

- TP and EP cannot currently be combined in nano-vLLM.
- Qwen3-30B-A3B has four KV heads, so nano-vLLM supports TP1, TP2, or TP4;
  KV-head replication for TP8 is not implemented.
- The number of experts must be divisible by the EP size.
