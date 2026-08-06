# Run nano-vLLM on custom prompts

This guide runs nano-vLLM directly, without `neval` or a benchmark dataset.

## Setup

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm
uv sync
```

Save the following as `run_prompts.py` in the project directory:

```python
import argparse
import json

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", help="JSON array of prompt strings")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--chat", action="store_true", help="apply chat template")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tp > 1 and args.ep > 1:
        raise SystemExit("TP and EP cannot currently be combined")

    prompts = list(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as file:
            file_prompts = json.load(file)
        if not isinstance(file_prompts, list) or not all(
            isinstance(prompt, str) for prompt in file_prompts
        ):
            raise SystemExit("--prompt-file must contain a JSON array of strings")
        prompts.extend(file_prompts)
    if not prompts:
        raise SystemExit("provide --prompt or --prompt-file")

    model_prompts = prompts
    if args.chat:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    llm = LLM(
        args.model,
        tensor_parallel_size=args.tp,
        expert_parallel_size=args.ep,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    outputs = llm.generate(model_prompts, sampling)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}\nCompletion: {output['text']}")


if __name__ == "__main__":
    main()
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

Pass `--prompt` more than once, or load a JSON array:

```json
[
  "Write a haiku about GPUs.",
  "What is mixture-of-experts inference?"
]
```

```bash
uv run python run_prompts.py --model /path/to/model --ep 8 --chat \
  --prompt-file prompts.json --temperature 0 --max-tokens 128
```

Temperature zero uses greedy argmax. Use a positive temperature for sampling.

## Current limits

- TP and EP cannot be enabled together.
- The model's number of experts must be divisible by EP.
- TP must divide both the attention-head and KV-head counts. For
  Qwen3-30B-A3B, supported TP sizes are 1, 2, and 4.
