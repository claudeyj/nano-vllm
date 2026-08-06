import argparse
import json

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nano-vLLM on command-line or JSON prompts."
    )
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument(
        "--prompt", action="append", default=[], help="prompt; may be repeated"
    )
    parser.add_argument("--prompt-file", help="JSON array of prompt strings")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--ep", type=int, default=1, help="expert parallel size")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--chat", action="store_true", help="apply chat template")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def load_prompts(args) -> list[str]:
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
    return prompts


def main():
    args = parse_args()
    if args.tp > 1 and args.ep > 1:
        raise SystemExit("TP and EP cannot currently be combined")

    prompts = load_prompts(args)
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
