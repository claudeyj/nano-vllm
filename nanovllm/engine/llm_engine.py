import atexit
from dataclasses import fields
from time import perf_counter
from uuid import uuid4
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


def _new_shm_name() -> str:
    return f"nanovllm_{uuid4().hex}"


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        world_size = max(config.tensor_parallel_size, config.expert_parallel_size)
        shm_name = _new_shm_name() if world_size > 1 else None
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner, args=(config, i, event, shm_name)
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events, shm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        generation_start = perf_counter()
        prefill_seconds = decode_seconds = 0.0
        prefill_tokens = decode_tokens = 0
        prefill_steps = decode_steps = 0
        completion_seconds = {}
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            step_seconds = perf_counter() - t
            if num_tokens > 0:
                prefill_steps += 1
                prefill_tokens += num_tokens
                prefill_seconds += step_seconds
                prefill_throughput = num_tokens / step_seconds
            else:
                decode_steps += 1
                decode_tokens -= num_tokens
                decode_seconds += step_seconds
                decode_throughput = -num_tokens / step_seconds
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                completion_seconds[seq_id] = perf_counter() - generation_start
                pbar.update(1)
        pbar.close()
        generation_seconds = perf_counter() - generation_start
        self.last_generation_stats = {
            "generation_seconds": generation_seconds,
            "prefill": {
                "steps": prefill_steps,
                "tokens": prefill_tokens,
                "seconds": prefill_seconds,
                "tokens_per_second": (
                    prefill_tokens / prefill_seconds if prefill_seconds else 0.0
                ),
            },
            "decode": {
                "steps": decode_steps,
                "tokens": decode_tokens,
                "seconds": decode_seconds,
                "tokens_per_second": (
                    decode_tokens / decode_seconds if decode_seconds else 0.0
                ),
            },
            "completion_seconds": [
                completion_seconds[seq_id]
                for seq_id in sorted(completion_seconds)
            ],
        }
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
