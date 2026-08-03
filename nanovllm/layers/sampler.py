import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy_mask = temperatures == 0
        greedy_tokens = logits.argmax(dim=-1)

        # Avoid division by zero for greedy rows. torch.where below guarantees
        # that their result comes from argmax, including deterministic
        # lowest-token-id tie breaking.
        sampling_temperatures = torch.where(
            greedy_mask, torch.ones_like(temperatures), temperatures
        )
        scaled_logits = logits / sampling_temperatures.unsqueeze(dim=1)
        probs = torch.softmax(scaled_logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
