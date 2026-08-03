import unittest

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class SamplingParamsTest(unittest.TestCase):

    def test_zero_temperature_is_allowed(self):
        self.assertEqual(SamplingParams(temperature=0).temperature, 0)

    def test_negative_temperature_is_rejected(self):
        with self.assertRaises(AssertionError):
            SamplingParams(temperature=-0.1)


class SamplerTest(unittest.TestCase):

    def test_zero_temperature_uses_deterministic_argmax_for_ties(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 3.0, 3.0, 2.0]])
        temperatures = torch.tensor([0.0])

        outputs = [sampler(logits, temperatures).item() for _ in range(5)]

        self.assertEqual(outputs, [1] * 5)

    def test_mixed_batch_keeps_greedy_rows_deterministic(self):
        sampler = Sampler()
        logits = torch.tensor([
            [1.0, 3.0, 3.0, 2.0],
            [0.0, 1.0, 2.0, 3.0],
        ])
        temperatures = torch.tensor([0.0, 1.0])

        for _ in range(5):
            self.assertEqual(sampler(logits, temperatures)[0].item(), 1)


if __name__ == "__main__":
    unittest.main()
