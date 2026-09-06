from types import SimpleNamespace
import unittest

import torch

from nanovllm_vllm_a100_moe import VllmA100MoePlugin


def _expert(value: float):
    return SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=torch.full((4, 3), value)
        ),
        down_proj=SimpleNamespace(
            weight=torch.full((3, 2), value + 10)
        ),
    )


class TestVllmA100MoePlugin(unittest.TestCase):

    def test_stacks_only_locally_owned_experts(self):
        layer = SimpleNamespace(
            enable_expert_parallel=True,
            expert_start=2,
            expert_end=4,
            experts=[_expert(float(i)) for i in range(4)],
        )
        plugin = VllmA100MoePlugin()

        w1, w2 = plugin._stack_local_weights(layer)

        self.assertEqual(w1.shape, (2, 4, 3))
        self.assertEqual(w2.shape, (2, 3, 2))
        torch.testing.assert_close(w1[:, 0, 0], torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(w2[:, 0, 0], torch.tensor([12.0, 13.0]))
        self.assertIs(plugin._stack_local_weights(layer)[0], w1)

    def test_expert_map_translates_global_to_local_ids(self):
        layer = SimpleNamespace(
            enable_expert_parallel=True,
            num_experts=8,
            num_local_experts=2,
            expert_start=4,
            expert_end=6,
        )
        plugin = VllmA100MoePlugin()

        expert_map = plugin._expert_map(layer, torch.device("cpu"))

        self.assertEqual(
            expert_map.tolist(), [-1, -1, -1, -1, 0, 1, -1, -1]
        )

    def test_non_ep_does_not_create_expert_map(self):
        layer = SimpleNamespace(enable_expert_parallel=False)

        self.assertIsNone(
            VllmA100MoePlugin()._expert_map(layer, torch.device("cpu"))
        )


if __name__ == "__main__":
    unittest.main()
