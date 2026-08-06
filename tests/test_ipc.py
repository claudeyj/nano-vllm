import unittest

from nanovllm.engine.llm_engine import _new_shm_name


class TestIPC(unittest.TestCase):

    def test_shared_memory_names_are_unique(self):
        first = _new_shm_name()
        second = _new_shm_name()

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("nanovllm_"))
        self.assertNotIn("/", first)


if __name__ == "__main__":
    unittest.main()
