import unittest
from bridge import EventIndex, supported_request


class BridgeTests(unittest.TestCase):
    def stored(self, **kwargs):
        return {
            "type": "BlockStored",
            "block_hashes": [10, 20],
            "parent_block_hash": None,
            "block_size": 2,
            "token_ids": [1, 2, 3, 4],
            "medium": "GPU",
            "group_idx": 0,
            **kwargs,
        }

    def test_store_remove_clear_and_duplicate(self):
        index = EventIndex(2)
        index.apply(0, [self.stored()])
        self.assertEqual(index.blocks["int:20"]["parent"], "int:10")
        index.apply(0, [self.stored()])
        self.assertEqual(len(index.batches), 1)
        index.apply(
            1,
            [
                {
                    "type": "BlockRemoved",
                    "block_hashes": [10],
                    "medium": "GPU",
                    "group_idx": 0,
                }
            ],
        )
        self.assertEqual(len(index.blocks), 1)
        index.apply(2, [{"type": "AllBlocksCleared"}])
        self.assertEqual(index.blocks, {})

    def test_gap_fails_closed_then_contiguous_replay_recovers(self):
        index = EventIndex(2)
        index.apply(0, [self.stored()])
        with self.assertRaises(ValueError):
            index.apply(2, [])
        self.assertFalse(index.synced)
        index.apply(1, [])
        index.apply(2, [])
        self.assertTrue(index.synced)
        self.assertEqual([b["sequence"] for b in index.after(0)], [1, 2])

    def test_bounded_history_requires_snapshot(self):
        index = EventIndex(2, buffer_steps=2)
        for i in range(4):
            index.apply(i, [])
        with self.assertRaises(ValueError):
            index.after(0)
        self.assertEqual([b["sequence"] for b in index.after(1)], [2, 3])

    def test_untrusted_layout_does_not_look_like_zero_cache(self):
        for changes in [
            {"block_size": 4},
            {"group_idx": 1},
            {"kv_cache_spec_kind": "SlidingWindowSpec"},
        ]:
            index = EventIndex(2)
            with self.assertRaises(ValueError):
                index.apply(0, [self.stored(**changes)])
            self.assertFalse(index.synced)

    def test_extra_keys_and_cpu_blocks_do_not_match_plain_gpu_requests(self):
        for changes in [
            {"medium": "CPU"},
            {"lora_name": "adapter"},
            {"extra_keys": [["salt"], None]},
        ]:
            index = EventIndex(2)
            index.apply(0, [self.stored(**changes)])
            self.assertEqual(index.blocks, {})
            self.assertTrue(index.synced)

    def test_only_supported_requests_get_precise_features(self):
        body = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        self.assertTrue(supported_request("/v1/chat/completions", body, "local"))
        for patch in [
            {"n": 2},
            {"cache_salt": "tenant"},
            {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        ]:
            self.assertFalse(
                supported_request("/v1/chat/completions", {**body, **patch}, "local")
            )
        self.assertFalse(supported_request("/v1/responses", body, "local"))
        self.assertFalse(
            supported_request(
                "/v1/completions", {"model": "local", "prompt": ["a", "b"]}, "local"
            )
        )


if __name__ == "__main__":
    unittest.main()
