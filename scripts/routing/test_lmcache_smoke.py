import unittest

from smoke_lmcache import counter_delta, lookup_layout


class CacheObservationTest(unittest.TestCase):
    @staticmethod
    def counters(queries, hits, created=10):
        return {
            "values": {
                "vllm:prefix_cache_queries_total": queries,
                "vllm:prefix_cache_hits_total": hits,
                "vllm:prefix_cache_queries_created": created,
                "vllm:prefix_cache_hits_created": created,
            }
        }

    def test_hit_rate_uses_window_deltas_not_lifetime_totals(self):
        delta = counter_delta(
            self.counters(4118, 2016), self.counters(10380, 5136), "prefix"
        )
        self.assertEqual(delta["queried_tokens"], 6262)
        self.assertEqual(delta["hit_tokens"], 3120)
        self.assertAlmostEqual(delta["token_hit_rate"], 3120 / 6262)
        idle = counter_delta(self.counters(1, 0), self.counters(1, 0), "prefix")
        self.assertIsNone(idle["token_hit_rate"])

    def test_missing_reset_and_inconsistent_counters_are_unknown(self):
        before = self.counters(100, 50)
        for after, status in (
            ({"values": {}}, "missing_counters"),
            (self.counters(10, 5), "counter_reset"),
            (self.counters(200, 100, created=11), "counter_reset"),
            (self.counters(101, 100), "inconsistent_counters"),
        ):
            with self.subTest(status=status):
                delta = counter_delta(before, after, "prefix")
                self.assertEqual(delta["status"], status)
                self.assertIsNone(delta["token_hit_rate"])

    def test_lookup_preserves_tier_and_distinguishes_empty_from_invalid(self):
        def result(layout, status=200):
            return {"status": status, "data": {"layout_info": layout}}

        layout = {"vllm-c": ["LocalCPUBackend", 1280]}
        self.assertEqual(lookup_layout(result(layout), 1347), layout)
        self.assertEqual(lookup_layout(result({}), 1347), {})
        self.assertIsNone(lookup_layout(result({}, status=404), 1347))
        for match in (None, ["CPU", 1348], ["CPU", True], ["CPU", 0], ["", 1280]):
            with self.subTest(match=match):
                self.assertIsNone(lookup_layout(result({"vllm-c": match}), 1347))


if __name__ == "__main__":
    unittest.main()
