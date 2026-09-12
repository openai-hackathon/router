"""Exercise completion-time routing against three local LMCache fixtures."""

import asyncio
import json
from pathlib import Path
import re
import tempfile
import unittest

from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

import test_policy_smoke as fixtures


class CompletionFixture(fixtures.FixtureControllerWorker):
    def __init__(self, index):
        super().__init__(index)
        self.identity = False
        self.running = self.waiting = 0
        self.kv_usage = 0.0
        self.metrics_fail = False
        self.metrics_calls = 0

        @self.app.get("/metrics")
        async def metrics():
            self.metrics_calls += 1
            if self.metrics_fail:
                return PlainTextResponse("synthetic scrape failure", status_code=503)
            labels = '{engine="0",model_name="local"}'
            return PlainTextResponse(
                f"vllm:num_requests_running{labels} {self.running}\n"
                f"vllm:num_requests_waiting{labels} {self.waiting}\n"
                f"vllm:kv_cache_usage_perc{labels} {self.kv_usage}\n"
            )

        # This fixture returns actual usage so the same request also reaches
        # the Router's dispatch-to-terminal calibration sample path.
        self.app.router.routes[:] = [
            route
            for route in self.app.router.routes
            if getattr(route, "path", None) != "/v1/chat/completions"
        ]

        @self.app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            headers = {"x-smoke-worker-id": self.worker_id}
            if body.get("smoke_retry"):
                self.retries += 1
                if self.retries == 1:
                    return JSONResponse(
                        {"error": "synthetic rejection"},
                        status_code=503,
                        headers=headers,
                    )
            if body.get("stream"):

                async def chunks():
                    yield b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
                    await self.release.wait()
                    yield b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
                    yield b'data: {"choices":[],"usage":{"completion_to'
                    yield b'kens":4,"prompt_tokens":5,"total_tokens":9}}\n\n'
                    yield b"data: [DO"
                    yield b"NE]\n\n"

                return StreamingResponse(
                    chunks(), media_type="text/event-stream", headers=headers
                )
            return JSONResponse(
                {
                    "id": "completion-smoke",
                    "worker": self.worker_id,
                    "request": body,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 4,
                        "total_tokens": 9,
                    },
                },
                headers=headers,
            )


class CompletionPolicySmokeTest(unittest.IsolatedAsyncioTestCase):
    # Reuse fixture lifecycle without inheriting the other smoke test methods.
    worker_factory = CompletionFixture
    asyncSetUp = fixtures.PolicySmokeTest.asyncSetUp
    asyncTearDown = fixtures.PolicySmokeTest.asyncTearDown
    wait_for = fixtures.PolicySmokeTest.wait_for

    async def test_cache_pressure_background_load_and_shared_fallback(self):
        binary = Path(__file__).resolve().parents[2] / "target/debug/vllm-router"
        self.assertTrue(binary.exists(), "run cargo build --bin vllm-router first")
        self.nodes[0].running = 4  # External traffic, absent from Router ledger.
        self.nodes[1].kv_usage = 0.2
        for url in self.urls:

            async def ready(url=url):
                return (await self.client.get(url + "/v1/models")).status_code == 200

            await self.wait_for(ready)

        config = {
            "ect_model": "completion_time",
            "backend_metrics": {
                "model": "local",
                "urls": {url: url + "/metrics" for url in self.urls},
                "poll_interval_ms": 50,
                "timeout_ms": 1000,
                "max_age_ms": 500,
            },
            "lmcache": {
                "identity_mode": "endpoint",
                "renderer_base_url": self.urls[0],
                "model": "local",
                "workers": {
                    url: {
                        "controller_url": url,
                        "instance_id": f"g{i}",
                        "block_size": 2,
                    }
                    for i, url in enumerate(self.urls)
                },
            },
            "completion_models": {
                f"g{i}": {
                    "fingerprint": "synthetic-fixture",
                    "calibration_version": "synthetic-completion-smoke-only",
                    "source": "synthetic",
                    "prompt_range": [1, 4096],
                    "output_range": [1, 16],
                    "concurrency_range": [0, 50],
                    "cache_fraction_range": [0.0, 1.0],
                    "backend_running_range": [0, 50],
                    "backend_waiting_range": [0, 50],
                    "kv_usage_range": [0.0, 1.0],
                    "output_prior": 4,
                    "coefficients": {
                        "intercept_ms": 100,
                        "prompt_token_ms": 10,
                        "output_token_ms": 10,
                        "prompt_output_token_ms": 0,
                        "cache_token_ms": 20,
                        "router_inflight_ms": 100,
                        "backend_running_ms": 100,
                        "backend_waiting_ms": 200,
                        "kv_usage_ms": 100,
                    },
                }
                for i in range(3)
            },
        }
        # Completion models consume net cache benefit; no CPU restoration
        # model or decomposed prefill/decode calibration is supplied.
        self.assertNotIn("restore_models", config)
        self.assertNotIn("cost_models", config)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            router_port, metrics_port = fixtures.port(), fixtures.port()
            base = f"http://127.0.0.1:{router_port}"
            metrics_url = f"http://127.0.0.1:{metrics_port}/metrics"
            log_path = Path(directory) / "router.log"
            with log_path.open("w+") as log:
                process = await asyncio.create_subprocess_exec(
                    str(binary),
                    "--worker-urls",
                    *self.urls,
                    "--policy",
                    "kv_batch_ect",
                    "--routing-state-config",
                    str(config_path),
                    "--port",
                    str(router_port),
                    "--prometheus-port",
                    str(metrics_port),
                    "--worker-startup-check-interval",
                    "1",
                    "--health-check-endpoint",
                    "/v1/models",
                    "--worker-startup-timeout-secs",
                    "10",
                    "--retry-max-retries",
                    "2",
                    stdout=log,
                    stderr=log,
                )
                try:

                    async def router_ready():
                        return (
                            await self.client.get(base + "/health")
                        ).status_code == 200 and all(
                            node.metrics_calls for node in self.nodes
                        )

                    await self.wait_for(router_ready)
                    body = {
                        "model": "local",
                        "messages": [{"role": "user", "content": "smoke"}],
                        "max_tokens": 4,
                    }

                    def completion_samples():
                        # Open a separate descriptor so reading logs does not
                        # reposition the Router's stdout/stderr file offset.
                        samples = []
                        for line in log_path.read_text().splitlines():
                            if "routing completion sample" not in line:
                                continue
                            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                            _, separator, payload = line.partition("sample=")
                            if separator:
                                try:
                                    sample, _ = json.JSONDecoder().raw_decode(payload)
                                except json.JSONDecodeError:
                                    continue  # A concurrently written last line.
                                samples.append(sample)
                        return samples

                    async def expect_sample_count(expected):
                        async def complete():
                            return len(completion_samples()) == expected

                        await self.wait_for(complete)

                    async def expect_selection(index, fallback="none"):
                        # A scrape can be in flight when a fixture changes.
                        # Wait for the observable decision instead of assuming
                        # a fixed delay or a fixed number of lookup calls.
                        async def selected():
                            before = (await self.client.get(metrics_url)).text
                            previous = fixtures.metric(
                                before,
                                "router_routing_decisions_total",
                                policy="kv_batch_ect",
                                ect_model="completion_time",
                                fallback=fallback,
                            )
                            response = await self.client.post(
                                base + "/v1/chat/completions", json=body
                            )
                            self.assertEqual(response.status_code, 200, response.text)
                            self.assertEqual(response.json()["request"], body)
                            self.assertEqual(
                                response.json()["usage"]["completion_tokens"], 4
                            )
                            after = (await self.client.get(metrics_url)).text
                            count = fixtures.metric(
                                after,
                                "router_routing_decisions_total",
                                policy="kv_batch_ect",
                                ect_model="completion_time",
                                fallback=fallback,
                            )
                            return (
                                response.json()["worker"] == f"g{index}"
                                and count > previous
                            )

                        await self.wait_for(selected)

                    # Raw CPU prefixes 5/3/0 yield unloaded costs 90/130/190.
                    # Background traffic makes g0 expensive despite most KV.
                    await expect_selection(1)  # Scores 490 / 150 / 190.
                    self.nodes[1].kv_usage = 0.9
                    await expect_selection(2)  # Scores 490 / 220 / 190.
                    self.nodes[0].running = 0
                    await expect_selection(0)  # Scores 90 / 220 / 190.
                    self.nodes[0].blocks.clear()
                    self.nodes[1].kv_usage = 0.2
                    await expect_selection(1)  # New lookup: 190 / 150 / 190.

                    # Keep a completion-time request streaming while a second
                    # request observes its reservation in the same predictor.
                    before = (await self.client.get(metrics_url)).text
                    sample_start = len(completion_samples())
                    self.nodes[1].release.clear()
                    stream_body = {
                        **body,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    async with self.client.stream(
                        "POST", base + "/v1/chat/completions", json=stream_body
                    ) as held:
                        self.assertEqual(held.status_code, 200)
                        self.assertEqual(held.headers["x-smoke-worker-id"], "g1")
                        chunks = held.aiter_bytes()
                        self.assertIn(b'"content":"done"', await chunks.__anext__())
                        during = (await self.client.get(metrics_url)).text
                        self.assertEqual(
                            fixtures.metric(during, "vllm_router_running_requests"), 1
                        )
                        self.assertEqual(
                            fixtures.metric(
                                during,
                                "vllm_router_running_requests",
                                worker=self.urls[1],
                            ),
                            1,
                        )
                        self.assertEqual(len(completion_samples()), sample_start)
                        self.assertEqual(
                            fixtures.metric(during, "router_dispatch_finished_total"),
                            fixtures.metric(before, "router_dispatch_finished_total"),
                        )
                        response = await self.client.post(
                            base + "/v1/chat/completions", json=body
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                        # Held g1 now costs 250 ms; g0/g2 each cost 190 ms.
                        other = min((0, 2), key=lambda i: self.urls[i])
                        self.assertEqual(response.json()["worker"], f"g{other}")
                        self.assertEqual(response.json()["request"], body)
                        during = (await self.client.get(metrics_url)).text
                        self.assertEqual(
                            fixtures.metric(during, "vllm_router_running_requests"), 1
                        )
                        self.nodes[1].release.set()
                        tail = b"".join([chunk async for chunk in chunks])
                        self.assertIn(b'"completion_tokens":4', tail)
                        self.assertIn(b"data: [DONE]\n\n", tail)
                    await expect_sample_count(sample_start + 2)
                    after = (await self.client.get(metrics_url)).text
                    self.assertEqual(
                        fixtures.metric(after, "vllm_router_running_requests"), 0
                    )
                    self.assertEqual(
                        fixtures.metric(after, "router_dispatch_finished_total")
                        - fixtures.metric(before, "router_dispatch_finished_total"),
                        2,
                    )
                    stream_samples = completion_samples()[sample_start:]
                    self.assertEqual(len({s["attempt_id"] for s in stream_samples}), 2)
                    stream_sample = next(
                        s for s in stream_samples if s["worker_id"] == "g1"
                    )
                    self.assertTrue(stream_sample["success"])
                    self.assertEqual(stream_sample["model_kind"], "completion_time")
                    self.assertEqual(stream_sample["output_tokens"], 4)
                    self.assertEqual(stream_sample["finish_reason"], "stop")
                    self.assertGreater(stream_sample["completion_ms"], 0)

                    # A 503 is one completed failed attempt, followed by one
                    # successful retry. Neither headers nor cleanup duplicates it.
                    before = after
                    sample_start = len(completion_samples())
                    retry_body = {**body, "smoke_retry": True}
                    response = await self.client.post(
                        base + "/v1/chat/completions", json=retry_body
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["worker"], "g1")
                    self.assertEqual(response.json()["request"], retry_body)
                    self.assertEqual(self.nodes[1].retries, 2)
                    await expect_sample_count(sample_start + 2)
                    retry_samples = completion_samples()[sample_start:]
                    self.assertEqual(len({s["attempt_id"] for s in retry_samples}), 2)
                    self.assertEqual(
                        [s["success"] for s in retry_samples], [False, True]
                    )
                    self.assertIsNone(retry_samples[0]["output_tokens"])
                    self.assertEqual(retry_samples[1]["output_tokens"], 4)
                    after = (await self.client.get(metrics_url)).text
                    for success in ("true", "false"):
                        self.assertEqual(
                            fixtures.metric(
                                after, "router_dispatch_finished_total", success=success
                            )
                            - fixtures.metric(
                                before,
                                "router_dispatch_finished_total",
                                success=success,
                            ),
                            1,
                        )
                    self.assertEqual(
                        fixtures.metric(after, "vllm_router_running_requests"), 0
                    )
                    self.assertEqual(
                        fixtures.metric(after, "router_dispatch_unknown_total"), 0
                    )

                    # The deterministic least-load tie winner remains healthy
                    # and selectable even when its own metrics fail.
                    fallback_worker = self.urls.index(min(self.urls))
                    self.nodes[fallback_worker].metrics_fail = True
                    self.assertEqual(
                        (
                            await self.client.get(
                                self.urls[fallback_worker] + "/v1/models"
                            )
                        ).status_code,
                        200,
                    )
                    await expect_selection(fallback_worker, "missing_backend_metrics")
                    text = (await self.client.get(metrics_url)).text
                    self.assertEqual(
                        fixtures.metric(text, "vllm_router_running_requests"), 0
                    )
                    self.assertEqual(
                        fixtures.metric(text, "router_dispatch_unknown_total"), 0
                    )
                    self.assertGreater(
                        fixtures.metric(text, "router_backend_metrics_errors_total"), 0
                    )
                    for node in self.nodes:
                        self.assertEqual(node.native_calls, 0)
                        self.assertGreaterEqual(node.lookup_calls, 5)
                        self.assertEqual(node.lookup_calls, node.health_calls)
                    log.seek(0)
                    output = log.read()
                    self.assertIn("routing completion sample", output)
                    sample_lines = [
                        line
                        for line in output.splitlines()
                        if "routing completion sample" in line
                    ]
                    self.assertTrue(
                        any('"output_tokens":4' in line for line in sample_lines),
                        "successful inference must emit actual completion usage",
                    )
                except BaseException:
                    log.seek(0)
                    print(log.read())
                    raise
                finally:
                    for node in self.nodes:
                        node.release.set()
                    if process.returncode is None:
                        process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except TimeoutError:
                        process.kill()
                        await process.wait()


if __name__ == "__main__":
    unittest.main()
