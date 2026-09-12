"""Smoke all policies through the real Router with three local CPU fixtures.

Fixtures cover the native snapshot contract and the observed LMCache lookup /
health wire shapes. Identity headers on LMCache fixtures are a test extension;
current live deployments without those headers must fall back.
"""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from prometheus_client.parser import text_string_to_metric_families
import uvicorn

from test_bridge_http import port


class FixtureWorker:
    def __init__(self, index):
        self.worker_id = f"g{index}"
        self.epoch = f"smoke-epoch-{index}"
        self.synced = True
        self.release = asyncio.Event()
        self.retries = 0
        self.app = app = FastAPI()
        self.blocks = [
            {
                "hash": f"b{i}",
                "parent": f"b{i - 1}" if i else None,
                "tokens": [2 * i + 1, 2 * i + 2],
            }
            for i in range(2 - index)
        ]

        def envelope():
            return {
                "schema_version": 1,
                "worker_id": self.worker_id,
                "engine_epoch": self.epoch,
            }

        @app.get("/health")
        async def health():
            return {}

        @app.get("/routing/info")
        async def info():
            return {
                **envelope(),
                "fingerprint": "smoke-fixture",
                "model": "local",
                "block_size": 2,
                "supported": True,
            }

        @app.get("/routing/state")
        async def state():
            return {
                **envelope(),
                "engine_ready": True,
                "synced": self.synced,
                "last_contiguous_sequence": 0,
                "metrics": {
                    "running": 0,
                    "waiting": 0,
                    "kv_usage_fraction": 0,
                    "sample_age_ms": 0,
                },
            }

        @app.get("/routing/kv-snapshot")
        async def snapshot():
            return {
                **envelope(),
                "sequence": 0,
                "synced": self.synced,
                "blocks": self.blocks,
            }

        @app.get("/routing/kv-events")
        async def events():
            return {**envelope(), "sequence": 0, "synced": self.synced, "batches": []}

        @app.post("/routing/render")
        async def render():
            return {
                "schema_version": 1,
                "fingerprint": "smoke-fixture",
                "token_ids": [1, 2, 3, 4, 5],
            }

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            headers = {
                "x-routing-worker-id": self.worker_id,
                "x-routing-engine-epoch": self.epoch,
            }
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
                    yield b'data: {"choices":[]}\n\n'
                    await self.release.wait()
                    yield b"data: [DO"
                    yield b"NE]\n\n"

                return StreamingResponse(
                    chunks(), media_type="text/event-stream", headers=headers
                )
            return JSONResponse(
                {"id": "smoke", "worker": self.worker_id, "request": body},
                headers=headers,
            )


def metric(text, name, **labels):
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(key) == value for key, value in labels.items())
    )


class PolicySmokeTest(unittest.IsolatedAsyncioTestCase):
    worker_factory = FixtureWorker

    async def asyncSetUp(self):
        self.nodes = [self.worker_factory(i) for i in range(3)]
        self.servers, self.tasks, self.urls = [], [], []
        for node in self.nodes:
            http_port = port()
            server = uvicorn.Server(
                uvicorn.Config(
                    node.app,
                    host="127.0.0.1",
                    port=http_port,
                    log_level="error",
                    lifespan="off",
                )
            )
            self.servers.append(server)
            self.tasks.append(asyncio.create_task(server.serve()))
            self.urls.append(f"http://127.0.0.1:{http_port}")
        self.client = httpx.AsyncClient(timeout=15)

    async def asyncTearDown(self):
        for node in self.nodes:
            node.release.set()
        await self.client.aclose()
        for server in self.servers:
            server.should_exit = True
        await asyncio.gather(*self.tasks)

    async def wait_for(self, check):
        for _ in range(100):
            try:
                if await check():
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.05)
        self.fail("smoke condition did not become true within 5 seconds")

    async def exercise(self, policy, first_worker, while_held, source="native"):
        binary = Path(__file__).resolve().parents[2] / "target/debug/vllm-router"
        self.assertTrue(binary.exists(), "run cargo build --bin vllm-router first")
        for url in self.urls:

            async def healthy(url=url):
                path = "/v1/models" if source == "lmcache" else "/health"
                return (await self.client.get(url + path)).status_code == 200

            await self.wait_for(healthy)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config = {
                "poll_interval_ms": 50,
                "cost_models": {
                    f"g{i}": {
                        "fingerprint": "smoke-fixture",
                        "calibration_version": "synthetic-smoke-only",
                        "prompt_range": [1, 4096],
                        "output_range": [1, 16],
                        "concurrency_range": [0, 50],
                        "output_prior": 4,
                        "prefill": [0, 1, 0],
                        "decode": [score, 0, 0],
                        "beta": 1,
                        "queue_ms": 0,
                    }
                    for i, score in enumerate([100, 80, 120])
                },
            }
            if source == "lmcache":
                config["lmcache"] = {
                    "renderer_base_url": self.urls[0],
                    "model": "local",
                    "fingerprint": "smoke-fixture",
                    "workers": {
                        url: {
                            "controller_url": url,
                            "instance_id": f"g{i}",
                            "block_size": 2,
                            "fingerprint": "smoke-fixture",
                        }
                        for i, url in enumerate(self.urls)
                    },
                }
                config["restore_models"] = {
                    f"g{i}": {
                        "fingerprint": "smoke-fixture",
                        "calibration_version": "synthetic-restore-test-only",
                        "location": "LocalCPUBackend",
                        "token_range": [1, 4096],
                        "fixed_ms": 3,
                        "per_token_ms": 0,
                    }
                    for i in range(3)
                }
            config_path.write_text(json.dumps(config))
            router_port, metrics_port = port(), port()
            base = f"http://127.0.0.1:{router_port}"
            metrics_url = f"http://127.0.0.1:{metrics_port}/metrics"
            with (Path(directory) / "router.log").open("w+") as log:
                process = await asyncio.create_subprocess_exec(
                    str(binary),
                    "--worker-urls",
                    *self.urls,
                    "--policy",
                    policy,
                    "--routing-state-config",
                    str(config_path),
                    "--port",
                    str(router_port),
                    "--prometheus-port",
                    str(metrics_port),
                    "--worker-startup-check-interval",
                    "1",
                    "--health-check-endpoint",
                    "/v1/models" if source == "lmcache" else "/health",
                    "--worker-startup-timeout-secs",
                    "10",
                    "--retry-max-retries",
                    "2",
                    stdout=log,
                    stderr=log,
                )
                try:

                    async def ready():
                        if (await self.client.get(base + "/health")).status_code != 200:
                            return False
                        if source == "lmcache":
                            return True
                        text = (await self.client.get(metrics_url)).text
                        samples = [
                            s
                            for f in text_string_to_metric_families(text)
                            for s in f.samples
                            if s.name == "router_backend_running"
                        ]
                        return len(samples) == 3

                    await self.wait_for(ready)
                    body = {
                        "model": "local",
                        "messages": [{"role": "user", "content": "smoke"}],
                        "max_tokens": 4,
                    }
                    async with self.client.stream(
                        "POST",
                        base + "/v1/chat/completions",
                        json={**body, "stream": True},
                    ) as held:
                        self.assertEqual(held.status_code, 200)
                        self.assertEqual(
                            held.headers["x-routing-worker-id"], f"g{first_worker}"
                        )
                        # Headers and the first chunk must not release the reservation.
                        chunks = held.aiter_bytes()
                        self.assertIn(b"data:", await chunks.__anext__())
                        text = (await self.client.get(metrics_url)).text
                        self.assertEqual(
                            metric(
                                text,
                                "vllm_router_running_requests",
                                worker=self.urls[first_worker],
                            ),
                            1,
                        )
                        response = await self.client.post(
                            base + "/v1/chat/completions", json=body
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(response.json()["worker"], f"g{while_held}")
                        self.assertEqual(response.json()["request"], body)
                        self.nodes[first_worker].release.set()
                        self.assertIn(
                            b"[DONE]", b"".join([chunk async for chunk in chunks])
                        )
                    response = await self.client.post(
                        base + "/v1/chat/completions",
                        json={**body, "smoke_retry": True},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(self.nodes[first_worker].retries, 2)
                    text = (await self.client.get(metrics_url)).text
                    self.assertEqual(metric(text, "vllm_router_running_requests"), 0)
                    self.assertEqual(metric(text, "router_dispatch_unknown_total"), 0)
                    self.assertGreater(
                        metric(
                            text,
                            "router_routing_decisions_total",
                            policy=policy,
                            fallback="none",
                        ),
                        0,
                    )
                    # Keep inference healthy while one KV stream loses trust.
                    previous_errors = metric(text, "router_telemetry_errors_total")
                    self.nodes[2].synced = False

                    async def invalidated():
                        text = (await self.client.get(metrics_url)).text
                        return (
                            metric(text, "router_telemetry_errors_total")
                            > previous_errors
                        )

                    if source == "native":
                        await self.wait_for(invalidated)
                    response = await self.client.post(
                        base + "/v1/chat/completions", json=body
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    expected = self.urls.index(min(self.urls))
                    self.assertEqual(response.json()["worker"], f"g{expected}")
                    text = (await self.client.get(metrics_url)).text
                    self.assertGreater(
                        metric(
                            text,
                            "router_routing_decisions_total",
                            policy=policy,
                            fallback="stale_kv" if source == "native" else "unknown_kv",
                        ),
                        0,
                    )
                    self.assertEqual(metric(text, "vllm_router_running_requests"), 0)
                    if source == "lmcache":
                        for node in self.nodes:
                            self.assertEqual(node.native_calls, 0)
                            self.assertEqual(node.lookup_calls, 5)
                            self.assertEqual(node.health_calls, 5)
                except BaseException:
                    log.seek(0)
                    print(log.read())
                    raise
                finally:
                    for node in self.nodes:
                        node.release.set()
                    if process.returncode is None:
                        process.terminate()
                    await asyncio.wait_for(process.wait(), 10)

    async def test_prefix_max(self):
        await self.exercise("prefix_max", 0, 0)

    async def test_least_load_kv(self):
        await self.exercise("least_load_kv", 0, 1)

    async def test_kv_batch_ect(self):
        await self.exercise("kv_batch_ect", 1, 0)


class FixtureControllerWorker(FixtureWorker):
    def __init__(self, index):
        super().__init__(index)
        self.native_calls = self.lookup_calls = self.health_calls = 0
        self.identity = True
        app = self.app

        @app.middleware("http")
        async def count_native(request, call_next):
            if request.url.path == "/health" and request.method == "GET":
                return JSONResponse({"detail": "Method Not Allowed"}, status_code=405)
            if request.url.path.startswith("/routing/"):
                self.native_calls += 1
            return await call_next(request)

        @app.get("/v1/models")
        async def models():
            return {"data": [{"id": "local"}]}

        def headers():
            return (
                {
                    "x-routing-worker-id": self.worker_id,
                    "x-routing-engine-epoch": self.epoch,
                }
                if self.identity
                else {}
            )

        @app.post("/v1/chat/completions/render")
        async def render():
            return {"token_ids": [1, 2, 3, 4, 5]}

        @app.post("/lookup")
        async def lookup(request: Request):
            self.lookup_calls += 1
            assert (await request.json())["tokens"] == [1, 2, 3, 4, 5]
            layout = (
                {self.worker_id: ["LocalCPUBackend", len(self.blocks) * 2]}
                if self.blocks
                else {}
            )
            return JSONResponse(
                {"event_id": "lookup-op", "layout_info": layout}, headers=headers()
            )

        @app.post("/health")
        async def health(request: Request):
            self.health_calls += 1
            assert (await request.json())["instance_id"] == self.worker_id
            return JSONResponse(
                {
                    "event_id": "health-op",
                    "error_codes": {"0": 0 if self.synced else 1},
                },
                headers=headers(),
            )


class LmCachePolicySmokeTest(PolicySmokeTest):
    worker_factory = FixtureControllerWorker

    async def exercise(self, policy, first_worker, while_held, source="lmcache"):
        await super().exercise(policy, first_worker, while_held, source)


if __name__ == "__main__":
    unittest.main()
