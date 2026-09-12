"""CPU integration: native events -> bridge -> Rust collector -> selection.

Build the Rust binary first with cargo build --bin vllm-router.
"""

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import httpx
import uvicorn

from bridge import make_app
from test_bridge_http import port


class RouterPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_router_prefers_cached_worker_and_reports_observed_evidence(
        self,
    ):
        root = Path(__file__).resolve().parents[2]
        binary = root / "target/debug/vllm-router"
        self.assertTrue(binary.exists(), "build vllm-router before this test")
        os.environ["BRIDGE_TEST_TOKEN"] = "test-token"
        headers = {"x-routing-token": "test-token"}
        servers, tasks, urls = [], [], []
        async with AsyncExitStack() as stack:
            try:
                for i in range(2):
                    http_port, event_port, replay_port, bridge_port = (
                        port(),
                        port(),
                        port(),
                        port(),
                    )
                    config = {
                        "worker_id": f"g{i}",
                        "auth_token_env": "BRIDGE_TEST_TOKEN",
                        "vllm_url": f"http://127.0.0.1:{http_port}",
                        "event_endpoint": f"tcp://127.0.0.1:{event_port}",
                        "replay_endpoint": f"tcp://127.0.0.1:{replay_port}",
                        "serving": {
                            "vllm_version": "0.29.0",
                            "dp_size": 1,
                            "cache_groups": 1,
                            "attention": "full",
                            "block_size": 2,
                            "model": "local",
                            "model_root": "fixture",
                        },
                    }
                    command = [
                        sys.executable,
                        str(Path(__file__).with_name("fake_vllm.py")),
                        "--port",
                        str(http_port),
                        "--events",
                        config["event_endpoint"],
                        "--replay",
                        config["replay_endpoint"],
                    ]
                    app = make_app(config, command)
                    await stack.enter_async_context(app.router.lifespan_context(app))
                    server = uvicorn.Server(
                        uvicorn.Config(
                            app,
                            host="127.0.0.1",
                            port=bridge_port,
                            lifespan="off",
                            log_level="error",
                        )
                    )
                    servers.append(server)
                    tasks.append(asyncio.create_task(server.serve()))
                    urls.append(f"http://127.0.0.1:{bridge_port}")
                async with httpx.AsyncClient(headers=headers, timeout=10) as client:
                    for url in urls:
                        for _ in range(100):
                            try:
                                status = (
                                    await client.get(url + "/routing/state")
                                ).json()
                                if status["engine_ready"] and status["synced"]:
                                    break
                            except httpx.TransportError:
                                pass
                            await asyncio.sleep(0.1)
                        else:
                            self.fail("bridge did not become ready")
                    body = {
                        "model": "local",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                    # Seed only the URL that would lose a deterministic tie.
                    warm = max(urls)
                    self.assertEqual(
                        (
                            await client.post(warm + "/v1/chat/completions", json=body)
                        ).status_code,
                        200,
                    )
                    for _ in range(30):
                        snapshot = (
                            await client.get(warm + "/routing/kv-snapshot")
                        ).json()
                        if len(snapshot.get("blocks", [])) == 2:
                            break
                        await asyncio.sleep(0.1)
                    self.assertEqual(len(snapshot["blocks"]), 2)
                    with tempfile.TemporaryDirectory() as directory:
                        config_path = Path(directory) / "routing.json"
                        config_path.write_text(
                            json.dumps(
                                {"header_env": {"x-routing-token": "BRIDGE_TEST_TOKEN"}}
                            )
                        )
                        router_port, metrics_port = port(), port()
                        with (Path(directory) / "router.log").open("w+") as log:
                            process = await asyncio.create_subprocess_exec(
                                str(binary),
                                "--worker-urls",
                                *urls,
                                "--policy",
                                "prefix_max",
                                "--port",
                                str(router_port),
                                "--prometheus-port",
                                str(metrics_port),
                                "--routing-state-config",
                                str(config_path),
                                "--worker-startup-check-interval",
                                "1",
                                "--worker-startup-timeout-secs",
                                "10",
                                stdout=log,
                                stderr=log,
                            )
                            try:
                                router = f"http://127.0.0.1:{router_port}"
                                for _ in range(100):
                                    try:
                                        if (
                                            await client.get(router + "/health")
                                        ).status_code == 200:
                                            break
                                    except httpx.TransportError:
                                        pass
                                    await asyncio.sleep(0.1)
                                else:
                                    self.fail("router did not become ready")
                                # Wait for one collector cycle before issuing inference.
                                await asyncio.sleep(0.8)
                                response = await client.post(
                                    router + "/v1/chat/completions", json=body
                                )
                                self.assertEqual(
                                    response.status_code, 200, response.text
                                )
                                self.assertEqual(response.json()["request"], body)
                                self.assertEqual(
                                    response.headers["x-routing-worker-id"],
                                    f"g{urls.index(warm)}",
                                )
                                log.seek(0)
                                events = log.read()
                                self.assertIn('"reusable_tokens":4', events)
                                metrics = (
                                    await client.get(
                                        f"http://127.0.0.1:{metrics_port}/metrics"
                                    )
                                ).text
                                self.assertIn('fallback="none"', metrics)
                            finally:
                                if process.returncode is None:
                                    process.terminate()
                                await asyncio.wait_for(process.wait(), 10)
            finally:
                for server in servers:
                    server.should_exit = True
                await asyncio.gather(*tasks)


if __name__ == "__main__":
    unittest.main()
