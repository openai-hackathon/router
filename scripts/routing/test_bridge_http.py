import asyncio
import os
from pathlib import Path
import socket
import sys
import unittest
import httpx
from bridge import make_app


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BridgeHttpTest(unittest.IsolatedAsyncioTestCase):
    async def test_supervised_engine_identity_replay_auth_render_and_stream(self):
        http_port, event_port, replay_port = port(), port(), port()
        os.environ["BRIDGE_TEST_TOKEN"] = "test-token"
        config = {
            "worker_id": "fixture",
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
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://bridge"
            ) as client:
                self.assertEqual((await client.get("/routing/info")).status_code, 401)
                client.headers["x-routing-token"] = "test-token"
                for _ in range(60):
                    state = (await client.get("/routing/state")).json()
                    if state["engine_ready"] and state["synced"]:
                        break
                    await asyncio.sleep(0.1)
                self.assertTrue(state["engine_ready"])
                self.assertTrue(state["synced"])
                info = (await client.get("/routing/info")).json()
                self.assertEqual(
                    (await client.get("/routing/kv-snapshot")).json()["blocks"], []
                )
                body = {
                    "model": "local",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                }
                rendered = await client.post(
                    "/routing/render",
                    json={"route": "/v1/chat/completions", "body": body},
                )
                self.assertEqual(rendered.json()["token_ids"], [1, 2, 3, 4, 5])
                response = await client.post("/v1/chat/completions", json=body)
                self.assertEqual(
                    response.content, b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
                )
                self.assertEqual(
                    response.headers["x-routing-engine-epoch"], info["engine_epoch"]
                )
                for _ in range(30):
                    snapshot = (await client.get("/routing/kv-snapshot")).json()
                    if snapshot.get("blocks"):
                        break
                    await asyncio.sleep(0.1)
                self.assertEqual(len(snapshot["blocks"]), 2)
                delta = await client.get(
                    "/routing/kv-events",
                    params={"epoch": info["engine_epoch"], "after": -1},
                )
                self.assertEqual(delta.json()["batches"][0]["sequence"], 0)
                self.assertEqual(
                    (
                        await client.get(
                            "/routing/kv-events", params={"epoch": "wrong"}
                        )
                    ).status_code,
                    409,
                )


if __name__ == "__main__":
    unittest.main()
