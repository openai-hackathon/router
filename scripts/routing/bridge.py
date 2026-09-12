#!/usr/bin/env python3
"""Supervised vLLM 0.29.0 telemetry bridge; no replica selection or scheduler edits.

Run with the same Python environment as vLLM. See docs/load_balancing/observed-kv.md.
The public listener belongs to this process; vLLM and ZMQ stay on loopback.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager


class EventIndex:
    """A complete event-derived snapshot, never an invented inventory of GPU KV."""

    def __init__(self, block_size, max_blocks=1_000_000, buffer_steps=10000):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.batches = deque(maxlen=buffer_steps)
        self.blocks = {}
        self.sequence = None
        self.synced = False

    @staticmethod
    def key(value):
        if value is None:
            return None
        if isinstance(value, bytes):
            return "hex:" + value.hex()
        if isinstance(value, int):
            return "int:" + str(value)
        raise ValueError("unsupported block hash encoding")

    def apply(self, sequence, events):
        if self.sequence is not None and sequence <= self.sequence:
            return
        expected = 0 if self.sequence is None else self.sequence + 1
        if sequence != expected:
            self.synced = False
            raise ValueError("event gap")
        self.synced = False
        converted = []
        for event in events:
            kind = event["type"]
            if kind == "AllBlocksCleared":
                self.blocks.clear()
                converted.append({"type": kind})
                continue
            # V1 is restricted to local GPU group 0. Unrecognized layouts fail
            # closed instead of reporting a partial multi-group prefix as exact.
            if event.get("medium") != "GPU" or event.get("locality") not in (
                None,
                "LOCAL",
            ):
                continue
            if event.get("group_idx") not in (None, 0):
                raise ValueError("unsupported cache group")
            hashes = [self.key(h) for h in event["block_hashes"]]
            if kind == "BlockRemoved":
                for h in hashes:
                    self.blocks.pop(h, None)
                converted.append({"type": kind, "hashes": hashes})
            elif kind == "BlockStored":
                if (
                    event.get("kv_cache_spec_kind") not in (None, "FullAttentionSpec")
                    or event.get("kv_cache_spec_sliding_window") is not None
                ):
                    raise ValueError("unsupported attention layout")
                if (
                    event.get("lora_id") not in (None, 0)
                    or event.get("lora_name") is not None
                    or any(event.get("extra_keys") or [])
                ):
                    # These blocks cannot match the supported unsalted text
                    # requests; retain ordering but don't index their tokens.
                    continue
                size = event["block_size"]
                tokens = event["token_ids"]
                if size != self.block_size or len(tokens) != size * len(hashes):
                    raise ValueError("invalid block shape")
                parent = self.key(event["parent_block_hash"])
                blocks = []
                for i, h in enumerate(hashes):
                    block = {
                        "hash": h,
                        "parent": parent,
                        "tokens": tokens[i * size : (i + 1) * size],
                    }
                    if h in self.blocks and self.blocks[h] != block:
                        raise ValueError("conflicting block identity")
                    self.blocks[h] = block
                    blocks.append(block)
                    parent = h
                    if len(self.blocks) > self.max_blocks:
                        raise ValueError("index capacity")
                converted.append({"type": kind, "blocks": blocks})
            else:
                raise ValueError("unsupported event")
        self.sequence = sequence
        self.batches.append({"sequence": sequence, "events": converted})
        self.synced = True

    def after(self, after):
        if not self.synced:
            raise ValueError("unsynced")
        batches = [b for b in self.batches if b["sequence"] > after]
        if after != (self.sequence if self.sequence is not None else -1) and (
            not batches or batches[0]["sequence"] != after + 1
        ):
            raise ValueError("snapshot required")
        return batches


def supported_request(route, body, model):
    if (
        route not in ("/v1/chat/completions", "/v1/completions")
        or body.get("model") != model
    ):
        return False
    if any(
        body.get(key) is not None
        for key in (
            "cache_salt",
            "lora_path",
            "prompt_embeds",
            "multi_modal_data",
            "mm_processor_kwargs",
            "kv_transfer_params",
            "ec_transfer_params",
        )
    ):
        return False
    if (
        body.get("n", 1) != 1
        or body.get("best_of", 1) not in (None, 1)
        or body.get("skip_reading_prefix_cache", False)
    ):
        return False
    if route == "/v1/completions":
        prompt = body.get("prompt")
        return isinstance(prompt, str) or (
            isinstance(prompt, list) and all(type(t) is int and t >= 0 for t in prompt)
        )
    for message in body.get("messages", []):
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list) or any(
            p.get("type") != "text" for p in content
        ):
            return False
    return True


def make_app(config, command):
    import httpx
    import msgspec
    import zmq
    import zmq.asyncio
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    descriptor = config["serving"]
    if (
        descriptor["vllm_version"] != "0.29.0"
        or descriptor["dp_size"] != 1
        or descriptor["cache_groups"] != 1
        or descriptor["attention"] != "full"
    ):
        raise ValueError(
            "this adapter supports vLLM 0.29.0, one DP engine and one full-attention cache group"
        )
    fingerprint = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    info = {
        "schema_version": 1,
        "worker_id": config["worker_id"],
        "engine_epoch": str(uuid.uuid4()),
        "fingerprint": fingerprint,
        "model": descriptor["model"],
        "block_size": descriptor["block_size"],
        "supported": True,
    }
    index = EventIndex(
        descriptor["block_size"],
        config.get("max_blocks", 1_000_000),
        config.get("buffer_steps", 10000),
    )
    base = config.get("vllm_url", "http://127.0.0.1:8000").rstrip("/")
    if not base.startswith("http://127.0.0.1:"):
        raise ValueError("supervised vLLM must listen on loopback")
    runtime = {"ready": False, "metrics": None, "sampled_at": None}
    token = os.environ[config.get("auth_token_env", "ROUTING_BRIDGE_TOKEN")]
    if not token:
        raise ValueError("empty bridge authentication token")
    process = None
    telemetry_client = None
    inference_client = None
    zmq_context = zmq.asyncio.Context()

    def envelope():
        return {k: info[k] for k in ("schema_version", "worker_id", "engine_epoch")}

    async def replay():
        # DEALER with an empty delimiter can receive all ROUTER replies, unlike
        # a default REQ socket which accepts only one reply per request.
        socket = zmq_context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(config.get("replay_endpoint", "tcp://127.0.0.1:5558"))
        start = 0 if index.sequence is None else index.sequence + 1
        try:
            await socket.send_multipart([b"", start.to_bytes(8, "big")])
            while True:
                frames = await asyncio.wait_for(socket.recv_multipart(), 2)
                if len(frames) != 4 or frames[0] != b"":
                    raise ValueError("invalid replay framing")
                _, topic, sequence, payload = frames
                if not payload:
                    index.synced = True  # Empty replay is valid before any inference.
                    return
                if topic != config.get("topic", "kv-events").encode():
                    raise ValueError("unexpected topic")
                batch = msgspec.msgpack.decode(payload)
                index.apply(int.from_bytes(sequence, "big"), batch[1])
        finally:
            socket.close()

    async def events(socket):
        while True:
            try:
                if await socket.poll(500):
                    topic, seq_bytes, payload = await socket.recv_multipart()
                    if (
                        topic != config.get("topic", "kv-events").encode()
                        or len(seq_bytes) != 8
                    ):
                        raise ValueError("invalid event framing")
                    sequence = int.from_bytes(seq_bytes, "big")
                    expected = 0 if index.sequence is None else index.sequence + 1
                    if sequence > expected:
                        index.synced = False
                        await replay()
                    index.apply(sequence, msgspec.msgpack.decode(payload)[1])
                else:
                    # Replay also acts as a publisher liveness check and learns
                    # silently dropped final batches even without a later PUB event.
                    await replay()
            except (
                ValueError,
                KeyError,
                TypeError,
                asyncio.TimeoutError,
                zmq.ZMQError,
            ):
                index.synced = False
                # A missing history cannot be rebuilt by clearing our index.
                # Recovery requires a replay covering the gap or engine restart.
                await asyncio.sleep(1)

    async def sample():
        while True:
            try:
                if process.returncode is not None:
                    runtime["ready"] = False
                    index.synced = False
                    return
                health = await telemetry_client.get(base + "/health")
                version = (await telemetry_client.get(base + "/version")).json()
                models = (await telemetry_client.get(base + "/v1/models")).json()[
                    "data"
                ]
                runtime["ready"] = (
                    health.is_success
                    and version.get("version") == descriptor["vllm_version"]
                    and any(
                        m["id"] == descriptor["model"]
                        and m["root"] == descriptor["model_root"]
                        for m in models
                    )
                )
                response = await telemetry_client.get(base + "/metrics")
                response.raise_for_status()
                from prometheus_client.parser import text_string_to_metric_families

                values = {}
                engines = set()
                for family in text_string_to_metric_families(response.text):
                    for metric in family.samples:
                        if metric.name in (
                            "vllm:num_requests_running",
                            "vllm:num_requests_waiting",
                            "vllm:kv_cache_usage_perc",
                        ):
                            engines.add(metric.labels.get("engine"))
                            if metric.labels.get("model_name") == descriptor["model"]:
                                values[metric.name] = metric.value
                if engines != {"0"}:
                    raise ValueError("endpoint must correspond to exactly one engine")
                runtime["metrics"] = {
                    "running": int(values["vllm:num_requests_running"]),
                    "waiting": int(values["vllm:num_requests_waiting"]),
                    "kv_usage_fraction": values["vllm:kv_cache_usage_perc"],
                }
                runtime["sampled_at"] = time.monotonic()
            except (httpx.HTTPError, ValueError, KeyError):
                runtime["ready"] = False
            await asyncio.sleep(1)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal process, telemetry_client, inference_client
        socket = zmq_context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, config.get("topic", "kv-events").encode())
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(config.get("event_endpoint", "tcp://127.0.0.1:5557"))
        backend_headers = {}
        if config.get("vllm_api_key_env"):
            backend_headers["Authorization"] = (
                "Bearer " + os.environ[config["vllm_api_key_env"]]
            )
        async with httpx.AsyncClient(
            timeout=3, headers=backend_headers
        ) as tc, httpx.AsyncClient(
            timeout=httpx.Timeout(1800, connect=10), headers=backend_headers
        ) as ic:
            telemetry_client, inference_client = tc, ic
            process = await asyncio.create_subprocess_exec(*command)
            tasks = [asyncio.create_task(events(socket)), asyncio.create_task(sample())]
            try:
                yield
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                socket.close()
                zmq_context.term()
                if process.returncode is None:
                    process.terminate()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(process.wait(), 10)
                    if process.returncode is None:
                        process.kill()
                        await process.wait()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def authentication(request, call_next):
        import hmac

        if not hmac.compare_digest(request.headers.get("x-routing-token", ""), token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/routing/info")
    async def routing_info():
        return info

    @app.get("/routing/state")
    async def state():
        sample = runtime["metrics"]
        if sample is not None:
            sample = {
                **sample,
                "sample_age_ms": int((time.monotonic() - runtime["sampled_at"]) * 1000),
            }
        # sample age describes our scrape, not engine publication freshness.
        return {
            **envelope(),
            "engine_ready": runtime["ready"],
            "metrics": sample,
            "last_contiguous_sequence": index.sequence,
            "synced": index.synced,
        }

    @app.get("/routing/kv-snapshot")
    async def snapshot():
        if not index.synced:
            raise HTTPException(503, "KV history incomplete")
        return {
            **envelope(),
            "sequence": index.sequence,
            "synced": index.synced,
            "blocks": list(index.blocks.values()),
        }

    @app.get("/routing/kv-events")
    async def delta(epoch: str, after: int = -1):
        if epoch != info["engine_epoch"]:
            raise HTTPException(409, "engine epoch changed")
        try:
            batches = index.after(after)
        except ValueError:
            raise HTTPException(409, "snapshot required")
        return {
            **envelope(),
            "sequence": index.sequence,
            "synced": index.synced,
            "batches": batches,
        }

    @app.post("/routing/render")
    async def render(request: Request):
        data = await request.json()
        route, body = data.get("route"), data.get("body", {})
        if not runtime["ready"] or not supported_request(
            route, body, descriptor["model"]
        ):
            raise HTTPException(422, "unsupported request or engine")
        response = await telemetry_client.post(base + route + "/render", json=body)
        if not response.is_success:
            raise HTTPException(422, "backend renderer rejected request")
        rendered = response.json()
        if isinstance(rendered, list):
            if len(rendered) != 1:
                raise HTTPException(422, "batch prompt unsupported")
            rendered = rendered[0]
        if (
            rendered.get("features")
            or rendered.get("cache_salt")
            or not rendered.get("token_ids")
        ):
            raise HTTPException(422, "unsupported rendered features")
        return {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "token_ids": rendered["token_ids"],
        }

    @app.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]
    )
    async def proxy(path: str, request: Request):
        if not runtime["ready"] or not index.synced:
            raise HTTPException(503, "engine or KV telemetry not ready")
        if path.startswith("routing/"):
            raise HTTPException(404)
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower()
            not in (
                "host",
                "content-length",
                "connection",
                "x-routing-token",
                "modal-key",
                "modal-secret",
            )
        }
        req = inference_client.build_request(
            request.method,
            base + "/" + path,
            params=request.query_params,
            content=await request.body(),
            headers=headers,
        )
        response = await inference_client.send(req, stream=True)
        output_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower()
            not in (
                "content-length",
                "transfer-encoding",
                "connection",
                "content-encoding",
            )
        }
        output_headers.update(
            {
                "x-routing-worker-id": info["worker_id"],
                "x-routing-engine-epoch": info["engine_epoch"],
            }
        )

        async def chunks():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            chunks(), status_code=response.status_code, headers=output_headers
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a supervised vllm command is required")
    with open(args.config) as file:
        config = json.load(file)
    import uvicorn

    uvicorn.run(
        make_app(config, command), host="0.0.0.0", port=args.port, access_log=False
    )


if __name__ == "__main__":
    main()
