"""CPU-only vLLM wire-protocol fixture used by test_bridge_http.py."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import msgspec
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

p = argparse.ArgumentParser()
p.add_argument("--port", type=int)
p.add_argument("--events")
p.add_argument("--replay")
a = p.parse_args()
context = zmq.asyncio.Context()
pub = context.socket(zmq.PUB)
pub.bind(a.events)
router = context.socket(zmq.ROUTER)
router.bind(a.replay)
buffer = []


async def replay():
    while True:
        identity, empty, start = await router.recv_multipart()
        for sequence, payload in buffer:
            if sequence >= int.from_bytes(start, "big"):
                await router.send_multipart(
                    [identity, b"", b"kv-events", sequence.to_bytes(8, "big"), payload]
                )
        await router.send_multipart(
            [identity, b"", b"", (2**64 - 1).to_bytes(8, "big"), b""]
        )


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(replay())
    try:
        yield
    finally:
        task.cancel()
        pub.close(0)
        router.close(0)
        context.term()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {}


@app.get("/version")
async def version():
    return {"version": "0.29.0"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "local", "root": "fixture"}]}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        'vllm:num_requests_running{engine="0",model_name="local"} 0\nvllm:num_requests_waiting{engine="0",model_name="local"} 0\nvllm:kv_cache_usage_perc{engine="0",model_name="local"} 0\n'
    )


@app.post("/v1/chat/completions/render")
async def render(request: Request):
    return {"token_ids": [1, 2, 3, 4, 5]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    sequence = len(buffer)
    payload = msgspec.msgpack.encode(
        [
            0.0,
            [
                {
                    "type": "BlockStored",
                    "block_hashes": [42, 43],
                    "parent_block_hash": None,
                    "token_ids": [1, 2, 3, 4],
                    "block_size": 2,
                    "medium": "GPU",
                    "group_idx": 0,
                }
            ],
            0,
        ]
    )
    buffer.append((sequence, payload))
    await pub.send_multipart([b"kv-events", sequence.to_bytes(8, "big"), payload])
    if body.get("stream"):

        async def chunks():
            yield b'data: {"choices":[]}\n\n'
            yield b"data: [DO"
            yield b"NE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")
    return {"id": "fixture", "request": body}


uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="error")
