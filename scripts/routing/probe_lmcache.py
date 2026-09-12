#!/usr/bin/env python3
"""Read-only LMCache discovery on explicitly supplied HTTP base URLs.

Uses documented controller APIs, with synthetic tokens for lookups. Does not
probe other hosts/ports, modify deployments, or print cached prompt/token data.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(url, timeout, body=None):
    started = time.monotonic()
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, raw = response.status, response.read(1_048_576)
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read(4096)
    except (urllib.error.URLError, TimeoutError) as error:
        return {"status": "transport_error", "error_type": type(error).__name__}
    try:
        data = json.loads(raw)
    except ValueError:
        data = raw.decode(errors="replace")
    return {
        "status": status,
        "seconds": round(time.monotonic() - started, 3),
        "data": data,
    }


def summarize(path, result):
    data = result.pop("data", None)
    if result["status"] != 200:
        if isinstance(data, dict) and isinstance(data.get("detail"), str):
            result["detail"] = data["detail"][:160]
        return result
    if path == "/openapi.json" and isinstance(data, dict):
        result["paths"] = sorted(data.get("paths", {}))
    elif path == "/metrics" and isinstance(data, str):
        result["lmcache_metric_names"] = sorted(
            {
                line.split("{")[0].split(" ")[0]
                for line in data.splitlines()
                if not line.startswith("#") and "lmcache" in line.lower()
            }
        )
    elif path == "/lookup" and isinstance(data, dict):
        result["layout_info"] = data.get("layout_info")
    elif path == "/directory/lookup" and isinstance(data, dict):
        # Keep the wire schema for discovery, not token content or cache keys.
        result["response_fields"] = sorted(data)
    elif path == "/controller/key-stats" and isinstance(data, dict):
        result["counts"] = {
            name: data.get(name)
            for name in (
                "total_key_count",
                "total_instance_count",
                "total_worker_count",
            )
        }
    elif path in ("/controller/workers", "/instances"):
        result["registration"] = data
    elif isinstance(data, dict):
        result["response_fields"] = sorted(data)
    return result


def probe(url, timeout, model):
    url = url.rstrip("/")
    # Real tokenizer output is used when the supplied URL is a vLLM endpoint.
    rendered = request(
        url + "/v1/chat/completions/render",
        timeout,
        {
            "model": "local",
            "messages": [{"role": "user", "content": "LMCache routing probe."}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 1,
        },
    )
    tokens = (
        rendered.get("data", {}).get("token_ids")
        if isinstance(rendered.get("data"), dict)
        else None
    )
    # A controller-only URL can still be tested for API availability. One
    # synthetic token cannot establish that cache contents are absent.
    lookup_tokens = tokens if tokens is not None else [1]
    calls = [
        ("/openapi.json", None),
        ("/metrics", None),
        ("/controller/workers", None),
        ("/controller/key-stats", None),
        ("/lookup/info", None),
        ("/lookup", {"tokens": lookup_tokens}),
        ("/instances", None),
        (
            "/directory/lookup",
            {
                "token_ids": lookup_tokens,
                "model_name": model,
                "world_size": 1,
                "cache_salt": "",
            },
        ),
    ]
    endpoints = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {
            path: executor.submit(request, url + path, timeout, body)
            for path, body in calls
        }
        for path, future in pending.items():
            endpoints[path] = summarize(path, future.result())
    return {
        "url": url,
        "render_status": rendered["status"],
        "synthetic_token_count": len(lookup_tokens),
        "tokens_from_renderer": tokens is not None,
        "endpoints": endpoints,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as executor:
        deployments = list(
            executor.map(
                lambda url: probe(url, args.timeout, args.model),
                dict.fromkeys(args.url),
            )
        )
    report = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "deployments": deployments,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    print(output)


if __name__ == "__main__":
    main()
