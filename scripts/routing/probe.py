#!/usr/bin/env python3
"""Inspect explicit deployment URLs and verify synthetic chat token fixtures.

No deployment mutation. --token-parity makes two small inference calls per URL.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def fetch(url, body=None):
    start = time.monotonic()
    request = urllib.request.Request(
        url,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
            try:
                data = json.loads(raw)
            except ValueError:
                data = raw.decode()
            return {
                "status": response.status,
                "seconds": round(time.monotonic() - start, 3),
                "data": data,
            }
    except urllib.error.HTTPError as error:
        return {"status": error.code, "seconds": round(time.monotonic() - start, 3)}
    except (urllib.error.URLError, TimeoutError):
        return {
            "status": "transport_error",
            "seconds": round(time.monotonic() - start, 3),
        }


def probe(url, token_parity):
    url = url.rstrip("/")
    result = {"url": url, "endpoints": {}}
    for path in [
        "/version",
        "/v1/models",
        "/load",
        "/metrics",
        "/server_info",
        "/routing/info",
    ]:
        response = fetch(url + path)
        if path == "/metrics" and isinstance(response.get("data"), str):
            response["data"] = [
                line
                for line in response["data"].splitlines()
                if line.startswith(
                    (
                        "vllm:num_requests_running{",
                        "vllm:num_requests_waiting{",
                        "vllm:kv_cache_usage_perc{",
                        "vllm:prefix_cache_hits_total{",
                    )
                )
            ]
        result["endpoints"][path] = response
    if token_parity:
        fixture = json.loads(
            (
                Path(__file__).parents[2]
                / "tests/fixtures/qwen3_0_6b_vllm_029_tokens.json"
            ).read_text()
        )
        result["token_parity"] = []
        for case in fixture["cases"]:
            body = case["request"]
            rendered = fetch(url + "/v1/chat/completions/render", body)
            actual = fetch(url + "/v1/chat/completions", body)
            tokens = actual.get("data", {}).get("prompt_token_ids")
            result["token_parity"].append(
                {
                    "case": case["name"],
                    "render_status": rendered["status"],
                    "inference_status": actual["status"],
                    "inference_seconds": actual["seconds"],
                    "matches_renderer": tokens is not None
                    and tokens == rendered.get("data", {}).get("token_ids"),
                    "matches_golden": tokens == case["token_ids"],
                    "prompt_tokens": len(tokens) if tokens else None,
                    "usage": actual.get("data", {}).get("usage"),
                    "metrics": actual.get("data", {}).get("metrics"),
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--token-parity", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    urls = list(dict.fromkeys(args.url))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda url: probe(url, args.token_parity), urls))
    output = json.dumps(results, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    print(output)


if __name__ == "__main__":
    main()
