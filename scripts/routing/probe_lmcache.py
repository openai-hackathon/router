#!/usr/bin/env python3
"""Read-only LMCache lookup on explicitly supplied HTTP base URLs.

Queries only POST /lookup on the controller, using the vLLM renderer for
synthetic tokens when available. Does not enumerate APIs, modify deployments,
or print cached prompt/token data. An empty result cannot prove cache absence.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def request(url, timeout, body=None, *, include_identity=False):
    started = time.monotonic()
    identity = {}
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, raw = response.status, response.read(1_048_576)
            if include_identity:
                identity = {
                    name: response.headers[name]
                    for name in ("x-routing-worker-id", "x-routing-engine-epoch")
                    if name in response.headers
                }
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
        **({"identity_headers": identity} if include_identity else {}),
    }


def summarize(result):
    data = result.pop("data", None)
    if result["status"] != 200:
        if isinstance(data, dict) and isinstance(data.get("detail"), str):
            result["detail"] = data["detail"][:160]
        return result
    if isinstance(data, dict):
        result["layout_info"] = data.get("layout_info")
    return result


def probe(url, timeout, model):
    url = url.rstrip("/")
    # Real tokenizer output is used when the supplied URL is a vLLM endpoint.
    rendered = request(
        url + "/v1/chat/completions/render",
        timeout,
        {
            "model": model,
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
    lookup = request(url + "/lookup", timeout, {"tokens": lookup_tokens})
    return {
        "url": url,
        "render_status": rendered["status"],
        "synthetic_token_count": len(lookup_tokens),
        "tokens_from_renderer": tokens is not None,
        "endpoints": {"/lookup": summarize(lookup)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--model", default="local", help="Served renderer model name")
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
