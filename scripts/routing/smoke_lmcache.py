#!/usr/bin/env python3
"""Opt-in live LMCache smoke: warm/repeat synthetic prompts and compare counters.

Uses only the supplied inference URL's renderer, chat API and metrics, plus
POST /lookup on the explicit controller URL. Does not clear or move cache.
Reports token parity, cache tier and counter deltas without logging prompts.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
import uuid

from probe_lmcache import request


def metrics(url, timeout):
    result = request(url + "/metrics", timeout)
    raw = result.pop("data", "")
    values = {}
    if result["status"] == 200 and isinstance(raw, str):
        for line in raw.splitlines():
            match = re.match(r"^([\w:]+)(?:\{.*\})?\s+([-+\w.e]+)(?:\s|$)", line)
            if not match:
                continue
            name = match[1]
            if not any(
                term in name.lower()
                for term in (
                    "prefix_cache",
                    "lmcache",
                    "num_requests_running",
                    "num_requests_waiting",
                    "kv_cache_usage",
                )
            ):
                continue
            value = float(match[2])
            if math.isfinite(value):
                values[name] = values.get(name, 0.0) + value
    return {
        **result,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "values": values,
    }


def counter_delta(before, after, prefix):
    names = [f"vllm:{prefix}_cache_{kind}" for kind in ("queries", "hits")]
    deltas = []
    for name in names:
        previous = before["values"].get(name + "_total")
        current = after["values"].get(name + "_total")
        if previous is None or current is None:
            return {"status": "missing_counters", "token_hit_rate": None}
        if current < previous or before["values"].get(name + "_created") != after[
            "values"
        ].get(name + "_created"):
            return {"status": "counter_reset", "token_hit_rate": None}
        deltas.append(current - previous)
    queries, hits = deltas
    if hits > queries:
        return {"status": "inconsistent_counters", "token_hit_rate": None}
    return {
        "status": "observed",
        "queried_tokens": queries,
        "hit_tokens": hits,
        "token_hit_rate": hits / queries if queries else None,
    }


def lookup_layout(result, prompt_tokens):
    data = result.get("data")
    layout = data.get("layout_info") if isinstance(data, dict) else None
    if result["status"] != 200 or not isinstance(layout, dict):
        return None
    for instance, match in layout.items():
        if (
            not isinstance(instance, str)
            or not instance
            or not isinstance(match, list)
            or len(match) != 2
            or not isinstance(match[0], str)
            or not match[0]
            or type(match[1]) is not int
            or not 0 < match[1] <= prompt_tokens
        ):
            return None
    return layout


def smoke(url, controller, timeout, model):
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inference_url": url,
        "controller_url": controller,
        "metrics_before": metrics(url, timeout),
        "samples": [],
    }
    for scenario, lines in enumerate((50, 100, 150)):
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": uuid.uuid4().hex
                    + "\n"
                    + "Routing selects workers using prefix reuse and request load.\n"
                    * lines
                    + "Reply OK.",
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 8,
            "return_token_ids": True,
        }
        rendered = request(url + "/v1/chat/completions/render", timeout, body)
        data = rendered.get("data")
        tokens = data.get("token_ids") if isinstance(data, dict) else None
        if not tokens or not 512 <= len(tokens) <= 3500:
            report["error"] = "render_failed_or_prompt_out_of_range"
            return report
        for repeat in range(2):
            before = request(controller + "/lookup", timeout, {"tokens": tokens})
            generated = request(
                url + "/v1/chat/completions", timeout, body, include_identity=True
            )
            response = generated.pop("data", {})
            after = request(controller + "/lookup", timeout, {"tokens": tokens})
            sample = {
                "scenario": scenario,
                "repeat": repeat,
                "prompt_tokens": len(tokens),
                "inference": generated,
                "token_parity": isinstance(response, dict)
                and response.get("prompt_token_ids") == tokens,
                "usage": response.get("usage") if isinstance(response, dict) else None,
                "request_metrics": response.get("metrics")
                if isinstance(response, dict)
                else None,
                "lookup_before": before,
                "lookup_after": after,
            }
            report["samples"].append(sample)
            print(json.dumps(sample), flush=True)
            if generated["status"] != 200 or not sample["token_parity"]:
                report["error"] = "inference_failed_or_token_mismatch"
                return report
            if lookup_layout(before, len(tokens)) is None or not lookup_layout(
                after, len(tokens)
            ):
                report["error"] = "cache_lookup_missing_or_invalid"
                return report
    # Engine metrics may publish later than the inference response. Bound the
    # wait; do not manufacture zero counts if a scrape fails or resets.
    expected = sum(sample["prompt_tokens"] for sample in report["samples"])
    for poll in range(6):
        after = metrics(url, timeout)
        native = counter_delta(report["metrics_before"], after, "prefix")
        if native.get("queried_tokens", -1) >= expected or poll == 5:
            break
        time.sleep(2)
    report["metrics_after"] = after
    report["native_prefix_delta"] = native
    report["external_prefix_delta"] = counter_delta(
        report["metrics_before"], after, "external_prefix"
    )
    report["expected_prompt_tokens"] = expected
    report["metric_window_matches_prompt_tokens"] = (
        native.get("queried_tokens") == expected
    )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="One inference worker base URL")
    parser.add_argument("--controller-url", help="Defaults to --url")
    parser.add_argument("--model", default="local")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = smoke(
        args.url.rstrip("/"),
        (args.controller_url or args.url).rstrip("/"),
        args.timeout,
        args.model,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}))
    if report.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
