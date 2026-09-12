#!/usr/bin/env python3
"""Generate synthetic ECT timing samples to exercise calibration tooling."""

import argparse
import itertools
import json
from pathlib import Path


TRUTH = {
    "g0": {
        "prefill": [10.0, 0.2, 0.00002],
        "decode": [3.0, 0.5, 0.0002],
        "beta": 0.15,
        "queue_ms": 10.0,
    },
    "g1": {
        "prefill": [8.0, 0.16, 0.000016],
        "decode": [2.4, 0.4, 0.00016],
        "beta": 0.25,
        "queue_ms": 5.0,
    },
    "g2": {
        "prefill": [12.0, 0.24, 0.000024],
        "decode": [3.6, 0.6, 0.00024],
        "beta": 0.1,
        "queue_ms": 15.0,
    },
}


def timings(row):
    model = TRUTH[row["worker_id"]]
    length, cached, output = (
        row["prompt_tokens"],
        row["reusable_tokens"],
        row["output_tokens"],
    )
    p, d = model["prefill"], model["decode"]
    prefill = (
        p[0] + p[1] * (length - cached) + p[2] * (length * length - cached * cached)
    )
    decode = d[0] + d[1] * output + d[2] * length * output
    return (
        prefill,
        decode,
        (prefill + decode) * (1 + model["beta"] * row["inflight"]) + model["queue_ms"],
    )


def samples():
    rows = []
    for worker_id in TRUTH:
        for split, lengths, outputs, loads in [
            ("train", [1024, 4096, 8192], [64, 128, 256], [0, 1, 3, 6]),
            ("validation", [2048, 6144], [128], [0, 2, 4]),
        ]:
            for index, (length, fraction, output, load) in enumerate(
                itertools.product(lengths, [0, 0.25, 0.5, 0.75], outputs, loads)
            ):
                row = {
                    "sample_id": f"{worker_id}-{split}-{index}",
                    "group_id": f"{split}-context-{length}-{fraction}",
                    "worker_id": worker_id,
                    "fingerprint": "synthetic-only",
                    "source": "synthetic",
                    "split": split,
                    "prompt_tokens": length,
                    "reusable_tokens": int(length * fraction),
                    "output_tokens": output,
                    "max_output_tokens": 256,
                    "inflight": load,
                }
                prefill, decode, completion = timings(row)
                row.update(
                    prefill_ms=prefill if load == 0 else None,
                    decode_ms=decode if load == 0 else None,
                    completion_ms=completion,
                )
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = samples()
    Path(args.output).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"source": "synthetic", "samples": len(rows)}))


if __name__ == "__main__":
    main()
