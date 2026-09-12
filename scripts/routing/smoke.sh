#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
cargo build --bin vllm-router
cargo test --test routing_policy_snapshot_test
uv run --no-project \
  --with fastapi --with httpx --with msgspec --with pyzmq \
  --with prometheus-client --with uvicorn --with numpy --with scipy \
  python -m unittest discover -s scripts/routing -p 'test_*.py'
