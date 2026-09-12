# vLLM Router

## Three KV routing policies added by this fork

All three policies run in the **Router** and share request tokenization, prefix evidence, the in-flight ledger, and the retry/streaming lifecycle. They differ in worker ranking. vLLM provides inference and telemetry without scheduler changes.

| CLI policy | Primary ordering | Secondary ordering | Purpose |
| --- | --- | --- | --- |
| `prefix_max` | Largest reusable prefix `H_j` | Fewest in-flight attempts `n_j` | Cache-first baseline |
| `least_load_kv` | Fewest in-flight attempts `n_j` | Largest reusable prefix `H_j` | Load-first baseline with cache tie-breaking |
| `kv_batch_ect` | Lowest estimated completion time `E_j` | Stable worker URL order | Cache-aware ECT using cache benefit, service capacity, and load |

The first two policies also break final ties by worker URL and do not apply session affinity. `n_j` counts this Router's unfinished attempts **before** dispatch. Selection and reservation are atomic; receiving HTTP 200 headers does not release a streaming reservation. Traffic sent directly to a backend is absent from this ledger. The new completion-time model reads backend running/waiting metrics separately to account for background traffic; it never adds those gauges to the ledger count.

The LMCache adapter reads each controller's `/lookup` result as a **restorable `LocalCPUBackend` prefix**. It preserves the raw matched length `C_j`, including partial blocks, and derives `H_j` by rounding down to complete vLLM blocks while leaving the final prompt token to be computed. CPU inventory does not establish GPU KV residency. The two baselines need no timing coefficients or global hit rate.

For new ECT deployments, use **`ect_model: "completion_time"`**. This mode predicts dispatch-to-terminal time directly, in milliseconds:

```text
E_j = intercept_ms
    + prompt_token_ms × L
    + output_token_ms × Ô
    + prompt_output_token_ms × L × Ô
    - cache_token_ms × C_j
    + router_inflight_ms × n_j
    + backend_running_ms × running_j
    + backend_waiting_ms × waiting_j
    + kv_usage_ms × kv_usage_fraction_j
```

`L` comes from actual renderer tokens; `Ô` is an output-length prior capped by the request's output limit. Cache credit represents the modeled net benefit of CPU cache evidence, including GPU overlap and restoration effects. KV usage is capacity pressure, not a request-specific hit estimate. Coefficients must be fitted jointly because the load features overlap. **This mode does not require or add `restore_models`, a separate queue correction, or a batch multiplier.** Omitting `ect_model` keeps the original `decomposed` model for compatibility.

Only `kv_batch_ect` applies bounded session affinity. With `x-session-id`, the last successful home must remain available, have an unexpired record, offer at least one more block of prefix than the best worker, and satisfy `E_home ≤ E_best + 0.015 × E_best + 10 ms`. Verified identity mode also requires the same engine epoch.

If any available candidate lacks required prefix evidence or a valid, applicable ECT model, **the whole decision falls back to least-load** and records the reason. Missing telemetry is never treated as a confirmed zero hit or an idle backend. See the [completion-time usage guide](docs/load_balancing/completion-time-routing.md) and the [three-worker configuration](examples/configs/completion_time_routing.json).

## Using the Router's LMCache adapter

The adapter is built into this fork and enabled with `--routing-state-config`. Before each dispatch attempt, it queries the configured controllers in parallel, performs render/lookup outside the selection lock, and routes from a shared snapshot. No separate adapter process is needed. Controllers must already be deployed; the Router does not install LMCache on workers.

Supported deployments use regular HTTP routing, one engine per endpoint (`--intra-node-data-parallel-size 1`), and text `/v1/chat/completions`, including text tool history and `chat_template_kwargs`. These policies do not support PD/IGW mode. The LMCache adapter explicitly falls back for Responses, Completions, multimodal, LoRA, and cache-salted requests. The completion-time model currently supports one output choice per request.

### 1. Create `routing.json`

For a complete three-worker ECT setup, start with [completion_time_routing.json](examples/configs/completion_time_routing.json). Its coefficients are labeled `synthetic` and demonstrate the schema; they are not measurements of any deployment. The smaller configuration below is sufficient for the two baselines.

Replace URLs, instance IDs, the model alias, and block sizes with deployment values. Each `workers` key is an **inference base URL** matching `--worker-urls`; `controller_url` can be the same or a separate URL. To add workers, extend the config and CLI lists. No Modal endpoint is hard-coded.

```json
{
  "lmcache": {
    "identity_mode": "endpoint",
    "renderer_base_url": "https://worker-one.example",
    "model": "local",
    "fingerprint": null,
    "workers": {
      "https://worker-one.example": {
        "controller_url": "https://worker-one.example",
        "instance_id": "worker-one",
        "block_size": 16,
        "fingerprint": null
      },
      "https://worker-two.example": {
        "controller_url": "https://worker-two.example",
        "instance_id": "worker-two",
        "block_size": 16,
        "fingerprint": null
      }
    }
  },
  "telemetry_timeout_ms": 10000,
  "render_timeout_ms": 10000,
  "max_evidence_age_ms": 20000
}
```

`identity_mode: "endpoint"` adopts the experiment's deployment assumption: identical model/tokenizer/serving settings and one fixed engine behind each endpoint. Missing fingerprints or epochs do not block routing. Restart the Router after replacing an engine behind an existing URL to clear old session state. Omitting this field selects the default `verified` mode, which requires verified fingerprints and engine identities.

`renderer_base_url` identifies one available vLLM renderer with matching settings; the adapter calls `POST /v1/chat/completions/render`. Each controller must accept `POST /lookup` with `{"tokens": [...]}` and `POST /health` with `{"instance_id": "worker-one"}`. Set `block_size` to the actual vLLM block size, separately from LMCache chunk size. Adjust the illustrated timeouts for your deployment.

### 2. Build and start this fork

```bash
cargo build --release --bin vllm-router

export WORKER_ONE="https://worker-one.example"
export WORKER_TWO="https://worker-two.example"
export POLICY="prefix_max"

./target/release/vllm-router \
  --host 127.0.0.1 --port 30000 \
  --worker-urls "$WORKER_ONE" "$WORKER_TWO" \
  --policy "$POLICY" \
  --routing-state-config routing.json \
  --intra-node-data-parallel-size 1 \
  --health-check-endpoint /v1/models \
  --worker-startup-check-interval 1
```

Change `POLICY` to `least_load_kv` or `kv_batch_ect` and restart with the same command to switch policies. Add every configured worker to `--worker-urls`, including the third worker when using the complete ECT example. Use the binary built from this fork; the PyPI release may not include these policies.

`--health-check-endpoint /v1/models` checks inference availability. The tested Modal controller endpoints reserve `/health` for POST and return 405 for GET, so the default GET `/health` is unsuitable. Configure telemetry authentication headers through environment variables in `header_env`; see the [adapter guide](docs/load_balancing/lmcache-adapter.md).

### 3. Send inference requests to the Router

```bash
curl --fail-with-body --max-time 180 \
  http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-session-id: demo-session' \
  -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false},"priority":0}'
```

The Router forwards the original JSON and inference stream. For OpenAI-compatible clients, set `base_url` to `http://127.0.0.1:30000/v1`. The illustrated `enable_thinking` option applies to the current Qwen deployment.

### 4. Enable Cache-aware ECT and inspect fallbacks

Changing the CLI policy alone does not provide usable ECT scores. For the new mode, add these top-level fields to the same JSON configuration:

| Field | Required configuration |
| --- | --- |
| `ect_model: "completion_time"` | Selects the direct completion-time model for all candidates |
| `completion_models[instance_id]` | Per-worker coefficients, provenance, version, output prior, and supported feature ranges |
| `backend_metrics` | Model name, inference URL → complete `/metrics` URL mapping, polling interval, timeout, and maximum sample age |

Every candidate needs its own LMCache binding, metrics binding, and completion model. Models can differ by worker. The collector runs in the background with a separate HTTP pool; dispatch only reads local snapshots. Running/waiting gauges must match the configured model and one engine. KV usage is required when its model coefficient is positive. Missing or expired observations trigger the common fallback.

Model `source` must be `measured`, `estimated`, or `synthetic`. Estimated coefficients can support an explicitly labeled experiment; the Router does not automatically turn `/metrics` into a calibrated model. The [completion-time guide](docs/load_balancing/completion-time-routing.md) explains per-attempt sample logs, offline calibration, supported ranges, and validation. The [design notes](docs/load_balancing/cache-aware-ect-design.md) explain why partial CPU prefixes, GPU/CPU overlap, and background benchmarks motivated the change. Phase-based and reuse-history enhancements remain future work.

Existing configs that omit `ect_model` retain the `decomposed` formula and its `cost_models` plus CPU-hit `restore_models` requirements. See [decomposed ECT calibration](docs/load_balancing/ect-calibration.md), [CPU restoration models](docs/load_balancing/lmcache-adapter.md#ect-restoration-model), and [observable inputs](docs/load_balancing/ect-observable-inputs.md). Endpoint identity mode skips fingerprint matching; version, numerical, and domain checks still apply.

```bash
curl --fail http://127.0.0.1:29000/metrics \
  | rg 'router_routing_decisions_total|router_lmcache_observations_total'
```

Inspect the `policy`, `identity_mode`, `ect_model`, and `fallback` labels on `router_routing_decisions_total`. `fallback="none"` means the configured ranking was used. Common reasons include `render_failed`, `unknown_kv`, `stale_kv`, `missing_completion_model`, and `missing_backend_metrics`; legacy mode can also report `missing_cost_model` or `missing_restore_model`. `router_lmcache_observations_total` records collection outcomes. A successful lookup alone does not mean every ECT input is available.

### Local smoke tests and further documentation

With Rust/Cargo and `uv` installed, test all three policies using local fixtures without GPUs or live controllers:

```bash
bash scripts/routing/smoke.sh

# Fixed synthetic case: select G0, G2, and G1 to distinguish the three rankings.
cargo run --example routing_policy_demo
```

The new three-worker completion-time smoke also changes background load, KV pressure, and lookup inventory to verify that each affects routing, then checks the shared fallback when one worker's metrics fail. See [Observed KV routing](docs/load_balancing/observed-kv.md) for the shared design and limitations, and [LMCache adapter](docs/load_balancing/lmcache-adapter.md) for data contracts and deployment details.
---

<p align="center">
| <a href="docs/load_balancing/README.md"><b>Documentation</b></a> | <a href="https://deepwiki.com/vllm-project/router"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://vllm-dev.slack.com/archives/C085AUU43NK"><b>Developer Slack</b></a> | <a href="docs/assets/WeChat.png"><b>WeChat</b></a> |
</p>

A high-performance and light-weight request forwarding system for vLLM large scale deployments, providing advanced load balancing methods and prefill/decode disaggregation support.

### Key Features

- **Core Architecture**: Request routing framework and async processing patterns
- **Load Balancing**: Multiple algorithms (cache-aware, power of two, consistent hashing, random, round robin)
- **Prefill-Decode Disaggregation**: Specialized routing for separated processing phases
- **Service Discovery**: Kubernetes-native worker management and health monitoring
- **Enterprise Features**: Circuit breakers, retry logic, metrics collection

## Quick Start

### Prerequisites

**Rust and Cargo:**
```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version

```

**Python with pip installed**

### Installation & Basic Usage

#### Rust Binary
```bash
# Build Rust components
cargo build --release
```

#### Python Package
Install from PyPI
```bash
pip install vllm-router                                                                                                                                                        ```

To build from source:
```bash    
pip install setuptools-rust wheel build
python -m build
pip install dist/*.whl

# Rebuild & reinstall in one step during development
python -m build && pip install --force-reinstall dist/*.whl
```

### Usage Examples

#### Standard Data Parallelism Routing
```bash
# Launch router with data parallelism (8 replicas per worker URL)
# When data-parallel-size > 1, the router automatically creates DP-aware workers
./target/release/vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8

# Alternative: using cargo run
cargo run --release -- \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8

# Alternative: using python launcher
vllm-router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8
```

#### Prefill-Decode Disaggregation
```bash
# When vLLM runs the NIXL connector, prefill/decode URLs are required.
# See a working example in scripts/llama3.1/ folder.
cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 \
    --prefill http://127.0.0.1:8082 \
    --decode http://127.0.0.1:8083 \
    --decode http://127.0.0.1:8084 \
    --decode http://127.0.0.1:8085 \
    --decode http://127.0.0.1:8086 \
    --host 127.0.0.1 \
    --port 8090 \
    --intra-node-data-parallel-size 1 \


# When vLLM runs the NCCL connector, ZMQ based discovery is supported.
# See a working example in scripts/install.sh
cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --vllm-discovery-address 0.0.0.0:30001 \
    --host 0.0.0.0 \
    --port 10001 \
    --prefill-policy consistent_hash \
    --decode-policy consistent_hash

# When vLLM runs the Mooncake connector, pass --kv-connector mooncake.
# The router queries each prefill node's Mooncake bootstrap server at startup
# to learn engine_id per DP rank, and injects transfer_id / remote_bootstrap_addr /
# remote_engine_id into each request's kv_transfer_params for P/D coordination.
cargo run --release -- \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --kv-connector mooncake \
    --prefill http://127.0.0.1:8081 \
    --prefill http://127.0.0.1:8082 \
    --decode http://127.0.0.1:8083 \
    --decode http://127.0.0.1:8084 \
    --host 127.0.0.1 \
    --port 8090 \
    --intra-node-data-parallel-size 1
```

## Configuration

### Authentication

Enable bearer-token validation by listing validation URLs (comma-separated) in `.env` via `API_KEY_VALIDATION_URLS` or passing `--api-key-validation-urls`.
When set, all HTTP endpoints require `Authorization: Bearer <token>` and tokens are validated with HTTP 200 responses.

```bash
# .env
API_KEY_VALIDATION_URLS=https://codebase.helmholtz.cloud/api/v4/user

# CLI override
vllm-router --api-key-validation-urls https://codebase.helmholtz.cloud/api/v4/user
```

### Metrics

Prometheus metrics endpoint available at `127.0.0.1:29000` by default.

```bash
# Custom metrics configuration
vllm-router \
    --worker-urls http://localhost:8080 http://localhost:8081 \
    --prometheus-host 0.0.0.0 \
    --prometheus-port 9000
```

### Retries and Circuit Breakers

#### Retry Configuration
Retries are enabled by default with exponential backoff and jitter:

```bash
vllm-router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --retry-max-retries 3 \
  --retry-initial-backoff-ms 100 \
  --retry-max-backoff-ms 10000 \
  --retry-backoff-multiplier 2.0 \
  --retry-jitter-factor 0.1
```

#### Circuit Breaker Configuration
Circuit breakers protect workers and provide automatic recovery:

```bash
vllm-router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --cb-failure-threshold 5 \
  --cb-success-threshold 2 \
  --cb-timeout-duration-secs 30 \
  --cb-window-duration-secs 60
```

**Circuit Breaker State Machine:**
- `Closed` → `Open` after N consecutive failures (failure-threshold)
- `Open` → `HalfOpen` after timeout (timeout-duration-secs)
- `HalfOpen` → `Closed` after M consecutive successes (success-threshold)

**Retry Policy:** Retries on HTTP status codes 408/429/500/502/503/504, with backoff/jitter between attempts.

### Request ID Tracking

Track requests across distributed systems with configurable headers:

```bash
# Use custom request ID headers
vllm-router \
    --worker-urls http://localhost:8080 \
    --request-id-headers x-trace-id x-request-id
```

**Default headers:** `x-request-id`, `x-correlation-id`, `x-trace-id`, `request-id`

### Load Balancing Policies

The router supports multiple load balancing policies:

| Policy | Description | Session Affinity | Use Case |
|--------|-------------|------------------|----------|
| `round_robin` | Sequential distribution across workers | No | General purpose, even distribution |
| `random` | Uniform random selection | No | Simple deployments |
| `consistent_hash` | Routes same session/user to same worker | Yes | Multi-turn chat, KV cache reuse |
| `power_of_two` | Picks least loaded of two random workers | No | Load-sensitive workloads |
| `cache_aware` | Optimizes for prefix cache hits | Yes | Repeated prompts, few-shot |

```bash
# Example: Using consistent_hash with HTTP header for session affinity
curl -X POST http://router:8000/v1/chat/completions \
  -H "X-Session-ID: my-session-123" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

For detailed configuration options, hash key priorities, and usage examples, see [Load Balancing Documentation](docs/load_balancing/README.md).

## Advanced Features

### Kubernetes Service Discovery

Automatic worker discovery and management in Kubernetes environments.

#### Basic Service Discovery

```bash
vllm-router \
    --service-discovery \
    --selector app=vllm-worker role=inference \
    --service-discovery-namespace default
```

### Command Line Arguments Reference

#### Service Discovery
- `--service-discovery`: Enable Kubernetes service discovery
- `--service-discovery-port`: Port for worker URLs (default: 8000)
- `--service-discovery-namespace`: Kubernetes namespace to watch
- `--selector`: Label selectors for regular mode (format: `key1=value1 key2=value2`)

## Development

### Troubleshooting

**VSCode Rust Analyzer Issues:**
Set `rust-analyzer.linkedProjects` to the absolute path of `Cargo.toml`:

```json
{
  "rust-analyzer.linkedProjects": ["/workspaces/vllm/vllm-router/Cargo.toml"]
}
```

### CI/CD Pipeline

The continuous integration pipeline includes comprehensive testing, benchmarking, and publishing:

#### Build & Test
1. **Build Wheels**: Uses `cibuildwheel` for manylinux x86_64 packages
2. **Build Source Distribution**: Creates source distribution for pip fallback
3. **Rust HTTP Server Benchmarking**: Performance testing of router overhead
4. **Basic Inference Testing**: End-to-end validation through the router
5. **PD Disaggregation Testing**: Benchmark and sanity checks for prefill-decode load balancing

#### Publishing
- **PyPI Publishing**: Wheels and source distributions published when version changes in `pyproject.toml`
- **Container Images**: Docker images published using `/docker/Dockerfile.router`

## Acknowledgement

This project is a fork of [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway), and we would like to explicitly acknowledge and thank the original authors for their work. At this stage, our fork includes only minimal changes to preserve the existing interface and ensure compatibility with vLLM. We anticipate further divergence as we pursue the roadmap we have in mind, which is the reason for creating the fork.
