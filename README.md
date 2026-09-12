# vLLM Router

## 本 fork 新增的三個 KV routing policies

三個 policy 都實作在 **Router**，共用 request tokenization、prefix 證據、in-flight ledger 與 retry／streaming lifecycle；只替換 worker 的排序規則。vLLM 提供推論與 telemetry，無須修改 scheduler。

| CLI policy | 第一排序條件 | 第二排序條件 | 用途 |
| --- | --- | --- | --- |
| `prefix_max` | 可重用 prefix `H_j` 最大 | in-flight `n_j` 最少 | 優先利用已有 cache 的 baseline |
| `least_load_kv` | in-flight `n_j` 最少 | 可重用 prefix `H_j` 最大 | 優先平衡負載，平手時利用 cache |
| `kv_batch_ect` | 預估完成時間 `E_j` 最小 | 固定 worker URL 順序 | 同時考慮 prefix、服務能力、負載與 cache 還原成本 |

前兩個 policy 最後也以 worker URL 的字典順序處理平手，不套用 session affinity。`n_j` 是派送這筆請求**之前**，Router 記錄的未完成 attempt 數；選擇與 reservation 原子化執行，收到 HTTP 200 headers 不會提早釋放 streaming 負載。實驗流量應全部經過同一個 Router，backend running／waiting metrics 用於核對，不與 ledger 相加。

目前 LMCache adapter 使用各 controller `/lookup` 回報的 **`LocalCPUBackend` 可還原 prefix** 作為 `H_j`，並限制為完整 vLLM blocks、保留最後 prompt token 的計算。這個數字表示 CPU cache inventory，不代表 GPU 已駐留 KV。兩個 baseline 不需要時間係數或 hit rate，就可以依這份 prefix 證據排序。

KV-Batch-ECT 的 CPU cache 成本模型如下，時間單位皆為毫秒：

```text
P_j = a0_j + a1_j × (L − H_j) + a2_j × (L² − H_j²)
D_j = d0_j + d1_j × Ô + d2_j × L × Ô
R_j = fixed_ms_j + per_token_ms_j × cached_tokens_j    # 有 CPU cache 時
E_j = (P_j + D_j + R_j) × (1 + β_j × n_j) + Q_j
```

`L` 來自實際 renderer tokens；`Ô` 是輸出長度 prior，受 request 的輸出上限限制；`cached_tokens_j` 是 lookup 原始回報的 CPU prefix 長度。沒有 CPU 命中時 `R_j = 0`。目前模型保守計入整段 CPU prefix 的還原，尚未扣掉未知的 GPU 重疊部分。

只有 `kv_batch_ect` 會套用有界 session affinity：提供 `x-session-id` 後，最近成功使用的 home 必須仍可用、紀錄未過期、比最佳 worker 多至少一個 block 的 prefix，且 `E_home ≤ E_best + 0.015 × E_best + 10 ms`，才保留 home。`verified` 模式另要求 epoch 相同。

任一可用候選的必要 prefix 資料缺失／過期，或 ECT 的成本模型缺失、無效、超出適用範圍時，**整次決策共同降級成 least-load**，並記錄原因。缺 telemetry 不會被當成已確認的零命中。

## Router 的 LMCache adapter 怎麼用

Adapter 已內建於此 fork 的 Router，透過 `--routing-state-config` 啟用。它在每次 dispatch attempt 前並行查各 worker 的 controller，在鎖外完成 render／lookup，再以共用 snapshot 選路；不需要另外啟動 adapter 程序。每個 worker 的 controller 必須已部署，Router 不會替 backend 安裝 LMCache。

目前支援 regular HTTP routing、每個 endpoint 一個 engine（`--intra-node-data-parallel-size 1`），以及文字 `/v1/chat/completions`，包含文字 tool history 與 `chat_template_kwargs`。這三個 policy 尚不支援 PD／IGW 模式；LMCache adapter 對 Responses、Completions、多模態、LoRA 與 cache-salted requests 會明確降級。

### 1. 建立 `routing.json`

將以下 URL、instance ID、model alias 與 block size 換成部署值。`workers` 的 key 是 **推論 base URL**，必須對應 CLI 的 `--worker-urls`；`controller_url` 可與推論 URL 相同或不同。增加第三台時，在 `workers` 和 CLI 各加一筆，程式沒有寫死任何 Modal endpoint。

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

`identity_mode: "endpoint"` 採用本輪實驗的假設：各台模型／tokenizer／serving 設定相同，每個 endpoint 固定對應一台 engine。缺少 fingerprint／epoch 不會阻擋選路；若重新部署或更換 URL 背後的 engine，重啟 Router 清掉舊 session 狀態。省略這個欄位會使用預設 `verified` 模式，要求已驗證的 fingerprint 與 engine 身分。

`renderer_base_url` 只需選一個可用、設定相同的 vLLM renderer，adapter 會呼叫其 `POST /v1/chat/completions/render`。每個 controller 需要接受 `POST /lookup`（body 為 `{"tokens": [...]}`）與 `POST /health`（body 為 `{"instance_id": "worker-one"}`）。`block_size` 要填 vLLM 的實際 block size，與 LMCache chunk size 分開。上述 timeout 是暖機後的實驗設定，可依部署調整。

### 2. 從這個 fork 編譯並啟動

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

將 `POLICY` 改成 `least_load_kv` 或 `kv_batch_ect` 後，用同一指令重啟即可切換。這裡使用本 fork 編譯的 binary；從 PyPI 安裝的版本不保證包含這三個新增 policy。

`--health-check-endpoint /v1/models` 用於 inference 健康檢查；已測過的 Modal controller 入口將 `/health` 留給 POST，GET 會回 405，因此不能用預設的 GET `/health`。Telemetry 若需要驗證 header，可在 config 的 `header_env` 指定環境變數，詳見 [adapter 文件](docs/load_balancing/lmcache-adapter.md)。

### 3. 把推論請求送到 Router

```bash
curl --fail-with-body --max-time 180 \
  http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-session-id: demo-session' \
  -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false},"priority":0}'
```

原始 JSON 與推論串流會經 Router 轉送。使用 OpenAI 相容 client 時，將 `base_url` 設成 `http://127.0.0.1:30000/v1`；上述 `enable_thinking` 設定適用於目前的 Qwen 部署。

### 4. 啟用 ECT 排序與確認是否降級

上述最小 config 足以使用兩個 baseline。**只把 policy 名稱改成 `kv_batch_ect`，尚不會取得有效 ECT 分數**：還要在同一份 JSON 頂層加入以 `instance_id` 為 key 的 `cost_models`，CPU 命中時另需 `restore_models`。

| Config 欄位 | 每台需要的內容 |
| --- | --- |
| `cost_models[instance_id]` | `fingerprint`、`calibration_version`、`prompt_range`、`output_range`、`concurrency_range`、`output_prior`、`prefill`、`decode`、`beta`、`queue_ms` |
| `restore_models[instance_id]` | `fingerprint`、`calibration_version`、`location: "LocalCPUBackend"`、`token_range`、`fixed_ms`、`per_token_ms` |

`prefill` 填 `[a0, a1, a2]`，`decode` 填 `[d0, d1, d2]`；各 `*_range` 填包含端點的 `[min, max]`。成本係數須為有限、非負數，且 token 長度／併發量落在配置範圍內。

Endpoint 模式略過成本模型的 fingerprint 比對，其餘版本、數值與範圍檢查仍保留。每台可以配置不同係數。程式不會自動從 `/metrics` 擬合係數；近似實驗需自行填入並標示估計版本，量測與校準方式見 [ECT 校準文件](docs/load_balancing/ect-calibration.md)、[CPU 還原模型](docs/load_balancing/lmcache-adapter.md#ect-restoration-model) 與 [現有資料可推估的項目](docs/load_balancing/ect-observable-inputs.md)。

後續 [Cache-aware ECT 研究提案](docs/load_balancing/cache-aware-ect-design.md) 整理了 partial CPU prefix、GPU/CPU 重疊與 backend 背景 benchmark 的實測影響，建議直接學習 cache 與負載對完成時間的淨影響。提案尚未替換以上公式；目前 ledger 也不包含直接送往 backend 的外部流量。

```bash
curl --fail http://127.0.0.1:29000/metrics \
  | rg 'router_routing_decisions_total|router_lmcache_observations_total'
```

查看 `router_routing_decisions_total` 的 `policy`、`identity_mode` 與 `fallback` labels；`fallback="none"` 表示這次有使用指定的排序。常見降級原因包括 `render_failed`、`unknown_kv`、`stale_kv`、`missing_cost_model`、`missing_restore_model`。`router_lmcache_observations_total` 則記錄各 worker 的資料取得結果，單純 lookup 成功不代表 ECT 模型已齊備。

### 本機 smoke 與更多文件

安裝 Rust/Cargo 與 `uv` 後，可以用本機 fixtures 測三個 policy，不需 GPU 或線上 controller：

```bash
bash scripts/routing/smoke.sh

# 固定的合成案例：依序選 G0、G2、G1，驗證三個排序的差異
cargo run --example routing_policy_demo
```

完整設計與限制見 [Observed KV routing](docs/load_balancing/observed-kv.md)；資料契約與部署設定見 [LMCache adapter](docs/load_balancing/lmcache-adapter.md)。

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
