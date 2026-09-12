# 以現有 telemetry 改進 Cache-aware ECT

日期：2026-09-12。本文整理目前程式、d 的重複查詢與 Router smoke，以及 upstream 的相關方法。**模型改版是研究提案，尚未替換 `kv_batch_ect` 的現行公式；本輪已實作的是 partial CPU prefix 的 adapter 修正。**

建議維持三個政策及共用資料管線，前兩個 baseline 保持原定排序。第三個政策改為根據 cache、工作量和實測延遲預估完成時間，名稱可稱 **Cache-aware ECT**；CLI 暫時保留 `kv_batch_ect` 相容性。目前的 `β × n` 只是併發負載近似，沒有觀測實際 batch 組成，名稱中的 Batch 容易讓人高估模型能力。

## 本輪資料改變了哪些判斷

### lookup 有有效資訊，也可能回傳不滿一個 native block 的尾端

目前 d 的 `POST /lookup` 仍只回 `event_id` 和 `layout_info`。後者的 tuple 第二項是 matched prefix length，adapter 將它命名為 `cached_tokens`；HTTP response 沒有新增名叫 `cached_tokens` 的欄位。這符合已部署的 LMCache legacy controller 契約。[LMCache lookup](https://docs.lmcache.ai/kv_cache_management/lookup.html)

[兩組 prefix 實驗](results/lmcache-d-prefix-semantics-2026-09-12.json) 各先送一筆新 synthetic prompt，再改變查詢 prefix：

| 查詢 | 第一組，L=1045 | 第二組，L=1043 |
| --- | ---: | ---: |
| 生成前，相同完整 tokens | 無命中 | 無命中 |
| 生成後，相同完整 tokens | 1045 | 1043 |
| 只查前 512 tokens | 512 | 512 |
| 只查前 1024 tokens | 1024 | 1024 |
| 去掉最後一個 token | 1024 | 1024 |
| 在尾端追加 16 tokens | 1024 | 1024 |
| 修改第一個 token | 無命中 | 無命中 |
| 修改 index 512 的 token | 512 | 512 |

第二組同一份完整 tokens 連查六次，全部為 1043。這次觀察到的是 warm-up 和查詢 prefix 改變帶來的差異，沒有觀察到固定查詢的隨機波動。不同 L 的結果不能串成同一個 prompt 的 cache 增長曲線。

原 adapter 錯把「raw CPU 長度必須是 vLLM block size 的倍數」當成契約，現在已修正：保留原始長度，排序用的 H 才向下取完整 blocks 並保留最後 token 的計算。LMCache 有儲存未滿 chunk 的配置，但這些 lookup 結果本身不能證明部署啟用了哪個旗標或唯一決定 chunk size。[LMCache 配置](https://docs.lmcache.ai/api_reference/configurations.html)

```text
cached_tokens = 原始 CPU matched prefix
H = floor(min(cached_tokens, L - 1) / native_block_size) × native_block_size

L=1043、cached_tokens=1043、block_size=16 → H=1040
```

### external hits 已不再全是零，但仍不能辨識純還原成本

[d-only Router smoke](results/router-d-partial-prefix-smoke-2026-09-12.json) 的最後一個窗口：

| 項目 | 觀測值 |
| --- | ---: |
| 這次 prompt L / output O | 1557 / 32 |
| dispatch 使用的 CPU lookup 長度 | 1557 |
| 各 prefill/decode/queue/inference completion count 增量 | 1 |
| native cache token 增量 | 1552 |
| external transfer token 增量 | 4 |
| local compute token 增量 | 1 |
| prefill elapsed 增量 | 108.213 ms |

三種 token source 加總與 L 相符，phase counts 也與單次完成一致，因此這是有用的線索。**它仍是 aggregate 窗口，沒有 request ID 關聯，且本輪未保存 generation histogram 的歸屬檢查，不能升格成完整的 per-request trace。**

可以確認觀測窗口已有 external hits；不能把 108.213 ms 全當成搬運 4 tokens 的成本，也不能說整段 1557 CPU tokens 都搬了一次。lookup 告訴我們 CPU 可用量，實際使用還受 GPU 重疊、最後 token 計算及 connector 行為影響。官方 source accounting 也分開 local cache 與 external tokens。[vLLM 0.29 stats](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/metrics/stats.py)

因此現行 `R = fixed + per_token × cached_tokens` 的整段收費假設可能高估 warm request 的代價；「不把未知還原填成免費」仍然必要，但不能把保守收費說成已量到的 transfer。要從有限 telemetry 選路，較可行的是學習 **這種 cache 證據對完成時間的淨影響**，不強求拆出不可辨識的物理階段。

### d 有背景 benchmark，Router ledger 不是整台的負載

使用者指出 d 同時由另一個 benchmark 使用。本輪六次請求的 metrics-before 都是 `running=2, waiting=0`，本 Router dispatch 前的 in-flight 則為 0；有些窗口只送一筆，completion count 卻增加 4、3 或 2。

如果看的 counter 是 **d 的 vLLM `/metrics`**，它涵蓋 backend 的其他流量，這個差異符合背景請求。若看的其實是本 Router 的 `router_dispatch_finished_total`，直接打 d 的外部流量不會增加它，必須另外追蹤 attempt/retry，不能用同一理由解釋。

多筆完成窗口已標成 `mixed_window_do_not_assign_phase_times`。即使某個窗口只完成一筆，其他請求也可能一直佔用 batch，所以本輪六筆都標成 `unloaded_baseline_eligible=false`。它們可以保留為有背景負載的 client latency 樣本；不能拿來擬合空載 P/D，也不能把混合窗口的 phase sum 指派給自己的 request。

## 相關專案提供的方向

| 專案 | 官方實作／文件中的做法 | 對我們的意義 |
| --- | --- | --- |
| production-stack | `LoadAwareRouter` 以 cache match ratio 減去相對負載懲罰；load 來自 Router 的 prefill+decode request 計數 | controller prefix 加負載已是既有方法；加權 score 可以先工作，但其單位不是完成毫秒 |
| NVIDIA Dynamo | 綜合已派出的 prompt 工作量、incoming uncached work、active KV blocks，並按 cache tier 給 credit | 可以在 Router 追蹤工作量與 phase，改善每筆 request 都只算 1 的近似 |
| llm-d latency scorer | 使用 TTFT/TPOT 預測與延遲目標，預測不可用時改用 composite score | 可直接預測可觀測延遲，不必先取得 scheduler 每一階段的精確狀態 |

來源：[production-stack routing code](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/routers/routing_logic.py)、[Dynamo routing concepts](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts)、[llm-d latency scorer](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/scheduling/scorer/latency/README.md)。查閱日期同本文；main/dev 文件可能繼續變動。

這些方法所用資料與部署不完全相同，不能把它們的預設權重當成 c/d 的實測係數。我們可借用特徵與量測方法；目前不能宣稱 cache+load 的組合本身是新的研究貢獻。

## 沒有新增 vLLM API，也能補哪些輸入

| 輸入 | 現有來源 | Router 現況／建議用途 |
| --- | --- | --- |
| L、CPU prefix C、C/L | renderer + 每台 lookup | 已接入；保留 raw C、block-normalized H、tier、觀測時間 |
| 本 Router 未完成數 | 共用 dispatch ledger | 已接入；保留原始 n，所有排序使用送出前 snapshot |
| backend running/waiting、counter rate、preemption | 已能存取的 `/metrics` | live script 有量；尚須 background collector 接到選路 snapshot，作背景壓力特徵 |
| 每筆實際完成延遲 | Router 的 dispatch、body 終止事件 | 需要新增採樣；可直接做第一版 completion predictor |
| TTFT、首輸出後 elapsed、實際 O | streaming body + usage | 需要新增採樣；非串流只量總時間，缺 usage 時不猜 token 數 |
| 等待首輸出的 prompt 工作量 | ledger 保存 L/C，遇首個模型輸出轉 phase | 需要新增欄位；是 workload proxy，不宣稱知道 GPU prefill progress |
| decode 中的 request 數與 context 長度總和 | 同一份 ledger | 需要新增欄位；不把總和宣稱成去重後的真實 GPU KV 佔用 |
| 最近同 prefix 成功時間、重複次數 | Router 的成功請求歷史 | 可增加 TTL 特徵；只能推測 residency 機率，不能當 GPU 命中證據 |
| 輸出長度 prior | 自己完成請求的 usage + finish_reason | 按 workload 分組；output 上限及被截斷樣本要處理，不能把 max_tokens 當平均輸出 |

backend 的全域 hit rate 只能描述最近 workload 的平均行為，無法推回任意新 prompt 的命中。全域 tokens/s 在多人使用時也是總吞吐，不是每筆 request 的 decode speed。

CPU→GPU 的純時間、即時 GPU prefix、精確 batch 組成與 engine epoch 仍不能唯一回推。endpoint 身分假設繼續依使用者指定沿用，不把 epoch 作為本次工作的前置要求。日後若增加 LMCache retrieve/transfer 指標，可再拆細模型；本輪公開 metrics 仍未看到 `lmcache:` samples。[LMCache metrics](https://docs.lmcache.ai/production/observability/metrics.html)

## 建議第三個政策如何修改

### 第一版：直接估計完成時間，吸收 cache 的淨收益

令 C 為 controller 的 raw CPU prefix；X 為選路當下的負載與近期重複特徵。對每個 endpoint 學習：

```text
E_j = completion_model_j(L, C, estimated_output, X)
```

訓練 target 是 **Router 向該 worker dispatch 至成功 body 終止的 elapsed**，包含 backend／Modal 在這個邊界內的排隊、推論和傳輸。這個版本可先用 non-streaming 樣本，避免因尚未有可靠 first-token hook 而卡住。先用少量參數、分桶或正則化模型；資料量不足時不要同時擬合一長串高度相關特徵。

此模型以 CPU prefix 作預測特徵，讓樣本學到 GPU 重疊與 CPU 還原的合成效果。它不輸出「搬了多少 tokens」，也不將 C 當精確 GPU H。快取效果可能依 L、負載、距上次重用時間而不同，不能只用全域 hit rate 或固定百分比折扣。

### 有可靠 streaming 樣本後，再分成兩段

```text
F_j = first_output_model_j(L, C, X)
D_j = effective_time_per_output_token_model_j(L, X)
E_j = F_j + max(estimated_output - 1, 0) × D_j
```

F 從 dispatch 量到第一個真正模型輸出，D 是首輸出之後的平均 elapsed/token。第一個 token 已在 F 內，不能再算一次。vLLM 的 prefill/decode timing 也以 first token 分界，但 Router 的 HTTP 觀測含傳輸差異，兩種樣本應分開。[vLLM timing 定義](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/metrics/stats.py#L484)

HTTP 200、role-only SSE、空 delta 都不是首個模型輸出；content、reasoning、tool-call 輸出需要相應處理。SSE chunk 可以包含多個 tokens，不能數 chunks 當 O。這是可觀測的延遲近似，不是精確 inter-token GPU 時間。

**以上模型已涵蓋 cache、負載與排隊效應，就不再另加整段 R、Q 或乘一次 `1+βn`。** 現行分解模型與新 empirical 模型應是第三個政策的不同 model version，不能把兩套成本疊在一起。

render/lookup 是目前選擇前已支付的共同前處理，另記 overhead 供整體 benchmark 評估，排序時不再對每台加一次。本輪 lookup 有約秒級延遲，應計入 Router E2E 報告；lookup RTT 本身不是 CPU 還原成本。

### 背景流量與 phase 要用共用 snapshot 表達

保留 `n_router`、`backend_running`、`backend_waiting`、metrics age 為不同特徵，校準時一起考慮。**不要使用 `n_router + running + waiting`**；有重疊，而且 snapshot 時刻與 Modal 排隊邊界不同。`max(n_router, running+waiting)` 可作明示的粗略壓力 proxy，仍不是已重建的精確總未完成數，優先使用分開的特徵。

部署模式應明示 `exclusive_router` 或 `shared_backend`（建議的新 metadata，尚未加入 config）。新模型若依賴 backend 壓力，在 collector stale／缺資料時須共同降級或使用另有校準的降階模型，不能把缺測填成 idle。若未來要改兩個 baseline 的 load 定義，需另立實驗名稱並一致比較；目前 `least_load_kv` 的 n 仍是 Router ledger，無法平衡它看不到的外部流量。

在 ledger 補 phase 時，可先把每筆的整段 prompt 工作量保留到首輸出，之後移到 decode request/context 計數；不需要預測每筆剩餘 tokens。對 CPU prefix 的工作折扣仍由模型決定，不能直接宣稱 `L-C` 就是剩餘 GPU 計算量。reservation、phase 更新和 release_once 沿用同一 lifecycle。

### 資料不足時，分數名稱要符合它的單位

若暫時只使用手填 `cache_credit - load_penalty`，可以稱 **Cache-aware Cost Routing**，輸出 `score_units=relative`。這可以是明示的實驗模式，不能偽裝成毫秒 ECT，也不能直接沿用 `+10 ms` 的 affinity 門檻。這些欄位和模式是提案，現行 config 尚未支援。

有實測 target、適用範圍和留出驗證後，近似模型也可以稱 Cache-aware ECT，不要求拆出所有物理成本。保留固定平手規則和有界 affinity；每個候選必須使用可比較的同種分數，不能一台有時間模型、另一台缺模型就改塞相對分數。

## 一個 Router 配至少三台 worker 時

這裡依使用者補充，假設是 **一個 Router 的 config 內有至少三個 inference workers**。現行 adapter 依 worker/config 清單查詢，沒有固定 c、d，也沒有把候選數寫成兩台；lookup 的並行數有界，不代表 worker 總數只能有該數量。

| Policy | 三台以上是否需要改排序 | 需要做的工作 |
| --- | --- | --- |
| `prefix_max` | 不需要：H 最大，再 n 最少 | 每台取得各自的 prefix，維持完整 blocks 和一致的 CPU inventory 語意 |
| `least_load_kv` | 不需要：n 最少，再 H 最大 | 共用 ledger 已支援多 worker；直接打 backend 的 benchmark 仍不在此 n 中 |
| `kv_batch_ect` | 保留「預估完成時間最小」的目標，改進預測模型 | 各 endpoint 的服務能力、背景壓力和 cache 淨收益分別建模，再比較同單位的分數 |

增加 worker 數不會自動補齊 ECT 係數。同模型也不保證不同 GPU、KV 容量與背景流量下的服務速度相同；不能只量 d，就把它的時間曲線複製到另外兩台。可以先共用模型形式，但各台係數需要樣本或明示且驗證過的共用條件。

每個 worker 都要有正確的 URL→controller→instance binding；現行模型以各自 instance ID 配置。**只要一台健康候選缺必要 prefix 或適用的成本模型，就會整次共同降級**；加到三台後要特別報告 fallback 比例，不能讓第三台長期缺資料卻把結果稱 ECT benchmark。這個保守規則暫時維持，不把未知 H 當零，也不默默移除缺 telemetry 的健康 worker。

多 worker 的排序差異已有三候選 fixture 驗證：原定案例依序選 G0、G2、G1；endpoint 模式也涵蓋三候選。真實 d-only smoke 只能驗證接入；接下來三台實機驗證要讓 cache、負載與預估服務時間形成不同偏好，並確認 chosen worker、reservation 後的 n 與 fallback 原因。

如果部署改成三個 **Router 程序**，則是另一個問題：現在 ledger 只在各程序內共享，彼此的 reservations 不會自動同步，需要協調或明確的跨 Router 負載方案。

## 校準與實作的具體順序

1. **共用採樣**：新增 per-attempt dispatch／first-output／terminal 時間、L/C、usage、finish reason、送出前負載 snapshot。取消／失聯不是正常完成時間；retry 分別記 attempt，request E2E 另記。不要保存 prompt 或 token IDs 到一般 log。
2. **背景 metrics collector**：依配置抓每台，處理新鮮度與 counter reset，與自己的 ledger 分開；派送 hot path 不同步 scrape。只使用 dispatch 當下已知特徵，避免把之後的 hit counters 偷放進 predictor。
3. **先訓練小型 completion predictor**：每台量不同 L、C/L、O 和負載。單筆 Router latency 可在背景 benchmark 下量測，但必須記負載；無法歸屬的 backend phase histogram 只作 aggregate。若要空載 P/D，另安排確實隔離的窗口。
4. **補 phase workload 與 streaming predictor**：在既有 ledger 增加 prompt/context 計數，驗證 first-output 與 terminal 只更新一次；非串流保留 coarse phase，不假裝知道首 token 時刻。
5. **接入第三個 ranker 的新 model version**：保存資料來源、sample count、日期、domain、validation error；保留舊 CLI 與共用 fallback，不改兩個 baseline。
6. **在保留的 workload 測路由效果**：依 prompt/session 分組切 train/test，避免同 prefix 洩漏；比較 client completion p50/p95、TTFT、吞吐、lookup overhead、fallback 比例，以及預測選擇的誤差。新 worker 或範圍外輸入先共同降級，不能靠永久忽略沒有樣本的 worker 維持表面成績。

需要特別修正的採樣位置：目前 [HTTP Router](../../src/routers/http/router.rs) 的 `record_generate_duration` 在 response 物件回傳時記錄，對 streaming 並非 body 完成；不可直接拿它訓練 completion 模型。[Dispatch ledger](../../src/core/dispatch.rs) 已處理釋放，但尚未保存上述時間與 phase 特徵。

目前完成的 regression 驗證：3 個 adapter 單元測試、15 個 snapshot 測試、26 個 Python smoke／calibration 測試及 Clippy 通過；d 的六筆真實 Router request 均成功，兩個 baseline 無 fallback，ECT 因未配置成本模型回報 `missing_cost_model`。單 worker smoke 驗證的是資料接入與生命週期，三個政策的選擇差異由多 worker fixtures 驗證，沒有用 d 的不同 prompt 延遲宣稱哪個政策更快。
