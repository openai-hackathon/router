# KV-Batch-ECT：現有 HTTP 資料可觀測與可推估的輸入

更新：2026-09-12。這份分析區分直接觀測、附條件推估與仍然未知的資訊；它不是已完成實機校準的宣告。

**LMCache adapter 已接入 Router；這輪依使用者指定，假設模型／serving 設定相同、endpoint 固定對應 engine，以 `lmcache.identity_mode: "endpoint"` 暫時略過身分驗證。ECT 仍缺適用範圍內的時間模型校準，以及實際 CPU cache 還原樣本。服務時間本身現在可以開始量測，不必等新的 per-request API。** c、d 的 `/metrics` 已包含 prefill、decode、queue 與 inference histogram，可以在隔離單請求窗口用 `_sum` 的差值取得各階段時間。

設定方式見 [LMCache adapter](lmcache-adapter.md)。以下仍列出哪些資訊實際未知，以區分部署假設與直接觀測；epoch／fingerprint 的缺項在 endpoint 模式不再阻擋選路。lookup／health 失敗、資料過期或成本模型缺失仍會共同降級。

## 已實測的範圍

使用既有 [c 六次請求結果](results/lmcache-c-smoke-2026-09-12.json)，並新增 [c/d 交叉結果](results/lmcache-cd-observations-2026-09-12.json)。新增實驗只對使用者指定的 c、d 呼叫已知 renderer、chat、metrics、lookup 與 health 路徑；未呼叫 a/b，未執行 clear、pin、move 或 compress。

新增實驗使用同一個 synthetic prompt。兩台 renderer 都產生 1042 tokens，四次生成回傳的 prompt token IDs 也完全一致。依序執行 c 首次、c repeat、d 首次、d repeat，每次 `max_tokens=8`。同時查兩邊 controller，以檢查各自看見的 cache 分布；四次 inference 本身依序執行，便於歸屬 metrics。

| 時點 | c 的 `/lookup` | d 的 `/lookup` |
| --- | --- | --- |
| 尚未送入這個新 prompt | `layout_info={}` | `layout_info={}` |
| c 首次生成後 | `vllm-c: [LocalCPUBackend, 1024]` | `layout_info={}` |
| c repeat 後 | `vllm-c: [LocalCPUBackend, 1024]` | `layout_info={}` |
| d 首次生成後 | `vllm-c: [LocalCPUBackend, 1024]` | `vllm-d: [LocalCPUBackend, 1024]` |
| d repeat 後 | 同上 | 同上 |

兩邊 controller 的 `POST /health` 分別帶自己的 instance ID，皆回 HTTP 200、`error_codes={"0":0}`。這次資料支持「兩邊可各自查本機 CPU inventory」；沒有證據顯示 c 的 controller 能列出 d，也不能由 c 缺少 a/b 推論 a/b 沒有 cache。

## 四類輸入目前的可用程度

| ECT 所需資訊 | 現有資料能提供什麼 | 不能據此宣稱什麼 | 建議標示 |
| --- | --- | --- | --- |
| 每台 worker 的 prefix 證據 | 正確 tokens 的 `/lookup` 回傳 instance、storage tier、matched prefix length；隔離 metrics 窗口可事後量到 native/external 實際命中 tokens | CPU inventory 不等於 GPU 已駐留 prefix；事後命中率不能預知另一個 prompt 的命中 | `controller_inventory`、`tier=LocalCPUBackend`、`observed_at`；GPU 證據仍需分開 |
| engine 身分與失效 | config 的 endpoint→instance 對應、health、exporter 的 process start / metric creation / counter reset，可作失效警訊 | instance 名稱、`engine="0"`、operation ID 都不是 engine epoch；穩定 counters 不代表 cache 沒 eviction | `identity=configured`、`engine_epoch=unknown`；弱失效訊號另存 |
| ECT 服務時間 | `_sum/_count` 差分可在隔離窗口取得 prefill/decode/queue；usage 提供實際輸出長度；Router 提供送出前 in-flight | 兩次樣本不足以產生可泛化的 P/D/β/Q；backend 時間不含完整網路與 Modal 外部排隊 | `measured_interval` 加上歸屬檢查、量測位置與模型適用範圍 |
| CPU cache 還原成本 | 可以設計 confirmed external-hit 的配對測量；日後取得 retrieve/GPU transfer 指標可直接量測 | 現有 repeat 的外部 hits 全為 0，不能從 repeat 加速或 lookup RTT 算出 CPU→GPU 還原成本 | 目前 `restore_cost=unknown`，不能填 0 |

## Prefix 與 hit rate：可取得，但要保留層級與時間

LMCache 的 legacy lookup 契約回傳 `(location, matched_prefix_length)`，以 instance ID 索引；`event_id` 是該 controller operation 的識別碼。[LMCache lookup 文件](https://docs.lmcache.ai/kv_cache_management/lookup.html)

因此目前有的是「這次查詢時 controller 報告的 CPU prefix」。它沒有 GPU block residency、資料寫入時間、cache epoch、事件序號或 eviction watermark。收到 HTTP response 的時間可以記為觀測時間，卻不能當成 inventory 內部的新鮮度保證。TTL 能限制 Router 重用舊結果多久，無法證明 controller 自己沒有落後。

對 c 六次請求窗口，直接量到：

```text
native queried tokens = 6262
native hit tokens     = 3120
native token hit rate = 3120 / 6262 = 49.8243%

external queried tokens = 3142
external hit tokens     = 0
external token hit rate = 0 / 3142 = 0%
```

這裡的 external 0 是「有觀測到 counter、窗口內增量為 0」，不是把缺資料填成 0。若 counter 缺失、reset、分母為 0 或窗口混入其他流量，應回 `unknown` 或只報 aggregate。LMCache lookup 命中率與 vLLM 已使用的 native/external 命中率是不同指標，不能混用分母。

新增 c/d 實驗把歸屬縮小到單筆 request：兩台首次生成各 native hits=0，repeat 各 native hits=1040；兩台的 `prompt_tokens_by_source_total` repeat 增量都是 `local_compute=2`、`local_cache_hit=1040`、`external_kv_transfer=0`。lookup 的 CPU prefix 則是 1024。**現在已實測到 GPU 命中與 CPU inventory 的數字不同。**

兩台 `cache_config_info` 都回 `block_size=16`。1040 符合 `floor((1042-1)/16)*16`。vLLM 0.29 的 native prefix lookup 只承認完整 blocks，且最多重用到 prompt length−1，以便產生最後 token 的 logits。[vLLM KV cache manager](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/kv_cache_manager.py#L208)

這些 native metrics 可用來驗證「上一筆真的用了多少」，不能建立任意下一筆 request 的完整 GPU index。由歷史成功請求推測還留有 prefix，只能是有 TTL 的 heuristic，不能升格成精確觀測。CPU lookup 回 512、1024、1536 也不足以唯一決定 LMCache chunk size；不要從這幾個倍數寫死 chunk size。

`kv_cache_usage_perc=0` 也不能推論「沒有可重用 GPU cache」。vLLM 的 free block queue 可以保留有 hash 的 cached blocks，等待再利用或 eviction。[vLLM block pool](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/block_pool.py#L30)

## 身分、配置與失效：可以做弱偵測，不能補出真實 epoch

目前可直接取得的配置部分如下，來自新增實驗保存的 `cache_config_info`：

| 欄位 | c | d |
| --- | --- | --- |
| native block size | 16 | 16 |
| GPU block 數 | 10720 | 4766 |
| KV token 容量 | 171520 | 76256 |
| prefix caching | enabled | enabled |
| native prefix hash algorithm | `sha256` | `sha256` |
| cache dtype | `auto` | `auto` |
| engine label | `0` | `0` |

這已足以取得配置的 token 容量，不必從 VRAM 猜；但 `auto` 沒有告訴我們實際 dtype 的 bytes，這個配置也沒有完整模型權重版本、tokenizer/chat template revision、GPU 型號與 engine epoch。兩台容量不同；相同 `model="local"` 和這一個 prompt 的 token parity，不能證明 serving fingerprint 相同或 KV 可互相交換。各台 ECT 係數也應分開校準。

建議 adapter config 明確綁定 `worker_url`、`controller_url`、`instance_id`、預期 serving fingerprint 與允許的 storage tier。c/d 都是配置值，不應出現在 adapter 的選擇邏輯。使用者聲明的 engine identity 只能標為 configured；只有與推論實際 engine 綁定、會在 engine 更換時改變的識別碼，才能標 verified。

現有 `process_start_time_seconds`、`*_created` 與 counters 可組成 exporter 的弱 incarnation 訊號：

```text
process start 改變 OR metric creation 改變 OR counter 倒退
    → 停用舊的 metrics 差分窗口，清掉 Router 暫存 lookup 結果

lookup / health timeout、health 非成功、instance 不符
    → 此來源 unknown/stale，不能繼續宣稱可信 prefix
```

以上是保守的失效處理建議。exporter 可以重啟而 engine 未換，engine 也可能變更而 exporter 尚未重建；同一個 endpoint 還可能切到另一個容器。這些訊號都不能被轉成「已證明舊 engine 結束」來釋放 ledger 的未知 attempt。`event_id` 是每次操作不同的 UUID，也不能當作可排序的 telemetry sequence。

沒有事件序列時，可以每次使用前重查 lookup、縮短結果 TTL、健康失敗即失效；仍然無法檢測未回報的 eviction 或保證選路到送達之間不變。原先嚴格規則保留在預設的 `verified` 模式；目前實驗使用 `endpoint` 模式，決策與 metrics 會標示模式，並讓未知 epoch 維持 null。

## 服務時間：已實測能由 histogram 差分取得

c/d 現有 `/metrics` 皆有：

```text
vllm:request_prefill_time_seconds_{sum,count}
vllm:request_decode_time_seconds_{sum,count}
vllm:request_queue_time_seconds_{sum,count}
vllm:request_inference_time_seconds_{sum,count}
vllm:time_to_first_token_seconds_{sum,count}
vllm:e2e_request_latency_seconds_{sum,count}
vllm:request_prefill_kv_computed_tokens_{sum,count}
vllm:request_prompt_tokens_{sum,count}
vllm:request_generation_tokens_{sum,count}
```

對同一 label set、未 reset 的安靜單請求窗口：

```text
Δ phase_count = 1
phase_ms = 1000 × (phase_sum_after − phase_sum_before)
```

本次四個窗口的各 phase count 都增加 1，prompt histogram 增量皆為 1042，generation histogram 與 usage 相符，沒有 counter creation 變動或 preemption 增量。native query 與 prompt source 加總也都等於 1042。以這些檢查支持窗口歸屬；它不是 engine epoch 或跨容器身分的證明。

| Worker / request | Native H | CPU lookup H | O | Prefill ms | Decode ms | Queue ms | Backend E2E ms | Client HTTP ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c 首次 | 0 | 生成前無報告；生成後 1024 | 3 | 86.916 | 62.387 | 0.021 | 157.178 | 1003 |
| c repeat | 1040 | 1024 | 3 | 64.007 | 63.088 | 0.021 | 134.572 | 996 |
| d 首次 | 0 | 生成前無報告；生成後 1024 | 8 | 438.838 | 311.470 | 0.021 | 760.811 | 1752 |
| d repeat | 1040 | 1024 | 8 | 98.085 | 295.771 | 0.023 | 402.422 | 1431 |

vLLM 0.29 的 prefill interval 從第一次 scheduled 到 first token；decode 從 first token 到 last token；queue 從 queued 到第一次 scheduled。這些是 engine 事件的經過時間，可能包含 preemption。prefill 已含產生第一個 output token，decode 不是再計一次全部 O 個 tokens 的 GPU kernel 純時間。[vLLM timing 定義](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/metrics/stats.py#L484)

因此現在可以開始累積「量測到的 P、D、Q」，但不要直接把上述四個數字當係數：只有一個 L、兩個 H 狀態、每台兩次、且沒有有負載的樣本。d 的首次較慢可能混有 warm-up 或配置差異；兩台實際 O 也不同，不能拿總時間比值當服務能力比值。

可行的後續校準程序：

1. 固定模型/serving fingerprint，記錄 Router 送出前的 `n_j`；unloaded 樣本確認無其他 Router 或外部請求。
2. 測多個 L、H 與 output 上限，保存實際 O 與 cache source。每筆前後 scrape，等待 logger 發布到預期 count，逾時就捨棄窗口。
3. 對每個窗口檢查 label set、counter reset、completion reason、prompt/O 增量、phase count；若有額外流量，保留 aggregate，不硬分配給單筆。
4. 以 unloaded P/D 擬合 baseline，再用多種送出前 n 聯合校準 β 與 Q；group 分開訓練/驗證，保存範圍與誤差。
5. concurrent 窗口若只能取得 N 筆 aggregate，可用 `Δsum/N` 評估群組平均；它不保留每筆 L/H/n 與 latency 的對應，不能假裝 N 筆獨立校準資料。

目前六次短回覆的 O 只有 3–8 tokens，無法代表實際 agent/tool workload 的 output prior。可以從現有 usage 開始累積 prior，但必須按 workload 分組；max_tokens 是上限，不能當成預期 O。

若想預測 Router 使用者感受到的完成時間，還需包含傳輸/HTTP preprocessing/Modal 路徑成本。表中同一筆 `Client HTTP − backend E2E` 約 846–1029 ms，是可量到的外部總差額；它混合網路、連線、平台與量測邊界，不能全部叫作 network RTT 或 backend queue。單次 lookup 的 HTTP 時間也約 0.84–1.03 秒，不能當成 GPU 或 CPU transfer 成本。這些 overhead 應分別量測，避免塞入 β/Q 又重複計算。

## CPU 還原成本：目前仍無可辨識樣本

目前 `/lookup` 證實 CPU inventory 存在；上述兩輪 smoke 的 external cache hit 增量皆為 0，新增 c/d 窗口的 `source="external_kv_transfer"` 也都是 0。這意味著這批請求沒有觀測到經 connector 使用的外部 cache tokens，**無法量出 CPU→GPU restore 成本**。這不證明 LMCache 不支援還原，只表示實驗沒有隔離出這條路徑。

要從既有可存取的 API 推估，至少需要一批符合以下條件的 request：lookup 顯示 CPU prefix；其 native 實际 H 比 CPU prefix 短；執行後 external transfer token counter 確實增加。可等待自然 GPU eviction，或另外設計保留 CPU、減少 GPU residency 的受控實驗；不能將一般 warm repeat 當成 CPU restore。

在相同 worker/L/O/n、原生 GPU 命中已量到且沒有其他外部 backend 的情況下，可擬合「還原增加的關鍵路徑時間」：

```text
H_usable = max(H_gpu, H_cpu)          # 相同起點的重疊 prefix，不能相加
R_tokens = max(0, H_cpu − H_gpu)      # 額外需要還原的部分

restore_effective_ms
    ≈ prefill_interval_ms
       − baseline_prefill_ms(L, H_usable)
```

這是模型殘差估計，需先確認該段 transfer 落在 prefill interval；若 transfer 在 scheduled 前排隊或與計算重疊，改以相符的整段 backend latency 擬合，不能把這個差值叫純 PCIe 傳輸時間。負殘差表示噪聲、重疊或 baseline 不合，應保留診斷/誤差而不是宣稱 restore=0。若 H_gpu 本來比 H_cpu 長，這筆就不是還原樣本。

另一條路是暴露 LMCache 的 `time_to_retrieve`、`retrieve_to_gpu_time`、`num_hit_tokens`、`retrieve_speed` 等版本相容的指標。官方 legacy metrics 文件列有這些指標，但目前這兩個公開 `/metrics` 沒有任何 `lmcache:` sample；不能假設文件上存在就代表部署已提供。[LMCache metrics reference](https://docs.lmcache.ai/production/observability/metrics.html)

在模型 layout、KV dtype、TP shard 與有效頻寬都確認後，標準 attention 的 bytes 模型也能提供粗估：

```text
KV_bytes(R) ≈ 2 × layers × local_KV_heads × head_dim × dtype_bytes × R
restore_ms  ≈ fixed_overhead_ms + 1000 × KV_bytes(R) / effective_bytes_per_second
```

這只適用於對應的 KV layout，忽略 padding/量化 metadata/壓縮與 overlap；目前還沒有實際 dtype、layout 與頻寬，因此只能作明示假設的先驗，不能稱校準結果。硬體標稱頻寬本身也不是 end-to-end 有效頻寬。

CPU 來源若要進 ECT，應對 tier 保留額外成本或使用包含該 tier 的獨立校準模型。不能把 CPU prefix 直接代入原本 GPU baseline 的 H 並省略 restore，否則節省的 prefill 被計入，搬運卻被當成免費。是否把 restore 納入 load multiplier，要由實測校準與量測邊界決定。

## 接入時可立即採用的規則

- Controller URL、instance 對應、fingerprint 與允許 tier 全部由 config 輸入；不同 worker 可以各自有 controller。
- 保存原始 lookup tier/length/operation ID、觀測時間與來源狀態；對不受支援的 tier 或 schema 明確降級。
- `layout_info={}` 只表示這次 controller 沒報告該 prefix。尚未驗證健康/身分/完整性的來源，不能據此產生精確 H=0。
- `H_gpu`、CPU inventory 與執行後的 cache hit counters 分開記錄；不以 hit rate 推任意 prompt 的 H。
- 先建立服務時間採樣與原生 cache 模型；engine epoch 與 CPU restore 未補齊時，完整 ECT 的 evidence/model gating 維持明確 fallback。

本次 live 工作已結束：2 次 render、4 次 inference、18 次 lookup、8 次 metrics、2 次 health，全為 HTTP 200；沒有仍執行中的請求。新增 JSON 不含 prompt 內容或 token IDs。
