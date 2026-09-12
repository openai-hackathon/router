//! Request-scoped LMCache placement adapter. Queries explicit controllers
//! outside the ledger lock; never fabricates native KV events or engine epochs.
use super::{features::RequestFeatures, PrefixEvidence, WorkerMetadata};
use crate::core::Worker;
use futures_util::{stream, FutureExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IdentityMode {
    #[default]
    Verified,
    /// Experiment assumption: equal serving configuration and fixed endpoints.
    Endpoint,
}

impl IdentityMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Endpoint => "endpoint",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LmCacheWorker {
    pub controller_url: String,
    pub instance_id: String,
    /// Pinned vLLM GPU block size; also bounds the final-logits prefix cap.
    pub block_size: usize,
    #[serde(default)]
    pub fingerprint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LmCacheConfig {
    #[serde(default)]
    pub identity_mode: IdentityMode,
    pub renderer_base_url: String,
    pub model: String,
    /// Operator-supplied renderer serving fingerprint, independently checked
    /// against each worker's fingerprint. Missing means explicit fallback.
    #[serde(default)]
    pub fingerprint: Option<String>,
    /// Keyed by exact inference worker URL, independently of Controller URL.
    pub workers: HashMap<String, LmCacheWorker>,
}

impl LmCacheConfig {
    pub fn validate(&self) -> Result<(), String> {
        let url = |value: &str| -> Result<(), String> {
            let parsed = reqwest::Url::parse(value).map_err(|_| "invalid LMCache URL")?;
            if !matches!(parsed.scheme(), "http" | "https")
                || parsed.host_str().is_none()
                || !parsed.username().is_empty()
                || parsed.password().is_some()
                || parsed.query().is_some()
                || parsed.fragment().is_some()
            {
                return Err("LMCache URLs must be explicit HTTP(S) base URLs; use header_env for credentials".into());
            }
            Ok(())
        };
        url(&self.renderer_base_url)?;
        if self.model.is_empty() || self.fingerprint.as_ref().is_some_and(String::is_empty) {
            return Err("LMCache model and supplied fingerprint must not be empty".into());
        }
        let mut instances = HashSet::new();
        let mut urls = HashSet::new();
        for (inference, worker) in &self.workers {
            url(inference)?;
            url(&worker.controller_url)?;
            if worker.instance_id.is_empty()
                || worker.block_size == 0
                || worker.fingerprint.as_ref().is_some_and(String::is_empty)
                || !instances.insert(&worker.instance_id)
                || !urls.insert(inference.trim_end_matches('/'))
            {
                return Err(
                    "LMCache workers require unique URLs/instance IDs and a positive block size"
                        .into(),
                );
            }
        }
        Ok(())
    }

    pub(super) fn worker(&self, url: &str) -> Option<&LmCacheWorker> {
        self.workers
            .iter()
            .find(|(key, _)| key.trim_end_matches('/') == url.trim_end_matches('/'))
            .map(|(_, worker)| worker)
    }
}

/// Exact native Chat rendering, restricted to the token-parity-verified text
/// path. Responses and non-text/cache-isolated requests remain unsupported.
pub async fn build_features(
    client: &reqwest::Client,
    config: &LmCacheConfig,
    route: &str,
    body: &Value,
    headers: Option<&http::HeaderMap>,
) -> RequestFeatures {
    let mut features = RequestFeatures::unsupported(body, headers);
    if route != "/v1/chat/completions"
        || features.model.as_deref() != Some(config.model.as_str())
        || [
            "cache_salt",
            "lora_request",
            "prompt_embeds",
            "multi_modal_data",
            "mm_processor_kwargs",
        ]
        .iter()
        .any(|key| body.get(key).is_some_and(|v| !v.is_null()))
        || body
            .get("messages")
            .and_then(Value::as_array)
            .is_none_or(|messages| {
                messages.iter().any(|message| {
                    message
                        .get("content")
                        .is_some_and(|v| !v.is_null() && !v.is_string())
                })
            })
    {
        return features;
    }
    features.fallback_reason = Some("render_failed");
    let result = client
        .post(format!(
            "{}{route}/render",
            config.renderer_base_url.trim_end_matches('/')
        ))
        .json(body)
        .send()
        .await;
    if let Ok(response) = result {
        if response.status().is_success() {
            #[derive(Deserialize)]
            struct Rendered {
                token_ids: Vec<u32>,
            }
            if let Ok(rendered) = response.json::<Rendered>().await {
                if !rendered.token_ids.is_empty() {
                    features.tokens = Some(rendered.token_ids);
                    features.fingerprint = config.fingerprint.clone();
                    features.fallback_reason = (config.identity_mode == IdentityMode::Verified
                        && config.fingerprint.is_none())
                    .then_some("missing_serving_fingerprint");
                }
            }
        }
    }
    features
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub metadata: WorkerMetadata,
    pub evidence: PrefixEvidence,
    pub(super) started: Instant,
}

impl Observation {
    pub fn evidence(&self) -> PrefixEvidence {
        let mut evidence = self.evidence.clone();
        if let PrefixEvidence::LmCacheObserved { age_ms, .. } = &mut evidence {
            // Include HTTP round trip and time spent waiting for other workers.
            *age_ms = self.started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        }
        evidence
    }
}

pub type Observations = HashMap<String, Observation>;

#[derive(Deserialize)]
struct Lookup {
    layout_info: HashMap<String, (String, usize)>,
}
#[derive(Deserialize)]
struct Health {
    error_codes: HashMap<String, i32>,
}

async fn observe(
    client: &reqwest::Client,
    config: &LmCacheConfig,
    worker: &LmCacheWorker,
    tokens: &[u32],
) -> Result<Observation, &'static str> {
    let started = Instant::now();
    let controller = worker.controller_url.trim_end_matches('/');
    let (lookup, health) = tokio::join!(
        client
            .post(format!("{controller}/lookup"))
            .json(&serde_json::json!({"tokens": tokens}))
            .send(),
        client
            .post(format!("{controller}/health"))
            .json(&serde_json::json!({"instance_id": worker.instance_id}))
            .send(),
    );
    let lookup = lookup.map_err(|_| "lookup_transport_error")?;
    let health = health.map_err(|_| "health_transport_error")?;
    if !lookup.status().is_success() || !health.status().is_success() {
        return Err("controller_http_error");
    }
    // Only engine identity explicitly supplied by the endpoint is usable.
    // Both replies must agree; event_id, metrics engine="0" and timestamps are
    // not restart epochs. Current unextended controllers retain None here.
    let identity = |headers: &http::HeaderMap| {
        let id = headers.get("x-routing-worker-id")?.to_str().ok()?;
        let epoch = headers.get("x-routing-engine-epoch")?.to_str().ok()?;
        (id == worker.instance_id && !epoch.is_empty()).then(|| epoch.to_owned())
    };
    let epoch = identity(lookup.headers())
        .filter(|epoch| identity(health.headers()).as_ref() == Some(epoch));
    let health = health
        .json::<Health>()
        .await
        .map_err(|_| "invalid_controller_health")?;
    if health.error_codes.len() != 1 || health.error_codes.get("0") != Some(&0) {
        return Err("unavailable_controller_instance");
    }
    let lookup = lookup
        .json::<Lookup>()
        .await
        .map_err(|_| "invalid_lookup")?;
    let (location, cached_tokens) = lookup
        .layout_info
        .get(&worker.instance_id)
        .cloned()
        .unwrap_or(("LocalCPUBackend".into(), 0));
    if location != "LocalCPUBackend" {
        return Err("unsupported_cache_tier");
    }
    if cached_tokens > tokens.len() {
        return Err("invalid_lookup_prefix");
    }
    // LMCache's matched prefix can include a partial native block. Preserve
    // that reported length for restoration accounting; only the prefix used
    // for routing is rounded down, with the final prompt token excluded.
    let reusable =
        cached_tokens.min(tokens.len().saturating_sub(1)) / worker.block_size * worker.block_size;
    Ok(Observation {
        metadata: WorkerMetadata {
            worker_id: worker.instance_id.clone(),
            model: config.model.clone(),
            fingerprint: worker.fingerprint.clone().unwrap_or_default(),
            engine_epoch: epoch.clone().unwrap_or_default(),
            block_size: worker.block_size,
        },
        evidence: PrefixEvidence::LmCacheObserved {
            tokens: reusable,
            cached_tokens,
            instance_id: worker.instance_id.clone(),
            location,
            engine_epoch: epoch,
            age_ms: 0,
        },
        started,
    })
}

pub async fn lookup_workers(
    client: &reqwest::Client,
    config: &LmCacheConfig,
    workers: &[Arc<dyn Worker>],
    features: &RequestFeatures,
) -> Observations {
    let Some(tokens) = features.tokens.as_ref() else {
        return HashMap::new();
    };
    let available: Vec<_> = workers
        .iter()
        .filter(|w| w.is_available())
        .cloned()
        .collect();
    let config = Arc::new(config.clone());
    let tokens = Arc::new(tokens.clone());
    let jobs: Vec<_> = available.into_iter().map(|worker| {
        let client = client.clone();
        let config = config.clone();
        let tokens = tokens.clone();
        async move {
            let result = match config.worker(worker.url()) {
                Some(configured) => observe(&client, &config, configured, &tokens).await,
                None => Err("missing_worker_config"),
            };
            metrics::counter!("router_lmcache_observations_total", "worker" => worker.url().to_owned(), "result" => result.as_ref().err().copied().unwrap_or("observed")).increment(1);
            result.ok().map(|observation| (worker.url().to_owned(), observation))
        }.boxed()
    }).collect();
    stream::iter(jobs)
        .buffer_unordered(8)
        .filter_map(|value| async { value })
        .collect()
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use axum::{response::IntoResponse, routing::post, Json};

    async fn controller(
        layout: Value,
        health: Value,
        identity: bool,
    ) -> (
        String,
        tokio::task::JoinHandle<()>,
        Arc<parking_lot::Mutex<Value>>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let layout = Arc::new(parking_lot::Mutex::new(layout));
        let current_layout = layout.clone();
        let response = move |body: Value| {
            let mut response = Json(body).into_response();
            if identity {
                response
                    .headers_mut()
                    .insert("x-routing-worker-id", "custom-instance".parse().unwrap());
                response
                    .headers_mut()
                    .insert("x-routing-engine-epoch", "engine-1".parse().unwrap());
            }
            response
        };
        let app = axum::Router::new()
            .route(
                "/lookup",
                post(move |Json(body): Json<Value>| async move {
                    assert_eq!(body["tokens"], serde_json::json!([1, 2, 3, 4, 5]));
                    let layout = current_layout.lock().clone();
                    response(
                        serde_json::json!({"event_id":"operation-not-epoch", "layout_info":layout}),
                    )
                }),
            )
            .route(
                "/health",
                post(move |Json(body): Json<Value>| async move {
                    assert_eq!(body["instance_id"], "custom-instance");
                    response(serde_json::json!({"error_codes":health}))
                }),
            );
        (
            url,
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }),
            layout,
        )
    }

    fn config_for(url: &str) -> LmCacheConfig {
        LmCacheConfig {
            identity_mode: IdentityMode::Verified,
            renderer_base_url: url.to_owned(),
            model: "local".into(),
            fingerprint: Some("fp".into()),
            workers: HashMap::from([(
                url.to_owned(),
                LmCacheWorker {
                    controller_url: url.to_owned(),
                    instance_id: "custom-instance".into(),
                    block_size: 2,
                    fingerprint: Some("fp".into()),
                },
            )]),
        }
    }

    async fn observations(layout: Value, health: Value, identity: bool) -> Observation {
        let (url, task, _) = controller(layout, health, identity).await;
        let config = config_for(&url);
        config.validate().unwrap();
        let workers: Vec<Arc<dyn Worker>> =
            vec![Arc::new(BasicWorker::new(url.clone(), WorkerType::Regular))];
        let mut features = RequestFeatures::unsupported(&Value::Null, None);
        features.tokens = Some(vec![1, 2, 3, 4, 5]);
        let result = lookup_workers(&reqwest::Client::new(), &config, &workers, &features).await;
        task.abort();
        result.get(&url).cloned().unwrap_or(Observation {
            metadata: WorkerMetadata {
                worker_id: String::new(),
                model: String::new(),
                fingerprint: String::new(),
                engine_epoch: String::new(),
                block_size: 2,
            },
            evidence: PrefixEvidence::Unknown,
            started: Instant::now(),
        })
    }

    #[tokio::test]
    async fn repeated_lookup_preserves_updated_partial_prefix_and_rounds_only_routing_tokens() {
        let (url, task, layout) =
            controller(serde_json::json!({}), serde_json::json!({"0": 0}), false).await;
        let config = config_for(&url);
        let client = reqwest::Client::new();
        // Same endpoint and request: no match, partial block, growth, complete
        // prompt, then eviction. Each observation must reflect the latest value.
        for (cached, expected) in [(0, 0), (1, 0), (3, 2), (5, 4), (2, 2)] {
            *layout.lock() = serde_json::json!({"custom-instance": ["LocalCPUBackend", cached]});
            let observed = observe(
                &client,
                &config,
                config.worker(&url).unwrap(),
                &[1, 2, 3, 4, 5],
            )
            .await
            .unwrap();
            assert!(matches!(
                observed.evidence,
                PrefixEvidence::LmCacheObserved {
                    tokens,
                    cached_tokens,
                    engine_epoch: None,
                    ..
                } if tokens == expected && cached_tokens == cached
            ));
        }
        task.abort();
    }

    #[tokio::test]
    async fn health_gates_cache_misses_and_scope_never_uses_another_instance() {
        let observed = observations(
            serde_json::json!({"other-instance":["LocalCPUBackend",4]}),
            serde_json::json!({"0":0}),
            true,
        )
        .await;
        assert!(matches!(
            observed.evidence,
            PrefixEvidence::LmCacheObserved {
                tokens: 0,
                engine_epoch: Some(_),
                ..
            }
        ));
        for health in [
            serde_json::json!({}),
            serde_json::json!({"0":1}),
            serde_json::json!({"0":0,"1":0}),
        ] {
            assert!(matches!(
                observations(serde_json::json!({}), health, true)
                    .await
                    .evidence,
                PrefixEvidence::Unknown
            ));
        }
    }

    #[tokio::test]
    async fn cache_tier_and_unverified_identity_survive_without_fabricated_epoch() {
        let observed = observations(
            serde_json::json!({"custom-instance":["LocalCPUBackend",4]}),
            serde_json::json!({"0":0}),
            false,
        )
        .await;
        assert_eq!(observed.metadata.worker_id, "custom-instance");
        assert!(observed.metadata.engine_epoch.is_empty());
        assert!(matches!(
            observed.evidence(),
            PrefixEvidence::LmCacheObserved {
                tokens: 4,
                cached_tokens: 4,
                engine_epoch: None,
                ..
            }
        ));
        for layout in [
            serde_json::json!({"custom-instance":["LocalCPUBackend",6]}),
            serde_json::json!({"custom-instance":["unknown-tier",4]}),
            serde_json::json!({"custom-instance":["LocalCPUBackend",-1]}),
            serde_json::json!({"custom-instance":["LocalCPUBackend",true]}),
            serde_json::json!({"custom-instance":["LocalCPUBackend",1.5]}),
        ] {
            assert!(matches!(
                observations(layout, serde_json::json!({"0":0}), true)
                    .await
                    .evidence,
                PrefixEvidence::Unknown
            ));
        }
    }
}
