use super::{
    config::RoutingConfig,
    kv::{Block, EventBatch, KvIndex},
    SharedRoutingState,
};
use crate::core::WorkerRegistry;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct WorkerInfo {
    pub schema_version: u32,
    pub worker_id: String,
    pub engine_epoch: String,
    pub fingerprint: String,
    pub model: String,
    pub block_size: usize,
    pub supported: bool,
}

impl WorkerInfo {
    fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != 1
            || self.worker_id.is_empty()
            || self.engine_epoch.is_empty()
            || self.fingerprint.is_empty()
            || self.model.is_empty()
            || self.block_size == 0
        {
            return Err("invalid_worker_info");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RuntimeMetrics {
    pub running: usize,
    pub waiting: usize,
    pub kv_usage_fraction: f64,
    pub sample_age_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RuntimeState {
    pub schema_version: u32,
    pub worker_id: String,
    pub engine_epoch: String,
    pub engine_ready: bool,
    pub metrics: Option<RuntimeMetrics>,
    pub last_contiguous_sequence: Option<u64>,
    pub synced: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KvSnapshot {
    pub schema_version: u32,
    pub worker_id: String,
    pub engine_epoch: String,
    pub sequence: Option<u64>,
    pub synced: bool,
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KvDelta {
    pub schema_version: u32,
    pub worker_id: String,
    pub engine_epoch: String,
    pub sequence: Option<u64>,
    pub synced: bool,
    pub batches: Vec<EventBatch>,
}

#[derive(Clone, Debug)]
pub struct WorkerTelemetry {
    pub info: WorkerInfo,
    pub index: Arc<KvIndex>,
    pub sequence: Option<u64>,
    pub synced: bool,
    pub observed_at: Instant,
}

impl WorkerTelemetry {
    pub fn snapshot(
        info: WorkerInfo,
        snapshot: KvSnapshot,
        limit: usize,
    ) -> Result<Self, &'static str> {
        info.validate()?;
        if snapshot.schema_version != 1
            || snapshot.worker_id != info.worker_id
            || snapshot.engine_epoch != info.engine_epoch
            || !snapshot.synced
        {
            return Err("invalid_snapshot");
        }
        let index = KvIndex::from_blocks(snapshot.blocks, info.block_size, limit)?;
        Ok(Self {
            info,
            index: Arc::new(index),
            sequence: snapshot.sequence,
            synced: true,
            observed_at: Instant::now(),
        })
    }

    pub fn apply(&mut self, delta: KvDelta, limit: usize) -> Result<(), &'static str> {
        // Never trust partially applied data if anything below fails.
        self.synced = false;
        if delta.schema_version != 1
            || delta.worker_id != self.info.worker_id
            || delta.engine_epoch != self.info.engine_epoch
            || !delta.synced
        {
            return Err("epoch_or_stream_mismatch");
        }
        for batch in delta.batches {
            if self.sequence.is_some_and(|seq| batch.sequence <= seq) {
                continue;
            }
            let expected = match self.sequence {
                Some(seq) => seq.checked_add(1).ok_or("sequence_overflow")?,
                None => 0,
            };
            if batch.sequence != expected {
                return Err("event_gap");
            }
            Arc::make_mut(&mut self.index).apply(&batch.events, self.info.block_size, limit)?;
            self.sequence = Some(batch.sequence);
        }
        if self.sequence != delta.sequence {
            return Err("event_gap");
        }
        self.synced = true;
        self.observed_at = Instant::now();
        Ok(())
    }
}

async fn get<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
) -> Result<T, String> {
    // Avoid propagating reqwest's URL/response body into logs (may contain tokens).
    client
        .get(url)
        .send()
        .await
        .map_err(|_| "telemetry_transport")?
        .error_for_status()
        .map_err(|_| "telemetry_status")?
        .json()
        .await
        .map_err(|_| "telemetry_schema".into())
}

async fn collect(
    url: String,
    client: reqwest::Client,
    state: Weak<SharedRoutingState>,
    config: RoutingConfig,
) {
    let mut local: Option<WorkerTelemetry> = None;
    let mut interval = tokio::time::interval(Duration::from_millis(config.poll_interval_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        let Some(shared) = state.upgrade() else {
            return;
        };
        let result: Result<(), String> = async {
            let runtime: RuntimeState = get(&client, &format!("{url}/routing/state")).await?;
            if runtime.schema_version != 1 || !runtime.engine_ready {
                return Err("engine_not_ready".into());
            }
            if local.as_ref().is_none_or(|t| {
                t.info.engine_epoch != runtime.engine_epoch
                    || t.info.worker_id != runtime.worker_id
                    || !t.synced
            }) {
                shared.invalidate(&url);
                let info: WorkerInfo = get(&client, &format!("{url}/routing/info")).await?;
                if info.worker_id != runtime.worker_id || info.engine_epoch != runtime.engine_epoch
                {
                    return Err("unstable_engine_identity".into());
                }
                let snapshot: KvSnapshot =
                    get(&client, &format!("{url}/routing/kv-snapshot")).await?;
                local = Some(WorkerTelemetry::snapshot(
                    info,
                    snapshot,
                    config.max_blocks_per_worker,
                )?);
            } else {
                let current = local.as_mut().unwrap();
                let after = current.sequence.map_or("-1".into(), |seq| seq.to_string());
                let mut event_url = url::Url::parse(&format!("{url}/routing/kv-events"))
                    .map_err(|_| "invalid_url")?;
                event_url
                    .query_pairs_mut()
                    .append_pair("epoch", &current.info.engine_epoch)
                    .append_pair("after", &after);
                let delta: KvDelta = get(&client, event_url.as_str()).await?;
                current.apply(delta, config.max_blocks_per_worker)?;
            }
            let current = local.as_mut().unwrap();
            if !runtime.synced || runtime.last_contiguous_sequence > current.sequence {
                return Err("event_gap".into());
            }
            current.observed_at = Instant::now();
            shared.install(&url, current.clone());
            if let Some(runtime) = runtime.metrics {
                if runtime.sample_age_ms <= config.max_evidence_age_ms
                    && runtime.kv_usage_fraction.is_finite()
                    && (0.0..=1.0).contains(&runtime.kv_usage_fraction)
                {
                    metrics::gauge!("router_backend_running", "worker" => url.clone())
                        .set(runtime.running as f64);
                    metrics::gauge!("router_backend_waiting", "worker" => url.clone())
                        .set(runtime.waiting as f64);
                    metrics::gauge!("router_backend_kv_usage_fraction", "worker" => url.clone())
                        .set(runtime.kv_usage_fraction);
                }
            }
            Ok(())
        }
        .await;
        if let Err(reason) = result {
            shared.invalidate(&url);
            local = None;
            metrics::counter!("router_telemetry_errors_total", "reason" => reason).increment(1);
            // Avoid polling an absent bridge four times a second forever.
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
}

#[derive(Debug)]
pub struct Collectors(tokio::task::JoinHandle<()>);
impl Drop for Collectors {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl Collectors {
    pub fn start(
        registry: Arc<WorkerRegistry>,
        state: &Arc<SharedRoutingState>,
        client: reqwest::Client,
    ) -> Self {
        let state = Arc::downgrade(state);
        Self(tokio::spawn(async move {
            let mut tasks = tokio::task::JoinSet::new();
            let mut active = HashMap::<String, tokio::task::AbortHandle>::new();
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                let Some(shared) = state.upgrade() else {
                    break;
                };
                let urls: HashSet<_> = registry.get_all_urls().into_iter().collect();
                active.retain(|url, task| {
                    if !urls.contains(url) {
                        task.abort();
                        shared.remove(url);
                        false
                    } else {
                        true
                    }
                });
                for url in urls {
                    active.entry(url.clone()).or_insert_with(|| {
                        tasks.spawn(collect(
                            url.trim_end_matches('/').to_string(),
                            client.clone(),
                            state.clone(),
                            shared.config.clone(),
                        ))
                    });
                }
                while tasks.try_join_next().is_some() {}
            }
        }))
    }
}
