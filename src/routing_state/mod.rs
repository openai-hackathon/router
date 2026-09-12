//! Shared request evidence, compatibility, KV index and session affinity.
//! DispatchLedger owns the selection/reservation lock for every policy.
pub mod config;
pub mod cost;
pub mod features;
pub mod kv;
pub mod lmcache;
pub mod selection;
pub mod telemetry;

use crate::{
    core::{
        dispatch::{DispatchLedger, Reservation},
        Worker,
    },
    policies::{LoadBalancingPolicy, RequestHeaders},
};
use features::RequestFeatures;
use parking_lot::Mutex;
pub use selection::{
    CandidateSnapshot, RoutingDecision, SelectionSnapshot, SessionHomeSnapshot, WorkerMetadata,
    WorkerSnapshot,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use telemetry::WorkerTelemetry;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ranking {
    PrefixMax,
    LeastLoadKv,
    KvBatchEct,
}
impl Ranking {
    pub fn name(self) -> &'static str {
        match self {
            Self::PrefixMax => "prefix_max",
            Self::LeastLoadKv => "least_load_kv",
            Self::KvBatchEct => "kv_batch_ect",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PrefixEvidence {
    Observed {
        tokens: usize,
        engine_epoch: String,
        sequence: Option<u64>,
        age_ms: u64,
    },
    /// Controller placement is restorable cache, not a GPU-resident KV claim.
    LmCacheObserved {
        tokens: usize,
        cached_tokens: usize,
        instance_id: String,
        location: String,
        engine_epoch: Option<String>,
        age_ms: u64,
    },
    Unknown,
    Stale,
    Unsupported,
}

#[derive(Clone, Debug)]
struct Home {
    worker_url: String,
    epoch: String,
    updated: Instant,
}
#[derive(Debug, Default)]
struct State {
    workers: HashMap<String, WorkerTelemetry>,
    controller_metadata: HashMap<String, WorkerMetadata>,
    controller_observed_at: HashMap<String, Instant>,
    sessions: HashMap<(String, String), Home>,
    identity_mismatches: HashMap<String, String>,
}

#[derive(Debug)]
pub struct SharedRoutingState {
    pub config: config::RoutingConfig,
    pub ledger: Arc<DispatchLedger>,
    state: Mutex<State>,
}

impl SharedRoutingState {
    pub fn new(config: config::RoutingConfig, ledger: Arc<DispatchLedger>) -> Self {
        Self {
            config,
            ledger,
            state: Mutex::new(State::default()),
        }
    }
    pub fn invalidate(&self, url: &str) {
        if let Some(worker) = self.state.lock().workers.get_mut(url) {
            worker.synced = false;
        }
    }
    pub fn remove(&self, url: &str) {
        self.state.lock().workers.remove(url);
    }
    pub fn install(&self, url: &str, mut telemetry: WorkerTelemetry) {
        // Never acquire the ledger lock while holding the evidence lock.
        self.ledger.observe_epoch(url, &telemetry.info.engine_epoch);
        let mut state = self.state.lock();
        if state.identity_mismatches.get(url) == Some(&telemetry.info.engine_epoch) {
            telemetry.synced = false;
        } else {
            state.identity_mismatches.remove(url);
        }
        state.workers.insert(url.to_owned(), telemetry);
    }
    pub fn renderer_url(&self) -> Option<String> {
        self.config.renderer_url.clone().or_else(|| {
            self.state
                .lock()
                .workers
                .iter()
                .filter(|(_, w)| {
                    w.synced
                        && w.info.supported
                        && w.observed_at.elapsed().as_millis()
                            <= self.config.max_evidence_age_ms as u128
                })
                .map(|(url, _)| url)
                .min()
                .map(|url| format!("{url}/routing/render"))
        })
    }

    /// Capture local evidence and pre-reservation loads under the ledger lock.
    /// Future data providers only need to produce this source-independent shape.
    pub fn snapshot(
        &self,
        workers: &[Arc<dyn Worker>],
        features: &RequestFeatures,
    ) -> SelectionSnapshot {
        let state = self.state.lock();
        let workers = workers
            .iter()
            .enumerate()
            .map(|(index, worker)| {
                let telemetry = state.workers.get(worker.url());
                let evidence = match telemetry {
                    None => PrefixEvidence::Unknown,
                    Some(t) if !t.info.supported || features.tokens.is_none() => {
                        PrefixEvidence::Unsupported
                    }
                    Some(t) if !t.synced => PrefixEvidence::Stale,
                    Some(t) => PrefixEvidence::Observed {
                        tokens: t
                            .index
                            .reusable_tokens(features.tokens.as_ref().unwrap(), t.info.block_size),
                        engine_epoch: t.info.engine_epoch.clone(),
                        sequence: t.sequence,
                        age_ms: t.observed_at.elapsed().as_millis().min(u64::MAX as u128) as u64,
                    },
                };
                WorkerSnapshot {
                    worker_index: index,
                    worker_url: worker.url().to_owned(),
                    available: worker.is_available(),
                    inflight: worker.load(),
                    metadata: telemetry.map(|t| WorkerMetadata {
                        worker_id: t.info.worker_id.clone(),
                        model: t.info.model.clone(),
                        fingerprint: t.info.fingerprint.clone(),
                        engine_epoch: t.info.engine_epoch.clone(),
                        block_size: t.info.block_size,
                    }),
                    evidence,
                }
            })
            .collect();
        let home = features
            .fingerprint
            .as_ref()
            .zip(features.session_id.as_ref())
            .and_then(|(fingerprint, session)| {
                state.sessions.get(&(fingerprint.clone(), session.clone()))
            })
            .map(|home| SessionHomeSnapshot {
                worker_url: home.worker_url.clone(),
                engine_epoch: home.epoch.clone(),
                age_ms: home.updated.elapsed().as_millis().min(u64::MAX as u128) as u64,
            });
        SelectionSnapshot { workers, home }
    }

    /// Called inside DispatchLedger's lock; no network or tokenization here.
    pub fn select(
        &self,
        workers: &[Arc<dyn Worker>],
        ranking: Ranking,
        features: &RequestFeatures,
    ) -> Option<usize> {
        self.select_with_observations(workers, ranking, features, None)
    }

    fn select_with_observations(
        &self,
        workers: &[Arc<dyn Worker>],
        ranking: Ranking,
        features: &RequestFeatures,
        observations: Option<&lmcache::Observations>,
    ) -> Option<usize> {
        let mut snapshot = self.snapshot(workers, features);
        if let Some(observations) = observations {
            let state = self.state.lock();
            for worker in &mut snapshot.workers {
                let observation = observations.get(&worker.worker_url);
                worker.metadata = observation.map(|o| o.metadata.clone());
                worker.evidence = observation.map_or(PrefixEvidence::Unknown, |o| {
                    if state.identity_mismatches.get(&worker.worker_url)
                        == Some(&o.metadata.engine_epoch)
                        || state
                            .controller_metadata
                            .get(&worker.worker_url)
                            .is_some_and(|m| m.engine_epoch != o.metadata.engine_epoch)
                    {
                        PrefixEvidence::Stale
                    } else {
                        o.evidence()
                    }
                });
            }
        }
        let decision = snapshot.decide(ranking, features, &self.config)?;
        metrics::counter!("router_routing_decisions_total", "policy" => ranking.name(), "fallback" => decision.fallback_reason.unwrap_or("none")).increment(1);
        tracing::info!(request_id = %features.request_id, policy = ranking.name(), fallback_reason = decision.fallback_reason,
            prompt_tokens = features.tokens.as_ref().map(Vec::len), output_limit = features.output_limit,
            candidates = %serde_json::to_string(&decision.candidates).unwrap_or_default(), chosen_worker = workers[decision.chosen_worker].url(), affinity = decision.affinity_applied, "routing decision");
        Some(decision.chosen_worker)
    }

    pub fn reserve(
        self: &Arc<Self>,
        workers: &[Arc<dyn Worker>],
        policy: Arc<dyn LoadBalancingPolicy>,
        features: &RequestFeatures,
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<Reservation> {
        self.reserve_with_observations(workers, policy, features, text, headers, None)
    }

    pub fn reserve_with_observations(
        self: &Arc<Self>,
        workers: &[Arc<dyn Worker>],
        policy: Arc<dyn LoadBalancingPolicy>,
        features: &RequestFeatures,
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
        observations: Option<&lmcache::Observations>,
    ) -> Option<Reservation> {
        if let Some(observations) = observations {
            for (url, observation) in observations {
                let meta = &observation.metadata;
                if !meta.engine_epoch.is_empty() {
                    // Concurrent HTTP lookups can arrive out of order. They
                    // cannot prove old inference attempts have terminated, so
                    // do not use them to release unknown ledger entries.
                    let mut state = self.state.lock();
                    if state
                        .controller_observed_at
                        .get(url)
                        .is_some_and(|time| *time >= observation.started)
                    {
                        continue;
                    }
                    if state.identity_mismatches.get(url) != Some(&meta.engine_epoch) {
                        state.identity_mismatches.remove(url);
                    }
                    state.controller_metadata.insert(url.clone(), meta.clone());
                    state
                        .controller_observed_at
                        .insert(url.clone(), observation.started);
                }
            }
        }
        let mut reservation = self
            .ledger
            .select_and_reserve(workers, policy.clone(), || match policy.ranking() {
                Some(ranking) => {
                    self.select_with_observations(workers, ranking, features, observations)
                }
                None => policy.select_worker_with_headers(workers, text, headers),
            })?;
        let epoch = observations
            .and_then(|o| o.get(reservation.worker.url()))
            .map(|o| o.metadata.engine_epoch.clone())
            .filter(|epoch| !epoch.is_empty())
            .or_else(|| {
                self.state
                    .lock()
                    .workers
                    .get(reservation.worker.url())
                    .map(|t| t.info.engine_epoch.clone())
            });
        if let Some(epoch) = epoch {
            reservation.set_epoch(epoch.clone());
            if let (Some(session), Some(fingerprint)) =
                (features.session_id.clone(), features.fingerprint.clone())
            {
                let weak = Arc::downgrade(self);
                let url = reservation.worker.url().to_owned();
                reservation.on_success(move || {
                    if let Some(shared) = weak.upgrade() {
                        let mut state = shared.state.lock();
                        let current = state
                            .workers
                            .get(&url)
                            .is_some_and(|w| w.info.engine_epoch == epoch && w.synced)
                            || state
                                .controller_metadata
                                .get(&url)
                                .is_some_and(|m| m.engine_epoch == epoch);
                        if !current || state.identity_mismatches.get(&url) == Some(&epoch) {
                            return;
                        }
                        state.sessions.retain(|_, home| {
                            home.updated.elapsed()
                                < Duration::from_secs(shared.config.session_ttl_secs)
                        });
                        if state.sessions.len() >= shared.config.max_sessions {
                            if let Some(oldest) = state
                                .sessions
                                .iter()
                                .min_by_key(|(_, home)| home.updated)
                                .map(|(key, _)| key.clone())
                            {
                                state.sessions.remove(&oldest);
                            }
                        }
                        state.sessions.insert(
                            (fingerprint, session),
                            Home {
                                worker_url: url,
                                epoch,
                                updated: Instant::now(),
                            },
                        );
                    }
                });
            }
        }
        Some(reservation)
    }

    pub fn validate_response_identity(&self, url: &str, headers: &http::HeaderMap) -> bool {
        let state = self.state.lock();
        let identity = state
            .workers
            .get(url)
            .map(|w| (&w.info.worker_id, &w.info.engine_epoch))
            .or_else(|| {
                state
                    .controller_metadata
                    .get(url)
                    .map(|m| (&m.worker_id, &m.engine_epoch))
            });
        let Some((worker_id, engine_epoch)) = identity else {
            return true;
        };
        let matches = headers
            .get("x-routing-worker-id")
            .and_then(|v| v.to_str().ok())
            == Some(worker_id.as_str())
            && headers
                .get("x-routing-engine-epoch")
                .and_then(|v| v.to_str().ok())
                == Some(engine_epoch.as_str());
        let epoch = engine_epoch.clone();
        drop(state);
        if !matches {
            self.invalidate(url);
            self.state
                .lock()
                .identity_mismatches
                .insert(url.to_owned(), epoch);
        }
        matches
    }
}

#[cfg(test)]
mod tests;
