//! Shared request evidence, compatibility, KV index and session affinity.
//! DispatchLedger owns the selection/reservation lock for every policy.
pub mod config;
pub mod cost;
pub mod features;
pub mod kv;
pub mod telemetry;

use crate::{
    core::{
        dispatch::{DispatchLedger, Reservation},
        Worker,
    },
    policies::{self, LoadBalancingPolicy, RequestHeaders},
};
use features::RequestFeatures;
use parking_lot::Mutex;
use serde::Serialize;
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

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PrefixEvidence {
    Observed {
        tokens: usize,
        engine_epoch: String,
        sequence: Option<u64>,
        age_ms: u64,
    },
    Unknown,
    Stale,
    Unsupported,
}

#[derive(Clone, Debug, Serialize)]
pub struct CandidateSnapshot {
    pub worker_index: usize,
    pub inflight: usize,
    pub reusable_tokens: usize,
    pub ect_ms: Option<f64>,
    pub tie_rank: usize,
    pub evidence: PrefixEvidence,
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

    /// Called inside DispatchLedger's lock; no network or tokenization here.
    pub fn select(
        &self,
        workers: &[Arc<dyn Worker>],
        ranking: Ranking,
        features: &RequestFeatures,
    ) -> Option<usize> {
        let state = self.state.lock();
        let mut ordered: Vec<_> = (0..workers.len()).collect();
        ordered.sort_by_key(|&i| workers[i].url());
        let mut fallback = features.fallback_reason;
        let mut candidates = Vec::new();
        for (tie_rank, index) in ordered.into_iter().enumerate() {
            let worker = &workers[index];
            if !worker.is_available() {
                continue;
            }
            let telemetry = state.workers.get(worker.url());
            if let Some(t) = telemetry {
                // Known incompatible serving configs are excluded; missing KV
                // evidence itself must never remove an otherwise usable worker.
                if features.model.as_deref().is_some_and(|m| m != t.info.model)
                    || features
                        .fingerprint
                        .as_deref()
                        .is_some_and(|f| f != t.info.fingerprint)
                {
                    continue;
                }
            }
            let evidence = match telemetry {
                None => PrefixEvidence::Unknown,
                Some(t) if !t.info.supported || features.tokens.is_none() => {
                    PrefixEvidence::Unsupported
                }
                Some(t)
                    if !t.synced
                        || t.observed_at.elapsed().as_millis()
                            > self.config.max_evidence_age_ms as u128 =>
                {
                    PrefixEvidence::Stale
                }
                Some(t) => PrefixEvidence::Observed {
                    tokens: t
                        .index
                        .reusable_tokens(features.tokens.as_ref().unwrap(), t.info.block_size),
                    engine_epoch: t.info.engine_epoch.clone(),
                    sequence: t.sequence,
                    age_ms: t.observed_at.elapsed().as_millis() as u64,
                },
            };
            let reusable = match &evidence {
                PrefixEvidence::Observed { tokens, .. } => *tokens,
                PrefixEvidence::Unknown => {
                    fallback.get_or_insert("unknown_kv");
                    0
                }
                PrefixEvidence::Stale => {
                    fallback.get_or_insert("stale_kv");
                    0
                }
                PrefixEvidence::Unsupported => {
                    fallback.get_or_insert("unsupported_kv");
                    0
                }
            };
            candidates.push(CandidateSnapshot {
                worker_index: index,
                inflight: worker.load(),
                reusable_tokens: reusable,
                ect_ms: None,
                tie_rank,
                evidence,
            });
        }
        if ranking == Ranking::KvBatchEct && fallback.is_none() {
            for c in &mut candidates {
                let info = &state.workers[workers[c.worker_index].url()].info;
                let result = self
                    .config
                    .cost_models
                    .get(&info.worker_id)
                    .ok_or("missing_cost_model")
                    .and_then(|model| {
                        model.predict(
                            &info.fingerprint,
                            features.tokens.as_ref().unwrap().len(),
                            c.reusable_tokens,
                            features.output_limit,
                            c.inflight,
                        )
                    });
                match result {
                    Ok((ect, _)) => c.ect_ms = Some(ect),
                    Err(reason) => {
                        fallback.get_or_insert(reason);
                    }
                }
            }
        }
        let mut chosen = if fallback.is_some() {
            candidates
                .iter()
                .min_by_key(|c| (c.inflight, c.tie_rank))
                .map(|c| c.worker_index)
        } else {
            match ranking {
                Ranking::PrefixMax => policies::prefix_max::choose(&candidates),
                Ranking::LeastLoadKv => policies::least_load_kv::choose(&candidates),
                Ranking::KvBatchEct => policies::kv_batch_ect::choose(&candidates).ok().flatten(),
            }
        }?;
        let mut affinity = false;
        if ranking == Ranking::KvBatchEct && fallback.is_none() {
            if let (Some(session), Some(fingerprint)) =
                (&features.session_id, &features.fingerprint)
            {
                if let Some(home) = state.sessions.get(&(fingerprint.clone(), session.clone())) {
                    if home.updated.elapsed() < Duration::from_secs(self.config.session_ttl_secs) {
                        if let Some(candidate) = candidates
                            .iter()
                            .find(|c| workers[c.worker_index].url() == home.worker_url)
                        {
                            let best = candidates
                                .iter()
                                .find(|c| c.worker_index == chosen)
                                .unwrap();
                            let info = &state.workers[&home.worker_url].info;
                            if info.engine_epoch == home.epoch
                                && candidate.reusable_tokens
                                    >= best.reusable_tokens.saturating_add(info.block_size)
                                && candidate.ect_ms.unwrap() <= 1.015 * best.ect_ms.unwrap() + 10.0
                            {
                                chosen = candidate.worker_index;
                                affinity = true;
                            }
                        }
                    }
                }
            }
        }
        metrics::counter!("router_routing_decisions_total", "policy" => ranking.name(), "fallback" => fallback.unwrap_or("none")).increment(1);
        tracing::info!(request_id = %features.request_id, policy = ranking.name(), fallback_reason = fallback,
            prompt_tokens = features.tokens.as_ref().map(Vec::len), output_limit = features.output_limit,
            candidates = %serde_json::to_string(&candidates).unwrap_or_default(), chosen_worker = workers[chosen].url(), affinity, "routing decision");
        Some(chosen)
    }

    pub fn reserve(
        self: &Arc<Self>,
        workers: &[Arc<dyn Worker>],
        policy: Arc<dyn LoadBalancingPolicy>,
        features: &RequestFeatures,
        text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<Reservation> {
        let mut reservation = self
            .ledger
            .select_and_reserve(workers, policy.clone(), || match policy.ranking() {
                Some(ranking) => self.select(workers, ranking, features),
                None => policy.select_worker_with_headers(workers, text, headers),
            })?;
        let epoch = self
            .state
            .lock()
            .workers
            .get(reservation.worker.url())
            .map(|t| t.info.engine_epoch.clone());
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
                        if state
                            .workers
                            .get(&url)
                            .is_none_or(|w| w.info.engine_epoch != epoch || !w.synced)
                        {
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
        let Some(worker) = state.workers.get(url) else {
            return true;
        };
        let matches = headers
            .get("x-routing-worker-id")
            .and_then(|v| v.to_str().ok())
            == Some(worker.info.worker_id.as_str())
            && headers
                .get("x-routing-engine-epoch")
                .and_then(|v| v.to_str().ok())
                == Some(worker.info.engine_epoch.as_str());
        let epoch = worker.info.engine_epoch.clone();
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
