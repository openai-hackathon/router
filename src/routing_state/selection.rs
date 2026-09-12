//! Source-independent policy inputs. An adapter supplies observations; selection
//! performs no tokenization, network access, load mutation, or cache mutation.
use super::{config::RoutingConfig, features::RequestFeatures, PrefixEvidence, Ranking};
use crate::policies;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerMetadata {
    pub worker_id: String,
    pub model: String,
    pub fingerprint: String,
    pub engine_epoch: String,
    /// Granularity of reusable prefix observations, in tokens.
    pub block_size: usize,
}

impl WorkerMetadata {
    fn valid(&self) -> bool {
        !self.worker_id.is_empty()
            && !self.model.is_empty()
            && !self.fingerprint.is_empty()
            && !self.engine_epoch.is_empty()
            && self.block_size > 0
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSnapshot {
    pub worker_index: usize,
    pub worker_url: String,
    pub available: bool,
    /// Read immediately before this attempt's reservation, under the ledger lock.
    pub inflight: usize,
    pub metadata: Option<WorkerMetadata>,
    pub evidence: PrefixEvidence,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionHomeSnapshot {
    pub worker_url: String,
    pub engine_epoch: String,
    pub age_ms: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionSnapshot {
    pub workers: Vec<WorkerSnapshot>,
    /// Only a successfully completed request may establish a home.
    pub home: Option<SessionHomeSnapshot>,
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

#[derive(Clone, Debug, Serialize)]
pub struct RoutingDecision {
    pub chosen_worker: usize,
    pub fallback_reason: Option<&'static str>,
    pub affinity_applied: bool,
    pub candidates: Vec<CandidateSnapshot>,
}

impl SelectionSnapshot {
    pub fn decide(
        &self,
        ranking: Ranking,
        features: &RequestFeatures,
        config: &RoutingConfig,
    ) -> Option<RoutingDecision> {
        let mut ordered: Vec<_> = self.workers.iter().filter(|w| w.available).collect();
        ordered.sort_by(|a, b| a.worker_url.cmp(&b.worker_url));
        let mut fallback = features.fallback_reason;
        if features.tokens.as_ref().is_none_or(Vec::is_empty)
            || features.fingerprint.as_ref().is_none_or(String::is_empty)
        {
            fallback.get_or_insert("missing_request_features");
        }
        let mut candidates = Vec::new();
        let mut inputs = HashMap::new();
        let mut identities = HashSet::new();
        let mut indices = HashSet::new();
        let mut urls = HashSet::new();
        for (tie_rank, worker) in ordered.into_iter().enumerate() {
            // Duplicate indices/URLs make the dispatch target ambiguous, even
            // under least-load fallback. Refuse such malformed snapshots.
            if !indices.insert(worker.worker_index) || !urls.insert(&worker.worker_url) {
                return None;
            }
            let metadata = worker.metadata.as_ref();
            if let Some(meta) = metadata.filter(|m| m.valid()) {
                if features.model.as_deref().is_some_and(|m| m != meta.model)
                    || features
                        .fingerprint
                        .as_deref()
                        .is_some_and(|f| f != meta.fingerprint)
                {
                    continue;
                }
                if !identities.insert(&meta.worker_id) {
                    fallback.get_or_insert("ambiguous_worker_identity");
                }
            }
            let evidence = match &worker.evidence {
                PrefixEvidence::Observed { age_ms, .. } if *age_ms > config.max_evidence_age_ms => {
                    PrefixEvidence::Stale
                }
                other => other.clone(),
            };
            let reusable = match &evidence {
                PrefixEvidence::Observed {
                    tokens,
                    engine_epoch,
                    ..
                } => {
                    let valid = metadata.is_some_and(|meta| {
                        meta.valid()
                            && meta.engine_epoch == *engine_epoch
                            && tokens.is_multiple_of(meta.block_size)
                            && features.tokens.as_ref().is_some_and(|t| *tokens <= t.len())
                    });
                    if !valid {
                        fallback.get_or_insert("invalid_kv_evidence");
                        0
                    } else {
                        *tokens
                    }
                }
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
                worker_index: worker.worker_index,
                inflight: worker.inflight,
                reusable_tokens: reusable,
                ect_ms: None,
                tie_rank,
                evidence,
            });
            inputs.insert(worker.worker_index, worker);
        }
        if ranking == Ranking::KvBatchEct && fallback.is_none() {
            for candidate in &mut candidates {
                let meta = inputs[&candidate.worker_index].metadata.as_ref()?;
                let prediction = config
                    .cost_models
                    .get(&meta.worker_id)
                    .ok_or("missing_cost_model")
                    .and_then(|model| {
                        model.predict(
                            &meta.fingerprint,
                            features.tokens.as_ref().unwrap().len(),
                            candidate.reusable_tokens,
                            features.output_limit,
                            candidate.inflight,
                        )
                    });
                match prediction {
                    Ok((ect, _)) => candidate.ect_ms = Some(ect),
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
        if ranking == Ranking::KvBatchEct && fallback.is_none() && features.session_id.is_some() {
            if let Some(home) = self
                .home
                .as_ref()
                .filter(|home| home.age_ms < config.session_ttl_secs.saturating_mul(1000))
            {
                if let Some(candidate) = candidates
                    .iter()
                    .find(|c| inputs[&c.worker_index].worker_url == home.worker_url)
                {
                    let best = candidates.iter().find(|c| c.worker_index == chosen)?;
                    let meta = inputs[&candidate.worker_index].metadata.as_ref()?;
                    let best_ect = best.ect_ms.unwrap();
                    // Evaluate the small allowance separately: 1.015 * 2000
                    // rounds below 2030 and would reject an exact 2040 ms home.
                    let affinity_limit = best_ect + 0.015 * best_ect + 10.0;
                    if meta.engine_epoch == home.engine_epoch
                        && candidate.reusable_tokens
                            >= best.reusable_tokens.saturating_add(meta.block_size)
                        && candidate.ect_ms.unwrap() <= affinity_limit
                    {
                        chosen = candidate.worker_index;
                        affinity = true;
                    }
                }
            }
        }
        Some(RoutingDecision {
            chosen_worker: chosen,
            fallback_reason: fallback,
            affinity_applied: affinity,
            candidates,
        })
    }
}
