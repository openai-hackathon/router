//! Source-independent policy inputs. An adapter supplies observations; selection
//! performs no tokenization, network access, load mutation, or cache mutation.
use super::{
    backend_load::BackendLoadSnapshot,
    completion::CompletionEstimate,
    config::{EctModel, RoutingConfig},
    cost::CostEstimate,
    features::RequestFeatures,
    PrefixEvidence, Ranking,
};
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
    #[serde(default)]
    pub backend_load: Option<BackendLoadSnapshot>,
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
    pub cost: Option<CostEstimate>,
    pub completion_cost: Option<CompletionEstimate>,
    pub backend_load: Option<BackendLoadSnapshot>,
    pub tie_rank: usize,
    pub evidence: PrefixEvidence,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoutingDecision {
    pub identity_mode: &'static str,
    pub ect_model: &'static str,
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
        let endpoint_identity = config.endpoint_identity();
        let mut ordered: Vec<_> = self.workers.iter().filter(|w| w.available).collect();
        ordered.sort_by(|a, b| a.worker_url.cmp(&b.worker_url));
        let mut fallback = features.fallback_reason;
        if features.tokens.as_ref().is_none_or(Vec::is_empty)
            || (!endpoint_identity && features.fingerprint.as_ref().is_none_or(String::is_empty))
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
            if let Some(meta) = metadata {
                if (!meta.model.is_empty()
                    && features.model.as_deref().is_some_and(|m| m != meta.model))
                    || (!endpoint_identity
                        && !meta.fingerprint.is_empty()
                        && features
                            .fingerprint
                            .as_deref()
                            .is_some_and(|f| f != meta.fingerprint))
                {
                    continue;
                }
                if (meta.valid() || (endpoint_identity && !meta.worker_id.is_empty()))
                    && !identities.insert(&meta.worker_id)
                {
                    fallback.get_or_insert("ambiguous_worker_identity");
                }
            }
            let evidence = match &worker.evidence {
                PrefixEvidence::Observed { age_ms, .. }
                | PrefixEvidence::LmCacheObserved { age_ms, .. }
                    if *age_ms > config.max_evidence_age_ms =>
                {
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
                            && features.fingerprint.as_deref() == Some(meta.fingerprint.as_str())
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
                PrefixEvidence::LmCacheObserved {
                    tokens,
                    cached_tokens,
                    instance_id,
                    location,
                    engine_epoch,
                    ..
                } => {
                    let valid = metadata.is_some_and(|meta| {
                        let identity_valid = if endpoint_identity {
                            config.lmcache.as_ref().is_some_and(|c| {
                                meta.model == c.model
                                    && c.worker(&worker.worker_url).is_some_and(|w| {
                                        w.instance_id == meta.worker_id
                                            && w.block_size == meta.block_size
                                            && meta.block_size > 0
                                    })
                            })
                        } else {
                            meta.valid()
                                && engine_epoch.as_deref() == Some(meta.engine_epoch.as_str())
                        };
                        identity_valid
                            && instance_id == &meta.worker_id
                            && location == "LocalCPUBackend"
                            && tokens.is_multiple_of(meta.block_size)
                            && tokens <= cached_tokens
                            && features
                                .tokens
                                .as_ref()
                                .is_some_and(|t| *tokens < t.len() && *cached_tokens <= t.len())
                    });
                    if !valid {
                        fallback.get_or_insert(if !endpoint_identity && engine_epoch.is_none() {
                            "unverified_lmcache_identity"
                        } else {
                            "invalid_lmcache_evidence"
                        });
                        0
                    } else {
                        *tokens
                    }
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
                cost: None,
                completion_cost: None,
                backend_load: worker.backend_load.clone(),
                tie_rank,
                evidence,
            });
            inputs.insert(worker.worker_index, worker);
        }
        if ranking == Ranking::KvBatchEct && fallback.is_none() {
            for candidate in &mut candidates {
                let meta = inputs[&candidate.worker_index].metadata.as_ref()?;
                if config.ect_model == EctModel::CompletionTime {
                    let prediction = (|| {
                        if features.num_choices != 1 {
                            return Err("unsupported_completion_choices");
                        }
                        let model = config
                            .completion_models
                            .get(&meta.worker_id)
                            .ok_or("missing_completion_model")?;
                        // This model is calibrated on controller CPU inventory,
                        // not a mix of native GPU evidence and CPU placement.
                        let PrefixEvidence::LmCacheObserved { cached_tokens, .. } =
                            &candidate.evidence
                        else {
                            return Err("unsupported_completion_evidence");
                        };
                        let metrics = config
                            .backend_metrics
                            .as_ref()
                            .ok_or("missing_backend_metrics")?;
                        if metrics.model != meta.model
                            || !metrics.urls.keys().any(|url| {
                                url.trim_end_matches('/')
                                    == inputs[&candidate.worker_index]
                                        .worker_url
                                        .trim_end_matches('/')
                            })
                        {
                            return Err("invalid_backend_metrics_binding");
                        }
                        let backend = candidate
                            .backend_load
                            .as_ref()
                            .ok_or("missing_backend_metrics")?;
                        if backend.age_ms > metrics.max_age_ms {
                            return Err("stale_backend_metrics");
                        }
                        model.estimate(
                            if endpoint_identity {
                                &model.fingerprint
                            } else {
                                &meta.fingerprint
                            },
                            features.tokens.as_ref().unwrap().len(),
                            *cached_tokens,
                            features.output_limit,
                            candidate.inflight,
                            backend.running,
                            backend.waiting,
                            backend.kv_usage_fraction,
                        )
                    })();
                    match prediction {
                        Ok(estimate) => {
                            candidate.ect_ms = Some(estimate.ect_ms);
                            candidate.completion_cost = Some(estimate);
                        }
                        Err(reason) => {
                            fallback.get_or_insert(reason);
                        }
                    }
                    continue;
                }
                let prediction = config
                    .cost_models
                    .get(&meta.worker_id)
                    .ok_or("missing_cost_model")
                    .and_then(|model| {
                        model.estimate(
                            // Endpoint mode assigns each cost model by configured
                            // instance; no serving fingerprint is claimed verified.
                            if endpoint_identity {
                                &model.fingerprint
                            } else {
                                &meta.fingerprint
                            },
                            features.tokens.as_ref().unwrap().len(),
                            candidate.reusable_tokens,
                            features.output_limit,
                            candidate.inflight,
                        )
                    })
                    .and_then(|estimate| {
                        if let PrefixEvidence::LmCacheObserved {
                            cached_tokens,
                            location,
                            ..
                        } = &candidate.evidence
                        {
                            if *cached_tokens > 0 {
                                let restore = config
                                    .restore_models
                                    .get(&meta.worker_id)
                                    .ok_or("missing_restore_model")?;
                                return restore.apply(
                                    if endpoint_identity {
                                        &restore.fingerprint
                                    } else {
                                        &meta.fingerprint
                                    },
                                    location,
                                    *cached_tokens,
                                    estimate,
                                );
                            }
                        }
                        Ok(estimate)
                    });
                match prediction {
                    Ok(estimate) => {
                        candidate.ect_ms = Some(estimate.ect_ms);
                        candidate.cost = Some(estimate);
                    }
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
                    if (endpoint_identity || meta.engine_epoch == home.engine_epoch)
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
            identity_mode: config.identity_mode_name(),
            ect_model: config.ect_model.name(),
            chosen_worker: chosen,
            fallback_reason: fallback,
            affinity_applied: affinity,
            candidates,
        })
    }
}
