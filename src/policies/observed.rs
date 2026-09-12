use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::{core::Worker, routing_state::Ranking};
use std::sync::Arc;

/// The shared Router builds features, validates evidence, selects and reserves.
/// Calls through the legacy text-only trait can only promise least-load.
#[derive(Debug)]
pub struct ObservedPolicy(pub Ranking);

impl LoadBalancingPolicy for ObservedPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _: Option<&str>,
        _: Option<&RequestHeaders>,
    ) -> Option<usize> {
        get_healthy_worker_indices(workers)
            .into_iter()
            .min_by(|&a, &b| {
                workers[a]
                    .load()
                    .cmp(&workers[b].load())
                    .then(workers[a].url().cmp(workers[b].url()))
            })
    }
    fn ranking(&self) -> Option<Ranking> {
        Some(self.0)
    }
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
