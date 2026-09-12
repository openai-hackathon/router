use crate::routing_state::CandidateSnapshot;
use std::cmp::Reverse;

pub fn choose(candidates: &[CandidateSnapshot]) -> Option<usize> {
    candidates
        .iter()
        .min_by_key(|c| (Reverse(c.reusable_tokens), c.inflight, c.tie_rank))
        .map(|c| c.worker_index)
}
