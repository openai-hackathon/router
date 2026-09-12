use crate::routing_state::CandidateSnapshot;

pub fn choose(candidates: &[CandidateSnapshot]) -> Result<Option<usize>, &'static str> {
    let mut best: Option<&CandidateSnapshot> = None;
    for candidate in candidates {
        let score = candidate.ect_ms.ok_or("missing_cost_model")?;
        if !score.is_finite() || score < 0.0 {
            return Err("invalid_cost_prediction");
        }
        if best.is_none_or(|current| {
            score
                .total_cmp(&current.ect_ms.unwrap())
                .then(candidate.tie_rank.cmp(&current.tie_rank))
                .is_lt()
        }) {
            best = Some(candidate);
        }
    }
    Ok(best.map(|c| c.worker_index))
}
