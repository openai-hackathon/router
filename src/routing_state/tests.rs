use super::*;
use crate::{
    core::{BasicWorker, WorkerType},
    policies::ObservedPolicy,
};
use kv::{Block, KvEvent};
use telemetry::{KvDelta, KvSnapshot, WorkerInfo, WorkerTelemetry};

fn workers() -> Vec<Arc<dyn Worker>> {
    (0..3)
        .map(|i| {
            Arc::new(BasicWorker::new(
                format!("http://g{i}"),
                WorkerType::Regular,
            )) as Arc<dyn Worker>
        })
        .collect()
}
fn features() -> RequestFeatures {
    RequestFeatures {
        request_id: "test".into(),
        session_id: None,
        tokens: Some((0..9).collect()),
        fingerprint: Some("f".into()),
        model: Some("local".into()),
        output_limit: Some(10),
        fallback_reason: None,
    }
}
fn telemetry(i: usize, blocks: usize) -> WorkerTelemetry {
    let blocks = (0..blocks)
        .map(|n| Block {
            hash: format!("b{n}"),
            parent: (n > 0).then(|| format!("b{}", n - 1)),
            tokens: vec![2 * n as u32, 2 * n as u32 + 1],
        })
        .collect();
    WorkerTelemetry::snapshot(
        WorkerInfo {
            schema_version: 1,
            worker_id: format!("g{i}"),
            engine_epoch: "epoch".into(),
            fingerprint: "f".into(),
            model: "local".into(),
            block_size: 2,
            supported: true,
        },
        KvSnapshot {
            schema_version: 1,
            worker_id: format!("g{i}"),
            engine_epoch: "epoch".into(),
            sequence: Some(0),
            synced: true,
            blocks,
        },
        100,
    )
    .unwrap()
}
fn state() -> (Arc<SharedRoutingState>, Vec<Arc<dyn Worker>>) {
    let state = Arc::new(SharedRoutingState::new(
        config::RoutingConfig::default(),
        Arc::new(DispatchLedger::default()),
    ));
    let workers = workers();
    for i in 0..3 {
        state.install(workers[i].url(), telemetry(i, 3 - i));
    }
    (state, workers)
}
fn candidate(i: usize, h: usize, n: usize, ect: f64) -> CandidateSnapshot {
    CandidateSnapshot {
        worker_index: i,
        inflight: n,
        reusable_tokens: h,
        ect_ms: Some(ect),
        tie_rank: i,
        evidence: PrefixEvidence::Unknown,
    }
}

#[test]
fn three_policies_choose_three_different_workers() {
    let c = [
        candidate(0, 6144, 6, 3060.0),
        candidate(1, 4096, 3, 2860.0),
        candidate(2, 0, 1, 3500.0),
    ];
    assert_eq!(policies::prefix_max::choose(&c), Some(0));
    assert_eq!(policies::least_load_kv::choose(&c), Some(2));
    assert_eq!(policies::kv_batch_ect::choose(&c), Ok(Some(1)));
}
#[test]
fn continuity_eviction_clear_and_epoch() {
    let mut t = telemetry(0, 4);
    let tokens: Vec<_> = (0..9).collect();
    assert_eq!(t.index.reusable_tokens(&tokens, 2), 8);
    assert_eq!(t.index.reusable_tokens(&tokens[..8], 2), 6);
    Arc::make_mut(&mut t.index)
        .apply(
            &[KvEvent::BlockRemoved {
                hashes: vec!["b1".into()],
            }],
            2,
            100,
        )
        .unwrap();
    assert_eq!(t.index.reusable_tokens(&tokens, 2), 2);
    Arc::make_mut(&mut t.index)
        .apply(&[KvEvent::AllBlocksCleared], 2, 100)
        .unwrap();
    assert_eq!(t.index.reusable_tokens(&tokens, 2), 0);
    let (state, workers) = state();
    let mut restarted = telemetry(0, 0);
    restarted.info.engine_epoch = "new".into();
    state.install(workers[0].url(), restarted);
    assert_eq!(
        state.select(&workers, Ranking::PrefixMax, &features()),
        Some(1)
    );
}
#[test]
fn sequence_gap_and_replay() {
    let mut t = telemetry(0, 2);
    let delta = |seq, batches| KvDelta {
        schema_version: 1,
        worker_id: "g0".into(),
        engine_epoch: "epoch".into(),
        sequence: Some(seq),
        synced: true,
        batches,
    };
    assert_eq!(
        t.apply(
            delta(
                2,
                vec![kv::EventBatch {
                    sequence: 2,
                    events: vec![]
                }]
            ),
            100
        ),
        Err("event_gap")
    );
    assert!(!t.synced);
    t.apply(
        delta(
            2,
            vec![
                kv::EventBatch {
                    sequence: 1,
                    events: vec![KvEvent::AllBlocksCleared],
                },
                kv::EventBatch {
                    sequence: 2,
                    events: vec![],
                },
            ],
        ),
        100,
    )
    .unwrap();
    assert!(t.synced);
    assert_eq!(t.sequence, Some(2));
    assert_eq!(t.index.reusable_tokens(&(0..9).collect::<Vec<_>>(), 2), 0);
}
#[test]
fn missing_stale_or_unsupported_evidence_falls_back_for_all_workers() {
    let (state, workers) = state();
    workers[0].increment_load();
    assert_eq!(
        state.select(&workers, Ranking::PrefixMax, &features()),
        Some(0)
    );
    state.invalidate(workers[2].url());
    assert_eq!(
        state.select(&workers, Ranking::PrefixMax, &features()),
        Some(1)
    );
    state.remove(workers[1].url());
    assert_eq!(
        state.select(&workers, Ranking::PrefixMax, &features()),
        Some(1)
    );
    let unsupported = RequestFeatures::unsupported(&serde_json::json!({"model":"local"}), None);
    assert_eq!(
        state.select(&workers, Ranking::KvBatchEct, &unsupported),
        Some(1)
    );
}
#[test]
fn missing_cost_model_keeps_the_worker_as_least_load_candidate() {
    let (state, workers) = state();
    workers[0].increment_load();
    workers[1].increment_load();
    assert_eq!(
        state.select(&workers, Ranking::KvBatchEct, &features()),
        Some(2)
    );
}
#[test]
fn measured_cost_model_checks_inputs_and_domain() {
    let mut model = cost::CostModel {
        fingerprint: "f".into(),
        calibration_version: "unit-test-only".into(),
        prompt_range: [1, 8192],
        output_range: [1, 100],
        concurrency_range: [0, 10],
        output_prior: 10,
        prefill: [0.0, 1.0, 0.0],
        decode: [500.0, 0.0, 0.0],
        beta: 0.4,
        queue_ms: 0.0,
    };
    assert_eq!(
        model.predict("f", 1000, 600, None, 6),
        Ok((3060.0000000000005, 10))
    );
    assert_eq!(
        model.predict("f", 9999, 0, None, 0),
        Err("outside_calibration_range")
    );
    assert_eq!(
        model.predict("other", 1000, 0, None, 0),
        Err("incompatible_cost_model")
    );
    model.prefill[1] = f64::NAN;
    assert_eq!(
        model.predict("f", 1000, 0, None, 0),
        Err("invalid_cost_prediction")
    );
    assert_eq!(
        policies::kv_batch_ect::choose(&[candidate(0, 0, 0, f64::INFINITY)]),
        Err("invalid_cost_prediction")
    );
}
#[test]
fn session_home_requires_success_epoch_and_bounded_cost() {
    let (mut state, workers) = state();
    // Configure distinct measured-model fixtures (not production defaults).
    let config = &mut Arc::get_mut(&mut state).unwrap().config;
    for (i, cost) in [101.0, 100.0, 200.0].into_iter().enumerate() {
        config.cost_models.insert(
            format!("g{i}"),
            cost::CostModel {
                fingerprint: "f".into(),
                calibration_version: "test".into(),
                prompt_range: [1, 100],
                output_range: [1, 100],
                concurrency_range: [0, 100],
                output_prior: 10,
                prefill: [cost, 0.0, 0.0],
                decode: [0.0; 3],
                beta: 0.0,
                queue_ms: 0.0,
            },
        );
    }
    let mut f = features();
    f.session_id = Some("session".into());
    let mut r = state
        .reserve(
            &workers,
            Arc::new(ObservedPolicy(Ranking::PrefixMax)),
            &f,
            None,
            None,
        )
        .unwrap();
    assert_eq!(r.worker.url(), "http://g0");
    assert_eq!(state.select(&workers, Ranking::KvBatchEct, &f), Some(1));
    r.dispatched();
    r.finish(true);
    assert_eq!(state.select(&workers, Ranking::KvBatchEct, &f), Some(0));
    state
        .state
        .lock()
        .sessions
        .values_mut()
        .for_each(|h| h.epoch = "wrong".into());
    assert_eq!(state.select(&workers, Ranking::KvBatchEct, &f), Some(1));
}
#[tokio::test]
async fn feature_builder_does_not_render_responses_as_chat() {
    let f = features::build(
        &reqwest::Client::new(),
        Some("http://invalid"),
        "/v1/responses",
        &serde_json::json!({"model":"local","input":"hello"}),
        None,
    )
    .await;
    assert_eq!(f.fallback_reason, Some("unsupported_request"));
    assert!(f.tokens.is_none());
}
#[test]
fn factory_and_config_roundtrip() {
    for name in ["prefix_max", "least_load_kv", "kv_batch_ect"] {
        let config: crate::config::PolicyConfig =
            serde_json::from_value(serde_json::json!({"type":name})).unwrap();
        assert_eq!(config.name(), name);
        assert_eq!(
            policies::PolicyFactory::create_from_config(&config).name(),
            name
        );
        assert_eq!(
            policies::PolicyRegistry::new(config)
                .get_default_policy()
                .name(),
            name
        );
        assert!(policies::PolicyFactory::create_by_name(name)
            .unwrap()
            .ranking()
            .is_some());
    }
    assert!(serde_json::from_value::<crate::config::PolicyConfig>(
        serde_json::json!({"type":"prefix_mxa"})
    )
    .is_err());
}
