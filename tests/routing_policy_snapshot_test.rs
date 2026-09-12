//! Acceptance tests for policy behavior without a controller or GPU.
use std::sync::{Arc, Barrier};
use vllm_router_rs::{
    core::{dispatch::DispatchLedger, BasicWorker, Worker, WorkerType},
    policies::ObservedPolicy,
    routing_state::{
        config::RoutingConfig, features::RequestFeatures, PrefixEvidence, Ranking,
        SelectionSnapshot, SessionHomeSnapshot,
    },
};

const RANKINGS: [Ranking; 3] = [
    Ranking::PrefixMax,
    Ranking::LeastLoadKv,
    Ranking::KvBatchEct,
];

fn scenario() -> (SelectionSnapshot, RequestFeatures, RoutingConfig) {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/routing_policy_scenario.json")).unwrap();
    let mut features = RequestFeatures::unsupported(&data["request"], None);
    // Policy scoring only needs length; this fixture does not claim token parity.
    features.tokens = Some(vec![
        0;
        data["request"]["prompt_tokens"].as_u64().unwrap()
            as usize
    ]);
    features.fingerprint = data["request"]["fingerprint"].as_str().map(str::to_owned);
    features.output_limit = data["request"]["output_limit"].as_u64().map(|v| v as usize);
    features.fallback_reason = None;
    (
        serde_json::from_value(data["snapshot"].clone()).unwrap(),
        features,
        serde_json::from_value(data["routing_config"].clone()).unwrap(),
    )
}

fn prefix(snapshot: &mut SelectionSnapshot, i: usize, value: usize) {
    if let PrefixEvidence::Observed { tokens, .. } = &mut snapshot.workers[i].evidence {
        *tokens = value;
    } else {
        panic!("test requires observed evidence");
    }
}

fn controller_evidence(snapshot: &mut SelectionSnapshot) {
    for worker in &mut snapshot.workers {
        let PrefixEvidence::Observed { tokens, .. } = worker.evidence else {
            panic!()
        };
        let meta = worker.metadata.as_ref().unwrap();
        worker.evidence = PrefixEvidence::LmCacheObserved {
            tokens,
            cached_tokens: tokens,
            instance_id: meta.worker_id.clone(),
            location: "LocalCPUBackend".into(),
            engine_epoch: Some(meta.engine_epoch.clone()),
            age_ms: 0,
        };
    }
}

#[test]
fn controller_cache_requires_identity_and_separate_restore_cost() {
    use vllm_router_rs::routing_state::cost::RestoreCostModel;
    let (mut snapshot, features, mut config) = scenario();
    controller_evidence(&mut snapshot);
    assert_eq!(
        snapshot
            .decide(Ranking::PrefixMax, &features, &config)
            .unwrap()
            .chosen_worker,
        0
    );
    let fallback = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert_eq!(fallback.fallback_reason, Some("missing_restore_model"));
    assert_eq!(fallback.candidates.len(), 3);
    for worker in &snapshot.workers {
        let meta = worker.metadata.as_ref().unwrap();
        config.restore_models.insert(
            meta.worker_id.clone(),
            RestoreCostModel {
                fingerprint: meta.fingerprint.clone(),
                calibration_version: "synthetic-restore-only".into(),
                location: "LocalCPUBackend".into(),
                token_range: [1, 8192],
                fixed_ms: 1000.0,
                per_token_ms: 0.0,
            },
        );
    }
    let decision = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert_eq!(decision.fallback_reason, None);
    // CPU restore changes the winner; the uncached worker is now fastest.
    assert_eq!(decision.chosen_worker, 2);
    for (candidate, expected) in decision.candidates.iter().zip([6460.0, 5060.0, 3500.0]) {
        assert!((candidate.ect_ms.unwrap() - expected).abs() < 1e-8);
    }
    if let PrefixEvidence::LmCacheObserved { engine_epoch, .. } = &mut snapshot.workers[0].evidence
    {
        *engine_epoch = None;
    }
    for ranking in RANKINGS {
        let decision = snapshot.decide(ranking, &features, &config).unwrap();
        assert_eq!(
            decision.fallback_reason,
            Some("unverified_lmcache_identity")
        );
        assert_eq!(decision.chosen_worker, 2);
    }
}

#[test]
fn controller_restore_rejects_invalid_models_and_partial_metadata_still_filters_compatibility() {
    use vllm_router_rs::routing_state::cost::RestoreCostModel;
    let (mut snapshot, features, mut config) = scenario();
    controller_evidence(&mut snapshot);
    for worker in &snapshot.workers {
        let meta = worker.metadata.as_ref().unwrap();
        config.restore_models.insert(
            meta.worker_id.clone(),
            RestoreCostModel {
                fingerprint: meta.fingerprint.clone(),
                calibration_version: "test-only".into(),
                location: "LocalCPUBackend".into(),
                token_range: [1, 8192],
                fixed_ms: 1.0,
                per_token_ms: 0.0,
            },
        );
    }
    let id = snapshot.workers[0]
        .metadata
        .as_ref()
        .unwrap()
        .worker_id
        .clone();
    for kind in ["nan", "negative", "fingerprint", "location", "range"] {
        let mut config = config.clone();
        let model = config.restore_models.get_mut(&id).unwrap();
        match kind {
            "nan" => model.fixed_ms = f64::NAN,
            "negative" => model.per_token_ms = -1.0,
            "fingerprint" => model.fingerprint = "different".into(),
            "location" => model.location = "remote".into(),
            _ => model.token_range = [1, 10],
        }
        let result = snapshot
            .decide(Ranking::KvBatchEct, &features, &config)
            .unwrap();
        assert!(result.fallback_reason.is_some(), "{kind}");
        assert_eq!(result.candidates.len(), 3);
    }
    let meta = snapshot.workers[2].metadata.as_mut().unwrap();
    meta.fingerprint = "different".into();
    meta.engine_epoch.clear();
    let decision = snapshot
        .decide(Ranking::PrefixMax, &features, &config)
        .unwrap();
    assert_eq!(decision.fallback_reason, None);
    assert_eq!(decision.candidates.len(), 2);
}

#[test]
fn original_scenario_computes_costs_and_selects_three_different_workers() {
    let (snapshot, features, config) = scenario();
    for (ranking, expected) in RANKINGS.into_iter().zip([0, 2, 1]) {
        let decision = snapshot.decide(ranking, &features, &config).unwrap();
        assert_eq!(decision.chosen_worker, expected);
        assert_eq!(decision.fallback_reason, None);
        assert!(!decision.affinity_applied);
        if ranking == Ranking::KvBatchEct {
            for (candidate, expected_ms) in decision.candidates.iter().zip([3060.0, 2860.0, 3500.0])
            {
                assert!((candidate.ect_ms.unwrap() - expected_ms).abs() < 1e-9);
                let estimate = candidate.cost.as_ref().unwrap();
                assert_eq!(estimate.ect_ms, candidate.ect_ms.unwrap());
                assert_eq!(estimate.uncached_tokens, 8192 - candidate.reusable_tokens);
                assert_eq!(estimate.estimated_output_tokens, 128);
                assert_eq!(estimate.decode_ms, 500.0);
                assert_eq!(estimate.queue_ms, 0.0);
                assert_eq!(estimate.calibration_version, "synthetic-test-only");
            }
        } else {
            assert!(decision.candidates.iter().all(|c| c.ect_ms.is_none()));
        }
    }
}

#[test]
fn baselines_preserve_lexicographic_priority_and_ignore_cost_models() {
    let (mut snapshot, features, mut config) = scenario();
    snapshot.workers.truncate(2);
    config.cost_models.clear();
    for left_h in [0, 16, 32] {
        for right_h in [0, 16, 32] {
            for left_n in 0..3 {
                for right_n in 0..3 {
                    prefix(&mut snapshot, 0, left_h);
                    prefix(&mut snapshot, 1, right_h);
                    snapshot.workers[0].inflight = left_n;
                    snapshot.workers[1].inflight = right_n;
                    let prefix_expected = usize::from(
                        (right_h, std::cmp::Reverse(right_n)) > (left_h, std::cmp::Reverse(left_n)),
                    );
                    let load_expected = usize::from(
                        (right_n, std::cmp::Reverse(right_h)) < (left_n, std::cmp::Reverse(left_h)),
                    );
                    for (ranking, expected) in [
                        (Ranking::PrefixMax, prefix_expected),
                        (Ranking::LeastLoadKv, load_expected),
                    ] {
                        let result = snapshot.decide(ranking, &features, &config).unwrap();
                        assert_eq!(result.chosen_worker, expected);
                        assert_eq!(result.fallback_reason, None);
                    }
                }
            }
        }
    }
}

#[test]
fn missing_stale_and_unsupported_candidates_force_common_least_load() {
    for ranking in RANKINGS {
        for (evidence, reason) in [
            (PrefixEvidence::Unknown, "unknown_kv"),
            (PrefixEvidence::Stale, "stale_kv"),
            (PrefixEvidence::Unsupported, "unsupported_kv"),
        ] {
            let (mut snapshot, features, config) = scenario();
            snapshot.workers[2].evidence = evidence;
            let result = snapshot.decide(ranking, &features, &config).unwrap();
            assert_eq!(result.chosen_worker, 2);
            assert_eq!(result.candidates.len(), 3);
            assert_eq!(result.fallback_reason, Some(reason));
            assert!(!result.affinity_applied);
        }
    }
}

#[test]
fn age_and_invalid_prefix_evidence_do_not_masquerade_as_cache_misses() {
    let (snapshot, features, config) = scenario();
    let mut bad_cases = Vec::new();
    let mut stale = snapshot.clone();
    if let PrefixEvidence::Observed { age_ms, .. } = &mut stale.workers[2].evidence {
        *age_ms = config.max_evidence_age_ms + 1;
    }
    bad_cases.push((stale, "stale_kv"));
    let mut overlong = snapshot.clone();
    prefix(&mut overlong, 2, 16384);
    bad_cases.push((overlong, "invalid_kv_evidence"));
    let mut partial_block = snapshot.clone();
    prefix(&mut partial_block, 2, 17);
    bad_cases.push((partial_block, "invalid_kv_evidence"));
    let mut epoch = snapshot.clone();
    epoch.workers[2].metadata.as_mut().unwrap().engine_epoch = "new".into();
    bad_cases.push((epoch, "invalid_kv_evidence"));
    let mut no_metadata = snapshot.clone();
    no_metadata.workers[2].metadata = None;
    bad_cases.push((no_metadata, "invalid_kv_evidence"));
    let mut zero_block = snapshot.clone();
    zero_block.workers[2].metadata.as_mut().unwrap().block_size = 0;
    bad_cases.push((zero_block, "invalid_kv_evidence"));
    for (snapshot, reason) in bad_cases {
        for ranking in RANKINGS {
            let result = snapshot.decide(ranking, &features, &config).unwrap();
            assert_eq!(result.chosen_worker, 2);
            assert_eq!(result.fallback_reason, Some(reason));
        }
    }
}

#[test]
fn unavailable_or_incompatible_workers_are_filtered_before_fallback() {
    let (mut snapshot, features, mut config) = scenario();
    snapshot.workers[2].evidence = PrefixEvidence::Unknown;
    snapshot.workers[2].available = false;
    config.cost_models.remove("g2");
    let result = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert_eq!(result.chosen_worker, 1);
    assert_eq!(result.fallback_reason, None);
    assert_eq!(result.candidates.len(), 2);
    snapshot.workers[2].available = true;
    snapshot.workers[2].metadata.as_mut().unwrap().fingerprint = "different-serving-config".into();
    assert_eq!(
        snapshot
            .decide(Ranking::KvBatchEct, &features, &config)
            .unwrap()
            .fallback_reason,
        None
    );
    snapshot
        .workers
        .iter_mut()
        .for_each(|w| w.available = false);
    assert!(snapshot
        .decide(Ranking::PrefixMax, &features, &config)
        .is_none());
}

#[test]
fn cost_failures_keep_all_candidates_and_do_not_affect_baselines() {
    for kind in [
        "missing",
        "nan",
        "infinite",
        "negative",
        "domain",
        "fingerprint",
        "prior",
    ] {
        let (snapshot, features, mut config) = scenario();
        let model = config.cost_models.get_mut("g2").unwrap();
        match kind {
            "missing" => {
                config.cost_models.remove("g2");
            }
            "nan" => model.beta = f64::NAN,
            "infinite" => model.decode[0] = f64::INFINITY,
            "negative" => model.queue_ms = -1.0,
            "domain" => model.concurrency_range = [2, 10],
            "fingerprint" => model.fingerprint = "wrong".into(),
            "prior" => model.output_prior = 0,
            _ => unreachable!(),
        }
        let result = snapshot
            .decide(Ranking::KvBatchEct, &features, &config)
            .unwrap();
        assert_eq!(result.chosen_worker, 2, "{kind}");
        assert!(result.fallback_reason.is_some(), "{kind}");
        assert_eq!(result.candidates.len(), 3);
        for ranking in [Ranking::PrefixMax, Ranking::LeastLoadKv] {
            assert_eq!(
                snapshot
                    .decide(ranking, &features, &config)
                    .unwrap()
                    .fallback_reason,
                None
            );
        }
    }
}

#[test]
fn affinity_is_bounded_and_requires_session_epoch_freshness_and_one_block() {
    let (mut snapshot, mut features, mut config) = scenario();
    features.session_id = Some("session".into());
    for (id, score) in [("g0", 2040.0), ("g1", 2000.0), ("g2", 5000.0)] {
        let model = config.cost_models.get_mut(id).unwrap();
        model.prefill = [score, 0.0, 0.0];
        model.decode = [0.0; 3];
        model.beta = 0.0;
    }
    snapshot.home = Some(SessionHomeSnapshot {
        worker_url: snapshot.workers[0].worker_url.clone(),
        engine_epoch: "epoch-0".into(),
        age_ms: 0,
    });
    let result = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert_eq!(result.chosen_worker, 0);
    assert!(result.affinity_applied);
    for kind in [
        "too_slow",
        "expired",
        "epoch",
        "less_than_block",
        "unhealthy",
        "no_session",
    ] {
        let (mut s, mut f, mut c) = (snapshot.clone(), features.clone(), config.clone());
        match kind {
            "too_slow" => c.cost_models.get_mut("g0").unwrap().prefill[0] += 0.001,
            "expired" => s.home.as_mut().unwrap().age_ms = c.session_ttl_secs * 1000,
            "epoch" => s.home.as_mut().unwrap().engine_epoch = "old".into(),
            "less_than_block" => prefix(&mut s, 0, 4096),
            "unhealthy" => s.workers[0].available = false,
            "no_session" => f.session_id = None,
            _ => unreachable!(),
        }
        let result = s.decide(Ranking::KvBatchEct, &f, &c).unwrap();
        assert_eq!(result.chosen_worker, 1, "{kind}");
        assert!(!result.affinity_applied, "{kind}");
    }
    for ranking in [Ranking::PrefixMax, Ranking::LeastLoadKv] {
        assert!(
            !snapshot
                .decide(ranking, &features, &config)
                .unwrap()
                .affinity_applied
        );
    }
}

#[test]
fn tie_breaking_is_independent_of_snapshot_order() {
    let (mut snapshot, features, mut config) = scenario();
    for i in 0..3 {
        prefix(&mut snapshot, i, 0);
        snapshot.workers[i].inflight = 0;
        let model = config.cost_models.get_mut(&format!("g{i}")).unwrap();
        model.prefill = [0.0; 3];
    }
    for ranking in RANKINGS {
        for _ in 0..3 {
            snapshot.workers.rotate_left(1);
            assert_eq!(
                snapshot
                    .decide(ranking, &features, &config)
                    .unwrap()
                    .chosen_worker,
                0
            );
        }
    }
}

#[test]
fn ambiguous_worker_identity_and_missing_features_are_explicit() {
    let (mut snapshot, mut features, config) = scenario();
    snapshot.workers[2].metadata.as_mut().unwrap().worker_id = "g0".into();
    assert_eq!(
        snapshot
            .decide(Ranking::PrefixMax, &features, &config)
            .unwrap()
            .fallback_reason,
        Some("ambiguous_worker_identity")
    );
    features.fingerprint = None;
    assert_eq!(
        snapshot
            .decide(Ranking::PrefixMax, &features, &config)
            .unwrap()
            .fallback_reason,
        Some("missing_request_features")
    );
    snapshot.workers[2].worker_index = 0;
    assert!(snapshot
        .decide(Ranking::PrefixMax, &features, &config)
        .is_none());
}

#[test]
fn all_policies_observe_concurrent_reservations_and_release_once() {
    for ranking in RANKINGS {
        let (mut template, features, mut config) = scenario();
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://fixture-g{i}"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        for i in 0..3 {
            prefix(&mut template, i, 0);
            let model = config.cost_models.get_mut(&format!("g{i}")).unwrap();
            model.prefill = [100.0, 0.0, 0.0];
        }
        let ledger = Arc::new(DispatchLedger::default());
        let barrier = Arc::new(Barrier::new(30));
        let threads: Vec<_> = (0..30)
            .map(|_| {
                let (ledger, workers, barrier, template, features, config) = (
                    ledger.clone(),
                    workers.clone(),
                    barrier.clone(),
                    template.clone(),
                    features.clone(),
                    config.clone(),
                );
                std::thread::spawn(move || {
                    barrier.wait();
                    ledger
                        .select_and_reserve(&workers, Arc::new(ObservedPolicy(ranking)), || {
                            let mut snapshot = template;
                            for observation in &mut snapshot.workers {
                                observation.inflight = workers[observation.worker_index].load();
                            }
                            snapshot
                                .decide(ranking, &features, &config)
                                .map(|d| d.chosen_worker)
                        })
                        .unwrap()
                })
            })
            .collect();
        let reservations: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(
            workers.iter().all(|worker| worker.load() == 10),
            "{ranking:?}"
        );
        for mut reservation in reservations {
            reservation.dispatched();
            reservation.finish(true);
            reservation.finish(true);
        }
        assert!(workers.iter().all(|worker| worker.load() == 0));
    }
}
