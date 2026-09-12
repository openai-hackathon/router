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

fn endpoint_scenario() -> (SelectionSnapshot, RequestFeatures, RoutingConfig) {
    let (mut snapshot, mut features, mut config) = scenario();
    controller_evidence(&mut snapshot);
    features.fingerprint = None;
    let mut workers = serde_json::Map::new();
    for worker in &mut snapshot.workers {
        let meta = worker.metadata.as_mut().unwrap();
        workers.insert(worker.worker_url.clone(), serde_json::json!({
            "controller_url":worker.worker_url, "instance_id":meta.worker_id, "block_size":meta.block_size
        }));
        meta.fingerprint.clear();
        meta.engine_epoch.clear();
        if let PrefixEvidence::LmCacheObserved { engine_epoch, .. } = &mut worker.evidence {
            *engine_epoch = None;
        }
        config.restore_models.insert(
            meta.worker_id.clone(),
            vllm_router_rs::routing_state::cost::RestoreCostModel {
                fingerprint: "assigned-by-endpoint-test-only".into(),
                calibration_version: "synthetic-restore-only".into(),
                location: "LocalCPUBackend".into(),
                token_range: [1, 8192],
                fixed_ms: 0.0,
                per_token_ms: 0.0,
            },
        );
    }
    config.lmcache = Some(
        serde_json::from_value(serde_json::json!({
            "identity_mode": "endpoint",
            "renderer_base_url": "http://renderer",
            "model": features.model,
            "workers": workers,
        }))
        .unwrap(),
    );
    config.lmcache.as_ref().unwrap().validate().unwrap();
    (snapshot, features, config)
}

fn completion_scenario() -> (SelectionSnapshot, RequestFeatures, RoutingConfig) {
    use vllm_router_rs::routing_state::{backend_load::BackendLoadSnapshot, config::EctModel};
    let (mut snapshot, features, mut config) = endpoint_scenario();
    config.ect_model = EctModel::CompletionTime;
    config.cost_models.clear();
    config.restore_models.clear();
    let mut urls = serde_json::Map::new();
    for worker in &mut snapshot.workers {
        urls.insert(
            worker.worker_url.clone(),
            format!("{}/metrics", worker.worker_url).into(),
        );
        worker.backend_load = Some(BackendLoadSnapshot {
            running: 0,
            waiting: 0,
            kv_usage_fraction: Some(0.0),
            age_ms: 0,
        });
        config.completion_models.insert(worker.metadata.as_ref().unwrap().worker_id.clone(),
            serde_json::from_value(serde_json::json!({
                "fingerprint":"", "calibration_version":"synthetic-completion-test", "source":"synthetic",
                "prompt_range":[1,8192], "output_range":[1,512], "concurrency_range":[0,64],
                "cache_fraction_range":[0.0,1.0], "backend_running_range":[0,64],
                "backend_waiting_range":[0,64], "output_prior":16,
                "coefficients": {"intercept_ms":100.0,"prompt_token_ms":1.0,"output_token_ms":1.0,
                    "prompt_output_token_ms":0.0,"cache_token_ms":0.9,"router_inflight_ms":5.0,
                    "backend_running_ms":30.0,"backend_waiting_ms":100.0,"kv_usage_ms":5000.0}
            })).unwrap());
    }
    config.backend_metrics = Some(
        serde_json::from_value(serde_json::json!({
            "model":"local", "urls":urls, "max_age_ms":1000
        }))
        .unwrap(),
    );
    snapshot.workers[0].backend_load.as_mut().unwrap().waiting = 40;
    (snapshot, features, config)
}

#[test]
fn completion_model_uses_cpu_prefix_background_work_and_kv_pressure() {
    let (mut snapshot, features, config) = completion_scenario();
    // Same original three-worker preferences, now with externally queued work.
    for (ranking, expected) in RANKINGS.into_iter().zip([0, 2, 1]) {
        let decision = snapshot.decide(ranking, &features, &config).unwrap();
        assert_eq!(decision.chosen_worker, expected);
        assert_eq!(decision.fallback_reason, None);
        if ranking == Ranking::KvBatchEct {
            assert!(decision
                .candidates
                .iter()
                .all(|c| c.cost.is_none() && c.completion_cost.is_some()));
            assert_eq!(decision.ect_model, "completion_time");
        }
    }
    snapshot.workers[0].backend_load.as_mut().unwrap().waiting = 0;
    let warm = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert_eq!(warm.chosen_worker, 0);
    // Gauge pressure does not overwrite request-specific CPU prefix evidence.
    snapshot.workers[0]
        .backend_load
        .as_mut()
        .unwrap()
        .kv_usage_fraction = Some(1.0);
    let pressured = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert_eq!(pressured.chosen_worker, 1);
    assert_eq!(
        pressured.candidates[0].reusable_tokens,
        warm.candidates[0].reusable_tokens
    );
    // A partial CPU tail gets credit even though native block-normalized H stays fixed.
    let before = pressured.candidates[0].ect_ms.unwrap();
    if let PrefixEvidence::LmCacheObserved { cached_tokens, .. } = &mut snapshot.workers[0].evidence
    {
        *cached_tokens += 1;
    }
    let partial = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert!((before - partial.candidates[0].ect_ms.unwrap() - 0.9).abs() < 1e-8);
    // Endpoint-specific service capacity can outweigh a cold cache.
    let mut faster = config;
    faster
        .completion_models
        .get_mut("g2")
        .unwrap()
        .coefficients
        .prompt_token_ms = 0.01;
    assert_eq!(
        snapshot
            .decide(Ranking::KvBatchEct, &features, &faster)
            .unwrap()
            .chosen_worker,
        2
    );
}

#[test]
fn completion_failures_fall_back_together_without_changing_baselines() {
    let (snapshot, features, config) = completion_scenario();
    for mode in 0..7 {
        let (mut snapshot, mut features, mut config) =
            (snapshot.clone(), features.clone(), config.clone());
        let reason = match mode {
            0 => {
                snapshot.workers[0].backend_load = None;
                "missing_backend_metrics"
            }
            1 => {
                snapshot.workers[0].backend_load.as_mut().unwrap().age_ms = 1001;
                "stale_backend_metrics"
            }
            2 => {
                snapshot.workers[0]
                    .backend_load
                    .as_mut()
                    .unwrap()
                    .kv_usage_fraction = None;
                "missing_backend_kv_usage"
            }
            3 => {
                config.completion_models.remove("g0");
                "missing_completion_model"
            }
            4 => {
                config
                    .completion_models
                    .get_mut("g0")
                    .unwrap()
                    .coefficients
                    .cache_token_ms = f64::NAN;
                "invalid_completion_prediction"
            }
            5 => {
                config
                    .completion_models
                    .get_mut("g0")
                    .unwrap()
                    .coefficients
                    .cache_token_ms = 1000.0;
                "invalid_completion_prediction"
            }
            _ => {
                features.num_choices = 2;
                "unsupported_completion_choices"
            }
        };
        let result = snapshot
            .decide(Ranking::KvBatchEct, &features, &config)
            .unwrap();
        assert_eq!(result.fallback_reason, Some(reason));
        assert_eq!(result.chosen_worker, 2);
        assert_eq!(result.candidates.len(), 3);
        for (ranking, expected) in [(Ranking::PrefixMax, 0), (Ranking::LeastLoadKv, 2)] {
            let result = snapshot.decide(ranking, &features, &config).unwrap();
            assert_eq!(result.fallback_reason, None);
            assert_eq!(result.chosen_worker, expected);
        }
    }
}

#[test]
fn three_worker_completion_config_is_loadable_and_model_names_are_strict() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/configs/completion_time_routing.json"
    );
    let config = RoutingConfig::load(Some(path)).unwrap();
    assert_eq!(
        config.ect_model,
        vllm_router_rs::routing_state::config::EctModel::CompletionTime
    );
    assert_eq!(config.completion_models.len(), 3);
    assert_eq!(config.backend_metrics.as_ref().unwrap().urls.len(), 3);
    for model in config.completion_models.values() {
        assert!(
            model
                .estimate("", 1024, 1023, Some(64), 1, 2, 1, Some(0.5))
                .unwrap()
                .ect_ms
                > 0.0
        );
    }
    let mut value = serde_json::to_value(config).unwrap();
    value["ect_model"] = "completin_time".into();
    assert!(serde_json::from_value::<RoutingConfig>(value).is_err());
}

#[test]
fn endpoint_identity_uses_configured_bindings_without_fabricating_verification() {
    let (snapshot, features, config) = endpoint_scenario();
    for (ranking, chosen) in RANKINGS.into_iter().zip([0, 2, 1]) {
        let result = snapshot.decide(ranking, &features, &config).unwrap();
        assert_eq!(result.chosen_worker, chosen);
        assert_eq!(result.identity_mode, "endpoint");
        assert_eq!(result.fallback_reason, None);
        assert!(result.candidates.iter().all(|c| matches!(
            c.evidence,
            PrefixEvidence::LmCacheObserved {
                engine_epoch: None,
                ..
            }
        )));
    }
    let mut strict = config.clone();
    strict.lmcache.as_mut().unwrap().identity_mode =
        vllm_router_rs::routing_state::lmcache::IdentityMode::Verified;
    assert!(snapshot
        .decide(Ranking::PrefixMax, &features, &strict)
        .unwrap()
        .fallback_reason
        .is_some());
    let mut serialized = serde_json::to_value(&config).unwrap();
    serialized["lmcache"]
        .as_object_mut()
        .unwrap()
        .remove("identity_mode");
    let default: RoutingConfig = serde_json::from_value(serialized.clone()).unwrap();
    assert_eq!(default.identity_mode_name(), "verified");
    serialized["lmcache"]["identity_mode"] = "endpont".into();
    assert!(serde_json::from_value::<RoutingConfig>(serialized).is_err());
    let mut unmapped = config.clone();
    unmapped
        .lmcache
        .as_mut()
        .unwrap()
        .workers
        .remove(&snapshot.workers[0].worker_url);
    let result = snapshot
        .decide(Ranking::PrefixMax, &features, &unmapped)
        .unwrap();
    assert_eq!(result.fallback_reason, Some("invalid_lmcache_evidence"));
    assert_eq!(result.candidates.len(), 3);
}

#[test]
fn partial_cpu_prefix_preserves_the_full_restoration_charge_and_range_check() {
    let (mut snapshot, features, mut config) = endpoint_scenario();
    if let PrefixEvidence::LmCacheObserved { cached_tokens, .. } = &mut snapshot.workers[0].evidence
    {
        *cached_tokens = 6145; // routing prefix remains 6144 complete-block tokens
    }
    config.restore_models.get_mut("g0").unwrap().per_token_ms = 1.0;
    for ranking in RANKINGS {
        let decision = snapshot.decide(ranking, &features, &config).unwrap();
        assert_eq!(decision.fallback_reason, None);
        let candidate = decision
            .candidates
            .iter()
            .find(|c| c.worker_index == 0)
            .unwrap();
        assert_eq!(candidate.reusable_tokens, 6144);
        if ranking == Ranking::KvBatchEct {
            let cost = candidate.cost.as_ref().unwrap();
            assert_eq!(cost.restore_ms, 6145.0);
            assert!((cost.ect_ms - 23953.0).abs() < 1e-8);
        }
    }
    config.restore_models.get_mut("g0").unwrap().token_range = [1, 6144];
    assert_eq!(
        snapshot
            .decide(Ranking::KvBatchEct, &features, &config)
            .unwrap()
            .fallback_reason,
        Some("outside_restore_calibration_range")
    );
}

#[test]
fn endpoint_mode_keeps_evidence_and_cost_fallback_and_bounded_affinity() {
    let (mut snapshot, mut features, mut config) = endpoint_scenario();
    let original = snapshot.workers[2].evidence.clone();
    for (evidence, reason) in [
        (PrefixEvidence::Unknown, "unknown_kv"),
        (PrefixEvidence::Stale, "stale_kv"),
    ] {
        snapshot.workers[2].evidence = evidence;
        for ranking in RANKINGS {
            assert_eq!(
                snapshot
                    .decide(ranking, &features, &config)
                    .unwrap()
                    .fallback_reason,
                Some(reason)
            );
        }
    }
    snapshot.workers[2].evidence = original;
    let id = snapshot.workers[0]
        .metadata
        .as_ref()
        .unwrap()
        .worker_id
        .clone();
    let restore = config.restore_models.remove(&id).unwrap();
    assert_eq!(
        snapshot
            .decide(Ranking::KvBatchEct, &features, &config)
            .unwrap()
            .fallback_reason,
        Some("missing_restore_model")
    );
    config.restore_models.insert(id, restore);
    for (worker, score) in snapshot.workers.iter().zip([111.5, 100.0, 500.0]) {
        let model = config
            .cost_models
            .get_mut(&worker.metadata.as_ref().unwrap().worker_id)
            .unwrap();
        model.prefill = [0.0; 3];
        model.decode = [score, 0.0, 0.0];
        model.beta = 0.0;
    }
    features.session_id = Some("endpoint-session".into());
    snapshot.home = Some(SessionHomeSnapshot {
        worker_url: snapshot.workers[0].worker_url.clone(),
        engine_epoch: String::new(),
        age_ms: 0,
    });
    let result = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert!(result.affinity_applied);
    assert_eq!(result.chosen_worker, 0);
    snapshot.home.as_mut().unwrap().age_ms = config.session_ttl_secs * 1000;
    let result = snapshot
        .decide(Ranking::KvBatchEct, &features, &config)
        .unwrap();
    assert!(!result.affinity_applied);
    assert_eq!(result.chosen_worker, 1);
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
