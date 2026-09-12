//! One reservation per backend attempt, shared by every regular HTTP policy.
use super::Worker;
use crate::{metrics::RouterMetrics, policies::LoadBalancingPolicy};
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc, time::Instant};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptState {
    Reserved,
    Dispatched,
    Unknown,
}

#[derive(Debug)]
struct Attempt {
    worker: Arc<dyn Worker>,
    policy: Arc<dyn LoadBalancingPolicy>,
    state: AttemptState,
    epoch: Option<String>,
}

#[derive(Debug, Default)]
pub struct DispatchLedger {
    attempts: Mutex<HashMap<Uuid, Attempt>>,
}

impl DispatchLedger {
    /// Selection reads loads while holding the same lock that increments them.
    /// The closure must not perform preprocessing or I/O.
    pub fn select_and_reserve(
        self: &Arc<Self>,
        workers: &[Arc<dyn Worker>],
        policy: Arc<dyn LoadBalancingPolicy>,
        select: impl FnOnce() -> Option<usize>,
    ) -> Option<Reservation> {
        let mut attempts = self.attempts.lock();
        let worker = workers.get(select()?)?.clone();
        let id = Uuid::new_v4();
        worker.increment_load();
        RouterMetrics::set_running_requests(worker.url(), worker.load());
        attempts.insert(
            id,
            Attempt {
                worker: worker.clone(),
                policy,
                state: AttemptState::Reserved,
                epoch: None,
            },
        );
        Some(Reservation {
            ledger: self.clone(),
            worker,
            id,
            on_success: None,
            dispatched_at: None,
            sample_context: None,
            output_tokens: None,
            finish_reason: None,
        })
    }

    fn release(&self, id: Uuid, success: bool) -> bool {
        let mut attempts = self.attempts.lock();
        let attempt = attempts.remove(&id);
        if let Some(attempt) = attempt {
            attempt.worker.decrement_load();
            drop(attempts);
            RouterMetrics::set_running_requests(attempt.worker.url(), attempt.worker.load());
            attempt
                .policy
                .on_request_complete(attempt.worker.url(), success);
            metrics::counter!("router_dispatch_finished_total", "success" => success.to_string())
                .increment(1);
            true
        } else {
            false
        }
    }

    /// Only an independently verified engine replacement confirms that old
    /// attempts cannot still be running. No timeout guesses or metric addition.
    pub fn observe_epoch(&self, worker_url: &str, epoch: &str) {
        let ids: Vec<_> = self
            .attempts
            .lock()
            .iter()
            .filter(|(_, a)| {
                a.worker.url() == worker_url && a.epoch.as_deref().is_some_and(|old| old != epoch)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.release(id, false);
        }
    }

    pub fn unknown_count(&self, worker_url: &str) -> usize {
        self.attempts
            .lock()
            .values()
            .filter(|a| a.worker.url() == worker_url && a.state == AttemptState::Unknown)
            .count()
    }
}

pub struct Reservation {
    ledger: Arc<DispatchLedger>,
    pub worker: Arc<dyn Worker>,
    pub id: Uuid,
    on_success: Option<Box<dyn FnOnce() + Send>>,
    dispatched_at: Option<Instant>,
    sample_context: Option<serde_json::Map<String, serde_json::Value>>,
    output_tokens: Option<u64>,
    finish_reason: Option<String>,
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reservation")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Reservation {
    /// Attach routing features only: callers must exclude prompts and token IDs.
    /// A completion is measured from dispatch, after rendering and lookup.
    pub fn set_sample_context(&mut self, context: serde_json::Value) {
        self.sample_context = context.as_object().cloned();
    }

    /// Observe usage already present in a JSON response or an SSE data event.
    /// This never asks the backend to change its response/streaming options.
    pub fn observe_response_usage(&mut self, value: &serde_json::Value) {
        let response = value.get("response").unwrap_or(value);
        if let Some(tokens) = response.get("usage").and_then(|usage| {
            usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
                .and_then(serde_json::Value::as_u64)
        }) {
            self.output_tokens = Some(tokens);
        }
        let reasons: Vec<_> = response
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|choice| choice.get("finish_reason").and_then(|v| v.as_str()))
            .collect();
        // Any truncated choice makes the aggregate usage a censored sample.
        let reason = if reasons.contains(&"length") {
            Some("length")
        } else {
            reasons.first().copied().or_else(|| {
                response
                    .get("finish_reason")
                    .and_then(serde_json::Value::as_str)
            })
        }
        .or_else(|| {
            let status = response.get("status").and_then(|v| v.as_str());
            match status {
                Some("completed") => Some("stop"),
                Some("incomplete")
                    if response
                        .pointer("/incomplete_details/reason")
                        .and_then(|v| v.as_str())
                        == Some("max_output_tokens") =>
                {
                    Some("length")
                }
                _ => None,
            }
        });
        if let Some(reason) = reason.filter(|reason| reason.len() <= 128) {
            self.finish_reason = Some(reason.to_owned());
        }
    }

    pub fn set_epoch(&mut self, epoch: String) {
        if let Some(attempt) = self.ledger.attempts.lock().get_mut(&self.id) {
            attempt.epoch = Some(epoch);
        }
    }
    pub fn on_success(&mut self, callback: impl FnOnce() + Send + 'static) {
        self.on_success = Some(Box::new(callback));
    }
    pub fn dispatched(&mut self) {
        if let Some(attempt) = self.ledger.attempts.lock().get_mut(&self.id) {
            attempt.state = AttemptState::Dispatched;
            self.dispatched_at.get_or_insert_with(Instant::now);
        }
    }
    /// Idempotent even when a terminal event, EOF and cleanup all observe it.
    pub fn finish(&mut self, success: bool) {
        let finished_at = Instant::now();
        if !self.ledger.release(self.id, success) {
            return;
        }
        if let (Some(started), Some(mut context)) = (self.dispatched_at, self.sample_context.take())
        {
            context.extend(
                serde_json::json!({
                    "attempt_id": self.id.to_string(),
                    "sample_id": self.id.to_string(),
                    "success": success,
                    "completion_ms": finished_at.duration_since(started).as_secs_f64() * 1000.0,
                    "output_tokens": self.output_tokens,
                    "finish_reason": self.finish_reason,
                })
                .as_object()
                .unwrap()
                .clone(),
            );
            let sample = serde_json::Value::Object(context);
            tracing::info!(sample = %sample, "routing completion sample");
        }
        if success {
            if let Some(callback) = self.on_success.take() {
                callback();
            }
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut attempts = self.ledger.attempts.lock();
        let Some(attempt) = attempts.get_mut(&self.id) else {
            return;
        };
        if attempt.state == AttemptState::Reserved {
            drop(attempts);
            self.finish(false);
        } else {
            attempt.state = AttemptState::Unknown;
            metrics::counter!("router_dispatch_unknown_total").increment(1);
            tracing::warn!(attempt_id = %self.id, worker = self.worker.url(), "backend termination unconfirmed; retaining in-flight reservation");
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        core::{BasicWorker, WorkerType},
        policies::RoundRobinPolicy,
    };

    pub(crate) fn capture_samples() -> (
        impl tracing::Subscriber + Send + Sync,
        Arc<Mutex<Vec<serde_json::Value>>>,
    ) {
        use tracing_subscriber::{layer::Context, prelude::*, Layer};
        struct Samples(Arc<Mutex<Vec<serde_json::Value>>>);
        impl<S: tracing::Subscriber> Layer<S> for Samples {
            fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
                #[derive(Default)]
                struct Visitor(Option<serde_json::Value>);
                impl tracing::field::Visit for Visitor {
                    fn record_debug(
                        &mut self,
                        field: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        if field.name() == "sample" {
                            self.0 = serde_json::from_str(&format!("{value:?}")).ok();
                        }
                    }
                }
                let mut visitor = Visitor::default();
                event.record(&mut visitor);
                if let Some(sample) = visitor.0 {
                    self.0.lock().push(sample);
                }
            }
        }
        let samples = Arc::new(Mutex::new(Vec::new()));
        (
            tracing_subscriber::registry().with(Samples(samples.clone())),
            samples,
        )
    }
    fn setup() -> (
        Arc<DispatchLedger>,
        Vec<Arc<dyn Worker>>,
        Arc<dyn LoadBalancingPolicy>,
    ) {
        (
            Arc::new(DispatchLedger::default()),
            vec![Arc::new(BasicWorker::new("w".into(), WorkerType::Regular))],
            Arc::new(RoundRobinPolicy::new()),
        )
    }
    #[test]
    fn release_once_and_keep_ambiguous_attempts() {
        let (ledger, workers, policy) = setup();
        let mut a = ledger
            .select_and_reserve(&workers, policy.clone(), || Some(0))
            .unwrap();
        a.dispatched();
        a.finish(true);
        a.finish(true);
        drop(a);
        assert_eq!(workers[0].load(), 0);
        let mut a = ledger
            .select_and_reserve(&workers, policy.clone(), || Some(0))
            .unwrap();
        a.set_epoch("old".into());
        a.dispatched();
        drop(a);
        let a = ledger
            .select_and_reserve(&workers, policy, || Some(0))
            .unwrap();
        drop(a); // Never sent, so safe to release.
        assert_eq!(workers[0].load(), 1);
        assert_eq!(ledger.unknown_count("w"), 1);
        ledger.observe_epoch("w", "old");
        assert_eq!(workers[0].load(), 1);
        ledger.observe_epoch("w", "new");
        assert_eq!(workers[0].load(), 0);
    }

    #[test]
    fn completion_samples_and_success_callbacks_are_emitted_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (subscriber, samples) = capture_samples();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        let (ledger, workers, policy) = setup();
        let mut reservation = ledger
            .select_and_reserve(&workers, policy, || Some(0))
            .unwrap();
        let callbacks = Arc::new(AtomicUsize::new(0));
        let count = callbacks.clone();
        reservation.on_success(move || {
            count.fetch_add(1, Ordering::SeqCst);
        });
        reservation.set_sample_context(serde_json::json!({
            "worker_id": "w", "backend_running": null, "completion_ms": 0,
        }));
        reservation.dispatched();
        std::thread::sleep(std::time::Duration::from_millis(2));
        reservation.dispatched(); // A repeated transition must not reset the clock.
        reservation.observe_response_usage(&serde_json::json!({
            "usage": {"completion_tokens": 17},
            "choices": [{"finish_reason": "stop"}],
        }));
        reservation.finish(true);
        reservation.finish(false);
        drop(reservation);
        let samples = samples.lock();
        assert_eq!(samples.len(), 1);
        let sample = &samples[0];
        assert_eq!(sample["worker_id"], "w");
        assert!(sample["backend_running"].is_null());
        assert_eq!(sample["attempt_id"], sample["sample_id"]);
        assert_eq!(sample["success"], true);
        assert_eq!(sample["output_tokens"], 17);
        assert_eq!(sample["finish_reason"], "stop");
        assert!(sample["completion_ms"].as_f64().unwrap() >= 2.0);
        assert_eq!(callbacks.load(Ordering::SeqCst), 1);
        assert_eq!(workers[0].load(), 0);
    }

    #[test]
    fn unconfirmed_or_undispatched_attempts_do_not_emit_completion_samples() {
        let (subscriber, samples) = capture_samples();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        let (ledger, workers, policy) = setup();
        for dispatch in [false, true] {
            let mut reservation = ledger
                .select_and_reserve(&workers, policy.clone(), || Some(0))
                .unwrap();
            reservation.set_sample_context(serde_json::json!({"worker_id": "w"}));
            if dispatch {
                reservation.dispatched();
            }
            drop(reservation);
        }
        assert!(samples.lock().is_empty());
        assert_eq!(ledger.unknown_count("w"), 1);
        ledger.observe_epoch("w", "new"); // Unknown identity cannot prove termination.
        assert!(samples.lock().is_empty());
    }

    #[test]
    fn failed_terminal_and_responses_usage_remain_distinct_from_success() {
        let (subscriber, samples) = capture_samples();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        let (ledger, workers, policy) = setup();
        let mut reservation = ledger
            .select_and_reserve(&workers, policy, || Some(0))
            .unwrap();
        reservation.set_sample_context(serde_json::json!({"worker_id": "w"}));
        reservation.dispatched();
        reservation.observe_response_usage(&serde_json::json!({
            "type": "response.incomplete",
            "response": {
                "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"output_tokens": 8},
            },
        }));
        reservation.finish(false);
        let samples = samples.lock();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0]["success"], false);
        assert_eq!(samples[0]["output_tokens"], 8);
        assert_eq!(samples[0]["finish_reason"], "length");
    }
    #[test]
    fn concurrent_selectors_see_reservations() {
        let ledger = Arc::new(DispatchLedger::default());
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(format!("w{i}"), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let barrier = Arc::new(std::sync::Barrier::new(30));
        let joins: Vec<_> = (0..30)
            .map(|_| {
                let (ledger, workers, barrier) = (ledger.clone(), workers.clone(), barrier.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    ledger
                        .select_and_reserve(&workers, Arc::new(RoundRobinPolicy::new()), || {
                            (0..workers.len()).min_by_key(|&i| workers[i].load())
                        })
                        .unwrap()
                })
            })
            .collect();
        let reservations: Vec<_> = joins.into_iter().map(|j| j.join().unwrap()).collect();
        assert!(workers.iter().all(|w| w.load() == 10));
        drop(reservations);
        assert!(workers.iter().all(|w| w.load() == 0));
    }
}
