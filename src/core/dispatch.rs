//! One reservation per backend attempt, shared by every regular HTTP policy.
use super::Worker;
use crate::{metrics::RouterMetrics, policies::LoadBalancingPolicy};
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};
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
        })
    }

    fn release(&self, id: Uuid, success: bool) -> bool {
        let attempt = self.attempts.lock().remove(&id);
        if let Some(attempt) = attempt {
            attempt.worker.decrement_load();
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
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reservation")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Reservation {
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
        }
    }
    /// Idempotent even when a terminal event, EOF and cleanup all observe it.
    pub fn finish(&mut self, success: bool) {
        if self.ledger.release(self.id, success) && success {
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
mod tests {
    use super::*;
    use crate::{
        core::{BasicWorker, WorkerType},
        policies::RoundRobinPolicy,
    };
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
