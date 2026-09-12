use crate::core::dispatch::Reservation;
use axum::{body::Body, response::Response};
use futures_util::StreamExt;

/// Incremental SSE framing: terminal markers may span arbitrary HTTP chunks.
#[derive(Default)]
struct TerminalEvents {
    line: Vec<u8>,
    data: Vec<u8>,
    oversized: bool,
}

impl TerminalEvents {
    fn feed(&mut self, bytes: &[u8], mut observe: impl FnMut(&serde_json::Value)) -> Option<bool> {
        let mut terminal = None;
        for &byte in bytes {
            if byte != b'\n' {
                if self.line.len() < 65536 {
                    self.line.push(byte);
                } else {
                    self.oversized = true;
                }
                continue;
            }
            if self.line.last() == Some(&b'\r') {
                self.line.pop();
            }
            if self.line.is_empty() {
                if !self.oversized {
                    let data = self.data.strip_suffix(b"\n").unwrap_or(&self.data);
                    if data == b"[DONE]" {
                        terminal.get_or_insert(true);
                    } else if let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) {
                        observe(&value);
                        if value.get("error").is_some() {
                            terminal = Some(false);
                        }
                        match value.get("type").and_then(|v| v.as_str()) {
                            Some("response.completed") => {
                                terminal.get_or_insert(true);
                            }
                            Some("response.failed" | "response.incomplete" | "error") => {
                                terminal = Some(false)
                            }
                            _ => {}
                        }
                    }
                }
                self.data.clear();
                self.oversized = false;
            } else if let Some(data) = self.line.strip_prefix(b"data:") {
                let data = data.strip_prefix(b" ").unwrap_or(data);
                if self.data.len() + data.len() < 65536 {
                    self.data.extend_from_slice(data);
                    self.data.push(b'\n');
                } else {
                    self.oversized = true;
                }
            }
            self.line.clear();
        }
        terminal
    }
}

pub async fn forward(
    response: reqwest::Response,
    mut reservation: Reservation,
    stream: bool,
    background: bool,
) -> Response {
    let status = response.status();
    let headers = crate::routers::header_utils::preserve_response_headers(response.headers());
    let sse = headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    if !stream || !status.is_success() {
        return match response.bytes().await {
            Ok(bytes) => {
                if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    reservation.observe_response_usage(&value);
                }
                if !background || !status.is_success() {
                    reservation.finish(status.is_success());
                }
                let mut response = Response::new(Body::from(bytes));
                *response.status_mut() = status;
                *response.headers_mut() = headers;
                response
            }
            Err(_) => {
                use axum::response::IntoResponse;
                (
                    http::StatusCode::BAD_GATEWAY,
                    "Failed to read backend response",
                )
                    .into_response()
            }
        };
    }
    let stream = futures_util::stream::unfold(
        (
            response.bytes_stream(),
            reservation,
            TerminalEvents::default(),
            false,
        ),
        move |(mut stream, mut reservation, mut events, failed)| async move {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    if sse {
                        if let Some(success) =
                            events.feed(&bytes, |value| reservation.observe_response_usage(value))
                        {
                            reservation.finish(success);
                        }
                    }
                    Some((
                        Ok::<_, reqwest::Error>(bytes),
                        (stream, reservation, events, failed),
                    ))
                }
                Some(Err(e)) => Some((Err(e), (stream, reservation, events, true))),
                None => {
                    // SSE EOF without a terminal event can be an interrupted generation.
                    if !sse && !background && !failed {
                        reservation.finish(true);
                    }
                    None
                }
            }
        },
    );
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn backend(
        body: &'static str,
        content_type: &'static str,
        status: http::StatusCode,
    ) -> (reqwest::Response, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move || async move {
                (status, [(http::header::CONTENT_TYPE, content_type)], body)
            }),
        );
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let response = reqwest::get(format!("http://{address}/")).await.unwrap();
        (response, task)
    }

    fn reserve() -> (
        std::sync::Arc<crate::core::dispatch::DispatchLedger>,
        Reservation,
    ) {
        use crate::{
            core::{BasicWorker, WorkerType},
            policies::RoundRobinPolicy,
        };
        use std::sync::Arc;
        let ledger = Arc::new(crate::core::dispatch::DispatchLedger::default());
        let mut reservation = ledger
            .select_and_reserve(
                &[Arc::new(BasicWorker::new(
                    "worker".into(),
                    WorkerType::Regular,
                ))],
                Arc::new(RoundRobinPolicy::new()),
                || Some(0),
            )
            .unwrap();
        reservation.dispatched();
        (ledger, reservation)
    }

    #[tokio::test]
    async fn headers_do_not_release_stream_but_terminal_event_does() {
        let (backend, task) = backend(
            "data: [DONE]\n\n",
            "text/event-stream",
            http::StatusCode::OK,
        )
        .await;
        let (_, reservation) = reserve();
        let worker = reservation.worker.clone();
        let response = forward(backend, reservation, true, false).await;
        assert_eq!(worker.load(), 1);
        axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(worker.load(), 0);
        task.abort();
    }

    #[tokio::test]
    async fn truncated_and_dropped_streams_stay_unknown() {
        let (subscriber, samples) = crate::core::dispatch::tests::capture_samples();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        for consume in [true, false] {
            let (backend, task) =
                backend("data: {}\n\n", "text/event-stream", http::StatusCode::OK).await;
            let (ledger, mut reservation) = reserve();
            reservation.set_sample_context(serde_json::json!({"worker_id": "worker"}));
            let worker = reservation.worker.clone();
            let response = forward(backend, reservation, true, false).await;
            if consume {
                axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .unwrap();
            } else {
                drop(response);
            }
            assert_eq!(worker.load(), 1);
            assert_eq!(ledger.unknown_count("worker"), 1);
            task.abort();
        }
        assert!(samples.lock().is_empty());
    }

    #[tokio::test]
    async fn rejected_stream_request_and_nonstream_release_once() {
        for status in [http::StatusCode::OK, http::StatusCode::SERVICE_UNAVAILABLE] {
            let (backend, task) = backend("{}", "application/json", status).await;
            let (_, reservation) = reserve();
            let worker = reservation.worker.clone();
            let response = forward(backend, reservation, status.is_server_error(), false).await;
            assert_eq!(response.status(), status);
            assert_eq!(worker.load(), 0);
            drop(response);
            assert_eq!(worker.load(), 0);
            task.abort();
        }
    }

    #[tokio::test]
    async fn nonstream_sample_waits_for_body_and_records_actual_usage() {
        use std::{convert::Infallible, sync::Arc, time::Duration};
        let (subscriber, samples) = crate::core::dispatch::tests::capture_samples();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        let gate = Arc::new(tokio::sync::Notify::new());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_gate = gate.clone();
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move || {
                let gate = server_gate.clone();
                async move {
                    let body = futures_util::stream::iter([Ok::<_, Infallible>(
                        "{\"usage\":",
                    )])
                    .chain(futures_util::stream::once(async move {
                        gate.notified().await;
                        Ok("{\"completion_tokens\":9},\"choices\":[{\"finish_reason\":\"stop\"}]}")
                    }));
                    Body::from_stream(body)
                }
            }),
        );
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let (_, mut reservation) = reserve();
        reservation.set_sample_context(serde_json::json!({"worker_id": "worker"}));
        let started = std::time::Instant::now();
        let backend = reqwest::get(format!("http://{address}/")).await.unwrap();
        let headers_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert!(samples.lock().is_empty());
        let worker = reservation.worker.clone();
        assert_eq!(worker.load(), 1);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            gate.notify_one();
        });
        let response = forward(backend, reservation, false, false).await;
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap()["usage"]
                ["completion_tokens"],
            9
        );
        let samples = samples.lock();
        assert_eq!(samples.len(), 1);
        assert!(samples[0]["completion_ms"].as_f64().unwrap() >= headers_ms + 20.0);
        assert_eq!(samples[0]["output_tokens"], 9);
        assert_eq!(samples[0]["finish_reason"], "stop");
        assert_eq!(worker.load(), 0);
        task.abort();
    }

    #[test]
    fn fragmented_usage_events_are_captured_before_a_single_terminal_sample() {
        let (subscriber, samples) = crate::core::dispatch::tests::capture_samples();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        let event = concat!(
            "data: {\"choices\":[{\"finish_reason\":\"length\"}]}\n\n",
            "data: {\"usage\":{\"completion_tokens\":32},\"choices\":[]}\n\n",
            "data: [DONE]\n\n",
        );
        for cut in 0..event.len() {
            let (_, mut reservation) = reserve();
            reservation.set_sample_context(serde_json::json!({"worker_id": "worker"}));
            let mut scanner = TerminalEvents::default();
            assert_eq!(
                scanner.feed(&event.as_bytes()[..cut], |value| reservation
                    .observe_response_usage(value)),
                None,
            );
            assert_eq!(samples.lock().len(), cut);
            let terminal = scanner.feed(&event.as_bytes()[cut..], |value| {
                reservation.observe_response_usage(value)
            });
            reservation.finish(terminal.unwrap());
            reservation.finish(true);
            let samples = samples.lock();
            assert_eq!(samples.len(), cut + 1);
            assert_eq!(samples[cut]["output_tokens"], 32);
            assert_eq!(samples[cut]["finish_reason"], "length");
        }
    }

    #[test]
    fn terminal_events_survive_every_chunk_boundary() {
        for event in [
            "data: [DONE]\n\n",
            "event: response.completed\r\ndata: {\"type\":\"response.completed\"}\r\n\r\n",
        ] {
            for cut in 0..event.len() {
                let mut scanner = TerminalEvents::default();
                assert_eq!(scanner.feed(&event.as_bytes()[..cut], |_| {}), None);
                assert_eq!(scanner.feed(&event.as_bytes()[cut..], |_| {}), Some(true));
            }
        }
    }
    #[test]
    fn content_is_not_a_terminal_event() {
        let mut scanner = TerminalEvents::default();
        assert_eq!(
            scanner.feed(b"data: {\"text\":\"data: [DONE]\"}\n\n", |_| {}),
            None
        );
        assert_eq!(scanner.feed(b"data: [DONE]", |_| {}), None);
    }
}
