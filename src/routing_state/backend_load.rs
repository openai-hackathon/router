//! Background backend load observations for deployments with external traffic.
//! These gauges overlap the Router ledger and must never be added to it.
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

const MAX_WORKERS: usize = 128;
const MAX_METRICS_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendMetricsConfig {
    pub model: String,
    /// Exact inference worker URL -> complete Prometheus metrics URL.
    pub urls: HashMap<String, String>,
    pub poll_interval_ms: u64,
    pub timeout_ms: u64,
    pub max_age_ms: u64,
}

impl Default for BackendMetricsConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            urls: HashMap::new(),
            poll_interval_ms: 1000,
            timeout_ms: 2000,
            max_age_ms: 3000,
        }
    }
}

impl BackendMetricsConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.model.trim().is_empty()
            || self.urls.is_empty()
            || self.urls.len() > MAX_WORKERS
            || self.poll_interval_ms == 0
            || self.timeout_ms == 0
            || self.max_age_ms == 0
        {
            return Err("backend metrics require a model, 1..=128 workers, and positive intervals and limits".into());
        }
        let mut inference_urls = HashSet::new();
        let mut metrics_urls = HashSet::new();
        for (inference, metrics) in &self.urls {
            for value in [inference, metrics] {
                let parsed = reqwest::Url::parse(value)
                    .map_err(|_| "invalid backend metrics HTTP(S) URL")?;
                if !matches!(parsed.scheme(), "http" | "https")
                    || parsed.host_str().is_none()
                    || !parsed.username().is_empty()
                    || parsed.password().is_some()
                    || parsed.query().is_some()
                    || parsed.fragment().is_some()
                {
                    return Err("backend metrics URLs must use HTTP(S) without credentials, queries, or fragments; use header_env for authentication".into());
                }
            }
            if !inference_urls.insert(inference.trim_end_matches('/'))
                || !metrics_urls.insert(metrics.trim_end_matches('/'))
            {
                return Err("backend metrics require unique worker and metrics URLs".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BackendLoadSnapshot {
    pub running: usize,
    pub waiting: usize,
    /// Backend GPU KV occupancy; never a request-specific prefix hit estimate.
    pub kv_usage_fraction: Option<f64>,
    /// Time since this scrape began, including its HTTP round trip.
    pub age_ms: u64,
}

#[derive(Debug)]
struct Sample {
    running: usize,
    waiting: usize,
    kv_usage_fraction: Option<f64>,
    started: Instant,
}

#[derive(Debug)]
pub struct BackendLoadCollector {
    samples: Arc<RwLock<HashMap<String, Sample>>>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    max_age: Duration,
}

impl Drop for BackendLoadCollector {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

impl BackendLoadCollector {
    pub fn spawn(client: reqwest::Client, config: BackendMetricsConfig) -> Self {
        let mut collector = Self {
            samples: Arc::new(RwLock::new(HashMap::new())),
            tasks: Vec::new(),
            max_age: Duration::from_millis(config.max_age_ms),
        };
        // RoutingConfig validates at startup. Keep direct callers fail-closed
        // instead of panicking on a zero interval or spawning unbounded tasks.
        if config.validate().is_err() {
            return collector;
        }
        for (worker, url) in config.urls {
            let worker = worker.trim_end_matches('/').to_owned();
            let samples = collector.samples.clone();
            let client = client.clone();
            let model = config.model.clone();
            collector.tasks.push(tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_millis(config.poll_interval_ms));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    interval.tick().await;
                    let started = Instant::now();
                    let result = scrape(&client, &url, &model, config.timeout_ms).await;
                    match result {
                        Ok((running, waiting, kv_usage_fraction)) => {
                            samples.write().insert(
                                worker.clone(),
                                Sample {
                                    running,
                                    waiting,
                                    kv_usage_fraction,
                                    started,
                                },
                            );
                        }
                        Err(reason) => {
                            samples.write().remove(&worker);
                            metrics::counter!("router_backend_metrics_errors_total", "reason" => reason)
                                .increment(1);
                        }
                    }
                }
            }));
        }
        collector
    }

    pub fn snapshot(&self, worker_url: &str) -> Option<BackendLoadSnapshot> {
        let samples = self.samples.read();
        let sample = samples.get(worker_url.trim_end_matches('/'))?;
        let age = sample.started.elapsed();
        if age > self.max_age {
            return None;
        }
        Some(BackendLoadSnapshot {
            running: sample.running,
            waiting: sample.waiting,
            kv_usage_fraction: sample.kv_usage_fraction,
            age_ms: age.as_millis().min(u64::MAX as u128) as u64,
        })
    }
}

async fn scrape(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    timeout_ms: u64,
) -> Result<(usize, usize, Option<f64>), &'static str> {
    let mut response = client
        .get(url)
        .timeout(Duration::from_millis(timeout_ms))
        .send()
        .await
        .map_err(|_| "metrics_transport_error")?
        .error_for_status()
        .map_err(|_| "metrics_http_error")?;
    if response
        .content_length()
        .is_some_and(|length| length > MAX_METRICS_BYTES as u64)
    {
        return Err("metrics_body_too_large");
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "metrics_transport_error")?
    {
        if chunk.len() > MAX_METRICS_BYTES - body.len() {
            return Err("metrics_body_too_large");
        }
        body.extend_from_slice(&chunk);
    }
    parse_metrics(
        std::str::from_utf8(&body).map_err(|_| "invalid_metrics_encoding")?,
        model,
    )
}

/// Require one sample per gauge and one matching engine. Aggregating multiple
/// engines would break the configured one-endpoint/one-worker assumption.
fn parse_metrics(body: &str, model: &str) -> Result<(usize, usize, Option<f64>), &'static str> {
    let mut running: Option<(String, f64)> = None;
    let mut waiting: Option<(String, f64)> = None;
    let mut kv_usage: Option<(String, f64)> = None;
    for line in body.lines() {
        let line = line.trim();
        let end = line
            .find(|c: char| c == '{' || c.is_whitespace())
            .unwrap_or(line.len());
        let is_kv_usage = &line[..end] == "vllm:kv_cache_usage_perc";
        let target = match &line[..end] {
            "vllm:num_requests_running" => &mut running,
            "vllm:num_requests_waiting" => &mut waiting,
            "vllm:kv_cache_usage_perc" => &mut kv_usage,
            _ => continue,
        };
        let (labels, value) = parse_labels(line[end..].trim_start())?;
        if labels.get("model_name").map(String::as_str) != Some(model) {
            continue;
        }
        let engine = labels
            .get("engine")
            .filter(|engine| !engine.is_empty())
            .ok_or("missing_metrics_engine")?;
        let mut fields = value.split_whitespace();
        let count: f64 = fields
            .next()
            .ok_or("missing_metrics_value")?
            .parse()
            .map_err(|_| "invalid_metrics_value")?;
        if !count.is_finite()
            || count < 0.0
            || (is_kv_usage && count > 1.0)
            || (!is_kv_usage && (count.fract() != 0.0 || count >= usize::MAX as f64))
        {
            return Err("invalid_metrics_value");
        }
        // Prometheus allows an optional numeric timestamp. OpenMetrics may
        // append an exemplar after '#'; neither changes the gauge value.
        match fields.next() {
            Some(extra)
                if extra != "#"
                    && (!extra.parse::<f64>().is_ok_and(f64::is_finite)
                        || fields.next().is_some_and(|extra| extra != "#")) =>
            {
                return Err("invalid_metrics_value");
            }
            _ => {}
        }
        if target.replace((engine.clone(), count)).is_some() {
            return Err("ambiguous_metrics_engine");
        }
    }
    match (running, waiting) {
        (Some((running_engine, running)), Some((waiting_engine, waiting)))
            if running_engine == waiting_engine
                && kv_usage
                    .as_ref()
                    .is_none_or(|(engine, _)| engine == &running_engine) =>
        {
            Ok((
                running as usize,
                waiting as usize,
                kv_usage.map(|(_, value)| value),
            ))
        }
        (Some(_), Some(_)) => Err("ambiguous_metrics_engine"),
        _ => Err("missing_backend_metrics"),
    }
}

/// Parse labels with Prometheus escaping, not comma splitting: label values
/// may contain commas, braces, escaped quotes, backslashes, and newlines.
fn parse_labels(input: &str) -> Result<(HashMap<String, String>, &str), &'static str> {
    let mut remaining = input.strip_prefix('{').ok_or("missing_metrics_labels")?;
    let mut labels = HashMap::new();
    loop {
        remaining = remaining.trim_start();
        if let Some(rest) = remaining.strip_prefix('}') {
            return Ok((labels, rest.trim_start()));
        }
        let key_length = remaining
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if key_length == 0 || remaining.as_bytes()[0].is_ascii_digit() {
            return Err("invalid_metrics_labels");
        }
        let key = &remaining[..key_length];
        remaining = remaining[key_length..]
            .trim_start()
            .strip_prefix('=')
            .ok_or("invalid_metrics_labels")?
            .trim_start()
            .strip_prefix('"')
            .ok_or("invalid_metrics_labels")?;
        let mut value = String::new();
        let mut chars = remaining.char_indices();
        let closing = loop {
            let (position, ch) = chars.next().ok_or("invalid_metrics_labels")?;
            match ch {
                '"' => break position,
                '\\' => value.push(match chars.next().map(|(_, ch)| ch) {
                    Some('n') => '\n',
                    Some('"') => '"',
                    Some('\\') => '\\',
                    _ => return Err("invalid_metrics_labels"),
                }),
                ch => value.push(ch),
            }
        };
        if labels.insert(key.to_owned(), value).is_some() {
            return Err("duplicate_metrics_label");
        }
        remaining = remaining[closing + 1..].trim_start();
        if let Some(rest) = remaining.strip_prefix(',') {
            remaining = rest;
        } else if !remaining.starts_with('}') {
            return Err("invalid_metrics_labels");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    const VALID: &str = concat!(
        "vllm:num_requests_running{engine=\"0\",model_name=\"local\"} 2\n",
        "vllm:num_requests_waiting{engine=\"0\",model_name=\"local\"} 1\n",
    );

    #[test]
    fn selects_one_model_and_parses_escaped_extra_labels() {
        let label_text =
            r#"model_name="local",description="comma, brace } quote \" slash \\ newline \n 中文""#;
        let labeled = format!("{{{label_text}}} 2");
        let (labels, value) = parse_labels(&labeled).unwrap();
        assert_eq!(
            labels["description"],
            "comma, brace } quote \" slash \\ newline \n 中文"
        );
        assert_eq!(value, "2");
        let body = format!(
            "# HELP ignored\n\
             vllm:num_requests_running{{engine=\"3\",model_name=\"other\"}} 19\n\
             vllm:num_requests_waiting{{engine=\"3\",model_name=\"other\"}} 20\n\
             vllm:num_requests_waiting_by_reason{{engine=\"0\",model_name=\"local\"}} 50\n\
             {}",
            VALID.replace("model_name=\"local\"", label_text),
        );
        assert_eq!(parse_metrics(&body, "local"), Ok((2, 1, None)));
        assert_eq!(parse_metrics(&body, "other"), Ok((19, 20, None)));
        assert!(parse_metrics(&body, "missing").is_err());
    }

    #[test]
    fn rejects_incomplete_ambiguous_and_invalid_observations() {
        for bad in ["NaN", "+Inf", "-Inf", "-1", "0.5", "1e100", "unknown"] {
            assert!(
                parse_metrics(&VALID.replace("} 2", &format!("}} {bad}")), "local").is_err(),
                "{bad}"
            );
        }
        for body in [
            VALID.lines().next().unwrap().to_owned(),
            format!("{VALID}{VALID}"),
            VALID.replacen("engine=\"0\"", "engine=\"1\"", 1),
            VALID.replace("engine=\"0\",", ""),
            VALID.replace("engine=\"0\",", "engine=\"0\",engine=\"0\","),
            VALID.replace("model_name=\"local\"", "model_name=\"local\\x\""),
        ] {
            assert!(parse_metrics(&body, "local").is_err(), "{body}");
        }
        assert_eq!(
            parse_metrics(&VALID.replace("} 2", "} 2e0 123456"), "local"),
            Ok((2, 1, None))
        );
    }

    #[test]
    fn kv_pressure_is_optional_but_must_match_the_single_engine() {
        let gauge = "vllm:kv_cache_usage_perc{engine=\"0\",model_name=\"local\"} 0.625\n";
        assert_eq!(
            parse_metrics(&format!("{VALID}{gauge}"), "local"),
            Ok((2, 1, Some(0.625)))
        );
        for invalid in [
            gauge.replace("0.625", "1.01"),
            gauge.replace("0.625", "-0.01"),
            gauge.replace("0.625", "NaN"),
            gauge.replace("engine=\"0\"", "engine=\"1\""),
            format!("{gauge}{gauge}"),
        ] {
            assert!(parse_metrics(&format!("{VALID}{invalid}"), "local").is_err());
        }
        assert_eq!(
            parse_metrics(
                &format!("{VALID}{}", gauge.replace("local", "other")),
                "local"
            ),
            Ok((2, 1, None))
        );
    }

    #[test]
    fn stale_unknown_and_invalid_configuration_are_not_idle() {
        let config = BackendMetricsConfig {
            model: "local".into(),
            urls: HashMap::from([("https://worker".into(), "https://worker/metrics".into())]),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        assert!(BackendMetricsConfig::default().validate().is_err());
        let mut duplicate = config.clone();
        duplicate
            .urls
            .insert("https://worker/".into(), "https://worker/metrics-2".into());
        assert!(duplicate.validate().is_err());
        let mut invalid = config.clone();
        invalid.urls = HashMap::from([(
            "https://worker".into(),
            "https://secret@worker/metrics".into(),
        )]);
        assert!(invalid.validate().is_err());
        assert!(serde_json::from_str::<BackendMetricsConfig>(
            r#"{"model":"local","urls":{"http://worker":"http://worker/metrics"},"unexpected":true}"#
        )
        .is_err());
        let collector = BackendLoadCollector {
            samples: Arc::new(RwLock::new(HashMap::from([(
                "https://worker".into(),
                Sample {
                    running: 2,
                    waiting: 1,
                    kv_usage_fraction: None,
                    started: Instant::now() - Duration::from_secs(1),
                },
            )]))),
            tasks: vec![],
            max_age: Duration::from_millis(10),
        };
        assert!(collector.snapshot("https://worker/").is_none());
        assert!(collector.snapshot("https://unknown").is_none());
    }

    #[tokio::test]
    async fn polling_failure_removes_previously_valid_load() {
        let fail = Arc::new(AtomicBool::new(false));
        let handler_fail = fail.clone();
        let app = axum::Router::new().route(
            "/metrics",
            axum::routing::get(move || {
                let fail = handler_fail.clone();
                async move {
                    if fail.load(Ordering::SeqCst) {
                        (axum::http::StatusCode::SERVICE_UNAVAILABLE, "unavailable")
                    } else {
                        (axum::http::StatusCode::OK, VALID)
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let collector = BackendLoadCollector::spawn(
            reqwest::Client::new(),
            BackendMetricsConfig {
                model: "local".into(),
                urls: HashMap::from([(base.clone(), format!("{base}/metrics"))]),
                poll_interval_ms: 10,
                max_age_ms: 60_000,
                ..Default::default()
            },
        );
        tokio::time::timeout(Duration::from_secs(2), async {
            while collector.snapshot(&base).is_none() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let sample = collector.snapshot(&format!("{base}/")).unwrap();
        assert_eq!((sample.running, sample.waiting), (2, 1));
        fail.store(true, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(2), async {
            while collector.snapshot(&base).is_some() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let task_abort_handles: Vec<_> = collector
            .tasks
            .iter()
            .map(|task| task.abort_handle())
            .collect();
        drop(collector);
        tokio::task::yield_now().await;
        assert!(task_abort_handles.iter().all(|task| task.is_finished()));
        server.abort();
    }

    #[tokio::test]
    async fn bounds_chunked_metrics_body_without_content_length() {
        let app = axum::Router::new().route(
            "/metrics",
            axum::routing::get(|| async {
                let chunks = [
                    Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b'x'; MAX_METRICS_BYTES])),
                    Ok(bytes::Bytes::from_static(b"x")),
                ];
                axum::body::Body::from_stream(futures_util::stream::iter(chunks))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/metrics", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        assert_eq!(
            scrape(&reqwest::Client::new(), &url, "local", 2000).await,
            Err("metrics_body_too_large")
        );
        server.abort();
    }
}
