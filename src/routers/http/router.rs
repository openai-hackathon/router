use crate::config::types::RetryConfig;
use crate::core::dispatch::{DispatchLedger, Reservation};
use crate::core::{
    is_retryable_status, BasicWorker, CircuitBreakerConfig, DPAwareWorker, HealthConfig,
    RetryExecutor, Worker, WorkerRegistry, WorkerType,
};
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, GenerationRequest,
    InferenceGenerateRequest, RerankRequest, RerankResponse, RerankResult, ResponsesRequest,
};
use crate::routers::header_utils;
use crate::routers::http::dp_utils;
use crate::routers::{RouterTrait, WorkerManagement};
use crate::routing_state::{
    config::RoutingConfig,
    features::{self, RequestFeatures},
    lmcache,
    telemetry::Collectors,
    SharedRoutingState,
};
use axum::body::to_bytes;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_LENGTH, header::CONTENT_TYPE, HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use reqwest::Client;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Regular router that uses injected load balancing policies
#[derive(Debug)]
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    worker_startup_timeout_secs: u64,
    worker_startup_check_interval_secs: u64,
    health_endpoint: String,
    intra_node_data_parallel_size: usize,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    routing_state: Arc<SharedRoutingState>,
    feature_client: Client,
    lookup_client: Client,
    _collectors: Option<Collectors>,
}

impl Router {
    /// Create a new router with injected policy and client
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        worker_urls: Vec<String>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Update active workers gauge
        RouterMetrics::set_active_workers(worker_urls.len());

        // Wait for workers to be healthy (skip if empty - for service discovery mode)
        if !worker_urls.is_empty() {
            Self::wait_for_healthy_workers_async(
                &worker_urls,
                ctx.router_config.worker_startup_timeout_secs,
                ctx.router_config.worker_startup_check_interval_secs,
                Some(ctx.client.clone()),
                &ctx.router_config.health_check.endpoint,
                ctx.router_config.health_check.timeout_secs,
            )
            .await?;
        }

        // Automatically expand to DP-aware workers when intra_node_data_parallel_size > 1
        let worker_urls = if ctx.router_config.intra_node_data_parallel_size > 1 {
            // worker address now in the format of "http://host:port@dp_rank"
            dp_utils::get_dp_aware_workers(
                &worker_urls,
                &ctx.router_config.api_key,
                ctx.router_config.intra_node_data_parallel_size,
            )
            .await
            .map_err(|e| format!("Failed to get dp-aware workers: {}", e))?
        } else {
            worker_urls
        };

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Register workers in the registry
        // In IGW mode, we need to fetch model info from workers
        let dp_size = ctx.router_config.intra_node_data_parallel_size;
        let health_config = HealthConfig {
            timeout_secs: ctx.router_config.health_check.timeout_secs,
            check_interval_secs: ctx.router_config.health_check.check_interval_secs,
            endpoint: ctx.router_config.health_check.endpoint.clone(),
            failure_threshold: ctx.router_config.health_check.failure_threshold,
            success_threshold: ctx.router_config.health_check.success_threshold,
        };
        for url in &worker_urls {
            // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
            // For now, create worker without model_id
            let worker_arc: Arc<dyn Worker> = if dp_size > 1 {
                let (base_url, dp_rank) = dp_utils::parse_worker_url(url);
                Arc::new(
                    DPAwareWorker::new(
                        base_url,
                        dp_rank.unwrap_or(0),
                        dp_size,
                        WorkerType::Regular,
                    )
                    .with_circuit_breaker_config(core_cb_config.clone())
                    .with_health_config(health_config.clone()),
                )
            } else {
                Arc::new(
                    BasicWorker::new(url.clone(), WorkerType::Regular)
                        .with_circuit_breaker_config(core_cb_config.clone())
                        .with_health_config(health_config.clone()),
                )
            };
            ctx.worker_registry.register(worker_arc.clone());

            // Notify PolicyRegistry about the new worker
            let model_id = worker_arc.model_id();
            let policy = ctx.policy_registry.on_worker_added(model_id, None);

            // If this is a cache-aware policy and it's the first worker for this model,
            // initialize it with the worker
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy
                    .as_any()
                    .downcast_ref::<crate::policies::CacheAwarePolicy>()
                {
                    let worker_dyn: Arc<dyn Worker> = worker_arc.clone();
                    cache_aware.init_workers(std::slice::from_ref(&worker_dyn));
                }
            }
        }

        let routing_config =
            RoutingConfig::load(ctx.router_config.routing_state_config.as_deref())?;
        let routing_state = Arc::new(SharedRoutingState::new(
            routing_config.clone(),
            Arc::new(DispatchLedger::default()),
        ));
        let headers = routing_config.headers()?;
        let feature_client = Client::builder()
            .default_headers(headers.clone())
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_millis(routing_config.render_timeout_ms))
            .build()
            .map_err(|e| e.to_string())?;
        let telemetry_client = Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_millis(routing_config.telemetry_timeout_ms))
            .build()
            .map_err(|e| e.to_string())?;
        let collectors = if routing_config.lmcache.is_none()
            && (ctx.policy_registry.get_default_policy().ranking().is_some()
                || ctx.router_config.routing_state_config.is_some())
        {
            Some(Collectors::start(
                ctx.worker_registry.clone(),
                &routing_state,
                telemetry_client.clone(),
            ))
        } else {
            None
        };

        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: ctx
                .router_config
                .worker_startup_check_interval_secs,
            health_endpoint: ctx.router_config.health_check.endpoint.clone(),
            intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
            routing_state,
            feature_client,
            lookup_client: telemetry_client,
            _collectors: collectors,
        })
    }

    /// Get the current list of worker URLs
    pub fn get_worker_urls(&self) -> Vec<String> {
        self.worker_registry.get_all_urls()
    }

    /// Get worker URLs for a specific model
    pub fn get_worker_urls_for_model(&self, model_id: Option<&str>) -> Vec<String> {
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };
        workers.iter().map(|w| w.url().to_string()).collect()
    }

    pub async fn wait_for_healthy_workers(
        worker_urls: &[String],
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval_secs: u64,
    ) -> Result<(), String> {
        if worker_urls.is_empty() {
            return Err(
                "Timeout waiting for workers to become healthy: no workers provided".to_string(),
            );
        }

        // Perform health check asynchronously
        Self::wait_for_healthy_workers_async(
            worker_urls,
            worker_startup_timeout_secs,
            worker_startup_check_interval_secs,
            None,
            "/health",
            2,
        )
        .await
    }

    async fn wait_for_healthy_workers_async(
        worker_urls: &[String],
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval_secs: u64,
        client: Option<Client>,
        health_endpoint: &str,
        timeout_secs: u64,
    ) -> Result<(), String> {
        // Extract unique base URLs (hosts) for health checks
        // This deduplicates DP-aware URLs like http://host:8081@0, @1, @2, @3
        // to only check http://host:8081 once
        use std::collections::HashSet;
        let mut unique_hosts = HashSet::new();
        let mut host_to_workers: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for url in worker_urls {
            // Extract base URL by removing @rank suffix if present
            let base_url = if let Some(at_pos) = url.rfind('@') {
                url[..at_pos].to_string()
            } else {
                url.clone()
            };

            unique_hosts.insert(base_url.clone());
            host_to_workers
                .entry(base_url)
                .or_default()
                .push(url.clone());
        }

        let unique_hosts_vec: Vec<String> = unique_hosts.into_iter().collect();

        info!(
            "Waiting for {} unique hosts (representing {} workers) to become healthy (timeout: {}s)",
            unique_hosts_vec.len(),
            worker_urls.len(),
            worker_startup_timeout_secs
        );

        let start_time = std::time::Instant::now();
        let client = client.unwrap_or_default();

        loop {
            if start_time.elapsed() > Duration::from_secs(worker_startup_timeout_secs) {
                error!(
                    "Timeout {}s waiting for hosts {:?} to become healthy. Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value",
                    worker_startup_timeout_secs, unique_hosts_vec
                );
                return Err(format!(
                    "Timeout {}s waiting for hosts {:?} to become healthy. Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value",
                    worker_startup_timeout_secs, unique_hosts_vec
                ));
            }

            // Perform health checks only on unique hosts (not per DP rank)
            let mut health_checks = Vec::new();
            for base_url in &unique_hosts_vec {
                let client_clone = client.clone();
                let url_clone = base_url.clone();
                let health_url = format!("{}{}", url_clone.trim_end_matches('/'), health_endpoint);

                let check_health = tokio::spawn(async move {
                    match client_clone
                        .get(&health_url)
                        .timeout(Duration::from_secs(timeout_secs))
                        .send()
                        .await
                    {
                        Ok(res) => {
                            if res.status().is_success() {
                                None
                            } else {
                                Some((url_clone, format!("status: {}", res.status())))
                            }
                        }
                        Err(_) => Some((url_clone, "not ready".to_string())),
                    }
                });

                health_checks.push(check_health);
            }

            // Wait for all health checks to complete
            let results = futures::future::join_all(health_checks).await;

            let mut unhealthy_hosts = Vec::new();
            let mut healthy_host_count = 0;

            for result in results {
                match result {
                    Ok(None) => {
                        healthy_host_count += 1;
                        // Host is healthy
                    }
                    Ok(Some((url, reason))) => {
                        unhealthy_hosts.push((url, reason));
                    }
                    Err(e) => {
                        unhealthy_hosts.push(("unknown".to_string(), format!("task error: {}", e)));
                    }
                }
            }

            if healthy_host_count > 0 {
                info!(
                    "{} out of {} unique hosts are healthy (representing {} workers)",
                    healthy_host_count,
                    unique_hosts_vec.len(),
                    worker_urls.len()
                );
                return Ok(());
            } else {
                debug!(
                   "Waiting for at least 1 of {} unique hosts to become healthy ({} unhealthy: {:?})",
                    unique_hosts_vec.len(),
                    unhealthy_hosts.len(),
                    unhealthy_hosts
                );
                tokio::time::sleep(Duration::from_secs(worker_startup_check_interval_secs)).await;
            }
        }
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(workers[0].url().to_string())
        }
    }

    #[allow(dead_code)]
    fn select_first_worker_for_model(&self, model_id: Option<&str>) -> Result<String, String> {
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };
        if workers.is_empty() {
            Err(format!(
                "No workers are available for model: {:?}",
                model_id
            ))
        } else {
            Ok(workers[0].url().to_string())
        }
    }

    pub async fn send_health_check(&self, worker_url: &str) -> Response {
        let health_url = if self.intra_node_data_parallel_size > 1 {
            // Need to extract the URL from "http://host:port@dp_rank"
            match dp_utils::extract_dp_rank(worker_url) {
                Ok((worker_url_prefix, _dp_rank)) => worker_url_prefix,
                Err(e) => {
                    error!("Failed to extract dp_rank for health check: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to extract dp_rank: {}", e),
                    )
                        .into_response();
                }
            }
        } else {
            worker_url
        };

        let request_builder = self.client.get(format!(
            "{}{}",
            health_url.trim_end_matches('/'),
            self.health_endpoint
        ));

        let response = match request_builder.send().await {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                match res.bytes().await {
                    Ok(body) => (status, body).into_response(),
                    Err(e) => {
                        error!(
                            worker_url = %health_url,
                            error = %e,
                            "Failed to read health response body"
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Err(e) => {
                error!(
                    worker_url = %health_url,
                    error = %e,
                    "Failed to send health request to worker"
                );
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to send request to worker {}: {}", health_url, e),
                )
                    .into_response()
            }
        };

        // Don't record metrics for health checks
        response
    }

    // Helper method to proxy GET requests to the first available worker
    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let incoming_headers = req.headers();
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let (base_url, dp_rank) = dp_utils::parse_worker_url(&worker_url);
                let url = format!("{}/{}", base_url, endpoint);
                let route_name = format!("/{}", endpoint);
                let mut request_builder =
                    dp_utils::add_dp_rank_header(self.client.get(&url), dp_rank);

                for (name, value) in headers {
                    let name_lc = name.to_lowercase();
                    // When the router selects a DP rank, it owns the
                    // X-data-parallel-rank header: skip any client-supplied
                    // value so the worker sees exactly one rank (ours).
                    if name_lc != "content-type"
                        && name_lc != "content-length"
                        && !(dp_rank.is_some() && name_lc == "x-data-parallel-rank")
                        && !header_utils::TRACE_HEADER_NAMES.contains(&name_lc.as_str())
                    {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match otel_http::send_client_request(
                    request_builder,
                    Some(incoming_headers),
                    ClientRequestOptions {
                        method: "GET",
                        url: &url,
                        route: Some(&route_name),
                        request_phase: None,
                    },
                )
                .await
                {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(axum::body::Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to read response: {}", e),
                            )
                                .into_response(),
                        }
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Request failed: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
        }
    }

    /// Convert axum HeaderMap to policy RequestHeaders (HashMap<String, String>)
    fn headers_to_request_headers(
        headers: Option<&HeaderMap>,
    ) -> Option<crate::policies::RequestHeaders> {
        headers.map(|h| {
            h.iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                })
                .collect()
        })
    }

    /// Select worker for a specific model considering circuit breaker state
    fn reserve_for_model(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        headers: Option<&HeaderMap>,
        features: &RequestFeatures,
        observations: Option<&lmcache::Observations>,
    ) -> Option<Reservation> {
        // Get workers for the specified model (O(1) lookup if model_id is provided)
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Convert headers for policies that need them (e.g., consistent_hash)
        let request_headers = Self::headers_to_request_headers(headers);

        self.routing_state.reserve_with_observations(
            &available,
            policy,
            features,
            text,
            request_headers.as_ref(),
            observations,
        )
    }

    #[cfg(test)]
    fn select_worker_for_model(
        &self,
        model: Option<&str>,
        text: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Option<Reservation> {
        self.reserve_for_model(
            model,
            text,
            headers,
            &RequestFeatures::unsupported(&serde_json::Value::Null, headers),
            None,
        )
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &str,
        model_id: Option<&str>,
    ) -> Response {
        let body = match serde_json::to_value(typed_req) {
            Ok(body) => body,
            Err(_) => return (StatusCode::BAD_REQUEST, "Invalid request body").into_response(),
        };
        self.route_json_request(
            headers,
            &body,
            route,
            model_id,
            typed_req.is_stream(),
            &typed_req.extract_text_for_routing(),
        )
        .await
    }

    async fn route_json_request(
        &self,
        headers: Option<&HeaderMap>,
        body: &serde_json::Value,
        route: &str,
        model_id: Option<&str>,
        is_stream: bool,
        text: &str,
    ) -> Response {
        let start = Instant::now();
        let policy = model_id.map_or_else(
            || self.policy_registry.get_default_policy(),
            |m| self.policy_registry.get_policy_or_default(m),
        );
        let features = if policy.ranking().is_some() {
            if let Some(config) = &self.routing_state.config.lmcache {
                lmcache::build_features(&self.feature_client, config, route, body, headers).await
            } else {
                features::build(
                    &self.feature_client,
                    self.routing_state.renderer_url().as_deref(),
                    route,
                    body,
                    headers,
                )
                .await
            }
        } else {
            RequestFeatures::unsupported(body, headers)
        };
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                // Controller I/O finishes before selection/reservation. Refresh
                // on each retry; the selector re-reads current loads and ages.
                let observations = if policy.ranking().is_some() {
                    if let Some(config) = &self.routing_state.config.lmcache {
                        let workers = match model_id {
                            Some(model) => self.worker_registry.get_by_model_fast(model),
                            None => self.worker_registry.get_all(),
                        };
                        Some(
                            lmcache::lookup_workers(
                                &self.lookup_client,
                                config,
                                &workers,
                                &features,
                            )
                            .await,
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };
                let reservation = match self.reserve_for_model(
                    model_id,
                    Some(text),
                    headers,
                    &features,
                    observations.as_ref(),
                ) {
                    Some(w) => w,
                    None => {
                        RouterMetrics::record_request_error(route, "no_available_workers");
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "No available workers (all circuits open or unhealthy)",
                        )
                            .into_response();
                    }
                };

                let worker = reservation.worker.clone();
                let response = self
                    .send_typed_request(headers, body, route, worker.url(), is_stream, reservation)
                    .await;
                let status = response.status();
                worker.record_outcome(status.is_success() || status.is_client_error());

                response
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                RouterMetrics::record_retry(route);
                RouterMetrics::record_retry_backoff_duration(delay, attempt);
            },
            // on_exhausted hook
            || RouterMetrics::record_retries_exhausted(route),
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            RouterMetrics::record_request(route);
            RouterMetrics::record_generate_duration(duration);
        } else if !is_retryable_status(response.status()) {
            RouterMetrics::record_request_error(route, "non_retryable_error");
        }

        response
    }

    // Helper: return base worker URL (strips DP suffix when enabled)
    fn worker_base_url(&self, worker_url: &str) -> String {
        if self.intra_node_data_parallel_size > 1 {
            if let Ok((prefix, _)) = dp_utils::extract_dp_rank(worker_url) {
                return prefix.to_string();
            }
        }
        worker_url.to_string()
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the vllm worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let worker_urls = self.get_worker_urls();
        if worker_urls.is_empty() {
            return (StatusCode::SERVICE_UNAVAILABLE, "No available workers").into_response();
        }

        let mut last_response: Option<Response> = None;
        for worker_url in worker_urls {
            let base = self.worker_base_url(&worker_url);

            let url = format!("{}/{}", base, endpoint);
            let route_name = format!("/{}", endpoint);
            let method_name = method.as_str().to_string();
            let mut request_builder = match method.clone() {
                Method::GET => self.client.get(&url),
                Method::POST => self.client.post(&url),
                _ => {
                    return (
                        StatusCode::METHOD_NOT_ALLOWED,
                        "Unsupported method for simple routing",
                    )
                        .into_response()
                }
            };

            if let Some(hdrs) = headers {
                for (name, value) in hdrs {
                    let name_lc = name.as_str().to_lowercase();
                    if name_lc != "content-type"
                        && name_lc != "content-length"
                        && !header_utils::TRACE_HEADER_NAMES.contains(&name_lc.as_str())
                    {
                        request_builder = request_builder.header(name, value);
                    }
                }
            }

            match otel_http::send_client_request(
                request_builder,
                headers,
                ClientRequestOptions {
                    method: &method_name,
                    url: &url,
                    route: Some(&route_name),
                    request_phase: None,
                },
            )
            .await
            {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    let response_headers = header_utils::preserve_response_headers(res.headers());
                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(axum::body::Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;
                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    format!("Failed to read response: {}", e),
                                )
                                    .into_response(),
                            );
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Request failed: {}", e),
                        )
                            .into_response(),
                    );
                }
            }
        }

        last_response
            .unwrap_or_else(|| (StatusCode::BAD_GATEWAY, "No worker response").into_response())
    }

    // Route a GET request with provided headers to a specific endpoint
    async fn route_get_request(&self, headers: Option<&HeaderMap>, endpoint: &str) -> Response {
        self.route_simple_request(headers, endpoint, Method::GET)
            .await
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &str,
        worker_url: &str,
        is_stream: bool,
        mut reservation: Reservation,
    ) -> Response {
        let (mut request_builder, extracted_dp_rank, request_url) =
            if self.intra_node_data_parallel_size > 1 {
                let (worker_url_prefix, dp_rank) = match dp_utils::extract_dp_rank(worker_url) {
                    Ok(tup) => tup,
                    Err(e) => {
                        error!("Failed to extract dp_rank: {}", e);
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to extract dp_rank: {}", e),
                        )
                            .into_response();
                    }
                };

                // Parse the request body
                let json_val = match serde_json::to_value(typed_req) {
                    Ok(j) => j,
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!("Convert into serde_json::Value failed: {}", e),
                        )
                            .into_response();
                    }
                };

                // Use the original json_val without modification

                let request_url = format!("{}{}", worker_url_prefix, route);
                (
                    self.client.post(&request_url).json(&json_val),
                    Some(dp_rank),
                    request_url,
                )
            } else {
                let request_url = format!("{}{}", worker_url, route);
                (
                    self.client.post(&request_url).json(typed_req),
                    None,
                    request_url,
                )
            };

        // Copy all headers from original request if provided, skipping
        // Content-Type/Content-Length (.json() sets them) and trace headers
        // (propagate_trace_headers below injects fresh context).
        if let Some(headers) = headers {
            for (name, value) in headers {
                if *name != CONTENT_TYPE
                    && name != "host"
                    && *name != CONTENT_LENGTH
                    && !header_utils::TRACE_HEADER_NAMES
                        .iter()
                        .any(|&th| name.as_str().eq_ignore_ascii_case(th))
                {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        // Add X-data-parallel-rank header for DP-aware routing
        if let Some(dp_rank) = extracted_dp_rank {
            request_builder = request_builder.header("X-data-parallel-rank", dp_rank.to_string());
        }

        reservation.dispatched();
        let res = match otel_http::send_client_request(
            request_builder,
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &request_url,
                route: Some(route),
                request_phase: Some("inference"),
            },
        )
        .await
        {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                if e.is_connect() || e.is_builder() {
                    reservation.finish(false);
                }

                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request failed: {}", e),
                )
                    .into_response();
            }
        };

        self.routing_state
            .validate_response_identity(worker_url, res.headers());
        let background = serde_json::to_value(typed_req)
            .ok()
            .and_then(|v| v.get("background").and_then(|v| v.as_bool()))
            .unwrap_or(false);
        super::response_lifecycle::forward(res, reservation, is_stream, background).await
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .default_headers(self.routing_state.config.headers()?)
            .timeout(Duration::from_secs(self.worker_startup_timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(self.worker_startup_timeout_secs) {
                error!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value",
                    self.worker_startup_timeout_secs, worker_url
                );
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value",
                    self.worker_startup_timeout_secs, worker_url
                ));
            }

            match client
                .get(format!(
                    "{}{}",
                    worker_url.trim_end_matches('/'),
                    self.health_endpoint
                ))
                .send()
                .await
            {
                Ok(res) => {
                    if res.status().is_success() {
                        if self.intra_node_data_parallel_size > 1 {
                            // Expand worker URL into multiple DP-aware URLs based on configured intra_node_data_parallel_size
                            // (e.g., "http://host:8000" → "http://host:8000@0", "@1", etc.)
                            // without querying the worker
                            let url_vec = vec![String::from(worker_url)];
                            let dp_url_vec = dp_utils::get_dp_aware_workers(
                                &url_vec,
                                &self.api_key,
                                self.intra_node_data_parallel_size,
                            )
                            .await
                            .map_err(|e| format!("Failed to get dp-aware workers: {}", e))?;
                            let mut worker_added: bool = false;
                            for dp_url in &dp_url_vec {
                                if self.worker_registry.get_by_url(dp_url).is_some() {
                                    warn!("Worker {} already exists", dp_url);
                                    continue;
                                }
                                info!("Added worker: {}", dp_url);
                                // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
                                let (base_url, dp_rank) = dp_utils::parse_worker_url(dp_url);
                                let new_worker = DPAwareWorker::new(
                                    base_url,
                                    dp_rank.unwrap_or(0),
                                    self.intra_node_data_parallel_size,
                                    WorkerType::Regular,
                                )
                                .with_circuit_breaker_config(self.circuit_breaker_config.clone());

                                let worker_arc: Arc<dyn Worker> = Arc::new(new_worker);
                                self.worker_registry.register(worker_arc.clone());

                                // Notify PolicyRegistry about the new worker
                                let model_id = worker_arc.model_id();
                                let policy = self.policy_registry.on_worker_added(model_id, None);

                                // If this is a cache-aware policy, update it with all workers for this model
                                if policy.name() == "cache_aware" {
                                    if let Some(cache_aware) = policy
                                        .as_any()
                                        .downcast_ref::<crate::policies::CacheAwarePolicy>(
                                    ) {
                                        let model_workers =
                                            self.worker_registry.get_by_model_fast(model_id);
                                        cache_aware.init_workers(&model_workers);
                                    }
                                }

                                worker_added = true;
                            }
                            if !worker_added {
                                return Err(format!("No worker added for {}", worker_url));
                            }
                        } else {
                            if self.worker_registry.get_by_url(worker_url).is_some() {
                                return Err(format!("Worker {} already exists", worker_url));
                            }
                            info!("Added worker: {}", worker_url);

                            // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
                            let new_worker =
                                BasicWorker::new(worker_url.to_string(), WorkerType::Regular)
                                    .with_circuit_breaker_config(
                                        self.circuit_breaker_config.clone(),
                                    );

                            let worker_arc = Arc::new(new_worker);
                            self.worker_registry.register(worker_arc.clone());

                            // Notify PolicyRegistry about the new worker
                            let model_id = worker_arc.model_id();
                            let policy = self.policy_registry.on_worker_added(model_id, None);

                            // If this is a cache-aware policy, add this worker to it
                            if policy.name() == "cache_aware" {
                                if let Some(cache_aware) = policy
                                    .as_any()
                                    .downcast_ref::<crate::policies::CacheAwarePolicy>(
                                ) {
                                    // Get all workers for this model
                                    let model_workers =
                                        self.worker_registry.get_by_model_fast(model_id);
                                    cache_aware.init_workers(&model_workers);
                                }
                            }
                        }

                        RouterMetrics::set_active_workers(self.worker_registry.get_all().len());

                        return Ok(format!("Successfully added worker: {}", worker_url));
                    } else {
                        debug!(
                            "Worker {} health check pending - status: {}",
                            worker_url,
                            res.status()
                        );
                        // if the url does not have http or https prefix, warn users
                        if !worker_url.starts_with("http://") && !worker_url.starts_with("https://")
                        {
                            warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                        }

                        tokio::time::sleep(Duration::from_secs(
                            self.worker_startup_check_interval_secs,
                        ))
                        .await;
                        continue;
                    }
                }
                Err(e) => {
                    debug!("Worker {} health check pending - error: {}", worker_url, e);

                    // if the url does not have http or https prefix, warn users
                    if !worker_url.starts_with("http://") && !worker_url.starts_with("https://") {
                        warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                    }

                    tokio::time::sleep(Duration::from_secs(
                        self.worker_startup_check_interval_secs,
                    ))
                    .await;
                    continue;
                }
            }
        }
    }

    pub fn remove_worker(&self, worker_url: &str) {
        if self.intra_node_data_parallel_size > 1 {
            // remove dp-aware workers in a prefix-matching fashion
            // without contacting the remote worker
            let mut removed_workers: Vec<String> = Vec::new();
            let worker_url_prefix = format!("{}@", worker_url);

            // Find and remove all workers with matching prefix
            let all_workers = self.worker_registry.get_all();
            for w in all_workers.iter() {
                if w.url().starts_with(&worker_url_prefix) {
                    // Get model_id before removing
                    let model_id = w.model_id().to_string();

                    if self.worker_registry.remove_by_url(w.url()).is_some() {
                        info!("Removed worker: {}", w.url());
                        removed_workers.push(w.url().to_string());

                        // Notify PolicyRegistry about the removed worker
                        self.policy_registry.on_worker_removed(&model_id);
                    } else {
                        warn!("Worker {} not found, skipping removal", w.url());
                    }
                }
            }

            RouterMetrics::set_active_workers(self.worker_registry.get_all().len());

            // If any models are using cache aware policy, remove the workers from the tree
            // Check each removed worker's model and get its policy
            for dp_url in removed_workers.iter() {
                if let Some(worker) = self.worker_registry.get_by_url(dp_url) {
                    let model_id = worker.model_id();
                    if let Some(policy) = self.policy_registry.get_policy(model_id) {
                        if let Some(cache_aware) = policy
                            .as_any()
                            .downcast_ref::<crate::policies::CacheAwarePolicy>()
                        {
                            cache_aware.remove_worker_by_url(dp_url);
                            info!("Removed worker from cache-aware tree: {}", dp_url);
                        }
                    }
                }
            }
        } else {
            // Get the worker first to extract model_id
            let model_id = if let Some(worker) = self.worker_registry.get_by_url(worker_url) {
                worker.model_id().to_string()
            } else {
                warn!("Worker {} not found, skipping removal", worker_url);
                return;
            };

            if self.worker_registry.remove_by_url(worker_url).is_some() {
                info!("Removed worker: {}", worker_url);

                // Notify PolicyRegistry about the removed worker
                self.policy_registry.on_worker_removed(&model_id);

                RouterMetrics::set_active_workers(self.worker_registry.get_all().len());
            }

            // If the model is using cache aware policy, remove the worker from the tree
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if let Some(cache_aware) = policy
                    .as_any()
                    .downcast_ref::<crate::policies::CacheAwarePolicy>()
                {
                    cache_aware.remove_worker_by_url(worker_url);
                    info!("Removed worker from cache-aware tree: {}", worker_url);
                }
            }
        }
    }

    async fn build_rerank_response(
        req: &RerankRequest,
        response: Response,
    ) -> anyhow::Result<Response> {
        let (_, response_body) = response.into_parts();
        let body_bytes = to_bytes(response_body, usize::MAX).await?;
        let rerank_results = serde_json::from_slice::<Vec<RerankResult>>(&body_bytes)?;
        let mut rerank_response =
            RerankResponse::new(rerank_results, req.model.clone(), req.rid.clone());
        rerank_response.sort_by_score();
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Ok(Json(rerank_response).into_response())
    }
}

use async_trait::async_trait;

#[async_trait]
impl WorkerManagement for Router {
    async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        Router::add_worker(self, worker_url).await
    }

    fn remove_worker(&self, worker_url: &str) {
        Router::remove_worker(self, worker_url)
    }

    fn get_worker_urls(&self) -> Vec<String> {
        Router::get_worker_urls(self)
    }
}

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        let workers = self.worker_registry.get_all();
        let unhealthy_servers: Vec<_> = workers
            .iter()
            .filter(|w| !w.is_healthy())
            .map(|w| w.url().to_string())
            .collect();

        if unhealthy_servers.is_empty() {
            (StatusCode::OK, "All servers healthy").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Unhealthy servers: {:?}", unhealthy_servers),
            )
                .into_response()
        }
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_inference_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &InferenceGenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/inference/v1/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn get_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_get_request(headers, &endpoint).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Record embeddings-specific metrics in addition to general request metrics
        let start = Instant::now();
        let res = self
            .route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await;

        // Embedding specific metrics
        if res.status().is_success() {
            RouterMetrics::record_embeddings_request();
            RouterMetrics::record_embeddings_duration(start.elapsed());
        } else {
            let error_type = format!("http_{}", res.status().as_u16());
            RouterMetrics::record_embeddings_error(&error_type);
        }

        res
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        if let Err(e) = body.validate() {
            return (StatusCode::BAD_REQUEST, e).into_response();
        }
        let response = self
            .route_typed_request(headers, body, "/v1/rerank", model_id)
            .await;
        if response.status().is_success() {
            match Self::build_rerank_response(body, response).await {
                Ok(rerank_response) => rerank_response,
                Err(e) => {
                    error!("Failed to build rerank response: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to build rerank response".to_string(),
                    )
                        .into_response();
                }
            }
        } else {
            response
        }
    }

    async fn flush_cache(&self) -> Response {
        // Get all worker URLs
        let worker_urls = self.get_worker_urls();

        // Send requests to all workers concurrently without headers
        let mut tasks = Vec::new();
        for worker_url in &worker_urls {
            let worker_url = if self.intra_node_data_parallel_size > 1 {
                // Need to extract the URL from "http://host:port@dp_rank"
                let (worker_url_prefix, _dp_rank) = match dp_utils::extract_dp_rank(worker_url) {
                    Ok(tup) => tup,
                    Err(e) => {
                        error!("Failed to extract dp_rank: {}", e);
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to extract dp_rank: {}", e),
                        )
                            .into_response();
                    }
                };
                worker_url_prefix
            } else {
                worker_url
            };
            let request_builder = self.client.post(format!("{}/flush_cache", worker_url));
            tasks.push(request_builder.send());
        }

        // Wait for all responses
        let results = futures_util::future::join_all(tasks).await;

        // Check if all succeeded
        let all_success = results.iter().all(|r| {
            r.as_ref()
                .map(|res| res.status().is_success())
                .unwrap_or(false)
        });

        if all_success {
            (StatusCode::OK, "Cache flushed on all servers").into_response()
        } else {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Cache flush failed on one or more servers",
            )
                .into_response()
        }
    }

    async fn get_worker_loads(&self) -> Response {
        let urls = self.get_worker_urls();
        let mut loads = Vec::new();

        // Get loads from all workers
        for url in &urls {
            let load = self.worker_registry.get_by_url(url).map_or(0, |w| w.load());
            loads.push(serde_json::json!({
                "worker": url,
                "load": load
            }));
        }

        Json(serde_json::json!({
            "workers": loads
        }))
        .into_response()
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }

    fn readiness(&self) -> Response {
        // Regular router is ready if it has at least one healthy worker
        let workers = self.worker_registry.get_all();
        let healthy_count = workers.iter().filter(|w| w.is_healthy()).count();
        let total_workers = workers.len();

        if healthy_count > 0 {
            Json(serde_json::json!({
                "status": "ready",
                "healthy_workers": healthy_count,
                "total_workers": total_workers
            }))
            .into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "not_ready",
                    "reason": "no healthy workers available",
                    "total_workers": total_workers
                })),
            )
                .into_response()
        }
    }

    /// Route a transparent proxy request to a backend worker
    /// Forwards the request as-is to a selected worker
    async fn route_transparent(
        &self,
        headers: Option<&HeaderMap>,
        path: &str,
        method: &Method,
        body: serde_json::Value,
    ) -> Response {
        if *method == Method::POST
            && matches!(
                path,
                "/v1/chat/completions"
                    | "/v1/completions"
                    | "/v1/responses"
                    | "/inference/v1/generate"
            )
        {
            let is_stream = body
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let text = serde_json::to_string(&body).unwrap_or_default();
            return self
                .route_json_request(headers, &body, path, None, is_stream, &text)
                .await;
        }
        debug!("Transparent proxy: routing {} {} to backend", method, path);

        // Select a worker (filter by availability like select_worker_for_model)
        let all_workers = self.worker_registry.get_all();
        let workers: Vec<Arc<dyn Worker>> = all_workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if workers.is_empty() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "No available workers".to_string(),
            )
                .into_response();
        }

        let policy = self.policy_registry.get_default_policy();
        let request_text = serde_json::to_string(&body).ok();
        let request_headers = Self::headers_to_request_headers(headers);
        let features = RequestFeatures::unsupported(&body, headers);
        let mut reservation = match self.routing_state.reserve(
            &workers,
            policy,
            &features,
            request_text.as_deref(),
            request_headers.as_ref(),
        ) {
            Some(reservation) => reservation,
            None => {
                return (StatusCode::SERVICE_UNAVAILABLE, "Failed to select a worker")
                    .into_response()
            }
        };
        let worker = reservation.worker.clone();
        let url = worker.endpoint_url(path);

        debug!("Transparent proxy: forwarding to {}", url);

        // Build the request
        let mut request_builder = match *method {
            Method::GET => self.client.get(&url),
            Method::POST => self.client.post(&url),
            Method::PUT => self.client.put(&url),
            Method::DELETE => self.client.delete(&url),
            Method::PATCH => self.client.patch(&url),
            Method::HEAD => self.client.head(&url),
            _ => {
                return (
                    StatusCode::METHOD_NOT_ALLOWED,
                    format!("Method {} not supported", method),
                )
                    .into_response();
            }
        };

        // Add X-data-parallel-rank header for DP-aware routing
        request_builder = dp_utils::add_dp_rank_header(request_builder, worker.dp_rank());

        // Add JSON body if not null/empty
        if !body.is_null() {
            request_builder = request_builder.json(&body);
        }

        // Add authorization if configured
        if let Some(ref key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", key));
        }

        if let Some(headers) = headers {
            for (name, value) in headers {
                if *name != CONTENT_TYPE
                    && name != "host"
                    && *name != CONTENT_LENGTH
                    && name != "authorization"
                    && !header_utils::TRACE_HEADER_NAMES.contains(&name.as_str())
                {
                    request_builder = request_builder.header(name, value);
                }
            }
            if self.api_key.is_none() {
                if let Some(value) = headers.get("authorization") {
                    request_builder = request_builder.header("authorization", value);
                }
            }
        }
        let background = body
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        reservation.dispatched();
        // Send request
        match otel_http::send_client_request(
            request_builder,
            headers,
            ClientRequestOptions {
                method: method.as_str(),
                url: &url,
                route: Some(path),
                request_phase: Some("inference"),
            },
        )
        .await
        {
            Ok(response) => {
                worker.record_outcome(
                    response.status().is_success() || response.status().is_client_error(),
                );
                super::response_lifecycle::forward(response, reservation, true, background).await
            }
            Err(e) => {
                if e.is_connect() || e.is_builder() {
                    reservation.finish(false);
                }
                worker.record_outcome(false);
                (
                    StatusCode::BAD_GATEWAY,
                    format!("Backend request failed: {}", e),
                )
                    .into_response()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;
    use std::collections::HashMap;

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        // Register test workers
        let worker1 = BasicWorker::new("http://worker1:8080".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://worker2:8080".to_string(), WorkerType::Regular);
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        Router {
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            health_endpoint: "/health".into(),
            intra_node_data_parallel_size: 1,
            api_key: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            routing_state: Arc::new(SharedRoutingState::new(
                RoutingConfig::default(),
                Arc::new(DispatchLedger::default()),
            )),
            feature_client: Client::new(),
            lookup_client: Client::new(),
            _collectors: None,
        }
    }

    #[tokio::test]
    async fn observed_policies_use_raw_chat_and_responses_and_release_retries() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_handler = attempts.clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let inference = axum::routing::post(move |Json(body): Json<serde_json::Value>| {
            let attempts = attempts_handler.clone();
            async move {
                let current = attempts.fetch_add(1, Ordering::SeqCst);
                if current % 2 == 0 {
                    return (StatusCode::SERVICE_UNAVAILABLE, "rejected").into_response();
                }
                Json(body).into_response()
            }
        });
        let app = axum::Router::new()
            .route("/v1/chat/completions", inference.clone())
            .route("/v1/responses", inference);
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        for config in [
            crate::config::PolicyConfig::PrefixMax,
            crate::config::PolicyConfig::LeastLoadKv,
            crate::config::PolicyConfig::KvBatchEct,
        ] {
            let mut router = create_test_regular_router();
            router.worker_registry = Arc::new(WorkerRegistry::new());
            let worker: Arc<dyn Worker> =
                Arc::new(BasicWorker::new(url.clone(), WorkerType::Regular));
            router.worker_registry.register(worker.clone());
            router.policy_registry = Arc::new(PolicyRegistry::new(config));
            router.retry_config.max_retries = 2;
            for route in ["/v1/chat/completions", "/v1/responses"] {
                let body = serde_json::json!({"model":"local", "messages":[{"role":"user","content":"hello"}], "priority":0, "chat_template_kwargs":{"enable_thinking":false}, "custom_extension":{"preserve":true}});
                let response = router
                    .route_transparent(None, route, &Method::POST, body.clone())
                    .await;
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
                assert_eq!(
                    serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
                    body
                );
                assert_eq!(worker.load(), 0);
                assert_eq!(router.routing_state.ledger.unknown_count(worker.url()), 0);
            }
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 12);
        server.abort();
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let urls = router.get_worker_urls();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_empty_list() {
        // Empty list will return error immediately
        let result = Router::wait_for_healthy_workers(&[], 1, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no workers provided"));
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_invalid_urls() {
        // This test will timeout quickly since the URLs are invalid
        let result =
            Router::wait_for_healthy_workers(&["http://nonexistent:8080".to_string()], 1, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Timeout"));
    }

    // =============================
    // Tests for transparent proxy header/availability fixes
    // =============================

    /// Create a test router with ConsistentHash policy instead of RoundRobin
    fn create_test_consistent_hash_router() -> Router {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::ConsistentHash { virtual_nodes: 100 },
        ));

        let worker1 = BasicWorker::new("http://worker1:8080".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://worker2:8080".to_string(), WorkerType::Regular);
        let worker3 = BasicWorker::new("http://worker3:8080".to_string(), WorkerType::Regular);
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));
        worker_registry.register(Arc::new(worker3));

        Router {
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            health_endpoint: "/health".into(),
            intra_node_data_parallel_size: 1,
            api_key: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            routing_state: Arc::new(SharedRoutingState::new(
                RoutingConfig::default(),
                Arc::new(DispatchLedger::default()),
            )),
            feature_client: Client::new(),
            lookup_client: Client::new(),
            _collectors: None,
        }
    }

    #[test]
    fn test_headers_to_request_headers_basic() {
        // Test that headers_to_request_headers correctly converts HeaderMap to HashMap
        let mut header_map = HeaderMap::new();
        header_map.insert("x-session-id", HeaderValue::from_static("session-123"));
        header_map.insert("content-type", HeaderValue::from_static("application/json"));
        header_map.insert("X-Custom-Header", HeaderValue::from_static("custom-value"));

        let result = Router::headers_to_request_headers(Some(&header_map));
        assert!(result.is_some());
        let headers = result.unwrap();

        // All keys should be lowercased
        assert_eq!(headers.get("x-session-id").unwrap(), "session-123");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
        assert_eq!(headers.get("x-custom-header").unwrap(), "custom-value");
    }

    #[test]
    fn test_headers_to_request_headers_none() {
        // Test that None headers produce None output
        let result = Router::headers_to_request_headers(None);
        assert!(result.is_none());
    }

    #[test]
    fn test_headers_to_request_headers_empty() {
        // Test that empty HeaderMap produces empty HashMap
        let header_map = HeaderMap::new();
        let result = Router::headers_to_request_headers(Some(&header_map));
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_select_worker_for_model_with_consistent_hash_uses_headers() {
        // Verify that select_worker_for_model passes headers through to the policy,
        // producing consistent routing for the same session ID
        let router = create_test_consistent_hash_router();

        let mut header_map = HeaderMap::new();
        header_map.insert("x-session-id", HeaderValue::from_static("sticky-session-1"));

        // Make multiple selections with the same headers - should all pick the same worker
        let mut selected_urls: Vec<String> = Vec::new();
        for _ in 0..10 {
            let worker = router
                .select_worker_for_model(None, Some(r#"{"prompt": "test"}"#), Some(&header_map))
                .expect("Should select a worker");
            selected_urls.push(worker.worker.url().to_string());
        }

        // All selections should go to the same worker (sticky routing)
        let first = &selected_urls[0];
        for (i, url) in selected_urls.iter().enumerate() {
            assert_eq!(
                url, first,
                "Request {} routed to {}, expected {} (session stickiness broken)",
                i, url, first
            );
        }
    }

    #[test]
    fn test_select_worker_for_model_filters_unavailable_workers() {
        // Verify that select_worker_for_model skips unhealthy workers
        let router = create_test_consistent_hash_router();

        // Mark worker1 and worker2 as unhealthy, leaving only worker3
        let all_workers = router.worker_registry.get_all();
        for w in &all_workers {
            if w.url() == "http://worker1:8080" || w.url() == "http://worker2:8080" {
                w.set_healthy(false);
            }
        }

        let worker = router
            .select_worker_for_model(None, Some(r#"{"prompt": "test"}"#), None)
            .expect("Should select the remaining healthy worker");

        assert_eq!(
            worker.worker.url(),
            "http://worker3:8080",
            "Should only select the healthy worker"
        );
    }

    #[test]
    fn test_select_worker_for_model_returns_none_when_all_unavailable() {
        // Verify that when all workers are unhealthy, None is returned
        let router = create_test_consistent_hash_router();

        // Mark all workers as unhealthy
        let all_workers = router.worker_registry.get_all();
        for w in &all_workers {
            w.set_healthy(false);
        }

        let result = router.select_worker_for_model(None, Some(r#"{"prompt": "test"}"#), None);
        assert!(
            result.is_none(),
            "Should return None when all workers are unavailable"
        );
    }

    #[test]
    fn test_consistent_hash_different_sessions_can_route_differently() {
        // Verify that different session IDs can route to different workers
        let router = create_test_consistent_hash_router();

        let mut worker_urls_seen = std::collections::HashSet::new();
        for i in 0..50 {
            let mut header_map = HeaderMap::new();
            let session_id = format!("session-{}", i);
            header_map.insert("x-session-id", HeaderValue::from_str(&session_id).unwrap());

            if let Some(worker) = router.select_worker_for_model(
                None,
                Some(r#"{"prompt": "test"}"#),
                Some(&header_map),
            ) {
                worker_urls_seen.insert(worker.worker.url().to_string());
            }
        }

        // With 50 different sessions and 3 workers, we should see at least 2 workers used
        assert!(
            worker_urls_seen.len() >= 2,
            "Expected distribution across workers, only used: {:?}",
            worker_urls_seen
        );
    }

    #[test]
    fn test_inline_header_conversion_matches_headers_to_request_headers() {
        // Verify that the inline header conversion pattern used in vllm_pd_router
        // produces the same result as Router::headers_to_request_headers.
        let mut header_map = HeaderMap::new();
        header_map.insert("X-Session-Id", HeaderValue::from_static("session-abc"));
        header_map.insert("Content-Type", HeaderValue::from_static("application/json"));
        header_map.insert("x-user-id", HeaderValue::from_static("user-42"));

        // Method 1: Router::headers_to_request_headers (used in router.rs)
        let method1 = Router::headers_to_request_headers(Some(&header_map)).unwrap();

        // Method 2: Inline conversion (used in vllm_pd_router.rs)
        let method2: HashMap<String, String> = header_map
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|v| (name.as_str().to_lowercase(), v.to_string()))
            })
            .collect();

        assert_eq!(
            method1, method2,
            "Both header conversion methods should produce identical results"
        );
    }

    /// Helper: start a minimal mock server that responds 200 on /health.
    async fn start_healthy_mock_server() -> (String, tokio::task::JoinHandle<()>) {
        use axum::{routing::get, Router as AxumRouter};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = AxumRouter::new().route("/health", get(|| async { "ok" }));
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_all_healthy() {
        let (url, _handle) = start_healthy_mock_server().await;
        let result = Router::wait_for_healthy_workers(&[url], 5, 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_partial_health() {
        // One healthy server + one unreachable URL.
        // The new behaviour succeeds when at least one host is healthy.
        let (healthy_url, _handle) = start_healthy_mock_server().await;
        let unreachable_url = "http://127.0.0.1:1".to_string(); // port 1 is unreachable

        let result = Router::wait_for_healthy_workers(&[healthy_url, unreachable_url], 5, 1).await;
        assert!(result.is_ok());
    }

    /// Helper: start a mock server that returns 503 on /health for a given
    /// duration, then switches to 200. Simulates a worker with a slow startup.
    async fn start_delayed_healthy_mock_server(
        delay: std::time::Duration,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use axum::{extract::State, http::StatusCode, routing::get, Router as AxumRouter};
        use std::sync::Arc;
        use tokio::net::TcpListener;

        let start = std::time::Instant::now();
        let ready_after = Arc::new(delay);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let app = AxumRouter::new()
            .route(
                "/health",
                get(
                    move |State((start, ready_after)): State<(
                        std::time::Instant,
                        Arc<std::time::Duration>,
                    )>| async move {
                        if start.elapsed() >= *ready_after {
                            StatusCode::OK
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        }
                    },
                ),
            )
            .with_state((start, ready_after));

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_dp_aware_dedup() {
        // DP-aware URLs like http://host:port@0, @1, @2 should be deduplicated
        // to a single /health check on http://host:port.
        let (base_url, _handle) = start_healthy_mock_server().await;
        let dp_urls: Vec<String> = (0..4)
            .map(|rank| format!("{}@{}", base_url, rank))
            .collect();

        let result = Router::wait_for_healthy_workers(&dp_urls, 5, 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_delayed_worker_becomes_routable_via_health_checker() {
        use crate::core::{BasicWorker, HealthConfig, WorkerRegistry, WorkerType};
        use std::sync::Arc;

        // Two workers: one immediately healthy, one delayed (503 for 2s, then 200).
        let (healthy_url, _h1) = start_healthy_mock_server().await;
        let (delayed_url, _h2) =
            start_delayed_healthy_mock_server(std::time::Duration::from_secs(2)).await;

        // Verify the delayed worker is genuinely unhealthy right now.
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{}/health", delayed_url))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 503);

        // ── Step 1: wait_for_healthy_workers (mirrors PdRouterBase::new startup) ──
        // This succeeds because the healthy worker responds immediately,
        // even though the delayed worker is still returning 503.
        let result =
            Router::wait_for_healthy_workers(&[healthy_url.clone(), delayed_url.clone()], 10, 1)
                .await;
        assert!(
            result.is_ok(),
            "Startup should succeed with at least one healthy worker"
        );

        // ── Step 2: register workers in the registry (mirrors PdRouterBase::new) ──
        let registry = Arc::new(WorkerRegistry::new());

        let healthy_worker = Arc::new(
            BasicWorker::new(healthy_url, WorkerType::Decode).with_health_config(HealthConfig {
                timeout_secs: 2,
                check_interval_secs: 1,
                endpoint: "/health".to_string(),
                failure_threshold: 3,
                success_threshold: 1,
            }),
        );
        registry.register(healthy_worker);

        let delayed_worker = Arc::new(
            BasicWorker::new(delayed_url, WorkerType::Decode).with_health_config(HealthConfig {
                timeout_secs: 2,
                check_interval_secs: 1,
                endpoint: "/health".to_string(),
                failure_threshold: 3,
                success_threshold: 1,
            }),
        );
        delayed_worker.set_healthy(false); // starts unhealthy
        registry.register(delayed_worker.clone());

        // Only the immediately-healthy worker should be available for routing.
        let healthy = registry.get_workers_filtered(None, None, None, true);
        assert_eq!(
            healthy.len(),
            1,
            "Only 1 worker should be healthy initially, got {}",
            healthy.len()
        );

        // ── Step 3: start background health checker (mirrors PdRouterBase::new) ──
        let health_checker = registry.start_health_checker(1);

        // ── Step 4: wait for delayed worker to recover ──
        // The mock server switches to 200 at t≈2s. With a 1s check interval
        // and success_threshold=1, the worker should be healthy by t≈3-4s.
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;

        // Both workers should now be available for routing.
        let healthy = registry.get_workers_filtered(None, None, None, true);
        assert_eq!(
            healthy.len(),
            2,
            "Both workers should be healthy after recovery, got {}",
            healthy.len()
        );
        assert!(
            delayed_worker.is_healthy(),
            "Delayed worker should have transitioned to healthy via health checker"
        );

        health_checker.shutdown().await;
    }
}
