use super::{
    cost::{CostModel, RestoreCostModel},
    lmcache::LmCacheConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// All URLs are explicit deployment URLs. Header values come from environment
/// variables, so the configuration and its Debug output never contain secrets.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RoutingConfig {
    pub renderer_url: Option<String>,
    pub header_env: HashMap<String, String>,
    pub poll_interval_ms: u64,
    pub telemetry_timeout_ms: u64,
    pub render_timeout_ms: u64,
    pub max_evidence_age_ms: u64,
    pub session_ttl_secs: u64,
    pub max_sessions: usize,
    pub max_blocks_per_worker: usize,
    pub cost_models: HashMap<String, CostModel>,
    pub restore_models: HashMap<String, RestoreCostModel>,
    pub lmcache: Option<LmCacheConfig>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            renderer_url: None,
            header_env: HashMap::new(),
            poll_interval_ms: 250,
            telemetry_timeout_ms: 2000,
            render_timeout_ms: 5000,
            max_evidence_age_ms: 3000,
            session_ttl_secs: 300,
            max_sessions: 10000,
            max_blocks_per_worker: 1_000_000,
            cost_models: HashMap::new(),
            restore_models: HashMap::new(),
            lmcache: None,
        }
    }
}

impl RoutingConfig {
    pub fn endpoint_identity(&self) -> bool {
        self.lmcache
            .as_ref()
            .is_some_and(|c| c.identity_mode == super::lmcache::IdentityMode::Endpoint)
    }

    pub fn identity_mode_name(&self) -> &'static str {
        self.lmcache
            .as_ref()
            .map_or("native", |c| c.identity_mode.name())
    }

    pub fn load(path: Option<&str>) -> Result<Self, String> {
        let config: Self = match path {
            Some(path) => serde_json::from_slice(
                &std::fs::read(path).map_err(|e| format!("routing config: {e}"))?,
            )
            .map_err(|e| format!("routing config: {e}"))?,
            None => Self::default(),
        };
        if config.poll_interval_ms == 0
            || config.telemetry_timeout_ms == 0
            || config.render_timeout_ms == 0
            || config.max_evidence_age_ms == 0
            || config.max_blocks_per_worker == 0
            || config.max_sessions == 0
        {
            return Err("routing config limits and intervals must be positive".into());
        }
        config.headers()?;
        if let Some(lmcache) = &config.lmcache {
            lmcache.validate()?;
            if config.renderer_url.is_some() {
                return Err(
                    "configure either the native renderer or lmcache.renderer_base_url".into(),
                );
            }
        }
        Ok(config)
    }

    pub fn headers(&self) -> Result<http::HeaderMap, String> {
        let mut headers = http::HeaderMap::new();
        for (header, variable) in &self.header_env {
            let value = std::env::var(variable).map_err(|_| {
                format!("missing routing credential environment variable: {variable}")
            })?;
            let name = http::HeaderName::from_bytes(header.as_bytes())
                .map_err(|_| "invalid routing header name")?;
            let mut value =
                http::HeaderValue::from_str(&value).map_err(|_| "invalid routing header value")?;
            value.set_sensitive(true);
            headers.insert(name, value);
        }
        Ok(headers)
    }
}
