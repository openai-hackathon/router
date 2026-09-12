use http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct RequestFeatures {
    pub request_id: String,
    pub session_id: Option<String>,
    pub tokens: Option<Vec<u32>>,
    pub fingerprint: Option<String>,
    pub model: Option<String>,
    pub output_limit: Option<usize>,
    pub num_choices: usize,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RenderedFeatures {
    pub schema_version: u32,
    pub fingerprint: String,
    pub token_ids: Vec<u32>,
}

impl RequestFeatures {
    pub fn unsupported(body: &Value, headers: Option<&HeaderMap>) -> Self {
        let session_id = headers
            .and_then(|h| h.get("x-session-id"))
            .and_then(|h| h.to_str().ok())
            .filter(|s| s.len() <= 256)
            .map(str::to_owned);
        Self {
            request_id: Uuid::new_v4().to_string(),
            session_id,
            tokens: None,
            fingerprint: None,
            model: body.get("model").and_then(Value::as_str).map(str::to_owned),
            output_limit: ["max_completion_tokens", "max_output_tokens", "max_tokens"]
                .iter()
                .find_map(|key| body.get(key).and_then(Value::as_u64))
                .and_then(|v| usize::try_from(v).ok()),
            num_choices: ["n", "best_of"]
                .iter()
                .try_fold(1usize, |choices, key| {
                    let count = match body.get(key).filter(|v| !v.is_null()) {
                        None => Some(1),
                        Some(value) => value.as_u64().and_then(|n| usize::try_from(n).ok()),
                    }?;
                    (count > 0).then_some(choices.max(count))
                })
                .unwrap_or(0),
            fallback_reason: Some("unsupported_request"),
        }
    }
}

/// Must receive the exact JSON that will be forwarded to inference. The helper
/// runs the pinned backend renderer, including tools and chat template kwargs.
pub async fn build(
    client: &reqwest::Client,
    renderer_url: Option<&str>,
    route: &str,
    body: &Value,
    headers: Option<&HeaderMap>,
) -> RequestFeatures {
    let mut features = RequestFeatures::unsupported(body, headers);
    if !matches!(route, "/v1/chat/completions" | "/v1/completions") {
        return features;
    }
    let Some(url) = renderer_url else {
        features.fallback_reason = Some("missing_renderer");
        return features;
    };
    features.fallback_reason = Some("render_failed");
    let result = client
        .post(url)
        .json(&serde_json::json!({"route": route, "body": body}))
        .send()
        .await;
    if let Ok(response) = result {
        if response.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
            features.fallback_reason = Some("unsupported_request");
        } else if response.status().is_success() {
            if let Ok(rendered) = response.json::<RenderedFeatures>().await {
                if rendered.schema_version == 1
                    && !rendered.fingerprint.is_empty()
                    && !rendered.token_ids.is_empty()
                {
                    features.tokens = Some(rendered.token_ids);
                    features.fingerprint = Some(rendered.fingerprint);
                    features.fallback_reason = None;
                }
            }
        }
    }
    features
}
