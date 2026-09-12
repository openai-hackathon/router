use serde::{Deserialize, Serialize};

/// Provenance is explicit: an estimated model is never presented as calibration.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompletionModelSource {
    Measured,
    Estimated,
    Synthetic,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionCoefficients {
    pub intercept_ms: f64,
    pub prompt_token_ms: f64,
    pub output_token_ms: f64,
    pub prompt_output_token_ms: f64,
    /// Net benefit of an observed CPU-cache token, including GPU overlap.
    pub cache_token_ms: f64,
    pub router_inflight_ms: f64,
    pub backend_running_ms: f64,
    pub backend_waiting_ms: f64,
    #[serde(default)]
    pub kv_usage_ms: f64,
}

/// Predicts dispatch-to-terminal time directly. Cache, queue and restoration
/// effects belong to the fitted response; no additional phase costs are added.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionModel {
    pub fingerprint: String,
    pub calibration_version: String,
    pub source: CompletionModelSource,
    pub prompt_range: [usize; 2],
    pub output_range: [usize; 2],
    pub concurrency_range: [usize; 2],
    pub cache_fraction_range: [f64; 2],
    pub backend_running_range: [usize; 2],
    pub backend_waiting_range: [usize; 2],
    #[serde(default = "default_kv_usage_range")]
    pub kv_usage_range: [f64; 2],
    pub output_prior: usize,
    pub coefficients: CompletionCoefficients,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CompletionEstimate {
    pub calibration_version: String,
    pub source: CompletionModelSource,
    pub estimated_output_tokens: usize,
    pub cached_tokens: usize,
    pub router_inflight: usize,
    pub backend_running: usize,
    pub backend_waiting: usize,
    pub kv_usage_fraction: Option<f64>,
    pub cache_credit_ms: f64,
    pub backend_pressure_ms: f64,
    pub ect_ms: f64,
}

fn default_kv_usage_range() -> [f64; 2] {
    [0.0, 1.0]
}

impl CompletionModel {
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &self,
        fingerprint: &str,
        prompt: usize,
        raw_cached: usize,
        output_limit: Option<usize>,
        inflight: usize,
        backend_running: usize,
        backend_waiting: usize,
        kv_usage_fraction: Option<f64>,
    ) -> Result<CompletionEstimate, &'static str> {
        if self.fingerprint != fingerprint || self.calibration_version.trim().is_empty() {
            return Err("incompatible_completion_model");
        }
        let output = output_limit.map_or(self.output_prior, |limit| limit.min(self.output_prior));
        if prompt == 0 || raw_cached > prompt || self.output_prior == 0 || output == 0 {
            return Err("invalid_completion_prediction");
        }
        for (value, range) in [
            (prompt, self.prompt_range),
            (output, self.output_range),
            (inflight, self.concurrency_range),
            (backend_running, self.backend_running_range),
            (backend_waiting, self.backend_waiting_range),
        ] {
            if range[0] > range[1] || !(range[0]..=range[1]).contains(&value) {
                return Err("outside_completion_calibration_range");
            }
        }
        let fraction = raw_cached as f64 / prompt as f64;
        let [minimum, maximum] = self.cache_fraction_range;
        if !minimum.is_finite()
            || !maximum.is_finite()
            || minimum < 0.0
            || maximum > 1.0
            || minimum > maximum
        {
            return Err("invalid_completion_prediction");
        }
        if !(minimum..=maximum).contains(&fraction) {
            return Err("outside_completion_calibration_range");
        }
        let c = &self.coefficients;
        if [
            c.intercept_ms,
            c.prompt_token_ms,
            c.output_token_ms,
            c.prompt_output_token_ms,
            c.cache_token_ms,
            c.router_inflight_ms,
            c.backend_running_ms,
            c.backend_waiting_ms,
            c.kv_usage_ms,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("invalid_completion_prediction");
        }
        let [kv_min, kv_max] = self.kv_usage_range;
        if !kv_min.is_finite()
            || !kv_max.is_finite()
            || kv_min < 0.0
            || kv_max > 1.0
            || kv_min > kv_max
        {
            return Err("invalid_completion_prediction");
        }
        let kv_pressure_ms = match kv_usage_fraction {
            Some(usage) if !usage.is_finite() || !(0.0..=1.0).contains(&usage) => {
                return Err("invalid_backend_kv_usage");
            }
            Some(usage) if !(kv_min..=kv_max).contains(&usage) => {
                return Err("outside_completion_calibration_range");
            }
            Some(usage) => usage * c.kv_usage_ms,
            None if c.kv_usage_ms > 0.0 => return Err("missing_backend_kv_usage"),
            None => 0.0,
        };
        let cache_credit_ms = raw_cached as f64 * c.cache_token_ms;
        let backend_pressure_ms = backend_running as f64 * c.backend_running_ms
            + backend_waiting as f64 * c.backend_waiting_ms
            + kv_pressure_ms;
        let ect_ms = c.intercept_ms
            + prompt as f64 * c.prompt_token_ms
            + output as f64 * c.output_token_ms
            + prompt as f64 * output as f64 * c.prompt_output_token_ms
            - cache_credit_ms
            + inflight as f64 * c.router_inflight_ms
            + backend_pressure_ms;
        // A bad cache credit must not make a worker appear free or disappear.
        // The shared selector handles this error with its all-candidate fallback.
        if !ect_ms.is_finite() || ect_ms <= 0.0 {
            return Err("invalid_completion_prediction");
        }
        Ok(CompletionEstimate {
            calibration_version: self.calibration_version.clone(),
            source: self.source,
            estimated_output_tokens: output,
            cached_tokens: raw_cached,
            router_inflight: inflight,
            backend_running,
            backend_waiting,
            kv_usage_fraction,
            cache_credit_ms,
            backend_pressure_ms,
            ect_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CompletionModel {
        CompletionModel {
            fingerprint: "model".into(),
            calibration_version: "measured:v1".into(),
            source: CompletionModelSource::Measured,
            prompt_range: [100, 2000],
            output_range: [1, 128],
            concurrency_range: [0, 8],
            cache_fraction_range: [0.0, 1.0],
            backend_running_range: [0, 16],
            backend_waiting_range: [0, 8],
            kv_usage_range: [0.0, 1.0],
            output_prior: 64,
            coefficients: CompletionCoefficients {
                intercept_ms: 100.0,
                prompt_token_ms: 1.0,
                output_token_ms: 10.0,
                prompt_output_token_ms: 0.01,
                cache_token_ms: 0.8,
                router_inflight_ms: 20.0,
                backend_running_ms: 30.0,
                backend_waiting_ms: 50.0,
                kv_usage_ms: 0.0,
            },
        }
    }

    #[test]
    fn uses_raw_cpu_prefix_and_separate_background_load_with_capped_output() {
        let estimate = model()
            .estimate("model", 1000, 513, Some(32), 1, 3, 2, None)
            .unwrap();
        assert_eq!(estimate.estimated_output_tokens, 32);
        assert_eq!(estimate.cached_tokens, 513);
        assert_eq!(estimate.backend_running, 3);
        assert_eq!(estimate.source, CompletionModelSource::Measured);
        assert!((estimate.ect_ms - 1539.6).abs() < 1e-9);
        let busy = model()
            .estimate("model", 1000, 513, Some(32), 1, 4, 2, None)
            .unwrap();
        assert!((busy.ect_ms - estimate.ect_ms - 30.0).abs() < 1e-9);
        let cold = model()
            .estimate("model", 1000, 0, Some(32), 1, 3, 2, None)
            .unwrap();
        assert!((cold.ect_ms - estimate.ect_ms - 410.4).abs() < 1e-9);
    }

    #[test]
    fn rejects_mismatches_invalid_predictions_and_all_feature_domain_exits() {
        let valid = model();
        assert_eq!(
            valid.estimate("other", 1000, 0, None, 0, 0, 0, None),
            Err("incompatible_completion_model")
        );
        for (prompt, output, router, running, waiting) in [
            (99, 64, 0, 0, 0),
            (1000, 0, 0, 0, 0),
            (1000, 64, 9, 0, 0),
            (1000, 64, 0, 17, 0),
            (1000, 64, 0, 0, 9),
        ] {
            assert!(valid
                .estimate(
                    "model",
                    prompt,
                    0,
                    Some(output),
                    router,
                    running,
                    waiting,
                    None
                )
                .is_err());
        }
        assert!(valid
            .estimate("model", 1000, 1001, None, 0, 0, 0, None)
            .is_err());
        for credit in [-1.0, f64::NAN, f64::INFINITY, 10_000.0] {
            let mut invalid = model();
            invalid.coefficients.cache_token_ms = credit;
            assert_eq!(
                invalid.estimate("model", 1000, 1000, None, 0, 0, 0, None),
                Err("invalid_completion_prediction")
            );
        }
        let mut limited = model();
        limited.cache_fraction_range = [0.0, 0.5];
        assert_eq!(
            limited.estimate("model", 1000, 501, None, 0, 0, 0, None),
            Err("outside_completion_calibration_range")
        );
        limited.cache_fraction_range = [f64::NAN, 1.0];
        assert!(limited
            .estimate("model", 1000, 0, None, 0, 0, 0, None)
            .is_err());
    }
    #[test]
    fn optional_kv_pressure_never_treats_unknown_as_observed_zero() {
        let mut calibrated = model();
        let unknown = calibrated
            .estimate("model", 1000, 0, None, 0, 0, 0, None)
            .unwrap();
        assert_eq!(unknown.kv_usage_fraction, None);
        calibrated.coefficients.kv_usage_ms = 200.0;
        assert_eq!(
            calibrated.estimate("model", 1000, 0, None, 0, 0, 0, None),
            Err("missing_backend_kv_usage")
        );
        let busy = calibrated
            .estimate("model", 1000, 0, None, 0, 0, 0, Some(0.75))
            .unwrap();
        assert_eq!(busy.backend_pressure_ms, 150.0);
        assert_eq!(busy.ect_ms - unknown.ect_ms, 150.0);
        assert!(calibrated
            .estimate("model", 1000, 0, None, 0, 0, 0, Some(f64::NAN))
            .is_err());
    }
}
