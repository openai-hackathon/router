use serde::{Deserialize, Serialize};

/// Measured coefficients in milliseconds. There are deliberately no synthetic
/// defaults: an absent, invalid or out-of-domain calibration causes fallback.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CostModel {
    pub fingerprint: String,
    pub calibration_version: String,
    pub prompt_range: [usize; 2],
    pub output_range: [usize; 2],
    pub concurrency_range: [usize; 2],
    pub output_prior: usize,
    pub prefill: [f64; 3],
    /// Intercept, output-token term, prompt-length * output-token term.
    pub decode: [f64; 3],
    pub beta: f64,
    pub queue_ms: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CostEstimate {
    pub calibration_version: String,
    pub uncached_tokens: usize,
    pub estimated_output_tokens: usize,
    pub prefill_ms: f64,
    pub restore_ms: f64,
    pub restore_calibration_version: Option<String>,
    pub decode_ms: f64,
    pub load_multiplier: f64,
    pub queue_ms: f64,
    pub ect_ms: f64,
}

impl CostModel {
    pub fn predict(
        &self,
        fingerprint: &str,
        prompt: usize,
        reusable: usize,
        output_limit: Option<usize>,
        inflight: usize,
    ) -> Result<(f64, usize), &'static str> {
        self.estimate(fingerprint, prompt, reusable, output_limit, inflight)
            .map(|estimate| (estimate.ect_ms, estimate.estimated_output_tokens))
    }

    pub fn estimate(
        &self,
        fingerprint: &str,
        prompt: usize,
        reusable: usize,
        output_limit: Option<usize>,
        inflight: usize,
    ) -> Result<CostEstimate, &'static str> {
        if self.fingerprint != fingerprint || self.calibration_version.is_empty() {
            return Err("incompatible_cost_model");
        }
        let output = output_limit.map_or(self.output_prior, |limit| limit.min(self.output_prior));
        for (value, range) in [
            (prompt, self.prompt_range),
            (output, self.output_range),
            (inflight, self.concurrency_range),
        ] {
            if range[0] > range[1] || !(range[0]..=range[1]).contains(&value) {
                return Err("outside_calibration_range");
            }
        }
        if reusable > prompt
            || self.output_prior == 0
            || self
                .prefill
                .iter()
                .chain(self.decode.iter())
                .chain([&self.beta, &self.queue_ms])
                .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err("invalid_cost_prediction");
        }
        let l = prompt as f64;
        let h = reusable as f64;
        let o = output as f64;
        let prefill =
            self.prefill[0] + self.prefill[1] * (l - h) + self.prefill[2] * (l * l - h * h);
        let decode = self.decode[0] + self.decode[1] * o + self.decode[2] * l * o;
        let load_multiplier = 1.0 + self.beta * inflight as f64;
        let ect = (prefill + decode) * load_multiplier + self.queue_ms;
        if [prefill, decode, load_multiplier, ect]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("invalid_cost_prediction");
        }
        Ok(CostEstimate {
            calibration_version: self.calibration_version.clone(),
            uncached_tokens: prompt - reusable,
            estimated_output_tokens: output,
            prefill_ms: prefill,
            restore_ms: 0.0,
            restore_calibration_version: None,
            decode_ms: decode,
            load_multiplier,
            queue_ms: self.queue_ms,
            ect_ms: ect,
        })
    }
}

/// An independently calibrated transfer/restore term for one cache tier.
/// Included in service time before applying the jointly fitted load multiplier.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RestoreCostModel {
    pub fingerprint: String,
    pub calibration_version: String,
    pub location: String,
    pub token_range: [usize; 2],
    pub fixed_ms: f64,
    pub per_token_ms: f64,
}

impl RestoreCostModel {
    pub fn apply(
        &self,
        fingerprint: &str,
        location: &str,
        tokens: usize,
        mut cost: CostEstimate,
    ) -> Result<CostEstimate, &'static str> {
        if self.fingerprint != fingerprint
            || self.location != location
            || self.calibration_version.is_empty()
        {
            return Err("incompatible_restore_model");
        }
        if !(self.token_range[0]..=self.token_range[1]).contains(&tokens) {
            return Err("outside_restore_calibration_range");
        }
        let restore = self.fixed_ms + self.per_token_ms * tokens as f64;
        if [self.fixed_ms, self.per_token_ms, restore]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid_restore_prediction");
        }
        cost.ect_ms += restore * cost.load_multiplier;
        if !cost.ect_ms.is_finite() {
            return Err("invalid_restore_prediction");
        }
        cost.restore_ms = restore;
        cost.restore_calibration_version = Some(self.calibration_version.clone());
        Ok(cost)
    }
}
