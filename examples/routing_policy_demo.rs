//! Offline policy demonstration. All cache observations and costs are synthetic.
use serde_json::json;
use vllm_router_rs::routing_state::{
    config::RoutingConfig, features::RequestFeatures, Ranking, SelectionSnapshot,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../tests/fixtures/routing_policy_scenario.json"
    ))?;
    let snapshot: SelectionSnapshot = serde_json::from_value(fixture["snapshot"].clone())?;
    let config: RoutingConfig = serde_json::from_value(fixture["routing_config"].clone())?;
    let request = &fixture["request"];
    let mut features = RequestFeatures::unsupported(request, None);
    // These placeholders express the fixture's length. They are not tokens
    // produced by a tokenizer and must never be sent to inference or lookup.
    features.tokens = Some(vec![0; request["prompt_tokens"].as_u64().unwrap() as usize]);
    features.fingerprint = request["fingerprint"].as_str().map(str::to_owned);
    features.output_limit = request["output_limit"].as_u64().map(|v| v as usize);
    features.fallback_reason = None;
    let mut decisions = serde_json::Map::new();
    for ranking in [
        Ranking::PrefixMax,
        Ranking::LeastLoadKv,
        Ranking::KvBatchEct,
    ] {
        let decision = snapshot
            .decide(ranking, &features, &config)
            .ok_or("no candidate")?;
        decisions.insert(ranking.name().into(), serde_json::to_value(decision)?);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "synthetic": true,
            "description": fixture["description"],
            "request": request,
            "decisions": decisions,
        }))?
    );
    Ok(())
}
