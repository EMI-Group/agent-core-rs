//! Every policy key: name, description, and all five card policy attributes.
use openjiuwen_common::error::Result;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool(
    name = "policy",
    description = "A tool with every policy key set",
    stateless = true,
    parallel_safe = false,
    idempotent = true,
    exposure = "deferred",
    execution = "client"
)]
pub async fn policy(args: Args) -> Result<i64> {
    Ok(args.value)
}

fn main() {
    let _ = policy_tool();
}
