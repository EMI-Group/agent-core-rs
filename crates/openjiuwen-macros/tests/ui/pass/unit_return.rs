//! Unit return type: the tool output serializes to `null`.
use openjiuwen_common::error::Result;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool(name = "notify")]
pub async fn notify(_args: Args) -> Result<()> {
    Ok(())
}

fn main() {
    let _ = notify_tool();
}
