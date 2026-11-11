//! Second parameter owned as `ToolContext`.
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::ToolContext;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool(name = "invocation")]
pub async fn invocation(args: Args, ctx: ToolContext) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "value": args.value,
        "invocation_id": ctx.invocation_id,
    }))
}

fn main() {
    let _ = invocation_tool();
}
