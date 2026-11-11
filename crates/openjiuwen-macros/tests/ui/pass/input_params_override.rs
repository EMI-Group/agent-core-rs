//! `input_params` supplies the schema, so the argument type needs no
//! `JsonSchema` derive.
use openjiuwen_common::error::Result;
use openjiuwen_macros::tool;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RawArgs {
    pub x: String,
}

#[tool(
    name = "raw",
    input_params = serde_json::json!({
        "type": "object",
        "properties": { "x": { "type": "string" } },
        "required": ["x"]
    })
)]
pub async fn raw(args: RawArgs) -> Result<String> {
    Ok(args.x)
}

fn main() {
    let _ = raw_tool();
}
