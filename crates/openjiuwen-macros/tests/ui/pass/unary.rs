//! Unary form: one argument struct, fallible return, model-facing name.
use openjiuwen_common::error::Result;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool(name = "double")]
pub async fn double(args: Args) -> Result<i64> {
    Ok(args.value * 2)
}

fn main() {
    let _ = double_tool();
    assert_eq!(DoubleTool::NAME, "double");
}
