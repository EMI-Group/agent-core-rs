//! Private function: the wrapper struct, constants, and constructor are private
//! and stay usable inside the defining module.
use openjiuwen_common::error::Result;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Args {
    value: i64,
}

#[tool]
async fn hidden(args: Args) -> Result<i64> {
    Ok(args.value)
}

fn main() {
    let _ = hidden_tool();
    assert_eq!(HiddenTool::NAME, "hidden");
}
