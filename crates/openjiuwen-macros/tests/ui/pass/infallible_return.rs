//! Infallible return type: the output is wrapped in `Ok` by the macro.
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

/// Triple the value.
#[tool]
pub async fn triple(args: Args) -> i64 {
    args.value * 3
}

fn main() {
    let _ = triple_tool();
    assert_eq!(TripleTool::DESCRIPTION, "Triple the value.");
}
