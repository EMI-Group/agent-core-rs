//! `pub(crate)` function: every generated item inherits that visibility.
use openjiuwen_common::error::Result;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct Args {
    pub(crate) value: i64,
}

#[tool(name = "scoped")]
pub(crate) async fn scoped(args: Args) -> Result<i64> {
    Ok(args.value)
}

fn main() {
    let _ = scoped_tool();
    assert_eq!(ScopedTool::NAME, "scoped");
}
