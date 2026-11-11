//! Description derived from the first doc-comment paragraph.
use openjiuwen_common::error::Result;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

/// Read the current value.
///
/// Args:
///     value: the value to read
#[tool]
pub async fn read_value(args: Args) -> Result<i64> {
    Ok(args.value)
}

fn main() {
    assert_eq!(ReadValueTool::DESCRIPTION, "Read the current value.");
    let _ = read_value_tool();
}
