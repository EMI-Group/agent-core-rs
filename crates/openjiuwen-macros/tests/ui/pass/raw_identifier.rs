//! Raw identifier function name: the generated names drop the `r#` marker.
use openjiuwen_common::error::Result;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool]
pub async fn r#type(args: Args) -> Result<i64> {
    Ok(args.value)
}

fn main() {
    let _ = type_tool();
    assert_eq!(TypeTool::NAME, "type");
    assert_eq!(TypeTool::DESCRIPTION, "type");
}
