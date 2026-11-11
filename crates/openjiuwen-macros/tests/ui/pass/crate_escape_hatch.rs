//! The `crate`/`common_crate` escape hatch: the generated paths follow the
//! aliases instead of the default crate roots.
use openjiuwen_foundation as my_foundation;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool(
    name = "escape",
    crate = "my_foundation",
    common_crate = "::openjiuwen_common"
)]
pub async fn escape(args: Args) -> openjiuwen_common::error::Result<i64> {
    Ok(args.value)
}

fn main() {
    let _ = escape_tool();
}
