//! At most two parameters are supported.
use openjiuwen_foundation::tool::ToolContext;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool]
async fn too_many(args: Args, ctx: &ToolContext, extra: i64) -> i64 {
    let _ = ctx;
    args.value + extra
}

fn main() {}
