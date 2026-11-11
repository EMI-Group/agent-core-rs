//! Second parameter borrowed as `&ToolContext`.
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::ToolContext;
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool(name = "read_session")]
pub async fn read_session(args: Args, ctx: &ToolContext) -> Result<Option<String>> {
    let _ = args.value;
    Ok(ctx.session_id.clone())
}

fn main() {
    let _ = read_session_tool();
}
