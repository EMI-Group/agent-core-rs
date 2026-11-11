//! `ToolContext` cannot be the first (argument) parameter.
use openjiuwen_foundation::tool::ToolContext;
use openjiuwen_macros::tool;

#[tool]
async fn only_context(ctx: ToolContext) -> i64 {
    let _ = ctx;
    1
}

fn main() {}
