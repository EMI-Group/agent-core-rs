//! The argument struct must be taken by value.
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool]
async fn borrowed(args: &Args) -> i64 {
    args.value
}

fn main() {}
