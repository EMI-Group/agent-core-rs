//! A method inside an `impl` block is not a free function.
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

pub struct Holder;

impl Holder {
    #[tool]
    async fn method(&self, args: Args) -> i64 {
        args.value
    }
}

fn main() {}
