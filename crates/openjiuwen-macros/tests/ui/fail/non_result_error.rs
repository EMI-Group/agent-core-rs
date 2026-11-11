//! A `Result` whose error type has no conversion into the shared error type
//! fails at the generated `?`, proving only `Result`-shaped returns are
//! propagated.
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug)]
struct CustomError;

impl std::fmt::Display for CustomError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("custom")
    }
}

impl std::error::Error for CustomError {}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Args {
    pub value: i64,
}

#[tool]
async fn confused(args: Args) -> Result<i64, CustomError> {
    if args.value < 0 {
        return Err(CustomError);
    }
    Ok(args.value)
}

fn main() {}
