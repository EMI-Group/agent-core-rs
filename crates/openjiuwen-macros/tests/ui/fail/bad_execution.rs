//! `execution` only accepts `"server"` or `"client"`.
use openjiuwen_macros::tool;

#[tool(execution = "local")]
async fn bad_execution() -> i64 {
    1
}

fn main() {}
