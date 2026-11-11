//! Every attribute key may appear at most once.
use openjiuwen_macros::tool;

#[tool(name = "first", name = "second")]
async fn duplicate_key() -> i64 {
    1
}

fn main() {}
