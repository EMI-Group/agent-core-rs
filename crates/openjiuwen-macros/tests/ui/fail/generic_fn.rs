//! Generic functions are rejected.
use openjiuwen_macros::tool;

#[tool]
async fn generic<T: Default>(_args: T) -> i64 {
    1
}

fn main() {}
