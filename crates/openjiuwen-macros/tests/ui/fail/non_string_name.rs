//! `name` must be a string literal.
use openjiuwen_macros::tool;

#[tool(name = 42)]
async fn non_string_name() -> i64 {
    1
}

fn main() {}
