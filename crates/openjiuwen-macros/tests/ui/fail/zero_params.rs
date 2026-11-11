//! A tool needs an argument parameter.
use openjiuwen_macros::tool;

#[tool]
async fn no_arguments() -> i64 {
    1
}

fn main() {}
