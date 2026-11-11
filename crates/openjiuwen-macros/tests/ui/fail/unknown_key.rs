//! Unknown attribute keys are rejected.
use openjiuwen_macros::tool;

#[tool(nme = "typo")]
async fn unknown_key() -> i64 {
    1
}

fn main() {}
