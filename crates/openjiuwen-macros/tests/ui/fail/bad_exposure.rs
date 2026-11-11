//! `exposure` only accepts `"direct"` or `"deferred"`.
use openjiuwen_macros::tool;

#[tool(exposure = "always")]
async fn bad_exposure() -> i64 {
    1
}

fn main() {}
