//! Compile-time contract for `#[openjiuwen_macros::tool]`.
//!
//! `tests/ui/pass/` covers every supported form; `tests/ui/fail/` pins the span
//! and message of every rejection. Regenerate the `.stderr` files with
//! `TRYBUILD=overwrite cargo test -p openjiuwen-macros --test compile`, inspect
//! them, then re-run without the variable.
#[test]
fn tool_attribute_contract() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/pass/*.rs");
    tests.compile_fail("tests/ui/fail/*.rs");
}
