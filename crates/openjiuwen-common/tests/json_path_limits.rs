//! Consumer-level checks for the bounded JSON path address space.
//!
//! `set_path` is reached from `openjiuwen-session` state keys and
//! `openjiuwen-graph` state updates, both of which can carry user- or
//! model-supplied paths, so an index that would pad a billion `Value`s must be a
//! typed error instead of an allocation.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::constants::MAX_COLLECTION_SIZE;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::utils::json::{get_path, set_path, MAX_JSON_INDEX};
use serde_json::json;

#[test]
fn the_documented_cap_reuses_the_collection_size_limit() {
    const { assert!(MAX_JSON_INDEX >= 65_536, "room for realistic state trees") };
    assert_eq!(MAX_JSON_INDEX, MAX_COLLECTION_SIZE);
}

#[test]
fn an_oversized_index_is_rejected_without_touching_the_tree() {
    let mut value = json!({"messages": [{"role": "user"}]});
    let before = value.clone();

    let error = set_path(&mut value, "messages[1000000000].role", json!("assistant")).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("1000000000"), "{message}");
    assert!(message.contains(&MAX_JSON_INDEX.to_string()), "{message}");
    assert_eq!(value, before, "a rejected path must not mutate the state");
    assert_eq!(get_path(&value, "messages[1000000000].role"), None);
}

#[test]
fn an_index_at_the_cap_is_still_addressable() {
    let mut value = json!({});
    set_path(
        &mut value,
        &format!("items[{MAX_JSON_INDEX}]"),
        json!("last"),
    )
    .unwrap();

    let items = value["items"].as_array().expect("padded array");
    assert_eq!(items.len(), MAX_JSON_INDEX + 1);
    assert_eq!(
        get_path(&value, &format!("items[{MAX_JSON_INDEX}]")),
        Some(&json!("last"))
    );

    // Beyond the cap the path is rejected rather than padded further.
    let error = set_path(
        &mut value,
        &format!("items[{}]", MAX_JSON_INDEX + 1),
        json!(1),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}
