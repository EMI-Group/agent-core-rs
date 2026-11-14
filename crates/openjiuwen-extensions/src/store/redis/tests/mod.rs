//! Unit tests for the live `Redis` store backends.
//!
//! Python counterpart: the reference tree has no dedicated `RedisStore` unit
//! test (`tests/unit_tests/extensions/store/` covers only the Elasticsearch
//! vector store and `GaussDB`) and ships no `Redis` document or vector backend at
//! all, so the key-value suites stand in for
//! `tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py` and the
//! document/vector suites verify the Rust-side extension directly.
//!
//! The child modules reach the backend's items through the re-exports below, and
//! a fake `Redis` server (`crate::redis::testsupport`) supplies every reply, so
//! the suite is local, offline, and deterministic.
pub(crate) use super::kv::{
    decode_value, escape_glob, expiry_millis, is_unknown_command, BYTES_TAG, TEXT_TAG,
};
pub(crate) use super::*;
pub(crate) use openjiuwen_core::foundation::store::{KvStore, KvValue};
pub(crate) use std::collections::BTreeMap;
pub(crate) use std::time::Duration;

#[allow(clippy::unwrap_used, clippy::expect_used)]
mod document;
mod kv;
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod vector;
