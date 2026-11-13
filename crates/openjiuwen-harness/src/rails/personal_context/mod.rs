//! Portable, offline subset of the Python `personal_context` package.
// The subtree is reached through `crate::rails`, whose module declaration is
// private until the rails module re-exports these items; the allows keep
// `-D warnings` green during that window.
#![allow(dead_code, unreachable_pub)]

mod atomic;
mod ingest;
mod link_book;
mod path;
mod provider;
mod rail;
mod search;
mod source;
mod status;
mod store;
