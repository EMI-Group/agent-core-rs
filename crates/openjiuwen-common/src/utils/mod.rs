//! General-purpose utilities: JSON helpers, ids and timestamps, string
//! sanitization/formatting, and hashing.
//!
//! These modules replace the Python `openjiuwen.core.common.utils` package
//! (`dict_utils`, `id_utils`, `string_utils`, `hash_util`) with idiomatic Rust
//! free functions. They are pure and side-effect free apart from reading the
//! system clock and the process id.

pub mod hash;
pub mod id;
pub mod json;
pub mod string;
