//! Drop-scoped filesystem and environment fixtures.
//!
//! These helpers are the Rust equivalents of the pytest fixtures the Python
//! test-suite relied on:
//!
//! - `tmp_path` becomes [`TempWorkspace`]: an owned temporary directory tree
//!   that is removed when the value is dropped.
//! - `monkeypatch.setenv("OPENJIUWEN_HOME", str(tmp_path / "home"))` becomes
//!   [`TempHomeGuard`]: a temporary home directory that is already pointed at by
//!   `OPENJIUWEN_HOME` for the guard's lifetime (and removed afterwards).
//! - `monkeypatch` environment cleanup becomes [`EnvGuard`]: every listed
//!   variable is cleared on construction and restored on drop, so a test never
//!   depends on — or leaks — the host environment.
//!
//! All three are scope guards: keep them alive in the test body, drop them (or
//! let them fall out of scope) at the end, and the process state is restored
//! even when an assertion panics.

mod env;
mod workspace;

pub use env::{EnvGuard, TempHomeGuard, ISOLATED_ENV_VARS};
pub use workspace::TempWorkspace;
