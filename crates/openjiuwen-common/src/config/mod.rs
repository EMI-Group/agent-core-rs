//! Configuration and environment resolution.
//!
//! The Rust SDK keeps the Python SDK's "no central config file" model
//! (`docs/ARCHITECTURE.md` §4): typed configuration structs live at their call
//! sites, and every runtime knob is read from an environment variable through
//! [`env`](mod@env). Filesystem locations for runtime state live in [`home`].
//!
//! [`load_dotenv`] is the only place a `.env` file is read. It exists for
//! binaries and examples; library constructors never call it, so embedding an
//! `openJiuwen` library in a larger application never mutates the process
//! environment behind the host application's back.

pub mod env;
pub mod home;

pub use home::OpenJiuwenHome;

use std::path::PathBuf;

use crate::error::{Error, Result};

/// Loads a `.env` file into the process environment.
///
/// This is the only `.env` entry point in the SDK and is meant for **binaries
/// and examples only**: library constructors never call it, because loading a
/// `.env` file mutates process-global state that the embedding application owns.
///
/// The lookup itself is delegated to `dotenvy::dotenv()`, which searches the
/// current directory and then its ancestors. Existing environment variables are
/// never overwritten by the file (`dotenvy` parity).
///
/// Returns `Ok(Some(path))` when a `.env` file was found and loaded, and
/// `Ok(None)` when no `.env` file exists anywhere in the search path — a missing
/// file is not an error, since production deployments configure the process
/// environment directly.
///
/// # Errors
///
/// Returns [`Error::Config`] when a `.env` file exists but cannot be read
/// (permissions, malformed contents, invalid encoding).
pub fn load_dotenv() -> Result<Option<PathBuf>> {
    match dotenvy::dotenv() {
        Ok(path) => Ok(Some(path)),
        Err(err) if err.not_found() => Ok(None),
        Err(err) => Err(Error::config(format!("failed to load .env file: {err}"))),
    }
}

/// Helpers shared by the `config` test modules.
///
/// Environment variables are process-global, so every test that reads or mutates
/// them takes the same [`test_support::ENV_LOCK`]; a single lock shared across
/// `env` and `home` keeps those tests from racing with each other.
#[cfg(test)]
pub(crate) mod test_support {
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard, PoisonError};

    /// Serializes every test that reads or mutates process environment variables.
    pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// An RAII guard that owns [`ENV_LOCK`], clears the managed variables, and
    /// restores their previous values on drop.
    ///
    /// Clearing on construction means assertions never depend on the host
    /// environment, and restoring on drop keeps the test process clean even when
    /// an assertion panics.
    pub(crate) struct EnvGuard {
        _lock: MutexGuard<'static, ()>,
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvGuard {
        /// Locks the environment and clears every name in `names`.
        pub(crate) fn new(names: &[&'static str]) -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
            let saved = names
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect();
            for name in names {
                std::env::remove_var(name);
            }
            Self { _lock: lock, saved }
        }

        /// Sets `name` to `value` for the lifetime of the guard.
        ///
        /// The receiver is not read; `&self` is what ties the mutation to the
        /// guard (and therefore to the lock) so the call cannot outlive it.
        #[allow(clippy::unused_self)]
        pub(crate) fn set(&self, name: &str, value: &str) {
            std::env::set_var(name, value);
        }

        /// Removes `name` for the lifetime of the guard.
        #[allow(clippy::unused_self)]
        pub(crate) fn remove(&self, name: &str) {
            std::env::remove_var(name);
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.saved.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}
