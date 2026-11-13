//! Row-stamp and key-validation helpers shared by the store backends.
//!
//! Every backend stamps a caller-supplied `0` with the millisecond clock and keeps two writes inside
//! the same millisecond ordered, so `InMemoryTeamStore` and `SqliteTeamStore` (feature `sqlite`)
//! report the same timestamps for the same call sequence.
//!
//! The helpers are `pub(crate)`: they are backend internals, not part of the
//! [`TeamStore`](crate::storage::TeamStore) contract.

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::unix_millis;

/// One monotonically increasing write stamp.
///
/// A real backend stamps `now()`; the store must also keep ordering stable when two writes land
/// inside the same millisecond, so it returns `max(now, previous + 1)`.
#[must_use]
pub(crate) fn refresh(previous: u64) -> u64 {
    let now = u64::try_from(unix_millis()).unwrap_or(0);
    now.max(previous.saturating_add(1))
}

/// Returns the caller-supplied stamp when it is set, else a fresh write stamp after `previous`.
#[must_use]
pub(crate) fn stamp(supplied: u64, previous: u64) -> u64 {
    if supplied == 0 {
        refresh(previous)
    } else {
        supplied
    }
}

/// Rejects an empty row key.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) whose message
/// is `"{label} must not be empty"`.
pub(crate) fn require_key(value: &str, label: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::validation(format!("{label} must not be empty")));
    }
    Ok(())
}
