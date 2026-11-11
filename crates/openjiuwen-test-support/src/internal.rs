//! Crate-private helpers shared by the mock server and the scripted model.
//!
//! Nothing in this module is public API: the items exist so the two fakes state
//! the Python-parity default answer and the `OpenAI` tool-call kind exactly once,
//! and so both share one poison-recovering mutex helper. They stay
//! `pub(crate)` and are deliberately not re-exported from the crate root.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Text served when a script is exhausted.
///
/// Python parity: `tests/unit_tests/fixtures/mock_llm.py` answers every
/// unmatched call with `"Default mock response"`, so both the mock server's
/// fallback and the scripted model's fallback turn use this exact string.
pub(crate) const DEFAULT_FALLBACK_TEXT: &str = "Default mock response";

/// The wire kind every built-in provider uses for tool calls.
pub(crate) const FUNCTION_KIND: &str = "function";

/// Locks `mutex`, recovering from a poisoned guard instead of panicking.
///
/// Test infrastructure must not turn one panicking test into a cascade of
/// failures for every later test in the same process, so a poisoned lock is
/// treated as "the data is still usable" and its guard is handed back.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_text_is_the_python_default() {
        assert_eq!(DEFAULT_FALLBACK_TEXT, "Default mock response");
        assert_eq!(FUNCTION_KIND, "function");
    }

    #[test]
    fn lock_recovers_a_poisoned_mutex() {
        let mutex = Mutex::new(1_u32);
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = mutex.lock().expect("first lock succeeds");
            *guard = 7;
            panic!("poison the lock on purpose");
        }));
        assert!(poisoned.is_err(), "the guarded section panicked");
        assert!(mutex.is_poisoned());

        let recovered = lock(&mutex);
        assert_eq!(*recovered, 7, "the data survives the poison");
        drop(recovered);
        assert_eq!(*lock(&mutex), 7, "later locks keep working");
    }
}
