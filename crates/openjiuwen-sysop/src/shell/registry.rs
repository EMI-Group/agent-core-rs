//! Session-scoped registry of live shell cancellations.
//!
//! Python parity: `shell_process_registry.py` (`SHELL_PROCESS_REGISTRY`,
//! `register_shell_process`, `unregister_shell_process`, `resolve_shell_session_id`). The
//! Python module maps a session id to the live `asyncio` subprocesses so a session can be
//! torn down; in Rust the backend registers a [`CancellationToken`] instead and selects on
//! it while the child runs.
//!
//! Cancellation is **cooperative**: cancelling a token only signals the backend's run
//! loop, which then kills the child process and reports `cancelled = true` on the result.
//! Cancelling a session is therefore not a synchronous process kill.
//!
//! ```
//! use openjiuwen_sysop::shell::registry::ShellProcessRegistry;
//! use tokio_util::sync::CancellationToken;
//!
//! let registry = ShellProcessRegistry::new();
//! let token = CancellationToken::new();
//! let guard = registry.register("session-1", token.clone());
//! assert_eq!(registry.active_sessions(), vec!["session-1".to_string()]);
//! assert_eq!(registry.cancel_session("session-1"), 1);
//! assert!(token.is_cancelled());
//! drop(guard);
//! ```

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use tokio_util::sync::CancellationToken;

/// Registrations grouped by session id.
type SessionMap = BTreeMap<String, BTreeMap<u64, CancellationToken>>;

/// Session-scoped registry of live shell cancellation tokens.
///
/// A local instance is useful in tests; [`ShellProcessRegistry::global`] is the
/// process-wide registry the backend registers with (Python's `SHELL_PROCESS_REGISTRY`).
#[derive(Debug, Clone, Default)]
pub struct ShellProcessRegistry {
    inner: Arc<Inner>,
}

impl ShellProcessRegistry {
    /// Builds an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `token` under `session_id` and returns the guard that deregisters it.
    ///
    /// A blank session id registers nothing and returns an inert guard, so dropping it is
    /// a no-op (Python's `resolve_shell_session_id` yields no session outside a bound
    /// context).
    #[must_use]
    pub fn register(&self, session_id: &str, token: CancellationToken) -> Registration {
        if session_id.trim().is_empty() {
            return Registration {
                inner: None,
                session_id: String::new(),
                id: 0,
            };
        }
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        self.sessions()
            .entry(session_id.to_string())
            .or_default()
            .insert(id, token);
        Registration {
            inner: Some(Arc::clone(&self.inner)),
            session_id: session_id.to_string(),
            id,
        }
    }

    /// Cancels every token registered for `session_id`, removing them.
    ///
    /// Returns the number of tokens cancelled; an unknown session returns `0`.
    #[must_use]
    pub fn cancel_session(&self, session_id: &str) -> usize {
        let tokens: Vec<CancellationToken> = {
            let mut sessions = self.sessions();
            sessions
                .remove(session_id)
                .map(|session| session.into_values().collect())
                .unwrap_or_default()
        };
        cancel_all_tokens(tokens)
    }

    /// Cancels every registered token, removing all registrations.
    ///
    /// Returns the number of tokens cancelled.
    #[must_use]
    pub fn cancel_all(&self) -> usize {
        let tokens: Vec<CancellationToken> = {
            let mut sessions = self.sessions();
            std::mem::take(&mut *sessions)
                .into_values()
                .flat_map(BTreeMap::into_values)
                .collect()
        };
        cancel_all_tokens(tokens)
    }

    /// Session ids with at least one live registration, sorted.
    #[must_use]
    pub fn active_sessions(&self) -> Vec<String> {
        self.sessions().keys().cloned().collect()
    }

    /// Returns `true` when `session_id` currently has at least one registration.
    #[must_use]
    pub fn is_registered(&self, session_id: &str) -> bool {
        self.sessions()
            .get(session_id)
            .is_some_and(|session| !session.is_empty())
    }

    /// The process-global registry (the documented singleton).
    ///
    /// This is one of the few pieces of global mutable state in the crate; it exists so
    /// that a caller can cancel a session's shell processes without threading a registry
    /// through the backend. Use [`ShellProcessRegistry::new`] for isolated instances.
    #[must_use]
    pub fn global() -> &'static ShellProcessRegistry {
        static REGISTRY: OnceLock<ShellProcessRegistry> = OnceLock::new();
        REGISTRY.get_or_init(Self::new)
    }

    /// Locks the session map, recovering from a poisoned lock.
    ///
    /// The registry holds no invariants that a panic could break, so poisoning is ignored
    /// rather than propagated (the methods are infallible by design).
    fn sessions(&self) -> MutexGuard<'_, SessionMap> {
        self.inner
            .sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Registrations shared by the registry and its guards.
#[derive(Debug, Default)]
struct Inner {
    sessions: Mutex<SessionMap>,
    next_id: AtomicU64,
}

/// RAII registration guard: dropping it deregisters the token.
#[derive(Debug)]
pub struct Registration {
    /// `None` for an inert guard (blank session id).
    inner: Option<Arc<Inner>>,
    session_id: String,
    id: u64,
}

impl Registration {
    /// The registry-assigned registration id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The session the token was registered under (empty for an inert guard).
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut sessions = inner
            .sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(session) = sessions.get_mut(&self.session_id) {
            session.remove(&self.id);
            if session.is_empty() {
                sessions.remove(&self.session_id);
            }
        }
    }
}

/// Cancels `tokens` outside any lock and returns how many were cancelled.
fn cancel_all_tokens(tokens: Vec<CancellationToken>) -> usize {
    let count = tokens.len();
    for token in tokens {
        token.cancel();
    }
    count
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn active_sessions_are_sorted_and_deduplicated() {
        let registry = ShellProcessRegistry::new();
        let first = CancellationToken::new();
        let second = CancellationToken::new();
        let other = CancellationToken::new();
        let _guard_a = registry.register("session-b", first);
        let _guard_b = registry.register("session-b", second);
        let _guard_c = registry.register("session-a", other);

        assert_eq!(
            registry.active_sessions(),
            vec!["session-a".to_string(), "session-b".to_string()]
        );
        assert!(registry.is_registered("session-a"));
        assert!(registry.is_registered("session-b"));
        assert!(!registry.is_registered("session-c"));
    }

    #[test]
    fn cancel_session_cancels_only_that_session() {
        let registry = ShellProcessRegistry::new();
        let first = CancellationToken::new();
        let second = CancellationToken::new();
        let other = CancellationToken::new();
        let _guard_a = registry.register("session-b", first.clone());
        let _guard_b = registry.register("session-b", second.clone());
        let _guard_c = registry.register("session-a", other.clone());

        assert_eq!(registry.cancel_session("session-b"), 2);
        assert!(first.is_cancelled());
        assert!(second.is_cancelled());
        assert!(!other.is_cancelled());
        assert!(!registry.is_registered("session-b"));
        assert!(registry.is_registered("session-a"));
    }

    #[test]
    fn dropping_the_guard_deregisters() {
        let registry = ShellProcessRegistry::new();
        let token = CancellationToken::new();
        let guard = registry.register("session", token.clone());
        assert!(registry.is_registered("session"));
        drop(guard);
        assert!(!registry.is_registered("session"));
        assert_eq!(registry.active_sessions(), Vec::<String>::new());
        assert!(!token.is_cancelled());

        let first = CancellationToken::new();
        let second = CancellationToken::new();
        let guard_a = registry.register("session", first);
        let _guard_b = registry.register("session", second);
        drop(guard_a);
        assert_eq!(registry.cancel_session("session"), 1);
    }

    #[test]
    fn cancel_all_clears_every_session() {
        let registry = ShellProcessRegistry::new();
        let first = CancellationToken::new();
        let second = CancellationToken::new();
        let _guard_a = registry.register("one", first.clone());
        let _guard_b = registry.register("two", second.clone());

        assert_eq!(registry.cancel_all(), 2);
        assert!(first.is_cancelled());
        assert!(second.is_cancelled());
        assert_eq!(registry.active_sessions(), Vec::<String>::new());
        assert_eq!(registry.cancel_all(), 0);
    }

    #[test]
    fn unknown_session_cancels_nothing() {
        let registry = ShellProcessRegistry::new();
        assert_eq!(registry.cancel_session("missing"), 0);
        assert!(!registry.is_registered("missing"));
    }

    #[test]
    fn blank_session_registers_nothing() {
        let registry = ShellProcessRegistry::new();
        let token = CancellationToken::new();
        let guard = registry.register("   ", token.clone());
        assert_eq!(registry.active_sessions(), Vec::<String>::new());
        assert_eq!(guard.session_id(), "");
        assert_eq!(registry.cancel_session(""), 0);
        drop(guard);
        assert_eq!(registry.active_sessions(), Vec::<String>::new());
        assert!(!token.is_cancelled());
    }

    #[test]
    fn registration_ids_are_unique_and_exposed() {
        let registry = ShellProcessRegistry::new();
        let _first = registry.register("s", CancellationToken::new());
        let second = registry.register("s", CancellationToken::new());
        assert_eq!(second.session_id(), "s");
        assert_eq!(second.id(), 1);
    }

    #[test]
    fn global_is_pointer_stable() {
        let first = ShellProcessRegistry::global();
        let second = ShellProcessRegistry::global();
        assert!(std::ptr::eq(first, second));
    }
}
