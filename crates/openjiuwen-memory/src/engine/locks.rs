//! Per-scope write serialization.
//!
//! Two concurrent writes to the same `(user_id, scope_id)` must not interleave their
//! dedupe scan and persistence, otherwise both could decide "no duplicate exists" and
//! store two identical records.
//! [`WriteLocks`] hands out one [`tokio::sync::Mutex`] per scope; the registry itself is
//! guarded by a short-lived `parking_lot` lock that is dropped before any `.await`.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::record::MemoryScope;

/// The mutex that serializes writers of one `(user_id, scope_id)`.
type ScopeLock = Arc<tokio::sync::Mutex<()>>;

/// The map held behind the short-lived registry lock.
type LockMap = HashMap<(String, String), ScopeLock>;

/// The registry of per-scope asynchronous write locks.
///
/// The lock map key is `(user_id, scope_id)`: `session_id`/`agent_id` do not create a
/// separate lock because records of the same user and scope share storage keys and the
/// dedupe scan covers the whole scope.
#[derive(Default)]
pub(crate) struct WriteLocks {
    /// Lock entries keyed by `(user_id, scope_id)`.
    locks: Mutex<LockMap>,
}

impl WriteLocks {
    /// Returns the lock of `scope`, creating it on first use.
    ///
    /// The returned `Arc` is cloned out of the registry so the caller can drop the
    /// `parking_lot` guard before awaiting the lock.
    pub(crate) fn lock_for(&self, scope: &MemoryScope) -> ScopeLock {
        let mut locks = self.locks.lock();
        Arc::clone(
            locks
                .entry((scope.user_id.clone(), scope.scope_id.clone()))
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_lock_per_user_and_scope() {
        let locks = WriteLocks::default();
        let alice = MemoryScope::new("alice", "ws");
        let first = locks.lock_for(&alice);
        let second = locks.lock_for(&MemoryScope::new("alice", "ws").session("s1"));
        assert!(
            Arc::ptr_eq(&first, &second),
            "session does not split the lock"
        );
        let other_scope = locks.lock_for(&MemoryScope::new("alice", "other"));
        assert!(!Arc::ptr_eq(&first, &other_scope));
        let other_user = locks.lock_for(&MemoryScope::new("bob", "ws"));
        assert!(!Arc::ptr_eq(&first, &other_user));
    }

    #[tokio::test]
    async fn a_held_scope_lock_serializes_writers() {
        let locks = Arc::new(WriteLocks::default());
        let scope = MemoryScope::new("alice", "ws");
        let lock = locks.lock_for(&scope);
        let guard = lock.lock().await;
        let contender = Arc::clone(&locks);
        let waiting = Arc::clone(&lock);
        let handle = tokio::spawn(async move {
            let _guard = waiting.lock().await;
            contender.lock_for(&scope);
        });
        // Yield so the spawned task reaches the pending lock acquisition.
        tokio::task::yield_now().await;
        assert!(!handle.is_finished());
        drop(guard);
        handle
            .await
            .expect("the spawned task completes once the guard is dropped");
    }
}
