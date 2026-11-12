//! The crate-private [`Teardown`]: an owned cleanup action that runs once.
//!
//! Python's runner relies on `finally` blocks and on the caller draining a
//! streaming generator; the Rust port cannot, because a consumer may drop a
//! stream at any point. A [`Teardown`] therefore owns its action and runs it
//! either on [`Teardown::finish`] (normal completion) or from [`Drop`] (early
//! drop), whichever comes first.
//!
//! The action is an opaque `async` block: agent runs post-run their session,
//! workflow runs close theirs, and team runs finish their session lease. Every
//! action logs its own failure, so a teardown problem never masks the run's
//! result.

use std::future::Future;

use futures::future::BoxFuture;

/// An owned teardown action that runs exactly once.
///
/// `Drop` spawns the action on the current Tokio runtime; when called outside a
/// runtime it logs a warning instead of panicking.
pub(crate) struct Teardown {
    /// The pending action, taken by whichever of `finish`/`drop` runs first.
    action: Option<BoxFuture<'static, ()>>,
}

impl Teardown {
    /// Wraps `action` as a teardown to run later.
    pub(crate) fn new<F>(action: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        Self {
            action: Some(Box::pin(action)),
        }
    }

    /// A teardown with nothing to do (a caller-owned resource).
    pub(crate) fn none() -> Self {
        Self { action: None }
    }

    /// Runs the action now, if it has not run yet.
    pub(crate) async fn finish(mut self) {
        if let Some(action) = self.action.take() {
            action.await;
        }
    }
}

impl Drop for Teardown {
    fn drop(&mut self) {
        let Some(action) = self.action.take() else {
            return;
        };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(action);
            }
            Err(_) => {
                tracing::warn!(
                    target: super::LOG_TARGET,
                    "no tokio runtime available; skipped run teardown"
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn counter() -> Arc<AtomicUsize> {
        Arc::new(AtomicUsize::new(0))
    }

    #[tokio::test]
    async fn finish_runs_the_action_exactly_once() {
        let counter = counter();
        let teardown = Teardown::new({
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        });
        teardown.finish().await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn finish_is_not_repeated_by_drop() {
        let counter = counter();
        let teardown = Teardown::new({
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        });
        teardown.finish().await;
        // Dropping the consumed teardown must not schedule the action again.
        tokio::task::yield_now().await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn drop_spawns_the_action() {
        let counter = counter();
        {
            let counter = Arc::clone(&counter);
            let _teardown = Teardown::new(async move {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }
        for _ in 0..100 {
            if counter.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_none_teardown_does_nothing() {
        Teardown::none().finish().await;
    }

    #[test]
    fn dropping_without_a_runtime_does_not_panic() {
        let counter = counter();
        {
            let counter = Arc::clone(&counter);
            let _teardown = Teardown::new(async move {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
