//! The caller-driven sync surface of the local store: `sync_now`, the
//! [`SyncTrigger`] points, and the opt-in background interval task.
//!
//! Ports the sync triggers of [`crate::lite::sync`] onto one store handle. Nothing here
//! runs on its own: a trigger is a point the caller reaches, and the only background task
//! is the one [`LocalCodingMemory::start_interval_sync`] explicitly starts. That task lives
//! on the caller's tokio runtime, wakes no more often than the configured interval, holds a
//! [`CancellationToken`], and is owned by the handle that started it — there is no global or
//! static state, so a second handle over the same root reports no task of its own.
//!
//! # What a sync does
//!
//! [`LocalCodingMemory::sync_now`] reconciles every derived representation with the sources
//! of truth:
//!
//! 1. the document is refreshed from disk — in the per-fact file layout this also rewrites
//!    the projection (see `crate::lite::local::facts_mode`);
//! 2. the derived full-text index is rebuilt from the cached document;
//! 3. the chunk index is refreshed when an embedding model is configured, and the pass
//!    reports the chunks it embedded (a second sync over unchanged text embeds nothing,
//!    because chunk vectors are read from the content-hash cache);
//! 4. the session logs under `sessions/` are counted.
//!
//! # Triggers
//!
//! - [`SyncTrigger::OnWrite`] reconciles the cached document only, and reports whether a
//!   reload actually happened.
//! - [`SyncTrigger::OnSessionStart`] always runs a full sync.
//! - [`SyncTrigger::OnInterval`] runs a full sync only when the deterministic
//!   [`IntervalScheduler`] of this handle says a run is due, so two calls in a row with no
//!   elapsed interval never run twice. The scheduler is created from the first interval the
//!   handle sees and re-created when a caller passes a different one; a zero interval is
//!   [`Error::Validation`](openjiuwen_common::error::Error::Validation).
//!
//! Lock order is the cross-process [`FileLock`] and then the state guard, everywhere; no
//! state guard is held across the embedding model call, which is why the embedding refresh
//! runs after the guards are released.
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::Result;

use crate::lite::lock::FileLock;
use crate::lite::session::SessionStore;
use crate::lite::sync::{IntervalScheduler, SyncReport, SyncTrigger};

use super::state::LocalCodingMemory;
use super::TARGET;

/// The trigger label a full [`LocalCodingMemory::sync_now`] run reports.
///
/// It is intentionally not one of [`SyncTrigger::ALL_KINDS`]: a direct call is not a
/// trigger point, and [`LocalCodingMemory::on_trigger`] re-labels its report with the
/// trigger that asked for it.
pub(super) const SYNC_NOW_LABEL: &str = "sync_now";

/// The opt-in background interval task of one store handle.
pub(super) struct IntervalTask {
    /// The token the loop watches for cancellation.
    pub(super) token: CancellationToken,
    /// The spawned loop.
    pub(super) task: JoinHandle<()>,
}

impl LocalCodingMemory {
    /// Reconciles the document, the derived indices, and the session count.
    ///
    /// See the module documentation for the steps. The call is idempotent: a second sync
    /// over unchanged inputs reports the same facts, chunks, and sessions, embeds nothing,
    /// and leaves every file byte-identical.
    ///
    /// # Errors
    ///
    /// Returns the lock, filesystem, parse, index, or embedding error of the failing step;
    /// nothing is swallowed.
    pub async fn sync_now(&self) -> Result<SyncReport> {
        let facts = self.reconcile_document().await?;
        let (chunks, embedded) = if self.has_embedding() {
            let stats = self.refresh_embeddings().await?;
            (stats.chunks, stats.embedded)
        } else {
            (0, 0)
        };
        let sessions = SessionStore::new(&self.inner.root).chunks().await?.len();
        Ok(SyncReport::new(
            SYNC_NOW_LABEL,
            facts,
            chunks,
            embedded,
            sessions,
        ))
    }

    /// Runs the trigger's sync when its precondition holds.
    ///
    /// [`SyncTrigger::OnWrite`] refreshes the cached document, [`SyncTrigger::OnSessionStart`]
    /// runs a full [`LocalCodingMemory::sync_now`], and [`SyncTrigger::OnInterval`] runs one
    /// only when this handle's interval gate is due. A trigger whose precondition does not
    /// hold answers a skipped report rather than silently succeeding.
    ///
    /// # Errors
    ///
    /// Returns the [`LocalCodingMemory::sync_now`] errors, plus
    /// [`Error::Validation`](openjiuwen_common::error::Error::Validation) for a zero
    /// interval.
    pub async fn on_trigger(&self, trigger: SyncTrigger) -> Result<SyncReport> {
        match trigger {
            SyncTrigger::OnWrite => self.sync_on_write().await,
            SyncTrigger::OnSessionStart => {
                let report = self.sync_now().await?;
                Ok(relabelled(&report, SyncTrigger::OnSessionStart.label()))
            }
            SyncTrigger::OnInterval(interval) => {
                if !self.interval_is_due(interval).await? {
                    return Ok(SyncReport::skipped(
                        SyncTrigger::OnInterval(interval).label(),
                    ));
                }
                let report = self.sync_now().await?;
                Ok(relabelled(
                    &report,
                    SyncTrigger::OnInterval(interval).label(),
                ))
            }
        }
    }

    /// Starts an opt-in background task that runs [`LocalCodingMemory::sync_now`] every
    /// `interval`, replacing any task this handle started before.
    ///
    /// The task is cancellation-aware and owned by this handle (its clones share it);
    /// [`LocalCodingMemory::stop_interval_sync`] stops it and dropping every handle ends it.
    /// A failing run is logged and does not stop the loop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) when
    /// `interval` is zero.
    pub async fn start_interval_sync(&self, interval: Duration) -> Result<()> {
        let _validated = IntervalScheduler::new(interval)?;
        self.stop_interval_sync().await;
        let store = self.clone();
        let token = CancellationToken::new();
        let watcher = token.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = watcher.cancelled() => break,
                    () = tokio::time::sleep(interval) => {
                        if let Err(error) = store.sync_now().await {
                            tracing::warn!(
                                target: TARGET,
                                error = %error,
                                "the interval sync failed"
                            );
                        }
                    }
                }
            }
        });
        *self.inner.interval_task.lock().await = Some(IntervalTask { token, task });
        Ok(())
    }

    /// Stops the interval task started by [`LocalCodingMemory::start_interval_sync`].
    ///
    /// A no-op when this handle started none.
    pub async fn stop_interval_sync(&self) {
        let active = self.inner.interval_task.lock().await.take();
        let Some(active) = active else {
            return;
        };
        active.token.cancel();
        active.task.abort();
        let _joined = active.task.await;
    }

    /// Whether this handle's interval task is running.
    ///
    /// Only the handle that started the task (or a clone of it) reports `true`; a handle
    /// opened separately over the same root has no task and reports `false`.
    pub async fn is_interval_sync_running(&self) -> bool {
        self.inner
            .interval_task
            .lock()
            .await
            .as_ref()
            .is_some_and(|active| !active.token.is_cancelled())
    }

    /// Refreshes the cached document and rebuilds the full-text index, returning the fact
    /// count. The embedding refresh is the caller's next step (no guard is held there).
    async fn reconcile_document(&self) -> Result<usize> {
        if !self.inner.fact_files {
            let mut state = self.inner.state.lock().await;
            let _reloaded = self.ensure_fresh(&mut state).await?;
            let facts = state.document.facts();
            let _indexed = self.rebuild_index_from(&facts).await?;
            return Ok(facts.len());
        }
        let guard = FileLock::acquire(&self.inner.lock_path).await?;
        let count = {
            let mut state = self.inner.state.lock().await;
            let _reloaded = self.refresh_facts(&mut state, true).await?;
            let facts = state.document.facts();
            let _indexed = self.rebuild_index_from(&facts).await?;
            facts.len()
        };
        drop(guard);
        Ok(count)
    }

    /// Refreshes the cached document only, as [`SyncTrigger::OnWrite`] asks.
    async fn sync_on_write(&self) -> Result<SyncReport> {
        let (reloaded, facts) = if self.inner.fact_files {
            let guard = FileLock::acquire(&self.inner.lock_path).await?;
            let outcome = {
                let mut state = self.inner.state.lock().await;
                let reloaded = self.refresh_facts(&mut state, true).await?;
                (reloaded, state.document.facts().len())
            };
            drop(guard);
            outcome
        } else {
            let mut state = self.inner.state.lock().await;
            let reloaded = self.ensure_fresh(&mut state).await?;
            (reloaded, state.document.facts().len())
        };
        if !reloaded {
            return Ok(SyncReport::skipped(SyncTrigger::OnWrite.label()));
        }
        let chunks = self.embedding_chunks().await;
        Ok(SyncReport::new(
            SyncTrigger::OnWrite.label(),
            facts,
            chunks,
            0,
            0,
        ))
    }

    /// Polls this handle's interval gate, creating it from the first interval seen and
    /// re-creating it when `interval` differs.
    async fn interval_is_due(&self, interval: Duration) -> Result<bool> {
        let mut slot = self.inner.interval.lock().await;
        if slot
            .as_ref()
            .is_none_or(|scheduler| scheduler.interval() != interval)
        {
            *slot = Some(IntervalScheduler::new(interval)?);
        }
        Ok(slot
            .as_mut()
            .is_some_and(|scheduler| scheduler.poll(Instant::now())))
    }
}

/// The report of `report` re-labelled as `trigger`.
fn relabelled(report: &SyncReport, trigger: &'static str) -> SyncReport {
    SyncReport::new(
        trigger,
        report.facts,
        report.chunks,
        report.embedded,
        report.sessions,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use openjiuwen_retrieval::embedding::Embedding;
    use tempfile::TempDir;

    use openjiuwen_common::error::Error as MemoryError;

    use crate::config::CodingMemoryConfig;
    use crate::lite::fact::CodingFact;
    use crate::lite::sync::SyncTrigger;
    use crate::record::MemoryScope;

    use super::super::embedding::test_support::CountingEmbedding;
    use super::super::LocalCodingMemory;

    /// The scope every test store and test fact uses.
    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    /// A project fact of [`scope`] named `name` with `content` as its body.
    fn fact(name: &str, content: &str) -> CodingFact {
        CodingFact::new(&scope(), name, content)
    }

    /// A builder over `temp` with the per-fact layout enabled.
    fn builder(temp: &TempDir) -> super::super::LocalCodingMemoryBuilder {
        LocalCodingMemory::builder(temp.path().to_path_buf(), CodingMemoryConfig::default())
            .scope(scope())
            .fact_files(true)
    }

    /// Opens a fact-file store rooted at `temp`.
    async fn open(temp: &TempDir) -> LocalCodingMemory {
        builder(temp)
            .open()
            .await
            .expect("the fact-file store opens")
    }

    /// Writes the session log the session count observes.
    async fn write_sessions(temp: &TempDir) {
        let directory = temp.path().join("sessions").join("2026-11-14");
        tokio::fs::create_dir_all(&directory)
            .await
            .expect("the session directory is created");
        tokio::fs::write(
            directory.join("x.jsonl"),
            "{\"role\":\"user\",\"content\":\"hello\"}\n{\"role\":\"assistant\",\"content\":\"hi\"}\n",
        )
        .await
        .expect("the session log is written");
    }

    #[tokio::test]
    async fn sync_now_reports_every_count_and_is_idempotent() {
        let temp = TempDir::new().expect("temp dir");
        let counter = Arc::new(CountingEmbedding::new());
        let embedding: Arc<dyn Embedding> = counter.clone();
        let store = builder(&temp)
            .embedding(embedding)
            .open()
            .await
            .expect("the embedded store opens");
        store
            .write(fact("build", "cargo build"))
            .await
            .expect("write");
        store
            .write(fact("test", "cargo test"))
            .await
            .expect("write");
        write_sessions(&temp).await;

        let first = store.sync_now().await.expect("the sync succeeds");
        assert!(first.ran);
        assert_eq!(first.trigger, "sync_now");
        assert_eq!(first.facts, 2);
        assert!(
            first.chunks >= 2,
            "every fact contributes a chunk: {first:?}"
        );
        assert_eq!(first.sessions, 2, "the session log records are counted");

        let calls = counter.document_calls();
        let second = store.sync_now().await.expect("the second sync succeeds");
        assert!(second.ran);
        assert_eq!(second.facts, first.facts);
        assert_eq!(second.chunks, first.chunks);
        assert_eq!(second.sessions, first.sessions);
        assert_eq!(second.embedded, 0, "unchanged text is a cache hit");
        assert_eq!(
            counter.document_calls(),
            calls,
            "a second sync must not call the model"
        );
    }

    #[tokio::test]
    async fn triggers_respect_their_preconditions() {
        let temp = TempDir::new().expect("temp dir");
        let store = open(&temp).await;
        store
            .write(fact("build", "cargo build"))
            .await
            .expect("write");

        let interval = Duration::from_secs(3600);
        let first = store
            .on_trigger(SyncTrigger::OnInterval(interval))
            .await
            .expect("the first interval run succeeds");
        assert!(first.ran);
        assert_eq!(first.trigger, "on_interval");
        assert_eq!(first.facts, 1);

        let second = store
            .on_trigger(SyncTrigger::OnInterval(interval))
            .await
            .expect("the second poll succeeds");
        assert!(!second.ran, "an unchanged interval gate must skip");
        assert_eq!(second.trigger, "on_interval");
        assert_eq!(second.facts, 0);

        let write = store
            .on_trigger(SyncTrigger::OnWrite)
            .await
            .expect("the write trigger succeeds");
        assert!(!write.ran, "nothing changed on disk");
        assert_eq!(write.trigger, "on_write");

        let start = store
            .on_trigger(SyncTrigger::OnSessionStart)
            .await
            .expect("the session-start trigger succeeds");
        assert!(start.ran, "a session start always runs");
        assert_eq!(start.trigger, "on_session_start");
        assert_eq!(start.facts, 1);

        let error = store
            .on_trigger(SyncTrigger::OnInterval(Duration::ZERO))
            .await
            .expect_err("a zero interval is rejected");
        assert!(matches!(error, MemoryError::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn the_interval_task_lifecycle_is_per_handle() {
        let temp = TempDir::new().expect("temp dir");
        let store = open(&temp).await;
        let other = open(&temp).await;
        assert!(!store.is_interval_sync_running().await);

        store
            .start_interval_sync(Duration::from_secs(3600))
            .await
            .expect("the task starts");
        assert!(store.is_interval_sync_running().await);
        assert!(
            !other.is_interval_sync_running().await,
            "the task is owned by one handle, never by a global"
        );

        store.stop_interval_sync().await;
        assert!(!store.is_interval_sync_running().await);

        store
            .start_interval_sync(Duration::from_secs(3600))
            .await
            .expect("the task starts again");
        store
            .start_interval_sync(Duration::from_secs(1800))
            .await
            .expect("the restart replaces the previous task");
        assert!(store.is_interval_sync_running().await);
        store.stop_interval_sync().await;
        assert!(!store.is_interval_sync_running().await);

        let error = store
            .start_interval_sync(Duration::ZERO)
            .await
            .expect_err("a zero interval is rejected");
        assert!(matches!(error, MemoryError::Validation(_)), "{error:?}");
        assert!(!store.is_interval_sync_running().await);
    }
}
