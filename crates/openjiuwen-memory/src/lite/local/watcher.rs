//! The debounced watcher lifecycle of the local store: install, stop, and query it.
//!
//! The watcher itself lives in [`crate::lite::watch`]; this module owns the bridge from
//! its events to a document reload, so externally edited Markdown becomes visible to
//! recall without any call having to notice the file first.
use tokio::sync::mpsc;

use openjiuwen_common::error::Result;

use crate::lite::watch::{MemoryWatcher, WatchEvent};

use super::layout::DEFAULT_DEBOUNCE;
use super::state::{ActiveWatch, LocalCodingMemory};
use super::TARGET;

impl LocalCodingMemory {
    /// Starts watching the root for `.md` changes and reindexes on every debounced burst.
    ///
    /// Calling it again replaces the previous watcher. Must be called from within a tokio
    /// runtime. External edits then become visible to [`LocalCodingMemory::search`]
    /// without any call having to notice the file first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) when a
    /// watcher is already running is *not* an error (the old one is stopped first); the
    /// watcher's own errors are returned when the filesystem watch cannot be installed.
    pub async fn start_watching(&self) -> Result<()> {
        self.stop_watching().await;
        let (sender, mut receiver) = mpsc::unbounded_channel::<WatchEvent>();
        let watcher = MemoryWatcher::start([self.inner.root.clone()], DEFAULT_DEBOUNCE, sender)?;
        let store = self.clone();
        let task = tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                tracing::debug!(
                    target: TARGET,
                    paths = event.paths.len(),
                    "memory watcher reported a markdown change"
                );
                let mut state = store.inner.state.lock().await;
                if let Err(error) = store.ensure_fresh(&mut state).await {
                    tracing::warn!(
                        target: TARGET,
                        error = %error,
                        "failed to reload the memory document after a filesystem change"
                    );
                }
            }
        });
        let mut slot = self.watch.lock().await;
        *slot = Some(ActiveWatch { watcher, task });
        Ok(())
    }

    /// Stops the watcher started by [`LocalCodingMemory::start_watching`].
    ///
    /// A no-op when no watcher is running.
    pub async fn stop_watching(&self) {
        let active = self.watch.lock().await.take();
        let Some(mut active) = active else {
            return;
        };
        active.watcher.stop().await;
        active.task.abort();
        let _joined = active.task.await;
    }

    /// Whether a watcher is currently running.
    pub async fn is_watching(&self) -> bool {
        self.watch
            .lock()
            .await
            .as_ref()
            .is_some_and(|active| active.watcher.is_running())
    }
}
