//! The last two LLM-bound windows and the prompt-cache diff between them.
//!
//! `prepare` records every assembled window here; `detect_window_change` diffs
//! the two most recent ones, so the diff stays O(messages) and never touches the
//! stored history.
use std::sync::PoisonError;
use std::time::Instant;

use crate::policy::{
    detect_context_window_change, ContextWindowChange, ContextWindowSnapshot, ResolvedContextWindow,
};

use super::ConversationContext;

impl ConversationContext {
    /// The context window the engine resolves for this context's configuration.
    #[must_use]
    pub fn resolved_window(&self) -> ResolvedContextWindow {
        self.config.resolved_window()
    }

    /// Diffs the most recently prepared window against the one before it.
    ///
    /// The diff is the KV-cache primitive:
    /// [`ContextWindowChange::Appended`] keeps the whole prefix warm, while a
    /// system, tool or content change invalidates the corresponding suffix.
    /// Before the first [`ConversationContext::prepare`] the answer is
    /// [`ContextWindowChange::Unchanged`] with zero messages, and the call never
    /// mutates the stored history.
    #[must_use]
    pub fn detect_window_change(&self) -> ContextWindowChange {
        let windows = self.windows.lock().unwrap_or_else(PoisonError::into_inner);
        match (&windows.previous, &windows.current) {
            (Some(previous), Some(current)) => detect_context_window_change(previous, current),
            (None, Some(current)) => detect_context_window_change(current, current),
            (_, None) => ContextWindowChange::Unchanged { messages: 0 },
        }
    }

    /// When the most recent LLM-bound window was prepared.
    #[must_use]
    pub fn last_context_window_access(&self) -> Option<Instant> {
        self.windows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .accessed
    }

    /// The most recent LLM-bound window, when one was prepared.
    #[must_use]
    pub(crate) fn last_window_snapshot(&self) -> Option<ContextWindowSnapshot> {
        self.windows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .current
            .clone()
    }

    /// Records `window` as the current LLM-bound window.
    pub(super) fn track_window(&self, window: ContextWindowSnapshot) {
        let mut windows = self.windows.lock().unwrap_or_else(PoisonError::into_inner);
        let previous = windows.current.take();
        windows.previous = previous;
        windows.current = Some(window);
        windows.accessed = Some(Instant::now());
    }
}
