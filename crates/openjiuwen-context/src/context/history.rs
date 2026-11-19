//! Stored history: the [`ModelContext`] implementation, mutation and statistics.
//!
//! Every mutation bumps the revision and drops the cached statistics, so a
//! per-role summary never describes a history that has already changed;
//! compression totals are merged into the statistics on every read.
use std::sync::atomic::Ordering;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role, TokenCounter};

use crate::stats::{compute_stats, ContextStats};
use crate::tokens::count_message;
use crate::window::dialogue_round_ranges;

use super::{ConversationContext, ModelContext};

/// What one in-place append changed.
///
/// The two totals describe the history right after the append, which is what
/// [`ContextEngine::observe`] reports; the compression counters
/// [`ModelContext::statistics`] adds are deliberately absent because an append
/// never compresses.
///
/// [`ContextEngine::observe`]: crate::engine::ContextEngine::observe
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AppendOutcome {
    /// Messages the rolling limit evicted.
    pub(crate) evicted: usize,
    /// Messages in the history after the append.
    pub(crate) total_messages: usize,
    /// Tokens in the history after the append.
    pub(crate) total_tokens: usize,
}

/// Whether a message carries no payload at all.
///
/// Only [`MessageContent::Text`] that is empty and a [`MessageContent::Parts`]
/// list with no entries count as empty; a part list holding an image is *not*
/// empty, so multimodal messages are never rejected.
fn is_empty_content(content: &MessageContent) -> bool {
    match content {
        MessageContent::Text(text) => text.is_empty(),
        MessageContent::Parts(parts) => parts.is_empty(),
    }
}

impl ConversationContext {
    /// Records one compression pass in the statistics.
    pub(crate) async fn record_compression(&self, tokens_before: usize, tokens_after: usize) {
        let mut totals = self.compression_totals.lock().await;
        totals.0 = totals.0.saturating_add(1);
        totals.1 = totals.1.saturating_add(tokens_before);
        totals.2 = totals.2.saturating_add(tokens_after);
    }

    /// Drops the cached statistics so the next read recomputes them.
    pub(super) async fn invalidate_stats(&self) {
        *self.stats_cache.lock().await = None;
    }

    /// Bumps the revision and invalidates the statistics cache.
    ///
    /// The message guard must already be released: the cache and the message
    /// vector are locked in that order.
    async fn bump_revision(&self) {
        self.revision.fetch_add(1, Ordering::SeqCst);
        self.invalidate_stats().await;
    }

    /// Rejects the messages this context cannot store.
    ///
    /// The rule is deliberately small: a user or tool message must carry a
    /// payload, because those roles are meaningless when empty. Assistant
    /// messages may be empty (a tool-call turn has no text) and system messages
    /// are left to the caller, matching Python's tolerant context.
    fn validate_message(message: &BaseMessage) -> Result<()> {
        match message.role {
            Role::User | Role::Tool if is_empty_content(&message.content) => {
                Err(Error::Validation(format!(
                    "{} message content must not be empty",
                    message.role
                )))
            }
            _ => Ok(()),
        }
    }

    /// Appends one message in place and applies the rolling message limit.
    ///
    /// This is the append seam for a hot path: the history vector is pushed to
    /// and trimmed under its own write lock, so `n` appends cost `O(n)` moves
    /// instead of the `O(n²)` clones that `messages()` + `set_messages()` copy,
    /// and the revision is bumped exactly once per append.
    ///
    /// Eviction removes the oldest non-system messages, never a system message
    /// and never the message just appended, until the history fits `limit`.
    ///
    /// Token totals come from the statistics cache: the append adds the new
    /// message's count and subtracts the count of every message it evicted,
    /// which is exact because [`TokenCounter::count_messages`] is defined as the
    /// per-message sum. A cold or contended cache costs one [`compute_stats`]
    /// over the trimmed history instead — never one per append. The round count
    /// is always recomputed from the trimmed history, because appending or
    /// evicting moves the last round's boundary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `message` has no payload but needs one.
    pub(crate) async fn append_message(
        &self,
        message: BaseMessage,
        limit: Option<usize>,
    ) -> Result<AppendOutcome> {
        Self::validate_message(&message)?;
        let added_tokens = count_message(self.counter.as_ref(), &message);
        let mut messages = self.messages.write().await;
        // The revision is read under the history lock, so a concurrent append
        // that has already pushed its message is always part of this append's
        // base.
        let revision_before = self.revision.load(Ordering::SeqCst);
        // The cache is probed with `try_lock` because the message guard is held
        // and blocking here would invert the cache → messages order
        // `statistics` uses; a contended cache only costs this append the fresh
        // computation below.
        let mut cached = match self.stats_cache.try_lock() {
            Ok(cache) => match cache.as_ref() {
                Some((cached, stats)) if *cached == revision_before => Some(stats.clone()),
                _ => None,
            },
            Err(_) => None,
        };
        if let Some(stats) = cached.as_mut() {
            stats.record_message(&message, added_tokens);
        }
        messages.push(message);
        let mut evicted = 0_usize;
        if let Some(limit) = limit {
            while messages.len() > limit {
                // The newest message (the one just appended) is always last and
                // is never evicted.
                let candidate = messages
                    .iter()
                    .enumerate()
                    .take(messages.len().saturating_sub(1))
                    .find(|(_, candidate)| candidate.role != Role::System)
                    .map(|(index, _)| index);
                let Some(index) = candidate else {
                    break;
                };
                let removed = messages.remove(index);
                if let Some(stats) = cached.as_mut() {
                    let tokens = count_message(self.counter.as_ref(), &removed);
                    stats.unrecord_message(&removed, tokens);
                }
                evicted += 1;
            }
        }
        let stats = match cached {
            Some(mut stats) => {
                stats.total_dialogues = dialogue_round_ranges(&messages).len();
                stats
            }
            None => compute_stats(&messages, self.counter.as_ref()),
        };
        let total_messages = messages.len();
        let total_tokens = stats.total_tokens;
        // Bumped inside the lock so the mutation and its revision are one step
        // for every other reader.
        let revision_after = self
            .revision
            .fetch_add(1, Ordering::SeqCst)
            .saturating_add(1);
        drop(messages);
        // Publish the updated statistics unless another append moved the
        // revision in the meantime; that append owns the newer totals.
        let mut cache = self.stats_cache.lock().await;
        if cache
            .as_ref()
            .is_none_or(|(cached, _)| *cached == revision_before)
        {
            *cache = Some((revision_after, stats));
        }
        drop(cache);
        Ok(AppendOutcome {
            evicted,
            total_messages,
            total_tokens,
        })
    }
}

#[async_trait]
impl ModelContext for ConversationContext {
    fn context_id(&self) -> &str {
        &self.context_id
    }

    fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn messages(&self) -> Vec<BaseMessage> {
        self.messages.read().await.clone()
    }

    async fn set_messages(&self, messages: Vec<BaseMessage>) -> Result<()> {
        *self.messages.write().await = messages;
        self.bump_revision().await;
        Ok(())
    }

    async fn add_message(&self, message: BaseMessage) -> Result<()> {
        Self::validate_message(&message)?;
        self.messages.write().await.push(message);
        self.bump_revision().await;
        Ok(())
    }

    async fn pop_messages(&self, count: usize) -> Result<Vec<BaseMessage>> {
        if count == 0 {
            return Err(Error::Validation(
                "pop size should be larger than 0".to_owned(),
            ));
        }
        let mut messages = self.messages.write().await;
        let keep = messages.len().saturating_sub(count);
        let popped = messages.split_off(keep);
        drop(messages);
        if !popped.is_empty() {
            self.bump_revision().await;
        }
        Ok(popped)
    }

    async fn clear_messages(&self) -> Result<()> {
        self.messages.write().await.clear();
        self.bump_revision().await;
        Ok(())
    }

    fn token_counter(&self) -> Arc<dyn TokenCounter> {
        self.counter.clone()
    }

    async fn statistics(&self) -> ContextStats {
        let revision = self.revision();
        let base = {
            let mut cache = self.stats_cache.lock().await;
            match cache.as_ref() {
                Some((cached, stats)) if *cached == revision => stats.clone(),
                _ => {
                    let messages = self.messages.read().await.clone();
                    let stats = compute_stats(&messages, self.counter.as_ref());
                    *cache = Some((revision, stats.clone()));
                    stats
                }
            }
        };
        let totals = *self.compression_totals.lock().await;
        ContextStats {
            compressions: totals.0,
            tokens_before_compression: totals.1,
            tokens_after_compression: totals.2,
            ..base
        }
    }

    fn revision(&self) -> u64 {
        self.revision.load(Ordering::SeqCst)
    }
}
