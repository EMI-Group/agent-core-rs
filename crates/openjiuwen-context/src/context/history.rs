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

use super::{ConversationContext, ModelContext};

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
