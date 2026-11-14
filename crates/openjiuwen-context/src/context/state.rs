//! Context persistence: `ContextState`, `save_state` and `load_state`.
//!
//! The state carries the stored messages, the revision they were saved at, and
//! the processor checkpoints taken from the pipeline, so a restored context
//! continues the saved revision and keeps cross-run processor counters.
use std::sync::atomic::Ordering;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};

use super::{ConversationContext, ModelContext};

/// The persistable state of a [`ConversationContext`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextState {
    /// The stored messages, oldest first.
    pub messages: Vec<BaseMessage>,
    /// The revision the messages were saved at.
    pub revision: u64,
    /// Process-level checkpoint state, keyed by processor name.
    ///
    /// Filled from
    /// [`ContextPipeline::save_state`](crate::processor::ContextPipeline::save_state)
    /// and restored through
    /// [`ContextPipeline::load_state`](crate::processor::ContextPipeline::load_state),
    /// so a processor's cross-run counters survive the process that wrote
    /// them. Empty when no processor keeps counters, and when the state was
    /// written before processors had any.
    pub processors: std::collections::BTreeMap<String, serde_json::Value>,
}

impl ConversationContext {
    /// Serializes the history and the processor checkpoints for persistence.
    pub async fn save_state(&self) -> ContextState {
        ContextState {
            messages: self.messages().await,
            revision: self.revision(),
            processors: self.pipeline.save_state(),
        }
    }

    /// Replaces the history with `state` and restores the processor counters it
    /// carries.
    ///
    /// The revision stored in `state` becomes the context's revision, so a
    /// restored context continues the sequence the saved one used.
    ///
    /// # Errors
    ///
    /// Currently infallible; the signature keeps the door open for
    /// implementations that persist eagerly.
    pub async fn load_state(&self, state: ContextState) -> Result<()> {
        let ContextState {
            messages,
            revision,
            processors,
        } = state;
        *self.messages.write().await = messages;
        self.revision.store(revision, Ordering::SeqCst);
        self.invalidate_stats().await;
        self.pipeline.load_state(&processors);
        Ok(())
    }
}
