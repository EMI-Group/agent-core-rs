//! Session persistence for the context pool.
//!
//! Every context of a session is stored under the session state key `context` as
//! one map of context id → serialized [`ContextState`]; a context that fails to
//! decode is logged and skipped, so one corrupt entry cannot block a session.
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use serde_json::{Map, Value};

use crate::context::{ContextState, ConversationContext, ModelContext};

use super::{ContextEngine, ContextOptions, STATE_KEY};

impl ContextEngine {
    /// Persists every context of `session` under the session state key `context`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when a state cannot be serialized or the
    /// session rejects the write.
    pub async fn save_contexts(&self, session: &dyn Session) -> Result<usize> {
        let session_id = session.id();
        let contexts: Vec<Arc<ConversationContext>> = {
            let pool = self.contexts.read().await;
            pool.values()
                .filter(|context| context.session_id() == session_id)
                .map(Arc::clone)
                .collect()
        };
        let mut states = Map::new();
        for context in &contexts {
            let state = context.save_state().await;
            let value = serde_json::to_value(state).map_err(|error| {
                Error::Session(format!(
                    "failed to serialize context '{}': {error}",
                    context.context_id()
                ))
            })?;
            states.insert(context.context_id().to_owned(), value);
        }
        let count = states.len();
        session
            .update_state(serde_json::json!({ (STATE_KEY): Value::Object(states) }))
            .await?;
        Ok(count)
    }

    /// Restores every context stored under the session state key `context`.
    ///
    /// Entries that do not decode are logged and skipped so one corrupt context
    /// cannot block the rest of a session. Restored contexts do not carry a
    /// session handle, because this method only borrows the session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the session state cannot be read, and
    /// [`Error::Config`] when a context cannot be created.
    pub async fn load_contexts(&self, session: &dyn Session) -> Result<usize> {
        let Some(value) = session.get_state(STATE_KEY).await? else {
            return Ok(0);
        };
        let Some(entries) = value.as_object() else {
            tracing::warn!("session context state is not an object; skipping it");
            return Ok(0);
        };
        let entries: Vec<(String, Value)> = entries
            .iter()
            .map(|(context_id, state)| (context_id.clone(), state.clone()))
            .collect();

        self.config.validate()?;
        let mut loaded = 0;
        for (context_id, value) in entries {
            let state: ContextState = match serde_json::from_value(value) {
                Ok(state) => state,
                Err(error) => {
                    tracing::warn!(%context_id, %error, "skipping corrupt context state");
                    continue;
                }
            };
            let context = self
                .ensure_context(&context_id, session.id(), ContextOptions::default())
                .await?;
            context.load_state(state).await?;
            loaded += 1;
        }
        Ok(loaded)
    }
}
