//! Seeding the kernel `"main"` conversation context from a persisted history.
//!
//! A host that reconstructed an earlier transcript (for example a
//! `persistent_agent::ResumePlan`, which is deliberately not imported here —
//! `agent` never depends on that module) has no other way to push those
//! messages into a normal turn: the kernel creates the `"main"` context on the
//! first model call and derives the conversation from that context alone.
//! [`crate::agent::DeepAgent::seed_history`] fills that context before the turn
//! runs, so the model sees the restored transcript ahead of the new query.
//!
//! ## Pool key
//!
//! The kernel pool keys every context by
//! `(session.id(), normalized context id)`; the main conversation is keyed by
//! `"main"` ([`openjiuwen_agent::usage::MAIN_CONTEXT_ID`]). Seeding therefore
//! targets exactly the entry the next `ReActAgent::invoke` looks up.
//!
//! ## Ordering
//!
//! Seed after the last [`crate::agent::DeepAgent::configure`] (or after
//! [`crate::agent::DeepAgentBuilder::build`]) and before the `invoke`/`stream`
//! it should affect: `configure` re-applies the kernel config, which replaces
//! the context engine and drops every pooled context. A seed performed earlier
//! is silently discarded by the next `configure`.
//!
//! ## Recipe
//!
//! - an existing context is replaced wholesale with
//!   [`openjiuwen_context::ModelContext::set_messages`];
//! - an absent context is created with [`openjiuwen_context::ContextOptions`]
//!   `history`, which the engine applies only when the pool key is new.
//!
//! Both branches leave the same state, so seeding is idempotent: seeding the
//! same history twice, or seeding again before any invocation, yields one copy.
//! An empty history is legal and simply leaves the context empty.

use std::sync::Arc;

use openjiuwen_agent::usage::MAIN_CONTEXT_ID;
use openjiuwen_common::error::Result;
use openjiuwen_context::{ContextOptions, ModelContext};
use openjiuwen_foundation::BaseMessage;
use openjiuwen_session::Session;

use super::DeepAgent;

impl DeepAgent {
    /// Seeds the kernel `"main"` conversation context of `session` from a
    /// persisted history.
    ///
    /// The context is the one the next `invoke`/`stream` on the same
    /// [`Session`] will use, so the seeded messages are sent to the model ahead
    /// of the new user query. An already existing context is replaced
    /// wholesale; an absent one is created with the history applied. Empty
    /// `history` is legal and leaves the context empty.
    ///
    /// Call this after the last [`Self::configure`] (or after
    /// [`crate::agent::DeepAgentBuilder::build`]) and before the turn:
    /// `configure` re-applies the kernel config and drops every pooled context,
    /// so an earlier seed is discarded. The operation is idempotent — seeding
    /// the same history again (or seeding twice before any invocation) leaves
    /// exactly one copy.
    ///
    /// # Errors
    ///
    /// Returns the kernel error unchanged:
    /// [`Error::config`](openjiuwen_common::error::Error::config) when the
    /// context configuration is rejected while creating an absent context, or
    /// the context's own error when a seeded message is rejected by
    /// [`ModelContext::set_messages`].
    pub async fn seed_history(
        &self,
        session: &Arc<dyn Session>,
        history: Vec<BaseMessage>,
    ) -> Result<()> {
        let engine = self.inner.context_engine();
        match engine.get_context(session.id(), MAIN_CONTEXT_ID).await {
            Some(context) => context.set_messages(history).await,
            None => engine
                .create_context_with(
                    MAIN_CONTEXT_ID,
                    ContextOptions {
                        session: Some(Arc::clone(session)),
                        history,
                        ..ContextOptions::default()
                    },
                )
                .await
                .map(|_| ()),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use openjiuwen_agent::{Agent, AgentCard};
    use openjiuwen_common::error::Error;
    use openjiuwen_foundation::BaseMessage;
    use openjiuwen_session::{AgentSession, Session};
    use openjiuwen_test_support::{ModelCall, ScriptedModel};
    use serde_json::json;

    use super::DeepAgent;
    use crate::agent::{load_state, seed_state, DeepAgentBuilder, SESSION_STATE_KEY};
    use crate::schema::DeepAgentConfig;

    /// Builds a fully wired agent with a supplied model (no lazy resolution).
    fn build_agent(model: ScriptedModel) -> Arc<DeepAgent> {
        DeepAgentBuilder::new(AgentCard::new("seed-agent").with_description("Seeding tests"))
            .config(DeepAgentConfig::default().with_language("en".to_string()))
            .model(Arc::new(model))
            .build()
            .expect("agent builds")
    }

    /// A fresh in-memory session with `id`.
    fn session(id: &str) -> Arc<dyn Session> {
        Arc::new(AgentSession::new(id))
    }

    /// Index of the first recorded message whose text contains `needle`.
    fn index_of(call: &ModelCall, needle: &str) -> usize {
        call.messages
            .iter()
            .position(|message| message.text().contains(needle))
            .unwrap_or_else(|| panic!("no message contains {needle:?}"))
    }

    /// How many recorded messages' text contains `needle`.
    fn text_count(call: &ModelCall, needle: &str) -> usize {
        call.messages
            .iter()
            .filter(|message| message.text().contains(needle))
            .count()
    }

    #[tokio::test]
    async fn seeded_history_reaches_the_model_before_the_new_query() {
        let scripted = ScriptedModel::text("fresh answer");
        let agent = build_agent(scripted.clone());
        let session = session("seed-history");

        agent
            .seed_history(
                &session,
                vec![
                    BaseMessage::user("earlier question"),
                    BaseMessage::assistant("earlier answer"),
                ],
            )
            .await
            .unwrap();

        agent
            .invoke(json!({"query": "new question"}), Some(session.clone()))
            .await
            .expect("invocation succeeds");

        let call = scripted.last_call().expect("one model call recorded");
        let new_query = index_of(&call, "new question");
        assert!(index_of(&call, "earlier question") < new_query);
        assert!(index_of(&call, "earlier answer") < new_query);
    }

    #[tokio::test]
    async fn seeding_is_idempotent_and_replaces_on_a_pool_hit() {
        let scripted = ScriptedModel::text("ok");
        let agent = build_agent(scripted.clone());
        let session = session("seed-idempotent");
        let history = vec![
            BaseMessage::user("earlier question"),
            BaseMessage::assistant("earlier answer"),
        ];

        // First seed creates the context, the second one hits the pool and
        // replaces the same history: exactly one copy remains.
        agent.seed_history(&session, history.clone()).await.unwrap();
        agent.seed_history(&session, history).await.unwrap();

        agent
            .invoke(json!({"query": "new question"}), Some(session.clone()))
            .await
            .expect("first invocation succeeds");
        let first = scripted.last_call().expect("one model call recorded");
        assert_eq!(text_count(&first, "earlier question"), 1);
        assert_eq!(text_count(&first, "earlier answer"), 1);

        // A different history on the already-created context replaces the whole
        // transcript instead of appending to it.
        agent
            .seed_history(&session, vec![BaseMessage::user("replacement question")])
            .await
            .unwrap();

        agent
            .invoke(json!({"query": "second question"}), Some(session.clone()))
            .await
            .expect("second invocation succeeds");
        let second = scripted.last_call().expect("two model calls recorded");
        assert_eq!(text_count(&second, "earlier question"), 0);
        assert_eq!(text_count(&second, "earlier answer"), 0);
        assert!(index_of(&second, "replacement question") < index_of(&second, "second question"));
    }

    #[tokio::test]
    async fn seeding_after_the_last_configure_reaches_the_model() {
        let scripted = ScriptedModel::text("ok");
        let agent = build_agent(scripted.clone());
        // `configure` re-applies the kernel config and drops every pooled
        // context, so the seed has to follow it — this is the documented order.
        agent.configure(DeepAgentConfig::default().with_language("en".to_string()));
        let session = session("seed-configure");

        agent
            .seed_history(
                &session,
                vec![BaseMessage::user("configured earlier question")],
            )
            .await
            .unwrap();

        agent
            .invoke(json!({"query": "configured question"}), Some(session))
            .await
            .expect("invocation succeeds");

        let call = scripted.last_call().expect("one model call recorded");
        assert!(
            index_of(&call, "configured earlier question") < index_of(&call, "configured question")
        );
    }

    #[tokio::test]
    async fn seed_state_replaces_the_deepagent_lane() {
        let session = session("seed-state");

        seed_state(session.as_ref(), json!({"iteration": 3, "plan_mode": null}))
            .await
            .unwrap();
        let loaded = load_state(session.as_ref()).await.unwrap();
        assert_eq!(loaded.iteration, 3);

        // Replace, not deep-merge: no key of the first payload survives.
        seed_state(
            session.as_ref(),
            json!({"iteration": 5, "host_key": "kept"}),
        )
        .await
        .unwrap();
        let payload = session
            .get_state(SESSION_STATE_KEY)
            .await
            .unwrap()
            .expect("payload present");
        assert_eq!(payload, json!({"iteration": 5, "host_key": "kept"}));
        assert_eq!(load_state(session.as_ref()).await.unwrap().iteration, 5);
    }

    #[tokio::test]
    async fn seed_state_rejects_non_object_payloads() {
        let session = session("seed-state-invalid");
        let error = seed_state(session.as_ref(), json!("nope"))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
    }
}
