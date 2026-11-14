#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! Integration tests for the external long-term-memory provider bridge.
//!
//! The suite pins the observable contract of
//! `ReActAgent::set_memory_provider` against a `ScriptedModel`: the provider's
//! prompt block reaches the identity system message, `prefetch` reaches the
//! model as a system message, the advertised tool schemas are mounted and a model
//! tool call is dispatched to `handle_tool_call`, `sync_turn` sees the turn,
//! `on_session_end` runs once for an owned session and never for a supplied one,
//! and a failing provider never changes the run's output.
//!
//! Everything is offline and deterministic: the provider is an in-process fake
//! that records its rail traffic, the model is a `ScriptedModel`, and assertions
//! read the recorded call history instead of timing anything.
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::{Agent, AgentCard, ReActAgent, ReActAgentConfig, ResultType};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, Model, Role};
use openjiuwen_memory::{
    AddOptions, AddOutcome, CallOptions, ConversationTurn, MemoryFilter, MemoryProvider,
    MemoryRecord, MemoryScope, MemoryUpdate, NewMemory, ProviderCapabilities, ProviderHealth,
    RecallHit, SearchOptions,
};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use parking_lot::Mutex;
use serde_json::{json, Value};

/// Everything the fake provider recorded, plus the values it answers with.
#[derive(Default)]
struct Recorded {
    /// Prefetch queries the rail asked for.
    queries: Vec<String>,
    /// What `prefetch` answers.
    prefetch: String,
    /// What `system_prompt_block` answers.
    block: String,
    /// `(user, assistant)` pairs `sync_turn` saw.
    synced: Vec<(String, String)>,
    /// `(tool, args)` pairs `handle_tool_call` saw.
    calls: Vec<(String, Value)>,
    /// Turn lists `on_session_end` saw.
    ends: Vec<Vec<ConversationTurn>>,
}

/// A provider that records its rail traffic instead of touching a backend.
struct Fake {
    recorded: Arc<Mutex<Recorded>>,
    tools: Vec<Value>,
    capabilities: ProviderCapabilities,
    sync_fails: bool,
    end_fails: bool,
}

impl Fake {
    /// A ready provider advertising one `remember` tool.
    fn new(recorded: &Arc<Mutex<Recorded>>) -> Self {
        Self {
            recorded: Arc::clone(recorded),
            tools: vec![json!({
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "stores a memory",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            })],
            capabilities: ProviderCapabilities::local(false),
            sync_fails: false,
            end_fails: false,
        }
    }

    fn boxed(self) -> Arc<dyn MemoryProvider> {
        Arc::new(self)
    }
}

#[async_trait]
impl MemoryProvider for Fake {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }

    async fn initialize(&self) -> Result<()> {
        Ok(())
    }

    async fn add(
        &self,
        _scope: &MemoryScope,
        _memories: Vec<NewMemory>,
        _options: &AddOptions,
    ) -> Result<AddOutcome> {
        Ok(AddOutcome::default())
    }

    async fn search(
        &self,
        _scope: &MemoryScope,
        _query: &str,
        _options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        Ok(Vec::new())
    }

    async fn get(&self, _scope: &MemoryScope, _id: &str) -> Result<Option<MemoryRecord>> {
        Ok(None)
    }

    async fn update(
        &self,
        _scope: &MemoryScope,
        _id: &str,
        _update: MemoryUpdate,
        _options: &CallOptions,
    ) -> Result<MemoryRecord> {
        Err(Error::unsupported("the fake provider stores no records"))
    }

    async fn delete(&self, _scope: &MemoryScope, _id: &str) -> Result<bool> {
        Ok(false)
    }

    async fn list(&self, _filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        Ok(Vec::new())
    }

    async fn clear(&self, _filter: &MemoryFilter) -> Result<usize> {
        Ok(0)
    }

    async fn health(&self) -> Result<ProviderHealth> {
        Ok(ProviderHealth::healthy())
    }

    fn get_tool_schemas(&self) -> Vec<Value> {
        self.tools.clone()
    }

    async fn handle_tool_call(&self, tool: &str, args: &Value) -> Value {
        self.recorded
            .lock()
            .calls
            .push((tool.to_owned(), args.clone()));
        json!({"result": "stored", "id": "mem-1"})
    }

    async fn prefetch(&self, query: &str) -> String {
        let mut recorded = self.recorded.lock();
        recorded.queries.push(query.to_owned());
        recorded.prefetch.clone()
    }

    async fn sync_turn(&self, user_msg: &str, assistant_msg: &str) -> Result<()> {
        self.recorded
            .lock()
            .synced
            .push((user_msg.to_owned(), assistant_msg.to_owned()));
        if self.sync_fails {
            return Err(Error::memory("sync failed"));
        }
        Ok(())
    }

    fn system_prompt_block(&self) -> String {
        self.recorded.lock().block.clone()
    }

    async fn on_session_end(&self, messages: &[ConversationTurn]) -> Result<()> {
        self.recorded.lock().ends.push(messages.to_vec());
        if self.end_fails {
            return Err(Error::memory("end failed"));
        }
        Ok(())
    }
}

fn recorded() -> Arc<Mutex<Recorded>> {
    Arc::new(Mutex::new(Recorded::default()))
}

fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}

fn texts_with_role(messages: &[BaseMessage], role: &Role) -> Vec<String> {
    messages
        .iter()
        .filter(|message| &message.role == role)
        .map(|message| message.text().into_owned())
        .collect()
}

fn agent(model: Arc<dyn Model>) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("react-provider"));
    agent.configure(ReActAgentConfig {
        prompt_template: vec![BaseMessage::system("you are terse")],
        ..ReActAgentConfig::new()
    });
    agent.set_model(model);
    agent
}

#[tokio::test]
async fn the_prompt_block_and_prefetch_reach_the_model_window() {
    let state = recorded();
    {
        let mut recorded = state.lock();
        recorded.block = "Use the memory tools for durable facts.".to_owned();
        recorded.prefetch = "recalled: the user prefers helix".to_owned();
    }
    let model = ScriptedModel::builder().respond_text("hello back").build();
    let agent = agent(shared(&model));
    agent.set_memory_provider(Fake::new(&state).boxed());

    let output = agent.invoke(json!("which editor?"), None).await.unwrap();
    assert_eq!(output.result_type, ResultType::Answer);

    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    let systems = texts_with_role(&calls[0].messages, &Role::System);
    assert_eq!(
        systems,
        vec![
            "you are terse\n\nUse the memory tools for durable facts.".to_owned(),
            "recalled: the user prefers helix".to_owned(),
        ],
        "{:?}",
        calls[0].messages
    );
    assert_eq!(state.lock().queries, vec!["which editor?".to_owned()]);
}

#[tokio::test]
async fn an_empty_provider_block_leaves_the_identity_alone() {
    let state = recorded();
    let model = ScriptedModel::builder().respond_text("hello").build();
    let agent = agent(shared(&model));
    agent.set_memory_provider(Fake::new(&state).boxed());

    agent.invoke(json!("hi"), None).await.unwrap();
    let systems = texts_with_role(&model.calls()[0].messages, &Role::System);
    assert_eq!(systems, vec!["you are terse".to_owned()]);
}

#[tokio::test]
async fn provider_tool_schemas_are_mounted_and_dispatched() {
    let state = recorded();
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "remember",
            json!({"text": "likes helix"}),
        ))
        .respond_text("noted")
        .build();
    let agent = agent(shared(&model));
    agent.set_memory_provider(Fake::new(&state).boxed());

    let names: Vec<String> = agent.tools().into_iter().map(|card| card.name).collect();
    assert_eq!(names, vec!["remember".to_owned()]);

    let output = agent
        .invoke(json!("remember I like helix"), None)
        .await
        .unwrap();
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "noted");
    assert_eq!(
        state.lock().calls,
        vec![("remember".to_owned(), json!({"text": "likes helix"}))]
    );

    // The provider's payload reached the model as the tool message.
    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    let tool_texts = texts_with_role(&calls[1].messages, &Role::Tool);
    assert_eq!(tool_texts.len(), 1, "{:?}", calls[1].messages);
    assert!(tool_texts[0].contains("mem-1"), "{}", tool_texts[0]);
    // The schema is visible to the model.
    let tools = calls[0].request.tools.clone().unwrap_or(Value::Null);
    assert_eq!(tools[0]["function"]["name"], json!("remember"));
}

#[tokio::test]
async fn sync_turn_sees_the_turn_and_on_session_end_runs_for_an_owned_session() {
    let state = recorded();
    let model = ScriptedModel::builder().respond_text("helix").build();
    let agent = agent(shared(&model));
    agent.set_memory_provider(Fake::new(&state).boxed());

    agent.invoke(json!("which editor?"), None).await.unwrap();

    let recorded = state.lock();
    assert_eq!(
        recorded.synced,
        vec![("which editor?".to_owned(), "helix".to_owned())]
    );
    assert_eq!(
        recorded.ends,
        vec![vec![
            ConversationTurn::new("user", "which editor?"),
            ConversationTurn::new("assistant", "helix"),
        ]]
    );
}

#[tokio::test]
async fn a_supplied_session_is_not_finalized_by_the_loop() {
    let state = recorded();
    let model = ScriptedModel::builder().respond_text("helix").build();
    let agent = agent(shared(&model));
    agent.set_memory_provider(Fake::new(&state).boxed());

    let session: Arc<dyn openjiuwen_session::Session> =
        Arc::new(openjiuwen_session::AgentSession::new("s1"));
    agent
        .invoke(
            json!({"query": "which editor?", "conversation_id": "s1"}),
            Some(session),
        )
        .await
        .unwrap();
    assert!(state.lock().ends.is_empty());

    // The caller finalizes the conversation itself.
    let turns = vec![ConversationTurn::new("user", "which editor?")];
    agent.end_memory_session(&turns).await.unwrap();
    assert_eq!(state.lock().ends, vec![turns]);
}

#[tokio::test]
async fn a_failing_sync_turn_and_session_end_do_not_change_the_output() {
    let state = recorded();
    let model = ScriptedModel::builder().respond_text("helix").build();
    let agent = agent(shared(&model));
    agent.set_memory_provider(
        Fake {
            sync_fails: true,
            end_fails: true,
            ..Fake::new(&state)
        }
        .boxed(),
    );

    let output = agent.invoke(json!("which editor?"), None).await.unwrap();
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(output.output, "helix");
    assert_eq!(state.lock().synced.len(), 1);
    assert_eq!(state.lock().ends.len(), 1);
}

#[tokio::test]
async fn an_unavailable_provider_is_still_mounted() {
    let state = recorded();
    let model = ScriptedModel::builder().respond_text("helix").build();
    let agent = agent(shared(&model));
    // No declared capability means `is_available()` answers `false`.
    agent.set_memory_provider(
        Fake {
            capabilities: ProviderCapabilities::none(),
            ..Fake::new(&state)
        }
        .boxed(),
    );

    assert!(agent.memory_provider().is_some());
    let names: Vec<String> = agent.tools().into_iter().map(|card| card.name).collect();
    assert_eq!(names, vec!["remember".to_owned()]);
    assert!(agent
        .rails()
        .rails()
        .iter()
        .any(|rail| rail.name() == "memory_provider"));

    let output = agent.invoke(json!("hi"), None).await.unwrap();
    assert_eq!(output.result_type, ResultType::Answer);
}

#[tokio::test]
async fn clear_memory_provider_removes_the_rail_and_keeps_the_tools() {
    let state = recorded();
    let model = ScriptedModel::builder().respond_text("helix").build();
    let agent = agent(shared(&model));
    agent.set_memory_provider(Fake::new(&state).boxed());
    assert!(agent
        .rails()
        .rails()
        .iter()
        .any(|rail| rail.name() == "memory_provider"));

    agent.clear_memory_provider();
    assert!(agent.memory_provider().is_none());
    assert!(!agent
        .rails()
        .rails()
        .iter()
        .any(|rail| rail.name() == "memory_provider"));
    let names: Vec<String> = agent.tools().into_iter().map(|card| card.name).collect();
    assert_eq!(names, vec!["remember".to_owned()]);

    // Later runs neither prefetch, nor sync the turn, nor finalize the session.
    agent.invoke(json!("which editor?"), None).await.unwrap();
    let recorded = state.lock();
    assert!(recorded.queries.is_empty());
    assert!(recorded.synced.is_empty());
    assert!(recorded.ends.is_empty());
}

#[tokio::test]
async fn end_memory_session_without_a_provider_is_a_no_op() {
    let model = ScriptedModel::builder().respond_text("helix").build();
    let agent = agent(shared(&model));
    agent
        .end_memory_session(&[ConversationTurn::new("user", "hi")])
        .await
        .unwrap();
}
