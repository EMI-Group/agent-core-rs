//! Unit tests for the provider bridge: prompt block, prefetch placement, turn
//! write-back, schema-derived tools, and the session-finalization gate.
#![allow(
    clippy::assert_is_empty,
    clippy::new_ret_no_self,
    clippy::unnecessary_literal_bound
)]
use super::*;
use crate::rails::RailEvent;
use async_trait::async_trait;
use openjiuwen_common::error::Error;
use openjiuwen_memory::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, ProviderCapabilities, ProviderHealth, RecallHit, SearchOptions,
};
use parking_lot::Mutex;
use serde_json::json;

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
    sync_fails: bool,
    end_fails: bool,
}

impl Fake {
    fn new(recorded: &Arc<Mutex<Recorded>>) -> Arc<dyn MemoryProvider> {
        Arc::new(Self {
            recorded: Arc::clone(recorded),
            tools: vec![json!({
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "stores a memory",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                }
            })],
            sync_fails: false,
            end_fails: false,
        })
    }
}

#[async_trait]
impl MemoryProvider for Fake {
    fn name(&self) -> &str {
        "fake"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::local(false)
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
        Err(Error::unsupported("fake provider does not store records"))
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
        json!({"result": "stored"})
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

fn system(text: &str) -> BaseMessage {
    BaseMessage::system(text)
}

fn context(event: RailEvent, messages: Vec<BaseMessage>) -> RailContext {
    RailContext::new(event).with_messages(messages)
}

#[test]
fn the_rail_is_named_memory_provider_by_default() {
    let state = recorded();
    let rail = ProviderMemoryRail::new(Fake::new(&state));
    assert_eq!(rail.name(), "memory_provider");
    assert_eq!(rail.provider().name(), "fake");
    assert_eq!(
        ProviderMemoryRail::new(Fake::new(&state))
            .with_name("other")
            .name(),
        "other"
    );
}

#[tokio::test]
async fn before_invoke_publishes_a_non_empty_prompt_block() {
    let state = recorded();
    state.lock().block = "Use the memory tools for durable facts.".to_owned();
    let rail = ProviderMemoryRail::new(Fake::new(&state));
    let mut ctx = context(RailEvent::BeforeInvoke, Vec::new());
    rail.before_invoke(&mut ctx).await.unwrap();
    assert_eq!(
        injected_prompt_block(&ctx),
        Some("Use the memory tools for durable facts.")
    );
}

#[tokio::test]
async fn an_empty_prompt_block_publishes_nothing() {
    let state = recorded();
    let rail = ProviderMemoryRail::new(Fake::new(&state));
    let mut ctx = context(RailEvent::BeforeInvoke, Vec::new());
    rail.before_invoke(&mut ctx).await.unwrap();
    assert!(injected_prompt_block(&ctx).is_none());
    assert!(!ctx.extra.contains_key(MEMORY_PROVIDER_BLOCK_KEY));
}

#[tokio::test]
async fn prefetch_lands_after_the_leading_system_messages_once() {
    let state = recorded();
    state.lock().prefetch = "recalled: the user prefers helix".to_owned();
    let rail = ProviderMemoryRail::new(Fake::new(&state));
    let mut ctx = context(
        RailEvent::BeforeModelCall,
        vec![
            system("identity"),
            system("second"),
            BaseMessage::user("hi"),
        ],
    );
    rail.before_model_call(&mut ctx).await.unwrap();
    assert_eq!(state.lock().queries, vec!["hi".to_owned()]);
    assert_eq!(ctx.messages.len(), 4);
    assert_eq!(ctx.messages[0].text(), "identity");
    assert_eq!(ctx.messages[1].text(), "second");
    assert_eq!(ctx.messages[2].text(), "recalled: the user prefers helix");
    assert!(ctx.messages[2].role.is_system());
    assert_eq!(ctx.messages[3].text(), "hi");

    // A second model call of the same invocation does not prefetch again.
    rail.before_model_call(&mut ctx).await.unwrap();
    assert_eq!(state.lock().queries, vec!["hi".to_owned()]);
    assert_eq!(ctx.messages.len(), 4);
}

#[tokio::test]
async fn an_empty_prefetch_or_a_blank_query_inserts_nothing() {
    let state = recorded();
    let rail = ProviderMemoryRail::new(Fake::new(&state));
    let mut ctx = context(RailEvent::BeforeModelCall, vec![BaseMessage::user("hi")]);
    rail.before_model_call(&mut ctx).await.unwrap();
    assert_eq!(ctx.messages.len(), 1);

    state.lock().prefetch = "recalled".to_owned();
    let mut blank = context(RailEvent::BeforeModelCall, vec![BaseMessage::user("   ")]);
    rail.before_model_call(&mut blank).await.unwrap();
    assert_eq!(blank.messages.len(), 1);
    // The blank query never reached the provider.
    assert_eq!(state.lock().queries, vec!["hi".to_owned()]);
}

#[tokio::test]
async fn a_model_call_without_a_user_message_does_not_prefetch() {
    let state = recorded();
    let rail = ProviderMemoryRail::new(Fake::new(&state));
    let mut ctx = context(RailEvent::BeforeModelCall, vec![system("identity")]);
    rail.before_model_call(&mut ctx).await.unwrap();
    assert!(state.lock().queries.is_empty());
}

#[tokio::test]
async fn after_invoke_syncs_the_turn_when_the_output_is_an_answer() {
    let state = recorded();
    let rail = ProviderMemoryRail::new(Fake::new(&state));
    let mut ctx = context(RailEvent::AfterInvoke, Vec::new())
        .with_input(crate::AgentInput::from_text("what editor?"));
    ctx.output = Some(crate::output::AgentOutput::answer("helix"));
    rail.after_invoke(&mut ctx).await.unwrap();
    assert_eq!(
        state.lock().synced,
        vec![("what editor?".to_owned(), "helix".to_owned())]
    );
}

#[tokio::test]
async fn after_invoke_skips_errors_and_interrupts() {
    let state = recorded();
    let rail = ProviderMemoryRail::new(Fake::new(&state));
    let mut error_ctx = context(RailEvent::AfterInvoke, Vec::new())
        .with_input(crate::AgentInput::from_text("what editor?"));
    error_ctx.output = Some(crate::output::AgentOutput::error("boom"));
    rail.after_invoke(&mut error_ctx).await.unwrap();

    let mut interrupt_ctx = context(RailEvent::AfterInvoke, Vec::new())
        .with_input(crate::AgentInput::from_text("what editor?"));
    interrupt_ctx.output = Some(crate::output::AgentOutput::interrupt(
        Vec::new(),
        Vec::new(),
    ));
    rail.after_invoke(&mut interrupt_ctx).await.unwrap();

    assert!(state.lock().synced.is_empty());
}

#[tokio::test]
async fn a_failing_sync_turn_does_not_fail_the_rail() {
    let state = recorded();
    let provider: Arc<dyn MemoryProvider> = Arc::new(Fake {
        recorded: Arc::clone(&state),
        tools: Vec::new(),
        sync_fails: true,
        end_fails: false,
    });
    let rail = ProviderMemoryRail::new(provider);
    let mut ctx =
        context(RailEvent::AfterInvoke, Vec::new()).with_input(crate::AgentInput::from_text("hi"));
    ctx.output = Some(crate::output::AgentOutput::answer("hello"));
    rail.after_invoke(&mut ctx).await.unwrap();
    assert_eq!(state.lock().synced.len(), 1);
}

#[test]
fn provider_tools_derive_cards_from_the_function_schema() {
    let state = recorded();
    let provider = Fake::new(&state);
    let tools = provider_tools(Arc::clone(&provider));
    assert_eq!(tools.len(), 1);
    let card = tools[0].card();
    assert_eq!(card.name, "remember");
    assert_eq!(card.description, "stores a memory");
    assert_eq!(card.input_params["type"], json!("object"));
    assert_eq!(card.metadata.get("provider"), Some(&json!("fake")));

    let cards = provider_tool_cards(&provider);
    assert_eq!(cards.len(), 1);
    assert_eq!(cards[0].name, "remember");
}

#[test]
fn malformed_schemas_are_skipped() {
    let state = recorded();
    let provider: Arc<dyn MemoryProvider> = Arc::new(Fake {
        recorded: Arc::clone(&state),
        tools: vec![
            json!({"type": "function"}),
            json!({"type": "function", "function": {"name": "   "}}),
            json!("not a schema"),
            json!({"function": {"name": "kept"}}),
        ],
        sync_fails: false,
        end_fails: false,
    });
    let tools = provider_tools(Arc::clone(&provider));
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].card().name, "kept");
    // A missing description and parameters fall back to empty values.
    assert_eq!(tools[0].card().description, "");
    assert_eq!(tools[0].card().input_params, json!({}));
    assert_eq!(provider_tool_cards(&provider).len(), 1);
}

#[tokio::test]
async fn a_provider_tool_dispatches_and_returns_the_payload_verbatim() {
    let state = recorded();
    let provider = Fake::new(&state);
    let tool = ProviderTool::from_schema(Arc::clone(&provider), &provider.get_tool_schemas()[0])
        .expect("the fixture schema derives a tool");
    let ctx = ToolContext::default();
    let output = tool
        .invoke(json!({"text": "likes helix"}), &ctx)
        .await
        .unwrap();
    assert_eq!(output, json!({"result": "stored"}));
    assert_eq!(
        state.lock().calls,
        vec![("remember".to_owned(), json!({"text": "likes helix"}))]
    );
}

#[tokio::test]
async fn finalize_session_runs_once_for_an_owned_answer() {
    let state = recorded();
    let provider: Arc<dyn MemoryProvider> = Fake::new(&state);
    let turns = vec![
        ConversationTurn::new("user", "what editor?"),
        ConversationTurn::new("assistant", "helix"),
    ];
    finalize_session(
        Some(&provider),
        &turns,
        true,
        &crate::output::AgentOutput::answer("helix"),
    )
    .await;
    assert_eq!(
        state.lock().ends,
        vec![vec![
            ConversationTurn::new("user", "what editor?"),
            ConversationTurn::new("assistant", "helix"),
        ]]
    );
}

#[tokio::test]
async fn finalize_session_skips_borrowed_sessions_and_interrupts() {
    let state = recorded();
    let provider: Arc<dyn MemoryProvider> = Fake::new(&state);
    let turns = vec![ConversationTurn::new("user", "hi")];
    finalize_session(
        Some(&provider),
        &turns,
        false,
        &crate::output::AgentOutput::answer("hello"),
    )
    .await;
    finalize_session(
        Some(&provider),
        &turns,
        true,
        &crate::output::AgentOutput::interrupt(Vec::new(), Vec::new()),
    )
    .await;
    assert!(state.lock().ends.is_empty());
}

#[tokio::test]
async fn a_failing_session_end_does_not_propagate() {
    let state = recorded();
    let provider: Arc<dyn MemoryProvider> = Arc::new(Fake {
        recorded: Arc::clone(&state),
        tools: Vec::new(),
        sync_fails: false,
        end_fails: true,
    });
    let turns = vec![ConversationTurn::new("user", "hi")];
    finalize_session(
        Some(&provider),
        &turns,
        true,
        &crate::output::AgentOutput::answer("hello"),
    )
    .await;
    assert_eq!(state.lock().ends.len(), 1);
}

#[tokio::test]
async fn finalize_session_without_a_provider_is_a_no_op() {
    let turns = vec![ConversationTurn::new("user", "hi")];
    finalize_session(
        None,
        &turns,
        true,
        &crate::output::AgentOutput::answer("hello"),
    )
    .await;
}

#[test]
fn finalize_turns_drops_blank_text() {
    let turns = vec![
        ConversationTurn::new("user", "hi"),
        ConversationTurn::new("assistant", "  "),
        ConversationTurn::new("assistant", "there"),
    ];
    assert_eq!(
        finalize_turns(&turns),
        vec![
            ConversationTurn::new("user", "hi"),
            ConversationTurn::new("assistant", "there"),
        ]
    );
}
