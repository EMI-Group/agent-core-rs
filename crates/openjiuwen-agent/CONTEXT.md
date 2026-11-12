# openjiuwen-agent

## Intent

Agent runtime of the kernel: the `Agent` seam, `ReActAgent` (model → tools → observation loop), abilities (tool mounting/execution), rails (model/tool/loop hooks), interrupts/HITL, and the two application agents `LLMAgent` and `WorkflowAgent`.
Ports Python `openjiuwen.core.single_agent` (`base.py`, `agents/react_agent.py`, `ability_manager.py`, `rail/`, `interrupt/`, `schema/`) and `openjiuwen.core.application` (`llm_agent/`, `workflow_agent/`) onto `openjiuwen-foundation` (`Model`, `BaseMessage`, `Tool`), `openjiuwen-session` (`Session`, `AgentSession`, `StreamItem`, `InteractionRequest`), `openjiuwen-context` (`ContextEngine`), `openjiuwen-workflow` (`Workflow`), `openjiuwen-memory` and `openjiuwen-security`.
The crate is the only place that owns the ReAct control flow; it never resolves tools or models from a global registry (the `Runner` injects them via `AbilityManager`/`set_model`).
Python's legacy controller machinery (`ControllerAgent`, `LLMController`, `WorkflowController`, intent-detection controllers, `Runner.resource_mgr` lookup) is NOT ported: `LLMAgent` is a thin configured `ReActAgent` composition and `WorkflowAgent` owns its `Workflow` values directly.
See `docs/PARITY.md` for the port status of each area.

## API Surface

### `lib.rs` — crate root

Re-exports the public seams: `Agent`, `AgentCard`, `AgentOutput`, `AgentChunk`, `ResultType`, `Observation`, `OutputFrame`, `AgentInput`; `ReActAgent`, `ReActAgentConfig`, `MaxIterationsPolicy`; `LlmAgent`, `LlmAgentConfig`, `ModelConfig`, `ModelInfo`; `WorkflowAgent`, `WorkflowAgentConfig`, `ConstrainConfig`, `DefaultResponse`; `AbilityManager`, `AddAbilityResult`, `ToolCallResult`, `ToolExecutionOptions`; rails (`AgentRail`, `RailContext`, `RailEvent`, `RailRegistry`, `GuardrailRail`, `MemoryRail`, `ModelBackupRail`, `BaseInterruptRail`, `ConfirmInterruptRail`, `AskUserRail`, `ToolFilter`); interrupts (`InterruptKind`, `InterruptRequest`, `InterruptAnswer`, `InterruptionState`, `match_answers`).

### Agent seam (`agent.rs`, `output.rs`, `card.rs`, `input.rs`)

```rust
#[async_trait]
pub trait Agent: Send + Sync {
    fn card(&self) -> &AgentCard;
    async fn invoke(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> BoxStream<'static, Result<AgentChunk>>;
}

pub struct AgentCard { pub id: String, pub name: String, pub description: String,
                       pub input_params: Option<Value>, pub output_params: Option<Value>,
                       pub interface_url: Option<String> }   // serde snake_case, id defaults to uuid hex
impl AgentCard { pub fn new(name: impl Into<String>) -> Self; /* with_* builders */ }
impl openjiuwen_common::card::Card for AgentCard { .. }

pub enum ResultType { Answer, Error, Interrupt }   // wire "answer" | "error" | "interrupt"

pub struct Observation { pub content: String, pub name: Option<String>, pub tool_call_id: Option<String> }

pub struct OutputFrame { #[serde(rename = "type")] pub kind: String, pub index: i64, pub payload: Value }
impl OutputFrame { pub fn interaction(index: i64, id: &str, value: Value) -> Self; pub fn to_value(&self) -> Value; }

pub struct AgentOutput {
    pub output: String,                                   // answer text; workflow agents: JSON text of the result
    pub result_type: ResultType,
    pub finish_reason: Option<String>,
    pub reasoning_present: bool,
    pub termination_reason: Option<String>,               // "max_iterations" | "token_budget" | "cancelled"
    pub truncated_by_max_iterations: bool,
    pub iterations_used: Option<usize>,
    pub observations: Vec<Observation>,
    pub finalization_error: Option<String>,
    pub state: Vec<OutputFrame>,                          // interrupt frames (kind "__interaction__")
    pub interrupt_ids: Vec<String>,
    pub component_ids: Vec<String>,
    #[serde(rename = "workflow_execution_state")] pub workflow_output: Option<WorkflowOutput>,
    pub data: Option<Value>,                              // structured payload (workflow result object)
    pub extra: BTreeMap<String, Value>,
}
impl AgentOutput { pub fn answer(text) -> Self; pub fn error(text) -> Self; pub fn interrupt(frames, ids) -> Self;
                   pub fn is_interrupt(&self) -> bool; pub fn to_value(&self) -> Value; }

pub struct AgentChunk { pub kind: String, pub index: i64, pub payload: Value }
// kind: "llm_output" | "llm_reasoning" | "llm_usage" | "answer" | "__interaction__" | "workflow_final" | "context.usage"
impl AgentChunk { pub fn new/kinds()/answer(&AgentOutput)/interaction(id, value)/to_value()/to_stream_item()/from_stream_item() }

pub struct AgentInput {
    pub query: InputQuery,                 // Text(String) | Interactive(InteractiveInput)
    pub conversation_id: Option<String>,
    pub parent_session_id: Option<String>,
    pub user_id: Option<String>,
    pub run_kind: Option<String>,
    pub max_iterations: Option<usize>,     // Python "_host_max_iterations"
    pub extra: BTreeMap<String, Value>,    // unreserved keys, usable as prompt template variables
}
impl AgentInput { pub fn parse(inputs: &Value) -> Result<Self>; pub fn query_text(&self) -> Option<&str>; }
```
`AgentOutput` serializes to the Python result dict: `output`, `result_type`, `finish_reason`, `reasoning_present`, `termination_reason`, `iterations_used`, `truncated_by_max_iterations`, `observations[{content,name,tool_call_id}]`, `finalization_error`, `state`, `interrupt_ids`, `component_ids`, `workflow_execution_state`, `data`, `extra`; `None`/empty fields are skipped.
`AgentChunk::to_stream_item`/`from_stream_item` bridge to `openjiuwen_session::stream::StreamItem` (`mode = Output`).

### ReAct (`react/`)

```rust
pub enum MaxIterationsPolicy { Error, FinalAnswer, ReturnObservations }  // wire "error"|"final_answer"|"return_observations"

pub struct ReActAgentConfig {                       // Python ReActAgentConfig field names/defaults
    pub model_name: String,                         // ""
    pub model_provider: String,                     // "openai"
    pub api_key: Option<SecretString>,
    pub api_base: Option<String>,
    pub custom_headers: BTreeMap<String, String>,
    pub prompt_template_name: String,               // ""
    pub prompt_template: Vec<BaseMessage>,          // Python List[{"role","content"}]; system roles feed the identity section
    pub max_iterations: usize,                      // 5
    pub max_iterations_hard_limit: Option<usize>,
    pub on_max_iterations: MaxIterationsPolicy,     // "error"
    pub model_client_config: Option<ModelClientConfig>,
    pub model_config: Option<ModelRequestConfig>,  // Python `model_config_obj`
    pub context_engine_config: ContextConfig,
    pub mem_scope_id: String,                       // ""
    pub parallel_tool_calls: bool,                  // true
    pub retain_tool_facts_on_abort: bool,           // false
    pub max_total_tokens: Option<usize>,            // None; extension over Python (budget across a run)
}
impl ReActAgentConfig { pub fn new() -> Self; /* configure_model, configure_model_provider, configure_prompt,
    configure_prompt_template, configure_max_iterations, configure_model_client, configure_parallel_tool_calls,
    configure_context_engine, configure_mem_scope, configure_custom_headers (all builder-style) */ }

pub struct ReActAgent { /* card, RwLock<ReActAgentConfig>, RwLock<Option<Arc<dyn Model>>>, RwLock<SystemPromptBuilder>,
                            RwLock<AbilityManager>, RwLock<Arc<RailRegistry>>, RwLock<Arc<ContextEngine>> */ }
impl ReActAgent {
    pub fn new(card: AgentCard) -> Self;
    pub fn configure(&self, config: ReActAgentConfig);          // idempotent; rebuilds prompt builder + context engine
    pub fn config(&self) -> ReActAgentConfig;
    pub fn set_model(&self, model: Arc<dyn Model>);
    pub fn model(&self) -> Result<Arc<dyn Model>>;               // Error::Config when unconfigured
    pub fn add_tool(&self, tool: Arc<dyn Tool>) -> AddAbilityResult;
    pub fn add_tools(&self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Vec<AddAbilityResult>;
    pub fn remove_tool(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn tools(&self) -> Vec<ToolCard>;                        // model-visible order
    pub fn register_rail(&self, rail: Arc<dyn AgentRail>);
    pub fn unregister_rail(&self, name: &str);
    pub fn rails(&self) -> Arc<RailRegistry>;
    pub fn prompt_builder(&self) -> SystemPromptBuilder;
}
impl Agent for ReActAgent { .. }
```
Loop semantics (Python `_inner_invoke`): iteration `1..=limit`; per iteration build the system prompt (identity section = rendered system-role `prompt_template` joined by `\n\n`), prepare the window through `ContextEngine::prepare`, fire `BeforeModelCall`, call the model (streaming when a session/publisher exists; `invoke` otherwise), append the assistant message, execute tool calls (parallel batches by `parallel_safe`) with `BeforeToolCall`/`AfterToolCall`, append tool messages, fire `AfterReactIteration`.
No tool calls → `{"output": content, "result_type": "answer"}`; empty content → one retry after injecting a notice, then `result_type: "error"` with `finish_reason`/`reasoning_present`.
Exhaustion → `termination_reason: "max_iterations"`, `iterations_used`, `truncated_by_max_iterations: true` and the configured policy (`Error` → error output; `ReturnObservations` → observations attached; `FinalAnswer` → one extra tool-free model call).
Interrupts pause the loop into an `AgentOutput` with `result_type: "interrupt"`, `state` frames and `interrupt_ids`; state is persisted under `__react_agent_interruption__` and cleared on resume.
Cancellation (`CancellationToken` from the session config / tool context) yields `Error::Cancelled` and sanitizes the partial turn when `retain_tool_facts_on_abort = false`.

### Abilities (`abilities/`)

```rust
pub struct AddAbilityResult { pub name: String, pub added: bool, pub reason: String }  // "added_tool"|"refreshed_tool"|"duplicate_tool"

pub struct AbilityManager { /* ordered registry: name -> Arc<dyn Tool>, revision counter, owner_id */ }
impl AbilityManager {
    pub fn new(owner_id: impl Into<String>) -> Self;
    pub fn add_tool(&mut self, tool: Arc<dyn Tool>) -> AddAbilityResult;   // duplicate name+same card id -> refreshed
    pub fn remove_tool(&mut self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn cards(&self) -> Vec<ToolCard>;
    pub fn tool_info(&self) -> Vec<Value>;             // openai_tools(&cards)
    pub fn revision(&self) -> u64;  pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn reorder(&mut self, names: &[String]);
    pub async fn execute(&self, calls: &[ToolCall], options: &ToolExecutionOptions) -> Vec<ToolCallResult>;
}
pub struct ToolExecutionOptions { pub parallel: bool, pub session_id: Option<String>,
                                  pub timeout: Option<Duration>, pub cancellation: CancellationToken,
                                  pub metadata: BTreeMap<String, Value> }
pub struct ToolCallResult { pub call: ToolCall, pub output: Option<Value>, pub text: String,
                            pub success: bool, pub error: Option<String> }
impl ToolCallResult { pub fn message(&self) -> BaseMessage; }   // BaseMessage::tool(id, text).with_name(name)

pub fn parse_tool_arguments(raw: &str) -> Result<Value>;        // empty -> {}; repairs truncated JSON
```
Execution order follows the model's call order; consecutive `parallel_safe` tools run concurrently, a non-parallel-safe tool is a barrier. Unknown tools and failures become error `ToolCallResult`s (never `Err`), so the model always sees a result message. Timeouts map to `"Tool '<name>' timed out after {}s"`.

### Rails (`rails/`)

```rust
pub enum RailEvent { BeforeInvoke, AfterInvoke, BeforeModelCall, AfterModelCall, OnModelException,
                     BeforeToolCall, AfterToolCall, OnToolException, AfterReactIteration }
pub struct RailContext {
    pub event: RailEvent, pub session: Option<Arc<dyn Session>>, pub agent_id: String,
    pub iteration: usize, pub input: Option<AgentInput>,
    pub messages: Vec<BaseMessage>, pub tools: Vec<ToolCard>,
    pub response: Option<ModelReply>, pub tool_call: Option<ToolCall>, pub tool_result: Option<ToolCallResult>,
    pub output: Option<AgentOutput>,           // AfterInvoke
    pub extra: BTreeMap<String, Value>,        // per-invoke scratch (Python ctx.extra)
    pub force_finish: Option<AgentOutput>, pub model_continue: bool,
    pub retry: Option<RetryRequest>, pub skip_tool: Option<String>,
    pub interrupts: Vec<InterruptRequest>, pub model_override: Option<Arc<dyn Model>>,
}
impl RailContext { pub fn request_force_finish(&mut self, out: AgentOutput); pub fn take_force_finish(&mut self) -> Option<AgentOutput>;
                   pub fn request_model_continue(&mut self); pub fn take_model_continue(&mut self) -> bool;
                   pub fn request_retry(&mut self, delay: Option<Duration>); pub fn take_retry(&mut self) -> Option<RetryRequest>;
                   pub fn skip_tool_with(&mut self, message: impl Into<String>);
                   pub fn request_interrupt(&mut self, request: InterruptRequest); pub fn is_interrupted(&self) -> bool; }

#[async_trait]
pub trait AgentRail: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> i32 { 50 }
    async fn before_invoke(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
    async fn after_invoke(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
    async fn before_model_call(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
    async fn after_model_call(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
    async fn on_model_exception(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
    async fn before_tool_call(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
    async fn after_tool_call(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
    async fn on_tool_exception(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
    async fn after_react_iteration(&self, _ctx: &mut RailContext) -> Result<()> { Ok(()) }
}

pub struct RailRegistry { /* stable-descending by priority */ }
impl RailRegistry { pub fn new() -> Self; pub fn register(&mut self, rail: Arc<dyn AgentRail>);
                    pub fn unregister(&mut self, name: &str); pub fn rails(&self) -> &[Arc<dyn AgentRail>];
                    pub async fn fire(&self, ctx: &mut RailContext) -> Result<()>; }
```
`fire` dispatches only the hook for `ctx.event`, in descending priority (stable), and stops on the first `Err` (which aborts the loop).

Built-in rails (no upper-layer dependency): `GuardrailRail` (`GuardrailPipeline` input/output/tool scans; blocking → `Error::Security`), `MemoryRail` (`LongTermMemory` recall into a system message on `BeforeModelCall`, message write on `AfterInvoke`; failures are logged, never fatal), `ModelBackupRail` (on model exception: swap to the next `Arc<dyn Model>` via `ctx.model_override` and `request_retry`).
Interrupt rails: `BaseInterruptRail`, `ConfirmInterruptRail` (kind `confirm`, message `"Please approve or reject?"`, payload `{approved, feedback, auto_confirm, persist_allow}`), `AskUserRail` (kind `ask_user`, answer becomes the tool result).
Rails living above this crate (harness `ConfirmInterruptRail` extensions, tool trackers, OTel tracing, steering) are out of scope — the trait and event set are the seam.

### Interrupts (`interrupt.rs`)

```rust
pub enum InterruptKind { Undeclared, Permission, Confirm, AskUser, SkillAuthorization, SkillEvolutionApproval,
                         TeamToolApproval, ClientTool, Custom(String) }   // wire strings "", "permission", ...
pub struct InterruptRequest {
    pub interrupt_id: String,       // "intr_<uuid hex>"
    pub kind: InterruptKind, pub owner: String, pub message: String,
    pub payload_schema: Value, pub auto_confirm_key: String,
    pub ui_options: Vec<Value>, pub metadata: BTreeMap<String, Value>,
    pub tool_name: String, pub tool_call_id: String, pub tool_args: String, pub index: Option<u32>,
}
impl InterruptRequest { pub fn new(kind, owner, message) -> Self; pub fn for_tool_call(kind, owner, message, call: &ToolCall) -> Self;
                        pub fn to_frame(&self, index: i64) -> OutputFrame; pub fn to_interaction_request(&self) -> InteractionRequest; }
pub struct InterruptAnswer { pub owner: String, pub kind: InterruptKind, pub value: Value, pub request_id: String, pub carried: bool }
pub struct MatchedAnswers { pub answers: BTreeMap<String, Value>, pub unmatched: Vec<String> }
pub fn match_answers(interactive: &InteractiveInput, pending: &[(String, String)]) -> MatchedAnswers; // (tool_call_id, interrupt_id)
pub fn approval_value(approved: bool, feedback: &str, auto_confirm: bool) -> Value;
pub fn parse_approval(value: &Value) -> Option<bool>;      // bool | {approved} | "y|yes|ok|approve|allow|a|always" ...
pub fn reject_feedback(value: &Value) -> String;           // default "User feedback: rejected\n action"

pub struct InterruptionState { pub iteration: usize, pub original_query: String,
                               pub assistant_message: Option<BaseMessage>,
                               pub requests: BTreeMap<String, InterruptRequest>,  // tool_call_id -> request
                               pub answers: BTreeMap<String, Value> }             // interactive answers keyed by call id
pub const INTERRUPTION_STATE_KEY: &str = "__react_agent_interruption__";
pub const AUTO_CONFIRM_STATE_KEY: &str = "__interrupt_auto_confirm__";
```

### Application agents (`llm_agent.rs`, `workflow_agent/`)

```rust
pub struct ConstrainConfig { pub reserved_max_chat_rounds: usize, pub max_iteration: usize }  // 10, 5
pub struct ModelInfo { pub api_base: String, pub api_key: Option<SecretString>, pub model_name: String,
                       pub temperature: Option<f32>, pub top_p: Option<f32>, pub streaming: bool,
                       pub timeout: Option<Duration>, pub custom_headers: BTreeMap<String, String> }
pub struct ModelConfig { pub model_provider: String, pub model_info: ModelInfo }

pub struct LlmAgentConfig { pub id: String, pub name: String, pub description: String, pub version: String,
                            pub prompt_template: Vec<BaseMessage>, pub constrain: ConstrainConfig,
                            pub model: Option<ModelConfig>, pub memory_scope_id: String,
                            pub agent_memory_config: AgentMemoryConfig }
impl LlmAgentConfig { pub fn new(id: impl Into<String>) -> Self; /* with_model, with_prompt_template, with_constrain */ }
pub fn create_llm_agent_config(id: impl Into<String>, model: ModelConfig) -> LlmAgentConfig;
pub struct LlmAgent { /* card + Arc<ReActAgent> */ }
impl LlmAgent { pub fn new(config: LlmAgentConfig) -> Result<Self>; pub fn set_prompt_template(&self, template: Vec<BaseMessage>);
                pub fn react(&self) -> &Arc<ReActAgent>; }
impl Agent for LlmAgent { .. }    // output dict filtered to {output, result_type}; interrupts pass through

pub struct DefaultResponse { pub text: Option<String> }
pub struct WorkflowAgentConfig { pub id: String, pub name: String, pub description: String, pub version: String,
                                 pub constrain: ConstrainConfig, pub default_response: DefaultResponse,
                                 pub global_params: BTreeMap<String, Value> }
pub struct WorkflowAgent { /* card, config, RwLock<Vec<Arc<Workflow>>>, model for intent detection */ }
impl WorkflowAgent {
    pub fn new(config: WorkflowAgentConfig) -> Self;
    pub fn add_workflows(&self, workflows: impl IntoIterator<Item = Arc<Workflow>>);
    pub fn remove_workflow(&self, id: &str) -> bool;
    pub fn workflows(&self) -> Vec<Arc<Workflow>>;
    pub fn set_model(&self, model: Arc<dyn Model>);            // used for multi-workflow intent detection
}
impl Agent for WorkflowAgent { .. }
```
`LLMAgent` maps `constrain.max_iteration → ReActAgentConfig::max_iterations` and `reserved_max_chat_rounds` → context window rounds (Python `mock_llm_agent` mapping).
`WorkflowAgent` selection: one workflow → run it directly; several → LLM intent detection over `description or name` when a model is configured, otherwise the first workflow; a failed detection falls back to `default_response.text` when set.
Inputs are dispatched through the start component with `{"query": ...}` plus the caller's extra keys; the workflow session is a child of the agent session (`WorkflowSessionOptions::parent_session_id`).
Resume: `AgentInput::query` being `InteractiveInput` (or `agent_state` containing an interrupted workflow) calls `Workflow::resume`.
Outputs: completed → `result_type: "answer"`, `output` = JSON text of the workflow result, `data`/`workflow_output` carry the structured `WorkflowOutput`; interrupted → `result_type: "interrupt"` with `workflow_output.state == InputRequired` and the `__interaction__` frames; `default_response` → `result_type: "answer"` with `data["status"] = "default_response"`.

### Session, history, streaming (`run.rs`, `prompt.rs`)

`run::resolve_session(session, agent_id, agent_name, conversation_id, parent_session_id)` returns an `AgentRun { session, owned }`: a caller session is used as-is; otherwise an `AgentSession` is created with `session_id = conversation_id || "default_session"` and committed/closed at the end (`owned`).
History lives in `openjiuwen-context`: the loop calls `ContextEngine::create_context`, `prepare`, `observe`, and `save_contexts` (session state key `"context"`), matching Python's `context_engine` usage.
Streaming: every frame is `AgentChunk`; when a session is present the loop also writes `StreamItem`s to the session (`llm_output`, `llm_reasoning`, `llm_usage`, `answer`, `__interaction__`).
Traces: the loop opens an agent span on `session.tracer()` (`InvokeType::Chain`), model spans (`Llm`), and tool spans (`Plugin`), finishing with outputs/status.
`prompt::SystemPromptBuilder` holds `PromptSection { name, content, priority, category }`; `build()` joins sections by `"\n\n"` in ascending priority (identity = 10), matching Python's builder ordering.

## Constraints

- Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, `tracing` logging (no stdout).
- Only this crate's files are edited; other crates are read-only consumers.
- Files stay around ≤1000 lines; split by responsibility (loop vs model step vs config vs execution).
- Public data types are `Serialize + Deserialize + Default`-tolerant with `#[serde(rename_all = "snake_case", default)]` and unknown-field tolerance; Python wire names are preserved.
- Model/tool failures never abort the ReAct loop except when no recovery exists: tool errors become tool messages, model errors get one recovery attempt (rail-driven) before propagating.
- No global mutable state: `ReActAgent`/`WorkflowAgent` use interior mutability (`parking_lot::RwLock`) and are shared as `Arc`.
- Tests are offline and deterministic, driven by `openjiuwen-test-support::ScriptedModel`; no network.

## Design Decisions

- **Two lineages merged into one loop**: Python's `application` agents (legacy controller stack) are re-expressed on top of the single `ReActAgent` loop; `LLMAgent` is a composition, not a second loop implementation.
- **Interrupts are outputs, not errors**: a pause yields `AgentOutput { result_type: Interrupt, state, interrupt_ids }` plus persisted `InterruptionState`; resuming is a normal `invoke` whose `query` is an `InteractiveInput`.
- **Rails are async trait objects with priority ordering**, fired by name; `RailContext` carries per-event data plus control flags (`force_finish`, `retry`, `skip_tool`, `interrupts`, `model_override`) mirroring Python's `AgentCallbackContext`.
- **Tool execution is rail-free**: `AbilityManager::execute` only executes; the loop fires tool rails around each call so execution stays reusable/testable.
- **Prompt window comes from `ContextEngine::prepare`** (system messages + tools + window), not from an agent-local buffer.
- **`AgentOutput.output` is a string**; structured payloads travel in `data`/`workflow_output` (documented deviation from Python's nested dicts).

## Test Strategy

- Unit tests beside modules: card/`AgentInput`/`AgentOutput` serde round-trips; tool argument repair; registry ordering/revision; `match_answers`/`parse_approval`; prompt section ordering; `RailRegistry` priority ordering.
- Integration tests in `tests/` with `ScriptedModel`: ReAct no-tool answer, single tool, parallel tools, tool error → tool message, max-iteration policies, empty response retry, streaming frame assembly, ability mounting, rails ordering/force-finish/model backup, guardrail blocking, memory rail injection, confirm/ask-user interrupt + resume, `LLMAgent`, `WorkflowAgent` over a simple workflow, card serde snapshots.

## Routing Table

- `./src/react/` → `ReActAgent`, config, iteration loop, model step (`react/CONTEXT.md`)
- `./src/abilities/` → ability manager, argument repair, tool execution (`abilities/CONTEXT.md`)
- `./src/rails/` → rail trait/context/registry and built-in rails (`rails/CONTEXT.md`)
- `./src/workflow_agent/` → `WorkflowAgent` and workflow selection (`workflow_agent/CONTEXT.md`)
- `./src/interrupt.rs` → interrupt data model and resume matching
- `./src/output.rs` → `AgentOutput`, `AgentChunk`, `Observation`, `OutputFrame`
- `./src/input.rs` → `AgentInput` parsing
- `./src/run.rs` → session acquisition/commit helpers
- `./src/prompt.rs` → system prompt sections/builder
- `./src/llm_agent.rs` → `LLMAgent` composition over `ReActAgent`
- `./tests/` → crate integration suites (`tests/CONTEXT.md`)
