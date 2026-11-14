# openjiuwen-agent

## Intent

Agent runtime of the kernel: the `Agent` seam, `ReActAgent` (model → tools → observation loop), abilities (tool mounting/execution), rails (model/tool/loop hooks), interrupts/HITL, and the two application agents `LLMAgent` and `WorkflowAgent`.
Ports Python `openjiuwen.core.single_agent` (`base.py`, `agents/react_agent.py`, `ability_manager.py`, `rail/`, `interrupt/`, `schema/`) and `openjiuwen.core.application` (`llm_agent/`, `workflow_agent/`) onto `openjiuwen-foundation` (`Model`, `BaseMessage`, `Tool`), `openjiuwen-session` (`Session`, `AgentSession`, `StreamItem`, `InteractionRequest`), `openjiuwen-context` (`ContextEngine`), `openjiuwen-workflow` (`Workflow`), `openjiuwen-memory` and `openjiuwen-security`.
The crate is the only place that owns the ReAct control flow; it never resolves tools or models from a global registry (the `Runner` injects them via `AbilityManager`/`set_model`).
Python's legacy controller machinery is now ported as the `controller` module: `ControllerAgent`/`LLMController`/`WorkflowController` and the `IntentDetectionController` lineage route events to registered agents/workflows, while `LLMAgent` stays a thin configured `ReActAgent` composition.
It also hosts the seams Python keeps beside the loop: tool exposure/MCP allowlists, client tools, per-tool resilience, the final-request check (`policy`), the lifecycle callback framework (`callbacks`), the steering queue (`steering`), and the Markdown skill library (`skills`).
`controller` (event → intent → task store routing with background scheduling, pause/cancel, per-conversation preemption, group routing and the task-op reasoner), `kv_cache` (cache identity, in-process prefix runtime, model-call affinity hook and the provider prefetch/offload action runtime), `operator` (tunable LLM/tool/memory/skill/context-evolve handles, the legacy `LLMCall` wrapper and experience staging), `evolve` (`ReActAgentEvolve`), and `prompt_pack` (YAML packs plus Python prompt-directory loading) are implemented modules with their own `CONTEXT.md`; `openjiuwen-core` re-exports them under the Python package paths.
See `docs/PARITY.md` for the port status of each area.

## API Surface

### `lib.rs` — crate root

Re-exports the public seams: `Agent`, `AgentCard`, `AgentOutput`, `AgentChunk`, `ResultType`, `Observation`, `OutputFrame`, `AgentInput`; `ReActAgent`, `ReActAgentConfig`, `MaxIterationsPolicy`; `LlmAgent`, `LlmAgentConfig`, `ModelConfig`, `ModelInfo`; `WorkflowAgent`, `WorkflowAgentConfig`, `ConstrainConfig`, `DefaultResponse`; `AbilityManager`, `AddAbilityResult`, `ToolCallResult`, `ToolExecutionOptions`; rails (`AgentRail`, `RailContext`, `RailEvent`, `RailRegistry`, `GuardrailRail`, `MemoryRail`, `ModelBackupRail`, `BaseInterruptRail`, `ConfirmInterruptRail`, `AskUserRail`, `ToolFilter`); interrupts (`InterruptKind`, `InterruptRequest`, `InterruptAnswer`, `InterruptionState`, `match_answers`).
The extension trees are `pub mod` only (not crate-root re-exports): `callbacks`, `steering`, `policy`, `skills`, `controller`, `kv_cache`, `operator`, `evolve`, `prompt_pack`.

### Agent seam (`agent.rs`, `output.rs`, `card.rs`, `input.rs`)

```rust
#[async_trait]
pub trait Agent: Send + Sync {
    fn card(&self) -> &AgentCard;
    async fn invoke(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> BoxStream<'static, Result<AgentChunk>>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]   // no Default derive; no schemars transform
#[serde(rename_all = "snake_case", default)]                            // unknown fields tolerated
pub struct AgentCard { pub id: String, pub name: String, pub description: String,
                       pub input_params: Option<Value>, pub output_params: Option<Value>,
                       pub interface_url: Option<String> }   // `description` is a String; optionals serialize as null (no skip_serializing_if)
impl Default for AgentCard { .. }   // manual: fresh 32-lowercase-hex uuid id, empty strings, None optionals
impl AgentCard { pub fn new(name: impl Into<String>) -> Self;
                 // consuming #[must_use] builders: with_id/with_name/with_description/with_input_params/with_output_params/with_interface_url
                 pub fn id(&self) -> &str; pub fn name(&self) -> &str; pub fn description(&self) -> &str; }
impl openjiuwen_common::card::Card for AgentCard { .. }   // Card::description() -> None when the String is empty
// No From/TryFrom conversions between AgentCard and CardBase or any other card type.

pub enum ResultType { Answer, Error, Interrupt }   // wire "answer" | "error" | "interrupt"

pub struct Observation { pub content: String, pub name: Option<String>, pub tool_call_id: Option<String>,
                         #[serde(default, skip_serializing_if = "Vec::is_empty")] pub parts: Vec<ContentPart> }  // openjiuwen_foundation::ContentPart; empty parts stay omitted (Python-shaped)

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
                   pub fn is_interrupt(&self) -> bool; pub fn to_value(&self) -> Value; pub fn from_value(value: Value) -> Result<Self>;
                   /* consuming builders: with_finish_reason, with_reasoning_present, with_iterations_used,
                      with_termination_reason, with_data, with_extra, with_observations */ }
impl OutputFrame { pub fn from_value(value: Value) -> Result<Self>; }   // Error::validation on non-object
impl AgentChunk { pub fn from_value(value: Value) -> Result<Self>; }

pub struct AgentChunk { pub kind: String, pub index: i64, pub payload: Value }
// kind: "llm_output" | "llm_reasoning" | "llm_usage" | "answer" | "__interaction__" | "workflow_final" | "context.usage" | "llm_route"
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
impl AgentInput { pub fn parse(inputs: &Value) -> Result<Self>; pub fn query_text(&self) -> Option<&str>;
                  pub fn interactive(&self) -> Option<&InteractiveInput>; pub fn from_text(text: impl Into<String>) -> Self; }
```
`AgentOutput` serializes to the Python result dict: `output`, `result_type`, `finish_reason`, `reasoning_present`, `termination_reason`, `iterations_used`, `truncated_by_max_iterations`, `observations[{content,name,tool_call_id}]`, `finalization_error`, `state`, `interrupt_ids`, `component_ids`, `workflow_execution_state`, `data`, `extra`; `None`/empty fields are skipped, and `reasoning_present`/`truncated_by_max_iterations` are omitted unless `true`.
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
    pub fn context_engine(&self) -> Arc<ContextEngine>;
    // side-seam setters (all default to the pre-wiring behavior)
    pub fn set_callback_framework(&self, Arc<CallbackFramework>);  pub fn callbacks(&self) -> Arc<CallbackFramework>;
    pub fn register_callback(&self, AgentCallbackEvent, Arc<dyn AgentCallback>);
    pub fn steering(&self) -> SteeringQueue;  pub fn steer(&self, text: impl Into<String>) -> usize;
    pub fn set_tool_exposure_policy(&self, ToolExposurePolicy);  pub fn tool_exposure_policy(&self) -> ToolExposurePolicy;
    pub fn set_mcp_tool_allowlist(&self, McpToolAllowlist);      pub fn mcp_tool_allowlist(&self) -> McpToolAllowlist;
    pub fn set_client_tools(&self, Arc<ClientToolRegistry>);     pub fn client_tools(&self) -> Arc<ClientToolRegistry>;
    pub fn set_tool_resilience(&self, ToolResiliencePolicy);     pub fn tool_resilience(&self) -> ToolResiliencePolicy;
    pub fn set_final_request_check(&self, Arc<dyn FinalRequestCheck>);
    pub fn final_request_check(&self) -> Option<Arc<dyn FinalRequestCheck>>;
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
                            pub success: bool, pub error: Option<String>, pub parts: Vec<ContentPart> }
impl ToolCallResult { pub fn success(call, output, text) -> Self; pub fn failure(call, error) -> Self;
                      pub fn with_parts(self, parts: Vec<ContentPart>) -> Self;
                      pub fn message(&self) -> BaseMessage; }   // MessageContent::Parts when parts exist, else text; tool_call_id + name preserved

pub mod parts;   // extract_parts(&Value) -> Vec<ContentPart>, part_summary(&ContentPart) -> String,
                 // render_with_parts(text, &[ContentPart]) -> String
pub struct SubAgentAbility;   // new(Arc<dyn Agent>), with_name/with_description/with_session, agent(), session()
pub struct WorkflowAbility;   // new(Arc<Workflow>), with_name/with_description, workflow()
pub struct McpToolAbility;    // new(card, Arc<dyn McpClient>, tool_name), with_options, remote_name(), server_name(), mcp_tool()
pub struct McpAbilityOptions; // new/with_namespace/with_allowlist/with_prefix/without_prefix
pub async fn load_mcp_abilities(client: Arc<dyn McpClient>, options: McpAbilityOptions) -> Result<Vec<Arc<dyn Tool>>>;
// MCP names are namespaced "<namespace>__<tool>" (default namespace = server name); allowlist key = server id.
```

pub fn parse_tool_arguments(raw: &str) -> Result<Value>;        // empty -> {}; repairs truncated JSON
```
Execution order follows the model's call order; consecutive `parallel_safe` tools run concurrently, a non-parallel-safe tool is a barrier. Unknown tools and failures become error `ToolCallResult`s (never `Err`), so the model always sees a result message. Timeouts map to `"Tool '<name>' timed out after {}s"`.
Per-tool resilience (`policy::ToolResiliencePolicy`) wraps each call: the per-tool timeout overrides `ToolExecutionOptions::timeout`, and `max_attempts` retries tool/timeout failures with `backoff_for(attempt)`; `Cancelled` and `Validation` are never retried.
Multimodal tool outputs: `abilities::parts::extract_parts` scans the raw output (`content_parts`, a `content` array of typed parts, a bare array, or a foundation `MessageContent`), the parts ride on `ToolCallResult::parts` and `Observation::parts`, and `ToolCallResult::message()` emits `MessageContent::Parts` (text first) only when parts exist — text-only results are unchanged.

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

### Extensions (`callbacks/`, `steering/`, `policy/`, `skills/`, `controller/`, `kv_cache/`, `operator/`, `evolve/`, `prompt_pack/`)

These modules are `pub mod` in `lib.rs` but not re-exported from the crate root; reach them by module path. Each has its own `CONTEXT.md` with the full API.

- `policy` — `ToolExposurePolicy`/`ToolExposureMode`/`McpToolAllowlist` (which tools reach the model), `ClientToolRegistry`/`ClientToolHandler`/`FnClientToolHandler` (tools the connected client executes), `ToolResilience`/`ToolResiliencePolicy` (per-tool timeout/attempts/backoff), `FinalRequestCheck`/`DefaultFinalRequestCheck`/`FinalRequestContext`/`FinalRequestDecision`.
- `callbacks` — `AgentCallbackEvent` (six events), typed payloads, the `AgentCallback` trait, and `CallbackFramework` (priority-ordered registry); a failing handler is logged, never fatal.
- `steering` — `SteeringMessage` and the cloneable `SteeringQueue` (`push`/`push_text`/`drain`/`drain_messages`/`drain_prompt`); the loop drains the backlog at the top of every iteration as one batch of `user` messages.
- `skills` — `Skill` (+ YAML front matter), `SkillManager` (`tokio::fs` discovery), `SkillUtil`, and `SkillRail` (renders the catalog as the first system message on `BeforeModelCall`).
- `controller` — `Controller`/`BaseController` (turn loop: event queue → intent recognizer → planner → executor) over a shared `TaskManager` (indexes, filters, hierarchy, `claim_task`, per-session persistence under `__controller_tasks__`); `TaskSchedulerLoop` (spawned scan loop with `Notify` wakeups, `max_concurrent_tasks`, `task_timeout_ms`, dependency gating from `metadata["depends_on"]`, `all_tasks_processed` sentinel); `TaskExecutor` pause/cancel hooks; `TaskQueue` (one running task per conversation, new message preempts); `GroupRouter` (`send_to_agent`/`publish`); the `reasoner` task-op layer (create/pause/resume/cancel/continue/supplement/modify behind the confidence gate); schemas, `ControllerAgent`, `WorkflowController`, `LLMController`, `IntentRecognizer`/`LlmIntentRecognizer`/`Planner`, and the `legacy` `IntentDetectionController`/`IntentDetector`.
- `kv_cache` — `KVCacheIdentity`/`KVCacheEntry`/`KVCacheLookup`/`KVCacheStats`, `KVCacheRuntimeProtocol`, `InProcessKVCacheRuntime` (LRU + session eviction + FNV-1a prefix hashing), `KVCacheAffinityConfig`/`KVCacheRuntimeConfig`, metadata helpers (`resolve_session_lineage`, `first_changed_index`, range kwargs, compressor/team identities), `KVCacheModelCallHook`, and the provider-action seam `KvCacheActionProvider`/`InProcessKvCacheActionProvider`/`KvCacheActionRuntime` (prefetch/offload/evict with admission, residency, fail-open and timeout budgets); `KVCacheBackend::{Redis, Distributed}` return `Error::Unsupported`.
- `operator` — `Operator`/`PreviewableOperator` traits with `TunableSpec`/`UpdateValue`/`ApplyResult` (full Python metadata: mode/effect/records/change type/lifecycle stage/pending id/errors/metadata) and the protocol normalizers; `LLMCallOperator` (prompt tunables + `render_messages`/`call`), `ToolCallOperator` (`apply_to_cards`), `MemoryCallOperator` (recall seam with `max_retries`), `SkillCallOperator` (selection), `SkillExperienceOperator` + `PendingChangeStore` staging, the distinct executing `legacy::LLMCall` wrapper (optimizer callback, invoke/stream), `ContextEvolveOperator`/`MetisContextEvolveOperator` (deterministic evolve policies; Metis `learn` is `Unsupported`).
- `evolve` — `ReActAgentEvolve` delegating execution to an inner `ReActAgent`, exposing `get_operators()` (`react_llm`, `react_tool`) whose parameter callbacks sync prompt templates back into the live config and tool descriptions through `DescriptionOverrideTool` wrappers.
- `prompt_pack` — `PromptPack` YAML schema (prompts/template/sections, tool policy, MCP allowlist, resilience, rails), `from_yaml`/`from_yaml_file`/`to_yaml`, `apply_to`/`apply_to_with` returning `AppliedPack` (built vs deferred rails), `PromptSectionsRail`, and `PromptDirectory`/`PromptPack::from_prompt_dir` for the Python prompt-file layouts (`cn/en/*.md`, `<name>.<lang>.md`, YAML front matter, language fallback, evolved hash).
- `rails::MemoryRail` — recall before the model call plus `BeforeInvoke` memory prompt variables: `ctx.extra["memory_variables"]` carries `sys_memory_variables` (filtered user variables as a JSON object) and `sys_long_term_memory` (labeled fragment/summary hits as a JSON array), rendered into `prompt_template` system messages once per invoke.
- `ReActAgent` seam setters: `set_callback_framework`, `callbacks`, `register_callback`, `steering`, `steer`, `set_tool_exposure_policy`, `set_mcp_tool_allowlist`, `set_client_tools`, `client_tools`, `set_tool_resilience`, `set_final_request_check` (details in `react/CONTEXT.md`).

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

## Notes for Agents

- The loop pauses a turn once: it fires `BeforeToolCall` for every tool call of the model turn in model order, drains each call's requests, and — when any request was recorded — executes no tool of that turn and returns all requests together in one `result_type: "interrupt"` output.
- A resumed turn replays the interrupted calls without a model call; requests the caller left unanswered re-pause with only the remaining requests, whose `interrupt_id`s are replayed rather than re-minted.
- Auto-confirm is seeded from the session's `__interrupt_auto_confirm__` into `ctx.extra["_interrupt_auto_confirm"]` on every invoke — fresh and resumed alike — matching Python's per-turn session read.
- The resume payload is the kernel `InteractiveInput` in the `query` slot: `{"query": {"user_inputs": {<tool_call_id>: answer}, "interrupt_answers": {<interrupt_id>: answer}}}`; `interrupt_answers` wins over `user_inputs`, and a top-level `resume`/`answers`/`interaction_id` key is not read (it lands in `extra` as a template variable).
- Resume reuses the caller's `Arc<dyn Session>` (state is in-process); an interactive query with no persisted `InterruptionState` is `Error::Validation`, while a text query with pending state resumes with zero answers and re-pauses with the same `interrupt_id`s.
- A pause stores each request under its tool-call id (the interrupt id when the call has none) and emits `{"id": interrupt_id, "value": <serialized InterruptRequest>}` frames; `InterruptRequest::to_interaction_request` exists for hosts but the loop never calls `Session::request_interaction` — it mirrors frames with `Session::write_stream`.
- `InterruptRequest` serializes with Python wire names `interrupt_kind`/`interrupt_owner` (deserialization also accepts the Rust `kind`/`owner`), so an `__interaction__` frame payload is `{id, value}` with a Python-shaped `value`.
- `AgentOutput::truncated_by_max_iterations` is omitted from the wire object unless it is `true`.
- `AgentOutput::termination_reason` is a free-form `Option<String>`, not an enum: `"max_iterations"` / `"token_budget"` / `"cancelled"` from the ReAct loop and `"workflow_error"` from `WorkflowAgent`.
- The `AgentChunk` frame-kind constants (`LLM_OUTPUT`, `LLM_REASONING`, `LLM_USAGE`, `ANSWER`, `INTERACTION`, `WORKFLOW_FINAL`, `CONTEXT_USAGE`, `LLM_ROUTE`) live in the `output` module but are not re-exported from the crate root; refer to them as `openjiuwen_agent::output::*`.
- `context.usage` frames are emitted after every successful `ContextEngine::prepare` (`iteration`, `message_count`, `tool_count`, `token_count`, `overflow`, `window_size`); `llm_route` frames are emitted only when a rail's `model_override` makes the run diverge from the primary model (`route`, `model`, `reason`).
- `abilities::AbilityManager::execute` batches runs of consecutive `parallel_safe` tools (capped by `MAX_PARALLEL_BATCH`), treats any other call as a barrier, and applies `ToolExecutionOptions::timeout` per call.
- `parse_tool_arguments` strips a Markdown fence and repairs trailing commas and truncation before reporting an error.
- `ReActAgent::tools()` returns owned `ToolCard` snapshots in model-visible order and the agent exposes no accessor to its `AbilityManager`, so callers needing `Arc<dyn Tool>` handles (or a name/description other than through `tool.card().name`/`.description`; the `Tool` trait has only `card()`) must keep their own references to the mounted tools.
- There is no `set_context_engine`/`with_context_engine`: `ReActAgent.engine` is private, built by `new`/`configure` from `ReActAgentConfig::context_engine_config`, and exposed only as an `Arc<ContextEngine>` clone through `context_engine()`; calling `configure` again replaces the engine and drops its contexts.
- `ContextConfig.processors` (a `Vec<ProcessorConfig>`) is the only way to install context processors on the agent's engine — the engine builds its pipeline per context and offers no `add_processor`.
- The loop only calls `ContextEngine::create_context`, `prepare`, `observe`, and `save_contexts` (the last writes session state key `context`); it never calls `compress`, `recover_from_model_exception`, `load_contexts`, or `observe_tool_result`, so compression happens only inside `prepare` and overflow recovery is not agent-driven.
- A `ContextEngine` works standalone with no agent (`create_context("main", None)`, `observe`, `prepare`/`compress`), so compressing a long transcript does not require a `ReActAgent`/`LlmAgent`; see `crates/openjiuwen-examples/examples/05_context_and_memory.rs` and `crates/openjiuwen-context/tests/engine.rs`.
- `LlmAgent` reaches the engine only through `LlmAgent::react()`; its `constrain.reserved_max_chat_rounds` is copied into `context_engine_config.default_window_rounds` and it exposes no other context setting.
- `LlmAgent` and `WorkflowAgent` define no tool methods of their own: mount/read on an `LlmAgent` through `LlmAgent::react()`, and `WorkflowAgent` has no tool surface at all.
- `WorkflowAgent` implements `invoke`/`stream`; `workflow_agent::select::select` asks the model to pick among several workflows from `description or name` and falls back to `default_response` and then the first workflow.
- `LlmAgentConfig::agent_memory_config` is host-visible only: `ReActAgentConfig` has no memory-toggle field, so it does not change the composed `ReAct` configuration.
- `rails::StageFlags` is a type alias of `openjiuwen_security::ScanTargets`.
- `react::model_step::ModelReply` is public because `RailContext::response` exposes it; the loop calls the model through `call_model_streaming` (the session-scoped path that emits `llm_reasoning`/`llm_output`/`llm_usage` frames) and uses `call_model` only for the tool-free `FinalAnswer` finalization call.
- `controller`, `evolve`, `kv_cache`, `operator`, and `prompt_pack` are implemented `pub mod` trees with their own `CONTEXT.md`; the crate root does not re-export their types, so reach them by module path.
- The controller has two execution modes: `BaseController::invoke`/`stream` stay single-turn and in-process, while `start()` launches the background `TaskSchedulerLoop` that only runs tasks the shared `TaskManager` holds (or that a reasoner/host added). `stop()` aborts the loop and every in-flight task; `shutdown()` also drains.
- Dependency scheduling is a documented extension: a task with `metadata["depends_on"] = [ids]` and status `Waiting` is promoted to `Submitted` once every dependency is `Completed`; Python declares the `waiting` status but never gates on it.
- The background scheduler admits in deterministic priority-descending order (`created_seq`, then `task_id` tiebreaks), treats `max_concurrent_tasks = 0` as unlimited (documented divergence from Python's bug), and emits `all_tasks_processed` once per idle stretch unless `suppress_completion_signal` is set.
- `TaskExecutor` pause/cancel hooks are defaulted, so every executor that does not support them keeps compiling; a `Submitted` cancel short-circuits to `Canceled`, terminal states are idempotent, and a resume flips `Paused` → `Submitted`.
- `TaskQueue` (legacy `IntentDetectionController`) allows one running task per conversation: a new message aborts the previous handler fire-and-forget, cancels the tracked run, then enqueues. `GroupRouter` is optional; without one `send_to_agent`/`publish` return `Error::Config` naming the method (and agent).
- `reasoner` is the task-operation layer behind Python's intent tools: `plan_ops` maps create/pause/resume/cancel/continue/supplement/modify intents, degrades to `Unknown` below `intent_confidence_threshold`, and `apply_task_ops` never aborts a batch.
- The KV-cache action runtime is a separate seam from prefix bookkeeping: `prepare` returns once the prefetch provider call starts (not completes), `suspend`/`release` schedule offload/evict, offload/evict wait for the active-inference counter to reach zero, every provider error/timeout fails open, and the in-process backend exists for deterministic hosts/tests.
- `KvCacheActionProvider` is the network boundary: real provider HTTP (Python's `model.prefetch_kvc`/`offload_kvc`/`evict_kvc`) is not reimplemented here; attach any provider implementation to `KVCacheActionRuntime`.
- `operator::legacy::LLMCall` is deliberately a distinct executing wrapper (live model, async optimizer callback, invoke/stream); it is not an alias of `LLMCallOperator`. Experience staging (`PendingChangeStore`) is process-local and never serialized, matching Python's in-memory `ExperienceManager`.
- `prompt_pack` accepts two additive forms: the Rust YAML document and Python prompt directories (`PromptDirectory`), both mapping onto `SectionSpec`/`PromptSectionsRail`; front matter is parsed but never re-rendered, and evolved-ness is computed from the body hash.
- `MemoryRail` publishes its memory prompt variables on `BeforeInvoke` (not on every model call); the loop renders them into the identity section once per invoke, so every model call of that invoke sees the same memory text. Memory values override same-named input fields and unknown placeholders stay literal.
- Test models are `openjiuwen_test_support::ScriptedModel` (aliased nowhere); the construction and inspection idioms are catalogued in `tests/CONTEXT.md`.

## Deferred Python Behavior

- Not ported: the `before_steering_drain` rail hook (`RailEvent` has no variant for it and `openjiuwen-harness` matches that enum exhaustively; the loop drains the `SteeringQueue` at its own safe point, rail-free).
- Not ported: the process-global `Runner.callback_framework`; `CallbackFramework` is a per-agent value owned by the host.
- Not ported: KV-cache Redis/distributed prefix backends (`KVCacheBackend::runtime` returns `Error::Unsupported`) and vendor HTTP transports for prefetch/offload/evict — the action runtime is a trait seam with a deterministic in-process backend.
- Not ported: controller `event_timeout_ms`, `history_limit`, `expose_reasoning`, and `stream_first_frame_timeout_ms` enforcement (no stream host/context engine), and Python's `ReasonerConfig`/`ProactiveIdentifier`/`Reflector` configuration surface; `max_concurrent_tasks = 0` means unlimited and scheduler admission is deterministic (documented divergences).
- Not ported: the ML-backed Metis learner (`MetisContextEvolveOperator::learn` is `Unsupported`), `approved_record_ids` subset approval, pending-change subject/trajectory/messages, and Python's `execute_updates` operator-map loop (callers drive operators directly).
- Not ported: `MemoryRail`'s `ctx.extra["is_resume"]` skip guard and `_original_query` recording (the Rust loop sets neither), and the legacy `LLMController`'s per-iteration memory re-query (the ReAct path renders once per invoke).
- Not ported: Python package manifests (`manifest.json`/`harness_config.yaml`) and `SKILL.md` skill packages as prompt sources — `prompt_pack` loads the markdown prompt-directory layouts and the YAML pack only.

## Test Strategy

- Unit tests beside every module (898) cover the seams in isolation: card/`AgentInput`/`AgentOutput` serde, `match_answers`/`parse_approval`, prompt section ordering, rail registry dispatch, guardrail stages, memory recall injection and memory prompt variables, model-backup ordering, interrupt rail flows, tool-argument repair, parallel tool batching, callback dispatch ordering, steering FIFO/thread sharing, exposure/allowlist projection, client-tool registry, resilience backoff, final-request decisions, skill front matter/discovery/rail, multimodal part extraction, controller routing/intents/tasks/legacy controllers, task-manager filters/indexes/persistence, background scheduling/timeouts/cancellation/sentinel, executor pause/cancel, `TaskQueue` preemption, group routing, reasoner task ops, KV-cache identity/LRU/eviction/metadata plus the action runtime (non-blocking prefetch, idle-gated offload, fail-open, revision guard), operator tunables/state/propagation plus the legacy `LLMCall`, memory `max_retries`, experience preview/staging, `ReActAgentEvolve` parameter sync, and prompt-pack YAML + directory/front-matter loading.
- `src/react/wiring/` holds the loop-integration suites for the six side seams (callbacks, steering, exposure/MCP/resilience, client tools, final-request check); they live under `src/` to observe crate-internal state (`react/wiring/CONTEXT.md`).
- `tests/agent_card.rs` (16) pins the wire contract: `AgentCard` field set/`Card` trait/JSON round-trip, `AgentOutput` result-type strings and `state` frames, the absent-when-empty key shapes, Python-shaped result dicts, and the `AgentChunk` ↔ `StreamItem` bridge.
- `tests/react_agent.rs` (12) drives the loop end to end: no-tool answer, single/parallel/failing tools, the three max-iteration policies, empty-response retry, prompt rendering, and input/config errors.
- `tests/streaming.rs` (4) assembles `llm_output`/`llm_reasoning`/`llm_usage`/`answer` frames and checks the session mirror.
- `tests/rails.rs` (7) proves rails steer a running agent: priority ordering per event, force-finish before the model and before a tool, `ModelBackupRail` failover, `GuardrailRail` blocking (`ErrorKind::Security`), `MemoryRail` recall + write-back, and `skip_tool`.
- `tests/interrupt.rs` (8) covers the HITL contract: confirm pause shape and persistence, approve resume, reject with and without feedback, session-seeded auto-confirm, ask-user answer as the tool result, two pending calls pausing once and resuming individually, and the streaming pause path emitting and mirroring its `__interaction__` frame.
- `tests/llm_agent.rs` (7) covers config mapping onto `ReActAgent`, provider fallback, invoke/stream output filtering, prompt-template override, and interrupt pass-through.
- `tests/workflow_agent.rs` (5) covers single-workflow invoke/stream, model-driven selection with default-response fallback, workflow interrupt/resume, and the no-workflow error.
- The crate suite (898 unit + 59 integration + 5 doc tests) is offline and deterministic on `ScriptedModel`; no network and no wall-clock assertions beyond short timeouts.

## Routing Table

- `./src/react/` → `ReActAgent`, config, iteration loop, model step, exposure, frames, hooks, pause/resume (`react/CONTEXT.md`)
- `./src/react/agent_loop/` → the invoke loop (`mod.rs`) and its unit tests (`agent_loop/CONTEXT.md`)
- `./src/react/wiring/` → `#[cfg(test)]` loop-integration suites for the six side seams (`react/wiring/CONTEXT.md`)
- `./src/abilities/` → ability manager, argument repair, tool execution (`abilities/CONTEXT.md`)
- `./src/rails/` → rail trait/context/registry and built-in rails (`rails/CONTEXT.md`)
- `./src/policy/` → tool exposure, MCP allowlist, client tools, per-tool resilience, final-request check (`policy/CONTEXT.md`)
- `./src/callbacks/` → lifecycle callback events, payloads, framework (`callbacks/CONTEXT.md`)
- `./src/steering/` → steering queue drained by the loop (`steering/CONTEXT.md`)
- `./src/skills/` → Markdown skill library, manager, `SkillRail` (`skills/CONTEXT.md`)
- `./src/controller/` → controller stack: turn loop, executors, intent recognition, legacy `IntentDetectionController` (`controller/CONTEXT.md`)
- `./src/controller/task_manager/` → shared task store: indexes, filters, hierarchy, claim/persistence (`task_manager/CONTEXT.md`)
- `./src/controller/reasoner/` → intent → task-op planning and application (`reasoner/CONTEXT.md`)
- `./src/controller/executors/` → executor trait/registry, built-ins, and the background scheduler (`executors/CONTEXT.md`)
- `./src/kv_cache/` → cache identity/prefix runtime/metadata, the model-call hook, and the prefetch/offload action runtime (`kv_cache/CONTEXT.md`)
- `./src/operator/` → tunable operators, rich update/apply results, experience staging (`operator/CONTEXT.md`)
- `./src/operator/legacy/` → the executing `LLMCall` wrapper distinct from `LLMCallOperator` (`operator/legacy/CONTEXT.md`)
- `./src/evolve/` → `ReActAgentEvolve` and operator parameter sync (`evolve/CONTEXT.md`)
- `./src/prompt_pack/` → YAML prompt packs, prompt directories/front matter, rails, section rail (`prompt_pack/CONTEXT.md`)
- `./src/workflow_agent/` → `WorkflowAgent` and workflow selection (`workflow_agent/CONTEXT.md`)
- `./src/interrupt.rs` → interrupt data model and resume matching
- `./src/output.rs` → `AgentOutput`, `AgentChunk`, `Observation`, `OutputFrame`
- `./src/input.rs` → `AgentInput` parsing
- `./src/run.rs` → session acquisition/commit helpers
- `./src/prompt.rs` → system prompt sections/builder
- `./src/llm_agent.rs` → `LLMAgent` composition over `ReActAgent`
- `./tests/` → crate integration suites (`tests/CONTEXT.md`)
