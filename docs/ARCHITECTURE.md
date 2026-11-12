# Architecture

This document defines the cross-cutting contracts of the Rust rewrite.
Per-crate details live in each crate's `CONTEXT.md`.

## 1. Crate layering

Dependencies point downward only; a crate may depend on crates in lower layers, never upward.
`openjiuwen-core` is the kernel umbrella and may depend on every kernel crate.
`openjiuwen` is the top-level umbrella and may depend on every library crate (never on the CLI binary).

| Layer | Crates | Notes |
|---|---|---|
| 0 | `openjiuwen-common`, `openjiuwen-macros`, `openjiuwen-harness-protocol` | No workspace dependencies (macros is a proc-macro crate; protocol is a pure SPI). |
| 1 | `openjiuwen-foundation`, `openjiuwen-sysop`, `openjiuwen-session`, `openjiuwen-graph`, `openjiuwen-retrieval` | Session may depend on foundation and common; foundation never depends on session. |
| 2 | `openjiuwen-workflow`, `openjiuwen-context`, `openjiuwen-memory`, `openjiuwen-security` | Workflow sits on graph + session + foundation; memory depends on foundation and retrieval (not context or session). |
| 3 | `openjiuwen-agent`, `openjiuwen-multiagent` | Agent composes foundation, session, workflow, context, memory, security, sysop. |
| 4 | `openjiuwen-runner`, `openjiuwen-core` | Runner owns the process-global orchestrator; core re-exports kernel crates. |
| 5 | `openjiuwen-extensions`, `openjiuwen-dev-tools`, `openjiuwen-symphony`, `openjiuwen-harness` | Depend on `openjiuwen-core` (and `openjiuwen-foundation` where a concrete trait impl is needed). |
| 6 | `openjiuwen-harness-providers`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, `openjiuwen-auto-harness`, `openjiuwen-agent-teams` | Depend on harness and/or core. |
| 7 | `openjiuwen` (umbrella), `openjiuwen-cli` | Top-level convenience and binary. |
| dev | `openjiuwen-test-support`, `openjiuwen-e2e`, `openjiuwen-examples` | Never depended on by library crates. |

Rules:
- If two crates need the same type, it belongs in the lowest common ancestor (usually `openjiuwen-common` or `openjiuwen-foundation`).
- Do not introduce cycles to mirror Python's function-local imports; use traits and registration instead.
- Python's function-local `core → harness/extensions/agent_teams` imports become extension traits implemented in the upper crate and registered at startup.

## 2. Shared contracts

These signatures are the intended public seams. Child architects may refine them, but changing a seam requires updating every dependent crate.

### 2.1 Cards, configs, and errors (`openjiuwen-common`)

```rust
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Serialized identity + metadata; crosses process boundaries.
pub trait Card: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
}

/// Common card fields reused by all concrete cards.
pub struct CardBase { pub id: String, pub name: String, pub description: Option<String>,
                      pub version: Option<String>, pub metadata: BTreeMap<String, Value> }

pub enum Error { Config(String), Validation(String), Llm(String), Tool(String), Workflow(String),
                 Graph(String), Session(String), Memory(String), Retrieval(String), SysOp(String),
                 Security(String), Mcp(String), Transport(String), Timeout(String),
                 NotFound(String), Unsupported(String), Cancelled, External(Box<dyn Error + Send + Sync>) }
```

The Python `Card`/`Config` split is preserved: `*Card` is serializable and registry-safe; `*Config` is process-local, may hold clients/credentials, and is passed to `configure`.

All serialized structs use `#[serde(rename_all = "snake_case", default)]`, tolerate unknown fields, and default missing collections to empty.

### 2.2 Messages, prompts, and models (`openjiuwen-foundation`)

```rust
pub enum Role { System, User, Assistant, Tool, Other(String) }
pub enum ContentPart { Text { text: String }, ImageUrl { url: String, detail: Option<String> },
                       Audio { data: String, format: String }, File { ... }, Other(Value) }
// Messages live in the `prompt` module, not `llm`; `llm` re-exports them.
pub struct BaseMessage { pub role: Role, pub content: MessageContent, pub name: Option<String>,
                         pub tool_calls: Vec<ToolCall>, pub tool_call_id: Option<String>,
                         pub reasoning_content: Option<String>, pub metadata: BTreeMap<String, Value> }
pub enum MessageContent { Text(String), Parts(Vec<ContentPart>) } // serde accepts both forms
pub struct PromptTemplate { pub name: String, pub content: TemplateContent,
                            pub placeholder_prefix: String, pub placeholder_suffix: String }

pub struct ModelClientConfig { pub provider: String, pub api_key: Option<SecretString>,
                               pub api_base: Option<String>, pub verify_ssl: bool,
                               pub timeout: Option<Duration>, pub max_retries: u32,
                               pub headers: BTreeMap<String, String>, pub extra: Value }
pub struct ModelRequestConfig { pub model_name: String, pub temperature: Option<f32>,
                                pub top_p: Option<f32>, pub max_tokens: Option<u32>,
                                pub stop: Vec<String>, pub tools: Option<Value>,
                                pub tool_choice: Option<Value>, pub stream: bool, pub extra: Value }

pub struct ChatResponse { pub id: String, pub model: String, pub choices: Vec<Choice>, pub usage: Usage }
pub enum StreamChunk { Delta { index: usize, content: Option<String>, reasoning: Option<String>,
                              tool_call: Option<ToolCallDelta>, role: Option<Role>,
                              finish_reason: Option<FinishReason> },
                       Usage(Usage), Done }

#[async_trait]
pub trait Model: Send + Sync {
    fn provider(&self) -> &str;
    fn model_name(&self) -> &str;
    async fn invoke(&self, messages: &[BaseMessage], request: &ModelRequestConfig) -> Result<ChatResponse>;
    fn stream(&self, messages: Vec<BaseMessage>, request: ModelRequestConfig)
        -> BoxStream<'static, Result<StreamChunk>>;
}

pub async fn init_model(client: ModelClientConfig, request: ModelRequestConfig) -> Result<Arc<dyn Model>>;
```

Built-in providers: `openai` (all OpenAI-compatible vendors, including Pangu/Qwen/DeepSeek/OpenAI via `api_base`) and `anthropic`.
Providers self-register in a `ModelProviderRegistry`; extension providers implement `ModelProvider` in `openjiuwen-extensions`.

### 2.3 Tools (`openjiuwen-foundation`)

```rust
pub struct ToolCard { pub id: String, pub name: String, pub description: String,
                      pub input_params: Value /* JSON Schema */,
                      pub exposure: ToolExposure, pub execution: ToolExecution,
                      pub parallel_safe: bool, pub metadata: BTreeMap<String, Value> }
pub struct ToolContext { pub session_id: Option<String>, pub invocation_id: String,
                         pub cancellation: CancellationToken, pub metadata: BTreeMap<String, Value>,
                         pub timeout: Option<Duration> }

#[async_trait]
pub trait Tool: Send + Sync {
    fn card(&self) -> &ToolCard;
    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value>;
    fn stream(self: Arc<Self>, inputs: Value, ctx: ToolContext)
        -> BoxStream<'static, Result<Value>> { /* one invoke */ }
    fn render_for_llm(&self, output: &Value) -> Option<String> { None }
}
```

Implementations: `FunctionTool` (closure/typed handler + JSON Schema), `RestfulApiTool`, `McpTool`.
`ToolCard.input_params` is generated from `schemars` where a typed handler exists; the Python `@tool` decorator maps to the `#[tool]` attribute macro in `openjiuwen-macros` or the `FunctionTool::builder`.

### 2.4 Sessions (`openjiuwen-session`)

```rust
#[async_trait]
pub trait Session: Send + Sync {
    fn id(&self) -> &str;
    fn parent_id(&self) -> Option<&str>;
    async fn get_state(&self, key: &str) -> Result<Option<Value>>;
    async fn update_state(&self, patch: Value) -> Result<()>;
    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>>;
    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()>;
    async fn write_stream(&self, item: StreamItem) -> Result<()>;
    async fn request_interaction(&self, request: InteractionRequest) -> Result<InteractionResponse>;
    fn trace(&self) -> &dyn TraceSink;
}
```

Concrete sessions: `WorkflowSession`, `AgentSession`, `AgentTeamSession`, `NodeSession`.
Checkpointing is a trait (`Checkpointer`): in-memory, filesystem/JSON, and optional SQLite backends.
Resume semantics are keyed by `session_id` exactly as in Python; global state is JSON, component state is namespaced by component id.

### 2.5 Graph and workflow (`openjiuwen-graph`, `openjiuwen-workflow`)

```rust
// graph
pub struct GraphState(pub Map<String, Value>);
#[async_trait] pub trait Node: Send + Sync { fn id(&self) -> &str;
    async fn run(&self, state: &mut GraphState, ctx: &GraphContext) -> Result<NodeOutcome>; }
pub struct CompiledGraph { /* pregel/BSP executor */ }
impl CompiledGraph { pub async fn invoke(&self, input: Value, session: Arc<dyn Session>) -> Result<Value>;
                     pub fn stream(&self, input: Value, session: Arc<dyn Session>) -> BoxStream<'static, Result<GraphEvent>>; }

// workflow
pub struct Workflow { /* builder + card */ }
impl Workflow {
    pub fn set_start_comp(&mut self, id: &str, comp: impl Component + 'static,
                          options: ComponentOptions) -> &mut Self;
    pub fn add_workflow_comp(&mut self, id: &str, comp: impl Component + 'static,
                             options: ComponentOptions) -> &mut Self;
    pub fn set_end_comp(&mut self, id: &str, comp: impl Component + 'static,
                        options: ComponentOptions) -> &mut Self;
    pub fn add_connection(&mut self, from: &str, to: &str) -> &mut Self;
    pub fn add_connections(&mut self, from: &[&str], to: &str) -> &mut Self;
    pub fn add_conditional_connection(&mut self, from: &str, to: &str,
                                      router: Arc<dyn Router>) -> &mut Self;
    pub fn add_stream_connection(&mut self, from: &str, to: &str) -> &mut Self;
    pub fn build(&self) -> Result<CompiledWorkflow>;
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>) -> Result<WorkflowOutput>;
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>) -> BoxStream<'static, Result<WorkflowChunk>>;
}
```

Data flow uses textual references `${node.field}` exactly as in Python; resolution happens in the engine, not in components.
Component capabilities (`invoke`, `stream`, `transform`, `collect`) determine legal edge kinds.

### 2.6 Agents (`openjiuwen-agent`) and runner (`openjiuwen-runner`)

```rust
#[async_trait]
pub trait Agent: Send + Sync {
    fn card(&self) -> &AgentCard;
    async fn invoke(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value, session: Option<Arc<dyn Session>>)
        -> BoxStream<'static, Result<AgentChunk>>;
}
```

`ReActAgent::configure(ReActAgentConfig)`, `WorkflowAgent::add_workflows`, and `LLMAgent::configure` mirror the Python call sites.
`AbilityManager` mounts tool cards on an agent; tools are resolved from the agent's registry or the `Runner` resource registry.

```rust
pub struct Runner { /* resources, callbacks, MQ, config */ }
impl Runner {
    pub fn new() -> Arc<Self>;
    pub fn global() -> Arc<Self>;             // process-global, documented singleton
    pub fn resource_mgr(&self) -> &Arc<ResourceMgr>;
    pub async fn run_agent(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> Result<AgentOutput>;
    pub fn run_agent_streaming(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<AgentChunk>>;
    pub async fn run_workflow(&self, workflow: impl Into<WorkflowRef>, inputs: Value, options: RunOptions) -> Result<WorkflowOutput>;
    pub fn run_workflow_streaming(&self, workflow: impl Into<WorkflowRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub async fn run_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<Value>;
    pub fn run_agent_team_streaming(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<Value>>;
}
```

`RunOptions` carries the optional caller session, deadline, cancellation token, session id, envs, and metadata; an id passed as `AgentRef`/`WorkflowRef`/`TeamRef` is resolved through `ResourceMgr`.

`ResourceMgr` registers tools, workflows, agents, and teams by id. Tests use local `Runner` instances; production code may use `Runner::global()` for parity.

### 2.7 Harness protocol (`openjiuwen-harness-protocol`)

Provider-neutral SPI with no workspace dependencies: `HarnessProtocol`, `HarnessProvider`, `HarnessModelControl`, plus serde models for events, results, and streams.
`openjiuwen-harness` implements the native provider; `openjiuwen-harness-providers` registers providers (`native` implemented; `native_v2` and `claudecode`, `codex`, `dsh` declared but returning `Error::Unsupported`, the last three behind the `vendor` feature).

## 3. Async and streaming model

- All I/O is async on tokio; CPU-bound work uses `spawn_blocking`.
- Public streaming APIs return `BoxStream<'static, Result<T>>` built with `async_stream::try_stream!`.
- Long-running loops check `CancellationToken` at await points; timeouts use `tokio::time::timeout`.
- Structured concurrency: spawned tasks are joined or explicitly detached with a documented owner; no detached task may outlive its `Runner` without being tracked.
- Shared mutable state uses `Arc<RwLock<..>>`/`Arc<Mutex<..>>`; locks are never held across `.await`.

## 4. Configuration and environment

- No central config file, matching Python. Configs are typed structs at call sites; YAML is only for packaged resources (prompts, templates, skills).
- Environment variables: `OPENJIUWEN_HOME`, `OPENJIUWEN_API_KEY`, `OPENJIUWEN_SERVER_URL`, `OPENJIUWEN_BASH_STRICT`, `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS`, `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE`, plus `API_KEY`, `API_BASE`, `MODEL_PROVIDER`, `MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`.
- `.env` loading is opt-in (`dotenvy`) at binary entry points only, never in library constructors.
- Secrets use a `SecretString` wrapper that redacts in `Debug` and logging; serialization preserves the value so configuration round-trips, and credentials are never logged.

## 5. Feature flags

Default features must compile with no external services.
- `openjiuwen-foundation`: `mcp` (default), `tiktoken` (default), `sqlite`, `live-tests`.
- `openjiuwen-session`: `sqlite` (optional).
- `openjiuwen-memory`: `sqlite` (optional).
- `openjiuwen-sysop`: `sandbox` (optional; HTTP sandbox gateway).
- `openjiuwen-symphony`: `sqlite` (optional asset store).
- `openjiuwen-extensions`: `a2a`, `otel`, `sqlite`, `redis`, `pulsar`, `mcp`, `live-tests` (all optional).
- `openjiuwen-harness`: `native` (default), `browser`/`cua`/`mobile` (optional external workers), `notify` (optional workspace watcher), `live-tests`.
- `openjiuwen-harness-providers`: `native` (default), `vendor` (declares the `claudecode`/`codex`/`dsh` surface), `live-tests`.
- `openjiuwen-agent-teams`: `mcp` (default), `distributed` (empty marker), `live-tests`.
- `openjiuwen`: `native` (default), `otel`, `a2a`, `sqlite`, `live-tests`.
- `openjiuwen-*`: `live-tests` is never enabled by default.
- The `native` provider is default-on in `openjiuwen-harness-providers`, and the umbrella depends on that crate non-optionally, so `openjiuwen::harness_providers::create_harness("native", …)` and `resolve_provider("native")` are always available; vendor providers require a direct dependency with `features = ["vendor"]`.

## 6. Testing infrastructure

- `openjiuwen-test-support` provides:
  - `MockLlmServer` — an axum-based OpenAI-compatible endpoint supporting chat completions, SSE streaming, tool calls, and scripted responses;
  - scripted `Model` fakes;
  - temporary workspace and session fixtures;
  - JSON snapshots for Cards.
- `openjiuwen-e2e` contains cross-crate scenarios: ReAct loop with tool use, workflow graph execution + streaming, session checkpoint/resume, memory recall, MCP stdio round-trip, and CLI smoke tests.
- Unit tests live next to modules; behavior-port tests reference the corresponding Python test file in a comment.
- Tests must be deterministic and run offline; any test needing network is `#[ignore]`d and documented.

## 7. Porting method

1. Read the Python module and its unit tests through a read-only investigator.
2. Preserve serialized names, default values, and observable behavior.
3. Replace Python-specific mechanisms (pydantic, asyncio globals, reflection) with Rust equivalents described above.
4. Port representative unit tests; document gaps in `docs/PARITY.md`.
5. Never modify the Python reference repository.
