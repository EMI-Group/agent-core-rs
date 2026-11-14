# operator/legacy

## Intent

The legacy single-call LLM wrapper: one model call (or one stream) plus an optimizer trace callback.
Ports `openjiuwen/core/operator/legacy/llm_call/base.py`.
It is distinct from `LLMCallOperator` in the parent module — that type is a tunable prompt handle with no call or callback surface — so an evolving consumer that needs Python's `LLMCall` uses this wrapper.

## API Surface

```rust
pub const DEFAULT_USER_PROMPT: &str = "{{query}}";

#[derive(Clone)] pub struct OptimizerTrace {
    pub llm_call_id: String, pub inputs: Value, pub response: String,
    pub session: Option<Arc<dyn openjiuwen_session::Session>>,
}
pub type OptimizerTraceFuture = Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>>;
pub type OptimizerCallback = Arc<dyn Fn(OptimizerTrace) -> OptimizerTraceFuture + Send + Sync>;

pub struct LLMCall;
impl LLMCall {
    pub fn new(model_name, Arc<dyn Model>, system_prompt: impl Into<TemplateContent>,
               user_prompt: impl Into<TemplateContent>) -> Self;   // freezes: system false, user true; id "llm_call"
    pub fn with_llm_call_id(self, id); pub fn with_freeze_system_prompt(self, bool);
    pub fn with_freeze_user_prompt(self, bool);
    pub fn model_name(); pub fn model() -> &Arc<dyn Model>;
    pub fn get_system_prompt() -> &PromptTemplate; pub fn get_user_prompt() -> &PromptTemplate;
    pub fn update_system_prompt(&mut self, impl Into<TemplateContent>) -> bool;
    pub fn update_user_prompt(&mut self, impl Into<TemplateContent>) -> bool;
    pub fn set_freeze_system_prompt(&mut self, bool); pub fn set_freeze_user_prompt(&mut self, bool);
    pub fn get_freeze_system_prompt(); pub fn get_freeze_user_prompt();
    pub fn get_optimizer_callback() -> Option<&OptimizerCallback>;
    pub fn set_optimizer_callback(&mut self, Option<OptimizerCallback>);
    pub async fn invoke(&self, inputs: &Value, session: Option<Arc<dyn Session>>,
                        history: Option<&[BaseMessage]>, tools: Option<&[Value]>) -> Result<BaseMessage>;
    pub fn stream(&self, inputs: Value, session: Option<Arc<dyn Session>>,
                  history: Option<Vec<BaseMessage>>, tools: Option<Vec<Value>>)
        -> BoxStream<'static, Result<StreamChunk>>;
}
```
`mod.rs` re-exports `LLMCall`, `OptimizerCallback`, `OptimizerTrace`, `OptimizerTraceFuture`, and this module's `DEFAULT_USER_PROMPT` (the parent module re-exports `LLMCall` and the optimizer types but keeps its own `DEFAULT_USER_PROMPT` from `llm_call.rs`).

## Constraints

- `LLMCall` mutates in place through the `update_*_prompt` methods; it is not `Clone` because it owns a mutable optimizer callback and a live model handle.
- Rendering uses `openjiuwen_foundation::PromptTemplate` with the `inputs` object as the value map; a non-object input renders the templates unchanged.
- Frozen prompt updates are a silent no-op returning `false`, matching Python (never an error).
- No retry loop, no error swallowing: model and callback errors propagate; no logging, no stdout.
- Files stay under ~1000 lines; tests live beside the implementation.

## Design Decisions

- Python's async callable is expressed as `Arc<dyn Fn(OptimizerTrace) -> OptimizerTraceFuture + Send + Sync>`; the trace bundles `(llm_call_id, inputs, response, session)` into one cloneable struct so the callback stays object-safe.
- `invoke` returns the model's `BaseMessage` only after the callback succeeds; `stream` yields every chunk first, then awaits the callback once and emits any callback error as one final `Err`.
- The system prompt's rendered messages are forced to the system role, mirroring Python's `SystemMessage(content=msg.content)` wrapper.
- `tools` are forwarded in provider wire shape as `request.tools = Value::Array(..)`; the wrapper performs no schema translation.

## Test Strategy

- 11 offline tests on `ScriptedModel`: message order (system → history → user), tool forwarding and model name, one optimizer trace with its id/inputs/response, callback-error propagation, model-error propagation without a callback, streamed delta accumulation plus one trace, stream model error with no trace, callback error as the final stream item, freeze no-ops, accessors, empty-user-prompt fallback, and non-object input rendering.
- Run with `cargo test -p openjiuwen-agent --offline operator::legacy`.

## Routing Table

- `./mod.rs` → module declaration and re-exports
- `./llm_call.rs` → `LLMCall`, `OptimizerTrace`, `OptimizerCallback`, `OptimizerTraceFuture`
