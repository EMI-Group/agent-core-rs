# openjiuwen-agent operator module

## Intent

Tunable parameter handles for self-evolution: an optimiser reads the tunables, mutates a handle, and the handle pushes the change to the live call path through a callback.
Ports `openjiuwen/core/operator/` (`base.py`, `llm_call/`, `tool_call/`, `memory_call/`, `skill_call/`, `context_evolve_call/`) onto this crate's `Model`, `ToolCard`, `Skill`, and `openjiuwen_memory` types.
`ReActAgentEvolve` (`../evolve/`) consumes this module; the `ParameterCallback` seam is the contract between a tuned handle and the agent configuration.

## API Surface

### `base.rs`

```rust
pub type ParameterCallback = Arc<dyn Fn(&str, &Value) + Send + Sync>;

pub enum TunableKind { Text, TextList, Number, Bool, Json }   // serde snake_case
pub struct TunableSpec { pub name: String, pub description: String, pub kind: TunableKind, pub frozen: bool }
impl TunableSpec { new(name, kind); with_description(..); with_frozen(bool);
                   validate(&self, value: &Value) -> Result<()> }

pub enum UpdateValue { Set(Value), Merge(Value), Append(Value) }  // serde {mode, payload}; Set wire name "replace" (accepts "set")
impl UpdateValue { set(v); merge(v); append(v); payload() -> &Value; mode() -> "replace"|"merge"|"append" }

pub struct ApplyResult { pub applied: bool, pub target: String, pub reason: String }
impl ApplyResult { applied(target); skipped(target, reason); ok() -> bool }

#[async_trait] pub trait Operator: Send + Sync {
    fn operator_id(&self) -> &str;
    fn get_tunables(&self) -> Vec<TunableSpec>;
    fn get_state(&self) -> Value;
    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult>;
    fn load_state(&mut self, state: Value) -> Result<()>;
    fn apply_update(&mut self, target: &str, update: UpdateValue) -> Result<ApplyResult>;  // default: apply_merge + set_parameter
}
#[async_trait] pub trait PreviewableOperator: Operator {
    fn preview_update(&self, target: &str, update: &UpdateValue) -> Result<ApplyResult>;
}
pub fn apply_merge(current: &Value, update: &UpdateValue) -> Value;  // objects merge shallowly, arrays/strings append, scalars replace
```

### `llm_call.rs` — `LLMCallOperator`

`new(operator_id)`; builders `with_system_prompt(Vec<BaseMessage>)`, `with_user_prompt`, `with_model`, `with_temperature(f32)`, `with_freeze_system_prompt(bool)`, `with_freeze_user_prompt(bool)`, `with_callback(ParameterCallback)`.
Accessors `system_prompt()`, `user_prompt()`, `freeze_system_prompt()`, `freeze_user_prompt()`, `model()`, `temperature()`.
Tunables `system_prompt` (TextList), `user_prompt` (Text), `model` (Text), `temperature` (Number); frozen ones report `frozen: true`.
`render_messages(&self, query: &str) -> Vec<BaseMessage>`; `async call(&self, model: &dyn Model, messages: &[BaseMessage], request: ModelRequestConfig) -> Result<ChatResponse>`.
Defaults: empty system prompt, user prompt `{{query}}`, system prompt evolvable, user prompt frozen.

### `tool_call.rs` — `ToolCallOperator`

`new(operator_id)`, `with_descriptions(IntoIterator<Item = (String, String)>)`, `with_callback`.
Tunables: `tool_description` (Json map) plus one `tool_description.<name>` (Text) per cached tool; none when the cache is empty.
`descriptions() -> &BTreeMap`, `description(name) -> Option<&str>`, `apply_to_cards(&self, cards: &mut [ToolCard])`.
`set_parameter` accepts the whole map or one per-tool target (text sets, `null` removes) and always fires the callback with the whole map.

### `memory_call.rs` — `MemoryCallOperator` / `MemoryRecall`

`new(operator_id)`, `with_callback`; accessors `top_k()`, `threshold()`, `memory_scope_id()`, `enabled()`.
Tunables `top_k` (Number, clamped to `0..=100`), `threshold` (Number, clamped to `0.0..=1.0`), `memory_scope_id` (Text), `enabled` (Bool).
`async recall(&self, memory: &dyn MemoryRecall, query: &str) -> Result<Vec<String>>`; disabled returns `Ok(vec![])` without touching the backend.

```rust
#[async_trait] pub trait MemoryRecall: Send + Sync {
    async fn recall_texts(&self, scope_id: &str, query: &str, top_k: usize, threshold: f32) -> Result<Vec<String>>;
}
```
Implemented for `openjiuwen_memory::LongTermMemory` (empty scope id maps to `DEFAULT_ID`) and for `Arc<T: MemoryRecall + ?Sized>`.

### `skill_call.rs` — `SkillCallOperator`

`new(operator_id)`, `with_selected_skills(names)`, `with_max_skills(n)`, `with_callback`; accessors `selected_skills()`, `max_skills()`.
Tunables `selected_skills` (TextList; one text is accepted as a single name) and `max_skills` (Number, default 5, clamped to `0..=100`).
`select<'a>(&self, skills: &'a [Skill], query: &str) -> Vec<&'a Skill>`: curated candidates → `Skill::matches` → fallback to all candidates → truncate to `max_skills` (0 selects nothing).

### `context_evolve.rs` — `ContextEvolveOperator` / `MetisContextEvolveOperator`

```rust
pub enum UpdatePolicy { Append, Replace, Merge }   // serde snake_case; parse(&str)
```
`ContextEvolveOperator::new(operator_id)`; builders `with_system_prompt`, `with_policy`, `with_max_messages`, `with_callback`; accessors `system_prompt()`, `policy()`, `max_messages()`.
Tunables `system_prompt` (Text), `policy` (Text), `max_messages` (Number; `0` = no truncation, clamped to `0..=1000`).
`evolve(&self, messages: &[BaseMessage]) -> Result<Vec<BaseMessage>>`: Append pushes a trailing system message, Replace rewrites the first system message (inserting one when absent), Merge concatenates onto the first system message, then truncation keeps the first system message plus the newest non-system turns.
`MetisContextEvolveOperator`: tunables `summary` and `max_messages`; injects `[metis-summary]\n\n<summary>` into the leading system message, then truncates.
`learn(&self, &[String]) -> Result<()>` always returns `Error::Unsupported` (the ML-backed Metis learner is not ported).

## Constraints

- Only this module's files are edited; consumers (`evolve/`, agents) read through the public API above.
- All values cross the seam as `serde_json::Value`; every state object is a JSON object so `load_state` can restore field by field.
- Files stay under ~500 lines; tests live in `#[cfg(test)] mod tests` beside each handle.
- Offline and deterministic: `openjiuwen_test_support::ScriptedModel` for the LLM call, fakes for the memory seam, no network.

## Design Decisions

- `set_parameter` returns `Result<ApplyResult>` instead of Python's `None`: unknown targets, frozen tunables, and type-mismatched values return `ApplyResult::skipped`, while structural state errors in `load_state` return `Error::Validation`.
- `get_tunables` includes frozen specs with `frozen: true` (Python omits them) so an optimiser can see the full parameter surface and the agent can re-derive its own policy.
- `UpdateValue` uses the Python `{mode, payload}` wire shape with modes `replace`/`merge`/`append`; `ApplyResult` is reduced to `{applied, target, reason}` because operator identity and update metadata are already known to the caller.
- Preview-only handles override `apply_update` to route into `preview_update`, so an update never mutates a preview handle (Python `PreviewableOperator` semantics).
- `LLMCallOperator` adds a real call path (Python's handle only stores prompts): the query comes from the last user message, earlier non-system messages are kept as history, and the operator's model/temperature overrides replace the request's fields.
- `MemoryCallOperator` is wired to the real `LongTermMemory::recall` through a local `MemoryRecall` seam, so the memory crate stays untouched and tests can fake recall.
- `SkillCallOperator` tunes selection rather than Python's skill-experience staging; experience persistence stays with the caller.

## Notes for Agents

- A successful `set_parameter` fires the callback exactly once with the target's canonical stored value; `load_state` fires once per field present in the state object, including frozen ones.
- `load_state` bypasses freeze markers; `set_parameter` does not.
- For `ToolCallOperator` the callback's value is always the whole description map, never a single entry, matching Python.
- `apply_merge` on a `Text` target with `UpdateValue::Append` concatenates text; on a `TextList`/array target it concatenates items; on a JSON object it merges shallowly.
- `MemoryCallOperator::recall` takes the memory implementation as an argument; the operator itself stores no memory handle.
- `MetisContextEvolveOperator::preview_update` validates `summary`/`max_messages` only; any other target is returned as `ApplyResult::skipped`, not an error.
- `LLMCallOperator::render_messages` renders system messages first, then one user message; `call` drops input system messages and keeps the pre-query history.

## Deferred Python Behavior

- Not ported: `openjiuwen/core/operator/legacy/llm_call` (`LLMCall` execution wrapper) — the Rust `LLMCallOperator::call` covers the call path.
- Not ported: Python's `max_retries` memory tunable and its `SkillExperienceOperator` staging lifecycle (`updates_generated -> local_apply_completed`).
- Not ported: the ML-backed Metis store; `MetisContextEvolveOperator::learn` returns `Error::Unsupported`.
- Not ported: Python's per-operator `ApplyResult` metadata (`operator_id`, `mode`, `effect`, `records`, `errors`, `change_type`) and the `agent_evolving` update-protocol constants.

## Test Strategy

- Unit tests beside each module (38 total): tunable metadata/freeze, callback firing, frozen skips, state round-trips, `apply_merge` semantics, render/substitution, a real `ScriptedModel` call asserting the rendered messages and request overrides, tool-description projection onto `ToolCard`s, memory off/on + parameter forwarding through a fake seam, skill matching/curation/caps, context policies and truncation, and preview-without-mutation.
- All tests are offline (`ScriptedModel`, in-memory memory engine) and run under `cargo test -p openjiuwen-agent --offline operator::`.

## Routing Table

- `./base.rs` → operator traits, tunables, structured updates, `apply_merge`
- `./llm_call.rs` → `LLMCallOperator`, prompt rendering, model call
- `./tool_call.rs` → `ToolCallOperator`, description map and `ToolCard` projection
- `./memory_call.rs` → `MemoryCallOperator`, `MemoryRecall` seam
- `./skill_call.rs` → `SkillCallOperator`, bounded skill selection
- `./context_evolve.rs` → `ContextEvolveOperator`, `MetisContextEvolveOperator`, `UpdatePolicy`
