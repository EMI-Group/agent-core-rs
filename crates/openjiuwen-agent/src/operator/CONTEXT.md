# openjiuwen-agent operator module

## Intent

Tunable parameter handles for self-evolution: an optimiser reads the tunables, mutates a handle, and the handle pushes the change to the live call path through a callback.
Ports `openjiuwen/core/operator/` (`base.py`, `llm_call/`, `tool_call/`, `memory_call/`, `skill_call/`, `context_evolve_call/`, `legacy/llm_call/`) plus the operator-facing contracts of `openjiuwen/agent_evolving/` (`types.py`, `protocols.py`, `update_execution.py`, `experience/`).
`ReActAgentEvolve` (`../evolve/`) consumes this module; the `ParameterCallback` seam is the contract between a tuned handle and the agent configuration.

## API Surface

### `base.rs`

```rust
pub type ParameterCallback = Arc<dyn Fn(&str, &Value) + Send + Sync>;

pub enum TunableKind { Text, TextList, Number, Bool, Json, SkillExperience }   // serde snake_case
pub struct TunableSpec { pub name: String, pub description: String, pub kind: TunableKind,
                         pub frozen: bool, pub path: String, pub constraint: Option<Value> }
impl TunableSpec { new(name, kind); with_description(..); with_frozen(bool); with_path(..); with_constraint(Value);
                   validate(&self, value: &Value) -> Result<()> }
// `path` and `constraint` are omitted from the wire form when empty/None; `SkillExperience` and `Json` accept any value.

pub enum UpdateMode { Replace, Append, Merge }        // wire "replace" (accepts "set"), "append", "merge"; parse/as_str/is_replace
pub enum UpdateEffect { State, PendingChange }        // wire "state" / "pending_change"; parse/as_str/is_state

pub struct UpdateValue { pub payload: Value, pub mode: UpdateMode, pub effect: UpdateEffect,
                         pub change_type: Option<String>, pub metadata: BTreeMap<String, Value> }
impl UpdateValue { new(payload, mode); set(..); merge(..); append(..);
                   with_effect(..); with_change_type(..); with_metadata(..); with_metadata_entry(..);
                   payload(); mode() -> "replace"|"append"|"merge"; is_state(); is_pending_change() }
// serde: snake_case, `mode`/`payload` always present, `effect`/`change_type`/`metadata` skipped at their defaults.

pub struct ApplyResult {
    pub operator_id: String, pub target: String, pub applied: bool,
    pub mode: UpdateMode, pub effect: UpdateEffect, pub value: Value,
    pub records: Vec<Value>, pub change_type: Option<String>, pub lifecycle_stage: Option<String>,
    pub pending_change_id: Option<String>, pub errors: Vec<String>,
    pub metadata: BTreeMap<String, Value>, pub reason: String,
}
impl ApplyResult { applied(target); skipped(target, reason); rejection(operator_id, target, &UpdateValue, errors);
                   with_operator_id/with_mode/with_effect/with_value/with_records/with_change_type/
                   with_lifecycle_stage/with_pending_change_id/with_error/with_metadata/with_metadata_entry;
                   ok() = applied && errors.is_empty() }
// serde: snake_case, empty fields skipped; `reason` is a Rust-only note from the legacy constructors.

#[async_trait] pub trait Operator: Send + Sync {
    fn operator_id(&self) -> &str;
    fn get_tunables(&self) -> Vec<TunableSpec>;
    fn get_state(&self) -> Value;
    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult>;
    fn load_state(&mut self, state: Value) -> Result<()>;
    fn apply_update(&mut self, target: &str, update: UpdateValue) -> Result<ApplyResult>;
}
#[async_trait] pub trait PreviewableOperator: Operator {
    fn preview_update(&self, target: &str, update: &UpdateValue) -> Result<ApplyResult>;
}
pub fn apply_merge(current: &Value, update: &UpdateValue) -> Value;   // objects shallow-merge, arrays/strings append, otherwise replace
```
`apply_update` default: a `pending_change`/non-state effect is rejected (`errors` naming `mode/effect`), an unknown target is rejected with `errors`, and a state update is merged/appended ([`apply_merge`]), validated against the spec, then delegated to `set_parameter`; the result carries the operator id, mode, effect, payload, change type, and metadata.
`PreviewableOperator` implementors override `apply_update` to route into `preview_update`; the trait itself adds only the preview method.

### `protocol.rs`

```rust
pub const REPLACE_MODE/APPEND_MODE/MERGE_MODE: &str;
pub const STATE_EFFECT/PENDING_CHANGE_EFFECT: &str;
pub const LOCAL_APPLY_COMPLETED: &str;            // "local_apply_completed"
pub const EXPERIENCES_TARGET: &str;               // "experiences"
pub const SKILL_EXPERIENCE_ENTRY: &str;           // "skill_experience_entry"
pub const EXPERIENCE_ENTRY: &str;                 // "experience_entry"
pub const TASK_MEMORY_TARGET: &str;               // "task_memory"
pub const TASK_MEMORY_ENTRY: &str;                // "task_memory_entry"

pub type UpdateKey = (String, String);            // (operator_id, target)
pub fn update_key(operator_id, target) -> UpdateKey;

pub fn normalize_update_value(value: impl Into<Value>, target: Option<&str>) -> UpdateValue;
pub fn normalize_updates<I>(updates: I) -> Vec<(UpdateKey, UpdateValue)>
where I: IntoIterator<Item = (UpdateKey, Value)>;

pub struct ApplySummary { pub total: usize, pub applied: usize, pub failed: usize }   // is_empty()
pub fn summarize_apply_results(results: &[ApplyResult]) -> ApplySummary;
```
Legacy `experiences` payloads normalise to append + pending change + `skill_experience_entry` (metadata mirrors the tag); every other target keeps replace + state.
A JSON object carrying a parseable `mode` and a `payload` is treated as an already-structured update (Rust has no `UpdateValue` instance to detect; Python's `isinstance` check is replaced by this shape check).

### `llm_call.rs` — `LLMCallOperator`

`new(operator_id)`; builders `with_system_prompt(Vec<BaseMessage>)`, `with_user_prompt`, `with_model`, `with_temperature(f32)`, `with_freeze_system_prompt(bool)`, `with_freeze_user_prompt(bool)`, `with_callback(ParameterCallback)`.
Accessors `system_prompt()`, `user_prompt()`, `freeze_system_prompt()`, `freeze_user_prompt()`, `model()`, `temperature()`.
Tunables `system_prompt` (TextList), `user_prompt` (Text), `model` (Text), `temperature` (Number); frozen ones report `frozen: true`.
`render_messages(&self, query: &str) -> Vec<BaseMessage>`; `async call(&self, model: &dyn Model, messages: &[BaseMessage], request: ModelRequestConfig) -> Result<ChatResponse>`.
Defaults: empty system prompt, user prompt `{{query}}`, system prompt evolvable, user prompt frozen.

### `tool_call.rs` — `ToolCallOperator`

`new(operator_id)`, `with_descriptions(IntoIterator<Item = (String, String)>)`, `with_callback`.
Tunables: `tool_description` (Json map) plus one `tool_description.<name>` (Text) per cached tool; none when the cache is empty.
`descriptions()`, `description(name)`, `apply_to_cards(&self, cards: &mut [ToolCard])`.
`set_parameter` accepts the whole map or one per-tool target (text sets, `null` removes) and always fires the callback with the whole map.

### `memory_call.rs` — `MemoryCallOperator` / `MemoryRecall`

`new(operator_id)`, `with_callback`; accessors `top_k()`, `threshold()`, `memory_scope_id()`, `enabled()`, `max_retries()`.
Tunables `top_k` (Number, `0..=100`), `threshold` (Number, `0.0..=1.0`), `memory_scope_id` (Text), `enabled` (Bool), `max_retries` (Number, `0..=5`, default 0, constraint `{"type":"int","min":0,"max":5}`).
`async recall(&self, memory: &dyn MemoryRecall, query: &str) -> Result<Vec<String>>`: disabled returns `Ok(vec![])` without touching the backend; a failing backend is retried up to `max_retries` additional attempts (no delay), and the last error is returned unchanged.

```rust
#[async_trait] pub trait MemoryRecall: Send + Sync {
    async fn recall_texts(&self, scope_id: &str, query: &str, top_k: usize, threshold: f32) -> Result<Vec<String>>;
}
```
Implemented for `openjiuwen_memory::LongTermMemory` (empty scope id maps to `DEFAULT_ID`) and for `Arc<T: MemoryRecall + ?Sized>`.

### `skill_call.rs` — `SkillCallOperator`

`new(operator_id)`, `with_selected_skills(names)`, `with_max_skills(n)`, `with_callback`; accessors `selected_skills()`, `max_skills()`.
Tunables `selected_skills` (TextList; one text is accepted as a single name) and `max_skills` (Number, default 5, `0..=100`).
`select<'a>(&self, skills: &'a [Skill], query: &str) -> Vec<&'a Skill>`: curated candidates → `Skill::matches` → fallback to all candidates → truncate to `max_skills` (0 selects nothing).

### `skill_experience.rs` — `SkillExperienceOperator`

```rust
pub struct SkillExperienceOperator;
impl SkillExperienceOperator {
    pub fn new(skill_name: impl Into<String>, callback: Option<ParameterCallback>) -> Self;  // operator_id = "skill_experience_<name>"
    pub fn with_callback(self, callback: ParameterCallback) -> Self;
    pub fn skill_name(&self) -> &str;
}
```
`get_tunables` is one `experiences` spec (kind `TunableKind::SkillExperience`, path `content`, constraint `{"type": "record"}`); `get_state` is always `{}` and `load_state` is a no-op.
`preview_update(target, update)` rejects in Python's order: wrong target (error names it, metadata copied), then `effect != pending_change` or a mode outside `{append, merge}` (error names `mode/effect`).
On acceptance `records` is the payload as a list — a non-list payload, including JSON `null`, is wrapped into a one-element list (Python's `isinstance(list)` check); `applied = !records.is_empty()`, `lifecycle_stage = Some("local_apply_completed")`, `change_type` comes from the update, and metadata is the update metadata plus `skill_name`.
`set_parameter` is the direct-compatibility notification path: it never mutates state, wraps a non-list payload into a list, fires the callback once with the list, and returns an applied pending-change result carrying the records; a `null` payload or a foreign target returns a non-applied result without firing.
`apply_update` routes to `preview_update`, so a preview handle never mutates.

### `staging.rs` — caller-owned pending-change lifecycle

```rust
pub struct LocalApplyPreview { pub skill_name: String, pub records: Vec<Value>, pub change_type: String,
                               pub lifecycle_stage: String, pub apply_results: Vec<ApplyResult> }
pub fn build_local_apply_preview(skill_name: &str, apply_results: &[ApplyResult]) -> Result<LocalApplyPreview>;

#[async_trait] pub trait SkillExperienceStore: Send + Sync {
    async fn append_record(&self, skill_name: &str, record: &Value) -> Result<()>;
}
pub struct InMemorySkillExperienceStore;   // new(), with_failure_after(n), records(), len(), is_empty()
                                          // injected failure is Error::memory(...); no new deps

pub enum PendingChangeStatus { PendingApproval, Persisted, Partial, Rejected }   // as_str() snake_case
pub struct PendingChange { pub change_id: String, pub skill_name: String, pub records: Vec<Value>,
                           pub change_type: String, pub metadata: BTreeMap<String, Value>,
                           pub created_seq: u64, pub status: PendingChangeStatus }
pub struct PendingChangeOutcome { pub change_id: String, pub skill_name: String, pub status: PendingChangeStatus,
                                  pub applied_count: usize, pub pending_count: usize,
                                  pub rejected_count: usize, pub errors: Vec<String> }

pub struct PendingChangeStore;   // new(), with_store(Arc<dyn SkillExperienceStore>), set_store(..),
                                 // stage(&LocalApplyPreview) -> PendingChange, list() -> Vec<PendingChange>, get(id),
                                 // async approve(id), async reject(id), async retry(id) -> Result<PendingChangeOutcome>
```
`build_local_apply_preview` drops `applied == false` results, concatenates their records in order, takes the last non-`None` change type (default `skill_experience_entry`), and returns `Error::Validation` for an applied result whose `lifecycle_stage` is not `local_apply_completed`.
`stage` mints `skill_evolve_<8 hex digits>` from a process-local counter, assigns a strictly increasing `created_seq`, and stores the snapshot as `pending_approval`.
`approve` writes every remaining record through the sink in order: full success removes the entry (`persisted`, `applied_count` = records), a write failure keeps the unwritten tail staged (`partial`, `applied_count` = successes, one error per failed write), and an empty change removes the entry with `persisted`/0.
`reject` removes the entry without writing and reports `rejected`.
`retry` commits only the remaining tail with the same partial semantics.
An unknown id is `Error::NotFound`; `approve`/`retry` without an installed sink is `Error::Config`.

### `context_evolve.rs` — `ContextEvolveOperator` / `MetisContextEvolveOperator`

```rust
pub enum UpdatePolicy { Append, Replace, Merge }   // serde snake_case; parse(&str)
```
`ContextEvolveOperator::new(operator_id)`; builders `with_system_prompt`, `with_policy`, `with_max_messages`, `with_callback`.
Tunables `system_prompt` (Text), `policy` (Text), `max_messages` (Number; `0` = no truncation, `0..=1000`).
`evolve(&self, messages: &[BaseMessage]) -> Result<Vec<BaseMessage>>`: Append pushes a trailing system message, Replace rewrites the first system message (inserting one when absent), Merge concatenates onto the first system message, then truncation keeps the first system message plus the newest non-system turns.
`MetisContextEvolveOperator`: tunables `summary` and `max_messages`; injects `[metis-summary]\n\n<summary>` into the leading system message, then truncates.
`learn(&self, &[String]) -> Result<()>` always returns `Error::Unsupported` (the ML-backed Metis learner is not ported).

### `legacy/` — `LLMCall`

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
               user_prompt: impl Into<TemplateContent>) -> Self;
    pub fn with_llm_call_id(self, id); pub fn with_freeze_system_prompt(self, bool);
    pub fn with_freeze_user_prompt(self, bool);
    pub fn model_name(); pub fn model() -> &Arc<dyn Model>;
    pub fn get_system_prompt() -> &PromptTemplate; pub fn get_user_prompt() -> &PromptTemplate;
    pub fn update_system_prompt(&mut self, impl Into<TemplateContent>) -> bool;   // frozen -> silent false
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
`LLMCall` is the callable, legacy wrapper and is distinct from `LLMCallOperator` (a tunable prompt handle): it renders both `PromptTemplate`s from the `inputs` object, appends `history`, drives one model call (or one stream), and then awaits the optimizer callback once with `(llm_call_id, inputs, response text, session)`.
Message order is rendered system messages (forced to the system role), then history, then rendered user messages; `tools` are forwarded as `request.tools = Value::Array(..)`, and an empty user prompt falls back to `{{query}}`.
No retry loop and no error swallowing: a model error propagates (`invoke`) or ends the stream (`stream`) before the callback runs, and a callback error propagates from `invoke` or is emitted as the stream's final `Err` item.

## Constraints

- Only this module's files are edited; consumers (`evolve/`, agents) read through the public API above.
- All values cross the seam as `serde_json::Value`; every state object is a JSON object so `load_state` can restore field by field.
- `ApplyResult`/`UpdateValue` are serde-serializable (snake_case, empty fields skipped) even though Python does not serialize them, because they cross this crate's public boundary.
- Files stay under ~1000 lines; tests live in `#[cfg(test)] mod tests` beside each handle.
- Offline and deterministic: `openjiuwen_test_support::ScriptedModel` for LLM calls, fakes for the memory and experience sinks, no network, no wall-clock ordering.

## Design Decisions

- `set_parameter` returns `Result<ApplyResult>` instead of Python's `None`: unknown targets, frozen tunables, and type-mismatched values return a non-applied result (`reason` for the legacy constructors, `errors` for structured updates), while structural state errors in `load_state` return `Error::Validation`.
- `get_tunables` includes frozen specs with `frozen: true` (Python omits them) so an optimiser can see the full parameter surface.
- `UpdateValue` is a struct, not the former three-variant enum, so it can carry Python's `effect`, `change_type`, and `metadata`; `set`/`merge`/`append` remain the constructors and `mode()` keeps its `&'static str` result.
- `UpdateMode`/`UpdateEffect` are typed enums with `parse`/`as_str`; mode deserialization accepts the legacy `set` spelling.
- `ApplyResult` keeps the Rust-only `reason` note from `applied`/`skipped` and adds Python's full field set; `ok()` now requires `errors` to be empty.
- `normalize_update_value` detects structured updates by JSON shape because Rust has no `UpdateValue` instance to type-check.
- The default `apply_update` rejects pending-change effects (only preview operators produce them) but still merges/appends state updates, keeping the Rust extension over Python's replace-only compatibility default.
- `LLMCallOperator` adds a real call path (Python's handle only stores prompts); `LLMCall` keeps Python's callable-wrapper role with an optimizer trace callback expressed as a boxed-future `Arc<dyn Fn>` so it stays object-safe and testable.
- `MemoryCallOperator` is wired to the real `LongTermMemory::recall` through a local `MemoryRecall` seam, so the memory crate stays untouched and tests can fake recall; `max_retries` retries only failures, synchronously with no delay.
- `SkillCallOperator` tunes selection rather than Python's skill-experience staging; `SkillExperienceOperator` ports the staging handle separately, so the two Python names that alias one class are distinct Rust types with distinct jobs.
- Pending changes stay process-local and are not serialized: Python keeps them in `ExperienceManager._pending_approval_snapshots`, and this crate exposes the same lifecycle through `PendingChangeStore` over an injected `SkillExperienceStore` sink.
- `PendingChangeStore::new` starts without a sink so a host decides where records go; `with_store`/`set_store` install or swap it, and a missing sink is `Error::Config` rather than a silent success.

## Notes for Agents

- A successful `set_parameter` fires the callback exactly once with the target's canonical stored value; `load_state` fires once per field present in the state object, including frozen ones.
- `load_state` bypasses freeze markers; `set_parameter` does not.
- For `ToolCallOperator` the callback's value is always the whole description map, never a single entry, matching Python.
- `apply_merge` on a `Text` target with `UpdateValue::append` concatenates text; on a `TextList`/array target it concatenates items; on a JSON object it merges shallowly.
- `MemoryCallOperator::recall` takes the memory implementation as an argument; the operator itself stores no memory handle.
- `MetisContextEvolveOperator::preview_update` validates `summary`/`max_messages` only; any other target is returned as a non-applied result, not an error.
- `SkillExperienceOperator::preview_update` wraps `null` into `[null]` (Python-identical) and therefore reports `applied = true`; only an empty array yields `applied = false`.
- `SkillExperienceOperator::set_parameter` is notification-only and reports `applied = pending_change effect`; use `preview_update` plus `PendingChangeStore::stage` for the real lifecycle.
- `PendingChangeStore` records `metadata` aggregated from the preview's applied `apply_results` (later results win) because `LocalApplyPreview` has no metadata field.
- `PendingChange::records` always holds the *unwritten remainder*: after a partial approve/retry it shrinks to the tail, and the entry disappears once it reaches zero.
- `LocalApplyPreview.lifecycle_stage` is always `local_apply_completed`; the error case is a result claiming a different stage.
- `LLMCall::stream` takes owned inputs/history/tools (the returned stream is `'static`), while `invoke` borrows them.
- `LLMCall` does not log; the callback is the observation seam, and its error is treated as the call's error.

## Deferred Python Behavior

- Not ported: the ML-backed Metis store; `MetisContextEvolveOperator::learn` returns `Error::Unsupported`.
- Not ported: Python's operator-map `execute_updates` loop (operators keyed by id with legacy `None` values); `normalize_update_value`, `normalize_updates`, and `summarize_apply_results` are ported, but batch execution is left to the caller.
- Not ported: `approved_record_ids` subset approval in `commit_pending_change`; `PendingChangeStore::approve` writes the whole remaining tail.
- Not ported: `PendingChange`'s `subject_kind`, `is_shared_records`, `trajectory`, and `messages` snapshots; the Rust snapshot keeps the records, change type, metadata, sequence, and status.
- Not ported: Python's `HostFacingExperienceResult` / `ExperienceApplyResult` host-facing classes; `PendingChangeOutcome` and `LocalApplyPreview` cover the same status/count contract.
- Not ported: Python's `SkillExperienceOperator` alias `SkillCallOperator = SkillExperienceOperator`; the Rust `SkillCallOperator` is the selection handle instead.
- Not ported: the `agent_evolving` apply-path constants outside the operator protocol (`approve`/`reject`/`retry` actions, evolution target/subject kinds, patch actions and sections).

## Test Strategy

- Unit tests beside each module cover the seams in isolation: tunable metadata/freeze/callback firing, state round-trips, `apply_merge` semantics, rich `UpdateValue`/`ApplyResult` serde and defaults, protocol normalisation/summaries, `LLMCallOperator` render/substitution and a real `ScriptedModel` call, tool-description projection, memory off/on + parameter forwarding + retry budgets through a fake seam, skill selection/curation/caps, skill-experience preview rejection/acceptance and direct-compat notification, staging approve/reject/partial-retry, context policies and truncation, and preview-without-mutation.
- The legacy wrapper suite (11 tests) drives `ScriptedModel` through `invoke` and `stream`, asserting message order, tool forwarding, one optimizer trace per call, callback-error and model-error propagation, freeze no-ops, accessors, and default-prompt fallback.
- The skill-experience suite (7 tests) pins the computed operator id, the single tunable, both rejection paths, append/merge acceptance, list wrapping (single/empty/null), callback-once state immutability, and `apply_update` preview routing.
- The staging suite (13 tests) covers preview construction/filtering/lifecycle validation, deterministic ids and ordering, approve-all/empty/reject, partial failure with tail retention and tail-only retry, unknown-id `NotFound` for all three transitions, a missing sink (`Config`), and outcome serde.
- Everything is offline (`ScriptedModel`, in-memory sinks) and runs under `cargo test -p openjiuwen-agent --offline operator::`.
- The crate suite is 711 unit tests + 59 integration tests + 5 doc tests; `cargo clippy --all-targets --all-features -- -D warnings` and `cargo doc --no-deps` are clean.

## Routing Table

- `./base.rs` → operator traits, tunables, structured updates/results, `apply_merge`
- `./protocol.rs` → evolution literals, update normalisation, result summaries
- `./llm_call.rs` → `LLMCallOperator`, prompt rendering, model call
- `./tool_call.rs` → `ToolCallOperator`, description map and `ToolCard` projection
- `./memory_call.rs` → `MemoryCallOperator`, `MemoryRecall` seam, retry budget
- `./skill_call.rs` → `SkillCallOperator`, bounded skill selection
- `./skill_experience.rs` → `SkillExperienceOperator`, preview-only experience records
- `./staging.rs` → `LocalApplyPreview`, `SkillExperienceStore`, `PendingChangeStore` lifecycle
- `./legacy/` → `LLMCall`, optimizer trace callback (`legacy/CONTEXT.md`)
- `./context_evolve.rs` → `ContextEvolveOperator`, `MetisContextEvolveOperator`, `UpdatePolicy`
