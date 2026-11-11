# Workflow built-in components

## Intent

Built-in `Component` implementations that the workflow engine drives for control flow and leaf work.
Each file ports one Python `openjiuwen.core.workflow.components` module onto `crate::component::Component`.
The control-flow components (`start`, `end`, `branch`, `loop`) have no external dependencies and are offline-testable.

## API Surface

### `start.rs` — `StartComponent`

`StartComponent` is `Debug + Default + Clone` with `new()`, `component_type()` = `"start"`, and abilities `[Invoke]`.
`invoke` returns the resolved inputs unchanged; a start node is a pure pass-through.

### `end.rs` — `EndComponent`

`EndConfig { response_template: Option<String> }` is serde snake_case with a `responseTemplate` alias, `Default`, and `EndConfig::new(template)`.
`EndComponent` is `Debug + Clone + Default` with `new()`, `with_config(EndConfig)`, and `config() -> &EndConfig`; `component_type()` = `"end"`; abilities `[Invoke, Stream, Collect, Transform]`.
The response template is segmented once at construction: a segment starting with `{{` and ending with `}}` is a variable whose path is the inner text verbatim, everything else is a literal, and empty segments are dropped.
Flat-mode `invoke` (no template) returns `{"output": <inputs without null-valued keys>}`, a non-object as `{"output": <inputs>}`, and `null` as `{"output": {}}`.
Template-mode `invoke` returns `{"response": "<joined segments>"}`; missing and `null` variables contribute nothing and non-string values render as JSON text.
Template-mode `stream`/`transform` yield one `{"response": …}` frame per segment piece in template order: a literal is one frame, a resolved array is one frame per element carrying the raw value, any other non-null value is one raw frame, and `null`/missing yields no frame.
Flat-mode `stream`/`transform` yield one `{"output": {key: value}}` frame per input entry (nulls unfiltered), one `{"output": inputs}` frame for non-object inputs, and no frames for `null`.
Template-mode `collect` returns the same joined `{"response": "…"}` value as `invoke`.
Flat-mode `collect` returns `{"output": {"a.b[0].c": <leaf>, …}}`, a flattened path→value map with `null` leaves omitted and `null` inputs mapped to `{"output": {}}`.

### `branch.rs` — `BranchComponent`

`BranchComponent` is `Debug + Clone` with `new(BranchRouter)`, `router() -> &Arc<BranchRouter>`, and `router_adapter() -> Arc<dyn openjiuwen_graph::Router>`; `component_type()` = `"branch"`; abilities `[Invoke]`.
`invoke` always returns `{}` and never errors because a branch only selects successors.
`route(source)` delegates to `BranchRouter::route` and returns `Some(Result<Vec<String>>)`; the first matching branch wins and no match is error `101021`.
`BranchRouterAdapter` is `Debug + Clone`, implements `openjiuwen_graph::Router` with `name()` = `"branch"`, and routes from `RouterContext::state().as_value()`.

### `loop_.rs` — `LoopComponent`

`LoopType { Number, Array, Condition }` is serde snake_case, `Default = Number`, with `as_str()`/`Display` returning `"number"`, `"array"`, `"condition"`.
`LoopConfig { loop_type, loop_number: Option<u64>, loop_array: Option<String>, bool_expression: Option<String>, max_iterations: u64 }` is serde snake_case with defaults (`max_iterations = 1000`) and chainable `new`/`with_loop_number`/`with_loop_array`/`with_bool_expression`/`with_max_iterations` builders.
`LoopComponent` has a manual `Debug`, `new(LoopConfig, Vec<Arc<dyn Component>>)`, `config()`, and `body_len()`; `component_type()` = `"loop"`; abilities `[Invoke]`.
`invoke` accepts an object or `null` and returns `{"loop_results": [<per-iteration output>, …]}`.
The effective cap is `min(config.max_iterations, session.config().loop_number_max_limit)`, falling back to `config.max_iterations` when the session exposes no configuration.
Each iteration starts from the input object overlaid with `{"index": <u64>, "item": <array element or null>, "previous": <last output or null>}`.
`Number` requires `loop_number` and fails with `101040` `loop_number exceeds maximum limit {cap}` when it exceeds the cap.
`Array` takes `loop_array` as an input key or a whole-string `${...}` reference, fails with `101040` `loop_array is not an array` for a non-array, and applies the same exceeded-limit error for a longer array.
`Condition` parses `bool_expression` once, re-evaluates it before every iteration against that iteration's inputs, stops on the first `false`, and fails with the exceeded-limit error when it would run past the cap.
The body runs sequentially per iteration, each later component receives the previous component's output, and the last output becomes both the iteration result and the next `previous`; an empty body echoes the iteration inputs.

### `llm.rs` — `LlmComponent`

`LlmResponseFormat { Text, Markdown, Json }` is serde snake_case with `Default = Text`.
`LlmCompConfig` carries `model`, `model_id`, `request` (`ModelRequestConfig`), the two prompt templates, `template_content`, `response_format`, `output_config`, and `enable_history`; all fields are reachable through chainable builders.
`LlmComponent::new(LlmCompConfig) -> Result<Self>` returns `Error::Validation("llm component requires a model")` when no model is configured; `component_type()` = `"llm"`; abilities `[Invoke, Stream]`.
Prompt assembly: system-only yields `[system(rendered), user("")]`, user-only yields one rendered user message, both yield `[system, user]`, and otherwise `template_content` is rendered message by message (empty content falls back to one empty user message).
Rendering substitutes `{{var}}` placeholders from a map of the input object's top-level keys plus the whole input under `"inputs"`; single braces stay literal.
`invoke` returns `{<output_key>: <assistant text>}` where `output_key()` is the first key of `output_config` or `"result"`.
`stream` yields one `{<output_key>: <delta>}` frame per non-empty content delta and one wrapped `101003` error on provider failure; `response_format` instruction injection and `enable_history` history insertion are deferred.

### `tool.rs` — `ToolComponent`

`ToolComponent::new(Arc<dyn Tool>)` derives its card from `Tool::card`; `component_type()` = `"tool"`; abilities `[Invoke]`.
`invoke` builds a `ToolContext` (session id, metadata, cancellation) and always returns a batch result: `{"error_code": 0, "error_message": "", "data": <value>}` on success.
Tool failures are captured, never propagated: `{"error_code": 102000, "error_message": <msg>, "data": null}` for `Tool`/`Validation` errors and `182012` for any other error kind.

### `retrieval.rs` — `KnowledgeRetrievalComponent`

`KnowledgeRetrievalComponent::new(Arc<SimpleKnowledgeBase>)` defaults to the `"query"` input key and `RetrievalConfig::default()`; `with_retrieval_config`/`with_query_key` adjust it; `component_type()` = `"knowledge_retrieval"`; abilities `[Invoke]`.
`invoke` returns `{"query": <q>, "documents": [{"text", "score", "metadata"}]}`; a missing/blank query is `101102`, a retrieval failure is wrapped as `101100`.
`pub(crate) fn resolve_query` (object key, else whole string input, else `None`) is shared with the memory component.

### `memory.rs` — `MemoryRetrievalComponent` / `MemoryWriteComponent`

Local seams `MemoryRecall { async fn recall(&self, query, scope_id, user_id) -> Result<Vec<Value>> }` and `MemoryRecord { async fn record(&self, messages, scope_id, user_id) -> Result<Value> }`; adapters over the real long-term memory land in `openjiuwen-memory`.
`MemoryRetrievalComponent::new(Arc<dyn MemoryRecall>, scope_id, user_id)` returns `{"memories": [...]}`; a blank query is `101200`, a recall failure `101201`.
`MemoryWriteComponent::new(Arc<dyn MemoryRecord>, scope_id, user_id)` reads `"messages"` (`Vec<BaseMessage>`), requires a non-empty array (`101150`), and returns the record result verbatim; failures are `101151`.

## Constraints

- Components implement `crate::component::Component`; they never reimplement graph scheduling, state resolution, or stream framing.
- Errors use `openjiuwen_common::error`; user-data problems become coded workflow errors and never panics.
- Missing references resolve to `null`; the control-flow components keep running or omit the frame rather than failing.
- Every type is serde-serializable where it is public configuration (`EndConfig`, `LoopType`, `LoopConfig`), snake_case, and tolerant of unknown fields.
- Public items are documented; files stay around or below 1000 lines; tests are deterministic and offline.

## Routing Table

- `./start.rs` → `StartComponent`, the pass-through entry node
- `./end.rs` → `EndComponent`/`EndConfig`, the exit node with flat and template output modes
- `./branch.rs` → `BranchComponent`/`BranchRouterAdapter`, condition-driven routing to successor ids
- `./loop_.rs` → `LoopComponent`/`LoopConfig`/`LoopType`, bounded number/array/condition iteration over a body list
- `./llm.rs` → LLM prompt assembly and model invocation component
- `./tool.rs` → tool invocation component
- `./retrieval.rs` → knowledge-retrieval component
- `./memory.rs` → memory recall and write components

## Test Strategy

Unit tests live beside each module and use `WorkflowSession` plus a plain `ComponentContext`.
`start` covers identity and pass-through of objects, `null`, and scalars.
`end` covers template segmentation (literals, adjacent variables, unclosed braces), flat filtering and null/scalar inputs, batch JSON-text stringification, missing variables, frame order and array expansion, `transform` parity with `stream`, nested flattening, the `responseTemplate` alias, and abilities.
`branch` covers first-match delegation, the `101021` no-match error, the empty invoke output, and adapter naming/`Arc` sharing; the adapter's `RouterContext` path is exercised by engine integration tests because `RouterContext` has no public constructor.
`loop_` covers number/array/condition plans, reference-path arrays, `previous` propagation, empty and multi-component bodies, the `1001` and session-tightened caps, a zero cap, every missing parameter, non-object inputs, and `LoopConfig` JSON round-trip/defaults.

## Notes for Agents

- **Batch template values are JSON text in Rust.** `EndComponent::invoke`/`collect` render non-string values with `serde_json::to_string` (`3`, `[1,2]`, `{"a":1}`), where Python uses `str()`/`repr` (`3`, `[1, 2]`, `{'a': 1}`); strings are used verbatim in both. Frame payloads always carry raw JSON values.
- **No live interleaving.** The end node's template frames are produced segment-by-segment from buffered inputs; concurrent streaming producers are composed through buffered arrays, so text and order match Python for finite streams but two producers never interleave live.
- **`loop_results` is a Rust-side normalized shape.** Python's loop group projects iterations through an output schema callback into the enclosing component state; the Rust port returns the collected iteration outputs in one wrapper object instead.
- **Branch default targets are a Rust extension.** `BranchRouter::default_target` lets a `branch` route without a matching condition; Python's router raises `101021` instead.
- **Flat-mode `collect` drops `null` leaves** (matching Python's `extract_leaf_nodes` pruning of empty values) but keeps other falsy scalars such as `0`, `""`, and `false`.
- `loop_`'s cancellation check runs once per iteration; a cancelled run surfaces `Error::Cancelled` rather than a loop error.
- The body runs in-process: component errors propagate unchanged, while loop configuration, array-shape, and cap problems use `crate::error::loop_execution_error` (`101040`).
