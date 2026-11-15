# Workflow built-in components

## Intent

Built-in `Component` implementations that the workflow engine drives for control flow and leaf work.
Each module ports one Python `openjiuwen.core.workflow.components` area onto `crate::component::Component`.
The control-flow components (`start`, `end`, `branch`, `loop`, `loop_group`, `advanced_loop`, `loop_break`, `loop_set_variable`, `http_request`) have no runtime collaborators and are offline-testable.
The leaf components take their collaborators through constructors: a `Model`, an `Arc<dyn Tool>`, a `SimpleKnowledgeBase`, the memory trait seams, or (for ReAct) an injected `AgentRunner`.
The `questioner` and `intent_detection` components persist their state and conversation in the session lanes, not in the component.

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
`LoopConfig { loop_type, loop_number: Option<u64>, loop_array: Option<String>, bool_expression: Option<String>, max_iterations: u64 }` is serde snake_case with defaults (`max_iterations = 1000`) and chainable builders.
`LoopComponent` has a manual `Debug`, `new(LoopConfig, Vec<Arc<dyn Component>>)`, `config()`, and `body_len()`; `component_type()` = `"loop"`; abilities `[Invoke]`.
`invoke` accepts an object or `null` and returns `{"loop_results": [<per-iteration output>, …]}`.
The effective cap is `min(config.max_iterations, session.config().loop_number_max_limit)`, falling back to `config.max_iterations` when the session exposes no configuration.
Each iteration starts from the input object overlaid with `{"index": <u64>, "item": <array element or null>, "previous": <last output or null>}`.
`Number` requires `loop_number` and fails with `101040` `loop_number exceeds maximum limit {cap}` when it exceeds the cap.
`Array` takes `loop_array` as an input key or a whole-string `${...}` reference, fails with `101040` `loop_array is not an array` for a non-array, and applies the same exceeded-limit error for a longer array.
`Condition` parses `bool_expression` once, re-evaluates it before every iteration against that iteration's inputs, stops on the first `false`, and fails with the exceeded-limit error when it would run past the cap.
The body runs sequentially per iteration, each later component receives the previous component's output, and the last output becomes both the iteration result and the next `previous`; an empty body echoes the iteration inputs.
The plan, cap, iteration inputs and body driver come from `loop_driver.rs`, so a `loop_break` request stops the loop without recording the interrupted round; missing or contradictory configuration is rejected by `validate_config` (`101030`) before any run.

### `loop_driver.rs` — `LoopBound` and the shared iteration driver

`LoopBound { Number, Array, Condition, UntilBreak }` is serde snake_case (`Default = Number`, `always_true` accepted as an `UntilBreak` alias) with `as_str()`/`Display` returning `"number"`, `"array"`, `"condition"`, `"until_break"`, `needs_break()`, and a `From<LoopType>` conversion; it is the `loop_type` field of the three container configurations.
The driver is `pub(crate)` and owns what all three containers share: `plan` (bound → `IterationPlan` plus a parsed condition, failing `101040` on a bad bound), `effective_limit` (`min(configured limit, session.config().loop_number_max_limit)`), `object_inputs` (object or `null`, any other JSON is `101040`), `resolve_variables` (initial loop variables templated against the inputs), `iteration_inputs`/`overlay_variables`, `run_spec` (body execution, marker folding, break handling, cancellation), and `container_output`.
`validate_bound` is the run-free check (`101030`): a zero limit, a missing/zero `loop_number`, a count above the limit, a blank `loop_array`, or a missing/unparsable `bool_expression`; `no_terminator` is the same code for a loop nothing can end.
The container wrapper is `{"loop_results": [<round output>, …], "loop_variables": {<name>: <value>}, "loop_count": <completed rounds>, "broken": <bool>}`; an interrupted round is not recorded, so `loop_count` is always `loop_results.len()`.

### `loop_group.rs` — `LoopGroupComponent`

`LoopGroupConfig { loop_type, loop_number_max_limit, loop_number, loop_array, bool_expression, loop_variables }` is serde snake_case with defaults (`max_iterations` alias for `loop_number_max_limit` — default 1000, `loop_condition` alias for `bool_expression`) and chainable builders (`with_loop_type`, `with_loop_number`, `with_loop_array`, `with_bool_expression`/`with_loop_condition`, `with_max_iterations`, `with_loop_variables`/`with_variable`).
`LoopGroupComponent::new(LoopGroupConfig, Vec<Arc<dyn Component>>)` with manual `Debug`, `config()`, `body_len()`, `has_break_component()` (immediate body scan); `component_type()` = `"loop_group"`; abilities `[Invoke]`.
`invoke` reports the container wrapper; `loop_variables` are exposed to the body and to a closing condition on every round and updated by `LoopSetVariableComponent`; `until_break` ignores the count/array/condition bound and runs until a `loop_break` asks to stop; reaching the cap without one is `101040`.
`validate_config` runs `validate_bound` and rejects an `until_break` group with no break component in its body.

### `advanced_loop.rs` — `AdvancedLoopComponent`

`AdvancedLoopConfig { max_iterations, loop_number, loop_array, bool_expression, loop_variables }` is serde snake_case with defaults (`loop_number_max_limit` alias for `max_iterations`, `loop_condition` alias for `bool_expression`) and the same chainable builders; `bound()` picks the active kind (`loop_number` > `loop_array` > condition-only > `until_break`).
`AdvancedLoopComponent::new(AdvancedLoopConfig, Vec<Arc<dyn Component>>)` with manual `Debug`, `config()`, `body_len()`, `has_break_component()`; `component_type()` = `"advanced_loop"`; abilities `[Invoke]`.
It re-evaluates termination before **every** round — the cap, the count, the array, the condition and the body's break signal each end the loop, per-iteration variable updates land before the next decision, and reaching the cap with no other terminator is `101040`.
`validate_config` applies `validate_bound` and rejects an unterminated configuration (no count, array, or condition, and no break component).

### `loop_control.rs` — `LoopBreakComponent` / `LoopSetVariableComponent`

`LOOP_BREAK_MARKER` = `"__loop_break__"` and `LOOP_SET_VARIABLES_MARKER` = `"__loop_set_variables__"` are the data markers that replace Python's injected `LoopController` and sibling-node output writes.
`LoopBreakConfig { condition: Option<String> }` is serde snake_case; `LoopBreakComponent::new()`, `with_config(LoopBreakConfig)`, `with_condition(expr)`, `config()`, `should_break(inputs) -> Result<bool>`; `component_type()` = `"loop_break"`; abilities `[Invoke]`.
`invoke` returns `{"__loop_break__": <bool>}` — `true` when no condition is configured or the condition holds against the node inputs, `false` when a configured condition does not hold.
`validate_config` rejects a blank or unparsable break condition (`101030`).
`LoopSetVariableConfig { variable_mapping: BTreeMap<String, Value> }` is serde snake_case; `LoopSetVariableComponent::new(mapping)`, `with_config(..)`, `config()`, `mapping()`, `assignments(inputs) -> Result<Map<String, Value>>`; `component_type()` = `"loop_set_variable"`; abilities `[Invoke]`.
`invoke` returns `{"__loop_set_variables__": {<name>: <value>}}` with every value resolved against the node inputs and every name normalized (`${loop.x}`, `loop.x`, `${x}` and `x` all name `x`).
`validate_config` rejects an empty mapping and an invalid variable name (`101031`); standalone use returns the marker as the component's whole output.

### Build-time configuration validation

`Component::validate_config` (default `Ok(())`) is called by `crate::engine::validate` for every registered component while a workflow is compiled, so a component's own configuration problem is reported with its status code before any run instead of surfacing mid-run.
The loop components are its first users: `loop`, `loop_group` and `advanced_loop` report a missing, zero, or out-of-range bound and an unterminatable container (`101030`), and the loop-body components report an unparsable break condition (`101030`) and an empty or invalid variable mapping (`101031`).
The check runs for every component registered as a node, so a container's bad bound fails `Workflow::build` with that coded message joined into the same compile error as every other build problem; a body component's check is reached through its own `validate_config` because a loop body is a plain component list, not a registered node.

### `llm/` — `LlmComponent`

`mod.rs` holds the component, `format.rs` the prompt rewrite plus answer shaping, `messages.rs` the conversation assembly and turn recording.
`LlmResponseFormat { Text, Markdown, Json }` is serde snake_case with `Default = Text`.
`LlmCompConfig` carries `model`, `model_id`, `request` (`ModelRequestConfig`), the two prompt templates, `template_content`, `response_format`, `markdown_instruction`/`json_instruction`, `output_config`, and `enable_history`; all fields have chainable builders.
`LlmComponent::new(LlmCompConfig) -> Result<Self>` returns `Error::Validation("llm component requires a model")` when no model is configured; `component_type()` = `"llm"`; abilities `[Invoke, Stream]`.
Prompt assembly: system-only yields `[system(rendered), user("")]`, user-only yields one rendered user message, both yield `[system, user]`, and otherwise `template_content` is rendered message by message (empty content falls back to one empty user message).
Rendering substitutes `{{var}}` placeholders from a map of the input object's top-level keys plus the whole input under `"inputs"`; single braces stay literal.
`format_prompt` rewrites only the **last** user message for `Markdown`/`Json` using the verbatim `DEFAULT_MARKDOWN_INSTRUCTION` / `DEFAULT_JSON_INSTRUCTION` (`${query}`, `${json_schema}` substitution; overrides via `markdown_instruction`/`json_instruction`); `Text` leaves the conversation untouched.
`shape_output` shapes the answer from `output_config`: `Text`/`Markdown` require exactly one configured key (`101002`), `Json` strips a fenced block, parses, projects the configured keys and reports `101004` for invalid JSON or a missing required field.
`invoke` returns `{<output_key>: <shaped answer>}` where `output_key()` is the first key of `output_config` or `"result"`.
`stream` yields one `{<output_key>: <delta>}` frame per non-empty content delta; a JSON-format stream is one non-streaming `Model::invoke` call yielded as a single frame; provider failures are a wrapped `101003` error.
`enable_history` inserts `[system?] + history + [current prompt]` and, after a successful call, appends the sent user text and the raw assistant reply (a streamed run records the accumulated text once); with it off nothing is read from or written to the history store.
No provider-side `response_format` parameter is ever sent.

### `sub_workflow.rs` — `SubWorkflowComponent`

`SUB_WORKFLOW_COMPONENT` = `"sub_workflow"`; `new(&Workflow) -> Result<Self>` compiles the child, `from_compiled(CompiledWorkflow)` wraps an existing one and takes `workflow_max_nesting_depth` from its config, and `with_cache_stream(bool)` / `with_max_nesting_depth(u32)` tune it; accessors `cache_stream()`, `max_nesting_depth()`, `compiled()`, `stream_output()`.
Abilities are `[Invoke, Stream]`; `component_type()` = `"sub_workflow"`.
Each run gets a fresh child `WorkflowSession` whose parent is the calling session (config, stream hub, tracer and checkpointer shared; its own state store, so child and parent node ids never collide) and `depth + 1` run options.
`invoke` returns the child end output unchanged, so downstream references read `${sub.output.<key>}`.
`stream` forwards the child's `{"type","index","payload"}` frame wires and drops the child's terminal `workflow_final`; with `cache_stream = true` the frames are merged (strings concatenate, arrays extend, every other type is last-wins — Python raises `TypeError` on numbers) and a later `invoke` answers from the cache.
A child `INPUT_REQUIRED` re-publishes every interaction wire on the parent context and fails with the first one, so the parent run reports `INPUT_REQUIRED` with the same questions.
`depth >= max_nesting_depth` fails with `[100100]` "workflow nesting hierarchy is too big, must <= {max}".
Nested resume: `child_options` reads the answers already forwarded from the parent session component lane `sub_workflow_answers` (`NESTED_ANSWERS_KEY`, a JSON object of `id → [answer, …]`, a lone non-array read as a one-element queue), folds Python's dotted `"<sub node id>.<child id>"` ids onto the bare child id (dropping the prefixed key so it is not consumed twice), appends the answers still queued in the parent run (`ctx.queued_answers()`), records the merged map back, and passes it as `WorkflowRunOptions::answers` with `depth + 1`.
The child still starts from a fresh session, so its own checkpoint snapshot and component lanes are not resumed; only the answers are replayed.

### `http_request/` — `HttpRequestComponent`

`config.rs` ports Python's configuration tree: `HttpComponentConfig`, `HttpRequestParamConfig`, `HttpHeadersConfig` (map or JSON template string), `HttpRequestBodyConfig`, `HttpAuthConfig` with `HttpAuthType`/`HttpAuthIn`, `HttpResponseHandlingConfig` with `HttpResponseFormat` (`autodetect`/`json`/`text`/`binary`/`buffer`) and `HttpResponseMode` (`full`/`on-success`/`on-error`), `HttpAdvancedOptionsConfig`, `HttpRetryConfig` with `HttpBackoffType` (`fixed`/`exponential`/`linear`), `HttpRateLimitConfig` with `HttpRateLimitUnit`, and `HttpContentType`.
The module is composed by responsibility beside `mod.rs` (the component, input merge, request build and send/replay flow): `config.rs` the configuration tree, `template.rs` the `{{key}}`/`${path}` templating, `body.rs` the request bodies and `multipart.rs` the `multipart/form-data` encoder, `auth.rs` with `digest.rs` and `sigv4.rs` the authentication schemes, `response.rs` the response projection, and `rate_limit.rs` the sliding-window limiter; `CONTEXT.md` in this directory owns the per-submodule API and wire details.
`HttpRequestComponent::new(HttpComponentConfig)`; `config() -> &HttpComponentConfig`; `component_type()` = `"http_request"`; abilities `[Invoke]` only (it still implements `stream`/`collect`/`transform` as single-result frames) so a plain node is never driven as `Transform`.
Requests run through `openjiuwen_common::http::HttpClient`.
`invoke` merges `inputs.method`/`headers`/`query_parameters` over the configuration and a non-null `inputs.body` into a JSON body, then templates `{{key}}` over `url`, `method`, header values, string query-parameter values, `body.json_data` (string form) and `body.text_data`; embedded `${path}` references resolve against the node inputs first (a Rust extension; auth secrets are left untouched).
The output is `{statusCode, headers, body, url, ok}` with Python's success logic, `response_data_property` narrowing, and `response_mode` gating the object.
`response_handling.throw_on_http_error` (default `true`) turns a failed status into a coded `COMPONENT_HTTP_ERROR` (`[102002]`) naming the status, `ok`, URL and a body preview; when `false` the 4xx/5xx response is ordinary data with `ok = false`.
`advanced_options.max_body_length` (default 1 MiB, `0` = unlimited) rejects an encoded body over the limit before sending with `[102001]`, naming both sizes.
`rate_limit_config.enabled` installs one sliding-window limiter per component instance whose delay is awaited before each send, honouring the run's cancellation token; `advanced_options.use_stream` is accepted and ignored (responses are buffered).
Body kinds: JSON, text, URL-encoded form, binary (base64-decoded), and `multipart_form` (boundary, CRLF-framed parts, `filename`/`content_type` on file parts), all sent through `HttpClient`.
Authentication: `basic`/`bearer`/`api_key` headers, challenge-driven `digest` replay (MD5/MD5-sess/SHA-256/SHA-256-sess, `qop=auth` when offered) and pre-flight AWS `SigV4` signing; an unanswerable challenge is ordinary response data with `ok = false`.
`response_format = binary` base64-encodes the bytes, and `response_format = buffer` uses the lossless Node-style `{"type": "Buffer", "data": [..u8..], "length": n}` wire object.
Transport, timeout, cancellation, size and client-build failures become `[102000]`.
Retries are enabled by `retry_config.enabled` and run inside `HttpClient` (`fixed` caps at `retry_delay`, `exponential` at `retry_delay * 2^max_retries`, `linear` is approximated by `fixed`).
Registered as the builtin `"http_request"` factory.

### `questioner/` — `QuestionerComponent`

`config.rs` ports `QuestionerConfig` (`model`/`model_id`, `response_type` — only `"reply_directly"`, `question_content`, `extract_fields_from_response`, `field_names`, `max_response` default 3, `with_chat_history`, `chat_history_max_rounds`, `extra_prompt_for_fields_extraction`, `example_content`, `accept_language`) and `FieldInfo` (`field_name`, `description`, `type`, `cn_field_name`, `required`, `default_value`); `prompts.rs` holds the verbatim zh/en extraction templates plus the continue-ask wording.
The component is `Debug` with `new(QuestionerConfig) -> Result<Self>`, `config()`, `component_type()` = `"questioner"`, abilities `[Invoke]`.
It runs a `start`/`user_interact`/`end` state machine persisted in the session component lane under `questioner_state` and reset after a completed round.
`question_content` is rendered with `{{field}}` placeholders over the node inputs (an unresolved placeholder renders `"None"`).
Field extraction calls the model, strips a code fence, parses the JSON object, filters it to the configured field names, drops empty values, converts `FieldInfo.type` (`string`/`integer`/`number`/`boolean`), and fills `default_value` for the rest.
Missing required fields produce a continue-ask question in the configured language; exceeding `max_response` is `[101074]`.
An open question returns the interrupt error from `ComponentContext::interact`, so the run halts as `INPUT_REQUIRED` and `resume` re-runs the component with the seeded answers.
Chat history is opt-in (`with_chat_history`, default off) with `chat_history_max_rounds` (default 5): each round records the user turn, builds the window from the session `__messages__` lane through `ctx.history()` (the last N rounds, a round starting at a user turn), renders it as `dialogue_history` in the extraction prompt (one `role<separator>content` line per message, `": "` in English and `"："` otherwise), appends the current query when the window is empty or ends with an assistant turn, and records the assistant turn afterwards (the follow-up question, or the extracted fields as compact JSON when the round completes).
The output is `{user_response, question, <fields>}` with empty `user_response`/`question` omitted.
Local codes are `101070..=101076`.

### `intent_detection.rs` (+ `intent_detection/`) — `IntentDetectionComponent` / `IntentRouter`

`IntentDetectionCompConfig` carries `model_id`, `category_name_list`, `user_prompt`, `example_content: Vec<String>`, `enable_history`, `chat_history_max_turn` (default 3), `accept_language`, plus the non-serde `model` and `keywords`.
The category model matches Python: index 0 is the default class (`Category0`/`分类0`, displayed `默认意图`/`Default intent`) followed by one class per configured name.
The model path asks the model and post-processes the outermost `{…}` block: the `class` field is sanitised and matched against `Category\d+`/`分类\d+`, with Python's unparsable/missing-class/out-of-list fallbacks to the default class plus a language-specific `reason`.
The rule path (no model configured) scores each category's keywords case-insensitively, counts distinct hits, breaks ties by the lowest class id, and reports `no keyword matched` for the default class; a category without keywords uses its display name.
The output is `{classification_id, category_name, reason}` with an empty `reason` omitted; abilities `[Invoke]`; local codes `101050..=101062` in `intent_detection/error.rs`.
`IntentRouter::new(component_id, Vec<(i64, Vec<String>)>, fallback)` implements `openjiuwen_graph::Router` for `add_conditional_connection`: it reads `classification_id` from the node output, routes the matching target list, and falls back for an unknown or absent id (an empty target list legally ends the run).
The component deliberately does **not** implement `Component::route`, because dispatch would then suppress the state write the router needs.
History is the session `__messages__` lane.

### `react_agent.rs` — `ReActAgentComponent` / `AgentRunner` / `ModelReActRunner`

`AgentRunner` is the async seam (`async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value>`) that `openjiuwen-agent` and `openjiuwen-harness` adapt to; this crate never depends on them.
`ReActAgentComponent::new(Arc<dyn AgentRunner>)`, `runner()`, and `from_model(ReActAgentConfig) -> Result<Self>`; `component_type()` = `"react_agent"`, abilities `[Invoke]` (one node frame per run).
`invoke` forwards the resolved inputs to the runner unchanged and normalizes the result to `{"output", "result_type"}`: `answer` for an ordinary value (an object's `output` key is unwrapped), `error` for a runner failure (`"Error in ReAct execution: <reason>"`, logged, Python parity, run completes), and `interrupt` for a workflow interrupt (the question becomes output data, a documented deviation).
`ReActAgentConfig` carries `max_iterations` (5), `on_max_iterations` (`error`/`final_answer`/`return_observations`), the system/user prompt templates, and the non-serde `model`, `tools` and `request`; `validate()` rejects bad policy/prompt configuration.
`ModelReActRunner` is the bundled runner: a bounded tool-calling loop over `Model` + `Arc<dyn Tool>` that feeds each tool result back as one tool message, runs tool calls sequentially, and reports `final_answer` as the last assistant text and `return_observations` as a plain observation array.

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
- Component-specific error codes live beside the component, not in `crate::error`.
- Missing references resolve to `null`; the control-flow components keep running or omit the frame rather than failing.
- Every public configuration type is serde-serializable, snake_case, and tolerant of unknown fields; runtime handles (`Model`, `Tool`, `AgentRunner`) are `#[serde(skip)]`.
- Public items are documented; files stay around or below 1000 lines; tests are deterministic and offline (loopback `wiremock` for HTTP).

## Routing Table

- `./start.rs` → `StartComponent`, the pass-through entry node
- `./end.rs` → `EndComponent`/`EndConfig`, the exit node with flat and template output modes
- `./branch.rs` → `BranchComponent`/`BranchRouterAdapter`, condition-driven routing to successor ids
- `./loop_.rs` → `LoopComponent`/`LoopConfig`/`LoopType`, bounded number/array/condition iteration over a body list
- `./loop_driver.rs` → `LoopBound` and the shared iteration driver (plan, cap, iteration inputs, variable/break folding, container wrapper, build-time bound validation)
- `./loop_group.rs` → `LoopGroupComponent`/`LoopGroupConfig`, variables plus the unbounded `until_break` container
- `./advanced_loop.rs` → `AdvancedLoopComponent`/`AdvancedLoopConfig`, per-iteration termination re-evaluation
- `./loop_control.rs` → `LoopBreakComponent`/`LoopBreakConfig` and `LoopSetVariableComponent`/`LoopSetVariableConfig` with their output markers
- `./llm/` → `mod.rs` (component + abilities + invoke/stream), `format.rs` (instruction rewrite, schema generation, output shaping), `messages.rs` (prompt assembly + history insertion)
- `./sub_workflow.rs` → `SubWorkflowComponent`, a compiled child workflow as one node
- `./http_request/` → `HttpRequestComponent` and its Python configuration tree (`CONTEXT.md` in that directory details every submodule)
- `./questioner/` → `QuestionerComponent`, `QuestionerConfig`/`FieldInfo`, the zh/en extraction prompts
- `./intent_detection.rs` → `IntentDetectionComponent`/`IntentDetectionCompConfig`/`IntentRouter`; `./intent_detection/` → local codes and the inline test suite
- `./react_agent.rs` → `ReActAgentComponent`, the `AgentRunner` seam, `ModelReActRunner`
- `./tool.rs` → tool invocation component
- `./retrieval.rs` → knowledge-retrieval component
- `./memory.rs` → memory recall and write components

## Test Strategy

Unit tests live beside each module and use `WorkflowSession` plus a plain `ComponentContext`.
`start` covers identity and pass-through of objects, `null`, and scalars.
`end` covers template segmentation (literals, adjacent variables, unclosed braces), flat filtering and null/scalar inputs, batch JSON-text stringification, missing variables, frame order and array expansion, `transform` parity with `stream`, nested flattening, the `responseTemplate` alias, and abilities.
`branch` covers first-match delegation, the `101021` no-match error, the empty invoke output, and adapter naming/`Arc` sharing; the adapter's `RouterContext` path is exercised by engine integration tests because `RouterContext` has no public constructor.
`loop_` covers number/array/condition plans, reference-path arrays, `previous` propagation, empty and multi-component bodies, the `1001` and session-tightened caps, a zero cap, every missing parameter, non-object inputs, break and loop-variable markers, and `LoopConfig` JSON round-trip/defaults.
`loop_driver` covers the bound plan for every `LoopBound`, `validate_bound`'s missing/zero/out-of-range/blank/unparsable cases, `no_terminator`, `object_inputs`, variable resolution, `iteration_inputs`/`overlay_variables`, marker folding, and the `container_output` wrapper.
`loop_group` covers the number/array/condition/`until_break` bounds, variable accumulation across rounds, break handling (including the first round), the session-tightened cap, non-object inputs, `validate_config`'s three rejections, `has_break_component`, alias parsing and the JSON round-trip.
`advanced_loop` covers condition-decided termination, the cap, variable updates before the next decision, break handling, the unterminated configuration, non-object inputs, `validate_config`, and the JSON round-trip.
`loop_control` covers `should_break` gating, the markers' exact shape, template resolution and name normalization, marker stripping, empty/invalid mappings, and every `validate_config` rejection.
`llm/format` covers instruction rewriting, `generate_json_schema`, fence stripping, required-field errors, and output projection; `llm/messages` covers template precedence, history splicing, and turn recording.
`start`, `end`, `branch`, `loop_`, `loop_driver`, `loop_group`, `advanced_loop`, `loop_control`, `llm/mod`, `llm/format`, `llm/messages`, `tool`, `retrieval`, `memory`, `http_request/mod`, `http_request/config`, `intent_detection`, `react_agent` and `sub_workflow` (card identity, nesting default, opt-in cache, cache merging, interaction-wire parsing) carry inline suites.
`questioner` has no inline suite: its state machine, `question_content` rendering and chat-history window are covered by `../tests/questioner_component.rs`.
The `http_request` submodules carry their own inline suites listed in `http_request/CONTEXT.md`'s test strategy.
Crate-level suites in `../tests/` (listed in `tests/CONTEXT.md`) exercise the same components inside real workflows.

## Notes for Agents

- **Batch template values are JSON text in Rust.** `EndComponent::invoke`/`collect` render non-string values with `serde_json::to_string` (`3`, `[1,2]`, `{"a":1}`), where Python uses `str()`/`repr` (`3`, `[1, 2]`, `{'a': 1}`); strings are used verbatim in both. Frame payloads always carry raw JSON values.
- **No live interleaving.** The end node's template frames are produced segment-by-segment from buffered inputs; concurrent streaming producers are composed through buffered arrays, so text and order match Python for finite streams but two producers never interleave live.
- **`loop_results` is a Rust-side normalized shape.** Python's loop group projects iterations through an output schema callback into the enclosing component state; the Rust port returns the collected iteration outputs in one wrapper object instead.
- **Branch default targets are a Rust extension.** `BranchRouter::default_target` lets a `branch` route without a matching condition; Python's router raises `101021` instead.
- **Flat-mode `collect` drops `null` leaves** (matching Python's `extract_leaf_nodes` pruning of empty values) but keeps other falsy scalars such as `0`, `""`, and `false`.
- `loop_`'s cancellation check runs once per iteration; a cancelled run surfaces `Error::Cancelled` rather than a loop error.
- The body runs in-process: component errors propagate unchanged, while loop configuration, array-shape, and cap problems use `crate::error::loop_execution_error` (`101040`).
- `SubWorkflowComponent` builds a fresh child `WorkflowSession` (shared checkpointer, own state store) for every run, so a parent `resume` replays the child from scratch and only the answers recorded under `sub_workflow_answers` carry over; child-side state such as questioner rounds, and any stored child snapshot, does not.
- `HttpRequestComponent` declares only `Invoke` on purpose: the engine's ability priority would otherwise drive a plain node as `Transform`.
- `HttpRequestComponent` deviations: digest and SigV4 authentication are implemented here while Python only declares the configuration; binary bodies/responses use base64 through JSON (`buffer` is the lossless form); JSON string bodies use `serde_json` instead of `ast.literal_eval`; Python's `response_handling.max_redirects` is not modelled; `advanced_options.disable_compression` and `use_stream` are accepted but not acted on; an API-key `in_location = body` is ignored, as in Python.
- The loop containers report the same documented wrapper (`loop_results`, `loop_variables`, `loop_count`, `broken`) and represent body intent as data (`__loop_break__`, `__loop_set_variables__`) instead of Python's injected `LoopController` and sibling-node writes; `LoopComponent` keeps its narrower `{"loop_results": …}` output.
- A loop body is a plain component list, so a body component's `validate_config` is not reached by `Workflow::build`; only a container registered as a node has its own configuration checked at build time.
- `QuestionerComponent` deviations: the model is injected via `with_model` (no `model_id` runner lookup), chat history lives in the session `__messages__` lane instead of Python's `ModelContext` window (so every history-enabled component on one session shares one conversation, entries are text only through `BaseMessage::text`, and a leading system turn is dropped when the window is trimmed), `chat_history_max_rounds` of `0` is clamped to one round (Python requires `> 0`), the completing assistant turn is compact JSON rather than `json.dumps` with separators, a `question_content` round writes no assistant turn, there is no `QuestionerExecutable` layer, and the state reset writes `null`s because the session lane deep-merges.
- `IntentDetectionComponent` deviations: the model is injected, history lives in `__messages__`, and JSON parsing is strict (`{'class': 'x'}` is rejected).
- `ReActAgentComponent` deviations: interrupts become `result_type: "interrupt"` data (the run completes), `on_max_iterations` policies return plain values instead of Python's rich dict, and the agent runtime (context engine, rails, skills, KV cache) is deferred to `AgentRunner` implementations.
