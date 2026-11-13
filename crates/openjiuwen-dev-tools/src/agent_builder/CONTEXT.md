# openjiuwen-dev-tools / agent_builder

## Intent

Ports `openjiuwen.dev_tools.agent_builder`: programmatic agent/workflow specs, the serialized platform DSL, the staged builder pipeline, and the kernel bridge.
The pipeline turns one caller request into a ready artifact (`LlmAgentDsl`, validated `WorkflowDsl`) or one specific clarifying question.
Every stage has a deterministic offline implementation; a `Model` is optional and only ever improves a stage, never gates it.
Internal structure is idiomatic Rust (serde models, typed errors, one async round trip per model-assisted stage), not a transliteration of Python's multi-agent state machine.

## API Surface

Core artifacts:
- `LlmAgentDsl` (`dsl.rs`) with `to_value`/`from_value`/`to_json_string`/`from_json_str`, plus blocks `DslConfigs`, `DslModel`, `DslModelInfo`, `PluginRef`, `WorkflowRef`, `DslDependencies`.
- `WorkflowDsl` (`workflow_dsl.rs`) with `DslNode`, `DslEdge`, `validate()`, `node(id)`, `to_json_string`, `from_json_str`.
- `LlmAgentSpec` and `WorkflowSpec` (`spec.rs`) with `to_dsl`, `to_json_string`, `from_json_str`, and `with_*` builders.
- `LlmAgentDsl::{to_agent_card, to_llm_agent_config, build_llm_agent}` (`bridge.rs`) map a document into `openjiuwen_agent::{AgentCard, LlmAgentConfig, LlmAgent}`.
- `HistoryManager` and `DialogueMessage` (`history.rs`) with `new(max)`, `add_message`, `add_user_message`, `add_assistant_message`, `get_history`, `get_latest_k(k)`, `len`, `is_empty`, `clear`, `max_history_size`; `DEFAULT_MAX_HISTORY_SIZE` is `20`.
- `BuildState` (`initial`/`clarifying`/`requesting`/`processing`/`completed`), `BuildStatus`, `BuildOutput`, `BuildResult`, and `map_state_to_status` (`outcome.rs`).
- `BuildResult::to_json` emits `status`, `session_id`, `agent_type`, and exactly one of `dsl`, `workflow`, `response`; `BuildOutput` serializes externally tagged, so its JSON is exactly `{"dsl": {...}}`, `{"workflow": {...}}`, or `{"response": "..."}`.

Builder entry points (`AgentBuilder`, `builder.rs`):
- `new(Arc<dyn Model>)`, `build_agent`, `build_agent_at`, `build_with_pipeline`, `history(session_id, k)`, `clear_session`, `status`, `model`.
- `AgentType` (`llm_agent`/`workflow`) and `now_millis`.
- `build_agent`/`build_agent_at` keep the legacy single round trip: one fenced JSON answer becomes the artifact, anything else becomes `BuildOutput::Response(text)` with state `Clarifying`.
- `build_with_pipeline` runs the staged pipeline; only a blank query errors.

Pipeline stages (new modules):
- `clarifier.rs` — `Clarifier::{new, with_model(Arc<dyn Model>)}`, `async clarify(&self, query, &HistoryManager) -> Result<Clarification>`; `Clarification { ready, question, summary }`.
- `designer.rs` — `WorkflowDesigner::{new, with_model}`, `async design(query)`, `design_offline(query)`; `WorkflowDesign { name, description, nodes, edges }`, `DesignNode { id, kind, title, description, config }`, `DesignEdge { source, target, condition }`, `DesignNodeKind::{Start, Llm, Branch, Tool, End}` with `as_str`/`from_str`/`Display`/`FromStr`; `WorkflowDesign::{validate, to_workflow_dsl}`.
- `dl_generator.rs` — `DlGenerator::{new, with_model}`, `generate(&WorkflowDesign)`, `async generate_with_model(&WorkflowDesign)`, `from_spec(&WorkflowSpec)`.
- `dl_reflector.rs` — `DlReflector::{new, with_model}`, `reflect(&WorkflowDsl)`, `async reflect_with_model(&WorkflowDsl, instructions)`; `Reflection { passed, issues, repaired }`, `ReflectionIssue { kind, message, node_id }`, `ReflectionIssueKind::{DanglingEdge, DuplicateNodeId, MissingStart, MissingEnd, Cycle, InvalidNode, UnreachableNode, MissingField, InvalidExpression, MissingDefaultBranch, InvalidReference, MissingConditionExpression, UnknownNodeType}`.
- `DlReflector::validate_dl(&DlDocument) -> DlValidation` — deterministic Python-parity per-node-type DL validation (model-free, never errors), with `DlValidation { passed, issues }` and `DEFAULT_MAX_RETRIES = 3`.
- `dl_transformer/` — the Python `SimpleIR` DL document (`DlDocument`, `DlNode`, `DlParameters`, `DlVariable`, `DlCondition`, `DlNodeType`) plus `DlTransformer::{new, transform, transform_with_resources, transform_str}` producing `PlatformWorkflow { nodes, edges }` (`PlatformNode` with numeric type id/`meta.position`/`data`, `PlatformEdge` with `source_port_id`), `PluginResource`, and `PlatformWorkflow::{to_value, to_json_string, to_workflow_dsl}`.
- `cycle_checker.rs` — `find_cycle(&WorkflowDsl) -> Option<Vec<String>>`, `has_cycle(&WorkflowDsl) -> bool`.
- `mermaid.rs` — `render_mermaid(&WorkflowDsl) -> String`, `render_design(&WorkflowDesign) -> String`, both emitting Python's `graph TD` grammar through one renderer.
- `utils.rs` — the deterministic `agent_builder/utils/utils.py` helpers: `extract_json_from_text`, `format_dialog_history`/`format_dialog_history_with`, `merge_dict_lists(existing, new_items, unique_key)`, `validate_session_id`/`is_valid_session_id`.
- `progress.rs` — the `agent_builder/utils/progress.py` weights: `ProgressStage` (15 stages, `as_str`/`Display`, serde snake_case), `LLM_AGENT_STAGE_WEIGHTS` (6), `WORKFLOW_STAGE_WEIGHTS` (9), `progress_for(agent_type, stage) -> u8`, and `BuildStatus::progress(stage)` as a derived view.
- `mod.rs` re-exports every type above plus `find_cycle`, `has_cycle`, `render_mermaid`, `render_design`, the `utils` helpers, the `progress` items, the DL model/transformer types, and `DlValidation`/`DEFAULT_MAX_RETRIES`.

## Pipeline (`build_with_pipeline`)

Sessions behave exactly like `build_agent_at`: a blank query is rejected before any session or model call, the session is created on first use with `HistoryManager::new(DEFAULT_MAX_HISTORY_SIZE)`, `agent_type` is set, the user turn is appended, and the state becomes `Processing`.
The clarifier receives a clone of the history taken *before* the user turn is appended, so the query is not duplicated in its prompt.
Stage 1 is always `Clarifier::with_model`:
- Not ready → `BuildOutput::Response` with the question, else the summary, else `DEFAULT_CLARIFICATION_QUESTION` ("Please describe what the agent should accomplish and what output it should produce."), state `Clarifying`.
- Ready + `AgentType::Workflow` → `WorkflowDesigner::design` (model with offline fallback) → `DlGenerator::generate_with_model` (model with deterministic fallback) → `DlReflector::reflect` (rule-based repair; a reflection error falls back to `DlGenerator::generate`), output `Workflow(repaired)`, state `Completed`.
- Ready + `AgentType::LlmAgent` → one model call with `LLM_AGENT_SYSTEM_PROMPT` over the retained turns, `extract_json_fence` + `LlmAgentDsl::from_value`, then `repair_llm_dsl`; a failed call, a choice-less answer, or an unparseable answer becomes `deterministic_agent_dsl(query, timestamp_ms)`, output `Dsl(dsl)`, state `Completed`.
The assistant turn records `dsl.to_json_string()`, `workflow.to_json_string()`, or the clarification text, and the final state is stored, so `status().resource_count` grows by two per call.
A model-assisted ready workflow uses three model calls (clarifier, designer, generator); a ready LLM agent uses two (clarifier, generation); a clarification uses one. `reflect_with_model` is not part of the pipeline — model refinement stays caller-driven.

Deterministic LLM-agent repair (`repair_llm_dsl`): blank `agent_id` → `UUIDv4`; blank `agent_type` → `react`; blank `name` → first 6 words (or first 24 chars for CJK-only text, fallback `Agent`); blank `description` → trimmed query truncated to 160 chars (fallback the name); blank `configs.system_prompt` → `"You are an assistant built for the following request: {query}"`; zero `create_time`/`update_time` → the supplied `timestamp_ms`.
`deterministic_agent_dsl` builds the same document through `LlmAgentSpec::new(...).to_dsl(uuid, timestamp_ms)`, so two calls with the same query and timestamp differ only in the generated id.

## Stage Contracts

### Clarifier (offline heuristic)

The model path sends one system prompt plus one message per retained turn, then the query as the final user message, and expects one fenced ```json `{"ready", "question", "summary"}`. A model error, missing choices, no fence, or a contract mismatch logs `tracing::debug!` and falls back to the heuristic — it never errors.
Offline context = every retained `user` turn (oldest first, `\n`-joined) + the query, lowercased for keyword matching. Four signals: goal, input (or a `${...}` placeholder), output, tools. `ready = goal && output`. When not ready, the question names the first missing signal in the order goal → input → output → tools; `summary` lists the detected signals as `"goal, input, output, tools"`. A blank query is `Error::validation`, before any model call.

### Workflow designer (offline synthesis)

`design_offline` splits the query on sentence terminators (`.;。；\n!?！？`), sequence markers ("then", "after that", "next", "首先", "然后", "接着", "最后", "，然后", "，再", case-insensitive), and numbered list markers, strips leading list markers, drops empties/duplicates, and caps at `MAX_STEPS = 4` (single-step fallback when nothing survives).
It emits `start` (id `start`, `inputs.query`) → one `llm` node per clause (`llm_1..llm_N`, deterministic title from the first 8 words or 20 CJK chars, config carries `instruction`, `inputs.query`, `outputs.result`) → an optional `branch` node (`branch`, config `{"conditions": ["default"]}`) when the lowercased query contains any of if/when/whether/condition/decision/branch/else/otherwise/如果/否则/当/是否/条件/分支 → `end` (id `end`). Edges connect in document order.
The workflow name is a 40-char slug of the first clause (CJK kept, other runs → `-`, fallback `workflow`); the description is the query truncated to 160 chars.
`WorkflowDesign::validate` requires at least one node, unique ids, resolvable edges, exactly one start node, and at least one end node.
`WorkflowDesign::to_workflow_dsl` is the single design→DSL converter: kind → `node_type`, `config.inputs`/`config.outputs` objects become the node maps, `data` is the whole node config, then the produced document is validated.
The model path asks for one fenced JSON design, parses `DesignNodeKind` case-insensitively and alias-tolerantly (`Output`→End, `Questioner`/`LLM`→Llm, `Plugin`/`Code`→Tool, `IntentDetection`/`Branch`→Branch), validates, and falls back to `design_offline` on any model/parse/validation failure.

### DL generator

`generate` is `design.to_workflow_dsl()` (validating). `from_spec` is `spec.to_dsl()` + `validate()`. `generate_with_model` computes the deterministic baseline first (an invalid design errors before any model call), then asks the model for one fenced workflow DSL document and accepts it only when `WorkflowDsl::validate` passes; any model failure or invalid document falls back to the baseline.

### DL reflector (repair rules)

`reflect` clones the document and repairs in this order, recording one `ReflectionIssue` per repair: zero nodes → `start`→`end` chain (`MissingStart` + `MissingEnd`); blank ids → `node_<index>`, blank titles → the id (`InvalidNode`); duplicate ids → later copies renamed `<id>_2`, `<id>_3`, … (`DuplicateNodeId`); edges with unknown endpoints dropped (`DanglingEdge`); missing start inserted at the front and connected to the first node (`MissingStart`); missing end appended and every sink connected (`MissingEnd`); unreachable nodes connected from the nearest earlier reachable node in document order (`UnreachableNode`); cycles broken by removing the back edge `path[len-2] -> path[len-1]` from `find_cycle`'s path (`Cycle`, message includes the path). The repaired document is then validated.
`passed = issues.is_empty()` and an already-valid document is returned unchanged. Repairs never mutate the input.
`reflect_with_model` runs `reflect` first, then accepts a model revision only when it parses, validates, and keeps the same node-id set; otherwise it returns the baseline and appends an `InvalidNode` issue `model revision rejected: <reason>` (the generic rejection marker). Model problems never surface as errors.

### DL reflector (per-node-type validation)

`validate_dl` is a separate, model-free validator over the Python-shaped `DlDocument`; it never errors and never mutates, and `passed == issues.is_empty()`.
It walks nodes in document order, accumulating each node's declared outputs as `"{node_id}.{name}"` so references may only use outputs declared at or before that point.
Basic checks: blank `id`/`type`/`description` → `MissingField`, repeated id → `DuplicateNodeId`, a type outside the nine wire names → `UnknownNodeType` (the node is then skipped), and a `next`/`conditions[].next` naming no node → `DanglingEdge`.
Per type: `Start` needs an output `query` described `用户输入` and a `next`; `LLM` needs non-empty `inputs`/`outputs`, non-blank `configs.system_prompt`/`configs.user_prompt`, and a `next`; `Questioner`/`Plugin`/`Code` need non-empty `inputs`/`outputs`, their `configs.prompt`/`configs.tool_id`/`configs.code`, and a `next`; `Output` needs inputs, `configs.template`, and a `next`; `End` needs inputs and `configs.template`.
`IntentDetection` needs inputs, `configs.prompt`, non-empty conditions, branch/description/next per condition, and, per non-default condition, an expression holding `${<node_id>.rawOutput}` and `contain`, plus a mandatory `default` condition.
`Branch` needs non-empty conditions, branch/description/next per condition, exactly one of `expression`/`expressions`, the operator as the second whitespace token from `{eq, not_eq, contain, not_contain, longer_than, longer_than_or_eq, short_than, short_than_or_eq, is_empty, is_not_empty}`, every `${…}` variable already declared, and a mandatory `default` condition.
Operator/reference/expression-shape problems are `InvalidExpression`/`InvalidReference`/`MissingConditionExpression`; absent fields are `MissingField`; a missing fallback is `MissingDefaultBranch`.
`DEFAULT_MAX_RETRIES = 3` is Python's repair-loop bound, exposed for callers that drive their own generate/validate loop; neither reflector entry point loops.

### Cycle checker

`find_cycle` is a three-colour DFS over edges in document order from roots in node order, ignoring endpoints that are not existing nodes (structural errors belong to `validate`). The result starts and ends at the repeated node (`["a","b","c","a"]`, self-loop `["a","a"]`) and is deterministic. `has_cycle` is `find_cycle(..).is_some()`.

### Mermaid

Both renderers feed one renderer, so `render_design` is byte-identical to `render_mermaid` on the equivalent DSL graph.
Output is Python's `simpleir_to_mermaid` grammar: the literal header `graph TD`, then one two-space-indented node line per node in document order, then one edge line per edge in edge-array order, with no trailing newline.
A node line is `  {id}[Node{id}: {label}]`; the label comes from the node `description` (never `title`) and falls back to the node id when the description is blank; only backtick and `"` are escaped, both to `'`.
An edge is `  {src} --> {dst}`; it becomes `  {src} -- {label} --> {dst}` only when the source node has more than one outgoing edge and the condition is non-blank; the label is the escaped condition (no `-->|…|` form, no shape-delimiter escaping).
Ids keep ASCII alphanumerics and `_`, map every other character to `_`, use `node_<index>` for an empty id, prefix `n_` when digit-leading, and resolve collisions with `_2`, `_3`, … in document order; edges follow the rendered ids.

### DL transformer (DL → platform DSL)

`DlTransformer` converts a validated `DlDocument` into a `PlatformWorkflow` deterministically in document order: `type` is the numeric platform id (`Start` 1, `End` 2, `LLM` 3, `Branch` 4, `IntentDetection` 6, `Questioner` 7, `Output` 9, `Code` 10, `Plugin` 19; an unknown type is a validation error), `meta.position` starts at `(0,0)` and advances by 20 on both axes per node, and `data.title` is the node description.
Inputs become `constant` variables (`schema.type` default `"string"`, `extra.index` per node starting at 0) or `ref` variables whose `content` is `[node, *_of_-reversed name]`; outputs are nested `object` schemas built by reversing `_of_` name segments, each leaf carrying `extra.index`.
Per-type rules: `Start`/`Code` make every output required; `LLM` derives `output_format` (multi-output text/markdown → json, single unknown → text) and carries `systemPrompt`/`prompt` templates, the fixed model config, and `response_format`; `Questioner` duplicates its prompt into `system_prompt`, sets `history_enable:false`/`max_response:3`, adds `user_response` (always) and `output` (when another output exists), requires everything but `output`, and injects `node_start.query` when it has no inputs; `Plugin` adds `error_code`/`error_message`/`data` defaults and a `plugin_param` (`pluginVersion` defaults to `draft`); `Code` carries `language:"python"` and the fixed `exception_config`; `Output` carries the `template` content; `Branch` emits `branches` with the operator/logic maps and per-condition `source_port_id`; `IntentDetection` builds deterministic intent ids and ports (`"0"` for default).
Edges come from `next` (unconditional, no port) or from conditions/intents (one per branch, `source_port_id` = branch or intent id); `Code` uses port `"0"`.
`PlatformWorkflow::to_value`/`to_json_string` apply Python's recursive null-dropping (nulls in objects and arrays disappear; `0`, `""`, and `false` survive), and `to_workflow_dsl` adapts the document to the simplified `WorkflowDsl` and validates it.
Every invalid input (unknown type, missing required config, malformed reference/operator) is `Error::validation` naming the node and field; no id is generated from a uuid or the clock.

## DSL Shape

The DSL is a typed subset of Python's document: agent identity, `configs`, `model`, plugin/workflow references, dependency id lists, and the two epoch-millisecond timestamps.
Python-only fields survive in `LlmAgentDsl::extra` (`#[serde(flatten)]`, a `BTreeMap`), notably `model_provider` and `constraints`, and are written back unchanged.
`DslConfigs::opening_remarks` is nested inside `configs` here, while Python keeps it at the document top level; the wire name is unchanged.
Unknown keys never make parsing fail, and a document without `agent_type` defaults to `"react"`.
A workflow document uses `nodes` (`id`, `type`, `title`, `description`, `inputs`, `outputs`, `data`) and `edges` (`source`, `target`, `condition`), with `type` keeping its Python wire name; `validate` requires at least one node, unique node ids, and resolvable edge endpoints, and messages name the offending id.

## DL Document Shape

The DL document is Python's `SimpleIR`: a bare JSON array of nodes (`DlDocument` serializes transparently), each `{id, type, description, next?, parameters}` with `parameters = {inputs, outputs, configs, conditions}`.
`type` keeps Python's capitalized wire names (`Start`, `End`, `LLM`, `Branch`, `IntentDetection`, `Questioner`, `Output`, `Code`, `Plugin`); `DlNodeType` matches them case-insensitively and exposes the numeric platform id, with no `Questioner`/`LLM` aliasing.
`inputs`/`outputs` are lists of `{name, value?, type?, description?, items?, properties?, required?}`, `configs` is a free-form string-keyed object, and `conditions` are `{branch, description, next?, expression?, expressions?, operator?, intent_id?}`.
Absent optional keys stay absent on serialization, and unknown keys survive through flattened `extra` maps, so a document round-trips without loss.
`DlDocument::validate` checks only structure: at least one node, non-blank unique ids, and resolvable `next`/condition references; node types are checked by `DlNodeType::from_wire` (and therefore by `validate_dl` and `DlTransformer`).

## Behavior of build_agent / build_agent_at

`build_agent` is `build_agent_at` with `now_millis()`; both keep session state in the builder keyed by the caller's session id.
A blank query is rejected as `Error::Validation` before any session exists, so `status`/`history` stay empty and the model is never called.
On the first call the session is created, the user turn is appended, and the stored state becomes `Processing`.
The model is called exactly once: one system message from the artifact prompt followed by one `user`/`assistant` message per retained history turn.
Request configuration is `ModelRequestConfig::new(model.model_name())`; a model error propagates and leaves the session in `Processing` with the user turn recorded so the caller can retry.
A response without choices is `Error::Llm("model returned no choices")`.
The answer text is scanned with `util::extract_json_fence`, then parsed as `LlmAgentDsl` or `WorkflowDsl` (plus `validate()`) for the requested type; any failure becomes `BuildOutput::Response(text)`.
A successful LLM-agent parse fills an empty `agent_id` with a fresh UUIDv4 and zero timestamps with the supplied `timestamp_ms`.
Success records `Completed`; the fallback records `Clarifying`; either way the assistant text is appended to the session history.
`map_state_to_status` yields `"clarifying"` (Initial for LLM agents, Clarifying), `"requesting"` (Initial for workflows, Requesting), `"processing"`, and `"completed"`.

## Constraints

- Only this directory may be edited here; sibling crates, `src/lib.rs`, `Cargo.toml`, `src/util.rs`, `tests/`, and `docs/` are out of scope.
- Errors are `openjiuwen_common::error::{Error, Result}` only: invalid user input is `Error::validation`, model failures in the legacy entry points propagate unchanged, a choice-less response is `Error::llm`; pipeline stages absorb model problems and fall back.
- All public data types are serde (`rename_all = "snake_case"`, `default`, unknown-field tolerant) and keep Python's JSON field names; core models derive `schemars::JsonSchema`.
- Time is a parameter everywhere except `now_millis`, and randomness is used only for a missing/generated `agent_id`, so tests never depend on the wall clock or generated ids.
- No `unsafe`, no `unwrap`/`expect`/`panic!` in library code, no printing; diagnostics go through `tracing::debug`.
- Async-first: model-assisted entry points are `async fn` and compose inside an existing tokio runtime; no blocking calls, no environment reads, no `block_on`.
- Files stay under the ~1000-line baseline: `designer.rs` (810), `builder.rs` (720), `dl_reflector.rs` (694) with its checks in `dl_reflector/dl_checks.rs` (497), `dl_transformer/model.rs` (954), `convert.rs` (551), `convert/nodes.rs` (599), and `platform.rs` (320); implementation modules carry no tests, which live in `designer/tests.rs`, `builder/tests.rs`, `builder/pipeline_tests.rs`, `dl_reflector/tests.rs` (1342), and `dl_transformer/convert/tests.rs` (1648) — test suites are allowed to exceed the baseline.
- A moved test module keeps its path (`builder::pipeline_tests` stays `builder::pipeline_tests`), so test bodies move verbatim and only the crate's existing `tests/` suites observe the public API.

## Design Decisions

- The builder owns `Arc<dyn Model>` supplied by the caller; stage constructors take the same `Arc`, so no stage constructs a client or reads provider configuration, and `Clarifier`/`WorkflowDesigner`/`DlGenerator`/`DlReflector` remain usable without a model.
- Deterministic cores are public and model-free: the clarification heuristic, offline design synthesis, design→DSL conversion, rule-based reflection, cycle detection, and Mermaid rendering are ordinary functions testable without a model.
- One model round trip per model-assisted stage replaces Python's state machine: a parseable answer is used, anything else is replaced by the stage's deterministic result (pipeline) or returned as a clarifying/plain response (legacy builder).
- `BuildState` adds `Clarifying` and `Requesting`, which Python only models as status strings beside its state enum.
- Session state is a private `SessionState`, so the public surface cannot corrupt a session's history invariants.
- `AgentBuilder` derives `Clone` (sessions are copied) and implements `Debug` manually because `dyn Model` is not `Debug`; stage types implement `Debug` as `has_model: bool`.
- `BuildOutput` keeps `LlmAgentDsl` inline; the `large_enum_variant` lint is allowed locally because the document is the payload callers hand to the platform.
- The bridge is total and infallible for card and configuration: an absent provider falls back to `"openai"` and absent constraints to `ConstrainConfig::default()`; only `build_llm_agent` can fail.
- `HistoryManager::new(0)` uses the default capacity of `20`, while `get_latest_k(0)` returns no message.
- Workflow pipeline reflection is rule-based; model-driven refinement lives behind `DlReflector::reflect_with_model` so the pipeline stays predictable and its model-call count bounded.
- The DL layer is separate from the simplified `WorkflowDsl`: `DlDocument` mirrors Python's `SimpleIR`, `PlatformWorkflow` mirrors the platform conversion, and `WorkflowDsl` stays the design-stage document; `PlatformWorkflow::to_workflow_dsl` bridges the two for structural validation and kernel use.
- Per-node-type DL validation is a pure report (`DlValidation`) beside the repairing `reflect`, so the pipeline's graph repair keeps its `passed` semantics and Python's field checks stay model-free and non-mutating.
- Mermaid output follows Python's grammar byte-for-byte (`graph TD`, `Node{id}: {description}` labels, and `-- label -->` only for multi-successor sources) instead of a shape-per-kind diagram, so diagrams match the Python tool.
- The DL transformer is deterministic by construction: no uuid, clock, or resource fetch on the conversion path, so identical documents always produce identical platform JSON.

## Deferred

- Python's full build-state machine stages — intention detector, multi-turn LLM clarifier dialogue, agent designer, workflow designer's 3-pass (basic/branch/reflection) prompts, DL `refine` mode, and the reflection-evaluate repair loop that feeds errors back to the designer — are replaced by the single-pass stages above.
- Python's resource retrieval/preprocessing (plugins, knowledge bases, prompt templates from the platform) has no counterpart; documents carry ids only, and the DL generator performs no asset search.
- Python's LLM-judged design review, `run_eval_query.py` shell harness, and `main.py` CLI driver for the builder are not ported; the CLI lives in `openjiuwen-cli`.
- The builder pipeline keeps the design→`WorkflowDsl` path: `DlDocument`/`DlTransformer` are a separate public conversion layer, and `WorkflowDesign::to_workflow_dsl` still keeps string node types (`start`/`llm`/`branch`/`tool`/`end`) with `DesignNodeKind` aliases (`Questioner`→`Llm`, `Output`→`End`, `Plugin`/`Code`→`Tool`, `IntentDetection`→`Branch`).
- `validate_dl` does not port every Python `Reflector` sub-check: duplicate input/output names, input `value` placeholder rules (multiple placeholders, undeclared references), and the optional `available_variable_types` validation are absent; the port covers node/type/field presence, branch and intent expressions, the operator set, declared-output references, default branches, and dangling links.
- Python's repair loop is not driven: `DEFAULT_MAX_RETRIES` is exposed, but neither `reflect` nor `validate_dl` retries and no generate→validate→repair cycle feeds errors back to the designer.
- `DlTransformer` replaces Python's uuid generation with deterministic ids (`intent_<node>_<index>` when no `intent_id` is present) and reads the branch operator as the second whitespace token instead of Python's first-substring match (which misreads `not_eq`/`not_contain`).
- `transform_with_resources` takes caller-supplied `PluginResource`s; Python's platform plugin/knowledge/prompt-template retrieval and `collect_plugin` preprocessing are not ported.
- The Mermaid renderer does not port Python's `When condition met[...]` regex extraction: the edge label is the condition text escaped as-is.
- Python's `utils/utils.py` `safe_json_loads`, `deep_merge_dict`, and `load_json_file` are not ported (`serde_json` covers safe parsing at the call site, and file loading belongs to the caller); the deterministic text/dict helpers are ported in `utils.rs`.
- Python's `utils/progress.py` `ProgressReporter`/`BuildProgress`/`ProgressStep`/`ProgressManager` and the process-global `progress_manager` singleton are not ported — no mutable per-session progress state, callbacks, or wall-clock timestamps; the port keeps the stage vocabulary and the two `_calculate_progress` weight tables, and build progress is observed through `BuildStatus::progress`/`progress_for` plus `AgentBuilder::status`.
- Multi-turn refinement after `Completed` is a plain second call: no state-machine re-entry, no error-correction pass over a previously produced artifact.
- Any streaming variant of the pipeline or the stages is not implemented.

## Test Strategy

- Unit tests live beside the code: `dsl`, `workflow_dsl`, `spec`, `bridge`, `history`, `outcome`, `clarifier`, `cycle_checker`, `dl_generator`, `mermaid`, `progress`, and `utils` each own an inline `#[cfg(test)] mod tests`, while `designer`, `dl_reflector`, and `builder` declare `#[cfg(test)] mod tests;` resolving to the sibling `designer/tests.rs`, `dl_reflector/tests.rs`, and `builder/tests.rs`; `builder` additionally declares `#[cfg(test)] mod pipeline_tests;` resolving to `builder/pipeline_tests.rs`.
- A test module in a sibling file opens with `use super::*;`, so it reaches the parent module's private items exactly as the inline module did.
- `cycle_checker`: acyclic chain, diamond, self-loop, three-node cycle, cycle behind an isolated root, dangling-edge tolerance, empty document, determinism.
- `clarifier`: blank validation with zero calls, ready/missing-signal questions, history-carried signals, model JSON path with message-order assertions, garbage/error fallbacks, serde round-trip.
- `designer`: offline single/multi-step synthesis, the 4-step cap, branch detection, CJK input, name slugging, blank validation with zero calls, model path with aliases, invalid/garbage/error fallbacks, `DesignNodeKind` alias matrix, `to_workflow_dsl` mapping, all `validate` rejections.
- `dl_generator`: deterministic parity with `to_workflow_dsl`, `from_spec` valid/invalid, model success assertions, invalid/garbage/error fallbacks, no-model passthrough.
- `dl_reflector`: every graph repair rule and repaired-document validation, `passed` semantics, valid-document passthrough, determinism, serde round-trips, model revision accepted only with the same id set, all rejection paths; plus `validate_dl`'s valid multi-node document, one failing fixture per node-type rule (basic fields, every required config/inputs/outputs/next, intent expressions, branch operator set/references/default, declaration order, dangling links), determinism, `DEFAULT_MAX_RETRIES`, and serde round-trips.
- `dl_transformer` (`model.rs` + `convert/tests.rs`): DL round-trip and structural validation, `DlNodeType` wire/numeric ids and unknown rejection; one full conversion fixture per node type (numeric ids, `meta.position` progression, `data`, edges/ports), `_of_` splitting (single/nested), constant-vs-ref inputs and index counters, plugin/questioner/branch defaults and operator/logic maps, code exception config, intent determinism, null-dropping, error paths, and `to_workflow_dsl` validation.
- `mermaid`: byte-exact `graph TD` output, `Node{id}: {description}` lines, description-over-title and blank-description fallbacks, quote/backtick escaping, multi-successor labelled edges vs single-successor unlabelled edges, label-less edges, id sanitizing/deduplication, determinism/document order, no trailing newline, empty document, and design/DSL equivalence.
- `utils`: json-tagged and untagged fences, first-of-several, multiline bodies, non-`json` info strings, no-fence and empty passthrough; history formatting with the default/custom/empty separator and blank roles; first-wins merging, falsy/missing keys, custom keys, duplicate handling; session ids accepted/rejected with the id named in the error.
- `progress`: both weight tables entry by entry, unlisted stages → 0, serde/`as_str`/`Display` round-trip of all 15 stages, table coverage, and `BuildStatus::progress` delegation with the serialized snapshot unchanged.
- Pipeline tests: scripted model with queued turns for clarification-only, full workflow (3 calls), unparseable LLM answer (2 calls, deterministic repair, fixed timestamps), model failure absorbed, fully offline workflow via `ScriptedModel::new([])`, blank-query validation, fallback determinism, and a legacy `build_agent_at` regression.
- Every test is offline and deterministic: `openjiuwen_test_support::ScriptedModel`, no network, no environment reads, no wall-clock or generated-id assertions beyond non-emptiness.

## Routing Table

- `./dsl.rs` → serialized LLM-agent document and its model/plugin/workflow/dependency blocks, with unknown keys preserved
- `./workflow_dsl.rs` → serialized workflow document, node/edge models, and structural validation
- `./spec.rs` → programmatic `LlmAgentSpec`/`WorkflowSpec` and their DSL conversion
- `./bridge.rs` → DSL-to-kernel mapping: `AgentCard`, `LlmAgentConfig`, `build_llm_agent`
- `./history.rs` → bounded `HistoryManager` and `DialogueMessage`
- `./builder.rs` → session-based `AgentBuilder`, `AgentType`, legacy single-round-trip build, `build_with_pipeline`, deterministic LLM-agent repair, prompts, and `now_millis`
- `./outcome.rs` → `BuildState`, `BuildStatus` (with the additive `progress(stage)` view), `BuildOutput`, `BuildResult`, and `map_state_to_status`, the result vocabulary re-exported from `mod.rs`
- `./clarifier.rs` → `Clarifier`/`Clarification`, model-assisted and keyword-heuristic requirement clarification
- `./designer.rs` → typed workflow design, `DesignNodeKind` parsing, validation, design→DSL conversion, `WorkflowDesigner` model/offline design
- `./dl_generator.rs` → `DlGenerator` deterministic and model-assisted DSL generation, spec conversion
- `./dl_reflector.rs` → `DlReflector`, graph-shape repair with issue reporting, optional model revision, `DEFAULT_MAX_RETRIES`, and the `validate_dl` per-node-type validation entry point
- `./dl_reflector/dl_checks.rs` → private per-node-type DL validation helpers (basic checks, required fields, intent/branch expressions, references, default branches)
- `./dl_transformer/` → Python `SimpleIR` DL model, the platform workflow document, and the DL→platform-DSL transformer
- `./dl_transformer/model.rs` → `DlDocument`/`DlNode`/`DlParameters`/`DlVariable`/`DlCondition`/`DlNodeType` and structural validation
- `./dl_transformer/platform.rs` → `PlatformWorkflow`/`PlatformNode`/`PlatformEdge`, recursive null-dropping, and the `to_workflow_dsl` adapter
- `./dl_transformer/convert.rs` → `DlTransformer` driver, `PluginResource`, variable/output builders, reference parsing, edge construction
- `./dl_transformer/convert/nodes.rs` → per-node-type `data` builders (start/end/llm/branch/intent/questioner/plugin/code/output)
- `./dl_transformer/convert/tests.rs` → transformer conversion fixtures, determinism, error paths, and `to_workflow_dsl` validation
- `./cycle_checker.rs` → deterministic `find_cycle`/`has_cycle` over the workflow graph
- `./mermaid.rs` → `render_mermaid`/`render_design` in Python's `graph TD` grammar
- `./progress.rs` → `ProgressStage`, the LLM-agent/workflow weight tables, and `progress_for`
- `./utils.rs` → fence extraction, dialogue-history formatting, first-wins dict-list merging, and session-id validation
- `./designer/tests.rs` → `designer`'s `mod tests`: offline synthesis, model path and aliases, validation, design→DSL conversion
- `./builder/tests.rs` → `builder`'s `mod tests`: legacy single-round-trip builds, result vocabulary, deterministic LLM-agent repair
- `./builder/pipeline_tests.rs` → `builder`'s `mod pipeline_tests`: `build_with_pipeline` stages, model-call counts, fallbacks
- `./dl_reflector/tests.rs` → `dl_reflector`'s `mod tests`: every graph repair rule, `passed` semantics, model-revision accept/reject paths, and the `validate_dl` valid/failing fixtures per node-type rule
- `./mod.rs` → module tree, crate-level re-exports, and the pipeline overview
- `../util.rs` → shared `extract_json_fence` and text helpers (sibling, read-only — escalate writes to the parent)
- `../../openjiuwen-agent/` → `AgentCard`, `LlmAgentConfig`, `LlmAgent`, `ReActAgent` consumed by the bridge (sibling crate, read-only — escalate writes to the parent)
