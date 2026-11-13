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
- `dl_reflector.rs` — `DlReflector::{new, with_model}`, `reflect(&WorkflowDsl)`, `async reflect_with_model(&WorkflowDsl, instructions)`; `Reflection { passed, issues, repaired }`, `ReflectionIssue { kind, message, node_id }`, `ReflectionIssueKind::{DanglingEdge, DuplicateNodeId, MissingStart, MissingEnd, Cycle, InvalidNode, UnreachableNode}`.
- `cycle_checker.rs` — `find_cycle(&WorkflowDsl) -> Option<Vec<String>>`, `has_cycle(&WorkflowDsl) -> bool`.
- `mermaid.rs` — `render_mermaid(&WorkflowDsl) -> String`, `render_design(&WorkflowDesign) -> String`.
- `utils.rs` — the deterministic `agent_builder/utils/utils.py` helpers: `extract_json_from_text`, `format_dialog_history`/`format_dialog_history_with`, `merge_dict_lists(existing, new_items, unique_key)`, `validate_session_id`/`is_valid_session_id`.
- `progress.rs` — the `agent_builder/utils/progress.py` weights: `ProgressStage` (15 stages, `as_str`/`Display`, serde snake_case), `LLM_AGENT_STAGE_WEIGHTS` (6), `WORKFLOW_STAGE_WEIGHTS` (9), `progress_for(agent_type, stage) -> u8`, and `BuildStatus::progress(stage)` as a derived view.
- `mod.rs` re-exports every type above plus `find_cycle`, `has_cycle`, `render_mermaid`, `render_design`, the `utils` helpers, and the `progress` items.

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

### Cycle checker

`find_cycle` is a three-colour DFS over edges in document order from roots in node order, ignoring endpoints that are not existing nodes (structural errors belong to `validate`). The result starts and ends at the repeated node (`["a","b","c","a"]`, self-loop `["a","a"]`) and is deterministic. `has_cycle` is `find_cycle(..).is_some()`.

### Mermaid

Both renderers feed one renderer, so `render_design` is byte-identical to `render_mermaid` on the equivalent DSL graph. Output: `flowchart TD`, then one two-space-indented node line in document order, then one edge line per edge in document order (no trailing newline).
Shapes: start/end → `id([label])`, llm → `id[label]`, branch → `id{label}`, tool → `id{{label}}`, unknown type → `id[label]`.
Labels prefer `title`, then `description`, then the id. Escaping replaces quotes/backticks with `'`, shape delimiters `[ ] { } ( ) | < >` with spaces, newlines/tabs with spaces, collapses whitespace, and falls back to the id when nothing remains. Edge labels come from `condition` and use `-->|label|` (edge labels keep `<`/`>` so `score > 0.5` reads naturally).
Ids keep ASCII alphanumerics and `_`, map every other character to `_`, prefix `n_` when digit-leading or empty (`node_<index>`), and resolve collisions with `_2`, `_3`, … in document order; edges follow the rendered ids.

## DSL Shape

The DSL is a typed subset of Python's document: agent identity, `configs`, `model`, plugin/workflow references, dependency id lists, and the two epoch-millisecond timestamps.
Python-only fields survive in `LlmAgentDsl::extra` (`#[serde(flatten)]`, a `BTreeMap`), notably `model_provider` and `constraints`, and are written back unchanged.
`DslConfigs::opening_remarks` is nested inside `configs` here, while Python keeps it at the document top level; the wire name is unchanged.
Unknown keys never make parsing fail, and a document without `agent_type` defaults to `"react"`.
A workflow document uses `nodes` (`id`, `type`, `title`, `description`, `inputs`, `outputs`, `data`) and `edges` (`source`, `target`, `condition`), with `type` keeping its Python wire name; `validate` requires at least one node, unique node ids, and resolvable edge endpoints, and messages name the offending id.

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
- Files stay under the ~1000-line baseline: `designer.rs` (810), `builder.rs` (720), and `dl_reflector.rs` (607) keep their implementation only, with the test modules in `designer/tests.rs`, `builder/tests.rs`, `builder/pipeline_tests.rs`, and `dl_reflector/tests.rs` loaded through `#[cfg(test)] mod …;`.
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

## Deferred

- Python's full build-state machine stages — intention detector, multi-turn LLM clarifier dialogue, agent designer, workflow designer's 3-pass (basic/branch/reflection) prompts, DL `refine` mode, and the reflection-evaluate repair loop that feeds errors back to the designer — are replaced by the single-pass stages above.
- Python's resource retrieval/preprocessing (plugins, knowledge bases, prompt templates from the platform) has no counterpart; documents carry ids only, and the DL generator performs no asset search.
- Python's LLM-judged design review, `run_eval_query.py` shell harness, and `main.py` CLI driver for the builder are not ported; the CLI lives in `openjiuwen-cli`.
- Python's DL→DSL transformer (`builders/workflow/dl_transformer/`) is not ported: there is no SimpleIR node list, no numeric `NodeType.dsl_type` ids (Start→1 … Plugin→19), no x/y `+20` position assignment, no `${node.a_of_b}` `_of_` splitting, no None-dropping recursive dict serialiser, and no per-converter defaults (plugin `toolID`/`pluginVersion: "draft"` and default outputs, questioner empty-input `node_start.query`, `BRANCH_OPERATOR_MAP`/`BRANCH_LOGIC_MAP`).
- `WorkflowDesign::to_workflow_dsl` is the only design→DSL converter and it keeps string node types (`start`/`llm`/`branch`/`tool`/`end`); Python DL node types outside that set (`Questioner`, `IntentDetection`, `Output`, `Code`, `Plugin`) exist only as `DesignNodeKind` parse aliases (`Questioner`→`Llm`, `Output`→`End`, `Plugin`/`Code`→`Tool`, `IntentDetection`→`Branch`).
- Reflection is graph-structural only: Python's per-node-type field checks (`_check_start_node` query/用户输入 output, `_check_llm_node` `configs.system_prompt`+`configs.user_prompt`, `_check_questioner_node` `configs.prompt`, `_check_plugin_node` `configs.tool_id`, `_check_code_node` `configs.code`, `_check_output_node`/End `configs.template`, intent/branch condition `operator`/`variable`/`default` rules, `_check_inputs_list`/`_check_outputs_list` placeholder resolution, `_check_next_missing`) are not ported; the port validates node presence, unique ids, resolvable edges, exactly one start, at least one end, reachability, and acyclicity.
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
- `dl_reflector`: every issue kind repaired and the repaired document validating, `passed` semantics, valid-document passthrough, determinism, serde round-trips, model revision accepted only with the same id set, all rejection paths.
- `mermaid`: shapes per kind, escaping of labels and conditions, id sanitizing/deduplication, determinism/document order, exactly-once node and edge lines, design/DSL equivalence.
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
- `./outcome.rs` → `BuildState`, `BuildStatus`, `BuildOutput`, `BuildResult`, and `map_state_to_status`, the result vocabulary re-exported from `mod.rs`
- `./clarifier.rs` → `Clarifier`/`Clarification`, model-assisted and keyword-heuristic requirement clarification
- `./designer.rs` → typed workflow design, `DesignNodeKind` parsing, validation, design→DSL conversion, `WorkflowDesigner` model/offline design
- `./dl_generator.rs` → `DlGenerator` deterministic and model-assisted DSL generation, spec conversion
- `./dl_reflector.rs` → `DlReflector`, rule-based repair with issue reporting, optional model revision
- `./cycle_checker.rs` → deterministic `find_cycle`/`has_cycle` over the workflow graph
- `./mermaid.rs` → `render_mermaid`/`render_design` Mermaid `flowchart TD` output
- `./progress.rs` → `ProgressStage`, the LLM-agent/workflow weight tables, and `progress_for`
- `./utils.rs` → fence extraction, dialogue-history formatting, first-wins dict-list merging, and session-id validation
- `./designer/tests.rs` → `designer`'s `mod tests`: offline synthesis, model path and aliases, validation, design→DSL conversion
- `./builder/tests.rs` → `builder`'s `mod tests`: legacy single-round-trip builds, result vocabulary, deterministic LLM-agent repair
- `./builder/pipeline_tests.rs` → `builder`'s `mod pipeline_tests`: `build_with_pipeline` stages, model-call counts, fallbacks
- `./dl_reflector/tests.rs` → `dl_reflector`'s `mod tests`: every repair rule, `passed` semantics, model-revision accept/reject paths
- `./mod.rs` → module tree, crate-level re-exports, and the pipeline overview
- `../util.rs` → shared `extract_json_fence` and text helpers (sibling, read-only — escalate writes to the parent)
- `../../openjiuwen-agent/` → `AgentCard`, `LlmAgentConfig`, `LlmAgent`, `ReActAgent` consumed by the bridge (sibling crate, read-only — escalate writes to the parent)
