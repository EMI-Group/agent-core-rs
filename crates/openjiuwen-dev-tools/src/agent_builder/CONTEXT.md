# openjiuwen-dev-tools / agent_builder

## Intent

Ports `openjiuwen.dev_tools.agent_builder`: programmatic agent/workflow specs, the serialized platform DSL, the staged builder pipeline, and the kernel bridge.
The pipeline turns one caller request into a ready artifact (`LlmAgentDsl`, validated `WorkflowDsl`) or one specific clarifying question, and reports where every stage's output came from.
Every stage has a deterministic offline implementation; a `Model` is optional and only ever improves a stage, never gates it.
Internal structure is idiomatic Rust (serde models, typed errors, futures streams), not a transliteration of Python's multi-agent state machine or its Markdown prompts.
The module tree is seventeen modules (`mod.rs`), plus private submodules under `builder`, `clarifier`, `designer`, `dl_reflector`, `dl_transformer`, `intention`, and `mermaid`.

## API Surface

Core artifacts:
- `LlmAgentDsl` (`dsl.rs`) with `to_value`/`from_value`/`to_json_string`/`from_json_str`, plus blocks `DslConfigs`, `DslModel`, `DslModelInfo`, `PluginRef`, `WorkflowRef`, `DslDependencies`.
- `WorkflowDsl` (`workflow_dsl.rs`) with `DslNode`, `DslEdge`, `validate()`, `node(id)`, `to_json_string`, `from_json_str`.
- `LlmAgentSpec` and `WorkflowSpec` (`spec.rs`) with `to_dsl`, `to_json_string`, `from_json_str`, and `with_*` builders.
- `LlmAgentDsl::{to_agent_card, to_llm_agent_config, build_llm_agent}` (`bridge.rs`) map a document into `openjiuwen_agent::{AgentCard, LlmAgentConfig, LlmAgent}`.
- `HistoryManager` and `DialogueMessage` (`history.rs`) with `new(max)`, `add_message`, `add_user_message`, `add_assistant_message`, `get_history`, `get_latest_k(k)`, `len`, `is_empty`, `clear`, `max_history_size`; `DEFAULT_MAX_HISTORY_SIZE` is `20`.

Result vocabulary and provenance (`outcome.rs`):
- `BuildState` (`initial`/`clarifying`/`requesting`/`processing`/`completed`), `BuildStatus` (`session_id`, `agent_type`, `state`, `resource_count`, plus the derived `progress(stage)` view), `BuildOutput`, `BuildResult`, and `map_state_to_status`.
- `BuildOutput` serializes externally tagged, so its JSON is exactly `{"dsl": {...}}`, `{"workflow": {...}}`, or `{"response": "..."}`; `BuildOutput::as_json` returns that value.
- `BuildResult` carries `status`, `session_id`, `agent_type`, `output`, `provenance`, and `repair`; `to_json` flattens the output and appends `"provenance"` only when `provenance.has_deviation()` and `"repair"` only when the report is non-empty, so the four Python keys are otherwise unchanged.
- `GenerationSource` is `{Model, Deterministic, Fallback}` with `as_str` (`"model"`, `"deterministic"`, `"fallback"`).
- `StageProvenance` is `{stage, source, detail}`; `GenerationProvenance` is `{stages}` with `new`, `is_empty`, `len`, `record(stage, source, detail)`, `record_model`, `record_deterministic`, `record_fallback`, `extend_from`, `has_fallback`, `has_deviation`, `fallback_count`, `stages_matching(source)`, and `warn_if_degraded` (one `tracing::warn!` per non-model stage naming its stage, source, and detail).
- `RepairReport` is `{attempts, notes}` with `is_empty`, `record_attempt`, and `record_note`.

Intention detection (`intention.rs`):
- `IntentionDetector::{new, with_model}`, `async detect_initial(query, history)`, `async detect_refine(query, history, mermaid)`.
- `IntentionVerdict { intent, source, detail }` and `BuildIntent { Build, Refine, Clarify }` with `as_str`/`Display`.

Requirement clarifier (`clarifier.rs` with `clarifier/rounds.rs`):
- `Clarifier::{new, with_model}`, `async clarify(query, history) -> Result<Clarification>` (one round), `async clarify_rounds(query, history, config)`, `async clarify_rounds_with(query, history, config, responder)`, and `stream_clarify(query, history)`.
- `Clarification { ready, question, summary, provenance }`; the single-pass verdict records exactly one `clarifying` provenance entry.
- `rounds` owns `ClarifyConfig { max_rounds }` (`new` maps `0` to the default, `resolved_max_rounds` is never zero), `DEFAULT_MAX_CLARIFY_ROUNDS = 3`, `ClarificationRound { index, question, answer, ready, summary, source }`, `ClarificationTermination { Ready, NoAnswer, RepeatedQuestion, RoundCap }` with `as_str`/`Display`, `ClarificationDialogue { rounds, clarification, termination, provenance }`, and `ClarifyEvent { Round, Completed }`.

Workflow designer (`designer.rs` with `designer/passes.rs`, `designer/passes/clauses.rs`, `designer/stream.rs`):
- `WorkflowDesigner::{new, with_model}`, `async design(query)`, `async design_with_provenance(query) -> Result<DesignOutcome>`, `design_offline(query)`, `design_basic_offline(query)`, `design_branch_offline(query, basic)`, `design_reflection_offline(query, with_branches) -> Result<(WorkflowDesign, Vec<String>)>`, `async design_three_pass(query)`, and `stream_design(query)`.
- `DesignOutcome { design, provenance }`; `WorkflowDesign { name, description, nodes, edges }` with `validate` and `to_workflow_dsl`; `DesignNode { id, kind (wire `type`), title, description, config }`; `DesignEdge { source, target, condition }`.
- `DesignNodeKind { Start, Llm, Branch, Tool, End }` with `as_str`/`Display`/`FromStr`/`TryFrom<String>` and tolerant alias parsing (`Output`→End, `Questioner`/`LLM`→Llm, `Plugin`/`Code`→Tool, `IntentDetection`→Branch).
- `passes` owns `DesignPassKind { Basic, Branch, Reflection }` (with `as_str`), `DesignPass { kind, source, detail, node_count, edge_count }`, and `ThreePassDesign { design, passes, provenance, notes }`.
- `stream` owns `DesignEvent { PassStarted(DesignPassKind), PassCompleted(DesignPass), Completed(ThreePassDesign) }`, externally tagged.

DL generation (`dl_generator.rs`):
- `DlGenerator::{new, with_model}`, `generate(&WorkflowDesign)`, `from_spec(&WorkflowSpec)`, `async generate_with_model(&WorkflowDesign)`, and `async refine(feedback, existing, mermaid) -> Result<Refinement>`.
- `Refinement { workflow, provenance }`; the provenance always holds exactly one `refining_dl` record.

DL reflection (`dl_reflector.rs` with `dl_reflector/dl_checks.rs`):
- `DlReflector::{new, with_model}`, `reflect(&WorkflowDsl)`, `async reflect_with_model(&WorkflowDsl, instructions)`, and `validate_dl(&DlDocument)`.
- `Reflection { passed, issues, repaired }`, `ReflectionIssue { kind, message, node_id }`, `DlValidation { passed, issues }`.
- `ReflectionIssueKind` has sixteen variants: `DanglingEdge`, `DuplicateNodeId`, `MissingStart`, `MissingEnd`, `Cycle`, `InvalidNode` (also the rejected-model-revision marker), `UnreachableNode`, `MissingField`, `InvalidExpression`, `MissingDefaultBranch`, `InvalidReference`, `MissingConditionExpression`, `UnknownNodeType`, `DuplicateVariable`, `MultiPlaceholderValue`, `UndeclaredReference`.
- `DEFAULT_MAX_RETRIES = 3` is Python's repair-loop bound; the pipeline's repair loop reads it, `reflect` and `validate_dl` never retry.

Builder (`builder.rs` with `builder/pipeline.rs`, `builder/stream.rs`):
- `AgentBuilder::{new, build_agent, build_agent_at, build_with_pipeline, build_with_pipeline_stream, record_result, history(session_id, k), clear_session, status, model}`.
- `AgentType` (`llm_agent`/`workflow`) with `as_str`/`Display`/`FromStr`; `now_millis`; `PipelineEvent`.
- A session is a private `SessionState { history, state, agent_type, workflow }`; `workflow` is the newest workflow document the session produced and the input to the next request's refinement decision.
- `pipeline` owns `run` (collapses the stream to its `Final`), `llm_agent_pipeline`, `generate_workflow`, `clarification_text`, `last_provenance`, `issue_note`, `joined_issue_notes`, and `repair_feedback`.
- `stream` owns `pipeline_stream` (the single implementation of the pipeline) and `PipelineEvent { StageStarted { stage }, StageCompleted { stage, source, detail }, Clarified(ClarificationRound), Repaired(RepairReport), Final(BuildResult) }`.

Support:
- `cycle_checker` — `find_cycle(&WorkflowDsl) -> Option<Vec<String>>`, `has_cycle`.
- `mermaid` — `render_mermaid(&WorkflowDsl)`, `render_design(&WorkflowDesign)`, and `when_condition_label(&str) -> Option<String>`; `mermaid/labels.rs` holds the id, endpoint, and label helpers.
- `dl_transformer` — the Python `SimpleIR` document (`DlDocument`, `DlNode`, `DlParameters`, `DlVariable`, `DlCondition`, `DlNodeType`), `DlTransformer::{new, transform, transform_with_resources, transform_str}`, `PluginResource`, and `PlatformWorkflow { nodes, edges }` with `to_value`/`to_json_string`/`to_workflow_dsl`.
- `utils` — `extract_json_from_text`, `format_dialog_history`/`format_dialog_history_with`, `merge_dict_lists`, `validate_session_id`/`is_valid_session_id`.
- `progress` — `ProgressStage` (15 stages, `as_str`/`Display`), `LLM_AGENT_STAGE_WEIGHTS` (6), `WORKFLOW_STAGE_WEIGHTS` (9), and `progress_for(agent_type, stage) -> u8`.
- `mod.rs` re-exports the public types above, the three stream/event vocabularies, and the deterministic helpers.

## Pipeline

`build_with_pipeline(query, session_id, agent_type, timestamp_ms)` runs the staged pipeline and `build_with_pipeline_stream(query, session_id, agent_type, timestamp_ms)` exposes the same run as a `PipelineEvent` stream; the stream is the single implementation, so both produce identical results.

Session handling: a blank query is rejected before any session exists; otherwise the session is created on first use with `HistoryManager::new(DEFAULT_MAX_HISTORY_SIZE)`, `agent_type` is set, the retained turns and the stored workflow artifact are snapshotted as `prior` and `previous`, the user turn is appended, and the state becomes `Processing`.
`prior` and `previous` are taken *before* the user turn, so no stage sees the query twice.
After the pipeline the result is persisted through `record_result`, which appends the assistant text (the serialized document, or the plain response), sets the state to `completed` for a document and `clarifying` for a response, and stores a produced workflow as the session's newest artifact.

Stage order and events:

1. **Intention** (`detecting_intention`) — `detect_refine` when the session holds a workflow artifact and `agent_type` is `Workflow`, else `detect_initial`; the detector is deterministic on a request with no retained turns and model-assisted afterwards.
2. **Refinement** (`refining_dl`) — when the verdict is `BuildIntent::Refine` and a workflow artifact exists, one `DlGenerator::refine(&query, &workflow, &render_mermaid(&workflow))` call, then the result is `completed` immediately (no clarify, design, or generation stage runs).
3. **Clarification** (`clarifying`) — `Clarifier::with_model(..).clarify_rounds(&query, &prior, &ClarifyConfig::new(DEFAULT_MAX_CLARIFY_ROUNDS))`, with a responder that always returns `None`, so the dialogue is exactly one round and terminates `Ready` or `NoAnswer`; every round is reported as `Clarified`, and a not-ready verdict ends the build with `BuildOutput::Response` and state `Clarifying`.
4. **LLM agent** (`generating_config`) — one generation round trip over `prior` plus the query, then `repair_llm_dsl`; output `Dsl`, state `Completed`.
5. **Workflow** — the designer's three passes streamed as designer events (`generating_workflow_design`, `generating_workflow_design`, `validating_dl`), then `generating_dl`, then the bounded repair loop.

The repair loop counts one `RepairReport::record_attempt` per `DlReflector::reflect` and reports each with `Repaired`.
A `reflect` error or a failed `DlGenerator::refine` logs at debug, reports the stage `Deterministic` with the reason, and ends the loop with the document it already has.
A refinement is used only when it differs from the repaired document and validates; otherwise the loop ends with the repaired document.
After a non-passing reflection the loop records the joined issue notes and feeds `repair_feedback(issues)` to `DlGenerator::refine` with the repaired document and its Mermaid diagram, so a refusal by the reflector costs one refine call.
The loop stops when `reflection.passed` or `repair.attempts >= DEFAULT_MAX_RETRIES` (3 reflections, hence at most 2 refinements), and never errors.

Provenance records the pipeline emits, in order: `detecting_intention`; `refining_dl` on the refine path; one `clarifying` per round; then `generating_config`, or the designer's two `generating_workflow_design` and one `validating_dl` records followed by `generating_dl` and one `refining_dl` per refinement.
The per-attempt reflect results are carried by `RepairReport` (`attempts`, `notes`), not by provenance.
`warn_if_degraded` is called once before each `Final`, so a degraded build logs one `warn!` per non-model stage.

Model-call counts on a first request (empty session history, so the intention verdict is deterministic): a workflow build makes 5 calls (clarifier, three designer passes, DL generator) and an LLM-agent build makes 2 (clarifier, generation); a request the clarifier rejects makes 1 and stops there.
A later request adds one intention call to whichever path it takes, and a refine path is intention plus refinement only (2 calls).
Each repair-loop re-attempt adds one refine call.

## Stage Contracts

### Intention detector

The deterministic initial rule lowercases the retained `user` turns followed by the query and yields `Build` when it counts at least two sequencing phrases (`, then `, ` then ``, `after that`, `next,`, `首先`, `然后`, `接着`, `最后`, `，然后`, `，再`, case-insensitive for ASCII), at least two numbered list items (`1.`, `2)`, `3、` at a word boundary), or one workflow keyword (`workflow`, `steps`, `pipeline`, `流程`, `步骤`, `工作流`); otherwise `Clarify`.
The deterministic refine rule yields `Refine` exactly when the diagram is non-blank and the query carries a modification keyword (`modify`, `change`, `add`, `remove`, `refine`, `adjust`, `update`, `fix`, `修改`, `改成`, `增加`, `删除`, `调整`, `优化`), else `Build`.
The model path sends one system prompt plus the formatted dialogue (and the current diagram, labelled `## Current Mermaid diagram`, for refinement) and expects one fenced `{"provide_process": bool}` or `{"need_refined": bool}`; `provide_process` maps to `Build`/`Clarify` and `need_refined` to `Refine`/`Build`.
A failed call, a choice-less answer, a missing fence, or a contract mismatch records `Fallback` with the reason and the rule's verdict; the detector never errors except on a blank query, checked before any model call.
`IntentionVerdict::detail` always explains the decision (`model answered need_refined=true`, `the request numbers 3 steps`, `the request does not ask to modify the diagram`, …).

### Clarifier (single-pass heuristic)

The model path sends one system prompt plus one message per retained turn, then the query as the final user message, and expects one fenced ```json `{"ready", "question", "summary"}`; strings are trimmed.
A model error, missing choices, no fence, or a contract mismatch logs `tracing::debug!` and falls back to the heuristic — it never errors.
Offline context = every retained `user` turn (oldest first, `\n`-joined) + the query, lowercased for keyword matching.
Four signals: goal, input (or a `${...}` placeholder), output, tools. `ready = goal && output`. When not ready, the question names the first missing signal in the order goal → input → output → tools; `summary` lists the detected signals as `"goal, input, output, tools"`.
`clarify` is `clarify_rounds_with` with one round and no responder, so a single-pass verdict and a dialogue's first round are the same judgement.

### Clarifier rounds

A dialogue judges the query against a clone of the caller's `HistoryManager`; each answered round appends its answer as one `user` turn to that working copy, and the caller's history is never mutated.
`responder` is asked only for a not-ready round and returns `None` (or a blank answer, which is treated as `None`) to stop.
The termination checks run in a fixed order: `Ready` (the round judged the request complete), `NoAnswer` (no or blank answer), `RepeatedQuestion` (the round's trimmed question equals the previous round's), `RoundCap` (`max_rounds` rounds ran), so a ready verdict is never reported as a repeat.
A dialogue always runs at least one round; `ClarifyConfig::new(0)` and `Default` both use `DEFAULT_MAX_CLARIFY_ROUNDS`.
`ClarificationDialogue::clarification` is the last round's verdict carrying the aggregate provenance, and `finish` warns once about any degraded round.
`stream_clarify` owns clones of the clarifier, query, and history (`Send + 'static`), runs the default config with no responder, and yields one `Round` per round followed by one `Completed`; a blank query yields one validation error and nothing else.

### Workflow designer (single pass and offline synthesis)

`design_offline` splits the query into at most `MAX_STEPS = 4` clauses at sentence terminators (`.`, `;`, `。`, `；`, `\n`, `!`, `?`, `！`, `？`), sequencing phrases, and numbered list markers, strips leading list markers, drops empty/non-alphabetic and duplicate clauses, and falls back to the whole query when nothing survives.
It emits `start` (id `start`, `inputs.query`) → one `llm` node per clause (`llm_1..llm_N`, deterministic title from the first 8 words or 20 CJK characters, config carrying `instruction`, `inputs.query`, `outputs.result`) → the legacy branch node (id `branch`, title `Decide`, `config.conditions = ["default"]`) when the lowercased query contains any of if/when/whether/condition/decision/branch/else/otherwise/如果/否则/当/是否/条件/分支 → `end` (id `end`); edges connect in document order.
The workflow name is a 40-character slug of the first clause (CJK kept, other runs → `-`, fallback `workflow`); the description is the query truncated to 160 characters.
`design` returns `design_with_provenance(..).design`: with no model the offline design is recorded `deterministic`; an accepted model answer is recorded `model`; a rejected one is logged at debug, replaced by the offline design, and recorded `fallback` with the rejection reason (capped at 160 characters).
`WorkflowDesign::validate` requires at least one node, unique ids, resolvable edges, exactly one start node, and at least one end node.
`WorkflowDesign::to_workflow_dsl` is the single design→DSL converter: kind → `node_type`, `config.inputs`/`config.outputs` objects become the node maps, `data` is the whole node config, then the produced document is validated.

### Workflow designer (three passes)

`design_three_pass` runs basic → branch → reflection and returns exactly three `DesignPass` records plus one provenance record per pass (the reflection pass records `validating_dl`, the others `generating_workflow_design`).
**Basic pass** (`design_basic_offline`): the offline rule above without any branch node and never calling a model; a model answer is accepted only when it parses from one fenced JSON design and `validate` passes, otherwise the baseline is used with `fallback` and a `model answer rejected: …` detail; the deterministic detail is `1 step synthesised` / `N steps synthesised`.
**Branch pass** (`design_branch_offline`): when the query holds a decision marker and the design has neither a branch node nor the id `branch`, one branch node is inserted before the end node with `config.conditions = ["condition", "default"]` and every edge into the end node is rewired through it; without a marker or when a branch already exists the design is returned unchanged, so the pass is idempotent; a model answer must parse, validate, and (only when the request holds a decision marker) actually contain a branch node.
**Reflection pass** (`design_reflection_offline`): repairs run in this order, each appending one note — an empty design becomes `start → end`; duplicate ids are renamed `<id>_2`, `<id>_3`, …; edges naming an unknown endpoint are dropped; a missing start is inserted at the front and connected to the first node; a missing end is appended and connected from every sink; each unreachable node is connected from the nearest earlier reachable node (the start node when none is earlier); cycles are broken by dropping the back edge `find_cycle` reports; the repaired design is then validated.
An accepted model reflection answer replaces the design and clears the notes; any rejection keeps the repaired baseline, appends `fallback`, and keeps the notes.
`design_reflection_offline` ignores its `query` argument, which only the model-backed pass reads.

### Workflow designer (stream)

`stream_design` owns a clone of the designer and the query (`Send + 'static`) and yields `PassStarted`/`PassCompleted` around each pass in order, then one `Completed(ThreePassDesign)`; a blank query yields one validation error and ends the stream.
The stream's `Completed` payload is the same value `design_three_pass` returns for the same inputs.

### DL generator

`generate` is `design.to_workflow_dsl()`; `from_spec` is `spec.to_dsl()` plus `validate()` and never calls a model.
`generate_with_model` computes the deterministic baseline first (an invalid design is a validation error regardless of the model), then asks the model for one fenced workflow document and accepts it only when `WorkflowDsl::validate` passes; any model failure or invalid document falls back to the baseline. Its result carries no source, so the pipeline infers one by comparing the answer with the baseline.
`refine` rejects blank feedback as a validation error before any model call; without a model it returns the document unchanged with a `deterministic` record (`no model configured; keeping the existing workflow`), and with one it sends the refine system contract plus a user message holding the feedback, the Mermaid diagram, and the document JSON.
A refinement answer is accepted only when the fenced-or-bare JSON parses as `WorkflowDsl` and validates; a model error, a choice-less answer, an unparseable answer, or an invalid document keeps the existing document with a `fallback` record naming the reason.
The accepted path's provenance holds one `model` record and no warning is emitted.

### DL reflector (repair rules)

`reflect` clones the document and repairs in this order, recording one `ReflectionIssue` per repair: an empty document becomes `start → end` (`MissingStart` + `MissingEnd`); blank ids become `node_<index>` and blank titles take the id (`InvalidNode`); duplicate ids → later copies renamed `<id>_2`, `<id>_3`, … (`DuplicateNodeId`); edges with unknown endpoints dropped (`DanglingEdge`); a missing start inserted at the front and connected to the first node (`MissingStart`); a missing end appended and every sink connected (`MissingEnd`); unreachable nodes connected from the nearest earlier reachable node (`UnreachableNode`); cycles broken by removing the back edge `path[len-2] -> path[len-1]` of `find_cycle`'s path (`Cycle`, message names the path). The repaired document is then validated.
`passed = issues.is_empty()` and an already-valid document is returned unchanged. Repairs never mutate the input.
`reflect_with_model` runs `reflect` first, then accepts a model revision only when it parses, validates, and keeps the same node id set; otherwise it returns the baseline plus one `InvalidNode` issue whose message starts with `model revision rejected: `. Model problems never surface as errors.

### DL reflector (per-node-type validation)

`validate_dl` is model-free, never errors, never mutates, and `passed == issues.is_empty()`.
It walks nodes in document order, accumulating each node's declared outputs as `"{node_id}.{name}"`, so a reference may only use outputs declared at or before that point.
Basic checks: blank `id`/`type`/`description` → `MissingField`, a repeated id → `DuplicateNodeId`, a type outside the nine wire names → `UnknownNodeType` (the node is then skipped), and a `next`/`conditions[].next` naming no node → `DanglingEdge`.
Per type: `Start` needs an output `query` described `用户输入` and a `next`; `LLM` needs non-empty `inputs`/`outputs`, non-blank `configs.system_prompt`/`configs.user_prompt`, and a `next`; `Questioner`/`Plugin`/`Code` need non-empty `inputs`/`outputs`, their `configs.prompt`/`configs.tool_id`/`configs.code`, and a `next`; `Output` needs inputs, `configs.template`, and a `next`; `End` needs inputs and `configs.template`.
`IntentDetection` needs inputs, `configs.prompt`, non-empty conditions, branch/description/next per condition, an expression holding `${<node_id>.rawOutput}` and `contain` per non-default condition, and a mandatory `default` condition.
`Branch` needs non-empty conditions, branch/description/next per condition, exactly one of `expression`/`expressions`, the operator as the second whitespace token from `{eq, not_eq, contain, not_contain, longer_than, longer_than_or_eq, short_than, short_than_or_eq, is_empty, is_not_empty}`, every `${…}` variable already declared, and a mandatory `default` condition.
Operator/reference/expression-shape problems are `InvalidExpression`/`InvalidReference`/`MissingConditionExpression`; absent fields are `MissingField`; a missing fallback is `MissingDefaultBranch`.
Every inspected input and output list is also checked variable by variable: a name repeated inside one list → `DuplicateVariable`, an input without a value → `MissingField` naming the variable, an output without a description → `MissingField` naming the variable, an input value with more than one `${...}` placeholder → `MultiPlaceholderValue`, and an input value whose single placeholder names no output declared by an earlier or the current node → `UndeclaredReference`. Non-string values are accepted without reference checks.
Variable `type` is never validated, because Python's `available_variable_types` set and its `check_type` flag are not ported.

### Cycle checker

`find_cycle` is a three-colour DFS over edges in document order from roots in node order, ignoring endpoints that are not existing nodes (structural errors belong to `validate`). The result starts and ends at the repeated node (`["a","b","c","a"]`, self-loop `["a","a"]`) and is deterministic. `has_cycle` is `find_cycle(..).is_some()`.

### Mermaid

Both entry points feed one renderer, so `render_design` is byte-identical to `render_mermaid` on the equivalent DSL graph.
Output is Python's `simpleir_to_mermaid` grammar: the literal header `graph TD`, then one two-space-indented node line per node in document order, then one edge line per edge in edge-array order, with no trailing newline.
A node line is `  {id}[Node{id}: {label}]`; the label comes from the node `description` (never `title`) and falls back to the original node id (then the rendered id) when the description is blank; only backtick and `"` are escaped, both to `'`.
An edge is `  {src} --> {dst}`; it becomes `  {src} -- {label} --> {dst}` only when the source node has more than one outgoing edge and the label is non-blank (outgoing edges counted over raw source ids, so sanitized ids do not change the decision).
The label is the escaped condition, except that a `When condition met[…]` description is first reduced by `when_condition_label` to the bracketed inner text (bracket-depth aware) or to the remainder with one leading `:`/`=`/`-` stripped.
Ids keep ASCII alphanumerics and `_`, map every other character to `_`, use `node_<index>` (1-based) for an empty id, prefix `n_` when digit-leading, and resolve collisions with `_2`, `_3`, … in document order; a dangling endpoint keeps its sanitized form.

### DL transformer

`DlTransformer` validates the document, then converts nodes in document order: `type` is the numeric platform id (`Start` 1, `End` 2, `LLM` 3, `Branch` 4, `IntentDetection` 6, `Questioner` 7, `Output` 9, `Code` 10, `Plugin` 19; an unknown type is a validation error), `meta.position` starts at `(0,0)` and steps by 20 on both axes, and `data.title` is the node description.
Inputs become `constant` variables (`schema.type` default `"string"`, `extra.index` per node starting at 0) or `ref` variables whose `content` is `[node, *_of_-reversed name]`; outputs are nested `object` schemas built by reversing `_of_` name segments, each leaf carrying `extra.index`.
Per-type rules: `Start`/`Code` make every output required; `LLM` derives `output_format` (multi-output text/markdown → json, single unknown → text) and carries `systemPrompt`/`prompt` templates, the fixed model config, and `response_format`; `Questioner` duplicates its prompt into `system_prompt`, sets `history_enable:false`/`max_response:3`, adds `user_response` (always) and `output` (when another output exists), requires everything but `output`, and injects `node_start.query` when it has no inputs; `Plugin` resolves each parameter per field to the first non-blank value of the first matched `PluginResource` (`toolId` equal to the node's non-blank `configs.tool_id`), the node's matching `configs` entry, and the field default (`"draft"` for `pluginVersion`, empty for the other four), and adds `error_code`/`error_message`/`data` defaults; `Code` carries `language:"python"` and the fixed `exception_config`; `Output` carries the `template` content; `Branch` emits `branches` with the operator/logic maps and per-condition `source_port_id`; `IntentDetection` resolves each condition's `intent_id` to a fresh uuid4 when absent and builds the intent ports (`"0"` for `default`) from those resolved ids, so a generated id always equals its edge's `source_port_id`.
Edges come from `next` (unconditional, no port) or from conditions/intents (one per branch, `source_port_id` = branch or intent id); `Code` uses port `"0"`.
`PlatformWorkflow::to_value`/`to_json_string` apply Python's recursive null-dropping (nulls in objects and arrays disappear; `0`, `""`, and `false` survive), and `to_workflow_dsl` adapts the document to the simplified `WorkflowDsl` and validates it.
Every invalid input (unknown type, missing required config, malformed reference/operator) is `Error::validation` naming the node and field.

## Constraints

- Only this directory may be edited here; sibling crates, `src/lib.rs`, `Cargo.toml`, `src/util.rs`, the crate's `tests/`, and `docs/` are out of scope.
- Errors are `openjiuwen_common::error::{Error, Result}` only: invalid user input is `Error::validation`, and a pipeline call fails only on a blank query or a stage that cannot represent its input (a design that does not convert, a designer stream without a result).
- Pipeline stages absorb model problems: a model failure is logged with `tracing::debug!` and the stage's deterministic result is used, while the aggregate degradation is logged once with `tracing::warn!` by `GenerationProvenance::warn_if_degraded`.
- The legacy `build_agent`/`build_agent_at` entry points keep Python's behavior: a model error propagates, a choice-less answer is `Error::llm`, and an unparseable answer becomes a plain-text `Response`.
- All public data types are serde (`rename_all = "snake_case"`, `default`, unknown-field tolerant) and keep Python's JSON field names; the core models derive `schemars::JsonSchema`.
- `provenance` and `repair` are additive on the wire: a plain-model first-attempt result keeps exactly the four Python keys in `BuildResult::to_json`, and the struct's container `default` lets a result serialized without them still deserialize.
- Time is a parameter everywhere except `now_millis`; randomness is limited to a missing/generated `agent_id` and to intent ids in `DlTransformer`, so tests never assert a clock value or a generated id.
- Async-first: model-assisted entry points are `async fn` and compose inside an existing tokio runtime; no blocking calls, no environment reads, no `block_on`.
- Streams are `BoxStream<'static, Result<_>>` built with `async_stream` and own their inputs, so a stream outlives the value that created it; `build_with_pipeline_stream` takes `&self` and never mutates a session, and the caller persists a `Final` result with `record_result`.
- No `unsafe`, no `unwrap`/`expect`/`panic!` in library code, no printing; diagnostics go through `tracing`.
- Files stay around the ~1000-line baseline: `dl_transformer/model.rs` (954), `designer/passes.rs` (859), `clarifier.rs` (820), `outcome.rs` (820), `mermaid.rs` (808), `dl_generator.rs` (757), `builder.rs` (749), `dl_reflector.rs` (718), `designer.rs` (652), `intention.rs` (511), `clarifier/rounds.rs` (391), `builder/stream.rs` (374), `mermaid/labels.rs` (307).
- The graph and pipeline implementation files carry no tests; suites live in inline `#[cfg(test)] mod tests` blocks (`outcome`, `clarifier`, `dl_generator`, `mermaid`, `mermaid/labels`) or in sibling modules (`designer/tests.rs`, `designer/passes/tests.rs`, `designer/stream/tests.rs`, `clarifier/rounds/tests.rs`, `intention/tests.rs`, `builder/tests.rs`, `builder/pipeline_tests.rs`, `builder/stream_tests.rs`, `dl_reflector/tests/`, `dl_transformer/convert/tests/`), each also under the ~1000-line baseline (largest: `designer/passes/tests.rs`, 789).
- A moved test module keeps its path (`builder::pipeline_tests`, `designer::passes::tests`), so test bodies move verbatim and only the crate's `tests/` suites observe the public API.
- Three lints are allowed locally and deliberately: `large_enum_variant` on `BuildOutput` and `PipelineEvent` (the document/result is the payload callers want), and `too_many_lines`/`cognitive_complexity` on `pipeline_stream` (one ordered narrative is clearer than a helper per stage).

## Design Decisions

- The builder owns `Arc<dyn Model>` supplied by the caller; every stage takes the same `Arc` or a `&dyn Model`, so no stage constructs a client or reads provider configuration, and every stage stays usable without a model.
- Deterministic cores are public and model-free: the intention rules, the clarification heuristic, design synthesis and repairs, validation, cycle detection, Mermaid rendering, and DL conversion all run without a model.
- Provenance is the one degradation channel: every stage records where its output came from, `BuildResult` hands the ordered records back, and `to_json` appends them only when some stage did not come from a model, so a clean build keeps its historical wire shape and no stage result type needs a `used_fallback` flag.
- A degraded stage is additionally visible through the `warn!` from `warn_if_degraded` and, on the workflow path, the `RepairReport` notes.
- The staged pipeline is one `PipelineEvent` stream; `build_with_pipeline` drives that stream and returns its `Final`, so the buffered and streamed builds cannot diverge.
- The stream holds `&self` only for the session snapshot: it clones the model, the retained turns, and the newest artifact, runs without borrowing the builder, and leaves persistence to `record_result`, which maps a document to `completed` and a response to `clarifying`.
- Clarification inside the pipeline is one round on purpose: the pipeline has no user to answer, so the bounded multi-round dialogue stays a caller-driven API (`clarify_rounds_with` with a responder) rather than an internal loop.
- The intention detector asks the model only once a session holds turns, so a first request costs no intention call and a first-request `Refine` verdict cannot fire (there is no artifact to refine).
- The repair loop never errors and is bounded by `DEFAULT_MAX_RETRIES`: a failed reflection or refinement keeps the best document so far, an exhausted budget still returns a validated document, and a refinement that is equal or invalid ends the loop instead of burning another attempt.
- `generate_with_model` keeps its single-document signature, so the pipeline infers `model` vs `fallback` by comparing the answer with the deterministic baseline; an answer identical to the baseline counts as a fallback.
- The three designer passes record typed `DesignPass` values rather than parsing Python's Markdown sections: each pass's acceptance rule is explicit, and the reflection notes make every applied repair visible.
- The single-pass designers keep a narrower shape than the three-pass pipeline: `design`/`design_offline` are the offline basic design plus a legacy `branch` node (`title` `Decide`, `config.conditions = ["default"]`), while the branch pass uses `["condition", "default"]`.
- Per-node-type DL validation is a pure report (`DlValidation`) beside the repairing `reflect`, so the pipeline's graph repair keeps its `passed` semantics and Python's field checks stay model-free and non-mutating.
- The DL layer is separate from the simplified `WorkflowDsl`: `DlDocument` mirrors Python's `SimpleIR`, `PlatformWorkflow` mirrors the platform conversion, and `WorkflowDsl` stays the design-stage document; `PlatformWorkflow::to_workflow_dsl` bridges the two.
- The DL transformer reproduces Python's uuid4 intent ids (an explicit `intent_id` is kept verbatim), so conversion is intentionally non-deterministic per call and only the intent ids differ between two conversions of the same document.
- Plugin parameters resolve resource → node config → field default, which is the closest match to Python's platform lookup without a platform client.
- Mermaid output follows Python's grammar byte-for-byte, including the multi-successor gate on labelled edges; only the `When condition met` label extraction deviates, because Python's regex captures a single character and the port keeps the intent (the bracketed text).
- Session state is a private `SessionState`, so the public surface cannot corrupt a session's history invariants; `AgentBuilder` derives `Clone` (sessions are copied) and implements `Debug` manually because `dyn Model` is not `Debug`, while stage types implement `Debug` as `has_model: bool`.
- `HistoryManager::new(0)` uses the default capacity of `20`, while `get_latest_k(0)` returns no message.
- `BuildState` adds `Clarifying` and `Requesting`, which Python only models as status strings beside its state enum; `map_state_to_status` reports the four reachable strings.
- The bridge is total and infallible for card and configuration: an absent provider falls back to `"openai"` and absent constraints to `ConstrainConfig::default()`; only `build_llm_agent` can fail.

## Deferred

- Python's `ProgressReporter`/`BuildProgress`/`ProgressStep`/`ProgressManager` and its process-global `progress_manager` singleton are not ported: there is no mutable per-session progress state, callback fan-out, or wall-clock progress timestamp, because progress here is a pure function of `(agent_type, stage)` observed through `BuildStatus::progress` and the `PipelineEvent` stream.
- Python's platform resource retrieval and preprocessing (plugins, knowledge bases, prompt templates, `collect_plugin`) has no counterpart: `transform_with_resources` takes caller-supplied `PluginResource`s and the DL generator performs no asset search, because the crate has no platform client.
- Variable `type` validation is not ported: Python's `available_variable_types` set and its optional `check_type` flag are absent, so a `type` key is never checked.
- Python's Markdown-section designer prompts and its `## New Workflow Design` extraction are replaced by one JSON contract per pass, so Python's exact prompt text and Markdown parsing are not reproduced.
- Python's clarifier runs two model round trips (basic-element extraction plus resource planning); this port answers with a single verdict and asks the missing detail, so its factor extraction and resource plan are not reproduced.
- No streaming variant exists for the DL generator, the DL reflector, or `DlGenerator::refine`; only the clarifier dialogue, the three designer passes, and the whole pipeline stream.
- Python's `main.py` and `run_eval_query.py` drivers for the builder are not ported; the CLI lives in `openjiuwen-cli`.
- `openjiuwen-dev-tools/tests/` integration coverage for this module is limited to the public seams (`agent_assets.rs`, `judge_and_pipeline.rs`, and `deferred_surface.rs` for the workflow-DSL validation contract); internal behavior is tested in-tree.

## Test Strategy

- Unit tests live beside the code: `outcome`, `clarifier` (the single-pass verdict), `dl_generator`, `mermaid`, and `mermaid/labels` each own an inline `#[cfg(test)] mod tests`; `designer`, `designer/passes`, `designer/stream`, `intention`, `builder`, `dl_reflector`, and `dl_transformer::convert` declare sibling modules.
- `builder.rs` declares three suites: `builder/tests.rs` (`mod tests`), `builder/pipeline_tests.rs` (`mod pipeline_tests`), and `builder/stream_tests.rs` (`mod stream_tests`).
- `dl_reflector/tests/mod.rs` groups six files: `support` (fixtures), `graph` (every graph repair rule), `repair` (`passed` semantics, reflection vocabulary, model-revision accept/reject), `node_rules` (basic and per-node-type field validation), `branch_rules` (intent and branch conditions), and `variable_rules` (duplicate names, input values, references, output descriptions).
- `dl_transformer/convert/tests/mod.rs` groups eight files: `support`, `basics` (node types, ids, positions, titles), `branches`, `intents` (rendered ports, generated vs explicit ids, per-conversion id drift), `defaults` (questioner/code/output), `plugins` (resource/config/field-default precedence), `variables` (`_of_` splitting, references, output schemas), and `serialization` (null-dropping, DSL validation, error paths).
- A test module in a sibling file opens with `use super::*;`, so it reaches the parent module's private items exactly as an inline module would; directory suites reach the grandparent through `use super::super::*;` and fixtures through `use super::support::*;`.
- `outcome`: state/status mapping over every pair, wire names and default tolerance, each `BuildOutput` variant, the flattened `BuildResult`, provenance record/aggregate helpers, `to_json` appending `provenance` only when degraded and `repair` only when non-empty, and round trips of both records.
- `clarifier` and `clarifier/rounds`: blank validation before any model call, ready/missing-signal questions, history-carried signals, model JSON path with message-order assertions, garbage/error fallbacks, the four terminations and their order, answer accumulation across rounds with an unmutated caller history, provenance aggregation, `clarify` matching the one-round dialogue, streaming events (one `Round` per round then `Completed`, one error for a blank query), config normalization, wire names, and JSON round trips including a legacy verdict without provenance.
- `intention`: deterministic initial rules (sequence markers, numbered steps, workflow keywords), retained user turns counting toward the process rule, the refine rule needing both a diagram and a modification keyword, blank validation with zero model calls, model answers overriding the rule with recorded detail, the diagram reaching the refine prompt, fallback for every unusable answer, and wire-name round trips.
- `designer` and `designer/passes`: offline single/multi-step synthesis, the 4-step cap, the legacy branch, CJK input, name slugging, blank validation, the model path with aliases, invalid/garbage/error fallbacks, `DesignNodeKind` alias matrix, `to_workflow_dsl` mapping, and every `validate` rejection; the pass suite covers each deterministic pass (including no model call when offline), branch-marker classes, branch idempotence and detail, every reflection repair with one note per repair, an unrepairable design, three-pass record order and determinism, per-pass model acceptance and rejection including a branch-marker answer without a branch node, and JSON round trips of `DesignPass`/`ThreePassDesign`.
- `designer/stream`: every pass reported in order, parity with `design_three_pass`, `Send + 'static`, blank-query rejection, per-pass provenance, fallbacks, and externally tagged serialization.
- `builder/tests`: legacy single-round-trip builds, metadata stamping, session isolation/clearing, `history(k)`, plain-text clarifications, blank validation, workflow completion and fallback, propagated model failures, clock use, legacy provenance (model and fallback), and `record_result` persisting turns in pairs.
- `builder/pipeline_tests`: the full workflow path with call-count and prompt assertions (5 calls), the deterministic fallback for every stage, a clarification-only request (1 call), LLM-agent repair and absorbed model failure, blank validation, reproducible fallbacks, the legacy entry point, provenance recording every stage, the repair loop reaching a fixed document (3 attempts, 2 notes, 7 calls) and exhausting its budget without error, the second-turn refine path (intention + refinement, stored artifact replaced), a first-turn refine intent being ignored, and model-vs-fallback provenance for an LLM agent.
- `builder/stream_tests`: stage order, parity with the buffered build, one `Repaired` event per attempt, a blank query yielding a single validation error, and `Send + 'static` with sessions left untouched.
- `dl_generator`: deterministic parity with `to_workflow_dsl`, `from_spec` valid/invalid, model success assertions, invalid/garbage/error fallbacks, no-model passthrough, and `refine`'s blank-feedback rejection, deterministic no-model path, accepted revision, and every fallback reason.
- `dl_reflector`: every graph repair and repaired-document validation, `passed` semantics, valid-document passthrough, determinism, serde round trips, model revision accepted only with the same id set, all rejection paths, `validate_dl`'s valid multi-node document, one failing fixture per node-type and variable rule, declaration order, dangling links, determinism, `DEFAULT_MAX_RETRIES`, and issue-kind wire names.
- `mermaid` and `mermaid/labels`: byte-exact `graph TD` output, `Node{id}: {description}` lines, description-over-title and blank fallbacks, quote/backtick escaping, multi-successor labelled edges vs single-successor unlabelled ones, `When condition met` label extraction (bracketed, unbracketed, one leading separator, nested brackets, unbalanced fallback) on labelled edges only, id sanitizing/deduplication, dangling endpoints, determinism/document order, no trailing newline, empty document, and design/DSL equivalence.
- `dl_transformer`: DL round trip and structural validation, `DlNodeType` wire/numeric ids and unknown rejection, one full conversion fixture per node type, `_of_` splitting, constant-vs-ref inputs and index counters, plugin/questioner/branch defaults and operator/logic maps, code exception config, intent determinism with explicit ids and id-only drift without them, null-dropping, error paths, and `to_workflow_dsl` validation.
- Every test is offline and deterministic: `openjiuwen_test_support::ScriptedModel` with queued turns and recorded calls, no network, no environment reads, no wall-clock or generated-id assertions beyond non-emptiness.

## Routing Table

- `./dsl.rs` → serialized LLM-agent document and its model/plugin/workflow/dependency blocks, with unknown keys preserved
- `./workflow_dsl.rs` → serialized workflow document, node/edge models, and structural validation
- `./spec.rs` → programmatic `LlmAgentSpec`/`WorkflowSpec` and their DSL conversion
- `./bridge.rs` → DSL-to-kernel mapping: `AgentCard`, `LlmAgentConfig`, `build_llm_agent`
- `./history.rs` → bounded `HistoryManager` and `DialogueMessage`
- `./outcome.rs` → `BuildState`, `BuildStatus`, `BuildOutput`, `BuildResult` with the conditional `to_json`, plus the provenance vocabulary (`GenerationSource`, `StageProvenance`, `GenerationProvenance`, `RepairReport`)
- `./builder.rs` → session-based `AgentBuilder`, `AgentType`, the legacy single-round-trip build, `build_with_pipeline`, `build_with_pipeline_stream`, `record_result`, deterministic LLM-agent repair, prompts, and `now_millis`
- `./builder/pipeline.rs` → the non-streaming driver `run`, the LLM-agent generation helper, workflow generation with baseline-based source inference, clarification text, issue notes, and repair feedback
- `./builder/stream.rs` → `pipeline_stream` (the pipeline implementation) and `PipelineEvent`
- `./builder/tests.rs` → `builder`'s `mod tests`: legacy builds, result vocabulary, provenance, `record_result`
- `./builder/pipeline_tests.rs` → `builder`'s `mod pipeline_tests`: stage order, call counts, fallbacks, repair loop, refine path
- `./builder/stream_tests.rs` → `builder`'s `mod stream_tests`: event order and parity, `Repaired` events, `Send + 'static`, untouched sessions
- `./intention.rs` → `IntentionDetector`/`IntentionVerdict`/`BuildIntent`, the deterministic initial/refine rules, and the two model contracts
- `./intention/tests.rs` → `intention`'s `mod tests`: rules, history contribution, model answers, fallbacks, wire names
- `./clarifier.rs` → `Clarifier`, `Clarification`, the model contract and keyword heuristic, plus the re-exported round types
- `./clarifier/rounds.rs` → the bounded dialogue: `ClarifyConfig`, `ClarificationRound`, `ClarificationTermination`, `ClarificationDialogue`, `ClarifyEvent`, and the stream
- `./clarifier/rounds/tests.rs` → `clarifier::rounds`' `mod tests`: terminations, accumulation, provenance, streaming
- `./designer.rs` → typed workflow design, `DesignNodeKind` parsing, validation, design→DSL conversion, `WorkflowDesigner` single-pass/offline/three-pass/stream entry points, `DesignOutcome`, and the shared clause limits
- `./designer/passes.rs` → the three pass drivers, `DesignPassKind`/`DesignPass`/`ThreePassDesign`, deterministic basic/branch/reflection rules, and model acceptance per pass
- `./designer/passes/clauses.rs` → private clause splitting (numbered, terminator, sequencing), list-marker stripping, step titles, slugify, truncation
- `./designer/passes/tests.rs` → `designer::passes`' `mod tests`: per-pass deterministic and model behavior, repairs, round trips
- `./designer/stream.rs` → `DesignEvent` and the streamed three-pass pipeline
- `./designer/stream/tests.rs` → `designer::stream`'s `mod tests`: event order, parity, blank query, provenance
- `./designer/tests.rs` → `designer`'s `mod tests`: offline synthesis, single-pass model path and aliases, validation, design→DSL conversion
- `./dl_generator.rs` → `DlGenerator` deterministic/model generation, spec conversion, `refine` and `Refinement`
- `./dl_reflector.rs` → `DlReflector`, graph-shape repair with issue reporting, optional model revision, `DEFAULT_MAX_RETRIES`, and `validate_dl`
- `./dl_reflector/dl_checks.rs` → private per-node-type validation helpers (basic checks, required fields, intent/branch expressions, references, default branches, variable checks)
- `./dl_reflector/tests/` → `dl_reflector`'s `mod tests` directory module: `graph`, `repair`, `node_rules`, `branch_rules`, `variable_rules`, and fixtures in `support`
- `./dl_transformer/` → Python `SimpleIR` DL model, the platform workflow document, and the DL→platform-DSL transformer
- `./dl_transformer/model.rs` → `DlDocument`/`DlNode`/`DlParameters`/`DlVariable`/`DlCondition`/`DlNodeType` and structural validation
- `./dl_transformer/platform.rs` → `PlatformWorkflow`/`PlatformNode`/`PlatformEdge`, recursive null-dropping, and `to_workflow_dsl`
- `./dl_transformer/convert.rs` → `DlTransformer` driver, `PluginResource`, variable/output builders, reference parsing, edge construction, intent-id resolution
- `./dl_transformer/convert/nodes.rs` → per-node-type `data` builders (start/end/llm/branch/intent/questioner/plugin/code/output)
- `./dl_transformer/convert/tests/` → conversion tests as a directory module: `basics`, `branches`, `intents`, `defaults`, `plugins`, `variables`, `serialization`, and fixtures in `support`
- `./cycle_checker.rs` → deterministic `find_cycle`/`has_cycle` over the workflow graph
- `./mermaid.rs` → `render_mermaid`/`render_design` in Python's `graph TD` grammar, the multi-successor edge gate, and the re-exported `when_condition_label`
- `./mermaid/labels.rs` → id sanitizing/deduplication, endpoint resolution, label escaping, `when_condition_label`, and `edge_label`
- `./progress.rs` → `ProgressStage`, the LLM-agent/workflow weight tables, and `progress_for`
- `./utils.rs` → fence extraction, dialogue-history formatting, first-wins dict-list merging, and session-id validation
- `./mod.rs` → module tree, crate-level re-exports, and the pipeline overview
- `../util.rs` → shared `extract_json_fence` and text helpers (sibling, read-only — escalate writes to the parent)
- `../../openjiuwen-agent/` → `AgentCard`, `LlmAgentConfig`, `LlmAgent`, `ReActAgent` consumed by the bridge (sibling crate, read-only — escalate writes to the parent)
