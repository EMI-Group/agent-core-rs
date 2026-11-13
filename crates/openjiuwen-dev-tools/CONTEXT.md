# openjiuwen-dev-tools

## Intent

Developer tooling for authoring, generating, evaluating, and tuning openJiuwen artifacts: prompts, skills, agents, and workflows.
Maps the Python package `openjiuwen.dev_tools` (`prompt_builder/`, `skill_creator/`, `agent_builder/`, `skill_evaluator/`, `tune/`).
The crate is a layer-5 consumer of the kernel: it builds on `openjiuwen-foundation` prompts and `Model`, on `openjiuwen-agent` cards/configs, and on `openjiuwen-common` errors and HTTP transport.
Audience: tool authors writing Rust programs (and the CLI/dev tooling above this crate) that generate, validate, evaluate, and optimize agent assets.
Python's dev tools are LLM-orchestration scripts with little typed surface; the Rust port keeps the observable artifacts (prompt text, skill bundles, agent DSL/configs, workflow designs, score reports, tuning traces) and replaces free-form LLM contracts with typed models, deterministic cores, and an optional `Model` seam.

## API Surface

Crate root `openjiuwen_dev_tools` with one module per Python subpackage plus a shared scoring module:

- `judge` — rubric-driven LLM judgement shared by `skill_evaluator` and `tune`: `Rubric`/`RubricDimension` (weights, scale, validation, weighted `normalize`), `Verdict`/`VerdictSource` (tolerant structured parsing of fenced JSON, tag sections, bare JSON; percent strings; provenance), `prompt::{system_prompt, render_rubric, rubric_messages}`, `ModelJudge` + `Judge`/`JudgeRequest`, and `judge_with_fallback` returning a deterministic verdict when the model call or parsing fails.
- `prompt_builder` — `MetaTemplateBuilder` (task description → standardized prompt), `FeedbackPromptBuilder` (feedback → rewritten prompt), `BadCasePromptBuilder` (evaluation cases → repaired prompt); bundled `zh-CN`/`en-US` prompt banks, deterministic formatting/parsing helpers.
- `skill_creator` — `SkillDefinition` (name/description/instructions/resources/metadata), `SkillCreator` (create from `SkillSpec` or from a `Model`, template rendering, bundle write/read, `create_from_url`), `SkillBundle` (harness-compatible `SKILL.md` + resources layout), and `skill_creator::omni` — the `skill_omni_creation` pipeline: `Fetcher` seam (`HttpFetcher` over `openjiuwen_common::http`, `ScriptedFetcher`), rule-based `HtmlDocument`/`resolve_url`, and `OmniCreation` stages scrape → download → generate → save → `OmniOutcome`.
  `skill_creator::omni::archive` is the archive layer: `ArchiveFormat`/`ArchiveEntry`, `detect`, `ArchiveUnpacker::{unpack, unpack_auto}` for tar, gzip (including gzip-wrapped tar) and a stored+DEFLATE ZIP reader, `ArchiveWriter::write_stored`, in-crate inflate and CRC-32, path safety, and per-entry/per-archive uncompressed-size caps.
  The rest of `omni` is the in-crate `sha256` helper (`digest`/`hex_digest`), the Python-parity image extension/MIME allow-lists (`SUPPORTED_EXTS`/`SUPPORTED_MIMES`, `image_extension`, `sniff_image`, `filter_images*`, `slugify`, `dedupe_links`), image persistence at `references/img_NN.<ext>` with `saved_path`/`sha256` provenance, `strip_hallucinated_images` against those persisted paths, and exactly one `## Reference Files` block per generated document.
- `agent_builder` — programmatic `LlmAgentSpec`/`WorkflowSpec` and serialized platform DSL (`LlmAgentDsl`, `WorkflowDsl`), the `to_agent_card`/`to_llm_agent_config`/`build_llm_agent` bridge into `openjiuwen-agent`, `HistoryManager`, and the session-based `AgentBuilder`.
  The staged pipeline is composed of `Clarifier`, `WorkflowDesigner`/`WorkflowDesign` (with `to_workflow_dsl`), `DlGenerator` (deterministic plus model), `DlReflector`/`Reflection` for graph repair plus `DlReflector::validate_dl` for Python's per-node-type DL rules, `cycle_checker::{find_cycle, has_cycle}`, `mermaid::{render_mermaid, render_design}` in Python's grammar (`graph TD`, `  {id}[Node{id}: {label}]`, and `-- label -->` only on a multi-successor source), and `AgentBuilder::build_with_pipeline` (clarify → design → generate → reflect → validate).
  `dl_transformer` is the DL→platform-DSL conversion layer (`DlDocument`/`DlNode`/`DlNodeType` → `PlatformWorkflow`): numeric node-type ids, `meta.position` steps of +20, `_of_` input/output variable splitting, plugin/questioner/branch/code defaults, recursive null-dropping, and `PlatformWorkflow::to_workflow_dsl`.
  `utils` (`extract_json_from_text`, `format_dialog_history`, `merge_dict_lists`, `validate_session_id`) and `progress` (`ProgressStage`, the two stage-weight tables, `progress_for`) carry the Python helper modules.
- `skill_evaluator` — `EvalCase`, deterministic `Assertion`/`CaseRunner` seams, `DesignReport`/`FunctionalReport`/`SafetyReport`/`FinalScore` models with the Python aggregation (`0.40·design + 0.60·pass_rate`, safety gate `0.80`), and `SkillEvaluator`.
  `judge_scoring` adds `design_rubric`/`safety_rubric`, `SkillJudge`, and `SkillEvaluator::with_judge`/`evaluate_with_judge` with deterministic fallback and `JudgeRecord` provenance.
  `eval` is the runnable harness: `load_cases_json`/`load_cases_file` (JSON array, `{"cases": […]}` object, single object, JSONL), `EvalAgent`/`ModelAgent`/`CaseRunnerAgent`, `EvalHarness::{run, run_query}`, `EvalRun::{to_json, to_json_string, to_jsonl, write_json, write_jsonl}`, and `QueryRequest`/`QueryOutcome` mirroring `run_eval_query.py`'s output-path resolution.
  `artifacts` writes the four Python JSON files — `llm_judge_score.json`, `skill_safety_score.json`, `skill_tester_report.json`, and `final_score.json` (skipped unless the pipeline completed) — through `ArtifactPaths`, the per-file `EvaluationReport::write_*_report` methods, and `write_artifacts`.
  Units: every `*_pct`, `contribution`, `pass_rate`, and gate threshold (`gate_threshold`, `safety_gate_threshold`, `DESIGN_GATE_THRESHOLD`) is a 0.0–1.0 fraction, while `design_score`, `design_score_max`, and `ScoreDimension::{score, max}` stay 0–100 points.
  Status tokens are `COMPLETED`, `BLOCKED_AT_E1`, and `BLOCKED_AT_E3`, gate roles are `GATE_E1`/`GATE_E2`/`GATE_E3` (a completed run reports only `e3_safety`), and `SafetyReport::auto_fail` is an `Option<AutoFailCode>` over AF1–AF5.
- `tune` — `Case`/`EvaluatedCase`/`CaseLoader` (Python field names), `Case::from_text` for the `PromptTaskCase` shape, the evaluator seam (`PromptRunner`/`ModelPromptRunner`/`ScriptedPromptRunner`, `CaseScorer`/`LabelScorer`, `CaseEvaluator`/`RunnerEvaluator`/`ModelEvaluator`/`DeterministicEvaluator`), `OptimizerConfig`, `OptimizationResult`, the four real searches (`InstructionOptimizer` beam search, `ExampleOptimizer` greedy example selection, `JointOptimizer` alternating ascent, `PromptSearchOptimizer` pool sweep), `Trainer`, and `optimize_prompt`.
  The `tool` layer covers tool-aware evaluation: `ToolStub`/`ToolStubs`, `ToolCallRecord`/`ToolTrace`, `RunOutcome`, and `ToolUsageScorer`, with `DEFAULT_MAX_TOOL_ROUNDS` bounding the model tool loop.
  The traced surface is `PromptRunner::run_traced`, `CaseScorer::score_with_trace`, `CaseEvaluator::evaluate_traced`/`score_cases_traced`/`tool_usage`, and `Trainer::evaluate_traced` → `EvaluationSummary` (`mean_score`, `results`, `tool_usage`); optimizer prompts render the case set's declared tools as a `<tools_description>` block.

Cross-module rules: `Case`/`EvaluatedCase` are owned by `tune` and consumed by `prompt_builder` (`BadCasePromptBuilder`) — never duplicated; rubric scoring lives in `judge` and is consumed, never re-implemented, by `skill_evaluator` and `tune`.
Tool descriptors use `openjiuwen_foundation::tool::ToolInfo` (the wire `{"type":"function","function":{…}}` shape); Python's flat `ToolInfo` is not reproduced.

## Constraints

- Only this crate may be edited here; workspace manifest, sibling crates, and `docs/` are out of scope.
- No new dependencies: HTTP is `openjiuwen_common::http` (workspace `reqwest` stack); HTML, tar, zip, gzip, DEFLATE, CRC-32, SHA-256, image sniffing, and URL resolution are implemented in-crate.
- Async-first: builders, optimizers, trainers, and the harness are `async fn`; streams return `BoxStream<'static, Result<String>>`; no `block_on`/`asyncio.run` bridging — synchronous Python entry points become `async` (documented deviation, e.g. `Trainer::train`).
- Errors: `openjiuwen_common::error::{Error, Result}` only; invalid user input is `Error::validation`, missing capability is `Error::unsupported`, model failures propagate as `Error::llm`.
- Model failures never silently degrade: an optional model path falls back to a documented deterministic path (never an empty result), and the fallback is visible (returned value, `VerdictSource`, or `used_fallback`).
- Serialization: public data types are `serde` (`rename_all = "snake_case", default`), unknown-field tolerant; DSL, report, `Case`, and `OptimizationResult` JSON keep the Python field names.
- No `unsafe`; `tracing` logging via `openjiuwen_common::logging`; no `println!`/`eprintln!`; no panics on user input.
- Files ≲1000 lines; one responsibility per file; shared helpers within a module, not duplicated across modules.
- Tests are offline and deterministic: `openjiuwen_test_support::ScriptedModel` for model-driven paths, `TempWorkspace` for filesystem bundles, `ScriptedFetcher`/loopback `TcpListener` for HTTP, fixed ids/timestamps — no wall-clock reads, no environment reads.
- Empty prompts, blank skill names, unknown template names, malformed frontmatter, empty rubrics, and out-of-range tuning parameters are validation errors, never silent fallbacks (Python's "warn and degrade" behavior is a documented deviation).

## Design Decisions

- Python builders own a `Model`; here every builder, optimizer, judge, and stage takes an optional `Arc<dyn Model>` (or `ScriptedModel` in tests), so configuration stays with the caller and nothing reads the environment.
- Deterministic cores are public: template rendering, message assembly, JSON-fence/tag parsing, skill frontmatter rendering/parsing, agent/workflow DSL transformation, Mermaid rendering, cycle detection, HTML/archive parsing, label comparison, and evaluation scoring are ordinary functions testable without a model; the model only supplies text.
- Every model-driven stage is validated structurally before it is accepted (DSL must `validate`, a design must pass `WorkflowDesign::validate`, a skill definition must `validate`, a judgement must parse); an unusable answer falls back to the deterministic path instead of erroring.
- `judge` is the single scoring LCA: `Rubric::normalize` is the only weighted-score implementation, `Verdict::parse` is the only verdict parser, and `ModelJudge::judge_with_fallback` is the only model→deterministic fallback contract.
- `tune`'s search is deterministic by construction: fixed mutation banks, ordered candidate generation, seeded shuffle, and a total tie-break (score desc → earlier candidate index → lexicographically smaller prompt or `case_id`). `DeterministicEvaluator` scores placeholder preservation, instruction structure, case term coverage, and degenerate-prompt hygiene so an offline run has a real gradient; `ModelPromptRunner` plus `LabelScorer` replace it when a model is available.
- Tool awareness stays additive in `tune`: a `ToolTrace` travels only through the traced methods and `EvaluationSummary`, `EvaluatedCase` gains no field, and an untraced evaluation scores exactly as before.
- `skill_evaluator` keeps one unit convention: fractions for every ratio and threshold, points only where Python reports points, and `round4`/`fraction_from_points` as the only scale conversions.
- Optimizer configuration is explicit (`OptimizerConfig`) and validated against the Python `TuneConstant` bounds; `optimize_prompt` is the no-config convenience over `InstructionOptimizer`.
- `agent_builder` keeps Python's two artifacts (platform DSL JSON and kernel config) and composes them; the legacy one-round-trip `build_agent`/`build_agent_at` stays available, while `build_with_pipeline` runs the staged clarifier → designer → generator → reflector pipeline with a deterministic fallback at every stage.
- `skill_evaluator`'s functional score stays assertion-based; the LLM judge covers design and safety only, and a `scan_safety` finding always overrides a passing model verdict (hard gate).
- `skill_creator::omni` keeps fetch, parsing, and generation separate: `Fetcher` is injected, HTML parsing is rule-based and tolerant, and the skill document is generated deterministically unless a model is supplied.

## Test Strategy

- Unit tests inside each module (`#[cfg(test)]`) cover deterministic transforms, validation errors, fallbacks, model-driven paths with `ScriptedModel`, and serialization round-trips; the `judge`, `tune::evaluator`, `omni/html`, `omni/pipeline`, `skill_evaluator::eval`, `skill_evaluator::artifacts`, and `agent_builder` modules keep their larger suites in sibling `tests.rs` files.
- The archive layer is tested per codec: `omni/archive/{zip,gzip,inflate}` keep `tests.rs` siblings while `tar.rs` keeps its own module tests, covering the format matrix, the stored-ZIP round trip through `ArchiveWriter::write_stored`, DEFLATE golden and dynamic streams, gzip members/CRC-ISIZE verification/trailing-garbage rejection, truncated-input validation errors, and the size caps.
- Tool-aware tuning is covered by `tune/tool.rs` (stub lookup, records, trace coverage/missing/usage) and `tune/evaluator/tests.rs` (the model tool loop, `ToolUsageScorer`, `evaluate_traced`, and the `EvaluationSummary` round trip).
- `skill_evaluator/artifacts.rs` covers the four artifact writers: exact key sets per file, re-read JSON, the skipped blocked `final_score.json`, nested-directory creation, and the non-directory rejection.
- `agent_builder/dl_transformer/convert/tests.rs` covers the DL→platform conversion (per-node-type data, ids/positions, null-dropping, `to_workflow_dsl`), and `agent_builder/dl_reflector/tests.rs` covers `validate_dl` per node type beside the graph-repair rules.
- Model-driven paths assert both the result and the request: queued `ScriptedModel` turns plus recorded call inspection.
- `crates/openjiuwen-dev-tools/tests/skill_lifecycle.rs` covers the `skill_creator`→`skill_evaluator` seam: author→bundle write/read→frontmatter re-parse in a `TempWorkspace`, then deterministic and `ModelCaseRunner` evaluation with the Python report JSON names, fraction-unit scores, the E3 safety gate, and the `BLOCKED_AT_E3` status.
- `crates/openjiuwen-dev-tools/tests/prompt_assets.rs` covers the `tune`→`prompt_builder` seam: Python-field `Case`/`EvaluatedCase` JSON feeding the bad-case, meta (`Plan`), and feedback (`Insert`) builders, plus buffered-vs-streamed parity.
- `crates/openjiuwen-dev-tools/tests/agent_assets.rs` covers the `agent_builder`→`openjiuwen_agent` seam: spec→DSL JSON→`LlmAgent`, workflow-spec validation, and session build/clarify/clear flows with the flattened `BuildResult` keys.
- `crates/openjiuwen-dev-tools/tests/tuning_pipeline.rs` covers the tuning workflow end-to-end: loader→split→optimizers→trainer, determinism, and the public evaluator seam.
- `crates/openjiuwen-dev-tools/tests/omni_bundle.rs` covers URL→bundle through `ScriptedFetcher`→`SkillCreator::create_from_url`→`SkillBundle::read`, plus the stored-ZIP round trip and the tar/ZIP/gzip detection and truncation matrix.
- `crates/openjiuwen-dev-tools/tests/judge_and_pipeline.rs` covers the judge→evaluator provenance path, the `EvalHarness` JSON/JSONL reports, and `AgentBuilder::build_with_pipeline` (workflow, clarification, and the Mermaid node/edge grammar including a labelled branch).
- `crates/openjiuwen-dev-tools/tests/deferred_surface.rs` pins the workflow-DSL structural validation contract.
- No network, no environment reads, no wall-clock dependence: ids and timestamps are parameters or fixed constants.

## Deferred

- Archive coverage stops at Python's supported subset: ZIP64, encrypted ZIP entries, ZIP compression methods other than stored (0) and DEFLATE (8), and gzip methods other than DEFLATE (8) are typed `Error::Unsupported`; the ZIP writer is stored-only, so nothing here writes compressed ZIP entries.
- Python `skill_omni_creation` Playwright/JS rendering, stealth/cookie handling, Pillow image decoding/size gates, and the multimodal image-relevance judge — replaced by static-DOM extraction, byte/format filters, and an optional text model; a browser/vision stack is out of scope.
- Agent-builder pipeline extras: intention detector, multi-turn clarifier dialogue, three-pass (basic/branch/reflection) designer prompts, DL refine mode, reflection feedback loop into the designer, platform resource retrieval, and streaming variants — replaced by single-pass stages with deterministic fallbacks; the CLI lives in `openjiuwen-cli`.
- Agent-builder conversion gaps: `DlTransformer` generates deterministic intent ids instead of Python's uuids and reads the branch operator as the second whitespace token, `transform_with_resources` takes caller-supplied `PluginResource`s instead of fetching platform assets, `validate_dl` skips Python's duplicate input/output-name and input-placeholder checks, the repair loop is caller-driven (`DEFAULT_MAX_RETRIES` is exposed but nothing retries), Mermaid does not port the `When condition met[...]` label extraction, and `utils`/`progress` keep only the deterministic helpers and weight tables (no `ProgressReporter` singleton or callbacks).
- `tune` dataset samplers, `tune/chat_agent` agent adapters, and the Python `optimizer/adopt`/`prompt_search` policy/memory/reward/drift services — the port keeps cases in memory and implements the observable propose→evaluate→best contract as stateless searches; the LLM-judge retry loop becomes `LabelScorer` plus `judge`.
  Concretely unported from `prompt_search`: the task/candidate/record models, `CompositeReward` (latency/token/cost/completeness/structured-validation components, min-max normalization, correctness gate, overfit and drift penalties), `ConvergenceDetector`, the Jaccard/JSONL prompt memory and history compressor, the LLM policy seam, and `ParameterSearcher`; the port also has no `LLMCall`/`TextualParameter` binding layer, so optimizers take prompt strings and `Case`s instead of agent parameters.
- `skill_evaluator`'s Python `create_subagent` ReAct/tool-execution orchestration and environment-driven model construction — replaced by the `EvalAgent` seam with caller-supplied configuration; functional scoring stays assertion-based (LLM judging covers design and safety).
- The human-readable eval reports (`llm_judge_report.md`, `final_report.md`, `pipeline_report.md`, `skill_safety_report.md`) are not written — only the four machine-readable JSON artifacts — and `FunctionalReport::timestamp` stays epoch milliseconds rather than an ISO-8601 string.
- `openjiuwen-dev-tools/tests/` integration coverage is limited to public-API seams; module internals are tested in-tree.

## Routing Table

- `./src/judge/` → rubric model, verdict parsing, judge prompt rendering, `ModelJudge` and the fallback contract (`judge/CONTEXT.md`)
- `./src/prompt_builder/` → prompt template banks, meta/feedback/bad-case builders, deterministic prompt helpers (`prompt_builder/CONTEXT.md`)
- `./src/skill_creator/` → skill definition model, validation, templating, `SKILL.md` bundle read/write, model-assisted creation, `create_from_url` (`skill_creator/CONTEXT.md`)
- `./src/skill_creator/omni/` → fetcher seam, HTML extraction, tar/ZIP/gzip unpacking with the stored-ZIP writer, SHA-256, image/link filtering, image persistence, the scrape→download→generate→save pipeline (`skill_creator/omni/CONTEXT.md`)
- `./src/agent_builder/` → agent specs, platform DSL, kernel bridge, session builder, clarifier/designer/DL generator/reflector (graph repair plus per-node-type `validate_dl`), DL→platform transformer, cycle checker, Python-grammar Mermaid, `utils`/`progress` helpers (`agent_builder/CONTEXT.md`)
- `./src/skill_evaluator/` → evaluation cases, assertions, reports and the fraction unit convention, deterministic scoring, judge rubrics/provenance, the eval harness, the four artifact writers (`skill_evaluator/CONTEXT.md`)
- `./src/tune/` → `Case`/`EvaluatedCase`/`CaseLoader`, the tool-stub layer and traced evaluator seam, prompt optimizers, trainer (`tune/CONTEXT.md`)
- `./src/lib.rs` → module tree, crate lints, module-level docs
- `./src/util.rs` → shared model-output parsing (JSON/list fences, tag extraction, truncation)
- `./tests/` → cross-module public-API integration suites (skill lifecycle, prompt assets, agent assets, tuning pipeline, omni bundle with the archive matrix, judge/pipeline with the Mermaid grammar, deferred surface)
- `./Cargo.toml` → crate manifest; `openjiuwen-test-support` is a dev-dependency only
- `../openjiuwen-foundation/` → `PromptTemplate`, `BaseMessage`, `Model`, `ToolInfo` (sibling — read-only, escalate writes to parent)
- `../openjiuwen-agent/` → `AgentCard`, `LlmAgentConfig`, `LlmAgent`, `ReActAgent` (sibling — read-only, escalate writes to parent)
