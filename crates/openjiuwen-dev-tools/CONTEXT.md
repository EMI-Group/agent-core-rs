# openjiuwen-dev-tools

## Intent

Developer tooling for authoring openJiuwen artifacts: prompts, skills, agents.
Maps the Python package `openjiuwen.dev_tools` (`prompt_builder/`, `skill_creator/`, `agent_builder/`, `skill_evaluator/`, `tune/`).
The crate is a layer-5 consumer of the kernel: it builds on `openjiuwen-foundation` prompts and `Model`, on `openjiuwen-agent` cards/configs, and on `openjiuwen-common` errors.
Audience: tool authors writing Rust programs (and the CLI/dev tooling above this crate) that generate, validate, and evaluate agent assets.
Python's dev tools are LLM-orchestration scripts with little typed surface; the Rust port keeps the observable artifacts (prompt text, skill bundles, agent DSL/configs, score reports) and replaces free-form LLM contracts with typed models plus optional `Model`-driven generation.

## API Surface

Crate root `openjiuwen_dev_tools` with one module per Python subpackage:

- `prompt_builder` — `MetaTemplateBuilder` (task description → standardized prompt), `FeedbackPromptBuilder` (feedback → rewritten prompt), `BadCasePromptBuilder` (evaluation cases → repaired prompt); bundled `zh-CN`/`en-US` prompt banks, deterministic formatting/parsing helpers.
- `skill_creator` — `SkillDefinition` (name/description/instructions/resources/metadata), `SkillCreator` (create from `SkillSpec` or from a `Model`, template rendering, bundle write/read), `SkillBundle` (harness-compatible `SKILL.md` + resources layout).
- `agent_builder` — programmatic `LlmAgentSpec`/`WorkflowSpec` and serialized platform DSL (`LlmAgentDsl`, `WorkflowDsl`), the `to_agent_card`/`to_llm_agent_config`/`build_llm_agent` bridge into `openjiuwen-agent`, `HistoryManager`, and the session-based `AgentBuilder`.
- `skill_evaluator` — `EvalCase`, deterministic `Assertion`/`CaseRunner` seams, `DesignReport`/`FunctionalReport`/`SafetyReport`/`FinalScore` models with the Python aggregation (`0.40·design + 0.60·pass_rate`, safety gate `0.80`), and `SkillEvaluator`.
- `tune` — `Case`/`EvaluatedCase`/`CaseLoader` (ported, pure) plus API-shaped stubs (`Error::Unsupported`) for optimizers, `Trainer`, and `optimize_prompt`.

Cross-module rule: `Case`/`EvaluatedCase` are owned by `tune` and consumed by `prompt_builder` (`BadCasePromptBuilder`) — never duplicated.
Tool descriptors use `openjiuwen_foundation::tool::ToolInfo` (the wire `{"type":"function","function":{…}}` shape); Python's flat `ToolInfo` is not reproduced.

## Constraints

- Only this crate may be edited here; workspace manifest, sibling crates, and `docs/` are out of scope.
- Async-first: builders that call a model are `async fn` plus a `stream_build` returning `BoxStream<'static, Result<String>>`; no `block_on`/`asyncio.run` bridging — synchronous Python entry points become `async` (documented deviation).
- Errors: `openjiuwen_common::error::{Error, Result}` only; invalid user input is `Error::validation`, missing capability is `Error::unsupported`, model failures propagate as `Error::llm`.
- Serialization: public data types are `serde` (`rename_all = "snake_case", default`), snake_case, unknown-field tolerant; DSL and report JSON keep the Python field names.
- No `unsafe`; `tracing` logging via `openjiuwen_common::logging`; no `println!`/`eprintln!`; no panics on user input.
- Files ≤1000 lines; one responsibility per file; shared helpers within a module, not duplicated across modules.
- Tests are offline and deterministic: `openjiuwen_test_support::ScriptedModel` for model-driven paths, `TempWorkspace` for filesystem bundles, fixed ids/timestamps for DSL and reports.
- Empty prompts, blank skill names, unknown template names, and malformed frontmatter are validation errors, never silent fallbacks (Python's "warn and degrade" behavior is a documented deviation).

## Design Decisions

- Python builders own a `Model` constructed from configs; here every builder takes `Arc<dyn Model>` (or `ScriptedModel` in tests), so configuration stays with the caller and no builder reads the environment.
- Deterministic cores are public: template rendering, message assembly, JSON-fence/tag parsing, skill frontmatter rendering/parsing, agent DSL transformation, and evaluation scoring are ordinary functions testable without a model; the model only supplies text.
- `MetaTemplateBuilder`'s registry keys are uniform (`register`/`get`/`pop` share the caller-supplied name); Python's hidden `META_TEMPLATE_` prefix asymmetry is not reproduced.
- `agent_builder` keeps Python's two artifacts (the platform DSL JSON and the kernel config) and composes them: a spec/DSL always converts into `AgentCard` + `LlmAgentConfig`, and `build_llm_agent` wires a `ReActAgent`-based `LlmAgent` when a model is supplied.
- Model-driven agent building is one round trip that must yield DSL JSON; a non-JSON answer is returned as a clarifying `Response` (Python's state machine behavior) instead of the multi-stage clarifier/designer pipeline.
- `skill_evaluator` replaces Python's prompt-defined rubric with deterministic structural scoring for design and expectation assertions for function, keeping the report field names and the aggregation formula; a `Model` judge can be layered on later through the same trait seams.
- `tune`'s optimizers depend on Python optimization internals and are stubs; `Case`, `EvaluatedCase`, and `CaseLoader` are fully ported because they are pure data models shared with prompt building.

## Test Strategy

- Unit tests inside each module (`#[cfg(test)]`) cover deterministic transforms, validation errors, and serialization round-trips.
- Model-driven paths use `ScriptedModel` with queued responses: prompt builders assert the rendered messages, skill creator asserts the parsed result, agent builder asserts DSL extraction, evaluator asserts scoring.
- `crates/openjiuwen-dev-tools/tests/` holds crate-level round-trip suites: skill bundle write→read in a `TempWorkspace`, DSL JSON round-trip, prompt-builder template banks.
- No network, no environment reads, no wall-clock dependence: ids and timestamps are parameters or fixed constants.

## Deferred

- Python `skill_omni_creation` scraping/download stages (Playwright, bs4, image filtering) — no port; `skill_creator` starts from text/spec input.
- Agent-builder LLM pipeline internals (LLM clarifier, workflow designer, DL generator/reflector, cycle checker, Mermaid rendering of designs) — only the DSL model plus JSON extraction are ported; Mermaid output is a stub.
- Prompt-tuning optimizers (`InstructionOptimizer`, `ExampleOptimizer`, `JointOptimizer`, `Trainer`, `PromptSearchOptimizer`) — `Error::Unsupported`.
- LLM-judge design/safety scoring, subagent orchestration, and the `run_eval_query.py` shell harness — superseded by the deterministic harness.

## Routing Table

- `./prompt_builder/` → prompt template banks, meta/feedback/bad-case builders, deterministic prompt helpers (`prompt_builder/CONTEXT.md`)
- `./skill_creator/` → skill definition model, validation, templating, `SKILL.md` bundle read/write, model-assisted creation (`skill_creator/CONTEXT.md`)
- `./agent_builder/` → agent specs, platform DSL, kernel bridge, session history, LLM-driven builder (`agent_builder/CONTEXT.md`)
- `./skill_evaluator/` → evaluation cases, assertion harness, report/score models, aggregation (`skill_evaluator/CONTEXT.md`)
- `./tune/` → `Case`/`EvaluatedCase`/`CaseLoader` and unsupported-optimizer stubs (`tune/CONTEXT.md`)
- `./src/lib.rs` → module tree, crate lints, shared crate-level re-exports
- `./Cargo.toml` → crate manifest; `openjiuwen-test-support` is a dev-dependency only
- `../openjiuwen-foundation/` → `PromptTemplate`, `BaseMessage`, `Model`, `ToolInfo` (sibling — read-only, escalate writes to parent)
- `../openjiuwen-agent/` → `AgentCard`, `LlmAgentConfig`, `LlmAgent`, `ReActAgent` (sibling — read-only, escalate writes to parent)
