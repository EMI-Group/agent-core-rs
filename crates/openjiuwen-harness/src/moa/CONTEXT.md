# openjiuwen-harness::moa (mixture of agents)

## Intent

Optional reference consultation for `DeepAgent`: generate independent proposals from several injected models, verify/rank them offline, and produce an integrated answer.
Ports `openjiuwen/harness/moa/` and `openjiuwen/harness/rails/moa_rail.py` from the Python SDK (v0.1.20).
The module is self-contained: it owns its config, state, prompts, proposer fan-out, verifier, aggregator, skill descriptor and harness rail.
It never opens a network connection by itself; every model handle is injected and every HTTP path belongs to the caller.
Python entry points and persisted JSON keys are preserved where the reference persists data; live Python dataclass fields that hold model objects are split into serde data plus injected handles.

## Module layout

- `config.rs` — `MoaConfig`, `VerifierConfig`, `JevConfig`, `MoaMode`, `VerifierSelection`, validation helpers and Python-compatible error messages.
- `state.rs` — `TurnKey`, `ProposalSource`, `ProposerFailure`, `ProposalAttempt`, `ProposalBatch`, `ProposalStore`, `MoaState`, `MoaSnapshot`, `ConsultationOutcome`, `ConsultationRecord`, `ConsultationHistory`.
- `redaction.rs` — `redact`: shared credential/PII redaction for verifier and transcript inputs.
- `prompts.rs` — `REFERENCE_SYSTEM_PROMPT`, `AGGREGATOR_SYSTEM_PROMPT`, reference-proposal rendering, transcript flattening, reference/aggregator/verifier message builders, `moa_section`.
- `consultation.rs` — bounded concurrent proposer fan-out over injected models with timeout, cancellation and partial-failure accounting.
- `verifier.rs` — `ProgressVerifier` seam, offline `StructuralVerifier`, model-driven `LlmVerifier`, structural proposal ranking (`rank_proposals`).
- `aggregate.rs` — `AggregationStrategy`, `Aggregator`: deterministic labelled concatenation by default, optional model synthesis.
- `rail.rs` — `MoaRail`: harness rail that publishes the MoA system-prompt section, gates on config, and exposes the last consultation outcome.
- `builtin_skills.rs` — embedded `consult-reference-models` skill descriptor and materialization helper.
- `skill_mount.rs` — `ConsultationSkillMount`: validates and mounts the consultation skill root into a `crate::skills::SkillLibrary`.
- `tests.rs` — offline integration-style suite for the whole module.
- `skills/consult-reference-models/SKILL.md` — embedded skill body (`include_str!`).

## API Surface

Config (`config.rs`):
- `MoaMode { AggregatorOnDemand, VerifierRouted, JevRouted }`, serde snake_case, `as_str`/`parse`; default `AggregatorOnDemand`.
- `VerifierSelection { Structural, Llm }`, serde snake_case; default `Structural`.
- `VerifierConfig { interval: usize, n_evaluations: usize, max_verifier_calls: Option<usize> }`, default `{3, 3, None}`, `validate` and builders (`with_interval`, `with_n_evaluations`, `with_max_verifier_calls`).
- `JevConfig { interval: usize, recent_messages: usize, timeout_secs: f64, max_jev_calls: Option<usize>, max_reference_calls: Option<usize> }`, default `{3, 5, 30.0, None, None}`, `validate`.
- `MoaConfig` fields: `reference_models: Vec<Arc<dyn Model>>`, `mode`, `verifier`, `verifier_selection`, `verifier_model: Option<Arc<dyn Model>>`, `aggregator_model: Option<Arc<dyn Model>>`, `max_proposals: Option<usize>`, `max_concurrency: usize`, `proposer_temperature: Option<f32>`, `aggregator_temperature: Option<f32>`, `max_reference_chars: usize`, `max_advice_chars: usize`, `reference_timeout: Duration`, `max_reference_calls: Option<usize>`, `consultation_skill_dir: Option<PathBuf>`, `jev: Option<JevConfig>`, `enabled: bool`, `history_limit: usize`, `rail_priority: i32`.
- `MoaConfig::new(reference_models)`, `validate() -> Result<()>`, `proposal_count() -> usize`, builder methods for every field, and a manual `Debug` that redacts model handles.
- Validation messages mirror Python: `MoA requires at least one reference model`, `unsupported MoA mode: {mode}`, `jev_routed requires a JevConfig`, `JEV and verifier routing are mutually exclusive; jev is only supported in jev_routed`, `verifier_routed requires a verifier model`, `{name} must be a positive integer`, `{name} must be a nonnegative integer or None`, `{name} must be finite and positive`, `consultation_skill_dir ...` messages.

State (`state.rs`):
- `TurnKey { turn: u64, iteration: u64 }` (serde, `new`).
- `ProposalSource { index: usize, text: String }`, `ProposerFailure { index: usize, error: String }`.
- `ProposalAttempt { key: Option<TurnKey>, injected: bool, reason: String, kept_chars: usize, fallback: bool }`.
- `ProposalBatch { sources, target, attempts, batch_id, disabled_reason, origin }`.
- `ProposalStore { active, ready, epoch, issued_ticket, accepted_ticket, closed }` with `activate`, `invalidate(reason, clear)`, `close(reason)`.
- `MoaState { problem, baseline: Vec<BaseMessage>, proposals, active_key, completed_turns, steps, previous_score: Option<f32>, reference_calls, verifier_calls, consultation_rejection, closed, jev_calls, last_jev_decision, last_jev_reason, invoke_id }` with `new`, `accepts_result(epoch)`, `snapshot`.
- `MoaSnapshot` serde struct with the diagnostics Python exposes (no reference text).
- `ConsultationOutcome { proposals, success_count, failed_count, failures, ranking, aggregate, verifier_score }` serde, `is_empty`, `best`.
- `ConsultationRecord { problem, outcome }`, `ConsultationHistory { limit, records }` bounded with `push`, `records`, `save(path)`, `load(path)`, `clear`.

Redaction (`redaction.rs`): `redact(&str) -> String` removes JSON-shaped secret values, bearer tokens, `api_key`/`token`/`password` assignments and `sk-` keys.

Prompts (`prompts.rs`): `REFERENCE_SYSTEM_PROMPT`, `AGGREGATOR_SYSTEM_PROMPT`, `MOA_PROMPT_SECTION_NAME`, `escape_xml`, `render_reference_proposals(sources, raw_limit)`, `consultation_transcript(messages)`, `reference_messages(transcript)`, `aggregation_messages(problem, sources)`, `verifier_messages(problem, steps)`, `moa_section(language, mode) -> PromptSection`.

Consultation (`consultation.rs`): `CONSULT_TOOL_NAME`, `Consultation { sources, success_count, total_models, failures }`, `consult(config, messages, cancel) -> Result<Consultation>`, `consult_problem(config, problem, messages, cancel) -> Result<Consultation>`.

Verifier (`verifier.rs`): `ProgressVerifier` async trait, `StructuralVerifier`, `LlmVerifier` (injected model, parses a 0..1 score), `RankedProposal { index, score, reasons }`, `score_proposal`, `rank_proposals(problem, sources)`, `score_progress(config, problem, steps)`, `rank_with_config(config, problem, sources)`.

Aggregation (`aggregate.rs`): `AggregationStrategy { Labelled, Synthesis }`, `Aggregator::new`, `Aggregator::labelled`, `Aggregator::aggregate(config, problem, sources, cancel) -> Result<String>`.

Rail (`rail.rs`): `MOA_RAIL_NAME`, `MOA_PROMPT_SECTION`, `MOA_STATE_EXTRA_KEY`, `MOA_OUTCOME_EXTRA_KEY`, `MoaRail::{new, config, prompt_section, last_outcome, snapshot, history, consult, save_history, load_history}`; implements `crate::rails::HarnessRail` with `name() = "moa"` and priority `rail_priority::MOA` by default.

Skills (`builtin_skills.rs`, `skill_mount.rs`): `CONSULT_SKILL_NAME`, `CONSULT_SKILL_DESCRIPTION`, `CONSULT_SKILL_BODY`, `validate_builtin_skill_name`, `consult_skill`, `materialize_builtin_skill`; `ConsultationSkillMount::{new, embedded, from_dir, skill, mount, unmount, validate}`.

Every public item is re-exported from this module root; the crate root adds the same names (see the parent report).

## Constraints

- `openjiuwen_common::error::{Error, Result}` only; no `unwrap`/`expect` outside `#[cfg(test)]`.
- Async-first tokio; injected models; cancellation via `tokio_util::sync::CancellationToken`; never hold a `parking_lot` guard across `.await`.
- No network, no filesystem writes except explicit history and skill-materialization helpers (caller-supplied paths); no globals.
- serde snake_case, tolerant of unknown fields; persisted JSON keeps Python keys where Python persists them.
- `tracing` only; no stdout/stderr.
- Files around 1000 lines; each file has a single responsibility.
- Pedantic clippy and `missing_docs` are enforced by `-D warnings`; document every public item and every `# Errors` case.
- Consultations fail open on partial proposer failure and fail only when every proposer fails; a failed proposer is recorded with its index and error.
- The rail never calls a model: `consult` is an explicit API the host calls; hooks only read config/outcome and publish the prompt section.
- `MoaConfig` holds live `Arc<dyn Model>` handles, so it is not serde; the serde-able knobs live in `VerifierConfig`/`JevConfig` and in the snapshot/outcome types.

## Test Strategy

- Offline only: `openjiuwen_test_support::ScriptedModel`, hand-built `BaseMessage`s, `tempfile` directories, `tokio_util::sync::CancellationToken`; no network, no wall-clock ordering.
- `config.rs`: defaults, builders, every validation message, serde of `MoaMode`/`VerifierConfig`/`JevConfig`, redacting `Debug`.
- `state.rs`: store activate/invalidate/close, snapshot diagnostics, outcome best/is_empty, bounded history eviction and JSON round-trip.
- `redaction.rs`: JSON secrets, bearer tokens, assignments and `sk-` keys; unrelated text untouched.
- `prompts.rs`: reference rendering wraps and escapes, limits bodies, transcript filtering/truncation, message builders, bilingual section.
- `consultation.rs`: fan-out preserves configured order, partial failure is recorded and does not sink the batch, all-fail returns an error, per-proposer character caps are applied, pre-cancelled token returns `Error::Cancelled`, timeout is recorded as a failure.
- `verifier.rs`: structural scores stay in 0..1, longer/structured proposals outrank thin ones, redaction applies before scoring, `LlmVerifier` parses a scripted score and rejects non-numeric output, progress scores detect stalled repetition, config selection picks structural vs LLM.
- `aggregate.rs`: labelled output shape and source order, empty sources, synthesis required-model error, synthesis through `ScriptedModel`, cancellation.
- `rail.rs`: disabled config publishes nothing, enabled config publishes exactly one `moa` section into `prompt_sections`, mode-specific text, priority/name, outcome publication, consult stores outcome/history, fresh state per invocation.
- `skill_mount.rs`: embedded descriptor, explicit directory validation and error messages, mount/unmount ownership, materialization round-trip.
- `tests.rs`: end-to-end offline scenario (proposers -> verifier -> aggregator -> rail -> history).
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --offline`, `--no-default-features`, `--all-features`, `cargo clippy -p openjiuwen-harness --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps` warning-free.

## Deferred items (typed)

- `jev_routed` decision-making (`decide_reference`) returns `Error::Unsupported` with reason `jev_routed requires an external System One client, which this port does not provide`; the config and state fields are present so callers can model it.
- LLM-as-judge proposal quality, multimodal proposals, vendor proposer clients and OTLP telemetry are out of scope; models are always injected handles.
- `consult_reference_models` tool registration and Python's ability-manager integration are not ported (the crate has no ability manager); `MoaRail::consult` is the replacement seam.
- Dynamic plugin/`sys.modules` loading of skills is not ported; the builtin skill body is embedded with `include_str!` and materialized on request.

## Routing Table

- `./config.rs` → configuration, validation, modes
- `./state.rs` → proposal store, moa state, snapshots, outcome and history
- `./redaction.rs` → credential redaction
- `./prompts.rs` → MoA prompts, rendering, transcript/message builders
- `./consultation.rs` → concurrent proposer fan-out
- `./verifier.rs` → structural + LLM verifier, proposal ranking
- `./aggregate.rs` → labelled/synthesis aggregation
- `./rail.rs` → `MoaRail` harness rail
- `./builtin_skills.rs` → embedded consultation skill descriptor
- `./skill_mount.rs` → consultation skill mounting/validation
- `./tests.rs` → offline module suite
- `./skills/consult-reference-models/SKILL.md` → embedded skill body
