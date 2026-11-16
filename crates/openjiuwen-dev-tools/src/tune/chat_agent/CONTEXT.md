# openjiuwen-dev-tools / tune / chat_agent

## Intent

Port `openjiuwen.dev_tools.tune.chat_agent` (`chat_config.py`, `chat_agent.py`) so a tuning run can be bound to an agent instead of the deterministic evaluator.
Python's `ChatAgent` is the simplest LLM chat agent wrapped around one `LLMCall` (`system_prompt`, `user_prompt`, `freeze_system_prompt`, `freeze_user_prompt`) and answers `dict(output=content, tool_calls=…)`; there is no Python runtime here, so the execution step is a caller-supplied runner seam.
The adapter owns the conversation assembly (candidate prompt → system message, case → user message) and the bindings that put the agent behind `InstructionOptimizer` and `Trainer`.
Everything is offline and deterministic: no environment reads, no network, no wall clock, no sleeps.

## API Surface

- `ChatConfig` — the typed settings a chat agent is built from: `agent_id`, `version`, `description`, `model_name`, `system_prompt`, `user_prompt`, `temperature: Option<f64>`, `top_p: Option<f64>`, `freeze_system_prompt`, `freeze_user_prompt`; fields are public and double as the readers (matching `OptimizerConfig`).
- `ChatConfig::new(agent_id, model_name)` (unvalidated, Python defaults elsewhere), builders `with_version`/`with_description`/`with_system_prompt`/`with_user_prompt`/`with_temperature`/`with_top_p`/`with_frozen_system_prompt`/`with_frozen_user_prompt`, and `validate()`.
- `ChatConfig::validate()` reports `Error::validation` for a blank `agent_id`, a blank `model_name`, a non-finite or out-of-`0.0..=2.0` `temperature`, a non-finite or out-of-`0.0..=1.0` `top_p`, and a `freeze_user_prompt` with a blank `user_prompt`; a blank frozen `system_prompt` is accepted (it means "no frozen block").
- `create_chat_config(agent_id, version, description, model_name) -> Result<ChatConfig>` — the port of Python's `create_chat_agent_config`, validating before it returns.
- `ChatRunner` — the `#[async_trait]`, `Send + Sync` execution seam: `async fn reply(&self, system: &str, user: &str) -> Result<String>`.
- `ModelChatRunner` — the model-backed default: `new(model)` (request name from the model), `from_config(&ChatConfig, model)` (request name, `temperature`, `top_p` from the config), `with_request`, `model()`, `request()`; one `Model::invoke` of `[system, user]` per reply, and `Error::llm` when the completion carries no choice.
- `ScriptedChatRunner` — `new`, `with_reply(marker, reply)`, `with_ordered_reply(reply)`, `with_fallback(reply)`, `remaining()`, `calls() -> Vec<ChatCall>`, `call_count()`, `last_call()`; `ChatCall { system, user }` is the recorded turn.
- `ChatAgent` — `Clone` (config cloned, runner shared): `new(config, Arc<dyn Model>)` (model-backed runner), `with_runner(config, Arc<dyn ChatRunner>)` (custom seam), both validating; readers `config()`, `runner()`.
- `ChatAgent::reply(&self, prompt, case) -> Result<String>` — the raw reply text for one candidate prompt over one case.
- `ChatAgent` implements `crate::tune::evaluator::PromptRunner`: the reply becomes the answer map `{"output": text}`.
- `ChatAgent::evaluator(&self) -> RunnerEvaluator` — `RunnerEvaluator::new(Arc::new(self.clone()))` over the shared runner, scored by the default `LabelScorer`.
- `ChatAgent::optimize(&self, prompt, cases) -> Result<OptimizationResult>` — `InstructionOptimizer::new().with_evaluator(Arc::new(self.evaluator()))`.
- `ChatAgent::train(&self, prompt, train, val) -> Result<OptimizationResult>` — `Trainer::new().with_evaluator(Arc::new(self.evaluator()))`.
- `CANDIDATE_BLOCK_HEADING` (`"## Candidate instructions"`) — the heading between a frozen system prompt and the candidate prompt.
- `chat_agent/mod.rs` is the doc and re-export surface; `tune/mod.rs` re-exports everything through `pub use chat_agent::*`, so these names also live at `openjiuwen_dev_tools::tune::*`.

## Prompt assembly

System message: the candidate prompt under optimisation, unless `freeze_system_prompt` is set and `system_prompt` is non-blank, in which case `"{frozen}\n\n## Candidate instructions\n\n{candidate}"` (both trimmed) keeps the persona pinned while an instruction block is tuned.
An unfrozen non-blank `system_prompt` is ignored: the candidate prompt is what tuning rewrites, so it replaces the baseline.
User message: a frozen `user_prompt` verbatim (no substitution, must be non-blank); otherwise `user_prompt` with its `{{name}}` placeholders substituted from the case inputs; otherwise `render_case_inputs(case)` (`[key]: value` lines).
Substitution is one left-to-right pass, so an inserted value is never itself substituted; an unsupplied placeholder and a stray `{{`/`}}` stay verbatim.
A substituted string value is rendered bare, any other JSON value compactly — the same rendering `render_case_inputs` uses.

## Constraints

- Only files under `src/tune/chat_agent/` are edited; `../mod.rs`, `Cargo.toml`, `src/lib.rs`, and `tests/*.rs` are out of scope.
- No new dependencies: `openjiuwen_common`, `openjiuwen_foundation`, `async-trait`, `serde`, `serde_json`, `tokio`; dev-dependency `openjiuwen_test_support`.
- Errors only from `openjiuwen_common::error`: `Error::validation` for an invalid configuration, `Error::not_found` for an unscripted `ScriptedChatRunner` call, `Error::llm` for a provider failure or a choiceless completion; `Error::Unsupported` is never returned here.
- No `unsafe`, no printing, no panics on user input (a poisoned lock is recovered instead of unwrapped); tests may `unwrap`.
- Every public item is documented with a `# Errors` section where it can fail (`missing_docs` + `missing_errors_doc` under `-D warnings`).
- One responsibility per file and files well under 1000 lines.

## Design Decisions

- The runner seam replaces Python's `Runner`/`Session`/`ContextEngine` plumbing: a tuning run never needs a session, a context window, or a tool registry, so `ChatRunner` takes the two rendered strings and returns text.
- `ChatAgent` is `Clone` so `evaluator(&self)` can hand a clone to `RunnerEvaluator` while sharing one `Arc<dyn ChatRunner>`; the caller and the evaluator therefore observe one conversation history (a `ScriptedChatRunner`'s recorded calls stay visible).
- Both constructors validate, so an agent that exists always has a valid configuration and the prompt-rendering rules never see an impossible state (a frozen-but-blank user prompt is rejected at construction).
- The candidate prompt is the system message because that is what Python's `LLMCallConfig` puts under optimisation; `freeze_system_prompt` keeps the configured prompt in front of it rather than ignoring the candidate, so a pinned persona and a tuned instruction coexist.
- `ChatAgent::optimize`/`train` attach only the evaluator; a model-driven rewrite and a custom budget stay with the caller, who composes `InstructionOptimizer::with_model`/`Trainer::with_config` around `agent.evaluator()` (the runner seam is not a `Model`, so the adapter cannot invent one).
- The answer map has exactly one `output` entry, like Python's `dict(output=…)`; `LabelScorer` already lets that single entry stand in for a missing label key, so prose replies are scored without a second scoring rule here.
- A case's declared tools stay on `Case::tools()`: the agent's seam carries text only, and the tool layer renders the declarations into the optimizer's rewrite prompt (`<tools_description>`), which the evaluator does not disturb.
- The model-backed runner uses `config.model_name` rather than the model's own name, because Python's `ChatAgentConfig.model` is the authority on the wire model; `temperature`/`top_p` are narrowed to the request's `f32` with a documented allow (both are validated into `0.0..=2.0`, so nothing meaningful is lost).
- `ScriptedChatRunner` resolves a marker in the user message first and the system message second (sorted marker order, so the choice is deterministic), then the call-order queue, then the fallback; every call is recorded before resolution, so a test can see an attempted-but-unscripted call.
- `frozen` prompts are trimmed at the block boundary only, so incidental author whitespace cannot produce a blank line between messages; the user prompt itself is passed through untouched.

## Test Strategy

- `config.rs` unit tests: the Python defaults, every builder, all validation rejections (blank id/model, temperature and `top_p` bounds plus `NaN`/infinities, frozen-blank user prompt) with the offending field named in the message, `serde` snake_case round-trip, per-field defaulting, unknown-field tolerance, and the permissive-deserialization/`validate`-reports-it split; plus `create_chat_config` accept/reject.
- `runner.rs` unit tests: marker replies, the system-message marker fallback, ordered replies and `remaining()`, marker-before-queue precedence, the fallback, `NotFound` on an unscripted call with the call still recorded, recorded call order, and the model path (`from_config` request shape, the two-message conversation and request name reaching `ScriptedModel`, `new` taking the model's own name, provider-failure propagation, `Debug`).
- `agent.rs` unit tests: construction validation for each rejected field, reader access through the shared runner, prompt rendering (candidate → system, frozen system block, blank frozen system fallback, unfrozen baseline ignored, template substitution including non-string values and unsupplied placeholders, no cascading substitution, malformed braces, empty prompt → `render_case_inputs`, frozen user prompt verbatim), the `{"output": text}` answer map, `evaluator()` scoring through the agent's runner plus the `NotFound` propagation, `optimize` end-to-end over `ScriptedModel` (early stop when the seed already passes, search rounds and the mutated candidates that reach the model when it does not), `train` end-to-end (baseline early stop) and its input validation, an offline `ScriptedChatRunner` optimize, declared tools reaching the rewrite prompt, and `Debug`.
- Every model path asserts both the result and the recorded conversation (`ScriptedModel::calls`, `ScriptedChatRunner::calls`); the whole suite runs offline in the default `cargo test -p openjiuwen-dev-tools`.
- The module `//!` doc example builds a `ChatAgent` from a `ScriptedChatRunner`, tunes one case, and asserts a perfect score, so the public surface stays runnable.

## Routing Table

- `./mod.rs` → module docs, the private submodules, and the public re-export surface
- `./config.rs` → `ChatConfig`, validation, builders, `create_chat_config`, serde surface
- `./runner.rs` → `ChatRunner`, `ModelChatRunner`, `ScriptedChatRunner`, `ChatCall`
- `./agent.rs` → `ChatAgent`, prompt assembly (`render_system`/`render_user`/`substitute`), the `PromptRunner` impl, and the `evaluator`/`optimize`/`train` adapters
- `../evaluator/` → `PromptRunner`, `render_case_inputs`, `RunnerEvaluator`, `LabelScorer` (parent — read-only here)
- `../optimizer/` → `InstructionOptimizer`, `Trainer`, `OptimizationResult` (parent — read-only here)
