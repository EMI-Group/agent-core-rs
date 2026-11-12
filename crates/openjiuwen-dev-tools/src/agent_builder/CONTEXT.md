# openjiuwen-dev-tools / agent_builder

## Intent

Ports `openjiuwen.dev_tools.agent_builder`: programmatic agent/workflow specs, the serialized platform DSL, session dialogue history, and a model-driven builder.
The builder turns one caller request into either a parsed `LlmAgentDsl`, a validated `WorkflowDsl`, or a plain-text clarifying answer, and keeps per-session state between calls.
Internal structure is idiomatic Rust (serde models, typed errors, one async round trip per request), not a transliteration of Python's stage classes.

## API Surface

- `LlmAgentDsl` (`dsl.rs`) with `to_value`/`from_value`/`to_json_string`/`from_json_str`, plus blocks `DslConfigs`, `DslModel`, `DslModelInfo`, `PluginRef`, `WorkflowRef`, `DslDependencies`.
- `WorkflowDsl` (`workflow_dsl.rs`) with `DslNode`, `DslEdge`, `validate()`, `node(id)`, `to_json_string`, `from_json_str`.
- `LlmAgentSpec` and `WorkflowSpec` (`spec.rs`) with `to_dsl`, `to_json_string`, `from_json_str`, and `with_*` builders.
- `LlmAgentDsl::{to_agent_card, to_llm_agent_config, build_llm_agent}` (`bridge.rs`) map a document into `openjiuwen_agent::{AgentCard, LlmAgentConfig, LlmAgent}`.
- `HistoryManager` and `DialogueMessage` (`history.rs`) with `new(max)`, `add_message`, `add_user_message`, `add_assistant_message`, `get_history`, `get_latest_k(k)`, `len`, `is_empty`, `clear`, `max_history_size`; `DEFAULT_MAX_HISTORY_SIZE` is `20`.
- `AgentBuilder` (`builder.rs`) with `new(Arc<dyn Model>)`, `build_agent`, `build_agent_at`, `history(session_id, k)`, `clear_session`, `status`, `model`.
- `AgentType` (`llm_agent`/`workflow`) and `now_millis` (`builder.rs`).
- `BuildState` (`initial`/`clarifying`/`requesting`/`processing`/`completed`), `BuildStatus`, `BuildOutput`, `BuildResult`, and `map_state_to_status` (`outcome.rs`), re-exported from `mod.rs`.
- `BuildResult::to_json` emits `status`, `session_id`, `agent_type`, and exactly one of `dsl`, `workflow`, `response`; there is no nested `output` key.
- `BuildOutput` serializes externally tagged, so its JSON is exactly `{"dsl": {...}}`, `{"workflow": {...}}`, or `{"response": "..."}`.

## DSL Shape

The DSL is a typed subset of Python's document: agent identity, `configs`, `model`, plugin/workflow references, dependency id lists, and the two epoch-millisecond timestamps.
Python-only fields survive in `LlmAgentDsl::extra` (`#[serde(flatten)]`, a `BTreeMap`), notably `model_provider` and `constraints`, and are written back unchanged.
`DslConfigs::opening_remarks` is nested inside `configs` here, while Python keeps it at the document top level; the wire name is unchanged.
Unknown keys never make parsing fail, and a document without `agent_type` defaults to `"react"`.
A workflow document uses `nodes` (`id`, `type`, `title`, `description`, `inputs`, `outputs`, `data`) and `edges` (`source`, `target`, `condition`), with `type` keeping its Python wire name.
`WorkflowDsl::validate` requires at least one node, unique node ids, and resolvable edge endpoints; failure messages name the offending id.

## Behavior of build_agent

`build_agent` is `build_agent_at` with `now_millis()`; both keep the session state in the builder keyed by the caller's session id.
A blank or whitespace-only query is rejected as `Error::Validation` before any session exists, so `status`/`history` stay empty and the model is never called.
On the first call the session is created (`HistoryManager::new(DEFAULT_MAX_HISTORY_SIZE)`), the user turn is appended, and the stored state becomes `Processing`.
The model is called exactly once: one system message from `build_system_prompt(agent_type)` followed by one `user`/`assistant` message per retained history turn (any role other than `"assistant"` is sent as `user`).
Request configuration is `ModelRequestConfig::new(model.model_name())`; a model error propagates and leaves the session in `Processing` with the user turn recorded so the caller can retry.
A response without choices is `Error::Llm("model returned no choices")` and also leaves the session untouched.
The answer text is scanned with `util::extract_json_fence`, then parsed as `LlmAgentDsl` or `WorkflowDsl` (plus `validate()`) for the requested type; any failure becomes `BuildOutput::Response(text)` rather than an error.
A successful LLM-agent parse fills an empty `agent_id` with a fresh UUIDv4 and zero timestamps with the supplied `timestamp_ms`; the workflow parse needs no stamping.
Success records `Completed` and the status string `"completed"`; the fallback records `Clarifying` and the status from `map_state_to_status`.
Either way the assistant text is appended to the session history, so `status().resource_count` grows by two per completed call.
`map_state_to_status` yields `"clarifying"` (Initial for LLM agents, Clarifying), `"requesting"` (Initial for workflows, Requesting), `"processing"` (Processing), and `"completed"` (Completed).

## Constraints

- Only this directory may be edited here; sibling crates, `src/lib.rs`, `Cargo.toml`, `src/util.rs`, and `docs/` are out of scope.
- Errors are `openjiuwen_common::error::{Error, Result}` only: invalid user input is `Error::validation`, model failures propagate unchanged, a choice-less response is `Error::llm`.
- All public data types are serde (`rename_all = "snake_case"`, `default`, unknown-field tolerant) and keep Python's JSON field names.
- Time is a parameter everywhere except `now_millis`, and randomness is used only for a missing DSL `agent_id`, so tests never depend on the wall clock or on generated ids.
- No `unsafe`, no `unwrap`/`expect`/`panic!` in library code, no printing; diagnostics go through `tracing::debug`.
- Async-first: `build_agent`/`build_agent_at` are `async fn` and compose inside an existing tokio runtime; no blocking calls and no environment reads.
- Files stay under 1000 lines with one responsibility each: `builder.rs` (~730) holds the session builder and stays within budget by delegating the result vocabulary to `outcome.rs` (~315), whose test module covers the full state/status matrix.

## Design Decisions

- The builder owns `Arc<dyn Model>` supplied by the caller, so no builder constructs a client or reads provider configuration.
- Description of the model contract is a `const` prompt per artifact (`LLM_AGENT_SYSTEM_PROMPT`, `WORKFLOW_SYSTEM_PROMPT`) that enumerates exactly the keys the parser reads.
- One model round trip replaces Python's state machine: a parseable fenced document completes the session, everything else is returned verbatim as a clarifying response (documented deviation).
- `BuildState` adds the `Clarifying` and `Requesting` response states that Python only models as status strings beside its state enum.
- Session state is a private `SessionState` (`history`, `state`, `agent_type`), so the public surface cannot corrupt a session's history length invariants.
- `AgentBuilder` derives `Clone` (sessions are copied) and implements `Debug` manually because `dyn Model` is not `Debug`; the output prints provider, model name, and session count.
- `BuildOutput` keeps the `LlmAgentDsl` inline; the `large_enum_variant` lint is allowed locally because the document is the payload callers hand to the platform.
- The bridge is total and infallible for card and configuration: an absent provider falls back to `"openai"` and absent constraints to `ConstrainConfig::default()`; only `build_llm_agent` can fail.
- `HistoryManager::new(0)` uses the default capacity of `20`, while `get_latest_k(0)` returns no message — Python's `k <= 0` meant "all", and this port keeps the accessor range-free but unambiguous.

## Deferred

- Python's build-state machine stages — intention detector, LLM clarifier, agent designer, workflow designer, DL generator/reflector, LLM cycle checker, and Mermaid rendering — are not ported.
- Python's resource retrieval/preprocessing (plugins, knowledge bases, prompt templates from the platform) has no counterpart; documents carry ids only.
- Multi-turn refinement after `Completed` is a plain second call: no state-machine re-entry, no error-correction pass over a previously produced artifact.
- LLM-judged design review of a produced document, and any streaming variant of `build_agent`, are not implemented.
- Python's `main.py`/CLI driver for the builder is not ported; the CLI lives in `openjiuwen-cli`.

## Test Strategy

- Unit tests live beside the code: `dsl`, `workflow_dsl`, `spec`, `bridge`, `history`, `builder`, and `outcome` each own a `#[cfg(test)] mod tests`.
- `builder` tests cover `AgentType`'s `Display`/`FromStr` wire names plus the session, prompt-assembly, and parse-fallback flows.
- `outcome` tests cover all five `BuildState` values against both `AgentType`s, the `BuildState`/`BuildStatus` serde wire names, all three `BuildOutput` variants, and the flattened `BuildResult::to_json` keys.
- Model-driven paths use `openjiuwen_test_support::ScriptedModel` with queued turns or a single failing turn; `call_count`/`calls` assert that exactly one round trip happens per request.
- Sessions are exercised for isolation, `clear_session`, `history(sid, Some(k))` limits, and `status().resource_count` transitions.
- Timestamps are asserted against an explicit `build_agent_at(.., 1_700_000_000_000)` value; only the clock-entry-point test asserts a positive `create_time`.
- Every test is offline and deterministic: no network, no environment reads, no generated-id assertions beyond non-emptiness.

## Routing Table

- `./dsl.rs` → serialized LLM-agent document and its model/plugin/workflow/dependency blocks, with unknown keys preserved
- `./workflow_dsl.rs` → serialized workflow document, node/edge models, and structural validation
- `./spec.rs` → programmatic `LlmAgentSpec`/`WorkflowSpec` and their DSL conversion
- `./bridge.rs` → DSL-to-kernel mapping: `AgentCard`, `LlmAgentConfig`, `build_llm_agent`
- `./history.rs` → bounded `HistoryManager` and `DialogueMessage`
- `./builder.rs` → session-based `AgentBuilder`, `AgentType`, prompt assembly, artifact parsing, and `now_millis`
- `./outcome.rs` → `BuildState`, `BuildStatus`, `BuildOutput`, `BuildResult`, and `map_state_to_status`, the result vocabulary re-exported from `mod.rs`
- `./mod.rs` → module tree and crate-level re-exports for this directory
- `../util.rs` → shared `extract_json_fence` and text helpers (sibling, read-only — escalate writes to the parent)
- `../../openjiuwen-agent/` → `AgentCard`, `LlmAgentConfig`, `LlmAgent`, `ReActAgent` consumed by the bridge (sibling crate, read-only — escalate writes to the parent)
