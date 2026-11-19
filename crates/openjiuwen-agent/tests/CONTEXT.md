# tests/

## Intent

Crate-level integration suites for `openjiuwen-agent`, driven by `openjiuwen-test-support::ScriptedModel` (offline, deterministic).

## API Surface

Suites (run with `cargo test -p openjiuwen-agent`):
- `agent_card.rs` → `AgentCard` field set/`Card` trait/round-trip, `AgentOutput` result-type wire strings and `state` frames, absent-when-empty key shapes, Python-shaped result dicts, `AgentChunk` ↔ `StreamItem` bridge.
- `react_agent.rs` → no-tool answer, single tool, parallel tools, tool error as tool message, max-iteration policies (`error`, `return_observations`, `final_answer`), empty-response retry, prompt/window contents, input/config errors.
- `streaming.rs` → `llm_output`/`llm_reasoning`/`llm_usage`/`answer` frame assembly from a streamed scripted model, plus frame mirroring into a supplied session.
- `rails.rs` → rails steering the running loop: priority ordering per event, force-finish before the model and before a tool, `ModelBackupRail` retry, `GuardrailRail` blocking (`ErrorKind::Security`), `MemoryRail` recall reaching the model + write-back, `skip_tool` replacement.
- `interrupt.rs` → confirm pause shape and persisted state, approve resume, reject with and without feedback, session-seeded auto-confirm, ask-user answer as tool result, two pending calls pausing once and resuming individually, and the streaming pause emitting plus mirroring its `__interaction__` frame.
- `llm_agent.rs` → config mapping onto `ReActAgent`, provider fallback, invoke/stream output filtering, prompt-template override, interrupt pass-through.
- `workflow_agent.rs` → single-workflow invoke/stream, model-driven selection with default-response fallback, workflow interrupt/resume, no-workflow error.
- `memory_provider.rs` → the external long-term-memory provider seam: `system_prompt_block` reaching the prompt, prefetch text injected once per invoke as a system message, provider tool schemas mounted and a model tool call dispatched through `handle_tool_call`, `sync_turn` receiving the user/assistant texts, `on_session_end` firing once for an owned session only, failures tolerated, and `clear_memory_provider` removing the rail.
- `kv_cache_seam.rs` → the KV-cache seam: an inert default-config hook, affinity recording the prefix and provider input tokens, an attached action runtime prefetching once and releasing its inference lease, typed `Error::Unsupported` for the Redis/Distributed backends with the installed hook surviving, an unchanged no-hook run, and a compile-level check of the crate-root promotions.
- `usage.rs` → the usage telemetry seam: pre-call `context.usage` frames carrying the measured block, one post-call frame per model call (provider token counts + snapshot) with its session mirror, the buffered snapshots read by `last_usage`/`drain_usage_events` on a non-streaming `invoke`, the provider report reaching `ContextEngine::kv_cache_usage`, the analyzer-less run (one pre-call frame, no snapshot, `Error::Config` from `usage_report`), `usage_report` reconciling a supplied provider total, and `configure` keeping an installed analyzer/recorder.
- `run_control.rs` → the run's escape hatches: the completion deadline ending a hung model stream, `ReActAgent::abort` cancelling one, a hung tool failing after `tool_timeout_secs`, and a dropped stream stopping its run.
- `controller_run_control.rs` → LLM-requested pause/cancel against the background scheduler: the executor hooks run, the run stops, and the recorded status survives the driver.

## Model fakes (`ScriptedModel`)

- `openjiuwen_test_support::ScriptedModel` is the only model fake in the crate (unit tests included); no suite uses `MockLlmServer`, `TempWorkspace`, `TempHomeGuard`, or `EnvGuard` — isolation comes from in-process sessions and temp dirs created inside a test.
- Construction, in rough frequency across `src/` + `tests/`: `ScriptedModel::builder().….build()` (61), `ScriptedModel::text("…")` (17), `ScriptedModel::new([ScriptedTurn::…])` (3).
- Builder calls used: `.completion(MockCompletion::…)` (34), `.respond_text("…")` (54, the answer served when the script runs out), `.stream_text("…")` (6), `.fail(ScriptedError::llm("boom"))` (2), `.model_name("…")` (only `src/rails/model_backup.rs`).
- `MockCompletion` variants used: `tool_call(name, json!({…}))` (29), `tool_calls(vec![MockToolCall::new(name, args), …])` (2, ids pinned with `.with_id("call_a")`), `text("a").with_usage(Usage::new(3, 4))` (1), `reasoning("think", "answer")` (2).
- Failure turns: `ScriptedTurn::fail("boom")` and `ScriptedTurn::fail_with(ScriptedError::llm("boom"))` (3, always through `ScriptedModel::new`), or the builder's `.fail(...)`.
- Inspection: `call_count()` (19), `calls()` (17 — index `calls[1].messages` or filter by `Role::Tool`), `last_call()` (6 — `.expect("one model call")` then `.messages` / `.request.tools`); each element is `openjiuwen_test_support::ModelCall { mode, messages, request }`.
- No crate file calls `remaining()`; nothing asserts on `mode`/`CallMode` either.
- `ScriptedModel` is `Clone` but not `Debug`, and the agent takes `Arc<dyn Model>`, so every suite has a `fn shared(model: &ScriptedModel) -> Arc<dyn Model> { Arc::new(model.clone()) }` (16 sites: `tests/react_agent.rs:24`, `tests/interrupt.rs:33`, `tests/rails.rs:44`, `tests/workflow_agent.rs:92`, `tests/streaming.rs:25`, plus per-module copies); it keeps the original handle for the call-history assertions.
- `tests/llm_agent.rs` casts inline instead: `agent.react().set_model(Arc::new(model.clone()) as Arc<dyn Model>)`.

## Constraints

- Every test is offline and deterministic; no wall-clock assertions beyond short timeouts and no external services.
- Shared fixtures live in `common/mod.rs` (`SESSION_ID`, session/agent builders, recording and failing tools, the input log); suites that need bespoke rails, stores, or workflows keep those helpers local to the file.
- A suite pulls the shared fixtures in with a plain `mod common;` (no `#[path]` attribute — `tests/common/mod.rs` is auto-discovered); `interrupt.rs` and `rails.rs` instead carry local `log`/`recording_tool` helpers and do not declare it.
- Every suite file opens with `#![allow(clippy::unwrap_used, clippy::expect_used)]` (plus `clippy::assert_is_empty` in most) so `unwrap`/`expect` in tests stay lint-clean; `common/mod.rs` adds `#![allow(dead_code, unreachable_pub)]` because each suite is a separate crate and only uses some helpers.
- All async tests are plain `#[tokio::test]` (current-thread runtime); no `flavor = "multi_thread"`, no `#[serial]`, and no `TempHomeGuard`/`EnvGuard`/global `Runner` — isolation comes from in-process sessions and `ScriptedModel`.
- Shared state in tests is `parking_lot::Mutex<Vec<_>>` held in an `Arc` (clone the `Arc` inside `Fn` tool handlers, then `lock()` inside the async block); the guard is never held across an `await`.
- Tests assert Python-compatible serialized names (`result_type`, `termination_reason`, `interrupt_kind`/`interrupt_owner`, `__interaction__`, `llm_output`, `workflow_final`).
- `interrupt.rs` and `rails.rs` keep their own local fixtures (`SESSION_ID`, `agent()`, `new_session()`, `recording_tool()`) instead of `mod common`, because they need bespoke rails/sessions.
- The resume payload the interrupt suites send is `{"query": {"user_inputs": {"<tool_call_id>": <answer>}, "raw_inputs": null, "interrupt_answers": {}}, "conversation_id": SESSION_ID}`; `interrupt_answers` is keyed by interrupt id and wins over the `user_inputs` tool-call-id entry (`openjiuwen_agent::match_answers`).

## Routing Table

- `./common/` → shared fixtures (tools, sessions, workflows)
