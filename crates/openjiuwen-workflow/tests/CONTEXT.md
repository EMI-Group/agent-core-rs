# tests (openjiuwen-workflow)

## Intent

Crate-level behavior tests for `openjiuwen-workflow`, driven only through the public API (`Workflow`, `ComponentOptions`, `WorkflowRunOptions`, built-in components, `ComponentRegistry`) plus `openjiuwen_session::WorkflowSession` and `openjiuwen_test_support::ScriptedModel`/`wiremock`.
Each file ports a Python unit test suite and cites the mirrored test in a comment (e.g. `// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_workflow`).
Everything is offline and deterministic: no external network, loopback `wiremock` where HTTP is involved, no sleeps for ordering, no stdout.

## API Surface

Shared fixtures live in `common/mod.rs` (`mod common;` in every suite, `#![allow(dead_code)]` where a suite does not use all of them):

- `session(id) -> Arc<dyn Session>` — `WorkflowSession::new(id)`, in-process, no checkpointer.
- `session_with_checkpointer(id, Arc<dyn Checkpointer>) -> Arc<dyn Session>` — via `WorkflowSessionOptions`.
- `card(id) -> WorkflowCard` — `WorkflowCard::new(id)`.
- Components: `EchoComponent` (`{"echo": inputs}`), `FixedComponent::new(output)` (records `calls()`/`call_count()`), `CounterComponent` (`{"count": n}`, one-based), `StreamingProducer::new(frames)` (`Stream`), `TransformProducer::new(frames)` (`Transform`), `CollectorComponent` (`{"collected": inputs}`), `InteractingComponent::new(question)` (HITL), `FailingComponent::new(reason)`, `WaitOnCancellationComponent`.
- `b_of_a_tool()` — `FunctionTool` fixture; `llm_component(model, user_template)` — `LlmComponent` with only a user prompt; `end_with_template(template)` — `EndComponent` with `EndConfig`.
- `assistant_text(&[Value]) -> String` — joins `result`/`response`/bare-string chunk text.

Per-file suites:

- `linear.rs` — `start → llm → end` with `ScriptedModel`: result shape, session global lane, cancellation, build failure, concurrent `invoke` on one `Workflow`.
- `references.rs` — `${node.field}` grammar (dotted, `[i]`, `[-1]`, `['k']`), whole-node refs, missing → `null`, literal rule, `outputs_schema` projection, root/`${inputs.x}` refs.
- `llm_component.rs` — prompt assembly from resolved inputs, `output_config` key, default `result` key, streaming deltas as buffered frames, two LLM nodes chaining.
- `llm_semantics.rs` — `enable_history` splicing/recording (and the history-disabled no-op), `response_format` prompt rewriting, JSON fence/validation/projection, one-frame JSON streaming.
- `streaming.rs` — indexed `end node stream` frames plus terminal `workflow_final`, `invoke` returning collected wires, custom frames, mode filtering, single `Err` on failure, `Collect` over a buffered producer, early drop, empty producer.
- `end_template.rs` — `{{path}}` templates: batch rendering, missing segments, array templates streaming one frame per element, producer projection, end node's own inputs.
- `branch.rs` — `BranchComponent`/`BranchRouter` selected-target routing, Rust-only default target, `101021` no-match, custom graph router, undeclared target, agreeing routers, empty route.
- `fan_in.rs` — CNF barrier versus `wait_for_all = false` first arrival: any-arrival firing, re-firing on a later arrival, not-yet-produced keys as `null`, layered superstep ordering.
- `builder_validation.rs` — all build-time problems joined into one `ErrorKind::Workflow` error; codes `100010`–`100015`; `ComponentOptions` reflected in the spec.
- `checkpoint_resume.rs` — `MemoryCheckpointer` resume after failure, HITL `INPUT_REQUIRED` + `resume`, snapshot lifecycle, seeded answers, superstep replay.
- `exception_recovery.rs` — `default_outputs` recovery after retries, `outputs_schema` projection of the defaults, downstream refs resolving; `interrupt` and stream failures still `[100054]`.
- `session_state.rs` — component vs global session lanes, `dump_state`, trace spans, cancellation via `CancellationToken` + `Notify`, session stream publication.
- `card_spec.rs` — `WorkflowCard`/`WorkflowSpec` JSON round-trip, unknown-field tolerance, `spec()` vs `config().spec`, `to_mermaid`, per-node spec defaults (including `exception_config: None`); mostly sync tests (no runtime).
- `draw.rs` — `Workflow::draw`/`to_svg` Mermaid vs SVG, determinism, XML escaping, rounded start/end, empty/single/disconnected specs, and `DrawFormat::Png` as a typed `Unsupported` rejection that leaves Mermaid/SVG untouched.
- `retrieval_component.rs` — `SimpleKnowledgeBase` + `HashEmbedding` + `InMemoryVectorStore`; `documents` output; `101102` on blank query.
- `tool_component.rs` — `ToolComponent` maps failures to `error_code` data (`0`, `102000`, `182012`) and resolves the tool input schema.
- `loop_component.rs` — number/array/condition loops, session limit cap, `101040` when the bound is missing.
- `http_component.rs` — `HttpRequestComponent` end to end against loopback `wiremock`: encoding, response projection, `response_data_property`/`response_mode`, 4xx/5xx as data, retries, `[102000]` transport failures, builtin `"http_request"` factory.
- `questioner_component.rs` — `start → questioner → end` over a checkpointed session, `ScriptedModel` extraction, `INPUT_REQUIRED` plus `resume` per question, continue-ask wording, `max_response`; opt-in chat history (prior turns reaching the extraction prompt, recorded user/assistant turns, round trimming, the default-off regression guard).
- `intent_detection.rs` — model path via `ScriptedModel`, offline keyword rule path (case-insensitive distinct hits, lowest-id ties, default class), `IntentRouter` successor selection from graph state.
- `react_agent.rs` — resolved inputs reach the `AgentRunner` unchanged, `{output, result_type}` normalization, error/interrupt data, `ModelReActRunner` tool round and every `on_max_iterations` policy.
- `sub_workflow.rs` — child end output passthrough, isolated child state lanes for reused node ids, nesting limit, forwarded frame wires, stream cache, relayed HITL as the parent's `INPUT_REQUIRED`, and resume answer forwarding (bare and dotted child interaction ids, replay of earlier answers on a second resume via the recorded parent lane).
- `single_component.rs` — `execute_single_component[_with]`: `inputs_schema` resolution, `outputs_schema` projection, interrupt reported as `INPUT_REQUIRED` with interaction wires, seeded answer completing the call, error propagation, `generate_workflow_key`, `WorkflowChunkType` wire values.
- `public_api.rs` — crate-root vs module-path parity (type identity for the HTTP config tree and the LLM/ReAct aliases), a Python-shaped `HttpComponentConfig` JSON round-trip with unknown keys, HTTP enum wire strings, max-iteration constants and exported status codes.

## Constraints

- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]` (library lints stay strict); several also blanket-allow `dead_code`/doc/clippy style lints because `mod common;` pulls unused fixtures.
- Tests only use public APIs — no `crate::` internals; `common/mod.rs` re-implements the fakes it needs on top of the `Component` trait.
- Gates: `cargo fmt -p openjiuwen-workflow`, `cargo test -p openjiuwen-workflow`, `cargo clippy -p openjiuwen-workflow --all-targets -- -D warnings`, all offline.
- `common::session` is the only session factory; the engine only needs the `Session` trait, so any implementation would work.
- Tests never mutate process-global state: no `EnvGuard`/`TempHomeGuard`, and no `#[serial]` anywhere in the workspace (`serial_test` is not a dependency).

## Routing Table

No child directories; every file is a top-level integration test target, listed above.

## Gotchas for Reuse Outside This Crate

- `common/mod.rs` is not importable from another crate's tests: copy `session`/`card`/`EchoComponent`/`FixedComponent` into the new crate (or use `openjiuwen-test-support` fixtures) rather than `#[path = ...]`-including it.
- From the `openjiuwen-e2e` crate (which depends only on the `openjiuwen` umbrella + test-support) the same items are reachable as `openjiuwen::core::workflow::*`, `openjiuwen::core::session::{WorkflowSession, Session, MemoryCheckpointer}`, and `openjiuwen::core::foundation::Model`.
- `invoke` needs a tokio runtime; the default `#[tokio::test]` (current_thread) is enough except for cancellation tests, which use `#[tokio::test(flavor = "multi_thread", worker_threads = 2)]` (`session_state.rs:390`).
- Output extraction: `EndComponent::new()` wraps non-null inputs as `{"output": …}`, so assertions read `output.result["output"]["answer"]`; a streaming end or an interrupted run makes `result` the JSON array of chunk wires.
- `invoke` merges object inputs into the session global lane, so both `${query}` and `${start.query}` resolve, but a bare `${field}` inside `outputs_schema` is `null`.
- HITL suites need `session_with_checkpointer` + `Workflow::resume`; a session without a checkpointer cannot resume an interrupted run.
- HTTP suites drive a real loopback server (`openjiuwen_test_support::wiremock`), so assertions use its `uri()` and never a hard-coded port.
