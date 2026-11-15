# openjiuwen-examples

## Intent

The runnable example set of the Rust rewrite: a curated tour of the SDK that a newcomer can read top to bottom.
Mirrors the Python reference tree `examples/` (read-only foreign repo), reduced to twenty-six example binaries (plus the shared `examples/common/mod.rs` helper module) that compile, run offline, and each teach one seam of the API.
Every example is a `cargo example` binary; the crate's library in `./src/` only holds the shared setup helpers the binaries use, so the helpers are unit-testable.
`tests/examples_offline.rs` executes every example binary in a clean offline environment, so the offline claim is enforced rather than asserted.
Examples that need a model run against `ScriptedModel` or the loopback `MockLlmServer`, and the rest use in-memory stores, hash embeddings, loopback sockets, or temp files; a real provider is reached only when `OPENJIUWEN_LIVE` and credentials are present.
Every example prints its narrative to stdout and diagnostics to stderr; the narrative output is deterministic, but identifiers such as generated UUIDs, temp paths, interrupt ids, and ephemeral ports vary between runs.

## Example Matrix

| Example | Python counterpart | Runs offline | Live switch |
| --- | --- | --- | --- |
| `01_hello_agent` — model + `ReActAgent`, one `invoke` | `examples/react_agent/build_react_agent.ipynb` (first steps); the SDK has no plain hello-world script | yes (`ScriptedModel`) | `OPENJIUWEN_LIVE=1` + `MODEL_NAME` + a key |
| `02_react_with_tools` — two tools, one round | `examples/rl_calculator/tools.py` (`@tool`), `examples/react_agent/build_react_agent.ipynb` | yes (scripted tool call) | — (offline only) |
| `03_sessions_and_resume` — checkpoint + interrupt/resume | `examples/client_tool/client_tool_demo.py`, `examples/harness/worktree/single_deepagent_e2e.py` | yes (`ScriptedModel` + temp `FileCheckpointer`) | — |
| `04_workflow_pipeline` — conditional edges, `${node.field}` | `examples/intelli_router/intelliRouter_demo.py`, `examples/mcp/*/client_as_workflow.py` | yes | — |
| `05_context_and_memory` — windowing/compression + long-term memory | `examples/context_evolver/quickstart.py`, `examples/graph_memory/showcase_graph_memory.py` | yes | — |
| `06_retrieval` — chunk, embed, store, search | `examples/retrieval/showcase_text_embedding.py`, `examples/store/es_vector_store_example.py` | yes (`HashEmbedding`) | — |
| `07_mcp_tool` — an MCP-exposed tool on an agent | `examples/mcp/stdio/client_as_tool.py` | yes (in-example `McpClient`) | `OPENJIUWEN_MCP_SERVER_CMD` runs a real stdio server |
| `08_deep_agent` — `DeepAgent` in a temp workspace, file-tool round | `examples/harness/deepagent_with_worktree.py`, `examples/skill_use/main.py` | yes (`ScriptedModel`) | — |
| `09_harness_provider` — `create_harness("native", …)` start/send/events/stop | no Python example (`create_harness` has zero hits under `examples/`) | yes (loopback `MockLlmServer` as `api_base`) | — |
| `10_agent_team` — `TeamAgentSpec` → build → invoke | `examples/multi_agent/team_hybrid.py`, `examples/multi_agent/builtin_teams/handoff_customer_service.py` | yes (loopback mock + injected model resolver) | — |
| `11_a2a_server_client` — A2A server + client on loopback | `examples/a2a/server_openjiuwen_a2a_for_sdk_client.py` + `examples/a2a/client_open_a2a_sdk_to_jiuwen_server.py` | yes (loopback, `--features a2a`) | — |
| `12_self_improvement` — auto-harness orchestration + evolving trainer | `examples/agent_evolving/react_agent_evolving.py`; auto-harness has no Python example | yes (fake runner/target) | — |
| `13_context_processors` — token accounting, truncation, tool-result offload, summarization, reload | `examples/context_evolver/quickstart.py` (engine-level; no example drives the processor seam) | yes (scripted summary, in-memory store) | `OPENJIUWEN_LIVE=1` (summary text) |
| `14_workflow_components` — sub-workflow, HTTP request, questioner HITL resume in one workflow | `examples/workflow_agent/multi_workflow_agent_demo/multi_workflow_agent_demo.py` (questioner only; no sub-workflow or HTTP component example) | yes (loopback `TcpListener` + `MemoryCheckpointer`) | — |
| `15_retrieval_quality` — BM25 ranking, query rewriting, DOCX/XLSX/PDF parsing | `examples/retrieval/showcase_query_rewriter.py` (LLM rewrite only; no sparse or parsing example) | yes (`Bm25Index` + `HashEmbedding` + generated fixtures) | `OPENJIUWEN_LIVE=1` (model rewrite) |
| `16_memory_lite` — `CodingMemory` over `MemoryEngine` and `LocalCodingMemory` (`MEMORY.md` + FTS5) | — (no Python example; Python `memory/lite/`) | yes (in-memory engine, temp workspace, SQLite FTS5) | — |
| `17_team_persistence` — SQLite team store plus the organisation layer | — (no Python example; Python `agent_teams/tools/database/` and `organization/`) | yes (temp SQLite file, no model) | — |
| `18_team_attach` — `AttachServer`/`AttachClient` over a Unix socket | — (no Python example; Python `agent_teams/external/` over shared storage and ZeroMQ) | yes (Unix socket in a temp dir) | — |
| `19_symphony_skills` — SKILL.md discovery, fingerprinted inventory, lexical search, registry/graph | — (no Python example; Python `symphony/discovery/`) | yes (temp files, no model) | — |
| `20_prompt_tuning` — `dev_tools::tune` cases, scoring, `InstructionOptimizer`, `Trainer` | `examples/agent_evolving/react_agent_evolving.py` | yes (`ScriptedPromptRunner` + `DeterministicEvaluator`) | `OPENJIUWEN_LIVE=1` (`ModelEvaluator`) |
| `21_native_tokenizers` — generated `tokenizer.json` + tiktoken `.model`, selector fallback ladder, artefact cache | — (no Python example; Python `context_engine/tokenizer/`) | yes (generated artefacts, no feature flag) | — |
| `22_context_telemetry` — `context.usage` snapshots plus `context.compression_state` records over a compressed transcript | — (no Python example) | yes (`ApproxTokenCounter` + scripted summary) | `OPENJIUWEN_LIVE=1` (summary text) |
| `23_memory_dreaming` — `DreamOrchestrator` ticks, deterministic consolidator, backoff, temp SQLite queue | — (no Python example; Python `memory/dreaming/`) | yes (temp SQLite, injected clock) | — |
| `24_runner_spawn_registry` — registry routing, in-process + external-command backends, health, `active_spawns` | — (no Python example) | yes (in-process echo + self re-exec) | — |
| `25_harness_moa_persistent` — MoA consult/rank/aggregate + journal replay and resume plan | — (no Python example) | yes (`ScriptedModel` + file journal) | — |
| `26_online_training_service` — task/capture/ledger/train/status over `MemoryOnlineStore` + local executor | — (no Python example; Python `agent_evolving/online/`) | yes (in-memory store, temp artefacts) | — |

Deliberately not ported: the research drops `examples/MANGO/` and `examples/PerStream/` (no `openjiuwen` imports), `mobile_gui/`, `lsp/`, the RL training pipelines (`rl_calculator`, `rl_nl2sql`, `jiuwenrl_online`), vendor-backed stores (Milvus/Elasticsearch/Chroma/OBS) and the notebooks — they need credentials, device farms, or training infrastructure and teach no SDK seam.

## Shared Support API

`src/support/` (library, unit-tested, no dev-only dependencies):

- `support::setup() -> Result<RunMode>` — loads `.env` when present, installs stderr `tracing`, prints the run mode banner, returns the mode.
- `support::mode::{RunMode, Evaluation, detect, detect_with}` — `RunMode::{Offline, Live { provider, model }}`; live requires `OPENJIUWEN_LIVE` (`LIVE_FLAG`) truthy **and** a model name **and** a credential, otherwise `Evaluation::missing` names the absent variables and the mode degrades to offline — never panics.
- `support::logging::init()` — idempotent stderr subscriber honoring `OPENJIUWEN_LOG_LEVEL` and `OPENJIUWEN_LOG_FORMAT`.
- `support::banner::{title, step, note, answer}` — the only stdout narrative helpers; stdout stays clean of logs.

`examples/common/mod.rs` (example-only helpers, may use dev-dependencies, reached with `mod common;`):

- `setup() -> anyhow::Result<RunMode>` — the `anyhow`-flavored wrapper every example calls first.
- `scripted(answer)`, `scripted_tool_round(tool, args, answer)`, `arc_model(ScriptedModel)` — build `Arc<dyn Model>` scripted fakes.
- `mock_server(items)` + `text_response(text)` — start a loopback `MockLlmServer` for examples that must exercise the real HTTP path.
- `live_model()` — async `init_model` from environment, used only on the live branch.
- `tempdir()` — a `TempDir` the example must keep alive for its whole run.
- `skip(reason, hint)` — prints why an optional path is off and returns `Ok(())` for a clean exit 0.

## Constraints

- Only this crate is writable from here; report sibling-crate or manifest needs upward instead of editing them.
- Every example: `fn main() -> anyhow::Result<()>` (or `#[tokio::main]` + `-> anyhow::Result<()>`), no `unwrap`/`expect`/`panic!` outside tests, no secrets printed (redact keys, print host/port only), and exit code 0 with an actionable "set X to run" message when a prerequisite is absent.
- Offline by default; the only network is loopback (`127.0.0.1`) or an explicitly configured live provider. No example writes outside its temp directory.
- One seam per file, roughly ≤300 lines; shared logic belongs in `src/support/` (library-safe) or `examples/common/mod.rs` (dev-only) — never copy-pasted between examples.
- Over-length examples are deliberate and all stay under the workspace 1000-line baseline: `04` (317, conditional edges plus the fan-in workaround), `05` (410, context + memory), `07` (310, in-example `McpClient` plus the opt-in stdio branch), `09` (330, provider registry plus the one-turn event walk), `12` (419, auto-harness + trainer), `13` (504, five processor stages plus the reload half), `14` (463, sub-workflow + HTTP + questioner), `15` (727, BM25 + rewriting + parsing plus hand-written ZIP and PDF fixture writers), `16` (396, engine-backed coding memory plus the local FTS workspace), `17` (798, team store + org layer plus the CAS and typed-`Unsupported` walks), `18` (643, hosted demo team + client walk + raw-frame probes), `19` (458, discovery + inventory + search + registry/graph), `20` (459, cases + scoring + optimisation + live switch), `21` (671, two artefact formats + five ladder rungs + cache misses), `22` (779, snapshot table + recorder states + reconciliation + control engine), `23` (610, queue seeding + retry/failure/cancel + backoff ladder), `24` (600, registry + routing + two backends + NDJSON child loop), `25` (729, MoA pipeline + rail hooks + journal replay + resume plan), `26` (549, five state machines + stop and deferred paths).
- Examples may use `anyhow`; the library in `src/` must not (it returns `openjiuwen_common::Error`).
- Each example file carries the clippy allowances it needs for binary ergonomics (`missing_errors_doc`, `missing_panics_doc`, `must_use_candidate`, `unused_async`, `items_after_statements`); `cargo clippy --all-targets -- -D warnings` must stay green.
- Examples must not depend on sibling crates directly: the umbrella `openjiuwen` is the only SDK dependency, plus the dev-only crates `openjiuwen-test-support`, `openjiuwen-macros`, `tokio-util` (for `CancellationToken`) and `tempfile`.
- Documented deviation from that rule: `openjiuwen-memory` and `openjiuwen-agent-teams` are dev-dependencies with `features = ["sqlite"]` because the umbrella exposes `sqlite` only as an opt-in feature and `16`, `17` and `23` need the memory and team SQLite backends directly; both are workspace crates, so the build stays offline.
- `chrono` is a dev-dependency for the same reason: `DreamOrchestrator::tick` takes an injected `DateTime<Utc>`, which a `TempDir` alone cannot supply.

## Routing Table

- `./src/` → shared setup helpers compiled into the library (`support::setup`, run-mode detection, logging, banner) — unit-tested here
- `./examples/` → the twenty-six runnable binaries and the shared example-only helper module `common/mod.rs` (see the matrix above)
- `./tests/` → `examples_offline.rs`, the smoke suite that runs every example binary in a clean offline environment
- `./README.md` → reader-facing index: how to run, offline vs live, API notes
- `./Cargo.toml` → the explicit `[[example]]` list, feature gates (`a2a`, marker `mcp`), dev-dependency split
- `../openjiuwen-test-support/` → `ScriptedModel`, `MockLlmServer`, fixtures used by the offline examples (sibling — read-only, escalate writes to parent)
- `../openjiuwen/` → the umbrella crate and `prelude` the examples import from (sibling — read-only, escalate writes to parent)

## Test Strategy

- `cargo test -p openjiuwen-examples` runs the library unit tests: run-mode detection over injected env lookups (offline default, live with flag+model+key, live flag without credentials degrading to offline, falsy flag values), subscriber installation, and banner formatting.
- The same command runs `tests/examples_offline.rs` (one `#[test]`), which executes every example executable built by the test profile in a clean offline environment (live `OPENJIUWEN_*`/provider variables removed, temp working directory, per-example timeout, captured stdout and stderr) and asserts exit code 0 with non-empty stdout — 25 examples with default features, 26 under `--features a2a`.
- `11_a2a_server_client` joins that run only with `--features a2a`; with the feature off its `required-features` gate means no example executable exists and the suite skips it.
- Compilation gates remain: `cargo build -p openjiuwen-examples --examples` and `--examples --features a2a,mcp` must compile every example, and `cargo clippy -p openjiuwen-examples --all-targets --no-deps -- -D warnings` must be clean.
- Each example is deterministic and idempotent: scripted models, in-memory stores, temp directories, no wall-clock assertions; the narrative output is deterministic, but identifiers such as generated UUIDs, temp paths, interrupt ids, and ephemeral ports vary between runs.
- Manual smoke run per example (`cargo run -p openjiuwen-examples --example 01_hello_agent`) is the acceptance check that the narrative and the "set X to run" messages are right.

## Notes for Agents

- Offline recipe: scripted model → `ScriptedModel`; real wire path → `MockLlmServer` (via `client_config()`/`request_config()`, or `api_base = server.base_url()` when the component builds its own client); workspace → `tempfile::TempDir` kept alive for the whole run.
- Session/checkpoint seam lives in `openjiuwen::core::session`: `AgentSession::new(id)` (no checkpointer, `save_checkpoint` fails with `ErrorKind::Session`) or `AgentSession::with_options(AgentSessionOptions { session_id, checkpointer: Some(Arc::clone(&store) as Arc<dyn Checkpointer>), ..Default::default() })`; `use openjiuwen::core::session::{Checkpointer, Session}` for the trait methods `save_checkpoint(Some(id))`, `resume(id)` (returns `Ok(false)` when absent), `checkpointer()`; two same-id sessions sharing one `Arc<dyn Checkpointer>` (e.g. `MemoryCheckpointer::new()` or `FileCheckpointer::new(dir)`) are how resume is demonstrated.
- Conditional edges (`04`): `add_conditional_connection(from, to, router)` registers **one candidate target per call**; the engine activates the union of node ids the router returns (`Router::route -> Result<Vec<String>>`). Returned ids must be declared nodes, and an empty vec ends the branch.
- `04` also works around a fan-in limit: one `set_end_comp` exists, `WorkflowOutput::result` is exactly `state[declared_end]`, and fan-in is CNF — a diamond is expressed by ending each branch in a component whose `Component::route` forwards to `end` rather than declaring two edges into it.
- Deep-agent file tool (`08`): `write_file` with `{"file_path": "<workspace-relative>", "content": "..."}`; `Observation` has no success flag, so a round is proven by reading the file back.
- Native harness turn (`09`): with `enable_task_loop=false` one turn makes exactly one model request and emits six events — `state_changed` ×2, `turn_lifecycle/started`, `output/answer`, `usage_updated`, `turn_lifecycle/finished`; `events()` is single-consumer, so drain with a timeout **and** an event cap.
- `10` injects `NativeMemberFactory::new(Arc::new(EnvModelResolver))` through `TeamBuildDeps::with_harness_factory`; the factory copies the resolved model's name and provider onto the member's `DeepAgentConfig`, so the resolver only supplies the client (`api_base`/`api_key`) and no wrapper model is needed.
- `11` uses `A2aServer::serve_in_background(CancellationToken::new())` + `ServerHandle::{local_addr, shutdown, join}`, and logs the ephemeral port to stderr and prints only the host on stdout, so unlike `09`/`10` (which print the mock server's base URL) the varying port never reaches stdout; `11`, `12`, `13`, `23` and `25` name `tokio_util::sync::CancellationToken` directly, which is why `tokio-util` is a dev-dependency.
- `13` drives `ContextPipeline` directly because `ContextEngine::compress` rewrites whole contexts and reports only before/after totals, and it constructs `SummarizationProcessor` with an explicit `Model` (scripted offline, `live_model()` in live mode).
- `14` builds `HttpComponentConfig` with `serde_json::from_value` because the nested configuration types are not re-exported, and it keeps the questioner in the parent because a sub-workflow `resume` re-runs the child from scratch without forwarding the parent's seeded answers.
- `15`'s ZIP writer is local because the crate's `parsing::zip` is reader-only, and the auto loader stamps `metadata.title` with the file stem over the `dc:title` the DOCX parser read.
- `16` requests `SearchMode::Keyword` because the default `Hybrid` logs a no-embedding warning when no model is attached.
- `17` records two store behaviours: `insert_message` writes the caller's stamp verbatim, and `add_dependency` records the edge without changing the dependent task's status.
- `18`'s demo handler returns a typed `Unsupported` for the five ops it does not model (`Broadcast`, `ClaimableTasks`, `TaskDetail`, `UpdateTask`, `InvokeTool`); the raw-frame probes assert the typed `validation` kind rather than printing it blindly.
- `19`'s `SkillFs` logs the absolute skill path to stderr as library diagnostics; the example renders its own diagnostics relative to the scan root so no temporary path reaches stdout.
- `20` carries `Case.tools` through the JSON round trip, but no evaluator consumes it; the tool declaration is data, not a scoring input.
- API gaps this set works around (do not "fix" them inside an example): `create_harness` has no fake-model injection, so `09` points `api_base` at a loopback server; there is no in-process MCP server and `FakeMcpClient` is `#[cfg(test)]`-only, so `07` implements a small `McpClient` and gates the real stdio path on `OPENJIUWEN_MCP_SERVER_CMD`; there is no `Interrupt` type — an interrupt is `AgentOutput::result_type == Interrupt` and resume is re-invoking the same agent with the same session and an interactive payload; `openjiuwen::prelude` carries the agent/model/session/harness surface (`01` and `08` import from it), so examples fall back to `openjiuwen::core::single_agent` only for items the prelude omits (`ConfirmInterruptRail`).
- `openjiuwen-foundation` always compiles MCP in this graph (nothing here disables its default `mcp` feature), so the crate's `mcp` feature is a documented marker; `a2a` is a real passthrough and gates example 11.
- `21`: `load_native_tokenizer` returns a `SelectedTokenizer` (only counts); use `NativeTokenizer::{from_json_str, load}` (or `from_tiktoken_model_file`) when encode/decode ids are wanted. Tokenizer `.json`/`.model` loading needs no Cargo feature. The artefact cache layout is `<cache>/huggingface/<id>/<revision>/tokenizer.json`; a miss is `resolve_cached() == None` plus a compact `last_error()` code (`tokenizer_not_in_local_cache`, `remote_download_disabled`, …), never an error, and artifact-cache helpers live at `context_engine::tokenizers::artifacts::*` rather than the crate root.
- `22`: `context.usage` snapshots are return values (`PreparedContext.usage`, `usage_snapshot`, `analyze`) and never session-stream frames; only `context.compression_state` records can reach a sink or session, and an engine without an analyzer answers a typed `config` error.
- `23`: `tick(now)` is deterministic (no clock read); `run_until_cancelled` sleeps in real time and returns `Err(Error::Cancelled)` on its normal exit; a consolidator must return a terminal state and must not touch the repository; the dream queue only moves `dream_jobs` through its state machine.
- `24`: `SpawnerRegistry` alone does not fill `active_spawns` — dispatch through `Runner::spawn_agent`/`spawn_agent_with`. The external-command child speaks NDJSON on stdout, so child mode must not print a banner; `tokio` has no `io-std` here, so the child loop uses blocking `std::io`. A dead idle child stays listed until a probe or `shutdown` observes it.
- `25`: `DeepAgentConfig` has no `moa` field, so MoA is standalone (`consult`/`rank_with_config`/`Aggregator`) or host-registered through `DeepAgentBuilder::rail`; the persistent agent is a file-based journal (`ExecutionLog`) plus `FileSessionStore`/`FileLinkStore`, and `resume(...)` rebuilds the window from the journal — no model is involved.
- `26`: capture only stages raw JSON under the task key and never promotes it into the training ledger, which `SampleLedger::save_once` (or the HTTP upload route) feeds; `PpoTrainingExecutor` is the typed-`Unsupported` GPU path, shown twice (health probe and a failed run).
- Adding an example means: a `[[example]]` entry in `Cargo.toml`, a row in the matrix above and in `README.md`, and a shared helper rather than duplicated setup code.
