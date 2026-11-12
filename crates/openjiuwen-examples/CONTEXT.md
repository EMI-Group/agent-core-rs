# openjiuwen-examples

## Intent

The runnable example set of the Rust rewrite: a curated tour of the SDK that a newcomer can read top to bottom.
Mirrors the Python reference tree `examples/` (read-only foreign repo), reduced to twelve files that compile, run offline, and each teach one seam of the API.
Every example is a `cargo example` binary; the crate's library in `./src/` only holds the shared setup helpers the binaries use, so the helpers are unit-testable.
The twelve files in `./examples/` are currently placeholders (`//! ...` doc line, a `TODO(example)` line, `fn main() {}`); the matrix below describes their intended content, not shipped behavior.
There is no `tests/` directory: the only tests are the library unit tests under `./src/`.
All examples are offline-capable: the six kernel examples and the harness examples run against `ScriptedModel` or the loopback `MockLlmServer`, and switch to a real provider only when `OPENJIUWEN_LIVE` and credentials are present.

## Example Matrix

| Example | Python counterpart | Runs offline | Live switch |
| --- | --- | --- | --- |
| `01_hello_agent` — model + `ReActAgent`, one `invoke` | `examples/react_agent/build_react_agent.ipynb` (first steps); the SDK has no plain hello-world script | yes (`ScriptedModel`) | `OPENJIUWEN_LIVE=1` + `MODEL_NAME` + a key |
| `02_react_with_tools` — two tools, one round | `examples/rl_calculator/tools.py` (`@tool`), `examples/react_agent/build_react_agent.ipynb` | yes (scripted tool call) | `OPENJIUWEN_LIVE=1` |
| `03_sessions_and_resume` — checkpoint + interrupt/resume | `examples/client_tool/client_tool_demo.py`, `examples/harness/worktree/single_deepagent_e2e.py` | yes | — |
| `04_workflow_pipeline` — conditional edges, `${node.field}` | `examples/intelli_router/intelliRouter_demo.py`, `examples/mcp/*/client_as_workflow.py` | yes | — |
| `05_context_and_memory` — windowing/compression + long-term memory | `examples/context_evolver/quickstart.py`, `examples/graph_memory/showcase_graph_memory.py` | yes | — |
| `06_retrieval` — chunk, embed, store, search | `examples/retrieval/showcase_text_embedding.py`, `examples/store/es_vector_store_example.py` | yes (`HashEmbedding`) | — |
| `07_mcp_tool` — an MCP-exposed tool on an agent | `examples/mcp/stdio/client_as_tool.py` | yes (in-example `McpClient`) | `OPENJIUWEN_MCP_SERVER_CMD` runs a real stdio server |
| `08_deep_agent` — `DeepAgent` in a temp workspace, file-tool round | `examples/harness/deepagent_with_worktree.py`, `examples/skill_use/main.py` | yes (`ScriptedModel`) | — |
| `09_harness_provider` — `create_harness("native", …)` start/send/events/stop | no Python example (`create_harness` has zero hits under `examples/`) | yes (loopback `MockLlmServer` as `api_base`) | — |
| `10_agent_team` — `TeamAgentSpec` → build → invoke | `examples/multi_agent/team_hybrid.py`, `examples/multi_agent/builtin_teams/handoff_customer_service.py` | yes (loopback mock + injected model resolver) | — |
| `11_a2a_server_client` — A2A server + client on loopback | `examples/a2a/server_openjiuwen_a2a_for_sdk_client.py` + `examples/a2a/client_open_a2a_sdk_to_jiuwen_server.py` | yes (loopback, `--features a2a`) | — |
| `12_self_improvement` — auto-harness orchestration + evolving trainer | `examples/agent_evolving/react_agent_evolving.py`; auto-harness has no Python example | yes (fake runner/target) | — |

Deliberately not ported: the research drops `examples/MANGO/` and `examples/PerStream/` (no `openjiuwen` imports), `mobile_gui/`, `lsp/`, the RL training pipelines (`rl_calculator`, `rl_nl2sql`, `jiuwenrl_online`), vendor-backed stores (Milvus/Elasticsearch/Chroma/OBS) and the notebooks — they need credentials, device farms, or training infrastructure and teach no SDK seam.

## Shared Support API

`src/support/` (library, unit-tested, no dev-only dependencies):

- `support::setup() -> Result<RunMode>` — loads `.env` when present, installs stderr `tracing`, prints the run mode banner, returns the mode.
- `support::mode::{RunMode, Evaluation, detect, detect_with}` — `RunMode::{Offline, Live { provider, model }}`; live requires `OPENJIUWEN_LIVE` (`LIVE_FLAG`) truthy **and** a model name **and** a credential, otherwise `Evaluation::missing` names the absent variables and the mode degrades to offline — never panics.
- `support::logging::init()` — idempotent stderr subscriber honoring `OPENJIUWEN_LOG_LEVEL`.
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
- Examples may use `anyhow`; the library in `src/` must not (it returns `openjiuwen_common::Error`).
- Each example file carries the clippy allowances it needs for binary ergonomics (`missing_errors_doc`, `missing_panics_doc`, `must_use_candidate`, `unused_async`, `items_after_statements`); `cargo clippy --all-targets -- -D warnings` must stay green.
- Examples must not depend on sibling crates directly: the umbrella `openjiuwen` is the only SDK dependency, plus dev-only `openjiuwen-test-support` and `openjiuwen-macros`.

## Routing Table

- `./src/` → shared setup helpers compiled into the library (`support::setup`, run-mode detection, logging, banner) — unit-tested here
- `./examples/` → the twelve runnable binaries and the shared example-only helper module `common/mod.rs` (see the matrix above)
- `./README.md` → reader-facing index: how to run, offline vs live, API notes
- `./Cargo.toml` → the explicit `[[example]]` list, feature gates (`a2a`, marker `mcp`), dev-dependency split
- `../openjiuwen-test-support/` → `ScriptedModel`, `MockLlmServer`, fixtures used by the offline examples (sibling — read-only, escalate writes to parent)
- `../openjiuwen/` → the umbrella crate and `prelude` the examples import from (sibling — read-only, escalate writes to parent)

## Test Strategy

- `cargo test -p openjiuwen-examples` runs the library unit tests: run-mode detection over injected env lookups (offline default, live with flag+model+key, live flag without credentials degrading to offline, falsy flag values) and banner formatting.
- The examples themselves are validated by compilation gates, not by tests: `cargo build -p openjiuwen-examples --examples` and `--examples --features a2a,mcp` must compile every example, and `cargo clippy -p openjiuwen-examples --all-targets --no-deps -- -D warnings` must be clean.
- Each example is deterministic and idempotent: scripted models, in-memory stores, temp directories, no wall-clock assertions; running one twice in a row must produce the same narrative.
- Manual smoke run per example (`cargo run -p openjiuwen-examples --example 01_hello_agent`) is the acceptance check that the narrative and the "set X to run" messages are right.

## Notes for Agents

- Offline recipe: scripted model → `ScriptedModel`; real wire path → `MockLlmServer` with `client_config()`/`request_config()`; workspace → `tempfile::TempDir`.
- Session/checkpoint seam lives in `openjiuwen::core::session`: `AgentSession::new(id)` (no checkpointer, `save_checkpoint` fails with `ErrorKind::Session`) or `AgentSession::with_options(AgentSessionOptions { session_id, checkpointer: Some(Arc::clone(&store) as Arc<dyn Checkpointer>), ..Default::default() })`; `use openjiuwen::core::session::{Checkpointer, Session}` for the trait methods `save_checkpoint(Some(id))`, `resume(id)` (returns `Ok(false)` when absent), `checkpointer()`; two same-id sessions sharing one `Arc<dyn Checkpointer>` (e.g. `MemoryCheckpointer::new()` or `FileCheckpointer::new(dir)`) are how resume is demonstrated.
- API gaps this set works around (do not "fix" them inside an example): `create_harness` has no fake-model injection, so `09` points `api_base` at a loopback server; `TeamAgentSpec::build()` resolves models from the environment, so `10` injects `NativeMemberFactory` with an `EnvModelResolver` override; there is no in-process MCP server and `FakeMcpClient` is `#[cfg(test)]`-only, so `07` implements a small `McpClient`; there is no `Interrupt` type — an interrupt is `AgentOutput::result_type == Interrupt` and resume is re-invoking the same agent with the same session and an interactive payload; `openjiuwen::prelude` omits `ReActAgent`/configs, so examples import from `openjiuwen::core::single_agent`.
- `openjiuwen-foundation` compiles MCP unconditionally, so the crate's `mcp` feature is a documented marker; `a2a` is a real passthrough and gates example 11.
- Adding an example means: a `[[example]]` entry in `Cargo.toml`, a row in the matrix above and in `README.md`, and a shared helper rather than duplicated setup code.
