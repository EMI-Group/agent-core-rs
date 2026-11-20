# openJiuwen (Rust) — root

## Intent

Rust rewrite of the openJiuwen Python agent SDK (reference: read-only foreign repo `/home/bill/Source/agent-core`, v0.1.20).
The objective is functional and interface parity with the Python SDK's public surface: agent runtime, LLM clients, prompts, tools/MCP, sessions and checkpoints, graph/workflow engine, context engine, memory, retrieval, sys-ops, guardrails, multi-agent teams, the DeepAgent (harness) framework, and the `openjiuwen` CLI.
Internal implementation is idiomatic Rust (tokio, serde, traits, thiserror), not a transliteration of Python internals.
Python package `openjiuwen.<pkg>` maps to crate `openjiuwen-<pkg>`; `openjiuwen-core` and `openjiuwen` are umbrella crates preserving the module paths.

## Implementation Status

The workspace contains 30 crates: the full kernel, harness/protocol/providers, agent teams, extensions, dev tools, symphony, RSI/auto-harness, agent-evolving, the `openjiuwen` umbrella, the `openjiuwen` binary, plus test/e2e/example crates.
The kernel (common, foundation, session, graph, workflow, context, memory, retrieval, sysop, security, agent, multiagent, runner) is implemented; its deferred areas are peripheral adapters and distributed modes.
The DeepAgent harness is implemented as a deep subset: task loop, workspace, permissions, 20 coding tools, rails, prompts, subagents, skills, native protocol, and CLI logic.
Agent teams are implemented in-process, including the `team-member` CLI and the `openjiuwen-team-mcp` MCP server.
A2A and OTel integrations, prompt/skill/agent dev tools, the symphony offline engine, the auto-harness orchestrator, and the agent-evolving evaluation/trainer core are implemented as documented subsets.
Runtime features that depend on Python research/ML stacks, vendor SDKs, external databases/brokers, or desktop/mobile transports are not reimplemented.
Those paths keep their API shape and return `openjiuwen_common::error::Error::Unsupported` with a reason instead of silently degrading.
The complete per-package matrix, deviations, and deferred list live in `docs/PARITY.md`.

## API Surface

- `openjiuwen` — top-level umbrella: `core`, `harness`, `harness_protocol`, `harness_providers`, `agent_teams`, `extensions`, `dev_tools`, `symphony`, `rsi`, `agent_evolving`, `auto_harness`, plus a curated `prelude`.
- `openjiuwen-core` — kernel umbrella: `common`, `foundation`, `session`, `graph`, `workflow`, `context_engine`, `memory`, `retrieval`, `sys_operation`, `security`, `single_agent`, `application`, `multi_agent`, `runner`.
- `openjiuwen-cli` — binary `openjiuwen` (`run`, `chat`); `team-member` and `openjiuwen-team-mcp` ship from `openjiuwen-agent-teams`.
- `crates/` — every workspace crate, layering, status, and per-crate routing.
- `crates/openjiuwen-examples/examples/` — 26 runnable examples mirroring the Python `examples/` tree (offline by default, live with `OPENJIUWEN_LIVE=1`); a few are feature-gated (`a2a`, `mcp`).
- `crates/openjiuwen-examples/src/support/` — the shared example harness (`RunMode`, logging, banner) used by every example.
- `docs/PARITY.md` — status legend, Python→Rust package matrix, wire compatibility, deferred features, intentional differences, verification.

## Constraints

- **Compatibility contract**: preserve OpenAI-compatible and Anthropic wire formats; `*Card` JSON field names and shapes; workflow `${node.field}` references; `OPENJIUWEN_*`, `API_KEY`, `API_BASE`, `MODEL_PROVIDER`, `MODEL_NAME`, `OPENAI_*` environment variables; and `openjiuwen` CLI commands. See `docs/PARITY.md`.
- **Async-first**: tokio runtime; no blocking calls on async tasks; public streams are `Send + 'static`; cancellation via `tokio_util::sync::CancellationToken`.
- **Errors**: `openjiuwen_common::error::{Error, Result}` across libraries; no panics on user input; `anyhow` only in binaries, examples, and tests.
- **Serialization**: every public data type is `serde`-serializable, snake_case, tolerant of unknown fields; Cards round-trip through JSON and expose JSON Schema via `schemars`.
- **Logging**: `tracing` through `openjiuwen_common::logging`; no `println!`/`eprintln!` in libraries.
- **Safety**: no `unsafe` (workspace lint `deny`); no undocumented global mutable state (the process-global `Runner` is the documented exception, with local instances supported for tests).
- **Size**: files around ≤1000 lines; split by responsibility; shared types live at the lowest common ancestor crate.
- **Tests**: unit tests beside modules and in `crates/<crate>/tests/`; cross-crate tests in `openjiuwen-e2e`; fake LLM server in `openjiuwen-test-support`; `cargo test --workspace` passes offline.
- **Deferred features are typed**: unimplemented capability paths return `Error::Unsupported` with a reason and are listed in `docs/PARITY.md`; silent no-ops are not acceptable.
- **HTTP stack** is pinned once in `[workspace.dependencies]`: `reqwest 0.12` (default-features off, `json`/`stream`/`rustls-tls`/`multipart`), `axum 0.8` (`json`/`tokio`/`http1`), `http = "1"`, `url 2` (`serde`), `bytes = "1"`; tests use `wiremock 0.6`.
- `hyper`, `tower` and `tower-http` have **no** `[workspace.dependencies]` entry — crates that need them pin inline (`tower = { version = "0.5", features = ["util"] }` as a dev-dependency in `openjiuwen-extensions` and `openjiuwen-agent-evolving`).
- `http` is a workspace entry but no crate declares it directly; consumers reach it through `axum::http` (e.g. `openjiuwen-extensions/tests/a2a_server.rs`) or via `reqwest::StatusCode`.
- `mockito` and `httpmock` are absent from the workspace, `Cargo.lock`, and the local registry cache — adding them would break `--offline` builds.
- **Offline builds**: every locked dependency (`reqwest 0.12.28`, `axum 0.8.9`, `wiremock 0.6.5`, `http 1.5.0`, `url 2.5.8`, `hyper 1.12.0`, `tower 0.5.3`, `bytes 1.12.1`) is present in `~/.cargo/registry/{cache,src}/index.crates.io-*/`, and `make test` (Makefile:7) runs the workspace offline.
- **Foreign repo** is read-only; never modify it. Delegate investigation to read-only investigators.
- `OPENJIUWEN_MODEL`, `OPENJIUWEN_PROVIDER` and `OPENJIUWEN_API_BASE` are read only by the harness CLI option resolver (`crates/openjiuwen-harness/src/cli/config.rs:301-337`); library clients read `MODEL_NAME`/`MODEL_PROVIDER`/`API_BASE`/`OPENAI_BASE_URL` (`crates/openjiuwen-common/src/config/env.rs:134-158`).

## Known Issues

- The upstream is `https://gitcode.com/openJiuwen/agent-core` (Python `pyproject.toml` `[project.urls]`, reference v0.1.20); `Cargo.toml:42` now points at the canonical `https://github.com/openJiuwen-ai/agent-core` while `README.md:3` still links `https://github.com/openjiuwen/openjiuwen`.
- Root `README.md` is stale: line 24 claims the CLI runs offline (only the examples do; `run`/`chat` fail without a provider key), line 73 says 12 examples (26 exist), line 74 says `live-tests` gates provider tests (it is an empty gate; `OPENJIUWEN_LIVE` switches the examples).

## Repository Facts

- 30 workspace crates under `crates/*`; `default-members` lists 27 (all but `openjiuwen-test-support`, `openjiuwen-e2e`, `openjiuwen-examples`), so a bare `cargo build`/`cargo test` skips the three dev crates.
- Every crate inherits `version = "0.1.0"`, `edition = "2021"`, `rust-version = "1.85"`, `license = "Apache-2.0"`, `repository`, and `publish = false` from `[workspace.package]`.
- Root `LICENSE` holds the standard Apache-2.0 text (201 lines, copied verbatim from the Python reference); there is also no CI configuration, no `rustfmt.toml`/`clippy.toml`/`deny.toml`, and no `rust-toolchain` file — the `Makefile` targets are the only gate.
- `[workspace.package] repository` is `https://github.com/openJiuwen-ai/agent-core` (the canonical original-project link); the Python upstream declares `https://gitcode.com/openJiuwen/agent-core.git`.
- Only `openjiuwen-common` declares a package `description`; no crate declares `homepage`, `documentation`, `keywords`, or `categories`, and all 30 set `publish = false`.
- Workspace lints (`Cargo.toml:168-178`): Rust `unsafe_code = "deny"`, `missing_docs`/`unreachable_pub` warn; Clippy `all`/`pedantic` warn plus `unwrap_used`/`expect_used` warn.
- No `unsafe` block, `unsafe fn`, `extern "C"`, or `#[no_mangle]` exists in tracked source; FFI is confined to dependencies (`sqlx` → bundled C SQLite via `libsqlite3-sys`, `nix`/`libc` syscalls, `ring`, `inotify` behind `notify`).
- External process boundaries are real and numerous: shells/PTYs (`openjiuwen-sysop`), `git` worktrees (`openjiuwen-harness`, `openjiuwen-agent-teams`), LSP servers, vendor CLIs (`openjiuwen-harness-providers`), rmcp stdio MCP servers (`openjiuwen-foundation`), spawned `team-member`/runner children, and `docker run`/`docker exec` in agent-evolving's evaluator pipeline.
- `println!`/`eprintln!` occur only in the `openjiuwen-cli` binary crate; library crates log through `tracing`.

## Routing Table

- `./crates/` → workspace crates, dependency layers, status table, and per-crate responsibilities (`crates/CONTEXT.md`)
- `./crates/openjiuwen-examples/` → runnable examples and their offline/live harness (`crates/openjiuwen-examples/CONTEXT.md`)
- `./docs/` → architecture, parity matrix, and deferred-scope documentation (`docs/CONTEXT.md`)
- `./docs/PARITY.md` → per-package status, wire compatibility, deferred features, intentional differences, verification
- `./Cargo.toml` → workspace manifest, shared dependency versions, lints
- `./Makefile` → build/test/lint/verify entry points
