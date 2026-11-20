# Docs

## Intent

Architecture and parity documentation for the Rust rewrite of the openJiuwen Python SDK (v0.1.20).
The Context Tree (`CONTEXT.md` per crate) is the primary source of intent and API surface; these documents hold the cross-cutting contracts every crate shares.

## Contents

- `ARCHITECTURE.md` — §1 the 27 crates in layers 0–7 plus the three development crates (30 in total); §2 the shared contracts (§2.1 cards/errors, §2.2 models, §2.3 tools, §2.4 sessions, §2.5 graph/workflow, §2.6 agents/runner, §2.7 harness protocol, §2.8 frame and event streams, §2.9 context engine, §2.10 memory providers, §2.11 observability, §2.12 resume and journal, §2.13 harness host seams); §3 async/streaming model; §4 configuration and environment; §5 feature flags; §6 testing infrastructure; §7 porting method.
- `PARITY.md` — the status legend, the per-Python-path package matrix, wire/behavior compatibility (models, cards, sessions, workflow data flow, environment variables, CLI commands, documented deviations), the per-crate deferred-feature table, the `Error::Unsupported` surface, the gaps that are not typed seams, intentional differences, and the verification gates.
- `BENCHMARKS.md` — the measurement appendix: environment, the exact commands and batch/repetition sizes, median and p95 for process start (wall, peak RSS) and two pure-local operations (prompt render, graph run), the raw per-repetition values, and the limits of what each row means.
- `bench/` — the harnesses and raw JSON behind `BENCHMARKS.md` (standalone Rust package plus Python scripts).

## API Surface

- `PARITY.md` §Package Matrix — one row per Python public path with the owning Rust crate, a status (`Implemented` / `Subset` / `Stub`), and the genuine remaining gaps.
- `PARITY.md` §CLI commands — the `openjiuwen` subcommands (`run`, `chat`, `sessions`, `lsp`, `cron`, `team`, `watch`, `auto-harness`), their flags including `--resume`/`--continue`/`--worktree`, and the exit codes.
- `PARITY.md` §Deferred Features — the authoritative per-crate gap list, each entry carrying a reason category (vendor SDK, external service/broker, GPU/ML stack, platform automation, incompatible format/design).
- `PARITY.md` §`Error::Unsupported` surface — the typed seams by module path, plus §Gaps that are not typed seams for behavioural shortfalls.
- `PARITY.md` §Verification — the green outcome, the gate commands, the `default-members` caveat, the hermeticity guarantees (offline, loopback, no wall clock), the `#[ignore]` inventory and what each ignored test is for, and the suites that assert the deferred surfaces.
- `ARCHITECTURE.md` §2 — the public seams whose signatures require updating every dependent crate when changed.
- `ARCHITECTURE.md` §2.13 — the harness seams a host wires: `MemoryToolBackend`/`WorkspaceMemoryBackend` for the memory tools and `SubAgentSpawner`/`RuntimeSubAgentSpawner` for subagents.
- `ARCHITECTURE.md` §5 — the complete feature-flag table: per-crate features, defaults and effects, the `default = []` crates, the empty marker features (`live-tests`, `openjiuwen-harness`'s `native`), the `#![cfg(feature = "…")]` suite gates and the umbrella forwarders.

## Constraints

- Documentation describes the current state; history lives in git, so a fixed gap is deleted rather than annotated as fixed.
- Keep the parity matrix and the deferred table current whenever a crate's scope changes.
- Write one sentence per line so git diffs stay granular, and keep each file under ~64 KB.
- Do not duplicate per-crate API details here; link to the crate `CONTEXT.md`.
- State only verified facts, and present tense throughout: no changelog wording, no dates, no "was X, now Y".
- Record how a claim is verified (the gate command, the package suite) instead of a test or scenario count, which drifts with the next commit.
- The verification section states the measured outcome of the gate: it names the failing targets when the gate is red, and claims a green run only while every gate passes.

## Notes for Agents

- A crate's own `CONTEXT.md` plus its source is the source of truth; `PARITY.md` and `ARCHITECTURE.md` are derived from them and go stale fastest.
- Re-verify a matrix row against the crate (and `rg -n "Unsupported" crates/<crate>/src`) rather than against the previous wording of the document.
- Every deferred entry needs a concrete reason: vendor SDK, external service/broker, GPU or ML stack, platform automation, or an incompatible format/design.
- A subsystem may be implemented while a documented seam still rejects a narrower path; record the narrower fact instead of a blanket "deferred".
- Racy-looking counts (ignored tests, scenarios, examples) are cheap to re-measure with `cargo test --list` or an `rg` count; prefer measuring over copying.
- A workspace-wide `cargo test --workspace` unifies features across crates, so a test that assumes an optional feature is off can pass under `-p <crate>` and fail in the workspace gate; measure both when a feature row changes.
- `Error::Unsupported` is a tuple variant `Error::Unsupported(String)` rendered `unsupported: <msg>`, built by `Error::unsupported(impl Into<String>)` in `crates/openjiuwen-common/src/error.rs`; its `ErrorKind` is `Unsupported`.
- Two crates funnel every deferred path through a local helper that still returns `Error::Unsupported`: `openjiuwen_symphony::error::unsupported(feature)` and `openjiuwen_rsi::deferred::not_implemented(surface, reason)`, so a bare `rg "Error::unsupported"` undercounts them.
- `openjiuwen-harness-protocol` expresses capability limits as `HarnessError::UnsupportedCapability`, never `Error::Unsupported`.
- A workspace `rg` for `Unsupported` also matches test fixtures and in-`src` `CONTEXT.md` prose; verify a crate's production seams through the `§Error::Unsupported surface` module list instead of trusting a hit count.

## Routing Table

- `./ARCHITECTURE.md` → cross-cutting contracts, crate layering, feature flags, testing infrastructure
- `./PARITY.md` → per-package parity matrix, deferred features, typed seams, verification gates
- `./BENCHMARKS.md` → environment, measured process and pure-local operations, raw samples, limitations
- `./bench/` → Rust and Python measurement harnesses plus the recorded raw results (`bench/CONTEXT.md`)
