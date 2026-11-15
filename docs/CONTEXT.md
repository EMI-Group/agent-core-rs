# Docs

## Intent

Architecture and parity documentation for the Rust rewrite of the openJiuwen Python SDK (v0.1.20).
The Context Tree (`CONTEXT.md` per crate) is the primary source of intent and API surface; these documents hold the cross-cutting contracts every crate shares.

## Contents

- `ARCHITECTURE.md` — §1 the 30 crates in layers 0–7 plus the three development crates; §2 the shared contracts (§2.1 cards/errors, §2.2 models, §2.3 tools, §2.4 sessions, §2.5 graph/workflow, §2.6 agents/runner, §2.7 harness protocol, §2.8 frame and event streams, §2.9 context engine, §2.10 memory providers, §2.11 observability, §2.12 resume and journal); §3 async/streaming model; §4 configuration and environment; §5 feature flags; §6 testing infrastructure; §7 porting method.
- `PARITY.md` — the status legend, the per-Python-path package matrix, wire/behavior compatibility, the per-crate deferred-feature table, the `Error::Unsupported` surface, intentional differences, and the verification gates.

## API Surface

- `PARITY.md` §Package Matrix — one row per Python public path with the owning Rust crate, a status (`Implemented` / `Subset` / `Stub`), and the genuine remaining gaps.
- `PARITY.md` §Deferred Features — the authoritative per-crate gap list, each entry carrying a reason category (vendor SDK, external service/broker, GPU/ML stack, platform automation, incompatible format/design).
- `PARITY.md` §`Error::Unsupported` surface — the typed seams by module path, plus §Gaps that are not typed seams for behavioural shortfalls.
- `PARITY.md` §Verification — the gate commands, the `default-members` caveat, and the `#[ignore]` inventory.
- `ARCHITECTURE.md` §2 — the public seams whose signatures require updating every dependent crate when changed.
- `ARCHITECTURE.md` §5 — the complete feature-flag table, including the `default = []` crates, the empty `live-tests` gates and the dead gates.

## Constraints

- Documentation describes the current state; history lives in git, so a fixed gap is deleted rather than annotated as fixed.
- Keep the parity matrix and the deferred table current whenever a crate's scope changes.
- Write one sentence per line so git diffs stay granular, and keep each file under ~60 KB.
- Do not duplicate per-crate API details here; link to the crate `CONTEXT.md`.
- State only verified facts, and present tense throughout: no changelog wording, no dates, no "was X, now Y".
- Parity numbers (test and scenario counts) come from a measured run of the gate, never from an earlier revision.

## Notes for Agents

- A crate's own `CONTEXT.md` plus its source is the source of truth; `PARITY.md` and `ARCHITECTURE.md` are derived from them and go stale fastest.
- Re-verify a matrix row against the crate (and `rg -n "Unsupported" crates/<crate>/src`) rather than against the previous wording of the document.
- Every deferred entry needs a concrete reason: vendor SDK, external service/broker, GPU or ML stack, platform automation, or an incompatible format/design.
- A subsystem may be implemented while a documented seam still rejects a narrower path; record the narrower fact instead of a blanket "deferred".
- Racy-looking counts (ignored tests, scenarios, examples) are cheap to re-measure with `cargo test --list` or an `rg` count; prefer measuring over copying.
