# Docs

## Intent

Architecture and parity documentation for the Rust rewrite.
The Context Tree (`CONTEXT.md` per crate) is the primary source of intent and API surface; these documents hold the cross-cutting contracts that every crate shares.

## Contents

- `ARCHITECTURE.md` — crate layering, shared Rust contracts (cards/errors, prompts/models, tools, sessions, graph/workflow, agents/runner, harness protocol, frame and event streams), async/streaming model, configuration and environment, feature flags, testing infrastructure.
- `PARITY.md` — status legend, the per-Python-path package matrix, wire/behavior compatibility, deferred features with their reasons, the `Error::Unsupported` surface, intentional differences, and verification.

## API Surface

- `PARITY.md` §Package Matrix — one row per Python public path, with the Rust crate that owns it, a status (`Implemented` / `Subset` / `Stub` / `Deferred`), and the genuine remaining gaps.
- `PARITY.md` §Deferred Features and §`Error::Unsupported` surface — the authoritative list of what is not implemented and why.
- `ARCHITECTURE.md` §2 — the public seams whose signatures require updating every dependent crate when changed.
- `ARCHITECTURE.md` §5 — the complete feature-flag table.

## Constraints

- Documentation describes the current state; history lives in git.
- Keep the parity matrix current whenever a crate's scope changes.
- Do not duplicate per-crate API details here; link to the crate `CONTEXT.md`.

## Notes for Agents

- A crate's own `CONTEXT.md` plus its source is the source of truth; `PARITY.md` is derived from them and goes stale fastest.
- Re-verify a matrix row against the crate (and `rg -n "Unsupported" crates/<crate>/src`) rather than against the previous wording of the doc.
- Every deferred entry needs a concrete reason: vendor SDK, external service/broker, ML/research stack, platform automation, or an incompatible format/design.
- Parity text states current behavior only; a fixed bug is deleted from `## Known Issues`-style prose, not annotated as fixed.
