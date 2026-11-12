# Docs

## Intent

Architecture and parity documentation for the Rust rewrite.
The Context Tree (`CONTEXT.md` per crate) is the primary source of intent and API surface; these documents hold the cross-cutting contracts that every crate shares.

## Contents

- `ARCHITECTURE.md` — crate layering, shared Rust contracts, async/streaming model, error strategy, feature flags, testing infrastructure.
- `PARITY.md` — Python → Rust feature matrix, intentional deviations, and explicitly deferred capabilities.

## Known Gaps

- Neither document states which cargo feature gates the `openjiuwen-harness-providers` provider registry, nor how a consumer enables the `native` provider; `ARCHITECTURE.md` only lists `openjiuwen-harness` feature `native-provider` (default) in §5.
- `create_harness` and `resolve_provider` are not mentioned anywhere in this directory.
- `harness_providers` appears only as the Python module name in the `PARITY.md` matrix; there is no API-surface description of the registry types.

## Constraints

- Documentation describes the current state; history lives in git.
- Keep the parity matrix current whenever a crate's scope changes.
- Do not duplicate per-crate API details here; link to the crate `CONTEXT.md`.
