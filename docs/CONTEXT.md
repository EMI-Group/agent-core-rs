# Docs

## Intent

Architecture and parity documentation for the Rust rewrite.
The Context Tree (`CONTEXT.md` per crate) is the primary source of intent and API surface; these documents hold the cross-cutting contracts that every crate shares.

## Contents

- `ARCHITECTURE.md` — crate layering, shared Rust contracts, async/streaming model, error strategy, feature flags, testing infrastructure.
- `PARITY.md` — Python → Rust feature matrix, intentional deviations, and explicitly deferred capabilities.

## Known Gaps

- No known documentation gaps; per-package status and feature gates are stated in `PARITY.md` and `ARCHITECTURE.md` §5.

## Constraints

- Documentation describes the current state; history lives in git.
- Keep the parity matrix current whenever a crate's scope changes.
- Do not duplicate per-crate API details here; link to the crate `CONTEXT.md`.
