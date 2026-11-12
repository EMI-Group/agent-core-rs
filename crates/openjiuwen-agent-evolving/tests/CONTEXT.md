# tests

## Intent

Offline integration suites for the portable agent-evolving core.
They exercise the public API end to end and never touch a network, the environment, or wall-clock time.

## Constraints

- Deterministic and offline; fakes implement `CaseRunner`/`TrainableTarget` locally.
- Use `tempfile` for checkpoint directories and `openjiuwen-test-support` fixtures where a model/agent is useful.
- Service tests use `tower::ServiceExt::oneshot` on `build_router`, not a bound port.

## Routing Table

- `./dataset.rs` → JSON/JSONL loading, validation, shuffle/split, serde
- `./evaluator.rs` → exact match, aggregation, batch validation, reports, fake runners
- `./checkpoint.rs` → file/memory store, save/load/resume, manager save policy
- `./trainer.rs` → baseline/epochs, early stop, resume, cancellation, budgets, callbacks
- `./service.rs` → `/health`, `/v1/evaluate`, 501 RL endpoint envelope
- `./rails.rs` → event translation and harness adapter
