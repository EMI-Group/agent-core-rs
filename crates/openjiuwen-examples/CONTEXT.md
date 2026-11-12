# openjiuwen-examples

## Intent

Runnable examples mirroring the Python `examples/` tree.

## API Surface

The crate is a scaffold stub: `openjiuwen_examples::crate_name() -> &'static str` is its only item.
`examples/` contains only `.gitkeep` — no runnable example targets exist yet.

## Constraints

- Depends only on the `openjiuwen` umbrella (`openjiuwen.workspace = true`, default features → `native`), plus `tokio`, `serde_json`, `anyhow`, `dotenvy`, `futures`.
- No direct dependency on `openjiuwen-harness-providers`; harness-provider APIs are reachable only transitively via `openjiuwen`.
- `default = []` — the crate declares no features of its own.
- Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging (examples may use `anyhow`).

## Routing Table

- `./src/` → `lib.rs` only (crate-name stub); no child routing.
- `./examples/` → empty except `.gitkeep`; examples added here become `cargo` example targets.
