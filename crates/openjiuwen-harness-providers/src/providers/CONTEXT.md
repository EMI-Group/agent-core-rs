# Built-in harness providers

## Intent

The compile-time provider table: one module per registry name in `PROVIDER_NAMES`.
`native` delegates to `openjiuwen_harness::{NativeHarness, NativeHarnessProvider}`; the three vendor modules are deferred stubs that keep the registration surface, cards, and pure-data configs while refusing to construct a harness.

## API Surface

- `mod.rs` — `builtin_provider(name) -> Option<Result<Arc<dyn ProviderSpec>>>` (the resolution table), re-exports of `native` and the `vendor`-gated modules, `is_builtin(name)`.
- `native` — `PROVIDER_NAME`, `CARD_NAME`, `provider()`, `card()`, and re-exports of `NativeHarness*`.
- `claudecode` / `codex` / `dsh` (feature `vendor`) — `PROVIDER_NAME`, `CARD_NAME`, `card()`, `provider()`, `model_config(...)`, and the provider config data types.
- `vendor.rs` (feature `vendor`) — shared helpers: the per-provider `Unsupported` message, JSON mapping helpers for the `{"model": {…}}` / flat fragments, and vendor card construction.

## Constraints

- A provider module never implements harness behavior; `native` only constructs the harness crate's provider and `vendor` modules only validate data.
- Registry names are Python strings (`native`, `claudecode`, `codex`, `dsh`); card names are metadata (`openjiuwen-native`, `claude-code`, `codex`, `deepseek-harness`).
- Vendor `create` returns `HarnessError::UnsupportedCapability` carrying the Python "SDK required" text; nothing spawns an external process.
- Vendor cards describe the *eventual* implementation (capability sets from the Python cards) and live behind `vendor`, so a default build never advertises them.
- Config types: serde `snake_case`, defaults matching Python, `from_json(&JsonObject)` rejects unknown keys, `api_key` is `SecretString` and redacted in `Debug`.
- Files ≤1000 lines; shared decoding helpers come from `../decode.rs`, never duplicated per provider.

## Routing Table

- `./mod.rs` → built-in resolution table and re-exports
- `./native.rs` → native provider wiring over `openjiuwen-harness`
- `./vendor.rs` → shared vendor stub helpers (feature `vendor`)
- `./claudecode.rs` → Claude Code card, config types, model fragment, unsupported `create`
- `./codex.rs` → Codex card, config types, model fragment, unsupported `create`
- `./dsh.rs` → DSH card, config types, flat model fragment, unsupported `create`
