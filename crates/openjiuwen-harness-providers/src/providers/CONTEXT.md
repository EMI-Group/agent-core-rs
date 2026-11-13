# Built-in harness providers

## Intent

The compile-time provider table: one module per registry name in `PROVIDER_NAMES`, resolved by `builtin_provider(name) -> Option<Result<Arc<dyn ProviderSpec>>>`.
Every declared provider is implemented: `native` delegates to `openjiuwen_harness::{NativeHarness, NativeHarnessProvider}`, `native_v2` hosts an `openjiuwen-agent-teams` team through `vendor_cli::VendorHarness`, and `claudecode`/`codex`/`dsh` are real subprocess adapters that drive the vendor CLIs over stdio.
Feature gating is expressed in `mod.rs`, never inside a provider module: without `native` the native entry is compiled out, without `vendor` the three vendor providers are unregistered, and without `native-v2` the team-hosted provider is absent. Each refused name reports `Error::Unsupported` naming the feature to rebuild with.

## API Surface

- `mod.rs` — `builtin_provider(name)`, the `#[cfg]`-gated module declarations, `native_provider()`, `native_v2_provider()`, `vendor_provider(name)`, and the compiled-out messages (`"harness provider 'native_v2' is disabled; rebuild with the `native-v2` feature …"`, `"vendor harness provider '{name}' is deferred; rebuild with the `vendor` feature …"`).
- `native` (`native`) — `PROVIDER_NAME`, `CARD_NAME`, `provider()`, `card()`, and re-exports of `NativeHarness*`; registered through `model_configuring_provider`.
- `native_v2` (`native-v2`) — `PROVIDER_NAME`, `CARD_NAME`, `IMPLEMENTATION_VERSION` (`0.2.0`), `DEFAULT_EVENT_BUFFER_CAPACITY`, `NativeV2Config`, `NativeV2HarnessProvider`, `NativeV2Driver`; registered through `plain_provider` (no model hook).
- `claudecode` / `codex` / `dsh` (`vendor`) — each a directory with `mod.rs` (config, enums, `card()`, `provider()`, `model_config`) and `harness.rs`/`frames.rs`/`wire.rs`; registered through `model_configuring_provider`.
- `vendor.rs` (`vendor`) — crate-internal shared helpers: `config_error_to_harness`, `model_fragment`, `nested_model_config`, `vendor_card`. Nothing here spawns a process or speaks a vendor protocol.

## Constraints

- A provider module never implements harness behavior: `native` constructs the harness crate's provider, `native_v2` constructs a driver over the team runtime, and the vendor modules decode configuration, build cards and hand a driver to `vendor_cli::VendorHarness`.
- A vendor `create` decodes the config, resolves the configured executable (and, for DSH, `DSH_HOME` and the argv) and returns an **unstarted** harness; nothing spawns before `start`, and a missing binary is a typed `HarnessError::Protocol` naming it.
- Registry names are Python strings (`native`, `native_v2`, `claudecode`, `codex`, `dsh`); card names are metadata (`openjiuwen-native`, `native_v2`, `claude-code`, `codex`, `deepseek-harness`).
- Cards declare only capabilities the transport really provides: DSH omits abort/steer/pause/checkpoint, Codex omits model control and pausing, `native_v2` omits checkpoint/MCP/model capabilities.
- Config types: serde `snake_case`, Python defaults, `from_json(&JsonObject)` rejects unknown keys, `api_key` is `SecretString` and `Debug` redacts the secret-bearing maps.
- Files ≤1000 lines; shared decoding helpers come from `../decode.rs`, shared vendor data helpers from `./vendor.rs`, shared transport from `../vendor_cli/` — never duplicated per provider.

## Routing Table

- `./mod.rs` → built-in resolution table, feature gating, and the compiled-out messages; unit tests for every feature combination
- `./native.rs` → native provider wiring over `openjiuwen-harness`
- `./vendor.rs` → shared vendor data helpers (config-error translation, model fragments, card construction)
- `./claudecode/` → Claude Code card, config, model fragment and the stream-json subprocess transport (`./claudecode/CONTEXT.md`)
- `./codex/` → Codex card, config, model fragment and the app-server JSON-RPC transport (`./codex/CONTEXT.md`)
- `./dsh/` → DSH card, config, flat model fragment and the JSON-RPC runtime transport (`./dsh/CONTEXT.md`)
- `./native_v2/` → team-hosted provider, create schema and `NativeV2Driver` (`./native_v2/CONTEXT.md`)
