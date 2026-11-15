# external_provider — external account providers (auth + model catalog)

## Intent

Port `openjiuwen/extensions/external_provider/` (v0.1.20): a registry of providers whose credentials live outside the SDK, plus the built-in `OpenAIAccount` implementation.
Two provider-neutral seams — [`ExternalAuthProvider`] (status / logout / access token) and [`ProviderModelCatalog`] (model discovery) — are registered by provider name and created through a typed configuration.
The device-authorization grant, token refresh, the auth-store file, and the cross-process lock around its writes are **not** reimplemented here: the module composes [`crate::llm::oauth`] (`OpenAiOAuthClient`, `FileTokenStore`, `OAuthToken`, `REFRESH_SKEW`, and now `login`/`status`/`logout` over `OAuthCredentials`).
The model catalog talks HTTP only through `openjiuwen_common::http::HttpClient`.

## API Surface

`base`:
- `AuthStatus { authenticated, auth_path: Option<PathBuf>, has_refresh_token, expires_at: Option<DateTime<Utc>>, needs_refresh, error: Option<String> }` — serde, `expires_at` on the wire as epoch seconds (an RFC 3339 string is accepted on input); `AuthStatus::unauthenticated(auth_path, code)`.
- `ExternalAuthProvider` (async, object-safe): `status() -> Result<AuthStatus>`, `logout() -> Result<bool>`, `resolve_access_token(force_refresh: bool) -> Result<SecretString>`.
- `ProviderModelCatalog` (async, object-safe): `list_model_ids(&ModelCatalogRequest) -> Result<Vec<String>>`.
- `ModelCatalogRequest { auth_manager: Option<Arc<dyn ExternalAuthProvider>>, access_token: Option<SecretString>, force_refresh: bool }` with `with_auth_manager`, `with_access_token`, `with_secret`, `with_force_refresh`; redacting `Debug`; deliberately not serde (trait object + secret).
- `ExternalProviderConfig { auth_path, base_url, cache_path, timeout (seconds, f64), refresh_skew (seconds, u64) }` — `#[serde(default)]`, tolerant of unknown keys, builders `with_auth_path`/`with_base_url`/`with_cache_path`/`with_timeout`/`with_refresh_skew`, `refresh_skew_or(fallback)`.
  This is the Rust replacement for Python's factory `**kwargs`: `timeout_seconds` → `timeout`, `refresh_skew_seconds` → `refresh_skew`.

`registry`:
- `ExternalProviderRegistry` (`new`, `with_builtins`, `global`): `register_auth_provider(provider, factory, replace) -> Result<Option<AuthProviderFactory>>`, `register_model_catalog(provider, factory, replace)`, `get_auth_provider_factory`, `get_model_catalog_factory`, `create_auth_provider(provider, &config)`, `create_model_catalog(provider, &config)`, `list_auth_providers`, `list_model_catalogs`.
- `AuthProviderFactory = Arc<dyn Fn(&ExternalProviderConfig) -> Result<Arc<dyn ExternalAuthProvider>> + Send + Sync>`; `ModelCatalogFactory` is the catalog twin.
- Free functions over `global()`: `register_external_auth_provider`, `get_auth_provider_factory`, `create_auth_provider`, `list_auth_providers`, `register_external_model_catalog`, `get_model_catalog_factory`, `create_model_catalog`, `list_model_catalogs`.

`openai_account`:
- Constants: `PROVIDER_NAME = "OpenAIAccount"`, `DEFAULT_BASE_URL`, `REFRESH_SKEW_SECONDS = 120`, `MODELS_CLIENT_VERSION = "1.0.0"`, `MODELS_CACHE_FILENAME`, `MODELS_CACHE_FILE_ENV = "OPENJIUWEN_OPENAI_ACCOUNT_MODELS_CACHE_FILE"`, `BASE_URL_ENV = "OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL"`, `DEFAULT_MODELS: [&str; 6]`.
- `OpenAiAccountAuthManager` (`new`, `with_config`, `with_store(Arc<dyn TokenStore>)`, `with_oauth_client`, `with_base_url`, `with_refresh_skew`; accessors `auth_path`, `base_url`, `refresh_skew`): `status`, `logout`, `load_tokens`, `save_tokens`, `refresh_tokens(force)`, `resolve_access_token(force_refresh)`; implements `ExternalAuthProvider`.
- Status codes: `MISSING_AUTH_CODE = "openai_account_auth_missing"`, `INVALID_STORE_CODE = "openai_account_auth_invalid_store"`, `STORE_READ_FAILED_CODE = "openai_account_auth_store_read_failed"`.
- `OpenAiAccountModelCatalog` (`new`, `with_config`, `with_http_client`, `with_base_url`, `with_cache_path`, `with_client_version`, `with_timeout`, `with_fixed_time`; accessors `base_url`, `cache_path`, `client_version`, `timeout`): `list_model_ids`, `fetch_models(&SecretString) -> Result<(Value, Vec<String>)>`, `read_cached() -> Option<Vec<String>>`, `write_cache(&Value, &[String])`; implements `ProviderModelCatalog`.
- Free functions: `parse_model_ids(&Value) -> Vec<String>`, `forward_compat_models(Vec<String>) -> Vec<String>`, `default_cache_path() -> Result<PathBuf>`, `DEFAULT_TIMEOUT_SECONDS = 10.0`.
- Factories wired into the builtins: `auth_provider_factory(&ExternalProviderConfig)`, `model_catalog_factory(&ExternalProviderConfig)`.

## Constraints

- Every HTTP call goes through `HttpClient`; no `reqwest` use here, no new dependency, no `unsafe`, no `println!`/`eprintln!` (only `tracing` with target `openjiuwen_foundation::llm::external_provider`).
- Errors use `openjiuwen_common::error::{Error, Result}`: `Error::Validation` for registry/config problems, `Error::Config` for a missing credential, `Error::Llm` for discovery wire/parse failures, `Error::External` for cache-store I/O.
- Tokens stay in `SecretString`; `Debug` for `ModelCatalogRequest` and the manager never prints a credential.
- No panics on user input or corrupt files: malformed JSON, non-object documents, non-finite timeouts, and missing home directories are typed errors or degraded statuses.
- Files stay around ≤1000 lines: `auth.rs` (manager), `catalog.rs` (discovery, parsing, cache), `base.rs` (traits/config), `registry.rs`, two sibling test suites.
- `src/llm/mod.rs` gains only `pub mod external_provider;`, five re-exports (`AuthStatus`, `ExternalAuthProvider`, `ExternalProviderConfig`, `ExternalProviderRegistry`, `ModelCatalogRequest`, `ProviderModelCatalog`), and one `//!` bullet; every deeper item is reached through `llm::external_provider::…`.

## Design Decisions

- **Provider keys fold separators.** Python keys are `strip().lower()`; this module also drops `-`, `_`, and whitespace (like `llm::normalize_provider_name`), so `OpenAIAccount`, `openai_account`, `openai-account`, and `openaiaccount` all hit one entry. `list_*` reports the spelling used at registration.
- **Registry errors are `Error::Validation`.** Python's `ExternalProviderRegistryError(ValueError)` has no Rust twin; the message keeps Python's text (`Unknown external auth provider: 'X'. Available providers: …`) so a ported test matches.
- **Factories are fallible and take a typed config.** Python constructs classes with `**kwargs`; a Rust path/URL/HTTP-stack resolution can fail, so the factory returns `Result` and the registry propagates it.
- **The process-global registry is sanctioned.** `ExternalProviderRegistry::global()` (`LazyLock` + `RwLock`) is the second documented mutable global beside `ModelProviderRegistry::global()`; local instances (`new`, `with_builtins`) keep tests and isolated components off it.
- **`status()` never fails for store trouble.** Missing / unreadable / corrupt stores become `authenticated: false` with a Python error code (mapped by `Error::kind()`: `External` → read-failed, otherwise invalid-store), matching Python's non-raising `status()`. `logout()` and `resolve_access_token()` do propagate store errors.
- **Refresh decisions are the reference implementation's**: refresh when the token expires within `refresh_skew`, or always with `force`; `OAuthToken::effective_expires_at` already falls back to the JWT `exp` claim, and a token with no resolvable expiry counts as expired (see `llm::oauth`), so it is refreshed on first use instead of being treated as valid forever.
- **Device login stays in `llm::oauth`.** Python's `OpenAIAccountAuthManager.start_device_login` / `poll_device_login` / `login_with_device_code` are not duplicated; `llm::oauth::login` runs the whole grant (and `status`/`logout` cover the other two commands) over `OAuthCredentials`, while a caller that needs the primitives uses `OpenAiOAuthClient` directly.
- **Catalog precedence is live → cache → defaults, and never fails.** A discovery or cache-write failure is logged at `debug` and the next fallback answers (Python swallows `OpenAIAccountModelListError` / `OSError`). The individual `fetch_models`/`write_cache` calls still return typed errors for direct callers.
- **Parsing rules are ported exactly**: entries from `models` then `data` (list or object keyed by ID with a `slug` fallback), `visibility` `hide`/`hidden` (trimmed, case-insensitive) skipped, ID from `slug`/`id`/`name`/`model`, sort by `(priority, id)` with `10_000` for absent/bool/non-numeric priority, dedupe, then append the forward-compat templates (`gpt-5.5`, `gpt-5.4-mini`, `gpt-5.4`, `gpt-5.3-codex-spark`).
- **The cache document is Python-shaped**: `{provider, base_url, client_version, fetched_at, model_ids, payload}` (keys sorted by serde_json's default map), `model_ids` already forward-compat-extended, written through `{name}.tmp` + rename, `fetched_at` pinned by `with_fixed_time` in tests.
- **`read_cached` returns `Option<Vec<String>>`**: `None` for absent/unreadable/non-object files, `Some` otherwise (possibly empty), so the caller owns the "empty means fall through" rule; Python returns `[]` for both.
- **The timeout is validated, not clamped from invalid input**: non-finite/non-positive → `Error::Validation`; a valid value below 5 s is raised to 5 s (Python's `max(5.0, …)`). Python would silently clamp a negative value to 5 s.
- **Proxy/TLS overrides are injected, not config fields**: `with_http_client(HttpClient)` carries the `HttpClientConfig` (proxy, TLS verification, retry policy), so no capability is lost and nothing returns `Error::Unsupported`.

## Known Gaps and Deviations

- The stored `auth_mode` marker is `openai_account_oauth` (what `llm::oauth` writes), not Python's `openai_account`; the `tokens` payload itself round-trips with Python.
- Python's `now`/`sleep`/`transport` injection points are represented by `with_fixed_time` and `with_http_client`; `max_retries` and `verify`/`proxy` live on `HttpClientConfig`.
- The catalog supports only the synchronous-equivalent request path: no `asyncio.to_thread` wrapper is needed because the Rust client is async.
- No `Error::Unsupported` is emitted by this subtree: every deferred Python capability (callable injection, unknown kwargs) is either replaced by an explicit seam or outside the public surface. The auth-store lock itself lives in `llm::oauth` and is unix-only there.

## Test Strategy

- `registry_tests.rs`: builtin availability and both factory kinds; alias folding (`OpenAIAccount`/`openai_account`/`openai-account`/`openaiaccount`); fresh instance per creation; unknown-provider message listing the available names; empty-name rejection; duplicate registration with and without `replace` (previous factory returned); local-vs-global isolation; free-function listings.
- `openai_account/tests.rs`: pure parsing (hidden/`supported_in_api`/forward-compat, keyed `models`/`data`, ID precedence, dedupe, priority shapes); wiremock live fetch asserting path, `client_version`, `Authorization: Bearer`, `Accept`, returned order, exact cache JSON and the absent `tmp` file; non-200 / invalid JSON / array payload / no visible models / empty token (with zero requests); cache fallback and defaults fallback; `write_cache`/`read_cached` round trip incl. corrupt and absent files; manager `status`/`logout`/forced and expiring refresh with persistence, missing and corrupt store codes, `providers.other` preservation, and catalog-through-manager bearer resolution; `default_cache_path` env precedence under a serialized env guard.
- Everything is offline: wiremock on loopback, `tempfile` directories, `InMemoryTokenStore`, fixed clock; no credentials and no network.
- Gates: `cargo test -p openjiuwen-foundation --offline`, the same with `--all-features`, `cargo clippy -p openjiuwen-foundation --all-targets --all-features --offline -- -D warnings`, and `cargo fmt -p openjiuwen-foundation -- --check`.

## Routing Table

- `./base.rs` → `AuthStatus`, `ExternalAuthProvider`, `ProviderModelCatalog`, `ModelCatalogRequest`, `ExternalProviderConfig`
- `./registry.rs` → `ExternalProviderRegistry`, factory type aliases, the process-global registry, free functions, key folding and error shapes
- `./openai_account/mod.rs` → provider constants, factory wiring, re-exports
- `./openai_account/auth.rs` → `OpenAiAccountAuthManager` (status/logout/load/save/refresh/resolve) and the status codes
- `./openai_account/catalog.rs` → `OpenAiAccountModelCatalog`, live fetch, `parse_model_ids`, forward-compat, cache read/write, `default_cache_path`
- `./registry_tests.rs` → registry behavior suite (module-level `#[cfg(test)]`)
- `./openai_account/tests.rs` → credential, discovery, parsing, and cache suites (module-level `#[cfg(test)]`)
