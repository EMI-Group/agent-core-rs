# oauth — OpenAI account OAuth transport and credential store

## Intent

Port the `OpenAIAccount` provider of the reference SDK: the OAuth 2.0 device authorization grant against `https://auth.openai.com`, the token exchange and refresh that rotate the refresh token, and the private auth-store file that keeps the credentials between runs.
The Python source lives under `openjiuwen/core/foundation/llm/` (the account/OAuth client); here the transport is implemented on `openjiuwen_common::http::HttpClient` instead of the vendor `openai` SDK.
The module also owns the three whole-flow entry points (`login`, `status`, `logout`) and the cross-process lock that serializes the auth store's read-modify-write cycles.
Only the token endpoints live in this module: sending a model request with the resulting bearer token — the Responses-API transport — is out of scope.

## API Surface

- `OAuthToken` — `access_token`, `refresh_token`, `id_token` (all `SecretString`), `expires_at: Option<DateTime<Utc>>`, `token_type` (`Bearer` by default), `scope`; `new` plus `with_refresh_token`, `with_id_token`, `with_expires_at`, `with_token_type`, `with_scope`, `effective_expires_at()` and `is_expired(skew)`; redacting `Debug`, canonical `Serialize`, tolerant `Deserialize` through a private `TokenPayload`.
- `TokenStore` — async trait (`load`, `save`, `clear`) and the persistence seam above the client.
- `InMemoryTokenStore` — process-memory backend with `new` and `Default`; the testable default.
- `FileTokenStore` — auth-store JSON backend: `new(path)`, `with_base_url`, `path`, `base_url`, `default_path`, `lock_path()`, `lock()` (bounded wait), `lock_with_timeout(Duration)`, `try_lock()` (never waits); the async `TokenStore` impl acquires the lock for its own save/clear cycle.
- `TokenFileLock` (`lock`) — exclusive advisory lock held on `<auth_file>.lock`: `auth_path`, `lock_path`, `base_url`, `load`, `save`, `clear` (all under the held lock), plus the synchronous acquisitions `acquire`, `acquire_with_timeout`, `try_acquire`; `lock_path(Path)` and `LOCK_TIMEOUT` (10 s) are exported too.
- `OAuthCredentials` (`credentials`) — the single argument of `login`/`status`/`logout`: `new()` (default auth path), `at_path(path)`, `from_file(FileTokenStore)`, `from_store(Arc<dyn TokenStore>)`, builders `with_client`, `with_skew`, accessors `store()`, `auth_path()`, `client()`, `skew()`; redacting `Debug`.
- `login(&OAuthCredentials) -> Result<LoginInfo>` — the full device grant: request the code, publish the verification URL and user code through `tracing` (`openjiuwen_foundation::llm::oauth`), poll, exchange, persist under the lock, and return the same pair plus a token summary.
- `status(&OAuthCredentials) -> Result<TokenStatus>` — reads the store under the lock; `logout(&OAuthCredentials) -> Result<bool>` — removes the credential and reports whether one was there (idempotent).
- `LoginInfo` — `verification_uri`, `user_code`, `expires_in`, `interval`, `token: TokenSummary`; serde, no secret fields.
- `TokenSummary` — `expires_at`, `token_type`, `scope`, `has_refresh_token`, `has_id_token`; `TokenSummary::of(&OAuthToken)`.
- `TokenStatus` — `auth_path: Option<PathBuf>`, `present`, `expires_at`, `expired`, `has_refresh_token`, `auth_mode: Option<String>`, plus `is_valid()`; serde (`expires_at` as epoch seconds).
- `DeviceCode` — pending device authorization: `user_code`, `device_auth_id`, `interval` (default 5 s, never below 3 s when parsed), `expires_in` (`0` means unspecified), `new`, `with_interval` (no floor, the test seam), `with_expires_in`.
- `OpenAiOAuthClient` — the HTTP side: `new`, `with_http_client`, `with_base_url`, `with_client_id`, `http_client`, `base_url`, `client_id`, `request_device_code`, `poll_device_token`, `poll_device_token_with_deadline`, `exchange_code`, `refresh`, `access_token`.
- Exported constants from `client`: `DEFAULT_ISSUER` (`https://auth.openai.com`), `DEFAULT_CLIENT_ID`, `DEVICE_CALLBACK_PATH` (`/deviceauth/callback`), `DEVICE_VERIFICATION_PATH` (`/codex/device`).
- Exported constants from `store`: `PROVIDER_KEY` (`OpenAIAccount`), `AUTH_MODE` (`openai_account_oauth`), `DEFAULT_BASE_URL` (`https://chatgpt.com/backend-api/codex`), `AUTH_FILE_ENV` (`OPENJIUWEN_AUTH_FILE`), `REFRESH_SKEW` (120 s).

## Wire Contract

- Device code: `POST {issuer}/api/accounts/deviceauth/usercode` with `{"client_id"}` → `DeviceCode`; a non-2xx answer becomes an error with a redacted, truncated body.
- The verification URL is derived, not returned: `{issuer}/codex/device` (`DEVICE_VERIFICATION_PATH`), the same page Python's `OPENAI_ACCOUNT_DEVICE_AUTH_URL` names.
- Poll: `POST {issuer}/api/accounts/deviceauth/token` with `{"device_auth_id", "user_code"}`; `403`/`404` mean "still pending", `200` either carries a token or `authorization_code` + `code_verifier`.
- A `200` carrying an authorization pair is exchanged through `exchange_code` with `redirect_uri = {issuer}/deviceauth/callback`; a missing code or verifier is an error.
- Exchange and refresh share `POST {issuer}/oauth/token`, form-encoded: `grant_type=authorization_code` with `code`, `redirect_uri`, `client_id`, `code_verifier`, or `grant_type=refresh_token` with `refresh_token`, `client_id`.
- Polling starts after `max(interval, 1 s)`, doubles while pending up to 10 s, and the whole flow is bounded by `min(device_code.expires_in, 15 min)`; a parsed `interval` is never below 3 s (Python parity: `max(3, interval)`).
- The expiry of a token response is resolved with the reference implementation's precedence: `expires_at` (epoch seconds or RFC 3339) → `now + max(expires_in, 0)` (numeric strings accepted) → the previous token's `expires_at` → the `exp` claim of a JWT `access_token` (base64url payload, signature never verified).
  A token whose expiry resolves from none of those is **treated as already expired** (see below), never as "valid forever".
  Serialization always writes `expires_at` in epoch seconds and never re-emits `expires_in`.
- A refresh response may rotate the refresh token; a field it omits (`refresh_token`, `id_token`, `scope`, `token_type`, expiry) is inherited from the token being replaced.
- `access_token` refreshes when `force_refresh` is set or the token `is_expired(REFRESH_SKEW)` (120 s skew, `effective_expires_at`), and saves the (possibly rotated) result before returning it.
- A rejected request is classified through the error message: `429` is rate limiting whose message says the credentials are still valid and no re-login is required; `401`/`403` or a vendor code `invalid_grant`/`invalid_token`/`invalid_request`/`refresh_token_reused` says the credentials are unusable and a re-login is required; anything else is a plain provider failure.
- Store envelope: `{"providers": {"OpenAIAccount": {"tokens": {...}, "auth_mode": "openai_account_oauth", "base_url": "..."}}}`; only those three keys of the entry are written.
- Path resolution: `$OPENJIUWEN_AUTH_FILE`, then `$OPENJIUWEN_HOME/auth.json`, then `~/.openjiuwen/auth.json` (`%USERPROFILE%` stands in for `$HOME` on Windows), with a leading `~` expanded.
- Lock file: `<auth_file>.lock` in the same directory, created on demand, never read or written, and intentionally never removed (removing it would let two processes lock two inodes).

## Expiry Rule

`OAuthToken::is_expired(skew)` has three distinct branches:

- a resolvable expiry already inside `skew` (or past) → stale;
- **no resolvable expiry** (no `expires_at`, no `expires_in`, no inherited previous expiry, and an access token that is not a JWT with a positive `exp`) → stale, so `access_token`/`resolve_access_token` attempt the refresh-token flow instead of handing out a credential of unknown age forever; without a stored refresh token the attempt surfaces the re-login `Error::Config`;
- a resolvable expiry beyond `skew` → valid, no network round trip.

## Constraints

- Every request goes through `openjiuwen_common::http::HttpClient`; never `reqwest` directly, never a hand-rolled retry loop.
- No token, authorization code, code verifier, or refresh token is logged or formatted into a message: credentials are `SecretString`, `Debug` redacts, the summaries report presence flags only, and error bodies pass through `security::redact_str` and are truncated to 500 characters.
  The device `user_code` and the verification URL are public by design and are the only device-flow values written to the log.
- Error variants: `Config` for "no stored token", "no refresh token", and an unresolvable auth path; `Llm` for non-2xx responses, bodies that are not a token, invalid JSON, and an expired device code; `Validation` for a malformed auth-store envelope; `External` for I/O, blocking-task failures, and lock contention (`try_lock` wraps `WouldBlock`); `Timeout` when a bounded lock wait runs out; `Unsupported` for lock acquisition on a platform without unix advisory locks.
  The rate-limited / re-login-required / plain-failure distinction lives in the message (no new error kind), and a vendor code is echoed only when it is one of the four known re-login codes.
- Blocking file IO — including the `flock` acquisition and its retry loop — runs on `tokio::task::spawn_blocking`; no blocking call on a runtime thread.
- Logging is `tracing` only (pending poll status at debug, the device pair and the completed login at info); no `println!`/`eprintln!`.
- Files stay under the ~1000-line rule (`store.rs` ~960, `client.rs` ~510, `credentials.rs` ~440, `lock.rs` ~385, `tests.rs` ~820, `refresh_tests.rs` ~320, `token_tests.rs` ~300, `credential_tests.rs` ~430, `lock_tests.rs` ~230, `support.rs` 63); no dependencies beyond the crate's existing `chrono`, `base64`, `serde`/`serde_json`, `url`, `nix` (unix `flock`), and dev-only `wiremock`/`tempfile`.

## Design Decisions

- `FileTokenStore` wraps each synchronous helper in `spawn_blocking`, cloning the path, base URL, and token first, so the store stays `Send + Sync` behind the async `TokenStore` trait.
- Writes are atomic: render pretty JSON, write `<name>.<pid>.tmp` in the same directory, `sync_all`, chmod to `0o600` on unix, rename over the target, then restrict the target again so an existing loose file is tightened.
- Every write (the store's `save`/`clear`, and therefore `login`/`logout`) runs under the cross-process lock: `flock(2)` with `LOCK_EX` on `<auth_file>.lock`, retried every 20 ms until `LOCK_TIMEOUT` (10 s) so the wait is bounded instead of an unbounded blocking call.
  Reads stay lock-free because the rename publishes a complete document; `status` still takes the lock so the token and the `auth_mode` marker come from the same document version.
  The guard (`TokenFileLock`) carries the token operations, so a caller holding the lock cannot accidentally re-acquire it (which would deadlock: `flock` locks belong to the open file description).
- The save is a read-merge-write: other providers' entries and unknown keys inside the `OpenAIAccount` entry survive; only `tokens`, `auth_mode`, and `base_url` are set.
- `clear` removes just the `providers.OpenAIAccount` entry, reports whether it was there, and leaves the file untouched when the entry is absent.
- Every omitted field of a refresh response — `refresh_token`, `id_token`, `scope`, `token_type`, and the expiry — is inherited from the previous token by one resolver (`TokenPayload::into_token`), so rotation and non-rotation take the same code path.
- The client holds no token cache: the store is the single source of truth, so several clients or processes cannot disagree about the current token.
- An unresolvable expiry means "expired now" rather than "never expires": a credential of unknown age is refreshed on first use, which is the only way to notice an issuer-side revocation. The refresh response's own lifetime then re-establishes a known expiry.
- The JWT `exp` fallback is read without verifying the signature: it only decides *when* to refresh, and a forged claim can at most cost one extra refresh whose result must still come from the issuer.
- `default_path` returns `Error::Config` instead of a relative fallback, so a credential file can never land in the working directory silently.
- `429` is reported as rate limiting whose message states that the stored credentials are still valid and no re-login is required, which keeps a caller from forcing a re-login; the vendor body is redacted and an unrecognised vendor code is never echoed, so an error message cannot smuggle a secret.
- `DeviceCode` is tolerant by design: a missing `interval` is 5 s and a smaller advertised one is raised to 3 s, a missing `expires_in` is 0, and both accept numbers or numeric strings; the `with_interval` builder applies no floor, so tests can drive the poll quickly.
- The issuer and store path are injectable (`with_base_url`, `with_http_client`, `FileTokenStore::new`, `OAuthCredentials::at_path`) so tests point at loopback and temporary directories.
- `login`/`status`/`logout` take one `OAuthCredentials` instead of loose arguments so a caller cannot pair a client with the wrong store; the file target keeps the lock while a custom `TokenStore` target skips it, and the store's own save/clear still lock for the file target.

## Test Strategy

- The suite is split by responsibility and declared from `mod.rs`: `tests.rs` (device flow, exchange, refresh rotation, `access_token`, file store, path resolution), `refresh_tests.rs` (refresh-response precedence and failure classification), `token_tests.rs` (expiry precedence and serialization), `credential_tests.rs` (login/status/logout), `lock_tests.rs` (cross-process lock), and `support.rs` (the shared mock-server client, expiry arithmetic, JWT builder, and secret reader).
- `tests.rs` covers the transport against `wiremock` on loopback: device-code request body, the three-second interval floor, redacted rejection, polling through pending to exchange, a token returned by the poll itself, expiry, the exact exchange and refresh form (three parameters, no `scope`/`redirect_uri`), refresh rotation and merge, and `access_token` refresh/force/no-network/no-token/no-refresh-token/past-JWT-claim, plus the refresh of a token with no resolvable expiry (which now stores the refreshed expiry).
- `credentials.rs`' test tree (`credential_tests.rs`) drives the whole login against a mocked device-code/poll/exchange sequence and asserts the returned `LoginInfo` (verification URL derived from the issuer, user code, lifetime, interval floor, token summary) and the exact stored document (`access_token`, `refresh_token`, `id_token`, `auth_mode`, `base_url`, the `.lock` file), plus that no rendered value or summary leaks a credential; a rejected device code creates no file. `status` covers absent/valid/expired/no-resolvable-expiry, an explicit skew, a custom store without a path, and a corrupt store; `logout` covers removal, idempotence (twice, and without a store), a custom store, and provider preservation.
- `lock_tests.rs` proves the two-handle semantics on unix: `try_lock` reports `Error::External` + `WouldBlock` while the other handle holds the lock and succeeds after the release, the bounded wait reports `Error::Timeout`, the guard reads/writes/clears under the lock, and concurrent read-modify-write rounds between the account store and a second "process" (same lock protocol) lose neither entry.
- File-store tests use `tempfile` directories: round trip, other-provider preservation, `clear`, `0o600` permissions with no leftover temporary file, and an unreadable store.
- `default_path` precedence is tested behind a process-environment mutex with restore, and never touches `~/.openjiuwen`.
- `token_tests.rs` covers `expires_in` → `expires_at` (including numeric strings and a non-positive lifetime), epoch and RFC 3339 expiries, the JWT `exp` fallback (unpadded and padded payloads, missing claim, malformed token, opaque token), the unresolvable-expiry branch (stale), canonical serialization, round trip, `Debug` redaction, and `DeviceCode` tolerance.
- Everything is offline: loopback sockets only, no credential, no wall-clock ordering, short poll deadlines and timeouts.

## Known Gaps

- No Responses-API (or any other model) transport that consumes the bearer token; the module stops at the token endpoints.
- No `/models` catalog cache and no account metadata fetch.
- No shipped keyring or secret-manager backend — `TokenStore` is the seam for one.
- No `OPENJIUWEN_*` override for the issuer or client id; those are constructor arguments only.
- The cross-process lock is unix-only: on other platforms `TokenFileLock`/`FileTokenStore::lock` return `Error::Unsupported` naming the missing support, and file writes stay atomic but unserialized.

## Routing Table

- `./mod.rs` → module documentation, `client`/`credentials`/`lock`/`store` declarations, and the crate-facing re-export list
- `./store.rs` → `OAuthToken` (expiry precedence incl. the JWT `exp` fallback and the unresolvable-expiry branch), `parse_token_payload`, `DeviceCode`, the `TokenStore` trait, the memory and file backends, envelope merge, path resolution, atomic owner-only writes
- `./client.rs` → `OpenAiOAuthClient`: device code, polling, exchange, refresh, `access_token`, and the redacted rate-limited / re-login-required / plain-failure error rendering
- `./credentials.rs` → `OAuthCredentials`, `LoginInfo`, `TokenSummary`, `TokenStatus`, and the `login`/`status`/`logout` entry points
- `./lock.rs` → `TokenFileLock`, `lock_path`, `LOCK_TIMEOUT`, the bounded/`try` acquisitions, and the non-unix `Unsupported` path
- `./tests.rs` → wiremock transport suites, tempfile store suites, and path-resolution tests
- `./refresh_tests.rs` → refresh-response precedence and the failure-classification branches
- `./token_tests.rs` → expiry precedence, the unresolvable-expiry rule, serialization, and `Debug` redaction
- `./credential_tests.rs` → login, status, and logout suites
- `./lock_tests.rs` → cross-process lock suite (unix only)
- `./support.rs` → shared test helpers (mock-server client, expiry arithmetic, JWT builder, secret reader)
