# oauth — OpenAI account OAuth transport and credential store

## Intent

Port the `OpenAIAccount` provider of the reference SDK: the OAuth 2.0 device authorization grant against `https://auth.openai.com`, the token exchange and refresh that rotate the refresh token, and the private auth-store file that keeps the credentials between runs.
The Python source lives under `openjiuwen/core/foundation/llm/` (the account/OAuth client); here the transport is implemented on `openjiuwen_common::http::HttpClient` instead of the vendor `openai` SDK.
Only the token endpoints live in this module: sending a model request with the resulting bearer token — the Responses-API transport — is out of scope.

## API Surface

- `OAuthToken` — `access_token`, `refresh_token`, `id_token` (all `SecretString`), `expires_at: Option<DateTime<Utc>>`, `token_type` (`Bearer` by default), `scope`; `new` plus `with_refresh_token`, `with_id_token`, `with_expires_at`, `with_token_type`, `with_scope`, `effective_expires_at()` and `is_expired(skew)`; redacting `Debug`, canonical `Serialize`, tolerant `Deserialize` through a private `TokenPayload`.
- `TokenStore` — async trait (`load`, `save`, `clear`) and the persistence seam above the client.
- `InMemoryTokenStore` — process-memory backend with `new` and `Default`; the testable default.
- `FileTokenStore` — auth-store JSON backend: `new(path)`, `with_base_url`, `path`, `base_url`, `default_path`.
- `DeviceCode` — pending device authorization: `user_code`, `device_auth_id`, `interval` (default 5 s, never below 3 s when parsed), `expires_in` (`0` means unspecified), `new`, `with_interval` (no floor, the test seam), `with_expires_in`.
- `OpenAiOAuthClient` — the HTTP side: `new`, `with_http_client`, `with_base_url`, `with_client_id`, `http_client`, `base_url`, `client_id`, `request_device_code`, `poll_device_token`, `poll_device_token_with_deadline`, `exchange_code`, `refresh`, `access_token`.
- Exported constants from `client`: `DEFAULT_ISSUER` (`https://auth.openai.com`), `DEFAULT_CLIENT_ID`, `DEVICE_CALLBACK_PATH` (`/deviceauth/callback`).
- Exported constants from `store`: `PROVIDER_KEY` (`OpenAIAccount`), `AUTH_MODE` (`openai_account_oauth`), `DEFAULT_BASE_URL` (`https://chatgpt.com/backend-api/codex`), `AUTH_FILE_ENV` (`OPENJIUWEN_AUTH_FILE`), `REFRESH_SKEW` (120 s).

## Wire Contract

- Device code: `POST {issuer}/api/accounts/deviceauth/usercode` with `{"client_id"}` → `DeviceCode`; a non-2xx answer becomes an error with a redacted, truncated body.
- Poll: `POST {issuer}/api/accounts/deviceauth/token` with `{"device_auth_id", "user_code"}`; `403`/`404` mean "still pending", `200` either carries a token or `authorization_code` + `code_verifier`.
- A `200` carrying an authorization pair is exchanged through `exchange_code` with `redirect_uri = {issuer}/deviceauth/callback`; a missing code or verifier is an error.
- Exchange and refresh share `POST {issuer}/oauth/token`, form-encoded: `grant_type=authorization_code` with `code`, `redirect_uri`, `client_id`, `code_verifier`, or `grant_type=refresh_token` with `refresh_token`, `client_id`.
- Polling starts after `max(interval, 1 s)`, doubles while pending up to 10 s, and the whole flow is bounded by `min(device_code.expires_in, 15 min)`; a parsed `interval` is never below 3 s (Python parity: `max(3, interval)`).
- The expiry of a token response is resolved with the reference implementation's precedence: `expires_at` (epoch seconds or RFC 3339) → `now + max(expires_in, 0)` (numeric strings accepted) → the previous token's `expires_at` → the `exp` claim of a JWT `access_token` (base64url payload, signature never verified).
  Nothing resolvable means "unknown", and an unknown expiry never refreshes.
  Serialization always writes `expires_at` in epoch seconds and never re-emits `expires_in`.
- A refresh response may rotate the refresh token; a field it omits (`refresh_token`, `id_token`, `scope`, `token_type`, expiry) is inherited from the token being replaced.
- `access_token` refreshes when `force_refresh` is set or the token `is_expired(REFRESH_SKEW)` (120 s skew, `effective_expires_at`), and saves the (possibly rotated) result before returning it.
- A rejected request is classified through the error message: `429` is rate limiting whose message says the credentials are still valid and no re-login is required; `401`/`403` or a vendor code `invalid_grant`/`invalid_token`/`invalid_request`/`refresh_token_reused` says the credentials are unusable and a re-login is required; anything else is a plain provider failure.
- Store envelope: `{"providers": {"OpenAIAccount": {"tokens": {...}, "auth_mode": "openai_account_oauth", "base_url": "..."}}}`; only those three keys of the entry are written.
- Path resolution: `$OPENJIUWEN_AUTH_FILE`, then `$OPENJIUWEN_HOME/auth.json`, then `~/.openjiuwen/auth.json` (`%USERPROFILE%` stands in for `$HOME` on Windows), with a leading `~` expanded.

## Constraints

- Every request goes through `openjiuwen_common::http::HttpClient`; never `reqwest` directly, never a hand-rolled retry loop.
- No token, authorization code, code verifier, or refresh token is logged or formatted into a message: credentials are `SecretString`, `Debug` redacts, and error bodies pass through `security::redact_str` and are truncated to 500 characters.
- Error variants: `Config` for "no stored token", "no refresh token", and an unresolvable auth path; `Llm` for non-2xx responses, bodies that are not a token, invalid JSON, and an expired device code; `Validation` for a malformed auth-store envelope; `External` for I/O and blocking-task failures.
  The rate-limited / re-login-required / plain-failure distinction lives in the message (no new error kind), and a vendor code is echoed only when it is one of the four known re-login codes.
- Blocking file IO runs on `tokio::task::spawn_blocking`; no blocking call on a runtime thread.
- Logging is `tracing` only (pending poll status at debug); no `println!`/`eprintln!`.
- Files stay under the ~1000-line rule (`store.rs` 831, `client.rs` 503, `tests.rs` 800, `refresh_tests.rs` ~300, `token_tests.rs` ~290, `support.rs` 63); no dependencies beyond the crate's existing `chrono`, `base64`, `serde`/`serde_json`, `url`, and dev-only `wiremock`/`tempfile`.

## Design Decisions

- `FileTokenStore` wraps each synchronous helper in `spawn_blocking`, cloning the path, base URL, and token first, so the store stays `Send + Sync` behind the async `TokenStore` trait.
- Writes are atomic: render pretty JSON, write `<name>.<pid>.tmp` in the same directory, `sync_all`, chmod to `0o600` on unix, rename over the target, then restrict the target again so an existing loose file is tightened.
- The save is a read-merge-write: other providers' entries and unknown keys inside the `OpenAIAccount` entry survive; only `tokens`, `auth_mode`, and `base_url` are set.
- `clear` removes just the `providers.OpenAIAccount` entry and leaves the file untouched when the entry is absent.
- Every omitted field of a refresh response — `refresh_token`, `id_token`, `scope`, `token_type`, and the expiry — is inherited from the previous token by one resolver (`TokenPayload::into_token`), so rotation and non-rotation take the same code path.
- The client holds no token cache: the store is the single source of truth, so several clients or processes cannot disagree about the current token.
- An unknown expiry means "not expired", so a token with no resolvable expiry does not trigger a refresh round-trip on every call; a skew too large for chrono saturates to expired.
- The JWT `exp` fallback is read without verifying the signature: it only decides *when* to refresh, and a forged claim can at most cost one extra refresh whose result must still come from the issuer.
- `default_path` returns `Error::Config` instead of a relative fallback, so a credential file can never land in the working directory silently.
- `429` is reported as rate limiting whose message states that the stored credentials are still valid and no re-login is required, which keeps a caller from forcing a re-login; the vendor body is redacted and an unrecognised vendor code is never echoed, so an error message cannot smuggle a secret.
- `DeviceCode` is tolerant by design: a missing `interval` is 5 s and a smaller advertised one is raised to 3 s, a missing `expires_in` is 0, and both accept numbers or numeric strings; the `with_interval` builder applies no floor, so tests can drive the poll quickly.
- The issuer and store path are injectable (`with_base_url`, `with_http_client`, `FileTokenStore::new`) so tests point at loopback and temporary directories.

## Test Strategy

- The suite is split by responsibility and declared from `mod.rs`: `tests.rs` (device flow, exchange, refresh rotation, `access_token`, file store, path resolution), `refresh_tests.rs` (refresh-response precedence and failure classification), `token_tests.rs` (expiry precedence and serialization), and `support.rs` (the shared mock-server client, expiry arithmetic, JWT builder, and secret reader).
- `tests.rs` covers the transport against `wiremock` on loopback: device-code request body, the three-second interval floor, redacted rejection, polling through pending to exchange, a token returned by the poll itself, expiry, the exact exchange and refresh form (three parameters, no `scope`/`redirect_uri`), refresh rotation and merge, `access_token` refresh/force/no-network/no-token/no-refresh-token/unresolvable-expiry/past-JWT-claim paths.
- `refresh_tests.rs` covers every expiry-precedence step (`expires_at` → `expires_in` → previous `expires_at` → JWT `exp` → unknown), refresh rotation with and without a rotated refresh token, `token_type` carry-over and rotation, and one wiremock test per failure branch (`429`, `401` + `invalid_grant`, `403` without a known code, `400` + `refresh_token_reused`, `500` + `server_error`, and a `200` carrying no token).
- File-store tests use `tempfile` directories: round trip, other-provider preservation, `clear`, `0o600` permissions with no leftover temporary file, and an unreadable store.
- `default_path` precedence is tested behind a process-environment mutex with restore, and never touches `~/.openjiuwen`.
- `token_tests.rs` covers `expires_in` → `expires_at` (including numeric strings and a non-positive lifetime), epoch and RFC 3339 expiries, the JWT `exp` fallback (unpadded and padded payloads, missing claim, malformed token, opaque token), canonical serialization, round trip, `Debug` redaction, and `DeviceCode` tolerance.
- Everything is offline: loopback sockets only, no credential, no wall-clock ordering, short poll deadlines and timeouts.

## Known Gaps

- No Responses-API (or any other model) transport that consumes the bearer token; the module stops at the token endpoints.
- No `/models` catalog cache and no account metadata fetch.
- No login/status/logout CLI: callers drive `request_device_code`, `poll_device_token`, and `access_token` themselves.
- No cross-process file lock; concurrent writers rely on the atomic rename alone.
- No `AuthStatus` type (the Python status snapshot) and no shipped keyring or secret-manager backend — `TokenStore` is the seam for one.
- No `OPENJIUWEN_*` override for the issuer or client id; those are constructor arguments only.

## Routing Table

- `./mod.rs` → module documentation, `client`/`store` declarations, and the crate-facing re-export list
- `./store.rs` → `OAuthToken` (expiry precedence incl. the JWT `exp` fallback), `parse_token_payload`, `DeviceCode`, the `TokenStore` trait, the memory and file backends, envelope merge, path resolution, atomic owner-only writes
- `./client.rs` → `OpenAiOAuthClient`: device code, polling, exchange, refresh, `access_token`, and the redacted rate-limited / re-login-required / plain-failure error rendering
- `./tests.rs` → wiremock transport suites, tempfile store suites, and path-resolution tests
- `./refresh_tests.rs` → refresh-response precedence and the failure-classification branches
- `./token_tests.rs` → expiry precedence, serialization, and `Debug` redaction
- `./support.rs` → shared test helpers (mock-server client, expiry arithmetic, JWT builder, secret reader)
