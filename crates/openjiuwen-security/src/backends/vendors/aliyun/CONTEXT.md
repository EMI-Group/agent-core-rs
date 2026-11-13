# vendors/aliyun — Aliyun Green (内容安全) adapter

## Intent

Adapter for Aliyun's Green text-moderation RPC API (`green-20220302`, action `TextModeration`), a vendor surface the Python SDK never shipped.
The adapter owns the RPC wire format (JSON body with a JSON-encoded `ServiceParameters`, `x-acs-*` headers, ACS3-HMAC-SHA256 signing) and maps `Data.Labels` / `Data.Result[].Confidence` onto `crate::RiskAssessment`.
No vendor SDK is used: requests go through `crate::backends::http::{http_client, json_body, resolve_timeout}` on the shared `openjiuwen_common::http::HttpClient`.
The adapter is a `GuardrailBackend`, so it plugs into `BackendGuardrail` and `GuardrailPipeline` like any other backend, and it never decides blocking.

## API Surface

```
consts:  DEFAULT_ENDPOINT "https://green-cip.cn-shanghai.aliyuncs.com/", DEFAULT_API_VERSION "2022-03-02",
         DEFAULT_SERVICE "nickname_detection", NAME "aliyun_green"
auth:    AliyunGreenAuth { None, Bearer(SecretString), Aksk { access_key_id, access_key_secret } } with kind() -> "none"|"bearer"|"aksk"
config:  AliyunGreenConfig { endpoint, auth, service, api_version, region, timeout /* f64 seconds, 30.0 */,
                             max_retries /* 2 */, risk_type /* "aliyun_green" */ } + Default + builders
backend: AliyunGreenBackend::new(config) -> Result<Self>   // blank endpoint or blank AK/SK => Error::Validation
         with_client(HttpClient) -> Self, config() -> &AliyunGreenConfig
         parse_response(&Value) -> Result<RiskAssessment>   // pure, fixture-testable
trait:   #[async_trait] impl GuardrailBackend   // name() == NAME
```

`AliyunGreenAuth` and `AliyunGreenConfig` are serde (`snake_case`, `#[serde(default)]`, unknown fields tolerated); `SecretString` keeps credentials redacted in `Debug`.

## Wire Contract

Request: `POST <endpoint>` (path `/`) with body `{"Service": "<service>", "ServiceParameters": "{\"content\":\"<scanned text>\"}"}`.
Headers: `host`, `x-acs-action: TextModeration`, `x-acs-version`, `x-acs-date` (`%Y-%m-%dT%H:%M:%SZ` UTC), `x-acs-signature-nonce` (UUID v4), `x-acs-content-sha256` (hex SHA-256 of the body bytes).
`Bearer` auth adds `Authorization: Bearer <token>`.
`Aksk` auth adds `Authorization: ACS3-HMAC-SHA256 Credential=<id>,SignedHeaders=<names>,Signature=<hex>`, signing `host;x-acs-action;x-acs-content-sha256;x-acs-date;x-acs-signature-nonce;x-acs-version` over `ACS3-HMAC-SHA256\n<sha256(canonical request)>`.
Response: `{"Code": 200, "Message": "OK", "RequestId": "...", "Data": {"Labels": "a,b", "Reason": "...", "Result": [{"Label": "a", "Confidence": 95.5, "RiskWords": "..."}]}}`.
`Code` != 200 / missing / non-integer => `Error::transport("aliyun green returned code <code>")`.
Labels are the union of `Data.Labels` (comma-separated, `nonLabel` ignored) and `Data.Result[].Label`, order-preserving and deduplicated.
A reply is safe only when that union is empty and no `Result` entry carries a positive confidence.
Risky replies map the maximum confidence through `>= 90` Critical, `>= 75` High, `>= 50` Medium, else Low (labels without a usable score => High); `risk_type` is the first label, falling back to `config.risk_type`; `confidence = max/100` rounded to 4 dp.
`details = {"provider": "aliyun", "service", "region", "labels": [union], "confidence", "request_id"}`; `RiskWords`, `Reason`, and the scanned text are never emitted.

## Constraints

- Empty/whitespace `GuardrailContext::text_view()` short-circuits to a safe assessment with no HTTP request.
- Errors are `openjiuwen_common::error::{Error, Result}` only; configuration problems are `Error::Validation`, transport/business-code/parse problems are `Error::Transport`.
- No credential, no scanned text, and no vendor matched words appear in `Debug`, `details`, logs, or error messages.
- No new dependencies: signing uses `sha2` (in-module HMAC-SHA256, RFC 2104), ids use `uuid`, timestamps use `chrono`.
- Tests are offline and loopback-only (`wiremock`); the signature helper is deterministic for fixed `(date, nonce, body)` inputs while the live adapter stamps the current time.
- Each file stays under ~1000 lines.

## Routing Table

- `./mod.rs` → constants, `AliyunGreenAuth`, `AliyunGreenConfig`, `AliyunGreenBackend`, request building, label/confidence mapping
- `./signing.rs` → HMAC-SHA256, hex encoding, canonical request, ACS3 `Authorization` builder and their unit tests
- `./tests.rs` → config/serde/`Debug`/validation tests, `wiremock` Bearer and AK/SK wire assertions, `parse_response` fixtures, empty-text short-circuit

## Test Strategy

- `signing.rs` unit tests pin HMAC-SHA256 against RFC 4231 vectors and assert the canonical-request layout plus signature determinism.
- `tests.rs` asserts the outgoing request (path, auth header shape, `x-acs-*` headers, body hash equals SHA-256 of the recorded body, JSON-string `ServiceParameters`) for both auth modes against a loopback `wiremock` server.
- `parse_response` fixtures cover every confidence band, `nonLabel`/blank labels, Result-sourced labels (the false-negative regression), label-only replies, business-code failures, and the absence of `RiskWords`/`Reason` in serialized details.
- Crate-level behavior through the pipeline lives in `crates/openjiuwen-security/tests/adapters.rs`.
- Gates: `cargo fmt -p openjiuwen-security`, `cargo test -p openjiuwen-security --offline`, `cargo clippy -p openjiuwen-security --all-targets --offline -- -D warnings`, `cargo doc -p openjiuwen-security --no-deps --offline`.
