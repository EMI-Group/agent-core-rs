# src/backends/vendors — vendor moderation adapters

## Intent

One adapter per vendor moderation API, translating the vendor wire format into a `crate::RiskAssessment`.
All three are `GuardrailBackend` implementations, so they plug into `BackendGuardrail` and `GuardrailPipeline` like any other backend and never decide blocking themselves.
They have no Python counterpart (Python ships only the `ApiModelBackend` transport seam); they exist because the crate's parity goal names OpenAI, Azure, and Aliyun moderation as the deferred adapters.
No vendor SDK is linked: requests are built with `HttpRequest` and sent through the shared `HttpClient`, so the whole surface is offline-testable against a loopback server.
They depend on `crate::backends::http` for client construction (`http_client`), status/JSON mapping (`json_body`), and timeout sanitising (`resolve_timeout`).

## API Surface

```rust
// openai
pub const DEFAULT_API_URL: &str = "https://api.openai.com/v1/moderations";
pub const DEFAULT_MODEL: &str = "omni-moderation-latest";
pub const NAME: &str = "openai_moderation";
pub struct OpenAiModerationConfig { api_url, api_key: Option<SecretString>, model: String, organization: Option<String>,
                                    timeout: f64, max_retries: u32, risk_type: String }
impl OpenAiModerationConfig { with_api_url / with_api_key / with_model / with_organization / with_timeout / with_max_retries }
impl OpenAiModerationBackend {
    pub fn new(config: OpenAiModerationConfig) -> Result<Self>;   // fills blank url/model, Error::Config on client failure
    pub fn with_client(self, HttpClient) -> Self;
    pub fn config(&self) -> &OpenAiModerationConfig;
    pub fn parse_response(&self, &Value) -> Result<RiskAssessment>;
}   // Clone

// azure
pub const DEFAULT_API_VERSION: &str = "2024-09-01";
pub const DEFAULT_CATEGORIES: [&str; 4] = ["Hate", "SelfHarm", "Sexual", "Violence"];
pub const NAME: &str = "azure_content_safety";
pub struct AzureContentSafetyConfig { endpoint, api_key: Option<SecretString>, api_version, categories: Vec<String>,
                                      timeout: f64, max_retries: u32, risk_type: String }
impl AzureContentSafetyConfig { with_endpoint / with_api_key / with_api_version / with_categories / with_timeout / with_max_retries }
impl AzureContentSafetyBackend {
    pub fn new(config) -> Result<Self>;      // blank endpoint => Error::Validation
    pub fn with_client(self, HttpClient) -> Self;
    pub fn config(&self) -> &AzureContentSafetyConfig;
    pub fn analyze_url(&self) -> Result<String>;
    pub fn parse_response(&self, &Value) -> Result<RiskAssessment>;
}

// aliyun
pub const DEFAULT_ENDPOINT: &str = "https://green-cip.cn-shanghai.aliyuncs.com/";
pub const DEFAULT_API_VERSION: &str = "2022-03-02";
pub const DEFAULT_SERVICE: &str = "nickname_detection";
pub const NAME: &str = "aliyun_green";
pub enum AliyunGreenAuth { None, Bearer(SecretString), Aksk { access_key_id: SecretString, access_key_secret: SecretString } }  // kind() -> "none"|"bearer"|"aksk"
pub struct AliyunGreenConfig { endpoint, auth, service, api_version, region, timeout: f64, max_retries: u32, risk_type }
impl AliyunGreenConfig { with_endpoint / with_auth / with_service / with_api_version / with_region / with_timeout / with_max_retries }
impl AliyunGreenBackend {
    pub fn new(config) -> Result<Self>;      // blank endpoint or blank Aksk half => Error::Validation
    pub fn with_client(self, HttpClient) -> Self;
    pub fn config(&self) -> &AliyunGreenConfig;
    pub fn parse_response(&self, &Value) -> Result<RiskAssessment>;
}   // Clone
```

Crate root re-exports the three backends and their configs plus `AliyunGreenAuth`; the `DEFAULT_*`/`NAME` constants stay under their module path (`openjiuwen_security::backends::vendors::openai::NAME`).
Every config is `serde` (`snake_case`, `#[serde(default)]`, unknown fields tolerated), keeps seconds as `f64` (Python-style timeouts), and hides credentials behind `SecretString`.
`Default` values: OpenAI — public URL at construction, no key, `omni-moderation-latest`, `30.0` s, 2 retries, risk type `openai_moderation`; Azure — no endpoint, API version `2024-09-01`, the four default categories, `30.0` s, 2 retries, risk type `azure_content_safety`; Aliyun — public endpoint, `AliyunGreenAuth::None`, `nickname_detection`, `2022-03-02`, region `cn-shanghai`, `30.0` s, 2 retries, risk type `aliyun_green`.

## Wire Contracts

**OpenAI** — `POST {api_url}` (default `https://api.openai.com/v1/moderations`) with `{"input": <text>, "model": <model>}`.
`OpenAI-Organization` is sent when configured non-blank; `Authorization: Bearer <key>` only when a non-empty key is set.
`results` missing, empty, or non-array → `Error::Transport`; `results[0].flagged != true` → safe assessment.
A flagged result takes its severity from the highest finite `category_scores` entry (`>= 0.90` Crit, `>= 0.75` High, `>= 0.50` Medium, else Low; ties go to the lexicographically smaller category) and reports that category as the risk type with the score as confidence.
A flagged result without scores reports `High` with the configured `risk_type` and confidence `1.0`.
`details`: `provider` = `"openai"`, `model` (result model, else response model, else configured), `categories` (sorted flagged names), `top_score` (4-decimal), `request_id` when the reply carries one.

**Azure** — `POST {endpoint}/contentsafety/text:analyze?api-version={version}` with `{"text": <text>, "categories": [...]}`.
`analyze_url` keeps a configured endpoint that already contains `contentsafety/text:analyze` (appending `api-version` only when it has no query) and trims a trailing `/`.
The resource key travels in `Ocp-Apim-Subscription-Key`, never as a bearer token, and is omitted when blank.
`categoriesAnalysis` missing/empty → `Error::Transport`; every severity `0` → safe.
Severities are clamped to `0..=7` (floats truncated, negatives/non-numerics read as `0`) and band as `1..=2` Low, `3..=4` Medium, `5..=6` High, `>= 7` Critical; the highest category names the risk type (ties keep the first in wire order) and confidence is `max_severity / 7` rounded to 4 decimals.
`details`: `provider` = `"azure"`, `categories` (name → severity map), `max_severity`, `blocklist_matches` (count only; blocklist terms are dropped).
A 2xx reply carrying an `error` object becomes `Error::Transport` naming the vendor `code`.

**Aliyun** — `POST {endpoint}` with the `TextModeration` RPC (version `2022-03-02`) and body `{"Service": <service>, "ServiceParameters": "{\"content\": <text>}"}`.
Headers `host`, `x-acs-action: TextModeration`, `x-acs-version`, `x-acs-date` (RFC 3339 UTC), `x-acs-signature-nonce` (UUID v4), `x-acs-content-sha256` (SHA-256 of the body).
`AliyunGreenAuth::Bearer` sends `Authorization: Bearer <token>`; `Aksk` builds `ACS3-HMAC-SHA256 Credential=<id>,SignedHeaders=host;x-acs-action;x-acs-content-sha256;x-acs-date;x-acs-signature-nonce;x-acs-version,Signature=<hex>` over the canonical request (`POST\n/\n<headers>\n<signed names>\n<body sha>`), signed with an in-module RFC 2104 HMAC-SHA256 over `sha2` (the access key secret is never transmitted).
`Code` absent, non-integer, or non-`200` → `Error::Transport` quoting the code and at most 120 characters of the collapsed `Message`.
`Data.Labels` is split on `,`, trimmed, and filtered; blank or `nonLabel`-only labels → safe, otherwise the first label is the risk type.
Confidence is the maximum `Data.Result[].Confidence` (`0..=100`) banded at `90`/`75`/`50` and reported as `max / 100` rounded to 4 decimals; no score reports `High` with confidence `1.0`.
`details`: `provider` = `"aliyun"`, `service`, `region`, `labels`, `confidence`, `request_id` when present; vendor `RiskWords`, `Reason`, and per-result `RiskWords` are deliberately dropped.

## Constraints

- `GuardrailBackend` is object-safe and async; `name()` returns the module `NAME` constant.
- Every adapter short-circuits blank/whitespace `text_view()` to `crate::backends::safe_assessment()` without any HTTP call.
- Secrets are `SecretString`; `Debug` prints presence/kind only, and serialization keeps plaintext for config round-trips.
- Findings never carry scanned text or vendor matched words; errors carry service, status, vendor code, and a bounded body excerpt only.
- Failure mapping is uniform: `Error::Validation` for unusable construction input, `Error::Transport` for status/decoding/vendor-code failures, `Error::Timeout` when the deadline expires, `Error::Config` when the shared client cannot be built.
- Retries come from the shared client (`429` and `5xx`, exponential backoff capped at 5 s); `max_retries = 0` sends exactly one request, which is what the test suites use.
- One request per scan and only the first result is graded (OpenAI `results[0]`); batch/streaming moderation is out of scope.
- `OpenAiModerationBackend` and `AliyunGreenBackend` derive `Clone` (their `HttpClient` is cloneable); `AzureContentSafetyBackend` does not derive `Clone` but is rebuildable from `config()`.
- Aliyun signing uses the wall clock (`chrono::Utc::now`) and a fresh UUID nonce, so tests assert header presence/shape and the deterministic canonical-request/signature unit vectors rather than a fixed wire signature.

## Routing Table

- `./mod.rs` → module list and the six re-exports
- `./openai.rs` → OpenAI moderation config/backend, score-band mapping, wire + loopback tests behind the module docs
- `./azure.rs` → Azure Content Safety config/backend, URL composition, severity mapping, wire + loopback tests
- `./aliyun/` → Aliyun Green adapter as a directory module (`src/backends/vendors/aliyun/CONTEXT.md`): `mod.rs` (config/auth/backend, RPC body, label/confidence mapping), `signing.rs` (ACS3-HMAC-SHA256), `tests.rs` (wire + loopback tests)

## Test Strategy

- Each module carries `#[cfg(test)] mod tests` over a loopback `wiremock` server: exact request path/headers/body, credential placement, benign vs flagged replies, band boundaries, missing/error payloads, and empty-text short-circuit.
- Config tests pin every default, the serde round-trip, unknown-field tolerance, and that `Debug` never prints a credential.
- Aliyun additionally pins SHA-256/HMAC-SHA256 test vectors and the determinism of the canonical request and signature; the live wire test mounts a `Bearer`/`Aksk` request and checks the header shape.
- Crate-level behaviour through the pipeline (blocking, failure policy, stage binding, no content leak) lives in `crates/openjiuwen-security/tests/adapters.rs`.
