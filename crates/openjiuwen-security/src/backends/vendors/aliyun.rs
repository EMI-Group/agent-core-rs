//! Aliyun Green (内容安全) text-moderation adapter.
//!
//! [`AliyunGreenBackend`] speaks the Aliyun Green `TextModeration` RPC
//! (`green-20220302`) over the crate-shared [`crate::backends::http`] helpers:
//! it posts `{"Service": ..., "ServiceParameters": "{\"content\": ...}"}` and
//! maps `Data.Labels` and the per-label `Confidence` scores onto a
//! [`crate::RiskAssessment`]. A label present means risk, the first label is
//! `risk_type`, and the vendor's `RiskWords`, `Reason`, and the scanned text
//! never appear in `details`, logs, or errors; empty text is safe without a
//! request. Requests use either a gateway `Bearer` token or the native
//! `ACS3-HMAC-SHA256` signature over six headers, so the public endpoint
//! accepts them without an SDK.
#![allow(clippy::module_name_repetitions)]

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};
use openjiuwen_common::security::SecretString;

use crate::backends::http::{http_client, json_body, resolve_timeout};
use crate::backends::safe_assessment;
use crate::guardrail::GuardrailBackend;
use crate::{GuardrailContext, RiskAssessment, RiskLevel};

/// Endpoint used when the configuration does not name one.
pub const DEFAULT_ENDPOINT: &str = "https://green-cip.cn-shanghai.aliyuncs.com/";

/// `x-acs-version` sent when the configuration does not name one.
pub const DEFAULT_API_VERSION: &str = "2022-03-02";

/// Moderation service sent when the configuration does not name one.
pub const DEFAULT_SERVICE: &str = "nickname_detection";

/// Stable backend name, mirrored in findings and logs.
pub const NAME: &str = "aliyun_green";

/// Default request timeout in seconds.
const DEFAULT_TIMEOUT: f64 = 30.0;
/// Retries after the first attempt.
const DEFAULT_MAX_RETRIES: u32 = 2;
/// Region recorded in `details` when none is configured.
const DEFAULT_REGION: &str = "cn-shanghai";
/// Risk type reserved for parity with the other vendor configs.
const DEFAULT_RISK_TYPE: &str = "aliyun_green";
/// First-retry backoff handed to the shared client, matching its default.
const DEFAULT_RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(200);
/// Log target shared with the other vendor adapters.
const LOG_TARGET: &str = "openjiuwen_security::backends::vendors::aliyun";
/// Provider label recorded in `details`.
const PROVIDER: &str = "aliyun";
/// Diagnostic label used in `HTTP` error messages.
const SERVICE: &str = "aliyun green";
/// RPC action carried by `x-acs-action`.
const ACTION: &str = "TextModeration";
/// Signature algorithm name, used for the string to sign and the header.
const SIGNATURE_ALGORITHM: &str = "ACS3-HMAC-SHA256";
/// Sorted, semicolon-joined names of the headers the signature covers.
const SIGNED_HEADER_NAMES: &str =
    "host;x-acs-action;x-acs-content-sha256;x-acs-date;x-acs-signature-nonce;x-acs-version";
/// The label the vendor reports for benign content.
const NON_LABEL: &str = "nonLabel";
/// Maximum vendor status-message characters quoted in an error.
const ERROR_MESSAGE_CHARS: usize = 120;
/// `Confidence` at or above this value reports [`RiskLevel::Critical`].
const CRITICAL_CONFIDENCE: f64 = 90.0;
/// `Confidence` at or above this value reports [`RiskLevel::High`].
const HIGH_CONFIDENCE: f64 = 75.0;
/// `Confidence` at or above this value reports [`RiskLevel::Medium`].
const MEDIUM_CONFIDENCE: f64 = 50.0;
/// Scale factor that rounds a reported confidence to four decimal places.
const CONFIDENCE_SCALE: f64 = 10_000.0;
/// HMAC block size in bytes (SHA-256 uses 64-byte blocks).
const HMAC_BLOCK_SIZE: usize = 64;

/// How a request authenticates against Aliyun Green.
///
/// The variants hold [`SecretString`] values, so `Debug` and logs show a
/// redaction marker while serialization keeps the plaintext for config
/// round-trips.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AliyunGreenAuth {
    /// No credential is attached.
    #[default]
    None,
    /// `Authorization: Bearer <token>` for gateway-fronted deployments.
    Bearer(SecretString),
    /// `ACS3-HMAC-SHA256` request signing with an Aliyun access key pair.
    Aksk {
        /// Access key id, named in the `Credential=` clause.
        access_key_id: SecretString,
        /// Access key secret, the `HMAC-SHA256` signing key; never transmitted.
        access_key_secret: SecretString,
    },
}

impl AliyunGreenAuth {
    /// Short label of the credential kind: `"none"`, `"bearer"`, or `"aksk"`.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bearer(_) => "bearer",
            Self::Aksk { .. } => "aksk",
        }
    }
}

/// Configuration of [`AliyunGreenBackend`].
///
/// Field names are `snake_case`, unknown JSON fields are tolerated, and a
/// missing field falls back to its default. Blank `service`, `api_version`, and
/// `region` values are replaced by their defaults when the backend is built; a
/// blank `endpoint` is rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct AliyunGreenConfig {
    /// Endpoint URL, [`DEFAULT_ENDPOINT`] unless overridden; `new` rejects a
    /// blank value.
    pub endpoint: String,
    /// How requests authenticate.
    pub auth: AliyunGreenAuth,
    /// Moderation service name; blank means [`DEFAULT_SERVICE`].
    pub service: String,
    /// `x-acs-version` value; blank means [`DEFAULT_API_VERSION`].
    pub api_version: String,
    /// Region label recorded in `details`; blank means `"cn-shanghai"`.
    pub region: String,
    /// Request timeout in seconds (`30.0` by default).
    pub timeout: f64,
    /// Retries after the first attempt (`2` by default).
    pub max_retries: u32,
    /// Fallback risk type kept for parity with the other vendor configs; the
    /// vendor's first label is used whenever a label is present.
    pub risk_type: String,
}

impl Default for AliyunGreenConfig {
    /// The documented defaults: the public endpoint, no credential, the
    /// `nickname_detection` service, `2022-03-02`, `30.0` s, two retries.
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_ENDPOINT.to_owned(),
            auth: AliyunGreenAuth::None,
            service: DEFAULT_SERVICE.to_owned(),
            api_version: DEFAULT_API_VERSION.to_owned(),
            region: DEFAULT_REGION.to_owned(),
            timeout: DEFAULT_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
            risk_type: DEFAULT_RISK_TYPE.to_owned(),
        }
    }
}

impl AliyunGreenConfig {
    /// Sets the endpoint URL; `new` rejects a blank value.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sets how requests authenticate.
    #[must_use]
    pub fn with_auth(mut self, auth: AliyunGreenAuth) -> Self {
        self.auth = auth;
        self
    }

    /// Sets the moderation service; blank means [`DEFAULT_SERVICE`].
    #[must_use]
    pub fn with_service(mut self, service: impl Into<String>) -> Self {
        self.service = service.into();
        self
    }

    /// Sets the `x-acs-version` value; blank means [`DEFAULT_API_VERSION`].
    #[must_use]
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
        self
    }

    /// Sets the region label recorded in `details`.
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Sets the request timeout in seconds.
    #[must_use]
    pub fn with_timeout(mut self, seconds: f64) -> Self {
        self.timeout = seconds;
        self
    }

    /// Sets the number of retries after the first attempt.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}

/// Backend that grades content through the Aliyun Green `TextModeration` RPC.
///
/// `analyze` short-circuits empty text, posts the remaining text, and grades
/// the reply in [`AliyunGreenBackend::parse_response`]. The backend owns one
/// shared [`HttpClient`], so it is cheap to clone and safe to share across
/// guardrails and pipelines.
#[derive(Clone)]
pub struct AliyunGreenBackend {
    config: AliyunGreenConfig,
    client: HttpClient,
}

impl AliyunGreenBackend {
    /// Builds a backend from `config`, filling the blank service, version, and
    /// region defaults and trimming the endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `config.endpoint` is blank or when an
    /// `Aksk` credential has a blank access key id or secret, and
    /// [`Error::Config`] when the shared `HTTP` client cannot be built.
    pub fn new(config: AliyunGreenConfig) -> Result<Self> {
        let config = normalize_config(config)?;
        if let AliyunGreenAuth::Aksk {
            access_key_id,
            access_key_secret,
        } = &config.auth
        {
            if access_key_id.expose_secret().trim().is_empty()
                || access_key_secret.expose_secret().trim().is_empty()
            {
                return Err(Error::validation(
                    "aliyun green backend requires both an access key id and an access key secret",
                ));
            }
        }
        let client = http_client(
            resolve_timeout(config.timeout),
            config.max_retries,
            DEFAULT_RETRY_INITIAL_BACKOFF,
        )?;
        tracing::debug!(
            target: LOG_TARGET,
            service = %config.service,
            auth = config.auth.kind(),
            "aliyun green backend initialised"
        );
        Ok(Self { config, client })
    }

    /// Uses `client` instead of the client built by [`AliyunGreenBackend::new`].
    #[must_use]
    pub fn with_client(mut self, client: HttpClient) -> Self {
        self.client = client;
        self
    }

    /// Returns the configuration this backend was built from.
    #[must_use]
    pub fn config(&self) -> &AliyunGreenConfig {
        &self.config
    }

    /// Grades one `TextModeration` reply.
    ///
    /// A non-`200` (or missing, or non-integer) `Code` is a transport failure.
    /// Missing, blank, or `nonLabel`-only `Data.Labels` is a safe assessment.
    /// Otherwise the highest `Result[].Confidence` (`0..=100`) picks the
    /// severity (`>= 90` [`RiskLevel::Critical`], `>= 75` [`RiskLevel::High`],
    /// `>= 50` [`RiskLevel::Medium`], else [`RiskLevel::Low`]; no score reports
    /// [`RiskLevel::High`]), the first label is `risk_type`, and the rounded
    /// confidence (`0..=1`) is reported. `details` carries the provider,
    /// service, region, trimmed labels, confidence, and the request id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when `Code` is absent, non-integer, or not
    /// `200`.
    pub fn parse_response(&self, response: &Value) -> Result<RiskAssessment> {
        if response.get("Code").and_then(Value::as_i64) != Some(200) {
            return Err(code_error(response));
        }
        let data = response.get("Data");
        let labels = data
            .and_then(|data| data.get("Labels"))
            .and_then(Value::as_str)
            .map(parse_labels)
            .unwrap_or_default();
        let Some(risk_type) = labels.first().cloned() else {
            return Ok(safe_assessment());
        };
        let max_confidence = data
            .and_then(|data| data.get("Result"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("Confidence").and_then(Value::as_f64))
            .filter(|confidence| confidence.is_finite())
            .reduce(f64::max);
        let (risk_level, confidence) = max_confidence.map_or((RiskLevel::High, 1.0), |max| {
            (level_for_confidence(max), round_confidence(max / 100.0))
        });

        let mut details = json!({
            "provider": PROVIDER,
            "service": self.config.service,
            "region": self.config.region,
            "labels": labels,
            "confidence": confidence,
        });
        if let Some(request_id) = response
            .get("RequestId")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            details["request_id"] = json!(request_id);
        }
        Ok(RiskAssessment {
            has_risk: true,
            risk_level,
            risk_type: Some(risk_type),
            confidence,
            details: Some(details),
        })
    }
}

impl fmt::Debug for AliyunGreenBackend {
    /// Prints the endpoint and settings; the credential is reduced to its kind.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AliyunGreenBackend")
            .field("endpoint", &self.config.endpoint)
            .field("service", &self.config.service)
            .field("auth", &self.config.auth.kind())
            .field("timeout", &self.config.timeout)
            .field("max_retries", &self.config.max_retries)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl GuardrailBackend for AliyunGreenBackend {
    fn name(&self) -> &str {
        NAME
    }

    /// Posts the context's text projection to the moderation endpoint.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the endpoint carries no host,
    /// [`Error::Transport`] for connection, status, business-code, and decoding
    /// failures, and [`Error::Timeout`] when the deadline expires. Errors never
    /// carry the scanned text.
    async fn analyze(&self, ctx: &GuardrailContext) -> Result<RiskAssessment> {
        let text = ctx.text_view();
        if text.trim().is_empty() {
            return Ok(safe_assessment());
        }
        let host = host_of(&self.config.endpoint)?;
        let body = request_body(&self.config.service, &text);
        let body_bytes = serde_json::to_vec(&body).map_err(|error| {
            Error::validation(format!("serializing aliyun green body: {error}"))
        })?;
        let content_sha256 = sha256_hex(&body_bytes);
        let date = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let nonce = uuid::Uuid::new_v4().to_string();

        let authorization = match &self.config.auth {
            AliyunGreenAuth::None => None,
            AliyunGreenAuth::Bearer(token) if token.is_empty() => None,
            AliyunGreenAuth::Bearer(token) => Some(format!("Bearer {}", token.expose_secret())),
            AliyunGreenAuth::Aksk {
                access_key_id,
                access_key_secret,
            } => Some(sign_authorization(
                access_key_id.expose_secret(),
                access_key_secret.expose_secret(),
                &host,
                &self.config.api_version,
                &date,
                &nonce,
                &content_sha256,
            )),
        };

        let mut request = HttpRequest::post(self.config.endpoint.as_str())
            .timeout(resolve_timeout(self.config.timeout))
            .header("host", host)
            .header("x-acs-action", ACTION)
            .header("x-acs-version", self.config.api_version.as_str())
            .header("x-acs-date", date)
            .header("x-acs-signature-nonce", nonce)
            .header("x-acs-content-sha256", content_sha256)
            .json(&body)?;
        if let Some(authorization) = authorization {
            request = request.header("Authorization", authorization);
        }

        let response = self.client.execute(request).await?;
        tracing::debug!(
            target: LOG_TARGET,
            status = response.status,
            "aliyun green request completed"
        );
        let reply = json_body(response, SERVICE)?;
        self.parse_response(&reply)
    }
}

/// Trims the endpoint and fills blank service, version, and region defaults.
fn normalize_config(mut config: AliyunGreenConfig) -> Result<AliyunGreenConfig> {
    let endpoint = config.endpoint.trim();
    if endpoint.is_empty() {
        return Err(Error::validation(
            "aliyun green backend requires an endpoint URL",
        ));
    }
    config.endpoint = endpoint.to_owned();
    config.service = or_default(config.service.trim(), DEFAULT_SERVICE);
    config.api_version = or_default(config.api_version.trim(), DEFAULT_API_VERSION);
    config.region = or_default(config.region.trim(), DEFAULT_REGION);
    Ok(config)
}

/// Returns `value` when non-empty, else `default`.
fn or_default(value: &str, default: &str) -> String {
    if value.is_empty() {
        default.to_owned()
    } else {
        value.to_owned()
    }
}

/// Builds the RPC payload, with `ServiceParameters` as a JSON-encoded string.
fn request_body(service: &str, text: &str) -> Value {
    let parameters = json!({ "content": text });
    json!({"Service": service, "ServiceParameters": parameters.to_string()})
}

/// Extracts `host[:port]` from the endpoint for signing and the `host` header.
fn host_of(endpoint: &str) -> Result<String> {
    let without_scheme = endpoint
        .split_once("://")
        .map_or(endpoint, |(_, rest)| rest);
    let authority = without_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    if host.is_empty() {
        return Err(Error::validation(
            "aliyun green backend requires an endpoint with a host",
        ));
    }
    Ok(host.to_owned())
}

/// Splits `Data.Labels` into trimmed, non-empty labels, dropping `nonLabel`.
fn parse_labels(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|label| !label.is_empty() && !label.eq_ignore_ascii_case(NON_LABEL))
        .map(str::to_owned)
        .collect()
}

/// Maps one vendor confidence (`0..=100`) onto a severity band.
fn level_for_confidence(confidence: f64) -> RiskLevel {
    if confidence >= CRITICAL_CONFIDENCE {
        RiskLevel::Critical
    } else if confidence >= HIGH_CONFIDENCE {
        RiskLevel::High
    } else if confidence >= MEDIUM_CONFIDENCE {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    }
}

/// Rounds a confidence to four decimal places for `details`.
fn round_confidence(confidence: f64) -> f64 {
    (confidence * CONFIDENCE_SCALE).round() / CONFIDENCE_SCALE
}

/// Builds the transport error for a non-`200` reply, quoting only the code and
/// a truncated, whitespace-collapsed status message.
fn code_error(response: &Value) -> Error {
    let label = match response.get("Code") {
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::Null) | None => "missing".to_owned(),
        Some(_) => "invalid".to_owned(),
    };
    let mut message = format!("aliyun green returned code {label}");
    if let Some(status) = response.get("Message").and_then(Value::as_str) {
        let collapsed = status.split_whitespace().collect::<Vec<_>>().join(" ");
        if !collapsed.is_empty() {
            message.push_str(": ");
            message.extend(collapsed.chars().take(ERROR_MESSAGE_CHARS));
        }
    }
    tracing::warn!(
        target: LOG_TARGET,
        code = %label,
        "aliyun green returned an error code"
    );
    Error::transport(message)
}

/// Composes the RFC 2104 HMAC-SHA256 of `data` under `key`.
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut block = [0_u8; HMAC_BLOCK_SIZE];
    if key.len() > HMAC_BLOCK_SIZE {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|byte| byte ^ 0x36));
    inner.update(data);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(block.map(|byte| byte ^ 0x5c));
    outer.update(inner);
    let mut digest = [0_u8; 32];
    digest.copy_from_slice(&outer.finalize());
    digest
}

/// Lowercase-hex HMAC-SHA256 digest of `data` under `key`.
fn hmac_sha256_hex(key: &[u8], data: &[u8]) -> String {
    hex_encode(&hmac_sha256(key, data))
}

/// Lowercase-hex SHA-256 digest of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

/// Encodes `bytes` as a lowercase hexadecimal string.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// Builds the `ACS3-HMAC-SHA256` canonical request.
fn canonical_request(
    host: &str,
    api_version: &str,
    date: &str,
    nonce: &str,
    body_sha: &str,
) -> String {
    let headers = format!(
        "host:{host}\n\
         x-acs-action:{ACTION}\n\
         x-acs-content-sha256:{body_sha}\n\
         x-acs-date:{date}\n\
         x-acs-signature-nonce:{nonce}\n\
         x-acs-version:{api_version}\n"
    );
    format!("POST\n/\n{headers}\n{SIGNED_HEADER_NAMES}\n{body_sha}")
}

/// Builds the `Authorization` value for an `Aksk` credential.
///
/// The signature is deterministic in `(host, api_version, date, nonce,
/// body_sha)`, so the same request inputs always produce the same header.
fn sign_authorization(
    access_key_id: &str,
    access_key_secret: &str,
    host: &str,
    api_version: &str,
    date: &str,
    nonce: &str,
    body_sha: &str,
) -> String {
    let canonical = canonical_request(host, api_version, date, nonce, body_sha);
    let string_to_sign = format!(
        "{SIGNATURE_ALGORITHM}\n{}",
        sha256_hex(canonical.as_bytes())
    );
    let signature = hmac_sha256_hex(access_key_secret.as_bytes(), string_to_sign.as_bytes());
    format!(
        "{SIGNATURE_ALGORITHM} Credential={access_key_id},\
         SignedHeaders={SIGNED_HEADER_NAMES},Signature={signature}"
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::ScanStage;

    /// Distinctive text the tests assert never leaks into details or errors.
    const SCANNED_TEXT: &str = "sensitive sample text 42";
    const BEARER_TOKEN: &str = "green-bearer-token";
    const ACCESS_KEY_ID: &str = "test-access-key-id";
    const ACCESS_KEY_SECRET: &str = "test-access-key-secret";

    fn offline_backend() -> AliyunGreenBackend {
        AliyunGreenBackend::new(AliyunGreenConfig::default()).unwrap()
    }

    fn backend_for(server: &MockServer, auth: AliyunGreenAuth) -> AliyunGreenBackend {
        AliyunGreenBackend::new(
            AliyunGreenConfig::default()
                .with_endpoint(server.uri())
                .with_auth(auth)
                .with_max_retries(0),
        )
        .unwrap()
    }

    async fn server_with_reply(reply: &Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reply))
            .expect(1)
            .mount(&server)
            .await;
        server
    }

    fn header<'a>(request: &'a wiremock::Request, name: &str) -> Option<&'a str> {
        request.headers.get(name)?.to_str().ok()
    }

    fn recorded_reply(confidence: f64) -> Value {
        json!({
            "Code": 200, "Message": "OK", "RequestId": "req-aliyun-1",
            "Data": {
                "Labels": "political_content,profanity",
                "Reason": "vendor reason that must stay private",
                "Result": [{
                    "Label": "political_content",
                    "Confidence": confidence,
                    "RiskWords": "secret-risk-word"
                }]
            }
        })
    }

    fn ctx(text: &str) -> GuardrailContext {
        GuardrailContext::text(text, ScanStage::UserInput)
    }

    /// Asserts the headers and body hash every signed request carries.
    fn assert_signed_request(sent: &wiremock::Request, host: &str) {
        assert_eq!(sent.method.as_str(), "POST");
        assert_eq!(sent.url.path(), "/");
        assert_eq!(header(sent, "x-acs-action"), Some(ACTION));
        assert_eq!(header(sent, "x-acs-version"), Some(DEFAULT_API_VERSION));
        assert_eq!(header(sent, "host"), Some(host));
        let date = header(sent, "x-acs-date").expect("date");
        assert!(
            chrono::DateTime::parse_from_rfc3339(date).is_ok(),
            "date {date}"
        );
        let nonce = header(sent, "x-acs-signature-nonce").expect("nonce");
        assert!(uuid::Uuid::parse_str(nonce).is_ok(), "nonce {nonce}");
        let expected_sha = sha256_hex(&sent.body);
        assert_eq!(
            header(sent, "x-acs-content-sha256"),
            Some(expected_sha.as_str())
        );
        let body: Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(body["Service"], json!(DEFAULT_SERVICE));
        let parameters = body["ServiceParameters"].as_str().expect("encoded");
        assert_eq!(parameters, json!({"content": SCANNED_TEXT}).to_string());
        assert_eq!(
            serde_json::from_str::<Value>(parameters).unwrap()["content"],
            json!(SCANNED_TEXT)
        );
    }

    #[test]
    fn digest_vectors_match_the_rfcs() {
        let vectors: [(&[u8], &[u8], &str); 3] = [
            (
                &[0x0b_u8; 20],
                b"Hi There",
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
            ),
            (
                b"Jefe",
                b"what do ya want for nothing?",
                "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
            ),
            // A longer-than-block-size key exercises the hashed-key path.
            (
                &[0xaa_u8; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First",
                "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
            ),
        ];
        for (key, data, expected) in vectors {
            assert_eq!(hmac_sha256_hex(key, data), expected);
        }
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(hex_encode(&[0x00, 0x0f, 0xff]), "000fff");
    }

    #[test]
    fn canonical_request_and_signature_are_deterministic() {
        let canonical = canonical_request(
            "green.example",
            "2022-03-02",
            "2026-01-01T00:00:00Z",
            "nonce-1",
            "abc123",
        );
        assert_eq!(
            canonical,
            "POST\n/\nhost:green.example\nx-acs-action:TextModeration\n\
             x-acs-content-sha256:abc123\nx-acs-date:2026-01-01T00:00:00Z\n\
             x-acs-signature-nonce:nonce-1\nx-acs-version:2022-03-02\n\n\
             host;x-acs-action;x-acs-content-sha256;x-acs-date;x-acs-signature-nonce;x-acs-version\nabc123"
        );
        let sign = |host: &str| {
            sign_authorization(
                ACCESS_KEY_ID,
                ACCESS_KEY_SECRET,
                host,
                "2022-03-02",
                "2026-01-01T00:00:00Z",
                "nonce-1",
                "abc123",
            )
        };
        let first = sign("green.example");
        assert_eq!(first, sign("green.example"));
        assert_ne!(first, sign("other.example"));
        let string_to_sign = format!("ACS3-HMAC-SHA256\n{}", sha256_hex(canonical.as_bytes()));
        let expected = format!(
            "ACS3-HMAC-SHA256 Credential={ACCESS_KEY_ID},SignedHeaders={SIGNED_HEADER_NAMES},Signature={}",
            hmac_sha256_hex(ACCESS_KEY_SECRET.as_bytes(), string_to_sign.as_bytes())
        );
        assert_eq!(first, expected);
        assert!(!first.contains(ACCESS_KEY_SECRET));
    }

    #[test]
    fn config_defaults_serde_auth_and_debug() {
        let defaults = AliyunGreenConfig::default();
        assert_eq!(defaults.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(defaults.auth, AliyunGreenAuth::None);
        assert_eq!(defaults.auth.kind(), "none");
        assert_eq!(defaults.service, DEFAULT_SERVICE);
        assert_eq!(defaults.api_version, DEFAULT_API_VERSION);
        assert_eq!(defaults.region, "cn-shanghai");
        assert!((defaults.timeout - 30.0).abs() < f64::EPSILON);
        assert_eq!(defaults.max_retries, 2);
        assert_eq!(defaults.risk_type, DEFAULT_RISK_TYPE);

        let config = defaults
            .with_endpoint("https://green.example/")
            .with_auth(AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)))
            .with_service("text_moderation")
            .with_api_version("2023-01-01")
            .with_region("cn-hangzhou")
            .with_timeout(12.5)
            .with_max_retries(3);
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(serialized.contains("\"timeout\":12.5"), "{serialized}");
        assert!(serialized.contains("\"service\":\"text_moderation\""));
        let restored: AliyunGreenConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(restored, config);

        let partial: AliyunGreenConfig = serde_json::from_str(
            r#"{"endpoint": "https://green.example/", "vendor": {"name": "aliyun"}}"#,
        )
        .unwrap();
        assert_eq!(partial.endpoint, "https://green.example/");
        assert_eq!(partial.auth, AliyunGreenAuth::None);
        assert_eq!(partial.service, DEFAULT_SERVICE);
        assert_eq!(partial.max_retries, 2);

        // Blank service/version/region fall back to their defaults.
        let backend = AliyunGreenBackend::new(
            AliyunGreenConfig::default()
                .with_service("  ")
                .with_api_version("")
                .with_region(" "),
        )
        .unwrap();
        assert_eq!(backend.config().service, DEFAULT_SERVICE);
        assert_eq!(backend.config().api_version, DEFAULT_API_VERSION);
        assert_eq!(backend.config().region, DEFAULT_REGION);
        assert_eq!(backend.name(), NAME);

        // Every auth variant round-trips and never prints its secrets.
        for auth in [
            AliyunGreenAuth::None,
            AliyunGreenAuth::Bearer(SecretString::from("bearer-secret-value")),
            AliyunGreenAuth::Aksk {
                access_key_id: SecretString::from("ak-id-secret"),
                access_key_secret: SecretString::from("ak-secret-value"),
            },
        ] {
            let serialized = serde_json::to_string(&auth).unwrap();
            let restored: AliyunGreenAuth = serde_json::from_str(&serialized).unwrap();
            assert_eq!(restored, auth);
            let debug = format!("{auth:?}");
            assert!(!debug.contains("bearer-secret-value"), "{debug}");
            assert!(!debug.contains("ak-id-secret"), "{debug}");
            assert!(!debug.contains("ak-secret-value"), "{debug}");
        }
        assert_eq!(
            AliyunGreenAuth::Bearer(SecretString::from("x")).kind(),
            "bearer"
        );
        assert_eq!(
            AliyunGreenAuth::Aksk {
                access_key_id: SecretString::from("x"),
                access_key_secret: SecretString::from("y"),
            }
            .kind(),
            "aksk"
        );
    }

    #[test]
    fn debug_never_prints_credentials() {
        let config = AliyunGreenConfig::default()
            .with_auth(AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)));
        let debug = format!("{config:?}");
        assert!(
            debug.contains("green-cip.cn-shanghai.aliyuncs.com"),
            "{debug}"
        );
        assert!(!debug.contains(BEARER_TOKEN), "{debug}");
        let debug = format!("{:?}", AliyunGreenBackend::new(config).unwrap());
        assert!(debug.contains("AliyunGreenBackend"), "{debug}");
        assert!(!debug.contains(BEARER_TOKEN), "{debug}");

        let config = AliyunGreenConfig::default().with_auth(AliyunGreenAuth::Aksk {
            access_key_id: SecretString::from(ACCESS_KEY_ID),
            access_key_secret: SecretString::from(ACCESS_KEY_SECRET),
        });
        let debug = format!("{:?}", AliyunGreenBackend::new(config).unwrap());
        assert!(debug.contains("aksk"), "{debug}");
        assert!(!debug.contains(ACCESS_KEY_ID), "{debug}");
        assert!(!debug.contains(ACCESS_KEY_SECRET), "{debug}");
    }

    #[test]
    fn new_rejects_blank_endpoint_and_blank_aksk() {
        let error =
            AliyunGreenBackend::new(AliyunGreenConfig::default().with_endpoint("   ")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        for (access_key_id, access_key_secret) in [
            ("", ACCESS_KEY_SECRET),
            (ACCESS_KEY_ID, ""),
            ("  ", ACCESS_KEY_SECRET),
        ] {
            let error = AliyunGreenBackend::new(AliyunGreenConfig::default().with_auth(
                AliyunGreenAuth::Aksk {
                    access_key_id: SecretString::from(access_key_id),
                    access_key_secret: SecretString::from(access_key_secret),
                },
            ))
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
        }

        // A blank bearer token is allowed and omitted from the request.
        let backend = AliyunGreenBackend::new(
            AliyunGreenConfig::default().with_auth(AliyunGreenAuth::Bearer(SecretString::from(""))),
        )
        .unwrap();
        assert_eq!(backend.config().auth.kind(), "bearer");
    }

    #[tokio::test]
    async fn bearer_request_matches_the_rpc_contract() {
        let server = server_with_reply(&recorded_reply(95.0)).await;
        let backend = backend_for(
            &server,
            AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)),
        );
        let assessment = backend.analyze(&ctx(SCANNED_TEXT)).await.unwrap();
        assert!(assessment.has_risk);
        let details = assessment.details.expect("details");
        assert!(!details.to_string().contains(SCANNED_TEXT));
        assert!(!details.to_string().contains("secret-risk-word"));

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        assert_signed_request(&received[0], &server.address().to_string());
        assert_eq!(
            header(&received[0], "authorization"),
            Some(format!("Bearer {BEARER_TOKEN}").as_str())
        );
    }

    #[tokio::test]
    async fn aksk_request_carries_the_acs3_signature() {
        let server = server_with_reply(&recorded_reply(80.0)).await;
        let backend = backend_for(
            &server,
            AliyunGreenAuth::Aksk {
                access_key_id: SecretString::from(ACCESS_KEY_ID),
                access_key_secret: SecretString::from(ACCESS_KEY_SECRET),
            },
        );
        backend.analyze(&ctx(SCANNED_TEXT)).await.unwrap();

        let received = server.received_requests().await.unwrap();
        let sent = &received[0];
        let host = server.address().to_string();
        assert_signed_request(sent, &host);
        let authorization = header(sent, "authorization").expect("authorization");
        assert!(
            authorization.starts_with("ACS3-HMAC-SHA256 Credential="),
            "{authorization}"
        );
        assert!(
            authorization.contains(&format!(
                "Credential={ACCESS_KEY_ID},SignedHeaders={SIGNED_HEADER_NAMES},Signature="
            )),
            "{authorization}"
        );
        assert!(!authorization.contains(ACCESS_KEY_SECRET));
        let signature = authorization.split("Signature=").nth(1).expect("signature");
        assert_eq!(signature.len(), 64, "{authorization}");
        assert!(
            signature
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "{authorization}"
        );

        // The received header matches the deterministic helper for the same inputs.
        assert_eq!(
            authorization,
            sign_authorization(
                ACCESS_KEY_ID,
                ACCESS_KEY_SECRET,
                &host,
                DEFAULT_API_VERSION,
                header(sent, "x-acs-date").expect("date"),
                header(sent, "x-acs-signature-nonce").expect("nonce"),
                header(sent, "x-acs-content-sha256").expect("sha"),
            )
        );
    }

    #[tokio::test]
    async fn empty_text_skips_requests_and_status_errors_are_transport() {
        let server = MockServer::start().await;
        let backend = backend_for(
            &server,
            AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)),
        );
        for text in ["", "   ", "\n\t "] {
            let assessment = backend.analyze(&ctx(text)).await.unwrap();
            assert!(!assessment.has_risk, "text {text:?}");
            assert_eq!(assessment.risk_level, RiskLevel::Safe);
            assert!(assessment.risk_type.is_none());
            assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
        }
        assert!(server.received_requests().await.unwrap().is_empty());

        // A blank bearer token is omitted rather than sent empty.
        let server = server_with_reply(&recorded_reply(95.0)).await;
        let backend = backend_for(&server, AliyunGreenAuth::Bearer(SecretString::from("")));
        backend.analyze(&ctx(SCANNED_TEXT)).await.unwrap();
        let received = server.received_requests().await.unwrap();
        assert_eq!(header(&received[0], "authorization"), None);

        // A non-2xx status is a transport error that never quotes the text.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(
                ResponseTemplate::new(403)
                    .set_body_json(json!({"Code": 403, "Message": "forbidden"})),
            )
            .mount(&server)
            .await;
        let error = backend_for(
            &server,
            AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)),
        )
        .analyze(&ctx(SCANNED_TEXT))
        .await
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(!error.to_string().contains(SCANNED_TEXT), "{error}");
    }

    #[test]
    fn parse_response_levels_details_and_vendor_content() {
        for (confidence, expected) in [
            (95.0, RiskLevel::Critical),
            (90.0, RiskLevel::Critical),
            (89.9, RiskLevel::High),
            (80.0, RiskLevel::High),
            (75.0, RiskLevel::High),
            (60.0, RiskLevel::Medium),
            (50.0, RiskLevel::Medium),
            (49.9, RiskLevel::Low),
            (20.0, RiskLevel::Low),
        ] {
            let assessment = offline_backend()
                .parse_response(&recorded_reply(confidence))
                .unwrap();
            assert!(assessment.has_risk);
            assert_eq!(assessment.risk_level, expected, "confidence {confidence}");
            assert_eq!(assessment.risk_type.as_deref(), Some("political_content"));
            let expected_reported = round_confidence(confidence / 100.0);
            assert!((assessment.confidence - expected_reported).abs() < f64::EPSILON);
        }

        let assessment = offline_backend()
            .parse_response(&recorded_reply(95.555_5))
            .unwrap();
        let details = assessment.details.expect("details");
        assert_eq!(details["provider"], json!("aliyun"));
        assert_eq!(details["service"], json!(DEFAULT_SERVICE));
        assert_eq!(details["region"], json!(DEFAULT_REGION));
        assert_eq!(details["labels"], json!(["political_content", "profanity"]));
        assert_eq!(details["request_id"], json!("req-aliyun-1"));
        let confidence = details["confidence"].as_f64().expect("confidence");
        assert!((confidence - 0.9556).abs() < 1e-9, "{confidence}");
        let serialized = details.to_string();
        assert!(!serialized.contains("secret-risk-word"), "{serialized}");
        assert!(!serialized.contains("vendor reason"), "{serialized}");
        assert!(!serialized.contains("RiskWords"), "{serialized}");
        assert!(!serialized.contains("Reason"), "{serialized}");

        // The highest confidence wins across results; labels without scores are high.
        let reply = json!({"Code": 200, "Data": {"Labels": "profanity", "Result": [
            {"Confidence": 20.0}, {"Confidence": 88.5}, {"Confidence": 60.0}
        ]}});
        let assessment = offline_backend().parse_response(&reply).unwrap();
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("profanity"));
        assert!((assessment.confidence - 0.885).abs() < f64::EPSILON);

        let assessment = offline_backend()
            .parse_response(&json!({"Code": 200, "Data": {"Labels": "political_content"}}))
            .unwrap();
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
        let details = assessment.details.expect("details");
        assert_eq!(details["confidence"].as_f64(), Some(1.0));
        assert!(details.get("request_id").is_none());
    }

    #[test]
    fn parse_response_safe_and_code_errors() {
        for labels in [
            json!(""),
            json!("   "),
            json!("nonLabel"),
            json!(" nonLabel ,  "),
            json!(",,"),
            json!("NONLABEL"),
        ] {
            let reply = json!({"Code": 200, "Data": {"Labels": labels, "Result": []}});
            let assessment = offline_backend().parse_response(&reply).unwrap();
            assert!(!assessment.has_risk, "labels {labels:?}");
            assert_eq!(assessment.risk_level, RiskLevel::Safe);
            assert!(assessment.risk_type.is_none());
            assert!(assessment.details.is_none());
        }
        for reply in [
            json!({"Code": 200, "Data": {"Result": []}}),
            json!({"Code": 200, "Data": {}}),
            json!({"Code": 200}),
        ] {
            let assessment = offline_backend().parse_response(&reply).unwrap();
            assert!(!assessment.has_risk, "reply {reply}");
            assert_eq!(assessment.risk_level, RiskLevel::Safe);
        }

        for reply in [
            json!({"Code": 500, "Message": "internal error"}),
            json!({"Message": "missing code"}),
            json!({"Code": "200"}),
            json!({"Code": 500.5}),
        ] {
            let error = offline_backend().parse_response(&reply).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Transport, "reply {reply}");
        }
        let error = offline_backend()
            .parse_response(&json!({"Code": 500, "Message": "bad\nrequest"}))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("aliyun green returned code 500: bad request"),
            "{error}"
        );
    }
}
