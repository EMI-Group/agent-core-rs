//! Aliyun Green (内容安全) text-moderation adapter.
//!
//! [`AliyunGreenBackend`] speaks the Aliyun Green `TextModeration` RPC
//! (`green-20220302`) over the crate-shared [`crate::backends::http`] helpers:
//! it posts `{"Service": ..., "ServiceParameters": "{\"content\": ...}"}` and
//! maps the label union of `Data.Labels` and `Data.Result[].Label` plus the
//! per-label `Confidence` scores onto a [`crate::RiskAssessment`]. The label
//! list is the union of both sources (the RPC can omit `Labels` while
//! returning the per-label detail list), any label or a positive `Confidence`
//! means risk, the first label is `risk_type`, and the vendor's `RiskWords`,
//! `Reason`, and the scanned text never appear in `details`, logs, or errors;
//! empty text is safe without a request. Requests use either a gateway `Bearer` token or the native
//! `ACS3-HMAC-SHA256` signature over six headers, so the public endpoint
//! accepts them without an SDK. A reply that does not carry the
//! `TextModeration` envelope is a validation error rather than a safe verdict.
//!
//! The signature machinery lives in the private `signing` submodule and the
//! suite exercising the adapter lives in `tests`.
#![allow(clippy::module_name_repetitions)]

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};
use openjiuwen_common::security::SecretString;

use crate::backends::http::{http_client, json_body, resolve_timeout};
use crate::backends::safe_assessment;
use crate::guardrail::GuardrailBackend;
use crate::{GuardrailContext, RiskAssessment, RiskLevel};

use self::signing::{sha256_hex, sign_authorization};

mod signing;

#[cfg(test)]
mod tests;

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
/// The label the vendor reports for benign content.
const NON_LABEL: &str = "nonLabel";
/// The envelope `Code` a successful moderation call reports.
const SUCCESS_CODE: i64 = 200;
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
    /// vendor's first label is used whenever a label is present, and a positive
    /// confidence reported without any label falls back to this value.
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
    /// A reply without the RPC envelope — no integer `Code`, or no `Data`
    /// object carrying `Labels` or `Result` — is a validation failure rather
    /// than a safe verdict, while an integer `Code` other than `200` is a
    /// transport failure. The labels are the trimmed, de-duplicated union of
    /// the comma-separated `Data.Labels` (first) and the `Data.Result[].Label`
    /// fields (`nonLabel` dropped from both), because the RPC may omit
    /// `Labels` and report only the per-label detail list. The reply is safe
    /// only when that union is empty and no `Result` entry reports a positive
    /// `Confidence`; otherwise
    /// the highest `Result[].Confidence` (`0..=100`) picks the severity (`>= 90`
    /// [`RiskLevel::Critical`], `>= 75` [`RiskLevel::High`], `>= 50`
    /// [`RiskLevel::Medium`], else [`RiskLevel::Low`]; labels without a usable
    /// score report [`RiskLevel::High`]). The first union label is `risk_type`,
    /// falling back to `config.risk_type` when only a confidence was reported,
    /// and the rounded confidence (`0..=1`) is recorded. `details` carries the
    /// provider, service, region, union labels, confidence, and the request id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the envelope is unrecognized and
    /// [`Error::Transport`] when `Code` is an integer other than `200`; neither
    /// message repeats any part of the payload.
    pub fn parse_response(&self, response: &Value) -> Result<RiskAssessment> {
        let Some(code) = response.get("Code").and_then(Value::as_i64) else {
            return Err(unrecognized_reply());
        };
        if code != SUCCESS_CODE {
            return Err(code_error(code, response));
        }
        let Some(data) = response
            .get("Data")
            .filter(|data| data.is_object())
            .filter(|data| data.get("Labels").is_some() || data.get("Result").is_some())
        else {
            return Err(unrecognized_reply());
        };
        let results = data
            .get("Result")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let labels = response_labels(Some(data), results);
        let max_confidence = results
            .iter()
            .filter_map(|entry| entry.get("Confidence").and_then(Value::as_f64))
            .filter(|confidence| confidence.is_finite())
            .reduce(f64::max);
        if labels.is_empty() && !max_confidence.is_some_and(|max| max > 0.0) {
            return Ok(safe_assessment());
        }
        let risk_type = labels
            .first()
            .cloned()
            .unwrap_or_else(|| self.config.risk_type.clone());
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
    /// [`Error::Validation`] when the payload cannot be scanned, the endpoint
    /// carries no host, or the reply does not carry the `TextModeration`
    /// envelope, [`Error::Transport`] for connection, status, business-code,
    /// and decoding failures, and [`Error::Timeout`] when the deadline expires.
    /// Errors never carry the scanned text.
    async fn analyze(&self, ctx: &GuardrailContext) -> Result<RiskAssessment> {
        // The checked projection, like every local detector: an unscannable
        // payload is a failure for the failure policy to resolve, never a pass.
        let text = ctx.text_view_checked()?;
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

/// Unions the `Data.Labels` list with the `Data.Result[].Label` fields.
///
/// The RPC may report only the per-label detail list, so both sources count:
/// `Labels` entries come first, then the `Result`-only labels, trimmed, blanks
/// and `nonLabel` dropped, duplicates removed in first-seen order.
fn response_labels(data: Option<&Value>, results: &[Value]) -> Vec<String> {
    let mut labels = data
        .and_then(|data| data.get("Labels"))
        .and_then(Value::as_str)
        .map(parse_labels)
        .unwrap_or_default();
    for label in results
        .iter()
        .filter_map(|entry| entry.get("Label").and_then(Value::as_str))
        .flat_map(parse_labels)
    {
        if !labels.contains(&label) {
            labels.push(label);
        }
    }
    labels
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

/// Builds the error reported when a `2xx` body does not carry the
/// `TextModeration` envelope.
///
/// Naming the fields keeps the mismatch actionable without repeating any part
/// of the payload, and the validation kind makes the pipeline block under the
/// fail-closed default instead of treating the reply as a clean verdict.
fn unrecognized_reply() -> Error {
    Error::validation(
        "aliyun green reply does not match the TextModeration envelope: expected an integer `Code` \
         and a `Data` object carrying `Labels` or `Result`",
    )
}

/// Builds the transport error for a non-`200` reply, quoting only the code and
/// a truncated, whitespace-collapsed status message.
fn code_error(code: i64, response: &Value) -> Error {
    let mut message = format!("aliyun green returned code {code}");
    if let Some(status) = response.get("Message").and_then(Value::as_str) {
        let collapsed = status.split_whitespace().collect::<Vec<_>>().join(" ");
        if !collapsed.is_empty() {
            message.push_str(": ");
            message.extend(collapsed.chars().take(ERROR_MESSAGE_CHARS));
        }
    }
    tracing::warn!(
        target: LOG_TARGET,
        code,
        "aliyun green returned an error code"
    );
    Error::transport(message)
}
