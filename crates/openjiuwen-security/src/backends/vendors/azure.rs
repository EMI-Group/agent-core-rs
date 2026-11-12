//! Azure AI Content Safety adapter (`POST /contentsafety/text:analyze`).
//!
//! [`AzureContentSafetyBackend`] posts the scanned text to the text-analysis
//! operation and turns the per-category `severity` output into a
//! [`RiskAssessment`]: `0` is safe, `1..=2` [`RiskLevel::Low`], `3..=4`
//! [`RiskLevel::Medium`], `5..=6` [`RiskLevel::High`], and `7` or more
//! [`RiskLevel::Critical`]. `risk_type` is the highest category, `confidence` is
//! the maximum severity divided by `7`, and `details` holds the category map,
//! the maximum severity, and the blocklist hit count.
//!
//! Requests carry the resource key in `Ocp-Apim-Subscription-Key` (never a
//! bearer token) and the configured categories, and empty text is safe without
//! an `HTTP` call. Scanned text and blocklist terms never appear in `details`,
//! logs, or errors; the tests drive recorded `text:analyze` fixtures through a
//! loopback server, so the suite stays offline.
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;
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

/// `api-version` sent when the configuration does not name one.
pub const DEFAULT_API_VERSION: &str = "2024-09-01";

/// Categories analysed when the configuration does not name any.
pub const DEFAULT_CATEGORIES: [&str; 4] = ["Hate", "SelfHarm", "Sexual", "Violence"];

/// Stable backend name used in findings and logs.
pub const NAME: &str = "azure_content_safety";

/// Operation path appended to a resource endpoint.
const ANALYZE_OPERATION: &str = "contentsafety/text:analyze";

/// Header carrying the Azure resource key.
const SUBSCRIPTION_KEY_HEADER: &str = "Ocp-Apim-Subscription-Key";

/// Default request timeout in seconds.
const DEFAULT_TIMEOUT_SECS: f64 = 30.0;

/// Number of retries performed after the first attempt.
const DEFAULT_MAX_RETRIES: u32 = 2;

/// First retry backoff handed to the shared client.
const RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(200);

/// Highest severity Azure reports (eight-level output); larger values clamp.
const MAX_SEVERITY: u32 = 7;

/// Scale used to round `confidence` to four decimal places.
const CONFIDENCE_SCALE: f64 = 10_000.0;

/// Log target shared with the other backends.
const LOG_TARGET: &str = "openjiuwen_security::backends";

/// Configuration for [`AzureContentSafetyBackend`].
///
/// Field names are `snake_case`, unknown JSON fields are tolerated, and a
/// missing field falls back to its default, so a partial configuration
/// document still loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct AzureContentSafetyConfig {
    /// Resource endpoint, for example `https://res.cognitiveservices.azure.com`.
    pub endpoint: String,
    /// Resource key sent as `Ocp-Apim-Subscription-Key`; redacted by
    /// [`SecretString`].
    pub api_key: Option<SecretString>,
    /// `api-version` query value; blank means [`DEFAULT_API_VERSION`].
    pub api_version: String,
    /// Categories to analyse; empty means [`DEFAULT_CATEGORIES`].
    pub categories: Vec<String>,
    /// Request timeout in seconds.
    pub timeout: f64,
    /// Number of retries performed after the first attempt.
    pub max_retries: u32,
    /// Risk name used when the detection carries no category name.
    pub risk_type: String,
}

impl Default for AzureContentSafetyConfig {
    /// The documented defaults: no endpoint, no key, the current API version,
    /// the four default categories, a 30 s timeout, two retries, and
    /// [`NAME`] as the fallback risk type.
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            api_key: None,
            api_version: DEFAULT_API_VERSION.to_owned(),
            categories: DEFAULT_CATEGORIES
                .iter()
                .map(|category| (*category).to_owned())
                .collect(),
            timeout: DEFAULT_TIMEOUT_SECS,
            max_retries: DEFAULT_MAX_RETRIES,
            risk_type: NAME.to_owned(),
        }
    }
}

impl AzureContentSafetyConfig {
    /// Sets the resource endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sets the resource key sent in `Ocp-Apim-Subscription-Key`.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<SecretString>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the `api-version`; blank keeps [`DEFAULT_API_VERSION`] in effect.
    #[must_use]
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
        self
    }

    /// Replaces the category list.
    #[must_use]
    pub fn with_categories(mut self, categories: Vec<String>) -> Self {
        self.categories = categories;
        self
    }

    /// Sets the request timeout in seconds; unusable values fall back to 30 s.
    #[must_use]
    pub fn with_timeout(mut self, timeout: f64) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the number of retries performed after the first attempt.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}

/// Azure AI Content Safety text-analysis backend.
///
/// The backend owns the endpoint composition, the request shape, and the
/// response parsing; the shared [`HttpClient`] only performs the exchange.
/// [`AzureContentSafetyBackend::parse_response`] and
/// [`AzureContentSafetyBackend::analyze_url`] are pure, so the wire contract is
/// testable without a server.
pub struct AzureContentSafetyBackend {
    config: AzureContentSafetyConfig,
    client: HttpClient,
}

impl AzureContentSafetyBackend {
    /// Builds a backend from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `config.endpoint` is blank or only
    /// whitespace, and [`Error::Config`] when the shared client cannot be
    /// built with the configured timeout and retry settings.
    pub fn new(config: AzureContentSafetyConfig) -> Result<Self> {
        if config.endpoint.trim().is_empty() {
            return Err(Error::validation(
                "azure content safety backend requires an endpoint URL",
            ));
        }
        let client = http_client(
            resolve_timeout(config.timeout),
            config.max_retries,
            RETRY_INITIAL_BACKOFF,
        )?;
        Ok(Self { config, client })
    }

    /// Uses `client` for every request instead of the constructed one.
    #[must_use]
    pub fn with_client(mut self, client: HttpClient) -> Self {
        self.client = client;
        self
    }

    /// The configuration the backend was built with.
    #[must_use]
    pub fn config(&self) -> &AzureContentSafetyConfig {
        &self.config
    }

    /// Composes the `text:analyze` URL from the configured endpoint.
    ///
    /// A trailing `/` is trimmed and the operation is appended; when the
    /// endpoint already contains `contentsafety/text:analyze` it is used as
    /// given, and `api-version` is appended only when the endpoint carries no
    /// query string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the configured endpoint is blank.
    pub fn analyze_url(&self) -> Result<String> {
        let endpoint = self.config.endpoint.trim().trim_end_matches('/');
        if endpoint.is_empty() {
            return Err(Error::validation(
                "azure content safety backend requires an endpoint URL",
            ));
        }
        let version = self.effective_api_version();
        Ok(if endpoint.contains(ANALYZE_OPERATION) {
            if endpoint.contains('?') {
                endpoint.to_owned()
            } else {
                format!("{endpoint}?api-version={version}")
            }
        } else {
            format!("{endpoint}/{ANALYZE_OPERATION}?api-version={version}")
        })
    }

    /// Converts a `text:analyze` response body into a [`RiskAssessment`].
    ///
    /// The category with the highest severity names the risk; ties keep the
    /// first category in wire order. When no detected risk carries a category
    /// name, the configured [`AzureContentSafetyConfig::risk_type`] is used.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when `categoriesAnalysis` is missing, empty,
    /// or not an array.
    pub fn parse_response(&self, response: &Value) -> Result<RiskAssessment> {
        let analysis = response
            .get("categoriesAnalysis")
            .and_then(Value::as_array)
            .filter(|entries| !entries.is_empty())
            .ok_or_else(no_category_analysis)?;

        let mut category_severities = BTreeMap::new();
        let mut max_severity = 0_u32;
        let mut max_category: Option<String> = None;
        for entry in analysis {
            let category = entry
                .get("category")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or_default();
            let severity = severity_of(entry.get("severity"));
            if !category.is_empty() {
                category_severities.insert(category.to_owned(), severity);
            }
            if severity > max_severity {
                max_severity = severity;
                max_category = (!category.is_empty()).then(|| category.to_owned());
            }
        }

        if max_severity == 0 {
            return Ok(safe_assessment());
        }

        let blocklist_matches = response
            .get("blocklistsMatch")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        Ok(RiskAssessment {
            has_risk: true,
            risk_level: risk_level_for(max_severity),
            risk_type: Some(self.risk_type_for(max_category.as_deref())),
            confidence: round_confidence(f64::from(max_severity) / f64::from(MAX_SEVERITY)),
            details: Some(json!({
                "provider": "azure",
                "categories": category_severities,
                "max_severity": max_severity,
                "blocklist_matches": blocklist_matches,
            })),
        })
    }

    /// The configured `api-version`, or [`DEFAULT_API_VERSION`] when blank.
    fn effective_api_version(&self) -> &str {
        let version = self.config.api_version.trim();
        if version.is_empty() {
            DEFAULT_API_VERSION
        } else {
            version
        }
    }

    /// The configured categories without blank entries, or
    /// [`DEFAULT_CATEGORIES`] when none remain.
    fn effective_categories(&self) -> Vec<&str> {
        let configured: Vec<&str> = self
            .config
            .categories
            .iter()
            .map(String::as_str)
            .map(str::trim)
            .filter(|category| !category.is_empty())
            .collect();
        if configured.is_empty() {
            DEFAULT_CATEGORIES.to_vec()
        } else {
            configured
        }
    }

    /// The risk name for a detected risk, falling back to the configuration.
    fn risk_type_for(&self, category: Option<&str>) -> String {
        if let Some(name) = category {
            let name = name.trim();
            if !name.is_empty() {
                return name.to_owned();
            }
        }
        let fallback = self.config.risk_type.trim();
        if fallback.is_empty() {
            NAME.to_owned()
        } else {
            fallback.to_owned()
        }
    }
}

impl fmt::Debug for AzureContentSafetyBackend {
    /// Prints the configuration, whose [`SecretString`] key stays redacted.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureContentSafetyBackend")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl GuardrailBackend for AzureContentSafetyBackend {
    fn name(&self) -> &str {
        NAME
    }

    async fn analyze(&self, ctx: &GuardrailContext) -> Result<RiskAssessment> {
        let text = ctx.text_view();
        if text.trim().is_empty() {
            return Ok(safe_assessment());
        }

        let body = json!({
            "text": text.as_ref(),
            "categories": self.effective_categories(),
        });
        let mut request = HttpRequest::post(self.analyze_url()?)
            .timeout(resolve_timeout(self.config.timeout))
            .json(&body)?;
        if let Some(key) = self.config.api_key.as_ref().filter(|key| !key.is_empty()) {
            request = request.header(SUBSCRIPTION_KEY_HEADER, key.expose_secret());
        }

        let response = self.client.execute(request).await?;
        let status = response.status;
        let payload = json_body(response, NAME)?;
        tracing::debug!(
            target: LOG_TARGET,
            backend = NAME,
            status,
            "azure content safety request completed"
        );

        if let Some(error) = payload.get("error").filter(|value| !value.is_null()) {
            let code = error
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or("unknown_error");
            tracing::warn!(
                target: LOG_TARGET,
                backend = NAME,
                code,
                "azure content safety reported an error"
            );
            return Err(Error::transport(format!(
                "azure content safety returned an error: {code}"
            )));
        }

        self.parse_response(&payload)
    }
}

/// Builds the error reported when a response carries no usable analysis.
fn no_category_analysis() -> Error {
    Error::transport("azure content safety returned no category analysis")
}

/// Maps a clamped severity to its risk level.
const fn risk_level_for(severity: u32) -> RiskLevel {
    match severity {
        0 => RiskLevel::Safe,
        1..=2 => RiskLevel::Low,
        3..=4 => RiskLevel::Medium,
        5..=6 => RiskLevel::High,
        _ => RiskLevel::Critical,
    }
}

/// Reads one `severity` value, clamping outliers into `0..=MAX_SEVERITY`.
///
/// Azure sends integers; fractional values are truncated toward zero so a
/// tolerant parser still produces a band, while negative and non-numeric values
/// count as `0` (safe).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn severity_of(value: Option<&Value>) -> u32 {
    let Some(value) = value else {
        return 0;
    };
    if let Some(severity) = value.as_i64() {
        return u32::try_from(severity.clamp(0, i64::from(MAX_SEVERITY))).unwrap_or(0);
    }
    let Some(severity) = value.as_f64() else {
        return 0;
    };
    if !severity.is_finite() || severity <= 0.0 {
        return 0;
    }
    if severity >= f64::from(MAX_SEVERITY) {
        return MAX_SEVERITY;
    }
    // `severity` is finite and in `(0, MAX_SEVERITY)`, so the cast is exact.
    severity.trunc() as u32
}

/// Rounds `confidence` to four decimal places.
fn round_confidence(value: f64) -> f64 {
    (value * CONFIDENCE_SCALE).round() / CONFIDENCE_SCALE
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_common::http::HttpClientConfig;

    use super::*;
    use crate::ScanStage;

    /// Text scanned by the end-to-end tests; no assertion may leak it.
    const SCANNED_TEXT: &str = "please analyse this prompt";

    /// Recorded `text:analyze` response with a `Hate` hit at severity 6.
    fn recorded_analysis() -> Value {
        analysis(&[("Hate", 6), ("SelfHarm", 0), ("Sexual", 0), ("Violence", 2)])
    }

    /// Builds a `text:analyze` response body from `(category, severity)` pairs.
    fn analysis(entries: &[(&str, i64)]) -> Value {
        json!({
            "blocklistsMatch": [],
            "categoriesAnalysis": entries
                .iter()
                .map(|(category, severity)| json!({"category": category, "severity": severity}))
                .collect::<Vec<_>>(),
        })
    }

    /// Configuration pointing at `server`, with retries disabled.
    fn config_for(server: &MockServer) -> AzureContentSafetyConfig {
        AzureContentSafetyConfig::default()
            .with_endpoint(server.uri())
            .with_api_key(SecretString::from("azure-subscription-key"))
            .with_max_retries(0)
    }

    /// A backend that needs no server for the pure parsing tests.
    fn offline_backend() -> AzureContentSafetyBackend {
        AzureContentSafetyBackend::new(
            AzureContentSafetyConfig::default()
                .with_endpoint("https://res.cognitiveservices.azure.com"),
        )
        .unwrap()
    }

    /// Mounts a single-shot `POST /contentsafety/text:analyze` mock.
    async fn server_with(response: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .and(query_param("api-version", DEFAULT_API_VERSION))
            .respond_with(response)
            .expect(1)
            .mount(&server)
            .await;
        server
    }

    fn header_value<'a>(request: &'a wiremock::Request, name: &str) -> Option<&'a str> {
        request.headers.get(name)?.to_str().ok()
    }

    fn body_json(request: &wiremock::Request) -> Value {
        serde_json::from_slice(&request.body).unwrap()
    }

    #[test]
    fn config_defaults_match_the_contract() {
        let config = AzureContentSafetyConfig::default();
        assert_eq!(config.endpoint, "");
        assert_eq!(config.api_key, None);
        assert_eq!(config.api_version, DEFAULT_API_VERSION);
        assert_eq!(
            config.categories,
            vec![
                "Hate".to_owned(),
                "SelfHarm".to_owned(),
                "Sexual".to_owned(),
                "Violence".to_owned(),
            ]
        );
        assert!((config.timeout - 30.0).abs() < f64::EPSILON);
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.risk_type, NAME);
        assert_eq!(DEFAULT_API_VERSION, "2024-09-01");
        assert_eq!(DEFAULT_CATEGORIES.len(), 4);
    }

    #[test]
    fn config_round_trips_and_tolerates_unknown_fields() {
        let config = AzureContentSafetyConfig::default()
            .with_endpoint("https://res.cognitiveservices.azure.com")
            .with_api_key(SecretString::from("azure-subscription-key"))
            .with_api_version("2023-10-01")
            .with_categories(vec!["Hate".to_owned(), "Violence".to_owned()])
            .with_timeout(12.5)
            .with_max_retries(3);

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"timeout\":12.5"), "{json}");
        assert!(json.contains("\"2023-10-01\""), "{json}");
        let restored: AzureContentSafetyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, config);

        let partial: AzureContentSafetyConfig = serde_json::from_value(json!({
            "endpoint": "https://res.cognitiveservices.azure.com",
            "future_field": {"nested": true},
        }))
        .unwrap();
        assert_eq!(partial.api_version, DEFAULT_API_VERSION);
        assert_eq!(
            partial.categories,
            AzureContentSafetyConfig::default().categories
        );
        assert_eq!(partial.max_retries, 2);
        // The key is plaintext in the serialized document so config round-trips.
        assert_eq!(
            partial.api_key.map(|key| key.expose_secret().to_owned()),
            None
        );
    }

    #[test]
    fn new_rejects_a_blank_endpoint() {
        for endpoint in ["", "   ", "\t\n"] {
            let error = AzureContentSafetyBackend::new(
                AzureContentSafetyConfig::default().with_endpoint(endpoint),
            )
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "endpoint {endpoint:?}");
            assert!(error.to_string().contains("endpoint"), "{error}");
        }
    }

    #[test]
    fn analyze_url_composes_the_operation() {
        let backend = |endpoint: &str, version: &str| {
            AzureContentSafetyBackend::new(
                AzureContentSafetyConfig::default()
                    .with_endpoint(endpoint)
                    .with_api_version(version),
            )
            .unwrap()
        };
        let base = "https://res.cognitiveservices.azure.com";

        for endpoint in [
            base.to_owned(),
            format!("{base}/"),
            format!("{base}///"),
            format!("  {base}/  "),
        ] {
            assert_eq!(
                backend(&endpoint, "").analyze_url().unwrap(),
                format!("{base}/contentsafety/text:analyze?api-version={DEFAULT_API_VERSION}"),
                "endpoint {endpoint:?}"
            );
        }

        assert_eq!(
            backend(base, "2023-10-01").analyze_url().unwrap(),
            format!("{base}/contentsafety/text:analyze?api-version=2023-10-01")
        );
        assert_eq!(
            backend(&format!("{base}/contentsafety/text:analyze"), "")
                .analyze_url()
                .unwrap(),
            format!("{base}/contentsafety/text:analyze?api-version={DEFAULT_API_VERSION}")
        );
        assert_eq!(
            backend(&format!("{base}/contentsafety/text:analyze/"), "")
                .analyze_url()
                .unwrap(),
            format!("{base}/contentsafety/text:analyze?api-version={DEFAULT_API_VERSION}")
        );
        assert_eq!(
            backend(
                &format!("{base}/contentsafety/text:analyze?api-version=2022-01-01"),
                "",
            )
            .analyze_url()
            .unwrap(),
            format!("{base}/contentsafety/text:analyze?api-version=2022-01-01"),
            "a full operation URL is used verbatim"
        );
    }

    #[tokio::test]
    async fn posts_the_text_with_the_subscription_key() {
        let server =
            server_with(ResponseTemplate::new(200).set_body_json(recorded_analysis())).await;
        let backend = AzureContentSafetyBackend::new(config_for(&server))
            .unwrap()
            .with_client(HttpClient::new(HttpClientConfig::default()).unwrap());
        assert_eq!(backend.name(), NAME);
        assert_eq!(backend.config().endpoint, server.uri());

        let ctx = GuardrailContext::text(SCANNED_TEXT, ScanStage::UserInput);
        let assessment = backend.analyze(&ctx).await.unwrap();

        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some("Hate"));
        assert!((assessment.confidence - 0.8571).abs() < 1e-9);
        let details = assessment.details.expect("details");
        assert_eq!(details["provider"], json!("azure"));
        assert_eq!(details["max_severity"], json!(6));
        assert_eq!(details["blocklist_matches"], json!(0));
        assert_eq!(details["categories"]["Hate"], json!(6));
        assert_eq!(details["categories"]["Violence"], json!(2));
        assert!(!details.to_string().contains(SCANNED_TEXT));

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let sent = &received[0];
        assert_eq!(sent.method.as_str(), "POST");
        assert_eq!(sent.url.path(), "/contentsafety/text:analyze");
        assert_eq!(sent.url.query(), Some("api-version=2024-09-01"));
        assert_eq!(
            header_value(sent, SUBSCRIPTION_KEY_HEADER),
            Some("azure-subscription-key")
        );
        assert_eq!(header_value(sent, "authorization"), None);
        assert_eq!(header_value(sent, "content-type"), Some("application/json"));
        assert_eq!(
            body_json(sent),
            json!({
                "text": SCANNED_TEXT,
                "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
            })
        );
    }

    #[tokio::test]
    async fn omits_the_key_header_without_a_credential_and_honours_categories() {
        let server =
            server_with(ResponseTemplate::new(200).set_body_json(recorded_analysis())).await;
        let config = AzureContentSafetyConfig::default()
            .with_endpoint(server.uri())
            .with_api_key(SecretString::from(""))
            .with_categories(vec!["Hate".to_owned(), " ".to_owned()])
            .with_max_retries(0);
        let backend = AzureContentSafetyBackend::new(config).unwrap();

        backend
            .analyze(&GuardrailContext::text(SCANNED_TEXT, ScanStage::UserInput))
            .await
            .unwrap();

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(header_value(&received[0], SUBSCRIPTION_KEY_HEADER), None);
        assert_eq!(header_value(&received[0], "authorization"), None);
        assert_eq!(body_json(&received[0])["categories"], json!(["Hate"]));
    }

    #[test]
    fn severity_bands_map_to_risk_levels() {
        let backend = offline_backend();
        for (severity, expected) in [
            (0, RiskLevel::Safe),
            (1, RiskLevel::Low),
            (2, RiskLevel::Low),
            (3, RiskLevel::Medium),
            (4, RiskLevel::Medium),
            (5, RiskLevel::High),
            (6, RiskLevel::High),
            (7, RiskLevel::Critical),
            (9, RiskLevel::Critical),
        ] {
            let assessment = backend
                .parse_response(&analysis(&[("Hate", severity)]))
                .unwrap();
            assert_eq!(assessment.risk_level, expected, "severity {severity}");
            assert_eq!(assessment.has_risk, severity > 0, "severity {severity}");
        }

        let assessment = backend.parse_response(&analysis(&[("Hate", 9)])).unwrap();
        assert_eq!(assessment.details.unwrap()["max_severity"], json!(7));
        assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn safe_when_every_severity_is_zero() {
        let assessment = offline_backend()
            .parse_response(&analysis(&[
                ("Hate", 0),
                ("SelfHarm", 0),
                ("Sexual", 0),
                ("Violence", 0),
            ]))
            .unwrap();
        assert!(!assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
        assert_eq!(assessment.risk_type, None);
        assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
        assert_eq!(assessment.details, None);
    }

    #[test]
    fn ties_keep_the_first_category_and_floats_truncate() {
        let backend = offline_backend();
        let tie = backend
            .parse_response(&analysis(&[("Hate", 4), ("Violence", 4)]))
            .unwrap();
        assert_eq!(tie.risk_type.as_deref(), Some("Hate"));

        let fractional = json!({
            "categoriesAnalysis": [
                {"category": "Violence", "severity": 2.9},
                {"category": "Hate", "severity": -3},
            ],
        });
        let assessment = backend.parse_response(&fractional).unwrap();
        assert_eq!(assessment.risk_level, RiskLevel::Low);
        assert_eq!(assessment.risk_type.as_deref(), Some("Violence"));
        let details = assessment.details.unwrap();
        assert_eq!(details["max_severity"], json!(2));
        assert_eq!(details["categories"]["Violence"], json!(2));
        assert_eq!(details["categories"]["Hate"], json!(0));
    }

    #[test]
    fn an_unnamed_detection_falls_back_to_the_configured_risk_type() {
        let response = json!({"categoriesAnalysis": [{"severity": 5}]});
        let assessment = offline_backend().parse_response(&response).unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some(NAME));

        let custom = AzureContentSafetyBackend::new(AzureContentSafetyConfig {
            risk_type: "custom_moderation".to_owned(),
            ..AzureContentSafetyConfig::default()
                .with_endpoint("https://res.cognitiveservices.azure.com")
        })
        .unwrap();
        let assessment = custom.parse_response(&response).unwrap();
        assert_eq!(assessment.risk_type.as_deref(), Some("custom_moderation"));
    }

    #[test]
    fn blocklist_matches_are_counted_without_terms() {
        let response = json!({
            "blocklistsMatch": [
                {"blocklistName": "profanity", "blocklistItemId": "item-1", "blocklistItemText": "secret-term"},
                {"blocklistName": "profanity", "blocklistItemId": "item-2", "blocklistItemText": "another-term"},
            ],
            "categoriesAnalysis": [{"category": "Hate", "severity": 4}],
        });
        let details = offline_backend()
            .parse_response(&response)
            .unwrap()
            .details
            .unwrap();
        assert_eq!(details["blocklist_matches"], json!(2));
        assert_eq!(details["max_severity"], json!(4));
        let rendered = details.to_string();
        assert!(!rendered.contains("secret-term"), "{rendered}");
        assert!(!rendered.contains("another-term"), "{rendered}");
        assert!(!rendered.contains("profanity"), "{rendered}");
    }

    #[test]
    fn missing_or_empty_category_analysis_is_a_transport_error() {
        let backend = offline_backend();
        for response in [
            json!({"blocklistsMatch": []}),
            json!({"categoriesAnalysis": []}),
            json!({"categoriesAnalysis": "Hate"}),
            json!({"categoriesAnalysis": null}),
        ] {
            let error = backend.parse_response(&response).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Transport, "{response}");
            assert!(
                error.to_string().contains("no category analysis"),
                "{error}"
            );
        }
    }

    #[tokio::test]
    async fn an_error_object_on_a_success_status_is_a_transport_error() {
        let server = server_with(ResponseTemplate::new(200).set_body_json(json!({
            "error": {"code": "InvalidRequest", "message": "the request was rejected"},
        })))
        .await;
        let backend = AzureContentSafetyBackend::new(config_for(&server)).unwrap();

        let error = backend
            .analyze(&GuardrailContext::text(SCANNED_TEXT, ScanStage::UserInput))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.to_string().contains("InvalidRequest"), "{error}");
        assert!(!error.to_string().contains(SCANNED_TEXT), "{error}");
    }

    #[tokio::test]
    async fn non_success_status_is_a_transport_error() {
        let server =
            server_with(ResponseTemplate::new(500).set_body_string("upstream unavailable")).await;
        let backend = AzureContentSafetyBackend::new(config_for(&server)).unwrap();

        let error = backend
            .analyze(&GuardrailContext::text(SCANNED_TEXT, ScanStage::UserInput))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.to_string().contains("500"), "{error}");
        assert!(
            error.to_string().contains("upstream unavailable"),
            "{error}"
        );
        assert!(!error.to_string().contains(SCANNED_TEXT), "{error}");
    }

    #[tokio::test]
    async fn empty_text_skips_the_http_request() {
        let server = MockServer::start().await;
        let backend = AzureContentSafetyBackend::new(config_for(&server)).unwrap();

        for text in ["", "   ", "\n\t "] {
            let assessment = backend
                .analyze(&GuardrailContext::text(text, ScanStage::UserInput))
                .await
                .unwrap();
            assert!(!assessment.has_risk, "text {text:?}");
            assert_eq!(assessment.risk_level, RiskLevel::Safe);
            assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
        }
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[test]
    fn debug_never_prints_the_subscription_key() {
        let config = AzureContentSafetyConfig::default()
            .with_endpoint("https://res.cognitiveservices.azure.com")
            .with_api_key(SecretString::from("super-secret-key"));
        let debug = format!("{config:?}");
        assert!(debug.contains("res.cognitiveservices.azure.com"), "{debug}");
        assert!(!debug.contains("super-secret-key"), "{debug}");

        let backend = AzureContentSafetyBackend::new(config).unwrap();
        let debug = format!("{backend:?}");
        assert!(debug.contains("AzureContentSafetyBackend"), "{debug}");
        assert!(debug.contains("res.cognitiveservices.azure.com"), "{debug}");
        assert!(!debug.contains("super-secret-key"), "{debug}");
    }
}
