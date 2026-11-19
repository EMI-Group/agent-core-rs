//! `OpenAI` moderation adapter (`POST /v1/moderations`).
//!
//! [`OpenAiModerationBackend`] speaks the `OpenAI` moderation API directly over the
//! crate-shared [`crate::backends::http`] helpers: it posts
//! `{"input": <text>, "model": <model>}` and maps the vendor verdict
//! (`flagged`, `categories`, `category_scores`) onto a
//! [`crate::RiskAssessment`].
//!
//! The highest `category_scores` value picks both the reported category name
//! and the severity band; the flagged category names, the reported model, and
//! the request id travel in `details`. The scanned text, the credential, and
//! every raw category payload stay out of `details`, logs, and errors.
//! Empty text short-circuits to a safe assessment without any request, while a
//! reply that does not carry the moderations schema is a validation error
//! rather than a safe verdict.
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeSet;
use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};
use openjiuwen_common::security::SecretString;

use crate::backends::http::{http_client, json_body, resolve_timeout};
use crate::backends::safe_assessment;
use crate::guardrail::GuardrailBackend;
use crate::{GuardrailContext, RiskAssessment, RiskLevel};

/// Endpoint used when [`OpenAiModerationConfig::api_url`] is blank.
pub const DEFAULT_API_URL: &str = "https://api.openai.com/v1/moderations";

/// Moderation model used when [`OpenAiModerationConfig::model`] is blank.
pub const DEFAULT_MODEL: &str = "omni-moderation-latest";

/// Stable backend name, mirrored in findings and logs.
pub const NAME: &str = "openai_moderation";

/// Log target shared with the other vendor adapters.
const LOG_TARGET: &str = "openjiuwen_security::backends::vendors::openai";

/// Request timeout used by [`OpenAiModerationConfig::default`], in seconds.
const DEFAULT_TIMEOUT: f64 = 30.0;

/// Retries after the first attempt used by [`OpenAiModerationConfig::default`].
const DEFAULT_MAX_RETRIES: u32 = 2;

/// Risk type reported when the response names no category.
const DEFAULT_RISK_TYPE: &str = "openai_moderation";

/// First-retry backoff handed to the shared client, matching its default.
const DEFAULT_RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(200);

/// Diagnostic label used in HTTP error messages.
const SERVICE: &str = "openai moderation";

/// Provider label recorded in `details`.
const PROVIDER: &str = "openai";

/// `category_scores` at or above this value report [`RiskLevel::Critical`].
const CRITICAL_SCORE: f64 = 0.90;

/// `category_scores` at or above this value report [`RiskLevel::High`].
const HIGH_SCORE: f64 = 0.75;

/// `category_scores` at or above this value report [`RiskLevel::Medium`].
const MEDIUM_SCORE: f64 = 0.50;

/// Scale factor that rounds a reported score to four decimal places.
const SCORE_SCALE: f64 = 10_000.0;

/// Configuration of [`OpenAiModerationBackend`].
///
/// Field names and defaults mirror the `OpenAI` moderation request: `api_url`,
/// `api_key`, `model`, `organization`, `timeout` (seconds, `30.0`),
/// `max_retries` (`2`), and the fallback `risk_type`
/// (`"openai_moderation"`). Blank `api_url` and `model` values are replaced by
/// [`DEFAULT_API_URL`] and [`DEFAULT_MODEL`] when the backend is built.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct OpenAiModerationConfig {
    /// Absolute endpoint URL; blank means [`DEFAULT_API_URL`].
    pub api_url: String,
    /// Optional bearer credential; redacted by [`SecretString`].
    pub api_key: Option<SecretString>,
    /// Moderation model name; blank means [`DEFAULT_MODEL`].
    pub model: String,
    /// Optional `OpenAI-Organization` header value.
    pub organization: Option<String>,
    /// Request timeout in seconds (`30.0` by default).
    pub timeout: f64,
    /// Retries after the first attempt (`2` by default).
    pub max_retries: u32,
    /// Risk type reported when the response names no category.
    pub risk_type: String,
}

impl Default for OpenAiModerationConfig {
    /// The documented defaults: the public endpoint, no credential, the
    /// `omni-moderation-latest` model, `30.0` s, two retries.
    fn default() -> Self {
        Self {
            api_url: String::new(),
            api_key: None,
            model: DEFAULT_MODEL.to_owned(),
            organization: None,
            timeout: DEFAULT_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
            risk_type: DEFAULT_RISK_TYPE.to_owned(),
        }
    }
}

impl OpenAiModerationConfig {
    /// Sets the endpoint URL; blank means [`DEFAULT_API_URL`].
    #[must_use]
    pub fn with_api_url(mut self, api_url: impl Into<String>) -> Self {
        self.api_url = api_url.into();
        self
    }

    /// Sets the bearer credential.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<SecretString>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the moderation model; blank means [`DEFAULT_MODEL`].
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the `OpenAI-Organization` header value.
    #[must_use]
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
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

/// Backend that grades content through the `OpenAI` moderation API.
///
/// `analyze` short-circuits empty text, posts the remaining text, and grades
/// the reply in [`OpenAiModerationBackend::parse_response`]. The backend owns
/// one shared [`HttpClient`], so it is cheap to clone and safe to share across
/// guardrails and pipelines.
#[derive(Clone)]
pub struct OpenAiModerationBackend {
    config: OpenAiModerationConfig,
    client: HttpClient,
}

impl OpenAiModerationBackend {
    /// Builds a backend from `config`, filling the blank URL/model defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the shared `HTTP` client cannot be built
    /// with the configured timeout and retry settings.
    pub fn new(config: OpenAiModerationConfig) -> Result<Self> {
        let config = normalize_config(config);
        let client = http_client(
            resolve_timeout(config.timeout),
            config.max_retries,
            DEFAULT_RETRY_INITIAL_BACKOFF,
        )?;
        tracing::debug!(
            target: LOG_TARGET,
            model = %config.model,
            max_retries = config.max_retries,
            "openai moderation backend initialised"
        );
        Ok(Self { config, client })
    }

    /// Uses `client` instead of the client built by [`OpenAiModerationBackend::new`].
    #[must_use]
    pub fn with_client(mut self, client: HttpClient) -> Self {
        self.client = client;
        self
    }

    /// Returns the configuration this backend was built from.
    #[must_use]
    pub fn config(&self) -> &OpenAiModerationConfig {
        &self.config
    }

    /// Grades one `/v1/moderations` reply.
    ///
    /// A result whose `flagged` is `false` is a safe assessment. A reply that
    /// does not carry the moderations schema — no `results` array, an empty
    /// one, a first entry that is not an object, or one without a boolean
    /// `flagged` — is a validation failure rather than a safe verdict, so it
    /// blocks under the fail-closed default instead of reading as clean. A
    /// flagged result reports
    /// `has_risk = true`, the severity of its highest `category_scores` value
    /// (`>= 0.90` [`RiskLevel::Critical`], `>= 0.75` [`RiskLevel::High`],
    /// `>= 0.50` [`RiskLevel::Medium`], else [`RiskLevel::Low`]; flagged without
    /// scores is [`RiskLevel::High`]), that category as `risk_type`, and its
    /// score as `confidence`. `details` carries the provider, the reported (or
    /// configured) model, the sorted flagged category names, the four-decimal
    /// top score, and the request id when the reply has one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the required fields when the reply
    /// does not match the moderations schema; the message never repeats any
    /// part of the payload.
    pub fn parse_response(&self, response: &Value) -> Result<RiskAssessment> {
        let result = response
            .get("results")
            .and_then(Value::as_array)
            .and_then(|results| results.first())
            .ok_or_else(unrecognized_reply)?;

        let flagged = result
            .get("flagged")
            .and_then(Value::as_bool)
            .ok_or_else(unrecognized_reply)?;

        if !flagged {
            return Ok(safe_assessment());
        }

        let scores = result.get("category_scores").and_then(Value::as_object);
        let top = scores.and_then(top_score);
        let (risk_level, risk_type, confidence) = match &top {
            Some((category, score)) => (level_for_score(*score), category.clone(), *score),
            None => (RiskLevel::High, self.config.risk_type.clone(), 1.0),
        };

        let mut details = Map::new();
        details.insert("provider".to_owned(), Value::String(PROVIDER.to_owned()));
        details.insert(
            "model".to_owned(),
            Value::String(reported_model(response, result, &self.config.model).to_owned()),
        );
        details.insert(
            "categories".to_owned(),
            Value::Array(
                flagged_categories(result)
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            ),
        );
        details.insert(
            "top_score".to_owned(),
            json!(round_score(top.as_ref().map_or(0.0, |(_, score)| *score))),
        );
        if let Some(request_id) = request_id(response) {
            details.insert(
                "request_id".to_owned(),
                Value::String(request_id.to_owned()),
            );
        }

        Ok(RiskAssessment {
            has_risk: true,
            risk_level,
            risk_type: Some(risk_type),
            confidence,
            details: Some(Value::Object(details)),
        })
    }
}

impl fmt::Debug for OpenAiModerationBackend {
    /// Prints the endpoint, model, and settings; the credential is reduced to
    /// its presence, so a backend can be logged while diagnosing a call.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiModerationBackend")
            .field("api_url", &self.config.api_url)
            .field("model", &self.config.model)
            .field("organization_present", &self.config.organization.is_some())
            .field("timeout", &self.config.timeout)
            .field("max_retries", &self.config.max_retries)
            .field("risk_type", &self.config.risk_type)
            .field("api_key_present", &self.has_api_key())
            .finish_non_exhaustive()
    }
}

impl OpenAiModerationBackend {
    /// Whether a non-empty credential is configured.
    fn has_api_key(&self) -> bool {
        self.config
            .api_key
            .as_ref()
            .is_some_and(|key| !key.is_empty())
    }
}

#[async_trait]
impl GuardrailBackend for OpenAiModerationBackend {
    fn name(&self) -> &str {
        NAME
    }

    /// Posts the context's text projection to the moderation endpoint.
    ///
    /// # Errors
    ///
    /// [`Error::Transport`] for connection, status, and decoding failures,
    /// [`Error::Timeout`] when the deadline expires, and [`Error::Validation`]
    /// for a payload the projection cannot read or a reply that does not match
    /// the moderations schema. Blank scannable text is safe without a request.
    /// Errors never carry the scanned text.
    async fn analyze(&self, ctx: &GuardrailContext) -> Result<RiskAssessment> {
        // The checked projection, like every local detector: an unscannable
        // payload is a failure for the failure policy to resolve, never a pass.
        let text = ctx.text_view_checked()?;
        if text.trim().is_empty() {
            return Ok(safe_assessment());
        }

        let body = json!({"input": &*text, "model": self.config.model});
        let mut request = HttpRequest::post(self.config.api_url.as_str())
            .json(&body)?
            .timeout(resolve_timeout(self.config.timeout));
        if let Some(organization) = self
            .config
            .organization
            .as_deref()
            .map(str::trim)
            .filter(|organization| !organization.is_empty())
        {
            request = request.header("OpenAI-Organization", organization);
        }
        if let Some(api_key) = self.config.api_key.as_ref().filter(|key| !key.is_empty()) {
            request = request.bearer(api_key.clone());
        }

        let response = self.client.execute(request).await?;
        tracing::debug!(
            target: LOG_TARGET,
            status = response.status,
            model = %self.config.model,
            "openai moderation request completed"
        );
        let reply = json_body(response, SERVICE)?;
        self.parse_response(&reply)
    }
}

/// Builds the error reported when a `2xx` body does not match the
/// `/v1/moderations` schema.
///
/// Naming the fields keeps the mismatch actionable without repeating any part
/// of the payload, and the validation kind makes the pipeline block under the
/// fail-closed default instead of treating the reply as a clean verdict.
fn unrecognized_reply() -> Error {
    Error::validation(
        "openai moderation reply does not match the moderations schema: expected a non-empty \
         `results` array whose first entry carries a boolean `flagged`",
    )
}

/// Replaces blank URLs and models with their documented defaults and trims the
/// surrounding whitespace.
fn normalize_config(mut config: OpenAiModerationConfig) -> OpenAiModerationConfig {
    let api_url = config.api_url.trim();
    config.api_url = if api_url.is_empty() {
        DEFAULT_API_URL.to_owned()
    } else {
        api_url.to_owned()
    };
    let model = config.model.trim();
    config.model = if model.is_empty() {
        DEFAULT_MODEL.to_owned()
    } else {
        model.to_owned()
    };
    config
}

/// Returns the highest-scoring category, breaking ties by the smaller name.
fn top_score(scores: &Map<String, Value>) -> Option<(String, f64)> {
    scores
        .iter()
        .filter_map(|(category, raw)| {
            raw.as_f64()
                .filter(|score| score.is_finite())
                .map(|score| (category.clone(), score))
        })
        .max_by(|(left_name, left), (right_name, right)| {
            left.total_cmp(right)
                .then_with(|| right_name.cmp(left_name))
        })
}

/// Maps one category score onto a severity band.
fn level_for_score(score: f64) -> RiskLevel {
    if score >= CRITICAL_SCORE {
        RiskLevel::Critical
    } else if score >= HIGH_SCORE {
        RiskLevel::High
    } else if score >= MEDIUM_SCORE {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    }
}

/// Rounds a score to four decimal places for `details`.
fn round_score(score: f64) -> f64 {
    (score * SCORE_SCALE).round() / SCORE_SCALE
}

/// Collects the category names the vendor flagged `true`, sorted.
fn flagged_categories(result: &Value) -> Vec<String> {
    let Some(categories) = result.get("categories").and_then(Value::as_object) else {
        return Vec::new();
    };
    categories
        .iter()
        .filter(|(_, value)| value.as_bool() == Some(true))
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Picks the model the reply reports, falling back to the configured one.
fn reported_model<'a>(response: &'a Value, result: &'a Value, configured: &'a str) -> &'a str {
    result
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .or_else(|| {
            response
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.trim().is_empty())
        })
        .unwrap_or(configured)
}

/// Returns the reply's request id when it carries a non-empty one.
fn request_id(response: &Value) -> Option<&str> {
    response
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::ScanStage;

    /// Recorded `omni-moderation-latest` reply for violent-hateful text.
    fn flagged_fixture() -> Value {
        json!({
            "id": "modr-3f8b1f0e0f0a4c9d",
            "model": "omni-moderation-latest",
            "results": [{
                "flagged": true,
                "categories": {
                    "harassment": false,
                    "harassment/threatening": false,
                    "hate": true,
                    "hate/threatening": false,
                    "self-harm": false,
                    "self-harm/instructions": false,
                    "self-harm/intent": false,
                    "sexual": false,
                    "sexual/minors": false,
                    "violence": true,
                    "violence/graphic": false
                },
                "category_scores": {
                    "harassment": 0.0312,
                    "harassment/threatening": 0.0021,
                    "hate": 0.93,
                    "hate/threatening": 0.11,
                    "self-harm": 0.0002,
                    "self-harm/instructions": 0.0001,
                    "self-harm/intent": 0.0001,
                    "sexual": 0.0003,
                    "sexual/minors": 0.00002,
                    "violence": 0.64,
                    "violence/graphic": 0.07
                },
                "category_applied_input_types": {
                    "hate": ["text"],
                    "violence": ["text"]
                }
            }]
        })
    }

    /// Recorded `omni-moderation-latest` reply for benign text.
    fn benign_fixture() -> Value {
        json!({
            "id": "modr-benign-0001",
            "model": "omni-moderation-latest",
            "results": [{
                "flagged": false,
                "categories": {
                    "harassment": false,
                    "hate": false,
                    "violence": false
                },
                "category_scores": {
                    "harassment": 0.0001,
                    "hate": 0.0002,
                    "violence": 0.0003
                },
                "category_applied_input_types": {}
            }]
        })
    }

    fn default_backend() -> OpenAiModerationBackend {
        OpenAiModerationBackend::new(OpenAiModerationConfig::default()).unwrap()
    }

    fn endpoint(server: &MockServer) -> String {
        format!("{}/v1/moderations", server.uri())
    }

    fn backend_for(server: &MockServer) -> OpenAiModerationBackend {
        OpenAiModerationBackend::new(OpenAiModerationConfig {
            api_url: endpoint(server),
            max_retries: 0,
            ..OpenAiModerationConfig::default()
        })
        .unwrap()
    }

    fn ctx(text: &str) -> GuardrailContext {
        GuardrailContext::text(text, ScanStage::UserInput)
    }

    /// Mounts a single-shot `POST /v1/moderations` mock answering `body`.
    async fn server_with(body: Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
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
    fn config_defaults_round_trip_and_tolerate_unknown_fields() {
        let defaults = OpenAiModerationConfig::default();
        assert_eq!(defaults.api_url, "");
        assert!(defaults.api_key.is_none());
        assert_eq!(defaults.model, DEFAULT_MODEL);
        assert!(defaults.organization.is_none());
        assert!((defaults.timeout - 30.0).abs() < f64::EPSILON);
        assert_eq!(defaults.max_retries, 2);
        assert_eq!(defaults.risk_type, "openai_moderation");

        let config = defaults
            .clone()
            .with_api_url("https://moderation.example/v1/moderations")
            .with_api_key("sk-openai-secret")
            .with_model("text-moderation-007")
            .with_organization("org-abc")
            .with_timeout(12.5)
            .with_max_retries(4);

        let serialized = serde_json::to_string(&config).unwrap();
        assert!(
            serialized.contains("\"api_url\":\"https://moderation.example/v1/moderations\""),
            "{serialized}"
        );
        assert!(serialized.contains("\"model\":\"text-moderation-007\""));
        assert!(serialized.contains("\"organization\":\"org-abc\""));
        assert!(serialized.contains("\"timeout\":12.5"));
        let restored: OpenAiModerationConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(restored, config);

        let partial: OpenAiModerationConfig = serde_json::from_str(
            r#"{"model": "text-moderation-007", "vendor": {"name": "openai"}}"#,
        )
        .unwrap();
        assert_eq!(partial.model, "text-moderation-007");
        assert_eq!(partial.api_url, "");
        assert_eq!(partial.max_retries, 2);
        assert_eq!(partial.risk_type, "openai_moderation");
    }

    #[test]
    fn debug_never_prints_the_credential() {
        let config = OpenAiModerationConfig::default()
            .with_api_key("sk-openai-secret")
            .with_organization("org-abc");
        let backend = OpenAiModerationBackend::new(config.clone()).unwrap();

        let config_debug = format!("{config:?}");
        assert!(!config_debug.contains("sk-openai-secret"), "{config_debug}");

        let backend_debug = format!("{backend:?}");
        assert!(
            backend_debug.contains("api_key_present: true"),
            "{backend_debug}"
        );
        assert!(backend_debug.contains("organization_present: true"));
        assert!(backend_debug.contains(DEFAULT_API_URL));
        assert!(
            !backend_debug.contains("sk-openai-secret"),
            "{backend_debug}"
        );
    }

    #[test]
    fn blank_url_and_model_fall_back_to_defaults() {
        for config in [
            OpenAiModerationConfig::default(),
            OpenAiModerationConfig::default()
                .with_api_url("   ")
                .with_model(" \t"),
        ] {
            let backend = OpenAiModerationBackend::new(config).unwrap();
            assert_eq!(backend.config().api_url, DEFAULT_API_URL);
            assert_eq!(backend.config().model, DEFAULT_MODEL);
            assert_eq!(backend.name(), NAME);
        }
    }

    #[tokio::test]
    async fn posts_the_text_and_parses_a_recorded_omni_moderation_verdict() {
        let server = server_with(flagged_fixture()).await;
        let config = OpenAiModerationConfig {
            api_url: endpoint(&server),
            api_key: Some(SecretString::from("sk-openai-test")),
            max_retries: 0,
            ..OpenAiModerationConfig::default()
        };
        let backend = OpenAiModerationBackend::new(config).unwrap();

        let assessment = backend
            .analyze(&ctx("I will hurt you badly"))
            .await
            .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Critical);
        assert_eq!(assessment.risk_type.as_deref(), Some("hate"));
        assert!((assessment.confidence - 0.93).abs() < 1e-9);

        let details = assessment.details.unwrap();
        assert_eq!(details["provider"], json!("openai"));
        assert_eq!(details["model"], json!("omni-moderation-latest"));
        assert_eq!(details["categories"], json!(["hate", "violence"]));
        assert!((details["top_score"].as_f64().unwrap() - 0.93).abs() < 1e-9);
        assert_eq!(details["request_id"], json!("modr-3f8b1f0e0f0a4c9d"));

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let sent = &received[0];
        assert_eq!(sent.method.as_str(), "POST");
        assert_eq!(sent.url.path(), "/v1/moderations");
        assert_eq!(
            header_value(sent, "authorization"),
            Some("Bearer sk-openai-test")
        );
        assert_eq!(header_value(sent, "openai-organization"), None);
        assert_eq!(
            body_json(sent),
            json!({"input": "I will hurt you badly", "model": DEFAULT_MODEL})
        );
    }

    #[tokio::test]
    async fn custom_endpoint_model_and_organization_reach_the_wire() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations-v2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(flagged_fixture()))
            .expect(1)
            .mount(&server)
            .await;

        let config = OpenAiModerationConfig::default()
            .with_api_url(format!("{}/moderations-v2", server.uri()))
            .with_model("text-moderation-007")
            .with_organization("org-abc")
            .with_max_retries(0);
        let backend = OpenAiModerationBackend::new(config).unwrap();

        backend.analyze(&ctx("some text")).await.unwrap();

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let sent = &received[0];
        assert_eq!(sent.url.path(), "/moderations-v2");
        assert_eq!(header_value(sent, "openai-organization"), Some("org-abc"));
        assert_eq!(header_value(sent, "authorization"), None);
        let body = body_json(sent);
        assert_eq!(body["model"], json!("text-moderation-007"));
        assert_eq!(body["input"], json!("some text"));
    }

    #[tokio::test]
    async fn benign_verdict_is_a_safe_assessment() {
        let server = server_with(benign_fixture()).await;
        let backend = backend_for(&server);

        let assessment = backend
            .analyze(&ctx("the weather is pleasant today"))
            .await
            .unwrap();
        assert!(!assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
        assert_eq!(assessment.risk_type, None);
        assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
        assert_eq!(assessment.details, None);

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(
            body_json(&received[0])["input"],
            json!("the weather is pleasant today")
        );
    }

    #[tokio::test]
    async fn flagged_violence_at_0_93_is_critical() {
        let server = server_with(json!({
            "id": "modr-violence-0001",
            "model": "omni-moderation-latest",
            "results": [{
                "flagged": true,
                "categories": {"violence": true, "hate": false},
                "category_scores": {"violence": 0.93, "hate": 0.02}
            }]
        }))
        .await;

        let assessment = backend_for(&server)
            .analyze(&ctx("some violent text"))
            .await
            .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Critical);
        assert_eq!(assessment.risk_type.as_deref(), Some("violence"));
        assert!((assessment.confidence - 0.93).abs() < 1e-9);

        let details = assessment.details.unwrap();
        assert_eq!(details["provider"], json!("openai"));
        assert_eq!(details["model"], json!("omni-moderation-latest"));
        assert_eq!(details["categories"], json!(["violence"]));
        assert!((details["top_score"].as_f64().unwrap() - 0.93).abs() < 1e-9);
        assert_eq!(details["request_id"], json!("modr-violence-0001"));
    }

    #[tokio::test]
    async fn unrecognized_reply_shapes_are_validation_errors() {
        for reply in [
            json!({}),
            json!({"id": "modr-empty", "model": "omni-moderation-latest"}),
            json!({"results": []}),
            json!({"results": {}}),
            json!({"results": [{}]}),
            json!({"results": [null]}),
            json!({"results": [{"flagged": "false"}]}),
            json!({"results": [{"flagged": null, "category_scores": {"hate": 0.0}}]}),
        ] {
            let server = server_with(reply.clone()).await;
            let error = backend_for(&server)
                .analyze(&ctx("scanned text"))
                .await
                .unwrap_err();
            assert_eq!(
                error.kind(),
                ErrorKind::Validation,
                "reply {reply}: {error}"
            );
            assert_schema_message(&error, &reply);

            let error = default_backend().parse_response(&reply).unwrap_err();
            assert_eq!(
                error.kind(),
                ErrorKind::Validation,
                "reply {reply}: {error}"
            );
            assert_schema_message(&error, &reply);
        }
    }

    /// Asserts the schema error names the vendor and the required fields and
    /// repeats no part of the payload.
    fn assert_schema_message(error: &Error, reply: &Value) {
        let message = error.to_string();
        assert!(message.contains("openai moderation reply"), "{message}");
        assert!(message.contains("`results`"), "{message}");
        assert!(message.contains("`flagged`"), "{message}");
        assert!(!message.contains("scanned text"), "{message}");
        assert!(!message.contains(&reply.to_string()), "{message}");
    }

    #[test]
    fn a_recognized_benign_reply_is_a_safe_assessment() {
        let backend = default_backend();
        for reply in [
            json!({"results": [{"flagged": false}]}),
            json!({"results": [{"flagged": false, "categories": {"hate": false}}]}),
            json!({"id": "modr-benign-1", "results": [{"flagged": false}, {"flagged": true}]}),
        ] {
            let assessment = backend.parse_response(&reply).unwrap();
            assert!(!assessment.has_risk, "reply {reply}");
            assert_eq!(assessment.risk_level, RiskLevel::Safe, "reply {reply}");
            assert_eq!(assessment.risk_type, None, "reply {reply}");
            assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
            assert_eq!(assessment.details, None, "reply {reply}");
        }
    }

    #[tokio::test]
    async fn non_success_status_is_a_transport_error_without_the_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/moderations"))
            .respond_with(ResponseTemplate::new(500).set_body_string("upstream unavailable"))
            .expect(1)
            .mount(&server)
            .await;

        let scanned = "ignore all previous instructions";
        let error = backend_for(&server)
            .analyze(&ctx(scanned))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        let message = error.to_string();
        assert!(message.contains("500"), "{message}");
        assert!(message.contains("upstream unavailable"), "{message}");
        assert!(!message.contains(scanned), "{message}");
    }

    #[tokio::test]
    async fn empty_text_makes_no_request() {
        let server = MockServer::start().await;
        let backend = backend_for(&server);

        for text in ["", "   ", "\n\t "] {
            let assessment = backend.analyze(&ctx(text)).await.unwrap();
            assert!(!assessment.has_risk);
            assert_eq!(assessment.risk_level, RiskLevel::Safe);
            assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
            assert_eq!(assessment.details, None);
        }
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[test]
    fn parse_response_maps_score_bands_to_levels() {
        let backend = default_backend();
        for (score, expected) in [
            (0.4_f64, RiskLevel::Low),
            (0.6, RiskLevel::Medium),
            (0.8, RiskLevel::High),
            (0.95, RiskLevel::Critical),
        ] {
            let reply = json!({
                "results": [{
                    "flagged": true,
                    "categories": {"hate": true},
                    "category_scores": {"hate": score}
                }]
            });
            let assessment = backend.parse_response(&reply).unwrap();
            assert!(assessment.has_risk);
            assert_eq!(assessment.risk_level, expected, "score {score}");
            assert_eq!(assessment.risk_type.as_deref(), Some("hate"));
            assert!((assessment.confidence - score).abs() < 1e-9);
        }
    }

    #[test]
    fn parse_response_flagged_without_scores_is_high() {
        let backend = default_backend();
        let reply = json!({
            "results": [{"flagged": true, "categories": {"violence": true}}]
        });
        let assessment = backend.parse_response(&reply).unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::High);
        assert_eq!(assessment.risk_type.as_deref(), Some(DEFAULT_RISK_TYPE));
        assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);

        let details = assessment.details.unwrap();
        assert_eq!(details["categories"], json!(["violence"]));
        assert_eq!(details["top_score"], json!(0.0));
        assert_eq!(details["model"], json!(DEFAULT_MODEL));
        assert_eq!(details["request_id"], Value::Null);

        let custom = OpenAiModerationBackend::new(OpenAiModerationConfig {
            risk_type: "vendor_flagged".to_owned(),
            ..OpenAiModerationConfig::default()
        })
        .unwrap();
        let assessment = custom
            .parse_response(&json!({"results": [{"flagged": true}]}))
            .unwrap();
        assert_eq!(assessment.risk_type.as_deref(), Some("vendor_flagged"));
        assert_eq!(assessment.details.unwrap()["categories"], json!([]));
    }

    #[test]
    fn parse_response_sorts_flagged_categories_and_rounds_the_top_score() {
        let backend = default_backend();
        let reply = json!({
            "id": "modr-multi",
            "results": [{
                "flagged": true,
                "categories": {
                    "violence": true,
                    "hate": true,
                    "harassment": false,
                    "self-harm": true
                },
                "category_scores": {
                    "violence": 0.612_345_678,
                    "hate": 0.51,
                    "harassment": 0.2,
                    "self-harm": 0.3
                }
            }]
        });
        let assessment = backend.parse_response(&reply).unwrap();
        assert_eq!(assessment.risk_level, RiskLevel::Medium);
        assert_eq!(assessment.risk_type.as_deref(), Some("violence"));

        let details = assessment.details.unwrap();
        assert_eq!(
            details["categories"],
            json!(["hate", "self-harm", "violence"])
        );
        assert_eq!(details["top_score"], json!(0.6123));
        assert_eq!(details["model"], json!(DEFAULT_MODEL));
    }

    #[test]
    fn parse_response_uses_the_reported_model_when_present() {
        let backend = default_backend();
        let reply = json!({
            "model": "text-moderation-007",
            "results": [{
                "flagged": true,
                "categories": {"hate": true},
                "category_scores": {"hate": 0.8}
            }]
        });
        let details = backend.parse_response(&reply).unwrap().details.unwrap();
        assert_eq!(details["model"], json!("text-moderation-007"));
    }
}
