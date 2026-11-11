//! Model selection, discovery, and endpoint control.
//!
//! Mirrors `openjiuwen.harness_protocol.models` (v0.1.20): [`ModelSelection`],
//! [`ModelEndpoint`], [`ModelChoice`], [`ModelOption`], and the auth-fallback request/event type
//! constants.
//!
//! [`ModelEndpoint::api_key`] is a credential: its manual `Debug` renders `Some("<redacted>")`
//! (Python's `field(repr=False)`) and it must never be logged. Constructors do not validate
//! implicitly — decode paths and callers call [`Validate::validate`].

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::json::JsonObject;
use crate::validation::Validate;

/// Provider interaction a harness sends before switching to its fallback model.
///
/// A host that completes it has persisted the switch.
pub const AUTH_FALLBACK_REQUEST_TYPE: &str = "auth_fallback";

/// Provider event announcing that the fallback model now serves the session.
pub const AUTH_FALLBACK_EVENT_TYPE: &str = "auth_fallback_activated";

/// Model and reasoning effort a harness should run its next turns on.
///
/// [`None`] keeps the current value of that field, so a selection may switch only the model, only
/// the effort, or both. Effort values are provider vocabulary (for example `low`/`high`); the
/// provider validates them, the protocol does not enumerate them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ModelSelection {
    /// Model identifier to switch to; `None` keeps the current model.
    #[serde(default)]
    pub model: Option<String>,
    /// Reasoning effort to switch to; `None` keeps the current effort.
    #[serde(default)]
    pub effort: Option<String>,
}

/// A host-supplied model endpoint, as opposed to the CLI's own login.
///
/// `provider` is the wire-protocol label of the endpoint (for example `openai`/`anthropic`);
/// providers map it, the protocol does not enumerate it. `api_key` is a credential and is never
/// logged: it is redacted by this type's manual `Debug`.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ModelEndpoint {
    /// Base URL of an OpenAI-compatible endpoint.
    #[serde(default)]
    pub api_base: Option<String>,
    /// Credential for the endpoint; redacted in `Debug` and never logged.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Wire-protocol label of the endpoint (for example `openai`).
    #[serde(default)]
    pub provider: Option<String>,
}

impl fmt::Debug for ModelEndpoint {
    /// Renders `api_key` as `Some("<redacted>")` instead of the credential.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelEndpoint")
            .field("api_base", &self.api_base)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("provider", &self.provider)
            .finish()
    }
}

/// The model a host wants a harness created on, and where it comes from.
///
/// `endpoint` is [`None`] for a model the vendor runtime offers on its own login (a built-in
/// model, the source `list_models` reports); otherwise the model is served by that host-supplied
/// endpoint. `model`/`effort` of [`None`] keep the runtime's default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ModelChoice {
    /// Model identifier to start the harness on.
    #[serde(default)]
    pub model: Option<String>,
    /// Reasoning effort to start the harness on.
    #[serde(default)]
    pub effort: Option<String>,
    /// Host-supplied endpoint serving the model; `None` means a built-in model.
    #[serde(default)]
    pub endpoint: Option<ModelEndpoint>,
}

impl ModelChoice {
    /// Returns whether the model runs on the vendor runtime's own login.
    #[must_use]
    pub fn is_builtin(&self) -> bool {
        self.endpoint.is_none()
    }
}

/// One model a provider offers, as reported by its own catalog.
///
/// `efforts` is empty when the model has no adjustable reasoning effort. `extensions` keeps
/// vendor-specific catalog data (resolved model id, feature flags) without widening the shared
/// fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ModelOption {
    /// Identifier the provider uses for this model.
    pub model_id: String,
    /// Human-readable name; defaults to an empty string.
    #[serde(default)]
    pub display_name: String,
    /// Human-readable description; defaults to an empty string.
    #[serde(default)]
    pub description: String,
    /// Adjustable reasoning efforts the model offers; defaults to empty.
    #[serde(default)]
    pub efforts: Vec<String>,
    /// Effort the provider selects when the caller does not choose one.
    #[serde(default)]
    pub default_effort: Option<String>,
    /// Whether this is the provider's default model; defaults to `false`.
    #[serde(default)]
    pub is_default: bool,
    /// Vendor-specific catalog data; defaults to an empty object.
    #[serde(default)]
    pub extensions: JsonObject,
}

/// Requires `value`, when provided, to contain something besides whitespace.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] with
/// `"{kind} {field} must be a non-empty string when provided"`.
fn require_trimmed_when_provided(
    kind: &str,
    field: &str,
    value: Option<&str>,
) -> HarnessResult<()> {
    if value.is_some_and(|text| text.trim().is_empty()) {
        return Err(HarnessError::validation(format!(
            "{kind} {field} must be a non-empty string when provided"
        )));
    }
    Ok(())
}

impl Validate for ModelSelection {
    /// Requires at least one of `model`/`effort`, and any provided value to be non-blank.
    fn validate(&self) -> HarnessResult<()> {
        if self.model.is_none() && self.effort.is_none() {
            return Err(HarnessError::validation(
                "model selection must set a model, an effort, or both",
            ));
        }
        for (field, value) in [("model", &self.model), ("effort", &self.effort)] {
            require_trimmed_when_provided("model selection", field, value.as_deref())?;
        }
        Ok(())
    }
}

impl Validate for ModelEndpoint {
    /// Requires at least one of `api_base`/`provider`, and every provided value to be non-blank.
    fn validate(&self) -> HarnessResult<()> {
        if self.api_base.is_none() && self.provider.is_none() {
            return Err(HarnessError::validation(
                "model endpoint must set an api_base, a provider, or both",
            ));
        }
        for (field, value) in [
            ("api_base", &self.api_base),
            ("api_key", &self.api_key),
            ("provider", &self.provider),
        ] {
            require_trimmed_when_provided("model endpoint", field, value.as_deref())?;
        }
        Ok(())
    }
}

impl Validate for ModelChoice {
    /// Requires at least one of `model`/`effort`, any provided value to be non-blank, and a valid
    /// `endpoint` when one is set.
    fn validate(&self) -> HarnessResult<()> {
        if self.model.is_none() && self.effort.is_none() {
            return Err(HarnessError::validation(
                "model choice must set a model, an effort, or both",
            ));
        }
        for (field, value) in [("model", &self.model), ("effort", &self.effort)] {
            require_trimmed_when_provided("model choice", field, value.as_deref())?;
        }
        if let Some(endpoint) = &self.endpoint {
            endpoint.validate()?;
        }
        Ok(())
    }
}

impl Validate for ModelOption {
    /// Requires a non-empty `model_id`, non-empty efforts, and a `default_effort` that belongs to
    /// `efforts`.
    fn validate(&self) -> HarnessResult<()> {
        if self.model_id.is_empty() {
            return Err(HarnessError::validation(
                "model option model_id must not be empty",
            ));
        }
        if self.efforts.iter().any(String::is_empty) {
            return Err(HarnessError::validation(
                "model option efforts must be non-empty strings",
            ));
        }
        if let Some(default_effort) = &self.default_effort {
            if !self.efforts.is_empty() && !self.efforts.contains(default_effort) {
                return Err(HarnessError::validation(
                    "model option default_effort must be one of its efforts",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{
        ModelChoice, ModelEndpoint, ModelOption, ModelSelection, AUTH_FALLBACK_EVENT_TYPE,
        AUTH_FALLBACK_REQUEST_TYPE,
    };
    use crate::{empty_json_object, from_str_validated, HarnessErrorKind, Validate};

    fn endpoint() -> ModelEndpoint {
        ModelEndpoint {
            api_base: Some("https://example.test/v1".to_owned()),
            api_key: Some("sk-secret-value".to_owned()),
            provider: Some("openai".to_owned()),
        }
    }

    fn option(model_id: &str, efforts: &[&str], default_effort: Option<&str>) -> ModelOption {
        ModelOption {
            model_id: model_id.to_owned(),
            display_name: String::new(),
            description: String::new(),
            efforts: efforts.iter().map(|effort| (*effort).to_owned()).collect(),
            default_effort: default_effort.map(str::to_owned),
            is_default: false,
            extensions: empty_json_object(),
        }
    }

    fn validation_message(result: Result<(), crate::HarnessError>) -> String {
        result.expect_err("invalid").message().to_owned()
    }

    #[test]
    fn selection_requires_model_or_effort_and_rejects_blanks() {
        let both_none = ModelSelection {
            model: None,
            effort: None,
        };
        assert_eq!(
            validation_message(both_none.validate()),
            "model selection must set a model, an effort, or both"
        );

        let effort_only = ModelSelection {
            model: None,
            effort: Some("high".to_owned()),
        };
        assert!(effort_only.validate().is_ok());

        let blank_model = ModelSelection {
            model: Some("   ".to_owned()),
            effort: None,
        };
        assert_eq!(
            validation_message(blank_model.validate()),
            "model selection model must be a non-empty string when provided"
        );
    }

    #[test]
    fn choice_requires_selection_and_validates_its_endpoint() {
        let both_none = ModelChoice {
            model: None,
            effort: None,
            endpoint: None,
        };
        // DESIGN §5.4 deliberately rejects both-`None`; Python's `ModelChoice` allows it.
        assert_eq!(
            validation_message(both_none.validate()),
            "model choice must set a model, an effort, or both"
        );

        let builtin = ModelChoice {
            model: Some("claude-sonnet".to_owned()),
            effort: None,
            endpoint: None,
        };
        assert!(builtin.validate().is_ok());
        assert!(builtin.is_builtin());

        let hosted = ModelChoice {
            model: Some("gpt-5".to_owned()),
            effort: Some("  ".to_owned()),
            endpoint: Some(endpoint()),
        };
        assert_eq!(
            validation_message(hosted.validate()),
            "model choice effort must be a non-empty string when provided"
        );

        let bad_endpoint = ModelChoice {
            model: Some("gpt-5".to_owned()),
            effort: None,
            endpoint: Some(ModelEndpoint {
                api_base: None,
                api_key: Some("sk-secret-value".to_owned()),
                provider: None,
            }),
        };
        assert!(!bad_endpoint.is_builtin());
        assert_eq!(
            validation_message(bad_endpoint.validate()),
            "model endpoint must set an api_base, a provider, or both"
        );
    }

    #[test]
    fn endpoint_validation_and_redacted_debug() {
        assert!(endpoint().validate().is_ok());
        assert_eq!(
            validation_message(
                ModelEndpoint {
                    api_base: Some("  ".to_owned()),
                    ..endpoint()
                }
                .validate()
            ),
            "model endpoint api_base must be a non-empty string when provided"
        );
        assert_eq!(
            validation_message(
                ModelEndpoint {
                    api_key: Some(String::new()),
                    ..endpoint()
                }
                .validate()
            ),
            "model endpoint api_key must be a non-empty string when provided"
        );
        assert_eq!(
            validation_message(
                ModelEndpoint {
                    provider: Some("  ".to_owned()),
                    ..endpoint()
                }
                .validate()
            ),
            "model endpoint provider must be a non-empty string when provided"
        );

        let debug = format!("{:?}", endpoint());
        assert!(
            debug.contains(r#"api_key: Some("<redacted>")"#),
            "unexpected debug output: {debug}"
        );
        assert!(!debug.contains("sk-secret-value"));
        assert!(debug.contains(r#"api_base: Some("https://example.test/v1")"#));
    }

    #[test]
    fn endpoint_round_trips_through_json() {
        let decoded: ModelEndpoint =
            from_str_validated(r#"{"api_base": "https://example.test/v1", "provider": "openai"}"#)
                .expect("valid endpoint");
        assert_eq!(decoded.api_key, None);
        assert_eq!(
            serde_json::to_value(&decoded).expect("serializes"),
            json!({
                "api_base": "https://example.test/v1",
                "api_key": null,
                "provider": "openai",
            })
        );
        let error =
            from_str_validated::<ModelEndpoint>(r#"{"api_key": "sk-1"}"#).expect_err("no target");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
    }

    #[test]
    fn model_option_checks_identity_efforts_and_default_membership() {
        assert!(option("gpt-5", &["low", "high"], Some("high"))
            .validate()
            .is_ok());
        assert!(option("gpt-5", &[], Some("high")).validate().is_ok());
        assert!(option("gpt-5", &["low"], None).validate().is_ok());

        assert_eq!(
            validation_message(option("", &["low"], None).validate()),
            "model option model_id must not be empty"
        );
        assert_eq!(
            validation_message(option("gpt-5", &["low", ""], None).validate()),
            "model option efforts must be non-empty strings"
        );
        assert_eq!(
            validation_message(option("gpt-5", &["low", "high"], Some("max")).validate()),
            "model option default_effort must be one of its efforts"
        );
    }

    #[test]
    fn model_option_decode_applies_defaults_and_requires_model_id() {
        let decoded: ModelOption =
            from_str_validated(r#"{"model_id": "gpt-5"}"#).expect("valid option");
        assert_eq!(
            decoded,
            ModelOption {
                model_id: "gpt-5".to_owned(),
                display_name: String::new(),
                description: String::new(),
                efforts: Vec::new(),
                default_effort: None,
                is_default: false,
                extensions: empty_json_object(),
            }
        );
        assert_eq!(
            serde_json::to_value(&decoded).expect("serializes"),
            json!({
                "model_id": "gpt-5",
                "display_name": "",
                "description": "",
                "efforts": [],
                "default_effort": null,
                "is_default": false,
                "extensions": {},
            })
        );
        let error =
            from_str_validated::<ModelOption>(r#"{"display_name": "GPT-5"}"#).expect_err("no id");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    }

    #[test]
    fn auth_fallback_constants_match_python() {
        assert_eq!(AUTH_FALLBACK_REQUEST_TYPE, "auth_fallback");
        assert_eq!(AUTH_FALLBACK_EVENT_TYPE, "auth_fallback_activated");
    }
}
