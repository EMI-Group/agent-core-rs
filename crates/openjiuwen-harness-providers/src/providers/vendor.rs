//! Shared helpers for the vendor providers (`claudecode`, `codex`, `dsh`).
//!
//! The three vendor providers are real subprocess harnesses; this module holds only what they
//! share, so no provider module duplicates it: the configuration-error translation, the
//! `{"model": {…}}` / flat create fragments, and card construction.
//!
//! Nothing here spawns a process or talks to a vendor CLI — the helpers are pure data transforms.
use openjiuwen_common::error::Error;
use openjiuwen_harness_protocol::{
    json_object_to_value, HarnessCapability, HarnessCard, HarnessError, HostCapability, JsonObject,
    JsonValue, ModelChoice,
};

/// Translates a vendor configuration error into the harness SPI error.
///
/// Validation messages are preserved verbatim; an [`Error::External`] keeps its source's `Display`
/// (the SDK/IO cause), and every other workspace variant keeps its `Display` payload. The vendor
/// `from_mapping` constructors in Python raise `ValueError`, so the result is always a
/// [`HarnessError::Validation`].
pub(crate) fn config_error_to_harness(error: Error) -> HarnessError {
    match error {
        Error::Validation(message) => HarnessError::validation(message),
        Error::External(error) => HarnessError::validation(error.to_string()),
        other => HarnessError::validation(other.to_string()),
    }
}

/// Inserts `value` under `key` when it is present, leaving the key absent otherwise.
///
/// Vendor fragments omit unset fields instead of writing `null`, matching Python's
/// `{key: value for ... if value is not None}` comprehensions.
pub(crate) fn insert_opt_string(object: &mut JsonObject, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        object.insert(key.to_owned(), JsonValue::String(value.to_owned()));
    }
}

/// Builds the inner `{"model", "effort", "provider", "api_base", "api_key"}` fragment of a choice.
///
/// `model` and `effort` are always reported when set; the endpoint fields are reported when the
/// choice carries an endpoint. `provider` (the endpoint's wire-protocol label) is only included
/// when `include_endpoint_provider` is set, because DSH takes a flat mapping that has no place for
/// it. Unset fields are omitted, so an empty choice yields an empty object.
#[must_use]
pub(crate) fn model_fragment(choice: &ModelChoice, include_endpoint_provider: bool) -> JsonObject {
    let mut fragment = JsonObject::new();
    insert_opt_string(&mut fragment, "model", choice.model.as_deref());
    insert_opt_string(&mut fragment, "effort", choice.effort.as_deref());
    if let Some(endpoint) = &choice.endpoint {
        if include_endpoint_provider {
            insert_opt_string(&mut fragment, "provider", endpoint.provider.as_deref());
        }
        insert_opt_string(&mut fragment, "api_base", endpoint.api_base.as_deref());
        insert_opt_string(&mut fragment, "api_key", endpoint.api_key.as_deref());
    }
    fragment
}

/// Builds the nested create fragment: `{"model": {…}}` plus `{"fallback_model": {…}}`.
///
/// The fallback entry is only present when `fallback` is [`Some`], matching Python's
/// `fallback_model` key: the vendor runtime switches to it on its own after an authentication
/// failure.
#[must_use]
pub(crate) fn nested_model_config(
    choice: &ModelChoice,
    fallback: Option<&ModelChoice>,
    include_endpoint_provider: bool,
) -> JsonObject {
    let mut config = JsonObject::new();
    config.insert(
        "model".to_owned(),
        json_object_to_value(&model_fragment(choice, include_endpoint_provider)),
    );
    if let Some(fallback) = fallback {
        config.insert(
            "fallback_model".to_owned(),
            json_object_to_value(&model_fragment(fallback, include_endpoint_provider)),
        );
    }
    config
}

/// Builds a vendor card from its identity, capability sets and fork modes.
///
/// The protocol version fields keep the [`HarnessCard::new`] defaults, and
/// `required_host_capabilities` stays empty: a vendor card describes what the eventual
/// implementation can exploit, never a hard host requirement.
#[must_use]
pub(crate) fn vendor_card(
    name: &str,
    version: &str,
    capabilities: impl IntoIterator<Item = HarnessCapability>,
    optional_host_capabilities: impl IntoIterator<Item = HostCapability>,
    fork_modes: &[&str],
) -> HarnessCard {
    let mut card = HarnessCard::new(name, version);
    card.capabilities = capabilities.into_iter().collect();
    card.optional_host_capabilities = optional_host_capabilities.into_iter().collect();
    card.fork_modes = fork_modes.iter().map(|mode| (*mode).to_owned()).collect();
    card
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_common::error::Error;
    use openjiuwen_harness_protocol::{
        HarnessCapability, HarnessErrorKind, HostCapability, JsonValue, ModelChoice, ModelEndpoint,
        PROTOCOL_VERSION,
    };
    use serde_json::json;

    use super::{config_error_to_harness, model_fragment, nested_model_config, vendor_card};

    /// Choice with a model, no effort and no endpoint.
    fn builtin_choice() -> ModelChoice {
        ModelChoice {
            model: Some("claude-sonnet-4".to_owned()),
            effort: None,
            endpoint: None,
        }
    }

    /// Choice served by a host-supplied endpoint.
    fn endpoint_choice() -> ModelChoice {
        ModelChoice {
            model: Some("openai/gpt-5".to_owned()),
            effort: Some("high".to_owned()),
            endpoint: Some(ModelEndpoint {
                api_base: Some("https://example.test/v1".to_owned()),
                api_key: Some("sk-test".to_owned()),
                provider: Some("openai".to_owned()),
            }),
        }
    }

    #[test]
    fn model_fragment_omits_unset_fields() {
        let fragment = model_fragment(&builtin_choice(), true);
        assert_eq!(fragment.len(), 1);
        assert_eq!(
            fragment.get("model"),
            Some(&JsonValue::String("claude-sonnet-4".to_owned()))
        );
    }

    #[test]
    fn model_fragment_includes_endpoint_fields() {
        let fragment = model_fragment(&endpoint_choice(), true);
        assert_eq!(fragment.get("effort"), Some(&json!("high")));
        assert_eq!(fragment.get("provider"), Some(&json!("openai")));
        assert_eq!(
            fragment.get("api_base"),
            Some(&json!("https://example.test/v1"))
        );
        assert_eq!(fragment.get("api_key"), Some(&json!("sk-test")));
    }

    #[test]
    fn model_fragment_can_omit_the_endpoint_provider() {
        let fragment = model_fragment(&endpoint_choice(), false);
        assert!(!fragment.contains_key("provider"));
        assert!(fragment.contains_key("api_base"));
        assert!(fragment.contains_key("api_key"));
    }

    #[test]
    fn model_fragment_of_an_empty_choice_is_empty() {
        let choice = ModelChoice {
            model: None,
            effort: None,
            endpoint: None,
        };
        assert!(model_fragment(&choice, true).is_empty());
    }

    #[test]
    fn nested_model_config_wraps_model_and_fallback() {
        let fallback = ModelChoice {
            model: Some("claude-haiku-4".to_owned()),
            effort: None,
            endpoint: None,
        };
        let config = nested_model_config(&builtin_choice(), Some(&fallback), true);
        assert_eq!(
            config.get("model"),
            Some(&json!({"model": "claude-sonnet-4"}))
        );
        assert_eq!(
            config.get("fallback_model"),
            Some(&json!({"model": "claude-haiku-4"}))
        );
    }

    #[test]
    fn nested_model_config_without_fallback_has_one_key() {
        let config = nested_model_config(&builtin_choice(), None, true);
        assert_eq!(config.len(), 1);
        assert!(!config.contains_key("fallback_model"));
    }

    #[test]
    fn vendor_card_sets_capability_sets_and_protocol_defaults() {
        let card = vendor_card(
            "claude-code",
            "0.1.0",
            [HarnessCapability::Steer, HarnessCapability::McpTools],
            [HostCapability::ToolApproval],
            &["full", "before"],
        );
        assert_eq!(card.name, "claude-code");
        assert_eq!(card.implementation_version, "0.1.0");
        assert_eq!(card.protocol_version, PROTOCOL_VERSION);
        assert!(card.supports(HarnessCapability::Steer));
        assert!(card.supports(HarnessCapability::McpTools));
        assert_eq!(card.capabilities.len(), 2);
        assert_eq!(
            card.optional_host_capabilities,
            std::collections::BTreeSet::from([HostCapability::ToolApproval])
        );
        assert!(card.required_host_capabilities.is_empty());
        assert_eq!(
            card.fork_modes,
            std::collections::BTreeSet::from(["before".to_owned(), "full".to_owned()])
        );
    }

    #[test]
    fn vendor_card_allows_empty_sets() {
        let card = vendor_card("dsh", "0.3.0", [], [], &[]);
        assert!(card.capabilities.is_empty());
        assert!(card.optional_host_capabilities.is_empty());
        assert!(card.fork_modes.is_empty());
    }

    #[test]
    fn validation_errors_keep_their_message() {
        let error = config_error_to_harness(Error::validation("model must be a string"));
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "model must be a string");
    }

    #[test]
    fn external_errors_keep_the_source_message() {
        let error = config_error_to_harness(Error::external(std::io::Error::other("sdk missing")));
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "sdk missing");
    }

    #[test]
    fn other_errors_render_with_their_prefix() {
        let error = config_error_to_harness(Error::config("bad workspace"));
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "config: bad workspace");
    }
}
