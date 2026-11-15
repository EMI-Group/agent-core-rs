//! Resource subscriptions for the `rmcp`-backed MCP client.
//!
//! `resources/subscribe` is optional in MCP: a server advertises
//! `capabilities.resources.subscribe` in its `initialize` result and only then
//! accepts the request. Everything here is a pure function of that negotiated
//! [`InitializeResult`] plus the resource URI, so the capability gate and the
//! request construction are unit-tested without a server; [`super::RmcpClient`]
//! adds the session, reconnect, and error-mapping plumbing on top.
//!
//! [`McpClient::subscribe_resource`](super::McpClient::subscribe_resource) and
//! [`McpClient::unsubscribe_resource`](super::McpClient::unsubscribe_resource)
//! report a typed [`Error::Unsupported`] naming the missing capability instead of
//! sending a request the server never promised to answer.

use openjiuwen_common::error::{Error, Result};
use rmcp::model::{InitializeResult, SubscribeRequestParam, UnsubscribeRequestParam};

/// The JSON-RPC method of a `resources/subscribe` request.
pub(super) const METHOD_SUBSCRIBE: &str = "resources/subscribe";
/// The JSON-RPC method of a `resources/unsubscribe` request.
pub(super) const METHOD_UNSUBSCRIBE: &str = "resources/unsubscribe";

/// Returns whether `info` advertises `capabilities.resources.subscribe: true`.
///
/// A missing initialize result (no session yet), missing `resources`
/// capabilities, and an absent or `false` `subscribe` flag all mean "not
/// negotiated", so a subscription request would be rejected by the server.
pub(super) fn advertises_subscriptions(info: Option<&InitializeResult>) -> bool {
    info.and_then(|info| info.capabilities.resources.as_ref())
        .and_then(|resources| resources.subscribe)
        .unwrap_or(false)
}

/// The typed error for a server that never advertised the capability.
///
/// `method` is the JSON-RPC method that was refused, so an operator reading the
/// error can tell which request was withheld and why.
pub(super) fn missing_capability(method: &str) -> Error {
    Error::unsupported(format!(
        "MCP server did not advertise the `resources.subscribe` capability in its initialize \
         result, so `{method}` cannot be sent"
    ))
}

/// Checks the negotiated capability before `method` is sent.
///
/// # Errors
///
/// Returns the [`missing_capability`] error when the server did not advertise
/// `resources.subscribe`.
pub(super) fn ensure_capability(info: Option<&InitializeResult>, method: &str) -> Result<()> {
    if advertises_subscriptions(info) {
        Ok(())
    } else {
        Err(missing_capability(method))
    }
}

/// Builds the `resources/subscribe` params for `uri`.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a blank URI: the request would address no
/// resource and the server could only reject it.
pub(super) fn subscribe_params(uri: &str) -> Result<SubscribeRequestParam> {
    Ok(SubscribeRequestParam {
        uri: validate_uri(uri)?.to_owned(),
    })
}

/// Builds the `resources/unsubscribe` params for `uri`.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a blank URI.
pub(super) fn unsubscribe_params(uri: &str) -> Result<UnsubscribeRequestParam> {
    Ok(UnsubscribeRequestParam {
        uri: validate_uri(uri)?.to_owned(),
    })
}

/// Returns `uri` unchanged, or [`Error::Validation`] when it is blank.
fn validate_uri(uri: &str) -> Result<&str> {
    if uri.trim().is_empty() {
        return Err(Error::validation(
            "MCP resource subscription requires a non-empty resource URI",
        ));
    }
    Ok(uri)
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// Builds an `initialize` result carrying `capabilities`.
    fn info(capabilities: Value) -> InitializeResult {
        serde_json::from_value(json!({
            "protocolVersion": "2025-06-18",
            "capabilities": capabilities,
            "serverInfo": {"name": "mock", "version": "1.0"}
        }))
        .expect("initialize result")
    }

    #[test]
    fn advertised_subscribe_capability_is_recognized() {
        let result = info(json!({"resources": {"subscribe": true}}));

        assert!(advertises_subscriptions(Some(&result)));
        assert!(ensure_capability(Some(&result), METHOD_SUBSCRIBE).is_ok());
    }

    #[test]
    fn missing_server_info_is_not_subscribable() {
        assert!(!advertises_subscriptions(None));
    }

    #[test]
    fn empty_capabilities_are_not_subscribable() {
        let result = info(json!({}));

        assert!(!advertises_subscriptions(Some(&result)));
    }

    #[test]
    fn resources_without_the_subscribe_flag_are_not_subscribable() {
        for capabilities in [
            json!({"resources": {}}),
            json!({"resources": {"subscribe": false}}),
            json!({"resources": {"listChanged": true}}),
        ] {
            let result = info(capabilities.clone());

            assert!(
                !advertises_subscriptions(Some(&result)),
                "capabilities {capabilities} were treated as subscribable"
            );
        }
    }

    #[test]
    fn missing_capability_names_the_withheld_method() {
        let error = ensure_capability(None, METHOD_SUBSCRIBE).expect_err("no capability");

        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().contains(METHOD_SUBSCRIBE), "{error}");
        assert!(error.message().contains("resources.subscribe"), "{error}");
        assert!(missing_capability(METHOD_UNSUBSCRIBE)
            .message()
            .contains(METHOD_UNSUBSCRIBE));
    }

    #[test]
    fn subscribe_params_carry_the_uri() {
        let params = subscribe_params("file:///tmp/report.md").expect("params");

        assert_eq!(params.uri, "file:///tmp/report.md");
        assert_eq!(
            serde_json::to_value(&params).expect("json"),
            json!({"uri": "file:///tmp/report.md"})
        );
    }

    #[test]
    fn unsubscribe_params_carry_the_uri() {
        let params = unsubscribe_params("file:///tmp/report.md").expect("params");

        assert_eq!(params.uri, "file:///tmp/report.md");
    }

    #[test]
    fn blank_uris_are_rejected_before_any_request() {
        for uri in ["", "   ", "\n\t"] {
            assert_eq!(
                subscribe_params(uri).expect_err("blank uri").kind(),
                ErrorKind::Validation
            );
            assert_eq!(
                unsubscribe_params(uri).expect_err("blank uri").kind(),
                ErrorKind::Validation
            );
        }
    }
}
