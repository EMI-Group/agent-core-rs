//! How the host delivers input to a harness and how it aborts the active turn.
//!
//! Mirrors `openjiuwen.harness_protocol.models` (v0.1.20): [`DeliveryMode`], [`AbortMode`],
//! [`HarnessInput`], and [`SendReceipt`].
//!
//! Python re-freezes `content`/`metadata` in `__post_init__`; [`JsonValue`] is already owned and
//! JSON-compatible, so value semantics replace deep immutability. Constructors do not validate
//! implicitly — decode paths and callers call [`Validate::validate`].

use serde::{Deserialize, Serialize};

use crate::errors::HarnessResult;
use crate::json::{empty_json_object, JsonObject, JsonValue};
use crate::validation::{require_non_empty, Validate};

/// How an input is delivered relative to the active turn.
///
/// Wire values are `auto`, `steer`, and `follow_up`; the default is `auto`.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMode {
    /// Let the harness pick the best delivery for the current turn.
    #[default]
    Auto,
    /// Inject the input into the turn that is already running.
    Steer,
    /// Queue the input as the next turn.
    FollowUp,
}

/// How the active turn should be aborted.
///
/// Wire values are `graceful` and `force`; the default is `graceful`.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum AbortMode {
    /// Abort after the running turn reaches a safe point.
    #[default]
    Graceful,
    /// Abort the running turn immediately.
    Force,
}

/// One input accepted by an external harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct HarnessInput {
    /// JSON payload of the input; `content` is required.
    pub content: JsonValue,
    /// Provider-specific input extensions; defaults to an empty object.
    #[serde(default)]
    pub metadata: JsonObject,
}

impl HarnessInput {
    /// Builds a text input whose `content` is a JSON string and whose `metadata` is empty.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: JsonValue::String(content.into()),
            metadata: empty_json_object(),
        }
    }
}

/// Acknowledgement that a harness accepted an input command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SendReceipt {
    /// Identifier the harness assigned to the accepted input.
    pub message_id: String,
    /// Identifier of the turn the input belongs to.
    pub turn_id: String,
    /// Delivery mode the harness actually accepted (which may differ from the request).
    pub accepted_mode: DeliveryMode,
}

impl Validate for SendReceipt {
    /// Checks `message_id` and `turn_id` in Python's order.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("send receipt message_id", &self.message_id)?;
        require_non_empty("send receipt turn_id", &self.turn_id)
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{AbortMode, DeliveryMode, HarnessInput, SendReceipt};
    use crate::Validate;

    #[test]
    fn delivery_and_abort_modes_default_and_serialize() {
        assert_eq!(DeliveryMode::default(), DeliveryMode::Auto);
        assert_eq!(AbortMode::default(), AbortMode::Graceful);
        for (mode, wire) in [
            (DeliveryMode::Auto, "auto"),
            (DeliveryMode::Steer, "steer"),
            (DeliveryMode::FollowUp, "follow_up"),
        ] {
            assert_eq!(serde_json::to_value(mode).expect("serializes"), json!(wire));
        }
        for (mode, wire) in [
            (AbortMode::Graceful, "graceful"),
            (AbortMode::Force, "force"),
        ] {
            assert_eq!(serde_json::to_value(mode).expect("serializes"), json!(wire));
        }
    }

    #[test]
    fn text_input_sets_string_content_and_empty_metadata() {
        let input = HarnessInput::text("hello");
        assert_eq!(input.content, json!("hello"));
        assert!(input.metadata.is_empty());
        assert_eq!(
            serde_json::to_value(&input).expect("serializes"),
            json!({"content": "hello", "metadata": {}})
        );
    }

    #[test]
    fn input_content_is_required_on_decode() {
        // `HarnessInput` has no `__post_init__` rules, so it is decoded with plain serde.
        let error = serde_json::from_str::<HarnessInput>(r#"{"metadata": {}}"#)
            .expect_err("missing content");
        assert!(error.to_string().contains("missing field `content`"));

        let decoded: HarnessInput =
            serde_json::from_str(r#"{"content": {"block": [1, true, null]}}"#)
                .expect("valid input");
        assert_eq!(decoded.content, json!({"block": [1, true, null]}));
        assert!(decoded.metadata.is_empty());
    }

    #[test]
    fn send_receipt_validates_identifiers_in_order() {
        let receipt = SendReceipt {
            message_id: "m1".to_owned(),
            turn_id: "t1".to_owned(),
            accepted_mode: DeliveryMode::Steer,
        };
        assert!(receipt.validate().is_ok());

        let missing_message = SendReceipt {
            message_id: String::new(),
            ..receipt.clone()
        };
        let error = missing_message.validate().expect_err("empty message id");
        assert_eq!(error.message(), "send receipt message_id must not be empty");

        let missing_turn = SendReceipt {
            turn_id: String::new(),
            ..receipt
        };
        let error = missing_turn.validate().expect_err("empty turn id");
        assert_eq!(error.message(), "send receipt turn_id must not be empty");
    }
}
