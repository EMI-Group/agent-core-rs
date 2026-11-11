//! Versioned checkpoint contract for third-party agent harnesses.
//!
//! Mirrors `openjiuwen.harness_protocol.checkpoints` (v0.1.20): [`MAX_CHECKPOINT_BYTES`],
//! [`CheckpointReason`], [`HarnessCheckpoint`], [`CheckpointSaveReceipt`], and the
//! [`HarnessCheckpointSink`] SPI.
//!
//! A checkpoint is opaque provider state scoped to one agent and host session: only the provider
//! named by [`HarnessCheckpoint::provider`] may interpret
//! [`data`](HarnessCheckpoint::data) and [`schema_version`](HarnessCheckpoint::schema_version);
//! the host persists the complete envelope unchanged.
//!
//! Python freezes `data` in `__post_init__`; [`JsonObject`] is already owned and JSON-compatible,
//! so value semantics replace deep immutability. Python's `sequence < 0` check has no analogue:
//! `sequence` is a `u64`, so a negative wire value is rejected by serde. Constructors do not
//! validate implicitly — decode paths and callers that need the Python invariants call
//! [`Validate::validate`].

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::json::JsonObject;
use crate::validation::{require_non_empty, require_optional_non_empty, Validate};

/// Largest compact-JSON encoding accepted for a checkpoint's `data` payload (4 MiB).
pub const MAX_CHECKPOINT_BYTES: usize = 4 * 1024 * 1024;

/// Reason an implementation publishes a checkpoint.
///
/// Wire values are `session_activated`, `turn_completed`, `state_changed`, `periodic`, and
/// `provider_requested`.
#[derive(
    Debug,
    Clone,
    Copy,
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
pub enum CheckpointReason {
    /// The harness session was activated (or reactivated) for the agent.
    SessionActivated,
    /// A turn finished and its result may now be persisted.
    TurnCompleted,
    /// Provider state changed outside a turn boundary.
    StateChanged,
    /// A periodic checkpoint published on the provider's own schedule.
    Periodic,
    /// The provider explicitly asked the host to persist a checkpoint.
    ProviderRequested,
}

/// Opaque provider state scoped to one agent and host session.
///
/// Only the provider named by [`provider`](Self::provider) may interpret
/// [`data`](Self::data) and [`schema_version`](Self::schema_version); the host persists the
/// complete envelope unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct HarnessCheckpoint {
    /// Provider identifier that owns the checkpoint payload.
    pub provider: String,
    /// Provider-defined version of the payload's shape.
    pub schema_version: String,
    /// Agent the checkpoint belongs to.
    pub agent_id: String,
    /// Host session the checkpoint belongs to.
    pub host_session_id: String,
    /// Identifier of this checkpoint, unique within the provider/agent/host-session scope.
    pub checkpoint_id: String,
    /// Monotonic sequence number within the provider/agent/host-session scope.
    ///
    /// Python checks `sequence < 0`; the `u64` field rejects a negative wire value in serde
    /// instead (see `DESIGN.md` §7).
    pub sequence: u64,
    /// Opaque provider payload; decoded as an empty object when absent.
    #[serde(default)]
    pub data: JsonObject,
    /// Provider-side session identifier, when the harness keeps a persistent session.
    pub provider_session_id: Option<String>,
    /// Provider-side revision of the payload, when the provider versions its state.
    pub revision: Option<String>,
}

impl HarnessCheckpoint {
    /// Returns the compact UTF-8 encoded size of [`data`](Self::data).
    ///
    /// The value equals `serde_json::to_vec(&data)?.len()` and is the number
    /// [`validate`](Validate::validate) compares against [`MAX_CHECKPOINT_BYTES`].
    /// A payload that cannot be encoded (impossible for a [`JsonObject`], which is
    /// JSON-compatible by construction) reports [`usize::MAX`] instead of panicking.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        serde_json::to_vec(&self.data).map_or(usize::MAX, |bytes| bytes.len())
    }
}

impl Validate for HarnessCheckpoint {
    /// Checks the identity fields, the optional ids, then the encoded payload size.
    ///
    /// The identity fields are checked in Python's order (`provider`, `schema_version`,
    /// `agent_id`, `host_session_id`, `checkpoint_id`); both optional ids share Python's combined
    /// message. Python's `sequence < 0` check is subsumed by the `u64` field.
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("checkpoint provider", &self.provider)?;
        require_non_empty("checkpoint schema_version", &self.schema_version)?;
        require_non_empty("checkpoint agent_id", &self.agent_id)?;
        require_non_empty("checkpoint host_session_id", &self.host_session_id)?;
        require_non_empty("checkpoint checkpoint_id", &self.checkpoint_id)?;

        // Python raises one shared message for either optional id, not the per-field wording.
        let optional_ids = [
            (
                "checkpoint provider_session_id",
                self.provider_session_id.as_deref(),
            ),
            ("checkpoint revision", self.revision.as_deref()),
        ];
        for (field_name, value) in optional_ids {
            if require_optional_non_empty(field_name, value).is_err() {
                return Err(HarnessError::validation(
                    "optional checkpoint ids must not be empty strings",
                ));
            }
        }

        if self.size_bytes() > MAX_CHECKPOINT_BYTES {
            return Err(HarnessError::validation(format!(
                "checkpoint data exceeds {MAX_CHECKPOINT_BYTES} bytes"
            )));
        }
        Ok(())
    }
}

/// Durable acknowledgement of one idempotent checkpoint write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct CheckpointSaveReceipt {
    /// Identifier of the checkpoint that was persisted.
    pub checkpoint_id: String,
    /// Sequence number of the persisted checkpoint.
    ///
    /// Python additionally rejects a negative sequence; the `u64` field rejects a negative wire
    /// value in serde instead (see `DESIGN.md` §7).
    pub sequence: u64,
    /// Storage revision the sink assigned to the persisted checkpoint.
    pub storage_revision: String,
}

impl Validate for CheckpointSaveReceipt {
    /// Checks that both ids are non-empty (Python's single combined message).
    fn validate(&self) -> HarnessResult<()> {
        if self.checkpoint_id.is_empty() || self.storage_revision.is_empty() {
            return Err(HarnessError::validation(
                "checkpoint save receipt ids must not be empty",
            ));
        }
        Ok(())
    }
}

/// Durable host-side destination for proactively published checkpoints.
///
/// Implementations must honour three obligations:
///
/// 1. Sequence numbers are monotonic within one provider/agent/host-session scope.
/// 2. Retrying the same `checkpoint_id` is idempotent and returns the original receipt.
/// 3. A different write at an existing or lower sequence, or a failed
///    `expected_storage_revision` comparison, returns
///    [`HarnessError::CheckpointConflict`] instead of overwriting newer state.
///
/// The trait is object-safe (`async-trait` + `Send + Sync`), so hosts can hold
/// `Arc<dyn HarnessCheckpointSink>`.
#[async_trait]
pub trait HarnessCheckpointSink: Send + Sync {
    /// Persists `checkpoint` durably and returns its storage revision.
    ///
    /// `reason` records why the provider published the checkpoint. When
    /// `expected_storage_revision` is `Some`, the write succeeds only if the stored revision for
    /// this provider/agent/host-session scope still matches it (compare-and-set).
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::CheckpointConflict`] when the checkpoint's sequence is not greater
    /// than the stored sequence, when `expected_storage_revision` does not match the stored
    /// revision, or when the same `checkpoint_id` is retried with different content.
    /// Transport or storage failures are reported as any other [`HarnessError`] variant.
    async fn save(
        &self,
        checkpoint: &HarnessCheckpoint,
        reason: CheckpointReason,
        expected_storage_revision: Option<&str>,
    ) -> HarnessResult<CheckpointSaveReceipt>;
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{CheckpointReason, CheckpointSaveReceipt, HarnessCheckpoint, MAX_CHECKPOINT_BYTES};
    use crate::json::empty_json_object;
    use crate::{from_str_validated, HarnessErrorKind, Validate};

    fn checkpoint() -> HarnessCheckpoint {
        HarnessCheckpoint {
            provider: "acme".to_owned(),
            schema_version: "1".to_owned(),
            agent_id: "agent-1".to_owned(),
            host_session_id: "session-1".to_owned(),
            checkpoint_id: "cp-1".to_owned(),
            sequence: 1,
            data: empty_json_object(),
            provider_session_id: None,
            revision: None,
        }
    }

    #[test]
    fn reason_wire_values_match_python() {
        for (reason, wire) in [
            (CheckpointReason::SessionActivated, "session_activated"),
            (CheckpointReason::TurnCompleted, "turn_completed"),
            (CheckpointReason::StateChanged, "state_changed"),
            (CheckpointReason::Periodic, "periodic"),
            (CheckpointReason::ProviderRequested, "provider_requested"),
        ] {
            assert_eq!(
                serde_json::to_value(reason).expect("serializes"),
                json!(wire)
            );
            let decoded: CheckpointReason =
                serde_json::from_value(json!(wire)).expect("deserializes");
            assert_eq!(decoded, reason);
        }
    }

    #[test]
    fn identity_fields_are_rejected_in_python_order() {
        for (mutate, message) in [
            (
                Box::new(|cp: &mut HarnessCheckpoint| cp.provider.clear())
                    as Box<dyn Fn(&mut HarnessCheckpoint)>,
                "checkpoint provider must not be empty",
            ),
            (
                Box::new(|cp: &mut HarnessCheckpoint| cp.schema_version.clear()),
                "checkpoint schema_version must not be empty",
            ),
            (
                Box::new(|cp: &mut HarnessCheckpoint| cp.agent_id.clear()),
                "checkpoint agent_id must not be empty",
            ),
            (
                Box::new(|cp: &mut HarnessCheckpoint| cp.host_session_id.clear()),
                "checkpoint host_session_id must not be empty",
            ),
            (
                Box::new(|cp: &mut HarnessCheckpoint| cp.checkpoint_id.clear()),
                "checkpoint checkpoint_id must not be empty",
            ),
        ] {
            let mut value = checkpoint();
            mutate(&mut value);
            let error = value.validate().expect_err("empty identity field");
            assert_eq!(error.message(), message);
            assert_eq!(error.kind(), HarnessErrorKind::Validation);
        }
        assert!(checkpoint().validate().is_ok());
    }

    #[test]
    fn optional_ids_share_the_python_message() {
        for (mutate, message) in [
            (
                Box::new(|cp: &mut HarnessCheckpoint| cp.provider_session_id = Some(String::new()))
                    as Box<dyn Fn(&mut HarnessCheckpoint)>,
                "optional checkpoint ids must not be empty strings",
            ),
            (
                Box::new(|cp: &mut HarnessCheckpoint| cp.revision = Some(String::new())),
                "optional checkpoint ids must not be empty strings",
            ),
        ] {
            let mut value = checkpoint();
            mutate(&mut value);
            let error = value.validate().expect_err("empty optional id");
            assert_eq!(error.message(), message);
        }

        let value = HarnessCheckpoint {
            provider_session_id: Some("provider-session".to_owned()),
            revision: Some("r1".to_owned()),
            ..checkpoint()
        };
        assert!(value.validate().is_ok());
    }

    #[test]
    fn size_bytes_matches_compact_encoding_and_bounds_data() {
        let empty = checkpoint();
        assert_eq!(empty.size_bytes(), 2, "an empty object encodes as `{{}}`");

        // `{"k":"<payload>"}` encodes to `8 + payload` bytes.
        let exact = HarnessCheckpoint {
            data: [(
                String::from("k"),
                json!("x".repeat(MAX_CHECKPOINT_BYTES - 8)),
            )]
            .into_iter()
            .collect(),
            ..checkpoint()
        };
        assert_eq!(exact.size_bytes(), MAX_CHECKPOINT_BYTES);
        assert!(exact.validate().is_ok(), "the limit itself is accepted");

        let oversize = HarnessCheckpoint {
            data: [(
                String::from("k"),
                json!("x".repeat(MAX_CHECKPOINT_BYTES - 7)),
            )]
            .into_iter()
            .collect(),
            ..checkpoint()
        };
        assert_eq!(oversize.size_bytes(), MAX_CHECKPOINT_BYTES + 1);
        let error = oversize.validate().expect_err("oversize data");
        assert_eq!(
            error.message(),
            format!("checkpoint data exceeds {MAX_CHECKPOINT_BYTES} bytes")
        );
    }

    #[test]
    fn decode_applies_defaults_and_rejects_negative_sequence() {
        let decoded: HarnessCheckpoint = from_str_validated(
            r#"{"provider": "acme", "schema_version": "1", "agent_id": "a",
                "host_session_id": "s", "checkpoint_id": "c", "sequence": 3}"#,
        )
        .expect("valid checkpoint");
        assert!(decoded.data.is_empty());
        assert_eq!(decoded.provider_session_id, None);
        assert_eq!(decoded.revision, None);
        assert_eq!(decoded.sequence, 3);

        let error = from_str_validated::<HarnessCheckpoint>(
            r#"{"provider": "acme", "schema_version": "1", "agent_id": "a",
                "host_session_id": "s", "checkpoint_id": "c", "sequence": -1}"#,
        )
        .expect_err("negative sequence");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    }

    #[test]
    fn save_receipt_requires_both_ids() {
        let receipt = CheckpointSaveReceipt {
            checkpoint_id: "cp-1".to_owned(),
            sequence: 7,
            storage_revision: "rev-1".to_owned(),
        };
        assert!(receipt.validate().is_ok());

        for invalid in [
            CheckpointSaveReceipt {
                checkpoint_id: String::new(),
                ..receipt.clone()
            },
            CheckpointSaveReceipt {
                storage_revision: String::new(),
                ..receipt.clone()
            },
        ] {
            let error = invalid.validate().expect_err("empty receipt id");
            assert_eq!(
                error.message(),
                "checkpoint save receipt ids must not be empty"
            );
        }

        assert_eq!(
            serde_json::to_value(&receipt).expect("serializes"),
            json!({"checkpoint_id": "cp-1", "sequence": 7, "storage_revision": "rev-1"})
        );
    }
}
