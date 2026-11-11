//! Round C host-service parity suite: checkpoint contracts and the checkpoint sink, synchronous
//! hooks, awaited interactions, and native-tool/MCP configuration.
//!
//! Ports the host-service cases of `tests/unit_tests/harness_protocol/test_protocol.py` from the
//! Python SDK (v0.1.20); each ported test names its Python counterpart in a `// Python:` comment.
//! Interaction and hook models are driven through their wire shape where the canonical value does
//! not matter, so the decode paths are exercised along with the rules. Async code runs on
//! `futures::executor::block_on`: no tokio, no sleeps, no network.

// Tests assert by panicking, which is the intent of a test suite.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::executor::block_on;
use serde::de::DeserializeOwned;
use serde_json::json;

use openjiuwen_harness_protocol::*;

/// Builds a JSON object from a literal, panicking when the literal is not an object.
fn object(value: JsonValue) -> JsonObject {
    json_object_from_value(value).expect("the literal is a JSON object")
}

/// A minimal checkpoint that satisfies every invariant.
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

/// A checkpoint with the given identity and sequence.
fn checkpoint_at(checkpoint_id: &str, sequence: u64) -> HarnessCheckpoint {
    HarnessCheckpoint {
        checkpoint_id: checkpoint_id.to_owned(),
        sequence,
        ..checkpoint()
    }
}

// ---------------------------------------------------------------------------
// Checkpoints (§5.11)
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_requires_identity_fields_in_order() {
    // Python: test_protocol.py::test_checkpoint_requires_identity_fields
    type Mutate = Box<dyn Fn(&mut HarnessCheckpoint)>;
    for (mutate, message) in [
        (
            Box::new(|cp: &mut HarnessCheckpoint| cp.provider.clear()) as Mutate,
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
        let error = value.validate().expect_err("an empty identity field");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), message);
    }

    checkpoint()
        .validate()
        .expect("a complete checkpoint is valid");

    // Python checks the fields in a fixed order, so clearing all five reports `provider` first.
    let all_empty = HarnessCheckpoint {
        provider: String::new(),
        schema_version: String::new(),
        agent_id: String::new(),
        host_session_id: String::new(),
        checkpoint_id: String::new(),
        ..checkpoint()
    };
    assert_eq!(
        all_empty
            .validate()
            .expect_err("all identity fields empty")
            .message(),
        "checkpoint provider must not be empty"
    );
}

#[test]
fn checkpoint_rejects_empty_optional_ids() {
    // Python: test_protocol.py::test_checkpoint_optional_ids_must_not_be_empty
    for value in [
        HarnessCheckpoint {
            provider_session_id: Some(String::new()),
            ..checkpoint()
        },
        HarnessCheckpoint {
            revision: Some(String::new()),
            ..checkpoint()
        },
    ] {
        let error = value.validate().expect_err("an empty optional id");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(
            error.message(),
            "optional checkpoint ids must not be empty strings"
        );
    }

    HarnessCheckpoint {
        provider_session_id: Some("provider-session".to_owned()),
        revision: Some("r1".to_owned()),
        ..checkpoint()
    }
    .validate()
    .expect("both optional ids may be set");
}

#[test]
fn checkpoint_decode_rejects_a_negative_sequence() {
    // Python: test_protocol.py::test_checkpoint_rejects_negative_sequence
    let error = from_str_validated::<HarnessCheckpoint>(
        r#"{"provider": "acme", "schema_version": "1", "agent_id": "a",
            "host_session_id": "s", "checkpoint_id": "c", "sequence": -1}"#,
    )
    .expect_err("a negative sequence is not a u64");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);

    let decoded: HarnessCheckpoint = from_str_validated(
        r#"{"provider": "acme", "schema_version": "1", "agent_id": "a",
            "host_session_id": "s", "checkpoint_id": "c", "sequence": 0}"#,
    )
    .expect("a zero sequence decodes");
    assert_eq!(decoded.sequence, 0);
}

#[test]
fn checkpoint_size_bytes_matches_the_compact_encoding() {
    // Python: test_protocol.py::test_checkpoint_size_bytes
    assert_eq!(
        checkpoint().size_bytes(),
        2,
        "an empty object encodes as `{{}}`"
    );

    let data = object(json!({"z": [1, true, null], "a": "text"}));
    let value = HarnessCheckpoint {
        data: data.clone(),
        ..checkpoint()
    };
    assert_eq!(
        value.size_bytes(),
        serde_json::to_vec(&data)
            .expect("the payload serializes")
            .len()
    );
    assert_eq!(
        serde_json::to_string(&data).expect("the payload serializes"),
        r#"{"a":"text","z":[1,true,null]}"#,
        "the measurement is compact, whitespace-free UTF-8 JSON"
    );
}

#[test]
fn checkpoint_rejects_oversize_data_and_accepts_the_exact_limit() {
    // Python: test_protocol.py::test_checkpoint_data_exceeds_limit
    let oversize = HarnessCheckpoint {
        data: [(
            String::from("k"),
            json!("x".repeat(MAX_CHECKPOINT_BYTES + 1)),
        )]
        .into_iter()
        .collect(),
        ..checkpoint()
    };
    assert!(oversize.size_bytes() > MAX_CHECKPOINT_BYTES);
    let error = oversize.validate().expect_err("oversize data");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(
        error.message(),
        format!("checkpoint data exceeds {MAX_CHECKPOINT_BYTES} bytes")
    );
    assert_eq!(error.message(), "checkpoint data exceeds 4194304 bytes");

    // `{"k":"<payload>"}` encodes to `8 + payload` bytes, so this lands exactly on the limit.
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
    exact.validate().expect("the limit itself is accepted");

    let encoded = serde_json::to_string(&exact).expect("the checkpoint serializes");
    let decoded: HarnessCheckpoint = from_str_validated(&encoded).expect("decodes at the limit");
    assert_eq!(decoded.size_bytes(), MAX_CHECKPOINT_BYTES);
}

#[test]
fn checkpoint_save_receipt_requires_both_ids() {
    // Python: test_protocol.py::test_checkpoint_save_receipt_requires_ids
    let receipt = CheckpointSaveReceipt {
        checkpoint_id: "cp-1".to_owned(),
        sequence: 7,
        storage_revision: "rev-1".to_owned(),
    };
    receipt.validate().expect("a complete receipt is valid");

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
        let error = invalid.validate().expect_err("an empty receipt id");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(
            error.message(),
            "checkpoint save receipt ids must not be empty"
        );
    }
}

// ---------------------------------------------------------------------------
// Fake `HarnessCheckpointSink` and its three documented obligations (§5.11)
// ---------------------------------------------------------------------------

/// In-memory state of [`RecordingSink`]: cached receipts, monotonic sequence, storage revision.
#[derive(Default)]
struct SinkState {
    receipts: BTreeMap<String, CheckpointSaveReceipt>,
    last_sequence: Option<u64>,
    storage_revision: Option<String>,
    revisions_assigned: u64,
}

/// Fake sink implementing idempotent retry, monotonic sequences, and compare-and-set.
#[derive(Default)]
struct RecordingSink {
    state: Mutex<SinkState>,
}

#[async_trait]
impl HarnessCheckpointSink for RecordingSink {
    async fn save(
        &self,
        checkpoint: &HarnessCheckpoint,
        _reason: CheckpointReason,
        expected_storage_revision: Option<&str>,
    ) -> HarnessResult<CheckpointSaveReceipt> {
        checkpoint.validate()?;
        let mut state = self.state.lock().expect("the sink mutex is not poisoned");
        if let Some(cached) = state.receipts.get(&checkpoint.checkpoint_id) {
            return Ok(cached.clone());
        }
        if let Some(last_sequence) = state.last_sequence {
            if checkpoint.sequence <= last_sequence {
                return Err(HarnessError::checkpoint_conflict(format!(
                    "sequence {} is not greater than {last_sequence}",
                    checkpoint.sequence
                )));
            }
        }
        if let Some(expected) = expected_storage_revision {
            if state.storage_revision.as_deref() != Some(expected) {
                return Err(HarnessError::checkpoint_conflict(format!(
                    "expected storage revision {expected} does not match the stored one"
                )));
            }
        }
        state.revisions_assigned += 1;
        let receipt = CheckpointSaveReceipt {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            sequence: checkpoint.sequence,
            storage_revision: format!("rev-{}", state.revisions_assigned),
        };
        state.last_sequence = Some(checkpoint.sequence);
        state.storage_revision = Some(receipt.storage_revision.clone());
        state
            .receipts
            .insert(receipt.checkpoint_id.clone(), receipt.clone());
        Ok(receipt)
    }
}

#[test]
fn checkpoint_sink_retry_returns_the_cached_receipt_without_bumping_the_revision() {
    // Python: test_protocol.py::test_checkpoint_sink_is_idempotent
    let sink = RecordingSink::default();
    let first = block_on(sink.save(&checkpoint(), CheckpointReason::TurnCompleted, None))
        .expect("the first save succeeds");
    assert_eq!(first.checkpoint_id, "cp-1");
    assert_eq!(first.sequence, 1);
    assert_eq!(first.storage_revision, "rev-1");

    let retry = block_on(sink.save(&checkpoint(), CheckpointReason::Periodic, None))
        .expect("retrying the same checkpoint_id succeeds");
    assert_eq!(retry, first, "the original receipt is returned unchanged");
    assert_eq!(
        sink.state
            .lock()
            .expect("the sink mutex")
            .revisions_assigned,
        1,
        "an idempotent retry does not assign a new revision"
    );
}

#[test]
fn checkpoint_sink_rejects_existing_or_lower_sequences() {
    // Python: test_protocol.py::test_checkpoint_sink_requires_monotonic_sequence
    let sink = RecordingSink::default();
    block_on(sink.save(
        &checkpoint_at("cp-1", 3),
        CheckpointReason::SessionActivated,
        None,
    ))
    .expect("the first save succeeds");

    for sequence in [3, 2] {
        let error = block_on(sink.save(
            &checkpoint_at("cp-2", sequence),
            CheckpointReason::StateChanged,
            None,
        ))
        .expect_err("a new checkpoint at an existing-or-lower sequence conflicts");
        assert_eq!(error.kind(), HarnessErrorKind::CheckpointConflict);
    }

    let advanced = block_on(sink.save(
        &checkpoint_at("cp-2", 4),
        CheckpointReason::StateChanged,
        None,
    ))
    .expect("a greater sequence succeeds");
    assert_eq!(advanced.storage_revision, "rev-2");
}

#[test]
fn checkpoint_sink_enforces_compare_and_set_revisions() {
    // Python: test_protocol.py::test_checkpoint_sink_cas
    let sink = RecordingSink::default();
    let first = block_on(sink.save(&checkpoint(), CheckpointReason::TurnCompleted, None))
        .expect("the first save succeeds");

    let stale = block_on(sink.save(
        &checkpoint_at("cp-2", 2),
        CheckpointReason::TurnCompleted,
        Some("rev-0"),
    ))
    .expect_err("a stale expected revision conflicts");
    assert_eq!(stale.kind(), HarnessErrorKind::CheckpointConflict);

    let matched = block_on(sink.save(
        &checkpoint_at("cp-2", 2),
        CheckpointReason::TurnCompleted,
        Some(&first.storage_revision),
    ))
    .expect("a matching compare-and-set succeeds");
    assert_eq!(matched.storage_revision, "rev-2");
}

// ---------------------------------------------------------------------------
// Hooks (§5.12)
// ---------------------------------------------------------------------------

#[test]
fn rewrite_tool_decision_requires_updated_arguments() {
    // Python: test_protocol.py::test_rewrite_tool_decision_requires_arguments
    let rewrite_without_arguments = ToolDecision {
        decision: ToolDecisionKind::Rewrite,
        reason: None,
        updated_arguments: None,
        additional_context: None,
    };
    let error = rewrite_without_arguments
        .validate()
        .expect_err("rewrite without arguments");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(
        error.message(),
        "rewrite tool decision requires updated_arguments"
    );

    ToolDecision {
        updated_arguments: Some(object(json!({"path": "/tmp/x"}))),
        ..rewrite_without_arguments
    }
    .validate()
    .expect("rewrite with arguments is valid");
}

#[test]
fn updated_arguments_are_only_valid_for_rewrite_decisions() {
    // Python: test_protocol.py::test_updated_arguments_require_rewrite
    for decision in [
        ToolDecisionKind::Allow,
        ToolDecisionKind::Deny,
        ToolDecisionKind::Ask,
        ToolDecisionKind::ProviderPolicy,
    ] {
        let error = ToolDecision {
            decision,
            reason: None,
            updated_arguments: Some(empty_json_object()),
            additional_context: None,
        }
        .validate()
        .expect_err("misplaced updated_arguments");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(
            error.message(),
            "updated_arguments is only valid for a rewrite tool decision"
        );
    }

    ToolDecision {
        decision: ToolDecisionKind::ProviderPolicy,
        reason: Some("defer to provider".to_owned()),
        updated_arguments: None,
        additional_context: None,
    }
    .validate()
    .expect("a non-rewrite decision needs no arguments");
}

#[test]
fn hook_models_round_trip_with_every_key_present() {
    // Python: test_protocol.py::test_hook_models_round_trip
    let prompt_context = BeforePromptContext {
        agent_name: "agent".to_owned(),
        provider_session_id: None,
        turn_id: Some("turn-1".to_owned()),
        prompt: "hi".to_owned(),
    };
    let encoded = serde_json::to_value(&prompt_context).expect("serializes");
    assert_eq!(
        encoded,
        json!({
            "agent_name": "agent",
            "provider_session_id": null,
            "turn_id": "turn-1",
            "prompt": "hi",
        })
    );
    assert_eq!(
        serde_json::from_value::<BeforePromptContext>(encoded).expect("round trip decodes"),
        prompt_context
    );

    let tool_context = BeforeToolContext {
        agent_name: "agent".to_owned(),
        provider_session_id: Some("provider-session".to_owned()),
        turn_id: None,
        call_id: "call-1".to_owned(),
        tool_name: "read_file".to_owned(),
        arguments: object(json!({"path": "/tmp/x"})),
    };
    let encoded = serde_json::to_value(&tool_context).expect("serializes");
    assert_eq!(
        encoded,
        json!({
            "agent_name": "agent",
            "provider_session_id": "provider-session",
            "turn_id": null,
            "call_id": "call-1",
            "tool_name": "read_file",
            "arguments": {"path": "/tmp/x"},
        })
    );
    assert_eq!(
        serde_json::from_value::<BeforeToolContext>(encoded).expect("round trip decodes"),
        tool_context
    );

    let decision = ToolDecision {
        decision: ToolDecisionKind::Rewrite,
        reason: Some("trim the path".to_owned()),
        updated_arguments: Some(object(json!({"path": "/tmp/x"}))),
        additional_context: None,
    };
    let encoded = serde_json::to_value(&decision).expect("serializes");
    assert_eq!(
        encoded,
        json!({
            "decision": "rewrite",
            "reason": "trim the path",
            "updated_arguments": {"path": "/tmp/x"},
            "additional_context": null,
        })
    );
    let decoded: ToolDecision = serde_json::from_value(encoded).expect("round trip decodes");
    assert_eq!(decoded, decision);
    decoded
        .validate()
        .expect("the round-tripped decision validates");
}

#[test]
fn hook_results_default_like_python() {
    // Python: test_protocol.py::test_hook_defaults
    let prompt_result: BeforePromptResult =
        serde_json::from_str(r#"{"prompt": "hi"}"#).expect("all defaults decode");
    assert!(prompt_result.continue_execution, "Python default is True");
    assert_eq!(
        serde_json::to_value(&prompt_result).expect("serializes"),
        json!({
            "prompt": "hi",
            "continue_execution": true,
            "reason": null,
            "additional_context": null,
        })
    );

    let after: AfterToolResult = serde_json::from_str("{}").expect("all defaults decode");
    assert_eq!(after, AfterToolResult::default());
    assert_eq!(
        serde_json::to_value(&after).expect("serializes"),
        json!({
            "replace_result": false,
            "updated_result": null,
            "additional_context": null,
        })
    );

    let after_context: AfterToolContext = serde_json::from_str(
        r#"{"agent_name": "agent", "provider_session_id": null, "turn_id": null,
            "call_id": "call-1", "tool_name": "shell", "arguments": {}}"#,
    )
    .expect("all defaults decode");
    assert_eq!(after_context.result, JsonValue::Null);
    assert_eq!(
        serde_json::to_value(&after_context).expect("serializes")["result"],
        json!(null)
    );

    let stop: StopHookContext = serde_json::from_str(
        r#"{"agent_name": "agent", "provider_session_id": null, "turn_id": null,
            "reason": null}"#,
    )
    .expect("all defaults decode");
    assert!(stop.metadata.is_empty());
    assert_eq!(
        serde_json::to_value(&stop).expect("serializes"),
        json!({
            "agent_name": "agent",
            "provider_session_id": null,
            "turn_id": null,
            "reason": null,
            "metadata": {},
        })
    );
}

// ---------------------------------------------------------------------------
// MCP transport matrix (§5.14)
// ---------------------------------------------------------------------------

/// A config with no transport target; each helper below adds exactly one.
fn mcp_config(transport: McpTransport, name: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.to_owned(),
        transport,
        command: Vec::new(),
        url: None,
        instance: None,
        env: BTreeMap::new(),
        headers: BTreeMap::new(),
    }
}

fn stdio_config() -> McpServerConfig {
    McpServerConfig {
        command: vec!["mcp-files".to_owned(), "--root".to_owned()],
        ..mcp_config(McpTransport::Stdio, "files")
    }
}

fn http_config() -> McpServerConfig {
    McpServerConfig {
        url: Some("https://mcp.example.test".to_owned()),
        ..mcp_config(McpTransport::Http, "search")
    }
}

fn in_process_config() -> McpServerConfig {
    McpServerConfig {
        instance: Some(McpInstance::new(42_u32)),
        ..mcp_config(McpTransport::InProcess, "local")
    }
}

#[test]
fn mcp_config_accepts_one_target_per_transport() {
    // Python: test_protocol.py::test_mcp_config_valid_transports
    stdio_config().validate().expect("stdio with a command");
    http_config().validate().expect("HTTP with a url");
    in_process_config()
        .validate()
        .expect("in-process with an instance");

    let mut with_env = stdio_config();
    with_env.env = BTreeMap::from([("TOKEN".to_owned(), "secret".to_owned())]);
    with_env.validate().expect("stdio may carry env");

    let mut with_headers = http_config();
    with_headers.headers = BTreeMap::from([("Authorization".to_owned(), "x".to_owned())]);
    with_headers.validate().expect("HTTP may carry headers");
}

#[test]
fn mcp_config_rejects_empty_names_and_missing_targets() {
    // Python: test_protocol.py::test_mcp_config_missing_target
    for (config, message) in [
        (
            McpServerConfig {
                name: String::new(),
                ..stdio_config()
            },
            "MCP server name must not be empty",
        ),
        (
            McpServerConfig {
                command: Vec::new(),
                ..stdio_config()
            },
            "stdio MCP server requires a non-empty command",
        ),
        (
            McpServerConfig {
                url: None,
                ..http_config()
            },
            "HTTP MCP server requires url",
        ),
        (
            McpServerConfig {
                instance: None,
                ..in_process_config()
            },
            "in-process MCP server requires instance",
        ),
    ] {
        let error = config.validate().expect_err(message);
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), message);
    }
}

#[test]
fn mcp_config_rejects_multiple_targets() {
    // Python: test_protocol.py::test_mcp_config_multiple_targets
    let expected = "MCP server requires exactly one of command, url, or instance";
    for config in [
        McpServerConfig {
            url: Some("https://mcp.example.test".to_owned()),
            ..stdio_config()
        },
        McpServerConfig {
            instance: Some(McpInstance::new("server")),
            ..http_config()
        },
        McpServerConfig {
            command: vec!["mcp-files".to_owned()],
            ..in_process_config()
        },
    ] {
        assert_eq!(config.validate().expect_err(expected).message(), expected);
    }
}

#[test]
fn mcp_config_rejects_misplaced_env_and_headers() {
    // Python: test_protocol.py::test_mcp_config_misplaced_env_and_headers
    let env = BTreeMap::from([("TOKEN".to_owned(), "secret".to_owned())]);
    let headers = BTreeMap::from([("Authorization".to_owned(), "x".to_owned())]);
    for (config, message) in [
        (
            McpServerConfig {
                env: env.clone(),
                ..http_config()
            },
            "MCP env is only valid for stdio transport",
        ),
        (
            McpServerConfig {
                env: env.clone(),
                ..in_process_config()
            },
            "MCP env is only valid for stdio transport",
        ),
        (
            McpServerConfig {
                headers: headers.clone(),
                ..stdio_config()
            },
            "MCP headers are only valid for HTTP transport",
        ),
        (
            McpServerConfig {
                headers: headers.clone(),
                ..in_process_config()
            },
            "MCP headers are only valid for HTTP transport",
        ),
    ] {
        assert_eq!(config.validate().expect_err(message).message(), message);
    }
}

#[test]
fn mcp_instance_is_opaque_and_downcasts() {
    // Python: test_protocol.py::test_mcp_instance_is_opaque
    let instance = McpInstance::new(7_u32);
    assert_eq!(instance.downcast_ref::<u32>(), Some(&7));
    assert!(instance.downcast_ref::<String>().is_none());
    assert_eq!(format!("{instance:?}"), "<opaque instance>");
    assert_eq!(
        instance.clone(),
        instance,
        "equality only compares presence"
    );
    assert_eq!(McpInstance::new("text"), instance);
    assert_eq!(
        format!("{:?}", in_process_config().instance),
        "Some(<opaque instance>)"
    );
}

#[test]
fn mcp_config_omits_the_instance_on_the_wire() {
    // Python: test_protocol.py::test_mcp_config_instance_is_not_serialized
    assert_eq!(
        serde_json::to_value(in_process_config()).expect("serializes"),
        json!({
            "name": "local",
            "transport": "in_process",
            "command": [],
            "url": null,
            "env": {},
            "headers": {},
        })
    );

    let decoded: McpServerConfig = from_str_validated(
        r#"{"name": "files", "transport": "stdio", "command": ["mcp-files"],
            "instance": {"opaque": true}}"#,
    )
    .expect("a stdio config decodes");
    assert!(
        decoded.instance.is_none(),
        "the instance handle has no wire shape"
    );
    assert_eq!(
        decoded,
        McpServerConfig {
            command: vec!["mcp-files".to_owned()],
            ..stdio_config()
        }
    );

    let mut with_headers = http_config();
    with_headers.headers = BTreeMap::from([("X-Token".to_owned(), "abc".to_owned())]);
    let decoded: McpServerConfig =
        from_str_validated(&serde_json::to_string(&with_headers).expect("serializes"))
            .expect("round trip");
    assert_eq!(decoded, with_headers);
}

// ---------------------------------------------------------------------------
// Tools (§5.14)
// ---------------------------------------------------------------------------

#[test]
fn tool_models_validate_and_default_like_python() {
    // Python: test_protocol.py::test_tool_models
    let definition = ToolDefinition {
        name: "read_file".to_owned(),
        description: String::new(),
        input_schema: object(json!({"type": "object"})),
    };
    definition
        .validate()
        .expect("an empty description is allowed");
    let error = ToolDefinition {
        name: String::new(),
        ..definition
    }
    .validate()
    .expect_err("an empty name");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(error.message(), "tool definition name must not be empty");

    let invocation = ToolInvocation {
        call_id: "call-1".to_owned(),
        name: "read_file".to_owned(),
        arguments: empty_json_object(),
    };
    invocation
        .validate()
        .expect("a complete invocation is valid");
    for invalid in [
        ToolInvocation {
            call_id: String::new(),
            ..invocation.clone()
        },
        ToolInvocation {
            name: String::new(),
            ..invocation
        },
    ] {
        assert_eq!(
            invalid
                .validate()
                .expect_err("an empty identity field")
                .message(),
            "tool invocation call_id and name must not be empty"
        );
    }

    let result: ToolExecutionResult =
        from_str_validated(r#"{"content": {"ok": true}}"#).expect("all defaults decode");
    assert!(!result.is_error);
    assert_eq!(
        serde_json::to_value(&result).expect("serializes"),
        json!({"content": {"ok": true}, "is_error": false})
    );
}

// ---------------------------------------------------------------------------
// Interaction wire fixtures (§5.13)
// ---------------------------------------------------------------------------

/// Every interaction tag, in Python's union order.
const INTERACTION_TAGS: [&str; 5] = [
    "tool_approval",
    "user_input",
    "mcp_elicitation",
    "dynamic_tool_call",
    "provider",
];

/// Extends `wire` with the variant-specific keys of `variant`.
fn merge(wire: &mut JsonValue, variant: JsonValue) {
    let JsonValue::Object(variant) = variant else {
        panic!("the variant is a JSON object");
    };
    wire.as_object_mut()
        .expect("the template is a JSON object")
        .extend(variant);
}

/// Canonical wire form of every interaction request: shared identity plus the variant payload.
fn request_wire(tag: &str) -> JsonValue {
    let mut wire = json!({"interaction_type": tag, "request_id": "req-1"});
    let variant = match tag {
        "tool_approval" => json!({
            "call_id": "call-1",
            "tool_name": "write_file",
            "arguments": {"path": "/tmp/x"},
        }),
        "user_input" => json!({"prompt": "Which file?", "choices": ["a", "b"]}),
        "mcp_elicitation" => json!({"server_name": "files", "prompt": "Project name?"}),
        "dynamic_tool_call" => json!({"call_id": "call-1", "tool_name": "lookup"}),
        "provider" => json!({
            "provider": "acme",
            "request_type": "quota_check",
            "schema_version": "1",
        }),
        other => panic!("unknown interaction tag {other}"),
    };
    merge(&mut wire, variant);
    wire
}

/// Canonical wire form of the response paired with `tag`.
fn response_wire(tag: &str) -> JsonValue {
    let mut wire = json!({"interaction_type": tag, "request_id": "req-1"});
    let variant = match tag {
        "tool_approval" => json!({"decision": "allow"}),
        "user_input" => json!({"status": "completed", "content": "a"}),
        "mcp_elicitation" => json!({"status": "completed", "values": {"name": "Ada"}}),
        "dynamic_tool_call" => json!({"status": "completed", "result": {"ok": true}}),
        "provider" => json!({"status": "declined"}),
        other => panic!("unknown interaction tag {other}"),
    };
    merge(&mut wire, variant);
    wire
}

/// Overrides the `request_id` of a wire form.
fn with_request_id(mut wire: JsonValue, request_id: &str) -> JsonValue {
    wire["request_id"] = json!(request_id);
    wire
}

/// Decodes the canonical request for `tag` through the validated decode path.
fn request(tag: &str) -> HarnessInteractionRequest {
    from_value_validated(request_wire(tag)).expect("the canonical request is valid")
}

/// Decodes the canonical response for `tag` through the validated decode path.
fn response(tag: &str) -> HarnessInteractionResponse {
    from_value_validated(response_wire(tag)).expect("the canonical response is valid")
}

/// Decodes the canonical response for `tag` answering `request_id`.
fn response_for(tag: &str, request_id: &str) -> HarnessInteractionResponse {
    from_value_validated(with_request_id(response_wire(tag), request_id)).expect("a valid response")
}

/// Decodes the concrete request type and converts it into the dispatch enum.
fn typed_request<T: DeserializeOwned + Into<HarnessInteractionRequest>>(
    tag: &str,
) -> HarnessInteractionRequest {
    serde_json::from_value::<T>(request_wire(tag))
        .expect("the canonical request decodes")
        .into()
}

/// Decodes the concrete response type and converts it into the dispatch enum.
fn typed_response<T: DeserializeOwned + Into<HarnessInteractionResponse>>(
    tag: &str,
) -> HarnessInteractionResponse {
    serde_json::from_value::<T>(response_wire(tag))
        .expect("the canonical response decodes")
        .into()
}

/// Python's shared interaction messages, kept short so the tables below stay one row per line.
const REQUEST_ID_EMPTY: &str = "interaction request_id must not be empty";
const SESSION_ID_EMPTY: &str = "interaction provider_session_id must not be empty";
const TURN_ID_EMPTY: &str = "interaction turn_id must not be empty";
const DEADLINE_INVALID: &str = "interaction deadline_at must be a finite Unix timestamp";
const RESPONSE_ID_EMPTY: &str = "interaction response request_id must not be empty";
const PROMPT_EMPTY: &str = "user input prompt must not be empty";
const TOOL_IDS_EMPTY: &str = "tool approval call_id and tool_name must not be empty";
const ELICITATION_EMPTY: &str = "MCP elicitation server_name and prompt must not be empty";
const DYNAMIC_TOOL_IDS_EMPTY: &str = "dynamic tool call_id and tool_name must not be empty";
const NAMESPACE_EMPTY: &str = "provider interaction namespace, type, and version must not be empty";

/// Returns the `interaction_type` tag of a request.
fn tag_of(request: &HarnessInteractionRequest) -> &'static str {
    match request {
        HarnessInteractionRequest::ToolApproval(_) => "tool_approval",
        HarnessInteractionRequest::UserInput(_) => "user_input",
        HarnessInteractionRequest::McpElicitation(_) => "mcp_elicitation",
        HarnessInteractionRequest::DynamicToolCall(_) => "dynamic_tool_call",
        HarnessInteractionRequest::Provider(_) => "provider",
    }
}

/// Returns the `interaction_type` tag of a response.
fn tag_of_response(response: &HarnessInteractionResponse) -> &'static str {
    match response {
        HarnessInteractionResponse::ToolApproval(_) => "tool_approval",
        HarnessInteractionResponse::UserInput(_) => "user_input",
        HarnessInteractionResponse::McpElicitation(_) => "mcp_elicitation",
        HarnessInteractionResponse::DynamicToolCall(_) => "dynamic_tool_call",
        HarnessInteractionResponse::Provider(_) => "provider",
    }
}

#[test]
fn interaction_request_common_rules_match_python_order() {
    // Python: test_protocol.py::test_interaction_request_common_rules
    for tag in INTERACTION_TAGS {
        request(tag)
            .validate()
            .expect("a complete request is valid");

        for (field, value, message) in [
            ("request_id", json!(""), REQUEST_ID_EMPTY),
            ("provider_session_id", json!(""), SESSION_ID_EMPTY),
            ("turn_id", json!(""), TURN_ID_EMPTY),
            ("deadline_at", json!(-1.0), DEADLINE_INVALID),
        ] {
            let mut wire = request_wire(tag);
            wire[field] = value;
            let error = from_value_validated::<HarnessInteractionRequest>(wire).expect_err(message);
            assert_eq!(error.kind(), HarnessErrorKind::Validation);
            assert_eq!(error.message(), message);
        }

        // Python checks `request_id` before the optional ids and the deadline.
        let mut wire = request_wire(tag);
        wire["request_id"] = json!("");
        wire["deadline_at"] = json!(-1.0);
        let error = from_value_validated::<HarnessInteractionRequest>(wire)
            .expect_err("request_id comes first");
        assert_eq!(error.message(), REQUEST_ID_EMPTY);
    }

    let mut wire = request_wire("user_input");
    wire["provider_session_id"] = json!("");
    wire["turn_id"] = json!("");
    let error =
        from_value_validated::<HarnessInteractionRequest>(wire).expect_err("session comes first");
    assert_eq!(error.message(), SESSION_ID_EMPTY);

    // Non-finite deadlines cannot ride in JSON, so the decoded model is mutated instead.
    let mut approval: ToolApprovalRequest =
        serde_json::from_value(request_wire("tool_approval")).expect("decodes");
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        approval.deadline_at = Some(bad);
        let error = approval.validate().expect_err("a non-finite deadline");
        assert_eq!(error.message(), DEADLINE_INVALID);
    }
    approval.deadline_at = Some(0.0);
    approval.validate().expect("a zero deadline is valid");
}

#[test]
fn interaction_request_type_rules_match_python() {
    // Python: test_protocol.py::test_interaction_request_type_rules
    for (tag, field, message) in [
        ("tool_approval", "call_id", TOOL_IDS_EMPTY),
        ("tool_approval", "tool_name", TOOL_IDS_EMPTY),
        ("user_input", "prompt", PROMPT_EMPTY),
        ("mcp_elicitation", "server_name", ELICITATION_EMPTY),
        ("mcp_elicitation", "prompt", ELICITATION_EMPTY),
        ("dynamic_tool_call", "call_id", DYNAMIC_TOOL_IDS_EMPTY),
        ("dynamic_tool_call", "tool_name", DYNAMIC_TOOL_IDS_EMPTY),
        ("provider", "provider", NAMESPACE_EMPTY),
        ("provider", "request_type", NAMESPACE_EMPTY),
        ("provider", "schema_version", NAMESPACE_EMPTY),
    ] {
        let mut wire = request_wire(tag);
        wire[field] = json!("");
        let error = from_value_validated::<HarnessInteractionRequest>(wire)
            .expect_err("an empty type-specific field");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), message);
    }
}

#[test]
fn interaction_responses_reject_empty_request_ids() {
    // Python: test_protocol.py::test_interaction_response_requires_request_id
    for tag in INTERACTION_TAGS {
        let wire = with_request_id(response_wire(tag), "");
        let error = from_value_validated::<HarnessInteractionResponse>(wire)
            .expect_err("an empty response request_id");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), RESPONSE_ID_EMPTY);
        response(tag)
            .validate()
            .expect("a paired response id is valid");
    }
}

#[test]
fn interaction_enums_round_trip_through_the_interaction_type_tag() {
    // Python: test_protocol.py::test_interaction_union_round_trip
    for tag in INTERACTION_TAGS {
        let request = request(tag);
        let encoded = serde_json::to_value(&request).expect("serializes");
        assert_eq!(encoded["interaction_type"], json!(tag));
        assert_eq!(request.request_id(), "req-1");
        let decoded: HarnessInteractionRequest =
            from_value_validated(encoded).expect("round trip decodes");
        assert_eq!(decoded, request);

        let response = response(tag);
        let encoded = serde_json::to_value(&response).expect("serializes");
        assert_eq!(encoded["interaction_type"], json!(tag));
        assert_eq!(response.request_id(), "req-1");
        let decoded: HarnessInteractionResponse =
            from_value_validated(encoded).expect("round trip decodes");
        assert_eq!(decoded, response);
    }
}

#[test]
fn from_conversions_cover_every_interaction_variant() {
    // Python: test_protocol.py::test_interaction_union_membership
    for converted in [
        typed_request::<ToolApprovalRequest>("tool_approval"),
        typed_request::<UserInputRequest>("user_input"),
        typed_request::<McpElicitationRequest>("mcp_elicitation"),
        typed_request::<DynamicToolCallRequest>("dynamic_tool_call"),
        typed_request::<ProviderInteractionRequest>("provider"),
    ] {
        assert_eq!(converted.request_id(), "req-1");
        assert_eq!(converted, request(tag_of(&converted)));
    }

    for converted in [
        typed_response::<ToolApprovalResponse>("tool_approval"),
        typed_response::<UserInputResponse>("user_input"),
        typed_response::<McpElicitationResponse>("mcp_elicitation"),
        typed_response::<DynamicToolCallResponse>("dynamic_tool_call"),
        typed_response::<ProviderInteractionResponse>("provider"),
    ] {
        assert_eq!(converted.request_id(), "req-1");
        assert_eq!(converted, response(tag_of_response(&converted)));
    }
}

#[test]
fn validate_interaction_response_accepts_only_the_paired_response() {
    // Python: test_protocol.py::test_validate_interaction_response
    for tag in INTERACTION_TAGS {
        let paired = response(tag);
        assert_eq!(
            validate_interaction_response(&request(tag), paired.clone())
                .expect("the paired response validates"),
            paired
        );
    }

    let error = validate_interaction_response(
        &request("tool_approval"),
        response_for("tool_approval", "resp-2"),
    )
    .expect_err("a mismatched id is a protocol error");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "interaction response id 'resp-2' does not match request 'req-1'"
    );

    let error = validate_interaction_response(&request("tool_approval"), response("user_input"))
        .expect_err("an unpaired response type is a protocol error");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "ToolApprovalRequest requires ToolApprovalResponse, got UserInputResponse"
    );

    let error =
        validate_interaction_response(&request("dynamic_tool_call"), response("mcp_elicitation"))
            .expect_err("an unpaired response type is a protocol error");
    assert_eq!(
        error.message(),
        "DynamicToolCallRequest requires DynamicToolCallResponse, got McpElicitationResponse"
    );
}

// ---------------------------------------------------------------------------
// Fake host services
// ---------------------------------------------------------------------------

/// Fake handler recording handled request ids and `(request_id, reason)` cancellations.
#[derive(Default)]
struct RecordingInteractionHandler {
    handled: Mutex<Vec<String>>,
    cancels: Mutex<Vec<(String, InteractionCancelReason)>>,
}

#[async_trait]
impl HarnessInteractionHandler for RecordingInteractionHandler {
    async fn handle(
        &self,
        request: &HarnessInteractionRequest,
    ) -> HarnessResult<HarnessInteractionResponse> {
        let request_id = request.request_id().to_owned();
        self.handled
            .lock()
            .expect("the handler mutex is not poisoned")
            .push(request_id.clone());
        Ok(response_for(tag_of(request), &request_id))
    }

    async fn cancel(&self, request_id: &str, reason: InteractionCancelReason) -> HarnessResult<()> {
        self.cancels
            .lock()
            .expect("the handler mutex is not poisoned")
            .push((request_id.to_owned(), reason));
        Ok(())
    }
}

/// Fake gateway exposing one echo tool and recording invocations.
#[derive(Default)]
struct RecordingGateway {
    invocations: Mutex<Vec<ToolInvocation>>,
}

#[async_trait]
impl ToolGateway for RecordingGateway {
    async fn definitions(&self) -> HarnessResult<Vec<ToolDefinition>> {
        Ok(vec![ToolDefinition {
            name: "echo".to_owned(),
            description: "echoes its arguments".to_owned(),
            input_schema: object(json!({"type": "object"})),
        }])
    }

    async fn invoke(&self, invocation: &ToolInvocation) -> HarnessResult<ToolExecutionResult> {
        self.invocations
            .lock()
            .expect("the gateway mutex is not poisoned")
            .push(invocation.clone());
        Ok(ToolExecutionResult {
            content: json!({"echo": invocation.arguments}),
            is_error: false,
        })
    }
}

/// Fake hook dispatcher recording the hook order and returning fixed decisions.
#[derive(Default)]
struct RecordingHooks {
    calls: Mutex<Vec<&'static str>>,
}

#[async_trait]
impl HarnessHookDispatcher for RecordingHooks {
    async fn before_prompt(
        &self,
        context: &BeforePromptContext,
    ) -> HarnessResult<BeforePromptResult> {
        self.calls
            .lock()
            .expect("the hook mutex is not poisoned")
            .push("before_prompt");
        Ok(BeforePromptResult {
            prompt: format!("guarded: {}", context.prompt),
            continue_execution: true,
            reason: None,
            additional_context: None,
        })
    }

    async fn before_tool(&self, context: &BeforeToolContext) -> HarnessResult<ToolDecision> {
        self.calls
            .lock()
            .expect("the hook mutex is not poisoned")
            .push("before_tool");
        Ok(ToolDecision {
            decision: ToolDecisionKind::Deny,
            reason: Some(format!("{} is denied", context.tool_name)),
            updated_arguments: None,
            additional_context: None,
        })
    }

    async fn after_tool(&self, _context: &AfterToolContext) -> HarnessResult<AfterToolResult> {
        self.calls
            .lock()
            .expect("the hook mutex is not poisoned")
            .push("after_tool");
        Ok(AfterToolResult {
            replace_result: true,
            updated_result: json!("redacted"),
            additional_context: None,
        })
    }

    async fn on_stop(&self, _context: &StopHookContext) -> HarnessResult<()> {
        self.calls
            .lock()
            .expect("the hook mutex is not poisoned")
            .push("on_stop");
        Ok(())
    }
}

/// Fake telemetry sink recording `(name, attributes)` pairs.
#[derive(Default)]
struct RecordingTelemetry {
    records: Mutex<Vec<(String, JsonObject)>>,
}

impl HarnessTelemetry for RecordingTelemetry {
    fn record(&self, name: &str, attributes: &JsonObject) {
        self.records
            .lock()
            .expect("the telemetry mutex is not poisoned")
            .push((name.to_owned(), attributes.clone()));
    }
}

#[test]
fn interaction_handler_echoes_the_request_id_and_records_cancellations() {
    // Python: test_protocol.py::test_interaction_handler_cancel
    let recorder = Arc::new(RecordingInteractionHandler::default());
    let handler: Arc<dyn HarnessInteractionHandler> = recorder.clone();

    let request: HarnessInteractionRequest =
        from_value_validated(with_request_id(request_wire("tool_approval"), "req-7"))
            .expect("a valid approval request");
    let response = block_on(handler.handle(&request)).expect("the handler answers requests");
    assert_eq!(response.request_id(), "req-7", "the request id is echoed");
    assert!(matches!(
        response,
        HarnessInteractionResponse::ToolApproval(ToolApprovalResponse {
            decision: ToolApprovalDecision::Allow,
            ..
        })
    ));

    block_on(handler.cancel("req-7", InteractionCancelReason::TurnAborted))
        .expect("an explicit cancellation is recorded");
    block_on(handler.cancel_default("req-8")).expect("cancel_default uses the Python default");

    assert_eq!(
        *recorder.handled.lock().expect("the handler mutex"),
        ["req-7"]
    );
    assert_eq!(
        *recorder.cancels.lock().expect("the handler mutex"),
        [
            (String::from("req-7"), InteractionCancelReason::TurnAborted),
            (
                String::from("req-8"),
                InteractionCancelReason::ProviderWithdrew
            ),
        ]
    );
}

#[test]
#[allow(clippy::too_many_lines)] // Five trait objects, one exercised method group each.
fn host_service_traits_are_object_safe() {
    // Python: test_protocol.py::test_host_service_trait_objects
    let sink: Arc<dyn HarnessCheckpointSink> = Arc::new(RecordingSink::default());
    let receipt = block_on(sink.save(&checkpoint(), CheckpointReason::ProviderRequested, None))
        .expect("a sink behind a trait object saves");
    assert_eq!(receipt.checkpoint_id, "cp-1");
    assert_eq!(receipt.storage_revision, "rev-1");

    let gateway_state = Arc::new(RecordingGateway::default());
    let gateway: Arc<dyn ToolGateway> = gateway_state.clone();
    let definitions = block_on(gateway.definitions()).expect("a gateway lists its tools");
    assert_eq!(definitions.len(), 1);
    assert_eq!(definitions[0].name, "echo");
    let result = block_on(gateway.invoke(&ToolInvocation {
        call_id: "call-1".to_owned(),
        name: "echo".to_owned(),
        arguments: object(json!({"text": "hi"})),
    }))
    .expect("a gateway executes a tool");
    assert_eq!(result.content, json!({"echo": {"text": "hi"}}));
    assert!(!result.is_error);
    assert_eq!(
        gateway_state
            .invocations
            .lock()
            .expect("the gateway mutex")
            .len(),
        1
    );

    let hook_state = Arc::new(RecordingHooks::default());
    let hooks: Arc<dyn HarnessHookDispatcher> = hook_state.clone();
    let prompt_result = block_on(hooks.before_prompt(&BeforePromptContext {
        agent_name: "agent".to_owned(),
        provider_session_id: None,
        turn_id: None,
        prompt: "hi".to_owned(),
    }))
    .expect("before_prompt runs");
    assert_eq!(prompt_result.prompt, "guarded: hi");
    let decision = block_on(hooks.before_tool(&BeforeToolContext {
        agent_name: "agent".to_owned(),
        provider_session_id: None,
        turn_id: None,
        call_id: "call-1".to_owned(),
        tool_name: "shell".to_owned(),
        arguments: empty_json_object(),
    }))
    .expect("before_tool runs");
    assert_eq!(decision.decision, ToolDecisionKind::Deny);
    decision.validate().expect("a denial needs no arguments");
    let after = block_on(hooks.after_tool(&AfterToolContext {
        agent_name: "agent".to_owned(),
        provider_session_id: None,
        turn_id: None,
        call_id: "call-1".to_owned(),
        tool_name: "shell".to_owned(),
        arguments: empty_json_object(),
        result: json!("raw"),
        error: None,
    }))
    .expect("after_tool runs");
    assert!(after.replace_result);
    assert_eq!(after.updated_result, json!("redacted"));
    block_on(hooks.on_stop(&StopHookContext {
        agent_name: "agent".to_owned(),
        provider_session_id: None,
        turn_id: None,
        reason: None,
        metadata: empty_json_object(),
    }))
    .expect("on_stop runs");
    assert_eq!(
        hook_state.calls.lock().expect("the hook mutex").as_slice(),
        ["before_prompt", "before_tool", "after_tool", "on_stop"]
    );

    let handler_state = Arc::new(RecordingInteractionHandler::default());
    let handler: Arc<dyn HarnessInteractionHandler> = handler_state.clone();
    let response =
        block_on(handler.handle(&request("user_input"))).expect("the handler answers user input");
    assert_eq!(response.request_id(), "req-1");
    block_on(handler.cancel("req-1", InteractionCancelReason::HarnessStopped))
        .expect("the handler accepts a cancellation");
    assert_eq!(
        handler_state
            .cancels
            .lock()
            .expect("the handler mutex")
            .len(),
        1
    );

    let telemetry_state = Arc::new(RecordingTelemetry::default());
    let telemetry: Arc<dyn HarnessTelemetry> = telemetry_state.clone();
    telemetry.record("turn.completed", &object(json!({"status": "completed"})));
    let records = telemetry_state.records.lock().expect("the telemetry mutex");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].0, "turn.completed");
    assert_eq!(records[0].1["status"], json!("completed"));
}
