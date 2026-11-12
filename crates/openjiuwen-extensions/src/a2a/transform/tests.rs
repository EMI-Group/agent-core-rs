#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;

/// A metadata map from a JSON literal.
fn metadata(entries: Value) -> Map<String, Value> {
    match entries {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

/// A text-only agent part.
fn text_part(text: &str) -> AgentPart {
    AgentPart {
        text: Some(text.to_owned()),
        ..AgentPart::default()
    }
}

#[test]
fn message_ids_are_fresh_lowercase_hex() {
    // ports `test_a2a_transformer.py::test_to_a2a_request` (message id part).
    let first = A2aTransformer::to_request(&json!({ "query": "a" }))
        .unwrap()
        .message
        .message_id;
    let second = A2aTransformer::to_request(&json!({ "query": "a" }))
        .unwrap()
        .message
        .message_id;
    assert_eq!(first.len(), 32, "uuid4().hex is 32 characters: {first}");
    assert!(
        first
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "expected lowercase hex, got {first}"
    );
    assert_ne!(first, second, "every request gets a fresh message id");
}

#[test]
fn to_request_builds_the_python_message() {
    // ports `test_a2a_transformer.py::test_to_a2a_request`.
    let request = A2aTransformer::to_request(&json!({
        "query": "what is up",
        "conversation_id": "conv-1",
        "sessionId": "ignored",
        "query_null": null,
        "attempts": 2,
        "extra": { "nested": true },
    }))
    .unwrap();

    assert_eq!(request.message.role, Role::User);
    assert_eq!(request.message.context_id.as_deref(), Some("conv-1"));
    assert_eq!(request.message.parts, vec![Part::text("what is up")]);
    assert_eq!(
        request.message.metadata,
        metadata(json!({ "attempts": 2, "extra": { "nested": true } }))
    );
    assert!(request.tenant.is_none());
    assert!(request.configuration.is_none());
    assert_eq!(
        serde_json::to_value(&request).unwrap(),
        json!({
            "message": {
                "messageId": request.message.message_id,
                "contextId": "conv-1",
                "role": "ROLE_USER",
                "parts": [{ "text": "what is up", "metadata": {} }],
                "metadata": { "attempts": 2, "extra": { "nested": true } },
                "extensions": [],
                "referenceTaskIds": [],
            },
            "metadata": {},
        })
    );
}

#[test]
fn to_request_stringifies_non_string_queries() {
    assert_eq!(
        A2aTransformer::to_request(&json!({ "query": 42 }))
            .unwrap()
            .message
            .parts,
        vec![Part::text("42")]
    );
    assert_eq!(
        A2aTransformer::to_request(&json!({ "query": { "a": 1 } }))
            .unwrap()
            .message
            .parts,
        vec![Part::text(r#"{"a":1}"#)]
    );
    // A null query produces no parts at all.
    assert_eq!(
        A2aTransformer::to_request(&json!({ "query": null }))
            .unwrap()
            .message
            .parts,
        Vec::new()
    );
    // So does a missing one.
    assert_eq!(
        A2aTransformer::to_request(&json!({}))
            .unwrap()
            .message
            .parts,
        Vec::new()
    );
}

#[test]
fn to_request_omits_the_context_id_when_no_session_is_given() {
    let request = A2aTransformer::to_request(&json!({ "query": "hi" })).unwrap();
    assert!(request.message.context_id.is_none());
    assert_eq!(
        serde_json::to_value(&request).unwrap()["message"]
            .get("contextId")
            .cloned(),
        None
    );
}

#[test]
fn conversation_id_wins_over_session_id() {
    assert_eq!(
        A2aTransformer::to_request(&json!({
            "conversation_id": "conv-1",
            "sessionId": "session-1",
        }))
        .unwrap()
        .message
        .context_id
        .as_deref(),
        Some("conv-1")
    );
    assert_eq!(
        A2aTransformer::to_request(&json!({ "sessionId": "session-1" }))
            .unwrap()
            .message
            .context_id
            .as_deref(),
        Some("session-1")
    );
    // An empty `conversation_id` is falsy in Python, so `sessionId` is used.
    assert_eq!(
        A2aTransformer::to_request(&json!({
            "conversation_id": "",
            "sessionId": "session-1",
        }))
        .unwrap()
        .message
        .context_id
        .as_deref(),
        Some("session-1")
    );
    assert_eq!(
        A2aTransformer::to_request(&json!({ "conversation_id": 7 }))
            .unwrap()
            .message
            .context_id
            .as_deref(),
        Some("7")
    );
    assert!(A2aTransformer::to_request(&json!({ "sessionId": null }))
        .unwrap()
        .message
        .context_id
        .is_none());
}

#[test]
fn to_request_rejects_non_object_inputs() {
    // ports the Python `TypeError` branch of `to_a2a_request`.
    for (value, name) in [
        (Value::Null, "null"),
        (json!([1, 2]), "array"),
        (json!("query"), "string"),
        (json!(1.5), "number"),
        (json!(true), "boolean"),
    ] {
        let error = A2aTransformer::to_request(&value).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            format!("validation: request must be an object, got {name}")
        );
    }
}

#[test]
fn from_request_reads_a_send_message_request() {
    // ports `test_a2a_transformer.py::test_from_a2a_request`.
    let payload = A2aTransformer::from_request(&json!({
        "message": {
            "messageId": "m1",
            "role": "ROLE_USER",
            "contextId": "ctx-1",
            "taskId": "task-1",
            "parts": [{ "text": "" }, { "text": "the query" }],
            "metadata": { "query": "overridden", "task_id": "stale", "extra": 2, "none": null },
        },
        "metadata": { "sessionId": "stale", "request_level": true },
    }))
    .unwrap();

    assert_eq!(
        payload,
        json!({
            "query": "the query",
            "task_id": "task-1",
            "sessionId": "ctx-1",
            "extra": 2,
            "request_level": true,
        })
    );
}

#[test]
fn from_request_reads_a_bare_message() {
    let payload = A2aTransformer::from_request(&json!({
        "role": "ROLE_AGENT",
        "contextId": "ctx-2",
        "parts": [{ "text": "hello" }],
        "metadata": { "k": 1 },
    }))
    .unwrap();
    assert_eq!(
        payload,
        json!({ "query": "hello", "sessionId": "ctx-2", "k": 1 })
    );

    // A message without text still yields its ids and metadata.
    let payload = A2aTransformer::from_request(&json!({
        "parts": [{ "data": { "a": 1 } }],
        "taskId": "t3",
    }))
    .unwrap();
    assert_eq!(payload, json!({ "task_id": "t3" }));
}

#[test]
fn from_request_reads_a_plain_payload() {
    let payload = A2aTransformer::from_request(&json!({
        "query": "raw",
        "metadata": { "query": "ignored", "a": 1, "b": null },
        "other": [1, 2],
    }))
    .unwrap();
    assert_eq!(payload, json!({ "query": "raw", "a": 1, "other": [1, 2] }));
}

#[test]
fn from_request_rejects_non_objects() {
    for (value, name) in [
        (Value::Null, "null"),
        (json!([1]), "array"),
        (json!("SendMessage"), "string"),
        (json!(7), "number"),
        (json!(false), "boolean"),
    ] {
        let error = A2aTransformer::from_request(&value).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            format!("validation: a2a request params must be an object, got {name}")
        );
    }
}

#[test]
fn from_request_reports_a_malformed_message() {
    let error = A2aTransformer::from_request(&json!({
        "message": { "parts": "not-a-list" },
    }))
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error
        .to_string()
        .starts_with("validation: invalid a2a message: "));
}

#[test]
fn metadata_merge_never_overrides_and_skips_nulls() {
    // Pins Python `_merge_metadata`.
    let mut payload = Map::new();
    payload.insert(String::from("kept"), json!("original"));
    merge_metadata(
        &mut payload,
        &metadata(json!({ "kept": "overridden", "fresh": 1, "none": null })),
    );
    assert_eq!(payload, metadata(json!({ "kept": "original", "fresh": 1 })));
}

#[test]
fn from_response_reads_a_message() {
    // ports `test_a2a_transformer.py::test_from_a2a_response_message`.
    let result = A2aTransformer::from_response(&json!({
        "messageId": "m1",
        "role": "ROLE_AGENT",
        "taskId": "task-1",
        "contextId": "ctx-1",
        "parts": [{ "text": "hello" }, { "data": { "chart": [1, 2] } }],
        "metadata": { "model": "fake" },
    }));

    assert_eq!(
        result,
        AgentResult {
            task_id: Some(String::from("task-1")),
            session_id: Some(String::from("ctx-1")),
            status: AgentStatus::Completed,
            artifacts: vec![AgentArtifact {
                artifact_id: Some(String::from("message")),
                name: None,
                description: None,
                parts: vec![
                    text_part("hello"),
                    AgentPart {
                        data: Some(json!({ "chart": [1, 2] })),
                        ..AgentPart::default()
                    },
                ],
                metadata: Map::new(),
            }],
            metadata: metadata(json!({ "model": "fake" })),
        }
    );
}

#[test]
fn from_response_reads_a_task() {
    // ports `test_a2a_transformer.py::test_from_a2a_response_task`; a task
    // has both `id` and `status`, so it must not be read as a status update.
    let result = A2aTransformer::from_response(&json!({
        "id": "task-1",
        "contextId": "ctx-1",
        "status": { "state": "TASK_STATE_WORKING" },
        "artifacts": [{
            "artifactId": "a1",
            "name": "summary",
            "parts": [{ "text": "body" }],
        }],
        "history": [],
        "metadata": { "k": "v" },
    }));

    assert_eq!(result.task_id.as_deref(), Some("task-1"));
    assert_eq!(result.session_id.as_deref(), Some("ctx-1"));
    assert_eq!(result.status, AgentStatus::Working);
    assert_eq!(
        result.artifacts,
        vec![AgentArtifact {
            artifact_id: Some(String::from("a1")),
            name: Some(String::from("summary")),
            description: None,
            parts: vec![text_part("body")],
            metadata: Map::new(),
        }]
    );
    assert_eq!(result.metadata, metadata(json!({ "k": "v" })));
}

#[test]
fn from_response_reads_a_status_update() {
    let result = A2aTransformer::from_response(&json!({
        "taskId": "task-1",
        "contextId": "ctx-1",
        "status": { "state": "TASK_STATE_INPUT_REQUIRED" },
        "metadata": { "m": 1 },
    }));
    assert_eq!(result.task_id.as_deref(), Some("task-1"));
    assert_eq!(result.session_id.as_deref(), Some("ctx-1"));
    assert_eq!(result.status, AgentStatus::InputRequired);
    assert_eq!(result.artifacts, Vec::new());
    assert_eq!(result.metadata, metadata(json!({ "m": 1 })));
}

#[test]
fn from_response_reads_an_artifact_update() {
    let result = A2aTransformer::from_response(&json!({
        "taskId": "task-1",
        "contextId": "ctx-1",
        "artifact": { "artifactId": "a1", "parts": [{ "text": "chunk" }] },
        "append": true,
        "lastChunk": false,
    }));
    assert_eq!(result.task_id.as_deref(), Some("task-1"));
    assert_eq!(result.session_id.as_deref(), Some("ctx-1"));
    assert_eq!(result.status, AgentStatus::Working);
    assert_eq!(
        result.artifacts,
        vec![AgentArtifact {
            artifact_id: Some(String::from("a1")),
            name: None,
            description: None,
            parts: vec![text_part("chunk")],
            metadata: Map::new(),
        }]
    );
}

#[test]
fn from_response_unwraps_stream_response_keys() {
    // The Python proto wrapper (`{"artifact_update": {...}}`) is unwrapped
    // before classification; both the snake and camel spellings work.
    let artifact_update = A2aTransformer::from_response(&json!({
        "artifact_update": {
            "taskId": "task-1",
            "artifact": { "artifactId": "a1", "parts": [] },
        },
    }));
    assert_eq!(artifact_update.status, AgentStatus::Working);
    assert_eq!(artifact_update.artifacts.len(), 1);

    let status_update = A2aTransformer::from_response(&json!({
        "statusUpdate": {
            "taskId": "task-2",
            "status": { "state": "TASK_STATE_COMPLETED" },
        },
    }));
    assert_eq!(status_update.task_id.as_deref(), Some("task-2"));
    assert_eq!(status_update.status, AgentStatus::Completed);
}

#[test]
fn from_response_of_an_unknown_event_is_completed() {
    for value in [
        json!({}),
        json!({ "somethingElse": true }),
        json!({ "status": "not-an-object", "id": "task-1" }),
        Value::Null,
        json!("a string"),
        json!([1, 2]),
        json!(42),
    ] {
        assert_eq!(
            A2aTransformer::from_response(&value),
            AgentResult {
                status: AgentStatus::Completed,
                ..AgentResult::default()
            },
            "{value} yields the fallback result"
        );
    }
}

#[test]
fn every_task_state_maps_to_its_python_status() {
    for (state, expected) in [
        (TaskState::Unspecified, AgentStatus::Unknown),
        (TaskState::Submitted, AgentStatus::Submitted),
        (TaskState::Working, AgentStatus::Working),
        (TaskState::Completed, AgentStatus::Completed),
        (TaskState::Failed, AgentStatus::Failed),
        (TaskState::Canceled, AgentStatus::Canceled),
        (TaskState::InputRequired, AgentStatus::InputRequired),
        (TaskState::Rejected, AgentStatus::Failed),
        (TaskState::AuthRequired, AgentStatus::InputRequired),
    ] {
        assert_eq!(
            map_task_state(state),
            expected,
            "{state} maps to {expected:?}"
        );
    }
}

#[test]
fn agent_status_serializes_to_the_python_task_status_values() {
    for (status, wire) in [
        (AgentStatus::Submitted, "submitted"),
        (AgentStatus::Working, "working"),
        (AgentStatus::Paused, "paused"),
        (AgentStatus::InputRequired, "input-required"),
        (AgentStatus::Completed, "completed"),
        (AgentStatus::Canceled, "canceled"),
        (AgentStatus::Failed, "failed"),
        (AgentStatus::Waiting, "waiting"),
        (AgentStatus::Unknown, "unknown"),
    ] {
        assert_eq!(status.as_str(), wire);
        assert_eq!(serde_json::to_value(status).unwrap(), json!(wire));
        assert_eq!(
            serde_json::from_value::<AgentStatus>(json!(wire)).unwrap(),
            status
        );
    }
    assert_eq!(AgentStatus::default(), AgentStatus::Unknown);
}

#[test]
fn agent_status_decodes_tolerantly() {
    for wire in [
        "input-required",
        "input_required",
        "inputRequired",
        "INPUT_REQUIRED",
    ] {
        assert_eq!(
            serde_json::from_value::<AgentStatus>(json!(wire)).unwrap(),
            AgentStatus::InputRequired,
            "{wire} decodes to input-required"
        );
    }
    for wire in ["cancelled", "cancelled  ", "CANCELED"] {
        assert_eq!(
            serde_json::from_value::<AgentStatus>(json!(wire)).unwrap(),
            AgentStatus::Canceled,
            "{wire} decodes to canceled"
        );
    }
    for value in [json!("mystery"), json!(""), Value::Null, json!(7)] {
        assert_eq!(
            serde_json::from_value::<AgentStatus>(value.clone()).unwrap(),
            AgentStatus::Unknown,
            "{value} decodes to unknown"
        );
    }
}

#[test]
fn terminal_and_unknown_status_predicates() {
    for status in [
        AgentStatus::Completed,
        AgentStatus::Canceled,
        AgentStatus::Failed,
        AgentStatus::InputRequired,
    ] {
        assert!(status.is_terminal(), "{status:?} is terminal");
    }
    for status in [
        AgentStatus::Submitted,
        AgentStatus::Working,
        AgentStatus::Paused,
        AgentStatus::Waiting,
        AgentStatus::Unknown,
    ] {
        assert!(!status.is_terminal(), "{status:?} is not terminal");
    }
    assert!(AgentStatus::Unknown.is_unknown());
    assert!(!AgentStatus::Working.is_unknown());
}

#[test]
fn agent_result_json_shape_matches_python() {
    let result = AgentResult {
        task_id: Some(String::from("task-1")),
        session_id: Some(String::from("ctx-1")),
        status: AgentStatus::Working,
        artifacts: vec![AgentArtifact {
            artifact_id: Some(String::from("a1")),
            parts: vec![text_part("body")],
            ..AgentArtifact::default()
        }],
        metadata: metadata(json!({ "k": "v" })),
    };
    let serialized = serde_json::to_value(&result).unwrap();
    assert_eq!(
        serialized,
        json!({
            "task_id": "task-1",
            "sessionId": "ctx-1",
            "status": "working",
            "artifacts": [{
                "artifactId": "a1",
                "parts": [{ "text": "body", "metadata": {} }],
                "metadata": {},
            }],
            "metadata": { "k": "v" },
        })
    );
    assert_eq!(
        serde_json::from_value::<AgentResult>(serialized).unwrap(),
        result
    );

    // Unset options are omitted rather than emitted as `null`.
    assert_eq!(
        serde_json::to_value(AgentResult::default()).unwrap(),
        json!({ "status": "unknown", "artifacts": [], "metadata": {} })
    );
}

#[test]
fn merge_results_concatenates_and_overrides() {
    // ports `a2a_client.py::_merge_agent_results`.
    let base = AgentResult {
        task_id: Some(String::from("base-task")),
        session_id: Some(String::from("base-session")),
        status: AgentStatus::Working,
        artifacts: vec![AgentArtifact {
            artifact_id: Some(String::from("a1")),
            ..AgentArtifact::default()
        }],
        metadata: metadata(json!({ "shared": 1, "base": true })),
    };

    // An `unknown` status and missing ids do not override the base.
    let merged = merge_results(
        base.clone(),
        AgentResult {
            status: AgentStatus::Unknown,
            artifacts: vec![AgentArtifact {
                artifact_id: Some(String::from("a2")),
                ..AgentArtifact::default()
            }],
            metadata: metadata(json!({ "shared": 2, "next": true })),
            ..AgentResult::default()
        },
    );
    assert_eq!(merged.task_id.as_deref(), Some("base-task"));
    assert_eq!(merged.session_id.as_deref(), Some("base-session"));
    assert_eq!(merged.status, AgentStatus::Working);
    assert_eq!(
        merged
            .artifacts
            .iter()
            .map(|artifact| artifact.artifact_id.clone().unwrap_or_default())
            .collect::<Vec<_>>(),
        vec![String::from("a1"), String::from("a2")]
    );
    assert_eq!(
        merged.metadata,
        metadata(json!({ "shared": 2, "base": true, "next": true }))
    );

    // Known statuses and set ids win.
    let merged = merge_results(
        merged,
        AgentResult {
            task_id: Some(String::from("next-task")),
            session_id: Some(String::from("next-session")),
            status: AgentStatus::Completed,
            ..AgentResult::default()
        },
    );
    assert_eq!(merged.task_id.as_deref(), Some("next-task"));
    assert_eq!(merged.session_id.as_deref(), Some("next-session"));
    assert_eq!(merged.status, AgentStatus::Completed);
    assert_eq!(merged.artifacts.len(), 2, "an empty update adds nothing");
}

#[test]
fn resolve_session_id_follows_python_precedence() {
    assert_eq!(
        resolve_session_id(&json!({ "conversation_id": "c", "sessionId": "s" })),
        Some(String::from("c"))
    );
    assert_eq!(
        resolve_session_id(&json!({ "sessionId": "s" })),
        Some(String::from("s"))
    );
    assert_eq!(
        resolve_session_id(&json!({ "conversation_id": 7 })),
        Some(String::from("7"))
    );
    assert_eq!(
        resolve_session_id(&json!({ "conversation_id": { "a": 1 } })),
        Some(String::from(r#"{"a":1}"#))
    );
    assert_eq!(
        resolve_session_id(&json!({ "conversation_id": "", "sessionId": "s" })),
        Some(String::from("s"))
    );
    assert_eq!(resolve_session_id(&json!({})), None);
    assert_eq!(
        resolve_session_id(&json!({ "conversation_id": null })),
        None
    );
    assert_eq!(resolve_session_id(&json!([1])), None);
    assert_eq!(resolve_session_id(&Value::Null), None);
}

#[test]
fn with_session_id_overrides_only_when_given() {
    let result = with_session_id(AgentResult::default(), Some("ctx-1"));
    assert_eq!(result.session_id.as_deref(), Some("ctx-1"));

    let untouched = with_session_id(result.clone(), None);
    assert_eq!(untouched, result);

    let overwritten = with_session_id(result, Some("ctx-2"));
    assert_eq!(overwritten.session_id.as_deref(), Some("ctx-2"));
}

#[test]
fn parts_round_trip_through_the_wire_types() {
    let agent_part = AgentPart {
        text: Some(String::from("hi")),
        raw: None,
        url: Some(String::from("https://example.test/a.txt")),
        data: None,
        filename: Some(String::from("a.txt")),
        media_type: Some(String::from("text/plain")),
        metadata: metadata(json!({ "k": 1 })),
    };
    let wire = to_a2a_part(&agent_part);
    assert_eq!(
        serde_json::to_value(&wire).unwrap(),
        json!({
            "text": "hi",
            "url": "https://example.test/a.txt",
            "filename": "a.txt",
            "mediaType": "text/plain",
            "metadata": { "k": 1 },
        })
    );
    assert_eq!(from_a2a_part(&wire), agent_part);

    let text_only = text_part("bare");
    assert_eq!(
        serde_json::to_value(to_a2a_part(&text_only)).unwrap(),
        json!({ "text": "bare", "metadata": {} })
    );
    assert_eq!(from_a2a_part(&Part::text("bare")), text_only);
}

#[test]
fn structured_data_survives_the_round_trip() {
    // Documented deviation: Python stringifies `data`, this side keeps JSON.
    let agent_part = AgentPart {
        data: Some(json!({ "chart": [1, 2], "ok": true })),
        ..AgentPart::default()
    };
    let wire = to_a2a_part(&agent_part);
    assert_eq!(wire.data, agent_part.data);
    assert_eq!(from_a2a_part(&wire), agent_part);
}

#[test]
fn empty_wire_strings_decode_to_none() {
    let wire = Part {
        text: Some(String::new()),
        raw: Some(String::new()),
        url: Some(String::new()),
        data: None,
        filename: Some(String::new()),
        media_type: Some(String::new()),
        metadata: Map::new(),
    };
    assert_eq!(from_a2a_part(&wire), AgentPart::default());

    let artifact = Artifact {
        artifact_id: String::new(),
        name: Some(String::new()),
        description: Some(String::new()),
        parts: Vec::new(),
        metadata: Map::new(),
        extensions: Vec::new(),
    };
    assert_eq!(from_a2a_artifact(&artifact), AgentArtifact::default());
}

#[test]
fn artifacts_convert_in_both_directions() {
    let wire = Artifact {
        artifact_id: String::from("a9"),
        name: Some(String::from("summary")),
        description: None,
        parts: vec![Part::text("body")],
        metadata: metadata(json!({ "m": 2 })),
        extensions: vec![String::from("urn:example:ext")],
    };
    let agent_artifact = from_a2a_artifact(&wire);
    assert_eq!(
        agent_artifact,
        AgentArtifact {
            artifact_id: Some(String::from("a9")),
            name: Some(String::from("summary")),
            description: None,
            parts: vec![text_part("body")],
            metadata: metadata(json!({ "m": 2 })),
        }
    );

    assert_eq!(
        to_a2a_artifact(&agent_artifact, None),
        Artifact {
            extensions: Vec::new(),
            ..wire
        }
    );
    assert_eq!(
        to_a2a_artifact(&agent_artifact, Some("generated")).artifact_id,
        "generated"
    );
}
