// Tests assert by panicking on failure, which is the intent of a test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde_json::json;

use super::super::transform::AgentPart;
use super::super::types::Artifact;
use super::*;
use openjiuwen_harness_protocol::{
    MonetaryAmount, TurnError, TurnTermination, TurnTerminationKind, TurnUsage,
};

/// Builds A2A message metadata.
fn metadata(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

/// Builds a harness JSON object.
fn json_object(pairs: &[(&str, Value)]) -> JsonObject {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

/// Builds a content block with no provider data.
fn block(block_id: &str, kind: &str, content: Value) -> ContentBlock {
    ContentBlock {
        block_id: block_id.to_owned(),
        kind: kind.to_owned(),
        content,
        data: JsonObject::new(),
    }
}

/// Builds an otherwise-empty turn result with the given status.
fn turn_result(status: TurnStatus) -> TurnResult {
    TurnResult {
        status,
        messages: Vec::new(),
        final_output: Value::Null,
        structured_output: Value::Null,
        stop_reason: None,
        termination: None,
        error: None,
        usage: None,
        cost: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        provider_data: JsonObject::new(),
    }
}

/// Asserts that the value looks like a fresh `uuid4().hex` identifier.
fn assert_hex_id(id: &str) {
    assert_eq!(id.len(), 32, "{id} has 32 characters");
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()), "{id} is hex");
}

#[test]
fn single_text_message_becomes_string_content() {
    let message = Message {
        message_id: "m1".to_owned(),
        role: Role::User,
        parts: vec![Part::text("hello")],
        metadata: metadata(&[("model", json!("m"))]),
        ..Message::default()
    };
    let input = message_to_harness_input(&message);
    assert_eq!(input.content, json!("hello"));
    assert_eq!(input.metadata.get("model"), Some(&json!("m")));
}

#[test]
fn multi_part_message_becomes_a_part_array() {
    let message = Message {
        parts: vec![
            Part::text("a"),
            Part {
                data: Some(json!({ "k": 1 })),
                ..Part::default()
            },
        ],
        ..Message::default()
    };
    let input = message_to_harness_input(&message);
    let items = input.content.as_array().expect("array content");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].get("text"), Some(&json!("a")));
    assert_eq!(items[1].get("data"), Some(&json!({ "k": 1 })));
    assert!(input.metadata.is_empty());
}

#[test]
fn text_with_another_content_field_stays_a_part_array() {
    let message = Message {
        parts: vec![Part {
            text: Some("a".to_owned()),
            data: Some(json!(1)),
            ..Part::default()
        }],
        ..Message::default()
    };
    let input = message_to_harness_input(&message);
    assert!(input.content.is_array(), "{}", input.content);
}

#[test]
fn message_without_parts_becomes_null_content() {
    let input = message_to_harness_input(&Message::default());
    assert_eq!(input.content, Value::Null);
    assert!(input.metadata.is_empty());
}

#[test]
fn text_input_becomes_one_text_part() {
    let message = harness_input_to_message(&HarnessInput::text("hello"), "m7");
    assert_eq!(message.message_id, "m7");
    assert_eq!(message.role, Role::Agent);
    assert_eq!(message.parts, vec![Part::text("hello")]);
    assert!(message.metadata.is_empty());
    assert!(message.context_id.is_none());
    assert!(message.task_id.is_none());
}

#[test]
fn part_array_input_round_trips_through_the_bridge() {
    let message = Message {
        parts: vec![Part::text("a"), Part::text("b")],
        metadata: metadata(&[("k", json!("v"))]),
        ..Message::default()
    };
    let input = message_to_harness_input(&message);
    let decoded = harness_input_to_message(&input, "m8");
    assert_eq!(decoded.parts, message.parts);
    assert_eq!(decoded.metadata, message.metadata);
    assert_eq!(decoded.role, Role::Agent);
    assert_eq!(decoded.message_id, "m8");
}

#[test]
fn undecodable_array_elements_are_skipped() {
    let input = HarnessInput {
        content: json!([{ "text": "keep" }, "not-a-part", 7, null]),
        metadata: JsonObject::new(),
    };
    let message = harness_input_to_message(&input, "m9");
    assert_eq!(message.parts, vec![Part::text("keep")]);
}

#[test]
fn parts_member_of_an_object_is_decoded() {
    let input = HarnessInput {
        content: json!({
            "parts": [
                { "text": "x" },
                { "url": "https://example.test/a.txt", "mediaType": "text/plain" },
            ],
            "ignored": true,
        }),
        metadata: JsonObject::new(),
    };
    let message = harness_input_to_message(&input, "m10");
    assert_eq!(
        message.parts,
        vec![
            Part::text("x"),
            Part {
                url: Some("https://example.test/a.txt".to_owned()),
                media_type: Some("text/plain".to_owned()),
                ..Part::default()
            },
        ]
    );
}

#[test]
fn non_part_content_becomes_a_data_part() {
    for content in [
        json!({ "kind": "custom", "payload": [1, 2] }),
        json!(42),
        json!(true),
        Value::Null,
    ] {
        let input = HarnessInput {
            content: content.clone(),
            metadata: json_object(&[("a", json!(1))]),
        };
        let message = harness_input_to_message(&input, "m11");
        assert_eq!(
            message.parts,
            vec![Part {
                data: Some(content.clone()),
                ..Part::default()
            }],
            "{content}"
        );
        assert_eq!(message.metadata.get("a"), Some(&json!(1)));
    }
}

#[test]
fn part_block_kind_precedence_is_text_then_data_then_url_then_raw() {
    let all_fields = Part {
        text: Some("t".to_owned()),
        raw: Some("r".to_owned()),
        url: Some("u".to_owned()),
        data: Some(json!(1)),
        filename: Some("f".to_owned()),
        media_type: Some("m".to_owned()),
        metadata: metadata(&[("k", json!(2))]),
    };
    let block = part_to_content_block(&all_fields, "b1");
    assert_eq!(block.block_id, "b1");
    assert_eq!(block.kind, "text");
    assert_eq!(block.content, json!({ "text": "t" }));
    assert_eq!(block.data.get("k"), Some(&json!(2)));

    let data_part = Part {
        raw: Some("r".to_owned()),
        url: Some("u".to_owned()),
        data: Some(json!({ "answer": 41 })),
        ..Part::default()
    };
    let block = part_to_content_block(&data_part, "b2");
    assert_eq!(block.kind, "data");
    assert_eq!(block.content, json!({ "answer": 41 }));

    let url_part = Part {
        raw: Some("r".to_owned()),
        url: Some("https://example.test/f.bin".to_owned()),
        ..Part::default()
    };
    let block = part_to_content_block(&url_part, "b3");
    assert_eq!(block.kind, "file");
    assert_eq!(
        block.content,
        json!({ "url": "https://example.test/f.bin" })
    );

    let raw_part = Part {
        raw: Some("AAEC".to_owned()),
        ..Part::default()
    };
    let block = part_to_content_block(&raw_part, "b4");
    assert_eq!(block.kind, "binary");
    assert_eq!(block.content, json!({ "raw": "AAEC" }));

    let block = part_to_content_block(&Part::default(), "b5");
    assert_eq!(block.kind, "empty");
    assert_eq!(block.content, Value::Null);
    assert!(block.data.is_empty());
}

#[test]
fn file_and_binary_blocks_carry_media_type_and_filename() {
    let url_part = Part {
        url: Some("https://example.test/f.pdf".to_owned()),
        filename: Some("f.pdf".to_owned()),
        media_type: Some("application/pdf".to_owned()),
        ..Part::default()
    };
    let block = part_to_content_block(&url_part, "b1");
    assert_eq!(
        block.content,
        json!({
            "url": "https://example.test/f.pdf",
            "media_type": "application/pdf",
            "filename": "f.pdf",
        })
    );
    let raw_part = Part {
        raw: Some("AAEC".to_owned()),
        filename: Some("f.pdf".to_owned()),
        media_type: Some("application/pdf".to_owned()),
        ..Part::default()
    };
    let block = part_to_content_block(&raw_part, "b2");
    assert_eq!(
        block.content,
        json!({
            "raw": "AAEC",
            "media_type": "application/pdf",
            "filename": "f.pdf",
        })
    );
}

#[test]
fn parts_round_trip_through_content_blocks() {
    let parts = [
        Part::text("plain"),
        Part {
            text: Some("with metadata".to_owned()),
            metadata: metadata(&[("k", json!("v"))]),
            ..Part::default()
        },
        Part {
            data: Some(json!({ "answer": 42 })),
            metadata: metadata(&[("k", json!(1))]),
            ..Part::default()
        },
        Part {
            url: Some("https://example.test/a.txt".to_owned()),
            media_type: Some("text/plain".to_owned()),
            filename: Some("a.txt".to_owned()),
            metadata: metadata(&[("k", json!(true))]),
            ..Part::default()
        },
        Part {
            raw: Some("AAEC".to_owned()),
            media_type: Some("application/octet-stream".to_owned()),
            metadata: metadata(&[("k", json!([1, 2]))]),
            ..Part::default()
        },
    ];
    for part in parts {
        let block = part_to_content_block(&part, "b1");
        assert_eq!(content_block_to_part(&block), part, "{part:?}");
    }
}

#[test]
fn text_block_reads_a_string_or_a_text_member_only() {
    let string_block = block("b1", "text", json!("plain"));
    assert_eq!(content_block_to_part(&string_block), Part::text("plain"));

    let member_block = block("b2", "text", json!({ "text": "member" }));
    assert_eq!(content_block_to_part(&member_block), Part::text("member"));

    let lossy_block = block("b3", "text", json!({ "text": 5 }));
    assert_eq!(content_block_to_part(&lossy_block), Part::default());
}

#[test]
fn empty_block_and_metadata_map_both_ways() {
    let empty = ContentBlock {
        block_id: "b1".to_owned(),
        kind: "empty".to_owned(),
        content: Value::Null,
        data: json_object(&[("signature", json!("s"))]),
    };
    let part = content_block_to_part(&empty);
    assert!(part.text.is_none());
    assert!(part.raw.is_none());
    assert!(part.url.is_none());
    assert!(part.data.is_none());
    assert_eq!(part.metadata.get("signature"), Some(&json!("s")));
}

#[test]
fn unknown_block_kind_falls_back_to_a_data_part() {
    let unknown = ContentBlock {
        block_id: "b1".to_owned(),
        kind: "reasoning".to_owned(),
        content: json!({ "text": "thinking", "signature": "s" }),
        data: json_object(&[("provider", json!("anthropic"))]),
    };
    let part = content_block_to_part(&unknown);
    assert_eq!(
        part.data,
        Some(json!({ "text": "thinking", "signature": "s" }))
    );
    assert_eq!(part.metadata.get("provider"), Some(&json!("anthropic")));
}

#[test]
fn file_block_with_a_non_object_content_yields_an_empty_part() {
    let file = block("b1", "file", json!("https://example.test/a.txt"));
    assert_eq!(content_block_to_part(&file), Part::default());
}

#[test]
fn turn_message_becomes_a_named_artifact() {
    let message = TurnMessage {
        message_id: "m1".to_owned(),
        role: MessageRole::Tool,
        content: vec![
            block("b1", "text", json!({ "text": "done" })),
            block("b2", "reasoning", json!({ "text": "why" })),
        ],
        data: json_object(&[("name", json!("search"))]),
    };
    let artifact = turn_message_to_artifact(&message);
    assert_eq!(artifact.artifact_id.as_deref(), Some("m1"));
    assert_eq!(artifact.name.as_deref(), Some("tool"));
    assert!(artifact.description.is_none());
    assert_eq!(
        artifact.parts,
        vec![
            AgentPart {
                text: Some("done".to_owned()),
                ..AgentPart::default()
            },
            AgentPart {
                data: Some(json!({ "text": "why" })),
                ..AgentPart::default()
            },
        ]
    );
    assert_eq!(artifact.metadata.get("name"), Some(&json!("search")));
}

#[test]
fn message_role_names_are_portable() {
    for (role, name) in [
        (MessageRole::User, "user"),
        (MessageRole::Assistant, "assistant"),
        (MessageRole::System, "system"),
        (MessageRole::Tool, "tool"),
    ] {
        assert_eq!(message_role_name(role), name);
    }
}

#[test]
fn completed_turn_becomes_a_completed_result_with_artifacts() {
    let result = TurnResult {
        status: TurnStatus::Completed,
        messages: vec![
            TurnMessage {
                message_id: "m1".to_owned(),
                role: MessageRole::User,
                content: vec![block("b1", "text", json!({ "text": "hi" }))],
                data: json_object(&[("trace", json!("t1"))]),
            },
            TurnMessage {
                message_id: "m2".to_owned(),
                role: MessageRole::Assistant,
                content: vec![block("b2", "text", json!({ "text": "hello" }))],
                data: JsonObject::new(),
            },
        ],
        stop_reason: Some("stop".to_owned()),
        final_output: json!("hello"),
        structured_output: json!({ "answer": "hello" }),
        duration_ms: Some(12),
        started_at: Some(1.5),
        completed_at: Some(1.75),
        ..turn_result(TurnStatus::Completed)
    };
    let agent = turn_result_to_agent_result(&result);

    assert_eq!(agent.status, AgentStatus::Completed);
    assert!(agent.task_id.is_none());
    assert!(agent.session_id.is_none());
    assert_eq!(agent.artifacts.len(), 2);
    assert_eq!(agent.artifacts[0].artifact_id.as_deref(), Some("m1"));
    assert_eq!(agent.artifacts[0].name.as_deref(), Some("user"));
    assert_eq!(agent.artifacts[0].metadata.get("trace"), Some(&json!("t1")));
    assert_eq!(agent.artifacts[1].name.as_deref(), Some("assistant"));
    assert_eq!(
        agent.artifacts[1].parts,
        vec![AgentPart {
            text: Some("hello".to_owned()),
            ..AgentPart::default()
        }]
    );

    assert_eq!(agent.metadata.get("stopReason"), Some(&json!("stop")));
    assert_eq!(agent.metadata.get("finalOutput"), Some(&json!("hello")));
    assert_eq!(
        agent.metadata.get("structuredOutput"),
        Some(&json!({ "answer": "hello" }))
    );
    assert_eq!(agent.metadata.get("durationMs"), Some(&json!(12)));
    assert_eq!(agent.metadata.get("startedAt"), Some(&json!(1.5)));
    assert_eq!(agent.metadata.get("completedAt"), Some(&json!(1.75)));
    assert!(!agent.metadata.contains_key("error"));
    assert!(!agent.metadata.contains_key("termination"));
    assert!(!agent.metadata.contains_key("usage"));
    assert!(!agent.metadata.contains_key("cost"));
}

#[test]
fn absent_and_null_turn_fields_are_not_copied() {
    let agent = turn_result_to_agent_result(&turn_result(TurnStatus::Completed));
    assert_eq!(agent.status, AgentStatus::Completed);
    assert_eq!(agent.artifacts, Vec::<AgentArtifact>::new());
    assert!(agent.metadata.is_empty(), "{:?}", agent.metadata);
}

#[test]
fn failed_turn_becomes_failed_with_the_error_object() {
    let result = TurnResult {
        status: TurnStatus::Failed,
        error: Some(TurnError {
            message: "boom".to_owned(),
            code: Some("E_BOOM".to_owned()),
            category: None,
            retryable: Some(true),
            provider_data: JsonObject::new(),
        }),
        ..turn_result(TurnStatus::Failed)
    };
    let agent = turn_result_to_agent_result(&result);
    assert_eq!(agent.status, AgentStatus::Failed);
    assert_eq!(agent.metadata["error"]["message"], "boom");
    assert_eq!(agent.metadata["error"]["code"], "E_BOOM");
    assert_eq!(agent.metadata["error"]["retryable"], json!(true));
    assert!(!agent.metadata.contains_key("termination"));
}

#[test]
fn interrupted_turn_becomes_input_required_with_the_termination() {
    let result = TurnResult {
        status: TurnStatus::Interrupted,
        termination: Some(TurnTermination {
            kind: TurnTerminationKind::Timeout,
            message: Some("too slow".to_owned()),
            code: None,
            provider_data: JsonObject::new(),
        }),
        ..turn_result(TurnStatus::Interrupted)
    };
    let agent = turn_result_to_agent_result(&result);
    assert_eq!(agent.status, AgentStatus::InputRequired);
    assert_eq!(agent.metadata["termination"]["kind"], "timeout");
    assert_eq!(agent.metadata["termination"]["message"], "too slow");
    assert!(!agent.metadata.contains_key("error"));
}

#[test]
fn usage_and_cost_are_copied_as_objects() {
    let result = TurnResult {
        status: TurnStatus::Completed,
        usage: Some(TurnUsage {
            input_tokens: Some(10),
            output_tokens: Some(2),
            ..TurnUsage::default()
        }),
        cost: Some(MonetaryAmount::new(1_500, "usd")),
        ..turn_result(TurnStatus::Completed)
    };
    let agent = turn_result_to_agent_result(&result);
    assert_eq!(agent.metadata["usage"]["input_tokens"], json!(10));
    assert_eq!(agent.metadata["usage"]["output_tokens"], json!(2));
    assert_eq!(agent.metadata["cost"]["micros"], json!(1_500));
    assert_eq!(agent.metadata["cost"]["currency"], "USD");
}

#[test]
fn provider_data_wins_over_derived_metadata() {
    let result = TurnResult {
        status: TurnStatus::Completed,
        stop_reason: Some("stop".to_owned()),
        final_output: json!("derived"),
        duration_ms: Some(7),
        provider_data: json_object(&[
            ("stopReason", json!("provider")),
            ("finalOutput", json!("provider output")),
            ("vendor", json!({ "id": "v1" })),
        ]),
        ..turn_result(TurnStatus::Completed)
    };
    let agent = turn_result_to_agent_result(&result);
    assert_eq!(agent.metadata.get("stopReason"), Some(&json!("provider")));
    assert_eq!(
        agent.metadata.get("finalOutput"),
        Some(&json!("provider output"))
    );
    assert_eq!(agent.metadata.get("vendor"), Some(&json!({ "id": "v1" })));
    assert_eq!(agent.metadata.get("durationMs"), Some(&json!(7)));
}

#[test]
fn agent_status_maps_onto_the_task_state_table() {
    for (status, state) in [
        (AgentStatus::Submitted, TaskState::Submitted),
        (AgentStatus::Working, TaskState::Working),
        (AgentStatus::Paused, TaskState::Working),
        (AgentStatus::InputRequired, TaskState::InputRequired),
        (AgentStatus::Completed, TaskState::Completed),
        (AgentStatus::Canceled, TaskState::Canceled),
        (AgentStatus::Failed, TaskState::Failed),
        (AgentStatus::Waiting, TaskState::Working),
        (AgentStatus::Unknown, TaskState::Unspecified),
    ] {
        assert_eq!(agent_status_to_task_state(status), state, "{status:?}");
    }
}

#[test]
fn task_id_precedence_is_argument_then_result_then_fresh() {
    let with_id = AgentResult {
        task_id: Some("result-task".to_owned()),
        ..AgentResult::default()
    };
    assert_eq!(
        agent_result_to_task(&with_id, Some("explicit"), "ctx").id,
        "explicit"
    );
    assert_eq!(
        agent_result_to_task(&with_id, None, "ctx").id,
        "result-task"
    );

    let without_id = AgentResult::default();
    let task = agent_result_to_task(&without_id, None, "ctx");
    assert_hex_id(&task.id);
}

#[test]
fn task_carries_status_context_artifacts_and_metadata() {
    let result = AgentResult {
        task_id: Some("result-task".to_owned()),
        session_id: Some("s1".to_owned()),
        status: AgentStatus::Working,
        artifacts: vec![
            AgentArtifact {
                artifact_id: Some("a1".to_owned()),
                name: Some("summary".to_owned()),
                description: Some("the summary".to_owned()),
                parts: vec![AgentPart {
                    text: Some("x".to_owned()),
                    ..AgentPart::default()
                }],
                metadata: metadata(&[("k", json!("v"))]),
            },
            AgentArtifact {
                parts: vec![AgentPart {
                    data: Some(json!({ "answer": 1 })),
                    ..AgentPart::default()
                }],
                ..AgentArtifact::default()
            },
        ],
        metadata: metadata(&[("turn", json!(3))]),
    };
    let task = agent_result_to_task(&result, Some("task-1"), "ctx-1");

    assert_eq!(task.id, "task-1");
    assert_eq!(task.context_id, "ctx-1");
    assert_eq!(task.status.state, TaskState::Working);
    assert!(task.status.message.is_none());
    assert!(task.status.timestamp.is_none());
    assert_eq!(task.history, Vec::<Message>::new());
    assert_eq!(task.metadata.get("turn"), Some(&json!(3)));

    assert_eq!(task.artifacts.len(), 2);
    assert_eq!(task.artifacts[0].artifact_id, "a1");
    assert_eq!(task.artifacts[0].name.as_deref(), Some("summary"));
    assert_eq!(task.artifacts[0].parts, vec![Part::text("x")]);
    assert_eq!(task.artifacts[0].metadata.get("k"), Some(&json!("v")));
    assert_hex_id(&task.artifacts[1].artifact_id);
    assert_eq!(
        task.artifacts[1].parts,
        vec![Part {
            data: Some(json!({ "answer": 1 })),
            ..Part::default()
        }]
    );
}

#[test]
fn unknown_status_produces_an_unspecified_task_state() {
    let task = agent_result_to_task(&AgentResult::default(), None, "ctx");
    assert_eq!(task.status.state, TaskState::Unspecified);
    assert_eq!(task.artifacts, Vec::<Artifact>::new());
    assert!(task.metadata.is_empty());
}
