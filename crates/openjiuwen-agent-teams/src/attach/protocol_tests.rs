#![allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
//! Unit tests for the attach wire contract: framing, serde, validation and frame IO.
//!
//! Everything runs offline with in-memory readers/writers; `tokio::io::duplex` exercises the
//! shared frame helpers without a socket.

use std::sync::Arc;

use openjiuwen_common::error::{Error, ErrorKind};
use serde_json::{json, Value};
use tokio::io::BufReader;

use super::protocol::{read_frame_line, to_data, write_all_line};
use super::*;
use crate::schema::{MemberSnapshot, TaskStatus};

/// Every op with its expected `name()`, one sample per variant.
fn sample_ops() -> Vec<(AttachOp, &'static str)> {
    vec![
        (AttachOp::ListMembers, "list_members"),
        (
            AttachOp::MemberStatus {
                member_name: "dev-1".to_owned(),
            },
            "member_status",
        ),
        (AttachOp::ReadInbox, "read_inbox"),
        (
            AttachOp::SendMember {
                to: "dev-2".to_owned(),
                content: "hello".to_owned(),
            },
            "send_member",
        ),
        (
            AttachOp::Broadcast {
                content: "hello all".to_owned(),
            },
            "broadcast",
        ),
        (
            AttachOp::ListTasks {
                status: Some(TaskStatus::Pending),
            },
            "list_tasks",
        ),
        (AttachOp::ClaimableTasks, "claimable_tasks"),
        (
            AttachOp::TaskDetail {
                task_id: "task-1".to_owned(),
            },
            "task_detail",
        ),
        (
            AttachOp::CreateTask {
                title: "Write tests".to_owned(),
                content: "Cover the protocol".to_owned(),
                task_id: Some("task-2".to_owned()),
                dependencies: vec!["task-1".to_owned()],
            },
            "create_task",
        ),
        (
            AttachOp::UpdateTask {
                task_id: "task-1".to_owned(),
                title: Some("Rename".to_owned()),
                content: None,
            },
            "update_task",
        ),
        (
            AttachOp::ClaimTask {
                task_id: "task-1".to_owned(),
            },
            "claim_task",
        ),
        (
            AttachOp::CompleteTask {
                task_id: "task-1".to_owned(),
            },
            "complete_task",
        ),
        (
            AttachOp::InvokeTool {
                name: "read_file".to_owned(),
                arguments: json!({"path": "a.txt"}),
            },
            "invoke_tool",
        ),
        (AttachOp::Snapshot, "snapshot"),
        (AttachOp::StopTeam, "stop_team"),
    ]
}

/// A handler double proving the trait is object safe and errors stay typed.
struct FakeHandler;

#[async_trait::async_trait]
impl AttachHandler for FakeHandler {
    async fn handle(&self, op: AttachOp) -> AttachResult<Value> {
        match op {
            AttachOp::Snapshot => Ok(json!({"team_name": "demo"})),
            other => Err(AttachError::remote(
                ErrorKind::NotFound,
                format!("no handler for `{}`", other.name()),
            )),
        }
    }
}

#[test]
fn request_wire_bytes_are_stable() {
    let frame = RequestFrame::new(7, &AttachOp::ListMembers).unwrap();
    let expected = b"{\"v\":1,\"id\":7,\"op\":\"list_members\",\"args\":{}}\n";
    assert_eq!(frame.encode().unwrap(), expected);
    assert_eq!(RequestFrame::decode(expected).unwrap(), frame);
    // A decoder also accepts the line without its trailing newline.
    assert_eq!(
        RequestFrame::decode(&expected[..expected.len() - 1]).unwrap(),
        frame
    );
}

#[test]
fn response_ok_wire_bytes_are_stable() {
    let frame = ResponseFrame::ok(7, json!({"team_name": "demo"}));
    let expected = b"{\"v\":1,\"id\":7,\"ok\":true,\"data\":{\"team_name\":\"demo\"}}\n";
    assert_eq!(frame.encode().unwrap(), expected);
    assert_eq!(ResponseFrame::decode(expected).unwrap(), frame);
}

#[test]
fn response_error_wire_bytes_are_stable() {
    let frame = ResponseFrame::err(7, AttachError::new(ErrorKind::NotFound, "task missing"));
    let expected =
        b"{\"v\":1,\"id\":7,\"ok\":false,\"error\":{\"kind\":\"not_found\",\"message\":\"task missing\"}}\n";
    assert_eq!(frame.encode().unwrap(), expected);
    assert_eq!(ResponseFrame::decode(expected).unwrap(), frame);
}

#[test]
fn every_op_round_trips_through_frames() {
    for (op, name) in sample_ops() {
        assert_eq!(op.name(), name);

        let args = op.args().unwrap();
        assert!(args.is_object(), "args for `{name}` must be an object");
        assert_eq!(AttachOp::from_parts(name, args).unwrap(), op);

        let frame = RequestFrame::new(42, &op).unwrap();
        assert_eq!(frame.v, PROTOCOL_VERSION);
        assert_eq!(frame.id, 42);
        assert_eq!(frame.op, name);
        assert_eq!(frame.op().unwrap(), op);

        let decoded = RequestFrame::decode(&frame.encode().unwrap()).unwrap();
        assert_eq!(decoded, frame);
        assert_eq!(decoded.op().unwrap(), op);
    }
}

#[test]
fn unit_variants_carry_empty_args() {
    for op in [
        AttachOp::ListMembers,
        AttachOp::ReadInbox,
        AttachOp::ClaimableTasks,
        AttachOp::Snapshot,
        AttachOp::StopTeam,
    ] {
        assert_eq!(op.args().unwrap(), json!({}));
        let name = op.name();
        assert_eq!(AttachOp::from_parts(name, json!({})).unwrap(), op);
    }
}

#[test]
fn list_tasks_status_is_optional_and_round_trips() {
    let all = AttachOp::ListTasks { status: None };
    assert_eq!(all.args().unwrap(), json!({"status": null}));
    assert_eq!(
        AttachOp::from_parts("list_tasks", all.args().unwrap()).unwrap(),
        all
    );
}

#[test]
fn unknown_request_fields_are_tolerated() {
    let line = br#"{"v":1,"id":3,"op":"read_inbox","args":{},"future":"ignored"}"#;
    let frame = RequestFrame::decode(line).unwrap();
    assert_eq!(frame.id, 3);
    assert_eq!(frame.op().unwrap(), AttachOp::ReadInbox);
}

#[test]
fn unknown_response_and_error_fields_are_tolerated() {
    let line = br#"{"v":1,"id":3,"ok":true,"data":null,"future":"ignored"}"#;
    assert!(ResponseFrame::decode(line).unwrap().ok);

    let error: AttachError =
        serde_json::from_str(r#"{"kind":"validation","message":"bad","detail":"ignored"}"#)
            .unwrap();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert_eq!(error.message, "bad");
}

#[test]
fn missing_args_defaults_to_empty_object() {
    let frame = RequestFrame::decode(br#"{"v":1,"id":1,"op":"snapshot"}"#).unwrap();
    assert_eq!(frame.args, json!({}));
    assert_eq!(frame.op().unwrap(), AttachOp::Snapshot);
}

#[test]
fn null_args_becomes_empty_object() {
    let frame = RequestFrame::decode(br#"{"v":1,"id":1,"op":"snapshot","args":null}"#).unwrap();
    assert_eq!(frame.args, json!({}));
}

#[test]
fn non_object_args_are_rejected() {
    let error = RequestFrame::decode(br#"{"v":1,"id":1,"op":"read_inbox","args":[]}"#).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("args"));
}

#[test]
fn malformed_json_is_a_protocol_error() {
    let error = RequestFrame::decode(b"this is not json\n").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("malformed request frame"));

    let error = ResponseFrame::decode(b"{]\n").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("malformed response frame"));
}

#[test]
fn unsupported_version_is_rejected_by_both_decoders() {
    let error = RequestFrame::decode(br#"{"v":2,"id":1,"op":"read_inbox","args":{}}"#).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("version"));

    let error = ResponseFrame::decode(br#"{"v":9,"id":1,"ok":true}"#).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("version"));
}

#[test]
fn missing_version_is_rejected() {
    let error = RequestFrame::decode(br#"{"id":1,"op":"read_inbox","args":{}}"#).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    let error = ResponseFrame::decode(br#"{"id":1,"ok":true}"#).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
}

#[test]
fn unknown_op_names_the_op() {
    // The envelope decodes without interpreting the op; the typed conversion rejects it.
    let frame = RequestFrame::decode(br#"{"v":1,"id":1,"op":"frobnicate","args":{}}"#).unwrap();
    let error = frame.op().unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("frobnicate"));

    let error = AttachOp::from_parts("frobnicate", json!({})).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("frobnicate"));
}

#[test]
fn unknown_op_with_non_object_args_names_the_op() {
    let error = AttachOp::from_parts("frobnicate", json!([1, 2])).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("frobnicate"));
    assert!(error.message.contains("JSON object"));
}

#[test]
fn wrong_field_type_is_a_protocol_error() {
    let error = AttachOp::from_parts("member_status", json!({"member_name": 7})).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("member_status"));
}

#[test]
fn empty_required_fields_are_validation_errors() {
    let cases = [
        (
            AttachOp::MemberStatus {
                member_name: String::new(),
            },
            "member_name",
        ),
        (
            AttachOp::SendMember {
                to: String::new(),
                content: "hi".to_owned(),
            },
            "to",
        ),
        (
            AttachOp::TaskDetail {
                task_id: "  ".to_owned(),
            },
            "task_id",
        ),
        (
            AttachOp::UpdateTask {
                task_id: String::new(),
                title: None,
                content: None,
            },
            "task_id",
        ),
        (
            AttachOp::ClaimTask {
                task_id: String::new(),
            },
            "task_id",
        ),
        (
            AttachOp::CompleteTask {
                task_id: String::new(),
            },
            "task_id",
        ),
        (
            AttachOp::CreateTask {
                title: String::new(),
                content: "body".to_owned(),
                task_id: None,
                dependencies: Vec::new(),
            },
            "title",
        ),
        (
            AttachOp::InvokeTool {
                name: String::new(),
                arguments: Value::Null,
            },
            "name",
        ),
    ];
    for (op, field) in cases {
        let error = op.validate().unwrap_err();
        assert_eq!(error.kind, ErrorKind::Validation);
        assert!(error.message.contains(field), "message: {}", error.message);
        assert!(
            error.message.contains(op.name()),
            "message: {}",
            error.message
        );
        assert!(RequestFrame::new(1, &op).is_err());
    }
}

#[test]
fn error_serde_round_trips() {
    let error = AttachError::new(ErrorKind::Transport, "socket closed");
    let text = serde_json::to_string(&error).unwrap();
    assert_eq!(text, r#"{"kind":"transport","message":"socket closed"}"#);
    let back: AttachError = serde_json::from_str(&text).unwrap();
    assert_eq!(back, error);
}

#[test]
fn unknown_error_kind_falls_back_to_external() {
    let error: AttachError =
        serde_json::from_str(r#"{"kind":"banana","message":"rotten"}"#).unwrap();
    assert_eq!(error.kind, ErrorKind::External);
    assert_eq!(error.message, "rotten");
}

#[test]
fn missing_error_fields_default() {
    let error: AttachError = serde_json::from_str("{}").unwrap();
    assert_eq!(error.kind, ErrorKind::External);
    assert_eq!(error.message, "");
}

#[test]
fn error_constructors_pick_the_right_kind() {
    assert_eq!(AttachError::protocol("x").kind, ErrorKind::Validation);
    assert_eq!(AttachError::transport("x").kind, ErrorKind::Transport);
    assert_eq!(AttachError::timeout("x").kind, ErrorKind::Timeout);
    assert_eq!(
        AttachError::remote(ErrorKind::NotFound, "x").kind,
        ErrorKind::NotFound
    );
}

#[test]
fn error_display_and_std_error() {
    let error = AttachError::new(ErrorKind::Timeout, "deadline");
    assert_eq!(error.to_string(), "timeout: deadline");
    assert_eq!(
        AttachError::new(ErrorKind::Cancelled, "").to_string(),
        "cancelled"
    );
    let dynamic: &dyn std::error::Error = &error;
    assert!(dynamic.source().is_none());
}

#[test]
fn from_error_preserves_kind() {
    let cases = [
        (Error::transport("x"), ErrorKind::Transport),
        (Error::timeout("x"), ErrorKind::Timeout),
        (Error::validation("x"), ErrorKind::Validation),
        (Error::not_found("x"), ErrorKind::NotFound),
        (Error::config("x"), ErrorKind::Config),
        (Error::unsupported("x"), ErrorKind::Unsupported),
        (Error::cancelled(), ErrorKind::Cancelled),
        (
            Error::external(std::io::Error::other("boom")),
            ErrorKind::External,
        ),
    ];
    for (error, kind) in cases {
        let attach = AttachError::from(error);
        assert_eq!(attach.kind, kind);
    }
}

#[test]
fn into_error_preserves_kind_and_message() {
    let kinds = [
        ErrorKind::Config,
        ErrorKind::Validation,
        ErrorKind::Tool,
        ErrorKind::Transport,
        ErrorKind::Timeout,
        ErrorKind::NotFound,
        ErrorKind::Unsupported,
        ErrorKind::Cancelled,
        ErrorKind::External,
    ];
    for kind in kinds {
        let error = AttachError::new(kind, "boom").into_error();
        assert_eq!(error.kind(), kind);
        if kind != ErrorKind::Cancelled {
            assert_eq!(error.message(), "boom");
        }
    }
    let converted: Error = AttachError::new(ErrorKind::Config, "no team").into();
    assert_eq!(converted.kind(), ErrorKind::Config);
}

#[test]
fn error_response_without_error_is_rejected() {
    let error = ResponseFrame::decode(br#"{"v":1,"id":2,"ok":false}"#).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains("error"));

    let manual = ResponseFrame {
        v: PROTOCOL_VERSION,
        id: 2,
        ok: false,
        data: None,
        error: None,
    };
    assert!(manual.into_result().is_err());
}

#[test]
fn success_response_without_data_is_null() {
    let frame = ResponseFrame::decode(br#"{"v":1,"id":9,"ok":true}"#).unwrap();
    assert_eq!(frame.into_result().unwrap(), Value::Null);
}

#[test]
fn response_error_decodes_into_result() {
    let frame = ResponseFrame::err(3, AttachError::remote(ErrorKind::Config, "no team"));
    assert_eq!(frame.v, PROTOCOL_VERSION);
    let error = ResponseFrame::decode(&frame.encode().unwrap())
        .unwrap()
        .into_result()
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Config);
    assert_eq!(error.message, "no team");
}

#[tokio::test]
async fn read_frame_line_reports_eof() {
    let mut reader = BufReader::new(tokio::io::empty());
    assert!(read_frame_line(&mut reader).await.unwrap().is_none());
}

#[tokio::test]
async fn read_frame_line_returns_a_partial_last_line() {
    let data = b"tail without newline".to_vec();
    let mut reader = BufReader::new(&data[..]);
    let line = read_frame_line(&mut reader).await.unwrap().unwrap();
    assert_eq!(line, data);
}

#[tokio::test]
async fn read_frame_line_rejects_an_oversized_line() {
    let mut data = vec![b'x'; MAX_FRAME_BYTES + 1];
    data.extend_from_slice(b"\n{\"v\":1,\"id\":1,\"op\":\"snapshot\",\"args\":{}}\n");
    let mut reader = BufReader::new(&data[..]);

    let error = read_frame_line(&mut reader).await.unwrap_err();
    assert_eq!(error.kind, ErrorKind::Validation);
    assert!(error.message.contains(&MAX_FRAME_BYTES.to_string()));

    // The rest of the oversized line was drained, so the next frame decodes normally.
    let next = read_frame_line(&mut reader).await.unwrap().unwrap();
    assert_eq!(RequestFrame::decode(&next).unwrap().id, 1);
}

#[tokio::test]
async fn frame_io_round_trips_over_duplex() {
    let (mut client, server) = tokio::io::duplex(4096);
    let frame = RequestFrame::new(
        11,
        &AttachOp::Broadcast {
            content: "hello".to_owned(),
        },
    )
    .unwrap();
    let bytes = frame.encode().unwrap();
    write_all_line(&mut client, &bytes).await.unwrap();

    let mut reader = BufReader::new(server);
    let line = read_frame_line(&mut reader).await.unwrap().unwrap();
    assert_eq!(line, bytes);
    assert_eq!(RequestFrame::decode(&line).unwrap(), frame);
}

#[test]
fn to_data_serializes_schema_values() {
    assert_eq!(to_data(&TaskStatus::Pending).unwrap(), json!("pending"));
    let member = MemberSnapshot {
        member_name: "dev-1".to_owned(),
        display_name: "Dev One".to_owned(),
        ..MemberSnapshot::default()
    };
    let data = to_data(&member).unwrap();
    assert_eq!(data["member_name"], "dev-1");
    assert!(data.is_object());
}

#[tokio::test]
async fn handler_is_object_safe_and_preserves_errors() {
    let handler: Arc<dyn AttachHandler> = Arc::new(FakeHandler);

    let data = handler.handle(AttachOp::Snapshot).await.unwrap();
    assert_eq!(data["team_name"], "demo");

    let error = handler.handle(AttachOp::StopTeam).await.unwrap_err();
    assert_eq!(error.kind, ErrorKind::NotFound);
    assert!(error.message.contains("stop_team"));
}
