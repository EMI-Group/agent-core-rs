//! Pure encode/decode tests for the JSON-lines member protocol.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_harness_protocol::HarnessState;
use serde_json::json;

use super::protocol::{MemberRequest, MemberResponse};
use crate::schema::RoundOutcome;

/// Asserts the exact wire text, the line form, and both parse paths for a request frame.
fn assert_request_round_trip(frame: &MemberRequest, wire: &str) {
    assert_eq!(frame.to_json().unwrap(), wire);
    assert_eq!(frame.to_line().unwrap(), format!("{wire}\n"));
    assert_eq!(&MemberRequest::parse_line(wire).unwrap(), frame);
    assert_eq!(
        &MemberRequest::parse_line(&frame.to_line().unwrap()).unwrap(),
        frame
    );
}

/// Asserts the exact wire text, the line form, and both parse paths for a response frame.
fn assert_response_round_trip(frame: &MemberResponse, wire: &str) {
    assert_eq!(frame.to_json().unwrap(), wire);
    assert_eq!(frame.to_line().unwrap(), format!("{wire}\n"));
    assert_eq!(&MemberResponse::parse_line(wire).unwrap(), frame);
    assert_eq!(
        &MemberResponse::parse_line(&frame.to_line().unwrap()).unwrap(),
        frame
    );
}

// --- requests -------------------------------------------------------------------------------

#[test]
fn start_request_round_trips() {
    assert_request_round_trip(
        &MemberRequest::Start {
            session_id: "team-session".into(),
        },
        r#"{"type":"start","session_id":"team-session"}"#,
    );
}

#[test]
fn send_request_round_trips() {
    assert_request_round_trip(
        &MemberRequest::Send {
            content: "hi".into(),
            immediate: false,
        },
        r#"{"type":"send","content":"hi","immediate":false}"#,
    );
    assert_request_round_trip(
        &MemberRequest::Send {
            content: "now".into(),
            immediate: true,
        },
        r#"{"type":"send","content":"now","immediate":true}"#,
    );
}

#[test]
fn abort_request_round_trips() {
    assert_request_round_trip(
        &MemberRequest::Abort { immediate: false },
        r#"{"type":"abort","immediate":false}"#,
    );
    assert_request_round_trip(
        &MemberRequest::Abort { immediate: true },
        r#"{"type":"abort","immediate":true}"#,
    );
}

#[test]
fn pause_request_round_trips() {
    assert_request_round_trip(&MemberRequest::Pause, r#"{"type":"pause"}"#);
}

#[test]
fn resume_request_round_trips() {
    assert_request_round_trip(
        &MemberRequest::Resume {
            query: Some("keep going".into()),
        },
        r#"{"type":"resume","query":"keep going"}"#,
    );
}

#[test]
fn resume_request_defaults_to_no_query() {
    let parsed = MemberRequest::parse_line(r#"{"type":"resume"}"#).unwrap();
    assert_eq!(parsed, MemberRequest::Resume { query: None });
    assert_eq!(
        parsed.to_json().unwrap(),
        r#"{"type":"resume","query":null}"#
    );
}

#[test]
fn stop_request_round_trips() {
    assert_request_round_trip(&MemberRequest::Stop, r#"{"type":"stop"}"#);
}

#[test]
fn send_request_defaults_content_and_immediate() {
    let parsed = MemberRequest::parse_line(r#"{"type":"send"}"#).unwrap();
    assert_eq!(
        parsed,
        MemberRequest::Send {
            content: String::new(),
            immediate: false,
        }
    );
}

#[test]
fn abort_request_defaults_immediate() {
    let parsed = MemberRequest::parse_line(r#"{"type":"abort"}"#).unwrap();
    assert_eq!(parsed, MemberRequest::Abort { immediate: false });
}

#[test]
fn request_frame_types() {
    let cases = [
        (
            MemberRequest::Start {
                session_id: "s".into(),
            },
            "start",
        ),
        (
            MemberRequest::Send {
                content: String::new(),
                immediate: false,
            },
            "send",
        ),
        (MemberRequest::Abort { immediate: false }, "abort"),
        (MemberRequest::Pause, "pause"),
        (MemberRequest::Resume { query: None }, "resume"),
        (MemberRequest::Stop, "stop"),
    ];
    for (frame, expected) in cases {
        assert_eq!(frame.frame_type(), expected);
        // The tag emitted by serde and the tag reported by `frame_type` must agree.
        assert!(frame
            .to_json()
            .unwrap()
            .contains(&format!("\"type\":\"{expected}\"")));
    }
}

// --- responses ------------------------------------------------------------------------------

#[test]
fn ready_response_round_trips() {
    assert_response_round_trip(
        &MemberResponse::Ready {
            session_id: "child-session".into(),
        },
        r#"{"type":"ready","session_id":"child-session"}"#,
    );
}

#[test]
fn state_response_round_trips_all_values() {
    let cases = [
        (HarnessState::Idle, "idle"),
        (HarnessState::Running, "running"),
        (HarnessState::Pausing, "pausing"),
        (HarnessState::Paused, "paused"),
        (HarnessState::Terminated, "terminated"),
    ];
    for (state, wire) in cases {
        let frame = MemberResponse::State { state };
        assert_response_round_trip(&frame, &format!(r#"{{"type":"state","state":"{wire}"}}"#));
    }
}

#[test]
fn round_response_round_trips_all_values() {
    let cases = [
        (RoundOutcome::Started, "started"),
        (RoundOutcome::Finished, "finished"),
        (RoundOutcome::Aborted, "aborted"),
        (RoundOutcome::Failed, "failed"),
    ];
    for (outcome, wire) in cases {
        let frame = MemberResponse::Round { outcome };
        assert_response_round_trip(&frame, &format!(r#"{{"type":"round","outcome":"{wire}"}}"#));
    }
}

#[test]
fn chunk_response_passes_json_through() {
    let chunk = json!({"delta": "hello", "index": 1, "nested": {"ok": true}});
    let frame = MemberResponse::Chunk {
        chunk: chunk.clone(),
    };
    let wire = frame.to_json().unwrap();
    let parsed = MemberResponse::parse_line(&wire).unwrap();
    match parsed {
        MemberResponse::Chunk { chunk: seen } => assert_eq!(seen, chunk),
        other => panic!("expected a chunk frame, got {other:?}"),
    }
    assert_eq!(
        MemberResponse::parse_line(r#"{"type":"chunk","chunk":{"delta":"hello"}}"#).unwrap(),
        MemberResponse::Chunk {
            chunk: json!({"delta": "hello"})
        }
    );
}

#[test]
fn error_response_carries_message() {
    assert_response_round_trip(
        &MemberResponse::Error {
            message: "start refused".into(),
        },
        r#"{"type":"error","message":"start refused"}"#,
    );
}

#[test]
fn response_frame_types() {
    let cases = [
        (
            MemberResponse::Ready {
                session_id: "s".into(),
            },
            "ready",
        ),
        (
            MemberResponse::State {
                state: HarnessState::Idle,
            },
            "state",
        ),
        (
            MemberResponse::Round {
                outcome: RoundOutcome::Started,
            },
            "round",
        ),
        (MemberResponse::Chunk { chunk: json!({}) }, "chunk"),
        (
            MemberResponse::Error {
                message: String::new(),
            },
            "error",
        ),
    ];
    for (frame, expected) in cases {
        assert_eq!(frame.frame_type(), expected);
        assert!(frame
            .to_json()
            .unwrap()
            .contains(&format!("\"type\":\"{expected}\"")));
    }
}

// --- tolerance and rejection ----------------------------------------------------------------

#[test]
fn unknown_fields_are_tolerated() {
    assert_eq!(
        MemberRequest::parse_line(r#"{"type":"stop","extra":1,"more":{"a":true}}"#).unwrap(),
        MemberRequest::Stop
    );
    assert_eq!(
        MemberResponse::parse_line(r#"{"type":"state","state":"idle","extra":[1,2]}"#).unwrap(),
        MemberResponse::State {
            state: HarnessState::Idle
        }
    );
}

#[test]
fn trailing_carriage_return_is_tolerated() {
    assert_eq!(
        MemberRequest::parse_line("{\"type\":\"pause\"}\r\n").unwrap(),
        MemberRequest::Pause
    );
    assert_eq!(
        MemberRequest::parse_line("\t {\"type\":\"stop\"}  \n").unwrap(),
        MemberRequest::Stop
    );
    assert_eq!(
        MemberResponse::parse_line("{\"type\":\"ready\",\"session_id\":\"s\"}\r\n").unwrap(),
        MemberResponse::Ready {
            session_id: "s".into()
        }
    );
}

#[test]
fn blank_request_line_is_rejected() {
    for line in ["", "\n", " \r\n\t"] {
        let err = MemberRequest::parse_line(line).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Validation);
        assert!(
            err.to_string().contains("empty member request frame"),
            "unexpected message: {err}"
        );
    }
}

#[test]
fn blank_response_line_is_rejected() {
    for line in ["", "\n", "  "] {
        let err = MemberResponse::parse_line(line).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Validation);
        assert!(
            err.to_string().contains("empty member response frame"),
            "unexpected message: {err}"
        );
    }
}

#[test]
fn unknown_request_type_is_rejected() {
    let err = MemberRequest::parse_line(r#"{"type":"bogus","content":"x"}"#).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    let message = err.to_string();
    assert!(
        message.contains("unknown member request frame type 'bogus'"),
        "unexpected message: {message}"
    );
}

#[test]
fn unknown_response_type_is_rejected() {
    let err = MemberResponse::parse_line(r#"{"type":"bogus"}"#).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(
        err.to_string()
            .contains("unknown member response frame type 'bogus'"),
        "unexpected message: {err}"
    );
}

#[test]
fn malformed_known_frame_names_the_type() {
    let err = MemberRequest::parse_line(r#"{"type":"send","content":5}"#).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(
        err.to_string()
            .contains("invalid member request frame 'send'"),
        "unexpected message: {err}"
    );
}

#[test]
fn invalid_json_is_rejected() {
    let err = MemberRequest::parse_line(r#"{"type":"send","content":"halo""#).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(
        err.to_string().contains("invalid member request frame"),
        "unexpected message: {err}"
    );
    let err = MemberResponse::parse_line("not json at all").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(
        err.to_string().contains("invalid member response frame"),
        "unexpected message: {err}"
    );
}

#[test]
fn huge_garbage_line_error_stays_bounded() {
    let ascii = "x".repeat(2000);
    let err = MemberResponse::parse_line(&ascii).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    let message = err.to_string();
    assert!(
        message.chars().count() < 300,
        "error message is {} chars long",
        message.chars().count()
    );
    assert!(message.contains('…'), "the echo should be truncated");

    // Multi-byte input must not split a char boundary or panic.
    let wide = "字".repeat(2000);
    let err = MemberRequest::parse_line(&wide).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().chars().count() < 300);
}

#[test]
fn parse_noisy_line_skips_blank_lines() {
    assert_eq!(MemberResponse::parse_noisy_line("").unwrap(), None);
    assert_eq!(MemberResponse::parse_noisy_line(" \r\n\t").unwrap(), None);
    assert_eq!(
        MemberResponse::parse_noisy_line(r#"{"type":"state","state":"paused"}"#).unwrap(),
        Some(MemberResponse::State {
            state: HarnessState::Paused
        })
    );
}

#[test]
fn parse_noisy_line_rejects_non_frame_output() {
    let err = MemberResponse::parse_line(r#"{"type":5}"#).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert_eq!(
        MemberResponse::parse_noisy_line(r#"{"type":5}"#)
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        MemberResponse::parse_noisy_line("starting up...")
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
}

// --- accessors ------------------------------------------------------------------------------

#[test]
fn state_change_reports_only_state_frames() {
    assert_eq!(
        MemberResponse::State {
            state: HarnessState::Running
        }
        .state_change(),
        Some(HarnessState::Running)
    );
    for other in [
        MemberResponse::Ready {
            session_id: "s".into(),
        },
        MemberResponse::Round {
            outcome: RoundOutcome::Started,
        },
        MemberResponse::Chunk { chunk: json!({}) },
        MemberResponse::Error {
            message: "e".into(),
        },
    ] {
        assert_eq!(other.state_change(), None);
    }
}

#[test]
fn round_outcome_reports_only_round_frames() {
    assert_eq!(
        MemberResponse::Round {
            outcome: RoundOutcome::Aborted
        }
        .round_outcome(),
        Some(RoundOutcome::Aborted)
    );
    for other in [
        MemberResponse::Ready {
            session_id: "s".into(),
        },
        MemberResponse::State {
            state: HarnessState::Idle,
        },
        MemberResponse::Chunk { chunk: json!({}) },
        MemberResponse::Error {
            message: "e".into(),
        },
    ] {
        assert_eq!(other.round_outcome(), None);
    }
}

#[test]
fn error_message_reports_only_error_frames() {
    assert_eq!(
        MemberResponse::Error {
            message: "start refused".into()
        }
        .error_message(),
        Some("start refused")
    );
    assert_eq!(
        MemberResponse::State {
            state: HarnessState::Idle
        }
        .error_message(),
        None
    );
}

#[test]
fn ready_accessors_report_the_child_session() {
    let ready = MemberResponse::Ready {
        session_id: "child-session".into(),
    };
    assert!(ready.is_ready());
    assert_eq!(ready.ready_session_id(), Some("child-session"));
    let other = MemberResponse::State {
        state: HarnessState::Idle,
    };
    assert!(!other.is_ready());
    assert_eq!(other.ready_session_id(), None);
}
