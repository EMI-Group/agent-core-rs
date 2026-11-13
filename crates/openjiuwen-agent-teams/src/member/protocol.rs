//! The JSON-lines member protocol spoken between a child-process member and its leader.
//!
//! # Framing
//!
//! Every frame is exactly one JSON object on one line, terminated by `\n`.
//! Leader-to-member frames decode to [`MemberRequest`] and arrive on the child's stdin;
//! member-to-leader frames decode to [`MemberResponse`] and arrive on the child's stdout.
//! The child must flush after every line: a frame that stays in a user-space buffer is invisible
//! to the leader, which is blocked waiting for it.
//! stderr is diagnostics only and is never parsed as protocol; a child may log there freely.
//! Blank lines on either stream are skipped, and unknown JSON fields are tolerated on every frame.
//!
//! # Requests (leader → member stdin)
//!
//! * `{"type":"start","session_id":"team-session"}` — begin the member's run.
//! * `{"type":"send","content":"hello","immediate":false}` — deliver user content to the member.
//! * `{"type":"abort","immediate":false}` — cancel the round in flight.
//! * `{"type":"pause"}` — ask the member to pause after the current step.
//! * `{"type":"resume","query":null}` — resume a paused member, optionally with fresh input.
//! * `{"type":"stop"}` — shut the member down; the child should exit `0`.
//!
//! # Responses (member → leader stdout)
//!
//! * `{"type":"ready","session_id":"child-session"}` — the member finished starting.
//! * `{"type":"state","state":"idle"}` — a harness state change (`idle`, `running`, `pausing`,
//!   `paused`, `terminated`).
//! * `{"type":"round","outcome":"started"}` — a round boundary (`started`, `finished`, `aborted`,
//!   `failed`).
//! * `{"type":"chunk","chunk":{"delta":"hello"}}` — one streaming frame; `chunk` is opaque JSON.
//! * `{"type":"error","message":"start refused"}` — the pending request failed.
//!
//! # Mapping onto the member seam
//!
//! * `ready` is consumed by [`MemberRuntime::start`](crate::member::MemberRuntime::start): it
//!   reports the child's own session id and never reaches observers.
//! * `state` maps to [`MemberObserver::on_state`](crate::member::MemberObserver::on_state).
//! * `round` maps to [`MemberObserver::on_round`](crate::member::MemberObserver::on_round), and the
//!   member's status machine walks it through
//!   [`TeamMember::note_round`](crate::member::TeamMember::note_round).
//! * `chunk` maps to [`MemberObserver::on_chunk`](crate::member::MemberObserver::on_chunk).
//! * `error` fails the pending request with an external-style detail and is logged; it does not
//!   change the member's lifecycle state, because the runtime decides whether to retry or fail it.
//! * A child exit ends the member: the runtime maps the closed stdout to
//!   [`HarnessState::Terminated`].
//!
//! # Transcript
//!
//! ```text
//! leader → {"type":"start","session_id":"team-session"}
//! leader ← {"type":"ready","session_id":"child-session"}
//! leader ← {"type":"state","state":"idle"}
//! leader → {"type":"send","content":"hello","immediate":true}
//! leader ← {"type":"round","outcome":"started"}
//! leader ← {"type":"chunk","chunk":{"delta":"hello"}}
//! leader ← {"type":"round","outcome":"finished"}
//! leader → {"type":"stop"}
//! (child exits 0)
//! ```
//!
//! ```
//! use openjiuwen_agent_teams::member::protocol::{MemberRequest, MemberResponse};
//!
//! let request = MemberRequest::Send { content: "hello".into(), immediate: false };
//! let line = request.to_line()?;
//! assert_eq!(line, "{\"type\":\"send\",\"content\":\"hello\",\"immediate\":false}\n");
//! assert_eq!(MemberRequest::parse_line(&line)?, request);
//!
//! let response = MemberResponse::parse_line("{\"type\":\"ready\",\"session_id\":\"abc\"}\r\n")?;
//! assert_eq!(response.ready_session_id(), Some("abc"));
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::schema::RoundOutcome;

/// Longest prefix of a malformed frame echoed back in an error message.
const ECHO_PREFIX_CHARS: usize = 120;

/// Wire tags of [`MemberRequest`], in declaration order.
const REQUEST_TYPES: [&str; 6] = ["start", "send", "abort", "pause", "resume", "stop"];

/// Wire tags of [`MemberResponse`], in declaration order.
const RESPONSE_TYPES: [&str; 5] = ["ready", "state", "round", "chunk", "error"];

/// One frame the leader writes to a child member's stdin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemberRequest {
    /// Begin the member's run; the child answers with a `ready` frame.
    Start {
        /// The team session the member joins.
        session_id: String,
    },
    /// Deliver user content to the member.
    Send {
        /// The content to hand to the member's agent; empty is allowed.
        #[serde(default)]
        content: String,
        /// Whether the member should interrupt the round in flight.
        #[serde(default)]
        immediate: bool,
    },
    /// Cancel the round in flight.
    Abort {
        /// Whether cancellation should skip graceful cleanup.
        #[serde(default)]
        immediate: bool,
    },
    /// Ask the member to pause after the current step.
    Pause,
    /// Resume a paused member, optionally carrying fresh input.
    Resume {
        /// Optional query handed to the member when it resumes.
        #[serde(default)]
        query: Option<String>,
    },
    /// Shut the member down; a well-behaved child exits `0`.
    Stop,
}

/// One frame a child member writes to its stdout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemberResponse {
    /// The member finished starting and owns a session.
    Ready {
        /// The session id the child is running under.
        session_id: String,
    },
    /// The member's harness state changed.
    State {
        /// The new harness state.
        state: HarnessState,
    },
    /// A member round boundary was reached.
    Round {
        /// How the round ended.
        outcome: RoundOutcome,
    },
    /// One streaming frame from the member.
    Chunk {
        /// Opaque JSON payload; the leader forwards it to observers unchanged.
        chunk: Value,
    },
    /// The pending request failed inside the child.
    Error {
        /// Human-readable failure detail.
        message: String,
    },
}

impl MemberRequest {
    /// The wire tag of this frame (`"start"`, `"send"`, …).
    #[must_use]
    pub fn frame_type(&self) -> &'static str {
        match self {
            Self::Start { .. } => "start",
            Self::Send { .. } => "send",
            Self::Abort { .. } => "abort",
            Self::Pause => "pause",
            Self::Resume { .. } => "resume",
            Self::Stop => "stop",
        }
    }

    /// Serializes the frame as one JSON object, without a trailing newline.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when the
    /// frame cannot be serialized (in practice never, for this data type).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| {
            Error::validation(format!(
                "member request frame '{}' is not serializable: {error}",
                self.frame_type()
            ))
        })
    }

    /// Serializes the frame as one newline-terminated JSON line.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when the
    /// frame cannot be serialized.
    pub fn to_line(&self) -> Result<String> {
        let mut line = self.to_json()?;
        line.push('\n');
        Ok(line)
    }

    /// Parses one line written by the leader.
    ///
    /// Trailing `\r`/`\n` and surrounding ASCII whitespace are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) for a
    /// blank line, invalid JSON, or an unknown frame `type`; the message names the offending frame
    /// type (or reports an empty frame) and echoes at most 120 characters of the line.
    pub fn parse_line(line: &str) -> Result<Self> {
        let body = frame_body(line);
        if body.is_empty() {
            return Err(Error::validation("empty member request frame"));
        }
        decode_frame(body, &REQUEST_TYPES, "member request")
    }
}

impl MemberResponse {
    /// The wire tag of this frame (`"ready"`, `"state"`, …).
    #[must_use]
    pub fn frame_type(&self) -> &'static str {
        match self {
            Self::Ready { .. } => "ready",
            Self::State { .. } => "state",
            Self::Round { .. } => "round",
            Self::Chunk { .. } => "chunk",
            Self::Error { .. } => "error",
        }
    }

    /// Serializes the frame as one JSON object, without a trailing newline.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when the
    /// frame cannot be serialized (in practice never, for this data type).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| {
            Error::validation(format!(
                "member response frame '{}' is not serializable: {error}",
                self.frame_type()
            ))
        })
    }

    /// Serializes the frame as one newline-terminated JSON line.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when the
    /// frame cannot be serialized.
    pub fn to_line(&self) -> Result<String> {
        let mut line = self.to_json()?;
        line.push('\n');
        Ok(line)
    }

    /// Parses one line written by the member.
    ///
    /// Trailing `\r`/`\n` and surrounding ASCII whitespace are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) for a
    /// blank line, invalid JSON, or an unknown frame `type`; the message names the offending frame
    /// type (or reports an empty frame) and echoes at most 120 characters of the line.
    pub fn parse_line(line: &str) -> Result<Self> {
        let body = frame_body(line);
        if body.is_empty() {
            return Err(Error::validation("empty member response frame"));
        }
        decode_frame(body, &RESPONSE_TYPES, "member response")
    }

    /// Parses one stdout line that may also be blank log noise.
    ///
    /// Children occasionally print blank lines to stdout; a blank line is not an error, it is
    /// simply not a frame.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) for a
    /// non-blank line that is not a valid member response frame.
    pub fn parse_noisy_line(line: &str) -> Result<Option<Self>> {
        if frame_body(line).is_empty() {
            return Ok(None);
        }
        Self::parse_line(line).map(Some)
    }

    /// The harness state carried by a `state` frame, if this is one.
    #[must_use]
    pub fn state_change(&self) -> Option<HarnessState> {
        match self {
            Self::State { state } => Some(*state),
            _ => None,
        }
    }

    /// The round outcome carried by a `round` frame, if this is one.
    #[must_use]
    pub fn round_outcome(&self) -> Option<RoundOutcome> {
        match self {
            Self::Round { outcome } => Some(*outcome),
            _ => None,
        }
    }

    /// The failure detail carried by an `error` frame, if this is one.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }

    /// Whether this frame is the `ready` answer to a `start` request.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }

    /// The child's session id carried by a `ready` frame, if this is one.
    #[must_use]
    pub fn ready_session_id(&self) -> Option<&str> {
        match self {
            Self::Ready { session_id } => Some(session_id),
            _ => None,
        }
    }
}

/// Strips the trailing line terminator and surrounding ASCII whitespace from one line.
fn frame_body(line: &str) -> &str {
    line.trim_matches(|c: char| c.is_ascii_whitespace())
}

/// Deserializes one trimmed frame body, turning every failure into a bounded validation error.
fn decode_frame<T>(body: &str, known: &[&str], label: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    match serde_json::from_str::<T>(body) {
        Ok(frame) => Ok(frame),
        Err(error) => {
            let detail = match raw_type(body) {
                Some(tag) if !known.contains(&tag.as_str()) => {
                    format!("unknown {label} frame type '{}'", preview(&tag))
                }
                Some(tag) => format!("invalid {label} frame '{}': {error}", preview(&tag)),
                None => format!("invalid {label} frame: {error}"),
            };
            Err(Error::validation(format!(
                "{detail} (line: '{}')",
                preview(body)
            )))
        }
    }
}

/// The `type` field of a JSON object that failed to decode as a frame, if it is a string.
fn raw_type(body: &str) -> Option<String> {
    serde_json::from_str::<Value>(body)
        .ok()?
        .get("type")?
        .as_str()
        .map(str::to_owned)
}

/// A short, char-boundary-safe prefix of `text` for error messages.
fn preview(text: &str) -> String {
    let mut shown: String = text.chars().take(ECHO_PREFIX_CHARS).collect();
    if text.chars().count() > ECHO_PREFIX_CHARS {
        shown.push('…');
    }
    shown
}
