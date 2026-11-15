//! The HTTP transport behind [`crate::spawn::remote::RemoteAgent`]: client
//! construction, request execution, status mapping, and the NDJSON stream
//! parser.
//!
//! [`crate::spawn::remote::RemoteClient`] is the seam and
//! [`crate::spawn::remote::RemoteAgent`] keeps the lifecycle; every request is
//! executed here, so the wire details — the `session_id`/`inputs` body, the status
//! mapping, and the one-frame-per-line stream grammar of
//! [`crate::spawn::protocol`] — have one home. The stream endpoint reuses the
//! subprocess wire: a `stream_chunk` frame carries one
//! [`openjiuwen_agent::output::AgentChunk`], an `error` frame is the stream's only
//! `Err`, `done`/`output` end it, and non-JSON line noise (keep-alives, blank
//! lines) is skipped at `trace`.
//!
//! Status mapping: `404` is [`openjiuwen_common::error::Error::NotFound`], any
//! other non-2xx is [`openjiuwen_common::error::Error::Transport`] with the
//! status and a whitespace-collapsed body snippet (at most [`BODY_SNIPPET_CHARS`]
//! chars), a `reqwest` timeout is [`openjiuwen_common::error::Error::Timeout`],
//! and a non-JSON success body or an invalid output object is
//! [`openjiuwen_common::error::Error::Validation`].

use super::{RemoteAgent, RemoteAgentConfig};
use crate::error;
use crate::spawn::protocol::{SpawnMessage, SpawnMessageKind};
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use std::time::Duration;

/// Longest error-body snippet echoed back in an error message.
pub(super) const BODY_SNIPPET_CHARS: usize = 256;

/// Builds the `reqwest` client from the configured timeout and headers.
pub(super) fn build_client(config: &RemoteAgentConfig) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    for (name, value) in &config.headers {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            Error::config(format!(
                "invalid remote agent header name '{name}': {error}"
            ))
        })?;
        let header_value = HeaderValue::from_str(value).map_err(|error| {
            Error::config(format!(
                "invalid remote agent header value for '{name}': {error}"
            ))
        })?;
        headers.insert(header_name, header_value);
    }
    reqwest::Client::builder()
        .timeout(config.effective_timeout())
        .default_headers(headers)
        .build()
        .map_err(|error| {
            error::transport(format!(
                "cannot build the remote agent HTTP client: {error}"
            ))
        })
}

/// The JSON body of an invocation: `session_id` always, `inputs` verbatim.
pub(super) fn request_body(config: &RemoteAgentConfig, inputs: &Value) -> Value {
    serde_json::json!({
        "session_id": config.session_id,
        "inputs": inputs,
    })
}

/// Maps a `reqwest` failure onto the crate's error vocabulary.
fn transport_error(
    agent_id: &str,
    label: &str,
    url: &str,
    timeout: Duration,
    error: &reqwest::Error,
) -> Error {
    if error.is_timeout() {
        return error::run_timeout(label, timeout);
    }
    error::transport(format!("{label} ('{agent_id}') to {url} failed: {error}"))
}

/// Runs one invoke request for `agent` over HTTP.
pub(super) async fn invoke_http(
    agent: &RemoteAgent,
    inputs: Value,
    timeout: Option<Duration>,
) -> Result<AgentOutput> {
    let url = agent.config().invoke_url()?;
    let client = agent.http_client()?;
    let deadline = timeout.unwrap_or_else(|| agent.config().effective_timeout());
    let payload = request_body(agent.config(), &inputs);
    tracing::debug!(
        agent_id = %agent.agent_id(),
        url = %url,
        "invoking a remote agent over HTTP"
    );
    let response = client
        .post(&url)
        .timeout(deadline)
        .json(&payload)
        .send()
        .await
        .map_err(|error| {
            transport_error(
                agent.agent_id(),
                "remote agent invoke",
                &url,
                deadline,
                &error,
            )
        })?;
    let status = response.status();
    let body = response.text().await.map_err(|error| {
        error::transport(format!(
            "remote agent '{}' invoke body from {url} cannot be read: {error}",
            agent.agent_id()
        ))
    })?;
    if !status.is_success() {
        return Err(map_status(agent.agent_id(), status.as_u16(), &url, &body));
    }
    let value: Value = serde_json::from_str(&body).map_err(|error| {
        Error::validation(format!(
            "remote agent '{}' returned a non-JSON body from {url}: {error}: {}",
            agent.agent_id(),
            body_snippet(&body)
        ))
    })?;
    AgentOutput::from_value(value).map_err(|error| {
        Error::validation(format!(
            "remote agent '{}' returned an invalid output object from {url}: {error}",
            agent.agent_id()
        ))
    })
}

/// Streams one invocation for `agent`, parsing the endpoint's NDJSON body.
pub(super) fn stream_http(
    agent: &RemoteAgent,
    inputs: Value,
    timeout: Option<Duration>,
) -> BoxStream<'static, Result<AgentChunk>> {
    let agent = agent.clone();
    Box::pin(async_stream::stream! {
        let url = match agent.config().stream_url() {
            Ok(url) => url,
            Err(error) => {
                yield Err(error);
                return;
            }
        };
        let client = match agent.http_client() {
            Ok(client) => client,
            Err(error) => {
                yield Err(error);
                return;
            }
        };
        let deadline = timeout.unwrap_or_else(|| agent.config().effective_timeout());
        let payload = request_body(agent.config(), &inputs);
        tracing::debug!(
            agent_id = %agent.agent_id(),
            url = %url,
            "streaming a remote agent over HTTP"
        );
        let response = match client
            .post(&url)
            .timeout(deadline)
            .json(&payload)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                yield Err(transport_error(
                    agent.agent_id(),
                    "remote agent stream",
                    &url,
                    deadline,
                    &error,
                ));
                return;
            }
        };
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            yield Err(map_status(agent.agent_id(), status, &url, &body));
            return;
        }
        let mut body = Box::pin(response.bytes_stream());
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(chunk) = body.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(error::transport(format!(
                        "remote agent '{}' stream from {url} failed: {error}",
                        agent.agent_id()
                    )));
                    return;
                }
            };
            buffer.extend_from_slice(&chunk);
            while let Some(line) = take_line(&mut buffer) {
                match parse_stream_line(&line) {
                    StreamLine::Skip => {}
                    StreamLine::Chunk(chunk) => yield Ok(chunk),
                    StreamLine::Error(error) => {
                        yield Err(error);
                        return;
                    }
                    StreamLine::End => return,
                }
            }
        }
        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer).into_owned();
            match parse_stream_line(&line) {
                StreamLine::Skip => {}
                StreamLine::Chunk(chunk) => yield Ok(chunk),
                StreamLine::Error(error) => yield Err(error),
                StreamLine::End => {}
            }
        }
    })
}

/// Asks the endpoint to cancel `request_id`.
///
/// The body is the crate's own documented layout, so it grows additively:
/// `{"task_id": …, "request_id": …, "reason": … | null}`. `task_id` is the key the
/// first version posted and is kept so an endpoint written against it keeps
/// working; `request_id` is the same id under its correlation name, and `reason`
/// is written as `null` when the caller has none. The endpoint decides what a
/// cancel means for the request it is running — this client only carries the id,
/// and a non-2xx answer is mapped like every other remote failure.
pub(super) async fn cancel_http(
    agent: &RemoteAgent,
    request_id: &str,
    reason: Option<&str>,
) -> Result<()> {
    if request_id.trim().is_empty() {
        return Err(Error::validation(
            "a remote cancel needs a non-blank request_id",
        ));
    }
    let Some(url) = agent.config().cancel_url()? else {
        return Err(error::unsupported(
            "remote agent cancel_task",
            "no cancel_path is configured for this RemoteAgent",
        ));
    };
    let client = agent.http_client()?;
    let deadline = agent.config().effective_timeout();
    let response = client
        .post(&url)
        .timeout(deadline)
        .json(&serde_json::json!({
            "task_id": request_id,
            "request_id": request_id,
            "reason": reason,
        }))
        .send()
        .await
        .map_err(|error| {
            transport_error(
                agent.agent_id(),
                "remote agent cancel_task",
                &url,
                deadline,
                &error,
            )
        })?;
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    let body = response.text().await.unwrap_or_default();
    Err(map_status(agent.agent_id(), status.as_u16(), &url, &body))
}

/// Maps a non-2xx status onto an error; `404` means the agent is unknown.
pub(super) fn map_status(agent_id: &str, status: u16, url: &str, body: &str) -> Error {
    let snippet = body_snippet(body);
    if status == 404 {
        return Error::NotFound(with_snippet(
            format!("remote agent '{agent_id}' is not registered at {url}"),
            &snippet,
        ));
    }
    error::transport(with_snippet(
        format!("remote agent '{agent_id}' returned status {status} from {url}"),
        &snippet,
    ))
}

/// Appends `: snippet` when there is a snippet to show.
fn with_snippet(message: String, snippet: &str) -> String {
    if snippet.is_empty() {
        message
    } else {
        format!("{message}: {snippet}")
    }
}

/// A whitespace-collapsed, at-most-[`BODY_SNIPPET_CHARS`]-char body snippet.
pub(super) fn body_snippet(body: &str) -> String {
    let collapsed = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= BODY_SNIPPET_CHARS {
        return collapsed;
    }
    let mut snippet: String = collapsed.chars().take(BODY_SNIPPET_CHARS - 1).collect();
    snippet.push('…');
    snippet
}

/// What one line of a remote stream means.
#[derive(Debug)]
pub(super) enum StreamLine {
    /// One chunk item.
    Chunk(AgentChunk),
    /// One `Err` item; the stream ends afterwards.
    Error(Error),
    /// The stream is complete.
    End,
    /// Noise or a frame this direction should not send: ignore it.
    Skip,
}

/// Parses one NDJSON line of a remote stream.
///
/// Non-JSON noise is skipped at `trace`; valid JSON that is not a spawn frame is
/// a [`Error::Validation`] that ends the stream.
pub(super) fn parse_stream_line(line: &str) -> StreamLine {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return StreamLine::Skip;
    }
    if serde_json::from_str::<Value>(trimmed).is_err() {
        tracing::trace!(
            target: "openjiuwen_runner::spawn",
            line = %body_snippet(trimmed),
            "skipping a non-JSON line of a remote agent stream"
        );
        return StreamLine::Skip;
    }
    let frame = match SpawnMessage::from_line(trimmed) {
        Ok(frame) => frame,
        Err(error) => return StreamLine::Error(error),
    };
    match frame.kind {
        SpawnMessageKind::StreamChunk => match AgentChunk::from_value(frame.payload) {
            Ok(chunk) => StreamLine::Chunk(chunk),
            Err(error) => StreamLine::Error(Error::validation(format!(
                "remote agent stream_chunk is not a valid AgentChunk frame: {error}"
            ))),
        },
        SpawnMessageKind::Done | SpawnMessageKind::Output => StreamLine::End,
        SpawnMessageKind::Error => StreamLine::Error(error::transport(format!(
            "remote agent stream reported an error: {}",
            frame_error_message(&frame.payload)
        ))),
        other => {
            tracing::trace!(
                target: "openjiuwen_runner::spawn",
                kind = ?other,
                "ignoring a remote agent stream frame"
            );
            StreamLine::Skip
        }
    }
}

/// The human-readable message of an `error` frame's payload.
fn frame_error_message(payload: &Value) -> String {
    match payload {
        Value::String(text) => text.clone(),
        Value::Object(map) => match map.get("message") {
            Some(Value::String(text)) => text.clone(),
            Some(other) => other.to_string(),
            None => payload.to_string(),
        },
        Value::Null => "no error detail was provided".to_owned(),
        other => other.to_string(),
    }
}

/// Removes and returns the next complete `\n`-terminated line of `buffer`.
pub(super) fn take_line(buffer: &mut Vec<u8>) -> Option<String> {
    let position = buffer.iter().position(|byte| *byte == b'\n')?;
    let line: Vec<u8> = buffer.drain(..=position).collect();
    Some(String::from_utf8_lossy(&line).into_owned())
}
