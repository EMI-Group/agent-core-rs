//! Shared mock-server state: the script queue, the fallback response, the
//! request recordings, the id sequence, and the request notifier.
//!
//! Every field is behind a short-lived [`std::sync::Mutex`] (or an atomic), so
//! a handler never holds a guard across an `.await`; a `MutexGuard` is `!Send`,
//! which turns an accidental lock-across-await into a compile error instead of a
//! deadlock.
//!
//! Ports the ordered-response queue of `tests/unit_tests/fixtures/mock_llm.py`
//! (`set_responses` → [`MockLlmServer::push`](super::MockLlmServer::push)) and
//! the default-answer fallback of `MockLLMModel`.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::StreamChunk;
use serde_json::Value;
use tokio::sync::Notify;

use crate::internal::DEFAULT_FALLBACK_TEXT;
use crate::responses::MockCompletion;

use super::wire::MockHttpError;

/// Re-exported so [`MockLlmServer`](super::MockLlmServer) can keep using
/// `state::lock` for its task handle without a second mutex helper.
pub(crate) use crate::internal::lock;

/// Prefix of the deterministic response ids (`chatcmpl-<prefix><seq>`).
pub(crate) const DEFAULT_ID_PREFIX: &str = "mock-";

/// Model name reported when a scripted turn names no model.
pub(crate) const DEFAULT_MODEL_NAME: &str = "mock-model";

/// One entry of the script queue: a whole completion, a frame sequence, or an
/// HTTP error.
///
/// The queue is mode-agnostic (one entry per request, streaming or not): a
/// [`ScriptedResponse::Completion`] is serialized as JSON for a non-streaming
/// request and expanded to SSE frames for a streaming one, so a test states a
/// turn once instead of keeping two scripts in sync.
#[derive(Debug, Clone)]
pub enum ScriptedResponse {
    /// A neutral assistant turn, projected to JSON or SSE by the server.
    Completion(MockCompletion),
    /// An explicit SSE frame sequence, emitted as given.
    Chunks(Vec<StreamChunk>),
    /// An HTTP error response that short-circuits the request.
    Error(MockHttpError),
}

impl ScriptedResponse {
    /// A plain text turn that finishes with `stop`.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Completion(MockCompletion::text(content))
    }

    /// Wraps a scripted completion.
    #[must_use]
    pub fn completion(completion: MockCompletion) -> Self {
        Self::Completion(completion)
    }

    /// Wraps an explicit frame sequence.
    #[must_use]
    pub fn chunks(chunks: Vec<StreamChunk>) -> Self {
        Self::Chunks(chunks)
    }

    /// An HTTP error with the given status and `OpenAI` error message.
    #[must_use]
    pub fn error(status: u16, message: impl Into<String>) -> Self {
        Self::Error(MockHttpError::new(status, message))
    }
}

impl From<MockCompletion> for ScriptedResponse {
    fn from(completion: MockCompletion) -> Self {
        Self::Completion(completion)
    }
}

/// One request the mock server answered (or rejected).
///
/// Recorded before the response is written, so a test can assert on exactly
/// what the client sent; `response_id` is the id the server stamped on the
/// answer, or `None` for a scripted error.
#[derive(Debug, Clone)]
pub struct RecordedRequest {
    /// HTTP method, for example `POST`.
    pub method: String,
    /// Request path, for example `/v1/chat/completions`.
    pub path: String,
    /// Query string without the leading `?`, when the request had one.
    pub query: Option<String>,
    /// Request headers with lowercased names; repeated values are joined with `", "`.
    pub headers: BTreeMap<String, String>,
    /// Parsed JSON body, or [`Value::Null`] when the body was not a JSON object.
    pub body: Value,
    /// Body text exactly as received (lossily decoded as UTF-8).
    pub raw_body: String,
    /// Whether the request asked for a streamed answer (`"stream": true`).
    pub stream: bool,
    /// Model named in the request body, when present.
    pub model: Option<String>,
    /// Id stamped on the response, or `None` when the script served an error.
    pub response_id: Option<String>,
}

impl RecordedRequest {
    /// Returns the value of `name`, matched case-insensitively.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    /// The `messages` entries of the recorded body, in order.
    #[must_use]
    pub fn messages(&self) -> Vec<Value> {
        self.body
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    }

    /// The text content of every message authored by `role`, in order.
    ///
    /// A string `content` is returned verbatim; a content-part array is
    /// concatenated from its `text` parts. Non-text content is skipped.
    #[must_use]
    pub fn texts_of_role(&self, role: &str) -> Vec<String> {
        self.messages()
            .iter()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some(role))
            .filter_map(message_text)
            .collect()
    }

    /// The text content of every `user` message, in order.
    #[must_use]
    pub fn user_texts(&self) -> Vec<String> {
        self.texts_of_role("user")
    }
}

/// Extracts the text of one recorded message, tolerating both content shapes.
fn message_text(message: &Value) -> Option<String> {
    match message.get("content")? {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let text: String = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect();
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        _ => None,
    }
}

/// Immutable server configuration plus the initial script.
#[derive(Debug)]
pub(crate) struct ServerConfig {
    /// Entries popped in order, one per request.
    pub script: Vec<ScriptedResponse>,
    /// Answer served when the queue is empty; never consumed.
    pub fallback: ScriptedResponse,
    /// Prefix of the deterministic response ids.
    pub id_prefix: String,
    /// Model name reported when a scripted turn names no model.
    pub default_model: String,
    /// Per-SSE-frame delay (zero disables the delay entirely).
    pub chunk_delay: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            script: Vec::new(),
            fallback: ScriptedResponse::text(DEFAULT_FALLBACK_TEXT),
            id_prefix: DEFAULT_ID_PREFIX.to_owned(),
            default_model: DEFAULT_MODEL_NAME.to_owned(),
            chunk_delay: Duration::ZERO,
        }
    }
}

/// The response decided for one request, plus the sequence number that names it.
#[derive(Debug)]
pub(crate) struct PoppedResponse {
    /// The popped script entry or the reusable fallback.
    pub response: ScriptedResponse,
    /// Sequence number used for the deterministic response id: the popped
    /// entry's ordinal, or the current counter value for the fallback.
    pub seq: u64,
}

/// Mutable state shared by the server task and the [`MockLlmServer`] handle.
///
/// [`MockLlmServer`]: super::MockLlmServer
#[derive(Debug)]
pub(crate) struct ServerState {
    /// Scripted answers, popped one per request.
    script: Mutex<VecDeque<ScriptedResponse>>,
    /// Answer served when `script` is empty; never consumed.
    fallback: Mutex<ScriptedResponse>,
    /// Requests seen so far, in arrival order.
    recordings: Mutex<Vec<RecordedRequest>>,
    /// Number of popped script entries; produces `chatcmpl-<prefix><seq>`.
    sequence: AtomicU64,
    /// Wakes [`ServerState::wait_for_requests_until`] callers after a recording.
    notify: Notify,
    /// Prefix of the deterministic response ids.
    id_prefix: String,
    /// Model name reported when a scripted turn names no model.
    default_model: String,
    /// Per-SSE-frame delay (zero disables the delay entirely).
    chunk_delay: Duration,
}

impl ServerState {
    /// Builds the state from a configuration and its initial script.
    pub(crate) fn new(config: ServerConfig) -> Self {
        Self {
            script: Mutex::new(config.script.into()),
            fallback: Mutex::new(config.fallback),
            recordings: Mutex::new(Vec::new()),
            sequence: AtomicU64::new(0),
            notify: Notify::new(),
            id_prefix: config.id_prefix,
            default_model: config.default_model,
            chunk_delay: config.chunk_delay,
        }
    }

    /// Appends one scripted answer to the queue.
    pub(crate) fn push(&self, response: ScriptedResponse) {
        lock(&self.script).push_back(response);
    }

    /// Appends several scripted answers, preserving order.
    pub(crate) fn push_many(&self, items: impl IntoIterator<Item = ScriptedResponse>) {
        lock(&self.script).extend(items);
    }

    /// Drops every queued answer; the fallback and the recordings stay.
    pub(crate) fn clear_script(&self) {
        lock(&self.script).clear();
    }

    /// Replaces the answer served when the queue is empty.
    pub(crate) fn set_fallback(&self, response: ScriptedResponse) {
        *lock(&self.fallback) = response;
    }

    /// Number of queued answers.
    pub(crate) fn queued(&self) -> usize {
        lock(&self.script).len()
    }

    /// Pops exactly one entry per request, falling back to the reusable answer.
    ///
    /// A popped entry bumps the sequence counter, so the first popped answer is
    /// number 1. The fallback does not bump it and reuses the current counter
    /// value, which keeps the ids deterministic across a whole test.
    pub(crate) fn pop(&self) -> PoppedResponse {
        let popped = lock(&self.script).pop_front();
        match popped {
            Some(response) => PoppedResponse {
                response,
                seq: self
                    .sequence
                    .fetch_add(1, Ordering::SeqCst)
                    .saturating_add(1),
            },
            None => PoppedResponse {
                response: lock(&self.fallback).clone(),
                seq: self.sequence.load(Ordering::SeqCst),
            },
        }
    }

    /// The deterministic response id for `seq`: `chatcmpl-<prefix><seq:04>`.
    pub(crate) fn response_id(&self, seq: u64) -> String {
        format!("chatcmpl-{}{seq:04}", self.id_prefix)
    }

    /// Model name reported when a scripted turn names no model.
    pub(crate) fn default_model(&self) -> &str {
        &self.default_model
    }

    /// Per-SSE-frame delay (zero means "emit as fast as possible").
    pub(crate) fn chunk_delay(&self) -> Duration {
        self.chunk_delay
    }

    /// Appends one recording and wakes every `wait_for_requests_until` caller.
    pub(crate) fn record(&self, request: RecordedRequest) {
        lock(&self.recordings).push(request);
        // The notification is raised outside the lock so a waiter that wakes up
        // can read the recording without contending for the guard.
        self.notify.notify_waiters();
    }

    /// Number of recorded requests.
    pub(crate) fn request_count(&self) -> usize {
        lock(&self.recordings).len()
    }

    /// Every recorded request, in arrival order.
    pub(crate) fn requests(&self) -> Vec<RecordedRequest> {
        lock(&self.recordings).clone()
    }

    /// The most recent recorded request, if any.
    pub(crate) fn last_request(&self) -> Option<RecordedRequest> {
        lock(&self.recordings).last().cloned()
    }

    /// Waits until at least `count` requests have been recorded, without a deadline.
    ///
    /// This is the primitive behind [`ServerState::wait_for_requests_timeout`]; it is
    /// crate-private precisely because it can wait forever. Every public waiter is bounded.
    pub(crate) async fn wait_for_requests_until(&self, count: usize) {
        loop {
            // Register for the wake-up before checking the counter, so a
            // recording that lands between the check and the await is not lost.
            // `enable()`'s result only says whether a wake-up was consumed; it
            // must never short-circuit the count check, because that would
            // return before `count` requests have arrived.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            let _ = notified.as_mut().enable();
            if self.request_count() >= count {
                return;
            }
            notified.await;
        }
    }

    /// Waits until `count` requests have been recorded, or fails after `timeout`.
    pub(crate) async fn wait_for_requests_timeout(
        &self,
        count: usize,
        timeout: Duration,
    ) -> Result<()> {
        if tokio::time::timeout(timeout, self.wait_for_requests_until(count))
            .await
            .is_ok()
        {
            return Ok(());
        }
        let seen = self.request_count();
        Err(Error::timeout(format!(
            "wait_for_requests({count}) timed out after {timeout:?}; saw {seen} request(s)"
        )))
    }
}
