//! Responses conversation state and client-side protocol fields.
//!
//! The `OpenAI` Responses API can continue a conversation with
//! `previous_response_id` and can cap how many tool calls one response may make
//! with `max_tool_calls`; both are ordinary protocol fields. This module owns
//! them: the typed accessors that read the caller's hints out of
//! [`ModelRequestConfig::extra`], the writer that validates and forwards them, and
//! the [`ResponsesSessionStore`] seam that remembers the last response id per
//! caller-supplied conversation key, so a follow-up call continues a conversation
//! without the caller threading the id by hand.
//!
//! Two hints are **client-side only** and never reach the wire: the
//! [`CONVERSATION_KEY`] (`conversation_id`) `extra` entry, and the per-client
//! conversation key configured with
//! [`with_conversation_key`](super::OpenAiResponsesClient::with_conversation_key).
//! Only `previous_response_id` and `max_tool_calls` are protocol fields.
//!
//! The transport does not implement the two Responses features that only work
//! against the live service — `background` mode and hosted tools — so both are
//! typed [`Error::Unsupported`] errors instead of a silent pass-through.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};

use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};

use super::super::config::ModelRequestConfig;
use super::request::EXTRA_BODY_KEY;

/// Key of the explicit `previous_response_id` protocol field in
/// [`ModelRequestConfig::extra`].
pub const PREVIOUS_RESPONSE_ID_KEY: &str = "previous_response_id";

/// Key of the caller's client-side conversation key in
/// [`ModelRequestConfig::extra`].
///
/// It scopes [`ResponsesSessionStore`] lookups and records, and is never written
/// to a request body.
pub const CONVERSATION_KEY: &str = "conversation_id";

/// Key of the `max_tool_calls` protocol field in [`ModelRequestConfig::extra`].
pub const MAX_TOOL_CALLS_KEY: &str = "max_tool_calls";

/// Key of the `background` mode flag in [`ModelRequestConfig::extra`].
const BACKGROUND_KEY: &str = "background";

/// Default number of conversations an [`InMemoryResponsesSessionStore`] keeps.
pub const DEFAULT_CONVERSATION_CAPACITY: usize = 256;

/// Remembers the last response id of a conversation.
///
/// A Responses follow-up continues a conversation by sending the previous
/// response's id as `previous_response_id`. This seam stores that mapping, so the
/// transport resolves it from a caller-supplied conversation key instead of
/// forcing the caller to thread the id manually.
///
/// Implementations are per-client state (see
/// [`with_session_store`](super::OpenAiResponsesClient::with_session_store)),
/// never a process global, and take `&self` so a shared store can be used behind
/// an [`Arc`](std::sync::Arc). A store is best-effort: a failed lookup or write
/// only costs conversation continuity, never the call itself.
pub trait ResponsesSessionStore: fmt::Debug + Send + Sync {
    /// The last response id recorded for `conversation`, if any.
    ///
    /// A blank conversation key always yields `None`.
    fn last_response_id(&self, conversation: &str) -> Option<String>;

    /// Records `response_id` as the last response id of `conversation`.
    ///
    /// A blank conversation key or a blank response id is ignored.
    fn remember_response_id(&self, conversation: &str, response_id: &str);
}

/// One conversation's ids, least recently used first.
#[derive(Debug, Default)]
struct Entries {
    /// Response ids by conversation key.
    ids: HashMap<String, String>,
    /// Conversation keys, least recently used at the front.
    recency: VecDeque<String>,
}

impl Entries {
    /// Marks `key` as the most recently used conversation.
    fn touch(&mut self, key: &str) {
        self.recency.retain(|entry| entry != key);
        self.recency.push_back(key.to_owned());
    }
}

/// A bounded, in-memory [`ResponsesSessionStore`].
///
/// The store keeps at most [`capacity`](Self::capacity) conversations and evicts
/// the least recently used one when a new conversation would exceed the cap (a
/// lookup counts as use), so an unbounded key space — one conversation key per
/// end user, say — cannot grow the process without limit.
///
/// It is plain in-memory state: nothing is persisted, so a restarted process
/// forgets every conversation and falls back to a fresh one. Share one store
/// between clients by handing the same [`Arc`](std::sync::Arc) to each, or keep
/// the default per-client store.
#[derive(Debug)]
pub struct InMemoryResponsesSessionStore {
    /// Maximum number of conversations kept.
    capacity: usize,
    /// Conversation state, guarded so the seam can take `&self`.
    entries: Mutex<Entries>,
}

impl InMemoryResponsesSessionStore {
    /// Creates a store holding [`DEFAULT_CONVERSATION_CAPACITY`] conversations.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CONVERSATION_CAPACITY)
    }

    /// Creates a store holding at most `capacity` conversations.
    ///
    /// `capacity` is clamped to at least one entry, so a store always remembers
    /// the most recent conversation.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Mutex::new(Entries::default()),
        }
    }

    /// The number of conversations this store keeps before evicting.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Locks the entries, recovering from a poisoned lock.
    ///
    /// A panicking writer leaves a consistent map behind, so recovering is
    /// preferable to propagating the poison into every later call.
    fn lock(&self) -> MutexGuard<'_, Entries> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Default for InMemoryResponsesSessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponsesSessionStore for InMemoryResponsesSessionStore {
    fn last_response_id(&self, conversation: &str) -> Option<String> {
        let key = conversation.trim();
        if key.is_empty() {
            return None;
        }
        let mut entries = self.lock();
        let id = entries.ids.get(key).cloned()?;
        entries.touch(key);
        Some(id)
    }

    fn remember_response_id(&self, conversation: &str, response_id: &str) {
        let key = conversation.trim();
        let id = response_id.trim();
        if key.is_empty() || id.is_empty() {
            return;
        }
        let mut entries = self.lock();
        entries.ids.insert(key.to_owned(), id.to_owned());
        entries.touch(key);
        while entries.ids.len() > self.capacity {
            let Some(oldest) = entries.recency.pop_front() else {
                break;
            };
            entries.ids.remove(&oldest);
        }
    }
}

/// A [`ResponsesSessionStore`] that remembers nothing.
///
/// Use it to opt out of conversation continuity entirely: no `previous_response_id`
/// is ever resolved from a store, and answered ids are dropped. An explicit
/// `extra["previous_response_id"]` still reaches the wire.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopResponsesSessionStore;

impl ResponsesSessionStore for NoopResponsesSessionStore {
    fn last_response_id(&self, _conversation: &str) -> Option<String> {
        None
    }

    fn remember_response_id(&self, _conversation: &str, _response_id: &str) {}
}

/// Reads the explicit `previous_response_id` from a request.
///
/// `null` and a blank string mean "not set". Any other non-string value is an
/// [`Error::Validation`] naming the field, so a malformed hint fails loudly
/// instead of being dropped.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `extra["previous_response_id"]` is neither
/// `null` nor a string.
pub fn previous_response_id(request: &ModelRequestConfig) -> Result<Option<String>> {
    match request.extra.get(PREVIOUS_RESPONSE_ID_KEY) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) => {
            let id = id.trim();
            Ok((!id.is_empty()).then(|| id.to_owned()))
        }
        Some(_) => Err(Error::validation(format!(
            "`{PREVIOUS_RESPONSE_ID_KEY}` must be a string response id"
        ))),
    }
}

/// Reads the caller's conversation key from a request.
///
/// `null` and a blank string mean "no conversation". Any other non-string value is
/// an [`Error::Validation`] naming the field.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `extra["conversation_id"]` is neither `null`
/// nor a string.
pub fn conversation_key(request: &ModelRequestConfig) -> Result<Option<String>> {
    match request.extra.get(CONVERSATION_KEY) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(key)) => {
            let key = key.trim();
            Ok((!key.is_empty()).then(|| key.to_owned()))
        }
        Some(_) => Err(Error::validation(format!(
            "`{CONVERSATION_KEY}` must be a string conversation key"
        ))),
    }
}

/// Reads the validated `max_tool_calls` limit from a request.
///
/// The value is an integer, written as a JSON number; a numeric string is
/// accepted too, because limits are often configured from an environment
/// variable. `null` means "not set". Zero, negative numbers, fractions, and
/// non-numeric strings are an [`Error::Validation`] naming the field.
///
/// The transport only forwards the limit: the service enforces it, so the value
/// is never applied client-side.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `extra["max_tool_calls"]` is set to
/// anything but a positive integer that fits a `u32`.
pub fn max_tool_calls(request: &ModelRequestConfig) -> Result<Option<u32>> {
    let value = match request.extra.get(MAX_TOOL_CALLS_KEY) {
        None | Some(Value::Null) => return Ok(None),
        Some(value) => value,
    };
    let parsed = match value {
        Value::Number(number) => number.as_u64().and_then(|limit| u32::try_from(limit).ok()),
        Value::String(text) => text.trim().parse::<u32>().ok(),
        _ => None,
    };
    match parsed {
        Some(limit) if limit > 0 => Ok(Some(limit)),
        _ => Err(Error::validation(format!(
            "`{MAX_TOOL_CALLS_KEY}` must be a positive integer no greater than {}, got `{value}`",
            u32::MAX
        ))),
    }
}

/// Whether `key` is a client-side hint that must never reach a request body.
#[must_use]
pub(crate) fn is_client_side_key(key: &str) -> bool {
    key == CONVERSATION_KEY || key == PREVIOUS_RESPONSE_ID_KEY || key == MAX_TOOL_CALLS_KEY
}

/// Writes the two protocol fields into a request body.
///
/// Both are written only when the request configures them, and both are
/// validated first, so a bad value fails the build instead of reaching the
/// provider.
pub(crate) fn apply_body_fields(
    body: &mut Map<String, Value>,
    request: &ModelRequestConfig,
) -> Result<()> {
    if let Some(previous) = previous_response_id(request)? {
        body.insert(PREVIOUS_RESPONSE_ID_KEY.to_owned(), Value::String(previous));
    }
    if let Some(limit) = max_tool_calls(request)? {
        body.insert(MAX_TOOL_CALLS_KEY.to_owned(), Value::from(u64::from(limit)));
    }
    Ok(())
}

/// Rejects the Responses features this transport does not implement.
///
/// `background` mode needs the live service to run a response asynchronously and
/// poll it, and hosted tools (web search, file search, computer use, ...) are
/// executed by the service rather than by the caller, so neither can be emulated
/// locally: both are typed [`Error::Unsupported`] with a reason instead of a
/// silently forwarded body field or a mangled tool definition.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] when `extra["background"]` is `true` (also
/// inside `extra["extra_body"]`) or when `tools` declares a hosted tool (web
/// search, file search, computer use, code interpreter, image generation, shell,
/// or MCP).
pub(crate) fn reject_unsupported(request: &ModelRequestConfig) -> Result<()> {
    if background_requested(request) {
        return Err(Error::unsupported(
            "`background` mode is not implemented: it requires the live OpenAI Responses \
             service to run the response asynchronously and poll `GET /responses/{id}`",
        ));
    }
    if let Some(kind) = hosted_tool_kind(request.tools.as_ref()) {
        return Err(Error::unsupported(format!(
            "hosted tool `{kind}` is not implemented: hosted tools are executed by the live \
             OpenAI Responses service, and this transport forwards `function` tools only"
        )));
    }
    Ok(())
}

/// Whether the request asks for `background` mode, at either `extra` level.
fn background_requested(request: &ModelRequestConfig) -> bool {
    let nested = request
        .extra
        .get(EXTRA_BODY_KEY)
        .and_then(Value::as_object)
        .and_then(|body| body.get(BACKGROUND_KEY));
    [request.extra.get(BACKGROUND_KEY), nested]
        .into_iter()
        .flatten()
        .any(|value| value.as_bool() == Some(true))
}

/// The first hosted tool type declared by the request.
///
/// Only the kinds the Responses service itself executes count as hosted; an
/// unfamiliar `type` (a vendor extension, or Python's caller-executed `custom`
/// tools) still passes through verbatim.
fn hosted_tool_kind(tools: Option<&Value>) -> Option<&str> {
    tools.and_then(Value::as_array)?.iter().find_map(|tool| {
        let kind = tool.get("type")?.as_str()?;
        is_hosted_tool(kind).then_some(kind)
    })
}

/// Whether `kind` is a tool the live Responses service executes.
fn is_hosted_tool(kind: &str) -> bool {
    kind.starts_with("web_search")
        || matches!(
            kind,
            "file_search"
                | "computer"
                | "computer_use_preview"
                | "code_interpreter"
                | "image_generation"
                | "local_shell"
                | "shell"
                | "mcp"
        )
}

/// One call's resolved conversation state.
#[derive(Debug, Clone)]
pub(crate) struct ConversationTurn {
    /// The per-call request, with `previous_response_id` resolved from the store.
    pub(crate) request: ModelRequestConfig,
    /// The key this turn's response id is recorded under, when scoped.
    pub(crate) key: Option<String>,
}

impl ConversationTurn {
    /// Records this turn's response id under the conversation key.
    ///
    /// A call without a conversation key records nothing, and a response without
    /// an id records nothing: both are provider-side shapes, not caller errors.
    pub(crate) fn record(&self, sessions: &dyn ResponsesSessionStore, response_id: Option<&str>) {
        let (Some(key), Some(response_id)) = (self.key.as_deref(), response_id) else {
            return;
        };
        let response_id = response_id.trim();
        if response_id.is_empty() {
            return;
        }
        sessions.remember_response_id(key, response_id);
        tracing::debug!(
            target: "openjiuwen_foundation::llm::responses",
            conversation = key,
            response_id,
            "recorded the Responses conversation state"
        );
    }
}

/// Resolves one call's conversation state against the client's store.
///
/// The request-level [`conversation_id`](CONVERSATION_KEY) wins over the
/// client-wide key. An explicit
/// [`previous_response_id`](PREVIOUS_RESPONSE_ID_KEY) always wins, and otherwise
/// the store's last id for the conversation is injected into the returned request
/// (which is what the body builder forwards). A call with no conversation key
/// never consults the store.
///
/// # Errors
///
/// Returns the [`Error::Validation`] of a malformed `conversation_id` or
/// `previous_response_id` hint.
pub(crate) fn resolve(
    request: &ModelRequestConfig,
    configured_key: Option<&str>,
    sessions: &dyn ResponsesSessionStore,
) -> Result<ConversationTurn> {
    let key = match conversation_key(request)? {
        Some(key) => Some(key),
        None => configured_key
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .map(str::to_owned),
    };

    let mut resolved = request.clone();
    if previous_response_id(request)?.is_none() {
        if let Some(id) = key
            .as_deref()
            .and_then(|key| sessions.last_response_id(key))
        {
            resolved = with_previous_response_id(request, &id);
        }
    }
    Ok(ConversationTurn {
        request: resolved,
        key,
    })
}

/// Clones `request` with `previous_response_id` set in its `extra`.
///
/// A non-object `extra` is replaced by an object holding the id, because a
/// non-object `extra` is never merged into a body anyway.
fn with_previous_response_id(
    request: &ModelRequestConfig,
    response_id: &str,
) -> ModelRequestConfig {
    let mut resolved = request.clone();
    let mut extra = resolved.extra.as_object().cloned().unwrap_or_default();
    extra.insert(
        PREVIOUS_RESPONSE_ID_KEY.to_owned(),
        Value::String(response_id.to_owned()),
    );
    resolved.extra = Value::Object(extra);
    resolved
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request() -> ModelRequestConfig {
        ModelRequestConfig::new("gpt-5")
    }

    #[test]
    fn max_tool_calls_accepts_positive_integers_and_numeric_strings() {
        assert_eq!(max_tool_calls(&request()).expect("unset"), None);
        let set = request().with_extra(json!({"max_tool_calls": 3}));
        assert_eq!(max_tool_calls(&set).expect("a number"), Some(3));
        let text = request().with_extra(json!({"max_tool_calls": " 7 "}));
        assert_eq!(max_tool_calls(&text).expect("a numeric string"), Some(7));
        let null = request().with_extra(json!({"max_tool_calls": null}));
        assert_eq!(max_tool_calls(&null).expect("null is unset"), None);
    }

    #[test]
    fn an_invalid_max_tool_calls_is_a_validation_error_naming_the_field() {
        for value in [
            json!(0),
            json!(-1),
            json!(2.5),
            json!("many"),
            json!(true),
            json!({"limit": 2}),
            json!(u64::from(u32::MAX) + 1),
        ] {
            let request = request().with_extra(json!({"max_tool_calls": value}));
            let error = max_tool_calls(&request).expect_err("invalid limits are rejected");
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
            assert!(
                error.to_string().contains("max_tool_calls"),
                "the error names the field: {error}"
            );
        }
    }

    #[test]
    fn conversation_hints_must_be_strings() {
        let scoped = request().with_extra(json!({"conversation_id": " conv-1 "}));
        assert_eq!(
            conversation_key(&scoped).expect("a key"),
            Some("conv-1".to_owned())
        );
        assert_eq!(
            conversation_key(&request().with_extra(json!({"conversation_id": "  "})))
                .expect("blank is unset"),
            None
        );
        let error = conversation_key(&request().with_extra(json!({"conversation_id": 7})))
            .expect_err("a number is not a key");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("conversation_id"));

        let error =
            previous_response_id(&request().with_extra(json!({"previous_response_id": []})))
                .expect_err("an array is not an id");
        assert!(error.to_string().contains("previous_response_id"));
    }

    #[test]
    fn the_store_is_bounded_and_evicts_the_least_recently_used_conversation() {
        let store = InMemoryResponsesSessionStore::with_capacity(2);
        assert_eq!(store.capacity(), 2);
        store.remember_response_id("a", "resp_a");
        store.remember_response_id("b", "resp_b");

        // Touching `a` makes `b` the least recently used conversation.
        assert_eq!(store.last_response_id("a").as_deref(), Some("resp_a"));
        store.remember_response_id("c", "resp_c");

        assert_eq!(store.last_response_id("c").as_deref(), Some("resp_c"));
        assert_eq!(store.last_response_id("a").as_deref(), Some("resp_a"));
        assert_eq!(
            store.last_response_id("b"),
            None,
            "the least recently used conversation is evicted"
        );
    }

    #[test]
    fn the_store_overwrites_a_conversation_and_ignores_blank_input() {
        let store = InMemoryResponsesSessionStore::with_capacity(1);
        store.remember_response_id("a", "resp_1");
        store.remember_response_id("a", "resp_2");
        assert_eq!(store.last_response_id("a").as_deref(), Some("resp_2"));

        store.remember_response_id("", "resp_3");
        store.remember_response_id("a", "  ");
        assert_eq!(store.last_response_id(""), None);
        assert_eq!(store.last_response_id("a").as_deref(), Some("resp_2"));
        assert_eq!(
            InMemoryResponsesSessionStore::with_capacity(0).capacity(),
            1,
            "a store always keeps at least one conversation"
        );
    }

    #[test]
    fn a_noop_store_remembers_nothing() {
        let store = NoopResponsesSessionStore;
        store.remember_response_id("a", "resp_1");
        assert_eq!(store.last_response_id("a"), None);
    }

    #[test]
    fn resolution_prefers_the_request_key_and_never_overrides_an_explicit_id() {
        let store = InMemoryResponsesSessionStore::new();
        store.remember_response_id("configured", "resp_c");
        store.remember_response_id("per-call", "resp_p");

        let scoped = request().with_extra(json!({"conversation_id": "per-call"}));
        let turn = resolve(&scoped, Some("configured"), &store).expect("resolves");
        assert_eq!(turn.key.as_deref(), Some("per-call"));
        assert_eq!(
            previous_response_id(&turn.request).expect("the id"),
            Some("resp_p".to_owned())
        );

        let unscoped = resolve(&request(), None, &store).expect("resolves");
        assert_eq!(unscoped.key, None);
        assert_eq!(
            previous_response_id(&unscoped.request).expect("no id"),
            None,
            "a call without a conversation key never consults the store"
        );

        let explicit = request().with_extra(json!({
            "conversation_id": "per-call",
            "previous_response_id": "resp_explicit",
        }));
        let turn = resolve(&explicit, None, &store).expect("resolves");
        assert_eq!(
            previous_response_id(&turn.request).expect("the id"),
            Some("resp_explicit".to_owned())
        );
    }

    #[test]
    fn a_scalar_extra_is_replaced_by_the_resolved_id() {
        let store = InMemoryResponsesSessionStore::new();
        store.remember_response_id("conv", "resp_1");
        let plain = request().with_extra(json!("compaction"));
        let turn = resolve(&plain, Some("conv"), &store).expect("resolves");
        assert_eq!(
            turn.request.extra,
            json!({"previous_response_id": "resp_1"}),
            "the resolved id needs an object to live in"
        );
    }

    #[test]
    fn recording_a_turn_ignores_a_blank_or_missing_id() {
        let store = InMemoryResponsesSessionStore::new();
        let turn = ConversationTurn {
            request: request(),
            key: Some("conv".to_owned()),
        };
        turn.record(&store, None);
        turn.record(&store, Some("  "));
        assert_eq!(store.last_response_id("conv"), None);
        turn.record(&store, Some(" resp_9 "));
        assert_eq!(store.last_response_id("conv").as_deref(), Some("resp_9"));

        let unscoped = ConversationTurn {
            request: request(),
            key: None,
        };
        unscoped.record(&store, Some("resp_10"));
        assert_eq!(store.last_response_id(""), None);
    }
}
