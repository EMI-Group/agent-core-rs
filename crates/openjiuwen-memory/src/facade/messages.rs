//! Conversation history, model-assisted extraction, and the heuristic fallback.
//!
//! [`LongTermMemory::add_messages`] ports Python's `add_messages`: it appends the
//! sanitized turns to the scope's message history and turns the conversation into
//! memory records. With an LLM attached one extraction call asks for
//! `{"facts": [...], "variables": {...}}`; without an LLM (or when the model fails) a
//! deterministic heuristic path writes one episodic record per long enough turn plus a
//! history summary.
//!
//! A model failure never fails `add_messages`: it degrades to the heuristic path with a
//! warning and an [`AddMemResult::degraded`] reason. A payload that parses but reports
//! neither a fact nor a variable counts as a failure too — an empty object must not
//! suppress the heuristics. Only cancellation is propagated, and already-appended history
//! stays.
//!
//! [`LongTermMemory::get_recent_messages`] reads the history back as foundation
//! [`BaseMessage`]s in chronological order, and [`LongTermMemory::get_message_page`]
//! walks the same history forward as keyset pages, so a caller can replay an arbitrarily
//! long conversation without an offset or a row number.
//!
//! Extracted variables are written through the projection-aware variable path
//! ([`LongTermMemory::assign_variable`]), so a variable inferred from a conversation is
//! readable both from the Python-shaped KV API and as a
//! [`MemoryType::Variable`] record.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::str::FromStr;

use chrono::Utc;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{ChatResponse, Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::tokens::TokenCounter;
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role};
use serde_json::{Map, Value};

use crate::config::AgentMemoryConfig;
use crate::record::{
    AddMemResult, AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope,
    MemoryType, NewMemory,
};
use crate::store::{MessagePage, StoredMessage};

use super::variables::validate_variable_name;
use super::{LongTermMemory, TARGET};

/// The system prompt of the extraction call.
const EXTRACTION_SYSTEM_PROMPT: &str = "You extract durable long-term memory from a conversation. \
Answer with one JSON object and nothing else: \
{\"facts\":[{\"content\":\"...\",\"type\":\"user_profile|semantic_memory|episodic_memory|summary\",\
\"importance\":0.5}],\"variables\":{\"name\":\"value\"}}. \
Only record facts that stay useful in later conversations; skip small talk.";

/// The facts and variables one extraction response produced.
struct ExtractionPayload {
    /// Candidate memories, in response order.
    facts: Vec<NewMemory>,
    /// Candidate variables; already coerced to strings and nulls dropped.
    variables: BTreeMap<String, String>,
}

impl ExtractionPayload {
    /// Returns `true` when the payload carries neither a fact nor a variable.
    ///
    /// A syntactically valid but empty answer (`{}`, `{"facts":[]}`) is not a usable
    /// extraction: accepting it would suppress the heuristic fallback and store nothing
    /// for a batch that has extractable turns, so the caller treats it as a failure.
    fn is_empty(&self) -> bool {
        self.facts.is_empty() && self.variables.is_empty()
    }
}

/// What one model-assisted extraction attempt produced.
enum ExtractionAttempt {
    /// No model is attached, so only the deterministic heuristics are available.
    ///
    /// This is the documented no-model path, not a degradation.
    NoModel,
    /// The model answered with a usable payload that has not been applied yet.
    Payload(ExtractionPayload),
    /// Every attempt failed (transport, parse, or an empty payload); the heuristics ran.
    ///
    /// The error is carried so the caller can name the reason in
    /// [`AddMemResult::degraded`].
    Failed(Error),
}

impl LongTermMemory {
    /// Appends conversation messages to the history and extracts memory from them.
    ///
    /// The messages are truncated to [`crate::config::MemoryEngineConfig::input_msg_max_len`]
    /// characters and appended to the scope's history first, so history is recorded even
    /// when extraction is disabled. Extraction then runs the model path when an LLM is
    /// attached and falls back to heuristics otherwise; facts are filtered by
    /// [`AgentMemoryConfig::kind_enabled`] and `max_facts`, and inferred variables are
    /// stored only for [`AgentMemoryConfig::mem_variables`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope, [`Error::Cancelled`] when the
    /// call is cancelled, and the storage/embedding error when a write fails. A model
    /// failure never fails the call: it degrades to the heuristic path and the reason is
    /// reported in [`AddMemResult::degraded`].
    pub async fn add_messages(
        &self,
        scope: &MemoryScope,
        messages: &[BaseMessage],
        agent_config: &AgentMemoryConfig,
        options: &AddOptions,
    ) -> Result<AddMemResult> {
        scope.validate()?;
        options.call.check()?;
        let mut result = AddMemResult::default();
        if messages.is_empty() {
            return Ok(result);
        }
        let sanitized = sanitize_messages(messages, self.config().input_msg_max_len);
        let now = Utc::now();
        options
            .call
            .run(self.engine.message_store().append(scope, &sanitized, now))
            .await??;

        let config = self.config();
        if !config.extraction.enabled || !agent_config.enable_long_term_mem {
            return Ok(result);
        }
        if !sanitized.iter().any(|message| message.role.is_user()) {
            tracing::debug!(
                target: TARGET,
                "message batch has no user turn; skipping memory extraction"
            );
            return Ok(result);
        }

        match self
            .extract_with_model(scope, &sanitized, agent_config, options)
            .await?
        {
            ExtractionAttempt::Payload(payload) => {
                self.apply_extraction(scope, agent_config, options, payload, &mut result)
                    .await?;
            }
            ExtractionAttempt::NoModel => {
                self.apply_heuristics(scope, agent_config, options, &sanitized, &mut result)
                    .await?;
            }
            ExtractionAttempt::Failed(error) => {
                tracing::warn!(
                    target: TARGET,
                    %error,
                    "memory extraction failed; falling back to heuristics"
                );
                result.degraded = Some(format!(
                    "model memory extraction failed ({error}); heuristic extraction ran instead"
                ));
                self.apply_heuristics(scope, agent_config, options, &sanitized, &mut result)
                    .await?;
            }
        }
        Ok(result)
    }

    /// Returns the `num` most recent messages of `scope`, oldest first.
    ///
    /// `num` is clamped to [`crate::config::MemoryEngineConfig::message_history_limit`].
    /// An empty or missing history yields an empty vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when the
    /// read fails.
    pub async fn get_recent_messages(
        &self,
        scope: &MemoryScope,
        num: usize,
    ) -> Result<Vec<BaseMessage>> {
        scope.validate()?;
        let limit = num.min(self.config().message_history_limit);
        let stored = self.engine.message_store().recent(scope, limit).await?;
        Ok(stored.into_iter().map(message_from_stored).collect())
    }

    /// Returns one keyset page of the history of `scope`, oldest row first.
    ///
    /// The page is ordered strictly ascending by `(timestamp, id)` and resumes after
    /// `cursor` (a token from a previous [`MessagePage::next_cursor`]); `None` starts at
    /// the oldest row. `filter` narrows the page and is `AND`ed with the scope clauses,
    /// so it can never widen the page past its scope.
    ///
    /// `page_size == 0` returns an empty page (`has_more == false`, no cursor) and a
    /// `page_size` larger than the history returns everything, so no clamping against
    /// [`crate::config::MemoryEngineConfig::message_history_limit`] applies — that limit
    /// belongs to the newest-N window of [`LongTermMemory::get_recent_messages`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or a malformed/tampered
    /// `cursor` and the storage error when the read fails.
    pub async fn get_message_page(
        &self,
        scope: &MemoryScope,
        page_size: usize,
        cursor: Option<&str>,
        filter: Option<&MemoryFilter>,
    ) -> Result<MessagePage> {
        scope.validate()?;
        self.engine
            .message_store()
            .page(scope, page_size, cursor, filter)
            .await
    }

    /// Returns one keyset page of the unfiltered history of `scope`.
    ///
    /// The convenience form of [`LongTermMemory::get_message_page`] with no extra
    /// [`MemoryFilter`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or a malformed/tampered
    /// `cursor` and the storage error when the read fails.
    pub async fn page_messages(
        &self,
        scope: &MemoryScope,
        page_size: usize,
        cursor: Option<&str>,
    ) -> Result<MessagePage> {
        self.get_message_page(scope, page_size, cursor, None).await
    }

    /// Deletes the conversation history rows of one tenant scope matching `filter`,
    /// returning how many were removed.
    ///
    /// Only message history is cleared; memory records and variables are untouched (use
    /// [`LongTermMemory::clear`] for those). `filter` must name a non-empty `user_id` and
    /// `scope_id` — build it with [`MemoryFilter::for_scope`] — and those clauses are
    /// `AND`ed into the delete, so no caller can delete another tenant's history (or the
    /// whole collection) by passing an empty filter; Python's
    /// `delete_by_user_and_scope` cannot express that either. `session_id`, `ids`, and the
    /// `created_after`/`created_before` bounds narrow the delete further; kind,
    /// importance, and text clauses do not apply to messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] (before deleting anything) when `filter` names no
    /// `user_id` or no `scope_id`, [`Error::Cancelled`] when the call is cancelled before
    /// the delete, and the storage error when the delete fails.
    pub async fn clear_messages(
        &self,
        filter: &MemoryFilter,
        options: &CallOptions,
    ) -> Result<usize> {
        crate::store::require_message_scope(filter)?;
        options.check()?;
        let deleted = options
            .run(self.engine.message_store().delete_for(filter))
            .await??;
        Ok(deleted)
    }

    /// Runs one extraction prompt, retrying on model or parse failures.
    ///
    /// Returns [`ExtractionAttempt::NoModel`] when there is no model to ask,
    /// [`ExtractionAttempt::Payload`] for a usable payload, and
    /// [`ExtractionAttempt::Failed`] when every attempt failed (the caller falls back to
    /// heuristics and records the reason). A syntactically valid but empty payload counts
    /// as a failure so it cannot suppress the heuristics. [`Error::Cancelled`] is returned
    /// as soon as the call is cancelled — a cancelled extraction is never retried.
    async fn extract_with_model(
        &self,
        scope: &MemoryScope,
        messages: &[BaseMessage],
        agent_config: &AgentMemoryConfig,
        options: &AddOptions,
    ) -> Result<ExtractionAttempt> {
        let Some(model) = self.engine.model() else {
            return Ok(ExtractionAttempt::NoModel);
        };
        let config = self.config();
        let history = self
            .engine
            .message_store()
            .recent(scope, config.extraction.use_history_messages)
            .await?;
        let prompt = vec![
            BaseMessage::system(EXTRACTION_SYSTEM_PROMPT),
            BaseMessage::user(build_extraction_prompt(&history, messages, agent_config)),
        ];
        let request = ModelRequestConfig::new(model.model_name());
        let attempts = config.extraction.model_retries.max(1);
        let mut last_error: Option<Error> = None;
        for _ in 0..attempts {
            options.call.check()?;
            let response = match options.call.run(model.invoke(&prompt, &request)).await {
                Ok(Ok(response)) => response,
                Ok(Err(error)) | Err(error) => {
                    if matches!(error, Error::Cancelled) {
                        return Err(error);
                    }
                    last_error = Some(error);
                    continue;
                }
            };
            match parse_extraction(&response) {
                Ok(payload) if payload.is_empty() => {
                    last_error = Some(Error::memory(
                        "memory extraction payload carried neither a fact nor a variable",
                    ));
                }
                Ok(payload) => return Ok(ExtractionAttempt::Payload(payload)),
                Err(error) => last_error = Some(error),
            }
        }
        Ok(ExtractionAttempt::Failed(last_error.unwrap_or_else(|| {
            Error::memory("memory extraction ran no attempt")
        })))
    }

    /// Applies a parsed extraction: facts through the engine, variables through storage.
    async fn apply_extraction(
        &self,
        scope: &MemoryScope,
        agent_config: &AgentMemoryConfig,
        options: &AddOptions,
        payload: ExtractionPayload,
        result: &mut AddMemResult,
    ) -> Result<()> {
        let config = self.config();
        let ExtractionPayload { facts, variables } = payload;
        let candidates: Vec<NewMemory> = facts
            .into_iter()
            .filter(|fact| agent_config.kind_enabled(fact.kind))
            .take(config.extraction.max_facts)
            .collect();
        if !candidates.is_empty() {
            let overrides = self.resolve_scope_overrides(scope).await?;
            let outcome = self
                .engine
                .add_many_with_embedding(scope, candidates, options, overrides.embedding())
                .await?;
            record_outcome(outcome, result);
        }
        for (name, value) in variables {
            if config
                .forbidden_variables
                .iter()
                .any(|forbidden| forbidden == &name)
            {
                continue;
            }
            if validate_variable_name(&name).is_err() {
                tracing::warn!(
                    target: TARGET,
                    variable = %name,
                    "skipping an extracted variable with an invalid name"
                );
                continue;
            }
            if !agent_config
                .mem_variables
                .iter()
                .any(|allowed| allowed == &name)
            {
                continue;
            }
            // The projection-aware write, so the extracted variable is a real variable
            // record rather than a KV-only entry. A name the write path rejects (for
            // example a case variant of a `forbidden_variables` entry) skips the variable
            // with a warning, exactly like an invalid name: extraction never fails the
            // turn. Cancellation and storage errors still propagate.
            match self.assign_variable(scope, &name, &value, options).await {
                Ok(_) => {
                    result.variables.insert(name, value);
                }
                Err(Error::Validation(message)) => {
                    tracing::warn!(
                        target: TARGET,
                        variable = %name,
                        %message,
                        "skipping an extracted variable the variable write path rejected"
                    );
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Writes the deterministic fallback records: episodes plus one history summary.
    async fn apply_heuristics(
        &self,
        scope: &MemoryScope,
        agent_config: &AgentMemoryConfig,
        options: &AddOptions,
        messages: &[BaseMessage],
        result: &mut AddMemResult,
    ) -> Result<()> {
        let config = self.config();
        let mut candidates = Vec::new();
        if agent_config.kind_enabled(MemoryType::EpisodicMemory) {
            for message in messages {
                if !(message.role.is_user() || message.role.is_assistant()) {
                    continue;
                }
                let text = message.text();
                let trimmed = text.trim();
                if trimmed.len() < config.extraction.min_message_len {
                    continue;
                }
                candidates.push(NewMemory::episodic(trimmed.to_owned()));
            }
            candidates.truncate(config.extraction.max_facts);
        }
        if agent_config.kind_enabled(MemoryType::Summary)
            && candidates.len() < config.extraction.max_facts
        {
            let history = self
                .engine
                .message_store()
                .recent(scope, config.extraction.use_history_messages.max(1))
                .await?;
            let text = render_history(&history);
            if !text.is_empty() {
                let summary = truncate_to_tokens(
                    &text,
                    config.single_turn_history_summary_max_token,
                    self.engine.token_counter(),
                );
                if !summary.is_empty() {
                    candidates.push(NewMemory::summary(summary));
                }
            }
        }
        if candidates.is_empty() {
            return Ok(());
        }
        let overrides = self.resolve_scope_overrides(scope).await?;
        let outcome = self
            .engine
            .add_many_with_embedding(scope, candidates, options, overrides.embedding())
            .await?;
        record_outcome(outcome, result);
        Ok(())
    }
}

/// Truncates every message's content to `max_chars` characters, keeping the role and the
/// remaining fields.
fn sanitize_messages(messages: &[BaseMessage], max_chars: usize) -> Vec<BaseMessage> {
    messages
        .iter()
        .map(|message| {
            let text = message.text();
            let mut sanitized = message.clone();
            sanitized.content = MessageContent::Text(text.chars().take(max_chars).collect());
            sanitized
        })
        .collect()
}

/// Maps one stored row back to a [`BaseMessage`]; unknown roles default to `User`.
fn message_from_stored(stored: StoredMessage) -> BaseMessage {
    let role = match Role::from(stored.role.as_str()) {
        Role::Other(raw) => {
            tracing::warn!(
                target: TARGET,
                role = %raw,
                "unknown stored message role; treating it as user"
            );
            Role::User
        }
        role => role,
    };
    BaseMessage {
        role,
        content: MessageContent::Text(stored.content),
        ..BaseMessage::default()
    }
}

/// Renders `role: content` lines for the user/assistant messages of a history.
fn render_history(history: &[StoredMessage]) -> String {
    history
        .iter()
        .filter(|message| message.role == "user" || message.role == "assistant")
        .map(|message| format!("{}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Builds the user prompt of the extraction call.
fn build_extraction_prompt(
    history: &[StoredMessage],
    messages: &[BaseMessage],
    agent_config: &AgentMemoryConfig,
) -> String {
    let mut prompt = String::new();
    if !history.is_empty() {
        prompt.push_str("Recent conversation history:\n");
        for message in history {
            let _ = writeln!(prompt, "{}: {}", message.role, message.content);
        }
        prompt.push('\n');
    }
    prompt.push_str("New conversation:\n");
    for message in messages {
        let _ = writeln!(prompt, "{}: {}", message.role.as_str(), message.text());
    }
    prompt.push('\n');
    prompt.push_str(
        "Extract durable memory facts from the new conversation and answer with JSON:\n\
         {\"facts\":[{\"content\":\"...\",\
         \"type\":\"user_profile|semantic_memory|episodic_memory|summary\",\
         \"importance\":0.5}],\"variables\":{\"name\":\"value\"}}\n",
    );
    if agent_config.mem_variables.is_empty() {
        // Naming the key keeps the answer non-empty: a payload with neither facts nor
        // variables is rejected and retried, so an ambiguous "empty object" instruction
        // would only burn the retry budget.
        prompt.push_str(
            "The agent exposes no variables: omit `variables` (or leave it empty) and \
             still report the facts you find.\n",
        );
    } else {
        let _ = writeln!(
            prompt,
            "Only these variables may be extracted: {}.",
            agent_config.mem_variables.join(", ")
        );
    }
    prompt.push_str("Answer with JSON only.\n");
    prompt
}

/// Parses an extraction response into facts and variables, tolerating fences.
fn parse_extraction(response: &ChatResponse) -> Result<ExtractionPayload> {
    let message = response
        .message()
        .ok_or_else(|| Error::memory("memory extraction response carried no message"))?;
    let text = message.text();
    let json = extract_json_object(&text)
        .ok_or_else(|| Error::memory("memory extraction response contained no JSON object"))?;
    let value: Value = serde_json::from_str(json)
        .map_err(|error| Error::memory(format!("malformed memory extraction JSON: {error}")))?;
    let Value::Object(object) = value else {
        return Err(Error::memory(
            "memory extraction JSON must be an object at the top level",
        ));
    };
    Ok(ExtractionPayload {
        facts: parse_facts(&object),
        variables: parse_variables(&object),
    })
}

/// Returns the JSON object inside `text`, stripping Markdown fences when present.
fn extract_json_object(text: &str) -> Option<&str> {
    let mut body = text.trim();
    if let Some(rest) = body.strip_prefix("```") {
        body = rest.split_once('\n').map_or(rest, |(_tag, rest)| rest);
        body = body.trim_end();
        body = body.strip_suffix("```").map_or(body, str::trim);
    }
    let start = body.find('{')?;
    let end = body.rfind('}')?;
    (start <= end).then(|| &body[start..=end])
}

/// Parses the `facts` array; unusable entries are skipped, never fatal.
///
/// A fact without a `type` is a semantic memory (the documented default). A fact with a
/// `type` the crate does not know is kept as a semantic memory with a warning naming the
/// value: [`MemoryType::Unknown`] is not an enabled kind, so keeping it would silently
/// drop a model-extracted fact (Python parity: an unknown type is tolerated, not fatal).
fn parse_facts(object: &Map<String, Value>) -> Vec<NewMemory> {
    let Some(Value::Array(items)) = object.get("facts") else {
        return Vec::new();
    };
    let mut facts = Vec::new();
    for item in items {
        let Value::Object(fact) = item else {
            continue;
        };
        let Some(content) = fact.get("content").and_then(Value::as_str) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }
        let kind = match fact.get("type").and_then(Value::as_str) {
            Some(raw) => match MemoryType::from_str(raw).unwrap_or_default() {
                // `Variable` is a projection-only kind and `Unknown` is not a kind at
                // all: neither is ever an enabled fact kind, so both would be dropped
                // by the agent's kind filter.
                kind @ (MemoryType::Unknown | MemoryType::Variable) => {
                    tracing::warn!(
                        target: TARGET,
                        kind = %raw,
                        parsed = %kind,
                        "unrecognized extracted fact type; treating the fact as semantic memory"
                    );
                    MemoryType::SemanticMemory
                }
                kind => kind,
            },
            None => MemoryType::SemanticMemory,
        };
        let mut memory = NewMemory::new(content.trim(), kind);
        if let Some(importance) = fact
            .get("importance")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
        {
            memory = memory.with_importance(importance.clamp(0.0, 1.0) as f32);
        }
        facts.push(memory);
    }
    facts
}

/// Parses the `variables` object, coercing scalars to strings and skipping nulls.
fn parse_variables(object: &Map<String, Value>) -> BTreeMap<String, String> {
    let Some(Value::Object(items)) = object.get("variables") else {
        return BTreeMap::new();
    };
    let mut variables = BTreeMap::new();
    for (name, value) in items {
        let text = match value {
            Value::String(text) => text.clone(),
            Value::Number(number) => number.to_string(),
            Value::Bool(flag) => flag.to_string(),
            Value::Null | Value::Array(_) | Value::Object(_) => continue,
        };
        variables.insert(name.clone(), text);
    }
    variables
}

/// Truncates `text` to `budget` tokens by accumulating whitespace-separated words.
///
/// The first word is always kept, so a budget of zero still yields one word; text that
/// already fits is returned unchanged (newlines preserved).
fn truncate_to_tokens(
    text: &str,
    budget: usize,
    counter: &(dyn TokenCounter + Send + Sync),
) -> String {
    if counter.count_text(text) <= budget {
        return text.to_owned();
    }
    let mut kept = String::new();
    for (index, word) in text.split_whitespace().enumerate() {
        let candidate = if kept.is_empty() {
            word.to_owned()
        } else {
            format!("{kept} {word}")
        };
        if index > 0 && counter.count_text(&candidate) > budget {
            break;
        }
        kept = candidate;
    }
    kept
}

/// Groups the records an engine write produced into an [`AddMemResult`] by kind.
///
/// The skipped and deleted ids the engine reported are preserved as counts, so a batch
/// whose inputs were all already stored (or whose rows a replace-style write removed) is
/// visible in the result instead of looking like an empty extraction.
fn record_outcome(outcome: AddOutcome, result: &mut AddMemResult) {
    result.skipped += outcome.skipped.len();
    result.deleted += outcome.deleted.len();
    for record in outcome.added.into_iter().chain(outcome.merged) {
        push_record(result, record);
    }
}

/// Appends one record to its kind bucket, warning on kinds Python never returns.
fn push_record(result: &mut AddMemResult, record: MemoryRecord) {
    match record.kind {
        MemoryType::UserProfile => result.user_profile.push(record),
        MemoryType::SemanticMemory => result.semantic_memory.push(record),
        MemoryType::EpisodicMemory => result.episodic_memory.push(record),
        MemoryType::Summary => result.summary.push(record),
        MemoryType::Coding => result.coding.push(record),
        MemoryType::Variable | MemoryType::Unknown => tracing::warn!(
            target: TARGET,
            kind = %record.kind,
            id = %record.id,
            "skipping an extracted memory of an unsupported kind"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use chrono::DateTime;
    use openjiuwen_common::error::Error as CommonError;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use tokio_util::sync::CancellationToken;

    use crate::config::MemoryEngineConfig;
    use crate::record::DedupePolicy;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn ts(offset: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed time")
    }

    async fn facade(config: MemoryEngineConfig) -> Arc<LongTermMemory> {
        LongTermMemory::in_memory(config).await.expect("facade")
    }

    fn long_messages() -> Vec<BaseMessage> {
        vec![
            BaseMessage::user("I always prefer dark mode in every editor"),
            BaseMessage::assistant("Noted, dark mode it is for your editors"),
        ]
    }

    #[tokio::test]
    async fn heuristics_group_episodes_and_one_summary() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        assert_eq!(result.episodic_memory.len(), 2);
        assert_eq!(result.summary.len(), 1);
        assert!(result.summary[0].content.contains("dark mode"));
        assert!(result.variables.is_empty());
        // The history is recorded regardless of extraction.
        let history = ltm
            .get_recent_messages(&scope(), 10)
            .await
            .expect("history");
        assert_eq!(history.len(), 2);
        // Both turns share the batch timestamp, so only the content pairing is asserted.
        let user_turn = history
            .iter()
            .find(|message| message.role.is_user())
            .expect("user turn");
        assert_eq!(
            user_turn.text().as_ref(),
            "I always prefer dark mode in every editor"
        );
        assert!(history.iter().any(|message| message.role.is_assistant()));
    }

    #[tokio::test]
    async fn disabling_episodic_memory_keeps_the_summary() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let agent = AgentMemoryConfig {
            enable_episodic_memory: false,
            ..Default::default()
        };
        let result = ltm
            .add_messages(&scope(), &long_messages(), &agent, &AddOptions::default())
            .await
            .expect("add messages");
        assert_eq!(result.episodic_memory.len(), 0);
        assert_eq!(result.summary.len(), 1);
    }

    #[tokio::test]
    async fn short_messages_and_a_disabled_summary_produce_nothing() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let short = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
        let agent = AgentMemoryConfig {
            enable_summary_memory: false,
            ..Default::default()
        };
        let result = ltm
            .add_messages(&scope(), &short, &agent, &AddOptions::default())
            .await
            .expect("add messages");
        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn max_facts_caps_the_heuristic_candidates() {
        let config = MemoryEngineConfig {
            extraction: crate::config::ExtractionConfig {
                max_facts: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let ltm = facade(config).await;
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        assert_eq!(result.len(), 1);
        assert_eq!(result.episodic_memory.len(), 1);
        assert_eq!(result.summary.len(), 0);
    }

    #[tokio::test]
    async fn content_is_truncated_to_input_msg_max_len() {
        let config = MemoryEngineConfig {
            input_msg_max_len: 5,
            ..Default::default()
        };
        let ltm = facade(config).await;
        ltm.add_messages(
            &scope(),
            &[BaseMessage::user("abcdefghij")],
            &AgentMemoryConfig::default(),
            &AddOptions::default(),
        )
        .await
        .expect("add messages");
        let history = ltm
            .get_recent_messages(&scope(), 10)
            .await
            .expect("history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].text().as_ref(), "abcde");
    }

    #[tokio::test]
    async fn extraction_disabled_returns_empty_but_records_history() {
        let config = MemoryEngineConfig {
            extraction: crate::config::ExtractionConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let ltm = facade(config).await;
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        assert!(result.is_empty());
        assert_eq!(
            ltm.get_recent_messages(&scope(), 10)
                .await
                .expect("history")
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn long_term_memory_disabled_returns_empty() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let agent = AgentMemoryConfig {
            enable_long_term_mem: false,
            ..Default::default()
        };
        let result = ltm
            .add_messages(&scope(), &long_messages(), &agent, &AddOptions::default())
            .await
            .expect("add messages");
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn a_batch_without_a_user_turn_extracts_nothing() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let result = ltm
            .add_messages(
                &scope(),
                &[BaseMessage::assistant("no user turn in this batch")],
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        assert!(result.is_empty());
        assert_eq!(
            ltm.get_recent_messages(&scope(), 10)
                .await
                .expect("history")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn the_model_payload_is_grouped_and_its_variables_are_set() {
        let payload = r#"{"facts":[{"content":"Alice prefers dark mode","type":"user_profile","importance":0.9},
            {"content":"Alice uses Rust","type":"semantic_memory","importance":2.5},
            {"content":"   ","type":"semantic_memory"}],"variables":{"theme":"dark","secret":null}}"#;
        let scripted = Arc::new(ScriptedModel::text(payload));
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(scripted.clone())
            .build()
            .expect("facade");
        let agent = AgentMemoryConfig {
            mem_variables: vec!["theme".to_owned()],
            ..Default::default()
        };
        let result = ltm
            .add_messages(&scope(), &long_messages(), &agent, &AddOptions::default())
            .await
            .expect("add messages");
        assert_eq!(scripted.call_count(), 1);
        assert_eq!(result.user_profile.len(), 1);
        assert_eq!(result.semantic_memory.len(), 1);
        assert!((result.semantic_memory[0].importance - 1.0).abs() < 1e-6);
        assert_eq!(result.episodic_memory.len(), 0);
        assert_eq!(
            result.variables.get("theme").map(String::as_str),
            Some("dark")
        );
        let stored = ltm
            .get_variables(&scope(), Some(&["theme".to_owned()]))
            .await
            .expect("variables");
        assert_eq!(stored.get("theme").map(String::as_str), Some("dark"));
        // A variable outside `mem_variables` is never persisted.
        assert_eq!(
            ltm.get_variables(&scope(), Some(&["secret".to_owned()]))
                .await
                .expect("variables")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn a_fenced_payload_is_accepted() {
        let payload = "```json\n{\"facts\":[{\"content\":\"Alice likes tea\",\
                       \"type\":\"semantic_memory\"}]}\n```";
        let scripted = Arc::new(ScriptedModel::text(payload));
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(scripted.clone())
            .build()
            .expect("facade");
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        assert_eq!(scripted.call_count(), 1);
        assert_eq!(result.semantic_memory.len(), 1);
        assert_eq!(result.semantic_memory[0].content, "Alice likes tea");
    }

    #[tokio::test]
    async fn malformed_model_output_falls_back_to_heuristics() {
        let scripted = Arc::new(ScriptedModel::text("no json here at all"));
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(scripted.clone())
            .build()
            .expect("facade");
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        assert!(scripted.call_count() >= 1);
        assert_eq!(result.episodic_memory.len(), 2);
        assert_eq!(result.summary.len(), 1);
        let degraded = result.degraded.expect("degradation reason");
        assert!(degraded.contains("heuristic"), "{degraded}");
    }

    #[tokio::test]
    async fn a_failing_model_reports_why_it_degraded_and_still_stores_facts() {
        // Every retry fails the same way, so the reported reason is the provider error.
        let scripted = Arc::new(ScriptedModel::new([
            ScriptedTurn::fail("provider exploded"),
            ScriptedTurn::fail("provider exploded"),
            ScriptedTurn::fail("provider exploded"),
        ]));
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(scripted.clone())
            .build()
            .expect("facade");
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        let degraded = result.degraded.clone().expect("degradation reason");
        assert!(degraded.contains("provider exploded"), "{degraded}");
        // The heuristic records are real: they are returned and persisted.
        assert_eq!(result.episodic_memory.len(), 2);
        assert_eq!(result.summary.len(), 1);
        let stored = ltm.list(&scope(), &[]).await.expect("list");
        assert_eq!(stored.len(), 3);
        assert_eq!(scripted.call_count(), 3, "every attempt failed");
    }

    #[tokio::test]
    async fn an_applied_payload_is_not_reported_as_degraded() {
        let scripted = Arc::new(ScriptedModel::text(
            r#"{"facts":[{"content":"Alice prefers tea","type":"user_profile"}]}"#,
        ));
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(scripted)
            .build()
            .expect("facade");
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        assert_eq!(result.user_profile.len(), 1);
        assert_eq!(result.degraded, None, "no fallback ran");
        assert_eq!(result.skipped, 0);
        assert_eq!(result.deleted, 0);
    }

    #[tokio::test]
    async fn an_unrecognized_fact_type_is_kept_as_semantic_memory() {
        let payload = r#"{"facts":[
            {"content":"Alice prefers tea over coffee","type":"preference"},
            {"content":"Alice works in Berlin","type":"variable"},
            {"content":"Alice ships Rust","type":"semantic_memory"}]}"#;
        let scripted = Arc::new(ScriptedModel::text(payload));
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(scripted)
            .build()
            .expect("facade");
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        // Nothing is dropped: the unrecognized types fall back to semantic memory.
        assert_eq!(result.semantic_memory.len(), 3);
        assert_eq!(result.episodic_memory.len(), 0);
        assert_eq!(result.degraded, None);
        assert!(result
            .semantic_memory
            .iter()
            .all(|record| record.kind == MemoryType::SemanticMemory));
        let stored = ltm.list(&scope(), &[]).await.expect("list");
        assert_eq!(stored.len(), 3);
        assert!(stored
            .iter()
            .any(|record| record.content == "Alice prefers tea over coffee"));
    }

    #[tokio::test]
    async fn an_empty_model_payload_does_not_suppress_the_heuristics() {
        for payload in ["{}", r#"{"facts":[]}"#] {
            let scripted = Arc::new(ScriptedModel::text(payload));
            let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
                .model(scripted.clone())
                .build()
                .expect("facade");
            let result = ltm
                .add_messages(
                    &scope(),
                    &long_messages(),
                    &AgentMemoryConfig::default(),
                    &AddOptions::default(),
                )
                .await
                .expect("add messages");
            assert!(
                scripted.call_count() >= 1,
                "{payload} came from a real model call"
            );
            assert_eq!(result.episodic_memory.len(), 2, "{payload}");
            assert_eq!(result.summary.len(), 1, "{payload}");
            assert!(result.degraded.is_some(), "{payload}");
        }
    }

    #[tokio::test]
    async fn a_truly_unextractable_batch_stays_empty_without_a_model() {
        // The no-model path is the documented heuristic route, not a degradation, and a
        // batch with nothing worth storing still yields an empty result.
        let ltm = facade(MemoryEngineConfig::default()).await;
        let short = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
        let agent = AgentMemoryConfig {
            enable_summary_memory: false,
            ..Default::default()
        };
        let result = ltm
            .add_messages(&scope(), &short, &agent, &AddOptions::default())
            .await
            .expect("add messages");
        assert!(result.is_empty());
        assert_eq!(result.degraded, None);
    }

    #[tokio::test]
    async fn a_skipped_duplicate_is_counted_in_the_result() {
        let scripted = Arc::new(ScriptedModel::text(
            r#"{"facts":[{"content":"Alice keeps every editor in dark mode",
                           "type":"semantic_memory"}]}"#,
        ));
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(scripted)
            .build()
            .expect("facade");
        ltm.add(
            &scope(),
            NewMemory::semantic("Alice keeps every editor in dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("seed record");
        let options = AddOptions::default().with_dedupe(DedupePolicy::Skip);
        let result = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &options,
            )
            .await
            .expect("add messages");
        assert_eq!(result.skipped, 1, "the duplicate is reported, not hidden");
        assert_eq!(result.semantic_memory.len(), 0);
        assert_eq!(result.degraded, None);
    }

    #[test]
    fn record_outcome_keeps_the_skipped_and_deleted_counts() {
        let mut result = AddMemResult::default();
        record_outcome(
            AddOutcome {
                skipped: vec!["skipped-1".to_owned(), "skipped-2".to_owned()],
                deleted: vec!["deleted-1".to_owned()],
                ..Default::default()
            },
            &mut result,
        );
        assert_eq!(result.skipped, 2);
        assert_eq!(result.deleted, 1);
        assert!(result.is_empty(), "counts are not records");
    }

    #[tokio::test]
    async fn forbidden_and_invalid_variables_are_skipped() {
        let payload = r#"{"facts":[],"variables":{"api_key":"x","bad/name":"y","city":"Berlin"}}"#;
        let scripted = Arc::new(ScriptedModel::text(payload));
        let config = MemoryEngineConfig {
            forbidden_variables: vec!["api_key".to_owned()],
            ..Default::default()
        };
        let ltm = LongTermMemory::builder(config)
            .model(scripted)
            .build()
            .expect("facade");
        let agent = AgentMemoryConfig {
            mem_variables: vec![
                "api_key".to_owned(),
                "bad/name".to_owned(),
                "city".to_owned(),
            ],
            ..Default::default()
        };
        let result = ltm
            .add_messages(&scope(), &long_messages(), &agent, &AddOptions::default())
            .await
            .expect("add messages");
        assert_eq!(result.variables.len(), 1);
        assert_eq!(
            result.variables.get("city").map(String::as_str),
            Some("Berlin")
        );
    }

    #[tokio::test]
    async fn a_cancelled_call_returns_cancelled_and_records_nothing() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let token = CancellationToken::new();
        token.cancel();
        let options = AddOptions::default().with_cancellation(token);
        let error = ltm
            .add_messages(
                &scope(),
                &long_messages(),
                &AgentMemoryConfig::default(),
                &options,
            )
            .await
            .expect_err("cancelled");
        assert!(matches!(error, CommonError::Cancelled), "{error:?}");
        assert_eq!(
            ltm.get_recent_messages(&scope(), 10)
                .await
                .expect("history")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn an_empty_batch_is_a_no_op() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let result = ltm
            .add_messages(
                &scope(),
                &[],
                &AgentMemoryConfig::default(),
                &AddOptions::default(),
            )
            .await
            .expect("add messages");
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn recent_messages_clamp_to_the_limit_and_are_chronological() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let messages = [
            BaseMessage::user("first message of the history"),
            BaseMessage::assistant("second message of the history"),
            BaseMessage::user("third message of the history"),
        ];
        for (index, message) in messages.iter().enumerate() {
            ltm.engine()
                .message_store()
                .append(
                    &scope(),
                    std::slice::from_ref(message),
                    ts(i64::try_from(index).expect("small index")),
                )
                .await
                .expect("append");
        }
        let recent = ltm.get_recent_messages(&scope(), 10).await.expect("recent");
        assert_eq!(recent.len(), 3);
        assert!(recent[0].role.is_user());
        assert!(recent[1].role.is_assistant());
        assert_eq!(recent[2].text().as_ref(), "third message of the history");

        let clamped = ltm.get_recent_messages(&scope(), 2).await.expect("clamped");
        assert_eq!(clamped.len(), 2);
        assert!(clamped[0].role.is_assistant());
        assert_eq!(clamped[1].text().as_ref(), "third message of the history");
    }

    #[tokio::test]
    async fn unknown_stored_roles_map_to_user() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let mut message = BaseMessage::user("native role message");
        message.role = Role::Other("native".to_owned());
        ltm.engine()
            .message_store()
            .append(&scope(), &[message], ts(0))
            .await
            .expect("append");
        let recent = ltm.get_recent_messages(&scope(), 10).await.expect("recent");
        assert_eq!(recent.len(), 1);
        assert!(recent[0].role.is_user());
    }

    #[test]
    fn json_objects_are_extracted_from_fences_and_prose() {
        assert_eq!(
            extract_json_object("```json\n{\"a\":1}\n```"),
            Some("{\"a\":1}")
        );
        assert_eq!(
            extract_json_object("sure: {\"a\":1} done"),
            Some("{\"a\":1}")
        );
        assert_eq!(extract_json_object("```json\n{\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(extract_json_object("no json"), None);
        assert_eq!(extract_json_object("} backwards {"), None);
    }

    #[test]
    fn token_truncation_keeps_the_budget_and_one_word() {
        let counter = openjiuwen_foundation::prompt::tokens::ApproxTokenCounter::new();
        assert_eq!(
            truncate_to_tokens("one two three", 100, &counter),
            "one two three"
        );
        let truncated = truncate_to_tokens("one two three four five six", 2, &counter);
        assert!(counter.count_text(&truncated) <= 2, "{truncated}");
        assert_eq!(truncate_to_tokens("single", 0, &counter), "single");
        assert_eq!(truncate_to_tokens("", 4, &counter), "");
    }

    /// Appends `count` user messages with fixed one-second-apart timestamps.
    async fn seeded_history(ltm: &LongTermMemory, scope: &MemoryScope, count: usize) {
        for index in 0..count {
            let text = format!("history message {index}");
            let offset = i64::try_from(index).expect("small index");
            let message = BaseMessage::user(text);
            ltm.engine()
                .message_store()
                .append(scope, &[message], ts(offset))
                .await
                .expect("append");
        }
    }

    /// Walks every page of a history and returns the contents in order.
    async fn walk(ltm: &LongTermMemory, scope: &MemoryScope, page_size: usize) -> Vec<String> {
        let mut contents = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let page = ltm
                .get_message_page(scope, page_size, cursor.as_deref(), None)
                .await
                .expect("page");
            assert!(
                page.items.len() <= page_size,
                "a page never exceeds its size"
            );
            contents.extend(page.items.iter().map(|message| message.content.clone()));
            let Some(next) = page.next_cursor.clone() else {
                assert!(!page.has_more, "no cursor on the last page");
                break;
            };
            assert!(page.has_more, "a cursor implies a later page");
            cursor = Some(next);
        }
        contents
    }

    #[tokio::test]
    async fn message_pages_walk_the_history_once_in_ascending_order() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        seeded_history(&ltm, &scope(), 5).await;
        assert_eq!(
            walk(&ltm, &scope(), 2).await,
            [
                "history message 0",
                "history message 1",
                "history message 2",
                "history message 3",
                "history message 4",
            ]
        );
        // Five rows in pages of two is three pages.
        let mut pages = 0;
        let mut cursor: Option<String> = None;
        loop {
            let page = ltm
                .page_messages(&scope(), 2, cursor.as_deref())
                .await
                .expect("page");
            pages += 1;
            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        assert_eq!(pages, 3);
        let first = ltm
            .page_messages(&scope(), 2, None)
            .await
            .expect("first page");
        assert_eq!(first.items.len(), 2);
        assert!(first.has_more);
        assert_eq!(first.items[0].content, "history message 0");
        assert_eq!(first.items[1].content, "history message 1");
    }

    #[tokio::test]
    async fn a_tampered_or_garbage_cursor_is_a_validation_error() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        seeded_history(&ltm, &scope(), 5).await;
        let error = ltm
            .get_message_page(&scope(), 2, Some("not-a-cursor"), None)
            .await
            .expect_err("garbage cursor");
        assert!(matches!(error, CommonError::Validation(_)), "{error:?}");

        let page = ltm
            .get_message_page(&scope(), 2, None, None)
            .await
            .expect("first page");
        let token = page.next_cursor.expect("cursor");
        let mut tampered = token.clone();
        let last = if tampered.ends_with('0') { '1' } else { '0' };
        tampered.pop();
        tampered.push(last);
        assert_ne!(tampered, token);
        let error = ltm
            .get_message_page(&scope(), 2, Some(&tampered), None)
            .await
            .expect_err("tampered cursor");
        assert!(matches!(error, CommonError::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn zero_page_size_and_a_missing_history_are_empty_pages() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let empty = ltm
            .get_message_page(&scope(), 2, None, None)
            .await
            .expect("missing history");
        assert_eq!(empty, MessagePage::default());
        assert!(!empty.has_more);
        assert!(empty.next_cursor.is_none());

        seeded_history(&ltm, &scope(), 3).await;
        let zero = ltm
            .get_message_page(&scope(), 0, None, None)
            .await
            .expect("zero page size");
        assert_eq!(zero, MessagePage::default());
        let oversized = ltm
            .get_message_page(&scope(), 100, None, None)
            .await
            .expect("oversized page");
        assert_eq!(oversized.items.len(), 3);
        assert!(!oversized.has_more);
        assert!(oversized.next_cursor.is_none());
    }

    #[tokio::test]
    async fn the_page_filter_cannot_widen_past_the_scope() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        seeded_history(&ltm, &scope(), 3).await;
        let foreign = MemoryScope::new("bob", "ws");
        ltm.engine()
            .message_store()
            .append(&foreign, &[BaseMessage::user("bob message")], ts(9))
            .await
            .expect("append bob");

        let page = ltm
            .get_message_page(&scope(), 10, None, Some(&MemoryFilter::new()))
            .await
            .expect("unfiltered clause");
        assert_eq!(page.items.len(), 3);
        assert!(page
            .items
            .iter()
            .all(|message| message.user_id == "alice" && message.scope_id == "ws"));

        let other_user = MemoryFilter {
            user_id: Some("bob".to_owned()),
            ..MemoryFilter::default()
        };
        let page = ltm
            .get_message_page(&scope(), 10, None, Some(&other_user))
            .await
            .expect("contradicting filter");
        assert!(page.items.is_empty(), "the scope clause always wins");

        let bounded = MemoryFilter {
            created_after: Some(ts(0)),
            ..MemoryFilter::default()
        };
        let page = ltm
            .get_message_page(&scope(), 10, None, Some(&bounded))
            .await
            .expect("bounded filter");
        assert_eq!(page.items.len(), 2);
        assert_eq!(page.items[0].content, "history message 1");
    }

    #[tokio::test]
    async fn the_legacy_recent_window_is_unchanged_by_pagination() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        seeded_history(&ltm, &scope(), 5).await;
        let _ = ltm
            .get_message_page(&scope(), 2, None, None)
            .await
            .expect("page");
        let recent = ltm.get_recent_messages(&scope(), 2).await.expect("recent");
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].text().as_ref(), "history message 3");
        assert_eq!(recent[1].text().as_ref(), "history message 4");
    }

    #[tokio::test]
    async fn clearing_one_tenant_leaves_the_other_tenant_history_intact() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let alice = scope();
        let bob = MemoryScope::new("bob", "ws").session("s1");
        seeded_history(&ltm, &alice, 3).await;
        seeded_history(&ltm, &bob, 2).await;

        let deleted = ltm
            .clear_messages(&MemoryFilter::for_scope(&alice), &CallOptions::default())
            .await
            .expect("clear alice");
        assert_eq!(deleted, 3);
        assert_eq!(
            ltm.get_recent_messages(&alice, 10)
                .await
                .expect("alice")
                .len(),
            0
        );
        let bob_history = ltm.get_recent_messages(&bob, 10).await.expect("bob");
        assert_eq!(bob_history.len(), 2, "bob's history is untouched");
        assert_eq!(bob_history[0].text().as_ref(), "history message 0");
        assert_eq!(bob_history[1].text().as_ref(), "history message 1");

        // The narrower forms still only reach the named tenant.
        let mut by_id = MemoryFilter::for_scope(&bob);
        by_id.ids = vec![
            ltm.engine()
                .message_store()
                .recent(&bob, 1)
                .await
                .expect("bob tail")
                .remove(0)
                .id,
        ];
        assert_eq!(
            ltm.clear_messages(&by_id, &CallOptions::default())
                .await
                .expect("clear one row"),
            1
        );
        assert_eq!(
            ltm.get_recent_messages(&bob, 10).await.expect("bob").len(),
            1
        );
        assert_eq!(
            ltm.get_recent_messages(&alice, 10)
                .await
                .expect("alice")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn clearing_without_a_scope_is_refused_and_deletes_nothing() {
        let ltm = facade(MemoryEngineConfig::default()).await;
        let alice = scope();
        let bob = MemoryScope::new("bob", "ws").session("s1");
        seeded_history(&ltm, &alice, 3).await;
        seeded_history(&ltm, &bob, 2).await;

        let unscoped = [
            (MemoryFilter::default(), vec!["user_id", "scope_id"]),
            (
                MemoryFilter {
                    user_id: Some("alice".to_owned()),
                    ..MemoryFilter::default()
                },
                vec!["scope_id"],
            ),
            (
                MemoryFilter {
                    scope_id: Some("ws".to_owned()),
                    ..MemoryFilter::default()
                },
                vec!["user_id"],
            ),
        ];
        for (filter, missing) in unscoped {
            let error = ltm
                .clear_messages(&filter, &CallOptions::default())
                .await
                .expect_err("an unscoped clear is refused");
            assert!(matches!(error, CommonError::Validation(_)), "{error:?}");
            let message = error.to_string();
            for field in missing {
                assert!(message.contains(field), "{field} missing from {message}");
            }
        }
        assert_eq!(
            ltm.get_recent_messages(&alice, 10)
                .await
                .expect("alice")
                .len(),
            3,
            "nothing was deleted"
        );
        assert_eq!(
            ltm.get_recent_messages(&bob, 10).await.expect("bob").len(),
            2,
            "nothing was deleted"
        );
    }

    #[tokio::test]
    async fn model_variables_are_projected_as_variable_records() {
        let scripted = Arc::new(ScriptedModel::text(
            r#"{"facts":[],"variables":{"theme":"dark"}}"#,
        ));
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .model(scripted)
            .build()
            .expect("facade");
        let agent = AgentMemoryConfig {
            mem_variables: vec!["theme".to_owned()],
            ..Default::default()
        };
        let result = ltm
            .add_messages(&scope(), &long_messages(), &agent, &AddOptions::default())
            .await
            .expect("add messages");
        assert_eq!(
            result.variables.get("theme").map(String::as_str),
            Some("dark")
        );
        assert_eq!(
            ltm.get_variables(&scope(), None)
                .await
                .expect("kv")
                .get("theme")
                .map(String::as_str),
            Some("dark")
        );
        let records = ltm.variable_records(&scope()).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].kind, MemoryType::Variable);
        assert_eq!(records[0].content, "theme=dark");
    }
}
