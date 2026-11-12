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
//! warning. Only cancellation is propagated, and already-appended history stays.
//!
//! [`LongTermMemory::get_recent_messages`] reads the history back as foundation
//! [`BaseMessage`]s in chronological order.

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
    AddMemResult, AddOptions, AddOutcome, MemoryRecord, MemoryScope, MemoryType, NewMemory,
};
use crate::store::StoredMessage;

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
    /// failure never fails the call: it degrades to the heuristic path.
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
            Some(payload) => {
                self.apply_extraction(scope, agent_config, options, payload, &mut result)
                    .await?;
            }
            None => {
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

    /// Runs one extraction prompt, retrying on model or parse failures.
    ///
    /// Returns `Ok(None)` when there is no model or every attempt failed (the caller
    /// falls back to heuristics), and [`Error::Cancelled`] as soon as the call is
    /// cancelled — a cancelled extraction is never retried.
    async fn extract_with_model(
        &self,
        scope: &MemoryScope,
        messages: &[BaseMessage],
        agent_config: &AgentMemoryConfig,
        options: &AddOptions,
    ) -> Result<Option<ExtractionPayload>> {
        let Some(model) = self.engine.model() else {
            return Ok(None);
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
                Ok(payload) => return Ok(Some(payload)),
                Err(error) => last_error = Some(error),
            }
        }
        if let Some(error) = last_error {
            tracing::warn!(
                target: TARGET,
                %error,
                "memory extraction failed; falling back to heuristics"
            );
        }
        Ok(None)
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
            let outcome = self.engine.add_many(scope, candidates, options).await?;
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
            self.set_variable(scope, &name, &value).await?;
            result.variables.insert(name, value);
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
        let outcome = self.engine.add_many(scope, candidates, options).await?;
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
        prompt.push_str("The agent exposes no variables: answer with an empty object.\n");
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
            Some(raw) => MemoryType::from_str(raw).unwrap_or(MemoryType::Unknown),
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
fn record_outcome(outcome: AddOutcome, result: &mut AddMemResult) {
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
    use openjiuwen_test_support::ScriptedModel;
    use tokio_util::sync::CancellationToken;

    use crate::config::MemoryEngineConfig;

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
}
