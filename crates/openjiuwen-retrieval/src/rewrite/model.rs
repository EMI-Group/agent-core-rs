//! LLM-backed query rewriting over the [`CompletionModel`] seam.
//!
//! Port of Python's `QueryRewriter.rewrite`: render the `intention_completion`
//! prompt with the recent history and the query, ask the model for one JSON
//! object, then repair the reply against the field schema. Python loads its
//! prompt from `prompts/intention_completion_{lang}.md` and keeps history in a
//! `ModelContext`; Rust takes the template and the turns from
//! [`ModelRewriteConfig`] and [`RewriteRequest`], so no filesystem or session
//! access is needed.
//!
//! Repair is deliberately lenient — missing fields default, scalars are coerced
//! to strings and single-element lists, JSON encoded inside a string is
//! decoded — because an LLM reply is untrusted. A reply that contains no JSON
//! object at all uses [`ModelRewriteConfig::fallback`] when configured and
//! otherwise fails with [`Error::Retrieval`], naming the parse failure.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::model::{extract_json_object, CompletionModel};

use super::heuristic::{HeuristicQueryRewriter, DEFAULT_HISTORY_WINDOW};
use super::{validate_query, QueryRewriter, RewriteOutput, RewriteRequest, TypoFix};

/// Character budget for the model reply quoted in a parse error.
const MAX_ERROR_PREVIEW_CHARS: usize = 500;

/// Key used when a non-object `references` value has to be wrapped.
const REFERENCES_WRAP_KEY: &str = "references";

/// Default rewrite prompt, a Rust port of the Python
/// `prompts/intention_completion_en.md` template.
///
/// `{history}` and `{query}` are the only placeholders; the JSON example keeps
/// its literal braces because the template is filled by explicit replacement
/// rather than `format!`.
pub const DEFAULT_REWRITE_PROMPT: &str = "\
You rewrite a user's latest query into a standalone search query.

Conversation history (oldest first):
{history}

Latest user query:
{query}

Reply with one JSON object and nothing else, using exactly these keys:
{
  \"before\": \"the query exactly as the user wrote it\",
  \"intention\": \"what the user wants, in one sentence\",
  \"standalone_query\": \"a self-contained query usable for retrieval\",
  \"references\": {\"term\": [\"what the term refers to\"]},
  \"missing\": [\"information the query omits but retrieval needs\"],
  \"typo\": [{\"original\": \"mispelled\", \"corrected\": \"misspelled\", \"reason\": \"spelling\"}],
  \"gibberish\": [\"tokens that carry no meaning\"],
  \"from_history\": \"the history turn the query depends on, or an empty string\"
}
";

/// Configuration for [`ModelQueryRewriter`].
#[derive(Clone)]
pub struct ModelRewriteConfig {
    /// Prompt template filled for every call; `{history}` and `{query}` are the
    /// recognised placeholders.
    pub prompt_template: String,
    /// Number of trailing history turns rendered into the prompt.
    pub history_window: usize,
    /// Rewriter used when the model reply contains no parsable JSON object.
    pub fallback: Option<Arc<dyn QueryRewriter>>,
}

impl fmt::Debug for ModelRewriteConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelRewriteConfig")
            .field("prompt_template", &self.prompt_template)
            .field("history_window", &self.history_window)
            .field("has_fallback", &self.fallback.is_some())
            .finish()
    }
}

impl Default for ModelRewriteConfig {
    fn default() -> Self {
        Self {
            prompt_template: DEFAULT_REWRITE_PROMPT.to_owned(),
            history_window: DEFAULT_HISTORY_WINDOW,
            fallback: None,
        }
    }
}

impl ModelRewriteConfig {
    /// Builder: replaces the prompt template.
    #[must_use]
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    /// Builder: sets how many trailing history turns reach the prompt.
    #[must_use]
    pub fn with_history_window(mut self, history_window: usize) -> Self {
        self.history_window = history_window;
        self
    }

    /// Builder: attaches the rewriter used when the reply is unparsable.
    #[must_use]
    pub fn with_fallback(mut self, fallback: Arc<dyn QueryRewriter>) -> Self {
        self.fallback = Some(fallback);
        self
    }
}

/// A query rewriter that asks a [`CompletionModel`] for a JSON rewrite.
///
/// Keyword extraction and expansion are not model calls: both derive from the
/// rewritten [`RewriteOutput::standalone_query`] with a
/// [`HeuristicQueryRewriter`], so they stay deterministic and cheap once the
/// rewrite has happened.
pub struct ModelQueryRewriter {
    /// Model answering the rewrite prompt.
    model: Arc<dyn CompletionModel>,
    /// Prompt template, history window, and fallback policy.
    config: ModelRewriteConfig,
    /// Deterministic keyword extractor applied to the rewritten query.
    extractor: HeuristicQueryRewriter,
}

impl fmt::Debug for ModelQueryRewriter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelQueryRewriter")
            .field("model", &self.model.name())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ModelQueryRewriter {
    /// Builds a rewriter that asks `model` and applies `config`.
    #[must_use]
    pub fn new(model: Arc<dyn CompletionModel>, config: ModelRewriteConfig) -> Self {
        Self {
            model,
            config,
            extractor: HeuristicQueryRewriter::default(),
        }
    }

    /// Builds a rewriter with the default prompt and no fallback.
    #[must_use]
    pub fn with_default_config(model: Arc<dyn CompletionModel>) -> Self {
        Self::new(model, ModelRewriteConfig::default())
    }

    /// The model used for rewriting.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn CompletionModel> {
        &self.model
    }

    /// The rewrite configuration in use.
    #[must_use]
    pub fn config(&self) -> &ModelRewriteConfig {
        &self.config
    }

    /// Renders the rewrite prompt for `request`.
    ///
    /// The query is substituted before the history, so a history turn that
    /// happens to contain the literal `{query}` placeholder cannot be expanded.
    #[must_use]
    pub fn render_prompt(&self, request: &RewriteRequest) -> String {
        let history = self.history_text(&request.history);
        self.config
            .prompt_template
            .replace("{query}", &request.query)
            .replace("{history}", &history)
    }

    /// Renders the trailing `history_window` non-blank turns as `user: <turn>`
    /// lines.
    fn history_text(&self, history: &[String]) -> String {
        let window = self.config.history_window.max(1);
        let turns: Vec<&String> = history
            .iter()
            .filter(|turn| !turn.trim().is_empty())
            .collect();
        let start = turns.len().saturating_sub(window);
        turns[start..]
            .iter()
            .map(|turn| format!("user: {}", turn.trim()))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

#[async_trait]
impl QueryRewriter for ModelQueryRewriter {
    async fn rewrite(&self, request: &RewriteRequest) -> Result<RewriteOutput> {
        validate_query(&request.query)?;
        let prompt = self.render_prompt(request);
        let reply = self.model.complete(&prompt).await?;
        match parse_output(&reply, &request.query) {
            Ok(output) => Ok(output),
            Err(error) => match self.config.fallback.as_ref() {
                Some(fallback) => fallback.rewrite(request).await,
                None => Err(error),
            },
        }
    }

    async fn keywords(&self, query: &str) -> Result<Vec<String>> {
        let request = RewriteRequest::new(query);
        let output = self.rewrite(&request).await?;
        Ok(self.extractor.extract_keywords(&output.standalone_query))
    }

    async fn expand(&self, query: &str) -> Result<Vec<String>> {
        let request = RewriteRequest::new(query);
        let output = self.rewrite(&request).await?;
        Ok(self.extractor.expand_query(&output.standalone_query))
    }
}

/// Parses and repairs a model reply into a [`RewriteOutput`].
fn parse_output(reply: &str, query: &str) -> Result<RewriteOutput> {
    let Some(json) = extract_json_object(reply) else {
        return Err(parse_error(reply, "no JSON object found"));
    };
    let value: Value = serde_json::from_str(json)
        .map_err(|error| parse_error(reply, &format!("invalid JSON: {error}")))?;
    let Value::Object(fields) = value else {
        return Err(parse_error(reply, "JSON value is not an object"));
    };

    let mut standalone_query = string_field(&fields, "standalone_query");
    if standalone_query.trim().is_empty() {
        query.trim().clone_into(&mut standalone_query);
    }

    Ok(RewriteOutput {
        before: string_field(&fields, "before"),
        intention: string_field(&fields, "intention"),
        standalone_query,
        references: object_field(fields.get("references")),
        missing: list_field(fields.get("missing")),
        typo: typo_field(fields.get("typo")),
        gibberish: list_field(fields.get("gibberish")),
        from_history: string_field(&fields, "from_history"),
    })
}

/// Builds the parse failure, quoting a bounded preview of the reply.
fn parse_error(reply: &str, detail: &str) -> Error {
    let preview: String = reply.trim().chars().take(MAX_ERROR_PREVIEW_CHARS).collect();
    Error::retrieval(format!(
        "failed to parse rewrite output: {detail}; content: {preview:?}"
    ))
}

/// Coerces a JSON value to a string.
///
/// Strings pass through; `null` becomes empty; numbers and booleans render as
/// their JSON text and arrays/objects as compact JSON.
fn force_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Reads a string field, defaulting to empty when absent or `null`.
fn string_field(fields: &Map<String, Value>, key: &str) -> String {
    fields.get(key).map_or_else(String::new, force_string)
}

/// Reads a list-of-strings field; scalars become single-element lists.
fn list_field(value: Option<&Value>) -> Vec<String> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(force_string)
            .filter(|item| !item.is_empty())
            .collect(),
        Some(other) => vec![force_string(other)],
    }
}

/// Reads the `references` object, decoding JSON encoded in a string.
fn object_field(value: Option<&Value>) -> Map<String, Value> {
    match value {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(fields)) => fields.clone(),
        Some(Value::String(text)) => {
            if let Ok(Value::Object(fields)) = serde_json::from_str::<Value>(text) {
                fields
            } else {
                wrap_object(Value::String(text.clone()))
            }
        }
        Some(other) => wrap_object(other.clone()),
    }
}

/// Wraps a non-object `references` value under a single key, as Python's
/// `_force_json` does.
fn wrap_object(value: Value) -> Map<String, Value> {
    let mut fields = Map::new();
    fields.insert(REFERENCES_WRAP_KEY.to_owned(), value);
    fields
}

/// Reads the `typo` field, repairing every entry into a [`TypoFix`].
fn typo_field(value: Option<&Value>) -> Vec<TypoFix> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => items.iter().map(typo_fix).collect(),
        Some(other) => vec![typo_fix(other)],
    }
}

/// Repairs a single typo entry; a bare string is treated as the original term.
fn typo_fix(value: &Value) -> TypoFix {
    match value {
        Value::Object(fields) => TypoFix {
            original: string_field(fields, "original"),
            corrected: string_field(fields, "corrected"),
            reason: string_field(fields, "reason"),
        },
        other => TypoFix {
            original: force_string(other),
            corrected: String::new(),
            reason: String::new(),
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::model::StaticCompletionModel;
    use async_trait::async_trait;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn rewriter(reply: &str) -> ModelQueryRewriter {
        ModelQueryRewriter::with_default_config(Arc::new(StaticCompletionModel::single(reply)))
    }

    fn well_formed() -> String {
        json!({
            "before": "what about it",
            "intention": "learn the rust book",
            "standalone_query": "what about the rust book",
            "references": {"it": ["the rust book"]},
            "missing": ["chapter"],
            "typo": [{"original": "rustt", "corrected": "rust", "reason": "spelling"}],
            "gibberish": ["asdf"],
            "from_history": "tell me about the rust book"
        })
        .to_string()
    }

    #[test]
    fn prompt_renders_both_placeholders_and_keeps_json_braces() {
        let rewriter =
            ModelQueryRewriter::with_default_config(Arc::new(StaticCompletionModel::single("{}")));
        let request = RewriteRequest::new("what about it")
            .with_history(["first turn", "tell me about the rust book"]);
        let prompt = rewriter.render_prompt(&request);
        assert!(prompt.contains("user: first turn"));
        assert!(prompt.contains("user: tell me about the rust book"));
        assert!(prompt.contains("Latest user query:\nwhat about it"));
        assert!(prompt.contains("\"standalone_query\""));
        assert!(!prompt.contains("{query}"));
        assert!(!prompt.contains("{history}"));
    }

    #[test]
    fn prompt_respects_the_history_window_and_skips_blank_turns() {
        let config = ModelRewriteConfig::default().with_history_window(1);
        let rewriter =
            ModelQueryRewriter::new(Arc::new(StaticCompletionModel::single("{}")), config);
        let request = RewriteRequest::new("it").with_history(["first turn", "second turn", "   "]);
        let prompt = rewriter.render_prompt(&request);
        assert!(!prompt.contains("first turn"));
        assert!(prompt.contains("user: second turn"));
    }

    #[test]
    fn parses_a_well_formed_reply() {
        let output = parse_output(&well_formed(), "what about it").unwrap();
        assert_eq!(output.before, "what about it");
        assert_eq!(output.intention, "learn the rust book");
        assert_eq!(output.standalone_query, "what about the rust book");
        assert_eq!(output.references["it"], json!(["the rust book"]));
        assert_eq!(output.missing, vec!["chapter"]);
        assert_eq!(output.typo, vec![TypoFix::new("rustt", "rust", "spelling")]);
        assert_eq!(output.gibberish, vec!["asdf"]);
        assert_eq!(output.from_history, "tell me about the rust book");
    }

    #[test]
    fn parses_json_surrounded_by_prose_and_fences() {
        let reply = format!(
            "Sure, here is the rewrite:\n```json\n{}\n```\nLet me know!",
            well_formed()
        );
        let output = parse_output(&reply, "what about it").unwrap();
        assert_eq!(output.standalone_query, "what about the rust book");
    }

    #[test]
    fn missing_fields_default_and_blank_standalone_query_falls_back_to_the_query() {
        let reply = json!({"before": "q", "standalone_query": "   "}).to_string();
        let output = parse_output(&reply, "  what is rust  ").unwrap();
        assert_eq!(output.standalone_query, "what is rust");
        assert_eq!(output.intention, "");
        assert!(output.references.is_empty());
        assert!(output.missing.is_empty());
        assert!(output.typo.is_empty());
        assert!(output.gibberish.is_empty());
        assert_eq!(output.from_history, "");
    }

    #[test]
    fn wrong_types_are_coerced() {
        let reply = json!({
            "before": null,
            "intention": 42,
            "standalone_query": "q",
            "references": "{\"it\": [\"that\"]}",
            "missing": "chapter",
            "typo": "rustt",
            "gibberish": ["ok", null],
            "from_history": false
        })
        .to_string();
        let output = parse_output(&reply, "query").unwrap();
        assert_eq!(output.before, "");
        assert_eq!(output.intention, "42");
        assert_eq!(output.references["it"], json!(["that"]));
        assert_eq!(output.missing, vec!["chapter"]);
        assert_eq!(output.typo, vec![TypoFix::new("rustt", "", "")]);
        assert_eq!(output.gibberish, vec!["ok"]);
        assert_eq!(output.from_history, "false");
    }

    #[test]
    fn non_object_references_and_typo_values_are_wrapped() {
        let reply = json!({"standalone_query": "q", "references": ["a"], "typo": 7}).to_string();
        let output = parse_output(&reply, "q").unwrap();
        assert_eq!(output.references["references"], json!(["a"]));
        assert_eq!(output.typo, vec![TypoFix::new("7", "", "")]);
    }

    #[tokio::test]
    async fn rewrite_sends_the_rendered_prompt_to_the_model() {
        let model = Arc::new(StaticCompletionModel::single(well_formed()));
        let rewriter = ModelQueryRewriter::with_default_config(model.clone());
        let request = RewriteRequest::new("what about it").with_history(["tell me about rust"]);
        let output = rewriter.rewrite(&request).await.unwrap();
        assert_eq!(output.standalone_query, "what about the rust book");
        assert_eq!(model.call_count(), 1);
        let prompt = model.last_prompt().unwrap();
        assert!(prompt.contains("tell me about rust"));
        assert!(prompt.contains("what about it"));
    }

    #[tokio::test]
    async fn malformed_reply_without_fallback_is_a_retrieval_error() {
        let rewriter = rewriter("I cannot help with that.");
        let error = rewriter
            .rewrite(&RewriteRequest::new("what is rust"))
            .await
            .expect_err("malformed reply must fail");
        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(
            error.message().contains("no JSON object found"),
            "unexpected message: {}",
            error.message()
        );
        assert!(error.message().contains("I cannot help with that."));
    }

    #[tokio::test]
    async fn invalid_json_without_fallback_is_a_retrieval_error() {
        let rewriter = rewriter("{\"standalone_query\": }");
        let error = rewriter
            .rewrite(&RewriteRequest::new("what is rust"))
            .await
            .expect_err("invalid JSON must fail");
        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(error.message().contains("invalid JSON"));
    }

    #[tokio::test]
    async fn malformed_reply_uses_the_fallback_rewriter() {
        let config = ModelRewriteConfig::default()
            .with_fallback(Arc::new(HeuristicQueryRewriter::default()));
        let rewriter = ModelQueryRewriter::new(
            Arc::new(StaticCompletionModel::single("no json here")),
            config,
        );
        let request =
            RewriteRequest::new("what about it").with_history(["tell me about the rust book"]);
        let output = rewriter.rewrite(&request).await.unwrap();
        assert_eq!(
            output.standalone_query,
            "tell me about the rust book what about it"
        );
        assert_eq!(output.from_history, "tell me about the rust book");
    }

    #[tokio::test]
    async fn model_failures_propagate_unchanged() {
        struct FailingModel;

        #[async_trait]
        impl CompletionModel for FailingModel {
            async fn complete(&self, _prompt: &str) -> Result<String> {
                Err(Error::timeout("model timed out"))
            }
        }

        let rewriter = ModelQueryRewriter::with_default_config(Arc::new(FailingModel));
        let error = rewriter
            .rewrite(&RewriteRequest::new("what is rust"))
            .await
            .expect_err("model failure must propagate");
        assert_eq!(error.kind(), ErrorKind::Timeout);
    }

    #[tokio::test]
    async fn blank_queries_never_reach_the_model() {
        let model = Arc::new(StaticCompletionModel::single(well_formed()));
        let rewriter = ModelQueryRewriter::with_default_config(model.clone());
        let error = rewriter
            .rewrite(&RewriteRequest::new("  "))
            .await
            .expect_err("blank query must fail");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn keywords_and_expansion_use_the_rewritten_query() {
        let reply = json!({
            "before": "what about it",
            "standalone_query": "the rust book chapter about async runtime",
            "references": {},
            "missing": [],
            "typo": [],
            "gibberish": [],
            "from_history": ""
        })
        .to_string();
        let rewriter = rewriter(&reply);
        assert_eq!(
            rewriter.keywords("what about it").await.unwrap(),
            vec!["chapter", "runtime", "async", "rust", "book"]
        );
        assert_eq!(
            rewriter.expand("what about it").await.unwrap()[0],
            "the rust book chapter about async runtime"
        );
    }
}
