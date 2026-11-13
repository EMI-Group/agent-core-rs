//! Query rewriting: turn a user's latest turn into a standalone retrieval query.
//!
//! Port of the Python `openjiuwen/core/retrieval/query_rewriter/` package. The
//! Python rewriter is LLM-only: it fills an `intention_completion` prompt with
//! the recent conversation, asks the model for a JSON object, and repairs the
//! reply against a fixed field schema. Rust keeps that wire shape in
//! [`RewriteOutput`] and ships two implementations:
//!
//! - [`HeuristicQueryRewriter`] — deterministic, model-free rewriting, keyword
//!   extraction, synonym expansion, and history resolution; the offline default.
//! - [`ModelQueryRewriter`] — the LLM-backed port over the
//!   [`crate::model::CompletionModel`] seam, with an optional fallback rewriter.
//!
//! `QueryRewriter` is the shared trait: every rewriter can rewrite a request,
//! extract keywords, and expand a query into retrieval variants.

mod heuristic;
mod model;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

pub use heuristic::{
    normalize_query, tokenize, HeuristicQueryRewriter, HeuristicRewriteConfig,
    DEFAULT_HISTORY_WINDOW, DEFAULT_MAX_KEYWORDS, DEFAULT_MAX_QUERIES,
};
pub use model::{ModelQueryRewriter, ModelRewriteConfig, DEFAULT_REWRITE_PROMPT};

/// A spelling correction reported by a rewriter.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypoFix {
    /// The term as the user wrote it.
    #[serde(default)]
    pub original: String,
    /// The corrected term.
    #[serde(default)]
    pub corrected: String,
    /// Why the term was corrected.
    #[serde(default)]
    pub reason: String,
}

impl TypoFix {
    /// Builds a correction record.
    #[must_use]
    pub fn new(
        original: impl Into<String>,
        corrected: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            original: original.into(),
            corrected: corrected.into(),
            reason: reason.into(),
        }
    }
}

/// The rewrite result, matching Python's field-for-field JSON shape.
///
/// Python builds this dict from the model reply after schema repair; Rust keeps
/// the same keys so a rewrite result can be logged, stored, or handed back to a
/// Python caller unchanged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RewriteOutput {
    /// The query exactly as the user sent it.
    pub before: String,
    /// A one-line statement of what the user wants.
    #[serde(default)]
    pub intention: String,
    /// The self-contained query to send to retrieval.
    pub standalone_query: String,
    /// Terms the query refers to, mapped to what they resolve to.
    #[serde(default)]
    pub references: Map<String, Value>,
    /// Information the query omits but retrieval would need.
    #[serde(default)]
    pub missing: Vec<String>,
    /// Spelling corrections applied while rewriting.
    #[serde(default)]
    pub typo: Vec<TypoFix>,
    /// Tokens judged meaningless (Python echoes them unchanged).
    #[serde(default)]
    pub gibberish: Vec<String>,
    /// The history turn the standalone query was completed from, or empty.
    #[serde(default)]
    pub from_history: String,
}

impl RewriteOutput {
    /// Builds an unchanged rewrite of `query`: `before`, `intention`, and
    /// `standalone_query` all mirror the input and every other field is empty.
    ///
    /// This is the shape returned when no rewriting is needed, and the fallback
    /// [`RewriteOutput`] always carries a usable `standalone_query`.
    #[must_use]
    pub fn from_query(query: impl Into<String>) -> Self {
        let query = query.into();
        Self {
            before: query.clone(),
            intention: query.clone(),
            standalone_query: query,
            ..Self::default()
        }
    }

    /// Whether the output carries a non-blank standalone query.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        !self.standalone_query.trim().is_empty()
    }

    /// Sets the reference map (builder style).
    #[must_use]
    pub fn with_references(mut self, references: Map<String, Value>) -> Self {
        self.references = references;
        self
    }
}

/// One rewrite call: the current query plus the conversation turns before it.
///
/// Python reads the history from a `ModelContext`; Rust takes the turns
/// directly so a rewriter is a pure function of its request. Turns are ordered
/// oldest-first and carry no role, matching the caller-owned history model of
/// the Python class.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewriteRequest {
    /// The user's latest turn.
    pub query: String,
    /// Previous turns, oldest first.
    #[serde(default)]
    pub history: Vec<String>,
}

impl RewriteRequest {
    /// Builds a request with no history.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            history: Vec::new(),
        }
    }

    /// Attaches conversation history, oldest first.
    #[must_use]
    pub fn with_history<I, S>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.history = history.into_iter().map(Into::into).collect();
        self
    }
}

/// Rewrites queries for retrieval.
///
/// `rewrite` is the primary operation; `keywords` and `expand` expose the two
/// derived views retrieval code usually needs. `rewrite_batch` defaults to a
/// sequential loop, which is what the model-backed rewriter wants (one model
/// call at a time) and what keeps request order observable.
#[async_trait]
pub trait QueryRewriter: Send + Sync {
    /// Rewrites `request` into a standalone retrieval query.
    async fn rewrite(&self, request: &RewriteRequest) -> Result<RewriteOutput>;

    /// Extracts the ranking keywords of `query`, most important first.
    async fn keywords(&self, query: &str) -> Result<Vec<String>>;

    /// Expands `query` into retrieval variants, original query first.
    async fn expand(&self, query: &str) -> Result<Vec<String>>;

    /// Rewrites every request in `requests`, preserving order.
    async fn rewrite_batch(&self, requests: &[RewriteRequest]) -> Result<Vec<RewriteOutput>> {
        let mut outputs = Vec::with_capacity(requests.len());
        for request in requests {
            outputs.push(self.rewrite(request).await?);
        }
        Ok(outputs)
    }
}

/// Rejects a blank query with [`Error::Validation`], as Python's rewriter does.
pub(crate) fn validate_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(Error::validation("query must be a non-empty string"));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn output_round_trips_through_the_python_json_shape() {
        let mut references = Map::new();
        references.insert("it".to_owned(), json!(["the rust book"]));
        let output = RewriteOutput {
            before: "what about it".to_owned(),
            intention: "learn about the rust book".to_owned(),
            standalone_query: "what about the rust book".to_owned(),
            references,
            missing: vec!["chapter".to_owned()],
            typo: vec![TypoFix::new("rustt", "rust", "spelling")],
            gibberish: vec!["asdf".to_owned()],
            from_history: "tell me about the rust book".to_owned(),
        };

        let value = serde_json::to_value(&output).unwrap();
        for key in [
            "before",
            "intention",
            "standalone_query",
            "references",
            "missing",
            "typo",
            "gibberish",
            "from_history",
        ] {
            assert!(value.get(key).is_some(), "missing JSON key {key}");
        }
        assert_eq!(value["typo"][0]["original"], json!("rustt"));
        assert_eq!(value["typo"][0]["corrected"], json!("rust"));
        assert_eq!(value["typo"][0]["reason"], json!("spelling"));

        let restored: RewriteOutput = serde_json::from_value(value).unwrap();
        assert_eq!(restored, output);
    }

    #[test]
    fn output_tolerates_missing_optional_fields_and_unknown_fields() {
        let value = json!({ "before": "q", "standalone_query": "q", "extra": 1 });
        let output: RewriteOutput = serde_json::from_value(value).unwrap();
        assert_eq!(output.before, "q");
        assert_eq!(output.standalone_query, "q");
        assert_eq!(output.intention, "");
        assert!(output.references.is_empty());
        assert!(output.is_usable());
    }

    #[test]
    fn from_query_mirrors_the_input_everywhere() {
        let output = RewriteOutput::from_query("  what is rust  ");
        assert_eq!(output.before, "  what is rust  ");
        assert_eq!(output.standalone_query, "  what is rust  ");
        assert_eq!(output.intention, "  what is rust  ");
        assert!(output.is_usable());
        assert!(output.missing.is_empty());
        assert!(output.references.is_empty());
        assert!(!RewriteOutput::default().is_usable());
    }

    #[test]
    fn request_builders_keep_history_order() {
        let request = RewriteRequest::new("it").with_history(["first", "second"]);
        assert_eq!(request.query, "it");
        assert_eq!(request.history, vec!["first", "second"]);
        assert_eq!(RewriteRequest::default().history, Vec::<String>::new());
    }

    #[tokio::test]
    async fn default_batch_rewrite_is_sequential_and_ordered() {
        let rewriter = HeuristicQueryRewriter::default();
        let requests = vec![
            RewriteRequest::new("what is rust"),
            RewriteRequest::new("python async"),
        ];
        let outputs = rewriter.rewrite_batch(&requests).await.unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].before, "what is rust");
        assert_eq!(outputs[1].before, "python async");
    }

    #[tokio::test]
    async fn blank_queries_are_validation_errors() {
        let rewriter = HeuristicQueryRewriter::default();
        let error = rewriter
            .rewrite(&RewriteRequest::new("   "))
            .await
            .expect_err("blank query must fail");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert_eq!(error.message(), "query must be a non-empty string");
    }
}
