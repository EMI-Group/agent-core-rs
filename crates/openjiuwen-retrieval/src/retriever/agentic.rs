//! Agentic retriever: iterative retrieve → read triples → rewrite, fused with RRF.
//!
//! Port of Python's `openjiuwen/core/retrieval/retriever/agentic_retriever.py`
//! `_retrieve_generic` loop. Python always calls an LLM for both steps (reading
//! triples out of the passages and judging whether they suffice); this Rust port
//! works fully offline by default:
//!
//! - triples come from the [`TripleExtractor`] seam, defaulting to the
//!   deterministic [`RuleTripleExtractor`];
//! - sufficiency is the fraction of the query's rewrite keywords that occur in
//!   the accumulated passage texts, compared against
//!   [`AgenticConfig::sufficiency_threshold`];
//! - the next query comes from the [`QueryRewriter`] seam.
//!
//! When a [`CompletionModel`] is supplied it replaces both model-driven steps
//! (Python's `_READ_PROMPT` and `_REWRITE_PROMPT`), and every model or parse
//! failure falls back to the deterministic path — the retrieve call never fails
//! because a model misbehaved.
//!
//! All iterations' hits are fused with [`rrf_fusion`]. Because fused scores are
//! reciprocal ranks rather than channel similarities, `score_threshold` is
//! applied to each iteration's channel scores **before** fusion; filtering the
//! fused list against a similarity threshold would drop everything.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::warn;

use super::fusion::{rrf_fusion, DEFAULT_RRF_K};
use super::{
    ensure_query, retrieval_to_search, search_result_to_retrieval, RetrieveMode, Retriever,
};
use crate::common::{Document, MetadataFilter, RetrievalResult, SearchResult, TextChunk};
use crate::graph::{RuleTripleExtractor, Triple, TripleExtractor, TripleMemory};
use crate::model::{extract_json_object, CompletionModel};
use crate::rewrite::{QueryRewriter, RewriteRequest};
use crate::vector_store::DOC_ID_FIELD;

/// Default number of iterations (Python's `max_iter` default).
pub const DEFAULT_MAX_ITER: usize = 2;

/// Default sufficiency threshold: this fraction of the query keywords must appear
/// in the retrieved passages before the loop stops early.
pub const DEFAULT_SUFFICIENCY_THRESHOLD: f32 = 0.6;

/// Default number of top passages read for triples per iteration.
pub const DEFAULT_PASSAGES_PER_ITERATION: usize = 5;

/// Triple-reading prompt, adapted from Python's `_READ_PROMPT`.
///
/// Placeholders are filled by literal replacement (`{docs}`, `{query}`,
/// `{facts}`) so the JSON braces in the instructions are left untouched.
pub const READ_PROMPT: &str = "\
Your task is to find facts that help answer an input question.
Present these facts as knowledge triples, structured as (\"subject\", \"predicate\", \"object\").
Now you are given some documents:
{docs}
Based on these documents and the preliminary facts provided below, find additional supporting facts \
that may help answer the question. If the information is insufficient, output only the relevant facts \
you can find.
Question: {query}
Facts: {facts}
Output the facts as a JSON array of triples, where each triple is an array of three strings.
Example output format: [[\"subject1\", \"predicate1\", \"object1\"], [\"subject2\", \"predicate2\", \"object2\"]]
";

/// Sufficiency/rewrite prompt, adapted from Python's `_REWRITE_PROMPT`.
///
/// Placeholders: `{query}`, `{triples}`, `{question_rewriting_history}`.
pub const REWRITE_PROMPT: &str = "\
Given a question and its associated retrieved knowledge triples, evaluate whether the triples by \
themselves are sufficient to answer the original question.
Respond with a JSON object in the following format:
{\"sufficient\": true/false, \"next_question\": \"string or null\"}
- If the triples are sufficient, set \"sufficient\" to true and \"next_question\" to null.
- If the triples are not sufficient, set \"sufficient\" to false and provide a next question that \
would retrieve the missing evidence. The next question must differ from every previous question.
Question History:
Original Question: {query}
{question_rewriting_history}
Knowledge triples:
{triples}
Response (JSON only, no additional text):
";

/// Tuning for [`AgenticRetriever`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AgenticConfig {
    /// Maximum number of retrieve → read → rewrite iterations.
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    /// Fraction of the query keywords that must appear before stopping early.
    #[serde(default = "default_sufficiency_threshold")]
    pub sufficiency_threshold: f32,
    /// How many top passages per iteration are read for triples.
    #[serde(default = "default_passages_per_iteration")]
    pub passages_per_iteration: usize,
}

fn default_max_iter() -> usize {
    DEFAULT_MAX_ITER
}

fn default_sufficiency_threshold() -> f32 {
    DEFAULT_SUFFICIENCY_THRESHOLD
}

fn default_passages_per_iteration() -> usize {
    DEFAULT_PASSAGES_PER_ITERATION
}

impl Default for AgenticConfig {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            sufficiency_threshold: DEFAULT_SUFFICIENCY_THRESHOLD,
            passages_per_iteration: DEFAULT_PASSAGES_PER_ITERATION,
        }
    }
}

impl AgenticConfig {
    /// Builds the default configuration (2 iterations, threshold `0.6`, 5 passages).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the iteration bound.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the sufficiency threshold.
    #[must_use]
    pub fn with_sufficiency_threshold(mut self, sufficiency_threshold: f32) -> Self {
        self.sufficiency_threshold = sufficiency_threshold;
        self
    }

    /// Sets how many passages are read per iteration.
    #[must_use]
    pub fn with_passages_per_iteration(mut self, passages_per_iteration: usize) -> Self {
        self.passages_per_iteration = passages_per_iteration;
        self
    }

    /// Validates the configuration.
    ///
    /// Python silently resets an invalid `max_iter` to 2; Rust reports it so the
    /// caller can see the value it actually asked for.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `max_iter` or `passages_per_iteration`
    /// is zero, or when `sufficiency_threshold` is not a finite number in `[0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if self.max_iter == 0 {
            return Err(Error::validation(
                "AgenticConfig max_iter must be greater than zero",
            ));
        }
        if self.passages_per_iteration == 0 {
            return Err(Error::validation(
                "AgenticConfig passages_per_iteration must be greater than zero",
            ));
        }
        if !self.sufficiency_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.sufficiency_threshold)
        {
            return Err(Error::validation(format!(
                "AgenticConfig sufficiency_threshold must be between 0 and 1, got {}",
                self.sufficiency_threshold
            )));
        }
        Ok(())
    }
}

/// Why an [`AgenticRetriever`] loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticStopReason {
    /// The accumulated passages covered enough query keywords (or the model said so).
    Sufficient,
    /// The loop ran its full iteration budget.
    MaxIterations,
    /// The rewriter returned a blank query.
    EmptyRewrite,
    /// The rewritten query repeated an earlier one, so no new evidence would appear.
    NoNewResults,
    /// A model call or reply could not be used, so the deterministic path took over.
    ModelFailure,
}

impl AgenticStopReason {
    /// The wire name of the reason (`"sufficient"`, `"max_iterations", ...`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sufficient => "sufficient",
            Self::MaxIterations => "max_iterations",
            Self::EmptyRewrite => "empty_rewrite",
            Self::NoNewResults => "no_new_results",
            Self::ModelFailure => "model_failure",
        }
    }
}

/// Deterministic, test-observable record of one agentic retrieval run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgenticTrace {
    /// Number of retrieve → read → rewrite iterations that ran.
    pub iterations: usize,
    /// Queries in order: the original query first, then each rewrite.
    pub queries: Vec<String>,
    /// Every triple read across the iterations, deduplicated in first-seen order.
    pub triples: Vec<Triple>,
    /// Why the loop stopped.
    pub stopped_reason: AgenticStopReason,
}

/// A [`Retriever`] that iteratively rewrites and fuses, on top of any retriever.
pub struct AgenticRetriever {
    retriever: Arc<dyn Retriever>,
    rewriter: Arc<dyn QueryRewriter>,
    model: Option<Arc<dyn CompletionModel>>,
    extractor: Arc<dyn TripleExtractor>,
    config: AgenticConfig,
}

impl std::fmt::Debug for AgenticRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgenticRetriever")
            .field("model", &self.model.as_ref().map(|model| model.name()))
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// One rewrite decision.
enum RewriteStep {
    /// Continue with a refined query.
    Query(String),
    /// The evidence is sufficient; stop.
    Sufficient,
    /// The rewrite produced nothing usable; stop.
    Empty,
}

impl AgenticRetriever {
    /// Builds an agentic retriever over `retriever`, rewriting with `rewriter`.
    ///
    /// Defaults: [`AgenticConfig::default`], the [`RuleTripleExtractor`], and no
    /// model (fully deterministic and offline).
    pub fn new(retriever: Arc<dyn Retriever>, rewriter: Arc<dyn QueryRewriter>) -> Self {
        Self {
            retriever,
            rewriter,
            model: None,
            extractor: Arc::new(RuleTripleExtractor::new()),
            config: AgenticConfig::default(),
        }
    }

    /// Replaces the configuration after validating it.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`AgenticConfig::validate`].
    pub fn with_config(mut self, config: AgenticConfig) -> Result<Self> {
        config.validate()?;
        self.config = config;
        Ok(self)
    }

    /// Supplies a completion model, enabling the model-driven read and rewrite steps.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Replaces the triple extractor used when no model is available.
    #[must_use]
    pub fn with_extractor(mut self, extractor: Arc<dyn TripleExtractor>) -> Self {
        self.extractor = extractor;
        self
    }

    /// The wrapped retriever.
    pub fn retriever(&self) -> &Arc<dyn Retriever> {
        &self.retriever
    }

    /// The query rewriter.
    pub fn rewriter(&self) -> &Arc<dyn QueryRewriter> {
        &self.rewriter
    }

    /// The completion model, when one is configured.
    pub fn model(&self) -> Option<&Arc<dyn CompletionModel>> {
        self.model.as_ref()
    }

    /// The triple extractor used without a model.
    pub fn extractor(&self) -> &Arc<dyn TripleExtractor> {
        &self.extractor
    }

    /// The configuration in force.
    pub fn config(&self) -> &AgenticConfig {
        &self.config
    }

    /// Runs the loop and returns the fused results together with their trace.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query or `top_k == 0` (Python
    /// raises the same `top_k is invalid` error), and propagates retriever,
    /// rewriter, and triple-extractor failures. Model and model-reply failures
    /// never fail the call: they degrade to the deterministic path and are
    /// reported through [`AgenticStopReason::ModelFailure`].
    pub async fn retrieve_with_trace(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<(Vec<RetrievalResult>, AgenticTrace)> {
        ensure_query(query)?;
        if top_k == 0 {
            return Err(Error::validation("top_k must be a positive integer"));
        }
        let mut queries: Vec<String> = vec![query.to_owned()];
        let mut memory = TripleMemory::new();
        let mut result_lists: Vec<Vec<SearchResult>> = Vec::new();
        let mut covered = String::new();
        let mut iterations = 0;
        let mut reason = AgenticStopReason::MaxIterations;
        let mut model_failed = false;
        let keywords = self.rewriter.keywords(query).await?;

        for turn in 1..=self.config.max_iter {
            iterations = turn;
            let current = queries.last().cloned().unwrap_or_else(|| query.to_owned());
            let mut hits = self
                .retriever
                .retrieve_search_results(&current, top_k, mode, filters)
                .await?;
            if let Some(threshold) = score_threshold {
                hits.retain(|hit| hit.score >= threshold);
            }
            hits.truncate(top_k);
            for hit in &hits {
                covered.push_str(&hit.text.to_lowercase());
                covered.push('\n');
            }
            let passages: Vec<SearchResult> = hits
                .iter()
                .take(self.config.passages_per_iteration)
                .cloned()
                .collect();
            let triples = self
                .read_triples(&current, &passages, &memory, &mut model_failed)
                .await?;
            memory.batch_extend(&triples);
            result_lists.push(hits);

            if sufficiency_ratio(&keywords, &covered) >= self.config.sufficiency_threshold {
                reason = AgenticStopReason::Sufficient;
                break;
            }
            if turn >= self.config.max_iter {
                break;
            }
            match self
                .next_query(query, &queries, &memory, &mut model_failed)
                .await?
            {
                RewriteStep::Sufficient => {
                    reason = AgenticStopReason::Sufficient;
                    break;
                }
                RewriteStep::Empty => {
                    reason = AgenticStopReason::EmptyRewrite;
                    break;
                }
                RewriteStep::Query(next) => {
                    if queries.iter().any(|seen| same_query(seen, &next)) {
                        reason = AgenticStopReason::NoNewResults;
                        break;
                    }
                    queries.push(next);
                }
            }
        }

        let mut fused = rrf_fusion(result_lists, DEFAULT_RRF_K);
        fused.truncate(top_k);
        let trace = AgenticTrace {
            iterations,
            queries,
            triples: memory.triples().to_vec(),
            stopped_reason: if model_failed {
                AgenticStopReason::ModelFailure
            } else {
                reason
            },
        };
        Ok((
            fused.into_iter().map(search_result_to_retrieval).collect(),
            trace,
        ))
    }

    /// Reads triples from `passages`, preferring the model and degrading to the
    /// configured extractor.
    async fn read_triples(
        &self,
        query: &str,
        passages: &[SearchResult],
        memory: &TripleMemory,
        model_failed: &mut bool,
    ) -> Result<Vec<Triple>> {
        if let Some(model) = self.model.as_ref() {
            match self.read_with_model(model, query, passages, memory).await {
                Ok(triples) => return Ok(triples),
                Err(error) => {
                    warn!(model = model.name(), error = %error, "model triple read failed, using the rule extractor");
                    *model_failed = true;
                }
            }
        }
        let chunks: Vec<TextChunk> = passages.iter().map(chunk_of).collect();
        self.extractor.extract(&chunks).await
    }

    /// Model-driven triple read (Python's `_read`).
    async fn read_with_model(
        &self,
        model: &Arc<dyn CompletionModel>,
        query: &str,
        passages: &[SearchResult],
        memory: &TripleMemory,
    ) -> Result<Vec<Triple>> {
        let docs = passages
            .iter()
            .map(|passage| passage.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let facts = {
            let triples = memory.triples_str();
            if triples.is_empty() {
                "None".to_owned()
            } else {
                triples
            }
        };
        let prompt = READ_PROMPT
            .replace("{docs}", &docs)
            .replace("{query}", query)
            .replace("{facts}", &facts);
        let reply = model.complete(&prompt).await?;
        parse_triples(&reply)
            .ok_or_else(|| Error::llm("model reply did not contain a JSON array of triples"))
    }

    /// Model-driven rewrite decision (Python's `_rewrite`), falling back to the
    /// configured rewriter on any model or parse failure.
    async fn next_query(
        &self,
        original: &str,
        queries: &[String],
        memory: &TripleMemory,
        model_failed: &mut bool,
    ) -> Result<RewriteStep> {
        if let Some(model) = self.model.as_ref() {
            match Self::rewrite_with_model(model, original, queries, memory).await {
                Ok(step) => return Ok(step),
                Err(error) => {
                    warn!(model = model.name(), error = %error, "model rewrite failed, using the rewriter");
                    *model_failed = true;
                }
            }
        }
        let request = rewrite_request(original, queries, memory);
        let output = self.rewriter.rewrite(&request).await?;
        let next = output.standalone_query.trim().to_owned();
        if next.is_empty() {
            return Ok(RewriteStep::Empty);
        }
        Ok(RewriteStep::Query(next))
    }

    /// Model-driven rewrite step, returning a parse error the caller degrades on.
    async fn rewrite_with_model(
        model: &Arc<dyn CompletionModel>,
        original: &str,
        queries: &[String],
        memory: &TripleMemory,
    ) -> Result<RewriteStep> {
        let history = if queries.len() > 1 {
            queries
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, query)| format!("Rewritten Question {}: {query}", index + 1))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            "(No rewriting steps yet)".to_owned()
        };
        let triples = {
            let triples = memory.triples_str();
            if triples.is_empty() {
                "None".to_owned()
            } else {
                triples
            }
        };
        let prompt = REWRITE_PROMPT
            .replace("{query}", original)
            .replace("{question_rewriting_history}", &history)
            .replace("{triples}", &triples);
        let reply = model.complete(&prompt).await?;
        let object = extract_json_object(&reply)
            .ok_or_else(|| Error::llm("model reply did not contain a JSON object"))?;
        let value: Value = serde_json::from_str(object)
            .map_err(|error| Error::llm(format!("model reply was not valid JSON: {error}")))?;
        let sufficient = value
            .get("sufficient")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let next = value
            .get("next_question")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|next| !next.is_empty());
        match (sufficient, next) {
            (true, _) | (false, None) => Ok(RewriteStep::Sufficient),
            (false, Some(next)) => Ok(RewriteStep::Query(next.to_owned())),
        }
    }
}

#[async_trait]
impl Retriever for AgenticRetriever {
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<RetrievalResult>> {
        let (results, _) = self
            .retrieve_with_trace(query, top_k, score_threshold, mode, filters)
            .await?;
        Ok(results)
    }

    async fn retrieve_search_results(
        &self,
        query: &str,
        top_k: usize,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        let (results, _) = self
            .retrieve_with_trace(query, top_k, None, mode, filters)
            .await?;
        Ok(results.into_iter().map(retrieval_to_search).collect())
    }
}

/// Builds a chunk for triple extraction from a retrieved passage.
fn chunk_of(passage: &SearchResult) -> TextChunk {
    let doc_id = passage
        .metadata
        .get(DOC_ID_FIELD)
        .and_then(Value::as_str)
        .unwrap_or(&passage.id)
        .to_owned();
    let document = Document::with_id(doc_id, passage.text.clone());
    TextChunk::from_document(&document, passage.text.clone(), Some(passage.id.clone()))
}

/// Builds the refinement request: earlier queries followed by the triple lines.
fn rewrite_request(original: &str, queries: &[String], memory: &TripleMemory) -> RewriteRequest {
    let mut history: Vec<String> = queries.to_vec();
    let triples = memory.triples_str();
    if !triples.is_empty() {
        history.extend(triples.lines().map(ToOwned::to_owned));
    }
    RewriteRequest::new(original).with_history(history)
}

/// Case-insensitive, whitespace-insensitive query equality.
fn same_query(left: &str, right: &str) -> bool {
    left.trim().to_lowercase() == right.trim().to_lowercase()
}

/// Fraction of `keywords` present in the accumulated passage text.
fn sufficiency_ratio(keywords: &[String], covered: &str) -> f32 {
    if keywords.is_empty() {
        return 0.0;
    }
    let present = keywords
        .iter()
        .filter(|keyword| {
            let keyword = keyword.trim().to_lowercase();
            keyword.is_empty() || covered.contains(&keyword)
        })
        .count();
    present as f32 / keywords.len() as f32
}

/// Parses a model reply into triples, accepting a JSON array of triples or an
/// object carrying a `triples`/`facts` array.
fn parse_triples(reply: &str) -> Option<Vec<Triple>> {
    if let Some(object) = extract_json_object(reply) {
        if let Ok(value) = serde_json::from_str::<Value>(object) {
            for key in ["triples", "facts"] {
                if let Some(triples) = value.get(key).and_then(Value::as_array) {
                    let triples = triples_from_array(triples);
                    if !triples.is_empty() {
                        return Some(triples);
                    }
                }
            }
        }
    }
    let array = slice_json_array(reply)?;
    let value: Value = serde_json::from_str(array).ok()?;
    let triples = triples_from_array(value.as_array()?);
    if triples.is_empty() {
        None
    } else {
        Some(triples)
    }
}

/// Slices from the first `[` to the last `]`, requiring a non-empty span.
fn slice_json_array(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end <= start {
        return None;
    }
    Some(&text[start..=end])
}

/// Converts a JSON array of triples, skipping entries that are not triples.
fn triples_from_array(values: &[Value]) -> Vec<Triple> {
    values.iter().filter_map(triple_from_value).collect()
}

/// Converts one lenient triple: `["s", "p", "o"]` or the object form.
fn triple_from_value(value: &Value) -> Option<Triple> {
    let triple = match value {
        Value::Array(items) if items.len() == 3 => {
            let mut parts = Vec::with_capacity(3);
            for item in items {
                parts.push(item.as_str()?.trim().to_owned());
            }
            Triple::new(parts[0].clone(), parts[1].clone(), parts[2].clone())
        }
        Value::Object(map) => {
            let component = |key: &str| map.get(key).and_then(Value::as_str).map(str::trim);
            Triple::new(
                component("subject")?,
                component("predicate")?,
                component("object")?,
            )
        }
        _ => return None,
    };
    if triple.has_blank_component() {
        return None;
    }
    Some(triple)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    use crate::graph::Triple;

    #[test]
    fn config_validation_matches_python_intent() {
        assert!(AgenticConfig::default().validate().is_ok());
        assert_eq!(AgenticConfig::default().max_iter, DEFAULT_MAX_ITER);
        assert_eq!(
            AgenticConfig::new()
                .with_max_iter(0)
                .validate()
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
        assert!(AgenticConfig::new()
            .with_passages_per_iteration(0)
            .validate()
            .is_err());
        for threshold in [-0.1, 1.1, f32::NAN] {
            let error = AgenticConfig::new()
                .with_sufficiency_threshold(threshold)
                .validate()
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("between 0 and 1"));
        }
        assert!(AgenticConfig::new()
            .with_sufficiency_threshold(0.0)
            .validate()
            .is_ok());
    }

    #[test]
    fn stop_reason_serializes_as_snake_case() {
        assert_eq!(AgenticStopReason::Sufficient.as_str(), "sufficient");
        assert_eq!(
            serde_json::to_value(AgenticStopReason::NoNewResults).unwrap(),
            serde_json::json!("no_new_results")
        );
        assert_eq!(
            serde_json::from_value::<AgenticStopReason>(serde_json::json!("model_failure"))
                .unwrap(),
            AgenticStopReason::ModelFailure
        );
    }

    #[test]
    fn triple_parsing_is_lenient_about_wrapping_prose() {
        let triples = parse_triples(
            "Here: [[\"Acme\", \"is_a\", \"company\"], [\"Acme\", \"has\", \"Rust\"]] done",
        )
        .unwrap();
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0], Triple::new("Acme", "is_a", "company"));

        let wrapped =
            parse_triples("```json\n{\"triples\": [[\"A\", \"b\", \"C\"], [\"oops\"]]}\n```")
                .unwrap();
        assert_eq!(wrapped, vec![Triple::new("A", "b", "C")]);

        let objects = parse_triples(
            "[{\"subject\": \"A\", \"predicate\": \"b\", \"object\": \"C\"}, {\"subject\": \"\"}]",
        )
        .unwrap();
        assert_eq!(objects, vec![Triple::new("A", "b", "C")]);
        assert!(parse_triples("no triples here").is_none());
        assert!(parse_triples("[[]]").is_none());
    }

    #[test]
    fn sufficiency_counts_present_keywords_only() {
        let keywords = vec!["ownership".to_owned(), "rust".to_owned()];
        assert_eq!(sufficiency_ratio(&keywords, "rust explains ownership"), 1.0);
        assert_eq!(sufficiency_ratio(&keywords, "rust only"), 0.5);
        assert_eq!(sufficiency_ratio(&keywords, ""), 0.0);
        assert_eq!(sufficiency_ratio(&[], "anything"), 0.0);
    }

    #[test]
    fn rewrite_requests_carry_queries_then_triple_lines() {
        let mut memory = TripleMemory::new();
        memory.extend(Triple::new("Acme", "is_a", "company"));
        let request = rewrite_request("who is acme", &["who is acme".to_owned()], &memory);
        assert_eq!(request.query, "who is acme");
        assert_eq!(
            request.history,
            vec!["who is acme".to_owned(), "(acme is_a company)".to_owned()]
        );
        assert!(same_query("Who is Acme ", "who is acme"));
        assert!(!same_query("who is acme", "who owns acme"));
    }
}
