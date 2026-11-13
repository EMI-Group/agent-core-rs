//! Progressive tool discovery: a BM25 index over tool cards and the `tool_search` tool.
//!
//! Ports `openjiuwen/harness/tools/tool_discovery/{tool_search,bm25}.py`. Deferred tools are
//! held by [`crate::tools::ToolRegistry`] and only advertised to the model after a search hit.
//!
//! # Indexing and scoring
//!
//! Each card becomes one document whose text is `"{name} {description}"`. Tokenisation is
//! deterministic: the text is lowercased and split on every non-alphanumeric character, so
//! `_` is a separator and `read_file` indexes as `read` + `file`; a query naming either the
//! verb or the noun therefore matches. A trailing plural `s` is dropped for tokens longer
//! than three characters (stem-light, no full stemming), and a run of CJK characters also
//! contributes its adjacent character bigrams, mirroring Python's `tokenize`.
//!
//! Scores are Okapi BM25 with `k1 = 1.2` and `b = 0.75`. Python's shared index uses
//! `k1 = 1.5`; the difference only reweights repeated query terms and never changes which
//! tools are eligible.
//!
//! The name is not given a separate term weight, because Python does not weight it either.
//! What Python does instead is rank `_contains_exact_tool_name` hits first, regardless of
//! BM25 score; here that match adds `EXACT_NAME_BOOST` to the score so the documented
//! "score descending, then name ascending" order still holds and stays deterministic.
//! Documents with a zero score and no exact-name match are dropped (Python behaviour).
//!
//! An empty, whitespace-only, or punctuation-only query returns no hits — Python's
//! `search` returns `[]` for a falsy query instead of scoring the whole corpus.
//!
//! # Model-facing contract
//!
//! `tool_search` takes `{ "query": <string>, "limit": <integer, optional> }`, where `limit`
//! defaults to `DEFAULT_LIMIT` and is clamped to `1..=MAX_LIMIT` (`0` therefore means
//! "one"). The payload is the usual six-key envelope whose `data` is
//! `{ "query": <string>, "results": [{ "name", "score", "description" }], "count": <n> }`.
//!
//! Differences from Python, all deliberate:
//!
//! - Python's `data` carries the raw result dicts with `parameters` schemas and has no
//!   `score`; here every hit is `{name, score, description}` and the parameter schema is not
//!   indexed result data, so a model that needs the schema must call the tool it picked.
//! - Python's `render_for_llm` prints each hit's parameter schema; that rendering is not
//!   ported for the same reason, so the default payload rendering applies.
//! - Python's `ToolSearchTool` takes an injected async search callback; here the tool owns an
//!   immutable [`ToolIndex`] and the registry decides when to rebuild it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::descriptions::description;
use super::support::{parse_args, schema_for, tool_card, ToolPayload};

/// BM25 term-frequency saturation parameter.
const K1: f64 = 1.2;

/// BM25 document-length normalisation parameter.
const B: f64 = 0.75;

/// Score added to a document whose tool name the query mentions verbatim.
///
/// Comfortably above the BM25 scores of a realistic tool registry, so an exact name keeps
/// Python's "exact-name match first" ordering even when another document shares more words.
const EXACT_NAME_BOOST: f64 = 10.0;

/// Result limit used when the model omits `limit` (Python `DEFAULT_TOOL_SEARCH_LIMIT`).
const DEFAULT_LIMIT: usize = 5;

/// Largest accepted `limit` (Python `MAX_TOOL_SEARCH_LIMIT`).
const MAX_LIMIT: usize = 20;

/// One tool returned by [`ToolIndex::search`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSearchHit {
    /// The model-facing tool name.
    pub name: String,
    /// BM25 score, plus `EXACT_NAME_BOOST` when the query names the tool verbatim.
    pub score: f64,
    /// The tool's model-facing description.
    pub description: String,
}

/// One indexed tool: its identity plus the tokenised search text.
#[derive(Debug, Clone, Default)]
struct IndexedTool {
    /// The model-facing tool name.
    name: String,
    /// The model-facing description, returned verbatim on a hit.
    description: String,
    /// The tokens of `"{name} {description}"`.
    tokens: Vec<String>,
}

/// An immutable BM25 index over the model-facing text of a set of tool cards.
///
/// Built once from the cards a caller wants to be discoverable and shared as an `Arc`; a
/// changed registry means a new index rather than a mutation, which keeps searches
/// lock-free.
#[derive(Debug, Clone, Default)]
pub struct ToolIndex {
    /// The indexed tools, in card order.
    docs: Vec<IndexedTool>,
    /// Number of documents containing each token.
    document_frequencies: BTreeMap<String, usize>,
    /// Mean document length in tokens; `0.0` for an empty index.
    average_length: f64,
}

impl ToolIndex {
    /// Builds an index over the `name` and `description` of `cards`.
    ///
    /// Card order only matters for equal-score, equal-name documents, which cannot occur in
    /// a registry whose names are unique.
    #[must_use]
    pub fn from_cards(cards: &[ToolCard]) -> Self {
        let docs: Vec<IndexedTool> = cards
            .iter()
            .map(|card| IndexedTool {
                name: card.name.clone(),
                description: card.description.clone(),
                tokens: tokenize(&format!("{} {}", card.name, card.description)),
            })
            .collect();

        let mut document_frequencies: BTreeMap<String, usize> = BTreeMap::new();
        for doc in &docs {
            let unique: BTreeSet<&str> = doc.tokens.iter().map(String::as_str).collect();
            for token in unique {
                *document_frequencies.entry(token.to_owned()).or_insert(0) += 1;
            }
        }

        let average_length = if docs.is_empty() {
            0.0
        } else {
            total_length(&docs) / documents_as_f64(docs.len())
        };

        Self {
            docs,
            document_frequencies,
            average_length,
        }
    }

    /// Number of indexed tools.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.docs.len()
    }

    /// Searches the index and returns at most `limit` hits.
    ///
    /// Hits are ordered by descending score, then by ascending name. A `limit` of `0` and an
    /// empty or whitespace-only query return no hits. Zero-score documents are dropped
    /// unless the query names the tool verbatim.
    #[must_use]
    pub fn search(&self, query: &str, limit: usize) -> Vec<ToolSearchHit> {
        if limit == 0 || self.docs.is_empty() || query.trim().is_empty() {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        let mut hits: Vec<ToolSearchHit> = self
            .docs
            .iter()
            .filter_map(|doc| {
                let exact_name_match = contains_exact_name(query, &doc.name);
                let score = self.bm25_score(&query_tokens, doc);
                if score <= 0.0 && !exact_name_match {
                    return None;
                }
                Some(ToolSearchHit {
                    name: doc.name.clone(),
                    score: if exact_name_match {
                        score + EXACT_NAME_BOOST
                    } else {
                        score
                    },
                    description: doc.description.clone(),
                })
            })
            .collect();

        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.name.cmp(&right.name))
        });
        hits.truncate(limit);
        hits
    }

    /// Scores one document against `query_tokens` (Okapi BM25).
    ///
    /// Duplicate query tokens count once, as in Python. The `usize` to `f64` conversions are
    /// lossless for any realistic corpus (a tool registry is far below 2^52 documents).
    #[allow(clippy::cast_precision_loss)]
    fn bm25_score(&self, query_tokens: &[String], doc: &IndexedTool) -> f64 {
        if self.average_length <= 0.0 {
            return 0.0;
        }
        let document_count = self.docs.len() as f64;
        let document_length = doc.tokens.len() as f64;
        let mut score = 0.0;
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for token in query_tokens {
            if !seen.insert(token.as_str()) {
                continue;
            }
            let term_frequency = doc
                .tokens
                .iter()
                .filter(|candidate| *candidate == token)
                .count() as f64;
            if term_frequency <= 0.0 {
                continue;
            }
            let document_frequency =
                self.document_frequencies.get(token).copied().unwrap_or(0) as f64;
            let inverse_document_frequency = (1.0
                + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();
            let denominator =
                term_frequency + K1 * (1.0 - B + B * document_length / self.average_length);
            score += inverse_document_frequency * term_frequency * (K1 + 1.0) / denominator;
        }
        score
    }
}

/// Sums the token counts of `docs`.
fn total_length(docs: &[IndexedTool]) -> f64 {
    documents_as_f64(docs.iter().map(|doc| doc.tokens.len()).sum())
}

/// Converts a token/document count to `f64`, which is lossless for realistic sizes.
#[allow(clippy::cast_precision_loss)]
fn documents_as_f64(count: usize) -> f64 {
    count as f64
}

/// Tokenises `text` for both indexing and querying.
///
/// The text is lowercased and split on every non-alphanumeric character; `_`, `-`, `.` and
/// punctuation are separators. Tokens are then expanded with a stem-light plural form and,
/// for CJK runs, with adjacent character bigrams. The expansion order is deterministic, so
/// two equal inputs always score identically.
fn tokenize(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    let mut tokens: Vec<String> = Vec::new();
    for word in lowered.split(|character: char| !character.is_alphanumeric()) {
        push_token(&mut tokens, word);
    }
    tokens
}

/// Appends one split word and its expansions to `tokens`.
fn push_token(tokens: &mut Vec<String>, word: &str) {
    if word.is_empty() {
        return;
    }
    if word.chars().any(is_cjk) {
        tokens.push(word.to_owned());
        let characters: Vec<char> = word.chars().collect();
        for pair in characters.windows(2) {
            tokens.push(pair.iter().collect());
        }
        return;
    }
    tokens.push(word.to_owned());
    if let Some(singular) = singularize(word) {
        tokens.push(singular);
    }
}

/// Returns the singular form of a simple plural, or `None`.
///
/// Only a trailing `s` is stripped, and only from words longer than three characters that do
/// not already end in `ss`, `us` or `is`, so `files`/`tools` match `file`/`tool` while
/// `class`, `status` and `analysis` do not collapse into unrelated tokens.
fn singularize(word: &str) -> Option<String> {
    if word.len() <= 3 || word.ends_with("ss") || word.ends_with("us") || word.ends_with("is") {
        return None;
    }
    word.strip_suffix('s').map(str::to_owned)
}

/// Whether `character` belongs to a CJK Unified Ideograph block.
fn is_cjk(character: char) -> bool {
    matches!(character as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF)
}

/// Whether `query` mentions `name` as a complete tool identifier.
///
/// Case-insensitive. An occurrence counts only when neither neighbour can be part of a tool
/// identifier (`[A-Za-z0-9_.-]`), so a Chinese sentence such as `请调用read_file读取` matches
/// `read_file` while `read_file_extra` does not.
fn contains_exact_name(query: &str, name: &str) -> bool {
    let query: Vec<char> = query.to_lowercase().chars().collect();
    let name: Vec<char> = name.trim().to_lowercase().chars().collect();
    if query.is_empty() || name.is_empty() || name.len() > query.len() {
        return false;
    }
    (0..=query.len() - name.len()).any(|start| {
        query[start..start + name.len()] == name[..]
            && !start
                .checked_sub(1)
                .and_then(|index| query.get(index))
                .is_some_and(|before| is_identifier_char(*before))
            && !query
                .get(start + name.len())
                .is_some_and(|after| is_identifier_char(*after))
    })
}

/// Whether `character` may be part of an ASCII tool identifier.
fn is_identifier_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-')
}

/// Arguments of the `tool_search` tool.
///
/// Python parity: `ToolSearchInput`. Only `query` is required; `limit` tolerates a numeric
/// string or a float because models routinely emit `"limit": "3"` or `"limit": 3.0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolSearchArgs {
    /// Query text describing the user's intent and the required capability.
    pub query: String,
    /// Maximum number of tools to return (default 5, clamped to 1..=20).
    #[serde(default, deserialize_with = "deserialize_opt_usize")]
    #[schemars(with = "usize")]
    pub limit: Option<usize>,
}

/// Deserializes an optional `usize` from a number or a numeric string.
///
/// Anything that is not a non-negative integer (a boolean, `null`, a non-numeric string)
/// deserializes as `None`, so a malformed `limit` falls back to the default instead of
/// failing the whole argument object. A float is truncated like Python's `int(value)`.
fn deserialize_opt_usize<'de, D>(deserializer: D) -> std::result::Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(json_usize))
}

/// Reads a non-negative integer out of a JSON number or a numeric string.
fn json_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .map(|number| usize::try_from(number).unwrap_or(usize::MAX))
            .or_else(|| number.as_f64().and_then(float_usize)),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Truncates a finite, non-negative float to `usize`, saturating at the integer bounds.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn float_usize(value: f64) -> Option<usize> {
    if value.is_finite() && value >= 0.0 {
        Some(value.floor() as usize)
    } else {
        None
    }
}

/// The `tool_search` tool: searches a [`ToolIndex`] over deferred tools.
///
/// The tool never returns [`Err`] for an expected failure: a malformed argument object becomes
/// a `success: false` payload. The only [`Err`] it produces is
/// [`openjiuwen_common::error::Error::Cancelled`] from the call's cancellation token.
pub struct ToolSearchTool {
    /// The immutable index this tool searches.
    index: Arc<ToolIndex>,
    /// The model-facing card, built once from the description language.
    card: ToolCard,
}

impl ToolSearchTool {
    /// Builds a `tool_search` tool over `index` with the description language `language`.
    #[must_use]
    pub fn new(index: Arc<ToolIndex>, language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            index,
            card: Self::card_for(&language),
        }
    }

    /// Builds the model-facing card for the description language `language`.
    ///
    /// The description text comes from [`super::descriptions`]; until that table gains a
    /// `tool_search` row the card carries an empty description.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        let mut input_params = schema_for::<ToolSearchArgs>();
        install_limit_metadata(&mut input_params);
        tool_card(
            "tool_search",
            description("tool_search", language),
            input_params,
            true,
        )
    }

    /// The index this tool searches.
    #[must_use]
    pub fn index(&self) -> &ToolIndex {
        &self.index
    }

    /// Resolves the effective result limit (Python `_normalize_tool_search_limit`).
    fn resolve_limit(limit: Option<usize>) -> usize {
        limit.map_or(DEFAULT_LIMIT, |limit| limit.clamp(1, MAX_LIMIT))
    }
}

/// Pins the Python `limit` metadata (`default`, `minimum`, `maximum`) into the card schema.
fn install_limit_metadata(input_params: &mut Value) {
    let Some(limit) = input_params
        .get_mut("properties")
        .and_then(|properties| properties.get_mut("limit"))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    limit.insert("default".to_owned(), json!(DEFAULT_LIMIT));
    limit.insert("minimum".to_owned(), json!(1));
    limit.insert("maximum".to_owned(), json!(MAX_LIMIT));
}

#[async_trait]
impl Tool for ToolSearchTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        let args = match parse_args::<ToolSearchArgs>(&inputs) {
            Ok(args) => args,
            Err(payload) => return Ok(payload.to_json()),
        };

        let limit = Self::resolve_limit(args.limit);
        let hits = self.index.search(&args.query, limit);
        let results: Vec<Value> = hits
            .iter()
            .map(|hit| {
                json!({
                    "name": hit.name.as_str(),
                    "score": hit.score,
                    "description": hit.description.as_str(),
                })
            })
            .collect();
        let count = results.len();

        Ok(ToolPayload::ok(json!({
            "query": args.query,
            "results": results,
            "count": count,
        }))
        .to_json())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::cmp::Ordering;

    use super::*;

    /// A card with `name` and `description` and an empty input schema.
    fn card(name: &str, description: &str) -> ToolCard {
        ToolCard::new(name, description, json!({}))
    }

    /// An index over the four filesystem-ish tools used by most tests.
    fn sample_index() -> ToolIndex {
        ToolIndex::from_cards(&[
            card(
                "read_file",
                "Read a text file and return line-numbered content",
            ),
            card(
                "write_file",
                "Write content to a file, creating or updating it",
            ),
            card("grep", "Search file contents with a regular expression"),
            card("bash", "Execute a bash command and return its output"),
        ])
    }

    /// Runs `tool_search` with `inputs` and returns the raw payload JSON.
    async fn invoke(index: Arc<ToolIndex>, inputs: Value) -> Value {
        let tool = ToolSearchTool::new(index, "en");
        tool.invoke(inputs, &ToolContext::new())
            .await
            .expect("tool_search never fails for expected input")
    }

    #[test]
    fn known_query_ranks_the_expected_tool_first() {
        let hits = sample_index().search("read the contents of a file", 5);
        assert_eq!(hits.first().map(|hit| hit.name.as_str()), Some("read_file"));
        assert!(hits.iter().all(|hit| hit.score > 0.0), "hits: {hits:?}");
    }

    #[test]
    fn underscore_names_index_as_separate_tokens() {
        let index = ToolIndex::from_cards(&[
            card("read_file", ""),
            card("write_file", ""),
            card("unrelated", ""),
        ]);
        assert_eq!(
            index
                .search("write", 5)
                .first()
                .map(|hit| hit.name.as_str()),
            Some("write_file")
        );
        assert_eq!(
            index
                .search("file", 5)
                .iter()
                .map(|hit| hit.name.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["read_file", "write_file"])
        );
    }

    #[test]
    fn stem_light_plurals_match_their_singular() {
        let hits = sample_index().search("files", 5);
        assert!(hits.iter().any(|hit| hit.name == "write_file"), "{hits:?}");
        assert_eq!(hits.first().map(|hit| hit.name.as_str()), Some("read_file"));
    }

    #[test]
    fn equal_scores_break_ties_by_name() {
        let index = ToolIndex::from_cards(&[
            card("zeta", "widget"),
            card("alpha", "widget"),
            card("beta", "widget"),
        ]);
        let names: Vec<String> = index
            .search("widget", 3)
            .into_iter()
            .map(|hit| hit.name)
            .collect();
        assert_eq!(names, vec!["alpha", "beta", "zeta"]);
    }

    #[test]
    fn exact_name_match_outranks_a_better_text_match() {
        let index = ToolIndex::from_cards(&[
            card("read_file", "read a file"),
            card("docs", "read_file helper for remote documents"),
        ]);
        let hits = index.search("read_file", 5);
        assert_eq!(hits.first().map(|hit| hit.name.as_str()), Some("read_file"));
        assert!(hits[0].score > EXACT_NAME_BOOST);

        // Inside a longer identifier the name does not count as an exact match.
        let partial = index.search("read_file_extra", 5);
        assert!(partial.iter().all(|hit| hit.score <= EXACT_NAME_BOOST));
    }

    #[test]
    fn empty_and_blank_queries_return_no_hits() {
        let index = sample_index();
        assert_eq!(index.search("", 5), Vec::new());
        assert_eq!(index.search("   ", 5), Vec::new());
        assert_eq!(index.search("!!!", 5), Vec::new());
    }

    #[test]
    fn zero_limit_returns_no_hits() {
        assert_eq!(sample_index().search("file", 0), Vec::new());
    }

    #[test]
    fn empty_index_has_no_documents_and_no_hits() {
        let index = ToolIndex::from_cards(&[]);
        assert_eq!(index.document_count(), 0);
        assert_eq!(index.search("file", 5), Vec::new());
    }

    #[test]
    fn document_count_counts_cards() {
        assert_eq!(sample_index().document_count(), 4);
    }

    #[test]
    fn search_is_case_insensitive() {
        let hits = sample_index().search("READ FILE", 5);
        assert_eq!(hits.first().map(|hit| hit.name.as_str()), Some("read_file"));
    }

    #[tokio::test]
    async fn no_match_query_is_a_success_with_no_results() {
        let payload = invoke(Arc::new(sample_index()), json!({"query": "kubernetes"})).await;
        assert_eq!(payload["success"], true);
        assert_eq!(payload["data"]["results"], json!([]));
        assert_eq!(payload["data"]["count"], 0);
        assert_eq!(payload["data"]["query"], "kubernetes");
    }

    #[tokio::test]
    async fn payload_shape_matches_the_documented_contract() {
        let payload = invoke(Arc::new(sample_index()), json!({"query": "read file"})).await;

        for key in [
            "success",
            "data",
            "error",
            "extracted_content",
            "include_extracted_content_only_once",
            "long_term_memory",
        ] {
            assert!(payload.get(key).is_some(), "missing envelope key {key}");
        }
        let data = payload["data"].as_object().unwrap();
        let mut keys: Vec<&str> = data.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["count", "query", "results"]);

        let first = &payload["data"]["results"][0];
        let mut hit_keys: Vec<&str> = first
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        hit_keys.sort_unstable();
        assert_eq!(hit_keys, vec!["description", "name", "score"]);
        assert_eq!(first["name"], "read_file");
        assert!(first["score"].as_f64().unwrap() > 0.0);
        assert!(first["description"]
            .as_str()
            .unwrap()
            .starts_with("Read a text file"));
        assert_eq!(
            payload["data"]["count"],
            payload["data"]["results"].as_array().unwrap().len()
        );
    }

    #[tokio::test]
    async fn limit_defaults_to_five_and_is_clamped() {
        let index = Arc::new(ToolIndex::from_cards(
            &(0..25)
                .map(|index| card(&format!("tool_{index}"), "sample capability"))
                .collect::<Vec<_>>(),
        ));

        let default = invoke(Arc::clone(&index), json!({"query": "sample"})).await;
        assert_eq!(default["data"]["count"], DEFAULT_LIMIT);

        let zero = invoke(Arc::clone(&index), json!({"query": "sample", "limit": 0})).await;
        assert_eq!(zero["data"]["count"], 1);

        let huge = invoke(Arc::clone(&index), json!({"query": "sample", "limit": 500})).await;
        assert_eq!(huge["data"]["count"], MAX_LIMIT);

        let numeric_string =
            invoke(Arc::clone(&index), json!({"query": "sample", "limit": "2"})).await;
        assert_eq!(numeric_string["data"]["count"], 2);

        let float = invoke(index, json!({"query": "sample", "limit": 3.9})).await;
        assert_eq!(float["data"]["count"], 3);
    }

    #[tokio::test]
    async fn missing_query_is_a_failure_payload() {
        let payload = invoke(Arc::new(sample_index()), json!({})).await;
        assert_eq!(payload["success"], false);
        let message = payload["error"].as_str().unwrap();
        assert!(
            message.starts_with("invalid tool arguments: "),
            "unexpected message: {message}"
        );
    }

    #[tokio::test]
    async fn non_object_arguments_are_a_protocol_failure() {
        let payload = invoke(Arc::new(sample_index()), json!([])).await;
        assert_eq!(payload["success"], false);
        assert_eq!(payload["error"], "tool arguments must be an object");
    }

    #[test]
    fn card_declares_the_python_name_and_schema() {
        let card = ToolSearchTool::card_for("en");
        assert_eq!(card.name, "tool_search");
        assert!(card.parallel_safe);
        assert_eq!(card.description, description("tool_search", "en"));

        let params = &card.input_params;
        assert_eq!(params["type"], "object");
        assert_eq!(params["required"], json!(["query"]));
        assert_eq!(params["properties"]["query"]["type"], "string");
        assert_eq!(params["properties"]["limit"]["type"], "integer");
        assert_eq!(params["properties"]["limit"]["default"], DEFAULT_LIMIT);
        assert_eq!(params["properties"]["limit"]["minimum"], 1);
        assert_eq!(params["properties"]["limit"]["maximum"], MAX_LIMIT);
    }

    #[test]
    fn card_language_selects_the_description_column() {
        let chinese = ToolSearchTool::card_for("cn");
        assert_eq!(chinese.name, "tool_search");
        assert_eq!(chinese.description, description("tool_search", "cn"));
        assert_eq!(ToolSearchTool::card_for("unknown").name, "tool_search");
    }

    #[test]
    fn tool_exposes_its_index() {
        let index = Arc::new(sample_index());
        let tool = ToolSearchTool::new(Arc::clone(&index), "en");
        assert_eq!(tool.index().document_count(), 4);
        assert_eq!(tool.card().name, "tool_search");
    }

    #[test]
    fn hits_sort_by_score_descending_then_name() {
        let hits = sample_index().search("file content", 10);
        for pair in hits.windows(2) {
            let order = pair[0].score.total_cmp(&pair[1].score);
            assert!(
                order != Ordering::Less,
                "hits are not sorted by descending score: {hits:?}"
            );
            if order == Ordering::Equal {
                assert!(pair[0].name <= pair[1].name);
            }
        }
    }
}
