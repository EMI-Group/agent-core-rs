//! Keyword/metadata and semantic capability retrieval.
//!
//! Python's symphony `retrieval` package answers one question: given a user request,
//! which capabilities should the planner consider? The Rust port offers two offline,
//! deterministic answers and one shared request/response pair:
//!
//! - [`crate::search::LexicalIndex`] — BM25 over
//!   [`crate::model::CapabilityDescriptor::retrieval_text`], with metadata filters
//!   ([`crate::search::SearchRequest`]).
//! - [`crate::search::SemanticIndex`] — embedding similarity over the same text, on
//!   top of `openjiuwen-retrieval`.
//!
//! [`crate::search::search_descriptors`] is the single orchestration entry point: it applies the
//! request's filters, ranks the surviving descriptors, applies `min_score`, and
//! truncates to `top_k`. The registry calls it with its catalog snapshot, so the
//! filtering and ranking rules live in exactly one place.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use openjiuwen_foundation::store::filter::matches;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::model::CapabilityDescriptor;

/// Deterministic BM25 keyword search over capability cards.
pub mod lexical;
/// Embedding-backed semantic capability search.
pub mod semantic;

pub use lexical::{tokenize, LexicalHit, LexicalIndex};
pub use semantic::SemanticIndex;

/// The number of hits [`SearchRequest`] returns when `top_k` is not set explicitly.
pub const DEFAULT_TOP_K: usize = 10;

/// A capability search request: free-text query, catalog filters, and paging.
///
/// Every field defaults, so `SearchRequest::default()` is a filter-only request that
/// returns the first [`DEFAULT_TOP_K`] capabilities by qualified id. `top_k == 0` means
/// unlimited.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SearchRequest {
    /// The free-text query; `None` or a blank string means "no keyword ranking".
    pub text: Option<String>,
    /// Restricts hits to these capability types (an empty list means every type).
    pub capability_types: Vec<String>,
    /// Requires every one of these tags (case-insensitive).
    pub tags: Vec<String>,
    /// Excludes capabilities carrying any of these tags (case-insensitive).
    pub exclude_tags: Vec<String>,
    /// Restricts hits to available/unavailable capabilities; `None` means both.
    pub available: Option<bool>,
    /// A Mongo-style JSON predicate matched against the descriptor JSON.
    pub metadata_filter: Option<serde_json::Value>,
    /// The maximum number of hits; `0` means unlimited and the default is
    /// [`DEFAULT_TOP_K`].
    pub top_k: usize,
    /// The inclusive minimum score a hit must reach.
    pub min_score: Option<f32>,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            text: None,
            capability_types: Vec::new(),
            tags: Vec::new(),
            exclude_tags: Vec::new(),
            available: None,
            metadata_filter: None,
            top_k: DEFAULT_TOP_K,
            min_score: None,
        }
    }
}

impl SearchRequest {
    /// Builds a keyword request for `text` with every other field at its default.
    #[must_use]
    pub fn for_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Self::default()
        }
    }
}

/// One search hit: the matched capability, its score, and the terms that matched.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SearchHit {
    /// The matched capability.
    pub descriptor: CapabilityDescriptor,
    /// The relevance score; higher is more relevant.
    pub score: f32,
    /// The distinct query terms the capability matched, sorted; empty for filter-only
    /// and semantic hits.
    pub matched_terms: Vec<String>,
}

/// Applies `request` to `descriptors` and returns the ranked hits.
///
/// Filters are applied first and combine with AND: `capability_types` (exact match),
/// `tags` (all present, case-insensitive), `exclude_tags` (none present),
/// `available`, and `metadata_filter` (via
/// [`openjiuwen_foundation::store::filter::matches`] over the descriptor JSON).
///
/// When `text` is present and not blank, the survivors are ranked with BM25 by
/// [`LexicalIndex`]. Otherwise every survivor is returned with a score of `0.0` and no
/// matched terms, sorted by qualified id. `min_score` is applied inclusively after
/// scoring, and the result is truncated to `top_k` unless `top_k == 0`.
///
/// # Errors
///
/// Returns [`Error::External`] when a descriptor cannot be serialized for the
/// `metadata_filter` predicate.
pub fn search_descriptors(
    descriptors: &[CapabilityDescriptor],
    request: &SearchRequest,
) -> Result<Vec<SearchHit>> {
    let mut filtered = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        if matches_filters(descriptor, request)? {
            filtered.push(descriptor.clone());
        }
    }

    let query = request
        .text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty());
    let Some(query) = query else {
        let mut hits: Vec<SearchHit> = filtered
            .into_iter()
            .map(|descriptor| SearchHit {
                descriptor,
                score: 0.0,
                matched_terms: Vec::new(),
            })
            .collect();
        sort_hits(&mut hits);
        return Ok(finalize(hits, request));
    };

    let mut by_id: BTreeMap<String, CapabilityDescriptor> = BTreeMap::new();
    for descriptor in filtered {
        by_id.insert(descriptor.qualified_id(), descriptor);
    }
    let index = LexicalIndex::build(by_id.values().cloned());
    let mut hits = Vec::with_capacity(by_id.len());
    for lexical in index.search(query, 0) {
        let Some(descriptor) = by_id.get(&lexical.qualified_id) else {
            continue;
        };
        hits.push(SearchHit {
            descriptor: descriptor.clone(),
            score: lexical.score,
            matched_terms: lexical.matched_terms,
        });
    }
    sort_hits(&mut hits);
    Ok(finalize(hits, request))
}

/// Evaluates the catalog filters of `request` against one capability.
///
/// # Errors
///
/// Returns [`Error::External`] when the descriptor cannot be serialized for the
/// `metadata_filter` predicate.
fn matches_filters(descriptor: &CapabilityDescriptor, request: &SearchRequest) -> Result<bool> {
    if !request.capability_types.is_empty()
        && !request
            .capability_types
            .iter()
            .any(|capability_type| capability_type == &descriptor.capability_type)
    {
        return Ok(false);
    }
    if !request
        .tags
        .iter()
        .all(|tag| has_tag(&descriptor.tags, tag))
    {
        return Ok(false);
    }
    if request
        .exclude_tags
        .iter()
        .any(|tag| has_tag(&descriptor.tags, tag))
    {
        return Ok(false);
    }
    if let Some(available) = request.available {
        if descriptor.available != available {
            return Ok(false);
        }
    }
    if let Some(filter) = request.metadata_filter.as_ref() {
        let document = serde_json::to_value(descriptor).map_err(Error::external)?;
        if !matches(filter, &document) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Reports whether `tags` contains `wanted`, ignoring case and surrounding blanks.
fn has_tag(tags: &[String], wanted: &str) -> bool {
    let wanted = wanted.trim().to_lowercase();
    tags.iter().any(|tag| tag.trim().to_lowercase() == wanted)
}

/// Orders hits by descending score, then ascending qualified id.
fn sort_hits(hits: &mut [SearchHit]) {
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left.descriptor
                    .qualified_id()
                    .cmp(&right.descriptor.qualified_id())
            })
    });
}

/// Applies `min_score` and `top_k` to ranked hits.
fn finalize(mut hits: Vec<SearchHit>, request: &SearchRequest) -> Vec<SearchHit> {
    if let Some(min_score) = request.min_score {
        hits.retain(|hit| hit.score >= min_score);
    }
    if request.top_k > 0 {
        hits.truncate(request.top_k);
    }
    hits
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
mod tests {
    use super::*;
    use crate::model::CapabilityKey;

    fn descriptor(
        capability_type: &str,
        id: &str,
        name: &str,
        description: &str,
    ) -> CapabilityDescriptor {
        CapabilityDescriptor::new(capability_type, id, name)
            .with_description(description)
            .with_source("builtin")
    }

    #[test]
    fn default_request_is_the_filter_only_request() {
        let request = SearchRequest::default();
        assert!(request.text.is_none());
        assert_eq!(request.top_k, DEFAULT_TOP_K);
        assert!(request.metadata_filter.is_none());
        assert!(request.min_score.is_none());

        let for_text = SearchRequest::for_text("sum numbers");
        assert_eq!(for_text.text.as_deref(), Some("sum numbers"));
        assert_eq!(
            for_text,
            SearchRequest {
                text: Some("sum numbers".to_string()),
                ..SearchRequest::default()
            }
        );
    }

    #[test]
    fn filter_only_requests_are_sorted_by_qualified_id() {
        let descriptors = [
            descriptor("tool", "b", "Beta", "second"),
            descriptor("workflow", "a", "Alpha", "first"),
        ];
        let hits = search_descriptors(&descriptors, &SearchRequest::default()).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].descriptor.qualified_id(), "tool:b");
        assert_eq!(hits[1].descriptor.qualified_id(), "workflow:a");
        assert!(hits.iter().all(|hit| hit.score == 0.0));
        assert!(hits.iter().all(|hit| hit.matched_terms.is_empty()));
    }

    #[test]
    fn filters_combine_with_and() {
        let descriptors = [
            descriptor("tool", "sum", "Sum", "adds two numbers")
                .with_tags(["math", "Numbers"])
                .with_available(true),
            descriptor("tool", "translate", "Translate", "translates text")
                .with_tags(["text"])
                .with_available(false),
            descriptor("skill", "weather", "Weather", "fetches the forecast")
                .with_tags(["net"])
                .with_available(true),
        ];

        let by_type = SearchRequest {
            capability_types: vec!["tool".to_string()],
            ..SearchRequest::default()
        };
        let hits = search_descriptors(&descriptors, &by_type).unwrap();
        assert_eq!(hits.len(), 2);

        let by_tag = SearchRequest {
            tags: vec!["MATH".to_string(), "numbers".to_string()],
            ..SearchRequest::default()
        };
        let hits = search_descriptors(&descriptors, &by_tag).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].descriptor.capability_id, "sum");

        let excluded = SearchRequest {
            exclude_tags: vec!["Net".to_string()],
            ..SearchRequest::default()
        };
        assert_eq!(
            search_descriptors(&descriptors, &excluded).unwrap().len(),
            2
        );

        let available = SearchRequest {
            available: Some(true),
            ..SearchRequest::default()
        };
        let hits = search_descriptors(&descriptors, &available).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.descriptor.available));

        let by_metadata = SearchRequest {
            metadata_filter: Some(serde_json::json!({"source": "builtin"})),
            ..SearchRequest::default()
        };
        assert_eq!(
            search_descriptors(&descriptors, &by_metadata)
                .unwrap()
                .len(),
            3
        );

        let miss = SearchRequest {
            metadata_filter: Some(serde_json::json!({"source": "remote"})),
            ..SearchRequest::default()
        };
        assert!(search_descriptors(&descriptors, &miss).unwrap().is_empty());
    }

    #[test]
    fn text_requests_rank_and_respect_min_score() {
        let descriptors = [
            descriptor("tool", "sum", "Sum", "adds two numbers"),
            descriptor("tool", "translate", "Translate", "translates text"),
        ];
        let request = SearchRequest {
            text: Some("adds numbers".to_string()),
            min_score: Some(0.1),
            ..SearchRequest::default()
        };
        let hits = search_descriptors(&descriptors, &request).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].descriptor.qualified_id(), "tool:sum");
        assert_eq!(hits[0].matched_terms, ["adds", "numbers"]);

        let too_high = SearchRequest {
            min_score: Some(f32::MAX),
            ..request
        };
        assert!(search_descriptors(&descriptors, &too_high)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn top_k_truncates_results() {
        let descriptors = [
            descriptor("tool", "a", "Shared Words", "shared"),
            descriptor("tool", "b", "Shared Words", "shared"),
            descriptor("tool", "c", "Shared Words", "shared"),
        ];
        let limited = SearchRequest {
            text: Some("shared".to_string()),
            top_k: 2,
            ..SearchRequest::default()
        };
        assert_eq!(search_descriptors(&descriptors, &limited).unwrap().len(), 2);

        let unlimited = SearchRequest {
            top_k: 0,
            ..limited
        };
        assert_eq!(
            search_descriptors(&descriptors, &unlimited).unwrap().len(),
            3
        );
    }

    #[test]
    fn request_round_trips_through_json_with_unknown_fields() {
        let request: SearchRequest = serde_json::from_value(serde_json::json!({
            "text": "sum",
            "top_k": 3,
            "future_field": true
        }))
        .unwrap();
        assert_eq!(request.text.as_deref(), Some("sum"));
        assert_eq!(request.top_k, 3);
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["min_score"], serde_json::Value::Null);
        assert_eq!(value["tags"], serde_json::json!([]));
    }

    #[test]
    fn blank_text_falls_back_to_filter_only_ordering() {
        let descriptors = [
            descriptor("tool", "b", "Beta", "second"),
            descriptor("tool", "a", "Alpha", "first"),
        ];
        let request = SearchRequest::for_text("   ");
        let hits = search_descriptors(&descriptors, &request).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].descriptor.qualified_id(), "tool:a");
        assert!(hits.iter().all(|hit| hit.score == 0.0));
    }

    #[test]
    fn keys_stay_available_for_registry_mapping() {
        let descriptor = descriptor("tool", "sum", "Sum", "adds");
        assert_eq!(
            CapabilityKey::new("tool", "sum").qualified(),
            descriptor.qualified_id()
        );
    }
}
