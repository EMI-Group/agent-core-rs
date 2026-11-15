//! Model-guided tree grouping with no silent lexical fallback.
//!
//! Python's `openjiuwen.symphony.retrieval.build.tree.grouping` groups skills in two phases:
//! `GROUP_DISCOVERY_PROMPT` invents the branch definitions, `GROUP_MERGE_PROMPT` merges the
//! passes, and `SKILL_ASSIGNMENT_PROMPT` assigns every skill. This module ports that idea as
//! one round trip: [`render_request`](crate::retrieval::build::grouping::llm::render_request)
//! describes the inventory and its bounds,
//! [`parse_reply`](crate::retrieval::build::grouping::llm::parse_reply) validates the answer,
//! and [`group_paths`](crate::retrieval::build::grouping::llm::group_paths) joins them over
//! the crate's [`CompletionModel`] seam.
//!
//! # The prompt contract
//!
//! `SYSTEM_PROMPT` asks for strict JSON only:
//! `{"groups":[{"path":["segment","segment"],"worker_ids":["…"]}]}`, with every listed worker
//! id assigned exactly once and a path of one to `max_depth - 1` lowercase slug segments
//! ordered from the most general to the most specific. The user payload is byte-stable JSON
//! (`{"max_depth","max_children","items":[{"worker_id","name","description"}]}`) built from
//! items sorted by `worker_id`, with the name and description truncated to
//! `NAME_LIMIT`/`DESCRIPTION_LIMIT` characters by `truncate_text`; no clock, no map order and
//! no locale decide a byte. A request with more than `MAX_REQUEST_ITEMS` items is an
//! [`Error::llm`](crate::error::Error) rather than a silently truncated inventory, because
//! batching is not implemented.
//!
//! # Reply validation
//!
//! [`parse_reply`](crate::retrieval::build::grouping::llm::parse_reply) is strict about
//! identity and tolerant about shape: `groups` must be an array, but a group may omit `path`
//! (meaning the root) or `worker_ids`. Each segment is slugged with [`slug_term`], blank
//! segments are dropped, repeated segments are dropped, and the path is truncated to
//! `max_depth - 1` segments, so an all-blank path puts its items at the root. The first
//! assignment of a `worker_id` wins; an id the inventory does not contain is dropped with a
//! `tracing::warn!`; every item the reply left out is dropped with a `tracing::warn!`. The
//! survivors keep reply order, so the result is deterministic.
//!
//! # Errors and fallbacks
//!
//! There is **no silent lexical fallback**: a failed call, a reply without a usable JSON
//! object, and a reply that assigns no known item are three distinct
//! [`Error::llm`](crate::error::Error) messages, each prefixed with `STAGE`. Deciding whether
//! that failure should degrade to a deterministic grouping is the caller's job (its
//! [the fallback mode](crate::llm::FallbackMode)), not this module's. An empty inventory is
//! answered with no assignments and no model call.

use std::collections::BTreeSet;

use serde::Serialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::llm::{complete_json, truncate_text, CompletionModel};
use crate::retrieval::build::tree_text::slug_term;

use super::{GroupAssignment, GroupingConfig, Limits, SourceItem, GENERAL_SEGMENT};

/// The stage name every error and warning message starts with.
pub const STAGE: &str = "llm_grouping";
/// The system instruction sent with every grouping request.
pub const SYSTEM_PROMPT: &str = "Group the capability items into a tree. Return JSON only: \
{\"groups\":[{\"path\":[\"segment\",\"segment\"],\"worker_ids\":[\"…\"]}]}. \
Assign every listed worker id exactly once, use each worker id at most once, and give each \
group a path of one to max_depth minus one lowercase slug segments ordered from the most \
general to the most specific. An empty path means the item belongs at the root.";
/// How many characters of an item's name are sent.
pub const NAME_LIMIT: usize = 120;
/// How many characters of an item's description are sent.
pub const DESCRIPTION_LIMIT: usize = 240;
/// The largest inventory one request may describe.
pub const MAX_REQUEST_ITEMS: usize = 1000;

/// One inventory entry of the request payload.
#[derive(Debug, Serialize)]
struct RequestItem {
    /// The stable item id the reply must echo.
    worker_id: String,
    /// The truncated display name.
    name: String,
    /// The truncated routing description.
    description: String,
}

/// The request payload, in its fixed field order.
#[derive(Debug, Serialize)]
struct GroupingRequest {
    /// The depth bound the reply's paths are validated against.
    max_depth: usize,
    /// The fan-out bound the reply is expected to respect.
    max_children: usize,
    /// The inventory, sorted by `worker_id`.
    items: Vec<RequestItem>,
}

/// Renders the request payload for `items`.
///
/// Items are sorted by `worker_id` (stable), so two calls over the same inventory return the
/// same bytes; the name and description are truncated by characters. The payload's field
/// order is `max_depth`, `max_children`, `items`.
///
/// # Errors
///
/// Returns [`Error::llm`] naming [`STAGE`] when the inventory holds more
/// than [`MAX_REQUEST_ITEMS`] items, or when the payload cannot be serialized.
pub fn render_request(items: &[SourceItem], config: &GroupingConfig) -> Result<String> {
    if items.len() > MAX_REQUEST_ITEMS {
        return Err(Error::llm(format!(
            "{STAGE}: {} items exceed the {MAX_REQUEST_ITEMS}-item request bound; \
             batching is not implemented — use a deterministic strategy or split the inventory",
            items.len()
        )));
    }
    let limits = Limits::from_config(config);
    let mut ordered: Vec<&SourceItem> = items.iter().collect();
    ordered.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
    let request = GroupingRequest {
        max_depth: limits.max_depth,
        max_children: limits.max_children,
        items: ordered
            .into_iter()
            .map(|item| RequestItem {
                worker_id: item.worker_id.clone(),
                name: truncate_text(&item.name, NAME_LIMIT).to_string(),
                description: truncate_text(&item.description, DESCRIPTION_LIMIT).to_string(),
            })
            .collect(),
    };
    serde_json::to_string(&request).map_err(|error| {
        Error::llm(format!(
            "{STAGE}: the request could not be serialized: {error}"
        ))
    })
}

/// Validates a model reply into one assignment per assigned item.
///
/// Each group's `path` is slugged, de-duplicated, stripped of blank segments and truncated to
/// `max_depth - 1` segments (an empty result is the root), and its `worker_ids` are claimed
/// first-wins. An id the inventory does not contain and an item the reply leaves out are both
/// dropped with a `tracing::warn!`; the survivors keep reply order.
#[must_use]
pub fn parse_reply(
    reply: &Value,
    items: &[SourceItem],
    config: &GroupingConfig,
) -> Vec<GroupAssignment> {
    let limit = Limits::from_config(config).max_depth.saturating_sub(1);
    let known: BTreeSet<&str> = items.iter().map(|item| item.worker_id.as_str()).collect();
    let mut claimed: BTreeSet<&str> = BTreeSet::new();
    let mut assignments = Vec::new();
    if let Some(groups) = reply.get("groups").and_then(Value::as_array) {
        for group in groups {
            let path = reply_path(group, limit);
            let Some(worker_ids) = group.get("worker_ids").and_then(Value::as_array) else {
                continue;
            };
            for worker_id in worker_ids.iter().filter_map(Value::as_str) {
                if !known.contains(worker_id) {
                    tracing::warn!(
                        worker_id = %worker_id,
                        "{}: the reply named an unknown worker id",
                        STAGE
                    );
                    continue;
                }
                if !claimed.insert(worker_id) {
                    continue;
                }
                assignments.push(GroupAssignment {
                    worker_id: worker_id.to_string(),
                    path: path.clone(),
                });
            }
        }
    }
    for item in items {
        if !claimed.contains(item.worker_id.as_str()) {
            tracing::warn!(
                worker_id = %item.worker_id,
                "{}: the reply left this item unassigned",
                STAGE
            );
        }
    }
    assignments
}

/// Reads and normalises one group's `path`.
fn reply_path(group: &Value, limit: usize) -> Vec<String> {
    let mut path: Vec<String> = Vec::new();
    let Some(segments) = group.get("path").and_then(Value::as_array) else {
        return path;
    };
    for segment in segments.iter().filter_map(Value::as_str) {
        if segment.trim().is_empty() {
            continue;
        }
        let slug = slug_term(segment, GENERAL_SEGMENT);
        if path.contains(&slug) {
            continue;
        }
        path.push(slug);
    }
    path.truncate(limit);
    path
}

/// Asks `model` to group `items`, returning one assignment per item the reply named.
///
/// An empty inventory answers no assignments without calling the model.
///
/// # Errors
///
/// Returns [`Error::llm`] naming [`STAGE`] when the request cannot be
/// rendered, when the call fails, when the reply holds no usable JSON object, or when the
/// reply assigns no known item. This module never substitutes a lexical grouping.
pub async fn group_paths(
    items: &[SourceItem],
    config: &GroupingConfig,
    model: &dyn CompletionModel,
) -> Result<Vec<GroupAssignment>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let request = render_request(items, config)?;
    match complete_json(Some(model), SYSTEM_PROMPT, &request).await {
        Ok(Some(reply)) => {
            let assignments = parse_reply(&reply, items, config);
            if assignments.is_empty() {
                return Err(Error::llm(format!(
                    "{STAGE}: the model reply assigned no known item"
                )));
            }
            Ok(assignments)
        }
        Ok(None) => Err(Error::llm(format!(
            "{STAGE}: the model reply contained no usable JSON object"
        ))),
        Err(error) => Err(Error::llm(format!(
            "{STAGE}: the model call failed: {error}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use openjiuwen_retrieval::StaticCompletionModel;

    use super::*;
    use crate::retrieval::types::RetrieverItem;

    /// Builds an item with a name and description, leaving every other field at its default.
    fn item(worker_id: &str, name: &str, description: &str) -> SourceItem {
        SourceItem {
            worker_id: worker_id.to_string(),
            item: RetrieverItem::new(worker_id, worker_id),
            name: name.to_string(),
            description: description.to_string(),
            ..SourceItem::default()
        }
    }

    /// The three-item inventory the reply tests share.
    fn inventory() -> Vec<SourceItem> {
        vec![
            item("a", "PDF Parser", "parses PDF documents"),
            item("b", "Spreadsheet Reader", "reads spreadsheets"),
            item("c", "Chart Builder", "builds charts"),
        ]
    }

    /// A path as the tests compare it.
    fn path(segments: &[&str]) -> Vec<String> {
        segments
            .iter()
            .map(|segment| (*segment).to_string())
            .collect()
    }

    #[test]
    fn render_request_is_byte_stable_and_sorted() {
        let config = GroupingConfig::default().with_max_depth(3);
        let request = render_request(&inventory(), &config).unwrap();
        assert_eq!(
            request,
            r#"{"max_depth":3,"max_children":8,"items":[{"worker_id":"a","name":"PDF Parser","description":"parses PDF documents"},{"worker_id":"b","name":"Spreadsheet Reader","description":"reads spreadsheets"},{"worker_id":"c","name":"Chart Builder","description":"builds charts"}]}"#
        );
        let reversed: Vec<SourceItem> = inventory().into_iter().rev().collect();
        assert_eq!(render_request(&reversed, &config).unwrap(), request);
    }

    #[test]
    fn render_request_truncates_the_name_and_description() {
        let long = item(
            "a",
            &"n".repeat(NAME_LIMIT + 20),
            &"d".repeat(DESCRIPTION_LIMIT + 20),
        );
        let request = render_request(&[long], &GroupingConfig::default()).unwrap();
        let decoded: Value = serde_json::from_str(&request).unwrap();
        let entry = &decoded["items"][0];
        assert_eq!(entry["name"].as_str().unwrap().chars().count(), NAME_LIMIT);
        assert_eq!(
            entry["description"].as_str().unwrap().chars().count(),
            DESCRIPTION_LIMIT
        );
    }

    #[test]
    fn render_request_bounds_the_item_count() {
        let items: Vec<SourceItem> = (0..=MAX_REQUEST_ITEMS)
            .map(|index| item(&format!("w{index}"), "Name", "Description"))
            .collect();
        let error = render_request(&items, &GroupingConfig::default()).unwrap_err();
        assert!(error.message().contains(STAGE), "{error}");
        assert!(
            error.message().contains("batching is not implemented"),
            "{error}"
        );
    }

    #[test]
    fn parse_reply_slugs_truncates_and_dedupes_paths() {
        let config = GroupingConfig::default().with_max_depth(3);
        let reply = json!({"groups": [
            {"path": ["Documents", "PDF", "Deep"], "worker_ids": ["a"]},
            {"path": ["", "Documents", "Documents", "  "], "worker_ids": ["b"]},
            {"path": ["", "   "], "worker_ids": ["c"]},
        ]});
        let assignments = parse_reply(&reply, &inventory(), &config);
        assert_eq!(
            assignments,
            vec![
                GroupAssignment {
                    worker_id: "a".to_string(),
                    path: path(&["documents", "pdf"]),
                },
                GroupAssignment {
                    worker_id: "b".to_string(),
                    path: path(&["documents"]),
                },
                GroupAssignment {
                    worker_id: "c".to_string(),
                    path: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn parse_reply_drops_unknown_ids_and_unassigned_items() {
        let reply = json!({"groups": [
            {"path": ["Docs"], "worker_ids": ["a", "ghost", 7]},
        ]});
        let assignments = parse_reply(&reply, &inventory(), &GroupingConfig::default());
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].worker_id, "a");
        assert_eq!(assignments[0].path, path(&["docs"]));
    }

    #[test]
    fn parse_reply_keeps_the_first_assignment() {
        let reply = json!({"groups": [
            {"path": ["First"], "worker_ids": ["a"]},
            {"path": ["Second"], "worker_ids": ["a"]},
        ]});
        let assignments = parse_reply(&reply, &inventory(), &GroupingConfig::default());
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].path, path(&["first"]));
    }

    #[test]
    fn parse_reply_tolerates_a_missing_or_malformed_groups_key() {
        let empty = parse_reply(&json!({}), &inventory(), &GroupingConfig::default());
        assert_eq!(empty, Vec::new());
        let wrong = parse_reply(
            &json!({"groups": "nope"}),
            &inventory(),
            &GroupingConfig::default(),
        );
        assert_eq!(wrong, Vec::new());
        let no_ids = parse_reply(
            &json!({"groups": [{"path": ["Docs"]}]}),
            &inventory(),
            &GroupingConfig::default(),
        );
        assert_eq!(no_ids, Vec::new());
    }

    #[tokio::test]
    async fn group_paths_parses_a_scripted_reply() {
        let model = StaticCompletionModel::single(
            json!({"groups": [
                {"path": ["Documents"], "worker_ids": ["a", "b"]},
                {"path": ["Visuals", "Charts"], "worker_ids": ["c"]},
            ]})
            .to_string(),
        );
        let assignments = group_paths(
            &inventory(),
            &GroupingConfig::default().with_max_depth(3),
            &model,
        )
        .await
        .unwrap();
        assert_eq!(
            assignments,
            vec![
                GroupAssignment {
                    worker_id: "a".to_string(),
                    path: path(&["documents"]),
                },
                GroupAssignment {
                    worker_id: "b".to_string(),
                    path: path(&["documents"]),
                },
                GroupAssignment {
                    worker_id: "c".to_string(),
                    path: path(&["visuals", "charts"]),
                },
            ]
        );
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn group_paths_does_not_call_the_model_for_no_items() {
        let model = StaticCompletionModel::single("{}");
        let assignments = group_paths(&[], &GroupingConfig::default(), &model)
            .await
            .unwrap();
        assert_eq!(assignments, Vec::new());
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn group_paths_reports_an_unusable_reply() {
        let model = StaticCompletionModel::single("no json here");
        let error = group_paths(&inventory(), &GroupingConfig::default(), &model)
            .await
            .unwrap_err();
        assert!(error.message().contains(STAGE), "{error}");
        assert!(error.message().contains("no usable JSON object"), "{error}");
    }

    #[tokio::test]
    async fn group_paths_reports_a_reply_with_no_known_item() {
        let model = StaticCompletionModel::single(json!({"groups": []}).to_string());
        let error = group_paths(&inventory(), &GroupingConfig::default(), &model)
            .await
            .unwrap_err();
        assert!(error.message().contains(STAGE), "{error}");
        assert!(
            error.message().contains("assigned no known item"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn group_paths_reports_a_failed_call() {
        let model = StaticCompletionModel::new(Vec::new());
        let error = group_paths(&inventory(), &GroupingConfig::default(), &model)
            .await
            .unwrap_err();
        assert!(error.message().contains(STAGE), "{error}");
        assert!(error.message().contains("model call failed"), "{error}");
    }
}
