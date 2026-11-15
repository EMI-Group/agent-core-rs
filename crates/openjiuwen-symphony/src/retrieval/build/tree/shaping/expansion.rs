//! Breadth-first expansion of a deterministic skeleton: the `expansion` tree-shaping stage.
//!
//! Python's `build/tree/expansion.py` grows an existing tree level by level with a model.
//! This stage ports that walk over a skeleton of
//! [`GroupAssignment`]s: the
//! deterministic grouping has already placed every item, and the model gets to split one
//! branch at a time.
//!
//! # Request and reply
//!
//! Branches are visited breadth first — the root first, then every branch prefix in sorted
//! order at the next depth — and each visit is one model call. The user payload is
//! `{"stage":"expansion","branch":["documents"],"depth":2,"children":["forms","pdf"],"items":[…]}`,
//! where `children` are the branch's current child segments, and each item carries
//! `worker_id`, `name`, `description`, `classification`, `tags` and its current `path`
//! below the branch. At most
//! [`ShapingConfig::max_items_per_prompt`](crate::retrieval::build::tree::shaping::ShapingConfig::max_items_per_prompt)
//! items are described (the first entries in `(worker_id, path)` order); the rest keep the
//! deterministic subtree the caller computed and are only reached by a later call.
//! The reply must be
//! `{"children":[{"segment":"documents","description":"…","worker_ids":["…"]}]}`.
//!
//! Every `worker_id` a reply names is moved from its current path below the branch to
//! `branch/segment`, keeping the rest of its path; a worker id that is not currently below
//! the branch, or that the reply names twice, is dropped and listed in
//! [`ShapingOutcome::dropped`](crate::retrieval::build::tree::shaping::ShapingOutcome::dropped).
//! Branch descriptions returned by the model are not persisted by this stage: the build
//! always derives them from the `cid` path with
//! [`branch_description`](crate::retrieval::build::tree::branch_description), so a shaped
//! tree never embeds prose that could drift between runs.
//!
//! # Bounds
//!
//! The walk makes at most
//! [`ShapingConfig::max_calls`](crate::retrieval::build::tree::shaping::ShapingConfig::max_calls)
//! calls: a branch left unvisited keeps its deterministic subtree, so the run still
//! succeeds. A prefix at the depth limit is never visited, and the final placements obey
//! the shared `max_depth`/`max_children` normalisation, so a reply can neither deepen the
//! tree past the bound nor overfill a branch.
//!
//! # The deterministic fallback
//!
//! A reply that carries no `children` array, or that moves nothing, keeps that branch's
//! deterministic subtree: the run warns once
//! ([`FallbackMode::Deterministic`](crate::llm::FallbackMode)) and continues with the next
//! branch. When every call fails, the run reports provenance
//! [`DeterministicFallback`](crate::retrieval::build::tree::shaping::ShapingProvenance::DeterministicFallback);
//! when at least one branch was expanded it reports
//! [`Model`](crate::retrieval::build::tree::shaping::ShapingProvenance::Model). With no model
//! configured or no call budget, the stage answers the caller's skeleton unchanged with
//! provenance
//! [`Skipped`](crate::retrieval::build::tree::shaping::ShapingProvenance::Skipped).
//! Under [`FallbackMode::Error`](crate::llm::FallbackMode) the first failed branch is
//! returned as an error instead.
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::llm::{string_field, string_list, CompletionModel};
use crate::retrieval::build::grouping::{GroupAssignment, GroupingConfig};
use crate::retrieval::build::scanner::SourceItem;
use crate::retrieval::build::tree_text::slug_term;

use super::{
    below, call_model, finalize, item_value, no_usable_reply, normalize_paths, on_model_failure,
    prompt_len, skipped, PathLimits, ShapingConfig, ShapingMode, ShapingOutcome, ShapingProvenance,
};

/// The system instruction sent with every expansion request.
const SYSTEM_PROMPT: &str = "Split one branch of a capability routing tree into child \
branches. Return JSON only: {\"children\":[{\"segment\":\"documents\",\"description\":\"…\",\"worker_ids\":[\"…\"]}]}. \
Use short lowercase slug segments, list each worker id once under the one child it belongs \
to, and return an empty list when the branch should not be split.";

/// Expands a deterministic skeleton branch by branch, breadth first.
///
/// # Errors
///
/// The first failed branch under
/// [`FallbackMode::Error`](crate::llm::FallbackMode) — the model invocation failure
/// unchanged, or `Error::Llm` when the reply carried nothing usable.
pub async fn expansion(
    items: &[SourceItem],
    skeleton: &[GroupAssignment],
    config: &GroupingConfig,
    shaping: &ShapingConfig,
    model: Option<&dyn CompletionModel>,
) -> Result<ShapingOutcome> {
    let limits = PathLimits::from_config(config);
    let Some(model) = model else {
        return Ok(skipped(skeleton, ShapingMode::Expansion));
    };
    if shaping.max_calls == 0 {
        return Ok(skipped(skeleton, ShapingMode::Expansion));
    }
    let mut assignments: Vec<GroupAssignment> = skeleton.to_vec();
    normalize_paths(&mut assignments, limits);
    let known: BTreeMap<&str, &SourceItem> = items
        .iter()
        .map(|item| (item.worker_id.as_str(), item))
        .collect();

    let mut queue: VecDeque<Vec<String>> = VecDeque::new();
    let mut visited: BTreeSet<Vec<String>> = BTreeSet::new();
    queue.push_back(Vec::new());
    visited.insert(Vec::new());
    let mut calls = 0_usize;
    let mut applied = 0_usize;
    let mut dropped: BTreeSet<String> = BTreeSet::new();
    let mut warned = false;

    while let Some(prefix) = queue.pop_front() {
        if calls >= shaping.max_calls {
            break;
        }
        if prefix.len() >= limits.max_segments() {
            continue;
        }
        let under: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, assignment)| below(&assignment.path, &prefix))
            .map(|(index, _)| index)
            .collect();
        if under.is_empty() {
            continue;
        }
        let request = branch_request(&prefix, &under, &assignments, &known, shaping);
        calls += 1;
        let mut merged = 0_usize;
        let mut failure: Option<Error> = None;
        match call_model(model, SYSTEM_PROMPT, &request).await {
            Ok(Some(value)) => match apply_children(&mut assignments, &prefix, &value) {
                Some((moved, unusable)) => {
                    merged = moved;
                    dropped.extend(unusable);
                    if moved == 0 {
                        failure = Some(no_usable_reply("expansion"));
                    }
                }
                None => failure = Some(no_usable_reply("expansion")),
            },
            Ok(None) => failure = Some(no_usable_reply("expansion")),
            Err(error) => failure = Some(error),
        }
        match failure {
            Some(error) => on_model_failure(shaping, &mut warned, "expansion", error)?,
            None => applied += merged,
        }

        let mut children: BTreeSet<Vec<String>> = BTreeSet::new();
        for assignment in &assignments {
            if below(&assignment.path, &prefix) {
                if let Some(child) = assignment.path.get(..=prefix.len()) {
                    children.insert(child.to_vec());
                }
            }
        }
        for child in children {
            if visited.insert(child.clone()) {
                queue.push_back(child);
            }
        }
    }

    let provenance = if calls == 0 {
        ShapingProvenance::Skipped
    } else if applied > 0 {
        ShapingProvenance::Model
    } else {
        ShapingProvenance::DeterministicFallback
    };
    Ok(finalize(
        assignments,
        limits,
        ShapingMode::Expansion,
        provenance,
        calls,
        dropped,
    ))
}

/// Renders the expansion request of one branch.
fn branch_request(
    prefix: &[String],
    under: &[usize],
    assignments: &[GroupAssignment],
    known: &BTreeMap<&str, &SourceItem>,
    shaping: &ShapingConfig,
) -> String {
    let mut ordered: Vec<&GroupAssignment> = under
        .iter()
        .filter_map(|index| assignments.get(*index))
        .collect();
    ordered.sort_by(|left, right| {
        (left.worker_id.as_str(), left.path.as_slice())
            .cmp(&(right.worker_id.as_str(), right.path.as_slice()))
    });
    let mut children: BTreeSet<String> = BTreeSet::new();
    for assignment in &ordered {
        if let Some(segment) = assignment.path.get(prefix.len()) {
            children.insert(segment.clone());
        }
    }
    let rendered: Vec<Value> = ordered
        .iter()
        .take(prompt_len(ordered.len(), shaping))
        .map(|assignment| {
            let mut value = known.get(assignment.worker_id.as_str()).map_or_else(
                || {
                    json!({
                        "worker_id": assignment.worker_id,
                        "name": "",
                        "description": "",
                        "classification": "",
                        "tags": [],
                    })
                },
                |item| item_value(item, shaping.description_limit),
            );
            let path = assignment
                .path
                .get(prefix.len()..)
                .unwrap_or_default()
                .join("/");
            if let Some(object) = value.as_object_mut() {
                object.insert("path".to_string(), Value::String(path));
            }
            value
        })
        .collect();
    json!({
        "stage": "expansion",
        "branch": prefix,
        "depth": prefix.len() + 1,
        "children": children.into_iter().collect::<Vec<String>>(),
        "items": rendered,
    })
    .to_string()
}

/// Merges one `{"children":[…]}` reply into the placements below `prefix`.
///
/// Returns how many worker ids moved and the ids the reply named that could not be placed,
/// or `None` when the reply carries no `children` array.
fn apply_children(
    assignments: &mut [GroupAssignment],
    prefix: &[String],
    value: &Value,
) -> Option<(usize, BTreeSet<String>)> {
    let children = value.get("children")?.as_array()?;
    let mut moved: BTreeSet<String> = BTreeSet::new();
    let mut unusable: BTreeSet<String> = BTreeSet::new();
    for child in children {
        let Some(segment) = string_field(child, "segment") else {
            continue;
        };
        let segment = slug_term(&segment, "");
        if segment.is_empty() {
            continue;
        }
        for worker_id in string_list(child.get("worker_ids")) {
            if moved.contains(&worker_id) || !place(assignments, prefix, &segment, &worker_id) {
                unusable.insert(worker_id);
                continue;
            }
            moved.insert(worker_id);
        }
    }
    Some((moved.len(), unusable))
}

/// Rewrites every placement of `worker_id` below `prefix` onto `prefix/segment`.
fn place(
    assignments: &mut [GroupAssignment],
    prefix: &[String],
    segment: &str,
    worker_id: &str,
) -> bool {
    let mut placed = false;
    for assignment in assignments.iter_mut() {
        if assignment.worker_id != worker_id || !below(&assignment.path, prefix) {
            continue;
        }
        let remainder = assignment
            .path
            .get(prefix.len() + 1..)
            .unwrap_or_default()
            .to_vec();
        let mut path = prefix.to_vec();
        path.push(segment.to_string());
        path.extend(remainder);
        assignment.path = path;
        placed = true;
    }
    placed
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_retrieval::StaticCompletionModel;

    use super::super::fixtures::{assignment, config, item, owned, paths, shaping};
    use super::*;
    use crate::error::ErrorKind;
    use crate::llm::FallbackMode;

    fn items() -> Vec<SourceItem> {
        vec![
            item("w1", "Alpha", "Writes documents"),
            item("w2", "Beta", "Reads media"),
        ]
    }

    fn skeleton() -> Vec<GroupAssignment> {
        vec![
            assignment("w1", &["general"]),
            assignment("w2", &["general"]),
        ]
    }

    fn nested_skeleton() -> Vec<GroupAssignment> {
        vec![
            assignment("w1", &["general", "pdf"]),
            assignment("w2", &["general", "pdf"]),
        ]
    }

    const SPLIT: &str = r#"{"children":[
        {"segment":"documents","worker_ids":["w1"]},
        {"segment":"media","worker_ids":["w2"]}
    ]}"#;

    #[tokio::test]
    async fn expansion_splits_the_root_with_the_model() {
        let model = StaticCompletionModel::single(SPLIT);
        let outcome = expansion(&items(), &skeleton(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Model);
        assert_eq!(outcome.model_calls, 1);
        assert_eq!(outcome.dropped, Vec::<String>::new());
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                ("w1".to_string(), owned(&["documents"])),
                ("w2".to_string(), owned(&["media"])),
            ]
        );
    }

    #[tokio::test]
    async fn expansion_stops_at_the_call_budget() {
        let model = StaticCompletionModel::single(SPLIT);
        let shaping = shaping().with_max_calls(1);
        let outcome = expansion(&items(), &skeleton(), &config(), &shaping, Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.model_calls, 1);
        assert_eq!(model.call_count(), 1);
        assert_eq!(outcome.provenance, ShapingProvenance::Model);
    }

    #[tokio::test]
    async fn expansion_keeps_the_subtree_of_an_unusable_branch() {
        let model = StaticCompletionModel::new(vec![SPLIT.to_string(), "no json here".to_string()]);
        let outcome = expansion(
            &items(),
            &nested_skeleton(),
            &config(),
            &shaping(),
            Some(&model),
        )
        .await
        .unwrap();
        // The root split applied; both deeper branches answered unusably.
        assert_eq!(outcome.provenance, ShapingProvenance::Model);
        assert_eq!(outcome.model_calls, 3);
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                ("w1".to_string(), owned(&["documents", "pdf"])),
                ("w2".to_string(), owned(&["media", "pdf"])),
            ]
        );
    }

    #[tokio::test]
    async fn expansion_falls_back_when_every_reply_is_unusable() {
        let model = StaticCompletionModel::single("garbage");
        let outcome = expansion(
            &items(),
            &nested_skeleton(),
            &config(),
            &shaping(),
            Some(&model),
        )
        .await
        .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::DeterministicFallback);
        assert_eq!(outcome.model_calls, 2);
        assert_eq!(
            paths(&outcome.assignments),
            paths(&nested_skeleton()),
            "the skeleton's own subtree stands"
        );
    }

    #[tokio::test]
    async fn expansion_error_mode_surfaces_the_failure() {
        let model = StaticCompletionModel::single("garbage");
        let shaping = shaping().with_fallback(FallbackMode::Error);
        let error = expansion(&items(), &skeleton(), &config(), &shaping, Some(&model))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("expansion"), "{error}");
    }

    #[tokio::test]
    async fn expansion_without_a_model_answers_the_skeleton() {
        let skeleton = vec![assignment("w1", &["Not Slugged"])];
        let outcome = expansion(&items(), &skeleton, &config(), &shaping(), None)
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Skipped);
        assert_eq!(outcome.model_calls, 0);
        assert_eq!(outcome.assignments, skeleton);
    }

    #[tokio::test]
    async fn expansion_truncates_to_the_depth_bound() {
        let skeleton = vec![
            assignment("w1", &["a", "b", "c", "d"]),
            assignment("w2", &["a", "b", "c", "d"]),
        ];
        let config = config().with_max_depth(3);
        let model = StaticCompletionModel::single("garbage");
        let outcome = expansion(&items(), &skeleton, &config, &shaping(), Some(&model))
            .await
            .unwrap();
        for (_, path) in paths(&outcome.assignments) {
            assert_eq!(path, owned(&["a", "b"]));
        }
    }

    #[tokio::test]
    async fn expansion_folds_the_overflow_under_other() {
        let items = vec![
            item("w1", "Alpha", "One"),
            item("w2", "Beta", "Two"),
            item("w3", "Gamma", "Three"),
            item("w4", "Delta", "Four"),
        ];
        let skeleton: Vec<GroupAssignment> = items
            .iter()
            .map(|item| assignment(&item.worker_id, &["general"]))
            .collect();
        let model = StaticCompletionModel::single(
            r#"{"children":[
                {"segment":"alpha","worker_ids":["w1"]},
                {"segment":"beta","worker_ids":["w2"]},
                {"segment":"gamma","worker_ids":["w3"]},
                {"segment":"delta","worker_ids":["w4"]}
            ]}"#,
        );
        let config = config().with_max_children(2);
        let outcome = expansion(&items, &skeleton, &config, &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.assignments.len(), items.len());
        let placed = super::super::path_map(&outcome.assignments);
        assert_eq!(placed.get("w1"), Some(&owned(&["alpha"])));
        assert_eq!(placed.get("w2"), Some(&owned(&["other"])));
        assert_eq!(placed.get("w3"), Some(&owned(&["other"])));
        assert_eq!(placed.get("w4"), Some(&owned(&["other"])));
    }

    #[tokio::test]
    async fn expansion_drops_ids_outside_the_branch() {
        // `ghost` is not an item and `w1` is claimed twice, so both are dropped.
        let model = StaticCompletionModel::single(
            r#"{"children":[
                {"segment":"documents","worker_ids":["w1","ghost"]},
                {"segment":"media","worker_ids":["w2","w1"]}
            ]}"#,
        );
        let outcome = expansion(&items(), &skeleton(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.dropped, vec!["ghost".to_string(), "w1".to_string()]);
        assert_eq!(outcome.assignments.len(), 2);
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                ("w1".to_string(), owned(&["documents"])),
                ("w2".to_string(), owned(&["media"])),
            ],
            "an item moves at most once"
        );
    }

    #[tokio::test]
    async fn expansion_prompts_are_byte_stable() {
        let first_model = StaticCompletionModel::single(SPLIT);
        let first = expansion(
            &items(),
            &skeleton(),
            &config(),
            &shaping(),
            Some(&first_model),
        )
        .await
        .unwrap();
        let second_model = StaticCompletionModel::single(SPLIT);
        let second = expansion(
            &items(),
            &skeleton(),
            &config(),
            &shaping(),
            Some(&second_model),
        )
        .await
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first_model.prompts(), second_model.prompts());
        assert!(first_model.prompts()[0].contains("\"stage\":\"expansion\""));
    }
}
