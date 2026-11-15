//! Merging equivalent sibling branches: the `equivalence` tree-shaping stage.
//!
//! Python's `build/tree/equivalence.py` asks a model which sibling branches of a built tree
//! mean the same thing and merges them. This stage ports that pass on top of a
//! deterministic pre-pass, so a tree with obvious duplicates is collapsed with no model at
//! all.
//!
//! # The deterministic pre-pass
//!
//! Sibling segments are grouped by
//! [`normalize_name_key`](crate::model::naming::normalize_name_key), and every group is
//! rewritten onto its lexicographically smallest member, so `PDF Fill`, `pdf_fill` and
//! `pdf-fill` become one branch. The canonical segment is a property of the group rather
//! than of the first insertion, so a rerun over the stage's own output changes nothing.
//! Paths are then slugged and held inside the shared depth and fan-out bounds.
//!
//! # Request and reply
//!
//! The user payload is
//! `{"stage":"equivalence","branches":[{"parent":["documents"],"segments":["pdf","pdf-fill"]}]}`
//! and lists every parent with at least two direct children, in parent order, at most
//! [`ShapingConfig::max_items_per_prompt`](crate::retrieval::build::tree::shaping::ShapingConfig::max_items_per_prompt)
//! of them. The reply must be
//! `{"merges":[{"parent":["documents"],"segments":["pdf","pdf-fill"],"into":"pdf"}]}`.
//!
//! A merge is applied only when its `parent` path is present and *every* entry of
//! `segments` is a direct child of that parent; anything else is ignored with a
//! `tracing::warn!`. An applied merge rewrites the matching segments of every placement
//! below the parent, keeping the rest of each path, so no placement is ever dropped.
//!
//! # The deterministic fallback
//!
//! With no model configured, or with
//! [`ShapingConfig::max_calls`](crate::retrieval::build::tree::shaping::ShapingConfig::max_calls)
//! at `0`, the stage answers the pre-pass result with provenance
//! [`Skipped`](crate::retrieval::build::tree::shaping::ShapingProvenance::Skipped). When a
//! model is called but fails or applies nothing,
//! [`FallbackMode::Deterministic`](crate::llm::FallbackMode) warns once and answers the
//! pre-pass result with provenance
//! [`DeterministicFallback`](crate::retrieval::build::tree::shaping::ShapingProvenance::DeterministicFallback);
//! [`FallbackMode::Error`](crate::llm::FallbackMode) returns the failure instead.
//!
//! Whatever the path, the answer must hold every input `worker_id` exactly once and keep
//! each path slugged and inside the bounds. A violation is `Error::Validation` naming
//! `assignments` (or `path`) — the stage never drops or invents an item, so a failure there
//! signals a bug rather than bad input.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::error::{validation, Result};
use crate::llm::{string_field, CompletionModel};
use crate::retrieval::build::grouping::{GroupAssignment, GroupingConfig};
use crate::retrieval::build::tree_text::slug_term;

use super::{
    below, bound_paths, call_model, finalize, no_usable_reply, normalize_paths, on_model_failure,
    path_field, prompt_len, raw_segments, segment_entries, slug_segments, strings_field,
    worker_counts, PathLimits, PathTrie, ShapingConfig, ShapingMode, ShapingOutcome,
    ShapingProvenance,
};

/// The system instruction sent with every equivalence request.
const SYSTEM_PROMPT: &str = "Merge semantically equivalent sibling branches of a capability \
routing tree. Return JSON only: {\"merges\":[{\"parent\":[\"documents\"],\"segments\":[\"pdf\",\"pdf-fill\"],\"into\":\"pdf\"}]}. \
Merge only branches that mean the same thing, keep the shortest segment as the target, and \
return an empty list when nothing should merge.";

/// Merges equivalent sibling branches, with a model when one is configured.
///
/// # Errors
///
/// The propagated model failure under
/// [`FallbackMode::Error`](crate::llm::FallbackMode), and `Error::Validation`
/// when the rewritten placements would not hold every worker id exactly once or would leave
/// the depth, fan-out or slug bounds.
pub async fn equivalence(
    assignments: &[GroupAssignment],
    config: &GroupingConfig,
    shaping: &ShapingConfig,
    model: Option<&dyn CompletionModel>,
) -> Result<ShapingOutcome> {
    let limits = PathLimits::from_config(config);
    let mut placements: Vec<GroupAssignment> = assignments.to_vec();
    let mut paths: Vec<Vec<String>> = placements
        .iter()
        .map(|assignment| raw_segments(&assignment.path, limits.max_segments()))
        .collect();
    collapse_equivalent(&mut paths, limits.max_segments());
    for (assignment, path) in placements.iter_mut().zip(paths) {
        assignment.path = path;
    }
    normalize_paths(&mut placements, limits);

    let Some(model) = model else {
        return finish(placements, limits, ShapingProvenance::Skipped, 0);
    };
    if shaping.max_calls == 0 {
        return finish(placements, limits, ShapingProvenance::Skipped, 0);
    }
    let request = merge_request(&placements, shaping);
    let mut warned = false;
    let merged = match call_model(model, SYSTEM_PROMPT, &request).await {
        Ok(Some(value)) => apply_merges(&mut placements, &value, limits),
        Ok(None) => None,
        Err(error) => {
            on_model_failure(shaping, &mut warned, "equivalence", error)?;
            return finish(
                placements,
                limits,
                ShapingProvenance::DeterministicFallback,
                1,
            );
        }
    };
    let Some(applied) = merged else {
        on_model_failure(
            shaping,
            &mut warned,
            "equivalence",
            no_usable_reply("equivalence"),
        )?;
        return finish(
            placements,
            limits,
            ShapingProvenance::DeterministicFallback,
            1,
        );
    };
    let provenance = if applied > 0 {
        ShapingProvenance::Model
    } else {
        ShapingProvenance::DeterministicFallback
    };
    finish(placements, limits, provenance, 1)
}

/// Collapses every group of sibling segments sharing a normalized name.
fn collapse_equivalent(paths: &mut [Vec<String>], limit: usize) {
    let mut trie = PathTrie::default();
    for (index, path) in paths.iter().enumerate() {
        trie.insert(index, path);
    }
    let trie = trie.collapse();
    let mut collapsed: Vec<Vec<String>> = vec![Vec::new(); paths.len()];
    trie.collect(&mut Vec::new(), &mut collapsed);
    for (path, collapsed) in paths.iter_mut().zip(collapsed) {
        *path = collapsed;
        path.truncate(limit);
    }
}

/// Renders the sibling groups worth merging.
fn merge_request(assignments: &[GroupAssignment], shaping: &ShapingConfig) -> String {
    let mut groups: BTreeMap<Vec<String>, BTreeSet<String>> = BTreeMap::new();
    for assignment in assignments {
        if let Some((segment, parent)) = assignment.path.split_last() {
            groups
                .entry(parent.to_vec())
                .or_default()
                .insert(segment.clone());
        }
    }
    let candidates = segment_entries(&groups);
    let limit = prompt_len(candidates.len(), shaping);
    let rendered: Vec<Value> = candidates.into_iter().take(limit).collect();
    json!({
        "stage": "equivalence",
        "branches": rendered,
    })
    .to_string()
}

/// Applies one `{"merges":[…]}` reply, returning how many merges it applied.
///
/// `None` means the reply carries no `merges` array. A merge whose parent is absent or
/// whose segments are not all direct children of that parent is ignored with a warning.
fn apply_merges(
    assignments: &mut [GroupAssignment],
    value: &Value,
    limits: PathLimits,
) -> Option<usize> {
    let merges = value.get("merges")?.as_array()?;
    let mut applied = 0_usize;
    for merge in merges {
        let parent = slug_segments(&path_field(merge, "parent"), limits.max_segments());
        let segments = slug_segments(&strings_field(merge, "segments"), limits.max_segments());
        let target =
            string_field(merge, "into").map_or_else(String::new, |into| slug_term(&into, ""));
        if segments.is_empty() || target.is_empty() {
            tracing::warn!(
                "tree shaping equivalence: ignoring a merge without segments or a target segment"
            );
            continue;
        }
        let present: BTreeSet<String> = assignments
            .iter()
            .filter(|assignment| below(&assignment.path, &parent))
            .filter_map(|assignment| assignment.path.get(parent.len()).cloned())
            .collect();
        if !segments.iter().all(|segment| present.contains(segment)) {
            tracing::warn!(
                target = target.as_str(),
                parent = parent.join("/").as_str(),
                "tree shaping equivalence: ignoring a merge naming a branch outside its parent"
            );
            continue;
        }
        let targets: BTreeSet<String> = segments.into_iter().collect();
        for assignment in assignments.iter_mut() {
            if !below(&assignment.path, &parent) {
                continue;
            }
            let replace = assignment
                .path
                .get(parent.len())
                .is_some_and(|segment| targets.contains(segment));
            if !replace {
                continue;
            }
            if let Some(slot) = assignment.path.get_mut(parent.len()) {
                slot.clone_from(&target);
            }
        }
        applied += 1;
    }
    Some(applied)
}

/// Normalizes, bounds, sorts and verifies the placements.
fn finish(
    placements: Vec<GroupAssignment>,
    limits: PathLimits,
    provenance: ShapingProvenance,
    model_calls: usize,
) -> Result<ShapingOutcome> {
    let expected = worker_counts(&placements);
    let outcome = finalize(
        placements,
        limits,
        ShapingMode::Equivalence,
        provenance,
        model_calls,
        BTreeSet::new(),
    );
    verify(&outcome.assignments, &expected, limits)?;
    Ok(outcome)
}

/// Proves the placements still hold every worker id once and respect the shared bounds.
fn verify(
    assignments: &[GroupAssignment],
    expected: &BTreeMap<String, usize>,
    limits: PathLimits,
) -> Result<()> {
    if worker_counts(assignments) != *expected {
        return Err(validation(
            "assignments",
            "the equivalence pass must preserve every worker id exactly once",
        ));
    }
    let mut paths: Vec<Vec<String>> = Vec::with_capacity(assignments.len());
    for assignment in assignments {
        if assignment.path.len() > limits.max_segments() {
            return Err(validation(
                "path",
                format!(
                    "{} segments exceed the limit of {}",
                    assignment.path.len(),
                    limits.max_segments()
                ),
            ));
        }
        for segment in &assignment.path {
            if segment.is_empty() || *segment != slug_term(segment, "") {
                return Err(validation(
                    "path",
                    format!("the segment {segment:?} is not a slug"),
                ));
            }
        }
        paths.push(assignment.path.clone());
    }
    let mut bounded = paths.clone();
    bound_paths(&mut bounded, limits);
    if bounded != paths {
        return Err(validation(
            "assignments",
            "the equivalence pass must keep every path inside the depth and fan-out bounds",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_retrieval::StaticCompletionModel;

    use super::super::fixtures::{assignment, config, owned, paths, shaping};
    use super::*;
    use crate::error::ErrorKind;
    use crate::llm::FallbackMode;

    const MERGE: &str = r#"{"merges":[
        {"parent":["documents"],"segments":["pdf","pdf-fill"],"into":"pdf"}
    ]}"#;

    fn siblings() -> Vec<GroupAssignment> {
        vec![
            assignment("w1", &["documents", "pdf"]),
            assignment("w2", &["documents", "pdf-fill"]),
        ]
    }

    #[tokio::test]
    async fn equivalence_collapses_normalized_siblings_without_a_model() {
        let input = vec![
            assignment("w1", &["PDF Fill"]),
            assignment("w2", &["pdf_fill"]),
            assignment("w3", &["PDF Fill", "forms"]),
        ];
        let outcome = equivalence(&input, &config(), &shaping(), None)
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Skipped);
        assert_eq!(outcome.model_calls, 0);
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                ("w1".to_string(), owned(&["pdf-fill"])),
                ("w2".to_string(), owned(&["pdf-fill"])),
                ("w3".to_string(), owned(&["pdf-fill", "forms"])),
            ]
        );
    }

    #[tokio::test]
    async fn equivalence_pre_pass_is_idempotent() {
        let input = vec![
            assignment("w1", &["PDF Fill"]),
            assignment("w2", &["pdf_fill"]),
            assignment("w3", &["PDF Fill", "forms"]),
        ];
        let first = equivalence(&input, &config(), &shaping(), None)
            .await
            .unwrap();
        let second = equivalence(&first.assignments, &config(), &shaping(), None)
            .await
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap()
        );
    }

    #[tokio::test]
    async fn equivalence_applies_a_model_merge_and_keeps_every_id() {
        let model = StaticCompletionModel::single(MERGE);
        let outcome = equivalence(&siblings(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Model);
        assert_eq!(outcome.model_calls, 1);
        assert_eq!(outcome.assignments.len(), 2);
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                ("w1".to_string(), owned(&["documents", "pdf"])),
                ("w2".to_string(), owned(&["documents", "pdf"])),
            ]
        );
    }

    #[tokio::test]
    async fn equivalence_ignores_a_merge_outside_the_tree() {
        let model = StaticCompletionModel::single(
            r#"{"merges":[
                {"parent":["documents"],"segments":["pdf","pdf-fill"],"into":"pdf"},
                {"parent":["missing"],"segments":["a","b"],"into":"a"},
                {"parent":["documents"],"segments":["pdf","ghost"],"into":"pdf"}
            ]}"#,
        );
        let outcome = equivalence(&siblings(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Model);
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                ("w1".to_string(), owned(&["documents", "pdf"])),
                ("w2".to_string(), owned(&["documents", "pdf"])),
            ]
        );
    }

    #[tokio::test]
    async fn equivalence_falls_back_on_a_garbage_reply() {
        let model = StaticCompletionModel::single("nothing to merge");
        let outcome = equivalence(&siblings(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::DeterministicFallback);
        assert_eq!(outcome.model_calls, 1);
        assert_eq!(paths(&outcome.assignments), paths(&siblings()));
    }

    #[tokio::test]
    async fn equivalence_treats_an_empty_merge_list_as_unusable() {
        let model = StaticCompletionModel::single(r#"{"merges":[]}"#);
        let outcome = equivalence(&siblings(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::DeterministicFallback);
    }

    #[tokio::test]
    async fn equivalence_error_mode_surfaces_the_failure() {
        let model = StaticCompletionModel::single("nothing to merge");
        let shaping = shaping().with_fallback(FallbackMode::Error);
        let error = equivalence(&siblings(), &config(), &shaping, Some(&model))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("equivalence"), "{error}");
    }

    #[tokio::test]
    async fn equivalence_holds_the_depth_and_fan_out_bounds() {
        let deep = vec![
            assignment("w1", &["a", "b", "c", "d"]),
            assignment("w2", &["a", "b", "c", "d"]),
        ];
        let deep_config = config().with_max_depth(3);
        let outcome = equivalence(&deep, &deep_config, &shaping(), None)
            .await
            .unwrap();
        for (_, path) in paths(&outcome.assignments) {
            assert_eq!(path, owned(&["a", "b"]));
        }

        let wide = vec![
            assignment("w1", &["alpha"]),
            assignment("w2", &["beta"]),
            assignment("w3", &["gamma"]),
            assignment("w4", &["delta"]),
        ];
        let wide_config = config().with_max_children(2);
        let outcome = equivalence(&wide, &wide_config, &shaping(), None)
            .await
            .unwrap();
        let placed = super::super::path_map(&outcome.assignments);
        assert_eq!(placed.get("w1"), Some(&owned(&["alpha"])));
        for worker_id in ["w2", "w3", "w4"] {
            assert_eq!(placed.get(worker_id), Some(&owned(&["other"])));
        }
    }

    #[tokio::test]
    async fn equivalence_prompts_are_byte_stable() {
        let first_model = StaticCompletionModel::single(MERGE);
        let first = equivalence(&siblings(), &config(), &shaping(), Some(&first_model))
            .await
            .unwrap();
        let second_model = StaticCompletionModel::single(MERGE);
        let second = equivalence(&siblings(), &config(), &shaping(), Some(&second_model))
            .await
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(first_model.prompts(), second_model.prompts());
        let prompt = first_model.last_prompt().unwrap();
        assert!(prompt.contains("\"stage\":\"equivalence\""), "{prompt}");
        assert!(
            prompt.contains("\"segments\":[\"pdf\",\"pdf-fill\"]"),
            "{prompt}"
        );
    }

    #[test]
    fn verify_rejects_a_lost_worker_id() {
        let limits = PathLimits {
            max_children: 8,
            max_depth: 6,
        };
        let expected = worker_counts(&siblings());
        assert!(verify(&siblings(), &expected, limits).is_ok());
        let lost = vec![assignment("w1", &["documents", "pdf"])];
        let error = verify(&lost, &expected, limits).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().starts_with("assignments:"), "{error}");
    }

    #[test]
    fn verify_rejects_a_path_outside_the_bounds() {
        let limits = PathLimits {
            max_children: 1,
            max_depth: 2,
        };
        let expected = worker_counts(&siblings());
        let error = verify(&siblings(), &expected, limits).unwrap_err();
        assert!(
            error.message().contains("bounds") || error.message().contains("path"),
            "{error}"
        );
    }
}
