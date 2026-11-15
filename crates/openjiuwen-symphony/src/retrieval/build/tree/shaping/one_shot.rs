//! One-prompt placement of every item: the `one_shot` tree-shaping stage.
//!
//! Python's `grouping.py` derives the whole branch structure of an index with a model,
//! then assigns every skill in a second pass. This stage collapses both passes into one
//! prompt that names the items and asks for their branch paths, which is what a small tree
//! needs.
//!
//! # Request and reply
//!
//! The user payload is
//! `{"stage":"one_shot","root":…,"max_depth":…,"max_children":…,"items":[…]}`; each item
//! carries `worker_id`, `name`, `description`, `classification` and `tags`, shortened to
//! [`ShapingConfig::description_limit`](crate::retrieval::build::tree::shaping::ShapingConfig::description_limit)
//! characters, and at most
//! [`ShapingConfig::max_items_per_prompt`](crate::retrieval::build::tree::shaping::ShapingConfig::max_items_per_prompt)
//! of them are sent (the first entries in `(worker_id, name)` order, so a prompt is
//! byte-stable). The reply must be
//! `{"assignments":[{"worker_id":"…","path":["documents","pdf"]}]}`; `path` may also be one
//! string joined with `/`, `.` or `>`.
//!
//! Every id the reply names must exist, and the first entry wins: an unknown id or a
//! repeat is dropped and listed in
//! [`ShapingOutcome::dropped`](crate::retrieval::build::tree::shaping::ShapingOutcome::dropped).
//! An item the reply omits — including every item a shortened prompt never showed — keeps
//! the path the deterministic grouping derives for it, so the stage always answers exactly
//! one placement per item and never loses one.
//!
//! # The deterministic fallback
//!
//! With no model configured, or with
//! [`ShapingConfig::max_calls`](crate::retrieval::build::tree::shaping::ShapingConfig::max_calls)
//! at `0`, the stage never calls a model and reports provenance
//! [`Skipped`](crate::retrieval::build::tree::shaping::ShapingProvenance::Skipped) with the
//! deterministic grouping ([`group_items`]).
//! When a model *is* called but the call fails or the reply is unusable, the same grouping
//! answers with provenance
//! [`DeterministicFallback`](crate::retrieval::build::tree::shaping::ShapingProvenance::DeterministicFallback)
//! after one `tracing::warn!`; under
//! [`FallbackMode::Error`](crate::llm::FallbackMode) the failure is returned instead.
//!
//! `group_items` reports
//! [`Error::Unsupported`](crate::error::Error) for the model/ML grouping strategies, so a
//! `one_shot` run over an `llm_grouping` configuration cannot answer deterministically and
//! propagates that typed error.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;

use crate::error::Result;
use crate::llm::CompletionModel;
use crate::retrieval::build::grouping::{group_items, GroupAssignment, GroupingConfig};
use crate::retrieval::build::scanner::SourceItem;

use super::{
    assemble, call_model, deterministic_outcome, deterministic_paths, finalize, item_value,
    no_usable_reply, on_model_failure, prompt_len, read_assignments, skipped, sorted_items,
    PathLimits, ShapingConfig, ShapingMode, ShapingOutcome, ShapingProvenance,
};

/// The system instruction sent with the one-shot placement request.
const SYSTEM_PROMPT: &str = "Place capability items in a routing tree. Return JSON only: \
{\"assignments\":[{\"worker_id\":\"…\",\"path\":[\"documents\",\"pdf\"]}]}. \
Use one entry per worker id, at most the given depth, and short lowercase slug segments; \
prefer two or three levels and fewer than the fan-out limit of children per branch.";

/// Places every item with one prompt, or with the deterministic grouping.
///
/// # Errors
///
/// The propagated model failure under
/// [`FallbackMode::Error`](crate::llm::FallbackMode), or the
/// [`Error::Unsupported`](crate::error::Error) the deterministic grouping reports for a
/// strategy this crate cannot run.
pub async fn one_shot(
    items: &[SourceItem],
    config: &GroupingConfig,
    shaping: &ShapingConfig,
    model: Option<&dyn CompletionModel>,
) -> Result<ShapingOutcome> {
    let limits = PathLimits::from_config(config);
    let Some(model) = model else {
        return Ok(skipped(&group_items(items, config)?, ShapingMode::OneShot));
    };
    if shaping.max_calls == 0 {
        return Ok(skipped(&group_items(items, config)?, ShapingMode::OneShot));
    }

    let known: BTreeSet<&str> = items.iter().map(|item| item.worker_id.as_str()).collect();
    let ordered = sorted_items(items);
    let rendered: Vec<serde_json::Value> = ordered
        .iter()
        .take(prompt_len(ordered.len(), shaping))
        .map(|item| item_value(item, shaping.description_limit))
        .collect();
    let request = json!({
        "stage": "one_shot",
        "root": config.root_segment,
        "max_depth": limits.max_depth,
        "max_children": limits.max_children,
        "items": rendered,
    })
    .to_string();

    let selection = match call_model(model, SYSTEM_PROMPT, &request).await {
        Ok(Some(value)) => read_assignments(&value, &known),
        Ok(None) => None,
        Err(error) => {
            let mut warned = false;
            on_model_failure(shaping, &mut warned, "one_shot", error)?;
            return deterministic_outcome(items, config, ShapingMode::OneShot, 1);
        }
    };
    let Some((model_paths, dropped)) = selection else {
        let mut warned = false;
        on_model_failure(
            shaping,
            &mut warned,
            "one_shot",
            no_usable_reply("one_shot"),
        )?;
        return deterministic_outcome(items, config, ShapingMode::OneShot, 1);
    };

    let fallback: Option<BTreeMap<String, Vec<String>>> = if items
        .iter()
        .all(|item| model_paths.contains_key(&item.worker_id))
    {
        None
    } else {
        Some(deterministic_paths(items, config)?)
    };
    let assignments: Vec<GroupAssignment> = assemble(items, &model_paths, fallback.as_ref());
    Ok(finalize(
        assignments,
        limits,
        ShapingMode::OneShot,
        ShapingProvenance::Model,
        1,
        dropped,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_retrieval::StaticCompletionModel;

    use super::super::fixtures::{config, item, map_of, paths, shaping};
    use super::*;
    use crate::error::ErrorKind;
    use crate::llm::FallbackMode;

    fn items() -> Vec<SourceItem> {
        vec![
            item("w1", "Alpha", "Writes documents"),
            item("w2", "Beta", "Reads media"),
        ]
    }

    #[tokio::test]
    async fn one_shot_places_items_from_the_model_reply() {
        let model = StaticCompletionModel::single(
            r#"{"assignments":[{"worker_id":"w1","path":["documents","pdf"]},{"worker_id":"w2","path":"media/video"}]}"#,
        );
        let outcome = one_shot(&items(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Model);
        assert_eq!(outcome.mode, ShapingMode::OneShot);
        assert_eq!(outcome.model_calls, 1);
        assert_eq!(outcome.dropped, Vec::<String>::new());
        assert_eq!(
            paths(&outcome.assignments),
            vec![
                (
                    "w1".to_string(),
                    vec!["documents".to_string(), "pdf".to_string()]
                ),
                (
                    "w2".to_string(),
                    vec!["media".to_string(), "video".to_string()]
                ),
            ]
        );
    }

    #[tokio::test]
    async fn one_shot_drops_unknown_and_repeated_ids() {
        let model = StaticCompletionModel::single(
            r#"{"assignments":[
                {"worker_id":"w1","path":["alpha"]},
                {"worker_id":"w1","path":["beta"]},
                {"worker_id":"ghost","path":["gamma"]},
                {"worker_id":"w2","path":["delta"]}
            ]}"#,
        );
        let outcome = one_shot(&items(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Model);
        assert_eq!(outcome.dropped, vec!["ghost".to_string(), "w1".to_string()]);
        let placed = map_of(&outcome.assignments);
        assert_eq!(placed.get("w1"), Some(&vec!["alpha".to_string()]));
        assert_eq!(placed.get("w2"), Some(&vec!["delta".to_string()]));
    }

    #[tokio::test]
    async fn one_shot_backfills_the_items_the_reply_omits() {
        let model = StaticCompletionModel::single(
            r#"{"assignments":[{"worker_id":"w1","path":["documents"]}]}"#,
        );
        let outcome = one_shot(&items(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Model);
        assert_eq!(outcome.dropped, Vec::<String>::new());
        assert_eq!(outcome.assignments.len(), 2);
        let deterministic = map_of(&group_items(&items(), &config()).unwrap());
        let placed = map_of(&outcome.assignments);
        assert_eq!(placed.get("w1"), Some(&vec!["documents".to_string()]));
        assert_eq!(placed.get("w2"), deterministic.get("w2"));
    }

    #[tokio::test]
    async fn one_shot_falls_back_on_a_garbage_reply() {
        let model = StaticCompletionModel::single("sorry, I cannot help with that");
        let outcome = one_shot(&items(), &config(), &shaping(), Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::DeterministicFallback);
        assert_eq!(outcome.model_calls, 1);
        assert_eq!(
            paths(&outcome.assignments),
            paths(&group_items(&items(), &config()).unwrap())
        );
    }

    #[tokio::test]
    async fn one_shot_without_a_model_skips_the_stage() {
        let outcome = one_shot(&items(), &config(), &shaping(), None)
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Skipped);
        assert_eq!(outcome.model_calls, 0);
        assert_eq!(
            paths(&outcome.assignments),
            paths(&group_items(&items(), &config()).unwrap())
        );
    }

    #[tokio::test]
    async fn one_shot_with_no_budget_skips_the_stage() {
        let model = StaticCompletionModel::single("{}");
        let shaping = shaping().with_max_calls(0);
        let outcome = one_shot(&items(), &config(), &shaping, Some(&model))
            .await
            .unwrap();
        assert_eq!(outcome.provenance, ShapingProvenance::Skipped);
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn one_shot_error_mode_surfaces_an_unusable_reply() {
        let model = StaticCompletionModel::single("no json here");
        let shaping = shaping().with_fallback(FallbackMode::Error);
        let error = one_shot(&items(), &config(), &shaping, Some(&model))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("one_shot"), "{error}");
        assert!(error.message().contains("no usable JSON object"), "{error}");
    }

    #[tokio::test]
    async fn one_shot_error_mode_surfaces_an_invocation_failure() {
        // An empty script fails the call itself instead of answering a reply.
        let model = StaticCompletionModel::new(Vec::new());
        let shaping = shaping().with_fallback(FallbackMode::Error);
        let error = one_shot(&items(), &config(), &shaping, Some(&model))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.message().contains("no scripted responses"), "{error}");
    }

    #[tokio::test]
    async fn one_shot_prompts_are_byte_stable() {
        let reply = r#"{"assignments":[{"worker_id":"w1","path":["documents"]}]}"#;
        let first_model = StaticCompletionModel::single(reply);
        let first = one_shot(&items(), &config(), &shaping(), Some(&first_model))
            .await
            .unwrap();
        let second_model = StaticCompletionModel::single(reply);
        let second = one_shot(&items(), &config(), &shaping(), Some(&second_model))
            .await
            .unwrap();
        assert_eq!(first, second);
        let prompts = first_model.prompts();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts, second_model.prompts());
        assert!(prompts[0].contains("\"stage\":\"one_shot\""));
        assert!(prompts[0].contains("\"worker_id\":\"w1\""));
    }

    #[tokio::test]
    async fn one_shot_describes_a_bounded_item_slice() {
        let model = StaticCompletionModel::single("unusable");
        let shaping = shaping().with_max_items_per_prompt(1);
        let _ = one_shot(&items(), &config(), &shaping, Some(&model))
            .await
            .unwrap();
        let prompt = model.last_prompt().unwrap();
        assert!(prompt.contains("\"worker_id\":\"w1\""), "{prompt}");
        assert!(!prompt.contains("\"worker_id\":\"w2\""), "{prompt}");
    }

    #[tokio::test]
    async fn one_shot_refuses_an_unsupported_grouping_strategy() {
        // The deterministic answer is `group_items`, which reports the deferral.
        let config = config()
            .with_strategy(crate::retrieval::build::grouping::GroupingStrategy::LlmGrouping);
        let error = one_shot(&items(), &config, &shaping(), None)
            .await
            .unwrap_err();
        assert!(crate::error::is_unsupported(&error), "{error}");
    }
}
