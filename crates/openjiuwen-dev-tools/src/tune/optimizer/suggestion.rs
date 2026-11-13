//! Deterministic candidate generation shared by every optimizer.
//!
//! The optimizers differ in *how* they walk the candidate space, not in what a
//! candidate is: a rewritten prompt. This module owns that shared vocabulary —
//! placeholder preservation, the deterministic mutation bank, the model rewrite
//! request, the deterministic tie-break, and bounded parallel scoring — so the
//! four search strategies stay small and agree on the details.
//!
//! A mutation is a stable, append-only instruction block identified by a marker
//! phrase. Applying a mutation whose marker already appears in the prompt is a
//! no-op, which keeps repeated rounds from stacking duplicates and makes the
//! search idempotent: the same seed always yields the same candidate list in the
//! same order.
//!
//! Python's optimizers ask a model for the whole next prompt and keep only the
//! last answer. Here the deterministic bank is the baseline and a model is an
//! optional extra candidate: a model failure or an unparsable answer logs and
//! falls back to the bank instead of failing the run.
//!
//! The rewrite request also carries the case set's tool declarations inside a
//! `<tools_description>` block, matching Python's `str(tools)` rendering.
//! The block is omitted entirely when no case declares a tool, so a tool-free
//! case set produces the same prompt as before.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_foundation::tool::ToolInfo;

use crate::tune::base::{Case, EvaluatedCase};
use crate::tune::evaluator::CaseEvaluator;
use crate::util::extract_tag;

/// One entry of the deterministic mutation bank.
pub(crate) struct Mutation {
    /// A phrase that appears in the prompt once the mutation is applied.
    pub(crate) marker: &'static str,
    /// The instruction block appended when the mutation is applied.
    pub(crate) text: &'static str,
}

/// The deterministic mutation bank, ordered so earlier entries are the safest.
pub(crate) const MUTATIONS: &[Mutation] = &[
    Mutation {
        marker: "think step by step",
        text: "Think step by step: first restate the request, then work through it, and finally \
               state the result.",
    },
    Mutation {
        marker: "output format",
        text: "Respond with a short final answer in the requested output format.",
    },
    Mutation {
        marker: "never invent facts",
        text: "Never invent facts; if the request is ambiguous, ask one clarifying question \
               instead of guessing.",
    },
    Mutation {
        marker: "before finishing, verify",
        text: "Before finishing, verify the answer against every requirement and fix any \
               mismatch.",
    },
    Mutation {
        marker: "follow the constraints",
        text: "Follow the constraints exactly and keep the response free of unrelated \
               commentary.",
    },
    Mutation {
        marker: "use the provided context only",
        text: "Use the provided context only, and explain the answer in one or two sentences.",
    },
];

/// The system prompt of a model rewrite request.
const REWRITE_SYSTEM_PROMPT: &str = "\
You are a prompt-engineering expert. Improve the given system prompt so the assistant answers \
the failing cases correctly. Keep the prompt's purpose and language, keep every {{placeholder}} \
intact, do not add the concrete cases as examples, and keep the result concise. Answer with \
exactly one block:\n\
<PROMPT_OPTIMIZED>\n\
[the improved prompt]\n\
</PROMPT_OPTIMIZED>";

/// Extracts the `{{placeholder}}` names used by `prompt`.
///
/// A placeholder is a `{{name}}` pair whose trimmed name is non-empty and spans
/// no line break. Malformed braces are ignored rather than rejected, so a
/// prompt with a stray `{{` still reports its well-formed placeholders.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::tune::placeholders;
/// let names = placeholders("Answer {{question}} in {{language}}.");
/// assert_eq!(names.len(), 2);
/// assert!(names.contains("question"));
/// ```
#[must_use]
pub fn placeholders(prompt: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut rest = prompt;
    while let Some(start) = rest.find("{{") {
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            break;
        };
        // A nested `{{` means the outer pair was malformed; resume at the inner
        // opener instead of swallowing both.
        if let Some(nested) = after[..end].find("{{") {
            rest = &after[nested..];
            continue;
        }
        let name = after[..end].trim();
        if !name.is_empty() && !name.contains('\n') {
            names.insert(name.to_owned());
        }
        rest = &after[end + 2..];
    }
    names
}

/// Appends the seed's missing placeholders to `candidate`.
///
/// The missing names are appended one per line exactly as `{{name}}`, matching
/// Python's repair step. A candidate that already carries every seed
/// placeholder is returned unchanged.
pub(crate) fn ensure_placeholders(seed: &str, candidate: &str) -> String {
    let seed_names = placeholders(seed);
    if seed_names.is_empty() {
        return candidate.to_owned();
    }
    let present = placeholders(candidate);
    let missing: Vec<String> = seed_names
        .difference(&present)
        .map(|name| format!("{{{{{name}}}}}"))
        .collect();
    if missing.is_empty() {
        return candidate.to_owned();
    }
    format!("{}\n{}", candidate.trim_end(), missing.join("\n"))
}

/// Applies every still-unapplied deterministic mutation to `base`.
///
/// Candidates keep bank order; a mutation whose marker already appears in
/// `base` (case-insensitively) is skipped.
pub(crate) fn deterministic_mutations(base: &str) -> Vec<String> {
    let lower = base.to_lowercase();
    MUTATIONS
        .iter()
        .filter(|mutation| !lower.contains(mutation.marker))
        .map(|mutation| format!("{}\n\n{}", base.trim_end(), mutation.text))
        .collect()
}

/// Ensures placeholders and removes duplicates while preserving order.
pub(crate) fn normalized_candidates(seed: &str, candidates: Vec<String>) -> Vec<String> {
    let mut visited = BTreeSet::new();
    let mut normalized = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let candidate = ensure_placeholders(seed, &candidate);
        if visited.insert(candidate.clone()) {
            normalized.push(candidate);
        }
    }
    normalized
}

/// The offline default evaluator: [`DeterministicEvaluator`] seeded with the
/// prompt under optimisation.
///
/// Every optimizer uses this when no evaluator was injected, so an optimizer
/// constructed with no arguments still works and stays offline.
pub(crate) fn default_evaluator(prompt: &str) -> Arc<dyn CaseEvaluator> {
    Arc::new(crate::tune::evaluator::DeterministicEvaluator::new(prompt))
}

/// A short `provider/model` description of an optional model, for `Debug`.
pub(crate) fn describe_model(model: Option<&Arc<dyn Model>>) -> String {
    match model {
        Some(model) => format!("{}/{}", model.provider(), model.model_name()),
        None => "none".to_owned(),
    }
}

/// Rejects a blank prompt and an empty case list.
pub(crate) fn validate_inputs(prompt: &str, cases: &[Case]) -> Result<()> {
    if prompt.trim().is_empty() {
        return Err(Error::validation("prompt must not be blank"));
    }
    if cases.is_empty() {
        return Err(Error::validation("cases must not be empty"));
    }
    Ok(())
}

/// Whether two scores are equal up to floating-point noise.
pub(crate) fn scores_equal(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-12
}

/// Whether `(score, prompt)` is better than `(other_score, other_prompt)`.
///
/// Higher score wins; on a tie the lexicographically smaller prompt wins. That
/// makes every optimizer's tie-breaking exact, total, and independent of the
/// order candidates happened to be generated in.
pub(crate) fn is_better(score: f64, prompt: &str, other_score: f64, other_prompt: &str) -> bool {
    if !scores_equal(score, other_score) {
        return score > other_score;
    }
    prompt < other_prompt
}

/// The index of the best `(score, prompt)` pair, in candidate order.
///
/// Ties are broken by the earlier index first and then by the lexicographically
/// smaller prompt, so the seed wins over an equal-scoring rewrite that follows
/// it, and the choice never depends on map iteration order.
pub(crate) fn select_best_index(scores: &[f64], prompts: &[String]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for index in 0..scores.len().min(prompts.len()) {
        match best {
            None => best = Some(index),
            Some(current) => {
                let better = scores[index] > scores[current]
                    || (scores_equal(scores[index], scores[current])
                        && prompts[index] < prompts[current]);
                if better {
                    best = Some(index);
                }
            }
        }
    }
    best
}

/// Scores every candidate over `cases`, `num_parallel` candidates at a time.
///
/// Results come back in input order regardless of completion order, so the
/// search stays deterministic.
///
/// # Errors
///
/// Propagates the first evaluation failure.
pub(crate) async fn score_candidates(
    evaluator: &dyn CaseEvaluator,
    prompts: &[String],
    cases: &[Case],
    num_parallel: usize,
) -> Result<Vec<f64>> {
    let mut scores = Vec::with_capacity(prompts.len());
    for batch in prompts.chunks(num_parallel.max(1)) {
        let futures = batch
            .iter()
            .map(|prompt| evaluator.score_cases(prompt, cases));
        for result in futures::future::join_all(futures).await {
            scores.push(result?.0);
        }
    }
    Ok(scores)
}

/// Generates candidate prompts, deterministically and optionally with a model.
pub(crate) struct Mutator {
    /// The model consulted for extra rewrites, when one is attached.
    model: Option<Arc<dyn Model>>,
}

impl Mutator {
    /// Creates a mutator over an optional model.
    pub(crate) fn new(model: Option<Arc<dyn Model>>) -> Self {
        Self { model }
    }

    /// The deterministic mutations applicable to `base`.
    ///
    /// Associated rather than a method: the bank does not depend on whether a
    /// model is attached, which keeps `Mutator`'s state about rewrites only.
    pub(crate) fn deterministic(base: &str) -> Vec<String> {
        deterministic_mutations(base)
    }

    /// Asks the model to rewrite `base`, using `bad_cases` as evidence.
    ///
    /// `cases` supplies the tool declarations described in the request, as
    /// Python's `InstructionOptimizer` does. The declarations are rendered into
    /// the prompt text only: `request.tools` stays unset, because Python
    /// describes the tools instead of offering them to the model.
    ///
    /// Returns `None` when no model is attached, the call fails, the answer has
    /// no `<PROMPT_OPTIMIZED>` block, or the rewrite equals `base`. A failure is
    /// logged and never propagated: the deterministic bank remains the baseline.
    pub(crate) async fn model_rewrite(
        &self,
        base: &str,
        bad_cases: &[EvaluatedCase],
        cases: &[Case],
    ) -> Option<String> {
        let model = self.model.as_ref()?;
        let messages = rewrite_messages(base, bad_cases, cases);
        let request = ModelRequestConfig::new(model.model_name());
        let completion = match model.invoke(&messages, &request).await {
            Ok(completion) => completion,
            Err(error) => {
                tracing::warn!(%error, "prompt rewrite model failed; keeping the deterministic bank");
                return None;
            }
        };
        let answer = completion
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        let Some(rewrite) = extract_tag(&answer, "PROMPT_OPTIMIZED") else {
            tracing::warn!(
                "prompt rewrite answer carried no <PROMPT_OPTIMIZED> block; keeping the deterministic bank"
            );
            return None;
        };
        if rewrite.trim().is_empty() || rewrite.trim() == base.trim() {
            return None;
        }
        Some(rewrite)
    }
}

/// Python's `str(tools)`: the declarations the case set shares, as compact
/// JSON; `"None"` when no case declares a tool.
///
/// Declarations are flattened across `cases` in first-appearance order and
/// deduplicated by `function.name` with the first declaration winning, so the
/// rendering is deterministic and independent of map iteration order.
pub(crate) fn tools_description(cases: &[Case]) -> String {
    let mut seen = BTreeSet::new();
    let mut declared: Vec<&ToolInfo> = Vec::new();
    for case in cases {
        for tool in case.tools().into_iter().flatten() {
            if seen.insert(tool.function.name.as_str()) {
                declared.push(tool);
            }
        }
    }
    if declared.is_empty() {
        return "None".to_owned();
    }
    match serde_json::to_string(&declared) {
        Ok(json) => json,
        Err(error) => {
            tracing::warn!(%error, "tool declarations could not be rendered as JSON");
            // One unrenderable declaration must not drop the others.
            let rendered: Vec<String> = declared
                .iter()
                .map(|tool| serde_json::to_string(tool).unwrap_or_default())
                .collect();
            format!("[{}]", rendered.join(","))
        }
    }
}

/// The `<tools_description>…</tools_description>` block, or `None` when no case
/// declares tools.
///
/// `None` keeps the rewrite prompt byte-identical for a tool-free case set.
pub(crate) fn tools_block(cases: &[Case]) -> Option<String> {
    if !cases.iter().any(Case::has_tools) {
        return None;
    }
    Some(format!(
        "<tools_description>{}</tools_description>",
        tools_description(cases)
    ))
}

/// Renders the rewrite conversation: the prompt under revision, the declared
/// tools when there are any, and the cases that scored worst.
fn rewrite_messages(base: &str, bad_cases: &[EvaluatedCase], cases: &[Case]) -> Vec<BaseMessage> {
    let mut evidence = String::new();
    for bad in bad_cases {
        let inputs = bad
            .inputs()
            .iter()
            .map(|(key, value)| format!("{key}:{}", compact(value)))
            .collect::<Vec<_>>()
            .join(" | ");
        let label = bad
            .label()
            .iter()
            .map(|(key, value)| format!("{key}:{}", compact(value)))
            .collect::<Vec<_>>()
            .join(" | ");
        let answer = bad
            .answer
            .iter()
            .map(|(key, value)| format!("{key}:{}", compact(value)))
            .collect::<Vec<_>>()
            .join(" | ");
        let _ = write!(
            evidence,
            "[question]: {inputs}\n[expected answer]: {label}\n[assistant answer]: {answer}\n\
             [reason]: {}\n===\n",
            bad.reason
        );
    }
    let mut user = format!("<prompt_base>\n{base}\n</prompt_base>\n\n");
    if let Some(block) = tools_block(cases) {
        user.push_str(&block);
        user.push_str("\n\n");
    }
    let _ = write!(
        user,
        "<bad_cases>\n{evidence}</bad_cases>\n\n\
         Rewrite the prompt so those cases are handled correctly."
    );
    vec![
        BaseMessage::system(REWRITE_SYSTEM_PROMPT.to_owned()),
        BaseMessage::user(user),
    ]
}

/// Renders a JSON value compactly, strings unquoted.
fn compact(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// The cases that scored worst, capped at `limit`, preserving case order.
pub(crate) fn worst_cases(evaluated: &[EvaluatedCase], limit: usize) -> Vec<EvaluatedCase> {
    let mut worst: Vec<&EvaluatedCase> = evaluated.iter().filter(|case| case.score < 1.0).collect();
    worst.sort_by(|left, right| {
        left.score
            .partial_cmp(&right.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.case_id().cmp(right.case_id()))
    });
    worst.into_iter().take(limit).cloned().collect()
}

/// Renders the case inputs/label as Python's `key:value | …` example line.
pub(crate) fn dict_line(data: &BTreeMap<String, serde_json::Value>) -> String {
    data.iter()
        .map(|(key, value)| format!("{key}:{}", compact(value)))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use serde_json::Value;

    use super::*;

    fn case(id: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(id.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(id.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    #[test]
    fn placeholders_parses_well_formed_names() {
        let names = placeholders("Answer {{question}} in {{ language }}; ignore {{bad and {{ok}}");
        assert_eq!(
            names.into_iter().collect::<Vec<_>>(),
            vec![
                "language".to_owned(),
                "ok".to_owned(),
                "question".to_owned()
            ]
        );
        assert!(placeholders("no placeholders here").is_empty());
        assert!(placeholders("{{}}").is_empty());
        assert!(placeholders("{{unclosed").is_empty());
        assert!(placeholders("{{two\nlines}}").is_empty());
    }

    #[test]
    fn ensure_placeholders_appends_only_missing_names() {
        let seed = "Answer {{question}} in {{language}}.";
        assert_eq!(
            ensure_placeholders(seed, "Answer {{question}}."),
            "Answer {{question}}.\n{{language}}"
        );
        assert_eq!(
            ensure_placeholders(seed, "Answer {{question}} in {{language}}."),
            "Answer {{question}} in {{language}}."
        );
        assert_eq!(
            ensure_placeholders("no placeholders", "anything"),
            "anything"
        );
        assert_eq!(
            ensure_placeholders(seed, ""),
            "\n{{language}}\n{{question}}",
            "the repair still restores every placeholder"
        );
    }

    #[test]
    fn mutations_are_ordered_and_idempotent() {
        let seed = "Answer the question.";
        let first = deterministic_mutations(seed);
        assert_eq!(first.len(), MUTATIONS.len());
        assert!(first[0].starts_with(seed));
        assert!(first[0].contains("Think step by step"));

        let applied = &first[0];
        let second = deterministic_mutations(applied);
        assert_eq!(second.len(), MUTATIONS.len() - 1);
        assert!(
            second
                .iter()
                .all(|candidate| candidate.matches("Think step by step").count() == 1),
            "the applied mutation is not re-applied"
        );
    }

    #[test]
    fn normalized_candidates_repair_and_deduplicate() {
        let candidates = vec![
            "Answer {{q}}.".to_owned(),
            "Answer {{q}}.".to_owned(),
            "Think {{q}}.".to_owned(),
        ];
        let normalized = normalized_candidates("Answer {{q}}.", candidates);
        assert_eq!(
            normalized,
            vec!["Answer {{q}}.".to_owned(), "Think {{q}}.".to_owned()]
        );
    }

    #[test]
    fn validate_inputs_rejects_blank_prompts_and_empty_cases() {
        let error = validate_inputs("  \n ", &[case("a")]).unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("prompt"));

        let error = validate_inputs("prompt", &[]).unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("cases"));

        assert!(validate_inputs("prompt", &[case("a")]).is_ok());
    }

    #[test]
    fn tie_breaks_follow_score_then_prompt() {
        assert!(is_better(0.8, "z", 0.5, "a"));
        assert!(!is_better(0.5, "z", 0.8, "a"));
        assert!(is_better(0.5, "a", 0.5, "b"), "smaller prompt wins");
        assert!(!is_better(0.5, "b", 0.5, "a"));
        assert!(scores_equal(0.5, 0.5 + 1e-15));
    }

    #[test]
    fn select_best_index_prefers_earlier_index_then_prompt() {
        let prompts = vec!["b".to_owned(), "a".to_owned(), "c".to_owned()];
        assert_eq!(select_best_index(&[0.5, 0.5, 0.5], &prompts), Some(1));
        assert_eq!(select_best_index(&[0.4, 0.9, 0.2], &prompts), Some(1));
        assert_eq!(select_best_index(&[], &prompts), None);

        let equal = vec!["same".to_owned(), "same".to_owned()];
        assert_eq!(select_best_index(&[0.7, 0.7], &equal), Some(0));
    }

    #[tokio::test]
    async fn score_candidates_keeps_input_order_across_batches() {
        /// Scores the prompt length so every candidate differs.
        struct Length;

        #[async_trait::async_trait]
        impl CaseEvaluator for Length {
            async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                let score = (prompt.chars().count() as f64 / 100.0).min(1.0);
                EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "")
            }
        }

        let prompts: Vec<String> = ["a", "aa", "aaa", "aaaa", "aaaaa"]
            .iter()
            .map(|text| (*text).to_owned())
            .collect();
        let scores = score_candidates(&Length, &prompts, &[case("c1")], 2)
            .await
            .unwrap();
        assert_eq!(scores.len(), 5);
        for window in scores.windows(2) {
            assert!(window[0] < window[1], "scores keep input order: {scores:?}");
        }
        // A zero parallelism is clamped instead of panicking.
        assert!(score_candidates(&Length, &prompts, &[case("c1")], 0)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn mutator_without_a_model_offers_only_the_bank() {
        let mutator = Mutator::new(None);
        assert_eq!(Mutator::deterministic("seed").len(), MUTATIONS.len());
        assert!(mutator
            .model_rewrite("seed", &[EvaluatedCase::from(case("c1"))], &[])
            .await
            .is_none());
    }

    #[tokio::test]
    async fn mutator_extracts_a_model_rewrite() {
        let model = ScriptedModel::text(
            "<思考>ignore</思考>\n<PROMPT_OPTIMIZED>Answer {{question}} carefully.</PROMPT_OPTIMIZED>",
        );
        let mutator = Mutator::new(Some(Arc::new(model.clone())));
        let bad = [EvaluatedCase::new(case("c1"), BTreeMap::new(), 0.0, "wrong").unwrap()];
        let rewrite = mutator
            .model_rewrite("Answer {{question}}.", &bad, &[])
            .await;
        assert_eq!(rewrite.as_deref(), Some("Answer {{question}} carefully."));

        let call = model.last_call().expect("one model call");
        assert!(call.messages[0].text().contains("PROMPT_OPTIMIZED"));
        let user = call.messages[1].text();
        assert!(user.contains("Answer {{question}}."), "{user}");
        assert!(user.contains("[question]"), "{user}");
        assert!(user.contains("wrong"), "{user}");
    }

    #[tokio::test]
    async fn mutator_degrades_on_unparsable_answers_and_failures() {
        let mutator = Mutator::new(Some(Arc::new(ScriptedModel::text("just a rewrite"))));
        assert!(mutator.model_rewrite("seed", &[], &[]).await.is_none());

        let identical = Mutator::new(Some(Arc::new(ScriptedModel::text(
            "<PROMPT_OPTIMIZED>seed</PROMPT_OPTIMIZED>",
        ))));
        assert!(identical.model_rewrite("seed", &[], &[]).await.is_none());

        let failing = Mutator::new(Some(Arc::new(ScriptedModel::new([ScriptedTurn::fail(
            "provider down",
        )]))));
        assert!(failing.model_rewrite("seed", &[], &[]).await.is_none());
    }

    /// A single function declaration with an empty object schema.
    fn lookup(description: &str) -> ToolInfo {
        ToolInfo::new(
            "lookup",
            description,
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }

    #[test]
    fn tools_description_renders_python_parity_json() {
        let tool = lookup("Look something up");
        let cases = vec![case("c1").with_tools(vec![tool.clone()])];
        assert_eq!(
            tools_description(&cases),
            serde_json::to_string(&vec![&tool]).unwrap(),
            "the declaration renders as the compact JSON array Python prints"
        );
        assert!(tools_description(&cases).starts_with("[{\"type\":\"function\""));
    }

    #[test]
    fn tools_description_is_none_without_declarations() {
        assert_eq!(tools_description(&[]), "None");
        assert_eq!(tools_description(&[case("c1")]), "None");
        assert_eq!(tools_description(&[case("c1").with_tools(vec![])]), "None");
    }

    #[test]
    fn tools_block_deduplicates_across_cases() {
        let first = lookup("First declaration");
        let replacement = lookup("Second declaration");
        let other = ToolInfo::new("search", "Search things", serde_json::json!({}));
        let cases = vec![
            case("c1").with_tools(vec![first]),
            case("c2").with_tools(vec![replacement, other]),
        ];
        let block = tools_block(&cases).expect("the case set declares tools");
        assert!(block.starts_with("<tools_description>"), "{block}");
        assert!(block.ends_with("</tools_description>"), "{block}");
        assert_eq!(block.matches("\"lookup\"").count(), 1, "{block}");
        assert_eq!(block.matches("\"search\"").count(), 1, "{block}");
        assert!(
            block.contains("First declaration"),
            "the first declaration wins: {block}"
        );
        assert!(!block.contains("Second declaration"), "{block}");
        let lookup_at = block.find("\"lookup\"").expect("lookup listed");
        let search_at = block.find("\"search\"").expect("search listed");
        assert!(lookup_at < search_at, "first-appearance order: {block}");
    }

    #[test]
    fn tools_block_is_absent_without_declarations() {
        let cases = vec![case("c1")];
        assert_eq!(tools_block(&cases), None);
        assert_eq!(tools_block(&[]), None);

        let messages = rewrite_messages("seed", &[EvaluatedCase::from(case("c1"))], &cases);
        let user = messages[1].text();
        assert!(!user.contains("<tools_description>"), "{user}");
        assert!(!user.contains("None"), "{user}");
        assert_eq!(
            user,
            "<prompt_base>\nseed\n</prompt_base>\n\n<bad_cases>\n\
             [question]: question:c1\n[expected answer]: answer:c1\n[assistant answer]: \n\
             [reason]: \n===\n</bad_cases>\n\n\
             Rewrite the prompt so those cases are handled correctly."
        );
    }

    #[test]
    fn rewrite_messages_render_the_tools_block() {
        let cases = vec![case("c1").with_tools(vec![lookup("Look something up")])];
        let bad = [EvaluatedCase::new(case("c1"), BTreeMap::new(), 0.0, "wrong").unwrap()];
        let messages = rewrite_messages("Answer {{question}}.", &bad, &cases);
        assert_eq!(messages.len(), 2);
        let user = messages[1].text();
        assert!(user.starts_with("<prompt_base>"), "{user}");
        let tools_at = user.find("<tools_description>").expect("tools block");
        assert!(user.contains("lookup"), "{user}");
        assert!(
            user.contains("</tools_description>"),
            "the block is closed: {user}"
        );
        let bad_at = user.find("<bad_cases>").expect("bad cases block");
        assert!(tools_at < bad_at, "tools precede the evidence: {user}");
        assert!(user.contains("Rewrite the prompt so those cases"));
    }

    #[test]
    fn worst_cases_filters_and_orders_by_score() {
        let evaluated = [
            EvaluatedCase::new(case("a"), BTreeMap::new(), 1.0, "").unwrap(),
            EvaluatedCase::new(case("b"), BTreeMap::new(), 0.5, "").unwrap(),
            EvaluatedCase::new(case("c"), BTreeMap::new(), 0.0, "").unwrap(),
        ];
        let worst = worst_cases(&evaluated, 2);
        assert_eq!(worst.len(), 2);
        assert_eq!(worst[0].case_id(), "c");
        assert_eq!(worst[1].case_id(), "b");
        assert_eq!(worst_cases(&evaluated, 5).len(), 2);
        assert_eq!(worst_cases(&evaluated, 0).len(), 0);
    }

    #[test]
    fn dict_line_renders_pythons_example_shape() {
        let mut data = BTreeMap::new();
        data.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
        data.insert("count".to_owned(), serde_json::json!(3));
        assert_eq!(dict_line(&data), "count:3 | question:2 + 2?");
    }
}
