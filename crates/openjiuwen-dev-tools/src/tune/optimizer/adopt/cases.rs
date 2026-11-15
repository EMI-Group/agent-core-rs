//! Failing/passing case selection and the deterministic "textual gradient".
//!
//! Ports `_get_bad_cases` from Python's `optimizer/adopt/adopt_optimizer.py`
//! plus the cases `utils.py`'s gradient prompts are rendered from. Python takes
//! every case scored exactly `0.0` as "bad" and every case scored exactly `1.0`
//! as "good", then `random.sample`s each list down to
//! [`DEFAULT_BAD_CASES_SAMPLE_NUM`]. This port keeps the exact-score split (a
//! partially scored case is evidence of neither) and replaces the random sample
//! with a deterministic selection ordered by case id, then score, so a run is
//! reproducible and a test can pin the selection.
//!
//! [`describe_gradient_context`] is the deterministic artifact that stands in for
//! Python's chain of LLM prompts: the rendered block names each failing case, its
//! answer, the expected label, and the evaluator's feedback, and is exactly what
//! a model-backed [`GradientPolicy`](super::GradientPolicy) would be asked to
//! summarize.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde_json::Value;

use crate::tune::EvaluatedCase;

/// Python `DEFAULT_BAD_CASES_SAMPLE_NUM`: how many failing/passing cases one
/// adoption run looks at.
pub const DEFAULT_BAD_CASES_SAMPLE_NUM: usize = 5;

/// The failing cases of `cases`, at most `limit` of them.
///
/// A case is failing when its score is exactly `0.0`. The selection is sorted by
/// case id, then score, and truncated, so the result depends only on the input
/// order-independent content. `limit == 0` selects nothing.
#[must_use]
pub fn select_bad_cases(cases: &[EvaluatedCase], limit: usize) -> Vec<EvaluatedCase> {
    select(cases, limit, 0.0)
}

/// The passing cases of `cases`, at most `limit` of them.
///
/// A case is passing when its score is exactly `1.0`. Ordering and truncation
/// match [`select_bad_cases`].
#[must_use]
pub fn select_good_cases(cases: &[EvaluatedCase], limit: usize) -> Vec<EvaluatedCase> {
    select(cases, limit, 1.0)
}

/// The cases scored exactly `score`, deterministically ordered and truncated.
///
/// The comparison is exact on purpose: Python's `_get_bad_cases` splits on
/// `case.score == 0.0` and `case.score == 1.0`, so a banded (partial) score is
/// evidence of neither and is excluded by both selections.
#[allow(clippy::float_cmp)] // Python's exact-score split, see above.
fn select(cases: &[EvaluatedCase], limit: usize, score: f64) -> Vec<EvaluatedCase> {
    if limit == 0 {
        return Vec::new();
    }
    let mut selected: Vec<EvaluatedCase> = cases
        .iter()
        .filter(|case| case.score == score)
        .cloned()
        .collect();
    selected.sort_by(compare);
    selected.truncate(limit);
    selected
}

/// Orders by case id, then by score, so equal scores keep a stable order.
fn compare(left: &EvaluatedCase, right: &EvaluatedCase) -> Ordering {
    left.case.case_id.cmp(&right.case.case_id).then_with(|| {
        left.score
            .partial_cmp(&right.score)
            .unwrap_or(Ordering::Equal)
    })
}

/// Renders the failing cases into the deterministic "textual gradient" block.
///
/// One block per case, in the order given:
///
/// ```text
/// case 0: <case id>
///   score: 0.000
///   input: {"query":"2 + 2?"}
///   answer: {"output":"5"}
///   expected: {"output":"4"}
///   feedback: <the evaluator's reason>
/// ```
///
/// A blank reason renders as `(no feedback recorded)` and a blank map as `{}`,
/// so the artifact never carries an empty line where a value belongs. An empty
/// input yields the single line `no failing cases`.
///
/// Python renders the same information into an LLM prompt; here the rendering is
/// the whole artifact, which keeps the crate model-free and the output testable.
#[must_use]
pub fn describe_gradient_context(bad_cases: &[EvaluatedCase]) -> String {
    if bad_cases.is_empty() {
        return "no failing cases".to_owned();
    }
    let mut lines = Vec::with_capacity(bad_cases.len() * 6);
    for (index, evaluated) in bad_cases.iter().enumerate() {
        lines.push(format!("case {index}: {}", evaluated.case.case_id));
        lines.push(format!("  score: {:.3}", evaluated.score));
        lines.push(format!("  input: {}", render_map(&evaluated.case.inputs)));
        lines.push(format!("  answer: {}", render_map(&evaluated.answer)));
        lines.push(format!("  expected: {}", render_map(&evaluated.case.label)));
        lines.push(format!("  feedback: {}", feedback(&evaluated.reason)));
    }
    lines.join("\n")
}

/// The evaluator's feedback, or the documented placeholder when it is blank.
fn feedback(reason: &str) -> &str {
    if reason.trim().is_empty() {
        "(no feedback recorded)"
    } else {
        reason
    }
}

/// Renders an input/label/answer map as compact JSON.
fn render_map(map: &BTreeMap<String, Value>) -> String {
    let object: serde_json::Map<String, Value> = map
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Value::Object(object).to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::tune::Case;

    use super::*;

    /// An evaluated case with `id`, `score`, and `reason`.
    fn evaluated(id: &str, score: f64, reason: &str) -> EvaluatedCase {
        let inputs = BTreeMap::from([("query".to_owned(), json!(format!("question {id}")))]);
        let label = BTreeMap::from([("output".to_owned(), json!(format!("answer {id}")))]);
        let case = Case::new(inputs, label).unwrap().with_case_id(id);
        let answer = BTreeMap::from([("output".to_owned(), json!(format!("given {id}")))]);
        EvaluatedCase::new(case, answer, score, reason).unwrap()
    }

    #[test]
    fn bad_cases_are_the_zero_scored_ones_in_case_id_order() {
        let cases = [
            evaluated("c", 0.0, "wrong"),
            evaluated("a", 0.0, "also wrong"),
            evaluated("b", 0.5, "partly right"),
            evaluated("d", 1.0, "right"),
        ];
        let bad = select_bad_cases(&cases, DEFAULT_BAD_CASES_SAMPLE_NUM);
        let ids: Vec<&str> = bad.iter().map(|case| case.case.case_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "c"]);
        // Reversing the input does not change the selection.
        let mut reversed = cases;
        reversed.reverse();
        let again = select_bad_cases(&reversed, DEFAULT_BAD_CASES_SAMPLE_NUM);
        assert_eq!(again, bad);
    }

    #[test]
    fn good_cases_are_the_perfectly_scored_ones() {
        let cases = [
            evaluated("b", 1.0, "right"),
            evaluated("a", 1.0, "right too"),
            evaluated("c", 0.9, "nearly"),
            evaluated("d", 0.0, "wrong"),
        ];
        let good = select_good_cases(&cases, DEFAULT_BAD_CASES_SAMPLE_NUM);
        let ids: Vec<&str> = good.iter().map(|case| case.case.case_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
        assert!(good
            .iter()
            .all(|case| (case.score - 1.0).abs() < f64::EPSILON));
    }

    #[test]
    fn the_limit_truncates_and_zero_selects_nothing() {
        let cases: Vec<EvaluatedCase> = ["a", "b", "c", "d"]
            .iter()
            .map(|id| evaluated(id, 0.0, "wrong"))
            .collect();
        assert_eq!(select_bad_cases(&cases, 0).len(), 0);
        assert_eq!(select_bad_cases(&cases, 1).len(), 1);
        assert_eq!(select_bad_cases(&cases, 2)[1].case.case_id, "b");
        assert_eq!(select_bad_cases(&cases, 99).len(), 4);
        assert_eq!(select_good_cases(&cases, 99).len(), 0);
        assert_eq!(select_bad_cases(&[], 5).len(), 0);
    }

    #[test]
    fn an_empty_gradient_context_says_so() {
        assert_eq!(describe_gradient_context(&[]), "no failing cases");
    }

    #[test]
    fn the_gradient_context_names_every_case_and_its_feedback() {
        let cases = [
            evaluated("a", 0.0, "quoted the wrong field"),
            evaluated("b", 0.0, ""),
        ];
        let rendered = describe_gradient_context(&cases);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 12);
        assert_eq!(lines[0], "case 0: a");
        assert_eq!(lines[1], "  score: 0.000");
        assert_eq!(lines[2], r#"  input: {"query":"question a"}"#);
        assert_eq!(lines[3], r#"  answer: {"output":"given a"}"#);
        assert_eq!(lines[4], r#"  expected: {"output":"answer a"}"#);
        assert_eq!(lines[5], "  feedback: quoted the wrong field");
        assert_eq!(lines[6], "case 1: b");
        assert_eq!(lines[11], "  feedback: (no feedback recorded)");
        // The rendering is a pure function of the cases.
        assert_eq!(rendered, describe_gradient_context(&cases));
    }

    #[test]
    fn empty_maps_render_as_empty_objects() {
        // `Case::new` rejects empty maps, but deserialized cases may still carry
        // them, so the rendering has to cope.
        let case = Case {
            inputs: BTreeMap::new(),
            label: BTreeMap::new(),
            tools: None,
            case_id: "empty".to_owned(),
        };
        let evaluated = EvaluatedCase::new(case, BTreeMap::new(), 0.0, "no answer").unwrap();
        let rendered = describe_gradient_context(&[evaluated]);
        assert!(rendered.contains("  input: {}"), "{rendered}");
        assert!(rendered.contains("  answer: {}"), "{rendered}");
        assert!(rendered.contains("  expected: {}"), "{rendered}");
    }
}
