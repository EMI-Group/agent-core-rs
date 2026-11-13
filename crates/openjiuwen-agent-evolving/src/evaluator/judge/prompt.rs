//! The judge prompt: rubric rendering and deterministic message construction.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use openjiuwen_core::foundation::prompt::BaseMessage;
use serde_json::Value;

use super::rubric::JudgeRubric;
use super::scale_of;
use crate::dataset::Case;

/// Renders a JSON object with deterministic (sorted) key order.
fn object_text(values: &BTreeMap<String, Value>) -> String {
    Value::Object(values.clone().into_iter().collect()).to_string()
}

/// Shortens `text` to `limit` characters, appending how much was dropped.
///
/// `limit == 0` disables truncation. Truncation never splits a character.
fn truncate_text(text: &str, limit: usize) -> String {
    if limit == 0 {
        return text.to_owned();
    }
    let total = text.chars().count();
    if total <= limit {
        return text.to_owned();
    }
    let mut truncated: String = text.chars().take(limit).collect();
    let _ = write!(truncated, "... [truncated {} chars]", total - limit);
    truncated
}

/// Builds the `[system, user]` messages one judge call sends.
///
/// The prompt is a pure function of its arguments: the system message states
/// the rubric and demands exactly one JSON verdict object, the user message
/// renders the case inputs, the expected label, and the prediction as compact
/// JSON (sorted keys, so the text is stable). The prediction is truncated to
/// `max_prediction_chars` characters — `0` means no truncation — with an explicit
/// `... [truncated N chars]` marker, so a huge agent transcript cannot push the
/// rubric out of the context window silently.
#[must_use]
pub fn build_judge_messages(
    case: &Case,
    prediction: &Value,
    rubric: &JudgeRubric,
    max_prediction_chars: usize,
) -> Vec<BaseMessage> {
    let scale = scale_of(rubric);
    let mut system = String::from(
        "You are an impartial evaluator scoring one agent response against an expected label.\n",
    );
    let _ = writeln!(
        system,
        "Score every criterion on a 0..{scale} scale, where 0 means unusable and {scale} means fully correct."
    );
    system.push_str("Rubric:\n");
    for criterion in &rubric.criteria {
        let description = criterion.description.trim();
        if description.is_empty() {
            let _ = writeln!(system, "- {} (weight {})", criterion.name, criterion.weight);
        } else {
            let _ = writeln!(
                system,
                "- {} (weight {}): {description}",
                criterion.name, criterion.weight
            );
        }
    }
    system.push_str(
        "Reply with exactly ONE strict JSON object and nothing else: no prose, no markdown fence, no commentary.\n",
    );
    system.push_str(
        "{\"score\": <number>, \"per_criterion\": {\"<criterion name>\": <number>}, \"reasoning\": \"<short justification>\"}\n",
    );
    let _ = writeln!(
        system,
        "Every criterion must appear in per_criterion, and both score and the per-criterion numbers use the 0..{scale} scale."
    );

    let mut user = String::from("Inputs:\n");
    user.push_str(&object_text(&case.inputs));
    user.push_str("\n\nExpected label:\n");
    user.push_str(&object_text(&case.label));
    user.push_str("\n\nPrediction:\n");
    user.push_str(&truncate_text(
        &prediction.to_string(),
        max_prediction_chars,
    ));

    vec![BaseMessage::system(system), BaseMessage::user(user)]
}

#[cfg(test)]
mod tests {
    use openjiuwen_core::foundation::prompt::Role;
    use serde_json::json;

    use super::super::test_support::case;
    use super::*;

    #[test]
    fn build_messages_is_deterministic_and_labelled() {
        let rubric = JudgeRubric::default_rubric();
        let prediction = json!({"output": "4"});
        let messages = build_judge_messages(&case(), &prediction, &rubric, 0);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);

        let system = messages[0].text().into_owned();
        for criterion in rubric.criteria() {
            assert!(system.contains(&criterion.name), "system: {system}");
        }
        assert!(system.contains("ONE strict JSON object"));
        assert!(system.contains("\"per_criterion\""));

        let user = messages[1].text().into_owned();
        assert!(user.starts_with("Inputs:\n{\"question\":\"2 + 2?\"}"));
        assert!(user.contains("Expected label:\n{\"answer\":\"4\"}"));
        assert!(user.contains("Prediction:\n{\"output\":\"4\"}"));

        let again = build_judge_messages(&case(), &prediction, &rubric, 0);
        assert_eq!(messages, again);
    }

    #[test]
    fn build_messages_truncates_the_prediction_on_a_char_boundary() {
        let rubric = JudgeRubric::default_rubric();
        let prediction = json!("héllo wörld");
        let messages = build_judge_messages(&case(), &prediction, &rubric, 5);
        let user = messages[1].text().into_owned();
        // The rendered prediction is the JSON string, quotes included.
        assert!(user.contains("\"héll"), "user: {user}");
        assert!(user.contains("[truncated"), "user: {user}");

        // `0` keeps everything.
        let messages = build_judge_messages(&case(), &prediction, &rubric, 0);
        assert_eq!(
            messages[1].text().into_owned(),
            build_judge_messages(&case(), &prediction, &rubric, 0)[1]
                .text()
                .into_owned()
        );
        assert!(messages[1].text().into_owned().ends_with("\"héllo wörld\""));
    }
}
