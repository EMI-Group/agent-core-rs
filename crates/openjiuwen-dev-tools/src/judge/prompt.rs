//! Prompt rendering for the rubric judge.
//!
//! Three pure functions build the conversation a [`ModelJudge`] sends:
//!
//! - [`system_prompt`] is the fixed judge persona and output contract.
//! - [`render_rubric`] renders the numbered dimensions with their weights,
//!   descriptions, guidance, and the shared scale.
//! - [`rubric_messages`] assembles exactly one system message and one user
//!   message carrying the rubric, the subject, the instructions, the optional
//!   reference, and the required answer format.
//!
//! Rendering is deterministic: the same request always yields byte-identical
//! messages, no environment or clock is read, and no `{{placeholder}}` survives
//! into the output.
//!
//! [`ModelJudge`]: crate::judge::ModelJudge

use std::fmt::Write as _;

use openjiuwen_foundation::prompt::BaseMessage;

use crate::judge::model_judge::JudgeRequest;
use crate::judge::rubric::{Rubric, DEFAULT_PASS_THRESHOLD};

/// The built-in judge persona and output contract.
const SYSTEM_PROMPT: &str = "\
You are a strict evaluation judge. Score the subject against the supplied rubric and justify \
every dimension with concrete evidence from the subject.
Do not reward length, confidence, or verbosity; reward only what the rubric asks for.
Use the full scoring scale, reserve the maximum for flawless work, and never invent facts about \
the subject.
Answer with the requested structured sections and nothing else.";

/// Returns the built-in system prompt.
///
/// The text is a compile-time constant, so it is stable across runs and
/// platforms.
#[must_use]
pub fn system_prompt() -> &'static str {
    SYSTEM_PROMPT
}

/// Renders `rubric` as numbered, human-readable scoring dimensions.
///
/// The output lists the rubric name and description, the shared scale, and one
/// numbered block per dimension with its weight, description, and guidance.
#[must_use]
pub fn render_rubric(rubric: &Rubric) -> String {
    let mut rendered = String::new();
    // Writing to a `String` is infallible, so the result needs no handling.
    let _ = write_rubric(&mut rendered, rubric);
    rendered
}

/// Appends the rendered rubric to `out`.
fn write_rubric(out: &mut String, rubric: &Rubric) -> std::fmt::Result {
    let name = &rubric.name;
    let description = &rubric.description;
    let scale = rubric.scale;
    writeln!(out, "Rubric: {name}")?;
    writeln!(out, "{description}")?;
    writeln!(out, "Scale: score every dimension from 0.0 to {scale:.1}.")?;
    let count = rubric.dimensions.len();
    let total_weight = rubric.max_total_weight();
    writeln!(out, "Dimensions ({count} total weight {total_weight:.2}):")?;

    for (index, dimension) in rubric.dimensions.iter().enumerate() {
        let position = index + 1;
        let name = &dimension.name;
        let weight = dimension.weight;
        writeln!(out, "{position}. {name} (weight {weight:.2})")?;
        let description = &dimension.description;
        writeln!(out, "   {description}")?;
        if !dimension.guidance.trim().is_empty() {
            let guidance = &dimension.guidance;
            writeln!(out, "   Guidance: {guidance}")?;
        }
    }

    writeln!(
        out,
        "Scores are weighted and normalized to a single 0.0 to 1.0 judgement; a total of at least \
         {DEFAULT_PASS_THRESHOLD} passes."
    )
}

/// Builds the judge conversation for `request`.
///
/// The result is exactly one system message ([`system_prompt`]) followed by one
/// user message that renders the rubric, the subject, the instructions, the
/// optional reference, and the required answer format (tag sections, with the
/// JSON alternative documented).
#[must_use]
pub fn rubric_messages(request: &JudgeRequest<'_>) -> Vec<BaseMessage> {
    vec![
        BaseMessage::system(system_prompt()),
        BaseMessage::user(render_request(request)),
    ]
}

/// Renders the user message of a judge request.
fn render_request(request: &JudgeRequest<'_>) -> String {
    let mut user = String::new();
    // Writing to a `String` is infallible, so the result needs no handling.
    let _ = write_request(&mut user, request);
    user
}

/// Appends the rendered user message of `request` to `out`.
fn write_request(out: &mut String, request: &JudgeRequest<'_>) -> std::fmt::Result {
    let rubric = request.rubric;
    out.push_str("## Rubric\n\n");
    out.push_str(&render_rubric(rubric));

    out.push_str("\n## Subject under evaluation\n\n");
    out.push_str(request.subject.trim());
    out.push('\n');

    out.push_str("\n## Instructions\n\n");
    out.push_str(request.instructions.trim());
    out.push('\n');

    if let Some(reference) = request.reference {
        out.push_str("\n## Reference\n\n");
        out.push_str(reference.trim());
        out.push('\n');
    }

    out.push_str("\n## Required answer format\n\n");
    out.push_str("Answer with these sections, in this order, and nothing else:\n\n");
    out.push_str("<score>0.0</score>\n");
    out.push_str("<reasoning>why the subject earned this score</reasoning>\n");
    for dimension in &rubric.dimensions {
        let name = &dimension.name;
        writeln!(out, "<dimension name=\"{name}\">0.0</dimension>")?;
    }
    out.push_str(
        "\nA single ```json object with the keys `score`, `passed`, `reasoning`, and \
         `dimension_scores` (name to score) is equally acceptable.\n",
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::rubric::RubricDimension;

    fn rubric() -> Rubric {
        Rubric::new("skill quality", "Does the skill do its job?")
            .with_dimension(
                RubricDimension::new("clarity", "Is it clear?")
                    .with_weight(2.0)
                    .with_guidance("Prefer concrete examples."),
            )
            .with_dimension(RubricDimension::new("depth", "Is it deep?"))
            .with_scale(10.0)
    }

    fn request<'a>(
        rubric: &'a Rubric,
        subject: &'a str,
        instructions: &'a str,
    ) -> JudgeRequest<'a> {
        JudgeRequest::new(rubric, subject, instructions)
    }

    #[test]
    fn system_prompt_is_a_stable_constant() {
        assert_eq!(system_prompt(), system_prompt());
        assert!(system_prompt().starts_with("You are a strict evaluation judge."));
        assert!(!system_prompt().contains("{{"));
    }

    #[test]
    fn messages_have_a_system_and_a_user_message() {
        let rubric = rubric();
        let messages = rubric_messages(&request(&rubric, "subject", "instructions"));
        assert_eq!(messages.len(), 2);
        assert!(messages[0].role.is_system());
        assert!(messages[1].role.is_user());
        assert_eq!(messages[0].text().into_owned(), system_prompt());
    }

    #[test]
    fn render_rubric_lists_every_dimension_with_weight_and_guidance() {
        let rendered = render_rubric(&rubric());
        assert!(rendered.contains("Rubric: skill quality"), "{rendered}");
        assert!(
            rendered.contains("Does the skill do its job?"),
            "{rendered}"
        );
        assert!(
            rendered.contains("Scale: score every dimension"),
            "{rendered}"
        );
        assert!(rendered.contains("10.0"), "{rendered}");
        assert!(rendered.contains("1. clarity (weight 2.00)"), "{rendered}");
        assert!(rendered.contains("Is it clear?"), "{rendered}");
        assert!(
            rendered.contains("Guidance: Prefer concrete examples."),
            "{rendered}"
        );
        assert!(rendered.contains("2. depth (weight 1.00)"), "{rendered}");
        assert!(rendered.contains("total weight 3.00"), "{rendered}");
    }

    #[test]
    fn render_rubric_omits_blank_guidance() {
        let rubric = Rubric::uniform("demo", "A demo", ["one"]);
        let rendered = render_rubric(&rubric);
        assert!(!rendered.contains("Guidance:"), "{rendered}");
    }

    #[test]
    fn user_message_carries_subject_instructions_and_answer_format() {
        let rubric = rubric();
        let messages = rubric_messages(&request(
            &rubric,
            "  The subject text.  ",
            "  Judge strictly.  ",
        ));
        let user = messages[1].text().into_owned();
        assert!(user.contains("## Rubric"), "{user}");
        assert!(user.contains("## Subject under evaluation"), "{user}");
        assert!(user.contains("The subject text."), "{user}");
        assert!(user.contains("## Instructions"), "{user}");
        assert!(user.contains("Judge strictly."), "{user}");
        assert!(user.contains("<score>0.0</score>"), "{user}");
        assert!(user.contains("<reasoning>"), "{user}");
        assert!(
            user.contains("<dimension name=\"clarity\">0.0</dimension>"),
            "{user}"
        );
        assert!(
            user.contains("<dimension name=\"depth\">0.0</dimension>"),
            "{user}"
        );
        assert!(user.contains("json"), "{user}");
    }

    #[test]
    fn reference_section_appears_only_when_provided() {
        let rubric = rubric();
        let without = rubric_messages(&request(&rubric, "subject", "instructions"));
        assert!(!without[1].text().contains("## Reference"));

        let with = rubric_messages(
            &request(&rubric, "subject", "instructions").with_reference("  the exemplar  "),
        );
        let user = with[1].text().into_owned();
        assert!(user.contains("## Reference"), "{user}");
        assert!(user.contains("the exemplar"), "{user}");
        assert!(!user.contains("  the exemplar  "), "{user}");
    }

    #[test]
    fn rendering_never_leaks_template_placeholders() {
        let rubric = rubric();
        let messages = rubric_messages(
            &request(&rubric, "subject", "instructions").with_reference("reference"),
        );
        for message in &messages {
            let text = message.text().into_owned();
            assert!(!text.contains("{{"), "{text}");
            assert!(!text.contains("}}"), "{text}");
        }
    }

    #[test]
    fn rendering_is_stable_for_the_same_request() {
        let rubric = rubric();
        let first = rubric_messages(&request(&rubric, "subject", "instructions"));
        let second = rubric_messages(&request(&rubric, "subject", "instructions"));
        assert_eq!(first, second);
        assert_eq!(
            first[1].text().into_owned(),
            render_request(&request(&rubric, "subject", "instructions"))
        );
    }
}
