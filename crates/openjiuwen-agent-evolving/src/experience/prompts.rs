//! Prompt templates for the experience draft and maintenance pipeline.
//!
//! Ports the prompt bodies from Python
//! `openjiuwen.agent_evolving.optimizer/skill_call/templates.py` and
//! `experience/scorer.py`. The placeholders are documented on each constant and
//! filled by [`render_template`], which replaces them in one pass so a value
//! containing another placeholder is never re-expanded.

/// Replaces `{key}` placeholders in one pass, without re-scanning replacements.
///
/// A brace group that names no placeholder is copied through verbatim, so a
/// prompt may include literal JSON examples without escaping.
#[must_use]
pub(crate) fn render_template(template: &str, values: &[(&str, &str)]) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        rendered.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            rendered.push_str(&rest[open..]);
            return rendered;
        };
        let key = &after[..close];
        match values.iter().find(|(name, _)| *name == key) {
            Some((_, value)) => rendered.push_str(value),
            None => {
                rendered.push('{');
                rendered.push_str(key);
                rendered.push('}');
            }
        }
        rest = &after[close + 1..];
    }
    rendered.push_str(rest);
    rendered
}

/// The draft prompt. Its documented placeholders are `{skill_name}`,
/// `{skill_content}`, `{signals}`, `{trajectories}`, `{existing}` and
/// `{user_query}`.
///
/// The output contract is Python's: a bare JSON array whose objects carry
/// `action` (`append`/`skip`), `skip_reason`, `target`, `section`, `summary`,
/// `keywords`, `content`, `merge_target`, `script_filename`, `script_language`
/// and `script_purpose`.
pub const DRAFT_PROMPT: &str = r#"
You are a Skill optimization expert. Based on the problem signals and the execution evidence below, draft reusable evolution experiences for the Skill.

## Output format (most important)
Reply with a valid JSON array of objects and nothing else.
Do not wrap the array in a Markdown code fence and do not add explanatory prose.
Every object carries these keys:
- action: "append" to add an experience, "skip" to reject the evidence
- skip_reason: irrelevant | duplicate | low_priority (only for skip, otherwise null)
- target: description | body | script
- section: Instructions | Examples | Troubleshooting | Scripts | Collaboration | Roles | Constraints | Workflow
- summary: one line of at most 160 characters
- keywords: an array of retrieval keywords, or null
- content: the Markdown lesson; escape newlines as \n and quotes as \"
- merge_target: the id of an existing experience to rewrite, or null
- script_filename, script_language, script_purpose: only for script experiences, otherwise null

## Rules
- Only generate experiences supported by the signals or by the execution evidence.
- Respect the skill's role and primary objective; do not invent unrelated behaviour.
- At most two text experiences (description/body) and one script experience.
- Script experiences use target "script" and section "Scripts".
- Skip environment, permission and network flukes with skip_reason "irrelevant".
- Do not repeat an existing experience; rewrite it through merge_target when it is incomplete.

## Current skill name
{skill_name}

## Current skill content
{skill_content}

## Signals
{signals}

## Trajectories
{trajectories}

## Existing experiences
{existing}

## User-specified optimization direction
{user_query}
"#;

/// The one-shot repair prompt sent when the first answer is not parseable.
///
/// Its documented placeholders are `{error}` (the parse failure) and
/// `{previous}` (a bounded preview of the rejected reply). It asks for the same
/// array contract as [`DRAFT_PROMPT`].
pub const FORMAT_REPAIR_PROMPT: &str = r#"
Your previous reply could not be parsed as the required JSON array. Fix it and reply again.

## Parse error
{error}

## Previous reply (truncated)
{previous}

## Target format
Reply with only a JSON array, starting with [ and ending with ].
Every object carries: action (append or skip), skip_reason, target (description, body or script), section, summary, keywords, content, merge_target, script_filename, script_language, script_purpose.
Use double quotes only, escape newlines inside strings as \n, and add no prose or Markdown fence.
"#;

/// The maintenance prompt behind [`ExperienceDrafter::simplify`].
///
/// Its documented placeholders are `{skill_name}`, `{records}` and
/// `{user_intent}`. The answer contract is a bare JSON array of
/// `{action, record_id, reason, merge_remove_ids, new_content}` objects with
/// `action` one of `DELETE | MERGE | REFINE | KEEP`.
pub const SIMPLIFY_PROMPT: &str = r#"
You maintain an experience library. Based on the scores and usage below, propose maintenance actions.

## Skill name
{skill_name}

## Current experiences (highest score first)
{records}

## Actions
- DELETE: drop a low-quality or stale experience
- MERGE: fold several similar experiences into one
- REFINE: rewrite one experience's content
- KEEP: leave the experience unchanged

## Rules
1. Delete experiences scored below 0.4 that were never used.
2. Merge highly similar experiences, keeping the highest score as the target.
3. Refine vague or badly formatted experiences.
4. Keep high-quality, frequently used experiences.

## Output format
Reply with only a JSON array. Every object carries:
- action: DELETE | MERGE | REFINE | KEEP
- record_id: the target experience id
- reason: why the action was chosen
- merge_remove_ids: ids folded into the target (MERGE only)
- new_content: the replacement content (REFINE or MERGE only)

## User intent
{user_intent}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_are_replaced_once_and_literals_survive() {
        let rendered = render_template(
            "a={a} b={b} missing={c} json={\"k\": 1}",
            &[("a", "{b}"), ("b", "two")],
        );
        assert_eq!(rendered, "a={b} b=two missing={c} json={\"k\": 1}");
    }

    #[test]
    fn prompts_document_every_placeholder() {
        for key in [
            "skill_name",
            "skill_content",
            "signals",
            "trajectories",
            "existing",
            "user_query",
        ] {
            assert!(DRAFT_PROMPT.contains(&format!("{{{key}}}")), "{key}");
        }
        assert!(FORMAT_REPAIR_PROMPT.contains("{error}"));
        assert!(FORMAT_REPAIR_PROMPT.contains("{previous}"));
        for key in ["skill_name", "records", "user_intent"] {
            assert!(SIMPLIFY_PROMPT.contains(&format!("{{{key}}}")), "{key}");
        }
    }
}
