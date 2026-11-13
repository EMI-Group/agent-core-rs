//! Prompt rendering and query selection over a set of skills.
//!
//! Ports the agent-facing half of
//! `openjiuwen/core/single_agent/skills/skill_util.py`: Python renders a fixed
//! instructional header plus one `"<i>.Skill name: …"` line per skill;
//! [`SkillUtil::render_prompt`] renders a Markdown `## Skills` section under a
//! character budget instead, so a large skill library can never crowd out the
//! conversation.
use super::skill::Skill;

/// Heading of the rendered skill prompt.
pub const SKILLS_HEADING: &str = "## Skills";

/// Marker appended when the rendered prompt exceeds its budget.
pub const TRUNCATION_MARKER: &str = "…";

/// Rendering and selection helpers over a slice of [`Skill`]s.
///
/// The type is a namespace: every method is an associated function, so
/// `SkillUtil::render_prompt(&skills, 2_000)` reads like the Python class.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SkillUtil;

impl SkillUtil {
    /// Renders one catalog line per skill, in input order.
    ///
    /// A line is `- name: description`, followed by ` [keyword, …]` when the
    /// skill declares keywords.
    #[must_use]
    pub fn render_catalog(skills: &[Skill]) -> String {
        skills
            .iter()
            .map(catalog_line)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Renders the `## Skills` prompt section, truncated at a skill boundary
    /// when it exceeds `max_chars` characters.
    ///
    /// The section is the heading, a blank line, and one catalog line per
    /// skill. Lines are added while the whole section — including the
    /// [`TRUNCATION_MARKER`] line that follows the last added one — still fits
    /// in `max_chars`; the rest are dropped and the marker is appended. The
    /// heading and the marker are always emitted, so a budget smaller than
    /// them still yields a marked section. An empty `skills` slice renders an
    /// empty string.
    #[must_use]
    pub fn render_prompt(skills: &[Skill], max_chars: usize) -> String {
        if skills.is_empty() {
            return String::new();
        }
        let lines: Vec<String> = skills.iter().map(catalog_line).collect();
        let mut rendered = String::from(SKILLS_HEADING);
        let mut used = char_len(SKILLS_HEADING);
        let mut emitted = 0_usize;
        let mut truncated = false;
        for (index, line) in lines.iter().enumerate() {
            let separator = if index == 0 { 2 } else { 1 };
            let reserve = if index + 1 < lines.len() {
                1 + char_len(TRUNCATION_MARKER)
            } else {
                0
            };
            let cost = separator + char_len(line);
            if used + cost + reserve <= max_chars {
                if index == 0 {
                    rendered.push_str("\n\n");
                } else {
                    rendered.push('\n');
                }
                rendered.push_str(line);
                used += cost;
                emitted += 1;
            } else {
                truncated = true;
                break;
            }
        }
        if truncated {
            if emitted == 0 {
                rendered.push_str("\n\n");
            } else {
                rendered.push('\n');
            }
            rendered.push_str(TRUNCATION_MARKER);
        }
        rendered
    }

    /// Returns the skills matching `query`, or every skill when none matches.
    ///
    /// Matching uses [`Skill::matches`]; the fallback keeps an unrelated query
    /// from hiding the whole library from the model.
    #[must_use]
    pub fn select(skills: &[Skill], query: &str) -> Vec<Skill> {
        let matched: Vec<Skill> = skills
            .iter()
            .filter(|skill| skill.matches(query))
            .cloned()
            .collect();
        if matched.is_empty() {
            skills.to_vec()
        } else {
            matched
        }
    }
}

/// Renders the catalog line of one skill.
fn catalog_line(skill: &Skill) -> String {
    let mut line = format!("- {}: {}", skill.name.trim(), skill.description.trim());
    let keywords: Vec<&str> = skill
        .keywords
        .iter()
        .map(|keyword| keyword.trim())
        .filter(|keyword| !keyword.is_empty())
        .collect();
    if !keywords.is_empty() {
        line.push_str(" [");
        line.push_str(&keywords.join(", "));
        line.push(']');
    }
    line.trim_end().to_owned()
}

/// Counts the characters (not bytes) of `text`.
fn char_len(text: &str) -> usize {
    text.chars().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skills() -> Vec<Skill> {
        vec![
            Skill::new("alpha", "Alpha skill").with_keywords(["a", "first"]),
            Skill::new("beta", "Beta skill"),
            Skill::new("gamma", "Gamma skill"),
        ]
    }

    #[test]
    fn catalog_lists_one_line_per_skill() {
        assert_eq!(
            SkillUtil::render_catalog(&skills()),
            "- alpha: Alpha skill [a, first]\n- beta: Beta skill\n- gamma: Gamma skill"
        );
        assert_eq!(SkillUtil::render_catalog(&[]), "");
        assert_eq!(
            SkillUtil::render_catalog(&[Skill::new("bare", "  ")]),
            "- bare:"
        );
    }

    #[test]
    fn prompt_has_the_heading_then_the_catalog() {
        assert_eq!(
            SkillUtil::render_prompt(&skills(), 10_000),
            "## Skills\n\n\
             - alpha: Alpha skill [a, first]\n\
             - beta: Beta skill\n- gamma: Gamma skill"
        );
        assert_eq!(SkillUtil::render_prompt(&[], 10_000), "");
    }

    #[test]
    fn prompt_truncates_at_a_skill_boundary_and_marks_it() {
        // "## Skills" (9) + "\n\n" (2) + alpha line (31) + "\n…" (2) = 44.
        let budget = 44;
        let rendered = SkillUtil::render_prompt(&skills(), budget);
        assert_eq!(rendered, "## Skills\n\n- alpha: Alpha skill [a, first]\n…");
        assert!(rendered.chars().count() <= budget);
        assert!(!rendered.contains("beta"));
        assert!(!rendered.contains("gamma"));
    }

    #[test]
    fn prompt_keeps_every_line_when_the_budget_allows_it() {
        let rendered = SkillUtil::render_prompt(&skills(), 200);
        assert!(!rendered.contains(TRUNCATION_MARKER));
        assert!(rendered.contains("gamma"));
    }

    #[test]
    fn prompt_marks_a_tiny_budget_without_any_skill() {
        assert_eq!(
            SkillUtil::render_prompt(&skills(), 0),
            format!("{SKILLS_HEADING}\n\n{TRUNCATION_MARKER}")
        );
    }

    #[test]
    fn select_matches_and_falls_back_to_everything() {
        let selected = SkillUtil::select(&skills(), "beta");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "beta");
        let fallback = SkillUtil::select(&skills(), "nothing matches");
        assert_eq!(fallback.len(), 3);
        let empty_query = SkillUtil::select(&skills(), "");
        assert_eq!(empty_query.len(), 3);
    }
}
