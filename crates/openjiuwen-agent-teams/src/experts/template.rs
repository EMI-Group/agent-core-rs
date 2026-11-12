//! Template → member composition and the read-only view of a bound `AgentGroup`.
//!
//! Python parity sources: `openjiuwen/agent_teams/experts/template_member.py` and
//! `openjiuwen/agent_teams/experts/bound_group.py`.
//!
//! A team member built from a template is two things:
//!
//! * a **persona** — the template's Markdown sections concatenated into the member's private
//!   prompt lane (see [`super::agent_group::LoadedAgentGroupPackage::load_member_persona`]);
//! * a **capability snapshot** — the template's skill directory names unioned into
//!   [`DeepAgentSpec::skills`], with skills/tools/rails/MCPs mounted by the member runtime.
//!
//! Python stores the snapshot in `DeepAgentSpec.agent_template_spec`; the Rust spec has no such
//! field, so the equivalent guarantee is that [`compose_template_member`] is idempotent: composing
//! the same template and persona twice returns the same spec rather than duplicating the persona
//! or the skill list.
//!
//! [`bind_group`] reads the group a team was assembled from back out of the package on disk: the
//! caller gets one [`BoundGroupMember`] per spawnable (non-leader) member, carrying exactly the
//! persona and launch kind that member needs.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::agent_group::{
    skill_basename, AgentTemplateCard, LoadedAgentGroupPackage, LEADER_MEMBER,
};
use crate::constants::DEFAULT_LANGUAGE;
use crate::schema::DeepAgentSpec;

/// One non-leader member of a bound `AgentGroup`, ready to be spawned as a teammate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct BoundGroupMember {
    /// Member name (`member_name` of the team roster).
    pub name: String,
    /// Display name in the package language, falling back to the manifest name and then `name`.
    pub display_name: String,
    /// Human-readable description in the package language.
    pub desc: String,
    /// Rendered persona text for the member's private prompt lane.
    pub persona: String,
    /// `true` when the member launches through an external CLI runtime.
    pub external_cli: bool,
}

/// The team's bound `AgentGroup`: its name and spawnable member templates.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct BoundAgentGroup {
    /// `AgentGroup` package name.
    pub name: String,
    /// Spawnable members, ordered by member name.
    pub members: Vec<BoundGroupMember>,
}

impl BoundAgentGroup {
    /// Builds an empty bound group for `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            members: Vec::new(),
        }
    }

    /// One member by name.
    #[must_use]
    pub fn member(&self, name: &str) -> Option<&BoundGroupMember> {
        self.members.iter().find(|member| member.name == name)
    }

    /// Member names in roster order.
    #[must_use]
    pub fn member_names(&self) -> Vec<&str> {
        self.members
            .iter()
            .map(|member| member.name.as_str())
            .collect()
    }
}

/// Binds a loaded package to its spawnable member descriptions.
///
/// The leader is never spawnable as a teammate and is skipped. Members whose persona cannot be
/// read from the package (the directory was removed or edited after loading) or is empty are
/// skipped with a warning rather than failing the whole binding, so one broken member does not
/// hide the rest.
#[must_use]
pub fn bind_group(package: &LoadedAgentGroupPackage) -> BoundAgentGroup {
    let mut bound = BoundAgentGroup::new(package.manifest.name.clone());
    for (name, card) in &package.templates {
        if name == LEADER_MEMBER {
            continue;
        }
        let persona = match package.load_member_persona(name) {
            Ok(persona) if !persona.trim().is_empty() => persona,
            Ok(_) => {
                tracing::debug!(member = %name, "agent group member has an empty persona; skipping");
                continue;
            }
            Err(error) => {
                tracing::warn!(member = %name, %error, "agent group member persona unavailable; skipping");
                continue;
            }
        };
        let display_name = card.display_label(DEFAULT_LANGUAGE, name);
        bound.members.push(BoundGroupMember {
            name: name.clone(),
            display_name: display_name.to_owned(),
            desc: card.description.resolve(DEFAULT_LANGUAGE).to_owned(),
            persona,
            external_cli: card.is_external(),
        });
    }
    bound
}

/// Assembles a template on a base spec: rendered persona plus the capability snapshot.
///
/// The returned spec is `base` with:
///
/// * `system_prompt` — the rendered persona, prepended to any existing prompt with one blank line
///   between them (an existing prompt that already starts with this persona is left untouched, so
///   composing twice is idempotent);
/// * `skills` — the union of the base skill names and the template's skill directory basenames,
///   deduplicated and sorted;
/// * `language` — the resolved language;
/// * every other field preserved.
///
/// Language precedence is `language`, then `base.language`, then [`DEFAULT_LANGUAGE`].
#[must_use]
pub fn compose_template_member(
    base: &DeepAgentSpec,
    template: &AgentTemplateCard,
    persona: &str,
    language: Option<&str>,
) -> DeepAgentSpec {
    let resolved_language = resolve_language(language, base.language.as_deref());
    let workspace = base
        .workspace
        .as_ref()
        .map_or("", |workspace| workspace.root_path.as_str());
    let rendered = render_persona(persona, resolved_language, workspace);

    let mut spec = base.clone();
    spec.system_prompt = merge_system_prompt(base.system_prompt.as_deref(), &rendered);
    spec.skills = merged_skills(base.skills.as_deref(), &template.skills);
    spec.language = Some(resolved_language.to_owned());
    spec
}

/// Renders a persona's template placeholders in one pass.
///
/// `{{language}}` and `{{workspace}}` are replaced with their values, and `{{persona}}` expands
/// once to the unrendered persona text (a package may use it to nest the raw text in a wrapper).
/// Replacements do not re-scan inserted content, so a persona is never expanded recursively.
#[must_use]
pub fn render_persona(persona: &str, language: &str, workspace: &str) -> String {
    persona
        .replace("{{language}}", language)
        .replace("{{workspace}}", workspace)
        .replace("{{persona}}", persona)
}

fn resolve_language<'a>(argument: Option<&'a str>, base: Option<&'a str>) -> &'a str {
    for candidate in [argument, base] {
        let Some(text) = candidate else {
            continue;
        };
        let text = text.trim();
        if !text.is_empty() {
            return text;
        }
    }
    DEFAULT_LANGUAGE
}

fn merge_system_prompt(existing: Option<&str>, rendered: &str) -> Option<String> {
    match existing {
        Some(current) if !current.trim().is_empty() => {
            if rendered.is_empty() || already_prefixed(current, rendered) {
                Some(current.to_owned())
            } else {
                Some(format!("{rendered}\n\n{current}"))
            }
        }
        _ => (!rendered.is_empty()).then(|| rendered.to_owned()),
    }
}

fn already_prefixed(existing: &str, rendered: &str) -> bool {
    existing == rendered
        || existing
            .strip_prefix(rendered)
            .is_some_and(|rest| rest.starts_with('\n'))
}

fn merged_skills(base: Option<&[String]>, template: &[String]) -> Option<Vec<String>> {
    let mut names = BTreeSet::new();
    for value in base.into_iter().flatten().chain(template) {
        let name = skill_basename(value);
        if !name.is_empty() {
            names.insert(name);
        }
    }
    if names.is_empty() {
        base.map(<[String]>::to_vec)
    } else {
        Some(names.into_iter().collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::schema::{ModelRef, WorkspaceSpec};

    fn template(name: &str, skills: &[&str]) -> AgentTemplateCard {
        AgentTemplateCard {
            name: name.to_owned(),
            skills: skills.iter().map(|skill| (*skill).to_owned()).collect(),
            ..AgentTemplateCard::default()
        }
    }

    fn base_spec() -> DeepAgentSpec {
        DeepAgentSpec {
            model: Some(ModelRef {
                name: None,
                model_name: Some("gpt-4o".to_owned()),
                provider: Some("openai".to_owned()),
                params: BTreeMap::new(),
            }),
            system_prompt: Some("Existing instructions.".to_owned()),
            skills: Some(vec!["shared".to_owned(), "a".to_owned()]),
            language: Some("en".to_owned()),
            workspace: Some(WorkspaceSpec {
                root_path: "/workspace/member".to_owned(),
                language: "en".to_owned(),
                stable_base: false,
            }),
            ..DeepAgentSpec::default()
        }
    }

    #[test]
    fn language_precedence_is_argument_then_spec_then_default() {
        let base = base_spec();
        let spec = compose_template_member(&base, &template("m", &[]), "", Some("cf"));
        assert_eq!(spec.language.as_deref(), Some("cf"));

        let spec = compose_template_member(&base, &template("m", &[]), "", None);
        assert_eq!(spec.language.as_deref(), Some("en"));

        let mut no_language = base;
        no_language.language = None;
        let spec = compose_template_member(&no_language, &template("m", &[]), "", None);
        assert_eq!(spec.language.as_deref(), Some(DEFAULT_LANGUAGE));

        let blank = compose_template_member(&no_language, &template("m", &[]), "", Some("  "));
        assert_eq!(blank.language.as_deref(), Some(DEFAULT_LANGUAGE));
    }

    #[test]
    fn persona_placeholders_are_rendered_once() {
        let persona = "speak {{language}} in {{workspace}}; raw={{persona}}";
        let rendered = render_persona(persona, "en", "/work");
        assert_eq!(
            rendered,
            "speak en in /work; raw=speak {{language}} in {{workspace}}; raw={{persona}}"
        );
    }

    #[test]
    fn composes_persona_prompt_and_skills_without_touching_other_fields() {
        let base = base_spec();
        let template = template("writer", &["skills/review", "a", "review"]);
        let spec = compose_template_member(
            &base,
            &template,
            "# Writer\nUse {{language}} in {{workspace}}.",
            Some("en"),
        );

        assert_eq!(
            spec.system_prompt.as_deref(),
            Some("# Writer\nUse en in /workspace/member.\n\nExisting instructions.")
        );
        assert_eq!(
            spec.skills,
            Some(vec![
                "a".to_owned(),
                "review".to_owned(),
                "shared".to_owned()
            ])
        );
        assert_eq!(spec.language.as_deref(), Some("en"));
        assert_eq!(spec.model, base.model);
        assert_eq!(spec.workspace, base.workspace);
        assert_eq!(spec.completion_timeout, base.completion_timeout);
    }

    #[test]
    fn compose_is_idempotent() {
        let base = base_spec();
        let template = template("writer", &["skills/review"]);
        let once = compose_template_member(&base, &template, "# Writer", Some("en"));
        let twice = compose_template_member(&once, &template, "# Writer", Some("en"));
        assert_eq!(once, twice);
    }

    #[test]
    fn bind_group_skips_the_leader_and_marks_external_members() {
        let temp = tempfile::tempdir().unwrap();
        let package = crate::experts::test_support::minimal_group_package(temp.path());
        let bound = bind_group(&package);

        assert_eq!(bound.name, "group");
        assert_eq!(bound.member_names(), vec!["member1", "member2"]);
        assert!(bound.member("leader").is_none());

        let member1 = bound.member("member1").unwrap();
        assert_eq!(member1.display_name, "member1 display");
        assert_eq!(member1.desc, "member1 description");
        assert!(member1.persona.contains("# member1"));
        assert!(member1.external_cli);

        let member2 = bound.member("member2").unwrap();
        assert!(!member2.external_cli);
    }
}
