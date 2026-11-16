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
//!
//! Binding never drops a member silently: every roster member that does not become a
//! [`BoundGroupMember`] is recorded in [`BoundAgentGroup::skipped`] as a [`SkippedMember`] naming
//! the member, a human-readable reason and a [`SkipReason`], and is logged with the package name.
//! The leader is always skipped by design (it is not spawnable); a member whose persona is empty
//! or no longer readable is skipped deliberately so one broken member does not hide the rest.

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

/// Why [`bind_group`] left an `AgentGroup` roster member out of the bound teammates.
///
/// [`SkipReason::Leader`] is the expected filter applied to every package; the persona variants
/// report a member the package on disk can no longer back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    /// The member is the group leader, which is never spawnable as a teammate.
    #[default]
    Leader,
    /// The member's persona is present but empty.
    EmptyPersona,
    /// The member's persona can no longer be read from the package on disk.
    PersonaUnreadable,
}

impl SkipReason {
    /// Stable wire name of this reason: `leader`, `empty_persona` or `persona_unreadable`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Leader => "leader",
            Self::EmptyPersona => "empty_persona",
            Self::PersonaUnreadable => "persona_unreadable",
        }
    }
}

/// One roster member [`bind_group`] did not bind as a spawnable teammate.
///
/// The entry exists so binding is never lossy in silence: `name` identifies the member, `reason`
/// is a human-readable sentence that names the member too, and `kind` is the same fact as data.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SkippedMember {
    /// Member name as it appears in the package roster.
    pub name: String,
    /// Human-readable reason for the skip, naming the member and the failing condition.
    pub reason: String,
    /// Classification of `reason` for programmatic handling.
    pub kind: SkipReason,
}

/// The team's bound `AgentGroup`: its name, spawnable member templates and skipped members.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct BoundAgentGroup {
    /// `AgentGroup` package name.
    pub name: String,
    /// Spawnable members, ordered by member name.
    pub members: Vec<BoundGroupMember>,
    /// Roster members that were not bound, ordered by member name.
    ///
    /// The leader is always recorded here (see [`SkipReason::Leader`]); a member whose persona is
    /// empty or unreadable appears with the matching [`SkipReason`] instead of vanishing.
    pub skipped: Vec<SkippedMember>,
}

impl BoundAgentGroup {
    /// Builds an empty bound group for `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            members: Vec::new(),
            skipped: Vec::new(),
        }
    }

    /// `true` when every non-leader roster member bound as a spawnable teammate.
    ///
    /// The leader is skipped by design on every package and is not counted.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.skipped
            .iter()
            .all(|skipped| skipped.kind == SkipReason::Leader)
    }

    /// Names of the skipped roster members, in member-name order.
    #[must_use]
    pub fn skipped_names(&self) -> Vec<&str> {
        self.skipped
            .iter()
            .map(|skipped| skipped.name.as_str())
            .collect()
    }

    /// The reason a roster member was skipped, or `None` when it bound as a teammate.
    #[must_use]
    pub fn skipped_reason(&self, name: &str) -> Option<&str> {
        self.skipped
            .iter()
            .find(|skipped| skipped.name == name)
            .map(|skipped| skipped.reason.as_str())
    }

    fn record_skip(&mut self, name: &str, kind: SkipReason, reason: impl Into<String>) {
        self.skipped.push(SkippedMember {
            name: name.to_owned(),
            reason: reason.into(),
            kind,
        });
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
///
/// Every skip is observable: it is logged with the package and member identity and appended to
/// [`BoundAgentGroup::skipped`], so a caller can tell "the package has only the leader" from "two
/// members were dropped".
#[must_use]
pub fn bind_group(package: &LoadedAgentGroupPackage) -> BoundAgentGroup {
    let mut bound = BoundAgentGroup::new(package.manifest.name.clone());
    for (name, card) in &package.templates {
        if name == LEADER_MEMBER {
            tracing::debug!(
                package = %bound.name,
                member = %name,
                "agent group leader is not spawnable as a teammate; skipping"
            );
            bound.record_skip(
                name,
                SkipReason::Leader,
                format!(
                    "member {name:?} is the agent group leader, which is not spawnable as a teammate"
                ),
            );
            continue;
        }
        let persona = match package.load_member_persona(name) {
            Ok(persona) if !persona.trim().is_empty() => persona,
            Ok(_) => {
                tracing::warn!(
                    package = %bound.name,
                    member = %name,
                    "agent group member has an empty persona; skipping"
                );
                bound.record_skip(
                    name,
                    SkipReason::EmptyPersona,
                    format!("member {name:?} has an empty persona"),
                );
                continue;
            }
            Err(error) => {
                tracing::warn!(
                    package = %bound.name,
                    member = %name,
                    %error,
                    "agent group member persona unavailable; skipping"
                );
                bound.record_skip(
                    name,
                    SkipReason::PersonaUnreadable,
                    format!("member {name:?} persona is unreadable: {error}"),
                );
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

    #[test]
    fn bind_group_binds_every_member_and_records_only_the_leader_skip() {
        let temp = tempfile::tempdir().unwrap();
        let package = crate::experts::test_support::minimal_group_package(temp.path());
        let bound = bind_group(&package);

        assert_eq!(bound.member_names(), vec!["member1", "member2"]);
        assert_eq!(bound.skipped_names(), vec!["leader"]);

        let leader = &bound.skipped[0];
        assert_eq!(leader.name, "leader");
        assert_eq!(leader.kind, SkipReason::Leader);
        assert_eq!(leader.kind.as_str(), "leader");
        assert!(leader.reason.contains("leader"), "{}", leader.reason);
        assert!(leader.reason.contains("not spawnable"), "{}", leader.reason);

        assert!(bound.is_complete());
        assert!(bound
            .skipped_reason("leader")
            .is_some_and(|reason| reason.contains("leader")));
        assert_eq!(bound.skipped_reason("member1"), None);
    }

    #[test]
    fn bind_group_surfaces_a_member_whose_persona_is_empty() {
        let temp = tempfile::tempdir().unwrap();
        let package = crate::experts::test_support::minimal_group_package(temp.path());
        let persona = package
            .package_dir
            .join("agents")
            .join("member2")
            .join("persona")
            .join("main.md");
        std::fs::write(&persona, "  \n").unwrap();

        let bound = bind_group(&package);

        assert_eq!(bound.member_names(), vec!["member1"]);
        assert_eq!(bound.skipped_names(), vec!["leader", "member2"]);
        assert!(!bound.is_complete());

        let skipped = bound
            .skipped
            .iter()
            .find(|skipped| skipped.name == "member2")
            .unwrap();
        assert_eq!(skipped.kind, SkipReason::EmptyPersona);
        assert_eq!(skipped.kind.as_str(), "empty_persona");
        assert!(skipped.reason.contains("member2"), "{}", skipped.reason);
        assert!(
            skipped.reason.contains("empty persona"),
            "{}",
            skipped.reason
        );
        assert_eq!(
            bound.skipped_reason("member2"),
            Some(skipped.reason.as_str())
        );
    }

    #[test]
    fn bind_group_surfaces_a_member_whose_persona_is_unreadable() {
        let temp = tempfile::tempdir().unwrap();
        let package = crate::experts::test_support::minimal_group_package(temp.path());
        std::fs::remove_dir_all(package.package_dir.join("agents").join("member2")).unwrap();

        let bound = bind_group(&package);

        assert_eq!(bound.member_names(), vec!["member1"]);
        assert!(!bound.is_complete());
        let skipped = bound
            .skipped
            .iter()
            .find(|skipped| skipped.name == "member2")
            .unwrap();
        assert_eq!(skipped.kind, SkipReason::PersonaUnreadable);
        assert_eq!(skipped.kind.as_str(), "persona_unreadable");
        assert!(skipped.reason.contains("member2"), "{}", skipped.reason);
        assert!(skipped.reason.contains("unreadable"), "{}", skipped.reason);
        assert!(
            skipped.reason.contains("agent directory not found"),
            "{}",
            skipped.reason
        );
    }

    #[test]
    fn skipped_members_round_trip_and_older_payloads_still_load() {
        let bound = BoundAgentGroup {
            name: "group".to_owned(),
            members: Vec::new(),
            skipped: vec![SkippedMember {
                name: "member2".to_owned(),
                reason: "member \"member2\" has an empty persona".to_owned(),
                kind: SkipReason::PersonaUnreadable,
            }],
        };

        let json = serde_json::to_value(&bound).unwrap();
        assert_eq!(json["skipped"][0]["name"], "member2");
        assert_eq!(json["skipped"][0]["kind"], "persona_unreadable");
        let back: BoundAgentGroup = serde_json::from_value(json).unwrap();
        assert_eq!(back, bound);

        let legacy: BoundAgentGroup = serde_json::from_value(serde_json::json!({
            "name": "group",
            "members": [],
            "a_field_from_the_future": true,
        }))
        .unwrap();
        assert_eq!(legacy.skipped, Vec::<SkippedMember>::new());
        assert!(legacy.is_complete());
        assert!(legacy.skipped_reason("member2").is_none());
    }
}
