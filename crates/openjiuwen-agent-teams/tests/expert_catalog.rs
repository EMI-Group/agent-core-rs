//! Expert catalog, `AgentGroup` package loading and template composition, through the public API.
//!
//! The suite pins six contracts a host — and the leader's expert tools — rely on:
//!
//! 1. `load_agent_group` accepts a well-formed JSON package (roster order does not matter, unknown
//!    manifest keys are tolerated) and exposes its cards, skills, connectors and personas;
//! 2. malformed packages are rejected with a validation error naming the offending path or entry,
//!    while an unknown member name is a not-found error;
//! 3. `bind_group` describes the spawnable members only, deterministically, and two packages loaded
//!    from different directories never mix;
//! 4. `compose_template_member` layers the rendered persona and the template's skills onto a base
//!    blueprint, preserving everything the template is silent about;
//! 5. the process-wide catalog slot round-trips the host's catalog and clears back to `None`;
//! 6. a blueprint whose teammate is a composed template builds a team whose stored member row
//!    carries the composed prompt.
//!
//! Everything runs offline against packages written into a `tempfile::TempDir`; no fixture lives
//! outside the temporary directory and nothing sleeps.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::AgentCard;
use openjiuwen_agent_teams::experts::{
    bind_group, clear_process_expert_catalog, compose_template_member, get_process_expert_catalog,
    load_agent_group, set_process_expert_catalog, AgentGroupMemberCard, AgentTemplateCard,
    ExpertCatalog, ExpertRef, LoadedAgentGroupPackage, PackageLocation,
};
use openjiuwen_agent_teams::runtime::SpecBuild;
use openjiuwen_agent_teams::schema::{
    DeepAgentSpec, MemberStatus, ModelRef, TeamRole, WorkspaceSpec,
};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen_common::error::{ErrorKind, Result};
use serde_json::{json, Map, Value};

use support::{build_deps, sample_spec, FakeHarnessFactory, FakeWorkspaceProvider};

/// Group name of the fixture package, which must equal its directory name.
const GROUP: &str = "writing";

/// Writes `payload` as pretty JSON, creating every missing parent directory.
fn write_json(path: &Path, payload: &Value) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, serde_json::to_string_pretty(payload).unwrap()).unwrap();
}

/// Writes `text`, creating every missing parent directory.
fn write_text(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, text).unwrap();
}

/// Applies `mutate` to the JSON object stored at `path` and writes it back.
fn patch_json(path: &Path, mutate: impl FnOnce(&mut Map<String, Value>)) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut value: Value = serde_json::from_str(&text).unwrap();
    mutate(value.as_object_mut().unwrap());
    std::fs::write(path, serde_json::to_string(&value).unwrap()).unwrap();
}

/// The persona text `load_agent_group` produces for a fixture member.
///
/// Three sections are concatenated in resolved-path order (`01_intro`, `02_rules`,
/// `extra/10_more`) and joined with a blank line; the sibling `notes.txt` is not Markdown and is
/// skipped. Front matter and placeholders are preserved verbatim at load time.
fn raw_persona(member: &str) -> String {
    format!(
        "# {member} intro\n\n# {member} rules\nUse {{{{language}}}} and {{{{workspace}}}}.\n\n# More"
    )
}

/// Writes `agents/<member>/manifest.json` plus its persona directory.
///
/// Every non-leader member declares one private skill (`narrative`), two tags, one connector and
/// a persona with three Markdown sections, one of them nested.
fn write_member(group: &Path, member: &str) {
    let dir = group.join("agents").join(member);
    let (description, display_name) = if member == "leader" {
        (
            json!({"cn": "带队", "en": "leads the crew"}),
            json!({"cn": "队长", "en": "Crew Lead"}),
        )
    } else {
        (
            json!({"cn": "负责起草", "en": "writes the drafts"}),
            json!({"cn": "写手", "en": "Writer"}),
        )
    };
    let mut manifest = json!({
        "package_type": "agent_template",
        "name": format!("{member} display"),
        "description": description,
        "display_name": display_name,
        "tags": ["prose", {"id": "copy"}],
        "connectors": ["web_search"],
        "persona": {"dir": "./persona"},
    });
    if member != "leader" {
        manifest["skills"] = json!(["narrative"]);
    }
    write_json(&dir.join("manifest.json"), &manifest);
    write_text(
        &dir.join("persona/01_intro.md"),
        &format!("# {member} intro"),
    );
    write_text(
        &dir.join("persona/02_rules.md"),
        &format!("# {member} rules\nUse {{{{language}}}} and {{{{workspace}}}}."),
    );
    write_text(&dir.join("persona/extra/10_more.md"), "# More");
    write_text(&dir.join("persona/notes.txt"), "ignored");
}

/// Writes a complete, valid `AgentGroup` package named `name` under `root` and returns its path.
///
/// The manifest declares `capabilities` (one localized object), one shared skill (`research`) that
/// is also auto-discovered next to `auto_skill`, and — deliberately — a roster that is not in
/// alphabetical order.
fn write_group(root: &Path, name: &str, roster: &[&str]) -> PathBuf {
    let group = root.join(name);
    write_json(
        &group.join("manifest.json"),
        &json!({
            "name": name,
            "package_type": "agent_group",
            "agents": roster,
            "description": {"cn": "写作团队", "en": "Writing team"},
            "display_name": {"cn": "写作组", "en": "Writing Crew"},
            "category": {"cn": "内容", "en": "content"},
            "version": "1.2.0",
            "instruction": "  keep the tone consistent  ",
            "capabilities": ["writing", {"id": "editing"}, "writing"],
            "skills": ["research"],
        }),
    );
    write_text(&group.join("skills/research/SKILL.md"), "# research\n");
    write_text(&group.join("skills/auto_skill/SKILL.md"), "# auto\n");
    for member in roster {
        write_member(&group, member);
    }
    write_text(&group.join("agents/leader/AGENT.md"), "# leader rules\n");
    group
}

/// Borrows a string list as `&str`s so assertions read without `to_owned`.
fn texts(values: &[String]) -> Vec<&str> {
    values.iter().map(String::as_str).collect()
}

/// Asserts that `dir` is rejected as a validation error mentioning `needle`.
///
/// `load_agent_group` reports every malformed package — including a missing directory — as
/// [`ErrorKind::Validation`] naming the offending path; not-found is reserved for an unknown
/// member name on an already loaded package.
#[track_caller]
fn assert_rejected(dir: &Path, needle: &str) {
    let error = load_agent_group(dir).expect_err("the package must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation, "for: {error}");
    assert!(
        error.to_string().contains(needle),
        "expected {needle:?} in: {error}"
    );
}

/// A hand-written [`ExpertCatalog`]: the crate ships the trait but no implementation, so a host's
/// catalog is modelled here over two fixed location lists.
struct FakeCatalog {
    /// Installed single-expert packages.
    experts: Vec<PackageLocation>,
    /// Installed `AgentGroup` packages.
    groups: Vec<PackageLocation>,
}

#[async_trait]
impl ExpertCatalog for FakeCatalog {
    async fn list_expert_packages(&self) -> Result<Vec<PackageLocation>> {
        Ok(self.experts.clone())
    }

    async fn locate_expert(&self, name: &str) -> Result<Option<PackageLocation>> {
        Ok(self.experts.iter().find(|item| item.name == name).cloned())
    }

    async fn list_group_packages(&self) -> Result<Vec<PackageLocation>> {
        Ok(self.groups.clone())
    }

    async fn locate_group(&self, name: &str) -> Result<Option<PackageLocation>> {
        Ok(self.groups.iter().find(|item| item.name == name).cloned())
    }
}

/// Clears the process-wide expert catalog on drop, so a panicking test cannot leak the singleton.
struct ClearCatalogOnDrop;

impl Drop for ClearCatalogOnDrop {
    fn drop(&mut self) {
        clear_process_expert_catalog();
    }
}

#[test]
fn loads_a_group_package_with_its_cards_skills_and_personas() {
    let temp = tempfile::tempdir().unwrap();
    // Roster order is `writer` first: the loader must key templates by name regardless.
    let dir = write_group(temp.path(), GROUP, &["writer", "leader"]);
    let package: LoadedAgentGroupPackage =
        load_agent_group(&dir).expect("the fixture package is valid");

    assert_eq!(package.package_dir, dir.canonicalize().unwrap());
    assert_eq!(package.manifest.name, GROUP);
    assert_eq!(package.manifest.description.resolve("en"), "Writing team");
    // `zh` is a `cn` synonym, so the Chinese text resolves through either request.
    assert_eq!(package.manifest.description.resolve("zh"), "写作团队");
    assert_eq!(package.manifest.display_label("en"), "Writing Crew");
    assert_eq!(package.manifest.display_label("cf"), "写作组");
    assert_eq!(package.manifest.category.as_deref(), Some("内容"));
    assert_eq!(package.manifest.version.as_deref(), Some("1.2.0"));
    assert_eq!(package.instruction(), "keep the tone consistent");
    assert_eq!(
        texts(package.capabilities()),
        vec!["writing", "editing"],
        "capabilities are deduplicated in declaration order"
    );

    // Templates are keyed (and therefore ordered) by member name, not roster order.
    assert_eq!(
        package.templates.keys().cloned().collect::<Vec<_>>(),
        vec!["leader".to_owned(), "writer".to_owned()]
    );
    let writer = package
        .template("writer")
        .expect("the writer template loaded");
    assert_eq!(writer.name, "writer display");
    assert_eq!(writer.description.resolve("en"), "writes the drafts");
    assert_eq!(
        writer.display_name.as_ref().map(|text| text.resolve("cn")),
        Some("写手")
    );
    assert_eq!(texts(&writer.tags), vec!["prose", "copy"]);
    assert_eq!(texts(&writer.connectors), vec!["web_search"]);
    // The private skill comes first, then the group's declared and auto-discovered skills.
    assert_eq!(
        texts(&writer.skills),
        vec!["narrative", "research", "auto_skill"]
    );
    assert_eq!(writer.runtime, "native");
    assert!(!writer.is_external());
    // A leader carries no private skill, only the shared ones; its `AGENT.md` is optional.
    let leader = package
        .template("leader")
        .expect("the leader template loaded");
    assert_eq!(texts(&leader.skills), vec!["research", "auto_skill"]);

    assert_eq!(
        package.load_member_persona("writer").unwrap(),
        raw_persona("writer")
    );

    // Member cards carry the `(group, name)` identity the expert tools address members by.
    let cards = package.member_cards();
    assert_eq!(
        texts(
            &cards
                .iter()
                .map(|card| card.name.clone())
                .collect::<Vec<_>>()
        ),
        vec!["leader", "writer"]
    );
    let writer_card = cards.iter().find(|card| card.name == "writer").unwrap();
    assert_eq!(writer_card.group, GROUP);
    assert_eq!(
        writer_card.expert_ref(),
        ExpertRef::new("writer").with_group(GROUP)
    );
    assert_eq!(writer_card.expert_ref().label(), "writing/writer");
    assert_eq!(writer_card.display_label("en"), "Writer");
    assert!(!writer_card.is_leader());
    assert!(cards.iter().any(AgentGroupMemberCard::is_leader));
}

#[test]
fn tolerates_unknown_manifest_keys_and_never_interprets_member_capabilities() {
    let temp = tempfile::tempdir().unwrap();
    let dir = write_group(temp.path(), GROUP, &["leader", "writer"]);
    patch_json(&dir.join("manifest.json"), |manifest| {
        manifest.insert("unknown_group_knob".to_owned(), json!({"nested": [1, 2]}));
        manifest.insert("tags".to_owned(), json!(["from_tags"]));
        manifest.insert("display_description".to_owned(), json!("unused"));
    });
    patch_json(&dir.join("agents/writer/manifest.json"), |manifest| {
        manifest.insert("tools".to_owned(), json!([{"type": "bash"}]));
        manifest.insert("model".to_owned(), json!({"model_name": "gpt-4o"}));
        manifest.insert("avatars".to_owned(), json!(["a.png"]));
        manifest.insert("unknown_member_knob".to_owned(), json!(true));
    });

    let package = load_agent_group(&dir).expect("unknown keys are tolerated");
    // `capabilities` wins over `tags` when both are declared.
    assert_eq!(texts(package.capabilities()), vec!["writing", "editing"]);
    let writer = package
        .template("writer")
        .expect("the writer template loaded");
    // `tools` and `model` are accepted by the loader but belong to the member runtime.
    assert_eq!(
        texts(&writer.skills),
        vec!["narrative", "research", "auto_skill"],
        "member capabilities are not interpreted by the loader"
    );
    assert_eq!(texts(&writer.tags), vec!["prose", "copy"]);
}

#[test]
fn rejects_a_missing_directory_or_an_unreadable_manifest() {
    let temp = tempfile::tempdir().unwrap();
    assert_rejected(
        &temp.path().join("absent"),
        "agent_group package directory not found",
    );

    let empty = temp.path().join("empty_group");
    std::fs::create_dir_all(&empty).unwrap();
    assert_rejected(&empty, "agent_group manifest missing");

    std::fs::write(empty.join("manifest.json"), "{ this is not json").unwrap();
    assert_rejected(&empty, "agent_group manifest is not valid JSON");

    std::fs::write(empty.join("manifest.json"), "[]").unwrap();
    assert_rejected(&empty, "agent_group manifest must contain a JSON object");
}

#[test]
fn rejects_invalid_roster_entries_and_member_manifests() {
    let temp = tempfile::tempdir().unwrap();
    let dir = write_group(temp.path(), GROUP, &["leader", "writer"]);

    patch_json(&dir.join("manifest.json"), |manifest| {
        manifest.insert("agents".to_owned(), json!(["leader", "Writer!"]));
    });
    assert_rejected(&dir, "invalid agent name in agent_group manifest");

    patch_json(&dir.join("manifest.json"), |manifest| {
        manifest.insert("agents".to_owned(), json!(["leader", "ghost"]));
    });
    assert_rejected(&dir, "agent directory not found: ghost");

    patch_json(&dir.join("manifest.json"), |manifest| {
        manifest.insert("agents".to_owned(), json!(["writer"]));
    });
    assert_rejected(&dir, "agent_group manifest agents must contain 'leader'");

    patch_json(&dir.join("manifest.json"), |manifest| {
        manifest.insert("agents".to_owned(), json!(["leader", "writer"]));
    });
    patch_json(&dir.join("agents/writer/manifest.json"), |manifest| {
        manifest.insert("package_type".to_owned(), json!("agent_group"));
    });
    assert_rejected(&dir, "must declare package_type='agent_template'");

    patch_json(&dir.join("agents/writer/manifest.json"), |manifest| {
        manifest.insert("package_type".to_owned(), json!("agent_template"));
        manifest.remove("persona");
    });
    assert_rejected(&dir, "must declare persona.dir");

    // A member with no Markdown under its persona directory is rejected too.
    patch_json(&dir.join("agents/writer/manifest.json"), |manifest| {
        manifest.insert("persona".to_owned(), json!({"dir": "./persona"}));
    });
    std::fs::remove_file(dir.join("agents/writer/persona/01_intro.md")).unwrap();
    std::fs::remove_file(dir.join("agents/writer/persona/02_rules.md")).unwrap();
    std::fs::remove_file(dir.join("agents/writer/persona/extra/10_more.md")).unwrap();
    assert_rejected(&dir, "persona.dir has no Markdown files");
}

#[test]
fn an_unknown_member_is_not_found_on_a_loaded_package() {
    let temp = tempfile::tempdir().unwrap();
    let package = load_agent_group(write_group(temp.path(), GROUP, &["leader", "writer"]))
        .expect("the fixture package is valid");

    let error = package
        .load_member_persona("ghost")
        .expect_err("an unknown member has no persona");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error
            .to_string()
            .contains("AgentGroup member not found: ghost"),
        "error: {error}"
    );
}

#[test]
fn binds_every_spawnable_member_in_name_order_without_the_leader() {
    let temp = tempfile::tempdir().unwrap();
    let dir = write_group(temp.path(), GROUP, &["writer", "leader", "editor"]);
    // One member launches through an external CLI; the bound view flags it.
    patch_json(&dir.join("agents/editor/manifest.json"), |manifest| {
        manifest.insert(
            "runtime".to_owned(),
            json!({"provider_name": "codex", "config": {"mode": "all"}}),
        );
    });
    let package = load_agent_group(&dir).expect("the fixture package is valid");

    let bound = bind_group(&package);
    assert_eq!(bound.name, GROUP);
    assert_eq!(bound.member_names(), vec!["editor", "writer"]);
    assert!(
        bound.member("leader").is_none(),
        "the leader is not spawnable"
    );

    let writer = bound.member("writer").expect("writer is bound");
    // A bound member resolves its display text in the package's default language (`cn`).
    assert_eq!(writer.display_name, "写手");
    assert_eq!(writer.desc, "负责起草");
    assert_eq!(writer.persona, raw_persona("writer"));
    assert!(!writer.external_cli);

    let editor = bound.member("editor").expect("editor is bound");
    assert!(editor.external_cli, "the external runtime is flagged");
    assert_eq!(editor.persona, raw_persona("editor"));

    let value = serde_json::to_value(&bound).unwrap();
    assert_eq!(value["name"], json!(GROUP));
    assert_eq!(
        value["members"]
            .as_array()
            .map(Vec::len)
            .unwrap_or_default(),
        2
    );
    assert_eq!(value["members"][0]["name"], json!("editor"));
    assert_eq!(value["members"][1]["name"], json!("writer"));
}

#[test]
fn two_packages_loaded_from_different_directories_stay_apart() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let writing = load_agent_group(write_group(first.path(), "writing", &["leader", "writer"]))
        .expect("the writing package is valid");
    let editorial = load_agent_group(write_group(
        second.path(),
        "editorial",
        &["leader", "editor"],
    ))
    .expect("the editorial package is valid");

    let bound_writing = bind_group(&writing);
    let bound_editorial = bind_group(&editorial);
    assert_eq!(bound_writing.name, "writing");
    assert_eq!(bound_editorial.name, "editorial");
    assert_eq!(bound_writing.member_names(), vec!["writer"]);
    assert_eq!(bound_editorial.member_names(), vec!["editor"]);
    assert!(bound_editorial.member("writer").is_none());
    assert_ne!(writing.package_dir, editorial.package_dir);
    assert_eq!(
        bound_writing.member("writer").unwrap().persona,
        raw_persona("writer")
    );
    assert_eq!(
        bound_editorial.member("editor").unwrap().persona,
        raw_persona("editor")
    );

    let writing_ref = writing
        .member_cards()
        .into_iter()
        .find(|card| card.name == "writer")
        .unwrap()
        .expert_ref();
    let editorial_ref = editorial
        .member_cards()
        .into_iter()
        .find(|card| card.name == "editor")
        .unwrap()
        .expert_ref();
    assert_eq!(writing_ref, ExpertRef::new("writer").with_group("writing"));
    assert_eq!(
        editorial_ref,
        ExpertRef::new("editor").with_group("editorial")
    );
}

#[test]
fn expert_ref_wire_shape_is_a_name_and_group_pair() {
    let single = ExpertRef::new("writer");
    assert_eq!(single.label(), "writer");
    let value = serde_json::to_value(&single).unwrap();
    assert_eq!(value, json!({"name": "writer", "group": null}));
    assert_eq!(single.to_value(), value);
    let parsed: ExpertRef = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(parsed, single);
    assert_eq!(parsed.group, None, "an absent group stays `None`");
    assert_eq!(serde_json::to_value(&parsed).unwrap(), value);

    let member = ExpertRef::new("narrative_designer").with_group("content");
    assert_eq!(member.label(), "content/narrative_designer");
    let value = serde_json::to_value(&member).unwrap();
    assert_eq!(
        value,
        json!({"name": "narrative_designer", "group": "content"})
    );
    assert_eq!(member.to_value(), value);
    assert_eq!(serde_json::from_value::<ExpertRef>(value).unwrap(), member);

    // Unknown fields are tolerated and an empty group is not a label prefix.
    let tolerant: ExpertRef =
        serde_json::from_value(json!({"name": "writer", "group": null, "extra": 1})).unwrap();
    assert_eq!(tolerant, single);
    let empty_group = ExpertRef {
        name: "writer".to_owned(),
        group: Some(String::new()),
    };
    assert_eq!(empty_group.label(), "writer");
}

#[test]
fn composes_the_rendered_persona_and_skills_onto_a_populated_base() {
    let temp = tempfile::tempdir().unwrap();
    let package = load_agent_group(write_group(temp.path(), GROUP, &["leader", "writer"]))
        .expect("the fixture package is valid");
    let template: &AgentTemplateCard = package.template("writer").expect("the writer template");
    let persona = package.load_member_persona("writer").unwrap();

    // The base owns the identity (card), the model and the workspace; the persona owns the
    // prompt prefix and the template owns the extra skills.
    let base = DeepAgentSpec {
        model: Some(ModelRef {
            name: None,
            model_name: Some("gpt-4o".to_owned()),
            provider: Some("openai".to_owned()),
            params: BTreeMap::new(),
        }),
        card: Some(
            AgentCard::new("Writer")
                .with_id("card-writer")
                .with_description("writes"),
        ),
        system_prompt: Some("Base instructions.".to_owned()),
        skills: Some(vec!["shared".to_owned(), "alpha".to_owned()]),
        language: Some("en".to_owned()),
        workspace: Some(WorkspaceSpec {
            root_path: "/workspace/writer".to_owned(),
            language: "en".to_owned(),
            stable_base: false,
        }),
        ..DeepAgentSpec::default()
    };

    let composed = compose_template_member(&base, template, &persona, Some("en"));
    assert_eq!(
        serde_json::to_value(&composed).unwrap(),
        json!({
            "model": {
                "name": null,
                "model_name": "gpt-4o",
                "provider": "openai",
                "params": {},
            },
            "card": {
                "id": "card-writer",
                "name": "Writer",
                "description": "writes",
                "input_params": null,
                "output_params": null,
                "interface_url": null,
            },
            "system_prompt": "# writer intro\n\n# writer rules\nUse en and /workspace/writer.\n\n# More\n\nBase instructions.",
            "tools": [],
            "mcps": [],
            "subagents": [],
            "rails": [],
            "enable_task_loop": true,
            "max_iterations": null,
            "workspace": {
                "root_path": "/workspace/writer",
                "language": "en",
                "stable_base": false,
            },
            "cwd": null,
            "project_root": null,
            "skills": ["alpha", "auto_skill", "narrative", "research", "shared"],
            "language": "en",
            "approval_required_tools": [],
            "completion_timeout": 600.0,
            "enable_security_rail": true,
            "prompt_mode": null,
        })
    );

    // Composition is idempotent: the persona is not duplicated and skills do not grow.
    let twice = compose_template_member(&composed, template, &persona, Some("en"));
    assert_eq!(twice, composed);
}

#[test]
fn composes_onto_an_empty_base_and_defaults_the_language() {
    let template = AgentTemplateCard::new("writer");
    let empty = DeepAgentSpec::default();
    let composed = compose_template_member(
        &empty,
        &template,
        "You work in {{workspace}} as a {{language}} writer.",
        None,
    );

    assert_eq!(
        composed.system_prompt.as_deref(),
        Some("You work in  as a cn writer."),
        "no workspace renders empty and the language falls back to the default"
    );
    assert_eq!(
        composed.skills, None,
        "an empty union keeps the base `None`"
    );
    assert_eq!(composed.language.as_deref(), Some("cn"));
    assert_eq!(composed.card, None);
    assert_eq!(composed.model, None);

    let value = serde_json::to_value(&composed).unwrap();
    assert_eq!(value["skills"], Value::Null);
    assert_eq!(value["language"], json!("cn"));
    assert_eq!(value["card"], Value::Null);
    assert_eq!(value["enable_task_loop"], json!(true));

    // A template with no skills preserves a declared base skill list, sorted and deduplicated.
    let with_skills = DeepAgentSpec {
        skills: Some(vec!["only".to_owned(), "only".to_owned()]),
        language: Some("cf".to_owned()),
        ..DeepAgentSpec::default()
    };
    let kept = compose_template_member(&with_skills, &template, "", Some(""));
    assert_eq!(kept.skills, Some(vec!["only".to_owned()]));
    assert_eq!(
        kept.language.as_deref(),
        Some("cf"),
        "a blank argument falls through"
    );
    assert_eq!(kept.system_prompt, None, "an empty persona adds no prompt");
}

#[tokio::test]
async fn catalog_accessors_round_trip_the_process_singleton() {
    let _guard = ClearCatalogOnDrop;
    clear_process_expert_catalog();
    assert!(get_process_expert_catalog().is_none());

    let temp = tempfile::tempdir().unwrap();
    let package = load_agent_group(write_group(temp.path(), GROUP, &["leader", "writer"]))
        .expect("the fixture package is valid");
    let group_location = PackageLocation::new(GROUP, package.package_dir.clone(), "local");
    let expert_location =
        PackageLocation::new("writer", package.package_dir.join("agents/writer"), "hub");
    let catalog = Arc::new(FakeCatalog {
        experts: vec![expert_location.clone()],
        groups: vec![group_location.clone()],
    });

    // The location-only seam answers both package kinds and reports a missing one as `None`.
    assert_eq!(
        catalog.list_expert_packages().await.unwrap(),
        vec![expert_location.clone()]
    );
    assert_eq!(
        catalog.locate_expert("writer").await.unwrap(),
        Some(expert_location.clone())
    );
    assert!(catalog.locate_expert("absent").await.unwrap().is_none());
    assert_eq!(
        catalog.list_group_packages().await.unwrap(),
        vec![group_location.clone()]
    );
    assert_eq!(
        catalog.locate_group(GROUP).await.unwrap(),
        Some(group_location.clone())
    );
    assert!(catalog.locate_group("absent").await.unwrap().is_none());

    // The process-wide slot hands the installed catalog back unchanged.
    let installed: Arc<dyn ExpertCatalog> = Arc::clone(&catalog) as Arc<dyn ExpertCatalog>;
    set_process_expert_catalog(Arc::clone(&installed));
    let returned = get_process_expert_catalog().expect("the catalog was installed");
    assert!(Arc::ptr_eq(&returned, &installed));
    assert_eq!(
        returned.locate_group(GROUP).await.unwrap(),
        Some(group_location.clone())
    );

    // A location records the package directory, serialized as a plain string.
    let value = serde_json::to_value(&group_location).unwrap();
    assert_eq!(value["name"], json!(GROUP));
    assert_eq!(value["source"], json!("local"));
    assert_eq!(
        value["path"].as_str().map(Path::new),
        Some(package.package_dir.as_path())
    );
    assert_eq!(
        serde_json::from_value::<PackageLocation>(value).unwrap(),
        group_location
    );

    clear_process_expert_catalog();
    assert!(get_process_expert_catalog().is_none());
}

#[tokio::test]
async fn builds_a_team_whose_teammate_row_carries_the_composed_prompt() {
    let temp = tempfile::tempdir().unwrap();
    let package = load_agent_group(write_group(temp.path(), GROUP, &["leader", "teammate"]))
        .expect("the fixture package is valid");
    let template = package.template("teammate").expect("the teammate template");
    let persona = package.load_member_persona("teammate").unwrap();

    // Compose the template onto the sample blueprint's shared teammate blueprint.
    let base = sample_spec();
    let blueprint = base.agents["teammate"].clone();
    let composed = compose_template_member(&blueprint, template, &persona, None);
    let composed_prompt = composed
        .system_prompt
        .clone()
        .expect("composition renders a prompt");
    assert!(
        composed_prompt.ends_with("do the work"),
        "the base prompt is preserved after the persona: {composed_prompt}"
    );
    assert_eq!(
        composed.skills,
        Some(vec![
            "auto_skill".to_owned(),
            "narrative".to_owned(),
            "research".to_owned()
        ])
    );

    let mut spec = base.clone();
    spec.agents.insert("teammate".to_owned(), composed.clone());
    assert_eq!(
        spec.member_spec("teammate").unwrap().base.prompt,
        composed_prompt
    );

    let factory = FakeHarnessFactory::new();
    let workspace = FakeWorkspaceProvider::new();
    let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
    let agent = spec
        .build_with(build_deps(&factory, &workspace, Some(Arc::clone(&store))))
        .await
        .expect("the template-derived blueprint builds");

    // Identity, roster and language follow the sample blueprint's contract.
    assert_eq!(agent.card().name, "alpha_team_leader");
    assert_eq!(agent.role(), TeamRole::Leader);
    assert_eq!(agent.member_name(), "team_leader");
    assert_eq!(agent.team_name(), "alpha");
    assert_eq!(agent.language(), "cn");
    assert_eq!(
        agent.roster(),
        vec![
            "dev-1".to_owned(),
            "team_leader".to_owned(),
            "teammate".to_owned()
        ]
    );
    assert_eq!(agent.blueprint().agents["teammate"], composed);

    // Only the leader is materialized eagerly, in the workspace the provider allocated for it.
    let calls = factory.calls();
    assert_eq!(calls.len(), 1, "only the leader is built eagerly");
    assert_eq!(calls[0].member_name, "team_leader");
    assert_eq!(calls[0].team_name, "alpha");
    assert_eq!(
        calls[0].workspace_root,
        workspace.root_for("team_leader").unwrap()
    );
    assert_eq!(workspace.allocations(), vec!["team_leader".to_owned()]);

    // The persisted teammate row carries the composed prompt; untouched rows keep theirs.
    let row = store
        .member("alpha", "teammate")
        .await
        .unwrap()
        .expect("the teammate has a row");
    assert_eq!(row.prompt, composed_prompt);
    assert_eq!(row.display_name, "Teammate");
    assert_eq!(row.role, TeamRole::Teammate);
    assert_eq!(row.status, MemberStatus::Unstarted);
    assert_eq!(row.model_ref, None);
    let dev = store
        .member("alpha", "dev-1")
        .await
        .unwrap()
        .expect("dev-1 has a row");
    assert_eq!(dev.prompt, "you are dev-1");
    let leader = store
        .member("alpha", "team_leader")
        .await
        .unwrap()
        .expect("the leader has a row");
    assert_eq!(leader.prompt, "lead the team");

    // The live roster exposes all three rows; the cold teammate has no runtime yet.
    assert_eq!(agent.snapshot().await.members.len(), 3);
    let teammate = agent
        .backend()
        .member("teammate")
        .expect("the teammate has a handle");
    assert_eq!(teammate.record().prompt, composed_prompt);
    assert!(
        teammate.runtime().is_none(),
        "a cold teammate has no runtime"
    );
}
