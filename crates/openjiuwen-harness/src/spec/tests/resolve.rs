//! Resolution of a template into harness parts: card, prompts, tools, skills,
//! sub-agents, rails, model hints and workspace layout.

use std::collections::BTreeMap;

use crate::schema::PromptMode;
use crate::spec::{
    resolve, resolve_with_source, DeepAgentSpec, ModelSpec, PromptFragment, SkillSpec,
    SubAgentSpec, WorkspaceHints,
};
use crate::workspace::Workspace;
use openjiuwen_common::error::ErrorKind;
use tempfile::TempDir;

use super::support::{temp_workspace, write};

/// Builds a fully populated in-memory template.
fn full_spec() -> DeepAgentSpec {
    DeepAgentSpec {
        id: "agent-7".to_string(),
        name: "solver".to_string(),
        description: "solves problems".to_string(),
        language: Some("en".to_string()),
        system_prompt: Some("You solve problems.".to_string()),
        prompt_sections: vec![
            PromptFragment {
                name: "late".to_string(),
                content: "Late rule.".to_string(),
                priority: 90,
            },
            PromptFragment {
                name: "early".to_string(),
                content: "Early rule.".to_string(),
                priority: 10,
            },
        ],
        model: Some(ModelSpec {
            provider: Some("scripted".to_string()),
            model: Some("scripted-model".to_string()),
            api_base: Some("http://localhost".to_string()),
            temperature: Some(0.25),
            top_p: Some(0.9),
            max_tokens: Some(512),
        }),
        tools: vec![
            "bash".to_string(),
            "read_file".to_string(),
            "web_search".to_string(),
        ],
        skills: vec![
            SkillSpec {
                root: "skills".to_string(),
                enabled: true,
            },
            SkillSpec {
                root: "disabled-skills".to_string(),
                enabled: false,
            },
        ],
        subagents: vec![SubAgentSpec {
            name: "helper".to_string(),
            description: "explores the codebase".to_string(),
            preset: Some("explore".to_string()),
            tools: vec!["read_file".to_string(), "grep".to_string()],
            model: Some(ModelSpec {
                model: Some("child-model".to_string()),
                ..ModelSpec::default()
            }),
            max_iterations: Some(9),
            ..SubAgentSpec::default()
        }],
        rails: vec![
            "core.security".to_string(),
            "skills".to_string(),
            "subagent".to_string(),
        ],
        workspace: WorkspaceHints {
            root: Some("./".to_string()),
            language: Some("en".to_string()),
            directories: vec!["notes".to_string()],
        },
        prompt_mode: Some(PromptMode::Minimal),
        max_iterations: Some(12),
        task_loop: true,
        restrict_to_work_dir: Some(true),
        metadata: BTreeMap::from([("owner".to_string(), serde_json::json!("tests"))]),
        source_path: None,
    }
}

#[test]
fn resolve_builds_every_harnes_part() {
    let (temp, workspace) = temp_workspace();
    let spec = full_spec();
    let resolved = resolve(&spec, &workspace, temp.path()).expect("resolve succeeds");

    assert_eq!(resolved.card.id, "agent-7");
    assert_eq!(resolved.card.name, "solver");
    assert_eq!(resolved.card.description, "solves problems");
    assert_eq!(resolved.language, "en");
    assert_eq!(
        resolved.system_prompt.as_deref(),
        Some("You solve problems.\n\nEarly rule.\n\nLate rule.")
    );
    assert_eq!(
        resolved
            .prompt_fragments
            .iter()
            .map(|fragment| fragment.name.as_str())
            .collect::<Vec<_>>(),
        ["early", "late"]
    );
    assert_eq!(resolved.tools.len(), 3);
    assert_eq!(resolved.tools[2].name(), "web_search");
    assert_eq!(resolved.rails, ["security", "skills", "subagent"]);
    assert_eq!(resolved.skills.len(), 2);
    assert!(resolved.skills[0].enabled);
    assert!(resolved.skills[0].mount.is_some());
    assert!(!resolved.skills[1].enabled);
    assert!(resolved.skills[1].mount.is_none());
    assert_eq!(resolved.skill_roots.len(), 1);
    assert_eq!(resolved.skill_roots[0], temp.path().join("skills"));
    assert_eq!(resolved.enabled_skill_roots().len(), 1);

    assert_eq!(resolved.subagents.len(), 1);
    let child = &resolved.subagents[0];
    assert_eq!(child.agent_card.name, "helper");
    assert_eq!(child.agent_card.description, "explores the codebase");
    assert_eq!(child.subagent_type.as_deref(), Some("explore_agent"));
    assert_eq!(child.tools, ["read_file", "grep"]);
    assert_eq!(child.max_iterations, Some(9));
    assert_eq!(
        child
            .factory_kwargs
            .get("model")
            .and_then(|value| value.get("model"))
            .and_then(serde_json::Value::as_str),
        Some("child-model")
    );
}

#[test]
fn relative_skill_roots_use_the_spec_directory() {
    let temp = TempDir::new().expect("temp dir");
    let spec_dir = temp.path().join("agents");
    write(&spec_dir.join("solver.yaml"), "name: solver\n");
    let path = spec_dir.join("solver.yaml");
    let mut spec = DeepAgentSpec::from_path(&path).expect("load template");
    spec.skills = vec![SkillSpec {
        root: "../shared-skills".to_string(),
        enabled: true,
    }];
    let workspace = Workspace::for_language(temp.path(), "en");
    let resolved = resolve_with_source(&spec, &workspace).expect("resolve");
    assert_eq!(resolved.skill_roots[0], spec_dir.join("../shared-skills"));
}

#[test]
fn to_config_maps_all_template_data() {
    let (temp, workspace) = temp_workspace();
    let resolved = resolve(&full_spec(), &workspace, temp.path()).expect("resolve");
    let config = resolved.to_config(workspace.clone());
    assert_eq!(config.card.name, "solver");
    assert_eq!(config.language, "en");
    assert!(config.enable_task_loop);
    assert_eq!(config.max_iterations, Some(12));
    assert_eq!(config.prompt_mode, PromptMode::Minimal);
    assert!(config.enable_security_rail);
    assert_eq!(config.subagents.len(), 1);
    assert_eq!(config.skills, resolved.enabled_skill_roots());
    assert_eq!(config.workspace.root_path, workspace.root_path);
    assert_eq!(config.model_provider.as_deref(), Some("scripted"));
    assert_eq!(config.model_name.as_deref(), Some("scripted-model"));
    assert_eq!(config.api_base.as_deref(), Some("http://localhost"));
    let request = config.model_config.expect("request config is built");
    assert_eq!(request.model_name, "scripted-model");
    assert_eq!(request.temperature, Some(0.25));
    assert_eq!(request.max_tokens, Some(512));
}

#[test]
fn a_non_empty_rail_list_without_security_disables_it() {
    let (temp, workspace) = temp_workspace();
    let mut spec = full_spec();
    spec.rails = vec!["skills".to_string()];
    let resolved = resolve(&spec, &workspace, temp.path()).expect("resolve");
    let config = resolved.to_config(workspace.clone());
    assert!(!config.enable_security_rail);

    spec.rails.clear();
    let resolved = resolve(&spec, &workspace, temp.path()).expect("resolve");
    let config = resolved.to_config(workspace);
    assert!(
        config.enable_security_rail,
        "the default is kept for an empty list"
    );
}

#[test]
fn workspace_applies_layout_hints() {
    let (temp, workspace) = temp_workspace();
    let resolved = resolve(&full_spec(), &workspace, temp.path()).expect("resolve");
    let workspace = resolved.workspace();
    assert_eq!(workspace.language, "en");
    assert!(
        workspace
            .directories
            .iter()
            .any(|node| node.name == "notes"),
        "the extra directory hint is appended"
    );
    // The default schema nodes are still there.
    assert!(workspace
        .directories
        .iter()
        .any(|node| node.name == "skills"));
}

#[test]
fn supplemental_tools_exclude_the_builder_defaults() {
    let (temp, workspace) = temp_workspace();
    let resolved = resolve(&full_spec(), &workspace, temp.path()).expect("resolve");
    let tools = resolved.supplemental_tools(&workspace);
    let names: Vec<String> = tools.iter().map(|tool| tool.card().name.clone()).collect();
    assert_eq!(names, ["web_search"]);

    let mut spec = full_spec();
    spec.tools = vec!["bash".to_string(), "read_file".to_string()];
    let resolved = resolve(&spec, &workspace, temp.path()).expect("resolve");
    assert!(resolved.supplemental_tools(&workspace).is_empty());
}

#[test]
fn unknown_subagent_preset_is_rejected() {
    let (temp, workspace) = temp_workspace();
    let mut spec = full_spec();
    spec.subagents[0].preset = Some("teleport".to_string());
    let error = resolve(&spec, &workspace, temp.path()).expect_err("unknown preset");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("teleport"));
}

#[test]
fn custom_subagent_without_preset_uses_its_description() {
    let (temp, workspace) = temp_workspace();
    let mut spec = full_spec();
    spec.subagents[0].preset = None;
    spec.subagents[0].description = "custom worker".to_string();
    let resolved = resolve(&spec, &workspace, temp.path()).expect("resolve");
    let child = &resolved.subagents[0];
    assert_eq!(child.agent_card.name, "helper");
    assert_eq!(child.agent_card.description, "custom worker");
    assert_eq!(child.subagent_type.as_deref(), Some("helper"));
    assert_eq!(child.role.as_deref(), Some("custom worker"));
}

#[test]
fn duplicate_rails_are_deduplicated() {
    let (temp, workspace) = temp_workspace();
    let mut spec = full_spec();
    spec.rails = vec![
        "security".to_string(),
        "core.security".to_string(),
        "core.confirm_interrupt".to_string(),
    ];
    let resolved = resolve(&spec, &workspace, temp.path()).expect("resolve");
    assert_eq!(resolved.rails, ["security", "permission_interrupt"]);
}
