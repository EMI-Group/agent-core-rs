//! Parsing and validation of `DeepAgentSpec` files in YAML, JSON and Markdown.

use std::collections::BTreeMap;

use crate::schema::PromptMode;
use crate::spec::{DeepAgentSpec, ModelSpec, SkillSpec, SpecFormat, SubAgentSpec};
use openjiuwen_common::error::ErrorKind;
use tempfile::TempDir;

use super::support::write;

#[test]
fn yaml_defaults_and_unknown_fields_are_tolerated() {
    let spec = DeepAgentSpec::parse(
        SpecFormat::Yaml,
        "name: helper\nfuture_field: 42\ntools: [read_file]\n",
    )
    .expect("valid YAML template");
    assert_eq!(spec.name, "helper");
    assert_eq!(spec.description, "");
    assert_eq!(spec.tools, ["read_file"]);
    assert_eq!(spec.skills.len(), 0);
    assert_eq!(spec.subagents.len(), 0);
    assert_eq!(spec.rails.len(), 0);
    assert!(!spec.task_loop);
    assert_eq!(spec.restrict_to_work_dir, None);
    assert_eq!(spec.prompt_mode, None);
    assert_eq!(spec.metadata, BTreeMap::new());
    assert!(spec.validate().is_ok());
}

#[test]
fn json_round_trips_through_serde() {
    let original = DeepAgentSpec {
        id: "my-id".to_string(),
        name: "json-agent".to_string(),
        description: "round trip".to_string(),
        tools: vec!["bash".to_string(), "web_search".to_string()],
        skills: vec![SkillSpec {
            root: "skills".to_string(),
            enabled: false,
        }],
        subagents: vec![SubAgentSpec {
            name: "helper".to_string(),
            preset: Some("explore".to_string()),
            max_iterations: Some(7),
            ..SubAgentSpec::default()
        }],
        rails: vec!["security".to_string(), "core.skill_use".to_string()],
        prompt_mode: Some(PromptMode::Minimal),
        model: Some(ModelSpec {
            provider: Some("openai".to_string()),
            model: Some("gpt-test".to_string()),
            temperature: Some(0.3),
            max_tokens: Some(128),
            ..ModelSpec::default()
        }),
        ..DeepAgentSpec::default()
    };
    let json = serde_json::to_string_pretty(&original).expect("serialize template");
    let parsed = DeepAgentSpec::parse(SpecFormat::Json, &json).expect("deserialize template");
    assert_eq!(parsed, original);
    assert!(parsed.validate().is_ok());
}

#[test]
fn markdown_body_becomes_the_system_prompt() {
    let spec = DeepAgentSpec::parse(
        SpecFormat::Markdown,
        "---\nname: md-agent\ndescription: from markdown\n---\nYou are helpful.\n",
    )
    .expect("valid Markdown template");
    assert_eq!(spec.name, "md-agent");
    assert_eq!(spec.description, "from markdown");
    assert_eq!(
        spec.system_prompt.as_deref().map(str::trim),
        Some("You are helpful.")
    );
}

#[test]
fn markdown_without_front_matter_is_rejected() {
    let error = DeepAgentSpec::parse(SpecFormat::Markdown, "just a prompt")
        .expect_err("a body without front matter is not a template");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("fenced YAML front matter"));
}

#[test]
fn file_stem_becomes_the_name_and_source_is_recorded() {
    let temp = TempDir::new().expect("temp dir");
    let path = temp.path().join("helper.yaml");
    write(&path, "description: no explicit name\ntools: [read_file]\n");
    let spec = DeepAgentSpec::from_path(&path).expect("load template");
    assert_eq!(spec.name, "helper");
    assert_eq!(spec.source_path.as_deref(), Some(path.as_path()));
    assert!(spec.label().contains("helper.yaml"));
}

#[test]
fn unknown_tool_names_the_tool_and_the_file() {
    let temp = TempDir::new().expect("temp dir");
    let path = temp.path().join("bad.yaml");
    write(&path, "name: bad\ntools: [nope]\n");
    let spec = DeepAgentSpec::from_path(&path).expect("load template");
    let error = spec.validate().expect_err("unknown tool is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(message.contains("nope"), "{message}");
    assert!(message.contains("bad.yaml"), "{message}");
}

#[test]
fn unknown_rail_is_rejected_with_the_known_set() {
    let temp = TempDir::new().expect("temp dir");
    let path = temp.path().join("bad-rail.yaml");
    write(&path, "name: bad-rail\nrails: [teleport]\n");
    let spec = DeepAgentSpec::from_path(&path).expect("load template");
    let error = spec.validate().expect_err("unknown rail is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(message.contains("teleport"), "{message}");
    assert!(message.contains("security"), "{message}");
}

#[test]
fn subagent_requires_a_preset_or_description() {
    let spec = DeepAgentSpec {
        name: "parent".to_string(),
        subagents: vec![SubAgentSpec {
            name: "child".to_string(),
            ..SubAgentSpec::default()
        }],
        ..DeepAgentSpec::default()
    };
    let error = spec.validate().expect_err("an empty sub-agent is rejected");
    assert!(error.message().contains("child"));
    assert!(error.message().contains("preset"));
}

#[test]
fn duplicate_subagent_names_are_rejected() {
    let spec = DeepAgentSpec {
        name: "parent".to_string(),
        subagents: vec![
            SubAgentSpec {
                name: "child".to_string(),
                preset: Some("explore".to_string()),
                ..SubAgentSpec::default()
            },
            SubAgentSpec {
                name: "child".to_string(),
                preset: Some("plan".to_string()),
                ..SubAgentSpec::default()
            },
        ],
        ..DeepAgentSpec::default()
    };
    let error = spec
        .validate()
        .expect_err("duplicate sub-agent is rejected");
    assert!(error.message().contains("duplicate sub-agent"));
}

#[test]
fn invalid_model_hints_are_rejected() {
    let spec = DeepAgentSpec {
        name: "bad-model".to_string(),
        model: Some(ModelSpec {
            max_tokens: Some(0),
            ..ModelSpec::default()
        }),
        ..DeepAgentSpec::default()
    };
    let error = spec.validate().expect_err("zero max_tokens is rejected");
    assert!(error.message().contains("max_tokens"));

    let spec = DeepAgentSpec {
        name: "bad-temp".to_string(),
        model: Some(ModelSpec {
            temperature: Some(f32::NAN),
            ..ModelSpec::default()
        }),
        ..DeepAgentSpec::default()
    };
    let error = spec.validate().expect_err("NaN temperature is rejected");
    assert!(error.message().contains("finite"));
}

#[test]
fn unsafe_workspace_hint_is_rejected() {
    let spec = DeepAgentSpec::parse(
        SpecFormat::Yaml,
        "name: traversal\nworkspace:\n  directories: ['../escape']\n",
    )
    .expect("parse template");
    let error = spec.validate().expect_err("traversal is rejected");
    assert!(error.message().contains("../escape"));
}

#[test]
fn unsupported_extension_is_rejected() {
    let temp = TempDir::new().expect("temp dir");
    let path = temp.path().join("agent.toml");
    write(&path, "name = 'x'");
    let error = DeepAgentSpec::from_path(&path).expect_err("extension is unsupported");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("supported"));
}

#[test]
fn format_detection_is_case_insensitive() {
    let temp = TempDir::new().expect("temp dir");
    let path = temp.path().join("agent.YAML");
    write(&path, "name: upper\n");
    let spec = DeepAgentSpec::from_path(&path).expect("uppercase extension loads");
    assert_eq!(spec.name, "upper");
}

#[test]
fn malformed_yaml_is_a_validation_error_naming_the_file() {
    let temp = TempDir::new().expect("temp dir");
    let path = temp.path().join("broken.yaml");
    write(&path, "name: [unclosed\n");
    let error = DeepAgentSpec::from_path(&path).expect_err("broken YAML is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("broken.yaml"));
}
