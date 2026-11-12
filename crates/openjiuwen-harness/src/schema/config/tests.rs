//! Unit tests for the `DeepAgent` configuration vocabulary.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use openjiuwen_agent::{AgentCard, MaxIterationsPolicy};
use openjiuwen_common::error::Error;
use openjiuwen_common::security::SecretString;
use serde_json::json;

use crate::schema::mode::{AgentMode, PromptMode};
use crate::workspace::Workspace;

use super::*;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

#[test]
fn deep_agent_config_defaults_match_python() {
    let config = DeepAgentConfig::default();
    assert!(config.model_name.is_none());
    assert!(config.model_provider.is_none());
    assert!(config.api_key.is_none());
    assert!(config.api_base.is_none());
    assert!(config.custom_headers.is_empty());
    assert!(config.model_config.is_none());
    assert!(config.model_client_config.is_none());
    assert!(config.card.name.is_empty(), "the default card has no name");
    assert_eq!(config.card.id.len(), 32, "a default card id is a uuid hex");
    assert!(config.system_prompt.is_none());
    assert!(config.max_iterations.is_none());
    assert!(config.max_iterations_hard_limit.is_none());
    assert_eq!(config.on_max_iterations, MaxIterationsPolicy::Error);
    assert!(!config.enable_task_loop);
    assert!(!config.enable_async_subagent);
    assert!(!config.enable_subagent_runtime);
    assert!(!config.add_general_purpose_agent);
    assert!((config.completion_timeout_secs - 600.0).abs() < f64::EPSILON);
    assert_eq!(config.workspace, Workspace::default());
    assert!(config.cwd.is_none());
    assert!(config.project_root.is_none());
    assert!(
        config.skills.is_empty(),
        "no skills are configured by default"
    );
    assert!(!config.enable_skill_discovery);
    assert_eq!(config.language, "cn");
    assert_eq!(config.prompt_mode, PromptMode::Full);
    assert!(
        config.subagents.is_empty(),
        "no sub-agents are configured by default"
    );
    assert_eq!(config.default_mode, AgentMode::Normal);
    assert!(!config.enable_plan_mode);
    assert!(config.parallel_tool_calls);
    assert!(config.restrict_to_work_dir);
    assert!(config.enable_tool_resilience_rail);
    assert!(!config.enable_task_planning);
    assert!(config.enable_security_rail);
    assert!(config.permissions.is_none());
    assert_eq!(
        config.task_loop_no_progress_guard,
        TaskLoopNoProgressGuardConfig::default()
    );
    assert!(config.skill_budget_max_skills.is_none());
    assert!(config.skill_budget_max_total_chars.is_none());
    assert!(config.invocation_log_dir.is_none());
    assert!(config.validate().is_ok(), "the defaults must validate");
}

#[test]
fn task_loop_no_progress_guard_defaults_match_python() {
    let guard = TaskLoopNoProgressGuardConfig::default();
    assert!(guard.enabled);
    assert_eq!(guard.max_consecutive_empty_answers, 3);
    assert_eq!(guard.min_answer_chars, 20);
}

#[test]
fn subagent_config_defaults_match_python() {
    let config = SubAgentConfig::default();
    assert!(
        config.agent_card.name.is_empty(),
        "the default card has no name"
    );
    assert_eq!(
        config.agent_card.id.len(),
        32,
        "a default card id is a uuid hex"
    );
    assert!(config.system_prompt.is_none());
    assert!(
        config.tools.is_empty(),
        "no tools are configured by default"
    );
    assert!(
        config.skills.is_empty(),
        "no skills are configured by default"
    );
    assert!(config.max_iterations.is_none());
    assert!(!config.enable_task_loop);
    assert!(config.language.is_none());
    assert!(config.prompt_mode.is_none());
    assert!(config.factory_name.is_none());
    assert!(config.factory_kwargs.is_empty());
    assert!(!config.enable_plan_mode);
    assert!(config.parallel_tool_calls);
    assert!(config.restrict_to_work_dir);
    assert!(config.display_name.is_none());
    assert!(config.role.is_none());
    assert!(config.subagent_type.is_none());
}

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

#[test]
fn json_round_trip_preserves_config() {
    let config = DeepAgentConfig::default()
        .with_model_name(Some("gpt-4o-mini".to_string()))
        .with_model_provider(Some("openai".to_string()))
        .with_api_key(Some(SecretString::new("sk-test")))
        .with_language("en".to_string())
        .with_max_iterations(Some(12))
        .with_enable_task_loop(true)
        .with_skills(vec!["alpha".to_string(), "beta".to_string()])
        .with_subagents(vec![SubAgentConfig::new(AgentCard::new("researcher"))])
        .with_default_mode(AgentMode::Plan)
        .with_prompt_mode(PromptMode::Minimal)
        .with_on_max_iterations(MaxIterationsPolicy::FinalAnswer);
    let value = serde_json::to_value(&config).expect("serialize config");
    let back: DeepAgentConfig = serde_json::from_value(value).expect("deserialize config");
    assert_eq!(back, config);
}

#[test]
fn unknown_fields_are_tolerated() {
    let value = json!({
        "unknown_field": 1,
        "nested": {"still": "unknown"},
        "language": "en",
        "parallel_tool_calls": false,
    });
    let config: DeepAgentConfig = serde_json::from_value(value).expect("unknown fields ignored");
    assert_eq!(config.language, "en");
    assert!(!config.parallel_tool_calls);
    assert_eq!(
        config.prompt_mode,
        PromptMode::Full,
        "missing fields use defaults"
    );
}

#[test]
fn subagent_config_json_round_trip() {
    let config = SubAgentConfig::new(AgentCard::new("reviewer"))
        .with_tools(vec!["read_file".to_string()])
        .with_max_iterations(Some(5))
        .with_factory_kwargs(BTreeMap::from([("depth".to_string(), json!(2))]));
    let value = serde_json::to_value(&config).expect("serialize subagent config");
    let back: SubAgentConfig = serde_json::from_value(value).expect("deserialize subagent config");
    assert_eq!(back, config);
}

#[test]
fn skills_accepts_a_list() {
    let config: DeepAgentConfig =
        serde_json::from_value(json!({"skills": ["alpha", "beta"]})).expect("list of skills");
    assert_eq!(config.skills, vec!["alpha".to_string(), "beta".to_string()]);
}

#[test]
fn skills_accepts_a_single_string() {
    let config: DeepAgentConfig =
        serde_json::from_value(json!({"skills": "alpha"})).expect("single skill");
    assert_eq!(config.skills, vec!["alpha".to_string()]);
}

#[test]
fn skills_rejects_other_json_types() {
    let error = serde_json::from_value::<DeepAgentConfig>(json!({"skills": 3}))
        .expect_err("a number is not a skill definition");
    assert!(error.to_string().contains("invalid type"), "{error}");
    let error = serde_json::from_value::<DeepAgentConfig>(json!({"skills": ["alpha", 3]}))
        .expect_err("a non-string element is rejected");
    assert!(!error.to_string().is_empty(), "{error}");
}

#[test]
fn completion_timeout_alias_is_accepted() {
    let config: DeepAgentConfig =
        serde_json::from_value(json!({"completion_timeout": 30.0})).expect("python alias");
    assert!((config.completion_timeout_secs - 30.0).abs() < f64::EPSILON);
    let config: DeepAgentConfig =
        serde_json::from_value(json!({"completion_timeout_secs": 45.0})).expect("rust name");
    assert!((config.completion_timeout_secs - 45.0).abs() < f64::EPSILON);
}

#[test]
fn debug_redacts_the_api_key() {
    let config =
        DeepAgentConfig::default().with_api_key(Some(SecretString::new("sk-super-secret")));
    let rendered = format!("{config:?}");
    assert!(
        !rendered.contains("sk-super-secret"),
        "the api key must never reach Debug: {rendered}"
    );
    assert!(
        rendered.contains("REDACTED"),
        "Debug must show the redaction marker: {rendered}"
    );
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn assert_config_error(config: &DeepAgentConfig, expected: &str) {
    let error = config
        .validate()
        .expect_err("configuration must be rejected");
    assert_eq!(error.to_string(), format!("config: {expected}"));
    assert!(matches!(error, Error::Config(_)));
}

#[test]
fn validate_rejects_hard_limit_with_task_loop() {
    let config = DeepAgentConfig::default()
        .with_enable_task_loop(true)
        .with_max_iterations_hard_limit(Some(5))
        .with_max_iterations(Some(3));
    assert_config_error(
        &config,
        "max_iterations_hard_limit requires single-round mode (enable_task_loop=False)",
    );
}

#[test]
fn validate_rejects_zero_hard_limit() {
    let config = DeepAgentConfig::default()
        .with_max_iterations_hard_limit(Some(0))
        .with_max_iterations(Some(1));
    assert_config_error(
        &config,
        "max_iterations_hard_limit must be a positive integer",
    );
}

#[test]
fn validate_rejects_missing_iterations_under_a_hard_limit() {
    let config = DeepAgentConfig::default().with_max_iterations_hard_limit(Some(5));
    assert_config_error(
        &config,
        "max_iterations must be positive and within max_iterations_hard_limit",
    );
}

#[test]
fn validate_rejects_zero_iterations_under_a_hard_limit() {
    let config = DeepAgentConfig::default()
        .with_max_iterations_hard_limit(Some(5))
        .with_max_iterations(Some(0));
    assert_config_error(
        &config,
        "max_iterations must be positive and within max_iterations_hard_limit",
    );
}

#[test]
fn validate_rejects_iterations_above_the_hard_limit() {
    let config = DeepAgentConfig::default()
        .with_max_iterations_hard_limit(Some(5))
        .with_max_iterations(Some(6));
    assert_config_error(
        &config,
        "max_iterations must be positive and within max_iterations_hard_limit",
    );
}

#[test]
fn validate_accepts_iterations_at_the_hard_limit() {
    for iterations in [1, 5] {
        let config = DeepAgentConfig::default()
            .with_max_iterations_hard_limit(Some(5))
            .with_max_iterations(Some(iterations));
        assert!(config.validate().is_ok(), "max_iterations={iterations}");
    }
}

#[test]
fn validate_rejects_standalone_zero_iterations() {
    let config = DeepAgentConfig::default().with_max_iterations(Some(0));
    assert_config_error(&config, "max_iterations must be a positive integer");
}

#[test]
fn validate_accepts_positive_iterations_without_a_ceiling() {
    let config = DeepAgentConfig::default().with_max_iterations(Some(3));
    assert!(config.validate().is_ok());
    let config = DeepAgentConfig::default().with_enable_task_loop(true);
    assert!(config.validate().is_ok());
}

#[test]
fn resolve_inner_react_max_iterations_matches_python() {
    assert_eq!(resolve_inner_react_max_iterations(None), usize::MAX);
    assert_eq!(resolve_inner_react_max_iterations(Some(5)), 5);
    assert_eq!(resolve_inner_react_max_iterations(Some(0)), 0);
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

#[test]
fn builders_override_every_field_type() {
    let config = DeepAgentConfig::default()
        .with_model_name(Some("model".to_string()))
        .with_model_provider(Some("provider".to_string()))
        .with_api_key(Some(SecretString::new("sk-test")))
        .with_api_base(Some("https://example.test".to_string()))
        .with_custom_headers(BTreeMap::from([("x".to_string(), "y".to_string())]))
        .with_card(AgentCard::new("agent"))
        .with_system_prompt(Some("prompt".to_string()))
        .with_max_iterations(Some(4))
        .with_on_max_iterations(MaxIterationsPolicy::ReturnObservations)
        .with_enable_task_loop(true)
        .with_enable_async_subagent(true)
        .with_enable_subagent_runtime(true)
        .with_add_general_purpose_agent(true)
        .with_completion_timeout_secs(30.0)
        .with_cwd(Some("work".into()))
        .with_project_root(Some("root".into()))
        .with_skills(vec!["skill".to_string()])
        .with_enable_skill_discovery(true)
        .with_language("en".to_string())
        .with_prompt_mode(PromptMode::None)
        .with_default_mode(AgentMode::Plan)
        .with_enable_plan_mode(true)
        .with_parallel_tool_calls(false)
        .with_restrict_to_work_dir(false)
        .with_enable_tool_resilience_rail(false)
        .with_enable_task_planning(true)
        .with_enable_security_rail(false)
        .with_skill_budget_max_skills(Some(5))
        .with_skill_budget_max_total_chars(Some(500))
        .with_invocation_log_dir(Some("logs".into()))
        .with_task_loop_no_progress_guard(TaskLoopNoProgressGuardConfig {
            enabled: false,
            max_consecutive_empty_answers: 1,
            min_answer_chars: 2,
        });
    assert_eq!(config.model_name.as_deref(), Some("model"));
    assert_eq!(config.model_provider.as_deref(), Some("provider"));
    assert_eq!(config.api_base.as_deref(), Some("https://example.test"));
    assert_eq!(
        config.custom_headers.get("x").map(String::as_str),
        Some("y")
    );
    assert_eq!(config.card.name, "agent");
    assert_eq!(config.system_prompt.as_deref(), Some("prompt"));
    assert_eq!(config.max_iterations, Some(4));
    assert_eq!(
        config.on_max_iterations,
        MaxIterationsPolicy::ReturnObservations
    );
    assert!(config.enable_task_loop);
    assert!(config.enable_async_subagent);
    assert!(config.enable_subagent_runtime);
    assert!(config.add_general_purpose_agent);
    assert!((config.completion_timeout_secs - 30.0).abs() < f64::EPSILON);
    assert_eq!(config.cwd.as_deref(), Some(std::path::Path::new("work")));
    assert_eq!(
        config.project_root.as_deref(),
        Some(std::path::Path::new("root"))
    );
    assert_eq!(config.skills, vec!["skill".to_string()]);
    assert!(config.enable_skill_discovery);
    assert_eq!(config.language, "en");
    assert_eq!(config.prompt_mode, PromptMode::None);
    assert_eq!(config.default_mode, AgentMode::Plan);
    assert!(config.enable_plan_mode);
    assert!(!config.parallel_tool_calls);
    assert!(!config.restrict_to_work_dir);
    assert!(!config.enable_tool_resilience_rail);
    assert!(config.enable_task_planning);
    assert!(!config.enable_security_rail);
    assert_eq!(config.skill_budget_max_skills, Some(5));
    assert_eq!(config.skill_budget_max_total_chars, Some(500));
    assert_eq!(
        config.invocation_log_dir.as_deref(),
        Some(std::path::Path::new("logs"))
    );
    assert!(!config.task_loop_no_progress_guard.enabled);
    assert!(config.api_key.is_some());
    assert_eq!(config.workspace, Workspace::default());
    assert!(
        config.subagents.is_empty(),
        "no sub-agents are configured by default"
    );
    assert!(config.permissions.is_none());
    assert!(config.max_iterations_hard_limit.is_none());
}

#[test]
fn subagent_builders_override_every_field() {
    let card = AgentCard::new("researcher");
    let config = SubAgentConfig::new(card.clone())
        .with_system_prompt(Some("be terse".to_string()))
        .with_tools(vec!["read_file".to_string()])
        .with_skills(vec!["rust".to_string()])
        .with_max_iterations(Some(7))
        .with_enable_task_loop(true)
        .with_language(Some("en".to_string()))
        .with_prompt_mode(Some(PromptMode::Minimal))
        .with_factory_name(Some("reviewer".to_string()))
        .with_factory_kwargs(BTreeMap::from([("k".to_string(), json!(1))]))
        .with_enable_plan_mode(true)
        .with_parallel_tool_calls(false)
        .with_restrict_to_work_dir(false)
        .with_display_name(Some("Researcher".to_string()))
        .with_role(Some("research".to_string()))
        .with_subagent_type(Some("general".to_string()))
        .with_agent_card(card.clone());
    assert_eq!(config.agent_card, card);
    assert_eq!(config.system_prompt.as_deref(), Some("be terse"));
    assert_eq!(config.tools, vec!["read_file".to_string()]);
    assert_eq!(config.skills, vec!["rust".to_string()]);
    assert_eq!(config.max_iterations, Some(7));
    assert!(config.enable_task_loop);
    assert_eq!(config.language.as_deref(), Some("en"));
    assert_eq!(config.prompt_mode, Some(PromptMode::Minimal));
    assert_eq!(config.factory_name.as_deref(), Some("reviewer"));
    assert_eq!(config.factory_kwargs.get("k"), Some(&json!(1)));
    assert!(config.enable_plan_mode);
    assert!(!config.parallel_tool_calls);
    assert!(!config.restrict_to_work_dir);
    assert_eq!(config.display_name.as_deref(), Some("Researcher"));
    assert_eq!(config.role.as_deref(), Some("research"));
    assert_eq!(config.subagent_type.as_deref(), Some("general"));
}

// ---------------------------------------------------------------------------
// JSON Schema
// ---------------------------------------------------------------------------

#[test]
fn json_schema_is_generated() {
    let schema =
        serde_json::to_value(schemars::schema_for!(DeepAgentConfig)).expect("serialize schema");
    assert_eq!(schema["type"], json!("object"));
    for field in [
        "model_name",
        "api_key",
        "model_config",
        "model_client_config",
        "card",
        "on_max_iterations",
        "workspace",
        "cwd",
        "project_root",
        "skills",
        "permissions",
        "invocation_log_dir",
    ] {
        assert!(
            schema["properties"].get(field).is_some(),
            "`{field}` must appear in the schema"
        );
    }
    let api_key = schema["properties"]["api_key"].to_string();
    assert!(api_key.contains("string"), "api_key is a string: {api_key}");
    assert!(api_key.contains("null"), "api_key is nullable: {api_key}");
    let on_max_iterations = schema["properties"]["on_max_iterations"].to_string();
    assert!(
        on_max_iterations.contains("string"),
        "on_max_iterations is a string enum: {on_max_iterations}"
    );
}

#[test]
fn subagent_json_schema_is_generated() {
    let schema =
        serde_json::to_value(schemars::schema_for!(SubAgentConfig)).expect("serialize schema");
    assert_eq!(schema["type"], json!("object"));
    assert!(schema["properties"].get("agent_card").is_some());
    assert!(schema["properties"].get("factory_kwargs").is_some());
    let guard = serde_json::to_value(schemars::schema_for!(TaskLoopNoProgressGuardConfig))
        .expect("serialize guard schema");
    assert_eq!(guard["type"], json!("object"));
    assert!(guard["properties"]
        .get("max_consecutive_empty_answers")
        .is_some());
}
