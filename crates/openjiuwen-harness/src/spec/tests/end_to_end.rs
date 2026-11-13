//! End-to-end: a template on disk is discovered, resolved, and mounted on a real
//! `DeepAgent` driven by a scripted model.

use std::sync::Arc;

use crate::spec::{
    build_agent, load_plugins, resolve_with_source, DeepAgentSpec, SpecBuildOptions, SpecRegistry,
};
use openjiuwen_agent::Agent;
use openjiuwen_session::AgentSession;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::json;

use super::support::{agents_dir, temp_workspace, write};

/// One template that exercises tools, skills, sub-agents, rails and prompts.
const SOLVER_YAML: &str = r"
name: solver
id: spec.solver
description: solves coding tasks
language: en
system_prompt: |
  You solve coding tasks.
tools:
  - bash
  - read_file
  - web_search
skills:
  - root: ../skills
    enabled: true
subagents:
  - name: helper
    preset: explore
    description: explores the repository
    tools: [read_file, grep]
rails: [security, skills, subagent]
workspace:
  language: en
  directories: [notes]
max_iterations: 4
";

#[tokio::test(flavor = "multi_thread")]
async fn a_disk_template_is_discovered_resolved_and_mounted() {
    let (temp, workspace) = temp_workspace();
    write(&agents_dir(temp.path()).join("solver.yaml"), SOLVER_YAML);
    let skills = temp.path().join(".openjiuwen").join("skills");
    std::fs::create_dir_all(&skills).expect("create the skill root");

    let registry = SpecRegistry::discover(&workspace, &[]).expect("discovery succeeds");
    let resolved = registry
        .resolve("solver", &workspace)
        .expect("resolve succeeds");
    assert_eq!(resolved.card.id, "spec.solver");
    assert!(resolved
        .subagents
        .iter()
        .any(|child| child.agent_card.name == "helper"));

    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "web_search",
            json!({"query": "rust"}),
        ))
        .respond_text("search attempted")
        .build();
    let scripted = model.clone();
    let agent = build_agent(
        resolved.clone(),
        Arc::new(model),
        SpecBuildOptions::default(),
    )
    .await
    .expect("the template builds a DeepAgent");

    let config = agent.config();
    assert_eq!(config.language, "en");
    assert_eq!(config.max_iterations, Some(4));
    assert!(config.enable_security_rail);
    assert_eq!(config.subagents.len(), 1);
    assert_eq!(config.subagents[0].agent_card.name, "helper");
    assert_eq!(config.skills.len(), 1);
    let mounted = std::fs::canonicalize(&config.skills[0]).expect("mounted root exists");
    let expected = std::fs::canonicalize(&skills).expect("expected root exists");
    assert_eq!(mounted, expected);
    assert!(config
        .workspace
        .directories
        .iter()
        .any(|node| node.name == "notes"));

    let session: Arc<dyn openjiuwen_session::Session> = Arc::new(AgentSession::new("spec-e2e"));
    let output = agent
        .invoke(json!({"query": "find the bug"}), Some(session))
        .await
        .expect("the turn succeeds");
    assert_eq!(output.output, "search attempted");
    assert_eq!(scripted.call_count(), 2);

    // The second model call carries the result of the extended tool the template
    // added; an unmounted tool would never produce this payload.
    let calls = scripted.calls();
    let result = calls[1]
        .messages
        .iter()
        .find(|message| message.tool_call_id.as_deref() == Some("call_web_search"))
        .expect("the web_search tool ran");
    assert!(
        result
            .text()
            .contains("web search backend is not configured"),
        "unexpected tool result: {}",
        result.text()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_template_without_extended_tools_builds_through_the_builder() {
    let (temp, workspace) = temp_workspace();
    let path = agents_dir(temp.path()).join("plain.md");
    write(
        &path,
        "---\nname: plain\ndescription: markdown template\ntools: [read_file]\n---\nYou are concise.\n",
    );
    let spec = DeepAgentSpec::from_path(&path).expect("load the markdown template");
    let resolved = resolve_with_source(&spec, &workspace).expect("resolve succeeds");
    assert_eq!(
        resolved.system_prompt.as_deref().map(str::trim),
        Some("You are concise.")
    );

    let model = ScriptedModel::text("plain answer");
    let agent = resolved
        .into_builder(Arc::new(model), workspace.clone())
        .expect("the builder wires")
        .build()
        .expect("the agent builds");
    let session: Arc<dyn openjiuwen_session::Session> = Arc::new(AgentSession::new("plain-e2e"));
    let output = agent
        .invoke(json!({"query": "hello"}), Some(session))
        .await
        .expect("the turn succeeds");
    assert_eq!(output.output, "plain answer");
    assert_eq!(agent.config().card.name, "plain");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plugin_template_builds_its_own_agent() {
    let (temp, workspace) = temp_workspace();
    let package = temp.path().join(".openjiuwen").join("plugins").join("acme");
    write(
        &package.join("plugin.yaml"),
        "name: acme\nversion: '1.0.0'\ntools: [web_search]\nrails: [security]\nskills: [skills]\nspec: agent.yaml\n",
    );
    std::fs::create_dir_all(package.join("skills")).expect("create the plugin skill root");
    write(
        &package.join("agent.yaml"),
        "name: acme-agent\ndescription: plugin supplied\ntools: [read_file]\n",
    );

    let plugins = load_plugins(&workspace).expect("plugins load");
    assert_eq!(plugins.len(), 1);
    let spec = plugins.specs()[0].clone();
    let resolved = resolve_with_source(&spec, &workspace).expect("the plugin template resolves");
    let model = ScriptedModel::text("plugin answer");
    let agent = build_agent(resolved, Arc::new(model), SpecBuildOptions::default())
        .await
        .expect("the plugin agent builds");

    let session: Arc<dyn openjiuwen_session::Session> = Arc::new(AgentSession::new("plugin-e2e"));
    let output = agent
        .invoke(json!({"query": "hello"}), Some(session))
        .await
        .expect("the turn succeeds");
    assert_eq!(output.output, "plugin answer");
    assert_eq!(agent.config().card.name, "acme-agent");
}
