//! Inline coverage of the pack schema, parsing, serialization, and application.
//!
//! Everything here is offline: tools are in-process `FunctionTool`s, models are
//! `ScriptedModel`s, and file fixtures live in `tempfile` directories.
use super::*;
use crate::agent::Agent;
use crate::card::AgentCard;
use crate::output::ResultType;
use crate::policy::ToolExposureMode;
use crate::prompt_pack::rail::PROMPT_SECTIONS_METADATA_KEY;
use crate::react::ReActAgentConfig;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::{
    BaseMessage, FunctionTool, Model, ModelRequestConfig, Role, Tool, ToolContext,
};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::{MockCompletion, ModelCall, ScriptedModel};
use parking_lot::Mutex;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

const FULL_PACK: &str = r#"
id: support-agent
name: Support Agent
description: Handles support requests
prompts:
  template:
    - { role: system, content: "You are a support agent." }
  sections:
    - { name: policy, content: "Never reveal internal ids.", priority: 20 }
    - { name: tone, content: "Be concise.", priority: 30 }
tools:
  policy: { mode: allowlist, allowed: [search, echo], hide_deferred: true }
  mcp_allowlist: { servers: { kb: [search] } }
rails:
  - { type: guardrail }
  - { type: confirm_interrupt }
  - { type: ask_user }
  - { type: skill, budget: 2000 }
metadata: { owner: team-a }
"#;

type Log = Arc<Mutex<Vec<Value>>>;

/// The tool schemas `call` sent, as `function.name` strings in request order.
fn tool_names(call: &ModelCall) -> Vec<String> {
    call.request
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| tool["function"]["name"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// The text of every message in `messages` that has `role`.
fn texts_with_role(messages: &[BaseMessage], role: &Role) -> Vec<String> {
    messages
        .iter()
        .filter(|message| &message.role == role)
        .map(|message| message.text().into_owned())
        .collect()
}

/// Hands a clone of `model` to the agent while the test keeps the original.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}

/// Builds an agent driven by `model`.
fn agent(model: &ScriptedModel) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("pack-test"));
    agent.set_model(shared(model));
    agent
}

/// A tool named `name` that answers with the fixed text `"tool-output"`.
fn tool(name: &str) -> Arc<dyn Tool> {
    let tool = FunctionTool::builder()
        .name(name)
        .description("test tool")
        .input_schema(json!({"type": "object"}))
        .handler(|_inputs: Value, _ctx: ToolContext| async move {
            Ok(Value::String("tool-output".to_owned()))
        })
        .build()
        .expect("the test tool fixture builds");
    Arc::new(tool)
}

/// Returns a fresh log a [`recording_tool`] writes into.
fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

/// A tool named `name` that records its parsed inputs into `log`.
fn recording_tool(name: &str, log: &Log) -> Arc<dyn Tool> {
    let log = Arc::clone(log);
    let tool = FunctionTool::builder()
        .name(name)
        .description("recording tool")
        .input_schema(json!({"type": "object"}))
        .handler(move |inputs: Value, _ctx: ToolContext| {
            let log = Arc::clone(&log);
            async move {
                log.lock().push(inputs);
                Ok(Value::String("tool-output".to_owned()))
            }
        })
        .build()
        .expect("the recording tool fixture builds");
    Arc::new(tool)
}

/// A resume payload answering the tool calls in `answers`.
fn resume(answers: &Value) -> Value {
    json!({
        "query": {
            "user_inputs": answers,
            "raw_inputs": null,
            "interrupt_answers": {},
        },
        "conversation_id": "s1",
    })
}

#[test]
fn a_full_pack_parses_every_field() {
    let pack = PromptPack::from_yaml(FULL_PACK).expect("the full pack parses");
    assert_eq!(pack.id, "support-agent");
    assert_eq!(pack.name, "Support Agent");
    assert_eq!(pack.description, "Handles support requests");

    assert_eq!(pack.prompts.template.len(), 1);
    assert_eq!(pack.prompts.template[0].role, Role::System);
    assert_eq!(pack.prompts.template[0].text(), "You are a support agent.");
    assert_eq!(pack.prompts.sections.len(), 2);
    assert_eq!(pack.prompts.sections[0].name, "policy");
    assert_eq!(pack.prompts.sections[0].priority, 20);
    assert_eq!(pack.prompts.sections[1].content, "Be concise.");

    let policy = pack.tools.policy.clone().expect("the policy parses");
    assert_eq!(policy.mode, ToolExposureMode::Allowlist);
    assert!(policy.allowed.contains("search"));
    assert!(policy.allowed.contains("echo"));
    assert!(policy.hide_deferred);
    assert_eq!(pack.to_tool_policy(), policy);
    let allowlist = pack
        .tools
        .mcp_allowlist
        .clone()
        .expect("the allowlist parses");
    assert!(allowlist.allows("kb", "search"));
    assert!(!allowlist.allows("kb", "write"));

    assert_eq!(
        pack.rail_specs(),
        [
            RailSpec::Guardrail,
            RailSpec::ConfirmInterrupt,
            RailSpec::AskUser,
            RailSpec::Skill { budget: Some(2000) },
        ]
    );
    assert_eq!(pack.metadata["owner"], json!("team-a"));
}

#[test]
fn a_minimal_pack_defaults_everything_else() {
    let pack = PromptPack::from_yaml("id: minimal\n").expect("the minimal pack parses");
    assert_eq!(pack.id, "minimal");
    assert_eq!(pack.name, "");
    assert_eq!(pack.description, "");
    assert_eq!(pack.prompts.template.len(), 0);
    assert_eq!(pack.prompts.sections.len(), 0);
    assert_eq!(pack.tools.policy, None);
    assert_eq!(pack.tools.mcp_allowlist, None);
    assert_eq!(pack.tools.resilience, None);
    assert_eq!(pack.rails.len(), 0);
    assert!(pack.metadata.is_empty());
    assert_eq!(pack.to_tool_policy(), ToolExposurePolicy::allow_all());
    assert_eq!(SectionSpec::default().priority, DEFAULT_SECTION_PRIORITY);
}

#[test]
fn malformed_documents_are_validation_errors() {
    let error = PromptPack::from_yaml("id: [unclosed").expect_err("invalid YAML");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("YAML"), "{error}");

    let error = PromptPack::from_yaml("- a\n- b\n").expect_err("not a mapping");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn a_pack_without_an_id_is_rejected_by_parsing_and_by_applying() {
    let error = PromptPack::from_yaml("name: no-id\n").expect_err("the id is required");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("id"), "{error}");

    let agent = agent(&ScriptedModel::text("ok"));
    let error = PromptPack::default()
        .apply_to(&agent)
        .expect_err("an empty id cannot be applied");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("id"), "{error}");
}

#[test]
fn unknown_keys_are_tolerated() {
    let pack = PromptPack::from_yaml(
        "id: tolerant\nfuture: {a: 1}\nprompts: {future: true}\ntools: {future: 1}\nmetadata: {m: 1}\n",
    )
    .expect("unknown keys are ignored");
    assert_eq!(pack.id, "tolerant");
    assert_eq!(pack.metadata["m"], json!(1));
    assert_eq!(pack.prompts.template.len(), 0);
}

#[test]
fn yaml_round_trips() {
    let pack = PromptPack::from_yaml(FULL_PACK).expect("the full pack parses");
    let yaml = pack.to_yaml().expect("the pack serializes");
    let again = PromptPack::from_yaml(&yaml).expect("the rendered pack parses");
    assert_eq!(again, pack);
}

#[tokio::test]
async fn from_yaml_file_reads_a_temp_file() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("pack.yaml");
    std::fs::write(&path, FULL_PACK).expect("the fixture is written");

    let pack = PromptPack::from_yaml_file(&path)
        .await
        .expect("the file parses");
    assert_eq!(pack.id, "support-agent");

    let error = PromptPack::from_yaml_file(dir.path().join("missing.yaml"))
        .await
        .expect_err("a missing file is an error");
    assert_eq!(error.kind(), ErrorKind::SysOp);
}

#[tokio::test]
async fn apply_to_sets_the_template_and_preserves_other_config() {
    let model = ScriptedModel::builder().respond_text("ok").build();
    let agent = agent(&model);
    let mut config = ReActAgentConfig::new();
    config
        .configure_model("gpt-4o", "openai")
        .configure_max_iterations(9);
    config.model_config = Some(ModelRequestConfig::new("gpt-4o"));
    agent.configure(config);

    let pack = PromptPack::from_yaml(FULL_PACK).expect("the full pack parses");
    let applied = pack.apply_to(&agent).expect("the pack applies");

    let config = agent.config();
    assert_eq!(config.prompt_template.len(), 1);
    assert_eq!(config.prompt_template[0].text(), "You are a support agent.");
    assert_eq!(config.max_iterations, 9);
    assert_eq!(config.model_name, "gpt-4o");
    assert_eq!(config.model_config, Some(ModelRequestConfig::new("gpt-4o")));

    assert_eq!(
        applied.rails_registered,
        [
            "prompt_sections",
            "guardrail",
            "confirm_interrupt",
            "ask_user",
        ]
    );
    assert_eq!(
        applied.deferred_rails,
        [RailSpec::Skill { budget: Some(2000) }]
    );
    assert!(!applied.is_complete());
}

#[tokio::test]
async fn apply_to_prepends_the_synthesized_identity_without_a_template() {
    let agent = agent(&ScriptedModel::text("ok"));
    let config = ReActAgentConfig {
        prompt_template: vec![BaseMessage::system("existing")],
        ..ReActAgentConfig::new()
    };
    agent.configure(config);

    let pack = PromptPack::from_yaml(
        "id: identity\nname: Support Agent\ndescription: Handles support requests\n",
    )
    .expect("the identity pack parses");
    pack.apply_to(&agent).expect("the pack applies");

    let template = agent.config().prompt_template;
    assert_eq!(template.len(), 2);
    assert_eq!(
        template[0].text(),
        "You are Support Agent.\n\nHandles support requests"
    );
    assert_eq!(template[1].text(), "existing");
}

#[test]
fn identity_messages_cover_name_description_and_both() {
    let name_only = PromptPack {
        name: "Agent".to_owned(),
        ..PromptPack::default()
    };
    assert_eq!(
        name_only.identity_message().unwrap().text(),
        "You are Agent."
    );

    let description_only = PromptPack {
        description: "Does things".to_owned(),
        ..PromptPack::default()
    };
    assert_eq!(
        description_only.identity_message().unwrap().text(),
        "Does things"
    );

    assert!(PromptPack::default().identity_message().is_none());
}

#[tokio::test]
async fn allowlist_policy_hides_a_tool_from_the_next_request() {
    let model = ScriptedModel::builder().respond_text("ok").build();
    let agent = agent(&model);
    agent.add_tool(tool("search"));
    agent.add_tool(tool("hidden"));

    let pack = PromptPack::from_yaml(
        "id: tools\ntools:\n  policy: { mode: allowlist, allowed: [search] }\n",
    )
    .expect("the tool pack parses");
    pack.apply_to(&agent).expect("the pack applies");

    let output = agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    assert_eq!(output.output, "ok");
    assert_eq!(
        tool_names(&model.last_call().expect("the model was called")),
        ["search"]
    );
    assert_eq!(
        agent.tool_exposure_policy(),
        ToolExposurePolicy::allowlist(["search"])
    );
}

#[test]
fn tool_policies_are_installed() {
    let agent = agent(&ScriptedModel::text("ok"));
    let pack = PromptPack::from_yaml(
        r"
id: policies
tools:
  policy: { mode: deny, allowed: [echo] }
  mcp_allowlist: { servers: { files: [read_file] } }
  resilience: { per_tool: { echo: { timeout: 1500, max_attempts: 3 } } }
",
    )
    .expect("the policy pack parses");
    pack.apply_to(&agent).expect("the pack applies");

    assert_eq!(agent.tool_exposure_policy().mode, ToolExposureMode::Deny);
    assert!(agent.mcp_tool_allowlist().allows("files", "read_file"));
    assert!(!agent.mcp_tool_allowlist().allows("files", "write_file"));
    let resilience = agent.tool_resilience().for_tool("echo");
    assert_eq!(resilience.timeout, Some(Duration::from_millis(1500)));
    assert_eq!(resilience.max_attempts, 3);
}

#[test]
fn rails_that_need_host_objects_are_deferred() {
    let agent = agent(&ScriptedModel::text("ok"));
    let pack = PromptPack::from_yaml(
        "id: rails\nrails:\n  - { type: skill }\n  - { type: model_backup }\n  - { type: custom, name: mine }\n",
    )
    .expect("the rail pack parses");
    let applied = pack.apply_to(&agent).expect("the pack applies");
    assert_eq!(applied.rails_registered.len(), 0);
    assert_eq!(applied.deferred_rails, pack.rails);
    assert!(!applied.is_complete());
}

#[tokio::test]
async fn apply_to_with_builds_the_skill_and_model_backup_rails() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(
        dir.path().join("deploy.md"),
        "---\nname: deploy\ndescription: Deploy the service\n---\n\nRun the deploy script.\n",
    )
    .expect("the skill fixture is written");
    let skills = SkillManager::load_dir(dir.path())
        .await
        .expect("the skills load");
    assert!(!skills.is_empty());

    let model = ScriptedModel::builder().respond_text("ok").build();
    let agent = agent(&model);
    let pack = PromptPack::from_yaml(
        "id: ctx\nrails:\n  - { type: skill, budget: 1000 }\n  - { type: model_backup }\n  - { type: custom, name: mine }\n",
    )
    .expect("the rail pack parses");
    let ctx = PromptPackContext::new()
        .with_skills(skills)
        .with_models([Arc::new(ScriptedModel::text("backup")) as Arc<dyn Model>]);
    let applied = pack
        .apply_to_with(&agent, &ctx)
        .expect("the pack applies with a context");

    assert_eq!(applied.rails_registered, ["skill", "model_backup"]);
    assert_eq!(
        applied.deferred_rails,
        [RailSpec::Custom {
            name: "mine".to_owned(),
            priority: None,
            options: BTreeMap::new(),
        }]
    );

    agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    let call = model.last_call().expect("the model was called");
    let system_texts = texts_with_role(&call.messages, &Role::System);
    assert!(
        system_texts.iter().any(|text| text.contains("deploy")),
        "the skill catalog reaches the model: {system_texts:?}"
    );
}

#[tokio::test]
async fn sections_reach_the_model_as_one_marked_system_message() {
    let model = ScriptedModel::builder().respond_text("ok").build();
    let agent = agent(&model);
    let pack = PromptPack::from_yaml(
        r#"
id: sections
prompts:
  sections:
    - { name: tone, content: "Be concise.", priority: 30 }
    - { name: policy, content: "Never reveal internal ids.", priority: 20 }
"#,
    )
    .expect("the section pack parses");
    let applied = pack.apply_to(&agent).expect("the pack applies");
    assert_eq!(applied.rails_registered, ["prompt_sections"]);

    agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    let call = model.last_call().expect("the model was called");
    let injected = call
        .messages
        .iter()
        .find(|message| message.metadata.contains_key(PROMPT_SECTIONS_METADATA_KEY))
        .expect("the sections message is injected");
    assert_eq!(injected.role, Role::System);
    assert_eq!(
        injected.text(),
        "## policy\nNever reveal internal ids.\n\n## tone\nBe concise."
    );
    assert_eq!(call.messages[0].text(), injected.text());
}

#[tokio::test]
async fn a_confirm_rail_from_a_pack_pauses_and_resumes() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("write", json!({"path": "x"})))
        .respond_text("done")
        .build();
    let agent = agent(&model);
    let log = log();
    agent.add_tool(recording_tool("write", &log));

    let pack = PromptPack::from_yaml(
        r"
id: confirm
rails:
  - { type: confirm_interrupt }
",
    )
    .expect("the confirm pack parses");
    let applied = pack.apply_to(&agent).expect("the pack applies");
    assert_eq!(applied.rails_registered, ["confirm_interrupt"]);

    let session: Arc<dyn Session> = Arc::new(AgentSession::new("s1"));
    let paused = agent
        .invoke(json!({"query": "do it"}), Some(Arc::clone(&session)))
        .await
        .expect("the run pauses");
    assert_eq!(paused.result_type, ResultType::Interrupt);
    assert_eq!(paused.interrupt_ids.len(), 1);
    assert!(log.lock().is_empty(), "the paused tool never runs");

    let resumed = agent
        .invoke(
            resume(&json!({"call_write": {"approved": true}})),
            Some(Arc::clone(&session)),
        )
        .await
        .expect("the run resumes");
    assert_eq!(resumed.result_type, ResultType::Answer);
    assert_eq!(resumed.output, "done");
    assert_eq!(log.lock().len(), 1, "the approved tool ran once");
}

#[test]
fn built_rails_report_the_names_the_pack_declares() {
    let agent = agent(&ScriptedModel::text("ok"));
    let pack = PromptPack::from_yaml("id: kinds\nrails:\n  - { type: guardrail }\n")
        .expect("the rail pack parses");
    let applied = pack.apply_to(&agent).expect("the pack applies");
    assert_eq!(applied.rails_registered, ["guardrail"]);
    assert_eq!(
        agent.rails().rails()[0].name(),
        "guardrail",
        "the rail is registered on the agent"
    );
}
