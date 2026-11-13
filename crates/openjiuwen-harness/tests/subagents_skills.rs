//! Integration suite for the harness sub-agent runtime and the skill library.
//!
//! Covers the built-in preset registry, the in-process [`SubAgentRuntime`] driven by a
//! local fake [`SubAgentFactory`], the model-facing `task_tool`/`subagent_*` tools, and
//! the skills side: recursive `SKILL.md` discovery, the selection budget, the persisted
//! disabled-state file, and owned skill-root mounts.
//!
//! Everything is offline and deterministic: the fake factory returns immediately, the
//! filesystem fixtures live in per-test `tempfile::TempDir` roots, and the only waits are
//! short deadlines that a completed turn satisfies at once.
//!
//! Size note: this suite is long by design — it is the single integration file for both the
//! sub-agent runtime (quota, LRU, transcripts, persistence) and the skill library.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::stream::{self, BoxStream};
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::tool::{Tool, ToolContext};
use openjiuwen_harness::schema::SubAgentConfig;
use openjiuwen_harness::skills::{
    Skill, SkillBudget, SkillLibrary, SkillRootMount, SKILL_FILENAME,
};
use openjiuwen_harness::subagents::{
    rfc3339_from_ms, subagent_tools, subagent_tools_with_transcript, ManualClock, SubAgentFactory,
    SubAgentOutcome, SubAgentPreset, SubAgentQuota, SubAgentRegistry, SubAgentRuntime,
    SubAgentSpawnRequest, SubAgentSpec, SubagentActivity, SubagentMetadata, SubagentStatus,
    TaskTool, TranscriptKind, EVICTION_REASON, GENERAL_PURPOSE_NAME,
};
use openjiuwen_session::Session;
use serde_json::{json, Value};

/// Deadline for a wait that the fake agents satisfy on their first turn.
const WAIT: Duration = Duration::from_millis(500);

/// Deadline for a wait whose ids are already final or unknown.
const IMMEDIATE: Duration = Duration::from_millis(10);

/// Bounded deadline of the gated turns and the lifecycle polling (never a sleep).
const BOUND: Duration = Duration::from_secs(5);

/// The frozen instant every [`ManualClock`] of this suite starts at.
const T0: i64 = 1_700_000_000_000;

/// The five Python preset names, in registry (alphabetical) order.
const PRESET_NAMES: [&str; 5] = [
    "code_agent",
    "explore_agent",
    "plan_agent",
    "research_agent",
    "verification_agent",
];

// ---------------------------------------------------------------------------
// Fakes: the runtime only needs the `Agent` and `SubAgentFactory` seams.
// ---------------------------------------------------------------------------

/// Minimal agent returning one canned answer; `stream` yields nothing.
struct FakeAgent {
    card: AgentCard,
    output: String,
}

#[async_trait::async_trait]
impl Agent for FakeAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        Ok(AgentOutput::answer(self.output.clone()))
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(stream::empty())
    }
}

/// What the fake factory observed when the runtime built a child agent.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CreatedAgent {
    subagent_type: Option<String>,
    max_iterations: Option<u32>,
    session_id: String,
}

/// Factory recording every child it builds and answering `done: <card name>`.
#[derive(Default)]
struct FakeFactory {
    created: Mutex<Vec<CreatedAgent>>,
}

impl FakeFactory {
    fn created(&self) -> Vec<CreatedAgent> {
        self.created
            .lock()
            .expect("factory recording lock is not poisoned")
            .clone()
    }
}

#[async_trait::async_trait]
impl SubAgentFactory for FakeFactory {
    async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>> {
        self.created
            .lock()
            .expect("factory recording lock is not poisoned")
            .push(CreatedAgent {
                subagent_type: config.subagent_type.clone(),
                max_iterations: config.max_iterations,
                session_id: session_id.to_owned(),
            });
        let card = AgentCard::new(config.agent_card.name.clone())
            .with_id(format!("child-agent-{session_id}"))
            .with_description(config.agent_card.description.clone());
        Ok(Arc::new(FakeAgent {
            card,
            output: format!("done: {}", config.agent_card.name),
        }))
    }
}

/// Factory that always fails, proving the runtime rolls the reservation back.
struct FailingFactory;

#[async_trait::async_trait]
impl SubAgentFactory for FailingFactory {
    async fn create(&self, _config: &SubAgentConfig, _session_id: &str) -> Result<Arc<dyn Agent>> {
        Err(Error::config("factory exploded"))
    }
}

/// Factory whose child agent parks its turn until the test opens the gate.
///
/// `tokio::sync::watch` replaces a sleep: the turn stays `Running` for as long as the test
/// needs to observe that phase, and the gate opens deterministically.
struct GatedFactory {
    gate: tokio::sync::watch::Sender<bool>,
}

impl GatedFactory {
    /// Creates the factory and the gate the test opens to release the turn.
    fn new() -> (tokio::sync::watch::Sender<bool>, Arc<Self>) {
        let (gate, _receiver) = tokio::sync::watch::channel(false);
        let sender = gate.clone();
        (gate, Arc::new(Self { gate: sender }))
    }
}

#[async_trait::async_trait]
impl SubAgentFactory for GatedFactory {
    async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>> {
        let card = AgentCard::new(config.agent_card.name.clone())
            .with_id(format!("child-agent-{session_id}"))
            .with_description(config.agent_card.description.clone());
        Ok(Arc::new(GatedAgent {
            card,
            gate: self.gate.clone(),
        }))
    }
}

/// Agent whose turn stays `Running` until its gate opens, then answers `done: <card name>`.
struct GatedAgent {
    card: AgentCard,
    gate: tokio::sync::watch::Sender<bool>,
}

#[async_trait::async_trait]
impl Agent for GatedAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let mut gate = self.gate.subscribe();
        if !*gate.borrow_and_update() {
            tokio::time::timeout(BOUND, gate.changed())
                .await
                .map_err(|_| Error::config("the gated sub-agent turn was never released"))?
                .map_err(|error| Error::config(format!("sub-agent gate closed: {error}")))?;
        }
        Ok(AgentOutput::answer(format!("done: {}", self.card.name)))
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(stream::empty())
    }
}

/// Builds a runtime over the English built-in presets.
fn builtin_runtime(
    factory: Arc<dyn SubAgentFactory>,
    max_subagents: usize,
    max_concurrent: usize,
) -> Arc<SubAgentRuntime> {
    let registry = Arc::new(SubAgentRegistry::with_builtin_presets("en"));
    Arc::new(
        SubAgentRuntime::new(registry, factory, "parent-s1")
            .with_limits(max_subagents, max_concurrent),
    )
}

/// Builds a runtime over the English built-in presets with `quota` and an injected clock.
///
/// Unlike [`builtin_runtime`] this keeps LRU eviction at the quota's setting, which is what
/// the quota, activity and eviction cases below exercise.
fn quota_runtime(
    factory: Arc<dyn SubAgentFactory>,
    quota: SubAgentQuota,
    clock: Arc<ManualClock>,
) -> Arc<SubAgentRuntime> {
    let registry = Arc::new(SubAgentRegistry::with_builtin_presets("en"));
    Arc::new(
        SubAgentRuntime::new(registry, factory, "parent-s1")
            .with_quota(quota)
            .with_clock(clock),
    )
}

/// Yields to the runtime until `predicate` holds, giving up at [`BOUND`].
///
/// The loop only yields, so the background turn task makes progress without a wall-clock sleep.
async fn wait_until(mut predicate: impl FnMut() -> bool, what: &str) {
    let deadline = tokio::time::Instant::now() + BOUND;
    while !predicate() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        tokio::task::yield_now().await;
    }
}

/// Returns the names of `tools` in declaration order.
fn tool_names(tools: &[Arc<dyn Tool>]) -> Vec<String> {
    tools.iter().map(|tool| tool.card().name.clone()).collect()
}

/// Returns the registered tool called `name`.
fn find_tool(tools: &[Arc<dyn Tool>], name: &str) -> Arc<dyn Tool> {
    tools
        .iter()
        .find(|tool| tool.card().name == name)
        .expect("tool is registered")
        .clone()
}

/// Returns the `success` flag of a tool payload.
fn succeeded(payload: &Value) -> bool {
    payload.get("success").and_then(Value::as_bool) == Some(true)
}

/// Returns the `error` text of a tool payload.
fn payload_error(payload: &Value) -> String {
    payload
        .get("error")
        .and_then(Value::as_str)
        .expect("failed payload carries an error string")
        .to_owned()
}

/// Returns the message of a quota refusal, panicking for any other error variant.
fn config_message(error: &Error) -> &str {
    match error {
        Error::Config(message) => message,
        other => panic!("expected a config error, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// A. Sub-agents
// ---------------------------------------------------------------------------

#[test]
fn builtin_presets_expose_the_python_configurations() {
    let registry = SubAgentRegistry::with_builtin_presets("en");
    assert_eq!(registry.names(), PRESET_NAMES);
    assert_eq!(registry.len(), PRESET_NAMES.len());

    let code = registry
        .get("code_agent")
        .expect("code preset is registered");
    assert_eq!(code.subagent_type.as_deref(), Some("code_agent"));
    assert_eq!(code.max_iterations, Some(50));
    assert!(!code.enable_task_loop);
    assert!(!code.enable_plan_mode);
    assert!(code.parallel_tool_calls);
    assert!(code.restrict_to_work_dir);
    assert_eq!(code.language.as_deref(), Some("en"));
    assert_eq!(code.factory_name.as_deref(), Some("code_agent"));
    assert!(code
        .system_prompt
        .as_deref()
        .is_some_and(|prompt| !prompt.is_empty()));
    assert!(code
        .agent_card
        .description
        .contains("senior software engineer"));
    assert_eq!(
        registry.get_spec("code_agent").and_then(|spec| spec.preset),
        Some(SubAgentPreset::Code)
    );

    // Python only leaves Explore and Plan unrestricted, and only Code/Research name a factory.
    let explore = registry
        .get("explore_agent")
        .expect("explore preset is registered");
    assert!(!explore.restrict_to_work_dir);
    assert_eq!(explore.factory_name, None);
    assert_eq!(
        registry
            .get_spec("explore_agent")
            .and_then(|spec| spec.preset),
        Some(SubAgentPreset::Explore)
    );

    let research = registry
        .get("research_agent")
        .expect("research preset is registered");
    assert_eq!(research.factory_name.as_deref(), Some("research_agent"));
    assert!(research.restrict_to_work_dir);

    let verification = registry
        .get("verification_agent")
        .expect("verification preset is registered");
    assert!(verification
        .agent_card
        .description
        .contains("Adversarial verification specialist"));
}

#[test]
fn chinese_presets_keep_python_names_and_localize_prompts() {
    let english = SubAgentRegistry::with_builtin_presets("en");
    let chinese = SubAgentRegistry::with_builtin_presets("cn");
    assert_eq!(chinese.names(), english.names());
    assert_eq!(chinese.names(), PRESET_NAMES);

    let code = chinese
        .get("code_agent")
        .expect("code preset is registered");
    assert_eq!(code.language.as_deref(), Some("cn"));
    assert!(code.agent_card.description.contains("资深软件工程师"));
    assert_eq!(code.max_iterations, Some(50));
    assert_eq!(code.subagent_type.as_deref(), Some("code_agent"));

    let english_code = english
        .get("code_agent")
        .expect("code preset is registered");
    assert_ne!(
        code.agent_card.description,
        english_code.agent_card.description
    );
    assert_ne!(code.system_prompt, english_code.system_prompt);
}

#[test]
fn general_purpose_custom_specs_and_list_stay_consistent() {
    let registry = SubAgentRegistry::with_builtin_presets("en");
    assert_eq!(registry.list().len(), registry.names().len());
    let listed: Vec<String> = registry
        .list()
        .into_iter()
        .map(|config| config.agent_card.name)
        .collect();
    assert_eq!(listed, registry.names());

    let general = registry.general_purpose();
    assert_eq!(general.agent_card.name, GENERAL_PURPOSE_NAME);
    assert_eq!(general.subagent_type.as_deref(), Some(GENERAL_PURPOSE_NAME));
    assert_eq!(general.max_iterations, None);
    assert!(!general.restrict_to_work_dir);
    assert_eq!(registry.len(), PRESET_NAMES.len() + 1);
    assert_eq!(registry.general_purpose(), general);
    assert_eq!(registry.len(), PRESET_NAMES.len() + 1);

    let custom = SubAgentConfig::new(AgentCard::new("custom_worker").with_description("custom"))
        .with_subagent_type(Some("custom_worker".to_owned()))
        .with_max_iterations(Some(7));
    registry.register(SubAgentSpec::new(custom.clone()));
    assert!(registry.contains("custom_worker"));
    assert_eq!(registry.get("custom_worker"), Some(custom));
    assert_eq!(
        registry
            .get_spec("custom_worker")
            .and_then(|spec| spec.preset),
        None
    );
    let mut expected_names = PRESET_NAMES.map(str::to_owned).to_vec();
    expected_names.push(GENERAL_PURPOSE_NAME.to_owned());
    expected_names.push("custom_worker".to_owned());
    expected_names.sort();
    assert_eq!(registry.names(), expected_names);
}

#[tokio::test]
async fn runtime_spawns_waits_and_reports_live_capacity() {
    let factory = Arc::new(FakeFactory::default());
    let runtime = builtin_runtime(factory.clone(), 2, 2);
    assert_eq!(runtime.parent_session_id(), "parent-s1");
    assert_eq!(runtime.capacity(), (0, 2));

    let first = runtime
        .spawn(SubAgentSpawnRequest::new("code_agent", "do the thing"))
        .await
        .expect("first spawn succeeds");
    let second = runtime
        .spawn(SubAgentSpawnRequest::new("explore_agent", "look around"))
        .await
        .expect("second spawn succeeds");
    assert_ne!(first, second);
    assert!(first.starts_with("parent-s1_sub_code_agent_"));
    assert!(second.starts_with("parent-s1_sub_explore_agent_"));
    assert_eq!(runtime.capacity(), (2, 2));

    // Duplicate ids are collapsed and input order is preserved.
    let outcomes = runtime
        .wait(&[first.clone(), second.clone(), first.clone()], WAIT)
        .await;
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].subagent_id, first);
    assert_eq!(outcomes[0].subagent_type, "code_agent");
    assert_eq!(outcomes[0].status, SubagentStatus::Completed);
    assert_eq!(outcomes[0].output.as_deref(), Some("done: code_agent"));
    assert_eq!(outcomes[0].agent_id, format!("child-agent-{first}"));
    assert_eq!(outcomes[1].subagent_id, second);
    assert_eq!(outcomes[1].status, SubagentStatus::Completed);
    assert_eq!(outcomes[1].output.as_deref(), Some("done: explore_agent"));
    assert!(outcomes.iter().all(SubAgentOutcome::is_final));

    assert_eq!(runtime.get_status(&first), SubagentStatus::Completed);
    let outcome = runtime.get_outcome(&first).expect("first outcome exists");
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(outcome.sub_session_id, first);
    assert_eq!(outcome.output.as_deref(), Some("done: code_agent"));

    let listed = runtime.list();
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].subagent_id, first);
    assert_eq!(listed[1].subagent_id, second);
    assert_eq!(listed[0].task_description, "do the thing");
    assert_eq!(listed[0].status, SubagentStatus::Completed);
    assert_ne!(listed[0].created_at, "");

    let created = factory.created();
    assert_eq!(created.len(), 2);
    assert_eq!(created[0].subagent_type.as_deref(), Some("code_agent"));
    assert_eq!(created[0].max_iterations, Some(50));
    assert_eq!(created[0].session_id, first);
    assert_eq!(created[1].subagent_type.as_deref(), Some("explore_agent"));
}

#[tokio::test]
async fn runtime_enforces_the_live_instance_cap() {
    let runtime = builtin_runtime(Arc::new(FakeFactory::default()), 2, 1);
    let first = runtime
        .spawn(SubAgentSpawnRequest::new("code_agent", "one"))
        .await
        .expect("first spawn succeeds");
    let second = runtime
        .spawn(SubAgentSpawnRequest::new("explore_agent", "two"))
        .await
        .expect("second spawn succeeds");

    let error = runtime
        .spawn(SubAgentSpawnRequest::new("plan_agent", "three"))
        .await
        .expect_err("the live-instance cap rejects the third spawn");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("subagent capacity exceeded"),
        "unexpected message: {error}"
    );
    assert!(
        error.to_string().contains("used=2, limit=2"),
        "unexpected message: {error}"
    );
    assert_eq!(runtime.capacity(), (2, 2));

    let unknown_type = runtime
        .spawn(SubAgentSpawnRequest::new("no_such_agent", "x"))
        .await
        .expect_err("unknown types are rejected");
    assert_eq!(unknown_type.kind(), ErrorKind::Validation);
    assert!(unknown_type
        .to_string()
        .contains("subagent type not found: no_such_agent"));

    runtime.wait(&[first, second], WAIT).await;
}

#[tokio::test]
async fn runtime_close_resume_and_unknown_ids_follow_python_semantics() {
    let runtime = builtin_runtime(Arc::new(FakeFactory::default()), 4, 2);

    // Unknown ids: NotFound status, no outcome, validation errors on every mutation.
    assert_eq!(runtime.get_status("missing"), SubagentStatus::NotFound);
    assert_eq!(runtime.get_outcome("missing"), None);
    let unknown = runtime.wait(&["missing".to_owned()], IMMEDIATE).await;
    assert_eq!(unknown.len(), 1);
    assert_eq!(unknown[0].status, SubagentStatus::NotFound);
    assert!(unknown[0].is_final());
    assert_eq!(
        runtime
            .close("missing")
            .await
            .expect_err("unknown close fails")
            .kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        runtime
            .resume("missing")
            .await
            .expect_err("unknown resume fails")
            .kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        runtime
            .send_input("missing", "hi", false)
            .await
            .expect_err("unknown input fails")
            .kind(),
        ErrorKind::Validation
    );

    let id = runtime
        .spawn(SubAgentSpawnRequest::new("code_agent", "do the thing"))
        .await
        .expect("spawn succeeds");
    runtime.wait(std::slice::from_ref(&id), WAIT).await;

    let previous = runtime.close(&id).await.expect("closing an idle instance");
    assert_eq!(previous, SubagentStatus::Completed);
    assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);
    assert_eq!(runtime.list(), Vec::<SubagentMetadata>::new());
    assert_eq!(runtime.capacity(), (0, 4));
    assert_eq!(
        runtime
            .send_input(&id, "again", false)
            .await
            .expect_err("closed instances reject input")
            .kind(),
        ErrorKind::Validation
    );

    let report = runtime
        .resume(&id)
        .await
        .expect("resuming a closed instance");
    assert!(report.restored);
    assert_eq!(report.status, SubagentStatus::Completed);
    assert_eq!(report.message, None);
    assert_eq!(runtime.get_status(&id), SubagentStatus::Completed);

    let live = runtime.resume(&id).await.expect("resuming a live instance");
    assert!(!live.restored);
    assert!(live.message.is_some());

    runtime
        .send_input(&id, "again", false)
        .await
        .expect("resumed instances accept input");
    let outcomes = runtime.wait(std::slice::from_ref(&id), WAIT).await;
    assert_eq!(outcomes[0].status, SubagentStatus::Completed);
    assert_eq!(outcomes[0].output.as_deref(), Some("done: code_agent"));
}

#[tokio::test]
async fn failed_factory_spawn_rolls_the_reservation_back() {
    let registry = Arc::new(SubAgentRegistry::with_builtin_presets("en"));
    let runtime = Arc::new(
        SubAgentRuntime::new(registry, Arc::new(FailingFactory), "parent-s1").with_limits(2, 1),
    );

    let mut request = SubAgentSpawnRequest::new("code_agent", "boom");
    request.subagent_id = Some("child-fail".to_owned());
    let error = runtime
        .spawn(request)
        .await
        .expect_err("the factory failure surfaces");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("factory exploded"));

    assert_eq!(runtime.capacity(), (0, 2));
    assert_eq!(runtime.get_status("child-fail"), SubagentStatus::NotFound);
    assert_eq!(runtime.list(), Vec::<SubagentMetadata>::new());
}

#[tokio::test]
async fn verification_agent_ids_are_sticky_and_live_duplicates_are_rejected() {
    let runtime = builtin_runtime(Arc::new(FakeFactory::default()), 4, 2);
    let first = runtime
        .spawn(SubAgentSpawnRequest::new("verification_agent", "verify"))
        .await
        .expect("first spawn succeeds");
    assert_eq!(first, "parent-s1_sub_verification_agent");

    let duplicate = runtime
        .spawn(SubAgentSpawnRequest::new(
            "verification_agent",
            "verify again",
        ))
        .await
        .expect_err("a live sticky id cannot be spawned twice");
    assert_eq!(duplicate.kind(), ErrorKind::Validation);
    assert!(duplicate.to_string().contains("already live"));

    runtime.wait(std::slice::from_ref(&first), WAIT).await;
    runtime.close(&first).await.expect("close the instance");
    let again = runtime
        .spawn(SubAgentSpawnRequest::new(
            "verification_agent",
            "verify again",
        ))
        .await
        .expect("the closed tombstone is reusable");
    assert_eq!(again, first);
}

#[tokio::test]
async fn subagent_tools_expose_python_names_and_payload_shape() {
    let runtime = builtin_runtime(Arc::new(FakeFactory::default()), 4, 2);
    let tools = subagent_tools(Arc::clone(&runtime));
    assert_eq!(
        tool_names(&tools),
        [
            "subagent_spawn",
            "subagent_wait",
            "subagent_list",
            "subagent_send_input",
            "subagent_close",
            "subagent_resume",
        ]
    );
    let ctx = ToolContext::new();

    // Expected failures are payloads, never `Err`.
    let spawn = find_tool(&tools, "subagent_spawn");
    let missing = spawn
        .invoke(json!({"subagent_type": "code_agent"}), &ctx)
        .await
        .expect("expected failures never return Err");
    assert!(!succeeded(&missing));
    assert_eq!(
        payload_error(&missing),
        "'subagent_type', 'task_description', 'display_name', and 'role' are required"
    );

    let spawned = spawn
        .invoke(
            json!({
                "subagent_type": "code_agent",
                "task_description": "do it",
                "display_name": "Coder",
                "role": "write code",
            }),
            &ctx,
        )
        .await
        .expect("spawn tool succeeds");
    assert!(succeeded(&spawned));
    let id = spawned["data"]["subagent_id"]
        .as_str()
        .expect("spawn returns a subagent id")
        .to_owned();
    assert!(id.starts_with("parent-s1_sub_code_agent_"));
    assert_eq!(spawned["data"]["sub_session_id"], id);
    assert!(spawned["data"]["task_id"]
        .as_str()
        .is_some_and(|task_id| task_id.len() == 32));

    let wait = find_tool(&tools, "subagent_wait");
    let waited = wait
        .invoke(json!({"subagent_ids": [id.clone()]}), &ctx)
        .await
        .expect("wait tool succeeds");
    assert!(succeeded(&waited));
    assert_eq!(waited["data"]["statuses"][id.as_str()], "completed");
    assert_eq!(waited["data"]["results"][id.as_str()], "done: code_agent");
    assert_eq!(waited["data"]["timed_out"], false);
    assert!(waited["data"]["output_files"].is_object());

    let unknown = wait
        .invoke(json!({"subagent_ids": ["nope"]}), &ctx)
        .await
        .expect("unknown ids are a payload");
    assert!(succeeded(&unknown));
    assert_eq!(unknown["data"]["statuses"]["nope"], "not_found");
    assert_eq!(unknown["data"]["timed_out"], false);

    let empty = wait
        .invoke(json!({"subagent_ids": []}), &ctx)
        .await
        .expect("an empty id list is a payload");
    assert!(!succeeded(&empty));
    assert!(payload_error(&empty).contains("non-empty list"));

    let bad_timeout = wait
        .invoke(json!({"subagent_ids": [id], "timeout_ms": "soon"}), &ctx)
        .await
        .expect("a bad deadline is a payload");
    assert!(!succeeded(&bad_timeout));
    assert!(payload_error(&bad_timeout).contains("must be an integer"));

    let list = find_tool(&tools, "subagent_list")
        .invoke(json!({}), &ctx)
        .await
        .expect("list tool succeeds");
    assert!(succeeded(&list));
    assert_eq!(list["data"]["capacity"]["used"], 1);
    assert_eq!(list["data"]["capacity"]["max"], 4);
    assert_eq!(list["data"]["subagents"][0]["subagent_id"], id);
    assert_eq!(list["data"]["subagents"][0]["display_name"], "Coder");
    assert_eq!(list["data"]["subagents"][0]["role"], "write code");
    assert_eq!(list["data"]["subagents"][0]["status"], "completed");
    assert_eq!(list["data"]["summary"]["live_count"], 1);
    assert_eq!(list["data"]["summary"]["closed_count"], 0);
}

#[tokio::test]
async fn subagent_close_and_resume_tools_round_trip() {
    let runtime = builtin_runtime(Arc::new(FakeFactory::default()), 4, 2);
    let tools = subagent_tools(Arc::clone(&runtime));
    let ctx = ToolContext::new();

    let id = runtime
        .spawn(SubAgentSpawnRequest::new("code_agent", "do it"))
        .await
        .expect("spawn succeeds");
    runtime.wait(std::slice::from_ref(&id), WAIT).await;

    let closed = find_tool(&tools, "subagent_close")
        .invoke(json!({"subagent_id": id}), &ctx)
        .await
        .expect("close tool succeeds");
    assert!(succeeded(&closed));
    assert_eq!(closed["data"]["previous_status"], "completed");
    assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);

    let resumed = find_tool(&tools, "subagent_resume")
        .invoke(json!({"subagent_id": id}), &ctx)
        .await
        .expect("resume tool succeeds");
    assert!(succeeded(&resumed));
    assert_eq!(resumed["data"]["status"], "completed");
    assert_eq!(resumed["data"]["restored"], true);

    let send = find_tool(&tools, "subagent_send_input")
        .invoke(json!({"subagent_id": id, "query": "again"}), &ctx)
        .await
        .expect("send-input tool succeeds");
    assert!(succeeded(&send));
    assert_eq!(send["data"]["subagent_id"], id);
    let outcomes = runtime.wait(std::slice::from_ref(&id), WAIT).await;
    assert_eq!(outcomes[0].output.as_deref(), Some("done: code_agent"));

    let unknown = find_tool(&tools, "subagent_close")
        .invoke(json!({"subagent_id": "nope"}), &ctx)
        .await
        .expect("unknown close is a payload");
    assert!(!succeeded(&unknown));
    assert!(payload_error(&unknown).contains("subagent not found: nope"));
}

#[tokio::test]
async fn task_tool_returns_payloads_instead_of_errors() {
    let runtime = builtin_runtime(Arc::new(FakeFactory::default()), 4, 2);
    let tool = TaskTool::new(Arc::clone(&runtime)).with_timeout(WAIT);
    let card = tool.card();
    assert_eq!(card.name, "task_tool");
    assert_ne!(card.description, "");
    assert_eq!(card.input_params["type"], "object");
    let mut required: Vec<String> = card.input_params["required"]
        .as_array()
        .expect("required is an array")
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    required.sort();
    assert_eq!(required, ["subagent_type", "task_description"]);
    assert_eq!(tool.timeout(), WAIT);

    let ctx = ToolContext::new();
    let answered = tool
        .invoke(
            json!({"subagent_type": "code_agent", "task_description": "do it"}),
            &ctx,
        )
        .await
        .expect("task_tool never returns Err for expected failures");
    assert!(succeeded(&answered));
    assert_eq!(answered["data"]["output"], "done: code_agent");
    assert_eq!(answered["data"]["subagent_type"], "code_agent");
    assert!(answered["data"]["agent_id"]
        .as_str()
        .is_some_and(|agent_id| agent_id.starts_with("child-agent-")));

    let unknown = tool
        .invoke(
            json!({"subagent_type": "no_such_agent", "task_description": "x"}),
            &ctx,
        )
        .await
        .expect("unknown types are a payload");
    assert!(!succeeded(&unknown));
    assert_eq!(
        payload_error(&unknown),
        "Subagent type 'no_such_agent' is not available through task_tool"
    );

    let incomplete = tool
        .invoke(json!({"subagent_type": "code_agent"}), &ctx)
        .await
        .expect("missing fields are a payload");
    assert!(!succeeded(&incomplete));
    assert_eq!(
        payload_error(&incomplete),
        "'subagent_type' and 'task_description' are required"
    );
}

// ---------------------------------------------------------------------------
// A2. Runtime quota, activity and eviction surfaces
// ---------------------------------------------------------------------------

#[tokio::test]
async fn quota_refusal_is_typed_and_a_freed_slot_admits_the_next_spawn() {
    let runtime = quota_runtime(
        Arc::new(FakeFactory::default()),
        SubAgentQuota::default()
            .with_max_subagents(1)
            .with_lru_eviction(false),
        Arc::new(ManualClock::new(T0)),
    );

    let first = runtime
        .spawn(SubAgentSpawnRequest::new("code_agent", "one"))
        .await
        .expect("the first spawn fits the live cap");
    runtime.wait(std::slice::from_ref(&first), WAIT).await;
    assert_eq!(runtime.capacity(), (1, 1));

    // The refusal is a typed config error with the Python message, never a panic.
    let refused = runtime
        .spawn(SubAgentSpawnRequest::new("explore_agent", "two"))
        .await
        .expect_err("the live cap refuses the second spawn");
    assert_eq!(refused.kind(), ErrorKind::Config);
    assert_eq!(
        config_message(&refused),
        "subagent capacity exceeded: used=1, limit=1"
    );
    assert_eq!(runtime.capacity(), (1, 1));
    assert_eq!(runtime.spawned_total(), 1);

    // Closing the idle instance frees the slot, so the next spawn is admitted.
    runtime
        .close(&first)
        .await
        .expect("an idle instance can be closed");
    assert_eq!(runtime.get_status(&first), SubagentStatus::Closed);
    assert_eq!(runtime.capacity(), (0, 1));

    let second = runtime
        .spawn(SubAgentSpawnRequest::new("explore_agent", "two"))
        .await
        .expect("the freed slot admits the next spawn");
    let outcomes = runtime.wait(std::slice::from_ref(&second), WAIT).await;
    assert_eq!(outcomes[0].status, SubagentStatus::Completed);
    assert_eq!(outcomes[0].output.as_deref(), Some("done: explore_agent"));
    assert_eq!(runtime.capacity(), (1, 1));
    assert_eq!(runtime.spawned_total(), 2);
}

#[tokio::test]
async fn per_parent_and_depth_limits_refuse_nested_spawns() {
    let runtime = quota_runtime(
        Arc::new(FakeFactory::default()),
        SubAgentQuota::default()
            .with_children_per_parent(1)
            .with_max_depth(2),
        Arc::new(ManualClock::new(T0)),
    );

    let root = runtime
        .spawn(SubAgentSpawnRequest::new("code_agent", "root"))
        .await
        .expect("the top-level spawn succeeds");
    runtime.wait(std::slice::from_ref(&root), WAIT).await;
    assert_eq!(
        runtime
            .get_metadata(&root)
            .expect("the root instance has metadata")
            .depth,
        1
    );

    // The root bucket is labelled by the parent session id.
    let sibling = runtime
        .spawn(SubAgentSpawnRequest::new("explore_agent", "sibling"))
        .await
        .expect_err("the root already holds its single live child");
    assert_eq!(sibling.kind(), ErrorKind::Config);
    assert_eq!(
        config_message(&sibling),
        "subagent child limit exceeded: parent=parent-s1, used=1, limit=1"
    );

    let mut nested = SubAgentSpawnRequest::new("plan_agent", "nested");
    nested.parent_subagent_id = Some(root.clone());
    let child = runtime
        .spawn(nested)
        .await
        .expect("the root's first nested child is admitted");
    runtime.wait(std::slice::from_ref(&child), WAIT).await;
    let child_metadata = runtime
        .get_metadata(&child)
        .expect("the child instance has metadata");
    assert_eq!(child_metadata.depth, 2);
    assert_eq!(
        child_metadata.parent_subagent_id.as_deref(),
        Some(root.as_str())
    );

    let mut second_child = SubAgentSpawnRequest::new("plan_agent", "second child");
    second_child.parent_subagent_id = Some(root.clone());
    let refused_child = runtime
        .spawn(second_child)
        .await
        .expect_err("the root has one live child");
    assert_eq!(refused_child.kind(), ErrorKind::Config);
    assert_eq!(
        config_message(&refused_child),
        format!("subagent child limit exceeded: parent={root}, used=1, limit=1")
    );

    let mut grandchild = SubAgentSpawnRequest::new("plan_agent", "grandchild");
    grandchild.parent_subagent_id = Some(child.clone());
    let too_deep = runtime
        .spawn(grandchild)
        .await
        .expect_err("depth 3 exceeds the nesting limit");
    assert_eq!(too_deep.kind(), ErrorKind::Config);
    assert_eq!(
        config_message(&too_deep),
        "subagent depth limit exceeded: depth=3, limit=2"
    );

    // Neither refusal left a live instance behind.
    assert_eq!(runtime.capacity(), (2, 10));
    assert_eq!(runtime.list().len(), 2);
}

#[tokio::test]
async fn activity_lifecycle_is_surfaced_by_subagent_list() {
    let (gate, factory) = GatedFactory::new();
    let clock = Arc::new(ManualClock::new(T0));
    let runtime = quota_runtime(factory, SubAgentQuota::default(), clock.clone());
    let tools = subagent_tools(Arc::clone(&runtime));
    let list = find_tool(&tools, "subagent_list");
    let ctx = ToolContext::new();

    let id = runtime
        .spawn(SubAgentSpawnRequest::new("code_agent", "lifecycle"))
        .await
        .expect("the gated spawn succeeds");

    // created: reserved in the `Queued` phase, stamped by the injected clock.
    let created = runtime
        .get_metadata(&id)
        .expect("the instance has metadata");
    assert_eq!(created.activity, SubagentActivity::Queued);
    assert_eq!(created.created_at, rfc3339_from_ms(T0));
    let queued = list
        .invoke(json!({}), &ctx)
        .await
        .expect("list tool succeeds");
    assert!(succeeded(&queued));
    assert_eq!(queued["data"]["subagents"][0]["activity"], "queued");
    assert_eq!(queued["data"]["capacity"]["running"], 0);

    // running: the gated turn owns the phase until the gate opens.
    wait_until(
        || {
            runtime
                .get_metadata(&id)
                .is_some_and(|metadata| metadata.activity == SubagentActivity::Running)
        },
        "the running activity phase",
    )
    .await;
    clock.advance(2_500);
    let running = list
        .invoke(json!({}), &ctx)
        .await
        .expect("list tool succeeds");
    assert_eq!(running["data"]["subagents"][0]["activity"], "running");
    assert_eq!(running["data"]["capacity"]["running"], 1);
    assert_eq!(running["data"]["capacity"]["max_running"], 5);
    assert_eq!(
        running["data"]["subagents"][0]["created_at"],
        rfc3339_from_ms(T0)
    );
    assert_eq!(running["data"]["summary"]["eviction"]["candidates"], 0);

    // idle: the finished turn parks the instance in the evictable phase.
    gate.send_replace(true);
    let outcomes = runtime.wait(std::slice::from_ref(&id), BOUND).await;
    assert_eq!(outcomes[0].status, SubagentStatus::Completed);
    let idle = runtime
        .get_metadata(&id)
        .expect("the instance has metadata");
    assert_eq!(idle.activity, SubagentActivity::Idle);
    assert_eq!(idle.status, SubagentStatus::Completed);
    assert_eq!(idle.last_used_at, rfc3339_from_ms(T0 + 2_500));
    let idle_list = list
        .invoke(json!({}), &ctx)
        .await
        .expect("list tool succeeds");
    assert_eq!(idle_list["data"]["subagents"][0]["activity"], "idle");
    assert_eq!(idle_list["data"]["capacity"]["running"], 0);
    assert_eq!(idle_list["data"]["summary"]["eviction"]["candidates"], 1);

    // closed: the terminal phase leaves a resumable tombstone.
    runtime.close(&id).await.expect("an idle instance closes");
    let closed = runtime
        .get_metadata(&id)
        .expect("the tombstone keeps its metadata");
    assert_eq!(closed.activity, SubagentActivity::Closed);
    assert_eq!(closed.closed_reason.as_deref(), Some("manual"));
    let closed_list = list
        .invoke(json!({}), &ctx)
        .await
        .expect("list tool succeeds");
    assert_eq!(
        closed_list["data"]["subagents"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(
        closed_list["data"]["closed_subagents"][0]["activity"],
        "closed"
    );
    assert_eq!(
        closed_list["data"]["closed_subagents"][0]["closed_reason"],
        "manual"
    );
    assert_eq!(closed_list["data"]["summary"]["live_count"], 0);
    assert_eq!(closed_list["data"]["summary"]["closed_count"], 1);
}

#[tokio::test]
async fn lru_eviction_is_reported_with_the_transcript_summary() {
    let clock = Arc::new(ManualClock::new(T0));
    let runtime = quota_runtime(
        Arc::new(FakeFactory::default()),
        SubAgentQuota::default().with_max_subagents(1),
        clock.clone(),
    );
    let tools = subagent_tools_with_transcript(Arc::clone(&runtime));
    let list = find_tool(&tools, "subagent_list");
    let ctx = ToolContext::new();

    let evicted_id = runtime
        .spawn(SubAgentSpawnRequest::new("code_agent", "first"))
        .await
        .expect("the first spawn fits the live cap");
    runtime.wait(std::slice::from_ref(&evicted_id), WAIT).await;
    assert_eq!(
        runtime
            .get_metadata(&evicted_id)
            .expect("the instance has metadata")
            .activity,
        SubagentActivity::Idle
    );
    runtime
        .record_tool_call(&evicted_id, "bash", "ls", Some(true))
        .expect("the runtime records child tool calls");
    let before = runtime
        .transcript_summary(&evicted_id)
        .expect("the transcript is retained");
    // spawn, the running status, the completion, then the recorded tool call.
    assert_eq!(before.count, 4);
    assert_eq!(before.last_kind, Some(TranscriptKind::ToolCall));

    clock.advance(1_000);
    let live_id = runtime
        .spawn(SubAgentSpawnRequest::new("explore_agent", "second"))
        .await
        .expect("the LRU policy reclaims the idle slot");
    let evicted = runtime
        .get_metadata(&evicted_id)
        .expect("the tombstone keeps its metadata");
    assert_eq!(evicted.status, SubagentStatus::Closed);
    assert_eq!(evicted.activity, SubagentActivity::Closed);
    assert_eq!(evicted.closed_reason.as_deref(), Some(EVICTION_REASON));
    assert_eq!(runtime.capacity(), (1, 1));
    assert_eq!(runtime.spawned_total(), 2);

    let outcomes = runtime.wait(std::slice::from_ref(&live_id), WAIT).await;
    assert_eq!(outcomes[0].status, SubagentStatus::Completed);

    let listed = list
        .invoke(json!({}), &ctx)
        .await
        .expect("list tool succeeds");
    assert!(succeeded(&listed));
    let data = &listed["data"];
    assert_eq!(data["capacity"]["used"], 1);
    assert_eq!(data["capacity"]["max"], 1);
    assert_eq!(data["capacity"]["spawned"], 2);
    assert_eq!(data["summary"]["live_count"], 1);
    assert_eq!(data["summary"]["closed_count"], 1);
    assert_eq!(data["summary"]["evicted_count"], 1);
    assert_eq!(data["summary"]["eviction"]["enabled"], true);

    let closed_row = &data["closed_subagents"][0];
    assert_eq!(closed_row["subagent_id"], evicted_id);
    assert_eq!(closed_row["status"], "closed");
    assert_eq!(closed_row["closed_reason"], "evicted");
    // The eviction entry is the newest transcript entry of the recycled instance.
    assert_eq!(closed_row["transcript"]["count"], 5);
    assert_eq!(closed_row["transcript"]["dropped"], 0);
    assert_eq!(closed_row["transcript"]["last_kind"], "evicted");
    assert_eq!(closed_row["transcript"]["last_content"], EVICTION_REASON);

    let live_row = &data["live_subagents"][0];
    assert_eq!(live_row["subagent_id"], live_id);
    assert_eq!(live_row["transcript"]["count"], 3);
    assert_eq!(live_row["transcript"]["last_kind"], "completion");
    assert_eq!(
        live_row["transcript"]["last_content"],
        "done: explore_agent"
    );

    // The rendering states the policy, the eviction count and both ledger rows.
    let rendered = list.render_for_llm(&listed);
    assert!(
        rendered.contains("Capacity: 1/1 subagents in use."),
        "unexpected rendering: {rendered}"
    );
    assert!(
        rendered.contains("eviction: enabled, evicted: 1"),
        "unexpected rendering: {rendered}"
    );
    assert!(rendered.contains(&evicted_id));
    assert!(rendered.contains(&live_id));
}

// ---------------------------------------------------------------------------
// B. Skills
// ---------------------------------------------------------------------------

/// Writes `<root>/<relative>/SKILL.md` with YAML front matter and returns the directory.
fn write_skill(root: &Path, relative: &str, name: &str, description: &str) -> PathBuf {
    let directory = root.join(relative);
    std::fs::create_dir_all(&directory).expect("skill directory is created");
    std::fs::write(
        directory.join(SKILL_FILENAME),
        format!("---\nname: {name}\ndescription: {description}\n---\n# {name}\n"),
    )
    .expect("SKILL.md is written");
    directory
}

/// Returns the names of `skills` in discovery order.
fn skill_names(skills: &[Skill]) -> Vec<String> {
    skills.iter().map(|skill| skill.name.clone()).collect()
}

#[tokio::test]
async fn discovery_finds_nested_skills_and_skips_directories_without_skill_file() {
    let temp = tempfile::tempdir().expect("temp skills root");
    let root = temp.path();
    let alpha_dir = write_skill(root, "alpha", "alpha", "Alpha writes production code.");
    // The front-matter `name` never overrides the directory name.
    let beta_dir = write_skill(root, "group/beta", "ignored-beta", "Beta reviews code.");
    // A directory without SKILL.md is not a skill.
    std::fs::create_dir_all(root.join("plain")).expect("plain directory is created");
    std::fs::write(root.join("plain/notes.md"), "not a skill").expect("notes are written");
    // A SKILL.md without front matter is a skill with the generated fallback description.
    let no_front = root.join("no_front");
    std::fs::create_dir_all(&no_front).expect("no_front directory is created");
    std::fs::write(no_front.join(SKILL_FILENAME), "# no front matter\n").expect("SKILL.md written");

    let library = SkillLibrary::with_roots([root.to_path_buf()]);
    let skills = library.discover().await.expect("discovery never fails");
    assert_eq!(skill_names(&skills), ["alpha", "beta", "no_front"]);

    assert_eq!(skills[0].name, "alpha");
    assert_eq!(skills[0].description, "Alpha writes production code.");
    assert_eq!(skills[0].directory, alpha_dir);
    assert_eq!(skills[0].path(), alpha_dir.join(SKILL_FILENAME));

    assert_eq!(skills[1].name, "beta");
    assert_eq!(skills[1].description, "Beta reviews code.");
    assert_eq!(skills[1].directory, beta_dir);

    assert_eq!(skills[2].name, "no_front");
    assert_eq!(
        skills[2].description,
        format!("Skill located in {}", no_front.display())
    );

    // Missing roots are skipped instead of failing discovery.
    let missing = root.join("does-not-exist");
    let none = SkillLibrary::with_roots([missing])
        .discover()
        .await
        .expect("missing roots never fail");
    assert_eq!(none, Vec::<Skill>::new());

    // Discovery reports disabled skills; only `select` filters them.
    let mut library = library;
    library.set_disabled("alpha", true);
    assert_eq!(library.discover().await.expect("discovery").len(), 3);
}

#[tokio::test]
async fn budget_limits_skill_count_and_description_characters() {
    let temp = tempfile::tempdir().expect("temp skills root");
    let root = temp.path();
    write_skill(root, "alpha", "alpha", "Alpha writes production code.");
    write_skill(
        root,
        "beta",
        "beta",
        "Beta reviews code carefully and thoroughly.",
    );

    let library = SkillLibrary::with_roots([root.to_path_buf()]);
    let skills = library.discover().await.expect("discovery never fails");
    assert_eq!(skills.len(), 2);

    let all = library.select(&skills, SkillBudget::unlimited());
    assert_eq!(skill_names(&all), ["alpha", "beta"]);
    assert_eq!(all, skills);

    let one = library.select(&skills, SkillBudget::unlimited().with_max_skills(1));
    assert_eq!(skill_names(&one), ["alpha"]);

    let none = library.select(&skills, SkillBudget::unlimited().with_max_skills(0));
    assert_eq!(none, Vec::<Skill>::new());

    // The character budget drops whole skills from the end of the selection.
    let total: usize = skills
        .iter()
        .map(|skill| skill.description.chars().count())
        .sum();
    let tight = library.select(
        &skills,
        SkillBudget::unlimited().with_max_total_chars(total - 1),
    );
    assert_eq!(skill_names(&tight), ["alpha"]);

    // A single skill over the cap is kept with a warning rather than dropped.
    let tiny = library.select(&skills, SkillBudget::unlimited().with_max_total_chars(1));
    assert_eq!(skill_names(&tiny), ["alpha"]);
}

#[tokio::test]
async fn disabled_state_filters_selection_and_persists_the_python_file_shape() {
    let temp = tempfile::tempdir().expect("temp skills root");
    let root = temp.path();
    write_skill(root, "alpha", "alpha", "Alpha writes production code.");
    write_skill(root, "beta", "beta", "Beta reviews code carefully.");

    let mut library = SkillLibrary::with_roots([root.to_path_buf()]);
    let skills = library.discover().await.expect("discovery never fails");
    assert_eq!(skills.len(), 2);
    assert!(!library.is_disabled("alpha"));

    library.set_disabled("alpha", true);
    assert!(library.is_disabled("alpha"));
    assert_eq!(
        library.disabled_names().iter().cloned().collect::<Vec<_>>(),
        ["alpha"]
    );
    assert_eq!(
        skill_names(&library.select(&skills, SkillBudget::unlimited())),
        ["beta"]
    );

    library.save_state(root).await.expect("state is saved");
    let path = root.join("skills_state.json");
    let text = std::fs::read_to_string(&path).expect("state file is readable");
    let persisted: Value = serde_json::from_str(&text).expect("state file is JSON");
    assert_eq!(
        persisted,
        json!({"skill_configs": {"alpha": {"enabled": false}}})
    );
    assert!(text.ends_with('\n'));

    let mut restored = SkillLibrary::with_roots([root.to_path_buf()]);
    restored.load_state(root).await;
    assert!(restored.is_disabled("alpha"));
    assert_eq!(restored.disabled_names(), library.disabled_names());
    assert_eq!(
        skill_names(&restored.select(&skills, SkillBudget::unlimited())),
        ["beta"]
    );

    let mut every_root = SkillLibrary::with_roots([root.to_path_buf()]);
    every_root.load_all_state().await;
    assert!(every_root.is_disabled("alpha"));

    library.set_disabled("alpha", false);
    assert!(!library.is_disabled("alpha"));
    assert!(library.disabled_names().is_empty());
}

#[test]
fn skill_root_mount_is_owned_once_and_releases_only_what_it_added() {
    let temp = tempfile::tempdir().expect("temp skills root");
    let directory = temp.path().join("skills");
    let mut roots: Vec<PathBuf> = Vec::new();

    let mut owned = SkillRootMount::mount(&mut roots, &directory);
    assert!(owned.owned());
    assert_eq!(roots.len(), 1);
    assert!(owned.is_mounted(&roots));
    assert!(owned.root().ends_with("skills"));

    let borrowed = SkillRootMount::mount(&mut roots, &directory);
    assert!(!borrowed.owned());
    assert_eq!(
        roots.len(),
        1,
        "a repeated mount must not duplicate the root"
    );
    assert!(borrowed.is_mounted(&roots));

    // `ensure_mounted` is idempotent on a mounted root.
    assert!(owned.ensure_mounted(&mut roots));
    assert_eq!(roots.len(), 1);

    // A mount that did not introduce the root never releases it.
    let mut borrowed = borrowed;
    assert!(!borrowed.unmount(&mut roots));
    assert!(owned.is_mounted(&roots));
    assert_eq!(roots.len(), 1);

    // The owning mount releases it once; repeated release is a no-op.
    assert!(owned.unmount(&mut roots));
    assert_eq!(roots, Vec::<PathBuf>::new());
    assert!(!owned.is_mounted(&roots));
    assert!(!owned.unmount(&mut roots));

    // Re-mounting after a release restores ownership.
    assert!(owned.ensure_mounted(&mut roots));
    assert!(owned.owned());
    assert_eq!(roots.len(), 1);
    assert!(owned.is_mounted(&roots));
}
