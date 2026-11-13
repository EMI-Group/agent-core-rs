use futures::stream::BoxStream;
use openjiuwen_agent::{AgentCard, AgentChunk, AgentOutput};
use openjiuwen_session::Session;
use tokio::sync::watch;

use super::*;
use crate::subagents::SubAgentSpec;

type Recorded = Arc<Mutex<Vec<(SubAgentConfig, String)>>>;
type Calls = Arc<Mutex<Vec<String>>>;

enum Mode {
    Echo,
    ErrorResult,
    Blocked(watch::Sender<bool>),
}

struct FakeAgent {
    card: AgentCard,
    calls: Calls,
    mode: Mode,
}

#[async_trait::async_trait]
impl Agent for FakeAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let query = inputs
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        self.calls.lock().push(query.clone());
        match &self.mode {
            Mode::Echo => Ok(AgentOutput::answer(format!("echo:{query}"))),
            Mode::ErrorResult => Ok(AgentOutput::error("model failed")),
            Mode::Blocked(gate) => {
                let mut gate = gate.subscribe();
                if !*gate.borrow_and_update() {
                    let _ = gate.changed().await;
                }
                Ok(AgentOutput::answer(format!("echo:{query}")))
            }
        }
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }
}

enum Behavior {
    Echo,
    Fail,
    ErrorResult,
    Blocked(watch::Sender<bool>),
}

struct FakeFactory {
    recorded: Recorded,
    calls: Calls,
    behavior: Behavior,
}

impl FakeFactory {
    fn new(behavior: Behavior) -> Arc<Self> {
        Arc::new(Self {
            recorded: Arc::default(),
            calls: Arc::default(),
            behavior,
        })
    }
}

#[async_trait::async_trait]
impl SubAgentFactory for FakeFactory {
    async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>> {
        self.recorded
            .lock()
            .push((config.clone(), session_id.to_owned()));
        let mode = match &self.behavior {
            Behavior::Echo => Mode::Echo,
            Behavior::Fail => return Err(Error::config("boom")),
            Behavior::ErrorResult => Mode::ErrorResult,
            Behavior::Blocked(gate) => Mode::Blocked(gate.clone()),
        };
        Ok(Arc::new(FakeAgent {
            card: config.agent_card.clone(),
            calls: Arc::clone(&self.calls),
            mode,
        }))
    }
}

fn build_runtime(names: &[&str], factory: Arc<FakeFactory>) -> SubAgentRuntime {
    let registry = SubAgentRegistry::new();
    for name in names {
        registry.register(SubAgentSpec::new(SubAgentConfig::new(AgentCard::new(
            *name,
        ))));
    }
    SubAgentRuntime::new(Arc::new(registry), factory, "parent-session")
}

async fn spawn(runtime: &SubAgentRuntime, subagent_type: &str, task: &str) -> SubAgentId {
    runtime
        .spawn(SubAgentSpawnRequest::new(subagent_type, task))
        .await
        .unwrap()
}

async fn wait_one(runtime: &SubAgentRuntime, id: &str) -> SubAgentOutcome {
    runtime
        .wait(&[id.to_owned()], Duration::from_secs(5))
        .await
        .pop()
        .unwrap()
}

async fn wait_running(runtime: &SubAgentRuntime, id: &str) {
    for _ in 0..200 {
        if runtime.get_status(id) == SubagentStatus::Running {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("subagent did not start: {id}");
}

fn blocked() -> (watch::Sender<bool>, Arc<FakeFactory>) {
    let (gate, _receiver) = watch::channel(false);
    (gate.clone(), FakeFactory::new(Behavior::Blocked(gate)))
}

#[tokio::test]
async fn spawn_runs_a_turn_to_completion() {
    let factory = FakeFactory::new(Behavior::Echo);
    let runtime = build_runtime(&["worker"], Arc::clone(&factory));
    let id = spawn(&runtime, "worker", "hello").await;
    assert!(matches!(
        runtime.get_status(&id),
        SubagentStatus::PendingInit | SubagentStatus::Running
    ));
    let outcome = wait_one(&runtime, &id).await;
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(outcome.output.as_deref(), Some("echo:hello"));
    assert_eq!(outcome.sub_session_id, id);
    assert_eq!(outcome.subagent_type, "worker");
    let recorded = factory.recorded.lock().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].1, id);
    assert_eq!(recorded[0].0.subagent_type.as_deref(), Some("worker"));
    let stored = runtime.get_outcome(&id).unwrap();
    assert_eq!(stored.agent_id, recorded[0].0.agent_card.id);
    assert_eq!(stored.status, SubagentStatus::Completed);
    let metadata = runtime.list();
    assert_eq!(metadata.len(), 1);
    assert_eq!(metadata[0].subagent_id, id);
    assert_eq!(metadata[0].task_description, "hello");
    assert_eq!(metadata[0].status, SubagentStatus::Completed);
    assert_ne!(metadata[0].created_at, "");
    assert_eq!(runtime.capacity(), (1, 10));
}

#[tokio::test]
async fn child_session_id_format_and_overrides() {
    let factory = FakeFactory::new(Behavior::Echo);
    let runtime = build_runtime(&["worker", "verification_agent"], Arc::clone(&factory));
    let generated = spawn(&runtime, "worker", "task").await;
    let suffix = generated
        .strip_prefix("parent-session_sub_worker_")
        .unwrap();
    assert_eq!(suffix.len(), 8);
    assert!(suffix
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    wait_one(&runtime, &generated).await;
    let sticky = spawn(&runtime, "verification_agent", "verify").await;
    assert_eq!(sticky, "parent-session_sub_verification_agent");
    let mut request = SubAgentSpawnRequest::new("worker", "custom");
    request.subagent_id = Some("explicit-id".to_owned());
    assert_eq!(runtime.spawn(request).await.unwrap(), "explicit-id");
    let recorded = factory.recorded.lock().clone();
    let sessions: Vec<&str> = recorded
        .iter()
        .map(|(_, session)| session.as_str())
        .collect();
    assert_eq!(
        sessions,
        vec![generated.as_str(), sticky.as_str(), "explicit-id"]
    );
}

#[tokio::test]
async fn spawn_validation_and_rollback() {
    let factory = FakeFactory::new(Behavior::Echo);
    let runtime = build_runtime(&["worker"], Arc::clone(&factory));
    let error = runtime
        .spawn(SubAgentSpawnRequest::new("ghost", "task"))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.contains("subagent type not found"))
    );
    let mut request = SubAgentSpawnRequest::new("worker", "first");
    request.subagent_id = Some("dup".to_owned());
    let id = runtime.spawn(request.clone()).await.unwrap();
    wait_one(&runtime, &id).await;
    let error = runtime.spawn(request).await.unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.contains("subagent already live"))
    );

    let failing = FakeFactory::new(Behavior::Fail);
    let runtime = build_runtime(&["worker"], failing);
    let mut request = SubAgentSpawnRequest::new("worker", "task");
    request.subagent_id = Some("will-fail".to_owned());
    let error = runtime.spawn(request).await.unwrap_err();
    assert!(matches!(&error, Error::Config(message) if message == "boom"));
    assert_eq!(runtime.capacity(), (0, 10));
    assert_eq!(runtime.get_status("will-fail"), SubagentStatus::NotFound);
    assert_eq!(runtime.list().len(), 0);
}

#[tokio::test]
async fn capacity_limit_and_slot_reuse() {
    let factory = FakeFactory::new(Behavior::Echo);
    let runtime = build_runtime(&["worker"], factory).with_limits(1, 1);
    let first = spawn(&runtime, "worker", "one").await;
    wait_one(&runtime, &first).await;
    let error = runtime
        .spawn(SubAgentSpawnRequest::new("worker", "two"))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::Config(message) if message == "subagent capacity exceeded: used=1, limit=1")
    );
    assert_eq!(runtime.capacity(), (1, 1));
    assert_eq!(
        runtime.close(&first).await.unwrap(),
        SubagentStatus::Completed
    );
    assert_eq!(runtime.capacity(), (0, 1));
    assert_ne!(spawn(&runtime, "worker", "two").await, first);
}

#[tokio::test]
async fn wait_times_out_while_the_agent_is_blocked() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "task").await;
    let outcomes = runtime
        .wait(std::slice::from_ref(&id), Duration::from_millis(50))
        .await;
    assert_eq!(outcomes.len(), 1);
    assert!(!outcomes[0].is_final());
    gate.send_replace(true);
    let outcome = wait_one(&runtime, &id).await;
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(outcome.output.as_deref(), Some("echo:task"));
}

#[tokio::test]
async fn error_output_maps_to_errored() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::ErrorResult));
    let id = spawn(&runtime, "worker", "task").await;
    let outcome = wait_one(&runtime, &id).await;
    assert_eq!(outcome.status, SubagentStatus::Errored);
    assert_eq!(outcome.error.as_deref(), Some("model failed"));
    assert!(outcome.output.is_none());
}

#[tokio::test]
async fn close_refuses_running_then_resume_revives() {
    let (gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "task").await;
    wait_running(&runtime, &id).await;
    let error = runtime.close(&id).await.unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.starts_with("cannot close running subagent:"))
    );
    gate.send_replace(true);
    wait_one(&runtime, &id).await;
    assert_eq!(runtime.close(&id).await.unwrap(), SubagentStatus::Completed);
    assert_eq!(runtime.close(&id).await.unwrap(), SubagentStatus::Closed);
    let report = runtime.resume(&id).await.unwrap();
    assert!(report.restored);
    assert_eq!(report.status, SubagentStatus::Completed);
    assert!(report.message.is_none());
    let live = runtime.resume(&id).await.unwrap();
    assert!(!live.restored);
    assert_eq!(
        live.message.as_deref(),
        Some("Instance is already live; use subagent_send_input directly.")
    );
    let error = runtime.resume("ghost").await.unwrap_err();
    assert!(matches!(&error, Error::Validation(message) if message.contains("subagent not found")));
}

#[tokio::test]
async fn send_input_validates_closed_and_unknown_ids() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let error = runtime
        .send_input("ghost", "query", false)
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.contains("subagent closed or not found"))
    );
    let id = spawn(&runtime, "worker", "task").await;
    wait_one(&runtime, &id).await;
    runtime.close(&id).await.unwrap();
    let error = runtime.send_input(&id, "query", false).await.unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.contains("subagent closed or not found"))
    );
}

#[tokio::test]
async fn send_input_after_completion_starts_a_new_turn() {
    let factory = FakeFactory::new(Behavior::Echo);
    let calls = Arc::clone(&factory.calls);
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "first").await;
    wait_one(&runtime, &id).await;
    runtime.send_input(&id, "second", false).await.unwrap();
    assert_eq!(runtime.get_status(&id), SubagentStatus::PendingInit);
    assert_eq!(
        wait_one(&runtime, &id).await.output.as_deref(),
        Some("echo:second")
    );
    assert_eq!(*calls.lock(), vec!["first".to_owned(), "second".to_owned()]);
}

#[tokio::test]
async fn queued_inputs_run_in_fifo_order() {
    let (gate, factory) = blocked();
    let calls = Arc::clone(&factory.calls);
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "task").await;
    wait_running(&runtime, &id).await;
    runtime.send_input(&id, "one", false).await.unwrap();
    runtime.send_input(&id, "two", false).await.unwrap();
    gate.send_replace(true);
    assert_eq!(
        wait_one(&runtime, &id).await.output.as_deref(),
        Some("echo:two")
    );
    assert_eq!(
        *calls.lock(),
        vec!["task".to_owned(), "one".to_owned(), "two".to_owned()]
    );
}

#[tokio::test]
async fn interrupt_aborts_the_running_turn_and_runs_the_queued_query() {
    let (gate, factory) = blocked();
    let calls = Arc::clone(&factory.calls);
    let runtime = build_runtime(&["worker"], factory);
    let id = spawn(&runtime, "worker", "original").await;
    wait_running(&runtime, &id).await;
    runtime.send_input(&id, "next", true).await.unwrap();
    assert_eq!(runtime.get_status(&id), SubagentStatus::Interrupted);
    gate.send_replace(true);
    let outcome = wait_one(&runtime, &id).await;
    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(outcome.output.as_deref(), Some("echo:next"));
    assert_eq!(
        *calls.lock(),
        vec!["original".to_owned(), "next".to_owned()]
    );
}

#[tokio::test]
async fn cancel_all_closes_live_instances_and_describe_list_separates_them() {
    let (_gate, factory) = blocked();
    let runtime = build_runtime(&["worker"], factory);
    let first = spawn(&runtime, "worker", "one").await;
    let second = spawn(&runtime, "worker", "two").await;
    wait_running(&runtime, &first).await;
    wait_running(&runtime, &second).await;
    let live = runtime.describe_list();
    assert_eq!(live["summary"]["live_count"], json!(2));
    assert_eq!(live["subagents"].as_array().map(Vec::len), Some(2));
    assert_eq!(live["live_subagents"].as_array().map(Vec::len), Some(2));
    assert_eq!(live["closed_subagents"].as_array().map(Vec::len), Some(0));
    assert_eq!(live["capacity"]["max"], json!(10));
    assert_eq!(
        runtime.cancel_all().await,
        vec![first.clone(), second.clone()]
    );
    assert_eq!(runtime.get_status(&first), SubagentStatus::Closed);
    assert_eq!(runtime.get_status(&second), SubagentStatus::Closed);
    let list = runtime.describe_list();
    assert_eq!(list["capacity"]["used"], json!(0));
    assert_eq!(list["subagents"].as_array().map(Vec::len), Some(0));
    assert_eq!(list["live_subagents"].as_array().map(Vec::len), Some(0));
    assert_eq!(list["closed_subagents"].as_array().map(Vec::len), Some(2));
    assert_eq!(list["summary"], json!({"live_count": 0, "closed_count": 2}));
}

#[tokio::test]
async fn unknown_ids_report_not_found() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    assert_eq!(runtime.get_status("ghost"), SubagentStatus::NotFound);
    assert!(runtime.get_outcome("ghost").is_none());
    let outcomes = runtime
        .wait(&["ghost".to_owned()], Duration::from_millis(50))
        .await;
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].subagent_id, "ghost");
    assert_eq!(outcomes[0].status, SubagentStatus::NotFound);
    assert!(outcomes[0].is_final());
    assert_eq!(runtime.parent_session_id(), "parent-session");
}

#[tokio::test]
async fn duplicate_wait_ids_are_collapsed() {
    let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
    let id = spawn(&runtime, "worker", "task").await;
    let outcomes = runtime
        .wait(&[id.clone(), id.clone()], Duration::from_secs(5))
        .await;
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, SubagentStatus::Completed);
}

#[test]
fn status_wire_strings_and_finality() {
    let cases = [
        ("pending_init", SubagentStatus::PendingInit, false),
        ("running", SubagentStatus::Running, false),
        ("completed", SubagentStatus::Completed, true),
        ("interrupted", SubagentStatus::Interrupted, false),
        ("errored", SubagentStatus::Errored, true),
        ("closed", SubagentStatus::Closed, true),
        ("not_found", SubagentStatus::NotFound, true),
    ];
    for (wire, status, final_) in cases {
        assert_eq!(status.as_str(), wire);
        assert_eq!(status.is_final(), final_);
    }
    let outcome = SubAgentOutcome {
        status: SubagentStatus::Completed,
        ..SubAgentOutcome::default()
    };
    assert!(outcome.is_final());
}

mod quota;
