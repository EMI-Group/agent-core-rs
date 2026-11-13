//! `metadata["organization"]` → chart, org task events, mailbox acknowledgement and delegation.
//!
//! The organisation seam is one optional block on the team spec: a chart (teams, members and
//! reporting lines) plus the delegation rules layered over it. When the block is absent the flat
//! in-process team is untouched; when it is present but broken the build fails; when it is valid the
//! build installs one [`OrgBinding`](crate::runtime::org_bind::OrgBinding) that carries the chart,
//! the delegation router, the cross-team task pool and the org mailbox.
//!
//! These tests drive that block end to end: the installed binding, the inert unconfigured team, the
//! two invalid-chart refusals, the `org_task` bus copy of a pool mutation, the leader-loop mailbox
//! drain and both sides of the delegation router.
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessContext, HarnessError,
    HarnessEventCursor, HarnessInput, HarnessProtocol, HarnessResult, HarnessState, SendReceipt,
    VecEventCursor,
};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::harness::{MemberHarnessContext, MemberHarnessFactory};
use crate::messager::{EventSink, Messager};
use crate::observability::{ObservabilityHooks, ObservableEvent, RecordingHooks};
use crate::org::OrgTaskSpec;
use crate::runtime::org_bind::ORG_TASK_EVENT;
use crate::runtime::team_agent::SpecBuild;
use crate::schema::{DeepAgentSpec, EventMessage, TaskGraphSpec, TeamAgentSpec, TeamTopic};

use super::{
    blueprint, deps, wait_until, wire, with_metadata, RecordingWorkspaceProvider, ScriptedFactory,
};

/// The chart every organisation test layers under `{"spec": …}`.
///
/// Two teams (`alpha` led by `boss`, `beta` led by `other`) and one declared member per team: the
/// smallest chart whose leaders resolve and whose reporting lines are all roots.
fn chart() -> Value {
    json!({
        "organization_id": "acme",
        "teams": [
            { "team_name": "alpha", "leader": "boss" },
            { "team_name": "beta", "leader": "other" }
        ],
        "members": [
            { "member_id": "boss", "team_name": "alpha" },
            { "member_id": "other", "team_name": "beta" }
        ]
    })
}

/// Returns the organisation block for `rules` over [`chart`].
fn block(rules: &Value) -> Value {
    json!({ "spec": chart(), "rules": rules })
}

/// Returns the blueprint under test carrying `block` as `metadata["organization"]`.
fn organized(block: Value) -> TeamAgentSpec {
    with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "organization",
        block,
    )
}

/// Builds `spec` over fresh fakes without unwrapping, so an invalid chart can be asserted.
async fn build_error(spec: &TeamAgentSpec) -> Error {
    let factory = ScriptedFactory::new();
    let workspace = RecordingWorkspaceProvider::new();
    match spec.build_with(deps(&factory, &workspace)).await {
        Ok(_) => panic!("the build accepted an invalid organisation"),
        Err(error) => error,
    }
}

/// A shared log of the events one recording bus sink observed.
type EventLog = Arc<Mutex<Vec<EventMessage>>>;

/// Subscribes an `observer` sink on `topics` and returns the shared event log.
fn record_events(messager: &Arc<dyn Messager>, topics: &[TeamTopic]) -> EventLog {
    let log: EventLog = Arc::new(Mutex::new(Vec::new()));
    let sink: EventSink = {
        let log = Arc::clone(&log);
        Arc::new(move |event| log.lock().push(event))
    };
    messager
        .subscribe("observer", topics, sink)
        .expect("the event observer could not subscribe");
    log
}

/// Returns a snapshot of `log`.
fn recorded(log: &EventLog) -> Vec<EventMessage> {
    log.lock().clone()
}

/// Factory that gives every member a harness whose `send` is always rejected.
struct FailingFactory {
    /// Harness built per member, so the test can inspect the leader's attempts.
    harnesses: Mutex<BTreeMap<String, Arc<FailingHarness>>>,
}
impl FailingFactory {
    /// Builds an empty factory.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            harnesses: Mutex::new(BTreeMap::new()),
        })
    }
    /// Returns the harness built for `member`, when it was built.
    fn harness(&self, member: &str) -> Option<Arc<FailingHarness>> {
        self.harnesses.lock().get(member).cloned()
    }
}
#[async_trait]
impl MemberHarnessFactory for FailingFactory {
    async fn create(
        &self,
        _spec: &DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>> {
        let harness = FailingHarness::new();
        self.harnesses
            .lock()
            .insert(ctx.member_name.clone(), Arc::clone(&harness));
        Ok(harness)
    }
    fn name(&self) -> &'static str {
        "failing"
    }
}
/// Idle harness that rejects every `send`, as a runtime with no running cycle would.
struct FailingHarness {
    /// State reported by [`HarnessProtocol::state`]; always `idle`, never terminated.
    state: Mutex<HarnessState>,
    /// Static capability card.
    card: HarnessCard,
    /// Observation buffer policy.
    buffer: EventBufferConfig,
    /// Texts the harness *accepted* (never, because `send` always fails).
    sends: Mutex<Vec<String>>,
    /// Number of rejected `send` calls.
    rejected: Mutex<usize>,
}
impl FailingHarness {
    /// Builds an idle, always-rejecting harness.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(HarnessState::Idle),
            card: HarnessCard::new("failing", "0.0.0"),
            buffer: EventBufferConfig::default(),
            sends: Mutex::new(Vec::new()),
            rejected: Mutex::new(0),
        })
    }
    /// Returns the texts the harness accepted.
    fn sends(&self) -> Vec<String> {
        self.sends.lock().clone()
    }
    /// Returns how many inputs were rejected.
    fn rejected(&self) -> usize {
        *self.rejected.lock()
    }
}
#[async_trait]
#[allow(clippy::needless_pass_by_value)]
impl HarnessProtocol for FailingHarness {
    fn card(&self) -> &HarnessCard {
        &self.card
    }
    fn state(&self) -> HarnessState {
        *self.state.lock()
    }
    fn provider_session_id(&self) -> Option<&str> {
        None
    }
    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.buffer
    }
    async fn start(&self, _context: HarnessContext) -> HarnessResult<()> {
        *self.state.lock() = HarnessState::Idle;
        Ok(())
    }
    async fn stop(&self) -> HarnessResult<()> {
        *self.state.lock() = HarnessState::Terminated;
        Ok(())
    }
    fn events(&self) -> Box<dyn HarnessEventCursor> {
        Box::new(VecEventCursor::new(Vec::new()))
    }
    fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        Box::new(VecEventCursor::new(Vec::new()))
    }
    async fn send(&self, _input: HarnessInput, _mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        *self.rejected.lock() += 1;
        Err(HarnessError::state("no cycle is running"))
    }
    async fn abort(&self, _mode: AbortMode) -> HarnessResult<()> {
        Ok(())
    }
    async fn pause(&self) -> HarnessResult<()> {
        *self.state.lock() = HarnessState::Paused;
        Ok(())
    }
    async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
        *self.state.lock() = HarnessState::Running;
        Ok(())
    }
    async fn export_checkpoint(
        &self,
    ) -> HarnessResult<Option<openjiuwen_harness_protocol::HarnessCheckpoint>> {
        Ok(None)
    }
}

#[tokio::test]
async fn configured_organisation_is_installed_on_the_build() {
    let spec = organized(block(&json!({
        "require_reporting_line": false,
        "max_delegation_depth": 3
    })));
    let wiring = wire(&spec, |deps| deps).await;

    let org = wiring
        .agent
        .org()
        .expect("the build did not install the organisation block");
    assert_eq!(org.organization_id(), "acme");

    let teams: Vec<String> = org.teams().into_iter().map(|(name, _)| name).collect();
    assert_eq!(teams, vec!["alpha".to_owned(), "beta".to_owned()]);

    let rules = org.rules();
    assert!(!rules.require_reporting_line);
    assert_eq!(rules.max_delegation_depth, 3);
    assert_eq!(org.spec().organization_id, "acme");
}

#[tokio::test]
async fn unconfigured_team_keeps_the_flat_board() {
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| deps).await;
    let agent = &wiring.agent;
    assert!(agent.org().is_none());

    let error = agent
        .create_org_task(OrgTaskSpec::new("alpha", "ship", "do it"))
        .await
        .expect_err("an unconfigured team created an org task");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.message().contains("organisation"),
        "unexpected message: {}",
        error.message()
    );

    let error = agent
        .delegate_org_task("task_missing", "beta", None)
        .await
        .expect_err("an unconfigured team delegated an org task");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.message().contains("organisation"),
        "unexpected message: {}",
        error.message()
    );

    let task = agent
        .create_task(TaskGraphSpec {
            title: "flat task".to_owned(),
            content: "the board still works".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .expect("the flat board refused a normal task");
    assert_eq!(task.title, "flat task");
}

#[tokio::test]
async fn invalid_charts_fail_the_build() {
    // A team whose declared leader is not a member of that team.
    let unknown_leader = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "organization",
        json!({
            "spec": {
                "organization_id": "acme",
                "teams": [
                    { "team_name": "alpha", "leader": "ghost" },
                    { "team_name": "beta", "leader": "other" }
                ],
                "members": [
                    { "member_id": "boss", "team_name": "alpha" },
                    { "member_id": "other", "team_name": "beta" }
                ]
            }
        }),
    );
    let error = build_error(&unknown_leader).await;
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("leader 'ghost'"),
        "unexpected message: {}",
        error.message()
    );

    // A reporting cycle between the two teams.
    let cycle = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "organization",
        json!({
            "spec": {
                "organization_id": "acme",
                "teams": [
                    { "team_name": "alpha", "leader": "boss", "reports_to": "beta" },
                    { "team_name": "beta", "leader": "other", "reports_to": "alpha" }
                ],
                "members": [
                    { "member_id": "boss", "team_name": "alpha" },
                    { "member_id": "other", "team_name": "beta" }
                ]
            }
        }),
    );
    let error = build_error(&cycle).await;
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("cycle"),
        "unexpected message: {}",
        error.message()
    );

    // A JSON null is "not configured", never a broken chart.
    let null_block = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "organization",
        Value::Null,
    );
    let wiring = wire(&null_block, |deps| deps).await;
    assert!(wiring.agent.org().is_none());
}

#[tokio::test]
async fn create_org_task_publishes_an_org_task_event() {
    let spec = organized(block(&json!({
        "require_reporting_line": false,
        "max_delegation_depth": 3
    })));
    let wiring = wire(&spec, |deps| deps).await;
    let events = record_events(wiring.agent.backend().messager(), &[TeamTopic::Task]);

    let created = wiring
        .agent
        .create_org_task(OrgTaskSpec::new("alpha", "ship", "do it"))
        .await
        .expect("the org task was not created");
    let task_id = created.task.task_id.clone();
    assert_ne!(task_id, "");

    wait_until("the org task event", || {
        recorded(&events).iter().any(|event| {
            event.event_type == ORG_TASK_EVENT
                && event.sender_id == "alpha"
                && event.payload.get("type").and_then(Value::as_str) == Some("created")
                && event.payload.get("task_id").and_then(Value::as_str) == Some(task_id.as_str())
        })
    })
    .await;
}

// The negative half of the acknowledgement contract: a leader turn whose `send` fails must leave the
// delivered org messages unacknowledged, so the next run retries them. The leader loop flushes the
// drained mailbox into the steering batch, and the acknowledgement only runs when that flush
// returned `true` (`src/runtime/leader.rs`, the settle tick and the post-loop `drop(events_keeper)`
// block). `flush` restores a rejected batch, so the post-loop flush fails again instead of reporting
// an empty batch as success — which is what makes the unacknowledged state observable here.
#[tokio::test]
async fn a_failed_leader_flush_leaves_the_organisation_mail_unacknowledged() {
    let spec = organized(block(&json!({ "require_reporting_line": false })));
    let factory = FailingFactory::new();
    let injected: Arc<dyn MemberHarnessFactory> =
        Arc::clone(&factory) as Arc<dyn MemberHarnessFactory>;
    let wiring = wire(&spec, move |deps| deps.with_harness_factory(injected)).await;
    let org = wiring
        .agent
        .org()
        .expect("the build did not install the organisation block");

    org.send_mail("beta", "alpha", "subject", "body")
        .await
        .expect("the org message was not sent");
    assert_eq!(
        org.unread("alpha").await.expect("the inbox failed").len(),
        1
    );

    // No leader query, so the only `send` the run performs is the org flush.
    let output = wiring
        .agent
        .invoke(Value::Null, None)
        .await
        .expect("the team run did not settle");
    assert_eq!(output.output, "");

    let leader = factory
        .harness("boss")
        .expect("no leader harness was built");
    assert!(
        leader.rejected() > 0,
        "the run never tried to flush the drained mailbox"
    );
    assert!(
        leader.sends().is_empty(),
        "a rejected steering batch reached the harness: {:?}",
        leader.sends()
    );
    let unread = org.unread("alpha").await.expect("the inbox failed");
    assert_eq!(
        unread.len(),
        1,
        "a failed flush acknowledged mail that never reached the leader: {unread:?}"
    );
    assert_eq!(unread[0].body, "body", "the message must not be lost");
}

#[tokio::test]
async fn mailbox_is_drained_and_acknowledged_after_the_flush() {
    let spec = organized(block(&json!({ "require_reporting_line": false })));
    let wiring = wire(&spec, |deps| deps).await;
    let org = wiring
        .agent
        .org()
        .expect("the build did not install the organisation block");

    org.send_mail("beta", "alpha", "subject", "body")
        .await
        .expect("the org message was not sent");
    assert_eq!(
        org.unread("alpha").await.expect("the inbox failed").len(),
        1
    );

    wiring
        .agent
        .invoke(Value::Null, None)
        .await
        .expect("the team run failed");

    let leader = wiring
        .factory
        .harness("boss")
        .expect("no leader harness was built");
    let sends = leader.sends();
    assert!(
        sends.iter().any(|line| line.starts_with("org beta:")),
        "leader sends: {sends:?}"
    );
    assert!(
        org.unread("alpha")
            .await
            .expect("the inbox failed")
            .is_empty(),
        "the delivered message was not acknowledged"
    );
}

#[tokio::test]
async fn refused_delegation_leaves_the_task_untouched() {
    let hooks = Arc::new(RecordingHooks::new());
    let injected_hooks = Arc::clone(&hooks);
    let injected: Arc<dyn ObservabilityHooks> = injected_hooks;
    let spec = organized(block(&json!({ "require_reporting_line": true })));
    let wiring = wire(&spec, move |deps| deps.with_observability_hook(injected)).await;
    let org = wiring
        .agent
        .org()
        .expect("the build did not install the organisation block");

    let created = wiring
        .agent
        .create_org_task(OrgTaskSpec::new("alpha", "ship", "do it"))
        .await
        .expect("the org task was not created");
    let task_id = created.task.task_id.clone();
    hooks.clear();

    let error = wiring
        .agent
        .delegate_org_task(&task_id, "beta", None)
        .await
        .expect_err("the router allowed a delegation without a reporting line");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("reporting line"),
        "unexpected message: {}",
        error.message()
    );

    let task = org
        .task(&task_id)
        .await
        .expect("the task lookup failed")
        .expect("the task vanished");
    assert_eq!(task.owner_team, "alpha");
    assert_eq!(task.target_team, "alpha");
    assert!(task.assignee.is_none());
    assert!(
        !hooks.kinds().contains(&"delegation"),
        "a refused hop recorded a delegation: {:?}",
        hooks.kinds()
    );
}

#[tokio::test]
async fn allowed_delegation_records_the_hook_and_publishes_the_event() {
    let hooks = Arc::new(RecordingHooks::new());
    let injected_hooks = Arc::clone(&hooks);
    let injected: Arc<dyn ObservabilityHooks> = injected_hooks;
    let spec = organized(block(&json!({
        "require_reporting_line": false,
        "max_delegation_depth": 3
    })));
    let wiring = wire(&spec, move |deps| deps.with_observability_hook(injected)).await;
    let events = record_events(wiring.agent.backend().messager(), &[TeamTopic::Task]);
    let org = wiring
        .agent
        .org()
        .expect("the build did not install the organisation block");

    let created = wiring
        .agent
        .create_org_task(OrgTaskSpec::new("alpha", "ship", "do it"))
        .await
        .expect("the org task was not created");
    let task_id = created.task.task_id.clone();
    hooks.clear();

    let result = wiring
        .agent
        .delegate_org_task(&task_id, "beta", None)
        .await
        .expect("the delegation was refused");
    assert!(result.is_ok(), "the delegation was not applied");

    let reasons: Vec<Option<String>> = hooks
        .events()
        .iter()
        .filter_map(|event| match &event.event {
            ObservableEvent::Delegation(delegation) => Some(delegation.reason.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(reasons, vec![Some("allowed".to_owned())]);

    wait_until("the delegated org task event", || {
        recorded(&events).iter().any(|event| {
            event.event_type == ORG_TASK_EVENT
                && event.payload.get("type").and_then(Value::as_str) == Some("delegated")
                && event.payload.get("task_id").and_then(Value::as_str) == Some(task_id.as_str())
        })
    })
    .await;

    let task = org
        .task(&task_id)
        .await
        .expect("the task lookup failed")
        .expect("the task vanished");
    assert_eq!(task.target_team, "beta");
}
