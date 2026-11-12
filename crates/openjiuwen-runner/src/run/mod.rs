//! The runner's six execution entry points and everything they share.
//!
//! Ports the execution half of Python `openjiuwen/core/runner/runner.py`
//! (`run_agent`, `run_agent_streaming`, `run_workflow`,
//! `run_workflow_streaming`, `_prepare_agent`, `_prepare_workflow`,
//! `_create_agent_session`) and the in-process path of `team_runner.py`
//! (`_run_base_team`).
//!
//! The module is deliberately split by responsibility so each file stays small
//! and readable:
//!
//! - [`agent`] — [`invoke_agent`] / [`stream_agent`];
//! - [`workflow`] — [`invoke_workflow`] / [`stream_workflow`];
//! - [`team`] — [`invoke_team`] / [`stream_team`];
//! - `span` — the crate-private [`RunSpan`] that mirrors a run onto the
//!   session tracer;
//! - `teardown` — the crate-private [`Teardown`] that runs an owned teardown
//!   action exactly once, on completion or on drop.
//!
//! Reference resolution happens before anything else: an id that is not
//! registered fails with [`openjiuwen_common::error::Error::NotFound`] before a
//! session is created or a callback is fired.
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use openjiuwen_agent::{Agent, LlmAgent, ReActAgent, WorkflowAgent};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_multiagent::{BaseTeam, Team};
use openjiuwen_security::GuardrailPipeline;
use openjiuwen_session::{AgentTeamSession, Checkpointer, Session, StreamMode};
use openjiuwen_workflow::Workflow;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::callback::CallbackFramework;
use crate::config::{RunnerConfig, CONVERSATION_ID_KEY};
use crate::resources::ResourceMgr;

mod agent;
mod span;
mod team;
mod teardown;
mod workflow;

pub use agent::{invoke_agent, stream_agent};
pub use team::{invoke_team, stream_team};
pub use workflow::{invoke_workflow, stream_workflow};

pub(crate) use span::RunSpan;
pub(crate) use teardown::Teardown;

/// `tracing` target every execution-path log uses.
pub(crate) const LOG_TARGET: &str = "openjiuwen_runner::run";

/// The owned execution context the facade hands down to one run.
///
/// It is cheap to clone (facts and `Arc`s only) and carries exactly what the
/// execution paths need: the registries for id resolution, the callback
/// framework for lifecycle events, the runner configuration (session-id and
/// deadline defaults), the checkpointer created by `Runner::start`, and the
/// guardrails bound for the run.
#[derive(Clone)]
pub struct RunServices {
    /// Registries resolving an id-based [`AgentRef`] / [`WorkflowRef`] /
    /// [`TeamRef`].
    pub resources: Arc<ResourceMgr>,
    /// Framework firing this crate's lifecycle events.
    pub callbacks: Arc<CallbackFramework>,
    /// Runner configuration (defaults for session id and run deadline).
    pub config: RunnerConfig,
    /// Checkpointer injected into the sessions the runner creates.
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Guardrails bound for the run, when any.
    pub guardrails: Option<Arc<GuardrailPipeline>>,
}

impl RunServices {
    /// Builds the execution context from its four required parts.
    #[must_use]
    pub fn new(
        resources: Arc<ResourceMgr>,
        callbacks: Arc<CallbackFramework>,
        config: RunnerConfig,
        checkpointer: Option<Arc<dyn Checkpointer>>,
    ) -> Self {
        Self {
            resources,
            callbacks,
            config,
            checkpointer,
            guardrails: None,
        }
    }

    /// Attaches the guardrails bound for the run.
    #[must_use]
    pub fn with_guardrails(mut self, pipeline: Arc<GuardrailPipeline>) -> Self {
        self.guardrails = Some(pipeline);
        self
    }
}

/// Per-run behaviour: session ownership, cancellation, deadline, and overrides.
///
/// [`Default`] means: no caller session, no deadline (the runner's
/// `default_run_timeout` still applies when configured), never cancelled, and a
/// session id taken from `inputs["conversation_id"]` else
/// [`RunnerConfig::default_session_id`].
#[derive(Clone)]
pub struct RunOptions {
    /// Caller-owned session used instead of creating one.
    ///
    /// The runner never commits, closes, or ends it.
    pub session: Option<Arc<dyn Session>>,
    /// Caller-owned team session used instead of creating one.
    pub team_session: Option<Arc<AgentTeamSession>>,
    /// Session id requested by the caller (`session="id"` in Python).
    pub session_id: Option<String>,
    /// Cancels the run when triggered.
    pub cancellation: CancellationToken,
    /// Per-run deadline; `None` falls back to
    /// [`RunnerConfig::default_run_timeout`] for agents and means "no deadline"
    /// for workflows (the engine owns its own deadline).
    pub timeout: Option<Duration>,
    /// Frame modes requested from a workflow stream.
    pub stream_modes: Option<Vec<StreamMode>>,
    /// Environment entries applied to sessions the runner creates.
    pub envs: BTreeMap<String, Value>,
    /// Checkpointer overriding the runner's for this run's sessions.
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Metadata copied into sessions the runner creates.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            session: None,
            team_session: None,
            session_id: None,
            cancellation: CancellationToken::new(),
            timeout: None,
            stream_modes: None,
            envs: BTreeMap::new(),
            checkpointer: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl fmt::Debug for RunOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunOptions")
            .field(
                "session",
                &self.session.as_ref().map(|session| session.id().to_owned()),
            )
            .field(
                "team_session",
                &self
                    .team_session
                    .as_ref()
                    .map(|session| session.id().to_owned()),
            )
            .field("session_id", &self.session_id)
            .field("cancellation", &self.cancellation)
            .field("timeout", &self.timeout)
            .field("stream_modes", &self.stream_modes)
            .field("envs", &self.envs.keys().collect::<Vec<_>>())
            .field("checkpointer", &self.checkpointer.is_some())
            .field("metadata", &self.metadata.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl RunOptions {
    /// Uses `session` instead of creating one; the runner will not commit it.
    #[must_use]
    pub fn with_session(mut self, session: Arc<dyn Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Uses `session` for a team run instead of creating one.
    #[must_use]
    pub fn with_team_session(mut self, session: Arc<AgentTeamSession>) -> Self {
        self.team_session = Some(session);
        self
    }

    /// Requests `session_id` for a session the runner creates.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Cancels the run when `cancellation` is triggered.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Bounds the run with `timeout`.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Requests `modes` from a workflow stream.
    #[must_use]
    pub fn with_stream_modes(mut self, modes: Vec<StreamMode>) -> Self {
        self.stream_modes = Some(modes);
        self
    }

    /// Adds one environment entry applied to created sessions.
    #[must_use]
    pub fn with_envs(mut self, key: impl Into<String>, value: Value) -> Self {
        self.envs.insert(key.into(), value);
        self
    }

    /// Overrides the checkpointer used by sessions created for this run.
    #[must_use]
    pub fn with_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Adds one metadata entry copied into created sessions.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A single agent to run: either a registry id or a live instance.
#[derive(Clone)]
pub enum AgentRef {
    /// An id resolved through [`ResourceMgr`] on every run (the provider is
    /// invoked once per lookup).
    Id(String),
    /// A live instance used as-is.
    Agent(Arc<dyn Agent>),
}

impl AgentRef {
    /// References the agent registered as `id`.
    #[must_use]
    pub fn id(id: impl Into<String>) -> Self {
        Self::Id(id.into())
    }

    /// References a live agent instance.
    #[must_use]
    pub fn of(agent: Arc<dyn Agent>) -> Self {
        Self::Agent(agent)
    }
}

impl fmt::Debug for AgentRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id(id) => f.debug_tuple("Id").field(id).finish(),
            Self::Agent(agent) => f.debug_tuple("Agent").field(&agent.card().id).finish(),
        }
    }
}

impl From<&str> for AgentRef {
    fn from(id: &str) -> Self {
        Self::Id(id.to_owned())
    }
}

impl From<String> for AgentRef {
    fn from(id: String) -> Self {
        Self::Id(id)
    }
}

impl From<Arc<dyn Agent>> for AgentRef {
    fn from(agent: Arc<dyn Agent>) -> Self {
        Self::Agent(agent)
    }
}

impl From<Arc<ReActAgent>> for AgentRef {
    fn from(agent: Arc<ReActAgent>) -> Self {
        Self::Agent(agent)
    }
}

impl From<Arc<LlmAgent>> for AgentRef {
    fn from(agent: Arc<LlmAgent>) -> Self {
        Self::Agent(agent)
    }
}

impl From<Arc<WorkflowAgent>> for AgentRef {
    fn from(agent: Arc<WorkflowAgent>) -> Self {
        Self::Agent(agent)
    }
}

/// A single workflow to run: either a registry id or a live instance.
#[derive(Clone)]
pub enum WorkflowRef {
    /// An id resolved through [`ResourceMgr`] on every run.
    Id(String),
    /// A live workflow used as-is.
    Workflow(Arc<Workflow>),
}

impl WorkflowRef {
    /// References the workflow registered as `id`.
    #[must_use]
    pub fn id(id: impl Into<String>) -> Self {
        Self::Id(id.into())
    }

    /// References a live workflow instance.
    #[must_use]
    pub fn of(workflow: Arc<Workflow>) -> Self {
        Self::Workflow(workflow)
    }
}

impl fmt::Debug for WorkflowRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id(id) => f.debug_tuple("Id").field(id).finish(),
            Self::Workflow(workflow) => f
                .debug_tuple("Workflow")
                .field(&workflow.card().id)
                .finish(),
        }
    }
}

impl From<&str> for WorkflowRef {
    fn from(id: &str) -> Self {
        Self::Id(id.to_owned())
    }
}

impl From<String> for WorkflowRef {
    fn from(id: String) -> Self {
        Self::Id(id)
    }
}

impl From<Arc<Workflow>> for WorkflowRef {
    fn from(workflow: Arc<Workflow>) -> Self {
        Self::Workflow(workflow)
    }
}

/// A single agent team to run: either a registry id or a live instance.
#[derive(Clone)]
pub enum TeamRef {
    /// An id resolved through [`ResourceMgr`] on every run.
    Id(String),
    /// A live team used as-is.
    Team(Arc<dyn Team>),
}

impl TeamRef {
    /// References the team registered as `id`.
    #[must_use]
    pub fn id(id: impl Into<String>) -> Self {
        Self::Id(id.into())
    }

    /// References a live team instance.
    #[must_use]
    pub fn of(team: Arc<dyn Team>) -> Self {
        Self::Team(team)
    }
}

impl fmt::Debug for TeamRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id(id) => f.debug_tuple("Id").field(id).finish(),
            Self::Team(team) => f.debug_tuple("Team").field(&team.card().id).finish(),
        }
    }
}

impl From<&str> for TeamRef {
    fn from(id: &str) -> Self {
        Self::Id(id.to_owned())
    }
}

impl From<String> for TeamRef {
    fn from(id: String) -> Self {
        Self::Id(id)
    }
}

impl From<Arc<dyn Team>> for TeamRef {
    fn from(team: Arc<dyn Team>) -> Self {
        Self::Team(team)
    }
}

impl From<Arc<BaseTeam>> for TeamRef {
    fn from(team: Arc<BaseTeam>) -> Self {
        Self::Team(team)
    }
}

/// Resolves `reference`, invoking the registry provider at most once.
///
/// # Errors
///
/// [`crate::error::not_found`] (`NotFound`) when the id is not registered.
pub(crate) fn resolve_agent(
    services: &RunServices,
    reference: &AgentRef,
) -> Result<Arc<dyn Agent>> {
    match reference {
        AgentRef::Id(id) => services
            .resources
            .agent(id)
            .ok_or_else(|| crate::error::not_found("agent", id)),
        AgentRef::Agent(agent) => Ok(Arc::clone(agent)),
    }
}

/// Resolves `reference` to a live workflow.
///
/// # Errors
///
/// [`crate::error::not_found`] (`NotFound`) when the id is not registered.
pub(crate) fn resolve_workflow(
    services: &RunServices,
    reference: &WorkflowRef,
) -> Result<Arc<Workflow>> {
    match reference {
        WorkflowRef::Id(id) => services
            .resources
            .workflow(id)
            .map(Arc::new)
            .ok_or_else(|| crate::error::not_found("workflow", id)),
        WorkflowRef::Workflow(workflow) => Ok(Arc::clone(workflow)),
    }
}

/// Resolves `reference` to a live team.
///
/// # Errors
///
/// [`crate::error::not_found`] (`NotFound`) when the id is not registered.
pub(crate) fn resolve_team(services: &RunServices, reference: &TeamRef) -> Result<Arc<dyn Team>> {
    match reference {
        TeamRef::Id(id) => services
            .resources
            .team(id)
            .ok_or_else(|| crate::error::not_found("team", id)),
        TeamRef::Team(team) => Ok(Arc::clone(team)),
    }
}

/// Resolves the session id of a run, in Python's precedence order.
///
/// `inputs["conversation_id"]` when it is a non-empty string, else
/// `options.session_id` (an empty string counts as absent), else
/// `config.default_session_id`.
pub(crate) fn resolve_session_id(
    inputs: &Value,
    options: &RunOptions,
    config: &RunnerConfig,
) -> String {
    inputs
        .get(CONVERSATION_ID_KEY)
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| options.session_id.clone().filter(|id| !id.is_empty()))
        .unwrap_or_else(|| config.default_session_id.clone())
}

/// Runs `future` under a cancellation token and an optional deadline.
///
/// The select is biased to the token first, so a cancelled run cannot win the
/// race by completing on the same poll. A deadline expiry returns
/// [`crate::error::run_timeout`] for `operation`.
pub(crate) async fn run_with_limits<F, T>(
    operation: &str,
    future: F,
    cancellation: &CancellationToken,
    deadline: Option<Duration>,
) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    match deadline {
        None => {
            tokio::select! {
                biased;
                () = cancellation.cancelled() => Err(Error::cancelled()),
                result = future => result,
            }
        }
        Some(limit) => {
            let sleep = tokio::time::sleep(limit);
            tokio::pin!(sleep);
            tokio::select! {
                biased;
                () = cancellation.cancelled() => Err(Error::cancelled()),
                () = &mut sleep => Err(crate::error::run_timeout(operation, limit)),
                result = future => result,
            }
        }
    }
}

/// One step of a guarded stream: a frame, or a control-flow verdict.
pub(crate) enum StreamStep<T> {
    /// The inner stream yielded (`None` when it ended).
    Frame(Option<Result<T>>),
    /// The run's cancellation token fired.
    Cancelled,
    /// The run's deadline expired.
    TimedOut,
}

/// Awaits the next frame of `inner`, racing it against cancellation and the
/// deadline. Biased to the token first, then the deadline, so a frame that is
/// ready on the same poll cannot mask either.
pub(crate) async fn next_stream_step<S, T>(
    inner: &mut S,
    cancellation: &CancellationToken,
    deadline_at: Option<(tokio::time::Instant, Duration)>,
) -> StreamStep<T>
where
    S: futures::Stream<Item = Result<T>> + Unpin,
{
    match deadline_at {
        None => {
            tokio::select! {
                biased;
                () = cancellation.cancelled() => StreamStep::Cancelled,
                frame = inner.next() => StreamStep::Frame(frame),
            }
        }
        Some((at, _)) => {
            let sleep = tokio::time::sleep_until(at);
            tokio::pin!(sleep);
            tokio::select! {
                biased;
                () = cancellation.cancelled() => StreamStep::Cancelled,
                () = &mut sleep => StreamStep::TimedOut,
                frame = inner.next() => StreamStep::Frame(frame),
            }
        }
    }
}

/// Records the outcome of a guarded stream on its span and its trace event.
pub(crate) fn record_stream_outcome(
    span: &RunSpan,
    kind: &'static str,
    run_id: &str,
    component: &str,
    session_id: &str,
    started: Instant,
    failure: Option<&Error>,
) {
    if let Some(error) = failure {
        span.fail(error);
        trace_failure(
            kind,
            run_id,
            component,
            session_id,
            started.elapsed(),
            error,
        );
    } else {
        span.finish(Value::Null);
        trace_finish(kind, run_id, component, session_id, started.elapsed());
    }
}

/// The effective run deadline: the caller's, else the runner's default.
pub(crate) fn effective_deadline(services: &RunServices, options: &RunOptions) -> Option<Duration> {
    options
        .timeout
        .or_else(|| services.config.default_run_timeout())
}

/// Projects a serializable value to JSON, falling back to `Null` when the
/// projection cannot be represented.
pub(crate) fn to_value<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// Emits the start event of one run.
pub(crate) fn trace_start(kind: &'static str, run_id: &str, component: &str, session_id: &str) {
    tracing::debug!(
        target: LOG_TARGET,
        run_id,
        kind,
        component,
        session_id,
        "run started"
    );
}

/// Emits the completion event of one run with its duration.
pub(crate) fn trace_finish(
    kind: &'static str,
    run_id: &str,
    component: &str,
    session_id: &str,
    elapsed: Duration,
) {
    let duration_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
    tracing::debug!(
        target: LOG_TARGET,
        run_id,
        kind,
        component,
        session_id,
        duration_ms,
        "run finished"
    );
}

/// Emits the failure event of one run with its duration.
pub(crate) fn trace_failure(
    kind: &'static str,
    run_id: &str,
    component: &str,
    session_id: &str,
    elapsed: Duration,
    error: &Error,
) {
    let duration_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
    tracing::warn!(
        target: LOG_TARGET,
        run_id,
        kind,
        component,
        session_id,
        duration_ms,
        error = %error,
        "run finished with an error"
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_session::{AgentSession, MemoryCheckpointer};
    use serde_json::json;

    fn services() -> RunServices {
        RunServices::new(
            ResourceMgr::new(),
            CallbackFramework::new(),
            RunnerConfig::default(),
            None,
        )
    }

    // ports tests/unit_tests/core/runner/test_runner.py (session id precedence)
    #[test]
    fn session_id_prefers_the_conversation_id() {
        let config = RunnerConfig::default().with_default_session_id("runner-default");
        let options = RunOptions::default().with_session_id("caller");
        assert_eq!(
            resolve_session_id(&json!({"conversation_id": "conv"}), &options, &config),
            "conv"
        );
    }

    #[test]
    fn session_id_falls_back_to_the_caller_then_the_config() {
        let config = RunnerConfig::default().with_default_session_id("runner-default");
        let options = RunOptions::default().with_session_id("caller");
        assert_eq!(resolve_session_id(&json!({}), &options, &config), "caller");
        assert_eq!(
            resolve_session_id(&json!({}), &RunOptions::default(), &config),
            "runner-default"
        );
    }

    #[test]
    fn empty_or_non_string_conversation_ids_are_absent() {
        let config = RunnerConfig::default().with_default_session_id("runner-default");
        let options = RunOptions::default().with_session_id("caller");
        for inputs in [
            json!({"conversation_id": ""}),
            json!({"conversation_id": 7}),
        ] {
            assert_eq!(resolve_session_id(&inputs, &options, &config), "caller");
        }
        assert_eq!(
            resolve_session_id(&json!(null), &RunOptions::default(), &config),
            "runner-default"
        );
        // A whitespace-only id is still a non-empty string, so it wins.
        assert_eq!(
            resolve_session_id(
                &json!({"conversation_id": "  "}),
                &RunOptions::default(),
                &config
            ),
            "  "
        );
    }

    #[test]
    fn an_empty_caller_session_id_is_absent() {
        let config = RunnerConfig::default().with_default_session_id("runner-default");
        let options = RunOptions::default().with_session_id("");
        assert_eq!(
            resolve_session_id(&json!({}), &options, &config),
            "runner-default"
        );
    }

    #[test]
    fn run_options_defaults_and_builders() {
        let defaults = RunOptions::default();
        assert!(defaults.session.is_none());
        assert!(defaults.team_session.is_none());
        assert!(defaults.session_id.is_none());
        assert!(defaults.timeout.is_none());
        assert!(defaults.stream_modes.is_none());
        assert!(defaults.envs.is_empty());
        assert!(defaults.checkpointer.is_none());
        assert!(defaults.metadata.is_empty());
        assert!(!defaults.cancellation.is_cancelled());

        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s-1"));
        let token = CancellationToken::new();
        let checkpointer = Arc::new(MemoryCheckpointer::new()) as Arc<dyn Checkpointer>;
        let options = RunOptions::default()
            .with_session(Arc::clone(&session))
            .with_session_id("s-1")
            .with_cancellation(token.clone())
            .with_timeout(Duration::from_millis(250))
            .with_stream_modes(vec![StreamMode::Output])
            .with_envs("K", json!("V"))
            .with_checkpointer(Arc::clone(&checkpointer))
            .with_metadata("trace", json!("abc"));
        assert!(Arc::ptr_eq(options.session.as_ref().unwrap(), &session));
        assert_eq!(options.session_id.as_deref(), Some("s-1"));
        assert_eq!(options.timeout, Some(Duration::from_millis(250)));
        assert_eq!(options.stream_modes, Some(vec![StreamMode::Output]));
        assert_eq!(options.envs.get("K"), Some(&json!("V")));
        assert!(Arc::ptr_eq(
            options.checkpointer.as_ref().unwrap(),
            &checkpointer
        ));
        assert_eq!(options.metadata.get("trace"), Some(&json!("abc")));
        token.cancel();
        assert!(options.cancellation.is_cancelled());

        let debug = format!("{options:?}");
        assert!(debug.contains("RunOptions"));
        assert!(!debug.contains("payload"));
    }

    #[test]
    fn agent_ref_from_impls_and_helpers() {
        let agent: Arc<dyn Agent> = Arc::new(TestAgent);
        assert!(matches!(AgentRef::from("a"), AgentRef::Id(id) if id == "a"));
        assert!(matches!(AgentRef::from("a".to_owned()), AgentRef::Id(id) if id == "a"));
        assert!(matches!(
            AgentRef::from(Arc::clone(&agent)),
            AgentRef::Agent(_)
        ));
        assert!(matches!(AgentRef::id("a"), AgentRef::Id(id) if id == "a"));
        assert!(matches!(AgentRef::of(agent.clone()), AgentRef::Agent(_)));
        assert!(matches!(
            AgentRef::from(Arc::new(TestAgent) as Arc<dyn Agent>),
            AgentRef::Agent(_)
        ));
        assert!(format!("{:?}", AgentRef::id("a")).contains("Id"));
        assert!(format!("{:?}", AgentRef::of(agent)).contains("test-agent"));
    }

    #[test]
    fn workflow_and_team_refs_expose_their_variants() {
        let workflow = Arc::new(Workflow::new(openjiuwen_workflow::WorkflowCard::new("wf")));
        assert!(matches!(WorkflowRef::from("wf"), WorkflowRef::Id(id) if id == "wf"));
        assert!(matches!(
            WorkflowRef::from("wf".to_owned()),
            WorkflowRef::Id(id) if id == "wf"
        ));
        assert!(matches!(
            WorkflowRef::from(Arc::clone(&workflow)),
            WorkflowRef::Workflow(_)
        ));
        assert!(matches!(WorkflowRef::id("wf"), WorkflowRef::Id(_)));
        assert!(matches!(
            WorkflowRef::of(workflow),
            WorkflowRef::Workflow(_)
        ));

        let team: Arc<dyn Team> = Arc::new(BaseTeam::new(
            openjiuwen_multiagent::TeamCard::new("crew"),
            openjiuwen_multiagent::TeamConfig::default(),
        ));
        assert!(matches!(TeamRef::from("crew"), TeamRef::Id(id) if id == "crew"));
        assert!(matches!(
            TeamRef::from("crew".to_owned()),
            TeamRef::Id(id) if id == "crew"
        ));
        assert!(matches!(TeamRef::from(Arc::clone(&team)), TeamRef::Team(_)));
        assert!(matches!(TeamRef::id("crew"), TeamRef::Id(_)));
        assert!(matches!(TeamRef::of(team), TeamRef::Team(_)));
    }

    #[test]
    fn run_services_builders() {
        let services = services();
        assert!(services.guardrails.is_none());
        let pipeline = Arc::new(GuardrailPipeline::new());
        let services = services.with_guardrails(Arc::clone(&pipeline));
        assert!(Arc::ptr_eq(
            services.guardrails.as_ref().unwrap(),
            &pipeline
        ));
    }

    #[tokio::test]
    async fn run_with_limits_reports_cancellation_before_completion() {
        let token = CancellationToken::new();
        token.cancel();
        let error = run_with_limits("run_agent", async { Ok(1) }, &token, None)
            .await
            .unwrap_err();
        assert!(error.is_cancelled());
    }

    #[tokio::test]
    async fn run_with_limits_reports_the_deadline() {
        let error = run_with_limits(
            "run_agent",
            async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(1)
            },
            &CancellationToken::new(),
            Some(Duration::from_millis(5)),
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "timeout: run_agent timed out after 5ms");
    }

    #[test]
    fn effective_deadline_prefers_the_run_option() {
        let mut services = services();
        services.config = services.config.with_default_run_timeout_secs(Some(10.0));
        assert_eq!(
            effective_deadline(&services, &RunOptions::default()),
            Some(Duration::from_secs(10))
        );
        assert_eq!(
            effective_deadline(
                &services,
                &RunOptions::default().with_timeout(Duration::from_secs(1))
            ),
            Some(Duration::from_secs(1))
        );
    }

    #[test]
    fn to_value_falls_back_to_null_only_for_unrepresentable_values() {
        assert_eq!(to_value(&json!({"a": 1})), json!({"a": 1}));
        assert_eq!(to_value(&f64::NAN), Value::Null);
    }

    use futures::stream::BoxStream;
    use openjiuwen_agent::{AgentCard, AgentChunk, AgentOutput};

    /// A minimal agent used only by the ref tests.
    struct TestAgent;

    #[async_trait::async_trait]
    impl Agent for TestAgent {
        fn card(&self) -> &AgentCard {
            static CARD: std::sync::OnceLock<AgentCard> = std::sync::OnceLock::new();
            CARD.get_or_init(|| {
                let mut card = AgentCard::new("test-agent");
                card.id = "test-agent".to_owned();
                card
            })
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            Ok(AgentOutput::default())
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::empty())
        }
    }
}
