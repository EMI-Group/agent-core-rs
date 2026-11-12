//! `10_agent_team` — a team blueprint, built and run: `TeamAgentSpec` → build → invoke.
//!
//! A team is a *blueprint* — a leader agent plus the teammates it may spawn — and a *runtime*
//! that executes it. This example fills the blueprint, validates it, activates it through
//! `TeamRuntimeManager`, prints the roster, runs one request through the leader, snapshots the
//! team and stops it. The manager owns the live-team pool, so activation is also the build step:
//! `TeamRuntimeManager::activate` validates the blueprint, builds the runtime with the injected
//! dependencies and starts the leader in one call.
//!
//! Offline by default, and this example always stays loopback: the member models are resolved by
//! a `ModelResolver` over `EnvModelResolver`, pointed at a `MockLlmServer` on `127.0.0.1` instead
//! of a provider. The run therefore makes real OpenAI-compatible HTTP calls, but only to the
//! mock, and a placeholder API key is used (it is never printed).
//!
//! One workaround worth naming: `NativeMemberFactory` hands the resolved model to the
//! `DeepAgent` builder but does not copy the model name into the `DeepAgentConfig`, so the kernel
//! sends chat requests with an empty `model_name` and the provider client rejects them. This
//! example's resolver therefore wraps the client in [`NamedModel`], which supplies the name the
//! kernel left blank. Remove the wrapper once the factory propagates the resolved name.
//!
//! The leader blueprint sets `enable_task_loop: false`, a deviation from the Python default of
//! `true`: the member then runs exactly one `ReAct` turn instead of the task-loop protocol, so a
//! single mock response answers the request and the run is deterministic. The `dev-1` entry in
//! `predefined_members` only enriches the printed roster — a predefined member is materialized as
//! a roster row and spawned by the leader on demand, so this example spends no model call on it.
//!
//! Python counterparts: `examples/multi_agent/team_hybrid.py` and
//! `examples/multi_agent/builtin_teams/handoff_customer_service.py`.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen::agent_teams::harness::native::{
    EnvModelResolver, ModelResolver, NativeMemberFactory,
};
use openjiuwen::agent_teams::harness::MemberHarnessFactory;
use openjiuwen::agent_teams::runtime::{TeamBuildDeps, TeamRuntimeManager};
use openjiuwen::agent_teams::schema::{
    DeepAgentSpec, MemberSpecBase, PredefinedMemberSpec, TeamAgentSpec,
};
use openjiuwen::agent_teams::workspace::{
    LocalWorkspaceProvider, MemberWorkspaceLayout, WorkspaceProvider,
};
use openjiuwen::core::common::error::Result as SdkResult;
use openjiuwen::core::foundation::llm::{ChatResponse, Model, ModelRequestConfig, StreamChunk};
use openjiuwen::core::foundation::BaseMessage;
use openjiuwen::core::single_agent::{AgentCard, ResultType};
use serde_json::json;

/// Team name written into the blueprint and used to address the live team.
const TEAM_NAME: &str = "demo-team";

/// Session the team is activated under; `stop_team` needs the same id.
const SESSION_ID: &str = "example-session";

/// Provider the resolver is pinned to, so the environment cannot change it.
const MODEL_PROVIDER: &str = "openai";

/// Model name the resolver hands to every member.
const MODEL_NAME: &str = "gpt-4o-mini";

/// Placeholder credential for the loopback server; never printed.
const PLACEHOLDER_API_KEY: &str = "test-key";

/// The single request this example sends to the leader.
const QUESTION: &str = "Draft a two-sentence plan for shipping the release";

/// The answer the offline mock server returns for the leader's one `ReAct` turn.
const MOCK_ANSWER: &str = "Freeze the release branch on Friday and run the full regression suite \
                            over the weekend; ship on Monday if it stays green, with the on-call \
                            engineer holding a 24-hour rollback window.";

/// Upper bound on activation (validate, build, start the leader).
const ACTIVATE_TIMEOUT: Duration = Duration::from_secs(20);

/// Upper bound on the run, so a misconfigured mock cannot hang the demo.
const RUN_TIMEOUT: Duration = Duration::from_secs(30);

/// Upper bound on the snapshot and the teardown.
const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(10);

/// Forwards the model name the kernel left blank on a chat request.
///
/// See the module header: `NativeMemberFactory` resolves a model but does not carry its name into
/// the `DeepAgentConfig`, so the kernel's per-call request has an empty `model_name`. Filling it
/// here keeps the rest of the example on the plain resolver path.
fn named(request: &ModelRequestConfig, name: &str) -> ModelRequestConfig {
    let mut request = request.clone();
    if request.model_name.trim().is_empty() {
        name.clone_into(&mut request.model_name);
    }
    request
}

/// A resolved model that supplies the kernel's missing model name on every call.
struct NamedModel {
    /// The client built by the resolver.
    inner: Arc<dyn Model>,
    /// Provider-facing model name, taken from the client.
    name: String,
}

#[async_trait]
impl Model for NamedModel {
    fn provider(&self) -> &str {
        self.inner.provider()
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> SdkResult<ChatResponse> {
        self.inner
            .invoke(messages, &named(request, &self.name))
            .await
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, SdkResult<StreamChunk>> {
        self.inner.stream(messages, named(&request, &self.name))
    }
}

/// Resolves members' models from the loopback mock server.
struct LoopbackResolver {
    /// Environment-driven resolver, pinned to the mock by its builder.
    env: EnvModelResolver,
}

impl LoopbackResolver {
    /// Wraps an already configured [`EnvModelResolver`].
    fn new(env: EnvModelResolver) -> Self {
        Self { env }
    }
}

#[async_trait]
impl ModelResolver for LoopbackResolver {
    async fn resolve(&self, model_name: Option<&str>) -> SdkResult<Arc<dyn Model>> {
        let model = self.env.resolve(model_name).await?;
        let name = model.model_name().to_owned();
        Ok(Arc::new(NamedModel { inner: model, name }))
    }
}

/// Builds the team blueprint: one leader blueprint and one predefined teammate.
fn blueprint() -> TeamAgentSpec {
    let mut spec = TeamAgentSpec {
        team_name: TEAM_NAME.to_owned(),
        team_display_name: Some("Demo Team".to_owned()),
        ..TeamAgentSpec::default()
    };

    // The blueprint of the leader lives under the reserved `agents["leader"]` key; the leader's
    // *member* name (`team_leader` by default) is what the roster shows.
    let leader = DeepAgentSpec {
        card: Some(AgentCard::new("demo-leader")),
        system_prompt: Some(
            "You lead a small delivery team. Plan briefly, and hand work to a teammate when \
             there is work to hand out."
                .to_owned(),
        ),
        // One ReAct turn instead of the task-loop protocol: deterministic, one model call.
        enable_task_loop: false,
        ..DeepAgentSpec::default()
    };
    spec.agents.insert("leader".to_owned(), leader);

    // A predefined teammate: it lands on the roster up front, and only costs a model call if the
    // leader actually spawns it.
    spec.predefined_members
        .push(PredefinedMemberSpec::Teammate(MemberSpecBase {
            member_name: "dev-1".to_owned(),
            display_name: "Dev One".to_owned(),
            desc: "Implements the plan the leader hands over.".to_owned(),
            ..MemberSpecBase::default()
        }));

    spec
}

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() -> Result<()> {
    let mode = common::setup()?;
    common::banner::title("10 agent team");

    if mode.is_live() {
        common::banner::note(
            "live mode is set, but this example always runs against the loopback mock server",
        );
    }

    // The mock speaks the real OpenAI-compatible protocol; spare answers cost nothing and keep
    // the script from running dry if the leader asks a follow-up question.
    let server = common::mock_server(vec![
        common::text_response(MOCK_ANSWER),
        common::text_response(MOCK_ANSWER),
        common::text_response(MOCK_ANSWER),
    ])
    .await
    .context("the loopback mock LLM server did not start")?;
    common::banner::note(&format!(
        "mock LLM server: {} (placeholder key; nothing leaves loopback)",
        server.base_url()
    ));

    common::banner::step(&format!("blueprint: {TEAM_NAME}"));
    let spec = blueprint();
    spec.validate().context("the team blueprint is not valid")?;
    common::banner::note(&format!(
        "validated: {} agent blueprint(s), {} predefined member(s)",
        spec.agents.len(),
        spec.predefined_members.len()
    ));

    // Member workspaces go under this example's temp directory, removed with the handle.
    let workspace_root = common::tempdir().context("no temporary workspace directory")?;
    let resolver = EnvModelResolver::from_env()
        .with_provider(MODEL_PROVIDER)
        .with_model_name(MODEL_NAME)
        .with_api_base(server.base_url())
        .with_api_key(PLACEHOLDER_API_KEY);
    let workspace: Arc<dyn WorkspaceProvider> = Arc::new(LocalWorkspaceProvider::new(
        MemberWorkspaceLayout::new(workspace_root.path()),
    ));
    let factory: Arc<dyn MemberHarnessFactory> = Arc::new(NativeMemberFactory::new(Arc::new(
        LoopbackResolver::new(resolver),
    )));
    let deps = TeamBuildDeps::default()
        .with_harness_factory(factory)
        .with_workspace(workspace);
    let manager = TeamRuntimeManager::new(deps);

    common::banner::step(&format!(
        "activating team '{TEAM_NAME}' for session '{SESSION_ID}'"
    ));
    let team = tokio::time::timeout(
        ACTIVATE_TIMEOUT,
        manager.activate(&spec, SESSION_ID, &serde_json::Value::Null),
    )
    .await
    .context("activating the team timed out")?
    .context("the team did not activate")?;

    common::banner::note(&format!(
        "team: {} (leader {}, session {SESSION_ID})",
        team.team_name(),
        team.leader_member_name()
    ));
    common::banner::note(&format!("roster: {}", team.roster().join(", ")));

    common::banner::step(&format!("invoking with: {QUESTION}"));
    let output = tokio::time::timeout(RUN_TIMEOUT, team.invoke(json!(QUESTION), None))
        .await
        .context("the team run timed out")?
        .context("the team run failed")?;

    let result_type = match output.result_type {
        ResultType::Answer => "answer",
        ResultType::Error => "error",
        ResultType::Interrupt => "interrupt",
    };
    common::banner::note(&format!("result type: {result_type}"));
    common::banner::answer(&output.output);

    common::banner::step("snapshotting the team");
    let snapshot = tokio::time::timeout(TEARDOWN_TIMEOUT, team.snapshot())
        .await
        .context("the snapshot timed out")?;
    let names: Vec<&str> = snapshot
        .members
        .iter()
        .map(|member| member.member_name.as_str())
        .collect();
    common::banner::note(&format!(
        "{} member row(s) [{}], state {:?}",
        snapshot.members.len(),
        names.join(", "),
        snapshot.state
    ));

    common::banner::step("stopping the team");
    let stopped = tokio::time::timeout(TEARDOWN_TIMEOUT, manager.stop_team(TEAM_NAME, SESSION_ID))
        .await
        .context("stopping the team timed out")?
        .context("the team did not stop cleanly")?;
    if stopped {
        common::banner::note("team stopped and removed from the live pool");
    } else {
        common::banner::note("the team was no longer in the live pool");
    }

    common::banner::note(&format!(
        "the mock server served {} chat-completions request(s)",
        server.requests().len()
    ));

    Ok(())
}
