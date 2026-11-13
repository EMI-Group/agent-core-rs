//! Team execution: [`invoke_team`] and [`stream_team`].
//!
//! Ports the in-process path of `team_runner.py::_run_base_team`: start the
//! team's transport, take a [`SessionLease`] for the run, drive
//! [`Team::invoke`]/[`Team::stream`], and finish the lease (unbind the owned
//! session, clean its topics, end its stream, and commit).
//!
//! Teams fire no callbacks: Python's `AgentTeamEvents` are point-to-point and
//! pub/sub events, which the multiagent bus owns end to end. The runner's
//! callback vocabulary therefore has no team invoke/stream events.
//!
//! # Deliberate deviation from Python
//!
//! Python unconditionally binds and unbinds the run's team session. The Rust
//! port follows the multiagent contract instead: a caller-provided
//! [`AgentTeamSession`] is used exactly as the caller bound it (a borrowed
//! [`SessionLease`] only ends its stream on finish), while a session the runner
//! creates is bound, unbound, cleaned, ended, and committed by the lease.

use std::sync::Arc;
use std::time::Instant;

use futures::stream::{BoxStream, StreamExt};
use openjiuwen_common::error::Result;
use openjiuwen_multiagent::SessionLease;
use openjiuwen_session::Session;
use serde_json::{json, Value};
use uuid::Uuid;

use super::{
    resolve_team, trace_failure, trace_finish, trace_start, RunOptions, RunServices, RunSpan,
    TeamRef, Teardown, LOG_TARGET,
};

/// Runs one agent-team invocation to completion.
///
/// The reference is resolved first (a missing id is
/// [`openjiuwen_common::error::Error::NotFound`]), the team's transport is
/// started (idempotently), and a [`SessionLease`] is taken: without a
/// caller-provided team session the lease creates and binds one, and its
/// [`SessionLease::finish`] unbinds it, cleans its topics, ends its stream, and
/// commits it. With a caller-provided session the lease borrows it and only ends
/// its stream on finish — the deliberate relaxation of Python's unconditional
/// bind/unbind documented on this module.
///
/// The result value is returned verbatim; no callback events are fired.
///
/// # Errors
///
/// `NotFound` for a missing id, whatever [`Team::start`](openjiuwen_multiagent::Team::start)
/// reports, whatever the strategy reports, and a lease-teardown failure (logged,
/// never replacing a successful result).
#[allow(clippy::needless_pass_by_value)]
pub async fn invoke_team(
    services: RunServices,
    team: TeamRef,
    inputs: Value,
    options: RunOptions,
) -> Result<Value> {
    let run_id = Uuid::new_v4().to_string();
    let started = Instant::now();

    let team = resolve_team(&services, &team)?;
    team.start().await?;

    let runtime = team.runtime();
    let card = team.card();
    let lease = SessionLease::acquire(&runtime, &card, &inputs, options.team_session.clone()).await;
    let session = Arc::clone(lease.session());
    // Observation seams: registered once the lease has the run's session, which is
    // the last point that does not fire a callback on this path. Teams always
    // attach the lease session (borrowed or owned) to the hub watcher, so a host
    // can answer the interactions the strategy publishes. The binding is declared
    // before the status guard so the guard finalises the run before the hub entry
    // disappears.
    let detail = json!({"component": card.id});
    let binding = crate::hitl::bind_run(
        services.hitl.as_ref(),
        crate::hitl::ActiveRunSpec::for_execution(
            &run_id,
            &card.id,
            Some(card.id.as_str()),
            Some(session.id()),
            &options.cancellation,
            detail.clone(),
        ),
        Some(Arc::clone(&session) as Arc<dyn Session>),
    );
    let mut status = crate::monitor::RunStatusGuard::begin(
        services.monitor.clone(),
        &run_id,
        crate::monitor::RunStart::new(crate::monitor::RunKind::Team, card.id.clone())
            .with_session(session.id())
            .with_detail(detail.clone()),
    );
    status.running(detail.clone());

    trace_start("team", &run_id, &card.id, session.id());
    let span = RunSpan::start(session.tracer(), "run_team", inputs.clone());
    let teardown = team_teardown(lease);

    let result = team.invoke(inputs, Some(Arc::clone(&session))).await;
    match result {
        Ok(value) => {
            span.finish(value.clone());
            teardown.finish().await;
            trace_finish("team", &run_id, &card.id, session.id(), started.elapsed());
            status.ok(detail);
            binding.unregister();
            Ok(value)
        }
        Err(error) => {
            span.fail(&error);
            teardown.finish().await;
            trace_failure(
                "team",
                &run_id,
                &card.id,
                session.id(),
                started.elapsed(),
                &error,
            );
            status.fail(&error, detail);
            binding.unregister();
            Err(error)
        }
    }
}

/// Runs one agent-team invocation as a stream of wire frames.
///
/// The stream is returned immediately, so a resolution or transport-start
/// failure is delivered as a one-element stream yielding that error. On success
/// the strategy's frames are forwarded verbatim (nothing is re-wrapped) and the
/// [`SessionLease`] is finished by the `Teardown` after the last frame or when
/// the consumer drops the stream early.
///
/// As with [`invoke_team`], no callback events are fired and a caller-provided
/// team session is left bound by the caller.
#[allow(clippy::needless_pass_by_value)]
pub fn stream_team(
    services: RunServices,
    team: TeamRef,
    inputs: Value,
    options: RunOptions,
) -> BoxStream<'static, Result<Value>> {
    let run_id = Uuid::new_v4().to_string();
    async_stream::stream! {
        let started = Instant::now();
        let team = match resolve_team(&services, &team) {
            Ok(team) => team,
            Err(error) => {
                yield Err(error);
                return;
            }
        };
        if let Err(error) = team.start().await {
            yield Err(error);
            return;
        }

        let runtime = team.runtime();
        let card = team.card();
        let lease =
            SessionLease::acquire(&runtime, &card, &inputs, options.team_session.clone()).await;
        let session = Arc::clone(lease.session());
        // Observation seams: registered once the lease has the run's session. Teams
        // always attach the lease session (borrowed or owned) to the hub watcher.
        // The strategy's frames stay forwarded verbatim; a dropped stream is
        // finished as cancelled by the status guard.
        let detail = json!({"component": card.id});
        let binding = crate::hitl::bind_run(
            services.hitl.as_ref(),
            crate::hitl::ActiveRunSpec::for_execution(
                &run_id,
                &card.id,
                Some(card.id.as_str()),
                Some(session.id()),
                &options.cancellation,
                detail.clone(),
            ),
            Some(Arc::clone(&session) as Arc<dyn Session>),
        );
        let mut status = crate::monitor::RunStatusGuard::begin(
            services.monitor.clone(),
            &run_id,
            crate::monitor::RunStart::new(crate::monitor::RunKind::Team, card.id.clone())
                .with_session(session.id())
                .with_detail(detail.clone()),
        );
        status.running(detail.clone());

        trace_start("team", &run_id, &card.id, session.id());
        let span = RunSpan::start(session.tracer(), "run_team_stream", inputs.clone());
        let teardown = team_teardown(lease);
        let mut inner = team.stream(inputs, Some(Arc::clone(&session)));

        while let Some(frame) = inner.next().await {
            match frame {
                Ok(value) => yield Ok(value),
                Err(error) => {
                    span.fail(&error);
                    trace_failure(
                        "team",
                        &run_id,
                        &card.id,
                        session.id(),
                        started.elapsed(),
                        &error,
                    );
                    status.fail(&error, detail.clone());
                    binding.unregister();
                    teardown.finish().await;
                    yield Err(error);
                    return;
                }
            }
        }

        span.finish(Value::Null);
        trace_finish("team", &run_id, &card.id, session.id(), started.elapsed());
        status.ok(detail);
        binding.unregister();
        teardown.finish().await;
    }
    .boxed()
}

/// Builds the teardown of a team run (`SessionLease::finish`).
fn team_teardown(lease: SessionLease) -> Teardown {
    Teardown::new(async move {
        if let Err(error) = lease.finish().await {
            tracing::warn!(
                target: LOG_TARGET,
                error = %error,
                "team session teardown failed"
            );
        }
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::stream::BoxStream;
    use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
    use openjiuwen_multiagent::{handoff_team, HandoffTeamConfig, Team, TeamAgent, TeamCard};
    use serde_json::json;

    use crate::callback::CallbackFramework;
    use crate::config::RunnerConfig;
    use crate::monitor::{RunEvent, RunPhase};
    use crate::resources::ResourceMgr;

    /// A team member that answers immediately, so no handoff is needed.
    struct Answering {
        card: AgentCard,
    }

    #[async_trait::async_trait]
    impl Agent for Answering {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            Ok(AgentOutput::answer("done"))
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::empty())
        }
    }

    /// A one-member handoff team: the member answers its first input.
    fn one_member_team() -> Arc<dyn Team> {
        let card = TeamCard::new("crew").with_id("crew");
        let team = handoff_team(card, HandoffTeamConfig::default());
        let member_card = AgentCard::new("solo").with_id("solo");
        let member: Arc<dyn Agent> = Arc::new(Answering {
            card: member_card.clone(),
        });
        team.add_agent(
            member_card,
            Arc::new(move || TeamAgent::new(Arc::clone(&member))),
        )
        .expect("the member registers");
        Arc::new(team)
    }

    /// A services bundle observing both seams.
    fn services(
        monitor: &Arc<crate::monitor::RunMonitor>,
        hub: &Arc<crate::hitl::InteractionHub>,
    ) -> RunServices {
        RunServices::new(
            ResourceMgr::new(),
            CallbackFramework::new(),
            RunnerConfig::default(),
            None,
        )
        .with_monitor(Arc::clone(monitor))
        .with_hitl(Arc::clone(hub))
    }

    /// Drains the transitions a monitor published, in order.
    fn phases(events: &mut tokio::sync::broadcast::Receiver<RunEvent>) -> Vec<RunPhase> {
        std::iter::from_fn(|| events.try_recv().ok())
            .map(|event| event.phase)
            .collect()
    }

    #[tokio::test]
    async fn invoke_mirrors_the_run_and_unregisters_it() {
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();

        let value = invoke_team(
            services(&monitor, &hub),
            TeamRef::of(one_member_team()),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(value["output"], json!("done"));
        assert!(hub.is_empty(), "the run was unregistered on completion");
        assert_eq!(
            phases(&mut transitions),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed]
        );
    }

    #[tokio::test]
    async fn streaming_mirrors_the_run_and_unregisters_it() {
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();

        let frames: Vec<Result<Value>> = stream_team(
            services(&monitor, &hub),
            TeamRef::of(one_member_team()),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .collect()
        .await;

        assert!(frames.iter().all(Result::is_ok), "unexpected: {frames:?}");
        assert!(hub.is_empty(), "the stream unregistered its run");
        assert_eq!(
            phases(&mut transitions),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed]
        );
    }
}
