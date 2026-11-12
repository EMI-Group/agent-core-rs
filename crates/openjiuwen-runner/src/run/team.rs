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
use serde_json::Value;
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
    trace_start("team", &run_id, &card.id, session.id());
    let span = RunSpan::start(session.tracer(), "run_team", inputs.clone());
    let teardown = team_teardown(lease);

    let result = team.invoke(inputs, Some(Arc::clone(&session))).await;
    match result {
        Ok(value) => {
            span.finish(value.clone());
            teardown.finish().await;
            trace_finish("team", &run_id, &card.id, session.id(), started.elapsed());
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
            Err(error)
        }
    }
}

/// Runs one agent-team invocation as a stream of wire frames.
///
/// The stream is returned immediately, so a resolution or transport-start
/// failure is delivered as a one-element stream yielding that error. On success
/// the strategy's frames are forwarded verbatim (nothing is re-wrapped) and the
/// [`SessionLease`] is finished by the [`Teardown`] after the last frame or when
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
                    teardown.finish().await;
                    yield Err(error);
                    return;
                }
            }
        }

        span.finish(Value::Null);
        trace_finish("team", &run_id, &card.id, session.id(), started.elapsed());
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
