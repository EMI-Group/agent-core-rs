//! The leader's coordination loop: one `select!` over chunks, bus events, cancellation and a
//! settle tick.
//!
//! Python parity source: the `CoordinationKernel` of
//! `openjiuwen/agent_teams/runtime/coordination.py`, reimplemented as one documented loop. The
//! kernel is an event-driven supervisor with batching, idle debounce and escalation heuristics;
//! this port keeps the *observable* contract and simplifies the policy:
//!
//! * exactly one `team.completed` marker, emitted last;
//! * deliveries never reach the leader runtime (they are persisted and routed elsewhere);
//! * the run ends when the leader is idle **and** the task board is settled, on two consecutive
//!   settle ticks so a between-turn idle does not finish the run;
//! * mailbox and board events become short steering lines, batched into one `send` per
//!   `steer_batch_size` lines.
//!
//! ```text
//! start -> deliver payloads -> send query -> loop {
//!     member chunk      -> tag, forward, accumulate the leader answer
//!     bus event         -> one steering line, batched and flushed per `steer_batch_size`
//!     cancelled         -> flush, close the gate, Error::cancelled()
//!     settle tick       -> flush a partial batch, evaluate "idle && settled" twice
//! } -> flush -> close the gate -> team.completed marker -> AgentOutput::answer(answer)
//! ```
//!
//! One run per team is supported: the bus receiver is a single-consumer value
//! (`TeamAgent::take_events`), so a second concurrent run sees chunks and the settle tick only.
use std::time::Duration;

use openjiuwen_agent::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use openjiuwen_session::Session;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::time::{self, MissedTickBehavior};
use tracing::{debug, warn};

use super::team_agent::TeamAgent;
use crate::schema::{tag_chunk, team_completed_frame, EventMessage, EventPayload, TEAM_COMPLETED};

/// How often an idle run re-reads the task board before concluding the team is done.
pub const SETTLE_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// How many consecutive settle ticks must observe "leader idle && board settled".
///
/// Two ticks (one interval apart) keep a between-turn idle from ending the run before the leader
/// starts its next turn.
const REQUIRED_SETTLE_TICKS: u32 = 2;

/// Chunk kind of an answer/reasoning output event (see [`crate::harness::mapping`]).
const CHUNK_LLM_OUTPUT: &str = "llm_output";

/// Chunk kind used when a forwarded chunk carries no `type` field.
const CHUNK_FALLBACK: &str = "chunk";

/// Sink the run loop forwards tagged frames into, when a streaming caller asked for them.
pub(crate) type ChunkSink = UnboundedSender<Result<AgentChunk>>;

/// Why the loop stopped.
enum RunExit {
    /// The leader was idle and the task board settled for two consecutive ticks.
    Settled,
    /// The caller cancelled the run.
    Cancelled,
}

/// Runs one team request to completion and returns the leader's answer.
///
/// `session` is accepted for API parity with [`openjiuwen_agent::Agent::invoke`] but unused beyond a
/// debug log: the harness owns the session (`HarnessMemberRuntime::start` already ran with the
/// team's session before this loop starts), so a caller-supplied session is never forwarded into a
/// live cycle. A second `start` of the same runtime would reject it anyway.
///
/// # Errors
///
/// Propagates start failures and leader runtime failures. A terminal leader state during the
/// initial `send` propagates; a "not ready" runtime error is logged and the loop keeps waiting
/// (the settle tick decides when the run is over). Cancellation returns [`Error::cancelled`].
// The loop is one documented linear sequence: splitting it would hide the order of the
// branches that the node contract pins.
#[allow(clippy::too_many_lines)]
pub(crate) async fn run(
    agent: TeamAgent,
    query: Option<String>,
    session: Option<Arc<dyn Session>>,
    deliveries: Vec<crate::interaction::InteractPayload>,
    sink: Option<ChunkSink>,
) -> Result<AgentOutput> {
    if let Some(session) = session.as_ref() {
        debug!(
            session = session.id(),
            "caller session is not forwarded into a live cycle"
        );
    }
    debug!(
        team = agent.team_name(),
        member = agent.member_name(),
        query = query.is_some(),
        deliveries = deliveries.len(),
        "team run starting"
    );
    agent.start().await?;
    for payload in deliveries {
        let kind = payload.kind();
        let result = agent.deliver(payload).await;
        if !result.ok {
            debug!(
                team = agent.team_name(),
                kind,
                reason = result.reason.as_deref().unwrap_or_default(),
                "delivery was not accepted"
            );
        }
    }

    let mut chunks = agent.subscribe_chunks();
    if let Some(query) = query {
        if let Err(error) = agent.runtime().send(query, false).await {
            if agent.state() == HarnessState::Terminated {
                return Err(error);
            }
            debug!(
                team = agent.team_name(),
                error = %error,
                "leader runtime was not ready for the initial query; waiting on the settle tick"
            );
        }
    }

    let (events_keeper, mut events) = if let Some(receiver) = agent.take_events() {
        (None, receiver)
    } else {
        // No bus receiver left (a second concurrent run): hold a live sender so the branch
        // never fires spuriously, and drain chunks/tick/cancellation only.
        let (tx, rx) = mpsc::unbounded_channel();
        (Some(tx), rx)
    };
    let cancel = agent.cancellation_token();
    let mut ticker = time::interval(SETTLE_POLL_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    let mut index: i64 = 0;
    let mut answer = String::new();
    let mut batch: Vec<String> = Vec::new();
    let mut idle_ticks: u32 = 0;
    let mut chunks_open = true;
    let mut events_open = true;

    let mut org_acks: Vec<String> = Vec::new();
    let exit = loop {
        tokio::select! {
            chunk = chunks.recv(), if chunks_open => {
                match chunk {
                    Some(chunk) => {
                        idle_ticks = 0;
                        forward_chunk(&agent, chunk, &mut index, &mut answer, sink.as_ref());
                    }
                    None => chunks_open = false,
                }
            }
            event = events.recv(), if events_open => {
                match event {
                    Some(event) => {
                        idle_ticks = 0;
                        if let Some(line) = steer_line(&agent, &event).await {
                            batch.push(line);
                            if batch.len() >= agent.steer_batch_size() {
                                flush(&agent, &mut batch).await;
                            }
                        }
                    }
                    None => events_open = false,
                }
            }
            () = cancel.cancelled() => break RunExit::Cancelled,
            _ = ticker.tick() => {
                // Drain output produced before the runtime went idle, so a tick can never
                // outrun a chunk that is already queued.
                while let Ok(chunk) = chunks.try_recv() {
                    idle_ticks = 0;
                    forward_chunk(&agent, chunk, &mut index, &mut answer, sink.as_ref());
                }
                drain_org_mail(&agent, &mut batch, &mut org_acks).await;
                if flush(&agent, &mut batch).await {
                    acknowledge_org(&agent, &mut org_acks).await;
                }
                if agent.state() == HarnessState::Idle && agent.backend().tasks().is_settled().await? {
                    idle_ticks += 1;
                    if idle_ticks >= REQUIRED_SETTLE_TICKS {
                        break RunExit::Settled;
                    }
                } else {
                    idle_ticks = 0;
                }
            }
        }
    };

    drop(events_keeper);
    if flush(&agent, &mut batch).await {
        acknowledge_org(&agent, &mut org_acks).await;
    }
    agent.gate().close();

    match exit {
        RunExit::Settled => {
            let marker = AgentChunk::new(
                TEAM_COMPLETED,
                index,
                tag_chunk(
                    agent.member_name(),
                    agent.role(),
                    serde_json::json!({ "payload": team_completed_frame() }),
                ),
            );
            if let Some(sink) = &sink {
                let _ = sink.send(Ok(marker));
            }
            debug!(
                team = agent.team_name(),
                answer_chars = answer.chars().count(),
                "team run settled"
            );
            Ok(AgentOutput::answer(answer))
        }
        RunExit::Cancelled => Err(Error::cancelled()),
    }
}

/// Tags `chunk` with the member identity, forwards it and accumulates the leader's answer.
fn forward_chunk(
    agent: &TeamAgent,
    chunk: Value,
    index: &mut i64,
    answer: &mut String,
    sink: Option<&ChunkSink>,
) {
    let kind = chunk
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or(CHUNK_FALLBACK)
        .to_owned();
    let payload = chunk
        .get("payload")
        .cloned()
        .unwrap_or_else(|| chunk.clone());
    if kind == CHUNK_LLM_OUTPUT {
        if let Some(text) = answer_text(&payload) {
            answer.push_str(text);
        }
    }
    let frame = AgentChunk::new(
        kind,
        *index,
        tag_chunk(agent.member_name(), agent.role(), chunk),
    );
    *index += 1;
    if let Some(sink) = sink {
        // A closed sink means the streaming caller went away; the run continues for the mailbox
        // and the board, so this is never fatal.
        let _ = sink.send(Ok(frame));
    }
}

/// Reads the answer text out of an `llm_output` payload: `content`, else `text`.
fn answer_text(payload: &Value) -> Option<&str> {
    payload
        .get("content")
        .and_then(Value::as_str)
        .or_else(|| payload.get("text").and_then(Value::as_str))
}

/// Sends one batched steering message to the leader, logging (never failing) a rejection.
///
/// A rejected batch is restored so the next tick retries it: the ids it carries must stay
/// unacknowledged until a flush succeeds.
async fn flush(agent: &TeamAgent, batch: &mut Vec<String>) -> bool {
    if batch.is_empty() {
        return true;
    }
    let joined = batch.join("\n");
    batch.clear();
    if let Err(error) = agent.runtime().send(joined.clone(), false).await {
        debug!(
            team = agent.team_name(),
            error = %error,
            "leader runtime did not accept a steering batch"
        );
        batch.push(joined);
        return false;
    }
    true
}

/// Drains the organisation mailbox for this team into the current steering batch.
///
/// One line per unread message; the ids are remembered so they can be acknowledged only after the
/// leader accepted the batch that carried them. An id is queued at most once, because a batch the
/// leader rejected is retried on the next tick while its messages stay unread. A team without an
/// organisation is a no-op, and a store failure is logged rather than ending the run.
async fn drain_org_mail(agent: &TeamAgent, batch: &mut Vec<String>, acks: &mut Vec<String>) {
    let Some(org) = agent.org() else {
        return;
    };
    match org.unread(agent.team_name()).await {
        Ok(messages) => {
            for message in messages {
                batch.push(format!(
                    "org {}: {}",
                    message.from_team,
                    message.preview(160)
                ));
                if !acks.contains(&message.message_id) {
                    acks.push(message.message_id);
                }
            }
        }
        Err(error) => debug!(
            team = agent.team_name(),
            %error,
            "failed to read the organisation mailbox"
        ),
    }
}

/// Acknowledges delivered org messages after a successful leader turn.
async fn acknowledge_org(agent: &TeamAgent, acks: &mut Vec<String>) {
    if acks.is_empty() {
        return;
    }
    let Some(org) = agent.org() else {
        acks.clear();
        return;
    };
    let ids = std::mem::take(acks);
    if let Err(error) = org.acknowledge(agent.team_name(), &ids).await {
        debug!(
            team = agent.team_name(),
            %error,
            count = ids.len(),
            "failed to acknowledge the organisation mailbox"
        );
    }
}

/// Translates one bus event into the leader's steering text.
///
/// Mail events read the leader's unread mailbox and mark it read (one line per row); board events
/// render the task id and its status; member events name the member and its status. Every other
/// event is ignored, and a store/transport failure is logged rather than ending the run.
async fn steer_line(agent: &TeamAgent, event: &EventMessage) -> Option<String> {
    let leader = agent.member_name();
    if event.sender_id == leader {
        return None;
    }
    let payload = match EventPayload::from_message(event) {
        Ok(payload) => payload,
        Err(error) => {
            debug!(
                team = agent.team_name(),
                event_type = %event.event_type,
                error = %error,
                "ignoring an unparsable bus event"
            );
            return None;
        }
    };
    match payload {
        EventPayload::Message(_) | EventPayload::Broadcast(_) => unread_lines(agent, leader).await,
        EventPayload::TaskStatusChanged(payload) => Some(format!(
            "task {}: {}",
            payload.task_id,
            payload.status.as_str()
        )),
        EventPayload::TaskCreated(payload) => Some(format!("task {}: created", payload.task_id)),
        EventPayload::TaskVerified(payload) => Some(format!(
            "task {}: {}",
            payload.task_id,
            payload.verdict.as_str()
        )),
        EventPayload::MemberStatusChanged(payload) => Some(format!(
            "member {} is {}",
            payload.member_name,
            payload.status.as_str()
        )),
        _ => None,
    }
}

/// Renders and acknowledges the leader's unread mailbox.
async fn unread_lines(agent: &TeamAgent, leader: &str) -> Option<String> {
    let backend = agent.backend();
    let unread = match backend.unread(leader).await {
        Ok(unread) => unread,
        Err(error) => {
            warn!(team = agent.team_name(), error = %error, "failed to read the leader mailbox");
            return None;
        }
    };
    if unread.is_empty() {
        return None;
    }
    let direct: Vec<String> = unread
        .iter()
        .filter(|row| !row.broadcast)
        .map(|row| row.message_id.clone())
        .collect();
    if !direct.is_empty() {
        if let Err(error) = backend.mark_read(leader, &direct).await {
            warn!(team = agent.team_name(), error = %error, "failed to mark the leader mailbox read");
        }
    }
    Some(
        unread
            .iter()
            .map(|row| {
                format!(
                    "{} -> {}: {}",
                    row.from_member_name,
                    row.to_member_name.as_deref().unwrap_or("all"),
                    row.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::runtime::team_agent::test_support::{
        blueprint, built, deps, RecordingWorkspaceProvider, ScriptedFactory,
    };
    use crate::runtime::team_agent::SpecBuild;
    use crate::schema::{is_team_event_marker, TaskGraphSpec};
    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    /// Upper bound for any single wait in these tests.
    const BOUND: Duration = Duration::from_secs(2);

    /// Drains `stream` to its end, failing the test when it does not finish within [`BOUND`].
    async fn drain(
        mut stream: futures::stream::BoxStream<'static, Result<AgentChunk>>,
    ) -> Vec<Result<AgentChunk>> {
        time::timeout(BOUND, async move {
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        })
        .await
        .expect("the run did not end within the bound")
    }

    /// Builds a leader agent for `teammates`.
    async fn agent_for(teammates: &[&str]) -> Arc<TeamAgent> {
        let (agent, _factory, _workspace) = built(&blueprint("alpha", "boss", teammates)).await;
        agent
    }

    #[tokio::test]
    async fn invoke_returns_the_leader_answer() {
        let agent = agent_for(&[]).await;
        let output = agent.invoke(json!("hello"), None).await.unwrap();
        assert_eq!(output.output, "done: boss");
        assert!(agent.is_started());
    }

    #[tokio::test]
    async fn the_stream_tags_the_answer_and_ends_with_one_completed_marker() {
        let agent = agent_for(&[]).await;
        let frames = drain(agent.stream(json!("hello"), None)).await;

        assert!(
            frames.len() >= 2,
            "an answer chunk and the marker: {frames:?}"
        );
        assert!(frames.iter().all(Result::is_ok), "{frames:?}");
        let first = frames.first().unwrap().as_ref().unwrap();
        assert_eq!(first.kind, CHUNK_LLM_OUTPUT);
        assert_eq!(first.payload["source_member"], "boss");
        assert_eq!(first.payload["role"], "leader");

        let last = frames.last().unwrap().as_ref().unwrap();
        assert_eq!(last.kind, TEAM_COMPLETED);
        assert!(is_team_event_marker(&last.payload));
        let markers = frames
            .iter()
            .filter(|item| {
                item.as_ref()
                    .is_ok_and(|chunk| is_team_event_marker(&chunk.payload))
            })
            .count();
        assert_eq!(markers, 1, "exactly one completed marker: {frames:?}");
        assert!(agent.gate().is_closed(), "the gate closes with the run");
    }

    #[tokio::test]
    async fn mailbox_events_become_one_steering_line() {
        let (agent, factory, _workspace) = built(&blueprint("alpha", "boss", &["dev-1"])).await;
        agent
            .backend()
            .send_message("dev-1", Some("boss"), "status?")
            .await
            .unwrap();
        let frames = drain(agent.stream(json!("hello"), None)).await;
        assert!(frames.iter().all(Result::is_ok), "{frames:?}");

        let sends = factory.harness("boss").unwrap().sends();
        assert!(sends.iter().any(|send| send == "hello"), "{sends:?}");
        assert!(
            sends
                .iter()
                .any(|send| send.contains("dev-1 -> boss: status?")),
            "{sends:?}"
        );
    }

    #[tokio::test]
    async fn an_open_task_keeps_the_stream_open_until_the_board_settles() {
        let agent = agent_for(&["dev-1"]).await;
        let record = agent
            .create_task(TaskGraphSpec {
                title: "ship it".to_owned(),
                content: "do the thing".to_owned(),
                ..TaskGraphSpec::default()
            })
            .await
            .unwrap();
        assert!(!agent.backend().tasks().is_settled().await.unwrap());

        let backend = agent.backend();
        let task_id = record.task_id.clone();
        let closer = tokio::spawn(async move {
            time::sleep(Duration::from_millis(150)).await;
            let mutation = backend.tasks().cancel(&task_id).await.unwrap();
            backend.publish_mutation(&mutation).await.unwrap();
        });

        let frames = drain(agent.stream(json!("hello"), None)).await;
        closer.await.unwrap();
        let last = frames.last().unwrap().as_ref().unwrap();
        assert!(is_team_event_marker(&last.payload), "{frames:?}");
        assert_eq!(last.kind, TEAM_COMPLETED);
        assert!(agent.backend().tasks().is_settled().await.unwrap());
    }

    #[tokio::test]
    async fn stop_ends_a_hung_stream_with_a_cancellation() {
        let factory = ScriptedFactory::new();
        factory.hanging_for("boss");
        let workspace = RecordingWorkspaceProvider::new();
        let agent = blueprint("alpha", "boss", &[])
            .build_with(deps(&factory, &workspace))
            .await
            .unwrap();

        let stopper = tokio::spawn({
            let agent = Arc::clone(&agent);
            async move {
                time::sleep(Duration::from_millis(150)).await;
                agent.stop().await.unwrap();
            }
        });
        let frames = drain(agent.stream(json!("hello"), None)).await;
        stopper.await.unwrap();

        assert_eq!(frames.len(), 1, "{frames:?}");
        match frames.first().unwrap() {
            Err(error) => assert_eq!(error.kind(), ErrorKind::Cancelled),
            Ok(chunk) => panic!("expected the cancellation, got {chunk:?}"),
        }
    }
}
