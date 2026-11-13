#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessContext, HarnessError,
    HarnessEvent, HarnessEventCursor, HarnessInput, HarnessProtocol, HarnessResult, OutputChannel,
    OutputEvent, OutputKind, OutputOperation, SendReceipt, TurnEventKind, TurnLifecycleEvent,
};
use parking_lot::Mutex;
use serde_json::json;
use tokio::io::{AsyncReadExt, DuplexStream};
use tokio::sync::mpsc::{self, UnboundedSender};

use super::*;

/// Bound for one read in the tests, so a protocol bug fails instead of hanging.
const TEST_TIMEOUT: Duration = Duration::from_secs(5);

/// The identity every framing test runs under.
fn context() -> ServeContext {
    ServeContext {
        team_name: "alpha".to_owned(),
        member_name: "dev-1".to_owned(),
        session_id: "s1".to_owned(),
        workspace_root: PathBuf::from("/tmp/workspace"),
        language: DEFAULT_LANGUAGE.to_owned(),
    }
}

/// A file-descriptor-free harness: records commands and scripts an observation stream.
struct FakeHarness {
    /// Current lifecycle state.
    state: Mutex<HarnessState>,
    /// Recorded commands, in call order.
    calls: Mutex<Vec<String>>,
    /// Static capability card.
    card: HarnessCard,
    /// Observation buffer policy.
    buffer: EventBufferConfig,
    /// The single observation receiver handed to the pump.
    events: Mutex<Option<mpsc::UnboundedReceiver<HarnessEvent>>>,
    /// Sender used to script observations.
    sender: UnboundedSender<HarnessEvent>,
    /// Sequence counter for observed events.
    sequence: AtomicU64,
}

impl FakeHarness {
    /// Builds a harness in `Terminated` (not started) state.
    fn new() -> Arc<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();
        Arc::new(Self {
            state: Mutex::new(HarnessState::Terminated),
            calls: Mutex::new(Vec::new()),
            card: HarnessCard::new("fake", "0.0.0"),
            buffer: EventBufferConfig::default(),
            events: Mutex::new(Some(receiver)),
            sender,
            sequence: AtomicU64::new(0),
        })
    }

    /// Returns the recorded commands.
    fn calls(&self) -> Vec<String> {
        self.calls.lock().clone()
    }

    /// Publishes one observation to the pump.
    fn emit(
        &self,
        payload: impl Into<openjiuwen_harness_protocol::HarnessEventPayload>,
        turn_id: &str,
    ) {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let event = HarnessEvent::new(sequence, 0.0, payload, "s1", "dev-1").with_turn_id(turn_id);
        let _ = self.sender.send(event);
    }
}

/// Cursor over the fake's channel-backed observation stream.
struct FakeCursor {
    /// Observation source.
    receiver: mpsc::UnboundedReceiver<HarnessEvent>,
}

#[async_trait]
impl HarnessEventCursor for FakeCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        Ok(self.receiver.recv().await)
    }

    async fn close(&mut self) -> HarnessResult<()> {
        self.receiver.close();
        Ok(())
    }
}

#[async_trait]
#[allow(clippy::needless_pass_by_value)]
impl HarnessProtocol for FakeHarness {
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
        self.calls.lock().push("start".to_owned());
        *self.state.lock() = HarnessState::Idle;
        Ok(())
    }

    async fn stop(&self) -> HarnessResult<()> {
        self.calls.lock().push("stop".to_owned());
        *self.state.lock() = HarnessState::Terminated;
        Ok(())
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        match self.events.lock().take() {
            Some(receiver) => Box::new(FakeCursor { receiver }),
            None => Box::new(openjiuwen_harness_protocol::VecEventCursor::new(Vec::new())),
        }
    }

    fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        Box::new(openjiuwen_harness_protocol::VecEventCursor::new(Vec::new()))
    }

    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        let content = input
            .content
            .as_str()
            .map_or_else(|| input.content.to_string(), str::to_owned);
        self.calls.lock().push(format!("send:{content}"));
        if *self.state.lock() == HarnessState::Terminated {
            return Err(HarnessError::state("the fake harness is not started"));
        }
        let turn = "turn-1";
        self.emit(
            TurnLifecycleEvent {
                kind: TurnEventKind::Started,
                result: None,
            },
            turn,
        );
        self.emit(
            OutputEvent {
                output_id: "out-1".to_owned(),
                kind: OutputKind::Text,
                content: input.content.clone(),
                operation: OutputOperation::Delta,
                channel: OutputChannel::Answer,
                content_index: 0,
                content_type: None,
                data: BTreeMap::new(),
            },
            turn,
        );
        self.emit(
            TurnLifecycleEvent {
                kind: TurnEventKind::Finished,
                result: None,
            },
            turn,
        );
        Ok(SendReceipt {
            message_id: "msg-1".to_owned(),
            turn_id: turn.to_owned(),
            accepted_mode: mode,
        })
    }

    async fn abort(&self, mode: AbortMode) -> HarnessResult<()> {
        let name = match mode {
            AbortMode::Graceful => "abort:graceful",
            AbortMode::Force => "abort:force",
        };
        self.calls.lock().push(name.to_owned());
        Ok(())
    }

    async fn pause(&self) -> HarnessResult<()> {
        self.calls.lock().push("pause".to_owned());
        Ok(())
    }

    async fn resume(&self, query: Option<HarnessInput>) -> HarnessResult<()> {
        let detail = query.map_or_else(
            || "none".to_owned(),
            |input| {
                input
                    .content
                    .as_str()
                    .map_or_else(|| input.content.to_string(), str::to_owned)
            },
        );
        self.calls.lock().push(format!("resume:{detail}"));
        Ok(())
    }

    async fn export_checkpoint(
        &self,
    ) -> HarnessResult<Option<openjiuwen_harness_protocol::HarnessCheckpoint>> {
        Ok(None)
    }
}

/// Factory handing out one [`FakeHarness`] per member, or failing on demand.
struct FakeFactory {
    /// Contexts recorded by `create`, in call order.
    calls: Mutex<Vec<MemberHarnessContext>>,
    /// Harnesses built, in call order.
    harnesses: Mutex<Vec<Arc<FakeHarness>>>,
    /// When set, `create` fails.
    failing: bool,
}

impl FakeFactory {
    /// Builds a factory that succeeds.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            harnesses: Mutex::new(Vec::new()),
            failing: false,
        })
    }

    /// Builds a factory whose `create` always fails.
    fn failing() -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            harnesses: Mutex::new(Vec::new()),
            failing: true,
        })
    }

    /// Returns the first context `create` received.
    fn first_context(&self) -> MemberHarnessContext {
        self.calls.lock().first().cloned().expect("no factory call")
    }

    /// Returns the first harness built.
    fn harness(&self) -> Arc<FakeHarness> {
        self.harnesses.lock().first().cloned().expect("no harness")
    }
}

#[async_trait]
impl MemberHarnessFactory for FakeFactory {
    async fn create(
        &self,
        _spec: &DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>> {
        self.calls.lock().push(ctx.clone());
        if self.failing {
            return Err(Error::config("fake harness factory failure"));
        }
        let harness = FakeHarness::new();
        self.harnesses.lock().push(Arc::clone(&harness));
        Ok(harness)
    }

    fn name(&self) -> &'static str {
        "fake"
    }
}

/// The leader end of the in-memory pipe: writes request lines and reads response frames.
struct Leader {
    /// The leader's duplex endpoint.
    stream: DuplexStream,
    /// Bytes read but not yet consumed as a line.
    buffer: Vec<u8>,
}

impl Leader {
    /// Writes one request line and flushes it.
    async fn send(&mut self, line: &str) {
        self.stream.write_all(line.as_bytes()).await.unwrap();
        self.stream.flush().await.unwrap();
    }

    /// Reads the next response frame, skipping blank noise lines.
    async fn next_frame(&mut self) -> MemberResponse {
        loop {
            if let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
                let line: Vec<u8> = self.buffer.drain(..=position).collect();
                let text = String::from_utf8(line).unwrap();
                if let Some(frame) = MemberResponse::parse_noisy_line(&text).unwrap() {
                    return frame;
                }
                continue;
            }
            let mut chunk = [0_u8; 4096];
            let read = tokio::time::timeout(TEST_TIMEOUT, self.stream.read(&mut chunk))
                .await
                .expect("the frame read timed out")
                .unwrap();
            assert!(read > 0, "the server closed the frame stream");
            self.buffer.extend_from_slice(&chunk[..read]);
        }
    }
}

/// Spawns the server over an in-memory pipe and returns the leader end plus its exit code.
fn spawn_server(factory: Arc<dyn MemberHarnessFactory>) -> (Leader, JoinHandle<i32>) {
    let (member_side, leader_side) = tokio::io::duplex(64 * 1024);
    let (read, write) = tokio::io::split(member_side);
    let context = context();
    let handle =
        tokio::spawn(async move { serve_with_context(read, write, factory, &context).await });
    (
        Leader {
            stream: leader_side,
            buffer: Vec::new(),
        },
        handle,
    )
}

/// Drives a `start` handshake and consumes its two frames.
async fn start_member(leader: &mut Leader) {
    leader
        .send("{\"type\":\"start\",\"session_id\":\"s1\"}\n")
        .await;
    assert!(leader.next_frame().await.is_ready());
    assert_eq!(
        leader.next_frame().await,
        MemberResponse::State {
            state: HarnessState::Idle
        }
    );
}

#[tokio::test]
async fn start_frame_answers_ready_then_state() {
    let factory = FakeFactory::new();
    let (mut leader, server) = spawn_server(factory.clone());

    leader
        .send("{\"type\":\"start\",\"session_id\":\"s1\"}\n")
        .await;
    assert_eq!(
        leader.next_frame().await,
        MemberResponse::Ready {
            session_id: "s1".to_owned()
        }
    );
    assert_eq!(
        leader.next_frame().await,
        MemberResponse::State {
            state: HarnessState::Idle
        }
    );

    leader.send("{\"type\":\"stop\"}\n").await;
    assert_eq!(server.await.unwrap(), 0);
}

#[tokio::test]
async fn the_start_frame_session_id_is_authoritative() {
    let factory = FakeFactory::new();
    let (mut leader, server) = spawn_server(factory.clone());

    leader
        .send("{\"type\":\"start\",\"session_id\":\"child-session\"}\n")
        .await;
    assert_eq!(
        leader.next_frame().await,
        MemberResponse::Ready {
            session_id: "child-session".to_owned()
        }
    );
    assert_eq!(
        leader.next_frame().await,
        MemberResponse::State {
            state: HarnessState::Idle
        }
    );
    assert_eq!(factory.first_context().session_id, "child-session");

    leader.send("{\"type\":\"stop\"}\n").await;
    assert_eq!(server.await.unwrap(), 0);
}

#[tokio::test]
async fn send_frame_streams_round_and_chunk_frames() {
    let factory = FakeFactory::new();
    let (mut leader, server) = spawn_server(factory.clone());
    start_member(&mut leader).await;

    leader
        .send("{\"type\":\"send\",\"content\":\"hello\",\"immediate\":false}\n")
        .await;
    assert_eq!(
        leader.next_frame().await,
        MemberResponse::Round {
            outcome: RoundOutcome::Started
        }
    );
    let MemberResponse::Chunk { chunk } = leader.next_frame().await else {
        panic!("expected a chunk frame");
    };
    assert_eq!(chunk["type"], json!("llm_output"));
    assert_eq!(chunk["payload"]["content"], json!("hello"));
    assert_eq!(
        leader.next_frame().await,
        MemberResponse::Round {
            outcome: RoundOutcome::Finished
        }
    );

    leader.send("{\"type\":\"stop\"}\n").await;
    assert_eq!(server.await.unwrap(), 0);
}

#[tokio::test]
async fn unparsable_line_is_answered_and_the_loop_keeps_serving() {
    let factory = FakeFactory::new();
    let (mut leader, server) = spawn_server(factory);

    leader.send("not json\n").await;
    let MemberResponse::Error { message } = leader.next_frame().await else {
        panic!("expected an error frame");
    };
    assert!(
        message.contains("not json"),
        "unexpected message: {message}"
    );

    // Blank lines are skipped, and the loop still serves a later request.
    leader.send("\n").await;
    start_member(&mut leader).await;
    leader.send("{\"type\":\"stop\"}\n").await;
    assert_eq!(server.await.unwrap(), 0);
}

#[tokio::test]
async fn send_before_start_is_answered_with_an_error() {
    let factory = FakeFactory::new();
    let (mut leader, server) = spawn_server(factory);

    leader
        .send("{\"type\":\"send\",\"content\":\"hello\",\"immediate\":false}\n")
        .await;
    let MemberResponse::Error { message } = leader.next_frame().await else {
        panic!("expected an error frame");
    };
    assert!(
        message.contains("not been started"),
        "unexpected message: {message}"
    );

    start_member(&mut leader).await;
    leader.send("{\"type\":\"stop\"}\n").await;
    assert_eq!(server.await.unwrap(), 0);
}

#[tokio::test]
async fn stop_frame_stops_the_member_and_exits_zero() {
    let factory = FakeFactory::new();
    let (mut leader, server) = spawn_server(factory.clone());
    start_member(&mut leader).await;

    leader.send("{\"type\":\"stop\"}\n").await;
    assert_eq!(server.await.unwrap(), 0);
    assert_eq!(factory.harness().calls(), vec!["start", "stop"]);
}

#[tokio::test]
async fn end_of_input_stops_the_member_and_exits_zero() {
    let factory = FakeFactory::new();
    let (mut leader, server) = spawn_server(factory.clone());
    start_member(&mut leader).await;

    drop(leader);
    assert_eq!(server.await.unwrap(), 0);
    assert_eq!(factory.harness().calls(), vec!["start", "stop"]);
}

#[tokio::test]
async fn factory_failure_answers_with_an_error_and_exits_one() {
    let factory = FakeFactory::failing();
    let (mut leader, server) = spawn_server(factory);

    leader
        .send("{\"type\":\"start\",\"session_id\":\"s1\"}\n")
        .await;
    let MemberResponse::Error { message } = leader.next_frame().await else {
        panic!("expected an error frame");
    };
    assert!(
        message.contains("fake harness factory failure"),
        "unexpected message: {message}"
    );
    assert_eq!(server.await.unwrap(), 1);
}

#[tokio::test]
async fn abort_pause_and_resume_are_forwarded_to_the_runtime() {
    let factory = FakeFactory::new();
    let (mut leader, server) = spawn_server(factory.clone());
    start_member(&mut leader).await;

    leader
        .send("{\"type\":\"abort\",\"immediate\":false}\n")
        .await;
    leader.send("{\"type\":\"pause\"}\n").await;
    leader
        .send("{\"type\":\"resume\",\"query\":\"fresh\"}\n")
        .await;
    leader.send("{\"type\":\"stop\"}\n").await;
    assert_eq!(server.await.unwrap(), 0);
    assert_eq!(
        factory.harness().calls(),
        vec![
            "start".to_owned(),
            "abort:graceful".to_owned(),
            "pause".to_owned(),
            "resume:fresh".to_owned(),
            "stop".to_owned(),
        ]
    );
}

#[test]
fn context_defaults_match_the_documented_contract() {
    let context = ServeContext::from_env_map(|_| None);
    assert_eq!(context.team_name, DEFAULT_TEAM_NAME);
    assert_eq!(context.member_name, SERVE_DEFAULT_MEMBER_NAME);
    assert_eq!(context.session_id, SERVE_DEFAULT_SESSION_ID);
    assert_eq!(context.language, DEFAULT_LANGUAGE);
    assert!(!context.workspace_root.as_os_str().is_empty());

    let env = |key: &str| match key {
        TEAM_NAME_ENV => Some("alpha".to_owned()),
        MEMBER_NAME_ENV => Some("dev-1".to_owned()),
        SESSION_ID_ENV => Some("sess-1".to_owned()),
        MEMBER_WORKSPACE_ENV => Some("/tmp/ws".to_owned()),
        _ => None,
    };
    let context = ServeContext::from_env_map(env);
    assert_eq!(context.team_name, "alpha");
    assert_eq!(context.member_name, "dev-1");
    assert_eq!(context.session_id, "sess-1");
    assert_eq!(context.workspace_root, PathBuf::from("/tmp/ws"));

    // Blank variables count as absent.
    let context = ServeContext::from_env_map(|_| Some(String::new()));
    assert_eq!(context.team_name, DEFAULT_TEAM_NAME);
    assert_eq!(context.session_id, SERVE_DEFAULT_SESSION_ID);
}

#[test]
fn a_blank_start_session_id_keeps_the_context_session() {
    let blank = context().with_session_id("   ");
    assert_eq!(blank.session_id, "s1");
    let named = context().with_session_id("s2");
    assert_eq!(named.session_id, "s2");
    assert_eq!(named.harness_context().session_id, "s2");
    assert_eq!(named.harness_context().member_name, "dev-1");
}
