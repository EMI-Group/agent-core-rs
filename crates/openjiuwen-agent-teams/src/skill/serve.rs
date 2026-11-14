//! The `team-member serve` child-process server: the member end of the JSON-lines protocol.
//!
//! [`StdioMemberRuntime`](crate::member::stdio::StdioMemberRuntime) launches a process member as
//! `team-member serve` and then speaks the frames documented in
//! [`crate::member::protocol`] over its stdin/stdout. This module is the other end: it reads one
//! [`MemberRequest`] per line from stdin, drives a [`MemberRuntime`] built through a
//! [`MemberHarnessFactory`], and writes [`MemberResponse`] frames to stdout.
//!
//! # Process contract
//!
//! * One JSON object per line, every line flushed, exactly as the protocol module requires.
//! * stdout carries protocol frames only; every human-readable diagnostic goes to stderr, and a
//!   failure to write stderr is ignored (a closed diagnostic stream must not panic the server).
//! * `start` builds and starts the member, then answers `ready` followed by exactly one `state`
//!   frame. The `start` frame's `session_id` is authoritative: a non-blank value replaces the
//!   environment-derived session id, and the harness context is built from it.
//! * `send`, `abort`, `pause` and `resume` are forwarded to the member runtime; a runtime failure
//!   becomes an `error` frame and the loop keeps serving.
//! * An unparsable request line (including invalid UTF-8) is answered with an `error` frame and
//!   the loop keeps serving; blank lines are skipped.
//! * `stop` or end of input stops the member and exits `0`. Teardown publishes the settled
//!   harness state (normally `terminated`) before the process exits.
//! * A harness that cannot be built or started is answered with an `error` frame and exits `1`.
//!
//! # Identity
//!
//! A spawned member process cannot inherit the leader's in-memory state, so the leader exports the
//! identity through the environment variables in [`crate::member::spawn`]
//! (`OPENJIUWEN_TEAM_NAME`, `OPENJIUWEN_MEMBER_NAME`, `OPENJIUWEN_SESSION_ID`,
//! `OPENJIUWEN_MEMBER_WORKSPACE`), plus its checkpoint settings
//! (`OPENJIUWEN_MEMBER_CHECKPOINTS`, `OPENJIUWEN_MEMBER_CHECKPOINT_DIR`). [`ServeContext::from_env`]
//! reads them and documents one default per absent variable. Language has no environment carrier in
//! the spawn contract, so a served member always starts in [`DEFAULT_LANGUAGE`].
//!
//! # Checkpoints
//!
//! The served member is wired like an in-process one: `build_member` applies the card id (its
//! member name, because the default spec carries no card), the policy and store of
//! [`ServeContext::checkpoint_config`], and the workspace root as the working directory. A directory
//! in the environment selects the file backend, so the records land under
//! `<dir>/<session>/<member>/checkpoint.json`; an absent one keeps the process-local in-memory
//! store, and `OPENJIUWEN_MEMBER_CHECKPOINTS=off` disables capture entirely.
//!
//! # Testability
//!
//! [`serve_with_context`] drives the same loop over any `AsyncRead`/`AsyncWrite` pair and an
//! injected factory, which is how the unit tests below exercise the framing without a process.
//!
//! ```no_run
//! use openjiuwen_agent_teams::skill::serve::{serve_stdio, ServeContext};
//!
//! # async fn demo() {
//! let context = ServeContext::from_env();
//! assert!(!context.session_id.is_empty());
//! let code = serve_stdio().await;
//! assert!(code == 0 || code == 1);
//! # }
//! ```
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::constants::{DEFAULT_LANGUAGE, DEFAULT_TEAM_NAME};
use crate::harness::{
    HarnessMemberRuntime, MemberCheckpointPolicy, MemberHarnessContext, MemberHarnessFactory,
    NativeMemberFactory,
};
use crate::member::protocol::{MemberRequest, MemberResponse};
use crate::member::spawn::{
    CHECKPOINT_DIR_ENV, CHECKPOINT_ENV, MEMBER_NAME_ENV, MEMBER_WORKSPACE_ENV, SESSION_ID_ENV,
    TEAM_NAME_ENV,
};
use crate::member::{MemberObserver, MemberRuntime};
use crate::runtime::spec_meta::{CheckpointConfig, CheckpointStoreConfig};
use crate::schema::{DeepAgentSpec, MemberMode, RoundOutcome};

/// Session id a served member runs under when `OPENJIUWEN_SESSION_ID` is absent and the `start`
/// frame names none.
pub const SERVE_DEFAULT_SESSION_ID: &str = "serve";

/// Member name used when `OPENJIUWEN_MEMBER_NAME` is absent.
pub const SERVE_DEFAULT_MEMBER_NAME: &str = "member";

/// Bound on how long teardown waits for the frame writer to drain before abandoning the task.
const WRITER_STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// Identity a served member process runs under.
///
/// The leader exports it through the environment; the `start` frame's session id overrides
/// [`session_id`](Self::session_id) once the handshake begins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeContext {
    /// Owning team name (`OPENJIUWEN_TEAM_NAME`, default [`DEFAULT_TEAM_NAME`]).
    pub team_name: String,
    /// This member's name (`OPENJIUWEN_MEMBER_NAME`, default [`SERVE_DEFAULT_MEMBER_NAME`]).
    pub member_name: String,
    /// Session id the member runs under (`OPENJIUWEN_SESSION_ID`, default
    /// [`SERVE_DEFAULT_SESSION_ID`]); a non-blank `start` frame value wins.
    pub session_id: String,
    /// Root directory the member's tools resolve against (`OPENJIUWEN_MEMBER_WORKSPACE`, default
    /// the process working directory, then `.` when even that cannot be read).
    pub workspace_root: PathBuf,
    /// Whether the member captures checkpoints (`OPENJIUWEN_MEMBER_CHECKPOINTS`, default `true`;
    /// `off`, `false` or `0` — case-insensitively — disables them).
    pub checkpoint_enabled: bool,
    /// Directory a file-backed checkpoint store writes into (`OPENJIUWEN_MEMBER_CHECKPOINT_DIR`,
    /// default `None`, which keeps the process-local in-memory store).
    pub checkpoint_dir: Option<PathBuf>,
    /// Prompt language; [`DEFAULT_LANGUAGE`] because the spawn contract carries no language.
    pub language: String,
}

impl ServeContext {
    /// Builds the context from the process environment.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_env_map(|key| std::env::var(key).ok())
    }

    /// Builds the context from an injectable environment reader, so tests never mutate the process
    /// environment.
    ///
    /// An absent *or blank* variable falls back to the documented default.
    #[must_use]
    pub fn from_env_map(get: impl Fn(&str) -> Option<String>) -> Self {
        let value = |key: &str| get(key).filter(|text| !text.trim().is_empty());
        Self {
            team_name: value(TEAM_NAME_ENV).unwrap_or_else(|| DEFAULT_TEAM_NAME.to_owned()),
            member_name: value(MEMBER_NAME_ENV)
                .unwrap_or_else(|| SERVE_DEFAULT_MEMBER_NAME.to_owned()),
            session_id: value(SESSION_ID_ENV)
                .unwrap_or_else(|| SERVE_DEFAULT_SESSION_ID.to_owned()),
            workspace_root: value(MEMBER_WORKSPACE_ENV).map_or_else(
                || std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                PathBuf::from,
            ),
            checkpoint_enabled: value(CHECKPOINT_ENV).is_none_or(|text| !is_checkpoint_off(&text)),
            checkpoint_dir: value(CHECKPOINT_DIR_ENV).map(PathBuf::from),
            language: DEFAULT_LANGUAGE.to_owned(),
        }
    }

    /// Sets the directory a file-backed checkpoint store writes into.
    ///
    /// The returned context captures checkpoints on the default policy, persisted through
    /// [`FileMemberCheckpointStore`](crate::harness::FileMemberCheckpointStore) rooted at `dir`.
    #[must_use]
    pub fn with_checkpoint_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.checkpoint_dir = Some(dir.into());
        self
    }

    /// The checkpoint wiring this served member runs with.
    ///
    /// [`checkpoint_enabled`](Self::checkpoint_enabled) `false` is a disabled policy, so the member
    /// captures nothing; otherwise the default policy runs on a file store rooted at
    /// [`checkpoint_dir`](Self::checkpoint_dir) when one is known and on the process-local
    /// in-memory store when it is not.
    #[must_use]
    pub fn checkpoint_config(&self) -> CheckpointConfig {
        if !self.checkpoint_enabled {
            return CheckpointConfig {
                policy: MemberCheckpointPolicy::disabled(),
                store: CheckpointStoreConfig::Memory,
            };
        }
        CheckpointConfig {
            policy: MemberCheckpointPolicy::default(),
            store: self
                .checkpoint_dir
                .clone()
                .map_or(CheckpointStoreConfig::Memory, |dir| {
                    CheckpointStoreConfig::File { dir: Some(dir) }
                }),
        }
    }

    /// Replaces the session id with the `start` frame's value, when it names one.
    ///
    /// A blank value is ignored, so a leader that omits the field keeps the environment-derived
    /// session id.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        let session_id = session_id.into();
        if !session_id.trim().is_empty() {
            self.session_id = session_id;
        }
        self
    }

    /// Builds the harness context a factory materializes this member from.
    #[must_use]
    pub fn harness_context(&self) -> MemberHarnessContext {
        MemberHarnessContext::new(
            self.team_name.clone(),
            self.member_name.clone(),
            self.session_id.clone(),
            self.workspace_root.clone(),
            self.language.clone(),
            MemberMode::BuildMode,
        )
    }
}

/// Serves one member over the process's stdin/stdout until `stop` or end of input.
///
/// This is the process entry point behind `team-member serve`; the default factory is
/// [`NativeMemberFactory::from_env`], so model credentials and the model name come from
/// `MODEL_PROVIDER`/`MODEL_NAME`/`API_KEY`/`API_BASE` through
/// [`EnvModelResolver`](crate::harness::EnvModelResolver).
///
/// Returns the process exit code: `0` after `stop` or end of input, `1` when the member cannot be
/// built or started (or the process streams cannot be opened), and `1` on non-Unix targets, where
/// `/dev/stdin` and `/dev/stdout` are unavailable without tokio's `io-std` feature.
pub async fn serve_stdio() -> i32 {
    serve_stdio_with(Arc::new(NativeMemberFactory::from_env())).await
}

/// [`serve_stdio`] with an injectable factory, for callers that own their own model resolution.
async fn serve_stdio_with(factory: Arc<dyn MemberHarnessFactory>) -> i32 {
    match process_streams().await {
        Ok((input, output)) => serve_with(input, output, factory).await,
        Err(error) => {
            diagnostic(&format!("cannot open the process stdin/stdout: {error}"));
            1
        }
    }
}

/// Serves one member over `input`/`output`, reading the identity from the process environment.
///
/// See [`serve_with_context`] for the context-injecting form used by tests.
pub async fn serve_with<R, W>(input: R, output: W, factory: Arc<dyn MemberHarnessFactory>) -> i32
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let context = ServeContext::from_env();
    serve_with_context(input, output, factory, &context).await
}

/// The single serve loop: reads one [`MemberRequest`] per line and writes [`MemberResponse`]
/// frames through one writer task, so frames never interleave.
///
/// Returns the process exit code described on [`serve_stdio`].
pub async fn serve_with_context<R, W>(
    input: R,
    output: W,
    factory: Arc<dyn MemberHarnessFactory>,
    context: &ServeContext,
) -> i32
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (frames, receiver) = mpsc::unbounded_channel();
    let writer = spawn_writer(output, receiver);
    let mut member: Option<Arc<dyn MemberRuntime>> = None;
    let mut lines = BufReader::new(input).lines();

    let exit_code = loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => {
                debug!("member serve: input closed");
                break 0;
            }
            Err(error) => {
                warn!(error = %error, "member serve: cannot read the request stream");
                let _ = frames.send(MemberResponse::Error {
                    message: format!("cannot read the member request stream: {error}"),
                });
                break 1;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let request = match MemberRequest::parse_line(&line) {
            Ok(request) => request,
            Err(error) => {
                warn!(error = %error, "member serve: unparsable request frame");
                let _ = frames.send(MemberResponse::Error {
                    message: error.to_string(),
                });
                continue;
            }
        };

        match request {
            MemberRequest::Start { session_id } => {
                if let Some(running) = member.as_ref() {
                    // A repeated `start` re-announces the member that already runs; the first
                    // handshake owns the session id.
                    debug!("member serve: start repeated, re-announcing the running member");
                    announce_ready(&frames, running.as_ref());
                    continue;
                }
                let start_context = context.clone().with_session_id(session_id);
                let built = match build_member(&factory, &start_context).await {
                    Ok(built) => built,
                    Err(error) => {
                        diagnostic(&format!("cannot build the member harness: {error}"));
                        let _ = frames.send(MemberResponse::Error {
                            message: error.to_string(),
                        });
                        break 1;
                    }
                };
                if let Err(error) = built.start(None).await {
                    diagnostic(&format!("cannot start the member harness: {error}"));
                    let _ = frames.send(MemberResponse::Error {
                        message: error.to_string(),
                    });
                    let _ = built.stop().await;
                    break 1;
                }
                // Subscribe after `start` so `ready` is the first announcement on the wire; the
                // pump's settled state is then published explicitly by `announce_ready`.
                built.subscribe(Arc::new(FrameObserver {
                    frames: frames.clone(),
                }));
                announce_ready(&frames, built.as_ref());
                member = Some(built);
            }
            MemberRequest::Stop => {
                debug!("member serve: stop requested");
                break 0;
            }
            other => {
                if let Err(message) = forward(member.as_deref(), other).await {
                    let _ = frames.send(MemberResponse::Error { message });
                }
            }
        }
    };

    if let Some(running) = member.take() {
        if let Err(error) = running.stop().await {
            warn!(error = %error, "member serve: the member failed to stop cleanly");
            diagnostic(&format!("the member failed to stop cleanly: {error}"));
        }
    }
    drop(frames);
    if tokio::time::timeout(WRITER_STOP_TIMEOUT, writer)
        .await
        .is_err()
    {
        warn!("member serve: the frame writer did not drain before the timeout");
    }
    exit_code
}

/// Builds the member's runtime from a default member spec and the served identity.
///
/// The runtime is wired exactly like an in-process member: its checkpoint identity is the spec card
/// id (the member name for the default spec, which carries no card), the policy and store come from
/// [`ServeContext::checkpoint_config`], and the working directory is the served workspace root. The
/// child process owns its store, so nothing is shared with the leader.
async fn build_member(
    factory: &Arc<dyn MemberHarnessFactory>,
    context: &ServeContext,
) -> Result<Arc<dyn MemberRuntime>> {
    let spec = DeepAgentSpec::default();
    let harness_context = context.harness_context();
    let harness = factory.create(&spec, &harness_context).await?;
    let card_id = crate::runtime::checkpoints::member_card_id(&spec, &context.member_name);
    let runtime = crate::runtime::checkpoints::wire_runtime(
        HarnessMemberRuntime::new(harness, context.session_id.clone()),
        &card_id,
        &context.checkpoint_config(),
        None,
        &context.workspace_root,
    );
    Ok(Arc::new(runtime))
}

/// Whether an `OPENJIUWEN_MEMBER_CHECKPOINTS` value disables checkpointing.
///
/// `off`, `false` and `0` disable it case-insensitively; every other value — and an absent
/// variable — leaves it enabled.
fn is_checkpoint_off(text: &str) -> bool {
    matches!(
        text.trim().to_ascii_lowercase().as_str(),
        "off" | "false" | "0"
    )
}

/// Writes the handshake answer: `ready` with the member's own session id, then one `state` frame.
fn announce_ready(frames: &UnboundedSender<MemberResponse>, member: &dyn MemberRuntime) {
    let _ = frames.send(MemberResponse::Ready {
        session_id: member.session_id().to_owned(),
    });
    let _ = frames.send(MemberResponse::State {
        state: member.state(),
    });
}

/// Forwards one non-lifecycle request to the running member.
///
/// Returns the failure detail when the member has not been started or the runtime rejects the
/// command; the caller turns it into an `error` frame.
async fn forward(
    member: Option<&dyn MemberRuntime>,
    request: MemberRequest,
) -> std::result::Result<(), String> {
    let Some(member) = member else {
        return Err("the member has not been started".to_owned());
    };
    let outcome = match request {
        MemberRequest::Send { content, immediate } => member.send(content, immediate).await,
        MemberRequest::Abort { immediate } => member.abort(immediate).await,
        MemberRequest::Pause => member.pause().await,
        MemberRequest::Resume { query } => member.resume(query).await,
        // `start` and `stop` are handled by the loop itself.
        MemberRequest::Start { .. } | MemberRequest::Stop => Ok(()),
    };
    outcome.map_err(|error| error.to_string())
}

/// Owns the single writer side of the protocol and flushes every frame.
fn spawn_writer<W>(output: W, mut frames: UnboundedReceiver<MemberResponse>) -> JoinHandle<()>
where
    W: AsyncWrite + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut output = output;
        while let Some(frame) = frames.recv().await {
            let line = match frame.to_line() {
                Ok(line) => line,
                Err(error) => {
                    warn!(error = %error, "member serve: dropped an unserializable frame");
                    continue;
                }
            };
            if let Err(error) = output.write_all(line.as_bytes()).await {
                diagnostic(&format!("cannot write the member frame stream: {error}"));
                break;
            }
            if let Err(error) = output.flush().await {
                diagnostic(&format!("cannot flush the member frame stream: {error}"));
                break;
            }
        }
        let _ = output.flush().await;
    })
}

/// Forwards one runtime observation onto the frame channel.
///
/// The callbacks are synchronous, so they hand the frame to the writer task instead of writing
/// themselves; that is what keeps frames from interleaving.
struct FrameObserver {
    /// The single writer channel.
    frames: UnboundedSender<MemberResponse>,
}

impl FrameObserver {
    /// Enqueues one frame, logging (never propagating) a closed writer.
    fn send(&self, frame: MemberResponse) {
        if self.frames.send(frame).is_err() {
            debug!("member serve: the frame writer is gone; dropping one observation");
        }
    }
}

impl MemberObserver for FrameObserver {
    fn on_state(&self, state: HarnessState) {
        self.send(MemberResponse::State { state });
    }

    fn on_round(&self, outcome: RoundOutcome) {
        self.send(MemberResponse::Round { outcome });
    }

    fn on_chunk(&self, chunk: Value) {
        self.send(MemberResponse::Chunk { chunk });
    }
}

/// Writes one diagnostic line to the process's stderr; a failed write is ignored.
fn diagnostic(message: &str) {
    let _ = writeln!(std::io::stderr(), "team-member serve: {message}");
}

/// Opens the process's stdin/stdout as async streams.
///
/// The workspace does not enable tokio's `io-std` feature (see
/// [`crate::mcp::stdio`]), so the process streams are reopened as `tokio::fs::File`s, the same
/// pattern the MCP stdio transport uses.
#[cfg(unix)]
async fn process_streams() -> Result<(tokio::fs::File, tokio::fs::File)> {
    let input = tokio::fs::File::open("/dev/stdin")
        .await
        .map_err(|error| Error::config(format!("cannot open /dev/stdin: {error}")))?;
    let output = tokio::fs::File::options()
        .write(true)
        .open("/dev/stdout")
        .await
        .map_err(|error| Error::config(format!("cannot open /dev/stdout: {error}")))?;
    Ok((input, output))
}

/// Reports the missing stdio transport on targets without `/dev/stdin`/`/dev/stdout`.
#[cfg(not(unix))]
async fn process_streams() -> Result<(tokio::fs::File, tokio::fs::File)> {
    Err(Error::unsupported(
        "team-member serve stdio transport (/dev/stdin, /dev/stdout)",
    ))
}

#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;
