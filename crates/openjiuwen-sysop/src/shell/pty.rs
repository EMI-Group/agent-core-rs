//! Pseudo-terminal streaming for shell commands.
//!
//! A child attached to a terminal line-buffers its own output, so a command that writes
//! progress lines reveals them as they happen instead of in one blob at exit. Python gets
//! that effect on Linux by wrapping the command in `stdbuf -oL -eL` (see
//! [`crate::shell::process::line_buffered_argv`]); this module offers the stronger
//! alternative of a real terminal via `openpty`, selected by
//! [`crate::config::LocalWorkConfig::pty_streaming`].
//!
//! Only the streaming API uses a terminal: a captured `execute_cmd` run needs stdout and
//! stderr apart, which a terminal cannot give (a terminal merges both into one stream), so
//! every chunk of a PTY-backed stream is [`StreamKind::Stdout`].
//!
//! The terminal is configured with the safe `nix::sys::termios` API only: `ECHO` is
//! cleared (the child's input is never echoed back into the output) and `OPOST`/`ONLCR`
//! are cleared so no carriage return is inserted before a newline. If `tcgetattr` or
//! `tcsetattr` fails, the child still runs with the terminal defaults and
//! [`pty_reader`] drops every `\r`, so streamed text is stable either way.
//!
//! Failure to allocate or attach a terminal is never an error: [`try_spawn`] returns
//! `None` and the caller falls back to pipes.

use std::fs::File;
use std::process::Stdio;

use futures::stream::BoxStream;
use nix::pty::{openpty, OpenptyResult};
use nix::sys::termios::{tcgetattr, tcsetattr, LocalFlags, OutputFlags, SetArg};
use nix::unistd::dup;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use crate::constants;
use crate::shell::process::{self, EngineEvent, RunOptions, Utf8Decoder};
use crate::shell::types::StreamKind;

/// A child attached to a pseudo-terminal plus the master side of that terminal.
pub(crate) struct PtyChild {
    /// The spawned child; it owns the slave side through its standard streams.
    pub(crate) child: Child,
    /// The parent's read end of the terminal.
    pub(crate) master: tokio::fs::File,
}

/// Opens a terminal and spawns `command` on the slave side of it.
///
/// Returns `None` — never an error — when the host cannot provide a PTY (`openpty` fails,
/// for example because `/dev/ptmx` is unavailable), when a descriptor cannot be
/// duplicated, or when the spawn itself fails; the caller then spawns the same command
/// with pipes. The command is consumed either way, so it must be fully configured before
/// the call.
pub(crate) fn try_spawn(mut command: Command) -> Option<PtyChild> {
    let OpenptyResult { master, slave } = match openpty(
        None::<&nix::pty::Winsize>,
        None::<&nix::sys::termios::Termios>,
    ) {
        Ok(pair) => pair,
        Err(error) => {
            tracing::debug!(%error, "no pseudo-terminal available; falling back to pipes");
            return None;
        }
    };
    configure_terminal(&slave);
    // A terminal merges the three standard streams, so the child gets the slave on all of
    // them; each stdio slot needs its own descriptor.
    let stdin = dup(&slave).ok()?;
    let stdout = dup(&slave).ok()?;
    let stderr = dup(&slave).ok()?;
    command
        .kill_on_drop(true)
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    command.process_group(0);
    let child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            tracing::debug!(%error, "failed to spawn a PTY command");
            return None;
        }
    };
    // The parent must not keep the slave open: the master only reports the end of the run
    // once every slave descriptor is closed.
    drop(slave);
    let master = tokio::fs::File::from_std(File::from(master));
    Some(PtyChild { child, master })
}

/// Clears echo and newline translation on the slave side, best effort.
///
/// `OPOST` off disables all output processing (including `ONLCR`, the `\n` → `\r\n`
/// translation), so the bytes reaching the master match pipe output. A failure is logged
/// and ignored: [`pty_reader`] also strips carriage returns.
fn configure_terminal(slave: &std::os::fd::OwnedFd) {
    let mut attributes = match tcgetattr(slave) {
        Ok(attributes) => attributes,
        Err(error) => {
            tracing::debug!(%error, "cannot read the terminal attributes; using defaults");
            return;
        }
    };
    attributes.local_flags.remove(LocalFlags::ECHO);
    attributes
        .output_flags
        .remove(OutputFlags::OPOST | OutputFlags::ONLCR);
    if let Err(error) = tcsetattr(slave, SetArg::TCSANOW, &attributes) {
        tracing::debug!(%error, "cannot apply the terminal attributes; using defaults");
    }
}

/// Streams the terminal output of `child` as [`EngineEvent`]s.
///
/// The pipe-path invariants carry over unchanged: bounded capture with the truncation
/// marker, one terminal event with the exit code, and a timeout or cancellation that kills
/// the child's process group. All data events are [`StreamKind::Stdout`] because a
/// terminal has a single stream.
pub(crate) fn stream_pty_child(
    child: Child,
    master: tokio::fs::File,
    options: RunOptions,
) -> BoxStream<'static, EngineEvent> {
    let (sender, receiver) = mpsc::channel(process::EVENT_CHANNEL_CAPACITY);
    tokio::spawn(pty_reader(master, options.max_output_bytes, sender));
    process::drive_child(child, options.timeout_secs, options.cancellation, receiver)
}

/// Reads the terminal master, sending bounded, carriage-return-free chunks to `sender`.
async fn pty_reader(
    mut master: tokio::fs::File,
    max_output_bytes: u64,
    sender: mpsc::Sender<EngineEvent>,
) {
    let limit = usize::try_from(max_output_bytes).unwrap_or(usize::MAX);
    let mut produced = 0_usize;
    let mut marked = false;
    let mut decoder = Utf8Decoder::default();
    let mut buffer = vec![0_u8; constants::DEFAULT_TEXT_CHUNK_SIZE];
    loop {
        let read = match master.read(&mut buffer).await {
            // EOF and a read failure both end the capture; on Linux the master reports
            // `EIO` once the last slave descriptor closes, which is the normal end.
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        if marked {
            // Past the budget: keep draining so the child never blocks on a full terminal.
            continue;
        }
        let take = limit.saturating_sub(produced).min(read);
        if take > 0 {
            produced += take;
            let text = strip_carriage_returns(&decoder.decode(&buffer[..take]));
            if !text.is_empty()
                && sender
                    .send(EngineEvent::Data {
                        kind: StreamKind::Stdout,
                        text,
                    })
                    .await
                    .is_err()
            {
                // The consumer went away: stop reading.
                return;
            }
        }
        if take < read {
            marked = true;
            let mut tail = strip_carriage_returns(&decoder.flush());
            tail.push_str(constants::OUTPUT_TRUNCATION_MARKER);
            if sender
                .send(EngineEvent::Data {
                    kind: StreamKind::Stdout,
                    text: tail,
                })
                .await
                .is_err()
            {
                return;
            }
        }
    }
    let tail = strip_carriage_returns(&decoder.flush());
    if !marked && !tail.is_empty() {
        let _ = sender
            .send(EngineEvent::Data {
                kind: StreamKind::Stdout,
                text: tail,
            })
            .await;
    }
}

/// Removes every carriage return from `text`.
///
/// With `OPOST` cleared the terminal produces none; the strip keeps the stream identical
/// when the terminal attributes could not be applied, and it also covers a child that
/// writes `\r` itself. The trade-off is that carriage-return progress bars lose their
/// overwrite behavior in a PTY stream.
fn strip_carriage_returns(text: &str) -> String {
    if text.contains('\r') {
        text.replace('\r', "")
    } else {
        text.to_string()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn carriage_returns_are_stripped() {
        assert_eq!(strip_carriage_returns("a\r\nb\r\n"), "a\nb\n");
        assert_eq!(strip_carriage_returns("plain\n"), "plain\n");
        assert_eq!(strip_carriage_returns(""), "");
    }

    #[cfg(unix)]
    #[test]
    fn terminal_is_allocated_and_configured() {
        let pair = match openpty(
            None::<&nix::pty::Winsize>,
            None::<&nix::sys::termios::Termios>,
        ) {
            Ok(pair) => pair,
            Err(error) => {
                eprintln!("skipping PTY test: {error}");
                return;
            }
        };
        configure_terminal(&pair.slave);
        let attributes = tcgetattr(&pair.slave).unwrap();
        assert!(!attributes.local_flags.contains(LocalFlags::ECHO));
        assert!(!attributes.output_flags.contains(OutputFlags::OPOST));
        assert!(!attributes.output_flags.contains(OutputFlags::ONLCR));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spawned_child_reports_output_through_the_terminal() {
        if let Err(error) = openpty(
            None::<&nix::pty::Winsize>,
            None::<&nix::sys::termios::Termios>,
        ) {
            eprintln!("skipping PTY test: {error}");
            return;
        }
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "echo pty-output"]);
        let Some(pty) = try_spawn(command) else {
            eprintln!("skipping PTY test: the command could not be attached to a terminal");
            return;
        };
        let options = RunOptions {
            max_output_bytes: 4096,
            ..RunOptions::default()
        };
        let mut events = stream_pty_child(pty.child, pty.master, options);
        let mut text = String::new();
        let mut terminal = None;
        while let Some(event) = futures::StreamExt::next(&mut events).await {
            match event {
                EngineEvent::Data { kind, text: chunk } => {
                    assert_eq!(kind, StreamKind::Stdout);
                    text.push_str(&chunk);
                }
                other => terminal = Some(other),
            }
        }
        assert_eq!(text, "pty-output\n");
        assert!(matches!(terminal, Some(EngineEvent::Exit { code: 0 })));
    }
}
