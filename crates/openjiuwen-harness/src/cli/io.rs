//! Injectable line-oriented IO used by the CLI.
//!
//! The runner and the REPL never touch `stdin`/`stdout` directly: they hold a
//! cloneable [`CliIo`] handle, so tests drive a scripted terminal instead of the
//! process streams. [`StdioReader`] and [`StdoutWriter`] are the real,
//! terminal-backed implementations; they move the blocking std calls into
//! [`tokio::task::spawn_blocking`] so no reactor thread ever blocks.
//!
//! A handle also carries two flags: whether it is interactive (which drives
//! fail-closed permission prompts) and whether ANSI colour is enabled (see
//! [`crate::cli::style`]). Both default to off for every explicit constructor, so
//! scripted terminals and one-shot runs keep producing escape-free, byte-exact
//! output.
//!
//! Failure mapping: OS and stdio failures (unreadable stdin, closed stdout,
//! `JoinError` from the blocking pool) carry no domain meaning, so they are
//! wrapped with [`Error::external`], which preserves the original error as the
//! [`std::error::Error::source`] for logging and `Debug`.

use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use tokio::sync::Mutex;

/// A source of input lines.
///
/// `read_line` returns the line with its trailing newline removed, and `None`
/// once the input is exhausted (including a closed stdin).
#[async_trait]
pub trait LineReader: Send {
    /// Reads the next line, or returns `None` at end of input.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the underlying input cannot be read.
    async fn read_line(&mut self) -> Result<Option<String>>;
}

/// A text sink: the terminal, or a test double.
#[async_trait]
pub trait Writer: Send {
    /// Writes `text` verbatim; no newline is appended.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the underlying sink cannot be written.
    async fn write(&mut self, text: &str) -> Result<()>;

    /// Flushes buffered text to the sink.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the flush fails.
    async fn flush(&mut self) -> Result<()>;
}

/// The guarded state behind a [`CliIo`]: the streams plus queued input lines.
struct IoState {
    /// Source of input lines.
    reader: Box<dyn LineReader>,
    /// Text sink.
    writer: Box<dyn Writer>,
    /// Lines returned by [`CliIo::prompt_line`] before the reader is consulted.
    pending: VecDeque<String>,
}

impl IoState {
    /// Builds the state of a handle over an explicit reader and writer.
    fn new(reader: Box<dyn LineReader>, writer: Box<dyn Writer>) -> Self {
        Self {
            reader,
            writer,
            pending: VecDeque::new(),
        }
    }
}

/// Cloneable handle over the CLI's input and output streams.
///
/// Both the runner and the permission host hold a clone: every call locks the
/// shared state for the duration of one read or write only, so no lock is ever
/// held across an `.await` of the agent stream.
#[derive(Clone)]
pub struct CliIo {
    inner: Arc<Mutex<IoState>>,
    interactive: bool,
    color: bool,
}

impl CliIo {
    /// Builds a handle from an explicit reader and writer (the test path).
    #[must_use]
    pub fn new(reader: Box<dyn LineReader>, writer: Box<dyn Writer>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(IoState::new(reader, writer))),
            interactive: false,
            color: false,
        }
    }

    /// Builds an interactive handle from an explicit reader and writer.
    ///
    /// Identical to [`CliIo::new`] except that the handle is marked interactive,
    /// so `CliPermissionHost` writes approval prompts and reads the answer from
    /// the injected streams instead of failing closed. This is the test path for
    /// permission-prompt behaviour.
    #[must_use]
    pub fn with_interactive(reader: Box<dyn LineReader>, writer: Box<dyn Writer>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(IoState::new(reader, writer))),
            interactive: true,
            color: false,
        }
    }

    /// Builds the interactive handle backed by the process stdio streams.
    ///
    /// The `interactive` flag is what later makes permission prompts
    /// answerable; a handle built with [`CliIo::new`] fails closed instead.
    #[must_use]
    pub fn stdio() -> Self {
        Self {
            inner: Arc::new(Mutex::new(IoState::new(
                Box::new(StdioReader),
                Box::new(StdoutWriter),
            ))),
            interactive: true,
            color: false,
        }
    }

    /// Builds the non-interactive stdio handle used by the one-shot `run` command.
    ///
    /// Reads stdin and writes stdout, but reports `is_interactive() == false`, so
    /// permission prompts fail closed exactly like Python's non-interactive `run`
    /// (no prompt is printed, the tool call is rejected).
    #[must_use]
    pub fn stdio_non_interactive() -> Self {
        Self {
            inner: Arc::new(Mutex::new(IoState::new(
                Box::new(StdioReader),
                Box::new(StdoutWriter),
            ))),
            interactive: false,
            color: false,
        }
    }

    /// Returns whether ANSI styling is enabled on this handle.
    ///
    /// Off for every constructor; the binary path opts in with
    /// [`CliIo::with_color`] once [`crate::cli::paths::color_enabled`] agrees.
    #[must_use]
    pub fn is_color(&self) -> bool {
        self.color
    }

    /// Returns this handle with ANSI styling turned on or off.
    ///
    /// The builder is cheap: the returned handle shares the same reader, writer
    /// and queued lines.
    #[must_use]
    pub fn with_color(self, enabled: bool) -> Self {
        Self {
            color: enabled,
            ..self
        }
    }

    /// Seeds lines that [`CliIo::prompt_line`] returns before consulting the reader.
    ///
    /// Each line is trimmed and blank lines are dropped, so a seed script may
    /// carry the same whitespace a human would. This is the offline analogue of
    /// pre-filling a line editor's history buffer: the REPL replays stored
    /// history through it, and tests script a transcript through it.
    pub async fn seed_lines(&self, lines: impl IntoIterator<Item = String>) {
        let mut guard = self.inner.lock().await;
        for line in lines {
            let line = line.trim();
            if !line.is_empty() {
                guard.pending.push_back(line.to_owned());
            }
        }
    }

    /// Returns the number of seed lines still queued.
    pub async fn pending_lines(&self) -> usize {
        self.inner.lock().await.pending.len()
    }

    /// Writes `prompt`, flushes it, then returns one line.
    ///
    /// A queued seed line (see [`CliIo::seed_lines`]) is returned first, in FIFO
    /// order and without touching the reader; only once the queue is empty is the
    /// reader asked for a line, which is trimmed as before.
    ///
    /// Returns `Ok(None)` at end of input.
    ///
    /// # Errors
    ///
    /// Returns the writer or reader error when prompting or reading fails.
    pub async fn prompt_line(&self, prompt: &str) -> Result<Option<String>> {
        let mut guard = self.inner.lock().await;
        let state = &mut *guard;
        state.writer.write(prompt).await?;
        state.writer.flush().await?;
        if let Some(line) = state.pending.pop_front() {
            return Ok(Some(line));
        }
        let line = state.reader.read_line().await?;
        Ok(line.map(|line| line.trim().to_owned()))
    }

    /// Writes `text` verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the sink cannot be written.
    pub async fn write(&self, text: &str) -> Result<()> {
        let mut guard = self.inner.lock().await;
        guard.writer.write(text).await
    }

    /// Writes `text` followed by a newline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the sink cannot be written.
    pub async fn write_line(&self, text: &str) -> Result<()> {
        self.write(&format!("{text}\n")).await
    }

    /// Writes a single newline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the sink cannot be written.
    pub async fn write_blank(&self) -> Result<()> {
        self.write("\n").await
    }

    /// Flushes the sink.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the flush fails.
    pub async fn flush(&self) -> Result<()> {
        let mut guard = self.inner.lock().await;
        guard.writer.flush().await
    }

    /// Returns whether this handle is backed by an interactive terminal.
    #[must_use]
    pub fn is_interactive(&self) -> bool {
        self.interactive
    }
}

/// [`LineReader`] over the process stdin.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdioReader;

#[async_trait]
impl LineReader for StdioReader {
    async fn read_line(&mut self) -> Result<Option<String>> {
        tokio::task::spawn_blocking(read_stdin_line)
            .await
            .map_err(Error::external)?
            .map_err(Error::external)
    }
}

/// [`Writer`] over the process stdout, flushed after every write.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdoutWriter;

#[async_trait]
impl Writer for StdoutWriter {
    async fn write(&mut self, text: &str) -> Result<()> {
        let text = text.to_owned();
        tokio::task::spawn_blocking(move || write_stdout(&text))
            .await
            .map_err(Error::external)?
            .map_err(Error::external)
    }

    async fn flush(&mut self) -> Result<()> {
        tokio::task::spawn_blocking(flush_stdout)
            .await
            .map_err(Error::external)?
            .map_err(Error::external)
    }
}

/// Blocks one thread until stdin yields a line; `None` means end of input.
fn read_stdin_line() -> std::io::Result<Option<String>> {
    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    let mut line = String::new();
    if handle.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\r', '\n']).to_owned()))
}

/// Writes to stdout and flushes, so streamed chunks appear immediately.
fn write_stdout(text: &str) -> std::io::Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(text.as_bytes())?;
    handle.flush()
}

/// Flushes stdout.
fn flush_stdout() -> std::io::Result<()> {
    std::io::stdout().flush()
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Scripted reader and writer doubles shared by the CLI node's test modules.
    //!
    //! The node's external suite (`tests/cli.rs`) and `tests.rs` keep their own
    //! copies; these live here so the `io` and `render` unit tests share one writer.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use openjiuwen_common::error::Result;

    use super::{CliIo, LineReader, Writer};

    /// [`LineReader`] that answers from a fixed script, then reports EOF.
    pub(crate) struct ScriptedReader {
        /// Lines still to hand out.
        pub(crate) lines: VecDeque<String>,
    }

    #[async_trait]
    impl LineReader for ScriptedReader {
        async fn read_line(&mut self) -> Result<Option<String>> {
            Ok(self.lines.pop_front())
        }
    }

    /// [`Writer`] that appends everything to a shared buffer.
    pub(crate) struct SharedWriter {
        /// Buffer every write is appended to.
        pub(crate) buffer: Arc<Mutex<String>>,
    }

    #[async_trait]
    impl Writer for SharedWriter {
        async fn write(&mut self, text: &str) -> Result<()> {
            self.buffer.lock().unwrap().push_str(text);
            Ok(())
        }

        async fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }

    /// Builds a handle over `lines` and a fresh shared output buffer.
    pub(crate) fn scripted_io(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
        let buffer = Arc::new(Mutex::new(String::new()));
        let reader = Box::new(ScriptedReader {
            lines: lines.iter().map(|line| (*line).to_owned()).collect(),
        });
        let writer = Box::new(SharedWriter {
            buffer: Arc::clone(&buffer),
        });
        (CliIo::new(reader, writer), buffer)
    }

    /// Returns everything written to `buffer` so far.
    pub(crate) fn output(buffer: &Arc<Mutex<String>>) -> String {
        buffer.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::{Arc, Mutex};

    use super::test_support::{output, scripted_io, SharedWriter};
    use super::*;

    /// Builds a handle over `lines` whose colour flag is left at its default.
    fn handle(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
        scripted_io(lines)
    }

    /// Builds an interactive handle over `lines` and a fresh buffer.
    fn interactive(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
        let buffer = Arc::new(Mutex::new(String::new()));
        let reader = Box::new(super::test_support::ScriptedReader {
            lines: lines.iter().map(|line| (*line).to_owned()).collect(),
        });
        let writer = Box::new(SharedWriter {
            buffer: Arc::clone(&buffer),
        });
        (CliIo::with_interactive(reader, writer), buffer)
    }

    #[tokio::test]
    async fn seeded_lines_come_out_before_reader_lines_in_fifo_order() {
        let (io, _buffer) = handle(&["from-reader"]);
        io.seed_lines(["first".to_owned(), "second".to_owned(), "third".to_owned()])
            .await;
        assert_eq!(io.pending_lines().await, 3);

        for expected in ["first", "second", "third", "from-reader"] {
            assert_eq!(
                io.prompt_line("> ").await.unwrap().as_deref(),
                Some(expected)
            );
        }
        assert_eq!(io.pending_lines().await, 0);
        assert_eq!(io.prompt_line("> ").await.unwrap(), None);
    }

    #[tokio::test]
    async fn blank_seeds_are_dropped_and_seeds_are_trimmed() {
        let (io, _buffer) = handle(&[]);
        io.seed_lines([
            "  padded  ".to_owned(),
            "   ".to_owned(),
            String::new(),
            "\ttabbed\t".to_owned(),
        ])
        .await;
        assert_eq!(io.pending_lines().await, 2);
        assert_eq!(io.prompt_line("").await.unwrap().as_deref(), Some("padded"));
        assert_eq!(io.prompt_line("").await.unwrap().as_deref(), Some("tabbed"));
        assert_eq!(io.prompt_line("").await.unwrap(), None);
    }

    #[tokio::test]
    async fn reader_is_used_after_the_queue_is_drained() {
        let (io, _buffer) = handle(&["reader line"]);
        io.seed_lines(["seed".to_owned()]).await;
        assert_eq!(io.prompt_line("").await.unwrap().as_deref(), Some("seed"));
        assert_eq!(
            io.prompt_line("").await.unwrap().as_deref(),
            Some("reader line")
        );
        assert_eq!(io.prompt_line("").await.unwrap(), None);
    }

    #[tokio::test]
    async fn prompt_text_is_written_before_the_first_seeded_line() {
        let (io, buffer) = handle(&[]);
        io.seed_lines(["seeded".to_owned()]).await;
        assert_eq!(
            io.prompt_line("You> ").await.unwrap().as_deref(),
            Some("seeded")
        );
        assert_eq!(output(&buffer), "You> ");
    }

    #[test]
    fn colour_flag_defaults_to_false_for_every_constructor() {
        assert!(!CliIo::new(Box::new(StdioReader), Box::new(StdoutWriter)).is_color());
        assert!(!CliIo::with_interactive(Box::new(StdioReader), Box::new(StdoutWriter)).is_color());
        assert!(!CliIo::stdio().is_color());
        assert!(!CliIo::stdio_non_interactive().is_color());
    }

    #[test]
    fn with_color_flips_the_flag_for_every_constructor() {
        assert!(CliIo::new(Box::new(StdioReader), Box::new(StdoutWriter))
            .with_color(true)
            .is_color());
        assert!(
            CliIo::with_interactive(Box::new(StdioReader), Box::new(StdoutWriter))
                .with_color(true)
                .is_color()
        );
        assert!(CliIo::stdio().with_color(true).is_color());
        assert!(CliIo::stdio_non_interactive().with_color(true).is_color());
        assert!(!CliIo::stdio().with_color(true).with_color(false).is_color());
    }

    #[test]
    fn with_color_keeps_the_interactivity_flag() {
        assert!(!CliIo::stdio_non_interactive()
            .with_color(true)
            .is_interactive());
        assert!(CliIo::stdio().with_color(true).is_interactive());
    }

    #[tokio::test]
    async fn with_color_shares_the_same_streams_and_queue() {
        let (io, buffer) = handle(&[]);
        io.seed_lines(["seeded".to_owned()]).await;
        let coloured = io.clone().with_color(true);
        assert_eq!(
            coloured.prompt_line("> ").await.unwrap().as_deref(),
            Some("seeded")
        );
        assert_eq!(io.pending_lines().await, 0);
        assert_eq!(output(&buffer), "> ");
    }

    #[tokio::test]
    async fn writer_keeps_text_verbatim_without_a_trailing_newline() {
        let (io, buffer) = handle(&[]);
        io.write("verbatim").await.unwrap();
        io.flush().await.unwrap();
        assert_eq!(output(&buffer), "verbatim");

        io.write_line("next").await.unwrap();
        assert_eq!(output(&buffer), "verbatimnext\n");

        io.write_blank().await.unwrap();
        assert_eq!(output(&buffer), "verbatimnext\n\n");
    }

    #[tokio::test]
    async fn interactive_handle_still_reads_scripted_lines() {
        let (io, buffer) = interactive(&["answer"]);
        assert!(io.is_interactive());
        assert_eq!(
            io.prompt_line("q? ").await.unwrap().as_deref(),
            Some("answer")
        );
        assert_eq!(output(&buffer), "q? ");
    }
}
