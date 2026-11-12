//! Injectable line-oriented IO used by the CLI.
//!
//! The runner and the REPL never touch `stdin`/`stdout` directly: they hold a
//! cloneable [`CliIo`] handle, so tests drive a scripted terminal instead of the
//! process streams. [`StdioReader`] and [`StdoutWriter`] are the real,
//! terminal-backed implementations; they move the blocking std calls into
//! [`tokio::task::spawn_blocking`] so no reactor thread ever blocks.
//!
//! Failure mapping: OS and stdio failures (unreadable stdin, closed stdout,
//! `JoinError` from the blocking pool) carry no domain meaning, so they are
//! wrapped with [`Error::external`], which preserves the original error as the
//! [`std::error::Error::source`] for logging and `Debug`.

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

/// The guarded pair of input reader and output writer.
type IoStreams = (Box<dyn LineReader>, Box<dyn Writer>);

/// Cloneable handle over the CLI's input and output streams.
///
/// Both the runner and the permission host hold a clone: every call locks the
/// shared state for the duration of one read or write only, so no lock is ever
/// held across an `.await` of the agent stream.
#[derive(Clone)]
pub struct CliIo {
    inner: Arc<Mutex<IoStreams>>,
    interactive: bool,
}

impl CliIo {
    /// Builds a handle from an explicit reader and writer (the test path).
    #[must_use]
    pub fn new(reader: Box<dyn LineReader>, writer: Box<dyn Writer>) -> Self {
        Self {
            inner: Arc::new(Mutex::new((reader, writer))),
            interactive: false,
        }
    }

    /// Builds the interactive handle backed by the process stdio streams.
    ///
    /// The `interactive` flag is what later makes permission prompts
    /// answerable; a handle built with [`CliIo::new`] fails closed instead.
    #[must_use]
    pub fn stdio() -> Self {
        Self {
            inner: Arc::new(Mutex::new((Box::new(StdioReader), Box::new(StdoutWriter)))),
            interactive: true,
        }
    }

    /// Writes `prompt`, flushes it, then reads and trims one line.
    ///
    /// Returns `Ok(None)` at end of input.
    ///
    /// # Errors
    ///
    /// Returns the writer or reader error when prompting or reading fails.
    pub async fn prompt_line(&self, prompt: &str) -> Result<Option<String>> {
        let mut guard = self.inner.lock().await;
        let (reader, writer) = &mut *guard;
        writer.write(prompt).await?;
        writer.flush().await?;
        let line = reader.read_line().await?;
        Ok(line.map(|line| line.trim().to_owned()))
    }

    /// Writes `text` verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the sink cannot be written.
    pub async fn write(&self, text: &str) -> Result<()> {
        let mut guard = self.inner.lock().await;
        guard.1.write(text).await
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
        guard.1.flush().await
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
