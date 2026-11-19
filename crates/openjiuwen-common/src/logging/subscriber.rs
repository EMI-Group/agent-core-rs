//! Subscriber configuration and conversion into a `tracing` subscriber.
//!
//! Port of Python `openjiuwen/core/common/logging/log_config.py` (the
//! user-visible knobs: level, format, destination, ANSI, filter string). Python's
//! `LogManager` installed handlers on the `logging` module; the Rust SDK instead
//! installs exactly one global `tracing` subscriber, and every logger name becomes
//! a `tracing` target (see [`super::targets`]).
//!
//! Initialization is idempotent by design: libraries may call [`init`] defensively
//! and a host application that already installed a subscriber is never disturbed.
#![allow(clippy::module_name_repetitions)]
use super::levels::{normalize_level, LevelFilter};
use crate::config::env;
use crate::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Weak};
use std::thread;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Output format of the installed subscriber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// Human-readable single-line records (default).
    #[default]
    Text,
    /// One JSON object per record, suitable for log collectors.
    Json,
}

impl LogFormat {
    /// Returns the lowercase format name (`"text"` / `"json"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

impl std::fmt::Display for LogFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for LogFormat {
    type Err = Error;
    /// Parses `"text"` or `"json"`, case-insensitively.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for any other value.
    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(Error::validation(format!(
                "invalid log format '{other}': expected 'text' or 'json'"
            ))),
        }
    }
}

/// Destination of the installed subscriber.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LogWriter {
    /// Standard output (default).
    #[default]
    Stdout,
    /// Standard error.
    Stderr,
    /// A file opened in append+create mode; parent directories are created.
    File(PathBuf),
}

/// Configuration for [`init_with`].
///
/// [`LoggingConfig::default`] is `INFO`, text format, stdout, ANSI enabled, with no
/// custom filter string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggingConfig {
    /// Maximum enabled level; used when [`LoggingConfig::filter`] is `None`.
    pub level: LevelFilter,
    /// Output format.
    pub format: LogFormat,
    /// Write destination.
    pub writer: LogWriter,
    /// Whether ANSI styling is enabled (ignored for JSON output).
    pub ansi: bool,
    /// Full `EnvFilter` directive string (for example `"openjiuwen.llm=debug,info"`).
    pub filter: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: super::levels::DEFAULT_CONFIG_LEVEL,
            format: LogFormat::default(),
            writer: LogWriter::default(),
            ansi: true,
            filter: None,
        }
    }
}

impl LoggingConfig {
    /// Overrides the level.
    #[must_use]
    pub fn with_level(mut self, level: LevelFilter) -> Self {
        self.level = level;
        self
    }
    /// Overrides the format.
    #[must_use]
    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.format = format;
        self
    }
    /// Overrides the destination.
    #[must_use]
    pub fn with_writer(mut self, writer: LogWriter) -> Self {
        self.writer = writer;
        self
    }
    /// Overrides ANSI styling.
    #[must_use]
    pub fn with_ansi(mut self, ansi: bool) -> Self {
        self.ansi = ansi;
        self
    }
    /// Sets a full `EnvFilter` directive string, which takes precedence over
    /// [`LoggingConfig::level`].
    #[must_use]
    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }
    /// Builds the configuration from `OPENJIUWEN_LOG_LEVEL`/`OPENJIUWEN_LOG_FORMAT`.
    ///
    /// Values that cannot be parsed are ignored (defaults are kept) because this
    /// constructor cannot report an error and a malformed environment variable
    /// must never prevent a process from starting. JSON output always disables
    /// ANSI styling.
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(level) = env::log_level() {
            config.level = normalize_level(&level);
        }
        if let Some(format) = env::log_format() {
            if let Ok(parsed) = LogFormat::from_str(&format) {
                config.format = parsed;
            }
        }
        if config.format == LogFormat::Json {
            config.ansi = false;
        }
        config
    }
}

/// Whether a global subscriber has been installed by this module.
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Installs the global subscriber using [`LoggingConfig::from_env`].
///
/// # Errors
///
/// Returns [`Error::Validation`] for an invalid filter directive and
/// [`Error::Config`] when the subscriber cannot be installed for a reason other
/// than an already-installed global subscriber.
pub fn init() -> Result<()> {
    init_with(&LoggingConfig::from_env())
}

/// Installs the global subscriber described by `config`.
///
/// The call is idempotent: a second call, or a call after another component
/// installed a global subscriber, returns `Ok(())` and leaves the existing
/// subscriber untouched.
///
/// # Errors
///
/// Returns [`Error::Validation`] for an invalid filter directive and
/// [`Error::Config`] when the log file cannot be opened or the subscriber cannot
/// be installed for a reason other than an already-installed global subscriber.
pub fn init_with(config: &LoggingConfig) -> Result<()> {
    if INITIALIZED.load(Ordering::SeqCst) || tracing::dispatcher::has_been_set() {
        INITIALIZED.store(true, Ordering::SeqCst);
        return Ok(());
    }
    let filter = build_filter(config)?;
    let writer = target_writer(&config.writer)?;
    let result = match config.format {
        LogFormat::Text => tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(true)
                    .with_ansi(config.ansi)
                    .with_writer(writer),
            )
            .try_init(),
        LogFormat::Json => tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(true)
                    .with_ansi(false)
                    .json()
                    .with_writer(writer),
            )
            .try_init(),
    };
    match result {
        Ok(()) => {
            INITIALIZED.store(true, Ordering::SeqCst);
            Ok(())
        }
        Err(err) if is_already_set(&err) => {
            INITIALIZED.store(true, Ordering::SeqCst);
            Ok(())
        }
        Err(err) => Err(Error::config(format!(
            "failed to install the logging subscriber: {err}"
        ))),
    }
}

/// Returns whether [`init`]/[`init_with`] installed (or observed) a subscriber.
///
/// A `false` result does not mean nothing is logged: a host application may have
/// installed its own subscriber through `tracing`.
#[must_use]
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

/// Builds the `EnvFilter` from the explicit filter string or the level.
fn build_filter(config: &LoggingConfig) -> Result<EnvFilter> {
    let spec = config
        .filter
        .clone()
        .unwrap_or_else(|| config.level.to_string());
    EnvFilter::try_new(&spec)
        .map_err(|err| Error::validation(format!("invalid log filter '{spec}': {err}")))
}

/// Detects "a global subscriber is already installed" from `try_init`'s error.
///
/// `tracing_subscriber` erases the concrete error inside `TryInitError`; its
/// `source()` returns the wrapped error's own source (which is `None` for
/// `SetGlobalDefaultError`), so the message is the only available signal.
fn is_already_set(error: &tracing_subscriber::util::TryInitError) -> bool {
    error.to_string().contains("already been set")
}

/// Destination adapter that implements [`MakeWriter`] for every [`LogWriter`].
enum TargetWriter {
    Stdout,
    Stderr,
    File(FileMakeWriter),
}

impl<'a> MakeWriter<'a> for TargetWriter {
    type Writer = Box<dyn Write + 'a>;
    fn make_writer(&'a self) -> Self::Writer {
        match self {
            Self::Stdout => Box::new(io::stdout()),
            Self::Stderr => Box::new(io::stderr()),
            Self::File(file) => Box::new(file.make_writer()),
        }
    }
}

/// Builds the destination adapter, opening the log file when needed.
fn target_writer(writer: &LogWriter) -> Result<TargetWriter> {
    Ok(match writer {
        LogWriter::Stdout => TargetWriter::Stdout,
        LogWriter::Stderr => TargetWriter::Stderr,
        LogWriter::File(path) => TargetWriter::File(FileMakeWriter::new(path)?),
    })
}

/// Number of records that may wait for the writer thread before a logging caller
/// is told the queue is full.
///
/// The queue is bounded so a stalled disk cannot grow memory without limit, and it
/// is drained by [`FileWriter::write`] with `try_send`, so a logging call never
/// blocks whatever thread produced the record (usually a tokio worker).
const WRITER_QUEUE_CAPACITY: usize = 4096;

/// One unit of work for the dedicated writer thread.
#[derive(Debug)]
enum WriterCommand {
    /// Append these bytes to the sink.
    Write(Vec<u8>),
    /// Flush the sink, then answer on the channel with the outcome.
    Flush(mpsc::Sender<io::Result<()>>),
}

/// State shared by [`FileMakeWriter`] and the [`FileWriter`]s it hands out.
///
/// The blocking file I/O lives on one dedicated thread; callers only enqueue (they
/// never hold this mutex across an I/O call) and read back the failure the writer
/// thread recorded.
#[derive(Debug)]
struct WriterShared {
    /// Queue to the writer thread; `None` once the thread has been shut down.
    commands: Mutex<Option<mpsc::SyncSender<WriterCommand>>>,
    /// First write/flush failure reported by the writer thread.
    failure: Mutex<Option<(io::ErrorKind, String)>>,
    /// The writer thread, joined when the last handle is dropped.
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

impl WriterShared {
    /// Spawns the writer thread that owns `sink`.
    fn start(sink: Box<dyn Write + Send>, capacity: usize) -> io::Result<Arc<Self>> {
        let (sender, receiver) = mpsc::sync_channel::<WriterCommand>(capacity);
        let shared = Arc::new(Self {
            commands: Mutex::new(Some(sender)),
            failure: Mutex::new(None),
            thread: Mutex::new(None),
        });
        let weak = Arc::downgrade(&shared);
        let handle = thread::Builder::new()
            .name("openjiuwen-log-writer".to_owned())
            .spawn(move || writer_loop(sink, &receiver, &weak))?;
        *shared.thread.lock() = Some(handle);
        Ok(shared)
    }

    /// Returns the recorded failure, if the sink has already failed.
    fn failure(&self) -> Option<io::Error> {
        self.failure
            .lock()
            .as_ref()
            .map(|(kind, message)| io::Error::new(*kind, message.clone()))
    }

    /// Records the first failure; later failures are follow-on symptoms.
    fn record_failure(&self, error: &io::Error) {
        let mut slot = self.failure.lock();
        if slot.is_none() {
            *slot = Some((error.kind(), error.to_string()));
        }
    }

    /// Queues one record without blocking.
    fn enqueue(&self, bytes: Vec<u8>) -> io::Result<()> {
        if let Some(error) = self.failure() {
            return Err(error);
        }
        let guard = self.commands.lock();
        let Some(sender) = guard.as_ref() else {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "the log writer thread has been shut down",
            ));
        };
        match sender.try_send(WriterCommand::Write(bytes)) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("the log writer queue is full ({WRITER_QUEUE_CAPACITY} records)"),
            )),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "the log writer thread has been shut down",
            )),
        }
    }

    /// Waits until every queued record has been written and the sink flushed.
    fn flush(&self) -> io::Result<()> {
        let (acknowledge, acknowledged) = mpsc::channel();
        {
            let guard = self.commands.lock();
            let Some(sender) = guard.as_ref() else {
                return self.failure().map_or(Ok(()), Err);
            };
            // `send` (not `try_send`) is right here: a flush is an explicit,
            // inherently synchronous request that must not drop earlier records.
            sender
                .send(WriterCommand::Flush(acknowledge))
                .map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "the log writer thread has been shut down",
                    )
                })?;
        }
        match acknowledged.recv() {
            Ok(result) => result.and_then(|()| self.failure().map_or(Ok(()), Err)),
            Err(_) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "the log writer thread has been shut down",
            )),
        }
    }
}

impl Drop for WriterShared {
    fn drop(&mut self) {
        // Dropping the queue lets the writer thread drain what is left and exit.
        drop(self.commands.lock().take());
        if let Some(handle) = self.thread.lock().take() {
            let _ = handle.join();
        }
    }
}

/// Body of the dedicated writer thread: writes every queued record in order.
fn writer_loop(
    mut sink: Box<dyn Write + Send>,
    receiver: &mpsc::Receiver<WriterCommand>,
    shared: &Weak<WriterShared>,
) {
    let record = |error: &io::Error| {
        if let Some(shared) = shared.upgrade() {
            shared.record_failure(error);
        }
    };

    while let Ok(command) = receiver.recv() {
        match command {
            WriterCommand::Write(bytes) => {
                // `write_all` retries short writes, so a partial write is never
                // reported as a complete one.
                if let Err(error) = sink.write_all(&bytes) {
                    record(&error);
                }
            }
            WriterCommand::Flush(acknowledge) => {
                let result = sink.flush();
                if let Err(error) = &result {
                    record(error);
                }
                let _ = acknowledge.send(result);
            }
        }
    }

    if let Err(error) = sink.flush() {
        record(&error);
    }
}

/// [`MakeWriter`] that appends every record to one shared log file.
///
/// The file is opened once (`append` + `create`), and every writer obtained from
/// [`FileMakeWriter::make_writer`] enqueues into one bounded queue drained by a
/// dedicated writer thread, so concurrent threads cannot interleave partial lines
/// and no logging call performs (or waits for) blocking file I/O.
///
/// Records are written in the order they were queued. [`FileMakeWriter::flush`]
/// waits for the queue to drain and reports the first write/flush failure; writes
/// issued after a failure also fail, so a caller can observe the problem instead of
/// losing records silently. Dropping the last handle closes the queue and joins the
/// writer thread.
///
/// Rotation is deliberately not implemented: a binary (the CLI) owns log rotation.
#[derive(Debug, Clone)]
pub struct FileMakeWriter {
    shared: Arc<WriterShared>,
}

impl FileMakeWriter {
    /// Opens `path` for appending, creating it and its parent directories.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the parent directory cannot be created, the
    /// file cannot be opened (permissions, a directory in the way, ...), or the
    /// writer thread cannot be spawned.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|err| {
                    Error::config(format!(
                        "failed to create log directory '{}': {err}",
                        parent.display()
                    ))
                })?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|err| {
                Error::config(format!(
                    "failed to open log file '{}': {err}",
                    path.display()
                ))
            })?;
        Self::with_sink(Box::new(file), WRITER_QUEUE_CAPACITY)
    }

    /// Builds the writer around any blocking sink; the queue capacity is injectable
    /// so tests can exercise backpressure and failure paths.
    fn with_sink(sink: Box<dyn Write + Send>, capacity: usize) -> Result<Self> {
        let shared = WriterShared::start(sink, capacity).map_err(|err| {
            Error::config(format!("failed to start the log writer thread: {err}"))
        })?;
        Ok(Self { shared })
    }

    /// Blocks until every queued record has been written and the file flushed.
    ///
    /// Logging itself never blocks; call this before a short-lived process exits if
    /// the tail of the log matters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when a write or the flush failed.
    pub fn flush(&self) -> Result<()> {
        self.shared
            .flush()
            .map_err(|err| Error::config(format!("flushing the log file failed: {err}")))
    }
}

impl<'a> MakeWriter<'a> for FileMakeWriter {
    type Writer = FileWriter;
    fn make_writer(&'a self) -> Self::Writer {
        FileWriter {
            shared: Arc::clone(&self.shared),
        }
    }
}

/// Handle returned by [`FileMakeWriter::make_writer`]; writes append to the file.
///
/// The handle only queues the record: [`Write::write`] returns as soon as the bytes
/// are accepted by the writer thread, and reports the real outcome — a full queue, a
/// shut-down thread, or a write failure already reported by the thread. Batching
/// belongs to the caller (the `tracing` layer writes one formatted record per call).
#[derive(Debug)]
pub struct FileWriter {
    shared: Arc<WriterShared>,
}

impl Write for FileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.shared.enqueue(buf.to_vec())?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.shared.flush()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::test_support::EnvGuard;
    use std::fs::File;
    use std::io::Read;
    use std::time::Duration;

    // port of tests/unit_tests/core/common/log/test_log_config.py
    #[test]
    fn default_config_matches_python_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, LevelFilter::INFO);
        assert_eq!(config.format, LogFormat::Text);
        assert_eq!(config.writer, LogWriter::Stdout);
        assert!(config.ansi);
        assert!(config.filter.is_none());
    }
    #[test]
    fn builders_override_every_field() {
        let config = LoggingConfig::default()
            .with_level(LevelFilter::DEBUG)
            .with_format(LogFormat::Json)
            .with_writer(LogWriter::Stderr)
            .with_ansi(false)
            .with_filter("openjiuwen.llm=debug");
        assert_eq!(config.level, LevelFilter::DEBUG);
        assert_eq!(config.format, LogFormat::Json);
        assert_eq!(config.writer, LogWriter::Stderr);
        assert!(!config.ansi);
        assert_eq!(config.filter.as_deref(), Some("openjiuwen.llm=debug"));
    }
    #[test]
    fn log_format_from_str_accepts_both_names_and_rejects_others() {
        assert_eq!("text".parse::<LogFormat>().unwrap(), LogFormat::Text);
        assert_eq!("JSON".parse::<LogFormat>().unwrap(), LogFormat::Json);
        assert_eq!(" Json ".parse::<LogFormat>().unwrap(), LogFormat::Json);
        let error = "yaml".parse::<LogFormat>().unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert_eq!(LogFormat::Text.as_str(), "text");
        assert_eq!(LogFormat::Json.to_string(), "json");
    }
    #[test]
    fn from_env_reads_level_and_format_and_disables_ansi_for_json() {
        let guard = EnvGuard::new(&["OPENJIUWEN_LOG_LEVEL", "OPENJIUWEN_LOG_FORMAT"]);
        guard.set("OPENJIUWEN_LOG_LEVEL", "critical");
        guard.set("OPENJIUWEN_LOG_FORMAT", "json");
        let config = LoggingConfig::from_env();
        assert_eq!(config.level, LevelFilter::ERROR);
        assert_eq!(config.format, LogFormat::Json);
        assert!(!config.ansi, "JSON output must not contain ANSI escapes");
    }
    #[test]
    fn from_env_ignores_invalid_values() {
        let guard = EnvGuard::new(&["OPENJIUWEN_LOG_LEVEL", "OPENJIUWEN_LOG_FORMAT"]);
        guard.set("OPENJIUWEN_LOG_LEVEL", "chatty");
        guard.set("OPENJIUWEN_LOG_FORMAT", "yaml");
        let config = LoggingConfig::from_env();
        assert_eq!(config.level, LevelFilter::WARN);
        assert_eq!(config.format, LogFormat::Text);
    }
    #[test]
    fn invalid_filter_directive_is_a_validation_error() {
        let error = build_filter(&LoggingConfig::default().with_filter("openjiuwen=notalevel"))
            .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }
    #[test]
    fn init_with_is_idempotent() {
        // The global subscriber can be installed only once per process, so this
        // test tolerates a subscriber installed by another test or the harness.
        assert!(init_with(&LoggingConfig::default()).is_ok());
        assert!(init_with(&LoggingConfig::default()).is_ok());
        assert!(is_initialized());
    }
    #[test]
    fn file_make_writer_appends_to_the_target_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("openjiuwen.log");
        let writer = FileMakeWriter::new(&path).unwrap();
        let mut handle = writer.make_writer();
        handle.write_all(b"first line\n").unwrap();
        handle.flush().unwrap();
        drop(handle);
        let mut handle = writer.make_writer();
        handle.write_all(b"second line\n").unwrap();
        handle.flush().unwrap();
        drop(handle);
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "first line\nsecond line\n");
    }
    #[test]
    fn file_make_writer_reports_unopenable_paths() {
        let dir = tempfile::tempdir().unwrap();
        // A directory cannot be opened as an append target.
        let error = FileMakeWriter::new(dir.path()).unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Config);
    }

    /// Sink whose writes always fail, standing in for a full or broken disk.
    struct FailingSink;

    impl Write for FailingSink {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "simulated write failure",
            ))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "simulated flush failure",
            ))
        }
    }

    /// Sink that blocks every write until it is released, recording what it got.
    #[derive(Clone, Default)]
    struct BlockingSink {
        released: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        written: Arc<Mutex<Vec<u8>>>,
    }

    impl BlockingSink {
        fn release(&self) {
            let (lock, condvar) = &*self.released;
            *lock.lock().unwrap() = true;
            condvar.notify_all();
        }

        fn written(&self) -> Vec<u8> {
            self.written.lock().clone()
        }
    }

    impl Write for BlockingSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let (lock, condvar) = &*self.released;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = condvar.wait(released).unwrap();
            }
            drop(released);
            self.written.lock().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn file_writer_reports_a_failing_sink() {
        let writer = FileMakeWriter::with_sink(Box::new(FailingSink), 8).unwrap();
        let mut handle = writer.make_writer();

        // The record may be accepted (queued before the thread fails) or already
        // rejected; either way the failure must surface instead of being swallowed.
        if let Err(error) = handle.write_all(b"record\n") {
            assert_eq!(error.kind(), io::ErrorKind::StorageFull);
        }

        let error = handle.flush().unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::StorageFull);
        assert!(
            error.to_string().contains("simulated"),
            "the real sink error is propagated: {error}"
        );

        // Once the writer thread has failed, later writes fail too.
        let error = handle.write(b"again").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::StorageFull);

        let error = writer.flush().unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Config);
        assert!(error.to_string().contains("simulated"), "{error}");
    }

    #[test]
    fn file_writer_does_not_block_the_caller_on_a_slow_sink() {
        let sink = BlockingSink::default();
        let writer = FileMakeWriter::with_sink(Box::new(sink.clone()), 8).unwrap();

        // The first record parks the writer thread inside `write`.
        let mut first = writer.make_writer();
        first.write_all(b"first\n").unwrap();

        // A second write from another thread must return without waiting for the
        // sink: the queue accepts it and the caller is released immediately.
        let mut second = writer.make_writer();
        let (done, finished) = mpsc::channel();
        let caller = thread::spawn(move || {
            let result = second.write_all(b"second\n");
            let _ = done.send(result.is_ok());
        });
        assert!(
            finished.recv_timeout(Duration::from_secs(10)).unwrap(),
            "a logging call must not wait for the file I/O"
        );
        caller.join().unwrap();
        assert!(
            sink.written().is_empty(),
            "nothing is written while the sink is blocked"
        );

        sink.release();
        writer.flush().unwrap();
        assert_eq!(
            sink.written(),
            b"first\nsecond\n".to_vec(),
            "queued records keep their order"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_writes_all_reach_the_file_intact_and_in_order() {
        const TASKS: usize = 8;
        const RECORDS: usize = 25;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent.log");
        let writer = FileMakeWriter::new(&path).unwrap();

        let mut tasks = Vec::new();
        for task in 0..TASKS {
            let writer = writer.clone();
            tasks.push(tokio::spawn(async move {
                for record in 0..RECORDS {
                    let mut handle = writer.make_writer();
                    handle
                        .write_all(format!("task-{task}-record-{record}\n").as_bytes())
                        .unwrap();
                }
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        writer.flush().unwrap();

        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), TASKS * RECORDS, "every record was written");

        let mut next_per_task = [0_usize; TASKS];
        for line in lines {
            // No interleaving or truncation: every line is one whole record.
            let rest = line
                .strip_prefix("task-")
                .unwrap_or_else(|| panic!("torn line: {line:?}"));
            let (task, record) = rest
                .split_once("-record-")
                .unwrap_or_else(|| panic!("torn line: {line:?}"));
            let task: usize = task.parse().unwrap();
            let record: usize = record.parse().unwrap();
            assert_eq!(
                record, next_per_task[task],
                "records from one task keep their order: {line:?}"
            );
            next_per_task[task] += 1;
        }
        assert_eq!(next_per_task, [RECORDS; TASKS]);
    }
}
