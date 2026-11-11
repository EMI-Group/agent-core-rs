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
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

/// [`MakeWriter`] that appends every record to one shared log file.
///
/// The file is opened once (`append` + `create`), and all writers obtained from
/// [`FileMakeWriter::make_writer`] share it through a mutex, so concurrent
/// threads cannot interleave partial lines. Rotation is deliberately not
/// implemented: a binary (the CLI) owns log rotation.
#[derive(Debug, Clone)]
pub struct FileMakeWriter {
    file: Arc<Mutex<File>>,
}

impl FileMakeWriter {
    /// Opens `path` for appending, creating it and its parent directories.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the parent directory cannot be created or
    /// the file cannot be opened (permissions, a directory in the way, ...).
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
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
        })
    }
}

impl<'a> MakeWriter<'a> for FileMakeWriter {
    type Writer = FileWriter;
    fn make_writer(&'a self) -> Self::Writer {
        FileWriter {
            file: Arc::clone(&self.file),
        }
    }
}

/// Handle returned by [`FileMakeWriter::make_writer`]; writes append to the file.
#[derive(Debug)]
pub struct FileWriter {
    file: Arc<Mutex<File>>,
}

impl Write for FileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // `tracing_subscriber`'s fmt layer discards the `io::Result` of its
        // writer, and a logging backend cannot abort the application either, so a
        // failed write is dropped instead of being surfaced as a panic. The
        // failure mode that matters (an unwritable path) is reported by
        // `FileMakeWriter::new`.
        let _ = self.file.lock().write(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        let _ = self.file.lock().flush();
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::test_support::EnvGuard;
    use std::io::Read;

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
}
