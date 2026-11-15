//! Collector-free OTLP-JSON span exporter: one append-only JSONL file per day.
//!
//! Port of `openjiuwen/extensions/observability/file_exporter.py` (v0.1.20).
//! Python's `TraceFileExporter` writes every finished span as a standalone
//! OTLP JSON `ExportTraceServiceRequest` line
//! (`resourceSpans` → `scopeSpans` → `spans` with exactly one span), carrying
//! `traceId`/`spanId`/`parentSpanId` as lowercase hex strings — the format a
//! Collector (or Langfuse) ingests directly via `POST /v1/traces`. Spans from
//! different traces interleave in one file: the collector rebuilds each trace
//! from the ids carried on every span, so physical ordering and per-trace files
//! are irrelevant, and replaying is just POSTing each line in turn.
//!
//! # Semantics kept from Python
//!
//! - One file per calendar day, `traces-<YYYY-MM-DD>.jsonl` directly under
//!   `root_dir`. The date is the **local** date ([`chrono::Local`]), so the file
//!   rolls over at local midnight at Python's `time.strftime('%Y-%m-%d')` does.
//! - Appends go straight to disk; the exporter buffers nothing, so
//!   [`SpanExporter::force_flush`] and [`SpanExporter::shutdown_with_timeout`]
//!   are no-ops and batching stays the `BatchSpanProcessor`'s job.
//! - A failed append (open/write/flush) is logged at `warn` and the export still
//!   reports success: telemetry never breaks the caller.
//! - Trace files whose mtime predates `retention_days` are lazily pruned at most
//!   every `CLEANUP_INTERVAL` (64) exports; `retention_days == 0` disables
//!   cleanup and cleanup never fails an export.
//!
//! # Runtime model
//!
//! The SDK's `BatchSpanProcessor` calls [`SpanExporter::export`] on its own
//! dedicated thread, exactly like [`crate::otel::otlp::OtlpHttpExporter`]; the
//! write path is synchronous file I/O and has no reactor requirement, and the
//! append lock is never held across an await (the returned future is already
//! ready).

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime};

use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use opentelemetry_sdk::Resource;
use tracing::warn;

use crate::otel::config::OtelTracerConfig;
use crate::otel::otlp::json;

/// Cleanup runs at most every N export cycles (Python's `_CLEANUP_INTERVAL`).
pub(crate) const CLEANUP_INTERVAL: u64 = 64;

/// Seconds in one retention day (Python's `_SECONDS_PER_DAY`).
const SECONDS_PER_DAY: u64 = 86_400;

/// Suffix of a trace file (Python's `entry.endswith(".jsonl")` filter).
const TRACE_FILE_SUFFIX: &str = ".jsonl";

/// Prefix of a trace file name (`traces-<YYYY-MM-DD>.jsonl`).
const TRACE_FILE_PREFIX: &str = "traces-";

/// `tracing` target for exporter-level failures.
const LOG_TARGET: &str = "openjiuwen_extensions::observability::file_exporter";

/// Appends finished spans to `<root_dir>/traces-<YYYY-MM-DD>.jsonl`, one
/// standalone single-span OTLP JSON request per line.
///
/// No collector, HTTP client or runtime is involved: the exporter writes plain
/// files, which is what makes it usable in offline and CI deployments. Pair it
/// with the provider's `BatchSpanProcessor` (what
/// [`init_otel_tracer`](crate::otel::init_otel_tracer) does) so ending a span
/// never blocks the business thread.
///
/// Every failure is fail-open: a directory that cannot be created, an append
/// that fails, and a cleanup that cannot stat or remove a file are all logged at
/// `warn` on target `openjiuwen_extensions::observability::file_exporter` and
/// never surface as an export error.
#[derive(Debug)]
pub struct TraceFileExporter {
    /// Directory the daily trace files live in.
    root_dir: PathBuf,
    /// Days a trace file is kept; `0` disables cleanup entirely.
    retention_days: u32,
    /// Resource the provider handed over in `set_resource`.
    resource: Mutex<Option<Resource>>,
    /// Serializes appends (Python's `_lock`) and counts export calls
    /// (Python's `_write_count`), so the cleanup interval cannot drift.
    appends: Mutex<u64>,
}

impl TraceFileExporter {
    /// Creates an exporter writing under `root_dir`, keeping files for
    /// `retention_days` days (`0` never prunes).
    ///
    /// The directory is created eagerly like Python's `os.makedirs(..., exist_ok=True)`;
    /// a failure is only logged, because a broken traces directory must not
    /// break span export or the provider that owns this exporter.
    #[must_use]
    pub fn new(root_dir: impl Into<PathBuf>, retention_days: u32) -> Self {
        let root_dir = root_dir.into();
        if let Err(error) = fs::create_dir_all(&root_dir) {
            warn!(
                target: LOG_TARGET,
                path = %root_dir.display(),
                %error,
                "cannot create the traces directory; spans keep being dropped for this exporter"
            );
        }
        Self {
            root_dir,
            retention_days,
            resource: Mutex::new(None),
            appends: Mutex::new(0),
        }
    }

    /// Creates an exporter from [`OtelTracerConfig::traces_dir`] and
    /// [`OtelTracerConfig::file_retention_days`].
    #[must_use]
    pub fn from_config(config: &OtelTracerConfig) -> Self {
        Self::new(config.traces_dir.clone(), config.file_retention_days)
    }

    /// The directory the daily trace files are written to.
    #[must_use]
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// The retention window in days; `0` means cleanup is disabled.
    #[must_use]
    pub fn retention_days(&self) -> u32 {
        self.retention_days
    }

    /// The trace file an export taken now would append to.
    ///
    /// The name comes from the **local** calendar date, matching Python's
    /// `time.strftime('%Y-%m-%d')`, so a process that runs across local midnight
    /// starts a new file.
    #[must_use]
    pub fn traces_file(&self) -> PathBuf {
        self.root_dir.join(trace_file_name())
    }

    /// Encodes one standalone OTLP JSON line per span.
    ///
    /// The body comes from the same encoder the OTLP/HTTP exporter uses
    /// (`crate::otel::otlp::json::encode_body`), so a line here is the request
    /// the collector would have received, minus the batching.
    fn encode_lines(&self, batch: &[SpanData]) -> Vec<String> {
        let resource = lock(&self.resource).clone();
        let mut lines = Vec::with_capacity(batch.len());
        for span in batch {
            let body = json::encode_body(resource.as_ref(), std::slice::from_ref(span));
            // `serde_json` writes compact output (`{"a":1,"b":2}`) where
            // Python's `json.dumps` defaults to `', '`/`': '`; the two bodies
            // are JSON-equivalent, and a collector parses either.
            match serde_json::to_string(&body) {
                Ok(mut line) => {
                    line.push('\n');
                    lines.push(line);
                }
                Err(error) => warn!(
                    target: LOG_TARGET,
                    span = %span.name,
                    %error,
                    "skipped a span line that could not be serialized"
                ),
            }
        }
        lines
    }

    /// Appends `lines` to today's file and returns the new export count.
    ///
    /// The append lock is held for the whole write, so two exporters sharing a
    /// directory (or the batch-processor thread racing shutdown) cannot
    /// interleave partial lines. An empty batch writes nothing but still counts
    /// as an export, exactly like Python.
    fn append(&self, lines: &[String]) -> u64 {
        let mut count = lock(&self.appends);
        if !lines.is_empty() {
            let path = self.traces_file();
            if let Err(error) = append_lines(&path, lines) {
                warn!(
                    target: LOG_TARGET,
                    path = %path.display(),
                    %error,
                    "trace file append failed"
                );
            }
        }
        *count = count.saturating_add(1);
        *count
    }

    /// Records one exported batch and runs the periodic cleanup.
    fn write_batch(&self, batch: &[SpanData]) {
        // `SpanData` is only produced after `Span::end`, so `end_time` is always
        // set; Python's defensive `end_time is None` branch has no Rust
        // equivalent and is intentionally absent.
        let lines = self.encode_lines(batch);
        if self.append(&lines) % CLEANUP_INTERVAL == 0 {
            self.cleanup_old_files();
        }
    }

    /// Deletes trace files whose mtime predates the retention cutoff.
    ///
    /// `retention_days == 0` skips the sweep. A missing directory (the exporter
    /// may never have written) and a file that vanished between listing and
    /// stat/remove are ignored; any other I/O error is logged once and skipped so
    /// a single bad file cannot abort the sweep.
    fn cleanup_old_files(&self) {
        if self.retention_days == 0 {
            return;
        }
        let entries = match fs::read_dir(&self.root_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return,
            Err(error) => {
                warn!(
                    target: LOG_TARGET,
                    path = %self.root_dir.display(),
                    %error,
                    "cannot list the traces directory"
                );
                return;
            }
        };
        let now = SystemTime::now();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    warn!(target: LOG_TARGET, %error, "cannot read a traces directory entry");
                    continue;
                }
            };
            let name = entry.file_name();
            if !name.to_string_lossy().ends_with(TRACE_FILE_SUFFIX) {
                continue;
            }
            let path = entry.path();
            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => {
                    warn!(target: LOG_TARGET, path = %path.display(), %error, "cannot stat a trace file");
                    continue;
                }
            };
            if !metadata.is_file() {
                continue;
            }
            let modified = match metadata.modified() {
                Ok(modified) => modified,
                Err(error) => {
                    warn!(target: LOG_TARGET, path = %path.display(), %error, "cannot read a trace file mtime");
                    continue;
                }
            };
            if !is_expired(modified, now, self.retention_days) {
                continue;
            }
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => {
                    warn!(target: LOG_TARGET, path = %path.display(), %error, "cannot prune a trace file");
                }
            }
        }
    }
}

impl SpanExporter for TraceFileExporter {
    /// Appends every span as one OTLP JSON line and always reports success.
    ///
    /// The future is ready on return: the write happens before it is built, like
    /// Python's synchronous `export`. An append failure is logged, not returned,
    /// so telemetry cannot fail the caller's request path.
    fn export(
        &self,
        batch: Vec<SpanData>,
    ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
        self.write_batch(&batch);
        std::future::ready(Ok(()))
    }

    /// No-op: nothing is buffered, because every export already hit disk.
    ///
    /// Python's `force_flush` returns `True`; the SDK equivalent of `true` is
    /// `Ok(())`.
    fn force_flush(&mut self) -> OTelSdkResult {
        Ok(())
    }

    /// No-op: nothing is buffered, so there is nothing to drain on shutdown.
    fn shutdown_with_timeout(&mut self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }

    fn set_resource(&mut self, resource: &Resource) {
        *lock(&self.resource) = Some(resource.clone());
    }
}

/// Reports whether a file modified at `modified` is older than `retention_days`
/// at `now`.
///
/// Python compares `getmtime(path) < time.time() - days * 86400`, i.e. an age
/// strictly greater than the window. `retention_days == 0` means "keep
/// everything" and never expires, matching the caller's early return.
pub(crate) fn is_expired(modified: SystemTime, now: SystemTime, retention_days: u32) -> bool {
    if retention_days == 0 {
        return false;
    }
    let window = u64::from(retention_days).saturating_mul(SECONDS_PER_DAY);
    now.duration_since(modified)
        .is_ok_and(|age| age > Duration::from_secs(window))
}

/// Appends complete lines with a single append-mode handle.
fn append_lines(path: &Path, lines: &[String]) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    for line in lines {
        file.write_all(line.as_bytes())?;
    }
    file.flush()
}

/// Today's trace file name, `traces-<YYYY-MM-DD>.jsonl` in local time.
fn trace_file_name() -> String {
    format!(
        "{TRACE_FILE_PREFIX}{}.jsonl",
        chrono::Local::now().format("%Y-%m-%d")
    )
}

/// Locks a mutex, recovering the data when a previous holder panicked.
///
/// Exporter callbacks are fail-open: a poisoned lock must not turn span export
/// into a panic on the batch-processor thread.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // port of tests/unit_tests/extensions/observability/test_file_exporter.py
    // (the retention predicate; the file-level behaviour is covered by
    // `tests/otel_file_exporter.rs`)

    #[test]
    fn retention_zero_never_expires() {
        let now = SystemTime::now();
        let ancient = now - Duration::from_secs(365 * SECONDS_PER_DAY);
        assert!(!is_expired(ancient, now, 0));
    }

    #[test]
    fn a_file_older_than_the_window_expires() {
        let now = SystemTime::now();
        // One second past a one-day window: Python's strict `<` cutoff is met.
        let old = now - Duration::from_secs(SECONDS_PER_DAY + 1);
        assert!(is_expired(old, now, 1));
        // Inside the window, and exactly at its edge, files are kept.
        let fresh = now - Duration::from_secs(60);
        assert!(!is_expired(fresh, now, 1));
        let edge = now - Duration::from_secs(SECONDS_PER_DAY);
        assert!(!is_expired(edge, now, 1));
    }

    #[test]
    fn a_future_mtime_never_expires() {
        let now = SystemTime::now();
        let future = now + Duration::from_secs(60);
        assert!(!is_expired(future, now, 7));
    }

    #[test]
    fn a_large_window_does_not_overflow() {
        let now = SystemTime::now();
        let old = now - Duration::from_secs(1);
        assert!(!is_expired(old, now, u32::MAX));
    }

    #[test]
    fn from_config_reads_the_file_exporter_fields() {
        let dir = tempfile::tempdir().unwrap();
        let config = OtelTracerConfig {
            traces_dir: dir.path().to_string_lossy().into_owned(),
            file_retention_days: 3,
            ..OtelTracerConfig::default()
        };
        let exporter = TraceFileExporter::from_config(&config);
        assert_eq!(exporter.root_dir(), dir.path());
        assert_eq!(exporter.retention_days(), 3);
        assert!(exporter.traces_file().starts_with(dir.path()));
    }

    #[test]
    fn trace_file_name_is_prefixed_and_jsonl_suffixed() {
        let name = trace_file_name();
        assert!(
            name.starts_with(TRACE_FILE_PREFIX),
            "unexpected name: {name}"
        );
        assert!(name.ends_with(TRACE_FILE_SUFFIX), "unexpected name: {name}");
        // `traces-YYYY-MM-DD.jsonl` has one fixed length for every date.
        assert_eq!(
            name.len(),
            TRACE_FILE_PREFIX.len() + 10 + TRACE_FILE_SUFFIX.len()
        );
    }
}
