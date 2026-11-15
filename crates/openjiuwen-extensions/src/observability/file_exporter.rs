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
//! are irrelevant, and replaying means posting every line with `POST` in turn.
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
//!   reports success: telemetry never breaks the caller. The loss is not silent
//!   for a host, though: every failure is recorded on the exporter
//!   ([`TraceFileExporter::dropped_spans`],
//!   [`TraceFileExporter::failed_exports`], [`TraceFileExporter::last_error`]),
//!   and a handle built with [`ExporterType::File`] exposes the exporter through
//!   `TracerHandle::file_exporter`, so a host can assert that export is working
//!   while `export` itself stays infallible.
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
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
///
/// The loss is nevertheless observable: [`Self::dropped_spans`],
/// [`Self::failed_exports`] and [`Self::last_error`] record what could not be
/// written, so a host can tell "no spans yet" from "every span is being dropped".
/// Clones share that state (and the append lock), exactly like
/// [`CaptureExporter`](crate::otel::provider::CaptureExporter), which is how a
/// provider handle keeps observing the exporter it handed to the SDK.
#[derive(Debug, Clone)]
pub struct TraceFileExporter {
    /// Everything a clone must share; behind an [`Arc`] so the handle keeps
    /// observing the provider's exporter.
    state: Arc<ExportState>,
}

/// The state shared by every clone of one [`TraceFileExporter`].
#[derive(Debug)]
struct ExportState {
    /// Directory the daily trace files live in.
    root_dir: PathBuf,
    /// Days a trace file is kept; `0` disables cleanup entirely.
    retention_days: u32,
    /// Resource the provider handed over in `set_resource`.
    resource: Mutex<Option<Resource>>,
    /// Serializes appends (Python's `_lock`) and counts export calls
    /// (Python's `_write_count`), so the cleanup interval cannot drift.
    appends: Mutex<u64>,
    /// Spans that never reached disk (serialization or append failure).
    dropped_spans: AtomicU64,
    /// Exports whose append failed as a whole.
    failed_exports: AtomicU64,
    /// The most recent failure, for a host that wants to diagnose it.
    last_error: Mutex<Option<String>>,
}

impl TraceFileExporter {
    /// Creates an exporter writing under `root_dir`, keeping files for
    /// `retention_days` days (`0` never prunes).
    ///
    /// The directory is created eagerly like Python's `os.makedirs(..., exist_ok=True)`;
    /// a failure is only logged, because a broken traces directory must not
    /// break span export or the provider that owns this exporter. It is recorded
    /// in [`Self::last_error`], and every span exported afterwards shows up in
    /// [`Self::dropped_spans`].
    #[must_use]
    pub fn new(root_dir: impl Into<PathBuf>, retention_days: u32) -> Self {
        let root_dir = root_dir.into();
        let state = ExportState {
            root_dir,
            retention_days,
            resource: Mutex::new(None),
            appends: Mutex::new(0),
            dropped_spans: AtomicU64::new(0),
            failed_exports: AtomicU64::new(0),
            last_error: Mutex::new(None),
        };
        let exporter = Self {
            state: Arc::new(state),
        };
        if let Err(error) = fs::create_dir_all(exporter.root_dir()) {
            warn!(
                target: LOG_TARGET,
                path = %exporter.root_dir().display(),
                %error,
                "cannot create the traces directory; spans keep being dropped for this exporter"
            );
            exporter.record_error(format!(
                "cannot create the traces directory {}: {error}",
                exporter.root_dir().display()
            ));
        }
        exporter
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
        &self.state.root_dir
    }

    /// The retention window in days; `0` means cleanup is disabled.
    #[must_use]
    pub fn retention_days(&self) -> u32 {
        self.state.retention_days
    }

    /// The number of spans that never reached disk.
    ///
    /// Counts the spans of every failed export (an append that could not open,
    /// write, or flush today's file) plus every span line that could not be
    /// serialized. A non-zero value means telemetry is being dropped even though
    /// [`SpanExporter::export`] keeps reporting success.
    #[must_use]
    pub fn dropped_spans(&self) -> u64 {
        self.state.dropped_spans.load(Ordering::Acquire)
    }

    /// The number of exports whose append failed as a whole.
    ///
    /// One failed export drops all of its spans; the counter therefore moves once
    /// per lost batch, while [`Self::dropped_spans`] moves by the batch size.
    #[must_use]
    pub fn failed_exports(&self) -> u64 {
        self.state.failed_exports.load(Ordering::Acquire)
    }

    /// The most recent failure, rendered for a host that wants to log or surface it.
    ///
    /// `None` while every export has reached disk. The value is a diagnostic
    /// message, never a credential: it names the path and the I/O error.
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        lock(&self.state.last_error).clone()
    }

    /// Records a failure message as [`Self::last_error`].
    fn record_error(&self, message: String) {
        *lock(&self.state.last_error) = Some(message);
    }

    /// Records `count` lost spans and the failure that lost them.
    fn record_dropped(&self, count: u64, message: String) {
        self.state.dropped_spans.fetch_add(count, Ordering::AcqRel);
        self.record_error(message);
    }

    /// The trace file an export taken now would append to.
    ///
    /// The name comes from the **local** calendar date, matching Python's
    /// `time.strftime('%Y-%m-%d')`, so a process that runs across local midnight
    /// starts a new file.
    #[must_use]
    pub fn traces_file(&self) -> PathBuf {
        self.root_dir().join(trace_file_name())
    }

    /// Encodes one standalone OTLP JSON line per span.
    ///
    /// The body comes from the same encoder the OTLP/HTTP exporter uses
    /// (`crate::otel::otlp::json::encode_body`), so a line here is the request
    /// the collector would have received, minus the batching. A span whose line
    /// cannot be serialized is counted as dropped and recorded as
    /// [`Self::last_error`], because it will never reach the trace file.
    fn encode_lines(&self, batch: &[SpanData]) -> Vec<String> {
        let resource = lock(&self.state.resource).clone();
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
                Err(error) => {
                    warn!(
                        target: LOG_TARGET,
                        span = %span.name,
                        %error,
                        "skipped a span line that could not be serialized"
                    );
                    self.record_dropped(
                        1,
                        format!("could not serialize the span {:?}: {error}", span.name),
                    );
                }
            }
        }
        lines
    }

    /// Appends `lines` to today's file and returns the new export count.
    ///
    /// The append lock is held for the whole write, so two exporters sharing a
    /// directory (or the batch-processor thread racing shutdown) cannot
    /// interleave partial lines. An empty batch writes nothing but still counts
    /// as an export, exactly like Python. A failed append drops the whole batch:
    /// it is counted in [`Self::failed_exports`] and [`Self::dropped_spans`], and
    /// it is kept as [`Self::last_error`], while the export itself stays
    /// successful.
    fn append(&self, lines: &[String]) -> u64 {
        let mut count = lock(&self.state.appends);
        if !lines.is_empty() {
            let path = self.traces_file();
            if let Err(error) = append_lines(&path, lines) {
                warn!(
                    target: LOG_TARGET,
                    path = %path.display(),
                    %error,
                    "trace file append failed"
                );
                // The batch is lost, but the export still reports success; the
                // counters are how a host observes the loss.
                self.state.failed_exports.fetch_add(1, Ordering::AcqRel);
                self.record_dropped(
                    u64::try_from(lines.len()).unwrap_or(u64::MAX),
                    format!("cannot append to {}: {error}", path.display()),
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
        if self.retention_days() == 0 {
            return;
        }
        let entries = match fs::read_dir(self.root_dir()) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return,
            Err(error) => {
                warn!(
                    target: LOG_TARGET,
                    path = %self.root_dir().display(),
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
            if !is_expired(modified, now, self.retention_days()) {
                continue;
            }
            match fs::remove_file(&path) {
                Ok(()) => {}
                // Removed by another process between listing and now.
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
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
    /// so telemetry cannot fail the caller's request path; it is recorded in
    /// [`TraceFileExporter::failed_exports`],
    /// [`TraceFileExporter::dropped_spans`] and
    /// [`TraceFileExporter::last_error`] instead, which is how a host observes
    /// that export stopped working.
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
        *lock(&self.state.resource) = Some(resource.clone());
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
    fn a_fresh_exporter_records_no_failure() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = TraceFileExporter::new(dir.path(), 7);
        assert_eq!(exporter.dropped_spans(), 0);
        assert_eq!(exporter.failed_exports(), 0);
        assert_eq!(exporter.last_error(), None);
    }

    #[test]
    fn a_failed_append_is_counted_and_reported_while_export_stays_infallible() {
        // A regular file where the traces directory should be: `create_dir_all`
        // and every append fail, which is exactly the fail-open path.
        let dir = tempfile::tempdir().unwrap();
        let not_a_dir = dir.path().join("not-a-directory");
        std::fs::write(&not_a_dir, b"x").unwrap();
        let exporter = TraceFileExporter::new(&not_a_dir, 0);
        assert!(
            exporter.last_error().is_some(),
            "the directory-creation failure must be recorded"
        );

        let count = exporter.append(&[String::from("{}\n"), String::from("{}\n")]);
        assert_eq!(count, 1, "a failed append still counts as an export");
        assert_eq!(exporter.failed_exports(), 1);
        assert_eq!(exporter.dropped_spans(), 2);
        let error = exporter.last_error().unwrap();
        assert!(
            error.contains(TRACE_FILE_PREFIX),
            "the error should name the trace file: {error}"
        );

        // A second, empty batch writes nothing and reports success, like Python.
        let outcome = futures::executor::block_on(exporter.export(Vec::new()));
        assert!(outcome.is_ok());
        assert_eq!(
            exporter.failed_exports(),
            1,
            "an empty batch cannot fail an append"
        );
    }

    #[test]
    fn clones_share_the_failure_state() {
        let dir = tempfile::tempdir().unwrap();
        let not_a_dir = dir.path().join("not-a-directory");
        std::fs::write(&not_a_dir, b"x").unwrap();
        let exporter = TraceFileExporter::new(&not_a_dir, 0);
        let clone = exporter.clone();
        clone.append(&[String::from("{}\n")]);
        assert_eq!(exporter.failed_exports(), 1);
        assert_eq!(exporter.dropped_spans(), 1);
        assert!(exporter.last_error().is_some());
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
    fn exporter_is_send_sync_and_static() {
        // The SDK owns the exporter for the provider lifetime and drives it from
        // the batch-processor thread.
        fn assert_bounds<T: Send + Sync + 'static>() {}
        assert_bounds::<TraceFileExporter>();
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
