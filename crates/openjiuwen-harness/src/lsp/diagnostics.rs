//! Buffering and delivery of `textDocument/publishDiagnostics` notifications.
//!
//! Ports `openjiuwen/harness/lsp/core/diagnostic_registry.py`: push diagnostics arrive
//! asynchronously from the server's read loop, are buffered per document, and are drained by
//! the agent loop at a point where it can turn them into model-visible text.
//!
//! # Deviation from Python
//!
//! Python exposes a process-global singleton (`LspDiagnosticRegistry.get_instance`). This
//! crate forbids undocumented global mutable state, so the registry is an owned value: a host
//! constructs one, shares it as an [`Arc`], and attaches it to each client it starts. `LspTool`
//! owns one registry; two tools in one process therefore never see each other's diagnostics.
//!
//! # Semantics
//!
//! - Publishing is version-aware: a non-empty publish whose `version` is lower than the stored
//!   version for that URI is dropped. A newer (or unversioned) publish replaces the stored set.
//! - An empty diagnostic list clears the URI: stored and already-delivered state is reset and a
//!   [`DiagnosticsChange::Cleared`] frame is emitted.
//! - Entries are deduplicated by `message|severity|line:character|code`, sorted by severity
//!   (unknown severity counts as 3), and capped at [`DiagnosticsConfig::max_per_file`] per
//!   document and [`DiagnosticsConfig::max_files`] documents (the least recently updated is
//!   evicted first).
//! - [`DiagnosticRegistry::drain`] mirrors Python's `get_and_clear`: merge the stored sets,
//!   dedupe within the round, suppress keys already delivered in an earlier round, sort by
//!   severity, apply the per-file and global caps, then mark what it returned as delivered.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::broadcast;

use super::client::LspClient;
use super::uri::uri_to_path;
use super::wire::{parse_diagnostics_event, Diagnostic, DiagnosticCode, DiagnosticsEvent};

/// Method name the registry listens for.
const PUBLISH_METHOD: &str = "textDocument/publishDiagnostics";

/// Severity used when a server omits it (LSP `Information`).
const DEFAULT_SEVERITY: u32 = 3;

/// Caps applied while buffering and delivering diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, rename_all = "snake_case")]
pub struct DiagnosticsConfig {
    /// Maximum diagnostics kept per document.
    pub max_per_file: usize,
    /// Maximum diagnostics returned by one [`DiagnosticRegistry::drain`].
    pub max_total: usize,
    /// Maximum documents buffered at once; the least recently updated is evicted.
    pub max_files: usize,
    /// Capacity of the change broadcast channel.
    pub channel_capacity: usize,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            max_per_file: 10,
            max_total: 30,
            max_files: 256,
            channel_capacity: 16,
        }
    }
}

/// All diagnostics [`DiagnosticRegistry::drain`] returned for one document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct DiagnosticsFile {
    /// `file://` URI of the document.
    pub uri: String,
    /// Document version the server computed the diagnostics for, when it reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<i64>,
    /// The (deduplicated, severity-sorted, capped) diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Native path decoded from `uri`; empty when the URI is not a `file://` URI.
    pub local_path: String,
}

/// One frame of the change stream returned by [`DiagnosticRegistry::subscribe`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DiagnosticsChange {
    /// A server published diagnostics for a document.
    Published(DiagnosticsEvent),
    /// A server resolved every diagnostic for a document.
    Cleared {
        /// `file://` URI whose diagnostics were cleared.
        uri: String,
    },
}

/// One buffered document.
#[derive(Debug, Clone)]
struct PendingEntry {
    /// Deduplicated, severity-sorted, capped diagnostics.
    diagnostics: Vec<Diagnostic>,
    /// Document version the server reported, when it reported one.
    version: Option<i64>,
    /// Monotonic arrival counter, used for the `max_files` eviction order.
    sequence: u64,
}

/// Registry state under one lock.
#[derive(Debug, Default)]
struct RegistryState {
    /// Buffered documents keyed by URI.
    pending: BTreeMap<String, PendingEntry>,
    /// Keys already returned by a previous [`DiagnosticRegistry::drain`], per URI.
    delivered: BTreeMap<String, BTreeSet<String>>,
    /// Monotonic arrival counter.
    sequence: u64,
}

/// Buffers push diagnostics and delivers them in bounded, deduplicated batches.
///
/// Owned by the host (see the module docs for the singleton deviation); share it with
/// [`Arc`] and call [`DiagnosticRegistry::attach`] once per [`LspClient`].
#[derive(Debug)]
pub struct DiagnosticRegistry {
    /// Caps and channel capacity.
    config: DiagnosticsConfig,
    /// Buffered and delivered state.
    state: Mutex<RegistryState>,
    /// Change stream fan-out.
    changes: broadcast::Sender<DiagnosticsChange>,
}

impl DiagnosticRegistry {
    /// Creates an empty registry with `config`.
    #[must_use]
    pub fn new(config: DiagnosticsConfig) -> Self {
        let (changes, _receiver) = broadcast::channel(config.channel_capacity.max(1));
        Self {
            config,
            state: Mutex::new(RegistryState::default()),
            changes,
        }
    }

    /// Number of buffered documents (not diagnostics).
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.state.lock().pending.len()
    }

    /// The currently buffered diagnostics for `uri`, if any.
    #[must_use]
    pub fn snapshot(&self, uri: &str) -> Option<Vec<Diagnostic>> {
        self.state
            .lock()
            .pending
            .get(uri)
            .map(|entry| entry.diagnostics.clone())
    }

    /// Drops every buffered diagnostic and the delivered-key history.
    pub fn clear(&self) {
        let mut state = self.state.lock();
        state.pending.clear();
        state.delivered.clear();
    }

    /// Subscribes to [`DiagnosticsChange`] frames.
    ///
    /// The receiver is `Send + 'static`, so a host may forward it to another task. Frames are
    /// best effort: publishing never blocks and a full channel drops the oldest frame.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<DiagnosticsChange> {
        self.changes.subscribe()
    }

    /// Buffers one publish notification and notifies subscribers.
    ///
    /// Empty diagnostic lists clear the document; see the module docs for the version and
    /// deduplication rules.
    pub fn publish(&self, event: DiagnosticsEvent) {
        if event.diagnostics.is_empty() {
            let mut state = self.state.lock();
            state.pending.remove(&event.uri);
            state.delivered.remove(&event.uri);
            drop(state);
            let _ = self
                .changes
                .send(DiagnosticsChange::Cleared { uri: event.uri });
            return;
        }
        let diagnostics = normalize(&event.diagnostics, self.config.max_per_file);
        {
            let mut state = self.state.lock();
            let stored_version = state
                .pending
                .get(&event.uri)
                .and_then(|entry| entry.version);
            if let (Some(stored), Some(incoming)) = (stored_version, event.version) {
                if incoming < stored {
                    tracing::debug!(
                        target: "openjiuwen_harness::lsp",
                        uri = %event.uri,
                        stored,
                        incoming,
                        "ignoring stale LSP diagnostics publish"
                    );
                    return;
                }
            }
            state.sequence = state.sequence.wrapping_add(1);
            let sequence = state.sequence;
            state.pending.insert(
                event.uri.clone(),
                PendingEntry {
                    diagnostics: diagnostics.clone(),
                    version: event.version,
                    sequence,
                },
            );
            evict_excess(&mut state, self.config.max_files);
        }
        let _ = self
            .changes
            .send(DiagnosticsChange::Published(DiagnosticsEvent {
                uri: event.uri,
                version: event.version,
                diagnostics,
            }));
    }

    /// Returns everything buffered, clears the buffer and remembers what was returned.
    ///
    /// Mirrors Python's `get_and_clear`: intra-round dedupe, cross-round dedupe, severity sort,
    /// per-file cap and global cap. Returns an empty vector when nothing new is buffered.
    #[must_use]
    pub fn drain(&self) -> Vec<DiagnosticsFile> {
        let mut state = self.state.lock();
        if state.pending.is_empty() {
            return Vec::new();
        }
        let mut entries: Vec<(String, PendingEntry)> =
            std::mem::take(&mut state.pending).into_iter().collect();
        entries.sort_by_key(|(_, entry)| entry.sequence);

        let mut selected: Vec<(String, PendingEntry)> = Vec::new();
        for (uri, mut entry) in entries {
            let delivered = state.delivered.get(&uri);
            let mut seen: BTreeSet<String> = BTreeSet::new();
            entry.diagnostics.retain(|diagnostic| {
                let key = diagnostic_key(diagnostic);
                if delivered.is_some_and(|known| known.contains(&key)) {
                    return false;
                }
                seen.insert(key)
            });
            if entry.diagnostics.is_empty() {
                continue;
            }
            sort_by_severity(&mut entry.diagnostics);
            entry.diagnostics.truncate(self.config.max_per_file);
            selected.push((uri, entry));
        }

        let mut files = Vec::new();
        let mut total = 0_usize;
        for (uri, entry) in selected {
            if total >= self.config.max_total {
                break;
            }
            let remaining = self.config.max_total - total;
            let mut diagnostics = entry.diagnostics;
            diagnostics.truncate(remaining);
            total += diagnostics.len();
            for diagnostic in &diagnostics {
                state
                    .delivered
                    .entry(uri.clone())
                    .or_default()
                    .insert(diagnostic_key(diagnostic));
            }
            files.push(DiagnosticsFile {
                local_path: uri_to_path(&uri).to_string_lossy().into_owned(),
                uri,
                version: entry.version,
                diagnostics,
            });
        }
        files
    }

    /// Registers the `textDocument/publishDiagnostics` handler on `client`.
    ///
    /// Safe to call before `initialize`: the client dispatches notifications whenever they
    /// arrive. Malformed notification payloads are ignored rather than buffered, and each
    /// client gets its own registration so two servers cannot share one handler slot.
    pub fn attach(self: &Arc<Self>, client: &LspClient) {
        let registry = Arc::clone(self);
        client.add_notification_handler(
            PUBLISH_METHOD,
            Arc::new(move |params: &Value| {
                if let Some(event) = parse_diagnostics_event(params) {
                    registry.publish(event);
                }
            }),
        );
    }
}

/// Deduplicates by key, sorts by severity and caps at `max_per_file`.
fn normalize(diagnostics: &[Diagnostic], max_per_file: usize) -> Vec<Diagnostic> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut kept: Vec<Diagnostic> = diagnostics
        .iter()
        .filter(|diagnostic| seen.insert(diagnostic_key(diagnostic)))
        .cloned()
        .collect();
    sort_by_severity(&mut kept);
    kept.truncate(max_per_file);
    kept
}

/// Stable-sorts diagnostics by severity, errors first; unknown severity counts as 3.
fn sort_by_severity(diagnostics: &mut [Diagnostic]) {
    diagnostics.sort_by_key(severity_of);
}

/// The severity used for ordering, defaulting an absent value to [`DEFAULT_SEVERITY`].
fn severity_of(diagnostic: &Diagnostic) -> u32 {
    diagnostic.severity.unwrap_or(DEFAULT_SEVERITY)
}

/// Evicts the least recently updated documents until at most `max_files` remain.
fn evict_excess(state: &mut RegistryState, max_files: usize) {
    let cap = max_files.max(1);
    while state.pending.len() > cap {
        let oldest = state
            .pending
            .iter()
            .min_by_key(|(_, entry)| entry.sequence)
            .map(|(uri, _)| uri.clone());
        let Some(oldest) = oldest else {
            break;
        };
        state.pending.remove(&oldest);
    }
}

/// The stable dedupe key: `message|severity|line:character|code`.
fn diagnostic_key(diagnostic: &Diagnostic) -> String {
    format!(
        "{}|{}|{}:{}|{}",
        diagnostic.message,
        severity_of(diagnostic),
        diagnostic.range.start.line,
        diagnostic.range.start.character,
        code_text(diagnostic.code.as_ref()),
    )
}

/// Renders a diagnostic code as text; absent codes render as an empty string.
fn code_text(code: Option<&DiagnosticCode>) -> String {
    match code {
        Some(DiagnosticCode::String(text)) => text.clone(),
        Some(DiagnosticCode::Number(number)) => number.to_string(),
        None => String::new(),
    }
}
