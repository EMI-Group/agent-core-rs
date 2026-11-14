//! Diagnostics-registry tests: version arbitration, clearing, caps, cross-round deduplication,
//! the change stream and an end-to-end push through a real [`LspClient`] and the fake server.
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use crate::lsp::client::LspClient;
use crate::lsp::diagnostics::{
    DiagnosticRegistry, DiagnosticsChange, DiagnosticsConfig, DiagnosticsFile,
};
use crate::lsp::protocol::{Position, Range};
use crate::lsp::transport::ChannelTransport;
use crate::lsp::wire::{Diagnostic, DiagnosticsEvent};

use super::fake_server::{self, FakeServer};

/// The root URI the end-to-end test initializes against.
const ROOT: &str = "file:///workspace";
/// The document URI the end-to-end test opens.
const DOC: &str = "file:///workspace/src/lib.rs";

/// Builds one diagnostic with an explicit severity and start position.
fn diagnostic(message: &str, severity: Option<u32>, line: u32, character: u32) -> Diagnostic {
    Diagnostic {
        message: message.to_owned(),
        severity,
        range: Range {
            start: Position::new(line, character),
            end: Position::new(line, character + 1),
        },
        source: Some("test".to_owned()),
        code: None,
        tags: Vec::new(),
        related_information: Vec::new(),
    }
}

/// Builds a publish event for `uri`.
fn event(uri: &str, version: Option<i64>, diagnostics: Vec<Diagnostic>) -> DiagnosticsEvent {
    DiagnosticsEvent {
        uri: uri.to_owned(),
        version,
        diagnostics,
    }
}

/// A registry with the given caps and otherwise default settings.
fn registry(max_per_file: usize, max_total: usize, max_files: usize) -> DiagnosticRegistry {
    DiagnosticRegistry::new(DiagnosticsConfig {
        max_per_file,
        max_total,
        max_files,
        ..DiagnosticsConfig::default()
    })
}

/// Polls `condition` for at most one second.
async fn eventually(mut condition: impl FnMut() -> bool) -> bool {
    for _ in 0..200 {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    false
}

#[test]
fn publish_dedupes_sorts_and_caps_per_file() {
    let registry = registry(2, 30, 256);
    registry.publish(event(
        "file:///workspace/a.rs",
        Some(1),
        vec![
            diagnostic("hint", Some(4), 0, 0),
            diagnostic("error", Some(1), 1, 0),
            // Same key as the first entry: dropped by the dedupe pass.
            diagnostic("hint", Some(4), 0, 0),
            diagnostic("info", None, 2, 0),
        ],
    ));

    let stored = registry
        .snapshot("file:///workspace/a.rs")
        .expect("buffered");
    assert_eq!(
        stored.len(),
        2,
        "per-file cap keeps only the most severe entries"
    );
    assert_eq!(stored[0].message, "error");
    assert_eq!(stored[0].severity, Some(1));
    // Unknown severity sorts as informational (3), so it precedes the hint (4).
    assert_eq!(stored[1].message, "info");
    assert_eq!(registry.pending_count(), 1);
}

#[test]
fn stale_versions_are_ignored_and_newer_ones_replace() {
    let registry = registry(10, 30, 256);
    let uri = "file:///workspace/a.rs";
    registry.publish(event(uri, Some(5), vec![diagnostic("new", Some(1), 0, 0)]));
    registry.publish(event(uri, Some(3), vec![diagnostic("old", Some(1), 0, 0)]));
    let stored = registry.snapshot(uri).expect("buffered");
    assert_eq!(stored.len(), 1);
    assert_eq!(
        stored[0].message, "new",
        "a lower version must not overwrite"
    );

    registry.publish(event(
        uri,
        Some(6),
        vec![diagnostic("newer", Some(1), 0, 0)],
    ));
    assert_eq!(registry.snapshot(uri).unwrap()[0].message, "newer");

    // An unversioned publish cannot be compared, so it replaces and stores `None`.
    registry.publish(event(
        uri,
        None,
        vec![diagnostic("unversioned", Some(2), 0, 0)],
    ));
    assert_eq!(registry.snapshot(uri).unwrap()[0].message, "unversioned");
}

#[test]
fn empty_publish_clears_and_resets_delivered_state() {
    let registry = registry(10, 30, 256);
    let uri = "file:///workspace/a.rs";
    let mut changes = registry.subscribe();
    registry.publish(event(uri, Some(1), vec![diagnostic("warn", Some(2), 0, 0)]));
    assert_eq!(
        changes.try_recv().expect("published frame"),
        DiagnosticsChange::Published(event(uri, Some(1), vec![diagnostic("warn", Some(2), 0, 0)]))
    );

    registry.publish(event(uri, Some(2), Vec::new()));
    assert_eq!(
        changes.try_recv().expect("cleared frame"),
        DiagnosticsChange::Cleared {
            uri: uri.to_owned()
        }
    );
    assert!(registry.snapshot(uri).is_none());
    assert_eq!(registry.pending_count(), 0);
    assert_eq!(registry.drain(), Vec::<DiagnosticsFile>::new());

    // The clear reset the delivered history: republishing the same diagnostic reports it again.
    registry.publish(event(uri, Some(3), vec![diagnostic("warn", Some(2), 0, 0)]));
    assert_eq!(registry.drain().len(), 1);
}

#[test]
fn max_files_evicts_the_least_recently_updated_uri() {
    let registry = registry(10, 30, 2);
    for index in 1..=3 {
        registry.publish(event(
            &format!("file:///workspace/{index}.rs"),
            Some(1),
            vec![diagnostic("warn", Some(2), 0, 0)],
        ));
    }
    assert!(registry.snapshot("file:///workspace/1.rs").is_none());
    assert!(registry.snapshot("file:///workspace/2.rs").is_some());
    assert!(registry.snapshot("file:///workspace/3.rs").is_some());
    assert_eq!(registry.pending_count(), 2);

    // Touching 2 makes 3 the oldest, so publishing 4 evicts 3.
    registry.publish(event(
        "file:///workspace/2.rs",
        Some(2),
        vec![diagnostic("error", Some(1), 0, 0)],
    ));
    registry.publish(event(
        "file:///workspace/4.rs",
        Some(1),
        vec![diagnostic("warn", Some(2), 0, 0)],
    ));
    assert!(registry.snapshot("file:///workspace/3.rs").is_none());
    assert!(registry.snapshot("file:///workspace/2.rs").is_some());
    assert!(registry.snapshot("file:///workspace/4.rs").is_some());
}

#[test]
fn drain_caps_globally_and_dedupes_across_rounds() {
    let registry = registry(2, 3, 256);
    registry.publish(event(
        "file:///workspace/a.rs",
        Some(1),
        vec![
            diagnostic("a1", Some(1), 0, 0),
            diagnostic("a2", Some(2), 1, 0),
            diagnostic("a3", Some(3), 2, 0),
        ],
    ));
    registry.publish(event(
        "file:///workspace/b.rs",
        Some(1),
        vec![
            diagnostic("b1", Some(1), 0, 0),
            diagnostic("b2", Some(2), 1, 0),
        ],
    ));

    let files: Vec<DiagnosticsFile> = registry.drain();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].uri, "file:///workspace/a.rs");
    assert_eq!(
        files[0]
            .diagnostics
            .iter()
            .map(|item| item.message.as_str())
            .collect::<Vec<_>>(),
        vec!["a1", "a2"]
    );
    assert_eq!(files[0].local_path, "/workspace/a.rs");
    assert_eq!(
        files[1].diagnostics.len(),
        1,
        "global cap clips the second file"
    );
    assert_eq!(files[1].diagnostics[0].message, "b1");
    assert_eq!(registry.pending_count(), 0);

    // Everything returned before is suppressed, even when it is published again.
    assert_eq!(registry.drain(), Vec::<DiagnosticsFile>::new());
    registry.publish(event(
        "file:///workspace/a.rs",
        Some(2),
        vec![
            diagnostic("a1", Some(1), 0, 0),
            diagnostic("a4", Some(1), 3, 0),
        ],
    ));
    let files = registry.drain();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].diagnostics.len(), 1);
    assert_eq!(files[0].diagnostics[0].message, "a4");
}

#[test]
fn clear_drops_buffered_and_delivered_state() {
    let registry = registry(10, 30, 256);
    let uri = "file:///workspace/a.rs";
    registry.publish(event(uri, Some(1), vec![diagnostic("warn", Some(2), 0, 0)]));
    assert_eq!(registry.drain().len(), 1);
    registry.publish(event(uri, Some(2), vec![diagnostic("warn", Some(2), 0, 0)]));
    assert_eq!(
        registry.drain(),
        Vec::<DiagnosticsFile>::new(),
        "already delivered"
    );

    registry.clear();
    assert_eq!(registry.pending_count(), 0);
    registry.publish(event(uri, Some(3), vec![diagnostic("warn", Some(2), 0, 0)]));
    assert_eq!(registry.drain().len(), 1, "clear also reset delivered keys");
}

#[tokio::test]
async fn did_open_publishes_diagnostics_through_the_client() {
    let registry = Arc::new(DiagnosticRegistry::new(DiagnosticsConfig::default()));
    let (transport, peer) = ChannelTransport::new(64 * 1024);
    let client = LspClient::new(Box::new(transport), Duration::from_secs(2));
    registry.attach(&client);
    let mut changes = registry.subscribe();
    let server: FakeServer = fake_server::spawn(peer);

    client.initialize(ROOT, json!({})).await.unwrap();
    client.did_open(DOC, "rust", "fn main() {}").await.unwrap();

    assert!(
        eventually(|| registry.snapshot(DOC).is_some()).await,
        "the fake publishes one warning after didOpen"
    );
    let stored = registry.snapshot(DOC).unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].message, "fake warning");
    assert_eq!(stored[0].severity, Some(2));

    let change = tokio::time::timeout(Duration::from_secs(1), changes.recv())
        .await
        .expect("a change frame")
        .expect("an open channel");
    match change {
        DiagnosticsChange::Published(event) => {
            assert_eq!(event.uri, DOC);
            assert_eq!(event.version, Some(1));
            assert_eq!(event.diagnostics.len(), 1);
        }
        DiagnosticsChange::Cleared { uri } => panic!("expected Published, got Cleared for {uri}"),
    }

    // The fake clears diagnostics when the changed text contains "clean".
    client
        .did_change(DOC, "fn main() { /* clean */ }")
        .await
        .unwrap();
    assert!(
        eventually(|| registry.snapshot(DOC).is_none()).await,
        "an empty publish clears the document"
    );
    assert!(server.wait_for_count("textDocument/didChange", 1).await);
    client.shutdown().await;
}

#[test]
fn delivered_history_is_bounded_by_max_files() {
    let registry = registry(10, 30, 4);
    // Two rounds, so six documents are drained even though the buffer holds four at a time.
    for index in 0..4 {
        registry.publish(event(
            &format!("file:///workspace/{index}.rs"),
            Some(1),
            vec![diagnostic("warn", Some(2), 0, 0)],
        ));
    }
    assert_eq!(registry.drain().len(), 4, "every document is reported once");
    for index in 4..6 {
        registry.publish(event(
            &format!("file:///workspace/{index}.rs"),
            Some(1),
            vec![diagnostic("warn", Some(2), 0, 0)],
        ));
    }
    assert_eq!(registry.drain().len(), 2);

    // Six delivery histories were recorded but `max_files` (4) is the bound, so the two oldest
    // were dropped: the oldest document reports again, a recent one stays suppressed.
    registry.publish(event(
        "file:///workspace/0.rs",
        Some(2),
        vec![diagnostic("warn", Some(2), 0, 0)],
    ));
    registry.publish(event(
        "file:///workspace/5.rs",
        Some(2),
        vec![diagnostic("warn", Some(2), 0, 0)],
    ));
    let files = registry.drain();
    assert_eq!(files.len(), 1, "only the evicted history reports again");
    assert_eq!(files[0].uri, "file:///workspace/0.rs");
}

#[test]
fn delivered_history_follows_evicted_documents_out_of_the_buffer() {
    let registry = registry(10, 30, 2);
    let first = "file:///workspace/1.rs";
    registry.publish(event(
        first,
        Some(1),
        vec![diagnostic("warn", Some(2), 0, 0)],
    ));
    assert_eq!(registry.drain().len(), 1);
    // Re-publish the first document (now suppressed), then push it out of the buffer with two
    // fresh documents; the eviction must take its delivered history with it.
    registry.publish(event(
        first,
        Some(2),
        vec![diagnostic("warn", Some(2), 0, 0)],
    ));
    for index in 2..=3 {
        registry.publish(event(
            &format!("file:///workspace/{index}.rs"),
            Some(1),
            vec![diagnostic("warn", Some(2), 0, 0)],
        ));
    }
    assert!(
        registry.snapshot(first).is_none(),
        "evicted from the buffer"
    );
    let files = registry.drain();
    assert_eq!(files.len(), 2);
    assert!(files.iter().all(|file| file.uri != first));
    // With the history gone the document is reportable again instead of being suppressed by a
    // key set the registry no longer has any reason to keep.
    registry.publish(event(
        first,
        Some(3),
        vec![diagnostic("warn", Some(2), 0, 0)],
    ));
    let files = registry.drain();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].uri, first);
}
