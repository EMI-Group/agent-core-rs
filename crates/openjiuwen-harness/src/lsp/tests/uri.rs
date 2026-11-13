//! `file://` URI conversion tests.
use std::path::{Path, PathBuf};

use crate::lsp::uri::{path_to_uri, uri_to_path};

#[cfg(unix)]
#[test]
fn path_to_uri_percent_encodes_spaces_and_keeps_slashes() {
    assert_eq!(
        path_to_uri(Path::new("/tmp/lsp dir/lib.rs")),
        "file:///tmp/lsp%20dir/lib.rs"
    );
}

#[cfg(unix)]
#[test]
fn uri_to_path_decodes_percent_escapes() {
    assert_eq!(
        uri_to_path("file:///tmp/lsp%20dir/lib.rs"),
        PathBuf::from("/tmp/lsp dir/lib.rs")
    );
    assert_eq!(
        uri_to_path("file:///tmp/a%23b%25c.rs"),
        PathBuf::from("/tmp/a#b%c.rs")
    );
}

#[test]
fn uri_to_path_leaves_non_file_uris_unchanged() {
    assert_eq!(
        uri_to_path("untitled:Untitled-1"),
        PathBuf::from("untitled:Untitled-1")
    );
}

#[test]
fn uri_round_trips_a_temporary_path_with_spaces() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a b").join("c.rs");
    let uri = path_to_uri(&path);
    assert!(uri.starts_with("file://"));
    assert!(uri.contains("a%20b"));
    assert_eq!(uri_to_path(&uri), path);
}

#[test]
fn uri_to_path_keeps_malformed_escapes_verbatim() {
    assert_eq!(
        uri_to_path("file:///tmp/a%zzb.rs"),
        PathBuf::from("/tmp/a%zzb.rs")
    );
}
