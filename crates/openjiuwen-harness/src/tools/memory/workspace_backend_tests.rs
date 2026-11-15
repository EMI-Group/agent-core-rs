//! Unit tests of [`WorkspaceMemoryBackend`](super::WorkspaceMemoryBackend).
//!
//! Kept beside `workspace_backend.rs` (through a `#[path]` module) so the implementation file
//! stays within the crate's per-file size budget.
use super::*;

/// Builds a backend over a throwaway English workspace.
fn backend() -> (tempfile::TempDir, WorkspaceMemoryBackend) {
    let dir = tempfile::tempdir().expect("temp workspace");
    let workspace = Workspace::for_language(dir.path(), "en");
    (dir, WorkspaceMemoryBackend::new(workspace))
}

#[test]
fn memory_paths_follow_the_python_mapping() {
    let (_dir, backend) = backend();

    let cases = [
        ("USER.md", "USER.md"),
        ("MEMORY.md", "memory/MEMORY.md"),
        ("memory/MEMORY.md", "memory/MEMORY.md"),
        ("2026-11-15.md", "memory/daily_memory/2026-11-15.md"),
        ("notes.md", "memory/notes.md"),
        ("nested/notes.md", "memory/notes.md"),
    ];
    for (path, expected) in cases {
        assert_eq!(
            backend
                .relative_target(MemoryScope::Memory, path)
                .expect("valid memory path"),
            expected,
            "{path}"
        );
    }
}

#[test]
fn coding_paths_require_a_markdown_name() {
    let (_dir, backend) = backend();

    assert_eq!(
        backend
            .relative_target(MemoryScope::CodingMemory, "projects/rust.md")
            .expect("valid coding path"),
        "coding_memory/rust.md"
    );
    let error = backend
        .relative_target(MemoryScope::CodingMemory, "notes.txt")
        .expect_err("txt is rejected");
    assert_eq!(error.message(), "Path must end with .md");
}

#[test]
fn traversal_and_empty_paths_are_validation_errors() {
    let (_dir, backend) = backend();

    for path in ["../x.md", "/etc/passwd", "\\evil.md", "a/../../b.md"] {
        let error = backend
            .relative_target(MemoryScope::Memory, path)
            .expect_err("unsafe path");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation,
            "{path}"
        );
    }
    assert_eq!(
        backend
            .relative_target(MemoryScope::Memory, "  ")
            .expect_err("blank path")
            .message(),
        "path is required"
    );
}

#[test]
fn line_windows_follow_python_offsets() {
    let lines = ["one", "two", "three", "four"];

    let window = LineWindow::of(&lines, 0, None);
    assert_eq!(window.text, "one\ntwo\nthree\nfour");
    assert_eq!((window.start_line, window.end_line), (1, 4));

    let window = LineWindow::of(&lines, 1, Some(2));
    assert_eq!(window.text, "one\ntwo");
    assert_eq!((window.start_line, window.end_line), (1, 2));

    let window = LineWindow::of(&lines, 3, Some(2));
    assert_eq!(window.text, "three\nfour");
    assert_eq!((window.start_line, window.end_line), (3, 4));

    let window = LineWindow::of(&lines, 9, Some(2));
    assert_eq!(window.text, "");
    assert_eq!((window.start_line, window.end_line), (5, 4));
}

#[test]
fn query_tokens_drop_short_words_and_duplicates() {
    assert_eq!(query_tokens("go to the SHOP"), ["the", "shop"]);
    assert_eq!(query_tokens("build BUILD steps"), ["build", "steps"]);
    assert_eq!(query_tokens("go"), ["go"]);
    assert_eq!(query_tokens(" ... "), [] as [String; 0]);
}

#[test]
fn coding_front_matter_round_trips() {
    let content = "---\nname: build\ndescription: how to build\ntype: project\n---\n\ncargo build";
    let chunk = CodingChunk::parse(content).expect("valid chunk");

    assert_eq!(chunk.front_matter.get("name"), Some("build"));
    assert_eq!(chunk.body, "cargo build");
    assert!(chunk.front_matter.get("created_at").is_some());
    let rendered = render_coding_document(&chunk.front_matter, &chunk.body);
    assert!(rendered.starts_with("---\nname: build\ndescription: how to build\ntype: project\n"));
    assert!(rendered.ends_with("---\n\ncargo build"));
}

#[test]
fn coding_front_matter_reports_the_python_messages() {
    let error = CodingChunk::parse("no front matter").expect_err("missing front matter");
    assert_eq!(error.message(), FRONT_MATTER_REQUIRED);

    let error = CodingChunk::parse("---\nname: x\ntype: project\n---\n\nbody")
        .expect_err("missing description");
    assert_eq!(error.message(), "Missing required field: description");

    let error = CodingChunk::parse("---\nname: x\ndescription: y\ntype: other\n---\n\nbody")
        .expect_err("bad type");
    assert_eq!(error.message(), CODING_TYPES_MESSAGE);

    let error = CodingChunk::parse("---\nname: x\ndescription: y\ntype: user\n---\n\n")
        .expect_err("empty body");
    assert_eq!(error.message(), EMPTY_BODY);
}

#[test]
fn daily_names_match_only_iso_dates() {
    assert!(is_daily_name("2026-11-15.md"));
    assert!(!is_daily_name("2026-11-15.MD"));
    assert!(!is_daily_name("26-11-15.md"));
    assert!(!is_daily_name("2026-11-15.md.bak"));
}
