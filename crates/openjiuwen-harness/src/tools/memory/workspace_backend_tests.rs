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
#[tokio::test]
async fn search_scopes_the_tree_and_filters_session_logs() {
    let dir = tempfile::tempdir().expect("temp workspace");
    for (path, text) in [
        (
            "memory/MEMORY.md",
            "# Memory\n\nArgoCD deploys the service.",
        ),
        (
            "memory/sessions/s1/log.jsonl",
            "{\"text\":\"ArgoCD rollout\"}",
        ),
        (
            "memory/sessions/s2/log.jsonl",
            "{\"text\":\"ArgoCD rollback\"}",
        ),
        (
            "coding_memory/build.md",
            "---\nname: build\ndescription: d\ntype: project\n---\n\nArgoCD",
        ),
        ("USER.md", "# User\n\nArgoCD owner."),
    ] {
        let path = dir.path().join(path);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("parent dir");
        std::fs::write(path, text).expect("fixture");
    }
    let workspace = Workspace::for_language(dir.path(), "en");
    let backend = WorkspaceMemoryBackend::new(workspace);

    let request = |scope: MemoryScope, session_key: Option<&str>| MemorySearchRequest {
        query: "argocd".to_owned(),
        max_results: 20,
        min_score: 0.0,
        session_key: session_key.map(str::to_owned),
        scope,
    };

    let memory = backend
        .search(request(MemoryScope::Memory, None))
        .await
        .expect("memory search");
    let paths: Vec<&str> = memory["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter_map(|hit| hit["path"].as_str())
        .collect();
    assert_eq!(
        paths,
        [
            "USER.md",
            "memory/MEMORY.md",
            "memory/sessions/s1/log.jsonl",
            "memory/sessions/s2/log.jsonl",
        ],
        "the long-term scope searches the memory tree plus USER.md"
    );

    let scoped = backend
        .search(request(MemoryScope::Memory, Some("s1")))
        .await
        .expect("scoped search");
    let paths: Vec<&str> = scoped["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter_map(|hit| hit["path"].as_str())
        .collect();
    assert_eq!(
        paths,
        [
            "USER.md",
            "memory/MEMORY.md",
            "memory/sessions/s1/log.jsonl"
        ],
        "session_key keeps only its own transcript"
    );

    let coding = backend
        .search(request(MemoryScope::CodingMemory, None))
        .await
        .expect("coding search");
    let paths: Vec<&str> = coding["results"]
        .as_array()
        .expect("results")
        .iter()
        .filter_map(|hit| hit["path"].as_str())
        .collect();
    assert_eq!(paths, ["coding_memory/build.md"]);
}

#[tokio::test]
async fn search_of_a_missing_tree_is_empty_not_an_error() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let backend = WorkspaceMemoryBackend::new(Workspace::for_language(dir.path(), "en"));

    for scope in [MemoryScope::Memory, MemoryScope::CodingMemory] {
        let data = backend
            .search(MemorySearchRequest {
                query: "anything".to_owned(),
                max_results: 5,
                min_score: 0.0,
                session_key: None,
                scope,
            })
            .await
            .expect("search of a missing tree");
        assert_eq!(data["results"].as_array().map(Vec::len), Some(0));
        assert_eq!(data["disabled"], serde_json::json!(false));
    }
}
