//! Unit tests for the `glob`, `list_files` and `grep` tools.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use serde_json::{json, Value};
use tempfile::TempDir;

use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_sysop::operation::FsOperation;
use openjiuwen_sysop::{LocalWorkConfig, SysOperation};

use crate::tools::support::ToolEnv;
use crate::workspace::Workspace;

use super::grep::{
    apply_head_limit, scan_grep_with, FsSource, GrepFile, GrepSource, CHUNK_BYTES, MAX_FILE_BYTES,
};
use super::*;

/// A local filesystem backend rooted at `dir`.
fn backend(dir: &TempDir) -> Option<Arc<dyn FsOperation>> {
    let operation = SysOperation::local(LocalWorkConfig::unrestricted(
        dir.path().to_string_lossy().to_string(),
    ))
    .unwrap();
    operation.fs()
}

/// A `glob` tool over `dir`.
fn glob_tool(dir: &TempDir) -> GlobTool {
    GlobTool::new(ToolEnv::new(Workspace::new(dir.path())), backend(dir))
}

/// A `list_files` tool over `dir`.
fn list_tool(dir: &TempDir) -> ListDirTool {
    ListDirTool::new(ToolEnv::new(Workspace::new(dir.path())), backend(dir))
}

/// A `grep` tool over `dir`.
fn grep_tool(dir: &TempDir) -> GrepTool {
    GrepTool::new(ToolEnv::new(Workspace::new(dir.path())))
}

/// Writes `content` to `relative` below `dir`, creating parent directories.
fn write(dir: &TempDir, relative: &str, content: &str) {
    let path = dir.path().join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, content).unwrap();
}

/// Invokes `tool` with a fresh context and unwraps the payload envelope.
async fn invoke<T: Tool>(tool: &T, inputs: Value) -> Value {
    tool.invoke(inputs, &ToolContext::new()).await.unwrap()
}

/// The string entries of an array value.
fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap().to_owned())
        .collect()
}

/// The sorted property names of a card.
fn properties(card: &ToolCard) -> Vec<String> {
    let mut keys: Vec<String> = card.input_params["properties"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    keys.sort_unstable();
    keys
}

#[test]
fn cards_expose_the_python_arguments() {
    let glob = GlobTool::card_for("en");
    assert_eq!((glob.name.as_str(), glob.parallel_safe), ("glob", true));
    assert_eq!(glob.input_params["required"], json!(["pattern"]));
    assert_eq!(properties(&glob), ["path", "pattern"]);

    let list = ListDirTool::card_for("en");
    assert_eq!(
        (list.name.as_str(), list.parallel_safe),
        ("list_files", true)
    );
    assert!(list.input_params.get("required").is_none());
    assert_eq!(properties(&list), ["path", "show_hidden"]);

    let grep = GrepTool::card_for("en");
    assert_eq!((grep.name.as_str(), grep.parallel_safe), ("grep", true));
    assert_eq!(grep.input_params["required"], json!(["pattern"]));
    assert_eq!(
        properties(&grep),
        [
            "-A",
            "-B",
            "-C",
            "-i",
            "-n",
            "context",
            "glob",
            "head_limit",
            "ignore_case",
            "multiline",
            "offset",
            "output_mode",
            "path",
            "pattern",
            "type",
        ]
    );
}

#[test]
fn brace_expansion_matches_python() {
    assert_eq!(GlobTool::expand_braces("*.rs"), ["*.rs"]);
    assert_eq!(GlobTool::expand_braces("*.{rs,toml}"), ["*.rs", "*.toml"]);
    assert_eq!(
        GlobTool::expand_braces("src/*.{rs, toml}"),
        ["src/*.rs", "src/*.toml"]
    );
    // Fewer than two alternatives and nested brace groups stay unchanged.
    assert_eq!(GlobTool::expand_braces("*.{rs}"), ["*.{rs}"]);
    assert_eq!(GlobTool::expand_braces("{a,{b,c}}"), ["{a,{b,c}}"]);
    assert_eq!(GlobTool::expand_braces("{a,b}/{c,d}").len(), 4);
}

#[test]
fn output_mode_parsing_follows_python() {
    assert_eq!(GrepOutputMode::parse(""), Some(GrepOutputMode::Content));
    assert_eq!(GrepOutputMode::parse("count"), Some(GrepOutputMode::Count));
    assert_eq!(
        GrepOutputMode::parse("files_with_matches"),
        Some(GrepOutputMode::FilesWithMatches)
    );
    assert_eq!(GrepOutputMode::parse("nope"), None);
    assert_eq!(GrepOutputMode::Content.as_str(), "content");
    assert_eq!(GrepOutputMode::Count.as_str(), "count");
}

#[tokio::test]
async fn glob_finds_nested_files_through_the_backend() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "a");
    write(&dir, "src/b.rs", "b");
    write(&dir, "src/deep/c.rs", "c");
    let tool = glob_tool(&dir);

    let out = invoke(&tool, json!({"pattern": "**/*.rs"})).await;

    assert_eq!(out["success"], json!(true));
    assert_eq!(out["error"], Value::Null);
    assert_eq!(out["data"]["numFiles"], json!(2));
    assert_eq!(out["data"]["count"], json!(2));
    assert_eq!(out["data"]["truncated"], json!(false));
    assert!(out["data"]["durationMs"].is_u64());
    let mut filenames = strings(&out["data"]["filenames"]);
    filenames.sort();
    assert_eq!(filenames, ["src/b.rs", "src/deep/c.rs"]);
    let matching = strings(&out["data"]["matching_files"]);
    assert_eq!(matching.len(), 2);
    assert!(matching.iter().all(|path| path.starts_with('/')));
    assert_eq!(tool.render_for_llm(&out).lines().count(), 2);
}

#[tokio::test]
async fn glob_relativizes_against_an_explicit_path() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "src/b.rs", "b");
    write(&dir, "a.rs", "a");
    let tool = glob_tool(&dir);

    let out = invoke(&tool, json!({"pattern": "*.rs", "path": "src"})).await;

    assert_eq!(out["success"], json!(true));
    assert_eq!(strings(&out["data"]["filenames"]), ["b.rs"]);
    assert_eq!(strings(&out["data"]["matching_files"]).len(), 1);
}

#[tokio::test]
async fn glob_keeps_the_first_hundred_results() {
    let dir = tempfile::tempdir().unwrap();
    for index in 0..105 {
        write(&dir, &format!("file{index:03}.txt"), "x");
    }
    let tool = glob_tool(&dir);

    let out = invoke(&tool, json!({"pattern": "*.txt"})).await;

    assert_eq!(out["data"]["numFiles"], json!(100));
    assert_eq!(out["data"]["count"], json!(100));
    assert_eq!(out["data"]["truncated"], json!(true));
    assert_eq!(strings(&out["data"]["filenames"]).len(), 100);
    assert_eq!(strings(&out["data"]["matching_files"]).len(), 100);
    assert!(tool.render_for_llm(&out).ends_with(
        "(Results truncated to the first 100 files; use a more specific path or pattern.)"
    ));
}

#[tokio::test]
async fn glob_reports_no_files_and_validation_failures() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "a");
    let tool = glob_tool(&dir);

    let empty = invoke(&tool, json!({"pattern": "*.md"})).await;
    assert_eq!(empty["success"], json!(true));
    assert_eq!(empty["data"]["numFiles"], json!(0));
    assert_eq!(empty["data"]["truncated"], json!(false));
    assert_eq!(strings(&empty["data"]["filenames"]), Vec::<String>::new());
    assert_eq!(tool.render_for_llm(&empty), "No files found.");

    for inputs in [json!({}), json!({"pattern": ""})] {
        let out = invoke(&tool, inputs).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("pattern is required"));
    }

    let escaping = invoke(&tool, json!({"pattern": "*", "path": "../"})).await;
    assert_eq!(escaping["success"], json!(false));
    assert!(escaping["error"]
        .as_str()
        .unwrap()
        .starts_with("Invalid path"));
}

#[tokio::test]
async fn missing_backends_are_failure_payloads() {
    let dir = tempfile::tempdir().unwrap();
    let env = ToolEnv::new(Workspace::new(dir.path()));
    for out in [
        invoke(
            &GlobTool::new(env.clone(), None),
            json!({"pattern": "*.rs"}),
        )
        .await,
        invoke(&ListDirTool::new(env, None), json!({})).await,
    ] {
        assert_eq!(out["success"], json!(false));
        assert_eq!(
            out["error"],
            json!("filesystem operation is not configured")
        );
    }
}

#[tokio::test]
async fn list_files_sorts_and_hides_dotted_entries() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "alpha/x.txt", "a");
    write(&dir, "beta/y.txt", "b");
    write(&dir, ".hidden/z.txt", "c");
    write(&dir, "b.txt", "b");
    write(&dir, "a.txt", "a");
    write(&dir, ".dot", "d");
    let tool = list_tool(&dir);

    let out = invoke(&tool, json!({})).await;

    assert_eq!(out["success"], json!(true));
    assert_eq!(strings(&out["data"]["files"]), ["a.txt", "b.txt"]);
    assert_eq!(strings(&out["data"]["dirs"]), ["alpha", "beta"]);
    assert_eq!(tool.render_for_llm(&out), "alpha/\nbeta/\na.txt\nb.txt");

    let hidden = invoke(&tool, json!({"show_hidden": true})).await;
    assert_eq!(
        strings(&hidden["data"]["files"]),
        [".dot", "a.txt", "b.txt"]
    );
    assert_eq!(
        strings(&hidden["data"]["dirs"]),
        [".hidden", "alpha", "beta"]
    );
}

/// Writes the fixture used by the grep content tests.
fn content_fixture(dir: &TempDir) {
    write(
        dir,
        "notes.txt",
        "alpha\nneedle here\nbeta\ngamma\nneedle again\ndelta\n",
    );
}

#[tokio::test]
async fn grep_content_mode_renders_matches_and_context() {
    let dir = tempfile::tempdir().unwrap();
    content_fixture(&dir);
    let tool = grep_tool(&dir);

    let out = invoke(&tool, json!({"pattern": "needle", "-C": 1})).await;

    assert_eq!(out["success"], json!(true));
    assert_eq!(out["error"], Value::Null);
    assert_eq!(out["data"]["mode"], json!("content"));
    assert_eq!(out["data"]["exit_code"], json!(0));
    assert_eq!(out["data"]["numLines"], json!(6));
    assert_eq!(out["data"]["count"], json!(6));
    assert_eq!(out["data"]["numFiles"], json!(0));
    assert_eq!(out["data"]["appliedOffset"], Value::Null);
    assert_eq!(out["data"]["appliedLimit"], Value::Null);
    assert_eq!(
        out["data"]["stdout"],
        json!(concat!(
            "notes.txt-1-alpha\nnotes.txt:2:needle here\nnotes.txt-3-beta\n",
            "notes.txt-4-gamma\nnotes.txt:5:needle again\nnotes.txt-6-delta"
        ))
    );
    assert_eq!(tool.render_for_llm(&out), out["data"]["stdout"]);

    let bare = invoke(&tool, json!({"pattern": "needle", "-n": false})).await;
    assert_eq!(
        bare["data"]["stdout"],
        json!("notes.txt:needle here\nnotes.txt:needle again")
    );

    let before = invoke(&tool, json!({"pattern": "gamma", "-B": 1, "-A": 0})).await;
    assert_eq!(
        before["data"]["stdout"],
        json!("notes.txt-3-beta\nnotes.txt:4:gamma")
    );
}

#[tokio::test]
async fn grep_ignore_case_and_the_i_flag() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "shout.txt", "HELLO\nworld\n");
    let tool = grep_tool(&dir);

    let sensitive = invoke(&tool, json!({"pattern": "hello"})).await;
    assert_eq!(sensitive["success"], json!(true));
    assert_eq!(sensitive["data"]["exit_code"], json!(1));

    let insensitive = invoke(&tool, json!({"pattern": "hello", "ignore_case": true})).await;
    assert_eq!(insensitive["data"]["exit_code"], json!(0));
    assert_eq!(insensitive["data"]["stdout"], json!("shout.txt:1:HELLO"));

    // `-i` wins whenever it is present, in either direction.
    let short = invoke(
        &tool,
        json!({"pattern": "hello", "ignore_case": false, "-i": true}),
    )
    .await;
    assert_eq!(short["data"]["stdout"], json!("shout.txt:1:HELLO"));

    let overridden = invoke(
        &tool,
        json!({"pattern": "hello", "ignore_case": true, "-i": false}),
    )
    .await;
    assert_eq!(overridden["data"]["exit_code"], json!(1));
}

#[tokio::test]
async fn grep_modes_and_glob_filter() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "src/a.rs", "token\ntoken\n");
    write(&dir, "b.txt", "token\n");
    write(&dir, "c.txt", "nothing\n");
    let tool = grep_tool(&dir);

    let filtered = invoke(&tool, json!({"pattern": "token", "glob": "*.rs"})).await;
    assert_eq!(
        filtered["data"]["stdout"],
        json!("src/a.rs:1:token\nsrc/a.rs:2:token")
    );

    let files = invoke(
        &tool,
        json!({"pattern": "token", "output_mode": "files_with_matches"}),
    )
    .await;
    assert_eq!(files["data"]["mode"], json!("files_with_matches"));
    assert_eq!(strings(&files["data"]["filenames"]), ["b.txt", "src/a.rs"]);
    assert_eq!(files["data"]["numFiles"], json!(2));
    assert_eq!(files["data"]["count"], json!(2));
    assert_eq!(files["data"]["stdout"], json!("b.txt\nsrc/a.rs"));

    let counts = invoke(&tool, json!({"pattern": "token", "output_mode": "count"})).await;
    assert_eq!(counts["data"]["mode"], json!("count"));
    assert_eq!(counts["data"]["stdout"], json!("b.txt:1\nsrc/a.rs:2"));
    assert_eq!(counts["data"]["numFiles"], json!(2));
    assert_eq!(counts["data"]["numMatches"], json!(3));
    assert_eq!(counts["data"]["count"], json!(3));
    assert_eq!(strings(&counts["data"]["filenames"]), Vec::<String>::new());
}

#[tokio::test]
async fn grep_applies_head_limit_and_offset() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "lines.txt", "x\nx\nx\nx\nx\n");
    let tool = grep_tool(&dir);

    // A numeric string is accepted for `head_limit`, like the shell tool's timeout.
    let limited = invoke(
        &tool,
        json!({"pattern": "x", "head_limit": "2", "path": "lines.txt"}),
    )
    .await;
    assert_eq!(
        limited["data"]["stdout"],
        json!("lines.txt:1:x\nlines.txt:2:x")
    );
    assert_eq!(limited["data"]["appliedLimit"], json!(2));
    assert_eq!(limited["data"]["appliedOffset"], Value::Null);
    assert_eq!(
        tool.render_for_llm(&limited),
        concat!(
            "lines.txt:1:x\nlines.txt:2:x\n",
            "(Results truncated to 2 entries; pass offset to see more.)"
        )
    );

    let skipped = invoke(
        &tool,
        json!({"pattern": "x", "head_limit": 2, "offset": 2, "path": "lines.txt"}),
    )
    .await;
    assert_eq!(
        skipped["data"]["stdout"],
        json!("lines.txt:3:x\nlines.txt:4:x")
    );
    assert_eq!(skipped["data"]["appliedOffset"], json!(2));
    assert_eq!(skipped["data"]["appliedLimit"], json!(2));

    let unlimited = invoke(
        &tool,
        json!({"pattern": "x", "head_limit": 0, "path": "lines.txt"}),
    )
    .await;
    assert_eq!(unlimited["data"]["numLines"], json!(5));
    assert_eq!(unlimited["data"]["appliedLimit"], Value::Null);
}

#[tokio::test]
async fn grep_without_matches_succeeds_with_exit_code_one() {
    let dir = tempfile::tempdir().unwrap();
    content_fixture(&dir);
    let tool = grep_tool(&dir);

    let out = invoke(&tool, json!({"pattern": "absent"})).await;

    assert_eq!(out["success"], json!(true));
    assert_eq!(out["data"]["exit_code"], json!(1));
    assert_eq!(out["data"]["stdout"], json!(""));
    assert_eq!(out["data"]["numLines"], json!(0));
    assert_eq!(tool.render_for_llm(&out), "No matches found.");
}

#[tokio::test]
async fn grep_skips_vcs_directories_and_binary_files() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, ".git/config.txt", "token\n");
    write(&dir, "a.txt", "token\n");
    write(&dir, "blob.bin", "\u{0}token\n");
    let tool = grep_tool(&dir);

    let out = invoke(&tool, json!({"pattern": "token"})).await;

    assert_eq!(out["data"]["stdout"], json!("a.txt:1:token"));
}

#[tokio::test]
async fn grep_validation_failures_match_the_python_messages() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "token\n");
    let tool = grep_tool(&dir);

    let cases = [
        (
            json!({"pattern": "token", "output_mode": "nope"}),
            "output_mode must be one of: content, files_with_matches, count",
        ),
        (
            json!({"pattern": "token", "type": "rust"}),
            "type filter requires ripgrep (rg) to be installed",
        ),
        (
            json!({"pattern": "token", "multiline": true}),
            "multiline search requires ripgrep (rg) to be installed",
        ),
    ];
    for (inputs, expected) in cases {
        let out = invoke(&tool, inputs).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!(expected));
    }

    for inputs in [json!({}), json!({"pattern": ""})] {
        let out = invoke(&tool, inputs).await;
        assert_eq!(out["error"], json!("pattern is required"));
    }

    let invalid = invoke(&tool, json!({"pattern": "("})).await;
    assert_eq!(invalid["success"], json!(false));
    assert!(invalid["error"]
        .as_str()
        .unwrap()
        .starts_with("invalid pattern: "));
}

#[test]
fn search_reports_validation_failures() {
    let dir = tempfile::tempdir().unwrap();
    let tool = grep_tool(&dir);

    let args: GrepArgs = serde_json::from_value(json!({"pattern": ""})).unwrap();
    let error = tool.search(&args).unwrap_err();
    assert_eq!(error.error.as_deref(), Some("pattern is required"));

    let args: GrepArgs = serde_json::from_value(json!({
        "pattern": "token",
        "output_mode": "Nope",
    }))
    .unwrap();
    let error = tool.search(&args).unwrap_err();
    assert_eq!(
        error.error.as_deref(),
        Some("output_mode must be one of: content, files_with_matches, count")
    );
}

#[test]
fn head_limit_helper_matches_python() {
    let items: Vec<String> = (0..5).map(|index| index.to_string()).collect();
    let all: Vec<String> = items.clone();
    assert_eq!(apply_head_limit(items.clone(), None, 0), (all, None));
    assert_eq!(
        apply_head_limit(items.clone(), Some(2), 1),
        (vec!["1".to_owned(), "2".to_owned()], Some(2))
    );
    assert_eq!(
        apply_head_limit(items, Some(0), 3),
        (vec!["3".to_owned(), "4".to_owned()], None)
    );
}

/// A [`GrepSource`] wrapper that records every candidate it opens.
struct CountingSource {
    inner: FsSource,
    opened: std::sync::Mutex<Vec<std::path::PathBuf>>,
}

impl CountingSource {
    fn new() -> Self {
        Self {
            inner: FsSource,
            opened: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// The file names opened so far, sorted by open order.
    fn opened(&self) -> Vec<String> {
        self.opened
            .lock()
            .unwrap()
            .iter()
            .filter_map(|path| {
                path.file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .map(str::to_owned)
            })
            .collect()
    }
}

impl GrepSource for CountingSource {
    fn size(&self, path: &std::path::Path) -> Option<u64> {
        self.inner.size(path)
    }

    fn open(&self, path: &std::path::Path) -> std::io::Result<Box<dyn GrepFile>> {
        self.opened.lock().unwrap().push(path.to_path_buf());
        self.inner.open(path)
    }
}

/// Parses `grep` arguments out of their JSON wire form.
fn grep_args(inputs: Value) -> GrepArgs {
    serde_json::from_value(inputs).unwrap()
}

#[tokio::test]
async fn grep_finds_a_match_that_spans_a_chunk_boundary() {
    let dir = tempfile::tempdir().unwrap();
    // `needle` starts three bytes before the first chunk ends, so the match itself is split
    // across the boundary between the two chunks of the same line.
    let mut content = String::new();
    content.push_str(&"q".repeat(CHUNK_BYTES - 3));
    content.push_str("needle");
    content.push_str(&"q".repeat(20));
    content.push('\n');
    content.push_str("tail\n");
    write(&dir, "long.txt", &content);

    let tool = grep_tool(&dir);
    let out = invoke(&tool, json!({"pattern": "needle"})).await;
    assert_eq!(out["data"]["exit_code"], json!(0));
    let stdout = out["data"]["stdout"].as_str().unwrap();
    assert!(stdout.starts_with("long.txt:1:"), "{}", &stdout[..40]);
    assert!(stdout.contains("needle"), "the boundary match is reported");
    assert!(
        stdout.len() > CHUNK_BYTES,
        "the whole line came through the carry"
    );

    // A pattern that only exists on the far side of the boundary is found too.
    let tail = invoke(&tool, json!({"pattern": "^tail$"})).await;
    assert_eq!(tail["data"]["stdout"], json!("long.txt:2:tail"));
}

#[test]
fn grep_stops_scanning_once_head_limit_is_exhausted() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "hit\nhit\nhit\nhit\n");
    write(&dir, "zzz.txt", "hit\n");
    let source = CountingSource::new();
    let data = scan_grep_with(
        dir.path(),
        &grep_args(json!({"pattern": "hit", "head_limit": 2})),
        &source,
    )
    .unwrap();
    assert_eq!(data["stdout"], json!("a.txt:1:hit\na.txt:2:hit"));
    assert_eq!(data["appliedLimit"], json!(2));
    assert_eq!(source.opened(), vec!["a.txt".to_string()]);
}

#[test]
fn grep_skips_files_above_the_size_cap_with_a_counted_note() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "hit\n");
    // One byte over the cap: the scan never opens it, it only counts it.
    let oversized = format!(
        "hit\n{}",
        "q".repeat(usize::try_from(MAX_FILE_BYTES).unwrap())
    );
    write(&dir, "big.txt", &oversized);
    let source = CountingSource::new();
    let data = scan_grep_with(dir.path(), &grep_args(json!({"pattern": "hit"})), &source).unwrap();
    assert_eq!(data["exit_code"], json!(0));
    assert_eq!(data["stdout"], json!("a.txt:1:hit"));
    assert_eq!(data["skippedLargeFiles"], json!(1));
    assert_eq!(
        data["note"],
        json!(format!(
            "skipped 1 file(s) larger than {MAX_FILE_BYTES} bytes"
        ))
    );
    assert_eq!(source.opened(), vec!["a.txt".to_string()]);
    // The note reaches the model, not just the payload.
    let tool = grep_tool(&dir);
    let rendered = tool.render_for_llm(&json!({"success": true, "data": data, "error": null}));
    assert!(
        rendered.contains("skipped 1 file(s) larger than"),
        "{rendered}"
    );
}
