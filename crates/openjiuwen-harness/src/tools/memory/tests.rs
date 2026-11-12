#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{Tool, ToolContext};

use super::super::descriptions;
use super::tool::NO_BACKEND_MESSAGE;
use super::*;

/// The canned backend reply of one test case.
enum Reply {
    Search(Value),
    Read(MemoryReadData),
    Write(MemoryWriteData),
    Edit(MemoryEditData),
}

impl Reply {
    /// Wraps this reply in a fresh recording backend.
    fn backend(self) -> Arc<FakeBackend> {
        let mut backend = FakeBackend::default();
        match self {
            Self::Search(data) => backend.search = Some(data),
            Self::Read(data) => backend.read = Some(data),
            Self::Write(data) => backend.write = Some(data),
            Self::Edit(data) => backend.edit = Some(data),
        }
        Arc::new(backend)
    }
}

/// A [`MemoryToolBackend`] that records each call as a compact string and replays a reply.
#[derive(Default)]
struct FakeBackend {
    calls: Mutex<Vec<String>>,
    search: Option<Value>,
    read: Option<MemoryReadData>,
    write: Option<MemoryWriteData>,
    edit: Option<MemoryEditData>,
    failure: Option<&'static str>,
}

impl FakeBackend {
    /// Records `call` and fails the call when a failure is configured.
    fn record(&self, call: String) -> Result<()> {
        self.calls.lock().unwrap().push(call);
        match self.failure {
            Some(message) => Err(Error::tool(message)),
            None => Ok(()),
        }
    }

    /// The calls seen so far, in order.
    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl MemoryToolBackend for FakeBackend {
    async fn search(&self, request: MemorySearchRequest) -> Result<Value> {
        self.record(format!(
            "search {} {} {} {:?} {:?}",
            request.query,
            request.max_results,
            request.min_score,
            request.session_key,
            request.scope
        ))?;
        self.search
            .clone()
            .ok_or_else(|| Error::tool("no search reply configured"))
    }

    async fn read(
        &self,
        scope: MemoryScope,
        path: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<MemoryReadData> {
        self.record(format!("read {scope:?} {path} {offset} {limit:?}"))?;
        self.read
            .clone()
            .ok_or_else(|| Error::tool("no read reply configured"))
    }

    async fn write(
        &self,
        scope: MemoryScope,
        path: &str,
        content: &str,
        append: bool,
    ) -> Result<MemoryWriteData> {
        self.record(format!("write {scope:?} {path} {content:?} {append}"))?;
        self.write
            .clone()
            .ok_or_else(|| Error::tool("no write reply configured"))
    }

    async fn edit(
        &self,
        scope: MemoryScope,
        path: &str,
        old_text: &str,
        new_text: &str,
    ) -> Result<MemoryEditData> {
        self.record(format!("edit {scope:?} {path} {old_text:?} {new_text:?}"))?;
        self.edit
            .clone()
            .ok_or_else(|| Error::tool("no edit reply configured"))
    }
}

/// Builds a tool of `kind` over `backend` in English.
fn tool(kind: MemoryToolKind, backend: Option<Arc<dyn MemoryToolBackend>>) -> MemoryTool {
    MemoryTool::new(kind, backend, "en")
}

/// Invokes `tool` with a fresh, uncancelled context and unwraps the payload.
async fn invoke(tool: &MemoryTool, inputs: Value) -> Value {
    Tool::invoke(tool, inputs, &ToolContext::new())
        .await
        .unwrap()
}

/// Parses `text` as JSON, so expected payloads stay on one line.
fn json_text(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

/// Arguments that satisfy every required argument of `kind`.
fn valid_args(kind: MemoryToolKind) -> Value {
    match kind {
        MemoryToolKind::Search => json!({"query": "q"}),
        MemoryToolKind::Get | MemoryToolKind::Read | MemoryToolKind::CodingRead => {
            json!({"path": "a.md"})
        }
        MemoryToolKind::Write | MemoryToolKind::CodingWrite => {
            json!({"path": "a.md", "content": "body"})
        }
        MemoryToolKind::Edit | MemoryToolKind::CodingEdit => {
            json!({"path": "a.md", "old_text": "a", "new_text": "b"})
        }
    }
}

/// A read result of `path` carrying `text` as lines `start..=end` of `total`.
fn read_data(path: &str, text: &str, total: usize, start: usize, end: usize) -> MemoryReadData {
    MemoryReadData {
        path: path.to_owned(),
        text: text.to_owned(),
        total_lines: total,
        start_line: start,
        end_line: end,
    }
}

/// A write result of `path`.
fn write_data(path: &str, appended: bool, existed: bool) -> MemoryWriteData {
    MemoryWriteData {
        path: path.to_owned(),
        full_path: format!("/mem/{path}"),
        appended,
        file_existed: existed,
    }
}

/// An edit result of `path`.
fn edit_data(path: &str, replacements: usize) -> MemoryEditData {
    MemoryEditData {
        path: path.to_owned(),
        replacements,
    }
}

#[test]
fn kinds_scope_and_lookup_follow_the_registry() {
    let names: Vec<&str> = MemoryToolKind::ALL.iter().map(|kind| kind.name()).collect();
    assert_eq!(
        names,
        [
            "memory_search",
            "memory_get",
            "write_memory",
            "read_memory",
            "edit_memory",
            "coding_memory_read",
            "coding_memory_write",
            "coding_memory_edit",
        ]
    );
    for kind in MemoryToolKind::ALL {
        assert_eq!(MemoryToolKind::from_name(kind.name()), Some(kind));
        let expected = match kind {
            MemoryToolKind::CodingRead
            | MemoryToolKind::CodingWrite
            | MemoryToolKind::CodingEdit => MemoryScope::CodingMemory,
            _ => MemoryScope::Memory,
        };
        assert_eq!(kind.scope(), Some(expected), "{}", kind.name());
    }
    assert_eq!(MemoryToolKind::from_name(""), None);
    assert_eq!(MemoryToolKind::from_name("Memory_Search"), None);
    assert_eq!(MemoryToolKind::from_name("read_file"), None);
}

#[test]
fn cards_carry_the_python_arguments_and_defaults() {
    let required = [
        (MemoryToolKind::Search, &["query"][..]),
        (MemoryToolKind::Get, &["path"][..]),
        (MemoryToolKind::Write, &["path", "content"][..]),
        (MemoryToolKind::Read, &["path"][..]),
        (MemoryToolKind::Edit, &["path", "old_text", "new_text"][..]),
        (MemoryToolKind::CodingRead, &["path"][..]),
        (MemoryToolKind::CodingWrite, &["path", "content"][..]),
        (
            MemoryToolKind::CodingEdit,
            &["path", "old_text", "new_text"][..],
        ),
    ];
    let properties = [
        (
            MemoryToolKind::Search,
            &["max_results", "min_score", "query", "session_key"][..],
        ),
        (MemoryToolKind::Get, &["from_line", "lines", "path"][..]),
        (MemoryToolKind::Write, &["append", "content", "path"][..]),
        (MemoryToolKind::Read, &["limit", "offset", "path"][..]),
        (MemoryToolKind::Edit, &["new_text", "old_text", "path"][..]),
        (MemoryToolKind::CodingRead, &["limit", "offset", "path"][..]),
        (
            MemoryToolKind::CodingWrite,
            &["append", "content", "path"][..],
        ),
        (
            MemoryToolKind::CodingEdit,
            &["new_text", "old_text", "path"][..],
        ),
    ];

    for kind in MemoryToolKind::ALL {
        let card = MemoryTool::card_for(kind, "en");
        assert_eq!(card.name, kind.name());
        assert!(card.parallel_safe);
        assert_eq!(
            card.description,
            descriptions::description(kind.name(), "en")
        );
        assert_eq!(card.input_params["type"], json!("object"));

        let index = MemoryToolKind::ALL
            .iter()
            .position(|candidate| *candidate == kind)
            .unwrap();
        assert_eq!(card.input_params["required"], json!(required[index].1));
        let mut keys: Vec<&str> = card.input_params["properties"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, properties[index].1, "{}", kind.name());

        let rendered = card.input_params.to_string();
        assert!(!rendered.contains("$ref"), "dangling reference: {rendered}");
        assert!(
            !rendered.contains("\"default\":null"),
            "null default: {rendered}"
        );
        // Unknown languages fall back to the Python default (`cn`).
        assert_eq!(
            MemoryTool::card_for(kind, "fr").description,
            descriptions::description(kind.name(), "cn")
        );
    }

    let search = MemoryTool::card_for(MemoryToolKind::Search, "en");
    let search = &search.input_params["properties"];
    assert_eq!(search["query"]["type"], json!("string"));
    assert_eq!(search["max_results"]["type"], json!("integer"));
    assert_eq!(search["max_results"]["default"], json!(10));
    assert_eq!(search["min_score"]["type"], json!("number"));
    assert_eq!(search["min_score"]["default"], json!(0.3));
    assert_eq!(search["session_key"]["type"], json!("string"));
    assert_eq!(search["session_key"].get("default"), None);

    for kind in [MemoryToolKind::Write, MemoryToolKind::CodingWrite] {
        let card = MemoryTool::card_for(kind, "en");
        let properties = &card.input_params["properties"];
        assert_eq!(properties["append"]["type"], json!("boolean"));
        assert_eq!(properties["append"]["default"], json!(false));
    }
    for kind in [MemoryToolKind::Read, MemoryToolKind::CodingRead] {
        let card = MemoryTool::card_for(kind, "en");
        let properties = &card.input_params["properties"];
        assert_eq!(properties["offset"]["type"], json!("integer"));
        assert_eq!(properties["limit"].get("default"), None);
    }
}

#[tokio::test]
async fn a_missing_backend_and_missing_arguments_report_python_messages() {
    for kind in MemoryToolKind::ALL {
        let out = invoke(&tool(kind, None), valid_args(kind)).await;
        assert_eq!(out["success"], json!(false), "{}", kind.name());
        assert_eq!(out["error"], json!(NO_BACKEND_MESSAGE), "{}", kind.name());
        assert_eq!(out["data"], Value::Null);
        for key in [
            "success",
            "data",
            "error",
            "extracted_content",
            "include_extracted_content_only_once",
            "long_term_memory",
        ] {
            assert!(out.get(key).is_some(), "missing payload key {key}");
        }
    }

    let cases: [(MemoryToolKind, Value, &str); 11] = [
        (MemoryToolKind::Search, json!({}), "query is required"),
        (
            MemoryToolKind::Search,
            json!({"query": ""}),
            "query is required",
        ),
        (MemoryToolKind::Get, json!({}), "path is required"),
        (
            MemoryToolKind::Get,
            json!({"path": null}),
            "path is required",
        ),
        (MemoryToolKind::Read, json!({}), "path is required"),
        (MemoryToolKind::CodingRead, json!({}), "path is required"),
        (
            MemoryToolKind::Write,
            json!({"content": "body"}),
            "path is required",
        ),
        (
            MemoryToolKind::Write,
            json!({"path": "a.md"}),
            "content is required",
        ),
        (
            MemoryToolKind::CodingWrite,
            json!({"path": "a.md"}),
            "content is required",
        ),
        (
            MemoryToolKind::Edit,
            json!({"path": "a.md"}),
            "old_text is required",
        ),
        (
            MemoryToolKind::CodingEdit,
            json!({"path": "a.md", "old_text": "a"}),
            "new_text is required",
        ),
    ];
    for (kind, inputs, message) in cases {
        let out = invoke(&tool(kind, None), inputs).await;
        assert_eq!(out["success"], json!(false), "{}", kind.name());
        assert_eq!(out["error"], json!(message), "{}", kind.name());
    }
}

#[tokio::test]
async fn every_kind_forwards_the_python_arguments() {
    let cases: [(MemoryToolKind, &str, Reply, &str); 9] = [
        (
            MemoryToolKind::Search,
            r#"{"query":"q"}"#,
            Reply::Search(json!({"results": []})),
            "search q 10 0.3 None Memory",
        ),
        (
            MemoryToolKind::Search,
            r#"{"query":"rust","max_results":3,"min_score":0.5,"session_key":"s1"}"#,
            Reply::Search(json!({"results": []})),
            r#"search rust 3 0.5 Some("s1") Memory"#,
        ),
        (
            MemoryToolKind::Get,
            r#"{"path":"a.md","from_line":2,"lines":3}"#,
            Reply::Read(read_data("a.md", "", 0, 0, 0)),
            "read Memory a.md 2 Some(3)",
        ),
        (
            MemoryToolKind::Read,
            r#"{"path":"a.md","offset":1}"#,
            Reply::Read(read_data("a.md", "", 0, 0, 0)),
            "read Memory a.md 1 None",
        ),
        (
            MemoryToolKind::CodingRead,
            r#"{"path":"s.md"}"#,
            Reply::Read(read_data("s.md", "", 0, 0, 0)),
            "read CodingMemory s.md 0 None",
        ),
        (
            MemoryToolKind::Write,
            r#"{"path":"a.md","content":"body"}"#,
            Reply::Write(write_data("a.md", false, false)),
            r#"write Memory a.md "body" false"#,
        ),
        (
            MemoryToolKind::CodingWrite,
            r#"{"path":"s.md","content":"","append":true}"#,
            Reply::Write(write_data("s.md", true, true)),
            r#"write CodingMemory s.md "" true"#,
        ),
        (
            MemoryToolKind::Edit,
            r#"{"path":"a.md","old_text":"a","new_text":"b"}"#,
            Reply::Edit(edit_data("a.md", 1)),
            r#"edit Memory a.md "a" "b""#,
        ),
        (
            MemoryToolKind::CodingEdit,
            r#"{"path":"s.md","old_text":"x","new_text":"y"}"#,
            Reply::Edit(edit_data("s.md", 1)),
            r#"edit CodingMemory s.md "x" "y""#,
        ),
    ];
    for (kind, inputs, reply, expected) in cases {
        let backend = reply.backend();
        invoke(&tool(kind, Some(backend.clone())), json_text(inputs)).await;
        assert_eq!(backend.calls(), [expected], "{}", kind.name());
    }
}

#[tokio::test]
async fn every_kind_maps_the_python_payload_and_render() {
    let search_results = r#"{"results":[{"citation":"notes.md:3","score":0.83,"snippet":"prefers rust"},{"citation":"people.md:1","score":0.5,"snippet":"alice"}],"provider":"lite"}"#;
    let cases: [(MemoryToolKind, &str, Reply, &str, &str); 9] = [
        (
            MemoryToolKind::Search,
            r#"{"query":"rust","max_results":2}"#,
            Reply::Search(json_text(search_results)),
            search_results,
            "notes.md:3 (score 0.83)\nprefers rust\n\npeople.md:1 (score 0.50)\nalice",
        ),
        (
            MemoryToolKind::Get,
            r#"{"path":"notes.md","from_line":2,"lines":3}"#,
            Reply::Read(read_data("memory/notes.md", "# Notes", 9, 2, 4)),
            r##"{"path":"memory/notes.md","text":"# Notes","totalLines":9,"fromLine":2,"toLine":4,"disabled":false}"##,
            "# Notes",
        ),
        (
            MemoryToolKind::Read,
            r#"{"path":"notes.md","limit":2}"#,
            Reply::Read(read_data("notes.md", "one\ntwo", 5, 1, 2)),
            r#"{"success":true,"path":"notes.md","content":"one\ntwo","totalLines":5,"start_line":1,"end_line":2,"truncated":true}"#,
            "one\ntwo",
        ),
        (
            MemoryToolKind::CodingRead,
            r#"{"path":"style.md"}"#,
            Reply::Read(read_data("style.md", "---", 3, 1, 3)),
            r#"{"success":true,"path":"style.md","content":"---","totalLines":3,"start_line":1,"end_line":3,"truncated":false}"#,
            "---",
        ),
        (
            MemoryToolKind::Write,
            r#"{"path":"notes.md","content":"body"}"#,
            Reply::Write(write_data("notes.md", false, true)),
            r#"{"success":true,"path":"notes.md","fullPath":"/mem/notes.md","appended":false,"fileExisted":true}"#,
            "Wrote memory file notes.md.",
        ),
        (
            MemoryToolKind::CodingWrite,
            r#"{"path":"style.md","content":"","append":true}"#,
            Reply::Write(write_data("style.md", true, true)),
            r#"{"success":true,"path":"style.md","fullPath":"/mem/style.md","appended":true,"fileExisted":true}"#,
            "Appended to memory file style.md.",
        ),
        (
            MemoryToolKind::Edit,
            r#"{"path":"notes.md","old_text":"a","new_text":"b"}"#,
            Reply::Edit(edit_data("notes.md", 2)),
            r#"{"success":true,"path":"notes.md","replaced":2}"#,
            "Edited memory file notes.md.",
        ),
        (
            MemoryToolKind::CodingEdit,
            r#"{"path":"style.md","old_text":"x","new_text":"y"}"#,
            Reply::Edit(edit_data("style.md", 1)),
            r#"{"success":true,"path":"style.md","replaced":1}"#,
            "Edited memory file style.md.",
        ),
        (
            MemoryToolKind::Get,
            r#"{"path":"empty.md"}"#,
            Reply::Read(read_data("empty.md", "", 0, 0, 0)),
            r#"{"path":"empty.md","text":"","totalLines":0,"fromLine":0,"toLine":0,"disabled":false}"#,
            "No memory content found.",
        ),
    ];
    for (kind, inputs, reply, data, render) in cases {
        let memory = tool(kind, Some(reply.backend()));
        let out = invoke(&memory, json_text(inputs)).await;
        assert_eq!(out["success"], json!(true), "{}", kind.name());
        assert_eq!(out["data"], json_text(data), "{}", kind.name());
        assert_eq!(out["error"], Value::Null, "{}", kind.name());
        assert_eq!(memory.render_for_llm(&out), render, "{}", kind.name());
    }
}

#[tokio::test]
async fn an_empty_search_renders_the_placeholder_and_disabled_fails() {
    let memory = tool(
        MemoryToolKind::Search,
        Some(Reply::Search(json!({})).backend()),
    );
    let out = invoke(&memory, json!({"query": "q"})).await;
    assert_eq!(out["success"], json!(true));
    assert_eq!(memory.render_for_llm(&out), "No matching memories found.");

    let data = json!({"disabled": true, "error": "memory is disabled", "results": []});
    let memory = tool(
        MemoryToolKind::Search,
        Some(Reply::Search(data.clone()).backend()),
    );
    let out = invoke(&memory, json!({"query": "q"})).await;
    assert_eq!(out["success"], json!(false));
    assert_eq!(out["error"], json!("memory is disabled"));
    assert_eq!(out["data"], data);
    assert_eq!(memory.render_for_llm(&out), "memory is disabled");
}

#[tokio::test]
async fn backend_errors_become_failure_payloads() {
    let backend = Arc::new(FakeBackend {
        failure: Some("disk is full"),
        ..FakeBackend::default()
    });
    for kind in MemoryToolKind::ALL {
        let out = invoke(&tool(kind, Some(backend.clone())), valid_args(kind)).await;
        assert_eq!(out["success"], json!(false), "{}", kind.name());
        assert_eq!(out["error"], json!("disk is full"), "{}", kind.name());
        assert_eq!(out["data"], Value::Null, "{}", kind.name());
    }
}

#[test]
fn render_falls_back_to_payload_text_for_non_envelopes() {
    let write = tool(MemoryToolKind::Write, None);
    assert_eq!(write.render_for_llm(&json!("plain")), "plain");
    assert_eq!(write.render_for_llm(&json!({"content": "raw"})), "raw");
    assert_eq!(
        write.render_for_llm(&json!({"success": false, "error": "boom"})),
        "boom"
    );

    let read = tool(MemoryToolKind::Read, None);
    assert_eq!(
        read.render_for_llm(&json!({"success": true, "data": {"content": "done"}})),
        "done"
    );
}

#[tokio::test]
async fn a_cancelled_context_returns_the_cancelled_error() {
    let memory = tool(MemoryToolKind::Search, None);
    let ctx = ToolContext::new();
    ctx.cancellation.cancel();

    let error = Tool::invoke(&memory, json!({"query": "q"}), &ctx)
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Cancelled));
}
