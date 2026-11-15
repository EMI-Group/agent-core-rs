//! The bounded per-invocation journal behind [`DeepAgentConfig::invocation_log_dir`].
//!
//! Python parity: `DeepAgent._invocation_log_context` opens a scoped
//! `invocation_log_scope(directory, agent_id, session_id, role)` around `invoke`, so
//! every turn of that invocation lands in one JSONL file under the configured
//! directory without touching process-global logger state.
//!
//! The Rust counterpart records one JSON object per turn of the same invocation:
//! agent id, session id, role, query, answer summary, tool-call count, iteration
//! count, status and the start/finish timestamps. Three rules keep it safe:
//!
//! - **bounded** — every text field is clipped to [`MAX_TEXT_CHARS`], one line never
//!   exceeds [`MAX_ENTRY_CHARS`] and the file keeps itself under [`MAX_FILE_BYTES`]
//!   by dropping its oldest half once the bound is reached;
//! - **warn-only** — an IO failure is logged through `tracing` and never fails the
//!   turn, exactly like the persistent-agent journal;
//! - **stateless** — the journal is created per invocation from the configuration, so
//!   there is no process-global registry and `None` simply means "no logging".
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_agent::{AgentOutput, ResultType};
use serde_json::json;
use tokio::io::AsyncWriteExt;

use crate::schema::DeepAgentConfig;

/// Longest query/answer text kept in one entry, in characters.
pub(crate) const MAX_TEXT_CHARS: usize = 1024;

/// Longest single JSONL line, in characters.
pub(crate) const MAX_ENTRY_CHARS: usize = 4096;

/// Size at which the journal drops its oldest half, in bytes.
pub(crate) const MAX_FILE_BYTES: u64 = 128 * 1024;

/// Identity of one invocation, fixed when the journal is opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InvocationMeta {
    /// Id of the agent that ran the turn.
    pub(crate) agent_id: String,
    /// Session the turn ran on.
    pub(crate) session_id: String,
    /// Role label of the agent (its card name), like Python's `role`.
    pub(crate) role: String,
    /// Query the turn answered.
    pub(crate) query: String,
}

/// What one finished turn contributed to the journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TurnRecord<'a> {
    /// Answer text of the turn.
    pub(crate) answer: &'a str,
    /// Terminal result kind (`answer`, `error` or `interrupt`).
    pub(crate) status: &'a str,
    /// Number of tool calls the turn made.
    pub(crate) tool_calls: usize,
    /// Number of model iterations the turn used, when known.
    pub(crate) iterations: Option<usize>,
}

/// The per-invocation journal of one `DeepAgent` turn set.
#[derive(Debug, Clone)]
pub(crate) struct InvocationJournal {
    /// File every turn of this invocation appends to.
    path: PathBuf,
    /// Identity recorded with every line.
    meta: InvocationMeta,
    /// Start of the invocation in epoch milliseconds.
    started_ms: u128,
}

impl InvocationJournal {
    /// Opens the journal of one invocation on `session_id`.
    ///
    /// Returns `None` when [`DeepAgentConfig::invocation_log_dir`] is unset, and also
    /// when the directory cannot be created: recording is an observation of the
    /// turn, never a precondition for it, so the caller proceeds unjournaled.
    pub(crate) async fn open(
        config: &DeepAgentConfig,
        agent_id: &str,
        role: &str,
        session_id: &str,
        query: &str,
    ) -> Option<Self> {
        let directory = config.invocation_log_dir.clone()?;
        let path = directory.join(format!(
            "{}-{}.jsonl",
            file_component(agent_id),
            file_component(session_id)
        ));
        if let Err(error) = tokio::fs::create_dir_all(&directory).await {
            tracing::warn!(
                directory = %directory.display(),
                error = %error,
                "invocation journal unavailable"
            );
            return None;
        }
        Some(Self {
            path,
            meta: InvocationMeta {
                agent_id: agent_id.to_string(),
                session_id: session_id.to_string(),
                role: role.to_string(),
                query: query.to_string(),
            },
            started_ms: now_ms(),
        })
    }

    /// The file this invocation appends to.
    #[must_use]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one line for a finished turn; a write failure is warn-only.
    pub(crate) async fn finish(&self, output: &AgentOutput) {
        let record = TurnRecord {
            answer: &output.output,
            status: status_of(output),
            tool_calls: output.observations.len(),
            iterations: output.iterations_used,
        };
        self.append(&record).await;
    }

    /// Appends one line for a turn that failed before producing an output.
    ///
    /// The error text stands in for the answer, so a failed turn is still visible in
    /// the journal instead of silently missing from it.
    pub(crate) async fn finish_error(&self, error: &str) {
        let record = TurnRecord {
            answer: error,
            status: "error",
            tool_calls: 0,
            iterations: None,
        };
        self.append(&record).await;
    }

    /// Renders and appends one record; a write failure is warn-only.
    async fn append(&self, record: &TurnRecord<'_>) {
        let line = render_entry(&self.meta, record, self.started_ms, now_ms());
        if let Err(error) = append_bounded(&self.path, &line, MAX_FILE_BYTES).await {
            tracing::warn!(
                path = %self.path.display(),
                error = %error,
                "failed to write the invocation journal entry"
            );
        }
    }
}

/// Renders one JSONL entry for `record`.
///
/// Text fields are clipped to [`MAX_TEXT_CHARS`]; if the rendered line still exceeds
/// [`MAX_ENTRY_CHARS`] (pathological ids), the query and answer are dropped so the
/// line stays readable JSON. The result never contains a newline.
#[must_use]
pub(crate) fn render_entry(
    meta: &InvocationMeta,
    record: &TurnRecord<'_>,
    started_ms: u128,
    finished_ms: u128,
) -> String {
    let mut entry = json!({
        "agent_id": meta.agent_id,
        "session_id": meta.session_id,
        "role": meta.role,
        "query": clip(&meta.query, MAX_TEXT_CHARS),
        "answer": clip(record.answer, MAX_TEXT_CHARS),
        "status": record.status,
        "tool_calls": record.tool_calls,
        "iterations": record.iterations,
        "started_at_ms": u64::try_from(started_ms).unwrap_or(u64::MAX),
        "finished_at_ms": u64::try_from(finished_ms).unwrap_or(u64::MAX),
        "duration_ms": u64::try_from(finished_ms.saturating_sub(started_ms)).unwrap_or(u64::MAX),
    });
    if serialize_len(&entry) > MAX_ENTRY_CHARS {
        if let Some(map) = entry.as_object_mut() {
            map.insert("query".to_string(), json!(clip(&meta.query, 32)));
            map.insert("answer".to_string(), json!(clip(record.answer, 32)));
        }
    }
    if serialize_len(&entry) > MAX_ENTRY_CHARS {
        if let Some(map) = entry.as_object_mut() {
            map.remove("query");
            map.remove("answer");
            map.insert("truncated".to_string(), json!(true));
        }
    }
    entry.to_string().replace(['\n', '\r'], " ")
}

/// Appends `line` to `path`, trimming the oldest entries when `max_bytes` is reached.
///
/// The file is created when missing. Callers bound `line` to [`MAX_ENTRY_CHARS`], so
/// the file stays at `max_bytes + MAX_ENTRY_CHARS` at most.
async fn append_bounded(path: &Path, line: &str, max_bytes: u64) -> std::io::Result<()> {
    if let Ok(metadata) = tokio::fs::metadata(path).await {
        if metadata.len() >= max_bytes {
            trim_oldest(path).await?;
        }
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(line.as_bytes()).await?;
    file.write_all(b"\n").await?;
    file.flush().await
}

/// Rewrites `path` with the newest whole lines that fit in half of [`MAX_FILE_BYTES`].
async fn trim_oldest(path: &Path) -> std::io::Result<()> {
    let content = tokio::fs::read_to_string(path).await?;
    let budget = usize::try_from(MAX_FILE_BYTES / 2).unwrap_or(usize::MAX);
    let mut kept: Vec<&str> = Vec::new();
    let mut size = 0;
    for line in content.lines().rev() {
        let length = line.len() + 1;
        if size + length > budget {
            break;
        }
        size += length;
        kept.push(line);
    }
    kept.reverse();
    let mut rewritten = kept.join("\n");
    if !rewritten.is_empty() {
        rewritten.push('\n');
    }
    tokio::fs::write(path, rewritten).await
}

/// Length of `entry` once serialized, in characters.
fn serialize_len(entry: &serde_json::Value) -> usize {
    entry.to_string().chars().count()
}

/// Clips `text` to `limit` characters, marking a truncated tail.
#[must_use]
pub(crate) fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(limit).collect();
    clipped.push('…');
    clipped
}

/// Maps a finished output onto the journal's `status` value.
fn status_of(output: &AgentOutput) -> &'static str {
    match output.result_type {
        ResultType::Answer => "answer",
        ResultType::Error => "error",
        ResultType::Interrupt => "interrupt",
    }
}

/// Makes `value` safe as one file-name component.
fn file_component(value: &str) -> String {
    let mapped: String = value
        .chars()
        .map(|character| match character {
            '/' | '\\' | ':' | '\0' | ' ' => '_',
            other => other,
        })
        .take(64)
        .collect();
    if mapped.is_empty() {
        "unknown".to_string()
    } else {
        mapped
    }
}

/// Milliseconds since the Unix epoch; `0` when the clock is before it.
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_agent::AgentOutput;

    use super::{
        append_bounded, clip, file_component, render_entry, InvocationJournal, InvocationMeta,
        TurnRecord, MAX_ENTRY_CHARS, MAX_TEXT_CHARS,
    };
    use crate::schema::DeepAgentConfig;

    /// A fixed invocation identity.
    fn meta() -> InvocationMeta {
        InvocationMeta {
            agent_id: "agent-1".to_string(),
            session_id: "session-1".to_string(),
            role: "deep_agent".to_string(),
            query: "count the files".to_string(),
        }
    }

    /// A turn record with the given answer.
    fn record(answer: &str) -> TurnRecord<'_> {
        TurnRecord {
            answer,
            status: "answer",
            tool_calls: 2,
            iterations: Some(3),
        }
    }

    #[test]
    fn an_entry_records_the_turn_fields() {
        let line = render_entry(&meta(), &record("four files"), 1_000, 1_250);
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["agent_id"], "agent-1");
        assert_eq!(value["session_id"], "session-1");
        assert_eq!(value["role"], "deep_agent");
        assert_eq!(value["query"], "count the files");
        assert_eq!(value["answer"], "four files");
        assert_eq!(value["status"], "answer");
        assert_eq!(value["tool_calls"], 2);
        assert_eq!(value["iterations"], 3);
        assert_eq!(value["started_at_ms"], 1_000);
        assert_eq!(value["duration_ms"], 250);
        assert!(!line.contains('\n'));
    }

    #[test]
    fn long_text_is_clipped_and_the_line_stays_parseable() {
        let long = "x".repeat(MAX_TEXT_CHARS * 3);
        let line = render_entry(&meta(), &record(&long), 0, 1);
        assert!(line.chars().count() <= MAX_ENTRY_CHARS, "{}", line.len());
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        let answer = value["answer"].as_str().unwrap();
        assert_eq!(answer.chars().count(), MAX_TEXT_CHARS + 1);
        assert!(answer.ends_with('…'));
    }

    #[test]
    fn the_entry_budget_survives_pathological_identifiers() {
        let mut weird = meta();
        weird.agent_id = "a".repeat(10_000);
        weird.session_id = "s".repeat(10_000);
        let line = render_entry(&weird, &record("answer"), 0, 1);
        assert!(line.chars().count() <= MAX_ENTRY_CHARS, "{}", line.len());
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["truncated"], true);
        assert_eq!(value["status"], "answer");
    }

    #[test]
    fn clip_keeps_short_text_and_marks_the_tail() {
        assert_eq!(clip("short", 10), "short");
        assert_eq!(clip("12345", 3), "123…");
    }

    #[test]
    fn file_components_are_sanitized() {
        assert_eq!(file_component("a/b:c"), "a_b_c");
        assert_eq!(file_component(""), "unknown");
        assert_eq!(file_component(&"z".repeat(100)).len(), 64);
    }

    #[tokio::test]
    async fn the_journal_appends_one_line_per_turn() {
        let temp = tempfile::tempdir().unwrap();
        let config = DeepAgentConfig::default().with_invocation_log_dir(Some(temp.path().into()));
        let journal = InvocationJournal::open(&config, "agent-1", "deep_agent", "s-1", "hi")
            .await
            .expect("the journal opens");
        journal.finish(&AgentOutput::answer("first")).await;
        journal.finish(&AgentOutput::answer("second")).await;

        let content = tokio::fs::read_to_string(journal.path()).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["answer"], "second");
        assert_eq!(second["status"], "answer");
    }

    #[tokio::test]
    async fn no_directory_means_no_journal() {
        let config = DeepAgentConfig::default();
        assert!(
            InvocationJournal::open(&config, "agent-1", "deep_agent", "s-1", "hi")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn an_unusable_directory_is_reported_and_skipped() {
        let temp = tempfile::tempdir().unwrap();
        let blocker = temp.path().join("logs");
        tokio::fs::write(&blocker, b"not a directory")
            .await
            .unwrap();
        let config = DeepAgentConfig::default().with_invocation_log_dir(Some(blocker));
        assert!(
            InvocationJournal::open(&config, "agent-1", "deep_agent", "s-1", "hi")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn the_file_drops_its_oldest_entries_at_the_bound() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("bound.jsonl");
        for index in 0..30 {
            let line = format!("{{\"n\":{index},\"pad\":\"{}\"}}", "p".repeat(40));
            append_bounded(&path, &line, 400).await.unwrap();
        }
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(
            content.len() < 400 + MAX_ENTRY_CHARS,
            "the journal grew past its bound: {}",
            content.len()
        );
        assert!(content.lines().count() < 30);
        let last: serde_json::Value =
            serde_json::from_str(content.lines().last().unwrap()).unwrap();
        assert_eq!(last["n"], 29);
    }
}
