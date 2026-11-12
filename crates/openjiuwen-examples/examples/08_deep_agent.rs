//! `08_deep_agent` — a `DeepAgent` in a temporary workspace doing a file-tool round.
//!
//! The harness tier puts a workspace, a generated system prompt, and a tool set on
//! top of the kernel `ReActAgent`; this example shows the smallest useful slice of
//! it. A `DeepAgentBuilder` receives a card, a `DeepAgentConfig`, a model, and a
//! `Workspace` rooted at a temporary directory, and mounts the default
//! `ToolRegistry` — every built-in tool, `write_file` included. The scripted model
//! asks for `write_file`, the harness resolves the relative path inside the
//! workspace, writes the file, feeds the result back to the model, and the second
//! scripted turn is the final answer.
//!
//! The outer task loop stays off (`enable_task_loop: false`): a single
//! deterministic file-tool round needs only the inner `ReAct` loop, and the task
//! loop would add its own planning rounds on top of it. The example then reads the
//! workspace back to prove the file really landed.
//!
//! Offline by default: a `ScriptedModel` drives the model calls and the workspace
//! is a `TempDir`, so nothing is written outside it.
//!
//! Python counterparts: `examples/harness/deepagent_with_worktree.py` and
//! `examples/skill_use/main.py`, reduced to one file-tool round.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use openjiuwen::harness::workspace::Workspace;
use openjiuwen::prelude::{
    Agent, AgentCard, AgentSession, DeepAgent, DeepAgentBuilder, DeepAgentConfig, Model,
    ResultType, Session,
};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::json;

/// Model name written into the config (the model itself is scripted).
const MODEL_NAME: &str = "scripted-model";

/// Session id; the deep agent persists its state under the card id and this id.
const SESSION_ID: &str = "deep-agent-1";

/// File the model asks the harness to write, relative to the workspace root.
const NOTE_FILE: &str = "notes.txt";

/// Exact content the model puts in the file.
const NOTE_BODY: &str = "DeepAgent workspace note: written by the write_file tool.\n";

/// What the model says once the tool result is back.
const FINAL_ANSWER: &str = "Wrote notes.txt in the workspace.";

/// Upper bound on the whole turn, so a misbehaving tool cannot hang the demo.
const TURN_TIMEOUT: Duration = Duration::from_secs(30);

/// Builds the deep agent: card, config, scripted model, and the temp workspace.
///
/// The builder mounts the default tool registry over the workspace, which is how
/// `write_file` becomes available to the model without an explicit `.tool(...)`.
fn build_agent(model: Arc<dyn Model>, workspace: &Workspace) -> Result<Arc<DeepAgent>> {
    let card = AgentCard::new("deep-agent").with_description("workspace note taker");

    let config = DeepAgentConfig {
        model_name: Some(MODEL_NAME.to_owned()),
        // One file-tool round: the inner ReAct loop is enough, and without the task
        // loop the run is exactly two model calls instead of planning rounds.
        enable_task_loop: false,
        language: "en".to_owned(),
        ..DeepAgentConfig::default()
    };

    DeepAgentBuilder::new(card)
        .config(config)
        .model(model)
        .workspace(workspace.clone())
        .build()
        .context("the deep agent failed to build")
}

/// The scripted script: ask for `write_file`, then answer.
fn scripted_model() -> ScriptedModel {
    ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "write_file",
            json!({"file_path": NOTE_FILE, "content": NOTE_BODY}),
        ))
        .respond_text(FINAL_ANSWER)
        .build()
}

/// Collects every file under `dir`, as paths relative to `root`, sorted.
///
/// Recursive and directory-sorted so two runs print the same listing.
fn collect_files(root: &Path, dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("cannot list {}", dir.display()))?;
    entries.sort_by_key(std::fs::DirEntry::file_name);

    let mut files = Vec::new();
    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("cannot stat {}", path.display()))?;
        if file_type.is_dir() {
            files.extend(collect_files(root, &path)?);
        } else {
            let relative = path
                .strip_prefix(root)
                .with_context(|| format!("{} is outside the workspace", path.display()))?;
            files.push(relative.to_path_buf());
        }
    }
    Ok(files)
}

/// Whether a rendered tool observation reports a successful write.
///
/// An `Observation` carries only the text the tool rendered — no success flag — so
/// the action word `write_file` prints (`Created …` / `Updated …`) is the signal.
fn observation_succeeded(text: &str) -> bool {
    text.starts_with("Created ") || text.starts_with("Updated ")
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = common::setup()?;
    common::banner::title("08 deep agent");

    if mode.is_live() {
        common::banner::note(
            "this example is offline-only: it ignores the live provider and keeps the scripted \
             model, because the point is the workspace file round, not the wire format",
        );
    }

    // The handle keeps the workspace directory alive for the whole run, and the
    // deep agent is rooted inside it, so no example path escapes the temp tree.
    let workspace_dir = common::tempdir()?;
    let workspace = Workspace::new(workspace_dir.path());

    common::banner::step(&format!("workspace: {}", workspace_dir.path().display()));

    let scripted = scripted_model();
    let model = scripted.clone();
    let agent = build_agent(Arc::new(model) as Arc<dyn Model>, &workspace)?;

    common::banner::step(&format!(
        "card: {} (id {})",
        agent.card().name,
        agent.card().id
    ));
    common::banner::note(&format!(
        "task loop: {}",
        if agent.config().enable_task_loop {
            "on"
        } else {
            "off (one ReAct round)"
        }
    ));
    common::banner::step(&format!(
        "asking the model to write {NOTE_FILE} with write_file"
    ));

    let session: Arc<dyn Session> = Arc::new(AgentSession::new(SESSION_ID));
    let output = tokio::time::timeout(
        TURN_TIMEOUT,
        agent.invoke(
            json!({"query": format!("Write a short note into {NOTE_FILE}.")}),
            Some(session),
        ),
    )
    .await
    .context("the invoke timed out")?
    .context("the invoke failed")?;

    let result_type = match output.result_type {
        ResultType::Answer => "answer",
        ResultType::Error => "error",
        ResultType::Interrupt => "interrupt",
    };
    common::banner::note(&format!("result type: {result_type}"));
    common::banner::note(&format!(
        "iterations used: {}",
        output.iterations_used.unwrap_or_default()
    ));
    common::banner::note(&format!("model calls: {}", scripted.call_count()));
    for observation in &output.observations {
        common::banner::note(&format!(
            "observation {}: {}",
            observation.name.as_deref().unwrap_or("<unnamed>"),
            if observation_succeeded(&observation.content) {
                "success"
            } else {
                "failure"
            }
        ));
        common::banner::note(&format!("  {}", observation.content.trim_end()));
    }
    common::banner::answer(&output.output);

    if output.result_type != ResultType::Answer {
        bail!("expected an answer from the deep agent, got {result_type}");
    }

    // Prove the round: read the workspace back, then check the file the tool wrote.
    common::banner::step("workspace contents after the round");
    let files = collect_files(workspace_dir.path(), workspace_dir.path())?;
    if files.is_empty() {
        bail!("the workspace is empty: the write_file round did not land");
    }
    for file in &files {
        common::banner::note(&file.display().to_string());
    }

    let note_path = workspace_dir.path().join(NOTE_FILE);
    let written = std::fs::read_to_string(&note_path).with_context(|| {
        format!(
            "{} is missing: the write_file round did not create it",
            note_path.display()
        )
    })?;
    if written != NOTE_BODY {
        bail!(
            "{NOTE_FILE} holds {:?}, expected {:?}",
            written.trim_end(),
            NOTE_BODY.trim_end()
        );
    }
    common::banner::step(&format!("content of {NOTE_FILE}"));
    common::banner::note(written.trim_end());

    common::banner::note(
        "the same DeepAgent would now keep this workspace across invocations; every file it \
         writes stays inside it",
    );

    Ok(())
}
