//! One-shot execution behind `openjiuwen run`.
//!
//! [`run_once`] is what the binary calls: it resolves the prompt (argument or
//! piped stdin), builds the model from [`CliOptions`], and hands both to
//! [`run_with_model`]. The latter is the test seam — it accepts a prebuilt
//! [`Model`] so tests drive a scripted model and never touch the network.
//!
//! A run streams exactly one agent invocation and renders it per
//! [`OutputFormat`]:
//!
//! - `text` — chunks are rendered to the terminal while they arrive.
//! - `json` — nothing is printed while streaming; one pretty-printed
//!   [`RunOutcome::to_json`] object is printed at the end.
//! - `stream-json` — one `{"type", "index", "payload"}` line per chunk plus a
//!   terminal `{"type": "result", …}` line so consumers learn the session id.
//!
//! Session records are append-only and best-effort: a failed write is logged and
//! never loses the answer.

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentChunk};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_session::{AgentSession, Session};
use serde_json::{json, Value};

use super::config::{CliOptions, DEFAULT_LANGUAGE, SESSION_PREFIX};
use super::io::CliIo;
use super::permissions::CliPermissionHost;
use super::render::RenderState;
use super::RunArgs;
use crate::agent::{create_deep_agent, DeepAgent};
use crate::permission::PermissionHost;
use crate::schema::DeepAgentConfig;
use crate::workspace::Workspace;

/// Message shown when a run has no prompt at all.
const EMPTY_PROMPT: &str = "A prompt argument is required, or pipe via stdin.";

/// Session role of a user message in a persisted record.
const ROLE_USER: &str = "user";

/// Session role of an assistant message in a persisted record.
const ROLE_ASSISTANT: &str = "assistant";

/// Result of one completed run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunOutcome {
    /// Final answer of the invocation.
    pub answer: String,
    /// Result type reported by the terminal `answer` frame (`answer`, `error`, …).
    pub result_type: String,
    /// Token usage captured from the last `llm_usage` frame, when reported.
    pub usage: Option<Value>,
    /// Session id the run used.
    pub session_id: String,
    /// Number of streamed chunks consumed.
    pub chunks: usize,
}

impl RunOutcome {
    /// Renders the outcome as the `json` output format's object.
    ///
    /// The keys are `answer`, `result_type`, `usage` and `session_id`; usage is
    /// `null` when the model reported none.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "answer": self.answer,
            "result_type": self.result_type,
            "usage": self.usage,
            "session_id": self.session_id,
        })
    }
}

/// Renders one streamed chunk as the `stream-json` output format's line.
#[must_use]
pub fn stream_json_line(chunk: &AgentChunk) -> Value {
    json!({
        "type": chunk.kind,
        "index": chunk.index,
        "payload": chunk.payload,
    })
}

/// Renders the terminal line of the `stream-json` output format.
///
/// It is [`RunOutcome::to_json`] with `"type": "result"` added, so a consumer
/// learns the session id without a separate channel.
#[must_use]
pub fn result_json_line(outcome: &RunOutcome) -> Value {
    let mut value = outcome.to_json();
    if let Value::Object(object) = &mut value {
        object.insert("type".to_owned(), Value::String("result".to_owned()));
    }
    value
}

/// Creates a new CLI session id: [`SESSION_PREFIX`] plus eight hex characters.
#[must_use]
pub fn session_id() -> String {
    format!(
        "{}{}",
        SESSION_PREFIX,
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    )
}

/// Builds a kernel session for `session_id`.
#[must_use]
pub fn new_session(session_id: &str) -> Arc<dyn Session> {
    Arc::new(AgentSession::new(session_id))
}

/// Builds the `DeepAgent` a CLI run or REPL turn executes on.
///
/// The configuration comes from [`CliOptions`]: the iteration budget, the model
/// identity (name, provider, key, base URL) and the CLI language. The working
/// directory is the process directory so the agent reads and writes the project
/// the user invoked it from, falling back to the workspace when the process
/// directory cannot be determined.
///
/// `io` is handed to [`CliPermissionHost`], which is what makes permission asks
/// answerable in the REPL and fail closed in a one-shot run.
///
/// # Errors
///
/// Returns whatever agent construction reports: [`Error::Config`] for an invalid
/// configuration, [`Error::Validation`] for an unusable workspace.
pub async fn build_agent(
    model: Arc<dyn Model>,
    options: &CliOptions,
    io: &CliIo,
) -> Result<Arc<DeepAgent>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| options.workspace.clone());
    let config = DeepAgentConfig {
        enable_task_planning: true,
        max_iterations: Some(options.max_iterations),
        language: DEFAULT_LANGUAGE.to_owned(),
        cwd: Some(cwd.clone()),
        project_root: Some(cwd),
        model_name: Some(options.model.clone()),
        model_provider: Some(options.provider.clone()),
        api_key: options.api_key.clone(),
        api_base: options.api_base.clone(),
        ..DeepAgentConfig::default()
    };
    let permissions = options.permissions();
    let host: Arc<dyn PermissionHost> = Arc::new(CliPermissionHost::new(
        io.clone(),
        session_id(),
        options.workspace.clone(),
        options.permissions(),
    ));
    create_deep_agent(
        model,
        None,
        Some(config),
        Vec::new(),
        Vec::new(),
        Some(Workspace::for_language(
            &options.workspace,
            DEFAULT_LANGUAGE,
        )),
        Vec::new(),
        Vec::new(),
        true,
        Some(options.max_iterations),
        Some(DEFAULT_LANGUAGE.to_owned()),
        None,
        false,
        Some(permissions),
        Some(host),
    )
    .await
}

/// Runs one prompt and returns what happened.
///
/// The prompt is resolved (and validated) before the model is built, so an empty
/// prompt fails fast without touching credentials or the network.
///
/// # Errors
///
/// Returns [`Error::Validation`] for an empty prompt, and whatever model or
/// agent construction reports.
pub async fn run_once(options: &CliOptions, args: &RunArgs, io: &CliIo) -> Result<RunOutcome> {
    let prompt = resolve_prompt(args, io).await?;
    let model = options.build_model().await?;
    // Hand the resolved prompt down so stdin is read once, not twice.
    let resolved = RunArgs {
        prompt: Some(prompt),
        output_format: args.output_format,
    };
    run_with_model(options, &resolved, io, model).await
}

/// Runs one prompt on a prebuilt model (the test seam).
///
/// See the module docs for the per-format output shapes.
///
/// # Errors
///
/// Returns [`Error::Validation`] for an empty prompt, and whatever agent
/// construction or the stream reports. Session-write failures are logged, never
/// returned.
pub async fn run_with_model(
    options: &CliOptions,
    args: &RunArgs,
    io: &CliIo,
    model: Arc<dyn Model>,
) -> Result<RunOutcome> {
    let prompt = resolve_prompt(args, io).await?;
    let agent = build_agent(model, options, io).await?;
    let run_session_id = session_id();
    let session = new_session(&run_session_id);
    let store = options.sessions();
    if let Err(error) = store.new_session(&run_session_id, &options.model) {
        tracing::warn!(%error, "failed to create CLI session record");
    }

    let mut render = RenderState::new();
    let mut chunks = 0_usize;
    let mut stream = agent.stream(json!({"query": prompt}), Some(session));
    while let Some(item) = stream.next().await {
        let chunk = item?;
        chunks += 1;
        match args.output_format {
            super::OutputFormat::Text => render.render(&chunk, Some(io), options.verbose).await?,
            super::OutputFormat::Json => render.render(&chunk, None, false).await?,
            super::OutputFormat::StreamJson => {
                let line =
                    serde_json::to_string(&stream_json_line(&chunk)).map_err(Error::external)?;
                io.write_line(&line).await?;
                render.render(&chunk, None, false).await?;
            }
        }
    }

    let outcome = RunOutcome {
        answer: render.answer().to_owned(),
        result_type: render.result_type().to_owned(),
        usage: render.usage.clone(),
        session_id: run_session_id.clone(),
        chunks,
    };

    match args.output_format {
        super::OutputFormat::Text => render.finish(Some(io)).await?,
        super::OutputFormat::Json => {
            let text = serde_json::to_string_pretty(&outcome.to_json()).map_err(Error::external)?;
            io.write_line(&text).await?;
        }
        super::OutputFormat::StreamJson => {
            let line =
                serde_json::to_string(&result_json_line(&outcome)).map_err(Error::external)?;
            io.write_line(&line).await?;
        }
    }

    if let Err(error) = store.append_message(&run_session_id, ROLE_USER, &prompt) {
        tracing::warn!(%error, "failed to persist CLI user message");
    }
    if let Err(error) = store.append_message(&run_session_id, ROLE_ASSISTANT, &outcome.answer) {
        tracing::warn!(%error, "failed to persist CLI assistant message");
    }

    Ok(outcome)
}

/// Resolves the prompt of a run: the argument when given, stdin otherwise.
///
/// A `None` prompt or the literal `-` reads stdin to end of input and joins the
/// lines with `\n`; the result is trimmed in both cases.
async fn resolve_prompt(args: &RunArgs, io: &CliIo) -> Result<String> {
    let prompt = match &args.prompt {
        Some(prompt) if prompt != "-" => prompt.trim().to_owned(),
        _ => {
            let mut lines = Vec::new();
            while let Some(line) = io.prompt_line("").await? {
                lines.push(line);
            }
            lines.join("\n").trim().to_owned()
        }
    };
    if prompt.is_empty() {
        return Err(Error::validation(EMPTY_PROMPT));
    }
    Ok(prompt)
}
