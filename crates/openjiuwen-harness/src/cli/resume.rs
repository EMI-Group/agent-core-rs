//! Resuming a persisted execution as a normal CLI turn.
//!
//! Resuming here means *continue an existing session id*: the turn runs on the
//! same kernel [`Session`](openjiuwen_session::Session) identity as the earlier
//! one, with the prior message window seeded into the kernel's `"main"`
//! conversation context (and, when the caller has it, the persisted `deepagent`
//! state lane restored), so the model sees the restored transcript ahead of the
//! new query. The CLI session record under `<home>/sessions/<id>.json` is
//! **continued, never truncated**.
//!
//! The CLI owns only plain data. [`ResumedTurn`] is a [`Debug`]/[`Clone`] value
//! with no store, clock or IO handle: the caller extracts the data from whatever
//! it persisted — a `persistent_agent::ResumePlan`, a session store of its own, a
//! journal replay — and maps it into these fields. Nothing here imports
//! `persistent_agent`; the module edge rule keeps `cli` out of it.
//!
//! Output is byte-identical to the non-resumed path because the turn is streamed
//! through the same tail the fresh-run path uses (`run::stream_run`): rendering,
//! per-format output and the warn-only transcript append all live there, and this
//! module only owns the seeding that happens before it.
//!
//! ## Ordering
//!
//! The seed must follow the last agent build (or `configure`) and precede the
//! turn: re-applying the agent configuration drops every pooled kernel context,
//! so an earlier seed is silently discarded. Both entry points here build the
//! agent first and seed immediately after, which is why callers never have to
//! think about it.

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::BaseMessage;
use serde_json::Value;

use super::config::CliOptions;
use super::io::CliIo;
use super::run::{self, RunOutcome};
use super::RunArgs;
use crate::agent::seed_state;

/// A persisted turn the CLI continues instead of starting a fresh session.
///
/// Built with [`ResumedTurn::new`] plus the [`ResumedTurn::with_history`] and
/// [`ResumedTurn::with_state`] setters. The value is plain data: the caller maps
/// its own persisted representation into these fields.
#[derive(Debug, Clone)]
pub struct ResumedTurn {
    /// Session id to continue (the kernel session and the CLI record share it).
    pub session_id: String,
    /// Prior message window, oldest first.
    pub history: Vec<BaseMessage>,
    /// Persisted `deepagent` state payload, when the caller has one.
    pub state: Option<Value>,
}

impl ResumedTurn {
    /// Starts a resumed turn over the session `session_id`.
    ///
    /// The history is empty and no state is restored until the setters are
    /// called, so the result describes a plain continuation of an existing
    /// session.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            history: Vec::new(),
            state: None,
        }
    }

    /// Sets the prior message window, oldest first.
    #[must_use]
    pub fn with_history(mut self, history: Vec<BaseMessage>) -> Self {
        self.history = history;
        self
    }

    /// Sets the persisted `deepagent` state payload to restore.
    ///
    /// The payload must be a JSON object; anything else is rejected by
    /// [`crate::agent::seed_state`] when the turn runs.
    #[must_use]
    pub fn with_state(mut self, state: Value) -> Self {
        self.state = Some(state);
        self
    }
}

/// Runs one prompt on a prebuilt model, continuing a persisted session.
///
/// The resumed twin of [`run_with_model`](super::run::run_with_model):
///
/// 1. the prompt is resolved (and validated) first, so an empty prompt fails
///    before any model or agent work;
/// 2. the agent is built from `options` exactly as a fresh run builds it;
/// 3. a kernel session with `resumed.session_id` is created and seeded — the
///    history (when non-empty) through
///    [`DeepAgent::seed_history`](crate::agent::DeepAgent::seed_history), then
///    the `deepagent` state (when present) through [`seed_state`];
/// 4. the CLI session record is created only when it does not exist yet, so an
///    earlier transcript is never truncated;
/// 5. the turn is handed to `run::stream_run`, which streams, renders and
///    appends the new user/assistant messages to that record.
///
/// Seed failures propagate: a history or state payload the kernel rejects is the
/// caller's data problem, not something to paper over.
///
/// # Errors
///
/// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) for
/// an empty prompt, whatever agent construction or the seed reports (including
/// [`Error::Validation`](openjiuwen_common::error::Error::Validation) for a
/// non-object state payload), the error of a malformed CLI session record, and
/// the stream error of the turn. Session-write failures after the record exists
/// are logged, never returned.
pub async fn run_with_model_resumed(
    options: &CliOptions,
    args: &RunArgs,
    io: &CliIo,
    model: Arc<dyn Model>,
    resumed: ResumedTurn,
) -> Result<RunOutcome> {
    let prompt = run::resolve_prompt(args, io).await?;
    let agent = run::build_agent(model, options, io).await?;
    let session = run::new_session(&resumed.session_id);

    // Seeding must follow the build: a configured agent drops pooled contexts.
    if !resumed.history.is_empty() {
        agent
            .seed_history(&session, resumed.history.clone())
            .await?;
    }
    if let Some(state) = resumed.state.clone() {
        seed_state(session.as_ref(), state).await?;
    }

    // `new_session` truncates, so it is only for a record that does not exist.
    let store = options.sessions();
    if store.load(&resumed.session_id)?.is_none() {
        if let Err(error) = store.new_session(&resumed.session_id, &options.model) {
            tracing::warn!(%error, "failed to create CLI session record");
        }
    }

    run::stream_run(
        options,
        args,
        io,
        &agent,
        &session,
        &resumed.session_id,
        &prompt,
    )
    .await
}

/// Runs one prompt on a model built from `options`, continuing a persisted
/// session.
///
/// The resumed twin of [`run_once`](super::run::run_once): the prompt is resolved
/// before the model is built, so an empty prompt fails without touching
/// credentials or the network, and the resolved prompt is then handed down to
/// [`run_with_model_resumed`].
///
/// # Errors
///
/// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) for
/// an empty prompt, whatever model construction reports, and everything
/// [`run_with_model_resumed`] reports.
pub async fn run_once_resumed(
    options: &CliOptions,
    args: &RunArgs,
    io: &CliIo,
    resumed: ResumedTurn,
) -> Result<RunOutcome> {
    let prompt = run::resolve_prompt(args, io).await?;
    let model = options.build_model().await?;
    // Hand the resolved prompt down so stdin is read once, not twice.
    let resolved = RunArgs {
        prompt: Some(prompt),
        output_format: args.output_format,
    };
    run_with_model_resumed(options, &resolved, io, model, resumed).await
}
