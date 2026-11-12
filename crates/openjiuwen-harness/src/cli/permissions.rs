//! Approval prompts and the permission host the CLI installs.
//!
//! When the permission rail raises an ask, the CLI answers it on the user's
//! behalf: [`render_prompt`] formats the Python-compatible prompt,
//! [`PermissionAnswer::parse`] maps one typed line onto a
//! [`PermissionConfirmResponse`], and [`CliPermissionHost`] wires both into
//! [`PermissionHost`] through the injectable [`super::io::CliIo`] handles, so the
//! REPL renders the stream and asks for approval through the same terminal.
//!
//! Failure policy is fail-closed: a non-interactive handle (a one-shot `run`, a
//! piped stdin) never prompts at all, and end of input or an IO error rejects the
//! call instead of approving it. Nothing here panics, nothing writes to stdout
//! directly, and the prompt text carries only the tool name and its arguments —
//! never credentials.

use std::fmt;
use std::path::PathBuf;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value;

use super::io::CliIo;
use crate::permission::{
    ConfirmationOutcome, PermissionConfirmResponse, PermissionConfirmationRequest, PermissionHost,
    PermissionsSection,
};

/// Longest argument value shown in a prompt before it is elided.
const MAX_ARGUMENT_VALUE_CHARS: usize = 200;

/// Maximum number of argument lines a prompt shows.
const MAX_ARGUMENT_LINES: usize = 3;

/// How the user answered an approval prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionAnswer {
    /// Run the call once.
    Approve,
    /// Run the call and remember the answer for the session.
    Always,
    /// Reject the call with no feedback for the model.
    Reject,
    /// Reject the call and pass the typed text back to the model as feedback.
    Feedback(String),
}

impl PermissionAnswer {
    /// Parses one typed answer line.
    ///
    /// The input is trimmed and lowercased for matching; anything unrecognised
    /// becomes [`PermissionAnswer::Feedback`] carrying the **original** trimmed
    /// text, so the user's wording reaches the model verbatim. An empty line
    /// approves once, which keeps pressing Enter the fast path.
    #[must_use]
    pub fn parse(answer: &str) -> Self {
        let trimmed = answer.trim();
        match trimmed.to_lowercase().as_str() {
            "" | "y" | "yes" | "ok" | "approve" | "allow" | "allow_once" | "true" | "1" => {
                Self::Approve
            }
            "a" | "always" | "allow_always" => Self::Always,
            "n" | "no" | "deny" | "reject" => Self::Reject,
            _ => Self::Feedback(trimmed.to_owned()),
        }
    }

    /// Maps the answer onto the rail's confirmation payload.
    ///
    /// Only [`PermissionAnswer::Always`] sets `auto_confirm`, so the rail decides
    /// whether the remembered rule is session-scoped or permanent. `persist_allow`
    /// stays `None` (the legacy "host did not say" value) rather than claiming a
    /// permanent YAML write the CLI does not perform yet.
    #[must_use]
    pub fn to_response(&self) -> PermissionConfirmResponse {
        match self {
            Self::Approve => PermissionConfirmResponse {
                approved: true,
                feedback: String::new(),
                auto_confirm: false,
                persist_allow: None,
            },
            Self::Always => PermissionConfirmResponse {
                approved: true,
                feedback: String::new(),
                auto_confirm: true,
                persist_allow: None,
            },
            Self::Reject => PermissionConfirmResponse {
                approved: false,
                feedback: String::new(),
                auto_confirm: false,
                persist_allow: None,
            },
            Self::Feedback(text) => PermissionConfirmResponse {
                approved: false,
                feedback: text.clone(),
                auto_confirm: false,
                persist_allow: None,
            },
        }
    }
}

/// Renders the approval prompt for one ask.
///
/// Layout (lines joined with `\n`, no trailing newline):
///
/// ```text
/// ⚠ Approve {tool}?
///   {key}: {value}
/// (y/yes, a/always, n/no, or feedback)
/// Approve? (y/a/n)
/// ```
///
/// At most three argument lines are shown, in the JSON object's insertion order;
/// non-object arguments yield no argument lines at all.
#[must_use]
pub fn render_prompt(tool: &str, arguments: &Value) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(MAX_ARGUMENT_LINES + 3);
    lines.push(format!("⚠ Approve {tool}?"));
    if let Some(object) = arguments.as_object() {
        lines.extend(
            object
                .iter()
                .take(MAX_ARGUMENT_LINES)
                .map(|(key, value)| format!("  {key}: {}", render_argument_value(value))),
        );
    }
    lines.push("(y/yes, a/always, n/no, or feedback)".to_owned());
    lines.push("Approve? (y/a/n)".to_owned());
    lines.join("\n")
}

/// Renders one argument value: strings raw, everything else as compact JSON.
fn render_argument_value(value: &Value) -> String {
    match value {
        Value::String(text) => truncate_argument(text),
        other => truncate_argument(&other.to_string()),
    }
}

/// Elides a value past [`MAX_ARGUMENT_VALUE_CHARS`] characters.
///
/// Counting characters (not bytes) keeps the cut on a UTF-8 boundary, so a
/// non-ASCII argument can never panic the prompt.
fn truncate_argument(text: &str) -> String {
    let mut chars = text.chars();
    let head: String = chars.by_ref().take(MAX_ARGUMENT_VALUE_CHARS).collect();
    if chars.next().is_some() {
        format!("{head}...")
    } else {
        head
    }
}

/// [`PermissionHost`] backed by the CLI's terminal.
///
/// It holds the resolved session id and workspace (so the rail can scope rules
/// and validate paths), the current `permissions:` section, and a [`CliIo`] clone
/// for the prompt itself. The section sits behind a synchronous [`RwLock`] that is
/// only read to clone the data out — it is never held across an `.await`.
pub struct CliPermissionHost {
    /// Terminal the prompt is written to and read from.
    io: CliIo,
    /// Session the approvals are scoped to.
    session_id: String,
    /// Workspace root handed to the file guard.
    workspace: PathBuf,
    /// Latest merged `permissions:` section, refreshed by persistence.
    permissions: RwLock<Option<PermissionsSection>>,
}

impl CliPermissionHost {
    /// Builds a host for one session.
    #[must_use]
    pub fn new(
        io: CliIo,
        session_id: String,
        workspace: PathBuf,
        permissions: PermissionsSection,
    ) -> Self {
        Self {
            io,
            session_id,
            workspace,
            permissions: RwLock::new(Some(permissions)),
        }
    }
}

impl fmt::Debug for CliPermissionHost {
    /// Debug omits the terminal handle (its writer is not printable) and prints
    /// the session-scoped state only. The section may hold user rules but no
    /// secrets, and `api_key` never reaches this struct.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CliPermissionHost")
            .field("io", &"<terminal>")
            .field("session_id", &self.session_id)
            .field("workspace", &self.workspace)
            .field("permissions", &self.permissions.read())
            .finish()
    }
}

#[async_trait]
impl PermissionHost for CliPermissionHost {
    fn session_id(&self) -> Option<String> {
        Some(self.session_id.clone())
    }

    fn resolve_workspace_dir(&self) -> Option<PathBuf> {
        Some(self.workspace.clone())
    }

    fn permissions_snapshot(&self) -> Option<PermissionsSection> {
        self.permissions.read().clone()
    }

    async fn request_confirmation(
        &self,
        request: &PermissionConfirmationRequest,
    ) -> ConfirmationOutcome {
        // Fail closed without prompting: a one-shot run has no user to ask, and a
        // silent approval would defeat the permission engine entirely.
        if !self.io.is_interactive() {
            return ConfirmationOutcome::Failed;
        }
        let prompt = render_prompt(&request.tool_name, &request.arguments);
        match self.io.prompt_line(&prompt).await {
            Ok(Some(line)) => {
                ConfirmationOutcome::Response(PermissionAnswer::parse(&line).to_response())
            }
            // End of input: no answer, so the call is rejected.
            Ok(None) => ConfirmationOutcome::Failed,
            Err(error) => {
                tracing::warn!(%error, "permission prompt failed");
                ConfirmationOutcome::Failed
            }
        }
    }

    async fn persist_session_allow_rule(&self, permissions: &PermissionsSection) -> bool {
        *self.permissions.write() = Some(permissions.clone());
        true
    }

    async fn persist_allow_rule(&self, _permissions: &PermissionsSection) -> bool {
        // No permanent YAML write-back yet; the rail then keeps the rule in session
        // scope instead of claiming it was written.
        false
    }
}
