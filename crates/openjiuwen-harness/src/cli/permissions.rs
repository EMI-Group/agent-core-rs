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
//!
//! "Always allow" answers are now persisted as YAML under the CLI home
//! ([`allow_rules_path`](super::paths::allow_rules_path), i.e.
//! `<home>/permissions.yaml`) when the host was built with
//! [`CliPermissionHost::with_allow_rules`]. The file is reloaded on the next run,
//! so a stored rule takes effect on the **next process start** rather than inside
//! the current session; within the session the rail's own auto-confirm handles
//! the repeat asks. A host built without rules keeps the previous behaviour and
//! never writes anything.

use std::fmt;
use std::path::PathBuf;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value;

use super::allow_rules::AllowRules;
use super::io::CliIo;
use super::style::Style;
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
/// non-object arguments yield no argument lines at all. The text is unstyled;
/// see [`render_prompt_styled`] for the coloured form.
#[must_use]
pub fn render_prompt(tool: &str, arguments: &Value) -> String {
    render_prompt_styled(tool, arguments, Style::plain())
}

/// Renders the approval prompt with `style` applied.
///
/// The layout is exactly [`render_prompt`]'s; only the colours differ:
///
/// - the `⚠ Approve {tool}?` header is bold yellow;
/// - each argument line is dim;
/// - the `(y/yes, a/always, n/no, or feedback)` legend is dim.
///
/// With [`Style::plain`] the result is byte-identical to [`render_prompt`], so
/// piped and one-shot output never carries escape sequences.
#[must_use]
pub fn render_prompt_styled(tool: &str, arguments: &Value, style: Style) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(MAX_ARGUMENT_LINES + 3);
    lines.push(style.bold_yellow(&format!("⚠ Approve {tool}?")));
    if let Some(object) = arguments.as_object() {
        lines.extend(
            object.iter().take(MAX_ARGUMENT_LINES).map(|(key, value)| {
                style.dim(&format!("  {key}: {}", render_argument_value(value)))
            }),
        );
    }
    lines.push(style.dim("(y/yes, a/always, n/no, or feedback)"));
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
///
/// The optional [`AllowRules`] handle makes "always allow" answers permanent: it
/// points at `<home>/permissions.yaml` and is written whenever the rail reports a
/// permanent rule. Without it the host keeps every rule in session scope and
/// never touches the disk.
pub struct CliPermissionHost {
    /// Terminal the prompt is written to and read from.
    io: CliIo,
    /// Session the approvals are scoped to.
    session_id: String,
    /// Workspace root handed to the file guard.
    workspace: PathBuf,
    /// Latest merged `permissions:` section, refreshed by persistence.
    permissions: RwLock<Option<PermissionsSection>>,
    /// Where permanent "always allow" rules are stored, when configured.
    allow_rules: Option<AllowRules>,
    /// Styling applied to the approval prompt.
    style: Style,
}

impl CliPermissionHost {
    /// Builds a host for one session.
    ///
    /// The host starts without permanent allow-rule persistence and with plain
    /// (unstyled) prompts; [`CliPermissionHost::with_allow_rules`] and
    /// [`CliPermissionHost::with_style`] opt into either behaviour.
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
            allow_rules: None,
            style: Style::plain(),
        }
    }

    /// Returns this host persisting "always allow" answers through `rules`.
    #[must_use]
    pub fn with_allow_rules(mut self, rules: AllowRules) -> Self {
        self.allow_rules = Some(rules);
        self
    }

    /// Returns this host rendering its prompts with `style`.
    #[must_use]
    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Returns the filesystem location permanent rules are written to, if any.
    ///
    /// `None` means the host was built without [`AllowRules`], so "always allow"
    /// answers stay session-scoped.
    #[must_use]
    pub fn permission_yaml_path(&self) -> Option<PathBuf> {
        self.allow_rules
            .as_ref()
            .map(|rules| rules.path().to_path_buf())
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
            .field("allow_rules", &self.allow_rules)
            .field("style", &self.style)
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
        let prompt = render_prompt_styled(&request.tool_name, &request.arguments, self.style);
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

    async fn persist_allow_rule(&self, permissions: &PermissionsSection) -> bool {
        let Some(rules) = &self.allow_rules else {
            // No permanent write-back configured; the rail then keeps the rule in
            // session scope instead of claiming it was written.
            return false;
        };
        match rules.persist(permissions) {
            Ok(()) => {
                tracing::debug!(
                    path = %rules.path().display(),
                    "persisted permanent allow rules"
                );
                true
            }
            Err(error) => {
                tracing::warn!(
                    path = %rules.path().display(),
                    %error,
                    "failed to persist permanent allow rules"
                );
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use openjiuwen_common::error::Result;
    use tempfile::TempDir;

    use super::*;
    use crate::cli::LineReader;
    use crate::cli::Writer;
    use crate::permission::PermissionLevel;

    /// [`LineReader`] answering from a fixed script, then EOF.
    struct ScriptedReader {
        /// Lines still to hand out.
        lines: VecDeque<String>,
    }

    #[async_trait]
    impl LineReader for ScriptedReader {
        async fn read_line(&mut self) -> Result<Option<String>> {
            Ok(self.lines.pop_front())
        }
    }

    /// [`Writer`] appending everything to a shared buffer.
    struct SharedWriter {
        /// Sink every write lands in.
        buffer: Arc<Mutex<String>>,
    }

    #[async_trait]
    impl Writer for SharedWriter {
        async fn write(&mut self, text: &str) -> Result<()> {
            self.buffer.lock().unwrap().push_str(text);
            Ok(())
        }

        async fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }

    /// Builds a scripted interactive handle plus its captured output.
    fn interactive_io(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
        let buffer = Arc::new(Mutex::new(String::new()));
        let reader = Box::new(ScriptedReader {
            lines: lines.iter().map(|line| (*line).to_owned()).collect(),
        });
        let writer = Box::new(SharedWriter {
            buffer: Arc::clone(&buffer),
        });
        (CliIo::with_interactive(reader, writer), buffer)
    }

    /// Builds a scripted non-interactive handle plus its captured output.
    fn non_interactive_io(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
        let buffer = Arc::new(Mutex::new(String::new()));
        let reader = Box::new(ScriptedReader {
            lines: lines.iter().map(|line| (*line).to_owned()).collect(),
        });
        let writer = Box::new(SharedWriter {
            buffer: Arc::clone(&buffer),
        });
        (CliIo::new(reader, writer), buffer)
    }

    /// Builds a host over a scripted terminal with the default section.
    fn host(io: CliIo) -> CliPermissionHost {
        CliPermissionHost::new(
            io,
            "cli-test".to_owned(),
            PathBuf::from("/workspace"),
            PermissionsSection::default(),
        )
    }

    /// Builds a section allowing `write_file` permanently.
    fn allow_write_file() -> PermissionsSection {
        let mut section = PermissionsSection::default();
        section
            .tools
            .insert("write_file".to_owned(), PermissionLevel::Allow);
        section
    }

    /// Builds a permission request as the interrupt rail would.
    fn request() -> PermissionConfirmationRequest {
        PermissionConfirmationRequest {
            tool_name: "write_file".to_owned(),
            tool_call_id: "call-1".to_owned(),
            arguments: serde_json::json!({"path": "a.txt"}),
            result: crate::permission::PermissionResult::new(PermissionLevel::Ask),
            auto_confirm_key: String::new(),
        }
    }

    #[test]
    fn styled_prompt_is_byte_identical_to_plain_when_styling_is_off() {
        let arguments = serde_json::json!({"path": "a.txt", "content": "hi"});
        let plain = render_prompt("write_file", &arguments);
        let styled = render_prompt_styled("write_file", &arguments, Style::plain());
        assert_eq!(plain, styled);
        assert!(!styled.contains('\x1b'));
        assert_eq!(
            render_prompt_styled("write_file", &arguments, Style::new(false)),
            plain
        );
    }

    #[test]
    fn styled_prompt_colours_header_arguments_and_legend() {
        let arguments = serde_json::json!({"path": "a.txt"});
        let styled = render_prompt_styled("write_file", &arguments, Style::new(true));
        assert!(
            styled.contains("\x1b[1;33m"),
            "header must be bold yellow:\n{styled}"
        );
        assert!(
            styled.contains("\x1b[2m"),
            "arguments must be dim:\n{styled}"
        );
        let lines: Vec<&str> = styled.split('\n').collect();
        assert!(lines[0].starts_with("\x1b[1;33m⚠ Approve write_file?"));
        assert!(lines[1].starts_with("\x1b[2m  path: a.txt"));
        assert!(lines[2].starts_with("\x1b[2m(y/yes, a/always, n/no, or feedback)"));
        assert_eq!(lines[3], "Approve? (y/a/n)");
    }

    #[tokio::test]
    async fn persist_allow_rule_without_rules_writes_nothing_and_returns_false() {
        let temp = TempDir::new().unwrap();
        let (io, _buffer) = interactive_io(&[]);
        let host = host(io);
        assert!(host.permission_yaml_path().is_none());
        assert!(!host.persist_allow_rule(&allow_write_file()).await);
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn persist_allow_rule_with_rules_writes_the_yaml_section() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("permissions.yaml");
        let rules = AllowRules::new(&path);
        let (io, _buffer) = interactive_io(&[]);
        let host = host(io).with_allow_rules(rules.clone());
        assert_eq!(host.permission_yaml_path().as_deref(), Some(path.as_path()));

        let section = allow_write_file();
        assert!(host.persist_allow_rule(&section).await);
        assert!(path.exists(), "the rules file must have been written");
        assert_eq!(rules.load().unwrap(), Some(section));

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("write_file: allow"),
            "unexpected YAML document:\n{text}"
        );
    }

    #[tokio::test]
    async fn persist_allow_rule_reports_false_when_the_write_fails() {
        let temp = TempDir::new().unwrap();
        // A directory occupying the rules path makes the atomic rename fail.
        let path = temp.path().join("permissions.yaml");
        std::fs::create_dir_all(&path).unwrap();
        let (io, _buffer) = interactive_io(&[]);
        let host = host(io).with_allow_rules(AllowRules::new(&path));
        assert!(!host.persist_allow_rule(&allow_write_file()).await);
    }

    #[tokio::test]
    async fn interactive_styled_host_renders_the_coloured_prompt() {
        let (io, buffer) = interactive_io(&["y"]);
        let host = host(io).with_style(Style::new(true));
        let outcome = host.request_confirmation(&request()).await;
        let rendered = buffer.lock().unwrap().clone();
        assert!(
            rendered.contains("\x1b[1;33m⚠ Approve write_file?"),
            "unexpected prompt:\n{rendered}"
        );
        match outcome {
            ConfirmationOutcome::Response(response) => assert!(response.approved),
            other => panic!("expected a response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_interactive_host_fails_closed_without_prompting() {
        // The script holds a valid approval: a failing outcome proves the prompt
        // was never rendered and the answer never consumed.
        let (io, buffer) = non_interactive_io(&["y"]);
        let host = host(io);
        let outcome = host.request_confirmation(&request()).await;
        assert_eq!(outcome, ConfirmationOutcome::Failed);
        assert!(
            buffer.lock().unwrap().is_empty(),
            "a non-interactive host must not render a prompt"
        );
    }
}
