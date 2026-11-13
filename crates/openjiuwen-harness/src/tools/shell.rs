//! Port of `openjiuwen/harness/tools/shell/`: the `bash` tool.
//!
//! The tool runs one command line through the sys-operation shell backend and renders the
//! captured output as a `Command:`/`Stdout:`/`Stderr:`/`Exit Code:` block, which is the
//! Python `BashTool` output shape. The card name (`bash`), the argument names, the
//! `max_output_chars` behaviour and every message are Python-compatible; the description
//! text comes from [`super::descriptions`].
//!
//! Deviations from Python, all deliberate:
//!
//! - Oversized output is truncated in place with a marker instead of being persisted to a
//!   temporary file with a preview.
//! - A non-zero exit is reported as a failed payload; Python's `interpret_exit_code`
//!   neutrality for `grep`/`diff`/`test`/`find` is not ported.
//! - `get_call_timeout`, the destructive-command warning, `sudo` rewriting, the
//!   `OPENJIUWEN_BASH_STRICT` injection scan, the `.agent_history` rm tracking and the
//!   `PowerShellTool` variant are deferred.
//! - The streaming `stream()` override is absent, so callers get the trait-default
//!   single-result stream.
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
};
use openjiuwen_sysop::operation::ShellOperation;
use openjiuwen_sysop::shell::types::{ShellExecParams, ShellType};

use super::descriptions::description;
use super::support::{parse_args, schema_for, tool_card, truncate_chars, ToolEnv, ToolPayload};

/// Timeout applied when the model omits `timeout` or passes `0`, in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Upper bound of the `timeout` argument, in seconds.
const MAX_TIMEOUT_SECS: u64 = 3600;

/// Output budget applied when the model omits `max_output_chars`, in characters.
const DEFAULT_MAX_OUTPUT_CHARS: u64 = 20_000;

/// Smallest positive output budget, in characters.
const MIN_MAX_OUTPUT_CHARS: u64 = 200;

/// Rendered in place of an empty captured stream.
const EMPTY_STREAM: &str = "(empty)";

/// Rendered in the Stderr slot when the combined output was truncated.
const TRUNCATED_STREAM: &str = "(see truncated output)";

/// Arguments of the `bash` tool.
///
/// Python parity: the `BashTool` invocation parameters. Only `command` is required; every
/// other field has a serde default and tolerates a value of the wrong type, because models
/// routinely emit `"timeout": "60"` or an unknown `shell_type`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct BashArgs {
    /// The command line to run through the shell.
    pub command: String,
    /// Timeout in seconds (default 300, max 3600).
    #[serde(default, deserialize_with = "deserialize_opt_u64")]
    #[schemars(with = "u64")]
    pub timeout: Option<u64>,
    /// Short description of what the command does, used for logging only.
    #[serde(default)]
    pub description: String,
    /// Start the command in the background and return its process id instead of waiting.
    #[serde(default)]
    pub run_in_background: bool,
    /// Working directory relative to the workspace root; defaults to the workspace root.
    #[serde(default)]
    pub workdir: String,
    /// Maximum characters of captured output to return; `0` means unlimited.
    #[serde(default, deserialize_with = "deserialize_opt_u64")]
    #[schemars(with = "u64")]
    pub max_output_chars: Option<u64>,
    /// Interpreter selection: `auto`, `cmd`, `powershell`, `bash` or `sh`.
    ///
    /// The schema is declared as a string and gets its enum members from
    /// `install_shell_type_enum`: a `$ref` into `$defs` would be stripped by
    /// [`schema_for`] and leave a dangling reference in the card.
    #[serde(default, deserialize_with = "deserialize_shell_type")]
    #[schemars(with = "String")]
    pub shell_type: ShellType,
}

impl BashArgs {
    /// The timeout to apply, in seconds.
    ///
    /// `None` and `0` select `DEFAULT_TIMEOUT_SECS`; anything else is clamped into
    /// `1..=3600`.
    #[must_use]
    pub fn resolve_timeout(&self) -> u64 {
        match self.timeout {
            None | Some(0) => DEFAULT_TIMEOUT_SECS,
            Some(value) => value.clamp(1, MAX_TIMEOUT_SECS),
        }
    }

    /// The output budget to apply, in characters.
    ///
    /// `None` selects `DEFAULT_MAX_OUTPUT_CHARS`; `0` means unlimited; any other value is
    /// raised to at least `MIN_MAX_OUTPUT_CHARS` so a tiny budget cannot hide all output.
    #[must_use]
    pub fn resolve_max_output_chars(&self) -> u64 {
        match self.max_output_chars {
            None => DEFAULT_MAX_OUTPUT_CHARS,
            Some(0) => 0,
            Some(value) => value.max(MIN_MAX_OUTPUT_CHARS),
        }
    }
}

/// Deserializes an optional `u64` from a number or a numeric string.
///
/// A missing field never reaches this function. Anything that is not a non-negative integer
/// (a float, a boolean, `null`, a non-numeric string) deserializes as `None` instead of
/// failing the whole argument object.
fn deserialize_opt_u64<'de, D>(deserializer: D) -> std::result::Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(json_u64))
}

/// Reads a non-negative integer out of a JSON number or a numeric string.
fn json_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Deserializes a [`ShellType`] leniently.
///
/// Unknown names, non-string values and `null` all become [`ShellType::Auto`], matching the
/// Python `ShellType.from_str` fallback. A missing field keeps the serde default (`Auto`).
fn deserialize_shell_type<'de, D>(deserializer: D) -> std::result::Result<ShellType, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(match value {
        Some(Value::String(text)) => ShellType::parse_lenient(&text),
        _ => ShellType::Auto,
    })
}

/// The `bash` tool: runs one command line through a [`ShellOperation`] backend.
///
/// The tool never returns [`Err`] for an expected failure — a rejected command, a non-zero
/// exit, a timeout or a missing backend all become a `success: false` payload. The only
/// [`Err`] it produces is [`openjiuwen_common::error::Error::Cancelled`] from the call's
/// cancellation token.
pub struct BashTool {
    /// Path resolution, permission gate and card language.
    env: ToolEnv,
    /// The shell backend; without one every call reports a failure payload.
    shell: Option<Arc<dyn ShellOperation>>,
    /// The model-facing card, built once from [`ToolEnv::language`].
    card: ToolCard,
}

impl BashTool {
    /// Builds a `bash` tool over `env`, running commands through `shell`.
    #[must_use]
    pub fn new(env: ToolEnv, shell: Option<Arc<dyn ShellOperation>>) -> Self {
        let card = Self::card_for(env.language());
        Self { env, shell, card }
    }

    /// Builds the model-facing card for the description language `language`.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        let mut input_params = schema_for::<BashArgs>();
        install_shell_type_enum(&mut input_params);
        tool_card("bash", description("bash", language), input_params, true)
    }

    /// Renders the Python `BashTool` result block.
    ///
    /// An empty stream renders as `EMPTY_STREAM`, so the model always sees which of the
    /// two streams produced output.
    #[must_use]
    pub fn render_content(command: &str, stdout: &str, stderr: &str, exit_code: i32) -> String {
        format!(
            "Command: {command}\nStdout: {}\n\nStderr: {}\nExit Code: {exit_code}",
            stream_text(stdout),
            stream_text(stderr)
        )
    }

    /// Runs the foreground path and returns the payload JSON.
    async fn execute_foreground(
        &self,
        shell: &dyn ShellOperation,
        params: ShellExecParams,
        command: &str,
        max_output_chars: u64,
    ) -> Value {
        let result = shell.execute_cmd(params).await;
        let operation_succeeded = result.is_success();
        let Some(output) = result.data else {
            return ToolPayload::fail(result.message).to_json();
        };

        let stdout = normalize_stdout(&output.stdout);
        let (stdout_slot, stderr_slot, truncated) =
            apply_output_budget(&stdout, &output.stderr, max_output_chars);
        let rendered = Self::render_content(command, &stdout_slot, &stderr_slot, output.exit_code);

        if operation_succeeded && output.is_success() {
            return ToolPayload::ok(json!({
                "content": rendered,
                "truncated": truncated,
            }))
            .to_json();
        }

        tracing::debug!(
            tool = "bash",
            exit_code = output.exit_code,
            timed_out = output.timed_out,
            cancelled = output.cancelled,
            "shell command did not succeed"
        );
        ToolPayload::fail_with_data(
            rendered.clone(),
            json!({
                "content": rendered,
                "exit_code": output.exit_code,
                "timed_out": output.timed_out,
                "truncated": truncated,
            }),
        )
        .to_json()
    }

    /// Runs the background path and returns the payload JSON.
    async fn execute_background(shell: &dyn ShellOperation, params: ShellExecParams) -> Value {
        let result = shell.execute_cmd_background(params).await;
        if !result.is_success() {
            return ToolPayload::fail(result.message).to_json();
        }
        let Some(output) = result.data else {
            return ToolPayload::fail(result.message).to_json();
        };
        ToolPayload::ok(json!({
            "pid": output.pid,
            "status": "started",
        }))
        .to_json()
    }
}

#[async_trait]
impl Tool for BashTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        let args = match parse_args::<BashArgs>(&inputs) {
            Ok(args) => args,
            Err(payload) => return Ok(payload.to_json()),
        };

        let command = args.command.trim().to_owned();
        if command.is_empty() {
            return Ok(ToolPayload::fail("command cannot be empty").to_json());
        }

        let workdir = match self
            .env
            .resolve_dir("bash", "workdir", &args.workdir, &inputs)
            .await
        {
            Ok(workdir) => workdir,
            Err(payload) => return Ok(payload.to_json()),
        };
        if !args.workdir.is_empty() && !is_directory(&workdir).await {
            return Ok(
                ToolPayload::fail(format!("workdir does not exist: {}", workdir.display()))
                    .to_json(),
            );
        }

        let Some(shell) = self.shell.clone() else {
            return Ok(ToolPayload::fail("shell operation is not configured").to_json());
        };

        ctx.check_cancelled()?;

        if !args.description.trim().is_empty() {
            tracing::debug!(
                tool = "bash",
                description = %args.description,
                command = %command,
                "running shell command"
            );
        }

        let params = ShellExecParams {
            command: command.clone(),
            cwd: Some(workdir.to_string_lossy().into_owned()),
            timeout_secs: Some(args.resolve_timeout()),
            shell_type: args.shell_type,
            session_id: ctx.session_id.clone(),
            ..ShellExecParams::default()
        };

        if args.run_in_background {
            return Ok(Self::execute_background(shell.as_ref(), params).await);
        }
        Ok(self
            .execute_foreground(
                shell.as_ref(),
                params,
                &command,
                args.resolve_max_output_chars(),
            )
            .await)
    }

    fn render_for_llm(&self, output: &Value) -> String {
        if let Some(pid) = output
            .get("data")
            .and_then(|data| data.get("pid"))
            .and_then(Value::as_u64)
        {
            return format!("Command started in background (pid {pid}).");
        }
        // Reproduce the trait default; calling `self.render_for_llm` here would recurse.
        match serde_json::from_value::<ToolOutput>(output.clone()) {
            Ok(parsed) => render_tool_output(&parsed),
            Err(_) => render_payload_text(output),
        }
    }
}

/// Writes the `shell_type` enum members into a [`BashArgs`] schema.
///
/// [`ShellType`] is a named type, so schemars would describe it as `$ref: "#/$defs/ShellType"`;
/// [`schema_for`] strips `$defs`, which would leave a dangling reference. The card carries the
/// members inline instead, matching the Python card.
fn install_shell_type_enum(schema: &mut Value) {
    let field = schema
        .get_mut("properties")
        .and_then(|properties| properties.get_mut("shell_type"));
    let Some(field) = field.and_then(Value::as_object_mut) else {
        return;
    };
    field.remove("$ref");
    field.insert("type".to_owned(), Value::String("string".to_owned()));
    field.insert(
        "enum".to_owned(),
        json!(["auto", "cmd", "powershell", "bash", "sh"]),
    );
}

/// Renders one captured stream, substituting [`EMPTY_STREAM`] when it is empty.
fn stream_text(text: &str) -> &str {
    if text.is_empty() {
        EMPTY_STREAM
    } else {
        text
    }
}

/// Strips leading whitespace-only lines from captured stdout, then trailing whitespace.
///
/// Python parity: `re.sub(r"^(\s*\n)+", "", stdout)` followed by `rstrip()`. Leading spaces
/// of the first line that carries text are kept.
fn normalize_stdout(text: &str) -> String {
    let mut rest = text;
    while let Some(index) = rest.find('\n') {
        if !rest[..index].trim().is_empty() {
            break;
        }
        rest = &rest[index + 1..];
    }
    rest.trim_end().to_owned()
}

/// Applies the `max_output_chars` budget to the combined stdout and stderr.
///
/// Returns `(stdout slot, stderr slot, truncated)`. The budget counts characters of
/// `stdout + "\n" + stderr`; when it bites, the truncated text fills the stdout slot and the
/// stderr slot points at it instead of repeating the tail. `max_chars == 0` is unlimited.
fn apply_output_budget(stdout: &str, stderr: &str, max_chars: u64) -> (String, String, bool) {
    let combined = if stderr.is_empty() {
        stdout.to_owned()
    } else {
        format!("{stdout}\n{stderr}")
    };
    let limit = usize::try_from(max_chars).unwrap_or(usize::MAX);
    let truncated = truncate_chars(&combined, limit);
    if !truncated.truncated {
        return (stdout.to_owned(), stderr.to_owned(), false);
    }
    let stderr_slot = if stderr.is_empty() {
        EMPTY_STREAM
    } else {
        TRUNCATED_STREAM
    };
    (truncated.text, stderr_slot.to_owned(), true)
}

/// Whether `path` is an existing directory.
async fn is_directory(path: &Path) -> bool {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata.is_dir(),
        Err(error) => {
            tracing::debug!(
                tool = "bash",
                path = %path.display(),
                %error,
                "workdir check failed"
            );
            false
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use openjiuwen_sysop::shell::types::{ShellExecParams, ShellOutput};
    use openjiuwen_sysop::{LocalWorkConfig, SysOperation};

    use super::*;
    use crate::workspace::Workspace;

    /// A tool env over a fresh temporary workspace.
    fn env(dir: &tempfile::TempDir) -> ToolEnv {
        ToolEnv::new(Workspace::new(dir.path())).with_language("en")
    }

    /// A tool over `dir` backed by the local shell, or `None` when the host shell cannot run
    /// a trivial command (the test then skips instead of failing on an exotic host).
    async fn local_tool(dir: &tempfile::TempDir) -> Option<BashTool> {
        let operation =
            SysOperation::local(LocalWorkConfig::unrestricted(dir.path().to_string_lossy()))
                .ok()?;
        let shell = operation.shell()?;
        let probe = shell
            .execute_cmd(ShellExecParams::new("echo openjiuwen"))
            .await;
        let usable = probe.data.as_ref().is_some_and(|output: &ShellOutput| {
            output.is_success() && output.stdout.contains("openjiuwen")
        });
        if !usable {
            eprintln!("skipping: the local shell backend cannot run `echo`");
            return None;
        }
        Some(BashTool::new(env(dir), Some(shell)))
    }

    /// Invokes `tool` with nothing cancelled.
    async fn invoke(tool: &BashTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    #[test]
    fn card_matches_the_python_bash_tool() {
        let card = BashTool::card_for("en");
        assert_eq!(card.name, "bash");
        assert_eq!(card.description, description("bash", "en"));
        assert!(card.parallel_safe);
        assert_eq!(card.id.len(), 32);

        let schema = &card.input_params;
        assert_eq!(schema["type"], "object");
        let properties = schema["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "command",
                "description",
                "max_output_chars",
                "run_in_background",
                "shell_type",
                "timeout",
                "workdir",
            ]
        );
        assert_eq!(schema["required"], json!(["command"]));

        // `#[schemars(inline)]` keeps the enum in the card; `schema_for` strips `$defs`, so a
        // `$ref` would resolve to nothing.
        let shell_type = &properties["shell_type"];
        assert_eq!(
            shell_type["enum"],
            json!(["auto", "cmd", "powershell", "bash", "sh"])
        );

        // `#[schemars(with = "u64")]` keeps the numeric arguments plain integers.
        assert_eq!(properties["timeout"]["type"], "integer");
        assert_eq!(properties["max_output_chars"]["type"], "integer");
        assert_eq!(properties["run_in_background"]["type"], "boolean");
    }

    #[test]
    fn card_uses_the_configured_language() {
        assert_eq!(
            BashTool::card_for("cn").description,
            description("bash", "cn")
        );

        let dir = tempfile::tempdir().unwrap();
        let tool = BashTool::new(
            ToolEnv::new(Workspace::new(dir.path())).with_language("cn"),
            None,
        );
        assert_eq!(tool.card().description, description("bash", "cn"));
        assert!(tool.is_parallel_safe());
    }

    #[test]
    fn timeout_and_output_budget_resolve_like_python() {
        let args = |value: Value| serde_json::from_value::<BashArgs>(value).unwrap();

        assert_eq!(args(json!({"command": "x"})).resolve_timeout(), 300);
        assert_eq!(
            args(json!({"command": "x", "timeout": 0})).resolve_timeout(),
            300
        );
        assert_eq!(
            args(json!({"command": "x", "timeout": 1})).resolve_timeout(),
            1
        );
        assert_eq!(
            args(json!({"command": "x", "timeout": 5000})).resolve_timeout(),
            3600
        );

        assert_eq!(
            args(json!({"command": "x"})).resolve_max_output_chars(),
            20_000
        );
        assert_eq!(
            args(json!({"command": "x", "max_output_chars": 0})).resolve_max_output_chars(),
            0
        );
        assert_eq!(
            args(json!({"command": "x", "max_output_chars": 100})).resolve_max_output_chars(),
            200
        );
        assert_eq!(
            args(json!({"command": "x", "max_output_chars": 5000})).resolve_max_output_chars(),
            5000
        );
    }

    #[test]
    fn argument_deserialization_is_lenient() {
        let args = serde_json::from_value::<BashArgs>(json!({
            "command": "x",
            "shell_type": "FISH",
            "timeout": "abc",
            "max_output_chars": "nope",
        }))
        .unwrap();
        assert_eq!(args.shell_type, ShellType::Auto);
        assert_eq!(args.timeout, None);
        assert_eq!(args.max_output_chars, None);
        assert_eq!(args.resolve_timeout(), 300);
        assert_eq!(args.resolve_max_output_chars(), 20_000);

        let strings = serde_json::from_value::<BashArgs>(json!({
            "command": "x",
            "shell_type": " BASH ",
            "timeout": "42",
            "max_output_chars": 300,
        }))
        .unwrap();
        assert_eq!(strings.shell_type, ShellType::Bash);
        assert_eq!(strings.timeout, Some(42));
        assert_eq!(strings.max_output_chars, Some(300));

        let wrong_types = serde_json::from_value::<BashArgs>(json!({
            "command": "x",
            "timeout": -5,
            "max_output_chars": true,
            "shell_type": 7,
        }))
        .unwrap();
        assert_eq!(wrong_types.timeout, None);
        assert_eq!(wrong_types.max_output_chars, None);
        assert_eq!(wrong_types.shell_type, ShellType::Auto);

        let defaults = serde_json::from_value::<BashArgs>(json!({"command": "ls"})).unwrap();
        assert_eq!(defaults.description, "");
        assert!(!defaults.run_in_background);
        assert_eq!(defaults.workdir, "");
        assert_eq!(defaults.shell_type, ShellType::Auto);
    }

    #[test]
    fn render_content_matches_the_python_block() {
        assert_eq!(
            BashTool::render_content("echo hi", "hi", "", 0),
            "Command: echo hi\nStdout: hi\n\nStderr: (empty)\nExit Code: 0"
        );
        assert_eq!(
            BashTool::render_content("false", "", "", 1),
            "Command: false\nStdout: (empty)\n\nStderr: (empty)\nExit Code: 1"
        );
        assert_eq!(
            BashTool::render_content("x", "out", "err", 2),
            "Command: x\nStdout: out\n\nStderr: err\nExit Code: 2"
        );
    }

    #[test]
    fn normalize_stdout_strips_leading_blank_lines_and_trailing_space() {
        assert_eq!(normalize_stdout("\n\nhello\n"), "hello");
        assert_eq!(normalize_stdout("  \n\t\nhello\n\n"), "hello");
        assert_eq!(normalize_stdout("  indented\n"), "  indented");
        assert_eq!(normalize_stdout(""), "");
        assert_eq!(normalize_stdout("\n \n"), "");
    }

    #[test]
    fn output_budget_truncates_the_combined_stream() {
        let (stdout, stderr, truncated) = apply_output_budget("short", "err", 20_000);
        assert_eq!(stdout, "short");
        assert_eq!(stderr, "err");
        assert!(!truncated);

        let unlimited = apply_output_budget(&"a".repeat(50), "", 0);
        assert_eq!(unlimited.0.len(), 50);
        assert!(!unlimited.2);

        let long = "a".repeat(400);
        let (stdout, stderr, truncated) = apply_output_budget(&long, "boom", 200);
        assert!(truncated);
        assert!(stdout.starts_with(&"a".repeat(200)));
        assert!(stdout.contains("output truncated"));
        assert_eq!(stderr, TRUNCATED_STREAM);

        let (stdout, stderr, truncated) = apply_output_budget(&long, "", 200);
        assert!(truncated);
        assert!(stdout.contains("output truncated"));
        assert_eq!(stderr, EMPTY_STREAM);
    }

    #[tokio::test]
    async fn parse_failure_and_blank_command_are_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let tool = BashTool::new(env(&dir), None);

        let not_an_object = invoke(&tool, json!([])).await;
        assert_eq!(not_an_object["success"], false);
        assert_eq!(not_an_object["error"], "tool arguments must be an object");

        for command in ["", "   ", "\n\t"] {
            let blank = invoke(&tool, json!({"command": command})).await;
            assert_eq!(blank["success"], false, "for {command:?}");
            assert_eq!(blank["error"], "command cannot be empty");
            assert_eq!(blank["data"], Value::Null);
        }
    }

    #[tokio::test]
    async fn missing_workdir_is_reported_with_its_resolved_path() {
        let dir = tempfile::tempdir().unwrap();
        let tool = BashTool::new(env(&dir), None);

        let payload = invoke(&tool, json!({"command": "pwd", "workdir": "nope"})).await;
        assert_eq!(payload["success"], false);
        let expected = format!(
            "workdir does not exist: {}",
            std::fs::canonicalize(dir.path())
                .unwrap()
                .join("nope")
                .display()
        );
        assert_eq!(payload["error"], expected.as_str());
        assert_eq!(payload["data"], Value::Null);
    }

    #[tokio::test]
    async fn missing_shell_backend_is_a_failure_payload() {
        let dir = tempfile::tempdir().unwrap();
        let tool = BashTool::new(env(&dir), None);

        let payload = invoke(&tool, json!({"command": "echo hi"})).await;
        assert_eq!(payload["success"], false);
        assert_eq!(payload["error"], "shell operation is not configured");
    }

    #[tokio::test]
    async fn cancellation_propagates_as_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let Some(tool) = local_tool(&dir).await else {
            return;
        };
        let ctx = ToolContext::new();
        ctx.cancellation.cancel();

        let error = tool
            .invoke(json!({"command": "echo hi"}), &ctx)
            .await
            .unwrap_err();
        assert!(error.is_cancelled(), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn echo_succeeds_and_renders_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let Some(tool) = local_tool(&dir).await else {
            return;
        };

        let payload = invoke(&tool, json!({"command": "echo hello"})).await;
        assert_eq!(payload["success"], true, "payload: {payload}");
        assert_eq!(payload["data"]["truncated"], false);
        let content = payload["data"]["content"].as_str().unwrap();
        assert!(content.contains("Command: echo hello"), "{content}");
        assert!(content.contains("Stdout: hello"), "{content}");
        assert!(content.contains("Exit Code: 0"), "{content}");
        assert!(content.contains("Stderr: (empty)"), "{content}");

        assert_eq!(tool.render_for_llm(&payload), content);
    }

    #[tokio::test]
    async fn non_zero_exit_is_a_failure_payload_with_the_rendered_block() {
        let dir = tempfile::tempdir().unwrap();
        let Some(tool) = local_tool(&dir).await else {
            return;
        };

        let payload = invoke(&tool, json!({"command": "echo boom >&2; exit 3"})).await;
        assert_eq!(payload["success"], false);
        assert_eq!(payload["data"]["exit_code"], 3);
        assert_eq!(payload["data"]["timed_out"], false);
        let content = payload["data"]["content"].as_str().unwrap();
        assert!(content.contains("Stdout: (empty)"), "{content}");
        assert!(content.contains("Stderr: boom"), "{content}");
        assert!(content.contains("Exit Code: 3"), "{content}");
        assert_eq!(payload["error"], content);
    }

    #[tokio::test]
    async fn timeout_keeps_the_partial_output() {
        let dir = tempfile::tempdir().unwrap();
        let Some(tool) = local_tool(&dir).await else {
            return;
        };

        let payload = invoke(
            &tool,
            json!({"command": "echo started; sleep 5", "timeout": 1}),
        )
        .await;
        assert_eq!(payload["success"], false);
        assert_eq!(payload["data"]["timed_out"], true, "payload: {payload}");
        let content = payload["data"]["content"].as_str().unwrap();
        assert!(content.contains("Stdout: started"), "{content}");
    }

    #[tokio::test]
    async fn background_start_reports_the_pid() {
        let dir = tempfile::tempdir().unwrap();
        let Some(tool) = local_tool(&dir).await else {
            return;
        };

        let payload = invoke(
            &tool,
            json!({"command": "sleep 1", "run_in_background": true}),
        )
        .await;
        assert_eq!(payload["success"], true, "payload: {payload}");
        assert_eq!(payload["data"]["status"], "started");
        let pid = payload["data"]["pid"].as_u64().unwrap();
        assert_eq!(
            tool.render_for_llm(&payload),
            format!("Command started in background (pid {pid}).")
        );
    }

    #[tokio::test]
    async fn oversized_output_is_truncated_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let Some(tool) = local_tool(&dir).await else {
            return;
        };

        let payload = invoke(
            &tool,
            json!({"command": "seq 1 20000", "max_output_chars": 200}),
        )
        .await;
        assert_eq!(payload["success"], true, "payload: {payload}");
        assert_eq!(payload["data"]["truncated"], true);
        let content = payload["data"]["content"].as_str().unwrap();
        assert!(content.contains("output truncated"), "{content}");
        assert!(content.contains("Stdout: 1\n2\n3"), "{content}");
        assert!(content.contains("Stderr: (empty)"), "{content}");
    }
}
