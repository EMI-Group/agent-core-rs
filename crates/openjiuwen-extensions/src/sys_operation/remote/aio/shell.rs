//! The kernel's [`ShellOperation`] over the AIO gateway envelope.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/aio.py`
//! reports completion as success even for a non-zero exit code and fails only on
//! a container-side timeout (`199004`); `execute_argv` has no Python counterpart
//! and is sent as the POSIX-quoted `execute_cmd` line. [`AioShell`] reproduces
//! that, and maps every gateway/transport failure to `199007`.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use openjiuwen_core::sys_operation::operation::{OpStream, ShellOperation};
use openjiuwen_core::sys_operation::result::{shell_failure, OpResult};
use openjiuwen_core::sys_operation::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput, ShellType,
    StreamKind,
};
use openjiuwen_core::sys_operation::status::SysOpStatus;

use super::super::semantics::{posix_quote, text_lines};
use super::super::transport::gateway_failure;
use super::{json_i32, json_u64, text_field, AioClient, DEFAULT_TIMEOUT_SECS, TIMEOUT_EXIT_CODE};

/// The kernel's four [`ShellOperation`] methods over the envelope.
pub(crate) struct AioShell {
    /// Shared envelope client.
    pub(super) client: Arc<AioClient>,
}

impl AioShell {
    /// Sends one `execute_cmd` envelope and maps its output.
    ///
    /// `operation` names the calling kernel method in failure messages, so
    /// `execute_argv` reports itself rather than `execute_cmd`.
    async fn run_command(
        &self,
        operation: &str,
        command: &str,
        cwd: Option<&str>,
        timeout: u64,
        environment: &BTreeMap<String, Option<String>>,
        shell_type: ShellType,
    ) -> OpResult<ShellOutput> {
        let payload = json!({
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
            "environment": environment,
            "shell_type": shell_type.as_str(),
        });
        let data = match self.client.invoke("shell", "execute_cmd", payload).await {
            Ok(data) => data,
            Err(error) => return gateway_failure(operation, &error),
        };
        let output = shell_output(command, cwd, &data);
        if output.timed_out {
            return OpResult {
                code: SysOpStatus::ShellExecutionError.code(),
                message: format!("execution timeout after {timeout} seconds"),
                data: Some(output),
            };
        }
        OpResult::success(output)
    }
}

#[async_trait]
impl ShellOperation for AioShell {
    async fn execute_cmd(&self, params: ShellExecParams) -> OpResult<ShellOutput> {
        if params.command.trim().is_empty() {
            return shell_failure("execute_cmd", "command can not be empty");
        }
        let timeout = params.effective_timeout_secs(DEFAULT_TIMEOUT_SECS);
        let cwd = normalized_cwd(params.cwd.as_deref());
        self.run_command(
            "execute_cmd",
            &params.command,
            cwd.as_deref(),
            timeout,
            &params.environment,
            params.shell_type,
        )
        .await
    }

    fn execute_cmd_stream(self: Arc<Self>, params: ShellExecParams) -> OpStream<ShellChunk> {
        Box::pin(
            futures::stream::once(async move {
                match self.execute_cmd(params).await {
                    OpResult {
                        code: 0,
                        message: _,
                        data: Some(output),
                    } => shell_chunks(&output),
                    failure => vec![OpResult {
                        code: failure.code,
                        message: failure.message,
                        data: Some(ShellChunk::terminal(0, -1)),
                    }],
                }
            })
            .flat_map(futures::stream::iter),
        )
    }

    async fn execute_cmd_background(
        &self,
        params: ShellExecParams,
    ) -> OpResult<ShellBackgroundOutput> {
        if params.command.trim().is_empty() {
            return shell_failure("execute_cmd_background", "command can not be empty");
        }
        let cwd = normalized_cwd(params.cwd.as_deref());
        let payload = json!({
            "command": params.command,
            "cwd": cwd,
            "environment": params.environment,
            "shell_type": params.shell_type.as_str(),
        });
        let data = match self
            .client
            .invoke("shell", "execute_cmd_background", payload)
            .await
        {
            Ok(data) => data,
            Err(error) => return gateway_failure("execute_cmd_background", &error),
        };
        OpResult::success(ShellBackgroundOutput {
            command: text_field(&data, "command").unwrap_or_else(|| params.command.clone()),
            cwd: text_field(&data, "cwd").unwrap_or_else(|| cwd.unwrap_or_else(|| ".".to_string())),
            pid: json_u64(data.get("pid")).and_then(|pid| u32::try_from(pid).ok()),
            error: text_field(&data, "error"),
        })
    }

    async fn execute_argv(&self, params: ShellArgvParams) -> OpResult<ShellOutput> {
        if params.argv.is_empty() {
            return shell_failure("execute_argv", "argv can not be empty");
        }
        let timeout = params.effective_timeout_secs(DEFAULT_TIMEOUT_SECS);
        let cwd = normalized_cwd(params.cwd.as_deref());
        let command = params
            .argv
            .iter()
            .map(|argument| posix_quote(argument))
            .collect::<Vec<_>>()
            .join(" ");
        self.run_command(
            "execute_argv",
            &command,
            cwd.as_deref(),
            timeout,
            &params.environment,
            ShellType::Auto,
        )
        .await
    }
}

/// Maps an `execute_cmd` payload onto the kernel's shell output.
fn shell_output(request_command: &str, cwd: Option<&str>, data: &Value) -> ShellOutput {
    let exit_code = json_i32(data.get("exit_code")).unwrap_or(0);
    ShellOutput {
        command: text_field(data, "command").unwrap_or_else(|| request_command.to_string()),
        cwd: text_field(data, "cwd").unwrap_or_else(|| cwd.unwrap_or(".").to_string()),
        exit_code,
        stdout: text_field(data, "stdout")
            .or_else(|| text_field(data, "output"))
            .unwrap_or_default(),
        stderr: text_field(data, "stderr").unwrap_or_default(),
        truncated: false,
        timed_out: exit_code == TIMEOUT_EXIT_CODE,
        cancelled: false,
        duration_ms: json_u64(data.get("duration_ms")).unwrap_or(0),
    }
}

/// Splits a completed shell run into stdout lines, stderr lines, and the
/// terminal chunk.
fn shell_chunks(output: &ShellOutput) -> Vec<OpResult<ShellChunk>> {
    let mut items = Vec::new();
    let mut index = 0_u64;
    for line in text_lines(&output.stdout) {
        items.push(OpResult::success(ShellChunk::new(
            line,
            StreamKind::Stdout,
            index,
        )));
        index += 1;
    }
    for line in text_lines(&output.stderr) {
        items.push(OpResult::success(ShellChunk::new(
            line,
            StreamKind::Stderr,
            index,
        )));
        index += 1;
    }
    items.push(OpResult::success(ShellChunk::terminal(
        index,
        output.exit_code,
    )));
    items
}

/// Normalizes the caller's working directory the way the Python providers do.
fn normalized_cwd(cwd: Option<&str>) -> Option<String> {
    cwd.filter(|value| !value.is_empty() && *value != ".")
        .map(str::to_string)
}
