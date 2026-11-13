//! The kernel's [`ShellOperation`] implemented once over a [`SandboxBackend`].
//!
//! Python parity: the providers under
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` all report completion as
//! success (a non-zero exit code is data, not an error) and turn only a container-side
//! timeout into `199004`. [`RemoteShell`] reproduces that on top of
//! [`SandboxBackend::execute`] / [`SandboxBackend::spawn_background`].

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use openjiuwen_core::sys_operation::operation::{OpStream, ShellOperation};
use openjiuwen_core::sys_operation::result::{shell_failure, OpResult};
use openjiuwen_core::sys_operation::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput, StreamKind,
};
use openjiuwen_core::sys_operation::status::SysOpStatus;

use super::super::semantics::text_lines;
use super::{elapsed_ms, ExecCommand, ExecRequest, SandboxBackend};

/// The kernel's four [`ShellOperation`] methods implemented over a [`SandboxBackend`].
pub(crate) struct RemoteShell<B: SandboxBackend> {
    backend: Arc<B>,
}

impl<B: SandboxBackend> RemoteShell<B> {
    /// Builds the wrapper.
    #[must_use]
    pub(crate) fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl<B: SandboxBackend> ShellOperation for RemoteShell<B> {
    async fn execute_cmd(&self, params: ShellExecParams) -> OpResult<ShellOutput> {
        if params.command.trim().is_empty() {
            return shell_failure("execute_cmd", "command can not be empty");
        }
        let timeout = params.effective_timeout_secs(300);
        let cwd = normalized_cwd(params.cwd.as_deref());
        let request = ExecRequest {
            cwd: cwd.clone(),
            environment: params.environment.clone(),
            timeout_secs: Some(timeout),
            ..ExecRequest::new(ExecCommand::shell(params.command.clone()))
        };
        let started = Instant::now();
        let result = match self.backend.execute(&request).await {
            Ok(result) => result,
            Err(error) => return shell_failure("execute_cmd", &error.message()),
        };
        let output = ShellOutput {
            command: params.command,
            cwd: cwd.unwrap_or_else(|| ".".to_string()),
            exit_code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
            truncated: false,
            timed_out: result.timed_out,
            cancelled: false,
            duration_ms: elapsed_ms(started),
        };
        if output.timed_out {
            return OpResult {
                code: SysOpStatus::ShellExecutionError.code(),
                message: format!("execution timeout after {timeout} seconds"),
                data: Some(output),
            };
        }
        OpResult::success(output)
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
        let timeout = params.effective_timeout_secs(300);
        let cwd = normalized_cwd(params.cwd.as_deref());
        let request = ExecRequest {
            cwd: cwd.clone(),
            environment: params.environment,
            timeout_secs: Some(timeout),
            ..ExecRequest::new(ExecCommand::shell(params.command.clone()))
        };
        match self.backend.spawn_background(&request).await {
            Ok(pid) => OpResult::success(ShellBackgroundOutput {
                command: params.command,
                cwd: cwd.unwrap_or_else(|| ".".to_string()),
                pid: Some(pid),
                error: None,
            }),
            Err(error) => shell_failure("execute_cmd_background", &error.message()),
        }
    }

    async fn execute_argv(&self, params: ShellArgvParams) -> OpResult<ShellOutput> {
        if params.argv.is_empty() {
            return shell_failure("execute_argv", "argv can not be empty");
        }
        let timeout = params.effective_timeout_secs(300);
        let cwd = normalized_cwd(params.cwd.as_deref());
        let command = ExecCommand::argv(params.argv);
        let command_line = command.to_shell_command();
        let request = ExecRequest {
            cwd: cwd.clone(),
            environment: params.environment,
            timeout_secs: Some(timeout),
            ..ExecRequest::new(command)
        };
        let started = Instant::now();
        let result = match self.backend.execute(&request).await {
            Ok(result) => result,
            Err(error) => return shell_failure("execute_argv", &error.message()),
        };
        let output = ShellOutput {
            command: command_line,
            cwd: cwd.unwrap_or_else(|| ".".to_string()),
            exit_code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
            truncated: false,
            timed_out: result.timed_out,
            cancelled: false,
            duration_ms: elapsed_ms(started),
        };
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

/// Splits a completed shell run into stdout lines, stderr lines, and the terminal chunk.
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
