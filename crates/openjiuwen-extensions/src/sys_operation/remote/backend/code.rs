//! The kernel's [`CodeOperation`] implemented once over a [`SandboxBackend`].
//!
//! Python parity: the providers under
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` run a snippet through the
//! language's inline form (`python3 -u -c`, `node -e`) or a temporary file, report
//! completion as success, and turn only a timeout into `199005`. [`RemoteCode`]
//! reproduces that on top of [`SandboxBackend::code_command`] /
//! [`SandboxBackend::code_cwd`] / [`SandboxBackend::execute`].

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use openjiuwen_core::sys_operation::code::types::{CodeChunk, CodeExecParams, CodeOutput};
use openjiuwen_core::sys_operation::operation::{CodeOperation, OpStream};
use openjiuwen_core::sys_operation::result::{code_failure, OpResult};
use openjiuwen_core::sys_operation::shell::types::StreamKind;
use openjiuwen_core::sys_operation::status::SysOpStatus;

use super::super::semantics::text_lines;
use super::{elapsed_ms, ExecRequest, SandboxBackend};

/// The kernel's two [`CodeOperation`] methods implemented over a [`SandboxBackend`].
pub(crate) struct RemoteCode<B: SandboxBackend> {
    backend: Arc<B>,
}

impl<B: SandboxBackend> RemoteCode<B> {
    /// Builds the wrapper.
    #[must_use]
    pub(crate) fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl<B: SandboxBackend> CodeOperation for RemoteCode<B> {
    async fn execute_code(&self, params: CodeExecParams) -> OpResult<CodeOutput> {
        let language = params.language;
        if params.code.trim().is_empty() {
            return OpResult {
                code: SysOpStatus::CodeExecutionError.code(),
                message: "code can not be empty".to_string(),
                data: Some(CodeOutput {
                    code_content: params.code,
                    language: language.as_str().to_string(),
                    exit_code: -1,
                    ..CodeOutput::default()
                }),
            };
        }
        let timeout = params.effective_timeout_secs(300);
        let mut environment: BTreeMap<String, Option<String>> = language
            .forced_env()
            .iter()
            .map(|(name, value)| ((*name).to_string(), Some((*value).to_string())))
            .collect();
        for (name, value) in params.environment {
            environment.insert(name, value);
        }
        let command = self
            .backend
            .code_command(&params.code, language, params.force_file);
        let request = ExecRequest {
            cwd: Some(self.backend.code_cwd(params.cwd.as_deref())),
            environment,
            timeout_secs: Some(timeout),
            ..ExecRequest::new(command)
        };
        let started = Instant::now();
        let result = match self.backend.execute(&request).await {
            Ok(result) => result,
            Err(error) => return code_failure("execute_code", &error.message()),
        };
        let output = CodeOutput {
            code_content: params.code,
            language: language.as_str().to_string(),
            exit_code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
            truncated: false,
            timed_out: result.timed_out,
            duration_ms: elapsed_ms(started),
        };
        if output.timed_out {
            return OpResult {
                code: SysOpStatus::CodeExecutionError.code(),
                message: format!("execution timeout after {timeout} seconds"),
                data: Some(output),
            };
        }
        OpResult::success(output)
    }

    fn execute_code_stream(self: Arc<Self>, params: CodeExecParams) -> OpStream<CodeChunk> {
        Box::pin(
            futures::stream::once(async move {
                match self.execute_code(params).await {
                    OpResult {
                        code: 0,
                        message: _,
                        data: Some(output),
                    } => code_chunks(&output),
                    failure => vec![OpResult {
                        code: failure.code,
                        message: failure.message,
                        data: Some(CodeChunk::terminal(0, -1)),
                    }],
                }
            })
            .flat_map(futures::stream::iter),
        )
    }
}

/// Splits a completed code run into stdout lines, stderr lines, and the terminal chunk.
fn code_chunks(output: &CodeOutput) -> Vec<OpResult<CodeChunk>> {
    let mut items = Vec::new();
    let mut index = 0_u64;
    for line in text_lines(&output.stdout) {
        items.push(OpResult::success(CodeChunk::new(
            line,
            StreamKind::Stdout,
            index,
        )));
        index += 1;
    }
    for line in text_lines(&output.stderr) {
        items.push(OpResult::success(CodeChunk::new(
            line,
            StreamKind::Stderr,
            index,
        )));
        index += 1;
    }
    items.push(OpResult::success(CodeChunk::terminal(
        index,
        output.exit_code,
    )));
    items
}
