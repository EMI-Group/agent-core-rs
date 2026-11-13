//! The kernel's [`CodeOperation`] over the AIO gateway envelope.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/aio.py`
//! runs the snippet in the container, reports completion as success, and fails
//! only on a timeout (`199005`). [`AioCode`] reproduces that, and maps every
//! gateway/transport failure to `199007`.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use openjiuwen_core::sys_operation::code::types::{
    CodeChunk, CodeExecParams, CodeOutput, Language,
};
use openjiuwen_core::sys_operation::operation::{CodeOperation, OpStream};
use openjiuwen_core::sys_operation::result::OpResult;
use openjiuwen_core::sys_operation::shell::types::StreamKind;
use openjiuwen_core::sys_operation::status::SysOpStatus;

use super::super::semantics::text_lines;
use super::super::transport::gateway_failure;
use super::{json_i32, json_u64, text_field, AioClient, DEFAULT_TIMEOUT_SECS, TIMEOUT_EXIT_CODE};

/// The kernel's two [`CodeOperation`] methods over the envelope.
pub(crate) struct AioCode {
    /// Shared envelope client.
    pub(super) client: Arc<AioClient>,
}

#[async_trait]
impl CodeOperation for AioCode {
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
        let timeout = params.effective_timeout_secs(DEFAULT_TIMEOUT_SECS);
        let payload = json!({
            "code": params.code,
            "language": language.as_str(),
            "timeout": timeout,
            "environment": params.environment,
            "cwd": params.cwd,
            "options": {"force_file": params.force_file},
        });
        let data = match self.client.invoke("code", "execute_code", payload).await {
            Ok(data) => data,
            Err(error) => return gateway_failure("execute_code", &error),
        };
        let output = code_output(&params.code, language, &data);
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

/// Maps an `execute_code` payload onto the kernel's code output.
fn code_output(request_code: &str, language: Language, data: &Value) -> CodeOutput {
    let exit_code = json_i32(data.get("exit_code")).unwrap_or(-1);
    let stderr = text_field(data, "stderr").unwrap_or_default();
    CodeOutput {
        code_content: text_field(data, "code_content").unwrap_or_else(|| request_code.to_string()),
        language: text_field(data, "language").unwrap_or_else(|| language.as_str().to_string()),
        exit_code,
        stdout: text_field(data, "stdout").unwrap_or_default(),
        timed_out: exit_code == TIMEOUT_EXIT_CODE
            || stderr.to_ascii_lowercase().contains("timed out"),
        stderr,
        truncated: false,
        duration_ms: json_u64(data.get("duration_ms")).unwrap_or(0),
    }
}

/// Splits a completed code run into stdout lines, stderr lines, and the
/// terminal chunk.
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
