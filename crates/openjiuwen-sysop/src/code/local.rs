//! Local code backend: snippet and script execution in a subprocess.
//!
//! Python parity: `CodeOperation`
//! (`openjiuwen/core/sys_operation/local/code_operation.py`). A short snippet is passed
//! inline (`python3 -u -c <code>` / `node -e <code>`); a long one, or one with
//! `force_file`, is written to a temporary file (`python3 -u <path>` / `node <path>`),
//! exactly like Python's `_build_subprocess_cmd`. The run uses the shared engine in
//! [`crate::shell::local::process`], so bounded capture, timeout, and cancellation behave
//! as they do for shell commands, and every failure is an [`OpResult`] value carrying
//! status `199005`, never a panic. A non-zero exit status is **not** an error (Python
//! parity): the run succeeded and reported that `exit_code`.
//!
//! ```
//! use openjiuwen_sysop::code::LocalCodeOperation;
//! use openjiuwen_sysop::config::LocalWorkConfig;
//!
//! let backend = LocalCodeOperation::new(LocalWorkConfig::default()).unwrap();
//! assert_eq!(backend.config().default_timeout_secs, 300);
//! ```
//!
//! Documented deviations from Python:
//! - Python's `"<language> is not supported"` branch is unrepresentable: [`Language`] is a
//!   closed enum, so an unsupported language cannot reach the backend.
//! - [`CodeExecParams::encoding`] is accepted for wire compatibility but ignored; both pipes
//!   are decoded as UTF-8 with replacement characters.
//! - `cwd` is **not** sandbox-checked, matching Python's `create_subprocess_exec` call.
//! - The temporary file is deleted by RAII ([`tempfile::NamedTempFile`]), which covers
//!   success, error, timeout, cancellation, and a dropped stream.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use async_trait::async_trait;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::Result;

use crate::code::types::{CodeChunk, CodeExecParams, CodeOutput, Language};
use crate::config::{EnvPolicy, LocalWorkConfig};
use crate::constants;
use crate::env::{self, ExecutionEnvironment};
use crate::operation::{CodeOperation, OpStream};
use crate::result::{code_failure, OpResult};
use crate::shell::local::process;
use crate::status::{self, SysOpStatus};

/// The local code backend: one configuration, optionally one cancellation token.
pub struct LocalCodeOperation {
    config: LocalWorkConfig,
    cancellation: Option<CancellationToken>,
}

impl LocalCodeOperation {
    /// Builds the backend; validates `config`.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::Validation` when `config` is invalid, for example when
    /// `max_output_bytes` is zero.
    pub fn new(config: LocalWorkConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            cancellation: None,
        })
    }

    /// Returns this backend with an extra per-instance cancellation token.
    ///
    /// Cancelling the token kills a running snippet; the result reports
    /// `"execution cancelled by user"`.
    #[must_use]
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    /// The configuration this backend was built from.
    #[must_use]
    pub fn config(&self) -> &LocalWorkConfig {
        &self.config
    }

    /// The timeout to apply, treating `0` as the configured default (Python `timeout or 300`).
    #[must_use]
    fn effective_timeout(&self, requested: u64) -> u64 {
        if requested == 0 {
            self.config.default_timeout_secs
        } else {
            requested
        }
    }

    /// The cancellation tokens fed to [`process::RunOptions`].
    #[must_use]
    fn cancellation_tokens(&self) -> Vec<CancellationToken> {
        self.cancellation.iter().cloned().collect()
    }
}

#[async_trait]
impl CodeOperation for LocalCodeOperation {
    async fn execute_code(&self, params: CodeExecParams) -> OpResult<CodeOutput> {
        let execution = "execute_code";
        let CodeExecParams {
            code,
            language,
            timeout_secs,
            environment,
            cwd,
            force_file,
            ..
        } = params;
        if code.trim().is_empty() {
            return code_failure(execution, "code can not be empty");
        }
        let Some(program) = resolve_runtime(execution, language) else {
            return code_failure(execution, &runtime_missing_message(language));
        };
        let cwd = match resolve_cwd(cwd.as_deref()).await {
            Ok(cwd) => cwd,
            Err(reason) => {
                tracing::warn!(execution, reason = %reason, "code working directory rejected");
                return code_failure(execution, &reason);
            }
        };
        let plan = match build_plan(program, &code, language, force_file).await {
            Ok(plan) => plan,
            Err(reason) => {
                tracing::warn!(execution, reason = %reason, "failed to prepare code execution");
                return code_failure(execution, &reason);
            }
        };
        tracing::debug!(
            execution,
            uses_file = plan.uses_file(),
            "prepared code execution"
        );
        let mut command = Command::new(&plan.program);
        command.args(&plan.args);
        if let Some(cwd) = &cwd {
            command.current_dir(cwd);
        }
        let environment = merged_environment(language, &self.config.env_policy, &environment);
        process::apply_environment(&mut command, &environment);
        let child = match process::spawn_child(command) {
            Ok(child) => child,
            Err(error) => {
                tracing::warn!(execution, %error, "failed to start code execution");
                return code_failure(
                    execution,
                    &format!("failed to start code execution: {error}"),
                );
            }
        };
        let timeout_secs = self.effective_timeout(timeout_secs);
        let options = process::RunOptions {
            timeout_secs,
            max_output_bytes: self.config.max_output_bytes,
            cancellation: self.cancellation_tokens(),
        };
        let outcome = process::run_captured(child, options).await;
        let mut output = CodeOutput {
            code_content: code,
            language: language.as_str().to_string(),
            exit_code: outcome.exit_code,
            stdout: outcome.stdout.text,
            stderr: outcome.stderr.text,
            truncated: outcome.stdout.truncated || outcome.stderr.truncated,
            timed_out: outcome.timed_out,
            duration_ms: outcome.duration_ms,
        };
        // The temporary file (if any) is removed here, before the result is returned.
        drop(plan);
        if outcome.timed_out {
            tracing::warn!(execution, timeout_secs, "code execution timed out");
            output.exit_code = -1;
            let reason = timeout_reason(timeout_secs);
            return code_failure_with_data(execution, &reason, output);
        }
        if outcome.cancelled {
            tracing::warn!(execution, "code execution cancelled");
            output.exit_code = -1;
            return code_failure_with_data(execution, "execution cancelled by user", output);
        }
        // Python parity: a non-zero exit status is a successful run with that status.
        OpResult {
            code: SysOpStatus::Success.code(),
            message: "Code executed successfully".to_string(),
            data: Some(output),
        }
    }

    fn execute_code_stream(
        self: std::sync::Arc<Self>,
        params: CodeExecParams,
    ) -> OpStream<CodeChunk> {
        Box::pin(async_stream::stream! {
            let execution = "execute_code_stream";
            let CodeExecParams {
                code,
                language,
                timeout_secs,
                environment,
                cwd,
                force_file,
                ..
            } = params;
            if code.trim().is_empty() {
                yield stream_failure(execution, "code can not be empty", 0);
                return;
            }
            let Some(program) = resolve_runtime(execution, language) else {
                let reason = runtime_missing_message(language);
                yield stream_failure(execution, &reason, 0);
                return;
            };
            let cwd = match resolve_cwd(cwd.as_deref()).await {
                Ok(cwd) => cwd,
                Err(reason) => {
                    yield stream_failure(execution, &reason, 0);
                    return;
                }
            };
            // Held for the whole stream so the temporary file outlives the child and is
            // deleted when the stream completes or is dropped.
            let plan = match build_plan(program, &code, language, force_file).await {
                Ok(plan) => plan,
                Err(reason) => {
                    yield stream_failure(execution, &reason, 0);
                    return;
                }
            };
            let mut command = Command::new(&plan.program);
            command.args(&plan.args);
            if let Some(cwd) = &cwd {
                command.current_dir(cwd);
            }
            let environment = merged_environment(language, &self.config.env_policy, &environment);
            process::apply_environment(&mut command, &environment);
            let child = match process::spawn_child(command) {
                Ok(child) => child,
                Err(error) => {
                    let reason = format!("failed to start code execution: {error}");
                    yield stream_failure(execution, &reason, 0);
                    return;
                }
            };
            let timeout_secs = self.effective_timeout(timeout_secs);
            let options = process::RunOptions {
                timeout_secs,
                max_output_bytes: self.config.max_output_bytes,
                cancellation: self.cancellation_tokens(),
            };
            let mut events = process::stream_child(child, options);
            let mut index = 0_u64;
            while let Some(event) = futures::StreamExt::next(&mut events).await {
                let item = match event {
                    process::EngineEvent::Data { kind, text } => OpResult {
                        code: SysOpStatus::Success.code(),
                        message: format!("Get {} stream successfully", kind.as_str()),
                        data: Some(CodeChunk::new(text, kind, index)),
                    },
                    process::EngineEvent::Exit { code } => OpResult {
                        code: SysOpStatus::Success.code(),
                        message: "Code executed successfully".to_string(),
                        data: Some(CodeChunk::terminal(index, code)),
                    },
                    process::EngineEvent::TimedOut { .. } => {
                        let reason = timeout_reason(timeout_secs);
                        stream_failure(execution, &reason, index)
                    }
                    process::EngineEvent::Cancelled { .. } => {
                        stream_failure(execution, "execution cancelled by user", index)
                    }
                };
                index += 1;
                yield item;
            }
        })
    }
}

/// A prepared invocation: interpreter, arguments, and the temp file holding a script.
struct CommandPlan {
    program: String,
    args: Vec<String>,
    temp_file: Option<tempfile::NamedTempFile>,
}

impl CommandPlan {
    /// `true` when the snippet runs from a temporary file rather than an inline argument.
    #[must_use]
    fn uses_file(&self) -> bool {
        self.temp_file.is_some()
    }
}

/// The platform argument-vector limit Python compares the snippet length against.
#[must_use]
fn command_line_limit() -> usize {
    if cfg!(windows) {
        constants::WINDOWS_COMMAND_LINE_LIMIT
    } else {
        constants::UNIX_COMMAND_LINE_LIMIT
    }
}

/// The flags Python's `exec_file` entries add before the path: `-u` for Python, none for
/// JavaScript. They differ from [`Language::inline_flags`], hence this table.
#[must_use]
const fn file_flags(language: Language) -> &'static [&'static str] {
    match language {
        Language::Python => &["-u"],
        Language::Javascript => &[],
    }
}

/// Python's `"<language> file not found error, ..."` reason.
///
/// The slot carries the language name (`"python"`), not the executable, and ends with a period.
#[must_use]
fn runtime_missing_message(language: Language) -> String {
    format!(
        "{} file not found error, please install and add it to your system environment variable PATH.",
        language.as_str()
    )
}

/// Python's timeout reason: `execution timeout after <n> seconds`.
#[must_use]
fn timeout_reason(timeout_secs: u64) -> String {
    format!("execution timeout after {timeout_secs} seconds")
}

/// Resolves the interpreter for `language`, logging the failure when it is absent.
fn resolve_runtime(execution: &str, language: Language) -> Option<String> {
    match which::which(language.executable()) {
        Ok(path) => Some(path.to_string_lossy().into_owned()),
        Err(error) => {
            tracing::warn!(
                execution,
                language = language.as_str(),
                %error,
                "code runtime not found on PATH"
            );
            None
        }
    }
}

/// Resolves `cwd` the way Python does: a directory or nothing, never sandbox-checked.
async fn resolve_cwd(cwd: Option<&str>) -> std::result::Result<Option<PathBuf>, String> {
    let Some(cwd) = cwd else {
        return Ok(None);
    };
    let path = PathBuf::from(cwd);
    match tokio::fs::metadata(&path).await {
        Ok(meta) if meta.is_dir() => Ok(Some(path)),
        _ => Err(format!("working directory does not exist: {cwd}")),
    }
}

/// Builds the invocation for `code`, matching Python's `_build_subprocess_cmd`.
///
/// The snippet runs inline while it fits the platform limit (character count, like Python's
/// `len(code)`) and goes to a temporary file otherwise; the write runs on the blocking pool.
async fn build_plan(
    program: String,
    code: &str,
    language: Language,
    force_file: bool,
) -> std::result::Result<CommandPlan, String> {
    if !force_file && code.chars().count() <= command_line_limit() {
        let mut args: Vec<String> = language
            .inline_flags()
            .iter()
            .map(|flag| (*flag).to_string())
            .collect();
        args.push(code.to_string());
        return Ok(CommandPlan {
            program,
            args,
            temp_file: None,
        });
    }
    let suffix = language.file_suffix();
    let body = code.to_string();
    let temp_file =
        tokio::task::spawn_blocking(move || -> std::io::Result<tempfile::NamedTempFile> {
            let mut file = tempfile::Builder::new().suffix(suffix).tempfile()?;
            file.write_all(body.as_bytes())?;
            file.flush()?;
            Ok(file)
        })
        .await
        .map_err(|error| format!("failed to create temporary file: {error}"))?
        .map_err(|error| format!("failed to create temporary file: {error}"))?;
    let path = temp_file.path().to_string_lossy().into_owned();
    let mut args: Vec<String> = file_flags(language)
        .iter()
        .map(|flag| (*flag).to_string())
        .collect();
    args.push(path);
    Ok(CommandPlan {
        program,
        args,
        temp_file: Some(temp_file),
    })
}

/// Computes the child environment, applying the language's forced variables **last**.
///
/// Precedence: [`EnvPolicy`], then the call-site overlay (`None` removes), then
/// [`Language::forced_env`], so a caller cannot disable `PYTHONUTF8=1` or
/// `NODE_DISABLE_COLORS=1`. A name the policy scrubs stays absent even when forced.
#[must_use]
fn merged_environment(
    language: Language,
    policy: &EnvPolicy,
    call: &BTreeMap<String, Option<String>>,
) -> Vec<(String, String)> {
    let overlay = ExecutionEnvironment::default();
    let mut merged: BTreeMap<String, String> =
        env::resolve_environment(policy, &overlay, Some(call))
            .into_iter()
            .collect();
    for (name, value) in language.forced_env() {
        if policy.scrubs(name) {
            continue;
        }
        merged.insert((*name).to_string(), (*value).to_string());
    }
    merged.into_iter().collect()
}

/// Builds a code failure that retains a partially filled [`CodeOutput`].
fn code_failure_with_data(
    execution: &str,
    reason: &str,
    output: CodeOutput,
) -> OpResult<CodeOutput> {
    let mut failure = code_failure::<CodeOutput>(execution, reason);
    failure.data = Some(output);
    failure
}

/// Builds the single error chunk a streamed code call yields when the run cannot start.
fn stream_failure(execution: &str, reason: &str, chunk_index: u64) -> OpResult<CodeChunk> {
    OpResult {
        code: SysOpStatus::CodeExecutionError.code(),
        message: status::code_execution_message(execution, reason),
        data: Some(CodeChunk::terminal(chunk_index, -1)),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::shell::types::StreamKind;
    use futures::StreamExt;
    use std::path::Path;

    /// `true` when a Python 3 interpreter exists; runtime tests skip otherwise.
    fn has_python() -> bool {
        which::which("python3").is_ok()
    }

    fn backend() -> LocalCodeOperation {
        LocalCodeOperation::new(LocalWorkConfig::default()).unwrap()
    }

    fn value<'a>(pairs: &'a [(String, String)], name: &str) -> Option<&'a str> {
        pairs
            .iter()
            .find(|(existing, _)| existing == name)
            .map(|(_, value)| value.as_str())
    }

    /// The extension of `path`, used to check the temporary file's name.
    fn extension(path: &str) -> Option<&str> {
        Path::new(path).extension().and_then(|ext| ext.to_str())
    }

    #[tokio::test]
    async fn empty_code_is_rejected() {
        let result = backend().execute_code(CodeExecParams::new(" \n\t")).await;
        assert_eq!(result.code, 199_005);
        assert!(
            result.message.contains("code can not be empty"),
            "{}",
            result.message
        );
        // Python's non-stream path carries no data.
        assert!(result.data.is_none());
    }

    #[test]
    fn runtime_missing_message_matches_python() {
        assert_eq!(
            runtime_missing_message(Language::Python),
            "python file not found error, please install and add it to your system environment variable PATH."
        );
        assert_eq!(
            runtime_missing_message(Language::Javascript),
            "javascript file not found error, please install and add it to your system environment variable PATH."
        );
        assert_eq!(timeout_reason(1), "execution timeout after 1 seconds");
    }

    #[tokio::test]
    async fn command_plan_switches_to_a_file() {
        let inline = build_plan("python3".to_string(), "print(1)", Language::Python, false)
            .await
            .unwrap();
        assert!(!inline.uses_file());
        assert_eq!(inline.program, "python3");
        assert_eq!(inline.args, vec!["-u", "-c", "print(1)"]);

        let js = build_plan(
            "node".to_string(),
            "console.log(1)",
            Language::Javascript,
            false,
        )
        .await
        .unwrap();
        assert_eq!(js.args, vec!["-e", "console.log(1)"]);

        let forced = build_plan("python3".to_string(), "print(1)", Language::Python, true)
            .await
            .unwrap();
        assert!(forced.uses_file());
        let forced_path = forced.args.last().unwrap().clone();
        assert_eq!(extension(&forced_path), Some("py"), "{forced_path}");
        assert_eq!(forced.args, vec!["-u".to_string(), forced_path.clone()]);
        let temp_path = PathBuf::from(&forced_path);
        assert!(temp_path.exists(), "the snippet must be on disk");
        drop(forced);
        assert!(
            !Path::new(&temp_path).exists(),
            "dropping the plan deletes {forced_path}"
        );

        let limit = command_line_limit();
        let at_limit = "x".repeat(limit);
        let plan = build_plan("python3".to_string(), &at_limit, Language::Python, false)
            .await
            .unwrap();
        assert!(!plan.uses_file(), "the limit itself still runs inline");

        let over_limit = "x".repeat(limit + 1);
        let plan = build_plan("node".to_string(), &over_limit, Language::Javascript, false)
            .await
            .unwrap();
        assert!(plan.uses_file());
        let path = plan.args.last().unwrap();
        assert_eq!(plan.args.len(), 1, "node takes no file flag");
        assert_eq!(extension(path), Some("js"), "{path}");
    }

    #[test]
    fn forced_environment_wins_over_caller_values() {
        let mut call = BTreeMap::new();
        call.insert("PYTHONUTF8".to_string(), Some("0".to_string()));
        call.insert("NODE_DISABLE_COLORS".to_string(), Some("0".to_string()));
        let policy = EnvPolicy::default();

        let python = merged_environment(Language::Python, &policy, &call);
        assert_eq!(value(&python, "PYTHONUTF8"), Some("1"));
        assert_eq!(value(&python, "PYTHONIOENCODING"), Some("utf-8"));

        let js = merged_environment(Language::Javascript, &policy, &call);
        assert_eq!(value(&js, "NODE_DISABLE_COLORS"), Some("1"));
        assert_eq!(
            value(&js, "PYTHONUTF8"),
            Some("0"),
            "the caller's value survives"
        );

        let scrubbed = EnvPolicy {
            scrub: vec!["PYTHONUTF8".to_string()],
            ..EnvPolicy::default()
        };
        let python = merged_environment(Language::Python, &scrubbed, &call);
        assert_eq!(
            value(&python, "PYTHONUTF8"),
            None,
            "scrub wins over forcing"
        );
        assert_eq!(value(&python, "PYTHONIOENCODING"), Some("utf-8"));
    }

    #[tokio::test]
    async fn python_prints_stdout() {
        // Skips on hosts without python3 (see the test strategy in CONTEXT.md).
        if !has_python() {
            return;
        }
        let result = backend()
            .execute_code(CodeExecParams::new("print('hello from python')"))
            .await;
        assert_eq!(result.code, 0, "{}", result.message);
        assert_eq!(result.message, "Code executed successfully");
        let output = result.data.unwrap();
        assert_eq!(output.exit_code, 0);
        assert!(
            output.stdout.contains("hello from python"),
            "{}",
            output.stdout
        );
        assert!(output.stderr.is_empty(), "{}", output.stderr);
        assert!(!output.timed_out && !output.truncated);
        assert_eq!(output.language, "python");
        assert_eq!(output.code_content, "print('hello from python')");
    }

    #[tokio::test]
    async fn non_zero_exit_is_not_an_error() {
        if !has_python() {
            return;
        }
        let result = backend()
            .execute_code(CodeExecParams::new("raise SystemExit(3)"))
            .await;
        assert_eq!(result.code, 0, "{}", result.message);
        assert_eq!(result.message, "Code executed successfully");
        assert_eq!(result.data.unwrap().exit_code, 3);
    }

    #[tokio::test]
    async fn timeout_is_reported() {
        if !has_python() {
            return;
        }
        let params = CodeExecParams::new("import time; time.sleep(5)").with_timeout_secs(1);
        let result = backend().execute_code(params).await;
        assert_eq!(result.code, 199_005);
        assert!(result.message.contains("timeout"), "{}", result.message);
        let output = result.data.unwrap();
        assert!(output.timed_out);
        assert_eq!(output.exit_code, -1);
    }

    #[tokio::test]
    async fn cwd_is_honoured() {
        if !has_python() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let name = dir
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let params = CodeExecParams::new("import os; print(os.getcwd())")
            .with_cwd(dir.path().to_string_lossy().into_owned());
        let output = backend().execute_code(params).await.data.unwrap();
        assert!(
            output.stdout.trim().ends_with(&name),
            "{} does not end with {name}",
            output.stdout
        );

        let missing = dir.path().join("does-not-exist");
        let params =
            CodeExecParams::new("print(1)").with_cwd(missing.to_string_lossy().into_owned());
        let result = backend().execute_code(params).await;
        assert_eq!(result.code, 199_005);
        assert!(
            result.message.contains("working directory does not exist"),
            "{}",
            result.message
        );
    }

    #[tokio::test]
    async fn stream_yields_chunks_then_terminal() {
        if !has_python() {
            return;
        }
        let backend = std::sync::Arc::new(backend());
        let items: Vec<_> = backend
            .execute_code_stream(CodeExecParams::new("print('streamed')"))
            .collect()
            .await;
        assert!(items.iter().all(|item| item.code == 0));
        let stdout: String = items
            .iter()
            .filter_map(|item| item.data.as_ref())
            .filter(|data| data.stream == Some(StreamKind::Stdout))
            .map(|data| data.text.as_str())
            .collect();
        assert!(stdout.contains("streamed"), "{stdout}");
        let last = items.last().unwrap().data.as_ref().unwrap();
        assert_eq!(last.exit_code, Some(0));
        let indices: Vec<u64> = items
            .iter()
            .filter_map(|item| item.data.as_ref().map(|data| data.chunk_index))
            .collect();
        assert!(
            indices.windows(2).all(|pair| pair[0] < pair[1]),
            "{indices:?}"
        );
    }

    #[tokio::test]
    async fn stream_empty_code_yields_one_error_chunk() {
        let backend = std::sync::Arc::new(backend());
        let items: Vec<_> = backend
            .execute_code_stream(CodeExecParams::new("  "))
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].code, 199_005);
        assert!(
            items[0].message.contains("code can not be empty"),
            "{}",
            items[0].message
        );
        let data = items[0].data.as_ref().unwrap();
        assert_eq!((data.chunk_index, data.exit_code), (0, Some(-1)));
    }
}
