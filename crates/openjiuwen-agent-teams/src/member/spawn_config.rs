//! Spawn-mode selection and the per-kind launch configuration for teammate members.
//!
//! [`MemberSpawnKind`] is the selector and [`MemberSpawnOptions`] bundles it with the launch
//! configuration for that kind (an OS process, an external CLI, an SSH endpoint).
//! A mode travels to a spawner on [`crate::member::spawn::MemberSpawnRequest::options`], so one
//! request shape carries both the in-process case and the child-process cases.
//! Only [`MemberSpawnKind::InProcess`] needs no child process: the member's runtime is built
//! in-process by the harness factory and the request's options are ignored.
//! The three child-process kinds ([`MemberSpawnKind::Process`], [`MemberSpawnKind::ExternalCli`],
//! [`MemberSpawnKind::Ssh`]) all share the JSON-lines member protocol in `crate::member::protocol`
//! over the child's stdin/stdout, so only the way the child is launched differs.
//!
//! Python parity sources: `openjiuwen/agent_teams/spawn/process_spawn.py`,
//! `spawn/external_cli_spawn.py` and `schema/ssh_transport.py`. The SSH field names follow
//! Python's `SshTransportConfig` (documents and wire payloads are shared); `program`, `options`
//! and `remote_command` are Rust additions, and Python's `asyncssh` client is replaced by the
//! local `ssh` binary.
//!
//! ```
//! use openjiuwen_agent_teams::member::spawn_config::{
//!     MemberSpawnKind, MemberSpawnOptions, StdioCommand,
//! };
//!
//! let options = MemberSpawnOptions::from_kind(MemberSpawnKind::Ssh);
//! assert!(!options.is_in_process());
//! assert_eq!(MemberSpawnOptions::default().kind, MemberSpawnKind::InProcess);
//! let command = StdioCommand::new("ssh").with_arg("-V");
//! assert_eq!(command.to_argv(), vec!["ssh".to_string(), "-V".to_string()]);
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// Where a teammate runs.
///
/// Wire values are `inprocess`, `process`, `external_cli` and `ssh`; the `inprocess` spelling
/// matches [`crate::schema::SpawnMode`] so a spec and a spawn request agree on the JSON name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberSpawnKind {
    /// The member runs as a task inside the leader's process; the Python default.
    #[default]
    #[serde(rename = "inprocess")]
    InProcess,
    /// The member runs as a child process speaking the member protocol (Python `process`).
    Process,
    /// The member is an external CLI agent (for example `claude` or `codex`) run as a child.
    ExternalCli,
    /// The member runs on a remote host reached through the `ssh` binary.
    Ssh,
}

impl MemberSpawnKind {
    /// Returns the wire value of the kind (`inprocess`, `process`, `external_cli`, `ssh`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InProcess => "inprocess",
            Self::Process => "process",
            Self::ExternalCli => "external_cli",
            Self::Ssh => "ssh",
        }
    }

    /// Returns `true` for [`Self::InProcess`], the only kind that needs no child process.
    #[must_use]
    pub fn is_in_process(&self) -> bool {
        matches!(self, Self::InProcess)
    }
}

impl From<crate::schema::SpawnMode> for MemberSpawnKind {
    fn from(mode: crate::schema::SpawnMode) -> Self {
        match mode {
            crate::schema::SpawnMode::InProcess => Self::InProcess,
            crate::schema::SpawnMode::Process => Self::Process,
        }
    }
}

/// A child command line: program, arguments, extra environment and working directory.
///
/// The command is data, not a process: [`Self::to_argv`] renders it for logging and tests and
/// [`Self::to_command`] turns it into a [`tokio::process::Command`] once the program has been
/// resolved.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StdioCommand {
    /// Program to run: either a bare name looked up on `PATH` or a path to an executable.
    pub program: String,
    /// Arguments passed to the program, in order.
    pub args: Vec<String>,
    /// Extra environment variables layered over the inherited environment.
    pub env: BTreeMap<String, String>,
    /// Working directory for the child; the parent's directory when unset.
    pub cwd: Option<PathBuf>,
}

impl StdioCommand {
    /// Creates a command for `program` with no arguments.
    #[must_use]
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            ..Self::default()
        }
    }

    /// Appends one argument.
    #[must_use]
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Appends every argument of `args`, in order.
    #[must_use]
    pub fn with_args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Adds one environment variable, keeping the inherited environment.
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Sets the child's working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Returns the configured program.
    #[must_use]
    pub fn program(&self) -> &str {
        &self.program
    }

    /// Returns the configured arguments.
    #[must_use]
    pub fn args(&self) -> &[String] {
        &self.args
    }

    /// Returns the extra environment variables (additive over the inherited environment).
    #[must_use]
    pub fn env(&self) -> &BTreeMap<String, String> {
        &self.env
    }

    /// Returns the configured working directory.
    #[must_use]
    pub fn cwd(&self) -> Option<&Path> {
        self.cwd.as_deref()
    }

    /// Resolves the program to an executable path.
    ///
    /// A program containing a path separator is used as given and must name an existing file; a
    /// bare name is looked up on `PATH` with `which`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when no program is configured, when a configured path is not an
    /// existing file, or when a bare name is not on `PATH`.
    pub fn resolve_program(&self) -> Result<PathBuf> {
        if self.program.trim().is_empty() {
            return Err(Error::config("spawn command has no program configured"));
        }
        if self.program.contains(std::path::MAIN_SEPARATOR) || self.program.contains('/') {
            let path = PathBuf::from(&self.program);
            if path.is_file() {
                return Ok(path);
            }
            return Err(Error::config(format!(
                "spawn program '{}' is not an existing file; install it or fix the configured program",
                path.display()
            )));
        }
        which::which(&self.program).map_err(|_| {
            Error::config(format!(
                "`{}` was not found on PATH; install it, pass an explicit program path, or set the spawn command in the member spec",
                self.program
            ))
        })
    }

    /// Renders the command line as `[program, args...]` without resolving the program.
    ///
    /// This never runs `which`, so it is safe for logging and for asserting argv shapes in tests.
    #[must_use]
    pub fn to_argv(&self) -> Vec<String> {
        let mut argv = Vec::with_capacity(self.args.len() + 1);
        argv.push(self.program.clone());
        argv.extend(self.args.iter().cloned());
        argv
    }

    /// Builds a piped child command from this configuration.
    ///
    /// The child's stdin, stdout and stderr are piped (the member protocol is carried over
    /// stdin/stdout) and `kill_on_drop` is set so a dropped handle does not leave an orphan.
    /// `env` is additive: it never clears the inherited environment, which is how credentials
    /// reach the child.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::resolve_program`] failures.
    pub fn to_command(&self) -> Result<Command> {
        let program = self.resolve_program()?;
        let mut cmd = Command::new(program);
        cmd.args(&self.args)
            .envs(&self.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = &self.cwd {
            cmd.current_dir(cwd);
        }
        Ok(cmd)
    }
}

/// Launch configuration for an OS-process member ([`MemberSpawnKind::Process`]).
///
/// The child is expected to speak the `crate::member::protocol` JSON-lines member protocol on
/// stdin/stdout. The spawner passes team/member/session identity through the
/// `OPENJIUWEN_TEAM_NAME`, `OPENJIUWEN_MEMBER_NAME`, `OPENJIUWEN_SESSION_ID` and
/// `OPENJIUWEN_MEMBER_WORKSPACE` environment variables; a spec may pre-set or override them
/// through [`StdioCommand::env`].
/// The default command is `team-member serve`; the `serve` subcommand of the `team-member` binary
/// is wired at the crate root, not by the member node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ProcessMemberConfig {
    /// Command line used to launch the member process.
    pub command: StdioCommand,
}

impl Default for ProcessMemberConfig {
    fn default() -> Self {
        Self {
            command: StdioCommand::new("team-member").with_arg("serve"),
        }
    }
}

/// Launch configuration for an external CLI member ([`MemberSpawnKind::ExternalCli`]).
///
/// Documented deviation from Python: `cli_agent` is a free-form label (Python
/// `ExternalCliAgentSpec.cli_agent`, for example `claude` or `codex`) and provider/model
/// negotiation through Python's `external/cli_agent/model_compat.py` is out of scope here.
/// The child CLI is expected to be configured through its own environment; credentials reach it
/// through the inherited environment plus [`StdioCommand::env`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExternalCliRuntimeConfig {
    /// Command line used to launch the external CLI.
    pub command: StdioCommand,
    /// Free-form CLI label (`claude`, `codex`, …); never interpreted by this crate.
    pub cli_agent: Option<String>,
    /// Provider name recorded for diagnostics only.
    pub provider: Option<String>,
    /// Model name recorded for diagnostics only.
    pub model: Option<String>,
}

/// Connection details for one reachable SSH endpoint ([`MemberSpawnKind::Ssh`]).
///
/// Field names deliberately mirror Python
/// `openjiuwen/agent_teams/schema/ssh_transport.py::SshTransportConfig` (`host`, `port`,
/// `username`, `key_file`, `password`, `agent`, `known_hosts`, `disable_host_key_check`,
/// `connect_timeout_s`, `use_exec`); `program`, `options` and `remote_command` are the Rust
/// additions.
/// Unlike Python's `asyncssh` client, this crate renders an argv for the local `ssh` binary, so
/// authentication must be expressible on that command line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SshTransportSpec {
    /// SSH client program; a bare name resolved on `PATH`, default `ssh`.
    pub program: String,
    /// Hostname or IP of the endpoint; required by [`Self::validate`].
    pub host: String,
    /// SSH port, passed as `-p`; default `22`.
    pub port: u16,
    /// Login user (Python `username`); omitted from the target when unset.
    pub username: Option<String>,
    /// Private key file (Python `key_file`), passed as `-i`.
    pub key_file: Option<PathBuf>,
    /// Password (Python `password`).
    ///
    /// Accepted for parity but never placed on the argv: the `ssh` client has no command-line
    /// password, so a password-only spec requires an `SSH_ASKPASS` helper in the environment.
    /// The value is otherwise unused and is never logged.
    pub password: Option<String>,
    /// Use the local ssh-agent (Python `agent`); adds no flag, the client already honours
    /// `SSH_AUTH_SOCK`.
    pub agent: bool,
    /// Optional known-hosts file (Python `known_hosts`), passed as
    /// `-o UserKnownHostsFile=<path>`.
    pub known_hosts: Option<PathBuf>,
    /// Skip host-key verification (Python `disable_host_key_check`).
    ///
    /// Host-key verification flags are never injected unless this is explicitly `true`.
    pub disable_host_key_check: bool,
    /// Connect timeout in seconds (Python `connect_timeout_s`), passed as
    /// `-o ConnectTimeout=<seconds>`.
    pub connect_timeout_s: Option<f64>,
    /// Prefix the remote command with `exec` (Python `use_exec`) so signals and the exit status
    /// pass through the remote shell.
    pub use_exec: bool,
    /// Extra `ssh -o Key=Value` options, rendered sorted by key.
    pub options: BTreeMap<String, String>,
    /// The member command line to run on the remote host.
    pub remote_command: String,
}

impl Default for SshTransportSpec {
    fn default() -> Self {
        Self {
            program: "ssh".to_string(),
            host: String::new(),
            port: 22,
            username: None,
            key_file: None,
            password: None,
            agent: false,
            known_hosts: None,
            disable_host_key_check: false,
            connect_timeout_s: Some(15.0),
            use_exec: true,
            options: BTreeMap::new(),
            remote_command: String::new(),
        }
    }
}

impl SshTransportSpec {
    /// Returns the remote target word: `user@host` when a username is set, else `host`.
    #[must_use]
    pub fn remote_target(&self) -> String {
        match &self.username {
            Some(username) if !username.is_empty() => format!("{username}@{}", self.host),
            _ => self.host.clone(),
        }
    }

    /// Renders the remote command as one POSIX shell word.
    ///
    /// The word is single-quoted and every embedded `'` is escaped as `'\''`, so the remote shell
    /// receives the command verbatim.
    /// When [`Self::use_exec`] is `true` the word carries an `exec ` prefix, which replaces the
    /// remote shell with the member process so signals and the exit status pass through (Python
    /// `use_exec`); when `false` the plain command is sent.
    #[must_use]
    pub fn quoted_remote_command(&self) -> String {
        let command = if self.use_exec {
            format!("exec {}", self.remote_command)
        } else {
            self.remote_command.clone()
        };
        quote_shell_word(&command)
    }

    /// Returns the exact `ssh` argv, without rendering the remote command twice.
    fn ssh_args(&self) -> Vec<String> {
        let mut args: Vec<String> = vec!["-p".to_string(), self.port.to_string()];
        if let Some(key_file) = &self.key_file {
            args.push("-i".to_string());
            args.push(key_file.display().to_string());
        }
        if let Some(seconds) = self.connect_timeout_s {
            args.push("-o".to_string());
            args.push(format!("ConnectTimeout={seconds}"));
        }
        if let Some(known_hosts) = &self.known_hosts {
            args.push("-o".to_string());
            args.push(format!("UserKnownHostsFile={}", known_hosts.display()));
        }
        if self.disable_host_key_check {
            args.push("-o".to_string());
            args.push("StrictHostKeyChecking=no".to_string());
        }
        for (key, value) in &self.options {
            args.push("-o".to_string());
            args.push(format!("{key}={value}"));
        }
        args.push("--".to_string());
        args.push(self.remote_target());
        args.push(self.quoted_remote_command());
        args
    }

    /// Returns the full `ssh` argv: [`Self::program`] followed by the options, `--`, the remote
    /// target and the single quoted remote command.
    ///
    /// `-p` is always emitted so the command line is explicit; `-i` and `-o ConnectTimeout=` only
    /// when configured; `-o StrictHostKeyChecking=no` only when
    /// [`Self::disable_host_key_check`] is `true`; one `-o Key=Value` per [`Self::options`] entry
    /// in sorted order. [`Self::agent`] adds no flag (the client already uses `SSH_AUTH_SOCK`) and
    /// [`Self::password`] never appears.
    #[must_use]
    pub fn to_argv(&self) -> Vec<String> {
        let mut argv = Vec::with_capacity(self.options.len() + 8);
        argv.push(self.program.clone());
        argv.extend(self.ssh_args());
        argv
    }

    /// Checks the fields an argv cannot do without.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] for an empty host and for an empty remote command, and
    /// [`Error::validation`] for port `0`.
    pub fn validate(&self) -> Result<()> {
        if self.host.trim().is_empty() {
            return Err(Error::config("ssh transport has no host configured"));
        }
        if self.port == 0 {
            return Err(Error::validation(
                "ssh transport port must be between 1 and 65535",
            ));
        }
        if self.remote_command.trim().is_empty() {
            return Err(Error::config(
                "ssh transport has no remote command configured",
            ));
        }
        Ok(())
    }

    /// Validates the spec and renders it as a [`StdioCommand`] running the local `ssh` client.
    ///
    /// The returned command inherits the parent environment only (`env` is empty) and runs in the
    /// parent's directory. Host-key verification flags are never injected unless
    /// [`Self::disable_host_key_check`] is explicitly `true`.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::validate`] failures.
    pub fn to_command(&self) -> Result<StdioCommand> {
        self.validate()?;
        Ok(StdioCommand {
            program: self.program.clone(),
            args: self.ssh_args(),
            env: BTreeMap::new(),
            cwd: None,
        })
    }
}

/// Wraps `value` in single quotes, escaping embedded quotes as `'\''`.
fn quote_shell_word(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('\'');
    for ch in value.chars() {
        if ch == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(ch);
        }
    }
    quoted.push('\'');
    quoted
}

/// How one member should be launched, plus the per-kind configuration for that launch.
///
/// One option object carries the mode and the launch details; only the kind matching
/// [`Self::kind`] is read by the matching spawner, so an OS-process config is harmless on an SSH
/// request.
/// An in-process team ignores the whole object, which is the default, so every spec that carries
/// no spawn options keeps working unchanged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberSpawnOptions {
    /// Where the member runs.
    pub kind: MemberSpawnKind,
    /// Launch configuration for [`MemberSpawnKind::Process`].
    pub process: Option<ProcessMemberConfig>,
    /// Launch configuration for [`MemberSpawnKind::ExternalCli`].
    pub external_cli: Option<ExternalCliRuntimeConfig>,
    /// Launch configuration for [`MemberSpawnKind::Ssh`].
    pub ssh: Option<SshTransportSpec>,
}

impl MemberSpawnOptions {
    /// Creates options selecting `kind` with no per-kind configuration yet.
    #[must_use]
    pub fn from_kind(kind: MemberSpawnKind) -> Self {
        Self {
            kind,
            ..Self::default()
        }
    }

    /// Attaches the OS-process launch configuration.
    #[must_use]
    pub fn with_process(mut self, process: ProcessMemberConfig) -> Self {
        self.process = Some(process);
        self
    }

    /// Attaches the external-CLI launch configuration.
    #[must_use]
    pub fn with_external_cli(mut self, external_cli: ExternalCliRuntimeConfig) -> Self {
        self.external_cli = Some(external_cli);
        self
    }

    /// Attaches the SSH launch configuration.
    #[must_use]
    pub fn with_ssh(mut self, ssh: SshTransportSpec) -> Self {
        self.ssh = Some(ssh);
        self
    }

    /// Returns `true` when the member runs in-process and the launch configuration is unused.
    #[must_use]
    pub fn is_in_process(&self) -> bool {
        self.kind.is_in_process()
    }
}
