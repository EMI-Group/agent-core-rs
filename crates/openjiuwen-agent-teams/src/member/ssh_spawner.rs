//! SSH-transport member spawner: run a teammate on a remote host through the `ssh` binary.
//!
//! Python parity sources: `openjiuwen/agent_teams/schema/ssh_transport.py`
//! (`SshTransportConfig`) and `external/cli_agent/claude/ssh_transport.py`.
//! Python opens the connection with `asyncssh`; this port deliberately shells out to the local
//! `ssh` client instead, because it needs no async-SSH dependency and inherits the user's own
//! client configuration.
//! SSH-based *attach* to an existing team (cross-process attach over SSH, a remote team store) and
//! vendor CLI SDKs remain deferred; this spawner only starts a member on a remote host.
//!
//! # Identity across the pipe
//!
//! The JSON-lines member protocol's `start` frame carries only `session_id`, so the team name, the
//! member name and the workspace root would be lost on the way to the remote process.
//! `sshd` drops channel environment unless the server allows it with `AcceptEnv`, so `SendEnv` is
//! not a reliable carrier either.
//! The identity therefore travels inside the remote command as a POSIX `export` prologue:
//! `export OPENJIUWEN_TEAM_NAME='<team>'; export OPENJIUWEN_MEMBER_NAME='<member>'; ` +
//! `export OPENJIUWEN_SESSION_ID='<session>'; export OPENJIUWEN_MEMBER_WORKSPACE='<workspace>'; `
//! followed by the configured remote command.
//! This mirrors Python's `export OPENJIUWEN_TEAM_JOIN=...` prefix in its SSH transport.
//! The local environment stays untouched (the request's identity environment is not applied): the
//! local child is `ssh` itself and the values belong on the far side.
//!
//! # Quoting
//!
//! `ssh` concatenates its trailing arguments into one string that the remote login shell parses
//! again, so the remote command is passed as a single argv element that
//! [`SshTransportSpec::quoted_remote_command`] wraps in single quotes.
//! The exported values are single-quoted the same way, with every embedded `'` escaped as `'\''`.
//! A spec must not rely on local shell expansion: nothing between the local `ssh` argv and the
//! remote shell is expanded by the local shell.
//!
//! # No injected flags
//!
//! The spawner adds no `ssh` flag of its own — no `SendEnv`, `BatchMode` or
//! `StrictHostKeyChecking` beyond what the spec asks for — so host-key verification, known hosts
//! and timeouts come only from [`SshTransportSpec`].
//! [`SshTransportSpec::password`] never appears on the argv (the `ssh` client has no command-line
//! password); a password-only spec needs an `SSH_ASKPASS` helper in the environment.
//!
//! ```no_run
//! use openjiuwen_agent_teams::member::spawn::{MemberSpawnRequest, MemberSpawner as _};
//! use openjiuwen_agent_teams::member::spawn_config::SshTransportSpec;
//! use openjiuwen_agent_teams::member::ssh_spawner::SshMemberSpawner;
//!
//! # async fn demo() -> openjiuwen_common::error::Result<()> {
//! let spawner = SshMemberSpawner::new(SshTransportSpec {
//!     host: "build-box".to_string(),
//!     remote_command: "team-member serve".to_string(),
//!     ..SshTransportSpec::default()
//! });
//! let request = MemberSpawnRequest::new("alpha", "dev-1", "team-session", "/tmp/dev-1");
//! let runtime = spawner.spawn(request).await?;
//! runtime.stop().await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use tracing::debug;

use super::runtime::MemberRuntime;
use super::spawn::{
    MemberSpawnRequest, MemberSpawner, MEMBER_NAME_ENV, MEMBER_WORKSPACE_ENV, SESSION_ID_ENV,
    TEAM_NAME_ENV,
};
use super::spawn_config::{MemberSpawnKind, SshTransportSpec, StdioCommand};
use super::stdio::StdioMemberRuntime;

/// Spawns teammates on a remote host through the `ssh` client.
///
/// The spawner owns one [`SshTransportSpec`] and the live [`StdioMemberRuntime`] handles of the
/// members it started, keyed by member name.
/// A request may carry its own SSH spec on
/// [`MemberSpawnOptions::ssh`](crate::member::spawn_config::MemberSpawnOptions::ssh) together with
/// `kind: ssh`; that spec then wins over the spawner's spec, so one team can reach different hosts.
/// A request of any other kind falls back to the spawner's spec, which keeps a `process` or
/// `external_cli` request from silently picking up SSH connection details.
/// The struct is `Send + Sync` (a `parking_lot::Mutex` guards the registry); the mutex is never
/// held across an `.await`.
#[derive(Debug)]
pub struct SshMemberSpawner {
    spec: SshTransportSpec,
    live: Mutex<HashMap<String, Arc<StdioMemberRuntime>>>,
}

impl SshMemberSpawner {
    /// Creates a spawner for `spec` with no live members.
    #[must_use]
    pub fn new(spec: SshTransportSpec) -> Self {
        Self {
            spec,
            live: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the spawner's own SSH configuration.
    ///
    /// A request-level `kind: ssh` spec overrides this for that request only; the stored spec is
    /// never mutated.
    #[must_use]
    pub fn spec(&self) -> &SshTransportSpec {
        &self.spec
    }

    /// Returns the effective SSH configuration for `request`.
    ///
    /// A request whose `options.kind` is [`MemberSpawnKind::Ssh`] and whose
    /// `options.ssh` is set wins over the spawner's spec; every other request uses the spawner's
    /// spec.
    fn effective_spec<'a>(&'a self, request: &'a MemberSpawnRequest) -> &'a SshTransportSpec {
        match &request.options.ssh {
            Some(spec) if request.options.kind == MemberSpawnKind::Ssh => spec,
            _ => &self.spec,
        }
    }

    /// Returns the remote command word for `request`, identity prologue included.
    ///
    /// The result is the raw command the remote login shell should parse: the `export` prologue
    /// followed by the effective spec's [`remote_command`](SshTransportSpec::remote_command).
    /// It is not quoted and carries no `exec ` prefix; [`Self::command_for`] adds both when it
    /// renders the argv.
    #[must_use]
    pub fn remote_command_for(&self, request: &MemberSpawnRequest) -> String {
        let spec = self.effective_spec(request);
        format!("{}{}", identity_prologue(request), spec.remote_command)
    }

    /// Returns the effective `ssh` launch command for `request`, identity prefix included.
    ///
    /// The command is the effective spec rendered by [`SshTransportSpec::to_command`], with its
    /// remote command replaced by the `export` prologue followed by the member command.
    /// The `exec ` prefix of [`SshTransportSpec::use_exec`] is placed between the prologue and the
    /// member command, so it applies to the member process and not to the `export` built-in.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::config`] for an empty host or an empty remote
    /// command and [`openjiuwen_common::error::Error::validation`] for port `0`, all surfacing
    /// from [`SshTransportSpec::validate`] before any process starts; propagates program
    /// resolution failures (a missing `ssh` binary).
    pub fn command_for(&self, request: &MemberSpawnRequest) -> Result<StdioCommand> {
        let base = self.effective_spec(request);
        // Validate the configured command, not the prologue-prefixed one: an empty
        // `remote_command` is a configuration error even when the request carries identity.
        base.validate()?;
        let member_command = if base.use_exec {
            format!("exec {}", base.remote_command)
        } else {
            base.remote_command.clone()
        };
        let mut spec = base.clone();
        spec.remote_command = format!("{}{}", identity_prologue(request), member_command);
        // The prefix is already part of `remote_command`; rendering it again would put `exec`
        // before the `export` prologue.
        spec.use_exec = false;
        spec.to_command()
    }

    /// Returns the effective `ssh` argv for `request` (what actually runs).
    ///
    /// This is exactly [`Self::command_for`] rendered as `[program, args...]`; it never resolves
    /// the program on `PATH`, so it is safe for logging and for asserting argv shapes.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::command_for`] failures.
    pub fn argv_for(&self, request: &MemberSpawnRequest) -> Result<Vec<String>> {
        self.command_for(request).map(|command| command.to_argv())
    }

    /// Returns `true` when the member's local `ssh` process is still running.
    ///
    /// A member that is not registered (never spawned, already despawned or replaced) is not
    /// running.
    #[must_use]
    pub fn is_running(&self, member_name: &str) -> bool {
        self.live
            .lock()
            .get(member_name)
            .is_some_and(|runtime| runtime.is_running())
    }

    /// Returns the registered member names, sorted.
    #[must_use]
    pub fn live_members(&self) -> Vec<String> {
        let mut names: Vec<String> = self.live.lock().keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns the local `ssh` process id for `member_name`, while it is alive.
    #[must_use]
    pub fn child_pid(&self, member_name: &str) -> Option<u32> {
        self.live
            .lock()
            .get(member_name)
            .and_then(|runtime| runtime.child_pid())
    }
}

#[async_trait]
impl MemberSpawner for SshMemberSpawner {
    /// Launches `ssh` for `request` and wraps its stdio in a [`StdioMemberRuntime`].
    ///
    /// The local environment is left untouched: identity crosses as the remote `export` prologue,
    /// so applying [`MemberSpawnRequest::apply_identity_env`]
    /// would only add unused variables to the local `ssh` process.
    /// The runtime is not started — [`MemberRuntime::start`] is the member lifecycle's call, so a
    /// failed handshake can still roll the roster row back.
    /// A re-spawn replaces the registry entry; the replaced runtime is dropped, which kills its
    /// local `ssh` process.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::command_for`] failures (invalid spec, missing `ssh` binary) and
    /// [`StdioMemberRuntime::spawn`] failures.
    async fn spawn(&self, request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>> {
        let command = self.command_for(&request)?;
        let runtime = StdioMemberRuntime::spawn(&command, request.session_id.clone()).await?;
        let runtime = Arc::new(runtime);
        debug!(member = %request.member_name, host = %self.effective_spec(&request).host, "spawned ssh member");
        self.live
            .lock()
            .insert(request.member_name.clone(), Arc::clone(&runtime));
        Ok(runtime)
    }

    /// Removes `member_name` and stops its local `ssh` process.
    ///
    /// Removal happens before the await so the mutex is never held across it; `stop` is idempotent,
    /// and an unknown member (or a second despawn) is `Ok(())`.
    ///
    /// # Errors
    ///
    /// Propagates [`MemberRuntime::stop`] failures.
    async fn despawn(&self, member_name: &str) -> Result<()> {
        let runtime = self.live.lock().remove(member_name);
        match runtime {
            Some(runtime) => runtime.stop().await,
            None => Ok(()),
        }
    }
}

/// Builds the POSIX `export` prologue that carries the member's identity to the remote host.
///
/// Variables are emitted in the order team, member, session, workspace; empty request values are
/// skipped, as in [`MemberSpawnRequest::identity_env`].
/// Every value is single-quoted with `'` escaped as `'\''`, so the remote shell receives it
/// verbatim and the whole prologue stays one shell statement sequence.
fn identity_prologue(request: &MemberSpawnRequest) -> String {
    let workspace = request.workspace_root.to_string_lossy().into_owned();
    let mut prologue = String::new();
    for (key, value) in [
        (TEAM_NAME_ENV, request.team_name.as_str()),
        (MEMBER_NAME_ENV, request.member_name.as_str()),
        (SESSION_ID_ENV, request.session_id.as_str()),
        (MEMBER_WORKSPACE_ENV, workspace.as_str()),
    ] {
        if value.is_empty() {
            continue;
        }
        prologue.push_str("export ");
        prologue.push_str(key);
        prologue.push('=');
        prologue.push_str(&quote_shell_word(value));
        prologue.push_str("; ");
    }
    prologue
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
