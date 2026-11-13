//! External-CLI member spawner: launch a third-party CLI as a protocol-speaking child process.
//!
//! [`ExternalCliMemberSpawner`] is the [`MemberSpawner`] for
//! [`MemberSpawnKind::ExternalCli`](crate::member::spawn_config::MemberSpawnKind::ExternalCli).
//! It clones the effective [`ExternalCliRuntimeConfig`] command, layers the request's identity
//! environment on top of it, and hands the result to
//! [`StdioMemberRuntime`](crate::member::stdio::StdioMemberRuntime) — the same JSON-lines protocol
//! runtime the OS-process spawner uses.
//! Only the launch details differ between the two kinds, so this spawner adds configuration
//! precedence and a per-member runtime registry, nothing else.
//!
//! Python parity source: `openjiuwen/agent_teams/spawn/external_cli_spawn.py`, which builds an
//! external-CLI member runtime and injects it into the in-process `TeamAgent` shell instead of
//! configuring a `DeepAgent`.
//!
//! # Configuration precedence
//!
//! The spawner's own [`ExternalCliRuntimeConfig`] is authoritative.
//! A request overrides it only when it explicitly asks for an external CLI
//! ([`MemberSpawnKind::ExternalCli`](crate::member::spawn_config::MemberSpawnKind::ExternalCli))
//! and carries [`MemberSpawnOptions::external_cli`](crate::member::spawn_config::MemberSpawnOptions::external_cli);
//! a request of any other kind keeps the spawner's configuration, so a mislabelled request cannot
//! silently change how members are launched.
//!
//! # Environment and credentials
//!
//! Credentials and provider settings reach the CLI through the inherited process environment plus
//! [`StdioCommand::env`](crate::member::spawn_config::StdioCommand::env).
//! The request's identity (`OPENJIUWEN_TEAM_NAME`, `OPENJIUWEN_MEMBER_NAME`,
//! `OPENJIUWEN_SESSION_ID`, `OPENJIUWEN_MEMBER_WORKSPACE`) is applied by
//! [`MemberSpawnRequest::apply_identity_env`], which never overrides an explicit value.
//! `cli_agent`, `provider` and `model` are diagnostic labels: they are logged at debug level and
//! never placed on the argv, and no environment variable is invented from them.
//!
//! # Parity gap (documented)
//!
//! Python's `openjiuwen.agent_teams.external.cli_agent.*` layer translates the member protocol per
//! vendor: `adapter.format_input` / `adapter.summarize_output_line` convert protocol frames to and
//! from each CLI's input and output format, `StdinPipeInjector` writes either plain text or
//! `INPUT_CODEX_PROTO` JSON lines, turn completion is detected from vendor markers such as
//! `task_complete` or `turn.completed`, and `model_compat.py` negotiates provider/model
//! compatibility.
//! None of that is ported here.
//! This spawner requires the CLI — or a wrapper script around it — to speak the
//! [`crate::member::protocol`] JSON-lines protocol on stdin/stdout directly.
//! A wrapper script that translates between the vendor CLI and the member protocol is the intended
//! bridge; the vendor-SDK adapters (the `claude` and `codex` harness providers) stay deferred.
//!
//! ```
//! use openjiuwen_agent_teams::member::external_cli_spawner::ExternalCliMemberSpawner;
//! use openjiuwen_agent_teams::member::spawn_config::{ExternalCliRuntimeConfig, StdioCommand};
//!
//! let config = ExternalCliRuntimeConfig {
//!     command: StdioCommand::new("my-cli-wrapper").with_arg("--member-protocol"),
//!     cli_agent: Some("claude".to_string()),
//!     ..ExternalCliRuntimeConfig::default()
//! };
//! let spawner = ExternalCliMemberSpawner::new(config);
//! assert_eq!(spawner.config().cli_agent.as_deref(), Some("claude"));
//! assert!(spawner.live_members().is_empty());
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use tracing::debug;

use crate::member::runtime::MemberRuntime;
use crate::member::spawn::{MemberSpawnRequest, MemberSpawner};
use crate::member::spawn_config::{ExternalCliRuntimeConfig, MemberSpawnKind};
use crate::member::stdio::StdioMemberRuntime;

/// Spawns teammates as third-party CLI child processes driven over the member protocol.
///
/// The spawner's configuration is authoritative; a request selects its own external-CLI config
/// only when it explicitly asks for one, mirroring the OS-process spawner's precedence.
/// The effective command is cloned, the request's identity environment is applied, and the child
/// is launched through [`StdioMemberRuntime`], so the member speaks the same JSON-lines protocol as
/// an OS-process member.
/// `cli_agent`, `provider` and `model` are diagnostic labels only: they are logged and never
/// reach the argv or the environment.
/// The spawner owns at most one runtime per member name; re-spawning replaces the entry (dropping
/// the replaced runtime kills its child) and [`MemberSpawner::despawn`] stops and removes it.
/// An unknown program surfaces from `StdioCommand::resolve_program` as
/// [`openjiuwen_common::error::ErrorKind::Config`], so a configuration mistake is distinguishable
/// from a launch failure.
#[derive(Debug)]
pub struct ExternalCliMemberSpawner {
    /// Launch configuration used by every request that does not bring its own.
    config: ExternalCliRuntimeConfig,
    /// Live runtimes keyed by member name; the value owns the child process.
    live: Mutex<HashMap<String, Arc<StdioMemberRuntime>>>,
}

impl ExternalCliMemberSpawner {
    /// Creates a spawner whose [`Self::config`] is authoritative for every request.
    #[must_use]
    pub fn new(config: ExternalCliRuntimeConfig) -> Self {
        Self {
            config,
            live: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the spawner-wide launch configuration.
    #[must_use]
    pub fn config(&self) -> &ExternalCliRuntimeConfig {
        &self.config
    }

    /// Returns `true` while the child spawned for `member_name` has not terminated.
    #[must_use]
    pub fn is_running(&self, member_name: &str) -> bool {
        self.live
            .lock()
            .get(member_name)
            .is_some_and(|runtime| runtime.is_running())
    }

    /// Returns the names of every member the spawner currently holds a runtime for, sorted.
    #[must_use]
    pub fn live_members(&self) -> Vec<String> {
        let mut names: Vec<String> = self.live.lock().keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns the operating-system process id of `member_name`'s child, if it is held.
    ///
    /// The pid is captured at spawn time; after the child exits the runtime has no live process,
    /// but the id is still reported until the entry is removed by [`MemberSpawner::despawn`].
    #[must_use]
    pub fn child_pid(&self, member_name: &str) -> Option<u32> {
        self.live
            .lock()
            .get(member_name)
            .and_then(|runtime| runtime.child_pid())
    }

    /// Returns the configuration that applies to `request`.
    ///
    /// The per-request configuration wins only when the request explicitly asks for an external
    /// CLI ([`MemberSpawnKind::ExternalCli`]) and carries one; otherwise the spawner's own config
    /// applies, whatever kind the request names.
    fn effective_config<'a>(
        &'a self,
        request: &'a MemberSpawnRequest,
    ) -> &'a ExternalCliRuntimeConfig {
        if request.options.kind == MemberSpawnKind::ExternalCli {
            if let Some(config) = request.options.external_cli.as_ref() {
                return config;
            }
        }
        &self.config
    }
}

#[async_trait]
impl MemberSpawner for ExternalCliMemberSpawner {
    /// Launches the effective external-CLI command as a protocol-speaking child process.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Config`] when the program is missing (the
    /// error from `StdioCommand::resolve_program`, unchanged), and
    /// [`openjiuwen_common::error::ErrorKind::Transport`] when the child cannot be started or its
    /// pipes are unavailable.
    async fn spawn(&self, request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>> {
        let config = self.effective_config(&request);
        debug!(
            team = %request.team_name,
            member = %request.member_name,
            cli_agent = config.cli_agent.as_deref().unwrap_or(""),
            provider = config.provider.as_deref().unwrap_or(""),
            model = config.model.as_deref().unwrap_or(""),
            program = %config.command.program(),
            "spawning external CLI member"
        );

        let mut command = config.command.clone();
        request.apply_identity_env(&mut command);
        let runtime = Arc::new(StdioMemberRuntime::spawn(&command, &request.session_id).await?);

        // The lock is released before the replaced runtime is dropped: its `Drop` kills the old
        // child, which must never happen with the registry lock held.
        let replaced = self
            .live
            .lock()
            .insert(request.member_name.clone(), Arc::clone(&runtime));
        if replaced.is_some() {
            debug!(
                member = %request.member_name,
                "replaced the live runtime of a re-spawned external CLI member"
            );
        }
        drop(replaced);
        Ok(runtime)
    }

    /// Stops and forgets the runtime spawned for `member_name`.
    ///
    /// An unknown member is not an error: the spawn may never have completed, and `despawn` is
    /// expected to be idempotent. The registry lock is released before the runtime is stopped.
    async fn despawn(&self, member_name: &str) -> Result<()> {
        let runtime = self.live.lock().remove(member_name);
        if let Some(runtime) = runtime {
            runtime.stop().await?;
        }
        Ok(())
    }
}
