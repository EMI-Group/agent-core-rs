//! The OS-process member spawner: one `team-member` child per teammate over the JSON-lines
//! member protocol.
//!
//! Python parity source: `openjiuwen/agent_teams/spawn/` (the child-process branch that Python
//! reaches through `Runner.spawn_agent`); the Rust port keeps the launch configuration as data
//! ([`ProcessMemberConfig`]) and owns the live [`StdioMemberRuntime`]s of one team.
//!
//! What this node does: pick a launch command, export the identity environment, start the child
//! through [`StdioMemberRuntime::spawn`], and keep the returned runtimes so a later `despawn` can
//! tear them down.
//! What this node does not do: the child's own behaviour.
//! The default command is `team-member serve`, whose `serve` subcommand is wired at the crate
//! root, and the child is expected to speak [`crate::member::protocol`] on its stdin/stdout.
//! Spawn-mode dispatch is also not decided here: a [`MemberSpawner`] is handed a request that has
//! already been routed to it, and this spawner spawns a process whatever
//! [`MemberSpawnKind`](crate::member::spawn_config::MemberSpawnKind) the request names.
//!
//! Launch rules, in order:
//!
//! 1. The spawner's own [`ProcessMemberConfig`] is the default launch command.
//! 2. When the request selects [`MemberSpawnKind::Process`] and carries
//!    [`MemberSpawnOptions::process`](crate::member::spawn_config::MemberSpawnOptions::process),
//!    that per-request configuration wins, so the runtime can route a per-member command.
//! 3. [`MemberSpawnRequest::apply_identity_env`] layers the team name, member name, session id and
//!    member workspace onto the chosen command, so the child learns its identity without a
//!    protocol handshake.
//! 4. The program is resolved before the child starts; a missing binary surfaces unchanged as
//!    [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config) from
//!    [`StdioCommand::resolve_program`], and the operator fixes it by installing `team-member` on
//!    `PATH` or by setting an explicit program path in the spec.
//!
//! The live runtimes are keyed by member name: re-spawning a member replaces the entry and the
//! replaced runtime is stopped (its own `Drop` kills a still-live child when it was the last
//! holder), so a re-spawn is a replace rather than an error, matching
//! [`SpawnRegistry::insert`](crate::member::spawn::SpawnRegistry::insert) for in-process members.
//! A failed launch registers nothing, so it never disturbs a runtime that is already live.
//!
//! ```
//! use openjiuwen_agent_teams::member::process_spawner::ProcessMemberSpawner;
//! use openjiuwen_agent_teams::member::spawn_config::ProcessMemberConfig;
//!
//! let spawner = ProcessMemberSpawner::new(ProcessMemberConfig::default());
//! assert_eq!(
//!     spawner.config().command.to_argv(),
//!     vec!["team-member".to_string(), "serve".to_string()]
//! );
//! assert!(spawner.live_members().is_empty());
//! assert!(!spawner.is_running("dev-1"));
//! assert!(spawner.child_pid("dev-1").is_none());
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use tracing::{debug, info, warn};

use super::runtime::MemberRuntime;
use super::spawn::{MemberSpawnRequest, MemberSpawner};
use super::spawn_config::{MemberSpawnKind, ProcessMemberConfig, StdioCommand};
use super::stdio::StdioMemberRuntime;

/// Spawns teammates as child processes that speak the JSON-lines member protocol.
///
/// The spawner holds one [`ProcessMemberConfig`] as the default launch command and the live
/// [`StdioMemberRuntime`] of every member it spawned, keyed by member name.
/// It is `Send + Sync`; the map is a [`parking_lot::Mutex`] held for one lookup or insertion and
/// never across an `.await`.
pub struct ProcessMemberSpawner {
    /// Default launch command, used whenever the request carries no process configuration.
    config: ProcessMemberConfig,
    /// Live runtimes, keyed by member name.
    live: Mutex<HashMap<String, Arc<StdioMemberRuntime>>>,
}

impl ProcessMemberSpawner {
    /// Builds a spawner from a launch configuration.
    #[must_use]
    pub fn new(config: ProcessMemberConfig) -> Self {
        Self {
            config,
            live: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the configured launch command.
    ///
    /// This is the spawner's default; a request that selects [`MemberSpawnKind::Process`] and
    /// carries its own [`ProcessMemberConfig`] overrides it for that spawn only.
    #[must_use]
    pub fn config(&self) -> &ProcessMemberConfig {
        &self.config
    }

    /// Returns `true` when a live runtime is registered for `member_name`.
    ///
    /// A registered runtime whose child has exited still counts as registered until
    /// [`MemberSpawner::despawn`] removes it; the child's own liveness is available through
    /// [`Self::child_pid`] and the runtime's state.
    #[must_use]
    pub fn is_running(&self, member_name: &str) -> bool {
        self.live.lock().contains_key(member_name)
    }

    /// Returns the registered member names, sorted.
    #[must_use]
    pub fn live_members(&self) -> Vec<String> {
        let mut names: Vec<String> = self.live.lock().keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns the pid of a live member's child process.
    ///
    /// Returns `None` for an unknown member, and also for a child whose pid the operating system
    /// did not report at spawn time.
    #[must_use]
    pub fn child_pid(&self, member_name: &str) -> Option<u32> {
        let runtime = self.live.lock().get(member_name).map(Arc::clone)?;
        runtime.child_pid()
    }

    /// Picks the effective command for `request` and applies the identity environment.
    ///
    /// Precedence: a request that selects [`MemberSpawnKind::Process`] and carries
    /// `options.process` supplies the command; every other request (including an in-process one)
    /// uses the spawner's own [`Self::config`].
    fn command_for(&self, request: &MemberSpawnRequest) -> StdioCommand {
        let config = match (request.options.kind, request.options.process.as_ref()) {
            (MemberSpawnKind::Process, Some(process)) => process,
            _ => &self.config,
        };
        let mut command = config.command.clone();
        request.apply_identity_env(&mut command);
        command
    }
}

impl Default for ProcessMemberSpawner {
    /// Creates a spawner launching `team-member serve`.
    fn default() -> Self {
        Self::new(ProcessMemberConfig::default())
    }
}

impl fmt::Debug for ProcessMemberSpawner {
    /// Prints the configured program and the number of live runtimes.
    ///
    /// The live children print through [`StdioMemberRuntime`]'s own `Debug`; only their count is
    /// shown here so a diagnostic never walks a locked map of processes.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessMemberSpawner")
            .field("program", &self.config.command.program())
            .field("args", &self.config.command.args())
            .field("live", &self.live.lock().len())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl MemberSpawner for ProcessMemberSpawner {
    /// Starts one child process for `request` and registers its runtime.
    ///
    /// The identity environment is applied to the chosen command before the program is resolved, so
    /// a missing binary fails with the operator-facing configuration error before anything starts.
    /// A successful launch replaces any runtime already registered for the same member and stops
    /// the replaced one; a failed launch leaves the registry untouched.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config) when the program
    /// cannot be resolved and
    /// [`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport) when the child
    /// cannot be spawned or one of its pipes is unavailable.
    async fn spawn(&self, request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>> {
        let command = self.command_for(&request);
        // The argv is safe to log; the environment values are not secrets today, but the habit is
        // to log the command line only.
        let argv = command.to_argv();
        debug!(
            team = %request.team_name,
            member = %request.member_name,
            session = %request.session_id,
            program = %command.program(),
            ?argv,
            "spawning member process"
        );
        let program = command.resolve_program()?;
        debug!(
            member = %request.member_name,
            program = %program.display(),
            "resolved the member process program"
        );
        let runtime = Arc::new(StdioMemberRuntime::spawn(&command, &request.session_id).await?);
        let pid = runtime.child_pid();
        let previous = self
            .live
            .lock()
            .insert(request.member_name.clone(), Arc::clone(&runtime));
        if let Some(previous) = previous {
            info!(
                team = %request.team_name,
                member = %request.member_name,
                "replacing the live member runtime; stopping the previous child"
            );
            if let Err(error) = previous.stop().await {
                // The new runtime is already live and registered: a teardown failure of the
                // replaced child is a diagnostic, not a reason to fail the spawn.
                warn!(
                    member = %request.member_name,
                    %error,
                    "could not stop the replaced member runtime"
                );
            }
        }
        info!(
            team = %request.team_name,
            member = %request.member_name,
            ?pid,
            "member process spawned"
        );
        Ok(runtime)
    }

    /// Removes the runtime for `member_name` and stops its child.
    ///
    /// The entry is removed before the child is stopped, so a teardown failure never leaves a dead
    /// entry registered; an unknown member is not an error.
    ///
    /// # Errors
    ///
    /// Propagates a [`MemberRuntime::stop`] failure after the entry has been removed.
    async fn despawn(&self, member_name: &str) -> Result<()> {
        let removed = self.live.lock().remove(member_name);
        let Some(runtime) = removed else {
            debug!(member = %member_name, "no live member runtime to despawn");
            return Ok(());
        };
        let pid = runtime.child_pid();
        let result = runtime.stop().await;
        match &result {
            Ok(()) => info!(member = %member_name, ?pid, "member process stopped"),
            Err(error) => warn!(
                member = %member_name,
                ?pid,
                %error,
                "member process teardown failed"
            ),
        }
        result
    }
}
