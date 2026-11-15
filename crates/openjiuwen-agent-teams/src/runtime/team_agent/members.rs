//! Member lifecycle: spawning, retiring and listing teammates plus workspace bookkeeping.
//!
//! Every teammate is a roster row whose harness comes from the injected factory; the leader's own
//! row is built by `super::super::build`. Workspaces are retained per member and released once.
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tracing::{debug, warn};

use super::super::build::teammate_record;
use super::super::member_roles;
use super::super::memory_bind;
use super::super::observer::MemberStreamObserver;
use super::{ensure_leader_role, TeamAgent};
use crate::kv_cache::MemberCacheKey;
use crate::member::{MemberSpawnRequest, MemberSpawner, TeamMember};
use crate::schema::{
    EventPayload, MemberLifetime, MemberStatus, MemberStatusChangedEvent, TaskGraphSpec,
    TaskRecord, TeamMemberSpec, TeamRole, TeamTopic,
};
use crate::workspace::{MemberWorkspace, WorkspaceRequest};

impl TeamAgent {
    /// Releases one workspace through the injected provider, logging (never failing) a problem.
    async fn release_workspace(&self, name: &str, workspace: MemberWorkspace) {
        match self
            .inner
            .spawner
            .deps()
            .workspace
            .release(&workspace)
            .await
        {
            Ok(()) => debug!(
                team = self.team_name(),
                member = name,
                root = %workspace.root.display(),
                "released the member workspace"
            ),
            Err(error) => warn!(
                team = self.team_name(),
                member = name,
                root = %workspace.root.display(),
                %error,
                "failed to release the member workspace"
            ),
        }
    }

    /// Retains `workspace` as `name`'s current workspace, releasing a stale one first.
    ///
    /// Retaining happens only after a member runtime exists: a failed spawn hands its workspace
    /// straight back instead of holding it until the team stops.
    async fn retain_workspace(&self, name: &str, workspace: MemberWorkspace) {
        let stale = self
            .inner
            .workspaces
            .lock()
            .insert(name.to_owned(), workspace);
        if let Some(stale) = stale {
            self.release_workspace(name, stale).await;
        }
    }

    /// Releases and forgets `name`'s retained workspace; a second call releases nothing.
    async fn release_member_workspace(&self, name: &str) {
        let workspace = self.inner.workspaces.lock().remove(name);
        if let Some(workspace) = workspace {
            self.release_workspace(name, workspace).await;
        }
    }

    /// Releases every retained workspace, taking them all out of the map before the first await.
    ///
    /// Draining first keeps a concurrent second teardown from releasing the same workspace twice.
    pub(super) async fn release_all_workspaces(&self) {
        let retained: Vec<(String, MemberWorkspace)> = {
            let mut guard = self.inner.workspaces.lock();
            std::mem::take(&mut *guard).into_iter().collect()
        };
        for (name, workspace) in retained {
            self.release_workspace(&name, workspace).await;
        }
    }

    /// Spawns one teammate: allocate its workspace, build its runtime, start it.
    ///
    /// The leader name is a no-op when the team is started. An already-running member is a no-op.
    /// A failed spawn leaves an `unstarted` row behind and returns the factory's error.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for a name on no roster, [`Error::Unsupported`] for an unsupported role
    /// (one with no local harness, e.g. `passive_human`; Python's `TeamRole.PASSIVE_HUMAN` has no
    /// harness, so no startup, restart or recovery path may spawn a runtime for it),
    /// [`Error::Validation`] when the team leader is not running, and the
    /// workspace/factory/runtime errors of the spawn itself.
    pub async fn spawn_member(&self, name: &str) -> Result<()> {
        if name == self.leader_member_name() {
            return if self.is_started() {
                Ok(())
            } else {
                Err(Error::validation(format!(
                    "team leader '{name}' is not running; start the team first"
                )))
            };
        }
        let Some(member_spec) = self.inner.spawner.member_spec(name) else {
            return Err(Error::not_found(format!(
                "member '{name}' not found in team '{}'",
                self.team_name()
            )));
        };
        if !member_roles::has_local_harness(member_spec.role_type) {
            return Err(member_roles::in_process_refusal(
                name,
                member_spec.role_type,
            ));
        }
        if let Some(existing) = self.inner.backend.member(name) {
            if matches!(
                existing.status(),
                MemberStatus::Starting
                    | MemberStatus::Ready
                    | MemberStatus::Busy
                    | MemberStatus::Paused
            ) {
                debug!(
                    team = self.team_name(),
                    member = name,
                    "member is already running"
                );
                return Ok(());
            }
        }

        let spawner = Arc::clone(&self.inner.spawner);
        let workspace = spawner
            .deps()
            .workspace
            .allocate(&WorkspaceRequest {
                team_name: self.team_name().to_owned(),
                session_id: self.session_id().to_owned(),
                member_name: name.to_owned(),
                spec: self.inner.blueprint.workspace.clone(),
                member_spec: spawner
                    .deep_agent_spec(name)
                    .and_then(|spec| spec.workspace.clone()),
                isolation: self.inner.isolation,
            })
            .await?;

        let request = MemberSpawnRequest::new(
            self.team_name(),
            name,
            self.session_id(),
            workspace.root.clone(),
        )
        .with_options(self.inner.spawn_options.clone());
        let runtime = match spawner.spawn(request).await {
            Ok(runtime) => runtime,
            Err(error) => {
                self.ensure_row(name, &member_spec).await;
                // The member never got a runtime, so nothing owns its workspace: hand it back now
                // rather than retaining it until a `stop` that may never come.
                self.release_workspace(name, workspace).await;
                return Err(error);
            }
        };
        // A live member owns its workspace until its shutdown or the team's stop.
        self.retain_workspace(name, workspace).await;

        let member = if let Some(member) = self.inner.backend.member(name) {
            member
        } else {
            let record = teammate_record(&spawner, name, &self.inner.routing);
            self.inner.store.insert_member(record.clone()).await?;
            let created = Arc::new(TeamMember::new(self.inner.store.clone(), record));
            self.inner.backend.add_member(Arc::clone(&created));
            created
        };
        if member.runtime().is_none() {
            member.observe(Arc::new(MemberStreamObserver::new(
                Arc::clone(&self.inner.activity),
                Arc::downgrade(&member),
                member.role(),
                name.to_owned(),
                self.inner.signals.clone(),
                Arc::clone(&self.inner.chunk_sinks),
            )));
        }
        member.attach_runtime(runtime);
        let previous = member.status();
        member.start(None).await?;
        // Register the KV identity only after the start succeeded: a failed start would otherwise
        // leave a live affinity entry for a member that never ran.
        self.register_kv_affinity(name);
        // Same rule for the memory join: a member that never became live never joined, so a failed
        // spawn leaves no join record (and no orphan handoff note behind) either.
        memory_bind::member_joined(self.inner.backend.memory().as_ref(), self.team_name(), name)
            .await;
        self.inner.signals.member_spawned(name, member.role());
        self.publish_member_status(name, previous, member.status())
            .await;
        Ok(())
    }

    /// Ensures an `unstarted` row exists for `name`, logging (never failing) a store problem.
    async fn ensure_row(&self, name: &str, member_spec: &TeamMemberSpec) {
        if self.inner.backend.member(name).is_some() {
            return;
        }
        let record = teammate_record(&self.inner.spawner, name, &self.inner.routing);
        debug!(
            team = self.team_name(),
            member = name,
            role = member_spec.role_type.as_str(),
            "materializing an unstarted roster row after a failed spawn"
        );
        if let Err(error) = self.inner.store.insert_member(record.clone()).await {
            warn!(team = self.team_name(), member = name, error = %error, "failed to persist the roster row");
            return;
        }
        self.inner
            .backend
            .add_member(Arc::new(TeamMember::new(self.inner.store.clone(), record)));
    }

    /// Publishes a member status change, logging (never failing) a transport problem.
    async fn publish_member_status(
        &self,
        name: &str,
        previous: MemberStatus,
        status: MemberStatus,
    ) {
        self.inner
            .signals
            .member_status(name, self.role_of(name), status);
        if previous == status {
            return;
        }
        let payload = EventPayload::MemberStatusChanged(MemberStatusChangedEvent {
            team_name: self.team_name().to_owned(),
            member_name: name.to_owned(),
            status,
            previous: Some(previous),
        });
        if let Err(error) = self
            .inner
            .backend
            .post_event(TeamTopic::Team, payload)
            .await
        {
            warn!(team = self.team_name(), member = name, error = %error, "failed to publish a member status change");
        }
    }

    /// Retires one member and returns the status its row settles on.
    ///
    /// After the runtime is stopped and the child despawned, the member's retained workspace is
    /// released through the injected provider (a release failure is logged, never propagated).
    /// A member that became live also leaves a handoff note in team memory (best-effort: a memory
    /// failure is logged and never blocks the retirement).
    /// Idempotent: a second shutdown of the same member releases nothing and joins nothing twice.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for an unknown member, the member runtime's teardown error (logged, not
    /// propagated) and the lifecycle machine's own validation errors.
    pub async fn shutdown_member(
        &self,
        name: &str,
        lifetime: MemberLifetime,
        force: bool,
    ) -> Result<MemberStatus> {
        let Some(member) = self.inner.backend.member(name) else {
            return Err(Error::not_found(format!(
                "member '{name}' not found in team '{}'",
                self.team_name()
            )));
        };
        let previous = member.status();
        // Only a member that became live has something to hand over; an unstarted row never joined.
        let live = memory_bind::is_live(previous);
        if let Some(runtime) = member.runtime() {
            if let Err(error) = runtime.stop().await {
                warn!(team = self.team_name(), member = name, error = %error, "failed to stop a member runtime");
            }
        }
        let status = member.shutdown(lifetime, force).await?;
        if live {
            memory_bind::member_left(self.inner.backend.memory().as_ref(), self.team_name(), name)
                .await;
        }
        if let Err(error) = self.inner.spawner.despawn(name).await {
            debug!(team = self.team_name(), member = name, %error, "failed to reap a member child process");
        }
        let _ = self.inner.kv_cache.release(name).await;
        self.release_member_workspace(name).await;
        self.inner
            .signals
            .member_stopped(name, member.role(), "shutdown");
        self.publish_member_status(name, previous, status).await;
        Ok(status)
    }

    /// Spawns every spawnable `unstarted` non-leader member, returning the first error after the
    /// loop.
    ///
    /// The leader is never started here, and a member without a local harness (e.g. a passive
    /// human, which Python gives no harness at all) is skipped instead of failing the sweep.
    ///
    /// # Errors
    ///
    /// Returns the first spawn error; every member the loop reached before it was still spawned.
    pub async fn auto_start_all(&self) -> Result<()> {
        let mut first_error = None;
        for member in self.inner.backend.members() {
            if member.role() == TeamRole::Leader
                || member.status() != MemberStatus::Unstarted
                || !member_roles::has_local_harness(member.role())
            {
                continue;
            }
            if let Err(error) = self.spawn_member(member.member_name()).await {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Creates one top-level task on the team's board and publishes its events.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] on a non-leader handle (`only the leader can create tasks`), the
    /// board's own validation (duplicate id, cycle, unknown dependency) and the store's errors.
    pub async fn create_task(&self, spec: TaskGraphSpec) -> Result<TaskRecord> {
        ensure_leader_role(self.role())?;
        let mutation = self.inner.backend.tasks().create_task(spec).await?;
        self.inner.backend.publish_mutation(&mutation).await?;
        Ok(mutation.task)
    }

    /// Sends `content` from this member, optionally addressed to `to`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an over-long body or an empty target list, and the store's or the
    /// transport's errors.
    pub async fn send_message(
        &self,
        to: Option<&str>,
        content: &str,
    ) -> Result<crate::storage::MessageRecord> {
        self.inner
            .backend
            .send_message(self.member_name(), to, content)
            .await
    }

    /// Returns the live roster handles, sorted by member name.
    ///
    /// # Errors
    ///
    /// This implementation never fails; the signature is async so a future store-backed roster
    /// does not change the seam.
    // Async for seam stability: a future store-backed roster reads asynchronously.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn members(&self) -> Vec<Arc<TeamMember>> {
        self.inner.backend.members()
    }

    /// Subscribes to this handle's runtime chunks.
    ///
    /// The fan-out is the run loop's input: every registered sender receives each chunk the
    /// member's harness observer reports, and a dropped receiver is pruned on the next chunk.
    #[must_use]
    pub fn subscribe_chunks(&self) -> UnboundedReceiver<Value> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.inner.chunk_sinks.lock().push(tx);
        rx
    }

    /// Registers one member's KV-cache affinity, deriving a child identity for a fork.
    fn register_kv_affinity(&self, member_name: &str) {
        let key = MemberCacheKey::new(self.team_name(), self.session_id(), member_name.to_owned());
        let model = self
            .inner
            .routing
            .model_ref_for(member_name, None, self.role_of(member_name))
            .unwrap_or_default();
        let forked = self.inner.spawner.forked_member(member_name);
        let identity = match forked {
            Some(child) => self
                .inner
                .kv_cache
                .register_child(&child.parent_member, key, &model),
            None => Ok(self.inner.kv_cache.register_member(key, &model)),
        };
        match identity {
            Ok(identity) => debug!(
                team = self.team_name(),
                member = member_name,
                cache = %identity.session_id,
                "registered the member's KV-cache affinity"
            ),
            Err(error) => debug!(
                team = self.team_name(),
                member = member_name,
                %error,
                "KV-cache affinity not registered (disabled or unknown parent)"
            ),
        }
    }
    /// Returns a roster member's role, defaulting to teammate for an unknown name.
    fn role_of(&self, member_name: &str) -> TeamRole {
        self.inner
            .backend
            .member(member_name)
            .map_or(TeamRole::Teammate, |member| member.role())
    }
}
