//! Per-member checkpoint wiring: the identity and the store a member runtime is built with.
//!
//! The runtime resolves checkpoint destinations in two steps and both live here so the leader build
//! path and every teammate spawn share one rule:
//!
//! 1. **Once per team** — [`team_checkpoint_store`] resolves the store the whole team shares and the
//!    build path writes it into [`TeamBuildDeps::checkpoints`](super::deps::TeamBuildDeps::checkpoints)
//!    before the spawner is built. The injected store always wins; an enabled policy with no
//!    injection mints one fresh in-memory store per built team, shared by the leader and every
//!    teammate, so a re-spawned member of the same team restores its checkpoint while a new team
//!    build starts empty (injecting a store is the seam for a wider scope), and the file backend has
//!    no team-wide root (each member's own directory is resolved at member-build time).
//! 2. **Once per member** — [`member_checkpoint_store`] resolves that member's store: the shared one
//!    when the build path supplied it, else its own file root (an explicit directory, or
//!    `<workspace root>/`[`CHECKPOINT_DIR`]), and ultimately a private in-memory store for a member
//!    runtime built directly from a spawner that never went through the build path.
//!
//! A disabled policy allocates **no** store at both steps, so a member configured with
//! `{"enabled": false}` does no checkpoint work at all and leaves nothing behind.
//!
//! Identity follows the harness adapter's convention: a member's checkpoints are keyed on
//! `(member session id, card id)`, where the member session is
//! [`member_session_id`](crate::harness::MemberHarnessContext::member_session_id)`(team session,
//! member name)` and the card id is [`member_card_id`]. [`wire_runtime`] applies the identity, the
//! policy, the working directory and the store to a freshly built runtime — before it is wrapped in
//! an `Arc`, because every one of those builders consumes `self`.

use std::path::Path;
use std::sync::Arc;

use crate::harness::{
    FileMemberCheckpointStore, HarnessMemberRuntime, InMemoryMemberCheckpointStore,
    MemberCheckpointStore,
};
use crate::schema::DeepAgentSpec;

use super::spec_meta::{CheckpointConfig, CheckpointStoreConfig};

/// Directory under a member workspace root used when the metadata names no checkpoint dir.
pub(crate) const CHECKPOINT_DIR: &str = ".openjiuwen/checkpoints";

/// Card id a member's checkpoints are keyed on: the spec card's id when non-blank, else the member
/// name.
///
/// A changed card invalidates a stored record, so the id is part of the checkpoint identity rather
/// than a label: a blueprint with a pinned card resumes, one without falls back to the stable
/// member name.
pub(crate) fn member_card_id(spec: &DeepAgentSpec, member_name: &str) -> String {
    spec.card
        .as_ref()
        .map(|card| card.id.trim())
        .filter(|id| !id.is_empty())
        .map_or_else(|| member_name.to_owned(), str::to_owned)
}

/// Team-level store resolved once at build time (injected into `TeamBuildDeps` so every member
/// shares it).
///
/// `None` when the policy is disabled; the injected store verbatim; `Memory` →
/// [`InMemoryMemberCheckpointStore::shared`]; `File` → `None` (each member resolves its own file
/// root at member-build time).
pub(crate) fn team_checkpoint_store(
    config: &CheckpointConfig,
    injected: Option<&Arc<dyn MemberCheckpointStore>>,
) -> Option<Arc<dyn MemberCheckpointStore>> {
    if !config.policy.is_enabled() {
        return None;
    }
    if let Some(store) = injected {
        return Some(Arc::clone(store));
    }
    match config.store {
        CheckpointStoreConfig::Memory => Some(InMemoryMemberCheckpointStore::shared()),
        CheckpointStoreConfig::File { .. } => None,
    }
}

/// Per-member store: `None` when the policy is disabled; the team-shared store when present;
/// `File(Some(dir))` → a file store rooted at `dir`; `File(None)` → a file store rooted at
/// `workspace_root.join(CHECKPOINT_DIR)`; `Memory` → a fresh in-memory store (the fallback for a
/// directly built spawner).
pub(crate) fn member_checkpoint_store(
    config: &CheckpointConfig,
    shared: Option<&Arc<dyn MemberCheckpointStore>>,
    workspace_root: &Path,
) -> Option<Arc<dyn MemberCheckpointStore>> {
    if !config.policy.is_enabled() {
        return None;
    }
    if let Some(store) = shared {
        return Some(Arc::clone(store));
    }
    match &config.store {
        CheckpointStoreConfig::Memory => Some(Arc::new(InMemoryMemberCheckpointStore::new())),
        CheckpointStoreConfig::File { dir: Some(dir) } => {
            Some(Arc::new(FileMemberCheckpointStore::new(dir.clone())))
        }
        CheckpointStoreConfig::File { dir: None } => Some(Arc::new(
            FileMemberCheckpointStore::new(workspace_root.join(CHECKPOINT_DIR)),
        )),
    }
}

/// Applies card id, store, policy and cwd to a freshly built runtime (before it is put in an `Arc`).
///
/// The store is attached only when [`member_checkpoint_store`] yields one, so a disabled policy
/// attaches no sink and captures nothing.
pub(crate) fn wire_runtime(
    runtime: HarnessMemberRuntime,
    card_id: &str,
    config: &CheckpointConfig,
    shared: Option<&Arc<dyn MemberCheckpointStore>>,
    workspace_root: &Path,
) -> HarnessMemberRuntime {
    let mut runtime = runtime
        .with_card_id(card_id)
        .with_checkpoint_policy(config.policy.clone())
        .with_cwd(workspace_root.to_string_lossy().into_owned());
    if let Some(store) = member_checkpoint_store(config, shared, workspace_root) {
        runtime = runtime.with_checkpoint_store(store);
    }
    runtime
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_agent::AgentCard;
    use openjiuwen_harness_protocol::HarnessState;

    use super::*;
    use crate::harness::{
        MemberCheckpointPolicy, MemberCheckpointRecord, MemberHarnessContext, NativeCheckpoint,
    };
    use crate::member::MemberRuntime;

    /// A spec whose card carries an explicit id.
    fn spec_with_card(id: &str) -> DeepAgentSpec {
        DeepAgentSpec {
            card: Some(AgentCard::new("card").with_id(id)),
            ..DeepAgentSpec::default()
        }
    }

    /// The default (enabled) configuration on the in-memory backend.
    fn enabled() -> CheckpointConfig {
        CheckpointConfig::default()
    }

    /// The default configuration with checkpointing switched off.
    fn disabled() -> CheckpointConfig {
        CheckpointConfig {
            policy: MemberCheckpointPolicy::disabled(),
            ..CheckpointConfig::default()
        }
    }

    /// One payload for the file-store round trips.
    fn record() -> MemberCheckpointRecord {
        MemberCheckpointRecord::new(
            NativeCheckpoint::new("session-1:dev-1", "card-1").with_state(HarnessState::Idle),
        )
    }

    #[test]
    fn member_card_id_prefers_the_spec_card_id() {
        assert_eq!(member_card_id(&spec_with_card("card-7"), "dev-1"), "card-7");
    }

    #[test]
    fn member_card_id_falls_back_to_the_member_name() {
        assert_eq!(member_card_id(&spec_with_card("   "), "dev-1"), "dev-1");
        assert_eq!(
            member_card_id(&DeepAgentSpec::default(), "dev-1"),
            "dev-1",
            "a spec without a card still has a stable identity"
        );
    }

    #[test]
    fn team_store_is_none_when_the_policy_is_disabled() {
        let injected = InMemoryMemberCheckpointStore::shared();
        assert!(team_checkpoint_store(&disabled(), Some(&injected)).is_none());
        assert!(team_checkpoint_store(&disabled(), None).is_none());
    }

    #[test]
    fn team_store_reuses_the_injected_store() {
        let injected = InMemoryMemberCheckpointStore::shared();
        let store = team_checkpoint_store(&enabled(), Some(&injected)).expect("an injected store");
        assert!(Arc::ptr_eq(&store, &injected));
        assert_eq!(store.name(), "memory");
    }

    #[test]
    fn team_store_creates_one_shared_memory_store() {
        let config = enabled();
        let first = team_checkpoint_store(&config, None).expect("a memory store");
        let second = team_checkpoint_store(&config, None).expect("a memory store");
        assert_eq!(first.name(), "memory");
        assert!(
            !Arc::ptr_eq(&first, &second),
            "each resolution mints the store the build path then pins"
        );
    }

    #[test]
    fn team_store_is_none_for_the_file_backend() {
        let config = CheckpointConfig {
            store: CheckpointStoreConfig::File { dir: None },
            ..enabled()
        };
        assert!(team_checkpoint_store(&config, None).is_none());
    }

    #[test]
    fn member_store_is_none_when_the_policy_is_disabled() {
        let shared = InMemoryMemberCheckpointStore::shared();
        assert!(
            member_checkpoint_store(&disabled(), Some(&shared), Path::new("/tmp/ws")).is_none()
        );
        assert!(member_checkpoint_store(&disabled(), None, Path::new("/tmp/ws")).is_none());
    }

    #[test]
    fn member_store_reuses_the_shared_store() {
        let shared = InMemoryMemberCheckpointStore::shared();
        let store = member_checkpoint_store(&enabled(), Some(&shared), Path::new("/tmp/ws"))
            .expect("the shared store");
        assert!(Arc::ptr_eq(&store, &shared));
    }

    #[test]
    fn member_store_falls_back_to_a_private_memory_store() {
        let store = member_checkpoint_store(&enabled(), None, Path::new("/tmp/ws"))
            .expect("a private memory store");
        assert_eq!(store.name(), "memory");
    }

    #[tokio::test]
    async fn member_store_roots_a_file_store_at_the_named_dir() {
        let directory = tempfile::tempdir().unwrap();
        let config = CheckpointConfig {
            store: CheckpointStoreConfig::File {
                dir: Some(directory.path().to_path_buf()),
            },
            ..enabled()
        };
        let store =
            member_checkpoint_store(&config, None, Path::new("/tmp/ws")).expect("a file store");
        assert_eq!(store.name(), "file");

        store.save(record()).await.unwrap();
        let expected =
            FileMemberCheckpointStore::new(directory.path()).path_for("session-1:dev-1", "card-1");
        assert!(expected.exists(), "{expected:?}");
    }

    #[tokio::test]
    async fn member_store_roots_an_unspecified_file_dir_under_the_workspace() {
        let workspace = tempfile::tempdir().unwrap();
        let config = CheckpointConfig {
            store: CheckpointStoreConfig::File { dir: None },
            ..enabled()
        };
        let store = member_checkpoint_store(&config, None, workspace.path()).expect("a file store");
        assert_eq!(store.name(), "file");

        store.save(record()).await.unwrap();
        let root = workspace.path().join(CHECKPOINT_DIR);
        let expected = FileMemberCheckpointStore::new(&root).path_for("session-1:dev-1", "card-1");
        assert!(expected.exists(), "{expected:?}");
    }

    #[test]
    fn wire_runtime_pins_identity_policy_cwd_and_store() {
        let shared = InMemoryMemberCheckpointStore::shared();
        let runtime = wire_runtime(
            HarnessMemberRuntime::new(scripted_harness(), "session-1:dev-1"),
            "card-7",
            &enabled(),
            Some(&shared),
            Path::new("/tmp/ws"),
        );
        assert_eq!(runtime.card_id(), "card-7");
        assert_eq!(runtime.cwd(), Some("/tmp/ws"));
        assert_eq!(
            runtime.checkpoint_policy(),
            &MemberCheckpointPolicy::default()
        );
        let store = runtime.checkpoint_store().expect("the shared store");
        assert!(Arc::ptr_eq(store, &shared));
    }

    #[test]
    fn wire_runtime_attaches_no_store_under_a_disabled_policy() {
        let shared = InMemoryMemberCheckpointStore::shared();
        let runtime = wire_runtime(
            HarnessMemberRuntime::new(scripted_harness(), "session-1:dev-1"),
            "card-7",
            &disabled(),
            Some(&shared),
            Path::new("/tmp/ws"),
        );
        assert!(runtime.checkpoint_store().is_none());
        assert!(!runtime.checkpoint_policy().is_enabled());
        assert_eq!(runtime.card_id(), "card-7");
    }

    #[test]
    fn wire_runtime_keeps_the_member_session_id() {
        let runtime = wire_runtime(
            HarnessMemberRuntime::new(scripted_harness(), "session-1:dev-1"),
            "dev-1",
            &disabled(),
            None,
            Path::new("/tmp/ws"),
        );
        assert_eq!(
            runtime.session_id(),
            MemberHarnessContext::member_session_id("session-1", "dev-1")
        );
    }

    /// The smallest harness that satisfies the protocol: no output, no events.
    fn scripted_harness() -> Arc<dyn openjiuwen_harness_protocol::HarnessProtocol> {
        Arc::new(silent_harness::SilentHarness::new())
    }

    /// Local silent harness, mirroring the fakes of the sibling suites without their dependency.
    mod silent_harness {
        use async_trait::async_trait;
        use openjiuwen_harness_protocol::{
            AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessCheckpoint,
            HarnessContext, HarnessEventCursor, HarnessInput, HarnessProtocol, HarnessResult,
            HarnessState, SendReceipt, VecEventCursor,
        };
        use parking_lot::Mutex;

        /// A harness that accepts every command and emits nothing.
        pub(super) struct SilentHarness {
            /// State reported by the protocol.
            state: Mutex<HarnessState>,
            /// Static capability card.
            card: HarnessCard,
            /// Observation buffer policy.
            buffer: EventBufferConfig,
        }

        impl SilentHarness {
            /// Builds an idle harness.
            pub(super) fn new() -> Self {
                Self {
                    state: Mutex::new(HarnessState::Idle),
                    card: HarnessCard::new("silent", "0.0.0"),
                    buffer: EventBufferConfig::default(),
                }
            }
        }

        #[async_trait]
        #[allow(clippy::needless_pass_by_value)]
        impl HarnessProtocol for SilentHarness {
            fn card(&self) -> &HarnessCard {
                &self.card
            }

            fn state(&self) -> HarnessState {
                *self.state.lock()
            }

            fn provider_session_id(&self) -> Option<&str> {
                None
            }

            fn event_buffer_config(&self) -> &EventBufferConfig {
                &self.buffer
            }

            async fn start(&self, _context: HarnessContext) -> HarnessResult<()> {
                *self.state.lock() = HarnessState::Idle;
                Ok(())
            }

            async fn stop(&self) -> HarnessResult<()> {
                *self.state.lock() = HarnessState::Terminated;
                Ok(())
            }

            fn events(&self) -> Box<dyn HarnessEventCursor> {
                Box::new(VecEventCursor::new(Vec::new()))
            }

            fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
                Box::new(VecEventCursor::new(Vec::new()))
            }

            async fn send(
                &self,
                _input: HarnessInput,
                mode: DeliveryMode,
            ) -> HarnessResult<SendReceipt> {
                Ok(SendReceipt {
                    message_id: "message-1".to_owned(),
                    turn_id: "turn-1".to_owned(),
                    accepted_mode: mode,
                })
            }

            async fn abort(&self, _mode: AbortMode) -> HarnessResult<()> {
                Ok(())
            }

            async fn pause(&self) -> HarnessResult<()> {
                *self.state.lock() = HarnessState::Paused;
                Ok(())
            }

            async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
                *self.state.lock() = HarnessState::Running;
                Ok(())
            }

            async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
                Ok(None)
            }
        }
    }
}
