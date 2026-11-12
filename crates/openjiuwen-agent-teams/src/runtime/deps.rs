//! The dependencies a team runtime is built from.
//!
//! [`TeamBuildDeps`] is the single injection point of the runtime: a harness factory that
//! materializes one harness per member, the persisted-state store, the workspace provider and the
//! team language. The Python reference reaches the same objects through module-level imports and
//! `TeamAgent.__init__` keyword arguments; this port threads them explicitly so a test can build a
//! runtime without touching the environment or the filesystem.
//!
//! Python parity source: `openjiuwen/agent_teams/team_agent.py` (`TeamAgent.__init__` builds the
//! database, the transport and the harness factory) and `openjiuwen/agent_teams/models/`
//! (the pool that feeds the harness model).
//!
//! Every field has a `with_*` override and a default that works in-process, so
//!
//! ```
//! use openjiuwen_agent_teams::runtime::deps::TeamBuildDeps;
//!
//! let deps = TeamBuildDeps::new().with_language("en");
//! assert_eq!(deps.resolved_language(), "en");
//! assert_eq!(TeamBuildDeps::new().resolved_language(), "cn");
//! ```
use std::fmt;
use std::sync::Arc;

use crate::harness::{MemberHarnessFactory, NativeMemberFactory};
use crate::schema::Language;
use crate::storage::{InMemoryTeamStore, TeamStore};
use crate::workspace::{LocalWorkspaceProvider, MemberWorkspaceLayout, WorkspaceProvider};

/// Directory under the system temporary directory that backs the default workspace provider.
const DEFAULT_WORKSPACE_DIR: &str = "openjiuwen-agent-teams";

/// Everything a [`TeamAgent`](super::team_agent::TeamAgent) needs from its host.
///
/// The struct is cheap to clone: every heavy member is behind an [`Arc`] and the language is a
/// short string. [`TeamBuildDeps::default`] wires the in-process implementations, so a runtime
/// built from `TeamBuildDeps::new()` runs entirely in memory with member workspaces under the
/// system temporary directory.
///
/// The Python runtime reads the same four inputs from `TeamAgentSpec` and the environment; here
/// they are injected so tests never depend on either.
#[derive(Clone)]
pub struct TeamBuildDeps {
    /// Builds one harness per member (Python: the `HarnessProtocol` factory).
    pub harness_factory: Arc<dyn MemberHarnessFactory>,
    /// Persists the team row, roster, task board and mailbox.
    pub store: Arc<dyn TeamStore>,
    /// Allocates the directory a member writes artifacts into.
    pub workspace: Arc<dyn WorkspaceProvider>,
    /// Declared team language (`cn`/`cf`/`en`); `None` means the schema default.
    pub language: Option<String>,
}

impl TeamBuildDeps {
    /// Builds the in-process defaults; identical to [`TeamBuildDeps::default`].
    ///
    /// The default factory is [`NativeMemberFactory::from_env`] (models resolved from
    /// `MODEL_PROVIDER`/`MODEL_NAME`), the store is [`InMemoryTeamStore`], the workspace provider is
    /// [`LocalWorkspaceProvider`] rooted at `<temp>/openjiuwen-agent-teams`, and no language is
    /// declared.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the harness factory, leaving every other field untouched.
    #[must_use]
    pub fn with_harness_factory(self, factory: Arc<dyn MemberHarnessFactory>) -> Self {
        Self {
            harness_factory: factory,
            ..self
        }
    }

    /// Replaces the store, leaving every other field untouched.
    #[must_use]
    pub fn with_store(self, store: Arc<dyn TeamStore>) -> Self {
        Self { store, ..self }
    }

    /// Replaces the workspace provider, leaving every other field untouched.
    #[must_use]
    pub fn with_workspace(self, workspace: Arc<dyn WorkspaceProvider>) -> Self {
        Self { workspace, ..self }
    }

    /// Declares the team language, leaving every other field untouched.
    ///
    /// The value is not validated here: an unknown string resolves to the schema default in
    /// [`TeamBuildDeps::resolved_language`], mirroring Python's tolerant `resolve_language`.
    #[must_use]
    pub fn with_language(self, language: impl Into<String>) -> Self {
        Self {
            language: Some(language.into()),
            ..self
        }
    }

    /// Resolves the declared language to its wire value, falling back to `cn`.
    ///
    /// Python parity: `resolve_language`, which treats an absent or unknown value as the default
    /// rather than failing, so a spec written against a newer catalogue still loads.
    #[must_use]
    pub fn resolved_language(&self) -> &'static str {
        Language::resolve(self.language.as_deref())
    }
}

impl Default for TeamBuildDeps {
    fn default() -> Self {
        Self {
            harness_factory: Arc::new(NativeMemberFactory::from_env()),
            store: Arc::new(InMemoryTeamStore::new()),
            workspace: Arc::new(LocalWorkspaceProvider::new(MemberWorkspaceLayout::new(
                std::env::temp_dir().join(DEFAULT_WORKSPACE_DIR),
            ))),
            language: None,
        }
    }
}

impl fmt::Debug for TeamBuildDeps {
    /// Prints the two named seams by identity rather than dumping their internals: a harness
    /// factory and a store are opaque trait objects, and their `name()` is the stable identity the
    /// runtime logs. Neither carries credentials, so this is safe to log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeamBuildDeps")
            .field("harness_factory", &self.harness_factory.name())
            .field("store", &self.store.name())
            .field("workspace", &"injected")
            .field("language", &self.language)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_harness_protocol::HarnessProtocol;

    use super::*;
    use crate::harness::{MemberHarnessContext, NATIVE_FACTORY_NAME};
    use crate::schema::DeepAgentSpec;
    use crate::storage::MEMORY_STORAGE_KIND;

    /// A factory that never builds anything, used to prove the `Debug`/`name()` wiring.
    struct FakeFactory;

    #[async_trait::async_trait]
    impl MemberHarnessFactory for FakeFactory {
        async fn create(
            &self,
            _spec: &DeepAgentSpec,
            _ctx: &MemberHarnessContext,
        ) -> Result<Arc<dyn HarnessProtocol>> {
            Err(Error::Unsupported("fake harness factory".to_owned()))
        }

        fn name(&self) -> &'static str {
            "fake"
        }
    }

    #[test]
    fn default_wiring_is_in_process() {
        let deps = TeamBuildDeps::new();
        assert_eq!(deps.harness_factory.name(), NATIVE_FACTORY_NAME);
        assert_eq!(deps.harness_factory.name(), "native");
        assert_eq!(deps.store.name(), MEMORY_STORAGE_KIND);
        assert_eq!(deps.store.name(), "memory");
        assert_eq!(deps.language, None);
        assert_eq!(deps.resolved_language(), "cn");
    }

    #[test]
    fn with_harness_factory_overrides_only_the_factory() {
        let base = TeamBuildDeps::default();
        let deps = base.clone().with_harness_factory(Arc::new(FakeFactory));
        assert_eq!(deps.harness_factory.name(), "fake");
        assert!(Arc::ptr_eq(&deps.store, &base.store));
        assert!(Arc::ptr_eq(&deps.workspace, &base.workspace));
        assert_eq!(deps.language, base.language);
        assert!(format!("{deps:?}").contains("fake"));
    }

    #[test]
    fn with_store_overrides_only_the_store() {
        let base = TeamBuildDeps::default();
        let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
        let deps = base.clone().with_store(Arc::clone(&store));
        assert!(Arc::ptr_eq(&deps.store, &store));
        assert!(!Arc::ptr_eq(&deps.store, &base.store));
        assert!(Arc::ptr_eq(&deps.harness_factory, &base.harness_factory));
        assert!(Arc::ptr_eq(&deps.workspace, &base.workspace));
        assert_eq!(deps.language, base.language);
    }

    #[test]
    fn with_workspace_overrides_only_the_workspace() {
        let base = TeamBuildDeps::default();
        let workspace: Arc<dyn WorkspaceProvider> = Arc::new(LocalWorkspaceProvider::new(
            MemberWorkspaceLayout::new(std::env::temp_dir().join("other-base")),
        ));
        let deps = base.clone().with_workspace(Arc::clone(&workspace));
        assert!(Arc::ptr_eq(&deps.workspace, &workspace));
        assert!(!Arc::ptr_eq(&deps.workspace, &base.workspace));
        assert!(Arc::ptr_eq(&deps.harness_factory, &base.harness_factory));
        assert!(Arc::ptr_eq(&deps.store, &base.store));
        assert_eq!(deps.language, base.language);
    }

    #[test]
    fn with_language_overrides_only_the_language() {
        let base = TeamBuildDeps::default();
        let deps = base.clone().with_language("en");
        assert_eq!(deps.language.as_deref(), Some("en"));
        assert_eq!(deps.resolved_language(), "en");
        assert!(Arc::ptr_eq(&deps.harness_factory, &base.harness_factory));
        assert!(Arc::ptr_eq(&deps.store, &base.store));
        assert!(Arc::ptr_eq(&deps.workspace, &base.workspace));
    }

    #[test]
    fn resolved_language_accepts_wire_values_and_falls_back() {
        let cases = [
            (None, "cn"),
            (Some("cn"), "cn"),
            (Some("cf"), "cf"),
            (Some("en"), "en"),
            (Some("bogus"), "cn"),
            (Some("EN"), "cn"),
            (Some(""), "cn"),
        ];
        for (declared, expected) in cases {
            let deps = TeamBuildDeps {
                language: declared.map(str::to_owned),
                ..TeamBuildDeps::default()
            };
            assert_eq!(deps.resolved_language(), expected, "declared {declared:?}");
        }
    }

    #[test]
    fn debug_names_the_seams_without_leaking_payloads() {
        let debug = format!("{:?}", TeamBuildDeps::default());
        assert!(debug.starts_with("TeamBuildDeps"));
        assert!(debug.contains("native"), "{debug}");
        assert!(debug.contains("memory"), "{debug}");
        assert!(debug.contains("injected"), "{debug}");
        assert!(debug.contains("language: None"), "{debug}");
    }
}
