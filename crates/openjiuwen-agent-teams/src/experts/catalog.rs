//! The host-owned expert-locating seam and its process-scoped installation point.
//!
//! Python parity source: `openjiuwen/agent_teams/experts/catalog.py`.
//!
//! The catalog answers two questions only — *which* expert packages are installed, and *where*
//! their directories are — and never opens a manifest. Parsing is the job of
//! [`super::agent_group::load_agent_group`]; keeping the split means a host can expose a hub or a
//! local install directory without importing the package format.
//!
//! ```no_run
//! use std::sync::Arc;
//! use openjiuwen_agent_teams::experts::ExpertCatalog;
//!
//! # async fn locate(catalog: Arc<dyn ExpertCatalog>) -> openjiuwen_common::error::Result<()> {
//! if let Some(location) = catalog.locate_expert("writer").await? {
//!     assert_eq!(location.source, "builtin");
//! }
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// One installed package directory and the host install source it came from.
///
/// Python parity: `PackageLocation`. The path is serialized as a plain string, so a member row
/// that records a location round-trips through JSON unchanged.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PackageLocation {
    /// Package name for a single expert, or the group name for an `AgentGroup` package.
    pub name: String,
    /// Absolute package root directory.
    pub path: PathBuf,
    /// Host install source, for example `builtin`, `local` or `hub`.
    pub source: String,
}

impl PackageLocation {
    /// Builds a package location.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            source: source.into(),
        }
    }
}

/// Identity of one expert template.
///
/// A single expert is identified by its package name; a member of an `AgentGroup` only by
/// `(group, member name)` because member names are unique within their group alone.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExpertRef {
    /// Package name of a single expert, or member name inside [`ExpertRef::group`].
    pub name: String,
    /// `AgentGroup` name; `None` for a single expert.
    pub group: Option<String>,
}

impl ExpertRef {
    /// Builds a reference to a single (non-group) expert.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            group: None,
        }
    }

    /// Adds the `AgentGroup` name, turning the reference into a group-member reference.
    #[must_use]
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// Returns `group/name` for a group member and `name` for a single expert.
    #[must_use]
    pub fn label(&self) -> String {
        match self.group.as_deref() {
            Some(group) if !group.is_empty() => format!("{group}/{}", self.name),
            _ => self.name.clone(),
        }
    }

    /// Returns the JSON form persisted on a member row: `{"name": ..., "group": ...}`.
    #[must_use]
    pub fn to_value(&self) -> Value {
        json!({ "name": self.name, "group": self.group })
    }
}

/// Locates installed expert packages; async and read-only.
///
/// Implementations return only installed packages whose directory exists and never read or
/// validate manifest content — a broken package is reported by the loader that reads it.
/// File access must not block the event loop.
///
/// The locating methods return `Ok(None)` when the named package is not installed; Python raises
/// `ValueError`, which is represented here as the [`Error::NotFound`] kind only for callers that
/// choose to treat `None` as an error.
///
/// [`Error::NotFound`]: openjiuwen_common::error::Error::not_found
#[async_trait]
pub trait ExpertCatalog: Send + Sync {
    /// Every installed single-expert (`agent_template`) package.
    async fn list_expert_packages(&self) -> Result<Vec<PackageLocation>>;

    /// One installed single-expert package, or `None` when it is not installed.
    async fn locate_expert(&self, name: &str) -> Result<Option<PackageLocation>>;

    /// Every installed `AgentGroup` package.
    async fn list_group_packages(&self) -> Result<Vec<PackageLocation>>;

    /// One installed `AgentGroup` package, or `None` when it is not installed.
    async fn locate_group(&self, name: &str) -> Result<Option<PackageLocation>>;
}

/// The single process-wide catalog slot.
///
/// This static and the process-wide team registry are the crate's two documented singletons.
/// The catalog is process-global rather than carried on a spec because leaders are assembled
/// while the spec is built, before any runtime-manager hook could reach them, and because
/// [`crate::schema::TeamAgentSpec`] must stay serializable.
///
/// Wrapping the option in a lock (rather than using a bare [`OnceLock`] of the catalog) keeps the
/// slot resettable: tests install a fake catalog and call [`clear_process_expert_catalog`] when
/// they finish, so no catalog leaks between cases. The [`OnceLock`] only initializes the lock
/// itself.
static PROCESS_EXPERT_CATALOG: OnceLock<RwLock<Option<Arc<dyn ExpertCatalog>>>> = OnceLock::new();

fn catalog_slot() -> &'static RwLock<Option<Arc<dyn ExpertCatalog>>> {
    PROCESS_EXPERT_CATALOG.get_or_init(|| RwLock::new(None))
}

/// Installs this process's expert catalog, replacing any previous one.
///
/// Called through the runtime manager when the host injects its catalog. The catalog gates the
/// leader's expert tools: with no catalog installed those tools do not exist.
pub fn set_process_expert_catalog(catalog: Arc<dyn ExpertCatalog>) {
    *catalog_slot().write() = Some(catalog);
}

/// Returns this process's expert catalog, or `None` when the host injected none.
#[must_use]
pub fn get_process_expert_catalog() -> Option<Arc<dyn ExpertCatalog>> {
    catalog_slot().read().clone()
}

/// Clears this process's expert catalog.
///
/// Hosts and tests use this to restore the "no catalog installed" state.
pub fn clear_process_expert_catalog() {
    *catalog_slot().write() = None;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct FakeCatalog {
        experts: Vec<PackageLocation>,
        groups: Vec<PackageLocation>,
    }

    #[async_trait]
    impl ExpertCatalog for FakeCatalog {
        async fn list_expert_packages(&self) -> Result<Vec<PackageLocation>> {
            Ok(self.experts.clone())
        }

        async fn locate_expert(&self, name: &str) -> Result<Option<PackageLocation>> {
            Ok(self.experts.iter().find(|item| item.name == name).cloned())
        }

        async fn list_group_packages(&self) -> Result<Vec<PackageLocation>> {
            Ok(self.groups.clone())
        }

        async fn locate_group(&self, name: &str) -> Result<Option<PackageLocation>> {
            Ok(self.groups.iter().find(|item| item.name == name).cloned())
        }
    }

    #[test]
    fn expert_ref_label_and_value_match_python() {
        let single = ExpertRef::new("writer");
        assert_eq!(single.label(), "writer");
        assert_eq!(single.to_value(), json!({"name": "writer", "group": null}));

        let member = ExpertRef::new("narrative_designer").with_group("content");
        assert_eq!(member.label(), "content/narrative_designer");
        assert_eq!(
            member.to_value(),
            json!({"name": "narrative_designer", "group": "content"})
        );
        assert_eq!(
            serde_json::from_value::<ExpertRef>(member.to_value()).unwrap(),
            member
        );
    }

    #[test]
    fn package_location_serializes_its_path_as_a_string() {
        let location = PackageLocation::new("writer", "/opt/experts/writer", "builtin");
        let value = serde_json::to_value(&location).unwrap();
        assert_eq!(
            value,
            json!({
                "name": "writer",
                "path": "/opt/experts/writer",
                "source": "builtin",
            })
        );
        assert_eq!(
            serde_json::from_value::<PackageLocation>(value).unwrap(),
            location
        );
    }

    #[tokio::test]
    async fn process_catalog_set_get_and_clear() {
        clear_process_expert_catalog();
        assert!(get_process_expert_catalog().is_none());

        let catalog: Arc<dyn ExpertCatalog> = Arc::new(FakeCatalog {
            experts: vec![PackageLocation::new("writer", "/opt/experts/writer", "hub")],
            groups: Vec::new(),
        });
        set_process_expert_catalog(catalog);

        let installed = get_process_expert_catalog().expect("catalog installed");
        let located = installed.locate_expert("writer").await.unwrap();
        assert_eq!(located.map(|item| item.source), Some("hub".to_owned()));

        clear_process_expert_catalog();
        assert!(get_process_expert_catalog().is_none());
    }

    #[tokio::test]
    async fn fake_catalog_reports_missing_packages_as_none() {
        let catalog = FakeCatalog::default();
        let result = catalog.locate_group("absent").await.unwrap();
        assert!(result.is_none());
    }
}
