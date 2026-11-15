//! Disk-backed product sessions: scope-isolated session storage and links.
//!
//! Ports the storage subset of `openjiuwen/core/session/session_controller/`
//! (v0.1.20): [`SessionScope`]/[`SessionScopeKey`] isolation keys (with
//! [`Scope`] and [`Subject`]), the `sessions.json`/`state.data` schema
//! ([`SessionMeta`], [`SessionRecord`]), the [`DiskSessionStore`] layout, and
//! the per-agent in-memory [`SessionController`] cache.
//!
//! The persisted `data` document is exactly the crate's
//! [`crate::SessionSnapshot`] JSON, so a stored record loads directly into a
//! concrete session through [`crate::Session::restore`].
//!
//! The session link graph ([`SessionLinks`], persisted as each agent's
//! `<base>/<agent_id>/links.json`) records how sessions descend from and fork
//! each other and traverses a session's lineage.
//! It is graph metadata only: state propagation belongs to the crates that own a
//! run loop, not to this storage layer.
//!
//! Deferred from the Python package: data-container/permission policies and the
//! process-global `GlobalSessionController` singleton.
//!
//! ```
//! use openjiuwen_session::session_controller::{Scope, SessionScope};
//!
//! let scope = SessionScope::custom("tenant-a");
//! assert_eq!(scope.scope, Scope::Custom("tenant-a".to_string()));
//! ```
//!
//! ```
//! use openjiuwen_session::session_controller::SessionLinks;
//!
//! let mut links = SessionLinks::new();
//! links.record_parent("child", "root").unwrap();
//! assert_eq!(links.chain("child").unwrap(), ["root", "child"]);
//! ```
pub mod links;
pub mod schema;
pub mod scope;
pub mod store;

pub use links::{LinkKind, LinksIndex, SessionLink, SessionLinks};
pub use schema::{
    ScopeSessionsMeta, SessionMeta, SessionRecord, SessionsIndex, DEFAULT_DATA_CONTAINER_TYPE,
};
pub use scope::{Scope, SessionScope, SessionScopeKey, Subject};
pub use store::{DiskSessionStore, SessionController};
