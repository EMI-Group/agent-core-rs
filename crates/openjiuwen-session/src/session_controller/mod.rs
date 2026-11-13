//! Disk-backed product sessions: scope-isolated session storage.
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
//! Deferred from the Python package: the downstream link graph
//! (`downstreams/`, `ChainSession`), data-container/permission policies, and the
//! process-global `GlobalSessionController` singleton.
//!
//! ```
//! use openjiuwen_session::session_controller::{Scope, SessionScope};
//!
//! let scope = SessionScope::custom("tenant-a");
//! assert_eq!(scope.scope, Scope::Custom("tenant-a".to_string()));
//! ```
pub mod schema;
pub mod scope;
pub mod store;

pub use schema::{
    ScopeSessionsMeta, SessionMeta, SessionRecord, SessionsIndex, DEFAULT_DATA_CONTAINER_TYPE,
};
pub use scope::{Scope, SessionScope, SessionScopeKey, Subject};
pub use store::{DiskSessionStore, SessionController};
