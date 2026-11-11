//! State stores and JSON path arithmetic for sessions.
//!
//! Ports `openjiuwen/core/session/state/` (v0.1.20): [`WorkflowState`] is the
//! Python `CommitState` with four lanes plus per-node staging, [`AgentState`] is
//! the agent/team state collection with `global_state` + `agent_state` lanes, and
//! [`paths`] holds the nested-path and merge helpers.
//!
//! Every store is synchronous and guarded by a single `parking_lot::RwLock`, so
//! callers never hold a lock across an `.await`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::Result;

pub mod agent;
pub mod paths;
pub mod workflow;

pub use agent::AgentState;
pub use workflow::WorkflowState;

/// Lane key for committed and staged IO state (`io_state`).
pub const IO_STATE_KEY: &str = "io_state";
/// Lane key for staged IO state (`io_state_updates`).
pub const IO_STATE_UPDATES_KEY: &str = "io_state_updates";
/// Lane key for global state (`global_state`).
pub const GLOBAL_STATE_KEY: &str = "global_state";
/// Lane key for staged global state (`global_state_updates`).
pub const GLOBAL_STATE_UPDATES_KEY: &str = "global_state_updates";
/// Lane key for component state (`comp_state`).
pub const COMP_STATE_KEY: &str = "comp_state";
/// Lane key for staged component state (`comp_state_updates`).
pub const COMP_STATE_UPDATES_KEY: &str = "comp_state_updates";
/// Lane key for workflow state (`workflow_state`).
pub const WORKFLOW_STATE_KEY: &str = "workflow_state";
/// Lane key for staged workflow state (`workflow_state_updates`).
pub const WORKFLOW_STATE_UPDATES_KEY: &str = "workflow_state_updates";
/// Lane key for trace state (`trace_state`).
pub const TRACE_STATE_KEY: &str = "trace_state";
/// Lane key for agent state (`agent_state`).
pub const AGENT_STATE_KEY: &str = "agent_state";

/// The named state lane a store exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateLane {
    /// Input/output lane (`io_state`).
    Io,
    /// Global lane (`global_state`).
    Global,
    /// Component lane (`comp_state`).
    Comp,
    /// Workflow lane (`workflow_state`).
    Workflow,
    /// Trace lane (`trace_state`).
    Trace,
}

impl StateLane {
    /// Returns the lane's `snake_case` key, as used in `dump()` output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Io => IO_STATE_KEY,
            Self::Global => GLOBAL_STATE_KEY,
            Self::Comp => COMP_STATE_KEY,
            Self::Workflow => WORKFLOW_STATE_KEY,
            Self::Trace => TRACE_STATE_KEY,
        }
    }
}

/// The storage seam every session uses to read and write state.
///
/// Implementors are synchronous and internally synchronized, so all methods may
/// be called from async code without holding a lock across an `.await`.
/// Component ids are nested paths (`a.b` addresses `{"a": {"b": ...}}`).
pub trait StateStore: Send + Sync + std::fmt::Debug {
    /// Reads the value at `key` from the global lane.
    ///
    /// An empty key returns the whole global object when the store exposes one.
    fn get_global(&self, key: &str) -> Option<Value>;

    /// Stages or writes an object `patch` into the global lane.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when `patch` is
    /// not a JSON object or a key is not a valid nested path.
    fn update_global(&self, patch: Value) -> Result<()>;

    /// Merges an object `patch` into the committed global lane, bypassing any
    /// staging queue.
    ///
    /// The default forwards to [`StateStore::update_global`]; stores with a
    /// staging lane ([`WorkflowState`]) override it so a session-level update
    /// is immediately visible to [`StateStore::get_global`].
    ///
    /// # Errors
    ///
    /// Same as [`StateStore::update_global`].
    fn write_global(&self, patch: Value) -> Result<()> {
        self.update_global(patch)
    }

    /// Reads component state by nested component id.
    ///
    /// Returns `None` for an empty id or a missing component.
    fn get_component(&self, component_id: &str) -> Option<Value>;

    /// Writes an object `patch` into the component lane under `component_id`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when
    /// `component_id` is empty or the write cannot be applied.
    fn update_component(&self, component_id: &str, patch: Value) -> Result<()>;

    /// Returns the whole committed component lane object.
    fn components(&self) -> Value;

    /// Returns the store's full lane snapshot (`dump()` shape).
    fn dump(&self) -> Value;

    /// Replaces the global and component lanes from a snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when either
    /// argument is not a JSON object.
    fn restore(&self, global: Value, components: Value) -> Result<()>;

    /// Returns the trace lane.
    fn trace_state(&self) -> Value;

    /// Merges an object `patch` into the trace lane.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when `patch` is
    /// not a JSON object or a key is not a valid nested path.
    fn update_trace(&self, patch: Value) -> Result<()>;
}

/// Returns `patch` as a JSON object map, or a validation error naming `what`.
pub(crate) fn ensure_object(patch: Value, what: &str) -> Result<Map<String, Value>> {
    match patch {
        Value::Object(map) => Ok(map),
        _ => Err(openjiuwen_common::error::Error::validation(format!(
            "{what} must be a JSON object"
        ))),
    }
}
