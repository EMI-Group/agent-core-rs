//! `AgentState` — the Python agent/team `StateCollection` port.
//!
//! Two lanes, `global_state` and `agent_state`, with no staging: updates apply
//! immediately, and a cloned store shares both lanes through one
//! `Arc<RwLock<Inner>>`. The trace lane is stored separately and is not part of
//! [`AgentState::dump`].
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::state::ensure_object;
use crate::state::paths;
use crate::state::{StateStore, AGENT_STATE_KEY, GLOBAL_STATE_KEY};

/// The shared lanes behind every [`AgentState`] view.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
struct Inner {
    global_state: Value,
    agent_state: Value,
    trace_state: Value,
}

impl Default for Inner {
    fn default() -> Self {
        Self {
            global_state: Value::Object(Map::new()),
            agent_state: Value::Object(Map::new()),
            trace_state: Value::Object(Map::new()),
        }
    }
}

/// Agent/team state: a `global_state` lane plus an `agent_state` lane.
///
/// Ports `openjiuwen/core/session/state/agent_state.py`. Unlike
/// [`crate::state::WorkflowState`] there is no per-node staging; every update is
/// applied immediately.
#[derive(Debug, Clone)]
pub struct AgentState {
    inner: Arc<RwLock<Inner>>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentState {
    /// Creates an empty agent state store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
        }
    }

    /// Reads the global lane at `key`; an empty key returns the whole lane.
    #[must_use]
    pub fn get_global(&self, key: &str) -> Option<Value> {
        let inner = self.inner.read();
        if key.is_empty() {
            return Some(inner.global_state.clone());
        }
        paths::get_value_cloned(&inner.global_state, key)
    }

    /// Merges an object `patch` into the global lane.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object or a key
    /// is not a valid nested path.
    pub fn update_global(&self, patch: Value) -> Result<()> {
        self.apply_global(patch)
    }

    /// Reads the agent lane at `key`; an empty key returns the whole lane.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Value> {
        let inner = self.inner.read();
        if key.is_empty() {
            return Some(inner.agent_state.clone());
        }
        paths::get_value_cloned(&inner.agent_state, key)
    }

    /// Merges an object `patch` into the agent lane.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object or a key
    /// is not a valid nested path.
    pub fn update(&self, patch: Value) -> Result<()> {
        self.apply_agent(patch)
    }

    /// Returns a deep copy of the whole agent lane.
    #[must_use]
    pub fn agent_state(&self) -> Value {
        self.inner.read().agent_state.clone()
    }

    /// Merges an object `patch` into the agent lane (alias of
    /// [`AgentState::update`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object or a key
    /// is not a valid nested path.
    pub fn update_agent(&self, patch: Value) -> Result<()> {
        self.apply_agent(patch)
    }

    /// Returns the trace lane.
    #[must_use]
    pub fn trace_state(&self) -> Value {
        self.trace_value()
    }

    /// Merges an object `patch` into the trace lane.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object or a key
    /// is not a valid nested path.
    pub fn update_trace(&self, patch: Value) -> Result<()> {
        self.patch_trace(patch)
    }

    /// Returns `{global_state, agent_state}`; the trace lane is never a key here.
    #[must_use]
    pub fn dump(&self) -> Value {
        self.dump_value()
    }

    /// Merges a patch into the global lane.
    fn apply_global(&self, patch: Value) -> Result<()> {
        let patch = ensure_object(patch, "global state patch")?;
        let mut inner = self.inner.write();
        paths::apply_patch(&mut inner.global_state, &Value::Object(patch))
    }

    /// Merges a patch into the agent lane.
    fn apply_agent(&self, patch: Value) -> Result<()> {
        let patch = ensure_object(patch, "agent state patch")?;
        let mut inner = self.inner.write();
        paths::apply_patch(&mut inner.agent_state, &Value::Object(patch))
    }

    /// Reads committed agent state by nested component id.
    fn read_component(&self, component_id: &str) -> Option<Value> {
        let inner = self.inner.read();
        paths::get_value_cloned(&inner.agent_state, component_id)
    }

    /// Writes committed agent state by nested component id.
    fn write_component(&self, component_id: &str, patch: Value) -> Result<()> {
        if component_id.is_empty() {
            return Err(Error::validation("component id must not be empty"));
        }
        let mut inner = self.inner.write();
        let mut wrapper = Map::new();
        wrapper.insert(component_id.to_string(), patch);
        paths::apply_patch(&mut inner.agent_state, &Value::Object(wrapper))
    }

    /// Replaces the global and agent lanes.
    fn restore_lanes(&self, global: Value, components: Value) -> Result<()> {
        if !global.is_object() {
            return Err(Error::validation(
                "restored global state must be a JSON object",
            ));
        }
        if !components.is_object() {
            return Err(Error::validation(
                "restored agent state must be a JSON object",
            ));
        }
        let mut inner = self.inner.write();
        inner.global_state = global;
        inner.agent_state = components;
        Ok(())
    }

    /// Returns the trace lane value.
    fn trace_value(&self) -> Value {
        self.inner.read().trace_state.clone()
    }

    /// Merges an object patch into the trace lane.
    fn patch_trace(&self, patch: Value) -> Result<()> {
        let patch = ensure_object(patch, "trace state patch")?;
        let mut inner = self.inner.write();
        paths::apply_patch(&mut inner.trace_state, &Value::Object(patch))
    }

    /// Returns the two dump lanes.
    fn dump_value(&self) -> Value {
        let inner = self.inner.read();
        let mut dump = Map::new();
        dump.insert(GLOBAL_STATE_KEY.to_string(), inner.global_state.clone());
        dump.insert(AGENT_STATE_KEY.to_string(), inner.agent_state.clone());
        Value::Object(dump)
    }
}

impl StateStore for AgentState {
    fn get_global(&self, key: &str) -> Option<Value> {
        let inner = self.inner.read();
        if key.is_empty() {
            return Some(inner.global_state.clone());
        }
        paths::get_value_cloned(&inner.global_state, key)
    }

    fn update_global(&self, patch: Value) -> Result<()> {
        self.apply_global(patch)
    }

    fn get_component(&self, component_id: &str) -> Option<Value> {
        if component_id.is_empty() {
            return None;
        }
        self.read_component(component_id)
    }

    fn update_component(&self, component_id: &str, patch: Value) -> Result<()> {
        self.write_component(component_id, patch)
    }

    fn components(&self) -> Value {
        self.inner.read().agent_state.clone()
    }

    fn dump(&self) -> Value {
        self.dump_value()
    }

    fn restore(&self, global: Value, components: Value) -> Result<()> {
        self.restore_lanes(global, components)
    }

    fn trace_state(&self) -> Value {
        self.trace_value()
    }

    fn update_trace(&self, patch: Value) -> Result<()> {
        self.patch_trace(patch)
    }
}
