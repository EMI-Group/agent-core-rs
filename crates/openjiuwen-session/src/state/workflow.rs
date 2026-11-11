//! `WorkflowState` — the Python `CommitState` port.
//!
//! Holds the four committed lanes (`io_state`, `global_state`, `comp_state`,
//! `workflow_state`) plus a plain-JSON `trace_state`, each with its own per-node
//! staging queue. Clones and node views share one `Arc<RwLock<Inner>>`, so
//! [`WorkflowState::node`] is O(1) and every view sees the same lanes and
//! pending updates.
//!
//! Component ids are nested paths: a component id `a.b` is stored as
//! `{"a": {"b": ...}}`, matching Python's `executable_id` nesting.
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::constants::DEFAULT_NODE_ID;
use crate::state::ensure_object;
use crate::state::paths;
use crate::state::{
    StateLane, StateStore, COMP_STATE_KEY, COMP_STATE_UPDATES_KEY, GLOBAL_STATE_KEY,
    GLOBAL_STATE_UPDATES_KEY, IO_STATE_KEY, IO_STATE_UPDATES_KEY, TRACE_STATE_KEY,
    WORKFLOW_STATE_KEY, WORKFLOW_STATE_UPDATES_KEY,
};

/// The shared lanes and staging queues behind every [`WorkflowState`] view.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
struct Inner {
    io_state: Value,
    global_state: Value,
    comp_state: Value,
    workflow_state: Value,
    trace_state: Value,
    io_updates: BTreeMap<String, Vec<Value>>,
    global_updates: BTreeMap<String, Vec<Value>>,
    comp_updates: BTreeMap<String, Vec<Value>>,
    workflow_updates: BTreeMap<String, Vec<Value>>,
}

impl Default for Inner {
    fn default() -> Self {
        Self {
            io_state: Value::Object(Map::new()),
            global_state: Value::Object(Map::new()),
            comp_state: Value::Object(Map::new()),
            workflow_state: Value::Object(Map::new()),
            trace_state: Value::Object(Map::new()),
            io_updates: BTreeMap::new(),
            global_updates: BTreeMap::new(),
            comp_updates: BTreeMap::new(),
            workflow_updates: BTreeMap::new(),
        }
    }
}

/// Workflow state with four committed lanes, per-node staging, and a trace lane.
///
/// This is the Rust port of Python's `CommitState`/`InMemoryState`. Clones share
/// all lanes by reference; [`WorkflowState::node`] creates a child view scoped to
/// a node id and parent id.
#[derive(Debug, Clone)]
pub struct WorkflowState {
    inner: Arc<RwLock<Inner>>,
    node_id: String,
    parent_id: String,
    workflow_only: bool,
}

impl Default for WorkflowState {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkflowState {
    /// Creates a workflow-only store: all lanes are empty and there is no
    /// visible global lane.
    #[must_use]
    pub fn new() -> Self {
        Self::with_global_lane(true)
    }

    /// Creates a store, choosing whether it is workflow-only.
    ///
    /// Mirrors Python `InMemoryState(global_state=...)`: `workflow_only == true`
    /// means the global lane is absent from whole-object reads (a workflow with
    /// no outer agent/team global state); `false` exposes it. `dump()` always
    /// includes the `global_state` key either way.
    #[must_use]
    pub fn with_global_lane(workflow_only: bool) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
            node_id: DEFAULT_NODE_ID.to_string(),
            parent_id: String::new(),
            workflow_only,
        }
    }

    /// Returns whether this store is workflow-only (global lane hidden).
    #[must_use]
    pub fn workflow_only(&self) -> bool {
        self.workflow_only
    }

    /// Creates an O(1) node view sharing this store's lanes and staging queues.
    #[must_use]
    pub fn node(&self, node_id: &str, parent_id: &str) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            node_id: node_id.to_string(),
            parent_id: parent_id.to_string(),
            workflow_only: self.workflow_only,
        }
    }

    /// Returns a deep copy of one committed lane.
    ///
    /// The `Global` lane is returned even for workflow-only stores; use
    /// [`WorkflowState::get_global`] for the public visibility rule.
    #[must_use]
    pub fn lane(&self, lane: StateLane) -> Value {
        let inner = self.inner.read();
        match lane {
            StateLane::Io => inner.io_state.clone(),
            StateLane::Global => inner.global_state.clone(),
            StateLane::Comp => inner.comp_state.clone(),
            StateLane::Workflow => inner.workflow_state.clone(),
            StateLane::Trace => inner.trace_state.clone(),
        }
    }

    /// Reads component state for this node from the component lane.
    ///
    /// An empty key returns this node's whole subtree; a non-empty key is
    /// resolved relative to that subtree. Returns `None` when absent.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Value> {
        let inner = self.inner.read();
        if key.is_empty() {
            return paths::get_value_cloned(&inner.comp_state, &self.node_id);
        }
        let scope = paths::get_by_prefix(&inner.comp_state, &self.node_id);
        paths::get_value_cloned(&scope, key)
    }

    /// Reads component state for this node as a [`Value`], with [`Value::Null`]
    /// for a missing path.
    ///
    /// This is the lossless counterpart of [`WorkflowState::get`]; an empty key
    /// returns this node's whole component subtree.
    #[must_use]
    pub fn get_by_prefix(&self, key: &str) -> Value {
        let inner = self.inner.read();
        let scope = paths::get_by_prefix(&inner.comp_state, &self.node_id);
        if key.is_empty() {
            scope
        } else {
            paths::get_by_prefix(&scope, key)
        }
    }

    /// Reads global state, falling back to the IO lane at the parent and then
    /// the node id (Python `StateCollection.get_global`).
    ///
    /// An empty key returns the whole global object, or `None` for a
    /// workflow-only store.
    #[must_use]
    pub fn get_global(&self, key: &str) -> Option<Value> {
        self.read_global(key)
    }

    /// Reads workflow-lane state at `key`; an empty key returns the whole lane.
    #[must_use]
    pub fn get_workflow_state(&self, key: &str) -> Option<Value> {
        let inner = self.inner.read();
        if key.is_empty() {
            return Some(inner.workflow_state.clone());
        }
        paths::get_value_cloned(&inner.workflow_state, key)
    }

    /// Returns this node's inputs: the IO lane subtree at the node id.
    #[must_use]
    pub fn inputs(&self) -> Value {
        let inner = self.inner.read();
        paths::get_by_prefix(&inner.io_state, &self.node_id)
    }

    /// Returns the outputs of `node_id` (or this node) scoped under the parent id.
    ///
    /// Resolves the parent subtree first and then `node_id`, matching Python
    /// `get_outputs`. Missing data yields [`Value::Null`].
    #[must_use]
    pub fn outputs(&self, node_id: Option<&str>) -> Value {
        let inner = self.inner.read();
        let target = node_id.filter(|id| !id.is_empty()).unwrap_or(&self.node_id);
        let scope = paths::get_by_prefix(&inner.io_state, &self.parent_id);
        if target.is_empty() {
            scope
        } else {
            paths::get_by_prefix(&scope, target)
        }
    }

    /// Projects `schema` against this node's IO and component state.
    ///
    /// Porcelain over Python `utils.get_by_schema`: the source is the IO subtree
    /// at the node id with the component subtree at the node id merged over it.
    /// A string is either a `${...}` reference (origin key lookup) or, at the top
    /// level, a literal nested path; the same string nested inside an object or
    /// list passes through unchanged. A list projects each item; an object maps
    /// each entry, recursing only into lists, objects, and `${...}` strings and
    /// passing other scalars through.
    #[must_use]
    pub fn get_by_schema(&self, schema: &Value) -> Value {
        let inner = self.inner.read();
        let mut data = paths::get_by_prefix(&inner.io_state, &self.node_id);
        let comp = paths::get_by_prefix(&inner.comp_state, &self.node_id);
        if data.is_object() && comp.is_object() {
            openjiuwen_common::utils::json::merge(&mut data, comp);
        } else if !comp.is_null() {
            data = comp;
        }
        project_schema(schema, &data, true)
    }

    /// Stages `patch` in the component lane under this node id.
    ///
    /// The queued entry is shaped `{node_id: patch}`, so committing nests the
    /// patch under the node id exactly like Python `StateCollection.update`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object.
    pub fn update(&self, patch: Value) -> Result<()> {
        let patch = ensure_object(patch, "component state patch")?;
        let mut inner = self.inner.write();
        inner
            .comp_updates
            .entry(self.node_id.clone())
            .or_default()
            .push(single_key_object(&self.node_id, Value::Object(patch)));
        Ok(())
    }

    /// Stages `patch` in the global lane under this node id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object.
    pub fn update_global(&self, patch: Value) -> Result<()> {
        self.stage_global(patch)
    }

    /// Stages `patch` in the workflow lane under this node id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object.
    pub fn update_workflow(&self, patch: Value) -> Result<()> {
        let patch = ensure_object(patch, "workflow state patch")?;
        let mut inner = self.inner.write();
        inner
            .workflow_updates
            .entry(self.node_id.clone())
            .or_default()
            .push(Value::Object(patch));
        Ok(())
    }

    /// Stages `patch` in the component lane, nested under `component_id`.
    ///
    /// The staging key is the component id and the queued entry is
    /// `{component_id: patch}`; committing applies it as a nested path, so
    /// `a.b` lands at `{"a": {"b": ...}}`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `component_id` is empty or `patch` is
    /// not a JSON object.
    pub fn update_component(&self, component_id: &str, patch: Value) -> Result<()> {
        self.stage_component(component_id, patch)
    }

    /// Commits staged updates, draining the queues in insertion order per node.
    ///
    /// `node_id == None` commits every queued node (Python `commit()`); otherwise
    /// only that node's queues are drained. Each patch is applied to its lane with
    /// [`paths::apply_patch`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a staged patch cannot be applied (for
    /// example a malformed path key).
    pub fn commit(&self, node_id: Option<&str>) -> Result<()> {
        let mut inner = self.inner.write();
        let io = take_queue(&mut inner.io_updates, node_id);
        let comp = take_queue(&mut inner.comp_updates, node_id);
        let global = take_queue(&mut inner.global_updates, node_id);
        let workflow = take_queue(&mut inner.workflow_updates, node_id);
        apply_queues(&mut inner.io_state, io)?;
        apply_queues(&mut inner.comp_state, comp)?;
        apply_queues(&mut inner.global_state, global)?;
        apply_queues(&mut inner.workflow_state, workflow)?;
        Ok(())
    }

    /// Discards staged updates for `node_id`, or for every node when `None`.
    pub fn rollback(&self, node_id: Option<&str>) {
        let mut inner = self.inner.write();
        if let Some(id) = node_id {
            inner.io_updates.remove(id);
            inner.comp_updates.remove(id);
            inner.global_updates.remove(id);
            inner.workflow_updates.remove(id);
        } else {
            inner.io_updates.clear();
            inner.comp_updates.clear();
            inner.global_updates.clear();
            inner.workflow_updates.clear();
        }
    }

    /// Returns the staged updates as `{lane_updates_key: {node_id: [patch, ...]}}`.
    #[must_use]
    pub fn updates(&self) -> Value {
        let inner = self.inner.read();
        let mut updates = Map::new();
        updates.insert(
            IO_STATE_UPDATES_KEY.to_string(),
            queue_to_value(&inner.io_updates),
        );
        updates.insert(
            GLOBAL_STATE_UPDATES_KEY.to_string(),
            queue_to_value(&inner.global_updates),
        );
        updates.insert(
            COMP_STATE_UPDATES_KEY.to_string(),
            queue_to_value(&inner.comp_updates),
        );
        updates.insert(
            WORKFLOW_STATE_UPDATES_KEY.to_string(),
            queue_to_value(&inner.workflow_updates),
        );
        Value::Object(updates)
    }

    /// Replaces staging queues from an [`WorkflowState::updates`] shaped value.
    ///
    /// Lanes absent from `updates` are left untouched; a `null` lane clears it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `updates` is not a JSON object, or a
    /// present lane is neither an object nor `null`, or a node's entry is not an
    /// array of patches.
    pub fn set_updates(&self, updates: &Value) -> Result<()> {
        let Some(updates) = updates.as_object() else {
            return Err(Error::validation("state updates must be a JSON object"));
        };
        let mut inner = self.inner.write();
        set_queue(&mut inner.io_updates, updates.get(IO_STATE_UPDATES_KEY))?;
        set_queue(
            &mut inner.global_updates,
            updates.get(GLOBAL_STATE_UPDATES_KEY),
        )?;
        set_queue(&mut inner.comp_updates, updates.get(COMP_STATE_UPDATES_KEY))?;
        set_queue(
            &mut inner.workflow_updates,
            updates.get(WORKFLOW_STATE_UPDATES_KEY),
        )?;
        Ok(())
    }

    /// Writes `data` directly into the IO lane under `node_id` (not staged).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `node_id` is empty or `data` cannot be
    /// applied (for example a malformed path key).
    pub fn set_outputs(&self, node_id: &str, data: Value) -> Result<()> {
        let mut inner = self.inner.write();
        let patch = single_key_object(node_id, data);
        paths::apply_patch(&mut inner.io_state, &patch)
    }

    /// Writes user inputs directly into the IO and global lanes (Python
    /// `commit_user_inputs`).
    ///
    /// For the default node the inputs merge at the lane roots; for any other
    /// node the IO lane nests them under the node id while the global lane still
    /// merges them at its root.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `inputs` is not a JSON object or a key
    /// is not a valid nested path.
    pub fn commit_user_inputs(&self, inputs: Value) -> Result<()> {
        let inputs = ensure_object(inputs, "user inputs")?;
        let mut inner = self.inner.write();
        if self.node_id == DEFAULT_NODE_ID {
            paths::apply_patch(&mut inner.io_state, &Value::Object(inputs.clone()))?;
        } else {
            let patch = single_key_object(&self.node_id, Value::Object(inputs.clone()));
            paths::apply_patch(&mut inner.io_state, &patch)?;
        }
        paths::apply_patch(&mut inner.global_state, &Value::Object(inputs))
    }

    /// Applies `patch` directly to one lane, bypassing staging.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object or a key
    /// is not a valid nested path.
    pub fn apply_to_lane(&self, lane: StateLane, patch: &Value) -> Result<()> {
        let mut inner = self.inner.write();
        let target = match lane {
            StateLane::Io => &mut inner.io_state,
            StateLane::Global => &mut inner.global_state,
            StateLane::Comp => &mut inner.comp_state,
            StateLane::Workflow => &mut inner.workflow_state,
            StateLane::Trace => &mut inner.trace_state,
        };
        paths::apply_patch(target, patch)
    }

    /// Returns the trace lane.
    #[must_use]
    pub fn trace_state(&self) -> Value {
        self.trace_value()
    }

    /// Merges an object `patch` into the trace lane.
    ///
    /// Python stores spans keyed by node id; this accepts any object patch and
    /// merges it, so `{node_id: span}` replaces that node's entry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `patch` is not a JSON object or a key
    /// is not a valid nested path.
    pub fn update_trace(&self, patch: Value) -> Result<()> {
        self.patch_trace(patch)
    }

    /// Returns all five lanes as `{io_state, global_state, comp_state,
    /// workflow_state, trace_state}`.
    ///
    /// Every key is always present; `global_state` is included even for
    /// workflow-only stores.
    #[must_use]
    pub fn dump(&self) -> Value {
        self.dump_value()
    }

    /// Reads global state, falling back to IO lanes at the parent and node ids.
    fn read_global(&self, key: &str) -> Option<Value> {
        if key.is_empty() {
            if self.workflow_only {
                return None;
            }
            return Some(self.inner.read().global_state.clone());
        }

        let inner = self.inner.read();
        if let Some(value) = paths::get_value_cloned(&inner.global_state, key) {
            return Some(value);
        }
        for prefix in [&self.parent_id, &self.node_id] {
            if prefix.is_empty() {
                continue;
            }
            let scope = paths::get_by_prefix(&inner.io_state, prefix);
            if let Some(value) = paths::get_value_cloned(&scope, key) {
                return Some(value);
            }
        }
        None
    }

    /// Stages a global-lane patch for this node id.
    fn stage_global(&self, patch: Value) -> Result<()> {
        let patch = ensure_object(patch, "global state patch")?;
        let mut inner = self.inner.write();
        inner
            .global_updates
            .entry(self.node_id.clone())
            .or_default()
            .push(Value::Object(patch));
        Ok(())
    }

    /// Stages a component-lane patch under an arbitrary component id.
    fn stage_component(&self, component_id: &str, patch: Value) -> Result<()> {
        if component_id.is_empty() {
            return Err(Error::validation("component id must not be empty"));
        }
        let patch = ensure_object(patch, "component state patch")?;
        let mut inner = self.inner.write();
        inner
            .comp_updates
            .entry(component_id.to_string())
            .or_default()
            .push(single_key_object(component_id, Value::Object(patch)));
        Ok(())
    }

    /// Reads committed component state by nested component id.
    fn read_component(&self, component_id: &str) -> Option<Value> {
        let inner = self.inner.read();
        paths::get_value_cloned(&inner.comp_state, component_id)
    }

    /// Writes committed component state by nested component id (not staged).
    fn write_component(&self, component_id: &str, patch: Value) -> Result<()> {
        if component_id.is_empty() {
            return Err(Error::validation("component id must not be empty"));
        }
        let mut inner = self.inner.write();
        let wrapper = single_key_object(component_id, patch);
        paths::apply_patch(&mut inner.comp_state, &wrapper)
    }

    /// Replaces the global and component lanes.
    fn restore_lanes(&self, global: Value, components: Value) -> Result<()> {
        if !global.is_object() {
            return Err(Error::validation(
                "restored global state must be a JSON object",
            ));
        }
        if !components.is_object() {
            return Err(Error::validation(
                "restored component state must be a JSON object",
            ));
        }
        let mut inner = self.inner.write();
        inner.global_state = global;
        inner.comp_state = components;
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

    /// Returns all five committed lanes.
    fn dump_value(&self) -> Value {
        let inner = self.inner.read();
        let mut dump = Map::new();
        dump.insert(IO_STATE_KEY.to_string(), inner.io_state.clone());
        dump.insert(GLOBAL_STATE_KEY.to_string(), inner.global_state.clone());
        dump.insert(COMP_STATE_KEY.to_string(), inner.comp_state.clone());
        dump.insert(WORKFLOW_STATE_KEY.to_string(), inner.workflow_state.clone());
        dump.insert(TRACE_STATE_KEY.to_string(), inner.trace_state.clone());
        Value::Object(dump)
    }
}

impl StateStore for WorkflowState {
    fn get_global(&self, key: &str) -> Option<Value> {
        self.read_global(key)
    }

    fn update_global(&self, patch: Value) -> Result<()> {
        self.stage_global(patch)
    }

    fn write_global(&self, patch: Value) -> Result<()> {
        self.apply_to_lane(crate::state::StateLane::Global, &patch)
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
        self.inner.read().comp_state.clone()
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

/// Projects a schema against `data`, following Python `get_by_schema`.
fn project_schema(schema: &Value, data: &Value, is_root: bool) -> Value {
    match schema {
        Value::String(text) => {
            let key = match paths::extract_origin_key(text) {
                Some(origin) => origin,
                None if is_root => text.clone(),
                None => return schema.clone(),
            };
            paths::get_value_cloned(data, &key).unwrap_or(Value::Null)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| project_schema(item, data, false))
                .collect(),
        ),
        Value::Object(map) => {
            let mut result = Map::new();
            for (key, value) in map {
                let nested = value.is_array()
                    || value.is_object()
                    || value.as_str().is_some_and(paths::is_ref_path);
                let projected = if nested {
                    project_schema(value, data, false)
                } else {
                    value.clone()
                };
                result.insert(key.clone(), projected);
            }
            Value::Object(result)
        }
        other => other.clone(),
    }
}

/// Builds a one-entry JSON object.
fn single_key_object(key: &str, value: Value) -> Value {
    let mut map = Map::new();
    map.insert(key.to_string(), value);
    Value::Object(map)
}

/// Serializes a staging queue map into `{node_id: [patch, ...]}`.
fn queue_to_value(queue: &BTreeMap<String, Vec<Value>>) -> Value {
    let mut result = Map::new();
    for (node_id, patches) in queue {
        result.insert(node_id.clone(), Value::Array(patches.clone()));
    }
    Value::Object(result)
}

/// Replaces a staging queue from a `{node_id: [patch, ...]}` value.
fn set_queue(queue: &mut BTreeMap<String, Vec<Value>>, value: Option<&Value>) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() {
        queue.clear();
        return Ok(());
    }
    let Some(entries) = value.as_object() else {
        return Err(Error::validation(
            "state updates lane must be a JSON object",
        ));
    };
    let mut parsed = BTreeMap::new();
    for (node_id, patches) in entries {
        let Some(patches) = patches.as_array() else {
            return Err(Error::validation(format!(
                "state updates for node `{node_id}` must be an array"
            )));
        };
        parsed.insert(node_id.clone(), patches.clone());
    }
    *queue = parsed;
    Ok(())
}

/// Drains a staging queue for `node_id` (or all nodes).
fn take_queue(queue: &mut BTreeMap<String, Vec<Value>>, node_id: Option<&str>) -> Vec<Vec<Value>> {
    match node_id {
        Some(id) => queue.remove(id).into_iter().collect(),
        None => std::mem::take(queue).into_values().collect(),
    }
}

/// Applies drained patch batches to a lane in insertion order.
fn apply_queues(state: &mut Value, queues: Vec<Vec<Value>>) -> Result<()> {
    for patches in queues {
        for patch in patches {
            paths::apply_patch(state, &patch)?;
        }
    }
    Ok(())
}
