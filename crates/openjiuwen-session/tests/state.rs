//! Integration tests for the session `state` module.
//!
//! Ports behavior from `openjiuwen/core/session/state/` (`CommitState`,
//! agent/team `StateCollection`) and `openjiuwen/core/session/utils.py`
//! (`update_dict`, `get_by_schema`, `is_ref_path`, `extract_origin_key`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use serde_json::{json, Map, Value};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::state::paths::{
    apply_patch, extract_origin_key, get_by_prefix, get_value, is_ref_path, remove_value, set_value,
};
use openjiuwen_session::state::{
    AgentState, StateLane, StateStore, WorkflowState, COMP_STATE_KEY, COMP_STATE_UPDATES_KEY,
    GLOBAL_STATE_KEY, GLOBAL_STATE_UPDATES_KEY, IO_STATE_KEY, TRACE_STATE_KEY, WORKFLOW_STATE_KEY,
    WORKFLOW_STATE_UPDATES_KEY,
};

// --- apply_patch (Python `update_dict`) ---------------------------------------

#[test]
fn apply_patch_merges_objects_and_applies_null_deletion_last() {
    let mut value = json!({"a": {"b": 1, "c": 2}, "list": [{"x": 1}, {"y": 2}]});
    apply_patch(
        &mut value,
        &json!({"a": {"c": 3, "d": 4}, "list": [9], "a.b": null}),
    )
    .unwrap();

    // `a` merges recursively, `list` is replaced, and `a.b` is deleted after the
    // writes in the same patch.
    assert_eq!(value, json!({"a": {"c": 3, "d": 4}, "list": [9]}));
}

#[test]
fn apply_patch_replaces_arrays_wholesale() {
    let mut value = json!({"a": [{"x": 1}, {"y": 2}], "b": [1, 2, 3]});
    apply_patch(&mut value, &json!({"a": [{"x": 9}], "b": []})).unwrap();
    assert_eq!(value, json!({"a": [{"x": 9}], "b": []}));

    // An object replaces an array and vice versa; only objects merge.
    apply_patch(&mut value, &json!({"a": {"x": 1}, "b": {"y": 2}})).unwrap();
    assert_eq!(value, json!({"a": {"x": 1}, "b": {"y": 2}}));
}

#[test]
fn apply_patch_resolves_dotted_and_bracketed_keys() {
    let mut value = json!({});
    apply_patch(&mut value, &json!({"a.b[0].c": 7, "a.b[1]": {"d": 1}})).unwrap();
    assert_eq!(value, json!({"a": {"b": [{"c": 7}, {"d": 1}]}}));
}

#[test]
fn apply_patch_scopes_recursive_keys_to_the_nested_object() {
    let mut value = json!({"a": {"b": {"c": 1}}});
    apply_patch(&mut value, &json!({"a": {"b.d": 2}})).unwrap();
    assert_eq!(value, json!({"a": {"b": {"c": 1, "d": 2}}}));
}

#[test]
fn apply_patch_rejects_non_object_patches() {
    let mut value = json!({"kept": 1});
    let error = apply_patch(&mut value, &json!([1, 2])).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(value, json!({"kept": 1}));
}

// --- nested path helpers ------------------------------------------------------

#[test]
fn path_helpers_read_write_and_remove() {
    let value = json!({"a": {"b": [10, 20]}});
    assert_eq!(get_value(&value, ""), Some(&value));
    assert_eq!(get_value(&value, "a.b[1]"), Some(&json!(20)));
    assert_eq!(get_value(&value, "a.missing"), None);

    let mut value = json!({});
    set_value(&mut value, "a.b[0].c", json!(7)).unwrap();
    assert_eq!(value, json!({"a": {"b": [{"c": 7}]}}));

    assert_eq!(get_by_prefix(&value, "a.b"), json!([{"c": 7}]));
    assert_eq!(get_by_prefix(&value, "missing"), Value::Null);
    assert_eq!(get_by_prefix(&value, ""), value);

    assert!(remove_value(&mut value, "a.b[0].c"));
    assert!(!remove_value(&mut value, "a.b[5]"));
    assert!(!remove_value(&mut value, "bad..path"));
    assert_eq!(value, json!({"a": {"b": [{}]}}));

    assert!(remove_value(&mut value, "a.b[0]"));
    assert_eq!(value, json!({"a": {"b": []}}));
}

#[test]
fn ref_path_helpers_detect_and_unwrap_references() {
    assert!(is_ref_path("${start123.p2}"));
    assert!(!is_ref_path("${}"));
    assert!(!is_ref_path("start123.p2"));
    assert!(!is_ref_path("${unclosed"));

    assert_eq!(
        extract_origin_key("${start123.p2}").as_deref(),
        Some("start123.p2")
    );
    assert_eq!(
        extract_origin_key("prefix${x.y}suffix").as_deref(),
        Some("x.y")
    );
    assert_eq!(extract_origin_key("start123.p2"), None);
    assert_eq!(extract_origin_key("${unclosed"), None);
}

// --- WorkflowState ------------------------------------------------------------

#[test]
fn node_view_shares_lanes_and_scopes_reads() {
    let state = WorkflowState::new();
    state
        .apply_to_lane(
            StateLane::Comp,
            &json!({"default": {"value": 1}, "parent": {"child": {"x": 2}}}),
        )
        .unwrap();

    assert_eq!(state.get("value"), Some(json!(1)));

    // Node state ids are executable ids (`parent.child`), like Python's
    // `create_executable_id`.
    let node = state.node("parent.child", "parent");
    assert_eq!(node.get("x"), Some(json!(2)));
    assert_eq!(node.get(""), Some(json!({"x": 2})));
    assert_eq!(node.get("missing"), None);
    assert_eq!(node.get_by_prefix(""), json!({"x": 2}));

    // Writes through the parent are visible in the child (shared lane).
    state
        .apply_to_lane(StateLane::Comp, &json!({"parent": {"child": {"y": 3}}}))
        .unwrap();
    assert_eq!(node.get("y"), Some(json!(3)));
    assert_eq!(node.get_by_prefix(""), json!({"x": 2, "y": 3}));
}

#[test]
fn node_view_shares_staging_queues() {
    let state = WorkflowState::new();
    state.update_global(json!({"g": 1})).unwrap();

    let node = state.node("child", "");
    node.commit(None).unwrap();

    assert_eq!(state.lane(StateLane::Global), json!({"g": 1}));
}

#[test]
fn staging_commit_and_updates_round_trip() {
    let state = WorkflowState::new();
    state.update(json!({"value": 1})).unwrap();

    // Staged, not committed yet.
    assert_eq!(state.get("value"), None);
    assert_eq!(
        state.updates()[COMP_STATE_UPDATES_KEY],
        json!({"default": [{"default": {"value": 1}}]})
    );

    state.commit(None).unwrap();
    assert_eq!(state.get("value"), Some(json!(1)));
    assert_eq!(state.updates()[COMP_STATE_UPDATES_KEY], json!({}));
}

#[test]
fn commit_with_node_id_only_drains_that_node() {
    let state = WorkflowState::new();
    state.update_global(json!({"g": 1})).unwrap();
    state.update_workflow(json!({"w": 1})).unwrap();
    state.update_component("comp.a", json!({"x": 1})).unwrap();

    state.commit(Some("default")).unwrap();
    assert_eq!(state.lane(StateLane::Global), json!({"g": 1}));
    assert_eq!(state.get_workflow_state("w"), Some(json!(1)));
    assert_eq!(
        state.updates()[COMP_STATE_UPDATES_KEY],
        json!({"comp.a": [{"comp.a": {"x": 1}}]})
    );

    state.commit(Some("comp.a")).unwrap();
    assert_eq!(state.components(), json!({"comp": {"a": {"x": 1}}}));
}

#[test]
fn rollback_clears_staged_updates() {
    let state = WorkflowState::new();
    state.update_global(json!({"lost": 1})).unwrap();
    state.update_workflow(json!({"lost": 2})).unwrap();

    let node = state.node("other", "");
    node.update_global(json!({"kept": 3})).unwrap();

    state.rollback(Some("default"));
    assert_eq!(
        state.updates()[GLOBAL_STATE_UPDATES_KEY],
        json!({"other": [{"kept": 3}]})
    );
    assert_eq!(state.updates()[WORKFLOW_STATE_UPDATES_KEY], json!({}));

    state.commit(None).unwrap();
    assert_eq!(state.lane(StateLane::Global), json!({"kept": 3}));
    assert_eq!(state.lane(StateLane::Workflow), json!({}));
}

#[test]
fn set_updates_restores_staging() {
    let source = WorkflowState::new();
    source.update_global(json!({"a": 1})).unwrap();
    source.update_workflow(json!({"b": 2})).unwrap();
    source.update(json!({"c": 3})).unwrap();
    let snapshot = source.updates();

    let target = WorkflowState::new();
    target.set_updates(&snapshot).unwrap();
    assert_eq!(target.updates(), snapshot);

    target.commit(None).unwrap();
    assert_eq!(target.lane(StateLane::Global), json!({"a": 1}));
    assert_eq!(target.lane(StateLane::Workflow), json!({"b": 2}));
    assert_eq!(target.get("c"), Some(json!(3)));
}

#[test]
fn component_ids_are_nested_paths() {
    let state = WorkflowState::new();
    state.update_component("a.b", json!({"v": 1})).unwrap();
    state.commit(None).unwrap();
    assert_eq!(state.components(), json!({"a": {"b": {"v": 1}}}));
    assert_eq!(
        StateStore::get_component(&state, "a.b"),
        Some(json!({"v": 1}))
    );

    // The trait writes the committed view directly, nesting by path.
    StateStore::update_component(&state, "c.d", json!({"w": 2})).unwrap();
    assert_eq!(
        StateStore::get_component(&state, "c.d"),
        Some(json!({"w": 2}))
    );

    assert_eq!(StateStore::get_component(&state, ""), None);
    let error = StateStore::update_component(&state, "", json!({"v": 3})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn commit_user_inputs_writes_io_and_global() {
    let state = WorkflowState::new();
    state.commit_user_inputs(json!({"q": "hi"})).unwrap();
    assert_eq!(state.lane(StateLane::Io), json!({"q": "hi"}));
    assert_eq!(state.lane(StateLane::Global), json!({"q": "hi"}));

    let node = state.node("parent.step", "parent");
    node.commit_user_inputs(json!({"q": "x"})).unwrap();
    assert_eq!(
        state.lane(StateLane::Io),
        json!({"q": "hi", "parent": {"step": {"q": "x"}}})
    );
    assert_eq!(state.lane(StateLane::Global), json!({"q": "x"}));
}

#[test]
fn schema_projection_reads_node_io_state() {
    let state = WorkflowState::new();
    state
        .set_outputs(
            "default",
            json!({"name": "n", "nested": {"value": 2}, "list": [1, 2]}),
        )
        .unwrap();

    assert_eq!(state.get_by_schema(&json!("${name}")), json!("n"));
    assert_eq!(state.get_by_schema(&json!("nested.value")), json!(2));
    assert_eq!(
        state.get_by_schema(
            &json!({"a": "${name}", "lit": 5, "s": "plain", "obj": {"b": "${nested.value}"}})
        ),
        json!({"a": "n", "lit": 5, "s": "plain", "obj": {"b": 2}})
    );
    assert_eq!(state.get_by_schema(&json!(["${name}", 7])), json!(["n", 7]));
    assert_eq!(state.get_by_schema(&json!("${missing}")), Value::Null);
}

#[test]
fn inputs_and_outputs_read_io_lane_scopes() {
    let state = WorkflowState::new();
    state
        .apply_to_lane(
            StateLane::Io,
            &json!({"parent": {"step": {"out": 1}}, "step": {"own": 2}}),
        )
        .unwrap();

    // Inputs are the IO subtree at the executable node id.
    let nested = state.node("parent.step", "parent");
    assert_eq!(nested.inputs(), json!({"out": 1}));

    // Outputs resolve `node_id` relative to the parent id (Python `get_outputs`).
    assert_eq!(nested.outputs(Some("step")), json!({"out": 1}));
    assert_eq!(nested.outputs(Some("other")), Value::Null);

    let root = state.node("step", "");
    assert_eq!(root.inputs(), json!({"own": 2}));
    assert_eq!(root.outputs(None), json!({"own": 2}));
    assert_eq!(root.outputs(Some("step")), json!({"own": 2}));
}

#[test]
fn dump_always_has_five_lanes_and_hides_global_for_workflow_only() {
    let state = WorkflowState::new();
    assert!(state.workflow_only());
    state
        .apply_to_lane(StateLane::Global, &json!({"g": 1}))
        .unwrap();

    assert_eq!(state.get_global(""), None);
    assert_eq!(state.get_global("g"), Some(json!(1)));

    let dump = state.dump();
    let lanes = dump.as_object().unwrap();
    assert_eq!(lanes.len(), 5);
    for key in [
        IO_STATE_KEY,
        GLOBAL_STATE_KEY,
        COMP_STATE_KEY,
        WORKFLOW_STATE_KEY,
        TRACE_STATE_KEY,
    ] {
        assert!(lanes.contains_key(key), "missing lane {key}");
    }
    assert_eq!(lanes.get(GLOBAL_STATE_KEY), Some(&json!({"g": 1})));

    let with_global = WorkflowState::with_global_lane(false);
    assert!(!with_global.workflow_only());
    assert_eq!(with_global.get_global(""), Some(json!({})));
}

#[test]
fn trace_lane_merges_object_patches() {
    let state = WorkflowState::new();
    state
        .update_trace(json!({"node": {"status": "start"}}))
        .unwrap();
    state.update_trace(json!({"node": {"elapsed": 5}})).unwrap();

    let expected = json!({"node": {"status": "start", "elapsed": 5}});
    assert_eq!(state.trace_state(), expected);
    assert_eq!(state.dump()[TRACE_STATE_KEY], expected);
}

#[test]
fn restore_replaces_global_and_component_lanes() {
    let state = WorkflowState::new();
    StateStore::restore(&state, json!({"g": 1}), json!({"a": {"b": 2}})).unwrap();
    assert_eq!(state.lane(StateLane::Global), json!({"g": 1}));
    assert_eq!(state.components(), json!({"a": {"b": 2}}));

    let error = StateStore::restore(&state, json!({"g": 1}), json!([1])).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

// --- AgentState ---------------------------------------------------------------

#[test]
fn agent_state_dump_shape_and_component_nesting() {
    let agent = AgentState::new();
    agent.update_global(json!({"g": 1})).unwrap();
    agent.update(json!({"a": 2})).unwrap();
    assert_eq!(agent.get_global("g"), Some(json!(1)));

    let dump = agent.dump();
    assert_eq!(
        dump,
        json!({"global_state": {"g": 1}, "agent_state": {"a": 2}})
    );
    assert_eq!(dump.as_object().unwrap().len(), 2);

    StateStore::update_component(&agent, "tool.invoke", json!({"x": 1})).unwrap();
    assert_eq!(
        agent.components(),
        json!({"a": 2, "tool": {"invoke": {"x": 1}}})
    );
    assert_eq!(
        StateStore::get_component(&agent, "tool.invoke"),
        Some(json!({"x": 1}))
    );
    assert_eq!(
        agent.get(""),
        Some(json!({"a": 2, "tool": {"invoke": {"x": 1}}}))
    );
    assert_eq!(StateStore::get_component(&agent, ""), None);

    agent.update_agent(json!({"b": 3})).unwrap();
    assert_eq!(agent.get("b"), Some(json!(3)));

    agent.update_trace(json!({"t": 1})).unwrap();
    assert_eq!(agent.trace_state(), json!({"t": 1}));
    assert_eq!(agent.dump().as_object().unwrap().len(), 2);
}

#[test]
fn agent_restore_replaces_both_lanes() {
    let agent = AgentState::new();
    StateStore::restore(&agent, json!({"g": 2}), json!({"c": 3})).unwrap();
    assert_eq!(
        agent.dump(),
        json!({"global_state": {"g": 2}, "agent_state": {"c": 3}})
    );

    let error = StateStore::restore(&agent, json!(1), json!({})).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn agent_state_is_isolated_from_workflow_state() {
    let agent = AgentState::new();
    let workflow = WorkflowState::new();
    agent.update(json!({"a": 1})).unwrap();
    agent.update_global(json!({"g": 1})).unwrap();

    assert_eq!(workflow.components(), json!({}));
    assert_eq!(workflow.lane(StateLane::Global), json!({}));
}

// --- concurrency and isolation ------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_updates_commit_all_keys() {
    let state = Arc::new(WorkflowState::new());
    let mut handles = Vec::new();

    for worker in 0..8 {
        let state = Arc::clone(&state);
        handles.push(tokio::spawn(async move {
            for step in 0..100 {
                let mut patch = Map::new();
                patch.insert(format!("k{worker}_{step}"), json!(step));
                state.update_global(Value::Object(patch)).unwrap();
                state.commit(None).unwrap();
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let global = state.lane(StateLane::Global);
    let entries = global.as_object().unwrap();
    assert_eq!(entries.len(), 800);
    for worker in 0..8 {
        for step in 0..100 {
            assert!(entries.contains_key(&format!("k{worker}_{step}")));
        }
    }
}

#[test]
fn independent_stores_do_not_share_lanes() {
    let first = WorkflowState::new();
    let second = WorkflowState::new();
    first.update_global(json!({"x": 1})).unwrap();
    first.commit(None).unwrap();

    assert_eq!(second.lane(StateLane::Global), json!({}));

    let a = AgentState::new();
    let b = AgentState::new();
    a.update(json!({"x": 1})).unwrap();
    assert_eq!(b.get("x"), None);
}
