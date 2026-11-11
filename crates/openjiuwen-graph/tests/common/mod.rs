//! Shared helpers for the graph integration tests.
//!
//! Each integration test is its own crate, so unused helpers are expected.
#![allow(
    dead_code,
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty
)]

use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_graph::{GraphContext, GraphState, NodeOutcome, SyncNode};
use openjiuwen_session::{Checkpointer, Session, WorkflowSession, WorkflowSessionOptions};

/// A workflow session with the stable id `s1`.
pub fn session() -> Arc<dyn Session> {
    Arc::new(WorkflowSession::new("s1"))
}

/// A workflow session backed by `checkpointer`.
pub fn session_with_checkpointer(checkpointer: Arc<dyn Checkpointer>) -> Arc<dyn Session> {
    Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some("s1".to_string()),
        checkpointer: Some(checkpointer),
        ..WorkflowSessionOptions::default()
    }))
}

/// A node that increments the state's `count` key.
pub fn counting(
    id: &'static str,
) -> SyncNode<impl Fn(&mut GraphState, &GraphContext) -> Result<NodeOutcome>> {
    SyncNode::new(id, move |state: &mut GraphState, _ctx: &GraphContext| {
        let count = state.get("count").and_then(Value::as_u64).unwrap_or(0);
        state.set("count", json!(count + 1));
        Ok(NodeOutcome::Continue)
    })
}

/// A node that records `ran_<id>` in the state.
pub fn marker(
    id: &'static str,
) -> SyncNode<impl Fn(&mut GraphState, &GraphContext) -> Result<NodeOutcome>> {
    SyncNode::new(id, move |state: &mut GraphState, _ctx: &GraphContext| {
        state.set(format!("ran_{id}"), json!(true));
        Ok(NodeOutcome::Continue)
    })
}

/// A node that does nothing.
pub fn noop(
    id: &'static str,
) -> SyncNode<impl Fn(&mut GraphState, &GraphContext) -> Result<NodeOutcome>> {
    SyncNode::new(id, |_state: &mut GraphState, _ctx: &GraphContext| {
        Ok(NodeOutcome::Continue)
    })
}
