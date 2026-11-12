//! Running a compiled workflow: blocking invoke, frame streaming, and HITL resume.
//!
//! Both entry points share the same wiring: merge the run inputs into the session's
//! global lane (Python's `commit_user_inputs`), seed the interaction state from
//! [`WorkflowRunOptions::interactive`], compile one graph for the run, and execute it
//! with a cancellation token, a superstep budget, and a timeout.
//!
//! - **invoke** waits for the run and projects it into a [`WorkflowOutput`]. A halt
//!   becomes `INPUT_REQUIRED` carrying the interaction wires and keeps the snapshot for
//!   `resume`; a completed run deletes the snapshot and returns the end component's value.
//! - **stream** publishes frames through the run's [`FrameWriter`] into an unbounded
//!   channel while the run executes in a spawned task, and aborts that task when the
//!   stream is dropped.
//!
//! The root state of every run carries a reserved `"inputs"` key with the original
//! inputs, so both `${query}` (root reference) and `${start.query}` resolve.

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_graph::{CheckpointerStore, ExecutionOptions, GraphStore};
use openjiuwen_session::stream::StreamMode;
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::engine::node::RunScope;
use crate::engine::{CompiledWorkflow, WorkflowRunOptions};
use crate::interrupt::InteractionState;
use crate::output::{
    output_chunk, FrameCollector, FrameWriter, WorkflowChunk, WorkflowExecutionState,
    WorkflowOutput, WORKFLOW_FINAL,
};

/// Reserved root-state key holding the run's original inputs.
const INPUTS_KEY: &str = "inputs";

impl CompiledWorkflow {
    /// Runs the workflow and returns its output.
    ///
    /// Object inputs are deep-merged into the session's global state lane before the run
    /// (Python's `commit_user_inputs`), so `${query}` resolves alongside
    /// `${start.query}`. A run that halts for human input returns
    /// [`WorkflowExecutionState::InputRequired`] with the interaction wires as `result`
    /// and leaves the graph snapshot in place for [`CompiledWorkflow::resume`].
    ///
    /// # Errors
    ///
    /// Propagates state, compilation, graph, timeout, and cancellation failures.
    pub async fn invoke(
        &self,
        inputs: Value,
        session: Arc<dyn Session>,
        options: WorkflowRunOptions,
    ) -> Result<WorkflowOutput> {
        run_blocking(self, inputs, session, options).await
    }

    /// Streams the run's frames.
    ///
    /// Frames are yielded in production order and filtered by
    /// [`WorkflowRunOptions::stream_modes`] when it is set (the terminal `workflow_final`
    /// frame is an output-mode frame and obeys the same filter). A successful run ends
    /// with one `workflow_final` frame carrying the end component's output (omitted when
    /// that output is `null`); a failed run yields exactly one [`Err`]. Dropping the
    /// stream aborts the in-flight run and keeps the graph snapshot when a checkpointer
    /// is configured.
    #[allow(clippy::too_many_lines)] // one linear pipeline reads better than fragments
    pub fn stream(
        &self,
        inputs: Value,
        session: Arc<dyn Session>,
        options: WorkflowRunOptions,
    ) -> futures::stream::BoxStream<'static, Result<WorkflowChunk>> {
        let workflow = self.clone();
        let stream = async_stream::try_stream! {
            let workflow_id = workflow.inner.card.id.clone();
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            let writer = Arc::new(FrameWriter::new(Some(sender), Arc::clone(&session)));
            let collector: FrameCollector = Arc::new(Mutex::new(Vec::new()));
            writer.collect_into(collector);

            if !options.resume && inputs.is_object() {
                session.update_state(inputs.clone()).await?;
            }
            let interactions = Arc::new(InteractionState::new());
            if let Some(interactive) = options.interactive.as_ref() {
                interactions.seed(interactive);
            }

            let store = store_for(&session, &workflow_id);
            let scope = Arc::new(RunScope::new(interactions, Arc::clone(&writer)));
            let graph = workflow.build_graph(
                Some(&scope),
                store.clone(),
                options.max_supersteps,
            )?;
            // Only the graph's node runtimes keep the writer now, so the receiver ends
            // when the spawned run finishes.
            drop(writer);

            let execution = ExecutionOptions {
                cancellation: options.cancellation.clone(),
                resume: options.resume,
                max_supersteps: options.max_supersteps,
                write_session_trace: true,
                ..ExecutionOptions::default()
            };
            let seed = seed_value(inputs);
            let run_session = Arc::clone(&session);
            let handle = tokio::spawn(async move { graph.run(seed, run_session, execution).await });
            // Held across every yield, so dropping the stream aborts the run.
            let guard = AbortOnDrop(Some(handle.abort_handle()));

            let modes = options.stream_modes.clone();
            while let Some(frame) = receiver.recv().await {
                if modes.as_ref().is_none_or(|modes| modes.contains(&frame.mode)) {
                    yield frame;
                }
            }

            let result = match handle.await {
                Ok(result) => result,
                Err(join_error) => Err(Error::external(join_error)
                    .with_context("a workflow run task failed")),
            }?;
            drop(guard);

            if result.halted.is_some() {
                // The interaction frames were already yielded by the run's writer and
                // the snapshot stays in place for `resume`.
                return;
            }
            if let Some(store) = store {
                store.delete(session.id(), Some(workflow_id.as_str())).await?;
            }
            if let Some(end_output) = result
                .state
                .get(&workflow.inner.end_id)
                .filter(|value| !value.is_null())
            {
                // The final frame is an output-mode frame, so it obeys the same filter.
                let wanted = modes
                    .as_ref()
                    .is_none_or(|modes| modes.contains(&StreamMode::Output));
                if wanted {
                    yield output_chunk(WORKFLOW_FINAL, 0, end_output.clone());
                }
            }
        };
        Box::pin(stream)
    }

    /// Resumes an interrupted run with `interactive` answers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Workflow`] when the session has no checkpointer, and otherwise
    /// propagates the same failures as [`CompiledWorkflow::invoke`].
    pub async fn resume(
        &self,
        interactive: openjiuwen_session::InteractiveInput,
        session: Arc<dyn Session>,
        options: WorkflowRunOptions,
    ) -> Result<WorkflowOutput> {
        if session.checkpointer().is_none() {
            return Err(Error::workflow(
                "interactive resume requested without a checkpointer",
            ));
        }
        let options = WorkflowRunOptions {
            resume: true,
            interactive: Some(interactive),
            ..options
        };
        run_blocking(self, Value::Null, session, options).await
    }
}

/// Runs `workflow` to completion (or to an interaction halt) and projects the result.
async fn run_blocking(
    workflow: &CompiledWorkflow,
    inputs: Value,
    session: Arc<dyn Session>,
    options: WorkflowRunOptions,
) -> Result<WorkflowOutput> {
    let inner = &workflow.inner;
    if !options.resume && inputs.is_object() {
        session.update_state(inputs.clone()).await?;
    }

    let interactions = Arc::new(InteractionState::new());
    if let Some(interactive) = options.interactive.as_ref() {
        interactions.seed(interactive);
    }
    let writer = Arc::new(FrameWriter::new(None, Arc::clone(&session)));
    let collector: FrameCollector = Arc::new(Mutex::new(Vec::new()));
    writer.collect_into(collector);

    let store = store_for(&session, &inner.card.id);
    let scope = Arc::new(RunScope::new(interactions, writer));
    let graph = workflow.build_graph(Some(&scope), store.clone(), options.max_supersteps)?;
    let execution = ExecutionOptions {
        cancellation: options.cancellation.clone(),
        resume: options.resume,
        max_supersteps: options.max_supersteps,
        write_session_trace: true,
        ..ExecutionOptions::default()
    };
    let seed = seed_value(inputs);

    let timeout = resolve_timeout(&session, options.timeout);
    let run = match timeout {
        Some(limit) => {
            tokio::time::timeout(limit, graph.run(seed, Arc::clone(&session), execution))
                .await
                .map_err(|_| {
                    Error::timeout(format!("workflow '{}' execution timed out", inner.card.id))
                })??
        }
        None => graph.run(seed, Arc::clone(&session), execution).await?,
    };

    if let Some(halt) = run.halted {
        return Ok(WorkflowOutput {
            result: halt.value.unwrap_or_else(|| Value::Array(Vec::new())),
            state: WorkflowExecutionState::InputRequired,
        });
    }
    if let Some(store) = store {
        store
            .delete(session.id(), Some(inner.card.id.as_str()))
            .await?;
    }
    let result = run.state.get(&inner.end_id).cloned().unwrap_or(Value::Null);
    Ok(WorkflowOutput {
        result,
        state: WorkflowExecutionState::Completed,
    })
}

/// The graph store backing this run, when the session is checkpointed.
fn store_for(session: &Arc<dyn Session>, workflow_id: &str) -> Option<Arc<dyn GraphStore>> {
    session.checkpointer().map(|checkpointer| {
        Arc::new(CheckpointerStore::new(checkpointer, workflow_id)) as Arc<dyn GraphStore>
    })
}

/// The run's root state: the inputs plus a reserved `"inputs"` key.
fn seed_value(inputs: Value) -> Value {
    let mut seed = if inputs.is_object() {
        inputs.clone()
    } else {
        Value::Object(Map::new())
    };
    if let Value::Object(map) = &mut seed {
        map.insert(INPUTS_KEY.to_owned(), inputs);
    }
    seed
}

/// The run's timeout: the explicit option, else the session's configured limit.
///
/// A configured limit of `0` means "no timeout", matching Python's disabled setting.
fn resolve_timeout(session: &Arc<dyn Session>, requested: Option<Duration>) -> Option<Duration> {
    if requested.is_some() {
        return requested;
    }
    let config = session.config()?;
    let millis = config.read().execute_timeout_ms;
    if millis == 0 {
        None
    } else {
        Some(Duration::from_millis(millis))
    }
}

/// Aborts a spawned run when the stream that owns it is dropped.
///
/// Held across every `yield` of [`CompiledWorkflow::stream`], so abandoning the stream
/// aborts the in-flight run instead of leaking it as an untracked background task.
struct AbortOnDrop(Option<tokio::task::AbortHandle>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::assert_is_empty)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn seed_value_adds_the_reserved_inputs_key() {
        assert_eq!(
            seed_value(json!({"query": "hi"})),
            json!({"query": "hi", "inputs": {"query": "hi"}})
        );
    }

    #[test]
    fn non_object_inputs_are_wrapped() {
        assert_eq!(seed_value(json!("plain")), json!({"inputs": "plain"}));
        assert_eq!(seed_value(Value::Null), json!({"inputs": null}));
    }

    #[test]
    fn abort_guard_tolerates_being_dropped_twice() {
        let guard = AbortOnDrop(None);
        drop(guard);
    }
}
