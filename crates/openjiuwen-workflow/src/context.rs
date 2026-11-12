//! The per-component execution context handed to [`crate::component::Component`] methods.
//!
//! [`ComponentContext`] is the seam between a component and the rest of a run: it exposes
//! the workflow session's state lanes, the frame writer, human-in-the-loop questions, and
//! the run's cancellation token. It owns no runnable state — clones share the same
//! session, writer, interaction state, and frame counter — so a component may clone it
//! into helper tasks.
//!
//! State reads accept both plain session paths (`"start.query"`) and whole-string
//! references (`"${start.query}"`); reads never fail on a missing path, they return
//! `Ok(None)`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::interaction::INTERACTION_FRAME_TYPE;
use openjiuwen_session::stream::StreamItem;
use openjiuwen_session::Session;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::interrupt::{interrupt_error, InteractionState, WorkflowInteraction};
use crate::output::{self, FrameWriter};
use crate::reference;

/// Trace frame type used for workflow trace payloads.
const TRACE_ITEM_TYPE: &str = "tracer_workflow";

/// Everything a component needs from the run that is executing it.
#[derive(Clone)]
pub struct ComponentContext {
    session: Arc<dyn Session>,
    component_id: String,
    workflow_id: String,
    executable_id: String,
    interactions: Arc<InteractionState>,
    writer: Arc<FrameWriter>,
    cancellation: CancellationToken,
    metadata: BTreeMap<String, Value>,
    frame_index: Arc<AtomicU64>,
}

impl fmt::Debug for ComponentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentContext")
            .field("component_id", &self.component_id)
            .field("workflow_id", &self.workflow_id)
            .field("executable_id", &self.executable_id)
            .field("cancelled", &self.cancellation.is_cancelled())
            .field("metadata", &self.metadata)
            .field("frame_index", &self.frame_index.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ComponentContext {
    /// Build a context for `component_id` inside workflow `workflow_id`.
    ///
    /// The context starts without seeded answers, with an uncancelled
    /// [`CancellationToken`], and with a writer that forwards frames to `session` only.
    #[must_use]
    pub fn new(
        session: Arc<dyn Session>,
        component_id: impl Into<String>,
        workflow_id: impl Into<String>,
    ) -> Self {
        let component_id = component_id.into();
        Self {
            writer: Arc::new(FrameWriter::new(None, Arc::clone(&session))),
            session,
            executable_id: component_id.clone(),
            component_id,
            workflow_id: workflow_id.into(),
            interactions: Arc::new(InteractionState::new()),
            cancellation: CancellationToken::new(),
            metadata: BTreeMap::new(),
            frame_index: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Use `interactions` for seeded answers and pending questions.
    #[must_use]
    pub fn with_interactions(mut self, interactions: Arc<InteractionState>) -> Self {
        self.interactions = interactions;
        self
    }

    /// Replace the frame writer (for example with one that also feeds a channel sink).
    #[must_use]
    pub fn with_writer(mut self, writer: Arc<FrameWriter>) -> Self {
        self.writer = writer;
        self
    }

    /// Use `cancellation` as the run's cancellation token.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Replace the free-form context metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Replace the executable id (the session lane id, `node` or `a.b`).
    ///
    /// It defaults to the component id; nested workflows make the two differ.
    #[must_use]
    pub fn with_executable_id(mut self, executable_id: impl Into<String>) -> Self {
        self.executable_id = executable_id.into();
        self
    }

    /// The session this component runs against.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.session
    }

    /// This component's id.
    #[must_use]
    pub fn component_id(&self) -> &str {
        &self.component_id
    }

    /// The enclosing workflow's id.
    #[must_use]
    pub fn workflow_id(&self) -> &str {
        &self.workflow_id
    }

    /// The executable id used for session component lanes.
    #[must_use]
    pub fn executable_id(&self) -> &str {
        &self.executable_id
    }

    /// Free-form metadata attached to this context.
    #[must_use]
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }

    /// Reads the workflow's global state lane.
    ///
    /// A whole-string reference (`"${start.query}"`) is resolved against the lane: the
    /// first path segment is read through [`Session::get_state`] and the remaining path
    /// is resolved locally. Any other argument is a plain session path. A missing value
    /// is `Ok(None)`, never an error.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::get_state`] failures (for example an unreadable state store).
    pub async fn get_state(&self, key: &str) -> Result<Option<Value>> {
        let Some(origin) = reference::extract_origin_key(key) else {
            return self.session.get_state(key).await;
        };
        let Some((root, rest)) = split_origin(origin) else {
            // A path that starts with an index (`"${[0].name}"`) selects from the
            // whole global lane.
            let Some(global) = self.session.get_state("").await? else {
                return Ok(None);
            };
            return Ok(reference::resolve_path(&global, origin));
        };
        let Some(value) = self.session.get_state(root).await? else {
            return Ok(None);
        };
        Ok(reference::resolve_path(&value, rest))
    }

    /// Deep-merges `patch` into the workflow's global state lane.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::update_state`] failures (a non-object patch or a write
    /// failure).
    pub async fn update_state(&self, patch: Value) -> Result<()> {
        self.session.update_state(patch).await
    }

    /// Reads `key` from this component's state lane.
    ///
    /// `key` is a path inside the lane and an empty `key` returns the whole lane; a
    /// missing lane or path is `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::get_component_state`] failures.
    pub async fn get_component_state(&self, key: &str) -> Result<Option<Value>> {
        let Some(lane) = self
            .session
            .get_component_state(self.component_id())
            .await?
        else {
            return Ok(None);
        };
        Ok(reference::resolve_path(&lane, key))
    }

    /// Deep-merges `patch` into this component's state lane.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::update_component_state`] failures (a non-object patch or a
    /// write failure).
    pub async fn update_component_state(&self, patch: Value) -> Result<()> {
        self.session
            .update_component_state(self.component_id(), patch)
            .await
    }

    /// Returns the session's global state lane as one JSON value.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::snapshot`] failures.
    pub async fn dump_state(&self) -> Result<Value> {
        Ok(self.session.snapshot().await?.global)
    }

    /// Writes an output-mode frame with the next index of this context.
    ///
    /// Indices are monotonically increasing per context and start at `0`; clones of a
    /// context share one counter. Custom and trace frames are unindexed and do not
    /// advance it.
    ///
    /// # Errors
    ///
    /// Propagates the frame writer's errors (for example a rejected `null` payload).
    pub async fn write_stream(&self, payload: Value) -> Result<()> {
        let index = self.next_frame_index();
        self.writer
            .write_output(output::OUTPUT_FRAME_TYPE, index, payload)
            .await
    }

    /// Writes a custom-mode frame (`{"type": "custom", …}`).
    ///
    /// # Errors
    ///
    /// Propagates the frame writer's errors.
    pub async fn write_custom_stream(&self, payload: Value) -> Result<()> {
        self.writer.write_custom(payload).await
    }

    /// Writes a trace-mode frame with the `tracer_workflow` frame type.
    ///
    /// # Errors
    ///
    /// Propagates the frame writer's errors.
    pub async fn trace(&self, data: Value) -> Result<()> {
        self.writer
            .write(StreamItem::trace(TRACE_ITEM_TYPE, data))
            .await
    }

    /// Asks the user a question and returns the answer.
    ///
    /// The interaction id is `question["id"]` when it is a string, else this
    /// component's id. A seeded answer (see [`InteractionState::seed`]) is returned
    /// directly. Otherwise the interaction is recorded as pending, one
    /// `__interaction__` frame carrying `{"id", "index", "question"}` is written, and
    /// the returned error makes the run halt as `INPUT_REQUIRED`; resume re-runs the
    /// component with the question seeded.
    ///
    /// # Errors
    ///
    /// Returns the interrupt error (recover the interaction with
    /// [`crate::interrupt::find_interrupt`]) when no answer is seeded, and propagates
    /// frame-writer failures.
    pub async fn interact(&self, question: Value) -> Result<Value> {
        let id = question
            .get("id")
            .and_then(Value::as_str)
            .map_or_else(|| self.component_id.clone(), str::to_owned);
        if let Some(answer) = self.interactions.take_answer(&id) {
            return Ok(answer);
        }
        let index = self.interactions.next_index(&id);
        let interaction = WorkflowInteraction::new(id.clone(), index, question.clone());
        self.interactions.push_pending(interaction.clone());
        let frame = json!({"id": id, "index": index, "question": question});
        self.writer
            .write(StreamItem::output(
                INTERACTION_FRAME_TYPE,
                Some(i64::from(index)),
                frame,
            ))
            .await?;
        Err(interrupt_error(interaction))
    }

    /// The run's cancellation token.
    #[must_use]
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Fails with [`Error::Cancelled`] when the run has been cancelled.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when [`ComponentContext::cancellation`] is cancelled.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            return Err(Error::cancelled());
        }
        Ok(())
    }

    /// Pops the next seeded answer for `id`.
    ///
    /// This is consuming, exactly like [`ComponentContext::interact`]: each seeded
    /// answer is delivered once. Seeding several answers for one id queues them in
    /// order.
    #[must_use]
    pub fn seeded_answer(&self, id: &str) -> Option<Value> {
        self.interactions.take_answer(id)
    }

    /// The next output frame index for this context.
    fn next_frame_index(&self) -> u32 {
        let current = self.frame_index.fetch_add(1, Ordering::Relaxed);
        u32::try_from(current).unwrap_or(u32::MAX)
    }
}

/// Splits a reference path into its root session key and the remaining path.
///
/// Returns `None` when the path does not start with a key (`"[0].a"`).
fn split_origin(origin: &str) -> Option<(&str, &str)> {
    let split_at = origin.find(['.', '[']).unwrap_or(origin.len());
    let root = &origin[..split_at];
    if root.is_empty() {
        return None;
    }
    let rest = origin[split_at..].trim_start_matches('.');
    Some((root, rest))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::stream::{StreamItem, StreamMode};
    use openjiuwen_session::{InteractiveInput, Session, WorkflowSession};
    use parking_lot::Mutex;
    use serde_json::json;

    use super::*;
    use crate::interrupt::find_interrupt;
    use crate::output::FrameCollector;

    fn session() -> Arc<dyn Session> {
        Arc::new(WorkflowSession::new("context-tests"))
    }

    fn context() -> ComponentContext {
        ComponentContext::new(session(), "comp", "wf")
    }

    /// A context whose writer feeds an in-memory collector.
    fn collecting_context(
        interactions: Arc<InteractionState>,
    ) -> (ComponentContext, FrameCollector) {
        let session = session();
        let writer = Arc::new(FrameWriter::new(None, Arc::clone(&session)));
        let collector: FrameCollector = Arc::new(Mutex::new(Vec::new()));
        writer.collect_into(Arc::clone(&collector));
        let ctx = ComponentContext::new(session, "comp", "wf")
            .with_interactions(interactions)
            .with_writer(writer);
        (ctx, collector)
    }

    fn frames(collector: &FrameCollector) -> Vec<StreamItem> {
        collector.lock().clone()
    }

    #[test]
    fn accessors_expose_the_constructed_ids() {
        let ctx = context()
            .with_executable_id("wf.comp")
            .with_metadata(BTreeMap::from([("k".to_owned(), json!(1))]));
        assert_eq!(ctx.component_id(), "comp");
        assert_eq!(ctx.workflow_id(), "wf");
        assert_eq!(ctx.executable_id(), "wf.comp");
        assert_eq!(ctx.session().id(), "context-tests");
        assert_eq!(ctx.metadata().get("k"), Some(&json!(1)));
        assert!(!ctx.cancellation().is_cancelled());
        assert!(ctx.check_cancelled().is_ok());
        assert!(format!("{ctx:?}").contains("component_id: \"comp\""));
    }

    #[test]
    fn executable_id_defaults_to_the_component_id() {
        assert_eq!(context().executable_id(), "comp");
    }

    #[tokio::test]
    async fn global_state_round_trips() {
        let ctx = context();
        ctx.update_state(json!({"query": "hello"})).await.unwrap();
        assert_eq!(ctx.get_state("query").await.unwrap(), Some(json!("hello")));
        assert_eq!(ctx.get_state("missing").await.unwrap(), None);
    }

    #[tokio::test]
    async fn get_state_resolves_reference_paths() {
        let ctx = context();
        ctx.update_state(json!({"start": {"nested": {"b": [10, 20]}, "a": 1}}))
            .await
            .unwrap();

        assert_eq!(ctx.get_state("${start.a}").await.unwrap(), Some(json!(1)));
        assert_eq!(
            ctx.get_state("${start.nested.b[1]}").await.unwrap(),
            Some(json!(20))
        );
        assert_eq!(
            ctx.get_state("${start.nested}").await.unwrap(),
            Some(json!({"b": [10, 20]}))
        );
        // Whole-node and missing-path forms.
        assert_eq!(
            ctx.get_state("${start}").await.unwrap(),
            Some(json!({"nested": {"b": [10, 20]}, "a": 1}))
        );
        assert_eq!(ctx.get_state("${missing.a}").await.unwrap(), None);
        assert_eq!(ctx.get_state("${start.missing}").await.unwrap(), None);
        assert_eq!(
            ctx.get_state("${start.a.deeper}").await.unwrap(),
            None,
            "a scalar in the middle of the path resolves to nothing"
        );
        // A leading index selects from the whole lane.
        assert_eq!(ctx.get_state("${[0]}").await.unwrap(), None);
    }

    #[tokio::test]
    async fn get_state_passes_plain_paths_to_the_session() {
        let ctx = context();
        ctx.update_state(json!({"start": {"a": 3}})).await.unwrap();
        assert_eq!(ctx.get_state("start.a").await.unwrap(), Some(json!(3)));
        assert_eq!(
            ctx.get_state("").await.unwrap(),
            Some(json!({"start": {"a": 3}}))
        );
        assert_eq!(ctx.get_state("nope").await.unwrap(), None);
    }

    #[tokio::test]
    async fn component_state_is_namespaced_per_component() {
        let session = session();
        let first = ComponentContext::new(Arc::clone(&session), "a", "wf");
        let second = ComponentContext::new(Arc::clone(&session), "b", "wf");

        first
            .update_component_state(json!({"step": 1, "nested": {"k": "v"}}))
            .await
            .unwrap();
        second
            .update_component_state(json!({"step": 2}))
            .await
            .unwrap();

        assert_eq!(
            first.get_component_state("step").await.unwrap(),
            Some(json!(1))
        );
        assert_eq!(
            second.get_component_state("step").await.unwrap(),
            Some(json!(2))
        );
        assert_eq!(
            first.get_component_state("nested.k").await.unwrap(),
            Some(json!("v"))
        );
        assert_eq!(
            first.get_component_state("").await.unwrap(),
            Some(json!({"step": 1, "nested": {"k": "v"}}))
        );
        assert_eq!(first.get_component_state("missing").await.unwrap(), None);

        // The global lane is untouched by component writes.
        assert_eq!(first.get_state("step").await.unwrap(), None);
        assert_eq!(first.dump_state().await.unwrap(), json!({}));
    }

    #[tokio::test]
    async fn dump_state_returns_the_global_lane() {
        let ctx = context();
        ctx.update_state(json!({"query": "hello", "n": 2}))
            .await
            .unwrap();
        assert_eq!(
            ctx.dump_state().await.unwrap(),
            json!({"query": "hello", "n": 2})
        );
    }

    #[tokio::test]
    async fn write_stream_uses_a_monotonic_index() {
        let (ctx, collector) = collecting_context(Arc::new(InteractionState::new()));

        ctx.write_stream(json!({"text": "a"})).await.unwrap();
        ctx.write_stream(json!({"text": "b"})).await.unwrap();
        ctx.write_custom_stream(json!({"custom": true}))
            .await
            .unwrap();

        let frames = frames(&collector);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].mode, StreamMode::Output);
        assert_eq!(frames[0].item_type, output::OUTPUT_FRAME_TYPE);
        assert_eq!(frames[0].index, Some(0));
        assert_eq!(frames[0].payload, json!({"text": "a"}));
        assert_eq!(frames[1].index, Some(1));
        assert_eq!(frames[1].payload, json!({"text": "b"}));
        assert_eq!(frames[2].mode, StreamMode::Custom);
        assert_eq!(frames[2].item_type, "custom");
        assert_eq!(frames[2].index, None);
        assert_eq!(frames[2].payload, json!({"custom": true}));
    }

    #[tokio::test]
    async fn trace_writes_a_trace_frame() {
        let (ctx, collector) = collecting_context(Arc::new(InteractionState::new()));
        ctx.trace(json!({"span": "s"})).await.unwrap();

        let frames = frames(&collector);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].mode, StreamMode::Trace);
        assert_eq!(frames[0].item_type, TRACE_ITEM_TYPE);
        assert_eq!(frames[0].payload, json!({"span": "s"}));
    }

    #[tokio::test]
    async fn cloned_contexts_share_the_frame_counter() {
        let (ctx, collector) = collecting_context(Arc::new(InteractionState::new()));
        let clone = ctx.clone();

        ctx.write_stream(json!(1)).await.unwrap();
        clone.write_stream(json!(2)).await.unwrap();

        let frames = frames(&collector);
        assert_eq!(frames[0].index, Some(0));
        assert_eq!(frames[1].index, Some(1));
    }

    #[tokio::test]
    async fn interact_returns_a_seeded_answer() {
        let interactions = Arc::new(InteractionState::new());
        let mut input = InteractiveInput::new();
        input.answer("q1", json!({"answer": "yes"}));
        interactions.seed(&input);
        let (ctx, collector) = collecting_context(Arc::clone(&interactions));

        let answer = ctx
            .interact(json!({"id": "q1", "text": "continue?"}))
            .await
            .unwrap();

        assert_eq!(answer, json!({"answer": "yes"}));
        assert!(
            frames(&collector).is_empty(),
            "a seeded answer writes no frame"
        );
        assert!(interactions.pending().is_empty());
        assert_eq!(ctx.seeded_answer("q1"), None, "the answer is consumed");
    }

    #[tokio::test]
    async fn interact_without_a_seed_halts_and_writes_a_frame() {
        let interactions = Arc::new(InteractionState::new());
        let (ctx, collector) = collecting_context(Arc::clone(&interactions));

        let error = ctx
            .interact(json!({"id": "q1", "text": "continue?"}))
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::External);
        let interaction = find_interrupt(&error).expect("interrupt is recoverable");
        assert_eq!(interaction.id, "q1");
        assert_eq!(interaction.index, 0);
        assert_eq!(interaction.value, json!({"id": "q1", "text": "continue?"}));
        assert_eq!(interactions.pending(), vec![interaction.clone()]);

        let frames = frames(&collector);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].mode, StreamMode::Output);
        assert_eq!(frames[0].item_type, INTERACTION_FRAME_TYPE);
        assert_eq!(frames[0].index, Some(0));
        assert_eq!(
            frames[0].payload,
            json!({"id": "q1", "index": 0, "question": {"id": "q1", "text": "continue?"}})
        );
    }

    #[tokio::test]
    async fn interact_defaults_the_id_to_the_component_id() {
        let interactions = Arc::new(InteractionState::new());
        let (ctx, _collector) = collecting_context(Arc::clone(&interactions));

        let error = ctx
            .interact(json!({"text": "continue?"}))
            .await
            .unwrap_err();
        assert_eq!(find_interrupt(&error).unwrap().id, "comp");
    }

    #[tokio::test]
    async fn interaction_indices_increment_per_id() {
        let interactions = Arc::new(InteractionState::new());
        let (ctx, collector) = collecting_context(Arc::clone(&interactions));

        for expected in 0..2 {
            let error = ctx.interact(json!({"id": "q1"})).await.unwrap_err();
            assert_eq!(find_interrupt(&error).unwrap().index, expected);
        }
        let error = ctx.interact(json!({"id": "q2"})).await.unwrap_err();
        assert_eq!(find_interrupt(&error).unwrap().index, 0);

        let indices: Vec<_> = frames(&collector).iter().filter_map(|f| f.index).collect();
        assert_eq!(indices, vec![0, 1, 0]);
    }

    #[tokio::test]
    async fn seeded_answer_pops_one_answer_per_call() {
        let interactions = Arc::new(InteractionState::new());
        let mut first = InteractiveInput::new();
        first.answer("q", json!(1));
        interactions.seed(&first);
        let mut second = InteractiveInput::new();
        second.answer("q", json!(2));
        interactions.seed(&second);
        let ctx = ComponentContext::new(session(), "comp", "wf")
            .with_interactions(Arc::clone(&interactions));

        assert_eq!(ctx.seeded_answer("q"), Some(json!(1)));
        assert_eq!(ctx.seeded_answer("q"), Some(json!(2)));
        assert_eq!(ctx.seeded_answer("q"), None);
        assert_eq!(ctx.seeded_answer("other"), None);
    }

    #[tokio::test]
    async fn check_cancelled_reports_a_cancelled_token() {
        let token = CancellationToken::new();
        let ctx = ComponentContext::new(session(), "comp", "wf").with_cancellation(token.clone());
        assert!(ctx.check_cancelled().is_ok());
        assert!(!ctx.cancellation().is_cancelled());

        token.cancel();
        let error = ctx.check_cancelled().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert!(ctx.cancellation().is_cancelled());
    }

    #[tokio::test]
    async fn write_stream_propagates_writer_errors() {
        let ctx = context();
        let error = ctx.write_stream(Value::Null).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}
