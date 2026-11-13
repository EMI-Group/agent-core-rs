//! Workflow interruptions and the seeded-answer state used to resume runs.
//!
//! Mirrors Python's human-in-the-loop behavior: a component asks a question,
//! the run halts with [`WorkflowInterrupt`] as the error source, and a resumed
//! run seeds [`openjiuwen_session::InteractiveInput`] answers into
//! [`InteractionState`] so the same question resolves without halting again.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use openjiuwen_common::error::Error;
use openjiuwen_session::InteractiveInput;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One question/answer interaction raised by a workflow component.
///
/// `index` is the per-interaction-id sequence number and `value` is the
/// question payload (the answer is delivered separately when resuming).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowInteraction {
    /// Interaction id, usually derived from the component id and index.
    pub id: String,
    /// Position within the interaction id's sequence.
    pub index: u32,
    /// Question payload.
    pub value: Value,
}

impl WorkflowInteraction {
    /// Build an interaction.
    #[must_use]
    pub fn new(id: impl Into<String>, index: u32, value: Value) -> Self {
        Self {
            id: id.into(),
            index,
            value,
        }
    }
}

/// Error payload marking a workflow run that halted for human input.
///
/// Kept separate from the question value so [`find_interrupt`] can walk an
/// error's source chain and recover the interaction after context wrappers.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowInterrupt {
    interaction: WorkflowInteraction,
}

impl WorkflowInterrupt {
    /// Build an interrupt from `interaction`.
    #[must_use]
    pub fn new(interaction: WorkflowInteraction) -> Self {
        Self { interaction }
    }

    /// The interaction that caused the halt.
    #[must_use]
    pub fn interaction(&self) -> &WorkflowInteraction {
        &self.interaction
    }
}

impl fmt::Display for WorkflowInterrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "workflow interrupted at '{}' (index {})",
            self.interaction.id, self.interaction.index
        )
    }
}

impl std::error::Error for WorkflowInterrupt {}

/// Wrap `interaction` in the error a component returns to halt the run.
pub fn interrupt_error(interaction: WorkflowInteraction) -> Error {
    Error::external(WorkflowInterrupt::new(interaction))
}

/// Recover the interaction carried by an interrupt error, if any.
///
/// Walks the [`std::error::Error::source`] chain and downcasts at each step, so
/// the interrupt stays findable after [`Error::with_context`] wraps it.
#[must_use]
pub fn find_interrupt(err: &Error) -> Option<&WorkflowInteraction> {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(current) = source {
        if let Some(interrupt) = current.downcast_ref::<WorkflowInterrupt>() {
            return Some(interrupt.interaction());
        }
        source = current.source();
    }
    None
}

/// Seeded answers, per-id sequence counters, and pending interactions.
///
/// Cloned records share no state; the graph adapter owns one instance per run.
#[derive(Debug, Default)]
pub struct InteractionState {
    answers: Mutex<BTreeMap<String, VecDeque<Value>>>,
    indices: Mutex<BTreeMap<String, u32>>,
    pending: Mutex<Vec<WorkflowInteraction>>,
}

impl InteractionState {
    /// An empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue every answer in `input`.
    ///
    /// Interrupt-addressed answers are queued first; node-addressed answers are queued
    /// afterwards only for ids that received an interrupt answer *in this call*, so the
    /// interrupt-addressed answer wins. An id whose queue already holds answers from an
    /// earlier [`InteractionState::append`]/[`InteractionState::seed`] call still takes
    /// the node-addressed answer, in FIFO order.
    pub fn seed(&self, input: &InteractiveInput) {
        let mut answers = self.answers.lock();
        let mut interrupt_addressed: BTreeSet<&str> = BTreeSet::new();
        for (id, value) in &input.interrupt_answers {
            answers
                .entry(id.clone())
                .or_default()
                .push_back(value.clone());
            interrupt_addressed.insert(id.as_str());
        }
        for (id, value) in &input.user_inputs {
            if interrupt_addressed.contains(id.as_str()) {
                continue;
            }
            answers
                .entry(id.clone())
                .or_default()
                .push_back(value.clone());
        }
    }

    /// Appends every answer of every id, in order.
    ///
    /// Used to replay answers that were already delivered in earlier resumes before the
    /// newly supplied ones (a nested run forwards what its child consumed).
    pub fn append(&self, answers: &BTreeMap<String, Vec<Value>>) {
        let mut queues = self.answers.lock();
        for (id, values) in answers {
            let queue = queues.entry(id.clone()).or_default();
            for value in values {
                queue.push_back(value.clone());
            }
        }
    }

    /// A copy of the remaining per-id answer queues, in first-delivery order.
    ///
    /// Reading does not consume anything; [`InteractionState::take_answer`] keeps its
    /// position in the queue.
    #[must_use]
    pub fn queued_answers(&self) -> BTreeMap<String, Vec<Value>> {
        self.answers
            .lock()
            .iter()
            .map(|(id, queue)| (id.clone(), queue.iter().cloned().collect()))
            .collect()
    }

    /// Pop the next seeded answer for `id`.
    #[must_use]
    pub fn take_answer(&self, id: &str) -> Option<Value> {
        let mut answers = self.answers.lock();
        let (value, empty) = {
            let queue = answers.get_mut(id)?;
            let value = queue.pop_front()?;
            (value, queue.is_empty())
        };
        if empty {
            answers.remove(id);
        }
        Some(value)
    }

    /// Whether an answer is queued for `id`.
    #[must_use]
    pub fn has_answer(&self, id: &str) -> bool {
        self.answers
            .lock()
            .get(id)
            .is_some_and(|queue| !queue.is_empty())
    }

    /// The next interaction index for `id`; the first call returns `0`.
    #[must_use]
    pub fn next_index(&self, id: &str) -> u32 {
        let mut indices = self.indices.lock();
        let next = indices.entry(id.to_owned()).or_insert(0);
        let current = *next;
        *next = next.wrapping_add(1);
        current
    }

    /// Record an interaction that is waiting for input.
    pub fn push_pending(&self, interaction: WorkflowInteraction) {
        self.pending.lock().push(interaction);
    }

    /// Remove and return every pending interaction in insertion order.
    #[must_use]
    pub fn drain_pending(&self) -> Vec<WorkflowInteraction> {
        std::mem::take(&mut *self.pending.lock())
    }

    /// A copy of the pending interactions in insertion order.
    #[must_use]
    pub fn pending(&self) -> Vec<WorkflowInteraction> {
        self.pending.lock().clone()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use serde_json::json;

    fn interaction(id: &str, index: u32) -> WorkflowInteraction {
        WorkflowInteraction::new(id, index, json!({"question": "continue?"}))
    }

    #[test]
    fn interaction_round_trips_and_tolerates_unknown_fields() {
        let item = interaction("c1", 2);
        let parsed: WorkflowInteraction = serde_json::from_value(json!({
            "id": "c1",
            "index": 2,
            "value": {"question": "continue?"},
            "extra": 1
        }))
        .unwrap();
        assert_eq!(parsed, item);
    }

    #[test]
    fn interrupt_display_names_the_interaction() {
        let interrupt = WorkflowInterrupt::new(interaction("c1", 4));
        assert_eq!(
            interrupt.to_string(),
            "workflow interrupted at 'c1' (index 4)"
        );
        assert_eq!(interrupt.interaction(), &interaction("c1", 4));
    }

    #[test]
    fn interrupt_is_found_directly_and_through_context() {
        let item = interaction("c1", 1);
        let direct = interrupt_error(item.clone());
        assert_eq!(find_interrupt(&direct), Some(&item));

        let wrapped = interrupt_error(item.clone()).with_context("component `c1` failed");
        assert_eq!(find_interrupt(&wrapped), Some(&item));

        let twice_wrapped = wrapped.with_context("workflow run failed");
        assert_eq!(find_interrupt(&twice_wrapped), Some(&item));
    }

    #[test]
    fn non_interrupt_errors_have_no_interaction() {
        assert_eq!(find_interrupt(&Error::validation("bad input")), None);
        assert_eq!(
            find_interrupt(&Error::external(std::io::Error::other("io"))),
            None
        );
    }

    #[test]
    fn seed_queues_interrupt_answers_before_user_inputs() {
        let state = InteractionState::new();
        let mut input = InteractiveInput::new();
        input
            .interrupt_answers
            .insert("i1".to_owned(), json!("interrupt"));
        input.user_inputs.insert("i1".to_owned(), json!("node"));
        input
            .user_inputs
            .insert("n1".to_owned(), json!("node-only"));
        state.seed(&input);

        assert!(state.has_answer("i1"));
        assert_eq!(state.take_answer("i1"), Some(json!("interrupt")));
        // The node-addressed answer for an id already queued is ignored.
        assert_eq!(state.take_answer("i1"), None);
        assert_eq!(state.take_answer("n1"), Some(json!("node-only")));
        assert_eq!(state.take_answer("n1"), None);
        assert!(!state.has_answer("n1"));
    }

    #[test]
    fn take_answer_is_fifo_per_id() {
        let state = InteractionState::new();
        let mut first = InteractiveInput::new();
        first.interrupt_answers.insert("q".to_owned(), json!(1));
        state.seed(&first);
        let mut second = InteractiveInput::new();
        second.interrupt_answers.insert("q".to_owned(), json!(2));
        state.seed(&second);

        assert_eq!(state.take_answer("q"), Some(json!(1)));
        assert_eq!(state.take_answer("q"), Some(json!(2)));
        assert_eq!(state.take_answer("q"), None);
        assert_eq!(state.take_answer("missing"), None);
    }

    #[test]
    fn a_later_node_addressed_seed_is_appended_after_the_queued_interrupt_answer() {
        let state = InteractionState::new();
        let mut interrupt = InteractiveInput::new();
        interrupt
            .interrupt_answers
            .insert("q".to_owned(), json!("i"));
        state.seed(&interrupt);
        // The rule is per call: this call carries only the node-addressed answer, so it is
        // queued after the interrupt answer the earlier call left behind.
        let mut node = InteractiveInput::new();
        node.user_inputs.insert("q".to_owned(), json!("n"));
        state.seed(&node);

        assert_eq!(
            state.queued_answers(),
            BTreeMap::from([("q".to_owned(), vec![json!("i"), json!("n")])])
        );
        assert_eq!(state.take_answer("q"), Some(json!("i")));
        assert_eq!(state.take_answer("q"), Some(json!("n")));
        assert_eq!(state.take_answer("q"), None);
    }

    #[test]
    fn append_replays_answers_in_order_and_queued_answers_does_not_consume() {
        let state = InteractionState::new();
        let mut interactive = InteractiveInput::new();
        interactive
            .interrupt_answers
            .insert("q".to_owned(), json!(3));
        state.append(&BTreeMap::from([
            ("q".to_owned(), vec![json!(1), json!(2)]),
            ("other".to_owned(), vec![json!("a")]),
        ]));
        state.seed(&interactive);

        assert_eq!(
            state.queued_answers(),
            BTreeMap::from([
                ("other".to_owned(), vec![json!("a")]),
                ("q".to_owned(), vec![json!(1), json!(2), json!(3)]),
            ])
        );
        assert_eq!(
            state.queued_answers(),
            BTreeMap::from([
                ("other".to_owned(), vec![json!("a")]),
                ("q".to_owned(), vec![json!(1), json!(2), json!(3)]),
            ]),
            "reading the queues does not consume them"
        );

        assert_eq!(state.take_answer("q"), Some(json!(1)));
        assert_eq!(
            state.queued_answers().get("q").cloned(),
            Some(vec![json!(2), json!(3)])
        );
        assert_eq!(state.take_answer("q"), Some(json!(2)));
        assert_eq!(state.take_answer("q"), Some(json!(3)));
        assert_eq!(
            state.queued_answers(),
            BTreeMap::from([("other".to_owned(), vec![json!("a")])]),
            "only the untouched id keeps a queue"
        );
    }

    #[test]
    fn appending_an_empty_map_changes_nothing() {
        let state = InteractionState::new();
        state.append(&BTreeMap::new());
        assert!(state.queued_answers().is_empty());
    }

    #[test]
    fn next_index_counts_per_id() {
        let state = InteractionState::new();
        assert_eq!(state.next_index("a"), 0);
        assert_eq!(state.next_index("a"), 1);
        assert_eq!(state.next_index("a"), 2);
        assert_eq!(state.next_index("b"), 0);
        assert_eq!(state.next_index("a"), 3);
    }

    #[test]
    fn pending_preserves_insertion_order() {
        let state = InteractionState::new();
        state.push_pending(interaction("a", 0));
        state.push_pending(interaction("b", 0));
        state.push_pending(interaction("a", 1));

        assert_eq!(
            state.pending(),
            vec![
                interaction("a", 0),
                interaction("b", 0),
                interaction("a", 1)
            ]
        );
        assert_eq!(
            state.drain_pending(),
            vec![
                interaction("a", 0),
                interaction("b", 0),
                interaction("a", 1)
            ]
        );
        assert!(state.pending().is_empty());
        assert!(state.drain_pending().is_empty());
    }
}
