//! The rail-event drain seam: interaction events queued by a rail reach a host.
//!
//! Rails have no event sink of their own: a rail that wants to publish an
//! [`InteractionEvent`] appends its [`InteractionEvent::to_output_value`] frame
//! to an array in [`HarnessRailContext::extra_mut`]. This module turns that
//! convention into a working channel.
//!
//! # Key convention
//!
//! A channel is recognized when its key is [`GOAL_EVENT_EXTRA_KEY`]
//! (`harness.goal_events`, the key [`crate::tools::goal::GoalRail`] uses) or
//! starts with [`RAIL_EVENTS_EXTRA_PREFIX`] (`harness.events.`); the value is an
//! array of event objects, or one event object. Every other `extra` entry is
//! left untouched, so rails stay free to keep unrelated scratch data there.
//!
//! Events of one firing point are drained in `extra` key order (a `BTreeMap`)
//! and then in array order, so a round's event order is deterministic.
//!
//! # Wiring
//!
//! [`RailEventDrain`] is a [`HarnessRail`] registered on both the harness
//! registry and the inner kernel agent with [`RAIL_EVENT_DRAIN_PRIORITY`] (the
//! lowest possible priority), so it runs last at every firing point of both
//! layers and moves whatever the other rails queued into the [`RailEventBus`].
//! The bus retains the events for [`crate::agent::DeepAgent::drain_rail_events`]
//! and forwards each one to the live invocation stream, so a streaming caller
//! observes `__interaction__` frames ordered with the inner agent's own frames.
//!
//! The bus is per agent and survives across rounds; it keeps at most its
//! capacity (dropping the oldest event, with a debug log) so a host that never
//! drains it cannot grow memory without bound.
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::output::INTERACTION;
use openjiuwen_agent::AgentChunk;
use openjiuwen_common::error::Result;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::rails::{HarnessRail, HarnessRailContext};
use crate::schema::InteractionEvent;
use crate::tools::goal::GOAL_EVENT_EXTRA_KEY;

/// Prefix of a generic interaction-event channel in `ctx.extra`.
///
/// Any `Vec<InteractionEvent>` (or single event object) queued under
/// `harness.events.<name>` is drained and forwarded.
pub const RAIL_EVENTS_EXTRA_PREFIX: &str = "harness.events.";

/// Retained events of a [`RailEventBus`] that no host has drained yet.
pub const DEFAULT_RAIL_EVENT_CAPACITY: usize = 256;

/// Stable name of the drain rail.
pub const RAIL_EVENT_DRAIN_NAME: &str = "rail_events";

/// Dispatch priority of the drain rail: the lowest possible, so it runs last.
pub const RAIL_EVENT_DRAIN_PRIORITY: i32 = i32::MIN;

/// Removes and returns the interaction events queued in `ctx.extra`.
///
/// Both documented channels ([`GOAL_EVENT_EXTRA_KEY`] and every
/// [`RAIL_EVENTS_EXTRA_PREFIX`] key) are taken out of `extra`, so a second call
/// on the same context yields nothing. Unrelated `extra` entries stay in place.
/// An entry that is not an event object is logged and skipped, never fatal.
#[must_use]
pub fn drain_rail_events(ctx: &mut HarnessRailContext) -> Vec<InteractionEvent> {
    let channels: Vec<String> = ctx
        .extra()
        .keys()
        .filter(|key| is_event_channel(key))
        .cloned()
        .collect();
    let mut events = Vec::new();
    for channel in channels {
        let Some(value) = ctx.extra_mut().remove(&channel) else {
            continue;
        };
        decode_channel(&channel, value, &mut events);
    }
    events
}

/// Whether `key` names a rail event channel.
#[must_use]
fn is_event_channel(key: &str) -> bool {
    key == GOAL_EVENT_EXTRA_KEY || key.starts_with(RAIL_EVENTS_EXTRA_PREFIX)
}

/// Appends the events of one channel `value` to `events`.
fn decode_channel(channel: &str, value: Value, events: &mut Vec<InteractionEvent>) {
    let entries = match value {
        Value::Array(entries) => entries,
        Value::Object(_) => vec![value],
        _ => {
            tracing::warn!(
                channel,
                "ignoring a rail event channel that is neither an array nor an event object"
            );
            return;
        }
    };
    for entry in entries {
        match serde_json::from_value::<InteractionEvent>(entry) {
            Ok(event) => events.push(event),
            Err(error) => {
                tracing::warn!(channel, %error, "ignoring an undecodable rail event");
            }
        }
    }
}

/// Retained rail-queued events plus the live stream they are forwarded to.
///
/// One bus belongs to one [`crate::agent::DeepAgent`]; the drain rail pushes
/// into it and the invocation sets the stream for its duration. All operations
/// take a short `parking_lot` lock and never await, so a push can be issued from
/// any rail hook without deadlocking the loop.
pub struct RailEventBus {
    /// Retained events, oldest first.
    events: Mutex<VecDeque<InteractionEvent>>,
    /// Maximum retained events; `0` retains nothing (stream only).
    capacity: usize,
    /// Events dropped because the buffer was full.
    dropped: AtomicU64,
    /// Stream of the running invocation, when one is streaming.
    stream: RwLock<Option<UnboundedSender<Result<AgentChunk>>>>,
}

impl Default for RailEventBus {
    fn default() -> Self {
        Self::new(DEFAULT_RAIL_EVENT_CAPACITY)
    }
}

impl RailEventBus {
    /// Builds a bus retaining at most `capacity` events (`0` retains nothing).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            events: Mutex::new(VecDeque::new()),
            capacity,
            dropped: AtomicU64::new(0),
            stream: RwLock::new(None),
        }
    }

    /// Records `event` and forwards it to the live stream, when one is set.
    ///
    /// A full buffer drops its oldest event; forwarding happens first, so a
    /// streaming host never misses an event because the buffer was full.
    pub fn push(&self, event: InteractionEvent) {
        self.forward(&event);
        if self.capacity == 0 {
            return;
        }
        let mut buffer = self.events.lock();
        if buffer.len() >= self.capacity {
            buffer.pop_front();
            let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::debug!(
                dropped,
                capacity = self.capacity,
                "rail event buffer is full; dropping the oldest event"
            );
        }
        buffer.push_back(event);
    }

    /// Removes and returns every retained event, oldest first.
    pub fn drain(&self) -> Vec<InteractionEvent> {
        let mut buffer = self.events.lock();
        buffer.drain(..).collect()
    }

    /// Number of retained events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.lock().len()
    }

    /// Whether no event is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }

    /// The retention bound (`0` = retains nothing).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Events dropped so far because the buffer was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Installs (`Some`) or removes (`None`) the stream of the running invocation.
    pub(crate) fn set_stream(&self, stream: Option<UnboundedSender<Result<AgentChunk>>>) {
        *self.stream.write() = stream;
    }

    /// Sends `event` to the live stream as an `__interaction__` frame.
    fn forward(&self, event: &InteractionEvent) {
        let frame = AgentChunk::new(INTERACTION, 0, event.to_output_value());
        if let Some(tx) = self.stream.read().as_ref() {
            if tx.send(Ok(frame)).is_err() {
                tracing::debug!("rail event stream is closed; the frame was dropped");
            }
        }
    }
}

impl fmt::Debug for RailEventBus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RailEventBus")
            .field("retained", &self.len())
            .field("capacity", &self.capacity)
            .field("dropped", &self.dropped())
            .finish()
    }
}

/// The rail that moves queued events of a firing point into a [`RailEventBus`].
///
/// It fires last (lowest priority) at every harness and kernel firing point, so
/// a rail that queues in `after_invoke` (or any other kernel hook) is covered
/// too. A rail registered *after* it with [`RAIL_EVENT_DRAIN_PRIORITY`] fires
/// after the drain; its events stay queued until the next firing point.
pub struct RailEventDrain {
    /// Bus receiving the drained events.
    bus: Arc<RailEventBus>,
}

impl RailEventDrain {
    /// Builds a drain rail writing into `bus`.
    #[must_use]
    pub fn new(bus: Arc<RailEventBus>) -> Self {
        Self { bus }
    }

    /// The bus this rail drains into.
    #[must_use]
    pub fn bus(&self) -> &Arc<RailEventBus> {
        &self.bus
    }

    /// Moves every queued event of `ctx` into the bus.
    fn drain(&self, ctx: &mut HarnessRailContext) {
        for event in drain_rail_events(ctx) {
            self.bus.push(event);
        }
    }
}

impl fmt::Debug for RailEventDrain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RailEventDrain")
            .field("bus", &self.bus)
            .finish()
    }
}

#[async_trait]
impl HarnessRail for RailEventDrain {
    fn name(&self) -> &str {
        RAIL_EVENT_DRAIN_NAME
    }

    fn priority(&self) -> i32 {
        RAIL_EVENT_DRAIN_PRIORITY
    }

    async fn before_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn after_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn after_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn on_model_exception(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn before_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn after_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn on_tool_exception(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn after_react_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn on_user_message(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn before_steering_drain(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn before_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }

    async fn after_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        self.drain(ctx);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_agent::output::INTERACTION;
    use openjiuwen_agent::RailEvent;
    use serde_json::{json, Value};

    use super::{
        drain_rail_events, RailEventBus, RailEventDrain, RAIL_EVENTS_EXTRA_PREFIX,
        RAIL_EVENT_DRAIN_PRIORITY,
    };
    use crate::rails::{HarnessRail, HarnessRailContext};
    use crate::schema::{InteractionEvent, InteractionEventType};
    use crate::tools::goal::GOAL_EVENT_EXTRA_KEY;

    fn context() -> HarnessRailContext {
        HarnessRailContext::for_event(RailEvent::BeforeInvoke)
    }

    fn goal(value: &str) -> InteractionEvent {
        InteractionEvent::goal_updated(Some(json!({"id": value})))
    }

    #[test]
    fn drain_takes_both_channel_conventions_and_keeps_other_entries() {
        let mut ctx = context();
        let goal_frame = goal("g-1").to_output_value();
        let custom = InteractionEvent::execution_error("code", "boom", None).to_output_value();
        ctx.extra_mut()
            .insert(GOAL_EVENT_EXTRA_KEY.to_string(), json!([goal_frame]));
        ctx.extra_mut().insert(
            format!("{RAIL_EVENTS_EXTRA_PREFIX}custom"),
            json!([custom.clone()]),
        );
        ctx.extra_mut()
            .insert(format!("{RAIL_EVENTS_EXTRA_PREFIX}single"), custom);
        ctx.extra_mut().insert("scratch".to_string(), json!({"keep": true}));
        ctx.extra_mut()
            .insert("harness.other".to_string(), json!([goal("nope")]));

        let events = drain_rail_events(&mut ctx);

        // Deterministic order: `harness.events.custom`, `harness.events.single`,
        // then `harness.goal_events` (BTreeMap key order).
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, InteractionEventType::ExecutionError);
        assert_eq!(events[1].event_type, InteractionEventType::ExecutionError);
        assert_eq!(events[2].event_type, InteractionEventType::GoalUpdated);
        assert_eq!(events[2].payload.get("goal"), Some(&json!({"id": "g-1"})));

        // The channels are consumed; unrelated entries survive.
        assert!(drain_rail_events(&mut ctx).is_empty());
        assert!(ctx.extra().contains_key("scratch"));
        assert!(ctx.extra().contains_key("harness.other"));
        assert!(!ctx.extra().contains_key(GOAL_EVENT_EXTRA_KEY));
    }

    #[test]
    fn drain_skips_undecodable_entries_and_non_channel_values() {
        let mut ctx = context();
        ctx.extra_mut().insert(
            GOAL_EVENT_EXTRA_KEY.to_string(),
            json!([{"type": "goal.updated", "payload": {"goal": null}}, "nope", 7]),
        );
        ctx.extra_mut()
            .insert(format!("{RAIL_EVENTS_EXTRA_PREFIX}broken"), json!("text"));

        let events = drain_rail_events(&mut ctx);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, InteractionEventType::GoalUpdated);
    }

    #[test]
    fn retained_buffer_is_bounded_and_reports_drops() {
        let bus = RailEventBus::new(2);
        assert_eq!(bus.capacity(), 2);
        assert!(bus.is_empty());
        bus.push(goal("1"));
        bus.push(goal("2"));
        bus.push(goal("3"));
        assert_eq!(bus.len(), 2);
        assert_eq!(bus.dropped(), 1);
        let events = bus.drain();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].payload.get("goal"), Some(&json!({"id": "2"})));
        assert_eq!(events[1].payload.get("goal"), Some(&json!({"id": "3"})));
        assert!(bus.is_empty());
    }

    #[test]
    fn zero_capacity_retains_nothing() {
        let bus = RailEventBus::new(0);
        bus.push(goal("1"));
        assert!(bus.is_empty());
        assert_eq!(bus.dropped(), 0);
    }

    #[tokio::test]
    async fn pushed_events_are_forwarded_as_interaction_frames() {
        let bus = std::sync::Arc::new(RailEventBus::default());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        bus.set_stream(Some(tx));

        let drain = RailEventDrain::new(bus.clone());
        assert_eq!(drain.name(), "rail_events");
        assert_eq!(drain.priority(), RAIL_EVENT_DRAIN_PRIORITY);
        assert_eq!(drain.priority(), i32::MIN);

        let mut ctx = context();
        ctx.extra_mut()
            .insert(GOAL_EVENT_EXTRA_KEY.to_string(), json!([goal("g").to_output_value()]));
        drain.before_invoke(&mut ctx).await.unwrap();
        assert!(ctx.extra().get(GOAL_EVENT_EXTRA_KEY).is_none());
        assert_eq!(bus.len(), 1);

        let chunk = rx.try_recv().unwrap().unwrap();
        assert_eq!(chunk.kind, INTERACTION);
        let frame: Value = chunk.payload;
        assert_eq!(frame["type"], json!("goal.updated"));
        assert_eq!(frame["index"], json!(0));
        assert_eq!(frame["payload"]["goal"], json!({"id": "g"}));

        bus.set_stream(None);
        drain.after_invoke(&mut ctx).await.unwrap();
        assert!(rx.try_recv().is_err());
    }
}
