//! Agents-as-tools hierarchical teams.
//!
//! Python parity: `openjiuwen/core/multi_agent/teams/hierarchical_tools/`. One
//! root agent sits above a flat member list, and every sub-agent is exposed to
//! the root's model as a tool: [`AgentTool`](crate::AgentTool) mounted on the
//! root's [`ToolHost`](crate::ToolHost). Unlike the message-bus topology the
//! sub-agents are not dispatched through the bus by the root (the model calls
//! them in-process), and the team's own `stream` bypasses the bus entirely: it
//! resolves the root instance and forwards its agent frames as team frames.
//!
//! Wiring is deferred: [`TeamStrategy::on_member_added`] only records a
//! `(parent, child)` pair, and the first run mounts the child tools on the
//! parent instance
//! ([`setup`](TeamStrategy::setup)). The wired instance is pinned in the runtime
//! so later `send`s reach the instance that owns the tools, instead of a fresh
//! provider instance without them (documented divergence from the Python
//! fresh-provider trap).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::Tool;
use openjiuwen_session::{AgentTeamSession, Session, StreamItem};
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::agent_tool::AgentTool;
use crate::card::TeamCard;
use crate::config::TeamConfig;
use crate::member::TeamAgent;
use crate::session::SessionLease;
use crate::strategy::{StrategyContext, TeamStrategy};
use crate::team::BaseTeam;

/// The strategy kind reported by an agents-as-tools hierarchical team.
const HIERARCHICAL_TOOLS_KIND: &str = "hierarchical_tools";

/// Frame kind of the team's own frames (Python's default `OutputSchema` type).
const TEAM_FRAME_KIND: &str = "message";

/// Value of `result_type` in the frame that reports a failed agent stream.
const ERROR_RESULT_TYPE: &str = "error";

/// Index of the team's own frames (Python's default `OutputSchema` index).
const TEAM_FRAME_INDEX: i64 = 0;

/// Configuration of an agents-as-tools hierarchical team.
///
/// Python parity: `HierarchicalTeamConfig`. It subclasses `TeamConfig`, so Rust
/// composes the team limits under [`HierarchicalToolsConfig::team`] while
/// serializing them flattened into the same JSON object. `root_agent` is the
/// entry point: it must be registered in the team runtime, and every member
/// added with that agent as its parent gets mounted on it as a tool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HierarchicalToolsConfig {
    /// The team limits shared by every strategy.
    #[serde(flatten)]
    pub team: TeamConfig,
    /// Card of the root (entry point) agent.
    pub root_agent: AgentCard,
}

impl HierarchicalToolsConfig {
    /// Builds a configuration with the Python defaults and `root_agent`.
    #[must_use]
    pub fn new(root_agent: AgentCard) -> Self {
        Self {
            root_agent,
            ..Self::default()
        }
    }
}

/// Shapes the message a root agent streams with.
///
/// Python parity: an object input keeps its keys and gets `conversation_id` and
/// `sender` overwritten; any other input is wrapped under `query` so the agent
/// receives the same string-or-object contract as a bus delivery.
fn shape_input(input: Value, session_id: &str, sender: &str) -> Value {
    match input {
        Value::Object(mut map) => {
            map.insert(
                "conversation_id".to_owned(),
                Value::String(session_id.to_owned()),
            );
            map.insert("sender".to_owned(), Value::String(sender.to_owned()));
            Value::Object(map)
        }
        other => json!({
            "query": other,
            "conversation_id": session_id,
            "sender": sender,
        }),
    }
}

/// The execution strategy of an agents-as-tools hierarchical team.
///
/// The strategy owns the root id plus the wiring state it accumulates between
/// construction and the first run: the pending `(parent, child)` pairs recorded
/// by membership events, the instances it wired, and a "wired" flag that makes
/// [`TeamStrategy::setup`] idempotent.
///
/// The interior state is `Arc`-shared so a `stream` can move a clone of the
/// strategy into its run task.
#[derive(Clone)]
pub struct HierarchicalToolsStrategy {
    /// Id of the root agent registered in the team runtime.
    root_id: String,
    /// Child cards queued for a parent, keyed by parent agent id.
    pending_children: Arc<RwLock<BTreeMap<String, Vec<AgentCard>>>>,
    /// Parent instances this strategy already wired, keyed by parent agent id.
    ///
    /// The runtime keeps pinned instances private, so the strategy remembers the
    /// instance it mounted tools on and reuses it for its own bus-free stream.
    handles: Arc<RwLock<BTreeMap<String, TeamAgent>>>,
    /// Serializes concurrent [`TeamStrategy::setup`] calls.
    setup_lock: Arc<Mutex<()>>,
    /// `true` once the queued children were mounted and pinned.
    wired: Arc<AtomicBool>,
}

impl std::fmt::Debug for HierarchicalToolsStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HierarchicalToolsStrategy")
            .field("root_id", &self.root_id)
            .field("pending_parents", &self.pending_children.read().len())
            .field("wired_parents", &self.handles.read().len())
            .field("wired", &self.wired.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl HierarchicalToolsStrategy {
    /// Builds the strategy for `config`.
    ///
    /// The team limits are not stored: the team object owns them.
    #[must_use]
    pub fn new(config: HierarchicalToolsConfig) -> Arc<Self> {
        Arc::new(Self {
            root_id: config.root_agent.id,
            pending_children: Arc::new(RwLock::new(BTreeMap::new())),
            handles: Arc::new(RwLock::new(BTreeMap::new())),
            setup_lock: Arc::new(Mutex::new(())),
            wired: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Returns `true` when nothing is queued and the hierarchy was wired.
    ///
    /// A later membership event queues new work, so this is not a one-way latch.
    fn is_wired(&self) -> bool {
        self.wired.load(Ordering::SeqCst) && self.pending_children.read().is_empty()
    }

    /// Resolves the instance that represents `parent`.
    ///
    /// An instance this strategy already wired wins (it owns the mounted child
    /// tools); otherwise the runtime's provider builds a fresh one.
    fn resolve_handle(&self, ctx: &StrategyContext, parent: &str) -> Option<TeamAgent> {
        if let Some(handle) = self.handles.read().get(parent) {
            return Some(handle.clone());
        }
        ctx.runtime
            .agent_provider(parent)
            .map(|provider| provider())
    }

    /// Mounts one [`AgentTool`] per queued child on `parent` and pins the pair.
    ///
    /// A missing parent, a parent without a tool surface, and a rejected mount
    /// are logged and skipped: Python treats the same situations as a failed
    /// attribute probe, never as an error.
    fn wire_parent(&self, ctx: &StrategyContext, parent: &str, children: &[AgentCard]) {
        if children.is_empty() {
            tracing::warn!(
                parent = %parent,
                "hierarchical_tools: no children queued for the parent; skipping"
            );
            return;
        }
        let Some(handle) = self.resolve_handle(ctx, parent) else {
            tracing::warn!(
                parent = %parent,
                "hierarchical_tools: parent agent is not registered in the runtime; \
                 skipping its child tools"
            );
            return;
        };
        handle.bind(&ctx.runtime, parent);
        match handle.tool_host() {
            None => tracing::warn!(
                parent = %parent,
                "hierarchical_tools: parent agent cannot host tools; skipping its child tools"
            ),
            Some(host) => {
                for child in children {
                    let tool: Arc<dyn Tool> =
                        Arc::new(AgentTool::new(child, Arc::downgrade(&ctx.runtime)));
                    if !host.mount_tool(tool) {
                        tracing::warn!(
                            parent = %parent,
                            child = %child.id,
                            "hierarchical_tools: the parent rejected the child tool"
                        );
                    }
                }
            }
        }
        // Later bus deliveries must reach the instance that owns the tools.
        ctx.runtime.pin_agent(parent, handle.clone());
        self.handles.write().insert(parent.to_owned(), handle);
    }

    /// Streams one root-agent run into the team session.
    ///
    /// The run bypasses the bus: the root is resolved from the runtime, bound,
    /// streamed in-process, and every agent chunk is written to the team session
    /// as a team frame. An agent failure becomes one `error` frame and ends the
    /// run successfully (Python swallows the agent error); a root that cannot be
    /// resolved is the run's error.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::NotFound`] when the root is not
    /// registered in the runtime, and whatever wiring the setup stage reports.
    async fn run_stream(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Arc<AgentTeamSession>,
    ) -> Result<()> {
        self.setup(ctx).await?;
        let handle = self.resolve_handle(ctx, &self.root_id).ok_or_else(|| {
            Error::not_found(format!(
                "Agent '{}' not registered in runtime",
                self.root_id
            ))
        })?;
        handle.bind(&ctx.runtime, &self.root_id);

        let shaped = shape_input(input, session.id(), &ctx.card.id);
        let mut frames = handle.agent().stream(shaped, None);
        while let Some(frame) = frames.next().await {
            match frame {
                Ok(chunk) => {
                    if let Err(error) = session.write_stream(chunk.to_stream_item()).await {
                        tracing::warn!(
                            root = %self.root_id,
                            %error,
                            "hierarchical_tools: failed to write an agent frame \
                             to the team stream"
                        );
                    }
                }
                Err(error) => {
                    let payload = json!({
                        "output": error.message(),
                        "result_type": ERROR_RESULT_TYPE,
                    });
                    let frame =
                        StreamItem::output(TEAM_FRAME_KIND, Some(TEAM_FRAME_INDEX), payload);
                    if let Err(write_error) = session.write_stream(frame).await {
                        tracing::warn!(
                            root = %self.root_id,
                            %write_error,
                            "hierarchical_tools: failed to write the stream error frame \
                             to the team stream"
                        );
                    }
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl TeamStrategy for HierarchicalToolsStrategy {
    fn kind(&self) -> &str {
        HIERARCHICAL_TOOLS_KIND
    }

    /// Queues a child for later mounting when a parent is given.
    ///
    /// Nothing is registered or mounted here: the hierarchy is wired on the
    /// first run, so a member added before its parent still works.
    fn on_member_added(
        &self,
        ctx: &StrategyContext,
        card: &AgentCard,
        options: &crate::MemberOptions,
    ) {
        let _ = ctx;
        let Some(parent) = options
            .parent_agent_id
            .as_deref()
            .filter(|parent| !parent.is_empty())
        else {
            return;
        };
        self.pending_children
            .write()
            .entry(parent.to_owned())
            .or_default()
            .push(card.clone());
    }

    /// Drops the queued children of a removed agent and the agent itself.
    fn on_member_removed(&self, ctx: &StrategyContext, agent_id: &str) {
        let _ = ctx;
        let mut pending = self.pending_children.write();
        pending.remove(agent_id);
        for children in pending.values_mut() {
            children.retain(|child| child.id != agent_id);
        }
        drop(pending);
        self.handles.write().remove(agent_id);
    }

    /// Requires a configured root that is registered in the runtime.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::Config`] when the root id is empty
    /// or the root was never registered.
    fn check_ready(&self, ctx: &StrategyContext) -> Result<()> {
        if self.root_id.is_empty() {
            return Err(Error::config(
                "No root agent configured in HierarchicalTeamConfig.",
            ));
        }
        if !ctx.runtime.has_agent(&self.root_id) {
            return Err(Error::config(format!(
                "Root agent '{}' is not registered in runtime. \
                 Call add_agent(root_card, root_provider) before invoke()/stream().",
                self.root_id
            )));
        }
        Ok(())
    }

    /// Mounts the queued child tools on their parents, once.
    ///
    /// The wired flag is checked before the setup lock and again after
    /// acquiring it (the double-checked locking Python performs with
    /// `asyncio.Lock`), and the pending map is snapshotted so no lock is held
    /// across the `.await` of a provider resolution.
    ///
    /// # Errors
    ///
    /// Never fails today: a parent that cannot be resolved or cannot host tools
    /// is logged and skipped, matching Python's silent attribute probe.
    async fn setup(&self, ctx: &StrategyContext) -> Result<()> {
        if self.is_wired() {
            return Ok(());
        }
        let _guard = self.setup_lock.lock().await;
        if self.is_wired() {
            return Ok(());
        }
        let pending = self.pending_children.read().clone();
        for (parent, children) in pending {
            self.wire_parent(ctx, &parent, &children);
        }
        self.pending_children.write().clear();
        self.wired.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Sends `input` to the root agent over the bus.
    ///
    /// # Errors
    ///
    /// Propagates the readiness check, the setup stage, the runtime's validation
    /// of the sender and recipient, the root's own failure, and the lease
    /// teardown.
    async fn invoke(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> Result<Value> {
        self.check_ready(ctx)?;
        self.setup(ctx).await?;
        let lease = SessionLease::acquire(&ctx.runtime, &ctx.card, &input, session).await;
        let session_id = lease.session_id().to_owned();
        let outcome = ctx
            .runtime
            .send(input, &self.root_id, &ctx.card.id, Some(&session_id), None)
            .await;
        let finished = lease.finish().await;
        match outcome {
            Err(error) => Err(error),
            Ok(value) => finished.map(|()| value),
        }
    }

    /// Streams the root agent in-process, bypassing the bus.
    ///
    /// A readiness or setup failure is yielded as the stream's error rather than
    /// raising out of this call, so a team stream always reports failures as
    /// data.
    fn stream(
        &self,
        ctx: &StrategyContext,
        input: Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>> {
        if let Err(error) = self.check_ready(ctx) {
            return futures::stream::once(async move { Err(error) }).boxed();
        }
        let strategy = self.clone();
        let run_ctx = ctx.clone();
        let run_input = input.clone();
        crate::session::team_stream(
            Arc::clone(&ctx.runtime),
            ctx.card.clone(),
            input,
            session,
            move |team_session| {
                let strategy = strategy.clone();
                let ctx = run_ctx.clone();
                let input = run_input.clone();
                async move { strategy.run_stream(&ctx, input, team_session).await }.boxed()
            },
        )
    }
}

/// Builds an agents-as-tools hierarchical team from its card and configuration.
#[must_use]
pub fn hierarchical_tools_team(card: TeamCard, config: HierarchicalToolsConfig) -> BaseTeam {
    BaseTeam::with_strategy(
        card,
        config.team.clone(),
        HierarchicalToolsStrategy::new(config),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use async_trait::async_trait;
    use futures::StreamExt;
    use openjiuwen_agent::{Agent, AgentChunk, AgentOutput, ReActAgent};
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::Session;
    use serde_json::json;

    use super::*;
    use crate::handoff::support::agent_card;
    use crate::runtime::{RuntimeConfig, TeamRuntime};
    use crate::team::Team;
    use crate::{AgentProvider, MemberOptions};

    /// One frame a [`StreamingAgent`] yields, in order.
    #[derive(Clone)]
    enum Frame {
        /// A regular agent chunk.
        Chunk(AgentChunk),
        /// A failure that ends the agent stream.
        Fail(String),
    }

    /// An [`Agent`] that answers `invoke` and replays `frames` on `stream`.
    struct StreamingAgent {
        /// The card the agent reports.
        card: AgentCard,
        /// Frames yielded by `stream`, in order.
        frames: Vec<Frame>,
        /// Inputs the agent was streamed with.
        seen: Arc<parking_lot::Mutex<Option<Value>>>,
    }

    impl StreamingAgent {
        /// Builds an agent named `name` that yields `frames`.
        fn new(name: &str, frames: Vec<Frame>) -> Arc<Self> {
            Arc::new(Self {
                card: agent_card(name),
                frames,
                seen: Arc::new(parking_lot::Mutex::new(None)),
            })
        }

        /// Returns the inputs of the last `stream` call.
        fn seen(&self) -> Option<Value> {
            self.seen.lock().clone()
        }
    }

    #[async_trait]
    impl Agent for StreamingAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            Ok(AgentOutput::answer("root"))
        }

        fn stream(
            &self,
            inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            *self.seen.lock() = Some(inputs);
            let frames = self.frames.clone();
            futures::stream::iter(frames.into_iter().map(|frame| match frame {
                Frame::Chunk(chunk) => Ok(chunk),
                Frame::Fail(message) => Err(Error::tool(message)),
            }))
            .boxed()
        }
    }

    fn config(id: &str) -> HierarchicalToolsConfig {
        HierarchicalToolsConfig::new(agent_card(id))
    }

    fn parent_options(parent: Option<&str>) -> MemberOptions {
        MemberOptions {
            parent_agent_id: parent.map(ToOwned::to_owned),
        }
    }

    fn provider(agent: Arc<dyn Agent>) -> AgentProvider {
        Arc::new(move || TeamAgent::new(Arc::clone(&agent)))
    }

    fn react_provider(agent: Arc<ReActAgent>) -> AgentProvider {
        Arc::new(move || TeamAgent::react(Arc::clone(&agent)))
    }

    fn context(strategy_config: &HierarchicalToolsConfig) -> StrategyContext {
        StrategyContext::new(
            TeamCard::new("team"),
            strategy_config.team.clone(),
            vec![],
            TeamRuntime::new(RuntimeConfig::new()),
        )
    }

    #[test]
    fn config_defaults_match_python() {
        let defaults = HierarchicalToolsConfig::default();
        assert_eq!(defaults.team, TeamConfig::default());
        assert_eq!(defaults.team.max_agents, 10);
        assert_eq!(defaults.team.max_concurrent_messages, 100);
        assert!((defaults.team.message_timeout - 30.0).abs() < f64::EPSILON);
        assert_eq!(defaults.root_agent.name, "");
        assert_eq!(defaults.root_agent.id.len(), 32);
    }

    #[test]
    fn config_new_sets_the_root_and_keeps_defaults() {
        let card = agent_card("root");
        let parsed = HierarchicalToolsConfig::new(card.clone());
        assert_eq!(parsed.root_agent, card);
        assert_eq!(parsed.team, TeamConfig::default());
    }

    #[test]
    fn config_json_is_flat_defaulted_and_tolerant() {
        let empty: HierarchicalToolsConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty.team, TeamConfig::default());

        let parsed: HierarchicalToolsConfig = serde_json::from_value(json!({
            "max_agents": 4,
            "message_timeout": 2.5,
            "root_agent": {"id": "root", "name": "root"},
            "surprise": true
        }))
        .unwrap();
        assert_eq!(parsed.team.max_agents, 4);
        assert_eq!(parsed.team.max_concurrent_messages, 100);
        assert!((parsed.team.message_timeout - 2.5).abs() < f64::EPSILON);
        assert_eq!(parsed.root_agent.id, "root");

        let wire = serde_json::to_value(&parsed).unwrap();
        assert_eq!(wire["max_agents"], json!(4));
        assert_eq!(wire["max_concurrent_messages"], json!(100));
        assert_eq!(wire["root_agent"]["id"], json!("root"));
        let reparsed: HierarchicalToolsConfig = serde_json::from_value(wire).unwrap();
        assert_eq!(reparsed, parsed);
    }

    #[test]
    fn kind_is_the_python_strategy_name() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        assert_eq!(strategy.kind(), "hierarchical_tools");
    }

    #[test]
    fn on_member_added_queues_children_only_with_a_parent() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));

        strategy.on_member_added(&ctx, &agent_card("a"), &parent_options(Some("root")));
        strategy.on_member_added(&ctx, &agent_card("b"), &parent_options(Some("root")));
        strategy.on_member_added(&ctx, &agent_card("orphan"), &parent_options(None));
        strategy.on_member_added(&ctx, &agent_card("blank"), &parent_options(Some("")));

        let pending = strategy.pending_children.read();
        assert_eq!(pending.len(), 1);
        let children = pending.get("root").unwrap();
        assert_eq!(
            children
                .iter()
                .map(|card| card.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        // Queuing has no registration side effect: wiring happens on setup.
        assert_eq!(ctx.runtime.agent_count(), 0);
    }

    #[test]
    fn on_member_removed_drops_queued_entries() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        strategy.on_member_added(&ctx, &agent_card("a"), &parent_options(Some("root")));
        strategy.on_member_added(&ctx, &agent_card("b"), &parent_options(Some("root")));
        strategy.on_member_added(&ctx, &agent_card("c"), &parent_options(Some("a")));

        strategy.on_member_removed(&ctx, "a");
        let pending = strategy.pending_children.read();
        assert!(pending.get("a").is_none());
        assert_eq!(
            pending
                .get("root")
                .unwrap()
                .iter()
                .map(|card| card.id.as_str())
                .collect::<Vec<_>>(),
            vec!["b"]
        );
        drop(pending);

        strategy.on_member_removed(&ctx, "root");
        assert!(strategy.pending_children.read().is_empty());
    }

    #[test]
    fn check_ready_rejects_an_empty_root() {
        let strategy = HierarchicalToolsStrategy::new(config(""));
        let error = strategy.check_ready(&context(&config(""))).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.message(),
            "No root agent configured in HierarchicalTeamConfig."
        );
    }

    #[test]
    fn check_ready_rejects_an_unregistered_root() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let error = strategy.check_ready(&context(&config("root"))).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.message(),
            "Root agent 'root' is not registered in runtime. Call \
             add_agent(root_card, root_provider) before invoke()/stream()."
        );
    }

    #[test]
    fn check_ready_accepts_a_registered_root() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        ctx.runtime.register_agent(
            agent_card("root"),
            provider(Arc::new(ReActAgent::new(agent_card("root")))),
        );
        strategy.check_ready(&ctx).unwrap();
    }

    #[test]
    fn shape_input_overwrites_session_keys_of_an_object() {
        let shaped = shape_input(
            json!({"query": "hi", "conversation_id": "stale", "sender": "stale", "extra": 1}),
            "session-1",
            "team",
        );
        assert_eq!(
            shaped,
            json!({
                "query": "hi",
                "conversation_id": "session-1",
                "sender": "team",
                "extra": 1,
            })
        );
    }

    #[test]
    fn shape_input_wraps_a_scalar_under_query() {
        assert_eq!(
            shape_input(json!("hi"), "session-1", "team"),
            json!({"query": "hi", "conversation_id": "session-1", "sender": "team"})
        );
        assert_eq!(
            shape_input(Value::Null, "session-1", "team"),
            json!({"query": null, "conversation_id": "session-1", "sender": "team"})
        );
    }

    #[tokio::test]
    async fn setup_mounts_child_tools_on_the_parent_and_pins_it() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        let root = Arc::new(ReActAgent::new(agent_card("root")));
        ctx.runtime
            .register_agent(agent_card("root"), react_provider(Arc::clone(&root)));
        strategy.on_member_added(&ctx, &agent_card("a"), &parent_options(Some("root")));
        strategy.on_member_added(&ctx, &agent_card("b"), &parent_options(Some("root")));

        strategy.setup(&ctx).await.unwrap();

        let tools = root.tools();
        assert_eq!(tools.len(), 2);
        let mut names: Vec<&str> = tools.iter().map(|card| card.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["a", "b"]);
        assert!(strategy.pending_children.read().is_empty());
        assert!(strategy.handles.read().contains_key("root"));

        // A second setup is a no-op: the tools are neither re-mounted nor lost.
        strategy.setup(&ctx).await.unwrap();
        assert_eq!(root.tools().len(), 2);
    }

    #[tokio::test]
    async fn setup_reuses_the_wired_instance_for_later_children() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        let root = Arc::new(ReActAgent::new(agent_card("root")));
        ctx.runtime
            .register_agent(agent_card("root"), react_provider(Arc::clone(&root)));

        strategy.on_member_added(&ctx, &agent_card("a"), &parent_options(Some("root")));
        strategy.setup(&ctx).await.unwrap();
        strategy.on_member_added(&ctx, &agent_card("b"), &parent_options(Some("root")));
        strategy.setup(&ctx).await.unwrap();

        assert_eq!(root.tools().len(), 2);
    }

    #[tokio::test]
    async fn setup_skips_a_parent_without_a_tool_host() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        ctx.runtime.register_agent(
            agent_card("root"),
            provider(Arc::new(ReActAgent::new(agent_card("root")))),
        );
        strategy.on_member_added(&ctx, &agent_card("a"), &parent_options(Some("root")));

        strategy.setup(&ctx).await.unwrap();
        assert!(strategy.pending_children.read().is_empty());
    }

    #[tokio::test]
    async fn setup_skips_an_unregistered_parent() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        strategy.on_member_added(&ctx, &agent_card("a"), &parent_options(Some("ghost")));

        strategy.setup(&ctx).await.unwrap();
        assert!(strategy.pending_children.read().is_empty());
        assert_eq!(ctx.runtime.agent_count(), 0);
    }

    #[tokio::test]
    async fn invoke_sends_the_input_to_the_root() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        let root = crate::handoff::support::FakeAgent::echo("root");
        ctx.runtime.register_agent(
            agent_card("root"),
            provider(Arc::clone(&root) as Arc<dyn Agent>),
        );

        let result = strategy
            .invoke(&ctx, json!({"query": "hi"}), None)
            .await
            .unwrap();
        assert_eq!(result["output"], json!("root:hi"));
        assert_eq!(result["result_type"], json!("answer"));
    }

    #[tokio::test]
    async fn stream_forwards_agent_frames_and_shapes_the_input() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        let agent = StreamingAgent::new(
            "root",
            vec![
                Frame::Chunk(AgentChunk::new("llm_output", 0, json!({"content": "hi"}))),
                Frame::Chunk(AgentChunk::new("answer", 1, json!({"output": "hi"}))),
            ],
        );
        ctx.runtime.register_agent(
            agent_card("root"),
            provider(Arc::clone(&agent) as Arc<dyn Agent>),
        );

        let frames: Vec<Value> = strategy
            .stream(&ctx, json!({"query": "hi"}), None)
            .map(|frame| frame.unwrap())
            .collect()
            .await;
        assert_eq!(
            frames,
            vec![
                json!({"type": "llm_output", "index": 0, "payload": {"content": "hi"}}),
                json!({"type": "answer", "index": 1, "payload": {"output": "hi"}}),
            ]
        );

        let seen = agent.seen().unwrap();
        assert_eq!(seen["query"], json!("hi"));
        assert_eq!(seen["sender"], json!(ctx.card.id));
        assert!(seen["conversation_id"]
            .as_str()
            .is_some_and(|id| !id.is_empty()));
    }

    #[tokio::test]
    async fn stream_turns_an_agent_failure_into_one_error_frame() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));
        let agent = StreamingAgent::new(
            "root",
            vec![
                Frame::Chunk(AgentChunk::new("llm_output", 0, json!({"content": "hi"}))),
                Frame::Fail("model exploded".to_owned()),
                Frame::Chunk(AgentChunk::new("answer", 1, json!({"output": "ignored"}))),
            ],
        );
        ctx.runtime.register_agent(
            agent_card("root"),
            provider(Arc::clone(&agent) as Arc<dyn Agent>),
        );

        let frames: Vec<Value> = strategy
            .stream(&ctx, json!("hi"), None)
            .map(|frame| frame.unwrap())
            .collect()
            .await;
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0]["type"], json!("llm_output"));
        assert_eq!(frames[1]["type"], json!("message"));
        assert_eq!(
            frames[1]["payload"],
            json!({"output": "model exploded", "result_type": "error"})
        );
    }

    #[tokio::test]
    async fn stream_reports_a_readiness_failure_as_an_error_frame_stream() {
        let strategy = HierarchicalToolsStrategy::new(config("root"));
        let ctx = context(&config("root"));

        let mut stream = strategy.stream(&ctx, json!("hi"), None);
        let error = stream.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error
            .message()
            .contains("Root agent 'root' is not registered"));
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn team_builder_attaches_the_strategy() {
        let team = hierarchical_tools_team(TeamCard::new("team"), config("root"));
        assert_eq!(team.kind(), "hierarchical_tools");
        assert_eq!(team.team_id(), "team");
        assert_eq!(team.config(), TeamConfig::default());
    }
}
