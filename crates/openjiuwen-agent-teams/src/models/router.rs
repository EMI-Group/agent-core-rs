//! Per-request model routing over a [`ModelPool`].
//!
//! Python parity source: `openjiuwen/agent_teams/models/allocator.py` (`RouterAllocator`,
//! `ByModelNameAllocator`, `RoundRobinModelAllocator`) plus the router/intelli-router strategies
//! that Python delegates to a separate client.
//!
//! # Composition with `runtime::ModelAllocator`
//!
//! The two components answer different questions and neither replaces the other:
//!
//! * [`crate::runtime::allocator::ModelAllocator`] runs **once per member at build time**. It assigns
//!   one pool entry to the leader/teammate and persists that choice as
//!   `MemberRecord::model_ref` (`model_name` + pool index), so a restarted team rehydrates the same
//!   endpoint and a member's model never changes mid-conversation.
//! * [`ModelRouter`] runs **once per request**. It chooses among the same pool to spread load
//!   (least-busy / round-robin), to honour an explicit model request, or to apply team routing rules.
//!   It keeps no persisted state: the cursor and the in-flight counters are process-local and
//!   worthless after a restart, so they are never written to the store.
//!
//! A team normally uses both: the allocator pins each member to an endpoint, and a router in front
//! of the provider call can still route a single request elsewhere (for example to the entry that
//! is not rate-limited). A team with no pool uses neither and every member keeps the model its own
//! spec declares.
//!
//! ```
//! use openjiuwen_agent_teams::models::{ModelPool, ModelRouter, RequestHint, RouterStrategy};
//! use openjiuwen_agent_teams::schema::{ModelPoolEntry, TeamRole};
//!
//! let entries = vec![
//!     ModelPoolEntry::new("gpt-4o", "openai", "https://api.openai.com/v1", "sk-a"),
//!     ModelPoolEntry::new("claude", "anthropic", "https://api.anthropic.com", "sk-b"),
//! ];
//! let router = ModelRouter::new(ModelPool::from_entries(&entries).expect("pool"), RouterStrategy::ByName);
//! let hint = RequestHint::for_role(TeamRole::Leader).with_model_name("claude");
//! let guard = router.acquire(&hint).expect("pool is not empty");
//! assert_eq!(guard.entry().model_name, "claude");
//! assert_eq!(router.total_in_flight(), 1);
//! drop(guard);
//! assert_eq!(router.total_in_flight(), 0);
//! ```

use std::fmt;

use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::schema::{ModelPoolEntry, ModelPoolStrategy, TeamRole};

use super::pool::ModelPool;

/// Error returned when no endpoint can be chosen because the team declares no pool.
const NO_MODEL_MESSAGE: &str = "no model configured for requests; set model_pool on the team spec";

/// How a [`ModelRouter`] picks an endpoint for one request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterStrategy {
    /// Serve the model the request named; fall back to the first entry.
    ByName,
    /// Rotate through the pool in pool order, ignoring the requested name; the default.
    #[default]
    RoundRobin,
    /// Serve the entry with the fewest in-flight requests, tie-broken by pool position.
    LeastBusy,
    /// Apply the router's [`RoutingRule`]s, then fall back to [`RouterStrategy::ByName`].
    RuleBased,
}

impl RouterStrategy {
    /// Returns the wire string of this strategy (`"by_name"`, `"round_robin"`, …).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ByName => "by_name",
            Self::RoundRobin => "round_robin",
            Self::LeastBusy => "least_busy",
            Self::RuleBased => "rule_based",
        }
    }
}

impl From<ModelPoolStrategy> for RouterStrategy {
    /// Maps the build-time allocation strategy onto the per-request strategy of the same name.
    ///
    /// `RoundRobin` → [`RouterStrategy::RoundRobin`], `ByModelName` → [`RouterStrategy::ByName`]:
    /// a team configured to spread members over the pool also spreads its requests, and a team
    /// configured by name keeps honouring the requested name.
    fn from(strategy: ModelPoolStrategy) -> Self {
        match strategy {
            ModelPoolStrategy::RoundRobin => Self::RoundRobin,
            ModelPoolStrategy::ByModelName => Self::ByName,
        }
    }
}

/// How demanding a request is, used by [`RuleMatcher`] thresholds.
///
/// Ordered from [`Complexity::Low`] to [`Complexity::High`], so `min_complexity` is an ordinary
/// `>=` comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Complexity {
    /// Cheap, small-context requests (classification, short replies).
    Low,
    /// Default tier for ordinary member work.
    Medium,
    /// Long-context or multi-step reasoning requests; the most expensive tier.
    High,
}

/// What the caller knows about a request, used both to route it and to match rules.
///
/// Every field is a hint: a strategy may ignore any of them, and a [`RuleMatcher`] treats a `None`
/// constraint as unconstrained.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RequestHint {
    /// Model name the caller asked for, if any.
    pub model_name: Option<String>,
    /// Role of the member issuing the request.
    pub role: Option<TeamRole>,
    /// Caller-defined task kind (`"review"`, `"coding"`, …), matched verbatim.
    pub task_kind: Option<String>,
    /// Estimated request complexity.
    pub complexity: Option<Complexity>,
    /// Whether the request carries tool definitions; `false` when unknown.
    pub needs_tools: bool,
}

impl RequestHint {
    /// Returns an empty hint: no model, no role, no task kind, no complexity, no tools.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a hint that only carries the member role.
    #[must_use]
    pub fn for_role(role: TeamRole) -> Self {
        Self {
            role: Some(role),
            ..Self::default()
        }
    }

    /// Sets the requested model name.
    #[must_use]
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    /// Sets the member role.
    #[must_use]
    pub fn with_role(mut self, role: TeamRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Sets the task kind.
    #[must_use]
    pub fn with_task_kind(mut self, task_kind: impl Into<String>) -> Self {
        self.task_kind = Some(task_kind.into());
        self
    }

    /// Sets the estimated complexity.
    #[must_use]
    pub fn with_complexity(mut self, complexity: Complexity) -> Self {
        self.complexity = Some(complexity);
        self
    }

    /// Sets whether the request carries tool definitions.
    #[must_use]
    pub fn with_needs_tools(mut self, needs_tools: bool) -> Self {
        self.needs_tools = needs_tools;
        self
    }
}

/// Conjunction of constraints a [`RoutingRule`] applies to a request.
///
/// Every `Some` field must match; `None` means "unconstrained". A [`RuleMatcher::default`] therefore
/// matches every request, which makes it a ususable catch-all rule.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RuleMatcher {
    /// Exact member role.
    pub role: Option<TeamRole>,
    /// Exact task kind (case-sensitive).
    pub task_kind: Option<String>,
    /// Exact complexity tier.
    pub complexity: Option<Complexity>,
    /// Minimum complexity tier (inclusive); a request with no complexity never matches.
    pub min_complexity: Option<Complexity>,
    /// Exact `needs_tools` value.
    pub needs_tools: Option<bool>,
}

impl RuleMatcher {
    /// Returns `true` when `hint` satisfies every constraint in this matcher.
    ///
    /// `min_complexity` is a `>=` test against [`RequestHint::complexity`]; a hint that declares no
    /// complexity cannot prove it clears the threshold and so does not match.
    #[must_use]
    pub fn matches(&self, hint: &RequestHint) -> bool {
        self.role.is_none_or(|role| hint.role == Some(role))
            && self
                .task_kind
                .as_deref()
                .is_none_or(|kind| hint.task_kind.as_deref() == Some(kind))
            && self
                .complexity
                .is_none_or(|complexity| hint.complexity == Some(complexity))
            && self
                .min_complexity
                .is_none_or(|min| hint.complexity.is_some_and(|complexity| complexity >= min))
            && self
                .needs_tools
                .is_none_or(|needs_tools| hint.needs_tools == needs_tools)
    }

    /// Constrains the matcher to one member role.
    #[must_use]
    pub fn with_role(mut self, role: TeamRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Constrains the matcher to one task kind.
    #[must_use]
    pub fn with_task_kind(mut self, task_kind: impl Into<String>) -> Self {
        self.task_kind = Some(task_kind.into());
        self
    }

    /// Constrains the matcher to one complexity tier.
    #[must_use]
    pub fn with_complexity(mut self, complexity: Complexity) -> Self {
        self.complexity = Some(complexity);
        self
    }

    /// Sets the minimum complexity tier this matcher accepts.
    #[must_use]
    pub fn with_min_complexity(mut self, min_complexity: Complexity) -> Self {
        self.min_complexity = Some(min_complexity);
        self
    }

    /// Constrains the matcher to requests that do (or do not) carry tools.
    #[must_use]
    pub fn with_needs_tools(mut self, needs_tools: bool) -> Self {
        self.needs_tools = Some(needs_tools);
        self
    }
}

/// One routing rule: when `matcher` matches, send the request to `target`.
///
/// Rules are ordered by `(priority asc, name asc)` — a lower priority number wins, and the name
/// breaks ties so the order is total and reproducible regardless of declaration order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RoutingRule {
    /// Stable rule name, also reported in [`RouterDecision::rule`].
    pub name: String,
    /// Ordering key: lower runs first.
    pub priority: u32,
    /// Constraint conjunction this rule applies.
    pub matcher: RuleMatcher,
    /// Model name this rule selects; must be pooled for the rule to fire.
    pub target: String,
}

impl RoutingRule {
    /// Builds a rule with priority `0`.
    #[must_use]
    pub fn new(name: impl Into<String>, matcher: RuleMatcher, target: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            priority: 0,
            matcher,
            target: target.into(),
        }
    }

    /// Sets this rule's ordering key (lower runs first).
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Returns `true` when this rule's matcher accepts `hint`.
    ///
    /// The target's presence in the pool is checked by the router, not here.
    #[must_use]
    pub fn matches(&self, hint: &RequestHint) -> bool {
        self.matcher.matches(hint)
    }
}

/// Why a [`RouterDecision`] picked the entry it did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteReason {
    /// The request named this model and the pool serves it.
    RequestedName,
    /// A [`RoutingRule`] selected this model.
    Rule,
    /// The round-robin cursor selected this position.
    RoundRobin,
    /// This entry had the fewest in-flight requests.
    LeastBusy,
    /// Nothing matched, so the first entry was used as the fallback.
    #[default]
    FirstEntry,
}

impl RouteReason {
    /// Returns the wire string of this reason (`"requested_name"`, `"round_robin"`, …).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RequestedName => "requested_name",
            Self::Rule => "rule",
            Self::RoundRobin => "round_robin",
            Self::LeastBusy => "least_busy",
            Self::FirstEntry => "first_entry",
        }
    }
}

/// The outcome of routing one request: the chosen endpoint plus why it was chosen.
///
/// This is a value snapshot, not a resource: acquiring a [`RouteGuard`] reserves capacity and
/// dropping it releases that capacity, while a decision on its own changes nothing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RouterDecision {
    /// The chosen endpoint, copied from the pool.
    pub entry: ModelPoolEntry,
    /// Position of [`RouterDecision::entry`] in the pool.
    pub index: usize,
    /// Why this entry was chosen.
    pub reason: RouteReason,
    /// Name of the [`RoutingRule`] that fired, when [`RouteReason::Rule`] applies.
    pub rule: Option<String>,
    /// Other pooled model names to try, in order, if this endpoint fails.
    pub fallback_chain: Vec<String>,
}

/// Process-local router state: rotation cursor plus in-flight counters.
#[derive(Debug, Default)]
struct RouterState {
    /// Round-robin position the next acquisition hands out.
    cursor: usize,
    /// In-flight (live [`RouteGuard`]) requests per pool position.
    in_flight: Vec<usize>,
}

/// Chooses a pooled endpoint for every request.
///
/// The router owns a [`ModelPool`] (validated data), a [`RouterStrategy`] and an ordered rule list;
/// its mutable state — a round-robin cursor and one in-flight counter per entry — lives behind a
/// `Mutex` so [`ModelRouter::acquire`] is atomic across tasks. Decisions are made in microseconds and
/// the lock is never held across an `.await`.
pub struct ModelRouter {
    /// Endpoints this router chooses among.
    pool: ModelPool,
    /// How an entry is chosen.
    strategy: RouterStrategy,
    /// Rules sorted by `(priority asc, name asc)`; empty unless the strategy is rule-based.
    rules: Vec<RoutingRule>,
    /// Cursor and in-flight counters.
    state: Mutex<RouterState>,
}

impl ModelRouter {
    /// Builds a router over an already-validated pool.
    ///
    /// The in-flight counter vector is sized to the pool, so [`ModelRouter::in_flight`] always
    /// reports one count per pool position.
    #[must_use]
    pub fn new(pool: ModelPool, strategy: RouterStrategy) -> Self {
        let in_flight = vec![0; pool.len()];
        Self {
            pool,
            strategy,
            rules: Vec::new(),
            state: Mutex::new(RouterState {
                cursor: 0,
                in_flight,
            }),
        }
    }

    /// Builds a router from raw entries, mapping the build-time strategy onto a router strategy.
    ///
    /// An empty entry slice yields a router over an empty pool rather than an error: a team
    /// without a pool is legal, and only a request then fails (see [`ModelRouter::decide`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the entries fail [`ModelPool::new`] validation.
    pub fn for_pool(entries: &[ModelPoolEntry], strategy: ModelPoolStrategy) -> Result<Self> {
        Ok(Self::new(
            ModelPool::from_entries(entries)?,
            RouterStrategy::from(strategy),
        ))
    }

    /// Attaches routing rules, sorted by `(priority asc, name asc)`.
    ///
    /// Rules only fire under [`RouterStrategy::RuleBased`]; attaching them to another strategy is
    /// allowed (the router reports them through [`ModelRouter::rules`]) but they are not consulted.
    #[must_use]
    pub fn with_rules(mut self, mut rules: Vec<RoutingRule>) -> Self {
        rules.sort_by(|left, right| {
            left.priority
                .cmp(&right.priority)
                .then_with(|| left.name.cmp(&right.name))
        });
        self.rules = rules;
        self
    }

    /// Returns the pool this router chooses among.
    #[must_use]
    pub fn pool(&self) -> &ModelPool {
        &self.pool
    }

    /// Returns the strategy this router applies.
    #[must_use]
    pub const fn strategy(&self) -> RouterStrategy {
        self.strategy
    }

    /// Returns the routing rules in the order they are evaluated.
    #[must_use]
    pub fn rules(&self) -> &[RoutingRule] {
        &self.rules
    }

    /// Previews the routing decision for `hint` without reserving anything.
    ///
    /// The round-robin cursor does not advance and no in-flight counter changes, so repeated calls
    /// with the same hint and unchanged state return the same decision.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the pool is empty — no endpoint is configured, so the caller
    /// must fall back to a member-level model.
    pub fn decide(&self, hint: &RequestHint) -> Result<RouterDecision> {
        let mut state = self.state.lock();
        self.choose(&mut state, hint, false)
    }

    /// Atomically chooses an endpoint for `hint` and reserves one in-flight slot on it.
    ///
    /// The returned [`RouteGuard`] releases the slot when dropped, including on an early return or a
    /// panic, so a failed request never leaks capacity. Round-robin advances its cursor here (not in
    /// [`ModelRouter::decide`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the pool is empty; nothing is reserved in that case.
    pub fn acquire(&self, hint: &RequestHint) -> Result<RouteGuard<'_>> {
        let decision = {
            let mut state = self.state.lock();
            self.choose(&mut state, hint, true)?
        };
        Ok(RouteGuard {
            router: self,
            decision,
        })
    }

    /// Returns the live in-flight request count per pool position.
    #[must_use]
    pub fn in_flight(&self) -> Vec<usize> {
        self.state.lock().in_flight.clone()
    }

    /// Returns the total number of live [`RouteGuard`]s across the pool.
    #[must_use]
    pub fn total_in_flight(&self) -> usize {
        self.state.lock().in_flight.iter().sum()
    }

    /// Chooses the entry for `hint`, optionally reserving it.
    ///
    /// `reserve` distinguishes [`ModelRouter::decide`] (preview) from [`ModelRouter::acquire`]
    /// (commit): only a committing call advances the round-robin cursor and increments the in-flight
    /// counter. Reservation happens once the choice is known, so an error leaves state untouched.
    fn choose(
        &self,
        state: &mut RouterState,
        hint: &RequestHint,
        reserve: bool,
    ) -> Result<RouterDecision> {
        if self.pool.is_empty() {
            return Err(Error::config(NO_MODEL_MESSAGE));
        }
        let len = self.pool.len();
        let (index, reason, rule) = match self.strategy {
            RouterStrategy::ByName => self.choose_by_name(hint),
            RouterStrategy::RoundRobin => {
                let index = state.cursor.checked_rem(len).unwrap_or(0);
                (index, RouteReason::RoundRobin, None)
            }
            RouterStrategy::LeastBusy => Self::choose_least_busy(state),
            RouterStrategy::RuleBased => self
                .choose_rule_based(hint)
                .unwrap_or_else(|| self.choose_by_name(hint)),
        };
        let entry = self.pool.get(index).cloned().ok_or_else(|| {
            // Unreachable: every strategy derives `index` from the pool itself.
            Error::config(format!("model_pool index {index} is out of range"))
        })?;
        let fallback_chain = self.fallback_chain(state, hint, index, rule.as_deref());
        if reserve {
            if self.strategy == RouterStrategy::RoundRobin {
                state.cursor = index.wrapping_add(1).checked_rem(len).unwrap_or(0);
            }
            if let Some(slot) = state.in_flight.get_mut(index) {
                *slot = slot.saturating_add(1);
            }
        }
        Ok(RouterDecision {
            entry,
            index,
            reason,
            rule,
            fallback_chain,
        })
    }

    /// Serves the requested model name, else the first entry.
    fn choose_by_name(&self, hint: &RequestHint) -> (usize, RouteReason, Option<String>) {
        match hint
            .model_name
            .as_deref()
            .and_then(|name| self.pool.find(name))
        {
            Some((index, _)) => (index, RouteReason::RequestedName, None),
            None => (0, RouteReason::FirstEntry, None),
        }
    }

    /// Picks the entry with the fewest in-flight requests, tie-broken by pool position.
    fn choose_least_busy(state: &RouterState) -> (usize, RouteReason, Option<String>) {
        let mut best_index = 0;
        let mut best_load = state.in_flight.first().copied().unwrap_or(0);
        for (index, load) in state.in_flight.iter().enumerate().skip(1) {
            if *load < best_load {
                best_index = index;
                best_load = *load;
            }
        }
        (best_index, RouteReason::LeastBusy, None)
    }

    /// Applies the first matching rule whose target is pooled.
    ///
    /// A rule whose target is missing from the pool is skipped rather than treated as a fatal
    /// configuration error: the pool may have been refreshed underneath an unchanged rule set, and
    /// the next matching rule (or the by-name fallback) still yields a usable endpoint.
    fn choose_rule_based(
        &self,
        hint: &RequestHint,
    ) -> Option<(usize, RouteReason, Option<String>)> {
        for rule in &self.rules {
            if !rule.matches(hint) {
                continue;
            }
            if let Some((index, _)) = self.pool.find(&rule.target) {
                return Some((index, RouteReason::Rule, Some(rule.name.clone())));
            }
        }
        None
    }

    /// Builds the ordered fallback chain for a decision on `chosen_index`.
    ///
    /// The chain holds *model names* the caller should try next if the chosen endpoint fails, in
    /// this order: the remaining matching rule targets (rule-based only, after the fired rule), then
    /// the strategy's own ordering of the other entries, then the first entry. Names are
    /// de-duplicated, only pooled names appear, and the chosen name never appears.
    fn fallback_chain(
        &self,
        state: &RouterState,
        hint: &RequestHint,
        chosen_index: usize,
        chosen_rule: Option<&str>,
    ) -> Vec<String> {
        let chosen_name = self
            .pool
            .get(chosen_index)
            .map_or("", |entry| entry.model_name.as_str())
            .to_owned();
        let mut chain: Vec<String> = Vec::new();

        if self.strategy == RouterStrategy::RuleBased {
            let mut after_chosen = chosen_rule.is_none();
            for rule in &self.rules {
                if Some(rule.name.as_str()) == chosen_rule {
                    after_chosen = true;
                    continue;
                }
                if !after_chosen || !rule.matches(hint) || !self.pool.contains(&rule.target) {
                    continue;
                }
                if rule.target != chosen_name && !chain.contains(&rule.target) {
                    chain.push(rule.target.clone());
                }
            }
        }

        for index in self.strategy_order(state, chosen_index) {
            let Some(entry) = self.pool.get(index) else {
                continue;
            };
            if entry.model_name != chosen_name && !chain.contains(&entry.model_name) {
                chain.push(entry.model_name.clone());
            }
        }

        if let Some(first) = self.pool.get(0) {
            if first.model_name != chosen_name && !chain.contains(&first.model_name) {
                chain.push(first.model_name.clone());
            }
        }
        chain
    }

    /// Returns the pool positions of the non-chosen entries in this strategy's own order.
    ///
    /// Round-robin continues its rotation after the chosen position, least-busy orders by
    /// `(in-flight asc, index asc)`, and the by-name/rule strategies keep pool order.
    fn strategy_order(&self, state: &RouterState, chosen_index: usize) -> Vec<usize> {
        let len = self.pool.len();
        if len == 0 {
            return Vec::new();
        }
        match self.strategy {
            RouterStrategy::RoundRobin => {
                (1..=len).map(|step| (chosen_index + step) % len).collect()
            }
            RouterStrategy::LeastBusy => {
                let mut indices: Vec<usize> = (0..len).collect();
                indices.sort_by_key(|&index| {
                    (state.in_flight.get(index).copied().unwrap_or(0), index)
                });
                indices.retain(|&index| index != chosen_index);
                indices
            }
            RouterStrategy::ByName | RouterStrategy::RuleBased => {
                (0..len).filter(|&index| index != chosen_index).collect()
            }
        }
    }
}

impl fmt::Debug for ModelRouter {
    /// Prints the strategy, pool size, rule count and live counters; no credentials.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.lock();
        f.debug_struct("ModelRouter")
            .field("strategy", &self.strategy.as_str())
            .field("pool", &self.pool.len())
            .field("rules", &self.rules.len())
            .field("cursor", &state.cursor)
            .field("in_flight", &state.in_flight)
            .finish()
    }
}

/// Reservation of one in-flight slot on a routed endpoint.
///
/// Holding a guard tells a [`RouterStrategy::LeastBusy`] router that the endpoint is busy; dropping
/// it releases the slot. `Drop` never panics and is the only way capacity returns to the pool, so an
/// error path or a panic anywhere in the request cannot leak a slot.
pub struct RouteGuard<'a> {
    /// Router that owns the reserved slot.
    router: &'a ModelRouter,
    /// The decision this guard reserves.
    decision: RouterDecision,
}

impl RouteGuard<'_> {
    /// Returns the full routing decision behind this guard.
    #[must_use]
    pub fn decision(&self) -> &RouterDecision {
        &self.decision
    }

    /// Returns the reserved endpoint.
    #[must_use]
    pub fn entry(&self) -> &ModelPoolEntry {
        &self.decision.entry
    }
}

impl fmt::Debug for RouteGuard<'_> {
    /// Prints the reserved decision.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RouteGuard")
            .field("decision", &self.decision)
            .finish()
    }
}

impl Drop for RouteGuard<'_> {
    /// Releases the reserved in-flight slot; saturates instead of panicking if state is inconsistent.
    fn drop(&mut self) {
        let mut state = self.router.state.lock();
        if let Some(slot) = state.in_flight.get_mut(self.decision.index) {
            *slot = slot.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests;
