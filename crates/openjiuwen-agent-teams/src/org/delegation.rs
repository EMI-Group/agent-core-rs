//! Delegation rules: may this node hand work to that node?
//!
//! The router answers one question — "is this delegation allowed?" — against an [`OrgChart`] and a
//! [`DelegationRules`] bundle, and answers it *without* touching the task pool: the caller supplies
//! the already-measured hop count and the target team's open-task count, so the decision is pure,
//! deterministic and cheap to unit test. The organisation task pool asks the router before it
//! creates a delegated task; nothing else is allowed to re-implement these rules.
//!
//! ## Decision order
//!
//! Checks run top-down and the first refusal wins, so a request is reported against the most
//! fundamental problem first (an unknown delegator beats a budget complaint):
//!
//! | # | Check | Refusal |
//! | --- | --- | --- |
//! | 1 | the delegator node exists | [`DelegationRefusal::UnknownDelegator`] |
//! | 2 | `to_team` is a declared team | [`DelegationRefusal::UnknownTargetTeam`] |
//! | 3 | `to_member` (or the target team's leader when unset) is a member of `to_team` | [`DelegationRefusal::UnknownTargetMember`] |
//! | 4 | the target is not the delegator | [`DelegationRefusal::SelfDelegation`] |
//! | 5 | `depth` ≤ [`DelegationRules::max_delegation_depth`] (unlimited when the maximum is `0`) | [`DelegationRefusal::DepthExceeded`] |
//! | 6 | the delegator's role may delegate to the target's role | [`DelegationRefusal::RoleNotAllowed`] |
//! | 7 | the delegation follows the reporting line (skipped when [`DelegationRules::require_reporting_line`] is `false`) | [`DelegationRefusal::ReportingLineRequired`] |
//! | 8 | the target team is below its task budget | [`DelegationRefusal::BudgetExhausted`] |
//!
//! A request that passes every check yields [`DelegationDecision::Allowed`] carrying the effective
//! depth (a request with `depth == 0` means "not measured yet" and is treated as `1`) and whether
//! the hop crosses a team boundary.
//!
//! ## Reporting line
//!
//! The reporting-line check is the interesting one. It is satisfied when any of the following
//! holds:
//!
//! * the delegation stays inside one team (`to_team == from_team`),
//! * `to_team` sits below `from_team` in the chart (`from_team` is an ancestor of `to_team`),
//! * the delegator's role is [`OrgRole::Owner`] (the owner may delegate anywhere), or
//! * the delegator node is an ancestor of the target node (a leader delegating down its own line,
//!   including across teams).
//!
//! Everything here is data plus a pure decision: no clock, no I/O, no global state.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::qualify_member;
use super::roles::{OrgChart, OrgRole};

/// A `from -> to` role transition in [`DelegationRules::allowed_pairs`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RolePair {
    /// Role of the delegating node.
    pub from: OrgRole,
    /// Role of the target node.
    pub to: OrgRole,
}

/// The policy half of the delegation decision.
///
/// Defaults follow the crate's conservative organisation defaults: three hops, a reporting line is
/// required, every role pair is allowed and task budgets are unlimited (a host that wants budgets
/// sets them per team or as a default). The type is plain data, so it round-trips through serde and
/// can be tuned per organisation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DelegationRules {
    /// Maximum delegation hops, measured from the original request (`1` = the first hop).
    /// `0` disables the check entirely.
    pub max_delegation_depth: usize,
    /// When `true`, a delegation must follow the reporting line (see the module docs).
    pub require_reporting_line: bool,
    /// Allowed role transitions; an empty list allows every pair.
    pub allowed_pairs: Vec<RolePair>,
    /// Per-team open-task budgets, keyed by team name.
    pub team_task_budget: BTreeMap<String, usize>,
    /// Budget for teams without an entry in [`Self::team_task_budget`]; `None` is unlimited.
    pub default_task_budget: Option<usize>,
}

impl Default for DelegationRules {
    /// Conservative defaults: depth `3`, reporting line required, no role or budget restriction.
    fn default() -> Self {
        Self {
            max_delegation_depth: 3,
            require_reporting_line: true,
            allowed_pairs: Vec::new(),
            team_task_budget: BTreeMap::new(),
            default_task_budget: None,
        }
    }
}

impl DelegationRules {
    /// Returns the default rules.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets [`Self::max_delegation_depth`] (`0` means unlimited).
    #[must_use]
    pub fn with_max_depth(mut self, max_delegation_depth: usize) -> Self {
        self.max_delegation_depth = max_delegation_depth;
        self
    }

    /// Sets [`Self::require_reporting_line`].
    #[must_use]
    pub fn with_require_reporting_line(mut self, require: bool) -> Self {
        self.require_reporting_line = require;
        self
    }

    /// Adds one allowed role transition to [`Self::allowed_pairs`].
    ///
    /// The first call turns the matrix on: once the list is non-empty every pair that was not
    /// added is refused. Adding the same pair twice is a no-op.
    #[must_use]
    pub fn allowing(mut self, from: OrgRole, to: OrgRole) -> Self {
        let pair = RolePair { from, to };
        if !self.allowed_pairs.contains(&pair) {
            self.allowed_pairs.push(pair);
        }
        self
    }

    /// Sets the open-task budget of `team` in [`Self::team_task_budget`].
    #[must_use]
    pub fn with_team_budget(mut self, team: impl Into<String>, budget: usize) -> Self {
        self.team_task_budget.insert(team.into(), budget);
        self
    }

    /// Sets [`Self::default_task_budget`], used for teams without a specific budget.
    #[must_use]
    pub fn with_default_budget(mut self, budget: usize) -> Self {
        self.default_task_budget = Some(budget);
        self
    }

    /// Returns `true` when a `from -> to` transition is allowed.
    ///
    /// An empty [`Self::allowed_pairs`] allows every pair.
    #[must_use]
    pub fn allows_pair(&self, from: OrgRole, to: OrgRole) -> bool {
        self.allowed_pairs.is_empty()
            || self
                .allowed_pairs
                .iter()
                .any(|pair| pair.from == from && pair.to == to)
    }

    /// Returns the open-task budget that applies to `team`.
    ///
    /// A per-team entry wins over [`Self::default_task_budget`]; `None` means unlimited.
    #[must_use]
    pub fn budget_for(&self, team: &str) -> Option<usize> {
        match self.team_task_budget.get(team) {
            Some(budget) => Some(*budget),
            None => self.default_task_budget,
        }
    }
}

/// One delegation attempt, as the caller observed it.
///
/// The request carries facts, not policies: who delegates, to which team (and member), how many
/// hops deep this delegation already is, and how busy the target team is. `depth == 0` means "not
/// measured yet" and is evaluated as `1`. The record is serde-defaultable, so a partially filled
/// request is a valid request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DelegationRequest {
    /// Team of the delegating member.
    pub from_team: String,
    /// Member id of the delegating member inside [`Self::from_team`].
    pub from_member: String,
    /// Team the work is delegated to.
    pub to_team: String,
    /// Target member inside [`Self::to_team`]; `None` means that team's leader.
    pub to_member: Option<String>,
    /// Hop count of this delegation (`1` = first hop, `0` = not measured yet).
    pub depth: usize,
    /// Open tasks the target team already has, when the caller knows them.
    pub open_tasks_in_target: Option<usize>,
}

impl DelegationRequest {
    /// Returns a first-hop request with no explicit target member and no task count.
    #[must_use]
    pub fn new(
        from_team: impl Into<String>,
        from_member: impl Into<String>,
        to_team: impl Into<String>,
    ) -> Self {
        Self {
            from_team: from_team.into(),
            from_member: from_member.into(),
            to_team: to_team.into(),
            ..Self::default()
        }
    }

    /// Sets [`Self::to_member`].
    #[must_use]
    pub fn with_target_member(mut self, member: impl Into<String>) -> Self {
        self.to_member = Some(member.into());
        self
    }

    /// Sets [`Self::depth`].
    #[must_use]
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Sets [`Self::open_tasks_in_target`].
    #[must_use]
    pub fn with_open_tasks(mut self, open: usize) -> Self {
        self.open_tasks_in_target = Some(open);
        self
    }
}

/// Why a delegation was refused.
///
/// The variants are wire-visible (`snake_case`, structurally tagged) and carry the numbers the
/// caller needs to explain the refusal; entity ids stay out of the refusal so the same value can be
/// logged from any layer. [`Self::code`] is the stable machine token, [`Self::message`] the human
/// sentence used by [`DelegationRouter::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelegationRefusal {
    /// The delegating member is not in the chart.
    UnknownDelegator,
    /// The target team is not in the chart.
    UnknownTargetTeam,
    /// The named target member does not belong to the target team.
    UnknownTargetMember,
    /// The target node is the delegator itself.
    SelfDelegation,
    /// A reporting line is required and this hop does not follow one.
    ReportingLineRequired,
    /// The hop count exceeds the configured maximum.
    DepthExceeded {
        /// Hop count the request asked for.
        requested: usize,
        /// Configured maximum.
        max: usize,
    },
    /// The role matrix does not allow this transition.
    RoleNotAllowed {
        /// Role of the delegating node.
        from: OrgRole,
        /// Role of the target node.
        to: OrgRole,
    },
    /// The target team is at or above its open-task budget.
    BudgetExhausted {
        /// Open tasks the target team already has.
        open: usize,
        /// Budget that applies to the target team.
        budget: usize,
    },
}

impl DelegationRefusal {
    /// Returns the stable machine token of this refusal.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnknownDelegator => "unknown_delegator",
            Self::UnknownTargetTeam => "unknown_target_team",
            Self::UnknownTargetMember => "unknown_target_member",
            Self::SelfDelegation => "self_delegation",
            Self::ReportingLineRequired => "reporting_line_required",
            Self::DepthExceeded { .. } => "depth_exceeded",
            Self::RoleNotAllowed { .. } => "role_not_allowed",
            Self::BudgetExhausted { .. } => "budget_exhausted",
        }
    }

    /// Returns the human sentence for this refusal.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::UnknownDelegator => "delegator is not part of the organisation".to_owned(),
            Self::UnknownTargetTeam => "target team is not part of the organisation".to_owned(),
            Self::UnknownTargetMember => "target member is not part of the target team".to_owned(),
            Self::SelfDelegation => "a node cannot delegate to itself".to_owned(),
            Self::ReportingLineRequired => "delegation must follow the reporting line".to_owned(),
            Self::DepthExceeded { requested, max } => {
                format!("delegation depth {requested} exceeds the maximum of {max}")
            }
            Self::RoleNotAllowed { from, to } => format!(
                "role transition '{} -> {}' is not allowed",
                from.as_str(),
                to.as_str()
            ),
            Self::BudgetExhausted { open, budget } => {
                format!("target team has {open} open tasks, at or above its budget of {budget}")
            }
        }
    }
}

impl fmt::Display for DelegationRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

/// The router's verdict on one [`DelegationRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelegationDecision {
    /// The delegation may proceed.
    Allowed {
        /// Effective hop count (`1` when the request carried `0`).
        depth: usize,
        /// `true` when the delegation crosses a team boundary.
        cross_team: bool,
    },
    /// The delegation is refused; the payload explains why.
    Refused {
        /// First failed check, in decision order.
        refusal: DelegationRefusal,
    },
}

impl DelegationDecision {
    /// Returns `true` for [`Self::Allowed`].
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed { .. })
    }

    /// Returns the refusal of a refused decision.
    #[must_use]
    pub fn refusal(&self) -> Option<&DelegationRefusal> {
        match self {
            Self::Allowed { .. } => None,
            Self::Refused { refusal } => Some(refusal),
        }
    }

    /// Returns the human sentence of a refused decision.
    #[must_use]
    pub fn reason(&self) -> Option<String> {
        self.refusal().map(DelegationRefusal::message)
    }
}

/// Decides whether a delegation is allowed against one chart and one rule set.
///
/// The router borrows the chart behind an [`Arc`], so every team runtime can hold its own router
/// over the same validated chart; the rules are values, so a host can specialise them per
/// organisation. The type is `Clone + Send + Sync` and every method is pure.
#[derive(Debug, Clone)]
pub struct DelegationRouter {
    chart: Arc<OrgChart>,
    rules: DelegationRules,
}

impl DelegationRouter {
    /// Returns a router over `chart` applying `rules`.
    #[must_use]
    pub fn new(chart: Arc<OrgChart>, rules: DelegationRules) -> Self {
        Self { chart, rules }
    }

    /// Returns the chart this router decides against.
    #[must_use]
    pub fn chart(&self) -> &Arc<OrgChart> {
        &self.chart
    }

    /// Returns the rules this router applies.
    #[must_use]
    pub fn rules(&self) -> &DelegationRules {
        &self.rules
    }

    /// Returns the verdict on `request` (see the module docs for the decision order).
    #[must_use]
    pub fn decide(&self, request: &DelegationRequest) -> DelegationDecision {
        match self.evaluate(request) {
            Ok(decision) => decision,
            Err(refusal) => DelegationDecision::Refused { refusal },
        }
    }

    /// Returns `true` when [`Self::decide`] allows `request`.
    #[must_use]
    pub fn can_delegate(&self, request: &DelegationRequest) -> bool {
        self.decide(request).is_allowed()
    }

    /// Validates `request`, returning the refusal as a validation error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] carrying the [`DelegationRefusal::message`] of the first failed
    /// check.
    pub fn check(&self, request: &DelegationRequest) -> Result<()> {
        match self.decide(request) {
            DelegationDecision::Allowed { .. } => Ok(()),
            DelegationDecision::Refused { refusal } => Err(Error::validation(refusal.message())),
        }
    }

    /// Runs the checks in decision order; the first refusal wins.
    fn evaluate(
        &self,
        request: &DelegationRequest,
    ) -> std::result::Result<DelegationDecision, DelegationRefusal> {
        let chart = self.chart.as_ref();
        let from_node = qualify_member(&request.from_team, &request.from_member);
        let from_role = chart
            .role_of(&from_node)
            .ok_or(DelegationRefusal::UnknownDelegator)?;
        let target_team = chart
            .team(&request.to_team)
            .ok_or(DelegationRefusal::UnknownTargetTeam)?;
        let target_member = match request.to_member.as_deref() {
            Some(member) => {
                chart
                    .member(&request.to_team, member)
                    .ok_or(DelegationRefusal::UnknownTargetMember)?;
                member
            }
            None => target_team.leader_or_default(),
        };
        let to_node = qualify_member(&request.to_team, target_member);
        let to_role = chart
            .role_of(&to_node)
            .ok_or(DelegationRefusal::UnknownTargetMember)?;

        if from_node == to_node {
            return Err(DelegationRefusal::SelfDelegation);
        }

        let depth = request.depth.max(1);
        let max = self.rules.max_delegation_depth;
        if max > 0 && depth > max {
            return Err(DelegationRefusal::DepthExceeded {
                requested: depth,
                max,
            });
        }

        if !self.rules.allows_pair(from_role, to_role) {
            return Err(DelegationRefusal::RoleNotAllowed {
                from: from_role,
                to: to_role,
            });
        }

        if self.rules.require_reporting_line
            && !self.follows_reporting_line(request, &from_node, &to_node, from_role)
        {
            return Err(DelegationRefusal::ReportingLineRequired);
        }

        if let Some(open) = request.open_tasks_in_target {
            if let Some(budget) = self.rules.budget_for(&request.to_team) {
                if open >= budget {
                    return Err(DelegationRefusal::BudgetExhausted { open, budget });
                }
            }
        }

        Ok(DelegationDecision::Allowed {
            depth,
            cross_team: request.to_team != request.from_team,
        })
    }

    /// Returns `true` when the hop follows the reporting line (see the module docs).
    fn follows_reporting_line(
        &self,
        request: &DelegationRequest,
        from_node: &str,
        to_node: &str,
        from_role: OrgRole,
    ) -> bool {
        let chart = self.chart.as_ref();
        request.to_team == request.from_team
            || chart.is_ancestor(&request.from_team, &request.to_team)
            || from_role == OrgRole::Owner
            || chart.is_ancestor(from_node, to_node)
    }
}

#[cfg(test)]
#[path = "delegation_tests.rs"]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod delegation_tests;
