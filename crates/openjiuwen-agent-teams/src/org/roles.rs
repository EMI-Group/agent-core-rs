//! Organisation chart: roles, teams, members and the reporting hierarchy.
//!
//! This module owns the *static* shape of an organisation — the nodes the in-process organisation
//! layer coordinates — resolved once into an [`OrgChart`], so every later question ("who leads
//! this team?", "is `beta` below `alpha`?", "may this member delegate there?") is a pure lookup.
//! Nothing here spawns, dispatches or persists: the task pool and mailbox of the sibling modules
//! write through [`crate::storage::TeamStore`] and never re-derive the chart.
//!
//! The Python reference (`openjiuwen/agent_teams/organization/schema.py`) models an organisation as
//! registered leader handles plus a task pool; it has no reporting tree. The chart below is the
//! Rust shape the organisation layer needs and keeps Python's vocabulary (`organization_id`,
//! `display_name`, `desc`, `reports_to`, `summary_team`) wherever a counterpart exists.
//!
//! ## Node addressing
//!
//! A **team node id** is its `team_name`; a **member node id** is the team-qualified member
//! reference [`qualify_member`] renders, `"<team>::<member>"`.
//! `reports_to` accepts either form, so a team can report to a team and a member to a member (or a
//! member to a team, which is how a detail member hangs off the team it serves). Team names may not
//! contain `"::"`, which keeps the two namespaces disjoint.
//!
//! ## What `OrganizationSpec::validate` enforces
//!
//! 1. `organization_id` passes `validate_org_id` (1–96 characters of `[A-Za-z0-9_.-]`).
//! 2. Team names are non-empty, contain no `"::"`, and are unique.
//! 3. Member ids are non-empty inside their team, each member's team is declared, and
//!    `(team, member)` pairs are unique.
//! 4. Every `reports_to` names an existing node and never the node itself.
//! 5. The union reporting graph is acyclic; the message is
//!    `reporting cycle: a -> b -> a` with the path rotated to start at its smallest node id, the
//!    same convention as the task board's `Circular dependency: …`.
//! 6. Every team's resolved leader (`leader`, else `"team_leader"`) is a member of that team.
//! 7. `owner`, when set, names an existing member node: a `"<team>::<member>"` reference, or a
//!    bare member id that is unique in the organisation.
//! 8. `summary_team`, when set, names a declared team whose role is [`OrgRole::Summary`].
//!
//! Every failure is `Error::validation` and names the offending id; hierarchy queries on an unknown
//! node fail with `Error::not_found`. The chart is deterministic — teams ordered by name, members by
//! `(team_name, member_id)`, every derived list sorted — and holds no clock, I/O or global state.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{qualify_member, split_qualified, validate_org_id, MEMBER_QUALIFIER};
use crate::constants::DEFAULT_LEADER_MEMBER_NAME;
use crate::taskboard::detect_cycle;

/// The role a node plays in the organisation chart.
///
/// Wire values are the plain Python-style strings `member`, `leader`, `owner` and `summary`; the
/// default is [`OrgRole::Member`]. The declaration order is also the ordering of the enum, but
/// permission questions are answered by [`OrgRole::can_lead`] / [`OrgRole::can_summarize`], never
/// by comparing roles.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum OrgRole {
    /// An ordinary member of a team.
    #[default]
    Member,
    /// Leads the team it belongs to (and, for a team node, an ordinary leading team).
    Leader,
    /// Owns the whole organisation; may lead and delegate anywhere.
    Owner,
    /// Runs the organisation's summary team.
    Summary,
}

impl OrgRole {
    /// Returns the wire string of this role.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Member => "member",
            Self::Leader => "leader",
            Self::Owner => "owner",
            Self::Summary => "summary",
        }
    }

    /// Returns `true` when a node with this role may lead a team.
    ///
    /// [`OrgRole::Leader`], [`OrgRole::Owner`] and [`OrgRole::Summary`] may lead; a plain
    /// [`OrgRole::Member`] may not.
    #[must_use]
    pub const fn can_lead(self) -> bool {
        matches!(self, Self::Leader | Self::Owner | Self::Summary)
    }

    /// Returns `true` when a node with this role may run the organisation's summary team.
    ///
    /// The owner may do anything, and [`OrgRole::Summary`] is the dedicated summary role.
    #[must_use]
    pub const fn can_summarize(self) -> bool {
        matches!(self, Self::Owner | Self::Summary)
    }
}

/// One team node of the chart.
///
/// `team_name` is both the display key and the node id; the team's tasks live under that same name
/// in the team store, so renaming a team is a chart change, not a data migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgTeam {
    /// Team name — the node id of this team.
    pub team_name: String,
    /// Human-readable name used in prompts and the monitor stream.
    pub display_name: String,
    /// One-line description of the team's remit.
    pub desc: String,
    /// Role of the team node; defaults to [`OrgRole::Leader`].
    pub role: OrgRole,
    /// Member id of the team leader inside this team; `None` (or an empty string) means
    /// [`DEFAULT_LEADER_MEMBER_NAME`].
    pub leader: Option<String>,
    /// Node this team reports to: a team name or a qualified member id.
    pub reports_to: Option<String>,
    /// Free-form metadata carried through serde untouched.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for OrgTeam {
    /// An empty team with the [`OrgRole::Leader`] team role and no reporting line.
    fn default() -> Self {
        Self {
            team_name: String::new(),
            display_name: String::new(),
            desc: String::new(),
            role: OrgRole::Leader,
            leader: None,
            reports_to: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl OrgTeam {
    /// Returns the leader member id, falling back to [`DEFAULT_LEADER_MEMBER_NAME`].
    ///
    /// An empty `leader` string is treated like `None`, so a tolerance-minded deserializer cannot
    /// produce a team that delegates work to the empty name.
    #[must_use]
    pub fn leader_or_default(&self) -> &str {
        match self.leader.as_deref() {
            Some(leader) if !leader.is_empty() => leader,
            _ => DEFAULT_LEADER_MEMBER_NAME,
        }
    }
}

/// One member node of the chart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgMember {
    /// Member id, unique inside `team_name`.
    pub member_id: String,
    /// Team the member belongs to; must be a declared team.
    pub team_name: String,
    /// Human-readable name used in prompts and the monitor stream.
    pub display_name: String,
    /// One-line description of the member's remit.
    pub desc: String,
    /// Role of the member; defaults to [`OrgRole::Member`].
    pub role: OrgRole,
    /// Node this member reports to: a qualified member id or a team name.
    pub reports_to: Option<String>,
    /// Free-form metadata carried through serde untouched.
    pub metadata: BTreeMap<String, Value>,
}

impl OrgMember {
    /// Returns this member's node id (`"<team>::<member>"`).
    #[must_use]
    pub fn node_id(&self) -> String {
        qualify_member(&self.team_name, &self.member_id)
    }
}

/// The declarative shape of one organisation, before any hierarchy query is possible.
///
/// Topology lives on the nodes themselves (`OrgTeam::reports_to`, `OrgMember::reports_to`) rather
/// than in a separate edge list, so a chart is one flat document. Call [`Self::validate`] — or hand
/// the spec to [`OrgChart::new`], which validates for you — before trusting it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrganizationSpec {
    /// Organisation id; also the namespace its task-pool rows are stored under.
    pub organization_id: String,
    /// Human-readable organisation name.
    pub display_name: String,
    /// Description of what the organisation does.
    pub desc: String,
    /// Qualified member id of the organisation owner.
    pub owner: Option<String>,
    /// The organisation's teams, in declaration order (queries re-sort).
    pub teams: Vec<OrgTeam>,
    /// The organisation's members, in declaration order (queries re-sort).
    pub members: Vec<OrgMember>,
    /// Team name of the summary team; it must carry [`OrgRole::Summary`].
    pub summary_team: Option<String>,
    /// Free-form metadata carried through serde untouched.
    pub metadata: BTreeMap<String, Value>,
}

impl OrganizationSpec {
    /// Returns an empty spec for `organization_id`.
    ///
    /// The id is not validated here — [`Self::validate`] owns that — so a caller can build a spec
    /// incrementally and validate once at the end.
    #[must_use]
    pub fn new(organization_id: impl Into<String>) -> Self {
        Self {
            organization_id: organization_id.into(),
            ..Self::default()
        }
    }

    /// Validates the organisation id, the node ids, the reporting tree and the cross references.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with a message naming the offending id when
    /// * `organization_id` is not a valid organisation id,
    /// * a team name is empty, contains `"::"` or repeats,
    /// * a member id is empty, repeats inside its team, or its team is not declared,
    /// * a `reports_to` names an unknown node or the node itself,
    /// * the reporting graph contains a cycle,
    /// * a team's resolved leader is not a member of that team,
    /// * `owner` is not an existing member node, or
    /// * `summary_team` is not a declared team with role [`OrgRole::Summary`].
    pub fn validate(&self) -> Result<()> {
        validate_org_id(&self.organization_id)?;

        let mut team_names: BTreeSet<&str> = BTreeSet::new();
        for team in &self.teams {
            if team.team_name.is_empty() {
                return Err(Error::validation("team name must not be empty"));
            }
            if team.team_name.contains(MEMBER_QUALIFIER) {
                return Err(Error::validation(format!(
                    "team name '{}' must not contain '{MEMBER_QUALIFIER}'",
                    team.team_name
                )));
            }
            if !team_names.insert(team.team_name.as_str()) {
                return Err(Error::validation(format!(
                    "duplicate team name '{}'",
                    team.team_name
                )));
            }
        }

        let mut member_nodes: BTreeSet<String> = BTreeSet::new();
        for member in &self.members {
            if member.member_id.is_empty() {
                return Err(Error::validation(format!(
                    "team '{}' has a member with an empty id",
                    member.team_name
                )));
            }
            let node_id = member.node_id();
            if !team_names.contains(member.team_name.as_str()) {
                return Err(Error::validation(format!(
                    "member '{node_id}' belongs to unknown team '{}'",
                    member.team_name
                )));
            }
            if !member_nodes.insert(node_id.clone()) {
                return Err(Error::validation(format!("duplicate member '{node_id}'")));
            }
        }

        let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for team in &self.teams {
            edges.entry(team.team_name.clone()).or_default();
        }
        for node_id in &member_nodes {
            edges.entry(node_id.clone()).or_default();
        }
        let node_ids: BTreeSet<String> = edges.keys().cloned().collect();
        for team in &self.teams {
            if let Some(parent) = team.reports_to.as_deref() {
                add_report_edge(&team.team_name, parent, &node_ids, &mut edges)?;
            }
        }
        for member in &self.members {
            if let Some(parent) = member.reports_to.as_deref() {
                add_report_edge(&member.node_id(), parent, &node_ids, &mut edges)?;
            }
        }
        if let Some(cycle) = detect_cycle(&edges) {
            return Err(Error::validation(format!(
                "reporting cycle: {}",
                cycle.join(" -> ")
            )));
        }

        for team in &self.teams {
            let leader = team.leader_or_default();
            if !member_nodes.contains(&qualify_member(&team.team_name, leader)) {
                return Err(Error::validation(format!(
                    "team '{}' leader '{leader}' is not a member of the team",
                    team.team_name
                )));
            }
        }

        if let Some(owner) = self.owner.as_deref() {
            if !self.is_member_node(owner, &member_nodes) {
                return Err(Error::validation(format!(
                    "owner '{owner}' is not an organisation member"
                )));
            }
        }

        if let Some(summary) = self.summary_team.as_deref() {
            match self.team(summary) {
                None => {
                    return Err(Error::validation(format!(
                        "summary_team '{summary}' is not a team of the organisation"
                    )))
                }
                Some(team) if team.role != OrgRole::Summary => {
                    return Err(Error::validation(format!(
                        "summary_team '{summary}' must have role '{}'",
                        OrgRole::Summary.as_str()
                    )))
                }
                Some(_) => {}
            }
        }

        Ok(())
    }

    /// Returns the team named `team_name`, if declared.
    #[must_use]
    pub fn team(&self, team_name: &str) -> Option<&OrgTeam> {
        self.teams.iter().find(|team| team.team_name == team_name)
    }

    /// Returns the member `member_id` of team `team_name`, if declared.
    #[must_use]
    pub fn member(&self, team_name: &str, member_id: &str) -> Option<&OrgMember> {
        self.members
            .iter()
            .find(|member| member.team_name == team_name && member.member_id == member_id)
    }

    /// Resolves the `owner` reference against the qualified member-node set.
    ///
    /// A `"<team>::<member>"` reference is accepted as-is; a bare member id is accepted only when
    /// exactly one declared member carries it, so the owner is never ambiguous.
    fn is_member_node(&self, owner: &str, member_nodes: &BTreeSet<String>) -> bool {
        if split_qualified(owner).is_some() {
            return member_nodes.contains(owner);
        }
        self.members
            .iter()
            .filter(|member| member.member_id == owner)
            .count()
            == 1
    }
}

/// Adds one `node -> parent` reporting edge, rejecting unknown and self-referencing targets.
fn add_report_edge(
    node_id: &str,
    parent: &str,
    node_ids: &BTreeSet<String>,
    edges: &mut BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    if node_id == parent {
        return Err(Error::validation(format!(
            "'{node_id}' cannot report to itself"
        )));
    }
    if !node_ids.contains(parent) {
        return Err(Error::validation(format!(
            "'{node_id}' reports to unknown node '{parent}'"
        )));
    }
    edges
        .entry(node_id.to_owned())
        .or_default()
        .insert(parent.to_owned());
    Ok(())
}

/// A borrowed view of one chart node, returned by [`OrgChart::node`].
#[derive(Debug, Clone, Copy)]
pub enum OrgNodeRef<'a> {
    /// A team node (node id: its team name).
    Team(&'a OrgTeam),
    /// A member node (node id: `"<team>::<member>"`).
    Member(&'a OrgMember),
}

impl OrgNodeRef<'_> {
    /// Returns the node id: the team name for a team, the qualified member id for a member.
    #[must_use]
    pub fn id(&self) -> String {
        match self {
            Self::Team(team) => team.team_name.clone(),
            Self::Member(member) => member.node_id(),
        }
    }

    /// Returns the node's role.
    #[must_use]
    pub fn role(&self) -> OrgRole {
        match self {
            Self::Team(team) => team.role,
            Self::Member(member) => member.role,
        }
    }

    /// Returns the node id this node reports to, if any.
    #[must_use]
    pub fn reports_to(&self) -> Option<&str> {
        match self {
            Self::Team(team) => team.reports_to.as_deref(),
            Self::Member(member) => member.reports_to.as_deref(),
        }
    }
}

/// A validated organisation chart: teams, members and the reporting tree they form.
///
/// Build it once with [`OrgChart::new`] (which runs `OrganizationSpec::validate`) and share it
/// between the delegation router, the task pool and the mailbox. Every accessor is a pure lookup or
/// a bounded walk of the reporting tree, so the chart is `Clone + Send + Sync` and cheap to share
/// behind an `Arc`.
///
/// The reporting tree is the union of team and member `reports_to` links. A node with no
/// `reports_to` is a root (depth `0`); several roots may coexist — an organisation chart is a
/// forest, not necessarily a single tree.
#[derive(Debug, Clone)]
pub struct OrgChart {
    spec: OrganizationSpec,
    teams: BTreeMap<String, OrgTeam>,
    members: BTreeMap<String, BTreeMap<String, OrgMember>>,
    parents: BTreeMap<String, String>,
    children: BTreeMap<String, BTreeSet<String>>,
}

impl OrgChart {
    /// Validates `spec` and resolves it into a chart.
    ///
    /// # Errors
    ///
    /// Propagates every [`OrganizationSpec::validate`] failure.
    pub fn new(spec: OrganizationSpec) -> Result<Self> {
        spec.validate()?;
        let teams = spec
            .teams
            .iter()
            .map(|team| (team.team_name.clone(), team.clone()))
            .collect();
        let mut members: BTreeMap<String, BTreeMap<String, OrgMember>> = BTreeMap::new();
        for member in &spec.members {
            members
                .entry(member.team_name.clone())
                .or_default()
                .insert(member.member_id.clone(), member.clone());
        }
        let mut parents = BTreeMap::new();
        let mut children: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for team in &spec.teams {
            children.entry(team.team_name.clone()).or_default();
            if let Some(parent) = &team.reports_to {
                parents.insert(team.team_name.clone(), parent.clone());
            }
        }
        for member in &spec.members {
            let node_id = member.node_id();
            children.entry(node_id.clone()).or_default();
            if let Some(parent) = &member.reports_to {
                parents.insert(node_id, parent.clone());
            }
        }
        for (node_id, parent) in &parents {
            children
                .entry(parent.clone())
                .or_default()
                .insert(node_id.clone());
        }
        Ok(Self {
            spec,
            teams,
            members,
            parents,
            children,
        })
    }

    /// Returns the spec this chart was built from.
    #[must_use]
    pub fn spec(&self) -> &OrganizationSpec {
        &self.spec
    }

    /// Returns the organisation id.
    #[must_use]
    pub fn organization_id(&self) -> &str {
        &self.spec.organization_id
    }

    /// Returns the node named `node_id` (a team name or a qualified member id).
    #[must_use]
    pub fn node(&self, node_id: &str) -> Option<OrgNodeRef<'_>> {
        if let Some(team) = self.teams.get(node_id) {
            return Some(OrgNodeRef::Team(team));
        }
        let (team_name, member_id) = split_qualified(node_id)?;
        self.member(team_name, member_id).map(OrgNodeRef::Member)
    }

    /// Returns the team named `team_name`, if declared.
    #[must_use]
    pub fn team(&self, team_name: &str) -> Option<&OrgTeam> {
        self.teams.get(team_name)
    }

    /// Returns the member `member_id` of team `team_name`, if declared.
    #[must_use]
    pub fn member(&self, team_name: &str, member_id: &str) -> Option<&OrgMember> {
        self.members.get(team_name)?.get(member_id)
    }

    /// Returns every team, sorted by team name.
    #[must_use]
    pub fn teams(&self) -> Vec<&OrgTeam> {
        self.teams.values().collect()
    }

    /// Returns every member, sorted by `(team_name, member_id)`.
    #[must_use]
    pub fn members(&self) -> Vec<&OrgMember> {
        self.members.values().flat_map(BTreeMap::values).collect()
    }

    /// Returns the members of `team_name`, sorted by member id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when `team_name` is not a declared team.
    pub fn team_members(&self, team_name: &str) -> Result<Vec<&OrgMember>> {
        if !self.teams.contains_key(team_name) {
            return Err(Error::not_found(format!(
                "unknown organization team '{team_name}'"
            )));
        }
        let mut found: Vec<&OrgMember> = Vec::new();
        if let Some(members) = self.members.get(team_name) {
            found.extend(members.values());
        }
        Ok(found)
    }

    /// Returns the role of `node_id`, if the node exists.
    #[must_use]
    pub fn role_of(&self, node_id: &str) -> Option<OrgRole> {
        self.node(node_id).map(|node| node.role())
    }

    /// Returns the node id `node_id` reports to, if any.
    #[must_use]
    pub fn parent_of(&self, node_id: &str) -> Option<&str> {
        self.parents.get(node_id).map(String::as_str)
    }

    /// Returns the direct reports of `node_id`, sorted by node id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when `node_id` is not a chart node.
    pub fn direct_reports(&self, node_id: &str) -> Result<Vec<String>> {
        self.require_node(node_id)?;
        let mut found: Vec<String> = Vec::new();
        if let Some(children) = self.children.get(node_id) {
            found.extend(children.iter().cloned());
        }
        Ok(found)
    }

    /// Returns every node below `node_id`, sorted by node id and excluding `node_id` itself.
    ///
    /// The walk is breadth-first and de-duplicates, so a node reachable through two parents (a
    /// shared detail member) is reported once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when `node_id` is not a chart node.
    pub fn descendants(&self, node_id: &str) -> Result<Vec<String>> {
        self.require_node(node_id)?;
        let mut found: BTreeSet<String> = BTreeSet::new();
        let mut stack: Vec<String> = vec![node_id.to_owned()];
        while let Some(current) = stack.pop() {
            if let Some(children) = self.children.get(&current) {
                for child in children {
                    if found.insert(child.clone()) {
                        stack.push(child.clone());
                    }
                }
            }
        }
        Ok(found.into_iter().collect())
    }

    /// Returns the ancestors of `node_id`: nearest parent first, root last.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when `node_id` is not a chart node.
    pub fn ancestors(&self, node_id: &str) -> Result<Vec<String>> {
        self.require_node(node_id)?;
        let mut chain: Vec<String> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut current = self.parents.get(node_id).cloned();
        while let Some(parent) = current {
            // `validate` already refused cycles; the guard keeps a hand-built chart from looping.
            if !seen.insert(parent.clone()) {
                break;
            }
            chain.push(parent.clone());
            current = self.parents.get(&parent).cloned();
        }
        Ok(chain)
    }

    /// Returns the depth of `node_id`: a root without a reporting line is `0`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when `node_id` is not a chart node.
    pub fn depth_of(&self, node_id: &str) -> Result<usize> {
        Ok(self.ancestors(node_id)?.len())
    }

    /// Returns `true` when `ancestor` sits strictly above `node` in the reporting tree.
    ///
    /// A node is never its own ancestor, and an unknown id is never an ancestor of anything.
    #[must_use]
    pub fn is_ancestor(&self, ancestor: &str, node: &str) -> bool {
        self.ancestors(node)
            .is_ok_and(|ancestors| ancestors.iter().any(|id| id == ancestor))
    }

    /// Returns the leader member id of `team_name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when `team_name` is not a declared team.
    pub fn leader_of(&self, team_name: &str) -> Result<&str> {
        match self.teams.get(team_name) {
            Some(team) => Ok(team.leader_or_default()),
            None => Err(Error::not_found(format!(
                "unknown organization team '{team_name}'"
            ))),
        }
    }

    /// Returns the team a node belongs to: the team name itself for a team node, the owning team
    /// for a member node.
    #[must_use]
    pub fn team_of_node(&self, node_id: &str) -> Option<&str> {
        if let Some((team_name, _)) = self.teams.get_key_value(node_id) {
            return Some(team_name.as_str());
        }
        let (team_name, member_id) = split_qualified(node_id)?;
        self.member(team_name, member_id)
            .map(|member| member.team_name.as_str())
    }

    /// Serializes the underlying spec as JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::external`] if the spec cannot be serialized (a non-string map key, in
    /// practice only reachable through a hand-built [`serde_json::Value`] `metadata`).
    pub fn to_value(&self) -> Result<Value> {
        serde_json::to_value(&self.spec).map_err(Error::external)
    }

    /// Fails with [`Error::not_found`] unless `node_id` is a chart node.
    fn require_node(&self, node_id: &str) -> Result<()> {
        if self.node(node_id).is_some() {
            Ok(())
        } else {
            Err(Error::not_found(format!(
                "unknown organization node '{node_id}'"
            )))
        }
    }
}

#[cfg(test)]
#[path = "roles_tests.rs"]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod roles_tests;
