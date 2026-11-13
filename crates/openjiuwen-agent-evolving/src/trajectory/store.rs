//! In-memory trajectory archive with deterministic grouping and statistics.
//!
//! Ports the query surface of Python
//! `openjiuwen.agent_evolving.trajectory.store` (session/case lookups over an
//! in-memory archive) and adds the offline aggregates the evolution pipeline
//! needs: success rate, mean score, and per-tool usage.
//!
//! Insertion order is preserved: [`TrajectoryStore::add`] replaces an existing
//! trajectory in place (same id), and grouping keeps the store's order inside
//! each group. Group maps are [`BTreeMap`]s, so their keys iterate sorted and
//! results are deterministic.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::model::{Trajectory, TrajectoryOutcome};

/// A process-local archive of recorded trajectories.
///
/// Serializes as `{"trajectories": [...]}`; a payload without that key (or an
/// empty one) deserializes to an empty store.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrajectoryStore {
    trajectories: Vec<Trajectory>,
}

impl TrajectoryStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one trajectory, replacing an existing entry with the same id.
    ///
    /// Replacement keeps the original position so grouping and iteration stay
    /// in insertion order. Returns `true` when an existing trajectory was
    /// replaced and `false` when the input was appended.
    pub fn add(&mut self, trajectory: Trajectory) -> bool {
        if let Some(slot) = self
            .trajectories
            .iter_mut()
            .find(|existing| existing.trajectory_id == trajectory.trajectory_id)
        {
            *slot = trajectory;
            return true;
        }
        self.trajectories.push(trajectory);
        false
    }

    /// The trajectory with `trajectory_id`, if stored.
    #[must_use]
    pub fn get(&self, trajectory_id: &str) -> Option<&Trajectory> {
        self.trajectories
            .iter()
            .find(|trajectory| trajectory.trajectory_id == trajectory_id)
    }

    /// Removes and returns the trajectory with `trajectory_id`.
    pub fn remove(&mut self, trajectory_id: &str) -> Option<Trajectory> {
        let position = self
            .trajectories
            .iter()
            .position(|trajectory| trajectory.trajectory_id == trajectory_id)?;
        Some(self.trajectories.remove(position))
    }

    /// Number of stored trajectories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    /// All stored trajectories, in insertion order.
    #[must_use]
    pub fn trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }

    /// Iterates the stored trajectories in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &Trajectory> + '_ {
        self.trajectories.iter()
    }

    /// Trajectories recorded for `session_id`, in insertion order.
    #[must_use]
    pub fn by_session(&self, session_id: &str) -> Vec<&Trajectory> {
        self.trajectories
            .iter()
            .filter(|trajectory| trajectory.session_id.as_deref() == Some(session_id))
            .collect()
    }

    /// Trajectories recorded for `case_id`, in insertion order.
    #[must_use]
    pub fn by_case(&self, case_id: &str) -> Vec<&Trajectory> {
        self.trajectories
            .iter()
            .filter(|trajectory| trajectory.case_id.as_deref() == Some(case_id))
            .collect()
    }

    /// Trajectories with `outcome`, in insertion order.
    #[must_use]
    pub fn by_outcome(&self, outcome: TrajectoryOutcome) -> Vec<&Trajectory> {
        self.trajectories
            .iter()
            .filter(|trajectory| trajectory.outcome == outcome)
            .collect()
    }

    /// Groups trajectories by `session_id`, sorted by session id.
    ///
    /// Trajectories without a session id are not groupable and are skipped;
    /// the vectors keep the store's insertion order.
    #[must_use]
    pub fn group_by_session(&self) -> BTreeMap<String, Vec<&Trajectory>> {
        let mut groups: BTreeMap<String, Vec<&Trajectory>> = BTreeMap::new();
        for trajectory in &self.trajectories {
            if let Some(session_id) = &trajectory.session_id {
                groups
                    .entry(session_id.clone())
                    .or_default()
                    .push(trajectory);
            }
        }
        groups
    }

    /// Groups trajectories by `case_id`, sorted by case id.
    ///
    /// Trajectories without a case id are skipped; the vectors keep the
    /// store's insertion order.
    #[must_use]
    pub fn group_by_case(&self) -> BTreeMap<String, Vec<&Trajectory>> {
        let mut groups: BTreeMap<String, Vec<&Trajectory>> = BTreeMap::new();
        for trajectory in &self.trajectories {
            if let Some(case_id) = &trajectory.case_id {
                groups.entry(case_id.clone()).or_default().push(trajectory);
            }
        }
        groups
    }

    /// Aggregate statistics over the whole store.
    ///
    /// An empty store reports zeros. `success_rate` divides successes by the
    /// total count (unknown outcomes count in the denominator, not as
    /// failures); `mean_score` averages the trajectories that recorded a
    /// score, and is `0.0` when none did. `tool_usage` counts how many
    /// trajectories used each distinct tool name (once per trajectory, not
    /// once per call).
    #[must_use]
    pub fn summary(&self) -> TrajectoryStats {
        let references: Vec<&Trajectory> = self.trajectories.iter().collect();
        let (count, success_count, failure_count) = outcome_counts(&references);
        TrajectoryStats {
            count,
            success_count,
            failure_count,
            success_rate: ratio(success_count, count),
            mean_score: mean_score(&references),
            tool_usage: tool_usage(&references),
        }
    }

    /// Per-session summaries, sorted by session id.
    ///
    /// Each summary uses the same statistics as [`summary`](Self::summary)
    /// restricted to the session's trajectories.
    #[must_use]
    pub fn group_summaries(&self) -> Vec<TrajectoryGroupSummary> {
        self.group_by_session()
            .into_iter()
            .map(|(key, group)| {
                let (count, success_count, _) = outcome_counts(&group);
                TrajectoryGroupSummary {
                    key,
                    count,
                    success_rate: ratio(success_count, count),
                    mean_score: mean_score(&group),
                }
            })
            .collect()
    }
}

/// Aggregate statistics of a [`TrajectoryStore`] or one session group.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrajectoryStats {
    /// Total number of trajectories.
    pub count: usize,
    /// Trajectories whose outcome is [`TrajectoryOutcome::Success`].
    pub success_count: usize,
    /// Trajectories whose outcome is [`TrajectoryOutcome::Failure`].
    pub failure_count: usize,
    /// `success_count / count`, or `0.0` for an empty store.
    pub success_rate: f64,
    /// Mean of the recorded scores, or `0.0` when none were recorded.
    pub mean_score: f64,
    /// Number of trajectories that used each distinct tool name.
    pub tool_usage: BTreeMap<String, usize>,
}

/// Statistics of one session group, keyed by the session id.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrajectoryGroupSummary {
    /// The group key (the session id).
    pub key: String,
    /// Total number of trajectories in the group.
    pub count: usize,
    /// `success_count / count`, or `0.0` for an empty group.
    pub success_rate: f64,
    /// Mean of the recorded scores, or `0.0` when none were recorded.
    pub mean_score: f64,
}

/// Counts total, successful, and failed trajectories.
fn outcome_counts(trajectories: &[&Trajectory]) -> (usize, usize, usize) {
    let mut success_count = 0usize;
    let mut failure_count = 0usize;
    for trajectory in trajectories {
        match trajectory.outcome {
            TrajectoryOutcome::Success => success_count += 1,
            TrajectoryOutcome::Failure => failure_count += 1,
            TrajectoryOutcome::Unknown => {}
        }
    }
    (trajectories.len(), success_count, failure_count)
}

/// Divides `part` by `whole`, reporting `0.0` when `whole` is zero.
fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// Averaged score of the trajectories that recorded one.
fn mean_score(trajectories: &[&Trajectory]) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for trajectory in trajectories {
        if let Some(score) = trajectory.score {
            total += score;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Per-tool trajectory counts.
fn tool_usage(trajectories: &[&Trajectory]) -> BTreeMap<String, usize> {
    let mut usage: BTreeMap<String, usize> = BTreeMap::new();
    for trajectory in trajectories {
        for tool in trajectory.tool_names() {
            *usage.entry(tool.to_owned()).or_default() += 1;
        }
    }
    usage
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::trajectory::model::{StepKind, TrajectoryStep};

    fn trajectory(
        id: &str,
        session: &str,
        outcome: TrajectoryOutcome,
        score: Option<f64>,
    ) -> Trajectory {
        let mut trajectory = Trajectory::new(id)
            .expect("valid id")
            .with_session(session)
            .with_outcome(outcome);
        if let Some(score) = score {
            trajectory = trajectory.with_score(score).expect("finite score");
        }
        trajectory.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("search"));
        trajectory
    }

    #[test]
    fn add_replaces_in_place() {
        let mut store = TrajectoryStore::new();
        assert!(!store.add(trajectory(
            "t-1",
            "s-1",
            TrajectoryOutcome::Success,
            Some(1.0)
        )));
        let mut replacement = Trajectory::new("t-1")
            .expect("valid id")
            .with_session("s-2")
            .with_outcome(TrajectoryOutcome::Failure);
        replacement.push(TrajectoryStep::new(StepKind::Message, "again"));
        assert!(store.add(replacement));
        assert_eq!(store.len(), 1);
        assert_eq!(
            store.get("t-1").map(|t| t.session_id.clone()),
            Some(Some("s-2".to_owned()))
        );
    }

    #[test]
    fn summary_is_hand_computed() {
        let mut store = TrajectoryStore::new();
        store.add(trajectory(
            "t-1",
            "s-1",
            TrajectoryOutcome::Success,
            Some(1.0),
        ));
        store.add(trajectory(
            "t-2",
            "s-1",
            TrajectoryOutcome::Failure,
            Some(0.0),
        ));
        store.add(trajectory("t-3", "s-2", TrajectoryOutcome::Unknown, None));
        let summary = store.summary();
        assert_eq!(summary.count, 3);
        assert_eq!(summary.success_count, 1);
        assert_eq!(summary.failure_count, 1);
        assert!((summary.success_rate - 1.0 / 3.0).abs() < f64::EPSILON);
        assert!((summary.mean_score - 0.5).abs() < f64::EPSILON);
        assert_eq!(summary.tool_usage.get("search"), Some(&3));

        let groups = store.group_summaries();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].key, "s-1");
        assert_eq!(groups[0].count, 2);
        assert!((groups[0].success_rate - 0.5).abs() < f64::EPSILON);
        assert!((groups[0].mean_score - 0.5).abs() < f64::EPSILON);
        assert_eq!(groups[1].key, "s-2");
        assert!((groups[1].success_rate - 0.0).abs() < f64::EPSILON);
        assert!((groups[1].mean_score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_store_reports_zeros() {
        let summary = TrajectoryStore::new().summary();
        assert_eq!(summary, TrajectoryStats::default());
        let groups: Vec<TrajectoryGroupSummary> = TrajectoryStore::new().group_summaries();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn store_serde_round_trips() {
        let mut store = TrajectoryStore::new();
        store.add(trajectory(
            "t-1",
            "s-1",
            TrajectoryOutcome::Success,
            Some(0.25),
        ));
        let payload = serde_json::to_value(&store).expect("serialize");
        assert_eq!(payload["trajectories"][0]["trajectory_id"], json!("t-1"));
        let restored: TrajectoryStore = serde_json::from_value(payload).expect("deserialize");
        assert_eq!(restored, store);
    }
}
