//! Deterministic clustering of parsed traces, the first stage of the experience pipeline.
//!
//! Python groups traces by their skill set and then runs FAISS K-Means over the query
//! embeddings. An offline build has no faiss, so this module keeps the two-stage shape
//! and replaces the K-Means step with a deterministic greedy clustering over the same
//! cosine-like quantity computed lexically:
//!
//! 1. **Skill stage** — traces are grouped by their sorted, deduped skill set. A trace
//!    with no skills forms its own skill-less group (the collector drops such traces
//!    before clustering).
//! 2. **Query stage** — inside a group, traces join the first cluster whose seed query has
//!    a token Jaccard similarity of at least [`CLUSTER_SIMILARITY_THRESHOLD`] to theirs;
//!    otherwise they start a new cluster. Traces are visited in trace-id order, so the
//!    assignment is deterministic and has no RNG.
//!
//! The centroid query of a cluster is the member whose token set is closest to the
//! cluster's mean token set (tokens present in at least half of the members), with ties
//! broken by trace id. Output is sorted by `(skill_ids, pattern)`, so two calls over the
//! same input always agree.
use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::experience::ranking::query_terms;
use crate::experience::trace::TraceRecord;

/// The minimum token Jaccard similarity for a trace to join an existing query cluster.
///
/// `0.6` keeps paraphrases of one request together while separating distinct requests
/// that happen to use the same skills. It replaces Python's K-Means cosine threshold,
/// which needs the faiss/ML stack.
pub const CLUSTER_SIMILARITY_THRESHOLD: f64 = 0.6;

/// One cluster of traces that share a skill set and a query topic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ClusteredQuery {
    /// The centroid query: the member query closest to the cluster's mean token set.
    pub pattern: String,
    /// The trace ids of the cluster's members, sorted.
    pub members: Vec<String>,
    /// The distinct non-blank member queries, in member order.
    pub query_examples: Vec<String>,
    /// The sorted, deduped skill ids shared by every member.
    pub skill_ids: Vec<String>,
    /// How many members succeeded.
    pub success_count: u32,
}

/// Clusters `records` first by skill set, then by normalized-query similarity.
///
/// The function is success-agnostic: failures are clustered alongside successes and
/// counted by [`ClusteredQuery::success_count`], so a caller that wants positive evidence
/// only filters first (the collector does). Empty input answers an empty list.
#[must_use]
pub fn cluster_traces(records: &[TraceRecord]) -> Vec<ClusteredQuery> {
    let mut skill_groups: BTreeMap<Vec<String>, Vec<&TraceRecord>> = BTreeMap::new();
    for record in records {
        skill_groups
            .entry(skill_key(record))
            .or_default()
            .push(record);
    }

    let mut clusters = Vec::new();
    for (skill_ids, mut group) in skill_groups {
        group.sort_by(|left, right| {
            left.trace_id
                .cmp(&right.trace_id)
                .then_with(|| left.timestamp.total_cmp(&right.timestamp))
                .then_with(|| left.query.cmp(&right.query))
        });
        clusters.extend(cluster_group(skill_ids, group));
    }
    clusters.sort_by(|left, right| {
        left.skill_ids
            .cmp(&right.skill_ids)
            .then_with(|| left.pattern.cmp(&right.pattern))
            .then_with(|| left.members.cmp(&right.members))
    });
    clusters
}

/// One in-progress cluster: its seed query tokens and its member traces.
struct GroupCluster<'a> {
    /// The tokens of the first member's query; the first-match comparison target.
    seed: BTreeSet<String>,
    /// The member traces in trace-id order.
    members: Vec<&'a TraceRecord>,
}

/// Clusters one skill group by normalized-query similarity.
fn cluster_group(skill_ids: Vec<String>, group: Vec<&TraceRecord>) -> Vec<ClusteredQuery> {
    let mut local: Vec<GroupCluster<'_>> = Vec::new();
    for record in group {
        let tokens = query_terms(&record.query);
        let target = local
            .iter_mut()
            .find(|cluster| jaccard(&cluster.seed, &tokens) >= CLUSTER_SIMILARITY_THRESHOLD);
        match target {
            Some(cluster) => cluster.members.push(record),
            None => local.push(GroupCluster {
                seed: tokens,
                members: vec![record],
            }),
        }
    }
    local
        .into_iter()
        .map(|cluster| populate_cluster(skill_ids.clone(), cluster.members))
        .collect()
}

/// Builds the public cluster record: pattern, members, examples, and success count.
fn populate_cluster(skill_ids: Vec<String>, members: Vec<&TraceRecord>) -> ClusteredQuery {
    let mut member_ids: Vec<String> = members
        .iter()
        .map(|member| member.trace_id.clone())
        .collect();
    member_ids.sort();
    member_ids.dedup();

    let mut query_examples: Vec<String> = Vec::new();
    for member in &members {
        let query = member.query.trim();
        if !query.is_empty() && !query_examples.iter().any(|known| known == query) {
            query_examples.push(query.to_string());
        }
    }

    let success_count =
        u32::try_from(members.iter().filter(|member| member.success).count()).unwrap_or(u32::MAX);
    ClusteredQuery {
        pattern: centroid_query(&members),
        members: member_ids,
        query_examples,
        skill_ids,
        success_count,
    }
}

/// Picks the member query closest to the cluster's mean token set.
///
/// The mean set holds the tokens present in at least half of the members. The member with
/// the highest Jaccard similarity to it wins; ties keep the earliest trace id, because
/// `members` is already in trace-id order and only a strictly higher score replaces it.
fn centroid_query(members: &[&TraceRecord]) -> String {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for member in members {
        for token in query_terms(&member.query) {
            *counts.entry(token).or_default() += 1;
        }
    }
    let mean: BTreeSet<String> = counts
        .into_iter()
        .filter(|(_, count)| *count * 2 >= members.len())
        .map(|(token, _)| token)
        .collect();

    let mut best: Option<(&TraceRecord, f64)> = None;
    for member in members {
        let similarity = jaccard(&query_terms(&member.query), &mean);
        if best.is_none_or(|(_, best_similarity)| similarity > best_similarity) {
            best = Some((member, similarity));
        }
    }
    best.map_or_else(String::new, |(member, _)| member.query.trim().to_string())
}

/// The sorted, deduped, non-blank skill ids of `record`.
pub(crate) fn skill_key(record: &TraceRecord) -> Vec<String> {
    normalized_skills(&record.skills)
}

/// The sorted, deduped, non-blank form of a skill-id list.
pub(crate) fn normalized_skills(skill_ids: &[String]) -> Vec<String> {
    let mut skills: Vec<String> = skill_ids
        .iter()
        .map(|skill| skill.trim().to_string())
        .filter(|skill| !skill.is_empty())
        .collect();
    skills.sort();
    skills.dedup();
    skills
}

/// The Jaccard similarity of two token sets, in `[0, 1]`.
///
/// Two empty sets are `1.0` (the queries carry no distinguishing terms), and an empty set
/// against a non-empty one is `0.0`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let union = left.union(right).count();
    if union == 0 {
        return 1.0;
    }
    left.intersection(right).count() as f64 / union as f64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn record(trace_id: &str, query: &str, skills: &[&str], success: bool) -> TraceRecord {
        TraceRecord {
            trace_id: trace_id.to_string(),
            session_id: "s".to_string(),
            request_id: trace_id.to_string(),
            query: query.to_string(),
            skills: skills.iter().map(|skill| (*skill).to_string()).collect(),
            success,
            ..TraceRecord::default()
        }
    }

    #[test]
    fn jaccard_handles_empty_sets() {
        let empty = BTreeSet::new();
        let tokens: BTreeSet<String> = ["a".to_string()].into_iter().collect();
        assert_eq!(jaccard(&empty, &empty), 1.0);
        assert_eq!(jaccard(&empty, &tokens), 0.0);
        assert_eq!(jaccard(&tokens, &tokens), 1.0);
    }

    #[test]
    fn clustering_splits_distinct_topics_inside_one_skill_set() {
        let records = vec![
            record("t1", "deploy the kubernetes cluster", &["ops"], true),
            record("t2", "deploy the kubernetes cluster now", &["ops"], true),
            record("t3", "bake a chocolate cake", &["ops"], false),
        ];
        let clusters = cluster_traces(&records);
        assert_eq!(clusters.len(), 2);
        // Sorted by (skill_ids, pattern): "bake ..." sorts before "deploy ...".
        assert_eq!(clusters[0].members, vec!["t3"]);
        assert_eq!(clusters[0].success_count, 0);
        assert_eq!(clusters[1].members, vec!["t1", "t2"]);
        assert_eq!(clusters[1].pattern, "deploy the kubernetes cluster now");
        assert_eq!(clusters[1].success_count, 2);
    }

    #[test]
    fn centroid_ties_break_on_trace_id() {
        let records = vec![
            record("b", "alpha beta gamma", &["x"], true),
            record("a", "alpha beta gamma", &["x"], true),
        ];
        let clusters = cluster_traces(&records);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members, vec!["a", "b"]);
        assert_eq!(clusters[0].pattern, "alpha beta gamma");
    }
}
