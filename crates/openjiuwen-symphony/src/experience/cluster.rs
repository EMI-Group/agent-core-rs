//! Deterministic k-means clustering of parsed traces, the first stage of the experience
//! pipeline.
//!
//! Python groups traces by their skill set and then runs FAISS K-Means over the query
//! embeddings. This port keeps that two-stage shape and replaces faiss with a small,
//! deterministic k-means over the vectors of the
//! [`Embedding`](openjiuwen_retrieval::Embedding) seam:
//!
//! 1. **Skill stage** — traces are grouped by their sorted, deduped skill set. A trace with
//!    no skills forms its own skill-less group (the collector drops such traces before
//!    clustering).
//! 2. **Query stage** — inside a group, every distinct query is embedded once, the vectors
//!    are L2-normalised, and k-means runs: `k = ceil(sqrt(n / 2))` bounded by
//!    [`ClusterConfig`], deterministic k-means++-style seeding (no RNG), then Lloyd
//!    iterations bounded by [`ClusterConfig::max_iterations`] and
//!    [`ClusterConfig::epsilon`].
//!
//! The centroid query of a cluster is the member whose vector has the highest cosine
//! similarity to the cluster's L2-normalised mean vector, with ties broken by trace id.
//! Output is sorted by `(skill_ids, pattern, members)`, so two calls over the same input
//! always agree.
//!
//! # Two entry points, one algorithm
//!
//! [`cluster_traces`] is the synchronous offline path: it runs the same k-means over
//! [`HashEmbedding`](openjiuwen_retrieval::HashEmbedding) vectors with
//! [`ClusterConfig::default`]. [`cluster_traces_with_embedding`] takes the caller's own
//! embedder and reports failures. The chosen embedder is the only difference between them —
//! the synchronous path never substitutes a lexical measure for the vectors, so a caller
//! that wants a model's semantic clusters must call the asynchronous entry point. There is
//! therefore no silent quality switch between the two.
//!
//! `HashEmbedding`'s futures are pure and ready on their first poll, so [`cluster_traces`]
//! drives [`cluster_traces_with_embedding`] through a single-poll helper built on an
//! `Arc`-based `std::task::Wake` waker and `Box::pin`: no `unsafe`, no private runtime and
//! no `futures` crate. Should a poll ever come back pending — impossible for
//! `HashEmbedding`, because this path never constructs a future that awaits real I/O — the
//! function logs a `tracing::warn!` and clusters over zero vectors instead of panicking.
//! The same fallback covers an embedding error, which `HashEmbedding` cannot produce
//! either.
//!
//! # Determinism rules
//!
//! * Traces inside a skill group are visited in trace-id order (timestamp and query break
//!   exact id ties).
//! * Seeding starts from the first point and always adds the point farthest from its
//!   nearest chosen centroid; a tie keeps the earliest trace id.
//! * Assignment picks the nearest centroid by squared euclidean distance on the normalised
//!   vectors; a tie keeps the lowest centroid index.
//! * An empty cluster is reseeded with the point farthest from its own centroid among the
//!   clusters that still hold more than one member (ties by trace id).
//! * A query with no direction (blank text, or a zero vector from the embedder) joins the
//!   group's first list, because its cosine similarity is `0.0` against every centroid.
//! * A mean vector that cancels to zero keeps the cluster's previous centroid.
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use openjiuwen_retrieval::{Embedding, HashEmbedding};

use crate::error::{validation, Error, Result};
use crate::experience::ranking::cosine_similarity;
use crate::experience::trace::TraceRecord;

/// The largest number of query clusters one skill group produces by default (`8`).
pub const DEFAULT_MAX_CLUSTERS: usize = 8;

/// The smallest number of query clusters one skill group produces by default (`1`).
pub const DEFAULT_MIN_CLUSTERS: usize = 1;

/// The default Lloyd iteration cap (`25`).
pub const DEFAULT_MAX_ITERATIONS: usize = 25;

/// The default largest-centroid-shift tolerance that stops the iterations (`1e-6`).
pub const DEFAULT_EPSILON: f64 = 1e-6;

/// The absolute tolerance applied to the optional centroid-merge comparison (`1e-9`).
///
/// The cosine of two L2-normalised vectors is `1.0` only up to floating-point rounding, so
/// the tolerance lets a threshold of `1.0` merge identical centroids without loosening the
/// comparison in any meaningful way.
const MERGE_TOLERANCE: f64 = 1e-9;

/// Tunables for the k-means query clustering.
///
/// Every field is `serde`-defaulted, so a partial document configures only what it changes.
/// [`ClusterConfig::default`] is what [`cluster_traces`] uses; the bounds are checked by
/// [`ClusterConfig::validate`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ClusterConfig {
    /// The largest number of query clusters one skill group may produce (`8` by default).
    ///
    /// This is the hard ceiling on the `ceil(sqrt(n / 2))` heuristic, not a target: a group
    /// of one member still yields one cluster.
    pub max_clusters: usize,
    /// The smallest number of query clusters one skill group may produce (`1` by default).
    ///
    /// Setting this to `2` forces a split even when the heuristic would keep a group whole;
    /// the members that carry no direction still share the group's first list.
    pub min_clusters: usize,
    /// The Lloyd iteration cap per skill group (`25` by default).
    ///
    /// Must be at least `1`, because a single iteration is what turns the seeds into means.
    pub max_iterations: usize,
    /// The largest-centroid-shift tolerance that ends the iterations early (`1e-6`).
    ///
    /// The shift is the largest euclidean distance a centroid moved in one iteration; the
    /// loop stops once the assignments are stable and the shift is at most this value.
    pub epsilon: f64,
    /// An optional post-pass merging clusters whose centroids are this similar (`None`).
    ///
    /// When set to `t`, cluster pairs whose L2-normalised mean vectors reach a cosine
    /// similarity of `t` are merged before the cluster records are built, lowest list
    /// indexes first, and the merged cluster's pattern is re-derived from all its members.
    /// [`ClusterConfig::validate`] requires a finite value in `[-1, 1]`.
    pub centroid_merge_threshold: Option<f64>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            max_clusters: DEFAULT_MAX_CLUSTERS,
            min_clusters: DEFAULT_MIN_CLUSTERS,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            epsilon: DEFAULT_EPSILON,
            centroid_merge_threshold: None,
        }
    }
}

impl ClusterConfig {
    /// Checks the bounds the k-means relies on.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending field when `max_clusters` or
    /// `min_clusters` is zero, when `min_clusters` exceeds `max_clusters`, when
    /// `max_iterations` is zero, when `epsilon` is not a finite non-negative number, or
    /// when `centroid_merge_threshold` is not a finite cosine in `[-1, 1]`.
    pub fn validate(&self) -> Result<()> {
        if self.max_clusters == 0 {
            return Err(validation("max_clusters", "must be at least 1"));
        }
        if self.min_clusters == 0 {
            return Err(validation("min_clusters", "must be at least 1"));
        }
        if self.min_clusters > self.max_clusters {
            return Err(validation(
                "min_clusters",
                format!("must not exceed max_clusters ({})", self.max_clusters),
            ));
        }
        if self.max_iterations == 0 {
            return Err(validation("max_iterations", "must be at least 1"));
        }
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(validation("epsilon", "must be a finite non-negative number"));
        }
        if let Some(threshold) = self.centroid_merge_threshold {
            if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
                return Err(validation(
                    "centroid_merge_threshold",
                    "must be a finite cosine similarity in [-1, 1]",
                ));
            }
        }
        Ok(())
    }
}

/// One cluster of traces that share a skill set and a query topic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ClusteredQuery {
    /// The centroid query: the member query closest to the cluster's mean vector.
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

/// Clusters `records` first by skill set, then by query embedding similarity.
///
/// This is the synchronous offline path: it runs the same k-means as
/// [`cluster_traces_with_embedding`] over
/// [`HashEmbedding`](openjiuwen_retrieval::HashEmbedding) vectors with
/// [`ClusterConfig::default`]. `HashEmbedding`'s futures are ready on their first poll, so a
/// single poll of the asynchronous entry point completes; a pending poll or an embedding
/// error (neither of which `HashEmbedding` can produce) is logged with `tracing::warn!` and
/// the clusters are built over zero vectors.
///
/// The function is success-agnostic: failures are clustered alongside successes and counted
/// by [`ClusteredQuery::success_count`], so a caller that wants positive evidence only
/// filters first (the collector does). Empty input answers an empty list.
#[must_use]
pub fn cluster_traces(records: &[TraceRecord]) -> Vec<ClusteredQuery> {
    let groups = skill_groups(records);
    let embedder = HashEmbedding::default();
    let config = ClusterConfig::default();
    let vectors = match poll_ready(embed_queries(&groups, &embedder)) {
        Some(Ok(vectors)) => vectors,
        Some(Err(error)) => {
            tracing::warn!(
                error = %error,
                "hash embedding failed; clustering over zero vectors"
            );
            HashMap::new()
        }
        None => {
            tracing::warn!(
                "hash embedding did not complete on its first poll; \
                 clustering over zero vectors"
            );
            HashMap::new()
        }
    };
    clustered_groups(&groups, &vectors, &config)
}

/// Clusters `records` first by skill set, then by k-means over `embedding`'s vectors.
///
/// Every distinct non-blank query is embedded once, in trace-id order, with a single
/// [`Embedding::embed_documents`] call; the vectors are L2-normalised (a zero vector stays
/// zero) and clustered per skill group as described in the [module docs](crate::experience).
/// `k` is `ceil(sqrt(n / 2))` clamped to [`ClusterConfig::min_clusters`] and
/// `min(max_clusters, n)`, and is further capped by the number of members that carry a
/// direction.
///
/// # Errors
///
/// Returns [`ClusterConfig::validate`]'s [`Error::Validation`] for an out-of-range config,
/// the embedder's own error when a call fails (no trace is dropped silently), or
/// [`Error::Retrieval`] when the embedder answers a different number of vectors than it was
/// given.
pub async fn cluster_traces_with_embedding(
    records: &[TraceRecord],
    embedding: &dyn Embedding,
    config: &ClusterConfig,
) -> Result<Vec<ClusteredQuery>> {
    config.validate()?;
    let groups = skill_groups(records);
    let vectors = embed_queries(&groups, embedding).await?;
    Ok(clustered_groups(&groups, &vectors, config))
}

/// Groups `records` by skill key, each group in trace-id order.
fn skill_groups(records: &[TraceRecord]) -> Vec<(Vec<String>, Vec<&TraceRecord>)> {
    let mut groups: BTreeMap<Vec<String>, Vec<&TraceRecord>> = BTreeMap::new();
    for record in records {
        groups.entry(skill_key(record)).or_default().push(record);
    }
    groups
        .into_iter()
        .map(|(skill_ids, mut members)| {
            members.sort_by(|left, right| {
                left.trace_id
                    .cmp(&right.trace_id)
                    .then_with(|| left.timestamp.total_cmp(&right.timestamp))
                    .then_with(|| left.query.cmp(&right.query))
            });
            (skill_ids, members)
        })
        .collect()
}

/// Embeds every distinct non-blank query once, keyed by the trimmed query text.
async fn embed_queries(
    groups: &[(Vec<String>, Vec<&TraceRecord>)],
    embedding: &dyn Embedding,
) -> Result<HashMap<String, Vec<f32>>> {
    let mut ordered: Vec<&TraceRecord> = groups
        .iter()
        .flat_map(|(_, members)| members.iter().copied())
        .collect();
    ordered.sort_by(|left, right| {
        left.trace_id
            .cmp(&right.trace_id)
            .then_with(|| left.timestamp.total_cmp(&right.timestamp))
            .then_with(|| left.query.cmp(&right.query))
    });

    let mut texts: Vec<String> = Vec::new();
    for record in ordered {
        let query = record.query.trim();
        if query.is_empty() || texts.iter().any(|known| known == query) {
            continue;
        }
        texts.push(query.to_string());
    }
    if texts.is_empty() {
        return Ok(HashMap::new());
    }

    let embedded = embedding.embed_documents(&texts, None).await?;
    if embedded.len() != texts.len() {
        return Err(Error::retrieval(format!(
            "embed_documents answered {} vectors for {} queries",
            embedded.len(),
            texts.len()
        )));
    }
    Ok(texts
        .into_iter()
        .zip(embedded)
        .map(|(text, vector)| (text, normalized(vector)))
        .collect())
}

/// Builds the cluster records for every skill group and sorts the result.
fn clustered_groups(
    groups: &[(Vec<String>, Vec<&TraceRecord>)],
    vectors: &HashMap<String, Vec<f32>>,
    config: &ClusterConfig,
) -> Vec<ClusteredQuery> {
    let mut clusters = Vec::new();
    for (skill_ids, members) in groups {
        clusters.extend(cluster_group(skill_ids, members, vectors, config));
    }
    clusters.sort_by(|left, right| {
        left.skill_ids
            .cmp(&right.skill_ids)
            .then_with(|| left.pattern.cmp(&right.pattern))
            .then_with(|| left.members.cmp(&right.members))
    });
    clusters
}

/// One trace of a skill group together with its L2-normalised query vector.
struct QueryPoint<'a> {
    /// The member trace.
    record: &'a TraceRecord,
    /// The normalised query vector, empty when the query is blank or unembedded.
    vector: Vec<f32>,
}

/// Clusters one skill group and turns each list of members into a [`ClusteredQuery`].
fn cluster_group(
    skill_ids: &[String],
    members: &[&TraceRecord],
    vectors: &HashMap<String, Vec<f32>>,
    config: &ClusterConfig,
) -> Vec<ClusteredQuery> {
    let points: Vec<QueryPoint<'_>> = members
        .iter()
        .map(|record| QueryPoint {
            record,
            vector: lookup_vector(vectors, &record.query),
        })
        .collect();
    let lists = cluster_lists(&points, config);
    let lists = match config.centroid_merge_threshold {
        Some(threshold) => merge_lists(lists, &points, threshold),
        None => lists,
    };
    lists
        .iter()
        .map(|list| cluster_record(skill_ids, &points, list))
        .collect()
}

/// The normalised vector of `query`, or an empty vector when it has none.
fn lookup_vector(vectors: &HashMap<String, Vec<f32>>, query: &str) -> Vec<f32> {
    let query = query.trim();
    if query.is_empty() {
        return Vec::new();
    }
    vectors.get(query).cloned().unwrap_or_default()
}

/// The number of query clusters for a skill group of `n` members.
///
/// `ceil(sqrt(n / 2))` is Python's faiss k-means default, clamped to
/// [`ClusterConfig::min_clusters`] and `min(max_clusters, n)`; a caller overrides it by
/// narrowing those bounds. One member yields one cluster and nine members yield three.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn target_clusters(n: usize, config: &ClusterConfig) -> usize {
    if n == 0 {
        return 0;
    }
    let heuristic = (n as f64 / 2.0).sqrt().ceil() as usize;
    let upper = config.max_clusters.min(n).max(1);
    let lower = config.min_clusters.min(upper).max(1);
    heuristic.clamp(lower, upper)
}

/// Partitions one skill group's points into cluster-member lists.
fn cluster_lists(points: &[QueryPoint<'_>], config: &ClusterConfig) -> Vec<Vec<usize>> {
    let directional: Vec<usize> = (0..points.len())
        .filter(|index| has_direction(&points[*index].vector))
        .collect();
    let mut lists = vec![Vec::new()];
    if !directional.is_empty() {
        let k = target_clusters(points.len(), config)
            .min(directional.len())
            .max(1);
        let vectors: Vec<Vec<f64>> = directional
            .iter()
            .map(|index| to_f64(&points[*index].vector))
            .collect();
        let (assignments, _) = run_kmeans(&vectors, k, config);
        let mut assigned = vec![Vec::new(); k];
        for (position, index) in directional.iter().enumerate() {
            assigned[assignments[position]].push(*index);
        }
        assigned.retain(|list| !list.is_empty());
        lists = assigned;
    }
    // A zero vector has the same similarity to every centroid, so its member joins the
    // group's first list.
    for (index, point) in points.iter().enumerate() {
        if has_direction(&point.vector) {
            continue;
        }
        match lists.first_mut() {
            Some(first) => first.push(index),
            None => lists.push(vec![index]),
        }
    }
    for list in &mut lists {
        list.sort_unstable();
    }
    lists
}

/// Runs seeding plus Lloyd iterations and answers the final assignments and centroids.
fn run_kmeans(
    vectors: &[Vec<f64>],
    k: usize,
    config: &ClusterConfig,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let mut centroids = seeded_centroids(vectors, k);
    let mut assignments = assign_clusters(vectors, &centroids);
    for _ in 0..config.max_iterations {
        let (next, shift) = updated_centroids(vectors, &assignments, &centroids);
        centroids = next;
        let next_assignments = assign_clusters(vectors, &centroids);
        let stable = next_assignments == assignments;
        assignments = next_assignments;
        if stable && shift <= config.epsilon {
            break;
        }
    }
    (assignments, centroids)
}

/// Picks `k` seeds k-means++-style without RNG.
///
/// The first seed is the first point, and every further seed is the point farthest from its
/// nearest chosen seed; a tie keeps the earliest point, so the seeds follow trace-id order.
fn seeded_centroids(vectors: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut chosen = vec![false; vectors.len()];
    if let Some(first) = vectors.first() {
        centroids.push(first.clone());
        chosen[0] = true;
    }
    while centroids.len() < k {
        let mut best: Option<(usize, f64)> = None;
        for (position, vector) in vectors.iter().enumerate() {
            if chosen[position] {
                continue;
            }
            let nearest = centroids
                .iter()
                .map(|centroid| squared_distance(vector, centroid))
                .fold(f64::INFINITY, f64::min);
            if best.is_none_or(|(_, distance)| nearest > distance) {
                best = Some((position, nearest));
            }
        }
        let Some((position, _)) = best else {
            break;
        };
        chosen[position] = true;
        centroids.push(vectors[position].clone());
    }
    centroids
}

/// Assigns every point to its nearest centroid, ties keeping the lowest index.
fn assign_clusters(vectors: &[Vec<f64>], centroids: &[Vec<f64>]) -> Vec<usize> {
    vectors
        .iter()
        .map(|vector| {
            let mut best = 0;
            let mut best_distance = f64::INFINITY;
            for (index, centroid) in centroids.iter().enumerate() {
                let distance = squared_distance(vector, centroid);
                if distance < best_distance {
                    best = index;
                    best_distance = distance;
                }
            }
            best
        })
        .collect()
}

/// One Lloyd update: the normalised mean of each cluster plus the largest centroid shift.
#[allow(clippy::cast_precision_loss)]
fn updated_centroids(
    vectors: &[Vec<f64>],
    assignments: &[usize],
    centroids: &[Vec<f64>],
) -> (Vec<Vec<f64>>, f64) {
    let dimension = vectors.iter().map(Vec::len).max().unwrap_or(0);
    let mut sums = vec![vec![0.0_f64; dimension]; centroids.len()];
    let mut counts = vec![0_usize; centroids.len()];
    for (position, cluster) in assignments.iter().copied().enumerate() {
        if let Some(sum) = sums.get_mut(cluster) {
            for (accumulator, value) in sum.iter_mut().zip(&vectors[position]) {
                *accumulator += value;
            }
            counts[cluster] += 1;
        }
    }

    let mut next: Vec<Vec<f64>> = Vec::with_capacity(centroids.len());
    for (cluster, sum) in sums.iter().enumerate() {
        if counts[cluster] == 0 {
            next.push(centroids[cluster].clone());
            continue;
        }
        let scale = 1.0 / counts[cluster] as f64;
        let mean: Vec<f64> = sum.iter().map(|value| value * scale).collect();
        next.push(unit_or(&mean, &centroids[cluster]));
    }
    reseed_empty(vectors, assignments, &mut counts, &mut next, centroids);

    let shift = centroids
        .iter()
        .zip(&next)
        .map(|(old, new)| squared_distance(old, new).sqrt())
        .fold(0.0_f64, f64::max);
    (next, shift)
}

/// Gives every empty cluster a point stolen from a cluster that still has more than one.
///
/// The farthest such point (ties by trace id) becomes the cluster's centroid — the
/// deterministic analogue of the usual "reseed from the worst-fitting point" repair. When
/// every cluster is a singleton the empty cluster keeps its previous centroid and is dropped
/// when the lists are built.
fn reseed_empty(
    vectors: &[Vec<f64>],
    assignments: &[usize],
    counts: &mut [usize],
    centroids: &mut [Vec<f64>],
    previous: &[Vec<f64>],
) {
    let mut used = vec![false; vectors.len()];
    for cluster in 0..centroids.len() {
        if counts[cluster] > 0 {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        for (position, vector) in vectors.iter().enumerate() {
            if used[position] {
                continue;
            }
            let source = assignments[position];
            if counts.get(source).is_none_or(|count| *count <= 1) {
                continue;
            }
            let distance = squared_distance(vector, &previous[source]);
            if best.is_none_or(|(_, best_distance)| distance > best_distance) {
                best = Some((position, distance));
            }
        }
        let Some((position, _)) = best else {
            continue;
        };
        used[position] = true;
        counts[assignments[position]] -= 1;
        centroids[cluster] = vectors[position].clone();
    }
}

/// Merges cluster lists whose centroids reach `threshold` cosine similarity.
///
/// Pairs are visited in ascending list-index order and the higher index is absorbed into the
/// lower one, so the merge order never depends on iteration order. Passes repeat until no
/// pair qualifies, because absorbing a list moves the survivor's centroid.
fn merge_lists(
    lists: Vec<Vec<usize>>,
    points: &[QueryPoint<'_>],
    threshold: f64,
) -> Vec<Vec<usize>> {
    let mut lists = lists;
    loop {
        let mut merged = false;
        let mut index = 0;
        while index < lists.len() {
            let mut other = index + 1;
            while other < lists.len() {
                let left = mean_vector(points, &lists[index]);
                let right = mean_vector(points, &lists[other]);
                if cosine_similarity(&left, &right) + MERGE_TOLERANCE >= threshold {
                    let donor = lists.remove(other);
                    lists[index].extend(donor);
                    lists[index].sort_unstable();
                    merged = true;
                } else {
                    other += 1;
                }
            }
            index += 1;
        }
        if !merged {
            return lists;
        }
    }
}

/// Builds the public cluster record: pattern, members, examples, and success count.
fn cluster_record(skill_ids: &[String], points: &[QueryPoint<'_>], list: &[usize]) -> ClusteredQuery {
    let mut member_ids: Vec<String> = list
        .iter()
        .map(|index| points[*index].record.trace_id.clone())
        .collect();
    member_ids.sort();
    member_ids.dedup();

    let mut query_examples: Vec<String> = Vec::new();
    for index in list {
        let query = points[*index].record.query.trim();
        if !query.is_empty() && !query_examples.iter().any(|known| known == query) {
            query_examples.push(query.to_string());
        }
    }

    let success_count = u32::try_from(
        list.iter()
            .filter(|index| points[**index].record.success)
            .count(),
    )
    .unwrap_or(u32::MAX);
    ClusteredQuery {
        pattern: centroid_query(points, list),
        members: member_ids,
        query_examples,
        skill_ids: skill_ids.to_vec(),
        success_count,
    }
}

/// Picks the member query with the highest cosine similarity to the cluster's mean vector.
///
/// Ties keep the earliest trace id. Blank member queries are only considered when every
/// member is blank, so an empty example never wins over a usable one.
fn centroid_query(points: &[QueryPoint<'_>], list: &[usize]) -> String {
    let mean = mean_vector(points, list);
    let has_text = list
        .iter()
        .any(|index| !points[*index].record.query.trim().is_empty());
    let mut best: Option<(usize, f64)> = None;
    for index in list {
        if has_text && points[*index].record.query.trim().is_empty() {
            continue;
        }
        let similarity = cosine_similarity(&points[*index].vector, &mean);
        if best.is_none_or(|(_, best_similarity)| similarity > best_similarity) {
            best = Some((*index, similarity));
        }
    }
    best.map_or_else(String::new, |(index, _)| {
        points[index].record.query.trim().to_string()
    })
}

/// The L2-normalised mean of the member vectors, or the zero vector when they cancel out.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn mean_vector(points: &[QueryPoint<'_>], list: &[usize]) -> Vec<f32> {
    if list.is_empty() {
        return Vec::new();
    }
    let dimension = list
        .iter()
        .map(|index| points[*index].vector.len())
        .max()
        .unwrap_or(0);
    let mut mean = vec![0.0_f64; dimension];
    for index in list {
        for (accumulator, value) in mean.iter_mut().zip(&points[*index].vector) {
            *accumulator += f64::from(*value);
        }
    }
    let scale = 1.0 / list.len() as f64;
    let averaged: Vec<f32> = mean.iter().map(|value| (value * scale) as f32).collect();
    normalized(averaged)
}

/// The squared euclidean distance of two vectors, tolerating a differing length.
fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

/// The unit vector of `vector`, or `fallback` when `vector` carries no direction.
fn unit_or(vector: &[f64], fallback: &[f64]) -> Vec<f64> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        vector.iter().map(|value| value / norm).collect()
    } else {
        fallback.to_vec()
    }
}

/// The L2-normalised form of `vector`; a zero vector stays zero.
#[allow(clippy::cast_possible_truncation)]
fn normalized(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value = (f64::from(*value) / norm) as f32;
        }
    }
    vector
}

/// Whether `vector` carries a direction, i.e. at least one non-zero component.
#[allow(clippy::float_cmp)]
fn has_direction(vector: &[f32]) -> bool {
    vector.iter().any(|value| *value != 0.0)
}

/// Widens an `f32` vector to `f64` for the distance arithmetic.
fn to_f64(vector: &[f32]) -> Vec<f64> {
    vector.iter().map(|value| f64::from(*value)).collect()
}

/// A waker that does nothing, used by [`poll_ready`].
struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

/// Polls `future` exactly once and answers its output when it is ready.
///
/// Only futures that never await real I/O may use this: the caller has no runtime to drive
/// them and a wake does nothing here.
fn poll_ready<F: Future>(future: F) -> Option<F::Output> {
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(output) => Some(output),
        Poll::Pending => None,
    }
}

/// The sorted, deduped, non-blank skill ids of `record`.
fn skill_key(record: &TraceRecord) -> Vec<String> {
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
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

    /// An embedder answering a fixed vector per query, and an empty vector otherwise.
    struct ScriptedEmbedding {
        vectors: HashMap<String, Vec<f32>>,
    }

    impl ScriptedEmbedding {
        fn new(entries: &[(&str, [f32; 3])]) -> Self {
            Self {
                vectors: entries
                    .iter()
                    .map(|(query, vector)| ((*query).to_string(), vector.to_vec()))
                    .collect(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Embedding for ScriptedEmbedding {
        fn model_name(&self) -> &str {
            "scripted"
        }

        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.vectors.get(text).cloned().unwrap_or_default())
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| self.vectors.get(text).cloned().unwrap_or_default())
                .collect())
        }
    }

    #[test]
    fn target_clusters_follows_the_documented_heuristic() {
        let config = ClusterConfig::default();
        assert_eq!(target_clusters(0, &config), 0);
        assert_eq!(target_clusters(1, &config), 1);
        assert_eq!(target_clusters(2, &config), 1);
        assert_eq!(target_clusters(8, &config), 2);
        assert_eq!(target_clusters(9, &config), 3);
        assert_eq!(target_clusters(64, &config), DEFAULT_MAX_CLUSTERS);

        let bounded = ClusterConfig {
            max_clusters: 2,
            min_clusters: 1,
            ..ClusterConfig::default()
        };
        assert_eq!(target_clusters(64, &bounded), 2);

        let forced = ClusterConfig {
            max_clusters: 4,
            min_clusters: 3,
            ..ClusterConfig::default()
        };
        assert_eq!(target_clusters(4, &forced), 3);
    }

    #[test]
    fn config_validation_names_the_offending_field() {
        let assert_invalid = |config: ClusterConfig, field: &str| {
            let error = config.validate().unwrap_err();
            assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
            assert!(error.message().starts_with(field), "{error}");
        };
        assert_invalid(
            ClusterConfig {
                max_clusters: 0,
                ..ClusterConfig::default()
            },
            "max_clusters",
        );
        assert_invalid(
            ClusterConfig {
                min_clusters: 0,
                ..ClusterConfig::default()
            },
            "min_clusters",
        );
        assert_invalid(
            ClusterConfig {
                max_clusters: 1,
                min_clusters: 2,
                ..ClusterConfig::default()
            },
            "min_clusters",
        );
        assert_invalid(
            ClusterConfig {
                max_iterations: 0,
                ..ClusterConfig::default()
            },
            "max_iterations",
        );
        assert_invalid(
            ClusterConfig {
                epsilon: -1.0,
                ..ClusterConfig::default()
            },
            "epsilon",
        );
        assert_invalid(
            ClusterConfig {
                centroid_merge_threshold: Some(2.0),
                ..ClusterConfig::default()
            },
            "centroid_merge_threshold",
        );
        assert!(ClusterConfig::default().validate().is_ok());
    }

    #[test]
    fn blank_queries_join_the_first_list() {
        let records = vec![
            record("t1", "  ", &["a"], true),
            record("t2", "deploy the cluster", &["a"], true),
            record("t3", "deploy the cluster now", &["a"], true),
        ];
        let clusters = cluster_traces(&records);
        // n = 3 gives k = 2, and the blank-query member shares the first list.
        assert_eq!(clusters.len(), 2);
        assert!(clusters
            .iter()
            .any(|cluster| cluster.members.contains(&"t1".to_string())));
        for cluster in &clusters {
            assert!(!cluster.query_examples.iter().any(String::is_empty));
        }
    }

    #[test]
    fn zero_vectors_stay_in_one_list() {
        let records = vec![
            record("t1", "   ", &["a"], true),
            record("t2", "", &["a"], true),
        ];
        let clusters = cluster_traces(&records);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members, vec!["t1", "t2"]);
        assert!(clusters[0].query_examples.is_empty());
        assert_eq!(clusters[0].pattern, "");
    }

    #[tokio::test]
    async fn scripted_centroids_are_distinct_and_stable() {
        let records = vec![
            record("t1", "alpha", &["a"], true),
            record("t2", "alpha", &["a"], true),
            record("t3", "alpha", &["a"], false),
            record("t4", "beta", &["a"], true),
            record("t5", "beta", &["a"], true),
            record("t6", "beta", &["a"], true),
        ];
        let embedder = ScriptedEmbedding::new(&[
            ("alpha", [1.0, 0.0, 0.0]),
            ("beta", [0.0, 0.9, 0.435]),
        ]);
        let config = ClusterConfig::default();
        let clusters = cluster_traces_with_embedding(&records, &embedder, &config)
            .await
            .unwrap();
        // n = 6 gives k = ceil(sqrt(3)) = 2.
        assert_eq!(clusters.len(), 2);
        let alpha = clusters
            .iter()
            .find(|cluster| cluster.members.contains(&"t1".to_string()))
            .unwrap();
        assert_eq!(alpha.members, vec!["t1", "t2", "t3"]);
        assert_eq!(alpha.pattern, "alpha");
        assert_eq!(alpha.query_examples, vec!["alpha"]);
        assert_eq!(alpha.success_count, 2);
        let beta = clusters
            .iter()
            .find(|cluster| cluster.members.contains(&"t4".to_string()))
            .unwrap();
        assert_eq!(beta.members, vec!["t4", "t5", "t6"]);
        assert_eq!(beta.pattern, "beta");
        assert_eq!(beta.success_count, 3);

        // Every input trace appears in exactly one cluster, and repeats are byte-stable.
        let mut seen: Vec<&str> = clusters
            .iter()
            .flat_map(|cluster| cluster.members.iter().map(String::as_str))
            .collect();
        seen.sort();
        assert_eq!(seen, vec!["t1", "t2", "t3", "t4", "t5", "t6"]);
        assert_eq!(
            clusters,
            cluster_traces_with_embedding(&records, &embedder, &config)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn max_clusters_one_collapses_a_skill_group() {
        let records = vec![
            record("t1", "alpha", &["a"], true),
            record("t2", "beta", &["a"], true),
        ];
        let embedder = ScriptedEmbedding::new(&[("alpha", [1.0, 0.0, 0.0]), ("beta", [0.0, 1.0, 0.0])]);
        let collapsed = ClusterConfig {
            max_clusters: 1,
            ..ClusterConfig::default()
        };
        let clusters = cluster_traces_with_embedding(&records, &embedder, &collapsed)
            .await
            .unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members, vec!["t1", "t2"]);
        // The mean of the two orthogonal vectors is their bisector, so the first member wins.
        assert_eq!(clusters[0].pattern, "alpha");
    }

    #[tokio::test]
    async fn centroid_merge_is_opt_in() {
        let records = vec![
            record("t1", "near one", &["a"], true),
            record("t2", "near one", &["a"], true),
            record("t3", "near two", &["a"], true),
            record("t4", "near two", &["a"], true),
        ];
        let embedder = ScriptedEmbedding::new(&[
            ("near one", [1.0, 0.01, 0.0]),
            ("near two", [1.0, 0.02, 0.0]),
        ]);
        let config = ClusterConfig::default();
        assert_eq!(
            cluster_traces_with_embedding(&records, &embedder, &config)
                .await
                .unwrap()
                .len(),
            2
        );

        let exact = ClusterConfig {
            centroid_merge_threshold: Some(1.0),
            ..ClusterConfig::default()
        };
        let unmerged = cluster_traces_with_embedding(&records, &embedder, &exact)
            .await
            .unwrap();
        assert_eq!(unmerged.len(), 2);

        let merged = ClusterConfig {
            centroid_merge_threshold: Some(0.99),
            ..ClusterConfig::default()
        };
        let clusters = cluster_traces_with_embedding(&records, &embedder, &merged)
            .await
            .unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members, vec!["t1", "t2", "t3", "t4"]);
    }

    #[test]
    fn identical_centroids_merge_at_threshold_one() {
        let records = vec![
            record("t1", "same", &["a"], true),
            record("t2", "same", &["a"], true),
        ];
        let points: Vec<QueryPoint<'_>> = records
            .iter()
            .map(|record| QueryPoint {
                record,
                vector: vec![1.0, 0.0],
            })
            .collect();
        let merged = merge_lists(vec![vec![0], vec![1]], &points, 1.0);
        assert_eq!(merged, vec![vec![0, 1]]);
        // Two orthogonal lists never merge, whatever the threshold.
        let orthogonal = vec![
            QueryPoint {
                record: &records[0],
                vector: vec![1.0, 0.0],
            },
            QueryPoint {
                record: &records[1],
                vector: vec![0.0, 1.0],
            },
        ];
        assert_eq!(
            merge_lists(vec![vec![0], vec![1]], &orthogonal, -1.0).len(),
            1
        );
        assert_eq!(
            merge_lists(vec![vec![0], vec![1]], &orthogonal, 0.5).len(),
            2
        );
    }

    #[tokio::test]
    async fn sync_path_matches_the_async_hash_embedding_path() {
        let records = vec![
            record("t1", "deploy the kubernetes cluster", &["ops"], true),
            record("t2", "deploy the kubernetes cluster now", &["ops"], true),
            record("t3", "bake a chocolate cake", &["ops"], false),
            record("t4", "summarize the report", &["summary"], true),
            record("t5", "   ", &["summary"], true),
        ];
        let clusters = cluster_traces(&records);
        let expected = cluster_traces_with_embedding(
            &records,
            &HashEmbedding::default(),
            &ClusterConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(clusters, expected);
    }

    #[tokio::test]
    async fn every_skill_group_is_clustered_independently() {
        let records = vec![
            record("t1", "alpha", &["a"], true),
            record("t2", "beta", &["a"], true),
            record("t3", "alpha", &["b"], true),
        ];
        let embedder =
            ScriptedEmbedding::new(&[("alpha", [1.0, 0.0, 0.0]), ("beta", [0.0, 1.0, 0.0])]);
        let clusters = cluster_traces_with_embedding(&records, &embedder, &ClusterConfig::default())
            .await
            .unwrap();
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].skill_ids, vec!["a"]);
        assert_eq!(clusters[0].members, vec!["t1", "t2"]);
        assert_eq!(clusters[1].skill_ids, vec!["b"]);
        assert_eq!(clusters[1].members, vec!["t3"]);
    }

    #[tokio::test]
    async fn an_invalid_config_is_rejected() {
        let records = vec![record("t1", "alpha", &["a"], true)];
        let embedder = ScriptedEmbedding::new(&[("alpha", [1.0, 0.0, 0.0])]);
        let config = ClusterConfig {
            max_iterations: 0,
            ..ClusterConfig::default()
        };
        let error = cluster_traces_with_embedding(&records, &embedder, &config)
            .await
            .unwrap_err();
        assert!(error.message().starts_with("max_iterations"), "{error}");
    }
}
