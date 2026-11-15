//! Embedding-backed k-means clustering of items into tree branches.
//!
//! Ports the clustering half of Python `openjiuwen.symphony.retrieval.build.tree.grouping`
//! (its `faiss.Kmeans` call and the `n_clusters` derivation next to it) onto
//! `openjiuwen-retrieval`'s [`Embedding`](openjiuwen_retrieval::Embedding) trait, so
//! [`GroupingStrategy::Kmeans`] and [`GroupingStrategy::FaissKmeans`] group items without
//! faiss, numpy or a completion model.
//!
//! # The embedded text
//!
//! [`embedding_text`](crate::retrieval::build::grouping::kmeans::embedding_text) builds one
//! document per item from five parts in this order: the `name` truncated to `NAME_LIMIT`
//! characters, the `description` truncated to `DESCRIPTION_LIMIT`, the `classification`
//! truncated to `NAME_LIMIT`, the first `TAG_LIMIT` non-blank tags joined with `", "`, and
//! the `content` truncated to `CONTENT_LIMIT`. Truncation is by `char` via `truncate_text`,
//! so multi-byte text stays intact. Blank parts are dropped and the rest are joined with
//! `"\n"`; when every part is blank the item's `worker_id` is embedded instead, because an
//! embedder rejects a blank input (`HashEmbedding` answers the zero vector, `OpenAiEmbedding`
//! an [`Error::Validation`](crate::error::Error)).
//!
//! # The cluster count
//!
//! [`resolve_clusters`](crate::retrieval::build::grouping::kmeans::resolve_clusters) derives
//! `k` from [`GroupingConfig::clusters`]: `None` means `ceil(item_count / max_children)`
//! (both bounds are normalized to at least `1` by the parent module's `Limits`
//! normalization), and every value — derived or explicit — is clamped into
//! `1..=min(item_count, max_children)`. `k` therefore never exceeds the branch fan-out, so
//! the caller's shared rebalancing pass has nothing to fold for these branches.
//!
//! # Initialisation, iteration and convergence
//!
//! Vectors are L2-normalised once (a zero-norm vector stays all zeros) and compared by
//! squared euclidean distance accumulated in `f64`, which is `d² = 2 - 2·cos` for unit
//! vectors — the same ordering as cosine similarity.
//!
//! The seeding is k-means++-style but seeded by a SHA-256 digest, so there is no RNG: the
//! first centroid is the item at index `hash % n`, where `hash` is the first eight digest
//! bytes (big-endian `u64`) of the `"\n"`-joined, `worker_id`-sorted item ids; every next
//! centroid is the item with the largest minimum squared distance to the already chosen
//! centroids (ties break on the lowest index).
//!
//! Lloyd's algorithm then runs at most `MAX_ITERATIONS` rounds. A round assigns every item to
//! its nearest centroid (ties break on the lowest centroid index), recomputes each centroid
//! as the normalised mean of its members' normalised vectors, reseeds every empty cluster
//! with the item farthest from its own centroid among items whose cluster holds more than one
//! member (ties break on the lowest index; with no such item the loop stops), and stops once
//! the summed squared movement of all centroids is at most `CONVERGENCE_EPSILON` as `f64`.
//! One final assignment pass runs after the loop, so the returned assignments always belong
//! to the returned centroids.
//!
//! # Labels
//!
//! A cluster is labelled by its members' dominant `leading_name_token` — the most frequent
//! token, ties broken by the lexicographically smallest one — slugged with [`slug_term`]. A
//! cluster whose members all yield no token is `cluster-{rank}`, where the rank is the
//! 1-based position of the cluster's smallest `worker_id` among the non-empty clusters.
//! Collisions are disambiguated with `-2`, `-3`, … (a suffixed label that is itself taken is
//! skipped too), so the label set is stable and unique.
//!
//! Each item's path is the single-element `vec![label]` truncated to `max_depth - 1`
//! segments, so `max_depth == 1` puts every item at the root. Everything here is offline and
//! deterministic: no RNG, no clock, no `HashMap`, and the result is one [`GroupAssignment`]
//! per item ordered by `worker_id`.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_retrieval::Embedding;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::llm::truncate_text;
use crate::retrieval::build::tree_text::slug_term;

use super::{leading_name_token, GroupAssignment, GroupingConfig, Limits, SourceItem};

/// The most Lloyd rounds a cluster run performs.
pub const MAX_ITERATIONS: usize = 32;
/// The summed squared centroid movement below which a run is converged.
pub const CONVERGENCE_EPSILON: f32 = 1e-6;
/// How many characters of an item's name (and classification) are embedded.
pub const NAME_LIMIT: usize = 120;
/// How many characters of an item's description are embedded.
pub const DESCRIPTION_LIMIT: usize = 240;
/// How many characters of an item's content are embedded.
pub const CONTENT_LIMIT: usize = 240;
/// How many of an item's tags are embedded.
pub const TAG_LIMIT: usize = 8;

/// The embedder text of one item.
///
/// The parts are the name (at most [`NAME_LIMIT`] characters), the description (at most
/// [`DESCRIPTION_LIMIT`]), the classification (at most [`NAME_LIMIT`]), the first
/// [`TAG_LIMIT`] non-blank tags joined with `", "`, and the content (at most
/// [`CONTENT_LIMIT`]). Blank parts are dropped and the rest are joined with `"\n"`; when
/// every part is blank the item's `worker_id` is returned, because an embedder rejects a
/// blank input.
///
/// ```
/// use openjiuwen_symphony::retrieval::build::grouping::kmeans::embedding_text;
/// use openjiuwen_symphony::retrieval::build::scanner::SourceItem;
///
/// let item = SourceItem {
///     worker_id: "w".to_string(),
///     name: "PDF Parser".to_string(),
///     classification: "documents/pdf".to_string(),
///     ..SourceItem::default()
/// };
/// assert_eq!(embedding_text(&item), "PDF Parser\ndocuments/pdf");
/// assert_eq!(
///     embedding_text(&SourceItem {
///         worker_id: "blank".to_string(),
///         ..SourceItem::default()
///     }),
///     "blank"
/// );
/// ```
#[must_use]
pub fn embedding_text(item: &SourceItem) -> String {
    let tags = item
        .tags
        .iter()
        .filter(|tag| !tag.trim().is_empty())
        .take(TAG_LIMIT)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let parts = [
        truncate_text(&item.name, NAME_LIMIT),
        truncate_text(&item.description, DESCRIPTION_LIMIT),
        truncate_text(&item.classification, NAME_LIMIT),
        tags.as_str(),
        truncate_text(&item.content, CONTENT_LIMIT),
    ];
    let joined = parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if joined.trim().is_empty() {
        item.worker_id.clone()
    } else {
        joined
    }
}

/// How many clusters `config` asks for, clamped to what `item_count` can support.
///
/// [`GroupingConfig::clusters`]
/// `None` derives `ceil(item_count / max_children)`; both bounds are normalized to at least
/// `1` first, so a `max_children` of `0` behaves like `1`. Every value — derived or explicit
/// — is then clamped into `1..=min(item_count, max_children)`; an `item_count` of `0`
/// answers `1` so the derivation never returns zero.
///
/// ```
/// use openjiuwen_symphony::retrieval::build::grouping::kmeans::resolve_clusters;
/// use openjiuwen_symphony::retrieval::build::grouping::GroupingConfig;
///
/// let config = GroupingConfig::default().with_max_children(4);
/// assert_eq!(resolve_clusters(&config, 9), 3);
/// let explicit = config.clone().with_clusters(Some(99));
/// assert_eq!(resolve_clusters(&explicit, 9), 4);
/// let zeroed = config.with_clusters(Some(0));
/// assert_eq!(resolve_clusters(&zeroed, 9), 1);
/// ```
#[must_use]
pub fn resolve_clusters(config: &GroupingConfig, item_count: usize) -> usize {
    let limits = Limits::from_config(config);
    let derived = config
        .clusters
        .unwrap_or_else(|| item_count.div_ceil(limits.max_children));
    derived.clamp(1, limits.max_children).min(item_count.max(1))
}

/// Groups `items` into at most `resolve_clusters` embedded clusters.
///
/// The result holds one [`GroupAssignment`] per item, ordered by `worker_id`; each path is
/// the cluster label as a single segment, truncated to `max_depth - 1` segments.
///
/// # Errors
///
/// Returns the embedder's own error, and
/// [`Error::retrieval`](crate::error::Error) when the embedder answers a number of vectors
/// that differs from the number of items.
pub async fn cluster_paths(
    items: &[SourceItem],
    config: &GroupingConfig,
    embedding: &dyn Embedding,
) -> Result<Vec<GroupAssignment>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let mut ordered: Vec<&SourceItem> = items.iter().collect();
    ordered.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
    let texts: Vec<String> = ordered.iter().map(|item| embedding_text(item)).collect();
    let vectors = embedding.embed_documents(&texts, None).await?;
    if vectors.len() != ordered.len() {
        return Err(Error::retrieval(format!(
            "k-means grouping: the embedding model returned {} vectors for {} items",
            vectors.len(),
            ordered.len()
        )));
    }
    let points: Vec<Vec<f32>> = vectors.into_iter().map(normalize).collect();
    let clusters = resolve_clusters(config, ordered.len());
    let centroids = seed_centroids(&points, &ordered, clusters);
    let (assignments, _centroids) = lloyd(&points, centroids);
    let labels = cluster_labels(&ordered, &assignments, clusters);
    let depth = Limits::from_config(config).max_depth.saturating_sub(1);
    Ok(ordered
        .iter()
        .enumerate()
        .map(|(index, item)| GroupAssignment {
            worker_id: item.worker_id.clone(),
            path: match labels
                .get(assignments.get(index).copied().unwrap_or_default())
                .and_then(|label| label.as_deref())
            {
                Some(label) if depth > 0 => vec![label.to_string()],
                _ => Vec::new(),
            },
        })
        .collect())
}

/// Scales `vector` to unit length, leaving a zero vector at zero.
fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for component in &mut vector {
            *component /= norm;
        }
    }
    vector
}

/// The squared euclidean distance between two vectors, accumulated in `f64`.
///
/// A shorter vector is treated as zero-padded, so an embedder that answers mixed dimensions
/// still produces a distance rather than a panic.
fn squared_distance(left: &[f32], right: &[f32]) -> f64 {
    let length = left.len().max(right.len());
    (0..length)
        .map(|index| {
            let left = f64::from(left.get(index).copied().unwrap_or(0.0));
            let right = f64::from(right.get(index).copied().unwrap_or(0.0));
            (left - right) * (left - right)
        })
        .sum()
}

/// The summed squared movement between two centroid sets.
fn squared_movement(previous: &[Vec<f32>], current: &[Vec<f32>]) -> f64 {
    previous
        .iter()
        .zip(current)
        .map(|(left, right)| squared_distance(left, right))
        .sum()
}

/// The deterministic seed index derived from the sorted item ids.
fn seed_index(items: &[&SourceItem]) -> usize {
    let joined = items
        .iter()
        .map(|item| item.worker_id.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let digest = Sha256::digest(joined.as_bytes());
    let mut bytes = [0_u8; 8];
    if let Some(head) = digest.get(..bytes.len()) {
        bytes.copy_from_slice(head);
    }
    usize::try_from(u64::from_be_bytes(bytes)).unwrap_or(usize::MAX)
}

/// Chooses the starting centroids with a hash-seeded farthest-point walk.
fn seed_centroids(points: &[Vec<f32>], items: &[&SourceItem], clusters: usize) -> Vec<Vec<f32>> {
    if points.is_empty() {
        return Vec::new();
    }
    let first = seed_index(items) % points.len();
    let mut centroids = vec![points.get(first).cloned().unwrap_or_default()];
    while centroids.len() < clusters.min(points.len()) {
        let mut best = 0_usize;
        let mut best_distance = f64::NEG_INFINITY;
        for (index, point) in points.iter().enumerate() {
            let distance = centroids
                .iter()
                .map(|centroid| squared_distance(point, centroid))
                .fold(f64::INFINITY, f64::min);
            if distance > best_distance {
                best_distance = distance;
                best = index;
            }
        }
        centroids.push(points.get(best).cloned().unwrap_or_default());
    }
    centroids
}

/// The nearest centroid index of `point`; ties break on the lowest index.
fn nearest(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0_usize;
    let mut best_distance = f64::INFINITY;
    for (index, centroid) in centroids.iter().enumerate() {
        let distance = squared_distance(point, centroid);
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    best
}

/// Assigns every point to its nearest centroid.
fn nearest_assignments(points: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    points
        .iter()
        .map(|point| nearest(point, centroids))
        .collect()
}

/// Counts the members of every cluster.
fn cluster_counts(assignments: &[usize], clusters: usize) -> Vec<usize> {
    let mut counts = vec![0_usize; clusters];
    for cluster in assignments {
        if let Some(count) = counts.get_mut(*cluster) {
            *count += 1;
        }
    }
    counts
}

/// Multiplies `sum` by `1 / count`, the mean of the members it accumulates.
#[allow(clippy::cast_precision_loss)] // a cluster never holds more than 2^24 items here
fn mean(mut sum: Vec<f32>, count: usize) -> Vec<f32> {
    let scale = 1.0 / count as f32;
    for component in &mut sum {
        *component *= scale;
    }
    sum
}

/// Recomputes every centroid as the normalised mean of its members, keeping the previous
/// value for an empty cluster (which is reseeded immediately afterwards).
fn recompute_centroids(
    points: &[Vec<f32>],
    assignments: &[usize],
    previous: &[Vec<f32>],
    counts: &[usize],
) -> Vec<Vec<f32>> {
    let dimension = previous.first().map_or(0, Vec::len);
    let mut sums = vec![vec![0.0_f32; dimension]; previous.len()];
    for (index, point) in points.iter().enumerate() {
        let Some(cluster) = assignments.get(index).copied() else {
            continue;
        };
        let Some(sum) = sums.get_mut(cluster) else {
            continue;
        };
        for (slot, value) in sum.iter_mut().zip(point) {
            *slot += value;
        }
    }
    sums.into_iter()
        .zip(previous)
        .enumerate()
        .map(
            |(cluster, (sum, kept))| match counts.get(cluster).copied() {
                Some(count) if count > 0 => normalize(mean(sum, count)),
                _ => kept.clone(),
            },
        )
        .collect()
}

/// The item farthest from its own centroid whose cluster holds more than one member.
///
/// Ties break on the lowest index, and `None` means no item can be moved without emptying
/// another cluster.
fn farthest_shared_item(
    points: &[Vec<f32>],
    assignments: &[usize],
    centroids: &[Vec<f32>],
    counts: &[usize],
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (index, point) in points.iter().enumerate() {
        let cluster = assignments.get(index).copied().unwrap_or_default();
        if counts.get(cluster).copied().unwrap_or(0) <= 1 {
            continue;
        }
        let Some(centroid) = centroids.get(cluster) else {
            continue;
        };
        let distance = squared_distance(point, centroid);
        if best.is_none_or(|(_, best_distance)| distance > best_distance) {
            best = Some((index, distance));
        }
    }
    best.map(|(index, _)| index)
}

/// Seeds every empty cluster with a moved item; `false` means no move was possible.
fn reseed_empty(
    points: &[Vec<f32>],
    assignments: &mut [usize],
    centroids: &mut [Vec<f32>],
    counts: &mut [usize],
) -> bool {
    for cluster in 0..centroids.len() {
        if counts.get(cluster).copied().unwrap_or(0) > 0 {
            continue;
        }
        let Some(index) = farthest_shared_item(points, assignments, centroids, counts) else {
            return false;
        };
        let origin = assignments.get(index).copied().unwrap_or_default();
        if let Some(slot) = assignments.get_mut(index) {
            *slot = cluster;
        }
        if let Some(count) = counts.get_mut(origin) {
            *count = count.saturating_sub(1);
        }
        if let Some(count) = counts.get_mut(cluster) {
            *count += 1;
        }
        if let Some(slot) = centroids.get_mut(cluster) {
            if let Some(point) = points.get(index) {
                slot.clone_from(point);
            }
        }
    }
    true
}

/// Runs Lloyd's algorithm, returning the final assignments and centroids.
fn lloyd(points: &[Vec<f32>], mut centroids: Vec<Vec<f32>>) -> (Vec<usize>, Vec<Vec<f32>>) {
    let mut assignments = nearest_assignments(points, &centroids);
    for _ in 0..MAX_ITERATIONS {
        let previous = centroids.clone();
        let mut counts = cluster_counts(&assignments, centroids.len());
        centroids = recompute_centroids(points, &assignments, &previous, &counts);
        if !reseed_empty(points, &mut assignments, &mut centroids, &mut counts) {
            break;
        }
        let movement = squared_movement(&previous, &centroids);
        assignments = nearest_assignments(points, &centroids);
        if movement <= f64::from(CONVERGENCE_EPSILON) {
            break;
        }
    }
    (assignments, centroids)
}

/// The label of every non-empty cluster, indexed by cluster.
fn cluster_labels(
    items: &[&SourceItem],
    assignments: &[usize],
    clusters: usize,
) -> Vec<Option<String>> {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); clusters];
    for (index, cluster) in assignments.iter().enumerate() {
        if let Some(slot) = members.get_mut(*cluster) {
            slot.push(index);
        }
    }
    let mut order: Vec<usize> = (0..clusters)
        .filter(|cluster| {
            members
                .get(*cluster)
                .is_some_and(|cluster_members| !cluster_members.is_empty())
        })
        .collect();
    // Member indices ascend with `worker_id`, so the first member is the smallest id.
    order.sort_by_key(|cluster| {
        members
            .get(*cluster)
            .and_then(|cluster_members| cluster_members.first().copied())
            .unwrap_or(usize::MAX)
    });
    let mut used = BTreeSet::new();
    let mut labels = vec![None; clusters];
    for (rank, cluster) in order.into_iter().enumerate() {
        let base = members
            .get(cluster)
            .and_then(|cluster_members| dominant_token(items, cluster_members))
            .map_or_else(
                || format!("cluster-{}", rank + 1),
                |token| slug_term(&token, "cluster"),
            );
        if let Some(slot) = labels.get_mut(cluster) {
            *slot = Some(unique_label(base, &mut used));
        }
    }
    labels
}

/// The most frequent leading name token among `members`, ties broken lexicographically.
fn dominant_token(items: &[&SourceItem], members: &[usize]) -> Option<String> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for index in members {
        let Some(item) = items.get(*index) else {
            continue;
        };
        if let Some(token) = leading_name_token(&item.name) {
            *counts.entry(token).or_default() += 1;
        }
    }
    let mut best: Option<(String, usize)> = None;
    for (token, count) in counts {
        if best
            .as_ref()
            .is_none_or(|(_, best_count)| count > *best_count)
        {
            best = Some((token, count));
        }
    }
    best.map(|(token, _)| token)
}

/// Returns `base`, suffixing `-2`, `-3`, … until the label is unused.
fn unique_label(base: String, used: &mut BTreeSet<String>) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    let mut index = 2_usize;
    loop {
        let attempt = format!("{base}-{index}");
        if used.insert(attempt.clone()) {
            return attempt;
        }
        index += 1;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_retrieval::HashEmbedding;

    use super::*;
    use crate::retrieval::types::RetrieverItem;

    /// Builds an item with a name and body, leaving every other field at its default.
    fn item(worker_id: &str, name: &str, content: &str) -> SourceItem {
        SourceItem {
            worker_id: worker_id.to_string(),
            item: RetrieverItem::new(worker_id, worker_id),
            name: name.to_string(),
            content: content.to_string(),
            ..SourceItem::default()
        }
    }

    /// Two groups of two items with disjoint vocabularies.
    fn disjoint_items() -> Vec<SourceItem> {
        vec![
            item("a", "PDF Document Parser", "pdf document parser"),
            item("b", "PDF Document Reader", "pdf document reader"),
            item("c", "SQL Database Index", "sql database index"),
            item("d", "SQL Database Query", "sql database query"),
        ]
    }

    #[test]
    fn embedding_text_joins_the_parts_in_order() {
        let item = SourceItem {
            worker_id: "w".to_string(),
            name: "PDF Parser".to_string(),
            description: "parses documents".to_string(),
            classification: "documents/pdf".to_string(),
            tags: vec!["docs".to_string(), "  ".to_string(), "pdf".to_string()],
            content: "body text".to_string(),
            ..SourceItem::default()
        };
        assert_eq!(
            embedding_text(&item),
            "PDF Parser\nparses documents\ndocuments/pdf\ndocs, pdf\nbody text"
        );
    }

    #[test]
    fn embedding_text_truncates_every_part_and_caps_the_tags() {
        let item = SourceItem {
            worker_id: "w".to_string(),
            name: "n".repeat(NAME_LIMIT + 40),
            description: "d".repeat(DESCRIPTION_LIMIT + 40),
            classification: "c".repeat(NAME_LIMIT + 40),
            tags: (0..TAG_LIMIT + 4)
                .map(|index| format!("t{index}"))
                .collect(),
            content: "x".repeat(CONTENT_LIMIT + 40),
            ..SourceItem::default()
        };
        let text = embedding_text(&item);
        let lines: Vec<&str> = text.split('\n').collect();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[0].chars().count(), NAME_LIMIT);
        assert_eq!(lines[1].chars().count(), DESCRIPTION_LIMIT);
        assert_eq!(lines[2].chars().count(), NAME_LIMIT);
        assert_eq!(lines[3], "t0, t1, t2, t3, t4, t5, t6, t7");
        assert_eq!(lines[4].chars().count(), CONTENT_LIMIT);
    }

    #[test]
    fn embedding_text_falls_back_to_the_worker_id() {
        let blank = SourceItem {
            worker_id: "blank-id".to_string(),
            name: "   ".to_string(),
            ..SourceItem::default()
        };
        assert_eq!(embedding_text(&blank), "blank-id");
    }

    #[test]
    fn resolve_clusters_derives_the_ceiling_and_clamps() {
        let config = GroupingConfig::default().with_max_children(4);
        assert_eq!(resolve_clusters(&config, 9), 3);
        assert_eq!(resolve_clusters(&config, 8), 2);
        assert_eq!(resolve_clusters(&config, 4), 1);
        assert_eq!(resolve_clusters(&config, 0), 1);
        for (clusters, expected) in [(0, 1), (1, 1), (3, 3)] {
            let explicit = config.clone().with_clusters(Some(clusters));
            assert_eq!(resolve_clusters(&explicit, 9), expected, "{clusters}");
        }
        // The fan-out bound wins over an explicit count.
        let many = config.clone().with_clusters(Some(999));
        assert_eq!(resolve_clusters(&many, 9), 4);
        // The item count wins over the fan-out bound.
        assert_eq!(resolve_clusters(&many, 2), 2);
        // A zero fan-out behaves like one.
        let zeroed = GroupingConfig::default().with_max_children(0);
        assert_eq!(resolve_clusters(&zeroed, 9), 1);
        let zeroed_explicit = zeroed.with_clusters(Some(999));
        assert_eq!(resolve_clusters(&zeroed_explicit, 9), 1);
    }

    #[tokio::test]
    async fn cluster_paths_returns_nothing_for_no_items() {
        let embedding = HashEmbedding::new(64);
        let assignments = cluster_paths(&[], &GroupingConfig::default(), &embedding)
            .await
            .unwrap();
        assert_eq!(assignments, Vec::new());
    }

    #[tokio::test]
    async fn cluster_paths_is_byte_identical_across_runs() {
        let embedding = HashEmbedding::new(64);
        let items = disjoint_items();
        let config = GroupingConfig::default().with_clusters(Some(2));
        let first = cluster_paths(&items, &config, &embedding).await.unwrap();
        let second = cluster_paths(&items, &config, &embedding).await.unwrap();
        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap()
        );
        let ids: Vec<&str> = first
            .iter()
            .map(|assignment| assignment.worker_id.as_str())
            .collect();
        assert_eq!(ids, vec!["a", "b", "c", "d"]);
        for assignment in &first {
            assert_eq!(assignment.path.len(), 1, "{assignment:?}");
        }
    }

    #[tokio::test]
    async fn cluster_paths_separates_disjoint_vocabularies() {
        let embedding = HashEmbedding::new(64);
        let items = disjoint_items();
        let assignments = cluster_paths(
            &items,
            &GroupingConfig::default().with_clusters(Some(2)),
            &embedding,
        )
        .await
        .unwrap();
        assert_eq!(assignments.len(), 4);
        let label = |worker_id: &str| {
            assignments
                .iter()
                .find(|assignment| assignment.worker_id == worker_id)
                .map(|assignment| assignment.path.clone())
                .unwrap()
        };
        assert_eq!(label("a"), label("b"));
        assert_eq!(label("c"), label("d"));
        assert_ne!(label("a"), label("c"));
    }

    #[tokio::test]
    async fn cluster_paths_labels_follow_the_dominant_name_token() {
        let embedding = HashEmbedding::new(64);
        let items = vec![
            item("a", "Parser Helper", "alpha"),
            item("b", "Parser Tool", "beta"),
        ];
        let assignments = cluster_paths(
            &items,
            &GroupingConfig::default().with_clusters(Some(1)),
            &embedding,
        )
        .await
        .unwrap();
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].path, vec!["parser".to_string()]);
        assert_eq!(assignments[1].path, vec!["parser".to_string()]);
    }

    #[tokio::test]
    async fn cluster_paths_places_items_at_the_root_when_max_depth_is_one() {
        let embedding = HashEmbedding::new(64);
        let items = disjoint_items();
        let assignments = cluster_paths(
            &items,
            &GroupingConfig::default()
                .with_max_depth(1)
                .with_clusters(Some(2)),
            &embedding,
        )
        .await
        .unwrap();
        assert_eq!(assignments.len(), 4);
        for assignment in &assignments {
            assert!(assignment.path.is_empty(), "{assignment:?}");
        }
    }
}
