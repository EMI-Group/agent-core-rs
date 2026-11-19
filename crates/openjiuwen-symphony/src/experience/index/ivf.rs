//! The deterministic IVF coarse quantiser behind the persisted experience index.
//!
//! An IVF index answers a query in two steps: a *coarse quantiser* finds the `nprobe`
//! centroids closest to the query, and only the inverted lists of those centroids are
//! scored exactly. This module owns the first step — training the centroids and assigning
//! every stored vector to one list — and [`CoarseIndex::probe`] owns the query half.
//!
//! # Why a flat k-means instead of faiss
//!
//! faiss trains its coarse quantiser with a randomized k-means that needs numpy's BLAS and
//! faiss' own inverted-list storage. Nothing in an offline build ships either, so this
//! module trains with a small, fully specified k-means that is deterministic by construction:
//!
//! 1. Every vector is L2-normalised (a zero vector stays zero), so squared euclidean
//!    distance orders points exactly like the inner product a cosine search uses. Centroids
//!    are normalised the same way after every update, keeping assignment and probing
//!    consistent with each other.
//! 2. Seeding is k-means++ *deterministic*: the first centroid is the vector of the smallest
//!    id, and each following centroid is the point farthest (squared euclidean) from its
//!    nearest already-chosen centroid, with ties broken by id.
//! 3. Lloyd iterations run at most [`MAX_LLOYD_ITERATIONS`] times (configurable with
//!    [`CoarseConfig`]) and stop once the largest centroid shift is at most
//!    [`LLOYD_EPSILON`]. Assignment breaks distance ties by list index.
//! 4. An empty list is reseeded with the point farthest from its assigned centroid (ties by
//!    id, never the same point twice in one iteration); when no candidate is left the
//!    centroid is kept as it was.
//!
//! No randomness, no clock and no hash-map iteration order reaches an output, so two builds
//! over the same vectors produce byte-identical centroids and lists.
//!
//! # Product quantisation is not applied
//!
//! Naming an index `ivf4096,pq16` records the sub-quantiser count in the canonical name and
//! the manifest, but no vector is compressed: the probed candidates are scored against their
//! exact stored vectors, so probe search is a recall trade-off and never an accuracy one.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::error::{validation, Result};

/// The maximum number of Lloyd iterations a coarse build runs.
pub(crate) const MAX_LLOYD_ITERATIONS: usize = 25;

/// The largest centroid shift (euclidean, on unit vectors) that ends the Lloyd loop.
pub(crate) const LLOYD_EPSILON: f64 = 1e-6;

/// The tuning of a coarse build.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CoarseConfig {
    /// The iteration cap; [`MAX_LLOYD_ITERATIONS`] by default.
    pub(crate) max_iterations: usize,
    /// The convergence threshold; [`LLOYD_EPSILON`] by default.
    pub(crate) epsilon: f64,
}

impl Default for CoarseConfig {
    fn default() -> Self {
        Self {
            max_iterations: MAX_LLOYD_ITERATIONS,
            epsilon: LLOYD_EPSILON,
        }
    }
}

/// The trained coarse quantiser: centroids plus one inverted-list index per item.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CoarseIndex {
    /// The number of coarse lists (`0` for an empty catalog).
    nlist: usize,
    /// The number of components in a vector and in a centroid.
    dimension: usize,
    /// The centroids, one per list, L2-normalised.
    centroids: Vec<Vec<f32>>,
    /// The item ids, aligned with `lists`, in ascending id order.
    ids: Vec<String>,
    /// The list index of each id, aligned with `ids`.
    lists: Vec<usize>,
}

impl CoarseIndex {
    /// The quantiser of an empty catalog: no lists, no centroids, no ids.
    pub(crate) fn empty() -> Self {
        Self {
            nlist: 0,
            dimension: 0,
            centroids: Vec::new(),
            ids: Vec::new(),
            lists: Vec::new(),
        }
    }

    /// Trains a quantiser over `ids` and `vectors` with the default [`CoarseConfig`].
    ///
    /// The inputs must be aligned and `ids` must be unique; `nlist` is clamped to
    /// `1..=ids.len()`, so asking for more lists than vectors simply gives every vector its
    /// own list.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) when the two slices
    /// have different lengths, a vector has no components or a different length than the
    /// first, or the ids are not unique.
    pub(crate) fn build(ids: &[String], vectors: &[Vec<f32>], nlist: usize) -> Result<Self> {
        Self::build_with(ids, vectors, nlist, CoarseConfig::default())
    }

    /// Trains a quantiser with an explicit iteration cap and convergence threshold.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::build`].
    pub(crate) fn build_with(
        ids: &[String],
        vectors: &[Vec<f32>],
        nlist: usize,
        config: CoarseConfig,
    ) -> Result<Self> {
        if ids.len() != vectors.len() {
            return Err(validation(
                "ivf",
                format!(
                    "{} ids do not match the {} vectors",
                    ids.len(),
                    vectors.len()
                ),
            ));
        }
        if ids.is_empty() {
            return Ok(Self::empty());
        }
        let dimension = vectors.first().map_or(0, Vec::len);
        if dimension == 0 {
            return Err(validation("ivf", "the vectors have no components"));
        }
        if let Some((id, row)) = ids
            .iter()
            .zip(vectors)
            .find(|(_, row)| row.len() != dimension)
        {
            return Err(validation(
                "ivf",
                format!(
                    "the vector of `{id}` has {} components but the index dimension is \
                     {dimension}",
                    row.len()
                ),
            ));
        }
        if let Some(duplicate) = first_duplicate(ids) {
            return Err(validation(
                "ivf",
                format!("the id `{duplicate}` appears more than once"),
            ));
        }

        let nlist = nlist.clamp(1, ids.len());
        let points: Vec<Vec<f32>> = vectors.iter().map(|vector| normalize(vector)).collect();
        let centroids = lloyd(&points, ids, seed(&points, ids, nlist), config);
        let lists = assign(&points, &centroids);
        Ok(Self {
            nlist: centroids.len(),
            dimension,
            centroids,
            ids: ids.to_vec(),
            lists,
        })
    }

    /// Rehydrates a persisted quantiser, validating its shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) when the parts are
    /// inconsistent: an empty id list with lists or centroids, an `nlist` that does not match
    /// the centroid count, more lists than ids, an id list that is not unique and ascending, a
    /// list index outside `0..nlist`, or a centroid whose length is not `dimension`.
    pub(crate) fn from_parts(
        dimension: usize,
        nlist: usize,
        ids: Vec<String>,
        centroids: Vec<Vec<f32>>,
        lists: Vec<usize>,
    ) -> Result<Self> {
        if centroids.len() != nlist {
            return Err(validation(
                "ivf",
                format!(
                    "nlist is {nlist} but {} centroids are stored",
                    centroids.len()
                ),
            ));
        }
        if ids.len() != lists.len() {
            return Err(validation(
                "ivf",
                format!(
                    "{} ids do not match the {} list entries",
                    ids.len(),
                    lists.len()
                ),
            ));
        }
        if nlist > ids.len() {
            return Err(validation(
                "ivf",
                format!(
                    "nlist {nlist} is larger than the {} stored vectors",
                    ids.len()
                ),
            ));
        }
        if nlist == 0 && !ids.is_empty() {
            return Err(validation(
                "ivf",
                format!("{} vectors have no coarse list", ids.len()),
            ));
        }
        for (position, centroid) in centroids.iter().enumerate() {
            if centroid.len() != dimension {
                return Err(validation(
                    "ivf",
                    format!(
                        "centroid {position} has {} components but the index dimension is \
                         {dimension}",
                        centroid.len()
                    ),
                ));
            }
        }
        for (id, list) in ids.iter().zip(&lists) {
            if *list >= nlist {
                return Err(validation(
                    "ivf",
                    format!("the list index {list} of `{id}` is outside 0..{nlist}"),
                ));
            }
        }
        let sorted: BTreeSet<&String> = ids.iter().collect();
        let ascending = ids.windows(2).all(|pair| pair[0] < pair[1]);
        if sorted.len() != ids.len() || !ascending {
            return Err(validation(
                "ivf",
                "the stored ids must be unique and sorted ascending",
            ));
        }
        Ok(Self {
            nlist,
            dimension,
            centroids,
            ids,
            lists,
        })
    }

    /// The number of coarse lists.
    pub(crate) fn nlist(&self) -> usize {
        self.nlist
    }

    /// The number of components in a vector and in a centroid.
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    /// The centroids, one per list.
    pub(crate) fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// The stored ids, in ascending order.
    pub(crate) fn ids(&self) -> &[String] {
        &self.ids
    }

    /// The list index of every id, aligned with [`Self::ids`].
    pub(crate) fn lists(&self) -> &[usize] {
        &self.lists
    }

    /// Assigns every stored id to a list without retraining the centroids.
    ///
    /// An id the quantiser already knows keeps its stored list; a new id is placed in the
    /// list of its nearest centroid (the nearest-point rule [`Self::build`] uses, ties broken
    /// by list index). This is what lets a new item join an existing IVF index instead of
    /// rebuilding the whole quantiser: the geometry is unchanged, only the inverted lists
    /// grow.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) when the quantiser has
    /// no coarse lists, the inputs are misaligned, a row is not [`Self::dimension`] long, or
    /// an id the quantiser knows is absent from `ids` (a removed item needs a rebuild, not a
    /// patched list).
    pub(crate) fn assign_all(&self, ids: &[String], vectors: &[Vec<f32>]) -> Result<Self> {
        if self.nlist == 0 {
            return Err(validation(
                "ivf",
                "the coarse quantiser has no lists to assign the vectors to",
            ));
        }
        if ids.len() != vectors.len() {
            return Err(validation(
                "ivf",
                format!(
                    "{} ids do not match the {} vectors",
                    ids.len(),
                    vectors.len()
                ),
            ));
        }
        if let Some((id, row)) = ids
            .iter()
            .zip(vectors)
            .find(|(_, row)| row.len() != self.dimension)
        {
            return Err(validation(
                "ivf",
                format!(
                    "the vector of `{id}` has {} components but the index dimension is {}",
                    row.len(),
                    self.dimension
                ),
            ));
        }
        if let Some(missing) = self.ids.iter().find(|id| ids.binary_search(id).is_err()) {
            return Err(validation(
                "ivf",
                format!(
                    "the coarse quantiser knows `{missing}`, which is no longer stored; the \
                     index has to be rebuilt"
                ),
            ));
        }
        let mut lists = Vec::with_capacity(ids.len());
        for (id, row) in ids.iter().zip(vectors) {
            lists.push(match self.ids.binary_search(id) {
                Ok(position) => self.lists[position],
                Err(_) => nearest_centroid(&normalize(row), &self.centroids),
            });
        }
        Self::from_parts(
            self.dimension,
            self.nlist,
            ids.to_vec(),
            self.centroids.clone(),
            lists,
        )
    }

    /// The ids of the `nprobe` lists nearest to `query`, in probe then id order.
    ///
    /// The query is L2-normalised first, so the inner product against a normalised centroid
    /// is a cosine. `nprobe` is clamped to `1..=nlist`; ties between equally near centroids
    /// break on the lower list index, which keeps the probe order stable. A query whose
    /// length does not match [`Self::dimension`] scores zero against every centroid and
    /// therefore probes the lowest-numbered lists.
    pub(crate) fn probe(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        if self.nlist == 0 {
            return Vec::new();
        }
        let normalised = normalize(query);
        let mut scored: Vec<(usize, f64)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(index, centroid)| (index, inner_product(&normalised, centroid)))
            .collect();
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let probe = nprobe.clamp(1, self.nlist);
        scored
            .into_iter()
            .take(probe)
            .map(|(index, _)| index)
            .collect()
    }

    /// The ids living in the `nprobe` nearest lists, in probe then id order.
    pub(crate) fn probed_ids(&self, query: &[f32], nprobe: usize) -> Vec<&str> {
        let probed = self.probe(query, nprobe);
        let mut ids = Vec::new();
        for list in probed {
            for (id, assigned) in self.ids.iter().zip(&self.lists) {
                if *assigned == list {
                    ids.push(id.as_str());
                }
            }
        }
        ids
    }
}

/// Seeds `nlist` centroids deterministically (k-means++ with an id tie-break).
fn seed(points: &[Vec<f32>], ids: &[String], nlist: usize) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(nlist);
    let mut chosen = BTreeSet::new();
    let first = smallest_id_index(ids);
    centroids.push(points[first].clone());
    chosen.insert(first);
    let mut nearest: Vec<f64> = points
        .iter()
        .map(|point| squared_distance(point, &points[first]))
        .collect();
    while centroids.len() < nlist {
        let Some(index) = farthest_unset(&nearest, ids, &chosen) else {
            break;
        };
        let centroid = points[index].clone();
        for (position, point) in points.iter().enumerate() {
            let distance = squared_distance(point, &centroid);
            if distance < nearest[position] {
                nearest[position] = distance;
            }
        }
        chosen.insert(index);
        centroids.push(centroid);
    }
    centroids
}

/// The index of the smallest id (`0` for the id-sorted input this module documents).
fn smallest_id_index(ids: &[String]) -> usize {
    let mut best = 0;
    for (index, id) in ids.iter().enumerate() {
        if id < &ids[best] {
            best = index;
        }
    }
    best
}

/// The unchosen point farthest from its nearest centroid, ties broken by id.
fn farthest_unset(nearest: &[f64], ids: &[String], chosen: &BTreeSet<usize>) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (index, distance) in nearest.iter().enumerate() {
        if chosen.contains(&index) {
            continue;
        }
        best = Some(match best {
            None => (index, *distance),
            Some((best_index, best_distance)) => {
                // `total_cmp` keeps an exact tie (equal distance) an `Equal`, which the
                // id comparison below breaks deterministically.
                let better = match (*distance).total_cmp(&best_distance) {
                    Ordering::Greater => true,
                    Ordering::Equal => ids[index] < ids[best_index],
                    Ordering::Less => false,
                };
                if better {
                    (index, *distance)
                } else {
                    (best_index, best_distance)
                }
            }
        });
    }
    best.map(|(index, _)| index)
}

/// Runs Lloyd iterations until the cap or the convergence threshold is reached.
fn lloyd(
    points: &[Vec<f32>],
    ids: &[String],
    centroids: Vec<Vec<f32>>,
    config: CoarseConfig,
) -> Vec<Vec<f32>> {
    let mut centroids = centroids;
    let nlist = centroids.len();
    if nlist == 0 || points.is_empty() {
        return centroids;
    }
    let mut assignment = vec![0_usize; points.len()];
    let iterations = config.max_iterations.max(1);
    for _ in 0..iterations {
        assign_into(points, &centroids, &mut assignment);
        let (mut next, counts) = means(points, &centroids, &assignment);
        let shift = largest_shift(&centroids, &next);
        reseed_empty(points, ids, &centroids, &assignment, &mut next, &counts);
        centroids = next;
        if shift <= config.epsilon {
            break;
        }
    }
    centroids
}

/// Assigns every point to its nearest centroid, ties broken by list index.
fn assign(points: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    let mut assignment = vec![0_usize; points.len()];
    assign_into(points, centroids, &mut assignment);
    assignment
}

/// The nearest centroid of one (already normalised) point, ties broken by list index.
fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for (index, centroid) in centroids.iter().enumerate() {
        let distance = squared_distance(point, centroid);
        // Strictly closer wins, so an exact tie keeps the lower list index.
        if distance < best_distance {
            best = index;
            best_distance = distance;
        }
    }
    best
}

/// Fills `assignment` with the nearest centroid of every point.
fn assign_into(points: &[Vec<f32>], centroids: &[Vec<f32>], assignment: &mut [usize]) {
    for (position, point) in points.iter().enumerate() {
        assignment[position] = nearest_centroid(point, centroids);
    }
}

/// The new normalised centroid of every list plus the population of each list.
///
/// An empty list keeps the centroid it had, so the returned `centroids` always has one entry
/// per list.
fn means(
    points: &[Vec<f32>],
    centroids: &[Vec<f32>],
    assignment: &[usize],
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut sums = vec![vec![0.0_f64; centroids.first().map_or(0, Vec::len)]; centroids.len()];
    let mut counts = vec![0_usize; centroids.len()];
    for (position, point) in points.iter().enumerate() {
        let list = assignment[position];
        counts[list] += 1;
        for (accumulator, value) in sums[list].iter_mut().zip(point) {
            *accumulator += f64::from(*value);
        }
    }
    let mut next = Vec::with_capacity(centroids.len());
    for (index, sum) in sums.iter().enumerate() {
        if counts[index] == 0 {
            next.push(centroids[index].clone());
            continue;
        }
        #[allow(clippy::cast_precision_loss)] // populations are far below 2^53
        let population = counts[index] as f64;
        // The catalog stores `f32` vectors, so a mean narrows back to `f32` on purpose.
        #[allow(clippy::cast_possible_truncation)]
        let mean: Vec<f32> = sum
            .iter()
            .map(|component| (component / population) as f32)
            .collect();
        next.push(normalize(&mean));
    }
    (next, counts)
}

/// The largest euclidean shift between the previous and the next centroids.
fn largest_shift(previous: &[Vec<f32>], next: &[Vec<f32>]) -> f64 {
    previous
        .iter()
        .zip(next)
        .map(|(old, new)| squared_distance(old, new).sqrt())
        .fold(0.0_f64, f64::max)
}

/// Gives every empty list the point farthest from its assigned centroid.
///
/// Ties break on id and one point is used at most once per iteration; with no candidate left
/// the empty centroid keeps the value [`means`] gave it.
fn reseed_empty(
    points: &[Vec<f32>],
    ids: &[String],
    previous: &[Vec<f32>],
    assignment: &[usize],
    next: &mut [Vec<f32>],
    counts: &[usize],
) {
    let mut used = BTreeSet::new();
    for list in 0..next.len() {
        if counts[list] > 0 {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        for (position, point) in points.iter().enumerate() {
            if used.contains(&position) {
                continue;
            }
            let distance = squared_distance(point, &previous[assignment[position]]);
            best = Some(match best {
                None => (position, distance),
                Some((best_position, best_distance)) => {
                    // An exact tie falls back to the smaller id, as everywhere else here.
                    let better = match distance.total_cmp(&best_distance) {
                        Ordering::Greater => true,
                        Ordering::Equal => ids[position] < ids[best_position],
                        Ordering::Less => false,
                    };
                    if better {
                        (position, distance)
                    } else {
                        (best_position, best_distance)
                    }
                }
            });
        }
        if let Some((position, _)) = best {
            next[list].clone_from(&points[position]);
            used.insert(position);
        }
    }
}

/// The L2-normalised copy of `vector`; a zero (or non-finite) vector stays zero.
fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm = vector
        .iter()
        .map(|component| f64::from(*component) * f64::from(*component))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return vec![0.0; vector.len()];
    }
    // The vector space is `f32`, so the normalised components narrow back on purpose.
    #[allow(clippy::cast_possible_truncation)]
    let normalized = vector
        .iter()
        .map(|component| (f64::from(*component) / norm) as f32)
        .collect();
    normalized
}

/// The squared euclidean distance between two vectors of equal length.
fn squared_distance(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| {
            let difference = f64::from(*left_value) - f64::from(*right_value);
            difference * difference
        })
        .sum()
}

/// The inner product of two vectors of equal length (`0.0` for a length mismatch).
fn inner_product(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() {
        return 0.0;
    }
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| f64::from(*left_value) * f64::from(*right_value))
        .sum()
}

/// The first id that appears more than once, if any.
fn first_duplicate(ids: &[String]) -> Option<&String> {
    let mut seen = BTreeSet::new();
    ids.iter().find(|id| !seen.insert(id.as_str()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    /// An id for the `index`-th test item, which keeps the input id-sorted.
    fn id(index: usize) -> String {
        format!("exp_{:04}", index + 1)
    }

    /// The ids of `count` items in ascending order.
    fn ids(count: usize) -> Vec<String> {
        (0..count).map(id).collect()
    }

    /// Three clearly separated axes, so a coarse build has an obvious answer.
    fn axes() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn two_builds_produce_identical_centroids_and_lists() {
        let vectors = axes();
        let names = ids(vectors.len());
        let first = CoarseIndex::build(&names, &vectors, 3).expect("build");
        let second = CoarseIndex::build(&names, &vectors, 3).expect("build again");
        assert_eq!(first, second);
        assert_eq!(first.nlist(), 3);
        assert_eq!(first.dimension(), 3);
        assert_eq!(first.centroids(), second.centroids());
        assert_eq!(first.lists(), second.lists());
        // Each duplicated axis lands in one list.
        assert_eq!(first.lists(), &[0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn seeding_starts_at_the_smallest_id_and_spreads_out() {
        let vectors = axes();
        let names = ids(vectors.len());
        let centroids = seed(&vectors, &names, 3);
        assert_eq!(centroids.len(), 3);
        // The first centroid is the vector of the smallest id.
        assert_eq!(centroids[0], normalize(&vectors[0]));
        // The second and third are the farthest points from what is already chosen.
        assert_eq!(centroids[1], normalize(&vectors[2]));
        assert_eq!(centroids[2], normalize(&vectors[4]));

        // A k-means++ pass over points supplied out of id order still starts at the smallest
        // id, because the seed looks the id up instead of assuming index 0. The second seed
        // is the equidistant point with the lower id (`exp_0003`), not the first one seen.
        let rotated = vec![vectors[4].clone(), vectors[0].clone(), vectors[2].clone()];
        let rotated_ids = vec![
            "exp_0005".to_string(),
            "exp_0001".to_string(),
            "exp_0003".into(),
        ];
        let centroids = seed(&rotated, &rotated_ids, 2);
        assert_eq!(centroids[0], normalize(&vectors[0]));
        assert_eq!(centroids[1], normalize(&vectors[2]));
    }

    #[test]
    fn seeding_ties_break_on_the_lower_id() {
        // Two points equidistant from the first chosen point: the lower id must win.
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, -1.0]];
        let names = ids(3);
        let centroids = seed(&vectors, &names, 2);
        assert_eq!(centroids[1], normalize(&vectors[1]));
    }

    #[test]
    fn duplicate_heavy_points_still_build_a_stable_partition() {
        // Seeding can only choose points, so a catalog of duplicated vectors always leaves a
        // list empty; that must not break the build, and a repeat build must be identical.
        let vectors = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];
        let names = ids(vectors.len());
        let index = CoarseIndex::build(&names, &vectors, 3).expect("build");
        assert_eq!(index.nlist(), 3);
        assert_eq!(index.lists().len(), vectors.len());
        assert!(index.lists().iter().all(|list| *list < index.nlist()));
        // A reseeded centroid is one of the stored vectors, never the zero vector an empty
        // mean would have produced.
        assert!(index
            .centroids()
            .iter()
            .all(|centroid| centroid.iter().any(|value| *value != 0.0)));
        assert_eq!(
            CoarseIndex::build(&names, &vectors, 3).expect("again"),
            index
        );
    }

    #[test]
    fn an_empty_list_is_reseeded_with_the_farthest_point() {
        let points = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let names = ids(3);
        // `exp_0003` sits farthest from the centroid it was assigned to, so it seeds list 1.
        let previous = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut next = previous.clone();
        reseed_empty(&points, &names, &previous, &[0, 1, 0], &mut next, &[3, 0]);
        assert_eq!(next[1], points[2]);

        // Two empty lists take the farthest two points, never the same point twice.
        let previous = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, -1.0]];
        let mut next = previous.clone();
        reseed_empty(
            &points,
            &names,
            &previous,
            &[0, 0, 0],
            &mut next,
            &[3, 0, 0],
        );
        assert_eq!(next[1], points[2]);
        assert_eq!(next[2], points[1]);

        // With no candidate left, the empty centroid keeps the value it was given.
        let previous = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut next = previous.clone();
        reseed_empty(
            &points[..1],
            &names[..1],
            &previous,
            &[0],
            &mut next,
            &[0, 0],
        );
        assert_eq!(next[0], points[0]);
        assert_eq!(next[1], previous[1]);
    }

    #[test]
    fn zero_vectors_stay_zero_and_do_not_poison_the_centroids() {
        let vectors = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![3.0, 4.0]];
        assert_eq!(normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
        assert_eq!(normalize(&[-3.0, 4.0]), vec![-0.6, 0.8]);
        let names = ids(vectors.len());
        let index = CoarseIndex::build(&names, &vectors, 2).expect("build");
        assert_eq!(index.lists().len(), 3);
        for centroid in index.centroids() {
            assert!(centroid.iter().all(|value| value.is_finite()));
        }
        // A zero query still probes deterministically instead of panicking.
        let probed = index.probe(&[0.0, 0.0], 2);
        assert_eq!(probed, vec![0, 1]);
    }

    #[test]
    fn an_nlist_larger_than_the_item_count_gives_every_item_a_list() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let names = ids(vectors.len());
        let index = CoarseIndex::build(&names, &vectors, 8).expect("build");
        assert_eq!(index.nlist(), 3);
        assert_eq!(index.lists(), &[0, 1, 2]);
        assert_eq!(index.probe(&[1.0, 0.0], 8), vec![0, 2, 1]);
    }

    #[test]
    fn probing_orders_by_similarity_with_a_list_index_tie_break() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let names = ids(vectors.len());
        let index = CoarseIndex::build(&names, &vectors, 3).expect("build");
        let probed = index.probe(&[1.0, 0.0], 1);
        assert_eq!(probed, vec![0]);
        assert_eq!(index.probed_ids(&[1.0, 0.0], 1), vec!["exp_0001"]);
        // Asking for more lists than exist clamps to all of them.
        assert_eq!(index.probe(&[1.0, 0.0], 9).len(), 3);
        // A zero query ties everywhere, so the lower list indexes win.
        assert_eq!(index.probe(&[0.0, 0.0], 2), vec![0, 1]);
        // A mismatched dimension scores zero everywhere and stays deterministic.
        assert_eq!(index.probe(&[1.0, 0.0, 0.0], 2), vec![0, 1]);
    }

    #[test]
    fn probed_candidate_sets_grow_with_the_probe_count() {
        let vectors: Vec<Vec<f32>> = (0..12)
            .map(|step| {
                #[allow(clippy::cast_precision_loss)]
                let angle = step as f32 * 0.5;
                vec![angle.cos(), angle.sin()]
            })
            .collect();
        let names = ids(vectors.len());
        let index = CoarseIndex::build(&names, &vectors, 6).expect("build");
        let query = [1.0, 0.0];
        let mut previous: BTreeSet<&str> = BTreeSet::new();
        for nprobe in 1..=index.nlist() {
            let current: BTreeSet<&str> = index.probed_ids(&query, nprobe).into_iter().collect();
            assert!(previous.is_subset(&current), "nprobe {nprobe}");
            assert!(current.len() >= previous.len(), "nprobe {nprobe}");
            previous = current;
        }
        assert_eq!(previous.len(), vectors.len());
    }

    #[test]
    fn an_empty_catalog_builds_and_probes_nothing() {
        let index = CoarseIndex::build(&[], &[], 8).expect("build");
        assert_eq!(index.nlist(), 0);
        assert!(index.centroids().is_empty());
        assert!(index.ids().is_empty());
        assert!(index.probe(&[1.0, 0.0], 4).is_empty());
        assert!(index.probed_ids(&[1.0, 0.0], 4).is_empty());
        assert_eq!(index, CoarseIndex::empty());
    }

    #[test]
    fn malformed_inputs_are_validation_errors() {
        let names = ids(2);
        let error = CoarseIndex::build(&names, &[vec![1.0]], 2).expect_err("aligned");
        assert!(
            error.message().contains("ids do not match"),
            "{}",
            error.message()
        );

        let error = CoarseIndex::build(&names, &[vec![], vec![1.0]], 2).expect_err("empty row");
        assert!(error.message().contains("no components"));

        let error =
            CoarseIndex::build(&names, &[vec![1.0], vec![1.0, 2.0]], 2).expect_err("dimension");
        assert!(error.message().contains("index dimension"));

        let repeated = vec!["exp_0001".to_string(), "exp_0001".to_string()];
        let error =
            CoarseIndex::build(&repeated, &[vec![1.0], vec![2.0]], 2).expect_err("duplicate id");
        assert!(error.message().contains("more than once"));
    }

    #[test]
    fn rehydrated_parts_are_validated() {
        let names = ids(2);
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let index = CoarseIndex::from_parts(2, 2, names.clone(), centroids.clone(), vec![0, 1])
            .expect("valid");
        assert_eq!(index.nlist(), 2);

        let cases: Vec<(&str, Result<CoarseIndex>)> = vec![
            (
                "nlist",
                CoarseIndex::from_parts(2, 3, names.clone(), centroids.clone(), vec![0, 1]),
            ),
            (
                "list entries",
                CoarseIndex::from_parts(2, 2, names.clone(), centroids.clone(), vec![0]),
            ),
            (
                "no coarse list",
                CoarseIndex::from_parts(2, 0, names.clone(), Vec::new(), vec![0, 0]),
            ),
            (
                "outside 0..2",
                CoarseIndex::from_parts(2, 2, names.clone(), centroids.clone(), vec![0, 2]),
            ),
            (
                "components",
                CoarseIndex::from_parts(3, 2, names.clone(), centroids.clone(), vec![0, 1]),
            ),
            (
                "sorted ascending",
                CoarseIndex::from_parts(
                    2,
                    2,
                    vec![names[1].clone(), names[0].clone()],
                    centroids.clone(),
                    vec![0, 1],
                ),
            ),
        ];
        for (needle, result) in cases {
            let error = result.expect_err(needle);
            assert!(
                error.message().contains(needle),
                "{needle}: {}",
                error.message()
            );
        }
    }

    #[test]
    fn the_iteration_cap_and_epsilon_are_configurable() {
        // A single iteration cannot refine the seeding much, so the output stays a pair of
        // seed vectors; the default build converges on the same partition here.
        let vectors = axes();
        let names = ids(vectors.len());
        let quick = CoarseIndex::build_with(
            &names,
            &vectors,
            2,
            CoarseConfig {
                max_iterations: 1,
                epsilon: 1e-6,
            },
        )
        .expect("quick build");
        assert_eq!(quick.nlist(), 2);
        let loose = CoarseIndex::build_with(
            &names,
            &vectors,
            2,
            CoarseConfig {
                max_iterations: MAX_LLOYD_ITERATIONS,
                epsilon: 1.0,
            },
        )
        .expect("loose build");
        assert_eq!(loose.nlist(), 2);
        assert_eq!(loose.lists().len(), vectors.len());
    }

    /// The `(pattern, skills)` pairs the index-level probe tests store, in insertion order.
    const CORPUS: [(&str, &[&str]); 8] = [
        ("Summarize the quarterly report", &["summarizer"]),
        ("Summarize the meeting notes", &["summarizer", "formatter"]),
        ("Report the deployment status", &["deployer"]),
        ("Deploy the service to kubernetes", &["deployer"]),
        (
            "Roll back the kubernetes release",
            &["deployer", "rollback"],
        ),
        ("Bake a chocolate cake", &["baking"]),
        ("Bake a cake for the party", &["baking", "party"]),
        ("Write a cake recipe", &["baking"]),
    ];

    /// Fills `index` with [`CORPUS`] over the deterministic offline embedder.
    async fn fill(index: &crate::experience::ExperienceIndex) {
        for (pattern, skills) in CORPUS {
            let item = crate::experience::ExperienceItem::new(pattern)
                .with_skill_ids(skills.iter().copied());
            index.add(item).await.expect("add");
        }
    }

    /// An index over the offline embedder with `algorithm`.
    fn index(algorithm: &str) -> crate::experience::ExperienceIndex {
        crate::experience::ExperienceIndex::new_with_algorithm(
            std::sync::Arc::new(openjiuwen_retrieval::HashEmbedding::new(8)),
            algorithm,
        )
        .expect("index algorithm")
    }

    /// The ids of `hits`, in order.
    fn hit_ids(hits: &[crate::experience::ExperienceHit]) -> Vec<String> {
        hits.iter().map(|hit| hit.item.id.clone()).collect()
    }

    #[tokio::test]
    async fn a_full_probe_reproduces_the_flat_neighbour_list() {
        let flat = index("flat");
        fill(&flat).await;
        let ivf = index("ivf4");
        fill(&ivf).await;
        assert_eq!(ivf.nlist(), 4);
        for query in ["deploy the service", "summarize the report", "bake a cake"] {
            let ranking = crate::experience::ExperienceRanking::new(0);
            let expected = hit_ids(&flat.search(query, &ranking).await.expect("flat"));
            ivf.set_nprobe(4);
            let exact = hit_ids(&ivf.search(query, &ranking).await.expect("ivf"));
            assert_eq!(exact, expected, "{query}");
            // A wide probe clamps to the list count, so it stays the flat answer.
            ivf.set_nprobe(usize::MAX);
            let wide = hit_ids(&ivf.search(query, &ranking).await.expect("wide"));
            assert_eq!(wide, expected, "{query}");
            // A single probed list keeps a non-empty subset of those neighbours.
            ivf.set_nprobe(0);
            let narrow = hit_ids(&ivf.search(query, &ranking).await.expect("narrow"));
            assert!(!narrow.is_empty(), "{query}");
            for id in &narrow {
                assert!(expected.contains(id), "{query} lost {id}");
            }
        }
    }

    #[tokio::test]
    async fn growing_the_probe_only_adds_candidates() {
        let ivf = index("ivf8");
        fill(&ivf).await;
        assert_eq!(ivf.nlist(), ivf.count());
        let ranking = crate::experience::ExperienceRanking::new(0);
        let query = "deploy the kubernetes release";
        let mut previous: Vec<String> = Vec::new();
        for nprobe in 1..=ivf.nlist() {
            ivf.set_nprobe(nprobe);
            let ids = hit_ids(&ivf.search(query, &ranking).await.expect("probe"));
            assert!(ids.len() >= previous.len(), "nprobe {nprobe}");
            for id in &previous {
                assert!(ids.contains(id), "nprobe {nprobe} dropped {id}");
            }
            previous = ids;
        }
        assert_eq!(previous.len(), ivf.count());
    }
}
