//! Case samplers: [`SequentialSampler`], [`RandomSampler`], and
//! [`StratifiedSampler`].
//!
//! The sampling seam Python's tune package reaches for when a trainer or an
//! optimizer should look at a smaller slice of a case set. Each sampler owns its
//! count (there is no per-call count argument), so one value is reusable across
//! case lists of different sizes, and every sampler is a pure function of
//! `(cases, configuration, seed)`: no clock, no environment, no global generator.
//!
//! Sampling never mutates the input: the result holds clones of the selected
//! cases, and the selected cases are a subsequence of the input multiset
//! ([`SequentialSampler`] keeps the input prefix, the shuffled samplers keep the
//! input's element identity and multiset).
//!
//! # Clamping
//!
//! A count larger than the case list clamps to the list's length, so a sampler
//! configured for 100 cases over a 3-case list returns those 3 cases. A count of
//! `0` is valid and yields an empty vector.

use std::collections::BTreeMap;

use openjiuwen_common::error::Result;
use serde_json::Value;

use crate::tune::base::Case;
use crate::tune::prng::Rng64;

/// The stride that derives a per-stratum seed from the sampler seed.
///
/// Each stratum at sorted position `i` is shuffled with a generator seeded from
/// `seed.wrapping_add((i as u64).wrapping_mul(STRATUM_SEED_STRIDE))`, so a stratum
/// keeps its own reproducible permutation and adding a stratum does not reorder
/// the ones before it.
const STRATUM_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// A caller-supplied stratum key function.
///
/// Boxed behind a [`StratifiedSampler`] so the sampler can own it and stay
/// `Send + Sync`; the alias keeps the field type readable.
type StratumKey = Box<dyn Fn(&Case) -> String + Send + Sync>;

/// The sampling seam over a case list.
///
/// The number of cases to keep is configured on the implementation, so a sampler
/// is a reusable value rather than a per-call parameter: build it once and hand
/// the same `&dyn CaseSampler` to every training run.
///
/// # Contract
///
/// - **Clamping.** A `count` larger than `cases.len()` clamps to `cases.len()`;
///   sampling a shorter list never fails and never repeats a case.
/// - **Zero.** A `count` of `0` is valid and returns an empty vector.
/// - **Determinism.** The same sampler over the same cases returns the same cases
///   in the same order on every call. Randomness is always seeded on the sampler
///   (see [`RandomSampler::new`] and [`StratifiedSampler::new`]) and drawn from
///   the crate's one `Rng64` generator, never from a global or the clock.
/// - **Identity.** Sampling selects a subsequence of the input's cases by
///   cloning them: no case is invented, edited, or duplicated, and
///   [`SequentialSampler`] additionally preserves the input order exactly.
///
/// Implementations are expected to derive [`Debug`] so a configuration can be
/// logged, and to be cheap to share; the trait is object safe and `Send + Sync`,
/// so `&dyn CaseSampler` or `Arc<dyn CaseSampler>` both work.
///
/// # Errors
///
/// A sampler that cannot honour its configuration reports
/// [`Error::validation`](openjiuwen_common::error::Error::validation). The
/// samplers in this module never fail (every count is clamped rather than
/// rejected), but the seam stays fallible so a future store-backed sampler can
/// report an I/O failure through the same signature.
pub trait CaseSampler: Send + Sync {
    /// Samples up to the configured number of cases from `cases`.
    ///
    /// See the [trait documentation](CaseSampler) for the clamping, zero-count,
    /// determinism, and identity contract every implementation follows.
    ///
    /// # Errors
    ///
    /// Whatever the implementation cannot recover from, as
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) or a
    /// more specific variant; the samplers in this module are infallible.
    fn sample(&self, cases: &[Case]) -> Result<Vec<Case>>;
}

/// Takes the first `count` cases in their input order.
///
/// The deterministic baseline: no shuffle, no allocation, no randomness. Useful
/// as the "take a quick look at the head of the set" sampler and as the reference
/// the seeded samplers are compared against in tests.
///
/// A `count` larger than the list clamps to the list's length, and `count == 0`
/// yields an empty vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequentialSampler {
    /// How many leading cases to keep, before clamping.
    count: usize,
}

impl SequentialSampler {
    /// Creates a sampler that keeps the first `count` cases.
    ///
    /// `count` is clamped when it exceeds the case list, so no value is rejected
    /// here; `0` is a valid count that samples nothing.
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self { count }
    }

    /// The configured count, before clamping.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns a copy of the sampler keeping `count` cases instead.
    #[must_use]
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = count;
        self
    }
}

impl CaseSampler for SequentialSampler {
    /// Clones the first [`count`](SequentialSampler::count) cases, or every case
    /// when the count exceeds the list.
    fn sample(&self, cases: &[Case]) -> Result<Vec<Case>> {
        Ok(cases.iter().take(self.count).cloned().collect())
    }
}

/// Keeps `count` cases drawn from a seeded Fisher-Yates permutation.
///
/// The whole list is shuffled with the crate's `Rng64` generator (the one
/// [`CaseLoader::shuffle`](crate::tune::CaseLoader::shuffle) uses), then the first
/// `count` entries of that permutation are taken. The same seed over the same
/// cases always yields the same sample, and the sample is a subsequence of the
/// input: a permutation reorders, it never duplicates.
///
/// The count defaults to "every case", so a fresh [`RandomSampler`] is a seeded
/// shuffle with no truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomSampler {
    /// The seed handed to the generator; `0` is remapped by `Rng64`.
    seed: u64,
    /// The maximum number of cases to keep, or `None` for the whole list.
    count: Option<usize>,
}

impl RandomSampler {
    /// Creates a sampler that shuffles with `seed` and keeps every case.
    ///
    /// Seed `0` is accepted and remapped by `Rng64` to its golden-ratio constant,
    /// so `RandomSampler::new(0)` is a valid, reproducible shuffle.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed, count: None }
    }

    /// The configured seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The configured count, or `None` when every case is sampled.
    #[must_use]
    pub fn count(&self) -> Option<usize> {
        self.count
    }

    /// Returns a copy of the sampler keeping at most `count` shuffled cases.
    #[must_use]
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = Some(count);
        self
    }
}

impl CaseSampler for RandomSampler {
    /// Shuffles the input indices with [`RandomSampler::seed`] and clones the
    /// first `count` of the resulting permutation.
    fn sample(&self, cases: &[Case]) -> Result<Vec<Case>> {
        let count = self
            .count
            .map_or(cases.len(), |count| count.min(cases.len()));
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut indices: Vec<usize> = (0..cases.len()).collect();
        Rng64::new(self.seed).shuffle(&mut indices);
        indices.truncate(count);
        Ok(indices
            .into_iter()
            .map(|index| cases[index].clone())
            .collect())
    }
}

/// Keeps a proportional sample of every stratum.
///
/// Cases are grouped by a stratum key — by default the case's
/// [`label`](Case::label) rendered canonically, so two cases with equal labels
/// always land in the same stratum — and each stratum contributes a share of the
/// requested count proportional to its size. Strata are visited in sorted key
/// order both when the shares are computed and when the sample is emitted, so the
/// output is deterministic.
///
/// # Default stratum key
///
/// The default key renders [`Case::label`] as a comma-separated list of
/// `"key"=value` entries in ascending key order, with each key JSON-quoted and
/// each value rendered as compact JSON. Objects inside a value have their keys
/// sorted recursively before rendering, and nested objects/arrays use `:`/`,`
/// separators inside `{}`/`[]`, so `{"b": 1, "a": 2}` and `{"a": 2, "b": 1}` both
/// render as `{"a":2,"b":1}`. A label `{"output": "4", "note": "ok"}` therefore
/// renders as `"note"="ok","output"="4"` (`BTreeMap` order, strings JSON-quoted, so
/// a key containing `=` or `,` cannot be confused with a separator). Values are
/// compared by that rendering, so a `1` and a `1.0` are different strata while a
/// re-serialized but equal label is the same stratum.
///
/// # Allocation
///
/// Shares use the largest-remainder method over integer arithmetic:
/// `exact_i = count * n_i / total` per stratum, each stratum first gets
/// `floor(exact_i)` seats, and the `count - Σ floors` leftover seats go one at a
/// time to the largest fractional remainders `(count * n_i) mod total`. Equal
/// remainders are broken by the lexicographically smaller stratum key first (the
/// earlier stratum index, because strata are visited in sorted key order), so the
/// allocation is total and reproducible. A stratum is never given more seats than
/// it has cases: an allocation beyond a stratum's size is redistributed to the
/// next-largest-remainder stratum that still has capacity, so the sample is
/// always exactly `min(count, cases.len())` cases long.
///
/// # Ordering and shuffling
///
/// Within a stratum the cases are shuffled with the crate's `Rng64` generator,
/// seeded per stratum from the sampler seed and the stratum's sorted position:
/// `seed.wrapping_add((position as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))`.
/// The allocated prefix of each shuffled stratum is then emitted, stratum by
/// stratum, in sorted key order — the output is grouped, never globally mixed.
///
/// A `count` larger than the case list clamps to the list's length (which returns
/// every case in stratum order, not input order), `count == 0` yields an empty
/// vector, and an empty case list yields an empty vector.
pub struct StratifiedSampler {
    /// The seed the per-stratum generators are derived from.
    seed: u64,
    /// The maximum number of cases to keep, or `None` for the whole list.
    count: Option<usize>,
    /// The caller's stratum key function, or `None` for the default label key.
    stratum_key: Option<StratumKey>,
}

impl StratifiedSampler {
    /// Creates a sampler that stratifies by the default label key and keeps every
    /// case.
    ///
    /// Seed `0` is accepted and remapped by `Rng64`, exactly as for
    /// [`RandomSampler::new`].
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            count: None,
            stratum_key: None,
        }
    }

    /// The configured seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The configured count, or `None` when every case is sampled.
    #[must_use]
    pub fn count(&self) -> Option<usize> {
        self.count
    }

    /// Returns a copy of the sampler keeping at most `count` cases.
    #[must_use]
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = Some(count);
        self
    }

    /// Returns a copy of the sampler grouping cases with `stratum_key`.
    ///
    /// The function receives each case by reference and returns the stratum it
    /// belongs to; a plain `fn(&Case) -> String` pointer satisfies the bound. The
    /// key must be a pure function of the case, because the grouping is recomputed
    /// on every [`sample`](CaseSampler::sample)/[`strata`](StratifiedSampler::strata)
    /// call and the result must stay reproducible.
    #[must_use]
    pub fn with_stratum_key(
        mut self,
        stratum_key: impl Fn(&Case) -> String + Send + Sync + 'static,
    ) -> Self {
        self.stratum_key = Some(Box::new(stratum_key));
        self
    }

    /// Groups `cases` into strata, in sorted key order.
    ///
    /// The returned map is `stratum key -> input indices`, where the indices are
    /// the positions of the grouped cases in `cases` and keep their input order.
    /// Exposed so a caller (or a test) can inspect the grouping the sampler uses
    /// without sampling; the sampler itself calls exactly this method.
    #[must_use]
    pub fn strata(&self, cases: &[Case]) -> BTreeMap<String, Vec<usize>> {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, case) in cases.iter().enumerate() {
            let key = self
                .stratum_key
                .as_ref()
                .map_or_else(|| default_stratum_key(case), |key| key(case));
            groups.entry(key).or_default().push(index);
        }
        groups
    }
}

impl std::fmt::Debug for StratifiedSampler {
    /// Reports the configuration without trying to print the key function.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StratifiedSampler")
            .field("seed", &self.seed)
            .field("count", &self.count)
            .field(
                "stratum_key",
                &if self.stratum_key.is_some() {
                    "custom"
                } else {
                    "label"
                },
            )
            .finish()
    }
}

impl CaseSampler for StratifiedSampler {
    /// Allocates the count over the strata, shuffles each stratum, and emits the
    /// allocated prefixes in sorted stratum-key order.
    fn sample(&self, cases: &[Case]) -> Result<Vec<Case>> {
        let count = self
            .count
            .map_or(cases.len(), |count| count.min(cases.len()));
        if count == 0 {
            return Ok(Vec::new());
        }
        let groups = self.strata(cases);
        let sizes: Vec<usize> = groups.values().map(Vec::len).collect();
        let allocation = allocate(count, &sizes);

        let mut sampled = Vec::with_capacity(count);
        for (position, indices) in groups.values().enumerate() {
            let mut ordered = indices.clone();
            let stratum_seed = self
                .seed
                .wrapping_add((position as u64).wrapping_mul(STRATUM_SEED_STRIDE));
            Rng64::new(stratum_seed).shuffle(&mut ordered);
            let take = allocation[position].min(ordered.len());
            sampled.extend(
                ordered
                    .into_iter()
                    .take(take)
                    .map(|index| cases[index].clone()),
            );
        }
        Ok(sampled)
    }
}

/// Allocates `count` seats over strata of the given `sizes` by largest remainder.
///
/// `sizes` must be in the order the strata are emitted (sorted key order), because
/// an index also breaks remainder ties: the earlier index — the lexicographically
/// smaller key — wins. The returned vector is parallel to `sizes`, sums to
/// `min(count, sizes.iter().sum())`, and never exceeds a stratum's size.
fn allocate(count: usize, sizes: &[usize]) -> Vec<usize> {
    let total: usize = sizes.iter().sum();
    let mut allocation = vec![0usize; sizes.len()];
    let count = count.min(total);
    if count == 0 {
        return allocation;
    }

    let total_wide = total as u128;
    let count_wide = count as u128;
    let mut remainders: Vec<(u128, usize)> = Vec::with_capacity(sizes.len());
    let mut assigned = 0usize;
    for (index, &size) in sizes.iter().enumerate() {
        let scaled = count_wide * size as u128;
        let floor = (scaled / total_wide) as usize;
        allocation[index] = floor;
        assigned += floor;
        remainders.push((scaled % total_wide, index));
    }

    // Largest fractional remainder first; ties go to the earlier stratum index,
    // which is the lexicographically smaller key because `sizes` is sorted.
    remainders.sort_unstable_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    let mut leftover = count - assigned;
    for &(_, index) in &remainders {
        if leftover == 0 {
            break;
        }
        if allocation[index] < sizes[index] {
            allocation[index] += 1;
            leftover -= 1;
        }
    }

    // Defensive redistribution: a stratum that is smaller than its allocation hands
    // its seat on to the next one with capacity, so the total stays exact.
    while leftover > 0 {
        let mut progressed = false;
        for index in 0..sizes.len() {
            if leftover == 0 {
                break;
            }
            if allocation[index] < sizes[index] {
                allocation[index] += 1;
                leftover -= 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    allocation
}

/// Renders a case's label as the default stratum key.
///
/// Entries are joined as `"key"=value` in ascending key order, with each key
/// JSON-quoted and every value rendered by [`render_value`]; see
/// [`StratifiedSampler`] for the exact contract.
fn default_stratum_key(case: &Case) -> String {
    let mut entries: Vec<(&String, &Value)> = case.label.iter().collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    entries
        .into_iter()
        .map(|(key, value)| format!("{}={}", render_string(key), render_value(value)))
        .collect::<Vec<String>>()
        .join(",")
}

/// Renders a JSON value as compact, key-sorted text.
///
/// Objects have their keys sorted recursively before rendering, so two equal
/// labels always render identically no matter how their maps were built. Scalars
/// keep their JSON spelling (a string stays quoted and escaped, a number keeps
/// `serde_json`'s representation).
fn render_value(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
            let rendered: Vec<String> = entries
                .into_iter()
                .map(|(key, value)| format!("{}:{}", render_string(key), render_value(value)))
                .collect();
            format!("{{{}}}", rendered.join(","))
        }
        Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(render_value).collect();
            format!("[{}]", rendered.join(","))
        }
        scalar => scalar.to_string(),
    }
}

/// Renders `text` the way JSON renders a string: quoted and escaped.
fn render_string(text: &str) -> String {
    Value::String(text.to_owned()).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A case whose inputs and label both carry `id`, so identity is readable.
    fn case(id: &str) -> Case {
        labelled("answer", Value::String(id.to_owned())).with_case_id(id)
    }

    /// A case with the single label entry `key = value` and a generated id.
    fn labelled(key: &str, value: Value) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String("q".to_owned()));
        let mut label = BTreeMap::new();
        label.insert(key.to_owned(), value);
        Case::new(inputs, label).unwrap()
    }

    /// A case carrying a full label map.
    fn with_label(label: Vec<(&str, Value)>) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String("q".to_owned()));
        let mut map = BTreeMap::new();
        for (key, value) in label {
            map.insert(key.to_owned(), value);
        }
        Case::new(inputs, map).unwrap()
    }

    /// The case ids of a sample, in order.
    fn ids(cases: &[Case]) -> Vec<String> {
        cases.iter().map(|case| case.case_id.clone()).collect()
    }

    /// A list of `count` cases with the ids `0..count`.
    fn cases(count: usize) -> Vec<Case> {
        (0..count).map(|index| case(&index.to_string())).collect()
    }

    /// A sorted list of a sample's case ids, for multiset comparisons.
    fn sorted_ids(cases: &[Case]) -> Vec<String> {
        let mut ids = ids(cases);
        ids.sort_unstable();
        ids
    }

    #[test]
    fn sequential_keeps_the_first_cases_in_order() {
        let input = cases(5);
        assert_eq!(
            ids(&SequentialSampler::new(3).sample(&input).unwrap()),
            ["0", "1", "2"]
        );
        assert_eq!(
            ids(&SequentialSampler::new(0).sample(&input).unwrap()),
            Vec::<String>::new()
        );
    }

    #[test]
    fn sequential_clamps_its_count_to_the_list() {
        let input = cases(2);
        assert_eq!(
            ids(&SequentialSampler::new(10).sample(&input).unwrap()),
            ["0", "1"]
        );
        assert_eq!(
            SequentialSampler::new(10).sample(&[]).unwrap(),
            Vec::<Case>::new()
        );
    }

    #[test]
    fn sequential_readers_and_builder_agree() {
        let sampler = SequentialSampler::new(2);
        assert_eq!(sampler.count(), 2);
        assert_eq!(sampler.with_count(7).count(), 7);
        assert_eq!(
            ids(&sampler.with_count(7).sample(&cases(9)).unwrap()).len(),
            7
        );
        assert_eq!(format!("{sampler:?}"), "SequentialSampler { count: 2 }");
    }

    #[test]
    fn random_is_deterministic_for_a_seed() {
        let input = cases(12);
        let sampler = RandomSampler::new(42).with_count(5);
        let first = sampler.sample(&input).unwrap();
        assert_eq!(ids(&first), ids(&sampler.sample(&input).unwrap()));
        assert_eq!(first.len(), 5);

        // Seed zero is remapped, so it behaves like its golden-ratio equivalent.
        assert_eq!(
            ids(&RandomSampler::new(0).sample(&input).unwrap()),
            ids(&RandomSampler::new(STRATUM_SEED_STRIDE)
                .sample(&input)
                .unwrap())
        );
    }

    #[test]
    fn random_defaults_to_every_case_and_preserves_the_multiset() {
        let input = cases(9);
        for seed in [0, 1, 2, 42, u64::MAX] {
            let sampler = RandomSampler::new(seed);
            assert_eq!(sampler.count(), None);
            let sample = sampler.sample(&input).unwrap();
            assert_eq!(sample.len(), input.len());
            assert_eq!(sorted_ids(&sample), sorted_ids(&input), "seed {seed}");
        }
        assert!(
            (0..8u64)
                .any(|seed| ids(&RandomSampler::new(seed).sample(&input).unwrap()) != ids(&input)),
            "at least one seed must reorder the input"
        );
    }

    #[test]
    fn random_clamps_and_accepts_a_zero_count() {
        let input = cases(3);
        let all = RandomSampler::new(7)
            .with_count(100)
            .sample(&input)
            .unwrap();
        assert_eq!(sorted_ids(&all), sorted_ids(&input));
        assert_eq!(
            RandomSampler::new(7).with_count(0).sample(&input).unwrap(),
            Vec::<Case>::new()
        );
        assert_eq!(
            RandomSampler::new(7).with_count(4).sample(&[]).unwrap(),
            Vec::<Case>::new()
        );
    }

    #[test]
    fn random_has_a_readable_debug_and_seed_reader() {
        let sampler = RandomSampler::new(11).with_count(2);
        assert_eq!(sampler.seed(), 11);
        assert_eq!(sampler.count(), Some(2));
        assert_eq!(
            format!("{sampler:?}"),
            "RandomSampler { seed: 11, count: Some(2) }"
        );
    }

    #[test]
    fn stratified_groups_equal_labels_together() {
        let input = vec![
            with_label(vec![("output", Value::String("a".to_owned()))]),
            with_label(vec![("output", Value::String("b".to_owned()))]),
            with_label(vec![("output", Value::String("a".to_owned()))]),
        ];
        let strata = StratifiedSampler::new(1).strata(&input);
        assert_eq!(strata.len(), 2);
        assert_eq!(
            strata.get("\"output\"=\"a\"").map(Vec::as_slice),
            Some(&[0, 2][..])
        );
        assert_eq!(
            strata.get("\"output\"=\"b\"").map(Vec::as_slice),
            Some(&[1][..])
        );
        // The map is ordered by its keys, so `"a"` precedes `"b"`.
        let keys: Vec<&String> = strata.keys().collect();
        assert_eq!(keys, ["\"output\"=\"a\"", "\"output\"=\"b\""]);
    }

    #[test]
    fn default_stratum_key_sorts_nested_object_keys() {
        let left: Value = serde_json::from_str(r#"{"b": 1, "a": {"y": 2, "x": 3}}"#).unwrap();
        let right: Value = serde_json::from_str(r#"{"a": {"x": 3, "y": 2}, "b": 1}"#).unwrap();
        assert_eq!(render_value(&left), render_value(&right));
        assert_eq!(render_value(&left), r#"{"a":{"x":3,"y":2},"b":1}"#);

        // A multi-entry label renders as sorted `"key"=value` pairs, JSON-quoted.
        let case = with_label(vec![
            ("output", Value::String("4".to_owned())),
            ("note", Value::String("ok".to_owned())),
        ]);
        assert_eq!(default_stratum_key(&case), r#""note"="ok","output"="4""#);

        // Arrays render in order, scalars keep their JSON spelling.
        assert_eq!(
            render_value(&serde_json::json!([1, "two", null, true])),
            r#"[1,"two",null,true]"#
        );
        assert_eq!(render_value(&serde_json::json!(1.5)), "1.5");
    }

    #[test]
    fn stratified_allocation_is_proportional_and_exact() {
        // Strata of 7, 2 and 1 cases, asking for 6: the exact shares are 4.2, 1.2
        // and 0.6, so the floors (4, 1, 0) leave one seat, which goes to the
        // largest remainder — the one-case stratum `c` (0.6), not the big one.
        let mut input = Vec::new();
        for group in ["a", "a", "a", "a", "a", "a", "a", "b", "b", "c"] {
            input.push(labelled("output", Value::String(group.to_owned())));
        }
        let sampler = StratifiedSampler::new(3).with_count(6);
        let sample = sampler.sample(&input).unwrap();
        assert_eq!(sample.len(), 6);
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for case in &sample {
            let key = case.label["output"].as_str().unwrap().to_owned();
            *counts.entry(key).or_default() += 1;
        }
        assert_eq!(counts.get("a"), Some(&4));
        assert_eq!(counts.get("b"), Some(&1));
        assert_eq!(counts.get("c"), Some(&1));
    }

    #[test]
    fn stratified_never_exceeds_a_stratum_and_stays_exact() {
        // 3 of 4 cases in stratum `a`, 1 in stratum `b`; asking for 2 gives both
        // floors 1 and one leftover seat, which may go either way but never
        // exceeds a stratum's size.
        let mut input = Vec::new();
        for group in ["a", "a", "a", "b"] {
            input.push(labelled("output", Value::String(group.to_owned())));
        }
        for count in 0..=5 {
            let sample = StratifiedSampler::new(9)
                .with_count(count)
                .sample(&input)
                .unwrap();
            assert_eq!(sample.len(), count.min(input.len()), "count {count}");
            let a = sample
                .iter()
                .filter(|case| case.label["output"] == Value::String("a".to_owned()))
                .count();
            assert!(a <= 3, "count {count} took {a} of 3 stratum `a` cases");
        }
        // A single-stratum list is a pure shuffle-and-truncate.
        let single = cases(6);
        let sample = StratifiedSampler::new(5)
            .with_count(4)
            .sample(&single)
            .unwrap();
        assert_eq!(sample.len(), 4);
        for case in &sample {
            assert!(single.contains(case));
        }
    }

    #[test]
    fn stratified_is_deterministic_and_emits_sorted_strata() {
        let mut input = Vec::new();
        for (index, group) in ["b", "a", "b", "a", "c"].iter().enumerate() {
            input.push(
                labelled("output", Value::String((*group).to_owned()))
                    .with_case_id(index.to_string()),
            );
        }
        let sampler = StratifiedSampler::new(17).with_count(4);
        let first = sampler.sample(&input).unwrap();
        assert_eq!(ids(&first), ids(&sampler.sample(&input).unwrap()));

        // Strata are emitted in sorted key order: `a` cases before `b`/`c` ones.
        let groups: Vec<&str> = first
            .iter()
            .map(|case| case.label["output"].as_str().unwrap())
            .collect();
        let mut sorted = groups.clone();
        sorted.sort_unstable();
        assert_eq!(groups, sorted);

        // A different seed only reshuffles within the strata: the allocation, and
        // therefore the per-stratum counts, stay the same.
        let other = StratifiedSampler::new(18)
            .with_count(4)
            .sample(&input)
            .unwrap();
        assert_eq!(other.len(), 4);
        let mut other_groups: Vec<&str> = other
            .iter()
            .map(|case| case.label["output"].as_str().unwrap())
            .collect();
        other_groups.sort_unstable();
        assert_eq!(other_groups, sorted);
    }

    #[test]
    fn stratified_defaults_to_every_case_and_accepts_a_custom_key() {
        let mut input = Vec::new();
        for (index, group) in ["x", "y", "x"].iter().enumerate() {
            input.push(
                labelled("output", Value::String((*group).to_owned()))
                    .with_case_id(index.to_string()),
            );
        }
        let all = StratifiedSampler::new(4).sample(&input).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(sorted_ids(&all), sorted_ids(&input));

        assert_eq!(
            StratifiedSampler::new(4)
                .with_count(0)
                .sample(&input)
                .unwrap(),
            Vec::<Case>::new()
        );
        assert_eq!(
            StratifiedSampler::new(4).with_count(2).sample(&[]).unwrap(),
            Vec::<Case>::new()
        );

        // A caller-supplied key, here a plain `fn` pointer: group by case-id parity.
        fn parity(case: &Case) -> String {
            let last = case.case_id.chars().last().unwrap_or('0');
            last.to_digit(10).unwrap_or(0).to_string()
        }
        let sampler = StratifiedSampler::new(4)
            .with_stratum_key(parity)
            .with_count(3);
        let strata = sampler.strata(&input);
        assert_eq!(strata.len(), 3, "one stratum per distinct parity");
        assert_eq!(
            sampler.strata(&input).keys().collect::<Vec<&String>>(),
            ["0", "1", "2"]
        );
        assert_eq!(sampler.sample(&input).unwrap().len(), 3);
        assert!(format!("{sampler:?}").contains("custom"));
        assert_eq!(
            format!("{:?}", StratifiedSampler::new(4)),
            "StratifiedSampler { seed: 4, count: None, stratum_key: \"label\" }"
        );
    }

    #[test]
    fn strata_reports_input_indices_in_order() {
        let mut input = Vec::new();
        for group in ["m", "n", "m", "m"] {
            input.push(labelled("output", Value::String(group.to_owned())));
        }
        let strata = StratifiedSampler::new(1).strata(&input);
        assert_eq!(
            strata.get("\"output\"=\"m\"").map(Vec::as_slice),
            Some(&[0, 2, 3][..])
        );
        assert_eq!(
            strata.get("\"output\"=\"n\"").map(Vec::as_slice),
            Some(&[1][..])
        );
        assert!(StratifiedSampler::new(1).strata(&[]).is_empty());
    }

    #[test]
    fn allocate_distributes_the_largest_remainders() {
        // 3 + 3 + 3 cases, count 4: every exact share is 1.333, so the floors sum
        // to 3 and the one leftover seat goes to the first (smallest) index.
        assert_eq!(allocate(4, &[3, 3, 3]), [2, 1, 1]);
        // 1 + 1 cases, count 1: exact 0.5 each, the tie goes to the first index.
        assert_eq!(allocate(1, &[1, 1]), [1, 0]);
        // Clamping and degenerate inputs.
        assert_eq!(allocate(9, &[2, 1]), [2, 1]);
        assert_eq!(allocate(0, &[2, 1]), [0, 0]);
        assert_eq!(allocate(3, &[]), Vec::<usize>::new());
        assert_eq!(allocate(3, &[0, 0]), [0, 0]);
    }

    #[test]
    fn samplers_are_usable_through_the_trait_object() {
        let input = cases(6);
        let samplers: Vec<Box<dyn CaseSampler>> = vec![
            Box::new(SequentialSampler::new(2)),
            Box::new(RandomSampler::new(2).with_count(2)),
            Box::new(StratifiedSampler::new(2).with_count(2)),
        ];
        for sampler in &samplers {
            let sample = sampler.sample(&input).unwrap();
            assert_eq!(sample.len(), 2);
            for case in &sample {
                assert!(input.contains(case));
            }
        }
    }
}
