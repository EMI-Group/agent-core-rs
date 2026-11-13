//! [`ParameterSearcher`]: sample an LLM-call parameter space around a baseline.
//!
//! Ports `ParameterSearcher` from
//! `openjiuwen/dev_tools/tune/trainer/trainer.py`. Python builds the cross
//! product of every tuned parameter's alternatives, drops the all-baseline
//! combination, samples a handful of the rest with `random.sample`, evaluates
//! each one through the trainer, and keeps the first candidate that beats the
//! current epoch's baseline score.
//!
//! The port keeps that shape and firms up the two places Python is not
//! reproducible:
//!
//! - Sampling uses the same seeded `xorshift64*` Fisher-Yates shuffle as
//!   [`CaseLoader::shuffle`](crate::tune::CaseLoader), so one seed always
//!   evaluates the same candidates in the same order instead of Python's
//!   unseeded `random.sample`.
//! - `search_best` never reads a variable that only exists inside its loop: a
//!   parameter set with no alternatives after dropping the baseline returns the
//!   baseline with `evaluated == 0` and empty `best_cases`, where Python raises
//!   `UnboundLocalError`.
//!
//! Candidate generation is deterministic and totally ordered: the search walks
//! the parameter map's keys in [`BTreeMap`] order with the first key outermost,
//! so the last key varies fastest and index `0` is the all-first-value
//! combination (the baseline). Nothing here reads the clock, the environment, or
//! the filesystem, and nothing calls a model: the caller supplies an evaluation
//! closure that returns a score plus the evaluated cases.
//!
//! A candidate is a parameter map, so the baseline's keys define the space:
//! [`candidate_space`] turns a baseline and a list of alternative parameter maps
//! into the per-key choice lists Python derives from `[base, *parameters]`, and
//! [`generate_candidates`] expands those lists into the cross product.

use std::collections::BTreeMap;
use std::future::Future;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tune::base::EvaluatedCase;

/// Python `DEFAULT_CANDIDATES_SAMPLE_NUM`: how many parameter candidates one
/// search evaluates.
pub const DEFAULT_CANDIDATES_SAMPLE_NUM: usize = 6;

/// A deterministic `xorshift64*` generator used by the candidate shuffle.
///
/// The algorithm matches the generator behind
/// [`CaseLoader::shuffle`](crate::tune::CaseLoader) — Marsaglia's xorshift64
/// pushed through a 64-bit multiply — and is repeated here because that one is
/// private to the loader module. It keeps the crate free of random-number
/// dependencies while staying reproducible.
struct Rng64 {
    /// The generator state; never zero after construction.
    state: u64,
}

impl Rng64 {
    /// Seeds the generator; seed `0` is remapped to the golden-ratio constant
    /// because xorshift requires a non-zero state.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// The next pseudo-random 64-bit word.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A pseudo-random index in `0..bound`.
    ///
    /// The modulo bias is irrelevant for shuffling a short candidate list.
    fn below(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}

/// Expands per-parameter choice lists into their cross product.
///
/// The keys are consumed in [`BTreeMap`] order with the first key outermost, so
/// the last key varies fastest and index `0` holds the first value of every key:
/// `{a: [a1, a2], b: [b1, b2]}` becomes
/// `[{a1, b1}, {a1, b2}, {a2, b1}, {a2, b2}]`.
///
/// An empty map yields no candidates, and so does any map with an empty value
/// list; the caller decides whether either is an error.
#[must_use]
pub fn generate_candidates(params: &BTreeMap<String, Vec<Value>>) -> Vec<BTreeMap<String, Value>> {
    if params.is_empty() {
        return Vec::new();
    }
    let mut combinations: Vec<BTreeMap<String, Value>> = vec![BTreeMap::new()];
    for (key, values) in params {
        if values.is_empty() {
            return Vec::new();
        }
        let mut next = Vec::with_capacity(combinations.len().saturating_mul(values.len()));
        for combination in &combinations {
            for value in values {
                let mut candidate = combination.clone();
                candidate.insert(key.clone(), value.clone());
                next.push(candidate);
            }
        }
        combinations = next;
    }
    combinations
}

/// Builds the per-parameter choice lists of a search space.
///
/// For every key of `base` (in [`BTreeMap`] order) the choices are
/// `[base[key], variants[0][key], variants[1][key], …]`, exactly the lists
/// Python derives from `[base_parameters, *parameters]`. Feed the result to
/// [`generate_candidates`] to enumerate the candidates.
///
/// # Errors
///
/// [`Error::validation`] naming the key and the variant index when a variant is
/// missing a key the baseline declares.
pub fn candidate_space(
    base: &BTreeMap<String, Value>,
    variants: &[BTreeMap<String, Value>],
) -> Result<BTreeMap<String, Vec<Value>>> {
    let mut space: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (key, value) in base {
        let mut choices = Vec::with_capacity(variants.len() + 1);
        choices.push(value.clone());
        for (index, variant) in variants.iter().enumerate() {
            let Some(alternative) = variant.get(key) else {
                return Err(Error::validation(format!(
                    "parameter variant {index} is missing the key `{key}` declared by the baseline"
                )));
            };
            choices.push(alternative.clone());
        }
        space.insert(key.clone(), choices);
    }
    Ok(space)
}

/// The result of a [`ParameterSearcher::search_best`] run.
///
/// Python returns the same four values as a tuple
/// `(best_score, best_parameters, best_cases, last_score)`; `evaluated` is the
/// Rust-only count of candidates that were actually scored, which makes a
/// truncated sample observable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ParameterSearchOutcome {
    /// The winning score: the baseline unless a candidate strictly beat it.
    pub best_score: f64,
    /// The winning parameter map.
    pub best_parameters: BTreeMap<String, Value>,
    /// The cases behind `best_score`; empty when nothing was evaluated.
    pub best_cases: Vec<EvaluatedCase>,
    /// The score of the last candidate evaluated, or `best_score` when none was.
    pub last_score: f64,
    /// How many candidates were evaluated.
    pub evaluated: usize,
}

/// Samples a parameter space and keeps the best group it finds.
///
/// Python: `openjiuwen.dev_tools.tune.trainer.ParameterSearcher`. The searcher
/// scores prompt-plus-parameter candidates through the caller's closure, so the
/// same search works with the deterministic evaluator, a model evaluator, or a
/// hand-written scorer.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSearcher {
    /// How many candidates one search evaluates at most.
    sample_num: usize,
    /// The shuffle seed; one seed always yields the same sample and order.
    seed: u64,
}

impl Default for ParameterSearcher {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterSearcher {
    /// Creates a searcher that samples [`DEFAULT_CANDIDATES_SAMPLE_NUM`]
    /// candidates with seed `1`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sample_num: DEFAULT_CANDIDATES_SAMPLE_NUM,
            seed: 1,
        }
    }

    /// Replaces the sample budget.
    #[must_use]
    pub fn with_sample_num(mut self, sample_num: usize) -> Self {
        self.sample_num = sample_num;
        self
    }

    /// Replaces the shuffle seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The sample budget.
    #[must_use]
    pub fn sample_num(&self) -> usize {
        self.sample_num
    }

    /// The shuffle seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Samples the parameter space around `base_parameters` and keeps the best
    /// group.
    ///
    /// `parameters` holds alternative parameter maps; every one must carry each
    /// key of `base_parameters`. The searcher
    ///
    /// 1. builds the space with [`candidate_space`] and the candidates with
    ///    [`generate_candidates`],
    /// 2. drops candidate `0` (the all-baseline combination),
    /// 3. shuffles the rest with the seeded Fisher-Yates and keeps the first
    ///    `min(sample_num, len)`,
    /// 4. evaluates them in that order through `evaluate`, replacing the winner
    ///    only when a score is **strictly** greater than `base_score` (so a tied
    ///    candidate never displaces the baseline or an earlier winner), and
    /// 5. falls back to the first evaluated candidate's cases when nothing beat
    ///    the baseline, matching Python's `best_cases` handling.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `base_score` is not finite, `sample_num` is
    /// zero, `base_parameters` declares no parameter, or a variant misses a base
    /// key; otherwise the first error `evaluate` returns.
    pub async fn search_best<F, Fut>(
        &self,
        base_score: f64,
        base_parameters: BTreeMap<String, Value>,
        parameters: &[BTreeMap<String, Value>],
        evaluate: F,
    ) -> Result<ParameterSearchOutcome>
    where
        F: Fn(&BTreeMap<String, Value>) -> Fut,
        Fut: Future<Output = Result<(f64, Vec<EvaluatedCase>)>>,
    {
        if !base_score.is_finite() {
            return Err(Error::validation(format!(
                "ParameterSearcher base_score must be finite, got {base_score}"
            )));
        }
        if self.sample_num == 0 {
            return Err(Error::validation(
                "ParameterSearcher sample_num must be at least 1, got 0",
            ));
        }
        if base_parameters.is_empty() {
            return Err(Error::validation(
                "ParameterSearcher base_parameters must declare at least one parameter",
            ));
        }

        let space = candidate_space(&base_parameters, parameters)?;
        let mut candidates = generate_candidates(&space);
        if !candidates.is_empty() {
            candidates.remove(0);
        }
        let mut rng = Rng64::new(self.seed);
        shuffle(&mut candidates, &mut rng);
        candidates.truncate(self.sample_num);

        let mut best_score = base_score;
        let mut best_parameters = base_parameters;
        let mut best_cases: Option<Vec<EvaluatedCase>> = None;
        let mut last_score = base_score;
        let mut evaluated = 0;

        for candidate in &candidates {
            let (score, cases) = evaluate(candidate).await?;
            evaluated += 1;
            last_score = score;
            if score > best_score {
                best_score = score;
                best_parameters = candidate.clone();
                best_cases = Some(cases);
            } else if best_cases.is_none() {
                // Python parity: the first evaluated candidate supplies the
                // cases when no candidate improved on the baseline.
                best_cases = Some(cases);
            }
        }

        Ok(ParameterSearchOutcome {
            best_score,
            best_parameters,
            best_cases: best_cases.unwrap_or_default(),
            last_score,
            evaluated,
        })
    }
}

/// Fisher-Yates shuffle driven by a seeded [`Rng64`].
fn shuffle(candidates: &mut [BTreeMap<String, Value>], rng: &mut Rng64) {
    for i in (1..candidates.len()).rev() {
        let j = rng.below(i + 1);
        candidates.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::tune::base::Case;

    /// The string `value` as JSON.
    fn text(value: &str) -> Value {
        Value::String(value.to_owned())
    }

    /// A parameter map from string pairs.
    fn entries(pairs: &[(&str, &str)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), text(value)))
            .collect()
    }

    /// A choice list from string values.
    fn values(items: &[&str]) -> Vec<Value> {
        items.iter().map(|item| text(item)).collect()
    }

    /// The parameter map of a candidate, rendered as `key=value` pairs.
    fn render(candidate: &BTreeMap<String, Value>) -> String {
        candidate
            .iter()
            .map(|(key, value)| format!("{key}={}", value.as_str().unwrap_or("?")))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// A minimal valid case with the given identifier.
    fn case(id: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), text(id));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), text(id));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    /// An evaluated case carrying the identifier of the candidate it came from.
    fn evaluated(id: &str, score: f64) -> EvaluatedCase {
        EvaluatedCase::new(case(id), BTreeMap::new(), score, "").unwrap()
    }

    /// The score a candidate declares in its `score` parameter.
    fn declared_score(candidate: &BTreeMap<String, Value>) -> f64 {
        candidate
            .get("score")
            .and_then(Value::as_str)
            .and_then(|raw| raw.parse::<f64>().ok())
            .unwrap_or(0.0)
    }

    /// The rendered parameter map as the identifiers of the evaluated cases.
    fn ids_of(cases: &[EvaluatedCase]) -> Vec<String> {
        cases
            .iter()
            .map(|evaluated| evaluated.case_id().to_owned())
            .collect()
    }

    #[test]
    fn the_sample_budget_matches_python() {
        assert_eq!(DEFAULT_CANDIDATES_SAMPLE_NUM, 6);
    }

    #[test]
    fn constructors_read_back_their_builders() {
        let searcher = ParameterSearcher::new();
        assert_eq!(searcher.sample_num(), DEFAULT_CANDIDATES_SAMPLE_NUM);
        assert_eq!(searcher.seed(), 1);
        assert_eq!(searcher, ParameterSearcher::default());

        let tuned = ParameterSearcher::new().with_sample_num(3).with_seed(42);
        assert_eq!(tuned.sample_num(), 3);
        assert_eq!(tuned.seed(), 42);
        assert!(format!("{tuned:?}").contains("ParameterSearcher"));
    }

    #[test]
    fn generate_candidates_walks_the_keys_outermost_first() {
        let mut params = BTreeMap::new();
        params.insert("a".to_owned(), values(&["a1", "a2"]));
        params.insert("b".to_owned(), values(&["b1", "b2"]));

        let candidates = generate_candidates(&params);
        let rendered: Vec<String> = candidates.iter().map(render).collect();
        assert_eq!(
            rendered,
            ["a=a1,b=b1", "a=a1,b=b2", "a=a2,b=b1", "a=a2,b=b2"]
        );
        assert_eq!(candidates[0]["a"], text("a1"));
        assert_eq!(candidates[0]["b"], text("b1"));
    }

    #[test]
    fn generate_candidates_handles_single_keys_and_single_values() {
        let mut single_key = BTreeMap::new();
        single_key.insert("only".to_owned(), values(&["x", "y", "z"]));
        let candidates = generate_candidates(&single_key);
        assert_eq!(
            candidates.iter().map(render).collect::<Vec<_>>(),
            ["only=x", "only=y", "only=z"]
        );

        let mut single_value = BTreeMap::new();
        single_value.insert("a".to_owned(), values(&["a1"]));
        single_value.insert("b".to_owned(), values(&["b1"]));
        assert_eq!(
            generate_candidates(&single_value)
                .iter()
                .map(render)
                .collect::<Vec<_>>(),
            ["a=a1,b=b1"]
        );
    }

    #[test]
    fn generate_candidates_yields_nothing_for_empty_input() {
        assert!(generate_candidates(&BTreeMap::new()).is_empty());

        let mut with_empty_list = BTreeMap::new();
        with_empty_list.insert("a".to_owned(), values(&["a1"]));
        with_empty_list.insert("b".to_owned(), Vec::new());
        assert!(generate_candidates(&with_empty_list).is_empty());
    }

    #[test]
    fn candidate_space_lists_the_baseline_first() {
        let base = entries(&[("a", "a0"), ("b", "b0")]);
        let variants = [entries(&[("a", "a1"), ("b", "b1")])];
        let space = candidate_space(&base, &variants).unwrap();

        assert_eq!(space.len(), 2);
        assert_eq!(space["a"], values(&["a0", "a1"]));
        assert_eq!(space["b"], values(&["b0", "b1"]));

        let rendered: Vec<String> = generate_candidates(&space).iter().map(render).collect();
        assert_eq!(rendered[0], "a=a0,b=b0");
        assert_eq!(
            rendered,
            ["a=a0,b=b0", "a=a0,b=b1", "a=a1,b=b0", "a=a1,b=b1"]
        );
    }

    #[test]
    fn candidate_space_without_variants_holds_the_baseline_only() {
        let base = entries(&[("a", "a0")]);
        let space = candidate_space(&base, &[]).unwrap();
        assert_eq!(space["a"], values(&["a0"]));
        assert_eq!(
            generate_candidates(&space)
                .iter()
                .map(render)
                .collect::<Vec<_>>(),
            ["a=a0"]
        );
    }

    #[test]
    fn candidate_space_of_an_empty_baseline_is_empty() {
        let variants = [entries(&[("a", "a1")])];
        let space = candidate_space(&BTreeMap::new(), &variants).unwrap();
        assert!(space.is_empty());
        assert!(generate_candidates(&space).is_empty());
    }

    #[test]
    fn candidate_space_rejects_a_variant_missing_a_key() {
        let base = entries(&[("a", "a0"), ("b", "b0")]);
        let variants = [
            entries(&[("a", "a1"), ("b", "b1")]),
            entries(&[("a", "a2")]),
        ];
        let error = candidate_space(&base, &variants).unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        let message = error.to_string();
        assert!(message.contains('b'), "names the key: {message}");
        assert!(message.contains('1'), "names the variant: {message}");
    }

    #[tokio::test]
    async fn search_best_rejects_an_invalid_baseline_score() {
        let searcher = ParameterSearcher::new();
        let base = entries(&[("a", "a0")]);
        for base_score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = searcher
                .search_best(base_score, base.clone(), &[], |_| async {
                    Ok((0.0, Vec::new()))
                })
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
            assert!(error.to_string().contains("base_score"), "{error}");
        }
    }

    #[tokio::test]
    async fn search_best_rejects_a_zero_sample_budget() {
        let searcher = ParameterSearcher::new().with_sample_num(0);
        let error = searcher
            .search_best(0.0, entries(&[("a", "a0")]), &[], |_| async {
                Ok((0.0, Vec::new()))
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("sample_num"), "{error}");
    }

    #[tokio::test]
    async fn search_best_rejects_empty_base_parameters() {
        let error = ParameterSearcher::new()
            .search_best(0.0, BTreeMap::new(), &[], |_| async {
                Ok((0.0, Vec::new()))
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("base_parameters"), "{error}");
    }

    #[tokio::test]
    async fn search_best_propagates_a_variant_validation_error() {
        let variants = [entries(&[("other", "x")])];
        let error = ParameterSearcher::new()
            .search_best(0.0, entries(&[("a", "a0")]), &variants, |_| async {
                Ok((0.0, Vec::new()))
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains('a'), "{error}");
    }

    #[tokio::test]
    async fn search_best_keeps_the_strictly_better_candidate() {
        let base = entries(&[("score", "0.2")]);
        let variants = [entries(&[("score", "0.9")])];
        let outcome = ParameterSearcher::new()
            .search_best(0.2, base, &variants, |candidate| {
                let score = declared_score(candidate);
                let id = render(candidate);
                async move { Ok((score, vec![evaluated(&id, score)])) }
            })
            .await
            .unwrap();

        assert!(
            (outcome.best_score - 0.9).abs() < 1e-12,
            "{}",
            outcome.best_score
        );
        assert_eq!(outcome.best_parameters["score"], text("0.9"));
        assert_eq!(ids_of(&outcome.best_cases), ["score=0.9"]);
        assert_eq!(outcome.evaluated, 1);
        assert!((outcome.last_score - 0.9).abs() < 1e-12);
    }

    #[tokio::test]
    async fn search_best_keeps_the_baseline_on_a_tie() {
        let base = entries(&[("t", "base")]);
        let variants = [entries(&[("t", "tied")])];
        let order = RefCell::new(Vec::new());
        let outcome = ParameterSearcher::new()
            .search_best(0.5, base.clone(), &variants, |candidate| {
                order.borrow_mut().push(render(candidate));
                async { Ok((0.5, vec![evaluated("tied", 0.5)])) }
            })
            .await
            .unwrap();

        assert!((outcome.best_score - 0.5).abs() < 1e-12);
        assert_eq!(outcome.best_parameters, base);
        assert_eq!(outcome.evaluated, 1);
        // The fallback fills the cases from the first evaluated candidate.
        let evaluated_order = order.into_inner();
        assert_eq!(evaluated_order, ["t=tied"]);
        assert_eq!(ids_of(&outcome.best_cases), ["tied"]);
    }

    #[tokio::test]
    async fn search_best_tracks_the_last_score_of_the_sample() {
        let base = entries(&[("p", "base")]);
        let variants = [
            entries(&[("p", "v1")]),
            entries(&[("p", "v2")]),
            entries(&[("p", "v3")]),
        ];
        let order = RefCell::new(Vec::new());
        let outcome = ParameterSearcher::new()
            .with_seed(11)
            .with_sample_num(3)
            .search_best(0.1, base.clone(), &variants, |candidate| {
                order.borrow_mut().push(render(candidate));
                let score = 0.9;
                let id = render(candidate);
                async move { Ok((score, vec![evaluated(&id, score)])) }
            })
            .await
            .unwrap();

        let evaluated_order = order.into_inner();
        assert_eq!(evaluated_order.len(), 3);
        assert!((outcome.best_score - 0.9).abs() < 1e-12);
        // The baseline loses to the first candidate, and ties keep the winner.
        assert_eq!(
            outcome.best_parameters["p"],
            text(evaluated_order[0].trim_start_matches("p="))
        );
        assert!((outcome.last_score - 0.9).abs() < 1e-12);
        assert_eq!(ids_of(&outcome.best_cases), [evaluated_order[0].clone()]);
    }

    #[tokio::test]
    async fn search_best_returns_the_baseline_without_candidates() {
        let base = entries(&[("a", "a0")]);
        let outcome = ParameterSearcher::new()
            .search_best(0.4, base.clone(), &[], |_| async {
                panic!("no candidate should be evaluated")
            })
            .await
            .unwrap();

        assert!((outcome.best_score - 0.4).abs() < 1e-12);
        assert_eq!(outcome.best_parameters, base);
        assert!(outcome.best_cases.is_empty());
        assert!((outcome.last_score - 0.4).abs() < 1e-12);
        assert_eq!(outcome.evaluated, 0);
    }

    #[tokio::test]
    async fn search_best_never_evaluates_the_baseline_combination() {
        let base = entries(&[("a", "a0"), ("b", "b0")]);
        let variants = [entries(&[("a", "a1"), ("b", "b1")])];
        let evaluated_parameters = RefCell::new(Vec::new());
        let outcome = ParameterSearcher::new()
            .search_best(0.0, base.clone(), &variants, |candidate| {
                evaluated_parameters.borrow_mut().push(candidate.clone());
                async { Ok((1.0, vec![evaluated("x", 1.0)])) }
            })
            .await
            .unwrap();

        let seen = evaluated_parameters.into_inner();
        assert_eq!(seen.len(), 3);
        assert!(seen.iter().all(|candidate| candidate != &base), "{seen:?}");
        assert_eq!(outcome.evaluated, 3);
        // Every candidate carries both keys, so every map is complete.
        assert!(seen
            .iter()
            .all(|candidate| candidate.len() == 2 && candidate.contains_key("a")));
    }

    #[tokio::test]
    async fn sampling_is_reproducible_for_a_seed() {
        let base = entries(&[("a", "a0"), ("b", "b0")]);
        let variants = [
            entries(&[("a", "a1"), ("b", "b1")]),
            entries(&[("a", "a2"), ("b", "b2")]),
            entries(&[("a", "a3"), ("b", "b3")]),
        ];
        let run = |seed: u64| {
            let base = base.clone();
            let variants = variants.clone();
            async move {
                let order = RefCell::new(Vec::new());
                let searcher = ParameterSearcher::new().with_seed(seed);
                let outcome = searcher
                    .search_best(0.0, base, &variants, |candidate| {
                        order.borrow_mut().push(render(candidate));
                        async { Ok((0.5, vec![evaluated("c", 0.5)])) }
                    })
                    .await
                    .unwrap();
                (order.into_inner(), outcome)
            }
        };

        let (first_order, first) = run(7).await;
        let (second_order, second) = run(7).await;
        assert_eq!(first_order, second_order);
        assert_eq!(first, second);
        assert_eq!(first.evaluated, DEFAULT_CANDIDATES_SAMPLE_NUM);
        assert_eq!(first_order.len(), DEFAULT_CANDIDATES_SAMPLE_NUM);
        // The sample is a subset of the 15 non-baseline candidates, without repeats.
        let mut unique = first_order.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), first_order.len(), "{first_order:?}");
        assert!(first_order.iter().all(|rendered| rendered != "a=a0,b=b0"));
    }

    #[tokio::test]
    async fn sampling_never_exceeds_the_available_candidates() {
        let base = entries(&[("a", "a0"), ("b", "b0")]);
        let variants = [
            entries(&[("a", "a1"), ("b", "b1")]),
            entries(&[("a", "a2"), ("b", "b2")]),
            entries(&[("a", "a3"), ("b", "b3")]),
        ];
        // 4 choices per key → 16 combinations, minus the baseline → 15.
        let outcome = ParameterSearcher::new()
            .with_sample_num(100)
            .search_best(0.0, base, &variants, |_| async {
                Ok((0.5, vec![evaluated("c", 0.5)]))
            })
            .await
            .unwrap();
        assert_eq!(outcome.evaluated, 15);

        let capped = ParameterSearcher::new()
            .with_sample_num(2)
            .search_best(
                0.0,
                entries(&[("a", "a0"), ("b", "b0")]),
                &variants,
                |_| async { Ok((0.5, vec![evaluated("c", 0.5)])) },
            )
            .await
            .unwrap();
        assert_eq!(capped.evaluated, 2);
    }

    #[tokio::test]
    async fn search_best_propagates_evaluation_errors() {
        let variants = [entries(&[("a", "a1")])];
        let error = ParameterSearcher::new()
            .search_best(0.0, entries(&[("a", "a0")]), &variants, |_| async {
                Err(Error::validation("evaluator exploded"))
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("evaluator exploded"), "{error}");
    }

    #[test]
    fn outcome_round_trips_through_json_and_defaults() {
        let outcome = ParameterSearchOutcome {
            best_score: 0.75,
            best_parameters: entries(&[("temperature", "0.2")]),
            best_cases: vec![evaluated("c1", 0.75)],
            last_score: 0.5,
            evaluated: 3,
        };
        let value = serde_json::to_value(&outcome).unwrap();
        let object = value.as_object().unwrap();
        for key in [
            "best_score",
            "best_parameters",
            "best_cases",
            "last_score",
            "evaluated",
        ] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: ParameterSearchOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(restored, outcome);

        let empty: ParameterSearchOutcome = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, ParameterSearchOutcome::default());
        assert!(empty.best_parameters.is_empty());
        assert!(empty.best_cases.is_empty());
        assert_eq!(empty.evaluated, 0);
        assert!(empty.best_score.abs() < f64::EPSILON);
        assert!(empty.last_score.abs() < f64::EPSILON);
    }
}
