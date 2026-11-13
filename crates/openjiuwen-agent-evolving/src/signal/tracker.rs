//! The in-process signal accumulator: dedupe, weighting, decay and ordering.
//!
//! Ports the tracker role of the Python `openjiuwen.agent_evolving.signal`
//! package (`review_feedback.py` / `team.py` accumulate signals before an
//! optimizer consumes them) without any clock or external store:
//!
//! - [`TrackedSignal`] — one deduplicated signal plus its hit count, accumulated
//!   weight and first/last sequence numbers.
//! - [`SignalTracker`] — the bounded map from
//!   [`EvolutionSignal::dedupe_key`] to [`TrackedSignal`], with deterministic
//!   eviction, noise decay and a total ordering for consumers.
//!
//! Ordering comes from a monotonically increasing sequence counter, never from
//! wall-clock time, so two runs fed the same signals produce byte-identical
//! output. Values loaded through serde are not re-validated: the tracker trusts
//! its producer and leaves bounds checks to [`TrackedSignal::signal`]'s
//! [`EvolutionSignal::validate`].

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::signal::types::EvolutionSignal;
use crate::signal::types::EvolutionTarget;

/// The default number of distinct dedupe keys a tracker keeps.
const DEFAULT_CAPACITY: usize = 256;

/// One accumulated signal: the stored record plus its tracker bookkeeping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TrackedSignal {
    /// The first recorded signal for this dedupe key; only `confidence` and
    /// `weight` change as duplicates arrive.
    pub signal: EvolutionSignal,
    /// How many signals have been merged into this entry, starting at `1`.
    pub hits: u64,
    /// The accumulated signal weight, the sum of every merged
    /// [`EvolutionSignal::weight`].
    pub weight: f64,
    /// The sequence number of the first recorded signal.
    pub first_seq: u64,
    /// The sequence number of the most recent recorded signal.
    pub last_seq: u64,
}

/// A bounded, deterministic accumulator of [`EvolutionSignal`]s.
///
/// Signals dedupe by [`EvolutionSignal::dedupe_key`]; duplicates bump the hit
/// count and the accumulated weight, and the highest confidence wins. When the
/// map grows past [`SignalTracker::capacity`] the weakest entry is evicted by
/// the ordered tuple `(weight, last_seq, key)`, so eviction is a pure function
/// of the recorded sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SignalTracker {
    /// The maximum number of distinct dedupe keys retained; `0` stores nothing.
    #[serde(default = "default_capacity")]
    pub capacity: usize,
    /// The multiplier [`SignalTracker::decay_all`] applies in `0.0..=1.0`.
    #[serde(default = "default_unit")]
    pub decay: f64,
    /// The dedupe key to tracked signal map, ordered by key.
    #[serde(default)]
    pub seen: BTreeMap<String, TrackedSignal>,
    /// The monotonic sequence counter behind first/last ordering.
    #[serde(default)]
    pub seq: u64,
}

impl Default for SignalTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalTracker {
    /// Creates an empty tracker with the default capacity and no decay.
    #[must_use]
    pub fn new() -> Self {
        Self {
            capacity: DEFAULT_CAPACITY,
            decay: 1.0,
            seen: BTreeMap::new(),
            seq: 0,
        }
    }

    /// Returns the tracker with `capacity` replaced.
    ///
    /// A capacity of `0` is allowed and simply stores nothing: every recorded
    /// signal is evicted immediately, though [`SignalTracker::record`] still
    /// reports the hit count it observed.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Returns the tracker with its decay factor replaced.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `decay` is not finite or outside `0.0..=1.0`.
    pub fn with_decay(mut self, decay: f64) -> Result<Self> {
        if !decay.is_finite() || !(0.0..=1.0).contains(&decay) {
            return Err(Error::validation(format!(
                "SignalTracker.decay must be finite and between 0.0 and 1.0, got {decay}"
            )));
        }
        self.decay = decay;
        Ok(self)
    }

    /// Records one signal and returns the resulting hit count for its key.
    ///
    /// A duplicate keeps the originally stored payload but raises its
    /// confidence to the maximum seen; the entry's `last_seq` moves and its
    /// `first_seq` does not. The oldest-weakest entry is evicted while the map
    /// exceeds the capacity.
    pub fn record(&mut self, signal: EvolutionSignal) -> u64 {
        let key = signal.dedupe_key();
        let hits = match self.seen.entry(key) {
            Entry::Occupied(entry) => {
                let tracked = entry.into_mut();
                tracked.hits += 1;
                tracked.weight += signal.weight;
                tracked.signal.confidence = tracked.signal.confidence.max(signal.confidence);
                self.seq += 1;
                tracked.last_seq = self.seq;
                tracked.hits
            }
            Entry::Vacant(entry) => {
                self.seq += 1;
                let seq = self.seq;
                let weight = signal.weight;
                entry.insert(TrackedSignal {
                    signal,
                    hits: 1,
                    weight,
                    first_seq: seq,
                    last_seq: seq,
                });
                1
            }
        };
        self.evict_overflow();
        hits
    }

    /// Records every signal from `signals` and returns how many were recorded.
    pub fn record_all<I: IntoIterator<Item = EvolutionSignal>>(&mut self, signals: I) -> usize {
        let mut recorded = 0;
        for signal in signals {
            self.record(signal);
            recorded += 1;
        }
        recorded
    }

    /// Multiplies every accumulated weight by [`SignalTracker::decay`], dropping
    /// entries whose weight is no longer positive.
    pub fn decay_all(&mut self) {
        for tracked in self.seen.values_mut() {
            tracked.weight *= self.decay;
        }
        self.seen.retain(|_, tracked| tracked.weight > 0.0);
    }

    /// The tracked signals ordered by accumulated weight, heaviest first.
    ///
    /// Ties break on the dedupe key ascending, so the order is total and
    /// reproducible across runs.
    #[must_use]
    pub fn signals(&self) -> Vec<&EvolutionSignal> {
        let mut entries: Vec<(&str, &TrackedSignal)> = self
            .seen
            .iter()
            .map(|(key, tracked)| (key.as_str(), tracked))
            .collect();
        entries.sort_by(|(left_key, left), (right_key, right)| {
            right
                .weight
                .total_cmp(&left.weight)
                .then_with(|| left_key.cmp(right_key))
        });
        entries
            .into_iter()
            .map(|(_, tracked)| &tracked.signal)
            .collect()
    }

    /// The first `n` signals of [`SignalTracker::signals`].
    #[must_use]
    pub fn top(&self, n: usize) -> Vec<&EvolutionSignal> {
        self.signals().into_iter().take(n).collect()
    }

    /// The number of distinct tracked signals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Whether the tracker holds no signals.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Groups [`SignalTracker::signals`] by target, preserving that order.
    #[must_use]
    pub fn group_by_target(&self) -> BTreeMap<EvolutionTarget, Vec<&EvolutionSignal>> {
        let mut grouped: BTreeMap<EvolutionTarget, Vec<&EvolutionSignal>> = BTreeMap::new();
        for signal in self.signals() {
            grouped.entry(signal.target).or_default().push(signal);
        }
        grouped
    }

    /// Removes and returns every tracked signal in [`SignalTracker::signals`]
    /// order, resetting the tracker's contents and sequence counter.
    pub fn drain(&mut self) -> Vec<EvolutionSignal> {
        let drained: Vec<EvolutionSignal> = self.signals().into_iter().cloned().collect();
        self.clear();
        drained
    }

    /// Clears every tracked signal and resets the sequence counter.
    pub fn clear(&mut self) {
        self.seen.clear();
        self.seq = 0;
    }

    /// Evicts entries until the map fits the capacity.
    ///
    /// The victim is the smallest `(weight, last_seq, key)` tuple, where the
    /// weight compares with [`f64::total_cmp`], so eviction is a deterministic
    /// function of the recorded sequence.
    fn evict_overflow(&mut self) {
        while self.seen.len() > self.capacity {
            let victim = self
                .seen
                .iter()
                .min_by(|(left_key, left), (right_key, right)| {
                    left.weight
                        .total_cmp(&right.weight)
                        .then_with(|| left.last_seq.cmp(&right.last_seq))
                        .then_with(|| left_key.cmp(right_key))
                })
                .map(|(key, _)| key.clone());
            let Some(victim) = victim else {
                return;
            };
            self.seen.remove(&victim);
        }
    }
}

/// Default capacity used when a serialized tracker omits the field.
fn default_capacity() -> usize {
    DEFAULT_CAPACITY
}

/// Default decay: no decay.
fn default_unit() -> f64 {
    1.0
}
