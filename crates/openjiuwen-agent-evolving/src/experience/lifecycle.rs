//! Lifecycle: promoting proven experience, decaying and pruning stale records.
//!
//! Python `experience/lifecycle.py` runs the online orchestrator (stage,
//! approve, apply, persist) around a model-driven optimizer; that pipeline is
//! deferred. What survives here is the deterministic maintenance pass an archive
//! needs regardless of who produced its records:
//!
//! - stale records (no update for [`LifecycleConfig::stale_seq_gap`] archive
//!   sequence steps) have their weight halved, then are pruned;
//! - records with enough hits and a high success rate are promoted to shared;
//! - [`ExperienceLifecycle::rebuild`] replays a record stream through the
//!   archive's dedupe rule so a rebuilt archive matches an incrementally built
//!   one.
//!
//! Staleness is measured in archive sequence numbers, never in wall-clock time,
//! so a lifecycle pass is reproducible.

use std::collections::BTreeSet;

use openjiuwen_common::error::Result;

use crate::experience::archive::ExperienceArchive;
use crate::experience::models::ExperienceRecord;

/// The weight multiplier applied to a stale record before pruning.
pub const STALE_DECAY_FACTOR: f64 = 0.5;

/// The thresholds one lifecycle pass applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifecycleConfig {
    /// The hit count a record needs before it can be promoted.
    pub min_hits: u64,
    /// The success rate a promoted record needs, in `0.0..=1.0`.
    pub promote_success_rate: f64,
    /// How many archive sequence steps without an update make a record stale.
    pub stale_seq_gap: u64,
    /// The weight below which a record is pruned.
    pub min_weight: f64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            min_hits: 3,
            promote_success_rate: 0.7,
            stale_seq_gap: 100,
            min_weight: 0.1,
        }
    }
}

/// What one [`ExperienceLifecycle::apply`] pass changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LifecycleReport {
    /// Ids promoted to shared, sorted ascending.
    pub promoted: Vec<String>,
    /// Ids pruned from the archive, sorted ascending.
    pub pruned: Vec<String>,
    /// How many records had their weight decayed for staleness.
    pub decayed: usize,
}

/// Applies promotion, decay and pruning to an archive.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExperienceLifecycle {
    /// The thresholds this lifecycle enforces.
    pub config: LifecycleConfig,
}

impl ExperienceLifecycle {
    /// Creates a lifecycle applying `config`.
    #[must_use]
    pub fn new(config: LifecycleConfig) -> Self {
        Self { config }
    }

    /// Runs one maintenance pass and reports what changed.
    ///
    /// The order is decay, prune, promote:
    /// 1. every record whose `archive.seq - updated_seq` exceeds
    ///    [`stale_seq_gap`](LifecycleConfig::stale_seq_gap) has its weight
    ///    multiplied by [`STALE_DECAY_FACTOR`] and is counted in `decayed`;
    /// 2. records with weight below [`min_weight`](LifecycleConfig::min_weight)
    ///    or that are stale are removed and their ids reported in `pruned`;
    /// 3. surviving records with at least [`min_hits`](LifecycleConfig::min_hits)
    ///    hits and a success rate of at least
    ///    [`promote_success_rate`](LifecycleConfig::promote_success_rate) are
    ///    marked shared and reported in `promoted`.
    ///
    /// Both id lists are sorted, so the report is deterministic.
    #[must_use]
    pub fn apply(&self, archive: &mut ExperienceArchive) -> LifecycleReport {
        let seq = archive.seq();
        let stale: BTreeSet<String> = archive
            .records()
            .iter()
            .filter(|record| seq.saturating_sub(record.updated_seq) > self.config.stale_seq_gap)
            .map(|record| record.record_id.clone())
            .collect();
        let mut decayed = 0;
        for record_id in &stale {
            let mut changed = false;
            archive.update(record_id, |record| {
                if record.weight > 0.0 {
                    record.weight *= STALE_DECAY_FACTOR;
                    changed = true;
                }
            });
            if changed {
                decayed += 1;
            }
        }
        let mut pruned: Vec<String> = archive
            .records()
            .iter()
            .filter(|record| {
                record.weight < self.config.min_weight || stale.contains(&record.record_id)
            })
            .map(|record| record.record_id.clone())
            .collect();
        pruned.sort();
        for record_id in &pruned {
            archive.remove(record_id);
        }
        let mut promoted: Vec<String> = archive
            .records()
            .iter()
            .filter(|record| !record.shared)
            .filter(|record| record.hits >= self.config.min_hits)
            .filter(|record| record.success_rate() >= self.config.promote_success_rate)
            .map(|record| record.record_id.clone())
            .collect();
        promoted.sort();
        for record_id in &promoted {
            archive.update(record_id, |record| {
                record.shared = true;
                record.updated_seq = seq;
            });
        }
        LifecycleReport {
            promoted,
            pruned,
            decayed,
        }
    }

    /// Rebuilds an archive from `records`, preserving dedupe.
    ///
    /// Every record is replayed through [`ExperienceArchive::add`], so a
    /// rebuilt archive has the same merge behaviour as an incrementally filled
    /// one. Invalid records are an error rather than a silent drop, because a
    /// rebuild is the moment to notice corrupt input.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::validation`] from the first record
    /// that fails [`ExperienceRecord::validate`].
    pub fn rebuild<I>(&self, records: I) -> Result<ExperienceArchive>
    where
        I: IntoIterator<Item = ExperienceRecord>,
    {
        let mut archive = ExperienceArchive::new();
        for record in records {
            archive.add(record)?;
        }
        Ok(archive)
    }
}
