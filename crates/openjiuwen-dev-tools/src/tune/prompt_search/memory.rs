//! Prompt memory: the retrieval seam and a dependency-light JSONL store.
//!
//! Ports `memory.py`. The memory stores optimization outcomes (prompt, reward,
//! task characteristics, metrics, observations, timestamp) and retrieves similar
//! prior optimizations to warm-start a policy. Callers with a semantic or
//! embedding store can implement [`PromptMemory`] directly; nothing here assumes
//! a particular backend.
//!
//! [`JsonlPromptMemory`] is append-oriented but rewrites atomically: a mutation
//! writes `records.jsonl.tmp` and renames it over `records.jsonl`, so a reader
//! never sees a half-written file. Malformed lines are skipped on load (and
//! counted by [`JsonlPromptMemory::skipped_lines`]) instead of failing the whole
//! store; identifiers and timestamps always come from the records — nothing here
//! reads a clock.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use openjiuwen_common::error::Result;

use super::models::{PromptRecord, PromptTaskSpec};

/// Splits `text` into the lowercased word-token set used for lexical similarity.
///
/// A token is a maximal run of Unicode alphanumerics or `_`, mirroring Python's
/// `re.findall(r"[\w]+", text)` after lowercasing. Blank input yields an empty
/// set. Shared by the prompt memory and the lexical drift judge so both use one
/// tokenizer.
pub(crate) fn token_set(text: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    let mut current = String::new();
    for character in text.to_lowercase().chars() {
        if character.is_alphanumeric() || character == '_' {
            current.push(character);
        } else if !current.is_empty() {
            tokens.insert(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.insert(current);
    }
    tokens
}

/// The Jaccard similarity of two token sets (`0.0` when either is empty).
pub(crate) fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count();
    let union = left.len() + right.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Locks `mutex`, recovering the data when a previous holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A persistent store of prior optimizations with similarity retrieval.
///
/// Implementations use interior mutability, so every method takes `&self`.
pub trait PromptMemory: Send + Sync {
    /// Persists a completed optimization record.
    fn store(&self, record: PromptRecord) -> Result<()>;

    /// The `top_k` records most similar to `task`, best first.
    fn search_similar(&self, task: &PromptTaskSpec, top_k: usize) -> Vec<PromptRecord>;

    /// The best-scoring record for the exact same objective, if any.
    fn best_for_objective(&self, objective: &str) -> Option<PromptRecord>;

    /// The unapplied records whose gain exceeds `threshold`, highest gain first.
    fn pending(&self, threshold: f64) -> Vec<PromptRecord>;

    /// Flags a record as applied so it stops appearing in [`PromptMemory::pending`].
    ///
    /// Returns whether a record with `record_id` exists. The applied timestamp is
    /// never read from a clock: a caller that needs one supplies it on the record
    /// it stores.
    fn mark_applied(&self, record_id: &str) -> Result<bool>;

    /// Every stored record, in insertion order.
    fn records(&self) -> Vec<PromptRecord>;
}

/// A memory that stores nothing.
///
/// Used when memory is disabled; every query answers empty.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullPromptMemory;

impl NullPromptMemory {
    /// Creates the no-op memory.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PromptMemory for NullPromptMemory {
    fn store(&self, _record: PromptRecord) -> Result<()> {
        Ok(())
    }

    fn search_similar(&self, _task: &PromptTaskSpec, _top_k: usize) -> Vec<PromptRecord> {
        Vec::new()
    }

    fn best_for_objective(&self, _objective: &str) -> Option<PromptRecord> {
        None
    }

    fn pending(&self, _threshold: f64) -> Vec<PromptRecord> {
        Vec::new()
    }

    fn mark_applied(&self, _record_id: &str) -> Result<bool> {
        Ok(false)
    }

    fn records(&self) -> Vec<PromptRecord> {
        Vec::new()
    }
}

/// A JSONL-backed memory with cheap lexical (token-overlap) retrieval.
///
/// One compact JSON record per line in `<dir>/records.jsonl`. No embedding
/// backend is required; the lexical score is enough to warm-start a policy from
/// past runs on similar objectives.
#[derive(Debug)]
pub struct JsonlPromptMemory {
    /// The `records.jsonl` path.
    path: PathBuf,
    /// The in-memory records, in insertion order.
    records: Mutex<Vec<PromptRecord>>,
    /// How many malformed lines were skipped when the store was loaded.
    skipped_lines: AtomicUsize,
}

impl JsonlPromptMemory {
    /// Opens (or creates in memory) the store under `directory`.
    ///
    /// `records.jsonl` is read when it exists. A line that is blank is ignored; a
    /// line that does not parse as a [`PromptRecord`] is skipped and counted, so
    /// one corrupt line cannot hide every good record.
    ///
    /// # Errors
    ///
    /// The filesystem error when the file exists but cannot be read.
    pub fn new(directory: impl AsRef<Path>) -> Result<Self> {
        let path = directory.as_ref().join("records.jsonl");
        let mut records = Vec::new();
        let mut skipped_lines = 0usize;
        if path.is_file() {
            let text = std::fs::read_to_string(&path)?;
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<PromptRecord>(line) {
                    Ok(record) => records.push(record),
                    Err(_) => skipped_lines += 1,
                }
            }
        }
        Ok(Self {
            path,
            records: Mutex::new(records),
            skipped_lines: AtomicUsize::new(skipped_lines),
        })
    }

    /// The `records.jsonl` path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// How many malformed lines were skipped when the store was loaded.
    #[must_use]
    pub fn skipped_lines(&self) -> usize {
        self.skipped_lines.load(AtomicOrdering::Relaxed)
    }

    /// Rewrites the whole file atomically (`records.jsonl.tmp` then rename).
    fn persist(&self, records: &[PromptRecord]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut content = String::new();
        for record in records {
            content.push_str(&serde_json::to_string(record)?);
            content.push('\n');
        }
        let temporary = self.path.with_extension("jsonl.tmp");
        std::fs::write(&temporary, content)?;
        std::fs::rename(&temporary, &self.path)?;
        Ok(())
    }
}

impl PromptMemory for JsonlPromptMemory {
    fn store(&self, record: PromptRecord) -> Result<()> {
        let mut records = lock(&self.records);
        records.push(record);
        self.persist(records.as_slice())
    }

    fn search_similar(&self, task: &PromptTaskSpec, top_k: usize) -> Vec<PromptRecord> {
        let records = lock(&self.records).clone();
        let query = token_set(&task.characteristics());
        if query.is_empty() || records.is_empty() {
            return by_reward_desc(records, top_k);
        }

        let mut scored: Vec<(f64, PromptRecord)> = Vec::new();
        for record in &records {
            let characteristics = if record.task_characteristics.trim().is_empty() {
                record.objective.as_str()
            } else {
                record.task_characteristics.as_str()
            };
            let overlap = jaccard(&query, &token_set(characteristics));
            if overlap > 0.0 {
                scored.push((overlap * (0.5 + 0.5 * record.reward), record.clone()));
            }
        }
        if scored.is_empty() {
            return by_reward_desc(records, top_k);
        }
        scored.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(_, record)| record).collect()
    }

    fn best_for_objective(&self, objective: &str) -> Option<PromptRecord> {
        let records = lock(&self.records);
        let mut best: Option<&PromptRecord> = None;
        for record in records
            .iter()
            .filter(|record| record.objective == objective)
        {
            if best.is_none_or(|best| record.reward > best.reward) {
                best = Some(record);
            }
        }
        best.cloned()
    }

    fn pending(&self, threshold: f64) -> Vec<PromptRecord> {
        let records = lock(&self.records);
        let mut pending: Vec<PromptRecord> = records
            .iter()
            .filter(|record| record.is_pending(threshold))
            .cloned()
            .collect();
        pending.sort_by(|left, right| {
            right
                .gain()
                .partial_cmp(&left.gain())
                .unwrap_or(Ordering::Equal)
        });
        pending
    }

    fn mark_applied(&self, record_id: &str) -> Result<bool> {
        let mut records = lock(&self.records);
        let Some(record) = records
            .iter_mut()
            .find(|record| record.record_id == record_id)
        else {
            return Ok(false);
        };
        record.applied = true;
        self.persist(records.as_slice())?;
        Ok(true)
    }

    fn records(&self) -> Vec<PromptRecord> {
        lock(&self.records).clone()
    }
}

/// Sorts records by reward descending (stable) and truncates to `top_k`.
fn by_reward_desc(mut records: Vec<PromptRecord>, top_k: usize) -> Vec<PromptRecord> {
    records.sort_by(|left, right| {
        right
            .reward
            .partial_cmp(&left.reward)
            .unwrap_or(Ordering::Equal)
    });
    records.truncate(top_k);
    records
}

#[cfg(test)]
mod tests {
    use openjiuwen_test_support::TempWorkspace;

    use super::*;
    use crate::tune::prompt_search::models::PromptTaskCase;

    fn task(objective: &str) -> PromptTaskSpec {
        PromptTaskSpec::new(objective)
            .with_cases(vec![PromptTaskCase::from_text("q", "a").unwrap()])
    }

    fn record(id: &str, characteristics: &str, reward: f64) -> PromptRecord {
        PromptRecord::new(id, "prompt", reward, "objective")
            .with_characteristics(characteristics)
            .with_created_at(1.0)
    }

    #[test]
    fn tokenization_and_jaccard() {
        let tokens = token_set("Write a HAIKU, about_autumn!");
        assert_eq!(
            tokens,
            BTreeSet::from([
                "a".to_owned(),
                "about_autumn".to_owned(),
                "haiku".to_owned(),
                "write".to_owned(),
            ])
        );
        assert!(token_set("   ").is_empty());
        assert_eq!(jaccard(&tokens, &tokens), 1.0);
        assert_eq!(jaccard(&tokens, &BTreeSet::new()), 0.0);
        let other = token_set("write a poem");
        assert!((jaccard(&tokens, &other) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn null_memory_answers_empty() {
        let memory = NullPromptMemory::new();
        memory.store(PromptRecord::new("r", "p", 1.0, "o")).unwrap();
        assert!(memory.records().is_empty());
        assert!(memory.search_similar(&task("objective"), 3).is_empty());
        assert!(memory.best_for_objective("objective").is_none());
        assert!(memory.pending(0.0).is_empty());
        assert!(!memory.mark_applied("r").unwrap());
    }

    #[test]
    fn store_and_reopen_round_trip() {
        let workspace = TempWorkspace::new().unwrap();
        let directory = workspace.path("memory");

        let memory = JsonlPromptMemory::new(&directory).unwrap();
        assert!(memory.records().is_empty());
        assert_eq!(memory.skipped_lines(), 0);
        assert!(memory.path().ends_with("records.jsonl"));

        memory
            .store(record("r1", "write a haiku", 0.4).with_baseline_reward(0.1))
            .unwrap();
        memory.store(record("r2", "write a poem", 0.9)).unwrap();
        drop(memory);

        let reopened = JsonlPromptMemory::new(&directory).unwrap();
        let records = reopened.records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].record_id, "r1");
        assert!((records[0].baseline_reward.unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(records[1].record_id, "r2");
        // The atomic rewrite leaves no temporary file behind.
        assert!(!directory.join("records.jsonl.tmp").exists());
    }

    #[test]
    fn malformed_lines_are_skipped_and_counted() {
        let workspace = TempWorkspace::new().unwrap();
        let directory = workspace.path("memory");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("records.jsonl"),
            "{\"record_id\":\"r1\",\"prompt\":\"p\",\"reward\":0.5,\"objective\":\"o\"}\n\
             not json\n\
             \n\
             {\"record_id\":\"r2\"}\n",
        )
        .unwrap();

        let memory = JsonlPromptMemory::new(&directory).unwrap();
        assert_eq!(memory.skipped_lines(), 1);
        assert_eq!(memory.records().len(), 2);
        assert_eq!(memory.records()[1].record_id, "r2");
    }

    #[test]
    fn search_ranks_by_lexical_overlap_and_reward() {
        let workspace = TempWorkspace::new().unwrap();
        let memory = JsonlPromptMemory::new(workspace.path("memory")).unwrap();
        memory
            .store(record("exact", "write a haiku about autumn", 0.4))
            .unwrap();
        memory
            .store(record("near", "write a haiku about spring", 1.0))
            .unwrap();
        memory
            .store(record("unrelated", "analyze the stock market", 1.0))
            .unwrap();

        let query = task("write a haiku about autumn");
        let results = memory.search_similar(&query, 3);
        assert_eq!(results.len(), 2);
        // 1.0 * (0.5 + 0.2) = 0.7 beats 0.6667 * (0.5 + 0.5) = 0.6667.
        assert_eq!(results[0].record_id, "exact");
        assert_eq!(results[1].record_id, "near");

        let top_one = memory.search_similar(&query, 1);
        assert_eq!(top_one.len(), 1);
        assert_eq!(top_one[0].record_id, "exact");
        assert!(memory.search_similar(&query, 0).is_empty());
    }

    #[test]
    fn search_falls_back_to_the_best_rewards() {
        let workspace = TempWorkspace::new().unwrap();
        let memory = JsonlPromptMemory::new(workspace.path("memory")).unwrap();
        memory.store(record("low", "write a haiku", 0.2)).unwrap();
        memory.store(record("high", "write a haiku", 0.9)).unwrap();

        // No objective tokens at all.
        let fallback = memory.search_similar(&task(""), 1);
        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback[0].record_id, "high");

        // No lexical overlap with either record.
        let fallback = memory.search_similar(&task("quantum chromodynamics"), 2);
        assert_eq!(
            fallback
                .iter()
                .map(|record| record.record_id.as_str())
                .collect::<Vec<_>>(),
            vec!["high", "low"]
        );

        // The stored objective stands in when characteristics are empty.
        let memory = JsonlPromptMemory::new(workspace.path("other")).unwrap();
        memory
            .store(PromptRecord::new("r", "p", 0.5, "write a haiku"))
            .unwrap();
        assert_eq!(memory.search_similar(&task("write a haiku"), 1).len(), 1);
    }

    #[test]
    fn best_for_objective_picks_the_highest_reward() {
        let workspace = TempWorkspace::new().unwrap();
        let memory = JsonlPromptMemory::new(workspace.path("memory")).unwrap();
        memory
            .store(PromptRecord::new("a", "p", 0.3, "same objective"))
            .unwrap();
        memory
            .store(PromptRecord::new("b", "p", 0.8, "same objective"))
            .unwrap();
        memory
            .store(PromptRecord::new("c", "p", 0.9, "other objective"))
            .unwrap();

        let best = memory.best_for_objective("same objective").unwrap();
        assert_eq!(best.record_id, "b");
        assert!(memory.best_for_objective("missing").is_none());
    }

    #[test]
    fn pending_tracks_gain_and_applied_flag() {
        let workspace = TempWorkspace::new().unwrap();
        let memory = JsonlPromptMemory::new(workspace.path("memory")).unwrap();
        memory
            .store(record("small", "c", 0.2).with_baseline_reward(0.1))
            .unwrap();
        memory
            .store(record("big", "c", 0.9).with_baseline_reward(0.1))
            .unwrap();
        memory
            .store(
                record("applied", "c", 1.0)
                    .with_baseline_reward(0.0)
                    .with_applied(true, Some(5.0)),
            )
            .unwrap();

        let pending = memory.pending(0.2);
        assert_eq!(
            pending
                .iter()
                .map(|record| record.record_id.as_str())
                .collect::<Vec<_>>(),
            vec!["big"]
        );
        assert_eq!(memory.pending(0.0).len(), 2);
    }

    #[test]
    fn mark_applied_persists_and_reports_unknown_ids() {
        let workspace = TempWorkspace::new().unwrap();
        let directory = workspace.path("memory");
        let memory = JsonlPromptMemory::new(&directory).unwrap();
        memory.store(record("r1", "c", 0.5)).unwrap();

        assert!(memory.mark_applied("r1").unwrap());
        assert!(!memory.mark_applied("missing").unwrap());
        assert!(memory.records()[0].applied);
        assert!(memory.records()[0].applied_at.is_none());
        drop(memory);

        let reopened = JsonlPromptMemory::new(&directory).unwrap();
        assert!(reopened.records()[0].applied);
    }
}
