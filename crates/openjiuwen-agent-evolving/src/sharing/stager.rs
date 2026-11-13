//! Bundle assembly: quality filtering, dedupe and the approval gate.
//!
//! Python's `ShareStager` screens records, wraps them and pushes them onto an
//! upload queue owned by an `ExperienceSharer`. The offline port collapses that
//! pipeline into one step — [`ShareStager::build_bundle`] turns local records
//! into a [`SharedSkillBundle`] — and keeps the approval semantics explicit in
//! [`ShareStager::publish`] and [`ShareStager::share`].
//!
//! The stager never mutates the input records and never writes them anywhere:
//! a record rejected here stays in the local archive untouched.

use std::collections::BTreeSet;

use openjiuwen_common::error::{Error, Result};

use crate::experience::{jaccard, tokens, ExperienceRecord};
use crate::sharing::backend::SharingBackend;
use crate::sharing::keywords::KeywordExtractor;
use crate::sharing::types::{SharedExperience, SharedSkillBundle, SharingMeta};

/// The default minimum record confidence a bundle accepts.
const DEFAULT_MIN_CONFIDENCE: f64 = 0.7;
/// The default maximum number of experiences a bundle carries.
const DEFAULT_MAX_EXPERIENCES: usize = 32;
/// The content-token Jaccard similarity at which two records are duplicates.
const DEDUPE_THRESHOLD: f64 = 0.85;
/// The character budget of one experience summary.
const SUMMARY_CHARS: usize = 160;
/// The character budget of the bundle-level summary.
const AGGREGATE_SUMMARY_CHARS: usize = 240;

/// Screens experience records and assembles shareable bundles.
///
/// Configuration is pure data; every method is deterministic and makes no model
/// call.
#[derive(Debug, Clone, PartialEq)]
pub struct ShareStager {
    /// The keyword/summary extractor applied to every kept record.
    pub extractor: KeywordExtractor,
    /// Records below this confidence are excluded, in `0.0..=1.0`.
    pub min_confidence: f64,
    /// Whether publishing a bundle needs explicit human approval.
    pub require_approval: bool,
    /// The maximum number of experiences kept (`0` means unlimited).
    pub max_experiences: usize,
}

impl Default for ShareStager {
    fn default() -> Self {
        Self {
            extractor: KeywordExtractor::default(),
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            require_approval: true,
            max_experiences: DEFAULT_MAX_EXPERIENCES,
        }
    }
}

impl ShareStager {
    /// Creates a stager around `extractor` with the default gates.
    #[must_use]
    pub fn new(extractor: KeywordExtractor) -> Self {
        Self {
            extractor,
            ..Self::default()
        }
    }

    /// Sets the minimum record confidence.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Sets whether publishing requires explicit approval.
    #[must_use]
    pub fn with_require_approval(mut self, require_approval: bool) -> Self {
        self.require_approval = require_approval;
        self
    }

    /// Sets the maximum number of experiences; `0` means unlimited.
    #[must_use]
    pub fn with_max_experiences(mut self, max_experiences: usize) -> Self {
        self.max_experiences = max_experiences;
        self
    }

    /// Builds a bundle from `records`, filtered by `meta`.
    ///
    /// Records are skipped — and their ids reported in
    /// [`StagedBundle::skipped`] — when they fail
    /// [`ExperienceRecord::validate`], fall below
    /// [`min_confidence`](Self::min_confidence), duplicate an already-kept
    /// record (content-token Jaccard at or above `0.85`, keeping the higher
    /// confidence, then more successes, then the smaller record id), or fall
    /// outside the [`max_experiences`](Self::max_experiences) cap (the cap keeps
    /// the highest-confidence records). Each kept record contributes its
    /// extracted keywords and a 160-character summary; the bundle aggregates
    /// keywords across the kept contents and a 240-character summary across the
    /// experience summaries.
    ///
    /// The bundled skill name and version come from `meta`, and the skill id is
    /// `meta.skill_name`: in this offline port the skill name is the stable
    /// package id, because there is no separate package document to hash.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when no record survives — a bundle with no
    /// experience must not be published — or when `meta.skill_name` is blank.
    pub fn build_bundle(
        &self,
        records: &[ExperienceRecord],
        meta: &SharingMeta,
    ) -> Result<StagedBundle> {
        let mut skipped: BTreeSet<String> = BTreeSet::new();
        let mut kept: Vec<&ExperienceRecord> = Vec::new();
        for record in records {
            if record.validate().is_err() || record.confidence < self.min_confidence {
                skipped.insert(record.record_id.clone());
                continue;
            }
            let candidate_tokens = tokens(&record.content);
            let duplicate = kept.iter().position(|existing| {
                jaccard(&candidate_tokens, &tokens(&existing.content)) >= DEDUPE_THRESHOLD
            });
            if let Some(index) = duplicate {
                if preferred(record, kept[index]) {
                    skipped.insert(kept[index].record_id.clone());
                    kept[index] = record;
                } else {
                    skipped.insert(record.record_id.clone());
                }
                continue;
            }
            kept.push(record);
        }
        kept.sort_by(|left, right| {
            right
                .confidence
                .total_cmp(&left.confidence)
                .then_with(|| right.successes.cmp(&left.successes))
                .then_with(|| left.record_id.cmp(&right.record_id))
        });
        if self.max_experiences > 0 && kept.len() > self.max_experiences {
            for record in &kept[self.max_experiences..] {
                skipped.insert(record.record_id.clone());
            }
            kept.truncate(self.max_experiences);
        }
        if kept.is_empty() {
            return Err(Error::validation(format!(
                "no shareable experiences for skill {:?}",
                meta.skill_name
            )));
        }
        let contents: Vec<String> = kept.iter().map(|record| record.content.clone()).collect();
        let experiences: Vec<SharedExperience> = kept
            .iter()
            .map(|record| {
                SharedExperience::new(
                    (*record).clone(),
                    self.extractor.extract(&record.content),
                    self.extractor.summarise(&record.content, SUMMARY_CHARS),
                    meta.clone(),
                )
            })
            .collect();
        let mut bundle = SharedSkillBundle::new(
            meta.skill_name.clone(),
            meta.skill_name.clone(),
            meta.skill_version.clone(),
            experiences,
        )?;
        bundle.keywords_aggregate = self.extractor.aggregate(&contents);
        let summaries: Vec<String> = bundle
            .experiences
            .iter()
            .map(|experience| experience.summary.clone())
            .filter(|summary| !summary.is_empty())
            .collect();
        bundle.summary_aggregate = self
            .extractor
            .summarise(&summaries.join(" "), AGGREGATE_SUMMARY_CHARS);
        Ok(StagedBundle {
            bundle,
            requires_approval: self.require_approval,
            skipped: skipped.into_iter().collect(),
        })
    }

    /// Publishes `staged` through `backend`, honouring the approval gate.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the bundle requires approval and `approved`
    /// is false, plus any backend publish failure.
    pub async fn publish(
        &self,
        backend: &dyn SharingBackend,
        staged: &StagedBundle,
        approved: bool,
    ) -> Result<String> {
        if staged.requires_approval && !approved {
            return Err(Error::validation(format!(
                "sharing bundle {} requires approval before publishing",
                staged.bundle.bundle_id
            )));
        }
        backend.publish(&staged.bundle).await
    }

    /// Builds and publishes a bundle in one call.
    ///
    /// Returns `Ok(None)` when the approval gate is closed, so callers can treat
    /// "waiting for approval" as a normal outcome rather than an error.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when no record survives, plus any backend publish
    /// failure.
    pub async fn share(
        &self,
        backend: &dyn SharingBackend,
        records: &[ExperienceRecord],
        meta: &SharingMeta,
        approved: bool,
    ) -> Result<Option<String>> {
        let staged = self.build_bundle(records, meta)?;
        if staged.requires_approval && !approved {
            return Ok(None);
        }
        Ok(Some(backend.publish(&staged.bundle).await?))
    }
}

/// A built bundle plus the stager's decisions about it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct StagedBundle {
    /// The assembled bundle.
    pub bundle: SharedSkillBundle,
    /// Whether the stager requires explicit approval before publishing.
    pub requires_approval: bool,
    /// The ids of records dropped by filtering, dedupe or the cap, sorted
    /// ascending.
    pub skipped: Vec<String>,
}

/// Whether `candidate` should replace an equally similar `existing` record.
fn preferred(candidate: &ExperienceRecord, existing: &ExperienceRecord) -> bool {
    match candidate.confidence.total_cmp(&existing.confidence) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => candidate
            .successes
            .cmp(&existing.successes)
            .then_with(|| existing.record_id.cmp(&candidate.record_id))
            .is_gt(),
    }
}
