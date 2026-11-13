//! Wire types for the sharing path: metadata, wrappers, bundles and queries.
//!
//! These are pure serde data. Unlike Python's `sharing/types.py`, none of them
//! carries a wall-clock timestamp: [`SharedSkillBundle::bundle_id`] is derived
//! from the skill id and the record ids (see [`SharedSkillBundle::new`]) so the
//! same records always publish under the same id. Python uses `uuid4`; the
//! deterministic id is the deliberate Rust difference, which keeps tests and
//! re-publishes reproducible.

use std::collections::BTreeSet;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::experience::ExperienceRecord;

/// The default upload trigger: a human approves each bundle before it leaves.
const DEFAULT_UPLOAD_TRIGGER: &str = "user_approval";
/// The default confidence for metadata built without an explicit value.
const DEFAULT_CONFIDENCE: f64 = 0.7;
/// The prefix every derived bundle id carries.
const BUNDLE_ID_PREFIX: &str = "bundle-";
/// The FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// The FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Per-experience metadata added on the sharing path.
///
/// The record itself is unchanged: provenance for *who* shares and *why* lives
/// here, never inside the archived [`ExperienceRecord`]. Missing JSON fields
/// fall back to [`SharingMeta::default`], and unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SharingMeta {
    /// The skill the experience belongs to; the stable package id in this port.
    pub skill_name: String,
    /// The skill version being shared, or `""` when unknown.
    pub skill_version: String,
    /// What triggered the upload, for example `"user_approval"`.
    pub upload_trigger: String,
    /// How trustworthy the uploader considers the experience, in `0.0..=1.0`.
    pub confidence: f64,
    /// The uploading user, when the caller knows it.
    pub source_user_id: Option<String>,
    /// A short excerpt of the feedback that produced the experience.
    pub feedback_excerpt: Option<String>,
    /// Retrieval keywords attached by the uploader.
    pub keywords: Vec<String>,
}

impl Default for SharingMeta {
    fn default() -> Self {
        Self {
            skill_name: String::new(),
            skill_version: String::new(),
            upload_trigger: DEFAULT_UPLOAD_TRIGGER.to_owned(),
            confidence: DEFAULT_CONFIDENCE,
            source_user_id: None,
            feedback_excerpt: None,
            keywords: Vec::new(),
        }
    }
}

impl SharingMeta {
    /// Creates metadata for `skill_name` at `skill_version`.
    ///
    /// The trigger stays `"user_approval"` and the confidence `0.7` until a
    /// builder overrides them; provenance fields start empty.
    #[must_use]
    pub fn new(skill_name: impl Into<String>, skill_version: impl Into<String>) -> Self {
        Self {
            skill_name: skill_name.into(),
            skill_version: skill_version.into(),
            ..Self::default()
        }
    }

    /// Sets the confidence.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `confidence` is not finite or outside
    /// `0.0..=1.0`.
    pub fn with_confidence(mut self, confidence: f64) -> Result<Self> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(Error::validation(format!(
                "SharingMeta.confidence must be finite and between 0.0 and 1.0, got {confidence}"
            )));
        }
        self.confidence = confidence;
        Ok(self)
    }

    /// Sets the uploading user id.
    #[must_use]
    pub fn with_source_user(mut self, source_user_id: impl Into<String>) -> Self {
        self.source_user_id = Some(source_user_id.into());
        self
    }

    /// Sets the feedback excerpt.
    #[must_use]
    pub fn with_feedback(mut self, feedback_excerpt: impl Into<String>) -> Self {
        self.feedback_excerpt = Some(feedback_excerpt.into());
        self
    }

    /// Sets the uploader's keywords, deduplicated and sorted.
    #[must_use]
    pub fn with_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut unique = BTreeSet::new();
        for keyword in keywords {
            let keyword = keyword.into();
            let keyword = keyword.trim();
            if !keyword.is_empty() {
                unique.insert(keyword.to_owned());
            }
        }
        self.keywords = unique.into_iter().collect();
        self
    }

    /// Checks the invariants a deserialized value may violate.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming a blank `skill_name`, a blank
    /// `upload_trigger`, or a confidence outside `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        if self.skill_name.trim().is_empty() {
            return Err(Error::validation(
                "SharingMeta.skill_name must not be empty",
            ));
        }
        if self.upload_trigger.trim().is_empty() {
            return Err(Error::validation(
                "SharingMeta.upload_trigger must not be empty",
            ));
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(Error::validation(format!(
                "SharingMeta.confidence must be finite and between 0.0 and 1.0, got {}",
                self.confidence
            )));
        }
        Ok(())
    }
}

/// One experience wrapped for sharing.
///
/// The archived record keeps its identity and counters; the wrapper adds the
/// extracted keywords, a short summary and the uploader's metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SharedExperience {
    /// The wrapped record, unchanged.
    pub record: ExperienceRecord,
    /// Retrieval keywords lifted from the record content.
    pub keywords: Vec<String>,
    /// A short human-readable summary of the record content.
    pub summary: String,
    /// The uploader metadata for this experience.
    pub sharing_meta: SharingMeta,
}

impl SharedExperience {
    /// Creates a wrapper from its parts.
    #[must_use]
    pub fn new(
        record: ExperienceRecord,
        keywords: Vec<String>,
        summary: impl Into<String>,
        sharing_meta: SharingMeta,
    ) -> Self {
        Self {
            record,
            keywords,
            summary: summary.into(),
            sharing_meta,
        }
    }
}

/// The smallest unit a [`SharingBackend`](crate::sharing::SharingBackend) stores:
/// every shared experience for one skill.
///
/// [`SharedSkillBundle::bundle_id`] is content-derived so republishing the same
/// records is idempotent on a key-value backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SharedSkillBundle {
    /// The deterministic id: `bundle-` plus 16 lowercase hex digits.
    pub bundle_id: String,
    /// The skill package the experiences belong to.
    pub skill_id: String,
    /// The human-readable skill name.
    pub skill_name: String,
    /// The skill version, or `""` when unknown.
    pub skill_version: String,
    /// Keywords merged across the wrapped experiences.
    pub keywords_aggregate: Vec<String>,
    /// A summary merged across the wrapped experiences.
    pub summary_aggregate: String,
    /// The wrapped experiences, in the order the stager selected them.
    pub experiences: Vec<SharedExperience>,
}

impl SharedSkillBundle {
    /// Creates a bundle and derives its deterministic `bundle_id`.
    ///
    /// The id is `format!("bundle-{:016x}", fnv1a64(key))` where `key` is the
    /// `skill_id` followed by each distinct record id sorted ascending, joined
    /// by newlines. `keywords_aggregate` and `summary_aggregate` start empty;
    /// the stager fills them. Deserialization keeps whatever id the payload
    /// carries, so a stored bundle round-trips unchanged.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `skill_id` is blank.
    pub fn new(
        skill_id: impl Into<String>,
        skill_name: impl Into<String>,
        skill_version: impl Into<String>,
        experiences: Vec<SharedExperience>,
    ) -> Result<Self> {
        let skill_id = skill_id.into();
        if skill_id.trim().is_empty() {
            return Err(Error::validation(
                "SharedSkillBundle.skill_id must not be empty",
            ));
        }
        let mut key = skill_id.clone();
        let record_ids: BTreeSet<&str> = experiences
            .iter()
            .map(|experience| experience.record.record_id.as_str())
            .collect();
        for record_id in record_ids {
            key.push('\n');
            key.push_str(record_id);
        }
        let bundle_id = format!("{BUNDLE_ID_PREFIX}{:016x}", fnv1a64(key.as_bytes()));
        Ok(Self {
            bundle_id,
            skill_id,
            skill_name: skill_name.into(),
            skill_version: skill_version.into(),
            keywords_aggregate: Vec::new(),
            summary_aggregate: String::new(),
            experiences,
        })
    }

    /// The number of wrapped experiences.
    #[must_use]
    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    /// Whether the bundle wraps no experiences.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }

    /// The distinct record ids, sorted ascending.
    #[must_use]
    pub fn record_ids(&self) -> Vec<&str> {
        let ids: BTreeSet<&str> = self
            .experiences
            .iter()
            .map(|experience| experience.record.record_id.as_str())
            .collect();
        ids.into_iter().collect()
    }
}

/// Hub metadata for the single immutable skill package under one `skill_id`.
///
/// The offline port keeps the descriptive half only; upload timestamps are not
/// modelled because the crate never reads a clock.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillPackageMeta {
    /// The stable package id, usually the skill name.
    pub skill_id: String,
    /// The human-readable skill name.
    pub skill_name: String,
    /// A one-paragraph description of the skill.
    pub description: String,
    /// Retrieval keywords for the package.
    pub keywords: Vec<String>,
}

impl SkillPackageMeta {
    /// Creates metadata for `skill_id`; the descriptive fields start empty.
    #[must_use]
    pub fn new(skill_id: impl Into<String>) -> Self {
        Self {
            skill_id: skill_id.into(),
            ..Self::default()
        }
    }
}

/// One row returned by a backend keyword search.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillSearchResult {
    /// The bundle the row points at.
    pub bundle_id: String,
    /// The skill package the bundle belongs to.
    pub skill_id: String,
    /// The human-readable skill name.
    pub skill_name: String,
    /// The bundle summary, used as the result description.
    pub description: String,
    /// How many experiences the bundle carries.
    pub experience_count: usize,
    /// The bundle's aggregated keywords.
    pub keywords: Vec<String>,
    /// The relevance score; higher is better.
    pub score: f64,
}

/// The query-side keyword set used to retrieve relevant bundles.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct QueryKeywords {
    /// Keywords to look for in a bundle's aggregate.
    pub keywords: Vec<String>,
    /// A free-text intent compared against the bundle summary.
    pub intent: String,
    /// The conversation excerpt the keywords were lifted from.
    pub raw_excerpt: String,
}

impl QueryKeywords {
    /// Creates a query from its parts.
    #[must_use]
    pub fn new(
        keywords: Vec<String>,
        intent: impl Into<String>,
        raw_excerpt: impl Into<String>,
    ) -> Self {
        Self {
            keywords,
            intent: intent.into(),
            raw_excerpt: raw_excerpt.into(),
        }
    }

    /// Whether the query carries neither keywords nor intent.
    ///
    /// [`raw_excerpt`](Self::raw_excerpt) is not searched, so an excerpt alone
    /// does not make the query non-empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keywords
            .iter()
            .all(|keyword| keyword.trim().is_empty())
            && self.intent.trim().is_empty()
    }
}

/// The FNV-1a 64-bit hash of `bytes`.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
