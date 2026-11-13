//! The experience record, its section vocabulary, and the query that finds it.
//!
//! Python keeps an experience as an `EvolutionRecord` with a wall-clock
//! timestamp and a free-form patch payload. This crate keeps a typed, clock-free
//! record instead: sequence counters (`created_seq`, `updated_seq`) stamp
//! recency against the owning [`ExperienceArchive`](crate::experience::ExperienceArchive)
//! so archives stay deterministic and reproducible from a fixed input order.
//!
//! [`ExperienceRecord::score`] is the single ranking formula every consumer
//! shares: `weight * confidence * (0.5 + success_rate)`, so a record that has
//! never been used still scores half of an equally weighted record with a
//! perfect success rate, and a record with no attempts is never ranked below
//! zero.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::signal::EvolutionTarget;

/// Where an experience applies inside a skill document.
///
/// The vocabulary mirrors the section targets an evolution signal can carry;
/// [`EvolutionTarget::Instruction`] and [`EvolutionTarget::Description`] both
/// live in the description layer, and targets without a section of their own
/// map to [`ExperienceSection::Other`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ExperienceSection {
    /// The skill description shown before the body.
    Description,
    /// The main instruction/body block of a skill.
    Body,
    /// A script or code asset belonging to the skill.
    Script,
    /// A tool implementation or its schema.
    Tool,
    /// Stored long-term memory.
    Memory,
    /// Anything the crate does not classify.
    #[default]
    Other,
}

impl ExperienceSection {
    /// The section as its `snake_case` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Description => "description",
            Self::Body => "body",
            Self::Script => "script",
            Self::Tool => "tool",
            Self::Memory => "memory",
            Self::Other => "other",
        }
    }
}

impl From<EvolutionTarget> for ExperienceSection {
    /// Projects a signal target onto the section it updates.
    ///
    /// `Instruction` folds into [`ExperienceSection::Description`] because
    /// instruction text lives in the description layer; `Skill`, `Context` and
    /// `Other` have no section of their own and map to
    /// [`ExperienceSection::Other`].
    fn from(target: EvolutionTarget) -> Self {
        match target {
            EvolutionTarget::Instruction | EvolutionTarget::Description => Self::Description,
            EvolutionTarget::Body => Self::Body,
            EvolutionTarget::Script => Self::Script,
            EvolutionTarget::Tool => Self::Tool,
            EvolutionTarget::Memory => Self::Memory,
            EvolutionTarget::Skill | EvolutionTarget::Context | EvolutionTarget::Other => {
                Self::Other
            }
        }
    }
}

/// One reusable lesson extracted from a signal, trajectory or failing case.
///
/// The record is pure serde data: no clocks, no model calls and no environment
/// reads. Unknown JSON fields are ignored so newer producers stay readable, and
/// [`ExperienceRecord::validate`] is the explicit check for the invariants
/// [`ExperienceRecord::new`] enforces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceRecord {
    /// Stable identifier of the record inside an archive.
    pub record_id: String,
    /// The skill the lesson applies to; never empty.
    pub skill_name: String,
    /// The section the lesson belongs to.
    pub section: ExperienceSection,
    /// The lesson text shown to a model or a human; never empty.
    pub content: String,
    /// Retrieval keywords, sorted and deduplicated by the builders.
    pub keywords: Vec<String>,
    /// Where the record came from, for example `signal` or `trajectory`.
    #[serde(default = "default_source")]
    pub source: String,
    /// How trustworthy the lesson is, in `0.0..=1.0`.
    #[serde(default = "default_unit")]
    pub confidence: f64,
    /// Relative importance used by ranking and decay; finite and non-negative.
    #[serde(default = "default_unit")]
    pub weight: f64,
    /// How often the record was presented or confirmed.
    pub hits: u64,
    /// How often acting on the record helped.
    pub successes: u64,
    /// How often acting on the record did not help.
    pub failures: u64,
    /// Whether the record may be shared with other users.
    pub shared: bool,
    /// Archive sequence number at insertion; `0` until the archive assigns one.
    pub created_seq: u64,
    /// Archive sequence number of the last update; `0` until assigned.
    pub updated_seq: u64,
    /// Free-form provenance; keys are stable and serialized sorted.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ExperienceRecord {
    fn default() -> Self {
        Self {
            record_id: String::new(),
            skill_name: String::new(),
            section: ExperienceSection::default(),
            content: String::new(),
            keywords: Vec::new(),
            source: default_source(),
            confidence: default_unit(),
            weight: default_unit(),
            hits: 0,
            successes: 0,
            failures: 0,
            shared: false,
            created_seq: 0,
            updated_seq: 0,
            metadata: BTreeMap::new(),
        }
    }
}

impl ExperienceRecord {
    /// Creates a validated record for `skill_name` carrying `content`.
    ///
    /// Keywords, provenance and counters start empty except `source`
    /// (`"experience"`), `confidence` and `weight` (`1.0`). The sequence
    /// counters stay `0` until an archive assigns them.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `record_id`, `skill_name` or `content` is
    /// blank after trimming.
    pub fn new(
        record_id: impl Into<String>,
        skill_name: impl Into<String>,
        section: ExperienceSection,
        content: impl Into<String>,
    ) -> Result<Self> {
        let record = Self {
            record_id: record_id.into(),
            skill_name: skill_name.into(),
            section,
            content: content.into(),
            ..Self::default()
        };
        record.validate()?;
        Ok(record)
    }

    /// Checks the invariants the constructor enforces on a deserialized value.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the first blank string, the confidence
    /// outside `0.0..=1.0`, or a non-finite/negative weight.
    pub fn validate(&self) -> Result<()> {
        if self.record_id.trim().is_empty() {
            return Err(Error::validation(
                "ExperienceRecord.record_id must not be empty",
            ));
        }
        if self.skill_name.trim().is_empty() {
            return Err(Error::validation(
                "ExperienceRecord.skill_name must not be empty",
            ));
        }
        if self.content.trim().is_empty() {
            return Err(Error::validation(
                "ExperienceRecord.content must not be empty",
            ));
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(Error::validation(format!(
                "ExperienceRecord.confidence must be finite and between 0.0 and 1.0, got {}",
                self.confidence
            )));
        }
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(Error::validation(format!(
                "ExperienceRecord.weight must be finite and non-negative, got {}",
                self.weight
            )));
        }
        Ok(())
    }

    /// The number of recorded attempts (`successes + failures`).
    #[must_use]
    pub fn attempts(&self) -> u64 {
        self.successes.saturating_add(self.failures)
    }

    /// The observed success rate in `0.0..=1.0`, or `0.0` with no attempts.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let attempts = self.attempts();
        if attempts == 0 {
            return 0.0;
        }
        self.successes as f64 / attempts as f64
    }

    /// The ranking score: `weight * confidence * (0.5 + success_rate)`.
    ///
    /// The `0.5` floor keeps a fresh record visible (half of its weighted
    /// confidence) instead of ranking it at zero, and fixes the score at
    /// `1.5 * weight * confidence` for a record with a perfect success rate.
    #[must_use]
    pub fn score(&self) -> f64 {
        self.weight * self.confidence * (0.5 + self.success_rate())
    }

    /// Returns the record with its skill name replaced.
    #[must_use]
    pub fn with_skill(mut self, skill_name: impl Into<String>) -> Self {
        self.skill_name = skill_name.into();
        self
    }

    /// Returns the record with its section replaced.
    #[must_use]
    pub fn with_section(mut self, section: ExperienceSection) -> Self {
        self.section = section;
        self
    }

    /// Returns the record with `keywords`, deduplicated and sorted.
    #[must_use]
    pub fn with_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut unique = std::collections::BTreeSet::new();
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

    /// Returns the record with its provenance replaced.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Returns the record with `confidence` replaced.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `confidence` is not finite or outside
    /// `0.0..=1.0`.
    pub fn with_confidence(mut self, confidence: f64) -> Result<Self> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(Error::validation(format!(
                "ExperienceRecord.confidence must be finite and between 0.0 and 1.0, got {confidence}"
            )));
        }
        self.confidence = confidence;
        Ok(self)
    }

    /// Returns the record with `weight` replaced.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `weight` is not finite or is negative.
    pub fn with_weight(mut self, weight: f64) -> Result<Self> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::validation(format!(
                "ExperienceRecord.weight must be finite and non-negative, got {weight}"
            )));
        }
        self.weight = weight;
        Ok(self)
    }

    /// Returns the record with one more hit and one more success.
    #[must_use]
    pub fn record_success(mut self) -> Self {
        self.hits = self.hits.saturating_add(1);
        self.successes = self.successes.saturating_add(1);
        self
    }

    /// Returns the record with one more hit and one more failure.
    #[must_use]
    pub fn record_failure(mut self) -> Self {
        self.hits = self.hits.saturating_add(1);
        self.failures = self.failures.saturating_add(1);
        self
    }

    /// Returns the record marked as shareable.
    #[must_use]
    pub fn mark_shared(mut self) -> Self {
        self.shared = true;
        self
    }
}

/// The retrieval request an [`ExperienceArchive`](crate::experience::ExperienceArchive)
/// answers.
///
/// Filters compose with AND; ranking is described by
/// [`ExperienceArchive::query`](crate::experience::ExperienceArchive::query).
/// `limit` of `0` means "no limit", otherwise at most `limit` matches are
/// returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceQuery {
    /// Free text to compare against record content.
    pub text: Option<String>,
    /// Keywords to match against record keywords.
    pub keywords: Vec<String>,
    /// Restrict to one skill.
    pub skill_name: Option<String>,
    /// Restrict to one section.
    pub section: Option<ExperienceSection>,
    /// Whether records marked `shared` are eligible.
    pub include_shared: bool,
    /// Maximum number of matches; `0` means unlimited.
    pub limit: usize,
}

impl Default for ExperienceQuery {
    fn default() -> Self {
        Self {
            text: None,
            keywords: Vec::new(),
            skill_name: None,
            section: None,
            include_shared: true,
            limit: 10,
        }
    }
}

impl ExperienceQuery {
    /// Creates an empty query: no filters, shared records included, limit 10.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the free text.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Sets the keywords.
    #[must_use]
    pub fn with_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.keywords = keywords.into_iter().map(Into::into).collect();
        self
    }

    /// Restricts the query to one skill.
    #[must_use]
    pub fn with_skill(mut self, skill_name: impl Into<String>) -> Self {
        self.skill_name = Some(skill_name.into());
        self
    }

    /// Restricts the query to one section.
    #[must_use]
    pub fn with_section(mut self, section: ExperienceSection) -> Self {
        self.section = Some(section);
        self
    }

    /// Sets whether shared records are eligible.
    #[must_use]
    pub fn include_shared(mut self, include_shared: bool) -> Self {
        self.include_shared = include_shared;
        self
    }

    /// Sets the match limit; `0` means unlimited.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Whether the query carries neither text nor keywords.
    #[must_use]
    pub fn is_filter_only(&self) -> bool {
        let text_empty = match self.text.as_deref() {
            Some(text) => text.trim().is_empty(),
            None => true,
        };
        text_empty
            && self
                .keywords
                .iter()
                .all(|keyword| keyword.trim().is_empty())
    }
}

/// One ranked query hit: the record, its score and the keywords that matched.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExperienceMatch {
    /// The matched record.
    pub record: ExperienceRecord,
    /// The ranking score; higher is better.
    pub score: f64,
    /// The query keywords found on the record, sorted ascending.
    pub matched_keywords: Vec<String>,
}

impl ExperienceMatch {
    /// Creates a match from its parts.
    #[must_use]
    pub fn new(record: ExperienceRecord, score: f64, matched_keywords: Vec<String>) -> Self {
        Self {
            record,
            score,
            matched_keywords,
        }
    }
}

/// Default provenance for a record built by hand.
fn default_source() -> String {
    "experience".to_owned()
}

/// Default for confidence and weight: fully trusted.
fn default_unit() -> f64 {
    1.0
}
