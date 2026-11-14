//! The online evolution orchestrator: precheck, draft, stage or persist.
//!
//! Ports the sequencing half of Python
//! `openjiuwen.agent_evolving.experience.online_orchestrator`: decide whether the
//! request can run at all, generate a draft from the skill document and the
//! evolution evidence, then either stage it for approval or persist it into the
//! experience archive.
//!
//! The Python orchestrator reads the skill store asynchronously; this port takes
//! the lookup outcome as plain booleans ([`EvolutionRequest::skill_exists`],
//! [`EvolutionRequest::has_definition`]) so the decision order stays pure and the
//! caller owns the store. Persisted records map the Python initial-score map
//! (`execution_failure` 0.65, `user_intent` 0.70, `script_artifact` 0.60,
//! `conversation_review` 0.50, default 0.6) onto
//! [`ExperienceRecord::confidence`], because a record has no separate score until
//! it is ranked.

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::experience::archive::{AddOutcome, ExperienceArchive};
use crate::experience::draft::{DraftExperience, EvolutionSubject, EvolutionTarget, EvolveDraft};
use crate::experience::drafter::{
    draft_keywords, existing_summaries, DraftInput, ExperienceDrafter,
};
use crate::experience::models::{ExperienceRecord, ExperienceSection};
use crate::signal::EvolutionSignal;
use crate::trajectory::Trajectory;

/// The Python initial record score per signal source.
pub const INITIAL_SCORE_BY_SIGNAL: [(&str, f64); 4] = [
    ("execution_failure", 0.65),
    ("user_intent", 0.70),
    ("script_artifact", 0.60),
    ("conversation_review", 0.50),
];

/// The score a record starts with when the source is not in
/// [`INITIAL_SCORE_BY_SIGNAL`].
pub const DEFAULT_INITIAL_SCORE: f64 = 0.6;

/// The provenance label used when the request carries no typed signal.
pub const DEFAULT_SOURCE: &str = "experience";

/// The initial confidence for a record drafted from `signal_type`.
///
/// The value becomes [`ExperienceRecord::confidence`]: the archive's ranking
/// formula applies the weight and the usage counters on top of it.
#[must_use]
pub fn initial_confidence(signal_type: &str) -> f64 {
    INITIAL_SCORE_BY_SIGNAL
        .iter()
        .find(|(name, _)| *name == signal_type.trim())
        .map_or(DEFAULT_INITIAL_SCORE, |(_, score)| *score)
}

/// The outcome status of one online evolution attempt.
///
/// The literals are Python's: `staged`, `auto_approved`,
/// `no_evolution_no_records`, `skipped_no_input`, `skipped_skill_not_found`,
/// `skipped_skill_definition_not_found`, `persistence_failed` and
/// `generation_failed`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionStatus {
    /// The draft is waiting for approval.
    Staged,
    /// The draft was persisted without approval.
    AutoApproved,
    /// The draft carried no experience.
    #[default]
    NoEvolutionNoRecords,
    /// The skill name or the signal list was empty.
    SkippedNoInput,
    /// The skill does not exist in the caller's store.
    SkippedSkillNotFound,
    /// The skill exists but has no definition document.
    SkippedSkillDefinitionNotFound,
    /// The archive rejected a record.
    PersistenceFailed,
    /// The drafter returned an error.
    GenerationFailed,
}

impl EvolutionStatus {
    /// The status as its Python wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Staged => "staged",
            Self::AutoApproved => "auto_approved",
            Self::NoEvolutionNoRecords => "no_evolution_no_records",
            Self::SkippedNoInput => "skipped_no_input",
            Self::SkippedSkillNotFound => "skipped_skill_not_found",
            Self::SkippedSkillDefinitionNotFound => "skipped_skill_definition_not_found",
            Self::PersistenceFailed => "persistence_failed",
            Self::GenerationFailed => "generation_failed",
        }
    }

    /// Whether the attempt stopped before draft generation.
    #[must_use]
    pub const fn is_skipped(self) -> bool {
        matches!(
            self,
            Self::SkippedNoInput
                | Self::SkippedSkillNotFound
                | Self::SkippedSkillDefinitionNotFound
        )
    }
}

/// The structured result of one online evolution attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OnlineEvolutionResult {
    /// What happened.
    pub status: EvolutionStatus,
    /// The subject the draft targeted; `None` only for
    /// [`EvolutionStatus::SkippedNoInput`], where no skill name was usable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<EvolutionSubject>,
    /// The staged drafts; at most one today, kept as a list so a future
    /// multi-draft staging step keeps the result shape.
    #[serde(default)]
    pub staged: Vec<EvolveDraft>,
    /// The ids the archive stored or merged into, in draft order.
    #[serde(default)]
    pub committed: Vec<String>,
    /// A human-readable explanation, mirroring Python's `message`.
    #[serde(default)]
    pub reason: String,
}

impl OnlineEvolutionResult {
    /// Builds a result from its parts.
    #[must_use]
    pub fn new(
        status: EvolutionStatus,
        subject: Option<EvolutionSubject>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            status,
            subject,
            staged: Vec::new(),
            committed: Vec::new(),
            reason: reason.into(),
        }
    }

    /// Whether the result carries neither a staged draft nor a stored record.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.staged.is_empty() && self.committed.is_empty()
    }
}

/// The borrowed input of one [`OnlineEvolutionOrchestrator::evolve`] call.
///
/// The two lookup flags model Python's `EvolutionStore` queries. They default to
/// `true` in [`EvolutionRequest::new`], so the common case needs no store call.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionRequest<'a> {
    /// The skill to evolve; must be non-empty for the pipeline to run.
    pub skill_name: &'a str,
    /// The current `SKILL.md` content.
    pub skill_content: &'a str,
    /// The signals attributed to the skill; empty means nothing to learn from.
    pub signals: &'a [EvolutionSignal],
    /// The runs behind the signals.
    pub trajectories: &'a [Trajectory],
    /// The user's requested direction; may be empty.
    pub user_query: &'a str,
    /// Whether the draft waits for approval (`true`) or is persisted directly.
    pub requires_approval: bool,
    /// Whether the caller's store knows the skill.
    pub skill_exists: bool,
    /// Whether the caller's store has the skill definition document.
    pub has_definition: bool,
}

impl<'a> EvolutionRequest<'a> {
    /// Creates a request that stages its draft for approval.
    #[must_use]
    pub fn new(
        skill_name: &'a str,
        skill_content: &'a str,
        signals: &'a [EvolutionSignal],
    ) -> Self {
        Self {
            skill_name,
            skill_content,
            signals,
            trajectories: &[],
            user_query: "",
            requires_approval: true,
            skill_exists: true,
            has_definition: true,
        }
    }

    /// Sets the trajectories.
    #[must_use]
    pub fn with_trajectories(mut self, trajectories: &'a [Trajectory]) -> Self {
        self.trajectories = trajectories;
        self
    }

    /// Sets the user's requested direction.
    #[must_use]
    pub fn with_user_query(mut self, user_query: &'a str) -> Self {
        self.user_query = user_query;
        self
    }

    /// Sets whether the draft waits for approval.
    #[must_use]
    pub fn with_requires_approval(mut self, requires_approval: bool) -> Self {
        self.requires_approval = requires_approval;
        self
    }

    /// Sets whether the caller's store knows the skill.
    #[must_use]
    pub fn with_skill_exists(mut self, skill_exists: bool) -> Self {
        self.skill_exists = skill_exists;
        self
    }

    /// Sets whether the caller's store has the skill definition document.
    #[must_use]
    pub fn with_definition(mut self, has_definition: bool) -> Self {
        self.has_definition = has_definition;
        self
    }
}

/// Runs the online evolution pipeline against an owned experience archive.
///
/// The orchestrator owns the archive so persistence and its dedupe/merge rules
/// stay in one place; read it through [`Self::archive`] and seed it through
/// [`Self::with_archive`].
#[derive(Debug, Clone)]
pub struct OnlineEvolutionOrchestrator {
    /// The drafter that turns evidence into a normalized draft.
    drafter: ExperienceDrafter,
    /// The archive auto-approved drafts are persisted into.
    archive: ExperienceArchive,
}

impl Default for OnlineEvolutionOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineEvolutionOrchestrator {
    /// Builds an offline orchestrator with an empty archive.
    #[must_use]
    pub fn new() -> Self {
        Self {
            drafter: ExperienceDrafter::new(),
            archive: ExperienceArchive::new(),
        }
    }

    /// Builds an orchestrator over `drafter` and an empty archive.
    #[must_use]
    pub fn with_drafter(drafter: ExperienceDrafter) -> Self {
        Self {
            drafter,
            archive: ExperienceArchive::new(),
        }
    }

    /// Replaces the archive.
    #[must_use]
    pub fn with_archive(mut self, archive: ExperienceArchive) -> Self {
        self.archive = archive;
        self
    }

    /// The drafter this orchestrator uses.
    #[must_use]
    pub fn drafter(&self) -> &ExperienceDrafter {
        &self.drafter
    }

    /// The archive auto-approved drafts are written to.
    #[must_use]
    pub fn archive(&self) -> &ExperienceArchive {
        &self.archive
    }

    /// The archive, for inspection or maintenance after a run.
    #[must_use]
    pub fn archive_mut(&mut self) -> &mut ExperienceArchive {
        &mut self.archive
    }

    /// Runs the pipeline for `request`.
    ///
    /// The decision order is Python's: empty input, missing skill, missing
    /// definition, then generation, then staging or persistence. Every outcome
    /// except an unusable skill name is a status, never an error.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::validation`] only when the trimmed
    /// `request.skill_name` is blank *and* `request.signals` is empty — in that
    /// case the result is the `skipped_no_input` status instead, so this
    /// signature never actually fails today.
    pub async fn evolve(&mut self, request: EvolutionRequest<'_>) -> Result<OnlineEvolutionResult> {
        let skill_name = request.skill_name.trim();
        if skill_name.is_empty() || request.signals.is_empty() {
            return Ok(OnlineEvolutionResult::new(
                EvolutionStatus::SkippedNoInput,
                None,
                "online evolution skipped because skill_name or signals are empty",
            ));
        }
        let subject = EvolutionSubject::for_skill(skill_name)?;
        if !request.skill_exists {
            return Ok(OnlineEvolutionResult::new(
                EvolutionStatus::SkippedSkillNotFound,
                Some(subject),
                format!("online evolution skipped because skill '{skill_name}' does not exist"),
            ));
        }
        if !request.has_definition {
            return Ok(OnlineEvolutionResult::new(
                EvolutionStatus::SkippedSkillDefinitionNotFound,
                Some(subject),
                format!(
                    "online evolution skipped because skill '{skill_name}' is missing SKILL.md"
                ),
            ));
        }
        let existing = existing_summaries(self.archive.by_skill(skill_name));
        let input = DraftInput::new(skill_name, request.skill_content, request.signals)
            .with_trajectories(request.trajectories)
            .with_user_query(request.user_query)
            .with_existing(existing);
        let generated = self.drafter.draft(&input).await;
        Ok(self.complete(&request, generated))
    }

    /// Finishes the pipeline for an already-generated draft.
    ///
    /// [`Self::evolve`] calls this with the drafter's result; a caller that
    /// generated the draft elsewhere (a replay, a cached draft, an operator)
    /// can call it directly. A generation error becomes
    /// [`EvolutionStatus::GenerationFailed`], an empty draft becomes
    /// [`EvolutionStatus::NoEvolutionNoRecords`], and only
    /// `requires_approval == false` persists — staging otherwise.
    pub fn complete(
        &mut self,
        request: &EvolutionRequest<'_>,
        generated: Result<EvolveDraft>,
    ) -> OnlineEvolutionResult {
        let draft = match generated {
            Ok(draft) => draft,
            Err(error) => {
                let subject = EvolutionSubject::for_skill(request.skill_name).ok();
                return OnlineEvolutionResult::new(
                    EvolutionStatus::GenerationFailed,
                    subject,
                    error.to_string(),
                );
            }
        };
        let subject = draft.subject.clone();
        if draft.is_empty() {
            return OnlineEvolutionResult::new(
                EvolutionStatus::NoEvolutionNoRecords,
                Some(subject.clone()),
                format!("no applied updates for skill={}", subject.name),
            );
        }
        if request.requires_approval {
            let mut result = OnlineEvolutionResult::new(
                EvolutionStatus::Staged,
                Some(subject.clone()),
                format!("evolution request staged for skill={}", subject.name),
            );
            result.staged.push(draft);
            return result;
        }
        let source = signal_source(request.signals);
        let mut result = OnlineEvolutionResult::new(
            EvolutionStatus::AutoApproved,
            Some(subject.clone()),
            format!("evolution request auto-approved for skill={}", subject.name),
        );
        match self.persist(&subject, &draft, source) {
            Ok(committed) => result.committed = committed,
            Err(error) => {
                result.status = EvolutionStatus::PersistenceFailed;
                result.reason = error.to_string();
            }
        }
        result
    }

    /// Stores every draft experience, returning the resulting record ids.
    ///
    /// A record that merges into a near-duplicate reports the id it merged into,
    /// so callers see the archive's final identity rather than the draft's.
    fn persist(
        &mut self,
        subject: &EvolutionSubject,
        draft: &EvolveDraft,
        source: &str,
    ) -> Result<Vec<String>> {
        let mut records = Vec::with_capacity(draft.experiences.len());
        for experience in &draft.experiences {
            records.push(build_record(subject, experience, source)?);
        }
        let mut committed = Vec::with_capacity(records.len());
        for record in records {
            match self.archive.add(record) {
                Ok(AddOutcome::Inserted(id) | AddOutcome::Merged { into: id, .. }) => {
                    committed.push(id);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(committed)
    }
}

/// Builds the archive record for one draft experience.
///
/// The summary, the draft reason and any script fields ride along in
/// [`ExperienceRecord::metadata`], because the record has no dedicated fields
/// for them and dropping them would lose the approval view's information.
fn build_record(
    subject: &EvolutionSubject,
    experience: &DraftExperience,
    source: &str,
) -> Result<ExperienceRecord> {
    let record_id = experience_id(&subject.name, experience);
    let mut record = ExperienceRecord::new(
        record_id,
        subject.name.clone(),
        experience_section(experience.target),
        experience.content.clone(),
    )?
    .with_source(source)
    .with_keywords(draft_keywords(&experience.content))
    .with_confidence(initial_confidence(source))?;
    record.metadata.insert(
        "summary".to_owned(),
        Value::String(experience.summary.clone()),
    );
    if !experience.reason.is_empty() {
        record.metadata.insert(
            "reason".to_owned(),
            Value::String(experience.reason.clone()),
        );
    }
    for (key, value) in [
        ("script_filename", &experience.script_filename),
        ("script_language", &experience.script_language),
        ("script_purpose", &experience.script_purpose),
    ] {
        if let Some(value) = value {
            record
                .metadata
                .insert(key.to_owned(), Value::String(value.clone()));
        }
    }
    Ok(record)
}

/// A deterministic record id derived from the subject and the experience.
fn experience_id(skill_name: &str, experience: &DraftExperience) -> String {
    let key = format!(
        "{skill_name}\u{1f}{}\u{1f}{}\u{1f}{}",
        experience.target.as_str(),
        experience.section.as_str(),
        experience.content
    );
    format!("ev-{:016x}", fnv1a64(key.as_bytes()))
}

/// Projects a draft target onto the archive's section vocabulary.
fn experience_section(target: EvolutionTarget) -> ExperienceSection {
    match target {
        EvolutionTarget::Description => ExperienceSection::Description,
        EvolutionTarget::Body => ExperienceSection::Body,
        EvolutionTarget::Script => ExperienceSection::Script,
    }
}

/// The provenance label of the request: its first non-empty signal type.
fn signal_source(signals: &[EvolutionSignal]) -> &str {
    signals
        .iter()
        .map(|signal| signal.signal_type.trim())
        .find(|signal_type| !signal_type.is_empty())
        .unwrap_or(DEFAULT_SOURCE)
}

/// The FNV-1a 64-bit hash of `bytes`.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{EvolutionSignal, EvolutionTarget as SignalTarget};
    use openjiuwen_common::error::Error;

    fn signal_with(kind: &str, excerpt: &str, confidence: f64) -> EvolutionSignal {
        EvolutionSignal::new(kind, SignalTarget::Body, excerpt)
            .expect("valid signal")
            .with_confidence(confidence)
            .expect("valid confidence")
    }

    fn signal(excerpt: &str) -> EvolutionSignal {
        signal_with("execution_failure", excerpt, 1.0)
    }

    fn request(signals: &[EvolutionSignal]) -> EvolutionRequest<'_> {
        EvolutionRequest::new("pdf-report", "# PDF report", signals)
    }

    #[test]
    fn initial_confidence_matches_the_python_map() {
        assert_eq!(initial_confidence("execution_failure"), 0.65);
        assert_eq!(initial_confidence(" user_intent "), 0.70);
        assert_eq!(initial_confidence("script_artifact"), 0.60);
        assert_eq!(initial_confidence("conversation_review"), 0.50);
        assert_eq!(initial_confidence("task_completed"), DEFAULT_INITIAL_SCORE);
        assert_eq!(initial_confidence(""), DEFAULT_INITIAL_SCORE);
    }

    #[test]
    fn status_literals_stay_python_compatible() {
        let literals: Vec<&str> = [
            EvolutionStatus::Staged,
            EvolutionStatus::AutoApproved,
            EvolutionStatus::NoEvolutionNoRecords,
            EvolutionStatus::SkippedNoInput,
            EvolutionStatus::SkippedSkillNotFound,
            EvolutionStatus::SkippedSkillDefinitionNotFound,
            EvolutionStatus::PersistenceFailed,
            EvolutionStatus::GenerationFailed,
        ]
        .iter()
        .map(|status| status.as_str())
        .collect();
        assert_eq!(
            literals,
            vec![
                "staged",
                "auto_approved",
                "no_evolution_no_records",
                "skipped_no_input",
                "skipped_skill_not_found",
                "skipped_skill_definition_not_found",
                "persistence_failed",
                "generation_failed",
            ]
        );
        assert!(EvolutionStatus::SkippedNoInput.is_skipped());
        assert!(!EvolutionStatus::Staged.is_skipped());
    }

    #[tokio::test]
    async fn precheck_statuses_are_returned_in_python_order() {
        let mut orchestrator = OnlineEvolutionOrchestrator::new();
        let signals = [signal("the parser timed out")];
        let none: [EvolutionSignal; 0] = [];
        let skipped = orchestrator.evolve(request(&none)).await.expect("result");
        assert_eq!(skipped.status, EvolutionStatus::SkippedNoInput);
        assert!(skipped.subject.is_none());

        let missing = orchestrator
            .evolve(request(&signals).with_skill_exists(false))
            .await
            .expect("result");
        assert_eq!(missing.status, EvolutionStatus::SkippedSkillNotFound);

        let undefined = orchestrator
            .evolve(request(&signals).with_definition(false))
            .await
            .expect("result");
        assert_eq!(
            undefined.status,
            EvolutionStatus::SkippedSkillDefinitionNotFound
        );
        assert!(orchestrator.archive().is_empty());
    }

    #[tokio::test]
    async fn staging_and_auto_approval_split_persistence() {
        let signals = [signal("the parser timed out on large PDFs")];
        let mut orchestrator = OnlineEvolutionOrchestrator::new();
        let staged = orchestrator
            .evolve(request(&signals))
            .await
            .expect("result");
        assert_eq!(staged.status, EvolutionStatus::Staged);
        assert_eq!(staged.staged.len(), 1);
        assert_eq!(staged.committed.len(), 0);
        assert!(orchestrator.archive().is_empty());

        let auto = orchestrator
            .evolve(request(&signals).with_requires_approval(false))
            .await
            .expect("result");
        assert_eq!(auto.status, EvolutionStatus::AutoApproved);
        assert_eq!(auto.committed.len(), 1);
        assert_eq!(auto.staged.len(), 0);
        let records = orchestrator.archive().records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].record_id, auto.committed[0]);
        assert_eq!(records[0].source, "execution_failure");
        assert_eq!(records[0].confidence, 0.65);
        assert_eq!(records[0].section, ExperienceSection::Body);
        assert!(records[0].metadata.contains_key("summary"));

        // The same evidence again merges into the same record.
        let repeat = orchestrator
            .evolve(request(&signals).with_requires_approval(false))
            .await
            .expect("result");
        assert_eq!(repeat.committed, auto.committed);
        assert_eq!(orchestrator.archive().len(), 1);
    }

    #[tokio::test]
    async fn low_confidence_evidence_produces_no_records() {
        let signals = [signal_with("execution_failure", "weak hint", 0.1)];
        let mut orchestrator = OnlineEvolutionOrchestrator::new();
        let result = orchestrator
            .evolve(request(&signals).with_requires_approval(false))
            .await
            .expect("result");
        assert_eq!(result.status, EvolutionStatus::NoEvolutionNoRecords);
        assert!(result.reason.contains("no applied updates"));
        assert!(orchestrator.archive().is_empty());
    }

    #[test]
    fn generation_and_persistence_failures_are_statuses() {
        let signals = [signal("the parser timed out")];
        let request = request(&signals);
        let mut orchestrator = OnlineEvolutionOrchestrator::new();

        let generated = orchestrator.complete(&request, Err(Error::llm("provider down")));
        assert_eq!(generated.status, EvolutionStatus::GenerationFailed);
        assert_eq!(generated.reason, "llm: provider down");

        let subject = EvolutionSubject::for_skill("pdf-report").expect("valid subject");
        let blank = EvolveDraft::new(
            subject,
            vec![DraftExperience {
                summary: "summary".to_owned(),
                content: "   ".to_owned(),
                ..DraftExperience::default()
            }],
        );
        let persisted =
            orchestrator.complete(&request.clone().with_requires_approval(false), Ok(blank));
        assert_eq!(persisted.status, EvolutionStatus::PersistenceFailed);
        assert_eq!(persisted.committed.len(), 0);
    }

    #[test]
    fn result_round_trips_and_serde_is_tolerant() {
        let result =
            OnlineEvolutionResult::new(EvolutionStatus::SkippedNoInput, None, "nothing to do");
        let encoded = serde_json::to_string(&result).expect("serializable");
        let decoded: OnlineEvolutionResult =
            serde_json::from_str(&encoded).expect("deserializable");
        assert_eq!(decoded, result);
        let status: EvolutionStatus =
            serde_json::from_str("\"skipped_skill_not_found\"").expect("literal");
        assert_eq!(status, EvolutionStatus::SkippedSkillNotFound);
        assert!(result.is_empty());
    }
}
