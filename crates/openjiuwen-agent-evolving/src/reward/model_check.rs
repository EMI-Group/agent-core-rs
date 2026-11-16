//! The judged scoring layer: claim statuses to a 0-100 score, and the report a
//! reward pack consumes.
//!
//! Ports the arithmetic of Python `openjiuwen.agent_evolving.reward.scoring` and
//! the judge stage of `openjiuwen.agent_evolving.reward.pipeline`: each criterion
//! is judged from its checks, the statuses are scored in code (never by the
//! model), and the report's total is the mean of the scored criteria.
//!
//! The overall metric carries Python's version stamp, [`SCHEMA_VERSION`]
//! (`"reward/3"`). A failed judge call never invents a score: the criterion's
//! claims become [`CriterionStatus::Unknown`] with the failure as their reason.
//!
//! Scores round half away from zero (`f64::round`), where Python rounds halves to
//! even; the difference is at most one point on an exact half.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::check::CheckTool;
use super::evidence::{
    CriterionKind, EvidenceCriterion, EvidenceGathered, EvidenceTask, GatheredCheck,
};
use super::models::RewardSignal;

/// The version stamped on the overall metric (Python `SCHEMA_VERSION`).
pub const SCHEMA_VERSION: &str = "reward/3";

/// The share of a criterion's score that its floor claims carry (Python
/// `FLOOR_WEIGHT`).
///
/// It matters when floor and challenge claims are scored together; this model
/// gives each criterion one [`CriterionKind`], so only one side is populated and
/// the weighted blend reduces to that side's mean.
pub const FLOOR_WEIGHT: f64 = 0.5;

/// A failed floor claim caps the criterion's score here (Python `FLOOR_FAIL_CAP`).
pub const FLOOR_FAIL_CAP: u32 = 25;

/// The dimension a judged report scores (the default pack's highest weight).
pub const REWARD_CHECK_DIMENSION: &str = "correctness";

/// The source stamped on a signal built from a judged report.
pub const REWARD_CHECK_SOURCE: &str = "reward_check";

/// The judge's instructions: judge one criterion's claims from the checks.
///
/// Python prepends `READER_NOTE` and appends the art-judge variant; both are
/// out of scope for this offline port, and the reasons here are still written for
/// the user who asked for the work.
pub const JUDGE_PROMPT: &str = "\
You judge each claim of one criterion about an agent's delivered work. Evidence: the checks that \
were run on the work (tool, output, values, errors), the workspace excerpts, the agent's final \
reply and the inputs when given. Files show intent; checks show behaviour.

Judge only this criterion's claims, each on its own subject: how the work looks belongs to another \
criterion, so a visual weakness never lowers a claim about behaviour, content or data. A check that \
failed to run is not a defect of the work; a check that ran and showed a failure is. When the work \
has no files, the final reply is the outcome. A skipped check settles nothing: a page whose source \
was refused stays unknown unless a browser check saw it, and a picture check reports the file's \
format and size, not what it shows.

Status per claim: met (evidence shows the whole claim), partial (evidence shows the work does only \
part of it), unmet (evidence shows it fails), unknown (no evidence either way). A claim about \
behaviour is met only when a check ran the work and observed it there; source code that implements \
it, with no run reaching it, makes it partial, with the reason \"seen in the source, not observed in \
a run\". Challenge claims are judged against the best work of this kind, not against the Task: mark \
one met only if the evidence shows it fully and you cannot name a concrete way top work does it \
better; competent but ordinary work is partial, and when in doubt choose partial.

The deliverable's own statements about itself are not evidence that they are true: a claim that its \
numbers, facts or counts are right is met only when a check recomputed or looked them up in the data \
they rest on. Limits of the evaluation are not defects of the work.

For each claim give \"reason\" (what you saw) in simplified technical style: one short sentence, \
active voice, one fact, no hedging. Then give \"summary\": the criterion as a whole in one or two \
short sentences (what works, the main gap), without advice.

Return JSON: {\"claims\": [{\"index\": 0, \"status\": \"met\", \"reason\": \"...\"}], \"summary\": \"...\"}";

/// The status of one claim, as the judge decides it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionStatus {
    /// The evidence shows the whole claim.
    Met,
    /// The evidence shows the work does only part of the claim.
    Partial,
    /// The evidence shows the claim fails.
    Unmet,
    /// No evidence either way; left out of the arithmetic.
    #[default]
    Unknown,
}

impl CriterionStatus {
    /// The Python spelling of the status.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Met => "met",
            Self::Partial => "partial",
            Self::Unmet => "unmet",
            Self::Unknown => "unknown",
        }
    }

    /// The arithmetic value (`met` `1.0`, `partial` `0.5`, `unmet` `0.0`), or
    /// `None` for [`Unknown`](CriterionStatus::Unknown), which is left out.
    #[must_use]
    pub fn value(self) -> Option<f64> {
        match self {
            Self::Met => Some(1.0),
            Self::Partial => Some(0.5),
            Self::Unmet => Some(0.0),
            Self::Unknown => None,
        }
    }

    /// Reads a status, tolerating case and surrounding quotes; anything else is
    /// `None`.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().trim_matches('"').to_ascii_lowercase().as_str() {
            "met" => Some(Self::Met),
            "partial" => Some(Self::Partial),
            "unmet" => Some(Self::Unmet),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }
}

/// Scores one criterion's claim statuses on `0..=100`, or `None` when nothing
/// was settled.
///
/// `met` is `1.0`, `partial` `0.5`, `unmet` `0.0`; unknown claims are left out. A
/// criterion with no settled claim is unscored: nothing was seen, so nothing is
/// scored. The score is `round(100 * mean)` of the settled claims, and any
/// [`unmet`](CriterionStatus::Unmet) claim of a [`Floor`](CriterionKind::Floor)
/// criterion caps it at [`FLOOR_FAIL_CAP`].
///
/// Python groups the claims by their per-claim level and blends the two group
/// means `FLOOR_WEIGHT : 1 - FLOOR_WEIGHT`; this model carries one level per
/// criterion, so exactly one group is populated and the blend reduces to that
/// group's mean.
#[must_use]
pub fn score_criterion(kind: CriterionKind, statuses: &[CriterionStatus]) -> Option<u32> {
    let settled: Vec<f64> = statuses
        .iter()
        .filter_map(|status| status.value())
        .collect();
    if settled.is_empty() {
        return None;
    }
    let (floor, challenge): (Vec<f64>, Vec<f64>) = match kind {
        CriterionKind::Floor => (settled, Vec::new()),
        CriterionKind::ProactiveChallenge => (Vec::new(), settled),
    };
    let fraction = match (floor.is_empty(), challenge.is_empty()) {
        (true, true) => return None,
        (false, true) => mean(&floor),
        (true, false) => mean(&challenge),
        (false, false) => FLOOR_WEIGHT * mean(&floor) + (1.0 - FLOOR_WEIGHT) * mean(&challenge),
    };
    let score = round_score(100.0 * fraction);
    let floor_failed = kind == CriterionKind::Floor && statuses.contains(&CriterionStatus::Unmet);
    Some(if floor_failed {
        score.min(FLOOR_FAIL_CAP)
    } else {
        score
    })
}

/// The mean of a non-empty slice; callers guarantee it is not empty.
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Rounds to a whole point, half away from zero, clamped to `0..=100`.
fn round_score(value: f64) -> u32 {
    value.round().clamp(0.0, 100.0) as u32
}

/// The report total: the mean of the scored criteria, `None` when none was
/// scored.
///
/// Like [`score_criterion`], the mean rounds half away from zero.
#[must_use]
pub fn total(scores: &[Option<u32>]) -> Option<u32> {
    let values: Vec<f64> = scores
        .iter()
        .filter_map(|score| score.map(f64::from))
        .collect();
    if values.is_empty() {
        None
    } else {
        Some(round_score(mean(&values)))
    }
}

/// One judged criterion: what was asked, what the judge said and what it scores.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CriterionReport {
    /// The criterion id the checks named.
    pub id: String,
    /// The criterion name.
    pub name: String,
    /// The claim level.
    pub kind: CriterionKind,
    /// The criterion's claims as the judge read them, one numbered line each.
    pub rubric: String,
    /// The criterion's overall status, derived from its score.
    pub status: CriterionStatus,
    /// The score in `0..=100`, or `None` when no claim was settled.
    pub score: Option<u32>,
    /// `(claim text, status, reason)` per claim, in judging order.
    pub claims: Vec<(String, CriterionStatus, String)>,
}

/// The judged result of one evidence run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CheckRewardReport {
    /// The mean of the scored criteria in `0..=100`, or `None` when nothing was
    /// scored.
    pub total: Option<u32>,
    /// One report per criterion, in task order.
    pub criteria: Vec<CriterionReport>,
    /// The judges' summaries, or a generated summary when they wrote none.
    pub summary: String,
    /// How many distinct checks the run gathered.
    pub used_checks: usize,
}

impl CheckRewardReport {
    /// The report for the criterion `id`, when it was judged.
    #[must_use]
    pub fn criterion(&self, id: &str) -> Option<&CriterionReport> {
        self.criteria.iter().find(|report| report.id == id)
    }
}

/// Judges an evidence run one criterion at a time and scores it in code.
#[derive(Clone)]
pub struct ModelRewardJudge {
    model: Arc<dyn Model>,
    request: ModelRequestConfig,
}

impl std::fmt::Debug for ModelRewardJudge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelRewardJudge")
            .field("model", &self.model.model_name())
            .field("request", &self.request)
            .finish()
    }
}

impl ModelRewardJudge {
    /// Creates a judge over `model`, sending `request` with every call.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, request: ModelRequestConfig) -> Self {
        Self { model, request }
    }

    /// The model the judge asks.
    #[must_use]
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// The request configuration sent with every judge call.
    #[must_use]
    pub fn request(&self) -> &ModelRequestConfig {
        &self.request
    }

    /// Judges `task` from `gathered` and scores the verdicts.
    ///
    /// Each criterion gets its own call and its own prompt: the task, the
    /// criterion, the checks that serve it, the workspace files, `final_reply`
    /// (or the task's own reply when the argument is blank) and the inputs.
    /// A verdict that is not JSON, or a call that fails, leaves that criterion
    /// [`Unknown`](CriterionStatus::Unknown) with the failure as the reason.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `task` declares no criterion: there is nothing
    /// to judge, and a report over no criteria would carry no score at all.
    pub async fn judge(
        &self,
        task: &EvidenceTask,
        gathered: &EvidenceGathered,
        final_reply: &str,
    ) -> Result<CheckRewardReport> {
        if task.criteria.is_empty() {
            return Err(Error::validation(
                "a reward report needs at least one criterion",
            ));
        }
        let reply = if final_reply.trim().is_empty() {
            task.final_reply.as_str()
        } else {
            final_reply
        };
        let mut criteria = Vec::with_capacity(task.criteria.len());
        let mut summaries = Vec::new();
        for criterion in &task.criteria {
            let (report, summary) = self.judge_criterion(task, criterion, gathered, reply).await;
            criteria.push(report);
            if let Some(summary) = summary {
                summaries.push(summary);
            }
        }
        let scores: Vec<Option<u32>> = criteria.iter().map(|report| report.score).collect();
        let total = total(&scores);
        let summary = if summaries.is_empty() {
            generated_summary(&criteria, total)
        } else {
            summaries.join("\n")
        };
        Ok(CheckRewardReport {
            total,
            criteria,
            summary,
            used_checks: gathered.distinct().len(),
        })
    }

    /// Judges one criterion; a failure is reported on the criterion, not raised.
    async fn judge_criterion(
        &self,
        task: &EvidenceTask,
        criterion: &EvidenceCriterion,
        gathered: &EvidenceGathered,
        final_reply: &str,
    ) -> (CriterionReport, Option<String>) {
        let checks = gathered.checks_for(&criterion.id);
        let rubric = rubric(criterion);
        let messages = [
            BaseMessage::system(JUDGE_PROMPT),
            BaseMessage::user(criterion_message(task, criterion, &checks, final_reply)),
        ];
        match self.model.invoke(&messages, &self.request).await {
            Ok(response) => {
                let content = response
                    .message()
                    .map_or_else(String::new, |message| message.text().into_owned());
                match json_payload(&content) {
                    Some(verdict) => verdict_report(criterion, rubric, &verdict),
                    None => {
                        failure_report(criterion, rubric, "the judge reply was not a JSON verdict")
                    }
                }
            }
            Err(error) => failure_report(
                criterion,
                rubric,
                &format!("the judge call failed: {error}"),
            ),
        }
    }
}

/// Turns a judged report into the reward signal a [`RewardPack`] aggregates.
///
/// The signal measures [`REWARD_CHECK_DIMENSION`] (`correctness`) on `0.0..=1.0`
/// with [`REWARD_CHECK_SOURCE`] as its source and the report summary as its
/// evidence. `None` when nothing was scored, so an unjudged run contributes no
/// signal instead of a made-up zero.
///
/// [`RewardPack`]: crate::reward::RewardPack
#[must_use]
pub fn to_reward_signal(report: &CheckRewardReport) -> Option<RewardSignal> {
    let total = report.total?;
    Some(
        RewardSignal::new(REWARD_CHECK_DIMENSION, f64::from(total) / 100.0)
            .with_source(REWARD_CHECK_SOURCE)
            .with_evidence(report.summary.clone()),
    )
}

/// The numbered claims of a criterion, as the judge reads them.
fn rubric(criterion: &EvidenceCriterion) -> String {
    criterion
        .claims
        .iter()
        .map(|claim| format!("{}. {}", claim.index, claim.text))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The user message of one criterion's judge call.
fn criterion_message(
    task: &EvidenceTask,
    criterion: &EvidenceCriterion,
    checks: &[&GatheredCheck],
    final_reply: &str,
) -> String {
    let checks: Vec<Value> = checks
        .iter()
        .map(|check| {
            json!({
                "id": check.id,
                "tool": check.tool.map(CheckTool::as_str),
                "note": "",
                "skipped": check.skipped,
                "error": check.error,
                "output": check.result,
            })
        })
        .collect();
    json!({
        "task": task.task,
        "criterion": criterion,
        "checks": checks,
        "workspace": task.workspace_files,
        "final_reply": final_reply,
        "inputs": task.inputs_excerpt,
        "metric": SCHEMA_VERSION,
    })
    .to_string()
}

/// Builds a report from a parsed verdict; unknown or missing rows are `unknown`.
fn verdict_report(
    criterion: &EvidenceCriterion,
    rubric: String,
    verdict: &Value,
) -> (CriterionReport, Option<String>) {
    let rows = claim_rows(verdict);
    let claims: Vec<(String, CriterionStatus, String)> = criterion
        .claims
        .iter()
        .map(|claim| {
            let row = rows.get(&claim.index);
            let status = row
                .and_then(|row| row.get("status"))
                .and_then(Value::as_str)
                .and_then(CriterionStatus::parse)
                .unwrap_or(CriterionStatus::Unknown);
            let reason = row
                .and_then(|row| row.get("reason"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            (claim.text.clone(), status, reason)
        })
        .collect();
    let statuses: Vec<CriterionStatus> = claims.iter().map(|(_, status, _)| *status).collect();
    let score = score_criterion(criterion.kind, &statuses);
    let summary = verdict
        .get("summary")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|summary| !summary.is_empty())
        .map(str::to_owned);
    (criterion_report(criterion, rubric, score, claims), summary)
}

/// A criterion whose judge failed: every claim unknown, with the failure as the
/// reason.
fn failure_report(
    criterion: &EvidenceCriterion,
    rubric: String,
    failure: &str,
) -> (CriterionReport, Option<String>) {
    let claims: Vec<(String, CriterionStatus, String)> = criterion
        .claims
        .iter()
        .map(|claim| {
            (
                claim.text.clone(),
                CriterionStatus::Unknown,
                failure.to_owned(),
            )
        })
        .collect();
    let report = criterion_report(criterion, rubric, None, claims);
    let summary = format!("{}: unjudged ({failure})", criterion.id);
    (report, Some(summary))
}

/// Assembles a [`CriterionReport`] and derives its overall status.
fn criterion_report(
    criterion: &EvidenceCriterion,
    rubric: String,
    score: Option<u32>,
    claims: Vec<(String, CriterionStatus, String)>,
) -> CriterionReport {
    CriterionReport {
        id: criterion.id.clone(),
        name: criterion.name.clone(),
        kind: criterion.kind,
        rubric,
        status: status_from_score(score),
        score,
        claims,
    }
}

/// The overall status implied by a score.
///
/// A scored criterion is `met` at `100`, `unmet` at `0` and `partial` between; an
/// unscored one is `unknown`.
fn status_from_score(score: Option<u32>) -> CriterionStatus {
    match score {
        None => CriterionStatus::Unknown,
        Some(100) => CriterionStatus::Met,
        Some(0) => CriterionStatus::Unmet,
        Some(_) => CriterionStatus::Partial,
    }
}

/// The verdict rows by claim index; a row without an index keeps its position.
fn claim_rows(verdict: &Value) -> BTreeMap<usize, &Value> {
    let mut rows = BTreeMap::new();
    let Some(items) = verdict.get("claims").and_then(Value::as_array) else {
        return rows;
    };
    for (position, row) in items.iter().enumerate() {
        if !row.is_object() {
            continue;
        }
        let index = row
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .unwrap_or(position);
        rows.insert(index, row);
    }
    rows
}

/// A summary for a run whose judges wrote none.
fn generated_summary(criteria: &[CriterionReport], total: Option<u32>) -> String {
    if criteria.is_empty() {
        return format!("{SCHEMA_VERSION}: no criterion was judged");
    }
    let scored = criteria
        .iter()
        .filter(|report| report.score.is_some())
        .count();
    match total {
        Some(total) => format!(
            "{} of {} criteria scored; total {total}/{SCHEMA_VERSION}",
            scored,
            criteria.len()
        ),
        None => format!("no criterion could be scored (0 of {})", criteria.len()),
    }
}

/// Reads the first JSON value out of a model reply, tolerating fenced blocks and
/// surrounding prose; `None` when there is none.
pub(crate) fn json_payload(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }
    let body = strip_fence(trimmed);
    if body != trimmed {
        if let Ok(value) = serde_json::from_str::<Value>(body) {
            return Some(value);
        }
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&trimmed[start..=end]).ok()
}

/// Removes a Markdown code fence, when the reply is wrapped in one.
fn strip_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let Some(newline) = rest.find('\n') else {
        return text;
    };
    let body = &rest[newline + 1..];
    body.strip_suffix("```").unwrap_or(body).trim()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use futures::StreamExt;
    use openjiuwen_common::error::Error;
    use openjiuwen_core::foundation::llm::{ChatResponse, Choice, StreamChunk};

    use super::*;
    use crate::reward::default_pack;

    /// A model that serves queued replies and counts its calls.
    struct FakeModel {
        replies: Mutex<VecDeque<Result<String>>>,
        calls: AtomicUsize,
    }

    impl FakeModel {
        fn new(replies: Vec<Result<String>>) -> Arc<Self> {
            Arc::new(Self {
                replies: Mutex::new(replies.into()),
                calls: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Model for FakeModel {
        fn provider(&self) -> &'static str {
            "fake"
        }

        fn model_name(&self) -> &'static str {
            "fake-judge"
        }

        async fn invoke(
            &self,
            _messages: &[BaseMessage],
            _request: &ModelRequestConfig,
        ) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.replies.lock().unwrap().pop_front() {
                Some(Ok(content)) => Ok(ChatResponse {
                    choices: vec![Choice {
                        message: BaseMessage::assistant(content),
                        ..Choice::default()
                    }],
                    ..ChatResponse::default()
                }),
                Some(Err(error)) => Err(error),
                None => Err(Error::validation("the fake judge ran out of replies")),
            }
        }

        fn stream(
            &self,
            _messages: Vec<BaseMessage>,
            _request: ModelRequestConfig,
        ) -> BoxStream<'static, Result<StreamChunk>> {
            futures::stream::empty().boxed()
        }
    }

    fn task(kind: CriterionKind) -> EvidenceTask {
        EvidenceTask::new("Summarise the notes")
            .with_workspace_files(["notes.txt"])
            .with_final_reply("See notes.txt")
            .with_criteria(vec![EvidenceCriterion::new("c1", "Content")
                .with_kind(kind)
                .with_claim("Names the score")
                .with_claim("Names the date")])
    }

    fn gathered() -> EvidenceGathered {
        EvidenceGathered {
            checks: vec![GatheredCheck {
                id: String::from("c1"),
                tool: Some(CheckTool::Read),
                criteria: vec![String::from("c1")],
                ok: true,
                repeat: false,
                result: String::from("path: notes.txt\nmatches: 1\n1: Score: 7"),
                screenshots: Vec::new(),
                skipped: None,
                error: None,
            }],
            rounds: 2,
            deduped: 0,
        }
    }

    #[test]
    fn statuses_map_to_their_arithmetic_values() {
        assert_eq!(CriterionStatus::Met.value(), Some(1.0));
        assert_eq!(CriterionStatus::Partial.value(), Some(0.5));
        assert_eq!(CriterionStatus::Unmet.value(), Some(0.0));
        assert_eq!(CriterionStatus::Unknown.value(), None);
        assert_eq!(CriterionStatus::parse(" MET "), Some(CriterionStatus::Met));
        assert_eq!(
            CriterionStatus::parse("\"partial\""),
            Some(CriterionStatus::Partial)
        );
        assert_eq!(CriterionStatus::parse("great"), None);
        assert_eq!(CriterionStatus::default(), CriterionStatus::Unknown);
    }

    #[test]
    fn score_criterion_matches_the_python_arithmetic() {
        use CriterionStatus::{Met, Partial, Unknown, Unmet};
        let floor = CriterionKind::Floor;
        let challenge = CriterionKind::ProactiveChallenge;

        // Nothing settled is unscored.
        assert_eq!(score_criterion(floor, &[]), None);
        assert_eq!(score_criterion(floor, &[Unknown, Unknown]), None);
        assert_eq!(score_criterion(challenge, &[Unknown]), None);

        // Floor claims: met 100, half met 75, a third met 50.
        assert_eq!(score_criterion(floor, &[Met, Met]), Some(100));
        assert_eq!(score_criterion(floor, &[Met, Partial]), Some(75));
        assert_eq!(score_criterion(floor, &[Met, Partial, Unmet]), Some(25));
        assert_eq!(score_criterion(floor, &[Met, Met, Partial]), Some(83));

        // Unknown claims are left out, floor or challenge.
        assert_eq!(score_criterion(floor, &[Unknown, Met]), Some(100));
        assert_eq!(
            score_criterion(challenge, &[Met, Unknown, Partial]),
            Some(75)
        );

        // Challenge claims are scored without the floor cap.
        assert_eq!(score_criterion(challenge, &[Met, Partial]), Some(75));
        assert_eq!(score_criterion(challenge, &[Unmet]), Some(0));

        // Round half away from zero: 0.125 -> 12.5 -> 13 (Python gives 12).
        assert_eq!(
            score_criterion(floor, &[Partial, Unmet, Unmet, Unmet]),
            Some(13)
        );
    }

    #[test]
    fn a_failed_floor_claim_caps_the_score() {
        use CriterionStatus::{Met, Partial, Unmet};
        let floor = CriterionKind::Floor;
        let challenge = CriterionKind::ProactiveChallenge;
        let statuses = [Met, Met, Met, Met, Met, Met, Met, Partial, Unmet];
        assert_eq!(score_criterion(floor, &statuses), Some(FLOOR_FAIL_CAP));
        assert_eq!(score_criterion(floor, &[Met, Unmet]), Some(FLOOR_FAIL_CAP));
        assert_eq!(score_criterion(floor, &[Unmet]), Some(0));
        // The same statuses on a challenge criterion are not capped.
        assert!(score_criterion(challenge, &statuses).unwrap() > FLOOR_FAIL_CAP);
    }

    #[test]
    fn total_is_the_mean_of_the_scored_criteria() {
        assert_eq!(total(&[]), None);
        assert_eq!(total(&[None, None]), None);
        assert_eq!(total(&[Some(100)]), Some(100));
        assert_eq!(total(&[Some(100), Some(50)]), Some(75));
        assert_eq!(total(&[Some(100), Some(50), None]), Some(75));
        // 100 + 75 + 25 = 200 / 3 = 66.67 -> 67.
        assert_eq!(total(&[Some(100), Some(75), Some(25)]), Some(67));
        // The mean rounds half away from zero: 12.5 -> 13.
        assert_eq!(total(&[Some(0), Some(25)]), Some(13));
    }

    #[tokio::test]
    async fn judge_scores_a_verdict_and_builds_a_signal() {
        let model = FakeModel::new(vec![Ok(String::from(
            r#"{"claims": [{"index": 0, "status": "met", "reason": "the score is stated"},
                          {"index": 1, "status": "partial", "reason": "the month is missing"}],
                "summary": "Content is mostly there; the date is vague."}"#,
        ))]);
        let judge = ModelRewardJudge::new(model.clone(), ModelRequestConfig::default());
        let report = judge
            .judge(&task(CriterionKind::Floor), &gathered(), "See notes.txt")
            .await
            .unwrap();

        assert_eq!(model.calls(), 1);
        assert_eq!(report.total, Some(75));
        assert_eq!(report.used_checks, 1);
        assert_eq!(
            report.summary,
            "Content is mostly there; the date is vague."
        );
        let criterion = report.criterion("c1").unwrap();
        assert_eq!(criterion.status, CriterionStatus::Partial);
        assert_eq!(criterion.score, Some(75));
        assert_eq!(criterion.rubric, "0. Names the score\n1. Names the date");
        assert_eq!(criterion.claims[0].1, CriterionStatus::Met);
        assert_eq!(criterion.claims[1].2, "the month is missing");
        assert!(report.criterion("c2").is_none());

        let signal = to_reward_signal(&report).unwrap();
        assert_eq!(signal.dimension, REWARD_CHECK_DIMENSION);
        assert!((signal.value - 0.75).abs() < f64::EPSILON);
        assert_eq!(signal.source.as_deref(), Some(REWARD_CHECK_SOURCE));
        assert_eq!(signal.evidence.as_deref(), Some(report.summary.as_str()));
        assert!(signal.validate().is_ok());

        let summary = default_pack().aggregate(&[signal]).unwrap();
        assert!((summary.total() - 0.75).abs() < 1e-9);
        assert!(summary.passed());
    }

    #[tokio::test]
    async fn judge_tolerates_garbage_and_scores_nothing() {
        let model = FakeModel::new(vec![Ok(String::from("I think it is quite good."))]);
        let judge = ModelRewardJudge::new(model, ModelRequestConfig::default());
        let report = judge
            .judge(&task(CriterionKind::Floor), &gathered(), "")
            .await
            .unwrap();
        assert_eq!(report.total, None);
        assert_eq!(report.criteria[0].status, CriterionStatus::Unknown);
        assert!(report.criteria[0].score.is_none());
        assert!(report.criteria[0]
            .claims
            .iter()
            .all(|(_, status, reason)| *status == CriterionStatus::Unknown
                && reason.contains("not a JSON verdict")));
        assert!(report.summary.contains("unjudged"));
        assert!(to_reward_signal(&report).is_none());

        // A verdict with no claims is judged, but nothing is settled.
        let model = FakeModel::new(vec![Ok(String::from("{}"))]);
        let judge = ModelRewardJudge::new(model, ModelRequestConfig::default());
        let report = judge
            .judge(&task(CriterionKind::Floor), &gathered(), "")
            .await
            .unwrap();
        assert_eq!(report.total, None);
        assert!(report.criteria[0].score.is_none());
    }

    #[tokio::test]
    async fn a_failed_judge_call_is_unknown_not_a_panic() {
        let model = FakeModel::new(vec![Err(Error::llm("provider down"))]);
        let judge = ModelRewardJudge::new(model, ModelRequestConfig::default());
        let report = judge
            .judge(&task(CriterionKind::Floor), &gathered(), "")
            .await
            .unwrap();
        assert_eq!(report.total, None);
        let reason = &report.criteria[0].claims[0].2;
        assert!(reason.contains("the judge call failed"), "reason: {reason}");
        assert!(reason.contains("provider down"));
        assert!(report.summary.contains("unjudged"));
        assert!(to_reward_signal(&report).is_none());
    }

    #[tokio::test]
    async fn judge_needs_a_criterion() {
        let model = FakeModel::new(Vec::new());
        let judge = ModelRewardJudge::new(model, ModelRequestConfig::default());
        let error = judge
            .judge(&EvidenceTask::default(), &EvidenceGathered::default(), "")
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
    }

    #[tokio::test]
    async fn judge_parses_fenced_json_and_missing_indices() {
        let model = FakeModel::new(vec![Ok(String::from(
            "Verdict:\n```json\n{\n  \"claims\": [\n    {\"status\": \"met\"},\n    {\"status\": \"unmet\", \"reason\": \"wrong\"}\n  ],\n  \"summary\": \"ok\"\n}\n```",
        ))]);
        let judge = ModelRewardJudge::new(model, ModelRequestConfig::default());
        let report = judge
            .judge(&task(CriterionKind::ProactiveChallenge), &gathered(), "")
            .await
            .unwrap();
        // Rows without an index keep their position: met then unmet -> 50.
        assert_eq!(report.criteria[0].score, Some(50));
        assert_eq!(report.criteria[0].claims[1].1, CriterionStatus::Unmet);
        assert_eq!(report.summary, "ok");
    }

    #[tokio::test]
    async fn a_generated_summary_names_the_scored_criteria() {
        let model = FakeModel::new(vec![Ok(String::from(
            r#"{"claims": [{"index": 0, "status": "met"}, {"index": 1, "status": "met"}]}"#,
        ))]);
        let judge = ModelRewardJudge::new(model, ModelRequestConfig::default());
        let report = judge
            .judge(&task(CriterionKind::Floor), &gathered(), "")
            .await
            .unwrap();
        assert_eq!(report.total, Some(100));
        assert_eq!(report.summary, "1 of 1 criteria scored; total 100/reward/3");
    }

    #[test]
    fn report_serde_round_trip_and_phase_defaults() {
        let report = CheckRewardReport {
            total: Some(75),
            criteria: vec![CriterionReport {
                id: String::from("c1"),
                name: String::from("Content"),
                kind: CriterionKind::Floor,
                rubric: String::from("0. Names the score"),
                status: CriterionStatus::Partial,
                score: Some(75),
                claims: vec![(
                    String::from("Names the score"),
                    CriterionStatus::Met,
                    String::from("stated"),
                )],
            }],
            summary: String::from("fine"),
            used_checks: 2,
        };
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["criteria"][0]["claims"][0][1], "met");
        assert_eq!(value["criteria"][0]["kind"], "floor");
        let restored: CheckRewardReport = serde_json::from_value(value).unwrap();
        assert_eq!(restored, report);
        let empty: CheckRewardReport = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, CheckRewardReport::default());

        // An unknown-field payload is tolerated.
        let permissive: CheckRewardReport =
            serde_json::from_str(r#"{"total": 40, "bogus": true}"#).unwrap();
        assert_eq!(permissive.total, Some(40));
    }

    #[test]
    fn judge_debug_and_accessors() {
        let judge =
            ModelRewardJudge::new(FakeModel::new(Vec::new()), ModelRequestConfig::default());
        assert_eq!(judge.model().model_name(), "fake-judge");
        assert_eq!(judge.request().model_name, "");
        assert!(format!("{judge:?}").contains("fake-judge"));
        assert!(format!("{judge:?}").contains("request"));
    }

    #[test]
    fn json_payload_accepts_prose_and_fences() {
        assert_eq!(json_payload("{\"a\": 1}"), Some(json!({"a": 1})));
        assert_eq!(json_payload("no json"), None);
        assert_eq!(
            json_payload("```json\n{\"a\": 1}\n```"),
            Some(json!({"a": 1}))
        );
        assert_eq!(
            json_payload("Here: {\"a\": 2} and done"),
            Some(json!({"a": 2}))
        );
        assert_eq!(json_payload("{ broken"), None);
    }

    #[test]
    fn signal_helpers_are_reachable() {
        // The default pack's aggregator consumes a signal with no weight surprises.
        let signal =
            RewardSignal::new(REWARD_CHECK_DIMENSION, 0.5).with_source(REWARD_CHECK_SOURCE);
        let summary = default_pack().aggregate(&[signal]).unwrap();
        assert!((summary.total() - 0.5).abs() < 1e-9);
    }
}
