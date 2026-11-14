//! The evidence agent: one model-driven loop that gathers check evidence.
//!
//! Ports the portable core of Python
//! `openjiuwen.agent_evolving.reward.evidence_agent`. Python runs a `DeepAgent` with
//! two tools (`run_checks`, `read_evidence`); this port keeps the shape but talks
//! to the model in plain messages: each round asks for a JSON batch of checks
//! (`{"checks": [...]}`), runs them through [`CheckRunner`], and feeds the results
//! back. The loop stops at the round limit or when the model returns an empty
//! check list, which is also what an unparsable reply becomes.
//!
//! The parts of Python's agent that need the harness tool loop — saved-result
//! paging (`read_evidence`), idle stopping, restart after a dropped stream, the
//! default screenshot walk and screenshot-carrying messages — are not ported
//! here; the checks themselves and the round limit are.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::check::{clip, CheckOutcome, CheckRunner, CheckTool, RewardCheck};
use super::model_check::json_payload;

/// The model turns one evidence run may cost (Python `ROUNDS`).
pub const DEFAULT_ROUNDS: usize = 4;

/// The characters of one check result fed back to the model (Python `SHOWN`).
pub const DEFAULT_MAX_RESULT_CHARS: usize = 7500;

/// The evidence agent's instructions.
///
/// `render_prompt` fills these placeholders before the message is sent:
///
/// - `{task}` — the task text;
/// - `{criteria}` — every criterion with its id, name, kind and claims as JSON;
/// - `{workspace_files}` — the workspace file list, one name per line;
/// - `{inputs_excerpt}` — the input materials excerpt, or `(none)`;
/// - `{final_reply}` — the agent's final reply, or `(none)`;
/// - `{rounds}` — how many model turns this run may cost.
pub const EVIDENCE_AGENT_PROMPT: &str = "\
You gather evidence about an agent's delivered work for a Judge who decides each claim of each \
criterion afterwards. The first message gives the Task, every criterion with its claims, the \
workspace, the agent's final reply and your limit of model turns. You have one tool: \
run_checks(checks=[...]), which runs a batch of checks and returns each result whole.

Put every check that does not depend on another one's result into the same call: your first call \
holds every check you can already plan, for all criteria at once. Before it, go claim by claim and \
decide what observation settles it (a value recomputed, a line found, a state reached), then write \
the check that reaches exactly that, naming in \"criteria\" what it serves. Make at most one later \
call, only to fix a failed check or follow up on what a result revealed. Never judge or score; \
gather facts, and print values rather than pass/fail verdicts.

Read code closely only when code is the deliverable. A page's source is not read: how a page looks \
and behaves is seen by using it, and using it needs the browser tool, which is not available in \
this run. When a check cannot show a claim, say which criteria still lack evidence and why instead \
of repeating the check: a check identical to one already run is not run again.

Checks measure; you stop. When every claim has evidence, or the rest cannot be settled with these \
tools, reply in one line without a check.

Task:
{task}

Criteria:
{criteria}

Workspace files:
{workspace_files}

Inputs:
{inputs_excerpt}

Final reply:
{final_reply}

You have {rounds} model turns. Return JSON: {\"checks\": [{\"tool\": \"read|python|browser\", \
\"criteria\": [\"c1\"], ...}]}; return {\"checks\": []} to stop.";

/// How a criterion's claims are judged: what the work must do, or what top work
/// does better.
///
/// Python's claim `level`, defaulting to [`CriterionKind::Floor`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionKind {
    /// A requirement the work must meet; any unmet floor claim caps the score.
    #[default]
    Floor,
    /// A claim judged against the best work of this kind.
    ProactiveChallenge,
}

impl CriterionKind {
    /// The Python spelling of the level.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::ProactiveChallenge => "proactive_challenge",
        }
    }
}

/// One claim of a criterion: an index and the text the judge reads.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvidenceClaim {
    /// The claim's position in its criterion, starting at `0`.
    pub index: usize,
    /// The claim itself.
    pub text: String,
}

impl EvidenceClaim {
    /// Creates a claim with its index and text.
    #[must_use]
    pub fn new(index: usize, text: impl Into<String>) -> Self {
        Self {
            index,
            text: text.into(),
        }
    }
}

/// One criterion to gather evidence for.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvidenceCriterion {
    /// The criterion id the checks name.
    pub id: String,
    /// The human-readable criterion name.
    pub name: String,
    /// The claim level.
    pub kind: CriterionKind,
    /// The claims, in judging order.
    pub claims: Vec<EvidenceClaim>,
}

impl EvidenceCriterion {
    /// Creates an empty criterion with `id` and `name`.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind: CriterionKind::Floor,
            claims: Vec::new(),
        }
    }

    /// Sets the claim level.
    #[must_use]
    pub fn with_kind(mut self, kind: CriterionKind) -> Self {
        self.kind = kind;
        self
    }

    /// Replaces the claims.
    #[must_use]
    pub fn with_claims(mut self, claims: Vec<EvidenceClaim>) -> Self {
        self.claims = claims;
        self
    }

    /// Appends one claim, numbering it after the claims already present.
    #[must_use]
    pub fn with_claim(mut self, text: impl Into<String>) -> Self {
        let index = self.claims.len();
        self.claims.push(EvidenceClaim::new(index, text));
        self
    }
}

/// Everything the evidence agent is told about the run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvidenceTask {
    /// The task the work answers.
    pub task: String,
    /// The workspace file list the agent can read.
    pub workspace_files: Vec<String>,
    /// An excerpt of the input materials, when the caller supplied any.
    pub inputs_excerpt: String,
    /// The agent's final reply.
    pub final_reply: String,
    /// The criteria to gather evidence for.
    pub criteria: Vec<EvidenceCriterion>,
}

impl EvidenceTask {
    /// Creates a task with no files, inputs, reply or criteria.
    #[must_use]
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            ..Self::default()
        }
    }

    /// Replaces the workspace file list.
    #[must_use]
    pub fn with_workspace_files<I, S>(mut self, files: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.workspace_files = files.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the input materials excerpt.
    #[must_use]
    pub fn with_inputs_excerpt(mut self, excerpt: impl Into<String>) -> Self {
        self.inputs_excerpt = excerpt.into();
        self
    }

    /// Sets the final reply.
    #[must_use]
    pub fn with_final_reply(mut self, reply: impl Into<String>) -> Self {
        self.final_reply = reply.into();
        self
    }

    /// Replaces the criteria.
    #[must_use]
    pub fn with_criteria(mut self, criteria: Vec<EvidenceCriterion>) -> Self {
        self.criteria = criteria;
        self
    }

    /// The criterion with `id`, when it is declared.
    #[must_use]
    pub fn criterion(&self, id: &str) -> Option<&EvidenceCriterion> {
        self.criteria.iter().find(|criterion| criterion.id == id)
    }
}

/// One check the evidence agent ran, as the judge and the later rounds see it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GatheredCheck {
    /// The check's id (`c1`, `c2`, ...), unique within the run.
    pub id: String,
    /// The tool that ran, when one could be resolved.
    pub tool: Option<CheckTool>,
    /// The criteria ids this check serves.
    pub criteria: Vec<String>,
    /// `true` when the check ran and succeeded.
    pub ok: bool,
    /// `true` when the check repeated one already run, so it was not run again.
    pub repeat: bool,
    /// The rendered result, clipped to the agent's limit.
    pub result: String,
    /// Screenshot labels (always empty in this port).
    pub screenshots: Vec<String>,
    /// Why the check was skipped, when it was.
    pub skipped: Option<String>,
    /// Why the check failed, when it did.
    pub error: Option<String>,
}

impl GatheredCheck {
    /// `true` when the check produced usable evidence: it ran, it was not a
    /// repeat, and nothing was skipped or failed.
    #[must_use]
    pub fn usable(&self) -> bool {
        self.ok && !self.repeat && self.skipped.is_none() && self.error.is_none()
    }
}

/// The evidence one evidence-agent run gathered.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvidenceGathered {
    /// Every check the run saw, repeats included, in execution order.
    pub checks: Vec<GatheredCheck>,
    /// How many model turns the run cost.
    pub rounds: usize,
    /// How many requested checks were repeats and were not run again.
    pub deduped: usize,
}

impl EvidenceGathered {
    /// The distinct checks serving `criterion`, in execution order.
    #[must_use]
    pub fn checks_for(&self, criterion: &str) -> Vec<&GatheredCheck> {
        self.checks
            .iter()
            .filter(|check| !check.repeat && check.criteria.iter().any(|id| id == criterion))
            .collect()
    }

    /// The checks that actually ran (repeats excluded).
    #[must_use]
    pub fn distinct(&self) -> Vec<&GatheredCheck> {
        self.checks.iter().filter(|check| !check.repeat).collect()
    }
}

/// Gathers check evidence for every criterion through one model-driven loop.
#[derive(Clone)]
pub struct EvidenceAgent {
    model: Arc<dyn Model>,
    request: ModelRequestConfig,
    max_rounds: usize,
    max_result_chars: usize,
}

impl std::fmt::Debug for EvidenceAgent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EvidenceAgent")
            .field("model", &self.model.model_name())
            .field("request", &self.request)
            .field("max_rounds", &self.max_rounds)
            .field("max_result_chars", &self.max_result_chars)
            .finish()
    }
}

impl EvidenceAgent {
    /// Creates an agent over `model`, defaulting to 4 rounds and 7500 result
    /// characters.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, request: ModelRequestConfig) -> Self {
        Self {
            model,
            request,
            max_rounds: DEFAULT_ROUNDS,
            max_result_chars: DEFAULT_MAX_RESULT_CHARS,
        }
    }

    /// Sets the model-turn limit, at least `1`.
    #[must_use]
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds.max(1);
        self
    }

    /// Sets the characters of one check result fed back to the model.
    #[must_use]
    pub fn with_max_result_chars(mut self, max_result_chars: usize) -> Self {
        self.max_result_chars = max_result_chars;
        self
    }

    /// The model-turn limit.
    #[must_use]
    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }

    /// The per-result character limit.
    #[must_use]
    pub fn max_result_chars(&self) -> usize {
        self.max_result_chars
    }

    /// The model the agent asks.
    #[must_use]
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Gathers evidence for `task`, running every check through `runner`.
    ///
    /// Each round sends the instructions plus the results so far, reads a
    /// `{"checks": [...]}` batch, runs the new checks (repeats are only named) and
    /// continues. The loop stops after [`max_rounds`](Self::max_rounds) turns or
    /// as soon as the model returns no check — which is also what an unparsable
    /// reply becomes, so a confused model ends the run instead of panicking.
    ///
    /// # Errors
    ///
    /// The model's own error. A check that cannot run (a missing file, a deferred
    /// browser or picture tool) is recorded on its [`GatheredCheck`] instead: the
    /// model reads it and can plan around it, exactly as Python's `run_one` shows
    /// a failed check to the agent.
    pub async fn gather(
        &self,
        task: &EvidenceTask,
        runner: &CheckRunner,
    ) -> Result<EvidenceGathered> {
        let mut messages = vec![
            BaseMessage::system(render_prompt(task, self.max_rounds)),
            BaseMessage::user(results_message(0, self.max_rounds, &[])),
        ];
        let mut gathered = EvidenceGathered::default();
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        let mut executed = 0_usize;
        for round in 1..=self.max_rounds {
            gathered.rounds = round;
            let response = self.model.invoke(&messages, &self.request).await?;
            let content = response
                .message()
                .map_or_else(String::new, |message| message.text().into_owned());
            let requests = parse_check_requests(&content);
            if requests.is_empty() {
                break;
            }
            executed = self
                .run_batch(task, runner, requests, &mut gathered, &mut seen, executed)
                .await;
            messages.push(BaseMessage::assistant(content));
            messages.push(BaseMessage::user(results_message(
                round,
                self.max_rounds,
                &gathered.checks,
            )));
        }
        Ok(gathered)
    }

    /// Runs one parsed batch, recording each check and counting repeats.
    async fn run_batch(
        &self,
        task: &EvidenceTask,
        runner: &CheckRunner,
        requests: Vec<Value>,
        gathered: &mut EvidenceGathered,
        seen: &mut BTreeMap<String, usize>,
        executed: usize,
    ) -> usize {
        let mut executed = executed;
        for raw in requests {
            let Ok(check) = serde_json::from_value::<RewardCheck>(raw) else {
                continue; // a malformed check object is dropped, never a panic
            };
            let criteria = serves_for(task, &check);
            let key = check.dedupe_key();
            if let Some(&index) = seen.get(&key) {
                let repeated = &gathered.checks[index];
                let id = repeated.id.clone();
                let ok = repeated.ok;
                gathered.checks.push(GatheredCheck {
                    id: id.clone(),
                    tool: repeated.tool,
                    criteria,
                    ok,
                    repeat: true,
                    result: format!("already run as {id}; use its result"),
                    skipped: None,
                    error: None,
                    ..GatheredCheck::default()
                });
                gathered.deduped += 1;
                continue;
            }
            executed += 1;
            let outcome = match runner.run(&check).await {
                Ok(outcome) => outcome,
                Err(error) => CheckOutcome::failed(error.to_string()),
            };
            let record = gathered_check(
                format!("c{executed}"),
                check.infer_tool(),
                criteria,
                outcome,
                self.max_result_chars,
            );
            seen.insert(key, gathered.checks.len());
            gathered.checks.push(record);
        }
        executed
    }
}

/// Builds the [`GatheredCheck`] a model reply becomes.
fn gathered_check(
    id: String,
    tool: Option<CheckTool>,
    criteria: Vec<String>,
    outcome: CheckOutcome,
    max_result_chars: usize,
) -> GatheredCheck {
    GatheredCheck {
        id,
        tool,
        criteria,
        ok: outcome.ok,
        repeat: false,
        result: clip(&outcome.result, max_result_chars),
        screenshots: outcome.screenshots,
        skipped: outcome.skipped,
        error: outcome.error,
    }
}

/// The criteria a check serves: those it names that exist, else every criterion.
fn serves_for(task: &EvidenceTask, check: &RewardCheck) -> Vec<String> {
    let known: BTreeSet<&str> = task
        .criteria
        .iter()
        .map(|criterion| criterion.id.as_str())
        .collect();
    let requested: Vec<String> = check
        .criteria
        .iter()
        .filter(|id| known.contains(id.as_str()))
        .cloned()
        .collect();
    if requested.is_empty() {
        task.criteria
            .iter()
            .map(|criterion| criterion.id.clone())
            .collect()
    } else {
        requested
    }
}

/// Reads a `{"checks": [...]}` batch out of a model reply; anything else is empty.
fn parse_check_requests(content: &str) -> Vec<Value> {
    match json_payload(content) {
        Some(Value::Object(map)) => match map.get("checks") {
            Some(Value::Array(items)) => items
                .iter()
                .filter(|item| item.is_object())
                .cloned()
                .collect(),
            _ => Vec::new(),
        },
        Some(Value::Array(items)) => items.into_iter().filter(Value::is_object).collect(),
        _ => Vec::new(),
    }
}

/// Fills [`EVIDENCE_AGENT_PROMPT`]'s placeholders.
fn render_prompt(task: &EvidenceTask, rounds: usize) -> String {
    let criteria = serde_json::to_string_pretty(&task.criteria).unwrap_or_default();
    let files = if task.workspace_files.is_empty() {
        String::from("(no files)")
    } else {
        task.workspace_files.join("\n")
    };
    let inputs = if task.inputs_excerpt.trim().is_empty() {
        String::from("(none)")
    } else {
        task.inputs_excerpt.clone()
    };
    let reply = if task.final_reply.trim().is_empty() {
        String::from("(none)")
    } else {
        task.final_reply.clone()
    };
    EVIDENCE_AGENT_PROMPT
        .replace("{task}", &task.task)
        .replace("{criteria}", &criteria)
        .replace("{workspace_files}", &files)
        .replace("{inputs_excerpt}", &inputs)
        .replace("{final_reply}", &reply)
        .replace("{rounds}", &rounds.to_string())
}

/// The user message of one round: the results gathered so far.
fn results_message(round: usize, rounds: usize, checks: &[GatheredCheck]) -> String {
    let results: Vec<Value> = checks
        .iter()
        .map(|check| {
            json!({
                "id": check.id,
                "tool": check.tool.map(CheckTool::as_str),
                "ok": check.ok,
                "repeat": check.repeat,
                "criteria": check.criteria,
                "skipped": check.skipped,
                "error": check.error,
                "result": check.result,
            })
        })
        .collect();
    json!({
        "round": round + 1,
        "rounds": rounds,
        "results": results,
        "next": "return checks as {\"checks\": [...]}, or {\"checks\": []} to stop",
    })
    .to_string()
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
    use serde_json::json;

    use super::*;
    use crate::evaluator_pipeline::FakeCommandRunner;

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
            "fake-evidence"
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
                None => Err(Error::validation(
                    "the fake evidence model ran out of replies",
                )),
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

    fn task() -> EvidenceTask {
        EvidenceTask::new("Summarise the notes")
            .with_workspace_files(["notes.txt"])
            .with_final_reply("See notes.txt")
            .with_criteria(vec![EvidenceCriterion::new("c1", "Content")
                .with_claim("Names the score")
                .with_claim("Names the date")])
    }

    fn runner(root: &std::path::Path) -> CheckRunner {
        CheckRunner::new(root, Arc::new(FakeCommandRunner::new().record_only()))
    }

    #[test]
    fn prompt_renders_every_placeholder() {
        let rendered = render_prompt(&task(), 4);
        assert!(rendered.contains("Summarise the notes"));
        assert!(rendered.contains("Names the score"));
        assert!(rendered.contains("notes.txt"));
        assert!(rendered.contains("See notes.txt"));
        assert!(rendered.contains("You have 4 model turns."));
        for placeholder in [
            "{task}",
            "{criteria}",
            "{workspace_files}",
            "{inputs_excerpt}",
            "{final_reply}",
            "{rounds}",
        ] {
            assert!(
                !rendered.contains(placeholder),
                "{placeholder} left in the prompt"
            );
        }

        let empty = render_prompt(&EvidenceTask::default(), 2);
        assert!(empty.contains("(no files)"));
        assert!(empty.contains("(none)"));
    }

    #[test]
    fn check_requests_parse_json_arrays_and_fences() {
        assert_eq!(parse_check_requests("no json here"), Vec::<Value>::new());
        let parsed = parse_check_requests(r#"{"checks": [{"tool": "read", "path": "a"}]}"#);
        assert_eq!(parsed.len(), 1);
        let parsed = parse_check_requests("[{\"tool\": \"read\"}, 3]");
        assert_eq!(parsed.len(), 1);
        let fenced = parse_check_requests(
            "here you go:\n```json\n{\"checks\": [{\"tool\": \"read\"}]}\n```",
        );
        assert_eq!(fenced.len(), 1);
        assert_eq!(
            parse_check_requests(r#"{"checks": "nope"}"#),
            Vec::<Value>::new()
        );
    }

    #[tokio::test]
    async fn gather_runs_the_checks_and_stops_on_an_empty_list() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("notes.txt"), "Score: 7\nDate: May\n").unwrap();
        let model = FakeModel::new(vec![
            Ok(String::from(
                r#"{"checks": [{"tool": "read", "path": "notes.txt", "pattern": "score", "criteria": ["c1"]}]}"#,
            )),
            Ok(String::from(r#"{"checks": []}"#)),
        ]);
        let agent = EvidenceAgent::new(model.clone(), ModelRequestConfig::default());
        let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();

        assert_eq!(gathered.rounds, 2);
        assert_eq!(model.calls(), 2);
        assert_eq!(gathered.checks.len(), 1);
        assert_eq!(gathered.deduped, 0);
        let check = &gathered.checks[0];
        assert_eq!(check.id, "c1");
        assert_eq!(check.tool, Some(CheckTool::Read));
        assert_eq!(check.criteria, ["c1"]);
        assert!(check.ok && check.usable());
        assert_eq!(check.result, "path: notes.txt\nmatches: 1\n1: Score: 7");
        assert_eq!(gathered.checks_for("c1").len(), 1);
        assert_eq!(gathered.checks_for("c2"), Vec::<&GatheredCheck>::new());
    }

    #[tokio::test]
    async fn gather_dedupes_repeats_and_serves_every_criterion() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("notes.txt"), "one\n").unwrap();
        let model = FakeModel::new(vec![
            Ok(String::from(
                r#"{"checks": [
                    {"tool": "read", "path": "notes.txt", "criteria": ["c1"], "note": "first"},
                    {"tool": "read", "path": "notes.txt", "criteria": ["c2"], "note": "second"}
                ]}"#,
            )),
            Ok(String::from(r#"{"checks": [{"path": "notes.txt"}]}"#)),
            Ok(String::from(r#"{"checks": []}"#)),
        ]);
        let mut two = task();
        two.criteria
            .push(EvidenceCriterion::new("c2", "Style").with_claim("Reads well"));
        let agent = EvidenceAgent::new(model, ModelRequestConfig::default());
        let gathered = agent.gather(&two, &runner(temp.path())).await.unwrap();

        assert_eq!(gathered.rounds, 3);
        assert_eq!(gathered.deduped, 2);
        assert_eq!(gathered.checks.len(), 3);
        assert_eq!(gathered.distinct().len(), 1);
        assert_eq!(gathered.checks[0].criteria, ["c1"]);
        let repeat = &gathered.checks[1];
        assert!(repeat.repeat);
        assert_eq!(repeat.id, "c1");
        // The repeat keeps the criteria the model named for it this time.
        assert_eq!(repeat.criteria, ["c2"]);
        assert!(repeat.result.contains("already run as c1"));
        assert!(!repeat.usable());
        // A check without explicit criteria serves every criterion.
        assert_eq!(gathered.checks[2].criteria, ["c1", "c2"]);
    }

    #[tokio::test]
    async fn gather_stops_on_an_unparsable_reply_and_on_the_round_limit() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("notes.txt"), "one\n").unwrap();
        let model = FakeModel::new(vec![Ok(String::from("I cannot help with that"))]);
        let agent = EvidenceAgent::new(model.clone(), ModelRequestConfig::default());
        let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
        assert_eq!(gathered.rounds, 1);
        assert_eq!(model.calls(), 1);
        assert_eq!(gathered.checks, Vec::new());

        let fresh = || {
            Ok(String::from(
                r#"{"checks": [{"tool": "read", "path": "notes.txt", "count": 1}]}"#,
            ))
        };
        let model = FakeModel::new(vec![fresh(), fresh(), fresh(), fresh(), fresh()]);
        let agent =
            EvidenceAgent::new(model.clone(), ModelRequestConfig::default()).with_max_rounds(2);
        let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
        assert_eq!(gathered.rounds, 2);
        assert_eq!(model.calls(), 2);
        // The same check in round two is a repeat, not a second execution.
        assert_eq!(gathered.deduped, 1);
    }

    #[tokio::test]
    async fn gather_records_check_errors_and_propagates_model_errors() {
        let temp = tempfile::tempdir().unwrap();
        let model = FakeModel::new(vec![
            Ok(String::from(
                r#"{"checks": [{"page": "index.html", "criteria": ["c1"]}]}"#,
            )),
            Ok(String::from(r#"{"checks": []}"#)),
        ]);
        let agent = EvidenceAgent::new(model, ModelRequestConfig::default());
        let gathered = agent.gather(&task(), &runner(temp.path())).await.unwrap();
        let error = gathered.checks[0].error.as_deref().unwrap();
        assert!(error.contains("playwright"), "unexpected error: {error}");
        assert!(!gathered.checks[0].usable());

        let model = FakeModel::new(vec![Err(Error::llm("provider down"))]);
        let agent = EvidenceAgent::new(model, ModelRequestConfig::default());
        let error = agent
            .gather(&task(), &runner(temp.path()))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Llm(_)), "unexpected: {error}");
    }

    #[test]
    fn criterion_and_task_builders_and_serde() {
        let criterion = EvidenceCriterion::new("c1", "Content")
            .with_kind(CriterionKind::ProactiveChallenge)
            .with_claim("a")
            .with_claim("b");
        assert_eq!(criterion.claims[1], EvidenceClaim::new(1, "b"));
        assert_eq!(criterion.kind.as_str(), "proactive_challenge");
        let task = EvidenceTask::new("t").with_criteria(vec![criterion.clone()]);
        assert_eq!(task.criterion("c1"), Some(&criterion));
        assert!(task.criterion("missing").is_none());

        let value = serde_json::to_value(&task).unwrap();
        assert_eq!(value["criteria"][0]["kind"], "proactive_challenge");
        let restored: EvidenceTask = serde_json::from_value(value).unwrap();
        assert_eq!(restored, task);

        let defaulted: EvidenceCriterion = serde_json::from_str(r#"{"id": "c9"}"#).unwrap();
        assert_eq!(defaulted.kind, CriterionKind::Floor);
        assert_eq!(defaulted.claims, Vec::new());
    }

    #[test]
    fn agent_defaults_and_debug_rendering() {
        let agent = EvidenceAgent::new(FakeModel::new(Vec::new()), ModelRequestConfig::default());
        assert_eq!(agent.max_rounds(), DEFAULT_ROUNDS);
        assert_eq!(agent.max_result_chars(), DEFAULT_MAX_RESULT_CHARS);
        assert_eq!(agent.model().model_name(), "fake-evidence");
        assert_eq!(
            agent.clone().with_max_rounds(0).max_rounds(),
            1,
            "the round limit is at least one"
        );
        let rendered = format!("{agent:?}");
        assert!(rendered.contains("fake-evidence"));
        assert!(rendered.contains("max_rounds"));
    }

    #[test]
    fn gathered_json_round_trips() {
        let gathered = EvidenceGathered {
            checks: vec![GatheredCheck {
                id: String::from("c1"),
                tool: Some(CheckTool::Python),
                criteria: vec![String::from("c1")],
                ok: true,
                repeat: false,
                result: String::from("exit_code: 0"),
                screenshots: Vec::new(),
                skipped: None,
                error: None,
            }],
            rounds: 1,
            deduped: 0,
        };
        let value = serde_json::to_value(&gathered).unwrap();
        assert_eq!(value["checks"][0]["tool"], "python");
        let restored: EvidenceGathered = serde_json::from_value(value).unwrap();
        assert_eq!(restored, gathered);
        assert_eq!(
            serde_json::from_str::<EvidenceGathered>("{}").unwrap(),
            EvidenceGathered::default()
        );
    }

    #[test]
    fn json_payload_is_shared_with_the_judge() {
        assert_eq!(json_payload("{\"a\": 1}"), Some(json!({"a": 1})));
        assert_eq!(json_payload("no"), None);
    }
}
