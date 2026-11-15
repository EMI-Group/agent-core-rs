//! The bounded multi-turn clarification loop.
//!
//! One clarification verdict ([`super::Clarifier::clarify`]) answers "is this
//! request complete, and if not, what is missing?".
//! A dialogue runs that same verdict repeatedly, feeding each answer back into a
//! working copy of the retained history, until the request is ready, the caller
//! stops answering, a round repeats the previous question, or the round cap is
//! reached.
//!
//! # Terminations
//!
//! [`ClarificationTermination`] names the four ways a dialogue ends:
//! [`Ready`](ClarificationTermination::Ready) when a round judged the request
//! complete; [`NoAnswer`](ClarificationTermination::NoAnswer) when the responder
//! returned `None` or a blank answer; [`RepeatedQuestion`](ClarificationTermination::RepeatedQuestion)
//! when a round asked exactly the same trimmed question as the previous round, so
//! the dialogue made no progress; and [`RoundCap`](ClarificationTermination::RoundCap)
//! when [`ClarifyConfig::max_rounds`] rounds have run.
//! The checks run in that order, so a ready verdict is never reported as a repeat
//! and an exhausted budget is only reached when the last two questions differ.
//!
//! # Working history
//!
//! The caller's [`HistoryManager`] is cloned for the dialogue and never mutated.
//! Each answered round appends its answer as one `user` turn to the working copy,
//! so a later round sees the accumulated dialogue exactly as the single-pass
//! clarifier would.
//!
//! # Provenance
//!
//! Each round contributes one `clarifying` provenance record to the
//! dialogue's [`GenerationProvenance`], oldest first, and the aggregate is also
//! the final [`Clarification::provenance`].
//! A degraded dialogue is logged at warn level once, when the dialogue is
//! finalized.

use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::super::history::HistoryManager;
use super::super::outcome::{GenerationProvenance, GenerationSource};
use super::{Clarification, Clarifier};

/// The round cap [`ClarifyConfig::new`] uses for a `0` argument.
pub const DEFAULT_MAX_CLARIFY_ROUNDS: usize = 3;

/// How many rounds a clarification dialogue may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ClarifyConfig {
    /// The maximum number of rounds; a dialogue always runs at least one.
    ///
    /// [`ClarifyConfig::new`] replaces `0` with
    /// [`DEFAULT_MAX_CLARIFY_ROUNDS`], and the loop treats a `0` assigned to this
    /// field directly as `1`.
    pub max_rounds: usize,
}

impl ClarifyConfig {
    /// Creates a configuration capping the dialogue at `max_rounds` rounds.
    ///
    /// `0` is replaced by [`DEFAULT_MAX_CLARIFY_ROUNDS`].
    #[must_use]
    pub fn new(max_rounds: usize) -> Self {
        let max_rounds = if max_rounds == 0 {
            DEFAULT_MAX_CLARIFY_ROUNDS
        } else {
            max_rounds
        };
        Self { max_rounds }
    }

    /// The number of rounds the loop may actually run: never zero.
    #[must_use]
    pub fn resolved_max_rounds(&self) -> usize {
        self.max_rounds.max(1)
    }
}

impl Default for ClarifyConfig {
    /// [`DEFAULT_MAX_CLARIFY_ROUNDS`] rounds.
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CLARIFY_ROUNDS)
    }
}

/// One round of a clarification dialogue.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ClarificationRound {
    /// The round's zero-based position in the dialogue.
    pub index: usize,
    /// The question this round asked; empty when the round was already ready.
    pub question: String,
    /// The answer the responder gave, when it gave a non-blank one.
    pub answer: Option<String>,
    /// Whether this round judged the request complete enough to design from.
    pub ready: bool,
    /// The requirement signals this round recognised.
    pub summary: String,
    /// Where this round's judgement came from.
    pub source: GenerationSource,
}

/// Why a clarification dialogue stopped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClarificationTermination {
    /// A round judged the request complete; the default for an empty dialogue.
    #[default]
    Ready,
    /// The responder gave no answer (or a blank one) to the last question.
    NoAnswer,
    /// A round asked exactly the same question as the previous round.
    RepeatedQuestion,
    /// The round cap was reached without a ready verdict.
    RoundCap,
}

impl ClarificationTermination {
    /// The termination's wire name.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NoAnswer => "no_answer",
            Self::RepeatedQuestion => "repeated_question",
            Self::RoundCap => "round_cap",
        }
    }
}

impl std::fmt::Display for ClarificationTermination {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A finished clarification dialogue.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ClarificationDialogue {
    /// Every round that ran, oldest first.
    pub rounds: Vec<ClarificationRound>,
    /// The last round's verdict, with the aggregate provenance attached.
    pub clarification: Clarification,
    /// Why the dialogue stopped.
    pub termination: ClarificationTermination,
    /// One `clarifying` provenance record per round, oldest first.
    pub provenance: GenerationProvenance,
}

/// One event of [`Clarifier::stream_clarify`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClarifyEvent {
    /// One finished round; exactly one per dialogue round.
    Round(ClarificationRound),
    /// The finished dialogue; always the last event of a successful stream.
    Completed(ClarificationDialogue),
}

/// Whether the loop may run another round after the one just settled.
enum RoundStep {
    /// The dialogue continues.
    Continue(ClarificationRound),
    /// The dialogue stops with the given reason.
    Stop(ClarificationRound, ClarificationTermination),
}

/// The loop's accumulating state.
struct RoundRunner {
    /// The caller's request, unchanged across rounds.
    query: String,
    /// The working copy of the retained history; answers are appended here.
    working: HistoryManager,
    /// The resolved round cap, at least one.
    max_rounds: usize,
    /// Every settled round, oldest first.
    rounds: Vec<ClarificationRound>,
    /// One record per settled round, oldest first.
    provenance: GenerationProvenance,
    /// The previous round's trimmed question, for the repeat check.
    previous_question: Option<String>,
}

impl RoundRunner {
    /// Starts a dialogue over a copy of `history`.
    fn new(query: &str, history: &HistoryManager, config: ClarifyConfig) -> Self {
        Self {
            query: query.to_owned(),
            working: history.clone(),
            max_rounds: config.resolved_max_rounds(),
            rounds: Vec::new(),
            provenance: GenerationProvenance::new(),
            previous_question: None,
        }
    }

    /// Runs the next round, asks the responder for an answer, and settles it.
    async fn run_round<F>(&mut self, clarifier: &Clarifier, responder: &mut F) -> RoundStep
    where
        F: FnMut(&ClarificationRound) -> Option<String>,
    {
        let index = self.rounds.len();
        let judgement = clarifier.judge(&self.query, &self.working).await;
        self.provenance.extend_from(judgement.provenance());
        let verdict = judgement.clarification;

        let mut round = ClarificationRound {
            index,
            question: verdict.question.clone(),
            answer: None,
            ready: verdict.ready,
            summary: verdict.summary.clone(),
            source: judgement.source,
        };

        if round.ready {
            self.settle(round.clone());
            return RoundStep::Stop(round, ClarificationTermination::Ready);
        }

        let answer = responder(&round).and_then(|answer| {
            let trimmed = answer.trim().to_owned();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        });
        let Some(answer) = answer else {
            self.settle(round.clone());
            return RoundStep::Stop(round, ClarificationTermination::NoAnswer);
        };

        let asked = round.question.trim().to_owned();
        let repeated = self.previous_question.as_deref() == Some(asked.as_str());
        round.answer = Some(answer.clone());
        self.working.add_user_message(answer);
        self.settle(round.clone());

        if repeated {
            return RoundStep::Stop(round, ClarificationTermination::RepeatedQuestion);
        }
        if self.rounds.len() >= self.max_rounds {
            return RoundStep::Stop(round, ClarificationTermination::RoundCap);
        }
        RoundStep::Continue(round)
    }

    /// Appends one settled round and remembers its question.
    fn settle(&mut self, round: ClarificationRound) {
        self.previous_question = Some(round.question.trim().to_owned());
        self.rounds.push(round);
    }

    /// Builds the finished dialogue, logging any degraded round once.
    fn finish(self, termination: ClarificationTermination) -> ClarificationDialogue {
        self.provenance.warn_if_degraded();
        let provenance = self.provenance;
        let clarification = match self.rounds.last() {
            Some(last) => Clarification {
                ready: last.ready,
                question: last.question.clone(),
                summary: last.summary.clone(),
                provenance: provenance.clone(),
            },
            None => Clarification {
                provenance: provenance.clone(),
                ..Clarification::default()
            },
        };
        ClarificationDialogue {
            rounds: self.rounds,
            clarification,
            termination,
            provenance,
        }
    }
}

/// Rejects a blank query before any model call.
fn validate_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(Error::validation("query must not be blank"));
    }
    Ok(())
}

impl Clarifier {
    /// Runs the bounded clarification dialogue without a responder.
    ///
    /// Every round is judged, but no answer is ever supplied, so the dialogue
    /// stops with [`ClarificationTermination::Ready`] on the first ready round
    /// and with [`ClarificationTermination::NoAnswer`] otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `query` is empty or whitespace only.
    pub async fn clarify_rounds(
        &self,
        query: &str,
        history: &HistoryManager,
        config: &ClarifyConfig,
    ) -> Result<ClarificationDialogue> {
        self.clarify_rounds_with(query, history, config, |_| None)
            .await
    }

    /// Runs the bounded clarification dialogue, asking `responder` for each
    /// round's answer.
    ///
    /// Round `i` judges `query` against a working copy of `history`: the
    /// caller's history is cloned, and every answered round appends its answer
    /// there as one `user` turn, so later rounds see the accumulated dialogue.
    /// The caller's `history` is never mutated.
    ///
    /// `responder` is asked once per not-ready round, with that round; it
    /// returns `None` (or a blank answer) to stop the dialogue. The dialogue
    /// stops on the first ready round, on a missing answer, on a round that
    /// repeats the previous question, or after `config.max_rounds` rounds.
    /// The final [`Clarification`] is the last round's verdict with the last
    /// question, and its provenance aggregates every round's record.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `query` is empty or whitespace only.
    /// A model failure is never an error: the round falls back to the
    /// deterministic heuristic and records a
    /// [`GenerationSource::Fallback`] provenance entry.
    pub async fn clarify_rounds_with<F>(
        &self,
        query: &str,
        history: &HistoryManager,
        config: &ClarifyConfig,
        mut responder: F,
    ) -> Result<ClarificationDialogue>
    where
        F: FnMut(&ClarificationRound) -> Option<String>,
    {
        validate_query(query)?;
        let mut runner = RoundRunner::new(query, history, *config);
        loop {
            if let RoundStep::Stop(_, termination) = runner.run_round(self, &mut responder).await {
                return Ok(runner.finish(termination));
            }
        }
    }

    /// Streams the clarification dialogue one round at a time.
    ///
    /// The stream owns clones of the clarifier, the query, and the history, so
    /// it is `Send + 'static` and needs no borrow of the caller.
    /// It runs [`Clarifier::clarify_rounds`] semantics with
    /// [`DEFAULT_MAX_CLARIFY_ROUNDS`] and no responder, yielding exactly one
    /// [`ClarifyEvent::Round`] per round followed by one
    /// [`ClarifyEvent::Completed`].
    /// Because there is no responder, the dialogue stops on the first ready
    /// round or with [`ClarificationTermination::NoAnswer`].
    /// A blank query yields one [`Error::Validation`] and nothing else.
    pub fn stream_clarify(
        &self,
        query: &str,
        history: &HistoryManager,
    ) -> BoxStream<'static, Result<ClarifyEvent>> {
        let clarifier = self.clone();
        let query = query.to_owned();
        let history = history.clone();

        Box::pin(async_stream::try_stream! {
            validate_query(&query)?;
            let config = ClarifyConfig::default();
            let mut runner = RoundRunner::new(&query, &history, config);
            loop {
                match runner.run_round(&clarifier, &mut |_| None).await {
                    RoundStep::Continue(round) => yield ClarifyEvent::Round(round),
                    RoundStep::Stop(round, termination) => {
                        yield ClarifyEvent::Round(round);
                        yield ClarifyEvent::Completed(runner.finish(termination));
                        break;
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests;
