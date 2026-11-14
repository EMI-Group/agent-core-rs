//! Write modes (`create`/`append`/`merge`/`replace`/`skip`), the pure
//! [`WritePlan`](crate::lite::write::WritePlan) decision, and the bounded conflict-retry
//! checker.
//!
//! Ports Python's `WriteMode` from `openjiuwen/core/memory/lite/conflict_types.py` and the
//! `_MAX_CONFLICT_RETRIES = 2` loop from `coding_memory_tool_ops.py`, extending both for the
//! local Markdown store. Python's `WriteResult` maps onto
//! [`WriteOutcome`](crate::lite::write::WriteOutcome): `success` means "not `Skipped`",
//! `path`/`type` live on the returned [`CodingFact`], and `conflict_detected`/`note` keep
//! their names.
//!
//! # Modes
//!
//! | [`WriteMode`](crate::lite::write::WriteMode) | Fact exists | No fact |
//! | --- | --- | --- |
//! | `create` | refuses with a typed memory error | writes it |
//! | `append` | joins both bodies with one blank line | writes it |
//! | `merge` (default) | incoming non-blank fields win, tags unioned, `name`/`type`/`scope` kept | writes it |
//! | `replace` | the incoming fact overwrites the stored one (`None` path clears it) | writes it |
//! | `skip` | changes nothing and reports the skip | writes it |
//!
//! Python has neither `merge` nor `replace`: the local store is name-keyed, so overwriting an
//! entry is naturally what a plain write does (`replace`), while `merge` is the friendlier
//! default because it only touches the fields the caller actually filled in.
//!
//! # Retry and backoff
//!
//! [`MemUpdateChecker::plan`](crate::lite::write::MemUpdateChecker::plan) is a pure decision
//! function — it never touches storage, so a caller computes the outcome from the snapshot it
//! read. The retry driver,
//! [`MemUpdateChecker::run`](crate::lite::write::MemUpdateChecker::run), wraps a
//! read-decide-write closure that returns `Ok(None)` when it observed the store change
//! underneath it (a lost race); the checker then sleeps `backoff * 2^attempt`, capped at
//! `max_backoff`, before retrying. Retries are bounded by `max_attempts` (default 2, i.e. one
//! retry), so `max_attempts == 1` means "no retry — a conflict is an error".
//!
//! # Deviation from Python: exhaustion is a typed error
//!
//! Python degrades to a last-writer-wins write once `_MAX_CONFLICT_RETRIES` is exhausted, so a
//! lost race can silently overwrite a concurrent edit. This port deliberately does not: an
//! exhausted budget returns [`Error::Memory`] naming the operation and the attempt count, so
//! the caller decides whether to re-read and rewrite. A conflict is never reported as a skip
//! or as a success, and `run` never writes on its own.
//!
//! # No global state
//!
//! The checker is an ordinary value holding only a
//! [`WritePolicy`](crate::lite::write::WritePolicy); it is constructed per store and
//! injected, so tests and multi-store processes stay isolated. Nothing is registered
//! process-globally and no background task, timer, or lock is owned here — the store's own
//! file lock (or the engine) serializes the actual mutation.

use std::fmt;
use std::future::Future;
use std::str::FromStr;
use std::time::Duration;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use tokio::time::sleep;

use crate::lite::fact::CodingFact;

/// Tracing target for the conflict-retry driver.
const TRACE_TARGET: &str = "openjiuwen_memory::lite::write";

/// What a write does when a same-named fact already exists.
///
/// Serializes as the `snake_case` names `create`, `append`, `merge`, `replace`, and `skip`;
/// the default is [`WriteMode::Merge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteMode {
    /// Fails when a fact of that name already exists (Python's conflict-detecting mode).
    Create,
    /// Appends the incoming body to the existing body, keeping both.
    Append,
    /// Incoming wins per field, tags are unioned, and an absent path keeps the stored one.
    #[default]
    Merge,
    /// The incoming fact replaces the stored one wholesale (a `None` path clears it).
    Replace,
    /// No write when the fact exists; the skip is reported, never silently swallowed.
    Skip,
}

impl WriteMode {
    /// Every write mode, in the canonical order `create`, `append`, `merge`, `replace`, `skip`.
    pub const ALL: [Self; 5] = [
        Self::Create,
        Self::Append,
        Self::Merge,
        Self::Replace,
        Self::Skip,
    ];

    /// The `snake_case` name of this mode, as serialized and parsed.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Append => "append",
            Self::Merge => "merge",
            Self::Replace => "replace",
            Self::Skip => "skip",
        }
    }
}

impl FromStr for WriteMode {
    /// A validation error listing the accepted mode names.
    type Err = Error;

    /// Parses a mode name, trimming and ASCII-lowercasing it first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending value when it is not one of
    /// `create`, `append`, `merge`, `replace`, or `skip`.
    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "create" => Ok(Self::Create),
            "append" => Ok(Self::Append),
            "merge" => Ok(Self::Merge),
            "replace" => Ok(Self::Replace),
            "skip" => Ok(Self::Skip),
            _ => Err(Error::validation(format!(
                "unknown write mode \"{raw}\"; expected one of \"create\", \"append\", \
                 \"merge\", \"replace\", or \"skip\""
            ))),
        }
    }
}

impl fmt::Display for WriteMode {
    /// Writes [`WriteMode::as_str`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Bounded retry settings for a conflicting write.
///
/// `max_attempts` counts every attempt, so `1` disables retrying; the delay before attempt
/// `n + 1` is `backoff * 2^n`, capped at `max_backoff`. The default is Python's
/// `_MAX_CONFLICT_RETRIES = 2` attempts (one retry) with a short 5 ms first delay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritePolicy {
    /// Total number of attempts, including the first one; must be at least 1.
    pub max_attempts: usize,
    /// Delay before the first retry; doubled on every subsequent retry.
    pub backoff: Duration,
    /// Upper bound for the doubling, so a large attempt count stays bounded.
    pub max_backoff: Duration,
}

impl Default for WritePolicy {
    /// Two attempts, a 5 ms first backoff, and an 80 ms cap.
    fn default() -> Self {
        Self {
            max_attempts: 2,
            backoff: Duration::from_millis(5),
            max_backoff: Duration::from_millis(80),
        }
    }
}

impl WritePolicy {
    /// Validates the policy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `max_attempts` when it is 0, and naming
    /// `backoff`/`max_backoff` when the first delay exceeds the cap.
    pub fn validate(&self) -> Result<()> {
        if self.max_attempts == 0 {
            return Err(Error::validation(
                "write policy `max_attempts` must be at least 1 (got 0)",
            ));
        }
        if self.backoff > self.max_backoff {
            return Err(Error::validation(format!(
                "write policy `backoff` ({:?}) must not exceed `max_backoff` ({:?})",
                self.backoff, self.max_backoff
            )));
        }
        Ok(())
    }

    /// Returns the policy with a different attempt budget.
    #[must_use]
    pub const fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Returns the policy with different backoff bounds.
    #[must_use]
    pub const fn with_backoff(mut self, backoff: Duration, max_backoff: Duration) -> Self {
        self.backoff = backoff;
        self.max_backoff = max_backoff;
        self
    }
}

/// The effect a write has, computed from the mode and the current snapshot.
///
/// Every variant carries the fact as it would be stored; [`WritePlan::Skipped`] carries the
/// stored fact instead (or `None` when there is none).
#[derive(Debug, Clone, PartialEq)]
pub enum WritePlan {
    /// The fact does not exist yet and is written as-is.
    Created(CodingFact),
    /// [`WriteMode::Replace`] applied to an existing fact.
    Replaced(CodingFact),
    /// [`WriteMode::Merge`] applied to an existing fact.
    Merged(CodingFact),
    /// [`WriteMode::Append`] applied to an existing fact.
    Appended(CodingFact),
    /// [`WriteMode::Skip`] left an existing fact untouched; the stored fact is returned.
    Skipped(Option<CodingFact>),
}

/// The result of a completed write.
///
/// The counterpart of Python's `WriteResult`; combined with [`MemUpdateChecker::run`] it also
/// reports how many attempts the write needed.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteOutcome {
    /// The fact as stored (the existing fact for a skip).
    pub fact: CodingFact,
    /// The mode that was actually applied ([`WriteMode::Skip`] when skipped).
    pub mode: WriteMode,
    /// Whether the write created a new fact.
    pub created: bool,
    /// The 1-based number of attempts the write used.
    pub attempts: usize,
    /// Whether an attempt was retried because the store changed, or `create` found the fact.
    pub conflict: bool,
    /// Human-readable reason for a skip or a conflict.
    pub note: Option<String>,
}

/// The retry driver plus the pure plan computation.
///
/// Holds only a [`WritePolicy`]: it is constructed per store, injected by the caller, and keeps
/// no global state. [`MemUpdateChecker::plan`] never touches storage and
/// [`MemUpdateChecker::run`] never writes — the closure passed to `run` owns the
/// read-decide-write sequence.
#[derive(Debug, Clone)]
pub struct MemUpdateChecker {
    /// The bounded retry settings this checker applies.
    policy: WritePolicy,
}

impl MemUpdateChecker {
    /// Builds a checker, validating `policy`.
    ///
    /// # Errors
    ///
    /// Propagates [`WritePolicy::validate`]: [`Error::Validation`] for an attempt budget of 0
    /// or a first backoff above the cap.
    pub fn new(policy: WritePolicy) -> Result<Self> {
        policy.validate()?;
        Ok(Self { policy })
    }

    /// The policy in effect.
    #[must_use]
    pub fn policy(&self) -> &WritePolicy {
        &self.policy
    }

    /// Exponential, capped: `backoff * 2^attempt`, saturating, capped at `max_backoff`.
    ///
    /// `attempt` is 0-based: `0` is the delay before the second attempt.
    #[must_use]
    pub fn backoff_for(&self, attempt: usize) -> Duration {
        let factor = u32::try_from(attempt)
            .ok()
            .and_then(|shift| 1_u32.checked_shl(shift))
            .unwrap_or(u32::MAX);
        self.policy
            .backoff
            .saturating_mul(factor)
            .min(self.policy.max_backoff)
    }

    /// Pure decision function for one write, given the current snapshot of the fact.
    ///
    /// `now` becomes the fact's `updated_at` on every variant that writes. Per
    /// [`WriteMode`]:
    ///
    /// - [`WriteMode::Create`] with a stored fact is [`Error::Memory`] naming the fact; a
    ///   `create` collision is not retried here, and the caller may still treat it as a
    ///   conflict.
    /// - [`WriteMode::Skip`] with a stored fact is [`WritePlan::Skipped`] carrying it; without
    ///   one the fact is written ([`WritePlan::Created`]), as for every other mode.
    /// - [`WriteMode::Replace`] keeps nothing from the stored fact, so a `None` incoming path
    ///   clears the stored path.
    /// - [`WriteMode::Append`] joins the trimmed bodies with one blank line, keeps the stored
    ///   `name`/`type`/`scope`/`tags`, and lets a non-blank incoming `description` or `path`
    ///   win. When the incoming body is empty, or is already the trailing part of the stored
    ///   body (equal, or preceded by a blank line), the append would change nothing and the
    ///   plan is [`WritePlan::Skipped`].
    /// - [`WriteMode::Merge`] takes a non-blank incoming `content`/`description`/`path` (a
    ///   blank description or absent path keeps the stored value), unions the tags in order
    ///   (stored tags first, then unseen incoming tags, case-sensitive), and keeps the stored
    ///   `name`, `fact_type`, and `scope`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending field when the incoming fact's name
    /// or scope is invalid, and [`Error::Memory`] for a `create` collision.
    pub fn plan(
        &self,
        mode: WriteMode,
        incoming: &CodingFact,
        existing: Option<&CodingFact>,
        now: DateTime<Utc>,
    ) -> Result<WritePlan> {
        incoming.validate()?;
        let Some(existing) = existing else {
            return Ok(WritePlan::Created(with_timestamp(incoming, now)));
        };
        match mode {
            WriteMode::Skip => Ok(WritePlan::Skipped(Some(existing.clone()))),
            WriteMode::Create => Err(Error::memory(format!(
                "coding fact \"{}\" already exists; write mode `create` refuses to overwrite it",
                incoming.name
            ))),
            WriteMode::Replace => Ok(WritePlan::Replaced(with_timestamp(incoming, now))),
            WriteMode::Merge => Ok(WritePlan::Merged(merged(existing, incoming, now))),
            WriteMode::Append => Ok(appended(existing, incoming, now)),
        }
    }

    /// Runs `op` until it reports completion, retrying on `Ok(None)` (a conflict) with
    /// exponential backoff; returns the value plus the 1-based attempt count.
    ///
    /// `op` receives the 0-based attempt index, performs one read-decide-write, and returns
    /// `Ok(Some(value))` when it committed, `Ok(None)` when the store changed underneath it,
    /// or an error that is propagated unchanged. When every attempt reports a conflict the
    /// call returns [`Error::Memory`] naming `what` and the attempt count: exhaustion is never
    /// turned into a silent skip, and `max_attempts == 1` means no retry at all.
    ///
    /// # Errors
    ///
    /// Propagates any error from `op` immediately, without retrying, and returns
    /// [`Error::Memory`] when the attempt budget is exhausted.
    pub async fn run<T, F, Fut>(&self, what: &str, mut op: F) -> Result<(T, usize)>
    where
        F: FnMut(usize) -> Fut,
        Fut: Future<Output = Result<Option<T>>>,
    {
        let mut attempt = 0_usize;
        loop {
            if let Some(value) = op(attempt).await? {
                return Ok((value, attempt + 1));
            }
            if attempt + 1 >= self.policy.max_attempts {
                return Err(Error::memory(format!(
                    "{what}: write conflicted on all {} attempt(s); the store kept changing",
                    self.policy.max_attempts
                )));
            }
            let delay = self.backoff_for(attempt);
            tracing::debug!(
                target: TRACE_TARGET,
                what,
                attempt = attempt + 1,
                delay_ms = %delay.as_millis(),
                "lite write conflicted; retrying after backoff"
            );
            sleep(delay).await;
            attempt += 1;
        }
    }
}

/// Returns `incoming` with `updated_at = now`.
fn with_timestamp(incoming: &CodingFact, now: DateTime<Utc>) -> CodingFact {
    let mut fact = incoming.clone();
    fact.updated_at = now;
    fact
}

/// `Some(value)` when `value` has non-whitespace content, otherwise `None`.
fn non_blank(value: &str) -> Option<&str> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

/// `Some(path)` when the path is set and non-blank.
fn non_blank_path(path: Option<&str>) -> Option<&str> {
    path.and_then(non_blank)
}

/// The [`WriteMode::Merge`] fact: incoming non-blank fields win, tags unioned.
fn merged(existing: &CodingFact, incoming: &CodingFact, now: DateTime<Utc>) -> CodingFact {
    let mut tags = existing.tags.clone();
    for tag in &incoming.tags {
        if !tags.contains(tag) {
            tags.push(tag.clone());
        }
    }
    CodingFact {
        name: existing.name.clone(),
        description: non_blank(&incoming.description)
            .map_or_else(|| existing.description.clone(), str::to_owned),
        fact_type: existing.fact_type,
        scope: existing.scope.clone(),
        content: non_blank(&incoming.content)
            .map_or_else(|| existing.content.clone(), str::to_owned),
        tags,
        path: non_blank_path(incoming.path.as_deref())
            .map(str::to_owned)
            .or_else(|| existing.path.clone()),
        updated_at: now,
    }
}

/// The [`WriteMode::Append`] plan: one blank line between the two bodies, or a skip when the
/// incoming body would add nothing.
fn appended(existing: &CodingFact, incoming: &CodingFact, now: DateTime<Utc>) -> WritePlan {
    let existing_body = existing.content.trim();
    let incoming_body = incoming.content.trim();
    if incoming_body.is_empty() || is_trailing_part(existing_body, incoming_body) {
        return WritePlan::Skipped(Some(existing.clone()));
    }
    let mut fact = existing.clone();
    fact.content = if existing_body.is_empty() {
        incoming_body.to_owned()
    } else {
        format!("{existing_body}\n\n{incoming_body}")
    };
    if let Some(description) = non_blank(&incoming.description) {
        description.clone_into(&mut fact.description);
    }
    if let Some(path) = non_blank_path(incoming.path.as_deref()) {
        fact.path = Some(path.to_owned());
    }
    fact.updated_at = now;
    WritePlan::Appended(fact)
}

/// Whether appending `incoming` to `existing` would change nothing: the bodies are equal, or
/// `incoming` already ends `existing` at a blank-line boundary.
fn is_trailing_part(existing: &str, incoming: &str) -> bool {
    existing == incoming
        || existing
            .strip_suffix(incoming)
            .is_some_and(|head| head.ends_with("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lite::CodingFactType;
    use crate::record::MemoryScope;
    use openjiuwen_common::error::ErrorKind;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn fact(name: &str, body: &str) -> CodingFact {
        CodingFact::new(&scope(), name, body)
    }

    fn checker(max_attempts: usize) -> MemUpdateChecker {
        MemUpdateChecker::new(WritePolicy {
            max_attempts,
            backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(4),
        })
        .unwrap()
    }

    #[test]
    fn write_mode_names_serde_and_default() {
        assert_eq!(WriteMode::default(), WriteMode::Merge);
        let names: Vec<&str> = WriteMode::ALL.iter().map(|mode| mode.as_str()).collect();
        assert_eq!(names, vec!["create", "append", "merge", "replace", "skip"]);
        for mode in WriteMode::ALL {
            let json = serde_json::to_string(&mode).unwrap();
            assert_eq!(json, format!("\"{}\"", mode.as_str()));
            assert_eq!(serde_json::from_str::<WriteMode>(&json).unwrap(), mode);
            assert_eq!(mode.to_string(), mode.as_str());
        }
        assert!(serde_json::from_str::<WriteMode>("\"overwrite\"").is_err());
    }

    #[test]
    fn write_mode_parses_trimmed_and_case_insensitively() {
        for mode in WriteMode::ALL {
            let upper = mode.as_str().to_ascii_uppercase();
            assert_eq!(format!("  {upper} ").parse::<WriteMode>().unwrap(), mode);
        }
        let err = "overwrite".parse::<WriteMode>().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Validation);
        assert!(err.to_string().contains("overwrite"), "{err}");
    }

    #[test]
    fn policy_defaults_validation_and_builders() {
        let policy = WritePolicy::default();
        assert_eq!(policy.max_attempts, 2);
        assert_eq!(policy.backoff, Duration::from_millis(5));
        assert_eq!(policy.max_backoff, Duration::from_millis(80));
        assert!(policy.validate().is_ok());

        let tweaked = policy
            .with_max_attempts(4)
            .with_backoff(Duration::from_millis(2), Duration::from_millis(9));
        assert_eq!(tweaked.max_attempts, 4);
        assert_eq!(tweaked.backoff, Duration::from_millis(2));
        assert!(tweaked.validate().is_ok());

        let zero = policy.with_max_attempts(0);
        let message = zero.validate().unwrap_err().to_string();
        assert!(message.contains("max_attempts"), "{message}");
        assert_eq!(
            MemUpdateChecker::new(zero).unwrap_err().kind(),
            ErrorKind::Validation
        );

        let inverted = policy.with_backoff(Duration::from_millis(50), Duration::from_millis(10));
        let message = inverted.validate().unwrap_err().to_string();
        assert!(message.contains("backoff"), "{message}");
        assert!(message.contains("max_backoff"), "{message}");
    }

    #[test]
    fn backoff_grows_exponentially_and_caps() {
        let checker = MemUpdateChecker::new(WritePolicy {
            max_attempts: 10,
            backoff: Duration::from_millis(5),
            max_backoff: Duration::from_millis(80),
        })
        .unwrap();
        let delays: Vec<u128> = (0..=5)
            .map(|attempt| checker.backoff_for(attempt).as_millis())
            .collect();
        assert_eq!(delays, vec![5, 10, 20, 40, 80, 80]);
        assert_eq!(checker.backoff_for(usize::MAX), Duration::from_millis(80));
    }

    #[test]
    fn plan_writes_when_no_fact_exists() {
        let checker = checker(2);
        let now = Utc::now();
        let incoming = fact("build", "how to build");
        for mode in WriteMode::ALL {
            let plan = checker.plan(mode, &incoming, None, now).unwrap();
            let WritePlan::Created(planned) = plan else {
                panic!("{mode} without a stored fact planned {plan:?}");
            };
            assert_eq!(planned.name, incoming.name);
            assert_eq!(planned.content, incoming.content);
            assert_eq!(planned.fact_type, incoming.fact_type);
            assert_eq!(planned.updated_at, now);
        }
    }

    #[test]
    fn plan_create_refuses_an_existing_fact() {
        let err = checker(2)
            .plan(
                WriteMode::Create,
                &fact("build", "new"),
                Some(&fact("build", "old")),
                Utc::now(),
            )
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Memory);
        assert!(err.to_string().contains("build"), "{err}");
    }

    #[test]
    fn plan_skip_keeps_the_stored_fact() {
        let stored = fact("build", "old");
        let plan = checker(2)
            .plan(
                WriteMode::Skip,
                &fact("build", "new"),
                Some(&stored),
                Utc::now(),
            )
            .unwrap();
        assert_eq!(plan, WritePlan::Skipped(Some(stored)));
    }

    #[test]
    fn plan_replace_overwrites_and_clears_a_missing_path() {
        let checker = checker(2);
        let now = Utc::now();
        let stored = fact("build", "old")
            .with_description("old desc")
            .with_tags(vec!["old".to_owned()])
            .with_path("docs/OLD.md");
        let incoming = fact("build", "new").with_description("new desc");

        let plan = checker
            .plan(WriteMode::Replace, &incoming, Some(&stored), now)
            .unwrap();
        let WritePlan::Replaced(planned) = plan else {
            panic!("replace planned {plan:?}");
        };
        assert_eq!(planned.content, "new");
        assert_eq!(planned.description, "new desc");
        assert_eq!(planned.tags, Vec::<String>::new());
        assert!(
            planned.path.is_none(),
            "a None path must clear the stored one"
        );
        assert_eq!(planned.updated_at, now);

        let with_path = incoming.with_path("docs/NEW.md");
        let plan = checker
            .plan(WriteMode::Replace, &with_path, Some(&stored), now)
            .unwrap();
        let WritePlan::Replaced(planned) = plan else {
            panic!("replace planned {plan:?}");
        };
        assert_eq!(planned.path.as_deref(), Some("docs/NEW.md"));
    }

    #[test]
    fn plan_merge_unions_fields_and_keeps_stored_identity() {
        let checker = checker(2);
        let now = Utc::now();
        let stored = fact("build", "old body")
            .with_description("old desc")
            .with_type(CodingFactType::Reference)
            .with_tags(vec!["a".to_owned(), "b".to_owned()])
            .with_path("docs/OLD.md");
        let incoming = fact("build", "new body")
            .with_description("new desc")
            .with_tags(vec!["b".to_owned(), "c".to_owned(), "A".to_owned()]);

        let plan = checker
            .plan(WriteMode::Merge, &incoming, Some(&stored), now)
            .unwrap();
        let WritePlan::Merged(planned) = plan else {
            panic!("merge planned {plan:?}");
        };
        assert_eq!(planned.name, "build");
        assert_eq!(planned.fact_type, CodingFactType::Reference);
        assert_eq!(planned.scope, scope());
        assert_eq!(planned.content, "new body");
        assert_eq!(planned.description, "new desc");
        assert_eq!(planned.tags, vec!["a", "b", "c", "A"]);
        assert_eq!(planned.path.as_deref(), Some("docs/OLD.md"));
        assert_eq!(planned.updated_at, now);

        let blank = fact("build", "   ").with_description("");
        let plan = checker
            .plan(WriteMode::Merge, &blank, Some(&stored), now)
            .unwrap();
        let WritePlan::Merged(planned) = plan else {
            panic!("merge planned {plan:?}");
        };
        assert_eq!(planned.content, "old body");
        assert_eq!(planned.description, "old desc");
        assert_eq!(planned.path.as_deref(), Some("docs/OLD.md"));
    }

    #[test]
    fn plan_append_joins_bodies_and_applies_incoming_extras() {
        let checker = checker(2);
        let now = Utc::now();
        let stored = fact("build", "  first line  ").with_tags(vec!["kept".to_owned()]);
        let incoming = fact("build", " second line ")
            .with_description("  more  ")
            .with_path("docs/NEW.md");

        let plan = checker
            .plan(WriteMode::Append, &incoming, Some(&stored), now)
            .unwrap();
        let WritePlan::Appended(planned) = plan else {
            panic!("append planned {plan:?}");
        };
        assert_eq!(planned.content, "first line\n\nsecond line");
        assert_eq!(planned.description, "  more  ");
        assert_eq!(planned.tags, vec!["kept"]);
        assert_eq!(planned.path.as_deref(), Some("docs/NEW.md"));
        assert_eq!(planned.updated_at, now);

        let bare = fact("build", "second line");
        let plan = checker
            .plan(WriteMode::Append, &bare, Some(&stored), now)
            .unwrap();
        let WritePlan::Appended(planned) = plan else {
            panic!("append planned {plan:?}");
        };
        assert_eq!(planned.description, "");
        assert!(planned.path.is_none());
    }

    #[test]
    fn plan_append_skips_a_no_op() {
        let checker = checker(2);
        let now = Utc::now();
        let stored = fact("build", "one\n\ntwo");
        for body in ["two", "one\n\ntwo", "  two  ", ""] {
            let plan = checker
                .plan(WriteMode::Append, &fact("build", body), Some(&stored), now)
                .unwrap();
            assert_eq!(
                plan,
                WritePlan::Skipped(Some(stored.clone())),
                "body {body:?} should be a no-op append"
            );
        }

        let plan = checker
            .plan(
                WriteMode::Append,
                &fact("build", "three"),
                Some(&stored),
                now,
            )
            .unwrap();
        let WritePlan::Appended(planned) = plan else {
            panic!("append planned {plan:?}");
        };
        assert_eq!(planned.content, "one\n\ntwo\n\nthree");

        let empty_stored = fact("build", "   ");
        let plan = checker
            .plan(
                WriteMode::Append,
                &fact("build", "fresh"),
                Some(&empty_stored),
                now,
            )
            .unwrap();
        let WritePlan::Appended(planned) = plan else {
            panic!("append planned {plan:?}");
        };
        assert_eq!(planned.content, "fresh");
    }

    #[test]
    fn plan_rejects_a_blank_incoming_name() {
        let err = checker(2)
            .plan(WriteMode::Merge, &fact("   ", "body"), None, Utc::now())
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Validation);
        assert!(err.to_string().contains("name"), "{err}");
    }

    #[tokio::test]
    async fn run_completes_on_the_first_attempt() {
        let mut calls = 0_usize;
        let (value, attempts) = checker(3)
            .run("sync fact", |attempt| {
                calls += 1;
                async move {
                    assert_eq!(attempt, 0);
                    Ok(Some(7_u8))
                }
            })
            .await
            .unwrap();
        assert_eq!((value, attempts, calls), (7, 1, 1));
    }

    #[tokio::test]
    async fn run_retries_a_conflict_then_succeeds() {
        let mut seen = Vec::new();
        let (value, attempts) = checker(3)
            .run("sync fact", |attempt| {
                seen.push(attempt);
                async move {
                    if attempt == 0 {
                        Ok(None)
                    } else {
                        Ok(Some(attempt))
                    }
                }
            })
            .await
            .unwrap();
        assert_eq!((value, attempts), (1, 2));
        assert_eq!(seen, vec![0, 1]);
    }

    #[tokio::test]
    async fn run_reports_exhaustion_as_a_memory_error() {
        let mut calls = 0_usize;
        let err = checker(3)
            .run::<u8, _, _>("sync fact", |_attempt| {
                calls += 1;
                async move { Ok(None) }
            })
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Memory);
        let message = err.to_string();
        assert!(message.contains("sync fact"), "{message}");
        assert!(message.contains('3'), "{message}");
        assert!(message.contains("conflicted"), "{message}");
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn run_with_one_attempt_does_not_retry() {
        let mut calls = 0_usize;
        let err = checker(1)
            .run::<u8, _, _>("sync fact", |_attempt| {
                calls += 1;
                async move { Ok(None) }
            })
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Memory);
        assert_eq!(calls, 1);

        let (value, attempts) = checker(1)
            .run("sync fact", |_attempt| async move { Ok(Some(1_u8)) })
            .await
            .unwrap();
        assert_eq!((value, attempts), (1, 1));
    }

    #[tokio::test]
    async fn run_propagates_an_error_without_retrying() {
        let mut calls = 0_usize;
        let err = checker(3)
            .run::<u8, _, _>("sync fact", |_attempt| {
                calls += 1;
                async move { Err(Error::validation("bad fact")) }
            })
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Validation);
        assert_eq!(calls, 1);
    }
}
