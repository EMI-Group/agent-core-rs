//! Pause/resume/abort flags and outcome locking for one persistent execution.
//!
//! A [`Control`] is the shared signal board of one execution: the task loop asks it whether it
//! may keep running ([`Control::is_paused`], [`Control::pause_reason`]), the host asks it to stop
//! ([`Control::request_abort`]) and native code reports a critical failure
//! ([`Control::mark_fatal`]). Once the terminal outcome is decided the board is frozen by
//! [`Control::lock_outcome`], so a late pause or abort cannot rewrite that outcome.
//!
//! Every method is synchronous and guards a [`parking_lot::Mutex`] for a single read or write, so
//! no guard is ever held across an `.await`.
//!
//! See `CONTEXT.md` in the parent directory for the API contract.
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::entry::PauseReason;

/// Why an execution was asked to stop before its normal end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbortReason {
    /// The initiator cancelled the execution.
    Cancel,
    /// The initiator stopped the execution.
    Stop,
}

impl AbortReason {
    /// Returns the persisted snake-case spelling of the reason.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancel => "cancel",
            Self::Stop => "stop",
        }
    }
}

/// A point-in-time copy of a [`Control`]'s flags.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ControlSnapshot {
    /// Whether the execution is currently paused.
    pub paused: bool,
    /// Why the execution paused, when it is paused.
    pub pause_reason: Option<PauseReason>,
    /// The accepted abort reason, when an abort was requested.
    pub abort_reason: Option<AbortReason>,
    /// Whether the terminal outcome has been decided and frozen.
    pub outcome_locked: bool,
    /// The first critical failure reported by native code.
    pub fatal: Option<String>,
    /// How often the execution was resumed.
    pub resumes: u64,
}

/// The flags of one execution, kept behind a single mutex.
#[derive(Debug, Default)]
struct ControlInner {
    paused: bool,
    pause_reason: Option<PauseReason>,
    abort_reason: Option<AbortReason>,
    outcome_locked: bool,
    fatal: Option<String>,
    resumes: u64,
}

impl ControlInner {
    /// Whether a new pause or abort request may still be accepted.
    const fn is_open(&self) -> bool {
        !self.outcome_locked && self.abort_reason.is_none()
    }
}

/// Pause, resume, abort and fatal-failure signals shared by the parts of one execution.
///
/// The board is created before the agent exists, so a control can arrive before anything is
/// listening for it; the first accepted abort reason wins and survives into
/// [`Control::lock_outcome`].
#[derive(Debug, Default)]
pub struct Control {
    inner: Mutex<ControlInner>,
}

impl Control {
    /// Creates a control board with no flags set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pauses the execution with `reason`.
    ///
    /// Returns `false` — without changing anything — when the outcome is locked, an abort was
    /// already requested or the execution is already paused.
    pub fn pause(&self, reason: PauseReason) -> bool {
        let mut inner = self.inner.lock();
        if !inner.is_open() || inner.paused {
            return false;
        }
        inner.paused = true;
        inner.pause_reason = Some(reason);
        true
    }

    /// Resumes a paused execution.
    ///
    /// Returns `false` — without changing anything — when the outcome is locked, an abort was
    /// requested or the execution is not paused.
    pub fn resume(&self) -> bool {
        let mut inner = self.inner.lock();
        if !inner.is_open() || !inner.paused {
            return false;
        }
        inner.paused = false;
        inner.pause_reason = None;
        inner.resumes = inner.resumes.saturating_add(1);
        true
    }

    /// Records the first abort reason before the outcome is locked.
    ///
    /// Returns `false` when the outcome is locked or an abort reason is already recorded; the
    /// first request wins.
    pub fn request_abort(&self, reason: AbortReason) -> bool {
        let mut inner = self.inner.lock();
        if !inner.is_open() {
            return false;
        }
        inner.abort_reason = Some(reason);
        true
    }

    /// Freezes the outcome and returns the abort reason accepted before the freeze, if any.
    pub fn lock_outcome(&self) -> Option<AbortReason> {
        let mut inner = self.inner.lock();
        inner.outcome_locked = true;
        inner.abort_reason
    }

    /// Records a critical failure, keeping the first one reported.
    pub fn mark_fatal(&self, message: impl Into<String>) {
        let mut inner = self.inner.lock();
        if inner.fatal.is_none() {
            inner.fatal = Some(message.into());
        }
    }

    /// Returns whether the execution is paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.inner.lock().paused
    }

    /// Returns why the execution paused, when it is paused.
    #[must_use]
    pub fn pause_reason(&self) -> Option<PauseReason> {
        self.inner.lock().pause_reason
    }

    /// Returns the accepted abort reason, when an abort was requested.
    #[must_use]
    pub fn abort_reason(&self) -> Option<AbortReason> {
        self.inner.lock().abort_reason
    }

    /// Returns whether the terminal outcome has been frozen.
    #[must_use]
    pub fn outcome_locked(&self) -> bool {
        self.inner.lock().outcome_locked
    }

    /// Returns the first critical failure reported, when one was.
    #[must_use]
    pub fn fatal(&self) -> Option<String> {
        self.inner.lock().fatal.clone()
    }

    /// Returns how often the execution was resumed.
    #[must_use]
    pub fn resumes(&self) -> u64 {
        self.inner.lock().resumes
    }

    /// Takes a consistent snapshot of every flag.
    #[must_use]
    pub fn snapshot(&self) -> ControlSnapshot {
        let inner = self.inner.lock();
        ControlSnapshot {
            paused: inner.paused,
            pause_reason: inner.pause_reason,
            abort_reason: inner.abort_reason,
            outcome_locked: inner.outcome_locked,
            fatal: inner.fatal.clone(),
            resumes: inner.resumes,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{AbortReason, Control, ControlSnapshot};
    use crate::persistent_agent::entry::PauseReason;

    #[test]
    fn abort_reasons_use_snake_case() {
        assert_eq!(
            serde_json::to_string(&AbortReason::Cancel).unwrap(),
            "\"cancel\""
        );
        assert_eq!(
            serde_json::to_string(&AbortReason::Stop).unwrap(),
            "\"stop\""
        );
        let decoded: AbortReason = serde_json::from_str("\"stop\"").unwrap();
        assert_eq!(decoded, AbortReason::Stop);
        assert_eq!(AbortReason::Cancel.as_str(), "cancel");
        assert_eq!(AbortReason::Stop.as_str(), "stop");
    }

    #[test]
    fn pause_and_resume_move_the_flags_together() {
        let control = Control::new();
        assert!(!control.is_paused());
        assert_eq!(control.pause_reason(), None);

        assert!(control.pause(PauseReason::AwaitingConfirmation));
        assert!(control.is_paused());
        assert_eq!(
            control.pause_reason(),
            Some(PauseReason::AwaitingConfirmation)
        );
        // Pausing twice is refused and keeps the first reason.
        assert!(!control.pause(PauseReason::Error));
        assert_eq!(
            control.pause_reason(),
            Some(PauseReason::AwaitingConfirmation)
        );
        assert_eq!(control.resumes(), 0);

        assert!(control.resume());
        assert!(!control.is_paused());
        assert_eq!(control.pause_reason(), None);
        assert_eq!(control.resumes(), 1);

        assert!(control.pause(PauseReason::Stopped));
        assert!(control.resume());
        assert_eq!(control.resumes(), 2);
    }

    #[test]
    fn resume_is_refused_when_not_paused() {
        let control = Control::new();
        assert!(!control.resume());
        assert_eq!(control.resumes(), 0);
        assert!(!control.is_paused());
    }

    #[test]
    fn abort_reason_is_first_wins() {
        let control = Control::new();
        assert_eq!(control.abort_reason(), None);
        assert!(control.request_abort(AbortReason::Cancel));
        assert!(!control.request_abort(AbortReason::Stop));
        assert_eq!(control.abort_reason(), Some(AbortReason::Cancel));

        // An aborted execution can neither pause nor resume.
        assert!(!control.pause(PauseReason::Stopped));
        assert!(!control.resume());
        assert!(!control.is_paused());
        assert_eq!(control.resumes(), 0);
    }

    #[test]
    fn locking_freezes_pause_and_abort() {
        let paused = Control::new();
        assert!(paused.pause(PauseReason::BudgetDenied));
        assert_eq!(paused.lock_outcome(), None);
        assert!(paused.outcome_locked());
        assert!(!paused.pause(PauseReason::Error));
        assert!(!paused.resume());
        assert!(!paused.request_abort(AbortReason::Cancel));
        assert_eq!(paused.abort_reason(), None);
        assert_eq!(
            paused.pause_reason(),
            Some(PauseReason::BudgetDenied),
            "locking does not clear the pause flags"
        );

        let aborted = Control::new();
        assert!(aborted.request_abort(AbortReason::Stop));
        assert_eq!(aborted.lock_outcome(), Some(AbortReason::Stop));
        assert!(aborted.outcome_locked());
        assert!(!aborted.request_abort(AbortReason::Cancel));
        assert_eq!(aborted.abort_reason(), Some(AbortReason::Stop));
        assert_eq!(aborted.lock_outcome(), Some(AbortReason::Stop));
    }

    #[test]
    fn fatal_keeps_the_first_message() {
        let control = Control::new();
        assert_eq!(control.fatal(), None);
        control.mark_fatal("first failure");
        control.mark_fatal("second failure");
        assert_eq!(control.fatal().as_deref(), Some("first failure"));
    }

    #[test]
    fn snapshot_reports_every_flag_and_round_trips() {
        let control = Control::new();
        assert_eq!(control.snapshot(), ControlSnapshot::default());

        assert!(control.pause(PauseReason::ContextUnfit));
        assert!(control.resume());
        assert!(control.pause(PauseReason::Error));
        assert!(control.request_abort(AbortReason::Cancel));
        control.mark_fatal("boom");
        let snapshot = control.snapshot();
        assert!(snapshot.paused);
        assert_eq!(snapshot.pause_reason, Some(PauseReason::Error));
        assert_eq!(snapshot.abort_reason, Some(AbortReason::Cancel));
        assert!(!snapshot.outcome_locked);
        assert_eq!(snapshot.fatal.as_deref(), Some("boom"));
        assert_eq!(snapshot.resumes, 1);

        let encoded = serde_json::to_string(&snapshot).unwrap();
        let decoded: ControlSnapshot = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, snapshot);

        // Missing fields fall back to the defaults.
        let sparse: ControlSnapshot = serde_json::from_str("{}").unwrap();
        assert_eq!(sparse, ControlSnapshot::default());
        let tolerant: ControlSnapshot =
            serde_json::from_str(r#"{"paused":true,"future_flag":7}"#).unwrap();
        assert!(tolerant.paused);
        assert_eq!(tolerant.resumes, 0);
    }
}
