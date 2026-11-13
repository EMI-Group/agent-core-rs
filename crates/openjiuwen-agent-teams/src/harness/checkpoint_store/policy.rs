//! Checkpoint capture knobs of one team member.
//!
//! The policy decides two things: whether a member captures checkpoints at all (and at which
//! lifecycle triggers), and how a harness consumes the checkpoint handed to it at start.
//! Trigger mapping onto the harness protocol is explicit rather than inferred: `on_turn` is a
//! finished turn (`CheckpointReason::TurnCompleted`), `on_state` a state change
//! (`CheckpointReason::StateChanged`) and `on_stop` the last capture before a member ends.
use openjiuwen_harness_protocol::ResumePolicy;
use serde::{Deserialize, Serialize};

/// How a member harness should use the checkpoint supplied at start.
///
/// Wire values are `new`, `if_available` and `require`; the mapping onto the protocol's own
/// [`ResumePolicy`] is [`MemberCheckpointPolicy::resume_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointResume {
    /// Start a fresh provider session and ignore any stored checkpoint (the default).
    #[default]
    New,
    /// Restore the stored checkpoint when one exists, and start fresh when none does.
    IfAvailable,
    /// Restore the stored checkpoint; starting fails when there is none to restore.
    Require,
}

/// Knobs for a member's checkpoint wiring.
///
/// Wire keys are `snake_case`, every field defaults, and unknown fields are tolerated so a newer
/// Python configuration keeps decoding. The defaults capture at turn and stop boundaries and
/// resume opportunistically: a member restarts from its latest checkpoint when one exists, and
/// bootstraps cleanly on a first run.
///
/// The booleans are independent trigger switches for unrelated lifecycle events, not a state
/// machine, so grouping them into enums would flatten the configuration shape this type mirrors.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberCheckpointPolicy {
    /// Master switch; `false` disables every capture and every resume attempt.
    pub enabled: bool,
    /// Capture after a turn finishes.
    pub on_turn: bool,
    /// Capture when the harness state changes (off by default: chatty and rarely useful).
    pub on_state: bool,
    /// Capture when the member stops.
    pub on_stop: bool,
    /// How a harness consumes the checkpoint supplied at start.
    pub resume: CheckpointResume,
}

impl Default for MemberCheckpointPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            on_turn: true,
            on_state: false,
            on_stop: true,
            resume: CheckpointResume::IfAvailable,
        }
    }
}

impl MemberCheckpointPolicy {
    /// A policy that neither captures nor resumes anything.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            on_turn: false,
            on_state: false,
            on_stop: false,
            resume: CheckpointResume::New,
        }
    }

    /// Whether checkpointing is on at all.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Maps the resume choice onto the protocol's resume policy.
    ///
    /// `New` → [`ResumePolicy::New`], `IfAvailable` → [`ResumePolicy::ResumeIfAvailable`],
    /// `Require` → [`ResumePolicy::RequireResume`].
    #[must_use]
    pub fn resume_policy(&self) -> ResumePolicy {
        match self.resume {
            CheckpointResume::New => ResumePolicy::New,
            CheckpointResume::IfAvailable => ResumePolicy::ResumeIfAvailable,
            CheckpointResume::Require => ResumePolicy::RequireResume,
        }
    }

    /// Whether a finished turn should be captured.
    #[must_use]
    pub fn captures_on_turn(&self) -> bool {
        self.enabled && self.on_turn
    }

    /// Whether a state change should be captured.
    #[must_use]
    pub fn captures_on_state(&self) -> bool {
        self.enabled && self.on_state
    }

    /// Whether stopping the member should be captured.
    #[must_use]
    pub fn captures_on_stop(&self) -> bool {
        self.enabled && self.on_stop
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{CheckpointResume, MemberCheckpointPolicy};
    use openjiuwen_harness_protocol::ResumePolicy;
    use serde_json::json;

    #[test]
    fn defaults_capture_on_turn_and_stop() {
        let policy = MemberCheckpointPolicy::default();
        assert!(policy.enabled);
        assert!(policy.on_turn);
        assert!(!policy.on_state);
        assert!(policy.on_stop);
        assert_eq!(policy.resume, CheckpointResume::IfAvailable);
        assert!(policy.is_enabled());
        assert!(policy.captures_on_turn());
        assert!(!policy.captures_on_state());
        assert!(policy.captures_on_stop());
    }

    #[test]
    fn disabled_turns_every_capture_off() {
        let policy = MemberCheckpointPolicy::disabled();
        assert!(!policy.is_enabled());
        assert!(!policy.captures_on_turn());
        assert!(!policy.captures_on_state());
        assert!(!policy.captures_on_stop());
        assert_eq!(policy.resume, CheckpointResume::New);
        assert_eq!(policy.resume_policy(), ResumePolicy::New);
    }

    #[test]
    fn master_switch_overrides_enabled_triggers() {
        let policy = MemberCheckpointPolicy {
            enabled: false,
            on_turn: true,
            on_state: true,
            on_stop: true,
            resume: CheckpointResume::Require,
        };
        assert!(!policy.captures_on_turn());
        assert!(!policy.captures_on_state());
        assert!(!policy.captures_on_stop());
        assert_eq!(policy.resume_policy(), ResumePolicy::RequireResume);
    }

    #[test]
    fn serde_round_trip_preserves_every_field() {
        let policy = MemberCheckpointPolicy {
            enabled: true,
            on_turn: false,
            on_state: true,
            on_stop: true,
            resume: CheckpointResume::Require,
        };
        let encoded = serde_json::to_value(&policy).unwrap();
        assert_eq!(
            encoded,
            json!({
                "enabled": true,
                "on_turn": false,
                "on_state": true,
                "on_stop": true,
                "resume": "require",
            })
        );
        let decoded: MemberCheckpointPolicy = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, policy);
    }

    #[test]
    fn unknown_fields_are_tolerated_and_missing_ones_default() {
        let decoded: MemberCheckpointPolicy =
            serde_json::from_value(json!({"enabled": false, "future_knob": 1})).unwrap();
        assert!(!decoded.enabled);
        assert!(decoded.on_turn);
        assert!(!decoded.on_state);
        assert!(decoded.on_stop);
        assert_eq!(decoded.resume, CheckpointResume::IfAvailable);
        assert!(!decoded.captures_on_turn());
    }

    #[test]
    fn resume_wire_values_are_snake_case() {
        for (resume, wire) in [
            (CheckpointResume::New, "new"),
            (CheckpointResume::IfAvailable, "if_available"),
            (CheckpointResume::Require, "require"),
        ] {
            assert_eq!(serde_json::to_value(resume).unwrap(), json!(wire));
            let decoded: CheckpointResume = serde_json::from_value(json!(wire)).unwrap();
            assert_eq!(decoded, resume);
        }
        assert_eq!(CheckpointResume::default(), CheckpointResume::New);
    }

    #[test]
    fn resume_maps_onto_the_protocol_policy() {
        for (resume, expected) in [
            (CheckpointResume::New, ResumePolicy::New),
            (
                CheckpointResume::IfAvailable,
                ResumePolicy::ResumeIfAvailable,
            ),
            (CheckpointResume::Require, ResumePolicy::RequireResume),
        ] {
            let policy = MemberCheckpointPolicy {
                resume,
                ..MemberCheckpointPolicy::default()
            };
            assert_eq!(policy.resume_policy(), expected);
        }
    }
}
