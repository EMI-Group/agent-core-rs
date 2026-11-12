//! Run-time decision: create a team, attach to it, resume it, or refuse the request.
//!
//! When a caller runs a team by name the runtime has to reconcile the caller's session and the
//! requested [`TeamLifecycle`] with whatever is already live under that name. The decision is a
//! pure function of that state — [`decide_run_action`] — so the rules are testable without a
//! runtime and the runtime itself stays free of the branching.
//!
//! Python parity source: `openjiuwen/agent_teams/runtime/` team run entry (the "existing team"
//! checks around `TeamRuntimeManager.activate`).
//!
//! Evaluation order is significant and follows the truth table in `src/runtime/CONTEXT.md`:
//! no live team, terminated, persistent-vs-temporary, cross-session, then same-session resume.

use serde::{Deserialize, Serialize};

use openjiuwen_harness_protocol::HarnessState;

use crate::schema::TeamLifecycle;

/// What the runtime should do with a run request.
///
/// Wire values are the `lower_snake_case` strings returned by [`RunActionKind::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunActionKind {
    /// No team is live under the requested name; build and start one.
    CreateTeam,
    /// A live, same-session temporary team exists; hand the caller its existing handle.
    AttachExisting,
    /// A persistent team exists under the name and was stopped or paused; revive it.
    ResumeExisting,
    /// The live team is paused and the caller may continue from where it stopped.
    ResumePaused,
    /// A temporary team is live in another session; it cannot serve this caller.
    RejectCrossSession,
    /// The team under this name has already terminated.
    RejectStopped,
    /// The live team is persistent but the caller asked for a temporary one.
    RejectLifecycle,
}

impl Default for RunActionKind {
    /// The first evaluation step: create a team when nothing is live.
    fn default() -> Self {
        Self::CreateTeam
    }
}

impl RunActionKind {
    /// Returns the wire string of this action kind.
    ///
    /// These names are visible to callers (CLI output, logs, serialized decisions).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CreateTeam => "create_team",
            Self::AttachExisting => "attach_existing",
            Self::ResumeExisting => "resume_existing",
            Self::ResumePaused => "resume_paused",
            Self::RejectCrossSession => "reject_cross_session",
            Self::RejectStopped => "reject_stopped",
            Self::RejectLifecycle => "reject_lifecycle",
        }
    }

    /// Returns `true` when the runtime must refuse the run request instead of serving it.
    #[must_use]
    pub const fn is_reject(self) -> bool {
        matches!(
            self,
            Self::RejectCrossSession | Self::RejectStopped | Self::RejectLifecycle
        )
    }
}

/// The decision [`decide_run_action`] produced, together with the parties it is about.
///
/// `team_name` and `session_id` echo the live team's name and the *requested* session, so a
/// rejection reason and the caller's session stay in one place. When no team is live the team
/// name is empty — there is nothing to echo yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunAction {
    /// What the runtime should do.
    pub kind: RunActionKind,
    /// Name of the live team, or empty when no team is live.
    pub team_name: String,
    /// Session the caller asked to run the team in.
    pub session_id: String,
    /// Human-readable explanation; set for every rejection.
    pub reason: Option<String>,
}

impl Default for RunAction {
    /// A `CreateTeam` decision with no team, session or reason attached.
    fn default() -> Self {
        Self {
            kind: RunActionKind::default(),
            team_name: String::new(),
            session_id: String::new(),
            reason: None,
        }
    }
}

impl RunAction {
    /// Builds an action without a reason.
    #[must_use]
    pub fn new(
        kind: RunActionKind,
        team_name: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            team_name: team_name.into(),
            session_id: session_id.into(),
            reason: None,
        }
    }

    /// Attaches the human-readable reason of a rejection.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// Snapshot of the team already registered under a name, as the decision needs it.
///
/// This mirrors Python's "existing team" lookup: its [`HarnessState`] decides liveness, its
/// [`TeamLifecycle`] decides cross-session behaviour, and `paused` decides whether the caller
/// continues a suspended run or merely attaches to an idle one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LiveTeamState {
    /// Name the team is registered under.
    pub team_name: String,
    /// Session the live team was started in.
    pub session_id: String,
    /// Persistence class of the live team.
    pub lifecycle: TeamLifecycle,
    /// Whether the live team is suspended and can continue from its checkpoint.
    pub paused: bool,
    /// Harness lifecycle state of the live team.
    pub state: HarnessState,
}

impl Default for LiveTeamState {
    /// An idle, temporary team with no name and no session.
    fn default() -> Self {
        Self {
            team_name: String::new(),
            session_id: String::new(),
            lifecycle: TeamLifecycle::default(),
            paused: false,
            state: HarnessState::Idle,
        }
    }
}

/// Decides what to do with a run request for a team name.
///
/// `live` is the team already registered under that name (if any), `requested_session` is the
/// caller's session, and `lifecycle` is the lifecycle the caller asked for. The steps are
/// evaluated in order, so a terminated team is rejected before session and lifecycle are
/// considered:
///
/// 1. no live team → [`RunActionKind::CreateTeam`];
/// 2. live team terminated → [`RunActionKind::RejectStopped`];
/// 3. live team persistent but a temporary run requested → [`RunActionKind::RejectLifecycle`];
/// 4. different session: temporary → [`RunActionKind::RejectCrossSession`], persistent →
///    [`RunActionKind::ResumeExisting`];
/// 5. same session: paused → [`RunActionKind::ResumePaused`], persistent →
///    [`RunActionKind::ResumeExisting`], otherwise → [`RunActionKind::AttachExisting`].
#[must_use]
pub fn decide_run_action(
    live: Option<&LiveTeamState>,
    requested_session: &str,
    lifecycle: TeamLifecycle,
) -> RunAction {
    let Some(live) = live else {
        return RunAction::new(RunActionKind::CreateTeam, "", requested_session);
    };
    let team = live.team_name.as_str();

    if live.state == HarnessState::Terminated {
        return RunAction::new(RunActionKind::RejectStopped, team, requested_session)
            .with_reason(format!("team '{team}' has stopped; start a new team run"));
    }

    if live.lifecycle == TeamLifecycle::Persistent && lifecycle == TeamLifecycle::Temporary {
        return RunAction::new(RunActionKind::RejectLifecycle, team, requested_session)
            .with_reason(format!(
                "team '{team}' is persistent; it cannot be re-activated as temporary"
            ));
    }

    if live.session_id != requested_session {
        return match live.lifecycle {
            TeamLifecycle::Temporary => {
                RunAction::new(RunActionKind::RejectCrossSession, team, requested_session)
                    .with_reason(format!(
                        "team '{team}' is already running in session '{}' and cannot serve session \
                         '{requested_session}'",
                        live.session_id
                    ))
            }
            TeamLifecycle::Persistent => {
                RunAction::new(RunActionKind::ResumeExisting, team, requested_session)
            }
        };
    }

    if live.paused {
        return RunAction::new(RunActionKind::ResumePaused, team, requested_session);
    }
    if live.lifecycle == TeamLifecycle::Persistent {
        return RunAction::new(RunActionKind::ResumeExisting, team, requested_session);
    }
    RunAction::new(RunActionKind::AttachExisting, team, requested_session)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const TEAM: &str = "demo_team";
    const SESSION: &str = "session-a";
    const OTHER_SESSION: &str = "session-b";

    fn live(lifecycle: TeamLifecycle, paused: bool, state: HarnessState) -> LiveTeamState {
        LiveTeamState {
            team_name: TEAM.to_owned(),
            session_id: SESSION.to_owned(),
            lifecycle,
            paused,
            state,
        }
    }

    fn temporary(paused: bool) -> LiveTeamState {
        live(TeamLifecycle::Temporary, paused, HarnessState::Idle)
    }

    fn persistent(paused: bool) -> LiveTeamState {
        live(TeamLifecycle::Persistent, paused, HarnessState::Idle)
    }

    #[test]
    fn no_live_team_creates_a_team() {
        let action = decide_run_action(None, OTHER_SESSION, TeamLifecycle::Temporary);
        assert_eq!(action.kind, RunActionKind::CreateTeam);
        assert_eq!(action.team_name, "");
        assert_eq!(action.session_id, OTHER_SESSION);
        assert_eq!(action.reason, None);
    }

    #[test]
    fn terminated_team_is_rejected() {
        let state = live(TeamLifecycle::Temporary, false, HarnessState::Terminated);
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Temporary);
        assert_eq!(action.kind, RunActionKind::RejectStopped);
        assert_eq!(action.team_name, TEAM);
        assert_eq!(action.session_id, SESSION);
        assert_eq!(
            action.reason.as_deref(),
            Some("team 'demo_team' has stopped; start a new team run")
        );
    }

    #[test]
    fn terminated_while_paused_is_still_rejected() {
        let state = live(TeamLifecycle::Persistent, true, HarnessState::Terminated);
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Persistent);
        assert_eq!(action.kind, RunActionKind::RejectStopped);
        assert_eq!(
            action.reason.as_deref(),
            Some("team 'demo_team' has stopped; start a new team run")
        );
    }

    #[test]
    fn persistent_team_cannot_be_run_as_temporary() {
        let state = persistent(false);
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Temporary);
        assert_eq!(action.kind, RunActionKind::RejectLifecycle);
        assert_eq!(action.team_name, TEAM);
        assert_eq!(action.session_id, SESSION);
        assert_eq!(
            action.reason.as_deref(),
            Some("team 'demo_team' is persistent; it cannot be re-activated as temporary")
        );
    }

    #[test]
    fn persistent_team_rejects_temporary_before_cross_session() {
        let state = LiveTeamState {
            session_id: OTHER_SESSION.to_owned(),
            ..persistent(false)
        };
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Temporary);
        assert_eq!(action.kind, RunActionKind::RejectLifecycle);
    }

    #[test]
    fn temporary_team_rejects_a_cross_session_caller() {
        let state = LiveTeamState {
            session_id: OTHER_SESSION.to_owned(),
            ..temporary(false)
        };
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Temporary);
        assert_eq!(action.kind, RunActionKind::RejectCrossSession);
        assert_eq!(action.team_name, TEAM);
        assert_eq!(action.session_id, SESSION);
        assert_eq!(
            action.reason.as_deref(),
            Some(
                "team 'demo_team' is already running in session 'session-b' and cannot serve \
                 session 'session-a'"
            )
        );
    }

    #[test]
    fn persistent_team_resumes_for_a_cross_session_caller() {
        let state = LiveTeamState {
            session_id: OTHER_SESSION.to_owned(),
            ..persistent(false)
        };
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Persistent);
        assert_eq!(action.kind, RunActionKind::ResumeExisting);
        assert_eq!(action.team_name, TEAM);
        assert_eq!(action.session_id, SESSION);
        assert_eq!(action.reason, None);
    }

    #[test]
    fn persistent_team_resumes_in_the_same_session() {
        let state = persistent(false);
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Persistent);
        assert_eq!(action.kind, RunActionKind::ResumeExisting);
        assert_eq!(action.session_id, SESSION);
    }

    #[test]
    fn persistent_paused_team_resumes_paused_in_the_same_session() {
        let state = persistent(true);
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Persistent);
        assert_eq!(action.kind, RunActionKind::ResumePaused);
    }

    #[test]
    fn temporary_paused_team_resumes_paused() {
        let state = temporary(true);
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Temporary);
        assert_eq!(action.kind, RunActionKind::ResumePaused);
        assert_eq!(action.team_name, TEAM);
        assert_eq!(action.reason, None);
    }

    #[test]
    fn temporary_team_attaches_in_the_same_session() {
        let state = temporary(false);
        let action = decide_run_action(Some(&state), SESSION, TeamLifecycle::Temporary);
        assert_eq!(action.kind, RunActionKind::AttachExisting);
        assert_eq!(action.team_name, TEAM);
        assert_eq!(action.session_id, SESSION);
        assert_eq!(action.reason, None);
    }

    #[test]
    fn wire_names_match_the_contract() {
        let expected = [
            (RunActionKind::CreateTeam, "create_team"),
            (RunActionKind::AttachExisting, "attach_existing"),
            (RunActionKind::ResumeExisting, "resume_existing"),
            (RunActionKind::ResumePaused, "resume_paused"),
            (RunActionKind::RejectCrossSession, "reject_cross_session"),
            (RunActionKind::RejectStopped, "reject_stopped"),
            (RunActionKind::RejectLifecycle, "reject_lifecycle"),
        ];
        for (kind, wire) in expected {
            assert_eq!(kind.as_str(), wire);
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{wire}\""));
            assert_eq!(serde_json::from_str::<RunActionKind>(&json).unwrap(), kind);
        }
    }

    #[test]
    fn only_reject_kinds_are_rejects() {
        let rejecting = [
            RunActionKind::RejectCrossSession,
            RunActionKind::RejectStopped,
            RunActionKind::RejectLifecycle,
        ];
        for kind in rejecting {
            assert!(kind.is_reject(), "{} should reject", kind.as_str());
        }
        let serving = [
            RunActionKind::CreateTeam,
            RunActionKind::AttachExisting,
            RunActionKind::ResumeExisting,
            RunActionKind::ResumePaused,
        ];
        for kind in serving {
            assert!(!kind.is_reject(), "{} should serve", kind.as_str());
        }
    }

    #[test]
    fn run_action_round_trips_through_json() {
        let action = decide_run_action(
            Some(&temporary(false)),
            OTHER_SESSION,
            TeamLifecycle::Temporary,
        );
        let json = serde_json::to_string(&action).unwrap();
        assert_eq!(
            json,
            "{\"kind\":\"reject_cross_session\",\"team_name\":\"demo_team\",\
             \"session_id\":\"session-b\",\"reason\":\"team 'demo_team' is already running in \
             session 'session-a' and cannot serve session 'session-b'\"}"
        );
        let back: RunAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back, action);
    }

    #[test]
    fn run_action_deserializes_with_missing_fields() {
        let action: RunAction = serde_json::from_str("{\"kind\":\"create_team\"}").unwrap();
        assert_eq!(
            action,
            RunAction {
                kind: RunActionKind::CreateTeam,
                team_name: String::new(),
                session_id: String::new(),
                reason: None,
            }
        );
    }

    #[test]
    fn live_team_state_round_trips_through_json() {
        let state = LiveTeamState {
            paused: true,
            ..persistent(false)
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: LiveTeamState = serde_json::from_str(&json).unwrap();
        assert_eq!(back, state);
        let partial: LiveTeamState = serde_json::from_str("{\"team_name\":\"demo_team\"}").unwrap();
        assert_eq!(partial.team_name, TEAM);
        assert_eq!(partial.lifecycle, TeamLifecycle::Temporary);
        assert_eq!(partial.state, HarnessState::Idle);
        assert!(!partial.paused);
    }
}
