//! Tests for [`super`] — the delegation rules, the request record and the router's decision order.

use openjiuwen_common::error::ErrorKind;
use serde_json::json;

use super::*;
use crate::org::roles::{OrgMember, OrgTeam, OrganizationSpec};

/// Builds a team node with the given ids and reporting line.
fn team(name: &str, role: OrgRole, leader: Option<&str>, reports_to: Option<&str>) -> OrgTeam {
    OrgTeam {
        team_name: name.to_owned(),
        display_name: name.to_owned(),
        desc: format!("{name} team"),
        role,
        leader: leader.map(str::to_owned),
        reports_to: reports_to.map(str::to_owned),
        metadata: BTreeMap::new(),
    }
}

/// Builds a member node with the given ids and reporting line.
fn member(team: &str, id: &str, role: OrgRole, reports_to: Option<&str>) -> OrgMember {
    OrgMember {
        member_id: id.to_owned(),
        team_name: team.to_owned(),
        display_name: id.to_owned(),
        desc: format!("{id} member"),
        role,
        reports_to: reports_to.map(str::to_owned),
        metadata: BTreeMap::new(),
    }
}

/// A chart with two roots: `alpha -> beta -> gamma`, and the unrelated team `delta`.
fn fixture() -> Arc<OrgChart> {
    let mut spec = OrganizationSpec::new("acme");
    spec.owner = Some("alpha::owner-1".to_owned());
    spec.teams = vec![
        team("alpha", OrgRole::Leader, Some("lead-a"), None),
        team("beta", OrgRole::Leader, None, Some("alpha")),
        team("gamma", OrgRole::Summary, Some("sum-1"), Some("beta")),
        team("delta", OrgRole::Leader, None, None),
    ];
    spec.members = vec![
        member("alpha", "lead-a", OrgRole::Leader, None),
        member("alpha", "owner-1", OrgRole::Owner, None),
        member("alpha", "dev-1", OrgRole::Member, Some("alpha::lead-a")),
        member(
            "beta",
            "team_leader",
            OrgRole::Leader,
            Some("alpha::lead-a"),
        ),
        member("beta", "dev-2", OrgRole::Member, Some("beta::team_leader")),
        member("gamma", "sum-1", OrgRole::Leader, Some("gamma")),
        member("gamma", "dev-3", OrgRole::Member, Some("gamma::sum-1")),
        member("delta", "team_leader", OrgRole::Leader, None),
    ];
    Arc::new(OrgChart::new(spec).unwrap())
}

/// A router over the fixture chart applying `rules`.
fn router(rules: DelegationRules) -> DelegationRouter {
    DelegationRouter::new(fixture(), rules)
}

/// A router over the fixture chart applying the default rules.
fn default_router() -> DelegationRouter {
    router(DelegationRules::new())
}

/// Extracts the refusal of `decision`, failing the test when it was allowed.
fn refusal(decision: &DelegationDecision) -> DelegationRefusal {
    assert!(
        !decision.is_allowed(),
        "expected a refusal, got {decision:?}"
    );
    *decision
        .refusal()
        .expect("refused decisions carry a refusal")
}

#[test]
fn intra_team_delegation_is_allowed() {
    let request = DelegationRequest::new("beta", "dev-2", "beta").with_target_member("team_leader");
    let decision = default_router().decide(&request);
    assert_eq!(
        decision,
        DelegationDecision::Allowed {
            depth: 1,
            cross_team: false
        }
    );
    assert!(decision.is_allowed());
    assert_eq!(decision.refusal(), None);
    assert_eq!(decision.reason(), None);
}

#[test]
fn descendant_team_delegation_is_allowed() {
    let request = DelegationRequest::new("alpha", "lead-a", "beta");
    assert_eq!(
        default_router().decide(&request),
        DelegationDecision::Allowed {
            depth: 1,
            cross_team: true
        }
    );
}

#[test]
fn the_owner_may_delegate_outside_the_reporting_line() {
    let request = DelegationRequest::new("alpha", "owner-1", "delta");
    assert_eq!(
        default_router().decide(&request),
        DelegationDecision::Allowed {
            depth: 1,
            cross_team: true
        }
    );
}

#[test]
fn upward_delegation_is_refused() {
    let request =
        DelegationRequest::new("beta", "team_leader", "alpha").with_target_member("lead-a");
    let decision = default_router().decide(&request);
    assert_eq!(refusal(&decision), DelegationRefusal::ReportingLineRequired);
    assert_eq!(
        decision.reason().as_deref(),
        Some("delegation must follow the reporting line")
    );
}

#[test]
fn lateral_team_delegation_is_refused_by_default() {
    let request = DelegationRequest::new("beta", "dev-2", "delta");
    assert_eq!(
        refusal(&default_router().decide(&request)),
        DelegationRefusal::ReportingLineRequired
    );
}

#[test]
fn self_delegation_is_refused() {
    let implicit = DelegationRequest::new("beta", "team_leader", "beta");
    assert_eq!(
        refusal(&default_router().decide(&implicit)),
        DelegationRefusal::SelfDelegation
    );
    let explicit =
        DelegationRequest::new("beta", "team_leader", "beta").with_target_member("team_leader");
    assert_eq!(
        refusal(&default_router().decide(&explicit)),
        DelegationRefusal::SelfDelegation
    );
}

#[test]
fn unknown_participants_are_refused() {
    let router = default_router();
    let unknown_delegator = DelegationRequest::new("ghost", "dev-1", "beta");
    assert_eq!(
        refusal(&router.decide(&unknown_delegator)),
        DelegationRefusal::UnknownDelegator
    );
    let unknown_member = DelegationRequest::new("alpha", "ghost", "beta");
    assert_eq!(
        refusal(&router.decide(&unknown_member)),
        DelegationRefusal::UnknownDelegator
    );
    let unknown_team = DelegationRequest::new("alpha", "dev-1", "ghost");
    assert_eq!(
        refusal(&router.decide(&unknown_team)),
        DelegationRefusal::UnknownTargetTeam
    );
    let unknown_target =
        DelegationRequest::new("alpha", "dev-1", "beta").with_target_member("ghost");
    assert_eq!(
        refusal(&router.decide(&unknown_target)),
        DelegationRefusal::UnknownTargetMember
    );
}

#[test]
fn the_hop_count_is_capped() {
    let within = DelegationRequest::new("alpha", "lead-a", "beta").with_depth(2);
    assert_eq!(
        router(DelegationRules::new().with_max_depth(2)).decide(&within),
        DelegationDecision::Allowed {
            depth: 2,
            cross_team: true
        }
    );
    let beyond = DelegationRequest::new("alpha", "lead-a", "beta").with_depth(3);
    assert_eq!(
        refusal(&router(DelegationRules::new().with_max_depth(2)).decide(&beyond)),
        DelegationRefusal::DepthExceeded {
            requested: 3,
            max: 2
        }
    );
    let unmeasured = DelegationRequest::new("alpha", "lead-a", "beta");
    assert_eq!(
        router(DelegationRules::new().with_max_depth(1)).decide(&unmeasured),
        DelegationDecision::Allowed {
            depth: 1,
            cross_team: true
        }
    );
    let unlimited = DelegationRequest::new("alpha", "lead-a", "beta").with_depth(9);
    assert_eq!(
        router(DelegationRules::new().with_max_depth(0)).decide(&unlimited),
        DelegationDecision::Allowed {
            depth: 9,
            cross_team: true
        }
    );
}

#[test]
fn the_role_matrix_is_enforced() {
    let rules = DelegationRules::new().allowing(OrgRole::Member, OrgRole::Leader);
    assert!(rules.allows_pair(OrgRole::Member, OrgRole::Leader));
    assert!(!rules.allows_pair(OrgRole::Leader, OrgRole::Member));

    let allowed = DelegationRequest::new("beta", "dev-2", "beta").with_target_member("team_leader");
    assert!(router(rules.clone()).decide(&allowed).is_allowed());

    let denied = DelegationRequest::new("beta", "team_leader", "beta").with_target_member("dev-2");
    let decision = router(rules).decide(&denied);
    assert_eq!(
        refusal(&decision),
        DelegationRefusal::RoleNotAllowed {
            from: OrgRole::Leader,
            to: OrgRole::Member
        }
    );
}

#[test]
fn the_task_budget_is_enforced_at_the_boundary() {
    let rules = DelegationRules::new().with_default_budget(2);
    let room = DelegationRequest::new("alpha", "lead-a", "beta").with_open_tasks(1);
    assert!(router(rules.clone()).decide(&room).is_allowed());
    let full = DelegationRequest::new("alpha", "lead-a", "beta").with_open_tasks(2);
    assert_eq!(
        refusal(&router(rules).decide(&full)),
        DelegationRefusal::BudgetExhausted { open: 2, budget: 2 }
    );
}

#[test]
fn a_team_budget_overrides_the_default() {
    let rules = DelegationRules::new()
        .with_default_budget(5)
        .with_team_budget("beta", 1);
    assert_eq!(rules.budget_for("beta"), Some(1));
    assert_eq!(rules.budget_for("gamma"), Some(5));
    let beta = DelegationRequest::new("alpha", "lead-a", "beta").with_open_tasks(1);
    assert_eq!(
        refusal(&router(rules.clone()).decide(&beta)),
        DelegationRefusal::BudgetExhausted { open: 1, budget: 1 }
    );
    let gamma = DelegationRequest::new("alpha", "lead-a", "gamma").with_open_tasks(4);
    assert!(router(rules).decide(&gamma).is_allowed());
}

#[test]
fn the_reporting_line_requirement_can_be_relaxed() {
    let request = DelegationRequest::new("beta", "dev-2", "delta");
    assert!(!router(DelegationRules::new()).decide(&request).is_allowed());
    let relaxed = DelegationRules::new().with_require_reporting_line(false);
    assert_eq!(
        router(relaxed).decide(&request),
        DelegationDecision::Allowed {
            depth: 1,
            cross_team: true
        }
    );
}

#[test]
fn every_refusal_has_a_stable_code_and_message() {
    let cases = [
        (DelegationRefusal::UnknownDelegator, "unknown_delegator"),
        (DelegationRefusal::UnknownTargetTeam, "unknown_target_team"),
        (
            DelegationRefusal::UnknownTargetMember,
            "unknown_target_member",
        ),
        (DelegationRefusal::SelfDelegation, "self_delegation"),
        (
            DelegationRefusal::ReportingLineRequired,
            "reporting_line_required",
        ),
        (
            DelegationRefusal::DepthExceeded {
                requested: 4,
                max: 3,
            },
            "depth_exceeded",
        ),
        (
            DelegationRefusal::RoleNotAllowed {
                from: OrgRole::Leader,
                to: OrgRole::Member,
            },
            "role_not_allowed",
        ),
        (
            DelegationRefusal::BudgetExhausted { open: 3, budget: 3 },
            "budget_exhausted",
        ),
    ];
    for (refusal, code) in cases {
        assert_eq!(refusal.code(), code);
        assert_eq!(refusal.to_string(), refusal.message());
        assert_ne!(refusal.message(), "");
        let value = serde_json::to_value(refusal).unwrap();
        assert_eq!(
            serde_json::from_value::<DelegationRefusal>(value).unwrap(),
            refusal
        );
    }
    assert_eq!(
        DelegationRefusal::DepthExceeded {
            requested: 4,
            max: 3
        }
        .message(),
        "delegation depth 4 exceeds the maximum of 3"
    );
    assert_eq!(
        DelegationRefusal::RoleNotAllowed {
            from: OrgRole::Leader,
            to: OrgRole::Member,
        }
        .message(),
        "role transition 'leader -> member' is not allowed"
    );
}

#[test]
fn check_reports_refusals_as_validation_errors() {
    let request =
        DelegationRequest::new("beta", "team_leader", "alpha").with_target_member("lead-a");
    let error = default_router().check(&request).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: delegation must follow the reporting line"
    );

    let allowed = DelegationRequest::new("beta", "dev-2", "beta").with_target_member("team_leader");
    let router = default_router();
    assert!(router.check(&allowed).is_ok());
    assert!(router.can_delegate(&allowed));
    assert!(!router.can_delegate(&request));
}

#[test]
fn rules_and_requests_round_trip_through_serde() {
    let defaults: DelegationRules = serde_json::from_value(json!({"unknown": 1})).unwrap();
    assert_eq!(defaults, DelegationRules::new());
    assert_eq!(defaults.max_delegation_depth, 3);
    assert!(defaults.require_reporting_line);
    assert_eq!(defaults.allowed_pairs, Vec::<RolePair>::new());
    assert_eq!(defaults.default_task_budget, None);

    let value = serde_json::to_value(DelegationRules::new()).unwrap();
    assert_eq!(value["max_delegation_depth"], json!(3));
    assert_eq!(value["require_reporting_line"], json!(true));
    assert_eq!(value["allowed_pairs"], json!([]));
    assert_eq!(value["team_task_budget"], json!({}));
    assert_eq!(value["default_task_budget"], json!(null));

    let parsed: DelegationRules = serde_json::from_value(json!({
        "max_delegation_depth": 5,
        "require_reporting_line": false,
        "allowed_pairs": [{"from": "leader", "to": "member"}],
        "team_task_budget": {"beta": 2},
        "default_task_budget": 7,
        "unknown": true,
    }))
    .unwrap();
    assert_eq!(parsed.max_delegation_depth, 5);
    assert!(!parsed.require_reporting_line);
    assert_eq!(
        parsed.allowed_pairs,
        vec![RolePair {
            from: OrgRole::Leader,
            to: OrgRole::Member
        }]
    );
    assert_eq!(parsed.budget_for("beta"), Some(2));
    assert_eq!(parsed.budget_for("gamma"), Some(7));
    let round: DelegationRules =
        serde_json::from_value(serde_json::to_value(&parsed).unwrap()).unwrap();
    assert_eq!(round, parsed);

    let request: DelegationRequest = serde_json::from_value(json!({
        "from_team": "beta",
        "from_member": "dev-2",
        "to_team": "beta",
    }))
    .unwrap();
    assert_eq!(request.depth, 0);
    assert_eq!(request.to_member, None);
    assert_eq!(request.open_tasks_in_target, None);
    let round: DelegationRequest =
        serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
    assert_eq!(round, request);
}

#[test]
fn decisions_serialize_structurally() {
    let allowed = DelegationDecision::Allowed {
        depth: 1,
        cross_team: false,
    };
    assert_eq!(
        serde_json::to_value(&allowed).unwrap(),
        json!({"allowed": {"depth": 1, "cross_team": false}})
    );
    assert_eq!(
        serde_json::from_value::<DelegationDecision>(
            json!({"allowed": {"depth": 1, "cross_team": false}})
        )
        .unwrap(),
        allowed
    );
    let refused = DelegationDecision::Refused {
        refusal: DelegationRefusal::SelfDelegation,
    };
    assert_eq!(
        serde_json::to_value(&refused).unwrap(),
        json!({"refused": {"refusal": "self_delegation"}})
    );
}

#[test]
fn rule_builders_are_deterministic() {
    let rules = DelegationRules::new()
        .allowing(OrgRole::Leader, OrgRole::Leader)
        .allowing(OrgRole::Leader, OrgRole::Leader)
        .with_max_depth(2)
        .with_default_budget(4)
        .with_team_budget("beta", 1);
    assert_eq!(rules.allowed_pairs.len(), 1);
    assert_eq!(rules.max_delegation_depth, 2);
    assert_eq!(rules.default_task_budget, Some(4));
    assert_eq!(rules.budget_for("beta"), Some(1));
    assert_eq!(DelegationRules::new().budget_for("unknown"), None);
    assert!(DelegationRules::new().allows_pair(OrgRole::Owner, OrgRole::Summary));
    assert!(!rules.allows_pair(OrgRole::Owner, OrgRole::Summary));
    assert_eq!(DelegationRules::default(), DelegationRules::new());
}

#[test]
fn the_router_shares_its_chart_with_every_caller() {
    fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
    assert_send_sync_clone::<DelegationRouter>();
    assert_send_sync_clone::<DelegationRules>();
    assert_send_sync_clone::<DelegationRequest>();

    let chart = fixture();
    let rules = DelegationRules::new().with_max_depth(2);
    let router = DelegationRouter::new(Arc::clone(&chart), rules.clone());
    assert!(Arc::ptr_eq(router.chart(), &chart));
    assert_eq!(router.chart().organization_id(), "acme");
    assert_eq!(router.rules(), &rules);
    let cloned = router.clone();
    assert_eq!(cloned.rules().max_delegation_depth, 2);
}
