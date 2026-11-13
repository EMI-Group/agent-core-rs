//! Tests for [`super`] — the organisation chart, its validation rules and its hierarchy queries.

use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};

use super::*;

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

/// A three-team chart with a single reporting tree rooted at the `alpha` team.
///
/// ```text
/// alpha (root)                alpha::owner-1 (root, the organisation owner)
///   alpha::lead-a
///     alpha::dev-1
///     beta::team_leader
///       beta::dev-2
///   beta
///     gamma
///       gamma::sum-1
///         gamma::dev-3
/// ```
fn three_level_spec() -> OrganizationSpec {
    let mut spec = OrganizationSpec::new("acme");
    spec.display_name = "Acme".to_owned();
    spec.desc = "three-level chart".to_owned();
    spec.owner = Some("alpha::owner-1".to_owned());
    spec.summary_team = Some("gamma".to_owned());
    spec.teams = vec![
        team("alpha", OrgRole::Leader, Some("lead-a"), None),
        team("beta", OrgRole::Leader, None, Some("alpha")),
        team("gamma", OrgRole::Summary, Some("sum-1"), Some("beta")),
    ];
    spec.members = vec![
        member("alpha", "lead-a", OrgRole::Leader, Some("alpha")),
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
    ];
    spec
}

/// The validated chart of [`three_level_spec`].
fn chart() -> OrgChart {
    OrgChart::new(three_level_spec()).unwrap()
}

/// Collects the team names of `teams()`.
fn team_names(chart: &OrgChart) -> Vec<String> {
    chart
        .teams()
        .iter()
        .map(|team| team.team_name.clone())
        .collect()
}

/// Collects the `(team, member)` pairs of `members()`.
fn member_pairs(chart: &OrgChart) -> Vec<(String, String)> {
    chart
        .members()
        .iter()
        .map(|member| (member.team_name.clone(), member.member_id.clone()))
        .collect()
}

#[test]
fn organisation_ids_follow_the_shared_grammar() {
    assert!(OrganizationSpec::new("acme.core-1").validate().is_ok());
    let error = OrganizationSpec::new("acme/core").validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error
        .to_string()
        .contains("invalid organization id 'acme/core'"));
}

#[test]
fn roles_expose_their_wire_strings_and_capabilities() {
    assert_eq!(OrgRole::default(), OrgRole::Member);
    let cases = [
        (OrgRole::Member, "member", false, false),
        (OrgRole::Leader, "leader", true, false),
        (OrgRole::Owner, "owner", true, true),
        (OrgRole::Summary, "summary", true, true),
    ];
    for (role, wire, can_lead, can_summarize) in cases {
        assert_eq!(role.as_str(), wire);
        assert_eq!(role.can_lead(), can_lead, "{wire} lead");
        assert_eq!(role.can_summarize(), can_summarize, "{wire} summarize");
        assert_eq!(serde_json::to_value(role).unwrap(), json!(wire));
        assert_eq!(
            serde_json::from_value::<OrgRole>(json!(wire)).unwrap(),
            role
        );
    }
    assert!(OrgRole::Member < OrgRole::Leader);
}

#[test]
fn duplicate_and_malformed_team_names_are_refused() {
    let mut duplicate = OrganizationSpec::new("acme");
    duplicate.teams = vec![
        team("alpha", OrgRole::Leader, None, None),
        team("alpha", OrgRole::Leader, None, None),
    ];
    let error = duplicate.validate().unwrap_err();
    assert!(error.to_string().contains("duplicate team name 'alpha'"));

    let mut qualified = OrganizationSpec::new("acme");
    qualified.teams = vec![team("alpha::dev", OrgRole::Leader, None, None)];
    let error = qualified.validate().unwrap_err();
    assert!(error.to_string().contains("must not contain '::'"));

    let mut empty = OrganizationSpec::new("acme");
    empty.teams = vec![team("", OrgRole::Leader, None, None)];
    let error = empty.validate().unwrap_err();
    assert!(error.to_string().contains("team name must not be empty"));
}

#[test]
fn members_must_be_unique_and_belong_to_a_declared_team() {
    let mut duplicate = OrganizationSpec::new("acme");
    duplicate.teams = vec![team("alpha", OrgRole::Leader, None, None)];
    duplicate.members = vec![
        member("alpha", "dev-1", OrgRole::Member, None),
        member("alpha", "dev-1", OrgRole::Member, None),
    ];
    let error = duplicate.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("duplicate member 'alpha::dev-1'"));

    let mut orphan = OrganizationSpec::new("acme");
    orphan.members = vec![member("ghost", "dev-1", OrgRole::Member, None)];
    let error = orphan.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("member 'ghost::dev-1' belongs to unknown team 'ghost'"));

    let mut unnamed = OrganizationSpec::new("acme");
    unnamed.teams = vec![team("alpha", OrgRole::Leader, None, None)];
    unnamed.members = vec![member("alpha", "", OrgRole::Member, None)];
    let error = unnamed.validate().unwrap_err();
    assert!(error.to_string().contains("empty id"));
}

#[test]
fn reporting_lines_must_name_existing_nodes() {
    let mut unknown = OrganizationSpec::new("acme");
    unknown.teams = vec![team("alpha", OrgRole::Leader, None, None)];
    unknown.members = vec![member(
        "alpha",
        "dev-1",
        OrgRole::Member,
        Some("alpha::ghost"),
    )];
    let error = unknown.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("'alpha::dev-1' reports to unknown node 'alpha::ghost'"));

    let mut itself = OrganizationSpec::new("acme");
    itself.teams = vec![team("alpha", OrgRole::Leader, None, Some("alpha"))];
    let error = itself.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("'alpha' cannot report to itself"));
}

#[test]
fn team_reporting_cycles_are_reported_from_the_smallest_id() {
    let mut spec = OrganizationSpec::new("acme");
    spec.teams = vec![
        team("gamma", OrgRole::Leader, None, Some("alpha")),
        team("alpha", OrgRole::Leader, None, Some("beta")),
        team("beta", OrgRole::Leader, None, Some("gamma")),
    ];
    let error = spec.validate().unwrap_err();
    assert_eq!(
        error.to_string(),
        "validation: reporting cycle: alpha -> beta -> gamma -> alpha"
    );
}

#[test]
fn member_reporting_cycles_are_reported_with_qualified_ids() {
    let mut spec = OrganizationSpec::new("acme");
    spec.teams = vec![team("alpha", OrgRole::Leader, None, None)];
    spec.members = vec![
        member("alpha", "a", OrgRole::Member, Some("alpha::b")),
        member("alpha", "b", OrgRole::Member, Some("alpha::a")),
    ];
    let error = spec.validate().unwrap_err();
    assert_eq!(
        error.to_string(),
        "validation: reporting cycle: alpha::a -> alpha::b -> alpha::a"
    );
}

#[test]
fn team_leaders_must_be_members_of_their_team() {
    let mut missing = OrganizationSpec::new("acme");
    missing.teams = vec![team("alpha", OrgRole::Leader, Some("ghost"), None)];
    let error = missing.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("team 'alpha' leader 'ghost' is not a member of the team"));

    let mut defaulted = OrganizationSpec::new("acme");
    defaulted.teams = vec![team("alpha", OrgRole::Leader, None, None)];
    defaulted.members = vec![member(
        "alpha",
        DEFAULT_LEADER_MEMBER_NAME,
        OrgRole::Leader,
        None,
    )];
    assert!(defaulted.validate().is_ok());
    assert_eq!(
        defaulted.teams[0].leader_or_default(),
        DEFAULT_LEADER_MEMBER_NAME
    );

    let blank = team("alpha", OrgRole::Leader, Some(""), None);
    assert_eq!(blank.leader_or_default(), DEFAULT_LEADER_MEMBER_NAME);
}

#[test]
fn the_owner_must_be_an_unambiguous_member() {
    let mut unknown = three_level_spec();
    unknown.owner = Some("alpha::ghost".to_owned());
    let error = unknown.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("owner 'alpha::ghost' is not an organisation member"));

    let mut bare = three_level_spec();
    bare.owner = Some("owner-1".to_owned());
    assert!(bare.validate().is_ok());

    let mut duplicated = three_level_spec();
    duplicated.members.push(member(
        "beta",
        "owner-1",
        OrgRole::Owner,
        Some("beta::team_leader"),
    ));
    duplicated.owner = Some("owner-1".to_owned());
    let error = duplicated.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("owner 'owner-1' is not an organisation member"));
}

#[test]
fn the_summary_team_must_exist_and_carry_the_summary_role() {
    let mut unknown = three_level_spec();
    unknown.summary_team = Some("ghost".to_owned());
    let error = unknown.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("summary_team 'ghost' is not a team of the organisation"));

    let mut wrong_role = three_level_spec();
    wrong_role.summary_team = Some("beta".to_owned());
    let error = wrong_role.validate().unwrap_err();
    assert!(error
        .to_string()
        .contains("summary_team 'beta' must have role 'summary'"));

    assert_eq!(
        chart().spec().summary_team.as_deref(),
        Some("gamma"),
        "the fixture's summary team is valid"
    );
}

#[test]
fn teams_and_members_are_returned_in_sorted_order() {
    let chart = chart();
    assert_eq!(team_names(&chart), vec!["alpha", "beta", "gamma"]);
    let expected: Vec<(String, String)> = [
        ("alpha", "dev-1"),
        ("alpha", "lead-a"),
        ("alpha", "owner-1"),
        ("beta", "dev-2"),
        ("beta", "team_leader"),
        ("gamma", "dev-3"),
        ("gamma", "sum-1"),
    ]
    .iter()
    .map(|(team, member)| ((*team).to_owned(), (*member).to_owned()))
    .collect();
    assert_eq!(member_pairs(&chart), expected);

    let beta = chart.team_members("beta").unwrap();
    let beta_ids: Vec<&str> = beta
        .iter()
        .map(|member| member.member_id.as_str())
        .collect();
    assert_eq!(beta_ids, vec!["dev-2", "team_leader"]);
    assert_eq!(chart.team_members("gamma").unwrap().len(), 2);
}

#[test]
fn hierarchy_queries_walk_the_reporting_tree() {
    let chart = chart();
    assert_eq!(chart.depth_of("alpha").unwrap(), 0);
    assert_eq!(chart.depth_of("alpha::owner-1").unwrap(), 0);
    assert_eq!(chart.depth_of("alpha::lead-a").unwrap(), 1);
    assert_eq!(chart.depth_of("beta").unwrap(), 1);
    assert_eq!(chart.depth_of("beta::dev-2").unwrap(), 3);
    assert_eq!(chart.depth_of("gamma::dev-3").unwrap(), 4);

    assert_eq!(
        chart.ancestors("gamma::dev-3").unwrap(),
        vec![
            "gamma::sum-1".to_owned(),
            "gamma".to_owned(),
            "beta".to_owned(),
            "alpha".to_owned(),
        ]
    );
    assert_eq!(chart.ancestors("alpha").unwrap(), Vec::<String>::new());

    assert_eq!(
        chart.descendants("alpha").unwrap(),
        vec![
            "alpha::dev-1".to_owned(),
            "alpha::lead-a".to_owned(),
            "beta".to_owned(),
            "beta::dev-2".to_owned(),
            "beta::team_leader".to_owned(),
            "gamma".to_owned(),
            "gamma::dev-3".to_owned(),
            "gamma::sum-1".to_owned(),
        ]
    );
    assert_eq!(
        chart.descendants("gamma::dev-3").unwrap(),
        Vec::<String>::new()
    );

    assert_eq!(
        chart.direct_reports("alpha::lead-a").unwrap(),
        vec!["alpha::dev-1".to_owned(), "beta::team_leader".to_owned()]
    );
    assert_eq!(
        chart.direct_reports("alpha").unwrap(),
        vec!["alpha::lead-a".to_owned(), "beta".to_owned()]
    );

    assert!(chart.is_ancestor("alpha", "gamma::dev-3"));
    assert!(chart.is_ancestor("beta", "gamma::dev-3"));
    assert!(chart.is_ancestor("gamma", "gamma::sum-1"));
    assert!(!chart.is_ancestor("alpha", "alpha"));
    assert!(!chart.is_ancestor("gamma", "alpha"));
    assert!(!chart.is_ancestor("alpha", "ghost"));
    assert!(!chart.is_ancestor("ghost", "alpha"));
}

#[test]
fn hierarchy_queries_on_unknown_nodes_are_not_found() {
    let chart = chart();
    for error in [
        chart.direct_reports("ghost").unwrap_err(),
        chart.descendants("ghost").unwrap_err(),
        chart.ancestors("ghost").unwrap_err(),
        chart.depth_of("ghost").unwrap_err(),
        chart.team_members("ghost").unwrap_err(),
        chart.leader_of("ghost").unwrap_err(),
    ] {
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("ghost"));
    }
}

#[test]
fn leaders_are_resolved_by_team() {
    let chart = chart();
    assert_eq!(chart.leader_of("alpha").unwrap(), "lead-a");
    assert_eq!(chart.leader_of("beta").unwrap(), DEFAULT_LEADER_MEMBER_NAME);
    assert_eq!(chart.leader_of("gamma").unwrap(), "sum-1");
    assert_eq!(chart.team("beta").unwrap().team_name, "beta");
    assert!(chart.team("ghost").is_none());
}

#[test]
fn nodes_resolve_by_team_name_or_qualified_member_id() {
    let chart = chart();
    let team_node = chart.node("beta").unwrap();
    assert_eq!(team_node.id(), "beta");
    assert_eq!(team_node.role(), OrgRole::Leader);
    assert_eq!(team_node.reports_to(), Some("alpha"));

    let member_node = chart.node("alpha::dev-1").unwrap();
    assert_eq!(member_node.id(), "alpha::dev-1");
    assert_eq!(member_node.role(), OrgRole::Member);
    assert_eq!(member_node.reports_to(), Some("alpha::lead-a"));

    assert!(chart.node("beta::ghost").is_none());
    assert!(chart.node("ghost").is_none());

    assert_eq!(chart.role_of("gamma"), Some(OrgRole::Summary));
    assert_eq!(chart.role_of("alpha::owner-1"), Some(OrgRole::Owner));
    assert_eq!(chart.role_of("nope"), None);

    assert_eq!(chart.parent_of("beta"), Some("alpha"));
    assert_eq!(chart.parent_of("alpha"), None);
    assert_eq!(chart.parent_of("ghost"), None);

    assert_eq!(chart.team_of_node("beta::dev-2"), Some("beta"));
    assert_eq!(chart.team_of_node("gamma"), Some("gamma"));
    assert_eq!(chart.team_of_node("ghost"), None);
    assert_eq!(chart.team_of_node("beta::ghost"), None);

    assert_eq!(chart.organization_id(), "acme");
    assert_eq!(chart.member("beta", "dev-2").unwrap().member_id, "dev-2");
    assert!(chart.member("beta", "ghost").is_none());
    assert!(chart.member("ghost", "dev-2").is_none());
}

#[test]
fn the_spec_round_trips_through_serde_with_unknown_fields() {
    let raw = json!({
        "organization_id": "acme",
        "unknown_org_field": {"nested": 1},
        "teams": [{"team_name": "alpha", "unknown_team_field": 1}],
        "members": [{
            "team_name": "alpha",
            "member_id": "team_leader",
            "unknown_member_field": true,
        }],
    });
    let spec: OrganizationSpec = serde_json::from_value(raw).unwrap();
    assert_eq!(spec.display_name, "");
    assert_eq!(spec.teams[0].role, OrgRole::Leader);
    assert_eq!(spec.members[0].role, OrgRole::Member);
    assert_eq!(spec.members[0].desc, "");
    spec.validate().unwrap();

    let value = serde_json::to_value(&spec).unwrap();
    let back: OrganizationSpec = serde_json::from_value(value).unwrap();
    assert_eq!(back, spec);
}

#[test]
fn the_chart_serializes_its_spec_losslessly() {
    let chart = chart();
    let value = chart.to_value().unwrap();
    assert_eq!(value["organization_id"], json!("acme"));
    assert_eq!(value["summary_team"], json!("gamma"));
    assert_eq!(value["teams"].as_array().unwrap().len(), 3);
    let back: OrganizationSpec = serde_json::from_value(value).unwrap();
    assert_eq!(back, three_level_spec());
}

#[test]
fn the_chart_and_its_nodes_are_shareable() {
    fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
    assert_send_sync_clone::<OrgChart>();
    assert_send_sync_clone::<OrganizationSpec>();
    assert_send_sync_clone::<OrgTeam>();
    assert_send_sync_clone::<OrgMember>();
    assert_send_sync_clone::<OrgNodeRef<'static>>();

    let chart = chart();
    let cloned = chart.clone();
    let value: Value = cloned.to_value().unwrap();
    assert_eq!(value["organization_id"], json!("acme"));
}
