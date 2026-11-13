//! The organisation layer driven end to end through a real team build.
//!
//! A team joins an organisation through one metadata block,
//! `metadata["organization"] = {"spec": OrganizationSpec, "rules": DelegationRules}`. That block is
//! parsed at build time into the backend's [`OrgBinding`]-shaped facade (`chart`, delegation
//! router, cross-team task pool and organisation mailbox), so everything below is what a caller of
//! the public API — CLI, MCP server, host — can actually observe:
//!
//! 1. a valid chart installs the binding and a broken one (bad id, dangling reporting line) fails
//!    the build with `Validation`;
//! 2. an allowed hop (reporting line satisfied, within depth and budget) is applied to the org
//!    task; self-delegation, an unknown target member and a cross-team sibling hop are refused with
//!    `Validation` and leave the pool untouched;
//! 3. the router itself enforces the depth cap, the per-team task budget and the role matrix;
//! 4. an org task runs `pending -> in_progress -> completed` in the reserved `org::<id>` namespace,
//!    carrying the sidecar, and a task aimed at a non-member is refused without a row;
//! 5. the organisation mailbox keeps a row unread until the leader turn that delivered it
//!    succeeded, and the persisted row carries the `org` protocol tag;
//! 6. a team without the block keeps the flat board and reports `Config` for the org-only calls.
//!
//! Everything is offline and deterministic: the shared `support` fakes script every member turn,
//! the store is in memory, and every wait is bounded by [`support::BOUND`].
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod support;
use openjiuwen_agent_teams::org::{
    org_namespace, DelegationDecision, DelegationRefusal, DelegationRequest, DelegationRules,
    OrgMember, OrgRole, OrgTaskEvent, OrgTaskSidecar, OrgTaskSpec, OrgTeam, OrganizationSpec,
    ORG_SYSTEM_SENDER,
};
use openjiuwen_agent_teams::runtime::{SpecBuild, TeamAgent};
use openjiuwen_agent_teams::schema::{TaskFilter, TaskGraphSpec, TaskStatus, TeamAgentSpec};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use serde_json::{json, Value};
use std::sync::Arc;
use support::{build_deps, FakeHarnessFactory, FakeWorkspaceProvider, BOUND};
/// Organisation id of the chart under test.
const ORG_ID: &str = "acme-labs";
/// Team the built `TeamAgent` runs.
const CORE: &str = "core";
/// Engineering team: a child of [`CORE`] that owns the working member.
const ENGINEERING: &str = "engineering";
/// Quality team: a sibling of [`ENGINEERING`], both reporting to [`CORE`].
const QA: &str = "qa";
/// Leader member id of a team that names no leader, which is exactly the runtime's default.
const LEADER: &str = "team_leader";
/// One team node; `role` is always [`OrgRole::Leader`] because a team leads itself.
fn team(name: &str, leader: &str, reports_to: Option<&str>) -> OrgTeam {
    OrgTeam {
        team_name: name.to_owned(),
        display_name: name.to_owned(),
        desc: format!("the {name} team"),
        role: OrgRole::Leader,
        leader: Some(leader.to_owned()),
        reports_to: reports_to.map(str::to_owned),
        ..OrgTeam::default()
    }
}
/// One member node reporting to `reports_to` (a qualified member id, or a team name).
fn member(team_name: &str, member_id: &str, role: OrgRole, reports_to: Option<&str>) -> OrgMember {
    OrgMember {
        member_id: member_id.to_owned(),
        team_name: team_name.to_owned(),
        display_name: member_id.to_owned(),
        desc: format!("{member_id} of {team_name}"),
        role,
        reports_to: reports_to.map(str::to_owned),
        ..OrgMember::default()
    }
}
/// Returns the chart under test.
///
/// `core` is the root, led by [`LEADER`] — the name the built team's own leader resolves to, which
/// is what makes `delegate_org_task` a known delegator. `engineering` and `qa` report to `core`
/// and are siblings of each other; `dev-1` reports to its own leader, `reviewer` to the `qa` one.
fn chart() -> OrganizationSpec {
    OrganizationSpec {
        organization_id: ORG_ID.to_owned(),
        display_name: "Acme Labs".to_owned(),
        desc: "the release organisation".to_owned(),
        owner: None,
        teams: vec![
            team(CORE, LEADER, None),
            team(ENGINEERING, "eng_leader", Some(CORE)),
            team(QA, "qa_leader", Some(CORE)),
        ],
        members: vec![
            member(CORE, LEADER, OrgRole::Leader, None),
            member(
                ENGINEERING,
                "eng_leader",
                OrgRole::Leader,
                Some("core::team_leader"),
            ),
            member(
                ENGINEERING,
                "dev-1",
                OrgRole::Member,
                Some("engineering::eng_leader"),
            ),
            member(QA, "qa_leader", OrgRole::Leader, Some("core::team_leader")),
            member(QA, "reviewer", OrgRole::Member, Some("qa::qa_leader")),
        ],
        summary_team: None,
        metadata: std::collections::BTreeMap::new(),
    }
}
/// Returns the delegation rules under test: two hops, reporting line required, a three-pair role
/// matrix and a two-task budget for [`QA`].
fn rules() -> DelegationRules {
    DelegationRules::new()
        .with_max_depth(2)
        .with_require_reporting_line(true)
        .allowing(OrgRole::Leader, OrgRole::Member)
        .allowing(OrgRole::Leader, OrgRole::Leader)
        .allowing(OrgRole::Member, OrgRole::Leader)
        .with_team_budget(QA, 2)
}
/// Wraps `org`/`rules` the way the runtime reads them: one `organization` metadata entry.
fn organization_metadata(org: &OrganizationSpec, rules: &DelegationRules) -> Value {
    json!({
        "spec": serde_json::to_value(org).expect("the chart serializes"),
        "rules": serde_json::to_value(rules).expect("the rules serialize"),
    })
}
/// Returns the sample blueprint renamed to [`CORE`] and carrying `org`/`rules` as its metadata.
fn organized(org: &OrganizationSpec, rules: &DelegationRules) -> TeamAgentSpec {
    let mut spec = flat_spec();
    spec.metadata
        .insert("organization".to_owned(), organization_metadata(org, rules));
    spec
}
/// Returns the sample blueprint without any organisation block.
fn flat_spec() -> TeamAgentSpec {
    TeamAgentSpec {
        team_name: CORE.to_owned(),
        ..support::sample_spec()
    }
}
/// One built team plus the fakes and the store its dependencies were built over.
struct OrgFixture {
    /// The team handle the build produced.
    agent: Arc<TeamAgent>,
    /// Harness factory: records one scripted harness per member.
    factory: Arc<FakeHarnessFactory>,
    /// The store injected into the build; the same instance the org layer writes through.
    store: Arc<InMemoryTeamStore>,
}
/// Builds `spec` over fresh fakes, keeping the error so an invalid chart can be asserted.
async fn try_build(spec: &TeamAgentSpec) -> Result<OrgFixture> {
    let factory = FakeHarnessFactory::new();
    let workspace = FakeWorkspaceProvider::new();
    let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
    let deps = build_deps(&factory, &workspace, Some(Arc::clone(&store)));
    let agent = spec.build_with(deps).await?;
    Ok(OrgFixture {
        agent,
        factory,
        store,
    })
}
/// Builds `spec`, panicking when the build fails.
async fn build(spec: &TeamAgentSpec) -> OrgFixture {
    match try_build(spec).await {
        Ok(fixture) => fixture,
        Err(error) => panic!("the team did not build: {error}"),
    }
}
/// Builds `spec` expecting a refusal and returns the error.
async fn build_error(spec: &TeamAgentSpec) -> Error {
    match try_build(spec).await {
        Ok(_) => panic!("the build accepted an invalid organisation"),
        Err(error) => error,
    }
}
/// Builds the organised chart under test.
async fn organized_fixture() -> OrgFixture {
    build(&organized(&chart(), &rules())).await
}
#[tokio::test]
async fn configured_chart_installs_the_organisation_binding() {
    let fixture = organized_fixture().await;
    let agent = &fixture.agent;
    assert_eq!(agent.team_name(), CORE);
    assert_eq!(agent.leader_member_name(), LEADER);
    let org = agent
        .backend()
        .org()
        .expect("backend().org() is empty for a configured organisation");
    assert!(agent.org().is_some(), "the handle lost the binding");
    assert_eq!(org.organization_id(), ORG_ID);
    assert_eq!(org.spec().organization_id, ORG_ID);
    assert_eq!(org.pool().namespace(), org_namespace(ORG_ID));
    assert_eq!(org.mailbox().namespace(), org_namespace(ORG_ID));
    let teams: Vec<(String, OrgRole)> = org.teams();
    assert_eq!(
        teams,
        vec![
            (CORE.to_owned(), OrgRole::Leader),
            (ENGINEERING.to_owned(), OrgRole::Leader),
            (QA.to_owned(), OrgRole::Leader),
        ]
    );
    let chart = org.chart();
    assert_eq!(chart.leader_of(ENGINEERING).unwrap(), "eng_leader");
    assert!(
        chart.is_ancestor(CORE, ENGINEERING),
        "core must lead engineering"
    );
    assert!(chart.is_ancestor(CORE, QA), "core must lead qa");
    assert!(
        !chart.is_ancestor(ENGINEERING, QA),
        "engineering and qa are siblings"
    );
    assert_eq!(chart.role_of("engineering::dev-1"), Some(OrgRole::Member));
    assert_eq!(chart.depth_of("engineering::dev-1").unwrap(), 2);
    assert_eq!(
        chart.direct_reports("engineering::eng_leader").unwrap(),
        vec!["engineering::dev-1".to_owned()]
    );
    let rules = org.rules();
    assert_eq!(rules.max_delegation_depth, 2);
    assert!(rules.require_reporting_line);
    assert_eq!(rules.allowed_pairs.len(), 3);
    assert_eq!(rules.budget_for(QA), Some(2));
    assert_eq!(rules.budget_for(ENGINEERING), None);
}
#[tokio::test]
async fn a_broken_chart_fails_the_build() {
    let bad_id = OrganizationSpec {
        organization_id: "acme labs".to_owned(),
        ..chart()
    };
    let error = build_error(&organized(&bad_id, &rules())).await;
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .message()
            .contains("invalid organization id 'acme labs'"),
        "unexpected message: {}",
        error.message()
    );
    let mut dangling = chart();
    dangling.teams[2].reports_to = Some("ghost".to_owned());
    let error = build_error(&organized(&dangling, &rules())).await;
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .message()
            .contains("'qa' reports to unknown node 'ghost'"),
        "unexpected message: {}",
        error.message()
    );
}
#[tokio::test]
async fn an_allowed_delegation_moves_the_org_task() {
    let fixture = organized_fixture().await;
    let agent = &fixture.agent;
    let org = agent.org().expect("no organisation was installed");
    let created = agent
        .create_org_task(OrgTaskSpec::new(CORE, "ship the release", "cut 1.0"))
        .await
        .expect("the org task was not created");
    let task_id = created.task.task_id.clone();
    assert_eq!(created.task.status, TaskStatus::Pending);
    assert_eq!(created.task.owner_team, CORE);
    assert_eq!(created.task.target_team, CORE);
    assert!(
        created.events.iter().any(
            |event| matches!(event, OrgTaskEvent::Created { task_id: id, .. } if id == &task_id)
        ),
        "the create emitted no `created` event: {:?}",
        created.events
    );
    assert_eq!(org.pool().open_task_count(ENGINEERING).await.unwrap(), 0);
    let result = agent
        .delegate_org_task(&task_id, ENGINEERING, Some("dev-1"))
        .await
        .expect("the router refused an allowed hop");
    assert!(
        result.is_ok(),
        "the delegation was refused: {:?}",
        result.reason()
    );
    let delegated = result.task.expect("the delegation carries the row");
    assert_eq!(delegated.target_team, ENGINEERING);
    assert_eq!(delegated.assignee.as_deref(), Some("engineering::dev-1"));
    assert_eq!(delegated.status, TaskStatus::Pending);
    assert_eq!(delegated.owner_team, CORE);
    assert_eq!(org.pool().open_task_count(ENGINEERING).await.unwrap(), 1);
    assert_eq!(org.pool().open_task_count(CORE).await.unwrap(), 0);
    let claimed = org
        .pool()
        .claim(&task_id, ENGINEERING, "dev-1")
        .await
        .expect("the claim failed");
    assert!(
        claimed.is_ok(),
        "the claim was refused: {:?}",
        claimed.reason()
    );
    assert_eq!(claimed.task.unwrap().status, TaskStatus::InProgress);
}
#[tokio::test]
async fn refused_delegations_leave_the_org_task_pool_untouched() {
    let fixture = organized_fixture().await;
    let agent = &fixture.agent;
    let org = agent.org().expect("no organisation was installed");
    let core_task = agent
        .create_org_task(OrgTaskSpec::new(CORE, "own work", "do it here"))
        .await
        .expect("the core task was not created")
        .task
        .task_id;
    let engineering_task = agent
        .create_org_task(OrgTaskSpec::new(ENGINEERING, "hand off", "do it there"))
        .await
        .expect("the engineering task was not created")
        .task
        .task_id;
    let core_before = org.pool().open_task_count(CORE).await.unwrap();
    let engineering_before = org.pool().open_task_count(ENGINEERING).await.unwrap();
    let qa_before = org.pool().open_task_count(QA).await.unwrap();
    // (1) Self-delegation: the target resolves to the delegating leader itself.
    let error = agent
        .delegate_org_task(&core_task, CORE, None)
        .await
        .expect_err("the router allowed a self delegation");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("cannot delegate to itself"),
        "unexpected message: {}",
        error.message()
    );
    // (2) A named target that is not a member of the target team.
    let error = agent
        .delegate_org_task(&core_task, QA, Some("ghost"))
        .await
        .expect_err("the router accepted an unknown target member");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .message()
            .contains("target member is not part of the target team"),
        "unexpected message: {}",
        error.message()
    );
    // (3) A cross-team hop between siblings, which no reporting line supports.
    let error = org
        .delegate(
            &engineering_task,
            ENGINEERING,
            "eng_leader",
            QA,
            Some("qa_leader"),
        )
        .await
        .expect_err("the router allowed a sibling delegation");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .message()
            .contains("delegation must follow the reporting line"),
        "unexpected message: {}",
        error.message()
    );
    assert_eq!(org.pool().open_task_count(CORE).await.unwrap(), core_before);
    assert_eq!(
        org.pool().open_task_count(ENGINEERING).await.unwrap(),
        engineering_before
    );
    assert_eq!(org.pool().open_task_count(QA).await.unwrap(), qa_before);
    for (task_id, owner) in [(&core_task, CORE), (&engineering_task, ENGINEERING)] {
        let task = org
            .task(task_id)
            .await
            .expect("the task lookup failed")
            .expect("the task vanished");
        assert_eq!(
            &task.target_team, owner,
            "a refused hop moved task {task_id}"
        );
        assert!(
            task.assignee.is_none(),
            "a refused hop assigned task {task_id}"
        );
        assert_eq!(task.status, TaskStatus::Pending);
    }
}
#[tokio::test]
async fn the_router_enforces_depth_budget_and_role_pairs() {
    let fixture = organized_fixture().await;
    let org = fixture.agent.org().expect("no organisation was installed");
    let router = org.router();
    let allowed = DelegationRequest::new(CORE, LEADER, ENGINEERING)
        .with_target_member("dev-1")
        .with_open_tasks(0);
    match router.decide(&allowed) {
        DelegationDecision::Allowed { depth, cross_team } => {
            assert_eq!(depth, 1, "a first hop reads as depth 1");
            assert!(cross_team, "core -> engineering crosses a team boundary");
        }
        DelegationDecision::Refused { refusal } => {
            panic!("the router refused an allowed hop: {refusal}")
        }
    }
    assert!(router.check(&allowed).is_ok());
    let too_deep = DelegationRequest::new(CORE, LEADER, ENGINEERING)
        .with_depth(3)
        .with_open_tasks(0);
    assert_eq!(
        router.decide(&too_deep).refusal(),
        Some(&DelegationRefusal::DepthExceeded {
            requested: 3,
            max: 2
        })
    );
    let error = router
        .check(&too_deep)
        .expect_err("the router allowed an over-deep hop");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .message()
            .contains("delegation depth 3 exceeds the maximum of 2"),
        "unexpected message: {}",
        error.message()
    );
    let over_budget = DelegationRequest::new(CORE, LEADER, QA).with_open_tasks(2);
    assert_eq!(
        router.decide(&over_budget).refusal(),
        Some(&DelegationRefusal::BudgetExhausted { open: 2, budget: 2 })
    );
    let bad_pair = DelegationRequest::new(ENGINEERING, "dev-1", QA).with_target_member("reviewer");
    assert_eq!(
        router.decide(&bad_pair).refusal(),
        Some(&DelegationRefusal::RoleNotAllowed {
            from: OrgRole::Member,
            to: OrgRole::Member
        })
    );
    assert_eq!(router.rules().max_delegation_depth, 2);
}
#[tokio::test]
async fn an_org_task_runs_from_create_to_complete_in_the_reserved_namespace() {
    let fixture = organized_fixture().await;
    let agent = &fixture.agent;
    let org = agent.org().expect("no organisation was installed");
    let namespace = org_namespace(ORG_ID);
    let created = agent
        .create_org_task(
            OrgTaskSpec::new(CORE, "ship the release", "cut the 1.0 candidate")
                .for_team(ENGINEERING),
        )
        .await
        .expect("the org task was not created");
    let task_id = created.task.task_id.clone();
    assert_eq!(created.task.status, TaskStatus::Pending);
    assert_eq!(created.task.depth, 0);
    assert_eq!(created.task.owner_team, CORE);
    assert_eq!(created.task.target_team, ENGINEERING);
    assert_eq!(created.task.created_by, "core::team_leader");
    let record = fixture
        .store
        .task(&namespace, &task_id)
        .await
        .expect("the store lookup failed")
        .expect("the org task did not land in the reserved namespace");
    assert_eq!(record.team_name, namespace);
    assert_eq!(record.title, "ship the release");
    assert_eq!(record.status, TaskStatus::Pending);
    assert!(
        fixture
            .store
            .task(CORE, &task_id)
            .await
            .expect("the board lookup failed")
            .is_none(),
        "an org task leaked onto the team's own board"
    );
    let rows = fixture
        .store
        .list_tasks(&namespace, &TaskFilter::default())
        .await
        .expect("the namespace listing failed");
    assert_eq!(rows.len(), 1, "the namespace holds {rows:?}");
    let note = record.note.as_deref().expect("the row carries no sidecar");
    let sidecar: OrgTaskSidecar = serde_json::from_str(note).expect("the sidecar did not decode");
    assert_eq!(sidecar.owner_team, CORE);
    assert_eq!(sidecar.target_team, ENGINEERING);
    assert_eq!(sidecar.depth, 0);
    assert!(sidecar.failure.is_none());
    let claimed = org
        .pool()
        .claim(&task_id, ENGINEERING, "dev-1")
        .await
        .expect("the claim failed");
    assert!(
        claimed.is_ok(),
        "the claim was refused: {:?}",
        claimed.reason()
    );
    let claimed_task = claimed.task.expect("the claim carries the row");
    assert_eq!(claimed_task.status, TaskStatus::InProgress);
    assert_eq!(claimed_task.assignee.as_deref(), Some("engineering::dev-1"));
    let completed = org
        .pool()
        .complete(&task_id, ENGINEERING, "dev-1", "the candidate is cut")
        .await
        .expect("the completion failed");
    assert!(
        completed.is_ok(),
        "the completion was refused: {:?}",
        completed.reason()
    );
    let done = completed.task.expect("the completion carries the row");
    assert_eq!(done.status, TaskStatus::Completed);
    assert_eq!(done.summary.as_deref(), Some("the candidate is cut"));
    assert!(done.is_terminal());
    assert_eq!(org.pool().open_task_count(ENGINEERING).await.unwrap(), 0);
    let record = fixture
        .store
        .task(&namespace, &task_id)
        .await
        .expect("the store lookup failed")
        .expect("the org task vanished");
    assert_eq!(record.status, TaskStatus::Completed);
    assert_eq!(record.assignee.as_deref(), Some("engineering::dev-1"));
}
#[tokio::test]
async fn a_task_for_a_non_member_target_is_refused() {
    let fixture = organized_fixture().await;
    let agent = &fixture.agent;
    let org = agent.org().expect("no organisation was installed");
    let unknown_team = agent
        .create_org_task(OrgTaskSpec::new(CORE, "leak work", "nowhere").for_team("ghost"))
        .await
        .expect_err("a task was created for an unknown team");
    assert_eq!(unknown_team.kind(), ErrorKind::Validation);
    assert!(
        unknown_team
            .message()
            .contains("unknown organization team 'ghost'"),
        "unexpected message: {}",
        unknown_team.message()
    );
    let unknown_creator = agent
        .create_org_task(OrgTaskSpec::new(CORE, "leak work", "nowhere").by("ghost"))
        .await
        .expect_err("a task was created by a non-member");
    assert_eq!(unknown_creator.kind(), ErrorKind::Validation);
    assert!(
        unknown_creator
            .message()
            .contains("creator 'ghost' is not a member of team 'core'"),
        "unexpected message: {}",
        unknown_creator.message()
    );
    let unknown_reviewer = agent
        .create_org_task(
            OrgTaskSpec::new(CORE, "leak work", "nowhere")
                .for_team(ENGINEERING)
                .reviewed_by(vec!["ghost".to_owned()]),
        )
        .await
        .expect_err("a task was created with a non-member reviewer");
    assert_eq!(unknown_reviewer.kind(), ErrorKind::Validation);
    assert!(
        unknown_reviewer
            .message()
            .contains("reviewer 'ghost' is not a member of team 'engineering'"),
        "unexpected message: {}",
        unknown_reviewer.message()
    );
    let rows = fixture
        .store
        .list_tasks(&org_namespace(ORG_ID), &TaskFilter::default())
        .await
        .expect("the namespace listing failed");
    assert!(rows.is_empty(), "a refused create wrote a row: {rows:?}");
    assert_eq!(org.pool().open_task_count(ENGINEERING).await.unwrap(), 0);
}
#[tokio::test]
async fn org_mail_is_acknowledged_only_after_the_leader_flush() {
    let fixture = organized_fixture().await;
    let agent = &fixture.agent;
    let org = agent.org().expect("no organisation was installed");
    let namespace = org_namespace(ORG_ID);
    let leader_mail = org
        .send_mail(QA, CORE, "release", "hand over the release")
        .await
        .expect("the org message was not sent");
    assert_eq!(leader_mail.from_team, QA);
    assert_eq!(leader_mail.to_team.as_deref(), Some(CORE));
    assert!(!leader_mail.read, "a freshly sent message is unread");
    let system_mail = org
        .mailbox()
        .notify(CORE, "the release train is frozen")
        .await
        .expect("the system notification was not sent");
    assert!(system_mail.is_system());
    assert_eq!(system_mail.kind.as_str(), "system");
    assert_eq!(system_mail.from_team, ORG_SYSTEM_SENDER);
    let rows = fixture
        .store
        .unread_messages(&namespace, CORE)
        .await
        .expect("the unread lookup failed");
    assert_eq!(rows.len(), 2, "expected two unread rows, got {rows:?}");
    for row in &rows {
        assert_eq!(row.protocol, "org", "the row lost its protocol tag");
        assert!(!row.is_read);
        assert_eq!(row.to_member_name.as_deref(), Some(CORE));
    }
    assert_eq!(org.mailbox().unread_count(CORE).await.unwrap(), 2);
    let leader = fixture
        .factory
        .harness(LEADER)
        .expect("no leader harness was built");
    assert!(
        leader.sends().is_empty(),
        "the leader was steered before any run: {:?}",
        leader.sends()
    );
    tokio::time::timeout(BOUND, agent.invoke(Value::Null, None))
        .await
        .expect("the run did not settle within the bound")
        .expect("the team run failed");
    let sends = leader.sends();
    let joined = sends.join("\n");
    assert!(
        joined.contains(&format!("org {QA}: hand over the release")),
        "the leader never received the org mail: {sends:?}"
    );
    assert!(
        joined.contains(&format!(
            "org {ORG_SYSTEM_SENDER}: the release train is frozen"
        )),
        "the leader never received the system notification: {sends:?}"
    );
    assert!(
        org.unread(CORE).await.expect("the inbox failed").is_empty(),
        "the delivered mail was not acknowledged"
    );
    assert_eq!(org.mailbox().unread_count(CORE).await.unwrap(), 0);
    assert!(
        fixture
            .store
            .unread_messages(&namespace, CORE)
            .await
            .expect("the unread lookup failed")
            .is_empty(),
        "the read state was not persisted for the acknowledged rows"
    );
}
#[tokio::test]
async fn a_team_without_an_organisation_keeps_the_flat_board() {
    let fixture = build(&flat_spec()).await;
    let agent = &fixture.agent;
    assert!(
        agent.org().is_none(),
        "a flat team installed an organisation"
    );
    assert!(agent.backend().org().is_none());
    let error = agent
        .create_org_task(OrgTaskSpec::new(CORE, "ship", "do it"))
        .await
        .expect_err("a flat team created an org task");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.message().contains("organisation"),
        "unexpected message: {}",
        error.message()
    );
    let error = agent
        .delegate_org_task("task_missing", ENGINEERING, None)
        .await
        .expect_err("a flat team delegated an org task");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.message().contains("organisation"),
        "unexpected message: {}",
        error.message()
    );
    let task = agent
        .create_task(TaskGraphSpec {
            title: "flat task".to_owned(),
            content: "the board still works".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .expect("the flat board refused a normal task");
    assert_eq!(task.title, "flat task");
    let board_rows = fixture
        .store
        .list_tasks(CORE, &TaskFilter::default())
        .await
        .expect("the board listing failed");
    assert_eq!(board_rows.len(), 1, "the flat board holds {board_rows:?}");
    let org_rows = fixture
        .store
        .list_tasks(&org_namespace(ORG_ID), &TaskFilter::default())
        .await
        .expect("the namespace listing failed");
    assert!(
        org_rows.is_empty(),
        "a flat team wrote org rows: {org_rows:?}"
    );
}
