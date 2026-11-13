//! `17_team_persistence` — a team that survives a restart, plus the organisation above it.
//!
//! Teaches the two durable seams of `openjiuwen::agent_teams`, the Rust port of Python
//! `openjiuwen/agent_teams/`:
//!
//! * `storage` — `TeamStore`, `open_store` and `StorageSpec`. The `sqlite` backend writes the
//!   team row, the roster, the task board, the dependency edges and the mailbox into one
//!   database file, so a second store opened on that file answers exactly like the first one
//!   (Python: `tools/database/` — `team_dao`, `member_dao`, `task_dao`, `message_dao` over
//!   `TeamDatabase`).
//! * `org` — `OrganizationSpec`/`OrgChart` (the reporting lines), `DelegationRouter` (which hops
//!   are legal) and `OrgTaskPool` (cross-team work, persisted through the *same* store under an
//!   `org::<id>` namespace; Python: `organization/`).
//!
//! Everything is offline and deterministic: a `tempfile` directory, no model, no network and no
//! harness session. The `sqlite` backend is reachable because the crate's dev-dependency enables
//! `openjiuwen-agent-teams/sqlite`.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use openjiuwen::agent_teams::org::{
    org_namespace, qualify_member, DelegationDecision, DelegationRequest, DelegationRouter,
    DelegationRules, OrgChart, OrgMember, OrgRole, OrgTask, OrgTaskFilter, OrgTaskOpResult,
    OrgTaskPool, OrgTaskSpec, OrgTeam, OrganizationSpec,
};
use openjiuwen::agent_teams::{
    open_store, MemberRecord, MemberStatus, MessageRecord, StorageSpec, TaskFilter, TaskRecord,
    TaskStatus, TeamInfoRecord, TeamRole, TeamStore,
};
use openjiuwen::core::common::error::ErrorKind;
use serde_json::json;

use common::banner;

// --- the cast --------------------------------------------------------------

/// Team every persisted row belongs to.
const TEAM: &str = "platform";
/// Leader member id of [`TEAM`]; also the organisation's owner and root delegator.
const LEADER: &str = "team_leader";
/// First teammate: builds what the team ships.
const BUILDER: &str = "builder";
/// Second teammate: reviews what the team ships.
const REVIEWER: &str = "reviewer";
/// Organisation id; its task-pool rows live under `org::<id>`.
const ORG_ID: &str = "acme-labs";
/// Database file inside the temporary directory.
const DB_FILE: &str = "team.db";
/// The task that must finish first.
const BUILD: &str = "task-build";
/// The task that depends on [`BUILD`].
const DESIGN: &str = "task-design";

// --- the mailbox -----------------------------------------------------------

/// Direct mailbox row id.
const MESSAGE_DIRECT: &str = "msg-direct";
/// Broadcast mailbox row id.
const MESSAGE_BROADCAST: &str = "msg-broadcast";
/// Caller-supplied stamp of [`MESSAGE_DIRECT`]; the store writes message stamps verbatim.
const STAMP_DIRECT: u64 = 1_700_000_000_000;
/// Caller-supplied stamp of [`MESSAGE_BROADCAST`], one millisecond later.
const STAMP_BROADCAST: u64 = 1_700_000_000_001;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;
    banner::title("17 team persistence");
    banner::note(&format!(
        "run mode: {}; no model is built here",
        mode.describe()
    ));

    let workspace = common::tempdir()?;
    let spec = sqlite_spec(&workspace.path().join(DB_FILE));

    // 1. Write the team, read it back, forget it, and read it again from a second store.
    let store = open_and_seed(&spec).await?;
    let before = snapshot(&store).await?;
    print_facts("before the restart", &before);
    drop(store);
    banner::title("restart on the same file");
    let store = open_store(&spec).context("failed to reopen the sqlite team store")?;
    banner::step(&format!(
        "a second store on `{DB_FILE}`: the first handle is gone, the file is all that is left"
    ));
    let after = snapshot(&store).await?;
    ensure!(
        before == after,
        "the restarted store answers differently: {after:?}"
    );
    print_facts("after the restart", &after);
    other_backends().await?;

    // 2. The organisation layer over the recovered rows, and its own durable task pool.
    org_layer(store).await?;

    banner::title("done");
    banner::answer(
        "the team, roster, task states and mailbox came back from the file; the chart routed its \
         work through the same database",
    );
    Ok(())
}

// --- part 1: the sqlite team store -----------------------------------------

/// The `StorageSpec` selecting the SQLite backend at `path`.
fn sqlite_spec(path: &Path) -> StorageSpec {
    let mut params = BTreeMap::new();
    params.insert("path".to_owned(), json!(path.to_string_lossy()));
    StorageSpec {
        kind: "sqlite".to_owned(),
        params,
    }
}

/// Opens the store and writes one of every row kind through the `TeamStore` trait.
async fn open_and_seed(spec: &StorageSpec) -> Result<Arc<dyn TeamStore>> {
    banner::title("sqlite team store");
    let store = open_store(spec).context("failed to open the sqlite team store")?;
    ensure!(
        store.name() == "sqlite",
        "expected the sqlite backend, got `{}`",
        store.name()
    );
    banner::step(&format!(
        "open_store(StorageSpec {{ kind: \"sqlite\", params: {{ path: \"{DB_FILE}\" }} }}) -> SqliteTeamStore"
    ));
    let team = TeamInfoRecord {
        display_name: "Platform".to_owned(),
        desc: "owns the release pipeline".to_owned(),
        leader_member_name: LEADER.to_owned(),
        ..TeamInfoRecord::new(TEAM)
    };
    store
        .upsert_team(team)
        .await
        .context("failed to write the team row")?;
    seed_roster(&store).await?;
    seed_board(&store).await?;
    seed_mailbox(&store).await?;
    Ok(store)
}

/// Inserts the leader and the two teammates, each started through the lifecycle CAS.
async fn seed_roster(store: &Arc<dyn TeamStore>) -> Result<()> {
    for (name, role, desc) in [
        (LEADER, TeamRole::Leader, "leads the team"),
        (BUILDER, TeamRole::Teammate, "builds the pipeline"),
        (REVIEWER, TeamRole::Teammate, "reviews the release"),
    ] {
        let member = MemberRecord {
            display_name: name.to_owned(),
            desc: desc.to_owned(),
            ..MemberRecord::new(TEAM, name, role)
        };
        store
            .insert_member(member)
            .await
            .with_context(|| format!("failed to insert `{name}`"))?;
        cas_member(
            store,
            name,
            MemberStatus::Unstarted,
            MemberStatus::Starting,
            true,
        )
        .await?;
        cas_member(
            store,
            name,
            MemberStatus::Starting,
            MemberStatus::Ready,
            true,
        )
        .await?;
    }
    // A replayed CAS loses instead of failing: the row is no longer `unstarted`.
    cas_member(
        store,
        LEADER,
        MemberStatus::Unstarted,
        MemberStatus::Starting,
        false,
    )
    .await?;
    banner::step(
        "roster: leader + two teammates, each unstarted -> starting -> ready by CAS \
         (a replayed CAS answers false)",
    );
    Ok(())
}

/// Writes two tasks plus the edge between them, then walks both through their states.
async fn seed_board(store: &Arc<dyn TeamStore>) -> Result<()> {
    for (task, title, content) in [
        (BUILD, "build the pipeline", "run the build stage"),
        (DESIGN, "design the pipeline", "draw the stages first"),
    ] {
        store
            .insert_task(TaskRecord::new(TEAM, task, title, content))
            .await
            .with_context(|| format!("failed to insert `{task}`"))?;
    }
    store
        .add_dependency(TEAM, DESIGN, BUILD)
        .await
        .context("failed to record the dependency edge")?;
    banner::step(&format!(
        "two tasks on the board; `{DESIGN}` depends on `{BUILD}`"
    ));
    banner::note(
        "add_dependency records the edge only: parking the dependent task in `blocked` is the task \
         board's job, not the store's",
    );

    // Every status change is one compare-and-set, so a replayed claim simply loses.
    cas_task(
        store,
        BUILD,
        TaskStatus::Pending,
        TaskStatus::InProgress,
        true,
    )
    .await?;
    cas_task(
        store,
        BUILD,
        TaskStatus::Pending,
        TaskStatus::InProgress,
        false,
    )
    .await?;
    for (task, from, to) in [
        (BUILD, TaskStatus::InProgress, TaskStatus::Completed),
        (DESIGN, TaskStatus::Pending, TaskStatus::InProgress),
        (DESIGN, TaskStatus::InProgress, TaskStatus::Completed),
    ] {
        cas_task(store, task, from, to, true).await?;
    }
    banner::step(
        "each status change is one CAS: pending -> in_progress -> completed, and the replayed \
         claim answers false",
    );
    Ok(())
}

/// One member lifecycle compare-and-set; `win` is the answer the walkthrough expects.
async fn cas_member(
    store: &Arc<dyn TeamStore>,
    member: &str,
    from: MemberStatus,
    to: MemberStatus,
    win: bool,
) -> Result<()> {
    let answer = store.transition_member(TEAM, member, from, to).await?;
    ensure!(
        answer == win,
        "`{member}` {} -> {} answered {answer}",
        from.as_str(),
        to.as_str()
    );
    Ok(())
}

/// One task compare-and-set; `win` is the answer the walkthrough expects.
async fn cas_task(
    store: &Arc<dyn TeamStore>,
    task: &str,
    from: TaskStatus,
    to: TaskStatus,
    win: bool,
) -> Result<()> {
    let answer = store.transition_task(TEAM, task, from, to).await?;
    ensure!(
        answer == win,
        "`{task}` {} -> {} answered {answer}",
        from.as_str(),
        to.as_str()
    );
    Ok(())
}

/// Writes one direct message and one broadcast, then clears both through their own read paths.
async fn seed_mailbox(store: &Arc<dyn TeamStore>) -> Result<()> {
    let direct = MessageRecord::direct(
        TEAM,
        MESSAGE_DIRECT,
        LEADER,
        BUILDER,
        "start with the build stage",
        STAMP_DIRECT,
    );
    store
        .insert_message(direct)
        .await
        .context("failed to insert the direct message")?;
    let broadcast = MessageRecord::broadcast(
        TEAM,
        MESSAGE_BROADCAST,
        LEADER,
        "standup at ten",
        STAMP_BROADCAST,
    );
    store
        .insert_message(broadcast)
        .await
        .context("failed to insert the broadcast")?;
    ensure!(
        store.unread_messages(TEAM, BUILDER).await?.len() == 2
            && store.unread_messages(TEAM, LEADER).await?.is_empty(),
        "the builder sees both rows and a member never receives its own broadcast"
    );
    store
        .mark_read(TEAM, BUILDER, &[MESSAGE_DIRECT.to_owned()])
        .await
        .context("failed to mark the direct message read")?;
    store
        .set_read_watermark(TEAM, BUILDER, STAMP_BROADCAST)
        .await
        .context("failed to advance the broadcast watermark")?;
    ensure!(
        store.message_count(TEAM, BUILDER).await? == 0,
        "the builder mailbox must be empty once both rows are read"
    );
    banner::step(
        "mailbox: 2 rows for the builder; `mark_read` clears the direct row, \
         `set_read_watermark` the broadcast -> 0 unread for it",
    );
    Ok(())
}

/// Reads every row kind back through the trait, asserts the shape, and returns the facts.
async fn snapshot(store: &Arc<dyn TeamStore>) -> Result<Vec<String>> {
    let team = store.team(TEAM).await?.context("the team row is missing")?;
    ensure!(
        team.leader_member_name == LEADER,
        "the persisted leader is not `{LEADER}`"
    );
    let roster = store
        .list_members(TEAM)
        .await
        .context("failed to list the roster")?;
    ensure!(
        roster.len() == 3 && roster.iter().all(|row| row.status == MemberStatus::Ready),
        "the roster lost a member or a lifecycle state"
    );
    let mut tasks = store
        .list_tasks(TEAM, &TaskFilter::default())
        .await
        .context("failed to list the board")?;
    // Sorted by id: the store orders by `updated_at`, which must not shape the story.
    tasks.sort_by(|left, right| left.task_id.cmp(&right.task_id));
    ensure!(
        tasks.len() == 2
            && tasks
                .iter()
                .all(|task| task.status == TaskStatus::Completed),
        "the task states are wrong"
    );
    ensure!(
        store.dependencies(TEAM, DESIGN).await? == vec![BUILD.to_owned()],
        "the dependency edge is missing"
    );
    let unread = store
        .unread_messages(TEAM, REVIEWER)
        .await
        .context("failed to read the reviewer mailbox")?;
    ensure!(
        unread.len() == 1 && unread[0].message_id == MESSAGE_BROADCAST,
        "the reviewer must have exactly the broadcast unread"
    );
    ensure!(
        store.read_watermark(TEAM, BUILDER).await? == Some(STAMP_BROADCAST)
            && store.message_count(TEAM, BUILDER).await? == 0,
        "the builder's read state is wrong"
    );
    Ok(vec![
        format!(
            "team `{}` (leader `{}`, dispatch `{}`)",
            team.display_name_or_default(),
            team.leader_member_name,
            team.dispatch_mode.as_str()
        ),
        format!(
            "roster {}",
            roster
                .iter()
                .map(|row| format!(
                    "{} ({}/{})",
                    row.member_name,
                    row.role.as_str(),
                    row.status.as_str()
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        format!(
            "board {}",
            tasks
                .iter()
                .map(|task| format!(
                    "{} ({} -> {})",
                    task.task_id,
                    task.title,
                    task.status.as_str()
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        format!(
            "`{DESIGN}` depends on `{BUILD}`; reviewer has 1 unread (\"{}\"); builder's watermark \
             kept, 0 unread for it",
            unread[0].content
        ),
    ])
}

/// Prints the facts a [`snapshot`] returned.
fn print_facts(when: &str, facts: &[String]) {
    for fact in facts {
        banner::step(&format!("{when}: {fact}"));
    }
}

/// Shows the other `StorageSpec` outcomes: an empty `memory` store and typed deferrals.
async fn other_backends() -> Result<()> {
    banner::title("the other backends");
    let memory =
        open_store(&StorageSpec::in_memory()).context("failed to open the memory store")?;
    ensure!(
        memory.team(TEAM).await?.is_none(),
        "a fresh in-memory store must hold no team row"
    );
    banner::step(
        "kind \"memory\": a fresh store is empty, so file storage is what makes the restart work",
    );
    for kind in ["postgresql", "mysql"] {
        let spec = StorageSpec {
            kind: kind.to_owned(),
            params: BTreeMap::new(),
        };
        let Err(err) = open_store(&spec) else {
            anyhow::bail!("storage backend '{kind}' must report a deferral, not degrade to memory");
        };
        ensure!(
            err.kind() == ErrorKind::Unsupported,
            "'{kind}' must be Unsupported, got {}",
            err.kind()
        );
        banner::step(&format!(
            "kind \"{kind}\": {} — {}",
            err.kind(),
            err.message()
        ));
    }
    Ok(())
}

// --- part 2: the organisation layer ----------------------------------------

/// Builds the chart, prints it, routes delegations and runs one task through the org pool.
async fn org_layer(store: Arc<dyn TeamStore>) -> Result<()> {
    banner::title("organisation chart");
    let chart = Arc::new(OrgChart::new(org_spec()).context("failed to build the chart")?);
    banner::step(&format!(
        "OrgChart::new validated `{}`: every reporting line names a node that exists",
        chart.organization_id()
    ));
    print_chart(&chart)?;
    bind_roster(&store, &chart).await?;

    banner::title("delegation router");
    // Not on this walkthrough: the organisation mailbox (`OrgMailbox`), the bounded fan-out
    // runner (`SwarmflowPlan` / `run_swarmflow`) and fork/compact. Each is a self-contained
    // seam of the same `org` module and deserves its own example.
    let rules = DelegationRules::new()
        .with_max_depth(1)
        .with_require_reporting_line(true)
        // Only a leader hands work down, so a member -> member hop stays refused.
        .allowing(OrgRole::Leader, OrgRole::Member)
        .allowing(OrgRole::Leader, OrgRole::Leader)
        // Two open tasks per team before the router reports the budget exhausted.
        .with_default_budget(2);
    let router = Arc::new(DelegationRouter::new(Arc::clone(&chart), rules));
    banner::step(
        "rules: one hop, reporting line required, leader -> member only, two open tasks per team",
    );
    for (label, request) in delegation_cases() {
        banner::step(&format!("{label}: {}", decision(&router, &request)));
    }
    org_task_round_trip(&store, &router).await
}

/// The delegation requests under test: one allowed hop and four refusals.
fn delegation_cases() -> Vec<(String, DelegationRequest)> {
    let hop = || {
        DelegationRequest::new(TEAM, LEADER, TEAM)
            .with_target_member(BUILDER)
            .with_open_tasks(0)
    };
    vec![
        (format!("leader -> {TEAM}::{BUILDER}"), hop()),
        (
            format!("{BUILDER} -> {TEAM}::{REVIEWER}"),
            DelegationRequest::new(TEAM, BUILDER, TEAM)
                .with_target_member(REVIEWER)
                .with_open_tasks(0),
        ),
        (
            format!("leader -> {TEAM}::{BUILDER} at depth 2"),
            hop().with_depth(2),
        ),
        (
            format!("leader -> {TEAM}::{BUILDER} with 2 open tasks"),
            hop().with_open_tasks(2),
        ),
        (
            "leader -> release (absent team)".to_owned(),
            DelegationRequest::new(TEAM, LEADER, "release"),
        ),
    ]
}

/// The declarative organisation: one team, its leader and the two members below it.
fn org_spec() -> OrganizationSpec {
    let leader = qualify_member(TEAM, LEADER);
    let team = OrgTeam {
        team_name: TEAM.to_owned(),
        display_name: "Platform".to_owned(),
        desc: "owns the release pipeline".to_owned(),
        role: OrgRole::Leader,
        leader: Some(LEADER.to_owned()),
        ..OrgTeam::default()
    };
    let member = |id: &str, role: OrgRole, reports_to: String, desc: &str| OrgMember {
        member_id: id.to_owned(),
        team_name: TEAM.to_owned(),
        display_name: id.to_owned(),
        desc: desc.to_owned(),
        role,
        reports_to: Some(reports_to),
        ..OrgMember::default()
    };
    OrganizationSpec {
        organization_id: ORG_ID.to_owned(),
        display_name: "Acme Labs".to_owned(),
        desc: "one platform team and its reporting lines".to_owned(),
        owner: Some(leader.clone()),
        teams: vec![team],
        members: vec![
            member(LEADER, OrgRole::Leader, TEAM.to_owned(), "leads the team"),
            member(
                BUILDER,
                OrgRole::Member,
                leader.clone(),
                "builds the pipeline",
            ),
            member(REVIEWER, OrgRole::Member, leader, "reviews the release"),
        ],
        ..OrganizationSpec::default()
    }
}

/// Prints the chart as indented lines, roots first.
fn print_chart(chart: &OrgChart) -> Result<()> {
    let mut nodes: Vec<String> = chart
        .teams()
        .iter()
        .map(|team| team.team_name.clone())
        .collect();
    nodes.extend(chart.members().iter().map(|member| member.node_id()));
    let mut roots: Vec<&String> = nodes
        .iter()
        .filter(|node| chart.parent_of(node).is_none())
        .collect();
    roots.sort();
    for root in roots {
        print_node(chart, root, 0)?;
    }
    Ok(())
}

/// Prints one chart node and, recursively, everything that reports to it.
fn print_node(chart: &OrgChart, node_id: &str, depth: usize) -> Result<()> {
    let role = chart.role_of(node_id).map_or("unknown", OrgRole::as_str);
    let label = chart.team(node_id).map_or_else(
        || node_id.to_owned(),
        |team| format!("{} (team)", team.display_name),
    );
    banner::step(&format!("{}{label} [{role}]", "  ".repeat(depth)));
    for child in chart
        .direct_reports(node_id)
        .with_context(|| format!("failed to read the reports of `{node_id}`"))?
    {
        print_node(chart, &child, depth + 1)?;
    }
    Ok(())
}

/// Binds the persisted roster to the chart nodes, mapping each team role to an org role.
async fn bind_roster(store: &Arc<dyn TeamStore>, chart: &OrgChart) -> Result<()> {
    banner::title("roster and chart agree");
    let roster = store
        .list_members(TEAM)
        .await
        .context("failed to read the roster for the chart")?;
    ensure!(roster.len() == 3, "the chart binds all three roster rows");
    for row in &roster {
        let node = chart.member(TEAM, &row.member_name).with_context(|| {
            format!(
                "roster member `{}` is missing from the chart",
                row.member_name
            )
        })?;
        // The roster has no second role vocabulary: the chart's `OrgRole` is derived from
        // `TeamRole`, never stored twice.
        ensure!(
            node.role == org_role_of(row.role),
            "`{}` is `{}` on the roster but `{}` in the chart",
            row.member_name,
            row.role.as_str(),
            node.role.as_str()
        );
        banner::step(&format!(
            "{} -> {} [{}]",
            row.member_name,
            node.node_id(),
            node.role.as_str()
        ));
    }
    Ok(())
}

/// Maps a stored team role onto the organisation role the chart expects.
fn org_role_of(role: TeamRole) -> OrgRole {
    match role {
        TeamRole::Leader => OrgRole::Leader,
        // Human-agent, bridge and external-CLI rows are plain workers on the chart.
        _ => OrgRole::Member,
    }
}

/// Renders one delegation verdict as a stable sentence.
fn decision(router: &DelegationRouter, request: &DelegationRequest) -> String {
    match router.decide(request) {
        DelegationDecision::Allowed { depth, cross_team } => {
            format!("allowed (depth {depth}, cross_team {cross_team})")
        }
        DelegationDecision::Refused { refusal } => {
            format!("refused [{}]: {}", refusal.code(), refusal.message())
        }
    }
}

/// Unwraps a pool answer: a refusal keeps its sentence, an applied operation keeps its row.
fn applied(answer: OrgTaskOpResult, what: &str) -> Result<OrgTask> {
    ensure!(
        answer.ok,
        "{what} was refused: {}",
        answer.reason().unwrap_or("no reason recorded")
    );
    answer
        .task
        .with_context(|| format!("{what} carried no task row"))
}

/// Creates and delegates one organisation task, then settles it through the pool.
async fn org_task_round_trip(
    store: &Arc<dyn TeamStore>,
    router: &Arc<DelegationRouter>,
) -> Result<()> {
    banner::title("organisation task pool");
    let pool = OrgTaskPool::new(Arc::clone(store), Arc::clone(router))
        .context("failed to build the organisation task pool")?;
    ensure!(
        pool.namespace() == org_namespace(ORG_ID),
        "the pool must write under `org::<organization_id>`"
    );
    banner::step(&format!(
        "pool rows use the same store, under the reserved namespace `{}`",
        pool.namespace()
    ));

    let created = pool
        .create(OrgTaskSpec::new(TEAM, "cut the release", "tag 1.0"))
        .await
        .context("failed to create the organisation task")?;
    ensure!(
        created.task.status == TaskStatus::Pending,
        "a new org task starts pending"
    );
    banner::step(&format!(
        "create `{}` -> pending, depth {}, owner team `{}`, {} event(s)",
        created.task.title,
        created.task.depth,
        created.task.owner_team,
        created.events.len()
    ));
    let task_id = created.task.task_id.clone();

    // A refused hop is a policy answer (`ok = false`), never an `Err`.
    let refused = pool
        .delegate(&task_id, LEADER, "release", None)
        .await
        .context("failed to delegate the organisation task")?;
    ensure!(!refused.ok, "a hop to an absent team must be refused");
    banner::step(&format!(
        "delegate to an absent team -> refused: {}",
        refused.reason().unwrap_or("no reason recorded")
    ));

    let delegated = applied(
        pool.delegate(&task_id, LEADER, TEAM, Some(BUILDER))
            .await
            .context("failed to delegate the organisation task")?,
        "the delegation",
    )?;
    ensure!(
        delegated.assignee.as_deref() == Some(qualify_member(TEAM, BUILDER).as_str()),
        "the delegation did not route the task to the builder"
    );
    banner::step(&format!(
        "delegate to {TEAM}::{BUILDER} -> allowed, assignee `{}`, status `{}`",
        delegated.assignee.as_deref().unwrap_or("unassigned"),
        delegated.status.as_str()
    ));
    finish_org_task(pool, store, &task_id).await
}

/// Claims, completes and then locates the org row through the plain store trait.
async fn finish_org_task(
    pool: OrgTaskPool,
    store: &Arc<dyn TeamStore>,
    task_id: &str,
) -> Result<()> {
    let claimed = applied(
        pool.claim(task_id, TEAM, BUILDER)
            .await
            .context("failed to claim the organisation task")?,
        "the claim",
    )?;
    banner::step(&format!(
        "claim by {TEAM}::{BUILDER} -> status `{}` (one CAS winner)",
        claimed.status.as_str()
    ));

    let completed = applied(
        pool.complete(task_id, TEAM, BUILDER, "published 1.0")
            .await
            .context("failed to complete the organisation task")?,
        "the completion",
    )?;
    banner::step(&format!(
        "complete -> status `{}`, summary `{}`, {} open task(s) left for {TEAM}",
        completed.status.as_str(),
        completed.summary.as_deref().unwrap_or("none"),
        pool.open_task_count(TEAM).await?
    ));
    ensure!(
        pool.list(&OrgTaskFilter::default()).await?.len() == 1,
        "the pool must hold exactly the one completed task"
    );

    // The organisation rows are visible through the plain store trait, in their own namespace.
    let org_rows = store
        .list_tasks(&org_namespace(ORG_ID), &TaskFilter::default())
        .await
        .context("failed to read the organisation namespace")?;
    let board_rows = store
        .list_tasks(TEAM, &TaskFilter::default())
        .await
        .context("failed to read the team board namespace")?;
    ensure!(
        org_rows.len() == 1 && board_rows.len() == 2,
        "one database must hold both namespaces without mixing them"
    );
    banner::step(&format!(
        "store rows: {} under `{}`, {} on the team board `{TEAM}` — one database, two namespaces",
        org_rows.len(),
        org_namespace(ORG_ID),
        board_rows.len()
    ));
    Ok(())
}
