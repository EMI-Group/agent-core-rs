//! Member lifecycle and mail, driven entirely through the crate's public API.
//!
//! The suite pins the observable contracts a host of `openjiuwen-agent-teams` relies on:
//!
//! 1. spawning a teammate builds exactly one harness through the injected factory, starts it once
//!    and settles the roster row on `ready`; a factory failure leaves the row `unstarted`;
//! 2. retirement follows the member status machine: `dormant` for a graceful `Dormant` lifetime,
//!    `shut_down` for a forced or `Shutdown` one, and a never-started row rejects dormancy while
//!    still accepting `shut_down`;
//! 3. `auto_start_all` hands every member a distinct directory under the workspace provider's
//!    root, allocated in roster order under the team's own session;
//! 4. mail is one persisted row per mailbox plus one bus event per send: `message` for a direct or
//!    multicast send (de-duplicated targets), `broadcast` for a team-wide one, and `mark_read`
//!    only clears the named direct ids;
//! 5. the leader handle creates tasks, the board row and its `task_created` event agree, and the
//!    roster accessors are sorted and agree with each other.
//!
//! Everything is offline: `support` answers every member input with one scripted observation
//! triple and allocates member directories under one temporary root, and every wait is bounded.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

mod support;

use std::collections::BTreeSet;

use openjiuwen_agent_teams::schema::{
    MemberLifetime, MemberStatus, TaskGraphSpec, TaskStatus, TeamRole, TeamTopic,
};
use openjiuwen_agent_teams::storage::{MessageRecord, TeamStore};
use openjiuwen_agent_teams::workspace::IsolationMode;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use tokio::time::timeout;

use support::{
    event_types, recorded, sample_spec, subscribe_observer, wait_until, Call, Fixture, BOUND,
};

#[tokio::test]
async fn spawning_a_teammate_builds_one_harness_and_leaves_a_ready_row() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;

    timeout(BOUND, agent.spawn_member("dev-1"))
        .await
        .expect("the spawn did not finish")
        .expect("dev-1 did not spawn");

    wait_until("dev-1 to be ready", || {
        agent
            .backend()
            .member("dev-1")
            .is_some_and(|member| member.status() == MemberStatus::Ready)
    })
    .await;

    let member = agent.backend().member("dev-1").expect("the dev-1 row");
    assert_eq!(member.role(), TeamRole::Teammate);
    assert!(member.runtime().is_some(), "a spawned row owns a brain");
    let stored = fixture
        .store
        .member("alpha", "dev-1")
        .await
        .unwrap()
        .expect("the roster row is persisted");
    assert_eq!(stored.status, MemberStatus::Ready, "the store agrees");

    let creations = fixture
        .factory
        .created_members()
        .iter()
        .filter(|name| name.as_str() == "dev-1")
        .count();
    assert_eq!(creations, 1, "one harness is built for dev-1");
    let harness = fixture.harness("dev-1");
    assert_eq!(
        harness.calls(),
        [Call::Start],
        "the harness was started exactly once"
    );

    // A second spawn is a no-op: no second harness and no second start.
    timeout(BOUND, agent.spawn_member("dev-1"))
        .await
        .expect("the idempotent spawn did not finish")
        .expect("the idempotent spawn failed");
    let creations = fixture
        .factory
        .created_members()
        .iter()
        .filter(|name| name.as_str() == "dev-1")
        .count();
    assert_eq!(creations, 1, "the second spawn built no harness");
    assert_eq!(
        harness.calls(),
        [Call::Start],
        "and it did not start the harness again"
    );
}

#[tokio::test]
async fn a_factory_failure_leaves_the_roster_row_unstarted() {
    let fixture = Fixture::new();
    fixture.factory.failing_for("dev-1");
    let agent = fixture.activate(&sample_spec(), "session-a").await;

    let error = timeout(BOUND, agent.spawn_member("dev-1"))
        .await
        .expect("the failing spawn did not finish")
        .expect_err("the failing factory must fail the spawn");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error
            .to_string()
            .contains("fake factory failure for 'dev-1'"),
        "the factory's own error is propagated: {error}"
    );

    let member = agent.backend().member("dev-1").expect("the row survives");
    assert_eq!(member.status(), MemberStatus::Unstarted);
    assert!(member.runtime().is_none(), "no brain is attached");
    let stored = fixture
        .store
        .member("alpha", "dev-1")
        .await
        .unwrap()
        .expect("the roster row is persisted");
    assert_eq!(stored.status, MemberStatus::Unstarted, "the store agrees");
    assert!(
        fixture.factory.harness("dev-1").is_none(),
        "the factory handed back no harness"
    );
}

#[tokio::test]
async fn spawning_an_unknown_member_is_not_found() {
    let fixture = Fixture::new();
    // Without the shared `agents["teammate"]` fallback there is no blueprint for an off-roster
    // name at all, which is the `not_found` half of `member_spec`.
    let mut spec = sample_spec();
    spec.agents.remove("teammate");
    let agent = fixture.activate(&spec, "session-a").await;
    let built = fixture.factory.created_members().len();

    let error = timeout(BOUND, agent.spawn_member("nobody"))
        .await
        .expect("the rejected spawn did not finish")
        .expect_err("an off-roster name must not spawn");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error
            .to_string()
            .contains("member 'nobody' not found in team 'alpha'"),
        "{error}"
    );
    assert_eq!(
        fixture.factory.created_members().len(),
        built,
        "an unknown member builds nothing"
    );
    assert!(
        agent.backend().member("nobody").is_none(),
        "a rejected spawn leaves no row behind"
    );
}

#[tokio::test]
async fn a_grammar_valid_name_falls_back_to_the_teammate_blueprint() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;

    // `TeamAgentSpec::member_spec` derives an unlisted name from `agents["teammate"]`, so the
    // spawn seam — not the roster — is what rejects a name it has no blueprint for.
    timeout(BOUND, agent.spawn_member("nobody"))
        .await
        .expect("the fallback spawn did not finish")
        .expect("a grammar-valid name spawns on the teammate blueprint");

    let member = agent
        .backend()
        .member("nobody")
        .expect("the row was created");
    assert_eq!(member.role(), TeamRole::Teammate);
    assert_eq!(member.status(), MemberStatus::Ready);
    assert!(member.runtime().is_some());
}

#[tokio::test]
async fn shutdown_member_settles_a_ready_member_on_the_requested_lifetime() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;
    for name in ["dev-1", "teammate"] {
        timeout(BOUND, agent.spawn_member(name))
            .await
            .expect("the spawn did not finish")
            .expect("the member did not spawn");
    }

    let dormant = agent
        .shutdown_member("dev-1", MemberLifetime::Dormant, false)
        .await
        .unwrap();
    assert_eq!(dormant, MemberStatus::Dormant);
    assert_eq!(
        agent.backend().member("dev-1").expect("the row").status(),
        MemberStatus::Dormant,
        "the cached row agrees with the returned status"
    );
    let stored = fixture
        .store
        .member("alpha", "dev-1")
        .await
        .unwrap()
        .expect("the roster row is persisted");
    assert_eq!(stored.status, MemberStatus::Dormant, "the store agrees");

    let forced = agent
        .shutdown_member("dev-1", MemberLifetime::Dormant, true)
        .await
        .unwrap();
    assert_eq!(
        forced,
        MemberStatus::ShutDown,
        "force escalates a dormant member past dormancy"
    );
    let stored = fixture
        .store
        .member("alpha", "dev-1")
        .await
        .unwrap()
        .expect("the roster row is persisted");
    assert_eq!(stored.status, MemberStatus::ShutDown);

    let shutdown = agent
        .shutdown_member("teammate", MemberLifetime::Shutdown, false)
        .await
        .unwrap();
    assert_eq!(shutdown, MemberStatus::ShutDown, "graceful shutdown");
    assert_eq!(
        agent
            .backend()
            .member("teammate")
            .expect("the row")
            .status(),
        MemberStatus::ShutDown
    );
}

#[tokio::test]
async fn shutdown_member_of_an_unstarted_row_rejects_dormancy_but_accepts_shutdown() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;
    assert_eq!(
        agent.backend().member("dev-1").expect("the row").status(),
        MemberStatus::Unstarted
    );

    let error = agent
        .shutdown_member("dev-1", MemberLifetime::Dormant, false)
        .await
        .expect_err("dormancy is unreachable from unstarted");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("member 'dev-1' cannot shut down from status 'unstarted'"),
        "{error}"
    );
    assert_eq!(
        agent.backend().member("dev-1").expect("the row").status(),
        MemberStatus::Unstarted,
        "a rejected retirement leaves the row untouched"
    );

    let status = agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, false)
        .await
        .unwrap();
    assert_eq!(
        status,
        MemberStatus::ShutDown,
        "a never-started member still retires to shut_down"
    );
    let stored = fixture
        .store
        .member("alpha", "dev-1")
        .await
        .unwrap()
        .expect("the roster row is persisted");
    assert_eq!(stored.status, MemberStatus::ShutDown);
}

#[tokio::test]
async fn shutdown_member_of_an_unknown_name_is_not_found() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;

    let error = agent
        .shutdown_member("nobody", MemberLifetime::Shutdown, true)
        .await
        .expect_err("an off-roster name cannot retire");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error
            .to_string()
            .contains("member 'nobody' not found in team 'alpha'"),
        "{error}"
    );
}

#[tokio::test]
async fn auto_start_all_gives_every_member_a_distinct_workspace_under_the_provider_root() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;

    timeout(BOUND, agent.auto_start_all())
        .await
        .expect("auto_start_all did not finish")
        .expect("a teammate did not start");

    for name in ["dev-1", "teammate"] {
        let member = agent.backend().member(name).expect("the roster row");
        assert_eq!(member.status(), MemberStatus::Ready, "{name}");
        assert!(member.runtime().is_some(), "{name} owns a brain");
    }

    let contexts = fixture.factory.calls();
    assert_eq!(contexts.len(), 3, "the leader plus two teammates");
    let mut roots: BTreeSet<std::path::PathBuf> = BTreeSet::new();
    for context in &contexts {
        assert_eq!(
            Some(context.workspace_root.clone()),
            fixture.workspace.root_for(&context.member_name),
            "the harness was handed the provider's directory for its member"
        );
        assert!(
            context.workspace_root.is_dir(),
            "{} exists on disk",
            context.workspace_root.display()
        );
        assert!(
            context.workspace_root.starts_with(fixture.workspace.root()),
            "{} lives under the provider root",
            context.workspace_root.display()
        );
        roots.insert(context.workspace_root.clone());
    }
    assert_eq!(roots.len(), 3, "every member owns a distinct root");

    let requests = fixture.workspace.requests();
    let requested: Vec<&str> = requests
        .iter()
        .map(|request| request.member_name.as_str())
        .collect();
    assert_eq!(
        requested,
        ["team_leader", "dev-1", "teammate"],
        "the leader allocates at build time, then the roster is spawned in sorted order"
    );
    assert_eq!(
        fixture.workspace.allocations(),
        ["team_leader", "dev-1", "teammate"],
        "every allocation is recorded once"
    );
    for request in &requests {
        assert_eq!(request.team_name, "alpha");
        assert_eq!(
            request.session_id,
            agent.session_id(),
            "every member runs under the team's own session"
        );
        assert_eq!(request.isolation, IsolationMode::None);
    }
    assert!(
        fixture.workspace.releases().is_empty(),
        "nothing is released while the team is live"
    );
}

#[tokio::test]
async fn a_direct_message_reaches_one_mailbox_with_one_bus_event() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;
    let recorder = subscribe_observer(
        agent.backend().messager(),
        &[TeamTopic::Message, TeamTopic::Team, TeamTopic::Task],
    );

    let record = agent.send_message(Some("dev-1"), "hello").await.unwrap();

    assert_eq!(record.team_name, "alpha");
    assert_eq!(
        record.from_member_name, "team_leader",
        "the handle is the sender"
    );
    assert_eq!(record.to_member_name.as_deref(), Some("dev-1"));
    assert!(!record.is_broadcast(), "a targeted message is direct");
    assert!(!record.is_read, "a fresh row is unread");

    let unread = agent.backend().unread("dev-1").await.unwrap();
    assert_eq!(unread.len(), 1, "one row in the target's mailbox");
    assert_eq!(unread[0].message_id, record.message_id);
    assert_eq!(unread[0].content, "hello");
    assert!(
        agent
            .backend()
            .unread("team_leader")
            .await
            .unwrap()
            .is_empty(),
        "the sender does not receive its own mail"
    );
    assert!(
        agent.backend().unread("teammate").await.unwrap().is_empty(),
        "a direct message reaches only its target"
    );

    assert_eq!(event_types(&recorder), ["message".to_owned()]);
    let events = recorded(&recorder);
    assert_eq!(events[0].sender_id, "team_leader", "the bus is stamped too");
    assert_eq!(events[0].payload["from_member_name"], json!("team_leader"));
    assert_eq!(events[0].payload["to_member_name"], json!("dev-1"));
    assert_eq!(events[0].payload["message_id"], json!(record.message_id));
}

#[tokio::test]
async fn a_broadcast_reaches_every_other_mailbox_with_one_bus_event() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;
    let recorder = subscribe_observer(agent.backend().messager(), &[TeamTopic::Message]);

    let record = agent.send_message(None, "all hands").await.unwrap();

    assert!(record.is_broadcast());
    assert_eq!(record.to_member_name, None);
    assert_eq!(record.from_member_name, "team_leader");
    for name in ["dev-1", "teammate"] {
        let unread = agent.backend().unread(name).await.unwrap();
        assert_eq!(unread.len(), 1, "{name} reads the broadcast");
        assert!(unread[0].is_broadcast());
        assert_eq!(
            unread[0].message_id, record.message_id,
            "one row, not one per member"
        );
    }
    assert!(
        agent
            .backend()
            .unread("team_leader")
            .await
            .unwrap()
            .is_empty(),
        "a member never reads its own broadcast"
    );

    assert_eq!(event_types(&recorder), ["broadcast".to_owned()]);
    let events = recorded(&recorder);
    assert_eq!(events[0].sender_id, "team_leader");
    assert_eq!(events[0].payload["from_member_name"], json!("team_leader"));
    assert_eq!(events[0].payload["message_id"], json!(record.message_id));
}

#[tokio::test]
async fn a_multicast_target_list_is_de_duplicated() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;
    let recorder = subscribe_observer(agent.backend().messager(), &[TeamTopic::Message]);

    let record = agent
        .backend()
        .send_message("team_leader", Some("dev-1,teammate"), "both")
        .await
        .unwrap();
    assert_eq!(
        record.to_member_name.as_deref(),
        Some("dev-1"),
        "the first row is returned"
    );
    assert_eq!(record.from_member_name, "team_leader");
    for name in ["dev-1", "teammate"] {
        let unread = agent.backend().unread(name).await.unwrap();
        assert_eq!(unread.len(), 1, "{name} receives the multicast");
        assert_eq!(unread[0].from_member_name, "team_leader");
    }
    assert_eq!(
        event_types(&recorder),
        ["message".to_owned(), "message".to_owned()],
        "one event per distinct target"
    );
    let events = recorded(&recorder);
    assert_eq!(events[0].payload["to_member_name"], json!("dev-1"));
    assert_eq!(events[0].sender_id, "team_leader");
    assert_eq!(events[1].payload["to_member_name"], json!("teammate"));
    assert_eq!(events[1].sender_id, "team_leader");

    let repeated = agent
        .backend()
        .send_message("dev-1", Some("dev-1,dev-1"), "twice?")
        .await
        .unwrap();
    assert_eq!(repeated.to_member_name.as_deref(), Some("dev-1"));
    assert_eq!(repeated.from_member_name, "dev-1");
    let unread = agent.backend().unread("dev-1").await.unwrap();
    assert_eq!(unread.len(), 2, "one row, not one per repeated target");
    assert_eq!(
        unread
            .iter()
            .filter(|message| message.content == "twice?")
            .count(),
        1,
        "the repeated target produced a single row"
    );
    assert_eq!(
        agent.backend().unread("teammate").await.unwrap().len(),
        1,
        "the repeated target does not leak into another mailbox"
    );
    assert_eq!(
        event_types(&recorder).len(),
        3,
        "one event per persisted row"
    );
    assert_eq!(recorded(&recorder)[2].sender_id, "dev-1");
}

#[tokio::test]
async fn mark_read_clears_only_the_named_direct_ids() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;
    let first = agent.send_message(Some("dev-1"), "one").await.unwrap();
    let second = agent.send_message(Some("dev-1"), "two").await.unwrap();

    agent
        .backend()
        .mark_read("dev-1", std::slice::from_ref(&first.message_id))
        .await
        .unwrap();

    let unread = agent.backend().unread("dev-1").await.unwrap();
    assert_eq!(unread.len(), 1, "only the named id was cleared");
    assert_eq!(unread[0].message_id, second.message_id);
    assert_eq!(unread[0].content, "two");

    // A broadcast id in the same batch is ignored: broadcast read state is a watermark.
    let broadcast = agent.send_message(None, "all").await.unwrap();
    agent
        .backend()
        .mark_read(
            "dev-1",
            &[broadcast.message_id.clone(), first.message_id.clone()],
        )
        .await
        .unwrap();
    let unread = agent.backend().unread("dev-1").await.unwrap();
    assert_eq!(
        unread.len(),
        2,
        "the broadcast plus the still-unread direct row"
    );
    assert!(unread.iter().any(MessageRecord::is_broadcast));
    assert_eq!(
        agent
            .backend()
            .store()
            .read_watermark("alpha", "dev-1")
            .await
            .unwrap(),
        None,
        "marking a broadcast never advances the watermark"
    );
}

#[tokio::test]
async fn only_the_leader_handle_creates_tasks_and_publishes_task_created() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;
    let recorder = subscribe_observer(agent.backend().messager(), &[TeamTopic::Task]);

    assert_eq!(
        agent.role(),
        TeamRole::Leader,
        "build returns the leader handle"
    );
    assert_eq!(
        agent
            .backend()
            .member("team_leader")
            .expect("the leader row")
            .role(),
        TeamRole::Leader
    );
    assert_eq!(
        agent
            .backend()
            .member("dev-1")
            .expect("the dev-1 row")
            .role(),
        TeamRole::Teammate
    );

    let record = agent
        .create_task(TaskGraphSpec {
            title: "ship it".to_owned(),
            content: "do the thing".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();

    assert_eq!(record.team_name, "alpha");
    let stored = agent
        .backend()
        .store()
        .task("alpha", &record.task_id)
        .await
        .unwrap()
        .expect("the task row is persisted");
    assert_eq!(stored.title, "ship it");
    assert_eq!(stored.content, "do the thing");
    assert_eq!(stored.status, TaskStatus::Pending);

    let types = event_types(&recorder);
    assert_eq!(
        types.first().map(String::as_str),
        Some("task_created"),
        "{types:?}"
    );
    assert!(
        recorded(&recorder)
            .iter()
            .all(|event| event.sender_id == "alpha"),
        "board events are stamped with the team, not a member"
    );

    // The role gate is the private `ensure_leader_role` inside `TeamAgent::create_task`, and a
    // teammate-mounted handle is not publicly constructible (`build` always returns the leader),
    // so the teammate rejection cannot be reached from this suite. What is reachable is the board
    // seam below it: `TaskBoard::create_task` carries no role gate of its own.
    let mutation = agent
        .backend()
        .tasks()
        .create_task(TaskGraphSpec {
            title: "board level".to_owned(),
            content: "no role gate here".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();
    assert_eq!(mutation.task.title, "board level");
    assert_eq!(mutation.task.status, TaskStatus::Pending);
    assert!(agent
        .backend()
        .store()
        .task("alpha", &mutation.task.task_id)
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn the_roster_accessors_agree_and_are_sorted() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), "session-a").await;

    let members = agent.members().await;
    let names: Vec<String> = members
        .iter()
        .map(|member| member.member_name().to_owned())
        .collect();
    assert_eq!(names, ["dev-1", "team_leader", "teammate"]);
    assert_eq!(
        agent.backend().member_names(),
        names,
        "the agent and the backend report the same roster"
    );
    assert!(
        names.windows(2).all(|pair| pair[0] < pair[1]),
        "the roster is sorted: {names:?}"
    );
    assert!(
        agent.backend().member("nobody").is_none(),
        "an off-roster name has no row"
    );
    assert_eq!(agent.backend().len(), 3);

    let leader = agent.member();
    assert_eq!(leader.member_name(), agent.leader_member_name());
    assert_eq!(leader.role(), TeamRole::Leader);
    assert_eq!(
        agent.runtime().session_id(),
        agent.session_id(),
        "the leader runtime runs under the team session"
    );

    timeout(BOUND, agent.spawn_member("dev-1"))
        .await
        .expect("the spawn did not finish")
        .expect("dev-1 did not spawn");
    let member = agent.backend().member("dev-1").expect("the dev-1 row");
    let runtime = member.runtime().expect("dev-1 owns a brain");
    assert_eq!(
        runtime.session_id(),
        agent.session_id(),
        "a spawned member runs under the team session"
    );
    assert_eq!(member.status(), MemberStatus::Ready);
    let stored = fixture
        .store
        .member("alpha", "dev-1")
        .await
        .unwrap()
        .expect("the roster row is persisted");
    assert_eq!(stored.member_name, member.member_name());
    assert_eq!(stored.role, member.role());
    assert_eq!(stored.status, member.status());
}
