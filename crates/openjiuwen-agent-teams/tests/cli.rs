//! The `team-member` CLI suite: command parsing, descriptor resolution and rendered output.
//!
//! This suite pins the observable contract of `openjiuwen_agent_teams::skill` from outside the
//! crate — the same surface an external `team-member` process sees:
//!
//! 1. **The two command trees.** Every operator subcommand and every member subcommand parses,
//!    required arguments are enforced, and the documented scope split holds: an operator-only word
//!    is a parse error in the member tree and vice versa, and `read_inbox` exists in neither (the
//!    inbox is the `inbox` word, not a teammate tool).
//! 2. **Connection flags.** `--descriptor-file`/`--descriptor-json` are `global`, so they parse
//!    before *and* after the subcommand (including after a nested `task …` subcommand);
//!    [`SkillCli::load_descriptor`] resolves `--descriptor-json` over `--descriptor-file` over
//!    `OPENJIUWEN_TEAM_JOIN`, blank flags count as absent, and a malformed descriptor is a
//!    validation error naming serde's reason.
//! 3. **Operator runs** against a real `StoreTeamClient` over `InMemoryTeamStore` + `TaskBoard`:
//!    the inbox line format and its consume-on-read cursor, the board views, the roster order, the
//!    acknowledgement lines, and the exit-code mapping (0 success, 1 operation failure, 2 a tree /
//!    usage mismatch).
//! 4. **Member runs**: the member mailbox, `view_task` rendering (including `null` for an unknown
//!    task id), `claim_task` for the descriptor's member, and `send_message` direct/broadcast.
//! 5. **The scope-independent `serve` stage.** `serve` belongs to neither scope tree (no help line,
//!    no parse), and it is dispatched before the join descriptor is loaded, so `serve --help`
//!    succeeds without one and a bad `serve` flag is a CLI-side error.
//!
//! Everything is offline and deterministic: rows are seeded with explicit timestamps, and the only
//! values not asserted byte-for-byte are the generated message/task ids (asserted by shape and by
//! reading the id back out of the store).
//!
//! The live-team resolver of `run_skill_cli` (against the process-global runtime registry) is
//! deliberately *not* exercised here: it is another node's wiring and would make the suite depend
//! on process-global state. `run_skill_cli` itself is only reached through its scope-independent
//! `serve` stage, which never resolves a team.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

use std::sync::Arc;

use openjiuwen_agent_teams::messager::InProcessMessager;
use openjiuwen_agent_teams::schema::{MemberStatus, TaskRecord, TaskStatus, TeamRole};
use openjiuwen_agent_teams::skill::cli::{parse_descriptor_args, CliCommand};
use openjiuwen_agent_teams::skill::{
    run_skill_cli, ClientScope, SkillCli, StoreTeamClient, TeamClient, TeamJoinDescriptor,
};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, MemberRecord, MessageRecord, TeamStore};
use openjiuwen_agent_teams::taskboard::TaskBoard;

/// The team every fixture joins.
const TEAM: &str = "alpha";

/// Program name every argument vector starts with.
const PROG: &str = "team-member";

// ---------------------------------------------------------------------------------------------
// Helpers: argument vectors, parsing and the joined-team fixture
// ---------------------------------------------------------------------------------------------

/// Builds a process-style argument vector: program name first, then `words`.
fn argv(words: &[&str]) -> Vec<String> {
    std::iter::once(PROG.to_string())
        .chain(words.iter().map(|word| (*word).to_string()))
        .collect()
}

/// Parses `words` in `scope`, panicking with clap's message when they do not parse.
fn parse(words: &[&str], scope: ClientScope) -> SkillCli {
    SkillCli::try_parse_from(argv(words), scope)
        .unwrap_or_else(|err| panic!("{words:?} should parse in the {scope:?} tree: {err}"))
}

/// Parses `words` in `scope` and returns the rendered parse error.
fn parse_err(words: &[&str], scope: ClientScope) -> String {
    match SkillCli::try_parse_from(argv(words), scope) {
        Ok(_) => panic!("{words:?} must not parse in the {scope:?} tree"),
        Err(err) => err.to_string(),
    }
}

/// Returns the `Debug` rendering of a parsed invocation.
///
/// The parsed command tree (`ScopeCommand`, `MemberCommand`, `OperatorCommand`, `TaskCommand`) and
/// the `SkillCli` fields are `pub(crate)`, so from an integration test the `Debug` impl is the
/// public route to a parsed variant and its field values. Behavioural assertions in the run tests
/// below pin the same variants end to end.
fn debug_of(words: &[&str], scope: ClientScope) -> String {
    format!("{:?}", parse(words, scope))
}

/// One joined team: the store, the board and the client the CLI drives.
struct Team {
    /// Persistence seam the assertions read back.
    store: Arc<dyn TeamStore>,
    /// Task board the operator scope drives.
    board: Arc<TaskBoard>,
    /// The client under test, joined as `member_name`.
    client: Arc<dyn TeamClient>,
}

impl Team {
    /// Builds a team joined by `member_name` under `scope`.
    fn new(scope: ClientScope, member_name: &str) -> Self {
        let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
        let board = Arc::new(TaskBoard::with_defaults(Arc::clone(&store), TEAM));
        let descriptor = TeamJoinDescriptor {
            session_id: "s1".to_string(),
            team_name: TEAM.to_string(),
            member_name: member_name.to_string(),
            scope,
            ..TeamJoinDescriptor::default()
        };
        let client: Arc<dyn TeamClient> = Arc::new(StoreTeamClient::new(
            descriptor,
            Arc::clone(&store),
            Arc::clone(&board),
            Arc::new(InProcessMessager::new()),
        ));
        Self {
            store,
            board,
            client,
        }
    }

    /// Builds an operator-scope team joined as `user`.
    fn operator() -> Self {
        Self::new(ClientScope::Operator, "user")
    }

    /// Builds a member-scope team joined as `dev-1`.
    fn member() -> Self {
        Self::new(ClientScope::Member, "dev-1")
    }

    /// Seeds one roster row with the default lifecycle status.
    async fn seed_member(&self, name: &str, role: TeamRole) {
        self.store
            .insert_member(MemberRecord::new(TEAM, name, role))
            .await
            .unwrap();
    }

    /// Seeds one roster row with an explicit lifecycle status.
    async fn seed_member_with(&self, name: &str, role: TeamRole, status: MemberStatus) {
        let mut record = MemberRecord::new(TEAM, name, role);
        record.status = status;
        self.store.insert_member(record).await.unwrap();
    }

    /// Seeds one task row with an explicit write stamp, so list ordering is deterministic.
    async fn seed_task(&self, task_id: &str, title: &str, updated_at: u64) {
        self.store
            .insert_task(TaskRecord {
                team_name: TEAM.to_string(),
                task_id: task_id.to_string(),
                title: title.to_string(),
                content: "body".to_string(),
                status: TaskStatus::Pending,
                created_by: Some("team_leader".to_string()),
                updated_at,
                ..TaskRecord::default()
            })
            .await
            .unwrap();
    }

    /// Seeds one unread direct message row.
    async fn seed_direct(&self, message_id: &str, from: &str, to: &str, content: &str, ts: u64) {
        self.store
            .insert_message(MessageRecord::direct(
                TEAM, message_id, from, to, content, ts,
            ))
            .await
            .unwrap();
    }

    /// Seeds one unread broadcast row.
    async fn seed_broadcast(&self, message_id: &str, from: &str, content: &str, ts: u64) {
        self.store
            .insert_message(MessageRecord::broadcast(
                TEAM, message_id, from, content, ts,
            ))
            .await
            .unwrap();
    }

    /// Runs `words` through [`SkillCli::run_with`], returning `(exit code, stdout, stderr)`.
    async fn run_with(&self, words: &[&str]) -> (i32, String, String) {
        let cli = parse(words, self.client.scope());
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = cli
            .run_with(Arc::clone(&self.client), &mut out, &mut err)
            .await;
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    /// Runs `words` through the single-writer [`SkillCli::run`], returning `(exit code, output)`.
    async fn run(&self, words: &[&str]) -> (i32, String) {
        let cli = parse(words, self.client.scope());
        let mut out: Vec<u8> = Vec::new();
        let code = cli.run(Arc::clone(&self.client), &mut out).await;
        (code, String::from_utf8(out).unwrap())
    }

    /// Returns the task row `task_id`, panicking when it is absent.
    async fn task(&self, task_id: &str) -> TaskRecord {
        self.store
            .task(TEAM, task_id)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("task {task_id} should be persisted"))
    }

    /// Returns the unread rows of `member`.
    async fn unread(&self, member: &str) -> Vec<MessageRecord> {
        self.store.unread_messages(TEAM, member).await.unwrap()
    }
}

/// Splits an acknowledgement line of the form `<prefix><id> -> <target>\n`.
fn direct_ack<'a>(out: &'a str, prefix: &str, target: &str) -> &'a str {
    let line = out
        .strip_suffix('\n')
        .unwrap_or_else(|| panic!("expected a trailing newline: {out:?}"));
    let rest = line
        .strip_prefix(prefix)
        .unwrap_or_else(|| panic!("expected the {prefix:?} prefix: {line:?}"));
    let (id, rendered_target) = rest
        .rsplit_once(" -> ")
        .unwrap_or_else(|| panic!("expected a ' -> ' separator: {rest:?}"));
    assert_eq!(
        rendered_target, target,
        "unexpected acknowledgement: {line:?}"
    );
    assert!(
        uuid::Uuid::parse_str(id).is_ok(),
        "the acknowledgement should carry a UUID message id: {id:?}"
    );
    id
}

/// Splits a generated task id out of the `created <task_id>` acknowledgement line.
fn created_id(out: &str) -> &str {
    out.strip_suffix('\n')
        .and_then(|line| line.strip_prefix("created "))
        .unwrap_or_else(|| panic!("expected a 'created <id>' line: {out:?}"))
}

/// Returns `true` when `text` is 32 lowercase hex characters.
fn is_minted_task_id(text: &str) -> bool {
    text.len() == 32
        && text
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
}

// ---------------------------------------------------------------------------------------------
// Operator tree: parsing
// ---------------------------------------------------------------------------------------------

#[test]
fn operator_subcommands_parse() {
    // Every operator word parses in the operator tree.
    for words in [
        vec!["inbox"],
        vec!["inbox", "--watch"],
        vec!["send", "dev-1", "hi"],
        vec!["broadcast", "all hands"],
        vec!["create_task", "write docs", "body"],
        vec!["create_task", "write docs", "body", "--task-id", "t-9"],
        vec!["task", "list"],
        vec!["task", "list", "--status", "pending"],
        vec!["task", "claimable"],
        vec!["task", "get", "t-1"],
        vec!["claim", "t-1"],
        vec!["complete", "t-1"],
        vec!["update", "t-1"],
        vec!["update", "t-1", "--title", "new"],
        vec!["update", "t-1", "--content", "body"],
        vec!["members"],
    ] {
        let _ = parse(&words, ClientScope::Operator);
    }
}

#[test]
fn operator_parsed_variants_and_fields() {
    // `--watch` is a flag on `inbox`, off by default.
    assert!(debug_of(&["inbox"], ClientScope::Operator).contains("watch: false"));
    assert!(debug_of(&["inbox", "--watch"], ClientScope::Operator).contains("watch: true"));

    // `send <TO> <CONTENT>` and `broadcast <CONTENT>` take positionals.
    let send = debug_of(&["send", "dev-1", "hi"], ClientScope::Operator);
    assert!(send.contains("Send"), "unexpected: {send}");
    assert!(send.contains("to: \"dev-1\""), "unexpected: {send}");
    assert!(send.contains("content: \"hi\""), "unexpected: {send}");
    assert!(debug_of(&["broadcast", "all hands"], ClientScope::Operator).contains("Broadcast"));

    // `create_task` carries the optional pinned id.
    let created = debug_of(
        &["create_task", "t", "c", "--task-id", "t-9"],
        ClientScope::Operator,
    );
    assert!(created.contains("CreateTask"), "unexpected: {created}");
    assert!(
        created.contains("task_id: Some(\"t-9\")"),
        "unexpected: {created}"
    );

    // `task` is a nested tree: list (with a status filter), claimable and get.
    assert!(debug_of(&["task", "list"], ClientScope::Operator).contains("List { status: None }"));
    assert!(debug_of(
        &["task", "list", "--status", "pending"],
        ClientScope::Operator
    )
    .contains("status: Some(\"pending\")"));
    assert!(debug_of(&["task", "claimable"], ClientScope::Operator).contains("Claimable"));
    let get = debug_of(&["task", "get", "t-1"], ClientScope::Operator);
    assert!(get.contains("Get"), "unexpected: {get}");
    assert!(get.contains("task_id: \"t-1\""), "unexpected: {get}");

    // `update --title/--content` are optional options.
    let update = debug_of(
        &["update", "t-1", "--title", "new", "--content", "body"],
        ClientScope::Operator,
    );
    assert!(update.contains("Update"), "unexpected: {update}");
    assert!(
        update.contains("title: Some(\"new\")"),
        "unexpected: {update}"
    );
    assert!(
        update.contains("content: Some(\"body\")"),
        "unexpected: {update}"
    );
    assert!(debug_of(&["update", "t-1"], ClientScope::Operator).contains("title: None"));

    assert!(debug_of(&["members"], ClientScope::Operator).contains("Members"));
    assert!(debug_of(&["claim", "t-1"], ClientScope::Operator).contains("Claim"));
    assert!(debug_of(&["complete", "t-1"], ClientScope::Operator).contains("Complete"));
}

#[test]
fn operator_required_arguments_are_enforced() {
    let cases: [(&[&str], &str); 7] = [
        (&["send"], "<TO>"),
        (&["send", "dev-1"], "<CONTENT>"),
        (&["broadcast"], "<CONTENT>"),
        (&["create_task"], "<TITLE>"),
        (&["create_task", "title"], "<CONTENT>"),
        (&["task", "get"], "<TASK_ID>"),
        (&["claim"], "<TASK_ID>"),
    ];
    for (words, missing) in cases {
        let message = parse_err(words, ClientScope::Operator);
        assert!(
            message.starts_with("validation: "),
            "{words:?} should be a validation error: {message}"
        );
        assert!(
            message.contains(missing),
            "{words:?} should name the missing {missing}: {message}"
        );
    }
    // `complete` and `update` need a task id too.
    assert!(parse_err(&["complete"], ClientScope::Operator).contains("<TASK_ID>"));
    assert!(parse_err(&["update"], ClientScope::Operator).contains("<TASK_ID>"));
}

#[test]
fn operator_forms_are_positional_not_options() {
    // The declarations are positionals: passing the option spelling is a parse error, so a caller
    // cannot rely on a `--to`/`--title` form that clap never declared.
    for words in [
        vec!["send", "--to", "dev-1", "--content", "hi"],
        vec!["create_task", "--title", "t", "--content", "c"],
        vec!["task", "get", "--task-id", "t-1"],
        vec!["claim", "--task-id", "t-1"],
    ] {
        let message = parse_err(&words, ClientScope::Operator);
        assert!(
            message.starts_with("validation: "),
            "{words:?} should be a validation error: {message}"
        );
    }
    // An unknown flag is refused even when the positionals are present.
    assert!(parse_err(&["members", "--nope"], ClientScope::Operator).contains("--nope"));
}

#[test]
fn operator_tree_rejects_member_words_and_unknown_commands() {
    // The member tool set is not reachable as operator words.
    for words in [
        vec!["view_task"],
        vec!["claim_task", "t-1", "claimed"],
        vec!["send_message", "dev-1", "hi"],
    ] {
        let message = parse_err(&words, ClientScope::Operator);
        assert!(
            message.starts_with("validation: "),
            "{words:?} should be refused by the operator tree: {message}"
        );
    }
    // `read_inbox` is not a command in either tree.
    assert!(parse_err(&["read_inbox"], ClientScope::Operator).contains("read_inbox"));
    // An unknown word names the offending token.
    assert!(parse_err(&["explode"], ClientScope::Operator).contains("explode"));
    // No subcommand at all is a usage error.
    assert!(parse_err(&[], ClientScope::Operator).starts_with("validation: "));
}

// ---------------------------------------------------------------------------------------------
// Member tree: parsing
// ---------------------------------------------------------------------------------------------

#[test]
fn member_subcommands_parse() {
    for words in [
        vec!["inbox"],
        vec!["inbox", "--watch"],
        vec!["view_task"],
        vec!["view_task", "--action", "list"],
        vec!["view_task", "--action", "claimable"],
        vec!["view_task", "--action", "get", "--task-id", "t-1"],
        vec!["view_task", "--status", "pending"],
        vec!["claim_task", "t-1", "claimed"],
        vec!["claim_task", "t-1", "completed"],
        vec!["send_message", "team_leader", "done"],
        vec!["send_message", "*", "all hands"],
    ] {
        let _ = parse(&words, ClientScope::Member);
    }
}

#[test]
fn member_parsed_variants_and_fields() {
    assert!(debug_of(&["inbox"], ClientScope::Member).contains("watch: false"));
    assert!(debug_of(&["inbox", "--watch"], ClientScope::Member).contains("watch: true"));

    let list = debug_of(&["view_task"], ClientScope::Member);
    assert!(list.contains("ViewTask"), "unexpected: {list}");
    assert!(list.contains("action: List"), "unexpected: {list}");
    assert!(list.contains("task_id: None"), "unexpected: {list}");
    assert!(list.contains("status: None"), "unexpected: {list}");

    let get = debug_of(
        &[
            "view_task",
            "--action",
            "get",
            "--task-id",
            "t-1",
            "--status",
            "pending",
        ],
        ClientScope::Member,
    );
    assert!(get.contains("action: Get"), "unexpected: {get}");
    assert!(get.contains("task_id: Some(\"t-1\")"), "unexpected: {get}");
    assert!(
        get.contains("status: Some(\"pending\")"),
        "unexpected: {get}"
    );

    assert!(
        debug_of(&["view_task", "--action", "claimable"], ClientScope::Member)
            .contains("action: Claimable")
    );

    let claimed = debug_of(&["claim_task", "t-1", "claimed"], ClientScope::Member);
    assert!(claimed.contains("ClaimTask"), "unexpected: {claimed}");
    assert!(
        claimed.contains("task_id: \"t-1\""),
        "unexpected: {claimed}"
    );
    assert!(claimed.contains("status: Claimed"), "unexpected: {claimed}");
    assert!(
        debug_of(&["claim_task", "t-1", "completed"], ClientScope::Member)
            .contains("status: Completed")
    );

    let send = debug_of(
        &["send_message", "team_leader", "done"],
        ClientScope::Member,
    );
    assert!(send.contains("SendMessage"), "unexpected: {send}");
    assert!(send.contains("to: \"team_leader\""), "unexpected: {send}");
    assert!(send.contains("content: \"done\""), "unexpected: {send}");
}

#[test]
fn member_required_arguments_are_enforced() {
    let cases: [(&[&str], &str); 5] = [
        (&["claim_task"], "<TASK_ID>"),
        (&["claim_task", "t-1"], "<STATUS>"),
        (&["send_message"], "<TO>"),
        (&["send_message", "team_leader"], "<CONTENT>"),
        (&["view_task", "--action", "bogus"], "bogus"),
    ];
    for (words, named) in cases {
        let message = parse_err(words, ClientScope::Member);
        assert!(
            message.starts_with("validation: "),
            "{words:?} should be a validation error: {message}"
        );
        assert!(
            message.contains(named),
            "{words:?} should name {named}: {message}"
        );
    }
    // The claim status is a closed value set.
    assert!(parse_err(&["claim_task", "t-1", "started"], ClientScope::Member).contains("started"));
}

#[test]
fn member_tree_rejects_operator_words_and_unknown_commands() {
    // The member CLI is `inbox | view_task | claim_task | send_message`: the wider teammate tool
    // catalogue (`create_task`, `update_task`, `list_tasks`, `list_members`, …) is not reachable as
    // a member command word, and neither is the operator surface.
    for words in [
        vec!["members"],
        vec!["send", "dev-1", "hi"],
        vec!["broadcast", "all hands"],
        vec!["claim", "t-1"],
        vec!["complete", "t-1"],
        vec!["update", "t-1"],
        vec!["create_task", "t", "c"],
        vec!["task", "list"],
        vec!["list_tasks"],
        vec!["update_task", "t-1"],
        vec!["list_members"],
        vec!["complete_task", "t-1"],
    ] {
        let message = parse_err(&words, ClientScope::Member);
        assert!(
            message.starts_with("validation: "),
            "{words:?} should be refused by the member tree: {message}"
        );
    }
    // `read_inbox` is not a member command: the mailbox is read with `inbox`.
    assert!(parse_err(&["read_inbox"], ClientScope::Member).contains("read_inbox"));
    assert!(parse_err(&["explode"], ClientScope::Member).contains("explode"));
    assert!(parse_err(&[], ClientScope::Member).starts_with("validation: "));
}

// ---------------------------------------------------------------------------------------------
// Scope-independent `serve` stage
// ---------------------------------------------------------------------------------------------

#[test]
fn serve_is_not_a_member_or_operator_command() {
    // `serve` is the child end of the member protocol, not a word in either control tree.
    for scope in [ClientScope::Member, ClientScope::Operator] {
        let message = parse_err(&["serve"], scope);
        assert!(
            message.starts_with("validation: "),
            "{scope:?} should refuse `serve`: {message}"
        );
        assert!(
            message.contains("serve"),
            "{scope:?} should name the refused word: {message}"
        );

        // Neither tree's help advertises it: help renders that tree's own command table.
        let help = parse_err(&["--help"], scope);
        assert!(
            help.contains("Usage: team-member"),
            "{scope:?} should render its own usage: {help}"
        );
        assert!(
            !help.contains("serve"),
            "{scope:?} must not list `serve`: {help}"
        );

        // A command parsed out of a tree is that tree's variant — `SkillCli::try_parse_from` never
        // yields `CliCommand::Serve`, which only the scope-independent stage produces.
        let command = parse(&["inbox"], scope).command().clone();
        assert!(
            matches!(command, CliCommand::Member(_) | CliCommand::Operator(_)),
            "{scope:?} should not yield a serve command: {command:?}"
        );
    }
}

#[test]
fn serve_help_is_answered_before_the_descriptor_stage() {
    // `serve` is dispatched before the join descriptor is loaded, so a descriptor that cannot
    // possibly load does not turn its help into a failure — and no `OPENJIUWEN_TEAM_JOIN` is needed.
    let code = run_skill_cli(argv(&[
        "--descriptor-file",
        "/nonexistent/team-join.json",
        "serve",
        "--help",
    ]));
    assert_eq!(code, 0, "help must not depend on a join descriptor");

    // The same connection flags without `serve` do fail in the descriptor stage (exit `2`), so the
    // `0` above proves the serve stage ran first rather than the descriptor scan succeeding.
    assert_eq!(
        run_skill_cli(argv(&[
            "--descriptor-file",
            "/nonexistent/team-join.json",
            "inbox",
        ])),
        2,
        "an unreadable descriptor is still a CLI-side error"
    );
}

#[test]
fn a_bad_serve_flag_is_a_cli_side_error() {
    // A flag clap never declared on `serve` is a usage mismatch: exit `2`, never the serve loop.
    assert_eq!(run_skill_cli(argv(&["serve", "--nope"])), 2);
    // The serve stage runs before the descriptor stage even when both would fail.
    assert_eq!(
        run_skill_cli(argv(&[
            "--descriptor-file",
            "/nonexistent/team-join.json",
            "serve",
            "--nope",
        ])),
        2
    );
}

// ---------------------------------------------------------------------------------------------
// Connection flags and descriptor resolution
// ---------------------------------------------------------------------------------------------

#[test]
fn descriptor_json_parses_before_and_after_the_subcommand() {
    let json =
        r#"{"session_id":"s1","team_name":"alpha","member_name":"dev-1","scope":"operator"}"#;
    let expected = TeamJoinDescriptor::from_json(json).unwrap();

    // Before the subcommand, in the operator tree.
    let before = parse(
        &["--descriptor-json", json, "members"],
        ClientScope::Operator,
    );
    assert_eq!(before.load_descriptor().unwrap(), expected);

    // After the subcommand, in the operator tree.
    let after = parse(
        &["members", "--descriptor-json", json],
        ClientScope::Operator,
    );
    assert_eq!(after.load_descriptor().unwrap(), expected);

    // After a nested subcommand too: the flags are `global`, not only root-level.
    let nested = parse(
        &["task", "list", "--descriptor-json", json],
        ClientScope::Operator,
    );
    assert_eq!(nested.load_descriptor().unwrap(), expected);

    // And in the member tree, with the file form after the subcommand.
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        file.path(),
        r#"{"team_name":"alpha","member_name":"dev-1","scope":"member"}"#,
    )
    .unwrap();
    let path = file.path().display().to_string();
    let member = parse(&["inbox", "--descriptor-file", &path], ClientScope::Member);
    let loaded = member.load_descriptor().unwrap();
    assert_eq!(loaded.team_name, "alpha");
    assert_eq!(loaded.member_name, "dev-1");
    assert_eq!(loaded.scope, ClientScope::Member);
}

#[test]
fn descriptor_precedence_is_json_then_file_then_env() {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), r#"{"team_name":"beta"}"#).unwrap();
    let path = file.path().display().to_string();

    // The inline JSON wins over the file.
    let both = parse(
        &[
            "--descriptor-json",
            r#"{"team_name":"alpha"}"#,
            "--descriptor-file",
            &path,
            "members",
        ],
        ClientScope::Operator,
    );
    assert_eq!(both.load_descriptor().unwrap().team_name, "alpha");

    // A blank inline JSON counts as absent (Python `if descriptor_json:`), so the file is read.
    let blank = parse(
        &[
            "--descriptor-json",
            "   ",
            "--descriptor-file",
            &path,
            "members",
        ],
        ClientScope::Operator,
    );
    assert_eq!(blank.load_descriptor().unwrap().team_name, "beta");

    // The environment form is the documented last resort; the injectable lookup pins it without
    // mutating the process environment.
    let from_env = TeamJoinDescriptor::from_env_map(|key| {
        (key == openjiuwen_agent_teams::skill::descriptor::TEAM_JOIN_ENV)
            .then(|| r#"{"team_name":"gamma","scope":"member"}"#.to_string())
    })
    .unwrap();
    assert_eq!(from_env.team_name, "gamma");
    assert_eq!(from_env.scope, ClientScope::Member);

    let missing = TeamJoinDescriptor::from_env_map(|_| None).unwrap_err();
    assert_eq!(
        missing.to_string(),
        "config: environment variable OPENJIUWEN_TEAM_JOIN is not set; cannot join team"
    );
    let blank_env = TeamJoinDescriptor::from_env_map(|_| Some(" ".to_string())).unwrap_err();
    assert!(blank_env.to_string().contains("is not set"));
}

#[test]
fn parse_descriptor_args_scans_both_flag_forms() {
    // Space-separated and `=`-joined forms, before or after the command word.
    let scanned =
        parse_descriptor_args(&argv(&["--descriptor-file", "join.json", "members"])).unwrap();
    assert_eq!(scanned.descriptor_file.as_deref(), Some("join.json"));
    assert_eq!(scanned.descriptor_json, None);

    let scanned = parse_descriptor_args(&argv(&[
        "members",
        "--descriptor-json={\"team_name\":\"alpha\"}",
    ]))
    .unwrap();
    assert_eq!(
        scanned.descriptor_json.as_deref(),
        Some(r#"{"team_name":"alpha"}"#)
    );
    assert_eq!(scanned.descriptor_file, None);

    // Unknown tokens belong to the scope parser and are skipped here.
    let scanned =
        parse_descriptor_args(&argv(&["send", "dev-1", "hi", "--descriptor-file=a.json"])).unwrap();
    assert_eq!(scanned.descriptor_file.as_deref(), Some("a.json"));

    // A repeated flag keeps its last value.
    let scanned = parse_descriptor_args(&argv(&[
        "--descriptor-json",
        "{\"team_name\":\"alpha\"}",
        "members",
        "--descriptor-json",
        "{\"team_name\":\"beta\"}",
    ]))
    .unwrap();
    assert_eq!(
        scanned.descriptor_json.as_deref(),
        Some(r#"{"team_name":"beta"}"#)
    );

    // The associated function is the same scanner.
    let scanned = SkillCli::parse_descriptor_args(&argv(&["members"])).unwrap();
    assert_eq!(
        scanned,
        openjiuwen_agent_teams::skill::DescriptorArgs::default()
    );

    // A flag without a value, or followed by another flag, is a validation error.
    let missing = parse_descriptor_args(&argv(&["members", "--descriptor-file"])).unwrap_err();
    assert_eq!(
        missing.to_string(),
        "validation: --descriptor-file requires a value"
    );
    let consumed = parse_descriptor_args(&argv(&[
        "members",
        "--descriptor-json",
        "--descriptor-file",
        "join.json",
    ]))
    .unwrap_err();
    assert_eq!(
        consumed.to_string(),
        "validation: --descriptor-json requires a value"
    );
}

#[test]
fn a_malformed_descriptor_is_a_validation_error() {
    let cli = parse(
        &["--descriptor-json", "{not json", "members"],
        ClientScope::Operator,
    );
    let error = cli.load_descriptor().unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("validation: malformed team join descriptor: "),
        "unexpected message: {error}"
    );
    // The descriptor body is only parsed by `load_descriptor`: the tree parser never inspects it,
    // so the same malformed JSON still reaches the scope parser and the unknown word is what fails.
    let message = parse_err(
        &["--descriptor-json", "{not json", "explode"],
        ClientScope::Operator,
    );
    assert!(message.contains("explode"), "unexpected message: {message}");
    assert!(
        !message.contains("malformed team join descriptor"),
        "the tree parser must not load the descriptor: {message}"
    );
}

// ---------------------------------------------------------------------------------------------
// Operator runs against the store-backed client
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn inbox_renders_the_seeded_rows_and_then_reports_it_empty() {
    let team = Team::operator();
    team.seed_broadcast("m-1", "team_leader", "all hands", 1000)
        .await;
    team.seed_direct("m-2", "team_leader", "user", "status?", 2000)
        .await;

    assert_eq!(
        team.run_with(&["inbox"]).await,
        (
            0,
            "[1000] team_leader -> * : all hands\n[2000] team_leader -> user : status?\n"
                .to_string(),
            String::new()
        )
    );
    // Reading consumed the rows: the direct row is flagged read and the broadcast watermark moved.
    assert_eq!(team.unread("user").await, Vec::new());
    assert_eq!(
        team.store.read_watermark(TEAM, "user").await.unwrap(),
        Some(1000)
    );
    // A second read finds nothing left.
    assert_eq!(
        team.run_with(&["inbox"]).await,
        (0, "(inbox empty)\n".to_string(), String::new())
    );
    // `--watch` performs the same single read.
    assert_eq!(
        team.run_with(&["inbox", "--watch"]).await,
        (0, "(inbox empty)\n".to_string(), String::new())
    );
}

#[tokio::test]
async fn task_views_render_the_board_rows() {
    let team = Team::operator();
    team.seed_member("dev-1", TeamRole::Teammate).await;
    team.seed_task("t-1", "first task", 500).await;
    team.seed_task("t-2", "second task", 900).await;

    // Newest write first, ties broken by task id.
    assert_eq!(
        team.run_with(&["task", "list"]).await,
        (
            0,
            "[t-2] pending second task (-)\n[t-1] pending first task (-)\n".to_string(),
            String::new()
        )
    );
    assert_eq!(
        team.run_with(&["task", "list", "--status", "pending"])
            .await,
        (
            0,
            "[t-2] pending second task (-)\n[t-1] pending first task (-)\n".to_string(),
            String::new()
        )
    );
    // A filter with no match prints nothing at all and still succeeds.
    assert_eq!(
        team.run_with(&["task", "list", "--status", "completed"])
            .await,
        (0, String::new(), String::new())
    );

    assert_eq!(
        team.run_with(&["task", "claimable"]).await,
        (
            0,
            "[t-2] second task\n[t-1] first task\n".to_string(),
            String::new()
        )
    );

    let claimable_only = Team::operator();
    claimable_only
        .seed_member("dev-1", TeamRole::Teammate)
        .await;
    claimable_only.seed_task("t-1", "first task", 500).await;
    claimable_only.board.claim("t-1", "dev-1").await.unwrap();
    assert_eq!(
        claimable_only.run_with(&["task", "claimable"]).await,
        (0, String::new(), String::new())
    );
}

#[tokio::test]
async fn task_get_renders_the_seven_line_block() {
    let team = Team::operator();
    team.seed_member("dev-1", TeamRole::Teammate).await;
    team.seed_task("t-1", "first task", 500).await;

    assert_eq!(
        team.run_with(&["task", "get", "t-1"]).await,
        (
            0,
            "id:        t-1\n\
             title:     first task\n\
             status:    pending\n\
             assignee:  -\n\
             blocked_by: -\n\
             blocks:     -\n\
             content:\n\
             body\n"
                .to_string(),
            String::new()
        )
    );

    // An unknown task is an operation failure, not a usage error.
    assert_eq!(
        team.run_with(&["task", "get", "nope"]).await,
        (1, String::new(), "Task 'nope' not found\n".to_string())
    );
    // An unknown status filter is refused before the board is queried.
    let (code, out, err) = team
        .run_with(&["task", "list", "--status", "shipped"])
        .await;
    assert_eq!((code, out), (1, String::new()));
    assert!(
        err.contains("unknown task status 'shipped'"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn members_renders_the_roster_sorted_by_name() {
    let team = Team::operator();
    team.seed_member("zeta", TeamRole::Teammate).await;
    team.seed_member("team_leader", TeamRole::Leader).await;
    team.seed_member_with("dev-1", TeamRole::Teammate, MemberStatus::Busy)
        .await;

    assert_eq!(
        team.run_with(&["members"]).await,
        (
            0,
            "dev-1 (teammate) [busy]\n\
             team_leader (leader) [unstarted]\n\
             zeta (teammate) [unstarted]\n"
                .to_string(),
            String::new()
        )
    );

    // An empty roster writes nothing and still succeeds.
    let empty = Team::operator();
    assert_eq!(
        empty.run_with(&["members"]).await,
        (0, String::new(), String::new())
    );
}

#[tokio::test]
async fn send_persists_one_direct_message_and_acknowledges_it() {
    let team = Team::operator();
    let (code, out, err) = team.run_with(&["send", "dev-1", "start now"]).await;
    assert_eq!((code, err.as_str()), (0, ""), "unexpected: {err}");
    let message_id = direct_ack(&out, "sent ", "dev-1").to_string();

    let rows = team.unread("dev-1").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].message_id, message_id);
    assert_eq!(rows[0].from_member_name, "user");
    assert_eq!(rows[0].to_member_name.as_deref(), Some("dev-1"));
    assert_eq!(rows[0].content, "start now");
    assert!(!rows[0].broadcast);
}

#[tokio::test]
async fn broadcast_persists_one_broadcast_row_and_acknowledges_it() {
    let team = Team::operator();
    let (code, out, err) = team.run_with(&["broadcast", "all hands"]).await;
    assert_eq!((code, err.as_str()), (0, ""), "unexpected: {err}");
    let line = out.strip_suffix('\n').expect("a trailing newline");
    let message_id = line
        .strip_prefix("broadcast ")
        .unwrap_or_else(|| panic!("expected a 'broadcast <id>' line: {line:?}"));
    assert!(uuid::Uuid::parse_str(message_id).is_ok());

    let rows = team.unread("dev-1").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].message_id, message_id);
    assert_eq!(rows[0].from_member_name, "user");
    assert!(rows[0].broadcast);
    assert_eq!(rows[0].content, "all hands");
}

#[tokio::test]
async fn refused_sends_exit_one_with_the_reason() {
    let team = Team::operator();
    let too_long = "x".repeat(20_001);
    let (code, out, err) = team.run_with(&["send", "dev-1", &too_long]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(
        err.starts_with("content too long: 20001 characters exceeds the limit of 20000"),
        "unexpected: {err}"
    );
    assert!(
        team.unread("dev-1").await.is_empty(),
        "nothing is persisted"
    );

    // A blank target is refused by the client before anything is written.
    let (code, out, err) = team.run_with(&["send", "   ", "hi"]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert_eq!(err, "validation: target member name is required\n");
}

#[tokio::test]
async fn create_task_acknowledges_and_persists_the_row() {
    let team = Team::operator();
    let (code, out, err) = team
        .run_with(&["create_task", "write docs", "acceptance"])
        .await;
    assert_eq!((code, err.as_str()), (0, ""), "unexpected: {err}");
    let task_id = created_id(&out).to_string();
    assert!(
        task_id.starts_with("task_") && is_minted_task_id(&task_id["task_".len()..]),
        "unexpected minted id: {task_id:?}"
    );
    let row = team.task(&task_id).await;
    assert_eq!(row.title, "write docs");
    assert_eq!(row.content, "acceptance");
    assert_eq!(row.status, TaskStatus::Pending);
    assert_eq!(row.created_by.as_deref(), Some("user"));

    // A pinned id is honoured verbatim.
    assert_eq!(
        team.run_with(&["create_task", "second", "body", "--task-id", "t-9"])
            .await,
        (0, "created t-9\n".to_string(), String::new())
    );
    assert_eq!(team.task("t-9").await.title, "second");

    // A duplicate pinned id is a board refusal, not a torn write.
    let (code, out, err) = team
        .run_with(&["create_task", "again", "body", "--task-id", "t-9"])
        .await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(err.contains("t-9"), "unexpected: {err}");
}

#[tokio::test]
async fn claim_and_complete_walk_a_seeded_task() {
    let team = Team::operator();
    team.seed_member("user", TeamRole::Teammate).await;
    team.seed_task("t-1", "first task", 500).await;

    assert_eq!(
        team.run_with(&["claim", "t-1"]).await,
        (0, "claimed t-1\n".to_string(), String::new())
    );
    let claimed = team.task("t-1").await;
    assert_eq!(claimed.status, TaskStatus::InProgress);
    assert_eq!(claimed.assignee.as_deref(), Some("user"));

    assert_eq!(
        team.run_with(&["complete", "t-1"]).await,
        (0, "completed t-1\n".to_string(), String::new())
    );
    assert_eq!(team.task("t-1").await.status, TaskStatus::Completed);

    // Claiming a completed task is an operation failure carrying the board's reason.
    let (code, out, err) = team.run_with(&["claim", "t-1"]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(
        err.contains("cannot be claimed from status 'completed'"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn operations_on_an_unknown_task_exit_one() {
    let team = Team::operator();

    let (code, out, err) = team.run_with(&["claim", "nope"]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert_eq!(err, "not_found: task 'nope' not found\n");

    let (code, out, err) = team.run_with(&["complete", "nope"]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert_eq!(err, "not_found: task 'nope' not found\n");

    let (code, out, err) = team.run_with(&["update", "nope", "--title", "t"]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert_eq!(err, "task 'nope' not found\n");
}

#[tokio::test]
async fn update_edits_a_pending_task_and_refuses_a_started_one() {
    let team = Team::operator();
    team.seed_task("t-1", "first task", 500).await;

    assert_eq!(
        team.run_with(&["update", "t-1", "--title", "better"]).await,
        (0, "updated t-1\n".to_string(), String::new())
    );
    let row = team.task("t-1").await;
    assert_eq!(row.title, "better");
    assert_eq!(row.content, "body");

    // Once the task has started, the operator edit path refuses it.
    assert_eq!(
        team.run_with(&["claim", "t-1"]).await,
        (0, "claimed t-1\n".to_string(), String::new())
    );
    let (code, out, err) = team
        .run_with(&["update", "t-1", "--content", "later"])
        .await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(
        err.contains("cannot be edited from status 'in_progress'"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn run_folds_diagnostics_into_the_single_writer() {
    let team = Team::operator();
    // `run` is the single-writer form: the failure reason lands on the same writer as the output.
    assert_eq!(
        team.run(&["task", "get", "nope"]).await,
        (1, "Task 'nope' not found\n".to_string())
    );
    assert_eq!(
        team.run(&["task", "get", "nope"]).await.0,
        1,
        "the failure exit code is preserved"
    );

    team.seed_task("t-1", "first task", 500).await;
    let (code, out) = team.run(&["task", "claimable"]).await;
    assert_eq!((code, out.as_str()), (0, "[t-1] first task\n"));
}

#[tokio::test]
async fn a_member_tree_command_under_an_operator_client_is_a_usage_error() {
    let team = Team::operator();
    let cli = parse(&["send_message", "dev-1", "hi"], ClientScope::Member);
    let mut out: Vec<u8> = Vec::new();
    let mut err: Vec<u8> = Vec::new();
    let code = cli
        .run_with(Arc::clone(&team.client), &mut out, &mut err)
        .await;
    assert_eq!(code, 2, "a tree mismatch is a CLI-side usage error");
    assert_eq!(out.len(), 0);
    assert_eq!(
        String::from_utf8(err).unwrap(),
        "internal: the operator command tree was not selected\n"
    );
}

// ---------------------------------------------------------------------------------------------
// Member runs against the store-backed client
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn member_inbox_renders_the_mailbox_and_consumes_it() {
    let team = Team::member();
    team.seed_direct("m-1", "team_leader", "dev-1", "hello", 1000)
        .await;

    assert_eq!(
        team.run_with(&["inbox"]).await,
        (
            0,
            "[1000] team_leader -> dev-1 : hello\n".to_string(),
            String::new()
        )
    );
    assert_eq!(team.unread("dev-1").await, Vec::new());
    assert_eq!(
        team.run_with(&["inbox"]).await,
        (0, "(inbox empty)\n".to_string(), String::new())
    );
    // Another member's mailbox is untouched.
    assert_eq!(team.unread("team_leader").await, Vec::new());
}

#[tokio::test]
async fn member_claim_task_claims_and_completes_for_the_descriptor_member() {
    let team = Team::member();
    team.seed_task("t-1", "first task", 500).await;

    assert_eq!(
        team.run_with(&["claim_task", "t-1", "claimed"]).await,
        (0, "claimed t-1\n".to_string(), String::new())
    );
    let claimed = team.task("t-1").await;
    assert_eq!(claimed.status, TaskStatus::InProgress);
    assert_eq!(claimed.assignee.as_deref(), Some("dev-1"));

    assert_eq!(
        team.run_with(&["claim_task", "t-1", "completed"]).await,
        (0, "completed t-1\n".to_string(), String::new())
    );
    assert_eq!(team.task("t-1").await.status, TaskStatus::Completed);
}

#[tokio::test]
async fn member_claim_task_refusal_names_the_task() {
    let team = Team::member();
    team.seed_task("t-1", "first task", 500).await;
    // Another member owns the task, so the board refuses the claim.
    team.board.claim("t-1", "dev-2").await.unwrap();

    let (code, out, err) = team.run_with(&["claim_task", "t-1", "claimed"]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(
        err.contains("already claimed by 'dev-2'"),
        "unexpected: {err}"
    );

    // An unknown task is a not-found failure.
    let (code, out, err) = team.run_with(&["claim_task", "nope", "claimed"]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(err.contains("task 'nope' not found"), "unexpected: {err}");
}

#[tokio::test]
async fn member_send_message_acks_and_persists() {
    let team = Team::member();
    let (code, out, err) = team
        .run_with(&["send_message", "team_leader", "starting"])
        .await;
    assert_eq!((code, err.as_str()), (0, ""), "unexpected: {err}");
    let message_id = direct_ack(&out, "sent ", "team_leader").to_string();

    let rows = team.unread("team_leader").await;
    let direct: Vec<_> = rows.iter().filter(|row| !row.broadcast).collect();
    assert_eq!(direct.len(), 1);
    assert_eq!(direct[0].message_id, message_id);
    assert_eq!(direct[0].from_member_name, "dev-1");
    assert_eq!(direct[0].content, "starting");

    // `*` broadcasts.
    let (code, out, err) = team.run_with(&["send_message", "*", "all hands"]).await;
    assert_eq!((code, err.as_str()), (0, ""), "unexpected: {err}");
    let line = out.strip_suffix('\n').expect("a trailing newline");
    let broadcast_id = line
        .strip_prefix("broadcast ")
        .unwrap_or_else(|| panic!("expected a 'broadcast <id>' line: {line:?}"));
    let rows = team.unread("team_leader").await;
    assert!(
        rows.iter()
            .any(|row| row.broadcast && row.message_id == broadcast_id),
        "the broadcast row is persisted"
    );
}

#[tokio::test]
async fn member_send_message_over_the_limit_is_refused() {
    let team = Team::member();
    let too_long = "x".repeat(20_001);
    let (code, out, err) = team
        .run_with(&["send_message", "team_leader", &too_long])
        .await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(err.contains("content too long"), "unexpected: {err}");
    assert_eq!(team.unread("team_leader").await, Vec::new());
}

#[tokio::test]
async fn member_view_task_renders_list_get_and_claimable() {
    let team = Team::member();
    team.seed_task("t-1", "first task", 500).await;

    assert_eq!(
        team.run_with(&["view_task"]).await,
        (
            0,
            "[t-1] pending first task (-)\n".to_string(),
            String::new()
        )
    );
    assert_eq!(
        team.run_with(&["view_task", "--action", "list", "--status", "pending"])
            .await,
        (
            0,
            "[t-1] pending first task (-)\n".to_string(),
            String::new()
        )
    );
    assert_eq!(
        team.run_with(&["view_task", "--action", "claimable"]).await,
        (0, "[t-1] first task\n".to_string(), String::new())
    );
    assert_eq!(
        team.run_with(&["view_task", "--action", "get", "--task-id", "t-1"])
            .await,
        (
            0,
            "id:        t-1\n\
             title:     first task\n\
             status:    pending\n\
             assignee:  -\n\
             blocked_by: -\n\
             blocks:     -\n\
             content:\n\
             body\n"
                .to_string(),
            String::new()
        )
    );
    // An unknown task id is rendered as JSON `null`, not an error.
    assert_eq!(
        team.run_with(&["view_task", "--action", "get", "--task-id", "nope"])
            .await,
        (0, "null\n".to_string(), String::new())
    );
}

#[tokio::test]
async fn member_view_task_on_an_empty_board_writes_nothing() {
    let team = Team::member();
    assert_eq!(
        team.run_with(&["view_task"]).await,
        (0, String::new(), String::new())
    );
    assert_eq!(
        team.run_with(&["view_task", "--action", "claimable"]).await,
        (0, String::new(), String::new())
    );
}

#[tokio::test]
async fn member_view_task_rejects_an_unknown_status_filter() {
    let team = Team::member();
    team.seed_task("t-1", "first task", 500).await;
    let (code, out, err) = team.run_with(&["view_task", "--status", "shipped"]).await;
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(
        err.contains("unknown task status 'shipped'"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn an_operator_tree_command_under_a_member_client_is_a_usage_error() {
    let team = Team::member();
    let cli = parse(&["members"], ClientScope::Operator);
    let mut out: Vec<u8> = Vec::new();
    let mut err: Vec<u8> = Vec::new();
    let code = cli
        .run_with(Arc::clone(&team.client), &mut out, &mut err)
        .await;
    assert_eq!(code, 2, "a tree mismatch is a CLI-side usage error");
    assert_eq!(out.len(), 0);
    assert_eq!(
        String::from_utf8(err).unwrap(),
        "internal: the member command tree was not selected\n"
    );
}

#[tokio::test]
async fn the_descriptor_identity_drives_the_client() {
    // The client the CLI runs against is always the descriptor's team/member/scope pair.
    let operator = Team::operator();
    assert_eq!(operator.client.scope(), ClientScope::Operator);
    assert_eq!(operator.client.member_name(), "user");

    let member = Team::member();
    assert_eq!(member.client.scope(), ClientScope::Member);
    assert_eq!(member.client.member_name(), "dev-1");

    // A member-scope client still exposes the roster through the seam (it is not a CLI word).
    member.seed_member("dev-1", TeamRole::Teammate).await;
    let roster = member.client.list_members().await.unwrap();
    assert_eq!(roster.len(), 1);
    assert_eq!(roster[0].member_name, "dev-1");
    assert_eq!(roster[0].role, TeamRole::Teammate);
    assert_eq!(roster[0].status, MemberStatus::Unstarted);
}
