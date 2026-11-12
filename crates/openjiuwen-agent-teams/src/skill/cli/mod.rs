//! The `team-member` command tree, the two-stage parse and the process entry point.
//!
//! Python parity source: `openjiuwen/agent_teams/skill/cli.py`. One invocation performs exactly one
//! operation and the exit code carries the outcome.
//!
//! The parse is two-staged, like Python's `parse_known_args`: the connection flags are scanned out
//! of the raw argv first ([`SkillCli::parse_descriptor_args`]), the join descriptor they point at
//! decides the [`ClientScope`], and only then is the scope-specific parser built
//! (`command::parse_member` or `command::parse_operator`). A malformed flag or descriptor
//! therefore fails before the subcommand is even considered, and an unknown subcommand fails with
//! the scope parser's own message.
//! `--help`/`-h` is the one exception: it is answered by the scope parser before a descriptor is
//! required, so help always reaches stdout and exits `0` (Python argparse parity).
//!
//! | Part | Responsibility |
//! | --- | --- |
//! | [`SkillCli`] | connection flags + the parsed scope command, and [`SkillCli::run`] |
//! | `command` | the clap trees, one root per scope |
//! | `output` | the plain-text line renderers |
//! | `member` | member-scope dispatch (the real teammate tool set) |
//! | `operator` | operator-scope dispatch (board, messaging, roster) |
//!
//! Exit codes: `0` success (including `--help` rendered on stdout, which needs no join descriptor),
//! `1` refused or unresolved team, `2` CLI-side error (unknown command or flag, missing argument,
//! malformed descriptor), `130` a cancelled or timed-out operation.

use std::io::Write;
use std::sync::Arc;

use openjiuwen_common::error::{Error, ErrorKind, Result};

use super::client::TeamClient;
use super::descriptor::{ClientScope, TeamJoinDescriptor};

pub(crate) mod command;
pub(crate) mod member;
pub(crate) mod operator;
pub(crate) mod output;

pub(crate) use command::{parse_member, parse_operator};
pub use command::{ClaimStatus, MemberCommand, OperatorCommand, TaskCommand, ViewAction};

/// Diagnostic written when an operation is refused without a reason.
const DEFAULT_REFUSAL: &str = "operation refused";

/// The connection flags shared by every `team-member` invocation.
///
/// They are scanned out of the raw argv before the scope-specific parser runs, so the descriptor
/// can be loaded first; the parser then accepts the same flags again (they are `global`), which is
/// what makes them legal before *and* after the subcommand.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DescriptorArgs {
    /// Path of a JSON team join descriptor (`--descriptor-file`).
    pub descriptor_file: Option<String>,
    /// Inline JSON team join descriptor (`--descriptor-json`).
    pub descriptor_json: Option<String>,
}

/// One parsed `team-member` invocation: where to connect, and what to do.
///
/// Build it with [`SkillCli::try_parse_from`] (which is told the scope by the caller that already
/// loaded the descriptor) and execute it with [`SkillCli::run`] against a [`TeamClient`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCli {
    /// Connection source the invocation was given.
    descriptor: DescriptorArgs,
    /// The subcommand, in the tree selected by the descriptor's scope.
    command: CliCommand,
}

/// The scope-specific command word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliCommand {
    /// Member-scope command (the teammate tool set).
    Member(MemberCommand),
    /// Operator-scope command (the outside team-control surface).
    Operator(OperatorCommand),
}

impl SkillCli {
    /// Scans the raw argv for the two connection flags, ignoring everything else.
    ///
    /// `argv` is the process argument vector, program name included. Both `--flag value` and
    /// `--flag=value` are accepted; a repeated flag keeps its last value; a flag without a value
    /// (or followed by another flag) is refused. Every unrecognized token belongs to the
    /// scope-specific parser and is skipped here.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `--descriptor-file` or `--descriptor-json` is missing its
    /// value.
    pub fn parse_descriptor_args(argv: &[String]) -> Result<DescriptorArgs> {
        parse_descriptor_args(argv)
    }

    /// Parses one scope's command tree out of `argv`.
    ///
    /// `argv` is the process argument vector, program name included; `scope` comes from the
    /// descriptor the caller already loaded.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] carrying clap's message for an unknown subcommand, an unknown
    /// flag or a missing required argument.
    pub fn try_parse_from(argv: Vec<String>, scope: ClientScope) -> Result<Self> {
        match scope {
            ClientScope::Member => {
                let (descriptor, command) = parse_member(argv)?;
                Ok(Self {
                    descriptor,
                    command: CliCommand::Member(command),
                })
            }
            ClientScope::Operator => {
                let (descriptor, command) = parse_operator(argv)?;
                Ok(Self {
                    descriptor,
                    command: CliCommand::Operator(command),
                })
            }
        }
    }

    /// Loads the join descriptor this invocation was given.
    ///
    /// Precedence is `--descriptor-json` > `--descriptor-file` > the
    /// [`super::descriptor::TEAM_JOIN_ENV`] environment variable, and a blank flag value counts as
    /// absent (Python `if descriptor_json:`).
    ///
    /// # Errors
    ///
    /// Propagates the loader's failure: [`Error::Config`] for an unreadable file or an unset
    /// environment variable, [`Error::Validation`] for malformed JSON.
    pub fn load_descriptor(&self) -> Result<TeamJoinDescriptor> {
        load_descriptor_from(&self.descriptor)
    }

    /// Returns the parsed subcommand, in the tree selected by the descriptor's scope.
    #[must_use]
    pub const fn command(&self) -> &CliCommand {
        &self.command
    }

    /// Returns the connection flags this invocation carried.
    #[must_use]
    pub const fn descriptor_args(&self) -> &DescriptorArgs {
        &self.descriptor
    }

    /// Runs the parsed command, writing every failure reason into the same writer as the output.
    ///
    /// This is the single-writer form: the diagnostics are buffered and appended to `out` after the
    /// operation, so a caller that only owns stdout still sees why an operation failed. Use
    /// [`SkillCli::run_with`] when stdout and stderr are separate.
    pub async fn run(&self, client: Arc<dyn TeamClient>, out: &mut dyn Write) -> i32 {
        let mut diagnostics: Vec<u8> = Vec::new();
        let code = self.run_with(client, out, &mut diagnostics).await;
        if !diagnostics.is_empty() {
            let _ = out.write_all(&diagnostics);
        }
        code
    }

    /// Runs the parsed command, writing output to `out` and failures to `err`.
    ///
    /// The tree is selected by the client's own [`TeamClient::scope`], so a client that does not
    /// match the parsed command tree is reported as an internal mismatch instead of running.
    pub async fn run_with(
        &self,
        client: Arc<dyn TeamClient>,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> i32 {
        match client.scope() {
            ClientScope::Member => member::dispatch(self, client, out, err).await,
            ClientScope::Operator => operator::dispatch(self, client, out, err).await,
        }
    }
}

/// Scans the raw argv for the two connection flags (see [`SkillCli::parse_descriptor_args`]).
///
/// # Errors
///
/// Returns [`Error::Validation`] when a connection flag is missing its value.
pub fn parse_descriptor_args(argv: &[String]) -> Result<DescriptorArgs> {
    let mut scanned = DescriptorArgs::default();
    let mut index = 0;
    while let Some(token) = argv.get(index).map(String::as_str) {
        if let Some(value) = token.strip_prefix("--descriptor-file=") {
            scanned.descriptor_file = Some(value.to_string());
        } else if let Some(value) = token.strip_prefix("--descriptor-json=") {
            scanned.descriptor_json = Some(value.to_string());
        } else if token == "--descriptor-file" {
            scanned.descriptor_file = Some(flag_value(argv, index, "--descriptor-file")?);
            index += 1;
        } else if token == "--descriptor-json" {
            scanned.descriptor_json = Some(flag_value(argv, index, "--descriptor-json")?);
            index += 1;
        }
        index += 1;
    }
    Ok(scanned)
}

/// Returns the value following the flag at `index`, or refuses a flag without one.
fn flag_value(argv: &[String], index: usize, flag: &str) -> Result<String> {
    match argv.get(index + 1) {
        Some(value) if !value.starts_with("--") => Ok(value.clone()),
        _ => Err(Error::validation(format!("{flag} requires a value"))),
    }
}

/// Loads the descriptor from the flags, falling back to the environment (see
/// [`SkillCli::load_descriptor`]).
fn load_descriptor_from(args: &DescriptorArgs) -> Result<TeamJoinDescriptor> {
    if let Some(raw) = non_blank(args.descriptor_json.as_deref()) {
        return TeamJoinDescriptor::from_json(raw);
    }
    if let Some(path) = non_blank(args.descriptor_file.as_deref()) {
        return TeamJoinDescriptor::from_file(path);
    }
    TeamJoinDescriptor::from_env()
}

/// Returns `value` unless it is absent or blank.
fn non_blank(value: Option<&str>) -> Option<&str> {
    value.filter(|text| !text.trim().is_empty())
}

/// Returns the exit code of an error: `130` for a cancelled or timed-out operation, `1` otherwise.
pub(crate) fn exit_code(err: &Error) -> i32 {
    match err.kind() {
        ErrorKind::Cancelled | ErrorKind::Timeout => 130,
        _ => 1,
    }
}

/// Writes `text` plus a trailing newline, mapping a write failure onto [`Error::validation`].
pub(crate) fn write_lined(out: &mut dyn Write, text: &str) -> Result<()> {
    writeln!(out, "{text}")
        .map_err(|err| Error::validation(format!("failed to write the command output: {err}")))
}

/// Writes `lines` one per line and returns the exit code (`0` also when there is nothing to write).
pub(crate) fn emit_lines(out: &mut dyn Write, err: &mut dyn Write, lines: &[String]) -> i32 {
    if lines.is_empty() {
        return 0;
    }
    emit_line(out, err, &lines.join("\n"))
}

/// Writes `text` and returns the exit code, reporting a write failure on `err`.
pub(crate) fn emit_line(out: &mut dyn Write, err: &mut dyn Write, text: &str) -> i32 {
    match write_lined(out, text) {
        Ok(()) => 0,
        Err(error) => fail(err, &error),
    }
}

/// Writes a CLI-side error onto the diagnostic writer and returns exit code `2`.
///
/// Used for the connection stage only (bad flag, unreadable descriptor, unknown subcommand): those
/// problems are the caller's, not the team's, so they are distinguished from an operation that was
/// refused (`1`) or an unresolved team (`1`).
fn fail_cli(err_out: &mut dyn Write, err: &Error) -> i32 {
    let _ = writeln!(err_out, "{err}");
    2
}

/// Writes an error onto the diagnostic writer and returns its exit code.
pub(crate) fn fail(err_out: &mut dyn Write, err: &Error) -> i32 {
    let _ = writeln!(err_out, "{err}");
    exit_code(err)
}

/// Writes an operation's refusal reason and returns exit code `1`.
pub(crate) fn refuse(err_out: &mut dyn Write, reason: Option<&str>) -> i32 {
    let _ = writeln!(err_out, "{}", reason.unwrap_or(DEFAULT_REFUSAL));
    1
}

/// Resolves the live team behind `descriptor`, when this process hosts one.
///
/// The team is resolved through `crate::runtime::try_global_registry()` and the manager's
/// `live(team_name)`, then wrapped over that team's own store, task board and messager (see
/// [`super::live::connect_live_client`], the single wiring point this CLI shares with the
/// `openjiuwen-team-mcp` server). A team this process does not host — including every team hosted
/// by *another* process, since cross-process attach is deferred — resolves to `None`, which the
/// connection stage reports as `no live team '<name>' in this process` and exit code `1`.
fn live_client(descriptor: &TeamJoinDescriptor) -> Option<Arc<dyn TeamClient>> {
    super::live::connect_live_client(descriptor)
}

/// Parses and executes one `team-member` invocation against `std::io` and returns the exit code.
///
/// This is what the `team-member` binary calls: the live team is resolved through `live_client`,
/// output goes to stdout and diagnostics to stderr (through `writeln!`, never `println!`). Every
/// CLI-side problem (bad flag, malformed descriptor, unknown command) exits `2`; a team this
/// process does not host exits `1`.
/// `--help` (and `-h`) is rendered on stdout and exits `0`, even without a join descriptor.
#[must_use]
pub fn run_skill_cli(argv: Vec<String>) -> i32 {
    let mut out = std::io::stdout();
    let mut err = std::io::stderr();
    run_cli_with_resolver(argv, &live_client, &mut out, &mut err)
}

/// Runs one invocation against an already-resolved client: the test hook of [`run_skill_cli`].
///
/// `live` is `None` when no team is hosted in this process, which is reported exactly like
/// [`run_skill_cli`]'s own resolution failure.
#[cfg(test)]
pub(crate) fn run_skill_cli_with(
    argv: Vec<String>,
    live: Option<Arc<dyn TeamClient>>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let resolver = move |_: &TeamJoinDescriptor| live.clone();
    run_cli_with_resolver(argv, &resolver, out, err)
}

/// The single flow behind [`run_skill_cli`] and [`run_skill_cli_with`].
///
/// The connection flags are scanned first, then the descriptor is loaded *once* and kept as a
/// `Result`: its scope decides which parser renders `--help` (stage 1, stdout, exit `0`) and which
/// parser reads the command (stage 2). A descriptor that does not load still renders help through
/// the default operator tree, so `--help` never depends on a join descriptor; a real command with
/// no descriptor keeps failing on stderr with exit `2`.
fn run_cli_with_resolver(
    argv: Vec<String>,
    resolve: &dyn Fn(&TeamJoinDescriptor) -> Option<Arc<dyn TeamClient>>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let descriptor_args = match SkillCli::parse_descriptor_args(&argv) {
        Ok(scanned) => scanned,
        Err(error) => return fail_cli(err, &error),
    };
    let descriptor = load_descriptor_from(&descriptor_args);
    let scope = descriptor
        .as_ref()
        .map_or(ClientScope::Operator, |loaded| loaded.scope);
    if let Some(text) = command::display_text(argv.clone(), scope) {
        // A write failure is ignored, exactly like `fail_cli`: help must not turn into an exit code.
        let _ = out.write_all(text.as_bytes());
        return 0;
    }
    let descriptor = match descriptor {
        Ok(descriptor) => descriptor,
        Err(error) => return fail_cli(err, &error),
    };
    let cli = match SkillCli::try_parse_from(argv, descriptor.scope) {
        Ok(cli) => cli,
        Err(error) => return fail_cli(err, &error),
    };
    let Some(client) = resolve(&descriptor) else {
        let _ = writeln!(
            err,
            "no live team '{}' in this process",
            descriptor.team_name
        );
        return 1;
    };
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = writeln!(err, "failed to start the async runtime: {error}");
            return 1;
        }
    };
    runtime.block_on(cli.run_with(client, out, err))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
pub(crate) mod test_support {
    //! The joined-team fixture the CLI suites share.

    use std::sync::Arc;

    use crate::messager::InProcessMessager;
    use crate::schema::{TaskGraphSpec, TeamRole};
    use crate::storage::{InMemoryTeamStore, MemberRecord, TeamStore};
    use crate::taskboard::TaskBoard;

    use super::super::client::TeamClient;
    use super::super::descriptor::{ClientScope, TeamJoinDescriptor};
    use super::super::store_client::StoreTeamClient;

    /// One team's store, board and client, built the way the runtime bridge builds them.
    pub(crate) struct CliFixture {
        /// Persistence seam (mailbox and roster rows the assertions read back).
        pub(crate) store: Arc<dyn TeamStore>,
        /// Task board the operator scope drives.
        pub(crate) board: Arc<TaskBoard>,
        /// The client under test.
        pub(crate) client: Arc<dyn TeamClient>,
    }

    impl CliFixture {
        /// Builds a fixture for team `alpha` joined by `member_name` under `scope`.
        pub(crate) fn new(scope: ClientScope, member_name: &str) -> Self {
            let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
            let board = Arc::new(TaskBoard::with_defaults(store.clone(), "alpha"));
            let descriptor = TeamJoinDescriptor {
                session_id: "s1".to_string(),
                team_name: "alpha".to_string(),
                member_name: member_name.to_string(),
                scope,
                ..TeamJoinDescriptor::default()
            };
            let client = Arc::new(StoreTeamClient::new(
                descriptor.clone(),
                store.clone(),
                board.clone(),
                Arc::new(InProcessMessager::new()),
            ));
            Self {
                store,
                board,
                client,
            }
        }

        /// Seeds one pending task titled `title`.
        pub(crate) async fn seed_task(&self, task_id: &str, title: &str) {
            let spec = TaskGraphSpec {
                title: title.to_string(),
                content: "body".to_string(),
                task_id: Some(task_id.to_string()),
                ..TaskGraphSpec::default()
            };
            self.board.create_task(spec).await.expect("seed task");
        }

        /// Seeds one roster row (`unstarted`, default execution status).
        pub(crate) async fn seed_member(&self, member_name: &str, role: TeamRole) {
            self.store
                .insert_member(MemberRecord::new("alpha", member_name, role))
                .await
                .expect("seed member");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write;

    use super::command::PROG;
    use super::test_support::CliFixture;
    use super::*;

    /// Builds a process-style argument vector: program name first, then `args`.
    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once(PROG.to_string())
            .chain(args.iter().map(|arg| (*arg).to_string()))
            .collect()
    }

    /// Parses `args` with the operator tree.
    fn operator(args: &[&str]) -> SkillCli {
        SkillCli::try_parse_from(argv(args), ClientScope::Operator).unwrap()
    }

    #[test]
    fn descriptor_args_are_scanned_before_and_after_the_subcommand() {
        let before =
            parse_descriptor_args(&argv(&["--descriptor-file", "join.json", "task", "list"]))
                .unwrap();
        assert_eq!(
            before,
            DescriptorArgs {
                descriptor_file: Some("join.json".to_string()),
                descriptor_json: None,
            }
        );

        let after = parse_descriptor_args(&argv(&[
            "task",
            "list",
            "--descriptor-json",
            "{\"team_name\":\"alpha\"}",
        ]))
        .unwrap();
        assert_eq!(
            after.descriptor_json.as_deref(),
            Some("{\"team_name\":\"alpha\"}")
        );
        assert_eq!(after.descriptor_file, None);
    }

    #[test]
    fn descriptor_args_accept_the_equals_form() {
        let args = parse_descriptor_args(&argv(&[
            "--descriptor-json={\"team_name\":\"alpha\"}",
            "--descriptor-file=/tmp/join.json",
        ]))
        .unwrap();
        assert_eq!(
            args.descriptor_json.as_deref(),
            Some("{\"team_name\":\"alpha\"}")
        );
        assert_eq!(args.descriptor_file.as_deref(), Some("/tmp/join.json"));
    }

    #[test]
    fn descriptor_flag_without_a_value_is_refused() {
        let missing = parse_descriptor_args(&argv(&["inbox", "--descriptor-file"])).unwrap_err();
        assert_eq!(
            missing.to_string(),
            "validation: --descriptor-file requires a value"
        );
        let followed_by_flag =
            parse_descriptor_args(&argv(&["--descriptor-json", "--watch"])).unwrap_err();
        assert_eq!(
            followed_by_flag.to_string(),
            "validation: --descriptor-json requires a value"
        );
    }

    #[test]
    fn unknown_tokens_are_ignored_and_the_last_flag_wins() {
        let args = parse_descriptor_args(&argv(&[
            "task",
            "--status",
            "pending",
            "--descriptor-file",
            "a.json",
            "get",
            "t-1",
            "--descriptor-file",
            "b.json",
        ]))
        .unwrap();
        assert_eq!(args.descriptor_file.as_deref(), Some("b.json"));
    }

    #[test]
    fn try_parse_from_keeps_the_connection_flags() {
        let cli = operator(&["members", "--descriptor-json", "{\"team_name\":\"alpha\"}"]);
        assert_eq!(
            cli.descriptor.descriptor_json.as_deref(),
            Some("{\"team_name\":\"alpha\"}")
        );
        assert_eq!(
            cli.command(),
            &CliCommand::Operator(OperatorCommand::Members)
        );

        let member = SkillCli::try_parse_from(
            argv(&["--descriptor-file", "join.json", "inbox"]),
            ClientScope::Member,
        )
        .unwrap();
        assert_eq!(
            member.descriptor.descriptor_file.as_deref(),
            Some("join.json")
        );
        assert_eq!(
            member.command(),
            &CliCommand::Member(MemberCommand::Inbox { watch: false })
        );
    }

    #[test]
    fn parsed_shapes_are_publicly_inspectable() {
        let operator_task = SkillCli::try_parse_from(
            argv(&[
                "--descriptor-json",
                "{\"team_name\":\"alpha\"}",
                "task",
                "get",
                "t-1",
            ]),
            ClientScope::Operator,
        )
        .unwrap();
        assert_eq!(
            operator_task.descriptor_args().descriptor_json.as_deref(),
            Some("{\"team_name\":\"alpha\"}")
        );
        assert_eq!(
            operator_task.command(),
            &CliCommand::Operator(OperatorCommand::Task {
                action: TaskCommand::Get {
                    task_id: "t-1".to_string(),
                },
            })
        );

        let operator_update = SkillCli::try_parse_from(
            argv(&["update", "t-1", "--title", "new"]),
            ClientScope::Operator,
        )
        .unwrap();
        assert_eq!(
            operator_update.command(),
            &CliCommand::Operator(OperatorCommand::Update {
                task_id: "t-1".into(),
                title: Some("new".into()),
                content: None,
            })
        );

        let member_inbox = SkillCli::try_parse_from(
            argv(&["inbox", "--watch", "--descriptor-file", "join.json"]),
            ClientScope::Member,
        )
        .unwrap();
        assert_eq!(
            member_inbox.descriptor_args().descriptor_file.as_deref(),
            Some("join.json")
        );
        assert_eq!(
            member_inbox.command(),
            &CliCommand::Member(MemberCommand::Inbox { watch: true })
        );

        let member_claim = SkillCli::try_parse_from(
            argv(&["claim_task", "t-1", "completed"]),
            ClientScope::Member,
        )
        .unwrap();
        assert_eq!(
            member_claim.command(),
            &CliCommand::Member(MemberCommand::ClaimTask {
                task_id: "t-1".into(),
                status: ClaimStatus::Completed,
            })
        );
    }

    #[test]
    fn descriptor_precedence_is_json_then_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        {
            let mut handle = file.as_file();
            handle.write_all(br#"{"team_name":"from-file"}"#).unwrap();
            handle.sync_all().unwrap();
        }

        let both = SkillCli {
            descriptor: DescriptorArgs {
                descriptor_file: Some(file.path().display().to_string()),
                descriptor_json: Some("{\"team_name\":\"from-json\"}".to_string()),
            },
            command: CliCommand::Operator(OperatorCommand::Members),
        };
        assert_eq!(both.load_descriptor().unwrap().team_name, "from-json");

        let only_file = SkillCli {
            descriptor: DescriptorArgs {
                descriptor_file: Some(file.path().display().to_string()),
                descriptor_json: None,
            },
            command: CliCommand::Operator(OperatorCommand::Members),
        };
        assert_eq!(only_file.load_descriptor().unwrap().team_name, "from-file");

        let missing = SkillCli {
            descriptor: DescriptorArgs {
                descriptor_file: Some("does-not-exist/join.json".to_string()),
                descriptor_json: None,
            },
            command: CliCommand::Operator(OperatorCommand::Members),
        };
        assert!(missing.load_descriptor().is_err());
    }

    #[test]
    fn exit_codes_follow_the_error_kind() {
        assert_eq!(exit_code(&Error::cancelled()), 130);
        assert_eq!(exit_code(&Error::timeout("too slow")), 130);
        assert_eq!(exit_code(&Error::validation("bad")), 1);
        assert_eq!(exit_code(&Error::config("missing")), 1);
    }

    #[test]
    fn unknown_command_exits_two() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&[
                "--descriptor-json",
                "{\"team_name\":\"alpha\",\"scope\":\"operator\"}",
                "explode",
            ]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 2);
        assert_eq!(out.len(), 0);
        assert!(String::from_utf8(err).unwrap().contains("explode"));
    }

    #[test]
    fn malformed_descriptor_json_exits_two() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&["--descriptor-json", "{not json", "members"]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 2);
        assert!(String::from_utf8(err)
            .unwrap()
            .contains("malformed team join descriptor"));
    }

    #[test]
    fn help_exits_zero_and_writes_stdout() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&[
                "--descriptor-json",
                "{\"team_name\":\"alpha\",\"member_name\":\"dev-1\",\"scope\":\"member\"}",
                "--help",
            ]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 0);
        assert_eq!(err.len(), 0, "help belongs on stdout, not stderr");
        let stdout = String::from_utf8(out).unwrap();
        assert!(
            stdout.contains("Usage: team-member"),
            "unexpected help: {stdout}"
        );
        assert!(stdout.contains("inbox"), "unexpected help: {stdout}");
    }

    /// Help is answered before the join descriptor is loaded, like Python argparse: a descriptor
    /// that would fail to load (here malformed JSON, and normally an absent
    /// `OPENJIUWEN_TEAM_JOIN`) cannot turn `--help` into a CLI-side error.
    /// This keeps the case independent of the ambient environment.
    #[test]
    fn help_does_not_require_a_descriptor() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&["--descriptor-json", "{not json", "--help"]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 0);
        assert_eq!(err.len(), 0, "a bad descriptor must not reach stderr");
        let stdout = String::from_utf8(out).unwrap();
        assert!(
            stdout.contains("Usage: team-member"),
            "unexpected help: {stdout}"
        );
    }

    #[test]
    fn help_after_the_subcommand_also_exits_zero() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&[
                "--descriptor-json",
                "{\"team_name\":\"alpha\",\"member_name\":\"dev-1\",\"scope\":\"member\"}",
                "inbox",
                "--help",
            ]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 0);
        assert_eq!(err.len(), 0, "help belongs on stdout, not stderr");
        assert!(!out.is_empty(), "the subcommand help should be rendered");
    }

    #[test]
    fn unknown_flag_still_exits_two() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&[
                "--descriptor-json",
                "{\"team_name\":\"alpha\",\"member_name\":\"dev-1\",\"scope\":\"member\"}",
                "--nope",
            ]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 2);
        assert_eq!(out.len(), 0);
        assert!(!err.is_empty(), "the clap diagnostic should reach stderr");
    }

    #[test]
    fn missing_value_exits_two_before_any_parsing() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&["members", "--descriptor-file"]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 2);
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "validation: --descriptor-file requires a value\n"
        );
    }

    #[test]
    fn a_descriptor_file_is_accepted_as_the_connection_source() {
        let file = tempfile::NamedTempFile::new().unwrap();
        {
            let mut handle = file.as_file();
            handle
                .write_all(br#"{"team_name":"alpha","scope":"operator"}"#)
                .unwrap();
            handle.sync_all().unwrap();
        }
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&[
                "--descriptor-file",
                &file.path().display().to_string(),
                "members",
            ]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 1);
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "no live team 'alpha' in this process\n"
        );
    }

    #[test]
    fn valid_descriptor_without_a_live_team_exits_one() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&[
                "--descriptor-json",
                "{\"team_name\":\"alpha\",\"scope\":\"operator\"}",
                "members",
            ]),
            None,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 1);
        assert_eq!(out.len(), 0);
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "no live team 'alpha' in this process\n"
        );
    }

    #[test]
    fn a_resolved_client_runs_against_its_own_writers() {
        let fixture = CliFixture::new(ClientScope::Operator, "user");
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run_skill_cli_with(
            argv(&[
                "--descriptor-json",
                "{\"team_name\":\"alpha\",\"scope\":\"operator\"}",
                "members",
            ]),
            Some(Arc::clone(&fixture.client)),
            &mut out,
            &mut err,
        );
        assert_eq!(code, 0);
        assert_eq!(err.len(), 0);
        assert_eq!(String::from_utf8(out).unwrap(), "");
    }

    #[test]
    fn run_folds_the_diagnostics_into_the_single_writer() {
        let fixture = CliFixture::new(ClientScope::Operator, "user");
        let cli = operator(&["task", "get", "nope"]);
        let mut out: Vec<u8> = Vec::new();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let code = runtime.block_on(cli.run(Arc::clone(&fixture.client), &mut out));
        assert_eq!(code, 1);
        assert_eq!(String::from_utf8(out).unwrap(), "Task 'nope' not found\n");
    }
}
