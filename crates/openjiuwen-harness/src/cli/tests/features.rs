//! Cross-feature tests of the extended CLI surface.
//!
//! One test per feature seam proves the phases 1–3 additions compose: the colour
//! flag and the styled permission prompt, input history, first-run onboarding,
//! instruction files, `mcp.json` discovery, permanent allow rules, the REPL
//! transcript (help, `!cmd`, `/sessions`, `/clear`, `/exit`), `/cost` and
//! `/status`, `/compact`, `/<skill>` turns, and escape-free `text` versus
//! single-object `json` output.
//!
//! Everything runs offline against the shared harness of [`super`]: scripted IO
//! handles, [`openjiuwen_test_support::ScriptedModel`], [`MapEnv`] instead of the
//! process environment, and temporary home/workspace directories. No test reads
//! the process environment, opens a socket, spawns a process or sleeps.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use tempfile::TempDir;

use super::*;
use crate::cli::allow_rules::AllowRules;
use crate::cli::compact::DEFAULT_KEEP_MESSAGES;
use crate::cli::history::History;
use crate::cli::instruction::{
    InstructionFiles, InstructionSource, PROJECT_MEMORY_HEADER, USER_MEMORY_HEADER,
};
use crate::cli::mcp_config;
use crate::cli::onboarding::{maybe_run, OnboardingOutcome, OnboardingWizard};
use crate::cli::paths::{self, NO_COLOR, OPENJIUWEN_MCP_CONFIG, OPENJIUWEN_NO_ONBOARDING};
use crate::cli::shell::{ShellOutput, ShellRunner};
use crate::cli::style::Style;
use crate::permission::{PermissionHost, PermissionLevel, PermissionsSection};
use crate::skills::SKILL_FILENAME;

/// [`ShellRunner`] that answers every command from one fixed captured output.
struct FixedShell {
    /// Output returned by every call.
    output: ShellOutput,
    /// Commands the REPL asked to run, in call order.
    commands: Mutex<Vec<String>>,
}

impl FixedShell {
    /// Builds a runner answering `output`.
    fn new(output: ShellOutput) -> Arc<Self> {
        Arc::new(Self {
            output,
            commands: Mutex::new(Vec::new()),
        })
    }

    /// Returns the recorded commands.
    fn commands(&self) -> Vec<String> {
        self.commands.lock().unwrap().clone()
    }
}

#[async_trait]
impl ShellRunner for FixedShell {
    async fn run(&self, command: &str, _cwd: &Path) -> Result<ShellOutput> {
        self.commands.lock().unwrap().push(command.to_owned());
        Ok(self.output.clone())
    }
}

/// Returns the workspace-rooted options of a temporary CLI home.
fn prepared_options(temp: &TempDir) -> CliOptions {
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    options
}

/// Builds a REPL over an explicit terminal and model.
async fn repl_over(options: &CliOptions, io: CliIo, model: Arc<dyn Model>) -> Repl {
    let agent = build_agent(model, options, &io).await.unwrap();
    Repl::with_io(agent, options.clone(), io)
}

/// Removes every `ESC [ … m` sequence, leaving the printable text.
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('\x1b') {
        out.push_str(&rest[..start]);
        match rest[start..].find('m') {
            Some(end) => rest = &rest[start + end + 1..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

#[test]
fn no_color_flag_parses_before_and_after_a_subcommand_and_gates_colour() {
    // The flag is global, so both positions parse to the same `Cli`.
    let before = cli(&["--no-color", "run", "hi"]);
    let after = cli(&["run", "hi", "--no-color"]);
    assert!(before.no_color, "flag before the subcommand");
    assert!(after.no_color, "flag after the subcommand");
    for parsed in [&before, &after] {
        assert!(matches!(parsed.command, Some(Command::Run(_))));
        assert!(!parsed.verbose);
        assert!(parsed.model.is_none());
    }
    assert!(!cli(&[]).no_color, "colour is on by default");
    assert!(!cli(&["chat"]).no_color);

    // `color_enabled` needs a terminal, no flag and no `NO_COLOR`.
    let clear = MapEnv::new();
    let blocked = MapEnv::from_pairs([(NO_COLOR, "1")]);
    for (flag, no_color, terminal, expected) in [
        (false, false, true, true),
        (false, true, true, false),
        (true, false, true, false),
        (true, true, true, false),
        (false, false, false, false),
        (false, true, false, false),
        (true, false, false, false),
    ] {
        let env: &dyn EnvSource = if no_color { &blocked } else { &clear };
        assert_eq!(
            paths::color_enabled(flag, env, terminal),
            expected,
            "flag={flag} no_color={no_color} terminal={terminal}"
        );
    }

    // The handle carries the resolved decision; `CliIo` defaults to plain.
    let (io, _buffer) = scripted_io(&[]);
    assert!(!io.is_color());
    assert!(io.with_color(true).is_color());
}

#[test]
fn styled_prompt_matches_the_plain_prompt_byte_for_byte() {
    let arguments = json!({"path": "a.txt", "content": "hello"});
    let plain = render_prompt("write_file", &arguments);
    assert_eq!(
        render_prompt_styled("write_file", &arguments, Style::plain()),
        plain
    );
    assert!(!plain.contains('\x1b'), "plain prompt: {plain:?}");

    let styled = render_prompt_styled("write_file", &arguments, Style::new(true));
    assert!(
        styled.contains("\x1b[1;33m⚠ Approve write_file?\x1b[0m"),
        "header: {styled:?}"
    );
    assert!(
        styled.contains("\x1b[2m  path: a.txt\x1b[0m"),
        "argument line: {styled:?}"
    );
    assert!(
        styled.contains("\x1b[2m(y/yes, a/always, n/no, or feedback)\x1b[0m"),
        "legend: {styled:?}"
    );
    // Only the colours differ: stripping the escapes restores the plain prompt.
    assert_eq!(strip_ansi(&styled), plain);
    // The last line stays unpainted, as in the plain prompt.
    assert!(styled.ends_with("\nApprove? (y/a/n)"));

    // A direct `Style` call is the other positive-colour seam.
    assert_eq!(Style::new(true).cyan("x"), "\x1b[36mx\x1b[0m");
    assert_eq!(Style::plain().cyan("x"), "x");
}

#[test]
fn history_round_trips_at_the_default_location() {
    let temp = TempDir::new().unwrap();
    let home = OpenJiuwenHome::from_root(temp.path());
    let env = MapEnv::new();
    let path = paths::history_path(&home, &env);
    assert_eq!(path, temp.path().join(paths::HISTORY_FILENAME));

    let mut history = History::new(&path);
    assert!(history.record("first").unwrap());
    assert!(history.record("  second  ").unwrap());
    assert!(!history.record("   ").unwrap(), "blank lines are skipped");
    assert!(!history.record("second").unwrap(), "repeats are skipped");
    assert_eq!(history.entries(), ["first".to_owned(), "second".to_owned()]);

    let mut reloaded = History::new(&path);
    assert_eq!(reloaded.load().unwrap(), 2);
    assert_eq!(reloaded.entries(), history.entries());

    // A smaller capacity keeps the newest window in memory and on disk.
    let capped_path = temp.path().join("capped-history.txt");
    let mut capped = History::with_capacity(&capped_path, 3);
    for line in ["one", "two", "three", "four"] {
        assert!(capped.record(line).unwrap(), "{line} should be retained");
    }
    assert_eq!(
        capped.entries(),
        ["two".to_owned(), "three".to_owned(), "four".to_owned()]
    );
    assert_eq!(
        std::fs::read_to_string(&capped_path).unwrap(),
        "two\nthree\nfour\n"
    );
    let mut again = History::with_capacity(&capped_path, 3);
    assert_eq!(again.load().unwrap(), 3);
    assert_eq!(again.entries(), capped.entries());
}

#[tokio::test]
async fn onboarding_writes_settings_only_on_an_interactive_first_run() {
    let temp = TempDir::new().unwrap();
    let home = OpenJiuwenHome::from_root(temp.path());
    let env = MapEnv::new();
    let settings_file = paths::settings_path(&home);

    // A non-interactive handle is skipped and writes nothing, even with input.
    let (io, _buffer) = scripted_io(&["y"]);
    assert_eq!(
        maybe_run(&io, &home, &env).await.unwrap(),
        OnboardingOutcome::Skipped
    );
    assert!(!settings_file.exists(), "no settings file was written");

    // An interactive first run persists the answers.
    let (io, buffer) = interactive_io(&["y", "anthropic", "", "claude-3-5-sonnet", "sk-test"]);
    let outcome = maybe_run(&io, &home, &env).await.unwrap();
    match &outcome {
        OnboardingOutcome::Configured { path, .. } => assert_eq!(path, &settings_file),
        other => panic!("expected Configured, got {other:?}"),
    }
    assert!(output(&buffer).contains("Config saved to"));

    let settings = Settings::load(&settings_file).unwrap().unwrap();
    assert_eq!(settings.provider.as_deref(), Some("anthropic"));
    assert_eq!(settings.model.as_deref(), Some("claude-3-5-sonnet"));
    assert_eq!(
        settings.api_base.as_deref(),
        Some("https://api.anthropic.com")
    );
    assert_eq!(
        settings
            .api_key
            .as_ref()
            .map(|key| key.expose_secret().to_owned())
            .as_deref(),
        Some("sk-test")
    );

    // Needed only on a fresh interactive home, and never once the env switches it off.
    let fresh = OpenJiuwenHome::from_root(temp.path().join("fresh"));
    let disabled = MapEnv::from_pairs([(OPENJIUWEN_NO_ONBOARDING, "1")]);
    assert!(!OnboardingWizard::needed_with(&home, &env, true));
    assert!(OnboardingWizard::needed_with(&fresh, &env, true));
    assert!(!OnboardingWizard::needed_with(&fresh, &disabled, true));
    assert!(!OnboardingWizard::needed_with(&fresh, &env, false));
    assert!(!paths::settings_path(&fresh).exists());
}

#[test]
fn instruction_files_load_with_the_documented_precedence() {
    let temp = TempDir::new().unwrap();
    let workspace = temp.path().join("workspace");
    let home = OpenJiuwenHome::from_root(temp.path().join("home"));
    std::fs::create_dir_all(&workspace).unwrap();
    home.ensure().unwrap();
    std::fs::write(workspace.join("IDENTITY.md"), "workspace identity\n").unwrap();
    std::fs::write(home.join("IDENTITY.md"), "home identity\n").unwrap();
    std::fs::write(home.join("OPENJIUWEN.md"), "home memory\n").unwrap();
    std::fs::write(workspace.join("OPENJIUWEN.md"), "project memory\n").unwrap();

    let files = InstructionFiles::load(&workspace, &home).unwrap();
    assert_eq!(
        files.identity.as_ref().map(|doc| doc.source),
        Some(InstructionSource::WorkspaceIdentity),
        "the workspace identity wins"
    );
    assert_eq!(files.memory.len(), 2);
    assert_eq!(files.memory[0].source, InstructionSource::HomeMemory);
    assert_eq!(files.memory[1].source, InstructionSource::WorkspaceMemory);

    let prompt = files.system_prompt().unwrap();
    assert!(
        prompt.starts_with("workspace identity"),
        "prompt: {prompt:?}"
    );
    assert!(!prompt.contains("home identity"), "prompt: {prompt:?}");
    let user = prompt.find(USER_MEMORY_HEADER).expect("user header");
    let project = prompt.find(PROJECT_MEMORY_HEADER).expect("project header");
    assert!(user < project, "memory order: {prompt:?}");
    assert!(prompt.contains("home memory"), "prompt: {prompt:?}");
    assert!(prompt.contains("project memory"), "prompt: {prompt:?}");
}

#[test]
fn mcp_config_load_for_reads_one_server_and_names_a_malformed_file() {
    let temp = TempDir::new().unwrap();
    let home = OpenJiuwenHome::from_root(temp.path().join("home"));
    let workspace = temp.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let path = temp.path().join("mcp.json");
    std::fs::write(
        &path,
        r#"{"mcpServers": {"echo": {"command": "echo", "args": ["hello"]}}}"#,
    )
    .unwrap();
    let env = MapEnv::from_pairs([(OPENJIUWEN_MCP_CONFIG, path.to_str().unwrap())]);

    let servers = mcp_config::load_for(&home, &workspace, &env).unwrap();
    assert_eq!(servers.len(), 1);
    let server = &servers[0];
    assert_eq!(server.server_name, "echo");
    assert_eq!(server.client_type.as_str(), "stdio");
    assert_eq!(server.params["command"], json!("echo"));
    assert_eq!(server.params["args"][0], json!("hello"));

    // Malformed JSON is a typed configuration error naming the file.
    std::fs::write(&path, "{not json").unwrap();
    let error = mcp_config::load_for(&home, &workspace, &env).unwrap_err();
    assert!(matches!(error, Error::Config(_)), "unexpected: {error:?}");
    assert!(
        error.to_string().contains(path.to_str().unwrap()),
        "the error names the file: {error}"
    );
}

#[tokio::test]
async fn allow_rules_persist_and_are_reported_by_the_permission_host() {
    let temp = TempDir::new().unwrap();
    let workspace = temp.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let home = OpenJiuwenHome::from_root(temp.path().join("home"));
    let path = paths::allow_rules_path(&home);
    let rules = AllowRules::for_home(&home);
    assert_eq!(rules.path(), path.as_path());

    let section = PermissionsSection {
        tools: BTreeMap::from([("write_file".to_owned(), PermissionLevel::Allow)]),
        ..PermissionsSection::default()
    };
    let (io, _buffer) = interactive_io(&[]);
    let host = CliPermissionHost::new(
        io,
        "cli-allow".to_owned(),
        workspace,
        PermissionsSection::default(),
    )
    .with_allow_rules(rules.clone());
    assert_eq!(host.permission_yaml_path(), Some(path.clone()));
    assert!(
        host.persist_allow_rule(&section).await,
        "the write is reported as done"
    );
    assert!(path.exists(), "the rules file was written");

    let loaded = rules.load().unwrap().unwrap();
    assert_eq!(
        loaded.tools.get("write_file"),
        Some(&PermissionLevel::Allow)
    );
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("write_file: allow"), "YAML: {text}");

    // A host without a rules file reports no path and claims no write.
    let (io, _buffer) = interactive_io(&[]);
    let plain = CliPermissionHost::new(
        io,
        "cli-plain".to_owned(),
        temp.path().join("workspace"),
        PermissionsSection::default(),
    );
    assert!(plain.permission_yaml_path().is_none());
    assert!(!plain.persist_allow_rule(&section).await);
}

#[tokio::test]
async fn repl_transcript_covers_help_shell_sessions_clear_and_exit() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let shell = FixedShell::new(ShellOutput {
        command: "echo hi".to_owned(),
        stdout: "hi\n".to_owned(),
        stderr: String::new(),
        code: Some(0),
    });
    let scripted = ScriptedModel::text("unused");
    let probe = scripted.clone();
    let (io, buffer) = scripted_io(&["/help", "!echo hi", "/sessions", "/clear", "/exit"]);
    let mut repl = repl_over(&options, io, Arc::new(scripted))
        .await
        .with_shell(shell.clone());

    let before = repl.session_id().to_owned();
    repl.run().await.unwrap();
    let after = repl.session_id().to_owned();
    assert_ne!(before, after, "/clear starts a new session");

    let text = output(&buffer);
    for needle in [
        "Available commands:",
        "  /compact            Compact the conversation history",
        "! <cmd>             Run a shell command",
        "hi\n",
        "No saved sessions.",
        "Goodbye!",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in {text:?}");
    }
    assert!(
        text.contains(&format!(
            "Cleared conversation and started a new session: {after}"
        )),
        "clear banner: {text:?}"
    );
    assert_eq!(probe.call_count(), 0, "no line reached the model");
    assert_eq!(shell.commands(), vec!["echo hi".to_owned()]);
}

#[tokio::test]
async fn repl_cost_and_status_report_token_usage_after_a_turn() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let scripted = ScriptedModel::builder()
        .completion(MockCompletion::text("hi").with_usage(openjiuwen_foundation::Usage::new(3, 4)))
        .build();
    let probe = scripted.clone();
    let (io, buffer) = scripted_io(&["hello", "/status", "/cost", "/exit"]);
    let mut repl = repl_over(&options, io, Arc::new(scripted)).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(
        text.contains("Input 3 | Output 4 | Total 7 tokens | Calls 1 | Tool calls 0"),
        "status line: {text:?}"
    );
    assert!(
        text.contains(
            "Token usage:\n  Input:  3 tokens\n  Output: 4 tokens\n  Total:  7 tokens\n  Calls:  1\n  Tool calls: 0"
        ),
        "cost block: {text:?}"
    );
    assert_eq!(probe.call_count(), 1);
}

#[tokio::test]
async fn repl_compact_drops_old_messages_and_rewrites_the_transcript() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let (io, buffer) = scripted_io(&["/compact", "/exit"]);
    // Answers stay above the task-loop no-progress floor so every turn reaches
    // the model, and short enough to remain inside the compaction budget.
    let answers = (0..5)
        .map(|turn| format!("acknowledged turn {turn} of the compact transcript"))
        .collect::<Vec<String>>();
    let mut builder = ScriptedModel::builder();
    for answer in &answers {
        builder = builder.respond_text(answer.clone());
    }
    let mut repl = repl_over(&options, io, Arc::new(builder.build())).await;

    // Five turns mirror ten messages; the retained window keeps the newest eight.
    for (turn, answer) in answers.iter().enumerate() {
        let outcome = repl.run_turn(&format!("message {turn}")).await.unwrap();
        assert_eq!(&outcome.answer, answer, "turn {turn}");
    }
    let id = repl.session_id().to_owned();
    repl.run().await.unwrap();

    assert!(
        output(&buffer).contains("Compacted conversation: 2 message(s) dropped, 8 retained."),
        "compaction report: {:?}",
        output(&buffer)
    );
    let record = options.sessions().load(&id).unwrap().unwrap();
    assert_eq!(record.messages.len(), DEFAULT_KEEP_MESSAGES);
    assert_eq!(record.messages[0].role, "user");
    assert_eq!(record.messages[0].content, "message 1");
    assert_eq!(record.messages[7].role, "assistant");
    assert_eq!(record.messages[7].content, answers[4]);
}

#[tokio::test]
async fn repl_compact_reports_nothing_for_a_short_conversation() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let (io, buffer) = scripted_io(&["hello", "/compact", "/exit"]);
    let mut repl = repl_over(
        &options,
        io,
        Arc::new(ScriptedModel::text("scripted answer")),
    )
    .await;
    let id = repl.session_id().to_owned();
    repl.run().await.unwrap();

    assert!(output(&buffer).contains("Nothing to compact."));
    let record = options.sessions().load(&id).unwrap().unwrap();
    assert_eq!(record.messages.len(), 2, "the transcript is untouched");
    assert_eq!(record.messages[0].content, "hello");
    assert_eq!(record.messages[1].content, "scripted answer");
}

#[tokio::test]
async fn skill_slash_command_wraps_the_skill_file_into_the_prompt() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let directory = options.workspace.join("skills").join("my-skill");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join(SKILL_FILENAME),
        "---\nname: my-skill\ndescription: Example skill\n---\n# Body\nDo the thing.",
    )
    .unwrap();

    let scripted = ScriptedModel::text("done");
    let probe = scripted.clone();
    let (io, buffer) = scripted_io(&["/my-skill do it", "/exit"]);
    let mut repl = repl_over(&options, io, Arc::new(scripted)).await;
    repl.run().await.unwrap();

    assert!(output(&buffer).contains("done"));
    let prompt = probe
        .last_call()
        .expect("one model call")
        .last_user_text()
        .expect("a user message");
    assert!(
        prompt.starts_with("<skill-instructions>\n"),
        "wrapper: {prompt:?}"
    );
    assert!(prompt.contains("# Body\nDo the thing."), "body: {prompt:?}");
    assert!(
        prompt.ends_with("</skill-instructions>\n\nUser request: do it"),
        "request: {prompt:?}"
    );
    // An unknown slash command stays unknown rather than becoming a skill turn.
    assert_eq!(
        repl.handle_input("/nope"),
        ReplCommand::Unknown("/nope".to_owned())
    );
}

#[tokio::test]
async fn text_output_is_escape_free_and_json_is_one_object() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let args = |format| RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: format,
    };

    let (io, buffer) = interactive_io(&[]);
    assert!(!io.is_color());
    let model = ScriptedModel::builder()
        .completion(
            MockCompletion::text("plain answer")
                .with_usage(openjiuwen_foundation::Usage::new(3, 4)),
        )
        .build();
    let outcome = run_with_model(&options, &args(OutputFormat::Text), &io, Arc::new(model))
        .await
        .unwrap();
    assert_eq!(outcome.answer, "plain answer");
    assert!(outcome.usage.is_some(), "usage was captured");
    let text = output(&buffer);
    assert!(text.contains("plain answer"), "text output: {text:?}");
    assert!(!text.contains('\x1b'), "text output: {text:?}");

    let (io, buffer) = interactive_io(&[]);
    let model = ScriptedModel::builder()
        .completion(
            MockCompletion::text("json answer").with_usage(openjiuwen_foundation::Usage::new(3, 4)),
        )
        .build();
    let outcome = run_with_model(&options, &args(OutputFormat::Json), &io, Arc::new(model))
        .await
        .unwrap();
    let text = output(&buffer);
    assert!(!text.contains('\x1b'), "json output: {text:?}");
    let trimmed = text.trim();
    assert!(
        trimmed.starts_with('{') && trimmed.ends_with('}'),
        "{text:?}"
    );
    let value: Value = serde_json::from_str(trimmed).unwrap();
    assert_object_keys(&value, &["answer", "result_type", "session_id", "usage"]);
    assert_eq!(value["answer"], json!("json answer"));
    assert_eq!(value["session_id"], json!(outcome.session_id));
    assert!(value["usage"].is_object(), "usage object: {value}");
}
