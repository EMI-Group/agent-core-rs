//! Tests for the spawn-mode configuration types in [`super::spawn_config`].
//!
//! Everything here is offline and deterministic: `PATH` lookups only use names that cannot exist,
//! and the only filesystem touches go through `tempfile`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use openjiuwen_common::error::ErrorKind;

use super::spawn_config::{
    ExternalCliRuntimeConfig, MemberSpawnKind, MemberSpawnOptions, ProcessMemberConfig,
    SshTransportSpec, StdioCommand,
};
use crate::schema::SpawnMode;

// --- MemberSpawnKind -------------------------------------------------------------------------

#[test]
fn spawn_kind_wire_values() {
    for (kind, wire) in [
        (MemberSpawnKind::InProcess, "inprocess"),
        (MemberSpawnKind::Process, "process"),
        (MemberSpawnKind::ExternalCli, "external_cli"),
        (MemberSpawnKind::Ssh, "ssh"),
    ] {
        assert_eq!(serde_json::to_value(kind).unwrap(), serde_json::json!(wire));
        let decoded: MemberSpawnKind = serde_json::from_value(serde_json::json!(wire)).unwrap();
        assert_eq!(decoded, kind);
        assert_eq!(kind.as_str(), wire);
    }
}

#[test]
fn spawn_kind_defaults_to_in_process() {
    assert_eq!(MemberSpawnKind::default(), MemberSpawnKind::InProcess);
    assert!(MemberSpawnKind::InProcess.is_in_process());
    for kind in [
        MemberSpawnKind::Process,
        MemberSpawnKind::ExternalCli,
        MemberSpawnKind::Ssh,
    ] {
        assert!(!kind.is_in_process());
    }
}

#[test]
fn spawn_kind_rejects_unknown_values() {
    for wire in ["in_process", "external", "externalcli", "", "LOCAL"] {
        assert!(
            serde_json::from_value::<MemberSpawnKind>(serde_json::json!(wire)).is_err(),
            "{wire} must not decode"
        );
    }
}

#[test]
fn spawn_kind_converts_from_the_spec_spawn_mode() {
    assert_eq!(
        MemberSpawnKind::from(SpawnMode::InProcess),
        MemberSpawnKind::InProcess
    );
    assert_eq!(
        MemberSpawnKind::from(SpawnMode::Process),
        MemberSpawnKind::Process
    );
    // The wire spelling is shared with the spec, so both enums agree on the in-process name.
    assert_eq!(
        serde_json::to_value(SpawnMode::InProcess).unwrap(),
        serde_json::to_value(MemberSpawnKind::InProcess).unwrap()
    );
}

// --- StdioCommand ----------------------------------------------------------------------------

#[test]
fn stdio_command_default_is_empty() {
    let command = StdioCommand::default();
    assert_eq!(command.program(), "");
    assert_eq!(command.args().len(), 0);
    assert!(command.env().is_empty());
    assert_eq!(command.cwd(), None);
    assert_eq!(command.to_argv(), vec![String::new()]);
}

#[test]
fn stdio_command_builders_accumulate() {
    let command = StdioCommand::new("team-member")
        .with_arg("serve")
        .with_args(["--team", "alpha"])
        .with_env("OPENJIUWEN_TEAM_NAME", "alpha")
        .with_cwd("/tmp/ws");

    assert_eq!(command.program(), "team-member");
    assert_eq!(command.args(), ["serve", "--team", "alpha"]);
    assert_eq!(
        command
            .env()
            .get("OPENJIUWEN_TEAM_NAME")
            .map(String::as_str),
        Some("alpha")
    );
    assert_eq!(command.cwd(), Some(std::path::Path::new("/tmp/ws")));
    assert_eq!(
        command.to_argv(),
        vec![
            "team-member".to_string(),
            "serve".to_string(),
            "--team".to_string(),
            "alpha".to_string(),
        ]
    );
}

#[test]
fn stdio_command_round_trips_through_json() {
    let mut env = BTreeMap::new();
    env.insert("API_KEY".to_string(), "secret".to_string());
    let command = StdioCommand {
        program: "team-member".to_string(),
        args: vec!["serve".to_string()],
        env,
        cwd: Some(PathBuf::from("/tmp/ws")),
    };

    let encoded = serde_json::to_value(&command).unwrap();
    assert_eq!(encoded["program"], serde_json::json!("team-member"));
    assert_eq!(encoded["cwd"], serde_json::json!("/tmp/ws"));

    let decoded: StdioCommand = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, command);
}

#[test]
fn stdio_command_tolerates_unknown_fields_and_missing_ones() {
    let decoded: StdioCommand = serde_json::from_str(
        r#"{"program":"ssh","args":["-V"],"env":{},"cwd":null,"future_knob":42}"#,
    )
    .unwrap();
    assert_eq!(decoded.program(), "ssh");
    assert_eq!(decoded.args(), ["-V"]);

    let sparse: StdioCommand = serde_json::from_str(r#"{"program":"ssh"}"#).unwrap();
    assert_eq!(sparse, StdioCommand::new("ssh"));
}

#[test]
fn resolve_program_rejects_an_empty_program() {
    let error = StdioCommand::default().resolve_program().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.message().contains("no program configured"));

    let error = StdioCommand::new("   ").resolve_program().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[test]
fn resolve_program_returns_an_existing_path_unchanged() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let configured = file.path().to_path_buf();
    let command = StdioCommand::new(configured.display().to_string());

    assert_eq!(command.resolve_program().unwrap(), configured);
}

#[test]
fn resolve_program_rejects_a_missing_path() {
    let missing = PathBuf::from("/nonexistent/openjiuwen/team-member");
    let error = StdioCommand::new(missing.display().to_string())
        .resolve_program()
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error
            .message()
            .contains("/nonexistent/openjiuwen/team-member"),
        "message must name the path: {}",
        error.message()
    );
}

#[test]
fn resolve_program_reports_a_bare_name_missing_from_path() {
    let bare = "openjiuwen-no-such-binary-xyz";
    let error = StdioCommand::new(bare).resolve_program().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.message().contains(bare),
        "message must name the program: {}",
        error.message()
    );
}

// --- ProcessMemberConfig / ExternalCliRuntimeConfig -------------------------------------------

#[test]
fn process_config_default_is_team_member_serve() {
    let config = ProcessMemberConfig::default();
    assert_eq!(config.command.program(), "team-member");
    assert_eq!(config.command.args(), ["serve"]);
    assert_eq!(
        serde_json::from_str::<ProcessMemberConfig>(r#"{"command":{"program":"spawn-helper"}}"#)
            .unwrap()
            .command
            .program(),
        "spawn-helper"
    );
}

#[test]
fn external_cli_config_round_trips() {
    let config = ExternalCliRuntimeConfig {
        command: StdioCommand::new("claude").with_arg("--print"),
        cli_agent: Some("claude".to_string()),
        provider: Some("anthropic".to_string()),
        model: Some("claude-sonnet".to_string()),
    };

    let encoded = serde_json::to_value(&config).unwrap();
    assert_eq!(encoded["cli_agent"], serde_json::json!("claude"));
    let decoded: ExternalCliRuntimeConfig = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, config);
    assert_eq!(ExternalCliRuntimeConfig::default().cli_agent, None);
}

// --- SshTransportSpec ------------------------------------------------------------------------

#[test]
fn ssh_spec_defaults_match_the_python_schema() {
    let spec = SshTransportSpec::default();
    assert_eq!(spec.program, "ssh");
    assert_eq!(spec.host, "");
    assert_eq!(spec.port, 22);
    assert_eq!(spec.connect_timeout_s, Some(15.0));
    assert!(spec.use_exec);
    assert!(!spec.agent);
    assert!(!spec.disable_host_key_check);
    assert!(spec.options.is_empty());
}

#[test]
fn ssh_spec_round_trips_through_json() {
    let spec = SshTransportSpec {
        program: "ssh".to_string(),
        host: "build-box".to_string(),
        port: 2222,
        username: Some("ci".to_string()),
        key_file: Some(PathBuf::from("/keys/id_ed25519")),
        password: Some("hunter2".to_string()),
        agent: true,
        known_hosts: Some(PathBuf::from("/etc/known_hosts")),
        disable_host_key_check: true,
        connect_timeout_s: Some(7.5),
        use_exec: false,
        options: BTreeMap::from([("Compression".to_string(), "yes".to_string())]),
        remote_command: "team-member serve".to_string(),
    };

    let encoded = serde_json::to_value(&spec).unwrap();
    assert_eq!(encoded["key_file"], serde_json::json!("/keys/id_ed25519"));
    assert_eq!(encoded["connect_timeout_s"], serde_json::json!(7.5));
    assert_eq!(encoded["use_exec"], serde_json::json!(false));

    let decoded: SshTransportSpec = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, spec);
}

#[test]
fn ssh_spec_tolerates_unknown_fields_and_fills_defaults() {
    let decoded: SshTransportSpec =
        serde_json::from_str(r#"{"host":"build-box","future_knob":true}"#).unwrap();
    assert_eq!(decoded.host, "build-box");
    assert_eq!(decoded.program, "ssh");
    assert_eq!(decoded.port, 22);
    assert!(decoded.use_exec);
}

#[test]
fn ssh_remote_target_includes_the_username_when_set() {
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        ..SshTransportSpec::default()
    };
    assert_eq!(spec.remote_target(), "build-box");

    let spec = SshTransportSpec {
        username: Some("ci".to_string()),
        ..spec
    };
    assert_eq!(spec.remote_target(), "ci@build-box");
}

#[test]
fn ssh_argv_for_a_default_spec_is_explicit() {
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };

    assert_eq!(
        spec.to_argv(),
        vec![
            "ssh".to_string(),
            "-p".to_string(),
            "22".to_string(),
            "-o".to_string(),
            "ConnectTimeout=15".to_string(),
            "--".to_string(),
            "build-box".to_string(),
            "'exec team-member serve'".to_string(),
        ]
    );
}

#[test]
fn ssh_argv_without_the_optional_flags() {
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        connect_timeout_s: None,
        use_exec: false,
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };

    assert_eq!(
        spec.to_argv(),
        vec![
            "ssh".to_string(),
            "-p".to_string(),
            "22".to_string(),
            "--".to_string(),
            "build-box".to_string(),
            "'team-member serve'".to_string(),
        ]
    );
}

#[test]
fn ssh_argv_orders_every_flag_and_sorts_options() {
    let spec = SshTransportSpec {
        program: "ssh".to_string(),
        host: "build-box".to_string(),
        port: 2222,
        username: Some("ci".to_string()),
        key_file: Some(PathBuf::from("/keys/id_ed25519")),
        known_hosts: Some(PathBuf::from("/etc/known_hosts")),
        disable_host_key_check: true,
        connect_timeout_s: Some(7.5),
        use_exec: true,
        options: BTreeMap::from([
            ("Compression".to_string(), "yes".to_string()),
            ("BatchMode".to_string(), "yes".to_string()),
        ]),
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };

    assert_eq!(
        spec.to_argv(),
        vec![
            "ssh".to_string(),
            "-p".to_string(),
            "2222".to_string(),
            "-i".to_string(),
            "/keys/id_ed25519".to_string(),
            "-o".to_string(),
            "ConnectTimeout=7.5".to_string(),
            "-o".to_string(),
            "UserKnownHostsFile=/etc/known_hosts".to_string(),
            "-o".to_string(),
            "StrictHostKeyChecking=no".to_string(),
            "-o".to_string(),
            "BatchMode=yes".to_string(),
            "-o".to_string(),
            "Compression=yes".to_string(),
            "--".to_string(),
            "ci@build-box".to_string(),
            "'exec team-member serve'".to_string(),
        ]
    );
}

#[test]
fn ssh_host_key_checking_is_never_disabled_implicitly() {
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };
    assert!(!spec
        .to_argv()
        .iter()
        .any(|arg| arg.contains("StrictHostKeyChecking")));

    let strict = SshTransportSpec {
        disable_host_key_check: true,
        ..spec
    };
    let argv = strict.to_argv();
    let index = argv
        .iter()
        .position(|arg| arg == "StrictHostKeyChecking=no")
        .expect("the flag is injected when host-key checks are disabled");
    assert_eq!(argv[index - 1], "-o");
}

#[test]
fn ssh_agent_and_password_add_no_flags() {
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        username: Some("ci".to_string()),
        password: Some("hunter2".to_string()),
        agent: true,
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };

    let with_flags = spec.to_argv();
    let without = SshTransportSpec {
        password: None,
        agent: false,
        ..spec
    }
    .to_argv();
    assert_eq!(with_flags, without);
    assert!(!with_flags.iter().any(|arg| arg.contains("hunter2")));
}

#[test]
fn ssh_quoted_remote_command_escapes_single_quotes() {
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        use_exec: false,
        remote_command: "team-member serve --name 'dev one'".to_string(),
        ..SshTransportSpec::default()
    };
    assert_eq!(
        spec.quoted_remote_command(),
        "'team-member serve --name '\\''dev one'\\'''".to_string()
    );
}

#[test]
fn ssh_use_exec_prefixes_the_quoted_remote_command() {
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };
    assert_eq!(spec.quoted_remote_command(), "'exec team-member serve'");

    let plain = SshTransportSpec {
        use_exec: false,
        ..spec
    };
    assert_eq!(plain.quoted_remote_command(), "'team-member serve'");
    assert!(plain.to_argv().last().unwrap().starts_with("'team-member"));
}

#[test]
fn ssh_validate_rejects_incomplete_specs() {
    let no_host = SshTransportSpec {
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };
    assert_eq!(no_host.validate().unwrap_err().kind(), ErrorKind::Config);

    let complete = SshTransportSpec {
        host: "build-box".to_string(),
        ..no_host
    };

    let no_port = SshTransportSpec {
        port: 0,
        ..complete.clone()
    };
    assert_eq!(
        no_port.validate().unwrap_err().kind(),
        ErrorKind::Validation
    );

    let no_command = SshTransportSpec {
        remote_command: String::new(),
        ..complete.clone()
    };
    assert_eq!(no_command.validate().unwrap_err().kind(), ErrorKind::Config);

    complete.validate().unwrap();
}

#[test]
fn ssh_to_command_keeps_the_argv_tail_and_the_inherited_environment() {
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        username: Some("ci".to_string()),
        key_file: Some(PathBuf::from("/keys/id_ed25519")),
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };

    let command = spec.to_command().unwrap();
    assert_eq!(command.program(), "ssh");
    assert_eq!(command.args(), &spec.to_argv()[1..]);
    assert!(command.env().is_empty(), "env stays inherited-only");
    assert_eq!(command.cwd(), None);

    let error = SshTransportSpec::default().to_command().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
}

// --- MemberSpawnOptions ----------------------------------------------------------------------

#[test]
fn spawn_options_default_to_in_process_without_launch_config() {
    let options = MemberSpawnOptions::default();
    assert_eq!(options.kind, MemberSpawnKind::InProcess);
    assert!(options.is_in_process());
    assert!(options.process.is_none());
    assert!(options.external_cli.is_none());
    assert!(options.ssh.is_none());
}

#[test]
fn spawn_options_builders_attach_per_kind_config() {
    let process = ProcessMemberConfig::default();
    let options =
        MemberSpawnOptions::from_kind(MemberSpawnKind::Process).with_process(process.clone());
    assert_eq!(options.kind, MemberSpawnKind::Process);
    assert!(!options.is_in_process());
    assert_eq!(options.process, Some(process));

    let external = MemberSpawnOptions::from_kind(MemberSpawnKind::ExternalCli)
        .with_external_cli(ExternalCliRuntimeConfig::default());
    assert_eq!(external.kind, MemberSpawnKind::ExternalCli);
    assert!(external.external_cli.is_some());

    let ssh = SshTransportSpec {
        host: "build-box".to_string(),
        ..SshTransportSpec::default()
    };
    let over_ssh = MemberSpawnOptions::from_kind(MemberSpawnKind::Ssh).with_ssh(ssh.clone());
    assert_eq!(over_ssh.ssh, Some(ssh));
}

#[test]
fn spawn_options_round_trip_through_json() {
    let options = MemberSpawnOptions::from_kind(MemberSpawnKind::Ssh).with_ssh(SshTransportSpec {
        host: "build-box".to_string(),
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    });

    let encoded = serde_json::to_value(&options).unwrap();
    assert_eq!(encoded["kind"], serde_json::json!("ssh"));
    assert_eq!(encoded["ssh"]["host"], serde_json::json!("build-box"));
    assert_eq!(encoded["process"], serde_json::Value::Null);

    let decoded: MemberSpawnOptions = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, options);

    let default: MemberSpawnOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(default, MemberSpawnOptions::default());
}
