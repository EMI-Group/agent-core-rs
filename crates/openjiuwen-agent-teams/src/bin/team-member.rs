//! `team-member`: the thin process wrapper around the crate's skill CLI.
//!
//! All parsing, descriptor resolution, client wiring and rendering live in
//! [`openjiuwen_agent_teams::run_skill_cli`]; this `main` only hands it the process argument
//! vector and forwards its exit code.
//!
//! **The program name is prepended.** The library's two-stage clap parse
//! (`SkillCli::parse_descriptor_args` followed by `SkillCli::try_parse_from`) expects the process
//! argument vector *with* `argv[0]` included, and clap treats that first element as the binary
//! name. Passing `std::env::args().skip(1)` would therefore swallow the first flag or subcommand
//! as the binary name and fail every real invocation, so the canonical `team-member` program name
//! is restored here. Non-UTF-8 arguments are passed through lossily rather than panicking.
//!
//! Process contract (owned by `run_skill_cli`, documented in `src/skill/cli/mod.rs`):
//!
//! * `0` — success, `--help` (rendered to stdout, no descriptor needed), and `serve` ending on
//!   `stop` or end of input;
//! * `1` — the operation was refused, no team could be resolved (no host in this process and no
//!   reachable attach socket), or `serve` could not build and start its member harness;
//! * `2` — a CLI-side error: a bad flag, a malformed descriptor, an unknown command, or anything
//!   else that fails the connection stage or the clap parse;
//! * `130` — the operation was cancelled or timed out.
//!
//! `serve` is the child end of the JSON-lines member protocol (see
//! [`openjiuwen_agent_teams::skill::serve`]); it reads its identity from the environment, needs
//! no join descriptor, and speaks the protocol over stdin/stdout until `stop` or end of input.
//!
//! Command output goes to stdout and diagnostics to stderr (the library writes through `writeln!`,
//! never `println!`).
fn main() {
    let argv: Vec<String> = std::iter::once(String::from("team-member"))
        .chain(
            std::env::args_os()
                .skip(1)
                .map(|arg| arg.to_string_lossy().into_owned()),
        )
        .collect();
    std::process::exit(openjiuwen_agent_teams::run_skill_cli(argv));
}
