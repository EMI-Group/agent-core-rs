//! `openjiuwen` CLI binary and run/repl front-end.
//!
//! Every piece of behaviour lives in [`openjiuwen_harness::cli`]: the clap
//! grammar ([`Cli`]), option resolution ([`openjiuwen_harness::cli::CliOptions`]),
//! the one-shot runner and the interactive REPL. This `main` owns only the
//! process boundary:
//!
//! 1. load `.env` if one exists (a missing or malformed file is not an error);
//! 2. install the `tracing` subscriber from `RUST_LOG` / `OPENJIUWEN_LOG_LEVEL`
//!    and `OPENJIUWEN_LOG_FORMAT`, written to **stderr** so stdout stays a clean
//!    result stream (`json` / `stream-json` output must not be mixed with logs);
//! 3. parse the command line with [`Cli::parse`], letting clap answer `--help`
//!    and `--version` (exit `0`) and reject bad input (exit `2`);
//! 4. run the harness CLI on the tokio runtime and translate a failure into a
//!    diagnostic on stderr plus exit code `1`.
//!
//! No user input can panic: parsing is delegated to clap and errors are values,
//! never `unwrap`s.
use std::process::ExitCode;

use clap::Parser;
use openjiuwen_common::logging::{self, LogWriter, LoggingConfig};
use openjiuwen_harness::cli::{run_cli, Cli};

/// Exit code of a successful invocation (and of `--help` / `--version`).
const EXIT_FAILURE: u8 = 1;

/// Process entry point: set up the process, parse, then hand over to the
/// harness CLI.
#[tokio::main]
async fn main() -> ExitCode {
    // A `.env` in the working directory is optional; its absence is the normal
    // case and must never stop the CLI from starting.
    let _ = dotenvy::dotenv();
    init_logging();

    // `parse()` exits the process itself: `0` after printing `--help` or
    // `--version`, `2` for a usage error (clap's documented exit code).
    let cli = Cli::parse();

    match run_cli(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(EXIT_FAILURE)
        }
    }
}

/// Installs the global `tracing` subscriber, always writing to stderr.
///
/// `RUST_LOG` wins when it is set (the conventional `tracing` filter syntax,
/// for example `openjiuwen_harness=debug,info`); otherwise the configuration
/// comes from `OPENJIUWEN_LOG_LEVEL` / `OPENJIUWEN_LOG_FORMAT`. An invalid
/// filter is reported as a warning and the defaults are installed instead, so a
/// typo in the environment cannot prevent the CLI from running.
fn init_logging() {
    let base = LoggingConfig::from_env().with_writer(LogWriter::Stderr);
    let config = match std::env::var("RUST_LOG") {
        Ok(filter) if !filter.trim().is_empty() => base.with_filter(filter.trim()),
        _ => base,
    };
    if let Err(err) = logging::init_with(&config) {
        eprintln!("warning: {err}; using the default log level");
        // `from_env` has no filter string, so its level is always a valid
        // directive and this second attempt cannot fail for the same reason.
        if let Err(err) =
            logging::init_with(&LoggingConfig::from_env().with_writer(LogWriter::Stderr))
        {
            eprintln!("warning: {err}");
        }
    }
}
