//! `openjiuwen sessions` — discover, inspect and prune resumable sessions.
//!
//! One conversation can leave state in three places (see
//! [`catalog`](self::catalog)): the CLI session records under `<home>/sessions`,
//! the persistent-agent execution journals and the persistent-agent session
//! records under the workspace's `persistent_agent` node. This command lists all
//! three, shows what a resume would seed, and prunes old entries.
//!
//! It is read-only apart from `prune`: listing works on a home and a workspace
//! that were never materialised, so no directory is created here.
//!
//! [`crate::extras`] uses the same catalogue for `run --resume`/`chat --resume`,
//! which is why the resume note each of them prints goes to **stderr**: stdout
//! keeps the run's own output format (`text`, `json`, `stream-json`).
//!
//! # Output
//!
//! `--json` (group-global) prints exactly one pretty JSON document on stdout:
//! the row array of `list`, the `show` object (`id`, `kind`, `created_at`,
//! `detail`, `messages`, `plan`, `record`), or the `prune` object (`dry_run`,
//! `pruned`, `entries`).
//!
//! Text mode prints one item per line:
//!
//! - `list`: `<id>\t<created_at or ->\t<kind>\t<detail>`, or `no sessions`;
//! - `show`: `session <id>`, `kind <kind>`, `created <created_at or ->`, the
//!   detail line, `messages <n>`, one `<role>: <content>` line per seeded
//!   message, and `plan execution <ref or -> resume-seq <n>` when a journal was
//!   traced;
//! - `prune`: `would prune <kind>\t<id>` per row for a dry run, `pruned
//!   <kind>\t<id>` otherwise, or `nothing to prune`.
pub(crate) mod catalog;

use std::path::PathBuf;

use openjiuwen_common::error::Result;
use openjiuwen_harness::cli::{CliOptions, SessionRecord};
use openjiuwen_harness::ResumePlan;

use crate::sessions::catalog::{Catalog, Row};

/// `openjiuwen sessions <COMMAND>`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct SessionsCli {
    /// Base directory holding the persistent-agent state (logs/, sessions/, links/).
    ///
    /// Defaults to the workspace's `persistent_agent` node.
    #[arg(long, global = true, value_name = "DIR")]
    pub(crate) data_dir: Option<PathBuf>,
    /// Print machine-readable JSON instead of text.
    #[arg(long, global = true)]
    pub(crate) json: bool,
    /// The session operation to run.
    #[command(subcommand)]
    pub(crate) command: SessionsCommand,
}

/// The session operations a terminal user needs.
#[derive(Debug, Clone, clap::Subcommand)]
pub(crate) enum SessionsCommand {
    /// List persisted sessions, newest first.
    List {
        /// Only show one kind of record.
        #[arg(long, value_enum)]
        kind: Option<KindArg>,
        /// How many rows to show (0 = all).
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Show one session and what resuming it would seed.
    Show {
        /// Session (or execution) id.
        id: String,
    },
    /// Delete persisted sessions.
    Prune {
        /// Delete sessions older than this many days.
        #[arg(long, value_name = "DAYS", conflicts_with = "all")]
        older_than: Option<f64>,
        /// Delete every session.
        #[arg(long)]
        all: bool,
        /// Report what would be deleted without deleting it.
        #[arg(long)]
        dry_run: bool,
    },
}

/// The stores `list --kind` can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum KindArg {
    /// CLI session records.
    Cli,
    /// Persistent-agent execution journals.
    Execution,
    /// Persistent-agent session records.
    State,
}

impl KindArg {
    /// Returns the row kind tag spelling this value selects.
    fn as_str(self) -> &'static str {
        match self {
            Self::Cli => catalog::KIND_CLI,
            Self::Execution => catalog::KIND_EXECUTION,
            Self::State => catalog::KIND_STATE,
        }
    }
}

/// Runs `openjiuwen sessions`.
///
/// # Errors
///
/// Returns the catalogue's typed errors (an unknown id, no resumable session,
/// `prune` without a selector) and option-resolution errors from the harness.
pub(crate) async fn run(args: &SessionsCli, base: &openjiuwen_harness::cli::Cli) -> Result<()> {
    let options = CliOptions::resolve(base)?;
    let catalog = Catalog::for_options(&options, args.data_dir.as_deref());
    match &args.command {
        SessionsCommand::List { kind, limit } => list(&catalog, *kind, *limit, args.json).await,
        SessionsCommand::Show { id } => show(&catalog, id, args.json).await,
        SessionsCommand::Prune {
            older_than,
            all,
            dry_run,
        } => prune(&catalog, *older_than, *all, *dry_run, args.json).await,
    }
}

/// Prints the persisted rows, newest first.
///
/// The kind filter is applied before `limit`, and `limit` 0 means "all".
async fn list(catalog: &Catalog, kind: Option<KindArg>, limit: usize, json: bool) -> Result<()> {
    let rows: Vec<Row> = catalog
        .list()
        .await?
        .into_iter()
        .filter(|row| match kind {
            Some(kind) => row.kind == kind.as_str(),
            None => true,
        })
        .take(if limit == 0 { usize::MAX } else { limit })
        .collect();
    if json {
        return print_json(&rows);
    }
    if rows.is_empty() {
        println!("no sessions");
        return Ok(());
    }
    for row in &rows {
        println!(
            "{}\t{}\t{}\t{}",
            row.id,
            row.created_at.as_deref().unwrap_or("-"),
            row.kind,
            row.detail
        );
    }
    Ok(())
}

/// Prints one session and the turn a resume would seed.
async fn show(catalog: &Catalog, id: &str, json: bool) -> Result<()> {
    let (row, resolved) = catalog.show(id).await?;
    if json {
        let messages: Vec<MessageView> = resolved
            .history
            .iter()
            .map(|message| MessageView {
                role: message.role.as_str().to_owned(),
                content: message.text().into_owned(),
            })
            .collect();
        return print_json(&ShowView {
            id: &row.id,
            kind: row.kind,
            created_at: row.created_at.as_deref(),
            detail: &row.detail,
            messages,
            plan: resolved.plan.as_ref(),
            record: resolved.record.as_ref(),
        });
    }
    println!("session {}", row.id);
    println!("kind {}", row.kind);
    println!("created {}", row.created_at.as_deref().unwrap_or("-"));
    println!("{}", row.detail);
    println!("messages {}", resolved.history.len());
    for message in &resolved.history {
        // One line per message, so a multi-line message cannot look like a row.
        println!(
            "{}: {}",
            message.role.as_str(),
            message.text().replace('\n', " ")
        );
    }
    if let Some(plan) = &resolved.plan {
        println!(
            "plan execution {} resume-seq {}",
            plan.cursor.recovery_ref.as_deref().unwrap_or("-"),
            plan.cursor.resume_seq
        );
    }
    Ok(())
}

/// Deletes (or reports) the rows the selectors match.
async fn prune(
    catalog: &Catalog,
    older_than: Option<f64>,
    all: bool,
    dry_run: bool,
    json: bool,
) -> Result<()> {
    let entries = catalog.prune(older_than, all, dry_run).await?;
    if json {
        return print_json(&PruneView {
            dry_run,
            pruned: entries.len(),
            entries: &entries,
        });
    }
    if entries.is_empty() {
        println!("nothing to prune");
        return Ok(());
    }
    let verb = if dry_run { "would prune" } else { "pruned" };
    for row in &entries {
        println!("{verb} {}\t{}", row.kind, row.id);
    }
    Ok(())
}

/// One seeded message as `sessions show --json` reports it.
#[derive(Debug, serde::Serialize)]
struct MessageView {
    /// Message role (`system`, `user`, `assistant`, `tool`, …).
    role: String,
    /// Plain text of the message.
    content: String,
}

/// The `sessions show --json` document.
#[derive(Debug, serde::Serialize)]
struct ShowView<'a> {
    /// Session (or execution) id.
    id: &'a str,
    /// Which store the resolution came from.
    kind: &'a str,
    /// RFC 3339 creation timestamp, when the store records one.
    created_at: Option<&'a str>,
    /// One-line description of the store entry.
    detail: &'a str,
    /// The messages a resume would seed, oldest first.
    messages: Vec<MessageView>,
    /// The traced resume plan, when a journal could be reopened.
    plan: Option<&'a ResumePlan>,
    /// The CLI session record behind the resume, when it came from one.
    record: Option<&'a SessionRecord>,
}

/// The `sessions prune --json` document.
#[derive(Debug, serde::Serialize)]
struct PruneView<'a> {
    /// Whether anything was actually deleted.
    dry_run: bool,
    /// How many rows the selector matched.
    pruned: usize,
    /// The matched rows.
    entries: &'a [Row],
}

/// Prints `value` as one pretty JSON document on stdout.
fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_tags_match_the_catalog_tags() {
        assert_eq!(KindArg::Cli.as_str(), "cli");
        assert_eq!(KindArg::Execution.as_str(), "execution");
        assert_eq!(KindArg::State.as_str(), "state");
    }
}
