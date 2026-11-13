//! CI gate runner over the [`CommandRunner`] seam.
//!
//! Port of `harness_rsi/auto_harness/infra/ci_gate_runner.py` (v0.1.20) and the
//! `ci_gates` section of `harness_rsi/auto_harness/resources/ci_gate.yaml`.
//! Python's runner is one class that both loads a gate list and executes each
//! gate through `bash -c`; this module keeps that shape while making the pure
//! half (config parsing, command normalisation, diff scoping, output
//! sanitising) independently testable.
//!
//! Exit codes are data, as everywhere else in [`crate::runtime`]: a gate that
//! exits non-zero is a [`GateOutcome`] with `passed: false`, and only a failure
//! to run a command at all is an [`openjiuwen_common::error::Error`].
//!
//! Python parity notes:
//!
//! - The gate list comes from a `gates` (or `ci_gates`) sequence; a missing key
//!   or a parse failure yields no gates at all, matching Python's
//!   `_load_gates` fail-closed helper.
//! - `python -m …` is rewritten to the configured interpreter, and a `make`
//!   command is rewritten to the equivalent Python tool invocation when `make`
//!   is not on `PATH` (`_normalize_command`).
//! - Lint and type-check gates scope findings to the lines a diff touched
//!   (`_get_diff_line_ranges` + `_filter_*_by_line_ranges`); the filtering here
//!   is generic — any `path:line` finding — because the per-tool JSON parsers
//!   belong to tools this port does not ship.
//! - Failure output is trimmed to 4000 characters, keeping the head plus the
//!   pytest failure sections Python's `_sanitize_failure_output` preserves.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize};
use tracing::warn;

use openjiuwen_common::error::{Error, Result};

use super::command::{CommandOutcome, CommandRunner, CommandSpec};

/// Marker a gate script can print to fail the gate regardless of its exit code.
///
/// Python uses the same literal (`__CI_CHECK_FAIL__`) so a composite Make target
/// that swallows a failing sub-command still fails the gate.
pub const FAIL_MARKER: &str = "__CI_CHECK_FAIL__";

/// Maximum number of characters kept from a failing gate's output.
pub const OUTPUT_LIMIT: usize = 4000;

/// Marker inserted where `sanitize_failure_output` drops output.
const TRUNCATION_MARKER: &str = "\n…[truncated]…\n";

/// Upper bound of the synthetic range used for "whole file changed".
const WHOLE_FILE_LINE: u32 = 1_000_000;

/// The gate definitions shipped with the Python reference, in this crate's shape.
const DEFAULT_GATES_YAML: &str = r#"# CI gate definitions, mirroring the `ci_gates` sequence of Python's
# `harness_rsi/auto_harness/resources/ci_gate.yaml` (v0.1.20).
ci_gates:
  - name: lint
    kind: lint
    command: "make check COMMITS=1"
    commits: 1
  - name: type-check
    kind: type-check
    command: "make type-check COMMITS=1"
    commits: 1
  - name: test
    kind: command
    command: "make test TESTFLAGS=-q"
"#;

/// Which check a gate performs.
///
/// Python dispatches on the gate *name* (`lint`/`check` and `type-check` get
/// line-range filtering, everything else runs as a plain shell command). The
/// Rust port keeps an explicit kind, inferring it from the name when the YAML
/// omits it, and treats an unknown kind as [`GateKind::Command`] rather than
/// refusing the gate list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GateKind {
    /// Linting: findings are scoped to changed lines when a scope is set.
    Lint,
    /// Static type checking: findings are scoped like [`GateKind::Lint`].
    TypeCheck,
    /// Any other shell gate (`test`, `build`, …).
    #[default]
    Command,
}

impl GateKind {
    /// The `snake_case` token used in YAML and serde output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lint => "lint",
            Self::TypeCheck => "type_check",
            Self::Command => "command",
        }
    }

    /// Parses a kind token, accepting the spellings Python's names use.
    ///
    /// Unknown tokens — including `"check"` aliases and vendor names — map to
    /// [`GateKind::Command`] instead of failing, matching the fail-open gate
    /// selection the YAML format relies on.
    #[must_use]
    pub fn from_token(token: &str) -> Self {
        match token.trim().to_ascii_lowercase().as_str() {
            "lint" | "check" => Self::Lint,
            "type-check" | "type_check" | "typecheck" | "mypy" => Self::TypeCheck,
            _ => Self::Command,
        }
    }
}

impl<'de> Deserialize<'de> for GateKind {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_yaml_ng::Value::deserialize(deserializer)?;
        Ok(value.as_str().map_or(Self::Command, Self::from_token))
    }
}

/// One configured CI gate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CiGateSpec {
    /// Gate name, used for selection and error reporting (`lint`, `test`, …).
    pub name: String,
    /// What the gate checks; inferred from `name` when the YAML omits it.
    pub kind: GateKind,
    /// Shell command line executed for the gate.
    pub command: String,
    /// Number of commits the diff scope covers; `0` means "no scope".
    pub commits: u32,
}

/// The result of one gate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GateOutcome {
    /// Gate name.
    pub name: String,
    /// Whether the gate passed.
    pub passed: bool,
    /// Sanitised output (at most [`OUTPUT_LIMIT`] characters).
    pub output: String,
}

/// The result of running one or more gates.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CiGateReport {
    /// Whether every selected gate passed.
    pub passed: bool,
    /// One outcome per selected gate, in configuration order.
    pub gates: Vec<GateOutcome>,
    /// Failed gate names with their output, joined by a blank line.
    pub errors: String,
}

/// Runs configured CI gates through a [`CommandRunner`].
///
/// Every command runs in `workspace` with a merged output stream, so a gate's
/// findings are one string in the order the tool printed them.
#[derive(Clone)]
pub struct CiGateRunner {
    runner: Arc<dyn CommandRunner>,
    workspace: PathBuf,
    python: Option<String>,
    gates: Vec<CiGateSpec>,
    timeout: Option<Duration>,
}

impl CiGateRunner {
    /// Creates a runner over `workspace` with the bundled gate definitions.
    ///
    /// Loading the bundled list matches Python's constructor, which reads
    /// `ci_gate.yaml` by default; use [`CiGateRunner::with_gates`] to replace it.
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>, workspace: impl Into<PathBuf>) -> Self {
        Self {
            runner,
            workspace: workspace.into(),
            python: None,
            gates: parse_gates(default_gates_yaml()),
            timeout: None,
        }
    }

    /// Replaces the configured gates.
    #[must_use]
    pub fn with_gates(mut self, gates: Vec<CiGateSpec>) -> Self {
        self.gates = gates;
        self
    }

    /// Sets the Python interpreter used for command normalisation.
    #[must_use]
    pub fn with_python(mut self, python: impl Into<String>) -> Self {
        self.python = Some(python.into());
        self
    }

    /// Sets the per-gate deadline (the runner's own default is used otherwise).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Returns the configured gates.
    #[must_use]
    pub fn gates(&self) -> &[CiGateSpec] {
        &self.gates
    }

    /// Returns the workspace gates run in.
    #[must_use]
    pub fn workspace(&self) -> &std::path::Path {
        &self.workspace
    }

    /// Returns the configured Python interpreter, if any.
    #[must_use]
    pub fn python(&self) -> Option<&str> {
        self.python.as_deref()
    }

    /// Returns the per-gate deadline, if any.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Runs the gates selected by `action`.
    ///
    /// `"all"` runs every configured gate, `"check"` is an alias for `"lint"`
    /// (Python's `_match_gates`), and any other value selects the gate whose
    /// name matches exactly. Running an empty configuration yields a failed
    /// report rather than a vacuous pass, so a missing gate list can never be
    /// mistaken for a green build.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `action` names no configured gate, and
    /// propagates [`CommandRunner::run`] errors (a missing shell executable, for
    /// example). A non-zero gate exit is data, not an error.
    pub async fn run(&self, action: &str) -> Result<CiGateReport> {
        let action = action.trim();
        let selected = if action == "all" {
            self.gates.iter().collect::<Vec<_>>()
        } else {
            let target = if action == "check" { "lint" } else { action };
            let matched = self
                .gates
                .iter()
                .filter(|gate| gate.name == target)
                .collect::<Vec<_>>();
            if matched.is_empty() {
                return Err(Error::validation(format!(
                    "no CI gate matches action `{action}`"
                )));
            }
            matched
        };
        if selected.is_empty() {
            return Ok(CiGateReport {
                passed: false,
                gates: Vec::new(),
                errors: "no CI gates configured".to_owned(),
            });
        }
        let make_available = which::which("make").is_ok();
        let mut gates = Vec::with_capacity(selected.len());
        for spec in selected {
            gates.push(self.run_gate(spec, make_available).await?);
        }
        let passed = gates.iter().all(|gate| gate.passed);
        let errors = gates
            .iter()
            .filter(|gate| !gate.passed && !gate.output.trim().is_empty())
            .map(|gate| format!("[{}]\n{}", gate.name, gate.output.trim()))
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok(CiGateReport {
            passed,
            gates,
            errors,
        })
    }

    /// Runs one gate, applying the diff scope for lint/type-check gates.
    async fn run_gate(&self, spec: &CiGateSpec, make_available: bool) -> Result<GateOutcome> {
        let normalized = normalize_command(&spec.command, self.python.as_deref(), make_available);
        let commits = commits_scope(spec);
        let ranges = if commits > 0 && matches!(spec.kind, GateKind::Lint | GateKind::TypeCheck) {
            self.changed_ranges(commits).await
        } else {
            None
        };
        let mut command = shell_command(&normalized).cwd(self.workspace.clone());
        if let Some(python) = &self.python {
            command = command.env("AUTO_HARNESS_PYTHON", python.clone());
        }
        if let Some(timeout) = self.timeout {
            command = command.timeout(timeout);
        }
        let outcome = self.runner.run(command).await?;
        Ok(gate_outcome(&spec.name, spec, &outcome, ranges.as_ref()))
    }

    /// Returns the changed-line scope of the last `commits` commits.
    ///
    /// Python falls back to the worktree diff when the committed range is empty
    /// (the agent's changes are usually uncommitted). Any failure — a git error,
    /// an unparseable diff — yields `None`, and the caller then reports the
    /// gate's unfiltered output.
    async fn changed_ranges(&self, commits: u32) -> Option<BTreeMap<String, Vec<(u32, u32)>>> {
        let range = format!("HEAD~{commits}..");
        if let Some(ranges) = self
            .diff_ranges(&["-U0", &range, "--diff-filter=ACMR"])
            .await
        {
            return Some(ranges);
        }
        self.diff_ranges(&["-U0", "HEAD", "--diff-filter=ACMR"])
            .await
    }

    /// Runs one `git diff -U0` invocation and parses its ranges.
    async fn diff_ranges(&self, args: &[&str]) -> Option<BTreeMap<String, Vec<(u32, u32)>>> {
        let spec = CommandSpec::new("git")
            .args(args.iter().copied())
            .cwd(self.workspace.clone())
            .merge_stderr();
        let outcome = match self.runner.run(spec).await {
            Ok(outcome) => outcome,
            Err(error) => {
                warn!("CI gate: could not run `git diff`: {error}");
                return None;
            }
        };
        if !outcome.success() {
            warn!("CI gate: `git diff` failed: {}", outcome.combined_output());
            return None;
        }
        let ranges = changed_line_ranges(outcome.combined_output());
        if ranges.is_empty() {
            None
        } else {
            Some(ranges)
        }
    }
}

/// Returns the gate definitions bundled with the crate.
///
/// The shape mirrors Python's `resources/ci_gate.yaml`: a `ci_gates` sequence of
/// `{name, kind, command, commits}` entries covering lint, type checking, and
/// the test suite. The Python file also carries rails, fix-loop, and budget
/// settings; those belong to [`crate::harness_rsi::auto_harness::AutoHarnessConfig`]
/// and are not part of the gate list.
#[must_use]
pub fn default_gates_yaml() -> &'static str {
    DEFAULT_GATES_YAML
}

/// Parses gate definitions from YAML.
///
/// Reads the `gates` sequence (or Python's `ci_gates` key) and returns one
/// [`CiGateSpec`] per entry. A parse failure, a missing key, or an entry
/// without both `name` and `command` yields no gate — Python's `_load_gates`
/// behaves the same way, and a skipped entry must not silently pass.
///
/// Missing `kind` is inferred from the name (`lint`/`check` → lint,
/// `type-check` → type check, anything else → command); missing `commits` is
/// derived from a `COMMITS=<n>` argument in the command when present.
#[must_use]
pub fn parse_gates(yaml: &str) -> Vec<CiGateSpec> {
    let Ok(value) = serde_yaml_ng::from_str::<serde_yaml_ng::Value>(yaml) else {
        warn!("CI gate config is not valid YAML; no gates configured");
        return Vec::new();
    };
    let Some(mapping) = value.as_mapping() else {
        return Vec::new();
    };
    let sequence = ["gates", "ci_gates"]
        .iter()
        .find_map(|key| mapping.get(serde_yaml_ng::Value::String((*key).to_owned())))
        .and_then(serde_yaml_ng::Value::as_sequence);
    let Some(sequence) = sequence else {
        return Vec::new();
    };
    let mut gates = Vec::with_capacity(sequence.len());
    for item in sequence {
        let Some(entry) = item.as_mapping() else {
            warn!("CI gate entry is not a mapping; skipped");
            continue;
        };
        let name = string_field(entry, "name").unwrap_or_default();
        let command = string_field(entry, "command").unwrap_or_default();
        if name.trim().is_empty() || command.trim().is_empty() {
            warn!("CI gate entry without both `name` and `command`; skipped");
            continue;
        }
        let kind = string_field(entry, "kind").map_or_else(
            || GateKind::from_token(&name),
            |token| GateKind::from_token(&token),
        );
        let commits = entry
            .get(serde_yaml_ng::Value::String("commits".to_owned()))
            .and_then(|value| {
                value
                    .as_u64()
                    .map_or_else(|| value.as_str()?.trim().parse::<u64>().ok(), Some)
            })
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or_else(|| extract_commits(&command));
        gates.push(CiGateSpec {
            name,
            kind,
            command,
            commits,
        });
    }
    gates
}

/// Reads one string field from a YAML mapping.
fn string_field(mapping: &serde_yaml_ng::Mapping, key: &str) -> Option<String> {
    mapping
        .get(serde_yaml_ng::Value::String(key.to_owned()))
        .and_then(serde_yaml_ng::Value::as_str)
        .map(str::to_owned)
}

/// Parses a `COMMITS=<n>` assignment from a gate command.
///
/// Python's `_extract_commits` regex, kept public because it decides the diff
/// scope of a lint gate; `0` means "no scope configured".
#[must_use]
pub fn extract_commits(command: &str) -> u32 {
    let Some(index) = command.find("COMMITS=") else {
        return 0;
    };
    let digits = command[index + "COMMITS=".len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse().unwrap_or(0)
}

/// The effective diff scope of a gate.
fn commits_scope(spec: &CiGateSpec) -> u32 {
    if spec.commits > 0 {
        spec.commits
    } else {
        extract_commits(&spec.command)
    }
}

/// The shell invocation for one gate command.
///
/// On Unix the gate runs under `sh -c`, on Windows under `cmd.exe /c`; standard
/// error is merged into standard output, `CI=1` tells tools they are running in
/// automation, and `VIRTUAL_ENV` is removed so a host virtualenv never becomes
/// the target (Python's `_command_env`).
#[must_use]
pub fn shell_command(command: &str) -> CommandSpec {
    #[cfg(windows)]
    let spec = CommandSpec::new("cmd.exe").args(["/c", command]);
    #[cfg(not(windows))]
    let spec = CommandSpec::new("sh").args(["-c", command]);
    spec.env("CI", "1").env_remove("VIRTUAL_ENV").merge_stderr()
}

/// Rewrites the gate commands Python cannot run verbatim.
///
/// Two rewrites, both from Python's `_normalize_command`:
///
/// - a leading `python -m …` becomes `<python> -m …`, so the configured
///   interpreter (usually the workspace virtualenv) wins over `PATH`;
/// - when `make` is not on `PATH`, `make test TESTFLAGS=<x>` becomes
///   `<python> -m pytest <x>`. Other `make` targets are returned unchanged —
///   Python does the same, on the grounds that a surprising rewrite is worse
///   than a visible failure.
///
/// When no interpreter is configured the bare name `python` is used. Commands
/// that match neither rule are returned byte-for-byte unchanged.
#[must_use]
pub fn normalize_command(command: &str, python: Option<&str>, make_available: bool) -> String {
    let stripped = command.trim();
    let python_exe = quote_path(python.unwrap_or("python"));
    if !stripped.starts_with("make ") {
        if stripped.starts_with("python -m ") {
            return stripped.replacen("python -m ", &format!("{python_exe} -m "), 1);
        }
        return command.to_owned();
    }
    if make_available {
        return command.to_owned();
    }
    let words = split_words(stripped);
    if words.first().map(String::as_str) != Some("make") {
        return command.to_owned();
    }
    let target = words.get(1).map(String::as_str).unwrap_or_default();
    if target != "test" {
        return command.to_owned();
    }
    let testflags = words[2..]
        .iter()
        .find_map(|word| word.strip_prefix("TESTFLAGS="))
        .map_or("", str::trim);
    format!("{python_exe} -m pytest {testflags}")
        .trim_end()
        .to_owned()
}

/// Quotes a path for a shell command line when it needs quoting.
fn quote_path(path: &str) -> String {
    let plain = !path.is_empty()
        && path
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "./_:\\-@%+,".contains(ch));
    if plain {
        path.to_owned()
    } else {
        format!("'{}'", path.replace('\'', "'\\''"))
    }
}

/// Splits a command line into words, honouring single and double quotes.
fn split_words(command: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut started = false;
    for ch in command.chars() {
        match quote {
            Some(active) => {
                if ch == active {
                    quote = None;
                } else {
                    current.push(ch);
                }
            }
            None => match ch {
                '\'' | '"' => {
                    quote = Some(ch);
                    started = true;
                }
                _ if ch.is_whitespace() => {
                    if started {
                        words.push(std::mem::take(&mut current));
                        started = false;
                    }
                }
                _ => {
                    current.push(ch);
                    started = true;
                }
            },
        }
    }
    if started {
        words.push(current);
    }
    words
}

/// Parses the changed new-side line ranges out of a `git diff -U0` document.
///
/// The result maps each repo-relative path to inclusive `(first, last)` ranges of
/// lines that exist in the working tree after the change — the lines an agent is
/// responsible for. Rules:
///
/// - a `+++ b/path` header selects the file; `+++ /dev/null` (a deletion) and
///   `--- a/path` do not;
/// - a hunk header `@@ -a[,b] +c[,d] @@` contributes `(c, c + d - 1)`, with a
///   missing count meaning one line and a zero count (a pure deletion) adding
///   nothing;
/// - a line starting with `@@` that is not a valid hunk header marks the whole
///   file as changed (`(1, 1_000_000)`), matching Python's conservative
///   fallback;
/// - adjacent and overlapping ranges are merged so the output is canonical.
#[must_use]
pub fn changed_line_ranges(diff: &str) -> BTreeMap<String, Vec<(u32, u32)>> {
    let mut ranges: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in diff.lines() {
        if let Some(raw_path) = line.strip_prefix("+++ ") {
            current = clean_diff_path(raw_path);
            if let Some(path) = &current {
                ranges.entry(path.clone()).or_default();
            }
            continue;
        }
        if line.starts_with("--- ") {
            continue;
        }
        if !line.starts_with("@@") {
            continue;
        }
        let Some(path) = &current else {
            continue;
        };
        let Some(entry) = ranges.get_mut(path) else {
            continue;
        };
        match parse_hunk_header(line) {
            Some((start, count)) => {
                if count > 0 {
                    let last = start.saturating_add(count - 1).min(WHOLE_FILE_LINE);
                    entry.push((start.min(WHOLE_FILE_LINE), last));
                }
            }
            None => entry.push((1, WHOLE_FILE_LINE)),
        }
    }
    for entry in ranges.values_mut() {
        merge_ranges(entry);
    }
    ranges
}

/// Normalises one diff path, dropping the deletion sentinel.
fn clean_diff_path(raw: &str) -> Option<String> {
    let raw = raw.split('\t').next().unwrap_or(raw).trim();
    if raw.is_empty() || raw == "/dev/null" {
        return None;
    }
    let stripped = raw
        .strip_prefix("b/")
        .or_else(|| raw.strip_prefix("a/"))
        .unwrap_or(raw);
    Some(normalize_path(stripped))
}

/// Normalises a path for comparison: forward slashes, no `./` prefix.
fn normalize_path(path: &str) -> String {
    let path = path.trim().replace('\\', "/");
    let path = path.trim_start_matches("./");
    path.trim_start_matches('/').to_owned()
}

/// Parses `@@ -a[,b] +c[,d] @@` into `(c, d)`.
fn parse_hunk_header(line: &str) -> Option<(u32, u32)> {
    let rest = line.strip_prefix("@@")?.trim_start();
    let plus = rest.find('+')?;
    let old_part = rest[..plus].trim().strip_prefix('-')?;
    parse_range(old_part)?;
    let new_part = &rest[plus + 1..];
    let end = new_part.find("@@")?;
    parse_range(new_part[..end].trim())
}

/// Parses `<start>[,<count>]`, defaulting a missing count to one.
fn parse_range(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.splitn(2, ',');
    let start = parts.next()?.trim().parse::<u32>().ok()?;
    let count = parts
        .next()
        .map_or(Some(1), |count| count.trim().parse::<u32>().ok())?;
    Some((start, count))
}

/// Sorts and merges overlapping or adjacent inclusive ranges.
fn merge_ranges(ranges: &mut Vec<(u32, u32)>) {
    ranges.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges.iter().copied() {
        match merged.last_mut() {
            Some((_, last)) if start <= last.saturating_add(1) => {
                *last = (*last).max(end);
            }
            _ => merged.push((start, end)),
        }
    }
    *ranges = merged;
}

/// Keeps only the findings whose `path:line` falls inside the changed ranges.
///
/// A finding is any line carrying a `path:line` location — ruff, mypy, pylint,
/// codespell, and compiler output all share that shape. Paths are normalised
/// (`\` → `/`, leading `./` and `/` removed) before matching, lines for files
/// absent from `ranges` are dropped, and the surviving lines keep their original
/// order.
#[must_use]
pub fn filter_findings(output: &str, ranges: &BTreeMap<String, Vec<(u32, u32)>>) -> String {
    output
        .lines()
        .filter(|line| {
            let Some((path, number)) = finding_location(line) else {
                return false;
            };
            ranges.get(&path).is_some_and(|entries| {
                entries
                    .iter()
                    .any(|(start, end)| number >= *start && number <= *end)
            })
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extracts `(normalised path, line number)` from a finding line.
fn finding_location(line: &str) -> Option<(String, u32)> {
    for (index, _) in line.match_indices(':') {
        let rest = &line[index + 1..];
        let digits = rest
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>();
        if digits.is_empty() {
            continue;
        }
        let after = rest[digits.len()..].chars().next();
        if after.is_some_and(|ch| ch != ':' && !ch.is_whitespace()) {
            continue;
        }
        let Ok(number) = digits.parse::<u32>() else {
            continue;
        };
        let path = line[..index].trim();
        if path.is_empty() {
            continue;
        }
        return Some((normalize_path(path), number));
    }
    None
}

/// Trims failure output to `limit` characters, keeping the useful parts.
///
/// Python's `_sanitize_failure_output` keeps only pytest's `FAILURES` and
/// `short test summary info` sections and drops everything else. That loses a
/// tool's own error line when pytest never started, so this port keeps both
/// ends: the head of the output (the command, the collection error) and, when a
/// pytest failure section is present, the tail starting there. Output that
/// already fits is returned unchanged; empty output becomes an empty string.
#[must_use]
pub fn sanitize_failure_output(text: &str, limit: usize) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    let total = text.chars().count();
    if total <= limit {
        return text.to_owned();
    }
    if limit == 0 {
        return String::new();
    }
    let head: String = text.chars().take(limit / 2).collect();
    let budget = limit
        .saturating_sub(head.chars().count())
        .saturating_sub(TRUNCATION_MARKER.chars().count());
    let tail: String = failure_section_start(text).map_or_else(
        || text.chars().skip(total.saturating_sub(budget)).collect(),
        |start| text.chars().skip(start).take(budget).collect(),
    );
    format!("{head}{TRUNCATION_MARKER}{tail}")
}

/// Byte offset of pytest's first failure section, when one exists.
fn failure_section_start(text: &str) -> Option<usize> {
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        if is_failure_header(line.trim()) {
            return Some(offset);
        }
        offset += line.chars().count();
    }
    None
}

/// Whether a line is a pytest `=…= FAILURES =…=` style section header.
fn is_failure_header(line: &str) -> bool {
    if line.len() < 3 || !line.starts_with('=') || !line.ends_with('=') {
        return false;
    }
    let inner = line.trim_matches('=').trim().to_ascii_lowercase();
    inner == "failures" || inner == "short test summary info"
}

/// Builds a [`GateOutcome`] from a finished command.
///
/// A gate passes when the command exited successfully **and** its output does
/// not contain [`FAIL_MARKER`]. `ranges` is the diff scope: lint and type-check
/// findings are filtered to it when present, and only for display, because the
/// exit code already decided the gate. The output is sanitised to
/// [`OUTPUT_LIMIT`] characters and the marker is removed from the text.
#[must_use]
pub fn gate_outcome(
    name: &str,
    spec: &CiGateSpec,
    outcome: &CommandOutcome,
    ranges: Option<&BTreeMap<String, Vec<(u32, u32)>>>,
) -> GateOutcome {
    let raw = outcome.combined_output();
    let scoped = match (ranges, spec.kind) {
        (Some(ranges), GateKind::Lint | GateKind::TypeCheck) => filter_findings(raw, ranges),
        _ => raw.to_owned(),
    };
    let output = sanitize_failure_output(&scoped, OUTPUT_LIMIT)
        .replace(FAIL_MARKER, "")
        .trim()
        .to_owned();
    GateOutcome {
        name: name.to_owned(),
        passed: outcome.success() && !raw.contains(FAIL_MARKER),
        output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_words_honours_quotes() {
        assert_eq!(
            split_words("make test TESTFLAGS=\"-q tests/x.py -k foo\""),
            vec![
                "make".to_owned(),
                "test".to_owned(),
                "TESTFLAGS=-q tests/x.py -k foo".to_owned()
            ]
        );
        assert_eq!(split_words("  a  b "), vec!["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn quote_path_quotes_only_when_needed() {
        assert_eq!(quote_path("/usr/bin/python3"), "/usr/bin/python3");
        assert_eq!(quote_path("/tmp/a b/python"), "'/tmp/a b/python'");
    }

    #[test]
    fn malformed_range_is_rejected() {
        assert_eq!(parse_range("12"), Some((12, 1)));
        assert_eq!(parse_range("12,0"), Some((12, 0)));
        assert_eq!(parse_range("x,2"), None);
        assert_eq!(parse_range("1,y"), None);
    }
}
