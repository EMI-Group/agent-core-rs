//! Reward checks: the check model and the offline `read` / `python` tools.
//!
//! Ports the offline half of Python `openjiuwen.agent_evolving.reward.checks`: the
//! check vocabulary the evidence agent emits, the `read` tool (folder listings,
//! line ranges, a case-insensitive pattern search, picture headers and a page
//! refusal) and the `python` tool (a script run in a scratch directory with a 60 s
//! limit, a stripped proxy environment and a bounded output).
//!
//! Two `read` paths differ from Python by construction, and both say so in their
//! result instead of degrading silently:
//!
//! - **A page's source is never read.** Python refuses to judge a page from its
//!   source at all — it reads a page only through a headless browser and raises
//!   when playwright is missing. This port bundles no browser, so a `.html`/`.htm`
//!   read is [`CheckOutcome::skipped`] with Python's "the source of a page is not
//!   read" message plus the reason (no playwright), which means a page cannot be
//!   scored from its source here either. `.js`/`.mjs`/`.css` are still readable as
//!   source, exactly because Python reads them as source when no browser is
//!   available.
//! - **A picture is reported by its header, not its pixels.** Python opens the
//!   file with Pillow and attaches it to the Judge's message as a model image
//!   part. This port bundles neither Pillow nor an image decoder, so a picture
//!   read returns its [header](crate::reward::picture::PictureInfo) — format,
//!   pixel size and byte count — and states in the result that the Judge sees the
//!   metadata, not the picture.
//!
//! `browser` checks stay deferred: they need playwright, and they return
//! [`Error::Unsupported`] naming it.
//!
//! Every side effect of the `python` tool goes through
//! [`CommandRunner`](crate::evaluator_pipeline::CommandRunner), so the check can
//! be tested offline with a scripted runner.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::evaluator_pipeline::{find_program, CommandRunner, CommandSpec};
use crate::reward::picture::{PictureInfo, UNREADABLE_PICTURE};

/// The characters kept from one check's output (Python `CHECK_OUTPUT`).
pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 6000;

/// The most pattern matches one `read` check reports (Python's `[:60]`).
pub const DEFAULT_MAX_HITS: usize = 60;

/// The most entries a folder listing shows (Python's `[:200]`).
pub const DEFAULT_LISTING_LIMIT: usize = 200;

/// The default number of lines a range `read` returns (Python's `count` default).
pub const DEFAULT_RANGE_COUNT: usize = 200;

/// The largest number of lines a range `read` returns (Python clamps to 400).
pub const MAX_RANGE_COUNT: usize = 400;

/// The wall-clock budget of one `python` check in milliseconds (Python `PYTHON_SECONDS`).
pub const PYTHON_TIMEOUT_MS: u64 = 60_000;

/// The proxy every `python` check is pointed at, so a check cannot reach the network.
pub const PROXY_SINK: &str = "http://127.0.0.1:9";

/// The source file suffixes a browser check runs instead of reads.
pub const PAGE_SUFFIXES: [&str; 5] = ["html", "htm", "js", "mjs", "css"];

/// The page suffixes whose source is refused outright: a page is judged by using
/// it in a browser, never by reading its source (Python's `PAGE_CODE` gate).
///
/// A script or a stylesheet (`.js`, `.mjs`, `.css`) is still read as source, which
/// is what Python does when no browser is available.
pub const PAGE_SOURCE_SUFFIXES: [&str; 2] = ["html", "htm"];

/// The picture suffixes read as a picture header instead of as text.
pub const IMAGE_SUFFIXES: [&str; 6] = ["png", "jpg", "jpeg", "webp", "gif", "bmp"];

/// The reason a `.html`/`.htm` read is skipped: Python's own wording (its `read`
/// tool's "source of a page is not read" skip) plus this port's reason for
/// refusing a page outright.
pub const PAGE_SOURCE_REFUSED: &str = "the source of a page is not read: use the page in the browser; find the names a probe needs with a probe (Object.keys(window), the page's buttons); this port has no browser (playwright), so a page cannot be judged here and cannot be scored from its source";

/// The field-to-tool rule (Python `TOOL_BY_FIELD`): a check that names only a
/// payload field still picks the tool that field belongs to.
const TOOL_BY_FIELD: [(&str, CheckTool); 4] = [
    ("actions", CheckTool::Browser),
    ("page", CheckTool::Browser),
    ("code", CheckTool::Python),
    ("path", CheckTool::Read),
];

/// A safety cap on how many files one listing walks before it stops collecting.
const LISTING_WALK_LIMIT: usize = 10_000;

/// Counter that keeps concurrent scratch directories apart.
///
/// It is the one piece of process-wide state in this module: a monotonic number
/// combined with the process id, so two runs never share a scratch directory. It
/// is never read back, so it is not observable state.
static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The tool a [`RewardCheck`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckTool {
    /// Read a workspace file or list a folder.
    Read,
    /// Run a Python script in a scratch directory.
    Python,
    /// Drive a page in a headless browser (deferred in this port).
    Browser,
}

impl CheckTool {
    /// The Python spelling of the tool, as it travels in check JSON.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Python => "python",
            Self::Browser => "browser",
        }
    }
}

/// One check the evidence agent asks for: a read, a Python script or a browser run.
///
/// Only the fields the offline tools use are modelled; a field the model leaves
/// out stays `None` or empty, and unknown fields are ignored. `tool` may be
/// absent: [`infer_tool`](RewardCheck::infer_tool) then derives it from the
/// payload, exactly as Python's `with_tool` does.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RewardCheck {
    /// The tool; inferred from the payload when absent.
    pub tool: Option<CheckTool>,
    /// `read`: the workspace-relative file or folder to inspect.
    pub path: Option<String>,
    /// `python`: the script to run.
    pub code: Option<String>,
    /// `read`: a case-insensitive pattern to search for.
    pub pattern: Option<String>,
    /// `read` without a pattern: how many lines to show from the first line.
    pub count: Option<usize>,
    /// `browser`: the page to open.
    pub page: Option<String>,
    /// `browser`: the actions to perform.
    pub actions: Vec<Value>,
    /// The ids of the criteria this check serves.
    pub criteria: Vec<String>,
    /// A note the model writes for the judge ("controlled setup: ...").
    pub note: Option<String>,
}

impl RewardCheck {
    /// Creates an empty check; [`with_tool`](RewardCheck::with_tool) and friends
    /// fill it in.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the tool explicitly.
    #[must_use]
    pub fn with_tool(mut self, tool: CheckTool) -> Self {
        self.tool = Some(tool);
        self
    }

    /// Sets the workspace-relative path of a `read` check.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Sets the script of a `python` check.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Sets the pattern of a `read` check.
    #[must_use]
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

    /// Sets how many lines a range `read` shows.
    #[must_use]
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = Some(count);
        self
    }

    /// Sets the page of a `browser` check.
    #[must_use]
    pub fn with_page(mut self, page: impl Into<String>) -> Self {
        self.page = Some(page.into());
        self
    }

    /// Adds one browser action.
    #[must_use]
    pub fn with_action(mut self, action: Value) -> Self {
        self.actions.push(action);
        self
    }

    /// Sets the criteria ids this check serves.
    #[must_use]
    pub fn with_criteria<I, S>(mut self, criteria: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.criteria = criteria.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the judge-facing note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// The tool this check runs: the explicit one, else the first field of
    /// `TOOL_BY_FIELD` the check actually carries.
    ///
    /// A field counts as carried when it is a non-empty string (or, for
    /// `actions`, a non-empty list) — Python's truthiness test.
    #[must_use]
    pub fn infer_tool(&self) -> Option<CheckTool> {
        self.tool.or_else(|| {
            TOOL_BY_FIELD
                .iter()
                .find_map(|(field, tool)| self.carries(field).then_some(*tool))
        })
    }

    /// `true` when the field named by the `TOOL_BY_FIELD` rule is present and
    /// non-empty.
    fn carries(&self, field: &str) -> bool {
        match field {
            "actions" => !self.actions.is_empty(),
            "page" => self.page.as_deref().is_some_and(|value| !value.is_empty()),
            "code" => self.code.as_deref().is_some_and(|value| !value.is_empty()),
            "path" => self.path.as_deref().is_some_and(|value| !value.is_empty()),
            _ => false,
        }
    }

    /// What makes two checks the same, so a repeat is not run again.
    ///
    /// The key is canonical JSON of the check without the bookkeeping fields
    /// (`criteria` and `note`), with the tool filled in by
    /// [`infer_tool`](RewardCheck::infer_tool): Python's `check_key` after
    /// `with_tool`. Object keys are sorted at every depth, so the same check
    /// always hashes to the same string.
    #[must_use]
    pub fn dedupe_key(&self) -> String {
        let mut map: BTreeMap<String, Value> = BTreeMap::new();
        if let Some(tool) = self.infer_tool() {
            map.insert("tool".to_owned(), Value::String(tool.as_str().to_owned()));
        }
        for (field, value) in [("path", &self.path), ("code", &self.code)] {
            if let Some(value) = value.as_deref().filter(|text| !text.is_empty()) {
                map.insert(field.to_owned(), Value::String(value.to_owned()));
            }
        }
        if let Some(pattern) = self.pattern.as_deref().filter(|text| !text.is_empty()) {
            map.insert("pattern".to_owned(), Value::String(pattern.to_owned()));
        }
        if let Some(count) = self.count {
            map.insert("count".to_owned(), Value::from(count));
        }
        if let Some(page) = self.page.as_deref().filter(|text| !text.is_empty()) {
            map.insert("page".to_owned(), Value::String(page.to_owned()));
        }
        if !self.actions.is_empty() {
            map.insert(
                "actions".to_owned(),
                canonical(&Value::Array(self.actions.clone())),
            );
        }
        let object = map.into_iter().collect();
        Value::Object(object).to_string()
    }
}

/// Sorts object keys at every depth so a canonical rendering is stable.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> = map
                .iter()
                .map(|(key, item)| (key.clone(), canonical(item)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        other => other.clone(),
    }
}

/// The outcome of one check: what it showed, or why it could not run.
///
/// A skipped check (an unavailable tool, a read without a path) carries the
/// reason in [`skipped`](CheckOutcome::skipped) and is never evidence for a
/// claim; a check that ran and failed is evidence that the work failed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CheckOutcome {
    /// `true` when the check ran and exited cleanly.
    pub ok: bool,
    /// The rendered result: paths, line counts, matching lines, process output.
    pub result: String,
    /// Labels of the screenshots a browser check took (always empty here).
    pub screenshots: Vec<String>,
    /// Why the check was not run; `None` when it ran.
    pub skipped: Option<String>,
    /// Why the check failed; `None` when it succeeded or was skipped.
    pub error: Option<String>,
}

impl CheckOutcome {
    /// A check that ran and produced `result`.
    #[must_use]
    pub fn ok(result: impl Into<String>) -> Self {
        Self {
            ok: true,
            result: result.into(),
            ..Self::default()
        }
    }

    /// A check that was not run, for the reason given.
    #[must_use]
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self {
            skipped: Some(reason.into()),
            ..Self::default()
        }
    }

    /// A check that ran and failed, for the reason given.
    #[must_use]
    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            error: Some(error.into()),
            ..Self::default()
        }
    }

    /// `true` only for a check that ran and succeeded.
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.ok && self.skipped.is_none() && self.error.is_none()
    }
}

/// Runs [`RewardCheck`]s inside one workspace, with bounded output.
///
/// The runner owns the workspace root every path is resolved against and the
/// command seam the `python` tool runs through. Limits are fields so a consumer
/// can tighten them; the defaults are Python's (`6000` output characters, `60`
/// pattern hits, `200` folder entries).
#[derive(Clone)]
pub struct CheckRunner {
    root: PathBuf,
    runner: Arc<dyn CommandRunner>,
    max_output_chars: usize,
    max_hits: usize,
    listing_limit: usize,
}

impl std::fmt::Debug for CheckRunner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckRunner")
            .field("root", &self.root)
            .field("runner", &self.runner)
            .field("max_output_chars", &self.max_output_chars)
            .field("max_hits", &self.max_hits)
            .field("listing_limit", &self.listing_limit)
            .finish()
    }
}

impl CheckRunner {
    /// Creates a runner over `root`, using `runner` for every external command.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            root: root.into(),
            runner,
            max_output_chars: DEFAULT_MAX_OUTPUT_CHARS,
            max_hits: DEFAULT_MAX_HITS,
            listing_limit: DEFAULT_LISTING_LIMIT,
        }
    }

    /// Sets the characters kept from one result; `0` disables truncation.
    #[must_use]
    pub fn with_max_output_chars(mut self, max_output_chars: usize) -> Self {
        self.max_output_chars = max_output_chars;
        self
    }

    /// Sets how many pattern hits a `read` check reports.
    #[must_use]
    pub fn with_max_hits(mut self, max_hits: usize) -> Self {
        self.max_hits = max_hits.max(1);
        self
    }

    /// Sets how many entries a folder listing shows.
    #[must_use]
    pub fn with_listing_limit(mut self, listing_limit: usize) -> Self {
        self.listing_limit = listing_limit.max(1);
        self
    }

    /// The workspace every path is resolved against.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Runs one check.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a deferred tool (`browser`), [`Error::validation`]
    /// for a path that leaves the workspace, and the runner's own errors. A check
    /// that names no tool, a `read` without a path, an unreadable picture, a page
    /// whose source is refused and an unavailable Python interpreter are reported in
    /// the outcome instead, because the evidence agent can act on them.
    ///
    /// A check has no cancellation token of its own: each is bounded by its own
    /// output and time limits, and the runner is shared.
    pub async fn run(&self, check: &RewardCheck) -> Result<CheckOutcome> {
        let Some(tool) = check.infer_tool() else {
            return Ok(CheckOutcome::skipped(
                "unknown tool: the check names no tool and carries no tool-specific field",
            ));
        };
        match tool {
            CheckTool::Read => self.run_read(check),
            CheckTool::Python => self.run_python(check).await,
            CheckTool::Browser => Err(Error::unsupported(
                "browser checks are deferred: they need playwright (headless browser automation), \
                 which this port does not bundle",
            )),
        }
    }

    /// Runs a `read` check: a folder listing, a pattern search or a line range.
    fn run_read(&self, check: &RewardCheck) -> Result<CheckOutcome> {
        let Some(path) = check.path.as_deref().filter(|path| !path.is_empty()) else {
            return Ok(CheckOutcome::skipped(
                "read needs a \"path\"; a saved result is read through the evidence results",
            ));
        };
        let target = inside(&self.root, path)?;
        if target.is_dir() {
            return Ok(self.folder_outcome(path, &target));
        }
        let suffix = target
            .extension()
            .map(|extension| extension.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if IMAGE_SUFFIXES.contains(&suffix.as_str()) {
            return Ok(picture_outcome(path, &target));
        }
        // Python's "use the page, do not read it" rule: a page's source is never
        // evidence of how the page looks or behaves. Python reads a page only
        // through a headless browser and fails the run when playwright is missing;
        // this port has no browser, so the read is skipped with that reason instead
        // of scoring the page from its source.
        if PAGE_SOURCE_SUFFIXES.contains(&suffix.as_str()) {
            return Ok(CheckOutcome::skipped(PAGE_SOURCE_REFUSED));
        }
        let bytes = std::fs::read(&target)
            .map_err(|error| Error::sys_op(format!("failed to read {path}: {error}")))?;
        let text = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => String::from("[binary file]"),
        };
        let lines: Vec<&str> = text.lines().collect();
        let outcome = match check
            .pattern
            .as_deref()
            .filter(|pattern| !pattern.is_empty())
        {
            Some(pattern) => self.pattern_outcome(path, &lines, pattern),
            None => self.range_outcome(path, &lines, check.count),
        };
        Ok(outcome)
    }

    /// Renders a folder as its sorted, workspace-relative file names.
    fn folder_outcome(&self, path: &str, target: &Path) -> CheckOutcome {
        let mut names = Vec::new();
        collect_files(target, target, &mut names);
        names.sort();
        names.truncate(self.listing_limit);
        let files = names.join("\n");
        CheckOutcome::ok(format!("path: {path}\nfolder; files:\n{files}"))
    }

    /// Renders the lines matching `pattern`, case-insensitively.
    ///
    /// The Python check compiles a regular expression and can show up to five
    /// context lines per hit; this reduced model carries neither a `context`
    /// field nor a regular-expression dependency, so it matches substrings and
    /// shows the matching lines only.
    fn pattern_outcome(&self, path: &str, lines: &[&str], pattern: &str) -> CheckOutcome {
        let needle = pattern.to_lowercase();
        let hits: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.to_lowercase().contains(&needle))
            .map(|(index, _)| index)
            .take(self.max_hits)
            .collect();
        let (body, shown) = whole_lines(
            hits.iter().map(|&index| (index + 1, lines[index])),
            self.max_output_chars,
        );
        let body = if body.is_empty() {
            String::from("(no line matches)")
        } else {
            body
        };
        let mut header = format!("path: {path}\nmatches: {}", hits.len());
        if shown < hits.len() {
            header.push_str("\ncut: too many matches to show; narrow the pattern");
        }
        CheckOutcome::ok(format!("{header}\n{body}"))
    }

    /// Renders a range of lines, never cutting one in half.
    fn range_outcome(&self, path: &str, lines: &[&str], count: Option<usize>) -> CheckOutcome {
        let count = count
            .unwrap_or(DEFAULT_RANGE_COUNT)
            .clamp(1, MAX_RANGE_COUNT);
        let end = count.min(lines.len());
        let (body, shown) = whole_lines(
            (1..=end).map(|number| (number, lines[number - 1])),
            self.max_output_chars,
        );
        let last = shown;
        let total = lines.len();
        CheckOutcome::ok(format!("path: {path}\nlines: 1-{last} of {total}\n{body}"))
    }

    /// Runs a `python` check in a scratch directory outside the workspace.
    async fn run_python(&self, check: &RewardCheck) -> Result<CheckOutcome> {
        let program = python_program()?;
        let code = check.code.as_deref().unwrap_or_default();
        let scratch = scratch_directory()?;
        let outcome = self.execute_python(program, code, &scratch).await;
        let _ = std::fs::remove_dir_all(&scratch);
        outcome
    }

    /// Writes `code` into `scratch` and runs it against the workspace.
    async fn execute_python(
        &self,
        program: &str,
        code: &str,
        scratch: &Path,
    ) -> Result<CheckOutcome> {
        let script = scratch.join("check.py");
        std::fs::write(&script, code).map_err(|error| {
            Error::sys_op(format!("failed to write the python check script: {error}"))
        })?;
        let temp = scratch.to_string_lossy().into_owned();
        let spec = CommandSpec::new(program)
            .arg("-I")
            .arg(script.to_string_lossy().into_owned())
            .with_cwd(&self.root)
            .with_env("HTTP_PROXY", PROXY_SINK)
            .with_env("HTTPS_PROXY", PROXY_SINK)
            .with_env("PYTHONIOENCODING", "utf-8")
            .with_env("PYTHONDONTWRITEBYTECODE", "1")
            .with_env("TMPDIR", temp.clone())
            .with_env("TEMP", temp.clone())
            .with_env("TMP", temp)
            .with_timeout_ms(PYTHON_TIMEOUT_MS);
        let cancel = CancellationToken::new();
        let result = self.runner.run(&spec, &cancel).await?;
        Ok(self.python_outcome(&result))
    }

    /// Renders the process result the way the judge reads it.
    fn python_outcome(&self, result: &crate::evaluator_pipeline::ExecResult) -> CheckOutcome {
        let stdout = clip(&result.stdout, self.max_output_chars);
        let stderr = clip(&result.stderr, self.max_output_chars / 2);
        let mut outcome = if result.timed_out {
            CheckOutcome::failed(format!("timed out after {PYTHON_TIMEOUT_MS} ms"))
        } else if result.success() {
            CheckOutcome::ok("")
        } else {
            CheckOutcome::failed(format!("exit code {}", result.returncode))
        };
        outcome.result = format!(
            "exit_code: {code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            code = if result.timed_out {
                String::from("none")
            } else {
                result.returncode.to_string()
            }
        );
        outcome
    }
}

/// Reads a picture's header as the check's result.
///
/// The file's bytes are never decoded: [`PictureInfo::parse`] reads the format's
/// own header, and a file it cannot read truthfully answers Python's "not a
/// readable picture" (an `ok` outcome, not an error), so the evidence agent
/// records what it saw instead of aborting the run.
fn picture_outcome(path: &str, target: &Path) -> CheckOutcome {
    let Ok(bytes) = std::fs::read(target) else {
        return CheckOutcome::ok(UNREADABLE_PICTURE);
    };
    match PictureInfo::parse(&bytes) {
        Some(info) => CheckOutcome::ok(info.evidence(path)),
        None => CheckOutcome::ok(UNREADABLE_PICTURE),
    }
}

/// Resolves `relative` against `root` and refuses a path that leaves it.
///
/// The check is lexical (Python's `Path.resolve` also follows symlinks; this port
/// does not, so a symlink inside the workspace can still point outside it).
///
/// # Errors
///
/// [`Error::validation`] for an empty path or one that resolves outside `root`.
pub fn inside(root: &Path, relative: &str) -> Result<PathBuf> {
    if relative.trim().is_empty() {
        return Err(Error::validation("a check path must not be empty"));
    }
    let base = lexical(root);
    let target = lexical(&root.join(relative));
    if target == base || target.starts_with(&base) {
        Ok(target)
    } else {
        Err(Error::validation(format!(
            "a check path must stay inside the workspace: {relative}"
        )))
    }
}

/// Removes `.` and applies `..` without touching the file system.
fn lexical(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

/// Collects the file paths under `dir`, relative to `base`.
fn collect_files(base: &Path, dir: &Path, names: &mut Vec<String>) {
    if names.len() >= LISTING_WALK_LIMIT {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_files(base, &path, names);
        } else if file_type.is_file() {
            if let Ok(relative) = path.strip_prefix(base) {
                names.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

/// Joins numbered lines, keeping whole lines until `limit` characters are used.
///
/// Returns the text and how many lines it actually holds, so a caller can report
/// that a result was cut.
fn whole_lines<'a>(
    numbered: impl Iterator<Item = (usize, &'a str)>,
    limit: usize,
) -> (String, usize) {
    let mut rows: Vec<String> = Vec::new();
    let mut size = 0_usize;
    for (number, line) in numbered {
        let row = format!("{number}: {line}");
        let row_chars = row.chars().count();
        if !rows.is_empty() && size + row_chars + 1 > limit {
            break;
        }
        let row: String = row.chars().take(limit).collect();
        size += row.chars().count() + 1;
        rows.push(row);
    }
    let shown = rows.len();
    (rows.join("\n"), shown)
}

/// Keeps `limit` characters of `text`, naming what was cut (Python `clip`).
pub(crate) fn clip(text: &str, limit: usize) -> String {
    let total = text.chars().count();
    if limit == 0 || total <= limit {
        return text.to_owned();
    }
    let mut clipped: String = text.chars().take(limit).collect();
    let _ = write!(clipped, "\n...[cut {} characters]", total - limit);
    clipped
}

/// The interpreter a `python` check runs, when one is on `PATH`.
fn python_program() -> Result<&'static str> {
    for program in ["python3", "python"] {
        if find_program(program).is_some() {
            return Ok(program);
        }
    }
    Err(Error::unsupported(
        "python checks need a python3 (or python) interpreter on PATH, and none was found",
    ))
}

/// Creates a unique scratch directory outside the workspace.
fn scratch_directory() -> Result<PathBuf> {
    let number = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("reward_check_{}_{number}", std::process::id()));
    std::fs::create_dir_all(&path).map_err(|error| {
        Error::sys_op(format!(
            "failed to create the python check scratch directory: {error}"
        ))
    })?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::Error;
    use serde_json::json;

    use super::*;
    use crate::evaluator_pipeline::{ExecResult, FakeCommandRunner, ProcessCommandRunner};
    use crate::reward::picture::fixtures;

    fn runner(root: &Path) -> CheckRunner {
        CheckRunner::new(root, Arc::new(FakeCommandRunner::new().record_only()))
    }

    fn write(root: &Path, name: &str, text: &str) {
        std::fs::write(root.join(name), text).unwrap();
    }

    #[test]
    fn tool_inference_follows_the_python_rule() {
        assert_eq!(RewardCheck::new().infer_tool(), None);
        assert_eq!(
            RewardCheck::new().with_path("a.txt").infer_tool(),
            Some(CheckTool::Read)
        );
        assert_eq!(
            RewardCheck::new().with_code("print(1)").infer_tool(),
            Some(CheckTool::Python)
        );
        assert_eq!(
            RewardCheck::new().with_page("index.html").infer_tool(),
            Some(CheckTool::Browser)
        );
        assert_eq!(
            RewardCheck::new()
                .with_action(json!({"type": "click"}))
                .infer_tool(),
            Some(CheckTool::Browser)
        );
        // An explicit tool wins over the payload.
        assert_eq!(
            RewardCheck::new()
                .with_tool(CheckTool::Read)
                .with_code("print(1)")
                .infer_tool(),
            Some(CheckTool::Read)
        );
        // Empty strings do not count as present.
        assert_eq!(
            RewardCheck {
                path: Some(String::new()),
                ..RewardCheck::new()
            }
            .infer_tool(),
            None
        );
    }

    #[test]
    fn dedupe_key_ignores_bookkeeping_and_sorts_actions() {
        let first = RewardCheck::new()
            .with_path("notes.txt")
            .with_criteria(["c1"])
            .with_note("first");
        let second = RewardCheck::new()
            .with_tool(CheckTool::Read)
            .with_path("notes.txt")
            .with_criteria(["c2"])
            .with_note("second");
        assert_eq!(first.dedupe_key(), second.dedupe_key());
        assert_ne!(
            first.dedupe_key(),
            RewardCheck::new().with_path("other.txt").dedupe_key()
        );

        let browser = |actions: Vec<Value>| RewardCheck {
            actions,
            ..RewardCheck::new()
        };
        let first = browser(vec![json!({"b": 1, "a": 2})]);
        let second = browser(vec![json!({"a": 2, "b": 1})]);
        assert_eq!(first.dedupe_key(), second.dedupe_key());
        assert!(first.dedupe_key().contains("\"tool\":\"browser\""));
    }

    #[test]
    fn inside_refuses_a_path_that_leaves_the_workspace() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        assert!(inside(root, "notes.txt").is_ok());
        assert!(inside(root, "./notes.txt").is_ok());
        assert!(inside(root, "sub/../notes.txt").is_ok());
        assert!(inside(root, "")
            .unwrap_err()
            .to_string()
            .contains("must not be empty"));
        let error = inside(root, "../escape.txt").unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
        assert!(inside(root, "/etc/passwd").is_err());
    }

    #[tokio::test]
    async fn read_range_shows_whole_lines_and_the_line_count() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "notes.txt", "one\ntwo\nthree\nfour\n");
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("notes.txt").with_count(2))
            .await
            .unwrap();
        assert!(outcome.ok);
        assert_eq!(
            outcome.result,
            "path: notes.txt\nlines: 1-2 of 4\n1: one\n2: two"
        );

        // `count` is clamped to 1..=400.
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("notes.txt").with_count(0))
            .await
            .unwrap();
        assert!(
            outcome.result.contains("lines: 1-1 of 4"),
            "{}",
            outcome.result
        );
    }

    #[tokio::test]
    async fn read_pattern_is_case_insensitive_and_reports_matches() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "notes.txt", "Score: 3\nnothing\nSCORE: 7\n");
        let outcome = runner(temp.path())
            .run(
                &RewardCheck::new()
                    .with_path("notes.txt")
                    .with_pattern("score"),
            )
            .await
            .unwrap();
        assert_eq!(
            outcome.result,
            "path: notes.txt\nmatches: 2\n1: Score: 3\n3: SCORE: 7"
        );

        let outcome = runner(temp.path())
            .run(
                &RewardCheck::new()
                    .with_path("notes.txt")
                    .with_pattern("absent"),
            )
            .await
            .unwrap();
        assert_eq!(
            outcome.result,
            "path: notes.txt\nmatches: 0\n(no line matches)"
        );
    }

    #[tokio::test]
    async fn read_pattern_caps_hits_and_output() {
        let temp = tempfile::tempdir().unwrap();
        let text = (1..=10).fold(String::new(), |mut text, number| {
            let _ = writeln!(text, "hit {number}");
            text
        });
        write(temp.path(), "notes.txt", &text);

        let runner = CheckRunner::new(temp.path(), Arc::new(FakeCommandRunner::new()))
            .with_max_hits(10)
            .with_max_output_chars(20);
        let outcome = runner
            .run(
                &RewardCheck::new()
                    .with_path("notes.txt")
                    .with_pattern("hit"),
            )
            .await
            .unwrap();
        assert!(outcome.result.contains("matches: 10"), "{}", outcome.result);
        assert!(
            outcome.result.contains("cut: too many matches"),
            "{}",
            outcome.result
        );

        let runner =
            CheckRunner::new(temp.path(), Arc::new(FakeCommandRunner::new())).with_max_hits(3);
        let outcome = runner
            .run(
                &RewardCheck::new()
                    .with_path("notes.txt")
                    .with_pattern("HIT"),
            )
            .await
            .unwrap();
        assert!(outcome.result.contains("matches: 3"), "{}", outcome.result);
    }

    #[tokio::test]
    async fn read_folder_lists_sorted_relative_files() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(temp.path().join("sub")).unwrap();
        write(temp.path(), "b.txt", "b");
        write(temp.path(), "a.txt", "a");
        std::fs::write(temp.path().join("sub/c.txt"), "c").unwrap();

        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("."))
            .await
            .unwrap();
        assert_eq!(
            outcome.result,
            "path: .\nfolder; files:\na.txt\nb.txt\nsub/c.txt"
        );

        let outcome = CheckRunner::new(temp.path(), Arc::new(FakeCommandRunner::new()))
            .with_listing_limit(1)
            .run(&RewardCheck::new().with_path("."))
            .await
            .unwrap();
        assert_eq!(outcome.result, "path: .\nfolder; files:\na.txt");
    }

    #[tokio::test]
    async fn read_reports_missing_paths_and_binary_files() {
        let temp = tempfile::tempdir().unwrap();
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("../escape.txt"))
            .await;
        assert!(
            matches!(outcome, Err(Error::Validation(_))),
            "unexpected: {outcome:?}"
        );

        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_tool(CheckTool::Read))
            .await
            .unwrap();
        assert!(outcome.skipped.unwrap().contains("needs a \"path\""));

        std::fs::write(temp.path().join("blob.bin"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("blob.bin"))
            .await
            .unwrap();
        assert!(
            outcome.result.contains("[binary file]"),
            "{}",
            outcome.result
        );

        std::fs::write(temp.path().join("missing.txt"), "").unwrap();
        assert!(std::fs::remove_file(temp.path().join("missing.txt")).is_ok());
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("missing.txt"))
            .await;
        assert!(
            matches!(outcome, Err(Error::SysOp(_))),
            "unexpected: {outcome:?}"
        );
    }

    #[tokio::test]
    async fn picture_reads_report_the_header_not_the_pixels() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("poster.png"), fixtures::png(800, 600)).unwrap();
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("poster.png"))
            .await
            .unwrap();
        assert!(outcome.succeeded(), "unexpected: {outcome:?}");
        assert_eq!(
            outcome.result,
            "path: poster.png\npicture file (800x600 PNG, 33 bytes)\n\
             the judge sees the picture's metadata (format and size), not its pixels"
        );
        // No picture is attached, because none was decoded.
        assert!(outcome.screenshots.is_empty(), "no browser, no screenshots");
    }

    #[tokio::test]
    async fn every_supported_picture_format_is_read_from_its_header() {
        let temp = tempfile::tempdir().unwrap();
        let cases: [(&str, Vec<u8>, &str, u32, u32); 7] = [
            ("a.png", fixtures::png(800, 600), "PNG", 800, 600),
            ("b.jpg", fixtures::jpeg(1920, 1080), "JPEG", 1920, 1080),
            ("c.jpeg", fixtures::jpeg(12, 34), "JPEG", 12, 34),
            ("d.gif", fixtures::gif(64, 32, 2), "animated GIF", 64, 32),
            ("e.webp", fixtures::webp_lossy(320, 240), "WEBP", 320, 240),
            ("f.webp", fixtures::webp_lossless(64, 48), "WEBP", 64, 48),
            ("g.bmp", fixtures::bmp(640, -480), "BMP", 640, 480),
        ];
        for (name, bytes, format, width, height) in cases {
            std::fs::write(temp.path().join(name), &bytes).unwrap();
            let outcome = runner(temp.path())
                .run(&RewardCheck::new().with_path(name))
                .await
                .unwrap();
            assert!(outcome.succeeded(), "{name}: {outcome:?}");
            let header = format!(
                "picture file ({width}x{height} {format}, {} bytes)",
                bytes.len()
            );
            assert!(
                outcome.result.contains(&header),
                "{name}: {}",
                outcome.result
            );
            assert!(
                outcome.result.contains("not its pixels"),
                "{name}: {}",
                outcome.result
            );
            assert!(
                outcome.result.starts_with(&format!("path: {name}\n")),
                "{name}: {}",
                outcome.result
            );
        }
    }

    #[tokio::test]
    async fn an_unreadable_picture_is_ok_evidence_not_an_error() {
        let temp = tempfile::tempdir().unwrap();
        // A file whose bytes are not a picture at all.
        write(temp.path(), "broken.png", "not really a picture");
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("broken.png"))
            .await
            .unwrap();
        assert!(outcome.succeeded(), "unexpected: {outcome:?}");
        assert_eq!(outcome.result, UNREADABLE_PICTURE);
        assert_eq!(outcome.result, "not a readable picture");

        // A truncated header.
        let mut truncated = fixtures::png(800, 600);
        truncated.truncate(18);
        std::fs::write(temp.path().join("short.jpg"), truncated).unwrap();
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("short.jpg"))
            .await
            .unwrap();
        assert_eq!(outcome.result, UNREADABLE_PICTURE);
        assert!(outcome.skipped.is_none());

        // A picture that is not there reads as unreadable too, as in Python.
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("missing.png"))
            .await
            .unwrap();
        assert_eq!(outcome.result, UNREADABLE_PICTURE);
    }

    #[tokio::test]
    async fn a_picture_above_pythons_limits_says_so_without_converting() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("wide.png"), fixtures::png(2000, 900)).unwrap();
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("wide.png"))
            .await
            .unwrap();
        assert!(
            outcome
                .result
                .contains("because its longest side (2000 px) is longer than 1600 px"),
            "{}",
            outcome.result
        );
        assert!(
            outcome.result.contains("this port does not re-encode"),
            "{}",
            outcome.result
        );

        // A small BMP is still outside what a vision API takes as it is.
        std::fs::write(temp.path().join("small.bmp"), fixtures::bmp(10, 10)).unwrap();
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("small.bmp"))
            .await
            .unwrap();
        assert!(
            outcome
                .result
                .contains("because BMP is not a format a model takes as it is"),
            "{}",
            outcome.result
        );
    }

    #[tokio::test]
    async fn an_uppercase_picture_suffix_is_still_a_picture() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("POSTER.PNG"), fixtures::png(20, 10)).unwrap();
        let outcome = runner(temp.path())
            .run(&RewardCheck::new().with_path("POSTER.PNG"))
            .await
            .unwrap();
        assert!(
            outcome
                .result
                .contains("picture file (20x10 PNG, 33 bytes)"),
            "{}",
            outcome.result
        );
    }

    #[tokio::test]
    async fn page_source_is_refused_while_scripts_and_styles_are_read() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "index.html", "the score is 7\n");
        write(temp.path(), "flat.htm", "the score is 7\n");
        write(temp.path(), "app.js", "const score = 7\n");
        write(temp.path(), "theme.css", ".score { color: red }\n");

        for name in ["index.html", "flat.htm"] {
            let outcome = runner(temp.path())
                .run(&RewardCheck::new().with_path(name))
                .await
                .unwrap();
            assert!(!outcome.ok, "{name}: {outcome:?}");
            assert_eq!(outcome.skipped.as_deref(), Some(PAGE_SOURCE_REFUSED));
            // Nothing of the page's source reaches the evidence.
            assert_eq!(outcome.result, "");
            assert!(outcome.error.is_none());
            let reason = outcome.skipped.unwrap();
            assert!(
                reason.contains("the source of a page is not read"),
                "{reason}"
            );
            assert!(reason.contains("playwright"), "{reason}");

            // A pattern does not sneak the source past the refusal.
            let outcome = runner(temp.path())
                .run(&RewardCheck::new().with_path(name).with_pattern("score"))
                .await
                .unwrap();
            assert_eq!(outcome.skipped.as_deref(), Some(PAGE_SOURCE_REFUSED));
            assert_eq!(outcome.result, "");
        }

        // A script and a stylesheet are read as source, as Python reads them.
        for name in ["app.js", "theme.css"] {
            let outcome = runner(temp.path())
                .run(&RewardCheck::new().with_path(name))
                .await
                .unwrap();
            assert!(outcome.succeeded(), "{name}: {outcome:?}");
            assert!(
                outcome.result.contains("score"),
                "{name}: {}",
                outcome.result
            );
        }
    }

    #[tokio::test]
    async fn browser_checks_are_typed_deferred() {
        let temp = tempfile::tempdir().unwrap();
        for check in [
            RewardCheck::new().with_page("index.html"),
            RewardCheck::new().with_action(json!({"type": "click"})),
        ] {
            let error = runner(temp.path()).run(&check).await.unwrap_err();
            assert!(
                matches!(error, Error::Unsupported(_)),
                "unexpected: {error}"
            );
            assert!(error.to_string().contains("playwright"));
        }

        let outcome = runner(temp.path()).run(&RewardCheck::new()).await.unwrap();
        assert!(outcome.skipped.unwrap().contains("unknown tool"));
    }

    #[tokio::test]
    async fn python_check_runs_through_the_command_seam() {
        if find_program("python3").is_none() && find_program("python").is_none() {
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let fake = Arc::new(FakeCommandRunner::new());
        fake.push_result("python3", "-I", ExecResult::ok("{\"score\": 7}\n"));
        fake.push_result("python", "-I", ExecResult::ok("{\"score\": 7}\n"));
        let runner = CheckRunner::new(temp.path(), fake.clone());

        let outcome = runner
            .run(&RewardCheck::new().with_code("print('{\"score\": 7}')"))
            .await
            .unwrap();
        assert!(outcome.ok, "{outcome:?}");
        assert!(
            outcome.result.contains("exit_code: 0"),
            "{}",
            outcome.result
        );
        assert!(
            outcome.result.contains("{\"score\": 7}"),
            "{}",
            outcome.result
        );

        let invocations = fake.invocations();
        assert_eq!(invocations.len(), 1);
        let spec = &invocations[0];
        assert!(spec.program == "python3" || spec.program == "python");
        assert_eq!(spec.args.first().map(String::as_str), Some("-I"));
        assert_eq!(spec.args.len(), 2);
        assert!(spec.args[1].ends_with("check.py"));
        assert_eq!(spec.cwd.as_deref(), Some(temp.path()));
        assert_eq!(
            spec.env.get("HTTP_PROXY").map(String::as_str),
            Some(PROXY_SINK)
        );
        assert_eq!(
            spec.env.get("HTTPS_PROXY").map(String::as_str),
            Some(PROXY_SINK)
        );
        assert_eq!(spec.timeout_ms, Some(PYTHON_TIMEOUT_MS));
        let temp_dir = spec.env.get("TMPDIR").expect("TMPDIR");
        assert!(Path::new(temp_dir).is_absolute());
        assert!(
            !Path::new(temp_dir).starts_with(temp.path()),
            "the scratch dir must be outside the workspace"
        );
        // The scratch directory is cleaned up after the check.
        assert!(
            !Path::new(temp_dir).exists(),
            "scratch dir left behind: {temp_dir}"
        );
    }

    #[tokio::test]
    async fn python_failure_and_timeout_are_evidence() {
        if find_program("python3").is_none() && find_program("python").is_none() {
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let fake = Arc::new(FakeCommandRunner::new());
        fake.push_result("python3", "-I", ExecResult::failure("boom\n", 2));
        fake.push_result("python", "-I", ExecResult::failure("boom\n", 2));
        fake.push_result("python3", "-I", ExecResult::timeout(""));
        fake.push_result("python", "-I", ExecResult::timeout(""));
        let runner = CheckRunner::new(temp.path(), fake);

        let outcome = runner
            .run(&RewardCheck::new().with_code("raise SystemExit(2)"))
            .await
            .unwrap();
        assert!(!outcome.ok);
        assert!(outcome.error.unwrap().contains("exit code 2"));
        assert!(outcome.result.contains("boom"), "{}", outcome.result);

        let outcome = runner
            .run(&RewardCheck::new().with_code("while True: pass"))
            .await
            .unwrap();
        assert!(!outcome.ok);
        assert!(outcome.error.unwrap().contains("timed out"));
        assert!(
            outcome.result.contains("exit_code: none"),
            "{}",
            outcome.result
        );
    }

    #[tokio::test]
    async fn real_runner_runs_a_checked_program_when_available() {
        // `true` prints nothing; `printf` needs an operand, so it is given one.
        let (program, stdout) = if find_program("true").is_some() {
            ("true", "")
        } else if find_program("printf").is_some() {
            ("printf", "ok\n")
        } else {
            return; // no POSIX tool on PATH: nothing to prove here
        };
        let mut spec = CommandSpec::new(program);
        if !stdout.is_empty() {
            // `printf` needs an operand; `true` is run bare.
            spec = spec.arg(stdout);
        }
        let result = ProcessCommandRunner::new()
            .run(&spec, &CancellationToken::new())
            .await
            .unwrap();
        assert!(result.success(), "unexpected: {result:?}");
        assert_eq!(result.stdout, stdout);
    }

    #[test]
    fn outcome_constructors_and_serde() {
        let ok = CheckOutcome::ok("done");
        assert!(ok.succeeded());
        assert!(!CheckOutcome::skipped("why not").succeeded());
        let failed = CheckOutcome::failed("boom");
        assert_eq!(failed.error.as_deref(), Some("boom"));
        assert!(!failed.succeeded());

        let value = serde_json::to_value(&ok).unwrap();
        assert_eq!(value["ok"], true);
        assert_eq!(value["screenshots"], json!([]));
        let restored: CheckOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(restored, ok);
        let tolerant: CheckOutcome = serde_json::from_str(r#"{"ok": true, "extra": 1}"#).unwrap();
        assert!(tolerant.ok);
    }

    #[test]
    fn clip_keeps_whole_text_and_names_the_cut() {
        assert_eq!(clip("abc", 5), "abc");
        assert_eq!(clip("abcdef", 3), "abc\n...[cut 3 characters]");
        assert_eq!(clip("abcdef", 0), "abcdef");
    }
}
