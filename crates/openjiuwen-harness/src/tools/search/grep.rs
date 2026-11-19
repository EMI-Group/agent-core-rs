//! The `grep` tool: an in-process, ripgrep-shaped content search.
//!
//! Python parity: `openjiuwen/harness/tools/filesystem.py::GrepTool`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
};

use super::super::descriptions::description;
use super::super::support::{parse_args, schema_for, tool_card, ToolEnv, ToolPayload};
use super::glob::GlobTool;
use super::{deserialize_flag, json_flag, relative_name, MISSING_PATTERN};

/// Number of emitted `grep` entries kept when the model omits `head_limit`.
const DEFAULT_HEAD_LIMIT: usize = 250;

/// Directories `grep` never descends into (`GrepTool.VCS_DIRECTORIES_TO_EXCLUDE`).
const VCS_DIRECTORIES: [&str; 6] = [".git", ".svn", ".hg", ".bzr", ".jj", ".sl"];

/// How many leading bytes of a file are sniffed for a NUL byte before `grep` calls it binary.
const BINARY_SNIFF_BYTES: usize = 8 * 1024;

/// Bytes read from a candidate file per chunk.
///
/// The bytes after the last newline of a chunk are carried into the next chunk, so a match
/// that spans a chunk boundary is still found; the carry is bounded by the file size cap.
pub(super) const CHUNK_BYTES: usize = 64 * 1024;

/// Largest candidate file `grep` scans; a bigger file is skipped and counted in the payload.
///
/// A workspace can hold build artefacts or data dumps far larger than any source file;
/// scanning them costs unbounded time and memory, so they are reported instead of grepped.
pub(super) const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Python message for an `output_mode` outside the three supported values.
const INVALID_OUTPUT_MODE: &str = "output_mode must be one of: content, files_with_matches, count";

/// Python message for the unsupported `type` filter.
const TYPE_NEEDS_RIPGREP: &str = "type filter requires ripgrep (rg) to be installed";

/// Python message for the unsupported multiline search.
const MULTILINE_NEEDS_RIPGREP: &str = "multiline search requires ripgrep (rg) to be installed";

/// Matching options for path-like glob filters.
const GLOB_MATCH_OPTIONS: glob::MatchOptions = glob::MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// The `output_mode` of a `grep` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GrepOutputMode {
    /// Emit the matching lines (the default).
    Content,
    /// Emit one relative path per matching file.
    FilesWithMatches,
    /// Emit `{path}:{count}` per matching file.
    Count,
}

impl GrepOutputMode {
    /// The wire name of the mode.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::FilesWithMatches => "files_with_matches",
            Self::Count => "count",
        }
    }

    /// Parses a wire name; an empty value selects [`Self::Content`], like Python.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "" | "content" => Some(Self::Content),
            "files_with_matches" => Some(Self::FilesWithMatches),
            "count" => Some(Self::Count),
            _ => None,
        }
    }
}

/// Arguments of the `grep` tool.
///
/// Python parity: the `GrepTool.invoke` parameters, including the ripgrep-style short flags
/// (`-B`, `-A`, `-C`, `-n`, `-i`) that models pass through verbatim. Only `pattern` is
/// required; numeric fields accept numbers or numeric strings and boolean fields accept the
/// Python truthy spellings.
// The Python parameter set really does carry four independent flags; splitting them into
// sub-structs would change the flat wire shape the model sees.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GrepArgs {
    /// Regular expression to search for.
    pub pattern: String,
    /// File or directory to search; defaults to the workspace root.
    #[serde(default)]
    #[schemars(with = "String")]
    pub path: Option<String>,
    /// Case-insensitive search.
    #[serde(default, deserialize_with = "deserialize_flag")]
    #[schemars(with = "bool")]
    pub ignore_case: bool,
    /// Space-separated glob filter tokens, e.g. `*.rs src/**/*.{ts,tsx}`.
    #[serde(default)]
    #[schemars(with = "String")]
    pub glob: Option<String>,
    /// Output mode: `content` (default), `files_with_matches` or `count`.
    #[serde(default)]
    #[schemars(with = "String")]
    pub output_mode: Option<String>,
    /// Lines of context before each match (`-B`, content mode only).
    #[serde(rename = "-B", default, deserialize_with = "deserialize_opt_u64")]
    #[schemars(with = "u64")]
    pub context_before: Option<u64>,
    /// Lines of context after each match (`-A`, content mode only).
    #[serde(rename = "-A", default, deserialize_with = "deserialize_opt_u64")]
    #[schemars(with = "u64")]
    pub context_after: Option<u64>,
    /// Lines of context on both sides of each match (`-C`, content mode only).
    #[serde(rename = "-C", default, deserialize_with = "deserialize_opt_u64")]
    #[schemars(with = "u64")]
    pub context_c: Option<u64>,
    /// Lines of context on both sides, overriding `-C`/`-B`/`-A`.
    #[serde(default, deserialize_with = "deserialize_opt_u64")]
    #[schemars(with = "u64")]
    pub context: Option<u64>,
    /// Show line numbers in content mode (`-n`, default `true`).
    #[serde(
        rename = "-n",
        default = "default_true",
        deserialize_with = "deserialize_flag_true"
    )]
    #[schemars(with = "bool")]
    pub line_numbers: bool,
    /// Shorthand for `ignore_case` (`-i`), overriding it whenever the key is present.
    #[serde(rename = "-i", default, deserialize_with = "deserialize_opt_flag")]
    #[schemars(with = "bool")]
    pub short_ignore_case: Option<bool>,
    /// ripgrep `--type` filter; unsupported without rg.
    #[serde(rename = "type", default)]
    #[schemars(with = "String")]
    pub file_type: Option<String>,
    /// Maximum number of emitted entries (250 by default, `0` = unlimited).
    #[serde(default, deserialize_with = "deserialize_opt_u64")]
    #[schemars(with = "u64")]
    pub head_limit: Option<u64>,
    /// Entries to skip before `head_limit` is applied.
    #[serde(default, deserialize_with = "deserialize_opt_u64")]
    #[schemars(with = "u64")]
    pub offset: Option<u64>,
    /// Multi-line search; unsupported without rg.
    #[serde(default, deserialize_with = "deserialize_flag")]
    #[schemars(with = "bool")]
    pub multiline: bool,
}

/// The `grep` tool: an in-process, ripgrep-shaped content search.
pub struct GrepTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl GrepTool {
    /// Builds the tool over `env`.
    #[must_use]
    pub fn new(env: ToolEnv) -> Self {
        let card = Self::card_for(env.language());
        Self { env, card }
    }

    /// Builds the model-facing card in `language`.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "grep",
            description("grep", language),
            schema_for::<GrepArgs>(),
            true,
        )
    }

    /// Searches the workspace-relative `args.path` and returns the payload data.
    ///
    /// This is the synchronous scanner; [`Self::invoke`] resolves the path through
    /// [`ToolEnv::resolve_dir`] first and then runs this off the async reactor. The base
    /// here is a plain join onto the workspace root.
    ///
    /// # Errors
    ///
    /// Returns a failure payload for the Python validation failures: a blank `pattern`, an
    /// unsupported `output_mode`, a `type` filter, `multiline`, an uncompilable regex or an
    /// invalid glob filter.
    pub fn search(&self, args: &GrepArgs) -> std::result::Result<Value, ToolPayload> {
        let base = lexical_base(&self.env.workspace().root_path, args.path.as_deref());
        scan_grep(&base, args)
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        if inputs.is_object()
            && inputs
                .get("pattern")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            return Ok(ToolPayload::fail(MISSING_PATTERN).to_json());
        }
        let args: GrepArgs = match parse_args(&inputs) {
            Ok(args) => args,
            Err(payload) => return Ok(payload.to_json()),
        };
        let base = match self
            .env
            .resolve_dir("grep", "path", args.path.as_deref().unwrap_or(""), &inputs)
            .await
        {
            Ok(base) => base,
            Err(payload) => return Ok(payload.to_json()),
        };
        // The scan walks directories and reads files synchronously; keep it off the reactor.
        match tokio::task::spawn_blocking(move || scan_grep(&base, &args)).await {
            Ok(Ok(data)) => Ok(ToolPayload::ok(data).to_json()),
            Ok(Err(payload)) => Ok(payload.to_json()),
            Err(error) => Ok(ToolPayload::fail(format!("grep scan failed: {error}")).to_json()),
        }
    }

    fn render_for_llm(&self, output: &Value) -> String {
        let Ok(parsed) = serde_json::from_value::<ToolOutput>(output.clone()) else {
            return render_payload_text(output);
        };
        let Some(data) = parsed
            .data
            .as_ref()
            .filter(|data| data.get("stdout").is_some())
        else {
            return render_tool_output(&parsed);
        };
        let mut matches = data
            .get("stdout")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if let Some(limit) = data.get("appliedLimit").and_then(Value::as_u64) {
            matches = format!(
                "{matches}\n(Results truncated to {limit} entries; pass offset to see more.)"
            );
        }
        if let Some(note) = data.get("note").and_then(Value::as_str) {
            matches = if matches.is_empty() {
                note.to_owned()
            } else {
                format!("{matches}\n({note})")
            };
        }
        if parsed.success {
            return if matches.is_empty() {
                "No matches found.".to_owned()
            } else {
                matches
            };
        }
        let error = parsed
            .error
            .clone()
            .filter(|error| !error.is_empty())
            .unwrap_or_else(|| {
                format!(
                    "Search failed with exit code {}.",
                    data.get("exit_code").and_then(Value::as_i64).unwrap_or(-1)
                )
            });
        if matches.is_empty() {
            error
        } else {
            format!("{error}\n{matches}")
        }
    }
}

/// Validates `args`, scans `base` and returns the ripgrep-shaped payload data.
fn scan_grep(base: &Path, args: &GrepArgs) -> std::result::Result<Value, ToolPayload> {
    scan_grep_with(base, args, &FsSource)
}

/// Validates `args`, scans `base` through `source` and returns the ripgrep-shaped payload data.
///
/// The scan collects at most one entry more than `offset + head_limit` (the sentinel that
/// proves truncation) and then stops walking candidates, so a bounded `head_limit` bounds the
/// work instead of merely slicing its result.
pub(super) fn scan_grep_with(
    base: &Path,
    args: &GrepArgs,
    source: &dyn GrepSource,
) -> std::result::Result<Value, ToolPayload> {
    let (mode, regex, filters) = prepare_grep(args)?;
    let (root, candidates) = candidate_files(base);
    let (before, after) = context_window(args);
    let context = ScanContext {
        root: &root,
        regex: &regex,
        filters: &filters,
        mode,
        before,
        after,
        line_numbers: args.line_numbers,
    };

    let mut budget = ScanBudget::new(args);
    let mut emitted: Vec<String> = Vec::new();
    let mut any_match = false;
    for path in candidates {
        if budget.saturated(emitted.len()) {
            break;
        }
        any_match |= scan_candidate(&context, source, &path, &mut emitted, &mut budget);
    }

    let offset = args.offset.map_or(0, to_usize);
    let (selected, applied_limit) = apply_head_limit(emitted, args.head_limit, offset);
    Ok(grep_data(
        mode,
        any_match,
        &selected,
        applied_limit,
        offset,
        budget.skipped_large,
    ))
}

/// Entry budget of one scan, derived from `offset`/`head_limit`.
struct ScanBudget {
    /// Entries the scan may collect: `offset + limit + 1`, the sentinel proving truncation.
    ///
    /// `usize::MAX` when `head_limit` is `0` (unlimited).
    capacity: usize,
    /// Candidate files skipped because they exceed [`MAX_FILE_BYTES`].
    skipped_large: usize,
}

impl ScanBudget {
    /// Derives the budget the same way [`apply_head_limit`] resolves the effective limit.
    fn new(args: &GrepArgs) -> Self {
        let limit = match args.head_limit {
            Some(0) => None,
            Some(limit) => Some(to_usize(limit)),
            None => Some(DEFAULT_HEAD_LIMIT),
        };
        let offset = args.offset.map_or(0, to_usize);
        let capacity = limit.map_or(usize::MAX, |limit| {
            offset.saturating_add(limit).saturating_add(1)
        });
        Self {
            capacity,
            skipped_large: 0,
        }
    }

    /// Whether `collected` entries already exhaust the budget.
    fn saturated(&self, collected: usize) -> bool {
        collected >= self.capacity
    }

    /// Records one candidate skipped by the size cap.
    fn note_skipped_large(&mut self) {
        self.skipped_large += 1;
    }
}

/// Source of candidate contents for the scan.
///
/// Production scans read the filesystem ([`FsSource`]); tests inject a counting source to
/// prove that the scan stops as soon as the budget is exhausted.
pub(super) trait GrepSource {
    /// Returns the length of `path` in bytes, or `None` when it is not a readable file.
    fn size(&self, path: &Path) -> Option<u64>;

    /// Opens `path` for chunked reading.
    fn open(&self, path: &Path) -> std::io::Result<Box<dyn GrepFile>>;
}

/// One candidate opened for chunked reading.
pub(super) trait GrepFile {
    /// Reads up to `buffer.len()` bytes, returning `Ok(0)` at end of file.
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize>;
}

impl GrepFile for std::fs::File {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        std::io::Read::read(self, buffer)
    }
}

/// The filesystem implementation of [`GrepSource`].
pub(super) struct FsSource;

impl GrepSource for FsSource {
    fn size(&self, path: &Path) -> Option<u64> {
        std::fs::metadata(path)
            .ok()
            .filter(std::fs::Metadata::is_file)
            .map(|meta| meta.len())
    }

    fn open(&self, path: &Path) -> std::io::Result<Box<dyn GrepFile>> {
        Ok(Box::new(std::fs::File::open(path)?))
    }
}

/// Validates the `grep` arguments, returning the compiled scan configuration.
///
/// # Errors
///
/// Returns the Python failure payload for a blank pattern, an unsupported output mode, the
/// `type`/`multiline` filters, an uncompilable regex or an invalid glob filter.
fn prepare_grep(
    args: &GrepArgs,
) -> std::result::Result<(GrepOutputMode, regex::Regex, Vec<glob::Pattern>), ToolPayload> {
    if args.pattern.is_empty() {
        return Err(ToolPayload::fail(MISSING_PATTERN));
    }
    let mode = GrepOutputMode::parse(args.output_mode.as_deref().unwrap_or_default())
        .ok_or_else(|| ToolPayload::fail(INVALID_OUTPUT_MODE))?;
    if args
        .file_type
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
    {
        return Err(ToolPayload::fail(TYPE_NEEDS_RIPGREP));
    }
    if args.multiline {
        return Err(ToolPayload::fail(MULTILINE_NEEDS_RIPGREP));
    }
    let regex = regex::RegexBuilder::new(&args.pattern)
        // Python `_as_bool(inputs.get("-i", inputs.get("ignore_case", False)))`: the short
        // flag wins whenever it is present, even when it is false.
        .case_insensitive(args.short_ignore_case.unwrap_or(args.ignore_case))
        .build()
        .map_err(|error| ToolPayload::fail(format!("invalid pattern: {error}")))?;
    let filters = compile_glob_filters(args.glob.as_deref())?;
    Ok((mode, regex, filters))
}

/// The per-run settings [`scan_candidate`] needs.
struct ScanContext<'a> {
    /// Directory the emitted relative paths are computed against.
    root: &'a Path,
    /// The compiled search pattern.
    regex: &'a regex::Regex,
    /// Compile glob filter tokens; empty means every file is scanned.
    filters: &'a [glob::Pattern],
    /// The output mode.
    mode: GrepOutputMode,
    /// Lines of leading context, content mode only.
    before: usize,
    /// Lines of trailing context, content mode only.
    after: usize,
    /// Whether content-mode lines carry a line number.
    line_numbers: bool,
}

/// Scans one candidate file, appending its entries to `emitted`.
///
/// Returns whether the file matched at all. Files that pass their glob filter are read in
/// [`CHUNK_BYTES`] chunks through `source`; a file above [`MAX_FILE_BYTES`] is skipped and
/// counted in `budget` instead.
fn scan_candidate(
    context: &ScanContext<'_>,
    source: &dyn GrepSource,
    path: &Path,
    emitted: &mut Vec<String>,
    budget: &mut ScanBudget,
) -> bool {
    let relative = relative_name(context.root, path);
    if !context.filters.is_empty()
        && !context
            .filters
            .iter()
            .any(|filter| glob_filter_matches(filter, path, &relative))
    {
        return false;
    }
    if source.size(path).is_some_and(|size| size > MAX_FILE_BYTES) {
        budget.note_skipped_large();
        tracing::debug!(
            path = %path.display(),
            cap = MAX_FILE_BYTES,
            "skipping oversized grep candidate"
        );
        return false;
    }
    match context.mode {
        GrepOutputMode::Content => scan_content(context, source, path, &relative, emitted, budget),
        GrepOutputMode::FilesWithMatches => {
            let mut found = false;
            let _ = read_lines(source, path, |_number, line| {
                found = context.regex.is_match(line);
                !found
            });
            if found {
                emitted.push(relative);
                true
            } else {
                false
            }
        }
        GrepOutputMode::Count => {
            let mut count = 0usize;
            let _ = read_lines(source, path, |_number, line| {
                if context.regex.is_match(line) {
                    count += 1;
                }
                true
            });
            if count == 0 {
                false
            } else {
                emitted.push(format!("{relative}:{count}"));
                true
            }
        }
    }
}

/// Streams the lines of `path`, stopping early once the entry budget is exhausted.
///
/// Matching lines render as `{path}:{line}:{text}`, context lines as `{path}-{line}-{text}`;
/// the line number is dropped when `line_numbers` is off. Blank lines are never emitted and
/// overlapping context groups are merged, like the whole-file scan this replaces.
fn scan_content(
    context: &ScanContext<'_>,
    source: &dyn GrepSource,
    path: &Path,
    relative: &str,
    emitted: &mut Vec<String>,
    budget: &ScanBudget,
) -> bool {
    let mut matched = false;
    let mut last_emitted = 0usize;
    let mut pending_after = 0usize;
    // The `before` window of the next match: lines already read but not emitted yet.
    let mut before: VecDeque<(usize, String)> = VecDeque::with_capacity(context.before.min(64));
    let _ = read_lines(source, path, |number, line| {
        if context.regex.is_match(line) {
            matched = true;
            for (context_number, context_line) in before.drain(..) {
                if context_number > last_emitted {
                    push_content_line(
                        context,
                        relative,
                        context_number,
                        &context_line,
                        false,
                        emitted,
                    );
                    last_emitted = context_number;
                }
            }
            push_content_line(context, relative, number, line, true, emitted);
            last_emitted = number;
            pending_after = context.after;
        } else if pending_after > 0 {
            pending_after -= 1;
            push_content_line(context, relative, number, line, false, emitted);
            last_emitted = number;
        } else {
            before.push_back((number, line.to_owned()));
            while before.len() > context.before {
                before.pop_front();
            }
        }
        !budget.saturated(emitted.len())
    });
    matched
}

/// Pushes one rendered content entry unless the line is blank.
fn push_content_line(
    context: &ScanContext<'_>,
    relative: &str,
    number: usize,
    line: &str,
    matched: bool,
    emitted: &mut Vec<String>,
) {
    if line.trim().is_empty() {
        return;
    }
    let separator = if matched { ':' } else { '-' };
    if context.line_numbers {
        emitted.push(format!("{relative}{separator}{number}{separator}{line}"));
    } else {
        emitted.push(format!("{relative}{separator}{line}"));
    }
}

/// Result of a chunked line read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineVisit {
    /// The whole file was read.
    Completed,
    /// `visit` asked to stop.
    Stopped,
    /// The file turned out to be binary and was not visited.
    Binary,
}

/// Calls `visit` with every line of `path`, in order, reading in bounded chunks.
///
/// `visit` returns whether reading should continue. The bytes after the last newline of a
/// chunk are carried into the next chunk, so a line (and therefore a match) spanning a chunk
/// boundary is still seen; the carry stays bounded by [`MAX_FILE_BYTES`]. A NUL byte in the
/// first [`BINARY_SNIFF_BYTES`] short-circuits to [`LineVisit::Binary`].
fn read_lines(
    source: &dyn GrepSource,
    path: &Path,
    mut visit: impl FnMut(usize, &str) -> bool,
) -> LineVisit {
    let mut file = match source.open(path) {
        Ok(file) => file,
        Err(error) => {
            tracing::debug!(path = %path.display(), %error, "skipping unreadable grep candidate");
            return LineVisit::Completed;
        }
    };
    let mut chunk = vec![0u8; CHUNK_BYTES];
    let mut carry: Vec<u8> = Vec::new();
    let mut line_number = 0usize;
    let mut sniffed = false;
    loop {
        let read = match file.read(&mut chunk) {
            Ok(read) => read,
            Err(error) => {
                tracing::debug!(path = %path.display(), %error, "skipping unreadable grep candidate");
                return LineVisit::Completed;
            }
        };
        if !sniffed {
            sniffed = true;
            if is_binary(&chunk[..read]) {
                tracing::debug!(path = %path.display(), "skipping binary grep candidate");
                return LineVisit::Binary;
            }
        }
        let mut data = &chunk[..read];
        while let Some(index) = data.iter().position(|byte| *byte == b'\n') {
            let mut line = std::mem::take(&mut carry);
            line.extend_from_slice(&data[..index]);
            data = &data[index + 1..];
            line_number += 1;
            if !visit(line_number, &decode_line(&line)) {
                return LineVisit::Stopped;
            }
        }
        carry.extend_from_slice(data);
        if read == 0 {
            break;
        }
    }
    if !carry.is_empty() {
        line_number += 1;
        if !visit(line_number, &decode_line(&carry)) {
            return LineVisit::Stopped;
        }
    }
    LineVisit::Completed
}

/// Decodes one line, dropping a `\r` line ending and replacing invalid UTF-8.
fn decode_line(bytes: &[u8]) -> String {
    let bytes = bytes.strip_suffix(b"\r").unwrap_or(bytes);
    String::from_utf8_lossy(bytes).into_owned()
}

/// Assembles the Python-shaped `grep` payload data.
fn grep_data(
    mode: GrepOutputMode,
    any_match: bool,
    selected: &[String],
    applied_limit: Option<usize>,
    offset: usize,
    skipped_large: usize,
) -> Value {
    let stdout = selected.join("\n");
    let mut data = Map::from_iter([
        ("stdout".to_owned(), json!(stdout)),
        ("stderr".to_owned(), json!("")),
        ("exit_code".to_owned(), json!(u8::from(!any_match))),
        ("mode".to_owned(), json!(mode.as_str())),
        (
            "appliedOffset".to_owned(),
            if offset > 0 {
                json!(offset)
            } else {
                Value::Null
            },
        ),
        (
            "appliedLimit".to_owned(),
            applied_limit.map_or(Value::Null, |limit| json!(limit)),
        ),
        ("skippedLargeFiles".to_owned(), json!(skipped_large)),
    ]);
    if skipped_large > 0 {
        data.insert(
            "note".to_owned(),
            json!(format!(
                "skipped {skipped_large} file(s) larger than {MAX_FILE_BYTES} bytes"
            )),
        );
    }

    let counts = |lines: &[String]| -> (usize, u64) {
        let parsed: Vec<u64> = lines
            .iter()
            .filter_map(|line| line.rsplit_once(':'))
            .filter_map(|(_, count)| count.parse::<u64>().ok())
            .collect();
        (parsed.len(), parsed.iter().sum())
    };
    let extras = match mode {
        GrepOutputMode::Content => json!({
            "content": stdout,
            "filenames": [],
            "numFiles": 0,
            "numLines": selected.len(),
            "count": selected.len(),
        }),
        GrepOutputMode::FilesWithMatches => json!({
            "filenames": selected,
            "numFiles": selected.len(),
            "count": selected.len(),
        }),
        GrepOutputMode::Count => {
            let (files, matches) = counts(selected);
            json!({
                "content": stdout,
                "filenames": [],
                "numFiles": files,
                "numMatches": matches,
                "count": matches,
            })
        }
    };
    if let Value::Object(extras) = extras {
        data.extend(extras);
    }
    Value::Object(data)
}

/// Returns the directory relative paths are computed against, plus the files to scan.
///
/// A `base` that is a regular file is scanned on its own and relativized against its parent.
fn candidate_files(base: &Path) -> (PathBuf, Vec<PathBuf>) {
    if base.is_file() {
        let root = base
            .parent()
            .map_or_else(|| base.to_path_buf(), Path::to_path_buf);
        return (root, vec![base.to_path_buf()]);
    }
    let walker = walkdir::WalkDir::new(base)
        .min_depth(1)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| !(entry.file_type().is_dir() && is_vcs_directory(entry.file_name())));
    let mut files = Vec::new();
    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::debug!(%error, "skipping unreadable grep candidate");
                continue;
            }
        };
        if entry.file_type().is_file() {
            files.push(entry.into_path());
        }
    }
    files.sort();
    (base.to_path_buf(), files)
}

/// Whether `name` is one of the VCS directories `grep` never descends into.
fn is_vcs_directory(name: &std::ffi::OsStr) -> bool {
    name.to_str()
        .is_some_and(|name| VCS_DIRECTORIES.contains(&name))
}

/// Whether the leading bytes of `bytes` mark it as binary (a NUL byte).
fn is_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(BINARY_SNIFF_BYTES).any(|byte| *byte == 0)
}

/// The `(before, after)` context window of `args`.
///
/// `context` takes precedence over `-C`, and either expands into both sides; otherwise `-B`
/// and `-A` apply independently.
fn context_window(args: &GrepArgs) -> (usize, usize) {
    if let Some(both) = args.context.or(args.context_c) {
        let both = to_usize(both);
        return (both, both);
    }
    (
        args.context_before.map_or(0, to_usize),
        args.context_after.map_or(0, to_usize),
    )
}

/// Splits the `glob` argument into filter tokens.
///
/// Tokens are whitespace separated; a token without braces is additionally split on `,`
/// (Python `_split_glob_patterns`).
fn split_glob_tokens(raw: Option<&str>) -> Vec<String> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    let mut tokens = Vec::new();
    for token in raw.split_whitespace() {
        if token.contains('{') && token.contains('}') {
            tokens.push(token.to_owned());
        } else {
            tokens.extend(
                token
                    .split(',')
                    .filter(|part| !part.is_empty())
                    .map(str::to_owned),
            );
        }
    }
    tokens
}

/// Compiles the `glob` filter tokens, expanding brace alternatives like ripgrep does.
fn compile_glob_filters(raw: Option<&str>) -> std::result::Result<Vec<glob::Pattern>, ToolPayload> {
    let mut filters = Vec::new();
    for token in split_glob_tokens(raw) {
        for pattern in GlobTool::expand_braces(&token) {
            let compiled = glob::Pattern::new(&pattern).map_err(|error| {
                ToolPayload::fail(format!("invalid glob filter '{pattern}': {error}"))
            })?;
            filters.push(compiled);
        }
    }
    Ok(filters)
}

/// Whether `path` passes one glob filter token.
///
/// A token without `/` matches the file name; a token with `/` matches the `/`-relative path
/// with `**` matching zero or more components.
fn glob_filter_matches(filter: &glob::Pattern, path: &Path, relative: &str) -> bool {
    if !filter.as_str().contains('/') {
        return path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| filter.matches(name));
    }
    glob_path_matches(filter.as_str(), Path::new(relative), 0)
}

/// Matches `pattern` against `relative`, retrying with each `**` component removed.
///
/// `glob::Pattern` needs at least one component per `**`, while ripgrep (like Python's
/// recursive glob) lets it match zero.
fn glob_path_matches(pattern: &str, relative: &Path, depth: usize) -> bool {
    if depth > 8 {
        return false;
    }
    let Ok(compiled) = glob::Pattern::new(pattern) else {
        return false;
    };
    if compiled.matches_path_with(relative, GLOB_MATCH_OPTIONS) {
        return true;
    }
    let mut components: Vec<&str> = pattern.split('/').collect();
    let Some(index) = components.iter().position(|component| *component == "**") else {
        return false;
    };
    components.remove(index);
    let variant = components.join("/");
    if variant.is_empty() {
        return false;
    }
    glob_path_matches(&variant, relative, depth + 1)
}

/// Applies `offset` and `head_limit` to the emitted entries.
///
/// Returns the kept entries and the effective limit when it truncated the list (`None` for
/// the unlimited `head_limit = 0` case and when nothing was dropped).
pub(super) fn apply_head_limit(
    items: Vec<String>,
    limit: Option<u64>,
    offset: usize,
) -> (Vec<String>, Option<usize>) {
    let start = offset.min(items.len());
    let remaining = items.len() - start;
    let effective = match limit {
        Some(0) => return (items.into_iter().skip(start).collect(), None),
        Some(limit) => to_usize(limit),
        None => DEFAULT_HEAD_LIMIT,
    };
    let selected = items.into_iter().skip(start).take(effective).collect();
    (selected, (remaining > effective).then_some(effective))
}

/// Joins a model-supplied relative path onto `root` without touching the filesystem.
///
/// [`GrepTool::invoke`] performs the sandboxed resolution first; this is the base used by the
/// synchronous [`GrepTool::search`] entry point.
fn lexical_base(root: &Path, raw: Option<&str>) -> PathBuf {
    let Some(raw) = raw.filter(|value| !value.is_empty()) else {
        return root.to_path_buf();
    };
    let candidate = Path::new(raw);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    }
}

/// Converts a model-supplied count to `usize`, saturating instead of wrapping.
fn to_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// The serde default of the `-n` flag.
fn default_true() -> bool {
    true
}

/// Deserializes an optional `u64` from a number or a numeric string.
///
/// Ports the `BashTool` leniency: anything that is not a non-negative integer deserializes as
/// `None` instead of failing the whole argument object.
fn deserialize_opt_u64<'de, D>(deserializer: D) -> std::result::Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(json_u64))
}

/// Reads a non-negative integer out of a JSON number or a numeric string.
fn json_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Deserializes an optional flag: absent stays `None`, `null` becomes `Some(false)`.
///
/// Python reads `-i` with `inputs.get("-i", inputs.get("ignore_case", False))`, so a present
/// key wins over `ignore_case` even when its value is null.
fn deserialize_opt_flag<'de, D>(deserializer: D) -> std::result::Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .map(|value| json_flag(value).unwrap_or(false)))
}

/// Deserializes a flag like [`deserialize_flag`], defaulting to `true` (the `-n` flag).
fn deserialize_flag_true<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(json_flag).unwrap_or(true))
}
