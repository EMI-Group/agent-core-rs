//! Port of `openjiuwen/harness/tools/filesystem.py`: the `glob`, `list_files` and `grep` tools.
//!
//! `glob` delegates to [`FsOperation::search_files`] and `list_files` to
//! [`FsOperation::list_files`] / [`FsOperation::list_directories`], so both reuse the
//! sys-operation sandbox and keep the Python payload shapes. `grep` runs an in-process
//! scanner (`walkdir` + `regex`) that emits ripgrep-shaped output, so it needs neither the
//! `rg` binary nor a shell.
//!
//! Deviations from Python, all deliberate:
//!
//! - `grep` never shells out: `--type` filters and multiline search report the Python
//!   "requires ripgrep (rg)" failures, and the Windows `Select-String` fallback is not
//!   ported.
//! - `grep` skips a file whose first [`BINARY_SNIFF_BYTES`] contain a NUL byte; ripgrep's
//!   `--max-columns` preview and colours are not reproduced.
//! - `glob` de-duplicates by absolute path and relativizes the returned names against the
//!   resolved base; the Python symlink normalization of `os.path.relpath` is replaced by a
//!   canonicalized-base retry.

// The tool modules are private and their surface is re-exported from `mod.rs`; these items
// are `pub` for that re-export, which lands with the registry wiring wave.
#![allow(unreachable_pub)]
// Python parity: the grep helpers report expected failures as `ToolPayload`, and the public
// signature is part of the tool contract. Boxing the error would change every caller, so the
// size/performance heuristic does not apply here.
#![allow(clippy::result_large_err)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
};
use openjiuwen_sysop::fs::types::{ListDirsParams, ListFilesParams, SearchFilesParams};
use openjiuwen_sysop::operation::FsOperation;

use super::descriptions::description;
use super::support::{parse_args, schema_for, tool_card, ToolEnv, ToolPayload};

/// Maximum number of files `glob` returns (`GlobTool.DEFAULT_MAX_RESULTS`).
const MAX_GLOB_RESULTS: usize = 100;

/// Number of emitted `grep` entries kept when the model omits `head_limit`.
const DEFAULT_HEAD_LIMIT: usize = 250;

/// Directories `grep` never descends into (`GrepTool.VCS_DIRECTORIES_TO_EXCLUDE`).
const VCS_DIRECTORIES: [&str; 6] = [".git", ".svn", ".hg", ".bzr", ".jj", ".sl"];

/// How many leading bytes of a file are sniffed for a NUL byte before `grep` calls it binary.
const BINARY_SNIFF_BYTES: usize = 8 * 1024;

/// Python message for an `output_mode` outside the three supported values.
const INVALID_OUTPUT_MODE: &str = "output_mode must be one of: content, files_with_matches, count";

/// Python message for the unsupported `type` filter.
const TYPE_NEEDS_RIPGREP: &str = "type filter requires ripgrep (rg) to be installed";

/// Python message for the unsupported multiline search.
const MULTILINE_NEEDS_RIPGREP: &str = "multiline search requires ripgrep (rg) to be installed";

/// Python message for a document search without a pattern.
const MISSING_PATTERN: &str = "pattern is required";

/// Python message for a directory search without a filesystem backend.
const MISSING_FS: &str = "filesystem operation is not configured";

/// Matching options for path-like glob filters.
const GLOB_MATCH_OPTIONS: glob::MatchOptions = glob::MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

// ---------------------------------------------------------------------------------------
// glob
// ---------------------------------------------------------------------------------------

/// Arguments of the `glob` tool.
///
/// Python parity: the `pattern`/`path` parameters of `GlobTool.invoke`. A missing or empty
/// `path` searches the workspace root.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct GlobArgs {
    /// The glob pattern to match file names against, e.g. `**/*.rs` or `src/**/*.{rs,toml}`.
    pub pattern: String,
    /// Directory to search in; defaults to the workspace root.
    #[serde(default)]
    #[schemars(with = "String")]
    pub path: Option<String>,
}

/// The `glob` tool: finds files by name pattern through a [`FsOperation`] backend.
pub struct GlobTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl GlobTool {
    /// Builds the tool over `env`, searching through `fs`.
    #[must_use]
    pub fn new(env: ToolEnv, fs: Option<Arc<dyn FsOperation>>) -> Self {
        let card = Self::card_for(env.language());
        Self { env, fs, card }
    }

    /// Builds the model-facing card in `language`.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "glob",
            description("glob", language),
            schema_for::<GlobArgs>(),
            true,
        )
    }

    /// Expands shell-style brace alternatives, one nesting level deep.
    ///
    /// `*.{rs,toml}` becomes `["*.rs", "*.toml"]`. A pattern without a brace pair, whose
    /// brace body still contains braces, or with fewer than two alternatives is returned
    /// unchanged as a single element. Every expanded alternative is expanded again, so
    /// `{a,b}/{c,d}` yields four patterns. Alternatives are trimmed.
    #[must_use]
    pub fn expand_braces(pattern: &str) -> Vec<String> {
        let Some((prefix, body, suffix)) = split_brace_group(pattern) else {
            return vec![pattern.to_owned()];
        };
        let alternatives: Vec<&str> = body
            .split(',')
            .map(str::trim)
            .filter(|alternative| !alternative.is_empty())
            .collect();
        if alternatives.len() < 2 {
            return vec![pattern.to_owned()];
        }
        let mut expanded = Vec::new();
        for alternative in alternatives {
            expanded.extend(Self::expand_braces(&format!(
                "{prefix}{alternative}{suffix}"
            )));
        }
        expanded
    }

    /// Runs the search and returns the payload.
    async fn execute(&self, inputs: &Value) -> ToolPayload {
        // Python reads `pattern` from the raw arguments and reports its absence with the
        // plain validation text rather than a schema error, so check it before parsing.
        if inputs.is_object()
            && inputs
                .get("pattern")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            return ToolPayload::fail(MISSING_PATTERN);
        }
        let args: GlobArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        if args.pattern.is_empty() {
            return ToolPayload::fail(MISSING_PATTERN);
        }

        let base = match self
            .env
            .resolve_dir("glob", "path", args.path.as_deref().unwrap_or(""), inputs)
            .await
        {
            Ok(base) => base,
            Err(payload) => return payload,
        };
        let Some(fs) = self.fs.as_ref() else {
            return ToolPayload::fail(MISSING_FS);
        };

        let started = Instant::now();
        let base_string = base.to_string_lossy().into_owned();
        let mut matching_files: Vec<String> = Vec::new();
        for pattern in Self::expand_braces(&args.pattern) {
            let result = fs
                .search_files(SearchFilesParams::new(base_string.clone(), pattern))
                .await;
            if result.is_failure() {
                return ToolPayload::fail(result.message);
            }
            let Some(data) = result.data else {
                continue;
            };
            for item in data.matching_files {
                if !matching_files.iter().any(|known| known == &item.path) {
                    matching_files.push(item.path);
                }
            }
        }

        let truncated = matching_files.len() > MAX_GLOB_RESULTS;
        matching_files.truncate(MAX_GLOB_RESULTS);
        let filenames: Vec<String> = matching_files
            .iter()
            .map(|path| relative_name(&base, &PathBuf::from(path)))
            .collect();
        let duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        ToolPayload::ok(json!({
            "durationMs": duration_ms,
            "numFiles": matching_files.len(),
            "filenames": filenames,
            "matching_files": matching_files,
            "truncated": truncated,
            "count": matching_files.len(),
        }))
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.execute(&inputs).await.to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        let Ok(parsed) = serde_json::from_value::<ToolOutput>(output.clone()) else {
            return render_payload_text(output);
        };
        let Some(data) = parsed.data.as_ref().filter(|data| data.is_object()) else {
            return render_tool_output(&parsed);
        };
        if !parsed.success {
            return render_tool_output(&parsed);
        }
        let files = string_list(data, "matching_files");
        if files.is_empty() {
            return "No files found.".to_owned();
        }
        let mut text = files.join("\n");
        if data
            .get("truncated")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            text = format!(
                "{text}\n(Results truncated to the first {} files; use a more specific path or pattern.)",
                files.len()
            );
        }
        text
    }
}

/// Splits `pattern` around its first brace group.
///
/// Returns `(prefix, body, suffix)`, or `None` when there is no complete brace pair or the
/// body itself contains braces (a nested group, which is not expanded).
fn split_brace_group(pattern: &str) -> Option<(&str, &str, &str)> {
    let open = pattern.find('{')?;
    let close = pattern[open + 1..].find('}')? + open + 1;
    let body = &pattern[open + 1..close];
    if body.contains(['{', '}']) {
        return None;
    }
    Some((&pattern[..open], body, &pattern[close + 1..]))
}

// ---------------------------------------------------------------------------------------
// list_files
// ---------------------------------------------------------------------------------------

/// Arguments of the `list_files` tool.
///
/// Python parity: the `path`/`show_hidden` parameters of `ListDirTool.invoke`. A missing or
/// empty `path` lists the workspace root, and hidden entries are hidden by default.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ListDirArgs {
    /// Directory to list; defaults to the workspace root.
    #[serde(default)]
    #[schemars(with = "String")]
    pub path: Option<String>,
    /// Include entries whose name starts with `.`.
    #[serde(default, deserialize_with = "deserialize_flag")]
    #[schemars(with = "bool")]
    pub show_hidden: bool,
}

/// The `list_files` tool: lists one directory level through a [`FsOperation`] backend.
pub struct ListDirTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl ListDirTool {
    /// Builds the tool over `env`, listing through `fs`.
    #[must_use]
    pub fn new(env: ToolEnv, fs: Option<Arc<dyn FsOperation>>) -> Self {
        let card = Self::card_for(env.language());
        Self { env, fs, card }
    }

    /// Builds the model-facing card in `language`.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "list_files",
            description("list_files", language),
            schema_for::<ListDirArgs>(),
            true,
        )
    }

    /// Lists one directory level and returns the payload.
    async fn execute(&self, inputs: &Value) -> ToolPayload {
        let args: ListDirArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        let base = match self
            .env
            .resolve_dir(
                "list_files",
                "path",
                args.path.as_deref().unwrap_or(""),
                inputs,
            )
            .await
        {
            Ok(base) => base,
            Err(payload) => return payload,
        };
        let Some(fs) = self.fs.as_ref() else {
            return ToolPayload::fail(MISSING_FS);
        };
        let base_string = base.to_string_lossy().into_owned();

        let files = fs
            .list_files(ListFilesParams::new(base_string.clone()))
            .await;
        if files.is_failure() {
            return ToolPayload::fail(format!("Failed to list files: {}", files.message));
        }
        let dirs = fs.list_directories(ListDirsParams::new(base_string)).await;
        if dirs.is_failure() {
            return ToolPayload::fail(format!("Failed to list directories: {}", dirs.message));
        }

        let mut files = entry_names(files.data, args.show_hidden);
        let mut dirs = entry_names(dirs.data, args.show_hidden);
        files.sort_unstable();
        dirs.sort_unstable();
        ToolPayload::ok(json!({"files": files, "dirs": dirs}))
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.execute(&inputs).await.to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        let Ok(parsed) = serde_json::from_value::<ToolOutput>(output.clone()) else {
            return render_payload_text(output);
        };
        let Some(data) = parsed.data.as_ref().filter(|data| data.is_object()) else {
            return render_tool_output(&parsed);
        };
        if !parsed.success {
            return render_tool_output(&parsed);
        }
        let mut entries: Vec<String> = string_list(data, "dirs")
            .into_iter()
            .map(|name| format!("{name}/"))
            .collect();
        entries.extend(string_list(data, "files"));
        if entries.is_empty() {
            "No entries found.".to_owned()
        } else {
            entries.join("\n")
        }
    }
}

/// Maps a listing to entry names, dropping dotted entries unless `show_hidden`.
fn entry_names(
    data: Option<openjiuwen_sysop::fs::types::ListFilesData>,
    show_hidden: bool,
) -> Vec<String> {
    let Some(data) = data else {
        return Vec::new();
    };
    data.list_items
        .into_iter()
        .map(|item| item.name)
        .filter(|name| show_hidden || !name.starts_with('.'))
        .collect()
}

/// Reads the strings of a `name` array in a payload `data` object.
fn string_list(data: &Value, name: &str) -> Vec<String> {
    data.get(name)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------------------
// grep
// ---------------------------------------------------------------------------------------

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
    /// Shorthand for `ignore_case` (`-i`); either being set makes the search insensitive.
    #[serde(rename = "-i", default, deserialize_with = "deserialize_flag")]
    #[schemars(with = "bool")]
    pub short_ignore_case: bool,
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

    let mut emitted: Vec<String> = Vec::new();
    let mut any_match = false;
    for path in candidates {
        any_match |= scan_candidate(&context, &path, &mut emitted);
    }

    let offset = args.offset.map_or(0, to_usize);
    let (selected, applied_limit) = apply_head_limit(emitted, args.head_limit, offset);
    Ok(grep_data(mode, any_match, &selected, applied_limit, offset))
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
        .case_insensitive(args.ignore_case || args.short_ignore_case)
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
/// Returns whether the file matched at all.
fn scan_candidate(context: &ScanContext<'_>, path: &Path, emitted: &mut Vec<String>) -> bool {
    let relative = relative_name(context.root, path);
    if !context.filters.is_empty()
        && !context
            .filters
            .iter()
            .any(|filter| glob_filter_matches(filter, path, &relative))
    {
        return false;
    }
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::debug!(path = %path.display(), %error, "skipping unreadable grep candidate");
            return false;
        }
    };
    if is_binary(&bytes) {
        tracing::debug!(path = %path.display(), "skipping binary grep candidate");
        return false;
    }
    let text = String::from_utf8_lossy(&bytes);
    match context.mode {
        GrepOutputMode::Content => push_content_matches(
            context.regex,
            &text,
            &relative,
            context.before,
            context.after,
            context.line_numbers,
            emitted,
        ),
        GrepOutputMode::FilesWithMatches => {
            if !text.lines().any(|line| context.regex.is_match(line)) {
                return false;
            }
            emitted.push(relative);
            true
        }
        GrepOutputMode::Count => {
            let count = text
                .lines()
                .filter(|line| context.regex.is_match(line))
                .count();
            if count == 0 {
                return false;
            }
            emitted.push(format!("{relative}:{count}"));
            true
        }
    }
}

/// Assembles the Python-shaped `grep` payload data.
fn grep_data(
    mode: GrepOutputMode,
    any_match: bool,
    selected: &[String],
    applied_limit: Option<usize>,
    offset: usize,
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
    ]);

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

/// Appends the matching lines of `text` and their context to `emitted`.
///
/// Matching lines render as `{path}:{line}:{text}`, context lines as
/// `{path}-{line}-{text}`; the line number is dropped when `show_numbers` is false. Blank
/// lines are dropped and overlapping context groups are merged. Returns whether `text`
/// matched at all.
fn push_content_matches(
    regex: &regex::Regex,
    text: &str,
    relative: &str,
    before: usize,
    after: usize,
    show_numbers: bool,
    emitted: &mut Vec<String>,
) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return false;
    }
    let mut is_match = vec![false; lines.len()];
    let mut in_group = vec![false; lines.len()];
    let mut any = false;
    for (index, line) in lines.iter().enumerate() {
        if !regex.is_match(line) {
            continue;
        }
        any = true;
        is_match[index] = true;
        let start = index.saturating_sub(before);
        let end = index.saturating_add(after).min(lines.len() - 1);
        for flag in &mut in_group[start..=end] {
            *flag = true;
        }
    }
    if !any {
        return false;
    }
    for (index, line) in lines.iter().enumerate() {
        if !in_group[index] || line.trim().is_empty() {
            continue;
        }
        let separator = if is_match[index] { ':' } else { '-' };
        let number = index + 1;
        emitted.push(if show_numbers {
            format!("{relative}{separator}{number}{separator}{line}")
        } else {
            format!("{relative}{separator}{line}")
        });
    }
    true
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
fn apply_head_limit(
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

/// Renders `path` relative to `root` with `/` separators, falling back to the absolute path.
fn relative_name(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).map(std::path::Path::to_path_buf);
    let relative = relative.or_else(|_| {
        std::fs::canonicalize(root)
            .ok()
            .and_then(|root| path.strip_prefix(&root).ok().map(Path::to_path_buf))
            .ok_or(())
    });
    match relative {
        Ok(relative) => relative.to_string_lossy().replace('\\', "/"),
        Err(()) => path.to_string_lossy().replace('\\', "/"),
    }
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

/// Deserializes a flag leniently; a missing, `null` or unrecognized value is `false`.
fn deserialize_flag<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(json_flag).unwrap_or(false))
}

/// Deserializes a flag like [`deserialize_flag`], defaulting to `true` (the `-n` flag).
fn deserialize_flag_true<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(json_flag).unwrap_or(true))
}

/// Reads a boolean out of the Python truthy spellings (`true`, `1`, `"yes"`, `"on"`).
fn json_flag(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::Number(number) => Some(number.as_f64().is_some_and(|number| number != 0.0)),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;
    use openjiuwen_sysop::{LocalWorkConfig, SysOperation};
    use tempfile::TempDir;

    /// A local filesystem backend rooted at `dir`.
    fn backend(dir: &TempDir) -> Option<Arc<dyn FsOperation>> {
        let operation = SysOperation::local(LocalWorkConfig::unrestricted(
            dir.path().to_string_lossy().to_string(),
        ))
        .unwrap();
        operation.fs()
    }

    /// A `glob` tool over `dir`.
    fn glob_tool(dir: &TempDir) -> GlobTool {
        GlobTool::new(ToolEnv::new(Workspace::new(dir.path())), backend(dir))
    }

    /// A `list_files` tool over `dir`.
    fn list_tool(dir: &TempDir) -> ListDirTool {
        ListDirTool::new(ToolEnv::new(Workspace::new(dir.path())), backend(dir))
    }

    /// A `grep` tool over `dir`.
    fn grep_tool(dir: &TempDir) -> GrepTool {
        GrepTool::new(ToolEnv::new(Workspace::new(dir.path())))
    }

    /// Writes `content` to `relative` below `dir`, creating parent directories.
    fn write(dir: &TempDir, relative: &str, content: &str) {
        let path = dir.path().join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    /// Invokes `tool` with a fresh context and unwraps the payload envelope.
    async fn invoke<T: Tool>(tool: &T, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    /// The string entries of an array value.
    fn strings(value: &Value) -> Vec<String> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_str().unwrap().to_owned())
            .collect()
    }

    /// The sorted property names of a card.
    fn properties(card: &ToolCard) -> Vec<String> {
        let mut keys: Vec<String> = card.input_params["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn cards_expose_the_python_arguments() {
        let glob = GlobTool::card_for("en");
        assert_eq!((glob.name.as_str(), glob.parallel_safe), ("glob", true));
        assert_eq!(glob.input_params["required"], json!(["pattern"]));
        assert_eq!(properties(&glob), ["path", "pattern"]);

        let list = ListDirTool::card_for("en");
        assert_eq!(
            (list.name.as_str(), list.parallel_safe),
            ("list_files", true)
        );
        assert!(list.input_params.get("required").is_none());
        assert_eq!(properties(&list), ["path", "show_hidden"]);

        let grep = GrepTool::card_for("en");
        assert_eq!((grep.name.as_str(), grep.parallel_safe), ("grep", true));
        assert_eq!(grep.input_params["required"], json!(["pattern"]));
        assert_eq!(
            properties(&grep),
            [
                "-A",
                "-B",
                "-C",
                "-i",
                "-n",
                "context",
                "glob",
                "head_limit",
                "ignore_case",
                "multiline",
                "offset",
                "output_mode",
                "path",
                "pattern",
                "type",
            ]
        );
    }

    #[test]
    fn brace_expansion_matches_python() {
        assert_eq!(GlobTool::expand_braces("*.rs"), ["*.rs"]);
        assert_eq!(GlobTool::expand_braces("*.{rs,toml}"), ["*.rs", "*.toml"]);
        assert_eq!(
            GlobTool::expand_braces("src/*.{rs, toml}"),
            ["src/*.rs", "src/*.toml"]
        );
        // Fewer than two alternatives and nested brace groups stay unchanged.
        assert_eq!(GlobTool::expand_braces("*.{rs}"), ["*.{rs}"]);
        assert_eq!(GlobTool::expand_braces("{a,{b,c}}"), ["{a,{b,c}}"]);
        assert_eq!(GlobTool::expand_braces("{a,b}/{c,d}").len(), 4);
    }

    #[test]
    fn output_mode_parsing_follows_python() {
        assert_eq!(GrepOutputMode::parse(""), Some(GrepOutputMode::Content));
        assert_eq!(GrepOutputMode::parse("count"), Some(GrepOutputMode::Count));
        assert_eq!(
            GrepOutputMode::parse("files_with_matches"),
            Some(GrepOutputMode::FilesWithMatches)
        );
        assert_eq!(GrepOutputMode::parse("nope"), None);
        assert_eq!(GrepOutputMode::Content.as_str(), "content");
        assert_eq!(GrepOutputMode::Count.as_str(), "count");
    }

    #[tokio::test]
    async fn glob_finds_nested_files_through_the_backend() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "a.txt", "a");
        write(&dir, "src/b.rs", "b");
        write(&dir, "src/deep/c.rs", "c");
        let tool = glob_tool(&dir);

        let out = invoke(&tool, json!({"pattern": "**/*.rs"})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(out["error"], Value::Null);
        assert_eq!(out["data"]["numFiles"], json!(2));
        assert_eq!(out["data"]["count"], json!(2));
        assert_eq!(out["data"]["truncated"], json!(false));
        assert!(out["data"]["durationMs"].is_u64());
        let mut filenames = strings(&out["data"]["filenames"]);
        filenames.sort();
        assert_eq!(filenames, ["src/b.rs", "src/deep/c.rs"]);
        let matching = strings(&out["data"]["matching_files"]);
        assert_eq!(matching.len(), 2);
        assert!(matching.iter().all(|path| path.starts_with('/')));
        assert_eq!(tool.render_for_llm(&out).lines().count(), 2);
    }

    #[tokio::test]
    async fn glob_relativizes_against_an_explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "src/b.rs", "b");
        write(&dir, "a.rs", "a");
        let tool = glob_tool(&dir);

        let out = invoke(&tool, json!({"pattern": "*.rs", "path": "src"})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(strings(&out["data"]["filenames"]), ["b.rs"]);
        assert_eq!(strings(&out["data"]["matching_files"]).len(), 1);
    }

    #[tokio::test]
    async fn glob_keeps_the_first_hundred_results() {
        let dir = tempfile::tempdir().unwrap();
        for index in 0..105 {
            write(&dir, &format!("file{index:03}.txt"), "x");
        }
        let tool = glob_tool(&dir);

        let out = invoke(&tool, json!({"pattern": "*.txt"})).await;

        assert_eq!(out["data"]["numFiles"], json!(100));
        assert_eq!(out["data"]["count"], json!(100));
        assert_eq!(out["data"]["truncated"], json!(true));
        assert_eq!(strings(&out["data"]["filenames"]).len(), 100);
        assert_eq!(strings(&out["data"]["matching_files"]).len(), 100);
        assert!(tool.render_for_llm(&out).ends_with(
            "(Results truncated to the first 100 files; use a more specific path or pattern.)"
        ));
    }

    #[tokio::test]
    async fn glob_reports_no_files_and_validation_failures() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "a.txt", "a");
        let tool = glob_tool(&dir);

        let empty = invoke(&tool, json!({"pattern": "*.md"})).await;
        assert_eq!(empty["success"], json!(true));
        assert_eq!(empty["data"]["numFiles"], json!(0));
        assert_eq!(empty["data"]["truncated"], json!(false));
        assert_eq!(strings(&empty["data"]["filenames"]), Vec::<String>::new());
        assert_eq!(tool.render_for_llm(&empty), "No files found.");

        for inputs in [json!({}), json!({"pattern": ""})] {
            let out = invoke(&tool, inputs).await;
            assert_eq!(out["success"], json!(false));
            assert_eq!(out["error"], json!("pattern is required"));
        }

        let escaping = invoke(&tool, json!({"pattern": "*", "path": "../"})).await;
        assert_eq!(escaping["success"], json!(false));
        assert!(escaping["error"]
            .as_str()
            .unwrap()
            .starts_with("Invalid path"));
    }

    #[tokio::test]
    async fn missing_backends_are_failure_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let env = ToolEnv::new(Workspace::new(dir.path()));
        for out in [
            invoke(
                &GlobTool::new(env.clone(), None),
                json!({"pattern": "*.rs"}),
            )
            .await,
            invoke(&ListDirTool::new(env, None), json!({})).await,
        ] {
            assert_eq!(out["success"], json!(false));
            assert_eq!(
                out["error"],
                json!("filesystem operation is not configured")
            );
        }
    }

    #[tokio::test]
    async fn list_files_sorts_and_hides_dotted_entries() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "alpha/x.txt", "a");
        write(&dir, "beta/y.txt", "b");
        write(&dir, ".hidden/z.txt", "c");
        write(&dir, "b.txt", "b");
        write(&dir, "a.txt", "a");
        write(&dir, ".dot", "d");
        let tool = list_tool(&dir);

        let out = invoke(&tool, json!({})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(strings(&out["data"]["files"]), ["a.txt", "b.txt"]);
        assert_eq!(strings(&out["data"]["dirs"]), ["alpha", "beta"]);
        assert_eq!(tool.render_for_llm(&out), "alpha/\nbeta/\na.txt\nb.txt");

        let hidden = invoke(&tool, json!({"show_hidden": true})).await;
        assert_eq!(
            strings(&hidden["data"]["files"]),
            [".dot", "a.txt", "b.txt"]
        );
        assert_eq!(
            strings(&hidden["data"]["dirs"]),
            [".hidden", "alpha", "beta"]
        );
    }

    /// Writes the fixture used by the grep content tests.
    fn content_fixture(dir: &TempDir) {
        write(
            dir,
            "notes.txt",
            "alpha\nneedle here\nbeta\ngamma\nneedle again\ndelta\n",
        );
    }

    #[tokio::test]
    async fn grep_content_mode_renders_matches_and_context() {
        let dir = tempfile::tempdir().unwrap();
        content_fixture(&dir);
        let tool = grep_tool(&dir);

        let out = invoke(&tool, json!({"pattern": "needle", "-C": 1})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(out["error"], Value::Null);
        assert_eq!(out["data"]["mode"], json!("content"));
        assert_eq!(out["data"]["exit_code"], json!(0));
        assert_eq!(out["data"]["numLines"], json!(6));
        assert_eq!(out["data"]["count"], json!(6));
        assert_eq!(out["data"]["numFiles"], json!(0));
        assert_eq!(out["data"]["appliedOffset"], Value::Null);
        assert_eq!(out["data"]["appliedLimit"], Value::Null);
        assert_eq!(
            out["data"]["stdout"],
            json!(concat!(
                "notes.txt-1-alpha\nnotes.txt:2:needle here\nnotes.txt-3-beta\n",
                "notes.txt-4-gamma\nnotes.txt:5:needle again\nnotes.txt-6-delta"
            ))
        );
        assert_eq!(tool.render_for_llm(&out), out["data"]["stdout"]);

        let bare = invoke(&tool, json!({"pattern": "needle", "-n": false})).await;
        assert_eq!(
            bare["data"]["stdout"],
            json!("notes.txt:needle here\nnotes.txt:needle again")
        );

        let before = invoke(&tool, json!({"pattern": "gamma", "-B": 1, "-A": 0})).await;
        assert_eq!(
            before["data"]["stdout"],
            json!("notes.txt-3-beta\nnotes.txt:4:gamma")
        );
    }

    #[tokio::test]
    async fn grep_ignore_case_and_the_i_flag() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "shout.txt", "HELLO\nworld\n");
        let tool = grep_tool(&dir);

        let sensitive = invoke(&tool, json!({"pattern": "hello"})).await;
        assert_eq!(sensitive["success"], json!(true));
        assert_eq!(sensitive["data"]["exit_code"], json!(1));

        let insensitive = invoke(&tool, json!({"pattern": "hello", "ignore_case": true})).await;
        assert_eq!(insensitive["data"]["exit_code"], json!(0));
        assert_eq!(insensitive["data"]["stdout"], json!("shout.txt:1:HELLO"));

        // `-i` wins even when `ignore_case` is explicitly false.
        let short = invoke(
            &tool,
            json!({"pattern": "hello", "ignore_case": false, "-i": true}),
        )
        .await;
        assert_eq!(short["data"]["stdout"], json!("shout.txt:1:HELLO"));
    }

    #[tokio::test]
    async fn grep_modes_and_glob_filter() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "src/a.rs", "token\ntoken\n");
        write(&dir, "b.txt", "token\n");
        write(&dir, "c.txt", "nothing\n");
        let tool = grep_tool(&dir);

        let filtered = invoke(&tool, json!({"pattern": "token", "glob": "*.rs"})).await;
        assert_eq!(
            filtered["data"]["stdout"],
            json!("src/a.rs:1:token\nsrc/a.rs:2:token")
        );

        let files = invoke(
            &tool,
            json!({"pattern": "token", "output_mode": "files_with_matches"}),
        )
        .await;
        assert_eq!(files["data"]["mode"], json!("files_with_matches"));
        assert_eq!(strings(&files["data"]["filenames"]), ["b.txt", "src/a.rs"]);
        assert_eq!(files["data"]["numFiles"], json!(2));
        assert_eq!(files["data"]["count"], json!(2));
        assert_eq!(files["data"]["stdout"], json!("b.txt\nsrc/a.rs"));

        let counts = invoke(&tool, json!({"pattern": "token", "output_mode": "count"})).await;
        assert_eq!(counts["data"]["mode"], json!("count"));
        assert_eq!(counts["data"]["stdout"], json!("b.txt:1\nsrc/a.rs:2"));
        assert_eq!(counts["data"]["numFiles"], json!(2));
        assert_eq!(counts["data"]["numMatches"], json!(3));
        assert_eq!(counts["data"]["count"], json!(3));
        assert_eq!(strings(&counts["data"]["filenames"]), Vec::<String>::new());
    }

    #[tokio::test]
    async fn grep_applies_head_limit_and_offset() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "lines.txt", "x\nx\nx\nx\nx\n");
        let tool = grep_tool(&dir);

        // A numeric string is accepted for `head_limit`, like the shell tool's timeout.
        let limited = invoke(
            &tool,
            json!({"pattern": "x", "head_limit": "2", "path": "lines.txt"}),
        )
        .await;
        assert_eq!(
            limited["data"]["stdout"],
            json!("lines.txt:1:x\nlines.txt:2:x")
        );
        assert_eq!(limited["data"]["appliedLimit"], json!(2));
        assert_eq!(limited["data"]["appliedOffset"], Value::Null);
        assert_eq!(
            tool.render_for_llm(&limited),
            concat!(
                "lines.txt:1:x\nlines.txt:2:x\n",
                "(Results truncated to 2 entries; pass offset to see more.)"
            )
        );

        let skipped = invoke(
            &tool,
            json!({"pattern": "x", "head_limit": 2, "offset": 2, "path": "lines.txt"}),
        )
        .await;
        assert_eq!(
            skipped["data"]["stdout"],
            json!("lines.txt:3:x\nlines.txt:4:x")
        );
        assert_eq!(skipped["data"]["appliedOffset"], json!(2));
        assert_eq!(skipped["data"]["appliedLimit"], json!(2));

        let unlimited = invoke(
            &tool,
            json!({"pattern": "x", "head_limit": 0, "path": "lines.txt"}),
        )
        .await;
        assert_eq!(unlimited["data"]["numLines"], json!(5));
        assert_eq!(unlimited["data"]["appliedLimit"], Value::Null);
    }

    #[tokio::test]
    async fn grep_without_matches_succeeds_with_exit_code_one() {
        let dir = tempfile::tempdir().unwrap();
        content_fixture(&dir);
        let tool = grep_tool(&dir);

        let out = invoke(&tool, json!({"pattern": "absent"})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(out["data"]["exit_code"], json!(1));
        assert_eq!(out["data"]["stdout"], json!(""));
        assert_eq!(out["data"]["numLines"], json!(0));
        assert_eq!(tool.render_for_llm(&out), "No matches found.");
    }

    #[tokio::test]
    async fn grep_skips_vcs_directories_and_binary_files() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, ".git/config.txt", "token\n");
        write(&dir, "a.txt", "token\n");
        write(&dir, "blob.bin", "\u{0}token\n");
        let tool = grep_tool(&dir);

        let out = invoke(&tool, json!({"pattern": "token"})).await;

        assert_eq!(out["data"]["stdout"], json!("a.txt:1:token"));
    }

    #[tokio::test]
    async fn grep_validation_failures_match_the_python_messages() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "a.txt", "token\n");
        let tool = grep_tool(&dir);

        let cases = [
            (
                json!({"pattern": "token", "output_mode": "nope"}),
                "output_mode must be one of: content, files_with_matches, count",
            ),
            (
                json!({"pattern": "token", "type": "rust"}),
                "type filter requires ripgrep (rg) to be installed",
            ),
            (
                json!({"pattern": "token", "multiline": true}),
                "multiline search requires ripgrep (rg) to be installed",
            ),
        ];
        for (inputs, expected) in cases {
            let out = invoke(&tool, inputs).await;
            assert_eq!(out["success"], json!(false));
            assert_eq!(out["error"], json!(expected));
        }

        for inputs in [json!({}), json!({"pattern": ""})] {
            let out = invoke(&tool, inputs).await;
            assert_eq!(out["error"], json!("pattern is required"));
        }

        let invalid = invoke(&tool, json!({"pattern": "("})).await;
        assert_eq!(invalid["success"], json!(false));
        assert!(invalid["error"]
            .as_str()
            .unwrap()
            .starts_with("invalid pattern: "));
    }

    #[test]
    fn search_reports_validation_failures() {
        let dir = tempfile::tempdir().unwrap();
        let tool = grep_tool(&dir);

        let args: GrepArgs = serde_json::from_value(json!({"pattern": ""})).unwrap();
        let error = tool.search(&args).unwrap_err();
        assert_eq!(error.error.as_deref(), Some("pattern is required"));

        let args: GrepArgs = serde_json::from_value(json!({
            "pattern": "token",
            "output_mode": "Nope",
        }))
        .unwrap();
        let error = tool.search(&args).unwrap_err();
        assert_eq!(
            error.error.as_deref(),
            Some("output_mode must be one of: content, files_with_matches, count")
        );
    }

    #[test]
    fn head_limit_helper_matches_python() {
        let items: Vec<String> = (0..5).map(|index| index.to_string()).collect();
        let all: Vec<String> = items.clone();
        assert_eq!(apply_head_limit(items.clone(), None, 0), (all, None));
        assert_eq!(
            apply_head_limit(items.clone(), Some(2), 1),
            (vec!["1".to_owned(), "2".to_owned()], Some(2))
        );
        assert_eq!(
            apply_head_limit(items, Some(0), 3),
            (vec!["3".to_owned(), "4".to_owned()], None)
        );
    }
}
