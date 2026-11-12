//! Lexical path extraction for the file guard and the shell guard.
//!
//! Ports `openjiuwen/harness/security/permission_engine/fileguard/file_tool_specs.py`
//! (the `FileToolSpec` registry behind [`tool_action`] and [`tool_path_args`]) and
//! the extraction half of
//! `fileguard/path_extract.py` (v0.1.20) behind [`extract_file_accesses`] and
//! [`extract_shell_accesses`].
//!
//! **Deviation from the Python reference.** Python resolves extracted paths with
//! `Path.resolve()`, which touches the filesystem, follows symlinks and fails for
//! paths that do not exist yet. This port performs **no filesystem IO** and
//! normalizes paths **lexically** instead ([`normalize_lexically`]): `.` components
//! are dropped and `..` components fold into the preceding component, with `..`
//! beyond an absolute root ignored. Decisions therefore depend only on the text of
//! the tool call, which keeps the permission check synchronous and cheap as
//! [`super::engine::PermissionEngine`] requires, and makes them independent of the
//! current filesystem state. The cost is that symlinks are not followed here — the
//! sysop layer and the sandbox own symlink escapes, while the file guard judges the
//! literal path.
//!
//! `~`, `$VAR` and `${VAR}` are expanded (HOME first, then `USERPROFILE`), matching
//! `os.path.expanduser`/`os.path.expandvars`; an unset variable leaves the token
//! untouched. Absolute POSIX paths and `Windows` drive paths (`C:\x`) are recognized
//! as absolute, everything else is joined onto the workspace root (or onto the shell
//! command's effective working directory). Shell tokens that cannot be resolved
//! statically (`$VAR`, backticks, `%VAR%`, URLs, redirections into descriptors) are
//! dropped rather than guessed, so the guard never invents a path.
//!
//! Every function is pure, never panics and never returns an error: an unparseable
//! token simply yields no access.

// Size note: ~800 lines of implementation plus an inline test module; the tests stay here so the module contract and its verification live together.

use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use super::model::{FileAction, PathAccess};

// Argument keys of the builtin file tools, in Python spec order
// (`fileguard/file_tool_specs.py`).
const READ_FILE_ARGS: &[&str] = &["file_path", "path"];
const WRITE_FILE_ARGS: &[&str] = &["file_path", "path"];
const EDIT_FILE_ARGS: &[&str] = &["file_path"];
const READ_TEXT_FILE_ARGS: &[&str] = &["file_path"];
const WRITE_TEXT_FILE_ARGS: &[&str] = &["file_path"];
const WRITE_ARGS: &[&str] = &["file_path"];
const READ_ARGS: &[&str] = &["file_path"];
const SEARCH_REPLACE_ARGS: &[&str] = &["file_path", "path", "target_file"];
const GREP_ARGS: &[&str] = &["path", "file_path"];
const GLOB_FILE_SEARCH_ARGS: &[&str] = &["glob_pattern"];
const GLOB_ARGS: &[&str] = &["pattern"];
const LIST_DIR_ARGS: &[&str] = &["path"];
const LIST_FILES_ARGS: &[&str] = &["path"];

/// Redirect operator kind found while tokenizing a shell command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Redirect {
    /// `>`, `>>` or `&>` — the target is written.
    Write,
    /// `<` — the target is read.
    Read,
}

impl Redirect {
    /// The [`FileAction`] a redirect of this kind performs.
    fn action(self) -> FileAction {
        match self {
            Self::Write => FileAction::Write,
            Self::Read => FileAction::Read,
        }
    }
}

/// Token category produced by [`tokenize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    /// A word (quotes already removed, escapes already resolved).
    Word,
    /// A shell operator such as `>`, `>>`, `2>&1`, `&>` or `|`.
    Operator,
}

/// One shell token with its category.
#[derive(Debug, Clone)]
struct Token {
    /// Token text, with quotes and backslash escapes removed.
    text: String,
    /// Whether the token is a word or an operator.
    kind: TokenKind,
}

/// Action a file tool performs (the Python `FileToolSpec.action` table).
///
/// Unknown tool names yield `None`; the caller decides how to treat them.
#[must_use]
pub fn tool_action(tool_name: &str) -> Option<FileAction> {
    tool_spec(tool_name).map(|(action, _)| action)
}

/// Argument keys carrying paths for a file tool, in the Python spec order.
///
/// Unknown tool names yield an empty slice.
#[must_use]
pub fn tool_path_args(tool_name: &str) -> &'static [&'static str] {
    match tool_spec(tool_name) {
        Some((_, keys)) => keys,
        None => &[],
    }
}

/// Write-ish commands whose positional arguments are written, not read.
///
/// Covers the Python `_WRITE_CMDS`, `_TRANSFER_CMDS` and `PowerShell` write cmdlets
/// (`rm`, `mv`, `cp`, `mkdir`, `touch`, `chmod`, `chown`, `tee`, `truncate`, `ln`,
/// `Remove-Item`, `New-Item`, `Set-Content`, `Add-Content`, `Move-Item`,
/// `Copy-Item`, `Out-File`, …) plus the common `cmd.exe` aliases.
///
/// The probe accepts a full path or a bare word: the basename is taken, lowercased
/// and stripped of a `.exe` suffix. `sed`/`perl` report `false` because they only
/// write with an in-place flag; [`extract_shell_accesses`] handles that case.
#[must_use]
pub fn is_write_command(command_word: &str) -> bool {
    let word = normalize_command_word(command_word);
    matches!(
        word.as_str(),
        "rm" | "rmdir"
            | "rd"
            | "del"
            | "erase"
            | "md"
            | "mkdir"
            | "mv"
            | "move"
            | "cp"
            | "copy"
            | "touch"
            | "chmod"
            | "chown"
            | "tee"
            | "truncate"
            | "ln"
            | "link"
            | "ren"
            | "rename"
            | "xcopy"
            | "remove-item"
            | "ri"
            | "new-item"
            | "ni"
            | "set-content"
            | "add-content"
            | "move-item"
            | "copy-item"
            | "out-file"
            | "tee-object"
            | "sc"
    )
}

/// Lexical path resolution against a workspace root.
///
/// Trims whitespace and one layer of surrounding quotes, expands `~`, `$VAR` and
/// `${VAR}`, treats absolute POSIX paths and `Windows` drive paths (`C:\x`) as
/// absolute (a drive path is kept verbatim, as Python does on POSIX), joins
/// everything else onto `workspace_root`, and finally folds `.`/`..` lexically with
/// [`normalize_lexically`]. Empty input yields `None`.
#[must_use]
pub fn resolve_path(raw: &str, workspace_root: &Path) -> Option<PathBuf> {
    resolve_against(raw, workspace_root)
}

/// Path-looking value heuristic used for shell tokens.
///
/// True for a non-blank value that is not a flag and either contains `/` or `\`,
/// starts with `~`, `.` or a drive letter, or carries a file extension.
#[must_use]
pub fn is_path_like(value: &str) -> bool {
    let token = strip_quotes(value.trim());
    if token.is_empty() || token.starts_with('-') {
        return false;
    }
    if matches!(token.chars().next(), Some('~' | '.')) {
        return true;
    }
    if has_drive_prefix(token) || token.contains('/') || token.contains('\\') {
        return true;
    }
    has_file_extension(token)
}

/// Accesses of a file tool call.
///
/// Every spec key present as a non-blank string yields one access carrying the
/// tool's [`tool_action`]. An unknown tool, a non-object argument payload or a
/// missing path argument yields an empty vector.
#[must_use]
pub fn extract_file_accesses(
    tool_name: &str,
    args: &Value,
    workspace_root: &Path,
) -> Vec<PathAccess> {
    let Some((action, keys)) = tool_spec(tool_name) else {
        return Vec::new();
    };
    let Some(object) = args.as_object() else {
        return Vec::new();
    };
    let mut accesses = Vec::new();
    for key in keys {
        let Some(raw) = object.get(*key).and_then(Value::as_str) else {
            continue;
        };
        if raw.trim().is_empty() {
            continue;
        }
        if let Some(path) = resolve_path(raw, workspace_root) {
            accesses.push(PathAccess::new(path, action));
        }
    }
    accesses
}

/// Accesses of a shell command.
///
/// The command is split on unquoted `;`, `|` and `&` (keeping `>&`/`<&`/`&>`
/// together), then tokenized quote-aware. Per segment the first word is the command
/// word (basename, lowercased, `.exe` stripped):
///
/// - a write-ish command ([`is_write_command`], or `sed`/`perl` with an in-place
///   flag) yields [`FileAction::Write`] for its positional path-like arguments;
///   every other command yields [`FileAction::Read`] (`cat`, `ls`, `grep`, `find`,
///   `Get-ChildItem`, `Get-Content`, …);
/// - `>`, `>>` and `&>` followed by a target yield a [`FileAction::Write`] access,
///   `<` followed by a target a [`FileAction::Read`] access;
/// - flags, single digits, `/dev/null`, descriptor duplications (`2>&1`), URLs and
///   tokens containing `$`, `%`, backtick or a shell separator are skipped;
/// - a leading `cd` contributes no access but moves the base directory for later
///   segments (a documented approximation of the Python per-segment `cwd`).
///
/// `workdir`, when non-blank, is resolved first (an absolute workdir wins, a
/// relative one joins the workspace) and becomes the base for relative command
/// paths. Identical `(path, action)` pairs are deduplicated, preserving first-seen
/// order.
#[must_use]
pub fn extract_shell_accesses(
    command: &str,
    workdir: Option<&str>,
    workspace_root: &Path,
) -> Vec<PathAccess> {
    let text = command.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let mut base = normalize_lexically(workspace_root);
    if let Some(raw) = workdir.map(str::trim).filter(|value| !value.is_empty()) {
        if let Some(resolved) = resolve_path(raw, workspace_root) {
            base = resolved;
        }
    }

    let mut accesses = Vec::new();
    for (index, segment) in split_segments(text).iter().enumerate() {
        collect_segment_accesses(segment, index == 0, &mut base, &mut accesses);
    }
    dedupe(accesses)
}

/// Lexical path normalization: drops `.`, folds `..` into the preceding component.
///
/// A `..` that would climb above the root of an absolute path is ignored; one that
/// precedes any component of a relative path is kept. An empty result becomes `.`.
/// No filesystem access, no symlink resolution.
#[must_use]
pub fn normalize_lexically(path: &Path) -> PathBuf {
    let absolute = path.is_absolute();
    let mut normalized = PathBuf::new();
    let mut depth = 0_usize;
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if depth > 0 {
                    normalized.pop();
                    depth -= 1;
                } else if !absolute {
                    normalized.push("..");
                }
            }
            Component::Normal(part) => {
                normalized.push(part);
                depth += 1;
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return PathBuf::from(".");
    }
    normalized
}

/// Resolves `raw` against an explicit base directory (workspace or shell workdir).
fn resolve_against(raw: &str, base: &Path) -> Option<PathBuf> {
    let text = strip_quotes(raw.trim());
    if text.is_empty() {
        return None;
    }
    let expanded = expand_env_vars(&expand_tilde(text));
    if expanded.is_empty() {
        return None;
    }
    // Python keeps Windows drive paths intact on POSIX instead of resolving them.
    #[cfg(not(windows))]
    if has_drive_prefix(&expanded) {
        return Some(PathBuf::from(expanded));
    }
    let path = PathBuf::from(&expanded);
    if path.is_absolute() {
        return Some(normalize_lexically(&path));
    }
    Some(normalize_lexically(&base.join(path)))
}

/// Returns the file-tool spec of `tool_name`: its action plus its argument keys.
fn tool_spec(tool_name: &str) -> Option<(FileAction, &'static [&'static str])> {
    match tool_name {
        "read_file" => Some((FileAction::Read, READ_FILE_ARGS)),
        "write_file" => Some((FileAction::Write, WRITE_FILE_ARGS)),
        "edit_file" => Some((FileAction::Write, EDIT_FILE_ARGS)),
        "read_text_file" => Some((FileAction::Read, READ_TEXT_FILE_ARGS)),
        "write_text_file" => Some((FileAction::Write, WRITE_TEXT_FILE_ARGS)),
        "write" => Some((FileAction::Write, WRITE_ARGS)),
        "read" => Some((FileAction::Read, READ_ARGS)),
        "search_replace" => Some((FileAction::Write, SEARCH_REPLACE_ARGS)),
        "grep" => Some((FileAction::Read, GREP_ARGS)),
        "glob_file_search" => Some((FileAction::Read, GLOB_FILE_SEARCH_ARGS)),
        "glob" => Some((FileAction::Read, GLOB_ARGS)),
        "list_dir" => Some((FileAction::Read, LIST_DIR_ARGS)),
        "list_files" => Some((FileAction::Read, LIST_FILES_ARGS)),
        _ => None,
    }
}

/// Basename of a command token, lowercased, with a `.exe` suffix removed.
fn normalize_command_word(word: &str) -> String {
    let cleaned = strip_quotes(word.trim()).replace('\\', "/");
    let base = cleaned.rsplit('/').next().unwrap_or(cleaned.as_str());
    let lower = base.to_ascii_lowercase();
    lower.strip_suffix(".exe").unwrap_or(&lower).to_owned()
}

/// Strips one layer of matching single or double quotes around a trimmed value.
fn strip_quotes(raw: &str) -> &str {
    let token = raw.trim();
    let bytes = token.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        if (first == b'"' || first == b'\'') && first == bytes[bytes.len() - 1] {
            return &token[1..token.len() - 1];
        }
    }
    token
}

/// True for `C:\` / `C:/` style `Windows` drive prefixes.
fn has_drive_prefix(text: &str) -> bool {
    let mut chars = text.chars();
    matches!(
        (chars.next(), chars.next(), chars.next()),
        (Some(letter), Some(':'), Some('/' | '\\')) if letter.is_ascii_alphabetic()
    )
}

/// True when the last `name.ext` component looks like a file extension.
fn has_file_extension(text: &str) -> bool {
    let Some((stem, extension)) = text.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && !extension.is_empty()
        && extension.chars().all(|ch| ch.is_ascii_alphanumeric())
}

/// User home directory: `HOME` first, then `USERPROFILE`.
fn home_dir() -> Option<PathBuf> {
    ["HOME", "USERPROFILE"].iter().find_map(|key| {
        std::env::var_os(key)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    })
}

/// Expands a leading `~` / `~/` / `~\` to the user home directory.
///
/// Anything else (including `~user`) is returned unchanged, as is a `~` when the
/// home directory cannot be determined.
fn expand_tilde(raw: &str) -> String {
    let home = home_dir();
    if raw == "~" {
        return home.map_or_else(
            || raw.to_owned(),
            |path| path.to_string_lossy().into_owned(),
        );
    }
    let rest = raw
        .strip_prefix("~/")
        .or_else(|| raw.strip_prefix("~\\"))
        .filter(|_| home.is_some());
    if let (Some(rest), Some(mut path)) = (rest, home) {
        for part in rest.split(['/', '\\']).filter(|part| !part.is_empty()) {
            path.push(part);
        }
        return path.to_string_lossy().into_owned();
    }
    raw.to_owned()
}

/// Expands `$VAR` and `${VAR}` references, leaving unset variables untouched.
fn expand_env_vars(raw: &str) -> String {
    if !raw.contains('$') {
        return raw.to_owned();
    }
    let mut expanded = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(index) = rest.find('$') {
        expanded.push_str(&rest[..index]);
        let tail = &rest[index + 1..];
        if let Some(braced) = tail.strip_prefix('{') {
            if let Some(end) = braced.find('}') {
                let name = &braced[..end];
                let value = if name.is_empty() {
                    None
                } else {
                    std::env::var(name).ok()
                };
                if let Some(value) = value {
                    expanded.push_str(&value);
                } else {
                    expanded.push_str("${");
                    expanded.push_str(name);
                    expanded.push('}');
                }
                rest = &braced[end + 1..];
                continue;
            }
        } else {
            let name_length = tail
                .chars()
                .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                .count();
            let name = &tail[..name_length];
            if !name.is_empty() && !name.starts_with(|ch: char| ch.is_ascii_digit()) {
                if let Ok(value) = std::env::var(name) {
                    expanded.push_str(&value);
                    rest = &tail[name_length..];
                    continue;
                }
            }
        }
        expanded.push('$');
        rest = tail;
    }
    expanded.push_str(rest);
    expanded
}

/// Splits a command on unquoted `;`, `|` and `&`, keeping `>&`/`<&`/`&>` together.
fn split_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut chars = command.chars().peekable();

    while let Some(ch) = chars.next() {
        if let Some(active) = quote {
            current.push(ch);
            if ch == active {
                quote = None;
            } else if ch == '\\' && active == '"' && matches!(chars.peek(), Some('"' | '\\')) {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            '\\' => {
                current.push(ch);
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            ';' | '|' | '&' => {
                let previous = current.trim_end().chars().next_back();
                let redirection = ch == '&'
                    && (matches!(previous, Some('>' | '<')) || matches!(chars.peek(), Some('>')));
                if redirection {
                    current.push(ch);
                    continue;
                }
                if ch != ';' && matches!(chars.peek(), Some(next) if *next == ch) {
                    chars.next();
                }
                segments.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    segments.push(current);
    segments
        .into_iter()
        .map(|segment| segment.trim().to_owned())
        .filter(|segment| !segment.is_empty())
        .collect()
}

/// Splits one segment into words and operators, removing quotes and escapes.
fn tokenize(segment: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut started = false;
    let mut quote: Option<char> = None;
    let mut chars = segment.chars().peekable();

    while let Some(ch) = chars.next() {
        if let Some(active) = quote {
            if ch == active {
                quote = None;
            } else if ch == '\\' && active == '"' && matches!(chars.peek(), Some('"' | '\\')) {
                if let Some(next) = chars.next() {
                    word.push(next);
                }
                started = true;
            } else {
                word.push(ch);
                started = true;
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                started = true;
            }
            '\\' => {
                // A backslash is a literal path character here: `Windows` tools pass
                // `C:\tmp\x` unquoted, and `shlex` in `posix=False` mode (the Python
                // reference) keeps it too.
                word.push(ch);
                started = true;
            }
            ' ' | '\t' | '\n' | '\r' => push_word(&mut tokens, &mut word, &mut started),
            '>' | '<' => {
                push_word(&mut tokens, &mut word, &mut started);
                tokens.push(Token {
                    text: collect_redirect(ch, &mut chars),
                    kind: TokenKind::Operator,
                });
            }
            '&' => {
                push_word(&mut tokens, &mut word, &mut started);
                tokens.push(Token {
                    text: collect_ampersand(&mut chars),
                    kind: TokenKind::Operator,
                });
            }
            '|' => {
                push_word(&mut tokens, &mut word, &mut started);
                let mut text = String::from('|');
                if matches!(chars.peek(), Some('|')) {
                    chars.next();
                    text.push('|');
                }
                tokens.push(Token {
                    text,
                    kind: TokenKind::Operator,
                });
            }
            ';' => {
                push_word(&mut tokens, &mut word, &mut started);
                tokens.push(Token {
                    text: String::from(';'),
                    kind: TokenKind::Operator,
                });
            }
            _ => {
                word.push(ch);
                started = true;
            }
        }
    }
    push_word(&mut tokens, &mut word, &mut started);
    tokens
}

/// Pushes the pending word (when non-empty) and resets the word state.
fn push_word(tokens: &mut Vec<Token>, word: &mut String, started: &mut bool) {
    if !word.is_empty() {
        tokens.push(Token {
            text: std::mem::take(word),
            kind: TokenKind::Word,
        });
    }
    word.clear();
    *started = false;
}

/// Collects a `<`/`>` redirect operator, including `>>`, `>&2` and `<&-`.
fn collect_redirect(first: char, chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut text = String::from(first);
    if matches!(chars.peek(), Some(next) if *next == first) {
        chars.next();
        text.push(first);
    }
    if matches!(chars.peek(), Some('&')) {
        chars.next();
        text.push('&');
        while let Some(next) = chars.peek() {
            if next.is_ascii_digit() || *next == '-' {
                text.push(*next);
                chars.next();
            } else {
                break;
            }
        }
    }
    text
}

/// Collects an `&` operator, folding `&>`/`&>>` redirections into one token.
fn collect_ampersand(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut text = String::from('&');
    if matches!(chars.peek(), Some('>')) {
        chars.next();
        text.push('>');
        if matches!(chars.peek(), Some('>')) {
            chars.next();
            text.push('>');
        }
    }
    text
}

/// Classifies a redirect operator token; `None` for descriptors and heredocs.
fn redirect_action(operator: &str) -> Option<Redirect> {
    let body = operator.trim_start_matches(|ch: char| ch.is_ascii_digit());
    match body {
        ">" | ">>" | "&>" | "&>>" => Some(Redirect::Write),
        "<" => Some(Redirect::Read),
        _ => None,
    }
}

/// True for `-i`, `-i.bak`, `-ni` and `--in-place[=…]` style in-place flags.
fn is_in_place_flag(flag: &str) -> bool {
    if flag == "--in-place" || flag.starts_with("--in-place=") {
        return true;
    }
    match flag.strip_prefix('-') {
        Some(body) if !body.starts_with('-') => body.contains('i'),
        _ => false,
    }
}

/// Collects the accesses of one shell segment, moving `base` for a leading `cd`.
fn collect_segment_accesses(
    segment: &str,
    first_segment: bool,
    base: &mut PathBuf,
    accesses: &mut Vec<PathAccess>,
) {
    let tokens = tokenize(segment);
    let Some(command) = tokens.first() else {
        return;
    };
    if command.kind != TokenKind::Word {
        return;
    }
    let word = normalize_command_word(&command.text);
    if word.is_empty() {
        return;
    }
    if word == "cd" {
        if first_segment {
            let target = tokens
                .iter()
                .skip(1)
                .filter(|token| token.kind == TokenKind::Word)
                .find(|token| clean_token(&token.text).is_some());
            if let Some(target) = target {
                if let Some(resolved) = resolve_against(&target.text, base) {
                    *base = resolved;
                }
            }
        }
        return;
    }

    let script_operand = matches!(word.as_str(), "sed" | "perl");
    let in_place = script_operand
        && tokens
            .iter()
            .any(|token| token.kind == TokenKind::Word && is_in_place_flag(&token.text));
    let action = if in_place || is_write_command(&word) {
        FileAction::Write
    } else {
        FileAction::Read
    };

    let mut script_skipped = !script_operand;
    let mut consumed: Option<usize> = None;
    for (position, token) in tokens.iter().enumerate().skip(1) {
        if consumed == Some(position) {
            continue;
        }
        match token.kind {
            TokenKind::Operator => {
                let Some(redirect) = redirect_action(&token.text) else {
                    continue;
                };
                let Some(target) = tokens.get(position + 1) else {
                    continue;
                };
                if target.kind != TokenKind::Word {
                    continue;
                }
                consumed = Some(position + 1);
                let Some(path) = resolve_target(&target.text, base, false) else {
                    continue;
                };
                accesses.push(PathAccess::new(path, redirect.action()));
            }
            TokenKind::Word => {
                if !script_skipped {
                    if token.text.trim().starts_with('-') {
                        continue;
                    }
                    script_skipped = true;
                    continue;
                }
                let Some(path) = resolve_target(&token.text, base, true) else {
                    continue;
                };
                accesses.push(PathAccess::new(path, action));
            }
        }
    }
}

/// Cleans a shell token, returning `None` for values that are not static paths.
fn clean_token(raw: &str) -> Option<&str> {
    let token = strip_quotes(raw.trim());
    if token.is_empty() || token.starts_with('-') {
        return None;
    }
    if token.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if token
        .chars()
        .any(|ch| matches!(ch, ';' | '&' | '|' | '$' | '%' | '`'))
    {
        return None;
    }
    if token.eq_ignore_ascii_case("/dev/null") || token.eq_ignore_ascii_case("nul") {
        return None;
    }
    if token.contains("://") {
        return None;
    }
    Some(token)
}

/// Resolves a cleaned shell token; `require_path_shape` applies to positional args.
fn resolve_target(raw: &str, base: &Path, require_path_shape: bool) -> Option<PathBuf> {
    let token = clean_token(raw)?;
    if require_path_shape && !is_path_like(token) {
        return None;
    }
    resolve_against(token, base)
}

/// Drops repeated `(path, action)` pairs, preserving first-seen order.
fn dedupe(accesses: Vec<PathAccess>) -> Vec<PathAccess> {
    let mut unique: Vec<PathAccess> = Vec::with_capacity(accesses.len());
    for access in accesses {
        if !unique.contains(&access) {
            unique.push(access);
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use serde_json::json;
    use FileAction::{Read, Write};

    /// Workspace root used by the IO-free tests; it never needs to exist.
    const WS: &str = "/ws";

    fn ws() -> &'static Path {
        Path::new(WS)
    }

    fn path(text: &str) -> PathBuf {
        PathBuf::from(text)
    }

    /// Runs a shell extraction without a workdir.
    fn shell(command: &str) -> Vec<PathAccess> {
        extract_shell_accesses(command, None, ws())
    }

    /// Compares accesses with `(path, action)` pairs so failures stay readable.
    fn assert_accesses(actual: &[PathAccess], expected: &[(&str, FileAction)]) {
        let actual: Vec<(String, FileAction)> = actual
            .iter()
            .map(|access| (access.path.display().to_string(), access.action))
            .collect();
        let expected: Vec<(String, FileAction)> = expected
            .iter()
            .map(|(path, action)| ((*path).to_owned(), *action))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn resolve_path_handles_relative_absolute_and_quoted_forms() {
        assert_eq!(resolve_path("a/b.txt", ws()), Some(path("/ws/a/b.txt")));
        assert_eq!(resolve_path("sub/../x/./y", ws()), Some(path("/ws/x/y")));
        assert_eq!(resolve_path("../x", ws()), Some(path("/x")));
        assert_eq!(resolve_path("/etc/passwd", ws()), Some(path("/etc/passwd")));
        assert_eq!(
            resolve_path("/etc/../var/log", ws()),
            Some(path("/var/log"))
        );
        assert_eq!(
            resolve_path("  \"quoted.txt\"  ", ws()),
            Some(path("/ws/quoted.txt"))
        );
        assert_eq!(
            resolve_path("'single.txt'", ws()),
            Some(path("/ws/single.txt"))
        );
    }

    #[test]
    fn resolve_path_treats_windows_drive_paths_as_absolute() {
        assert_eq!(
            resolve_path("C:\\tmp\\x.txt", ws()),
            Some(path("C:\\tmp\\x.txt"))
        );
        assert_eq!(
            resolve_path("D:/data/x.txt", ws()),
            Some(path("D:/data/x.txt"))
        );
    }

    #[test]
    fn resolve_path_rejects_empty_input() {
        for raw in ["", "   ", "\"\""] {
            assert_eq!(resolve_path(raw, ws()), None, "raw {raw:?}");
        }
    }

    #[test]
    fn tool_action_table_matches_the_python_spec() {
        for (tool, action) in [
            ("read_file", Read),
            ("write_file", Write),
            ("edit_file", Write),
            ("read_text_file", Read),
            ("write_text_file", Write),
            ("write", Write),
            ("read", Read),
            ("search_replace", Write),
            ("grep", Read),
            ("glob_file_search", Read),
            ("glob", Read),
            ("list_dir", Read),
            ("list_files", Read),
        ] {
            assert_eq!(tool_action(tool), Some(action), "tool {tool}");
        }
        for tool in ["bash", "todo_create", ""] {
            assert_eq!(tool_action(tool), None, "tool {tool:?}");
        }
    }

    #[test]
    fn tool_path_args_table_matches_the_python_spec() {
        for (tool, keys) in [
            ("read_file", vec!["file_path", "path"]),
            ("write_file", vec!["file_path", "path"]),
            ("edit_file", vec!["file_path"]),
            ("read_text_file", vec!["file_path"]),
            ("write_text_file", vec!["file_path"]),
            ("write", vec!["file_path"]),
            ("read", vec!["file_path"]),
            ("search_replace", vec!["file_path", "path", "target_file"]),
            ("grep", vec!["path", "file_path"]),
            ("glob_file_search", vec!["glob_pattern"]),
            ("glob", vec!["pattern"]),
            ("list_dir", vec!["path"]),
            ("list_files", vec!["path"]),
        ] {
            assert_eq!(tool_path_args(tool), keys.as_slice(), "tool {tool}");
        }
        assert!(tool_path_args("bash").is_empty());
        assert!(tool_path_args("unknown_tool").is_empty());
    }

    #[test]
    fn extract_file_accesses_uses_the_tool_action() {
        assert_accesses(
            &extract_file_accesses("read_file", &json!({"file_path": "/ws/a.txt"}), ws()),
            &[("/ws/a.txt", Read)],
        );
        assert_accesses(
            &extract_file_accesses("edit_file", &json!({"file_path": "src/main.rs"}), ws()),
            &[("/ws/src/main.rs", Write)],
        );
    }

    #[test]
    fn extract_file_accesses_reads_every_spec_key() {
        assert_accesses(
            &extract_file_accesses(
                "search_replace",
                &json!({
                    "file_path": "/ws/a.txt",
                    "path": "/ws/b.txt",
                    "target_file": "/ws/c.txt",
                }),
                ws(),
            ),
            &[
                ("/ws/a.txt", Write),
                ("/ws/b.txt", Write),
                ("/ws/c.txt", Write),
            ],
        );
    }

    #[test]
    fn extract_file_accesses_ignores_non_string_and_blank_values() {
        for args in [
            json!({"file_path": 42}),
            json!({"file_path": "   ", "path": null}),
            json!({}),
        ] {
            assert!(
                extract_file_accesses("read_file", &args, ws()).is_empty(),
                "args {args}"
            );
        }
    }

    #[test]
    fn extract_file_accesses_is_empty_for_unknown_tools_and_payloads() {
        assert!(extract_file_accesses("bash", &json!({"command": "ls"}), ws()).is_empty());
        assert!(extract_file_accesses("read_file", &json!(null), ws()).is_empty());
        assert!(extract_file_accesses("read_file", &json!(["a"]), ws()).is_empty());
    }

    #[test]
    fn extract_shell_accesses_classifies_reads_and_writes() {
        assert_accesses(&shell("cat /ws/a.txt"), &[("/ws/a.txt", Read)]);
        assert_accesses(&shell("ls -la /ws/sub"), &[("/ws/sub", Read)]);
        assert_accesses(&shell("cat a.txt"), &[("/ws/a.txt", Read)]);
        assert_accesses(&shell("rm -rf /ws/b"), &[("/ws/b", Write)]);
        assert_accesses(&shell("mkdir /ws/new"), &[("/ws/new", Write)]);
    }

    #[test]
    fn extract_shell_accesses_recognizes_redirections() {
        assert_accesses(&shell("echo hi > /ws/out.txt"), &[("/ws/out.txt", Write)]);
        assert_accesses(&shell("echo hi >> /ws/out.txt"), &[("/ws/out.txt", Write)]);
        assert_accesses(&shell("cat < /ws/in.txt"), &[("/ws/in.txt", Read)]);
        assert_accesses(
            &shell("cat /ws/in.txt > /ws/out.txt"),
            &[("/ws/in.txt", Read), ("/ws/out.txt", Write)],
        );
    }

    #[test]
    fn extract_shell_accesses_ignores_descriptor_duplication_and_noise() {
        for command in [
            "ls 2>&1",
            "ls >&2",
            "cat a 2>&1 | grep a",
            "ls /dev/null",
            "ls 2",
            "echo hi >&-",
        ] {
            assert!(shell(command).is_empty(), "command {command:?}");
        }
    }

    #[test]
    fn extract_shell_accesses_skips_unresolvable_tokens() {
        for command in [
            "cat $HOME/secret.txt",
            "cat ${HOME}/secret.txt",
            "cat `pwd`/x.txt",
            "cat %USERPROFILE%\\x.txt",
            "curl https://example.com/x",
        ] {
            assert!(shell(command).is_empty(), "command {command:?}");
        }
    }

    #[test]
    fn extract_shell_accesses_resolves_relative_to_the_workdir() {
        assert_accesses(
            &extract_shell_accesses("cat ../outside/x", Some("sub"), ws()),
            &[("/ws/outside/x", Read)],
        );
        assert_accesses(
            &extract_shell_accesses("cat ../x", Some("/other"), ws()),
            &[("/x", Read)],
        );
        assert_accesses(
            &extract_shell_accesses("cat x.txt", Some("  "), ws()),
            &[("/ws/x.txt", Read)],
        );
    }

    #[test]
    fn extract_shell_accesses_deduplicates_pairs() {
        assert_accesses(&shell("cat /ws/a.txt /ws/a.txt"), &[("/ws/a.txt", Read)]);
        assert_accesses(
            &shell("cat /ws/a.txt; rm /ws/a.txt"),
            &[("/ws/a.txt", Read), ("/ws/a.txt", Write)],
        );
    }

    #[test]
    fn extract_shell_accesses_never_emits_shell_syntax() {
        assert!(shell("bash -c 'ls a; rm -rf b; echo done'").is_empty());
        let compound = shell("cat /ws/a.txt; rm -rf /ws/b");
        assert_accesses(&compound, &[("/ws/a.txt", Read), ("/ws/b", Write)]);
        for access in &compound {
            let text = access.path.to_string_lossy();
            assert!(
                !text.contains([';', '&', '|', '$']),
                "shell syntax leaked into {text}"
            );
        }
    }

    #[test]
    fn extract_shell_accesses_moves_the_base_for_a_leading_cd() {
        assert_accesses(
            &shell("cd /ws/sub && cat x.txt"),
            &[("/ws/sub/x.txt", Read)],
        );
        assert!(shell("cd /ws/sub").is_empty(), "cd alone is not an access");
        assert_accesses(
            &shell("ls /ws; cd /tmp; cat x.txt"),
            &[("/ws", Read), ("/ws/x.txt", Read)],
        );
    }

    #[test]
    fn extract_shell_accesses_handles_in_place_editors() {
        assert_accesses(&shell("sed -i 's/a/b/' /ws/f.txt"), &[("/ws/f.txt", Write)]);
        assert_accesses(&shell("sed 's/a/b/' /ws/f.txt"), &[("/ws/f.txt", Read)]);
        assert_accesses(&shell("sed -ni s/a/b/ /ws/f.txt"), &[("/ws/f.txt", Write)]);
    }

    #[test]
    fn extract_shell_accesses_handles_windows_commands() {
        assert_accesses(
            &shell("Remove-Item C:\\tmp\\x.txt"),
            &[("C:\\tmp\\x.txt", Write)],
        );
        assert_accesses(
            &shell("Get-Content C:\\tmp\\x.txt"),
            &[("C:\\tmp\\x.txt", Read)],
        );
        // Quoted `Windows` paths keep their backslashes.
        assert_accesses(
            &shell("Get-Content \"C:\\tmp\\x.txt\""),
            &[("C:\\tmp\\x.txt", Read)],
        );
    }

    #[test]
    fn is_write_command_covers_write_and_transfer_commands() {
        for word in [
            "rm",
            "rmdir",
            "mv",
            "cp",
            "mkdir",
            "touch",
            "chmod",
            "chown",
            "tee",
            "truncate",
            "ln",
            "Remove-Item",
            "New-Item",
            "Set-Content",
            "Add-Content",
            "Move-Item",
            "Copy-Item",
            "Out-File",
        ] {
            assert!(is_write_command(word), "{word} should be a write command");
        }
        for word in ["cat", "ls", "grep", "find", "sed", "Get-ChildItem", ""] {
            assert!(
                !is_write_command(word),
                "{word} should not be a write command"
            );
        }
        assert!(is_write_command("C:\\Windows\\System32\\rm.exe"));
        assert!(is_write_command("/bin/MV"));
    }

    #[test]
    fn is_path_like_recognizes_path_shapes() {
        for value in [
            "a.txt", "sub/dir", "sub\\dir", "./x", "../x", "~", "~/x", ".hidden", "C:\\x", "/abs",
        ] {
            assert!(is_path_like(value), "{value} should look like a path");
        }
        for value in ["", "   ", "-x", "--flag", "word", "2", "&&", "2>&1"] {
            assert!(!is_path_like(value), "{value} should not look like a path");
        }
    }

    #[test]
    fn normalize_lexically_folds_dot_and_dot_dot() {
        assert_eq!(
            normalize_lexically(Path::new("/ws/./a/../b")),
            path("/ws/b")
        );
        assert_eq!(normalize_lexically(Path::new("relative/../x")), path("x"));
        assert_eq!(normalize_lexically(Path::new("../x")), path("../x"));
        assert_eq!(normalize_lexically(Path::new("/ws/../..")), path("/"));
        assert_eq!(normalize_lexically(Path::new(".")), path("."));
        assert_eq!(normalize_lexically(Path::new("/ws/a/")), path("/ws/a"));
    }
}
