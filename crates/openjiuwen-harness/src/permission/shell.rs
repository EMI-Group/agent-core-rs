//! Conservative shell command analysis for permission decisions.
//!
//! Ports `permission_engine/toolguard/command_canonicalize.py` and the fallback
//! scanner of `toolguard/shell_ast.py`. The Python reference prefers tree-sitter;
//! this module replaces it with a lexer that fails closed: anything it cannot
//! prove simple is reported as [`ShellKind::TooComplex`] or
//! [`ShellKind::ParseUnavailable`] so downstream policy widens to `ask`.
//!
//! Only `std` is used here; the module has no dependencies on other `permission`
//! modules, which keeps it usable from every layer of the engine.
#![allow(clippy::module_name_repetitions)]

/// Interpreter basenames that make a pipeline sink dangerous.
///
/// Mirrors `_INTERPRETER_SINK_NAMES` in `toolguard/tool_policy.py`: a command
/// piped into one of these interpreters executes arbitrary payloads.
pub const INTERPRETER_SINKS: &[&str] = &[
    "bash",
    "sh",
    "zsh",
    "dash",
    "ash",
    "fish",
    "python",
    "python3",
    "pythonw",
    "py",
    "node",
    "nodejs",
    "ruby",
    "perl",
    "pwsh",
    "powershell",
    "cmd",
    "iex",
    "invoke-expression",
    "invoke-command",
];

/// Outcome of analysing a shell command for permission purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellKind {
    /// The lexer proved the command simple; `subcommands` is trustworthy.
    Simple,
    /// The command contains structure the permission view refuses to trust.
    TooComplex,
    /// The lexer could not tokenize the command safely (quotes/escapes).
    ParseUnavailable,
}

/// Structural flags discovered by the conservative scanner.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShellFlags {
    /// `&&`, `||` or `;` appeared outside quotes.
    pub compound: bool,
    /// A single `|` (not `||`) appeared outside quotes.
    pub pipeline: bool,
    /// `$(`, a backtick or `${` appeared outside single quotes.
    pub substitution: bool,
    /// `>` or `<` appeared outside quotes after fd-alias stripping.
    pub redirect: bool,
    /// A `(...)` or `{ ... }` group appeared outside quotes.
    pub subshell: bool,
    /// `<<` appeared outside single quotes.
    pub heredoc: bool,
}

impl ShellFlags {
    /// Returns `true` when any structural flag is set.
    #[must_use]
    pub fn has_risky_structure(&self) -> bool {
        self.compound
            || self.pipeline
            || self.substitution
            || self.redirect
            || self.subshell
            || self.heredoc
    }
}

/// Permission view of a shell command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    /// Confidence level of the analysis.
    pub kind: ShellKind,
    /// The canonicalized command (one launcher layer unwrapped).
    pub canonical: String,
    /// Top-level commands in order; empty unless [`ShellKind::Simple`].
    pub subcommands: Vec<String>,
    /// Structural flags discovered by the scanner.
    pub flags: ShellFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quote {
    None,
    Single,
    Double,
}

/// Returns a permission-view copy of `command`, unwrapping one launcher layer.
///
/// Recognised launchers are `cmd /c` (or `cmd //c`), `pwsh`/`powershell
/// -Command|-c`, and POSIX shells (`bash`, `sh`, `zsh`, `dash`, `ash`) invoked
/// with a `-c`-style switch group such as `-lc`. One layer of surrounding
/// quotes is removed from the payload, and leading shell separators left in
/// front of the unwrapped command are dropped.
///
/// The command used for execution is never rewritten.
#[must_use]
pub fn canonicalize_shell_command(command: &str) -> String {
    let text = command.trim();
    if text.is_empty() {
        return String::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let Some(payload) = unwrap_launcher(&chars) else {
        return text.to_string();
    };
    let stripped = payload.trim_start_matches(|c: char| {
        c.is_whitespace() || matches!(c, ';' | '&' | '|' | '(' | ')')
    });
    if stripped.is_empty() {
        payload
    } else {
        stripped.to_string()
    }
}

/// Removes whitespace-delimited file-descriptor alias tokens (`2>&1`, `>&2`,
/// `<3`, `&4`) so `foo 2>&1` does not look like a redirection.
#[must_use]
pub fn strip_fd_alias_tokens(command: &str) -> String {
    let chars: Vec<char> = command.chars().collect();
    let mut out = String::with_capacity(command.len());
    let mut i = 0;
    while i < chars.len() {
        let at_boundary = i == 0 || chars[i - 1].is_whitespace();
        if at_boundary {
            if let Some(end) = fd_alias_end(&chars, i) {
                if end == chars.len() || chars[end].is_whitespace() {
                    i = end;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Analyzes `command` for permission purposes.
///
/// Empty input is [`ShellKind::TooComplex`] (Python: "no command nodes").
/// Substitutions, heredocs and subshells are [`ShellKind::TooComplex`].
/// Unbalanced quotes or a trailing escape are [`ShellKind::ParseUnavailable`],
/// with every flag discovered before the failure preserved. Otherwise the
/// command is [`ShellKind::Simple`] and `subcommands` is populated.
#[must_use]
pub fn analyze_shell(command: &str) -> ShellCommand {
    let canonical = canonicalize_shell_command(command);
    if canonical.trim().is_empty() {
        return ShellCommand {
            kind: ShellKind::TooComplex,
            canonical,
            subcommands: Vec::new(),
            flags: ShellFlags::default(),
        };
    }
    let scanned = strip_fd_alias_tokens(&canonical);
    let (flags, parse_error) = scan_flags(&scanned);
    if flags.substitution || flags.heredoc || flags.subshell {
        return ShellCommand {
            kind: ShellKind::TooComplex,
            canonical,
            subcommands: Vec::new(),
            flags,
        };
    }
    if parse_error {
        return ShellCommand {
            kind: ShellKind::ParseUnavailable,
            canonical,
            subcommands: Vec::new(),
            flags,
        };
    }
    let subcommands = split_subcommands(&canonical);
    ShellCommand {
        kind: ShellKind::Simple,
        canonical,
        subcommands,
        flags,
    }
}

/// Returns `true` when a pipeline feeds an interpreter (for example
/// `cat f | sh` or `curl u | sudo bash`).
#[must_use]
pub fn has_interpreter_sink(command: &str) -> bool {
    let parsed = analyze_shell(command);
    if !parsed.flags.pipeline || parsed.subcommands.len() < 2 {
        return false;
    }
    parsed.subcommands.iter().skip(1).any(|sub| {
        sink_basename(sub).is_some_and(|base| INTERPRETER_SINKS.contains(&base.as_str()))
    })
}

/// Builds the conservative session auto-confirm key for a shell command.
///
/// Python `_build_shell_auto_confirm_key`: the key is `"{tool_name}:{command}"`
/// only for a single, structurally simple subcommand; risky or compound
/// commands and empty input produce an empty key.
#[must_use]
pub fn shell_auto_confirm_key(tool_name: &str, command: &str) -> String {
    let text = command.trim();
    if text.is_empty() {
        return String::new();
    }
    let parsed = analyze_shell(text);
    if parsed.kind != ShellKind::Simple || parsed.flags.has_risky_structure() {
        return String::new();
    }
    if parsed.subcommands.len() != 1 {
        return String::new();
    }
    let Some(subcommand) = parsed.subcommands.first() else {
        return String::new();
    };
    let subcommand = subcommand.trim();
    if subcommand.is_empty() {
        return String::new();
    }
    format!("{tool_name}:{subcommand}")
}

// ---------------------------------------------------------------------------
// Canonicalization helpers
// ---------------------------------------------------------------------------

/// Skips whitespace starting at `pos`, returning the next character index.
fn skip_ws(chars: &[char], mut pos: usize) -> usize {
    while pos < chars.len() && chars[pos].is_whitespace() {
        pos += 1;
    }
    pos
}

/// Reads one whitespace-delimited word starting at `pos`, unquoting it.
///
/// Returns the unquoted value and the index just past the word. Unterminated
/// quotes consume the rest of the input.
fn next_word(chars: &[char], pos: usize) -> Option<(String, usize)> {
    let mut i = skip_ws(chars, pos);
    if i >= chars.len() {
        return None;
    }
    let mut value = String::new();
    while i < chars.len() && !chars[i].is_whitespace() {
        match chars[i] {
            '\'' => {
                i += 1;
                while i < chars.len() && chars[i] != '\'' {
                    value.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    i += 1;
                }
            }
            '"' => {
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        value.push(chars[i + 1]);
                        i += 2;
                    } else {
                        value.push(chars[i]);
                        i += 1;
                    }
                }
                if i < chars.len() {
                    i += 1;
                }
            }
            '\\' => {
                if i + 1 < chars.len() {
                    value.push(chars[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            c => {
                value.push(c);
                i += 1;
            }
        }
    }
    Some((value, i))
}

/// Removes one layer of matching single or double quotes from `text`.
fn strip_one_quote_layer(text: &str) -> String {
    let mut chars = text.chars();
    let (Some(first), Some(last)) = (chars.next(), chars.next_back()) else {
        return text.to_string();
    };
    if first.len_utf8() + last.len_utf8() > text.len() {
        return text.to_string();
    }
    if (first == '"' || first == '\'') && first == last {
        text[first.len_utf8()..text.len() - last.len_utf8()].to_string()
    } else {
        text.to_string()
    }
}

/// Returns the trimmed, unquoted payload that follows a launcher switch.
fn payload_after(chars: &[char], pos: usize) -> String {
    let rest: String = chars[pos..].iter().collect();
    let trimmed = rest.trim();
    strip_one_quote_layer(trimmed).trim().to_string()
}

/// Returns the lowercased final path component of `token`, without `.exe`.
fn command_basename(token: &str) -> String {
    let normalized = token.replace('\\', "/");
    let base = normalized.rsplit('/').next().unwrap_or_default();
    let base = base.trim_matches(|c: char| c == '"' || c == '\'');
    let lower = base.to_ascii_lowercase();
    let stem = lower.strip_suffix(".exe").unwrap_or(lower.as_str());
    stem.to_string()
}

/// Returns `true` for a `-Command`/`-c` style PowerShell switch.
fn is_powershell_command_switch(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    lower == "-command" || lower == "-c"
}

/// Returns `true` for a generic `-x` / `--long-option` switch.
fn is_generic_switch(word: &str) -> bool {
    let Some(rest) = word.strip_prefix('-') else {
        return false;
    };
    let rest = rest.strip_prefix('-').unwrap_or(rest);
    let Some(first) = rest.chars().next() else {
        return false;
    };
    first.is_ascii_alphabetic() && rest.chars().all(|c| c.is_ascii_alphabetic() || c == '-')
}

/// Returns `true` for a single-dash POSIX shell switch group containing `c`.
fn is_shell_c_switch(word: &str) -> bool {
    let Some(rest) = word.strip_prefix('-') else {
        return false;
    };
    if rest.is_empty() || rest.starts_with('-') {
        return false;
    }
    rest.chars().all(|c| c.is_ascii_alphabetic())
        && rest.chars().any(|c| c.eq_ignore_ascii_case(&'c'))
}

/// Unwraps exactly one recognised launcher layer, if present at the start.
fn unwrap_launcher(chars: &[char]) -> Option<String> {
    let (first, first_end) = next_word(chars, 0)?;
    let base = command_basename(&first);
    if base == "cmd" {
        let (switch, switch_end) = next_word(chars, first_end)?;
        let switch = switch.to_ascii_lowercase();
        if switch == "/c" || switch == "//c" {
            return Some(payload_after(chars, switch_end));
        }
        return None;
    }
    if base == "pwsh" || base == "powershell" {
        return unwrap_powershell(chars, first_end);
    }
    if matches!(base.as_str(), "bash" | "sh" | "zsh" | "dash" | "ash") {
        return unwrap_posix_shell(chars, first_end);
    }
    None
}

/// Unwraps `pwsh`/`powershell [-switches] -Command|-c <payload>`.
fn unwrap_powershell(chars: &[char], start: usize) -> Option<String> {
    let mut pos = start;
    loop {
        let (word, end) = next_word(chars, pos)?;
        if is_powershell_command_switch(&word) {
            return Some(payload_after(chars, end));
        }
        if !is_generic_switch(&word) {
            return None;
        }
        pos = end;
    }
}

/// Unwraps `bash|sh|zsh|dash|ash [-switches] -<flags containing c> <payload>`.
fn unwrap_posix_shell(chars: &[char], start: usize) -> Option<String> {
    let mut pos = start;
    loop {
        let (word, end) = next_word(chars, pos)?;
        if is_shell_c_switch(&word) {
            return Some(payload_after(chars, end));
        }
        if !is_generic_switch(&word) {
            return None;
        }
        pos = end;
    }
}

// ---------------------------------------------------------------------------
// fd alias stripping
// ---------------------------------------------------------------------------

/// Returns the end of an ASCII digit run starting at `pos`.
fn ascii_digits_end(chars: &[char], mut pos: usize) -> usize {
    while pos < chars.len() && chars[pos].is_ascii_digit() {
        pos += 1;
    }
    pos
}

/// Matches an fd-alias token at `start`; returns its exclusive end index.
fn fd_alias_end(chars: &[char], start: usize) -> Option<usize> {
    let digits1 = ascii_digits_end(chars, start);
    if digits1 > start && chars.get(digits1) == Some(&'>') && chars.get(digits1 + 1) == Some(&'&') {
        let digits2 = ascii_digits_end(chars, digits1 + 2);
        if digits2 > digits1 + 2 {
            return Some(digits2);
        }
    }
    if chars.get(start) == Some(&'>') && chars.get(start + 1) == Some(&'&') {
        let digits = ascii_digits_end(chars, start + 2);
        if digits > start + 2 {
            return Some(digits);
        }
    }
    if chars.get(start) == Some(&'<') {
        let digits = ascii_digits_end(chars, start + 1);
        if digits > start + 1 {
            return Some(digits);
        }
    }
    if chars.get(start) == Some(&'&') {
        let digits = ascii_digits_end(chars, start + 1);
        if digits > start + 1 {
            return Some(digits);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Structure scanning
// ---------------------------------------------------------------------------

/// Scans `text` for shell structure, respecting quotes and backslash escapes.
///
/// Returns the discovered flags plus whether tokenization failed (unbalanced
/// quote or trailing escape).
fn scan_flags(text: &str) -> (ShellFlags, bool) {
    let chars: Vec<char> = text.chars().collect();
    let mut flags = ShellFlags::default();
    let mut quote = Quote::None;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        match quote {
            Quote::Single => {
                if c == '\'' {
                    quote = Quote::None;
                }
                i += 1;
            }
            Quote::Double => {
                match c {
                    '\\' => escaped = true,
                    '"' => quote = Quote::None,
                    '`' => flags.substitution = true,
                    '$' if matches!(next, Some('(' | '{')) => flags.substitution = true,
                    _ => {}
                }
                i += 1;
            }
            Quote::None => match c {
                '\\' => {
                    escaped = true;
                    i += 1;
                }
                '\'' => {
                    quote = Quote::Single;
                    i += 1;
                }
                '"' => {
                    quote = Quote::Double;
                    i += 1;
                }
                '`' => {
                    flags.substitution = true;
                    i += 1;
                }
                '$' => {
                    if matches!(next, Some('(' | '{')) {
                        flags.substitution = true;
                    }
                    i += 1;
                }
                '&' => {
                    if next == Some('&') {
                        flags.compound = true;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                '|' => {
                    if next == Some('|') {
                        flags.compound = true;
                        i += 2;
                    } else {
                        flags.pipeline = true;
                        i += 1;
                    }
                }
                ';' => {
                    flags.compound = true;
                    i += 1;
                }
                '<' => {
                    flags.redirect = true;
                    if next == Some('<') {
                        flags.heredoc = true;
                    }
                    i += 1;
                }
                '>' => {
                    flags.redirect = true;
                    i += 1;
                }
                '(' | ')' | '{' | '}' => {
                    flags.subshell = true;
                    i += 1;
                }
                _ => i += 1,
            },
        }
    }
    let parse_error = quote != Quote::None || escaped;
    (flags, parse_error)
}

// ---------------------------------------------------------------------------
// Subcommand splitting
// ---------------------------------------------------------------------------

/// Pushes a trimmed, unquoted piece unless it is empty.
fn push_piece(pieces: &mut Vec<String>, current: &str) {
    let trimmed = current.trim();
    let stripped = strip_one_quote_layer(trimmed);
    let piece = stripped.trim();
    if !piece.is_empty() {
        pieces.push(piece.to_string());
    }
}

/// Splits a command on unquoted top-level `&&`, `||`, `;` and `|`.
fn split_subcommands(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut quote = Quote::None;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if escaped {
            current.push(c);
            escaped = false;
            i += 1;
            continue;
        }
        match quote {
            Quote::Single => {
                current.push(c);
                if c == '\'' {
                    quote = Quote::None;
                }
                i += 1;
            }
            Quote::Double => {
                current.push(c);
                match c {
                    '\\' => escaped = true,
                    '"' => quote = Quote::None,
                    _ => {}
                }
                i += 1;
            }
            Quote::None => match c {
                '\\' => {
                    current.push(c);
                    escaped = true;
                    i += 1;
                }
                '\'' => {
                    quote = Quote::Single;
                    current.push(c);
                    i += 1;
                }
                '"' => {
                    quote = Quote::Double;
                    current.push(c);
                    i += 1;
                }
                '&' if next == Some('&') => {
                    push_piece(&mut pieces, &current);
                    current.clear();
                    i += 2;
                }
                '|' if next == Some('|') => {
                    push_piece(&mut pieces, &current);
                    current.clear();
                    i += 2;
                }
                '|' | ';' => {
                    push_piece(&mut pieces, &current);
                    current.clear();
                    i += 1;
                }
                _ => {
                    current.push(c);
                    i += 1;
                }
            },
        }
    }
    push_piece(&mut pieces, &current);
    pieces
}

// ---------------------------------------------------------------------------
// Interpreter sinks
// ---------------------------------------------------------------------------

/// Returns the effective first-word basename of a subcommand, skipping a
/// leading `sudo`/`doas` prefix and its options.
fn sink_basename(subcommand: &str) -> Option<String> {
    let mut words = subcommand.split_whitespace();
    let first = words.next()?;
    let mut base = command_basename(first);
    if base == "sudo" || base == "doas" {
        let mut candidate = words.next();
        let mut found = None;
        while let Some(word) = candidate {
            if word.starts_with('-') {
                candidate = words.next();
            } else {
                found = Some(command_basename(word));
                break;
            }
        }
        base = found?;
    }
    Some(base)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        analyze_shell, canonicalize_shell_command, has_interpreter_sink, shell_auto_confirm_key,
        strip_fd_alias_tokens, ShellFlags, ShellKind,
    };

    #[test]
    fn canonicalize_unwraps_launchers() {
        assert_eq!(canonicalize_shell_command("cmd /c dir"), "dir");
        assert_eq!(canonicalize_shell_command("  cmd /C  dir  "), "dir");
        assert_eq!(canonicalize_shell_command("cmd //c dir"), "dir");
        assert_eq!(
            canonicalize_shell_command("pwsh -Command \"Get-Item\""),
            "Get-Item"
        );
        assert_eq!(
            canonicalize_shell_command("powershell -NoProfile -c 'Get-Item'"),
            "Get-Item"
        );
        assert_eq!(canonicalize_shell_command("bash -lc \"ls -la\""), "ls -la");
        assert_eq!(canonicalize_shell_command("sh -c 'ls'"), "ls");
        assert_eq!(
            canonicalize_shell_command("/bin/bash -ec \"ls -la\""),
            "ls -la"
        );
    }

    #[test]
    fn canonicalize_leaves_plain_commands_unchanged() {
        assert_eq!(canonicalize_shell_command("git status"), "git status");
        assert_eq!(
            canonicalize_shell_command("echo bash -c 'rm'"),
            "echo bash -c 'rm'"
        );
        assert_eq!(canonicalize_shell_command("  ls   -la  "), "ls   -la");
        assert_eq!(
            canonicalize_shell_command("bash script.sh"),
            "bash script.sh"
        );
        assert_eq!(canonicalize_shell_command(""), "");
    }

    #[test]
    fn fd_alias_tokens_are_stripped() {
        assert_eq!(strip_fd_alias_tokens("ls 2>&1"), "ls ");
        assert_eq!(strip_fd_alias_tokens("cmd >&2"), "cmd ");
        assert_eq!(strip_fd_alias_tokens("cmd <3"), "cmd ");
        assert_eq!(strip_fd_alias_tokens("cmd &4"), "cmd ");
        // Not whitespace delimited: Python leaves these alone.
        assert_eq!(strip_fd_alias_tokens("echo 2>&1;"), "echo 2>&1;");
        let parsed = analyze_shell("ls 2>&1");
        assert!(!parsed.flags.redirect);
        assert!(!parsed.flags.has_risky_structure());
        assert_eq!(parsed.kind, ShellKind::Simple);
    }

    #[test]
    fn flags_are_detected_per_operator() {
        let compound = analyze_shell("git status && git diff").flags;
        assert!(compound.compound && !compound.pipeline);
        assert!(analyze_shell("a || b").flags.compound);
        assert!(analyze_shell("a; b").flags.compound);

        let pipeline = analyze_shell("ls | grep x").flags;
        assert!(pipeline.pipeline && !pipeline.compound);

        assert!(analyze_shell("echo hi > f").flags.redirect);
        assert!(analyze_shell("wc -l < f").flags.redirect);

        assert!(analyze_shell("echo $(id)").flags.substitution);
        assert!(analyze_shell("echo ${HOME}").flags.substitution);
        assert!(analyze_shell("echo `id`").flags.substitution);

        assert!(analyze_shell("cat <<EOF").flags.heredoc);
        assert!(analyze_shell("(cd /tmp && ls)").flags.subshell);

        // Quoted operators are inert.
        let quoted = analyze_shell("echo 'a && b | c > d'").flags;
        assert_eq!(quoted, ShellFlags::default());
    }

    #[test]
    fn complex_structure_widens_to_too_complex() {
        assert_eq!(analyze_shell("echo $(id)").kind, ShellKind::TooComplex);
        assert_eq!(analyze_shell("echo `id`").kind, ShellKind::TooComplex);
        assert_eq!(analyze_shell("cat <<EOF").kind, ShellKind::TooComplex);
        assert_eq!(analyze_shell("(cd /tmp)").kind, ShellKind::TooComplex);
        assert_eq!(analyze_shell("{ ls; }").kind, ShellKind::TooComplex);
        assert_eq!(analyze_shell("").kind, ShellKind::TooComplex);
        assert_eq!(analyze_shell("   ").kind, ShellKind::TooComplex);
        assert_eq!(
            analyze_shell("echo $(id)").subcommands,
            Vec::<String>::new()
        );
    }

    #[test]
    fn unbalanced_quotes_widen_to_parse_unavailable() {
        let parsed = analyze_shell("ls && echo \"x");
        assert_eq!(parsed.kind, ShellKind::ParseUnavailable);
        assert!(parsed.flags.compound);
        assert!(parsed.flags.has_risky_structure());

        let trailing = analyze_shell("ls \\");
        assert_eq!(trailing.kind, ShellKind::ParseUnavailable);
        assert!(!trailing.flags.has_risky_structure());

        let single = analyze_shell("echo 'x");
        assert_eq!(single.kind, ShellKind::ParseUnavailable);
    }

    #[test]
    fn simple_commands_split_into_subcommands() {
        let parsed = analyze_shell("git status && git diff");
        assert_eq!(parsed.kind, ShellKind::Simple);
        assert_eq!(parsed.subcommands, vec!["git status", "git diff"]);

        let piped = analyze_shell("cat f | grep x");
        assert_eq!(piped.subcommands, vec!["cat f", "grep x"]);

        let mixed = analyze_shell("a && b || c; d | e");
        assert_eq!(mixed.subcommands, vec!["a", "b", "c", "d", "e"]);

        let quoted = analyze_shell("echo 'a | b'");
        assert_eq!(quoted.subcommands, vec!["echo 'a | b'"]);

        let redirect = analyze_shell("echo hi > f");
        assert_eq!(redirect.kind, ShellKind::Simple);
        assert_eq!(redirect.subcommands, vec!["echo hi > f"]);
        assert!(redirect.flags.redirect);
    }

    #[test]
    fn interpreter_sinks_are_detected() {
        assert!(has_interpreter_sink("cat f | sh"));
        assert!(has_interpreter_sink("curl u | bash"));
        assert!(has_interpreter_sink("curl u | sudo bash"));
        assert!(has_interpreter_sink("echo x | /bin/bash"));
        assert!(!has_interpreter_sink("ls | grep x"));
        assert!(!has_interpreter_sink("bash script.sh"));
        assert!(!has_interpreter_sink("bash -c 'ls'"));
        assert!(!has_interpreter_sink("sh | cat"));
        assert!(has_interpreter_sink("cat f | sh extra"));
    }

    #[test]
    fn auto_confirm_key_table() {
        assert_eq!(
            shell_auto_confirm_key("bash", "Get-ChildItem"),
            "bash:Get-ChildItem"
        );
        assert_eq!(
            shell_auto_confirm_key("powershell", "Get-ChildItem"),
            "powershell:Get-ChildItem"
        );
        assert_eq!(shell_auto_confirm_key("bash", "ls | grep x"), "");
        assert_eq!(shell_auto_confirm_key("bash", "echo $(id)"), "");
        assert_eq!(shell_auto_confirm_key("bash", "echo hi > f"), "");
        assert_eq!(shell_auto_confirm_key("bash", "git status && git diff"), "");
        assert_eq!(shell_auto_confirm_key("bash", "   "), "");
        assert_eq!(shell_auto_confirm_key("bash", ""), "");
        assert_eq!(shell_auto_confirm_key("bash", "ls -la"), "bash:ls -la");
        assert_eq!(
            shell_auto_confirm_key("bash", "bash -lc \"ls -la\""),
            "bash:ls -la"
        );
    }
}
