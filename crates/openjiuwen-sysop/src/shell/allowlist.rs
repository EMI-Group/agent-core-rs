//! First-token allowlist and regex deny list for shell commands.
//!
//! Python parity: `_check_command_safety` / `_check_allowlist` in
//! `openjiuwen/core/sys_operation/local/shell_operation.py`.
//!
//! Two rules are reproduced deliberately because they are easy to get wrong:
//!
//! - The allowlist matches the **first token** of the command line, so `rm -rf /tmp/x` is
//!   rejected when the allowlist does not contain `rm`. A `None` or empty allowlist allows
//!   everything (Python's default `LocalWorkConfig.shell_allowlist` is a 30-entry list).
//! - A token that is immediately followed by more text (`"echo"-x`) is rejected instead of
//!   being treated as `echo`: the character after the token must be whitespace.
//!
//! The deny list is **opt-in**: Python's built-in table is empty, so a command such as
//! `rm -rf /` passes unless the caller configures
//! [`crate::config::LocalWorkConfig::dangerous_patterns`].
//!
//! ```
//! use openjiuwen_sysop::shell::allowlist::CommandFilter;
//!
//! let filter = CommandFilter::from_parts(Some(vec!["echo".to_string()]), None).unwrap();
//! assert!(filter.allows_command("echo hi"));
//! assert!(!filter.allows_command("rm -rf /tmp/x"));
//! assert_eq!(CommandFilter::first_token("echo hi").as_deref(), Some("echo"));
//! ```

use openjiuwen_common::error::{Error, Result};
use regex::{Regex, RegexBuilder};

use crate::config::LocalWorkConfig;

/// First-token allowlist plus the optional case-insensitive regex deny list.
#[derive(Debug, Clone)]
pub struct CommandFilter {
    allowlist: Option<Vec<String>>,
    normalized: Vec<String>,
    dangerous: Option<Vec<(String, Regex)>>,
}

impl CommandFilter {
    /// Builds a filter from `config`, compiling and validating `dangerous_patterns`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when a dangerous
    /// pattern is not a valid regular expression.
    pub fn new(config: &LocalWorkConfig) -> Result<Self> {
        Self::from_parts(
            config.shell_allowlist.clone(),
            config.dangerous_patterns.clone(),
        )
    }

    /// Builds a filter from explicit parts.
    ///
    /// An absent (`None`) or empty allowlist allows every command; an absent (`None`) deny
    /// list disables the safety check (Python's default). Entries are kept verbatim so
    /// [`Self::allowlist`] reports exactly what the caller configured.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when a dangerous
    /// pattern is not a valid regular expression.
    pub fn from_parts(
        allowlist: Option<Vec<String>>,
        dangerous_patterns: Option<Vec<String>>,
    ) -> Result<Self> {
        let normalized = allowlist
            .as_ref()
            .map(|entries| entries.iter().map(|entry| normalize(entry)).collect())
            .unwrap_or_default();
        let dangerous = match dangerous_patterns {
            Some(patterns) => {
                let mut compiled = Vec::with_capacity(patterns.len());
                for pattern in patterns {
                    // Python uses `re.search(pattern, command, re.IGNORECASE)`.
                    let regex = RegexBuilder::new(&pattern)
                        .case_insensitive(true)
                        .build()
                        .map_err(|error| {
                            Error::validation(format!(
                                "invalid dangerous pattern '{pattern}': {error}"
                            ))
                        })?;
                    compiled.push((pattern, regex));
                }
                Some(compiled)
            }
            None => None,
        };
        Ok(Self {
            allowlist,
            normalized,
            dangerous,
        })
    }

    /// The allowlist exactly as configured (`None` = every command is allowed).
    #[must_use]
    pub fn allowlist(&self) -> Option<&[String]> {
        self.allowlist.as_deref()
    }

    /// Returns `true` when `command`'s first token is allowlisted.
    ///
    /// A `None` or empty allowlist allows everything. A command whose first token cannot
    /// be parsed (for example an unterminated quote) or whose token is immediately
    /// followed by more text is rejected.
    #[must_use]
    pub fn allows_command(&self, command: &str) -> bool {
        if self.allows_everything() {
            return true;
        }
        match Self::first_token(command) {
            Some(token) => self.matches_entry(&token),
            None => false,
        }
    }

    /// Applies the allowlist rule to a bare program path (used by `execute_argv` on
    /// `argv[0]`).
    #[must_use]
    pub fn allows_program(&self, program: &str) -> bool {
        self.allows_everything() || self.matches_entry(program)
    }

    /// Returns the first dangerous pattern matching `command` (case-insensitive), if any.
    ///
    /// The returned string is the raw pattern as configured, matching Python's
    /// `_check_command_safety` label.
    #[must_use]
    pub fn dangerous_match(&self, command: &str) -> Option<&str> {
        self.dangerous.as_ref().and_then(|patterns| {
            patterns
                .iter()
                .find(|(_, regex)| regex.is_match(command))
                .map(|(raw, _)| raw.as_str())
        })
    }

    /// Full gate applied before a child process is spawned.
    ///
    /// # Errors
    ///
    /// Returns the Python rejection text: `command rejected for safety: <pattern>` for a
    /// deny-list match, or `command not allowed by allowlist` when the first token is not
    /// allowlisted. The deny list is checked first (Python order).
    pub fn check(&self, command: &str) -> std::result::Result<(), String> {
        if let Some(pattern) = self.dangerous_match(command) {
            return Err(format!("command rejected for safety: {pattern}"));
        }
        if !self.allows_command(command) {
            return Err("command not allowed by allowlist".to_string());
        }
        Ok(())
    }

    /// Extracts the first token of a command line with a quote-aware splitter.
    ///
    /// Leading whitespace is skipped. A token that starts with `'` or `"` is read up to
    /// the matching quote (inside `"` a backslash escapes the next character) and returned
    /// unquoted. Returns `None` for unparsable input: an unterminated quote, an empty
    /// line, or text glued directly onto the token (`"echo"-x`, which Python rejects with
    /// its `remainder[0].isspace()` check).
    #[must_use]
    pub fn first_token(command: &str) -> Option<String> {
        let text = prepare(command);
        let (token, end) = split_first_token(&text)?;
        if let Some(next) = text.get(end..).and_then(|rest| rest.chars().next()) {
            if !next.is_whitespace() {
                return None;
            }
        }
        Some(token)
    }

    /// Whether the configured allowlist accepts every command.
    fn allows_everything(&self) -> bool {
        self.normalized.is_empty()
    }

    /// Applies the exact-or-path-suffix rule to one value.
    fn matches_entry(&self, value: &str) -> bool {
        let candidate = normalize(value);
        self.normalized
            .iter()
            .any(|entry| candidate == *entry || candidate.ends_with(&format!("/{entry}")))
    }
}

/// Strips surrounding whitespace and, on Windows, one leading bare `&` caller marker.
#[cfg(windows)]
fn prepare(command: &str) -> String {
    let text = command.trim();
    if !text.starts_with("&&") {
        if let Some(rest) = text.strip_prefix('&') {
            return rest.trim_start().to_string();
        }
    }
    text.to_string()
}

/// Strips surrounding whitespace (POSIX has no `&` marker handling).
#[cfg(not(windows))]
fn prepare(command: &str) -> String {
    command.trim().to_string()
}

/// Splits the first token off `text`, returning it and the byte offset just past it.
///
/// Quotes are removed from a quoted token; the offset always points just after the token
/// in `text` so the caller can inspect what follows.
fn split_first_token(text: &str) -> Option<(String, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut index = 0;
    while index < chars.len() && chars[index].1.is_whitespace() {
        index += 1;
    }
    let (start, first) = *chars.get(index)?;
    if first == '\'' || first == '"' {
        let quote = first;
        let mut token = String::new();
        let mut cursor = index + 1;
        while let Some(&(offset, character)) = chars.get(cursor) {
            if quote == '"' && character == '\\' {
                let &(_, escaped) = chars.get(cursor + 1)?;
                token.push(escaped);
                cursor += 2;
                continue;
            }
            if character == quote {
                return Some((token, offset + character.len_utf8()));
            }
            token.push(character);
            cursor += 1;
        }
        // Unterminated quote: Python's `shlex.split` raises `ValueError`.
        return None;
    }
    let mut cursor = index;
    while cursor < chars.len() && !chars[cursor].1.is_whitespace() {
        cursor += 1;
    }
    let end = match chars.get(cursor) {
        Some(&(offset, _)) => offset,
        None => text.len(),
    };
    Some((text[start..end].to_string(), end))
}

/// Normalizes an allowlist entry or command token for comparison.
///
/// POSIX: identity (case-sensitive, no `.exe` handling). Windows: `\` becomes `/`, the
/// value is lowercased, and one trailing `.exe` is dropped — Python's
/// `os.name == "nt"` branch.
#[cfg(windows)]
fn normalize(value: &str) -> String {
    let lowered = value.replace('\\', "/").to_lowercase();
    lowered.strip_suffix(".exe").unwrap_or(&lowered).to_string()
}

/// Normalizes an allowlist entry or command token for comparison (POSIX identity).
#[cfg(not(windows))]
fn normalize(value: &str) -> String {
    value.to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn filter(allowlist: &[&str]) -> CommandFilter {
        CommandFilter::from_parts(
            Some(allowlist.iter().map(|entry| (*entry).to_string()).collect()),
            None,
        )
        .unwrap()
    }

    #[test]
    fn default_allowlist_allows_echo() {
        let filter = CommandFilter::new(&LocalWorkConfig::default()).unwrap();
        assert!(filter.allows_command("echo hi"));
        assert_eq!(filter.allowlist().map(<[String]>::len), Some(30));
    }

    #[test]
    fn unlisted_first_token_is_rejected() {
        let filter = filter(&["echo", "node"]);
        assert!(!filter.allows_command("rm -rf /tmp/x"));
        assert!(filter.allows_command("node script.js"));
    }

    #[test]
    fn none_or_empty_allowlist_allows_everything() {
        let none = CommandFilter::from_parts(None, None).unwrap();
        assert!(none.allows_command("rm -rf /"));
        assert!(none.allows_program("/usr/bin/anything"));
        assert_eq!(none.allowlist(), None);

        let empty = filter(&[]);
        assert!(empty.allows_command("rm -rf /"));
        assert_eq!(empty.allowlist().map(<[String]>::len), Some(0));
    }

    #[test]
    fn absolute_program_path_matches_leaf_entry() {
        let filter = filter(&["node"]);
        assert!(filter.allows_command("\"/usr/bin/node\" --version"));
        assert!(filter.allows_command("/usr/local/bin/node x.js"));
        assert!(filter.allows_program("/usr/local/bin/node"));
        assert!(!filter.allows_program("/usr/local/bin/printf"));
    }

    #[test]
    fn explicit_path_entry_is_literal() {
        let filter = filter(&["/usr/bin/node"]);
        assert!(filter.allows_command("/usr/bin/node script.js"));
        assert!(!filter.allows_command("node script.js"));
    }

    #[test]
    fn glued_suffix_is_rejected() {
        let filter = filter(&["echo", "node"]);
        assert!(!filter.allows_command("\"echo\"-x"));
        assert!(!filter.allows_command("\"node\"-unlisted.exe"));
        assert_eq!(CommandFilter::first_token("\"echo\"-x"), None);
    }

    #[test]
    fn unterminated_quote_is_rejected() {
        let filter = filter(&["echo"]);
        assert!(!filter.allows_command("\"echo --version"));
        assert_eq!(CommandFilter::first_token("\"echo --version"), None);
        assert_eq!(CommandFilter::first_token("'echo hi"), None);
        assert_eq!(CommandFilter::first_token("   "), None);
    }

    #[test]
    fn leading_whitespace_is_ignored_and_operators_are_not() {
        let filter = filter(&["echo"]);
        assert!(filter.allows_command("   echo hi"));
        assert_eq!(
            CommandFilter::first_token("  echo hi").as_deref(),
            Some("echo")
        );
        // On POSIX the first token of `&& echo hi` is `&&`, which is not allowlisted.
        assert!(!filter.allows_command("&& echo hi"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_marker_and_exe_suffix_are_normalized() {
        let filter = filter(&["node"]);
        assert!(filter.allows_command("& node app.js"));
        assert!(filter.allows_command("&& node app.js"));
        assert!(filter.allows_program(r"C:\Program Files\nodejs\node.exe"));
        assert!(filter.allows_program("NODE.EXE"));
    }

    #[cfg(not(windows))]
    #[test]
    fn posix_matching_is_case_sensitive() {
        let filter = filter(&["echo"]);
        assert!(!filter.allows_command("Echo hi"));
        assert!(filter.allows_command("echo hi"));
    }

    #[test]
    fn quoted_tokens_are_unquoted() {
        assert_eq!(
            CommandFilter::first_token("'/opt/my app/tool' --flag").as_deref(),
            Some("/opt/my app/tool")
        );
        assert_eq!(
            CommandFilter::first_token("\"say \\\"hi\\\"\"").as_deref(),
            Some("say \"hi\"")
        );
    }

    #[test]
    fn dangerous_match_is_case_insensitive_and_raw() {
        let filter = CommandFilter::from_parts(None, Some(vec![r"rm\s+-rf".to_string()])).unwrap();
        assert_eq!(filter.dangerous_match("rm -rf /"), Some(r"rm\s+-rf"));
        assert_eq!(filter.dangerous_match("RM -RF /"), Some(r"rm\s+-rf"));
        assert_eq!(filter.dangerous_match("echo ok"), None);

        let without = CommandFilter::from_parts(None, None).unwrap();
        assert_eq!(without.dangerous_match("rm -rf /"), None);
    }

    #[test]
    fn check_reports_python_messages() {
        let filter = CommandFilter::from_parts(
            Some(vec!["echo".to_string()]),
            Some(vec![r"rm\s+-rf".to_string()]),
        )
        .unwrap();
        assert_eq!(filter.check("echo hi"), Ok(()));
        assert_eq!(
            filter.check("rm -rf /"),
            Err("command rejected for safety: rm\\s+-rf".to_string())
        );
        assert_eq!(
            filter.check("ls -la"),
            Err("command not allowed by allowlist".to_string())
        );
    }

    #[test]
    fn invalid_regex_is_a_validation_error() {
        let error =
            CommandFilter::from_parts(None, Some(vec!["(unclosed".to_string()])).unwrap_err();
        assert!(error.message().contains("invalid dangerous pattern"));
        assert!(error.message().contains("(unclosed"));
    }

    #[test]
    fn deny_list_runs_before_allowlist() {
        let filter = CommandFilter::from_parts(
            Some(vec!["rm".to_string()]),
            Some(vec![r"rm\s+-rf".to_string()]),
        )
        .unwrap();
        assert!(filter.check("rm -rf /").is_err());
        assert_eq!(
            filter.check("rm -rf /").unwrap_err(),
            "command rejected for safety: rm\\s+-rf"
        );
    }
}
