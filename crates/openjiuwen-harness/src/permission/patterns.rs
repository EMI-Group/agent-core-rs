//! Pattern matchers shared by the permission pipeline.
//!
//! Ports `openjiuwen/harness/security/permission_engine/toolguard/pattern_matchers.py`
//! plus the private helpers `_match_glob` (`fileguard/file_guard.py`) and
//! `match_net_pattern` (`netguard/net_guard.py`) of the Python reference.
//!
//! Three matcher semantics live here and must not be conflated:
//!
//! 1. [`match_wildcard`] — shell-command globbing. `*` and `?` never consume shell
//!    metacharacters (`;`, `|`, `&`, backticks, `$`, quotes, redirections), newlines,
//!    CR or NUL, so a wildcard rule can never be widened into command injection.
//! 2. [`match_glob`] — POSIX path globbing. `*` and `?` never cross `/`, while `**`
//!    crosses directory boundaries.
//! 3. [`PolicyMatcher`] — the tiered-policy matcher, selecting exact equality, shell
//!    wildcard, or `re:` regex semantics from the pattern shape.
//!
//! [`match_net_pattern`] matches fetch URLs either by literal prefix plus authority
//! equality or by hostname wildcard; [`extract_host`] exposes the hostname parser it
//! uses.
//!
//! All functions are pure, never panic, and fail closed: a malformed input or a regex
//! build error yields `false`.

use regex::Regex;

/// Character class a shell wildcard may consume: everything except shell
/// metacharacters, backslash, newlines and NUL (Python `_WILDCARD_CHARS`).
const SHELL_WILDCARD_CLASS: &str = r"[^;|&`$()<>\\\n\r\x00]";

/// Character class a URL wildcard may consume: URL path, query and fragment
/// characters, but never backslash (Python `_URL_WILDCARD_CHARS`).
const URL_WILDCARD_CLASS: &str = r"[-a-zA-Z0-9._/:?#&=%+~@,;\[\]']";

/// Characters escaped before wildcard substitution, mirroring Python's
/// `to_escape = set(".+^${}()|[]\\")`.
const REGEX_META: &[char] = &['.', '+', '^', '$', '{', '}', '(', ')', '|', '[', ']', '\\'];

/// Shell-command wildcard match (Python `match_wildcard(value, pattern)`).
///
/// Backslashes are normalized to `/` in both arguments. `?` matches exactly one
/// character and `*` matches zero or more characters of
/// the shell wildcard class; a trailing `" *"` additionally makes the separating
/// space optional, so `ls *` matches both `ls` and `ls -la`. `[` and `]` are
/// literals, and `**` therefore degrades to `*`.
///
/// Matching is anchored (a full match) so a value carrying extra shell syntax such
/// as `git status; rm -rf /` can never satisfy `git status *`. Matching is
/// case-insensitive on `Windows` only. Empty input or a regex build error returns
/// `false`.
///
/// One deviation from the Python reference: when the pattern ends in `" *"`, the
/// Python helper leaves any earlier `*` as a raw regex quantifier; this port still
/// expands every earlier `*` to a wildcard.
#[must_use]
pub fn match_wildcard(value: &str, pattern: &str) -> bool {
    if value.is_empty() || pattern.is_empty() {
        return false;
    }
    let value = value.replace('\\', "/");
    let pattern = pattern.replace('\\', "/");
    let escaped = escape_regex_meta(&pattern).replace('?', SHELL_WILDCARD_CLASS);
    let body = escaped.strip_suffix(" *").map_or_else(
        || escaped.replace('*', &format!("{SHELL_WILDCARD_CLASS}*")),
        |head| format!("{head}( {SHELL_WILDCARD_CLASS}*)?"),
    );
    anchored_regex(&body).is_some_and(|regex| regex.is_match(&value))
}

/// POSIX path glob match (Python `file_guard._match_glob`).
///
/// Backslashes are normalized to `/` in both arguments. `**/` matches any number
/// of leading directories (including none), `**` matches anything, `*` matches any
/// run of non-`/` characters and `?` matches exactly one non-`/` character; every
/// other character is a literal. Matching is anchored, case-sensitive on every
/// platform, and a regex build error returns `false`.
///
/// Consequences worth remembering: `**/.env` matches `/home/u/.env`,
/// `**/.ssh/**` matches `/home/u/.ssh/id_rsa`, and `/home/u/**` does *not* match
/// `/home/u` itself because the trailing `/` is required.
#[must_use]
pub fn match_glob(pattern: &str, path: &str) -> bool {
    let pattern = pattern.replace('\\', "/");
    let path = path.replace('\\', "/");
    let mut body = String::with_capacity(pattern.len() * 2);
    let mut index = 0;
    while index < pattern.len() {
        let rest = &pattern[index..];
        if rest.starts_with("**/") {
            body.push_str("(?:.*/)?");
            index += 3;
        } else if rest.starts_with("**") {
            body.push_str(".*");
            index += 2;
        } else if rest.starts_with('*') {
            body.push_str("[^/]*");
            index += 1;
        } else if rest.starts_with('?') {
            body.push_str("[^/]");
            index += 1;
        } else if let Some(ch) = rest.chars().next() {
            body.push_str(&regex::escape(&ch.to_string()));
            index += ch.len_utf8();
        } else {
            break;
        }
    }
    anchored_regex(&body).is_some_and(|regex| regex.is_match(&path))
}

/// Matcher used by the tiered policy for commands, paths and tool arguments.
///
/// [`PolicyMatcher::new`] picks the semantics from the pattern shape, in Python
/// order:
///
/// 1. a `re:` prefix (case-insensitive) selects [`PolicyMatcher::Regex`] and the
///    remainder is compiled once; matching is an unanchored search like Python
///    `re.search`, and a compile error makes every [`PolicyMatcher::matches`] call
///    return `false`;
/// 2. otherwise a pattern containing `*`, `?` or `[` selects
///    [`PolicyMatcher::Wildcard`] through [`match_wildcard`];
/// 3. otherwise [`PolicyMatcher::Exact`] matches after trimming both sides.
///
/// Regex and wildcard matching are case-insensitive on `Windows` only; exact
/// matching is always case-sensitive. The pattern itself is trimmed on
/// construction.
#[derive(Debug, Clone)]
pub enum PolicyMatcher {
    /// Case-sensitive equality after trimming both sides.
    Exact(String),
    /// Shell wildcard semantics via [`match_wildcard`].
    Wildcard(String),
    /// `re:` regular expression, matched with unanchored search semantics.
    Regex {
        /// The full trimmed pattern, including the `re:` prefix.
        source: String,
        /// The compiled expression, or `None` when it failed to compile.
        compiled: Option<Regex>,
    },
}

impl PolicyMatcher {
    /// Builds a matcher, selecting exact, wildcard or regex semantics.
    #[must_use]
    pub fn new(pattern: &str) -> Self {
        let trimmed = pattern.trim();
        if trimmed
            .get(..3)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("re:"))
        {
            let expression = trimmed.get(3..).unwrap_or_default().trim();
            let compiled = compile_regex(expression);
            return Self::Regex {
                source: trimmed.to_owned(),
                compiled,
            };
        }
        if trimmed.contains(['*', '?', '[']) {
            return Self::Wildcard(trimmed.to_owned());
        }
        Self::Exact(trimmed.to_owned())
    }

    /// Returns the trimmed pattern this matcher was built from.
    #[must_use]
    pub fn pattern(&self) -> &str {
        match self {
            Self::Exact(pattern) | Self::Wildcard(pattern) => pattern,
            Self::Regex { source, .. } => source,
        }
    }

    /// Returns whether `value` matches this pattern.
    ///
    /// A [`PolicyMatcher::Regex`] whose expression failed to compile never matches.
    #[must_use]
    pub fn matches(&self, value: &str) -> bool {
        match self {
            Self::Exact(pattern) => value.trim() == pattern,
            Self::Wildcard(pattern) => match_wildcard(value, pattern),
            Self::Regex { compiled, .. } => regex_matches(compiled.as_ref(), value),
        }
    }
}

impl PartialEq for PolicyMatcher {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Exact(left), Self::Exact(right))
            | (Self::Wildcard(left), Self::Wildcard(right))
            | (Self::Regex { source: left, .. }, Self::Regex { source: right, .. }) => {
                left == right
            }
            _ => false,
        }
    }
}

impl Eq for PolicyMatcher {}

/// URL policy match (Python `net_guard.match_net_pattern(pattern, url)`).
///
/// Note the argument order of the Rust port: value first, pattern second. Both
/// arguments are trimmed and have backslashes normalized to `/`; empty input
/// returns `false`.
///
/// - A pattern containing `://` and no wildcard must be a literal prefix of the
///   URL *and* share its authority, so `https://blocked.example/secret` matches
///   `https://blocked.example/secretly` while `https://example.com` does not match
///   `https://example.com.evil.com/x`, `https://example.com@evil.com/x` (userinfo
///   belongs to the authority) or `https://example.com:8443/x` (different port).
/// - A pattern containing `://` and `*` or `?` is a full-string URL glob where the
///   wildcards consume the URL wildcard class, case-sensitively.
/// - Any other pattern is a hostname wildcard: the lowercased hostname of the URL
///   (see [`extract_host`]) is matched against the lowercased pattern with
///   [`match_wildcard`], so `localhost`, `127.0.0.1` and `*.internal.example` work.
///
/// Malformed URLs or patterns, such as an unclosed `IPv6` bracket (`https://[`),
/// return `false` rather than panicking.
#[must_use]
pub fn match_net_pattern(url: &str, pattern: &str) -> bool {
    let url = url.trim().replace('\\', "/");
    let pattern = pattern.trim().replace('\\', "/");
    if url.is_empty() || pattern.is_empty() {
        return false;
    }
    if pattern.contains("://") {
        if pattern.contains('*') || pattern.contains('?') {
            return match_url_glob(&url, &pattern);
        }
        let Some(url_netloc) = netloc(&url) else {
            return false;
        };
        let Some(pattern_netloc) = netloc(&pattern) else {
            return false;
        };
        if !is_valid_netloc(url_netloc) || !is_valid_netloc(pattern_netloc) {
            return false;
        }
        return url.starts_with(&pattern) && url_netloc == pattern_netloc;
    }
    extract_host(&url).is_some_and(|host| match_wildcard(&host, &pattern.to_lowercase()))
}

/// Extracts the lowercased hostname from a URL or bare authority.
///
/// The scheme, any `user@` userinfo, a port suffix and a trailing dot are removed;
/// the host stops at the first `/`, `?` or `#`. The result is `None` for empty or
/// host-less input. It is used by [`match_net_pattern`] and by the URL guard.
///
/// ```
/// use openjiuwen_harness::permission::patterns::extract_host;
///
/// assert_eq!(
///     extract_host("https://user@API.Example.COM:8443/x").as_deref(),
///     Some("api.example.com")
/// );
/// assert_eq!(extract_host("localhost:8080/path").as_deref(), Some("localhost"));
/// assert_eq!(extract_host("https:///x"), None);
/// ```
#[must_use]
pub fn extract_host(url: &str) -> Option<String> {
    let text = url.trim();
    if text.is_empty() {
        return None;
    }
    let authority = text.split_once("://").map_or(text, |(_, rest)| rest);
    let end = authority.find(['/', '?', '#']).unwrap_or(authority.len());
    let authority = &authority[..end];
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let host = strip_port(host_port).trim_end_matches('.').to_lowercase();
    if host.is_empty() {
        None
    } else {
        Some(host)
    }
}

/// Matches a full URL against a wildcard pattern (Python `_match_url_glob`).
fn match_url_glob(value: &str, pattern: &str) -> bool {
    let escaped = escape_regex_meta(pattern).replace('?', URL_WILDCARD_CLASS);
    let body = escaped.replace('*', &format!("{URL_WILDCARD_CLASS}*"));
    anchored_regex(&body).is_some_and(|regex| regex.is_match(value))
}

/// Returns the authority of a URL: the text after `://` up to a path, query or
/// fragment delimiter.
fn netloc(url: &str) -> Option<&str> {
    let (_, rest) = url.split_once("://")?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Rejects authority strings with unbalanced IPv6 brackets, which Python's
/// `urlparse` refuses with `ValueError`.
fn is_valid_netloc(netloc: &str) -> bool {
    match (netloc.find('['), netloc.find(']')) {
        (None, None) => true,
        (Some(open), Some(close)) if open < close => {
            let tail = &netloc[close + 1..];
            tail.is_empty() || tail.starts_with(':')
        }
        _ => false,
    }
}

/// Removes a `:port` suffix, unwrapping a bracketed IPv6 host.
fn strip_port(host_port: &str) -> &str {
    if let Some(rest) = host_port.strip_prefix('[') {
        if let Some((host, _)) = rest.split_once(']') {
            return host;
        }
        return host_port;
    }
    host_port
        .rsplit_once(':')
        .map_or(host_port, |(host, _)| host)
}

/// Escapes Python's regex metacharacter set without escaping `*` or `?`, which
/// the callers substitute with their wildcard classes.
fn escape_regex_meta(pattern: &str) -> String {
    let mut escaped = String::with_capacity(pattern.len());
    for ch in pattern.chars() {
        if REGEX_META.contains(&ch) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

/// Anchors `body` as a full-string match, case-insensitively on `Windows` only.
fn anchored_regex(body: &str) -> Option<Regex> {
    let source = if cfg!(windows) {
        format!(r"(?i)\A(?:{body})\z")
    } else {
        format!(r"\A(?:{body})\z")
    };
    Regex::new(&source).ok()
}

/// Matches a compiled regex only when compilation succeeded.
fn regex_matches(compiled: Option<&Regex>, value: &str) -> bool {
    compiled.is_some_and(|regex| regex.is_match(value))
}

/// Compiles a `re:` expression for [`PolicyMatcher::Regex`], case-insensitively on
/// `Windows` only.
fn compile_regex(expression: &str) -> Option<Regex> {
    let source = if cfg!(windows) {
        format!("(?i){expression}")
    } else {
        expression.to_owned()
    };
    Regex::new(&source).ok()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    // ----- match_wildcard -------------------------------------------------

    #[test]
    fn wildcard_matches_commands_and_arguments() {
        assert!(match_wildcard("ls", "ls"));
        assert!(match_wildcard("ls -la", "ls *"));
        assert!(match_wildcard("git status", "git status"));
        assert!(match_wildcard("dir /b *.docx", "dir *"));
    }

    #[test]
    fn wildcard_trailing_space_star_is_optional() {
        assert!(match_wildcard("ls", "ls *"));
        assert!(match_wildcard("ls -la", "ls *"));
        assert!(!match_wildcard("lsx", "ls *"));
        assert!(!match_wildcard("l", "ls *"));
    }

    #[test]
    fn wildcard_rejects_shell_injection() {
        assert!(!match_wildcard("git status; rm -rf /", "git status *"));
        assert!(!match_wildcard("git status && rm", "git status *"));
        assert!(!match_wildcard("git status|cat", "git status *"));
        assert!(!match_wildcard("git status$(id)", "git status *"));
        assert!(!match_wildcard("git status\n", "git status *"));
        assert!(!match_wildcard("git status\r", "git status *"));
        assert!(!match_wildcard("git status\0", "git status *"));
        assert!(!match_wildcard("git status中文", "git status *"));
        assert!(!match_wildcard("echo $HOME", "echo *"));
        assert!(!match_wildcard("echo `id`", "echo *"));
    }

    #[test]
    fn wildcard_accepts_non_ascii_payloads() {
        assert!(match_wildcard("echo 中", "echo *"));
        assert!(match_wildcard("date +中", "date +?"));
    }

    #[test]
    fn wildcard_question_mark_matches_exactly_one_char() {
        assert!(match_wildcard("date +%", "date +?"));
        assert!(!match_wildcard("date +%Y", "date +?"));
        assert!(!match_wildcard("date +", "date +?"));
    }

    #[test]
    fn wildcard_escape_regex_metacharacters() {
        assert!(match_wildcard("file[1].txt", "file[1].txt"));
        assert!(!match_wildcard("file1.txt", "file[1].txt"));
        assert!(match_wildcard("echo $HOME", "echo $HOME"));
        assert!(match_wildcard("a.b", "a.b"));
        assert!(!match_wildcard("axb", "a.b"));
    }

    #[test]
    fn wildcard_double_star_degrades_to_single_star() {
        assert!(match_wildcard("abc", "**"));
        assert!(match_wildcard("abc", "*"));
        assert!(!match_wildcard("abc;def", "**"));
    }

    #[test]
    fn wildcard_normalizes_backslashes() {
        assert!(match_wildcard(r"dir\file", "dir/*"));
        assert!(match_wildcard("dir/file", r"dir\*"));
    }

    #[test]
    fn wildcard_refuses_empty_inputs() {
        assert!(!match_wildcard("", "*"));
        assert!(!match_wildcard("ls", ""));
        assert!(!match_wildcard("", ""));
    }

    #[test]
    #[cfg(not(windows))]
    fn wildcard_is_case_sensitive_off_windows() {
        assert!(!match_wildcard("LS -LA", "ls *"));
        assert!(!match_wildcard("ls -la", "LS *"));
    }

    #[test]
    #[cfg(windows)]
    fn wildcard_is_case_insensitive_on_windows() {
        assert!(match_wildcard("LS -LA", "ls *"));
    }

    // ----- match_glob -----------------------------------------------------

    #[test]
    fn glob_double_star_matches_dotfiles_anywhere() {
        assert!(match_glob("**/.env", "/home/u/.env"));
        assert!(match_glob("**/.env", ".env"));
        assert!(match_glob("**/.ssh/**", "/home/u/.ssh/id_rsa"));
        assert!(!match_glob("**/.ssh/**", "/home/u/.ssh"));
    }

    #[test]
    fn glob_double_star_requires_the_trailing_separator() {
        assert!(match_glob("/home/u/**", "/home/u/a"));
        assert!(match_glob("/home/u/**", "/home/u/a/b"));
        assert!(!match_glob("/home/u/**", "/home/u"));
    }

    #[test]
    fn glob_single_star_never_crosses_slash() {
        assert!(match_glob("*.txt", "a.txt"));
        assert!(!match_glob("*.txt", "a/b.txt"));
        assert!(match_glob("/home/*/x", "/home/u/x"));
        assert!(!match_glob("/home/*/x", "/home/u/v/x"));
    }

    #[test]
    fn glob_question_mark_matches_one_non_slash_char() {
        assert!(match_glob("a?c", "abc"));
        assert!(!match_glob("a?c", "ac"));
        assert!(!match_glob("a?c", "a/c"));
    }

    #[test]
    fn glob_is_case_sensitive_on_every_platform() {
        assert!(!match_glob("*.TXT", "a.txt"));
        assert!(!match_glob("/Home/U/*", "/home/u/a.txt"));
    }

    #[test]
    fn glob_normalizes_backslashes() {
        assert!(match_glob(r"home\u\*.txt", "home/u/a.txt"));
        assert!(match_glob("home/u/*.txt", r"home\u\a.txt"));
    }

    // ----- PolicyMatcher --------------------------------------------------

    #[test]
    fn policy_matcher_selects_exact_and_trims() {
        let matcher = PolicyMatcher::new("  git status  ");
        assert!(matches!(matcher, PolicyMatcher::Exact(_)));
        assert_eq!(matcher.pattern(), "git status");
        assert!(matcher.matches(" git status "));
        assert!(!matcher.matches("git  status"));
    }

    #[test]
    fn policy_matcher_exact_is_case_sensitive() {
        let matcher = PolicyMatcher::new("git status");
        assert!(matcher.matches("git status"));
        assert!(!matcher.matches("GIT STATUS"));
    }

    #[test]
    fn policy_matcher_selects_wildcard_for_glob_characters() {
        for pattern in ["git *", "date +?", "file[1]"] {
            assert!(
                matches!(PolicyMatcher::new(pattern), PolicyMatcher::Wildcard(_)),
                "{pattern} should select wildcard semantics"
            );
        }
        let matcher = PolicyMatcher::new("git *");
        assert!(matcher.matches("git status"));
        assert!(!matcher.matches("git status; rm -rf /"));
        assert!(PolicyMatcher::new("date +?").matches("date +%"));
        assert!(PolicyMatcher::new("file[1]").matches("file[1]"));
        assert!(!PolicyMatcher::new("file[1]").matches("file1"));
    }

    #[test]
    fn policy_matcher_selects_regex_for_re_prefix() {
        let matcher = PolicyMatcher::new(r"re:git\s+status");
        assert!(matches!(matcher, PolicyMatcher::Regex { .. }));
        assert_eq!(matcher.pattern(), r"re:git\s+status");
        assert!(matcher.matches("git   status"));
        assert!(matcher.matches("prefix git status suffix"));
        assert!(!matcher.matches("gitstatus"));
    }

    #[test]
    fn policy_matcher_regex_prefix_is_case_insensitive() {
        let matcher = PolicyMatcher::new("RE:drop\\s+table");
        assert!(matches!(matcher, PolicyMatcher::Regex { .. }));
        assert!(matcher.matches("drop table"));
    }

    #[test]
    fn policy_matcher_invalid_regex_never_matches() {
        let matcher = PolicyMatcher::new("re:[");
        assert!(matches!(matcher, PolicyMatcher::Regex { .. }));
        assert!(!matcher.matches("["));
        assert!(!matcher.matches("anything"));
    }

    #[test]
    fn policy_matcher_equality_uses_the_pattern() {
        assert_eq!(PolicyMatcher::new("git *"), PolicyMatcher::new(" git * "));
        assert_ne!(PolicyMatcher::new("git *"), PolicyMatcher::new("git ?"));
        assert_ne!(
            PolicyMatcher::new("git status"),
            PolicyMatcher::new("GIT STATUS")
        );
        assert_eq!(PolicyMatcher::new("re:a+"), PolicyMatcher::new("re:a+"));
    }

    // ----- match_net_pattern ---------------------------------------------

    #[test]
    fn net_literal_prefix_alone_is_not_enough() {
        assert!(match_net_pattern(
            "https://blocked.example/secretly",
            "https://blocked.example/secret"
        ));
        assert!(match_net_pattern(
            "https://blocked.example/secret",
            "https://blocked.example/secret"
        ));
        assert!(!match_net_pattern(
            "https://blocked.example/public",
            "https://blocked.example/secret"
        ));
    }

    #[test]
    fn net_requires_authority_equality() {
        assert!(match_net_pattern(
            "https://example.com/x",
            "https://example.com"
        ));
        assert!(!match_net_pattern(
            "https://example.com",
            "https://example.com.evil.com/x"
        ));
        assert!(!match_net_pattern(
            "https://example.com.evil.com/x",
            "https://example.com"
        ));
        assert!(!match_net_pattern(
            "https://example.com@evil.com/x",
            "https://example.com"
        ));
        assert!(!match_net_pattern(
            "https://example.com:8443/x",
            "https://example.com"
        ));
    }

    #[test]
    fn net_url_glob_matches_full_urls() {
        assert!(match_net_pattern(
            "https://evil.example/admin/users",
            "https://evil.example/admin*"
        ));
        assert!(!match_net_pattern(
            "https://evil.example/other",
            "https://evil.example/admin*"
        ));
        assert!(!match_net_pattern(
            "https://evil.example/Admin/users",
            "https://evil.example/admin*"
        ));
        assert!(match_net_pattern(
            "https://evil.example/admin1",
            "https://evil.example/admin?"
        ));
        assert!(!match_net_pattern(
            "https://evil.example/admin12",
            "https://evil.example/admin?"
        ));
    }

    #[test]
    fn net_hostname_patterns_match_bare_hosts() {
        assert!(match_net_pattern("http://localhost/", "localhost"));
        assert!(match_net_pattern("http://localhost:80/", "localhost"));
        assert!(match_net_pattern(
            "http://127.0.0.1:443/health",
            "127.0.0.1"
        ));
        assert!(match_net_pattern("http://user@localhost/x", "localhost"));
        assert!(match_net_pattern(
            "https://API.Example.COM/x",
            "api.example.com"
        ));
        assert!(!match_net_pattern("https://example.org/x", "example.com"));
    }

    #[test]
    fn net_hostname_wildcard_excludes_the_apex() {
        assert!(match_net_pattern(
            "https://api.internal.example/v1",
            "*.internal.example"
        ));
        assert!(!match_net_pattern(
            "https://internal.example/",
            "*.internal.example"
        ));
        assert!(match_net_pattern(
            "https://API.Internal.Example/v1",
            "*.internal.example"
        ));
    }

    #[test]
    fn net_rejects_empty_and_malformed_inputs() {
        assert!(!match_net_pattern("", "localhost"));
        assert!(!match_net_pattern("http://a.example", ""));
        assert!(!match_net_pattern("   ", "localhost"));
        assert!(!match_net_pattern("https://[", "https://["));
        assert!(!match_net_pattern("https://[", "https://example.com"));
        assert!(!match_net_pattern("https://example.com", "https://["));
        assert!(!match_net_pattern("://", "https://"));
    }

    // ----- extract_host ---------------------------------------------------

    #[test]
    fn extract_host_strips_scheme_userinfo_and_port() {
        assert_eq!(
            extract_host("https://User@API.Example.COM:8443/x").as_deref(),
            Some("api.example.com")
        );
        assert_eq!(
            extract_host("localhost:8080/path").as_deref(),
            Some("localhost")
        );
        assert_eq!(extract_host("http://[::1]:8080/").as_deref(), Some("::1"));
        assert_eq!(
            extract_host("https://example.com./x").as_deref(),
            Some("example.com")
        );
    }

    #[test]
    fn extract_host_returns_none_for_hostless_input() {
        assert_eq!(extract_host(""), None);
        assert_eq!(extract_host("   "), None);
        assert_eq!(extract_host("https:///x"), None);
    }
}
