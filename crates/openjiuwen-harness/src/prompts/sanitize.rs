//! Interpolation sanitizers for values that reach the prompt.
//!
//! Ports `openjiuwen/harness/prompts/sanitize.py`: model- or user-supplied text is
//! stripped of the characters that could break a prompt template (`<`, `>`, `{`, `}`,
//! `[`, `]`, the backtick, `$`), of any run of three or more dots, and of the literal
//! two-character sequences `\n` / `\r` (a backslash followed by `n` or `r`, not real
//! newlines), then truncated to a character budget.
//!
//! Implemented by hand rather than with a regex: the rules are a single left-to-right
//! pass, which keeps multi-byte input intact and cannot backtrack.

/// Default character budget of [`sanitize_user_content`].
pub const DEFAULT_MAX_USER_CONTENT_LEN: usize = 2000;

/// Characters removed wherever they appear.
const STRIPPED_CHARS: [char; 8] = ['<', '>', '{', '}', '[', ']', '`', '$'];

/// Sanitises a path for interpolation into a prompt, without truncation.
#[must_use]
pub fn sanitize_path(input: &str) -> String {
    strip_injections(input)
}

/// Sanitises user-supplied content and truncates it to [`DEFAULT_MAX_USER_CONTENT_LEN`].
#[must_use]
pub fn sanitize_user_content(input: &str) -> String {
    sanitize_user_content_with_limit(input, DEFAULT_MAX_USER_CONTENT_LEN)
}

/// Sanitises user-supplied content and truncates it to at most `max_len` characters.
///
/// Truncation counts characters, not bytes, and adds no ellipsis: the result may end
/// mid-sentence but never mid-character.
#[must_use]
pub fn sanitize_user_content_with_limit(input: &str, max_len: usize) -> String {
    strip_injections(input).chars().take(max_len).collect()
}

/// Removes the injection-relevant characters and escapes from `input`.
///
/// A run of more than two consecutive dots is dropped entirely; one or two dots are
/// kept. Runs are consumed greedily so a longer run never leaves a dot behind.
fn strip_injections(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(current) = chars.next() {
        if STRIPPED_CHARS.contains(&current) {
            continue;
        }
        if current == '.' {
            let mut dots = 1_usize;
            while chars.peek() == Some(&'.') {
                chars.next();
                dots += 1;
            }
            if dots < 3 {
                for _ in 0..dots {
                    output.push('.');
                }
            }
            continue;
        }
        if current == '\\' {
            if matches!(chars.peek(), Some('n' | 'r')) {
                chars.next();
            } else {
                output.push('\\');
            }
            continue;
        }
        output.push(current);
    }

    output
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        sanitize_path, sanitize_user_content, sanitize_user_content_with_limit,
        DEFAULT_MAX_USER_CONTENT_LEN,
    };

    #[test]
    fn default_limit_is_two_thousand() {
        assert_eq!(DEFAULT_MAX_USER_CONTENT_LEN, 2000);
        let long = "a".repeat(2500);
        assert_eq!(sanitize_user_content(&long).chars().count(), 2000);
        assert_eq!(sanitize_path(&long).chars().count(), 2500);
    }

    #[test]
    fn strips_template_characters() {
        assert_eq!(
            sanitize_user_content("a<b>c{d}e[f]g`h$i"),
            "abcdefghi".to_string()
        );
        assert_eq!(sanitize_path("/tmp/{evil}/file"), "/tmp/evil/file");
        assert_eq!(sanitize_path("/tmp/$HOME/x"), "/tmp/HOME/x");
    }

    #[test]
    fn keeps_ordinary_text_and_single_dots() {
        let text = "Hello, world! 你好，世界。 (a) - b _ c + d = e / f";
        assert_eq!(sanitize_user_content(text), text);
        assert_eq!(sanitize_user_content("/a/b/c.txt"), "/a/b/c.txt");
        assert_eq!(sanitize_user_content(".."), "..".to_string());
        assert_eq!(sanitize_user_content("a.b"), "a.b".to_string());
    }

    #[test]
    fn drops_runs_of_three_or_more_dots() {
        assert_eq!(sanitize_user_content("a...b"), "ab".to_string());
        assert_eq!(sanitize_user_content("a.....b"), "ab".to_string());
        assert_eq!(sanitize_user_content("a.... .b"), "a .b".to_string());
        assert_eq!(sanitize_user_content("..."), String::new());
        assert_eq!(
            sanitize_user_content("../..//etc"),
            "../..//etc".to_string()
        );
    }

    #[test]
    fn drops_literal_backslash_n_and_r_only() {
        assert_eq!(sanitize_user_content(r"a\nb"), "ab".to_string());
        assert_eq!(sanitize_user_content(r"a\rb"), "ab".to_string());
        assert_eq!(sanitize_user_content(r"a\tb"), r"a\tb".to_string());
        // Real newlines survive: they are structure, not an escape sequence.
        assert_eq!(sanitize_user_content("a\nb"), "a\nb".to_string());
        // Any other backslash is ordinary text and survives.
        assert_eq!(sanitize_user_content("abc\\"), "abc\\".to_string());
        assert_eq!(sanitize_user_content(r"C:\Users"), r"C:\Users".to_string());
        // `\\n` drops the pair `\n` and keeps the leading backslash.
        assert_eq!(sanitize_user_content(r"\\n"), "\\".to_string());
    }

    #[test]
    fn truncation_counts_characters_and_adds_no_ellipsis() {
        let text = "你好世界你好世界";
        let truncated = sanitize_user_content_with_limit(text, 4);
        assert_eq!(truncated, "你好世界");
        assert_eq!(truncated.chars().count(), 4);
        assert!(!truncated.ends_with('…'));
        assert_eq!(
            sanitize_user_content_with_limit("abc", 10),
            "abc".to_string()
        );
        assert_eq!(sanitize_user_content_with_limit("abc", 0), String::new());
    }

    #[test]
    fn limit_applies_after_stripping() {
        // The stripped characters do not consume the budget.
        assert_eq!(
            sanitize_user_content_with_limit("<<<abcd", 4),
            "abcd".to_string()
        );
    }

    #[test]
    fn sanitize_path_never_truncates() {
        let long = "x".repeat(5000);
        assert_eq!(sanitize_path(&long).chars().count(), 5000);
    }
}
