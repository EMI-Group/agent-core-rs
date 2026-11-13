//! Dependency-free ANSI styling for the CLI's terminal output.
//!
//! Colour is opt-in: a [`Style`] built by [`Style::plain`] (which is also [`Style::default`]) returns every string unchanged.
//! Plain output therefore stays byte-identical to the unstyled output, which is what piped runs and the `json`/`stream-json` output formats rely on.
//! [`color_enabled`](crate::cli::paths::color_enabled) decides the flag from `--no-color`, `NO_COLOR` and tty detection; this module only formats strings.
//! Nothing here writes anywhere: every helper returns a `String` that the caller hands to [`CliIo`](crate::cli::CliIo).

/// Escape sequence restoring the terminal's default attributes.
pub const RESET: &str = "\x1b[0m";

/// Cyan foreground.
pub const CYAN: &str = "\x1b[36m";

/// Green foreground.
pub const GREEN: &str = "\x1b[32m";

/// Yellow foreground.
pub const YELLOW: &str = "\x1b[33m";

/// Red foreground.
pub const RED: &str = "\x1b[31m";

/// Dimmed (faint) text.
pub const DIM: &str = "\x1b[2m";

/// Bold text.
pub const BOLD: &str = "\x1b[1m";

/// Bold cyan text.
pub const BOLD_CYAN: &str = "\x1b[1;36m";

/// Bold yellow text.
pub const BOLD_YELLOW: &str = "\x1b[1;33m";

/// Whether ANSI styling is enabled, plus the helpers that apply it.
///
/// A `Style` is a plain `Copy` flag: it holds no writer and no global state, so
/// the same value may be shared between the renderer and the REPL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Style {
    enabled: bool,
}

impl Style {
    /// Returns a style that emits no escape sequences (the default).
    #[must_use]
    pub fn plain() -> Self {
        Self { enabled: false }
    }

    /// Returns a style that emits escape sequences only when `enabled` is true.
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Returns whether escape sequences are emitted.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Wraps `text` in `code` … [`RESET`] when styling is enabled.
    ///
    /// Returns `text` unchanged when styling is disabled, and never wraps empty
    /// text, so blank lines never carry escape bytes in either mode.
    #[must_use]
    pub fn paint(&self, code: &str, text: &str) -> String {
        if !self.enabled || text.is_empty() {
            return text.to_owned();
        }
        format!("{code}{text}{RESET}")
    }

    /// Paints `text` in [`CYAN`].
    #[must_use]
    pub fn cyan(&self, text: &str) -> String {
        self.paint(CYAN, text)
    }

    /// Paints `text` in [`GREEN`].
    #[must_use]
    pub fn green(&self, text: &str) -> String {
        self.paint(GREEN, text)
    }

    /// Paints `text` in [`YELLOW`].
    #[must_use]
    pub fn yellow(&self, text: &str) -> String {
        self.paint(YELLOW, text)
    }

    /// Paints `text` in [`RED`].
    #[must_use]
    pub fn red(&self, text: &str) -> String {
        self.paint(RED, text)
    }

    /// Paints `text` in [`DIM`].
    #[must_use]
    pub fn dim(&self, text: &str) -> String {
        self.paint(DIM, text)
    }

    /// Paints `text` in [`BOLD`].
    #[must_use]
    pub fn bold(&self, text: &str) -> String {
        self.paint(BOLD, text)
    }

    /// Paints `text` in [`BOLD_CYAN`].
    #[must_use]
    pub fn bold_cyan(&self, text: &str) -> String {
        self.paint(BOLD_CYAN, text)
    }

    /// Paints `text` in [`BOLD_YELLOW`].
    #[must_use]
    pub fn bold_yellow(&self, text: &str) -> String {
        self.paint(BOLD_YELLOW, text)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// One styling helper paired with the colour code it applies.
    type Helper = fn(&Style, &str) -> String;

    /// Every helper of [`Style`], paired with the colour code it applies.
    fn helpers() -> [(Helper, &'static str); 8] {
        [
            (Style::cyan, CYAN),
            (Style::green, GREEN),
            (Style::yellow, YELLOW),
            (Style::red, RED),
            (Style::dim, DIM),
            (Style::bold, BOLD),
            (Style::bold_cyan, BOLD_CYAN),
            (Style::bold_yellow, BOLD_YELLOW),
        ]
    }

    #[test]
    fn plain_style_leaves_text_untouched_for_every_helper() {
        let style = Style::plain();
        assert!(!style.enabled());
        for (helper, _code) in helpers() {
            assert_eq!(helper(&style, "plain text"), "plain text");
        }
        assert_eq!(style.paint(CYAN, "plain text"), "plain text");
    }

    #[test]
    fn enabled_style_emits_code_text_and_reset_byte_exactly() {
        let style = Style::new(true);
        assert!(style.enabled());
        assert_eq!(style.cyan("hi"), "\x1b[36mhi\x1b[0m");
        assert_eq!(style.dim("hi"), "\x1b[2mhi\x1b[0m");
        assert_eq!(style.red("boom"), "\x1b[31mboom\x1b[0m");
        assert_eq!(style.bold_yellow("warn"), "\x1b[1;33mwarn\x1b[0m");
        assert_eq!(style.paint(RESET, "x"), "\x1b[0mx\x1b[0m");
    }

    #[test]
    fn every_helper_uses_its_own_code_when_enabled() {
        let style = Style::new(true);
        for (helper, code) in helpers() {
            assert_eq!(
                helper(&style, "t"),
                format!("{code}t{RESET}"),
                "unexpected code for {code:?}"
            );
        }
    }

    #[test]
    fn empty_text_is_never_wrapped() {
        for style in [Style::plain(), Style::new(true)] {
            assert_eq!(style.paint(CYAN, ""), "");
            for (helper, _code) in helpers() {
                assert_eq!(helper(&style, ""), "");
            }
        }
    }

    #[test]
    fn default_style_equals_plain() {
        assert_eq!(Style::default(), Style::plain());
        assert!(!Style::default().enabled());
    }

    #[test]
    fn plain_output_contains_no_escape_byte() {
        let style = Style::plain();
        let mut rendered = String::new();
        for (helper, _code) in helpers() {
            rendered.push_str(&helper(&style, "line\n"));
        }
        assert!(!rendered.contains('\x1b'));
    }
}
