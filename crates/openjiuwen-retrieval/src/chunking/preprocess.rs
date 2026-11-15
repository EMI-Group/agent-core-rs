//! Text preprocessing applied before chunking.
//!
//! Port of Python
//! `openjiuwen/core/retrieval/indexing/processor/chunker/text_preprocessor.py`:
//! the [`TextPreprocessor`] trait, [`WhitespaceNormalizer`],
//! [`URLEmailRemover`], [`SpecialCharacterNormalizer`], and the ordered
//! [`PreprocessingPipeline`] that applies them in sequence.
//!
//! Preprocessors are pure, synchronous, and infallible: an empty input is
//! returned unchanged, and no preprocessor reports an error. They run only
//! through a pipeline (or through [`crate::chunking::TextChunker`]), never
//! inside a [`crate::chunking::Chunker`] implementation.
//!
//! ## Redundant-symbol removal without lookarounds
//!
//! Python's `SpecialCharacterNormalizer` compiles
//! `(?![\u4e00-\u9fa5，。！？；：“”‘’（）【】《》、…·])(?![#*\-_]{2,})([^\w\s#*_\-\|]){2,}`.
//! Rust's `regex` crate has no lookarounds, so the equivalent behaviour is
//! assembled from a plain run matcher ([`REDUNDANT_RUN_PATTERN`]) plus a manual
//! scan:
//!
//! * `[^\w\s#*_\-\|]{2,}` is expressible as-is and finds every maximal run of
//!   two or more redundant symbols (a redundant symbol is neither a word
//!   character, nor whitespace, nor one of `#`, `*`, `_`, `-`, `|`).
//! * The second lookahead is vacuous: `#`, `*`, `-`, and `_` are excluded from
//!   the run class, so a run can never start on a markdown marker — `##`
//!   headings and `**bold**` emphasis survive untouched.
//! * The first lookahead (protect CJK punctuation and the `U+4E00..=U+9FA5`
//!   ideographs at the *start* of a run) is reproduced by the scan in
//!   [`strip_redundant_symbols`]: Python's engine fails the lookahead at a
//!   protected symbol, retries one character later, and then matches greedily to
//!   the end of the run. The protected prefix is therefore kept and the rest of
//!   the run is removed, provided at least two characters remain from the retry
//!   position (the ideograph half of the class is redundant in Python too, since
//!   ideographs are word characters and never enter the run class).
use std::fmt;
use std::sync::LazyLock;

use regex::{NoExpand, Regex};

/// Symbolic characters Python's first lookahead keeps at the start of a run.
const PROTECTED_SYMBOLS: &str = "，。！？；：“”‘’（）【】《》、…·";

/// The character run Python's redundant-symbol pattern removes.
///
/// Kept public for documentation and tests; the compiled form is cached in
/// the private `redundant_run_pattern`.
pub const REDUNDANT_RUN_PATTERN: &str = r"[^\w\s#*_\-|]{2,}";

/// Processes text before it reaches a chunker.
///
/// Implementations must be pure: the same input always yields the same output,
/// and an empty input is returned unchanged.
pub trait TextPreprocessor: Send + Sync {
    /// Returns the cleaned text.
    fn process(&self, text: &str) -> String;
}

/// Collapses every whitespace run into a single space and trims the ends.
///
/// Python parity: `re.sub(r"\s+", " ", text).strip()`, so tabs, newlines, and
/// repeated spaces all become one space and leading/trailing whitespace is
/// dropped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WhitespaceNormalizer;

impl WhitespaceNormalizer {
    /// Creates a normalizer.
    pub const fn new() -> Self {
        Self
    }
}

impl TextPreprocessor for WhitespaceNormalizer {
    fn process(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        whitespace_pattern()
            .replace_all(text, " ")
            .trim()
            .to_owned()
    }
}

/// Removes URLs and e-mail addresses.
///
/// The three patterns are applied in Python's order: full URLs (`http://`,
/// `https://`, `www.`), then bare domains ending in `.com`/`.net`/`.org`/`.cn`,
/// then e-mail addresses. Every match is replaced by [`URLEmailRemover::replacement`],
/// which defaults to the empty string. The replacement is inserted literally:
/// `$` and `\` have no special meaning.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct URLEmailRemover {
    remove_urls: bool,
    remove_emails: bool,
    replacement: String,
}

impl URLEmailRemover {
    /// Creates a remover that drops both URLs and e-mail addresses.
    ///
    /// This is Python's default (`remove_urls=True`, `remove_emails=True`,
    /// `replacement=""`).
    pub const fn new() -> Self {
        Self {
            remove_urls: true,
            remove_emails: true,
            replacement: String::new(),
        }
    }

    /// Configures whether URLs (and bare domains) are removed.
    #[must_use]
    pub fn with_remove_urls(mut self, remove_urls: bool) -> Self {
        self.remove_urls = remove_urls;
        self
    }

    /// Configures whether e-mail addresses are removed.
    #[must_use]
    pub fn with_remove_emails(mut self, remove_emails: bool) -> Self {
        self.remove_emails = remove_emails;
        self
    }

    /// Configures the literal replacement text.
    #[must_use]
    pub fn with_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.replacement = replacement.into();
        self
    }

    /// Returns whether URLs are removed.
    pub fn remove_urls(&self) -> bool {
        self.remove_urls
    }

    /// Returns whether e-mail addresses are removed.
    pub fn remove_emails(&self) -> bool {
        self.remove_emails
    }

    /// Returns the replacement text.
    pub fn replacement(&self) -> &str {
        &self.replacement
    }
}

impl TextPreprocessor for URLEmailRemover {
    fn process(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let mut current = text.to_owned();
        if self.remove_urls {
            current = url_pattern()
                .replace_all(&current, NoExpand(&self.replacement))
                .into_owned();
            current = domain_pattern()
                .replace_all(&current, NoExpand(&self.replacement))
                .into_owned();
        }
        if self.remove_emails {
            current = email_pattern()
                .replace_all(&current, NoExpand(&self.replacement))
                .into_owned();
        }
        current
    }
}

/// Removes control characters and redundant symbol runs, then applies mappings.
///
/// Python parity order: strip `[\x00-\x1F\x7F]`, strip every run of two or more
/// redundant symbols, apply [`SpecialCharacterNormalizer::chars_to_replace`] in
/// insertion order, then drop every character listed in
/// [`SpecialCharacterNormalizer::chars_to_remove`].
/// CJK punctuation and ideographs at the start of a run survive; see the module
/// documentation for the exact lookahead equivalence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialCharacterNormalizer {
    chars_to_remove: String,
    chars_to_replace: Vec<(String, String)>,
}

impl SpecialCharacterNormalizer {
    /// Creates a normalizer with no extra removals or replacements.
    pub const fn new() -> Self {
        Self {
            chars_to_remove: String::new(),
            chars_to_replace: Vec::new(),
        }
    }

    /// Sets the characters removed from the text (Python's `chars_to_remove`).
    #[must_use]
    pub fn with_chars_to_remove(mut self, chars_to_remove: impl Into<String>) -> Self {
        self.chars_to_remove = chars_to_remove.into();
        self
    }

    /// Sets the whole replacement map (Python's `chars_to_replace` dict).
    ///
    /// Pairs are applied in order, each as a plain substring replacement.
    #[must_use]
    pub fn with_chars_to_replace(
        mut self,
        chars_to_replace: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        self.chars_to_replace = chars_to_replace.into_iter().collect();
        self
    }

    /// Adds one `from` → `to` mapping, appended after the existing ones.
    #[must_use]
    pub fn with_replacement(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.chars_to_replace.push((from.into(), to.into()));
        self
    }

    /// Returns the characters removed from the text.
    pub fn chars_to_remove(&self) -> &str {
        &self.chars_to_remove
    }

    /// Returns the ordered replacement pairs.
    pub fn chars_to_replace(&self) -> &[(String, String)] {
        &self.chars_to_replace
    }
}

impl TextPreprocessor for SpecialCharacterNormalizer {
    fn process(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let without_controls = control_pattern().replace_all(text, "").into_owned();
        let mut current = strip_redundant_symbols(&without_controls);
        for (from, to) in &self.chars_to_replace {
            if !from.is_empty() {
                current = current.replace(from.as_str(), to);
            }
        }
        if !self.chars_to_remove.is_empty() {
            let removals: Vec<char> = self.chars_to_remove.chars().collect();
            if !removals.is_empty() {
                current.retain(|c| !removals.contains(&c));
            }
        }
        current
    }
}

/// Applies a list of preprocessors in order.
///
/// Python parity: `process` threads the text through every preprocessor
/// sequentially; an empty pipeline is the identity.
#[derive(Default)]
pub struct PreprocessingPipeline {
    preprocessors: Vec<Box<dyn TextPreprocessor>>,
}

impl PreprocessingPipeline {
    /// Creates a pipeline from preprocessors applied in the given order.
    pub fn new(preprocessors: Vec<Box<dyn TextPreprocessor>>) -> Self {
        Self { preprocessors }
    }

    /// Appends a preprocessor to the end of the pipeline.
    pub fn add_preprocessor(&mut self, preprocessor: Box<dyn TextPreprocessor>) {
        self.preprocessors.push(preprocessor);
    }

    /// Runs every preprocessor in order and returns the final text.
    pub fn process(&self, text: &str) -> String {
        let mut current = text.to_owned();
        for preprocessor in &self.preprocessors {
            current = preprocessor.process(&current);
        }
        current
    }

    /// Returns the number of preprocessors.
    pub fn len(&self) -> usize {
        self.preprocessors.len()
    }

    /// Returns whether the pipeline holds no preprocessor (the identity).
    pub fn is_empty(&self) -> bool {
        self.preprocessors.is_empty()
    }

    /// Builds Python's `TextChunker` `preprocess_options` pair.
    ///
    /// Python applies `WhitespaceNormalizer` first when `normalize_whitespace`
    /// is set and `URLEmailRemover` second when `remove_url_email` is set, so an
    /// option that is off contributes no preprocessor.
    pub fn from_options(normalize_whitespace: bool, remove_url_email: bool) -> Self {
        let mut preprocessors: Vec<Box<dyn TextPreprocessor>> = Vec::new();
        if normalize_whitespace {
            preprocessors.push(Box::new(WhitespaceNormalizer::new()));
        }
        if remove_url_email {
            preprocessors.push(Box::new(URLEmailRemover::new()));
        }
        Self::new(preprocessors)
    }
}

impl fmt::Debug for PreprocessingPipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreprocessingPipeline")
            .field("len", &self.preprocessors.len())
            .finish()
    }
}

/// Removes Python's redundant symbol runs while sparing protected CJK symbols.
fn strip_redundant_symbols(text: &str) -> String {
    let pattern = redundant_run_pattern();
    if !pattern.is_match(text) {
        return text.to_owned();
    }
    let mut output = String::with_capacity(text.len());
    let mut consumed = 0;
    for run_match in pattern.find_iter(text) {
        output.push_str(&text[consumed..run_match.start()]);
        let run = run_match.as_str();
        let mut keep = run.len();
        for (offset, character) in run.char_indices() {
            if is_protected_symbol(character) {
                continue;
            }
            // Python's engine retries one character later, so the protected
            // prefix survives and the remainder is removed only when at least
            // two characters are left to match.
            if run[offset..].chars().count() >= 2 {
                keep = offset;
            }
            break;
        }
        output.push_str(&run[..keep]);
        consumed = run_match.end();
    }
    output.push_str(&text[consumed..]);
    output
}

/// Returns whether `character` protects the start of a redundant symbol run.
fn is_protected_symbol(character: char) -> bool {
    PROTECTED_SYMBOLS.contains(character) || ('\u{4E00}'..='\u{9FA5}').contains(&character)
}

/// Compiles one of the module's hard-coded regular expressions.
///
/// The patterns are literals covered by unit tests, so construction can only
/// fail on a programming error.
#[allow(clippy::expect_used)]
fn compile_pattern(pattern: &str) -> Regex {
    Regex::new(pattern).expect("preprocessor patterns are valid regular expressions")
}

/// `\s+`, Python's whitespace-run pattern.
fn whitespace_pattern() -> &'static Regex {
    static PATTERN: LazyLock<Regex> = LazyLock::new(|| compile_pattern(r"\s+"));
    &PATTERN
}

/// `https?://\S+|www\.\S+`, Python's URL pattern.
fn url_pattern() -> &'static Regex {
    static PATTERN: LazyLock<Regex> = LazyLock::new(|| compile_pattern(r"https?://\S+|www\.\S+"));
    &PATTERN
}

/// Python's bare-domain pattern.
fn domain_pattern() -> &'static Regex {
    static PATTERN: LazyLock<Regex> = LazyLock::new(|| {
        compile_pattern(r"(?:https?://|www\.)?\S+?\.(?:com|net|org|cn)(?:[/?#]\S*)?\b")
    });
    &PATTERN
}

/// `\b[\w\.-]+@[\w\.-]+\.\w+\b`, Python's e-mail pattern.
fn email_pattern() -> &'static Regex {
    static PATTERN: LazyLock<Regex> =
        LazyLock::new(|| compile_pattern(r"\b[\w.-]+@[\w.-]+\.\w+\b"));
    &PATTERN
}

/// `[\x00-\x1F\x7F]`, Python's control character pattern.
fn control_pattern() -> &'static Regex {
    static PATTERN: LazyLock<Regex> = LazyLock::new(|| compile_pattern(r"[\x00-\x1F\x7F]"));
    &PATTERN
}

/// The compiled [`REDUNDANT_RUN_PATTERN`].
fn redundant_run_pattern() -> &'static Regex {
    static PATTERN: LazyLock<Regex> = LazyLock::new(|| compile_pattern(REDUNDANT_RUN_PATTERN));
    &PATTERN
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // Port of tests/unit_tests/core/retrieval/indexing/processor/chunker/test_text_preprocessor.py
    #[test]
    fn whitespace_normalizer_collapses_runs_and_trims() {
        let normalizer = WhitespaceNormalizer::new();
        assert_eq!(normalizer.process("This is a test"), "This is a test");
        assert_eq!(
            normalizer.process("This   is    a     test"),
            "This is a test"
        );
        assert_eq!(normalizer.process("This\nis\na\ntest"), "This is a test");
        assert_eq!(normalizer.process("This\tis\ta\ttest"), "This is a test");
        assert_eq!(
            normalizer.process("This  \n\t  is  \n\t  a  \n\t  test"),
            "This is a test"
        );
        assert_eq!(normalizer.process("   This is a test   "), "This is a test");
        assert_eq!(normalizer.process(""), "");
        assert_eq!(normalizer.process(" \n\t "), "");
    }

    #[test]
    fn url_email_remover_drops_urls_and_emails_by_default() {
        let remover = URLEmailRemover::new();
        assert!(remover.remove_urls());
        assert!(remover.remove_emails());
        assert_eq!(remover.replacement(), "");

        for text in [
            "Visit http://example.com for more info",
            "Visit https://example.com for more info",
            "Visit www.example.com for more info",
        ] {
            let cleaned = remover.process(text);
            assert_eq!(cleaned, "Visit  for more info");
            assert!(!cleaned.contains("example"));
        }
        assert!(!remover
            .process("Contact us at test@example.com for support")
            .contains("@example.com"));
        assert_eq!(remover.process(""), "");
    }

    #[test]
    fn url_email_remover_honours_replacement_and_switches() {
        let replacer = URLEmailRemover::new().with_replacement("[URL]");
        assert_eq!(
            replacer.process("Visit http://example.com for more info"),
            "Visit [URL] for more info"
        );
        // The replacement applies to URLs and e-mails alike (Python parity).
        assert_eq!(
            replacer.process("Contact test@example.com"),
            "Contact [URL]"
        );

        let email_only = URLEmailRemover::new().with_replacement("[EMAIL]");
        assert_eq!(
            email_only.process("Contact test@example.com"),
            "Contact [EMAIL]"
        );

        let urls_off = URLEmailRemover::new().with_remove_urls(false);
        assert_eq!(
            urls_off.process("Visit http://example.com for more info"),
            "Visit http://example.com for more info"
        );
        let emails_off = URLEmailRemover::new().with_remove_emails(false);
        assert_eq!(
            emails_off.process("Contact test@example.io"),
            "Contact test@example.io"
        );
        // A `.com` address still matches Python's bare-domain URL pattern.
        assert_eq!(emails_off.process("Contact test@example.com"), "Contact ");

        // The replacement is inserted literally: `$` is not a group reference.
        let literal = URLEmailRemover::new().with_replacement("$1\\");
        assert_eq!(
            literal.process("Visit www.example.com now"),
            "Visit $1\\ now"
        );
    }

    #[test]
    fn url_email_remover_matches_bare_domains_and_paths() {
        let remover = URLEmailRemover::new();
        assert_eq!(
            remover.process("see example.com/docs?page=2 here"),
            "see  here"
        );
        assert_eq!(remover.process("example.cn is a domain"), " is a domain");
        assert_eq!(remover.process("no links here"), "no links here");
    }

    #[test]
    fn special_character_normalizer_removes_controls() {
        let normalizer = SpecialCharacterNormalizer::new();
        let text = "Test\u{0}text\u{1f}with\u{7f}control";
        let cleaned = normalizer.process(text);
        assert_eq!(cleaned, "Testtextwithcontrol");
        assert_eq!(normalizer.process(""), "");
        assert_eq!(normalizer.chars_to_remove(), "");
        assert_eq!(normalizer.chars_to_replace().len(), 0);
    }

    #[test]
    fn special_character_normalizer_applies_mappings_then_removals() {
        let normalizer = SpecialCharacterNormalizer::new()
            .with_replacement("&", "and")
            .with_replacement("@", "at");
        let cleaned = normalizer.process("Tom & Jerry @ home");
        assert_eq!(cleaned, "Tom and Jerry at home");
        assert!(!cleaned.contains('&'));
        assert!(!cleaned.contains('@'));

        let remover = SpecialCharacterNormalizer::new().with_chars_to_remove("!@#");
        assert_eq!(
            remover.process("Test!text@with#special"),
            "Testtextwithspecial"
        );
        assert_eq!(remover.chars_to_remove(), "!@#");
    }

    #[test]
    fn special_character_normalizer_strips_redundant_symbol_runs() {
        let normalizer = SpecialCharacterNormalizer::new();
        // Two or more non-word symbols collapse away.
        assert_eq!(normalizer.process("hello!!!world"), "helloworld");
        assert_eq!(normalizer.process("a~~~b"), "ab");
        assert_eq!(normalizer.process("a→→→b"), "ab");
        // A single symbol is left alone.
        assert_eq!(normalizer.process("hello!world"), "hello!world");
        assert_eq!(normalizer.process("a~b"), "a~b");
    }

    #[test]
    fn special_character_normalizer_keeps_cjk_punctuation_and_markdown() {
        let normalizer = SpecialCharacterNormalizer::new();
        // Protected CJK punctuation at the start of a run survives: Python's
        // lookahead fails there and the retry needs two characters to match.
        assert_eq!(normalizer.process("你好！！世界"), "你好！！世界");
        assert_eq!(normalizer.process("再见！！！"), "再见！！！");
        assert_eq!(normalizer.process("看，。！"), "看，。！");
        assert_eq!(normalizer.process("！!"), "！!");
        assert_eq!(normalizer.process("…·"), "…·");
        // A removable suffix after the protected prefix still goes away.
        assert_eq!(normalizer.process("你好！！！!!!"), "你好！！！");
        assert_eq!(normalizer.process("！！!!"), "！！");
        // A run that starts on an unprotected symbol is removed whole, even
        // when protected symbols follow it.
        assert_eq!(normalizer.process("!!！"), "");
        // Markdown markers are never redundant symbols.
        assert_eq!(normalizer.process("## Heading"), "## Heading");
        assert_eq!(
            normalizer.process("**bold** and _italic_"),
            "**bold** and _italic_"
        );
        assert_eq!(normalizer.process("a | b | c"), "a | b | c");
        assert_eq!(normalizer.process("---"), "---");
    }

    #[test]
    fn pipeline_applies_preprocessors_in_order() {
        let pipeline = PreprocessingPipeline::new(vec![
            Box::new(WhitespaceNormalizer::new()),
            Box::new(URLEmailRemover::new()),
        ]);
        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());
        assert_eq!(
            pipeline.process("Visit   http://example.com   now"),
            "Visit  now"
        );

        let empty = PreprocessingPipeline::default();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.process("  untouched  "), "  untouched  ");
    }

    #[test]
    fn pipeline_order_is_observable() {
        let mut pipeline = PreprocessingPipeline::default();
        pipeline.add_preprocessor(Box::new(
            SpecialCharacterNormalizer::new().with_replacement("a", "b"),
        ));
        pipeline.add_preprocessor(Box::new(
            SpecialCharacterNormalizer::new().with_replacement("b", "c"),
        ));
        assert_eq!(pipeline.len(), 2);
        // The first preprocessor rewrites `a` to `b`, the second rewrites `b` to `c`.
        assert_eq!(pipeline.process("aaa"), "ccc");
    }

    #[test]
    fn from_options_mirrors_python_preprocess_options() {
        assert!(PreprocessingPipeline::from_options(false, false).is_empty());
        assert_eq!(PreprocessingPipeline::from_options(true, false).len(), 1);
        assert_eq!(PreprocessingPipeline::from_options(false, true).len(), 1);
        let both = PreprocessingPipeline::from_options(true, true);
        assert_eq!(both.len(), 2);
        // Whitespace normalization runs first, then URL removal.
        assert_eq!(
            both.process("Visit   http://example.com   now"),
            "Visit  now"
        );
    }
}
