//! Host-side semantics shared by the remote sandbox providers.
//!
//! Python parity: the helpers every provider module in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` copies verbatim —
//! `_normalize_read_params`, `_validate_read_params`, `_split_text_lines`,
//! `_select_text_lines`, the text newline transform, chunking for streaming
//! reads, `shlex.quote` for shell command lines, and the `list_files`
//! sort/filter pair.
//!
//! These are deliberately pure host-side functions: the providers call them
//! before and after their transports, so the same slicing, quoting, and ordering
//! behavior applies whether the bytes come from a local file or a container.

use openjiuwen_core::sys_operation::fs::types::{FileSystemItem, LineRange, ReadMode, SortBy};

/// Default chunk size of a streaming read, matching the Python providers.
pub(crate) const DEFAULT_STREAM_CHUNK_SIZE: usize = 8192;

/// Validated, normalized text-selection request (Python `_normalize_read_params`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LineSelection {
    /// Keep only the first `head` lines.
    pub head: Option<usize>,
    /// Keep only the last `tail` lines.
    pub tail: Option<usize>,
    /// Keep only the 1-based inclusive `(start, end)` window.
    pub line_range: Option<(usize, usize)>,
}

/// Normalizes `head`/`tail` and validates the combination against `mode`.
///
/// `head == Some(0)` and `tail == Some(0)` mean "no limit" and become `None`,
/// exactly like the Python providers.
///
/// # Errors
///
/// Returns Python's message text as a `String` when a line selection is asked
/// for in bytes mode, or when more than one selection is given.
pub(crate) fn normalize_line_selection(
    mode: ReadMode,
    head: Option<usize>,
    tail: Option<usize>,
    line_range: Option<LineRange>,
) -> std::result::Result<LineSelection, String> {
    let head = head.filter(|value| *value > 0);
    let tail = tail.filter(|value| *value > 0);
    if !mode.is_text() && (head.is_some() || tail.is_some() || line_range.is_some()) {
        return Err(
            "Parameters 'head', 'tail', and 'line_range' are only supported in text mode"
                .to_string(),
        );
    }
    let mut specified = Vec::new();
    if head.is_some() {
        specified.push("head");
    }
    if tail.is_some() {
        specified.push("tail");
    }
    if line_range.is_some() {
        specified.push("line_range");
    }
    if specified.len() > 1 {
        return Err(format!(
            "{} cannot be specified simultaneously",
            specified.join(" and ")
        ));
    }
    Ok(LineSelection {
        head,
        tail,
        line_range: line_range.map(|range| (range.start, range.end)),
    })
}

/// Applies a [`LineSelection`] to `text` with Python's slicing semantics.
///
/// The text is split with `splitlines(keepends=True)` semantics, so a caller
/// that reads only whole lines still sees the terminators. An invalid window
/// (`start == 0` or `end < start`) and an out-of-range window both yield the
/// empty string, matching the Python providers.
pub(crate) fn select_text_lines(text: &str, selection: &LineSelection) -> String {
    if let Some((start, end)) = selection.line_range {
        if start == 0 || end < start {
            return String::new();
        }
        let lines = split_lines(text);
        if start > lines.len() || end > lines.len() {
            return String::new();
        }
        return lines[start - 1..end].concat();
    }
    if let Some(tail) = selection.tail {
        let lines = split_lines(text);
        let start = lines.len().saturating_sub(tail);
        return lines[start..].concat();
    }
    if let Some(head) = selection.head {
        return split_lines(text).into_iter().take(head).collect();
    }
    text.to_string()
}

/// Text-mode write transform: a newline before and/or after the content.
pub(crate) fn apply_text_newlines(content: &str, prepend: bool, append: bool) -> String {
    let mut out = String::with_capacity(content.len() + usize::from(prepend) + usize::from(append));
    if prepend {
        out.push('\n');
    }
    out.push_str(content);
    if append {
        out.push('\n');
    }
    out
}

/// Splits `text` into lines keeping the terminators; empty text has no lines.
pub(crate) fn text_lines(text: &str) -> Vec<String> {
    split_lines(text).into_iter().map(str::to_string).collect()
}

/// Splits `bytes` into fixed-size slices; `chunk_size == 0` means
/// [`DEFAULT_STREAM_CHUNK_SIZE`].
pub(crate) fn byte_chunks(bytes: &[u8], chunk_size: usize) -> Vec<Vec<u8>> {
    let chunk_size = if chunk_size == 0 {
        DEFAULT_STREAM_CHUNK_SIZE
    } else {
        chunk_size
    };
    bytes.chunks(chunk_size).map(<[u8]>::to_vec).collect()
}

/// The extension of a file name including the dot (`".txt"`).
///
/// Follows Python's `os.path.splitext`: a dotfile such as `.bashrc` has no
/// extension, and `archive.tar.gz` yields `".gz"`.
pub(crate) fn file_extension(name: &str) -> Option<String> {
    let base = basename(name);
    let index = base.rfind('.')?;
    if index == 0 {
        return None;
    }
    Some(base[index..].to_string())
}

/// The last component of `path`, ignoring trailing separators (Python `basename`).
pub(crate) fn basename(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    match trimmed.rfind('/') {
        Some(index) => trimmed[index + 1..].to_string(),
        None => trimmed.to_string(),
    }
}

/// POSIX shell single-quote escaping, equivalent to Python's `shlex.quote`.
///
/// The empty string becomes `''`, a string made only of shell-safe characters
/// passes through unchanged, and anything else is wrapped in single quotes with
/// embedded quotes escaped as `'\''`.
pub(crate) fn posix_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    let safe = value.chars().all(|character| {
        character.is_ascii_alphanumeric()
            || matches!(
                character,
                '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-'
            )
    });
    if safe {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Sorts `items` in place by name, modification time, or size.
pub(crate) fn sort_items(items: &mut [FileSystemItem], sort_by: SortBy, descending: bool) {
    items.sort_by(|left, right| {
        let ordering = match sort_by {
            SortBy::Name => left.name.cmp(&right.name),
            SortBy::ModifiedTime => left.modified_time.cmp(&right.modified_time),
            SortBy::Size => left.size.cmp(&right.size),
        };
        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });
}

/// Keeps entries whose `type` matches one of `file_types` exactly.
///
/// `None` and an empty list keep every entry, mirroring the Python providers'
/// `if file_types:` guard.
#[allow(clippy::ref_option)] // The borrowed `Option` is the shape the providers call.
pub(crate) fn filter_file_types(
    items: Vec<FileSystemItem>,
    file_types: &Option<Vec<String>>,
) -> Vec<FileSystemItem> {
    match file_types {
        Some(types) if !types.is_empty() => items
            .into_iter()
            .filter(|item| {
                item.file_type
                    .as_ref()
                    .is_some_and(|extension| types.iter().any(|type_name| type_name == extension))
            })
            .collect(),
        _ => items,
    }
}

/// Splits text into lines keeping terminators, like `splitlines(keepends=True)`.
fn split_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split_inclusive('\n').collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn zero_head_and_tail_mean_no_limit() {
        let selection = normalize_line_selection(ReadMode::Text, Some(0), Some(0), None).unwrap();
        assert_eq!(selection, LineSelection::default());

        // Zero is normalized before the bytes-mode check, so it stays valid.
        let selection = normalize_line_selection(ReadMode::Bytes, Some(0), None, None).unwrap();
        assert_eq!(selection, LineSelection::default());
    }

    #[test]
    fn bytes_mode_rejects_every_selection() {
        let expected =
            "Parameters 'head', 'tail', and 'line_range' are only supported in text mode";
        assert_eq!(
            normalize_line_selection(ReadMode::Bytes, Some(2), None, None).unwrap_err(),
            expected
        );
        assert_eq!(
            normalize_line_selection(ReadMode::Bytes, None, Some(2), None).unwrap_err(),
            expected
        );
        assert_eq!(
            normalize_line_selection(ReadMode::Bytes, None, None, Some(LineRange::new(1, 2)))
                .unwrap_err(),
            expected
        );
    }

    #[test]
    fn only_one_selection_is_allowed() {
        assert_eq!(
            normalize_line_selection(ReadMode::Text, Some(1), Some(2), None).unwrap_err(),
            "head and tail cannot be specified simultaneously"
        );
        assert_eq!(
            normalize_line_selection(ReadMode::Text, None, Some(2), Some(LineRange::new(1, 2)))
                .unwrap_err(),
            "tail and line_range cannot be specified simultaneously"
        );
        assert_eq!(
            normalize_line_selection(ReadMode::Text, Some(1), None, Some(LineRange::new(1, 2)))
                .unwrap_err(),
            "head and line_range cannot be specified simultaneously"
        );
        assert_eq!(
            normalize_line_selection(ReadMode::Text, Some(1), Some(2), Some(LineRange::new(1, 2)))
                .unwrap_err(),
            "head and tail and line_range cannot be specified simultaneously"
        );
    }

    #[test]
    fn a_single_selection_passes_validation() {
        let head = normalize_line_selection(ReadMode::Text, Some(3), None, None).unwrap();
        assert_eq!(head.head, Some(3));
        let tail = normalize_line_selection(ReadMode::Text, None, Some(3), None).unwrap();
        assert_eq!(tail.tail, Some(3));
        let range =
            normalize_line_selection(ReadMode::Text, None, None, Some(LineRange::new(2, 4)))
                .unwrap();
        assert_eq!(range.line_range, Some((2, 4)));
        let none = normalize_line_selection(ReadMode::Text, None, None, None).unwrap();
        assert_eq!(none, LineSelection::default());
    }

    fn selection(
        head: Option<usize>,
        tail: Option<usize>,
        line_range: Option<(usize, usize)>,
    ) -> LineSelection {
        LineSelection {
            head,
            tail,
            line_range,
        }
    }

    #[test]
    fn head_keeps_the_first_lines_with_terminators() {
        let text = "one\ntwo\nthree\nfour\n";
        assert_eq!(
            select_text_lines(text, &selection(Some(3), None, None)),
            "one\ntwo\nthree\n"
        );
        assert_eq!(
            select_text_lines(text, &selection(Some(10), None, None)),
            text
        );
        assert_eq!(
            select_text_lines("no trailing newline", &selection(Some(1), None, None)),
            "no trailing newline"
        );
    }

    #[test]
    fn tail_keeps_the_last_lines() {
        let text = "one\ntwo\nthree\nfour\n";
        assert_eq!(
            select_text_lines(text, &selection(None, Some(2), None)),
            "three\nfour\n"
        );
        assert_eq!(
            select_text_lines(text, &selection(None, Some(99), None)),
            text
        );
        assert_eq!(
            select_text_lines("a\nb", &selection(None, Some(1), None)),
            "b"
        );
    }

    #[test]
    fn line_range_is_one_based_and_inclusive() {
        let text = "one\ntwo\nthree\nfour\nfive\n";
        assert_eq!(
            select_text_lines(text, &selection(None, None, Some((2, 4)))),
            "two\nthree\nfour\n"
        );
        assert_eq!(
            select_text_lines(text, &selection(None, None, Some((3, 3)))),
            "three\n"
        );
        assert_eq!(
            select_text_lines(text, &selection(None, None, Some((4, 2)))),
            ""
        );
        assert_eq!(
            select_text_lines(text, &selection(None, None, Some((0, 2)))),
            ""
        );
        assert_eq!(
            select_text_lines(text, &selection(None, None, Some((2, 9)))),
            ""
        );
        assert_eq!(
            select_text_lines(text, &selection(None, None, Some((9, 10)))),
            ""
        );
    }

    #[test]
    fn no_selection_returns_the_text_unchanged() {
        assert_eq!(
            select_text_lines("as\nis\n", &LineSelection::default()),
            "as\nis\n"
        );
        assert_eq!(select_text_lines("", &selection(Some(3), None, None)), "");
        assert_eq!(select_text_lines("", &selection(None, Some(3), None)), "");
    }

    #[test]
    fn newline_transform_matches_python() {
        assert_eq!(apply_text_newlines("hi", true, false), "\nhi");
        assert_eq!(apply_text_newlines("hi", false, true), "hi\n");
        assert_eq!(apply_text_newlines("hi", true, true), "\nhi\n");
        assert_eq!(apply_text_newlines("hi", false, false), "hi");
        assert_eq!(apply_text_newlines("", true, false), "\n");
    }

    #[test]
    fn text_lines_keeps_terminators_and_the_last_segment() {
        assert_eq!(
            text_lines("a\nb\nc"),
            vec!["a\n".to_string(), "b\n".to_string(), "c".to_string()]
        );
        assert_eq!(text_lines(""), Vec::<String>::new());
        assert_eq!(text_lines("\n"), vec!["\n".to_string()]);
    }

    #[test]
    fn byte_chunks_uses_the_default_for_zero() {
        assert_eq!(DEFAULT_STREAM_CHUNK_SIZE, 8192);
        let bytes = vec![7_u8; DEFAULT_STREAM_CHUNK_SIZE + 1];
        assert_eq!(byte_chunks(&bytes, 0).len(), 2);
        assert_eq!(byte_chunks(&bytes, 0)[1], vec![7_u8]);

        assert_eq!(
            byte_chunks(b"abcde", 2),
            vec![b"ab".to_vec(), b"cd".to_vec(), b"e".to_vec()]
        );
        assert_eq!(byte_chunks(b"", 4), Vec::<Vec<u8>>::new());
    }

    #[test]
    fn file_extension_follows_splitext() {
        assert_eq!(file_extension("notes.txt"), Some(".txt".to_string()));
        assert_eq!(
            file_extension("/tmp/dir/archive.tar.gz"),
            Some(".gz".to_string())
        );
        assert_eq!(file_extension(".bashrc"), None);
        assert_eq!(file_extension("dir.d/.hidden"), None);
        assert_eq!(file_extension("noext"), None);
        assert_eq!(file_extension("trailing."), Some(".".to_string()));
    }

    #[test]
    fn basename_matches_python() {
        assert_eq!(basename("/tmp/dir/file.txt"), "file.txt");
        assert_eq!(basename("file.txt"), "file.txt");
        assert_eq!(basename("/tmp/dir/"), "dir");
        assert_eq!(basename("/"), "");
        assert_eq!(basename(""), "");
    }

    #[test]
    fn posix_quote_matches_shlex_quote() {
        assert_eq!(posix_quote(""), "''");
        assert_eq!(
            posix_quote("plain-1.0_a/b:c,d@e%f+g=h"),
            "plain-1.0_a/b:c,d@e%f+g=h"
        );
        assert_eq!(posix_quote("hello world"), "'hello world'");
        assert_eq!(posix_quote("it's"), "'it'\\''s'");
        assert_eq!(posix_quote("a$b"), "'a$b'");
    }

    fn item(name: &str, modified_time: &str, size: u64, file_type: Option<&str>) -> FileSystemItem {
        FileSystemItem {
            name: name.to_string(),
            path: format!("/w/{name}"),
            size,
            modified_time: modified_time.to_string(),
            is_directory: file_type.is_none(),
            file_type: file_type.map(str::to_string),
        }
    }

    #[test]
    fn sorting_uses_the_requested_key() {
        let mut items = vec![
            item("b.txt", "2024-01-02 00:00:00", 10, Some(".txt")),
            item("a.rs", "2024-01-03 00:00:00", 30, Some(".rs")),
            item("c.md", "2024-01-01 00:00:00", 20, Some(".md")),
        ];
        sort_items(&mut items, SortBy::Name, false);
        assert_eq!(
            items
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a.rs", "b.txt", "c.md"]
        );
        sort_items(&mut items, SortBy::Name, true);
        assert_eq!(
            items
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["c.md", "b.txt", "a.rs"]
        );
        sort_items(&mut items, SortBy::ModifiedTime, false);
        assert_eq!(
            items
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["c.md", "b.txt", "a.rs"]
        );
        sort_items(&mut items, SortBy::Size, false);
        assert_eq!(
            items
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["b.txt", "c.md", "a.rs"]
        );
        sort_items(&mut items, SortBy::Size, true);
        assert_eq!(
            items
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a.rs", "c.md", "b.txt"]
        );
    }

    #[test]
    fn filtering_keeps_exact_type_matches() {
        let items = vec![
            item("a.txt", "", 1, Some(".txt")),
            item("b.rs", "", 1, Some(".rs")),
            item("dir", "", 0, None),
            item("c.TXT", "", 1, Some(".TXT")),
        ];
        let filtered = filter_file_types(items.clone(), &Some(vec![".txt".to_string()]));
        assert_eq!(
            filtered
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a.txt"],
            "matching is exact, so '.TXT' is not '.txt'"
        );
        assert_eq!(filter_file_types(items.clone(), &None).len(), 4);
        assert_eq!(filter_file_types(items, &Some(Vec::new())).len(), 4);
    }
}
