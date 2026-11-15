//! Pure slicing, validation, and chunking helpers for the local filesystem backend.
//!
//! Everything here is synchronous, allocation-light, and free of I/O so it can be tested
//! without a filesystem. The local backend ([`crate::fs::local`]) uses these helpers to
//! reproduce Python's `read_file` parameter handling
//! (`_ReadParams.validate_binary_mode` / `validate_mutually_exclusive`,
//! `_read_head` / `_read_tail` / `_read_line_range`, `str.splitlines(True)`) and its
//! `write_file` newline handling.
//!
//! ```
//! use openjiuwen_sysop::fs::slice::{split_lines_keep_ends, take_first_chunk};
//!
//! assert_eq!(split_lines_keep_ends("a\r\nb\n"), vec!["a\r\n", "b\n"]);
//! assert_eq!(take_first_chunk(b"abcdef", 2), (&b"ab"[..], true));
//! ```

use crate::fs::types::{FileContent, LineRange, ReadFileParams, ReadMode, WriteFileParams};

/// One normalized line selection.
///
/// Python parity: `head`, `tail`, and `line_range` after
/// `_ReadParams.validate_non_negative` turned `0` into `None`. A `usize` cannot be
/// negative, so negative selections (`head=-1`, `tail=-1`) arrive as huge values through
/// the JSON API and are clamped by [`select_lines`] rather than producing an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSelection {
    /// Keep the first `n` lines.
    Head(usize),
    /// Keep the last `n` lines.
    Tail(usize),
    /// Keep the 1-based, inclusive range `start..=end`.
    LineRange {
        /// First line to keep, 1-based.
        start: usize,
        /// Last line to keep, 1-based and inclusive.
        end: usize,
    },
}

/// Validates a read request and normalizes `head`/`tail`/`line_range`.
///
/// The checks run in Python's order — bytes-mode rejection first, then the mutual
/// exclusion of the three line selections — and only **non-zero** values participate, so
/// `head = Some(0)` means "not specified" and a `line_range` of `0..=0` is ignored too.
///
/// # Errors
///
/// Returns Python's reason text (without the `199003` status envelope) when a line
/// selection is used in bytes mode or when two selections are combined.
pub fn validate_read_params(params: &ReadFileParams) -> Result<Option<LineSelection>, String> {
    let head = params.head.filter(|value| *value != 0);
    let tail = params.tail.filter(|value| *value != 0);
    let line_range = params
        .line_range
        .filter(|range: &LineRange| range.start != 0 || range.end != 0);

    if params.mode == ReadMode::Bytes && (head.is_some() || tail.is_some() || line_range.is_some())
    {
        return Err(
            "Parameters 'head', 'tail', and 'line_range' are only supported in text mode"
                .to_string(),
        );
    }
    if tail.is_some() && head.is_some() {
        return Err("tail and head cannot be specified simultaneously".to_string());
    }
    if tail.is_some() && line_range.is_some() {
        return Err("tail and line_range cannot be specified simultaneously".to_string());
    }
    if head.is_some() && line_range.is_some() {
        return Err("head and line_range cannot be specified simultaneously".to_string());
    }
    if let Some(count) = tail {
        return Ok(Some(LineSelection::Tail(count)));
    }
    if let Some(count) = head {
        return Ok(Some(LineSelection::Head(count)));
    }
    Ok(line_range.map(|range| LineSelection::LineRange {
        start: range.start,
        end: range.end,
    }))
}

/// Splits `text` into lines, keeping each line terminator (Python `str.splitlines(True)`).
///
/// `\n`, `\r\n` (kept verbatim as one terminator), and a lone `\r` terminate a line; a
/// trailing terminator therefore produces no extra empty element and empty text yields no
/// lines. Exotic Unicode line separators (`\v`, `\f`, `\u{1c}`–`\u{1e}`, `\u{85}`,
/// `\u{2028}`, `\u{2029}`) are **not** treated as terminators — a documented deviation
/// from Python that keeps chunk boundaries aligned with `\n`-readers such as `str::lines`.
#[must_use]
pub fn split_lines_keep_ends(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0_usize;
    let mut index = 0_usize;
    while index < bytes.len() {
        match bytes[index] {
            b'\n' => {
                lines.push(&text[start..=index]);
                index += 1;
                start = index;
            }
            b'\r' => {
                let end = if bytes.get(index + 1) == Some(&b'\n') {
                    index + 2
                } else {
                    index + 1
                };
                lines.push(&text[start..end]);
                index = end;
                start = index;
            }
            _ => index += 1,
        }
    }
    if start < bytes.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Applies a normalized selection to `lines`.
///
/// Head keeps the first `n` lines, tail the last `n`, and a line range is 1-based and
/// inclusive: `start == 0 || end == 0 || start > end` yields an empty result (Python
/// returns an empty string, not an error) and `end` past EOF clamps to the end. `None`
/// keeps every line. The result borrows from the input; no line is copied.
#[must_use]
pub fn select_lines<'a>(lines: &[&'a str], selection: Option<LineSelection>) -> Vec<&'a str> {
    match selection {
        None => lines.to_vec(),
        Some(LineSelection::Head(count)) => lines.iter().take(count).copied().collect(),
        Some(LineSelection::Tail(count)) => {
            let start = lines.len().saturating_sub(count);
            lines.get(start..).map(<[&str]>::to_vec).unwrap_or_default()
        }
        Some(LineSelection::LineRange { start, end }) => {
            if start == 0 || end == 0 || start > end {
                return Vec::new();
            }
            let from = start - 1;
            if from >= lines.len() {
                return Vec::new();
            }
            let to = end.min(lines.len());
            lines
                .get(from..to)
                .map(<[&str]>::to_vec)
                .unwrap_or_default()
        }
    }
}

/// Takes the first `chunk_size` bytes and reports whether they were cut.
///
/// `chunk_size == 0` keeps the whole buffer (Python's "unlimited" read).
#[must_use]
pub fn take_first_chunk(bytes: &[u8], chunk_size: usize) -> (&[u8], bool) {
    if chunk_size == 0 || bytes.len() <= chunk_size {
        return (bytes, false);
    }
    match bytes.get(..chunk_size) {
        Some(chunk) => (chunk, true),
        None => (bytes, false),
    }
}

/// Normalizes a requested streaming chunk size: `0` means "use the default".
#[must_use]
pub const fn effective_chunk_size(requested: usize, default: usize) -> usize {
    if requested == 0 {
        default
    } else {
        requested
    }
}

/// Byte length of `line` once encoded with `encoding` (Python `len(line.encode(encoding))`).
///
/// # Errors
///
/// Returns the encoder's message when `encoding` is not a known write encoding or cannot
/// represent `line`. Callers that report a size to the user use this so the failure
/// surfaces instead of a made-up length.
pub fn try_encoded_len(line: &str, encoding: &str) -> Result<u64, String> {
    let len = crate::encoding::encode_text(line, encoding)
        .map_err(|error| error.message().to_string())?
        .len();
    Ok(u64::try_from(len).unwrap_or(u64::MAX))
}
/// Byte length of `line` once encoded with `encoding`, as a bounded-size **estimate**.
///
/// When `encoding` cannot represent `line` (an unknown name, or a character the codec does
/// not cover) the UTF-8 length is returned and a warning is logged, so sizing a buffer never
/// fails. This is only safe for a size probe: a write must go through [`write_bytes`], which
/// propagates the encoder error, and a caller that reports the size to the user should use
/// [`try_encoded_len`] so the failure is visible.
#[must_use]
pub fn encoded_len(line: &str, encoding: &str) -> u64 {
    match try_encoded_len(line, encoding) {
        Ok(len) => len,
        Err(reason) => {
            tracing::warn!(
                encoding,
                %reason,
                "size probe fell back to the UTF-8 length"
            );
            u64::try_from(line.len()).unwrap_or(u64::MAX)
        }
    }
}

/// Encodes the content of a write request exactly the way Python's `write_file` does.
///
/// Text content gets a leading `"\n"` when `prepend_newline` is set and a trailing one
/// when `append_newline` is set (both default per [`WriteFileParams`]), then it is encoded
/// with `params.encoding`. Byte content is written verbatim — the newline flags do not
/// apply to it.
///
/// # Errors
///
/// Returns the encoder's message (without the `199003` status envelope) when
/// `params.encoding` is not a recognized write encoding or cannot represent the content, so
/// `write_file` turns it into a `199003` failure instead of writing different bytes than the
/// caller asked for.
pub fn write_bytes(params: &WriteFileParams) -> Result<Vec<u8>, String> {
    match &params.content {
        FileContent::Bytes(bytes) => Ok(bytes.clone()),
        FileContent::Text(text) => {
            let mut content = String::with_capacity(text.len() + 2);
            if params.prepend_newline {
                content.push('\n');
            }
            content.push_str(text);
            if params.append_newline {
                content.push('\n');
            }
            crate::encoding::encode_text(&content, &params.encoding)
                .map_err(|error| error.message().to_string())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn text_read() -> ReadFileParams {
        ReadFileParams::text("a.txt")
    }

    #[test]
    fn zero_values_are_not_selections() {
        let mut params = text_read();
        params.head = Some(0);
        params.tail = Some(0);
        params.line_range = Some(LineRange::new(0, 0));
        assert_eq!(validate_read_params(&params).unwrap(), None);
    }

    #[test]
    fn bytes_mode_rejection_wins_over_mutual_exclusion() {
        let mut params = ReadFileParams::bytes("a.bin");
        params.head = Some(1);
        params.tail = Some(2);
        let error = validate_read_params(&params).unwrap_err();
        assert_eq!(
            error,
            "Parameters 'head', 'tail', and 'line_range' are only supported in text mode"
        );

        // A zero-valued selection does not trigger the bytes-mode rejection.
        let mut params = ReadFileParams::bytes("a.bin");
        params.head = Some(0);
        assert_eq!(validate_read_params(&params).unwrap(), None);
    }

    #[test]
    fn each_pair_reports_its_own_python_message() {
        let mut params = text_read();
        params.tail = Some(2);
        params.head = Some(1);
        assert_eq!(
            validate_read_params(&params).unwrap_err(),
            "tail and head cannot be specified simultaneously"
        );

        let mut params = text_read();
        params.tail = Some(2);
        params.line_range = Some(LineRange::new(1, 2));
        assert_eq!(
            validate_read_params(&params).unwrap_err(),
            "tail and line_range cannot be specified simultaneously"
        );

        let mut params = text_read();
        params.head = Some(2);
        params.line_range = Some(LineRange::new(1, 2));
        assert_eq!(
            validate_read_params(&params).unwrap_err(),
            "head and line_range cannot be specified simultaneously"
        );
    }

    #[test]
    fn single_selection_is_normalized() {
        let mut params = text_read();
        params.head = Some(3);
        assert_eq!(
            validate_read_params(&params).unwrap(),
            Some(LineSelection::Head(3))
        );

        let mut params = text_read();
        params.tail = Some(4);
        assert_eq!(
            validate_read_params(&params).unwrap(),
            Some(LineSelection::Tail(4))
        );

        let mut params = text_read();
        params.line_range = Some(LineRange::new(2, 5));
        assert_eq!(
            validate_read_params(&params).unwrap(),
            Some(LineSelection::LineRange { start: 2, end: 5 })
        );
    }

    #[test]
    fn split_keeps_native_terminators() {
        assert_eq!(split_lines_keep_ends(""), Vec::<&str>::new());
        assert_eq!(split_lines_keep_ends("abc"), vec!["abc"]);
        assert_eq!(split_lines_keep_ends("a\nb\n"), vec!["a\n", "b\n"]);
        assert_eq!(split_lines_keep_ends("a\r\nb\r\n"), vec!["a\r\n", "b\r\n"]);
        assert_eq!(split_lines_keep_ends("a\rb"), vec!["a\r", "b"]);
        assert_eq!(split_lines_keep_ends("a\r\r\n"), vec!["a\r", "\r\n"]);
        assert_eq!(
            split_lines_keep_ends("héllo\nwörld"),
            vec!["héllo\n", "wörld"]
        );
    }

    #[test]
    fn head_tail_and_range_selection() {
        let lines = vec!["1\n", "2\n", "3\n", "4\n"];
        assert_eq!(select_lines(&lines, None), lines);
        assert_eq!(
            select_lines(&lines, Some(LineSelection::Head(2))),
            vec!["1\n", "2\n"]
        );
        assert_eq!(select_lines(&lines, Some(LineSelection::Head(99))), lines);
        assert_eq!(
            select_lines(&lines, Some(LineSelection::Tail(2))),
            vec!["3\n", "4\n"]
        );
        assert_eq!(
            select_lines(&lines, Some(LineSelection::Tail(99))),
            lines,
            "a tail longer than the file keeps everything"
        );
        assert_eq!(
            select_lines(&lines, Some(LineSelection::LineRange { start: 2, end: 3 })),
            vec!["2\n", "3\n"]
        );
        assert_eq!(
            select_lines(&lines, Some(LineSelection::LineRange { start: 3, end: 99 })),
            vec!["3\n", "4\n"],
            "the end clamps to EOF"
        );
    }

    #[test]
    fn invalid_ranges_are_empty_not_errors() {
        let lines = vec!["1\n", "2\n"];
        for selection in [
            LineSelection::LineRange { start: 0, end: 2 },
            LineSelection::LineRange { start: 1, end: 0 },
            LineSelection::LineRange { start: 2, end: 1 },
            LineSelection::LineRange { start: 9, end: 10 },
        ] {
            assert!(
                select_lines(&lines, Some(selection)).is_empty(),
                "{selection:?} must select nothing"
            );
        }
        assert_eq!(
            select_lines(&[], Some(LineSelection::Tail(3))),
            Vec::<&str>::new()
        );
        assert_eq!(
            select_lines(&lines, Some(LineSelection::Head(0))),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn chunks_respect_the_requested_size() {
        assert_eq!(take_first_chunk(b"abcdef", 0), (&b"abcdef"[..], false));
        assert_eq!(take_first_chunk(b"abcdef", 6), (&b"abcdef"[..], false));
        assert_eq!(take_first_chunk(b"abcdef", 2), (&b"ab"[..], true));
        assert_eq!(take_first_chunk(b"", 4), (&b""[..], false));
    }

    #[test]
    fn chunk_normalization_uses_the_default_for_zero() {
        assert_eq!(effective_chunk_size(0, 8192), 8192);
        assert_eq!(effective_chunk_size(10, 8192), 10);
    }

    #[test]
    fn encoded_length_tracks_the_encoding() {
        assert_eq!(encoded_len("abc", "utf-8"), 3);
        assert_eq!(encoded_len("é", "utf-8"), 2);
        assert_eq!(encoded_len("é", "utf-16le"), 2);
        // Latin-1 and the CJK codecs report their real byte counts, not the UTF-8 length.
        assert_eq!(encoded_len("é", "latin-1"), 1);
        assert_eq!(encoded_len("中文", "gb18030"), 4);
        assert_eq!(encoded_len("中文", "big5"), 4);
        assert_eq!(encoded_len("日本語", "shift-jis"), 6);
        // The estimate never fails: a name the encoder rejects degrades to the UTF-8 length.
        assert_eq!(encoded_len("abc", "not-an-encoding"), 3);
        assert_eq!(encoded_len("中", "latin-1"), 3);
    }

    #[test]
    fn try_encoded_len_reports_encoding_failures() {
        assert_eq!(try_encoded_len("中文", "gbk"), Ok(4));
        assert_eq!(try_encoded_len("é", "latin-1"), Ok(1));
        assert_eq!(try_encoded_len("", "gb18030"), Ok(0));
        let reason = try_encoded_len("中", "latin-1").unwrap_err();
        assert!(reason.contains("latin-1"), "{reason}");
        assert!(reason.contains("U+4E2D"), "{reason}");
        assert!(try_encoded_len("x", "koi8-r")
            .unwrap_err()
            .contains("koi8-r"));
    }

    #[test]
    fn write_bytes_applies_newline_flags_to_text_only() {
        let params = WriteFileParams::new("a.txt", "hi");
        assert_eq!(write_bytes(&params).unwrap(), b"\nhi".to_vec());

        let params = WriteFileParams::raw("a.txt", "hi");
        assert_eq!(write_bytes(&params).unwrap(), b"hi".to_vec());

        let mut params = WriteFileParams::raw("a.txt", "hi");
        params.append_newline = true;
        assert_eq!(write_bytes(&params).unwrap(), b"hi\n".to_vec());

        let mut params = WriteFileParams::new("a.bin", vec![1_u8, 2]);
        params.append_newline = true;
        assert_eq!(
            write_bytes(&params).unwrap(),
            vec![1_u8, 2],
            "byte content ignores the newline flags"
        );
    }
}
