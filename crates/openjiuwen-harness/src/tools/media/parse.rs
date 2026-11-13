//! Bounds-checked byte parsing helpers and the media failure type.
//!
//! The media tools never decode their inputs: they walk the raw header bytes with a
//! [`Cursor`] and report only what the container declares. Every read is bounds-checked and
//! yields `None` instead of panicking, and a malformed or truncated header becomes a
//! [`Failure`] whose message names the problem (for example `truncated PNG header`), which
//! the tools turn into a `success: false` payload.

use std::fmt;

/// Which media family a tool inspects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    /// Image headers: PNG, JPEG, GIF, WebP, BMP.
    Image,
    /// Audio headers: WAV, FLAC, MP3, OGG.
    Audio,
}

impl MediaKind {
    /// The payload `kind` string: `"image"` or `"audio"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }
}

impl fmt::Display for MediaKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A header that could not be parsed.
///
/// The message is meant for the model: it names the format and the broken field, for
/// example `truncated WAV header: missing fmt chunk`, and is used verbatim as the payload
/// `error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// Human-readable description of the problem.
    message: String,
}

impl Failure {
    /// Builds a failure carrying `message`.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The failure message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Failure {}

/// Result of parsing one media header.
pub type ParseResult<T> = std::result::Result<T, Failure>;

/// A bounds-checked read cursor over a byte slice.
///
/// Every accessor returns `None` (or `false` for the skipping helpers) once the requested
/// range runs past the end of the slice, so a truncated header can never panic a parser.
#[derive(Debug, Clone, Copy)]
pub struct Cursor<'a> {
    /// The bytes being read.
    data: &'a [u8],
    /// Offset of the next unread byte; never greater than `data.len()`.
    position: usize,
}

impl<'a> Cursor<'a> {
    /// Starts a cursor at the first byte of `data`.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// The whole slice this cursor reads from.
    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Total length of the slice, in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the slice is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Offset of the next unread byte.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Number of unread bytes.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Takes the next `count` bytes and advances past them.
    #[must_use]
    pub fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.position.checked_add(count)?;
        let taken = self.data.get(self.position..end)?;
        self.position = end;
        Some(taken)
    }

    /// Advances past the next `count` bytes.
    ///
    /// Returns `false` (leaving the position untouched) when fewer than `count` bytes remain.
    #[must_use]
    pub fn skip(&mut self, count: usize) -> bool {
        self.take(count).is_some()
    }

    /// Moves the cursor to the absolute `position`.
    ///
    /// Returns `false` (leaving the position untouched) when `position` is past the end.
    #[must_use]
    pub fn seek(&mut self, position: usize) -> bool {
        if position > self.data.len() {
            return false;
        }
        self.position = position;
        true
    }

    /// Reads one byte.
    #[must_use]
    pub fn u8(&mut self) -> Option<u8> {
        self.take(1)?.first().copied()
    }

    /// Reads a big-endian `u16`.
    #[must_use]
    pub fn be_u16(&mut self) -> Option<u16> {
        be_u16_at(self.take(2)?, 0)
    }

    /// Reads a big-endian 24-bit integer.
    #[must_use]
    pub fn be_u24(&mut self) -> Option<u32> {
        be_u24_at(self.take(3)?, 0)
    }

    /// Reads a big-endian `u32`.
    #[must_use]
    pub fn be_u32(&mut self) -> Option<u32> {
        be_u32_at(self.take(4)?, 0)
    }

    /// Reads a big-endian `u64`.
    #[must_use]
    pub fn be_u64(&mut self) -> Option<u64> {
        be_u64_at(self.take(8)?, 0)
    }

    /// Reads a little-endian `u16`.
    #[must_use]
    pub fn le_u16(&mut self) -> Option<u16> {
        le_u16_at(self.take(2)?, 0)
    }

    /// Reads a little-endian 24-bit integer.
    #[must_use]
    pub fn le_u24(&mut self) -> Option<u32> {
        le_u24_at(self.take(3)?, 0)
    }

    /// Reads a little-endian `u32`.
    #[must_use]
    pub fn le_u32(&mut self) -> Option<u32> {
        le_u32_at(self.take(4)?, 0)
    }

    /// Reads a little-endian `u64`.
    #[must_use]
    pub fn le_u64(&mut self) -> Option<u64> {
        le_u64_at(self.take(8)?, 0)
    }
}

/// Reads a big-endian `u16` at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn be_u16_at(bytes: &[u8], offset: usize) -> Option<u16> {
    let [first, second] = window::<2>(bytes, offset)?;
    Some(u16::from_be_bytes([first, second]))
}

/// Reads a little-endian `u16` at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn le_u16_at(bytes: &[u8], offset: usize) -> Option<u16> {
    let [first, second] = window::<2>(bytes, offset)?;
    Some(u16::from_le_bytes([first, second]))
}

/// Reads a big-endian 24-bit integer at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn be_u24_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let [first, second, third] = window::<3>(bytes, offset)?;
    Some(u32::from_be_bytes([0, first, second, third]))
}

/// Reads a little-endian 24-bit integer at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn le_u24_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let [first, second, third] = window::<3>(bytes, offset)?;
    Some(u32::from_le_bytes([first, second, third, 0]))
}

/// Reads a big-endian `u32` at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn be_u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let [first, second, third, fourth] = window::<4>(bytes, offset)?;
    Some(u32::from_be_bytes([first, second, third, fourth]))
}

/// Reads a little-endian `u32` at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn le_u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let [first, second, third, fourth] = window::<4>(bytes, offset)?;
    Some(u32::from_le_bytes([first, second, third, fourth]))
}

/// Reads a little-endian `i32` at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn le_i32_at(bytes: &[u8], offset: usize) -> Option<i32> {
    let [first, second, third, fourth] = window::<4>(bytes, offset)?;
    Some(i32::from_le_bytes([first, second, third, fourth]))
}

/// Reads a big-endian `u64` at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn be_u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    let value = window::<8>(bytes, offset)?;
    Some(u64::from_be_bytes(value))
}

/// Reads a little-endian `u64` at `offset`, or `None` when the slice is too short.
#[must_use]
pub fn le_u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    let value = window::<8>(bytes, offset)?;
    Some(u64::from_le_bytes(value))
}

/// Copies a fixed-size window of `bytes` starting at `offset`.
fn window<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    let end = offset.checked_add(N)?;
    <[u8; N]>::try_from(bytes.get(offset..end)?).ok()
}

/// Renders the first four bytes of `bytes` for an error message.
///
/// Printable ASCII (including spaces) is shown as characters (`VP8 `), anything else as lowercase
/// hex (`89504e47`); an empty slice renders as `<empty>`.
#[must_use]
pub fn fourcc(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let head: Vec<u8> = bytes.iter().copied().take(4).collect();
    if head.is_empty() {
        return "<empty>".to_owned();
    }
    let mut rendered = String::with_capacity(head.len() * 2);
    if head
        .iter()
        .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
    {
        for byte in &head {
            rendered.push(char::from(*byte));
        }
    } else {
        for byte in &head {
            let _ = write!(rendered, "{byte:02x}");
        }
    }
    rendered
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cursor_reads_both_endiannesses() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(cursor.be_u16(), Some(0x0102));
        assert_eq!(cursor.position(), 2);
        assert_eq!(cursor.le_u16(), Some(0x0403));
        assert_eq!(cursor.be_u24(), Some(0x05_06_07));
        assert_eq!(cursor.remaining(), 1);
        assert_eq!(cursor.be_u32(), None);
        assert_eq!(cursor.position(), 7);
    }

    #[test]
    fn cursor_reads_little_endian_u32_and_u64() {
        let bytes = [
            0x01, 0x00, 0x00, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
        ];
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(cursor.le_u32(), Some(1));
        assert_eq!(cursor.le_u64(), Some(0x0102_0304_0506_0708));
        assert_eq!(cursor.remaining(), 0);
    }

    #[test]
    fn cursor_never_reads_past_the_end() {
        let bytes = [0xFF, 0xD8];
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(cursor.take(3), None);
        assert_eq!(cursor.position(), 0);
        assert_eq!(cursor.take(2), Some([0xFF, 0xD8].as_slice()));
        assert_eq!(cursor.u8(), None);
        assert!(!cursor.skip(1));
        assert_eq!(cursor.remaining(), 0);
        assert_eq!(cursor.len(), 2);
        assert!(!cursor.is_empty());
    }

    #[test]
    fn cursor_seek_is_bounds_checked() {
        let bytes = [0, 1, 2, 3];
        let mut cursor = Cursor::new(&bytes);
        assert!(!cursor.seek(5));
        assert_eq!(cursor.position(), 0);
        assert!(cursor.seek(2));
        assert_eq!(cursor.u8(), Some(2));
        assert!(cursor.seek(4));
        assert_eq!(cursor.remaining(), 0);
    }

    #[test]
    fn offset_helpers_are_bounds_checked() {
        let bytes = [0x00, 0x01, 0x02, 0x03];
        assert_eq!(be_u16_at(&bytes, 0), Some(0x0001));
        assert_eq!(le_u16_at(&bytes, 2), Some(0x0302));
        assert_eq!(be_u24_at(&bytes, 1), Some(0x01_02_03));
        assert_eq!(le_u24_at(&bytes, 0), Some(0x02_01_00));
        assert_eq!(be_u32_at(&bytes, 0), Some(0x0001_0203));
        assert_eq!(le_u32_at(&bytes, 0), Some(0x0302_0100));
        assert_eq!(be_u64_at(&bytes, 0), None);
        assert_eq!(le_u64_at(&bytes, 0), None);
        assert_eq!(be_u16_at(&bytes, usize::MAX), None);
        assert_eq!(be_u24_at(&bytes, 2), None);
    }

    #[test]
    fn fourcc_renders_ascii_and_hex() {
        assert_eq!(fourcc(b"WEBPmore"), "WEBP");
        assert_eq!(fourcc(&[0x89, 0x50, 0x4E, 0x47]), "89504e47");
        assert_eq!(fourcc(b"AB"), "AB");
        assert_eq!(fourcc(b""), "<empty>");
    }

    #[test]
    fn failure_and_kind_render_for_the_model() {
        let failure = Failure::new("truncated PNG header");
        assert_eq!(failure.message(), "truncated PNG header");
        assert_eq!(failure.to_string(), "truncated PNG header");
        assert_eq!(MediaKind::Image.as_str(), "image");
        assert_eq!(MediaKind::Audio.as_str(), "audio");
        assert_eq!(MediaKind::Image.to_string(), "image");
        assert_eq!(MediaKind::Audio.to_string(), "audio");
    }
}
