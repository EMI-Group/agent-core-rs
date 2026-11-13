//! Archive sniffing and `ustar` unpacking for the [`super::pipeline`] scrape stage.
//!
//! The `skill_omni_creation` pipeline accepts a URL that may resolve to an
//! archive rather than an HTML page, so [`detect`] identifies the container from
//! its magic bytes and [`ArchiveUnpacker`] turns a `ustar` archive into ordered
//! [`ArchiveEntry`] values.
//!
//! `tar` is parsed in-process: no crate is added to the locked, offline
//! dependency set. `zip` and `gzip` containers are recognized but deliberately
//! not unpacked — [`ArchiveUnpacker::unpack`] returns
//! [`Error::unsupported`](openjiuwen_common::error::Error::unsupported) naming
//! the format instead of silently degrading.
//!
//! Every entry path is validated before it is returned: absolute paths, `..`
//! components, and backslashes are rejected, so a caller can never write outside
//! the bundle root.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
/// The size of one `tar` header or data block.
const BLOCK: usize = 512;
/// Offset of the `tar` name field.
const NAME_OFFSET: usize = 0;
/// Length of the `tar` name field.
const NAME_LEN: usize = 100;
/// Offset of the `tar` size field.
const SIZE_OFFSET: usize = 124;
/// Length of the `tar` size field.
const SIZE_LEN: usize = 12;
/// Offset of the `tar` checksum field.
const CHECKSUM_OFFSET: usize = 148;
/// Length of the `tar` checksum field.
const CHECKSUM_LEN: usize = 8;
/// Offset of the `tar` typeflag field.
const TYPEFLAG_OFFSET: usize = 156;
/// Offset of the `tar` magic field.
const MAGIC_OFFSET: usize = 257;
/// Length of the `tar` magic field.
const MAGIC_LEN: usize = 6;
/// Offset of the `ustar` prefix field.
const PREFIX_OFFSET: usize = 345;
/// Length of the `ustar` prefix field.
const PREFIX_LEN: usize = 155;
/// The POSIX `ustar` magic value, which enables the `prefix` field.
const USTAR_MAGIC: &[u8] = b"ustar\0";
/// A container format recognized by [`detect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveFormat {
    /// A `tar` archive, including the POSIX `ustar` flavor.
    Tar,
    /// A `zip` archive.
    Zip,
    /// A `gzip`-compressed stream.
    Gzip,
}
impl ArchiveFormat {
    /// The lowercase wire name of the format: `"tar"`, `"zip"`, or `"gzip"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tar => "tar",
            Self::Zip => "zip",
            Self::Gzip => "gzip",
        }
    }
}
impl fmt::Display for ArchiveFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
/// One entry read out of an archive.
///
/// `path` is the archive-relative path with `/` separators; `bytes` holds the
/// file contents and is empty for directories; `is_dir` marks directory entries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ArchiveEntry {
    /// Archive-relative path of the entry.
    pub path: String,
    /// File contents; empty for directory entries.
    pub bytes: Vec<u8>,
    /// Whether the entry is a directory.
    pub is_dir: bool,
}
/// Associated functions for sniffing and unpacking archives.
///
/// This type is never instantiated: it only groups the entry points so call
/// sites read as [`ArchiveUnpacker::unpack`] / [`ArchiveUnpacker::unpack_auto`],
/// matching the associated-function style of `SkillBundle`.
#[derive(Debug, Clone, Copy)]
pub struct ArchiveUnpacker;
impl ArchiveUnpacker {
    /// Unpacks `bytes` as `format`, returning entries in archive order.
    ///
    /// `tar` (including POSIX `ustar`) is parsed in-process. `zip` and `gzip`
    /// are deferred: their decompression would need a crate that the offline
    /// workspace does not provide.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for malformed, truncated, or unsafe `tar` content, and
    /// [`Error::unsupported`](openjiuwen_common::error::Error::unsupported) for
    /// `zip` and `gzip`.
    pub fn unpack(format: ArchiveFormat, bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
        match format {
            ArchiveFormat::Tar => unpack_tar(bytes),
            ArchiveFormat::Zip | ArchiveFormat::Gzip => Err(Error::unsupported(format!(
                "{format} archive unpacking requires a compression dependency that the offline \
                 workspace does not provide"
            ))),
        }
    }
    /// Sniffs the format with [`detect`] and then [`unpack`](Self::unpack)s it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when the bytes carry no recognized archive signature, plus everything
    /// [`unpack`](Self::unpack) can return.
    pub fn unpack_auto(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
        let format = detect(bytes).ok_or_else(|| {
            Error::validation(format!(
                "unrecognized archive: no gzip, zip, or ustar signature in {} bytes",
                bytes.len()
            ))
        })?;
        Self::unpack(format, bytes)
    }
}
/// Sniffs the archive format of `bytes`, or `None` when nothing matches.
///
/// Recognized signatures are `gzip` (`1f 8b`), `zip` (`PK\x03\x04` and the empty
/// archive `PK\x05\x06`), and `tar` (`ustar` at byte offset 257).
#[must_use]
pub fn detect(bytes: &[u8]) -> Option<ArchiveFormat> {
    if bytes.starts_with(&[0x1f, 0x8b]) {
        Some(ArchiveFormat::Gzip)
    } else if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
        Some(ArchiveFormat::Zip)
    } else if bytes.get(MAGIC_OFFSET..MAGIC_OFFSET + 5) == Some(b"ustar".as_slice()) {
        Some(ArchiveFormat::Tar)
    } else {
        None
    }
}
/// Parses a `tar` archive, appending entries in archive order.
fn unpack_tar(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
    let mut entries = Vec::new();
    let mut offset = 0usize;
    let mut previous_zero = false;
    let mut long_name: Option<String> = None;
    while let Some(end) = offset.checked_add(BLOCK) {
        let Some(header) = bytes.get(offset..end) else {
            break;
        };
        if header.iter().all(|&byte| byte == 0) {
            if previous_zero {
                break;
            }
            previous_zero = true;
            offset = end;
            continue;
        }
        previous_zero = false;
        verify_checksum(header, offset)?;
        let size = parse_size(header, offset)?;
        let data_start = offset.checked_add(BLOCK).ok_or_else(|| {
            Error::validation(format!("tar header offset overflow at byte {offset}"))
        })?;
        let data_end = data_start.checked_add(size).ok_or_else(|| {
            Error::validation(format!(
                "tar entry at byte {offset} declares an unaddressable size of {size} bytes"
            ))
        })?;
        let data = bytes.get(data_start..data_end).ok_or_else(|| {
            Error::validation(format!(
                "truncated tar entry data at byte {offset}: {size} bytes declared but only {} \
                 remain",
                bytes.len().saturating_sub(data_start)
            ))
        })?;
        let padding = (BLOCK - size % BLOCK) % BLOCK;
        let typeflag = header.get(TYPEFLAG_OFFSET).copied().unwrap_or(0);
        match typeflag {
            b'0' | b'\0' => {
                let path = entry_path(header, &mut long_name)?;
                entries.push(ArchiveEntry {
                    path,
                    bytes: data.to_vec(),
                    is_dir: false,
                });
            }
            b'5' => {
                let path = entry_path(header, &mut long_name)?;
                entries.push(ArchiveEntry {
                    path,
                    bytes: Vec::new(),
                    is_dir: true,
                });
            }
            b'L' => {
                long_name = Some(payload_text(data));
            }
            // GNU long linkname, PAX per-file (`x`) and global (`g`) headers are
            // consumed by size and otherwise ignored.
            b'K' | b'x' | b'g' => {}
            // Hard links, symlinks, devices, and FIFOs carry no file contents.
            _ => {
                long_name = None;
            }
        }
        offset = data_start
            .checked_add(size)
            .and_then(|base| base.checked_add(padding))
            .ok_or_else(|| {
                Error::validation(format!("tar entry offset overflow after byte {offset}"))
            })?;
    }
    Ok(entries)
}
/// Resolves the next entry path, preferring a GNU long-name override.
fn entry_path(header: &[u8], long_name: &mut Option<String>) -> Result<String> {
    match long_name.take() {
        Some(name) => validate_path(&name),
        None => validate_path(&header_path(header)),
    }
}
/// Reads `prefix/name` from a `tar` header, honoring the `ustar` prefix field.
fn header_path(header: &[u8]) -> String {
    let name = text_field(header, NAME_OFFSET, NAME_LEN);
    let magic = field(header, MAGIC_OFFSET, MAGIC_LEN);
    let prefix = if magic == USTAR_MAGIC {
        text_field(header, PREFIX_OFFSET, PREFIX_LEN)
    } else {
        String::new()
    };
    if prefix.is_empty() {
        name
    } else {
        format!("{prefix}/{name}")
    }
}
/// Rejects paths the pipeline could use to escape the bundle root.
fn validate_path(path: &str) -> Result<String> {
    if path.is_empty() {
        return Err(Error::validation("unsafe archive path: empty entry name"));
    }
    if path.starts_with('/') || is_windows_absolute(path) {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: absolute paths are not allowed"
        )));
    }
    if path.split('/').any(|component| component == "..") {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: `..` components are not allowed"
        )));
    }
    if path.contains('\\') {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: backslashes are not allowed"
        )));
    }
    Ok(path.to_string())
}
/// Whether `path` starts with a Windows drive prefix such as `C:`.
fn is_windows_absolute(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}
/// Verifies the `tar` header checksum at `offset`.
///
/// The stored octal value is compared against the unsigned sum with the
/// checksum field read as ASCII spaces; the `GNU` signed variant (each byte
/// interpreted as `i8`) is accepted too.
fn verify_checksum(header: &[u8], offset: usize) -> Result<()> {
    let stored = parse_number(field(header, CHECKSUM_OFFSET, CHECKSUM_LEN)).ok_or_else(|| {
        Error::validation(format!(
            "invalid tar checksum field in header at byte {offset}"
        ))
    })?;
    let mut unsigned = 0u64;
    let mut signed = 0i64;
    for (index, &byte) in header.iter().enumerate() {
        let byte = if (CHECKSUM_OFFSET..CHECKSUM_OFFSET + CHECKSUM_LEN).contains(&index) {
            b' '
        } else {
            byte
        };
        unsigned += u64::from(byte);
        signed += i64::from(i8::from_ne_bytes([byte]));
    }
    if stored == unsigned || i64::try_from(stored).is_ok_and(|value| value == signed) {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "tar checksum mismatch in header at byte {offset}: stored {stored}, computed \
             {unsigned}"
        )))
    }
}
/// Parses the octal (or `GNU` base-256) size field of a `tar` header.
fn parse_size(header: &[u8], offset: usize) -> Result<usize> {
    let value = parse_number(field(header, SIZE_OFFSET, SIZE_LEN)).ok_or_else(|| {
        Error::validation(format!("invalid tar size field in header at byte {offset}"))
    })?;
    usize::try_from(value).map_err(|_| {
        Error::validation(format!(
            "tar entry at byte {offset} declares a size too large to address: {value}"
        ))
    })
}
/// Parses a numeric `tar` header field.
///
/// Octal fields are read after leading `NUL`s and spaces, stopping at the first
/// non-octal byte. A high bit set in the first byte selects the `GNU` base-256
/// encoding, read big-endian with the high bit treated as a positive sign bit.
fn parse_number(raw: &[u8]) -> Option<u64> {
    if raw.first().is_some_and(|byte| byte & 0x80 != 0) {
        let mut value = 0u64;
        for (index, &byte) in raw.iter().enumerate() {
            let byte = if index == 0 { byte & 0x7f } else { byte };
            value = value.checked_mul(256)?.checked_add(u64::from(byte))?;
        }
        return Some(value);
    }
    let start = raw.iter().position(|&byte| byte != 0 && byte != b' ')?;
    let mut value = 0u64;
    let mut seen = false;
    for &byte in &raw[start..] {
        if !(b'0'..=b'7').contains(&byte) {
            break;
        }
        value = value.checked_mul(8)?.checked_add(u64::from(byte - b'0'))?;
        seen = true;
    }
    seen.then_some(value)
}
/// Reads an ASCII header field, trimming trailing `NUL`s and surrounding spaces.
fn text_field(header: &[u8], offset: usize, len: usize) -> String {
    let raw = field(header, offset, len);
    let end = raw.iter().position(|&byte| byte == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end])
        .trim_matches(' ')
        .to_string()
}
/// Reads the text payload of a GNU long-name header up to its `NUL` terminator.
fn payload_text(data: &[u8]) -> String {
    let end = data
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(data.len());
    String::from_utf8_lossy(&data[..end]).into_owned()
}
/// Bounds-checked slice of one header field.
fn field(header: &[u8], offset: usize, len: usize) -> &[u8] {
    header
        .get(offset..offset.saturating_add(len))
        .unwrap_or(&[])
}
#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    fn write_text(header: &mut [u8], offset: usize, len: usize, text: &str) {
        let bytes = text.as_bytes();
        let end = bytes.len().min(len);
        header[offset..offset + end].copy_from_slice(&bytes[..end]);
    }
    fn write_octal(header: &mut [u8], offset: usize, len: usize, value: u64) {
        let text = format!("{:0width$o}\0", value, width = len - 1);
        header[offset..offset + len].copy_from_slice(text.as_bytes());
    }
    fn write_checksum(header: &mut [u8; BLOCK], signed: bool) {
        for byte in &mut header[CHECKSUM_OFFSET..CHECKSUM_OFFSET + CHECKSUM_LEN] {
            *byte = b' ';
        }
        let sum: i64 = if signed {
            header
                .iter()
                .map(|&byte| i64::from(i8::from_ne_bytes([byte])))
                .sum()
        } else {
            header.iter().map(|&byte| i64::from(byte)).sum()
        };
        let text = format!("{sum:06o}\0 ");
        header[CHECKSUM_OFFSET..CHECKSUM_OFFSET + CHECKSUM_LEN].copy_from_slice(text.as_bytes());
    }
    fn header_with(name: &str, size: usize, typeflag: u8, prefix: &str) -> [u8; BLOCK] {
        let mut header = [0u8; BLOCK];
        write_text(&mut header, NAME_OFFSET, NAME_LEN, name);
        write_octal(&mut header, SIZE_OFFSET, SIZE_LEN, size as u64);
        write_octal(&mut header, 100, 8, 0o644);
        header[TYPEFLAG_OFFSET] = typeflag;
        header[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC_LEN].copy_from_slice(USTAR_MAGIC);
        if !prefix.is_empty() {
            write_text(&mut header, PREFIX_OFFSET, PREFIX_LEN, prefix);
        }
        write_checksum(&mut header, false);
        header
    }
    fn entry_with(name: &str, data: &[u8], typeflag: u8, prefix: &str) -> Vec<u8> {
        let mut archive = header_with(name, data.len(), typeflag, prefix).to_vec();
        archive.extend_from_slice(data);
        let padding = (BLOCK - data.len() % BLOCK) % BLOCK;
        archive.resize(archive.len() + padding, 0);
        archive
    }
    fn entry(name: &str, data: &[u8], typeflag: u8) -> Vec<u8> {
        entry_with(name, data, typeflag, "")
    }
    fn unpack_tar_bytes(archive: &[u8]) -> Result<Vec<ArchiveEntry>> {
        ArchiveUnpacker::unpack(ArchiveFormat::Tar, archive)
    }
    #[test]
    fn detect_recognizes_known_signatures() {
        assert_eq!(detect(&[0x1f, 0x8b, 0x08, 0x00]), Some(ArchiveFormat::Gzip));
        assert_eq!(detect(b"PK\x03\x04rest"), Some(ArchiveFormat::Zip));
        assert_eq!(detect(b"PK\x05\x06"), Some(ArchiveFormat::Zip));
        assert_eq!(
            detect(&header_with("a.txt", 0, b'0', "")),
            Some(ArchiveFormat::Tar)
        );
        assert_eq!(detect(b"plain text, not an archive"), None);
        assert_eq!(ArchiveFormat::Gzip.to_string(), "gzip");
    }
    #[test]
    fn unpacks_regular_nested_and_directory_entries_in_order() {
        let prefix = "d".repeat(60);
        let name = format!("{}.txt", "f".repeat(36));
        let long_path = format!("{prefix}/{name}");
        let mut archive = Vec::new();
        archive.extend_from_slice(&entry("a.txt", b"hello", b'0'));
        archive.extend_from_slice(&entry_with("guide.md", b"hey", b'0', "docs"));
        archive.extend_from_slice(&entry("docs/", b"", b'5'));
        archive.extend_from_slice(&entry_with(&name, b"long", b'0', &prefix));
        let entries = unpack_tar_bytes(&archive).expect("tar parses");
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].path, "a.txt");
        assert_eq!(entries[0].bytes, b"hello");
        assert!(!entries[0].is_dir);
        assert_eq!(entries[1].path, "docs/guide.md");
        assert_eq!(entries[1].bytes, b"hey");
        assert!(!entries[1].is_dir);
        assert_eq!(entries[2].path, "docs/");
        assert_eq!(entries[2].bytes, Vec::<u8>::new());
        assert!(entries[2].is_dir);
        assert_eq!(entries[3].path, long_path);
        assert_eq!(entries[3].bytes, b"long");
        assert!(!entries[3].is_dir);
    }
    #[test]
    fn rejects_bad_checksum_and_accepts_gnu_signed_checksum() {
        let mut broken = entry("a.txt", b"data", b'0');
        broken[0] = b'z';
        let error = unpack_tar_bytes(&broken).expect_err("checksum must fail");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("byte 0"), "{error}");

        let mut signed_header = header_with("caf\u{e9}.txt", 0, b'0', "");
        write_checksum(&mut signed_header, true);
        let entries = unpack_tar_bytes(&signed_header).expect("signed checksum parses");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "caf\u{e9}.txt");
    }
    #[test]
    fn rejects_traversal_absolute_and_backslash_paths() {
        for path in ["../evil.txt", "/etc/passwd", "a\\b.txt"] {
            let error = unpack_tar_bytes(&entry(path, b"", b'0')).expect_err("path must fail");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains(&format!("{path:?}")), "{error}");
        }
    }
    #[test]
    fn rejects_truncated_entry_data() {
        let mut archive = header_with("big.bin", 10, b'0', "").to_vec();
        archive.extend_from_slice(b"abcd");
        let error = unpack_tar_bytes(&archive).expect_err("truncation must fail");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("byte 0"), "{error}");
        assert!(error.to_string().contains("truncated"), "{error}");
        assert!(error.to_string().contains("10"), "{error}");
    }
    #[test]
    fn two_zero_blocks_terminate_the_archive() {
        let mut archive = entry("first.txt", b"1", b'0');
        archive.extend_from_slice(&[0u8; 2 * BLOCK]);
        archive.extend_from_slice(&entry("ignored.txt", b"2", b'0'));
        let entries = unpack_tar_bytes(&archive).expect("tar parses");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "first.txt");
        assert_eq!(entries[0].bytes, b"1");
    }
    #[test]
    fn skips_pax_headers_and_reads_the_following_file() {
        let mut archive = entry("PaxHeaders/real.txt", b"12 path=ignored\n", b'x');
        archive.extend_from_slice(&entry("real.txt", b"ok", b'0'));
        let entries = unpack_tar_bytes(&archive).expect("tar parses");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "real.txt");
        assert_eq!(entries[0].bytes, b"ok");
    }
    #[test]
    fn skips_symlinks_and_keeps_later_entries() {
        let mut archive = entry("link", b"", b'2');
        archive.extend_from_slice(&entry("after.txt", b"after", b'0'));
        let entries = unpack_tar_bytes(&archive).expect("tar parses");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "after.txt");
        assert_eq!(entries[0].bytes, b"after");
    }
    #[test]
    fn zip_and_gzip_are_deferred_with_a_typed_error() {
        for (format, name) in [(ArchiveFormat::Zip, "zip"), (ArchiveFormat::Gzip, "gzip")] {
            let error =
                ArchiveUnpacker::unpack(format, b"irrelevant").expect_err("must be deferred");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            let message = error.to_string();
            assert!(message.starts_with("unsupported:"), "{message}");
            assert!(message.contains(name), "{message}");
            assert!(message.contains("compression dependency"), "{message}");
        }
    }
    #[test]
    fn unpack_auto_rejects_unknown_bytes() {
        let error = ArchiveUnpacker::unpack_auto(b"plain bytes").expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("archive"), "{error}");
    }
}
