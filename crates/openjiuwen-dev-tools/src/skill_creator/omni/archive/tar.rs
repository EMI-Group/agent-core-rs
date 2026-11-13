//! POSIX `ustar` / GNU `tar` parsing.
//!
//! A `tar` archive is a sequence of 512-byte headers, each followed by the
//! entry's data padded to a 512-byte boundary. Two consecutive zero blocks
//! terminate the archive. The parser accepts:
//!
//! - octal size fields and the GNU base-256 encoding;
//! - unsigned checksums and the GNU signed variant;
//! - the POSIX `ustar` `prefix` field, and GNU long names (`L`) and long
//!   linknames (`K`), both consumed by their declared size;
//! - PAX per-file (`x`) and global (`g`) headers, skipped by size;
//! - regular files (`0`/`\0`) and directories (`5`).
//!
//! Hard links, symlinks, devices, and FIFOs carry no file contents and are
//! ignored. Every retained path goes through
//! [`validate_path`](super::validate_path), and every declared size is charged
//! to [`CapTracker`](super::CapTracker) before its data slice is copied.
use super::{validate_path, ArchiveEntry, CapTracker};
use openjiuwen_common::error::{Error, Result};

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

/// Parses a `tar` archive, appending entries in archive order.
///
/// # Errors
///
/// Returns [`Error::validation`] naming the offending byte offset for a bad
/// checksum, an unparsable size, truncated entry data, an unsafe path, or an
/// entry above the size caps.
pub(crate) fn unpack(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
    let mut entries = Vec::new();
    let mut caps = CapTracker::new();
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
        caps.charge(size, &format!("tar entry at byte {offset}"))?;
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
    use crate::skill_creator::omni::archive::{
        ArchiveFormat, ArchiveUnpacker, MAX_ENTRY_UNCOMPRESSED_BYTES,
    };
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
    fn applies_a_long_name_header_to_the_next_file() {
        let long = format!("dir/{}.txt", "n".repeat(120));
        let mut archive = entry("shortname", long.as_bytes(), b'L');
        archive.extend_from_slice(&entry("shortname", b"content", b'0'));
        let entries = unpack_tar_bytes(&archive).expect("tar parses");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, long);
        assert_eq!(entries[0].bytes, b"content");
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
    fn rejects_a_declared_size_above_the_per_entry_cap() {
        let mut archive =
            header_with("huge.bin", MAX_ENTRY_UNCOMPRESSED_BYTES + 1, b'0', "").to_vec();
        // The cap is enforced from the declared size, before the data is read.
        archive.extend_from_slice(&[0u8; 8]);
        let error = unpack_tar_bytes(&archive).expect_err("declared size must fail");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("per-entry cap"), "{error}");
    }

    #[test]
    fn accepts_a_base_256_size_field() {
        let mut header = [0u8; BLOCK];
        write_text(&mut header, NAME_OFFSET, NAME_LEN, "big.bin");
        // GNU base-256: the high bit of the first byte marks the encoding.
        let size = 300u64;
        header[SIZE_OFFSET] = 0x80;
        header[SIZE_OFFSET + SIZE_LEN - 2] = 0x01;
        header[SIZE_OFFSET + SIZE_LEN - 1] = (size - 256) as u8;
        write_octal(&mut header, 100, 8, 0o644);
        header[TYPEFLAG_OFFSET] = b'0';
        header[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC_LEN].copy_from_slice(USTAR_MAGIC);
        write_checksum(&mut header, false);
        let mut archive = header.to_vec();
        archive.extend(std::iter::repeat_n(b'x', 300));
        let entries = unpack_tar_bytes(&archive).expect("base-256 size parses");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].bytes.len(), 300);
    }
}
