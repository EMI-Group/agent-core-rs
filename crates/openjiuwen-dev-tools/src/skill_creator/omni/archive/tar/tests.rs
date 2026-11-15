//! Unit tests for the `tar` reader, built from synthetic raw-byte archives.

use super::*;
use crate::skill_creator::omni::archive::{
    ArchiveFormat, ArchiveUnpacker, MAX_ENTRY_UNCOMPRESSED_BYTES,
};
use openjiuwen_common::error::ErrorKind;

// --- archive fixtures ------------------------------------------------------

/// Writes ASCII `text` into a header field, truncating at `len`.
fn write_text(header: &mut [u8], offset: usize, len: usize, text: &str) {
    let bytes = text.as_bytes();
    let end = bytes.len().min(len);
    header[offset..offset + end].copy_from_slice(&bytes[..end]);
}

/// Writes `value` as a zero-padded octal field terminated by `NUL`.
fn write_octal(header: &mut [u8], offset: usize, len: usize, value: u64) {
    let text = format!("{:0width$o}\0", value, width = len - 1);
    header[offset..offset + len].copy_from_slice(text.as_bytes());
}

/// Writes the checksum field for the header, `GNU`-signed when `signed`.
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

/// A `ustar` header without a checksum yet.
fn raw_header(name: &str, size: usize, typeflag: u8) -> [u8; BLOCK] {
    let mut header = [0u8; BLOCK];
    write_text(&mut header, NAME_OFFSET, NAME_LEN, name);
    write_octal(&mut header, SIZE_OFFSET, SIZE_LEN, size as u64);
    write_octal(&mut header, MODE_OFFSET, MODE_LEN, 0o644);
    header[TYPEFLAG_OFFSET] = typeflag;
    header[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC_LEN].copy_from_slice(USTAR_MAGIC);
    header
}

/// A complete `ustar` header, optionally carrying a `prefix`.
fn header_with(name: &str, size: usize, typeflag: u8, prefix: &str) -> [u8; BLOCK] {
    let mut header = raw_header(name, size, typeflag);
    if !prefix.is_empty() {
        write_text(&mut header, PREFIX_OFFSET, PREFIX_LEN, prefix);
    }
    write_checksum(&mut header, false);
    header
}

/// A complete header whose linkname field holds `link`.
fn header_link(name: &str, typeflag: u8, link: &str) -> [u8; BLOCK] {
    let mut header = raw_header(name, 0, typeflag);
    write_text(&mut header, LINKNAME_OFFSET, LINKNAME_LEN, link);
    write_checksum(&mut header, false);
    header
}

/// Pads `data` with zero bytes up to the next 512-byte boundary.
fn push_padded(archive: &mut Vec<u8>, data: &[u8]) {
    archive.extend_from_slice(data);
    let padding = (BLOCK - data.len() % BLOCK) % BLOCK;
    archive.resize(archive.len() + padding, 0);
}

/// One header declaring `declared_size` followed by `data` (padded to a block).
///
/// `declared_size` may differ from `data.len()` to model a PAX `size` override.
fn raw_entry(name: &str, declared_size: usize, data: &[u8], typeflag: u8) -> Vec<u8> {
    let mut archive = header_with(name, declared_size, typeflag, "").to_vec();
    push_padded(&mut archive, data);
    archive
}

/// One header plus its data, the header declaring exactly `data.len()` bytes.
fn entry_with(name: &str, data: &[u8], typeflag: u8, prefix: &str) -> Vec<u8> {
    let mut archive = header_with(name, data.len(), typeflag, prefix).to_vec();
    push_padded(&mut archive, data);
    archive
}

/// One header plus its data, without a `ustar` prefix.
fn entry(name: &str, data: &[u8], typeflag: u8) -> Vec<u8> {
    entry_with(name, data, typeflag, "")
}

/// One regular file entry whose mode field holds `mode` (or all `NUL`s).
fn file_with_mode(name: &str, mode: Option<u32>, data: &[u8]) -> Vec<u8> {
    let mut header = raw_header(name, data.len(), b'0');
    match mode {
        Some(mode) => write_octal(&mut header, MODE_OFFSET, MODE_LEN, u64::from(mode)),
        None => header[MODE_OFFSET..MODE_OFFSET + MODE_LEN].fill(0),
    }
    write_checksum(&mut header, false);
    let mut archive = header.to_vec();
    push_padded(&mut archive, data);
    archive
}

/// One PAX record: `<len> <key>=<value>\n`, with the self-referential length.
fn pax_record(key: &str, value: &str) -> String {
    let body = format!(" {key}={value}\n");
    let mut len = body.len() + 1;
    loop {
        let next = len.to_string().len() + body.len();
        if next == len {
            return format!("{len}{body}");
        }
        len = next;
    }
}

/// A PAX extended-header payload holding `records` in order.
fn pax_payload(records: &[(&str, &str)]) -> Vec<u8> {
    records
        .iter()
        .map(|(key, value)| pax_record(key, value))
        .collect::<String>()
        .into_bytes()
}

/// Unpacks `archive` as `tar`.
fn unpack_tar_bytes(archive: &[u8]) -> Result<Vec<ArchiveEntry>> {
    ArchiveUnpacker::unpack(ArchiveFormat::Tar, archive)
}

/// The absolute offset of the first extended-header payload.
const FIRST_PAYLOAD: usize = BLOCK;
/// The absolute offset of the header that follows a padded extended header.
const SECOND_HEADER: usize = 2 * BLOCK;

// --- headers and data ------------------------------------------------------

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
    assert_eq!(entries[0].kind, ArchiveEntryKind::File);
    assert_eq!(entries[1].path, "docs/guide.md");
    assert_eq!(entries[1].bytes, b"hey");
    assert!(!entries[1].is_dir);
    assert_eq!(entries[2].path, "docs/");
    assert_eq!(entries[2].bytes, Vec::<u8>::new());
    assert!(entries[2].is_dir);
    assert_eq!(entries[2].kind, ArchiveEntryKind::Directory);
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
fn accepts_a_base_256_size_field() {
    let mut header = raw_header("big.bin", 0, b'0');
    // GNU base-256: the high bit of the first byte marks the encoding.
    let size = 300u64;
    header[SIZE_OFFSET..SIZE_OFFSET + SIZE_LEN].fill(0);
    header[SIZE_OFFSET] = 0x80;
    header[SIZE_OFFSET + SIZE_LEN - 2] = 0x01;
    header[SIZE_OFFSET + SIZE_LEN - 1] = (size - 256) as u8;
    write_checksum(&mut header, false);
    let mut archive = header.to_vec();
    archive.extend(std::iter::repeat_n(b'x', 300));
    let entries = unpack_tar_bytes(&archive).expect("base-256 size parses");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].bytes.len(), 300);
}

#[test]
fn rejects_a_declared_size_above_the_per_entry_cap() {
    let mut archive = header_with("huge.bin", MAX_ENTRY_UNCOMPRESSED_BYTES + 1, b'0', "").to_vec();
    // The cap is enforced from the declared size, before the data is read.
    archive.extend_from_slice(&[0u8; 8]);
    let error = unpack_tar_bytes(&archive).expect_err("declared size must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("per-entry cap"), "{error}");
}

// --- modes -----------------------------------------------------------------

#[test]
fn reports_the_declared_mode_and_none_for_a_missing_one() {
    let mut archive = file_with_mode("regular.txt", Some(0o100_644), b"data");
    archive.extend_from_slice(&file_with_mode("mode-less.txt", None, b"data2"));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].mode, Some(0o100_644));
    assert_eq!(entries[1].mode, None);
}

// --- end of archive --------------------------------------------------------

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
fn a_header_after_the_end_of_archive_marker_is_never_parsed() {
    let mut archive = entry("first.txt", b"1", b'0');
    archive.extend_from_slice(&[0u8; 2 * BLOCK]);
    // A block whose checksum is unusable: parsing it would fail the archive.
    archive.extend_from_slice(&[0xffu8; BLOCK]);
    let entries = unpack_tar_bytes(&archive).expect("trailing garbage is ignored");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "first.txt");
}

#[test]
fn a_single_zero_block_before_a_header_continues() {
    let mut archive = entry("first.txt", b"1", b'0');
    archive.extend_from_slice(&[0u8; BLOCK]);
    archive.extend_from_slice(&entry("second.txt", b"2", b'0'));
    let entries = unpack_tar_bytes(&archive).expect("a lone zero block is tolerated");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].path, "first.txt");
    assert_eq!(entries[1].path, "second.txt");
    assert_eq!(entries[1].bytes, b"2");
}

#[test]
fn a_trailing_partial_block_is_ignored() {
    let mut archive = entry("first.txt", b"1", b'0');
    // 200 bytes cannot hold a header, and they are not a valid one either.
    archive.extend_from_slice(&[0xab; 200]);
    let entries = unpack_tar_bytes(&archive).expect("the partial block is ignored");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "first.txt");
}

// --- GNU long names and linknames ------------------------------------------

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
fn applies_gnu_long_name_and_linkname_to_a_symlink() {
    let long_path = format!("dir/{}.txt", "n".repeat(120));
    let long_target = format!("../{}", "t".repeat(120));
    // Neither payload carries a `NUL` terminator: both are used in full.
    let mut archive = entry("ShortName", long_path.as_bytes(), b'L');
    archive.extend_from_slice(&entry("ShortLink", long_target.as_bytes(), b'K'));
    archive.extend_from_slice(&header_link("short", b'2', "ignored"));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, long_path);
    assert_eq!(entries[0].kind, ArchiveEntryKind::Symlink);
    assert_eq!(entries[0].link_target, Some(long_target));
    assert_eq!(entries[0].bytes, Vec::<u8>::new());
    assert!(!entries[0].is_dir);
    assert_eq!(entries[0].mode, Some(0o644));
}

#[test]
fn applies_a_nul_terminated_gnu_long_name() {
    let mut archive = entry("ShortName", b"docs/nul.txt\0", b'L');
    archive.extend_from_slice(&entry("ShortName", b"body", b'0'));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "docs/nul.txt");
    assert_eq!(entries[0].bytes, b"body");
    assert_eq!(entries[0].link_target, None);
}

#[test]
fn a_gnu_long_name_applies_to_the_next_header_only() {
    let long = format!("{}.txt", "n".repeat(120));
    let mut archive = entry("ShortName", long.as_bytes(), b'L');
    archive.extend_from_slice(&entry("first.txt", b"1", b'0'));
    archive.extend_from_slice(&entry("second.txt", b"2", b'0'));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].path, long);
    assert_eq!(entries[1].path, "second.txt");
}

// --- links -----------------------------------------------------------------

#[test]
fn exposes_symlink_and_hardlink_entries_in_archive_order() {
    let mut archive = entry("a.txt", b"a", b'0');
    archive.extend_from_slice(&header_link("link", b'2', "a.txt"));
    archive.extend_from_slice(&header_link("hard", b'1', "a.txt"));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].kind, ArchiveEntryKind::File);
    assert_eq!(entries[0].link_target, None);
    assert_eq!(entries[1].kind, ArchiveEntryKind::Symlink);
    assert_eq!(entries[1].path, "link");
    assert_eq!(entries[1].link_target.as_deref(), Some("a.txt"));
    assert_eq!(entries[1].mode, Some(0o644));
    assert_eq!(entries[2].kind, ArchiveEntryKind::HardLink);
    assert_eq!(entries[2].path, "hard");
    assert_eq!(entries[2].link_target.as_deref(), Some("a.txt"));
    for entry in &entries[1..] {
        assert!(entry.bytes.is_empty(), "{entry:?}");
        assert!(!entry.is_dir, "{entry:?}");
    }
}

#[test]
fn keeps_an_absolute_or_parent_relative_link_target_verbatim() {
    let mut archive = header_link("escape", b'2', "../../etc/passwd").to_vec();
    archive.extend_from_slice(&header_link("absolute", b'1', "/etc/passwd"));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 2);
    // The entry path is validated; the target is only recorded.
    assert_eq!(entries[0].path, "escape");
    assert_eq!(entries[0].link_target, Some("../../etc/passwd".to_owned()));
    assert_eq!(entries[1].link_target, Some("/etc/passwd".to_owned()));
}

// --- PAX extended headers --------------------------------------------------

#[test]
fn applies_pax_path_and_size_records_to_the_next_entry_only() {
    let pax = pax_payload(&[("path", "docs/real.txt"), ("size", "8")]);
    let mut archive = entry("PaxHeaders/real.txt", &pax, b'x');
    // The header's octal field declares 3 bytes; the PAX record resizes it to 8.
    archive.extend_from_slice(&raw_entry("real.txt", 3, b"12345678", b'0'));
    archive.extend_from_slice(&entry("second.txt", b"ok", b'0'));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].path, "docs/real.txt");
    assert_eq!(entries[0].bytes, b"12345678");
    assert_eq!(entries[1].path, "second.txt");
    assert_eq!(entries[1].bytes, b"ok");
}

#[test]
fn applies_pax_global_records_to_every_following_entry() {
    let global = pax_payload(&[("path", "global/one.txt"), ("uname", "docs")]);
    let mut archive = entry("PaxHeaders/global", &global, b'g');
    archive.extend_from_slice(&entry("one.txt", b"1", b'0'));
    let local = pax_payload(&[("path", "local/two.txt")]);
    archive.extend_from_slice(&entry("PaxHeaders/two", &local, b'x'));
    archive.extend_from_slice(&entry("two.txt", b"2", b'0'));
    archive.extend_from_slice(&entry("three.txt", b"3", b'0'));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].path, "global/one.txt");
    assert_eq!(entries[0].bytes, b"1");
    // The per-file record wins for its own entry only.
    assert_eq!(entries[1].path, "local/two.txt");
    assert_eq!(entries[1].bytes, b"2");
    assert_eq!(entries[2].path, "global/one.txt");
    assert_eq!(entries[2].bytes, b"3");
}

#[test]
fn applies_pax_linkpath_to_a_symlink() {
    let pax = pax_payload(&[("linkpath", "pax/target.txt")]);
    let mut archive = entry("PaxHeaders/link", &pax, b'x');
    archive.extend_from_slice(&header_link("link", b'2', "header-target.txt"));
    archive.extend_from_slice(&header_link("next", b'2', "header-target.txt"));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].path, "link");
    assert_eq!(
        entries[0].link_target.as_deref(),
        Some("pax/target.txt"),
        "the PAX linkpath replaces the header linkname"
    );
    assert_eq!(
        entries[1].link_target.as_deref(),
        Some("header-target.txt"),
        "the per-file record does not leak into the next entry"
    );
}

#[test]
fn a_pax_linkpath_wins_over_a_gnu_long_linkname() {
    let mut archive = entry("ShortLink", b"gnu/target.txt", b'K');
    let pax = pax_payload(&[("linkpath", "pax/target.txt")]);
    archive.extend_from_slice(&entry("PaxHeaders/link", &pax, b'x'));
    archive.extend_from_slice(&header_link("link", b'2', "header-target.txt"));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].link_target.as_deref(), Some("pax/target.txt"));
}

#[test]
fn a_pax_path_wins_over_a_gnu_long_name() {
    let mut archive = entry("ShortName", b"gnu/name.txt", b'L');
    let pax = pax_payload(&[("path", "pax/name.txt")]);
    archive.extend_from_slice(&entry("PaxHeaders/name", &pax, b'x'));
    archive.extend_from_slice(&entry("ShortName", b"body", b'0'));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "pax/name.txt");
    assert_eq!(entries[0].bytes, b"body");
}

#[test]
fn ignores_pax_records_it_does_not_interpret() {
    let pax = pax_payload(&[
        ("mtime", "1700000000.123456"),
        ("uid", "1000"),
        ("gid", "1000"),
        ("uname", "docs"),
        ("gname", "staff"),
        ("comment", "an ignored note"),
    ]);
    let mut archive = entry("PaxHeaders/plain.txt", &pax, b'x');
    archive.extend_from_slice(&entry("plain.txt", b"text", b'0'));
    let entries = unpack_tar_bytes(&archive).expect("tar parses");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "plain.txt");
    assert_eq!(entries[0].bytes, b"text");
}

#[test]
fn rejects_a_pax_record_with_a_non_decimal_length() {
    let mut archive = entry("PaxHeaders/bad", b"?? path=a.txt\n", b'x');
    archive.extend_from_slice(&entry("a.txt", b"a", b'0'));
    let error = unpack_tar_bytes(&archive).expect_err("bad length must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains(&format!("byte {FIRST_PAYLOAD}")),
        "{error}"
    );
}

#[test]
fn rejects_a_pax_record_whose_length_does_not_end_in_a_newline() {
    let mut archive = entry("PaxHeaders/bad", b"5 path=a.txt\n", b'x');
    archive.extend_from_slice(&entry("a.txt", b"a", b'0'));
    let error = unpack_tar_bytes(&archive).expect_err("bad length must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains(&format!("byte {FIRST_PAYLOAD}")),
        "{error}"
    );
    assert!(error.to_string().contains("newline"), "{error}");
}

#[test]
fn rejects_a_truncated_pax_record() {
    let mut archive = entry("PaxHeaders/bad", b"40 path=a.txt\n", b'x');
    archive.extend_from_slice(&entry("a.txt", b"a", b'0'));
    let error = unpack_tar_bytes(&archive).expect_err("truncated record must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("truncated"), "{error}");
    assert!(
        error.to_string().contains(&format!("byte {FIRST_PAYLOAD}")),
        "{error}"
    );
}

#[test]
fn rejects_a_pax_record_without_a_separator() {
    let mut archive = entry("PaxHeaders/bad", b"8 patha\n", b'x');
    archive.extend_from_slice(&entry("a.txt", b"a", b'0'));
    let error = unpack_tar_bytes(&archive).expect_err("a missing `=` must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains(&format!("byte {FIRST_PAYLOAD}")),
        "{error}"
    );
    assert!(error.to_string().contains("separator"), "{error}");
}

#[test]
fn names_the_offset_of_a_later_malformed_pax_record() {
    let first = pax_record("path", "a.txt");
    let payload = format!("{first}zz key=value\n");
    let mut archive = entry("PaxHeaders/bad", payload.as_bytes(), b'x');
    archive.extend_from_slice(&entry("a.txt", b"a", b'0'));
    let error = unpack_tar_bytes(&archive).expect_err("bad length must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let offset = FIRST_PAYLOAD + first.len();
    assert!(
        error.to_string().contains(&format!("byte {offset}")),
        "{error}"
    );
}

#[test]
fn rejects_a_non_decimal_pax_size_record() {
    let pax = pax_payload(&[("size", "many")]);
    let mut archive = entry("PaxHeaders/size", &pax, b'x');
    archive.extend_from_slice(&entry("a.txt", b"", b'0'));
    let error = unpack_tar_bytes(&archive).expect_err("a bad size must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("size"), "{error}");
    assert!(
        error.to_string().contains(&format!("byte {SECOND_HEADER}")),
        "{error}"
    );
}

#[test]
fn rejects_a_pax_size_above_the_per_entry_cap() {
    let oversized = (MAX_ENTRY_UNCOMPRESSED_BYTES + 1).to_string();
    let pax = pax_payload(&[("size", &oversized)]);
    let mut archive = entry("PaxHeaders/size", &pax, b'x');
    archive.extend_from_slice(&entry("a.txt", b"", b'0'));
    let error = unpack_tar_bytes(&archive).expect_err("the cap must apply to a PAX size");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("per-entry cap"), "{error}");
}
