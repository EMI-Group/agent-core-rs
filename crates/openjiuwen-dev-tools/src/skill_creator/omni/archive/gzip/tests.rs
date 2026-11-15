//! Unit tests for RFC 1952 gzip member parsing.
//!
//! Every fixture is built in-test from stored (uncompressed) DEFLATE blocks, so
//! the bytes are exact and no compressor or network is involved. The aggregate
//! cap test builds a fixed-Huffman stream whose 258-byte back-references expand a
//! ~210 KiB fixture past the 32 MiB payload cap instead of shipping a 32 MiB
//! constant.

use super::*;
use crate::skill_creator::omni::archive::ArchiveEntryKind;
use openjiuwen_common::error::ErrorKind;

/// The size of one `tar` header or data block.
const TAR_BLOCK: usize = 512;

// --- tar fixtures ----------------------------------------------------------

/// Writes ASCII `text` into a `tar` header field, truncating at `len`.
fn write_text(header: &mut [u8], offset: usize, len: usize, text: &str) {
    let bytes = text.as_bytes();
    let end = bytes.len().min(len);
    header[offset..offset + end].copy_from_slice(&bytes[..end]);
}

/// Writes `value` as a `NUL`-terminated octal field of `len` bytes.
fn write_octal(header: &mut [u8], offset: usize, len: usize, value: u64) {
    let text = format!("{:0width$o}\0", value, width = len - 1);
    header[offset..offset + len].copy_from_slice(text.as_bytes());
}

/// Fills the `tar` checksum field with the unsigned header sum.
fn write_checksum(header: &mut [u8; TAR_BLOCK]) {
    for byte in &mut header[148..156] {
        *byte = b' ';
    }
    let sum: u64 = header.iter().map(|&byte| u64::from(byte)).sum();
    write_text(header, 148, 8, &format!("{sum:06o}\0 "));
}

/// Builds one POSIX `ustar` header for `path` with `size` bytes of content.
fn tar_header(path: &str, size: usize, typeflag: u8) -> [u8; TAR_BLOCK] {
    let mut header = [0u8; TAR_BLOCK];
    write_text(&mut header, 0, 100, path);
    write_octal(&mut header, 100, 8, 0o644);
    write_octal(
        &mut header,
        124,
        12,
        u64::try_from(size).expect("a tar size fits u64"),
    );
    header[156] = typeflag;
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    write_checksum(&mut header);
    header
}

/// Builds a `ustar` archive from `(path, content, typeflag)` entries.
fn tar_archive(entries: &[(&str, &[u8], u8)]) -> Vec<u8> {
    let mut archive = Vec::new();
    for (path, content, typeflag) in entries {
        archive.extend_from_slice(&tar_header(path, content.len(), *typeflag));
        archive.extend_from_slice(content);
        let padding = (TAR_BLOCK - content.len() % TAR_BLOCK) % TAR_BLOCK;
        archive.extend(std::iter::repeat_n(0u8, padding));
    }
    archive.extend_from_slice(&[0u8; 2 * TAR_BLOCK]);
    archive
}

/// A small archive with one directory and one file entry.
fn sample_tar() -> Vec<u8> {
    tar_archive(&[
        ("docs/", b"", b'5'),
        ("docs/readme.txt", b"hello gzip\n", b'0'),
    ])
}

// --- gzip fixtures ---------------------------------------------------------

/// Wraps `payload` in one final stored DEFLATE block.
fn stored_deflate(payload: &[u8]) -> Vec<u8> {
    let len = u16::try_from(payload.len()).expect("a stored block holds at most 65535 bytes");
    let mut out = Vec::with_capacity(payload.len() + 5);
    // BFINAL = 1, BTYPE = 00, then five zero padding bits.
    out.push(0x01);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// The optional member-header fields a test wants, plus trailer overrides.
struct Member {
    /// The `CM` byte; defaults to DEFLATE.
    method: u8,
    /// Extra `FLG` bits (`FTEXT`, reserved bits) OR-ed into the computed flags.
    extra_flags: u8,
    /// The `FEXTRA` payload; emitted only when non-empty.
    extra: Vec<u8>,
    /// The `FNAME` value.
    name: Option<&'static str>,
    /// The `FCOMMENT` value.
    comment: Option<&'static str>,
    /// Whether an `FHCRC` field is emitted.
    header_crc: bool,
    /// An explicit `FHCRC` value; defaults to the correct one.
    header_crc_value: Option<u16>,
    /// An explicit trailer `CRC32`; defaults to the payload's CRC.
    crc_value: Option<u32>,
    /// An explicit trailer `ISIZE`; defaults to the payload length.
    isize_value: Option<u32>,
}

impl Default for Member {
    fn default() -> Self {
        Self {
            method: CM_DEFLATE,
            extra_flags: 0,
            extra: Vec::new(),
            name: None,
            comment: None,
            header_crc: false,
            header_crc_value: None,
            crc_value: None,
            isize_value: None,
        }
    }
}

/// Wraps `payload` in a stored DEFLATE block and a correct trailer.
fn member_with(payload: &[u8], options: &Member) -> Vec<u8> {
    let crc = crc32(payload);
    let isize = u32::try_from(payload.len()).expect("a fixture fits 32 bits");
    member_from_deflate(&stored_deflate(payload), crc, isize, options)
}

/// Builds a member around a pre-built DEFLATE payload and trailer values.
fn member_from_deflate(deflate: &[u8], crc: u32, isize: u32, options: &Member) -> Vec<u8> {
    let mut flags = options.extra_flags;
    if !options.extra.is_empty() {
        flags |= FLG_FEXTRA;
    }
    if options.name.is_some() {
        flags |= FLG_FNAME;
    }
    if options.comment.is_some() {
        flags |= FLG_FCOMMENT;
    }
    if options.header_crc {
        flags |= FLG_FHCRC;
    }
    // Magic, CM, FLG, then MTIME/XFL/OS.
    let mut out = vec![0x1f, 0x8b, options.method, flags, 0, 0, 0, 0, 0x00, 0xff];
    if !options.extra.is_empty() {
        let len = u16::try_from(options.extra.len()).expect("extra fits 16 bits");
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&options.extra);
    }
    if let Some(name) = options.name {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
    }
    if let Some(comment) = options.comment {
        out.extend_from_slice(comment.as_bytes());
        out.push(0);
    }
    if options.header_crc {
        let computed =
            u16::try_from(crc32(&out) & 0xFFFF).expect("the masked CRC is exactly 16 bits");
        let value = options.header_crc_value.unwrap_or(computed);
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(deflate);
    out.extend_from_slice(&options.crc_value.unwrap_or(crc).to_le_bytes());
    out.extend_from_slice(&options.isize_value.unwrap_or(isize).to_le_bytes());
    out
}

/// A DEFLATE bit writer: bits land in the output byte least-significant bit
/// first, and each call emits its value most-significant bit first, which is how
/// RFC 1951 packs Huffman codes. Raw bit fields such as `BTYPE` are therefore
/// passed pre-reversed.
struct BitWriter {
    /// The bytes written so far.
    bytes: Vec<u8>,
    /// The number of bits written so far.
    bit: usize,
}

impl BitWriter {
    /// An empty writer.
    const fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit: 0,
        }
    }

    /// Writes the low `count` bits of `value`, most-significant bit first.
    fn write(&mut self, value: u32, count: u32) {
        for shift in (0..count).rev() {
            if self.bit % 8 == 0 {
                self.bytes.push(0);
            }
            if (value >> shift) & 1 == 1 {
                let index = self.bit / 8;
                self.bytes[index] |= 1 << (self.bit % 8);
            }
            self.bit += 1;
        }
    }
}

/// A fixed-Huffman block that expands to at least `total` bytes.
///
/// One literal is followed by length-258/distance-1 matches, so every 13 input
/// bits yield 258 output bytes (`symbol 285` is the 8-bit fixed code `0xc5`,
/// distance symbol 0 takes 5 bits).
fn high_expansion_deflate(total: usize) -> Vec<u8> {
    let mut writer = BitWriter::new();
    // BFINAL = 1, then BTYPE = 01 (fixed Huffman) packed least-significant bit
    // first, which is `0b10` for this writer.
    writer.write(1, 1);
    writer.write(0b10, 2);
    writer.write(0x71, 8); // literal 'A' (fixed code 0x30 + 0x41)
    for _ in 0..total.div_ceil(258) {
        writer.write(0xC5, 8); // length symbol 285: 258 bytes
        writer.write(0, 5); // distance symbol 0: one byte back
    }
    writer.write(0, 7); // end of block (symbol 256)
    writer.bytes
}

/// The validation error produced by a fixture, after asserting its kind.
fn validation_error(bytes: &[u8], case: &str) -> (ErrorKind, String) {
    let error = unpack(bytes).expect_err(case);
    (error.kind(), error.to_string())
}

// --- tests -----------------------------------------------------------------

#[test]
fn a_gzip_wrapped_tar_unpacks_through_detection() {
    let tar = sample_tar();
    let bytes = member_with(&tar, &Member::default());
    assert_eq!(detect(&bytes), Some(ArchiveFormat::Gzip));

    let entries = unpack(&bytes).expect("a gzip-wrapped tar unpacks");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].path, "docs/");
    assert!(entries[0].is_dir, "the directory entry is marked");
    assert_eq!(
        entries[0].bytes,
        b"".as_slice(),
        "directory entries are empty"
    );
    assert_eq!(entries[1].path, "docs/readme.txt");
    assert_eq!(entries[1].bytes, b"hello gzip\n".as_slice());
    assert!(!entries[1].is_dir);

    let auto = ArchiveUnpacker::unpack_auto(&bytes).expect("detection routes to gzip");
    assert_eq!(auto, entries);
}

#[test]
fn a_gzip_wrapped_zip_unpacks_through_detection() {
    use crate::skill_creator::omni::archive::ArchiveWriter;

    let entries = vec![
        ArchiveEntry {
            path: "notes/readme.txt".to_owned(),
            bytes: b"hello zip\n".to_vec(),
            is_dir: false,
            ..ArchiveEntry::default()
        },
        ArchiveEntry {
            path: "assets/".to_owned(),
            bytes: Vec::new(),
            is_dir: true,
            kind: ArchiveEntryKind::Directory,
            ..ArchiveEntry::default()
        },
    ];
    let zip = ArchiveWriter::write_stored(&entries).expect("writes a stored zip");
    let bytes = member_with(&zip, &Member::default());
    assert_eq!(detect(&bytes), Some(ArchiveFormat::Gzip));

    let unpacked = ArchiveUnpacker::unpack_auto(&bytes).expect("a gzip-wrapped zip unpacks");
    assert_eq!(unpacked, entries);
}

#[test]
fn optional_header_fields_are_skipped() {
    let tar = sample_tar();
    let all = Member {
        extra: b"AP\x02\x00hi".to_vec(),
        name: Some("page.tar"),
        comment: Some("scraped by the omni pipeline"),
        extra_flags: 0x01, // FTEXT: carries no meaning
        ..Member::default()
    };
    let entries = unpack(&member_with(&tar, &all)).expect("FEXTRA, FNAME, FCOMMENT decode");
    assert_eq!(entries.len(), 2);

    let name_only = Member {
        name: Some("readme.tar"),
        ..Member::default()
    };
    let entries = unpack(&member_with(&tar, &name_only)).expect("FNAME alone decodes");
    assert_eq!(entries[1].path, "docs/readme.txt");

    let comment_only = Member {
        comment: Some("no name"),
        ..Member::default()
    };
    let entries = unpack(&member_with(&tar, &comment_only)).expect("FCOMMENT alone decodes");
    assert_eq!(entries.len(), 2);
}

#[test]
fn the_header_crc_is_verified() {
    let tar = sample_tar();
    let good = Member {
        header_crc: true,
        name: Some("page.tar"),
        ..Member::default()
    };
    let entries = unpack(&member_with(&tar, &good)).expect("a correct FHCRC decodes");
    assert_eq!(entries.len(), 2);

    let bad = Member {
        header_crc: true,
        header_crc_value: Some(0x1234),
        ..Member::default()
    };
    let (kind, message) = validation_error(&member_with(&tar, &bad), "a wrong FHCRC is rejected");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("header CRC mismatch"), "{message}");
    assert!(message.contains("0x1234"), "{message}");
    assert!(message.contains("member 0 at offset 0"), "{message}");
}

#[test]
fn a_crc32_mismatch_is_reported() {
    let tar = sample_tar();
    let mut bytes = member_with(&tar, &Member::default());
    // Flip a byte of the trailer CRC, which sits eight bytes before the end.
    let crc_offset = bytes.len() - TRAILER_LEN;
    bytes[crc_offset] ^= 0xFF;
    let computed = crc32(&tar);
    let stored = computed ^ 0xFF;
    let (kind, message) = validation_error(&bytes, "a wrong CRC-32 is rejected");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("CRC-32 mismatch"), "{message}");
    assert!(message.contains("member 0 at offset 0"), "{message}");
    assert!(message.contains(&format!("{stored:#010x}")), "{message}");
    assert!(message.contains(&format!("{computed:#010x}")), "{message}");
}

#[test]
fn an_isize_mismatch_is_reported() {
    let tar = sample_tar();
    let size = u32::try_from(tar.len()).expect("a fixture fits 32 bits");
    let options = Member {
        isize_value: Some(size + 1),
        ..Member::default()
    };
    let (kind, message) = validation_error(&member_with(&tar, &options), "a wrong ISIZE fails");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("ISIZE mismatch"), "{message}");
    assert!(
        message.contains(&format!("stores {}", size + 1)),
        "{message}"
    );
    assert!(message.contains(&format!("is {size} bytes")), "{message}");
}

#[test]
fn concatenated_members_form_one_payload() {
    let tar = sample_tar();
    let split = 700;
    let mut bytes = member_with(&tar[..split], &Member::default());
    bytes.extend_from_slice(&member_with(&tar[split..], &Member::default()));

    let entries = unpack(&bytes).expect("a tar split across two members parses");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].bytes, b"hello gzip\n".as_slice());
    assert_eq!(
        ArchiveUnpacker::unpack_auto(&bytes).expect("detection finds gzip"),
        entries
    );
}

#[test]
fn concatenated_members_that_do_not_form_an_archive_are_rejected() {
    let mut bytes = member_with(b"hello ", &Member::default());
    bytes.extend_from_slice(&member_with(b"world", &Member::default()));
    let (kind, message) = validation_error(&bytes, "two text members form no archive");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("not an archive"), "{message}");
    assert!(message.contains("11 bytes"), "{message}");
}

#[test]
fn a_non_archive_payload_is_rejected() {
    let (kind, message) = validation_error(
        &member_with(b"hello world", &Member::default()),
        "text is not an archive",
    );
    assert_eq!(kind, ErrorKind::Validation);
    assert!(
        message.contains("gzip payload is not an archive"),
        "{message}"
    );
    assert!(message.contains("11 bytes"), "{message}");
}

#[test]
fn a_nested_gzip_payload_is_rejected_as_such() {
    let inner = member_with(&sample_tar(), &Member::default());
    let (kind, message) = validation_error(&member_with(&inner, &Member::default()), "nested gzip");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("nested gzip"), "{message}");
    assert!(
        message.contains(&format!("{} bytes", inner.len())),
        "{message}"
    );
}

#[test]
fn truncated_headers_are_rejected() {
    let cases: &[(&[u8], &str)] = &[
        (&[0x1f], "half a magic"),
        (&[0x1f, 0x8b], "magic only"),
        (&[0x1f, 0x8b, CM_DEFLATE], "magic and CM"),
        (
            &[0x1f, 0x8b, CM_DEFLATE, FLG_FEXTRA, 0, 0, 0, 0, 0xff],
            "FEXTRA without XLEN",
        ),
    ];
    for (bytes, case) in cases {
        let (kind, message) = validation_error(bytes, case);
        assert_eq!(kind, ErrorKind::Validation, "{case}");
        assert!(
            message.contains("truncated gzip member at offset 0"),
            "{case}: {message}"
        );
    }

    let mut named = vec![0x1f, 0x8b, CM_DEFLATE, FLG_FNAME, 0, 0, 0, 0, 0xff];
    named.extend_from_slice(b"page.tar");
    let (kind, message) = validation_error(&named, "unterminated FNAME");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("no NUL terminator"), "{message}");
    assert!(message.contains("file name"), "{message}");
}

#[test]
fn a_non_deflate_compression_method_is_unsupported() {
    let error = unpack(&[0x1f, 0x8b, 0x07, 0x00]).expect_err("method 7 is not DEFLATE");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("compression method 7"), "{message}");
    assert!(message.contains("only 8"), "{message}");
}

#[test]
fn reserved_flag_bits_are_rejected() {
    let (kind, message) = validation_error(
        &[0x1f, 0x8b, CM_DEFLATE, FLG_RESERVED],
        "reserved flag bits are rejected",
    );
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("reserved FLG bits"), "{message}");
    assert!(message.contains("0xe0"), "{message}");
}

#[test]
fn an_empty_stream_is_rejected() {
    let (kind, message) = validation_error(&[], "empty input");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("empty gzip stream"), "{message}");

    let (kind, message) = validation_error(b"definitely not gzip", "wrong magic");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("not a gzip stream"), "{message}");
}

#[test]
fn trailing_garbage_after_a_member_is_rejected() {
    let tar = sample_tar();
    let mut bytes = member_with(&tar, &Member::default());
    let member_len = bytes.len();
    bytes.extend_from_slice(b"junk");
    let (kind, message) = validation_error(&bytes, "trailing garbage");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(
        message.contains("trailing bytes after the final gzip member"),
        "{message}"
    );
    assert!(
        message.contains(&format!("offset {member_len}")),
        "{message}"
    );

    let mut short = member_with(&tar, &Member::default());
    short.push(0x1f);
    let (kind, message) = validation_error(&short, "a one-byte tail");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(
        message.contains("too short to hold a member magic"),
        "{message}"
    );
}

#[test]
fn path_traversal_inside_a_wrapped_tar_is_rejected() {
    let tar = tar_archive(&[("../evil.txt", b"boom", b'0')]);
    let (kind, message) = validation_error(&member_with(&tar, &Member::default()), "../evil.txt");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("unsafe archive path"), "{message}");
    assert!(message.contains(".."), "{message}");
}

#[test]
fn the_member_budget_shrinks_with_the_bytes_already_produced() {
    assert_eq!(remaining_cap(0), MAX_ARCHIVE_UNCOMPRESSED_BYTES);
    assert_eq!(remaining_cap(MAX_ARCHIVE_UNCOMPRESSED_BYTES - 1), 1);
    assert_eq!(remaining_cap(MAX_ARCHIVE_UNCOMPRESSED_BYTES), 0);
    assert_eq!(remaining_cap(MAX_ARCHIVE_UNCOMPRESSED_BYTES + 1), 0);
}

#[test]
fn the_aggregate_cap_bounds_a_highly_expanding_member() {
    let deflate = high_expansion_deflate(MAX_ARCHIVE_UNCOMPRESSED_BYTES + 1);
    let bytes = member_from_deflate(&deflate, 0, 0, &Member::default());
    let (kind, message) = validation_error(&bytes, "the payload cap is enforced");
    assert_eq!(kind, ErrorKind::Validation);
    assert!(message.contains("gzip member 0 at offset 0"), "{message}");
    assert!(message.contains("output limit"), "{message}");
    assert!(
        message.contains(&MAX_ARCHIVE_UNCOMPRESSED_BYTES.to_string()),
        "{message}"
    );
}
