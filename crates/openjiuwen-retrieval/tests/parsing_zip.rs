#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]
//! Integration tests for the in-crate ZIP container reader.
//!
//! Every archive is built byte by byte in this file (local headers, central
//! directory, end-of-central-directory record), so the tests check the reader
//! against the format specification rather than against a producer library.
//! The deflated entry reuses the canned raw `DEFLATE` payload of the "hello"
//! `zlib` stream (`78 9c | cb 48 cd c9 c9 07 00 | 06 2c 02 15`). Everything is
//! offline and deterministic.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::parsing::{crc32, zip::compression_method_name, ZipArchive};

/// Name of the stored entry used by most tests.
const STORED_NAME: &str = "hello.txt";
/// Name of the deflated entry used by most tests.
const DEFLATED_NAME: &str = "word/document.xml";
/// Raw `DEFLATE` payload of the canned "hello" stream (fixed Huffman, no `zlib` framing).
const HELLO_DEFLATE: [u8; 7] = [0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00];
/// `CRC-32` of `b"hello world"`.
const HELLO_WORLD_CRC: u32 = 0x0D4A_1185;
/// `CRC-32` of `b"hello"`, the payload `HELLO_DEFLATE` expands to.
const HELLO_CRC: u32 = 0x3610_A686;

/// One entry to place in a hand-built archive.
struct EntrySpec {
    /// Entry name as stored in both headers.
    name: String,
    /// Compression method (0 stored, 8 deflate, anything else for error tests).
    method: u16,
    /// General-purpose flags.
    flags: u16,
    /// Bytes actually stored in the archive.
    compressed: Vec<u8>,
    /// `CRC-32` recorded in both headers.
    crc: u32,
    /// Uncompressed size recorded in both headers.
    uncompressed_size: u32,
}

impl EntrySpec {
    /// Builds an uncompressed entry whose headers carry the real size and `CRC-32`.
    fn stored(name: &str, payload: &[u8]) -> Self {
        Self {
            name: name.to_owned(),
            method: 0,
            flags: 0,
            compressed: payload.to_vec(),
            crc: crc32(payload),
            uncompressed_size: payload.len() as u32,
        }
    }

    /// Builds a `deflate` entry from an already compressed payload.
    fn deflated(name: &str, payload: &[u8], compressed: &[u8]) -> Self {
        Self {
            name: name.to_owned(),
            method: 8,
            flags: 0,
            compressed: compressed.to_vec(),
            crc: crc32(payload),
            uncompressed_size: payload.len() as u32,
        }
    }
}

/// Builds ZIP archives byte by byte (local headers, central directory, EOCD).
#[derive(Default)]
struct ZipBuilder {
    /// Local headers plus entry data, in insertion order.
    data: Vec<u8>,
    /// Central-directory records, in insertion order.
    directory: Vec<u8>,
    /// Number of entries added so far.
    count: u16,
}

impl ZipBuilder {
    /// Appends one entry and its central-directory record.
    fn add(&mut self, spec: &EntrySpec) {
        let name = spec.name.as_bytes();
        let offset = self.data.len() as u32;

        self.data.extend_from_slice(&0x0403_4B50u32.to_le_bytes());
        self.data.extend_from_slice(&20u16.to_le_bytes());
        self.data.extend_from_slice(&spec.flags.to_le_bytes());
        self.data.extend_from_slice(&spec.method.to_le_bytes());
        self.data.extend_from_slice(&0u16.to_le_bytes()); // modification time
        self.data.extend_from_slice(&0u16.to_le_bytes()); // modification date
        self.data.extend_from_slice(&spec.crc.to_le_bytes());
        self.data
            .extend_from_slice(&(spec.compressed.len() as u32).to_le_bytes());
        self.data
            .extend_from_slice(&spec.uncompressed_size.to_le_bytes());
        self.data
            .extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.data.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        self.data.extend_from_slice(name);
        self.data.extend_from_slice(&spec.compressed);

        self.directory
            .extend_from_slice(&0x0201_4B50u32.to_le_bytes());
        self.directory.extend_from_slice(&20u16.to_le_bytes());
        self.directory.extend_from_slice(&20u16.to_le_bytes());
        self.directory.extend_from_slice(&spec.flags.to_le_bytes());
        self.directory.extend_from_slice(&spec.method.to_le_bytes());
        self.directory.extend_from_slice(&0u16.to_le_bytes());
        self.directory.extend_from_slice(&0u16.to_le_bytes());
        self.directory.extend_from_slice(&spec.crc.to_le_bytes());
        self.directory
            .extend_from_slice(&(spec.compressed.len() as u32).to_le_bytes());
        self.directory
            .extend_from_slice(&spec.uncompressed_size.to_le_bytes());
        self.directory
            .extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // comment length
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // disk number start
        self.directory.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        self.directory.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        self.directory.extend_from_slice(&offset.to_le_bytes());
        self.directory.extend_from_slice(name);

        self.count += 1;
    }

    /// Seals the archive with its central directory and EOCD record.
    fn finish(self) -> Vec<u8> {
        self.finish_with_comment("")
    }

    /// Seals the archive with an archive comment.
    fn finish_with_comment(self, comment: &str) -> Vec<u8> {
        let directory_offset = self.data.len() as u32;
        let directory_size = self.directory.len() as u32;
        let mut bytes = self.data;
        bytes.extend_from_slice(&self.directory);
        bytes.extend_from_slice(&0x0605_4B50u32.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes()); // this disk
        bytes.extend_from_slice(&0u16.to_le_bytes()); // central directory disk
        bytes.extend_from_slice(&self.count.to_le_bytes());
        bytes.extend_from_slice(&self.count.to_le_bytes());
        bytes.extend_from_slice(&directory_size.to_le_bytes());
        bytes.extend_from_slice(&directory_offset.to_le_bytes());
        bytes.extend_from_slice(&(comment.len() as u16).to_le_bytes());
        bytes.extend_from_slice(comment.as_bytes());
        bytes
    }
}

/// Builds the archive shared by the read-path tests.
fn sample_archive() -> Vec<u8> {
    let mut builder = ZipBuilder::default();
    builder.add(&EntrySpec::stored(STORED_NAME, b"hello world"));
    builder.add(&EntrySpec::deflated(
        DEFLATED_NAME,
        b"hello",
        &HELLO_DEFLATE,
    ));
    builder.add(&EntrySpec::stored("word/", b""));
    builder.finish()
}

#[test]
fn lists_and_extracts_stored_and_deflated_entries() {
    assert_eq!(HELLO_WORLD_CRC, crc32(b"hello world"));
    assert_eq!(HELLO_CRC, crc32(b"hello"));

    let archive = ZipArchive::from_bytes(sample_archive()).unwrap();
    assert_eq!(archive.len(), 3);
    assert!(!archive.is_empty());
    assert_eq!(
        archive.entry_names(),
        vec![STORED_NAME, DEFLATED_NAME, "word/"]
    );
    assert!(archive.has_entry(STORED_NAME));
    assert!(!archive.has_entry("missing.txt"));

    let stored = archive.entry(STORED_NAME).unwrap();
    assert_eq!(stored.compression_method, 0);
    assert_eq!(stored.compression_method_name(), "stored");
    assert_eq!(stored.uncompressed_size, 11);
    assert_eq!(stored.compressed_size, 11);
    assert_eq!(stored.crc32, HELLO_WORLD_CRC);
    assert!(!stored.is_directory());
    assert!(!stored.is_utf8_name());

    let deflated = archive.entry(DEFLATED_NAME).unwrap();
    assert_eq!(deflated.compression_method, 8);
    assert_eq!(deflated.compression_method_name(), "deflate");
    assert_eq!(deflated.uncompressed_size, 5);
    assert_eq!(deflated.compressed_size, HELLO_DEFLATE.len() as u32);
    assert_eq!(deflated.crc32, HELLO_CRC);

    let directory = archive.entry("word/").unwrap();
    assert!(directory.is_directory());

    assert_eq!(archive.read_entry(STORED_NAME).unwrap(), b"hello world");
    assert_eq!(archive.read_entry(DEFLATED_NAME).unwrap(), b"hello");
    assert_eq!(archive.read_text(DEFLATED_NAME).unwrap(), "hello");

    // Directory entries carry no payload.
    let error = archive.read_entry("word/").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("is a directory"), "{error}");

    let error = archive.read_entry("nope.xml").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains("not found"), "{error}");

    // Lookups are exact: case and separators are part of the identity.
    assert!(!archive.has_entry("HELLO.TXT"));
    assert!(!archive.has_entry("word\\document.xml"));
}

#[test]
fn entry_names_and_archive_metadata_survive_a_comment() {
    let mut builder = ZipBuilder::default();
    builder.add(&EntrySpec::stored(STORED_NAME, b"hello world"));
    let archive = ZipArchive::from_bytes(builder.finish_with_comment("built by hand")).unwrap();
    assert_eq!(archive.entry_names(), vec![STORED_NAME]);
    assert_eq!(archive.read_text(STORED_NAME).unwrap(), "hello world");
    assert!(archive.archive_size() > 22);
}

#[test]
fn read_text_decodes_utf8_and_replaces_invalid_bytes() {
    let text = "héllo wörld\n";
    let mut spec = EntrySpec::stored("notes/übersicht.md", text.as_bytes());
    spec.flags = 0x0800; // UTF-8 name flag
    let mut builder = ZipBuilder::default();
    builder.add(&spec);
    let archive = ZipArchive::from_bytes(builder.finish()).unwrap();

    assert!(archive.entry("notes/übersicht.md").unwrap().is_utf8_name());
    assert_eq!(archive.read_text("notes/übersicht.md").unwrap(), text);

    // Invalid UTF-8 payload bytes degrade to the replacement character.
    let mut builder = ZipBuilder::default();
    builder.add(&EntrySpec::stored("raw.bin", &[0x61, 0xFF, 0x62]));
    let archive = ZipArchive::from_bytes(builder.finish()).unwrap();
    assert_eq!(archive.read_text("raw.bin").unwrap(), "a\u{fffd}b");
}

#[test]
fn read_entry_ignore_case_handles_ascii_casing() {
    let mut builder = ZipBuilder::default();
    builder.add(&EntrySpec::stored("Word/Document.XML", b"hello"));
    let archive = ZipArchive::from_bytes(builder.finish()).unwrap();

    assert!(archive.entry("word/document.xml").is_none());
    assert_eq!(
        archive.read_entry_ignore_case("word/document.xml").unwrap(),
        b"hello"
    );
    assert_eq!(
        archive.read_entry_ignore_case("Word/Document.XML").unwrap(),
        b"hello"
    );
    let error = archive
        .read_entry_ignore_case("word/missing.xml")
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[test]
fn corrupted_payload_fails_the_crc_check() {
    let mut builder = ZipBuilder::default();
    builder.add(&EntrySpec::stored(STORED_NAME, b"hello world"));
    let mut bytes = builder.finish();
    // The stored payload starts after the 30-byte local header and the 9-byte name.
    let payload_start = 30 + STORED_NAME.len();
    bytes[payload_start] ^= 0x01;

    let archive = ZipArchive::from_bytes(bytes).unwrap();
    let error = archive.read_entry(STORED_NAME).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(error.to_string().contains("CRC-32 mismatch"), "{error}");
    // The message names the entry and both checksums.
    assert!(error.to_string().contains(STORED_NAME), "{error}");
    assert!(error.to_string().contains("d4a1185"), "{error}");
}

#[test]
fn size_mismatch_is_rejected_before_the_crc_check() {
    let mut spec = EntrySpec::stored(STORED_NAME, b"hello world");
    spec.uncompressed_size = 4;
    let mut builder = ZipBuilder::default();
    builder.add(&spec);
    let archive = ZipArchive::from_bytes(builder.finish()).unwrap();

    let error = archive.read_entry(STORED_NAME).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(error.to_string().contains("size mismatch"), "{error}");
}

#[test]
fn corrupt_deflate_payload_is_reported() {
    // `0x07` is an invalid block type (BTYPE=11), so the stream cannot be inflated.
    let mut builder = ZipBuilder::default();
    let mut spec = EntrySpec::deflated(DEFLATED_NAME, b"hello", &[0x07]);
    spec.crc = HELLO_CRC;
    builder.add(&spec);
    let archive = ZipArchive::from_bytes(builder.finish()).unwrap();

    let error = archive.read_entry(DEFLATED_NAME).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("invalid DEFLATE block type"),
        "{error}"
    );
}

#[test]
fn unsupported_compression_methods_are_rejected() {
    // Method 12 (bzip2) is not implemented. Build the archive with a stored payload
    // so only the method field can be at fault.
    let mut spec = EntrySpec::stored("compressed.bin", b"payload");
    spec.method = 12;
    let mut builder = ZipBuilder::default();
    builder.add(&spec);
    let error = ZipArchive::from_bytes(builder.finish()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("compression method 12"), "{message}");
    assert!(message.contains("bzip2"), "{message}");
    assert!(message.contains("compressed.bin"), "{message}");

    assert_eq!(compression_method_name(0), "stored");
    assert_eq!(compression_method_name(8), "deflate");
    assert_eq!(compression_method_name(12), "bzip2");
    assert_eq!(compression_method_name(777), "unknown");
}

#[test]
fn encrypted_entries_are_rejected() {
    let mut spec = EntrySpec::stored(STORED_NAME, b"secret");
    spec.flags = 0x0001;
    let mut builder = ZipBuilder::default();
    builder.add(&spec);
    let error = ZipArchive::from_bytes(builder.finish()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("encrypted"), "{error}");
}

#[test]
fn structural_damage_is_reported() {
    // No EOCD record at all.
    let error = ZipArchive::from_bytes(b"PK\x03\x04 truncated".to_vec()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("end-of-central-directory"),
        "{error}"
    );

    // A truncated archive: the EOCD signature survives but its record is short.
    let bytes = sample_archive();
    let error = ZipArchive::from_bytes(bytes[..bytes.len() - 30].to_vec()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    // An empty buffer is not an archive.
    let error = ZipArchive::from_bytes(Vec::new()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn zip64_and_split_archives_are_unsupported() {
    // EOCD with the ZIP64 entry-count sentinel.
    let mut zip64 = Vec::new();
    zip64.extend_from_slice(&0x0605_4B50u32.to_le_bytes());
    zip64.extend_from_slice(&[0u8; 4]);
    zip64.extend_from_slice(&0xFFFFu16.to_le_bytes());
    zip64.extend_from_slice(&0xFFFFu16.to_le_bytes());
    zip64.extend_from_slice(&0u32.to_le_bytes());
    zip64.extend_from_slice(&0u32.to_le_bytes());
    zip64.extend_from_slice(&0u16.to_le_bytes());
    let error = ZipArchive::from_bytes(zip64).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("ZIP64"), "{error}");

    // Split archive: disk numbers are non-zero.
    let mut split = Vec::new();
    split.extend_from_slice(&0x0605_4B50u32.to_le_bytes());
    split.extend_from_slice(&1u16.to_le_bytes());
    split.extend_from_slice(&0u16.to_le_bytes());
    split.extend_from_slice(&0u16.to_le_bytes());
    split.extend_from_slice(&0u16.to_le_bytes());
    split.extend_from_slice(&0u32.to_le_bytes());
    split.extend_from_slice(&0u32.to_le_bytes());
    split.extend_from_slice(&0u16.to_le_bytes());
    let error = ZipArchive::from_bytes(split).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("multi-disk"), "{error}");
}

#[test]
fn local_header_offset_outside_the_archive_is_rejected() {
    let mut builder = ZipBuilder::default();
    builder.add(&EntrySpec::stored(STORED_NAME, b"hello world"));
    let mut bytes = builder.finish();
    // The single central-directory record starts right after the entry data.
    let directory_offset = bytes.len() - 22 - (46 + STORED_NAME.len());
    let local_header_offset = directory_offset + 42;
    bytes[local_header_offset..local_header_offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());

    let archive = ZipArchive::from_bytes(bytes).unwrap();
    let error = archive.read_entry(STORED_NAME).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("truncated ZIP structure"),
        "{error}"
    );
}

#[tokio::test]
async fn open_reads_archives_from_disk() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("document.docx");
    std::fs::write(&path, sample_archive()).unwrap();

    let archive = ZipArchive::open(&path).await.unwrap();
    assert_eq!(
        archive.entry_names(),
        vec![STORED_NAME, DEFLATED_NAME, "word/"]
    );
    assert_eq!(archive.read_text(STORED_NAME).unwrap(), "hello world");

    let missing = directory.path().join("missing.docx");
    let error = ZipArchive::open(&missing).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains("does not exist"), "{error}");
}
