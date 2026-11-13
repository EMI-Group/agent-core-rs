//! Unit tests for the stored-entry ZIP writer.

use super::*;
use crate::parsing::inflate::crc32;
use crate::parsing::zip::{CENTRAL_HEADER_SIZE, EOCD_SIZE, LOCAL_HEADER_SIZE};
use crate::parsing::ZipArchive;
use openjiuwen_common::error::ErrorKind;

/// The 22-byte end-of-central-directory record of an empty archive.
fn expected_eocd() -> Vec<u8> {
    let mut bytes = b"PK\x05\x06".to_vec();
    bytes.extend_from_slice(&[0; EOCD_SIZE - 4]);
    bytes
}

#[test]
fn entries_round_trip_through_the_reader() {
    let mut writer = ZipWriter::new();
    writer.add_stored("a.txt", b"hi".to_vec()).unwrap();
    writer.add_stored("empty.bin", Vec::new()).unwrap();
    let xml = "<w:document/>".repeat(4).into_bytes();
    writer.add_stored("文档.xml", xml.clone()).unwrap();
    assert_eq!(writer.len(), 3);
    assert!(!writer.is_empty());

    let bytes = writer.finish().unwrap();
    let archive = ZipArchive::from_bytes(bytes.clone()).unwrap();
    assert_eq!(
        archive.entry_names(),
        vec!["a.txt", "empty.bin", "文档.xml"]
    );
    assert_eq!(archive.read_entry("a.txt").unwrap(), b"hi".to_vec());
    assert_eq!(archive.read_entry("empty.bin").unwrap(), Vec::<u8>::new());
    assert_eq!(archive.read_entry("文档.xml").unwrap(), xml);
    assert_eq!(archive.archive_size(), bytes.len());
}

#[test]
fn an_empty_writer_yields_only_the_eocd_record() {
    let bytes = ZipWriter::new().finish().unwrap();
    assert_eq!(bytes, expected_eocd());
    assert_eq!(bytes.len(), EOCD_SIZE);

    let archive = ZipArchive::from_bytes(bytes).unwrap();
    assert!(archive.is_empty());
    assert_eq!(archive.len(), 0);
    assert_eq!(archive.entry_names().len(), 0);
}

#[test]
fn entries_can_be_chained_and_keep_insertion_order() {
    let mut writer = ZipWriter::new();
    writer
        .add_stored("word/document.xml", "<w:document/>".to_owned())
        .unwrap()
        .add_stored("docProps/core.xml", "<cp:coreProperties/>".to_owned())
        .unwrap();
    assert_eq!(writer.len(), 2);
    assert!(!writer.is_empty());

    let archive = ZipArchive::from_bytes(writer.finish().unwrap()).unwrap();
    assert_eq!(
        archive.entry_names(),
        vec!["word/document.xml", "docProps/core.xml"]
    );
    for entry in archive.entries() {
        assert_eq!(entry.compression_method, COMPRESSION_METHOD_STORED);
        assert!(entry.is_utf8_name(), "{entry:?}");
        assert_eq!(entry.compressed_size, entry.uncompressed_size);
    }
    assert_eq!(
        archive.read_entry("docProps/core.xml").unwrap(),
        "<cp:coreProperties/>".as_bytes().to_vec()
    );
}

#[test]
fn directory_entries_round_trip() {
    let mut writer = ZipWriter::new();
    writer.add_stored("word/", Vec::new()).unwrap();
    writer
        .add_stored("word/document.xml", "<w:document/>".to_owned())
        .unwrap();

    let archive = ZipArchive::from_bytes(writer.finish().unwrap()).unwrap();
    let directory = archive.entry("word/").unwrap();
    assert!(directory.is_directory());
    assert_eq!(directory.crc32, 0);
    assert_eq!(directory.compressed_size, 0);
    assert_eq!(directory.uncompressed_size, 0);
    assert!(archive.read_entry("word/").is_err());
    assert_eq!(
        archive.read_entry("word/document.xml").unwrap(),
        "<w:document/>".as_bytes().to_vec()
    );
}

#[test]
fn invalid_names_are_rejected() {
    let mut writer = ZipWriter::new();

    let error = writer.add_stored("", b"x".to_vec()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("must not be empty"), "{error}");

    let long_name = "a".repeat(MAX_NAME_BYTES + 1);
    let error = writer.add_stored(long_name, b"x".to_vec()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("65536"), "{error}");
    assert!(error.to_string().contains("65535"), "{error}");

    assert!(writer.is_empty(), "a rejected entry must not be written");
}

#[test]
fn the_entry_count_limit_is_enforced() {
    let mut writer = ZipWriter::new();
    for index in 0..MAX_ENTRIES {
        writer.add_stored(format!("e{index}"), Vec::new()).unwrap();
    }
    assert_eq!(writer.len(), MAX_ENTRIES);

    let error = writer.add_stored("one-too-many", Vec::new()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("65535 entries"), "{error}");
    assert_eq!(writer.len(), MAX_ENTRIES);
}

#[test]
fn size_and_offset_limits_are_reported() {
    let error = checked_size(usize::MAX, "big.bin").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("4294967295"), "{error}");
    assert!(error.to_string().contains("big.bin"), "{error}");

    let error = checked_offset(usize::MAX).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("4294967295"), "{error}");
    assert!(error.to_string().contains("offset"), "{error}");

    assert_eq!(checked_size(2, "a.txt").unwrap(), 2);
    assert_eq!(checked_offset(37).unwrap(), 37);
}

#[test]
fn a_single_entry_matches_the_specified_byte_layout() {
    // CRC-32 of "hi": `crc32(b"hi")`.
    assert_eq!(crc32(b"hi"), 0xD893_2AAC);

    let mut writer = ZipWriter::new();
    writer.add_stored("a.txt", b"hi".to_vec()).unwrap();
    let bytes = writer.finish().unwrap();

    let expected: &[u8] = &[
        0x50, 0x4b, 0x03, 0x04, // local file header signature
        0x14, 0x00, // version needed to extract: 2.0
        0x00, 0x08, // flags: UTF-8 name
        0x00, 0x00, // compression method: stored
        0x00, 0x00, // last modification time
        0x00, 0x00, // last modification date
        0xac, 0x2a, 0x93, 0xd8, // CRC-32 of "hi"
        0x02, 0x00, 0x00, 0x00, // compressed size
        0x02, 0x00, 0x00, 0x00, // uncompressed size
        0x05, 0x00, // file name length
        0x00, 0x00, // extra field length
        b'a', b'.', b't', b'x', b't', // file name
        b'h', b'i', // payload
        0x50, 0x4b, 0x01, 0x02, // central directory file header signature
        0x14, 0x00, // version made by: 2.0
        0x14, 0x00, // version needed to extract: 2.0
        0x00, 0x08, // flags: UTF-8 name
        0x00, 0x00, // compression method: stored
        0x00, 0x00, // last modification time
        0x00, 0x00, // last modification date
        0xac, 0x2a, 0x93, 0xd8, // CRC-32 of "hi"
        0x02, 0x00, 0x00, 0x00, // compressed size
        0x02, 0x00, 0x00, 0x00, // uncompressed size
        0x05, 0x00, // file name length
        0x00, 0x00, // extra field length
        0x00, 0x00, // file comment length
        0x00, 0x00, // disk number where the entry starts
        0x00, 0x00, // internal file attributes
        0x00, 0x00, 0x00, 0x00, // external file attributes
        0x00, 0x00, 0x00, 0x00, // local header offset
        b'a', b'.', b't', b'x', b't', // file name
        0x50, 0x4b, 0x05, 0x06, // end-of-central-directory signature
        0x00, 0x00, // number of this disk
        0x00, 0x00, // disk with the central directory
        0x01, 0x00, // entries on this disk
        0x01, 0x00, // total entries
        0x33, 0x00, 0x00, 0x00, // central directory size: 46 + 5
        0x25, 0x00, 0x00, 0x00, // central directory offset: 30 + 5 + 2
        0x00, 0x00, // archive comment length
    ];

    let directory_offset = LOCAL_HEADER_SIZE + 5 + 2;
    assert_eq!(
        bytes.len(),
        directory_offset + CENTRAL_HEADER_SIZE + 5 + EOCD_SIZE
    );
    assert_eq!(bytes.len(), expected.len());
    assert_eq!(bytes, expected);

    assert_eq!(
        &bytes[..4],
        b"PK\x03\x04",
        "local header starts at offset 0"
    );
    assert_eq!(
        &bytes[directory_offset..directory_offset + 4],
        b"PK\x01\x02",
        "central directory starts at offset {directory_offset}"
    );
    assert_eq!(
        &bytes[directory_offset + CENTRAL_HEADER_SIZE + 5..][..4],
        b"PK\x05\x06"
    );
}
