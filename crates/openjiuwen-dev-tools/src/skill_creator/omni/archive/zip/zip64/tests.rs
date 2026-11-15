//! Unit tests for the `ZIP64` end record, the locator, and the `ZIP64` extra
//! field.
//!
//! Every fixture is assembled from the byte layout itself, so the decoders are
//! checked against PKWARE APPNOTE 6.3 rather than against any writer in this
//! crate. Nothing touches the network, the environment, or the wall clock.

use super::*;
use openjiuwen_common::error::ErrorKind;

/// Appends a little-endian `u16`.
fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian `u32`.
fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian `u64`.
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// The fields of a hand-built `ZIP64` end-of-central-directory record.
#[derive(Debug, Clone, Copy)]
struct RecordFields {
    /// Declared size of the fields after the size field.
    declared: u64,
    /// The disk the record's archive starts on.
    disk: u32,
    /// The disk the central directory starts on.
    cd_disk: u32,
    /// Entries on this disk, `None` for the same count as the total.
    on_disk: Option<u64>,
    /// Total number of entries.
    total: u64,
    /// Central-directory size.
    cd_size: u64,
    /// Central-directory offset.
    cd_offset: u64,
}

impl Default for RecordFields {
    fn default() -> Self {
        Self {
            declared: ZIP64_EOCD_MIN_REMAINING,
            disk: 0,
            cd_disk: 0,
            on_disk: None,
            total: 0,
            cd_size: 0,
            cd_offset: 0,
        }
    }
}

/// Builds a `ZIP64` end-of-central-directory record.
fn eocd_record(fields: RecordFields) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(ZIP64_EOCD_SIGNATURE);
    put_u64(&mut out, fields.declared);
    put_u16(&mut out, 45);
    put_u16(&mut out, 45);
    put_u32(&mut out, fields.disk);
    put_u32(&mut out, fields.cd_disk);
    put_u64(&mut out, fields.on_disk.unwrap_or(fields.total));
    put_u64(&mut out, fields.total);
    put_u64(&mut out, fields.cd_size);
    put_u64(&mut out, fields.cd_offset);
    out
}

/// Builds a `ZIP64` end-of-central-directory locator.
fn locator(record_offset: u64, total_disks: u32) -> Vec<u8> {
    let mut out = ZIP64_LOCATOR_SIGNATURE.to_vec();
    put_u32(&mut out, 0);
    put_u64(&mut out, record_offset);
    put_u32(&mut out, total_disks);
    out
}

/// Builds the classic end-of-central-directory record an archive always ends
/// with.
fn classic_eocd(total: u16, cd_size: u32, cd_offset: u32) -> Vec<u8> {
    let mut out = b"PK\x05\x06".to_vec();
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u16(&mut out, total);
    put_u16(&mut out, total);
    put_u32(&mut out, cd_size);
    put_u32(&mut out, cd_offset);
    put_u16(&mut out, 0);
    out
}

/// Builds a `ZIP64` extra field (header id `0x0001`) holding `values` in order.
fn zip64_extra(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    put_u16(&mut out, ZIP64_EXTRA_ID);
    put_u16(&mut out, (values.len() * 8) as u16);
    for value in values {
        put_u64(&mut out, *value);
    }
    out
}

#[test]
fn find_directory_follows_the_locator() {
    let record = eocd_record(RecordFields {
        total: 3,
        cd_size: 200,
        cd_offset: 100,
        ..RecordFields::default()
    });
    let mut bytes = record;
    let locator_offset = bytes.len();
    bytes.extend_from_slice(&locator(0, 1));
    assert_eq!(bytes.len(), locator_offset + ZIP64_LOCATOR_FIXED);
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let directory = find_directory(&bytes, eocd_offset)
        .expect("reads the locator")
        .expect("the archive is ZIP64");
    assert_eq!(directory.total_entries, 3);
    assert_eq!(directory.cd_offset, 100);
    assert_eq!(directory.cd_size, 200);
}

#[test]
fn find_directory_accepts_a_record_without_a_locator() {
    let mut bytes = eocd_record(RecordFields {
        total: 1,
        cd_size: 46,
        cd_offset: 8,
        ..RecordFields::default()
    });
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let directory = find_directory(&bytes, eocd_offset)
        .expect("reads a locator-free record")
        .expect("the archive is ZIP64");
    assert_eq!(directory.total_entries, 1);
    assert_eq!(directory.cd_offset, 8);
    assert_eq!(directory.cd_size, 46);
}

#[test]
fn find_directory_reports_none_without_zip64_structures() {
    let bytes = classic_eocd(0, 0, 0);
    assert!(find_directory(&bytes, 0)
        .expect("classic archive")
        .is_none());
}

#[test]
fn locator_pointing_past_the_end_is_rejected() {
    let mut bytes = locator(10_000, 1);
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));
    let eocd_offset = ZIP64_LOCATOR_FIXED;
    let error = find_directory(&bytes, eocd_offset).expect_err("past the end");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("byte 0"), "{message}");
    assert!(message.contains("10000"), "{message}");
}

#[test]
fn locator_naming_a_second_disk_is_rejected() {
    let mut bytes = eocd_record(RecordFields::default());
    let locator_offset = bytes.len();
    bytes.extend_from_slice(&locator(0, 2));
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let error = find_directory(&bytes, eocd_offset).expect_err("multi-disk");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains(&format!("byte {locator_offset}")),
        "{message}"
    );
    assert!(message.contains("multi-disk"), "{message}");
}

#[test]
fn eocd_missing_signature_is_rejected() {
    let mut bytes = vec![0_u8; ZIP64_EOCD_FIXED];
    bytes.extend_from_slice(&locator(0, 1));
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let error = find_directory(&bytes, eocd_offset).expect_err("no signature");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("byte 0"), "{message}");
    assert!(message.contains("expected the signature"), "{message}");
}

#[test]
fn eocd_declared_size_below_the_fixed_fields_is_rejected() {
    let record = eocd_record(RecordFields {
        declared: 20,
        ..RecordFields::default()
    });
    // Truncate the physical record to the 20 bytes it declares.
    let mut bytes = record;
    bytes.truncate(ZIP64_EOCD_HEADER + 20);
    assert_eq!(bytes.len(), ZIP64_EOCD_HEADER + 20);
    bytes.extend_from_slice(&locator(0, 1));
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let error = find_directory(&bytes, eocd_offset).expect_err("short record");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("byte 0"), "{error}");
}

#[test]
fn eocd_declared_size_past_the_input_is_rejected() {
    let mut bytes = eocd_record(RecordFields {
        declared: 400,
        ..RecordFields::default()
    });
    bytes.extend_from_slice(&locator(0, 1));
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let error = find_directory(&bytes, eocd_offset).expect_err("truncated record");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("truncated"), "{error}");
}

#[test]
fn eocd_not_ending_on_the_following_structure_is_rejected() {
    // The locator starts twelve bytes after the record's declared end, which is
    // what a truncated or misdeclared record looks like.
    let mut bytes = eocd_record(RecordFields::default());
    bytes.extend_from_slice(&[0_u8; 12]);
    let locator_offset = bytes.len();
    bytes.extend_from_slice(&locator(0, 1));
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let error = find_directory(&bytes, eocd_offset).expect_err("misaligned record");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("byte 0"), "{message}");
    assert!(
        message.contains(&format!("byte {locator_offset}")),
        "{message}"
    );
}

#[test]
fn eocd_naming_a_second_disk_is_rejected() {
    let record = eocd_record(RecordFields {
        disk: 1,
        total: 1,
        cd_size: 46,
        ..RecordFields::default()
    });
    let mut bytes = record;
    bytes.extend_from_slice(&locator(0, 1));
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let error = find_directory(&bytes, eocd_offset).expect_err("multi-disk");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("multi-disk"), "{error}");
}

#[test]
fn eocd_entry_count_above_the_directory_size_is_rejected() {
    let record = eocd_record(RecordFields {
        total: 4,
        cd_size: 100,
        ..RecordFields::default()
    });
    let mut bytes = record;
    bytes.extend_from_slice(&locator(0, 1));
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&classic_eocd(0, 0, 0));

    let error = find_directory(&bytes, eocd_offset).expect_err("impossible count");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains('4'), "{message}");
    assert!(message.contains("100"), "{message}");
}

#[test]
fn parse_extra_reads_the_escaped_fields_in_zip64_order() {
    let extra = zip64_extra(&[11, 22, 33]);
    let escapes = Zip64Escapes {
        declared: true,
        compressed: true,
        local_offset: true,
    };
    let parsed =
        parse_extra(&extra, 0, extra.len(), escapes, "a.txt", 0).expect("reads every value");
    assert_eq!(parsed.declared, Some(11));
    assert_eq!(parsed.compressed, Some(22));
    assert_eq!(parsed.local_offset, Some(33));
}

#[test]
fn parse_extra_reads_only_the_escaped_values() {
    let extra = zip64_extra(&[77]);
    let escapes = Zip64Escapes {
        declared: false,
        compressed: false,
        local_offset: true,
    };
    let parsed = parse_extra(&extra, 0, extra.len(), escapes, "a.txt", 0).expect("reads offset");
    assert_eq!(parsed.declared, None);
    assert_eq!(parsed.compressed, None);
    assert_eq!(parsed.local_offset, Some(77));
}

#[test]
fn parse_extra_skips_other_extra_fields() {
    let mut extra = Vec::new();
    // A `UT` (0x5455) timestamp field before the `ZIP64` one.
    put_u16(&mut extra, 0x5455);
    put_u16(&mut extra, 5);
    extra.extend_from_slice(b"\x03\x00\x00\x00\x00");
    extra.extend_from_slice(&zip64_extra(&[9]));

    let escapes = Zip64Escapes {
        declared: true,
        ..Zip64Escapes::default()
    };
    let parsed = parse_extra(&extra, 0, extra.len(), escapes, "a.txt", 0).expect("finds ZIP64");
    assert_eq!(parsed.declared, Some(9));
}

#[test]
fn parse_extra_short_payload_is_rejected() {
    let mut extra = Vec::new();
    put_u16(&mut extra, ZIP64_EXTRA_ID);
    put_u16(&mut extra, 4);
    extra.extend_from_slice(&[0_u8; 4]);

    let escapes = Zip64Escapes {
        declared: true,
        ..Zip64Escapes::default()
    };
    let error = parse_extra(&extra, 0, extra.len(), escapes, "a.txt", 46).expect_err("short");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("a.txt"), "{message}");
    assert!(message.contains("byte 46"), "{message}");
}

#[test]
fn parse_extra_without_a_zip64_field_is_rejected() {
    let mut extra = Vec::new();
    put_u16(&mut extra, 0x5455);
    put_u16(&mut extra, 0);
    let escapes = Zip64Escapes {
        declared: true,
        ..Zip64Escapes::default()
    };
    let error = parse_extra(&extra, 0, extra.len(), escapes, "a.txt", 46).expect_err("no ZIP64");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("a.txt"), "{message}");
    assert!(message.contains("0x0001"), "{message}");
}

#[test]
fn parse_extra_field_running_past_the_extra_data_is_rejected() {
    let mut extra = Vec::new();
    put_u16(&mut extra, ZIP64_EXTRA_ID);
    put_u16(&mut extra, 32);
    extra.extend_from_slice(&[0_u8; 4]);

    let escapes = Zip64Escapes {
        declared: true,
        ..Zip64Escapes::default()
    };
    let error = parse_extra(&extra, 0, extra.len(), escapes, "a.txt", 46).expect_err("truncated");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("32"), "{error}");
}

#[test]
fn parse_extra_is_not_read_when_nothing_is_escaped() {
    // The offset is deliberately outside the input: an unescaped record has no
    // reason to read its extra data at all.
    let parsed = parse_extra(
        b"",
        usize::MAX,
        usize::MAX,
        Zip64Escapes::default(),
        "a.txt",
        0,
    )
    .expect("no escaped fields");
    assert_eq!(parsed.declared, None);
    assert_eq!(parsed.compressed, None);
    assert_eq!(parsed.local_offset, None);
}

#[test]
fn resolve_prefers_the_fixed_value_and_needs_a_zip64_value_otherwise() {
    assert_eq!(resolve(7, None, "size", "a.txt", 0).expect("plain"), 7);
    assert_eq!(
        resolve(ZIP64_MARKER, Some(9), "size", "a.txt", 0).expect("escaped"),
        9
    );
    let error = resolve(ZIP64_MARKER, None, "size", "a.txt", 46).expect_err("no value");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("a.txt"), "{message}");
    assert!(message.contains("byte 46"), "{message}");
}

#[test]
fn digital_signature_records_are_recognized_and_skipped() {
    let mut bytes = DIGITAL_SIGNATURE.to_vec();
    put_u16(&mut bytes, 4);
    bytes.extend_from_slice(b"abcd");
    assert!(is_digital_signature(&bytes, 0));
    assert_eq!(skip_digital_signature(&bytes, 0), 10);

    // No signature, no skip.
    assert!(!is_digital_signature(b"PK\x03\x04", 0));
    assert_eq!(skip_digital_signature(b"PK\x03\x04", 0), 0);

    // An empty signature data section skips only the header.
    let mut empty = DIGITAL_SIGNATURE.to_vec();
    put_u16(&mut empty, 0);
    assert_eq!(skip_digital_signature(&empty, 0), 6);

    // A declared length past the input stops at the end of the input.
    let mut long = DIGITAL_SIGNATURE.to_vec();
    put_u16(&mut long, 4000);
    assert_eq!(skip_digital_signature(&long, 0), long.len());
    assert_eq!(skip_digital_signature(&long, long.len()), long.len());
}

#[test]
fn to_usize_keeps_addressable_values() {
    assert_eq!(to_usize(0, "zero").expect("zero"), 0);
    assert_eq!(to_usize(4096, "size").expect("size"), 4096);
}
