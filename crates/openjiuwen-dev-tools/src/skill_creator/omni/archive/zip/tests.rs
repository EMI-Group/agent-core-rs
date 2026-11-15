//! Reader/writer tests for the in-crate `zip` implementation.
//!
//! Every fixture is assembled here from the byte layout itself (never with
//! [`ArchiveWriter`], except for the round-trip tests), so the reader is checked
//! against the format rather than against its own writer. Nothing touches the
//! network, the environment, or the wall clock.

use super::writer::{put_u16, put_u32};
use super::zip64::{
    ZIP64_EOCD_FIXED, ZIP64_EOCD_SIGNATURE, ZIP64_EXTRA_ID, ZIP64_LOCATOR_FIXED,
    ZIP64_LOCATOR_SIGNATURE, ZIP64_MARKER,
};
use super::*;
use crate::skill_creator::omni::archive::{
    detect, ArchiveEntryKind, ArchiveFormat, ArchiveUnpacker, MAX_ENTRY_UNCOMPRESSED_BYTES,
};
use openjiuwen_common::error::ErrorKind;

/// The fixed header fields a test may vary; [`Fields::stored`] fills the
/// self-consistent defaults for a stored entry.
#[derive(Debug, Clone, Copy, Default)]
struct Fields {
    /// General-purpose flags.
    flags: u16,
    /// Compression method.
    method: u16,
    /// CRC-32 of the uncompressed payload.
    crc: u32,
    /// Compressed payload size.
    packed: u32,
    /// Uncompressed payload size.
    declared: u32,
    /// External attributes.
    external: u32,
}

impl Fields {
    /// Fields of a stored entry whose payload is `data`.
    fn stored(data: &[u8]) -> Self {
        Self {
            flags: 0,
            method: METHOD_STORED,
            crc: crc32(data),
            packed: data.len() as u32,
            declared: data.len() as u32,
            external: 0,
        }
    }
}

/// Builds a local file header; `sizes` overrides the stored size fields.
fn local_header(name: &[u8], fields: Fields, extra: &[u8], sizes: Option<(u32, u32)>) -> Vec<u8> {
    let (packed, declared) = sizes.unwrap_or((fields.packed, fields.declared));
    let mut out = Vec::new();
    out.extend_from_slice(LOCAL_SIGNATURE);
    put_u16(&mut out, VERSION);
    put_u16(&mut out, fields.flags);
    put_u16(&mut out, fields.method);
    put_u16(&mut out, DOS_TIME);
    put_u16(&mut out, DOS_DATE);
    put_u32(&mut out, fields.crc);
    put_u32(&mut out, packed);
    put_u32(&mut out, declared);
    put_u16(&mut out, name.len() as u16);
    put_u16(&mut out, extra.len() as u16);
    out.extend_from_slice(name);
    out.extend_from_slice(extra);
    out
}

/// The three 32-bit values of a central-directory record, so a test can escape
/// any of them with the `ZIP64` marker.
#[derive(Debug, Clone, Copy)]
struct CentralValues {
    /// Compressed size.
    packed: u32,
    /// Uncompressed size.
    declared: u32,
    /// Local-header offset.
    offset: u32,
}

/// Builds a central-directory record with its own extra data and explicit
/// 32-bit values.
fn central_record_parts(
    name: &[u8],
    fields: Fields,
    extra: &[u8],
    values: CentralValues,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(CENTRAL_SIGNATURE);
    put_u16(&mut out, VERSION);
    put_u16(&mut out, VERSION);
    put_u16(&mut out, fields.flags);
    put_u16(&mut out, fields.method);
    put_u16(&mut out, DOS_TIME);
    put_u16(&mut out, DOS_DATE);
    put_u32(&mut out, fields.crc);
    put_u32(&mut out, values.packed);
    put_u32(&mut out, values.declared);
    put_u16(&mut out, name.len() as u16);
    put_u16(&mut out, extra.len() as u16);
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u32(&mut out, fields.external);
    put_u32(&mut out, values.offset);
    out.extend_from_slice(name);
    out.extend_from_slice(extra);
    out
}

/// Builds a central-directory record for `name` at local-header `offset`.
fn central_record(name: &[u8], fields: Fields, offset: u32) -> Vec<u8> {
    central_record_parts(
        name,
        fields,
        b"",
        CentralValues {
            packed: fields.packed,
            declared: fields.declared,
            offset,
        },
    )
}

/// Builds an end-of-central-directory record with an arbitrary comment.
fn eocd_with_comment(total: u16, cd_size: u32, cd_offset: u32, comment: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(EOCD_SIGNATURE);
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u16(&mut out, total);
    put_u16(&mut out, total);
    put_u32(&mut out, cd_size);
    put_u32(&mut out, cd_offset);
    put_u16(&mut out, comment.len() as u16);
    out.extend_from_slice(comment);
    out
}

/// Builds a comment-free end-of-central-directory record.
fn eocd(total: u16, cd_size: u32, cd_offset: u32) -> Vec<u8> {
    eocd_with_comment(total, cd_size, cd_offset, b"")
}

/// Builds an end-of-central-directory record that carries the `ZIP64` escape
/// values in every field a `zip` archive escapes.
fn eocd_escaped() -> Vec<u8> {
    eocd_with_comment(u16::MAX, ZIP64_MARKER, ZIP64_MARKER, b"")
}

/// One entry of a hand-assembled archive.
struct Spec<'a> {
    /// Entry name as stored in the archive.
    name: &'a [u8],
    /// Header fields, shared by both headers.
    fields: Fields,
    /// Local-header extra field.
    extra: &'a [u8],
    /// Local-header size override (a streamed entry keeps them zero).
    local_sizes: Option<(u32, u32)>,
    /// Compressed payload written after the local header.
    payload: &'a [u8],
}

impl<'a> Spec<'a> {
    /// A stored entry whose payload is `payload`.
    fn stored(name: &'a [u8], payload: &'a [u8]) -> Self {
        Self {
            name,
            fields: Fields::stored(payload),
            extra: b"",
            local_sizes: None,
            payload,
        }
    }
}

/// Assembles local records, their central directory, and an end record.
fn assemble(specs: &[Spec<'_>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut offsets = Vec::with_capacity(specs.len());
    for spec in specs {
        offsets.push(bytes.len() as u32);
        bytes.extend_from_slice(&local_header(
            spec.name,
            spec.fields,
            spec.extra,
            spec.local_sizes,
        ));
        bytes.extend_from_slice(spec.payload);
    }
    let cd_offset = bytes.len();
    for (spec, &offset) in specs.iter().zip(&offsets) {
        bytes.extend_from_slice(&central_record(spec.name, spec.fields, offset));
    }
    let cd_size = bytes.len() - cd_offset;
    bytes.extend_from_slice(&eocd(specs.len() as u16, cd_size as u32, cd_offset as u32));
    bytes
}

/// Assembles an archive around one entry.
fn single(spec: Spec<'_>) -> Vec<u8> {
    assemble(&[spec])
}

/// Builds a one-entry archive whose central record is well formed but whose
/// local region is `tail`, placed after a comment-carrying end record so the
/// local region really is the end of the file.
fn archive_with_tail(name: &[u8], fields: Fields, tail: &[u8]) -> Vec<u8> {
    let cd_size = CENTRAL_FIXED + name.len();
    let local_offset = cd_size + EOCD_FIXED;
    let mut bytes = central_record(name, fields, local_offset as u32);
    debug_assert_eq!(bytes.len(), cd_size);
    bytes.extend_from_slice(&eocd_with_comment(1, cd_size as u32, 0, tail));
    bytes
}

/// Appends a little-endian `u64`, the width every `ZIP64` field uses.
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
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

/// Builds a `ZIP64` end-of-central-directory record with the default 44-byte
/// declared size.
fn zip64_eocd(total: u64, cd_size: u64, cd_offset: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(ZIP64_EOCD_SIGNATURE);
    put_u64(&mut out, 44);
    put_u16(&mut out, 45);
    put_u16(&mut out, 45);
    put_u32(&mut out, 0);
    put_u32(&mut out, 0);
    put_u64(&mut out, total);
    put_u64(&mut out, total);
    put_u64(&mut out, cd_size);
    put_u64(&mut out, cd_offset);
    out
}

/// Builds a `ZIP64` end-of-central-directory locator.
fn zip64_locator(record_offset: u64, total_disks: u32) -> Vec<u8> {
    let mut out = ZIP64_LOCATOR_SIGNATURE.to_vec();
    put_u32(&mut out, 0);
    put_u64(&mut out, record_offset);
    put_u32(&mut out, total_disks);
    out
}

/// The `ZIP64` structures a hand-assembled archive carries.
///
/// The classic end record always carries the `ZIP64` escape values, which is what
/// a real `ZIP64` writer emits.
#[derive(Debug, Clone, Copy, Default)]
struct Zip64Layout {
    /// Write the `ZIP64` end-of-central-directory record.
    record: bool,
    /// Write the `ZIP64` locator that names it.
    locator: bool,
    /// Write an archive digital-signature record after the central directory.
    digital_signature: bool,
}

/// One entry of a hand-assembled `ZIP64` archive.
struct Zip64Spec<'a> {
    /// Entry name as stored in the archive.
    name: &'a [u8],
    /// Header fields, shared by both headers.
    fields: Fields,
    /// Compressed payload written after the local header.
    payload: &'a [u8],
    /// Escape both sizes and carry them in a `ZIP64` extra field.
    escape_sizes: bool,
    /// Escape the local-header offset and carry it in the same extra field.
    escape_offset: bool,
}

/// Assembles an archive with the requested `ZIP64` structures.
fn assemble_zip64(specs: &[Zip64Spec<'_>], layout: Zip64Layout) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut offsets = Vec::with_capacity(specs.len());
    for spec in specs {
        offsets.push(bytes.len() as u64);
        bytes.extend_from_slice(&local_header(spec.name, spec.fields, b"", None));
        bytes.extend_from_slice(spec.payload);
    }
    let cd_offset = bytes.len();
    for (spec, &offset) in specs.iter().zip(&offsets) {
        let mut values = Vec::new();
        if spec.escape_sizes {
            values.push(u64::from(spec.fields.declared));
            values.push(u64::from(spec.fields.packed));
        }
        if spec.escape_offset {
            values.push(offset);
        }
        let extra = zip64_extra(&values);
        let values = CentralValues {
            packed: if spec.escape_sizes {
                ZIP64_MARKER
            } else {
                spec.fields.packed
            },
            declared: if spec.escape_sizes {
                ZIP64_MARKER
            } else {
                spec.fields.declared
            },
            offset: if spec.escape_offset {
                ZIP64_MARKER
            } else {
                offset as u32
            },
        };
        bytes.extend_from_slice(&central_record_parts(
            spec.name,
            spec.fields,
            &extra,
            values,
        ));
    }
    let cd_size = bytes.len() - cd_offset;
    if layout.digital_signature {
        bytes.extend_from_slice(super::zip64::DIGITAL_SIGNATURE);
        put_u16(&mut bytes, 0);
    }
    let record_offset = bytes.len();
    if layout.record {
        bytes.extend_from_slice(&zip64_eocd(
            specs.len() as u64,
            cd_size as u64,
            cd_offset as u64,
        ));
    }
    if layout.locator {
        bytes.extend_from_slice(&zip64_locator(record_offset as u64, 1));
    }
    bytes.extend_from_slice(&eocd_escaped());
    bytes
}

#[test]
fn writer_round_trips_files_and_directories() {
    let entries = vec![
        ArchiveEntry {
            path: "readme.txt".to_owned(),
            bytes: b"hello zip\n".to_vec(),
            is_dir: false,
            mode: Some(UNIX_FILE_MODE),
            ..ArchiveEntry::default()
        },
        ArchiveEntry {
            path: "docs/".to_owned(),
            bytes: Vec::new(),
            is_dir: true,
            kind: ArchiveEntryKind::Directory,
            mode: Some(UNIX_DIRECTORY_MODE),
            ..ArchiveEntry::default()
        },
        ArchiveEntry {
            path: "café/naïve.txt".to_owned(),
            bytes: b"accented".to_vec(),
            is_dir: false,
            mode: Some(UNIX_FILE_MODE),
            ..ArchiveEntry::default()
        },
        ArchiveEntry {
            path: "docs/nested/deep.txt".to_owned(),
            bytes: b"deep".to_vec(),
            is_dir: false,
            mode: Some(UNIX_FILE_MODE),
            ..ArchiveEntry::default()
        },
    ];

    let bytes = ArchiveWriter::write_stored(&entries).expect("writes");
    assert_eq!(detect(&bytes), Some(ArchiveFormat::Zip));
    assert_eq!(&bytes[..4], LOCAL_SIGNATURE);
    assert_eq!(read_u16_le(&bytes, 4).expect("version"), VERSION);
    assert_eq!(read_u16_le(&bytes, 6).expect("flags"), 0);
    assert_eq!(read_u16_le(&bytes, 8).expect("method"), METHOD_STORED);
    assert_eq!(read_u16_le(&bytes, 10).expect("time"), DOS_TIME);
    assert_eq!(read_u16_le(&bytes, 12).expect("date"), DOS_DATE);

    // The writer's fixed modes come back on read, which is why the entries above
    // declare them.
    let decoded = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect("reads");
    assert_eq!(decoded, entries);
    assert_eq!(
        ArchiveWriter::write_stored(&entries).expect("writes again"),
        bytes,
        "the writer is deterministic"
    );
}

#[test]
fn stored_entries_read_from_hand_built_bytes() {
    let utf8_name: &[u8] = "naïve.txt".as_bytes();
    let cp437_name = [0x82_u8, b'.', b't', b'x', b't']; // `é.txt` in CP437
    let archive = assemble(&[
        Spec {
            name: utf8_name,
            fields: Fields {
                flags: FLAG_UTF8,
                ..Fields::stored(b"one")
            },
            ..Spec::stored(utf8_name, b"one")
        },
        Spec::stored(&cp437_name, b"two"),
    ]);

    let entries = ArchiveUnpacker::unpack_auto(&archive).expect("reads");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].path, "naïve.txt");
    assert_eq!(entries[0].bytes, b"one");
    assert!(!entries[0].is_dir);
    assert_eq!(entries[1].path, "é.txt");
    assert_eq!(entries[1].bytes, b"two");
    // A DOS-written record declares no Unix mode.
    assert_eq!(entries[0].mode, None);
}

#[test]
fn cp437_decoding_maps_the_upper_half() {
    assert_eq!(decode_cp437(b"plain"), "plain");
    assert_eq!(decode_cp437(b"a\x82\xc4\xff"), "aé─\u{a0}");
    assert_eq!(decode_cp437(&[0x80]), "Ç");
    assert_eq!(decode_cp437(&[0xA4]), "ñ");
    assert_eq!(decode_cp437(&[0xE1]), "ß");
}

#[test]
fn utf8_flagged_invalid_name_is_rejected() {
    let fields = Fields {
        flags: FLAG_UTF8,
        ..Fields::stored(b"")
    };
    let archive = single(Spec {
        name: b"bad\xff.txt",
        fields,
        ..Spec::stored(b"bad\xff.txt", b"")
    });
    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("invalid utf-8");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("UTF-8"), "{error}");
}

#[test]
fn deflate_entries_inflate_end_to_end() {
    // BFINAL = 1, BTYPE = 00 (stored block), LEN = 5, NLEN = !5, then the bytes.
    let payload = [
        0x01_u8, 0x05, 0x00, 0xFA, 0xFF, b'h', b'e', b'l', b'l', b'o',
    ];
    let fields = Fields {
        method: METHOD_DEFLATE,
        crc: crc32(b"hello"),
        packed: payload.len() as u32,
        declared: 5,
        ..Fields::default()
    };
    let archive = single(Spec {
        name: b"deflated.txt",
        fields,
        ..Spec::stored(b"deflated.txt", &payload)
    });

    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect("reads");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "deflated.txt");
    assert_eq!(entries[0].bytes, b"hello");
    assert!(!entries[0].is_dir);
}

#[test]
fn crc_mismatch_names_the_entry_and_both_values() {
    let fields = Fields {
        crc: 0xDEAD_BEEF,
        ..Fields::stored(b"hello")
    };
    let archive = single(Spec {
        name: b"bad-crc.txt",
        fields,
        ..Spec::stored(b"bad-crc.txt", b"hello")
    });

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("crc mismatch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("bad-crc.txt"), "{message}");
    assert!(message.contains("0xdeadbeef"), "{message}");
    assert!(
        message.contains(&format!("0x{:08x}", crc32(b"hello"))),
        "{message}"
    );
}

#[test]
fn truncated_central_directory_names_the_offset() {
    // A central record cut short: the end record still describes it.
    let fields = Fields::stored(b"hello");
    let mut bytes = local_header(b"a.txt", fields, b"", None);
    bytes.extend_from_slice(b"hello");
    let cd_offset = bytes.len();
    let record = central_record(b"a.txt", fields, 0);
    bytes.extend_from_slice(&record[..20]);
    bytes.extend_from_slice(&eocd(1, 20, cd_offset as u32));
    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("cut record");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains(&format!("offset {cd_offset}")),
        "{error}"
    );

    // A central directory that starts past the end of the archive.
    let mut bytes = local_header(b"a.txt", fields, b"", None);
    bytes.extend_from_slice(b"hello");
    let past_end = bytes.len() as u32 + 64;
    bytes.extend_from_slice(&eocd(1, 46, past_end));
    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("past the end");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains(&format!("byte {past_end}")),
        "{error}"
    );
}

#[test]
fn truncated_local_header_names_the_offset() {
    let fields = Fields::stored(b"hello");
    let header = local_header(b"a.txt", fields, b"", None);
    let tail = &header[..12];
    let bytes = archive_with_tail(b"a.txt", fields, tail);
    let local_offset = CENTRAL_FIXED + b"a.txt".len() + EOCD_FIXED;
    let name_length_offset = local_offset + 26;

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("truncated header");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains(&format!("offset {name_length_offset}")),
        "expected the name-length offset {name_length_offset}: {error}"
    );
}

#[test]
fn truncated_payload_names_the_offset() {
    let fields = Fields::stored(b"0123456789");
    let mut tail = local_header(b"data.txt", fields, b"", None);
    tail.extend_from_slice(b"012");
    let data_start =
        CENTRAL_FIXED + b"data.txt".len() + EOCD_FIXED + LOCAL_FIXED + b"data.txt".len();
    let bytes = archive_with_tail(b"data.txt", fields, &tail);

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("short payload");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains(&format!("offset {data_start}")),
        "expected the data offset {data_start}: {error}"
    );
}

#[test]
fn local_header_sizes_are_ignored() {
    let fields = Fields::stored(b"streamed");
    let archive = single(Spec {
        name: b"streamed.txt",
        fields,
        local_sizes: Some((0, 0)),
        ..Spec::stored(b"streamed.txt", b"streamed")
    });

    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect("reads");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].bytes, b"streamed");
}

#[test]
fn unknown_compression_method_is_unsupported() {
    let fields = Fields {
        method: 12,
        ..Fields::stored(b"")
    };
    let archive = single(Spec {
        name: b"bzip2.txt",
        fields,
        ..Spec::stored(b"bzip2.txt", b"")
    });

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("method 12");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("method 12"), "{message}");
    assert!(message.contains("bzip2.txt"), "{message}");
}

#[test]
fn encrypted_entries_are_unsupported() {
    let fields = Fields {
        flags: FLAG_ENCRYPTED,
        ..Fields::stored(b"secret")
    };
    let archive = single(Spec {
        name: b"secret.txt",
        fields,
        ..Spec::stored(b"secret.txt", b"secret")
    });

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("encrypted");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("encrypt"), "{message}");
    assert!(message.contains("secret.txt"), "{message}");
}

#[test]
fn zip64_end_record_and_locator_unpack_like_the_classic_twin() {
    let payload: &[u8] = b"hello zip64\n";
    let classic = single(Spec::stored(b"a.txt", payload));
    let zip64 = assemble_zip64(
        &[Zip64Spec {
            name: b"a.txt",
            fields: Fields::stored(payload),
            payload,
            escape_sizes: false,
            escape_offset: false,
        }],
        Zip64Layout {
            record: true,
            locator: true,
            ..Zip64Layout::default()
        },
    );

    let classic_entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &classic).expect("classic");
    let zip64_entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &zip64).expect("zip64");
    assert_eq!(zip64_entries, classic_entries);
    assert_eq!(zip64_entries[0].bytes, payload);
}

#[test]
fn zip64_end_record_without_a_locator_is_tolerated() {
    let payload: &[u8] = b"tolerated\n";
    let zip64 = assemble_zip64(
        &[Zip64Spec {
            name: b"a.txt",
            fields: Fields::stored(payload),
            payload,
            escape_sizes: false,
            escape_offset: false,
        }],
        Zip64Layout {
            record: true,
            ..Zip64Layout::default()
        },
    );

    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &zip64).expect("reads");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "a.txt");
    assert_eq!(entries[0].bytes, payload);
}

#[test]
fn per_entry_zip64_extras_supply_sizes_and_offsets() {
    // BFINAL = 1, BTYPE = 00 (stored block), LEN = 5, NLEN = !5, then the bytes.
    let deflated: &[u8] = &[0x01, 0x05, 0x00, 0xFA, 0xFF, b'h', b'e', b'l', b'l', b'o'];
    let stored: &[u8] = b"stored bytes\n";
    let stored_fields = Fields::stored(stored);
    let deflate_fields = Fields {
        method: METHOD_DEFLATE,
        crc: crc32(b"hello"),
        packed: deflated.len() as u32,
        declared: 5,
        ..Fields::default()
    };

    let classic = assemble(&[
        Spec {
            name: b"stored.txt",
            fields: stored_fields,
            ..Spec::stored(b"stored.txt", stored)
        },
        Spec {
            name: b"deflated.txt",
            fields: deflate_fields,
            ..Spec::stored(b"deflated.txt", deflated)
        },
    ]);
    let zip64 = assemble_zip64(
        &[
            Zip64Spec {
                name: b"stored.txt",
                fields: stored_fields,
                payload: stored,
                escape_sizes: true,
                escape_offset: true,
            },
            Zip64Spec {
                name: b"deflated.txt",
                fields: deflate_fields,
                payload: deflated,
                escape_sizes: true,
                escape_offset: false,
            },
        ],
        Zip64Layout {
            record: true,
            locator: true,
            ..Zip64Layout::default()
        },
    );

    let classic_entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &classic).expect("classic");
    let zip64_entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &zip64).expect("zip64");
    assert_eq!(zip64_entries, classic_entries);
    assert_eq!(zip64_entries[1].bytes, b"hello");
}

#[test]
fn zip64_archive_with_a_digital_signature_record_unpacks() {
    let payload: &[u8] = b"signed\n";
    let zip64 = assemble_zip64(
        &[Zip64Spec {
            name: b"a.txt",
            fields: Fields::stored(payload),
            payload,
            escape_sizes: true,
            escape_offset: true,
        }],
        Zip64Layout {
            record: true,
            locator: true,
            digital_signature: true,
        },
    );

    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &zip64).expect("reads");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "a.txt");
    assert_eq!(entries[0].bytes, payload);
}

#[test]
fn zip64_escape_values_without_any_zip64_structure_are_rejected() {
    let fields = Fields::stored(b"hello");
    let mut bytes = local_header(b"a.txt", fields, b"", None);
    bytes.extend_from_slice(b"hello");
    let cd_offset = bytes.len();
    bytes.extend_from_slice(&central_record(b"a.txt", fields, 0));
    let eocd_offset = bytes.len();
    bytes.extend_from_slice(&eocd_escaped());

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("escape values");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("ZIP64"), "{message}");
    assert!(
        message.contains(&format!("byte {eocd_offset}")),
        "expected the end-record offset {eocd_offset}: {message}"
    );
    assert!(cd_offset < eocd_offset);
}

#[test]
fn zip64_locator_pointing_past_the_end_is_rejected() {
    let fields = Fields::stored(b"hello");
    let mut bytes = local_header(b"a.txt", fields, b"", None);
    bytes.extend_from_slice(b"hello");
    bytes.extend_from_slice(&central_record(b"a.txt", fields, 0));
    let locator_offset = bytes.len();
    let record_offset = u64::from(u32::try_from(bytes.len()).expect("fits")) + 1024;
    bytes.extend_from_slice(&zip64_locator(record_offset, 1));
    bytes.extend_from_slice(&eocd_escaped());

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("past the end");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains(&format!("byte {locator_offset}")),
        "expected the locator offset {locator_offset}: {message}"
    );
    assert!(message.contains(&record_offset.to_string()), "{message}");
}

#[test]
fn truncated_zip64_end_record_is_rejected() {
    // A record that stops after its disk fields, with the locator naming it.
    let mut bytes = Vec::new();
    let record_offset = bytes.len();
    bytes.extend_from_slice(ZIP64_EOCD_SIGNATURE);
    put_u64(&mut bytes, 44);
    put_u16(&mut bytes, 45);
    put_u16(&mut bytes, 45);
    put_u32(&mut bytes, 0);
    put_u32(&mut bytes, 0);
    let locator_offset = bytes.len();
    bytes.extend_from_slice(&zip64_locator(record_offset as u64, 1));
    bytes.extend_from_slice(&eocd_escaped());

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("truncated record");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains(&format!("byte {record_offset}")),
        "expected the record offset {record_offset}: {message}"
    );
    assert!(
        message.contains(&format!("byte {locator_offset}")),
        "expected the locator offset {locator_offset}: {message}"
    );
}

#[test]
fn zip64_extra_field_that_stops_early_is_rejected() {
    let payload: &[u8] = b"hello";
    let fields = Fields::stored(payload);
    let mut bytes = local_header(b"a.txt", fields, b"", None);
    bytes.extend_from_slice(payload);
    let cd_offset = bytes.len();
    // Four bytes of `ZIP64` payload are not a 64-bit uncompressed size.
    let extra = [0x01_u8, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00];
    let record = central_record_parts(
        b"a.txt",
        fields,
        &extra,
        CentralValues {
            packed: fields.packed,
            declared: ZIP64_MARKER,
            offset: 0,
        },
    );
    bytes.extend_from_slice(&record);
    bytes.extend_from_slice(&eocd(1, record.len() as u32, cd_offset as u32));

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("short extra");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("a.txt"), "{message}");
    assert!(message.contains("ZIP64"), "{message}");
}

#[test]
fn escaped_size_without_a_zip64_extra_field_is_rejected() {
    let payload: &[u8] = b"hello";
    let fields = Fields::stored(payload);
    let mut bytes = local_header(b"a.txt", fields, b"", None);
    bytes.extend_from_slice(payload);
    let cd_offset = bytes.len();
    let record = central_record_parts(
        b"a.txt",
        fields,
        b"",
        CentralValues {
            packed: fields.packed,
            declared: ZIP64_MARKER,
            offset: 0,
        },
    );
    bytes.extend_from_slice(&record);
    bytes.extend_from_slice(&eocd(1, record.len() as u32, cd_offset as u32));

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("no extra field");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("a.txt"), "{message}");
    assert!(message.contains("0x0001"), "{message}");
}

#[test]
fn symlink_entries_carry_their_target() {
    let target: &[u8] = b"../a.txt";
    let fields = Fields {
        external: UNIX_SYMLINK_MODE << 16,
        ..Fields::stored(target)
    };
    let archive = single(Spec {
        name: b"links/a.txt",
        fields,
        ..Spec::stored(b"links/a.txt", target)
    });

    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect("reads");
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].kind,
        ArchiveEntryKind::Symlink,
        "S_IFLNK in the external attributes marks a symlink"
    );
    assert_eq!(entries[0].link_target.as_deref(), Some("../a.txt"));
    assert_eq!(entries[0].bytes, Vec::<u8>::new());
    assert!(!entries[0].is_dir);
    assert_eq!(entries[0].mode, Some(UNIX_SYMLINK_MODE));
}

#[test]
fn symlink_entry_paths_stay_validated_but_targets_are_opaque() {
    // A hostile target is only a string in the returned record.
    let target: &[u8] = b"../../etc/passwd";
    let fields = Fields {
        external: UNIX_SYMLINK_MODE << 16,
        ..Fields::stored(target)
    };
    let archive = single(Spec {
        name: b"link.txt",
        fields,
        ..Spec::stored(b"link.txt", target)
    });
    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect("reads");
    assert_eq!(
        entries[0].link_target.as_deref(),
        Some("../../etc/passwd"),
        "the target is never resolved or written"
    );

    // The entry path itself still goes through the shared validation.
    let archive = single(Spec {
        name: b"../link.txt",
        fields,
        ..Spec::stored(b"../link.txt", target)
    });
    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("unsafe path");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("unsafe archive path"), "{error}");
}

#[test]
fn writer_round_trips_a_symlink_entry() {
    let entries = vec![
        ArchiveEntry {
            path: "a.txt".to_owned(),
            bytes: b"target file\n".to_vec(),
            is_dir: false,
            mode: Some(UNIX_FILE_MODE),
            ..ArchiveEntry::default()
        },
        ArchiveEntry {
            path: "links/a.txt".to_owned(),
            bytes: Vec::new(),
            is_dir: false,
            kind: ArchiveEntryKind::Symlink,
            link_target: Some("../a.txt".to_owned()),
            mode: Some(UNIX_SYMLINK_MODE),
        },
    ];

    let bytes = ArchiveWriter::write_stored(&entries).expect("writes");
    let symlink_attributes = (UNIX_SYMLINK_MODE << 16).to_le_bytes();
    assert!(
        bytes
            .windows(4)
            .any(|window| window == symlink_attributes.as_slice()),
        "the external attributes carry S_IFLNK"
    );
    let decoded = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect("reads");
    assert_eq!(decoded, entries);
    assert_eq!(decoded[1].kind, ArchiveEntryKind::Symlink);
    assert_eq!(decoded[1].link_target.as_deref(), Some("../a.txt"));
}

#[test]
fn writer_takes_a_symlink_target_from_bytes_and_rejects_a_missing_one() {
    let entries = vec![ArchiveEntry {
        path: "link.txt".to_owned(),
        bytes: b"../a.txt".to_vec(),
        is_dir: false,
        kind: ArchiveEntryKind::Symlink,
        ..ArchiveEntry::default()
    }];
    let bytes = ArchiveWriter::write_stored(&entries).expect("writes the target in bytes");
    let decoded = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect("reads");
    assert_eq!(decoded[0].link_target.as_deref(), Some("../a.txt"));

    let entries = vec![ArchiveEntry {
        path: "link.txt".to_owned(),
        bytes: Vec::new(),
        is_dir: false,
        kind: ArchiveEntryKind::Symlink,
        ..ArchiveEntry::default()
    }];
    let error = ArchiveWriter::write_stored(&entries).expect_err("no target");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("no target"), "{error}");
}

#[test]
fn unix_modes_are_reported_when_the_archive_declares_them() {
    let file_fields = Fields {
        external: UNIX_FILE_MODE << 16,
        ..Fields::stored(b"body")
    };
    let directory_fields = Fields {
        external: (UNIX_DIRECTORY_MODE << 16) | DOS_DIRECTORY_ATTRIBUTE,
        ..Fields::default()
    };
    let archive = assemble(&[
        Spec {
            name: b"a.txt",
            fields: file_fields,
            ..Spec::stored(b"a.txt", b"body")
        },
        Spec {
            name: b"docs",
            fields: directory_fields,
            ..Spec::stored(b"docs", b"")
        },
    ]);

    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect("reads");
    assert_eq!(entries[0].mode, Some(UNIX_FILE_MODE));
    assert_eq!(entries[0].kind, ArchiveEntryKind::File);
    assert_eq!(entries[1].mode, Some(UNIX_DIRECTORY_MODE));
    assert_eq!(entries[1].kind, ArchiveEntryKind::Directory);
    assert!(entries[1].is_dir);
}

#[test]
fn declared_size_above_the_cap_is_rejected_before_allocating() {
    let declared = (MAX_ENTRY_UNCOMPRESSED_BYTES + 1) as u32;
    // An empty final stored DEFLATE block, so the payload stays two bytes.
    let payload = [0x03_u8, 0x00];
    let fields = Fields {
        method: METHOD_DEFLATE,
        crc: 0,
        packed: payload.len() as u32,
        declared,
        ..Fields::default()
    };
    let archive = single(Spec {
        name: b"bomb.txt",
        fields,
        ..Spec::stored(b"bomb.txt", &payload)
    });

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("cap");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("per-entry"), "{message}");
    assert!(message.contains(&declared.to_string()), "{message}");
}

#[test]
fn unsafe_entry_names_are_rejected() {
    for name in [
        &b"../evil.txt"[..],
        &b"/etc/passwd"[..],
        &b"a\\b.txt"[..],
        &b"C:evil.txt"[..],
    ] {
        let archive = single(Spec::stored(name, b"payload"));
        let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("unsafe name");
        assert_eq!(
            error.kind(),
            ErrorKind::Validation,
            "{:?}",
            String::from_utf8_lossy(name)
        );
        assert!(error.to_string().contains("unsafe archive path"), "{error}");
    }
}

#[test]
fn empty_archive_has_no_entries() {
    let mut bytes = EOCD_SIGNATURE.to_vec();
    bytes.extend_from_slice(&[0_u8; EOCD_FIXED - EOCD_SIGNATURE.len()]);
    assert_eq!(bytes.len(), EOCD_FIXED);
    assert_eq!(detect(&bytes), Some(ArchiveFormat::Zip));
    assert_eq!(
        ArchiveUnpacker::unpack_auto(&bytes).expect("empty archive"),
        Vec::<ArchiveEntry>::new()
    );
}

#[test]
fn writer_rejects_invalid_entries() {
    let empty: Vec<ArchiveEntry> = Vec::new();
    let error = ArchiveWriter::write_stored(&empty).expect_err("no entries");
    assert_eq!(error.kind(), ErrorKind::Validation);

    let directory_with_content = vec![ArchiveEntry {
        path: "docs/".to_owned(),
        bytes: b"x".to_vec(),
        is_dir: true,
        kind: ArchiveEntryKind::Directory,
        ..ArchiveEntry::default()
    }];
    let error = ArchiveWriter::write_stored(&directory_with_content).expect_err("directory bytes");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("directory"), "{error}");

    let file_with_slash = vec![ArchiveEntry {
        path: "docs/".to_owned(),
        bytes: Vec::new(),
        is_dir: false,
        ..ArchiveEntry::default()
    }];
    let error = ArchiveWriter::write_stored(&file_with_slash).expect_err("trailing slash");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("ends with `/`"), "{error}");

    let unsafe_path = vec![ArchiveEntry {
        path: "../escape.txt".to_owned(),
        bytes: Vec::new(),
        is_dir: false,
        ..ArchiveEntry::default()
    }];
    let error = ArchiveWriter::write_stored(&unsafe_path).expect_err("unsafe path");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("unsafe archive path"), "{error}");
}

#[test]
fn directory_marked_by_attributes_has_no_bytes() {
    let fields = Fields {
        external: DOS_DIRECTORY_ATTRIBUTE,
        ..Fields::default()
    };
    let archive = single(Spec {
        name: b"docs",
        fields,
        ..Spec::stored(b"docs", b"")
    });

    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect("reads");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "docs");
    assert!(entries[0].is_dir);
    assert_eq!(entries[0].bytes, Vec::<u8>::new());
    assert_eq!(
        entries[0].mode, None,
        "a zero high half declares no Unix mode"
    );
}

/// A 330-byte archive written by Python's `zipfile` (Unix `create_system`, a
/// `zlib`-deflated entry, a stored entry, and a directory), so the reader is
/// checked against bytes no implementation here produced.
const PYTHON_ZIP: &[u8] = &[
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x64, 0x6f,
    0x63, 0x73, 0x2f, 0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21,
    0x00, 0x98, 0x06, 0x2f, 0x1c, 0x0d, 0x00, 0x00, 0x00, 0xa0, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00,
    0x00, 0x64, 0x6f, 0x63, 0x73, 0x2f, 0x67, 0x75, 0x69, 0x64, 0x65, 0x2e, 0x6d, 0x64, 0x53, 0x56,
    0x70, 0x2f, 0xcd, 0x4c, 0x49, 0xe5, 0x52, 0x1e, 0xa4, 0x34, 0x00, 0x50, 0x4b, 0x03, 0x04, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0xea, 0x59, 0x06, 0x6c, 0x0d, 0x00, 0x00,
    0x00, 0x0d, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x70, 0x6c, 0x61, 0x69, 0x6e, 0x2e, 0x74,
    0x78, 0x74, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x64, 0x20, 0x62, 0x79, 0x74, 0x65, 0x73, 0x0a, 0x50,
    0x4b, 0x01, 0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x64, 0x6f, 0x63,
    0x73, 0x2f, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x21, 0x00, 0x98, 0x06, 0x2f, 0x1c, 0x0d, 0x00, 0x00, 0x00, 0xa0, 0x00, 0x00, 0x00, 0x0d, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x23, 0x00, 0x00, 0x00,
    0x64, 0x6f, 0x63, 0x73, 0x2f, 0x67, 0x75, 0x69, 0x64, 0x65, 0x2e, 0x6d, 0x64, 0x50, 0x4b, 0x01,
    0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0xea, 0x59, 0x06,
    0x6c, 0x0d, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x5b, 0x00, 0x00, 0x00, 0x70, 0x6c, 0x61, 0x69, 0x6e,
    0x2e, 0x74, 0x78, 0x74, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00,
    0xa5, 0x00, 0x00, 0x00, 0x8f, 0x00, 0x00, 0x00, 0x00, 0x00,
];

#[test]
fn reads_an_archive_written_by_python_zipfile() {
    assert_eq!(PYTHON_ZIP.len(), 330);
    assert_eq!(detect(PYTHON_ZIP), Some(ArchiveFormat::Zip));

    let entries = ArchiveUnpacker::unpack_auto(PYTHON_ZIP).expect("reads a real archive");
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].path, "docs/");
    assert!(entries[0].is_dir);
    assert_eq!(entries[0].bytes, Vec::<u8>::new());
    assert_eq!(entries[0].mode, Some(0o600));
    assert_eq!(entries[1].path, "docs/guide.md");
    assert!(!entries[1].is_dir);
    assert_eq!(entries[1].bytes, b"# Guide\n".repeat(20));
    assert_eq!(entries[1].mode, Some(0o600));
    assert_eq!(entries[2].path, "plain.txt");
    assert_eq!(entries[2].bytes, b"stored bytes\n");
    assert_eq!(entries[2].mode, Some(0o600));
}

#[test]
fn zip64_structures_are_no_longer_refused() {
    // A locator that names an absent record is invalid input, not a deferred
    // capability.
    let mut bytes = zip64_locator(0, 1);
    bytes.extend_from_slice(&eocd(0, 0, 0));
    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("empty record");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(!error.to_string().contains("not implemented"), "{error}");
    assert_eq!(ZIP64_LOCATOR_FIXED, 20);
    assert_eq!(ZIP64_EOCD_FIXED, 56);
}
