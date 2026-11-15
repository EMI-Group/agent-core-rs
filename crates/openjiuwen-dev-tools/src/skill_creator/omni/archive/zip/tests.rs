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
use crate::skill_creator::omni::archive::crc::crc32;
use crate::skill_creator::omni::archive::{
    detect, ArchiveEntryKind, ArchiveFormat, ArchiveUnpacker, UnpackOptions,
    MAX_ENTRY_UNCOMPRESSED_BYTES,
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
    /// DOS modification time, written by both headers.
    time: u16,
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
            time: DOS_TIME,
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
    put_u16(&mut out, fields.time);
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
    put_u16(&mut out, fields.time);
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

/// Fields of a `ZipCrypto` entry whose plaintext is `plain` and whose encrypted
/// payload is `cipher` bytes long.
fn encrypted_fields(plain: &[u8], cipher: &[u8], method: u16) -> Fields {
    Fields {
        flags: FLAG_ENCRYPTED,
        method,
        packed: cipher.len() as u32,
        ..Fields::stored(plain)
    }
}

#[test]
fn zipcrypto_stored_entry_decrypts_with_the_password() {
    let plain: &[u8] = b"top secret payload\n";
    let cipher = zipcrypto::encrypt("hunter2", plain, zipcrypto::CheckByte::Crc32(crc32(plain)));
    let archive = single(Spec {
        name: b"secret.txt",
        fields: encrypted_fields(plain, &cipher, METHOD_STORED),
        extra: b"",
        local_sizes: None,
        payload: &cipher,
    });

    let options = UnpackOptions::new().with_password("hunter2");
    let entries = ArchiveUnpacker::unpack_with_options(ArchiveFormat::Zip, &archive, &options)
        .expect("the stored payload decrypts");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "secret.txt");
    assert_eq!(entries[0].bytes, plain);
    assert!(!entries[0].is_dir);

    // Detection plus the options reaches exactly the same entry.
    let auto = ArchiveUnpacker::unpack_auto_with_options(&archive, &options).expect("sniffs");
    assert_eq!(auto, entries);
}

#[test]
fn zipcrypto_deflate_entry_decrypts_before_inflating() {
    // BFINAL = 1, BTYPE = 00 (stored block), LEN = 5, NLEN = !5, then "hello".
    let deflated: &[u8] = &[0x01, 0x05, 0x00, 0xFA, 0xFF, b'h', b'e', b'l', b'l', b'o'];
    let cipher = zipcrypto::encrypt(
        "hunter2",
        deflated,
        zipcrypto::CheckByte::Crc32(crc32(b"hello")),
    );
    let archive = single(Spec {
        name: b"deflated-secret.txt",
        fields: encrypted_fields(b"hello", &cipher, METHOD_DEFLATE),
        extra: b"",
        local_sizes: None,
        payload: &cipher,
    });

    let entries = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &archive,
        &UnpackOptions::new().with_password("hunter2"),
    )
    .expect("decrypts then inflates");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "deflated-secret.txt");
    assert_eq!(entries[0].bytes, b"hello");
}

#[test]
fn a_wrong_password_is_a_validation_error() {
    let plain: &[u8] = b"guess me if you can\n";
    let cipher = zipcrypto::encrypt("hunter2", plain, zipcrypto::CheckByte::Crc32(crc32(plain)));
    let archive = single(Spec {
        name: b"secret.txt",
        fields: encrypted_fields(plain, &cipher, METHOD_STORED),
        extra: b"",
        local_sizes: None,
        payload: &cipher,
    });

    let error = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &archive,
        &UnpackOptions::new().with_password("hunter3"),
    )
    .expect_err("a wrong password never yields data");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("secret.txt"), "{message}");
    assert!(message.contains("password"), "{message}");
    assert!(message.contains("does not decrypt"), "{message}");
}

#[test]
fn an_encrypted_entry_needs_the_option_with_password() {
    let plain: &[u8] = b"locked away\n";
    let cipher = zipcrypto::encrypt("hunter2", plain, zipcrypto::CheckByte::Crc32(crc32(plain)));
    let archive = single(Spec {
        name: b"secret.txt",
        fields: encrypted_fields(plain, &cipher, METHOD_STORED),
        extra: b"",
        local_sizes: None,
        payload: &cipher,
    });

    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("no password");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("secret.txt"), "{message}");
    assert!(message.contains("encrypted"), "{message}");
    assert!(
        message.contains("UnpackOptions::with_password"),
        "the error tells the caller how to supply the password: {message}"
    );

    // `unpack_auto` reaches the same typed answer, and the options API reads it.
    let error = ArchiveUnpacker::unpack_auto(&archive).expect_err("no password");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let entries = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &archive,
        &UnpackOptions::new().with_password("hunter2"),
    )
    .expect("the password decrypts the entry");
    assert_eq!(entries[0].bytes, plain);

    // An empty password is refused before decryption starts.
    let error = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &archive,
        &UnpackOptions::new().with_password(""),
    )
    .expect_err("an empty password is a validation error");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("empty"), "{error}");
}

#[test]
fn a_data_descriptor_entry_checks_the_dos_time() {
    /// A DOS time of 12:00:00, whose high byte is `0x18`.
    const TIME: u16 = 0x180F;
    let plain: &[u8] = b"streamed secret payload";
    assert_ne!(
        crc32(plain) >> 24,
        u32::from(TIME >> 8),
        "the fixture only proves the time is used when the two check bytes differ"
    );
    let cipher = zipcrypto::encrypt("hunter2", plain, zipcrypto::CheckByte::DosTime(TIME));
    let fields = Fields {
        flags: FLAG_ENCRYPTED | FLAG_DATA_DESCRIPTOR,
        time: TIME,
        ..encrypted_fields(plain, &cipher, METHOD_STORED)
    };
    let archive = single(Spec {
        name: b"streamed-secret.txt",
        fields,
        extra: b"",
        // A streamed entry keeps its local sizes zero and writes a data
        // descriptor, so the central directory stays authoritative.
        local_sizes: Some((0, 0)),
        payload: &cipher,
    });

    let entries = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &archive,
        &UnpackOptions::new().with_password("hunter2"),
    )
    .expect("the DOS time is the check byte");
    assert_eq!(entries[0].bytes, plain);

    // Encrypting the same payload against the CRC-32 proves the flag decides the
    // source: the DOS-time check rejects that header.
    let wrong_source =
        zipcrypto::encrypt("hunter2", plain, zipcrypto::CheckByte::Crc32(crc32(plain)));
    let archive = single(Spec {
        name: b"streamed-secret.txt",
        fields: Fields {
            flags: FLAG_ENCRYPTED | FLAG_DATA_DESCRIPTOR,
            time: TIME,
            ..encrypted_fields(plain, &wrong_source, METHOD_STORED)
        },
        extra: b"",
        local_sizes: Some((0, 0)),
        payload: &wrong_source,
    });
    let error = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &archive,
        &UnpackOptions::new().with_password("hunter2"),
    )
    .expect_err("a data-descriptor entry checks the DOS time, not the CRC-32");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn a_truncated_encryption_header_is_rejected() {
    let plain: &[u8] = b"eighteen bytes here";
    let cipher = zipcrypto::encrypt("hunter2", plain, zipcrypto::CheckByte::Crc32(crc32(plain)));
    let truncated = &cipher[..7];
    let archive = single(Spec {
        name: b"short-header.txt",
        fields: encrypted_fields(plain, truncated, METHOD_STORED),
        extra: b"",
        local_sizes: None,
        payload: truncated,
    });

    let error = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &archive,
        &UnpackOptions::new().with_password("hunter2"),
    )
    .expect_err("7 bytes cannot hold the 12-byte header");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("encryption header"), "{error}");
}

#[test]
fn a_mixed_archive_keeps_central_directory_order() {
    let secret: &[u8] = b"encrypted second entry";
    let cipher = zipcrypto::encrypt(
        "hunter2",
        secret,
        zipcrypto::CheckByte::Crc32(crc32(secret)),
    );
    let archive = assemble(&[
        Spec {
            name: b"secret.txt",
            fields: encrypted_fields(secret, &cipher, METHOD_STORED),
            extra: b"",
            local_sizes: None,
            payload: &cipher,
        },
        Spec::stored(b"plain.txt", b"plain first entry"),
    ]);

    let entries = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &archive,
        &UnpackOptions::new().with_password("hunter2"),
    )
    .expect("both entries read");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].path, "secret.txt");
    assert_eq!(entries[0].bytes, secret);
    assert_eq!(entries[1].path, "plain.txt");
    assert_eq!(entries[1].bytes, b"plain first entry");

    // Without the password the encrypted entry fails even though the plain one
    // would read, because the archive is unpacked as a unit.
    let error =
        ArchiveUnpacker::unpack(ArchiveFormat::Zip, &archive).expect_err("needs the password");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("secret.txt"), "{error}");
}

#[test]
fn winzip_aes_encryption_is_unsupported() {
    let payload: &[u8] = b"0123456789abcdef";
    let archive = single(Spec {
        name: b"aes.txt",
        fields: Fields {
            flags: FLAG_ENCRYPTED,
            method: METHOD_AES,
            ..Fields::stored(payload)
        },
        extra: b"",
        local_sizes: None,
        payload,
    });

    for options in [
        UnpackOptions::new(),
        UnpackOptions::new().with_password("hunter2"),
    ] {
        let error = ArchiveUnpacker::unpack_with_options(ArchiveFormat::Zip, &archive, &options)
            .expect_err("method 99");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(message.contains("WinZip AES"), "{message}");
        assert!(message.contains("aes.txt"), "{message}");
    }
}

#[test]
fn the_winzip_aes_extra_field_is_unsupported() {
    // A `0x9901` extra field with the `WinZip` AES payload: version 2, vendor `AE`,
    // AES-256, and the real method (stored, 0).
    let mut extra = Vec::new();
    put_u16(&mut extra, 0x9901);
    put_u16(&mut extra, 7);
    extra.extend_from_slice(&[0x02, 0x00, b'A', b'E', 0x03, 0x00, 0x00]);
    let payload: &[u8] = b"0123456789abcdef";
    let fields = Fields {
        flags: FLAG_ENCRYPTED,
        ..Fields::stored(payload)
    };

    let mut bytes = local_header(b"aes-extra.txt", fields, &extra, None);
    bytes.extend_from_slice(payload);
    let cd_offset = bytes.len();
    let values = CentralValues {
        packed: fields.packed,
        declared: fields.declared,
        offset: 0,
    };
    let record = central_record_parts(b"aes-extra.txt", fields, &extra, values);
    let cd_size = record.len();
    bytes.extend_from_slice(&record);
    bytes.extend_from_slice(&eocd(1, cd_size as u32, cd_offset as u32));

    let error = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        &bytes,
        &UnpackOptions::new().with_password("hunter2"),
    )
    .expect_err("the AES extra field marks a scheme this reader cannot decrypt");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("WinZip AES"), "{message}");
    assert!(message.contains("0x9901"), "{message}");
    assert!(message.contains("aes-extra.txt"), "{message}");
}

#[test]
fn other_encryption_schemes_are_unsupported() {
    for (flag, needle) in [
        (FLAG_STRONG_ENCRYPTION, "strong encryption"),
        (FLAG_MASKED_LOCAL_HEADERS, "encrypted central directory"),
    ] {
        let payload: &[u8] = b"0123456789abcdef";
        let archive = single(Spec {
            name: b"strong.txt",
            fields: Fields {
                flags: FLAG_ENCRYPTED | flag,
                ..Fields::stored(payload)
            },
            extra: b"",
            local_sizes: None,
            payload,
        });
        let error = ArchiveUnpacker::unpack_with_options(
            ArchiveFormat::Zip,
            &archive,
            &UnpackOptions::new().with_password("hunter2"),
        )
        .expect_err(needle);
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{needle}");
        let message = error.to_string();
        assert!(message.contains(needle), "{message}");
        assert!(message.contains("strong.txt"), "{message}");
    }
}

/// A real Info-ZIP encrypted archive, produced by
/// `zip -e -P hunter2 deflated.zip secret.txt` for a 640-byte `secret.txt`.
///
/// Its entry sets general-purpose flags `0x0009` (encrypted plus a data
/// descriptor) and uses `DEFLATE` (method 8), so it pins interoperability with a
/// writer this crate does not share code with: a real DOS-time password check
/// byte and a real `DEFLATE` stream inside a real `ZipCrypto` layer.
const INFO_ZIP_ENCRYPTED: &[u8] = &[
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x09, 0x00, 0x08, 0x00, 0x17, 0x18, 0x70, 0x5d, 0x43, 0x1f,
    0xac, 0xe4, 0x25, 0x00, 0x00, 0x00, 0x80, 0x02, 0x00, 0x00, 0x0a, 0x00, 0x1c, 0x00, 0x73, 0x65,
    0x63, 0x72, 0x65, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x55, 0x54, 0x09, 0x00, 0x03, 0x5d, 0x01, 0xfa,
    0x6a, 0x5d, 0x01, 0xfa, 0x6a, 0x75, 0x78, 0x0b, 0x00, 0x01, 0x04, 0xe8, 0x03, 0x00, 0x00, 0x04,
    0x64, 0x00, 0x00, 0x00, 0x7d, 0x0f, 0x91, 0xae, 0xbf, 0x96, 0xbf, 0xa0, 0x07, 0x96, 0x77, 0xe9,
    0xdf, 0x1c, 0x8e, 0x3d, 0xdd, 0x65, 0xa0, 0xba, 0x34, 0x63, 0x59, 0x9d, 0x22, 0x08, 0xaf, 0x35,
    0x41, 0xb2, 0x3c, 0x0d, 0x9c, 0xe3, 0xfc, 0x88, 0x5c, 0x50, 0x4b, 0x07, 0x08, 0x43, 0x1f, 0xac,
    0xe4, 0x25, 0x00, 0x00, 0x00, 0x80, 0x02, 0x00, 0x00, 0x50, 0x4b, 0x01, 0x02, 0x1e, 0x03, 0x14,
    0x00, 0x09, 0x00, 0x08, 0x00, 0x17, 0x18, 0x70, 0x5d, 0x43, 0x1f, 0xac, 0xe4, 0x25, 0x00, 0x00,
    0x00, 0x80, 0x02, 0x00, 0x00, 0x0a, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0xa4, 0x81, 0x00, 0x00, 0x00, 0x00, 0x73, 0x65, 0x63, 0x72, 0x65, 0x74, 0x2e, 0x74, 0x78,
    0x74, 0x55, 0x54, 0x05, 0x00, 0x03, 0x5d, 0x01, 0xfa, 0x6a, 0x75, 0x78, 0x0b, 0x00, 0x01, 0x04,
    0xe8, 0x03, 0x00, 0x00, 0x04, 0x64, 0x00, 0x00, 0x00, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x01, 0x00, 0x50, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00,
];

#[test]
fn reads_an_encrypted_archive_written_by_info_zip() {
    let expected = "hello zipcrypto\n".repeat(40);
    let entries = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        INFO_ZIP_ENCRYPTED,
        &UnpackOptions::new().with_password("hunter2"),
    )
    .expect("the Info-ZIP fixture decrypts");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "secret.txt");
    assert_eq!(entries[0].bytes, expected.as_bytes());

    // Without a password the entry asks for one; a wrong one fails the check byte.
    let error =
        ArchiveUnpacker::unpack(ArchiveFormat::Zip, INFO_ZIP_ENCRYPTED).expect_err("no password");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Zip,
        INFO_ZIP_ENCRYPTED,
        &UnpackOptions::new().with_password("hunter3"),
    )
    .expect_err("wrong password");
    assert_eq!(error.kind(), ErrorKind::Validation);
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
