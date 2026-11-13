//! Reader/writer tests for the in-crate `zip` implementation.
//!
//! Every fixture is assembled here from the byte layout itself (never with
//! [`ArchiveWriter`], except for the round-trip test), so the reader is checked
//! against the format rather than against its own writer. Nothing touches the
//! network, the environment, or the wall clock.

use super::*;
use crate::skill_creator::omni::archive::{
    detect, ArchiveFormat, ArchiveUnpacker, MAX_ENTRY_UNCOMPRESSED_BYTES,
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

/// Builds a central-directory record for `name` at local-header `offset`.
fn central_record(name: &[u8], fields: Fields, offset: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(CENTRAL_SIGNATURE);
    put_u16(&mut out, VERSION);
    put_u16(&mut out, VERSION);
    put_u16(&mut out, fields.flags);
    put_u16(&mut out, fields.method);
    put_u16(&mut out, DOS_TIME);
    put_u16(&mut out, DOS_DATE);
    put_u32(&mut out, fields.crc);
    put_u32(&mut out, fields.packed);
    put_u32(&mut out, fields.declared);
    put_u16(&mut out, name.len() as u16);
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u32(&mut out, fields.external);
    put_u32(&mut out, offset);
    out.extend_from_slice(name);
    out
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

#[test]
fn writer_round_trips_files_and_directories() {
    let entries = vec![
        ArchiveEntry {
            path: "readme.txt".to_owned(),
            bytes: b"hello zip\n".to_vec(),
            is_dir: false,
        },
        ArchiveEntry {
            path: "docs/".to_owned(),
            bytes: Vec::new(),
            is_dir: true,
        },
        ArchiveEntry {
            path: "café/naïve.txt".to_owned(),
            bytes: b"accented".to_vec(),
            is_dir: false,
        },
        ArchiveEntry {
            path: "docs/nested/deep.txt".to_owned(),
            bytes: b"deep".to_vec(),
            is_dir: false,
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
fn zip64_archives_are_unsupported() {
    // A ZIP64 end-of-central-directory locator directly before the end record.
    let mut bytes = ZIP64_LOCATOR_SIGNATURE.to_vec();
    bytes.extend_from_slice(&[0_u8; ZIP64_LOCATOR_FIXED - 4]);
    bytes.extend_from_slice(&eocd(0, 0, 0));
    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("locator");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("ZIP64"), "{error}");

    // A ZIP64 end-of-central-directory record directly before the end record.
    let mut bytes = ZIP64_EOCD_SIGNATURE.to_vec();
    bytes.extend_from_slice(&[0_u8; ZIP64_EOCD_FIXED - 4]);
    bytes.extend_from_slice(&eocd(0, 0, 0));
    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("record");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("ZIP64"), "{error}");

    // A 0xffffffff central-directory size, the ZIP64 escape value.
    let fields = Fields::stored(b"hello");
    let mut bytes = local_header(b"a.txt", fields, b"", None);
    bytes.extend_from_slice(b"hello");
    let cd_offset = bytes.len() as u32;
    bytes.extend_from_slice(&central_record(b"a.txt", fields, 0));
    bytes.extend_from_slice(&eocd(1, ZIP64_MARKER, cd_offset));
    let error = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &bytes).expect_err("size marker");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("ZIP64"), "{error}");
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
    }];
    let error = ArchiveWriter::write_stored(&directory_with_content).expect_err("directory bytes");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("directory"), "{error}");

    let file_with_slash = vec![ArchiveEntry {
        path: "docs/".to_owned(),
        bytes: Vec::new(),
        is_dir: false,
    }];
    let error = ArchiveWriter::write_stored(&file_with_slash).expect_err("trailing slash");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("ends with `/`"), "{error}");

    let unsafe_path = vec![ArchiveEntry {
        path: "../escape.txt".to_owned(),
        bytes: Vec::new(),
        is_dir: false,
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
    assert_eq!(entries[1].path, "docs/guide.md");
    assert!(!entries[1].is_dir);
    assert_eq!(entries[1].bytes, b"# Guide\n".repeat(20));
    assert_eq!(entries[2].path, "plain.txt");
    assert_eq!(entries[2].bytes, b"stored bytes\n");
}
