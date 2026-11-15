//! Unit tests for the traditional PKWARE (`ZipCrypto`) primitives.
//!
//! The archive-level behaviour (an encrypted entry inside a whole `zip`) is
//! covered by [`super::super::tests`], which can assemble archives; these tests
//! pin the key schedule, the password check byte, and the failure modes of
//! [`decrypt`] directly against hand-built payloads.

use super::*;
use crate::skill_creator::omni::archive::crc::crc32;

#[test]
fn crc32_update_agrees_with_the_shared_table_crc() {
    // Folding a buffer through the single-byte step and inverting the result is
    // the table implementation in `archive::crc`, so the key schedule and the
    // entry CRC check cannot drift apart.
    for buffer in [
        &b""[..],
        &b"a"[..],
        &b"123456789"[..],
        &b"The quick brown fox jumps over the lazy dog"[..],
        &[0x00, 0xFF, 0x80, 0x7F][..],
    ] {
        let mut state = 0xFFFF_FFFFu32;
        for &byte in buffer {
            state = crc32_update(state, byte);
        }
        assert_eq!(!state, crc32(buffer), "buffer {buffer:02x?}");
    }
}

#[test]
fn the_keys_start_at_the_published_values_and_fold_the_password() {
    // An empty password is the scheme's identity: the keys keep their seeds.
    let empty = Keys::new(b"");
    assert_eq!(empty.key0, KEY0_INIT);
    assert_eq!(empty.key1, KEY1_INIT);
    assert_eq!(empty.key2, KEY2_INIT);

    // A password moves every key, and the fold is order-sensitive.
    let keys = Keys::new(b"hunter2");
    assert_ne!(keys.key0, KEY0_INIT);
    assert_ne!(keys.key2, KEY2_INIT);
    assert_ne!((keys.key0, keys.key1, keys.key2), {
        let other = Keys::new(b"2retnuh");
        (other.key0, other.key1, other.key2)
    });

    // `key1` is a linear congruential step over `key0`'s low byte, so folding the
    // same byte twice cannot leave the keys unchanged.
    let mut updated = empty;
    updated.update(b'x');
    assert_ne!(updated.key0, empty.key0);
    assert_ne!(updated.key1, empty.key1);
    assert_ne!(updated.key2, empty.key2);
}

#[test]
fn encryption_round_trips_through_decryption() {
    for check in [CheckByte::Crc32(0xCBF4_3926), CheckByte::DosTime(0x180F)] {
        let plaintext = b"the payload the writer encrypted";
        let payload = encrypt("hunter2", plaintext, check);
        assert_eq!(payload.len(), plaintext.len() + HEADER_LEN);
        let decrypted = decrypt(&payload, "hunter2", check, "secret.txt").expect("round trip");
        assert_eq!(decrypted, plaintext);
    }
}

#[test]
fn an_empty_payload_round_trips_to_an_empty_payload() {
    let check = CheckByte::Crc32(0);
    let payload = encrypt("hunter2", b"", check);
    assert_eq!(payload.len(), HEADER_LEN);
    assert_eq!(
        decrypt(&payload, "hunter2", check, "empty.txt").expect("header only"),
        b""
    );
}

#[test]
fn a_wrong_password_is_a_validation_error_naming_the_entry() {
    let check = CheckByte::Crc32(0x1234_5678);
    let payload = encrypt("hunter2", b"payload", check);
    let error = decrypt(&payload, "hunter3", check, "secret.txt").expect_err("wrong password");
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    let message = error.to_string();
    assert!(message.contains("secret.txt"), "{message}");
    assert!(message.contains("password"), "{message}");
    assert!(message.contains("does not decrypt"), "{message}");
    // The message names both one-byte values, so a caller can tell a wrong
    // password from a corrupt header.
    assert!(message.contains("0x12"), "{message}");
}

#[test]
fn the_check_byte_source_matters() {
    let time = CheckByte::DosTime(0x180F);
    let payload = encrypt("hunter2", b"payload", time);
    // A DOS time of `0x180F` checks `0x18`, so the same header also matches a
    // CRC-32 whose high byte is `0x18` — only the byte itself is compared.
    assert!(decrypt(&payload, "hunter2", time, "secret.txt").is_ok());
    assert!(decrypt(
        &payload,
        "hunter2",
        CheckByte::Crc32(0x1800_0000),
        "secret.txt"
    )
    .is_ok());
    let error = decrypt(
        &payload,
        "hunter2",
        CheckByte::Crc32(0xFF00_0000),
        "secret.txt",
    )
    .expect_err("a different high byte must fail");
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("0xff"), "{error}");
}

#[test]
fn payloads_shorter_than_the_header_are_rejected() {
    for length in [0, 1, 11] {
        let payload = vec![0x5a; length];
        let error = decrypt(&payload, "hunter2", CheckByte::Crc32(0), "short.txt")
            .expect_err("no room for the encryption header");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        let message = error.to_string();
        assert!(message.contains("short.txt"), "{message}");
        assert!(message.contains("encryption header"), "{message}");
    }
}

#[test]
fn the_aes_extra_field_is_detected_by_its_header_id() {
    /// Builds `(id, payload)` tuples as a `zip` extra block.
    fn extra(fields: &[(u16, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (id, data) in fields {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(data.len() as u16).to_le_bytes());
            out.extend_from_slice(data);
        }
        out
    }

    let unrelated = extra(&[(0x5455, &[0x03, 0x00, 0x00, 0x00, 0x00])]);
    assert!(!has_aes_extra(&unrelated, 0, unrelated.len()));

    let aes = extra(&[(0x5455, &[0x03]), (0x9901, &[0x02, 0x00, b'A', b'E', 0x03])]);
    assert!(has_aes_extra(&aes, 0, aes.len()));
    // A leading timestamp field is walked over, so the marker is still found.
    assert!(has_aes_extra(&aes, 0, aes.len() - 1));

    // A block that ends before the AES field's header, an offset past the input,
    // a trailing fragment shorter than a header, and address overflow all just
    // end the scan instead of failing.
    assert!(!has_aes_extra(&aes, 0, 5));
    assert!(!has_aes_extra(&aes, aes.len() + 8, aes.len()));
    assert!(!has_aes_extra(&[0x01, 0x99], 0, 2));
    assert!(!has_aes_extra(&aes, usize::MAX - 1, 8));
    // A declared field longer than the block hides nothing: the walk stops.
    let declared_too_long = extra(&[(0x5455, &[0x01])]);
    assert!(!has_aes_extra(
        &declared_too_long,
        0,
        declared_too_long.len()
    ));
    // A declared block past the input is not a scan either.
    assert!(!has_aes_extra(
        &declared_too_long,
        0,
        declared_too_long.len() + 1
    ));
}
