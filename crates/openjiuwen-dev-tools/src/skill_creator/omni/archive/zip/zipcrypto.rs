//! Traditional PKWARE encryption (`ZipCrypto`) for `zip` entries.
//!
//! `zip` knows three password schemes, and this module owns exactly one of them:
//! the traditional PKWARE encryption that general-purpose flag bit 0 announces
//! (PKWARE APPNOTE 6.1 §6.1, also called `ZipCrypto`). `WinZip` AES (compression
//! method `99` plus the `0x9901` extra field), PKWARE strong encryption (flag bit
//! 6), and encrypted central directories (flag bit 13) stay typed
//! [`Error::unsupported`](openjiuwen_common::error::Error::unsupported) in
//! [`super`], because each of them needs a real cipher this crate does not ship.
//!
//! A `ZipCrypto` payload is a 12-byte encryption header followed by the
//! compressed data, both mixed with a keystream by XOR. The keystream comes from three
//! 32-bit keys that the password seeds with the fixed values `0x12345678`,
//! `0x23456789`, and `0x34567890`. Every recovered byte then updates the keys:
//!
//! - `key0` is the running CRC-32 of the plaintext, one byte at a time;
//! - `key1` is `(key1 + (key0 & 0xFF)) * 134775813 + 1`, a 32-bit LCG;
//! - `key2` is a running CRC-32 of `key1`'s high byte.
//!
//! The next keystream byte is `((key2 | 2) * ((key2 | 2) ^ 1) >> 8) & 0xFF`, where
//! the `key2 | 2` value is truncated to 16 bits first, exactly as the reference
//! implementation's `unsigned short` temporary does.
//!
//! The last byte of the encryption header is a one-byte password check: byte 11
//! of the decrypted header must equal the high byte of the entry's CRC-32, or —
//! when the entry carries a data descriptor (flag bit 3), which means the local
//! header's CRC field is zero — the high byte of its DOS modification time. A
//! mismatch is a [`Error::validation`] naming the entry and the password, so a
//! wrong password never reaches the decompressor and never returns corrupt data.
//!
//! No CRC table is duplicated here: [`crc32_update`] is the same single-byte
//! reflected CRC-32 step the shared table in [`super::super::crc`] encodes,
//! computed arithmetically, and `Some` vector tests pin both to each other.

use openjiuwen_common::error::{Error, Result};

/// Bytes of the encryption header that precede every `ZipCrypto` payload.
pub(crate) const HEADER_LEN: usize = 12;

/// Header id of the `WinZip` AES extra field, which marks a non-`ZipCrypto` entry.
const AES_EXTRA_ID: u16 = 0x9901;

/// The reflected IEEE CRC-32 polynomial, the same one the shared table uses.
const POLYNOMIAL: u32 = 0xEDB8_8320;

/// Initial value of the first key, before the password is folded in.
const KEY0_INIT: u32 = 0x1234_5678;

/// Initial value of the second key, before the password is folded in.
const KEY1_INIT: u32 = 0x2345_6789;

/// Initial value of the third key, before the password is folded in.
const KEY2_INIT: u32 = 0x3456_7890;

/// The multiplier of the second key's linear congruential update.
const KEY1_MULTIPLIER: u32 = 134_775_813;

/// Whether the extra data of a central-directory record carries the `WinZip` AES
/// extra field (header id `0x9901`).
///
/// The walk is deliberately tolerant: an extra block that runs past the input,
/// an extra field that overruns the declared block, and a trailing fragment
/// shorter than the four-byte field header all simply end the scan. The AES
/// marker can only *add* a typed refusal, so it must never turn an otherwise
/// readable archive into a validation error.
pub(crate) fn has_aes_extra(bytes: &[u8], extra_offset: usize, extra_len: usize) -> bool {
    let end = match extra_offset.checked_add(extra_len) {
        Some(end) => end,
        None => return false,
    };
    let extra = match bytes.get(extra_offset..end) {
        Some(extra) => extra,
        None => return false,
    };
    let mut field = 0usize;
    while let Some(header) = field.checked_add(4).and_then(|end| extra.get(field..end)) {
        let id = u16::from_le_bytes([header[0], header[1]]);
        let size = usize::from(u16::from_le_bytes([header[2], header[3]]));
        if id == AES_EXTRA_ID {
            return true;
        }
        field = match field
            .checked_add(4)
            .and_then(|start| start.checked_add(size))
        {
            Some(next) => next,
            None => return false,
        };
    }
    false
}

/// The single-byte CRC-32 step the `ZipCrypto` key schedule needs.
///
/// `crc` is the running (non-inverted) CRC-32 state and `byte` the next plaintext
/// byte; folding a whole buffer through this step and inverting the result is
/// exactly [`super::super::crc::crc32`], which the module's tests assert.
pub(crate) fn crc32_update(crc: u32, byte: u8) -> u32 {
    let mut entry = (crc ^ u32::from(byte)) & 0xFF;
    // Eight reflected bit steps, the arithmetic the table in `crc` precomputes.
    let mut bit = 0;
    while bit < 8 {
        entry = if entry & 1 == 1 {
            (entry >> 1) ^ POLYNOMIAL
        } else {
            entry >> 1
        };
        bit += 1;
    }
    (crc >> 8) ^ entry
}

/// The three `ZipCrypto` keys, seeded from the password and updated per byte.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Keys {
    /// The running CRC-32 of the plaintext.
    key0: u32,
    /// The linear congruential key.
    key1: u32,
    /// The running CRC-32 of `key1`'s high byte.
    key2: u32,
}

impl Keys {
    /// Seeds the three keys from `password`'s bytes.
    ///
    /// An empty password is the documented identity of the scheme — the keys keep
    /// their initial values — which is why a caller wanting a password-protected
    /// entry must supply a non-empty one; [`super::super::UnpackOptions::validate`]
    /// rejects an empty password before decryption starts.
    pub(crate) fn new(password: &[u8]) -> Self {
        let mut keys = Self {
            key0: KEY0_INIT,
            key1: KEY1_INIT,
            key2: KEY2_INIT,
        };
        for &byte in password {
            keys.update(byte);
        }
        keys
    }

    /// The next keystream byte, the value that turns a ciphertext byte into plaintext.
    fn stream_byte(&self) -> u8 {
        // The reference implementation truncates to `unsigned short` first.
        let temp = (self.key2 | 2) & 0xFFFF;
        (temp.wrapping_mul(temp ^ 1) >> 8) as u8
    }

    /// Folds one plaintext byte into the key schedule.
    pub(crate) fn update(&mut self, plain: u8) {
        self.key0 = crc32_update(self.key0, plain);
        self.key1 = self
            .key1
            .wrapping_add(self.key0 & 0xFF)
            .wrapping_mul(KEY1_MULTIPLIER)
            .wrapping_add(1);
        self.key2 = crc32_update(self.key2, (self.key1 >> 24) as u8);
    }

    /// Decrypts one ciphertext byte and folds the recovered plaintext into the
    /// key schedule.
    fn decrypt_byte(&mut self, cipher: u8) -> u8 {
        let plain = cipher ^ self.stream_byte();
        self.update(plain);
        plain
    }
}

/// The value the encryption header's last byte is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CheckByte {
    /// The high byte of the entry's CRC-32, used by a normal entry.
    Crc32(u32),
    /// The high byte of the entry's DOS modification time, used when the entry
    /// carries a data descriptor (general-purpose flag bit 3) and its local
    /// header CRC field is therefore zero.
    DosTime(u16),
}

impl CheckByte {
    /// The one-byte password check a decrypted header must end with.
    const fn value(self) -> u8 {
        match self {
            Self::Crc32(crc) => (crc >> 24) as u8,
            Self::DosTime(time) => (time >> 8) as u8,
        }
    }
}

/// Decrypts a `ZipCrypto` payload, verifying the header's password check byte.
///
/// `payload` is the whole encrypted region the central directory describes: the
/// 12-byte encryption header followed by the ciphertext of the compressed (or
/// stored) data. The returned vector is the decrypted remainder, so the caller
/// decompresses exactly what the header preceded.
///
/// # Errors
///
/// Returns [`Error::validation`] naming `path` when the payload is shorter than
/// the 12-byte encryption header, or when the decrypted header's last byte does
/// not match `check` — the signature of a wrong password.
pub(crate) fn decrypt(
    payload: &[u8],
    password: &str,
    check: CheckByte,
    path: &str,
) -> Result<Vec<u8>> {
    let header = payload.get(..HEADER_LEN).ok_or_else(|| {
        Error::validation(format!(
            "truncated encrypted zip entry {path:?}: the payload holds {} bytes, fewer than the \
             {HEADER_LEN}-byte ZipCrypto encryption header it declares",
            payload.len()
        ))
    })?;
    let cipher = payload.get(HEADER_LEN..).unwrap_or_default();
    let mut keys = Keys::new(password.as_bytes());
    let mut last = 0_u8;
    for &byte in header {
        last = keys.decrypt_byte(byte);
    }
    let expected = check.value();
    if last != expected {
        return Err(Error::validation(format!(
            "zip entry {path:?} does not decrypt with the supplied password: the ZipCrypto \
             encryption header ends with 0x{last:02x} but the entry requires 0x{expected:02x}"
        )));
    }
    Ok(cipher.iter().map(|&byte| keys.decrypt_byte(byte)).collect())
}

/// Encrypts `plaintext` the way a `ZipCrypto` writer does, for tests only.
///
/// The 11 random header bytes are a fixed pattern, so a fixture assembled from
/// this helper is byte-for-byte reproducible; only the caller-supplied payload
/// and `check` vary. The output is the 12-byte header (the check byte included)
/// followed by the ciphertext, the exact inverse of [`decrypt`].
#[cfg(test)]
pub(crate) fn encrypt(password: &str, plaintext: &[u8], check: CheckByte) -> Vec<u8> {
    /// The deterministic stand-in for the 11 random header bytes.
    const HEADER_PATTERN: [u8; HEADER_LEN - 1] = [
        0x5a, 0x17, 0xc3, 0x40, 0x9e, 0x2b, 0x71, 0x8c, 0x36, 0xd4, 0x0f,
    ];
    let mut keys = Keys::new(password.as_bytes());
    let mut out = Vec::with_capacity(HEADER_LEN + plaintext.len());
    for &byte in HEADER_PATTERN.iter().chain([check.value()].iter()) {
        out.push(byte ^ keys.stream_byte());
        keys.update(byte);
    }
    for &byte in plaintext {
        out.push(byte ^ keys.stream_byte());
        keys.update(byte);
    }
    out
}

#[cfg(test)]
mod tests;
