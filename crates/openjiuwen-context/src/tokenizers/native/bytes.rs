//! The GPT-2/Llama byte↔unicode alphabet used by byte-level vocabularies.
//!
//! JSON cannot hold raw bytes, so a byte-level vocabulary stores every byte as a printable scalar.
//! The table here is the one `HuggingFace` and `tiktoken` share, reproduced locally so no dependency is needed.
//! Printable ASCII and most of Latin-1 map to themselves; the remaining 68 bytes map to `U+0100..U+0143`.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The byte↔unicode alphabet.
struct Alphabet {
    /// The scalar a byte-level vocabulary stores for each byte value.
    to_char: [char; 256],
    /// The byte value of every mapped scalar.
    to_byte: HashMap<char, u8>,
}

impl Alphabet {
    /// Builds the alphabet from the safe printable ranges.
    fn new() -> Self {
        let mut to_char = [char::from(0_u8); 256];
        let mut safe = [false; 256];
        let mut push = |byte: u8, to_char: &mut [char; 256]| {
            to_char[usize::from(byte)] = char::from(byte);
            safe[usize::from(byte)] = true;
        };
        for byte in b'!'..=b'~' {
            push(byte, &mut to_char);
        }
        for byte in 0xA1..=0xAC {
            push(byte, &mut to_char);
        }
        for byte in 0xAE..=0xFF {
            push(byte, &mut to_char);
        }
        let mut next = 256_u32;
        for value in 0..256_u32 {
            let byte = u8::try_from(value).unwrap_or(0);
            if safe[usize::from(byte)] {
                continue;
            }
            to_char[usize::from(byte)] = char::from_u32(next).unwrap_or('\u{fffd}');
            next += 1;
        }
        let to_byte = to_char
            .iter()
            .enumerate()
            .map(|(value, ch)| (*ch, u8::try_from(value).unwrap_or(0)))
            .collect();
        Self { to_char, to_byte }
    }
}

/// The shared alphabet, built on first use.
static ALPHABET: OnceLock<Alphabet> = OnceLock::new();

/// Returns the shared alphabet.
fn alphabet() -> &'static Alphabet {
    ALPHABET.get_or_init(Alphabet::new)
}

/// The scalar a byte-level vocabulary stores for `byte`.
pub(crate) fn byte_to_char(byte: u8) -> char {
    alphabet().to_char[usize::from(byte)]
}

/// The byte value of a byte-level scalar, when the scalar belongs to the alphabet.
pub(crate) fn char_to_byte(ch: char) -> Option<u8> {
    alphabet().to_byte.get(&ch).copied()
}

/// Rewrites `text` as the printable scalars of its UTF-8 bytes.
pub(crate) fn encode(text: &str) -> String {
    text.bytes().map(byte_to_char).collect()
}

/// Recovers the bytes a vocabulary token stands for.
///
/// A scalar outside the alphabet cannot come from the byte mapping, so it contributes its own UTF-8 bytes.
pub(crate) fn token_bytes(token: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(token.len());
    let mut buffer = [0_u8; 4];
    for ch in token.chars() {
        match char_to_byte(ch) {
            Some(byte) => bytes.push(byte),
            None => bytes.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes()),
        }
    }
    bytes
}

/// The full table, for tooling that builds byte-level vocabularies.
pub(crate) fn table() -> &'static [char; 256] {
    &alphabet().to_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alphabet_covers_every_byte_once() {
        let mut seen = std::collections::HashSet::new();
        for byte in 0..=u8::MAX {
            let ch = byte_to_char(byte);
            assert!(seen.insert(ch), "byte {byte} maps to a scalar already used");
            assert_eq!(char_to_byte(ch), Some(byte));
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        assert_eq!(byte_to_char(b'a'), 'a');
        assert_eq!(byte_to_char(b'!'), '!');
        assert_eq!(byte_to_char(b' '), '\u{0120}');
    }

    #[test]
    fn encode_and_decode_round_trip_multibyte_text() {
        for text in ["hello", "héllo", "😀", "中文", "\u{0}\u{7f}"] {
            let mapped = encode(text);
            assert!(mapped.chars().all(|ch| char_to_byte(ch).is_some()));
            assert_eq!(String::from_utf8_lossy(&token_bytes(&mapped)), text);
        }
    }

    #[test]
    fn token_bytes_falls_back_to_utf8() {
        // `é` is itself a byte-alphabet scalar (byte 0xE9); `中` is not.
        assert_eq!(token_bytes("é"), vec![0xE9]);
        assert_eq!(token_bytes("中"), "中".as_bytes());
        assert_eq!(token_bytes("\u{0120}"), b" ");
    }
}
