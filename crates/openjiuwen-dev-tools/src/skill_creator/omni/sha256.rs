//! A dependency-free SHA-256 implementation (FIPS 180-4).
//!
//! The omni download stage deduplicates image payloads the way Python's
//! `stage_02_download.py` does: `hashlib.sha256(data).hexdigest()` becomes the
//! identity of one downloaded payload, so two URLs serving the same bytes
//! collapse into a single asset. That is the only use here — this is not a
//! general-purpose crypto module and it deliberately does not authenticate
//! anything, it only fingerprints payloads.
//!
//! The implementation is the plain Merkle–Damgård/FIPS 180-4 construction with
//! the standard 64 round constants, big-endian word packing, a single `0x80`
//! padding byte, zero fill to the 56-byte mark and an eight-byte big-endian bit
//! length. It never panics and never allocates beyond the padded copy of the
//! input, so it stays usable on arbitrary scraped bytes.
/// The number of bytes in one SHA-256 compression block.
const BLOCK_BYTES: usize = 64;
/// The eight initial hash values (FIPS 180-4 §5.3.3).
const INITIAL_STATE: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];
/// The sixty-four round constants (FIPS 180-4 §4.2.2), the first 32 bits of the
/// fractional parts of the cube roots of the first 64 primes.
const ROUND_CONSTANTS: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];
/// The lowercase hex digits used by [`hex_digest`].
const HEX_DIGITS: [u8; 16] = *b"0123456789abcdef";
/// Computes the SHA-256 digest of `bytes`.
///
/// The result is the 32-byte big-endian digest, exactly what Python's
/// `hashlib.sha256(data).digest()` returns. Use [`hex_digest`] when the value is
/// stored or compared as text (the omni pipeline keys its payload dedupe on the
/// hex form, matching `hashlib.sha256(data).hexdigest()`).
#[must_use]
pub fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut state = INITIAL_STATE;
    for block in padded(bytes).chunks_exact(BLOCK_BYTES) {
        compress(&mut state, block);
    }
    let mut output = [0_u8; 32];
    for (index, word) in state.iter().enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    output
}
/// Computes the lowercase hex SHA-256 digest of `bytes`.
///
/// Always 64 ASCII characters, so it is a stable, fixed-width dedupe key. This
/// mirrors Python's `hashlib.sha256(data).hexdigest()`; the value is the same.
#[must_use]
pub fn hex_digest(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in digest(bytes) {
        hex.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
        hex.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
    }
    hex
}
/// Appends the FIPS 180-4 padding and the message length to a copy of `bytes`.
///
/// The copy always ends on a 64-byte boundary: one `0x80` byte, `0…0` up to 56
/// bytes modulo the block size, then the 64-bit big-endian bit length. The
/// length multiplication saturates through `wrapping_mul`, which only matters
/// for inputs no machine can hold.
fn padded(bytes: &[u8]) -> Vec<u8> {
    let remainder = (bytes.len() + 1) % BLOCK_BYTES;
    let zeros = if remainder <= 56 {
        56 - remainder
    } else {
        BLOCK_BYTES + 56 - remainder
    };
    let mut message = Vec::with_capacity(bytes.len() + 1 + zeros + 8);
    message.extend_from_slice(bytes);
    message.push(0x80);
    message.resize(message.len() + zeros, 0);
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    message.extend_from_slice(&bit_length.to_be_bytes());
    message
}
/// Mixes one 64-byte block into the running hash state.
///
/// `block` must be exactly [`BLOCK_BYTES`] long; every caller slices it from
/// [`padded`], which guarantees the length. The function indexes the block
/// directly (rather than returning an `Option`), because a shorter block cannot
/// occur on any path into it.
fn compress(state: &mut [u32; 8], block: &[u8]) {
    let mut schedule = [0_u32; 64];
    for (index, word) in block.chunks_exact(4).enumerate() {
        schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
    }
    for index in 16..64 {
        let previous = schedule[index - 15];
        let lower = schedule[index - 2];
        let small_sigma_0 = previous.rotate_right(7) ^ previous.rotate_right(18) ^ (previous >> 3);
        let small_sigma_1 = lower.rotate_right(17) ^ lower.rotate_right(19) ^ (lower >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(small_sigma_0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(small_sigma_1);
    }
    let mut working = *state;
    for index in 0..64 {
        let big_sigma_1 =
            working[4].rotate_right(6) ^ working[4].rotate_right(11) ^ working[4].rotate_right(25);
        let choose = (working[4] & working[5]) ^ (!working[4] & working[6]);
        let first = working[7]
            .wrapping_add(big_sigma_1)
            .wrapping_add(choose)
            .wrapping_add(ROUND_CONSTANTS[index])
            .wrapping_add(schedule[index]);
        let big_sigma_0 =
            working[0].rotate_right(2) ^ working[0].rotate_right(13) ^ working[0].rotate_right(22);
        let majority =
            (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
        let second = big_sigma_0.wrapping_add(majority);
        working[7] = working[6];
        working[6] = working[5];
        working[5] = working[4];
        working[4] = working[3].wrapping_add(first);
        working[3] = working[2];
        working[2] = working[1];
        working[1] = working[0];
        working[0] = first.wrapping_add(second);
    }
    for (slot, value) in state.iter_mut().zip(working) {
        *slot = slot.wrapping_add(value);
    }
}
#[cfg(test)]
mod tests {
    use super::{digest, hex_digest};
    /// The published SHA-256 vectors from FIPS 180-4 / the NIST test suite.
    const VECTORS: [(&str, &str); 4] = [
        (
            "",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            "abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
        (
            "The quick brown fox jumps over the lazy dog",
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
        ),
    ];
    #[test]
    fn matches_published_vectors() {
        for (input, expected) in VECTORS {
            assert_eq!(hex_digest(input.as_bytes()), expected, "{input:?}");
        }
    }
    #[test]
    fn matches_the_million_a_vector() {
        let payload = vec![b'a'; 1_000_000];
        assert_eq!(
            hex_digest(&payload),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
    #[test]
    fn pads_across_the_fifty_six_and_sixty_four_byte_boundaries() {
        for (length, expected) in [
            (
                55,
                "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            ),
            (
                56,
                "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
            ),
            (
                63,
                "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34",
            ),
            (
                64,
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            ),
            (
                65,
                "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0",
            ),
            (
                119,
                "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb",
            ),
            (
                120,
                "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c",
            ),
            (
                127,
                "c57e9278af78fa3cab38667bef4ce29d783787a2f731d4e12200270f0c32320a",
            ),
            (
                128,
                "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e",
            ),
        ] {
            assert_eq!(hex_digest(&vec![b'a'; length]), expected, "length {length}");
        }
    }
    #[test]
    fn chunking_the_input_does_not_change_the_digest() {
        let expected = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
        let source = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let mut reassembled = Vec::new();
        for chunk in source.chunks(3) {
            reassembled.extend_from_slice(chunk);
        }
        assert_eq!(reassembled.as_slice(), source.as_slice());
        assert_eq!(hex_digest(&reassembled), expected);
        let mut long = Vec::new();
        for _ in 0..1000 {
            long.extend_from_slice(b"a");
        }
        assert_eq!(
            hex_digest(&long),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }
    #[test]
    fn digest_is_the_byte_form_of_the_hex_form() {
        for (input, expected) in VECTORS {
            let bytes = digest(input.as_bytes());
            assert_eq!(bytes.len(), 32);
            for (index, byte) in bytes.iter().enumerate() {
                let pair = expected.get(index * 2..index * 2 + 2).unwrap_or_default();
                assert_eq!(*byte, u8::from_str_radix(pair, 16).unwrap(), "{input:?}");
            }
            assert_eq!(hex_digest(input.as_bytes()).len(), 64);
        }
        assert!(hex_digest(b"payload")
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
    }
    #[test]
    fn distinct_payloads_have_distinct_digests() {
        assert_ne!(digest(b""), digest(b"\0"));
        assert_ne!(digest(b"payload"), digest(b"payload "));
        assert_ne!(digest(&[0_u8; 64]), digest(&[0_u8; 65]));
    }
}
