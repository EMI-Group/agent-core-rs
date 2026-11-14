//! SHA-256 digests for stable harness-side identifiers.
//!
//! Python derives `PersonalContext` source ids (`src_<32 hex>`), source-link
//! identities and KVC sub-session scopes from SHA-256, so a Rust port that must
//! produce the same ids needs the same digest. The workspace pins `sha2` in
//! `[workspace.dependencies]`, but `openjiuwen-harness` does not declare it and
//! the harness manifest is owned by another layer, so this module carries a
//! small self-contained FIPS 180-4 implementation.
//!
//! This is an identifier digest, not a security primitive: it makes ids stable
//! and collision-resistant across runs, and it is never used to authenticate,
//! sign or encrypt anything. Secrets never enter it.

const K: [u32; 64] = [
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

#[allow(clippy::many_single_char_names)]
fn compress(state: &mut [u32; 8], block: &[u8]) {
    let mut schedule = [0u32; 64];
    for (slot, chunk) in schedule.iter_mut().zip(block.chunks_exact(4)) {
        *slot = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for index in 16..64 {
        let first = schedule[index - 15];
        let second = schedule[index - 2];
        let s0 = first.rotate_right(7) ^ first.rotate_right(18) ^ (first >> 3);
        let s1 = second.rotate_right(17) ^ second.rotate_right(19) ^ (second >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (round, word) in K.iter().zip(schedule.iter()) {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(choose)
            .wrapping_add(*round)
            .wrapping_add(*word);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(value);
    }
}

/// Returns the lowercase hex SHA-256 digest of `input` (64 characters).
pub(crate) fn sha256_hex(input: impl AsRef<[u8]>) -> String {
    let input = input.as_ref();
    let mut state = INITIAL_STATE;
    let mut blocks = input.chunks_exact(64);
    for block in &mut blocks {
        compress(&mut state, block);
    }

    let mut tail = Vec::with_capacity(64);
    tail.extend_from_slice(blocks.remainder());
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    let bit_len = (input.len() as u64).wrapping_mul(8);
    tail.extend_from_slice(&bit_len.to_be_bytes());
    for block in tail.chunks_exact(64) {
        compress(&mut state, block);
    }

    state
        .iter()
        .fold(String::with_capacity(64), |mut digest, word| {
            const NIBBLES: [u8; 16] = *b"0123456789abcdef";
            for shift in (0..32).step_by(4).rev() {
                let nibble = usize::try_from((word >> shift) & 0xF).unwrap_or_default();
                digest.push(char::from(NIBBLES[nibble]));
            }
            digest
        })
}

/// Returns the first `hex_chars` characters of the hex digest of `input`.
///
/// `hex_chars` is clamped to the 64 characters a digest has, so an oversized
/// request degrades to the full digest instead of panicking.
pub(crate) fn sha256_prefix(input: impl AsRef<[u8]>, hex_chars: usize) -> String {
    let digest = sha256_hex(input);
    let end = hex_chars.min(digest.len());
    digest[..end].to_string()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// FIPS 180-4 / NIST examples.
    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const TWO_BLOCKS: &str = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";

    #[test]
    fn matches_the_published_vectors() {
        assert_eq!(sha256_hex(""), EMPTY);
        assert_eq!(sha256_hex("abc"), ABC);
        assert_eq!(
            sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            TWO_BLOCKS
        );
    }

    #[test]
    fn covers_the_padding_boundaries() {
        // 55 bytes: the length still fits in the first padded block.
        assert_eq!(
            sha256_hex("a".repeat(55)),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
        );
        // 56 bytes: the length forces a second padded block.
        assert_eq!(
            sha256_hex("a".repeat(56)),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
        );
        // 64 bytes: exactly one block plus padding.
        assert_eq!(
            sha256_hex("a".repeat(64)),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
        );
    }

    #[test]
    fn prefix_truncates_and_clamps() {
        assert_eq!(sha256_prefix("abc", 12), &ABC[..12]);
        assert_eq!(sha256_prefix("abc", 0), "");
        assert_eq!(sha256_prefix("abc", 1000), ABC);
        assert_eq!(sha256_prefix("abc", 12).len(), 12);
    }

    #[test]
    fn digests_are_lowercase_hex() {
        let digest = sha256_hex("openjiuwen");
        assert_eq!(digest.len(), 64);
        assert!(digest
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));
    }
}
