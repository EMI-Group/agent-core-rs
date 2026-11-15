//! HTTP Digest authentication (RFC 7616 / RFC 2617), challenge-driven.
//!
//! The component sends the request unauthenticated first; a `401` carrying a
//! `WWW-Authenticate: Digest …` challenge is answered by replaying the identical request
//! once with an `Authorization: Digest …` header (see
//! [`super::HttpRequestComponent`]'s invoke path). Everything here is pure — no I/O and no
//! clock — so the response digest reproduces exactly for a fixed challenge.
//!
//! # Algorithms
//!
//! `MD5`, `MD5-sess`, `SHA-256`, and `SHA-256-sess` are supported. MD5 is implemented
//! in-crate (RFC 1321) because no MD5 crate is a workspace dependency; SHA-256 comes from
//! `sha2`. A challenge naming any other algorithm is not answered, so the `401` is
//! returned as ordinary response data.
//!
//! # `qop`
//!
//! `auth` is used when the server offers it, otherwise the legacy RFC 2069 form without
//! `qop`/`nc`/`cnonce` is sent. `auth-int` is never selected (it needs an entity hash the
//! transport does not expose).
//!
//! # Credentials
//!
//! `username`/`password` come from [`super::HttpAuthConfig`]; the component rejects a
//! missing pair with a typed input-parameter error before sending anything.

use std::fmt::Write as _;

use sha2::{Digest as _, Sha256};
use uuid::Uuid;

/// The `qop` this module answers a challenge with.
const QOP_AUTH: &str = "auth";

/// The nonce count of the single replay this component performs.
const NONCE_COUNT: &str = "00000001";

/// The values one `WWW-Authenticate: Digest …` challenge carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DigestChallenge {
    /// The protection space (`realm`).
    pub(crate) realm: String,
    /// The server nonce.
    pub(crate) nonce: String,
    /// The opaque server state, echoed back when present.
    pub(crate) opaque: Option<String>,
    /// The digest algorithm.
    pub(crate) algorithm: Algorithm,
    /// `Some("auth")` when the server offers the `auth` quality of protection.
    pub(crate) qop: Option<String>,
}

impl DigestChallenge {
    /// Parses a `WWW-Authenticate` value; `None` when it is not a usable digest challenge.
    pub(crate) fn parse(header: &str) -> Option<Self> {
        let rest = digest_parameters(header)?;
        let parameters = parameters(rest);
        let lookup = |name: &str| {
            parameters
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        };
        let nonce = lookup("nonce").filter(|value| !value.is_empty())?;
        let algorithm = Algorithm::parse(lookup("algorithm").as_deref().unwrap_or("MD5"))?;
        let qop = lookup("qop").and_then(|offered| {
            offered
                .split(',')
                .map(str::trim)
                .any(|value| value.eq_ignore_ascii_case(QOP_AUTH))
                .then(|| QOP_AUTH.to_owned())
        });
        Some(Self {
            realm: lookup("realm").unwrap_or_default(),
            nonce,
            opaque: lookup("opaque").filter(|value| !value.is_empty()),
            algorithm,
            qop,
        })
    }
}

/// Strips the `Digest` scheme prefix, returning the parameter list.
fn digest_parameters(header: &str) -> Option<&str> {
    const SCHEME: &str = "digest";
    let trimmed = header.trim_start();
    if trimmed.len() <= SCHEME.len() || !trimmed[..SCHEME.len()].eq_ignore_ascii_case(SCHEME) {
        return None;
    }
    let rest = &trimmed[SCHEME.len()..];
    if !rest.starts_with([' ', '\t']) {
        return None;
    }
    Some(rest.trim_start())
}

/// Splits `key=value` pairs, honouring quoted values.
fn parameters(input: &str) -> Vec<(String, String)> {
    let bytes = input.as_bytes();
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        while index < bytes.len() && (bytes[index].is_ascii_whitespace() || bytes[index] == b',') {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }
        let key_start = index;
        while index < bytes.len() && bytes[index] != b'=' {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }
        let key = input[key_start..index].trim().to_ascii_lowercase();
        index += 1;
        while index < bytes.len() && bytes[index] == b' ' {
            index += 1;
        }
        let value = if index < bytes.len() && bytes[index] == b'"' {
            index += 1;
            let mut raw: Vec<u8> = Vec::new();
            while index < bytes.len() {
                match bytes[index] {
                    b'"' => {
                        index += 1;
                        break;
                    }
                    b'\\' if index + 1 < bytes.len() => {
                        raw.push(bytes[index + 1]);
                        index += 2;
                    }
                    other => {
                        raw.push(other);
                        index += 1;
                    }
                }
            }
            String::from_utf8_lossy(&raw).into_owned()
        } else {
            let start = index;
            while index < bytes.len() && bytes[index] != b',' {
                index += 1;
            }
            input[start..index].trim().to_owned()
        };
        if !key.is_empty() {
            pairs.push((key, value));
        }
    }
    pairs
}

/// The digest algorithms this module can compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Algorithm {
    /// `MD5`.
    Md5,
    /// `MD5-sess`.
    Md5Sess,
    /// `SHA-256`.
    Sha256,
    /// `SHA-256-sess`.
    Sha256Sess,
}

impl Algorithm {
    /// Parses a challenge algorithm name; `None` for an unsupported one.
    fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_uppercase().as_str() {
            "MD5" => Some(Self::Md5),
            "MD5-SESS" => Some(Self::Md5Sess),
            "SHA-256" => Some(Self::Sha256),
            "SHA-256-SESS" => Some(Self::Sha256Sess),
            _ => None,
        }
    }

    /// The wire name echoed in the `Authorization` header.
    fn name(self) -> &'static str {
        match self {
            Self::Md5 => "MD5",
            Self::Md5Sess => "MD5-sess",
            Self::Sha256 => "SHA-256",
            Self::Sha256Sess => "SHA-256-sess",
        }
    }

    /// Whether the algorithm hashes the nonce and cnonce into `HA1`.
    fn is_session(self) -> bool {
        matches!(self, Self::Md5Sess | Self::Sha256Sess)
    }

    /// Lower-case hexadecimal digest of `data`.
    fn hash(self, data: &[u8]) -> String {
        match self {
            Self::Md5 | Self::Md5Sess => md5_hex(data),
            Self::Sha256 | Self::Sha256Sess => hex_encode(&Sha256::digest(data)),
        }
    }
}

/// Builds the `Authorization` header of the replay for `challenge`.
pub(crate) fn authorization(
    challenge: &DigestChallenge,
    username: &str,
    password: &str,
    method: &str,
    uri: &str,
) -> String {
    authorization_with(
        challenge,
        username,
        password,
        method,
        uri,
        &client_nonce(),
        NONCE_COUNT,
    )
}

/// Builds the `Authorization` header with an injected cnonce and nonce count.
fn authorization_with(
    challenge: &DigestChallenge,
    username: &str,
    password: &str,
    method: &str,
    uri: &str,
    cnonce: &str,
    nonce_count: &str,
) -> String {
    let algorithm = challenge.algorithm;
    let mut ha1 = algorithm.hash(format!("{username}:{}:{password}", challenge.realm).as_bytes());
    if algorithm.is_session() {
        ha1 = algorithm.hash(format!("{ha1}:{}:{cnonce}", challenge.nonce).as_bytes());
    }
    let ha2 = algorithm.hash(format!("{}:{uri}", method.to_ascii_uppercase()).as_bytes());
    let response = match challenge.qop.as_deref() {
        Some(qop) => algorithm.hash(
            format!(
                "{ha1}:{}:{nonce_count}:{cnonce}:{qop}:{ha2}",
                challenge.nonce
            )
            .as_bytes(),
        ),
        None => algorithm.hash(format!("{ha1}:{}:{ha2}", challenge.nonce).as_bytes()),
    };

    let mut header = format!(
        "Digest username=\"{}\", realm=\"{}\", nonce=\"{}\", uri=\"{}\"",
        quote(username),
        quote(&challenge.realm),
        quote(&challenge.nonce),
        quote(uri),
    );
    if let Some(qop) = challenge.qop.as_deref() {
        let _ = write!(
            header,
            ", qop={qop}, nc={nonce_count}, cnonce=\"{}\"",
            quote(cnonce)
        );
    }
    let _ = write!(
        header,
        ", response=\"{response}\", algorithm={}",
        algorithm.name()
    );
    if let Some(opaque) = challenge.opaque.as_deref() {
        let _ = write!(header, ", opaque=\"{}\"", quote(opaque));
    }
    header
}

/// Escapes `text` for a quoted header parameter.
fn quote(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// A fresh client nonce (`cnonce`): 32 hexadecimal characters.
fn client_nonce() -> String {
    Uuid::new_v4().simple().to_string()
}

/// Lower-case hexadecimal MD5 of `data` (RFC 1321).
pub(crate) fn md5_hex(data: &[u8]) -> String {
    hex_encode(&md5(data))
}

/// The per-round left-rotation amounts of MD5 (RFC 1321 §3.4).
const MD5_SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
const MD5_CONSTANTS: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0xf61e_2562,
    0xc040_b340,
    0x265e_5a51,
    0xe9b6_c7aa,
    0xd62f_105d,
    0x0244_1453,
    0xd8a1_e681,
    0xe7d3_fbc8,
    0x21e1_cde6,
    0xc337_07d6,
    0xf4d5_0d87,
    0x455a_14ed,
    0xa9e3_e905,
    0xfcef_a3f8,
    0x676f_02d9,
    0x8d2a_4c8a,
    0xfffa_3942,
    0x8771_f681,
    0x6d9d_6122,
    0xfde5_380c,
    0xa4be_ea44,
    0x4bde_cfa9,
    0xf6bb_4b60,
    0xbebf_bc70,
    0x289b_7ec6,
    0xeaa1_27fa,
    0xd4ef_3085,
    0x0488_1d05,
    0xd9d4_d039,
    0xe6db_99e5,
    0x1fa2_7cf8,
    0xc4ac_5665,
    0xf429_2244,
    0x432a_ff97,
    0xab94_23a7,
    0xfc93_a039,
    0x655b_59c3,
    0x8f0c_cc92,
    0xffef_f47d,
    0x8584_5dd1,
    0x6fa8_7e4f,
    0xfe2c_e6e0,
    0xa301_4314,
    0x4e08_11a1,
    0xf753_7e82,
    0xbd3a_f235,
    0x2ad7_d2bb,
    0xeb86_d391,
];
/// MD5 of `data` (RFC 1321), implemented in-crate.
fn md5(data: &[u8]) -> [u8; 16] {
    let mut message = data.to_vec();
    let bit_len = u64::try_from(data.len())
        .unwrap_or(u64::MAX)
        .wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_le_bytes());

    let mut state = [0x6745_2301_u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    for chunk in message.chunks(64) {
        let mut words = [0_u32; 16];
        for (index, word) in chunk.chunks(4).enumerate() {
            words[index] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        }
        let mut registers = state;
        for (index, (&shift, &constant)) in MD5_SHIFTS.iter().zip(MD5_CONSTANTS.iter()).enumerate()
        {
            let (mixed, word_index) = match index {
                0..=15 => (
                    (registers[1] & registers[2]) | (!registers[1] & registers[3]),
                    index,
                ),
                16..=31 => (
                    (registers[3] & registers[1]) | (!registers[3] & registers[2]),
                    (5 * index + 1) % 16,
                ),
                32..=47 => (
                    registers[1] ^ registers[2] ^ registers[3],
                    (3 * index + 5) % 16,
                ),
                _ => (
                    registers[2] ^ (registers[1] | !registers[3]),
                    (7 * index) % 16,
                ),
            };
            let next = registers[1].wrapping_add(
                mixed
                    .wrapping_add(registers[0])
                    .wrapping_add(constant)
                    .wrapping_add(words[word_index])
                    .rotate_left(shift),
            );
            registers = [registers[3], next, registers[1], registers[2]];
        }
        for (slot, value) in state.iter_mut().zip(registers.iter()) {
            *slot = slot.wrapping_add(*value);
        }
    }

    let mut digest = [0_u8; 16];
    for (index, value) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    digest
}

/// Lower-case hexadecimal encoding of `bytes`.
fn hex_encode(bytes: &[u8]) -> String {
    const LOWER: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(LOWER[usize::from(byte >> 4)]));
        encoded.push(char::from(LOWER[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn md5_matches_the_rfc_1321_vectors() {
        for (input, expected) in [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
            (
                "The quick brown fox jumps over the lazy dog",
                "9e107d9d372bb6826bd81d3542a419d6",
            ),
        ] {
            assert_eq!(md5_hex(input.as_bytes()), expected, "input {input:?}");
        }
    }

    #[test]
    fn a_challenge_is_parsed_with_every_parameter() {
        let challenge = DigestChallenge::parse(
            "Digest realm=\"testrealm@host.com\", qop=\"auth,auth-int\", \
             nonce=\"dcd98b7102dd2f0e8b11d0f600bfb0c093\", \
             opaque=\"5ccc069c403ebaf9f0171e9517f40e41\", algorithm=MD5",
        )
        .unwrap();
        assert_eq!(challenge.realm, "testrealm@host.com");
        assert_eq!(challenge.nonce, "dcd98b7102dd2f0e8b11d0f600bfb0c093");
        assert_eq!(
            challenge.opaque.as_deref(),
            Some("5ccc069c403ebaf9f0171e9517f40e41")
        );
        assert_eq!(challenge.algorithm, Algorithm::Md5);
        assert_eq!(challenge.qop.as_deref(), Some("auth"));
    }

    #[test]
    fn unusable_challenges_are_rejected() {
        // Another scheme, a missing nonce, and an unknown algorithm are all unusable.
        assert!(DigestChallenge::parse("Basic realm=\"x\"").is_none());
        assert!(DigestChallenge::parse("Digest realm=\"x\"").is_none());
        assert!(
            DigestChallenge::parse("Digest realm=\"x\", nonce=\"n\", algorithm=SHA-512").is_none()
        );
        // A non-digest scheme with a digest-prefixed name is not accepted either.
        assert!(DigestChallenge::parse("Digestly realm=\"x\"").is_none());
    }

    #[test]
    fn a_challenge_without_qop_uses_the_legacy_form() {
        let challenge =
            DigestChallenge::parse("Digest realm=\"realm@x\", nonce=\"abc123\"").unwrap();
        assert!(challenge.qop.is_none());
        assert_eq!(challenge.algorithm, Algorithm::Md5, "MD5 is the default");
        let header = authorization_with(&challenge, "mufasa", "pw", "get", "/path", "c1", "1");
        // HA1 = md5("mufasa:realm@x:pw"), HA2 = md5("GET:/path").
        assert!(
            header.contains("response=\"6309b495fc9c75e8f4d10930face9cde\""),
            "unexpected header: {header}"
        );
        assert!(!header.contains("qop"));
        assert!(!header.contains("nc="));
        assert!(!header.contains("cnonce"));
        assert!(header.contains("algorithm=MD5"));
    }

    #[test]
    fn the_rfc_2617_vector_reproduces() {
        let challenge = DigestChallenge::parse(
            "Digest realm=\"testrealm@host.com\", qop=\"auth,auth-int\", \
             nonce=\"dcd98b7102dd2f0e8b11d0f600bfb0c093\", \
             opaque=\"5ccc069c403ebaf9f0171e9517f40e41\"",
        )
        .unwrap();
        let header = authorization_with(
            &challenge,
            "Mufasa",
            "Circle Of Life",
            "GET",
            "/dir/index.html",
            "0a4f113b",
            "00000001",
        );
        assert!(header.starts_with("Digest username=\"Mufasa\""));
        assert!(header.contains("realm=\"testrealm@host.com\""));
        assert!(header.contains("nonce=\"dcd98b7102dd2f0e8b11d0f600bfb0c093\""));
        assert!(header.contains("uri=\"/dir/index.html\""));
        assert!(header.contains("qop=auth"));
        assert!(header.contains("nc=00000001"));
        assert!(header.contains("cnonce=\"0a4f113b\""));
        assert!(
            header.contains("response=\"6629fae49393a05397450978507c4ef1\""),
            "unexpected header: {header}"
        );
        assert!(header.contains("opaque=\"5ccc069c403ebaf9f0171e9517f40e41\""));
    }

    #[test]
    fn the_sha256_and_session_variants_use_their_own_hash() {
        let challenge = DigestChallenge::parse(
            "Digest realm=\"realm@x\", nonce=\"abc123\", qop=\"auth\", algorithm=SHA-256",
        )
        .unwrap();
        let header = authorization_with(
            &challenge, "mufasa", "pw", "GET", "/path", "cnonce1", "00000001",
        );
        assert!(
            header.contains(
                "response=\"2c7fb96d4200304472da46d0a1f1fe73b569242e8c2d8f4bf2223e092651f455\""
            ),
            "unexpected header: {header}"
        );
        assert!(header.contains("algorithm=SHA-256"));

        let challenge = DigestChallenge::parse(
            "Digest realm=\"realm@x\", nonce=\"abc123\", qop=\"auth\", algorithm=MD5-sess",
        )
        .unwrap();
        let header = authorization_with(
            &challenge, "mufasa", "pw", "GET", "/path", "cnonce1", "00000001",
        );
        assert!(
            header.contains("response=\"879ed7d0b538e7c64e04fb3b37b49135\""),
            "unexpected header: {header}"
        );
        assert!(header.contains("algorithm=MD5-sess"));
    }

    #[test]
    fn a_generated_authorization_uses_the_auth_quality_of_protection() {
        let challenge = DigestChallenge::parse(
            "Digest realm=\"r\", nonce=\"n\", qop=\"auth-int,auth\", algorithm=SHA-256-sess",
        )
        .unwrap();
        let header = authorization(&challenge, "user", "pass", "POST", "/p?x=1");
        assert!(header.contains("qop=auth,"), "unexpected header: {header}");
        assert!(header.contains("nc=00000001"));
        assert!(header.contains("algorithm=SHA-256-sess"));
        // `cnonce` is a fresh 32-character hexadecimal value.
        let cnonce = header
            .split("cnonce=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .unwrap();
        assert_eq!(cnonce.len(), 32);
        assert!(cnonce
            .chars()
            .all(|character| character.is_ascii_hexdigit()));
    }

    #[test]
    fn quoted_values_are_escaped() {
        assert_eq!(quote("a\"b\\c"), "a\\\"b\\\\c");
    }
}
