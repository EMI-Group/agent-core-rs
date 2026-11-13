//! Table-driven CRC-32 (IEEE, polynomial `0xEDB88320`), shared by `zip` and `gzip`.
//!
//! Both container formats store the same CRC-32 of their uncompressed payload,
//! so one implementation serves [`super::zip`] and [`super::gzip`].

/// The IEEE CRC-32 polynomial in reflected form.
const POLYNOMIAL: u32 = 0xEDB8_8320;

/// The lookup table, built once at compile time.
const TABLE: [u32; 256] = build_table();

/// Builds the 256-entry reflected CRC-32 table.
const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0usize;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 1 {
                (value >> 1) ^ POLYNOMIAL
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

/// Computes the CRC-32 (IEEE) of `bytes`.
///
/// The classic check vector holds: `crc32(b"123456789") == 0xCBF4_3926`.
#[must_use]
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut state = 0xFFFF_FFFFu32;
    for &byte in bytes {
        let index = ((state ^ u32::from(byte)) & 0xFF) as usize;
        state = (state >> 8) ^ TABLE[index];
    }
    !state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_known_check_vectors() {
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }
}
