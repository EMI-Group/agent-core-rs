//! FLAC header parsing: the `fLaC` stream marker and the mandatory first `STREAMINFO` metadata
//! block, which carries the sample rate, channel count, bit depth and total sample count.
//!
//! A leading `ID3v2` tag is skipped (some taggers add one) and its length is reported. Audio frames
//! are never read.

use serde::Serialize;

use super::super::parse::{be_u16_at, be_u64_at, fourcc, Cursor, Failure, ParseResult};
use super::strip_id3v2;

/// Length of a `STREAMINFO` payload, in bytes.
const STREAMINFO_LEN: usize = 34;

/// The `STREAMINFO` block type.
const STREAMINFO_TYPE: u8 = 0;

/// Fields declared by the FLAC `STREAMINFO` block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FlacInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u8,
    /// Bits per sample.
    pub bits_per_sample: u8,
    /// Total number of inter-channel samples; `0` when the encoder left it unknown.
    pub total_samples: u64,
    /// Duration in milliseconds, when the total sample count is known.
    pub duration_ms: Option<u64>,
    /// Smallest block size in samples.
    pub min_block_size: u16,
    /// Largest block size in samples.
    pub max_block_size: u16,
    /// Length of a skipped leading `ID3v2` tag, in bytes (`0` when there is none).
    pub id3v2_bytes: usize,
}

/// Parses a FLAC header.
///
/// # Errors
///
/// Returns a [`Failure`] when the stream marker is missing or truncated, when the metadata block
/// header or length is truncated, when the first block is not `STREAMINFO`, or when `STREAMINFO`
/// does not have its mandatory 34-byte payload.
pub fn parse(bytes: &[u8]) -> ParseResult<FlacInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated FLAC header: STREAMINFO is incomplete")
    }

    let body = strip_id3v2(bytes);
    let id3v2_bytes = bytes.len().saturating_sub(body.len());
    let mut cursor = Cursor::new(body);
    let Some(marker) = cursor.take(4) else {
        return Err(Failure::new(
            "truncated FLAC header: stream marker is incomplete",
        ));
    };
    if marker != b"fLaC".as_slice() {
        return Err(Failure::new(format!(
            "unsupported audio format: {}",
            fourcc(marker)
        )));
    }
    let Some(header) = cursor.u8() else {
        return Err(Failure::new(
            "truncated FLAC header: metadata block header is incomplete",
        ));
    };
    let kind = header & 0x7F;
    let Some(length) = cursor.be_u24() else {
        return Err(Failure::new(
            "truncated FLAC header: metadata block length is incomplete",
        ));
    };
    if kind != STREAMINFO_TYPE {
        return Err(Failure::new(format!(
            "invalid FLAC header: first metadata block is type {kind}, expected STREAMINFO (0)"
        )));
    }
    if length != u32::try_from(STREAMINFO_LEN).unwrap_or_default() {
        return Err(Failure::new(format!(
            "invalid FLAC header: STREAMINFO is {length} bytes, expected {STREAMINFO_LEN}"
        )));
    }
    let Some(payload) = cursor.take(STREAMINFO_LEN) else {
        return Err(incomplete());
    };
    let min_block_size = be_u16_at(payload, 0).ok_or_else(incomplete)?;
    let max_block_size = be_u16_at(payload, 2).ok_or_else(incomplete)?;
    // The packed 64-bit field at offset 10 holds 20-bit sample rate, 3-bit channels minus one,
    // 5-bit bits-per-sample minus one and a 36-bit total sample count.
    let packed = be_u64_at(payload, 10).ok_or_else(incomplete)?;
    let sample_rate = u32::try_from(packed >> 44).unwrap_or_default();
    let channels = u8::try_from((packed >> 41) & 0x07).unwrap_or_default() + 1;
    let bits_per_sample = u8::try_from((packed >> 36) & 0x1F).unwrap_or_default() + 1;
    let total_samples = packed & 0x0F_FFFF_FFFF;
    if sample_rate == 0 {
        return Err(Failure::new("invalid FLAC header: zero sample rate"));
    }
    Ok(FlacInfo {
        sample_rate,
        channels,
        bits_per_sample,
        total_samples,
        duration_ms: duration_ms(total_samples, sample_rate),
        min_block_size,
        max_block_size,
        id3v2_bytes,
    })
}

/// Computes the duration of `total_samples` at `sample_rate` hertz.
fn duration_ms(total_samples: u64, sample_rate: u32) -> Option<u64> {
    if total_samples == 0 || sample_rate == 0 {
        return None;
    }
    Some(total_samples * 1000 / u64::from(sample_rate))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds a 34-byte `STREAMINFO` payload.
    fn stream_info(sample_rate: u32, channels: u8, bits: u8, total_samples: u64) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&4096u16.to_be_bytes());
        payload.extend_from_slice(&4096u16.to_be_bytes());
        payload.extend_from_slice(&[0, 0, 0]);
        payload.extend_from_slice(&[0, 0, 0]);
        let packed = (u64::from(sample_rate) << 44)
            | (u64::from(channels - 1) << 41)
            | (u64::from(bits - 1) << 36)
            | total_samples;
        payload.extend_from_slice(&packed.to_be_bytes());
        payload.extend_from_slice(&[0; 16]);
        payload
    }

    /// Builds a FLAC stream from a raw `STREAMINFO` payload with `block_type`.
    fn flac(block_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut bytes = b"fLaC".to_vec();
        bytes.push(0x80 | block_type);
        bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_be_bytes()[1..]);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn reads_streaminfo_fields() {
        let info = parse(&flac(0, &stream_info(44_100, 2, 16, 44_100))).unwrap();
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.total_samples, 44_100);
        assert_eq!(info.duration_ms, Some(1000));
        assert_eq!(info.min_block_size, 4096);
        assert_eq!(info.max_block_size, 4096);
        assert_eq!(info.id3v2_bytes, 0);
    }

    #[test]
    fn reads_mono_and_high_resolution_streams() {
        let info = parse(&flac(0, &stream_info(192_000, 1, 24, 192_000))).unwrap();
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 24);
        assert_eq!(info.duration_ms, Some(1000));
    }

    #[test]
    fn unknown_total_samples_leave_duration_unknown() {
        let info = parse(&flac(0, &stream_info(48_000, 2, 16, 0))).unwrap();
        assert_eq!(info.total_samples, 0);
        assert_eq!(info.duration_ms, None);
    }

    #[test]
    fn skips_and_reports_an_id3v2_tag() {
        let mut bytes = b"ID3\x04\x00\x00\x00\x00\x00\x05".to_vec();
        bytes.extend_from_slice(&[0; 5]);
        bytes.extend_from_slice(&flac(0, &stream_info(44_100, 2, 16, 44_100)));
        let info = parse(&bytes).unwrap();
        assert_eq!(info.id3v2_bytes, 15);
        assert_eq!(info.duration_ms, Some(1000));
    }

    #[test]
    fn truncated_marker_is_reported() {
        assert_eq!(
            parse(b"fLa").unwrap_err().message(),
            "truncated FLAC header: stream marker is incomplete"
        );
        assert_eq!(
            parse(&[]).unwrap_err().message(),
            "truncated FLAC header: stream marker is incomplete"
        );
    }

    #[test]
    fn unsupported_magic_is_reported() {
        assert_eq!(
            parse(b"OggSnot flac").unwrap_err().message(),
            "unsupported audio format: OggS"
        );
    }

    #[test]
    fn truncated_block_header_and_length_are_reported() {
        assert_eq!(
            parse(b"fLaC").unwrap_err().message(),
            "truncated FLAC header: metadata block header is incomplete"
        );
        assert_eq!(
            parse(b"fLaC\x80\x00").unwrap_err().message(),
            "truncated FLAC header: metadata block length is incomplete"
        );
        let mut bytes = b"fLaC\x80\x00\x00\x22".to_vec();
        bytes.extend_from_slice(&[0; 10]);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated FLAC header: STREAMINFO is incomplete"
        );
    }

    #[test]
    fn a_first_block_that_is_not_streaminfo_is_reported() {
        let error = parse(&flac(4, &[0; 8])).unwrap_err();
        assert_eq!(
            error.message(),
            "invalid FLAC header: first metadata block is type 4, expected STREAMINFO (0)"
        );
    }

    #[test]
    fn a_streaminfo_with_the_wrong_length_is_reported() {
        let error = parse(&flac(0, &[0; 33])).unwrap_err();
        assert_eq!(
            error.message(),
            "invalid FLAC header: STREAMINFO is 33 bytes, expected 34"
        );
    }

    #[test]
    fn a_zero_sample_rate_is_rejected() {
        let error = parse(&flac(0, &stream_info(0, 2, 16, 44_100))).unwrap_err();
        assert_eq!(error.message(), "invalid FLAC header: zero sample rate");
    }
}
