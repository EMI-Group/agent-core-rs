//! MPEG audio (MP3) header parsing: an optional `ID3v2` tag, the first MPEG frame header, and the
//! `Xing`/`Info` tag that declares the frame count.
//!
//! Nothing is decoded: the first frame header supplies version, layer, bitrate, sample rate and
//! channel mode, the `Xing` frame count supplies the exact duration when present, and otherwise
//! the duration is estimated from the stream size at the nominal bitrate.

use serde::Serialize;

use super::super::parse::{be_u32_at, Failure, ParseResult};
use super::strip_id3v2;

/// Highest number of bytes scanned for the first frame header.
const MAX_SEARCH_BYTES: usize = 1 << 20;

/// MPEG-1 Layer I bitrates, indexed by the header's bitrate index (`0` and `15` are invalid).
const MPEG1_LAYER1_BITRATES: [u32; 16] = [
    0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0,
];

/// MPEG-1 Layer II bitrates, indexed by the header's bitrate index.
const MPEG1_LAYER2_BITRATES: [u32; 16] = [
    0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0,
];

/// MPEG-1 Layer III bitrates, indexed by the header's bitrate index.
const MPEG1_LAYER3_BITRATES: [u32; 16] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
];

/// MPEG-2 and MPEG-2.5 Layer I bitrates, indexed by the header's bitrate index.
const MPEG2_LAYER1_BITRATES: [u32; 16] = [
    0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0,
];

/// MPEG-2 and MPEG-2.5 Layer II/III bitrates, indexed by the header's bitrate index.
const MPEG2_LAYER23_BITRATES: [u32; 16] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
];

/// MPEG-1 sample rates, indexed by the header's sample-rate index.
const MPEG1_SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 32_000];

/// MPEG-2 sample rates, indexed by the header's sample-rate index.
const MPEG2_SAMPLE_RATES: [u32; 3] = [22_050, 24_000, 16_000];

/// MPEG-2.5 sample rates, indexed by the header's sample-rate index.
const MPEG25_SAMPLE_RATES: [u32; 3] = [11_025, 12_000, 8_000];

/// Fields declared by the first MPEG frame header plus the `Xing`/`Info` frame count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Mp3Info {
    /// MPEG version: `1`, `2` or `2.5`.
    pub version: &'static str,
    /// MPEG layer: `I`, `II` or `III`.
    pub layer: &'static str,
    /// Nominal bitrate in kbit/s.
    pub bitrate_kbps: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count: `1` (mono) or `2`.
    pub channels: u8,
    /// Duration in milliseconds: exact from the `Xing` frame count when present, otherwise
    /// estimated from the stream size at the nominal bitrate.
    pub duration_ms: Option<u64>,
    /// Frame count declared by the `Xing`/`Info` tag.
    pub xing_frames: Option<u32>,
    /// Length of a skipped leading `ID3v2` tag, in bytes (`0` when there is none).
    pub id3v2_bytes: usize,
}

/// A decoded MPEG frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    /// MPEG version spelling.
    version: &'static str,
    /// MPEG layer spelling.
    layer: &'static str,
    /// Nominal bitrate in kbit/s.
    bitrate_kbps: u32,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Channel count.
    channels: u8,
    /// Whether a 16-bit CRC follows the header.
    crc_present: bool,
    /// PCM samples per frame.
    samples_per_frame: u32,
    /// Bytes of side information after the header (and CRC).
    side_info_bytes: usize,
}

/// Parses an MP3 header.
///
/// # Errors
///
/// Returns a [`Failure`] when the stream (after any `ID3v2` tag) is empty or contains no valid MPEG
/// frame header within the first `MAX_SEARCH_BYTES` bytes.
pub fn parse(bytes: &[u8]) -> ParseResult<Mp3Info> {
    let body = strip_id3v2(bytes);
    let id3v2_bytes = bytes.len().saturating_sub(body.len());
    let Some((offset, header)) = first_frame(body) else {
        let scanned = body.len().min(MAX_SEARCH_BYTES);
        return Err(Failure::new(if body.is_empty() {
            "truncated MP3 header: stream is empty".to_owned()
        } else {
            format!("truncated MP3 header: no MPEG frame header found in the first {scanned} bytes")
        }));
    };
    let xing_frames = xing_frames(body, offset, &header);
    let duration_ms = match xing_frames {
        Some(frames) => Some(
            u64::from(frames) * u64::from(header.samples_per_frame) * 1000
                / u64::from(header.sample_rate),
        ),
        None => constant_bitrate_duration(body, offset, header.bitrate_kbps),
    };
    Ok(Mp3Info {
        version: header.version,
        layer: header.layer,
        bitrate_kbps: header.bitrate_kbps,
        sample_rate: header.sample_rate,
        channels: header.channels,
        duration_ms,
        xing_frames,
        id3v2_bytes,
    })
}

/// Whether `bytes` starts with a plausible MPEG audio frame header.
///
/// Used by the audio dispatch to tell an MP3 stream (possibly after an `ID3v2` tag) from an
/// unsupported container without committing to a full parse.
#[must_use]
pub fn looks_like_frame(bytes: &[u8]) -> bool {
    first_frame(bytes).is_some()
}

/// Finds the first valid frame header and its offset.
fn first_frame(bytes: &[u8]) -> Option<(usize, Header)> {
    let limit = bytes.len().min(MAX_SEARCH_BYTES);
    (0..limit).find_map(|offset| parse_header(bytes, offset).map(|header| (offset, header)))
}

/// Decodes the four-byte frame header at `offset`.
fn parse_header(bytes: &[u8], offset: usize) -> Option<Header> {
    let raw = be_u32_at(bytes, offset)?;
    // Eleven sync bits, then a version/layer pair that must not be "reserved".
    if raw >> 21 != 0x7FF {
        return None;
    }
    let version_bits = (raw >> 19) & 0x03;
    let layer_bits = (raw >> 17) & 0x03;
    if version_bits == 1 || layer_bits == 0 {
        return None;
    }
    let bitrate_index = usize::try_from((raw >> 12) & 0x0F).ok()?;
    let sample_rate_index = usize::try_from((raw >> 10) & 0x03).ok()?;
    if bitrate_index == 0 || bitrate_index == 15 || sample_rate_index == 3 {
        return None;
    }
    let crc_present = ((raw >> 16) & 0x01) == 0;
    let mono = ((raw >> 6) & 0x03) == 3;
    let bitrate_kbps = bitrate_for(version_bits, layer_bits, bitrate_index)?;
    let sample_rate = match version_bits {
        3 => MPEG1_SAMPLE_RATES.get(sample_rate_index).copied()?,
        2 => MPEG2_SAMPLE_RATES.get(sample_rate_index).copied()?,
        _ => MPEG25_SAMPLE_RATES.get(sample_rate_index).copied()?,
    };
    let (version, layer, samples_per_frame, side_info_bytes) = match (version_bits, layer_bits) {
        (3, 3) => ("1", "I", 384, if mono { 17 } else { 32 }),
        (3, 2) => ("1", "II", 1152, if mono { 17 } else { 32 }),
        (3, _) => ("1", "III", 1152, if mono { 17 } else { 32 }),
        (2, 3) => ("2", "I", 384, if mono { 9 } else { 17 }),
        (2, 2) => ("2", "II", 1152, if mono { 9 } else { 17 }),
        (2, _) => ("2", "III", 576, if mono { 9 } else { 17 }),
        (_, 3) => ("2.5", "I", 384, if mono { 9 } else { 17 }),
        (_, 2) => ("2.5", "II", 1152, if mono { 9 } else { 17 }),
        (_, _) => ("2.5", "III", 576, if mono { 9 } else { 17 }),
    };
    Some(Header {
        version,
        layer,
        bitrate_kbps,
        sample_rate,
        channels: if mono { 1 } else { 2 },
        crc_present,
        samples_per_frame,
        side_info_bytes,
    })
}

/// Looks up the nominal bitrate for a version/layer pair and bitrate index.
fn bitrate_for(version_bits: u32, layer_bits: u32, index: usize) -> Option<u32> {
    let table = if version_bits == 3 {
        match layer_bits {
            3 => &MPEG1_LAYER1_BITRATES,
            2 => &MPEG1_LAYER2_BITRATES,
            _ => &MPEG1_LAYER3_BITRATES,
        }
    } else if layer_bits == 3 {
        &MPEG2_LAYER1_BITRATES
    } else {
        &MPEG2_LAYER23_BITRATES
    };
    let bitrate = table.get(index).copied()?;
    (bitrate > 0).then_some(bitrate)
}

/// Reads the frame count from a `Xing`/`Info` tag in the first frame.
fn xing_frames(bytes: &[u8], frame_offset: usize, header: &Header) -> Option<u32> {
    let crc_bytes = usize::from(header.crc_present) * 2;
    let tag_offset = frame_offset
        .checked_add(4)?
        .checked_add(crc_bytes)?
        .checked_add(header.side_info_bytes)?;
    let kind = bytes.get(tag_offset..tag_offset.checked_add(4)?)?;
    if kind != b"Xing".as_slice() && kind != b"Info".as_slice() {
        return None;
    }
    let flags = be_u32_at(bytes, tag_offset.checked_add(4)?)?;
    if flags & 0x01 == 0 {
        return None;
    }
    be_u32_at(bytes, tag_offset.checked_add(8)?)
}

/// Estimates the duration of a constant-bitrate stream from its size.
fn constant_bitrate_duration(bytes: &[u8], frame_offset: usize, bitrate_kbps: u32) -> Option<u64> {
    if bitrate_kbps == 0 {
        return None;
    }
    let audio_bytes = bytes.len().saturating_sub(frame_offset);
    let bits = u64::try_from(audio_bytes).unwrap_or_default() * 8;
    // bits / (kbit/s * 1000) seconds == bits / kbit/s milliseconds.
    Some(bits / u64::from(bitrate_kbps))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A MPEG-1 Layer III 44.1 kHz stereo frame header.
    const MPEG1_STEREO: [u8; 4] = [0xFF, 0xFB, 0x90, 0x00];

    /// A MPEG-2 Layer III 22.05 kHz mono frame header.
    const MPEG2_MONO: [u8; 4] = [0xFF, 0xF3, 0x90, 0xC0];

    /// Builds `header`, its side information and an optional `Xing` frame count.
    fn stream(header: [u8; 4], side_info: usize, frames: Option<u32>) -> Vec<u8> {
        let mut bytes = header.to_vec();
        bytes.extend(std::iter::repeat_n(0u8, side_info));
        if let Some(frames) = frames {
            bytes.extend_from_slice(b"Xing");
            bytes.extend_from_slice(&1u32.to_be_bytes());
            bytes.extend_from_slice(&frames.to_be_bytes());
        }
        bytes
    }

    #[test]
    fn reads_mpeg1_layer3_header_and_xing_duration() {
        let info = parse(&stream(MPEG1_STEREO, 32, Some(100))).unwrap();
        assert_eq!(info.version, "1");
        assert_eq!(info.layer, "III");
        assert_eq!(info.bitrate_kbps, 128);
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.xing_frames, Some(100));
        // 100 frames * 1152 samples / 44100 Hz.
        assert_eq!(info.duration_ms, Some(2612));
        assert_eq!(info.id3v2_bytes, 0);
    }

    #[test]
    fn reads_mpeg2_mono_header_and_short_frames() {
        let info = parse(&stream(MPEG2_MONO, 9, Some(100))).unwrap();
        assert_eq!(info.version, "2");
        assert_eq!(info.layer, "III");
        assert_eq!(info.channels, 1);
        assert_eq!(info.bitrate_kbps, 80);
        assert_eq!(info.sample_rate, 22_050);
        // 100 frames * 576 samples / 22050 Hz.
        assert_eq!(info.duration_ms, Some(2612));
    }

    #[test]
    fn estimates_the_duration_without_a_xing_tag() {
        let mut bytes = stream(MPEG1_STEREO, 32, None);
        bytes.resize(16_000, 0);
        let info = parse(&bytes).unwrap();
        assert_eq!(info.xing_frames, None);
        // 16000 bytes at 128 kbit/s.
        assert_eq!(info.duration_ms, Some(1000));
    }

    #[test]
    fn skips_and_reports_an_id3v2_tag() {
        let mut bytes = b"ID3\x04\x00\x00\x00\x00\x00\x0A".to_vec();
        bytes.extend_from_slice(&[0; 10]);
        bytes.extend_from_slice(&stream(MPEG1_STEREO, 32, Some(100)));
        let info = parse(&bytes).unwrap();
        assert_eq!(info.id3v2_bytes, 20);
        assert_eq!(info.duration_ms, Some(2612));
    }

    #[test]
    fn finds_the_frame_after_stray_leading_bytes() {
        let mut bytes = vec![0x00, 0x11, 0x22];
        bytes.extend_from_slice(&stream(MPEG1_STEREO, 32, Some(100)));
        let info = parse(&bytes).unwrap();
        assert_eq!(info.sample_rate, 44_100);
    }

    #[test]
    fn empty_and_frameless_streams_are_reported() {
        assert_eq!(
            parse(&[]).unwrap_err().message(),
            "truncated MP3 header: stream is empty"
        );
        assert_eq!(
            parse(b"\xFF\xFB").unwrap_err().message(),
            "truncated MP3 header: no MPEG frame header found in the first 2 bytes"
        );
        assert_eq!(
            parse(b"XXXX").unwrap_err().message(),
            "truncated MP3 header: no MPEG frame header found in the first 4 bytes"
        );
    }

    #[test]
    fn reserved_header_fields_are_rejected() {
        // Reserved version (01).
        assert!(!looks_like_frame(&[0xFF, 0xEB, 0x90, 0x00]));
        // Reserved layer (00).
        assert!(!looks_like_frame(&[0xFF, 0xF9, 0x90, 0x00]));
        // Free-format bitrate index (0000).
        assert!(!looks_like_frame(&[0xFF, 0xFB, 0x00, 0x00]));
        // Reserved sample-rate index (11).
        assert!(!looks_like_frame(&[0xFF, 0xFB, 0x9C, 0x00]));
        // No sync word.
        assert!(!looks_like_frame(&[0x00, 0xFB, 0x90, 0x00]));
        assert!(looks_like_frame(&MPEG1_STEREO));
    }

    #[test]
    fn bitrate_table_is_checked_for_every_version_and_layer() {
        assert_eq!(bitrate_for(3, 3, 14), Some(448));
        assert_eq!(bitrate_for(3, 2, 14), Some(384));
        assert_eq!(bitrate_for(3, 1, 14), Some(320));
        assert_eq!(bitrate_for(2, 3, 14), Some(256));
        assert_eq!(bitrate_for(2, 2, 14), Some(160));
        assert_eq!(bitrate_for(2, 1, 14), Some(160));
        assert_eq!(bitrate_for(3, 1, 0), None);
        assert_eq!(bitrate_for(3, 1, 15), None);
    }

    #[test]
    fn a_truncated_xing_tag_leaves_duration_to_the_estimate() {
        let mut bytes = MPEG1_STEREO.to_vec();
        bytes.extend(std::iter::repeat_n(0u8, 32));
        bytes.extend_from_slice(b"Xing");
        let info = parse(&bytes).unwrap();
        assert_eq!(info.xing_frames, None);
        assert!(info.duration_ms.is_some_and(|ms| ms > 0));
    }

    #[test]
    fn xing_without_the_frame_count_flag_is_ignored() {
        let mut bytes = MPEG1_STEREO.to_vec();
        bytes.extend(std::iter::repeat_n(0u8, 32));
        bytes.extend_from_slice(b"Xing");
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(&100u32.to_be_bytes());
        let info = parse(&bytes).unwrap();
        assert_eq!(info.xing_frames, None);
    }
}
