//! WAV header parsing: the RIFF container, the `fmt ` chunk and the `data` chunk size used to
//! compute the duration.
//!
//! The sample data is skipped, never read: duration comes from `data` bytes divided by the
//! declared byte rate.

use serde::Serialize;

use super::super::parse::{fourcc, le_u16_at, le_u32_at, Cursor, Failure, ParseResult};

/// Highest number of RIFF chunks the walk visits.
const MAX_CHUNKS: usize = 4096;

/// The `WAVE_FORMAT_EXTENSIBLE` compatibility tag.
const EXTENSIBLE_TAG: u16 = 0xFFFE;

/// The `data` chunk size used by streaming writers to mean "until the end of the file".
const UNKNOWN_DATA_SIZE: u32 = u32::MAX;

/// Fields declared by the WAV `fmt ` and `data` chunks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WavInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Bits per sample.
    pub bits_per_sample: u16,
    /// Raw format tag from the `fmt ` chunk (`1` is PCM, `3` is IEEE float).
    pub format_tag: u16,
    /// Name of the effective sample format (`pcm`, `float`, `alaw`, …).
    pub format_name: &'static str,
    /// Bytes per second of audio.
    pub byte_rate: u32,
    /// Bytes per sample frame.
    pub block_align: u16,
    /// Size of the `data` chunk in bytes, when the file declares one.
    pub data_bytes: Option<u32>,
    /// Duration in milliseconds, computed from `data_bytes` and `byte_rate`.
    pub duration_ms: Option<u64>,
}

/// The fields of one `fmt ` chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FormatChunk {
    /// Raw format tag.
    tag: u16,
    /// Channel count.
    channels: u16,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Bytes per second.
    byte_rate: u32,
    /// Bytes per sample frame.
    block_align: u16,
    /// Bits per sample.
    bits_per_sample: u16,
    /// Effective compatibility tag of a `WAVE_FORMAT_EXTENSIBLE` sub-format.
    subformat: Option<u16>,
}

/// Parses a WAV header.
///
/// # Errors
///
/// Returns a [`Failure`] when the RIFF container is truncated or not `RIFF`/`WAVE`, when a chunk
/// length or the `fmt ` chunk is truncated, when no `fmt ` chunk is present, or when the chunk
/// declares zero channels.
pub fn parse(bytes: &[u8]) -> ParseResult<WavInfo> {
    if bytes.len() < 12 {
        return Err(Failure::new(
            "truncated WAV header: RIFF container is incomplete",
        ));
    }
    if !bytes.starts_with(b"RIFF") {
        return Err(Failure::new(format!(
            "unsupported audio format: {}",
            fourcc(bytes)
        )));
    }
    if bytes.get(8..12) != Some(b"WAVE".as_slice()) {
        let kind = bytes.get(8..).unwrap_or_default();
        return Err(Failure::new(format!(
            "unsupported audio format: {}",
            fourcc(kind)
        )));
    }
    let mut cursor = Cursor::new(bytes);
    let _ = cursor.seek(12);
    let mut format: Option<FormatChunk> = None;
    let mut data_bytes: Option<u32> = None;
    for _ in 0..MAX_CHUNKS {
        let Some(kind) = cursor.take(4) else {
            break;
        };
        let Some(size) = cursor.le_u32() else {
            return Err(Failure::new(
                "truncated WAV header: chunk length is incomplete",
            ));
        };
        let padded = usize::try_from(size).unwrap_or(usize::MAX);
        if kind == b"fmt ".as_slice() {
            let Some(payload) = cursor.take(padded) else {
                return Err(Failure::new(
                    "truncated WAV header: fmt chunk is incomplete",
                ));
            };
            format = Some(parse_format(payload)?);
        } else {
            if kind == b"data".as_slice() {
                data_bytes = Some(size);
            }
            if !cursor.skip(padded) {
                break;
            }
        }
        // RIFF pads every chunk to an even length.
        if size % 2 == 1 && !cursor.skip(1) {
            break;
        }
    }
    let Some(format) = format else {
        return Err(Failure::new("truncated WAV header: missing fmt chunk"));
    };
    if format.channels == 0 {
        return Err(Failure::new("invalid WAV header: zero channels"));
    }
    Ok(WavInfo {
        sample_rate: format.sample_rate,
        channels: format.channels,
        bits_per_sample: format.bits_per_sample,
        format_tag: format.tag,
        format_name: format_name(format.tag, format.subformat),
        byte_rate: format.byte_rate,
        block_align: format.block_align,
        data_bytes,
        duration_ms: duration_ms(data_bytes, format.byte_rate),
    })
}

/// Reads the `fmt ` chunk fields.
fn parse_format(payload: &[u8]) -> ParseResult<FormatChunk> {
    fn incomplete() -> Failure {
        Failure::new("truncated WAV header: fmt chunk is incomplete")
    }

    let tag = le_u16_at(payload, 0).ok_or_else(incomplete)?;
    // The sub-format GUID of a `WAVE_FORMAT_EXTENSIBLE` chunk starts at byte 24; its first two
    // bytes hold the compatibility tag.
    let subformat = if tag == EXTENSIBLE_TAG {
        le_u16_at(payload, 24)
    } else {
        None
    };
    Ok(FormatChunk {
        tag,
        channels: le_u16_at(payload, 2).ok_or_else(incomplete)?,
        sample_rate: le_u32_at(payload, 4).ok_or_else(incomplete)?,
        byte_rate: le_u32_at(payload, 8).ok_or_else(incomplete)?,
        block_align: le_u16_at(payload, 12).ok_or_else(incomplete)?,
        bits_per_sample: le_u16_at(payload, 14).ok_or_else(incomplete)?,
        subformat,
    })
}

/// The name of a WAV sample format.
fn format_name(tag: u16, subformat: Option<u16>) -> &'static str {
    let effective = if tag == EXTENSIBLE_TAG {
        subformat.unwrap_or(tag)
    } else {
        tag
    };
    match effective {
        1 | EXTENSIBLE_TAG => "pcm",
        3 => "float",
        6 => "alaw",
        7 => "mulaw",
        0x11 => "adpcm",
        0x50 => "mpeg",
        _ => "unknown",
    }
}

/// Computes the duration of `data_bytes` at `byte_rate` bytes per second.
fn duration_ms(data_bytes: Option<u32>, byte_rate: u32) -> Option<u64> {
    let bytes = data_bytes?;
    if bytes == UNKNOWN_DATA_SIZE || byte_rate == 0 {
        return None;
    }
    Some(u64::from(bytes) * 1000 / u64::from(byte_rate))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds a `fmt ` payload with `tag` and the standard PCM geometry.
    fn fmt_payload(tag: u16, channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
        let block_align = channels * bits / 8;
        let byte_rate = sample_rate * u32::from(block_align);
        let mut payload = Vec::new();
        payload.extend_from_slice(&tag.to_le_bytes());
        payload.extend_from_slice(&channels.to_le_bytes());
        payload.extend_from_slice(&sample_rate.to_le_bytes());
        payload.extend_from_slice(&byte_rate.to_le_bytes());
        payload.extend_from_slice(&block_align.to_le_bytes());
        payload.extend_from_slice(&bits.to_le_bytes());
        payload
    }

    /// Builds a RIFF/WAVE stream from `(chunk id, payload)` pairs.
    fn wav(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut body = b"WAVE".to_vec();
        for (kind, payload) in chunks {
            body.extend_from_slice(*kind);
            body.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
            body.extend_from_slice(payload);
            if payload.len() % 2 == 1 {
                body.push(0);
            }
        }
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&body);
        bytes
    }

    /// A one-second 44.1 kHz 16-bit stereo PCM WAV.
    fn pcm(channels: u16) -> Vec<u8> {
        wav(&[
            (b"fmt ", fmt_payload(1, channels, 44_100, 16)),
            (b"data", vec![0; 176_400]),
        ])
    }

    #[test]
    fn reads_pcm_format_and_duration() {
        let info = parse(&pcm(2)).unwrap();
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.format_tag, 1);
        assert_eq!(info.format_name, "pcm");
        assert_eq!(info.byte_rate, 176_400);
        assert_eq!(info.block_align, 4);
        assert_eq!(info.data_bytes, Some(176_400));
        assert_eq!(info.duration_ms, Some(1000));
    }

    #[test]
    fn skips_unknown_and_odd_sized_chunks() {
        let info = parse(&wav(&[
            (b"LIST", vec![0; 5]),
            (b"fmt ", fmt_payload(1, 1, 48_000, 16)),
            (b"fact", vec![0; 4]),
            (b"data", vec![0; 96_000]),
        ]))
        .unwrap();
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.duration_ms, Some(1000));
    }

    #[test]
    fn names_the_common_sample_formats() {
        for (tag, expected) in [
            (1, "pcm"),
            (3, "float"),
            (6, "alaw"),
            (7, "mulaw"),
            (99, "unknown"),
        ] {
            assert_eq!(format_name(tag, None), expected, "tag {tag}");
        }
        assert_eq!(format_name(EXTENSIBLE_TAG, Some(3)), "float");
        assert_eq!(format_name(EXTENSIBLE_TAG, None), "pcm");
    }

    #[test]
    fn streaming_data_size_leaves_duration_unknown() {
        let mut bytes = wav(&[(b"fmt ", fmt_payload(1, 2, 44_100, 16))]);
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&UNKNOWN_DATA_SIZE.to_le_bytes());
        let info = parse(&bytes).unwrap();
        assert_eq!(info.data_bytes, Some(UNKNOWN_DATA_SIZE));
        assert_eq!(info.duration_ms, None);
    }

    #[test]
    fn zero_byte_rate_leaves_duration_unknown() {
        let mut payload = fmt_payload(1, 2, 44_100, 16);
        payload[8..12].copy_from_slice(&0u32.to_le_bytes());
        let info = parse(&wav(&[(b"fmt ", payload), (b"data", vec![0; 400])])).unwrap();
        assert_eq!(info.duration_ms, None);
    }

    #[test]
    fn truncated_container_is_reported() {
        assert_eq!(
            parse(b"RIFF\0\0\0\0WAV").unwrap_err().message(),
            "truncated WAV header: RIFF container is incomplete"
        );
        assert_eq!(
            parse(&[]).unwrap_err().message(),
            "truncated WAV header: RIFF container is incomplete"
        );
    }

    #[test]
    fn unsupported_containers_are_reported() {
        assert_eq!(
            parse(b"OGGS\0\0\0\0OGGS").unwrap_err().message(),
            "unsupported audio format: OGGS"
        );
        assert_eq!(
            parse(b"RIFF\0\0\0\0AVI LIST").unwrap_err().message(),
            "unsupported audio format: AVI "
        );
    }

    #[test]
    fn missing_and_truncated_fmt_chunk_are_reported() {
        assert_eq!(
            parse(&wav(&[(b"data", vec![0; 8])])).unwrap_err().message(),
            "truncated WAV header: missing fmt chunk"
        );

        let mut bytes = wav(&[]);
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 0]);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated WAV header: fmt chunk is incomplete"
        );

        let mut bytes = wav(&[]);
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 0, 2, 0]);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated WAV header: fmt chunk is incomplete"
        );
    }

    #[test]
    fn a_chunk_length_without_a_chunk_body_is_reported() {
        let mut bytes = wav(&[]);
        bytes.extend_from_slice(b"fmt ");
        let error = parse(&bytes).unwrap_err();
        assert_eq!(
            error.message(),
            "truncated WAV header: chunk length is incomplete"
        );
    }

    #[test]
    fn reads_the_extensible_subformat_tag() {
        let mut payload = fmt_payload(EXTENSIBLE_TAG, 2, 44_100, 32);
        payload.extend_from_slice(&22u16.to_le_bytes());
        payload.extend_from_slice(&32u16.to_le_bytes());
        payload.extend_from_slice(&3u32.to_le_bytes());
        payload.extend_from_slice(&3u16.to_le_bytes());
        payload.extend_from_slice(&[0; 12]);
        let info = parse(&wav(&[(b"fmt ", payload), (b"data", vec![0; 352_800])])).unwrap();
        assert_eq!(info.format_tag, EXTENSIBLE_TAG);
        assert_eq!(info.format_name, "float");
        assert_eq!(info.duration_ms, Some(1000));
    }

    #[test]
    fn zero_channels_are_rejected() {
        let error = parse(&wav(&[(b"fmt ", fmt_payload(1, 0, 44_100, 16))])).unwrap_err();
        assert_eq!(error.message(), "invalid WAV header: zero channels");
    }
}
