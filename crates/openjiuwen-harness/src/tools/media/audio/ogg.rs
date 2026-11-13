//! OGG header parsing: the first page's Vorbis identification header (`\x01vorbis`) or Opus head
//! (`OpusHead`), giving codec, sample rate and channel count.
//!
//! Duration is best effort: it comes from the granule position of the last page found in the
//! final `TAIL_BYTES` bytes of the file, divided by the sample rate. No packet is decoded.

use serde::Serialize;

use super::super::parse::{
    fourcc, le_i32_at, le_u16_at, le_u32_at, le_u64_at, Failure, ParseResult,
};

/// Bytes at the end of the stream scanned for the final page header.
const TAIL_BYTES: usize = 64 * 1024;

/// Ogg page header length, in bytes, without the segment table.
const PAGE_HEADER_LEN: usize = 27;

/// The granule position value that means "no packet finishes on this page".
const NO_GRANULE: u64 = u64::MAX;

/// Opus always plays back at 48 kHz; the granule position counts those samples.
const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Fields declared by the first OGG page's identification packet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OggInfo {
    /// Codec name: `vorbis` or `opus`.
    pub codec: &'static str,
    /// Sample rate in Hz used for duration: the Vorbis rate, or `48000` for Opus.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u8,
    /// Duration in milliseconds from the final granule position, when it is known.
    pub duration_ms: Option<u64>,
    /// Granule position of the last page, in samples.
    pub granule: Option<u64>,
    /// Vorbis nominal bitrate in bit/s.
    pub nominal_bitrate: Option<i32>,
    /// Opus pre-skip, in 48 kHz samples.
    pub pre_skip: Option<u16>,
    /// Opus input sample rate as declared by the encoder.
    pub input_sample_rate: Option<u32>,
}

/// Parses an OGG header.
///
/// # Errors
///
/// Returns a [`Failure`] when the capture pattern or page header is truncated, when the page
/// version is not `0`, when the segment table or first packet is truncated, when the codec is
/// neither Vorbis nor Opus, or when the identification header is too short.
pub fn parse(bytes: &[u8]) -> ParseResult<OggInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated OGG header: page header is incomplete")
    }

    if !bytes.starts_with(b"OggS") {
        return Err(Failure::new(format!(
            "unsupported audio format: {}",
            fourcc(bytes)
        )));
    }
    let Some(page) = bytes.get(..PAGE_HEADER_LEN) else {
        return Err(incomplete());
    };
    let version = page.get(4).copied().ok_or_else(incomplete)?;
    if version != 0 {
        return Err(Failure::new(format!(
            "invalid OGG header: page version {version}"
        )));
    }
    let segments = usize::from(page.get(26).copied().ok_or_else(incomplete)?);
    let Some(table) = bytes.get(PAGE_HEADER_LEN..PAGE_HEADER_LEN.saturating_add(segments)) else {
        return Err(Failure::new(
            "truncated OGG header: segment table is incomplete",
        ));
    };
    let payload_len = table.iter().fold(0usize, |total, size| {
        total.saturating_add(usize::from(*size))
    });
    let payload_start = PAGE_HEADER_LEN.saturating_add(segments);
    let Some(payload) = bytes.get(payload_start..payload_start.saturating_add(payload_len)) else {
        return Err(Failure::new(
            "truncated OGG header: first packet is incomplete",
        ));
    };
    let mut info = if payload.starts_with(b"\x01vorbis") {
        parse_vorbis(payload)?
    } else if payload.starts_with(b"OpusHead") {
        parse_opus(payload)?
    } else {
        return Err(Failure::new(format!(
            "unsupported OGG codec: {}",
            fourcc(payload)
        )));
    };
    let granule = last_granule(bytes);
    info.granule = granule;
    info.duration_ms = granule
        .filter(|position| *position != 0 && *position != NO_GRANULE)
        .and_then(|position| duration_ms(position, info.sample_rate));
    Ok(info)
}

/// Reads the Vorbis identification header, which starts with `\x01vorbis`.
fn parse_vorbis(payload: &[u8]) -> ParseResult<OggInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated OGG header: Vorbis identification header is incomplete")
    }

    let channels = payload.get(11).copied().ok_or_else(incomplete)?;
    let sample_rate = le_u32_at(payload, 12).ok_or_else(incomplete)?;
    if sample_rate == 0 || channels == 0 {
        return Err(Failure::new(
            "invalid OGG header: zero Vorbis sample rate or channels",
        ));
    }
    Ok(OggInfo {
        codec: "vorbis",
        sample_rate,
        channels,
        duration_ms: None,
        granule: None,
        nominal_bitrate: le_i32_at(payload, 20),
        pre_skip: None,
        input_sample_rate: None,
    })
}

/// Reads the Opus head, which starts with `OpusHead`.
fn parse_opus(payload: &[u8]) -> ParseResult<OggInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated OGG header: Opus head is incomplete")
    }

    let channels = payload.get(9).copied().ok_or_else(incomplete)?;
    let pre_skip = le_u16_at(payload, 10).ok_or_else(incomplete)?;
    let input_sample_rate = le_u32_at(payload, 12).ok_or_else(incomplete)?;
    if channels == 0 {
        return Err(Failure::new("invalid OGG header: zero Opus channel count"));
    }
    Ok(OggInfo {
        codec: "opus",
        sample_rate: OPUS_SAMPLE_RATE,
        channels,
        duration_ms: None,
        granule: None,
        nominal_bitrate: None,
        pre_skip: Some(pre_skip),
        input_sample_rate: Some(input_sample_rate),
    })
}

/// Finds the granule position of the last page header in the final [`TAIL_BYTES`] bytes.
fn last_granule(bytes: &[u8]) -> Option<u64> {
    let start = bytes.len().saturating_sub(TAIL_BYTES);
    let tail = bytes.get(start..)?;
    let mut offset = tail.len().checked_sub(PAGE_HEADER_LEN)?;
    loop {
        if tail.get(offset..offset.checked_add(4)?) == Some(b"OggS".as_slice()) {
            if let Some(granule) = le_u64_at(tail, offset.checked_add(6)?) {
                return Some(granule);
            }
        }
        offset = offset.checked_sub(1)?;
    }
}

/// Converts a granule position to milliseconds.
fn duration_ms(granule: u64, sample_rate: u32) -> Option<u64> {
    if sample_rate == 0 {
        return None;
    }
    Some(granule * 1000 / u64::from(sample_rate))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds one Ogg page carrying a single packet in a single segment.
    fn page(header_type: u8, granule: u64, payload: &[u8]) -> Vec<u8> {
        let mut bytes = b"OggS".to_vec();
        bytes.push(0);
        bytes.push(header_type);
        bytes.extend_from_slice(&granule.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.push(1);
        bytes.push(u8::try_from(payload.len()).unwrap());
        bytes.extend_from_slice(payload);
        bytes
    }

    /// A Vorbis identification payload for `channels` at `sample_rate`.
    fn vorbis(channels: u8, sample_rate: u32, nominal: i32) -> Vec<u8> {
        let mut payload = vec![0x01];
        payload.extend_from_slice(b"vorbis");
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.push(channels);
        payload.extend_from_slice(&sample_rate.to_le_bytes());
        payload.extend_from_slice(&0i32.to_le_bytes());
        payload.extend_from_slice(&nominal.to_le_bytes());
        payload.extend_from_slice(&0i32.to_le_bytes());
        payload.push(0);
        payload.push(1);
        payload
    }

    /// An Opus head payload.
    fn opus(channels: u8, pre_skip: u16, input_rate: u32) -> Vec<u8> {
        let mut payload = b"OpusHead".to_vec();
        payload.push(1);
        payload.push(channels);
        payload.extend_from_slice(&pre_skip.to_le_bytes());
        payload.extend_from_slice(&input_rate.to_le_bytes());
        payload.extend_from_slice(&0i16.to_le_bytes());
        payload.push(0);
        payload
    }

    #[test]
    fn reads_vorbis_identification_and_duration() {
        let mut bytes = page(0, 0, &vorbis(2, 48_000, 128_000));
        bytes.extend_from_slice(&page(4, 48_000, &[0]));
        let info = parse(&bytes).unwrap();
        assert_eq!(info.codec, "vorbis");
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.nominal_bitrate, Some(128_000));
        assert_eq!(info.granule, Some(48_000));
        assert_eq!(info.duration_ms, Some(1000));
        assert_eq!(info.pre_skip, None);
    }

    #[test]
    fn reads_the_opus_head() {
        let mut bytes = page(0, 0, &opus(2, 312, 44_100));
        bytes.extend_from_slice(&page(4, 96_000, &[0]));
        let info = parse(&bytes).unwrap();
        assert_eq!(info.codec, "opus");
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.pre_skip, Some(312));
        assert_eq!(info.input_sample_rate, Some(44_100));
        assert_eq!(info.duration_ms, Some(2000));
    }

    #[test]
    fn a_single_page_leaves_duration_unknown() {
        let info = parse(&page(0, 0, &vorbis(1, 44_100, 0))).unwrap();
        assert_eq!(info.channels, 1);
        assert_eq!(info.granule, Some(0));
        assert_eq!(info.duration_ms, None);
    }

    #[test]
    fn truncated_page_header_is_reported() {
        assert_eq!(
            parse(b"OggS\x00\x00").unwrap_err().message(),
            "truncated OGG header: page header is incomplete"
        );
        assert_eq!(
            parse(b"Ogg").unwrap_err().message(),
            "unsupported audio format: Ogg"
        );
        assert_eq!(
            parse(&[]).unwrap_err().message(),
            "unsupported audio format: <empty>"
        );
    }

    #[test]
    fn an_unsupported_page_version_is_reported() {
        let mut bytes = page(0, 0, &vorbis(1, 44_100, 0));
        bytes[4] = 1;
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "invalid OGG header: page version 1"
        );
    }

    #[test]
    fn truncated_segment_table_and_packet_are_reported() {
        let mut bytes = page(0, 0, &vorbis(1, 44_100, 0));
        bytes.truncate(PAGE_HEADER_LEN + 1);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated OGG header: first packet is incomplete"
        );

        let mut bytes = page(0, 0, &vorbis(1, 44_100, 0));
        bytes[26] = 2;
        bytes.truncate(PAGE_HEADER_LEN + 1);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated OGG header: segment table is incomplete"
        );
    }

    #[test]
    fn an_unsupported_codec_is_reported() {
        let error = parse(&page(0, 0, b"Speex   header")).unwrap_err();
        assert_eq!(error.message(), "unsupported OGG codec: Spee");
    }

    #[test]
    fn truncated_codec_headers_are_reported() {
        let mut payload = vec![0x01];
        payload.extend_from_slice(b"vorbis");
        assert_eq!(
            parse(&page(0, 0, &payload)).unwrap_err().message(),
            "truncated OGG header: Vorbis identification header is incomplete"
        );

        let mut payload = b"OpusHead".to_vec();
        payload.push(1);
        assert_eq!(
            parse(&page(0, 0, &payload)).unwrap_err().message(),
            "truncated OGG header: Opus head is incomplete"
        );
    }

    #[test]
    fn zero_sample_rate_and_channels_are_rejected() {
        assert_eq!(
            parse(&page(0, 0, &vorbis(2, 0, 0))).unwrap_err().message(),
            "invalid OGG header: zero Vorbis sample rate or channels"
        );
        assert_eq!(
            parse(&page(0, 0, &opus(0, 0, 48_000)))
                .unwrap_err()
                .message(),
            "invalid OGG header: zero Opus channel count"
        );
    }

    #[test]
    fn the_last_page_granule_is_used_for_the_duration() {
        let mut bytes = page(0, 0, &vorbis(2, 8_000, 0));
        bytes.extend_from_slice(&page(0, 4_000, &[0]));
        bytes.extend_from_slice(&page(4, 8_000, &[0]));
        assert_eq!(parse(&bytes).unwrap().duration_ms, Some(1000));
    }
}
