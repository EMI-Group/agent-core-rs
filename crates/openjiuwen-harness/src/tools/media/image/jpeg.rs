//! JPEG header parsing: a marker walk to the frame header (`SOF`) plus the EXIF orientation
//! tag carried by `APP1`.
//!
//! The walk stops at the first `SOS` (scan) marker, at `EOI`, when the segment budget
//! ([`MAX_SEGMENTS`]) runs out, or when a segment is truncated, so only header bytes are read
//! and the entropy-coded scan data is never touched.

use serde::Serialize;

use super::super::parse::{
    be_u16_at, be_u32_at, fourcc, le_u16_at, le_u32_at, Cursor, Failure, ParseResult,
};

/// The two-byte start-of-image marker.
const SOI: [u8; 2] = [0xFF, 0xD8];

/// Highest number of markers the walk visits before it stops.
const MAX_SEGMENTS: usize = 4096;

/// Highest number of IFD0 entries inspected for the EXIF orientation tag.
const MAX_IFD_ENTRIES: u16 = 512;

/// The EXIF orientation tag id.
const ORIENTATION_TAG: u16 = 0x0112;

/// Fields declared by a JPEG frame header plus the EXIF orientation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JpegInfo {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Component (channel) count: `1` grayscale, `3` YCbCr, `4` CMYK/YCCK.
    pub components: u8,
    /// Sample precision in bits, usually `8`.
    pub precision: u8,
    /// Whether the frame uses progressive DCT (`SOF2`).
    pub progressive: bool,
    /// EXIF orientation (`1`–`8`) when the file carries an `APP1` Exif block.
    pub orientation: Option<u16>,
}

/// A parsed frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Frame {
    /// Sample precision in bits.
    precision: u8,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Component count.
    components: u8,
    /// Whether the frame is progressive.
    progressive: bool,
}

/// Parses a JPEG header.
///
/// # Errors
///
/// Returns a [`Failure`] when the start-of-image marker is missing, when a segment that must be
/// read is truncated or has an impossible length, or when no frame header with a non-zero
/// dimension is found before the scan data starts.
pub fn parse(bytes: &[u8]) -> ParseResult<JpegInfo> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(2) != Some(SOI.as_slice()) {
        return Err(Failure::new(format!(
            "unsupported image format: {}",
            fourcc(bytes)
        )));
    }
    let mut frame: Option<Frame> = None;
    let mut orientation = None;
    for _ in 0..MAX_SEGMENTS {
        let Some(marker) = next_marker(&mut cursor) else {
            break;
        };
        match marker {
            // Standalone markers carry no length.
            0x01 | 0xD0..=0xD7 => {}
            // End of image or start of scan: the header is over.
            0xD9 | 0xDA => break,
            0xE1 => {
                let payload = read_segment(&mut cursor, marker)?;
                if orientation.is_none() {
                    orientation = exif_orientation(payload);
                }
            }
            _ if is_start_of_frame(marker) => {
                let payload = read_segment(&mut cursor, marker)?;
                frame = Some(parse_frame(payload, marker)?);
            }
            _ => {
                let _ = read_segment(&mut cursor, marker)?;
            }
        }
    }
    let Some(frame) = frame else {
        return Err(Failure::new(
            "truncated JPEG header: no frame header (SOF) found",
        ));
    };
    if frame.width == 0 || frame.height == 0 {
        return Err(Failure::new(format!(
            "invalid JPEG header: zero dimension {}x{}",
            frame.width, frame.height
        )));
    }
    Ok(JpegInfo {
        width: frame.width,
        height: frame.height,
        components: frame.components,
        precision: frame.precision,
        progressive: frame.progressive,
        orientation,
    })
}

/// Reads the next marker code, skipping fill bytes and `FF 00` stuffing.
fn next_marker(cursor: &mut Cursor<'_>) -> Option<u8> {
    loop {
        if cursor.u8()? != 0xFF {
            continue;
        }
        let mut marker = cursor.u8()?;
        while marker == 0xFF {
            marker = cursor.u8()?;
        }
        if marker == 0x00 {
            continue;
        }
        return Some(marker);
    }
}

/// Whether `marker` introduces a frame header (`SOF0`–`SOF15`, minus `DHT`, `JPG` and `DAC`).
fn is_start_of_frame(marker: u8) -> bool {
    matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC)
}

/// Reads the payload of a marker segment whose `length` field counts itself.
fn read_segment<'a>(cursor: &mut Cursor<'a>, marker: u8) -> ParseResult<&'a [u8]> {
    let Some(length) = cursor.be_u16() else {
        return Err(Failure::new(format!(
            "truncated JPEG header: segment FF{marker:02X} has no length"
        )));
    };
    if length < 2 {
        return Err(Failure::new(format!(
            "invalid JPEG header: segment FF{marker:02X} length {length}"
        )));
    }
    let size = usize::from(length - 2);
    cursor.take(size).ok_or_else(|| {
        Failure::new(format!(
            "truncated JPEG header: segment FF{marker:02X} is incomplete"
        ))
    })
}

/// Reads the frame header fields.
fn parse_frame(payload: &[u8], marker: u8) -> ParseResult<Frame> {
    fn incomplete() -> Failure {
        Failure::new("truncated JPEG header: frame header is incomplete")
    }

    let precision = payload.first().copied().ok_or_else(incomplete)?;
    let height = be_u16_at(payload, 1).ok_or_else(incomplete)?;
    let width = be_u16_at(payload, 3).ok_or_else(incomplete)?;
    let components = payload.get(5).copied().ok_or_else(incomplete)?;
    Ok(Frame {
        precision,
        width: u32::from(width),
        height: u32::from(height),
        components,
        progressive: matches!(marker, 0xC2 | 0xCA),
    })
}

/// Reads the EXIF orientation tag from an `APP1` payload.
///
/// A payload that is not an `Exif\0\0` TIFF block (for example XMP) yields `None`, as does an
/// orientation tag that is missing, truncated, or of a type other than `SHORT`.
fn exif_orientation(payload: &[u8]) -> Option<u16> {
    let tiff = payload.strip_prefix(b"Exif\0\0")?;
    let little_endian = if tiff.starts_with(b"II") {
        true
    } else if tiff.starts_with(b"MM") {
        false
    } else {
        return None;
    };
    let magic = if little_endian {
        le_u16_at(tiff, 2)
    } else {
        be_u16_at(tiff, 2)
    }?;
    if magic != 42 {
        return None;
    }
    let offset = if little_endian {
        le_u32_at(tiff, 4)?
    } else {
        be_u32_at(tiff, 4)?
    };
    let directory = usize::try_from(offset).ok()?;
    let count = if little_endian {
        le_u16_at(tiff, directory)?
    } else {
        be_u16_at(tiff, directory)?
    };
    for index in 0..count.min(MAX_IFD_ENTRIES) {
        let entry = directory
            .checked_add(2)?
            .checked_add(usize::from(index) * 12)?;
        let tag = if little_endian {
            le_u16_at(tiff, entry)?
        } else {
            be_u16_at(tiff, entry)?
        };
        if tag != ORIENTATION_TAG {
            continue;
        }
        let kind = if little_endian {
            le_u16_at(tiff, entry + 2)?
        } else {
            be_u16_at(tiff, entry + 2)?
        };
        if kind != 3 {
            return None;
        }
        return if little_endian {
            le_u16_at(tiff, entry + 8)
        } else {
            be_u16_at(tiff, entry + 8)
        };
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds a marker segment with `payload`.
    fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
        let length = u16::try_from(payload.len() + 2).unwrap();
        let mut bytes = vec![0xFF, marker];
        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    /// Builds a frame header payload.
    fn frame_payload(precision: u8, width: u16, height: u16, components: u8) -> Vec<u8> {
        let mut payload = vec![precision];
        payload.extend_from_slice(&height.to_be_bytes());
        payload.extend_from_slice(&width.to_be_bytes());
        payload.push(components);
        payload.extend(std::iter::repeat_n(0u8, usize::from(components) * 3));
        payload
    }

    /// Builds a big-endian Exif `APP1` payload carrying `orientation`.
    fn exif_payload(orientation: u16) -> Vec<u8> {
        let mut payload = b"Exif\0\0MM\0*".to_vec();
        payload.extend_from_slice(&8u32.to_be_bytes());
        payload.extend_from_slice(&1u16.to_be_bytes());
        payload.extend_from_slice(&ORIENTATION_TAG.to_be_bytes());
        payload.extend_from_slice(&3u16.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&orientation.to_be_bytes());
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload
    }

    /// A baseline JPEG with one 512x256 frame and the given optional extras.
    fn jpeg(extras: &[Vec<u8>]) -> Vec<u8> {
        let mut bytes = SOI.to_vec();
        for extra in extras {
            bytes.extend_from_slice(extra);
        }
        bytes.extend_from_slice(&segment(0xC0, &frame_payload(8, 512, 256, 3)));
        bytes.extend_from_slice(&[0xFF, 0xD9]);
        bytes
    }

    #[test]
    fn reads_frame_dimensions_and_components() {
        let info = parse(&jpeg(&[])).unwrap();
        assert_eq!(info.width, 512);
        assert_eq!(info.height, 256);
        assert_eq!(info.components, 3);
        assert_eq!(info.precision, 8);
        assert!(!info.progressive);
        assert_eq!(info.orientation, None);
    }

    #[test]
    fn reads_exif_orientation_before_the_frame_header() {
        let info = parse(&jpeg(&[segment(0xE1, &exif_payload(6))])).unwrap();
        assert_eq!(info.orientation, Some(6));
        assert_eq!(info.width, 512);
    }

    #[test]
    fn reads_progressive_frames_and_other_segments() {
        let mut bytes = SOI.to_vec();
        bytes.extend_from_slice(&segment(0xE0, b"JFIF\0"));
        bytes.extend_from_slice(&segment(0xDB, &[0; 65]));
        bytes.extend_from_slice(&segment(0xC2, &frame_payload(12, 40, 30, 1)));
        let info = parse(&bytes).unwrap();
        assert!(info.progressive);
        assert_eq!(info.precision, 12);
        assert_eq!(info.width, 40);
        assert_eq!(info.height, 30);
        assert_eq!(info.components, 1);
    }

    #[test]
    fn exif_is_ignored_when_absent_or_not_orientation() {
        assert_eq!(
            parse(&jpeg(&[segment(0xE1, b"http://ns.adobe.com/xap/")]))
                .unwrap()
                .orientation,
            None
        );
        // The IFD lists an unrelated tag only.
        let mut payload = b"Exif\0\0II*\0".to_vec();
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload.extend_from_slice(&0x010F_u16.to_le_bytes());
        payload.extend_from_slice(&3u16.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&[0, 0, 0, 0]);
        payload.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            parse(&jpeg(&[segment(0xE1, &payload)]))
                .unwrap()
                .orientation,
            None
        );
    }

    #[test]
    fn reads_little_endian_exif_orientation() {
        let mut payload = b"Exif\0\0II*\0".to_vec();
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload.extend_from_slice(&ORIENTATION_TAG.to_le_bytes());
        payload.extend_from_slice(&3u16.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&8u16.to_le_bytes());
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            parse(&jpeg(&[segment(0xE1, &payload)]))
                .unwrap()
                .orientation,
            Some(8)
        );
    }

    #[test]
    fn missing_start_of_image_is_reported() {
        let error = parse(b"not a jpeg at all").unwrap_err();
        assert_eq!(error.message(), "unsupported image format: not ");
        assert_eq!(
            parse(&[]).unwrap_err().message(),
            "unsupported image format: <empty>"
        );
    }

    #[test]
    fn truncated_segment_is_reported() {
        let mut bytes = SOI.to_vec();
        bytes.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00]);
        let error = parse(&bytes).unwrap_err();
        assert_eq!(
            error.message(),
            "truncated JPEG header: segment FFC0 is incomplete"
        );
    }

    #[test]
    fn truncated_length_and_impossible_length_are_reported() {
        let mut bytes = SOI.to_vec();
        bytes.extend_from_slice(&[0xFF, 0xC0, 0x00]);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated JPEG header: segment FFC0 has no length"
        );

        let mut bytes = SOI.to_vec();
        bytes.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x01]);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "invalid JPEG header: segment FFC0 length 1"
        );
    }

    #[test]
    fn a_jpeg_without_a_frame_header_is_reported() {
        let error = parse(&SOI).unwrap_err();
        assert_eq!(
            error.message(),
            "truncated JPEG header: no frame header (SOF) found"
        );
    }

    #[test]
    fn zero_dimension_is_reported() {
        let mut bytes = SOI.to_vec();
        bytes.extend_from_slice(&segment(0xC0, &frame_payload(8, 0, 0, 3)));
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "invalid JPEG header: zero dimension 0x0"
        );
    }
}
