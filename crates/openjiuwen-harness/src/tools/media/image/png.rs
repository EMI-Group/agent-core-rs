//! PNG header parsing: the eight-byte signature, the mandatory `IHDR` chunk, and the
//! `tRNS`/`acTL` chunks that reveal palette transparency and animation.
//!
//! Only the header is read: the chunk list is walked for those two hints and parsing stops at
//! `IEND`, the first truncated chunk, or `MAX_CHUNKS` chunks, so no image data is decoded.

use serde::Serialize;

use super::super::parse::{be_u32_at, fourcc, Cursor, Failure, ParseResult};

/// The eight-byte PNG signature.
pub const SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Highest number of chunks the header scan walks before it stops.
const MAX_CHUNKS: usize = 4096;

/// Fields declared by a PNG `IHDR` chunk, plus the optional `acTL`/`tRNS` hints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PngInfo {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bits per sample per channel.
    pub bit_depth: u8,
    /// PNG color type code: `0`, `2`, `3`, `4` or `6`.
    pub color_type: u8,
    /// Name of the color type (`grayscale`, `truecolor`, `indexed`, …).
    pub color_type_name: &'static str,
    /// Whether an alpha channel is present (alpha color type or a `tRNS` chunk).
    pub has_alpha: bool,
    /// Whether the image uses Adam7 interlacing.
    pub interlaced: bool,
    /// Frame count from an `acTL` chunk; `None` for a still image.
    pub frames: Option<u32>,
}

/// Parses a PNG header.
///
/// # Errors
///
/// Returns a [`Failure`] when the signature is truncated or wrong, when the first chunk is not
/// a complete 13-byte `IHDR`, or when the declared width or height is zero.
pub fn parse(bytes: &[u8]) -> ParseResult<PngInfo> {
    let mut cursor = Cursor::new(bytes);
    let Some(signature) = cursor.take(8) else {
        return Err(Failure::new(
            "truncated PNG header: signature is incomplete",
        ));
    };
    if signature != SIGNATURE.as_slice() {
        return Err(Failure::new(format!(
            "unsupported image format: {}",
            fourcc(signature)
        )));
    }
    let mut info = parse_ihdr(&mut cursor)?;
    // Skip the IHDR CRC before walking the remaining chunks.
    let _ = cursor.skip(4);
    let (has_transparency, frames) = scan_chunks(&mut cursor);
    info.has_alpha = info.has_alpha || has_transparency;
    info.frames = frames;
    Ok(info)
}

/// Reads the mandatory leading `IHDR` chunk.
fn parse_ihdr(cursor: &mut Cursor<'_>) -> ParseResult<PngInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated PNG header: IHDR is incomplete")
    }

    let Some(length) = cursor.be_u32() else {
        return Err(Failure::new("truncated PNG header: missing IHDR chunk"));
    };
    let Some(kind) = cursor.take(4) else {
        return Err(Failure::new("truncated PNG header: missing IHDR chunk"));
    };
    if kind != b"IHDR".as_slice() {
        return Err(Failure::new(format!(
            "invalid PNG header: first chunk is {}, expected IHDR",
            fourcc(kind)
        )));
    }
    if length != 13 {
        return Err(Failure::new(format!(
            "invalid PNG header: IHDR length {length}, expected 13"
        )));
    }
    let Some(payload) = cursor.take(13) else {
        return Err(incomplete());
    };
    let width = be_u32_at(payload, 0).ok_or_else(incomplete)?;
    let height = be_u32_at(payload, 4).ok_or_else(incomplete)?;
    let bit_depth = payload.get(8).copied().ok_or_else(incomplete)?;
    let color_type = payload.get(9).copied().ok_or_else(incomplete)?;
    let interlace = payload.get(12).copied().ok_or_else(incomplete)?;
    if width == 0 || height == 0 {
        return Err(Failure::new(format!(
            "invalid PNG header: zero dimension {width}x{height}"
        )));
    }
    Ok(PngInfo {
        width,
        height,
        bit_depth,
        color_type,
        color_type_name: color_type_name(color_type),
        has_alpha: matches!(color_type, 4 | 6),
        interlaced: interlace != 0,
        frames: None,
    })
}

/// The name of a PNG color type.
fn color_type_name(color_type: u8) -> &'static str {
    match color_type {
        0 => "grayscale",
        2 => "truecolor",
        3 => "indexed",
        4 => "grayscale_alpha",
        6 => "truecolor_alpha",
        _ => "unknown",
    }
}

/// Walks the chunk list for a `tRNS` chunk and the `acTL` frame count.
///
/// The scan is best effort: a truncated chunk, a chunk header without its CRC, or the
/// [`MAX_CHUNKS`] budget simply ends it, because the dimensions are already known.
fn scan_chunks(cursor: &mut Cursor<'_>) -> (bool, Option<u32>) {
    let mut has_transparency = false;
    let mut frames = None;
    for _ in 0..MAX_CHUNKS {
        let Some(length) = cursor.be_u32() else {
            break;
        };
        let Ok(length) = usize::try_from(length) else {
            break;
        };
        let Some(kind) = cursor.take(4) else {
            break;
        };
        let Some(payload) = cursor.take(length) else {
            break;
        };
        if kind == b"tRNS".as_slice() {
            has_transparency = true;
        }
        if kind == b"acTL".as_slice() {
            frames = be_u32_at(payload, 0).or(frames);
        }
        if kind == b"IEND".as_slice() {
            break;
        }
        // Skip the CRC; PNG does not need it for a metadata read.
        if !cursor.skip(4) {
            break;
        }
    }
    (has_transparency, frames)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds one chunk with a zero CRC.
    fn chunk(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_be_bytes());
        bytes.extend_from_slice(&kind);
        bytes.extend_from_slice(payload);
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes
    }

    /// Builds an `IHDR` chunk.
    fn ihdr(width: u32, height: u32, bit_depth: u8, color_type: u8) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&width.to_be_bytes());
        payload.extend_from_slice(&height.to_be_bytes());
        payload.extend_from_slice(&[bit_depth, color_type, 0, 0, 0]);
        chunk(*b"IHDR", &payload)
    }

    /// Builds a PNG stream from `chunks`.
    fn png(chunks: &[Vec<u8>]) -> Vec<u8> {
        let mut bytes = SIGNATURE.to_vec();
        for chunk in chunks {
            bytes.extend_from_slice(chunk);
        }
        bytes
    }

    #[test]
    fn reads_ihdr_dimensions_depth_and_color_type() {
        let info = parse(&png(&[ihdr(640, 480, 8, 6)])).unwrap();
        assert_eq!(info.width, 640);
        assert_eq!(info.height, 480);
        assert_eq!(info.bit_depth, 8);
        assert_eq!(info.color_type, 6);
        assert_eq!(info.color_type_name, "truecolor_alpha");
        assert!(info.has_alpha);
        assert!(!info.interlaced);
        assert_eq!(info.frames, None);
    }

    #[test]
    fn indexed_png_gets_alpha_and_frames_from_later_chunks() {
        let mut animation = Vec::new();
        animation.extend_from_slice(&3u32.to_be_bytes());
        animation.extend_from_slice(&0u32.to_be_bytes());
        let info = parse(&png(&[
            ihdr(2, 2, 8, 3),
            chunk(*b"tRNS", &[0, 0, 0]),
            chunk(*b"acTL", &animation),
            chunk(*b"IDAT", &[1, 2, 3]),
            chunk(*b"IEND", &[]),
        ]))
        .unwrap();
        assert_eq!(info.color_type_name, "indexed");
        assert!(info.has_alpha);
        assert_eq!(info.frames, Some(3));
    }

    #[test]
    fn interlaced_flag_is_reported() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&16u32.to_be_bytes());
        payload.extend_from_slice(&9u32.to_be_bytes());
        payload.extend_from_slice(&[1, 0, 0, 0, 1]);
        let info = parse(&png(&[chunk(*b"IHDR", &payload)])).unwrap();
        assert!(info.interlaced);
        assert_eq!(info.bit_depth, 1);
        assert_eq!(info.color_type_name, "grayscale");
    }

    #[test]
    fn truncated_signature_is_reported() {
        let error = parse(&SIGNATURE[..5]).unwrap_err();
        assert_eq!(
            error.message(),
            "truncated PNG header: signature is incomplete"
        );
        assert_eq!(
            parse(&[]).unwrap_err().message(),
            "truncated PNG header: signature is incomplete"
        );
    }

    #[test]
    fn truncated_ihdr_is_reported() {
        let mut bytes = SIGNATURE.to_vec();
        bytes.extend_from_slice(&[0, 0, 0, 13]);
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&[0, 0, 0, 2]);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated PNG header: IHDR is incomplete"
        );

        let header_only = png(&[ihdr(1, 1, 8, 0)]);
        let truncated = &header_only[..12];
        assert_eq!(
            parse(truncated).unwrap_err().message(),
            "truncated PNG header: missing IHDR chunk"
        );
    }

    #[test]
    fn wrong_first_chunk_is_reported() {
        let error = parse(&png(&[chunk(*b"IDAT", &[0])])).unwrap_err();
        assert_eq!(
            error.message(),
            "invalid PNG header: first chunk is IDAT, expected IHDR"
        );
    }

    #[test]
    fn non_ihdr_length_and_zero_dimensions_are_rejected() {
        let error = parse(&png(&[chunk(*b"IHDR", &[0; 12])])).unwrap_err();
        assert_eq!(
            error.message(),
            "invalid PNG header: IHDR length 12, expected 13"
        );

        let error = parse(&png(&[ihdr(0, 10, 8, 0)])).unwrap_err();
        assert_eq!(error.message(), "invalid PNG header: zero dimension 0x10");
    }

    #[test]
    fn a_chunk_that_lies_about_its_length_stops_the_scan() {
        let mut bytes = png(&[ihdr(4, 4, 8, 2)]);
        bytes.extend_from_slice(&[0x00, 0x00, 0xFF, 0xFF]);
        bytes.extend_from_slice(b"tRNS");
        bytes.extend_from_slice(&[0, 0]);
        let info = parse(&bytes).unwrap();
        assert_eq!(info.width, 4);
        assert!(!info.has_alpha);
    }
}
