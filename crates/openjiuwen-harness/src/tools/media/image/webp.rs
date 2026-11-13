//! WebP header parsing: the RIFF container, the `VP8X` extended header (canvas size, alpha and
//! animation flags, frame count) and the `VP8 `/`VP8L` bitstream headers for lossy and lossless
//! images.
//!
//! Only chunk headers are walked; the first `VP8X`, `VP8 ` and `VP8L` payloads are kept and the
//! rest is skipped, so no image data is decoded.

use serde::Serialize;

use super::super::parse::{fourcc, le_u16_at, le_u24_at, le_u32_at, Cursor, Failure, ParseResult};

/// Highest number of RIFF chunks the walk visits.
const MAX_CHUNKS: usize = 4096;

/// `VP8X` flag bit 1: the file is an animation.
const FLAG_ANIMATION: u8 = 0x02;

/// `VP8X` flag bit 4: any frame has transparency.
const FLAG_ALPHA: u8 = 0x10;

/// Fields declared by a WebP header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WebpInfo {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Container variant: `extended`, `lossy` or `lossless`.
    pub variant: &'static str,
    /// Whether the alpha flag (or the lossless alpha bit) is set.
    pub has_alpha: bool,
    /// Whether the animation flag is set.
    pub animated: bool,
    /// Number of `ANMF` frames found; `1` for a still image.
    pub frames: u32,
}

/// Parses a WebP header.
///
/// # Errors
///
/// Returns a [`Failure`] when the RIFF container or a chunk length is truncated, when the
/// container is not `RIFF`/`WEBP`, when no `VP8X`, `VP8 ` or `VP8L` chunk carries the image
/// dimensions, or when a bitstream header is malformed.
pub fn parse(bytes: &[u8]) -> ParseResult<WebpInfo> {
    if bytes.len() < 12 {
        return Err(Failure::new(
            "truncated WebP header: RIFF container is incomplete",
        ));
    }
    if !bytes.starts_with(b"RIFF") {
        return Err(Failure::new(format!(
            "unsupported image format: {}",
            fourcc(bytes)
        )));
    }
    if bytes.get(8..12) != Some(b"WEBP".as_slice()) {
        let kind = bytes.get(8..).unwrap_or_default();
        return Err(Failure::new(format!(
            "unsupported image format: {}",
            fourcc(kind)
        )));
    }
    let mut cursor = Cursor::new(bytes);
    let _ = cursor.seek(12);
    let mut extended: Option<&[u8]> = None;
    let mut lossy: Option<&[u8]> = None;
    let mut lossless: Option<&[u8]> = None;
    let mut frames = 0u32;
    for _ in 0..MAX_CHUNKS {
        let Some(kind) = cursor.take(4) else {
            break;
        };
        let Some(size) = cursor.le_u32() else {
            return Err(Failure::new(
                "truncated WebP header: chunk length is incomplete",
            ));
        };
        let Ok(size) = usize::try_from(size) else {
            break;
        };
        let Some(payload) = cursor.take(size) else {
            return Err(Failure::new(format!(
                "truncated WebP header: {} chunk is incomplete",
                fourcc(kind)
            )));
        };
        if kind == b"VP8X".as_slice() {
            extended = extended.or(Some(payload));
        } else if kind == b"VP8 ".as_slice() {
            lossy = lossy.or(Some(payload));
        } else if kind == b"VP8L".as_slice() {
            lossless = lossless.or(Some(payload));
        } else if kind == b"ANMF".as_slice() {
            frames = frames.saturating_add(1);
        }
        // RIFF pads every chunk to an even length.
        if size % 2 == 1 && !cursor.skip(1) {
            break;
        }
    }
    if let Some(payload) = extended {
        return parse_extended(payload, frames);
    }
    if let Some(payload) = lossless {
        return parse_lossless(payload);
    }
    if let Some(payload) = lossy {
        return parse_lossy(payload);
    }
    Err(Failure::new(
        "truncated WebP header: no VP8X, VP8 or VP8L chunk found",
    ))
}

/// Reads the extended (`VP8X`) header: canvas size, flags and frame count.
fn parse_extended(payload: &[u8], frames: u32) -> ParseResult<WebpInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated WebP header: VP8X chunk is incomplete")
    }

    let flags = payload.first().copied().ok_or_else(incomplete)?;
    let width = le_u24_at(payload, 4)
        .ok_or_else(incomplete)?
        .saturating_add(1);
    let height = le_u24_at(payload, 7)
        .ok_or_else(incomplete)?
        .saturating_add(1);
    Ok(WebpInfo {
        width,
        height,
        variant: "extended",
        has_alpha: flags & FLAG_ALPHA != 0,
        animated: flags & FLAG_ANIMATION != 0,
        frames: frames.max(1),
    })
}

/// Reads a lossless (`VP8L`) bitstream header.
fn parse_lossless(payload: &[u8]) -> ParseResult<WebpInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated WebP header: VP8L chunk is incomplete")
    }

    if payload.first() != Some(&0x2F) {
        return Err(Failure::new("invalid WebP header: bad VP8L signature"));
    }
    let bits = le_u32_at(payload, 1).ok_or_else(incomplete)?;
    let width = (bits & 0x3FFF).saturating_add(1);
    let height = ((bits >> 14) & 0x3FFF).saturating_add(1);
    Ok(WebpInfo {
        width,
        height,
        variant: "lossless",
        has_alpha: (bits >> 28) & 0x01 == 1,
        animated: false,
        frames: 1,
    })
}

/// Reads a lossy (`VP8 `) bitstream header.
fn parse_lossy(payload: &[u8]) -> ParseResult<WebpInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated WebP header: VP8 chunk is incomplete")
    }

    let start_code = payload.get(3..6);
    if start_code != Some([0x9D, 0x01, 0x2A].as_slice()) {
        return Err(Failure::new("invalid WebP header: bad VP8 start code"));
    }
    let width = u32::from(le_u16_at(payload, 6).ok_or_else(incomplete)?) & 0x3FFF;
    let height = u32::from(le_u16_at(payload, 8).ok_or_else(incomplete)?) & 0x3FFF;
    if width == 0 || height == 0 {
        return Err(Failure::new(format!(
            "invalid WebP header: zero dimension {width}x{height}"
        )));
    }
    Ok(WebpInfo {
        width,
        height,
        variant: "lossy",
        has_alpha: false,
        animated: false,
        frames: 1,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds a RIFF/WebP container from `(fourcc, payload)` chunks.
    fn riff(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut body = b"WEBP".to_vec();
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

    /// A lossy `VP8 ` bitstream header payload.
    fn lossy_payload(width: u16, height: u16) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0x9D, 0x01, 0x2A];
        payload.extend_from_slice(&width.to_le_bytes());
        payload.extend_from_slice(&height.to_le_bytes());
        payload
    }

    /// An extended `VP8X` payload with `flags` and a `width`x`height` canvas.
    fn extended_payload(flags: u8, width: u32, height: u32) -> Vec<u8> {
        let mut payload = vec![flags, 0, 0, 0];
        payload.extend_from_slice(&(width - 1).to_le_bytes()[..3]);
        payload.extend_from_slice(&(height - 1).to_le_bytes()[..3]);
        payload
    }

    #[test]
    fn reads_lossy_dimensions() {
        let info = parse(&riff(&[(b"VP8 ", lossy_payload(320, 240))])).unwrap();
        assert_eq!(info.width, 320);
        assert_eq!(info.height, 240);
        assert_eq!(info.variant, "lossy");
        assert!(!info.has_alpha);
        assert!(!info.animated);
        assert_eq!(info.frames, 1);
    }

    #[test]
    fn reads_lossless_dimensions_and_alpha() {
        let bits = 0x63u32 | (0x31u32 << 14) | (1u32 << 28);
        let mut payload = vec![0x2F];
        payload.extend_from_slice(&bits.to_le_bytes());
        let info = parse(&riff(&[(b"VP8L", payload)])).unwrap();
        assert_eq!(info.width, 100);
        assert_eq!(info.height, 50);
        assert_eq!(info.variant, "lossless");
        assert!(info.has_alpha);
        assert_eq!(info.frames, 1);
    }

    #[test]
    fn reads_extended_canvas_flags_and_frames() {
        let mut chunks: Vec<(&[u8; 4], Vec<u8>)> = vec![(
            b"VP8X",
            extended_payload(FLAG_ALPHA | FLAG_ANIMATION, 64, 32),
        )];
        chunks.push((b"ANMF", vec![0; 16]));
        chunks.push((b"ANMF", vec![0; 16]));
        let info = parse(&riff(&chunks)).unwrap();
        assert_eq!(info.width, 64);
        assert_eq!(info.height, 32);
        assert_eq!(info.variant, "extended");
        assert!(info.has_alpha);
        assert!(info.animated);
        assert_eq!(info.frames, 2);
    }

    #[test]
    fn truncated_container_is_reported() {
        assert_eq!(
            parse(b"RIFF\0\0\0\0WEB").unwrap_err().message(),
            "truncated WebP header: RIFF container is incomplete"
        );
        assert_eq!(
            parse(&[]).unwrap_err().message(),
            "truncated WebP header: RIFF container is incomplete"
        );
    }

    #[test]
    fn unknown_magic_is_reported() {
        assert_eq!(
            parse(b"NOTARIFF AT ALL").unwrap_err().message(),
            "unsupported image format: NOTA"
        );
        let error = parse(b"RIFF\0\0\0\0WAVEfmt ").unwrap_err();
        assert_eq!(error.message(), "unsupported image format: WAVE");
    }

    #[test]
    fn truncated_chunk_is_reported() {
        let mut bytes = riff(&[(b"VP8 ", vec![1, 2, 3, 4])]);
        bytes.truncate(bytes.len() - 1);
        let error = parse(&bytes).unwrap_err();
        assert_eq!(
            error.message(),
            "truncated WebP header: VP8  chunk is incomplete"
        );
    }

    #[test]
    fn missing_dimension_chunk_is_reported() {
        let error = parse(&riff(&[(b"ICCP", vec![0; 4])])).unwrap_err();
        assert_eq!(
            error.message(),
            "truncated WebP header: no VP8X, VP8 or VP8L chunk found"
        );
    }

    #[test]
    fn malformed_bitstream_headers_are_reported() {
        let error = parse(&riff(&[(b"VP8 ", vec![0; 10])])).unwrap_err();
        assert_eq!(error.message(), "invalid WebP header: bad VP8 start code");

        let error = parse(&riff(&[(b"VP8L", vec![0x2E, 0, 0, 0, 0])])).unwrap_err();
        assert_eq!(error.message(), "invalid WebP header: bad VP8L signature");

        let info = parse(&riff(&[(b"VP8L", vec![0x2F, 0])])).unwrap_err();
        assert_eq!(
            info.message(),
            "truncated WebP header: VP8L chunk is incomplete"
        );

        let error = parse(&riff(&[(b"VP8 ", vec![0, 0, 0, 0x9D, 0x01, 0x2A])])).unwrap_err();
        assert_eq!(
            error.message(),
            "truncated WebP header: VP8 chunk is incomplete"
        );
    }

    #[test]
    fn zero_lossy_dimensions_are_rejected() {
        let error = parse(&riff(&[(b"VP8 ", lossy_payload(0, 0))])).unwrap_err();
        assert_eq!(error.message(), "invalid WebP header: zero dimension 0x0");
    }
}
