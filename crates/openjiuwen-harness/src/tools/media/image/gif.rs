//! GIF header parsing: the logical screen descriptor plus a best-effort block walk that counts
//! image descriptors (frames) and detects the transparency flag of a graphic control extension.
//!
//! The walk stops at the trailer (`0x3B`), at the first truncated block, or when the block
//! budget ([`MAX_BLOCKS`]) runs out; LZW data is skipped, never decoded.

use serde::Serialize;

use super::super::parse::{fourcc, le_u16_at, Cursor, Failure, ParseResult};

/// Highest number of top-level blocks the frame walk visits.
const MAX_BLOCKS: usize = 65_536;

/// Highest number of data sub-blocks one image or extension may contain.
const MAX_SUB_BLOCKS: usize = 65_536;

/// The graphic control extension label.
const GRAPHIC_CONTROL_LABEL: u8 = 0xF9;

/// Fields declared by a GIF logical screen descriptor, plus the animation and transparency
/// hints found by the block walk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GifInfo {
    /// Canvas width in pixels.
    pub width: u16,
    /// Canvas height in pixels.
    pub height: u16,
    /// Declared version: `GIF87a` or `GIF89a`.
    pub version: &'static str,
    /// Number of image descriptors found, i.e. the frame count.
    pub frames: u32,
    /// Whether more than one frame is present.
    pub animated: bool,
    /// Whether a graphic control extension declares a transparent color.
    pub has_alpha: bool,
    /// Whether the logical screen descriptor carries a global color table.
    pub global_color_table: bool,
    /// Bits per primary color in the global color table, `1`–`8`.
    pub color_resolution: u8,
}

/// Parses a GIF header.
///
/// # Errors
///
/// Returns a [`Failure`] when the signature or logical screen descriptor is truncated, when the
/// version is not `GIF87a`/`GIF89a`, when a declared global color table is truncated, or when the
/// declared width or height is zero.
pub fn parse(bytes: &[u8]) -> ParseResult<GifInfo> {
    fn incomplete() -> Failure {
        Failure::new("truncated GIF header: logical screen descriptor is incomplete")
    }

    let mut cursor = Cursor::new(bytes);
    let Some(signature) = cursor.take(6) else {
        return Err(Failure::new(
            "truncated GIF header: signature is incomplete",
        ));
    };
    let version = if signature == b"GIF87a".as_slice() {
        "GIF87a"
    } else if signature == b"GIF89a".as_slice() {
        "GIF89a"
    } else {
        return Err(Failure::new(format!(
            "unsupported image format: {}",
            fourcc(signature)
        )));
    };
    let Some(descriptor) = cursor.take(7) else {
        return Err(incomplete());
    };
    let width = le_u16_at(descriptor, 0).ok_or_else(incomplete)?;
    let height = le_u16_at(descriptor, 2).ok_or_else(incomplete)?;
    let packed = descriptor.get(4).copied().ok_or_else(incomplete)?;
    if width == 0 || height == 0 {
        return Err(Failure::new(format!(
            "invalid GIF header: zero dimension {width}x{height}"
        )));
    }
    let global_color_table = packed & 0x80 != 0;
    let color_resolution = ((packed >> 4) & 0x07) + 1;
    if global_color_table && !cursor.skip(color_table_bytes(packed)) {
        return Err(Failure::new(
            "truncated GIF header: global color table is incomplete",
        ));
    }
    let (frames, has_alpha) = scan_blocks(&mut cursor);
    Ok(GifInfo {
        width,
        height,
        version,
        frames,
        animated: frames > 1,
        has_alpha,
        global_color_table,
        color_resolution,
    })
}

/// Byte length of a color table whose size bits are `packed & 0x07`.
fn color_table_bytes(packed: u8) -> usize {
    3 * (1usize << ((packed & 0x07) + 1))
}

/// Walks the block list for image descriptors and the transparency flag.
fn scan_blocks(cursor: &mut Cursor<'_>) -> (u32, bool) {
    let mut frames = 0u32;
    let mut has_alpha = false;
    for _ in 0..MAX_BLOCKS {
        let Some(introducer) = cursor.u8() else {
            break;
        };
        match introducer {
            // Extension introducer followed by a label.
            0x21 => {
                let Some(label) = cursor.u8() else {
                    break;
                };
                let advanced = if label == GRAPHIC_CONTROL_LABEL {
                    read_graphic_control(cursor, &mut has_alpha)
                } else {
                    skip_sub_blocks(cursor)
                };
                if !advanced {
                    break;
                }
            }
            // Image descriptor.
            0x2C => {
                frames = frames.saturating_add(1);
                if !skip_image(cursor) {
                    break;
                }
            }
            // Trailer (`0x3B`) or an unknown block ends the walk.
            _ => break,
        }
    }
    (frames, has_alpha)
}

/// Reads a graphic control extension, recording its transparency flag.
fn read_graphic_control(cursor: &mut Cursor<'_>, has_alpha: &mut bool) -> bool {
    let Some(size) = cursor.u8() else {
        return false;
    };
    let Some(block) = cursor.take(usize::from(size)) else {
        return false;
    };
    if block.first().is_some_and(|packed| packed & 0x01 != 0) {
        *has_alpha = true;
    }
    cursor.u8() == Some(0)
}

/// Skips an image descriptor, its local color table and its LZW data.
fn skip_image(cursor: &mut Cursor<'_>) -> bool {
    let Some(descriptor) = cursor.take(9) else {
        return false;
    };
    let Some(packed) = descriptor.get(8).copied() else {
        return false;
    };
    if packed & 0x80 != 0 && !cursor.skip(color_table_bytes(packed)) {
        return false;
    }
    // LZW minimum code size, then the compressed sub-blocks.
    cursor.u8().is_some() && skip_sub_blocks(cursor)
}

/// Skips a run of length-prefixed sub-blocks up to the zero terminator.
fn skip_sub_blocks(cursor: &mut Cursor<'_>) -> bool {
    for _ in 0..MAX_SUB_BLOCKS {
        let Some(size) = cursor.u8() else {
            return false;
        };
        if size == 0 {
            return true;
        }
        if !cursor.skip(usize::from(size)) {
            return false;
        }
    }
    false
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds a GIF stream with a logical screen descriptor and `body` blocks.
    fn gif(width: u16, height: u16, packed: u8, body: &[u8]) -> Vec<u8> {
        let mut bytes = b"GIF89a".to_vec();
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.push(packed);
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(body);
        bytes
    }

    /// A graphic control extension, optionally declaring a transparent color.
    fn graphic_control(transparent: bool) -> Vec<u8> {
        vec![
            0x21,
            GRAPHIC_CONTROL_LABEL,
            0x04,
            u8::from(transparent),
            0x0A,
            0x00,
            0x00,
            0x00,
        ]
    }

    /// A 1x1 image descriptor with one data sub-block.
    fn image_block() -> Vec<u8> {
        let mut bytes = vec![0x2C];
        bytes.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0, 0x00]);
        bytes.push(0x02);
        bytes.extend_from_slice(&[0x02, 0x4C, 0x01]);
        bytes.push(0x00);
        bytes
    }

    /// An animated GIF body with `frames` images and a trailer.
    fn frames(frames: usize) -> Vec<u8> {
        let mut body = graphic_control(true);
        for _ in 0..frames {
            body.extend_from_slice(&image_block());
        }
        body.push(0x3B);
        body
    }

    #[test]
    fn reads_screen_descriptor_and_one_frame() {
        let bytes = gif(640, 480, 0x70, &frames(1));
        let info = parse(&bytes).unwrap();
        assert_eq!(info.width, 640);
        assert_eq!(info.height, 480);
        assert_eq!(info.version, "GIF89a");
        assert_eq!(info.frames, 1);
        assert!(!info.animated);
        assert!(info.has_alpha);
        assert!(!info.global_color_table);
        assert_eq!(info.color_resolution, 8);
    }

    #[test]
    fn counts_animation_frames() {
        let info = parse(&gif(2, 2, 0x00, &frames(3))).unwrap();
        assert_eq!(info.frames, 3);
        assert!(info.animated);
    }

    #[test]
    fn opaque_gif_has_no_alpha() {
        let mut body = graphic_control(false);
        body.extend_from_slice(&image_block());
        body.push(0x3B);
        assert!(!parse(&gif(1, 1, 0, &body)).unwrap().has_alpha);
    }

    #[test]
    fn global_color_table_is_skipped() {
        let mut bytes = b"GIF87a".to_vec();
        bytes.extend_from_slice(&[3, 0, 3, 0, 0x80, 0, 0]);
        // Two color entries of three bytes each.
        bytes.extend_from_slice(&[0; 6]);
        bytes.extend_from_slice(&frames(1));
        let info = parse(&bytes).unwrap();
        assert!(info.global_color_table);
        assert_eq!(info.color_resolution, 1);
        assert_eq!(info.frames, 1);
        assert_eq!(info.version, "GIF87a");
    }

    #[test]
    fn truncated_signature_is_reported() {
        assert_eq!(
            parse(b"GIF8").unwrap_err().message(),
            "truncated GIF header: signature is incomplete"
        );
        assert_eq!(
            parse(b"NOTGIF").unwrap_err().message(),
            "unsupported image format: NOTG"
        );
        assert_eq!(
            parse(&[]).unwrap_err().message(),
            "truncated GIF header: signature is incomplete"
        );
    }

    #[test]
    fn truncated_screen_descriptor_is_reported() {
        let mut bytes = b"GIF89a".to_vec();
        bytes.extend_from_slice(&[1, 0, 1]);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated GIF header: logical screen descriptor is incomplete"
        );
    }

    #[test]
    fn truncated_global_color_table_is_reported() {
        let mut bytes = b"GIF89a".to_vec();
        bytes.extend_from_slice(&[1, 0, 1, 0, 0x80, 0, 0]);
        bytes.extend_from_slice(&[0; 3]);
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated GIF header: global color table is incomplete"
        );
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        assert_eq!(
            parse(&gif(0, 4, 0, &[])).unwrap_err().message(),
            "invalid GIF header: zero dimension 0x4"
        );
    }

    #[test]
    fn truncated_frame_data_keeps_the_frame_count() {
        let mut body = image_block();
        body.truncate(body.len() - 1);
        let info = parse(&gif(1, 1, 0, &body)).unwrap();
        assert_eq!(info.frames, 1);
        assert!(!info.animated);
    }
}
