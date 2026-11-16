//! Picture headers: what format a picture file is and how big it is.
//!
//! Python's `read` check opens a picture with Pillow, turns it into a model
//! image part and sends it to the Judge. This port bundles neither Pillow nor an
//! image decoder, so a picture check reports the file's **header** instead: its
//! format, its pixel size and its byte size. Nothing is decoded, nothing is
//! re-encoded and no pixels reach a model, which the rendered result says
//! explicitly rather than implying the Judge sees the picture.
//!
//! The parsers are dependency-free and only read what the format's own header
//! states:
//!
//! - PNG — the `IHDR` chunk (width and height, both big-endian `u32`);
//! - JPEG — the first frame header (`SOFn`), reached by walking the segment
//!   lengths from `SOI` past `APPn`/`DQT`/`DHT`/… (height and width, big-endian);
//! - GIF — the logical screen descriptor, plus a walk of the block stream to
//!   count frames (so an animation is reported as such);
//! - WebP — the `VP8 `, `VP8L` and `VP8X` chunk headers of a `RIFF` container;
//! - BMP — the `BITMAPINFOHEADER` (or the older 12-byte core header), with a
//!   bottom-up (`positive`) and top-down (`negative`) height both read.
//!
//! Anything else — an unknown magic, a truncated header, a zero dimension — is
//! not a readable picture, which is exactly what Python's `picture_or_none`
//! answers, so [`CheckOutcome::ok`](crate::reward::CheckOutcome::ok) carries the
//! text and no error is raised.
//!
//! Python's send limits ([`PICTURE_SIDE`], [`PICTURE_BYTES`] and the formats a
//! vision API takes as they are, [`MODEL_PICTURE_FORMATS`]) are reported as
//! information: [`PictureInfo::rewrite_reasons`] names why Python would
//! re-encode the file as a JPEG, and the rendered result says that this port did
//! not convert anything.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

/// The longest side, in pixels, of a picture Python sends to a model.
pub const PICTURE_SIDE: u32 = 1600;

/// The largest picture file, in bytes, Python sends to a model unchanged.
pub const PICTURE_BYTES: usize = 3_000_000;

/// The formats a vision API takes as they are (Python `MODEL_FORMATS`).
pub const MODEL_PICTURE_FORMATS: [PictureFormat; 4] = [
    PictureFormat::Png,
    PictureFormat::Jpeg,
    PictureFormat::Gif,
    PictureFormat::Webp,
];

/// What Python's `read` check answers for a file that is not a usable picture.
pub const UNREADABLE_PICTURE: &str = "not a readable picture";

/// The eight bytes every PNG starts with.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// The picture formats this reader recognises.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PictureFormat {
    /// A PNG (`IHDR`).
    Png,
    /// A JPEG (`SOFn`).
    Jpeg,
    /// A GIF (`GIF87a`/`GIF89a`).
    Gif,
    /// A WebP (`VP8 `/`VP8L`/`VP8X` in a `RIFF` container).
    Webp,
    /// A BMP (`BITMAPINFOHEADER` or the 12-byte core header).
    Bmp,
    /// A file whose header names no format this reader knows.
    #[default]
    Unknown,
}

impl PictureFormat {
    /// The spelling a vision API and Python's Pillow use (`PNG`, `JPEG`, …).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Gif => "GIF",
            Self::Webp => "WEBP",
            Self::Bmp => "BMP",
            Self::Unknown => "unknown",
        }
    }

    /// The format a file's first bytes name, or [`Unknown`](Self::Unknown).
    #[must_use]
    pub fn sniff(bytes: &[u8]) -> Self {
        if bytes.starts_with(&PNG_SIGNATURE) {
            Self::Png
        } else if bytes.starts_with(&[0xff, 0xd8]) {
            Self::Jpeg
        } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
            Self::Gif
        } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP".as_slice()) {
            Self::Webp
        } else if bytes.starts_with(b"BM") {
            Self::Bmp
        } else {
            Self::Unknown
        }
    }
}

/// What a picture file's header says: its format, its pixel size and its bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PictureInfo {
    /// The format the header names.
    pub format: PictureFormat,
    /// The width in pixels, as the header states it.
    pub width: u32,
    /// The height in pixels, as the header states it.
    pub height: u32,
    /// The size of the whole file in bytes.
    pub bytes: usize,
    /// `true` for a GIF or WebP whose block stream holds more than one frame.
    pub animated: bool,
}

impl PictureInfo {
    /// Reads the header of `bytes`, or `None` when it is not a readable picture.
    ///
    /// A truncated or corrupt header, an unknown format and a zero dimension all
    /// answer `None`, mirroring Python's `picture_or_none`.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let format = PictureFormat::sniff(bytes);
        let (width, height, animated) = match format {
            PictureFormat::Png => {
                let (width, height) = png_size(bytes)?;
                (width, height, false)
            }
            PictureFormat::Jpeg => {
                let (width, height) = jpeg_size(bytes)?;
                (width, height, false)
            }
            PictureFormat::Gif => gif_size(bytes)?,
            PictureFormat::Webp => webp_size(bytes)?,
            PictureFormat::Bmp => {
                let (width, height) = bmp_size(bytes)?;
                (width, height, false)
            }
            PictureFormat::Unknown => return None,
        };
        if width == 0 || height == 0 {
            return None;
        }
        Some(Self {
            format,
            width,
            height,
            bytes: bytes.len(),
            animated,
        })
    }

    /// The longer of the two sides, in pixels.
    #[must_use]
    pub fn longest_side(&self) -> u32 {
        self.width.max(self.height)
    }

    /// Why Python would re-encode this file as a JPEG before a model sees it.
    ///
    /// The reasons are Python's own send rule (`picture_data`): a format a vision
    /// API does not take as it is, a file above [`PICTURE_BYTES`], or a longest
    /// side above [`PICTURE_SIDE`]. Empty means Python would send the file as it
    /// is. Nothing here converts anything.
    #[must_use]
    pub fn rewrite_reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if !MODEL_PICTURE_FORMATS.contains(&self.format) {
            reasons.push(format!(
                "{} is not a format a model takes as it is",
                self.format.as_str()
            ));
        }
        if self.bytes > PICTURE_BYTES {
            reasons.push(format!("the file is larger than {PICTURE_BYTES} bytes"));
        }
        if self.longest_side() > PICTURE_SIDE {
            reasons.push(format!(
                "its longest side ({} px) is longer than {PICTURE_SIDE} px",
                self.longest_side()
            ));
        }
        reasons
    }

    /// The format as it is shown to the Judge (`PNG`, `animated GIF`, …).
    #[must_use]
    pub fn render_format(&self) -> String {
        if self.animated {
            format!("animated {}", self.format.as_str())
        } else {
            self.format.as_str().to_owned()
        }
    }

    /// The result a picture `read` check reports.
    ///
    /// The header, not the pixels: the third line says which of the two the Judge
    /// sees, so the deviation from Python (which sends the picture itself) is
    /// stated rather than implied.
    #[must_use]
    pub fn evidence(&self, path: &str) -> String {
        let mut text = format!(
            "path: {path}\npicture file ({width}x{height} {format}, {bytes} bytes)\n\
             the judge sees the picture's metadata (format and size), not its pixels",
            width = self.width,
            height = self.height,
            format = self.render_format(),
            bytes = self.bytes,
        );
        let reasons = self.rewrite_reasons();
        if !reasons.is_empty() {
            let _ = write!(
                text,
                "\nnote: Python would re-encode it as a JPEG with a longest side of {PICTURE_SIDE} px \
                 before the judge sees it, because {}; this port does not re-encode, so nothing was \
                 converted",
                reasons.join(" and ")
            );
        }
        text
    }
}

/// Reads a big-endian `u32` at `start`.
fn be_u32(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_be_bytes(
        bytes.get(start..start + 4)?.try_into().ok()?,
    ))
}

/// Reads a big-endian `u16` at `start`.
fn be_u16(bytes: &[u8], start: usize) -> Option<u16> {
    Some(u16::from_be_bytes(
        bytes.get(start..start + 2)?.try_into().ok()?,
    ))
}

/// Reads a little-endian `u16` at `start`.
fn le_u16(bytes: &[u8], start: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(start..start + 2)?.try_into().ok()?,
    ))
}

/// Reads a little-endian `u32` at `start`.
fn le_u32(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(start..start + 4)?.try_into().ok()?,
    ))
}

/// Reads a little-endian 24-bit unsigned integer at `start`.
fn le_u24(bytes: &[u8], start: usize) -> Option<u32> {
    let raw = bytes.get(start..start + 3)?;
    Some(u32::from(raw[0]) | (u32::from(raw[1]) << 8) | (u32::from(raw[2]) << 16))
}

/// The width and height of a PNG's `IHDR` chunk.
fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    // The signature (8 bytes), the chunk length (4) and the chunk type (4).
    if be_u32(bytes, 8)? != 13 || bytes.get(12..16)? != b"IHDR" {
        return None;
    }
    Some((be_u32(bytes, 16)?, be_u32(bytes, 20)?))
}

/// The width and height of a JPEG's first frame header, walking inter-marker segments.
fn jpeg_size(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut index = 2_usize; // past SOI
    while index < bytes.len() {
        if bytes[index] != 0xff {
            index += 1;
            continue;
        }
        let mut marker_index = index + 1;
        while bytes.get(marker_index) == Some(&0xff) {
            marker_index += 1; // fill bytes before the marker
        }
        let marker = *bytes.get(marker_index)?;
        index = marker_index + 1;
        match marker {
            // Stuffed byte, SOI and the standalone markers (RSTn, TEM) carry no length.
            0x00 | 0x01 | 0xd0..=0xd8 => continue,
            // Entropy-coded data or the end of the image: no frame header was found.
            0xd9 | 0xda => return None,
            _ => {}
        }
        let length = usize::from(be_u16(bytes, index)?);
        if length < 2 {
            return None;
        }
        // SOF0..SOF15, minus DHT (c4), JPG (c8) and DAC (cc), which share the range.
        let frame_header = matches!(marker, 0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf);
        if frame_header {
            let payload = bytes.get(index + 2..index + length)?;
            if payload.len() < 5 {
                return None;
            }
            let height = u32::from(be_u16(payload, 1)?);
            let width = u32::from(be_u16(payload, 3)?);
            return Some((width, height));
        }
        index = index.checked_add(length)?;
    }
    None
}

/// The GIF logical screen size and whether the block stream holds more than one frame.
fn gif_size(bytes: &[u8]) -> Option<(u32, u32, bool)> {
    // The logical screen descriptor follows the 6-byte header.
    let width = u32::from(le_u16(bytes, 6)?);
    let height = u32::from(le_u16(bytes, 8)?);
    let packed = *bytes.get(10)?;
    let mut index = 13_usize; // header (6) + logical screen descriptor (7)
    if packed & 0x80 != 0 {
        // A global colour table of 3 * 2^(packed & 0x07) * 3-bit entries.
        index = index.checked_add(3 * (2_usize << (packed & 0x07)))?;
    }
    let mut frames = 0_usize;
    while index < bytes.len() {
        match bytes[index] {
            0x2c => {
                frames += 1;
                let descriptor = bytes.get(index + 1..index + 10)?;
                let local = descriptor[8];
                index = index.checked_add(10)?;
                if local & 0x80 != 0 {
                    index = index.checked_add(3 * (2_usize << (local & 0x07)))?;
                }
                // The LZW minimum code size, then the data sub-blocks.
                index = skip_sub_blocks(bytes, index.checked_add(1)?)?;
            }
            0x21 => {
                // An extension: its label byte, then the data sub-blocks.
                index = skip_sub_blocks(bytes, index.checked_add(2)?)?;
            }
            0x3b => break, // trailer
            _ => break,    // an unknown block: stop counting rather than guess
        }
    }
    Some((width, height, frames > 1))
}

/// Skips a GIF data sub-block chain starting at `index`, returning the next block.
fn skip_sub_blocks(bytes: &[u8], mut index: usize) -> Option<usize> {
    loop {
        let size = usize::from(*bytes.get(index)?);
        index = index.checked_add(1)?;
        if size == 0 {
            return Some(index);
        }
        index = index.checked_add(size)?;
        if index > bytes.len() {
            return None;
        }
    }
}

/// The canvas size of a WebP container, read from its first `VP8 `, `VP8L` or `VP8X` chunk.
fn webp_size(bytes: &[u8]) -> Option<(u32, u32, bool)> {
    let fourcc: [u8; 4] = bytes.get(12..16)?.try_into().ok()?;
    let payload = bytes.get(20..)?; // past the chunk's own fourcc and length
    match &fourcc {
        b"VP8 " => {
            let frame_tag = *payload.first()?;
            if frame_tag & 0x01 != 0 {
                return None; // not a key frame: it carries no picture
            }
            match payload.get(3..6)? {
                [0x9d, 0x01, 0x2a] => {}
                _ => return None, // the start code every lossy frame carries
            }
            let width = u32::from(le_u16(payload, 6)?) & 0x3fff;
            let height = u32::from(le_u16(payload, 8)?) & 0x3fff;
            Some((width, height, false))
        }
        b"VP8L" => {
            if *payload.first()? != 0x2f {
                return None; // the lossless signature byte
            }
            let bits = le_u32(payload, 1)?;
            let width = (bits & 0x3fff) + 1;
            let height = ((bits >> 14) & 0x3fff) + 1;
            Some((width, height, false))
        }
        b"VP8X" => {
            let flags = *payload.first()?;
            let width = le_u24(payload, 4)? + 1;
            let height = le_u24(payload, 7)? + 1;
            Some((width, height, flags & 0x02 != 0))
        }
        _ => None,
    }
}

/// The width and height of a BMP's DIB header.
fn bmp_size(bytes: &[u8]) -> Option<(u32, u32)> {
    let header_size = le_u32(bytes, 14)?;
    match header_size {
        // BITMAPCOREHEADER: unsigned 16-bit dimensions.
        12 => Some((u32::from(le_u16(bytes, 18)?), u32::from(le_u16(bytes, 20)?))),
        // BITMAPINFOHEADER and its longer variants: signed 32-bit dimensions, a
        // negative height meaning a top-down bitmap.
        size if size >= 40 => {
            let width = i32::from_le_bytes(bytes.get(18..22)?.try_into().ok()?);
            let height = i32::from_le_bytes(bytes.get(22..26)?.try_into().ok()?);
            if width <= 0 {
                return None;
            }
            Some((width.unsigned_abs(), height.unsigned_abs()))
        }
        _ => None,
    }
}

/// Hand-built header fixtures: no binary assets and no encoded payloads.
///
/// Each fixture is the smallest byte string whose header states a size, so the
/// parsers are exercised without an image decoder on either side.
#[cfg(test)]
pub(crate) mod fixtures {
    /// A PNG carrying only its signature and `IHDR` chunk.
    pub(crate) fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::from(super::PNG_SIGNATURE);
        bytes.extend_from_slice(&13_u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 6, 0, 0, 0]); // bit depth, colour type, compression, filter, interlace
        bytes.extend_from_slice(&[0, 0, 0, 0]); // the CRC is not read
        bytes
    }

    /// A JPEG made of `SOI`, an `APP0` segment, one `SOF0` frame header and `EOI`.
    pub(crate) fn jpeg(width: u16, height: u16) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xd8];
        bytes.extend_from_slice(&[0xff, 0xe0, 0x00, 0x10]);
        bytes.extend_from_slice(b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00");
        bytes.extend_from_slice(&[0xff, 0xc0, 0x00, 0x11, 0x08]);
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&[0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        bytes.extend_from_slice(&[0xff, 0xd9]);
        bytes
    }

    /// A GIF with `frames` image descriptors and no colour tables.
    pub(crate) fn gif(width: u16, height: u16, frames: usize) -> Vec<u8> {
        let mut bytes = Vec::from(b"GIF89a".as_slice());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&[0x00, 0x00, 0x00]);
        for _ in 0..frames {
            bytes.push(0x2c); // image descriptor
            bytes.extend_from_slice(&[0, 0, 0, 0]); // left, top
            bytes.extend_from_slice(&width.to_le_bytes());
            bytes.extend_from_slice(&height.to_le_bytes());
            bytes.push(0x00); // no local colour table
            bytes.extend_from_slice(&[0x02, 0x02, 0x44, 0x01, 0x00]); // LZW size, one sub-block, terminator
        }
        bytes.push(0x3b);
        bytes
    }

    /// Wraps `payload` in a `RIFF`/`WEBP` container as `fourcc`.
    fn riff(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut body = Vec::from(b"WEBP".as_slice());
        body.extend_from_slice(fourcc);
        body.extend_from_slice(
            &u32::try_from(payload.len())
                .expect("a fixture payload fits in u32")
                .to_le_bytes(),
        );
        body.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            body.push(0); // RIFF chunk padding
        }
        let mut bytes = Vec::from(b"RIFF".as_slice());
        bytes.extend_from_slice(
            &u32::try_from(body.len())
                .expect("a fixture body fits in u32")
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&body);
        bytes
    }

    /// A lossy (`VP8 `) WebP header.
    pub(crate) fn webp_lossy(width: u16, height: u16) -> Vec<u8> {
        let mut payload = vec![0x10, 0x00, 0x00, 0x9d, 0x01, 0x2a];
        payload.extend_from_slice(&width.to_le_bytes());
        payload.extend_from_slice(&height.to_le_bytes());
        payload.extend_from_slice(&[0, 0, 0, 0]);
        riff(b"VP8 ", &payload)
    }

    /// A lossless (`VP8L`) WebP header.
    pub(crate) fn webp_lossless(width: u32, height: u32) -> Vec<u8> {
        let bits = ((height - 1) << 14) | (width - 1);
        let mut payload = vec![0x2f];
        payload.extend_from_slice(&bits.to_le_bytes());
        riff(b"VP8L", &payload)
    }

    /// An extended (`VP8X`) WebP header, with the animation flag when `animated`.
    pub(crate) fn webp_extended(width: u32, height: u32, animated: bool) -> Vec<u8> {
        let mut payload = vec![if animated { 0x02 } else { 0x00 }, 0, 0, 0];
        let canvas_width = (width - 1).to_le_bytes();
        let canvas_height = (height - 1).to_le_bytes();
        payload.extend_from_slice(&canvas_width[..3]);
        payload.extend_from_slice(&canvas_height[..3]);
        riff(b"VP8X", &payload)
    }

    /// A BMP with a 40-byte `BITMAPINFOHEADER`; a negative height is top-down.
    pub(crate) fn bmp(width: i32, height: i32) -> Vec<u8> {
        let mut bytes = Vec::from(b"BM".as_slice());
        bytes.extend_from_slice(&54_u32.to_le_bytes()); // file size
        bytes.extend_from_slice(&[0, 0, 0, 0]); // reserved
        bytes.extend_from_slice(&54_u32.to_le_bytes()); // pixel data offset
        bytes.extend_from_slice(&40_u32.to_le_bytes()); // BITMAPINFOHEADER
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes()); // planes
        bytes.extend_from_slice(&24_u16.to_le_bytes()); // bits per pixel
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // compression
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // image size
        bytes.extend_from_slice(&[0; 16]); // resolution and palette counts
        bytes
    }

    /// A BMP with the 12-byte `BITMAPCOREHEADER`.
    pub(crate) fn bmp_core(width: u16, height: u16) -> Vec<u8> {
        let mut bytes = Vec::from(b"BM".as_slice());
        bytes.extend_from_slice(&26_u32.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(&26_u32.to_le_bytes());
        bytes.extend_from_slice(&12_u32.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes()); // planes
        bytes.extend_from_slice(&24_u16.to_le_bytes()); // bits per pixel
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    fn parse(bytes: &[u8]) -> PictureInfo {
        PictureInfo::parse(bytes).expect("a readable picture header")
    }

    #[test]
    fn png_ihdr_gives_the_size_and_byte_count() {
        let info = parse(&png(800, 600));
        assert_eq!(info.format, PictureFormat::Png);
        assert_eq!((info.width, info.height), (800, 600));
        assert_eq!(info.bytes, 33);
        assert!(!info.animated);
        assert_eq!(info.format.as_str(), "PNG");
    }

    #[test]
    fn jpeg_frame_header_is_reached_through_the_app_segment() {
        let info = parse(&jpeg(1920, 1080));
        assert_eq!(info.format, PictureFormat::Jpeg);
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.format.as_str(), "JPEG");
        // A baseline frame header is preceded here by APP0, so the segment walk ran.
        assert!(info.bytes > 2 + 2 + 2);
    }

    #[test]
    fn gif_logical_screen_and_frame_stream_are_read() {
        let still = parse(&gif(64, 32, 1));
        assert_eq!(still.format, PictureFormat::Gif);
        assert_eq!((still.width, still.height), (64, 32));
        assert!(!still.animated);

        let animated = parse(&gif(64, 32, 3));
        assert!(animated.animated);
        assert_eq!(animated.render_format(), "animated GIF");
    }

    #[test]
    fn webp_reads_all_three_chunk_kinds() {
        let lossy = parse(&webp_lossy(320, 240));
        assert_eq!(lossy.format, PictureFormat::Webp);
        assert_eq!((lossy.width, lossy.height), (320, 240));

        let lossless = parse(&webp_lossless(1024, 768));
        assert_eq!((lossless.width, lossless.height), (1024, 768));

        let extended = parse(&webp_extended(5000, 4000, false));
        assert_eq!((extended.width, extended.height), (5000, 4000));
        assert!(!extended.animated);

        let animated = parse(&webp_extended(20, 10, true));
        assert!(animated.animated);
        assert_eq!((animated.width, animated.height), (20, 10));
    }

    #[test]
    fn bmp_reads_both_dib_headers_and_a_top_down_height() {
        let info = parse(&bmp(640, 480));
        assert_eq!(info.format, PictureFormat::Bmp);
        assert_eq!((info.width, info.height), (640, 480));

        let top_down = parse(&bmp(640, -480));
        assert_eq!((top_down.width, top_down.height), (640, 480));

        let core = parse(&bmp_core(16, 8));
        assert_eq!((core.width, core.height), (16, 8));
        assert_eq!(core.format.as_str(), "BMP");
    }

    #[test]
    fn sniff_names_every_supported_format() {
        assert_eq!(PictureFormat::sniff(&png(1, 1)), PictureFormat::Png);
        assert_eq!(PictureFormat::sniff(&jpeg(1, 1)), PictureFormat::Jpeg);
        assert_eq!(PictureFormat::sniff(&gif(1, 1, 1)), PictureFormat::Gif);
        assert_eq!(PictureFormat::sniff(&webp_lossy(1, 1)), PictureFormat::Webp);
        assert_eq!(PictureFormat::sniff(&bmp(1, 1)), PictureFormat::Bmp);
        assert_eq!(
            PictureFormat::sniff(b"not a picture"),
            PictureFormat::Unknown
        );
        assert_eq!(PictureFormat::sniff(b""), PictureFormat::Unknown);
        assert_eq!(PictureFormat::Unknown.as_str(), "unknown");
    }

    #[test]
    fn truncated_and_corrupt_headers_are_not_a_picture() {
        // An empty file and a signature with nothing behind it.
        assert!(PictureInfo::parse(b"").is_none());
        assert!(PictureInfo::parse(&PNG_SIGNATURE).is_none());
        // A PNG whose first chunk is not a 13-byte IHDR.
        let mut mislabelled = png(10, 10);
        mislabelled[12..16].copy_from_slice(b"IDAT");
        assert!(PictureInfo::parse(&mislabelled).is_none());
        // A PNG cut in the middle of the height field.
        assert!(PictureInfo::parse(&png(10, 10)[..18]).is_none());
        // A JPEG with no frame header at all.
        assert!(PictureInfo::parse(&[0xff, 0xd8, 0xff, 0xd9]).is_none());
        assert!(PictureInfo::parse(&[0xff, 0xd8, 0xff, 0xc0, 0x00]).is_none());
        // A GIF cut inside the logical screen descriptor, and one with no trailer.
        assert!(PictureInfo::parse(&gif(4, 4, 1)[..10]).is_none());
        assert!(PictureInfo::parse(b"GIF89a").is_none());
        // A WebP whose first chunk is not a picture chunk.
        let mut webp = webp_lossy(4, 4);
        webp[12..16].copy_from_slice(b"ICCP");
        assert!(PictureInfo::parse(&webp).is_none());
        // A WebP cut inside the VP8 start code.
        assert!(PictureInfo::parse(&webp_lossy(4, 4)[..22]).is_none());
        // A lossy WebP whose first byte says "not a key frame".
        let mut interframe = webp_lossy(4, 4);
        interframe[20] = 0x01;
        assert!(PictureInfo::parse(&interframe).is_none());
        // A BMP with an unknown header size, a zero width, and a cut header.
        let mut unknown_header = bmp(4, 4);
        unknown_header[14..18].copy_from_slice(&7_u32.to_le_bytes());
        assert!(PictureInfo::parse(&unknown_header).is_none());
        assert!(PictureInfo::parse(&bmp(0, 4)).is_none());
        assert!(PictureInfo::parse(&bmp(4, 0)).is_none());
        assert!(PictureInfo::parse(&bmp(4, 4)[..16]).is_none());
        // Bytes that name no format.
        assert!(PictureInfo::parse(b"not a picture").is_none());
    }

    #[test]
    fn evidence_states_that_only_metadata_reaches_the_judge() {
        let info = parse(&png(800, 600));
        assert_eq!(
            info.evidence("poster.png"),
            "path: poster.png\npicture file (800x600 PNG, 33 bytes)\n\
             the judge sees the picture's metadata (format and size), not its pixels"
        );
        // A JPEG of a normal size needs no note: Python would send it as it is.
        assert!(info.rewrite_reasons().is_empty());
        assert_eq!(info.longest_side(), 800);
    }

    #[test]
    fn oversized_and_unshippable_formats_are_reported_as_python_rewrites() {
        let side = parse(&png(2000, 100));
        assert_eq!(side.rewrite_reasons().len(), 1);
        let text = side.evidence("wide.png");
        assert!(
            text.contains("because its longest side (2000 px) is longer than 1600 px"),
            "{text}"
        );
        assert!(text.contains("this port does not re-encode"), "{text}");

        let mut heavy = png(10, 10);
        heavy.resize(PICTURE_BYTES + 1, 0);
        let heavy = parse(&heavy);
        assert_eq!(heavy.bytes, PICTURE_BYTES + 1);
        let text = heavy.evidence("heavy.png");
        assert!(
            text.contains("the file is larger than 3000000 bytes"),
            "{text}"
        );

        // BMP is not a format a vision API takes as it is, whatever its size.
        let bitmap = parse(&bmp(10, 10));
        assert_eq!(bitmap.rewrite_reasons().len(), 1);
        assert!(
            bitmap
                .evidence("small.bmp")
                .contains("BMP is not a format a model takes as it is"),
            "{}",
            bitmap.evidence("small.bmp")
        );

        // Every reason at once.
        let mut huge = bmp(4000, 3000);
        huge.resize(PICTURE_BYTES + 1, 0);
        let huge = parse(&huge);
        assert_eq!(huge.rewrite_reasons().len(), 3);
        assert!(huge.evidence("huge.bmp").contains(" and "));
    }

    #[test]
    fn picture_info_round_trips_through_json_and_tolerates_unknown_fields() {
        let info = parse(&gif(8, 6, 2));
        let value = serde_json::to_value(&info).unwrap();
        assert_eq!(value["format"], "gif");
        assert_eq!(value["width"], 8);
        assert_eq!(value["animated"], serde_json::Value::Bool(true));
        let restored: PictureInfo = serde_json::from_value(value).unwrap();
        assert_eq!(restored, info);

        let defaulted: PictureInfo =
            serde_json::from_str(r#"{"format": "png", "unknown_field": 1}"#).unwrap();
        assert_eq!(defaulted.format, PictureFormat::Png);
        assert_eq!(defaulted.width, 0);
        assert_eq!(PictureInfo::default().format, PictureFormat::Unknown);
    }
}
