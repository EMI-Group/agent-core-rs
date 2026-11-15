//! Image dimensions read from header bytes, without decoding any pixels.
//!
//! Python's `skill_omni_creation` pipeline measures images with Pillow
//! (`Image.open(...).size`) to apply its 80 px decorative-image gate. Pillow is
//! a decoding library; this port replaces it with [`image_dimensions`], a pure
//! scanner over the documented *header* fields of the five container formats the
//! pipeline keeps, so the gate still sees real dimensions while the crate gains
//! no image-decoding dependency.
//!
//! Supported containers, recognized by magic bytes:
//!
//! - PNG — `IHDR` width/height as big-endian `u32`s;
//! - JPEG — the first `SOFn` frame header, after skipping `APPn`/`DQT`/`DHT`/…
//!   segments by their declared lengths;
//! - GIF — the logical screen descriptor, little-endian `u16`s;
//! - WebP — the first chunk, whether lossy (`VP8 `), lossless (`VP8L`) or
//!   extended (`VP8X`);
//! - PNM (`P1`…`P6`) — the decimal ASCII width and height of the plain/raw
//!   bitmap family.
//!
//! Every parser is total: a truncated, mislabelled or otherwise malformed
//! header yields `None` and never panics, reads out of bounds, or loops forever.
//! A container this module does not understand (`BMP`, `TIFF`, `AVIF`, `SVG`,
//! an archive, plain text) also yields `None`.
/// The eight-byte PNG signature (`\x89PNG\r\n\x1a\n`).
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
/// The `IHDR` chunk type, the mandatory first PNG chunk.
const PNG_HEADER_CHUNK: [u8; 4] = *b"IHDR";
/// Minimum PNG length: the signature plus the `IHDR` length/type/width/height.
const PNG_HEADER_BYTES: usize = 24;
/// The `JPEG` start-of-image marker (`FF D8`).
const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];
/// The shortest `SOFn` segment: length, precision, height (2) and width (2).
const JPEG_FRAME_HEADER_BYTES: u16 = 7;
/// The three-byte `VP8` keyframe start code.
const WEBP_LOSSY_SYNC: [u8; 3] = [0x9D, 0x01, 0x2A];
/// The `VP8L` lossless signature byte.
const WEBP_LOSSLESS_SIGNATURE: u8 = 0x2F;
/// Mask for the 14-bit wide WebP dimension fields.
const WEBP_BITSTREAM_MASK: u32 = 0x3FFF;
/// Mask for the 24-bit wide `VP8X` canvas dimension fields.
const WEBP_CANVAS_MASK: u32 = 0x00FF_FFFF;

/// Reads an image's `(width, height)` in pixels from its header bytes.
///
/// Recognized payloads, in match order: PNG, JPEG, GIF, WebP and the PNM family
/// (`P1`…`P6`). The values come from the container header only — no pixel data
/// is decoded, no checksum is verified, and no allocation is made — so a caller
/// with a downloaded payload can apply a dimension gate cheaply.
///
/// Returns `None` when the magic bytes are unknown or when the header of a
/// recognized container is truncated or inconsistent (a missing `IHDR` chunk, a
/// JPEG without a frame header, a WebP whose first chunk is not `VP8 `/`VP8L`/
/// `VP8X`, a PNM header without two decimal numbers). The function is total, so
/// it never panics whatever slice it is handed.
///
/// # Examples
///
/// ```
/// use openjiuwen_dev_tools::skill_creator::omni::filter::image_dimensions;
///
/// let mut gif = b"GIF89a".to_vec();
/// gif.extend_from_slice(&640u16.to_le_bytes());
/// gif.extend_from_slice(&480u16.to_le_bytes());
/// assert_eq!(image_dimensions(&gif), Some((640, 480)));
/// assert_eq!(image_dimensions(b"not an image"), None);
/// ```
#[must_use]
pub fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.starts_with(&PNG_SIGNATURE) {
        return png_dimensions(bytes);
    }
    if bytes.starts_with(&JPEG_SOI) {
        return jpeg_dimensions(bytes);
    }
    if bytes.starts_with(b"GIF") {
        return gif_dimensions(bytes);
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12).is_some_and(|tag| tag == b"WEBP") {
        return webp_dimensions(bytes);
    }
    if bytes.starts_with(b"P") {
        return pnm_dimensions(bytes);
    }
    None
}

/// PNG: the big-endian `u32` pair of the mandatory first `IHDR` chunk.
///
/// Layout: signature (8), chunk length (4), `IHDR` (4), width (4), height (4) —
/// 24 bytes, with the width at offset 16 and the height at offset 20.
fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < PNG_HEADER_BYTES || !bytes.starts_with(&PNG_SIGNATURE) {
        return None;
    }
    if bytes.get(12..16)? != PNG_HEADER_CHUNK.as_slice() {
        return None;
    }
    Some((read_u32_be(bytes, 16)?, read_u32_be(bytes, 20)?))
}

/// JPEG: the `(width, height)` of the first start-of-frame segment.
///
/// The walk starts after `FF D8`, skips `FF` fill bytes before every marker, and
/// leaves every counted segment by its declared 16-bit length (which excludes
/// the two length bytes). `TEM` (`0x01`) and `RSTn` (`0xD0`…`0xD7`) carry no
/// length and no payload, so they only advance one byte. `SOF0`…`SOF15` ends the
/// walk, except `DHT` (`0xC4`), `JPG` (`0xC8`) and `DAC` (`0xCC`), which reuse
/// that marker range without being frame headers. `SOI`/`EOI`/`0x00` before a
/// frame header, a segment whose length is below its own two bytes, and a
/// truncated buffer all yield `None`.
fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.get(0..2)? != JPEG_SOI.as_slice() {
        return None;
    }
    let mut offset = 2usize;
    loop {
        if *bytes.get(offset)? != 0xFF {
            return None;
        }
        let mut marker = offset + 1;
        while *bytes.get(marker)? == 0xFF {
            marker += 1;
        }
        // The `0xFF` directly before the marker byte: SOF payload fields are
        // addressed relative to it.
        let marker_start = marker.checked_sub(1)?;
        match *bytes.get(marker)? {
            // Stuffed `0x00` and `SOI`/`EOI` only appear outside a header walk.
            0x00 | 0xD8 | 0xD9 => return None,
            // `TEM` and the restart markers carry no length field.
            0x01 | 0xD0..=0xD7 => offset = marker.checked_add(1)?,
            _ => {
                let length = usize::from(read_u16_be(bytes, marker.checked_add(1)?)?);
                if length < 2 {
                    return None;
                }
                if is_jpeg_frame_marker(*bytes.get(marker)?) {
                    if length < usize::from(JPEG_FRAME_HEADER_BYTES) {
                        return None;
                    }
                    let height = read_u16_be(bytes, marker_start.checked_add(5)?)?;
                    let width = read_u16_be(bytes, marker_start.checked_add(7)?)?;
                    return Some((u32::from(width), u32::from(height)));
                }
                offset = marker.checked_add(1)?.checked_add(length)?;
            }
        }
    }
}

/// Whether a JPEG marker is a start-of-frame header (`SOF0`…`SOF15`).
///
/// `DHT` (`0xC4`), `JPG` (`0xC8`) and `DAC` (`0xCC`) share the range but carry
/// tables rather than frame geometry, so they are not frame headers.
fn is_jpeg_frame_marker(marker: u8) -> bool {
    matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC)
}

/// GIF: the little-endian `u16` pair of the logical screen descriptor.
///
/// Layout: `GIF87a`/`GIF89a` (6), width (2), height (2).
fn gif_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if !(bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return None;
    }
    Some((
        u32::from(read_u16_le(bytes, 6)?),
        u32::from(read_u16_le(bytes, 8)?),
    ))
}

/// WebP: the dimensions of the first chunk of the `RIFF`/`WEBP` container.
///
/// The chunk header starts at offset 12 (`VP8 `, `VP8L` or `VP8X` + its 32-bit
/// size), so the payload begins at offset 20:
///
/// - `VP8 ` (lossy) — the 3-byte frame start code `9D 01 2A` at payload offset
///   3, then the 14-bit width and height as little-endian `u16`s at payload
///   offsets 6 and 8;
/// - `VP8L` (lossless) — the `0x2F` signature, then the 14-bit `width - 1` and
///   `height - 1` packed little-endian into the next four bytes;
/// - `VP8X` (extended) — the 24-bit `width - 1` at payload offset 4 and the
///   24-bit `height - 1` at payload offset 7, both little-endian.
///
/// Any other first chunk (an unknown or an animated/animation-frames chunk), a
/// wrong signature, or a payload too short for its own header yields `None`.
fn webp_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let fourcc = bytes.get(12..16)?;
    let payload = 20usize;
    match fourcc {
        b"VP8 " => {
            if bytes.get(payload.checked_add(3)?..payload.checked_add(6)?)?
                != WEBP_LOSSY_SYNC.as_slice()
            {
                return None;
            }
            let width = read_u16_le(bytes, payload.checked_add(6)?)? & 0x3FFF;
            let height = read_u16_le(bytes, payload.checked_add(8)?)? & 0x3FFF;
            Some((u32::from(width), u32::from(height)))
        }
        b"VP8L" => {
            if *bytes.get(payload)? != WEBP_LOSSLESS_SIGNATURE {
                return None;
            }
            let packed = read_u32_le(bytes, payload.checked_add(1)?)?;
            Some((
                (packed & WEBP_BITSTREAM_MASK).checked_add(1)?,
                ((packed >> 14) & WEBP_BITSTREAM_MASK).checked_add(1)?,
            ))
        }
        b"VP8X" => Some((
            (read_u24_le(bytes, payload.checked_add(4)?)? & WEBP_CANVAS_MASK).checked_add(1)?,
            (read_u24_le(bytes, payload.checked_add(7)?)? & WEBP_CANVAS_MASK).checked_add(1)?,
        )),
        _ => None,
    }
}

/// PNM: the decimal ASCII width and height that follow the `P1`…`P6` magic.
///
/// The magic must be followed by whitespace; both numbers are decimal ASCII and
/// `#` starts a comment that runs to the end of the line, so comments and
/// whitespace may separate the magic, the width and the height. The remaining
/// header fields (`maxval`, and the pixel data) are not read. A magic outside
/// `P1`…`P6`, a missing number, a number that does not fit `u32`, or an
/// unterminated comment yields `None`.
fn pnm_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let magic = bytes.get(0..2)?;
    if magic.first() != Some(&b'P') || !(b'1'..=b'6').contains(magic.get(1)?) {
        return None;
    }
    if !bytes.get(2)?.is_ascii_whitespace() {
        return None;
    }
    let mut cursor = 2usize;
    let width = read_pnm_number(bytes, &mut cursor)?;
    let height = read_pnm_number(bytes, &mut cursor)?;
    Some((width, height))
}

/// Reads the next decimal number of a PNM header, skipping blanks and comments.
///
/// Advances `cursor` past the digits of the number it read and returns its
/// value; returns `None` at the end of the payload or when the digits do not fit
/// a `u32`. A `#` comment is skipped up to (and including) its line break, as
/// the PNM specification requires.
fn read_pnm_number(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    loop {
        let byte = *bytes.get(*cursor)?;
        if byte == b'#' {
            let rest = bytes.get(*cursor..)?;
            let end = rest
                .iter()
                .position(|value| *value == b'\n' || *value == b'\r')?;
            *cursor = cursor.checked_add(end)?.checked_add(1)?;
            continue;
        }
        if byte.is_ascii_whitespace() {
            *cursor = cursor.checked_add(1)?;
            continue;
        }
        break;
    }
    let start = *cursor;
    let mut value: u32 = 0;
    while let Some(byte) = bytes.get(*cursor) {
        if !byte.is_ascii_digit() {
            break;
        }
        let digit = u32::from(byte.checked_sub(b'0')?);
        value = value.checked_mul(10)?.checked_add(digit)?;
        *cursor = cursor.checked_add(1)?;
    }
    if *cursor == start {
        return None;
    }
    Some(value)
}

/// Reads a big-endian `u32` at `offset`, or `None` when it does not fit.
fn read_u32_be(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_be_bytes(read_array(bytes, offset)?))
}

/// Reads a little-endian `u32` at `offset`, or `None` when it does not fit.
fn read_u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(read_array(bytes, offset)?))
}

/// Reads a big-endian `u16` at `offset`, or `None` when it does not fit.
fn read_u16_be(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_be_bytes(read_array(bytes, offset)?))
}

/// Reads a little-endian `u16` at `offset`, or `None` when it does not fit.
fn read_u16_le(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(read_array(bytes, offset)?))
}

/// Reads a little-endian 24-bit integer at `offset`, or `None` when it does not
/// fit.
fn read_u24_le(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(3)?;
    let slice = bytes.get(offset..end)?;
    let first = u32::from(*slice.first()?);
    let second = u32::from(*slice.get(1)?);
    let third = u32::from(*slice.get(2)?);
    Some(first | (second << 8) | (third << 16))
}

/// Copies the fixed-size byte run at `offset`, or `None` when it does not fit.
fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    let end = offset.checked_add(N)?;
    bytes.get(offset..end)?.try_into().ok()
}

#[cfg(test)]
mod tests;
