//! BMP header parsing: the file header plus a `BITMAPINFOHEADER` (or the older
//! `BITMAPCOREHEADER`) giving width, height, bit depth and compression.
//!
//! The pixel array is never touched: everything reported comes from the two leading headers.

use serde::Serialize;

use super::super::parse::{fourcc, le_i32_at, le_u16_at, le_u32_at, Cursor, Failure, ParseResult};

/// Size of the OS/2 1.x `BITMAPCOREHEADER`, in bytes.
const CORE_HEADER_SIZE: u32 = 12;

/// Smallest `BITMAPINFOHEADER` variant that carries a compression field.
const INFO_HEADER_SIZE: u32 = 40;

/// Fields declared by a BMP header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BmpInfo {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels (always positive; see [`Self::top_down`]).
    pub height: u32,
    /// Whether rows are stored bottom-up (the header height was negative).
    pub top_down: bool,
    /// Bits per pixel: `1`, `4`, `8`, `16`, `24` or `32`.
    pub bit_count: u16,
    /// Compression code; `0` is uncompressed.
    pub compression: u32,
    /// Name of the compression code (`none`, `rle8`, `bitfields`, …).
    pub compression_name: &'static str,
    /// Size of the DIB header that followed the file header.
    pub dib_header_size: u32,
    /// Offset of the pixel array, in bytes from the start of the file.
    pub data_offset: u32,
}

/// Parses a BMP header.
///
/// # Errors
///
/// Returns a [`Failure`] when the signature, file header or DIB header is truncated, when the
/// signature is not `BM`, when the DIB header is neither `BITMAPCOREHEADER` nor a
/// `BITMAPINFOHEADER` variant, or when the declared width is not positive or a dimension is zero.
pub fn parse(bytes: &[u8]) -> ParseResult<BmpInfo> {
    fn header_incomplete() -> Failure {
        Failure::new("truncated BMP header: file header is incomplete")
    }

    let mut cursor = Cursor::new(bytes);
    let Some(signature) = cursor.take(2) else {
        return Err(Failure::new(
            "truncated BMP header: signature is incomplete",
        ));
    };
    if signature != b"BM".as_slice() {
        return Err(Failure::new(format!(
            "unsupported image format: {}",
            fourcc(signature)
        )));
    }
    // The file size is not needed: the byte size comes from the file itself.
    let Some(_file_size) = cursor.le_u32() else {
        return Err(header_incomplete());
    };
    if !cursor.skip(4) {
        return Err(header_incomplete());
    }
    let Some(data_offset) = cursor.le_u32() else {
        return Err(header_incomplete());
    };
    let Some(dib_header_size) = cursor.le_u32() else {
        return Err(header_incomplete());
    };
    let (width, height, top_down, bit_count, compression) = if dib_header_size == CORE_HEADER_SIZE {
        read_core_header(&mut cursor)?
    } else if dib_header_size >= INFO_HEADER_SIZE {
        read_info_header(&mut cursor)?
    } else {
        return Err(Failure::new(format!(
            "unsupported BMP header: DIB header size {dib_header_size}"
        )));
    };
    if width == 0 || height == 0 {
        return Err(Failure::new(format!(
            "invalid BMP header: zero dimension {width}x{height}"
        )));
    }
    Ok(BmpInfo {
        width,
        height,
        top_down,
        bit_count,
        compression,
        compression_name: compression_name(compression),
        dib_header_size,
        data_offset,
    })
}

/// Reads an OS/2 1.x `BITMAPCOREHEADER`, which has no compression field.
fn read_core_header(cursor: &mut Cursor<'_>) -> ParseResult<(u32, u32, bool, u16, u32)> {
    fn incomplete() -> Failure {
        Failure::new("truncated BMP header: BITMAPCOREHEADER is incomplete")
    }

    let Some(header) = cursor.take(8) else {
        return Err(incomplete());
    };
    let width = le_u16_at(header, 0).ok_or_else(incomplete)?;
    let height = le_u16_at(header, 2).ok_or_else(incomplete)?;
    let bit_count = le_u16_at(header, 6).ok_or_else(incomplete)?;
    Ok((u32::from(width), u32::from(height), false, bit_count, 0))
}

/// Reads the shared prefix of every `BITMAPINFOHEADER` variant.
///
/// Returns `(width, height, top_down, bit_count, compression)`.
fn read_info_header(cursor: &mut Cursor<'_>) -> ParseResult<(u32, u32, bool, u16, u32)> {
    fn incomplete() -> Failure {
        Failure::new("truncated BMP header: BITMAPINFOHEADER is incomplete")
    }

    let Some(header) = cursor.take(16) else {
        return Err(incomplete());
    };
    let raw_width = le_i32_at(header, 0).ok_or_else(incomplete)?;
    let raw_height = le_i32_at(header, 4).ok_or_else(incomplete)?;
    let bit_count = le_u16_at(header, 10).ok_or_else(incomplete)?;
    let compression = le_u32_at(header, 12).ok_or_else(incomplete)?;
    if raw_width <= 0 {
        return Err(Failure::new(format!(
            "invalid BMP header: width {raw_width}"
        )));
    }
    Ok((
        raw_width.unsigned_abs(),
        raw_height.unsigned_abs(),
        raw_height < 0,
        bit_count,
        compression,
    ))
}

/// The name of a BMP compression code.
fn compression_name(compression: u32) -> &'static str {
    match compression {
        0 => "none",
        1 => "rle8",
        2 => "rle4",
        3 => "bitfields",
        4 => "jpeg",
        5 => "png",
        6 => "alphabitfields",
        11 => "cmyk",
        12 => "cmykrle8",
        13 => "cmykrle4",
        _ => "unknown",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds a `BITMAPINFOHEADER` BMP of `width`x`height` with `bit_count` and `compression`.
    fn info_bmp(width: i32, height: i32, bit_count: u16, compression: u32) -> Vec<u8> {
        let mut bytes = b"BM".to_vec();
        bytes.extend_from_slice(&54u32.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(&54u32.to_le_bytes());
        bytes.extend_from_slice(&INFO_HEADER_SIZE.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&bit_count.to_le_bytes());
        bytes.extend_from_slice(&compression.to_le_bytes());
        bytes.extend_from_slice(&[0; 20]);
        bytes
    }

    #[test]
    fn reads_info_header_fields() {
        let info = parse(&info_bmp(4, 3, 24, 0)).unwrap();
        assert_eq!(info.width, 4);
        assert_eq!(info.height, 3);
        assert!(!info.top_down);
        assert_eq!(info.bit_count, 24);
        assert_eq!(info.compression, 0);
        assert_eq!(info.compression_name, "none");
        assert_eq!(info.dib_header_size, INFO_HEADER_SIZE);
        assert_eq!(info.data_offset, 54);
    }

    #[test]
    fn negative_height_means_top_down() {
        let info = parse(&info_bmp(10, -20, 32, 3)).unwrap();
        assert_eq!(info.height, 20);
        assert!(info.top_down);
        assert_eq!(info.bit_count, 32);
        assert_eq!(info.compression_name, "bitfields");
    }

    #[test]
    fn reads_core_header() {
        let mut bytes = b"BM".to_vec();
        bytes.extend_from_slice(&26u32.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(&26u32.to_le_bytes());
        bytes.extend_from_slice(&CORE_HEADER_SIZE.to_le_bytes());
        bytes.extend_from_slice(&7u16.to_le_bytes());
        bytes.extend_from_slice(&9u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&8u16.to_le_bytes());
        let info = parse(&bytes).unwrap();
        assert_eq!(info.width, 7);
        assert_eq!(info.height, 9);
        assert_eq!(info.bit_count, 8);
        assert_eq!(info.compression_name, "none");
        assert!(!info.top_down);
    }

    #[test]
    fn truncated_headers_are_reported() {
        assert_eq!(
            parse(b"B").unwrap_err().message(),
            "truncated BMP header: signature is incomplete"
        );
        assert_eq!(
            parse(b"BM\0\0").unwrap_err().message(),
            "truncated BMP header: file header is incomplete"
        );
        let mut bytes = b"BM".to_vec();
        bytes.extend_from_slice(&[0; 12]);
        bytes.extend_from_slice(&INFO_HEADER_SIZE.to_le_bytes());
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "truncated BMP header: BITMAPINFOHEADER is incomplete"
        );
    }

    #[test]
    fn unsupported_magic_and_dib_header_are_reported() {
        assert_eq!(
            parse(b"XX\0\0\0\0").unwrap_err().message(),
            "unsupported image format: XX"
        );
        let mut bytes = info_bmp(1, 1, 24, 0);
        bytes[14..18].copy_from_slice(&16u32.to_le_bytes());
        assert_eq!(
            parse(&bytes).unwrap_err().message(),
            "unsupported BMP header: DIB header size 16"
        );
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        assert_eq!(
            parse(&info_bmp(0, 1, 24, 0)).unwrap_err().message(),
            "invalid BMP header: width 0"
        );
        assert_eq!(
            parse(&info_bmp(-4, 1, 24, 0)).unwrap_err().message(),
            "invalid BMP header: width -4"
        );
        assert_eq!(
            parse(&info_bmp(1, 0, 24, 0)).unwrap_err().message(),
            "invalid BMP header: zero dimension 1x0"
        );
    }
}
