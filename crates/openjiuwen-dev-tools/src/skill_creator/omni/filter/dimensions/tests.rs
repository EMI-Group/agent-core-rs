use super::*;

/// A `png` header: signature, `IHDR` chunk length/type, width, height, and the
/// five remaining `IHDR` fields.
fn png(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(&13u32.to_be_bytes());
    bytes.extend_from_slice(&PNG_HEADER_CHUNK);
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&[8, 2, 0, 0, 0]);
    bytes
}

/// A `gif` header with an explicit version tag and logical screen descriptor.
fn gif(version: &[u8], width: u16, height: u16) -> Vec<u8> {
    let mut bytes = version.to_vec();
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&[0x00, 0x00, 0x00]);
    bytes
}

/// One `jpeg` segment: `FF <marker>`, its 16-bit length, then `payload`.
fn jpeg_segment(marker: u8, payload: &[u8]) -> Vec<u8> {
    let length = u16::try_from(payload.len() + 2).unwrap();
    let mut bytes = vec![0xFF, marker];
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

/// A `SOFn` segment for `width`x`height`.
fn jpeg_frame(marker: u8, width: u16, height: u16) -> Vec<u8> {
    let mut payload = vec![8u8];
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.extend_from_slice(&[3, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1]);
    jpeg_segment(marker, &payload)
}

/// A `jpeg` stream: `SOI`, then `segments`, then `EOI`.
fn jpeg(segments: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = JPEG_SOI.to_vec();
    for segment in segments {
        bytes.extend_from_slice(segment);
    }
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    bytes
}

/// An `APP0`/`JFIF` segment, optionally repeated `times` times.
fn app0(times: usize) -> Vec<Vec<u8>> {
    let payload = [
        0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, b'J', b'F', b'I', b'F',
    ];
    (0..times).map(|_| jpeg_segment(0xE0, &payload)).collect()
}

/// A `webp` container holding exactly one chunk.
fn webp(fourcc: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&u32::try_from(12 + payload.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(b"WEBP");
    bytes.extend_from_slice(&fourcc);
    bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

/// A lossy `VP8 ` chunk payload: the frame tag, the start code, `(w, h)`.
fn vp8_payload(width: u16, height: u16) -> Vec<u8> {
    let mut payload = vec![0x10, 0x00, 0x00];
    payload.extend_from_slice(&WEBP_LOSSY_SYNC);
    payload.extend_from_slice(&width.to_le_bytes());
    payload.extend_from_slice(&height.to_le_bytes());
    payload.extend_from_slice(&[0, 0, 0, 0]);
    payload
}

/// A lossless `VP8L` chunk payload: the signature byte, then `(w - 1, h - 1)`.
fn vp8l_payload(width: u32, height: u32) -> Vec<u8> {
    let packed = ((width - 1) & WEBP_BITSTREAM_MASK) | (((height - 1) & WEBP_BITSTREAM_MASK) << 14);
    let mut payload = vec![WEBP_LOSSLESS_SIGNATURE];
    payload.extend_from_slice(&packed.to_le_bytes());
    payload.extend_from_slice(&[0, 0, 0, 0]);
    payload
}

/// An extended `VP8X` chunk payload: flag/reserved bytes, then `(w - 1, h - 1)`
/// as 24-bit little-endian values.
fn vp8x_payload(width: u32, height: u32) -> Vec<u8> {
    let mut payload = vec![0x10, 0x00, 0x00, 0x00];
    payload.extend_from_slice((width - 1).to_le_bytes().get(..3).unwrap());
    payload.extend_from_slice((height - 1).to_le_bytes().get(..3).unwrap());
    payload
}

#[test]
fn png_dimensions_read_the_ihdr_chunk() {
    assert_eq!(image_dimensions(&png(1, 1)), Some((1, 1)));
    assert_eq!(image_dimensions(&png(640, 480)), Some((640, 480)));
    assert_eq!(
        image_dimensions(&png(u32::MAX, 0x0102_0304)),
        Some((u32::MAX, 0x0102_0304))
    );
    // Trailing chunks and pixel data are irrelevant to the header scan.
    let mut payload = png(1200, 600);
    payload.extend_from_slice(&[0u8; 4096]);
    assert_eq!(image_dimensions(&payload), Some((1200, 600)));
}

#[test]
fn png_dimensions_require_the_header_chunk() {
    let mut wrong_chunk = png(10, 10);
    wrong_chunk[12..16].copy_from_slice(b"IDAT");
    assert_eq!(image_dimensions(&wrong_chunk), None);
    // One byte short of the 24-byte minimum.
    let truncated = png(10, 10).get(..23).unwrap().to_vec();
    assert_eq!(image_dimensions(&truncated), None);
}

#[test]
fn gif_dimensions_read_the_screen_descriptor() {
    assert_eq!(image_dimensions(&gif(b"GIF87a", 1, 1)), Some((1, 1)));
    assert_eq!(
        image_dimensions(&gif(b"GIF89a", 640, 480)),
        Some((640, 480))
    );
    assert_eq!(
        image_dimensions(&gif(b"GIF89a", u16::MAX, 0x0102)),
        Some((65_535, 0x0102))
    );
}

#[test]
fn gif_dimensions_reject_unknown_versions_and_short_headers() {
    assert_eq!(image_dimensions(b"GIF88a\x01\x00\x01\x00"), None);
    assert_eq!(image_dimensions(b"GIF89a\x01\x00\x01"), None);
    assert_eq!(image_dimensions(b"GIF89"), None);
}

#[test]
fn jpeg_dimensions_skip_app_segments() {
    let mut segments = app0(2);
    segments.push(jpeg_frame(0xC0, 640, 480));
    assert_eq!(image_dimensions(&jpeg(&segments)), Some((640, 480)));
}

#[test]
fn jpeg_dimensions_accept_every_frame_marker() {
    for marker in [
        0xC0u8, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    ] {
        assert_eq!(
            image_dimensions(&jpeg(&[jpeg_frame(marker, 320, 200)])),
            Some((320, 200)),
            "{marker:#04X}"
        );
    }
}

#[test]
fn jpeg_dimensions_skip_table_markers_that_reuse_the_frame_range() {
    for marker in [0xC4u8, 0xC8, 0xCC] {
        let table = jpeg_segment(marker, &[0u8; 32]);
        let segments = [table, jpeg_frame(0xC0, 100, 50)];
        assert_eq!(
            image_dimensions(&jpeg(&segments)),
            Some((100, 50)),
            "{marker:#04X}"
        );
    }
}

#[test]
fn jpeg_dimensions_skip_fill_bytes() {
    let mut bytes = JPEG_SOI.to_vec();
    bytes.extend_from_slice(&[0xFF, 0xFF]);
    bytes.extend_from_slice(&jpeg_frame(0xC0, 32, 24));
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    assert_eq!(image_dimensions(&bytes), Some((32, 24)));
}

#[test]
fn jpeg_dimensions_skip_lengthless_markers() {
    let mut bytes = JPEG_SOI.to_vec();
    // `TEM` (`FF 01`) and every restart marker (`FF D0`..`FF D7`).
    bytes.extend_from_slice(&[0xFF, 0x01]);
    for marker in 0xD0u8..=0xD7 {
        bytes.extend_from_slice(&[0xFF, marker]);
    }
    bytes.extend_from_slice(&jpeg_frame(0xC2, 16, 9));
    assert_eq!(image_dimensions(&bytes), Some((16, 9)));
}

#[test]
fn jpeg_dimensions_reject_streams_without_a_frame_header() {
    assert_eq!(image_dimensions(&jpeg(&app0(1))), None);
    assert_eq!(image_dimensions(&jpeg(&[])), None);
    // A frame header whose own length field does not cover height and width.
    assert_eq!(
        image_dimensions(&jpeg(&[jpeg_segment(0xC0, &[8, 0, 0])])),
        None
    );
}

#[test]
fn jpeg_dimensions_reject_malformed_segments() {
    // Declared length below its own two bytes.
    assert_eq!(
        image_dimensions(&[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x01]),
        None
    );
    // A declared length that runs past the payload.
    assert_eq!(
        image_dimensions(&[0xFF, 0xD8, 0xFF, 0xE0, 0xFF, 0xF0, 0x00]),
        None
    );
    // A marker byte that is not preceded by `FF`.
    assert_eq!(image_dimensions(&[0xFF, 0xD8, 0x00, 0xE0]), None);
    // A stuffed zero byte outside entropy-coded data.
    assert_eq!(
        image_dimensions(&[0xFF, 0xD8, 0xFF, 0x00, 0xFF, 0xD9]),
        None
    );
    // `EOI` before any frame header.
    assert_eq!(image_dimensions(&[0xFF, 0xD8, 0xFF, 0xD9]), None);
    // A second `SOI` before any frame header.
    assert_eq!(image_dimensions(&[0xFF, 0xD8, 0xFF, 0xD8]), None);
}

#[test]
fn webp_dimensions_read_the_lossy_chunk() {
    assert_eq!(
        image_dimensions(&webp(*b"VP8 ", &vp8_payload(640, 480))),
        Some((640, 480))
    );
    assert_eq!(
        image_dimensions(&webp(*b"VP8 ", &vp8_payload(1, 1))),
        Some((1, 1))
    );
    // Only the low 14 bits of each dimension field are significant.
    assert_eq!(
        image_dimensions(&webp(*b"VP8 ", &vp8_payload(0x3FFF, 0x4001))),
        Some((16_383, 1))
    );
}

#[test]
fn webp_dimensions_read_the_lossless_chunk() {
    assert_eq!(
        image_dimensions(&webp(*b"VP8L", &vp8l_payload(1, 1))),
        Some((1, 1))
    );
    assert_eq!(
        image_dimensions(&webp(*b"VP8L", &vp8l_payload(1024, 768))),
        Some((1024, 768))
    );
    assert_eq!(
        image_dimensions(&webp(*b"VP8L", &vp8l_payload(16_384, 16_384))),
        Some((16_384, 16_384))
    );
}

#[test]
fn webp_dimensions_read_the_extended_chunk() {
    assert_eq!(
        image_dimensions(&webp(*b"VP8X", &vp8x_payload(1, 1))),
        Some((1, 1))
    );
    assert_eq!(
        image_dimensions(&webp(*b"VP8X", &vp8x_payload(1920, 1080))),
        Some((1920, 1080))
    );
    assert_eq!(
        image_dimensions(&webp(*b"VP8X", &vp8x_payload(16_777_216, 2))),
        Some((16_777_216, 2))
    );
}

#[test]
fn webp_dimensions_reject_unknown_or_broken_chunks() {
    assert_eq!(
        image_dimensions(&webp(*b"VP8N", &vp8_payload(10, 10))),
        None
    );
    assert_eq!(
        image_dimensions(&webp(*b"ANMF", &vp8x_payload(10, 10))),
        None
    );
    // A lossy chunk without the start code.
    let mut payload = vp8_payload(10, 10);
    payload[3] = 0x00;
    assert_eq!(image_dimensions(&webp(*b"VP8 ", &payload)), None);
    // A lossless chunk without the signature byte.
    let mut payload = vp8l_payload(10, 10);
    payload[0] = 0x00;
    assert_eq!(image_dimensions(&webp(*b"VP8L", &payload)), None);
    // Truncated payloads for all three layouts.
    assert_eq!(
        image_dimensions(&webp(*b"VP8 ", &vp8_payload(10, 10))[..28]),
        None
    );
    assert_eq!(
        image_dimensions(&webp(*b"VP8L", &vp8l_payload(10, 10))[..24]),
        None
    );
    assert_eq!(
        image_dimensions(&webp(*b"VP8X", &vp8x_payload(10, 10))[..28]),
        None
    );
}

#[test]
fn pnm_dimensions_read_every_magic() {
    for magic in ["P1", "P2", "P3", "P4", "P5", "P6"] {
        let header = format!("{magic}\n640 480\n255\n");
        assert_eq!(
            image_dimensions(header.as_bytes()),
            Some((640, 480)),
            "{magic}"
        );
    }
    assert_eq!(image_dimensions(b"P5 1 1\n255\n"), Some((1, 1)));
    assert_eq!(image_dimensions(b"P4\t16\r\n9\r\n"), Some((16, 9)));
}

#[test]
fn pnm_dimensions_skip_comments_and_whitespace() {
    assert_eq!(
        image_dimensions(b"P5\n# written by hand\n640 # width\n\t480\n255\n"),
        Some((640, 480))
    );
    assert_eq!(
        image_dimensions(b"P6\r\n#one\r#two\r1920 1080\r\n255\r\n"),
        Some((1920, 1080))
    );
}

#[test]
fn pnm_dimensions_reject_broken_headers() {
    // No whitespace after the magic.
    assert_eq!(image_dimensions(b"P512 12"), None);
    // A comment and no numbers.
    assert_eq!(image_dimensions(b"P5\n# only a comment\n"), None);
    // Only one number.
    assert_eq!(image_dimensions(b"P5\n640\n"), None);
    // An unterminated comment.
    assert_eq!(image_dimensions(b"P5\n640 # trailing"), None);
    // A number that does not fit a `u32`.
    assert_eq!(image_dimensions(b"P5\n99999999999 1\n"), None);
    // Magic outside `P1`..`P6`.
    assert_eq!(image_dimensions(b"P7\n640 480\n"), None);
    assert_eq!(image_dimensions(b"P0\n640 480\n"), None);
}

#[test]
fn unknown_and_short_payloads_yield_none() {
    for payload in [
        b"".as_slice(),
        b"\x89",
        b"\xFF",
        b"GI",
        b"\x89PN",
        b"RIFF",
        b"RIFF\x00\x00\x00\x00WEB",
        b"BM\x00\x00\x00\x00",
        b"hello world",
        b"<svg xmlns=\"http://example.com\"/>",
        b"<?xml version=\"1.0\"?><svg/>",
        b"II*\x00",
        b"avif",
    ] {
        assert_eq!(image_dimensions(payload), None, "{payload:?}");
    }
}

#[test]
fn truncated_headers_are_safe_and_consistent() {
    let mut jpeg_segments = app0(1);
    jpeg_segments.push(jpeg_frame(0xC0, 640, 480));
    let payloads = [
        png(640, 480),
        gif(b"GIF89a", 640, 480),
        jpeg(&jpeg_segments),
        webp(*b"VP8 ", &vp8_payload(640, 480)),
        webp(*b"VP8L", &vp8l_payload(640, 480)),
        webp(*b"VP8X", &vp8x_payload(640, 480)),
        b"P5\n640 480\n255\n".to_vec(),
    ];
    for payload in &payloads {
        let minimum = (0..=payload.len())
            .find(|length| image_dimensions(payload.get(..*length).unwrap()) == Some((640, 480)))
            .unwrap();
        for length in 0..=payload.len() {
            let dims = image_dimensions(payload.get(..length).unwrap());
            if length >= minimum {
                assert_eq!(dims, Some((640, 480)), "prefix of {length} bytes");
            }
        }
    }
}
