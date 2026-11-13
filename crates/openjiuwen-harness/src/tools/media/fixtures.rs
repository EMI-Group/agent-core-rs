//! Minimal valid media headers used by the dispatch and end-to-end tool tests.
//!
//! Each helper returns the smallest header that is valid for its container, hand-built from the
//! format specification (a few dozen bytes), so the tests stay offline, deterministic and free of
//! binary fixtures. Per-format nuances are covered by the unit tests inside each parser module.

/// A 2x2 `RGBA` PNG.
pub(crate) fn png() -> Vec<u8> {
    let mut bytes = super::image::png::SIGNATURE.to_vec();
    let mut header = Vec::new();
    header.extend_from_slice(&2u32.to_be_bytes());
    header.extend_from_slice(&2u32.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    bytes.extend_from_slice(&13u32.to_be_bytes());
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes
}

/// A 2x2 baseline JPEG with three components.
pub(crate) fn jpeg() -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08];
    bytes.extend_from_slice(&2u16.to_be_bytes());
    bytes.extend_from_slice(&2u16.to_be_bytes());
    bytes.push(3);
    bytes.extend_from_slice(&[1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    bytes
}

/// A 1x1 `GIF89a` with one opaque frame.
pub(crate) fn gif() -> Vec<u8> {
    let mut bytes = b"GIF89a".to_vec();
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&[0, 0, 0]);
    bytes.push(0x2C);
    bytes.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0, 0x00, 0x02]);
    bytes.extend_from_slice(&[0x02, 0x4C, 0x01, 0x00]);
    bytes.push(0x3B);
    bytes
}

/// A 100x50 lossless WebP without alpha.
pub(crate) fn webp() -> Vec<u8> {
    let bits = 0x63u32 | (0x31u32 << 14);
    let mut payload = vec![0x2F];
    payload.extend_from_slice(&bits.to_le_bytes());
    let mut body = b"WEBP".to_vec();
    body.extend_from_slice(b"VP8L");
    body.extend_from_slice(
        &u32::try_from(payload.len())
            .unwrap_or_default()
            .to_le_bytes(),
    );
    body.extend_from_slice(&payload);
    if payload.len() % 2 == 1 {
        body.push(0);
    }
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or_default().to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes
}

/// A 4x3 24-bit uncompressed BMP.
pub(crate) fn bmp() -> Vec<u8> {
    let mut bytes = b"BM".to_vec();
    bytes.extend_from_slice(&54u32.to_le_bytes());
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes.extend_from_slice(&54u32.to_le_bytes());
    bytes.extend_from_slice(&40u32.to_le_bytes());
    bytes.extend_from_slice(&4i32.to_le_bytes());
    bytes.extend_from_slice(&3i32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&24u16.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&[0; 20]);
    bytes
}

/// A 1 kHz, 16-bit stereo WAV lasting one second.
pub(crate) fn wav() -> Vec<u8> {
    let mut body = b"WAVE".to_vec();
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&16u32.to_le_bytes());
    let mut format = Vec::new();
    format.extend_from_slice(&1u16.to_le_bytes());
    format.extend_from_slice(&2u16.to_le_bytes());
    format.extend_from_slice(&44_100u32.to_le_bytes());
    format.extend_from_slice(&176_400u32.to_le_bytes());
    format.extend_from_slice(&4u16.to_le_bytes());
    format.extend_from_slice(&16u16.to_le_bytes());
    body.extend_from_slice(&format);
    body.extend_from_slice(b"data");
    body.extend_from_slice(&176_400u32.to_le_bytes());
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or_default().to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes
}

/// A 44.1 kHz, 16-bit stereo FLAC holding one second of audio.
pub(crate) fn flac() -> Vec<u8> {
    let mut bytes = b"fLaC".to_vec();
    // Last metadata block, type 0 (STREAMINFO), 34 bytes.
    bytes.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]);
    let mut stream_info = Vec::new();
    stream_info.extend_from_slice(&4096u16.to_be_bytes());
    stream_info.extend_from_slice(&4096u16.to_be_bytes());
    stream_info.extend_from_slice(&[0, 0, 0]);
    stream_info.extend_from_slice(&[0, 0, 0]);
    let sample_rate = 44_100u64;
    let packed = (sample_rate << 44) | (1 << 41) | (0x0F << 36) | 0xAC44;
    stream_info.extend_from_slice(&packed.to_be_bytes());
    stream_info.extend_from_slice(&[0; 16]);
    bytes.extend_from_slice(&stream_info);
    bytes
}

/// A 128 kbps, 44.1 kHz stereo MP3 whose Xing header declares 100 frames.
pub(crate) fn mp3() -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xFB, 0x90, 0x00];
    // MPEG-1 stereo side information, then the Xing header with the frame count.
    bytes.extend_from_slice(&[0; 32]);
    bytes.extend_from_slice(b"Xing");
    bytes.extend_from_slice(&1u32.to_be_bytes());
    bytes.extend_from_slice(&100u32.to_be_bytes());
    bytes
}

/// A 48 kHz stereo Vorbis OGG lasting one second.
pub(crate) fn ogg() -> Vec<u8> {
    let mut identification = vec![0x01];
    identification.extend_from_slice(b"vorbis");
    identification.extend_from_slice(&0u32.to_le_bytes());
    identification.push(2);
    identification.extend_from_slice(&48_000u32.to_le_bytes());
    identification.extend_from_slice(&0i32.to_le_bytes());
    identification.extend_from_slice(&128_000i32.to_le_bytes());
    identification.extend_from_slice(&0i32.to_le_bytes());
    identification.push(0);
    identification.push(1);
    let mut bytes = ogg_page(0, 0, 0, &identification);
    bytes.extend_from_slice(&ogg_page(4, 48_000, 1, &[0]));
    bytes
}

/// Builds one Ogg page carrying a single packet in a single segment.
fn ogg_page(header_type: u8, granule: u64, sequence: u32, payload: &[u8]) -> Vec<u8> {
    let mut page = b"OggS".to_vec();
    page.push(0);
    page.push(header_type);
    page.extend_from_slice(&granule.to_le_bytes());
    page.extend_from_slice(&1u32.to_le_bytes());
    page.extend_from_slice(&sequence.to_le_bytes());
    page.extend_from_slice(&0u32.to_le_bytes());
    page.push(1);
    page.push(u8::try_from(payload.len()).unwrap_or_default());
    page.extend_from_slice(payload);
    page
}
