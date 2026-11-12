//! Pure-Rust reading of binary document containers.
//!
//! Binary office documents are compressed archives: DOCX and XLSX are ZIP
//! containers of XML parts, and PDF content streams are often `zlib`-wrapped
//! `DEFLATE` data. This module provides that container layer without adding a
//! dependency on a compression crate:
//!
//! - [`inflate`](mod@inflate) — raw `DEFLATE` (RFC 1951), the `zlib` wrapper (RFC 1950), the
//!   `Adler-32`/`CRC-32` checksums, and the [`DEFAULT_MAX_OUTPUT_BYTES`]
//!   decompression-bomb guard.
//! - [`zip`](mod@zip) — the ZIP archive reader ([`ZipArchive`], [`ZipEntry`]): central
//!   directory parsing, entry listing, and CRC-checked entry extraction.
//!
//! The document parsers themselves (DOCX/XLSX/PDF to [`crate::Document`]) sit on
//! top of this layer and are not part of it.
//!
//! # Example
//!
//! ```
//! use openjiuwen_retrieval::parsing::{crc32, inflate, zlib_decompress, ZipArchive};
//!
//! // One final stored block holding "hi".
//! assert_eq!(inflate(&[0x01, 0x02, 0x00, 0xfd, 0xff, b'h', b'i'], 1024).unwrap(), b"hi");
//! assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
//! # let _ = |data: &[u8]| zlib_decompress(data, 1024);
//! # let _: Option<ZipArchive> = None;
//! ```

pub mod inflate;
pub mod zip;

pub use inflate::{adler32, crc32, inflate, zlib_decompress, DEFAULT_MAX_OUTPUT_BYTES};
pub use zip::{ZipArchive, ZipEntry};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn module_root_reexports_the_container_layer() {
        let stream = [
            0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00, 0x06, 0x2c, 0x02, 0x15,
        ];
        assert_eq!(
            zlib_decompress(&stream, DEFAULT_MAX_OUTPUT_BYTES).unwrap(),
            b"hello"
        );
        assert_eq!(
            inflate(&[0x01, 0x00, 0x00, 0xff, 0xff], 16).unwrap(),
            Vec::<u8>::new()
        );
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        let empty = ZipArchive::from_bytes(vec![
            0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])
        .unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.entry_names().len(), 0);
        let _: Option<&ZipEntry> = None;
    }
}
