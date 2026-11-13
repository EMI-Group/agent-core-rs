//! Pure-Rust reading of binary documents.
//!
//! Binary office documents are compressed archives: DOCX, XLSX, PPTX, and ODT are
//! ZIP containers of XML parts, and PDF content streams are often `zlib`-wrapped
//! `DEFLATE` data. This module provides both layers without adding a dependency
//! on a compression or office-format crate:
//!
//! - [`inflate`](mod@inflate) — raw `DEFLATE` (RFC 1951), the `zlib` wrapper (RFC 1950), the
//!   `Adler-32`/`CRC-32` checksums, and the [`DEFAULT_MAX_OUTPUT_BYTES`]
//!   decompression-bomb guard.
//! - [`zip`](mod@zip) — the ZIP archive reader ([`ZipArchive`], [`ZipEntry`]) with
//!   central-directory parsing and CRC-checked extraction, plus the
//!   stored-entry [`ZipWriter`] used to build fixtures without a ZIP crate.
//! - [`xml`](mod@xml) — a tolerant XML pull parser ([`XmlDocument`], [`XmlElement`])
//!   for the `OOXML` and `OpenDocument` vocabularies.
//! - [`docx`](mod@docx), [`xlsx`](mod@xlsx), [`pptx`](mod@pptx), [`odt`](mod@odt),
//!   [`pdf`](mod@pdf) — the document formats themselves, rendered to
//!   Markdown-flavoured text or tabular data.
//!
//! Images contribute no text: there is no OCR or rendering stack, so image parts
//! are reported as out of scope rather than guessed at.
//!
//! # Example
//!
//! ```
//! use openjiuwen_retrieval::parsing::{crc32, inflate, xml, zlib_decompress, ZipArchive};
//!
//! // One final stored block holding "hi".
//! assert_eq!(inflate(&[0x01, 0x02, 0x00, 0xfd, 0xff, b'h', b'i'], 1024).unwrap(), b"hi");
//! assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
//! let document = xml::parse("<w:p><w:t>hi</w:t></w:p>").unwrap();
//! assert_eq!(document.root().unwrap().find_first("t").unwrap().text, "hi");
//! # let _ = |data: &[u8]| zlib_decompress(data, 1024);
//! # let _: Option<ZipArchive> = None;
//! ```

pub mod docx;
pub mod inflate;
pub mod odt;
pub mod pdf;
pub mod pptx;
pub mod xlsx;
pub mod xml;
pub mod zip;

pub use docx::{DocxBlock, DocxDocument, DocxParser, CORE_PROPERTIES_PART, DOCUMENT_PART};
pub use inflate::{adler32, crc32, inflate, zlib_decompress, DEFAULT_MAX_OUTPUT_BYTES};
pub use odt::{OdtDocument, OdtParser, CONTENT_PART, META_PART};
pub use pdf::{PdfDocument, PdfParser, KERNING_SPACE_THRESHOLD};
pub use pptx::{
    PptxDocument, PptxParser, PptxSlide, NOTES_SLIDES_DIR, PRESENTATION_PART,
    PRESENTATION_RELS_PART, SLIDES_DIR,
};
pub use xlsx::{
    SheetRef, XlsxParser, XlsxSheet, XlsxWorkbook, SHARED_STRINGS_PART, WORKBOOK_PART,
    WORKBOOK_RELS_PART,
};
pub use xml::{local_name, parse as parse_xml, XmlDocument, XmlElement, MAX_DEPTH};
pub use zip::{ZipArchive, ZipEntry, ZipWriter};

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
        let writer = ZipWriter::new();
        assert!(writer.is_empty());
        assert!(ZipArchive::from_bytes(writer.finish().unwrap())
            .unwrap()
            .is_empty());
        let _: Option<&ZipEntry> = None;
    }

    #[test]
    fn module_root_reexports_the_document_parsers() {
        let document = parse_xml("<a><b>text</b></a>").unwrap();
        assert_eq!(local_name("w:p"), "p");
        assert_eq!(document.root().unwrap().name, "a");
        assert_eq!(MAX_DEPTH, 256);
        assert!(DocxParser::new().parse_bytes(b"not a zip").is_err());
        assert!(XlsxParser::new().parse_bytes(b"not a zip").is_err());
        assert!(PptxParser::new().parse_bytes(b"not a zip").is_err());
        assert!(OdtParser::new().parse_bytes(b"not a zip").is_err());
        let pdf = PdfParser::new().parse_bytes(b"%PDF-1.4\n%%EOF").unwrap();
        assert!(!pdf.has_text);
        assert_eq!(KERNING_SPACE_THRESHOLD, -100.0);
        assert_eq!(DOCUMENT_PART, "word/document.xml");
        assert_eq!(CORE_PROPERTIES_PART, "docProps/core.xml");
        assert_eq!(WORKBOOK_PART, "xl/workbook.xml");
        assert_eq!(WORKBOOK_RELS_PART, "xl/_rels/workbook.xml.rels");
        assert_eq!(SHARED_STRINGS_PART, "xl/sharedStrings.xml");
        assert_eq!(PRESENTATION_PART, "ppt/presentation.xml");
        assert_eq!(PRESENTATION_RELS_PART, "ppt/_rels/presentation.xml.rels");
        assert_eq!(SLIDES_DIR, "ppt/slides/");
        assert_eq!(NOTES_SLIDES_DIR, "ppt/notesSlides/");
        assert_eq!(CONTENT_PART, "content.xml");
        assert_eq!(META_PART, "meta.xml");
        let workbook = XlsxWorkbook::default();
        assert!(workbook.is_empty());
        assert!(workbook.sheet(SheetRef::Index(0)).is_none());
        let _: Option<DocxDocument> = None;
        let _: Option<DocxBlock> = None;
        let _: Option<XlsxSheet> = None;
        let _: Option<PptxDocument> = None;
        let _: Option<PptxSlide> = None;
        let _: Option<OdtDocument> = None;
        let _: Option<PdfDocument> = None;
    }
}
