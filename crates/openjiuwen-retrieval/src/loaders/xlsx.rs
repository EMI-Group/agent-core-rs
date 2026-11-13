//! `XLSX` loader: row and column documents per worksheet.
//!
//! Port of the Excel half of the Python `ExcelParser`
//! (`indexing/processor/parser/excel_parser.py`): the workbook is parsed by
//! [`XlsxParser`] and every sheet is converted with `XlsxWorkbook::to_documents`
//! into the same row/column document scheme the [`CsvLoader`](super::CsvLoader)
//! emits. The document-id base is the file stem, which mirrors Python's
//! `base_id = doc_id or doc` and keeps ids readable (`people_s0_r2`).

use async_trait::async_trait;

use openjiuwen_common::error::Result;
use openjiuwen_common::logging::debug;

use crate::common::Document;
use crate::parsing::XlsxParser;

use super::{file_extension, file_stem, DocumentLoader};

/// Loads `.xlsx` workbooks as row documents followed by column documents.
#[derive(Debug, Default, Clone, Copy)]
pub struct XlsxLoader;

impl XlsxLoader {
    /// Creates an `XLSX` loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentLoader for XlsxLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".xlsx"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let workbook = XlsxParser::new().parse_path(source).await?;
        if workbook.is_empty() {
            debug!(file = source, "loaded an XLSX workbook with no sheets");
            return Ok(Vec::new());
        }
        let documents = workbook.to_documents(&file_stem(source));
        debug!(
            file = source,
            documents = documents.len(),
            "loaded XLSX workbook"
        );
        Ok(documents)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::Value;

    use crate::loaders::fixtures::build_zip;

    /// One-sheet workbook with a header row and one data row.
    const WORKBOOK_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="People" sheetId="1" r:id="rId1"/></sheets>
</workbook>"#;

    /// Relationship pointing at the single worksheet.
    const RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#;

    /// Worksheet: `Name`/`City` header plus one data row.
    const SHEET_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>Name</t></is></c>
      <c r="B1" t="inlineStr"><is><t>City</t></is></c>
    </row>
    <row r="2">
      <c r="A2" t="inlineStr"><is><t>alice</t></is></c>
      <c r="B2" t="inlineStr"><is><t>paris</t></is></c>
    </row>
  </sheetData>
</worksheet>"#;

    /// Builds the one-sheet workbook archive.
    fn workbook_archive() -> Vec<u8> {
        build_zip(&[
            ("xl/workbook.xml", WORKBOOK_XML),
            ("xl/_rels/workbook.xml.rels", RELS_XML),
            ("xl/worksheets/sheet1.xml", SHEET_XML),
        ])
    }

    #[tokio::test]
    async fn load_emits_row_and_column_documents_keyed_by_the_file_stem() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.xlsx");
        std::fs::write(&path, workbook_archive()).unwrap();

        let documents = XlsxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "Name: alice, City: paris",
                "Column name: Name. Values: alice",
                "Column name: City. Values: paris",
            ]
        );
        assert_eq!(documents[0].id, "people_s0_r2");
        assert_eq!(documents[1].id, "people_s0_c0");
        assert_eq!(documents[2].id, "people_s0_c1");
        assert_eq!(
            documents[0].metadata.get("sheet_name"),
            Some(&Value::from("People"))
        );
        assert_eq!(
            documents[0].metadata.get("source_type"),
            Some(&Value::from("row"))
        );
        assert_eq!(
            documents[1].metadata.get("column_name"),
            Some(&Value::from("Name"))
        );
    }

    #[tokio::test]
    async fn empty_workbook_yields_no_documents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.xlsx");
        std::fs::write(
            &path,
            build_zip(&[("xl/workbook.xml", "<workbook><sheets/></workbook>")]),
        )
        .unwrap();

        let documents = XlsxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents.len(), 0);
    }

    #[tokio::test]
    async fn missing_and_corrupt_workbooks_report_typed_errors() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.xlsx");
        let error = XlsxLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);

        let corrupt = dir.path().join("corrupt.xlsx");
        std::fs::write(&corrupt, b"not a zip").unwrap();
        let error = XlsxLoader::new()
            .load(corrupt.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let no_workbook = dir.path().join("nopart.xlsx");
        std::fs::write(
            &no_workbook,
            build_zip(&[("xl/sharedStrings.xml", "<sst/>")]),
        )
        .unwrap();
        let error = XlsxLoader::new()
            .load(no_workbook.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(error.to_string().contains("xl/workbook.xml"));
    }

    #[test]
    fn supports_xlsx_only_case_insensitively() {
        let loader = XlsxLoader::new();
        assert!(loader.supports("a.xlsx"));
        assert!(loader.supports("A.XLSX"));
        assert!(!loader.supports("a.xls"));
        assert!(!loader.supports("a.csv"));
    }
}
