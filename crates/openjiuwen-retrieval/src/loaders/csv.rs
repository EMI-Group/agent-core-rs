//! `CSV`/`TSV` loader emitting row-wise and column-wise documents.
//!
//! Port of the `CSV`/`TSV` half of the Python `ExcelParser` (`excel_parser.py`):
//! every data row becomes one document (row retrieval), and every header column
//! becomes one document (column retrieval). The `XLSX` branch needs `openpyxl`
//! and is out of scope, so this loader only handles delimited text.
//!
//! Python delegates parsing to the stdlib `csv` module; the Rust port carries a
//! small RFC-4180-style parser instead of adding a dependency.

use async_trait::async_trait;
use serde_json::{Map, Value};

use openjiuwen_common::error::Result;

use crate::common::Document;

use super::{
    apply_load_options, file_extension, file_stem, read_source, DocumentBytes, DocumentLoader,
    LoadOptions,
};

/// The sheet index stamped on every document; delimited files have one sheet.
const SHEET_INDEX: usize = 0;

/// Loads `.csv` and `.tsv` files as row documents followed by column documents.
#[derive(Debug, Clone, Copy)]
pub struct CsvLoader {
    include_header: bool,
    delimiter: Option<u8>,
}

impl CsvLoader {
    /// Creates a loader that prefixes every row with its column name.
    #[must_use]
    pub fn new() -> Self {
        Self {
            include_header: true,
            delimiter: None,
        }
    }

    /// Keeps (the default) or drops the `Header: value` prefixes.
    #[must_use]
    pub fn with_include_header(mut self, include_header: bool) -> Self {
        self.include_header = include_header;
        self
    }

    /// Overrides the delimiter inferred from the file extension.
    #[must_use]
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = Some(delimiter);
        self
    }
}

impl Default for CsvLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentLoader for CsvLoader {
    fn supports(&self, source: &str) -> bool {
        matches!(file_extension(source).as_str(), ".csv" | ".tsv")
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let raw = read_source(source).await?;
        Ok(documents_from_delimited(
            source,
            &raw,
            self.include_header,
            self.delimiter,
        ))
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let raw = String::from_utf8_lossy(&source.bytes);
        let mut documents =
            documents_from_delimited(&source.name, &raw, self.include_header, self.delimiter);
        apply_load_options(&source.name, &mut documents, options);
        Ok(documents)
    }
}

/// Turns delimited content into row documents followed by column documents.
///
/// `name` is the path or in-memory source name: it supplies the extension for
/// the delimiter, the id base, and the sheet name (`file_stem(name)`), exactly
/// as Python's `base_id = doc_id or doc` uses the source string.
fn documents_from_delimited(
    name: &str,
    raw: &str,
    include_header: bool,
    delimiter: Option<u8>,
) -> Vec<Document> {
    // Python opens the file with `utf-8-sig`, which drops a leading BOM.
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    let rows = parse_delimited(raw, resolve_delimiter(delimiter, name));
    if rows.is_empty() {
        return Vec::new();
    }
    build_documents(&rows, &file_stem(name), name, include_header)
}

/// Resolves the delimiter: explicit configuration wins over the extension.
fn resolve_delimiter(configured: Option<u8>, source: &str) -> u8 {
    configured.unwrap_or(if file_extension(source) == ".tsv" {
        b'\t'
    } else {
        b','
    })
}

/// Builds the row documents followed by the column documents.
///
/// `base` is the `source` string as passed to [`CsvLoader::load`], matching
/// Python's `base_id = doc_id or doc`: every generated id is prefixed with it.
fn build_documents(
    rows: &[Vec<String>],
    sheet_name: &str,
    base: &str,
    include_header: bool,
) -> Vec<Document> {
    let mut documents = Vec::new();
    let Some(header_row) = rows.first() else {
        return documents;
    };
    let headers: Vec<&str> = header_row.iter().map(|cell| cell.trim()).collect();
    let data_rows = &rows[1..];

    for (offset, row) in data_rows.iter().enumerate() {
        // Python enumerates data rows from 2, so the first data row is row 2.
        let row_index = offset + 2;
        let mut parts: Vec<String> = Vec::new();
        for (column, cell) in row.iter().enumerate() {
            // Python zips headers with the row, so extra cells are dropped.
            let Some(header) = headers.get(column).copied() else {
                break;
            };
            let value = cell.trim();
            if include_header {
                if !header.is_empty() {
                    parts.push(format!("{header}: {value}"));
                }
            } else if !value.is_empty() {
                parts.push(value.to_owned());
            }
        }
        if parts.is_empty() {
            continue;
        }
        documents.push(Document {
            id: format!("{base}_s{SHEET_INDEX}_r{row_index}"),
            text: parts.join(", "),
            metadata: record_metadata(sheet_name, Some(row_index), "row"),
        });
    }

    for (column, header) in headers.iter().enumerate() {
        // Empty header cells (and every cell when the header is suppressed) are
        // named positionally so column documents always have a stable name.
        let column_name = if include_header && !header.is_empty() {
            (*header).to_owned()
        } else {
            format!("column_{column}")
        };
        let values: Vec<&str> = data_rows
            .iter()
            .filter_map(|row| row.get(column))
            .map(|cell| cell.trim())
            .filter(|value| !value.is_empty())
            .collect();
        let text = if values.is_empty() {
            format!("Column name: {column_name}. Values: (empty)")
        } else {
            format!("Column name: {column_name}. Values: {}", values.join(", "))
        };
        let mut metadata = record_metadata(sheet_name, None, "column");
        metadata.insert("column_name".to_owned(), Value::String(column_name));
        documents.push(Document {
            id: format!("{base}_s{SHEET_INDEX}_c{column}"),
            text,
            metadata,
        });
    }

    documents
}

/// Builds the `{sheet_name, row_index?, source_type}` metadata block.
fn record_metadata(
    sheet_name: &str,
    row_index: Option<usize>,
    source_type: &str,
) -> Map<String, Value> {
    let mut metadata = Map::new();
    metadata.insert(
        "sheet_name".to_owned(),
        Value::String(sheet_name.to_owned()),
    );
    if let Some(row_index) = row_index {
        metadata.insert("row_index".to_owned(), Value::from(row_index));
    }
    metadata.insert(
        "source_type".to_owned(),
        Value::String(source_type.to_owned()),
    );
    metadata
}

/// Parses RFC-4180-style delimited text into rows of fields.
///
/// Handles double-quoted fields (`""` escapes, embedded delimiters and newlines),
/// `CRLF`/`LF` record breaks, and a trailing line break (which does not create an
/// extra row). A blank line yields an empty row, like Python's `csv.reader`.
fn parse_delimited(text: &str, delimiter: u8) -> Vec<Vec<String>> {
    let bytes = text.as_bytes();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field: Vec<u8> = Vec::new();
    let mut in_quotes = false;
    let mut field_started = false;
    let mut index = 0;

    while index < bytes.len() {
        let byte = bytes[index];
        if in_quotes {
            if byte == b'"' {
                if bytes.get(index + 1) == Some(&b'"') {
                    field.push(b'"');
                    index += 2;
                } else {
                    in_quotes = false;
                    index += 1;
                }
            } else {
                field.push(byte);
                index += 1;
            }
            continue;
        }
        match byte {
            b'"' if field.is_empty() => {
                in_quotes = true;
                field_started = true;
                index += 1;
            }
            b'\r' => {
                finish_record(&mut rows, &mut row, &mut field, &mut field_started);
                index += usize::from(bytes.get(index + 1) == Some(&b'\n')) + 1;
            }
            b'\n' => {
                finish_record(&mut rows, &mut row, &mut field, &mut field_started);
                index += 1;
            }
            byte if byte == delimiter => {
                row.push(finish_field(&mut field));
                field_started = false;
                index += 1;
            }
            byte => {
                field.push(byte);
                field_started = true;
                index += 1;
            }
        }
    }

    if !row.is_empty() || field_started {
        row.push(finish_field(&mut field));
        rows.push(row);
    }
    rows
}

/// Takes the accumulated bytes of one field as (lossy) `UTF-8`.
fn finish_field(field: &mut Vec<u8>) -> String {
    let value = String::from_utf8_lossy(field).into_owned();
    field.clear();
    value
}

/// Ends the current record, pushing an empty row for a blank line.
fn finish_record(
    rows: &mut Vec<Vec<String>>,
    row: &mut Vec<String>,
    field: &mut Vec<u8>,
    field_started: &mut bool,
) {
    if row.is_empty() && !*field_started {
        rows.push(Vec::new());
    } else {
        row.push(finish_field(field));
        rows.push(std::mem::take(row));
    }
    *field_started = false;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn csv_emits_row_then_column_documents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.CSV");
        std::fs::write(&path, "name,city\nalice,paris\nbob,\n").unwrap();

        let documents = CsvLoader::new().load(path.to_str().unwrap()).await.unwrap();
        let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "name: alice, city: paris",
                "name: bob, city: ",
                "Column name: name. Values: alice, bob",
                "Column name: city. Values: paris",
            ]
        );

        let base = path.to_str().unwrap();
        assert_eq!(documents[0].id, format!("{base}_s0_r2"));
        assert_eq!(documents[1].id, format!("{base}_s0_r3"));
        assert_eq!(documents[2].id, format!("{base}_s0_c0"));
        assert_eq!(documents[3].id, format!("{base}_s0_c1"));
        assert_eq!(
            documents[0].metadata.get("row_index"),
            Some(&Value::from(2_u64))
        );
        assert_eq!(
            documents[0].metadata.get("source_type"),
            Some(&Value::String("row".to_owned()))
        );
        assert_eq!(
            documents[2].metadata.get("column_name"),
            Some(&Value::String("name".to_owned()))
        );
        assert_eq!(
            documents[2].metadata.get("source_type"),
            Some(&Value::String("column".to_owned()))
        );
    }

    #[tokio::test]
    async fn header_only_file_emits_columns_with_empty_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.csv");
        std::fs::write(&path, "a,b\n").unwrap();

        let documents = CsvLoader::new().load(path.to_str().unwrap()).await.unwrap();
        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].text, "Column name: a. Values: (empty)");
        assert_eq!(documents[1].text, "Column name: b. Values: (empty)");
    }

    #[tokio::test]
    async fn include_header_false_keeps_values_only_and_positional_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("values.csv");
        std::fs::write(&path, "a,b\nx,\ny,z\n").unwrap();

        let documents = CsvLoader::new()
            .with_include_header(false)
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "x",
                "y, z",
                "Column name: column_0. Values: x, y",
                "Column name: column_1. Values: z",
            ]
        );
        assert_eq!(
            documents[2].metadata.get("column_name"),
            Some(&Value::String("column_0".to_owned()))
        );
    }

    #[tokio::test]
    async fn tsv_delimiter_and_quoted_fields_follow_rfc_4180() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.tsv");
        std::fs::write(
            &path,
            "id\tnote\r\n1\t\"hello, world\"\r\n2\t\"say \"\"hi\"\"\"\r\n",
        )
        .unwrap();

        let documents = CsvLoader::new().load(path.to_str().unwrap()).await.unwrap();
        assert_eq!(documents[0].text, "id: 1, note: hello, world");
        assert_eq!(documents[1].text, "id: 2, note: say \"hi\"");
        assert_eq!(documents[2].text, "Column name: id. Values: 1, 2");
        assert_eq!(
            documents[3].text,
            "Column name: note. Values: hello, world, say \"hi\""
        );
    }

    #[tokio::test]
    async fn custom_delimiter_overrides_the_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pipes.csv");
        std::fs::write(&path, "a|b\n1|2\n").unwrap();

        let documents = CsvLoader::new()
            .with_delimiter(b'|')
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents[0].text, "a: 1, b: 2");
    }

    #[tokio::test]
    async fn empty_file_yields_no_documents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nothing.csv");
        std::fs::write(&path, "").unwrap();

        let documents = CsvLoader::new().load(path.to_str().unwrap()).await.unwrap();
        assert_eq!(documents, Vec::<Document>::new());
    }

    #[tokio::test]
    async fn load_bytes_matches_the_path_result() {
        let source = DocumentBytes::new("people.CSV", b"name,city\nalice,paris\nbob,\n".to_vec());
        let documents = CsvLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "name: alice, city: paris",
                "name: bob, city: ",
                "Column name: name. Values: alice, bob",
                "Column name: city. Values: paris",
            ]
        );
        assert_eq!(documents[0].id, "people.CSV_s0_r2");
        assert_eq!(documents[1].id, "people.CSV_s0_r3");
        assert_eq!(documents[2].id, "people.CSV_s0_c0");
        assert_eq!(documents[3].id, "people.CSV_s0_c1");
        assert_eq!(
            documents[0].metadata.get("sheet_name"),
            Some(&Value::String("people".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("row_index"),
            Some(&Value::from(2_u64))
        );
    }

    #[tokio::test]
    async fn load_bytes_prefixes_multi_document_ids_with_doc_id_and_title() {
        let source = DocumentBytes::new("people.csv", b"name,city\nalice,paris\n".to_vec());
        let options = LoadOptions::new()
            .with_doc_id("doc-base")
            .with_title("People");
        let documents = CsvLoader::new()
            .load_bytes(&source, &options)
            .await
            .unwrap();
        assert_eq!(documents[0].id, "doc-base_s0_r2");
        assert_eq!(documents[1].id, "doc-base_s0_c0");
        for document in &documents {
            assert_eq!(
                document.metadata.get("doc_id"),
                Some(&Value::String(document.id.clone()))
            );
            assert_eq!(
                document.metadata.get("title"),
                Some(&Value::String("People".to_owned()))
            );
        }
    }

    #[tokio::test]
    async fn load_bytes_strips_a_leading_bom() {
        let source = DocumentBytes::new("bom.csv", "\u{feff}a,b\n1,2\n".as_bytes().to_vec());
        let documents = CsvLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents[0].text, "a: 1, b: 2");
        assert_eq!(documents[1].text, "Column name: a. Values: 1");
    }

    #[tokio::test]
    async fn load_bytes_resolves_the_delimiter_from_the_name() {
        let tsv = DocumentBytes::new("table.tsv", b"id\tnote\n1\thi\n".to_vec());
        let documents = CsvLoader::new()
            .load_bytes(&tsv, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents[0].text, "id: 1, note: hi");

        let custom = DocumentBytes::new("pipes.csv", b"a|b\n1|2\n".to_vec());
        let documents = CsvLoader::new()
            .with_delimiter(b'|')
            .load_bytes(&custom, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents[0].text, "a: 1, b: 2");
    }

    #[tokio::test]
    async fn load_bytes_of_an_empty_file_yields_no_documents() {
        let source = DocumentBytes::new("nothing.csv", Vec::new());
        let documents = CsvLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents, Vec::<Document>::new());
    }

    #[test]
    fn parser_handles_blank_lines_quotes_and_trailing_newlines() {
        assert_eq!(
            parse_delimited("a,b\n\nc\n", b','),
            vec![
                vec!["a".to_owned(), "b".to_owned()],
                Vec::new(),
                vec!["c".to_owned()],
            ]
        );
        assert_eq!(parse_delimited("x\n", b','), vec![vec!["x".to_owned()]]);
        assert_eq!(
            parse_delimited("\"a\nb\",c", b','),
            vec![vec!["a\nb".to_owned(), "c".to_owned()]]
        );
        assert_eq!(parse_delimited("\"\"", b','), vec![vec![String::new()]]);
        assert_eq!(parse_delimited("", b','), Vec::<Vec<String>>::new());
    }

    #[test]
    fn loader_extension_and_sheet_name() {
        let loader = CsvLoader::new();
        assert!(loader.supports("a.csv"));
        assert!(loader.supports("a.TSV"));
        assert!(!loader.supports("a.xlsx"));
        assert_eq!(resolve_delimiter(None, "a.tsv"), b'\t');
        assert_eq!(resolve_delimiter(None, "a.csv"), b',');
        assert_eq!(resolve_delimiter(Some(b';'), "a.csv"), b';');
        assert_eq!(file_stem("/tmp/data.v2.csv"), "data.v2");
    }
}
