//! `XLSX` (`SpreadsheetML`) reading.
//!
//! Port of the Excel half of the Python `ExcelParser`
//! (`indexing/processor/parser/excel_parser.py`) without `openpyxl`: the
//! workbook, relationship, shared-string, and worksheet parts are read straight
//! out of the `ZIP` container and turned into [`TableData`] tables.
//!
//! Part resolution follows the `OPC` packaging rules closely enough for real
//! workbooks:
//!
//! - `xl/workbook.xml` lists the sheets (`<sheet name="…" r:id="rIdN"/>`);
//! - `xl/_rels/workbook.xml.rels` maps each `rIdN` to a part target, which is
//!   normalised (leading `/`, `./`, `../`) against the `xl` directory;
//! - a workbook without a relationships part falls back to the conventional
//!   `xl/worksheets/sheetN.xml` path for the sheet's position;
//! - `xl/sharedStrings.xml` is optional, and `<si>` entries concatenate every
//!   `<t>` descendant, so rich-text runs come out as one string (phonetic runs
//!   `<rPh>` are skipped).
//!
//! Cell decoding: `t="s"` resolves a shared-string index, `t="inlineStr"` reads
//! `<is>`, `t="b"` maps `1`/`0` to `true`/`false`, `t="str"`/`t="e"`/`t="d"`
//! keep their literal text, and anything else is numeric. Numbers keep their
//! literal sheet text except for a fractional part that is entirely zeros
//! (`12.0` → `12`), which keeps row and column documents free of `openpyxl`'s
//! float formatting. Blank and missing cells are padded so every row is the
//! width of the widest column reference, and `A1`-style references (including
//! `AA`, `AB`, …) place each cell in its own column.
//!
//! The first row that holds any text becomes the header, matching the
//! reference's "first row is the header" rule for a normal sheet; rows after it
//! are data rows, and the generated documents reuse
//! [`TableData::to_documents`] (row documents `{base}_s{sheet}_r{row}`,
//! column documents `{base}_s{sheet}_c{column}`, `sheet_name`/`source_type`/
//! `row_index`/`column_name` metadata).
//!
//! A sheet keeps no row numbers: `row_index` therefore counts document rows with
//! the header as row 1, which matches the reference when the header is the
//! sheet's first row.

use std::collections::HashMap;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::debug;
use serde::{Deserialize, Serialize};

use crate::common::Document;
use crate::multimodal::TableData;

use super::xml::{local_name, parse, XmlDocument, XmlElement};
use super::ZipArchive;

/// Archive path of the workbook part.
pub const WORKBOOK_PART: &str = "xl/workbook.xml";

/// Archive path of the workbook relationships part.
pub const WORKBOOK_RELS_PART: &str = "xl/_rels/workbook.xml.rels";

/// Archive path of the shared-string table.
pub const SHARED_STRINGS_PART: &str = "xl/sharedStrings.xml";

/// Directory the workbook's relationship targets are relative to.
const WORKBOOK_DIR: &str = "xl";

/// Longest accepted `A1`-style column prefix, which bounds the decoded index.
const MAX_COLUMN_LETTERS: usize = 8;

/// One worksheet: its name, workbook position, and decoded table.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct XlsxSheet {
    /// Sheet name (`<sheet name="…"/>`), or `Sheet{n}` when the attribute is absent.
    pub name: String,
    /// Zero-based position in the workbook.
    pub index: usize,
    /// Header row plus data rows, padded to a common width.
    pub table: TableData,
}

impl XlsxSheet {
    /// Builds this sheet's row and column documents.
    ///
    /// Delegates to [`TableData::to_documents`] and re-stamps the sheet index, so
    /// ids are `{base_id}_s{index}_r{row}` and `{base_id}_s{index}_c{column}`.
    pub fn to_documents(&self, base_id: &str) -> Vec<Document> {
        self.table
            .to_documents(base_id, Some(&self.name))
            .into_iter()
            .map(|document| retarget_sheet(document, self.index))
            .collect()
    }
}

/// A parsed workbook: its sheets in workbook order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct XlsxWorkbook {
    /// Sheets in workbook order; an empty workbook yields an empty list.
    pub sheets: Vec<XlsxSheet>,
}

impl XlsxWorkbook {
    /// Returns the sheet selected by name (`"Sheet1"`) or position (`0`).
    ///
    /// ```
    /// # use openjiuwen_retrieval::parsing::xlsx::{XlsxSheet, XlsxWorkbook};
    /// let workbook = XlsxWorkbook { sheets: vec![XlsxSheet { name: "Data".to_owned(), ..XlsxSheet::default() }] };
    /// assert!(workbook.sheet("Data").is_some());
    /// assert!(workbook.sheet(0).is_some());
    /// assert!(workbook.sheet("Missing").is_none());
    /// ```
    pub fn sheet<'a>(&self, which: impl Into<SheetRef<'a>>) -> Option<&XlsxSheet> {
        match which.into() {
            SheetRef::Name(name) => self.sheets.iter().find(|sheet| sheet.name == name),
            SheetRef::Index(index) => self.sheets.get(index),
        }
    }

    /// Returns the sheet names in workbook order.
    pub fn sheet_names(&self) -> Vec<&str> {
        self.sheets
            .iter()
            .map(|sheet| sheet.name.as_str())
            .collect()
    }

    /// Number of sheets.
    pub fn len(&self) -> usize {
        self.sheets.len()
    }

    /// Returns `true` when the workbook holds no sheets.
    pub fn is_empty(&self) -> bool {
        self.sheets.is_empty()
    }

    /// Builds the row and column documents of every sheet, sheet-major.
    ///
    /// This is the reference `ExcelParser` document shape: row documents first,
    /// then column documents, for each sheet in order.
    pub fn to_documents(&self, base_id: &str) -> Vec<Document> {
        self.sheets
            .iter()
            .flat_map(|sheet| sheet.to_documents(base_id))
            .collect()
    }
}

/// Selects a sheet by name or by workbook position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SheetRef<'a> {
    /// Match the sheet name exactly.
    Name(&'a str),
    /// Match the zero-based workbook position.
    Index(usize),
}

impl<'a> From<&'a str> for SheetRef<'a> {
    fn from(name: &'a str) -> Self {
        Self::Name(name)
    }
}

impl From<usize> for SheetRef<'_> {
    fn from(index: usize) -> Self {
        Self::Index(index)
    }
}

/// Reads `XLSX` workbooks.
#[derive(Debug, Clone, Copy, Default)]
pub struct XlsxParser;

impl XlsxParser {
    /// Creates a parser.
    pub fn new() -> Self {
        Self
    }

    /// Parses a whole `.xlsx` file from its bytes.
    ///
    /// # Errors
    ///
    /// Propagates the [`ZipArchive`] errors (a corrupt container) and returns
    /// [`Error::Retrieval`] when a required part is missing (the message names the
    /// part). Malformed part XML is [`Error::Validation`]. A workbook with no
    /// sheets is not an error.
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<XlsxWorkbook> {
        let archive = ZipArchive::from_bytes(bytes.to_vec())?;
        self.parse_archive(&archive)
    }

    /// Parses an already opened archive.
    ///
    /// # Errors
    ///
    /// Same as [`XlsxParser::parse_bytes`].
    pub fn parse_archive(&self, archive: &ZipArchive) -> Result<XlsxWorkbook> {
        let workbook = read_part(archive, WORKBOOK_PART)?;
        let relationships = read_relationships(archive);
        let shared_strings = read_shared_strings(archive)?;
        let sheet_elements = workbook
            .root()
            .and_then(|root| root.find_first("sheets"))
            .map(|sheets| sheets.children_named("sheet"))
            .unwrap_or_default();
        let mut parsed = Vec::with_capacity(sheet_elements.len());
        for (index, element) in sheet_elements.iter().enumerate() {
            let name = element
                .attr("name")
                .map_or_else(|| format!("Sheet{}", index + 1), str::to_owned);
            let target = sheet_target(archive, &relationships, element, index)?;
            let markup = read_part_text(archive, &target)?;
            let table = build_table(&markup, &shared_strings)
                .map_err(|error| error.with_context(&target))?;
            parsed.push(XlsxSheet { name, index, table });
        }
        Ok(XlsxWorkbook { sheets: parsed })
    }

    /// Parses the `.xlsx` file at `path` (read with `tokio::fs`).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for a missing file, [`Error::Transport`] for any other
    /// I/O failure, plus every [`XlsxParser::parse_bytes`] error.
    pub async fn parse_path(&self, path: impl AsRef<std::path::Path>) -> Result<XlsxWorkbook> {
        let path = path.as_ref();
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => {
                    Error::NotFound(format!("File {} does not exist", path.display()))
                }
                _ => Error::Transport(format!("reading {}: {error}", path.display())),
            })?;
        self.parse_bytes(&bytes)
    }
}

/// Reads one required package part as text.
fn read_part_text(archive: &ZipArchive, part: &str) -> Result<String> {
    if !archive.has_entry(part) {
        return Err(Error::Retrieval(format!("XLSX archive has no {part} part")));
    }
    archive.read_text(part)
}

/// Reads and parses one required package part.
fn read_part(archive: &ZipArchive, part: &str) -> Result<XmlDocument> {
    let markup = read_part_text(archive, part)?;
    parse(&markup).map_err(|error| error.with_context(part))
}

/// Reads the workbook relationship map (`rId` → `Target`).
///
/// A missing or unreadable relationships part yields an empty map and the
/// conventional worksheet paths are used instead.
fn read_relationships(archive: &ZipArchive) -> HashMap<String, String> {
    let mut relationships = HashMap::new();
    if !archive.has_entry(WORKBOOK_RELS_PART) {
        return relationships;
    }
    let markup = match archive.read_text(WORKBOOK_RELS_PART) {
        Ok(markup) => markup,
        Err(error) => {
            debug!("ignoring unreadable {WORKBOOK_RELS_PART}: {error}");
            return relationships;
        }
    };
    let document = match parse(&markup) {
        Ok(document) => document,
        Err(error) => {
            debug!("ignoring unparsable {WORKBOOK_RELS_PART}: {error}");
            return relationships;
        }
    };
    let Some(root) = document.root() else {
        return relationships;
    };
    for relationship in root.find_all("Relationship") {
        if let (Some(id), Some(target)) = (relationship.attr("Id"), relationship.attr("Target")) {
            relationships.insert(id.trim().to_owned(), target.trim().to_owned());
        }
    }
    relationships
}

/// Reads the shared-string table; an absent part means "no shared strings".
fn read_shared_strings(archive: &ZipArchive) -> Result<Vec<String>> {
    if !archive.has_entry(SHARED_STRINGS_PART) {
        return Ok(Vec::new());
    }
    let markup = archive.read_text(SHARED_STRINGS_PART)?;
    let document = parse(&markup).map_err(|error| error.with_context(SHARED_STRINGS_PART))?;
    Ok(document
        .root()
        .map(|root| {
            root.find_all("si")
                .iter()
                .map(|entry| shared_string_text(entry))
                .collect()
        })
        .unwrap_or_default())
}

/// Returns the archive path of `sheet`'s worksheet part.
fn sheet_target(
    archive: &ZipArchive,
    relationships: &HashMap<String, String>,
    sheet: &XmlElement,
    index: usize,
) -> Result<String> {
    let name = sheet.attr("name").unwrap_or_default();
    if let Some(id) = sheet.attr("r:id").or(sheet.attr("id")) {
        if let Some(target) = relationships.get(id.trim()) {
            return Ok(resolve_part(WORKBOOK_DIR, target));
        }
    }
    let fallback = format!("{WORKBOOK_DIR}/worksheets/sheet{}.xml", index + 1);
    if archive.has_entry(&fallback) {
        return Ok(fallback);
    }
    Err(Error::Retrieval(format!(
        "XLSX sheet {name:?} has no workbook relationship target and no {fallback} part"
    )))
}

/// Resolves a relationship target against the workbook directory.
fn resolve_part(base_dir: &str, target: &str) -> String {
    match target.strip_prefix('/') {
        Some(absolute) => normalize_part(absolute),
        None => normalize_part(&format!("{base_dir}/{target}")),
    }
}

/// Collapses `.`/`..` segments and duplicate slashes in a package path.
fn normalize_part(path: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    segments.join("/")
}

/// Builds one sheet's table from its worksheet XML.
fn build_table(markup: &str, shared_strings: &[String]) -> Result<TableData> {
    let document = parse(markup)?;
    let Some(root) = document.root() else {
        return Ok(TableData::default());
    };
    let Some(sheet_data) = root.find_first("sheetData") else {
        return Ok(TableData::default());
    };
    let mut grid: Vec<(usize, Vec<String>)> = Vec::new();
    let mut next_row = 1usize;
    for row in sheet_data.children_named("row") {
        let row_number = row
            .attr("r")
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|number| *number > 0)
            .unwrap_or(next_row);
        next_row = row_number + 1;
        let mut cells: Vec<String> = Vec::new();
        let mut next_column = 0usize;
        for cell in row.children_named("c") {
            let column = cell
                .attr("r")
                .and_then(column_index_from_reference)
                .unwrap_or(next_column);
            next_column = column + 1;
            if cells.len() <= column {
                cells.resize(column + 1, String::new());
            }
            cells[column] = cell_value(cell, shared_strings);
        }
        grid.push((row_number, cells));
    }
    grid.sort_by_key(|(number, _)| *number);

    let Some(start) = grid
        .iter()
        .position(|(_, cells)| cells.iter().any(|cell| !cell.trim().is_empty()))
    else {
        return Ok(TableData::default());
    };
    let mut grid = grid.into_iter().skip(start);
    let Some((header_number, mut headers)) = grid.next() else {
        return Ok(TableData::default());
    };
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut previous = header_number;
    for (number, cells) in grid {
        // Preserve the gaps so document row indices stay contiguous.
        for _ in (previous + 1)..number {
            rows.push(Vec::new());
        }
        rows.push(cells);
        previous = number;
    }
    let width = headers
        .len()
        .max(rows.iter().map(Vec::len).max().unwrap_or(0));
    headers.resize(width, String::new());
    for row in &mut rows {
        row.resize(width, String::new());
    }
    Ok(TableData::new(headers, rows))
}

/// Returns the cell value of `c` in document text form.
fn cell_value(cell: &XmlElement, shared_strings: &[String]) -> String {
    let kind = cell.attr("t").unwrap_or("n").trim();
    match kind {
        "s" => cell
            .find_first("v")
            .and_then(|value| value.text_content().trim().parse::<usize>().ok())
            .and_then(|index| shared_strings.get(index).cloned())
            .unwrap_or_default(),
        "inlineStr" => cell
            .find_first("is")
            .map(shared_string_text)
            .unwrap_or_default(),
        "b" => match raw_value(cell).as_str() {
            "1" => "true".to_owned(),
            "0" => "false".to_owned(),
            other => other.to_owned(),
        },
        "n" | "" => normalize_number(&raw_value(cell)),
        _ => raw_value(cell),
    }
}

/// Returns the trimmed `<v>` text of a cell (`""` when absent).
fn raw_value(cell: &XmlElement) -> String {
    cell.find_first("v")
        .map(XmlElement::text_content)
        .map(|text| text.trim().to_owned())
        .unwrap_or_default()
}

/// Drops a fractional part that is entirely zeros (`12.0` → `12`).
fn normalize_number(raw: &str) -> String {
    let Some((integer, fraction)) = raw.split_once('.') else {
        return raw.to_owned();
    };
    let signed_integer = integer
        .strip_prefix('+')
        .or_else(|| integer.strip_prefix('-'))
        .unwrap_or(integer);
    let integral = !signed_integer.is_empty() && signed_integer.chars().all(|c| c.is_ascii_digit());
    let zeros = !fraction.is_empty() && fraction.chars().all(|character| character == '0');
    if integral && zeros {
        integer.to_owned()
    } else {
        raw.to_owned()
    }
}

/// Decodes the column part of an `A1` reference (`A1` → 0, `AA3` → 26).
fn column_index_from_reference(reference: &str) -> Option<usize> {
    let mut index = 0usize;
    let mut letters = 0usize;
    for byte in reference.bytes() {
        if !byte.is_ascii_alphabetic() {
            break;
        }
        let value = usize::from(byte.to_ascii_uppercase().wrapping_sub(b'A')) + 1;
        index = index.checked_mul(26)?.checked_add(value)?;
        letters += 1;
        if letters > MAX_COLUMN_LETTERS {
            return None;
        }
    }
    if letters == 0 {
        return None;
    }
    index.checked_sub(1)
}

/// Concatenates the `<t>` descendants of a shared or inline string.
///
/// Rich-text runs (`<r><t>…</t></r>`) are flattened; phonetic runs (`<rPh>`) are
/// Japanese furigana and are skipped so the string stays plain.
fn shared_string_text(element: &XmlElement) -> String {
    let mut text = String::new();
    collect_string_text(element, &mut text);
    text
}

/// Appends the `<t>` text of `element` and its descendants to `text`.
fn collect_string_text(element: &XmlElement, text: &mut String) {
    for child in &element.children {
        match local_name(&child.name) {
            "rPh" | "phoneticPr" => {}
            "t" => text.push_str(&child.text),
            _ => {
                text.push_str(&child.text);
                collect_string_text(child, text);
            }
        }
    }
}

/// Re-stamps `_s0_` in a generated id with the sheet's real index.
///
/// [`TableData`] stamps every id with sheet `0` because it carries no sheet
/// index; the last `_s0_` segment is the sheet marker, so only that one changes.
fn retarget_sheet(mut document: Document, index: usize) -> Document {
    if index != 0 {
        if let Some(position) = document.id.rfind("_s0_") {
            let replacement = format!("_s{index}_");
            document
                .id
                .replace_range(position..position + 4, &replacement);
        }
    }
    document
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::Value;

    use super::*;

    const SHEET: &str = r#"<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>Price</t></is></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>12.0</v></c><c r="C2" t="b"><v>1</v></c></row>
    <row r="4"><c r="C4" t="str"><v>gap</v></c></row>
  </sheetData>
</worksheet>"#;

    #[test]
    fn builds_a_table_from_shared_inline_numeric_and_boolean_cells() {
        let shared = vec!["Name".to_owned(), "alice".to_owned()];
        let table = build_table(SHEET, &shared).unwrap();
        assert_eq!(table.headers, vec!["Name", "Price", ""]);
        assert_eq!(table.rows.len(), 3);
        assert_eq!(table.rows[0], vec!["alice", "12", "true"]);
        assert_eq!(table.rows[1], vec!["", "", ""]);
        assert_eq!(table.rows[2], vec!["", "", "gap"]);
        assert_eq!(table.row_text(0).unwrap(), "Name: alice, Price: 12");
        assert_eq!(
            table.column_text(2),
            "Column name: Column 3. Values: true, gap"
        );
    }

    #[test]
    fn column_references_decode_multi_letter_names() {
        assert_eq!(column_index_from_reference("A1"), Some(0));
        assert_eq!(column_index_from_reference("Z9"), Some(25));
        assert_eq!(column_index_from_reference("AA1"), Some(26));
        assert_eq!(column_index_from_reference("AB3"), Some(27));
        assert_eq!(column_index_from_reference("XFD1"), Some(16_383));
        assert_eq!(column_index_from_reference("1"), None);
        assert_eq!(column_index_from_reference(""), None);
    }

    #[test]
    fn numbers_keep_their_literal_text_except_integral_zeros() {
        assert_eq!(normalize_number("12.0"), "12");
        assert_eq!(normalize_number("-3.00"), "-3");
        assert_eq!(normalize_number("+4.0"), "+4");
        assert_eq!(normalize_number("1.50"), "1.50");
        assert_eq!(normalize_number("1.5"), "1.5");
        assert_eq!(normalize_number("7"), "7");
        assert_eq!(normalize_number(""), "");
        assert_eq!(normalize_number("1."), "1.");
    }

    #[test]
    fn shared_strings_flatten_rich_text_and_skip_phonetics() {
        let markup =
            parse("<si><r><t>ab</t></r><r><t>cd</t></r><rPh><t>furigana</t></rPh><t>!</t></si>")
                .unwrap();
        assert_eq!(shared_string_text(markup.root().unwrap()), "abcd!");
    }

    #[test]
    fn relationship_targets_are_normalised() {
        assert_eq!(
            resolve_part("xl", "worksheets/sheet1.xml"),
            "xl/worksheets/sheet1.xml"
        );
        assert_eq!(
            resolve_part("xl", "/xl/worksheets/sheet2.xml"),
            "xl/worksheets/sheet2.xml"
        );
        assert_eq!(
            resolve_part("xl", "./worksheets/./sheet3.xml"),
            "xl/worksheets/sheet3.xml"
        );
        assert_eq!(
            resolve_part("xl", "../xl/worksheets/sheet4.xml"),
            "xl/worksheets/sheet4.xml"
        );
        assert_eq!(resolve_part("xl", "//a//b.xml"), "a/b.xml");
        assert_eq!(normalize_part("a/../../b"), "b");
    }

    #[test]
    fn sheets_are_addressable_by_name_and_index() {
        let workbook = XlsxWorkbook {
            sheets: vec![
                XlsxSheet {
                    name: "First".to_owned(),
                    index: 0,
                    table: TableData::new(vec!["h".to_owned()], vec![vec!["v".to_owned()]]),
                },
                XlsxSheet {
                    name: "Second".to_owned(),
                    index: 1,
                    table: TableData::new(vec!["h".to_owned()], vec![vec!["w".to_owned()]]),
                },
            ],
        };
        assert_eq!(workbook.sheet_names(), vec!["First", "Second"]);
        assert_eq!(workbook.len(), 2);
        assert!(!workbook.is_empty());
        assert_eq!(workbook.sheet("Second").unwrap().index, 1);
        assert_eq!(workbook.sheet(1).unwrap().name, "Second");
        assert!(workbook.sheet(2).is_none());

        let documents = workbook.to_documents("book");
        assert_eq!(
            documents
                .iter()
                .map(|doc| doc.id.as_str())
                .collect::<Vec<_>>(),
            vec!["book_s0_r2", "book_s0_c0", "book_s1_r2", "book_s1_c0"]
        );
        assert_eq!(
            documents[2].metadata.get("sheet_name"),
            Some(&Value::String("Second".to_owned()))
        );
    }

    #[test]
    fn sheet_ids_survive_a_base_id_that_contains_the_marker() {
        let sheet = XlsxSheet {
            name: "S".to_owned(),
            index: 3,
            table: TableData::new(vec!["h".to_owned()], vec![vec!["v".to_owned()]]),
        };
        let documents = sheet.to_documents("raw_s0_input");
        assert_eq!(documents[0].id, "raw_s0_input_s3_r2");
        assert_eq!(documents[1].id, "raw_s0_input_s3_c0");
    }

    #[test]
    fn missing_workbook_part_is_reported_as_retrieval() {
        let empty = vec![
            0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let archive = ZipArchive::from_bytes(empty).unwrap();
        let error = XlsxParser::new().parse_archive(&archive).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(error.to_string().contains(WORKBOOK_PART), "{error}");
    }
}
