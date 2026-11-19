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
//! The shared-string table may legitimately be absent: a workbook that uses
//! inline strings only needs no `xl/sharedStrings.xml` part, and such a sheet
//! parses fine. A part that does exist must be readable, parsable, and rooted,
//! and every `t="s"` cell must resolve to an entry in it — a missing or
//! non-numeric index, or an index past the end of the table, is a typed
//! [`Error::Retrieval`] naming the worksheet part and the cell rather than an
//! empty string, so a broken shared-string table can never pass as a workbook
//! whose text happens to be blank. The `<sst>` `count`/`uniqueCount`
//! declarations are informational only and are never checked against the real
//! `<si>` count.
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

/// Largest zero-based column a spreadsheet can address (`XFD`, the 16 384th).
const MAX_COLUMN_INDEX: usize = 16_383;
/// Largest one-based row number a spreadsheet can address (1 048 576).
const MAX_ROW_NUMBER: usize = 1_048_576;

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
    /// part). Malformed worksheet or workbook XML is [`Error::Validation`]. A
    /// shared-string part that exists but cannot be read, parsed, or rooted is
    /// [`Error::Retrieval`], as is a `t="s"` cell whose index does not resolve to
    /// an entry in the table. A workbook with no sheets is not an error.
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
            let table = build_table(&markup, &target, &shared_strings)?;
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

/// The workbook's shared-string table plus whether the part exists at all.
///
/// The two states are genuinely different: a workbook that stores every string
/// inline has no `xl/sharedStrings.xml` part (an absent table, no error), while
/// a present but empty table means a `t="s"` cell can never resolve. `build_table`
/// uses `present` to word the error message.
#[derive(Debug, Default, Clone)]
struct SharedStrings {
    /// One decoded string per `<si>` entry, in declaration order.
    values: Vec<String>,
    /// `true` when `xl/sharedStrings.xml` exists in the archive.
    present: bool,
}

impl SharedStrings {
    /// The table of a workbook without a shared-string part.
    fn absent() -> Self {
        Self::default()
    }

    /// A present table holding `values`.
    fn new(values: Vec<String>) -> Self {
        Self {
            values,
            present: true,
        }
    }

    /// Number of entries in the table (`0` when the part is absent).
    fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns the entry at `index`, or `None` when the index is out of range.
    fn get(&self, index: usize) -> Option<&str> {
        self.values.get(index).map(String::as_str)
    }
}

/// Re-kinds a shared-string failure as [`Error::Retrieval`] naming the part.
///
/// [`parse`] reports malformed XML as [`Error::Validation`]; the workbooks that
/// depend on the table cannot be decoded either way, so the shared-string layer
/// reports one kind and keeps the original message.
fn shared_strings_failure(error: &Error, reason: &str) -> Error {
    Error::Retrieval(format!("shared strings {reason}: {}", error.message()))
        .with_context(SHARED_STRINGS_PART)
}

/// Reads the shared-string table.
///
/// An absent `xl/sharedStrings.xml` is legitimate (inline-string-only workbooks)
/// and yields an absent table. A part that exists but cannot be read, parsed, or
/// rooted is an [`Error::Retrieval`] naming the part, because every `t="s"` cell
/// depends on it; see [`SharedStrings`].
fn read_shared_strings(archive: &ZipArchive) -> Result<SharedStrings> {
    if !archive.has_entry(SHARED_STRINGS_PART) {
        return Ok(SharedStrings::absent());
    }
    let markup = archive
        .read_text(SHARED_STRINGS_PART)
        .map_err(|error| shared_strings_failure(&error, "cannot be read"))?;
    let document =
        parse(&markup).map_err(|error| shared_strings_failure(&error, "cannot be parsed"))?;
    let Some(root) = document.root() else {
        return Err(Error::Retrieval("the part has no root element".to_owned())
            .with_context(SHARED_STRINGS_PART));
    };
    Ok(SharedStrings::new(
        root.find_all("si")
            .iter()
            .map(|entry| shared_string_text(entry))
            .collect(),
    ))
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
///
/// `part` is the worksheet's archive path, used for error messages; the table's
/// `t="s"` cells are resolved against `shared_strings`.
fn build_table(markup: &str, part: &str, shared_strings: &SharedStrings) -> Result<TableData> {
    let document = parse(markup).map_err(|error| error.with_context(part))?;
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
        if row_number > MAX_ROW_NUMBER {
            return Err(Error::Validation(format!(
                "worksheet {part} row {row_number} is past the spreadsheet row limit {MAX_ROW_NUMBER}"
            )));
        }
        next_row = row_number.checked_add(1).ok_or_else(|| {
            Error::Validation(format!(
                "worksheet {part} row {row_number} cannot advance the row counter"
            ))
        })?;
        let mut cells: Vec<String> = Vec::new();
        let mut next_column = 0usize;
        for cell in row.children_named("c") {
            let column = resolve_column(cell, next_column, part, row_number)?;
            next_column = column.checked_add(1).ok_or_else(|| {
                Error::Validation(format!(
                    "worksheet {part} cell {} cannot advance the column counter",
                    cell_reference(cell, column, row_number)
                ))
            })?;
            if cells.len() <= column {
                cells.resize(column + 1, String::new());
            }
            cells[column] = cell_value(cell, column, part, row_number, shared_strings)?;
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
///
/// `column` is the already-decoded zero-based column and `part` the worksheet's
/// archive path; both only feed error messages. A `t="s"` cell whose index is
/// missing, non-numeric, or outside [`SharedStrings`] is an
/// [`Error::Retrieval`] rather than an empty string, so an unreadable
/// shared-string table can never masquerade as a blank sheet. Every other cell
/// kind is decoded exactly as before and never fails.
fn cell_value(
    cell: &XmlElement,
    column: usize,
    part: &str,
    row_number: usize,
    shared_strings: &SharedStrings,
) -> Result<String> {
    let kind = cell.attr("t").unwrap_or("n").trim();
    Ok(match kind {
        "s" => {
            let reference = cell_reference(cell, column, row_number);
            let index = raw_value(cell).parse::<usize>().map_err(|_| {
                Error::Retrieval(format!(
                    "worksheet {part} cell {reference} has t=\"s\" but no numeric shared-string index"
                ))
            })?;
            let Some(value) = shared_strings.get(index) else {
                let reason = if shared_strings.present {
                    format!(
                        "the shared-string table has {} entries",
                        shared_strings.len()
                    )
                } else {
                    format!(
                        "{SHARED_STRINGS_PART} is absent (the shared-string table has 0 entries)"
                    )
                };
                return Err(Error::Retrieval(format!(
                    "worksheet {part} cell {reference} references shared string {index} but {reason}"
                )));
            };
            value.to_owned()
        }
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
    })
}

/// Returns the `A1`-style reference of a cell for error messages.
///
/// The `r` attribute wins when present; a cell without one is addressed from its
/// decoded column and row (`column` `1` of row `2` is `B2`), so the message still
/// points at the offending cell.
fn cell_reference(cell: &XmlElement, column: usize, row_number: usize) -> String {
    match cell.attr("r").map(str::trim) {
        Some(reference) if !reference.is_empty() => reference.to_owned(),
        _ => format!("{}{row_number}", column_reference(column)),
    }
}

/// Encodes a zero-based column index as `A1`-style letters (`0` → `A`, `26` → `AA`).
fn column_reference(index: usize) -> String {
    let mut letters = String::new();
    let mut value = index.saturating_add(1);
    while value > 0 {
        let remainder = (value - 1) % 26;
        letters.push(char::from(b'A' + u8::try_from(remainder).unwrap_or(0)));
        value = (value - 1) / 26;
    }
    letters.chars().rev().collect()
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

/// Resolves a cell's zero-based column from its `r` reference.
///
/// A cell without a usable `r` attribute falls back to `fallback` (the column
/// after the previous cell); a named column must stay inside the spreadsheet's
/// last column ([`MAX_COLUMN_INDEX`]), so `AAAAAAAA1`/`XFE1` are a
/// [`Error::Validation`] rather than an allocation-sized column index.
fn resolve_column(
    cell: &XmlElement,
    fallback: usize,
    part: &str,
    row_number: usize,
) -> Result<usize> {
    let reference = cell
        .attr("r")
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let column = match reference {
        Some(reference) => column_index_from_reference(reference).ok_or_else(|| {
            Error::Validation(format!(
                "worksheet {part} cell {reference} does not carry a usable column reference"
            ))
        })?,
        None => fallback,
    };
    if column > MAX_COLUMN_INDEX {
        return Err(Error::Validation(format!(
            "worksheet {part} cell {} addresses column {column}, past the spreadsheet column limit {} (XFD)",
            cell_reference(cell, column, row_number),
            MAX_COLUMN_INDEX
        )));
    }
    Ok(column)
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
    use crate::parsing::ZipWriter;

    const SHEET: &str = r#"<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>Price</t></is></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>12.0</v></c><c r="C2" t="b"><v>1</v></c></row>
    <row r="4"><c r="C4" t="str"><v>gap</v></c></row>
  </sheetData>
</worksheet>"#;

    const SHEET_PART: &str = "xl/worksheets/sheet1.xml";

    #[test]
    fn builds_a_table_from_shared_inline_numeric_and_boolean_cells() {
        let shared = SharedStrings::new(vec!["Name".to_owned(), "alice".to_owned()]);
        let table = build_table(SHEET, SHEET_PART, &shared).unwrap();
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
    fn column_references_encode_a1_letters() {
        assert_eq!(column_reference(0), "A");
        assert_eq!(column_reference(25), "Z");
        assert_eq!(column_reference(26), "AA");
        assert_eq!(column_reference(27), "AB");
        assert_eq!(column_reference(701), "ZZ");
        assert_eq!(column_reference(702), "AAA");
        assert_eq!(column_reference(16_383), "XFD");
    }

    /// Builds a table from `markup`, expecting a typed error.
    fn table_error(markup: &str, shared_strings: &SharedStrings) -> Error {
        build_table(markup, SHEET_PART, shared_strings).unwrap_err()
    }

    /// Asserts a [`Error::Retrieval`] whose message contains every fragment.
    fn assert_retrieval(error: &Error, fragments: &[&str]) {
        assert_eq!(error.kind(), ErrorKind::Retrieval, "{error}");
        let message = error.to_string();
        for fragment in fragments {
            assert!(
                message.contains(fragment),
                "{fragment:?} missing from {message}"
            );
        }
    }

    #[test]
    fn a_shared_string_reference_without_a_table_is_an_error() {
        let markup = r#"<worksheet><sheetData>
  <row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c></row>
  <row r="2"><c r="B2" t="s"><v>3</v></c></row>
</sheetData></worksheet>"#;
        let error = table_error(markup, &SharedStrings::absent());
        assert_retrieval(&error, &["B2", "shared string 3", "0 entries", SHEET_PART]);
    }

    #[test]
    fn a_shared_string_index_past_the_end_of_the_table_is_an_error() {
        let markup = r#"<worksheet><sheetData>
  <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
  <row r="2"><c r="B2" t="s"><v>5</v></c></row>
</sheetData></worksheet>"#;
        let shared = SharedStrings::new(vec!["Name".to_owned(), "alice".to_owned()]);
        let error = table_error(markup, &shared);
        assert_retrieval(&error, &["B2", "shared string 5", "2 entries"]);
    }

    #[test]
    fn an_inline_string_sheet_needs_no_shared_string_table() {
        let markup = r#"<worksheet><sheetData>
  <row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c><c r="B1" t="inlineStr"><is><t>Price</t></is></c></row>
  <row r="2"><c r="A2" t="inlineStr"><is><t>alice</t></is></c><c r="B2"><v>12.0</v></c></row>
</sheetData></worksheet>"#;
        let table = build_table(markup, SHEET_PART, &SharedStrings::absent()).unwrap();
        assert_eq!(table.headers, vec!["Name", "Price"]);
        assert_eq!(table.rows, vec![vec!["alice".to_owned(), "12".to_owned()]]);
    }

    #[test]
    fn a_cell_without_a_reference_still_names_its_column_in_the_error() {
        // The second cell has no `r` attribute, so its reference comes from the
        // decoded column (B) and the row number.
        let markup = r#"<worksheet><sheetData>
  <row r="2"><c t="s"><v>0</v></c><c t="s"><v>7</v></c></row>
</sheetData></worksheet>"#;
        let shared = SharedStrings::new(vec!["only".to_owned()]);
        assert_retrieval(&table_error(markup, &shared), &["cell B2"]);
    }

    #[test]
    fn a_shared_string_cell_without_a_numeric_index_is_an_error() {
        let shared = SharedStrings::new(vec!["first".to_owned(), "second".to_owned()]);
        let non_numeric = r#"<worksheet><sheetData>
  <row r="3"><c r="C3" t="s"><v>first</v></c></row>
</sheetData></worksheet>"#;
        let missing = r#"<worksheet><sheetData>
  <row r="4"><c r="D4" t="s"><f>1+1</f></c></row>
</sheetData></worksheet>"#;
        let expected = "has t=\"s\" but no numeric shared-string index";
        assert_retrieval(&table_error(non_numeric, &shared), &["cell C3", expected]);
        assert_retrieval(&table_error(missing, &shared), &["cell D4", expected]);
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

    /// Builds an archive from stored `(name, contents)` parts.
    fn archive(parts: &[(&str, &str)]) -> ZipArchive {
        let mut writer = ZipWriter::new();
        for (name, contents) in parts {
            writer.add_stored(*name, contents.as_bytes()).unwrap();
        }
        ZipArchive::from_bytes(writer.finish().unwrap()).unwrap()
    }

    #[test]
    fn missing_workbook_part_is_reported_as_retrieval() {
        let empty = vec![
            0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let archive = ZipArchive::from_bytes(empty).unwrap();
        let error = XlsxParser::new().parse_archive(&archive).unwrap_err();
        assert_retrieval(&error, &[WORKBOOK_PART]);
    }

    #[test]
    fn an_absent_shared_strings_part_is_not_an_error() {
        let archive = archive(&[(WORKBOOK_PART, "<workbook/>")]);
        let shared = read_shared_strings(&archive).unwrap();
        assert!(!shared.present);
        assert_eq!(shared.len(), 0);
    }

    #[test]
    fn an_unreadable_shared_strings_part_is_retrieval() {
        for contents in ["<sst><si><t>broken", "<!-- only a comment -->"] {
            let archive = archive(&[(SHARED_STRINGS_PART, contents)]);
            let error = read_shared_strings(&archive).unwrap_err();
            assert_retrieval(&error, &[SHARED_STRINGS_PART]);
        }
    }

    #[test]
    fn shared_string_declarations_larger_than_the_table_are_tolerated() {
        let archive = archive(&[(
            SHARED_STRINGS_PART,
            r#"<sst count="99" uniqueCount="99"><si><t>Name</t></si><si><t>alice</t></si></sst>"#,
        )]);
        let shared = read_shared_strings(&archive).unwrap();
        assert!(shared.present);
        assert_eq!(shared.len(), 2);
        let markup = r#"<worksheet><sheetData>
  <row r="1"><c r="A1" t="s"><v>0</v></c></row>
  <row r="2"><c r="A2" t="s"><v>1</v></c></row>
</sheetData></worksheet>"#;
        let table = build_table(markup, SHEET_PART, &shared).unwrap();
        assert_eq!(table.headers, vec!["Name"]);
        assert_eq!(table.rows, vec![vec!["alice".to_owned()]]);
    }

    #[test]
    fn an_unresolvable_shared_string_fails_the_parser() {
        let archive = archive(&[
            (
                WORKBOOK_PART,
                r#"<workbook><sheets><sheet name="Data"/></sheets></workbook>"#,
            ),
            (
                SHEET_PART,
                r#"<worksheet><sheetData>
  <row r="1"><c r="A1" t="s"><v>0</v></c></row>
</sheetData></worksheet>"#,
            ),
        ]);
        let error = XlsxParser::new().parse_archive(&archive).unwrap_err();
        assert_retrieval(&error, &[SHEET_PART, "cell A1", "shared string 0"]);
    }
}
