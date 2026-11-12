//! Tabular data: Markdown rendering, row/column documents, and table chunks.
//!
//! Port of the row/column document scheme of the Python `ExcelParser`
//! (`indexing/processor/parser/excel_parser.py`) without the `openpyxl`
//! dependency: callers bring a [`TableData`] value (from `CSV`, a future `XLSX`
//! reader, or a Markdown table) and this module converts it into the same row
//! documents, column documents, and text chunks the reference implementation
//! emits. Python converts `DOCX` tables to Markdown text with its own renderer;
//! [`TableData::to_markdown`] is the Rust equivalent.
//!
//! Naming parity: an empty or missing header cell is named `Column {n}` with
//! 1-based `n`, as Python does for column documents; row documents skip cells
//! whose header is empty (also Python's behavior). A fully header-less table
//! synthesizes `Column {n}` names for its row documents.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::common::{Document, Metadata, TextChunk};

/// The sheet index stamped on every generated id; a [`TableData`] is one sheet.
const SHEET_INDEX: usize = 0;

/// Header row plus data rows.
///
/// Cell text is used verbatim; blank cells are preserved so that row and column
/// documents keep their column alignment.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TableData {
    /// Column names, in table order.
    pub headers: Vec<String>,
    /// Data rows, each aligned with `headers` (short rows are tolerated).
    pub rows: Vec<Vec<String>>,
}

impl TableData {
    /// Creates a table from its header row and data rows.
    #[must_use]
    pub fn new(headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        Self { headers, rows }
    }

    /// Renders the table as a Markdown table.
    ///
    /// Pipes are escaped as `\|`, backslashes as `\\`, and line breaks as `<br>`,
    /// so [`TableData::from_markdown`] restores the original cells. A table with
    /// no headers renders only its rows (and cannot be parsed back with headers).
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut lines = Vec::new();
        if !self.headers.is_empty() {
            lines.push(render_row(&self.headers));
            let separators = vec!["---"; self.headers.len()].join(" | ");
            lines.push(format!("| {separators} |"));
        }
        for row in &self.rows {
            lines.push(render_row(row));
        }
        lines.join("\n")
    }

    /// Parses a Markdown table back into headers and rows.
    ///
    /// Tolerates a missing `| --- |` separator row, surrounding whitespace, blank
    /// lines, and rows without leading/trailing pipes. The first non-separator
    /// line becomes the header row.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the input contains no table row at all.
    pub fn from_markdown(markdown: &str) -> Result<Self> {
        let mut rows: Vec<Vec<String>> = Vec::new();
        for line in markdown.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let cells = parse_row(line);
            if cells.iter().all(|cell| is_separator_cell(cell)) {
                continue;
            }
            rows.push(cells);
        }
        let mut rows = rows.into_iter();
        let Some(headers) = rows.next() else {
            return Err(Error::Validation(
                "table markdown contains no rows".to_owned(),
            ));
        };
        Ok(Self {
            headers,
            rows: rows.collect(),
        })
    }

    /// Returns the values of `index` for every row that has that cell.
    #[must_use]
    pub fn column(&self, index: usize) -> Vec<&str> {
        self.rows
            .iter()
            .filter_map(|row| row.get(index))
            .map(String::as_str)
            .collect()
    }

    /// Returns the header name of `index`, falling back to `Column {index+1}`.
    #[must_use]
    pub fn column_name(&self, index: usize) -> String {
        self.headers
            .get(index)
            .map(|header| header.trim())
            .filter(|header| !header.is_empty())
            .map_or_else(|| format!("Column {}", index + 1), str::to_owned)
    }

    /// Renders the column document text for `index`.
    ///
    /// Python parity: `"Column name: X. Values: a, b"`, with `(empty)` when every
    /// value is blank.
    #[must_use]
    pub fn column_text(&self, index: usize) -> String {
        let name = self.column_name(index);
        let values: Vec<&str> = self
            .column(index)
            .into_iter()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .collect();
        if values.is_empty() {
            format!("Column name: {name}. Values: (empty)")
        } else {
            format!("Column name: {name}. Values: {}", values.join(", "))
        }
    }

    /// Renders the row document text for `index`.
    ///
    /// Python parity: `"{header}: {value}, …"` for every cell with a non-empty
    /// header, blank values included; `None` when the row has no such cell.
    #[must_use]
    pub fn row_text(&self, index: usize) -> Option<String> {
        let row = self.rows.get(index)?;
        let mut parts: Vec<String> = Vec::new();
        for (column, cell) in row.iter().enumerate() {
            let name = if self.headers.is_empty() {
                format!("Column {}", column + 1)
            } else {
                match self.headers.get(column) {
                    Some(header) if !header.trim().is_empty() => header.trim().to_owned(),
                    Some(_) => continue,
                    // Python zips headers with the row, dropping extra cells.
                    None => break,
                }
            };
            parts.push(format!("{name}: {}", cell.trim()));
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }

    /// Renders the row document text of every row that has one.
    #[must_use]
    pub fn row_texts(&self) -> Vec<String> {
        (0..self.rows.len())
            .filter_map(|index| self.row_text(index))
            .collect()
    }

    /// Builds row documents followed by column documents.
    ///
    /// Row ids are `{base_id}_s0_r{row}` with 1-based numbering that includes the
    /// header row (the first data row is 2), matching the `CSV` loader and Python
    /// `ExcelParser`. Column ids are `{base_id}_s0_c{column}`, 0-based.
    #[must_use]
    pub fn to_documents(&self, base_id: &str, sheet_name: Option<&str>) -> Vec<Document> {
        let mut documents = self.row_documents(base_id, sheet_name);
        documents.extend(self.to_column_documents(base_id, sheet_name));
        documents
    }

    /// Builds only the column documents (`source_type: "column"`).
    #[must_use]
    pub fn to_column_documents(&self, base_id: &str, sheet_name: Option<&str>) -> Vec<Document> {
        (0..self.column_count())
            .map(|column| {
                let mut metadata = sheet_metadata("column", sheet_name);
                metadata.insert(
                    "column_name".to_owned(),
                    Value::String(self.column_name(column)),
                );
                Document {
                    id: format!("{base_id}_s{SHEET_INDEX}_c{column}"),
                    text: self.column_text(column),
                    metadata,
                }
            })
            .collect()
    }

    /// Builds one table chunk plus one chunk per row.
    ///
    /// The first chunk holds the whole Markdown table (`source_type: "table"`);
    /// every following chunk holds a row text (`source_type: "row"` with its
    /// `row_index`). All chunks carry `modality: "table"` and a `sheet_name`
    /// entry, which is `null` because [`TableData`] carries no sheet name.
    #[must_use]
    pub fn to_chunks(&self, chunk_prefix: &str) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let markdown = self.to_markdown();
        if !markdown.is_empty() {
            chunks.push(TextChunk {
                id: format!("{chunk_prefix}_table"),
                text: markdown,
                doc_id: chunk_prefix.to_owned(),
                metadata: chunk_metadata("table", None),
                embedding: None,
            });
        }
        for index in 0..self.rows.len() {
            let Some(text) = self.row_text(index) else {
                continue;
            };
            let row_index = index + 2;
            chunks.push(TextChunk {
                id: format!("{chunk_prefix}_r{row_index}"),
                text,
                doc_id: chunk_prefix.to_owned(),
                metadata: chunk_metadata("row", Some(row_index)),
                embedding: None,
            });
        }
        chunks
    }

    /// Builds only the row documents (`source_type: "row"`).
    fn row_documents(&self, base_id: &str, sheet_name: Option<&str>) -> Vec<Document> {
        (0..self.rows.len())
            .filter_map(|index| {
                self.row_text(index).map(|text| Document {
                    id: format!("{base_id}_s{SHEET_INDEX}_r{}", index + 2),
                    text,
                    metadata: {
                        let mut metadata = sheet_metadata("row", sheet_name);
                        metadata.insert("row_index".to_owned(), Value::from(index + 2));
                        metadata
                    },
                })
            })
            .collect()
    }

    /// The number of column documents: the header width, or the widest row when
    /// the table has no header row (Python substitutes `Column {n}` names).
    fn column_count(&self) -> usize {
        if self.headers.is_empty() {
            self.rows.iter().map(Vec::len).max().unwrap_or(0)
        } else {
            self.headers.len()
        }
    }
}

/// Renders one Markdown row, including its leading and trailing pipes.
fn render_row(cells: &[String]) -> String {
    let mut line = String::from("|");
    for cell in cells {
        line.push(' ');
        line.push_str(&escape_cell(cell));
        line.push_str(" |");
    }
    line
}

/// Escapes a cell so [`TableData::from_markdown`] can restore it.
fn escape_cell(cell: &str) -> String {
    let normalized = cell.replace("\r\n", "\n").replace('\r', "\n");
    let mut escaped = String::with_capacity(normalized.len());
    for character in normalized.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '|' => escaped.push_str("\\|"),
            '\n' => escaped.push_str("<br>"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Restores the cell text produced by [`escape_cell`].
fn unescape_cell(cell: &str) -> String {
    cell.replace("<br>", "\n")
}

/// Splits one Markdown row into (still escaped, untrimmed) cells.
fn parse_row(line: &str) -> Vec<String> {
    let inner = line.strip_prefix('|').unwrap_or(line);
    let inner = strip_trailing_pipe(inner);
    split_cells(inner)
        .into_iter()
        .map(|cell| unescape_cell(cell.trim()))
        .collect()
}

/// Drops a trailing pipe unless it is escaped (`\|`).
fn strip_trailing_pipe(text: &str) -> &str {
    let Some(stripped) = text.strip_suffix('|') else {
        return text;
    };
    let backslashes = stripped.chars().rev().take_while(|c| *c == '\\').count();
    if backslashes % 2 == 1 {
        text
    } else {
        stripped
    }
}

/// Splits on unescaped pipes, honoring the `\|` and `\\` escapes.
fn split_cells(text: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '\\' if matches!(characters.peek(), Some('|' | '\\')) => {
                if let Some(escaped) = characters.next() {
                    current.push(escaped);
                }
            }
            '|' => cells.push(std::mem::take(&mut current)),
            other => current.push(other),
        }
    }
    cells.push(current);
    cells
}

/// Returns `true` for a `---`/`:---:` style separator cell.
fn is_separator_cell(cell: &str) -> bool {
    let trimmed = cell.trim().trim_matches(':');
    !trimmed.is_empty() && trimmed.chars().all(|character| character == '-')
}

/// Builds the `{sheet_name, source_type}` metadata shared by row/column documents.
fn sheet_metadata(source_type: &str, sheet_name: Option<&str>) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert(
        "sheet_name".to_owned(),
        sheet_name.map_or(Value::Null, |name| Value::String(name.to_owned())),
    );
    metadata.insert(
        "source_type".to_owned(),
        Value::String(source_type.to_owned()),
    );
    metadata
}

/// Builds the chunk metadata for a table or row chunk.
fn chunk_metadata(source_type: &str, row_index: Option<usize>) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert("modality".to_owned(), Value::String("table".to_owned()));
    metadata.insert(
        "source_type".to_owned(),
        Value::String(source_type.to_owned()),
    );
    if let Some(row_index) = row_index {
        metadata.insert("row_index".to_owned(), Value::from(row_index));
    }
    metadata.insert("sheet_name".to_owned(), Value::Null);
    metadata
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// The sample table used by most tests.
    fn sample() -> TableData {
        TableData::new(
            vec!["name".to_owned(), "city".to_owned()],
            vec![
                vec!["alice".to_owned(), "paris".to_owned()],
                vec!["bob".to_owned(), String::new()],
            ],
        )
    }

    #[test]
    fn markdown_rendering_uses_a_header_and_separator() {
        assert_eq!(
            sample().to_markdown(),
            "| name | city |\n| --- | --- |\n| alice | paris |\n| bob |  |"
        );
        assert_eq!(TableData::default().to_markdown(), "");
        assert_eq!(
            TableData::new(Vec::new(), vec![vec!["x".to_owned()]]).to_markdown(),
            "| x |"
        );
    }

    #[test]
    fn markdown_round_trips_escaped_pipes_and_newlines() {
        let table = TableData::new(
            vec!["a|b".to_owned(), "line".to_owned()],
            vec![
                vec!["x|y".to_owned(), "one\ntwo".to_owned()],
                vec!["back\\slash".to_owned(), String::new()],
            ],
        );
        let markdown = table.to_markdown();
        assert!(markdown.contains(r"a\|b"));
        assert!(markdown.contains("one<br>two"));
        assert_eq!(TableData::from_markdown(&markdown).unwrap(), table);

        // A trailing escaped pipe must not be mistaken for the row delimiter.
        let trailing = TableData::new(vec!["h".to_owned()], vec![vec!["v|".to_owned()]]);
        assert_eq!(
            TableData::from_markdown(&trailing.to_markdown()).unwrap(),
            trailing
        );
    }

    #[test]
    fn markdown_parsing_tolerates_loose_input() {
        let table = TableData::from_markdown("a | b\n1 | 2\n\n3 | 4\n").unwrap();
        assert_eq!(table.headers, vec!["a", "b"]);
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.rows[1], vec!["3", "4"]);

        let with_alignment = TableData::from_markdown("| a | b |\n|:---|---:|\n| 1 | 2 |").unwrap();
        assert_eq!(with_alignment.headers, vec!["a", "b"]);
        assert_eq!(with_alignment.rows, vec![vec!["1", "2"]]);

        let error = TableData::from_markdown("   ").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("contains no rows"));
    }

    #[test]
    fn row_and_column_texts_match_the_excel_parser() {
        let table = sample();
        assert_eq!(
            table.row_texts(),
            vec!["name: alice, city: paris", "name: bob, city: "]
        );
        assert_eq!(table.column(0), vec!["alice", "bob"]);
        assert_eq!(table.column_name(1), "city");
        assert_eq!(
            table.column_text(0),
            "Column name: name. Values: alice, bob"
        );
        assert_eq!(table.column_text(1), "Column name: city. Values: paris");

        // Empty headers are named positionally; blank values yield `(empty)`.
        let positional = TableData::new(
            vec![String::new(), "b".to_owned()],
            vec![vec!["1".to_owned(), String::new()]],
        );
        assert_eq!(positional.column_name(0), "Column 1");
        assert_eq!(positional.column_text(1), "Column name: b. Values: (empty)");
        // Row documents skip cells without a header, like Python.
        assert_eq!(positional.row_text(0).unwrap(), "b: ");

        // A header-less table synthesizes positional names for row documents.
        let headerless = TableData::new(Vec::new(), vec![vec!["1".to_owned(), "2".to_owned()]]);
        assert_eq!(headerless.row_text(0).unwrap(), "Column 1: 1, Column 2: 2");
        assert_eq!(
            headerless.column_text(1),
            "Column name: Column 2. Values: 2"
        );

        // Rows longer than the header drop the extra cells.
        let ragged = TableData::new(
            vec!["h".to_owned()],
            vec![vec!["v".to_owned(), "x".to_owned()]],
        );
        assert_eq!(ragged.row_text(0).unwrap(), "h: v");
        assert_eq!(
            TableData::new(vec!["h".to_owned()], Vec::new()).row_text(3),
            None
        );
    }

    #[test]
    fn documents_follow_the_csv_id_and_metadata_scheme() {
        let documents = sample().to_documents("base", Some("Sheet1"));
        assert_eq!(documents.len(), 4);
        assert_eq!(documents[0].id, "base_s0_r2");
        assert_eq!(documents[1].id, "base_s0_r3");
        assert_eq!(documents[2].id, "base_s0_c0");
        assert_eq!(documents[3].id, "base_s0_c1");
        assert_eq!(documents[0].text, "name: alice, city: paris");
        assert_eq!(
            documents[0].metadata.get("source_type"),
            Some(&Value::String("row".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("row_index"),
            Some(&Value::from(2_u64))
        );
        assert_eq!(
            documents[0].metadata.get("sheet_name"),
            Some(&Value::String("Sheet1".to_owned()))
        );
        assert_eq!(
            documents[2].metadata.get("source_type"),
            Some(&Value::String("column".to_owned()))
        );
        assert_eq!(
            documents[2].metadata.get("column_name"),
            Some(&Value::String("name".to_owned()))
        );
        assert!(documents[2].metadata.get("row_index").is_none());

        // An unnamed sheet keeps the key with a JSON null.
        let unnamed = sample().to_documents("base", None);
        assert_eq!(unnamed[0].metadata.get("sheet_name"), Some(&Value::Null));
    }

    #[test]
    fn column_documents_can_be_built_alone() {
        let documents = sample().to_column_documents("base", Some("Sheet1"));
        assert_eq!(documents.len(), 2);
        assert_eq!(documents[1].id, "base_s0_c1");
        assert_eq!(documents[1].text, "Column name: city. Values: paris");
        assert_eq!(sample().to_column_documents("base", None).len(), 2);
    }

    #[test]
    fn chunks_carry_the_table_metadata() {
        let chunks = sample().to_chunks("kb/demo.csv");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].id, "kb/demo.csv_table");
        assert_eq!(chunks[0].doc_id, "kb/demo.csv");
        assert_eq!(chunks[0].text, sample().to_markdown());
        assert_eq!(
            chunks[0].metadata.get("modality"),
            Some(&Value::String("table".to_owned()))
        );
        assert_eq!(
            chunks[0].metadata.get("source_type"),
            Some(&Value::String("table".to_owned()))
        );
        assert_eq!(chunks[0].metadata.get("sheet_name"), Some(&Value::Null));
        assert!(chunks[0].metadata.get("row_index").is_none());

        assert_eq!(chunks[1].id, "kb/demo.csv_r2");
        assert_eq!(chunks[1].text, "name: alice, city: paris");
        assert_eq!(chunks[1].doc_id, "kb/demo.csv");
        assert_eq!(
            chunks[1].metadata.get("source_type"),
            Some(&Value::String("row".to_owned()))
        );
        assert_eq!(
            chunks[1].metadata.get("row_index"),
            Some(&Value::from(2_u64))
        );
        assert_eq!(
            chunks[1].metadata.get("modality"),
            Some(&Value::String("table".to_owned()))
        );
        assert!(chunks.iter().all(|chunk| chunk.embedding.is_none()));

        assert_eq!(TableData::default().to_chunks("empty").len(), 0);
    }
}
