# src/parsing — containers, XML, and document parsers

## Intent

Read binary documents without adding a compression, office-format, or PDF dependency.
`DOCX`, `XLSX`, `PPTX`, and `ODT` are ZIP archives of XML parts and PDF content streams are often `zlib`-wrapped `DEFLATE`, so the module owns that container layer (reader plus a minimal writer), a tolerant XML parser, and the five document parsers built on them.
PDF content streams are run through their `/Filter` chain (`ASCIIHexDecode`, `ASCII85Decode`, `RunLengthDecode`, `FlateDecode`) and `/DecodeParms` predictors before the text operators are read, and a stream that cannot be decoded is recorded on the document rather than dropped.
Everything is offline and deterministic: no network, no OCR, no external binaries.
Parsers return data structures (`DocxDocument`, `XlsxWorkbook`, `PptxDocument`, `OdtDocument`, `PdfDocument`); turning them into `crate::Document`s for indexing is a separate step, and XLSX is the only format that already exposes a `to_documents` conversion (via `TableData`).
`ZipWriter` writes stored entries only — enough for DOCX/XLSX/PPTX/ODT fixtures in tests and tools without a ZIP crate.
`PPTX` and `ODT` are documented Rust additions: the Python reference has no parser for either, and no bytes-input API.
Image text extraction (OCR) is unsupported in every format: images contribute no text, and no parser guesses captions.

## API Surface

### Containers

- `inflate::inflate(data, max_output) -> Result<Vec<u8>>` — raw `DEFLATE` (RFC 1951): stored blocks, fixed Huffman, dynamic Huffman (code-length alphabet with repeat codes 16/17/18), LSB-first bit reader, length/distance base+extra tables, and back-reference copying.
- `inflate::zlib_decompress(data, max_output) -> Result<Vec<u8>>` — validates the `zlib` header (CM=8, CINFO<=7, FCHECK, no FDICT), inflates the payload, verifies the trailing `Adler-32`.
- `inflate::adler32(data) -> u32` and `inflate::crc32(data) -> u32` — table-driven standard checksums (`CRC-32` uses the reflected `0xEDB8_8320` polynomial).
- `inflate::DEFAULT_MAX_OUTPUT_BYTES` — 64 MiB decompression-bomb guard; the ZIP reader and the PDF reader pass it to the decoder.
- `zip::ZipArchive` — `from_bytes(Vec<u8>)`, async `open(path)` (`tokio::fs`, no blocking I/O), `entries`, `entry_names`, `entry`, `has_entry`, `len`, `is_empty`, `archive_size`, `read_entry`, `read_entry_ignore_case`, `read_text`.
- `zip::ZipEntry` — `name`, `compression_method`, `flags`, `crc32`, `compressed_size`, `uncompressed_size`, `local_header_offset`, plus `is_directory`, `is_utf8_name`, `compression_method_name`.
- `zip::compression_method_name(method) -> &'static str` — APPNOTE names used in `Unsupported` messages.
- `zip::COMPRESSION_METHOD_STORED` (0) and `zip::COMPRESSION_METHOD_DEFLATE` (8) — the only supported methods.
- `zip::writer::ZipWriter` — `new()`/`Default`, `add_stored(name, data) -> Result<&mut Self>` (writes a local header plus the payload immediately, so calls chain), `finish(self) -> Result<Vec<u8>>` (appends the central directory and EOCD), `len()`, `is_empty()`; stored entries only, UTF-8 name flag set, timestamps/attributes/comments/extra fields empty, `CRC-32` from `crate::parsing::crc32`, entries kept in insertion order.
- `ZipWriter` limits, all `Error::Validation` naming the value and limit: empty name, name over `65_535` bytes, more than `65_535` entries, payload over `u32::MAX`, archive offset over `u32::MAX`.

### XML (`xml`)

- `xml::parse(&str) -> Result<XmlDocument>` — prolog/`<?xml …?>`, comments, `CDATA`, markup declarations (`<!DOCTYPE …>`), self-closing tags, nested elements, single/double-quoted attributes, entity decoding (`&amp; &lt; &gt; &quot; &apos;`, `&#NN;`, `&#xNN;`; unknown entities kept verbatim), namespace prefixes preserved (`w:p`), whitespace text preserved, a leading BOM skipped.
- `xml::MAX_DEPTH` (256) — maximum element nesting; deeper input is rejected so recursion cannot overflow the stack.
- `xml::XmlDocument` — `prolog: Option<String>`, `roots: Vec<XmlElement>`, `root()` (first top-level element); several roots are tolerated for XML fragments.
- `xml::XmlElement` — `name`, `attributes: Vec<(String, String)>`, `children`, `text` (direct text, `CDATA` included), plus `find_first`, `find_all`, `children_named`, `attr`, `text_content`, `local_name`.
- Name matching: a name without `:` also matches by local name (`find_first("p")` finds `<w:p>`), a prefixed name matches exactly; `find_first`/`find_all` search descendants only.
- `xml::local_name(name) -> &str` — strips a namespace prefix.

### DOCX (`docx`)

- `docx::DocxParser` — `new()`, `parse_bytes(&[u8])`, `parse_archive(&ZipArchive)`, async `parse_path(path)`.
- `docx::DocxDocument` — `text` (rendered blocks joined by `\n`), `blocks`, `title: Option<String>` (`dc:title`), `paragraph_count` (paragraph and heading blocks), `table_count`, `is_empty()`.
- `docx::DocxBlock` — `Paragraph { text }`, `Heading { level, text }` (`level` is the `#` count), `Table { markdown }`; `render()` and `text()`.
- `docx::DOCUMENT_PART` (`word/document.xml`) and `docx::CORE_PROPERTIES_PART` (`docProps/core.xml`).
- Body walk: direct `w:body` children only, with `w:sdt`/`w:sdtContent`/`w:customXml`/`w:smartTag` transparent; `w:p` → paragraph/heading, `w:tbl` → Markdown table, everything else ignored.
- Paragraph text: `w:t` runs, `w:tab`/`w:ptab` → tab, `w:br`/`w:cr` → newline, empty paragraphs dropped.
- Heading mapping from `w:pPr/w:pStyle/@w:val` (case-insensitive, spaces ignored): `Title` → 1, `Subtitle` → 2, `HeadingN` (1–9) → `N + 1`; the reference Python `Title=#, H1=##, H2=###` convention.
- Tables are rendered by `TableData::to_markdown` (pipes escaped, cell paragraphs joined by newlines); the first row is the header.
- Missing `word/document.xml` → `Error::Retrieval` naming the part; a corrupt ZIP is propagated; a missing or broken `docProps/core.xml` only drops the title (logged at `debug`).
- Deviation: the reference appends a blank line after each table, this port joins blocks with a single `\n`.

### XLSX (`xlsx`)

- `xlsx::XlsxParser` — `new()`, `parse_bytes(&[u8])`, `parse_archive(&ZipArchive)`, async `parse_path(path)`.
- `xlsx::XlsxWorkbook` — `sheets: Vec<XlsxSheet>`, `sheet(name_or_index)` (`"Sheet1"` or `0`, via `SheetRef`), `sheet_names()`, `len()`, `is_empty()`, `to_documents(base_id)`.
- `xlsx::XlsxSheet` — `name`, `index`, `table: TableData`, `to_documents(base_id)`.
- `xlsx::SheetRef` — `Name(&str)` / `Index(usize)`, both `From`-convertible for `sheet(…)`.
- Constants: `WORKBOOK_PART` (`xl/workbook.xml`), `WORKBOOK_RELS_PART` (`xl/_rels/workbook.xml.rels`), `SHARED_STRINGS_PART` (`xl/sharedStrings.xml`).
- Part resolution: `r:id` → relationship `Target`, normalised against `xl` (leading `/`, `./`, `../`, duplicate slashes); without a relationships part (or without the id) the conventional `xl/worksheets/sheetN.xml` path is used; a missing part → `Error::Retrieval` naming it.
- Cells: `t="s"` shared-string lookup, `t="inlineStr"` (`<is>`), `t="b"` (`1`/`0` → `true`/`false`), `t="str"`/`t="e"`/`t="d"` literal, otherwise numeric literal text with an all-zero fraction trimmed (`12.0` → `12`); missing/blank cells are padded so rows are rectangular.
- Shared strings are resolved strictly: the table is a private `SharedStrings { values, present }` carrier, because "no `xl/sharedStrings.xml` part" (inline-string-only workbooks — legitimate, no error) and "index past the end of a present table" are different states.
- An unresolvable `t="s"` lookup is `Error::Retrieval` naming the worksheet part, the cell reference and the table size, never an empty cell: `worksheet xl/worksheets/sheet1.xml cell B2 references shared string 3 but xl/sharedStrings.xml is absent (the shared-string table has 0 entries)`, `… references shared string 5 but the shared-string table has 2 entries`, and `… cell C3 has t="s" but no numeric shared-string index` for a missing or non-numeric `<v>`.
- A `sharedStrings.xml` that exists but cannot be read, parsed, or rooted is `Error::Retrieval` with `SHARED_STRINGS_PART` as context (the tolerant XML parser's `Error::Validation` is re-kinded, so the whole shared-string layer reports one kind).
- The cell reference in those messages is the `r` attribute when present and otherwise synthesised from the decoded column index plus the row number (`column_reference`/`cell_reference`).
- The `<sst count="…" uniqueCount="…">` declarations are informational: a declaration that disagrees with the real `<si>` count is tolerated, and only a real lookup miss is an error. Sheets without `t="s"` cells never touch the table.
- Column placement uses `A1`-style references (multi-letter columns `AA`, `AB`, … up to 8 letters); rows are keyed by `r` and gaps are preserved.
- The first non-empty row is the header; documents come from `TableData::to_documents` with the sheet index re-stamped (`{base}_s{sheet}_r{row}`, `{base}_s{sheet}_c{column}`), `sheet_name`/`source_type`/`row_index`/`column_name` metadata included.
- `row_index` counts document rows with the header as row 1, since the table carries no sheet row numbers (matches the reference when the header is the sheet's first row).
- A workbook with no sheets is an empty `sheets` list, not an error; `<rPh>` phonetic runs are skipped.

### PPTX (`pptx`)

- `pptx::PptxParser` — `new()`, `parse_bytes(&[u8])`, `parse_archive(&ZipArchive)`, async `parse_path(path)`.
- `pptx::PptxDocument` — `text`, `slides: Vec<PptxSlide>`, `slide_count`, `title: Option<String>`, `problems: Vec<String>` (`#[serde(default)]`), `is_empty()`, `slide(index)` (1-based; `0` and out-of-range yield `None`).
- `pptx::PptxSlide` — `index` (1-based), `text` (rendered body lines without the `## Slide` heading and without notes), `title: Option<String>`, `notes: Option<String>`, `has_notes: bool`, `missing_part: Option<String>` (`#[serde(default)]`; the unresolved relationship reference for a placeholder slide, `None` for a real one).
- Constants: `PRESENTATION_PART` (`ppt/presentation.xml`), `PRESENTATION_RELS_PART` (`ppt/_rels/presentation.xml.rels`), `SLIDES_DIR` (`ppt/slides/`), `NOTES_SLIDES_DIR` (`ppt/notesSlides/`), `CORE_PROPERTIES_PART` (`docProps/core.xml`, not re-exported at the parsing root because DOCX defines the same path).
- Slide order: `p:sldIdLst`/`p:sldId@r:id` resolved through the presentation relationships into a private ordered slot list (`SlideSlot`; each slot is either a resolved part or an unresolved entry with a reference and a reason). Targets are normalised against `ppt/`, handling a leading `/`, `./`, `../`, and duplicate slashes.
- A slot never disappears: an entry with no `r:id`, an id the relationships part does not map, a target that is not a `ppt/slides/*.xml` part, or a target part absent from the archive keeps its deck position as a placeholder `PptxSlide` (`missing_part` set, `title` `None`, no notes, `text` = `! missing slide part: {reference}`; `<no r:id>` when the attribute is absent). `slide_count` counts placeholders, which is what makes the loss visible.
- `PptxDocument::problems` holds one message per unresolved entry, in deck order and naming the 1-based position, the relationship id, the target when known and the reason (`slide 2: rId7 is not mapped by ppt/_rels/presentation.xml.rels`); each is also logged once at `warn!`.
- The numeric `ppt/slides/slideN.xml` sort (`slide2` precedes `slide10`) is used only when the presentation part declares no slides at all — a missing presentation or relationships part, no `p:sldIdLst`, or an empty `p:sldIdLst`. Entries that exist but resolve to nothing keep their placeholders instead of falling back.
- A malformed presentation or relationships part still propagates `Error::Validation`.
- Text: `<p:sp>` and `<p:grpSp>` shapes are walked in document order; each `<a:p>` becomes one line from its `<a:t>` runs concatenated with no separator (`<a:br>` → newline), trimmed, with empty lines dropped; `<p:graphicFrame>` tables render through `TableData::to_markdown` (first row header, rows padded to the widest, pipes escaped); pictures, charts, and `SmartArt` contribute nothing.
- Slide `title`: the `title`/`ctrTitle` placeholder's first non-empty line, otherwise the slide's first non-empty body line.
- Speaker notes: `ppt/notesSlides/notesSlideN.xml`, matched by the slide part's number (a name convention, not the slide's own relationships); `sldNum` placeholders are skipped, `has_notes` reflects part presence, `notes` is `Some` only when the extracted text is non-empty.
- Document text: `## Slide {n}`, the slide body lines, then `### Notes` plus the notes lines when present; blocks joined by `\n`, empty blocks dropped. A placeholder slide's `! missing slide part: …` line therefore appears under its own `## Slide N` heading.
- Title: `dc:title` from `docProps/core.xml`; a missing or broken part only logs `debug`.
- Errors: non-ZIP bytes → `Error::Validation`; no slide at all → `Error::Validation` (`"PPTX has no slides: …"`); malformed slide, notes, presentation, or relationships XML → `Error::Validation` with the part path as context; unresolved individual slides are never an error (they become placeholders plus `problems` entries); images and OCR are out of scope.

### ODT (`odt`)

- `odt::OdtParser` — `new()`, `parse_bytes(&[u8])`, `parse_archive(&ZipArchive)`, async `parse_path(path)`.
- `odt::OdtDocument` — `text`, `paragraph_count`, `table_count`, `title: Option<String>`, `unsupported_elements: Vec<String>` (`#[serde(default)]`), `is_empty()`, `text()`.
- Constants: `CONTENT_PART` (`content.xml`), `META_PART` (`meta.xml`).
- Body walk: `office:body`/`office:text` children in document order; `text:section`, `text:list`, `text:list-item`, and `text:list-header` are transparent, so lists flatten into consecutive lines.
- `text:h` → `#`×`text:outline-level` heading (parsed `u8`, clamped to 1–9, default 1); `text:p` → paragraph line; both are trimmed, emit nothing when empty, and count toward `paragraph_count`.
- `table:table` → rows from `table:table-row`, cells from `table:table-cell` and `table:covered-table-cell`, cell text = its paragraphs joined by newline, rows padded to the widest, first row header, rendered by `TableData::to_markdown`; empty tables are dropped and `table_count` counts emitted tables.
- Inline content: direct text is appended, `text:s` → `text:c` spaces (default 1, capped at 1000), `text:tab` → tab, `text:line-break` → newline, `text:span`/`text:a`/`text:bookmark-ref` recurse; `text:note`, `draw:*`, and nested tables are skipped, and each is recorded in `unsupported_elements` instead of being lost.
- `OdtDocument::unsupported_elements` — one entry per distinct unsupported/ignored element, rendered as `<prefixed local name> (×<count>)` (e.g. `draw:image (×2)`) and sorted by element name so the list is deterministic; a clean document reports an empty vector.
- Recorded elements: every `draw:*` name in a skipped subtree (`draw:frame` and the `draw:image`/`draw:object` it contains are reported separately), `text:note`, a nested `table:table`, and every block-level element the walker does not render (`text:table-of-content`, `text:index-*`, `text:tracked-changes`, `office:annotation`, unknown names such as `foo:bar`). The transparent containers and the rendered `text:h`/`text:p`/`table:table` are never recorded.
- Each distinct recorded element is logged once at `warn!` with its count.
- Known limitation: `XmlElement` keeps an element's direct text concatenated in one field (no positional interleaving), so direct text renders before child elements inside the same paragraph; interleaved-in-child content (the normal ODF shape) is unaffected.
- Title: `dc:title` from `meta.xml`; a missing or unparsable meta part only logs `debug`.
- Errors: missing `content.xml` → `Error::Validation` (`"ODT archive has no content.xml"`); malformed `content.xml` → `Error::Validation` with the part path; non-ZIP bytes → `Error::Validation`; images and OCR are out of scope.

### PDF (`pdf`)

- `pdf::PdfParser` — `new()`, `parse_bytes(&[u8])`, async `parse_path(path)`.
- `pdf::PdfDocument` — `text` (non-empty pages joined by `\n`), `pages` (one trimmed string per decoded content stream, in file order), `page_count`, `has_text`, `is_empty()`, plus the completeness markers `incomplete`, `skipped_filters`, and `skipped_streams`, all `#[serde(default)]`.
- `PdfDocument::incomplete` — `true` when at least one stream was skipped, so `text` is partial.
- `PdfDocument::skipped_filters` — display names of the filters on skipped streams, deduplicated in first-seen order.
- `PdfDocument::skipped_streams` — one record per skipped stream, `"stream {0-based ordinal}: {reason}"`, where the reason names the image payload filter or the filter whose decode failed.
- `pdf::KERNING_SPACE_THRESHOLD` (-100.0) — negative `TJ` adjustments beyond it insert a space.
- Text operators: `Tj`, `TJ` (strings and kerning numbers), `'`/`"` (newline first), `Td`/`TD`/`T*`/`Tm`/`BT`/`ET` (idempotent newline).
- String decoding: literal strings with `\n \r \t \b \f \( \) \\`, balanced parentheses, line continuations and `\ddd` octal escapes; hex strings `<…>` (half byte completed with `0`); a UTF-16BE BOM switches either form to UTF-16BE, otherwise UTF-8 with a Latin-1 fallback.
- Stream dictionary: `pdf::dictionary` reads the window before the `stream` keyword — `/Filter` (one `/Name` or a `[ … ]` array whose order is the decode order, long and abbreviated forms) and `/DecodeParms`/`/DP` (one `<< … >>` dictionary or an array aligned with the filters, `null` entries and indirect `N 0 R` values falling back to the defaults).
- Filters (`pdf::filters`): `Filter` resolves `ASCIIHexDecode`/`AHx`, `ASCII85Decode`/`A85`, `LZWDecode`/`LZW`, `FlateDecode`/`Fl`, `RunLengthDecode`/`RL`, `CCITTFaxDecode`/`CCF`, `DCTDecode`/`DCT`, `JPXDecode`, `JBIG2Decode`, and `Crypt`, plus `Unknown(name)`; `is_image_payload()` marks CCITT/DCT/JPX/JBIG2 and `aborts_extraction()` marks LZW/Crypt.
- Decoders (`pdf::filters::decode`, dispatched in dictionary order and each bounded by `DEFAULT_MAX_OUTPUT_BYTES`): `ascii_hex_decode` (hex pairs, `>` optional at the end, odd nibble completed low), `ascii85_decode` (`!`..`u` digits, `z` only between groups, `~>` optional at the end, 2..=4-character final group), `run_length_decode` (literal runs, repeat runs `257-len`, `128` end marker optional), `flate_decode` (`zlib_decompress`); an empty chain means the payload already is the content stream.
- Predictors (`pdf::predictor`): `PredictorParams` (`predictor`, `columns`, `colors`, `bits_per_component`, defaults 1/1/1/8) is applied after the filter that names it — `1` is a passthrough, `2` is `TIFF` horizontal differencing for 1/2/4/8/16-bit samples, and `10`..`15` are the PNG row filters (None/Sub/Up/Average/Paeth over 8-bit components).
- Skipped, recorded in the document markers with a `warn` log: a stream whose chain names an image payload filter, an unknown filter name, or a payload the filter chain or predictor cannot undo (the record says `"{filter} decode failed: {error}"`).
- Rejected with `Error::Unsupported`: `/Encrypt` anywhere in the file ("encrypted PDF"), compressed object streams (`/Type /ObjStm`, which need cross-reference parsing), a `/LZWDecode` or `/Crypt` stream (`"… cannot be decoded: text extraction cannot continue"`), and a predictor value or bit depth that cannot be applied (`Predictor 3`, PNG predictor with `/BitsPerComponent 16`, TIFF predictor with `/BitsPerComponent 12`, `/Columns 0`).
- A scanned or text-free document parses to `text == ""`, `pages == []`, `has_text == false` — never an error.
- Page order is stream order (no xref table is parsed): one page per decoded content stream.

### Re-exports

- `parsing::{inflate, zlib_decompress, adler32, crc32, DEFAULT_MAX_OUTPUT_BYTES, ZipArchive, ZipEntry, ZipWriter, DocxParser, DocxDocument, DocxBlock, DOCUMENT_PART, CORE_PROPERTIES_PART, PptxParser, PptxDocument, PptxSlide, PRESENTATION_PART, PRESENTATION_RELS_PART, SLIDES_DIR, NOTES_SLIDES_DIR, OdtParser, OdtDocument, CONTENT_PART, META_PART, PdfParser, PdfDocument, KERNING_SPACE_THRESHOLD, XlsxParser, XlsxWorkbook, XlsxSheet, SheetRef, WORKBOOK_PART, WORKBOOK_RELS_PART, SHARED_STRINGS_PART, parse_xml, XmlDocument, XmlElement, MAX_DEPTH, local_name}`.
- `pptx::CORE_PROPERTIES_PART` stays module-level only (same value as the DOCX constant re-exported at the root).

## Error Conventions

- `Error::Validation` — structurally invalid data: bad `DEFLATE`/`zlib`/ZIP structure, malformed XML (unclosed element, mismatched end tag, unterminated comment/`CDATA`/processing instruction/attribute, text outside the root, nesting over `MAX_DEPTH`), malformed workbook/presentation/content part XML, `ZipWriter` limit violations, a PPTX archive with no slide at all (`"PPTX has no slides: …"`), and an ODT archive without `content.xml`.
- `Error::Retrieval` — data ended early or failed integrity (`Adler-32`/`CRC-32` mismatch, truncated streams), a DOCX/XLSX archive missing a required part (`word/document.xml`, `xl/workbook.xml`, a worksheet), an unusable `xl/sharedStrings.xml` part, and a `t="s"` cell whose shared-string index cannot be resolved.
- `Error::Unsupported` — deliberately unimplemented variants: `zlib` preset dictionaries, ZIP64/split/encrypted archives, non-stored/deflate methods, encrypted PDFs, PDF object streams, PDF `/LZWDecode` and `/Crypt` streams, and a PDF predictor or bit depth the crate cannot apply.
- `Error::NotFound` — a requested ZIP entry or input file does not exist.
- `Error::Transport` — any other I/O failure while reading a file.
- Part parse failures carry their part path as context (`Error::with_context`) so the message names the part.

## Constraints

- No new dependencies: everything is plain Rust over the existing workspace crates (`openjiuwen-common` only).
- `#![forbid(unsafe_code)]`; files are read with `tokio::fs` so nothing blocks the runtime.
- Every allocation is bounded by `max_output`/`DEFAULT_MAX_OUTPUT_BYTES` before bytes are appended (PDF filter decoders and predictors included); XML nesting is bounded by `MAX_DEPTH`, PDF arrays by depth and item count.
- PDF filters and predictors are pure functions over bytes: no panics on crafted input, typed errors instead, `tracing` (`warn`/`debug`) only.
- ZIP paths are byte-oriented: `/` separators and exact, case-sensitive lookups; `read_entry_ignore_case` is the documented ASCII-case-insensitive fallback; `ZipWriter` never normalises a name it is given.
- Malformed input never panics: every slice access is bounds-checked or preceded by a length check.
- Public data types are `serde`-serializable (`snake_case` with `#[serde(default)]` on documents), except the XML tree, which is an internal intermediate representation.
- Files stay around ≤900 lines: `pdf/tests.rs`, `pptx/tests.rs`, `odt/tests.rs`, and `zip/writer/tests.rs` hold the larger test bodies so the implementation files fit.
- Logging uses `openjiuwen_common::logging` (`debug`); libraries never print.

## Testing

- `inflate/tests.rs` (unit, beside the module): canned vectors (`zlib` "hello", a reference-encoder dynamic block with back-references), a hand-derived fixed-Huffman stream for `"A"`, stored-block streams, generated fixed/dynamic/dynamic-with-repeats streams over payloads including all bytes 0..=255, and every documented error path.
- `xml.rs` unit tests: nesting/attributes/ordered children, entity decoding (named/decimal/hex/unknown), comments/`CDATA`/declarations/self-closing tags, preserved whitespace and multiple roots, every structural error, the depth guard, and local-name matching rules.
- `docx.rs` unit tests: block rendering (headings, tabs/breaks, tables with escaped pipes, content-control wrappers), style-id → level mapping, cell text joining, blank paragraphs, missing document part.
- `xlsx.rs` unit tests: shared/inline/numeric/boolean cells, gap rows and padding, multi-letter column references, number normalisation, rich-text shared strings with phonetic runs, relationship target normalisation, per-sheet ids and metadata, missing workbook part, and the shared-string strictness ladder (absent table with a `t="s"` cell, index past the end, synthesised cell reference, non-numeric/missing `<v>`, unusable part, over-declared `count` still tolerated, inline-only sheet needing no table, end-to-end unresolvable lookup).
- `pdf/tests.rs` unit tests: `Tj`/`TJ`/quote operators, kerning spaces, literal escapes/octal/nesting/continuations, hex and UTF-16BE strings, page ordering, `parse_filter_chain`/`parse_decode_parms` (single names, abbreviations, array order, `/Filter` alone, DCT image payload, LZW abort), end-to-end ASCIIHex/ASCII85/RunLength/chained/predicted streams, recorded skips, default markers, stream keyword delimiting, encrypted/object-stream rejections.
- `pdf/filters.rs`, `pdf/predictor.rs`, and `pdf/dictionary.rs` keep their unit tests inline: every decoder's terminator/whitespace/overflow/bound case, both predictors' row reconstructions and unsupported parameters, dictionary keys versus look-alike names (`/Filters`, `/FlateDecode`, `/MyFilter`), `null`/indirect parameter entries, and array alignment.
- `zip/writer/tests.rs` unit tests: stored-entry round-trips through `ZipArchive` (names, contents, order, `archive_size`, non-ASCII names), the empty archive, chaining, directory entries, the UTF-8/stored flags, every documented limit, and a hand-asserted single-entry byte layout.
- `pptx/tests.rs` unit tests (fixtures via `ZipWriter`): presentation relationship order beating numeric order, numeric fallback (`slide2` before `slide10`), multi-run paragraphs and `<a:br>`, table Markdown with escaped pipes, notes and `sldNum` skipping, `dc:title`, missing slides, malformed parts, non-ZIP bytes, `parse_path`, and the placeholder path (unmapped `r:id`, absent target part, `<p:sldId>` without `r:id`, a non-slide target, an empty `sldIdLst` still falling back, clean decks reporting no problems, and serde tolerance for the new fields).
- `odt/tests.rs` unit tests (fixtures via `ZipWriter`): headings with outline levels, paragraphs, inline space runs/tabs/line breaks/spans, tables with covered cells and escaped pipes, flattened lists, empty documents, missing/malformed `content.xml`, `meta.xml` title, `parse_path`, JSON round-trip, and `unsupported_elements` (recorded element set, counted repetition, sorted order, clean documents reporting an empty list, legacy JSON without the field).
- `tests/parsing_office_strictness.rs` (integration): the three office-reader silent-loss paths stated against the public API — XLSX lookup misses (absent table, index past the end, unparsable part, over-declared `count`) and the inline-only workbook that needs no table, PPTX unresolved slides keeping their deck position with `missing_part`/`problems`, the numeric fallback for an empty `sldIdLst`, and ODT recorded elements; plus serde tolerance for payloads without the new fields.
- `tests/parsing_documents.rs` (integration): byte-built stored ZIP fixtures (`build_zip`) and `zlib_stored` PDF streams cover DOCX end to end (title, blocks, empty document, missing part, corrupt ZIP, `parse_path`), XLSX (two sheets with relative and absolute relationship targets, shared/inline/numeric/boolean cells, gap rows, four-column width, row and column documents with ids/metadata, missing parts, no-relationship fallback, empty workbook, `parse_path`), and PDF (uncompressed text, `TJ` kerning plus escapes and hex, `FlateDecode`, multi-stream ordering, skipped-image-stream recording, `/Encrypt`, `/ObjStm`, text-free documents with the completeness markers, `parse_path`).
- `tests/parsing_pdf_filters.rs` (integration): the full filter pipeline with byte-built fixtures — ASCIIHex, ASCII85 (with a `z` group and a partial final group), RunLength, the `/ASCII85Decode`→`/ASCIIHexDecode` and `/ASCII85Decode`→`/RunLengthDecode` chains, Flate plus `/Predictor 12`, a `[null << /Predictor 11 >>]` parameter array, `/LZWDecode` and `/LZW` rejections, DCT/garbage-Flate/unknown-filter recording and dedup, and the serialised `incomplete`/`skipped_filters`/`skipped_streams` markers.
- `tests/parsing_zip.rs` (integration): archives assembled byte by byte, stored and deflated entries, CRC/size failures, unsupported method/encryption/ZIP64/split archives, missing EOCD, `read_text`, `read_entry_ignore_case`, `open` against `tempfile`.
- Tests are offline and deterministic; no wall-clock, no network, no random ordering.

## Routing Table

- `./inflate.rs` → public `DEFLATE`/`zlib` API, checksums, decoder internals (`BitReader`, `Huffman`, block decoders)
- `./inflate/tests.rs` → unit tests plus the in-test `DEFLATE` stream writers
- `./zip.rs` → `ZipArchive`/`ZipEntry`, EOCD and central-directory parsing, entry extraction and verification
- `./zip/writer.rs` → `ZipWriter`: local headers, payloads, central directory and EOCD emission, format limits
- `./zip/writer/tests.rs` → ZIP writer unit tests and round-trips
- `./xml.rs` → tolerant XML tree parser used by the office-document readers
- `./docx.rs` → `DocxParser`/`DocxDocument`/`DocxBlock`, WordprocessingML body rendering
- `./xlsx.rs` → `XlsxParser`/`XlsxWorkbook`/`XlsxSheet`, workbook relationships, the strict shared-string table, worksheet cells
- `./pptx.rs` → `PptxParser`/`PptxDocument`/`PptxSlide`, slide order and relationships, unresolved-slide placeholders and `problems`, shape text, tables, speaker notes
- `./pptx/tests.rs` → PPTX unit tests and `ZipWriter`-built fixtures
- `./odt.rs` → `OdtParser`/`OdtDocument`, OpenDocument body walk, inline content, tables, `meta.xml` title, unsupported-element reporting
- `./odt/tests.rs` → ODT unit tests and `ZipWriter`-built fixtures
- `./pdf.rs` → `PdfParser`/`PdfDocument`, stream scanning, the decode pipeline (`decode_stream`, `ExtractedStreams`), content-stream text operators
- `./pdf/filters.rs` → `Filter` and the ASCIIHex/ASCII85/RunLength/Flate decoders
- `./pdf/predictor.rs` → `PredictorParams` and the TIFF/PNG predictor reconstruction
- `./pdf/dictionary.rs` → `/Filter` chain and `/DecodeParms` window parsing
- `./pdf/tests.rs` → PDF unit tests
- `./mod.rs` → module root, re-exports, and the document-layer overview
