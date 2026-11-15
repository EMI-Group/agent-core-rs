//! Integration suite for the omni document-extraction route.
//!
//! `OmniCreation::scrape` (and therefore `SkillCreator::create_from_url` /
//! `OmniCreation::run`) dispatches a fetched resource to the kernel retrieval
//! loaders when its URL's last path segment names a document the loaders read,
//! or when the response `content-type` names one. This suite drives that route
//! through the public API only: a document is served by `ScriptedFetcher`, the
//! scrape stage produces a `ScrapedPage` whose text comes from the loader, and
//! the deterministic generator turns it into a `SkillDefinition` whose
//! instructions carry the extracted text instead of a UTF-8 approximation of
//! the container bytes.
//!
//! Office documents are `ZIP` packages, so every `DOCX`/`XLSX`/`PPTX`/`ODT`
//! fixture here is assembled with the crate's public stored-`ZIP` writer
//! `ArchiveWriter::write_stored`, holding only the package parts the loader
//! reads. Nothing is inflated, so the fixtures need no office suite and no
//! compression library.
//!
//! Everything is offline and deterministic: the fetcher is scripted, paths come
//! from a `TempWorkspace`, ids are fixed by the fixture builders, and no
//! environment value, wall clock, network socket, or sleep is used.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fmt::Write as _;
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::skill_creator::omni::archive::{ArchiveEntry, ArchiveWriter};
use openjiuwen_dev_tools::skill_creator::omni::{
    Heading, OmniCreation, ScrapedPage, ScriptedFetcher,
};
use openjiuwen_dev_tools::skill_creator::{SkillBundle, SkillCreator, SkillDefinition};
use openjiuwen_test_support::TempWorkspace;

/// The `WordprocessingML` namespace a `DOCX` body declares.
const WORD_NAMESPACE: &str = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
/// The `SpreadsheetML` namespace an `XLSX` part declares.
const SHEET_NAMESPACE: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
/// The office relationship namespace shared by `XLSX` and `PPTX` parts.
const RELATIONSHIP_NAMESPACE: &str =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
/// The package relationship namespace used by `_rels` parts.
const PACKAGE_RELATIONSHIP_NAMESPACE: &str =
    "http://schemas.openxmlformats.org/package/2006/relationships";
/// The `PresentationML` namespace a `PPTX` slide declares.
const PRESENTATION_NAMESPACE: &str = "http://schemas.openxmlformats.org/presentationml/2006/main";
/// The `DrawingML` namespace a `PPTX` shape's text declares.
const DRAWING_NAMESPACE: &str = "http://schemas.openxmlformats.org/drawingml/2006/main";
/// The `OpenDocument` office namespace an `ODT` part declares.
const OFFICE_NAMESPACE: &str = "urn:oasis:names:tc:opendocument:xmlns:office:1.0";
/// The `OpenDocument` text namespace an `ODT` body declares.
const TEXT_NAMESPACE: &str = "urn:oasis:names:tc:opendocument:xmlns:text:1.0";

/// The text a `PDF` content stream carries in these fixtures.
const PDF_TEXT: &str = "Hello PDF";

/// Escapes the XML metacharacters that can appear in element text.
fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escapes the XML metacharacters that can appear in a double-quoted attribute.
fn escape_attribute(text: &str) -> String {
    escape_xml(text).replace('"', "&quot;")
}

/// Returns the `A1`-style column name of a zero-based column position.
fn column_name(position: usize) -> String {
    let mut name = String::new();
    let mut position = position;
    loop {
        let letter = u8::try_from(position % 26).unwrap_or(0);
        name.insert(0, char::from(b'A' + letter));
        if position < 26 {
            return name;
        }
        position = position / 26 - 1;
    }
}

/// Serializes `(path, body)` pairs as a stored `ZIP` archive.
fn stored_zip(entries: &[(&str, &str)]) -> Vec<u8> {
    let staged: Vec<ArchiveEntry> = entries
        .iter()
        .map(|(path, body)| ArchiveEntry {
            path: (*path).to_owned(),
            bytes: body.as_bytes().to_vec(),
            ..ArchiveEntry::default()
        })
        .collect();
    ArchiveWriter::write_stored(&staged).expect("writes a stored zip")
}

/// A minimal file-level `PDF` document holding an uncompressed text stream.
fn pdf_bytes(content: &str) -> Vec<u8> {
    let stream = format!(
        "1 0 obj\n<< /Length {} >>\nstream\n{content}\nendstream\nendobj",
        content.len()
    );
    format!("%PDF-1.7\n{stream}\n%%EOF\n").into_bytes()
}

/// A `DOCX` package whose body holds one paragraph per entry.
fn docx_package(paragraphs: &[&str]) -> Vec<u8> {
    let mut body = String::new();
    for paragraph in paragraphs {
        let _ = write!(
            body,
            "<w:p><w:r><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p>",
            escape_xml(paragraph)
        );
    }
    let document = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <w:document xmlns:w=\"{WORD_NAMESPACE}\"><w:body>{body}</w:body></w:document>"
    );
    stored_zip(&[("word/document.xml", document.as_str())])
}

/// An `XLSX` package with one worksheet whose first row is the header row.
fn xlsx_package(sheet: &str, rows: &[&[&str]]) -> Vec<u8> {
    let mut shared: Vec<String> = Vec::new();
    let mut sheet_data = String::new();
    for (row_index, row) in rows.iter().enumerate() {
        let row_number = row_index + 1;
        let _ = write!(sheet_data, "<row r=\"{row_number}\">");
        for (column_index, cell) in row.iter().enumerate() {
            if cell.trim().is_empty() {
                continue;
            }
            let reference = format!("{}{row_number}", column_name(column_index));
            let index = if let Some(index) = shared.iter().position(|existing| existing == cell) {
                index
            } else {
                shared.push((*cell).to_owned());
                shared.len() - 1
            };
            let _ = write!(
                sheet_data,
                "<c r=\"{reference}\" t=\"s\"><v>{index}</v></c>"
            );
        }
        sheet_data.push_str("</row>");
    }
    let worksheet = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <worksheet xmlns=\"{SHEET_NAMESPACE}\"><sheetData>{sheet_data}</sheetData></worksheet>"
    );
    let workbook = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <workbook xmlns=\"{SHEET_NAMESPACE}\" xmlns:r=\"{RELATIONSHIP_NAMESPACE}\">\
         <sheets><sheet name=\"{}\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>",
        escape_attribute(sheet)
    );
    let relationships = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <Relationships xmlns=\"{PACKAGE_RELATIONSHIP_NAMESPACE}\">\
         <Relationship Id=\"rId1\" Type=\"worksheet\" Target=\"worksheets/sheet1.xml\"/>\
         </Relationships>"
    );
    let mut shared_body = String::new();
    for value in &shared {
        let _ = write!(shared_body, "<si><t>{}</t></si>", escape_xml(value));
    }
    let shared_strings = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <sst xmlns=\"{SHEET_NAMESPACE}\" count=\"{count}\" uniqueCount=\"{count}\">\
         {shared_body}</sst>",
        count = shared.len()
    );
    stored_zip(&[
        ("xl/workbook.xml", workbook.as_str()),
        ("xl/_rels/workbook.xml.rels", relationships.as_str()),
        ("xl/sharedStrings.xml", shared_strings.as_str()),
        ("xl/worksheets/sheet1.xml", worksheet.as_str()),
    ])
}

/// One `PPTX` slide part with a title placeholder and one body shape.
fn slide_part(title: &str, paragraphs: &[&str]) -> String {
    let mut shapes = format!(
        "<p:sp><p:nvSpPr><p:nvPr><p:ph type=\"title\"/></p:nvPr></p:nvSpPr><p:txBody>\
         <a:bodyPr/><a:p><a:r><a:t>{}</a:t></a:r></a:p></p:txBody></p:sp>",
        escape_xml(title)
    );
    if !paragraphs.is_empty() {
        let mut body = String::new();
        for paragraph in paragraphs {
            let _ = write!(
                body,
                "<a:p><a:r><a:t>{}</a:t></a:r></a:p>",
                escape_xml(paragraph)
            );
        }
        let _ = write!(
            shapes,
            "<p:sp><p:nvSpPr><p:nvPr><p:ph idx=\"1\"/></p:nvPr></p:nvSpPr><p:txBody>\
             <a:bodyPr/>{body}</p:txBody></p:sp>"
        );
    }
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <p:sld xmlns:p=\"{PRESENTATION_NAMESPACE}\" xmlns:a=\"{DRAWING_NAMESPACE}\">\
         <p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>"
    )
}

/// A `PPTX` package with one slide per `(title, paragraphs)` entry.
fn pptx_package(slides: &[(&str, &[&str])]) -> Vec<u8> {
    let mut slide_ids = String::new();
    let mut relationship_entries = String::new();
    let mut parts: Vec<(String, String)> = Vec::new();
    for (position, (title, paragraphs)) in slides.iter().enumerate() {
        let number = position + 1;
        let _ = write!(
            slide_ids,
            "<p:sldId id=\"{}\" r:id=\"rId{number}\"/>",
            255 + number
        );
        let _ = write!(
            relationship_entries,
            "<Relationship Id=\"rId{number}\" Type=\"slide\" Target=\"slides/slide{number}.xml\"/>"
        );
        parts.push((
            format!("ppt/slides/slide{number}.xml"),
            slide_part(title, paragraphs),
        ));
    }
    let presentation = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <p:presentation xmlns:p=\"{PRESENTATION_NAMESPACE}\" \
         xmlns:r=\"{RELATIONSHIP_NAMESPACE}\"><p:sldIdLst>{slide_ids}</p:sldIdLst>\
         </p:presentation>"
    );
    let relationships = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <Relationships xmlns=\"{PACKAGE_RELATIONSHIP_NAMESPACE}\">\
         {relationship_entries}</Relationships>"
    );
    let mut entries: Vec<(&str, &str)> = vec![
        ("ppt/presentation.xml", presentation.as_str()),
        ("ppt/_rels/presentation.xml.rels", relationships.as_str()),
    ];
    entries.extend(
        parts
            .iter()
            .map(|(name, body)| (name.as_str(), body.as_str())),
    );
    stored_zip(&entries)
}

/// An `ODT` package whose body holds one paragraph per entry.
fn odt_package(paragraphs: &[&str]) -> Vec<u8> {
    let mut body = String::new();
    for paragraph in paragraphs {
        let _ = write!(body, "<text:p>{}</text:p>", escape_xml(paragraph));
    }
    let content = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <office:document-content xmlns:office=\"{OFFICE_NAMESPACE}\" \
         xmlns:text=\"{TEXT_NAMESPACE}\"><office:body><office:text>{body}</office:text>\
         </office:body></office:document-content>"
    );
    stored_zip(&[("content.xml", content.as_str())])
}

/// Scrapes `url` from a fetcher serving `body` with `content_type`.
async fn scrape_resource(url: &str, content_type: &str, body: Vec<u8>) -> ScrapedPage {
    let fetcher = ScriptedFetcher::new();
    fetcher.with_page(url, content_type, body);
    OmniCreation::new(Arc::new(fetcher))
        .scrape(url)
        .await
        .expect("the resource is scraped")
}

/// Renders the deterministic definition of `page` with no downloaded resources.
fn definition_of(page: &ScrapedPage) -> SkillDefinition {
    OmniCreation::new(Arc::new(ScriptedFetcher::new()))
        .generate(page, &[])
        .expect("the page renders a deterministic definition")
}

/// Returns the content of the `references/sources.md` resource of `definition`.
fn sources_of(definition: &SkillDefinition) -> &str {
    definition
        .resources
        .iter()
        .find(|resource| resource.path == "references/sources.md")
        .map(|resource| resource.content.as_str())
        .expect("the definition carries a sources resource")
}

#[tokio::test]
async fn a_pdf_url_is_extracted_by_the_retrieval_loader() {
    let url = "https://docs.example.com/files/guide.pdf";
    let body = pdf_bytes(&format!("BT ({PDF_TEXT}) Tj ET"));
    // The raw container bytes are printable ASCII, so a fallback that decoded
    // them as text would keep the PDF operators; the loader's text does not.
    assert!(body.windows(4).any(|window| window == b"%PDF"));

    let page = scrape_resource(url, "application/pdf", body).await;
    assert_eq!(page.title, "guide");
    assert!(page.text.contains(PDF_TEXT), "{:?}", page.text);
    assert!(!page.text.contains("%PDF"), "{:?}", page.text);
    assert!(page.headings.is_empty(), "{:?}", page.headings);
    assert!(page.links.is_empty(), "{:?}", page.links);
    assert!(page.images.is_empty(), "{:?}", page.images);

    let definition = definition_of(&page);
    assert_eq!(definition.name, "guide");
    assert!(
        definition.instructions.starts_with("# guide"),
        "{:?}",
        definition.instructions
    );
    assert!(definition.instructions.contains(PDF_TEXT));
    assert!(!definition.instructions.contains("%PDF"));
    assert!(!definition.instructions.contains("stream"));
    assert!(sources_of(&definition).contains(&format!("- Source: {url}")));
    assert!(sources_of(&definition).contains("- Title: guide"));
}

#[tokio::test]
async fn docx_and_xlsx_packages_reach_the_skill_definition() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let docx_url = "https://docs.example.com/files/notes.docx";
    let fetcher = Arc::new(ScriptedFetcher::new());
    fetcher.with_page(
        docx_url,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        docx_package(&["Intro", "Body"]),
    );

    // The whole public route: scrape, download (no candidates), generate, save.
    let outcome = SkillCreator::new()
        .create_from_url(fetcher.clone(), docx_url, &workspace.path("skills"))
        .await
        .expect("the docx becomes a bundle");
    assert_eq!(outcome.definition.name, "notes");
    assert!(
        outcome.definition.instructions.contains("Intro Body"),
        "{:?}",
        outcome.definition.instructions
    );
    let document = outcome.bundle_path.join("SKILL.md");
    assert!(document.is_file(), "SKILL.md must exist at {document:?}");
    let reloaded = SkillBundle::read(&outcome.bundle_path)
        .await
        .expect("the bundle reads back");
    assert_eq!(reloaded.instructions, outcome.definition.instructions);
    assert_eq!(fetcher.calls(), vec![docx_url.to_owned()]);

    let xlsx_url = "https://docs.example.com/files/people.xlsx";
    let page = scrape_resource(
        xlsx_url,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        xlsx_package("People", &[&["Name", "City"], &["alice", "paris"]]),
    )
    .await;
    assert_eq!(page.title, "people");
    assert!(page.text.contains("alice"), "{:?}", page.text);
    assert!(page.text.contains("paris"), "{:?}", page.text);

    let definition = definition_of(&page);
    assert!(definition.instructions.contains("alice"));
    assert!(definition.instructions.contains("paris"));
    assert!(sources_of(&definition).contains("- Title: people"));
}

#[tokio::test]
async fn pptx_and_odt_packages_reach_the_skill_definition() {
    let pptx_url = "https://docs.example.com/files/talk.pptx";
    let page = scrape_resource(
        pptx_url,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        pptx_package(&[("Deck", &["Agenda"])]),
    )
    .await;
    // The slide loader prefixes each slide with a level-2 markdown heading and
    // stamps the file stem as the title.
    assert_eq!(page.text, "## Slide 1\nDeck\nAgenda");
    assert_eq!(page.title, "talk");
    assert_eq!(
        page.headings,
        vec![Heading {
            level: 2,
            text: "Slide 1".to_owned(),
        }]
    );

    let definition = definition_of(&page);
    assert!(definition.instructions.contains("## Steps"));
    assert!(definition.instructions.contains("- **Slide 1**"));
    assert!(definition.instructions.contains("Deck"));
    assert!(definition.instructions.contains("Agenda"));
    assert!(sources_of(&definition).contains("- Title: talk"));

    let odt_url = "https://docs.example.com/files/memo.odt";
    let page = scrape_resource(
        odt_url,
        "application/vnd.oasis.opendocument.text",
        odt_package(&["Intro", "Body"]),
    )
    .await;
    assert_eq!(page.text, "Intro\nBody");
    assert_eq!(page.title, "memo");
    assert!(page.headings.is_empty(), "{:?}", page.headings);

    let definition = definition_of(&page);
    assert_eq!(definition.name, "memo");
    assert!(definition.instructions.contains("Intro Body"));
}

#[tokio::test]
async fn a_content_type_dispatches_a_suffixless_url_to_the_same_loader() {
    let body = pdf_bytes(&format!("BT ({PDF_TEXT}) Tj ET"));
    let extension_url = "https://docs.example.com/guide.pdf";
    let header_url = "https://docs.example.com/guide";

    let by_extension = scrape_resource(extension_url, "application/pdf", body.clone()).await;
    // The header ignores case and carries a `charset` parameter; the URL names
    // no extension, so only the header can select the loader.
    let by_content_type =
        scrape_resource(header_url, "APPLICATION/PDF; charset=binary", body).await;

    assert_eq!(by_content_type.text, by_extension.text);
    assert_eq!(by_content_type.title, "guide");
    assert_eq!(by_content_type.title, by_extension.title);
    assert_eq!(by_content_type.headings, by_extension.headings);
    assert_eq!(
        by_content_type.url, header_url,
        "the page reports the URL it was fetched from"
    );

    let extension_definition = definition_of(&by_extension);
    let content_type_definition = definition_of(&by_content_type);
    assert_eq!(
        content_type_definition.instructions,
        extension_definition.instructions
    );
    assert!(content_type_definition.instructions.contains(PDF_TEXT));
}

#[tokio::test]
async fn a_failed_document_loader_falls_back_to_the_text_route() {
    let url = "https://docs.example.com/files/broken.pdf";
    let body = b"plain words, not a pdf".to_vec();
    let page = scrape_resource(url, "application/pdf", body).await;

    // The loader cannot read these bytes, so the plain-text route delivers them
    // verbatim: no title (the loader would have stamped the file stem) and no
    // markdown structure.
    assert_eq!(page.text, "plain words, not a pdf");
    assert_eq!(page.title, "");
    assert!(page.headings.is_empty(), "{:?}", page.headings);
    assert!(page.code_blocks.is_empty(), "{:?}", page.code_blocks);

    let definition = definition_of(&page);
    assert!(definition.instructions.contains("plain words, not a pdf"));
}

#[tokio::test]
async fn a_document_yielding_no_text_names_the_url_and_format() {
    let url = "https://docs.example.com/files/empty.pdf";
    let fetcher = ScriptedFetcher::new();
    // Whitespace is neither loader text nor usable plain text, so the document
    // format is unusable and the stage must say so instead of returning a blank
    // page.
    fetcher.with_page(url, "application/pdf", b" \n\t\n".to_vec());

    let error = OmniCreation::new(Arc::new(fetcher))
        .scrape(url)
        .await
        .expect_err("an empty document is a validation error");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("extracted no text"), "{message}");
    assert!(message.contains(url), "{message}");
    assert!(message.contains(".pdf"), "{message}");
}

#[tokio::test]
async fn an_archive_page_unpacks_its_entries_as_text() {
    let url = "https://docs.example.com/files/site.zip";
    let index = "<!DOCTYPE html>\n<html>\n<head><title>Widget docs</title></head>\n<body><h1>Widget</h1></body>\n</html>";
    let guide = "# Archive Guide\n\nRun the widget.";
    let fetcher = ScriptedFetcher::new();
    fetcher.with_page(
        url,
        "application/zip",
        stored_zip(&[("index.html", index), ("notes.md", guide)]),
    );

    let page = OmniCreation::new(Arc::new(fetcher))
        .scrape(url)
        .await
        .expect("the archive is unpacked");

    // `archive_page` reads every file entry as markdown text in archive order:
    // the `index.html` payload is delivered verbatim (it is not HTML-parsed, so
    // it contributes no links or images) and the level-1 ATX heading of the
    // markdown entry becomes the title and heading.
    assert!(page.text.contains("<!DOCTYPE html>"), "{:?}", page.text);
    assert!(page.text.contains("<h1>Widget</h1>"), "{:?}", page.text);
    assert!(page.text.contains("Run the widget."), "{:?}", page.text);
    assert_eq!(page.title, "Archive Guide");
    assert_eq!(
        page.headings,
        vec![Heading {
            level: 1,
            text: "Archive Guide".to_owned(),
        }]
    );
    assert!(page.links.is_empty(), "{:?}", page.links);
    assert!(page.images.is_empty(), "{:?}", page.images);

    let definition = definition_of(&page);
    assert!(definition.instructions.contains("# Archive Guide"));
    assert!(definition.instructions.contains("Run the widget."));
    assert!(sources_of(&definition).contains(&format!("- Source: {url}")));
}
