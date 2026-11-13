#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end tests for the multimodal module exposed at
//! `openjiuwen_retrieval::multimodal`.
//!
//! Ports the offline behavior of the Python `MultimodalDocument`,
//! `ImageCaptioner`/`ImageParser`, and `ExcelParser` tests. Everything is
//! deterministic: the image loader uses the offline captioner, temp files come
//! from `tempfile`, and generated 32-character data ids are only checked for
//! shape.
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::multimodal::{
    parse_data_uri, to_data_uri, ImageCaptioner, ImageLoader, Modality, MultimodalDocument,
    OfflineCaptioner, TableData,
};
use openjiuwen_retrieval::{Document, DocumentLoader};
use serde_json::{json, Value};

/// Writes `bytes` into `dir/name` and returns the path as a `String`.
fn write_file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    path.to_str().unwrap().to_owned()
}

#[test]
fn data_uris_round_trip_and_reject_malformed_input() {
    let uri = to_data_uri(Modality::Image, "image/png", b"\x89PNG\r\n");
    assert_eq!(uri, "data:image/png;base64,iVBORw0K");
    let (mime, bytes) = parse_data_uri(&uri).unwrap();
    assert_eq!(mime, "image/png");
    assert_eq!(bytes, b"\x89PNG\r\n");

    let audio = to_data_uri(Modality::Audio, "audio/wav", b"RIFF");
    assert_eq!(audio, "data:audio/wav;base64,UklGRg==");
    assert_eq!(
        parse_data_uri(&audio),
        Some(("audio/wav".to_owned(), b"RIFF".to_vec()))
    );

    for malformed in [
        "",
        "https://example.com/a.png",
        "data:image/png;base64",
        "data:image/png;base64,***",
        "data:;base64,AAAA",
        "data:image/png,plain",
    ] {
        assert_eq!(parse_data_uri(malformed), None, "{malformed}");
    }
}

#[test]
fn content_shapes_cover_every_modality() {
    let mut document = MultimodalDocument::new("fallback text");
    document.add_part(Modality::Text, "what is this?", "");
    document.add_part(Modality::Image, "https://example.com/a.png", "");
    document.add_part(Modality::Video, "https://example.com/a.mp4", "video-key");
    document.add_part(
        Modality::Audio,
        "data:audio/wav;base64,UklGRg==",
        "audio-key",
    );

    assert_eq!(
        document.content().unwrap(),
        vec![
            json!({"type": "text", "text": "what is this?"}),
            json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}),
            json!({
                "type": "video_url",
                "video_url": {"url": "https://example.com/a.mp4"},
                "uuid": "video-key",
            }),
            json!({
                "type": "input_audio",
                "input_audio": {"data": "data:audio/wav;base64,UklGRg==", "format": "wav"},
                "uuid": "audio-key",
            }),
        ]
    );
    assert_eq!(MultimodalDocument::new("empty").content().unwrap().len(), 0);

    let mut bad_audio = MultimodalDocument::new("");
    bad_audio.add_part(Modality::Audio, "https://example.com/a.wav", "");
    let error = bad_audio.content().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("data:audio/"));
}

#[test]
fn strip_distinguishes_empty_documents() {
    assert!(MultimodalDocument::new("").strip().is_none());
    assert!(MultimodalDocument::default().strip().is_none());
    assert_eq!(
        MultimodalDocument::new("body").strip().unwrap().text,
        "body"
    );

    let mut with_part = MultimodalDocument::new("");
    with_part.add_part(Modality::Image, "https://example.com/a.png", "");
    assert!(with_part.strip().is_some());
}

#[test]
fn add_field_validates_and_tags_non_text_parts() {
    let mut document = MultimodalDocument::with_id("doc-1", "body");
    document.add_field(Modality::Text, "plain").unwrap();
    document
        .add_field(Modality::Image, "https://example.com/a.png")
        .unwrap();
    assert_eq!(document.modality_summary().len(), 2);
    assert_eq!(document.modality_summary().get("image"), Some(&1));
    assert_eq!(document.parts[0].data_id, "");
    assert_eq!(document.parts[1].data_id.len(), 32);

    let error = document.add_field(Modality::Image, "a.png").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(document
        .add_field(Modality::Audio, "https://example.com/a.wav")
        .is_err());
}

#[tokio::test]
async fn add_file_encodes_a_temp_image() {
    let dir = tempfile::tempdir().unwrap();
    let source = write_file(&dir, "pixel.PNG", b"\x89PNG");

    let mut document = MultimodalDocument::new("");
    document.add_file(Modality::Image, &source).await.unwrap();
    assert_eq!(document.parts.len(), 1);
    let part = &document.parts[0];
    assert_eq!(part.modality, Modality::Image);
    assert_eq!(part.data, "data:image/png;base64,iVBORw==");
    assert_eq!(part.data_id.len(), 32);
    assert!(part.data_id.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(
        document.content().unwrap(),
        vec![json!({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw=="},
            "uuid": part.data_id,
        })]
    );

    let error = document
        .add_file(Modality::Image, dir.path().join("absent.jpg"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

/// A captioner proving that injection replaces the offline degradation path.
struct UpperCaseCaptioner;

#[async_trait::async_trait]
impl ImageCaptioner for UpperCaseCaptioner {
    async fn caption(
        &self,
        image_paths: &[String],
    ) -> Result<Vec<String>, openjiuwen_common::error::Error> {
        Ok(image_paths
            .iter()
            .map(|path| format!("caption of {path}"))
            .collect())
    }
}

#[tokio::test]
async fn image_loader_produces_captioned_documents() {
    let dir = tempfile::tempdir().unwrap();
    let source = write_file(&dir, "cat.png", b"\x89PNG\r\n\x1a\n");

    let documents = ImageLoader::new().load(&source).await.unwrap();
    assert_eq!(documents.len(), 1);
    let document = &documents[0];
    assert_eq!(document.text, "[image] cat.png");
    assert_eq!(
        document.metadata.get("image_path"),
        Some(&Value::String(source.clone()))
    );
    assert_eq!(
        document.metadata.get("mime_type"),
        Some(&Value::String("image/png".to_owned()))
    );
    assert_eq!(
        document.metadata.get("byte_size"),
        Some(&Value::from(8_u64))
    );
    assert_eq!(
        document.metadata.get("modality"),
        Some(&Value::String("image".to_owned()))
    );
    assert_eq!(
        document.metadata.get("caption_status"),
        Some(&Value::String("offline".to_owned()))
    );
    assert_eq!(document.id.len(), 36);

    // The injected captioner wins and marks itself as custom.
    let injected = ImageLoader::new()
        .with_captioner(Arc::new(UpperCaseCaptioner))
        .load(&source)
        .await
        .unwrap();
    assert_eq!(injected[0].text, format!("caption of {source}"));
    assert_eq!(
        injected[0].metadata.get("caption_status"),
        Some(&Value::String("custom".to_owned()))
    );

    // The data URI is opt-in metadata.
    let with_uri = ImageLoader::new()
        .with_data_uri(true)
        .load(&source)
        .await
        .unwrap();
    assert_eq!(
        with_uri[0].metadata.get("data_uri"),
        Some(&Value::String(
            "data:image/png;base64,iVBORw0KGgo=".to_owned()
        ))
    );
}

#[tokio::test]
async fn image_loader_reports_missing_unsupported_and_empty_files() {
    let dir = tempfile::tempdir().unwrap();
    let loader = ImageLoader::new();

    let missing = dir.path().join("absent.gif");
    let error = loader.load(missing.to_str().unwrap()).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);

    let error = loader.load("scan.pdf").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("Unsupported image format: .pdf"));

    let empty = write_file(&dir, "empty.png", b"");
    let documents = loader.load(&empty).await.unwrap();
    assert_eq!(documents[0].text, "");
    assert_eq!(
        documents[0].metadata.get("byte_size"),
        Some(&Value::from(0_u64))
    );
}

#[tokio::test]
async fn offline_captioner_describes_files_by_name() {
    let captioner = OfflineCaptioner::new();
    let captions = captioner
        .caption(&["/tmp/a/cat.png".to_owned(), "dog.jpg".to_owned()])
        .await
        .unwrap();
    assert_eq!(captions, vec!["[image] cat.png", "[image] dog.jpg"]);
    assert_eq!(captioner.caption_status(), "offline");

    let custom = OfflineCaptioner::with_prefix("fig ");
    assert_eq!(
        custom.caption(&["x/y.png".to_owned()]).await.unwrap(),
        vec!["fig y.png"]
    );
}

#[test]
fn table_markdown_round_trips_escaped_cells() {
    let table = TableData::new(
        vec!["a|b".to_owned(), "notes".to_owned()],
        vec![
            vec!["1".to_owned(), "line one\nline two".to_owned()],
            vec!["2".to_owned(), "pipe | inside".to_owned()],
        ],
    );
    let markdown = table.to_markdown();
    assert!(markdown.starts_with("| a\\|b | notes |"));
    assert!(markdown.contains("| --- | --- |"));
    assert!(markdown.contains("line one<br>line two"));
    assert_eq!(TableData::from_markdown(&markdown).unwrap(), table);

    // Missing separator and alignment markers are tolerated.
    let loose =
        TableData::from_markdown("| h1 | h2 |\n|:---|:---:|\n| a | b |\n| c | d |\n").unwrap();
    assert_eq!(loose.headers, vec!["h1", "h2"]);
    assert_eq!(loose.rows.len(), 2);
    assert_eq!(loose.rows[1], vec!["c", "d"]);
}

#[test]
fn table_documents_match_the_csv_scheme() {
    let table = TableData::new(
        vec!["name".to_owned(), "city".to_owned()],
        vec![
            vec!["alice".to_owned(), "paris".to_owned()],
            vec!["bob".to_owned(), String::new()],
        ],
    );

    let documents = table.to_documents("data/people.csv", Some("people"));
    let ids: Vec<&str> = documents
        .iter()
        .map(|document| document.id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec![
            "data/people.csv_s0_r2",
            "data/people.csv_s0_r3",
            "data/people.csv_s0_c0",
            "data/people.csv_s0_c1",
        ]
    );
    assert_eq!(documents[0].text, "name: alice, city: paris");
    assert_eq!(documents[1].text, "name: bob, city: ");
    assert_eq!(
        documents[0].metadata.get("source_type"),
        Some(&json!("row"))
    );
    assert_eq!(documents[0].metadata.get("row_index"), Some(&json!(2)));
    assert_eq!(
        documents[0].metadata.get("sheet_name"),
        Some(&json!("people"))
    );
    assert_eq!(documents[2].text, "Column name: name. Values: alice, bob");
    assert_eq!(
        documents[2].metadata.get("column_name"),
        Some(&json!("name"))
    );
    assert_eq!(
        documents[2].metadata.get("source_type"),
        Some(&json!("column"))
    );

    let columns = table.to_column_documents("t", None);
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[1].id, "t_s0_c1");
    assert_eq!(columns[1].text, "Column name: city. Values: paris");
    assert_eq!(columns[1].metadata.get("sheet_name"), Some(&Value::Null));

    // Plain documents can be fed to the text-only pipeline unchanged.
    let plain: Vec<Document> = documents;
    assert!(plain.iter().all(|document| !document.text.is_empty()));
}

#[test]
fn table_chunks_carry_modality_metadata() {
    let table = TableData::new(
        vec!["a".to_owned(), "b".to_owned()],
        vec![vec!["1".to_owned(), "2".to_owned()]],
    );
    let chunks = table.to_chunks("sheet-1");

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].id, "sheet-1_table");
    assert_eq!(chunks[0].doc_id, "sheet-1");
    assert_eq!(chunks[0].text, "| a | b |\n| --- | --- |\n| 1 | 2 |");
    assert_eq!(chunks[0].metadata.get("modality"), Some(&json!("table")));
    assert_eq!(chunks[0].metadata.get("source_type"), Some(&json!("table")));
    assert_eq!(chunks[0].metadata.get("sheet_name"), Some(&Value::Null));
    assert!(chunks[0].metadata.get("row_index").is_none());

    assert_eq!(chunks[1].id, "sheet-1_r2");
    assert_eq!(chunks[1].text, "a: 1, b: 2");
    assert_eq!(chunks[1].metadata.get("source_type"), Some(&json!("row")));
    assert_eq!(chunks[1].metadata.get("row_index"), Some(&json!(2)));
    assert_eq!(chunks[1].metadata.get("modality"), Some(&json!("table")));
    assert!(chunks.iter().all(|chunk| chunk.embedding.is_none()));

    assert_eq!(TableData::default().to_chunks("empty").len(), 0);
}

#[test]
fn multimodal_documents_round_trip_through_json() {
    let mut document = MultimodalDocument::with_id("doc-7", "caption");
    document.metadata.insert("origin".to_owned(), json!("scan"));
    document.add_part(Modality::Image, "data:image/png;base64,iVBORw==", "abc");

    let value = serde_json::to_value(&document).unwrap();
    assert_eq!(value.get("id_"), Some(&json!("doc-7")));
    assert!(value.get("id").is_none());
    assert_eq!(value["parts"][0]["modality"], json!("image"));

    let restored: MultimodalDocument = serde_json::from_value(value).unwrap();
    assert_eq!(restored, document);

    // Unknown fields are tolerated and missing fields default.
    let lenient: MultimodalDocument =
        serde_json::from_value(json!({"text": "body", "future": {"x": 1}})).unwrap();
    assert_eq!(lenient.text, "body");
    assert_eq!(lenient.parts.len(), 0);
    assert_eq!(lenient.id, "");
}
