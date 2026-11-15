//! Tests for the staged omni pipeline.
//!
//! Everything runs offline and deterministically: [`ScriptedFetcher`] serves the
//! network inputs, [`ScriptedModel`] supplies model answers, and
//! [`TempWorkspace`] provides the bundle roots.

use std::sync::Arc;

use openjiuwen_test_support::{ScriptedModel, ScriptedTurn, TempWorkspace};
use serde_json::json;

use super::*;
use crate::skill_creator::omni::archive::{ArchiveEntry, ArchiveWriter};
use crate::skill_creator::omni::fetch::ScriptedFetcher;
use crate::skill_creator::SkillBundle;

/// The main page URL served in tests.
const MAIN_URL: &str = "https://docs.example.com/guide";
/// The single subpage linked from the main page.
const SUBPAGE_URL: &str = "https://docs.example.com/guide/details";
/// The single image referenced from the main page.
const IMAGE_URL: &str = "https://docs.example.com/guide/screenshot.png";
/// A second image with a different extension, used to prove the saved extension.
const PHOTO_URL: &str = "https://docs.example.com/guide/photo.jpeg";

/// The main HTML page: title, description, headings, text, one link, one image.
fn main_html() -> String {
    format!(
        "<!doctype html><html><head><title>Install the Widget</title>\
         <meta name=\"description\" content=\"Learn to install the widget.\"></head><body>\
         <h1>Install the Widget</h1>\
         <h2>Steps</h2>\
         <p>Download the package, then run the installer.</p>\
         <a href=\"{SUBPAGE_URL}\">Details</a>\
         <img src=\"{IMAGE_URL}\" alt=\"install screenshot\">\
         </body></html>"
    )
}

/// The subpage HTML: its own title, heading, and text.
fn subpage_html() -> &'static str {
    "<!doctype html><html><head><title>Details</title></head><body>\
     <h2>Requirements</h2><p>Rust 1.75 or later.</p></body></html>"
}

/// A minimal `png` payload padded to `size` bytes (at least 8).
fn png_body(size: usize) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.resize(size.max(bytes.len()), 0);
    bytes
}

/// A minimal `jpeg` payload padded to `size` bytes (at least 3).
fn jpeg_body(size: usize) -> Vec<u8> {
    let mut bytes = b"\xFF\xD8\xFF".to_vec();
    bytes.resize(size.max(bytes.len()), 0);
    bytes
}

/// A page with two images of different formats plus the usual subpage link.
fn two_image_fetcher() -> ScriptedFetcher {
    let fetcher = ScriptedFetcher::new();
    fetcher.with_html(
        MAIN_URL,
        format!(
            "<!doctype html><html><head><title>Install the Widget</title></head><body>\
             <h1>Install the Widget</h1>\
             <p>Download the package, then run the installer.</p>\
             <img src=\"{IMAGE_URL}\" alt=\"install screenshot\">\
             <img src=\"{PHOTO_URL}\" alt=\"install photo\">\
             <a href=\"{SUBPAGE_URL}\">Details</a>\
             </body></html>"
        ),
    );
    fetcher.with_page(IMAGE_URL, "image/png", png_body(2048));
    fetcher.with_page(PHOTO_URL, "image/jpeg", jpeg_body(4096));
    fetcher.with_html(SUBPAGE_URL, subpage_html());
    fetcher
}

/// A fetcher serving the main page, its subpage, and its image.
fn page_fetcher() -> ScriptedFetcher {
    let fetcher = ScriptedFetcher::new();
    fetcher.with_html(MAIN_URL, main_html());
    fetcher.with_html(SUBPAGE_URL, subpage_html());
    fetcher.with_page(IMAGE_URL, "image/png", png_body(2048));
    fetcher
}

/// The scraped main page.
async fn scraped_main() -> ScrapedPage {
    OmniCreation::new(Arc::new(page_fetcher()))
        .scrape(MAIN_URL)
        .await
        .expect("scrapes")
}

/// One `ustar` header for a regular file, with a valid checksum.
fn tar_header(name: &str, size: usize, typeflag: u8) -> [u8; 512] {
    assert!(name.len() < 100, "tar name field overflow: {name}");
    let mut header = [0_u8; 512];
    header[..name.len()].copy_from_slice(name.as_bytes());
    let size_field = format!("{size:o}");
    let size_bytes = size_field.as_bytes();
    header[124..124 + size_bytes.len()].copy_from_slice(size_bytes);
    header[156] = typeflag;
    header[257..262].copy_from_slice(b"ustar");
    for byte in &mut header[148..156] {
        *byte = b' ';
    }
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    let checksum = format!("{checksum:06o}\0 ");
    header[148..156].copy_from_slice(checksum.as_bytes());
    header
}

/// The padding needed to align a payload of `size` bytes to the next block.
fn padding(size: usize) -> usize {
    (512 - size % 512) % 512
}

/// A `tar` archive holding one markdown file and one binary file.
fn tar_archive() -> Vec<u8> {
    let guide = "# Install\n\nRun the installer.\n\n```sh\ncargo install widget\n```";
    let mut binary = b"logo payload".to_vec();
    binary.push(0);
    let mut archive = Vec::new();
    let bytes = guide.as_bytes();
    archive.extend_from_slice(&tar_header("guide.md", bytes.len(), b'0'));
    archive.extend_from_slice(bytes);
    archive.resize(archive.len() + padding(bytes.len()), 0);
    archive.extend_from_slice(&tar_header("assets/logo.bin", binary.len(), b'0'));
    archive.extend_from_slice(&binary);
    archive.resize(archive.len() + padding(binary.len()), 0);
    archive.extend_from_slice(&[0_u8; 1024]);
    archive
}

/// CRC-32 (IEEE) of `bytes`, in the form a gzip trailer stores it.
fn crc32(bytes: &[u8]) -> u32 {
    let mut state = 0xFFFF_FFFFu32;
    for &byte in bytes {
        state ^= u32::from(byte);
        for _ in 0..8 {
            state = if state & 1 == 1 {
                (state >> 1) ^ 0xEDB8_8320
            } else {
                state >> 1
            };
        }
    }
    !state
}

/// One gzip member holding `payload` as a single stored (uncompressed) DEFLATE
/// block: the fixed empty header, `BFINAL=1`/`BTYPE=00`, `LEN`/`NLEN`, the raw
/// payload, then the CRC-32 and `ISIZE` trailer. No gzip writer exists because
/// nothing in the crate needs to write one outside this test.
fn gzip_stored(payload: &[u8]) -> Vec<u8> {
    let length = u16::try_from(payload.len()).expect("a stored block payload fits 16 bits");
    let size = u32::try_from(payload.len()).expect("the payload length fits 32 bits");
    let mut member = vec![0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff];
    member.push(0x01);
    member.extend_from_slice(&length.to_le_bytes());
    member.extend_from_slice(&(!length).to_le_bytes());
    member.extend_from_slice(payload);
    member.extend_from_slice(&crc32(payload).to_le_bytes());
    member.extend_from_slice(&size.to_le_bytes());
    member
}

#[tokio::test]
async fn url_to_bundle_end_to_end() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let outcome = OmniCreation::new(Arc::new(page_fetcher()))
        .run(MAIN_URL, workspace.root())
        .await
        .expect("runs");

    assert!(outcome.resources.contains(&SOURCES_RESOURCE.to_owned()));
    assert!(outcome
        .resources
        .contains(&"references/page-00.md".to_owned()));
    assert!(tokio::fs::try_exists(&outcome.bundle_path)
        .await
        .expect("probe"));
    assert!(outcome
        .bundle_path
        .ends_with(outcome.definition.name.as_str()));

    let parsed = SkillBundle::read(&outcome.bundle_path)
        .await
        .expect("reads the bundle");
    assert_eq!(parsed.name, outcome.definition.name);
    assert_eq!(parsed.description, outcome.definition.description);
    assert_eq!(parsed.instructions, outcome.definition.instructions);
    parsed.validate().expect("valid definition");

    assert_eq!(outcome.definition.name, "install-the-widget");
    assert!(outcome.definition.instructions.contains("## Steps"));
    assert!(outcome
        .definition
        .instructions
        .contains("references/page-00.md"));
}

#[tokio::test]
async fn tar_page_input_yields_markdown_entries() {
    let fetcher = ScriptedFetcher::new();
    fetcher.with_page(
        "https://example.com/skill.tar",
        "application/x-tar",
        tar_archive(),
    );
    let page = OmniCreation::new(Arc::new(fetcher))
        .scrape("https://example.com/skill.tar")
        .await
        .expect("scrapes the archive");

    assert_eq!(page.title, "Install");
    assert_eq!(
        page.headings,
        vec![Heading {
            level: 1,
            text: "Install".to_owned(),
        }]
    );
    assert_eq!(page.code_blocks, vec!["cargo install widget".to_owned()]);
    assert!(page.text.contains("Run the installer."));
    assert!(!page.text.contains("logo"), "binary entry must be skipped");
}

#[tokio::test]
async fn zip_and_gzip_page_inputs_unpack() {
    let zip_url = "https://example.com/skill.zip";
    let gzip_url = "https://example.com/skill.tar.gz";
    let zip_bytes = ArchiveWriter::write_stored(&[ArchiveEntry {
        path: "guide.md".to_owned(),
        bytes: b"# Zip Guide\n\nbody text\n".to_vec(),
        is_dir: false,
        ..ArchiveEntry::default()
    }])
    .expect("writes the zip");

    let fetcher = ScriptedFetcher::new();
    fetcher.with_page(zip_url, "application/zip", zip_bytes);
    fetcher.with_page(gzip_url, "application/gzip", gzip_stored(&tar_archive()));
    let creation = OmniCreation::new(Arc::new(fetcher));

    let page = creation
        .scrape(zip_url)
        .await
        .expect("unpacks the zip input");
    assert_eq!(page.title, "Zip Guide");
    assert!(page.text.contains("body text"), "{:?}", page.text);

    let page = creation
        .scrape(gzip_url)
        .await
        .expect("unpacks the gzip-wrapped tar input");
    assert_eq!(
        page.headings,
        vec![Heading {
            level: 1,
            text: "Install".to_owned(),
        }]
    );
    assert!(page.text.contains("Run the installer."), "{:?}", page.text);
    assert!(!page.text.contains("logo"), "binary entry must be skipped");

    // A truncated container is unusable input, not a deferred capability.
    let truncated = ScriptedFetcher::new();
    truncated.with_page(zip_url, "application/zip", b"PK\x03\x04payload".to_vec());
    truncated.with_page(gzip_url, "application/gzip", vec![0x1f, 0x8b, 0x08, 0x00]);
    let creation = OmniCreation::new(Arc::new(truncated));
    for url in [zip_url, gzip_url] {
        let error = creation.scrape(url).await.expect_err("truncated container");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation,
            "{url}: {error}"
        );
    }
}

#[tokio::test]
async fn undecodable_containers_surface_unsupported_with_the_codec() {
    let xz_url = "https://example.com/skill.tar.xz";
    let zstd_url = "https://example.com/skill.tar.zst";
    let fetcher = ScriptedFetcher::new();
    fetcher.with_page(
        xz_url,
        "application/x-xz",
        b"\xFD7zXZ\x00xz-stream-body".to_vec(),
    );
    fetcher.with_page(
        zstd_url,
        "application/zstd",
        vec![0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x00],
    );
    let creation = OmniCreation::new(Arc::new(fetcher));

    for (url, codec) in [(xz_url, "xz"), (zstd_url, "zstd")] {
        let error = creation
            .scrape(url)
            .await
            .expect_err("no decoder is bundled for this codec");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported,
            "{url}: {error}"
        );
        let message = error.to_string();
        assert!(message.contains(codec), "{url}: {message}");
        assert!(message.contains("decoder"), "{url}: {message}");
    }
}

#[tokio::test]
async fn caps_disable_and_truncate_work() {
    let fetcher = page_fetcher();
    let creation = OmniCreation::new(Arc::new(fetcher.clone()))
        .with_config(OmniConfig::default().with_max_images(0));
    let page = creation.scrape(MAIN_URL).await.expect("scrapes");
    creation.download(&page).await.expect("downloads");
    let calls = fetcher.calls();
    assert!(!calls.iter().any(|url| url == IMAGE_URL), "{calls:?}");
    assert!(calls.iter().any(|url| url == SUBPAGE_URL), "{calls:?}");

    let fetcher = page_fetcher();
    let creation = OmniCreation::new(Arc::new(fetcher.clone()))
        .with_config(OmniConfig::default().with_max_links(0));
    let page = creation.scrape(MAIN_URL).await.expect("scrapes");
    creation.download(&page).await.expect("downloads");
    let calls = fetcher.calls();
    assert!(!calls.iter().any(|url| url == SUBPAGE_URL), "{calls:?}");

    let fetcher = page_fetcher();
    let creation = OmniCreation::new(Arc::new(fetcher))
        .with_config(OmniConfig::default().with_max_resources(1));
    let page = creation.scrape(MAIN_URL).await.expect("scrapes");
    let resources = creation.download(&page).await.expect("downloads");
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0].url, IMAGE_URL);
}

#[tokio::test]
async fn blank_pages_are_rejected() {
    let blank_url = "https://example.com/blank";
    let fetcher = ScriptedFetcher::new();
    fetcher.with_html(
        blank_url,
        "<html><head><style>body{color:red}</style></head><body><!-- hidden --></body></html>",
    );
    let error = OmniCreation::new(Arc::new(fetcher))
        .scrape(blank_url)
        .await
        .expect_err("blank page");
    assert!(error.to_string().contains(blank_url), "{error}");
}

#[tokio::test]
async fn model_assisted_generation_uses_the_fenced_json() {
    let page = scraped_main().await;
    let answer = format!(
        "Here you go:\n```json\n{}\n```\n",
        serde_json::to_string_pretty(&json!({
            "name": "model-widget",
            "description": "Install the widget with confidence.",
            "instructions": "# Install\n\n1. Download the package."
        }))
        .expect("payload serializes")
    );
    let creation = OmniCreation::new(Arc::new(ScriptedFetcher::new()))
        .with_model(Arc::new(ScriptedModel::text(answer)));
    let definition = creation
        .generate_with_model(&page, &[])
        .await
        .expect("models the skill");

    assert_eq!(definition.name, "model-widget");
    assert_eq!(
        definition.description,
        "Install the widget with confidence."
    );
    assert_eq!(
        definition.instructions,
        "# Install\n\n1. Download the package.\n\n## Reference Files\n\n\
         - `references/sources.md` — source page provenance, links, and image references"
    );
    assert_eq!(
        definition
            .instructions
            .matches("## Reference Files")
            .count(),
        1
    );
    assert_eq!(definition.resources.len(), 1);
    assert_eq!(definition.resources[0].path, SOURCES_RESOURCE);
    definition.validate().expect("valid definition");
}

#[tokio::test]
async fn model_failures_fall_back_to_the_deterministic_skill() {
    let page = scraped_main().await;
    let deterministic = OmniCreation::new(Arc::new(ScriptedFetcher::new()))
        .generate(&page, &[])
        .expect("generates");

    let garbage = OmniCreation::new(Arc::new(ScriptedFetcher::new()))
        .with_model(Arc::new(ScriptedModel::text("Sure, here you go")));
    let fallback = garbage
        .generate_with_model(&page, &[])
        .await
        .expect("falls back without a fence");
    assert_eq!(fallback, deterministic);

    let malformed = OmniCreation::new(Arc::new(ScriptedFetcher::new())).with_model(Arc::new(
        ScriptedModel::text("```json\n{\"name\": \"bad name\"}\n```"),
    ));
    let fallback = malformed
        .generate_with_model(&page, &[])
        .await
        .expect("falls back on an incomplete payload");
    assert_eq!(fallback, deterministic);

    let failing = OmniCreation::new(Arc::new(ScriptedFetcher::new())).with_model(Arc::new(
        ScriptedModel::new([ScriptedTurn::fail("provider down")]),
    ));
    let fallback = failing
        .generate_with_model(&page, &[])
        .await
        .expect("falls back on a model error");
    assert_eq!(fallback, deterministic);
}

#[tokio::test]
async fn model_assisted_generation_requires_a_model() {
    let page = scraped_main().await;
    let error = OmniCreation::new(Arc::new(ScriptedFetcher::new()))
        .generate_with_model(&page, &[])
        .await
        .expect_err("no model attached");
    assert!(error.to_string().contains("with_model"), "{error}");
}

#[tokio::test]
async fn the_pipeline_is_deterministic() {
    let first = TempWorkspace::new().expect("first workspace");
    let second = TempWorkspace::new().expect("second workspace");
    let creation = OmniCreation::new(Arc::new(page_fetcher()));

    let left = creation.run(MAIN_URL, first.root()).await.expect("runs");
    let right = creation.run(MAIN_URL, second.root()).await.expect("runs");
    assert_eq!(left.definition, right.definition);
    assert_eq!(left.resources, right.resources);

    let left_document = tokio::fs::read(left.bundle_path.join("SKILL.md"))
        .await
        .expect("reads the first document");
    let right_document = tokio::fs::read(right.bundle_path.join("SKILL.md"))
        .await
        .expect("reads the second document");
    assert_eq!(left_document, right_document);
}

#[test]
fn config_validation_rejects_unusable_bounds() {
    assert!(OmniConfig::default().validate().is_ok());
    assert!(OmniConfig::default()
        .with_max_links(0)
        .with_max_images(0)
        .with_max_resources(0)
        .validate()
        .is_ok());

    let error = OmniConfig::default()
        .with_min_image_bytes(0)
        .validate()
        .expect_err("zero floor");
    assert!(error.to_string().contains("min_image_bytes"), "{error}");

    let error = OmniConfig::default()
        .with_max_image_bytes(0)
        .validate()
        .expect_err("zero ceiling");
    assert!(error.to_string().contains("max_image_bytes"), "{error}");

    let error = OmniConfig::default()
        .with_max_resource_bytes(0)
        .validate()
        .expect_err("zero resource ceiling");
    assert!(error.to_string().contains("max_resource_bytes"), "{error}");

    let error = OmniConfig::default()
        .with_min_image_bytes(4096)
        .with_max_image_bytes(1024)
        .validate()
        .expect_err("inverted bounds");
    assert!(error.to_string().contains("max_image_bytes"), "{error}");

    let filter = OmniConfig::default().resource_filter();
    assert_eq!(filter.min_image_bytes, 1024);
    assert_eq!(filter.max_image_bytes, 8 * 1024 * 1024);
    assert!(!filter.allow_cross_host);
}

#[tokio::test]
async fn download_assigns_image_paths_and_provenance() {
    let creation = OmniCreation::new(Arc::new(two_image_fetcher()));
    let page = creation.scrape(MAIN_URL).await.expect("scrapes");
    let resources = creation.download(&page).await.expect("downloads");

    assert_eq!(resources.len(), 3, "{resources:#?}");
    let screenshot = &resources[0];
    assert_eq!(screenshot.url, IMAGE_URL);
    assert_eq!(screenshot.alt, "install screenshot");
    assert_eq!(
        screenshot.saved_path.as_deref(),
        Some("references/img_00.png")
    );
    assert_eq!(screenshot.bytes, png_body(2048));
    assert_eq!(screenshot.sha256, sha256::hex_digest(&screenshot.bytes));
    assert_eq!(screenshot.sha256.len(), 64);

    let photo = &resources[1];
    assert_eq!(photo.url, PHOTO_URL);
    assert_eq!(photo.alt, "install photo");
    assert_eq!(
        photo.saved_path.as_deref(),
        Some("references/img_01.jpeg"),
        "the saved extension follows the URL suffix"
    );
    assert_eq!(photo.bytes, jpeg_body(4096));
    assert_eq!(photo.sha256, sha256::hex_digest(&photo.bytes));
    assert_ne!(photo.sha256, screenshot.sha256);

    let subpage = &resources[2];
    assert_eq!(subpage.url, SUBPAGE_URL);
    assert_eq!(subpage.saved_path, None);
    assert_eq!(subpage.alt, "");
    assert_eq!(subpage.sha256, sha256::hex_digest(&subpage.bytes));
    assert_eq!(subpage.sha256.len(), 64);
}

#[tokio::test]
async fn save_with_images_writes_binary_assets_beside_the_text_assets() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let creation = OmniCreation::new(Arc::new(page_fetcher()));
    let page = creation.scrape(MAIN_URL).await.expect("scrapes");
    let downloaded = creation.download(&page).await.expect("downloads");
    let definition = creation.generate(&page, &downloaded).expect("generates");

    let bundle = creation
        .save_with_images(&definition, &[], &downloaded, workspace.root())
        .await
        .expect("saves");
    assert_eq!(bundle, workspace.root().join(definition.skill_dir_name()));
    assert!(tokio::fs::try_exists(bundle.join("SKILL.md"))
        .await
        .expect("probe"));

    for resource in &definition.resources {
        let written = tokio::fs::read_to_string(bundle.join(&resource.path))
            .await
            .expect("text asset");
        assert_eq!(written, resource.content, "{}", resource.path);
    }

    let image = downloaded
        .iter()
        .find_map(|resource| resource.saved_path.as_deref())
        .expect("one kept image");
    assert_eq!(image, "references/img_00.png");
    let written = tokio::fs::read(bundle.join(image))
        .await
        .expect("image asset");
    assert_eq!(written, png_body(2048), "the payload is byte-for-byte");

    let parsed = SkillBundle::read(&bundle).await.expect("reads the bundle");
    let mut expected: Vec<&str> = definition
        .resources
        .iter()
        .map(|resource| resource.path.as_str())
        .collect();
    expected.sort_unstable();
    let paths: Vec<&str> = parsed
        .resources
        .iter()
        .map(|resource| resource.path.as_str())
        .collect();
    assert_eq!(paths, expected, "binary assets never become text resources");
    assert!(!paths.iter().any(|path| path.starts_with("references/img_")));
}

#[tokio::test]
async fn save_with_images_rejects_unsafe_image_paths() {
    let definition = SkillDefinition::new("demo-skill", "A demo skill", "# Demo\n\nBody.");
    for saved_path in [
        "../escape.png",
        "/tmp/evil.png",
        "references/../../evil.png",
        "references\\img_00.png",
    ] {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let creation = OmniCreation::new(Arc::new(ScriptedFetcher::new()));
        let images = [DownloadedResource {
            url: "https://example.com/a.png".to_owned(),
            content_type: Some("image/png".to_owned()),
            bytes: png_body(2048),
            alt: "screenshot".to_owned(),
            sha256: String::new(),
            saved_path: Some(saved_path.to_owned()),
        }];
        let error = creation
            .save_with_images(&definition, &[], &images, workspace.root())
            .await
            .expect_err("unsafe image path");
        assert!(error.to_string().starts_with("validation:"), "{error}");
        assert!(
            error.to_string().contains(&format!("{saved_path:?}")),
            "{error}"
        );
        assert!(
            !tokio::fs::try_exists(workspace.root().join(definition.skill_dir_name()))
                .await
                .expect("probe"),
            "nothing must be written for {saved_path}"
        );
    }
}

#[tokio::test]
async fn run_lists_text_resources_before_saved_images() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let outcome = OmniCreation::new(Arc::new(page_fetcher()))
        .run(MAIN_URL, workspace.root())
        .await
        .expect("runs");

    assert_eq!(
        outcome.resources,
        vec![
            SOURCES_RESOURCE.to_owned(),
            "references/page-00.md".to_owned(),
            "references/img_00.png".to_owned(),
        ]
    );
    assert!(
        tokio::fs::try_exists(outcome.bundle_path.join("references/img_00.png"))
            .await
            .expect("probe")
    );
}

/// An image URL whose payload is below the 1 KiB floor, so it is filtered out.
const TINY_URL: &str = "https://docs.example.com/guide/tiny.png";

/// A fetcher serving one kept image, one filtered-out image, and one subpage.
fn kept_and_filtered_image_fetcher() -> ScriptedFetcher {
    let fetcher = ScriptedFetcher::new();
    fetcher.with_html(
        MAIN_URL,
        format!(
            "<!doctype html><html><head><title>Install the Widget</title></head><body>\
             <h1>Install the Widget</h1>\
             <p>Download the package, then run the installer.</p>\
             <img src=\"{IMAGE_URL}\" alt=\"install screenshot\">\
             <img src=\"{TINY_URL}\" alt=\"tiny shot\">\
             <a href=\"{SUBPAGE_URL}\">Details</a>\
             </body></html>"
        ),
    );
    fetcher.with_page(IMAGE_URL, "image/png", png_body(2048));
    fetcher.with_page(TINY_URL, "image/png", png_body(64));
    fetcher.with_html(SUBPAGE_URL, subpage_html());
    fetcher
}

/// The scraped page and its downloaded resources of [`kept_and_filtered_image_fetcher`].
async fn kept_and_filtered_downloads() -> (ScrapedPage, Vec<DownloadedResource>) {
    let creation = OmniCreation::new(Arc::new(kept_and_filtered_image_fetcher()));
    let page = creation.scrape(MAIN_URL).await.expect("scrapes");
    let downloaded = creation.download(&page).await.expect("downloads");
    (page, downloaded)
}

/// A model returning one fenced skill definition with `instructions` verbatim.
fn instructions_model(instructions: &str) -> ScriptedModel {
    let answer = format!(
        "```json\n{}\n```\n",
        serde_json::to_string_pretty(&json!({
            "name": "model-widget",
            "description": "Install the widget with confidence.",
            "instructions": instructions,
        }))
        .expect("payload serializes")
    );
    ScriptedModel::text(answer)
}

#[tokio::test]
async fn deterministic_document_references_saved_images_by_path() {
    let (page, downloaded) = kept_and_filtered_downloads().await;
    assert_eq!(
        downloaded
            .iter()
            .filter_map(|resource| resource.saved_path.as_deref())
            .collect::<Vec<_>>(),
        vec!["references/img_00.png"],
        "{downloaded:#?}"
    );

    let creation = OmniCreation::new(Arc::new(ScriptedFetcher::new()));
    let definition = creation.generate(&page, &downloaded).expect("generates");
    let instructions = &definition.instructions;

    assert!(
        instructions.contains("![install screenshot](references/img_00.png)"),
        "{instructions}"
    );
    assert!(
        !instructions.contains(&format!("- install screenshot — {IMAGE_URL}")),
        "a kept image is not repeated as a URL line: {instructions}"
    );
    assert!(
        instructions.contains(&format!("- tiny shot — {TINY_URL}")),
        "a filtered-out image stays a URL line: {instructions}"
    );

    assert_eq!(
        instructions.matches("## Reference Files").count(),
        1,
        "{instructions}"
    );
    let saved = "- `references/img_00.png` — install screenshot";
    let sources = "- `references/sources.md` — source page provenance, links, and image references";
    let saved_at = instructions.find(saved).expect("saved image line");
    let sources_at = instructions.find(sources).expect("sources line");
    assert!(saved_at < sources_at, "{instructions}");
    assert!(
        instructions
            .find("## Reference Images")
            .expect("image section")
            < instructions
                .find("## Reference Files")
                .expect("files section"),
        "{instructions}"
    );
    definition.validate().expect("valid definition");
}

#[tokio::test]
async fn sources_resource_names_the_saved_image_path() {
    let (page, downloaded) = kept_and_filtered_downloads().await;
    let creation = OmniCreation::new(Arc::new(ScriptedFetcher::new()));
    let resources = creation.text_resources(&page, &downloaded);
    let sources = resources
        .iter()
        .find(|resource| resource.path == SOURCES_RESOURCE)
        .expect("sources resource");

    assert!(
        sources.content.contains(&format!(
            "- install screenshot — {IMAGE_URL} (saved as references/img_00.png)"
        )),
        "{}",
        sources.content
    );
    assert!(
        sources
            .content
            .contains(&format!("- tiny shot — {TINY_URL}")),
        "{}",
        sources.content
    );
}

#[tokio::test]
async fn model_generation_keeps_only_saved_image_references() {
    let (page, downloaded) = kept_and_filtered_downloads().await;
    let creation = OmniCreation::new(Arc::new(ScriptedFetcher::new())).with_model(Arc::new(
        instructions_model(
            "# Install\n\n![ok](references/img_00.png)\n\n\
             ![hallucinated](references/img_99.png)\n\n1. Download the package.",
        ),
    ));
    let definition = creation
        .generate_with_model(&page, &downloaded)
        .await
        .expect("models the skill");
    let instructions = &definition.instructions;

    assert!(
        instructions.contains("![ok](references/img_00.png)"),
        "{instructions}"
    );
    assert!(!instructions.contains("img_99.png"), "{instructions}");
    assert_eq!(
        instructions.matches("## Reference Files").count(),
        1,
        "{instructions}"
    );
    assert!(
        instructions.contains("- `references/img_00.png` — install screenshot"),
        "{instructions}"
    );
    assert!(
        instructions.contains("- `references/sources.md` — source page provenance"),
        "{instructions}"
    );
    assert_eq!(definition.resources[0].path, SOURCES_RESOURCE);
    definition.validate().expect("valid definition");
}

#[tokio::test]
async fn model_hallucination_only_lines_disappear_and_collapse() {
    let (page, downloaded) = kept_and_filtered_downloads().await;
    let creation = OmniCreation::new(Arc::new(ScriptedFetcher::new())).with_model(Arc::new(
        instructions_model(
            "# Install\n\n![hallucinated](references/img_99.png)\n\n1. Step.\n\n\
             ![gone](references/img_98.png)",
        ),
    ));
    let definition = creation
        .generate_with_model(&page, &downloaded)
        .await
        .expect("models the skill");

    assert!(
        definition
            .instructions
            .starts_with("# Install\n\n1. Step.\n\n## Reference Files"),
        "{}",
        definition.instructions
    );
    assert!(
        !definition.instructions.contains("\n\n\n"),
        "{}",
        definition.instructions
    );
    assert!(
        !definition.instructions.contains("img_9"),
        "{}",
        definition.instructions
    );
}

#[tokio::test]
async fn blank_instructions_after_stripping_fall_back_to_the_deterministic_skill() {
    let (page, downloaded) = kept_and_filtered_downloads().await;
    let deterministic = OmniCreation::new(Arc::new(ScriptedFetcher::new()))
        .generate(&page, &downloaded)
        .expect("generates");

    let creation = OmniCreation::new(Arc::new(ScriptedFetcher::new())).with_model(Arc::new(
        instructions_model("![hallucinated](references/img_99.png)"),
    ));
    let definition = creation
        .generate_with_model(&page, &downloaded)
        .await
        .expect("falls back to the deterministic generator");

    assert_eq!(definition, deterministic);
    definition.validate().expect("valid definition");
}

#[tokio::test]
async fn model_generation_without_images_has_no_image_references() {
    let page = scraped_main().await;
    let creation = OmniCreation::new(Arc::new(ScriptedFetcher::new()))
        .with_model(Arc::new(instructions_model("# Install\n\n1. Step.")));
    let definition = creation
        .generate_with_model(&page, &[])
        .await
        .expect("models the skill");
    let instructions = &definition.instructions;

    assert!(!instructions.contains("!["), "{instructions}");
    assert!(!instructions.contains("references/img_"), "{instructions}");
    assert!(
        !instructions.contains("## Reference Images"),
        "{instructions}"
    );
    assert!(
        instructions.contains("- `references/sources.md` — source page provenance"),
        "{instructions}"
    );
    assert_eq!(
        instructions.matches("## Reference Files").count(),
        1,
        "{instructions}"
    );
}
