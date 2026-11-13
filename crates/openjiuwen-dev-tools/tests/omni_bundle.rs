//! Cross-module integration suite for URL-to-skill creation.
//!
//! Drives the public `skill_creator::omni` pipeline through
//! [`ScriptedFetcher`]: one HTML page is scraped, converted into a deterministic
//! skill definition, written as a `SKILL.md` bundle, and read back with
//! [`SkillBundle::read`]. The suite also pins the fetcher contract (unknown URL,
//! non-HTTP scheme) and the archive policy (tar in-process, zip/gzip deferred).
//!
//! Everything runs offline: the fetcher is scripted, paths come from a
//! [`TempWorkspace`], and no timestamp or environment value is read.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::skill_creator::omni::{
    detect, resolve_url, slugify, ArchiveFormat, ArchiveUnpacker, HttpFetcher, ScriptedFetcher,
};
use openjiuwen_dev_tools::skill_creator::{SkillBundle, SkillCreator, SkillDefinition};
use openjiuwen_test_support::TempWorkspace;

/// The page every bundle test scrapes.
const SOURCE: &str = "https://docs.example.com/install";

/// A documentation page carrying every extractable field.
const PAGE: &str = r#"<!DOCTYPE html>
<html>
  <head>
    <title>Install the Widget</title>
    <meta name="description" content="Learn to install the widget quickly.">
  </head>
  <body>
    <h1>Install the Widget</h1>
    <p>Follow the steps below to install the widget on your machine.</p>
    <h2>Steps</h2>
    <pre><code>widget install --fast</code></pre>
    <a href="https://docs.example.com/guide/advanced">Advanced guide</a>
    <a href="/guide/faq">FAQ</a>
    <img src="/img/diagram.png" alt="architecture diagram">
  </body>
</html>"#;

/// A fetcher serving [`PAGE`] at [`SOURCE`].
fn fetcher() -> Arc<ScriptedFetcher> {
    let fetcher = Arc::new(ScriptedFetcher::new());
    fetcher.with_html(SOURCE, PAGE);
    fetcher
}

/// The SHA-independent identity fields a bundle round trip must preserve.
fn identity(definition: &SkillDefinition) -> (String, String, String) {
    (
        definition.name.clone(),
        definition.description.clone(),
        definition.instructions.clone(),
    )
}

#[tokio::test]
async fn a_page_becomes_a_validated_bundle() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let fetcher = fetcher();
    let outcome = SkillCreator::new()
        .create_from_url(fetcher.clone(), SOURCE, &workspace.path("skills"))
        .await
        .expect("the scripted page becomes a bundle");

    assert_eq!(outcome.definition.name, "install-the-widget");
    assert_eq!(
        outcome.definition.description,
        "Learn to install the widget quickly."
    );
    assert!(outcome
        .definition
        .instructions
        .contains("# Install the Widget"));
    assert!(outcome.definition.instructions.contains("## Code Examples"));
    assert!(outcome
        .definition
        .instructions
        .contains("widget install --fast"));
    outcome
        .definition
        .validate()
        .expect("the generated definition validates");

    let document = outcome.bundle_path.join("SKILL.md");
    assert!(document.is_file(), "SKILL.md must exist at {document:?}");
    let sources = outcome.bundle_path.join("references/sources.md");
    assert!(sources.is_file(), "the source list must be written");
    assert!(outcome
        .resources
        .iter()
        .any(|path| path == "references/sources.md"));

    let loaded = SkillBundle::read(&outcome.bundle_path)
        .await
        .expect("the written bundle reads back");
    loaded
        .validate()
        .expect("the reloaded definition validates");
    assert_eq!(identity(&loaded), identity(&outcome.definition));

    assert_eq!(fetcher.calls().first().map(String::as_str), Some(SOURCE));
}

#[tokio::test]
async fn the_same_page_produces_the_same_definition() {
    let first_workspace = TempWorkspace::new().expect("a temporary workspace");
    let second_workspace = TempWorkspace::new().expect("a temporary workspace");

    let first = SkillCreator::new()
        .create_from_url(fetcher(), SOURCE, &first_workspace.path("one"))
        .await
        .expect("the first run succeeds");
    let second = SkillCreator::new()
        .create_from_url(fetcher(), SOURCE, &second_workspace.path("two"))
        .await
        .expect("the second run succeeds");

    assert_eq!(identity(&first.definition), identity(&second.definition));
    assert_eq!(first.resources, second.resources);
    assert_eq!(
        first.bundle_path.file_name(),
        second.bundle_path.file_name(),
        "the skill directory name is derived from the title"
    );
}

#[tokio::test]
async fn transport_and_scheme_failures_keep_their_kinds() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");

    let missing = SkillCreator::new()
        .create_from_url(
            Arc::new(ScriptedFetcher::new()),
            "https://docs.example.com/missing",
            &workspace.path("missing"),
        )
        .await
        .expect_err("an unregistered URL is a transport failure");
    assert_eq!(missing.kind(), ErrorKind::Transport);
    assert!(missing
        .to_string()
        .contains("https://docs.example.com/missing"));

    let scheme = SkillCreator::new()
        .create_from_url(
            Arc::new(HttpFetcher::new()),
            "file:///tmp/widget.html",
            &workspace.path("scheme"),
        )
        .await
        .expect_err("only http(s) URLs are fetched");
    assert_eq!(scheme.kind(), ErrorKind::Validation);
    assert!(scheme.to_string().contains("http or https"));
}

#[test]
fn archives_are_detected_and_compressed_formats_are_deferred() {
    let zip = b"PK\x03\x04\x14\x00\x00\x00";
    assert_eq!(detect(zip), Some(ArchiveFormat::Zip));
    let error = ArchiveUnpacker::unpack_auto(zip).expect_err("zip is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("zip"), "got {message}");
    assert!(message.contains("compression dependency"), "got {message}");

    let gzip = [0x1f, 0x8b, 0x08, 0x00];
    assert_eq!(detect(&gzip), Some(ArchiveFormat::Gzip));
    let error = ArchiveUnpacker::unpack_auto(&gzip).expect_err("gzip is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("gzip"), "got {message}");
    assert!(message.contains("compression dependency"), "got {message}");

    let unknown = ArchiveUnpacker::unpack_auto(b"not an archive at all")
        .expect_err("unrecognized bytes are rejected");
    assert_eq!(unknown.kind(), ErrorKind::Validation);
    assert!(unknown.to_string().contains("archive"));
}

#[test]
fn a_hand_built_tar_unpacks_into_named_entries() {
    let content = b"hello tar\n";
    let mut archive = Vec::new();
    archive.extend_from_slice(&tar_header("docs/readme.txt", content.len(), b'0'));
    archive.extend_from_slice(content);
    archive.resize(archive.len() + (512 - content.len() % 512) % 512, 0);
    archive.extend_from_slice(&[0_u8; 1024]);

    assert_eq!(detect(&archive), Some(ArchiveFormat::Tar));
    let entries = ArchiveUnpacker::unpack_auto(&archive).expect("ustar unpacks in-process");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "docs/readme.txt");
    assert_eq!(entries[0].bytes, content);
    assert!(!entries[0].is_dir);
}

#[test]
fn the_public_omni_helpers_normalize_text_and_links() {
    assert_eq!(slugify("  Install the Widget!! "), "install-the-widget");
    assert_eq!(slugify("install_the_widget"), "install-the-widget");
    assert_eq!(slugify("!!!"), "");

    assert_eq!(
        resolve_url("https://docs.example.com/guide/", "/install"),
        Some("https://docs.example.com/install".to_string())
    );
    assert_eq!(
        resolve_url("https://docs.example.com/guide/", "advanced"),
        Some("https://docs.example.com/guide/advanced".to_string())
    );
    assert_eq!(resolve_url("https://docs.example.com/guide/", "#top"), None);
    assert_eq!(
        resolve_url("https://docs.example.com/guide/", "mailto:docs@example.com"),
        None
    );
    assert_eq!(resolve_url("not a url", "/install"), None);
}

/// Builds one 512-byte `ustar` header with a valid checksum.
fn tar_header(name: &str, size: usize, typeflag: u8) -> [u8; 512] {
    let mut header = [0_u8; 512];
    header[..name.len()].copy_from_slice(name.as_bytes());
    header[100..107].copy_from_slice(b"0000644");
    header[108..115].copy_from_slice(b"0000000");
    header[116..123].copy_from_slice(b"0000000");
    header[124..135].copy_from_slice(format!("{size:011o}").as_bytes());
    header[136..147].copy_from_slice(b"00000000000");
    header[156] = typeflag;
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    header[148..156].fill(b' ');
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
    header
}
