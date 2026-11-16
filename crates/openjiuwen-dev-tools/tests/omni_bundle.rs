//! Cross-module integration suite for URL-to-skill creation.
//!
//! Drives the public `skill_creator::omni` pipeline through
//! [`ScriptedFetcher`]: one HTML page is scraped, converted into a deterministic
//! skill definition, written as a `SKILL.md` bundle, and read back with
//! [`SkillBundle::read`]. The suite also pins the fetcher contract (unknown URL,
//! non-HTTP scheme) and the archive policy (tar, zip, and gzip all unpack
//! in-process).
//!
//! The second half covers the rest of the public `skill_creator::omni` surface:
//! the [`GenerationProvenance`] of an [`OmniOutcome`] (deterministic template vs
//! drafting model, plus the legacy JSON that predates the field), the
//! [`UnpackOptions`] password path against a hand-built `ZipCrypto` archive
//! (missing, wrong, and correct password, and the formats that ignore it), the
//! `ArchiveEntryKind` of crafted `zip` and `tar` entries (symlinks, hard links,
//! PAX and GNU long names), the codecs the reader recognizes but refuses, and
//! [`image_dimensions`] composed with the decorative-image gate.
//!
//! Everything runs offline: the fetcher is scripted, paths come from a
//! [`TempWorkspace`], and no timestamp or environment value is read.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::skill_creator::omni::archive::ArchiveWriter;
use openjiuwen_dev_tools::skill_creator::omni::{
    detect, detect_named, image_dimensions, is_probable_icon, resolve_url, slugify, ArchiveEntry,
    ArchiveEntryKind, ArchiveFormat, ArchiveUnpacker, GenerationProvenance, HttpFetcher,
    OmniOutcome, ScriptedFetcher, UnpackOptions,
};
use openjiuwen_dev_tools::skill_creator::{SkillBundle, SkillCreator, SkillDefinition};
use openjiuwen_test_support::{ScriptedModel, TempWorkspace};

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
fn archives_are_detected_and_compressed_formats_unpack() {
    let zip = ArchiveWriter::write_stored(&[ArchiveEntry {
        path: "guide.md".to_owned(),
        bytes: b"# Guide\n".to_vec(),
        is_dir: false,
        ..ArchiveEntry::default()
    }])
    .expect("writes a zip");
    assert_eq!(detect(&zip), Some(ArchiveFormat::Zip));
    let entries = ArchiveUnpacker::unpack_auto(&zip).expect("zip unpacks");
    assert_eq!(entries[0].path, "guide.md");
    assert_eq!(entries[0].bytes, b"# Guide\n");

    let error = ArchiveUnpacker::unpack_auto(b"PK\x03\x04payload").expect_err("truncated zip");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error =
        ArchiveUnpacker::unpack_auto(&[0x1f, 0x8b, 0x08, 0x00]).expect_err("truncated gzip");
    assert_eq!(detect(&[0x1f, 0x8b, 0x08, 0x00]), Some(ArchiveFormat::Gzip));
    assert_eq!(error.kind(), ErrorKind::Validation);
    let unknown = ArchiveUnpacker::unpack_auto(b"not an archive at all").expect_err("unknown");
    assert_eq!(unknown.kind(), ErrorKind::Validation);
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
    tar_entry(name, size, typeflag, "")
}

/// Builds one 512-byte `ustar` header, optionally carrying a `linkname`.
fn tar_entry(name: &str, size: usize, typeflag: u8, linkname: &str) -> [u8; 512] {
    let mut header = [0_u8; 512];
    header[..name.len()].copy_from_slice(name.as_bytes());
    header[100..107].copy_from_slice(b"0000644");
    header[108..115].copy_from_slice(b"0000000");
    header[116..123].copy_from_slice(b"0000000");
    header[124..135].copy_from_slice(format!("{size:011o}").as_bytes());
    header[136..147].copy_from_slice(b"00000000000");
    header[156] = typeflag;
    header[157..157 + linkname.len()].copy_from_slice(linkname.as_bytes());
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    header[148..156].fill(b' ');
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
    header
}

// --- generation provenance -------------------------------------------------

/// The fenced JSON a drafting model answers with.
fn model_answer(name: &str) -> String {
    format!(
        "```json\n{}\n```",
        serde_json::json!({
            "name": name,
            "description": "Install the widget with confidence.",
            "instructions": "# Install\n\n1. Download the package."
        })
    )
}

#[tokio::test]
async fn an_outcome_reports_the_generator_that_produced_the_document() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let deterministic = SkillCreator::new()
        .create_from_url(fetcher(), SOURCE, &workspace.path("deterministic"))
        .await
        .expect("the deterministic run succeeds");
    assert_eq!(
        deterministic.provenance,
        GenerationProvenance::Deterministic
    );
    assert_eq!(deterministic.provenance.as_str(), "deterministic");
    assert_eq!(deterministic.definition.name, "install-the-widget");

    let modelled =
        SkillCreator::with_model(Arc::new(ScriptedModel::text(model_answer("model-widget"))))
            .create_from_url(fetcher(), SOURCE, &workspace.path("modelled"))
            .await
            .expect("the model run succeeds");
    assert_eq!(modelled.provenance, GenerationProvenance::Model);
    assert_eq!(modelled.provenance.as_str(), "model");
    assert_eq!(modelled.definition.name, "model-widget");
    assert!(modelled
        .definition
        .instructions
        .contains("1. Download the package."));
    assert_ne!(modelled.bundle_path, deterministic.bundle_path);

    // The wire token is snake_case, and an unknown key is tolerated.
    let mut payload = serde_json::to_value(&modelled).expect("the outcome serializes");
    assert_eq!(payload["provenance"], serde_json::json!("model"));
    payload["unknown"] = serde_json::json!("tolerated");

    // JSON written before the field existed deserializes to the serde default,
    // which is the deterministic generator.
    let mut legacy = payload.clone();
    legacy
        .as_object_mut()
        .expect("an object")
        .remove("provenance");
    let decoded: OmniOutcome = serde_json::from_value(legacy).expect("legacy JSON deserializes");
    assert_eq!(decoded.provenance, GenerationProvenance::Deterministic);
    assert_eq!(decoded.definition, modelled.definition);
    assert_eq!(decoded.resources, modelled.resources);
    assert_eq!(decoded.bundle_path, modelled.bundle_path);
}

// --- the ZipCrypto password option ----------------------------------------

/// The password of the encrypted fixture archive.
const PASSWORD: &str = "hunter2";

#[test]
fn an_encrypted_zip_entry_needs_the_password_option() {
    let plain: &[u8] = b"# Secret guide\n";
    let cipher = zipcrypto_encrypt(PASSWORD, plain, (crc32(plain) >> 24) as u8);
    let archive = build_zip(&[ZipSpec {
        name: "refs/secret.md",
        payload: &cipher,
        declared: u32_len(plain.len()),
        crc: crc32(plain),
        flags: FLAG_ENCRYPTED,
        external: 0,
    }]);
    assert_eq!(detect(&archive), Some(ArchiveFormat::Zip));

    // (a) No password: the answer names the entry and the option to set.
    let error = ArchiveUnpacker::unpack_auto(&archive)
        .expect_err("an encrypted entry cannot be read without its password");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("refs/secret.md"), "{message}");
    assert!(message.contains("with_password"), "{message}");

    let error = ArchiveUnpacker::unpack_auto_with_options(&archive, &UnpackOptions::new())
        .expect_err("the default options carry no password");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("refs/secret.md"), "{error}");

    // (b) A wrong password fails the entry's ZipCrypto check byte.
    let wrong = UnpackOptions::new().with_password("not-the-password");
    let error = ArchiveUnpacker::unpack_auto_with_options(&archive, &wrong)
        .expect_err("a wrong password does not decrypt");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("does not decrypt"), "{error}");

    // (c) The right password extracts the entry.
    let entries = ArchiveUnpacker::unpack_auto_with_options(
        &archive,
        &UnpackOptions::new().with_password(PASSWORD),
    )
    .expect("the correct password decrypts the entry");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, "refs/secret.md");
    assert_eq!(entries[0].bytes, plain);
    assert_eq!(entries[0].kind, ArchiveEntryKind::File);

    // An empty password is a validation error, never an attempt.
    let error = UnpackOptions::new()
        .with_password("")
        .validate()
        .expect_err("an empty password is a mistake");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("empty"), "{error}");

    // tar and gzip cannot declare a password, so they ignore it.
    let tar = sample_tar();
    let plain_tar = ArchiveUnpacker::unpack(ArchiveFormat::Tar, &tar).expect("tar unpacks");
    assert_eq!(plain_tar.len(), 2);
    let tar_with_password = ArchiveUnpacker::unpack_with_options(
        ArchiveFormat::Tar,
        &tar,
        &UnpackOptions::new().with_password(PASSWORD),
    )
    .expect("tar ignores the password");
    assert_eq!(tar_with_password, plain_tar);

    let gzip = gzip_member(&tar);
    assert_eq!(detect(&gzip), Some(ArchiveFormat::Gzip));
    let unpacked = ArchiveUnpacker::unpack_auto(&gzip).expect("gzip-wrapped tar unpacks");
    assert_eq!(unpacked, plain_tar);
    let gzip_with_password = ArchiveUnpacker::unpack_auto_with_options(
        &gzip,
        &UnpackOptions::new().with_password(PASSWORD),
    )
    .expect("gzip ignores the password");
    assert_eq!(gzip_with_password, plain_tar);
}

// --- entry kinds -----------------------------------------------------------

#[test]
fn zip_entries_report_their_unix_kind_and_link_target() {
    let target: &[u8] = b"guide.md";
    let contents: &[u8] = b"# Guide\n";
    let archive = build_zip(&[
        ZipSpec {
            name: "refs/latest.md",
            payload: target,
            declared: u32_len(target.len()),
            crc: crc32(target),
            flags: 0,
            external: UNIX_SYMLINK_MODE << 16,
        },
        ZipSpec {
            name: "guide.md",
            payload: contents,
            declared: u32_len(contents.len()),
            crc: crc32(contents),
            flags: 0,
            external: UNIX_FILE_MODE << 16,
        },
        ZipSpec {
            name: "refs/",
            payload: b"",
            declared: 0,
            crc: 0,
            flags: 0,
            external: (UNIX_DIRECTORY_MODE << 16) | 0x10,
        },
    ]);

    let entries = ArchiveUnpacker::unpack_auto(&archive).expect("the crafted zip unpacks");
    assert_eq!(entries.len(), 3);

    // The link entry carries its target as text, not as file contents.
    assert_eq!(entries[0].path, "refs/latest.md");
    assert_eq!(entries[0].kind, ArchiveEntryKind::Symlink);
    assert_eq!(entries[0].link_target.as_deref(), Some("guide.md"));
    assert_eq!(entries[0].bytes, Vec::<u8>::new());
    assert!(!entries[0].is_dir);
    assert_eq!(entries[0].mode, Some(UNIX_SYMLINK_MODE));

    // A regular entry stays a file, with its contents and no link target.
    assert_eq!(entries[1].path, "guide.md");
    assert_eq!(entries[1].kind, ArchiveEntryKind::File);
    assert_eq!(entries[1].link_target, None);
    assert_eq!(entries[1].bytes, contents);
    assert!(!entries[1].is_dir);
    assert_eq!(entries[1].mode, Some(UNIX_FILE_MODE));

    // The trailing slash plus the DOS attribute marks a directory.
    assert_eq!(entries[2].path, "refs/");
    assert_eq!(entries[2].kind, ArchiveEntryKind::Directory);
    assert!(entries[2].is_dir);
    assert_eq!(entries[2].bytes, Vec::<u8>::new());
    assert_eq!(entries[2].link_target, None);
}

#[test]
fn tar_extended_headers_and_link_entries_decode() {
    let mut archive = Vec::new();

    // A PAX per-file header whose `path` and `linkpath` win over the next header.
    let pax = [
        pax_record("path", "refs/pax-link.md"),
        pax_record("linkpath", "guide/real.md"),
    ]
    .concat();
    archive.extend_from_slice(&tar_entry("PaxHeaders/link", pax.len(), b'x', ""));
    push_block(&mut archive, pax.as_bytes());
    archive.extend_from_slice(&tar_entry("placeholder", 0, b'2', "ignored-target"));

    // GNU long name (`L`) and long linkname (`K`), both applying to the next
    // header only.
    let long_name = format!("docs/{}.md", "n".repeat(120));
    let long_link = format!("targets/{}.md", "t".repeat(120));
    archive.extend_from_slice(&tar_entry("././@LongLink", long_name.len() + 1, b'L', ""));
    push_block(&mut archive, format!("{long_name}\0").as_bytes());
    archive.extend_from_slice(&tar_entry("././@LongLink", long_link.len() + 1, b'K', ""));
    push_block(&mut archive, format!("{long_link}\0").as_bytes());
    archive.extend_from_slice(&tar_entry("short", 0, b'1', "short-target"));

    // Plain links: the header's own `linkname` field, with no extended header.
    archive.extend_from_slice(&tar_entry("refs/plain-link.md", 0, b'2', "guide.md"));
    archive.extend_from_slice(&tar_entry("refs/plain-hard.md", 0, b'1', "guide.md"));
    archive.extend_from_slice(&[0_u8; 1024]);

    let entries = ArchiveUnpacker::unpack_auto(&archive).expect("the crafted tar unpacks");
    assert_eq!(entries.len(), 4);

    assert_eq!(entries[0].path, "refs/pax-link.md");
    assert_eq!(entries[0].kind, ArchiveEntryKind::Symlink);
    assert_eq!(entries[0].link_target.as_deref(), Some("guide/real.md"));
    assert_eq!(entries[0].bytes, Vec::<u8>::new());
    assert!(!entries[0].is_dir);

    assert_eq!(entries[1].path, long_name);
    assert_eq!(entries[1].kind, ArchiveEntryKind::HardLink);
    assert_eq!(entries[1].link_target.as_deref(), Some(long_link.as_str()));
    assert_eq!(entries[1].bytes, Vec::<u8>::new());

    assert_eq!(entries[2].path, "refs/plain-link.md");
    assert_eq!(entries[2].kind, ArchiveEntryKind::Symlink);
    assert_eq!(entries[2].link_target.as_deref(), Some("guide.md"));
    assert!(!entries[2].is_dir);

    assert_eq!(entries[3].path, "refs/plain-hard.md");
    assert_eq!(entries[3].kind, ArchiveEntryKind::HardLink);
    assert_eq!(entries[3].link_target.as_deref(), Some("guide.md"));
}

// --- codecs the reader recognizes but refuses ------------------------------

#[test]
fn the_undecodable_codecs_are_detected_and_refused_by_name() {
    let codecs = [
        (ArchiveFormat::Bzip2, "bzip2", b"BZh9".as_slice()),
        (ArchiveFormat::Xz, "xz", b"\xFD7zXZ\x00".as_slice()),
        (
            ArchiveFormat::Zstd,
            "zstd",
            [0x28, 0xB5, 0x2F, 0xFD].as_slice(),
        ),
    ];
    for (format, name, magic) in codecs {
        assert_eq!(detect(magic), Some(format), "{name}");
        let error = ArchiveUnpacker::unpack(format, magic).expect_err(name);
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{name}");
        assert!(error.to_string().contains(name), "{name}: {error}");

        let error = ArchiveUnpacker::unpack_auto(magic).expect_err(name);
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{name}");
        assert!(error.to_string().contains(name), "{name}: {error}");
    }

    // Magic bytes win over the name, and the name is the fallback.
    assert_eq!(
        detect_named("skill.tar.bz2", b"BZh9"),
        Some(ArchiveFormat::Bzip2)
    );
    assert_eq!(
        detect_named("SKILL.tar.xz", b"plain text"),
        Some(ArchiveFormat::Xz)
    );
    assert_eq!(
        detect_named("skill.tar.zst", b"plain text"),
        Some(ArchiveFormat::Zstd)
    );
    assert_eq!(detect_named("skill.tzst", b""), Some(ArchiveFormat::Zstd));
    assert_eq!(detect_named("skill.tbz2", b""), Some(ArchiveFormat::Bzip2));
    assert_eq!(detect_named("skill.txz", b""), Some(ArchiveFormat::Xz));
    assert_eq!(detect_named("skill.tar", b"plain text"), None);
    assert_eq!(detect(b"BZh0"), None);
    assert_eq!(detect(&[0x28, 0xB5, 0x2F]), None);
}

// --- image dimensions ------------------------------------------------------

#[test]
fn image_dimensions_read_container_headers() {
    // PNG: signature, IHDR length/type, then big-endian width and height.
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&13_u32.to_be_bytes());
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&640_u32.to_be_bytes());
    png.extend_from_slice(&480_u32.to_be_bytes());
    png.extend_from_slice(&[8, 6, 0, 0, 0]);
    assert_eq!(image_dimensions(&png), Some((640, 480)));

    // GIF: the little-endian logical screen descriptor.
    let mut gif = b"GIF89a".to_vec();
    gif.extend_from_slice(&1200_u16.to_le_bytes());
    gif.extend_from_slice(&60_u16.to_le_bytes());
    assert_eq!(image_dimensions(&gif), Some((1200, 60)));

    // JPEG: SOI, a skipped APP0 segment, then SOF0 with height 480 / width 640.
    let mut jpeg = vec![0xFF, 0xD8];
    jpeg.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
    jpeg.extend_from_slice(&[0_u8; 14]);
    jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11]);
    jpeg.push(8);
    jpeg.extend_from_slice(&480_u16.to_be_bytes());
    jpeg.extend_from_slice(&640_u16.to_be_bytes());
    jpeg.push(3);
    jpeg.extend_from_slice(&[0_u8; 9]);
    assert_eq!(image_dimensions(&jpeg), Some((640, 480)));

    // Truncated, mislabelled, and unknown payloads have no dimensions.
    assert_eq!(image_dimensions(&png[..20]), None);
    assert_eq!(image_dimensions(b"\x89PNG\r\n\x1a\n"), None);
    assert_eq!(image_dimensions(b"GIF89a\x01"), None);
    assert_eq!(image_dimensions(b"\xFF\xD8"), None);
    assert_eq!(image_dimensions(b"RIFF\0\0\0\0AVI "), None);
    assert_eq!(image_dimensions(b"P7 640 480"), None);
    assert_eq!(image_dimensions(b"not an image at all"), None);
    assert_eq!(image_dimensions(&[]), None);

    // The decorative-image gate consumes exactly these dimensions: Python's
    // 80 px rule drops a 79x79 icon and keeps an 80x80 image.
    assert!(!is_probable_icon(
        "https://docs.example.com/img/banner.gif",
        Some(1200),
        Some(60)
    ));
    assert!(is_probable_icon(
        "https://docs.example.com/img/bullet.png",
        Some(79),
        Some(79)
    ));
    assert!(!is_probable_icon(
        "https://docs.example.com/img/logo.png",
        Some(80),
        Some(80)
    ));
    assert!(!is_probable_icon(
        "https://docs.example.com/img/photo.jpg",
        Some(640),
        Some(480)
    ));
    assert!(is_probable_icon(
        "https://docs.example.com/img/spacer.gif",
        None,
        None
    ));
}

// --- fixture helpers -------------------------------------------------------

/// `zip` general-purpose flag bit 0: the entry is encrypted.
const FLAG_ENCRYPTED: u16 = 0x0001;
/// Unix mode of a regular file (`0o100644`), shifted into the external
/// attributes' high half.
const UNIX_FILE_MODE: u32 = 0o100_644;
/// Unix mode of a directory (`0o040755`).
const UNIX_DIRECTORY_MODE: u32 = 0o040_755;
/// Unix mode of a symbolic link (`0o120777`).
const UNIX_SYMLINK_MODE: u32 = 0o120_777;

/// One entry of the hand-built `zip` containers below.
struct ZipSpec<'a> {
    /// Archive-relative entry name.
    name: &'a str,
    /// Bytes written after the local header: the plaintext, or the
    /// `ZipCrypto` ciphertext when the entry is encrypted.
    payload: &'a [u8],
    /// Uncompressed size declared by both headers.
    declared: u32,
    /// CRC-32 of the plaintext.
    crc: u32,
    /// General-purpose flags.
    flags: u16,
    /// External attributes.
    external: u32,
}

/// Narrows a fixture length to the 32-bit `zip` size/offset fields.
fn u32_len(value: usize) -> u32 {
    u32::try_from(value).expect("a fixture fits 32 bits")
}

/// Narrows a fixture length to the 16-bit `zip` name/count fields.
fn u16_len(value: usize) -> u16 {
    u16::try_from(value).expect("a fixture fits 16 bits")
}

/// Appends a little-endian `u16`.
fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Appends a little-endian `u32`.
fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Assembles stored local records, their central directory, and an end record.
///
/// The fixture is written from the `zip` byte layout rather than through
/// [`ArchiveWriter`], so the reader is checked against the format and the tests
/// can craft the encryption flag, the external attributes, and a declared size
/// that differs from the payload.
fn build_zip(entries: &[ZipSpec<'_>]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut offsets = Vec::with_capacity(entries.len());
    for entry in entries {
        offsets.push(u32_len(out.len()));
        out.extend_from_slice(b"PK\x03\x04");
        put_u16(&mut out, 20);
        put_u16(&mut out, entry.flags);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0x0021);
        put_u32(&mut out, entry.crc);
        put_u32(&mut out, u32_len(entry.payload.len()));
        put_u32(&mut out, entry.declared);
        put_u16(&mut out, u16_len(entry.name.len()));
        put_u16(&mut out, 0);
        out.extend_from_slice(entry.name.as_bytes());
        out.extend_from_slice(entry.payload);
    }
    let cd_offset = u32_len(out.len());
    for (entry, offset) in entries.iter().zip(&offsets) {
        out.extend_from_slice(b"PK\x01\x02");
        put_u16(&mut out, 20);
        put_u16(&mut out, 20);
        put_u16(&mut out, entry.flags);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0x0021);
        put_u32(&mut out, entry.crc);
        put_u32(&mut out, u32_len(entry.payload.len()));
        put_u32(&mut out, entry.declared);
        put_u16(&mut out, u16_len(entry.name.len()));
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u32(&mut out, entry.external);
        put_u32(&mut out, *offset);
        out.extend_from_slice(entry.name.as_bytes());
    }
    let cd_size = u32_len(out.len()) - cd_offset;
    out.extend_from_slice(b"PK\x05\x06");
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u16(&mut out, u16_len(entries.len()));
    put_u16(&mut out, u16_len(entries.len()));
    put_u32(&mut out, cd_size);
    put_u32(&mut out, cd_offset);
    put_u16(&mut out, 0);
    out
}

/// One reflected CRC-32 step over a non-inverted running state.
///
/// The in-crate CRC-32 and the `ZipCrypto` key schedule are private, so the
/// fixtures below carry their own eight-line copy of the same arithmetic.
fn crc32_step(state: u32, byte: u8) -> u32 {
    let mut entry = (state ^ u32::from(byte)) & 0xFF;
    for _ in 0..8 {
        entry = if entry & 1 == 1 {
            (entry >> 1) ^ 0xEDB8_8320
        } else {
            entry >> 1
        };
    }
    (state >> 8) ^ entry
}

/// The CRC-32 (IEEE) of `bytes`.
fn crc32(bytes: &[u8]) -> u32 {
    let mut state = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        state = crc32_step(state, byte);
    }
    !state
}

/// The three `ZipCrypto` keys, seeded from a password and updated per byte.
struct ZipCryptoKeys {
    /// The running CRC-32 of the plaintext.
    key0: u32,
    /// The linear congruential key.
    key1: u32,
    /// The running CRC-32 of `key1`'s high byte.
    key2: u32,
}

impl ZipCryptoKeys {
    /// Seeds the keys from `password`'s bytes.
    fn new(password: &str) -> Self {
        let mut keys = Self {
            key0: 0x1234_5678,
            key1: 0x2345_6789,
            key2: 0x3456_7890,
        };
        for &byte in password.as_bytes() {
            keys.update(byte);
        }
        keys
    }

    /// The next keystream byte.
    fn stream_byte(&self) -> u8 {
        let temp = (self.key2 | 2) & 0xFFFF;
        (temp.wrapping_mul(temp ^ 1) >> 8) as u8
    }

    /// Folds one plaintext byte into the key schedule.
    fn update(&mut self, plain: u8) {
        self.key0 = crc32_step(self.key0, plain);
        self.key1 = self
            .key1
            .wrapping_add(self.key0 & 0xFF)
            .wrapping_mul(134_775_813)
            .wrapping_add(1);
        self.key2 = crc32_step(self.key2, (self.key1 >> 24) as u8);
    }
}

/// Encrypts `plaintext` the way a traditional PKWARE (`ZipCrypto`) writer does.
///
/// The 11 random header bytes are a fixed pattern, so the fixture is
/// byte-for-byte reproducible; `check` is the header's password check byte, the
/// high byte of the plaintext's CRC-32 for an entry without a data descriptor.
fn zipcrypto_encrypt(password: &str, plaintext: &[u8], check: u8) -> Vec<u8> {
    /// The deterministic stand-in for the 11 random header bytes.
    const HEADER_PATTERN: [u8; 11] = [
        0x5a, 0x17, 0xc3, 0x40, 0x9e, 0x2b, 0x71, 0x8c, 0x36, 0xd4, 0x0f,
    ];
    let mut keys = ZipCryptoKeys::new(password);
    let mut out = Vec::with_capacity(12 + plaintext.len());
    for byte in HEADER_PATTERN.into_iter().chain([check]) {
        out.push(byte ^ keys.stream_byte());
        keys.update(byte);
    }
    for &byte in plaintext {
        out.push(byte ^ keys.stream_byte());
        keys.update(byte);
    }
    out
}

/// Wraps `payload` in one final stored DEFLATE block.
fn stored_deflate(payload: &[u8]) -> Vec<u8> {
    let len = u16::try_from(payload.len()).expect("a stored block holds at most 65535 bytes");
    let mut out = Vec::with_capacity(payload.len() + 5);
    out.push(0x01);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Wraps `payload` in one `gzip` member with a correct trailer.
fn gzip_member(payload: &[u8]) -> Vec<u8> {
    let isize = u32::try_from(payload.len()).expect("a fixture fits 32 bits");
    let mut out = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
    out.extend_from_slice(&stored_deflate(payload));
    out.extend_from_slice(&crc32(payload).to_le_bytes());
    out.extend_from_slice(&isize.to_le_bytes());
    out
}

/// Appends `data` to `archive`, padded to the next 512-byte boundary.
fn push_block(archive: &mut Vec<u8>, data: &[u8]) {
    archive.extend_from_slice(data);
    let padding = (512 - data.len() % 512) % 512;
    archive.resize(archive.len() + padding, 0);
}

/// One PAX record: `<len> <key>=<value>\n`, with the self-referential length.
fn pax_record(key: &str, value: &str) -> String {
    let body = format!(" {key}={value}\n");
    let mut len = body.len() + 1;
    loop {
        let next = len.to_string().len() + body.len();
        if next == len {
            return format!("{len}{body}");
        }
        len = next;
    }
}

/// A small `tar` archive with one directory and one file entry, both terminated
/// by the two mandatory zero blocks.
fn sample_tar() -> Vec<u8> {
    let mut archive = Vec::new();
    for (path, content, typeflag) in [
        ("docs/", b"".as_slice(), b'5'),
        ("docs/readme.txt", b"hello gzip\n".as_slice(), b'0'),
    ] {
        archive.extend_from_slice(&tar_entry(path, content.len(), typeflag, ""));
        push_block(&mut archive, content);
    }
    archive.extend_from_slice(&[0_u8; 1024]);
    archive
}
