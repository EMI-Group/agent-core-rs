//! Archive sniffing, packing, and unpacking for the [`super::pipeline`] scrape stage.
//!
//! The `skill_omni_creation` pipeline accepts a URL that may resolve to an
//! archive rather than an HTML page, so [`detect`] identifies the container from
//! its magic bytes and [`ArchiveUnpacker`] turns it into ordered
//! [`ArchiveEntry`] values. [`ArchiveWriter`] produces deterministic stored
//! `zip` bytes for tests and for callers that need to round-trip entries.
//!
//! Every decodable format is parsed in-process with no compression dependency:
//!
//! - `tar` (POSIX `ustar` and GNU flavors) in `tar`;
//! - `zip` (stored and DEFLATE, plus a stored writer) in `zip`;
//! - `gzip` (RFC 1952, including gzip-wrapped `tar`/`zip`) in `gzip`;
//! - DEFLATE (RFC 1951) and CRC-32 (IEEE) live in the shared `inflate` and
//!   `crc` modules.
//!
//! `bzip2`, `xz`, and `zstd` are recognized by [`detect`]/[`detect_named`] so a
//! caller can name the container it was handed, but no bzip2, xz, or zstd crate
//! is available offline, so [`ArchiveUnpacker::unpack`] answers them with
//! [`Error::unsupported`] naming the codec instead of guessing.
//!
//! `zip` is the only container here that can carry an encrypted entry.
//! Traditional PKWARE encryption (`ZipCrypto`, general-purpose flag bit 0) is
//! decrypted when the caller supplies a password through [`UnpackOptions`], the
//! opt-in companion of [`ArchiveUnpacker::unpack_with_options`] and
//! [`ArchiveUnpacker::unpack_auto_with_options`]; `unpack`/`unpack_auto` stay the
//! no-password shorthand, so an encrypted entry read through them is a
//! [`Error::validation`] naming the entry and the option to set. The three
//! schemes this reader does not implement stay typed [`Error::unsupported`]:
//! `WinZip` AES (compression method `99` or the `0x9901` extra field), PKWARE
//! strong encryption (flag bit 6), and an encrypted central directory (flag bit
//! 13). A password is ignored for `tar` and `gzip`, which cannot declare one,
//! and it only reaches a top-level `zip`: a `zip` nested inside a `gzip` wrapper
//! is read by the `gzip` decoder's own no-password path.
//!
//! Every entry path is validated before it is returned: absolute paths, `..`
//! components, and backslashes are rejected, so a caller can never write outside
//! the bundle root. Unpacking is bounded by [`MAX_ENTRY_UNCOMPRESSED_BYTES`] per
//! entry and [`MAX_ARCHIVE_UNCOMPRESSED_BYTES`] per archive, so a hostile archive
//! fails with a validation error instead of exhausting memory.
mod crc;
mod gzip;
mod inflate;
mod tar;
mod zip;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Largest number of bytes a single unpacked entry may declare: 8 MiB.
///
/// This matches the default [`HttpFetcher`](super::fetch::HttpFetcher) body cap,
/// so an archive entry can never be larger than the response that carried it.
pub const MAX_ENTRY_UNCOMPRESSED_BYTES: usize = 8 * 1024 * 1024;

/// Largest number of bytes every entry of one archive may declare in total: 32 MiB.
pub const MAX_ARCHIVE_UNCOMPRESSED_BYTES: usize = 32 * 1024 * 1024;

/// A container format recognized by [`detect`] or [`detect_named`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveFormat {
    /// A `tar` archive, including the POSIX `ustar` flavor.
    Tar,
    /// A `zip` archive.
    Zip,
    /// A `gzip`-compressed stream.
    Gzip,
    /// A `bzip2`-compressed stream, recognized but not decodable here.
    Bzip2,
    /// An `xz`/LZMA2-compressed stream, recognized but not decodable here.
    Xz,
    /// A `zstd` frame, recognized but not decodable here.
    Zstd,
}

impl ArchiveFormat {
    /// The lowercase wire name of the format: `"tar"`, `"zip"`, `"gzip"`,
    /// `"bzip2"`, `"xz"`, or `"zstd"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tar => "tar",
            Self::Zip => "zip",
            Self::Gzip => "gzip",
            Self::Bzip2 => "bzip2",
            Self::Xz => "xz",
            Self::Zstd => "zstd",
        }
    }
}

impl fmt::Display for ArchiveFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What an [`ArchiveEntry`] describes.
///
/// [`ArchiveEntryKind::File`] is the default, so a decoded entry that carries
/// contents is unambiguous and an older payload without a `kind` key still
/// deserializes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveEntryKind {
    /// A regular file entry; `bytes` holds its contents.
    #[default]
    File,
    /// A directory entry; `is_dir` is `true` and `bytes` is empty.
    Directory,
    /// A symbolic link entry; its target is in `link_target` and `bytes` is empty.
    Symlink,
    /// A hard link entry; its target is in `link_target` and `bytes` is empty.
    HardLink,
}

/// One entry read out of an archive.
///
/// `path` is the archive-relative path with `/` separators; `bytes` holds the
/// file contents and is empty for directories and links; `is_dir` marks
/// directory entries and stays `true` exactly when `kind` is
/// [`ArchiveEntryKind::Directory`]; `kind` names the entry shape; a symlink or
/// hardlink entry carries its target in `link_target` with empty `bytes`; and
/// `mode` is the permission bits the container declared, when it declares any.
///
/// `kind`, `link_target`, and `mode` all default, so a payload written before
/// they existed still deserializes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ArchiveEntry {
    /// Archive-relative path of the entry.
    pub path: String,
    /// File contents; empty for directory, symlink, and hardlink entries.
    pub bytes: Vec<u8>,
    /// Whether the entry is a directory; `true` exactly when `kind` is
    /// [`ArchiveEntryKind::Directory`].
    pub is_dir: bool,
    /// The entry shape: file, directory, symlink, or hardlink.
    pub kind: ArchiveEntryKind,
    /// Target path of a symlink or hardlink entry, `None` for every other kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_target: Option<String>,
    /// Permission bits declared by the container, `None` when it declares none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<u32>,
}

/// Options for unpacking an archive that needs more than its bytes.
///
/// Today the only option is the `ZipCrypto` password: a `zip` entry whose
/// general-purpose flag bit 0 is set can only be read when its password is
/// supplied here. Every format that cannot be encrypted (`tar`) or that declares
/// no encryption (`gzip`) ignores the password, and an empty password is a
/// validation error rather than a silent attempt, because the scheme's empty
/// password is its identity.
///
/// The default value carries no password, exactly like
/// [`ArchiveUnpacker::unpack`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct UnpackOptions {
    /// The password of a traditional PKWARE (`ZipCrypto`) encrypted `zip` entry.
    ///
    /// Its UTF-8 bytes seed the entry's keys, which is exact for the ASCII
    /// passwords every `zip` writer accepts; a password an archiver interpreted
    /// in another code page cannot be reproduced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
}

impl UnpackOptions {
    /// Options that carry no password.
    #[must_use]
    pub const fn new() -> Self {
        Self { password: None }
    }

    /// The same options with `password` set for `ZipCrypto` entries.
    #[must_use]
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.password = Some(password.into());
        self
    }

    /// Rejects an empty password.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `password` is `Some("")`: an empty
    /// password leaves the `ZipCrypto` keys at their initial values, so it can
    /// only ever be a mistake.
    pub fn validate(&self) -> Result<()> {
        if self.password.as_deref() == Some("") {
            return Err(Error::validation(
                "invalid unpack options: the password is empty, and a ZipCrypto entry needs its \
                 non-empty password",
            ));
        }
        Ok(())
    }
}

/// Associated functions for sniffing and unpacking archives.
///
/// This type is never instantiated: it only groups the entry points so call
/// sites read as [`ArchiveUnpacker::unpack`] / [`ArchiveUnpacker::unpack_auto`],
/// matching the associated-function style of `SkillBundle`. Passwords travel
/// through the `*_with_options` variants.
#[derive(Debug, Clone, Copy)]
pub struct ArchiveUnpacker;

impl ArchiveUnpacker {
    /// Unpacks `bytes` as `format`, returning entries in archive order.
    ///
    /// `tar`, `zip`, and `gzip` are decoded in-process. `bzip2`, `xz`, and
    /// `zstd` are recognized containers this reader cannot decode.
    ///
    /// This is the no-password shorthand for
    /// [`unpack_with_options`](Self::unpack_with_options).
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for malformed, truncated, unsafe, or
    /// oversized content and for an encrypted `zip` entry, which needs a
    /// password, and [`Error::unsupported`] for capabilities this in-process
    /// implementation deliberately does not provide (`WinZip` AES, PKWARE strong
    /// encryption and encrypted central directories, compression methods other
    /// than stored and DEFLATE, and the `bzip2`, `xz`, and `zstd` codecs, which
    /// no bundled decoder handles).
    pub fn unpack(format: ArchiveFormat, bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
        Self::unpack_with_options(format, bytes, &UnpackOptions::new())
    }

    /// Unpacks `bytes` as `format` with `options`, returning entries in archive
    /// order.
    ///
    /// A `zip` entry protected by traditional PKWARE encryption (`ZipCrypto`) is
    /// decrypted with [`UnpackOptions::password`]; every other format ignores it,
    /// because `tar` cannot encrypt an entry and `gzip` declares no encryption.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for unusable options (an empty password),
    /// malformed, truncated, unsafe, or oversized content, and an encrypted entry
    /// with no password or a password that fails the entry's check byte; and
    /// [`Error::unsupported`] for `WinZip` AES, PKWARE strong encryption, an
    /// encrypted central directory, compression methods other than stored and
    /// DEFLATE, and the `bzip2`, `xz`, and `zstd` codecs.
    pub fn unpack_with_options(
        format: ArchiveFormat,
        bytes: &[u8],
        options: &UnpackOptions,
    ) -> Result<Vec<ArchiveEntry>> {
        options.validate()?;
        match format {
            ArchiveFormat::Tar => tar::unpack(bytes),
            ArchiveFormat::Zip => match options.password.as_deref() {
                Some(password) => zip::unpack_with_password(bytes, Some(password)),
                None => zip::unpack(bytes),
            },
            ArchiveFormat::Gzip => gzip::unpack(bytes),
            ArchiveFormat::Bzip2 => Err(decoder_unavailable("bzip2")),
            ArchiveFormat::Xz => Err(decoder_unavailable("xz")),
            ArchiveFormat::Zstd => Err(decoder_unavailable("zstd")),
        }
    }

    /// Sniffs the format with [`detect`] and then [`unpack`](Self::unpack)s it.
    ///
    /// This is the no-password shorthand for
    /// [`unpack_auto_with_options`](Self::unpack_auto_with_options).
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the bytes carry no recognized archive
    /// signature, plus everything [`unpack`](Self::unpack) can return — in
    /// particular [`Error::unsupported`] naming `bzip2`, `xz`, or `zstd`.
    pub fn unpack_auto(bytes: &[u8]) -> Result<Vec<ArchiveEntry>> {
        let format = detect(bytes).ok_or_else(|| undetected(bytes.len()))?;
        Self::unpack(format, bytes)
    }

    /// Sniffs the format with [`detect`] and then
    /// [`unpack_with_options`](Self::unpack_with_options) it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the bytes carry no recognized archive
    /// signature, plus everything
    /// [`unpack_with_options`](Self::unpack_with_options) can return.
    pub fn unpack_auto_with_options(
        bytes: &[u8],
        options: &UnpackOptions,
    ) -> Result<Vec<ArchiveEntry>> {
        let format = detect(bytes).ok_or_else(|| undetected(bytes.len()))?;
        Self::unpack_with_options(format, bytes, options)
    }
}

/// The [`Error::validation`] answer for bytes with no archive signature.
fn undetected(len: usize) -> Error {
    Error::validation(format!(
        "unrecognized archive: no gzip, zip, ustar, bzip2, xz, or zstd signature in {len} bytes"
    ))
}

/// The [`Error::unsupported`] answer for a container this reader cannot decode.
///
/// No `bzip2`, `xz`, or `zstd` crate is available in this offline dependency
/// set, so the message names the codec in the user's terms.
fn decoder_unavailable(codec: &str) -> Error {
    Error::unsupported(format!(
        "{codec} archives are not supported: no {codec} decoder is bundled with this offline \
         in-process reader"
    ))
}

/// Sniffs the archive format of `bytes`, or `None` when nothing matches.
///
/// Recognized signatures are `gzip` (`1f 8b`), `zip` (`PK\x03\x04` and the empty
/// archive `PK\x05\x06`), `tar` (`ustar` at byte offset 257), `bzip2` (`BZh`
/// followed by the block-size digit `1`–`9`), `xz` (`fd 37 7a 58 5a 00`), and
/// `zstd` (`28 b5 2f fd`).
#[must_use]
pub fn detect(bytes: &[u8]) -> Option<ArchiveFormat> {
    if bytes.starts_with(&[0x1f, 0x8b]) {
        Some(ArchiveFormat::Gzip)
    } else if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
        Some(ArchiveFormat::Zip)
    } else if bytes.get(257..262) == Some(b"ustar".as_slice()) {
        Some(ArchiveFormat::Tar)
    } else if bytes.starts_with(b"BZh") && matches!(bytes.get(3).copied(), Some(b'1'..=b'9')) {
        Some(ArchiveFormat::Bzip2)
    } else if bytes.starts_with(b"\xFD7zXZ\x00") {
        Some(ArchiveFormat::Xz)
    } else if bytes.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        Some(ArchiveFormat::Zstd)
    } else {
        None
    }
}

/// Sniffs `bytes` with [`detect`], falling back to the filename `name`.
///
/// Magic bytes win, so a mislabelled file is classified by its content. When the
/// content carries no recognized signature, the lowercase filename suffix
/// decides: `.tar.bz2`/`.tbz2`/`.tbz` → [`ArchiveFormat::Bzip2`],
/// `.tar.xz`/`.txz` → [`ArchiveFormat::Xz`], `.tar.zst`/`.tzst`/`.zstd` →
/// [`ArchiveFormat::Zstd`], and anything else → `None`.
#[must_use]
pub fn detect_named(name: &str, bytes: &[u8]) -> Option<ArchiveFormat> {
    if let Some(format) = detect(bytes) {
        return Some(format);
    }
    // The name is lowercased first, so a suffix table keeps the comparison
    // case-insensitive without a per-suffix literal.
    let lower = name.to_ascii_lowercase();
    let ends_with_any = |suffixes: &[&str]| suffixes.iter().any(|suffix| lower.ends_with(suffix));
    if ends_with_any(&[".tar.bz2", ".tbz2", ".tbz"]) {
        Some(ArchiveFormat::Bzip2)
    } else if ends_with_any(&[".tar.xz", ".txz"]) {
        Some(ArchiveFormat::Xz)
    } else if ends_with_any(&[".tar.zst", ".tzst", ".zstd"]) {
        Some(ArchiveFormat::Zstd)
    } else {
        None
    }
}

pub use zip::ArchiveWriter;

/// Rejects paths a caller could use to escape the bundle root.
///
/// The rules are shared by every format: a path must be non-empty, relative
/// (no leading `/` and no Windows drive prefix), free of `..` components and
/// backslashes, and `/`-separated.
pub(crate) fn validate_path(path: &str) -> Result<String> {
    if path.is_empty() {
        return Err(Error::validation("unsafe archive path: empty entry name"));
    }
    if path.starts_with('/') || is_windows_absolute(path) {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: absolute paths are not allowed"
        )));
    }
    if path.split('/').any(|component| component == "..") {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: `..` components are not allowed"
        )));
    }
    if path.contains('\\') {
        return Err(Error::validation(format!(
            "unsafe archive path {path:?}: backslashes are not allowed"
        )));
    }
    Ok(path.to_string())
}

/// Whether `path` starts with a Windows drive prefix such as `C:`.
fn is_windows_absolute(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Bounds-checked slice of `bytes`, naming `offset` in the error.
pub(crate) fn slice_at(bytes: &[u8], offset: usize, len: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::validation(format!("byte offset overflow at {offset} + {len}")))?;
    bytes.get(offset..end).ok_or_else(|| {
        Error::validation(format!(
            "truncated input: {len} bytes needed at offset {offset}, but the input is {} bytes",
            bytes.len()
        ))
    })
}

/// Reads a little-endian `u16` at `offset`.
pub(crate) fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16> {
    let raw = slice_at(bytes, offset, 2)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

/// Reads a little-endian `u32` at `offset`.
pub(crate) fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = slice_at(bytes, offset, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

/// Enforces the per-entry and aggregate uncompressed-size caps.
///
/// `tar` and `zip` declare entry sizes in their headers; charging a declared
/// size before allocating keeps a hostile archive from exhausting memory.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CapTracker {
    total: usize,
}

impl CapTracker {
    /// A tracker that has charged nothing yet.
    pub(crate) const fn new() -> Self {
        Self { total: 0 }
    }

    /// Charges `len` bytes to the entry and aggregate caps.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] naming `label` when `len` exceeds
    /// [`MAX_ENTRY_UNCOMPRESSED_BYTES`] or the running total would exceed
    /// [`MAX_ARCHIVE_UNCOMPRESSED_BYTES`].
    pub(crate) fn charge(&mut self, len: usize, label: &str) -> Result<()> {
        if len > MAX_ENTRY_UNCOMPRESSED_BYTES {
            return Err(Error::validation(format!(
                "{label} declares {len} bytes, above the per-entry cap of \
                 {MAX_ENTRY_UNCOMPRESSED_BYTES} bytes"
            )));
        }
        let total = self
            .total
            .checked_add(len)
            .ok_or_else(|| Error::validation(format!("{label}: the archive size overflows")))?;
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES {
            return Err(Error::validation(format!(
                "{label}: the archive expands past the aggregate cap of \
                 {MAX_ARCHIVE_UNCOMPRESSED_BYTES} bytes"
            )));
        }
        self.total = total;
        Ok(())
    }

    /// Bytes charged so far, for tests that assert the running total.
    #[cfg(test)]
    pub(crate) const fn total(self) -> usize {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_recognizes_known_signatures() {
        assert_eq!(detect(&[0x1f, 0x8b, 0x08, 0x00]), Some(ArchiveFormat::Gzip));
        assert_eq!(detect(b"PK\x03\x04rest"), Some(ArchiveFormat::Zip));
        assert_eq!(detect(b"PK\x05\x06"), Some(ArchiveFormat::Zip));
        assert_eq!(detect(b"plain text, not an archive"), None);
        assert_eq!(ArchiveFormat::Gzip.to_string(), "gzip");
        assert_eq!(ArchiveFormat::Tar.as_str(), "tar");
    }

    /// A minimal `bzip2` stream prefix: the signature plus the level digit.
    const BZIP2_MAGIC: &[u8] = b"BZh9";
    /// The `xz` stream magic.
    const XZ_MAGIC: &[u8] = b"\xFD7zXZ\x00";
    /// The `zstd` frame magic.
    const ZSTD_MAGIC: &[u8] = &[0x28, 0xB5, 0x2F, 0xFD];

    #[test]
    fn detect_recognizes_the_undecodable_codec_signatures() {
        assert_eq!(detect(BZIP2_MAGIC), Some(ArchiveFormat::Bzip2));
        assert_eq!(
            detect(b"BZh1trailing payload"),
            Some(ArchiveFormat::Bzip2),
            "every block-size digit 1-9 is a valid bzip2 signature"
        );
        assert_eq!(detect(XZ_MAGIC), Some(ArchiveFormat::Xz));
        assert_eq!(detect(ZSTD_MAGIC), Some(ArchiveFormat::Zstd));

        // The bzip2 signature needs the level digit right after `BZh`.
        assert_eq!(detect(b"BZh"), None);
        assert_eq!(detect(b"BZh0"), None);
        assert_eq!(detect(b"BZ"), None);
        // The magics stay exact: no prefix or suffix match.
        assert_eq!(detect(b"\xFD7zXZ"), None);
        assert_eq!(detect(&[0x28, 0xB5, 0x2F]), None);
    }

    #[test]
    fn the_undecodable_formats_report_their_wire_names() {
        for (format, name) in [
            (ArchiveFormat::Bzip2, "bzip2"),
            (ArchiveFormat::Xz, "xz"),
            (ArchiveFormat::Zstd, "zstd"),
        ] {
            assert_eq!(format.as_str(), name);
            assert_eq!(format.to_string(), name);
        }
    }

    #[test]
    fn detect_named_prefers_magic_bytes_then_the_suffix() {
        // Magic bytes win, so the name cannot mislead the classifier.
        assert_eq!(detect_named("notes.txt", XZ_MAGIC), Some(ArchiveFormat::Xz));
        assert_eq!(
            detect_named("bundle.zip", BZIP2_MAGIC),
            Some(ArchiveFormat::Bzip2)
        );

        // With no magic, the lowercase suffix decides, case-insensitively.
        for name in [
            "skill.tar.bz2",
            "skill.tbz2",
            "skill.tbz",
            "SKILL.TAR.BZ2",
            "Skill.TBZ2",
        ] {
            assert_eq!(
                detect_named(name, b""),
                Some(ArchiveFormat::Bzip2),
                "{name}"
            );
        }
        for name in ["skill.tar.xz", "skill.txz", "SKILL.TXZ"] {
            assert_eq!(detect_named(name, b""), Some(ArchiveFormat::Xz), "{name}");
        }
        for name in ["skill.tar.zst", "skill.tzst", "skill.zstd", "SKILL.TAR.ZST"] {
            assert_eq!(detect_named(name, b""), Some(ArchiveFormat::Zstd), "{name}");
        }

        // Plain text under an unknown or unsupported suffix stays unrecognized.
        for name in ["skill.tar", "skill.zip", "skill.html", "", "bz2"] {
            assert_eq!(detect_named(name, b"plain text"), None, "{name}");
        }
    }

    #[test]
    fn the_undecodable_codecs_report_a_typed_unsupported_error() {
        for (format, codec, magic) in [
            (ArchiveFormat::Bzip2, "bzip2", BZIP2_MAGIC),
            (ArchiveFormat::Xz, "xz", XZ_MAGIC),
            (ArchiveFormat::Zstd, "zstd", ZSTD_MAGIC),
        ] {
            let error = ArchiveUnpacker::unpack(format, magic).expect_err(codec);
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Unsupported,
                "{codec}"
            );
            let message = error.to_string();
            assert!(message.contains(codec), "{codec}: {message}");
            assert!(message.contains("decoder"), "{codec}: {message}");

            // Detection plus unpacking reaches the same typed answer.
            let error = ArchiveUnpacker::unpack_auto(magic).expect_err(codec);
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Unsupported,
                "{codec}"
            );
            assert!(error.to_string().contains(codec), "{codec}: {error}");
        }
    }

    #[test]
    fn entries_carry_kind_link_target_and_mode_through_json() {
        let entry = ArchiveEntry {
            path: "refs/latest.md".to_owned(),
            bytes: Vec::new(),
            is_dir: false,
            kind: ArchiveEntryKind::Symlink,
            link_target: Some("guide.md".to_owned()),
            mode: Some(0o644),
        };
        let json = serde_json::to_value(&entry).expect("serializes");
        assert_eq!(json["kind"], "symlink");
        assert_eq!(json["link_target"], "guide.md");
        assert_eq!(json["mode"], 0o644);
        assert_eq!(json["is_dir"], false);

        let decoded: ArchiveEntry = serde_json::from_value(json).expect("deserializes");
        assert_eq!(decoded, entry);

        // Absent optional fields stay out of the payload rather than becoming
        // `null`, so an unchanged entry serializes exactly as it used to.
        let file = ArchiveEntry {
            path: "a.txt".to_owned(),
            bytes: b"a".to_vec(),
            is_dir: false,
            ..ArchiveEntry::default()
        };
        let json = serde_json::to_value(&file).expect("serializes");
        assert!(json.get("link_target").is_none(), "{json}");
        assert!(json.get("mode").is_none(), "{json}");
        assert_eq!(json["kind"], "file");

        // Every kind round-trips through the snake_case names.
        for (kind, name) in [
            (ArchiveEntryKind::File, "file"),
            (ArchiveEntryKind::Directory, "directory"),
            (ArchiveEntryKind::Symlink, "symlink"),
            (ArchiveEntryKind::HardLink, "hard_link"),
        ] {
            let entry = ArchiveEntry {
                kind,
                ..ArchiveEntry::default()
            };
            let json = serde_json::to_value(&entry).expect("serializes");
            assert_eq!(json["kind"], name);
            let decoded: ArchiveEntry = serde_json::from_value(json).expect("deserializes");
            assert_eq!(decoded.kind, kind);
        }
    }

    #[test]
    fn entries_without_the_new_fields_deserialize_with_defaults() {
        let legacy = serde_json::json!({"path": "a.txt", "bytes": [], "is_dir": false});
        let entry: ArchiveEntry = serde_json::from_value(legacy).expect("legacy payload");
        assert_eq!(entry.path, "a.txt");
        assert_eq!(entry.bytes, Vec::<u8>::new());
        assert!(!entry.is_dir);
        assert_eq!(entry.kind, ArchiveEntryKind::File);
        assert_eq!(entry.link_target, None);
        assert_eq!(entry.mode, None);
    }

    #[test]
    fn validate_path_accepts_relative_paths_and_rejects_escapes() {
        assert_eq!(validate_path("a/b.txt").expect("relative path"), "a/b.txt");
        for path in ["", "/etc/passwd", "C:evil", "../evil", "a/../b", "a\\b"] {
            let error = validate_path(path).expect_err(path);
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
        }
    }

    #[test]
    fn little_endian_readers_are_bounds_checked() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09];
        assert_eq!(read_u16_le(&bytes, 0).expect("u16"), 0x0201);
        assert_eq!(read_u32_le(&bytes, 0).expect("u32"), 0x0403_0201);
        let error = read_u32_le(&bytes, 7).expect_err("truncated");
        assert!(error.to_string().contains("offset 7"), "{error}");
        let error = slice_at(&bytes, usize::MAX, 2).expect_err("overflow");
        assert!(error.to_string().contains("overflow"), "{error}");
    }

    #[test]
    fn cap_tracker_enforces_per_entry_and_aggregate_bounds() {
        let mut caps = CapTracker::new();
        // Four full entries exactly exhaust the 32 MiB aggregate cap.
        for _ in 0..4 {
            caps.charge(MAX_ENTRY_UNCOMPRESSED_BYTES, "entry")
                .expect("a full entry fits");
        }
        assert_eq!(caps.total(), MAX_ARCHIVE_UNCOMPRESSED_BYTES);
        let error = caps
            .charge(MAX_ENTRY_UNCOMPRESSED_BYTES, "entry")
            .expect_err("aggregate cap");
        assert!(error.to_string().contains("aggregate"), "{error}");
        assert_eq!(caps.total(), MAX_ARCHIVE_UNCOMPRESSED_BYTES);

        let error = CapTracker::new()
            .charge(MAX_ENTRY_UNCOMPRESSED_BYTES + 1, "entry")
            .expect_err("per-entry cap");
        assert!(error.to_string().contains("per-entry"), "{error}");
    }

    #[test]
    fn unpack_options_default_to_no_password_and_round_trip_through_json() {
        let empty = UnpackOptions::default();
        assert_eq!(empty, UnpackOptions::new());
        assert_eq!(empty.password, None);
        // An absent password stays out of the payload rather than becoming `null`.
        assert_eq!(
            serde_json::to_value(&empty).expect("serializes"),
            serde_json::json!({})
        );

        let options = UnpackOptions::new().with_password("hunter2");
        let json = serde_json::to_value(&options).expect("serializes");
        assert_eq!(json, serde_json::json!({"password": "hunter2"}));
        let decoded: UnpackOptions = serde_json::from_value(json).expect("deserializes");
        assert_eq!(decoded, options);

        // A legacy payload without the key, and unknown keys, both deserialize.
        let legacy: UnpackOptions =
            serde_json::from_value(serde_json::json!({})).expect("no password");
        assert_eq!(legacy, empty);
        let tolerant: UnpackOptions = serde_json::from_value(serde_json::json!({
            "password": "secret",
            "unknown": 1,
        }))
        .expect("unknown fields are tolerated");
        assert_eq!(tolerant.password.as_deref(), Some("secret"));
    }

    #[test]
    fn unpack_options_reject_an_empty_password() {
        UnpackOptions::new()
            .validate()
            .expect("no password is usable");
        UnpackOptions::new()
            .with_password("hunter2")
            .validate()
            .expect("a password is usable");
        let error = UnpackOptions::new()
            .with_password("")
            .validate()
            .expect_err("an empty password is a mistake");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("empty"), "{error}");

        // The dispatcher validates before it looks at a single byte, so an empty
        // password is the same typed answer for every format.
        let error = ArchiveUnpacker::unpack_with_options(
            ArchiveFormat::Tar,
            b"not even a tar",
            &UnpackOptions::new().with_password(""),
        )
        .expect_err("empty password");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("empty"), "{error}");
    }

    #[test]
    fn a_password_is_ignored_by_the_formats_that_cannot_be_encrypted() {
        // A `tar` payload carrying no entry at all: two zero blocks. The password
        // cannot make it any less readable, and an empty archive is valid.
        let archive = [0_u8; 1024];
        let entries = ArchiveUnpacker::unpack_with_options(
            ArchiveFormat::Tar,
            &archive,
            &UnpackOptions::new().with_password("hunter2"),
        )
        .expect("tar ignores the password");
        assert_eq!(entries, Vec::new());
    }
}
