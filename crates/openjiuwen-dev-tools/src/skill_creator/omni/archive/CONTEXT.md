# skill_creator/omni/archive

## Intent

Archive sniffing, unpacking, and `zip` writing for the omni scrape stage, implemented entirely in-crate with no compression dependency.
The `skill_omni_creation` pipeline accepts a URL that may resolve to an archive rather than an HTML page, so `detect`/`detect_named` identify the container and `ArchiveUnpacker` turns it into ordered `ArchiveEntry` values.
`ArchiveWriter` produces deterministic stored `zip` bytes so callers and tests can round-trip entries through the reader.
Decodable containers: `ustar`/GNU `tar` (including PAX and GNU long-name records), `zip` (stored + DEFLATE, classic 32-bit and ZIP64, `ZipCrypto` with a password), and `gzip` (RFC 1952, including a gzip-wrapped `tar` or `zip`).
`bzip2`, `xz`, and `zstd` are recognized (magic bytes or filename suffix) but not decodable: no such crate is available offline, so they are a typed `Unsupported` naming the codec.
Every entry path is validated before it is returned, so a caller can never write outside the bundle root.

## API Surface

- `ArchiveFormat::{Tar, Zip, Gzip, Bzip2, Xz, Zstd}` with `as_str()`/`Display` (`"tar"`, `"zip"`, `"gzip"`, `"bzip2"`, `"xz"`, `"zstd"`).
- `detect(bytes) -> Option<ArchiveFormat>` — `1f 8b` → gzip, `PK\x03\x04` / `PK\x05\x06` → zip, `ustar` at byte 257 → tar, `BZh` + a level digit → bzip2, `FD 37 7A 58 5A 00` → xz, `28 B5 2F FD` → zstd.
- `detect_named(name, bytes) -> Option<ArchiveFormat>` — `detect` first, else a case-insensitive suffix fallback (`.tar.bz2`/`.tbz2`/`.tbz`, `.tar.xz`/`.txz`, `.tar.zst`/`.tzst`/`.zstd`).
- `ArchiveEntry { path, bytes, is_dir, kind, link_target, mode }` — serde `rename_all = "snake_case", default`; `is_dir` stays true exactly for `kind == Directory`; a `Symlink`/`HardLink` entry carries its target in `link_target` with empty `bytes`; `mode` is the Unix mode when the container declares one; old JSON without `kind`/`link_target`/`mode` still deserializes (`File`/`None`/`None`).
- `ArchiveEntryKind::{File, Directory, Symlink, HardLink}` — serde `snake_case` (wire name `hard_link`), `Default` = `File`.
- `UnpackOptions { password: Option<String> }` — serde `snake_case` + `default`, `new()`, `with_password()`, `validate()` (rejects `Some("")`); the password is UTF-8 and is ignored by `tar`/`gzip`.
- `ArchiveUnpacker::unpack(format, bytes)` / `unpack_auto(bytes)` — the no-password shorthand; `unpack_auto` sniffs with `detect` and rejects unknown bytes as `Error::validation`.
- `ArchiveUnpacker::unpack_with_options(format, bytes, &UnpackOptions)` / `unpack_auto_with_options(bytes, &UnpackOptions)` — the password-carrying entry points (associated functions on a never-instantiated type).
- `ArchiveWriter::write_stored(entries: &[ArchiveEntry]) -> Result<Vec<u8>>` — stored-only zip writer (fixed DOS timestamp 1980-01-01 00:00:00, deterministic bytes, external attributes carrying `0o120777` for a `Symlink` — whose target comes from `link_target`, else `bytes` — `0o040755` for a directory, `0o100644` otherwise).
- Public caps: `MAX_ENTRY_UNCOMPRESSED_BYTES` (8 MiB, matching the `HttpFetcher` body default) and `MAX_ARCHIVE_UNCOMPRESSED_BYTES` (32 MiB).
- Crate-internal: `crc::crc32(bytes) -> u32` (IEEE, table-driven), `inflate::inflate(bytes, max_output) -> Result<Vec<u8>>`, `inflate::inflate_stream(bytes, max_output) -> Result<InflateOutcome>` (`InflateOutcome { data, consumed }`), `validate_path`, `slice_at`, `read_u16_le`/`read_u32_le`, and `CapTracker::{new, charge}`.

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`: malformed/truncated/unsafe input is `Error::validation` naming a byte offset (and the member, entry, or cap); an encrypted entry read without a password is `Error::validation` naming the entry and `UnpackOptions::with_password`; a capability this offline reader does not ship is `Error::unsupported` naming the codec or scheme exactly (bzip2/xz/zstd, `WinZip` AES, PKWARE strong encryption, an encrypted central directory, zip compression method other than 0/8, gzip method other than 8, a 64-bit ZIP64 value wider than `usize`).
- No panics, no unbounded allocation: `tar` and `zip` charge their declared sizes through `CapTracker` before copying, decrypting, or inflating, and DEFLATE output is bounded by an explicit cap.
- No new dependencies; `unsafe` is denied workspace-wide; implementation files stay under ~1000 lines.
- Work is bounded by the public caps, so a zip bomb or a highly expanding gzip member is an `Error::validation`, not an OOM.
- Every public item is documented (workspace `missing_docs`, clippy pedantic, `-D warnings`).
- `zip` names: UTF-8 flag bit 11 decodes strictly, otherwise UTF-8 with a CP437 fallback table; backslashes are never normalised and always fail `validate_path`.
- `zip` symlink targets are stored verbatim (an absolute or `..`-climbing target stays an opaque string in the returned record); only the entry path is validated, and link entries are never followed or written.
- `zip` is the only container that can carry encryption, and a password reaches only a top-level `zip`: a `zip` nested in a `gzip` wrapper goes through `gzip`'s own no-password path.
- `gzip` verifies the `FHCRC` header CRC when present, every member's CRC-32 and `ISIZE`, and rejects trailing garbage and nested gzip payloads.

## Deferrals

- `bzip2`, `xz`, and `zstd` archives and gzip-style wrappers around them are recognized and refused with `Error::unsupported` naming the codec: no bzip2/xz/zstd crate is in the workspace or the offline registry.
- `WinZip` AES (compression method `99` or extra field `0x9901`), PKWARE strong encryption (general-purpose bit 6), an encrypted central directory (bit 13), and zip compression methods other than stored (0) and DEFLATE (8) are `Error::unsupported`.
- `ZipCrypto` is the only implemented scheme; it has no password-strength check beyond the 12-byte header's one-byte check, which is a property of the format, not of this reader.
- The zip writer emits stored entries only (no DEFLATE writer, no archive comment, no ZIP64 writer); data-descriptor archives are read using the central directory as the authoritative size source.
- Devices and FIFOs in a `tar` archive are still skipped (they carry no contents); PAX records other than `path`/`linkpath`/`size` are parsed and ignored (`mtime`/`uid`/`gid`/`uname`/`gname`/`comment`), and `SCHILY.*`/vendor-specific records are ignored.

## Known Issues

- `archive/zip/tests.rs` is ~1600 lines: it holds the shared synthetic-archive builders plus the classic, ZIP64, ZipCrypto, symlink and rejection matrices, so it is deliberately one long suite rather than a per-concern split.

## Routing Table

- `./mod.rs` → the public surface: `ArchiveFormat`, `detect`/`detect_named`, `ArchiveEntry`/`ArchiveEntryKind`, `UnpackOptions`, `ArchiveUnpacker`, `ArchiveWriter` re-export, caps, `CapTracker`, shared path validation and little-endian byte helpers
- `./tar/` → `ustar`/GNU tar parsing: checksums, base-256 sizes, PAX `x`/`g` overrides, GNU `L`/`K`, symlink/hardlink entries with their target, modes, end-of-archive (`./tar/tests.rs` holds its unit tests)
- `./zip/` → `archive/zip/mod.rs` walks the end records, the central directory (classic, ZIP64, escape values) and dispatches each entry; `zip/payload.rs` locates, decrypts, inflates and verifies one payload; `zip/zip64.rs` reads the ZIP64 EOCD/locator and per-entry `0x0001` extra field; `zip/zipcrypto.rs` holds the ZipCrypto key schedule and stream decryption (with a `#[cfg(test)]` encryptor); `zip/writer.rs` is the stored writer; `./zip/tests.rs`, `./zip/zip64/tests.rs` and `./zip/zipcrypto/tests.rs` hold their suites
- `./gzip.rs` → RFC 1952 header/member parsing, concatenated members, CRC-32 + ISIZE verification, payload archive detection (`./gzip/tests.rs` holds its unit tests)
- `./inflate.rs` → RFC 1951 DEFLATE decoder (stored, fixed, dynamic Huffman, LZ77, output cap, consumed-byte reporting; `./inflate/tests.rs` holds its unit tests)
- `./crc.rs` → shared table-driven CRC-32 (IEEE)
- `./CONTEXT.md` → this document
