# skill_creator/omni/archive

## Intent

Archive sniffing, unpacking, and `zip` writing for the omni scrape stage, implemented entirely in-crate with no compression dependency.
The `skill_omni_creation` pipeline accepts a URL that may resolve to an archive rather than an HTML page, so `detect` identifies the container from its magic bytes and `ArchiveUnpacker` turns it into ordered `ArchiveEntry` values.
`ArchiveWriter` produces deterministic stored `zip` bytes so callers and tests can round-trip entries through the reader.
Supported containers: `ustar`/GNU `tar`, `zip` (stored + DEFLATE, classic 32-bit), and `gzip` (RFC 1952, including a gzip-wrapped `tar` or `zip`).
Every entry path is validated before it is returned, so a caller can never write outside the bundle root.

## API Surface

- `ArchiveFormat::{Tar, Zip, Gzip}` with `as_str()`/`Display` (`"tar"`, `"zip"`, `"gzip"`).
- `detect(bytes) -> Option<ArchiveFormat>` — `1f 8b` → gzip, `PK\x03\x04` / `PK\x05\x06` → zip, `ustar` at byte 257 → tar.
- `ArchiveEntry { path, bytes, is_dir }` — serde `rename_all = "snake_case", default`; `is_dir` entries carry empty bytes.
- `ArchiveUnpacker::unpack(format, bytes)` / `ArchiveUnpacker::unpack_auto(bytes)` — associated functions on a never-instantiated type; `unpack_auto` sniffs with `detect` and rejects unknown bytes as `Error::validation`.
- `ArchiveWriter::write_stored(entries: &[ArchiveEntry]) -> Result<Vec<u8>>` — stored-only zip writer (fixed DOS timestamp 1980-01-01 00:00:00, deterministic bytes, external attributes marking directories).
- Public caps: `MAX_ENTRY_UNCOMPRESSED_BYTES` (8 MiB, matching the `HttpFetcher` body default) and `MAX_ARCHIVE_UNCOMPRESSED_BYTES` (32 MiB).
- Crate-internal: `crc::crc32(bytes) -> u32` (IEEE, table-driven), `inflate::inflate(bytes, max_output) -> Result<Vec<u8>>`, `inflate::inflate_stream(bytes, max_output) -> Result<InflateOutcome>` (`InflateOutcome { data, consumed }`), `validate_path`, `slice_at`, `read_u16_le`/`read_u32_le`, and `CapTracker::{new, charge}`.

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`: malformed/truncated/unsafe input is `Error::validation` naming a byte offset (and the member, entry, or cap); unimplemented capability is `Error::unsupported` naming the exact reason (ZIP64, encryption, compression method other than zip 0/8 or gzip 8).
- No panics, no unbounded allocation: `tar` and `zip` charge their declared sizes through `CapTracker` before copying or inflating, and DEFLATE output is bounded by an explicit cap.
- No new dependencies; `unsafe` is denied workspace-wide; files stay under ~1000 lines.
- Work is bounded by the public caps, so a zip bomb or a highly expanding gzip member is an `Error::validation`, not an OOM.
- Every public item is documented (workspace `missing_docs`, clippy pedantic, `-D warnings`).
- `zip` names: UTF-8 flag bit 11 decodes strictly, otherwise UTF-8 with a CP437 fallback table; backslashes are never normalised and always fail `validate_path`.
- `gzip` verifies the `FHCRC` header CRC when present, every member's CRC-32 and `ISIZE`, and rejects trailing garbage and nested gzip payloads.

## Deferrals

- ZIP64 archives (EOCD locator/record, `0xFFFF_FFFF` size or offset markers, a 65535 entry count) return `Error::unsupported` naming ZIP64; classic 32-bit archives are fully supported.
- Encrypted zip entries (general-purpose bit 0) and zip compression methods other than stored (0) and DEFLATE (8) return `Error::unsupported`.
- gzip compression methods other than DEFLATE (8) return `Error::unsupported`; `FTEXT` carries no meaning and is ignored.
- The zip writer emits stored entries only (no DEFLATE writer, no archive comment); data-descriptor archives are read using the central directory as the authoritative size source.
- `tar` ignores symlinks, hard links, devices, and FIFOs, and reads PAX/long-name headers only to skip them (no `path=`/`linkpath=` attribute parsing).

## Routing Table

- `./mod.rs` → the public surface: `ArchiveFormat`, `detect`, `ArchiveEntry`, `ArchiveUnpacker`, `ArchiveWriter` re-export, caps, `CapTracker`, shared path validation and little-endian byte helpers
- `./tar.rs` → `ustar`/GNU tar parsing (headers, checksums, base-256 sizes, long names, PAX skipping)
- `./zip.rs` → end-of-central-directory scan, ZIP64/encryption/method rejections, central-directory and local-header parsing, stored/DEFLATE extraction, stored writer (`./zip/tests.rs` holds its unit tests)
- `./gzip.rs` → RFC 1952 header/member parsing, concatenated members, CRC-32 + ISIZE verification, payload archive detection (`./gzip/tests.rs` holds its unit tests)
- `./inflate.rs` → RFC 1951 DEFLATE decoder (stored, fixed, dynamic Huffman, LZ77, output cap, consumed-byte reporting; `./inflate/tests.rs` holds its unit tests)
- `./crc.rs` → shared table-driven CRC-32 (IEEE)
- `./CONTEXT.md` → this document
