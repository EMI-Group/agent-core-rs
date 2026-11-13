# skill_creator/omni/archive

## Intent

Archive sniffing, unpacking, and `zip` writing for the omni scrape stage, implemented entirely in-crate with no compression dependency.
The `skill_omni_creation` pipeline accepts a URL that may resolve to an archive rather than an HTML page, so `detect` identifies the container from its magic bytes and `ArchiveUnpacker` turns it into ordered `ArchiveEntry` values.
Supported containers: `ustar`/GNU `tar`, `zip` (stored + deflate), and `gzip` (including a gzip-wrapped `tar`).
Every entry path is validated before it is returned so a caller can never write outside the bundle root.

## API Surface

- `ArchiveFormat::{Tar, Zip, Gzip}` with `as_str()`/`Display` (`"tar"`, `"zip"`, `"gzip"`).
- `detect(bytes) -> Option<ArchiveFormat>` — `1f 8b` → gzip, `PK\x03\x04` / `PK\x05\x06` / `PK\x07\x08` → zip, `ustar` at 257 → tar.
- `ArchiveEntry { path, bytes, is_dir }` — serde snake_case, `define_entry`-free plain struct, `is_dir` empty bytes for directories.
- `ArchiveUnpacker::unpack(format, bytes)` / `ArchiveUnpacker::unpack_auto(bytes)` — associated functions on a never-instantiated type.
- `ArchiveWriter::write_stored(entries: &[ArchiveEntry]) -> Result<Vec<u8>>` — stored-only zip writer (fixed 1980-01-01 DOS timestamp, deterministic bytes).
- Public caps: `MAX_ENTRY_UNCOMPRESSED_BYTES` (8 MiB) and `MAX_ARCHIVE_UNCOMPRESSED_BYTES` (32 MiB).
- `crc::crc32(bytes) -> u32` (IEEE, table-driven) and `inflate::inflate(bytes, max_output) -> Result<Vec<u8>>` are internal to the module tree but unit-tested.

## Constraints

- Errors are `openjiuwen_common::error::{Error, Result}`: malformed/truncated/unsafe input is `Error::validation` naming a byte offset; unimplemented capability is `Error::unsupported` naming the exact reason (ZIP64, compression method other than 0/8, encryption).
- No panics, no unbounded allocation: never allocate more than the declared size, and inflate is bounded by an explicit output cap.
- No new dependencies; `unsafe` is denied workspace-wide; files stay under ~1000 lines.
- Work is bounded by the public caps, so a zip bomb is a `Error::validation`, not an OOM.
- Every public item is documented (workspace `missing_docs`, clippy pedantic, `-D warnings`).

## Routing Table

- `./mod.rs` → the public surface: `ArchiveFormat`, `detect`, `ArchiveEntry`, `ArchiveUnpacker`, `ArchiveWriter`, caps, shared path validation and byte helpers
- `./tar.rs` → `ustar`/GNU tar parsing (headers, checksums, long names, PAX skipping)
- `./zip.rs` → end-of-central-directory scan, central-directory and local-header parsing, stored/deflate extraction, stored writer
- `./gzip.rs` → RFC 1952 header/member parsing, concatenated members, CRC32 + ISIZE verification
- `./inflate.rs` → RFC 1951 DEFLATE decoder (stored, fixed, dynamic Huffman, LZ77)
- `./crc.rs` → shared table-driven CRC-32 (IEEE)
