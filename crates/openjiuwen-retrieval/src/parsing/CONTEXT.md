# src/parsing — DEFLATE and ZIP containers

## Intent

Read the compressed container layer under binary office documents without adding a compression dependency.
DOCX and XLSX are ZIP archives of XML parts; PDF content streams are often `zlib`-wrapped `DEFLATE`.
This module stops at containers: verified entry bytes in, verified entry bytes out.
The DOCX/XLSX/PDF-to-`Document` parsers and any XML parsing sit above it and are not implemented here yet.

## API Surface

- `inflate::inflate(data, max_output) -> Result<Vec<u8>>` — raw `DEFLATE` (RFC 1951): stored blocks, fixed Huffman, dynamic Huffman (code-length alphabet with repeat codes 16/17/18), LSB-first bit reader, length/distance base+extra tables, and back-reference copying.
- `inflate::zlib_decompress(data, max_output) -> Result<Vec<u8>>` — validates the `zlib` header (CM=8, CINFO<=7, FCHECK, no FDICT), inflates the payload, verifies the trailing `Adler-32`.
- `inflate::adler32(data) -> u32` and `inflate::crc32(data) -> u32` — table-driven standard checksums (`CRC-32` uses the reflected `0xEDB8_8320` polynomial).
- `inflate::DEFAULT_MAX_OUTPUT_BYTES` — 64 MiB decompression-bomb guard; the ZIP reader passes it to `inflate`.
- `zip::ZipArchive` — `from_bytes(Vec<u8>)`, async `open(path)` (`tokio::fs`, no blocking I/O), `entries`, `entry_names`, `entry`, `has_entry`, `len`, `is_empty`, `archive_size`, `read_entry`, `read_entry_ignore_case`, `read_text`.
- `zip::ZipEntry` — `name`, `compression_method`, `flags`, `crc32`, `compressed_size`, `uncompressed_size`, `local_header_offset`, plus `is_directory`, `is_utf8_name`, `compression_method_name`.
- `zip::compression_method_name(method) -> &'static str` — APPNOTE names used in `Unsupported` messages.
- `zip::COMPRESSION_METHOD_STORED` (0) and `zip::COMPRESSION_METHOD_DEFLATE` (8) — the only supported methods.
- Crate-root re-exports from this module: `parsing::{inflate, zlib_decompress, adler32, crc32, ZipArchive, ZipEntry, DEFAULT_MAX_OUTPUT_BYTES}`.

## Error Conventions

- `Error::Validation` — structurally invalid data: invalid block type, bad Huffman code/length, invalid distance, over-subscribed code, truncated-inconsistent stored block, output over `max_output`, bad `zlib` header fields, missing/invalid ZIP signatures, local-vs-central header mismatch, directory entry read as a file.
- `Error::Retrieval` — data ended early or failed integrity: truncated `DEFLATE`/`zlib` stream, `Adler-32` mismatch, ZIP `CRC-32` mismatch, ZIP uncompressed-size mismatch.
- `Error::Unsupported` — deliberately unimplemented variants: `zlib` preset dictionaries, ZIP64 archives, split/multi-disk archives, encrypted entries, compression methods other than 0 and 8 (the message names the method).
- `Error::NotFound` — a requested ZIP entry, or the file passed to `ZipArchive::open`, does not exist.

## Constraints

- No new dependencies: the decoder is plain Rust over the existing workspace crates.
- `#![forbid(unsafe_code)]`; ZIP files are read with `tokio::fs` so nothing blocks the runtime.
- Every allocation is bounded by `max_output`/`DEFAULT_MAX_OUTPUT_BYTES` before bytes are appended.
- ZIP paths are byte-oriented: `/` separators and exact, case-sensitive lookups; `read_entry_ignore_case` is the documented ASCII-case-insensitive fallback.
- Malformed input never panics: every slice access is bounds-checked through byte readers.
- Files stay around ≤900 lines: `inflate.rs` holds the public API, the decoder internals, and the checksums; `inflate/tests.rs` holds the unit tests and the in-test `DEFLATE` writer; `zip.rs` holds the archive reader.

## Testing

- `inflate/tests.rs` (unit, beside the module): canned vectors (`zlib` "hello", a reference-encoder dynamic block with back-references), a hand-derived fixed-Huffman stream for `"A"`, stored-block streams, generated fixed/dynamic/dynamic-with-repeats streams over payloads including all bytes 0..=255, and every documented error path.
- `tests/parsing_zip.rs` (integration): archives assembled byte by byte (local headers, central directory, EOCD), stored and deflated entries, CRC and size failures, unsupported method/encryption/ZIP64/split archives, missing EOCD, `read_text`, `read_entry_ignore_case`, and `open` against `tempfile`.
- Tests are offline and deterministic; no wall-clock, no network, no random ordering.

## Routing Table

- `./inflate.rs` → public `DEFLATE`/`zlib` API, checksums, decoder internals (`BitReader`, `Huffman`, block decoders)
- `./inflate/tests.rs` → unit tests plus the in-test `DEFLATE` stream writers
- `./zip.rs` → `ZipArchive`/`ZipEntry`, EOCD and central-directory parsing, entry extraction and verification
- `./mod.rs` → module root, re-exports, and the container-layer overview
