# media

## Intent

Image and audio metadata tools read from the container header only — no decoding crates, no pixels, no samples, no network.
Ports the metadata half of `openjiuwen/harness/tools/multimodal/` (`audio.py` `AudioMetadataTool`, `vision.py`) as a model-free, offline subset.
Every field comes from a bounds-checked byte walk, so a truncated or malformed file yields a typed `success=false` payload naming the problem instead of a panic.
Format detection is by magic bytes, never by file extension, so a mislabelled file still parses.
Upper layers only see the two tools and the dispatch functions; the per-format parsers are internal detail exposed for tests.

## API Surface

### Tools — `image.rs`, `audio.rs`

- `image::TOOL_NAME = "image_metadata"`; `ImageMetadataArgs { image_path: String }` with aliases `path`, `file_path`, `image_path_or_url` and `#[serde(default)]`; `ImageMetadataTool` — `new(env: ToolEnv, language: impl Into<String>)`, `card_for(language: &str) -> ToolCard` (impl `Tool`); card `required = ["image_path"]`.
- `audio::TOOL_NAME = "audio_metadata"`; `AudioMetadataArgs { audio_path: String }` with aliases `path`, `file_path`, `audio_path_or_url`; `AudioMetadataTool` — `new(env, language)`, `card_for(language) -> ToolCard`; card `required = ["audio_path"]`.
- Both resolve the model path through `ToolEnv::resolve_path(TOOL_NAME, "<field>", raw, &args)` like `read_file`, check `ToolContext::check_cancelled`, then read the resolved file with `tokio::fs` (size gate [`MAX_MEDIA_BYTES`]).
- Success payload: `data = {kind: "image"|"audio", format, byte_size, ...format fields}`; failure payload: `success=false` with `error` text such as `unsupported image format: XXXX`, `truncated PNG header: IHDR is incomplete`, `empty file: no audio header bytes`.
- `parse_image_header(&[u8]) -> ParseResult<ImageMetadata>` and `parse_audio_header(&[u8]) -> ParseResult<AudioMetadata>` do the magic-byte dispatch; `ImageMetadata`/`AudioMetadata` expose `format()`, `width()`/`height()` (image), `sample_rate()`/`channels()`/`duration_ms()` (audio), and `to_data(byte_size) -> Value`.
- `audio::duration_seconds(ms: u64) -> f64` renders the Python `duration_seconds` field (two decimals).

### Format fields — images

- `png::PngInfo` — `width`, `height`, `bit_depth`, `color_type`, `color_type_name`, `has_alpha` (`tRNS` or alpha colour type), `interlaced`, `frames` (`acTL` animation frame count).
- `jpeg::JpegInfo` — `width`, `height`, `components`, `precision`, `progressive`, `orientation` (EXIF `0x0112` from `APP1`).
- `gif::GifInfo` — `width`, `height`, `version` (`87a`/`89a`), `frames`, `animated`, `has_alpha` (graphic-control transparency), `global_color_table`, `color_resolution`.
- `webp::WebpInfo` — `width`, `height`, `variant` (`extended`/`lossy`/`lossless`), `has_alpha`, `animated`, `frames`.
- `bmp::BmpInfo` — `width`, `height` (absolute, negative height reports `top_down`), `top_down`, `bit_count`, `compression`, `compression_name`, `dib_header_size`, `data_offset`.

### Format fields — audio

- `wav::WavInfo` — `sample_rate`, `channels`, `bits_per_sample`, `format_tag`, `format_name`, `byte_rate`, `block_align`, `data_bytes`, `duration_ms`.
- `flac::FlacInfo` — `sample_rate`, `channels`, `bits_per_sample`, `total_samples`, `duration_ms`, `min_block_size`, `max_block_size`, `id3v2_bytes`.
- `mp3::Mp3Info` — `sample_rate`, `channels`, `mpeg_version`, `layer`, `bitrate_kbps`, `duration_ms` (`Xing`/`Info` frame count, else CBR estimate), `xing_frames`, `id3v2_bytes`.
- `ogg::OggInfo` — `sample_rate`, `channels`, `codec` (`vorbis`/`opus`), `duration_ms` (last-page granule), `granule`, `nominal_bitrate`, `pre_skip`, `input_sample_rate`.

### Shared — `parse.rs`

- `MediaKind { Image, Audio }` with `as_str()`/`Display`; `Failure` (`new`, `message()`, `Display`, `std::error::Error`); `ParseResult<T> = Result<T, Failure>`.
- `Cursor<'a>` — `new`, `data`, `len`, `is_empty`, `position`, `remaining`, `take`, `skip`, `seek`, `u8`, `be_u16/u24/u32/u64`, `le_u16/u24/u32/u64`; all return `Option` on a short read, never panic.
- Free helpers `be_u16_at`, `le_u16_at`, `be_u24_at`, `le_u24_at`, `be_u32_at`, `le_u32_at`, `le_i32_at`, `be_u64_at`, `le_u64_at` and `fourcc(&[u8]) -> String` (printable ASCII as-is, otherwise lowercase hex, empty → `<empty>`).

## Constraints

- Only files under `src/tools/media/` belong to this node; `src/tools/mod.rs` (registry, `BuiltinTool`, `tool_names`) and `src/tools/descriptions.rs` are owned elsewhere — the tools are not registered yet and the description table has no entries, so `description("image_metadata"|"audio_metadata", lang)` returns `""` today.
- Expected failures are `success=false` payloads or `Failure` values, never `Err`/panic; `Err` is reserved for protocol violations and cancellation.
- No `unwrap`/`expect`/`panic!` outside `#[cfg(test)]`, no `unsafe`, no `println!`/`eprintln!`; `missing_docs` applies to every public item.
- Paths go only through `ToolEnv::resolve_path`; an absolute path inside the workspace is accepted, anything outside (`..`, outside the root, `http(s)` URLs) is refused.
- No new dependencies: `serde`, `serde_json`, `schemars`, `async-trait`, `tokio`, `tracing` only; detection never shells out or spawns a decoder.
- Files stay ≲1000 lines (largest is `audio.rs` at ~530).
- `#![allow(clippy::module_name_repetitions)]` (format-named types like `png::PngInfo`) and `#![allow(clippy::result_large_err)]` (`ToolPayload` is the legitimate error type, as in `support.rs`) are documented at the top of `mod.rs`.

## Design decisions

- **Deviation — no model, no service**: Python delegates transcription/OCR/visual Q&A/ACR identification to a vision or audio model; here the tools are pure header readers, so they run offline and deterministically.
- **Deviation — metadata only**: nothing is decoded; duration comes from declared sizes, `Xing` frame counts, `STREAMINFO` totals or the last OGG page granule, and is `null` when the header does not declare enough.
- **Deviation — no URL inputs**: `image_path_or_url`/`audio_path_or_url` are accepted as aliases, but an `http(s)` value fails the workspace path check instead of being fetched.
- **Deviation — one payload envelope**: both tools always return the six-key `ToolPayload` JSON, so a parse failure is `success=false` with the message in `error` rather than a Python exception.
- **Deviation — schema `required`**: `schemars` treats a `#[serde(default)]` field as optional and ignores `#[schemars(required)]`, so `card_for` post-processes `input_params` with a local `require_field` to restore Python's `required` list.
- Walks are bounded (`MAX_CHUNKS`/`MAX_SEGMENTS`/`MAX_BLOCKS`) so a hostile header cannot spin; every length is validated before `take`.
- Unknown magic → `unsupported image format: <4cc>` / `unsupported audio format: <4cc>`; zero-length input → `empty file: no image header bytes` / `empty file: no audio header bytes`.

## Deferred Python behaviour

- OCR, transcription, captioning, visual question answering, ACR/audio-fingerprint identification and any model call.
- URL/HTTP(S) inputs, multipart uploads and remote asset resolution.
- Pixel/sample decoding, thumbnails, waveform or peak extraction, cover-art/ID3 frame parsing beyond the tag length.
- Additional containers (TIFF, HEIF/AVIF, ICO, SVG, MP4/M4A, OGG-FLAC, AIFF) and MP3/VBR per-frame refinement beyond the first frame plus `Xing`.
- Python's `AudioMetadataTool`/vision tool registration in the tool registry and their description table entries.

## Routing Table

- `./mod.rs` → module docs, module declarations, flat re-exports, `MAX_MEDIA_BYTES`, `read_media_bytes`, `require_field`
- `./parse.rs` → `MediaKind`, `Failure`, `ParseResult`, `Cursor`, endian helpers, `fourcc`
- `./image.rs` → `image_metadata` tool, `ImageMetadataArgs`, `ImageMetadata` dispatch and payload
- `./image/png.rs` → PNG `IHDR` + `tRNS`/`acTL` walk
- `./image/jpeg.rs` → JPEG marker walk, `SOF` frame, `APP1`/EXIF orientation
- `./image/gif.rs` → GIF logical screen descriptor, block/frame walk, transparency
- `./image/webp.rs` → RIFF `VP8X`/`VP8 `/`VP8L`/`ANMF`
- `./image/bmp.rs` → BMP file header and `BITMAPINFOHEADER`/`BITMAPCOREHEADER`
- `./audio.rs` → `audio_metadata` tool, `AudioMetadata` dispatch and payload, `duration_seconds`, `strip_id3v2`
- `./audio/wav.rs` → RIFF/WAVE `fmt ` + `data` chunks
- `./audio/flac.rs` → FLAC `STREAMINFO`
- `./audio/mp3.rs` → MPEG frame header, bitrate tables, `Xing`/`Info`
- `./audio/ogg.rs` → OGG page walk, Vorbis identification header, Opus head, granule duration
- `./fixtures.rs` → hand-built minimal valid headers for every format (`#[cfg(test)]` only)

## Test Strategy

- 113 unit tests live beside the parsers: `parse` (cursor reads, endian helpers, `fourcc`), one module per format, plus the two tool modules.
- Each format module covers its happy path, a truncated header, an unknown/foreign magic and a zero-dimension or zero-rate rejection; `image.rs`/`audio.rs` add dispatch-by-magic coverage across every format.
- Tool tests use `tempfile` workspaces and hand-written fixture bytes: payload shape (`kind`, `format`, `byte_size`, per-format fields), extension disagreement (`media.bin` holding PNG bytes), Python-style path aliases, a missing path, malformed and empty files, and a path outside the workspace being refused.
- `fixtures.rs` builds every fixture from a few dozen literal bytes — no binary assets, no network, no wall clock.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --offline`, `cargo clippy -p openjiuwen-harness --all-targets --offline -- -D warnings`, `cargo test -p openjiuwen-harness --no-default-features --offline` all clean.
