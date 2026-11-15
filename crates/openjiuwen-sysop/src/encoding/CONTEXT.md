# encoding — text encoding detection, BOM handling, binary sniffing, decoding, encoding

## Intent

Owns everything the filesystem backend needs to turn bytes into text and text into bytes:
`detect_text_encoding` (Python `local/_text_encoding.py`), BOM handling, the binary sniff,
every decoder, and the matching encoders.
UTF-8/UTF-16/Latin-1 codecs live in `mod.rs`; the legacy CJK encodings (GB18030/GBK, Big5,
Shift-JIS) decode in `cjk` from generated tables and encode in `encode` from the reverse of
those same tables.
Every encoding this crate decodes can also be written, and each one round-trips through
`decode_text`/`encode_text` with the same name.

## API Surface

- `mod.rs` — `TextEncoding` (seven variants, canonical `as_str`, lenient `parse` with legacy
  aliases, `is_cjk`), `bom_encoding`, `strip_bom`, `detect_text_encoding`,
  `detected_encoding_name`, `is_binary`, `decode_text`, `encode_text`, and the three BOM
  constants; all of these are re-exported at `crate::encoding` as before.
- `mod.rs` private helpers — `detect_cjk` (strict GB18030/Big5 candidates, GB18030 wins ties),
  `text_quality` (Python `_text_quality` scoring over at most 2 Mi characters),
  `character_weight`, `detect_utf16_without_bom`, `decode_utf16`, `encode_utf16` (UTF-16
  writes), `is_control`, `is_whitespace`, `ratio_exceeds`, and `SUPPORTED_WRITE_ENCODINGS`
  (the list every write-encoding error message spells out).
- `cjk` — `CjkEncoding` (`Gb18030`/`Big5`/`ShiftJis`, `from_text_encoding`, `text_encoding`),
  `decode`, `decode_strict` (returns `None` when anything is invalid, used by detection), the
  incremental `CjkDecoder` (`new`, `push`, `finish`, `invalid_input`, `encoding`), and the
  crate-internal reverse geometry the encoder builds on: `two_byte_shape`,
  `two_byte_bytes`, `decode_two_byte`, `decode_four_byte`, `four_byte_index`,
  `four_byte_bytes`, `halfwidth_katakana_byte`, `decode_halfwidth_katakana`,
  `HALFWIDTH_KATAKANA_BYTES`.
- `encode` — crate-internal write side: `encode_latin1`, `encode_cjk` plus the
  `Code`/reverse-map machinery described under “Write support” below.
- `tables` — generated data only: `TWO_BYTE` (row-major `u16`) per encoding plus
  `FOUR_BYTE_RUNS`/`FOUR_BYTE_MAX_INDEX` for GB18030's four-byte index space.
- `TextEncoding::parse` accepts `utf-8`/`utf8`/`us-ascii`/`ascii`, `utf-16le`/`utf16le`/`utf-16`,
  `utf-16be`, `latin-1`/`latin1`/`iso-8859-1`/`l1`, `gb18030`/`gbk`/`gb2312`/`cp936`,
  `big5`/`big5-hkscs`/`cp950`, and `shift-jis`/`shiftjis`/`sjis`/`cp932` (case-insensitive,
  `_` normalized to `-`).

## Write support

`encode_text(text, encoding) -> Result<Vec<u8>>` writes every parsed encoding, and its
`&str` signature and `Result<Vec<u8>>` return are part of the public contract.
`utf-8` copies the UTF-8 bytes, `utf-16le`/`utf-16be` write the UTF-16 code units for the
declared endianness, and `latin-1` writes one byte per `U+0000..=U+00FF` character (`é` is
the single byte `0xE9`; anything above `U+00FF` is a typed failure, never a UTF-8 fallback).
`gb18030` (also selected by `gbk`/`gb2312`/`cp936`) writes the GBK two-byte area, then
GB18030's four-byte form for everything outside it (the BMP and the linear supplementary
region), `big5` (`cp950`, `big5-hkscs` aliases) writes its two-byte area, and `shift-jis`
(`sjis`, `cp932`) writes its two-byte area plus the single-byte halfwidth katakana block.
`auto` is a read policy: writing with it is refused, because a write must declare its bytes.

### Reverse lookup mechanism and cost

The two-byte encoders are built as the inverse of the generated decode tables, so no second
table is committed: one `OnceLock<Vec<(char, Code)>>` per CJK encoding holds that encoding's
reverse map, built on first use and reused for the process lifetime (`(char, Code)` is 12
bytes, so the GB18030 map is 23940 entries — about 280 KB — Big5 is ~14k entries (~160 KB)
and Shift-JIS ~7k (~80 KB)).
Entries are sorted by character and looked up with a binary search (~15 comparisons), never a
per-character scan over the table.
Each position is verified by decoding its bytes back before insertion, and positions that
decode to the same character collapse to one deterministic winner: the shortest code, ties
broken by the smallest byte sequence (Big5's four duplicate mappings therefore resolve to
`0xA2CC` for `十`, where Python's encoder picks `0xA451`).
Shift-JIS halfwidth katakana is arithmetic (`U+FF61..=U+FF9F` ↔ `0xA1..=0xDF`) and needs no
table; GB18030's four-byte form is one binary search over the 207 range runs plus the
documented index algebra, with the candidate bytes decoded back before they are returned —
no 1.1M-entry supplementary-plane map.
ASCII text never builds a map (the ASCII fast path returns before the lookup), so an
ASCII-only write allocates nothing beyond its output.

## Error rules

- An unknown encoding name is `Error::config` and the message lists the writable encodings
  (`encoding 'koi8-r' is not recognized; supported encodings are utf-8, utf-16le, utf-16be,
  latin-1, gb18030 (gbk), big5, and shift-jis`); it is never `Unsupported`.
- `auto`/`AUTO_ENCODING` is `Error::config` with `'auto' cannot be used for writing; declare
  an encoding`.
- A character the declared encoding cannot represent is `Error::validation`, e.g.
  `character U+4E2D ('中') cannot be represented in encoding 'latin-1'`; it names both the
  encoding and the code point so the caller can fix the content.
- `decode_text` keeps `Error::unsupported` for an unknown read name (unchanged) and
  `Error::validation` for content rejected by `BinaryPolicy::Reject`.

## Constraints

- Public paths are stable: `TextEncoding`, `decode_text`, `encode_text`, `is_binary`,
  `detect_text_encoding`, `bom_encoding`, `strip_bom`, `detected_encoding_name` and the BOM
  constants must stay importable from `crate::encoding`.
- `encode_text` never silently writes UTF-8 for another declared encoding and never returns
  `Unsupported`; a write either produces the declared codec's bytes or a typed error.
- Never panic on malformed bytes or unrepresentable text: invalid or truncated sequences
  decode to `U+FFFD` (Python `errors="replace"`), and encoding failures are returned as
  errors.
- Work per byte is bounded: a two-byte decode is one array read, a two-byte encode is one
  binary search, a four-byte index is a binary search over 207 runs, and the streaming decoder
  buffer holds at most three bytes.
- `TextEncoding::as_str` reports the canonical covering name, so `gbk`, `gb2312` and `cp936`
  all report `gb18030`.
- Shift-JIS is never auto-detected (Python does not consider it either); it needs an explicit
  name because its bytes are normally valid GB18030/Big5 too.
- Tables are generated, never edited: rerun `scripts/gen_cjk_tables.py` (and `--check` in a
  review); no build or test needs Python, and the encoders read the committed tables instead
  of regenerating them.
- Deliberately decode-only: none. Two encode deviations stand: Shift-JIS rejects the
  vendor-specific single-byte aliases whose bytes would not decode back (Python maps `¥` to
  `0x5C`, which decodes as `\`), and Big5's duplicate mappings pick the smallest code rather
  than Python's choice.

## Routing Table

- `./mod.rs` → `TextEncoding`, detection, BOM, binary sniff, `decode_text`, `encode_text`
- `./cjk.rs` → the bounded CJK decoders, their row/column lookups, and the reverse geometry
- `./encode.rs` → Latin-1 and the lazy CJK reverse maps that back `encode_text`
- `./tables/mod.rs` → description of the generated data and its module layout
- `./tables/gb18030.rs`, `./tables/big5.rs`, `./tables/shift_jis.rs` → generated tables
- `../../scripts/gen_cjk_tables.py` → authoring-time generator and self-verification
- `../../tests/encoding_cjk.rs` → offline end-to-end fixtures for names, decoding and detection
- `../../tests/encoding_write.rs` → offline round-trip and byte-fixture suite for writing
