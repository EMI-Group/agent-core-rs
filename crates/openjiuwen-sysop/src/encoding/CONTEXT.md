# encoding — text encoding detection, BOM handling, binary sniffing, decoding
## Intent
Owns everything the filesystem backend needs to turn bytes into text: `detect_text_encoding`
(Python `local/_text_encoding.py`), BOM handling, the binary sniff, and every decoder.
UTF-8/UTF-16/Latin-1 decode in `mod.rs`; the legacy CJK encodings (GB18030/GBK, Big5,
Shift-JIS) live in `cjk` and are table driven.
Decoding is decode-only: writes stay UTF-8 and `encode_text` rejects the legacy names.
## API Surface
- `mod.rs` — `TextEncoding` (seven variants, canonical `as_str`, lenient `parse` with legacy
  aliases, `is_cjk`), `bom_encoding`, `strip_bom`, `detect_text_encoding`,
  `detected_encoding_name`, `is_binary`, `decode_text`, `encode_text`, and the three BOM
  constants; all of these are re-exported at `crate::encoding` as before.
- `mod.rs` private helpers — `detect_cjk` (strict GB18030/Big5 candidates, GB18030 wins ties),
  `text_quality` (Python `_text_quality` scoring over at most 2 Mi characters),
  `character_weight`, `detect_utf16_without_bom`, `decode_utf16`, `encode_utf16`, `is_control`,
  `is_whitespace`, `ratio_exceeds`.
- `cjk` — `CjkEncoding` (`Gb18030`/`Big5`/`ShiftJis`, `from_text_encoding`, `text_encoding`),
  `decode`, `decode_strict` (returns `None` when anything is invalid, used by detection), and
  the incremental `CjkDecoder` (`new`, `push`, `finish`, `invalid_input`, `encoding`).
- `tables` — generated data only: `TWO_BYTE` (row-major `u16`) per encoding plus
  `FOUR_BYTE_RUNS`/`FOUR_BYTE_MAX_INDEX` for GB18030's four-byte index space.
## Constraints
- Public paths are stable: `TextEncoding`, `decode_text`, `encode_text`, `is_binary`,
  `detect_text_encoding`, `bom_encoding`, `strip_bom`, `detected_encoding_name` and the BOM
  constants must stay importable from `crate::encoding`.
- Never panic on malformed bytes: invalid or truncated sequences become `U+FFFD`, and a
  truncated tail contributes exactly one replacement (Python `errors="replace"`).
- Work per byte is bounded: a two-byte lookup is one array read, a four-byte index is a
  binary search over 207 runs, and the streaming buffer holds at most three bytes.
- `TextEncoding::as_str` reports the canonical covering name, so `gbk`, `gb2312` and `cp936`
  all report `gb18030`.
- Shift-JIS is never auto-detected (Python does not consider it either); it needs an explicit
  name because its bytes are normally valid GB18030/Big5 too.
- Tables are generated, never edited: rerun `scripts/gen_cjk_tables.py` (and `--check` in a
  review); no build or test needs Python.
## Routing Table
- `./mod.rs` → `TextEncoding`, detection, BOM, binary sniff, `decode_text`/`encode_text`
- `./cjk.rs` → the bounded CJK decoders and their row/column lookups
- `./tables/mod.rs` → description of the generated data and its module layout
- `./tables/gb18030.rs`, `./tables/big5.rs`, `./tables/shift_jis.rs` → generated tables
- `../../scripts/gen_cjk_tables.py` → authoring-time generator and self-verification
- `../../tests/encoding_cjk.rs` → offline end-to-end fixtures for names, decoding and detection
