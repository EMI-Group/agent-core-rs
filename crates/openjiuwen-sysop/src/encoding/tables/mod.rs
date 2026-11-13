//! Generated lookup tables for the legacy CJK decoders.
//!
//! Every file in this directory is data, not code: `scripts/gen_cjk_tables.py` emits it
//! from Python's built-in codecs and asserts that the tables and the decoder reproduce
//! `bytes.decode(encoding)` for every code position. Do not edit these files by hand —
//! rerun the generator instead, and prefer `--check` in a review to prove they are current.
//!
//! Layout:
//!
//! - [`gb18030`] — dense two-byte table (126 lead rows × 190 trail columns) plus 207
//!   range-compressed runs for the four-byte index space.
//! - [`big5`] — dense two-byte table (89 lead rows × 157 trail columns).
//! - [`shift_jis`] — dense two-byte table (60 lead rows × 188 trail columns); the
//!   halfwidth katakana block is positional and therefore lives in the decoder.
//!
//! A `0` entry means "unmapped" (Big5 and Shift-JIS only); GB18030 maps every two-byte
//! position and uses the table density instead of a sentinel.

pub(crate) mod big5;
pub(crate) mod gb18030;
pub(crate) mod shift_jis;
