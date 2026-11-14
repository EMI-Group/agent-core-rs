//! Pure-Rust loading of a `tiktoken` `.model` BPE vocabulary.
//!
//! A `tiktoken.model` file is one `<base64-encoded-token-bytes> <rank>` entry per line, so the vocabulary is decoded with `base64` and `serde_json` alone — no `tiktoken` runtime, no network.
//! Encoding follows the `tiktoken` byte-pair algorithm: the text is pre-tokenized, each piece is turned into its UTF-8 bytes, and the adjacent pair with the lowest rank merges until no ranked pair remains.
//! The pre-tokenization rule is configurable: [`PretokenizePattern::Kimi`] (the default, matching Python's `TiktokenModelCounter`) and [`PretokenizePattern::Gpt2`] (the `cl100k_base`/GPT-2 rule).
//!
//! Deviations from Python, all deliberate:
//!
//! - The Kimi rule is the default for every vocabulary, matching `TiktokenModelCounter`; pointing it at a non-Kimi `.model` file produces the same plausible-but-different counts Python does.
//! - Special tokens are allowed by default (`allowed_special="all"` parity): an exact special-token text always encodes as one id.
//! - Unicode classification uses the standard library's `char::is_alphabetic`/`is_numeric`/`is_lowercase` plus a combining-mark range table instead of the full `\p{Han}`/`\p{L}`/`\p{M}` tables, which would need a regex engine or a Unicode database.
//! - A byte with no single-byte vocabulary entry is skipped while encoding and reported in [`TiktokenModel::warnings`], where Python's `tiktoken` would raise.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use base64::prelude::{Engine as _, BASE64_STANDARD};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The Kimi K2/K2.5/K2.6/K2.7 special-token map, used when the repository ships no `tokenizer_config.json`.
const KIMI_K2_SPECIAL_TOKENS: &[(&str, u32)] = &[
    ("[BOS]", 163_584),
    ("[EOS]", 163_585),
    ("<|im_end|>", 163_586),
    ("<|im_user|>", 163_587),
    ("<|im_assistant|>", 163_588),
    ("<|start_header_id|>", 163_590),
    ("<|end_header_id|>", 163_591),
    ("[EOT]", 163_593),
    ("<|im_system|>", 163_594),
    ("<|tool_calls_section_begin|>", 163_595),
    ("<|tool_calls_section_end|>", 163_596),
    ("<|tool_call_begin|>", 163_597),
    ("<|tool_call_argument_begin|>", 163_598),
    ("<|tool_call_end|>", 163_599),
    ("<|im_middle|>", 163_601),
    ("<|media_begin|>", 163_602),
    ("<|media_content|>", 163_603),
    ("<|media_end|>", 163_604),
    ("<|media_pad|>", 163_605),
    ("<think>", 163_606),
    ("</think>", 163_607),
    ("[UNK]", 163_838),
    ("[PAD]", 163_839),
];

/// The Kimi K3 special-token map; K3 shares the base BPE with K2 but renames several control tokens.
const KIMI_K3_SPECIAL_TOKENS: &[(&str, u32)] = &[
    ("[BOS]", 163_584),
    ("[EOS]", 163_585),
    ("<|end_of_msg|>", 163_586),
    ("<|open|>", 163_587),
    ("<|close|>", 163_588),
    ("<|sep|>", 163_589),
    ("[start_header_id]", 163_590),
    ("[end_header_id]", 163_591),
    ("[EOT]", 163_593),
    ("<|media_begin|>", 163_602),
    ("<|media_content|>", 163_603),
    ("<|media_end|>", 163_604),
    ("<|media_pad|>", 163_605),
    ("<osagent_mode>", 163_649),
    ("[UNK]", 163_838),
    ("[PAD]", 163_839),
];

/// How many unused `<|reserved_token_N|>` ids a vocabulary reserves above its base size.
const RESERVED_SPECIAL_TOKENS: u32 = 256;

/// The pre-tokenization rule applied before the BPE merges.
///
/// The two rules differ in how they cut words, contractions, digits and whitespace; everything after the split shares the same merge algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PretokenizePattern {
    /// The Kimi rule (`KIMI_TIKTOKEN_PAT_STR`): Han runs, case-aware word runs with attached contractions, 1–3 digit runs, and the trailing-whitespace lookahead.
    #[default]
    Kimi,
    /// The GPT-2/`cl100k_base` rule: contractions, optional-space word/digit/symbol runs, and the trailing-whitespace lookahead.
    Gpt2,
}

/// The special-token table of a `tiktoken` vocabulary.
///
/// Construction always pads the 256 ids directly above the mergeable vocabulary with `<|reserved_token_N|>` entries, mirroring Python's `_fill_reserved_special_tokens`.
#[derive(Debug, Clone, Default)]
pub struct TiktokenSpecials {
    /// The `(content, id)` entries, longest content first so the first match is the longest one.
    entries: Vec<(String, u32)>,
    /// Text to id.
    by_text: HashMap<String, u32>,
    /// Id to text.
    by_id: HashMap<u32, String>,
    /// The first characters an entry can start with, for fast rejection.
    starts: HashSet<char>,
}

impl TiktokenSpecials {
    /// Builds a table from `(content, id)` pairs and pads the reserved ids above `base_token_count`.
    ///
    /// Empty contents are ignored and a repeated content keeps its last id, matching a Python `dict` update.
    #[must_use]
    pub fn new(entries: Vec<(String, u32)>, base_token_count: usize) -> Self {
        let mut by_text: HashMap<String, u32> = HashMap::new();
        for (content, id) in entries {
            if !content.is_empty() {
                by_text.insert(content, id);
            }
        }
        let mut used: HashSet<u32> = by_text.values().copied().collect();
        let base = u32::try_from(base_token_count).unwrap_or(u32::MAX);
        for id in base..base.saturating_add(RESERVED_SPECIAL_TOKENS) {
            if used.insert(id) {
                by_text.insert(format!("<|reserved_token_{id}|>"), id);
            }
        }
        Self::from_map(by_text)
    }

    /// The built-in fallback for `model`: the K3 map when the name contains `kimi-k3`, else the K2 map.
    ///
    /// The name is matched case-insensitively, like Python's `casefold()` check.
    #[must_use]
    pub fn for_model(model: &str, base_token_count: usize) -> Self {
        let name = model.to_lowercase();
        let defaults = if name.contains("kimi-k3") {
            KIMI_K3_SPECIAL_TOKENS
        } else {
            KIMI_K2_SPECIAL_TOKENS
        };
        let entries = defaults
            .iter()
            .map(|(content, id)| ((*content).to_owned(), *id))
            .collect();
        Self::new(entries, base_token_count)
    }

    /// Reads the `added_tokens_decoder` section of a `tokenizer_config.json` document.
    ///
    /// `Ok(None)` means the document declares no usable entry, so the caller should fall back to [`TiktokenSpecials::for_model`]; a malformed document is [`Error::Validation`].
    /// Individual entries without a string `content` or an integer id are skipped, as Python does.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `json` is not valid JSON.
    pub fn from_config_json(json: &str, base_token_count: usize) -> Result<Option<Self>> {
        let document: serde_json::Value = serde_json::from_str(json).map_err(|error| {
            Error::Validation(format!("tokenizer_config.json is malformed: {error}"))
        })?;
        let Some(decoder) = document
            .get("added_tokens_decoder")
            .and_then(serde_json::Value::as_object)
        else {
            return Ok(None);
        };
        let mut entries = Vec::with_capacity(decoder.len());
        for (raw_id, item) in decoder {
            let Some(content) = item.get("content").and_then(serde_json::Value::as_str) else {
                continue;
            };
            if content.is_empty() {
                continue;
            }
            let Ok(id) = raw_id.parse::<u32>() else {
                continue;
            };
            entries.push((content.to_owned(), id));
        }
        if entries.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self::new(entries, base_token_count)))
    }

    /// Builds the sorted tables from a content→id map.
    fn from_map(by_text: HashMap<String, u32>) -> Self {
        let mut entries: Vec<(String, u32)> = by_text
            .iter()
            .map(|(content, id)| (content.clone(), *id))
            .collect();
        // Longest first, so the first match at a position is the longest one.
        entries.sort_by_key(|(content, _)| std::cmp::Reverse(content.chars().count()));
        let by_id = by_text
            .iter()
            .map(|(content, id)| (*id, content.clone()))
            .collect();
        let starts = by_text
            .keys()
            .filter_map(|content| content.chars().next())
            .collect();
        Self {
            entries,
            by_text,
            by_id,
            starts,
        }
    }

    /// The number of declared special tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no special token is declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The id declared for `content`.
    #[must_use]
    pub fn token_id(&self, content: &str) -> Option<u32> {
        self.by_text.get(content).copied()
    }

    /// The content declared for `id`.
    #[must_use]
    pub fn text(&self, id: u32) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    /// Every declared content, sorted.
    #[must_use]
    pub fn texts(&self) -> Vec<&str> {
        let mut texts: Vec<&str> = self
            .entries
            .iter()
            .map(|(content, _)| content.as_str())
            .collect();
        texts.sort_unstable();
        texts
    }

    /// The longest special token matching at `position`, if any.
    fn match_at(&self, text: &str, position: usize) -> Option<(u32, usize)> {
        let first = text[position..].chars().next()?;
        if !self.starts.contains(&first) {
            return None;
        }
        self.entries.iter().find_map(|(content, id)| {
            text[position..]
                .starts_with(content.as_str())
                .then_some((*id, position + content.len()))
        })
    }
}

/// A loaded `tiktoken` `.model` vocabulary plus its special tokens and pre-tokenization rule.
///
/// The type is immutable, `Send` and `Sync`; loading happens once per context/model binding and every [`TiktokenModel::encode`] afterwards is pure computation over the in-memory tables.
#[derive(Clone)]
pub struct TiktokenModel {
    /// Token bytes to rank, for the merge loop.
    ranks: HashMap<Vec<u8>, u32>,
    /// Rank to token bytes, for decoding.
    tokens: HashMap<u32, Vec<u8>>,
    /// The special-token table.
    specials: TiktokenSpecials,
    /// The pre-tokenization rule.
    pattern: PretokenizePattern,
    /// The longest mergeable token, in bytes.
    max_token_bytes: usize,
    /// Everything the load could not reproduce exactly.
    warnings: Vec<String>,
}

impl std::fmt::Debug for TiktokenModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TiktokenModel")
            .field("mergeable_tokens", &self.ranks.len())
            .field("special_tokens", &self.specials.len())
            .field("pattern", &self.pattern)
            .field("max_token_bytes", &self.max_token_bytes)
            .field("warnings", &self.warnings)
            .finish_non_exhaustive()
    }
}

impl TiktokenModel {
    /// Parses a `.model` document held in memory, using the Kimi rule and no special tokens.
    ///
    /// Blank lines and lines starting with `#` are ignored. Attach special tokens with [`TiktokenModel::with_special_tokens`], or load a file with [`TiktokenModel::from_file`] to read the sibling `tokenizer_config.json` and the built-in Kimi fallback.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the 1-based line number for a malformed entry, a base64 failure, a non-integer rank, a repeated rank or token, and when the document declares no entry at all.
    pub fn from_text(text: &str) -> Result<Self> {
        let mut ranks: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut tokens: HashMap<u32, Vec<u8>> = HashMap::new();
        for (index, raw) in text.lines().enumerate() {
            let line_number = index + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (encoded, rank_text) = parse_line(line, line_number)?;
            let token = decode_token(encoded, line_number)?;
            let rank = parse_rank(rank_text, line_number)?;
            if ranks.contains_key(token.as_slice()) {
                return Err(Error::Validation(format!(
                    "tiktoken model line {line_number}: duplicate token"
                )));
            }
            if tokens.contains_key(&rank) {
                return Err(Error::Validation(format!(
                    "tiktoken model line {line_number}: duplicate rank {rank}"
                )));
            }
            ranks.insert(token.clone(), rank);
            tokens.insert(rank, token);
        }
        if ranks.is_empty() {
            return Err(Error::Validation(
                "tiktoken model vocabulary is empty".to_owned(),
            ));
        }
        let mut warnings = Vec::new();
        let missing = (0_u16..=255)
            .filter(|value| {
                let byte = u8::try_from(*value).unwrap_or(0);
                !ranks.contains_key(std::slice::from_ref(&byte))
            })
            .count();
        if missing > 0 {
            warnings.push(format!(
                "{missing} single-byte tokens are missing; text using them is skipped while encoding"
            ));
        }
        let max_token_bytes = ranks.keys().map(Vec::len).max().unwrap_or(0);
        Ok(Self {
            ranks,
            tokens,
            specials: TiktokenSpecials::default(),
            pattern: PretokenizePattern::Kimi,
            max_token_bytes,
            warnings,
        })
    }

    /// Reads a `.model` file and attaches its special tokens.
    ///
    /// The sibling `tokenizer_config.json` wins when it declares an `added_tokens_decoder` entry; otherwise the built-in Kimi map is used, with K3 selected when the file stem contains `kimi-k3`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the file does not exist and [`Error::Validation`] when it cannot be read or parsed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with_model(path, "")
    }

    /// Reads a `.model` file and resolves its special tokens for `model`.
    ///
    /// `model` is the tokenizer model id (for example `moonshotai/Kimi-K3`); a name containing `kimi-k3` selects the K3 fallback map, anything else K2.
    /// An empty name falls back to the file stem, and the sibling `tokenizer_config.json` always wins when it declares tokens.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when the file does not exist and [`Error::Validation`] when it cannot be read or parsed.
    pub fn from_file_with_model(path: impl AsRef<Path>, model: &str) -> Result<Self> {
        let path = path.as_ref();
        let text = read_model_text(path)?;
        let mut vocabulary = Self::from_text(&text)?;
        vocabulary.specials = load_specials(path, model, vocabulary.mergeable_count());
        Ok(vocabulary)
    }

    /// Replaces the pre-tokenization rule.
    #[must_use]
    pub fn with_pattern(mut self, pattern: PretokenizePattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Replaces the special-token table.
    #[must_use]
    pub fn with_special_tokens(mut self, specials: TiktokenSpecials) -> Self {
        self.specials = specials;
        self
    }

    /// The pre-tokenization rule in use.
    #[must_use]
    pub const fn pattern(&self) -> PretokenizePattern {
        self.pattern
    }

    /// The special-token table.
    #[must_use]
    pub const fn special_tokens(&self) -> &TiktokenSpecials {
        &self.specials
    }

    /// The id declared for the exact special-token text `content`.
    #[must_use]
    pub fn special_token_id(&self, content: &str) -> Option<u32> {
        self.specials.token_id(content)
    }

    /// The number of mergeable vocabulary entries, without special tokens.
    #[must_use]
    pub fn mergeable_count(&self) -> usize {
        self.ranks.len()
    }

    /// The number of declared ids: mergeable entries plus special tokens, like `tiktoken`'s `n_vocab`.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.ranks.len() + self.specials.len()
    }

    /// The longest mergeable token, in bytes.
    #[must_use]
    pub const fn max_token_bytes(&self) -> usize {
        self.max_token_bytes
    }

    /// The bytes of mergeable token `id`.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(&id).map(Vec::as_slice)
    }

    /// Everything the load could not reproduce exactly, in the order it was found.
    #[must_use]
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Encodes `text` into token ids.
    ///
    /// Special tokens are matched first and always become exactly one id (`allowed_special="all"` parity); the text between them is pre-tokenized and merged.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if text.is_empty() {
            return ids;
        }
        if self.specials.is_empty() {
            self.encode_plain(text, &mut ids);
            return ids;
        }
        let mut plain_start = 0;
        let mut cursor = 0;
        while cursor < text.len() {
            if let Some((id, end)) = self.specials.match_at(text, cursor) {
                if plain_start < cursor {
                    self.encode_plain(&text[plain_start..cursor], &mut ids);
                }
                ids.push(id);
                cursor = end;
                plain_start = end;
            } else {
                cursor += text[cursor..].chars().next().map_or(1, char::len_utf8);
            }
        }
        if plain_start < text.len() {
            self.encode_plain(&text[plain_start..], &mut ids);
        }
        ids
    }

    /// Decodes token ids by concatenating their bytes, replacing invalid UTF-8 lossily.
    ///
    /// Special tokens decode to their exact text; ids the vocabulary does not declare are skipped.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(token) = self.tokens.get(id) {
                bytes.extend_from_slice(token);
            } else if let Some(text) = self.specials.text(*id) {
                bytes.extend_from_slice(text.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Splits `text` into the pieces the merge loop sees, without encoding them.
    ///
    /// The split is the configured rule's; the result always covers the whole input, one character at a time for a character no alternative matches.
    #[must_use]
    pub fn pretokenize(&self, text: &str) -> Vec<String> {
        self.pretokenize_refs(text)
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    /// Encodes `text` ignoring special tokens.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in self.pretokenize_refs(text) {
            self.encode_piece(piece.as_bytes(), out);
        }
    }

    /// Splits `text` into borrowed pre-tokenized pieces.
    fn pretokenize_refs<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let chars: Vec<char> = text.chars().collect();
        let mut bounds = Vec::with_capacity(chars.len() + 1);
        let mut offset = 0;
        for ch in &chars {
            bounds.push(offset);
            offset += ch.len_utf8();
        }
        bounds.push(offset);
        let mut pieces = Vec::new();
        let mut position = 0;
        while position < chars.len() {
            let end = self
                .pattern
                .match_len(&chars, position)
                .unwrap_or(position + 1)
                .max(position + 1)
                .min(chars.len());
            pieces.push(&text[bounds[position]..bounds[end]]);
            position = end;
        }
        pieces
    }

    /// Runs the byte-pair merge loop over one pre-tokenized piece.
    fn encode_piece(&self, bytes: &[u8], out: &mut Vec<u32>) {
        if bytes.is_empty() {
            return;
        }
        let mut symbols: Vec<(Vec<u8>, u32)> = Vec::with_capacity(bytes.len());
        for byte in bytes {
            let single = [*byte];
            let Some(id) = self.ranks.get(single.as_slice()) else {
                continue;
            };
            symbols.push((single.to_vec(), *id));
        }
        while symbols.len() > 1 {
            let Some(index) = self.best_pair(&symbols) else {
                break;
            };
            let left = symbols[index].0.clone();
            let right = symbols[index + 1].0.clone();
            let mut joined = Vec::with_capacity(left.len() + right.len());
            joined.extend_from_slice(&left);
            joined.extend_from_slice(&right);
            let Some(id) = self.ranks.get(joined.as_slice()).copied() else {
                break;
            };
            let mut cursor = 0;
            while cursor + 1 < symbols.len() {
                if symbols[cursor].0 == left && symbols[cursor + 1].0 == right {
                    symbols[cursor].0.clone_from(&joined);
                    symbols[cursor].1 = id;
                    symbols.remove(cursor + 1);
                }
                cursor += 1;
            }
        }
        out.extend(symbols.iter().map(|(_, id)| *id));
    }

    /// The index of the adjacent pair whose concatenation has the lowest rank.
    fn best_pair(&self, symbols: &[(Vec<u8>, u32)]) -> Option<usize> {
        let mut best: Option<(u32, usize)> = None;
        for index in 0..symbols.len().saturating_sub(1) {
            let mut joined =
                Vec::with_capacity(symbols[index].0.len() + symbols[index + 1].0.len());
            joined.extend_from_slice(&symbols[index].0);
            joined.extend_from_slice(&symbols[index + 1].0);
            let Some(rank) = self.ranks.get(joined.as_slice()).copied() else {
                continue;
            };
            if best.is_none_or(|(current, _)| rank < current) {
                best = Some((rank, index));
            }
        }
        best.map(|(_, index)| index)
    }
}

/// Splits one non-empty `.model` line into its two fields.
fn parse_line(line: &str, line_number: usize) -> Result<(&str, &str)> {
    let mut fields = line.split_whitespace();
    let (Some(encoded), Some(rank)) = (fields.next(), fields.next()) else {
        return Err(Error::Validation(format!(
            "tiktoken model line {line_number}: expected `<base64-token> <rank>`"
        )));
    };
    if fields.next().is_some() {
        return Err(Error::Validation(format!(
            "tiktoken model line {line_number}: unexpected extra fields"
        )));
    }
    Ok((encoded, rank))
}

/// Decodes the base64 token of one `.model` line.
fn decode_token(encoded: &str, line_number: usize) -> Result<Vec<u8>> {
    let token = BASE64_STANDARD.decode(encoded).map_err(|error| {
        Error::Validation(format!(
            "tiktoken model line {line_number}: token is not valid base64 ({error})"
        ))
    })?;
    if token.is_empty() {
        return Err(Error::Validation(format!(
            "tiktoken model line {line_number}: token decodes to no bytes"
        )));
    }
    Ok(token)
}

/// Parses the rank of one `.model` line.
fn parse_rank(rank: &str, line_number: usize) -> Result<u32> {
    rank.parse::<u32>().map_err(|_| {
        Error::Validation(format!(
            "tiktoken model line {line_number}: rank `{rank}` is not an integer"
        ))
    })
}

/// Reads the `.model` document.
fn read_model_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            Error::NotFound(format!(
                "tiktoken model artifact not found: {}",
                path.display()
            ))
        } else {
            Error::Validation(format!(
                "tiktoken model artifact {} could not be read: {error}",
                path.display()
            ))
        }
    })
}

/// Resolves the special-token table for a loaded `.model` file.
///
/// The sibling `tokenizer_config.json` wins; otherwise the built-in map is selected by `model`, or by the file stem when `model` is blank.
fn load_specials(path: &Path, model: &str, base_token_count: usize) -> TiktokenSpecials {
    let stem = path
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or_default();
    let config = path.parent().map_or_else(
        || PathBuf::from("tokenizer_config.json"),
        |parent| parent.join("tokenizer_config.json"),
    );
    if config.is_file() {
        match std::fs::read_to_string(&config) {
            Ok(text) => match TiktokenSpecials::from_config_json(&text, base_token_count) {
                Ok(Some(specials)) => return specials,
                Ok(None) => tracing::debug!(
                    target: "openjiuwen_context::tokenizers::native",
                    path = %config.display(),
                    "tokenizer_config.json declares no added tokens; using the built-in fallback"
                ),
                Err(error) => tracing::warn!(
                    target: "openjiuwen_context::tokenizers::native",
                    %error,
                    path = %config.display(),
                    "tokenizer_config.json is malformed; using the built-in fallback"
                ),
            },
            Err(error) => tracing::warn!(
                target: "openjiuwen_context::tokenizers::native",
                %error,
                path = %config.display(),
                "tokenizer_config.json could not be read; using the built-in fallback"
            ),
        }
    }
    let name = if model.trim().is_empty() { stem } else { model };
    TiktokenSpecials::for_model(name, base_token_count)
}

/// Whether `text` starts like a `tiktoken` `.model` document rather than a JSON artifact.
///
/// The first meaningful line must be `<base64-token> <integer>`; comments and blank lines are skipped.
pub(crate) fn looks_like_tiktoken_model(text: &str) -> bool {
    for raw in text.lines().take(64) {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let (Some(encoded), Some(rank)) = (fields.next(), fields.next()) else {
            return false;
        };
        if fields.next().is_some() || rank.parse::<u32>().is_err() {
            return false;
        }
        return BASE64_STANDARD.decode(encoded).is_ok();
    }
    false
}

impl PretokenizePattern {
    /// The absolute end index of the first match at `start`, or `None` when no alternative matches.
    fn match_len(self, chars: &[char], start: usize) -> Option<usize> {
        match self {
            Self::Kimi => kimi_match(chars, start),
            Self::Gpt2 => gpt2_match(chars, start),
        }
    }
}

/// One alternative of a rule, returning the absolute end index.
type Matcher = fn(&[char], usize) -> Option<usize>;

/// Runs the alternatives in order, returning the first match.
fn first_match(chars: &[char], start: usize, alternatives: &[Matcher]) -> Option<usize> {
    alternatives
        .iter()
        .find_map(|alternative| alternative(chars, start))
}

/// The Kimi alternation, in the order of `KIMI_TIKTOKEN_PAT_STR`.
fn kimi_match(chars: &[char], start: usize) -> Option<usize> {
    first_match(
        chars,
        start,
        &[
            han_run,
            kimi_lowercase_word,
            kimi_uppercase_word,
            digit_run_kimi,
            symbol_run,
            whitespace_to_newline,
            trailing_whitespace,
            whitespace_run,
        ],
    )
}

/// The GPT-2/`cl100k_base` alternation.
fn gpt2_match(chars: &[char], start: usize) -> Option<usize> {
    first_match(
        chars,
        start,
        &[
            contraction,
            gpt2_letters,
            gpt2_numbers,
            gpt2_symbols,
            trailing_whitespace,
            whitespace_run,
        ],
    )
}

/// Whether `ch` belongs to the Han script, over the unified-ideograph and compatibility ranges.
fn is_han(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x2E80..=0x2EFF
            | 0x2F00..=0x2FDF
            | 0x3005
            | 0x3007
            | 0x3021..=0x3029
            | 0x3038..=0x303B
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2EBEF
            | 0x2F800..=0x2FA1F
            | 0x30000..=0x323AF
    )
}

/// Whether `ch` is a combining mark (`\p{M}`-like) in one of the common blocks.
fn is_mark(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x0591..=0x05BD
            | 0x05BF
            | 0x05C1..=0x05C2
            | 0x0610..=0x061A
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06DC
            | 0x0730..=0x074A
            | 0x07A6..=0x07B0
            | 0x0900..=0x0903
            | 0x093A..=0x094F
            | 0x0951..=0x0957
            | 0x0962..=0x0963
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20F0
            | 0x302A..=0x302F
            | 0x3099..=0x309A
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
    )
}

/// `\p{L}` without Han: alphabetic characters outside the Han script.
fn is_letter(ch: char) -> bool {
    ch.is_alphabetic() && !is_han(ch)
}

/// `\p{N}`: any numeric character.
fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`-like: a non-lowercase letter or a mark.
fn is_upper_like(ch: char) -> bool {
    is_mark(ch) || (is_letter(ch) && !ch.is_lowercase())
}

/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`-like: a lowercase letter or a mark.
fn is_lower_like(ch: char) -> bool {
    is_mark(ch) || (is_letter(ch) && ch.is_lowercase())
}

/// `[^\r\n\p{L}\p{N}]`: a character a word run may carry in front of itself.
fn is_word_lead(ch: char) -> bool {
    ch != '\r' && ch != '\n' && !ch.is_alphabetic() && !is_number(ch)
}

/// `[^\s\p{L}\p{N}]`: a character of a symbol run.
fn is_symbol(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphabetic() && !is_number(ch)
}

/// `[\r\n]`: a line break.
fn is_newline(ch: char) -> bool {
    ch == '\r' || ch == '\n'
}

/// `[\p{Han}]+`.
fn han_run(chars: &[char], start: usize) -> Option<usize> {
    if !is_han(chars[start]) {
        return None;
    }
    let mut end = start;
    while end < chars.len() && is_han(chars[end]) {
        end += 1;
    }
    Some(end)
}

/// `'s|'t|'re|'ve|'m|'ll|'d`, case-insensitively.
fn contraction(chars: &[char], start: usize) -> Option<usize> {
    if chars.get(start) != Some(&'\'') {
        return None;
    }
    ["s", "t", "re", "ve", "m", "ll", "d"]
        .iter()
        .find_map(|suffix| {
            let mut position = start + 1;
            for expected in suffix.chars() {
                if chars
                    .get(position)
                    .is_none_or(|ch| !ch.eq_ignore_ascii_case(&expected))
                {
                    return None;
                }
                position += 1;
            }
            Some(position)
        })
}

/// `[^\r\n\p{L}\p{N}]? [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]* [\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+ contraction?`.
fn kimi_lowercase_word(chars: &[char], start: usize) -> Option<usize> {
    for lead in [true, false] {
        let mut position = start;
        if lead {
            if !is_word_lead(chars[start]) {
                continue;
            }
            position += 1;
        }
        let upper_start = position;
        while position < chars.len() && is_upper_like(chars[position]) {
            position += 1;
        }
        let upper_count = position - upper_start;
        for available in (0..=upper_count).rev() {
            let lower_start = upper_start + available;
            if lower_start >= chars.len() || !is_lower_like(chars[lower_start]) {
                continue;
            }
            let mut end = lower_start;
            while end < chars.len() && is_lower_like(chars[end]) {
                end += 1;
            }
            return Some(contraction(chars, end).unwrap_or(end));
        }
    }
    None
}

/// `[^\r\n\p{L}\p{N}]? [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+ [\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]* contraction?`.
fn kimi_uppercase_word(chars: &[char], start: usize) -> Option<usize> {
    for lead in [true, false] {
        let mut position = start;
        if lead {
            if !is_word_lead(chars[start]) {
                continue;
            }
            position += 1;
        }
        let upper_start = position;
        while position < chars.len() && is_upper_like(chars[position]) {
            position += 1;
        }
        if position == upper_start {
            continue;
        }
        while position < chars.len() && is_lower_like(chars[position]) {
            position += 1;
        }
        return Some(contraction(chars, position).unwrap_or(position));
    }
    None
}

/// `\p{N}{1,3}`.
fn digit_run_kimi(chars: &[char], start: usize) -> Option<usize> {
    if !is_number(chars[start]) {
        return None;
    }
    let mut end = start;
    while end < chars.len() && end - start < 3 && is_number(chars[end]) {
        end += 1;
    }
    Some(end)
}

/// ` ?[^\s\p{L}\p{N}]+[\r\n]*`.
fn symbol_run(chars: &[char], start: usize) -> Option<usize> {
    for lead in [true, false] {
        let mut position = start;
        if lead {
            if chars[start] != ' ' {
                continue;
            }
            position += 1;
        }
        let symbol_start = position;
        while position < chars.len() && is_symbol(chars[position]) {
            position += 1;
        }
        if position == symbol_start {
            continue;
        }
        while position < chars.len() && is_newline(chars[position]) {
            position += 1;
        }
        return Some(position);
    }
    None
}

/// `\s*[\r\n]+`.
fn whitespace_to_newline(chars: &[char], start: usize) -> Option<usize> {
    let mut end = start;
    let mut last_newline = None;
    while end < chars.len() && chars[end].is_whitespace() {
        if is_newline(chars[end]) {
            last_newline = Some(end);
        }
        end += 1;
    }
    last_newline.map(|index| index + 1)
}

/// `\s+(?!\S)`.
fn trailing_whitespace(chars: &[char], start: usize) -> Option<usize> {
    let mut end = start;
    while end < chars.len() && chars[end].is_whitespace() {
        end += 1;
    }
    let run = end - start;
    if run == 0 {
        return None;
    }
    if end == chars.len() {
        return Some(end);
    }
    // `\s+` backtracks one character: the next one is then whitespace, so the
    // lookahead succeeds.
    (run > 1).then_some(end - 1)
}

/// `\s+`.
fn whitespace_run(chars: &[char], start: usize) -> Option<usize> {
    let mut end = start;
    while end < chars.len() && chars[end].is_whitespace() {
        end += 1;
    }
    (end > start).then_some(end)
}

/// ` ?\p{L}+`.
fn gpt2_letters(chars: &[char], start: usize) -> Option<usize> {
    optional_space_run(chars, start, char::is_alphabetic)
}

/// ` ?\p{N}+`.
fn gpt2_numbers(chars: &[char], start: usize) -> Option<usize> {
    optional_space_run(chars, start, is_number)
}

/// ` ?[^\s\p{L}\p{N}]+`.
fn gpt2_symbols(chars: &[char], start: usize) -> Option<usize> {
    optional_space_run(chars, start, is_symbol)
}

/// Matches ` ?` followed by one or more `class` characters, trying the space first.
fn optional_space_run(chars: &[char], start: usize, class: impl Fn(char) -> bool) -> Option<usize> {
    for lead in [true, false] {
        let mut position = start;
        if lead {
            if chars[start] != ' ' {
                continue;
            }
            position += 1;
        }
        let run_start = position;
        while position < chars.len() && class(chars[position]) {
            position += 1;
        }
        if position > run_start {
            return Some(position);
        }
    }
    None
}
