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

mod pretokenize;
mod specials;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use base64::prelude::{Engine as _, BASE64_STANDARD};
use openjiuwen_common::error::{Error, Result};

pub use pretokenize::PretokenizePattern;
pub use specials::TiktokenSpecials;

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
