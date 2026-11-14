//! The special-token table of a `tiktoken` vocabulary.
//!
//! The table holds the built-in Kimi K2/K3 maps, the `added_tokens_decoder` reader and the reserved-id padding, and answers the longest-match lookup the encoder uses.

use std::collections::{HashMap, HashSet};

use openjiuwen_common::error::{Error, Result};

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
    pub(super) fn match_at(&self, text: &str, position: usize) -> Option<(u32, usize)> {
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
