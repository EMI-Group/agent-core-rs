# src/parsers — model-output parsers

## Intent

Turn the raw output of a moderation/jailbreak model into a `RiskAssessment`.
Ports `openjiuwen/core/security/guardrail/context.py` (`ModelOutputParser`, `BertBinaryParser`, `QwenGuardParser`) with the same accepted dialects and thresholds, because deployments point these parsers at fixed model endpoints.
Pure and synchronous: parsing performs no I/O and never panics, but it is **fallible** — an unrecognizable reply is an error, never a silent `Safe` grade — and a recognized reply with no usable verdict errs upward instead of grading `Safe`.

## API Surface

```rust
pub trait ModelOutputParser: Send + Sync {
    fn name(&self) -> &str;                                              // "bert_binary" / "qwen_guard"
    fn parse(&self, output: &Value) -> openjiuwen_common::error::Result<RiskAssessment>;
}
pub struct BertBinaryParser { pub risk_type: String, pub confidence_thresholds: Thresholds, pub attack_class_id: i64 }
pub struct Thresholds { pub low: f64, pub medium: f64, pub high: f64 }   // default 0.7 / 0.85 / 0.95
pub struct QwenGuardParser { pub risk_type: String, pub default_risk_level: RiskLevel }
```

Constructors take the Python defaults except for the verdict fallback (`BertBinaryParser::new()` = `risk_type "attack_detected"`, `attack_class_id 1`; `QwenGuardParser::new()` = `risk_type "content_risk"`, `default_risk_level Critical`) plus builders for thresholds/attack class/risk type/default level.

## Error contract

`parse` returns `Err(Error::validation)`, whose `ErrorKind` is `Validation`, when the output matches no accepted dialect.
The message names the parser and its dialects and never echoes the payload: `bert_binary parser: unrecognized model output (expected predicted_class/confidence, label/score, probabilities or logits with at least two values, a two-element score array, or a bare numeric score)` and `qwen_guard parser: unrecognized model output dialect (expected a safety/categories object, an analysis object, a judgment object, a severity_level object, a risk_level object, an embedded JSON object, or a 'Safety:'/'Categories:' text block)`.
Rejected for both parsers: `null`, booleans, empty objects, objects whose keys match no dialect, and free text without a verdict marker.
Rejected additionally: a shortened `probabilities`/`logits` array or a top-level array with fewer than two values (`bert_binary`), and every number or array (`qwen_guard`).
Accepted but unmappable is not safe: a recognized shape whose level *value* is unknown (`safety: "totally-unheard-of"`, `severity_level: "elevated"`) or *missing* (`analysis: {}`, `judgment: null`, `severity_level: ""`, a `Categories:`-only text block) maps to `QwenGuardParser::default_risk_level`, which is `RiskLevel::Critical` by default, so it blocks under the default `block_level = Critical` instead of grading `Safe` like Python.
`QwenGuardParser::with_default_risk_level(RiskLevel::Safe)` is the explicit opt-in for Python's behaviour.
`BertBinaryParser` errs upward too: a predicted attack class whose reply supplies no usable confidence (absent, non-numeric or non-finite) is `RiskLevel::High`, never `Safe`; a predicted safe class stays `Safe`.

## Constraints

- `RiskAssessment`/`RiskLevel` come from the crate root; errors come from `openjiuwen_common::error`; no new error type.
- No regex at match time for JSON dialects; the Qwen text helpers use static `LazyLock<Regex>` (case-insensitive) with `#[allow(clippy::expect_used)]`.
- Never include the model output in the assessment `details` (only derived fields), so findings stay safe to log.
- Unit tests live beside the parsers (`bert.rs`, `qwen.rs`) and cover the tables below, including the failure cases and the no-payload-leak rule.

## BertBinaryParser semantics (port of `context.py:106`)

`extract_prediction` returns `Option<(predicted_class, Option<f64>)>` from the first matching shape, in this order, so "no dialect matched" (outer `None`) is distinguishable from a real `(0, 0.0)` prediction, and "matched but no usable confidence" (inner `None`) from a real `0.0`:

1. object with `predicted_class` → `class = as_int(predicted_class)`, `conf = as_f64(confidence)` or `None` when absent/non-numeric/non-finite.
2. object with `label` → `class = as_int(label)`, `conf = as_f64(score, then confidence)`, same `None` rule.
3. object with `probabilities` (array length ≥ 2) → `class = 1 if probs[1] > probs[0] else 0`, `conf = max(probs)` (always present).
4. object with `logits` (array length ≥ 2) → softmax over all logits (`exp`; if the sum ≤ 0 use `[0.5, 0.5]`), then the same argmax/max rule (always present).
5. top-level array length ≥ 2 → same argmax/max rule (always present).
6. number `v` → `conf = v`, `class = attack_class_id if conf > 0.5 else 0` (strict `>`; the bare number is always a present confidence, `0.0` when non-finite).
7. anything else (including a shortened `probabilities`/`logits`/top-level array) → `None` → `Error::validation`.

Level mapping: `class != attack_class_id` → `SAFE` regardless of confidence; an absent or non-finite confidence for a predicted attack → `HIGH` (errs upward, the level a confident attack gets); otherwise `conf < low → SAFE`, `low ≤ conf < medium → LOW`, `medium ≤ conf < high → MEDIUM`, `conf ≥ high → HIGH`.
Emission: `has_risk = level != Safe`; `risk_type = Some(self.risk_type)` only when risky, else `None`; `details = {"predicted_class": class, "attack_confidence": conf if class == attack_class_id else 1 - conf}` with an absent confidence reported as `0.0` so the key set and values stay stable; `confidence = conf` (`0.0` when absent).
`attack_class_id = 0` flips the meaning of "positive class" (tested in Python).

## QwenGuardParser semantics (port of `context.py:207`)

Risk map (exact, case-insensitive after trim): `"safe" → Safe`, `"low" → Low`, `"controversial"`/`"medium" → Medium`, `"unsafe"`/`"high" → High`, `"critical" → Critical`.
`_map_risk_level(s)`: lowercase+trim; exact key match wins; otherwise the first containment key (`controversial`, `unsafe`, `safe`, checked longest first) contained in the string; otherwise `default_risk_level`.
Containment is deliberately limited to those three words: `low` is a substring of `below`, so the severity words are exact-match only.

Input dispatch:
1. object → pick dialect by first present key: `safety` (standard) → `analysis` (full) → `judgment` (simple) → `severity_level` (api) → `risk_level`/`level` (generic); no such key → `Error::validation`.
2. string → try JSON (whole string when it starts `{`/ends `}`, else the substring from the first `{` to the last `}`); on success dispatch as an object (an embedded `{}` is an error); otherwise parse the standard text format.
3. any other value (`null`, boolean, number, array) → `Error::validation`.

Standard text format: recognized when `extract_safety` finds a verdict or `extract_categories` finds a category; text with neither is `Error::validation`.

Dialect details:
- standard: `safety` (default `""`, i.e. no verdict), `categories` (array or comma/whitespace-separated string; `risk_type` = first category else `self.risk_type`), details `{"safety", "categories"}`.
- full: `analysis.risk_level` (default `""`), `analysis.risk_categories[0]` as `risk_type` (fallback `self.risk_type`), details `{"risk_categories", "evidence", "language" (default `"unknown"`), "decision" from `data.decision` (default `"unknown"`), "version" from `data.version` (default `"unknown"`)}`.
- simple: `judgment` (default `""`), details `{"reason", "language", "judgment"}`; `risk_type` = `self.risk_type` when risky.
- api: `severity_level` (default `""`), details `{"reason", "language", "severity_level"}`.
- generic: `risk_level` or `level` (default `""`), `risk_type` (default `self.risk_type`), details = whole object (redaction happens at report time).

Text extraction: `safety:\s*(\w+)` (case-insensitive), falling back to line splitting on `safety:`, then to the first keyword in `["unsafe", "controversial", "safe"]` found in the text, else `"unknown"` (which is unmappable, so it takes `default_risk_level`).
`categories?:\s*(.+?)(?:\n|$)` (case-insensitive), also accepting a line keyed `category`; values split on commas/whitespace.
Emission: `has_risk = level != Safe`; `risk_type = Some(...)` only when risky (categories[0] or the dialect's risk type), `confidence = 0.0`, details as above.

## Routing Table

- `./mod.rs` → `ModelOutputParser` trait, `Thresholds`, factory helpers
- `./bert.rs` → `BertBinaryParser` + unit tests for the extraction table, the missing-confidence upward error and the unrecognized-output errors
- `./qwen.rs` → `QwenGuardParser` + unit tests for every dialect, the unmappable/missing-verdict upward error and the unrecognized-output errors
