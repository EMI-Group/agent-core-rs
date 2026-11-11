# src/parsers — model-output parsers

## Intent

Turn the raw output of a moderation/jailbreak model into a `RiskAssessment`.
Ports `openjiuwen/core/security/guardrail/context.py` (`ModelOutputParser`, `BertBinaryParser`, `QwenGuardParser`) with the same accepted dialects and thresholds, because deployments point these parsers at fixed model endpoints.
Pure and synchronous: parsing never performs I/O, so each parser is a plain `fn parse(&self, output: &Value) -> RiskAssessment`.

## API Surface

```rust
pub trait ModelOutputParser: Send + Sync {
    fn name(&self) -> &str;                                  // "bert_binary" / "qwen_guard"
    fn parse(&self, output: &Value) -> RiskAssessment;
}
pub struct BertBinaryParser { pub risk_type: String, pub confidence_thresholds: Thresholds, pub attack_class_id: i64 }
pub struct Thresholds { pub low: f64, pub medium: f64, pub high: f64 }   // default 0.7 / 0.85 / 0.95
pub struct QwenGuardParser { pub risk_type: String, pub default_risk_level: RiskLevel }
```
Constructors take the Python defaults (`BertBinaryParser::new()` = `risk_type "attack_detected"`, `attack_class_id 1`; `QwenGuardParser::new()` = `risk_type "content_risk"`, `default_risk_level Safe`) plus builders for thresholds/attack class/risk type.
`parse` is total: unrecognized input yields a `SAFE` assessment, never an error or panic.

## Constraints

- `RiskAssessment`/`RiskLevel` come from the crate root; no new error type.
- No regex at match time for JSON dialects; the Qwen text helpers use static `LazyLock<Regex>` (case-insensitive) with `#[allow(clippy::expect_used)]`.
- Never include the model output in the assessment `details` (only derived fields), so findings stay safe to log.
- Unit tests live beside the parsers (`bert.rs`, `qwen.rs`) and cover the tables below; the crate-level `tests/prompt_injection.rs` exercises them through a guardrail.

## BertBinaryParser semantics (port of `context.py:106`)

`parse` extracts `(predicted_class, confidence)` from the first matching shape, in this order:

1. object with `predicted_class` → `class = as_int(predicted_class)`, `conf = as_f64(confidence, 0.0)`.
2. object with `label` → `class = as_int(label)`, `conf = as_f64(score, as_f64(confidence, 0.0))`.
3. object with `probabilities` (array length ≥ 2) → `class = 1 if probs[1] > probs[0] else 0`, `conf = max(probs)`.
4. object with `logits` (array length ≥ 2) → softmax over all logits (`exp`; if the sum ≤ 0 use `[0.5, 0.5]`), then the same argmax/max rule.
5. top-level array length ≥ 2 → same argmax/max rule.
6. number `v` → `conf = v`, `class = attack_class_id if conf > 0.5 else 0` (strict `>`).
7. anything else → `(0, 0.0)`.

Level mapping: `class != attack_class_id` → `SAFE` regardless of confidence; otherwise `conf < low → SAFE`, `low ≤ conf < medium → LOW`, `medium ≤ conf < high → MEDIUM`, `conf ≥ high → HIGH`.
Emission: `has_risk = level != Safe`; `risk_type = Some(self.risk_type)` only when risky, else `None`; `details = {"predicted_class": class, "attack_confidence": conf if class == attack_class_id else 1 - conf}`; `confidence = conf`.
`attack_class_id = 0` flips the meaning of "positive class" (tested in Python).

## QwenGuardParser semantics (port of `context.py:207`)

Risk map: `"safe" → Safe`, `"controversial" → Medium`, `"unsafe" → High`.
`_map_risk_level(s)`: lowercase+trim; exact key match wins; otherwise the first map key (checked in descending key length: `controversial`, `unsafe`, `safe`) contained in the string; otherwise `default_risk_level`.

Input dispatch:
1. object → pick dialect by first present key: `safety` (standard) → `analysis` (full) → `judgment` (simple) → `severity_level` (api) → generic.
2. string → try JSON (whole string when it starts `{`/ends `}`, else the substring from the first `{` to the last `}`); on success dispatch as an object; otherwise parse the standard text format.
3. other values → `str()` then treat as text.

Dialect details:
- standard: `safety` (default `"safe"`), `categories` (array or comma/whitespace-separated string; `risk_type` = first category else `self.risk_type`), details `{"safety", "categories"}`.
- full: `analysis.risk_level` (default `"safe"`), `analysis.risk_categories[0]` as `risk_type` (fallback `self.risk_type`), details `{"risk_categories", "evidence", "language" (default `"unknown"`), "decision" from `data.decision` (default `"unknown"`), "version" from `data.version` (default `"unknown"`)}`.
- simple: `judgment` (default `"Safe"`), details `{"reason", "language", "judgment"}`; `risk_type` = `self.risk_type` when risky.
- api: `severity_level` (default `"safe"`), details `{"reason", "language", "severity_level"}`.
- generic: `risk_level` or `level` (default `"safe"`), `risk_type` (default `self.risk_type`), details = whole object (redaction happens at report time).

Text extraction: `safety:\s*(\w+)` (case-insensitive), falling back to line splitting on `safety:`, then to the first keyword in `["unsafe", "controversial", "safe"]` found in the text, else `"unknown"`.
`categories?:\s*(.+?)(?:\n|$)` (case-insensitive), also accepting a line keyed `category`; values split on commas/whitespace.
Emission: `has_risk = level != Safe`; `risk_type = Some(...)` only when risky (categories[0] or the dialect's risk type), `confidence = 0.0`, details as above.

## Routing Table

- `./mod.rs` → `ModelOutputParser` trait, `Thresholds`, factory helpers
- `./bert.rs` → `BertBinaryParser` + unit tests for the extraction table
- `./qwen.rs` → `QwenGuardParser` + unit tests for every dialect
