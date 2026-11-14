//! `SwarmFlow` script generation.
//!
//! Ports Python `openjiuwen.symphony.flow.codegen`: a recipe's capability
//! combination is mechanically translated into a `SwarmFlow` Python script with one
//! `await agent(...)` call per capability in topological order. The only allowed
//! operator is `agent` from `swarmflow`, so the generated script has a fixed
//! shape that the static review gate can check without executing anything.
//!
//! # Script shape
//!
//! ```text
//! """SwarmFlow script generated from recipe <recipe_id>."""
//!
//! from swarmflow import agent
//!
//! META = {'name': '<meta-name>', 'phases': ['step_1', 'step_2']}
//!
//! TASK = '<task description>'
//!
//! async def run(args):
//!     """按 skill pack 拓扑序执行能力组合。"""
//!     step_1 = await agent('...', label='<id>', phase='step_1')
//!     step_2 = await agent('...', label='<id>', phase='step_2')
//!     return step_2
//! ```
//!
//! Every interpolated value goes through [`python_string_literal`], a
//! `repr`-like single-quote escaper that keeps UTF-8 intact (Python's
//! `ensure_ascii=False` behaviour). Rendering is deterministic: the Kahn
//! topological order emits ready nodes by sorted id and a step's predecessors are
//! sorted, so the same recipe always yields the same bytes.
//!
//! # Validator approximation
//!
//! [`validate_generated_script`] is a line-oriented approximation of Python's
//! `ast`-based whitelist, not a Python parser. It masks string literals and
//! comments, then checks bracket/quote balance, the `META`/`TASK`/`run` markers,
//! plain and disallowed `from … import` statements, disallowed calls, and the
//! review gate's forbidden code tokens. A syntax error Python's AST would report
//! inside an expression (for example a missing colon) is not detected here; the
//! generated shape never contains one.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{validation, Error, Result};
pub use crate::flow::model::normalize_meta_name;
use crate::flow::model::{content_hash, topological_order, ExperienceRecipe, TraceNode};

/// The module a generated script may import from.
pub const SWARMFLOW_MODULE: &str = "swarmflow";
/// The only operator a generated script may import and call.
pub const ALLOWED_OPERATOR: &str = "agent";
/// The artifact path of a generated script inside a package.
pub const SCRIPT_FILENAME: &str = "swarmflow/run.py";
/// The canonical-materials key holding the generated script.
pub const SCRIPT_MATERIAL_KEY: &str = "swarmflow_script";
/// The canonical-materials key holding the recipe's trigger conditions.
pub const TRIGGER_CONDITIONS_KEY: &str = "trigger_conditions";
/// The label used when a recipe node declares no description.
pub const DEFAULT_STEP_DESCRIPTION: &str = "协作成员能力";
/// The `meta.name` fallback used when a recipe has no usable name.
pub const META_NAME_FALLBACK: &str = "skill-pack";
/// The generated identifier prefix (`combo-<12 hex>`).
pub const IDENTIFIER_PREFIX: &str = "combo-";
/// How many hex characters of the materials hash name the combination.
pub const IDENTIFIER_HASH_LENGTH: usize = 12;
/// The recipe-id prefix stripped when deriving the identifier fallback.
pub const RECIPE_ID_PREFIX: &str = "recipe_";

/// Words that begin a Python statement rather than a call expression.
///
/// Python's AST sees `def run(args)` as a function definition and `if (x)` as a
/// parenthesised condition, so the line-oriented validator must not read them as
/// calls to `run` or `if`.
const NON_CALL_KEYWORDS: [&str; 21] = [
    "def", "class", "if", "elif", "else", "for", "while", "return", "with", "except", "assert",
    "del", "raise", "lambda", "yield", "not", "in", "is", "and", "or", "import",
];

/// One generated `SwarmFlow` script plus the values it interpolates.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SwarmflowScript {
    /// The `meta.name` identifier (`combo-<12 hex>` or a normalized recipe name).
    pub identifier: String,
    /// The rendered Python source.
    pub source: String,
    /// The `META` literal as data (`{"name", "phases"}`).
    pub meta: Value,
    /// The task description assigned to `TASK`.
    pub task: String,
    /// The capability ids in topological order.
    pub steps: Vec<String>,
}

/// Derives Python's `combo-<12 hex>` identifier from the canonical materials.
///
/// The hash is the crate's content hash of `materials` (canonical JSON), and
/// `recipe_id` supplies the fallback used when the raw identifier normalizes to
/// nothing. The result always satisfies [`is_valid_meta_name`](crate::flow::model::is_valid_meta_name).
#[must_use]
pub fn swarmflow_identifier(materials: &Value, recipe_id: &str) -> String {
    let digest = content_hash(materials);
    let short: String = digest.chars().take(IDENTIFIER_HASH_LENGTH).collect();
    let stripped = recipe_id
        .strip_prefix(RECIPE_ID_PREFIX)
        .unwrap_or(recipe_id);
    let fallback = if stripped.is_empty() {
        "combo"
    } else {
        stripped
    };
    normalize_meta_name(&format!("{IDENTIFIER_PREFIX}{short}"), fallback)
}

/// Translates a recipe into a `SwarmFlow` script.
///
/// Steps follow [`crate::flow::model::topological_order`],
/// each `await agent(...)` prompt names the capability, its description, and the
/// earlier steps that feed it, and `run(args)` returns the last step. An empty
/// combination still produces a valid script whose `run` returns `None`.
///
/// Python's packager passes no member descriptions, so every prompt falls back to
/// `协作成员能力`; this port reads each recipe node's description instead and uses
/// that same fallback only when a node declares none.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error) when the combination is
/// cyclic, so no complete topological order exists.
pub fn generate_swarmflow_script(recipe: &ExperienceRecipe) -> Result<SwarmflowScript> {
    let structure = &recipe.combination_structure;
    let order = topological_order(structure);
    if order.len() != structure.nodes.len() {
        return Err(validation(
            "recipe.combination_structure",
            "capability combination is not acyclic",
        ));
    }
    let known: BTreeSet<&str> = order.iter().map(String::as_str).collect();
    let mut upstream: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for edge in &structure.edges {
        if known.contains(edge.source.as_str()) && known.contains(edge.target.as_str()) {
            upstream
                .entry(edge.target.clone())
                .or_default()
                .insert(edge.source.clone());
        }
    }
    let identifier = if recipe.name.trim().is_empty() {
        let materials = serde_json::to_value(structure).map_err(crate::error::Error::external)?;
        swarmflow_identifier(&materials, &recipe.recipe_id)
    } else {
        normalize_meta_name(&recipe.name, META_NAME_FALLBACK)
    };
    let phases: Vec<String> = (1..=order.len())
        .map(|index| format!("step_{index}"))
        .collect();
    let meta = serde_json::json!({"name": identifier, "phases": phases});
    let task = recipe.applicability_text("task_description").to_string();

    let mut lines: Vec<String> = Vec::new();
    lines.push(python_docstring(&format!(
        "SwarmFlow script generated from recipe {}.",
        recipe.recipe_id
    )));
    lines.push(String::new());
    lines.push(format!("from {SWARMFLOW_MODULE} import {ALLOWED_OPERATOR}"));
    lines.push(String::new());
    lines.push(format!("META = {}", meta_literal(&identifier, &phases)));
    lines.push(String::new());
    lines.push(format!("TASK = {}", python_string_literal(&task)));
    lines.push(String::new());
    lines.push("async def run(args):".to_string());
    if order.is_empty() {
        lines.push("    return None".to_string());
        lines.push(String::new());
        return Ok(SwarmflowScript {
            identifier,
            source: lines.join("\n"),
            meta,
            task,
            steps: order,
        });
    }
    lines.push("    \"\"\"按 skill pack 拓扑序执行能力组合。\"\"\"".to_string());
    for (index, capability_id) in order.iter().enumerate() {
        let step = index + 1;
        let description = step_description(recipe, capability_id);
        let inputs: Vec<String> = upstream
            .get(capability_id)
            .map(|sources| sources.iter().cloned().collect())
            .unwrap_or_default();
        let upstream_text = if inputs.is_empty() {
            "。输入来自任务参数".to_string()
        } else {
            format!("。上一步产出：{}", inputs.join(" + "))
        };
        let prompt =
            format!("使用能力 {capability_id}（{description}）完成本阶段任务{upstream_text}。");
        let phase = format!("step_{step}");
        lines.push(format!(
            "    step_{step} = await {ALLOWED_OPERATOR}({}, label={}, phase={})",
            python_string_literal(&prompt),
            python_string_literal(capability_id),
            python_string_literal(&phase),
        ));
    }
    lines.push(format!("    return step_{}", order.len()));
    lines.push(String::new());
    Ok(SwarmflowScript {
        identifier,
        source: lines.join("\n"),
        meta,
        task,
        steps: order,
    })
}

/// Adds the optional script keys to a pack's canonical materials.
///
/// The generated script enters under [`SCRIPT_MATERIAL_KEY`] and the recipe's
/// trigger conditions under `trigger_conditions`, each only when present, so a
/// pack without them keeps its previous content hash.
pub(crate) fn extend_materials(
    materials: &mut Value,
    script: Option<&str>,
    trigger_conditions: &str,
) {
    let Some(object) = materials.as_object_mut() else {
        return;
    };
    if let Some(script) = script {
        object.insert(
            SCRIPT_MATERIAL_KEY.to_string(),
            Value::String(script.to_string()),
        );
    }
    if !trigger_conditions.trim().is_empty() {
        object.insert(
            TRIGGER_CONDITIONS_KEY.to_string(),
            Value::String(trigger_conditions.to_string()),
        );
    }
}

/// Adds the generated script to a pack's rendered artifacts, when present.
pub(crate) fn insert_script_artifact(
    artifacts: &mut BTreeMap<String, String>,
    script: Option<&str>,
) {
    if let Some(script) = script {
        artifacts.insert(SCRIPT_FILENAME.to_string(), script.to_string());
    }
}

/// Reads the optional script artifact of a saved pack.
///
/// A missing file is `None` (a pack without a generated script); every other
/// I/O failure maps to [`crate::error::Error::External`].
pub(crate) async fn read_script_artifact(directory: &Path) -> Result<Option<String>> {
    match tokio::fs::read_to_string(directory.join(SCRIPT_FILENAME)).await {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(Error::external(std::io::Error::new(
            error.kind(),
            format!("read {SCRIPT_FILENAME}: {error}"),
        ))),
    }
}

/// Returns the description of one recipe node, or the documented fallback.
fn step_description(recipe: &ExperienceRecipe, capability_id: &str) -> String {
    recipe
        .combination_structure
        .nodes
        .get(capability_id)
        .map(|value| TraceNode::from_combination_value(capability_id, value).description)
        .map(|description| description.trim().to_string())
        .filter(|description| !description.is_empty())
        .unwrap_or_else(|| DEFAULT_STEP_DESCRIPTION.to_string())
}

/// Renders Python's `repr` of a string: single quotes, escaped control characters.
///
/// Backslashes, single quotes, and the three common control characters are
/// escaped; any other character below `0x20` or `0x7f` becomes `\xNN`. Every other
/// character, including multi-byte CJK, is emitted verbatim, matching Python's
/// `repr` under `ensure_ascii=False`.
#[must_use]
pub fn python_string_literal(value: &str) -> String {
    let mut literal = String::with_capacity(value.len() + 2);
    literal.push('\'');
    for character in value.chars() {
        match character {
            '\\' => literal.push_str("\\\\"),
            '\'' => literal.push_str("\\'"),
            '\n' => literal.push_str("\\n"),
            '\r' => literal.push_str("\\r"),
            '\t' => literal.push_str("\\t"),
            '\u{0008}' => literal.push_str("\\x08"),
            '\u{000c}' => literal.push_str("\\x0c"),
            other if (other as u32) < 0x20 || other as u32 == 0x7f => {
                let _ = write!(literal, "\\x{:02x}", other as u32);
            }
            other => literal.push(other),
        }
    }
    literal.push('\'');
    literal
}

/// Renders Python's `repr` of `META`: `{'name': '…', 'phases': ['step_1', …]}`.
fn meta_literal(identifier: &str, phases: &[String]) -> String {
    let phases: Vec<String> = phases
        .iter()
        .map(|phase| python_string_literal(phase))
        .collect();
    format!(
        "{{'name': {}, 'phases': [{}]}}",
        python_string_literal(identifier),
        phases.join(", ")
    )
}

/// Renders a single-line triple-quoted Python docstring.
///
/// Backslashes and every double quote are escaped, so the content can never close
/// the literal early and parses back to the original text.
fn python_docstring(text: &str) -> String {
    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"\"\"{escaped}\"\"\"")
}

/// Statically validates a generated `SwarmFlow` script.
///
/// The checks mirror Python `validate_generated_script` as closely as a
/// line-oriented scanner can: string literals and comments are masked (so their
/// contents cannot spoof a token), brackets and quotes must balance, the `META`,
/// `TASK`, and `async def run(args):` markers must be present, only
/// `from swarmflow import agent` may be imported, only `agent(...)` may be called,
/// and the review gate's forbidden code tokens are rejected.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error) naming the first offending
/// line (or the missing marker).
pub fn validate_generated_script(source: &str) -> Result<()> {
    let masked = mask_source(source)?;
    for (marker, prefix) in [
        ("META", "META ="),
        ("TASK", "TASK ="),
        ("run", "async def run(args):"),
    ] {
        if !masked
            .lines()
            .any(|line| line.trim_start().starts_with(prefix))
        {
            return Err(validation(
                "script",
                format!("missing `{marker}` marker (`{prefix}`)"),
            ));
        }
    }
    for (index, line) in masked.lines().enumerate() {
        check_line(index + 1, line)?;
    }
    Ok(())
}

/// Masks one string literal (single, double, or triple quoted) in place.
///
/// `index` is advanced past the literal, `line` tracks the 1-based source line,
/// and the literal's characters, including its quotes, become spaces so their
/// contents cannot spoof a token.
fn mask_string_literal(
    characters: &[char],
    masked: &mut String,
    index: &mut usize,
    character: char,
    line: &mut usize,
) -> Result<()> {
    let triple = *index + 2 < characters.len()
        && characters[*index + 1] == character
        && characters[*index + 2] == character;
    let width = if triple { 3 } else { 1 };
    let start_line = *line;
    for _ in 0..width {
        masked.push(' ');
    }
    *index += width;
    let mut closed = false;
    while *index < characters.len() {
        let current = characters[*index];
        if current == '\\' {
            masked.push(' ');
            if *index + 1 < characters.len() {
                if characters[*index + 1] == '\n' {
                    masked.push('\n');
                    *line += 1;
                } else {
                    masked.push(' ');
                }
                *index += 2;
            } else {
                *index += 1;
            }
            continue;
        }
        if current == character {
            let terminates = if triple {
                *index + 2 < characters.len()
                    && characters[*index + 1] == character
                    && characters[*index + 2] == character
            } else {
                true
            };
            if terminates {
                for _ in 0..width {
                    masked.push(' ');
                }
                *index += width;
                closed = true;
                break;
            }
        }
        if current == '\n' && !triple {
            return Err(validation(
                "script",
                format!("line {start_line}: unterminated string literal"),
            ));
        }
        if current == '\n' {
            *line += 1;
        }
        masked.push(if current == '\n' { '\n' } else { ' ' });
        *index += 1;
    }
    if !closed {
        return Err(validation(
            "script",
            format!("line {start_line}: unterminated string literal"),
        ));
    }
    Ok(())
}
/// Masks string literals and comments, and validates bracket/quote balance.
///
/// The returned string has the same line structure as `source`: string contents,
/// comment text, and quote delimiters are replaced by spaces, so a line number in
/// the masked text is a line number in the source.
fn mask_source(source: &str) -> Result<String> {
    let characters: Vec<char> = source.chars().collect();
    let mut masked = String::with_capacity(source.len());
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut line = 1usize;
    let mut index = 0usize;
    while index < characters.len() {
        let character = characters[index];
        match character {
            '\n' => {
                masked.push('\n');
                line += 1;
                index += 1;
            }
            '#' => {
                while index < characters.len() && characters[index] != '\n' {
                    masked.push(' ');
                    index += 1;
                }
            }
            '\'' | '"' => {
                mask_string_literal(&characters, &mut masked, &mut index, character, &mut line)?;
            }
            '(' | '[' | '{' => {
                stack.push((character, line));
                masked.push(character);
                index += 1;
            }
            ')' | ']' | '}' => {
                let expected = match character {
                    ')' => '(',
                    ']' => '[',
                    _ => '{',
                };
                match stack.pop() {
                    Some((open, _)) if open == expected => {
                        masked.push(character);
                        index += 1;
                    }
                    _ => {
                        return Err(validation(
                            "script",
                            format!("line {line}: unbalanced `{character}`"),
                        ));
                    }
                }
            }
            other => {
                masked.push(other);
                index += 1;
            }
        }
    }
    if let Some((open, at)) = stack.pop() {
        return Err(validation(
            "script",
            format!("line {at}: unclosed `{open}`"),
        ));
    }
    Ok(masked)
}

/// Checks one masked line against the import, call, and token whitelists.
fn check_line(number: usize, line: &str) -> Result<()> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return Ok(());
    }
    let lowered = trimmed.to_ascii_lowercase();
    for token in crate::flow::review::FORBIDDEN_CODE_TOKENS {
        if lowered.contains(token) {
            return Err(validation(
                "script",
                format!("line {number}: forbidden token `{token}`"),
            ));
        }
    }
    if trimmed == "import" || trimmed.starts_with("import ") {
        return Err(validation(
            "script",
            format!("line {number}: plain import statements are not allowed"),
        ));
    }
    if let Some(rest) = trimmed.strip_prefix("from ") {
        return check_from_import(number, rest);
    }
    check_calls(number, trimmed)
}

/// Checks one `from <module> import <names>` statement.
fn check_from_import(number: usize, rest: &str) -> Result<()> {
    let mut parts = rest.split_whitespace();
    let module = parts.next().unwrap_or_default();
    if module != SWARMFLOW_MODULE {
        return Err(validation(
            "script",
            format!("line {number}: disallowed import from `{module}`"),
        ));
    }
    if parts.next() != Some("import") {
        return Err(validation(
            "script",
            format!("line {number}: malformed from-import statement"),
        ));
    }
    for name in parts {
        let name =
            name.trim_matches(|character: char| matches!(character, ',' | '(' | ')' | '\\' | '*'));
        if name.is_empty() {
            continue;
        }
        if name != ALLOWED_OPERATOR {
            return Err(validation(
                "script",
                format!("line {number}: disallowed operator import `{name}`"),
            ));
        }
    }
    Ok(())
}

/// Checks that every call expression on one line names the allowed operator.
fn check_calls(number: usize, line: &str) -> Result<()> {
    let characters: Vec<char> = line.chars().collect();
    let mut previous_word: Option<String> = None;
    let mut previous_is_dot = false;
    let mut index = 0usize;
    while index < characters.len() {
        let character = characters[index];
        if character.is_alphanumeric() || character == '_' {
            let start = index;
            while index < characters.len()
                && (characters[index].is_alphanumeric() || characters[index] == '_')
            {
                index += 1;
            }
            let word: String = characters[start..index].iter().collect();
            let mut next = index;
            while next < characters.len() && characters[next] == ' ' {
                next += 1;
            }
            let is_call = next < characters.len() && characters[next] == '(';
            let is_definition = previous_word
                .as_deref()
                .is_some_and(|previous| NON_CALL_KEYWORDS.contains(&previous));
            if is_call && !previous_is_dot && !is_definition && word != ALLOWED_OPERATOR {
                return Err(validation(
                    "script",
                    format!("line {number}: disallowed call `{word}`"),
                ));
            }
            previous_is_dot = false;
            previous_word = Some(word);
            continue;
        }
        if character.is_whitespace() {
            index += 1;
            continue;
        }
        previous_is_dot = character == '.';
        previous_word = None;
        index += 1;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn string_literals_escape_quotes_and_keep_unicode() {
        assert_eq!(python_string_literal("plain"), "'plain'");
        assert_eq!(python_string_literal("it's"), "'it\\'s'");
        assert_eq!(python_string_literal("a\nb\tc"), "'a\\nb\\tc'");
        assert_eq!(python_string_literal("摘要"), "'摘要'");
        assert_eq!(python_string_literal("back\\slash"), "'back\\\\slash'");
    }

    #[test]
    fn masked_source_rejects_unbalanced_brackets() {
        assert!(mask_source("META = {'name': 'x'}").is_ok());
        assert!(mask_source("META = {'name': 'x'").is_err());
        assert!(mask_source("x = (]").is_err());
        assert!(mask_source("s = 'unterminated").is_err());
        assert!(mask_source("s = \"\"\"open").is_err());
        assert!(mask_source("s = '''multi\nline'''").is_ok());
        assert!(mask_source("x = 1  # comment with ) bracket").is_ok());
    }

    #[test]
    fn identifiers_strip_the_recipe_prefix() {
        let materials = serde_json::json!({"a": 1});
        let identifier = swarmflow_identifier(&materials, "recipe_abc");
        assert!(identifier.starts_with("combo-"));
        assert_eq!(identifier.len(), "combo-".len() + 12);
        assert!(crate::flow::model::is_valid_meta_name(&identifier));
        assert_eq!(
            swarmflow_identifier(&materials, "recipe_abc"),
            swarmflow_identifier(&materials, "recipe_abc")
        );
        assert_ne!(
            swarmflow_identifier(&materials, "recipe_abc"),
            swarmflow_identifier(&serde_json::json!({"a": 2}), "recipe_abc")
        );
    }
}
