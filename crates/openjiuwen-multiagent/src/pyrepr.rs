//! Python literal parsing and Python-style rendering of JSON values.
//!
//! Python's multi-agent tooling occasionally routes payloads through `str()`
//! (for example a tool result rendered into a model prompt) and later reads them
//! back with `json.loads` and an `ast.literal_eval` fallback
//! (`teams/handoff/handoff_signal.py`). [`literal_or_json`] generalises that
//! fallback, and [`render_python_value`] is the writer side: the rendering a
//! model sees is what [`parse_python_literal`] must read back.
//!
//! # Grammar
//!
//! The accepted language is Python's literal subset, whitespace-insensitive
//! (space, tab, newline, carriage return, form feed, vertical tab):
//!
//! ```text
//! value   := dict | set | list | tuple | string | number | "None" | "True" | "False"
//! dict    := "{" [ pair { "," pair } [ "," ] ] "}"
//! set     := "{" value { "," value } [ "," ] "}"
//! list    := "[" [ value { "," value } [ "," ] ] "]"
//! tuple   := "(" ")" | "(" value "," [ value { "," value } [ "," ] ] ")" | "(" value ")"
//! pair    := key ":" value
//! string  := ( "'" | "\"" ) { char | escape } ( "'" | "\"" )
//! number  := [ "+" | "-" ] digits [ "." digits ] [ ( "e" | "E" ) [ "+" | "-" ] digits ]
//! ```
//!
//! Accepted escapes are `\\`, `\'`, `\"`, `\n`, `\t`, `\r`, `\0`, `\a`, `\b`,
//! `\f`, `\v`, `\xNN` (two hex digits) and `\uNNNN` (four hex digits).
//! `_` may separate digits but never lead, trail or double.
//!
//! Tuples and sets both map to JSON arrays (JSON has no tuple or set type), a
//! parenthesized single value is that value (`(1)` is `1`, `(1,)` is `[1]`), and
//! dict keys are rendered to JSON strings: strings as-is, numbers through their
//! JSON text, `True`/`False`/`None` as those words. `{}` is the empty dict,
//! matching Python where the empty set is spelled `set()`.
//!
//! # Rejected input
//!
//! Anything outside that grammar yields `None` — never a panic and never a
//! partial value: trailing garbage, unterminated strings, unknown escapes,
//! bytes/raw/f-string prefixes (`b'..'`, `f'..'`), dict unpacking (`{**x}`),
//! comprehensions, base-prefixed integers (`0x10`, `0o10`, `0b10`), complex
//! numbers (`1j`), `inf`/`nan` and floats that overflow to infinity, integers
//! that do not fit `i64`, un-hashable dict keys, and nesting deeper than
//! [`MAX_LITERAL_DEPTH`] levels.
//!
//! # Rendering
//!
//! [`render_python_value`] matches Python's `str()` for JSON values: a
//! top-level string is bare, strings inside a container are single-quoted with
//! `\\`/`'` and control characters escaped (Python `repr` style), `True`/
//! `False`/`None` stand for booleans and null, containers are printed on one
//! line with `, ` separators, and dicts keep their insertion order. One
//! deliberate difference from `CPython` `repr`: a string is always single-quoted
//! (with `'` escaped), where Python would switch to double quotes for text that
//! contains `'` but no `"`. Both forms parse back to the same value. Numbers use
//! [`serde_json::Number`]'s shortest round-trip text, so `1.0` renders as
//! `"1.0"` and small magnitudes as `"1e-5"` (Rust/ryu spelling) rather than
//! `CPython`'s zero-padded `"1e-05"`; integral floats keep the `.0` suffix.

use std::fmt::Write as _;

use serde_json::{Map, Number, Value};

/// Maximum nesting level [`parse_python_literal`] accepts.
///
/// A value nested deeper returns `None` instead of risking unbounded recursion.
pub const MAX_LITERAL_DEPTH: usize = 64;

/// Parses `input` as a Python literal.
///
/// Returns `None` on any syntax error, unsupported construct, rejected number,
/// or nesting beyond [`MAX_LITERAL_DEPTH`]. See the module docs for the grammar.
#[must_use]
pub fn parse_python_literal(input: &str) -> Option<Value> {
    let mut parser = Parser {
        chars: input.chars().collect(),
        pos: 0,
    };
    parser.skip_whitespace();
    let value = parser.parse_value(0)?;
    parser.skip_whitespace();
    (parser.pos == parser.chars.len()).then_some(value)
}

/// Parses `input` as JSON, falling back to a Python literal.
///
/// This is the Rust counterpart of Python's `json.loads` → `ast.literal_eval`
/// fallback: a payload that is valid JSON always parses as JSON, so JSON's
/// spelling wins where the two grammars overlap.
#[must_use]
pub fn literal_or_json(input: &str) -> Option<Value> {
    serde_json::from_str::<Value>(input)
        .ok()
        .or_else(|| parse_python_literal(input))
}

/// Renders `value` the way Python's `str()` renders the equivalent object.
///
/// A top-level string is returned bare (no quotes); strings nested in arrays or
/// objects are single-quoted like Python `repr`. See the module docs for the
/// number formatting rule.
#[must_use]
pub fn render_python_value(value: &Value) -> String {
    let mut out = String::new();
    render_into(value, false, &mut out);
    out
}

/// Appends the Python rendering of `value` to `out`.
fn render_into(value: &Value, nested: bool, out: &mut String) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => {
            if nested {
                push_quoted(text, out);
            } else {
                out.push_str(text);
            }
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                render_into(item, true, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                push_quoted(key, out);
                out.push_str(": ");
                render_into(item, true, out);
            }
            out.push('}');
        }
    }
}

/// Appends `text` as a single-quoted Python string literal (Python `repr` style).
fn push_quoted(text: &str, out: &mut String) {
    out.push('\'');
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character.is_control() => {
                let code = character as u32;
                if code <= 0xFF {
                    let _ = write!(out, "\\x{code:02x}");
                } else {
                    let _ = write!(out, "\\u{code:04x}");
                }
            }
            character => out.push(character),
        }
    }
    out.push('\'');
}

/// Renders a dict key to the JSON string it becomes.
///
/// Only scalars are valid Python dict keys; arrays and objects return `None`.
fn scalar_key(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Bool(true) => Some("True".to_owned()),
        Value::Bool(false) => Some("False".to_owned()),
        Value::Null => Some("None".to_owned()),
        Value::Number(number) => Some(number.to_string()),
        Value::Array(_) | Value::Object(_) => None,
    }
}

/// Recursive-descent parser over the characters of one literal.
struct Parser {
    /// The input, split into characters so multi-byte text is indexed safely.
    chars: Vec<char>,
    /// Byte-independent cursor into [`chars`](Parser::chars).
    pos: usize,
}

impl Parser {
    /// Returns the character under the cursor.
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    /// Returns the character `offset` positions after the cursor.
    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    /// Consumes and returns the character under the cursor.
    fn bump(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.pos += 1;
        Some(character)
    }

    /// Consumes `expected`, reporting whether it was there.
    fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// Skips Python-insignificant whitespace.
    fn skip_whitespace(&mut self) {
        while matches!(
            self.peek(),
            Some(' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}')
        ) {
            self.pos += 1;
        }
    }

    /// Enters one more container level, or rejects at the depth limit.
    fn enter(depth: usize) -> Option<usize> {
        if depth >= MAX_LITERAL_DEPTH {
            None
        } else {
            Some(depth + 1)
        }
    }

    /// Parses one value nested `depth` container levels deep.
    fn parse_value(&mut self, depth: usize) -> Option<Value> {
        self.skip_whitespace();
        match self.peek()? {
            '[' => self.parse_list(depth),
            '(' => self.parse_tuple(depth),
            '{' => self.parse_brace(depth),
            '\'' | '"' => self.parse_string().map(Value::String),
            _ => self.parse_atom(),
        }
    }

    /// Parses a number or a `None`/`True`/`False` keyword.
    fn parse_atom(&mut self) -> Option<Value> {
        match self.peek()? {
            '-' | '+' | '.' | '0'..='9' => self.parse_number(),
            character if character.is_ascii_alphabetic() || character == '_' => {
                self.parse_keyword()
            }
            _ => None,
        }
    }

    /// Parses a bare keyword, rejecting every other identifier.
    fn parse_keyword(&mut self) -> Option<Value> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            self.pos += 1;
        }
        let word: String = self.chars[start..self.pos].iter().collect();
        match word.as_str() {
            "None" => Some(Value::Null),
            "True" => Some(Value::Bool(true)),
            "False" => Some(Value::Bool(false)),
            _ => None,
        }
    }

    /// Reads a run of decimal digits into `text`, allowing `_` only between two
    /// digits.
    ///
    /// Returns `None` on a misplaced separator, `Some(has_digit)` otherwise.
    fn read_digits(&mut self, text: &mut String) -> Option<bool> {
        let mut has_digit = false;
        while let Some(character) = self.peek() {
            if character.is_ascii_digit() {
                text.push(character);
                has_digit = true;
                self.pos += 1;
            } else if character == '_' {
                let next_is_digit = self.peek_at(1).is_some_and(|next| next.is_ascii_digit());
                if !has_digit || !next_is_digit {
                    return None;
                }
                self.pos += 1;
            } else {
                break;
            }
        }
        Some(has_digit)
    }

    /// Parses a decimal integer or float.
    fn parse_number(&mut self) -> Option<Value> {
        let negative = match self.peek() {
            Some('-') => {
                self.pos += 1;
                true
            }
            Some('+') => {
                self.pos += 1;
                false
            }
            _ => false,
        };
        let mut text = String::new();
        let int_digits = self.read_digits(&mut text)?;
        let mut is_float = false;
        let mut fraction_digits = false;
        if self.peek() == Some('.') {
            is_float = true;
            self.pos += 1;
            text.push('.');
            fraction_digits = self.read_digits(&mut text)?;
        }
        if !int_digits && !fraction_digits {
            return None;
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            self.pos += 1;
            text.push('e');
            if matches!(self.peek(), Some('-' | '+')) {
                text.push(self.bump()?);
            }
            if !self.read_digits(&mut text)? {
                return None;
            }
        }
        if !is_float {
            let value: i64 = text.parse().ok()?;
            let value = if negative {
                value.checked_neg()?
            } else {
                value
            };
            return Some(Value::Number(Number::from(value)));
        }
        if text.starts_with('.') {
            text.insert(0, '0');
        }
        if text.ends_with('.') {
            text.push('0');
        }
        let value: f64 = text.parse().ok()?;
        let value = if negative { -value } else { value };
        Some(Value::Number(Number::from_f64(value)?))
    }

    /// Parses a single- or double-quoted string.
    fn parse_string(&mut self) -> Option<String> {
        let quote = self.bump()?;
        let mut out = String::new();
        loop {
            match self.bump()? {
                character if character == quote => return Some(out),
                '\\' => {
                    let escape = self.bump()?;
                    out.push(self.decode_escape(escape)?);
                }
                '\n' => return None,
                character => out.push(character),
            }
        }
    }

    /// Decodes one escape sequence after the backslash.
    fn decode_escape(&mut self, escape: char) -> Option<char> {
        Some(match escape {
            '\\' => '\\',
            '\'' => '\'',
            '"' => '"',
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            '0' => '\0',
            'a' => '\u{7}',
            'b' => '\u{8}',
            'f' => '\u{c}',
            'v' => '\u{b}',
            'x' => char::from_u32(self.decode_hex(2)?)?,
            'u' => char::from_u32(self.decode_hex(4)?)?,
            _ => return None,
        })
    }

    /// Reads exactly `digits` hexadecimal digits.
    fn decode_hex(&mut self, digits: usize) -> Option<u32> {
        let mut value = 0_u32;
        for _ in 0..digits {
            value = value * 16 + self.bump()?.to_digit(16)?;
        }
        Some(value)
    }

    /// Parses a list literal.
    fn parse_list(&mut self, depth: usize) -> Option<Value> {
        let depth = Self::enter(depth)?;
        self.bump()?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.eat(']') {
            return Some(Value::Array(items));
        }
        loop {
            items.push(self.parse_value(depth)?);
            self.skip_whitespace();
            if self.eat(',') {
                self.skip_whitespace();
                if self.eat(']') {
                    return Some(Value::Array(items));
                }
                continue;
            }
            if self.eat(']') {
                return Some(Value::Array(items));
            }
            return None;
        }
    }

    /// Parses a tuple literal, including the single-element trailing-comma form.
    ///
    /// `()` is an empty tuple, `(a, b)` a tuple, and `(a)` is just `a`.
    fn parse_tuple(&mut self, depth: usize) -> Option<Value> {
        let depth = Self::enter(depth)?;
        self.bump()?;
        self.skip_whitespace();
        if self.eat(')') {
            return Some(Value::Array(Vec::new()));
        }
        let first = self.parse_value(depth)?;
        self.skip_whitespace();
        if !self.eat(',') {
            // A parenthesized value is not a tuple.
            return self.eat(')').then_some(first);
        }
        let mut items = vec![first];
        loop {
            self.skip_whitespace();
            if self.eat(')') {
                return Some(Value::Array(items));
            }
            items.push(self.parse_value(depth)?);
            self.skip_whitespace();
            if !self.eat(',') {
                return self.eat(')').then(|| Value::Array(items));
            }
        }
    }

    /// Parses a dict or set literal, which share the braces.
    fn parse_brace(&mut self, depth: usize) -> Option<Value> {
        let depth = Self::enter(depth)?;
        self.bump()?;
        self.skip_whitespace();
        if self.eat('}') {
            return Some(Value::Object(Map::new()));
        }
        let first = self.parse_value(depth)?;
        self.skip_whitespace();
        if self.eat(':') {
            return self.parse_dict_body(depth, first);
        }
        let mut items = vec![first];
        loop {
            if self.eat('}') {
                return Some(Value::Array(items));
            }
            if !self.eat(',') {
                return None;
            }
            self.skip_whitespace();
            if self.eat('}') {
                return Some(Value::Array(items));
            }
            items.push(self.parse_value(depth)?);
            self.skip_whitespace();
        }
    }

    /// Parses the rest of a dict literal after the first key's colon.
    fn parse_dict_body(&mut self, depth: usize, first_key: Value) -> Option<Value> {
        let mut map = Map::new();
        let mut key = first_key;
        loop {
            self.skip_whitespace();
            let value = self.parse_value(depth)?;
            map.insert(scalar_key(&key)?, value);
            self.skip_whitespace();
            if self.eat('}') {
                return Some(Value::Object(map));
            }
            if !self.eat(',') {
                return None;
            }
            self.skip_whitespace();
            if self.eat('}') {
                return Some(Value::Object(map));
            }
            key = self.parse_value(depth)?;
            self.skip_whitespace();
            if !self.eat(':') {
                return None;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keywords_parse() {
        assert_eq!(parse_python_literal("None"), Some(Value::Null));
        assert_eq!(parse_python_literal("True"), Some(Value::Bool(true)));
        assert_eq!(parse_python_literal("False"), Some(Value::Bool(false)));
        assert_eq!(parse_python_literal("  True \n"), Some(Value::Bool(true)));
        assert_eq!(parse_python_literal("true"), None);
        assert_eq!(parse_python_literal("none"), None);
    }

    #[test]
    fn integers_parse_with_sign_and_separators() {
        assert_eq!(parse_python_literal("0"), Some(json!(0)));
        assert_eq!(parse_python_literal("42"), Some(json!(42)));
        assert_eq!(parse_python_literal("-42"), Some(json!(-42)));
        assert_eq!(parse_python_literal("+7"), Some(json!(7)));
        assert_eq!(parse_python_literal("1_000_000"), Some(json!(1_000_000)));
        assert_eq!(parse_python_literal("1_"), None);
        assert_eq!(parse_python_literal("1__0"), None);
        assert_eq!(parse_python_literal("9223372036854775808"), None);
        assert_eq!(parse_python_literal("0x10"), None);
    }

    #[test]
    fn floats_parse_with_exponent_and_leading_dot() {
        assert_eq!(parse_python_literal("1.5"), Some(json!(1.5)));
        assert_eq!(parse_python_literal("-1.5"), Some(json!(-1.5)));
        assert_eq!(parse_python_literal(".5"), Some(json!(0.5)));
        assert_eq!(parse_python_literal("5."), Some(json!(5.0)));
        assert_eq!(parse_python_literal("1e3"), Some(json!(1000.0)));
        assert_eq!(parse_python_literal("1E-3"), Some(json!(0.001)));
        assert_eq!(parse_python_literal("1_0.5"), Some(json!(10.5)));
        assert_eq!(parse_python_literal("1.5e+2"), Some(json!(150.0)));
        assert_eq!(parse_python_literal("1e"), None);
        assert_eq!(parse_python_literal("1e400"), None);
        assert_eq!(parse_python_literal("."), None);
    }

    #[test]
    fn strings_parse_with_both_quote_styles_and_escapes() {
        assert_eq!(parse_python_literal("'abc'"), Some(json!("abc")));
        assert_eq!(parse_python_literal("\"abc\""), Some(json!("abc")));
        assert_eq!(parse_python_literal("''"), Some(json!("")));
        assert_eq!(parse_python_literal(r"'a\'b'"), Some(json!("a'b")));
        assert_eq!(parse_python_literal(r#""a\"b""#), Some(json!("a\"b")));
        assert_eq!(parse_python_literal(r"'a\\b'"), Some(json!("a\\b")));
        assert_eq!(
            parse_python_literal(r"'a\nb\tc\rd\0'"),
            Some(json!("a\nb\tc\rd\0"))
        );
        assert_eq!(parse_python_literal(r"'\x41\u4e2d'"), Some(json!("A中")));
        assert_eq!(parse_python_literal(r"'\x4'"), None);
        assert_eq!(parse_python_literal(r"'\u12'"), None);
        assert_eq!(parse_python_literal(r"'\q'"), None);
        assert_eq!(parse_python_literal("'abc"), None);
        assert_eq!(parse_python_literal("'a\nb'"), None);
        assert_eq!(parse_python_literal("'中'"), Some(json!("中")));
    }

    #[test]
    fn lists_parse_including_trailing_commas() {
        assert_eq!(parse_python_literal("[]"), Some(json!([])));
        assert_eq!(parse_python_literal("[1, 2, 3]"), Some(json!([1, 2, 3])));
        assert_eq!(parse_python_literal("[1, 2, 3,]"), Some(json!([1, 2, 3])));
        assert_eq!(
            parse_python_literal("[ 1 , [ 'a' , None ] , ]"),
            Some(json!([1, ["a", null]]))
        );
        assert_eq!(parse_python_literal("[1 2]"), None);
        assert_eq!(parse_python_literal("[,]"), None);
        assert_eq!(parse_python_literal("[1"), None);
    }

    #[test]
    fn tuples_map_to_arrays() {
        assert_eq!(parse_python_literal("()"), Some(json!([])));
        assert_eq!(parse_python_literal("(1,)"), Some(json!([1])));
        assert_eq!(parse_python_literal("(1, 2)"), Some(json!([1, 2])));
        assert_eq!(parse_python_literal("(1, 2,)"), Some(json!([1, 2])));
        assert_eq!(parse_python_literal("(1)"), Some(json!(1)));
        assert_eq!(parse_python_literal("('a')"), Some(json!("a")));
        assert_eq!(
            parse_python_literal("(1, (2, 3))"),
            Some(json!([1, [2, 3]]))
        );
        assert_eq!(parse_python_literal("(1"), None);
    }

    #[test]
    fn sets_map_to_arrays() {
        assert_eq!(parse_python_literal("{1}"), Some(json!([1])));
        assert_eq!(
            parse_python_literal("{1, 'a', True}"),
            Some(json!([1, "a", true]))
        );
        assert_eq!(parse_python_literal("{1,}"), Some(json!([1])));
        assert_eq!(parse_python_literal("{1, 2"), None);
    }

    #[test]
    fn dicts_render_scalar_keys_to_json_strings() {
        assert_eq!(parse_python_literal("{}"), Some(json!({})));
        assert_eq!(parse_python_literal("{'a': 1}"), Some(json!({"a": 1})));
        assert_eq!(
            parse_python_literal("{'a': 1, \"b\": [2],}"),
            Some(json!({"a": 1, "b": [2]}))
        );
        assert_eq!(parse_python_literal("{1: 'x'}"), Some(json!({"1": "x"})));
        assert_eq!(
            parse_python_literal("{1.5: 'x'}"),
            Some(json!({"1.5": "x"}))
        );
        assert_eq!(parse_python_literal("{True: 1}"), Some(json!({"True": 1})));
        assert_eq!(parse_python_literal("{None: 1}"), Some(json!({"None": 1})));
        assert_eq!(
            parse_python_literal("{'a': 1, 'a': 2}"),
            Some(json!({"a": 2}))
        );
        assert_eq!(parse_python_literal("{[1]: 2}"), None);
        assert_eq!(parse_python_literal("{'a' 1}"), None);
        assert_eq!(parse_python_literal("{'a': }"), None);
        assert_eq!(parse_python_literal("{**x}"), None);
        assert_eq!(parse_python_literal("{'a': 1"), None);
    }

    #[test]
    fn nested_handoff_payload_round_trips() {
        let payload = json!({
            "__handoff_to__": "reviewer",
            "__handoff_message__": "check it",
            "meta": {"count": 2, "tags": ["a", "b"], "ok": true, "none": null}
        });
        let rendered = render_python_value(&payload);
        let parsed = parse_python_literal(&rendered).expect("renders a parsable literal");
        assert_eq!(parsed, payload);
        assert_eq!(literal_or_json(&rendered), Some(payload));
    }

    #[test]
    fn trailing_garbage_is_rejected() {
        assert_eq!(parse_python_literal("1 2"), None);
        assert_eq!(parse_python_literal("[1] x"), None);
        assert_eq!(parse_python_literal("'a' 'b'"), None);
        assert_eq!(parse_python_literal("None()"), None);
        assert_eq!(parse_python_literal("TrueFalse"), None);
        assert_eq!(parse_python_literal(""), None);
        assert_eq!(parse_python_literal("   "), None);
    }

    #[test]
    fn unsupported_constructs_are_rejected() {
        assert_eq!(parse_python_literal("b'abc'"), None);
        assert_eq!(parse_python_literal("r'abc'"), None);
        assert_eq!(parse_python_literal("f'{x}'"), None);
        assert_eq!(parse_python_literal("[x for x in y]"), None);
        assert_eq!(parse_python_literal("{k: v for k, v in y}"), None);
        assert_eq!(parse_python_literal("1j"), None);
        assert_eq!(parse_python_literal("1.5j"), None);
        assert_eq!(parse_python_literal("inf"), None);
        assert_eq!(parse_python_literal("nan"), None);
        assert_eq!(parse_python_literal("set()"), None);
        assert_eq!(parse_python_literal("..."), None);
        assert_eq!(parse_python_literal("dict(a=1)"), None);
    }

    #[test]
    fn depth_limit_is_enforced() {
        let at_limit = format!(
            "{}1{}",
            "[".repeat(MAX_LITERAL_DEPTH),
            "]".repeat(MAX_LITERAL_DEPTH)
        );
        assert!(parse_python_literal(&at_limit).is_some());

        let too_deep = format!(
            "{}1{}",
            "[".repeat(MAX_LITERAL_DEPTH + 1),
            "]".repeat(MAX_LITERAL_DEPTH + 1)
        );
        assert_eq!(parse_python_literal(&too_deep), None);

        let mixed = format!(
            "{}1{}",
            "([{".repeat(MAX_LITERAL_DEPTH),
            "}])".repeat(MAX_LITERAL_DEPTH)
        );
        assert_eq!(parse_python_literal(&mixed), None);

        let unfriendly = "(".repeat(10_000);
        assert_eq!(parse_python_literal(&unfriendly), None);
    }

    #[test]
    fn literal_or_json_prefers_json() {
        assert_eq!(literal_or_json(r#"{"a": 1}"#), Some(json!({"a": 1})));
        assert_eq!(literal_or_json("{'a': 1}"), Some(json!({"a": 1})));
        assert_eq!(literal_or_json("[1, 2]"), Some(json!([1, 2])));
        assert_eq!(literal_or_json("None"), Some(Value::Null));
        assert_eq!(literal_or_json("nonsense"), None);
        assert_eq!(literal_or_json(""), None);
    }

    #[test]
    fn render_matches_python_str_for_scalars() {
        assert_eq!(render_python_value(&Value::Null), "None");
        assert_eq!(render_python_value(&json!(true)), "True");
        assert_eq!(render_python_value(&json!(false)), "False");
        assert_eq!(render_python_value(&json!(42)), "42");
        assert_eq!(render_python_value(&json!(-1)), "-1");
        assert_eq!(render_python_value(&json!(1.0)), "1.0");
        assert_eq!(render_python_value(&json!(0.5)), "0.5");
        assert_eq!(render_python_value(&json!("plain")), "plain");
        assert_eq!(render_python_value(&json!("")), "");
    }

    #[test]
    fn render_quotes_strings_inside_containers() {
        assert_eq!(render_python_value(&json!([])), "[]");
        assert_eq!(render_python_value(&json!({})), "{}");
        assert_eq!(render_python_value(&json!(["a"])), "['a']");
        assert_eq!(
            render_python_value(&json!(["it's", true, null, 1])),
            "['it\\'s', True, None, 1]"
        );
        assert_eq!(
            render_python_value(&json!({"k": 1, "j": [1, 2]})),
            "{'k': 1, 'j': [1, 2]}"
        );
        assert_eq!(render_python_value(&json!(["a\nb"])), "['a\\nb']");
        assert_eq!(render_python_value(&json!(["\\"])), "['\\\\']");
        assert_eq!(render_python_value(&json!(["\u{7}"])), "['\\x07']");
        assert_eq!(render_python_value(&json!(["中"])), "['中']");
        assert_eq!(
            render_python_value(&json!({"a": {"b": "c"}})),
            "{'a': {'b': 'c'}}"
        );
    }

    #[test]
    fn render_keeps_integral_floats_distinguishable() {
        let rendered = render_python_value(&json!(1.0));
        assert!(rendered.ends_with(".0"), "rendered {rendered}");
        assert_eq!(render_python_value(&json!(-2.5)), "-2.5");
    }
}
