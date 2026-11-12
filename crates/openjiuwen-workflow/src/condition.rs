//! Condition expressions, branches, and the branch router.
//!
//! Ports `openjiuwen/core/workflow/components/condition/expression.py` and
//! `…/flow/branch_router.py`. Python rewrites the expression into Python source and
//! evaluates it through `ast`; this port keeps the surface syntax but compiles it with a
//! small recursive-descent parser into an owned AST, so evaluation is deterministic and
//! never touches the host language.
//!
//! Grammar, loosest binding first:
//!
//! ```text
//! or         := and ("||" | "or" and)*
//! and        := comparison ("&&" | "and" comparison)*
//! comparison := unary (("==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "not_in" | "not" "in") unary)?
//! unary      := ("!" | "not") unary | primary
//! primary    := number | string | "true" | "false" | "null" | "${path}"
//!             | ("len" | "length" | "is_empty" | "is_not_empty") "(" or ")"
//!             | "(" or ")"
//! ```
//!
//! `!` binds tighter than a comparison, so `!${a} == true` means `(!${a}) == true`;
//! Python's `not` binds looser, and the tighter binding is the documented Rust
//! behaviour. Chained comparisons (`1 < x < 3`) are rejected as trailing garbage.
//!
//! `${path}` resolves through [`crate::reference::resolve_path`] and a missing path is
//! `null`. A bare value is true when it is a non-zero number, a non-empty
//! string/array/object, or `true`. Equality compares JSON values numerically
//! (`1 == 1.0`), ordering across JSON types is an evaluation error, and membership never
//! fails.
//!
//! ```
//! use serde_json::json;
//! use openjiuwen_workflow::condition::Condition;
//!
//! let condition = Condition::new("len(${start.items}) > 1 && ${start.ready}").unwrap();
//! let source = json!({"start": {"items": [1, 2], "ready": true}});
//! assert!(condition.evaluate(&source).unwrap());
//! ```
//!
//! Deviations from Python: the tighter `!`, `length()` as an alias of `len()`, an empty
//! `${}` reference is a syntax error (Python reads it as the empty path), and the
//! nesting-depth guard reports a syntax error instead of Python's evaluation error.

use std::cmp::Ordering;

use serde_json::{Number, Value};

use openjiuwen_common::error::Result;

use crate::error::{branch_execution_error, expression_eval_error, expression_syntax_error};
use crate::reference::resolve_path;

/// Maximum number of characters a condition expression may contain.
///
/// Mirrors Python's `MAX_EXPRESSION_LENGTH`; exceeding it is a syntax error.
const MAX_EXPRESSION_LENGTH: usize = 5_000;

/// Maximum parser recursion depth.
///
/// Mirrors Python's `MAX_AST_DEPTH` intent (reject hostile nesting instead of
/// overflowing the stack). The counter tracks parser frames, so a parenthesis pair
/// costs more than one level.
const MAX_PARSE_DEPTH: usize = 100;

/// A parsed condition expression.
///
/// The source text is kept for tracing and diagnostics; evaluation uses the compiled
/// AST, so an expression is parsed exactly once.
#[derive(Debug, Clone)]
pub struct Condition {
    expression: String,
    ast: Expr,
}

impl Condition {
    /// Parses `expression`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Workflow`](openjiuwen_common::error::Error::Workflow) with code
    /// `101024` (`EXPRESSION_SYNTAX_ERROR`) for an empty expression, an unknown token,
    /// unbalanced parentheses, trailing input, or an expression longer than 5000
    /// characters.
    pub fn new(expression: impl Into<String>) -> Result<Self> {
        let expression = expression.into();
        if expression.chars().count() > MAX_EXPRESSION_LENGTH {
            return Err(expression_syntax_error());
        }
        let ast = Parser::parse(&expression)?;
        Ok(Self { expression, ast })
    }

    /// The expression source, exactly as it was supplied.
    #[must_use]
    pub fn expression(&self) -> &str {
        &self.expression
    }

    /// Evaluates the expression against `source`, which supplies `${path}` values.
    ///
    /// # Errors
    ///
    /// Returns code `101025` (`EXPRESSION_EVAL_ERROR`) when the expression cannot be
    /// evaluated, for example ordering mutually incomparable values or calling a helper
    /// on a scalar.
    pub fn evaluate(&self, source: &Value) -> Result<bool> {
        Ok(is_truthy(&eval(&self.ast, source)?))
    }
}

/// One ordered branch of a [`BranchRouter`].
#[derive(Debug, Clone)]
pub struct Branch {
    /// Optional identifier used for tracing; Python's `branch_id`.
    pub name: Option<String>,
    /// Component ids this branch routes to when its condition holds.
    pub targets: Vec<String>,
    /// The condition that selects this branch.
    pub condition: Condition,
}

impl Branch {
    /// A branch without a name.
    #[must_use]
    pub fn new(condition: Condition, targets: Vec<String>) -> Self {
        Self {
            name: None,
            targets,
            condition,
        }
    }

    /// Sets the tracing name, consuming and returning the branch.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Ordered branches plus an optional default target.
///
/// `route` evaluates the branches in insertion order and returns the targets of the
/// first match; Python's `BranchRouter` has the same first-match-wins rule.
#[derive(Debug, Clone, Default)]
pub struct BranchRouter {
    branches: Vec<Branch>,
    default_targets: Option<Vec<String>>,
}

impl BranchRouter {
    /// An empty router without branches and without a default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a branch; insertion order is preserved and decides precedence.
    #[must_use]
    pub fn add_branch(&mut self, branch: Branch) -> &mut Self {
        self.branches.push(branch);
        self
    }

    /// Routes to `targets` when no branch condition matches.
    #[must_use]
    pub fn default_target(&mut self, targets: Vec<String>) -> &mut Self {
        self.default_targets = Some(targets);
        self
    }

    /// Every target this router can choose, deduplicated and in insertion order.
    ///
    /// Branch targets come first, followed by the default targets. Python exposes the
    /// same set as `all_targets`; the order makes it usable for graph registration.
    #[must_use]
    pub fn targets(&self) -> Vec<String> {
        let ordered = self
            .branches
            .iter()
            .flat_map(|branch| &branch.targets)
            .chain(self.default_targets.iter().flatten());
        let mut targets: Vec<String> = Vec::new();
        for target in ordered {
            if !targets.contains(target) {
                targets.push(target.clone());
            }
        }
        targets
    }

    /// Returns the targets of the first branch whose condition holds.
    ///
    /// # Errors
    ///
    /// Propagates an expression evaluation error (`101025`) from a branch condition, and
    /// returns `101021` (`COMPONENT_BRANCH_EXECUTION_ERROR`) when nothing matches and no
    /// default target is configured.
    pub fn route(&self, source: &Value) -> Result<Vec<String>> {
        for branch in &self.branches {
            if branch.condition.evaluate(source)? {
                return Ok(branch.targets.clone());
            }
        }
        if let Some(targets) = &self.default_targets {
            return Ok(targets.clone());
        }
        Err(branch_execution_error(
            "branch meeting the condition was not found",
        ))
    }
}

/// The comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareOp {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

impl CompareOp {
    /// The operator as it is written in an expression.
    const fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::Lt => "<",
            Self::Le => "<=",
        }
    }

    /// Whether `ordering` satisfies this operator.
    fn satisfied_by(self, ordering: Ordering) -> bool {
        match self {
            Self::Eq => ordering == Ordering::Equal,
            Self::Ne => ordering != Ordering::Equal,
            Self::Gt => ordering == Ordering::Greater,
            Self::Ge => ordering != Ordering::Less,
            Self::Lt => ordering == Ordering::Less,
            Self::Le => ordering != Ordering::Greater,
        }
    }
}

/// The value helpers an expression may call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HelperKind {
    Len,
    IsEmpty,
    IsNotEmpty,
}

/// `in` versus `not in`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MembershipCheck {
    In,
    NotIn,
}

/// The operator between the two operands of a comparison.
#[derive(Debug, Clone, Copy)]
enum Operator {
    Compare(CompareOp),
    Member(MembershipCheck),
}

/// The compiled form of an expression.
#[derive(Debug, Clone)]
enum Expr {
    Literal(Value),
    Reference(String),
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Compare(CompareOp, Box<Self>, Box<Self>),
    Membership(MembershipCheck, Box<Self>, Box<Self>),
    Helper(HelperKind, Box<Self>),
}

/// One lexical token.
#[derive(Debug, Clone)]
enum Token {
    Literal(Value),
    Reference(String),
    Compare(CompareOp),
    Helper(HelperKind),
    In,
    NotIn,
    And,
    Or,
    Not,
    LParen,
    RParen,
}

/// Splits `input` into tokens, dropping whitespace.
fn tokenize(input: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while let Some(&current) = chars.get(cursor) {
        let ahead = chars.get(cursor + 1).copied();
        let (token, next) = match current {
            c if c.is_whitespace() => {
                cursor += 1;
                continue;
            }
            '(' => (Token::LParen, cursor + 1),
            ')' => (Token::RParen, cursor + 1),
            '&' if ahead == Some('&') => (Token::And, cursor + 2),
            '|' if ahead == Some('|') => (Token::Or, cursor + 2),
            '!' if ahead == Some('=') => (Token::Compare(CompareOp::Ne), cursor + 2),
            '!' => (Token::Not, cursor + 1),
            '=' if ahead == Some('=') => (Token::Compare(CompareOp::Eq), cursor + 2),
            '>' if ahead == Some('=') => (Token::Compare(CompareOp::Ge), cursor + 2),
            '>' => (Token::Compare(CompareOp::Gt), cursor + 1),
            '<' if ahead == Some('=') => (Token::Compare(CompareOp::Le), cursor + 2),
            '<' => (Token::Compare(CompareOp::Lt), cursor + 1),
            '\'' | '"' => {
                let (text, next) = scan_string(&chars, cursor, current)?;
                (Token::Literal(Value::String(text)), next)
            }
            '$' => {
                let (path, next) = scan_reference(&chars, cursor)?;
                (Token::Reference(path), next)
            }
            '-' | '0'..='9' => {
                let (number, next) = scan_number(&chars, cursor)?;
                (Token::Literal(number), next)
            }
            c if c.is_alphabetic() || c == '_' => scan_word(&chars, cursor)?,
            _ => return Err(expression_syntax_error()),
        };
        tokens.push(token);
        cursor = next;
    }
    Ok(tokens)
}

/// Scans a quoted string literal, honouring backslash escapes.
///
/// Known escapes are `\n`, `\t`, `\r`, `\0`, `\\`, `\'` and `\"`; any other escaped
/// character stands for itself, matching Python's lenient handling.
fn scan_string(chars: &[char], start: usize, quote: char) -> Result<(String, usize)> {
    let mut cursor = start + 1;
    let mut text = String::new();
    while let Some(&current) = chars.get(cursor) {
        if current == '\\' {
            let Some(&escaped) = chars.get(cursor + 1) else {
                return Err(expression_syntax_error());
            };
            text.push(match escaped {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '0' => '\0',
                other => other,
            });
            cursor += 2;
        } else if current == quote {
            return Ok((text, cursor + 1));
        } else {
            text.push(current);
            cursor += 1;
        }
    }
    Err(expression_syntax_error())
}

/// Scans a `${path}` reference and returns its path.
fn scan_reference(chars: &[char], start: usize) -> Result<(String, usize)> {
    if chars.get(start + 1) != Some(&'{') {
        return Err(expression_syntax_error());
    }
    let mut cursor = start + 2;
    let mut path = String::new();
    while let Some(&current) = chars.get(cursor) {
        match current {
            '}' => {
                let path = path.trim();
                if path.is_empty() {
                    return Err(expression_syntax_error());
                }
                return Ok((path.to_string(), cursor + 1));
            }
            '{' => return Err(expression_syntax_error()),
            _ => {
                path.push(current);
                cursor += 1;
            }
        }
    }
    Err(expression_syntax_error())
}

/// Scans an integer or float literal, with an optional leading `-`.
fn scan_number(chars: &[char], start: usize) -> Result<(Value, usize)> {
    let mut cursor = start;
    let mut text = String::new();
    if chars.get(cursor) == Some(&'-') {
        text.push('-');
        cursor += 1;
    }
    let mut digits = 0;
    while let Some(&current) = chars.get(cursor) {
        if current.is_ascii_digit() {
            text.push(current);
            digits += 1;
            cursor += 1;
        } else {
            break;
        }
    }
    if digits == 0 {
        return Err(expression_syntax_error());
    }
    if chars.get(cursor) == Some(&'.') {
        text.push('.');
        cursor += 1;
        let mut fraction = 0;
        while let Some(&current) = chars.get(cursor) {
            if current.is_ascii_digit() {
                text.push(current);
                fraction += 1;
                cursor += 1;
            } else {
                break;
            }
        }
        if fraction == 0 {
            return Err(expression_syntax_error());
        }
    }
    if !text.contains('.') {
        if let Ok(integer) = text.parse::<i64>() {
            return Ok((Value::from(integer), cursor));
        }
    }
    let float = text.parse::<f64>().map_err(|_| expression_syntax_error())?;
    let number = Number::from_f64(float).ok_or_else(expression_syntax_error)?;
    Ok((Value::Number(number), cursor))
}

/// Scans a keyword, boolean/null literal, or helper name.
fn scan_word(chars: &[char], start: usize) -> Result<(Token, usize)> {
    let mut cursor = start;
    let mut word = String::new();
    while let Some(&current) = chars.get(cursor) {
        if current.is_alphanumeric() || current == '_' {
            word.push(current);
            cursor += 1;
        } else {
            break;
        }
    }
    let token = match word.as_str() {
        "and" => Token::And,
        "or" => Token::Or,
        "not" => Token::Not,
        "in" => Token::In,
        "not_in" => Token::NotIn,
        "true" | "True" => Token::Literal(Value::Bool(true)),
        "false" | "False" => Token::Literal(Value::Bool(false)),
        "null" | "None" => Token::Literal(Value::Null),
        "len" | "length" => Token::Helper(HelperKind::Len),
        "is_empty" => Token::Helper(HelperKind::IsEmpty),
        "is_not_empty" => Token::Helper(HelperKind::IsNotEmpty),
        _ => return Err(expression_syntax_error()),
    };
    Ok((token, cursor))
}

/// A recursive-descent parser over a token vector.
struct Parser {
    tokens: Vec<Token>,
    position: usize,
    depth: usize,
}

impl Parser {
    /// Parses a whole expression, rejecting trailing tokens.
    fn parse(input: &str) -> Result<Expr> {
        let mut parser = Self {
            tokens: tokenize(input)?,
            position: 0,
            depth: 0,
        };
        let expression = parser.parse_expression()?;
        if parser.peek().is_some() {
            return Err(expression_syntax_error());
        }
        Ok(expression)
    }

    /// Parses an `||` chain; also the recursion point for parentheses and helper args.
    fn parse_expression(&mut self) -> Result<Expr> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(expression_syntax_error());
        }
        let parsed = self.parse_or();
        self.depth -= 1;
        parsed
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Some(Token::Or)) {
            self.position += 1;
            let rhs = self.parse_and()?;
            lhs = Expr::Or(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_comparison()?;
        while matches!(self.peek(), Some(Token::And)) {
            self.position += 1;
            let rhs = self.parse_comparison()?;
            lhs = Expr::And(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let lhs = self.parse_unary()?;
        let Some(operator) = self.take_operator() else {
            return Ok(lhs);
        };
        let rhs = self.parse_unary()?;
        let (lhs, rhs) = (Box::new(lhs), Box::new(rhs));
        Ok(match operator {
            Operator::Compare(op) => Expr::Compare(op, lhs, rhs),
            Operator::Member(check) => Expr::Membership(check, lhs, rhs),
        })
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(expression_syntax_error());
        }
        let parsed = if matches!(self.peek(), Some(Token::Not)) {
            self.position += 1;
            self.parse_unary()
                .map(|operand| Expr::Not(Box::new(operand)))
        } else {
            self.parse_primary()
        };
        self.depth -= 1;
        parsed
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.advance() {
            Some(Token::Literal(value)) => Ok(Expr::Literal(value)),
            Some(Token::Reference(path)) => Ok(Expr::Reference(path)),
            Some(Token::Helper(kind)) => self.parse_helper(kind),
            Some(Token::LParen) => {
                let inner = self.parse_expression()?;
                self.expect(|token| matches!(token, Token::RParen))?;
                Ok(inner)
            }
            Some(_) | None => Err(expression_syntax_error()),
        }
    }

    fn parse_helper(&mut self, kind: HelperKind) -> Result<Expr> {
        self.expect(|token| matches!(token, Token::LParen))?;
        let argument = self.parse_expression()?;
        self.expect(|token| matches!(token, Token::RParen))?;
        Ok(Expr::Helper(kind, Box::new(argument)))
    }

    /// Consumes the comparison operator at the cursor, including two-token `not in`.
    fn take_operator(&mut self) -> Option<Operator> {
        let operator = match self.peek()? {
            Token::Compare(op) => Operator::Compare(*op),
            Token::In => Operator::Member(MembershipCheck::In),
            Token::NotIn => Operator::Member(MembershipCheck::NotIn),
            Token::Not if matches!(self.peek_after(1), Some(Token::In)) => {
                self.position += 1;
                Operator::Member(MembershipCheck::NotIn)
            }
            _ => return None,
        };
        self.position += 1;
        Some(operator)
    }

    fn expect(&mut self, check: impl Fn(&Token) -> bool) -> Result<()> {
        if self.peek().is_some_and(check) {
            self.position += 1;
            Ok(())
        } else {
            Err(expression_syntax_error())
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn peek_after(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.position + offset)
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.position).cloned();
        if token.is_some() {
            self.position += 1;
        }
        token
    }
}

/// Evaluates an AST node against `source`.
fn eval(expr: &Expr, source: &Value) -> Result<Value> {
    match expr {
        Expr::Literal(value) => Ok(value.clone()),
        Expr::Reference(path) => Ok(resolve_path(source, path).unwrap_or(Value::Null)),
        Expr::Not(operand) => Ok(Value::Bool(!is_truthy(&eval(operand, source)?))),
        Expr::And(lhs, rhs) => {
            if !is_truthy(&eval(lhs, source)?) {
                return Ok(Value::Bool(false));
            }
            Ok(Value::Bool(is_truthy(&eval(rhs, source)?)))
        }
        Expr::Or(lhs, rhs) => {
            if is_truthy(&eval(lhs, source)?) {
                return Ok(Value::Bool(true));
            }
            Ok(Value::Bool(is_truthy(&eval(rhs, source)?)))
        }
        Expr::Compare(op, lhs, rhs) => compare(*op, &eval(lhs, source)?, &eval(rhs, source)?),
        Expr::Membership(check, lhs, rhs) => {
            let found = contains(&eval(lhs, source)?, &eval(rhs, source)?);
            let matches = match check {
                MembershipCheck::In => found,
                MembershipCheck::NotIn => !found,
            };
            Ok(Value::Bool(matches))
        }
        Expr::Helper(kind, arg) => eval_helper(*kind, &eval(arg, source)?),
    }
}

/// Applies a comparison operator, reporting incomparable operands as an evaluation error.
fn compare(op: CompareOp, lhs: &Value, rhs: &Value) -> Result<Value> {
    let result = match op {
        CompareOp::Eq => values_equal(lhs, rhs),
        CompareOp::Ne => !values_equal(lhs, rhs),
        CompareOp::Gt | CompareOp::Ge | CompareOp::Lt | CompareOp::Le => {
            let ordering = order(lhs, rhs).ok_or_else(|| {
                expression_eval_error(&format!(
                    "cannot compare {} {} {}",
                    type_name(lhs),
                    op.symbol(),
                    type_name(rhs)
                ))
            })?;
            op.satisfied_by(ordering)
        }
    };
    Ok(Value::Bool(result))
}

/// Orders two values of the same comparable JSON type; `None` when they are incomparable.
fn order(lhs: &Value, rhs: &Value) -> Option<Ordering> {
    match (lhs, rhs) {
        (Value::Number(left), Value::Number(right)) => numeric_order(left, right),
        (Value::String(left), Value::String(right)) => Some(left.cmp(right)),
        _ => None,
    }
}

/// Orders two JSON numbers, comparing integers exactly when possible.
fn numeric_order(left: &Number, right: &Number) -> Option<Ordering> {
    if let (Some(left), Some(right)) = (left.as_i64(), right.as_i64()) {
        return Some(left.cmp(&right));
    }
    if let (Some(left), Some(right)) = (left.as_u64(), right.as_u64()) {
        return Some(left.cmp(&right));
    }
    left.as_f64()?.partial_cmp(&right.as_f64()?)
}

/// Equality with numeric coercion, so `1` equals `1.0` like it does in Python.
fn values_equal(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::Number(left), Value::Number(right)) => {
            matches!(numeric_order(left, right), Some(Ordering::Equal))
        }
        _ => lhs == rhs,
    }
}

/// Membership that never errors: array equality, object keys, and string substrings.
fn contains(needle: &Value, haystack: &Value) -> bool {
    match haystack {
        Value::Array(items) => items.iter().any(|item| values_equal(needle, item)),
        Value::Object(map) => needle.as_str().is_some_and(|key| map.contains_key(key)),
        Value::String(text) => needle.as_str().is_some_and(|sub| text.contains(sub)),
        _ => false,
    }
}

/// Applies a value helper.
fn eval_helper(kind: HelperKind, argument: &Value) -> Result<Value> {
    match kind {
        HelperKind::Len => Ok(Value::from(collection_len(argument)?)),
        HelperKind::IsEmpty => Ok(Value::Bool(is_empty(argument)?)),
        HelperKind::IsNotEmpty => Ok(Value::Bool(!is_empty(argument)?)),
    }
}

/// `len(value)`: `null` counts as 0, strings count characters.
fn collection_len(value: &Value) -> Result<usize> {
    match value {
        Value::Null => Ok(0),
        Value::String(text) => Ok(text.chars().count()),
        Value::Array(items) => Ok(items.len()),
        Value::Object(map) => Ok(map.len()),
        other => Err(expression_eval_error(&format!(
            "len() operate on non-sequence value of type {}",
            type_name(other)
        ))),
    }
}

/// `is_empty(value)`: `null` is empty, scalars are an evaluation error.
fn is_empty(value: &Value) -> Result<bool> {
    match value {
        Value::Null => Ok(true),
        Value::String(text) => Ok(text.is_empty()),
        Value::Array(items) => Ok(items.is_empty()),
        Value::Object(map) => Ok(map.is_empty()),
        other => Err(expression_eval_error(&format!(
            "cannot check emptiness of {} type",
            type_name(other)
        ))),
    }
}

/// Truthiness of a bare value, used by `!`, `&&`, `||`, and `evaluate`.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// The JSON type name used in evaluation error messages.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A source covering the shapes the tests reference.
    fn state() -> Value {
        json!({"a": {
            "b": [1, 2, 3], "flag": false, "text": "hello", "count": 3,
            "items": [{"name": "x"}, {"name": "y"}], "empty": [], "obj": {},
        }})
    }

    fn cond(expression: &str) -> Condition {
        Condition::new(expression).unwrap()
    }

    fn check(expression: &str, source: &Value) -> bool {
        cond(expression).evaluate(source).unwrap()
    }

    /// Asserts every `(expression, expected)` case against `state()`.
    fn cases(entries: &[(&str, bool)]) {
        let source = state();
        for (expression, expected) in entries {
            assert_eq!(check(expression, &source), *expected, "{expression}");
        }
    }

    /// Asserts every case against an empty source.
    fn cases_empty(entries: &[(&str, bool)]) {
        let source = json!({});
        for (expression, expected) in entries {
            assert_eq!(check(expression, &source), *expected, "{expression}");
        }
    }

    fn syntax_error(expression: &str) {
        let error = Condition::new(expression).unwrap_err();
        assert!(
            error.message().contains("[101024]"),
            "{expression:?}: {error}"
        );
    }

    /// Asserts an evaluation error and returns its rendered message.
    fn eval_error(expression: &str, source: &Value) -> String {
        let error = cond(expression).evaluate(source).unwrap_err();
        assert!(
            error.message().contains("[101025]"),
            "{expression}: {error}"
        );
        error.message().into_owned()
    }

    #[test]
    fn and_binds_tighter_than_or() {
        cases_empty(&[
            ("true || false && false", true),
            ("true and true", true),
            ("false or true", true),
            ("(true || false) && false", false),
            ("false || false", false),
            ("true && false", false),
        ]);
    }

    #[test]
    fn not_binds_tighter_than_comparison() {
        cases(&[
            // `!` applies to the reference, so this is (!false) == true.
            ("!${a.flag} == true", true),
            ("!(${a.flag} == true)", true),
            ("not ${a.flag}", true),
            ("!(1 > 2)", true),
            ("!!${a.flag}", false),
        ]);
    }

    #[test]
    fn comparison_operators() {
        cases(&[
            ("${a.count} == 3", true),
            ("${a.count} != 4", true),
            ("${a.count} > 2", true),
            ("${a.count} >= 3", true),
            ("${a.count} < 4", true),
            ("${a.count} <= 3", true),
            ("${a.text} > 'abc'", true),
            ("${a.text} < 'abc'", false),
            ("${a.count} == 4", false),
            ("1 == 1.0", true),
            ("1 != 1.0", false),
            ("-1 < 0", true),
            ("2.5 > 2", true),
            ("3.0 <= 3", true),
            ("-2.5 == -2.5", true),
        ]);
    }

    #[test]
    fn membership_operators() {
        cases(&[
            ("1 in ${a.b}", true),
            ("4 not_in ${a.b}", true),
            ("4 not in ${a.b}", true),
            ("'name' in ${a.items[0]}", true),
            ("'nope' not_in ${a.obj}", true),
            ("'ell' in ${a.text}", true),
            // Membership never errors: a missing haystack is null, which holds
            // nothing, so `not in` is true and `in` is false.
            ("'x' not_in ${missing}", true),
            ("'zzz' in ${a.text}", false),
            ("1 in ${a.text}", false),
            ("null in ${a.count}", false),
            ("'x' in ${missing}", false),
        ]);
    }

    #[test]
    fn helper_functions() {
        cases(&[
            ("len(${a.text}) == 5", true),
            ("len(${a.b}) == 3", true),
            ("len(${a.obj}) == 0", true),
            ("len(${missing}) == 0", true),
            ("length(${a.text}) == 5", true),
            ("is_empty(${a.empty})", true),
            ("is_empty(${missing})", true),
            ("is_not_empty(${a.b})", true),
            ("!is_empty(${a.text})", true),
            ("is_empty(${a.text})", false),
            ("is_not_empty(${a.empty})", false),
        ]);
    }

    #[test]
    fn helper_type_errors_are_eval_errors() {
        let source = state();
        let message = eval_error("is_empty(${a.count})", &source);
        assert!(message.contains("cannot check emptiness of number"));
        let message = eval_error("len(${a.flag})", &source);
        assert!(message.contains("non-sequence value of type bool"));
    }

    #[test]
    fn ordering_across_types_is_an_eval_error() {
        let source = json!({});
        for expression in ["'a' > 1", "null < 1", "true > false", "len('a') > 'x'"] {
            eval_error(expression, &source);
        }
        // Equality across types is legal and simply false.
        cases_empty(&[("'a' == 1", false), ("'a' != 1", true)]);
    }

    #[test]
    fn missing_references_resolve_to_null() {
        cases(&[
            ("${missing} == null", true),
            ("!${missing}", true),
            ("${a.missing.deep} == None", true),
            ("${missing} != null", false),
            ("${missing}", false),
        ]);
    }

    #[test]
    fn references_support_indexes_and_projection() {
        cases(&[
            ("${a.b[0]} == 1", true),
            ("${a.b[-1]} == 3", true),
            ("${a.items[1].name} == 'y'", true),
            // Array projection is the documented reference extension.
            ("'y' in ${a.items.name}", true),
        ]);
    }

    #[test]
    fn string_literals_stay_literal() {
        cases(&[
            ("'${a.text}' == '${a.text}'", true),
            ("\"prefix ${a.text}\" == 'prefix ${a.text}'", true),
            ("len('${a.text}') == 9", true),
            ("'${a.text}' == ${a.text}", false),
        ]);
    }

    #[test]
    fn string_escapes() {
        cases_empty(&[
            (r#"'it\'s' == "it's""#, true),
            ("len(\"a\\\"b\") == 3", true),
            ("'line\\nbreak' == \"line\\nbreak\"", true),
            ("len('back\\\\slash') == 10", true),
        ]);
    }

    #[test]
    fn literal_aliases_and_truthiness() {
        cases(&[
            ("True && !False", true),
            ("None == null", true),
            ("-1", true),
            ("'x'", true),
            ("true", true),
            ("False || None", false),
            ("0", false),
            ("0.0", false),
            ("''", false),
            ("null", false),
            ("${a.obj}", false),
        ]);
    }

    #[test]
    fn short_circuit_skips_evaluation_errors() {
        cases(&[
            ("!(false && is_empty(${a.count}))", true),
            ("true || is_empty(${a.count})", true),
            ("false && is_empty(${a.count})", false),
        ]);
    }

    #[test]
    fn syntax_errors_report_code_101024() {
        for expression in [
            "",
            "   ",
            "true &&",
            "(true",
            "true)",
            "true @ false",
            "true === false",
            "&& true",
            "unknown_identifier",
            "len",
            "len(true",
            "len(true, false)",
            "${}",
            "${unclosed",
            "$unclosed",
            "1 < 2 < 3",
            "-",
            "1.",
        ] {
            syntax_error(expression);
        }
    }

    #[test]
    fn condition_exposes_source_and_derives() {
        let expression = "  ${a.count} > 2  ";
        let parsed = cond(expression);
        assert_eq!(parsed.expression(), expression);
        assert!(format!("{:?}", parsed.clone()).contains("count"));
    }

    #[test]
    fn expression_length_and_nesting_caps() {
        let body = "a".repeat(MAX_EXPRESSION_LENGTH - 2);
        let at_limit = format!("'{body}'");
        assert_eq!(at_limit.chars().count(), MAX_EXPRESSION_LENGTH);
        assert!(check(&at_limit, &json!({})));
        syntax_error(&format!("'{body}x'"));
        syntax_error(&format!("{}true{}", "(".repeat(500), ")".repeat(500)));
    }

    #[test]
    fn router_returns_the_first_matching_branch_targets() {
        let mut router = BranchRouter::new();
        let _ = router
            .add_branch(Branch::new(cond("${a.count} > 10"), vec!["big".into()]))
            .add_branch(Branch::new(cond("${a.count} > 1"), vec!["medium".into()]))
            .add_branch(Branch::new(cond("${a.count} > 0"), vec!["small".into()]));
        // The second and third branches both match; insertion order decides.
        assert_eq!(router.route(&state()).unwrap(), vec!["medium".to_string()]);
    }

    #[test]
    fn router_default_target_and_no_match_error() {
        let mut router = BranchRouter::new();
        let _ = router
            .add_branch(Branch::new(cond("${a.count} > 10"), vec!["big".into()]))
            .default_target(vec!["fallback".into()]);
        assert_eq!(
            router.route(&state()).unwrap(),
            vec!["fallback".to_string()]
        );

        let error = BranchRouter::new().route(&state()).unwrap_err();
        assert_eq!(
            error.message(),
            "[101021] component branch execution error, \
             error='branch meeting the condition was not found'"
        );

        // A failing condition propagates instead of being treated as no match.
        let mut failing = BranchRouter::new();
        let _ = failing.add_branch(Branch::new(
            cond("is_empty(${a.count})"),
            vec!["bad".into()],
        ));
        assert!(failing
            .route(&state())
            .unwrap_err()
            .message()
            .contains("[101025]"));
    }

    #[test]
    fn router_targets_preserve_order_and_deduplicate() {
        let mut router = BranchRouter::new();
        let _ = router
            .add_branch(Branch::new(cond("true"), vec!["a".into(), "b".into()]))
            .add_branch(Branch::new(cond("true"), vec!["b".into(), "c".into()]))
            .default_target(vec!["d".into(), "a".into()]);
        assert_eq!(router.targets(), vec!["a", "b", "c", "d"]);
        assert_eq!(BranchRouter::default().targets(), Vec::<String>::new());

        let named = Branch::new(cond("true"), vec!["x".into()]).with_name("named");
        assert_eq!(named.name.as_deref(), Some("named"));
        assert!(format!("{:?}", named.clone()).contains("named"));
    }
}
