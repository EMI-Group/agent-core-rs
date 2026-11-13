//! The deterministic per-node-type DL format checks behind `DlReflector::validate_dl`.
//!
//! Ports the `Reflector` class of
//! `openjiuwen/dev_tools/agent_builder/builders/workflow/dl_reflector.py` onto the
//! typed [`DlDocument`] model: every rule is a pure inspection of one node, so a
//! validation pass never mutates the document, never calls a model, and never
//! fails.
//!
//! # Check order
//!
//! Nodes are visited in document order. Inside a node the checks run as: blank
//! `id`, duplicate `id`, blank `type`, blank `description`, output declaration,
//! the node type, then the per-type rules. A node whose `type` is not one of the
//! nine [`DlNodeType`] wire names records `UnknownNodeType` and skips its
//! per-type rules, because those rules are type-specific. A blank `type` records
//! only `MissingField` (the type is absent rather than unrecognized).
//!
//! Node ids are collected while walking, so `DuplicateNodeId` names the node
//! whose id repeats. Outputs are declared as `"{node_id}.{output_name}"` when the
//! node is reached, so a branch expression may only reference an output declared
//! by an earlier or the current node. A blank `id` is reported against the id the
//! document carries (the empty string), so every node-level issue names a node.
//!
//! After the walk every non-blank `next` link — the node's own `next` and each
//! condition's `next` — that does not name an existing node id records
//! `DanglingEdge`. The message names the missing id; the issue's `node_id` is the
//! node holding the link.
//!
//! # Reference checks
//!
//! Only branch conditions resolve `${...}` placeholders. The branch operator is
//! the second whitespace-separated token of the expression, and the operator must
//! be one of the ten operators Python's `Reflector` accepts. Every placeholder's
//! inner text must be a declared `"{node_id}.{output_name}"` string. Matching is
//! case-sensitive, exactly as Python compares its wire strings.
//!
//! [`DlDocument`]: crate::agent_builder::dl_transformer::model::DlDocument
//! [`DlNodeType`]: crate::agent_builder::dl_transformer::model::DlNodeType

use std::collections::BTreeSet;

use serde_json::Value;

use super::{DlValidation, ReflectionIssue, ReflectionIssueKind};
use crate::agent_builder::dl_transformer::model::{DlCondition, DlDocument, DlNode, DlNodeType};

/// The branch operators Python's `Reflector` accepts as the second token.
const BRANCH_OPERATORS: [&str; 10] = [
    "eq",
    "not_eq",
    "contain",
    "not_contain",
    "longer_than",
    "longer_than_or_eq",
    "short_than",
    "short_than_or_eq",
    "is_empty",
    "is_not_empty",
];

/// The branch-expression marker naming the fallback branch.
const DEFAULT_BRANCH: &str = "default";

/// The description Python requires on the Start node's `query` output.
const START_QUERY_DESCRIPTION: &str = "用户输入";

/// Validates `document` and reports every issue found.
///
/// See the module documentation for the check order and the reference rules.
pub(super) fn validate_dl(document: &DlDocument) -> DlValidation {
    let mut checks = Checks::default();
    for node in &document.nodes {
        checks.check_node(node);
    }
    checks.check_links(document);
    DlValidation {
        passed: checks.issues.is_empty(),
        issues: checks.issues,
    }
}

/// The mutable state of one validation pass.
#[derive(Debug, Default)]
struct Checks {
    /// `"{node_id}.{output_name}"` for every output declared so far.
    declared_outputs: BTreeSet<String>,
    /// Every node id met so far, for duplicate detection.
    seen_ids: BTreeSet<String>,
    /// The issues found so far, in node order then check order.
    issues: Vec<ReflectionIssue>,
}

impl Checks {
    /// Records one issue against `node`.
    fn node_issue(&mut self, kind: ReflectionIssueKind, node: &DlNode, message: String) {
        self.issues.push(ReflectionIssue {
            kind,
            message,
            node_id: Some(node.id.clone()),
        });
    }

    /// Records a `MissingField` issue naming the node and the field.
    fn missing(&mut self, node: &DlNode, field: &str, problem: &str) {
        self.node_issue(
            ReflectionIssueKind::MissingField,
            node,
            format!("node `{}`: required field `{field}` {problem}", node.id),
        );
    }

    /// Records a `MissingField` issue for one condition field.
    fn condition_missing(&mut self, node: &DlNode, index: usize, field: &str, problem: &str) {
        self.missing(
            node,
            &format!("parameters.conditions[{index}].{field}"),
            problem,
        );
    }

    /// Runs the basic checks, then the per-type checks, for one node.
    fn check_node(&mut self, node: &DlNode) {
        if node.id.trim().is_empty() {
            self.missing(node, "id", "is blank");
        }
        if !self.seen_ids.insert(node.id.clone()) {
            let id = &node.id;
            self.node_issue(
                ReflectionIssueKind::DuplicateNodeId,
                node,
                format!("node `{id}`: duplicate node id `{id}`"),
            );
        }
        let blank_type = node.node_type.trim().is_empty();
        if blank_type {
            self.missing(node, "type", "is blank");
        }
        if node.description.trim().is_empty() {
            self.missing(node, "description", "is blank");
        }

        self.declare_outputs(node);

        let kind = match node.kind() {
            Ok(kind) => kind,
            Err(_) if blank_type => return,
            Err(_) => {
                let node_type = &node.node_type;
                self.node_issue(
                    ReflectionIssueKind::UnknownNodeType,
                    node,
                    format!("node `{}`: unknown DL node type `{node_type}`", node.id),
                );
                return;
            }
        };
        match kind {
            DlNodeType::Start => self.check_start(node),
            DlNodeType::End => self.check_end(node),
            DlNodeType::Llm => self.check_llm(node),
            DlNodeType::Branch => self.check_branch(node),
            DlNodeType::IntentDetection => self.check_intent_detection(node),
            DlNodeType::Questioner => self.check_questioner(node),
            DlNodeType::Output => self.check_output(node),
            DlNodeType::Code => self.check_code(node),
            DlNodeType::Plugin => self.check_plugin(node),
        }
    }

    /// Declares `"{id}.{name}"` for every output of `node`.
    fn declare_outputs(&mut self, node: &DlNode) {
        for output in &node.parameters.outputs {
            self.declared_outputs
                .insert(format!("{}.{}", node.id, output.name));
        }
    }

    /// Requires a non-empty `parameters.inputs` list.
    fn check_inputs(&mut self, node: &DlNode) {
        if node.parameters.inputs.is_empty() {
            self.missing(node, "parameters.inputs", "must be a non-empty list");
        }
    }

    /// Requires a non-empty `parameters.outputs` list.
    fn check_outputs(&mut self, node: &DlNode) {
        if node.parameters.outputs.is_empty() {
            self.missing(node, "parameters.outputs", "must be a non-empty list");
        }
    }

    /// Requires a non-blank string under `parameters.configs[key]`.
    fn check_config(&mut self, node: &DlNode, key: &str) {
        let non_blank =
            node.parameters.configs.get(key).is_some_and(
                |value| matches!(value, Value::String(text) if !text.trim().is_empty()),
            );
        if !non_blank {
            self.missing(
                node,
                &format!("parameters.configs.{key}"),
                "must be a non-blank string",
            );
        }
    }

    /// Requires a present, non-blank `next` link.
    fn check_next(&mut self, node: &DlNode) {
        if blank(node.next.as_deref()) {
            self.missing(node, "next", "is missing or blank");
        }
    }

    /// Checks a Start node: the `query` output and the `next` link.
    fn check_start(&mut self, node: &DlNode) {
        let has_query = node.parameters.outputs.iter().any(|output| {
            output.name == "query" && output.description.as_deref() == Some(START_QUERY_DESCRIPTION)
        });
        if !has_query {
            self.missing(
                node,
                "parameters.outputs",
                "must declare `query` with description `用户输入`",
            );
        }
        self.check_next(node);
    }

    /// Checks an End node: inputs and the `template` configuration.
    fn check_end(&mut self, node: &DlNode) {
        self.check_inputs(node);
        self.check_config(node, "template");
    }

    /// Checks an Output node: inputs, `template`, and `next`.
    fn check_output(&mut self, node: &DlNode) {
        self.check_inputs(node);
        self.check_config(node, "template");
        self.check_next(node);
    }

    /// Checks an LLM node: variables, both prompts, and `next`.
    fn check_llm(&mut self, node: &DlNode) {
        self.check_inputs(node);
        self.check_outputs(node);
        self.check_config(node, "system_prompt");
        self.check_config(node, "user_prompt");
        self.check_next(node);
    }

    /// Checks a Questioner node: variables, `prompt`, and `next`.
    fn check_questioner(&mut self, node: &DlNode) {
        self.check_inputs(node);
        self.check_outputs(node);
        self.check_config(node, "prompt");
        self.check_next(node);
    }

    /// Checks a Plugin node: variables, `tool_id`, and `next`.
    fn check_plugin(&mut self, node: &DlNode) {
        self.check_inputs(node);
        self.check_outputs(node);
        self.check_config(node, "tool_id");
        self.check_next(node);
    }

    /// Checks a Code node: variables, `code`, and `next`.
    fn check_code(&mut self, node: &DlNode) {
        self.check_inputs(node);
        self.check_outputs(node);
        self.check_config(node, "code");
        self.check_next(node);
    }

    /// Checks an `IntentDetection` node: inputs, prompt, and intent conditions.
    fn check_intent_detection(&mut self, node: &DlNode) {
        self.check_inputs(node);
        self.check_config(node, "prompt");
        if node.parameters.conditions.is_empty() {
            self.missing(node, "parameters.conditions", "must be a non-empty list");
            return;
        }

        let anchor = format!("${{{}.rawOutput}}", node.id);
        let mut has_default = false;
        for (index, condition) in node.parameters.conditions.iter().enumerate() {
            self.check_condition_fields(node, index, condition);
            let expression = condition.expression.as_deref().unwrap_or("");
            if expression == DEFAULT_BRANCH {
                has_default = true;
                continue;
            }
            if !expression.contains(&anchor) {
                self.condition_missing(
                    node,
                    index,
                    "expression",
                    &format!("must reference `{anchor}`"),
                );
            }
            if !expression.contains("contain") {
                self.node_issue(
                    ReflectionIssueKind::InvalidExpression,
                    node,
                    format!(
                        "node `{}`: condition {index} `expression` must use the `contain` operator",
                        node.id
                    ),
                );
            }
        }

        if !has_default {
            self.missing_default_branch(node);
        }
    }

    /// Checks a Branch node: route names, expressions, references, and default.
    fn check_branch(&mut self, node: &DlNode) {
        if node.parameters.conditions.is_empty() {
            self.missing(node, "parameters.conditions", "must be a non-empty list");
            return;
        }

        let mut has_default = false;
        for (index, condition) in node.parameters.conditions.iter().enumerate() {
            self.check_condition_fields(node, index, condition);
            let expression = condition.expression.as_deref();
            if expression == Some(DEFAULT_BRANCH) {
                has_default = true;
            }
            match (expression, condition.expressions.as_ref()) {
                (Some(single), None) => {
                    if single != DEFAULT_BRANCH {
                        self.check_branch_expression(node, index, single);
                    }
                }
                (None, Some(expressions)) => {
                    for expression in expressions {
                        self.check_branch_expression(node, index, expression);
                    }
                }
                _ => {
                    self.node_issue(
                        ReflectionIssueKind::MissingConditionExpression,
                        node,
                        format!(
                            "node `{}`: condition {index} must declare exactly one of \
                             `expression` or `expressions`",
                            node.id
                        ),
                    );
                }
            }
        }

        if !has_default {
            self.missing_default_branch(node);
        }
    }

    /// Requires non-blank `branch`, `description`, and `next` condition fields.
    fn check_condition_fields(&mut self, node: &DlNode, index: usize, condition: &DlCondition) {
        for (field, value) in [
            ("branch", condition.branch.as_str()),
            ("description", condition.description.as_str()),
        ] {
            if value.trim().is_empty() {
                self.condition_missing(node, index, field, "is missing or blank");
            }
        }
        if blank(condition.next.as_deref()) {
            self.condition_missing(node, index, "next", "is missing or blank");
        }
    }

    /// Records the mandatory fallback branch of a branching node.
    fn missing_default_branch(&mut self, node: &DlNode) {
        self.node_issue(
            ReflectionIssueKind::MissingDefaultBranch,
            node,
            format!(
                "node `{}`: the conditions must declare a `default` branch",
                node.id
            ),
        );
    }

    /// Checks one branch expression's operator and `${...}` references.
    fn check_branch_expression(&mut self, node: &DlNode, index: usize, expression: &str) {
        match expression.split_whitespace().nth(1) {
            None => {
                self.node_issue(
                    ReflectionIssueKind::InvalidExpression,
                    node,
                    format!(
                        "node `{}`: condition {index} `expression` `{expression}` has no operator",
                        node.id
                    ),
                );
            }
            Some(operator) if !BRANCH_OPERATORS.contains(&operator) => {
                self.node_issue(
                    ReflectionIssueKind::InvalidExpression,
                    node,
                    format!(
                        "node `{}`: condition {index} uses the unsupported operator `{operator}`",
                        node.id
                    ),
                );
            }
            Some(_) => {}
        }

        for placeholder in placeholders(expression) {
            if !self.declared_outputs.contains(placeholder) {
                self.node_issue(
                    ReflectionIssueKind::InvalidReference,
                    node,
                    format!(
                        "node `{}`: condition {index} references the undeclared variable `${{{placeholder}}}`",
                        node.id
                    ),
                );
            }
        }
    }

    /// Records a `DanglingEdge` for every link naming a missing node.
    fn check_links(&mut self, document: &DlDocument) {
        let ids: BTreeSet<&str> = document
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .filter(|id| !id.trim().is_empty())
            .collect();

        for node in &document.nodes {
            if let Some(next) = dangling(node.next.as_deref(), &ids) {
                self.node_issue(
                    ReflectionIssueKind::DanglingEdge,
                    node,
                    format!("node `{}`: `next` names the unknown node `{next}`", node.id),
                );
            }
            for (index, condition) in node.parameters.conditions.iter().enumerate() {
                if let Some(next) = dangling(condition.next.as_deref(), &ids) {
                    self.node_issue(
                        ReflectionIssueKind::DanglingEdge,
                        node,
                        format!(
                            "node `{}`: condition {index} `next` names the unknown node `{next}`",
                            node.id
                        ),
                    );
                }
            }
        }
    }
}

/// Reports `true` when a `next` link is absent or blank.
fn blank(link: Option<&str>) -> bool {
    link.is_none_or(|next| next.trim().is_empty())
}

/// The `next` link when it is non-blank and names no existing node.
fn dangling<'a>(link: Option<&'a str>, ids: &BTreeSet<&str>) -> Option<&'a str> {
    link.filter(|next| !next.trim().is_empty() && !ids.contains(*next))
}

/// The inner text of every well-formed `${...}` placeholder, in order.
///
/// An unclosed `${` ends the scan, an empty `${}` is ignored, and a nested `${`
/// restarts the scan there, so a malformed outer placeholder does not swallow
/// the references inside it.
fn placeholders(expression: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = expression[cursor..].find("${") {
        let start = cursor + offset;
        let body = &expression[start + 2..];
        let Some(end) = body.find('}') else {
            break;
        };
        let inner = &body[..end];
        match inner.find("${") {
            Some(nested) => cursor = start + 2 + nested,
            None if inner.is_empty() => cursor = start + 2,
            None => {
                found.push(inner);
                cursor = start + 2 + end + 1;
            }
        }
    }
    found
}
