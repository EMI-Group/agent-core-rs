//! [`ToolBinding`]: resolves the tools a case declares to runnable stubs.
//!
//! Python's tuning loop hands one agent both the parameter binding and the tools
//! it may call, and `TraceNode.tools` records that declaration set. The Rust port
//! already has one tool model — [`Case::tools`](crate::tune::Case::tools) declares
//! [`ToolInfo`](openjiuwen_foundation::tool::ToolInfo) descriptors and
//! [`ToolStubs`] holds the runnable fixtures — so this module adds only the join
//! between them: which declared tools have a stub, which do not, and what a run
//! actually used.
//!
//! A missing stub is reported, not invented: [`ToolBinding::bind_case`] is
//! permissive (a partial binding is still useful for diagnostics), while
//! [`ToolBinding::validate_case`] is the strict entry point for a caller that
//! wants an error before running anything.
use crate::tune::tool::{ToolStubs, ToolTrace};
use crate::tune::Case;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
/// Which of a case's declared tools a [`ToolBinding`] can actually run.
///
/// Serializes with `snake_case` field names, ignores unknown fields, and defaults
/// every field, so a partial payload still parses.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolBindingReport {
    /// The tool names the case declared, in declaration order, deduplicated.
    pub declared: Vec<String>,
    /// The declared names that have a stub, in declaration order.
    pub bound: Vec<String>,
    /// The declared names without a stub, sorted.
    ///
    /// Sorted rather than declaration-ordered so two reports over the same
    /// declaration set compare equal regardless of authoring order.
    pub missing: Vec<String>,
}
impl ToolBindingReport {
    /// Whether every declared tool has a stub (true for a tool-free case).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }
    /// How many declared tools lack a stub.
    #[must_use]
    pub fn missing_count(&self) -> usize {
        self.missing.len()
    }
}
/// The join between a case's declared tools and the stubs that can run them.
///
/// Holds no per-case state: the same binding serves every case, and every report
/// is derived from the case's own declaration list.
#[derive(Debug, Clone, Default)]
pub struct ToolBinding {
    /// The runnable fixtures, keyed by tool name.
    stubs: ToolStubs,
}
impl ToolBinding {
    /// Wraps the stub table.
    pub fn new(stubs: ToolStubs) -> Self {
        Self { stubs }
    }
    /// The wrapped stub table.
    #[must_use]
    pub fn tool_stubs(&self) -> &ToolStubs {
        &self.stubs
    }
    /// The tool names a case declares, in declaration order, deduplicated.
    ///
    /// A case declares [`ToolInfo`](openjiuwen_foundation::tool::ToolInfo)
    /// descriptors, not handlers, so it can never produce stubs on its own; this
    /// helper exists to spell that out at the call site that reads the
    /// declaration list.
    #[must_use]
    pub fn declare_only(case: &Case) -> Vec<String> {
        case.tool_names().into_iter().map(str::to_owned).collect()
    }
    /// Partitions a case's declared tools into bound and missing.
    ///
    /// A case that declares no tools reports three empty vectors and is not an
    /// error; a partial binding reports the unbound names in
    /// [`missing`](ToolBindingReport::missing) instead of failing, so the caller
    /// can decide whether to run with what is bound.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the case declares a blank tool name: a report
    /// keyed by an empty name could not be matched against a stub.
    pub fn bind_case(&self, case: &Case) -> Result<ToolBindingReport> {
        let declared = Self::declare_only(case);
        if let Some(blank) = declared.iter().find(|name| name.trim().is_empty()) {
            return Err(Error::validation(format!(
                "case '{}' declares a blank tool name (offending name: {blank:?})",
                case.case_id
            )));
        }
        let mut bound: Vec<String> = Vec::with_capacity(declared.len());
        let mut missing: Vec<String> = Vec::new();
        for name in &declared {
            if self.stubs.contains(name) {
                bound.push(name.clone());
            } else {
                missing.push(name.clone());
            }
        }
        missing.sort();
        Ok(ToolBindingReport {
            declared,
            bound,
            missing,
        })
    }
    /// Rejects a case whose declared tools are not all bound.
    ///
    /// The strict counterpart to [`bind_case`](Self::bind_case), which tolerates a
    /// partial binding: this reports the *first* unbound name in declaration order
    /// (plus how many are missing) so an author sees the tool they wrote first,
    /// while the report keeps `missing` sorted for stable comparisons.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the case, the first unbound tool, and the count
    /// of unbound tools; also when the case declares a blank tool name.
    pub fn validate_case(&self, case: &Case) -> Result<()> {
        let report = self.bind_case(case)?;
        if report.missing.is_empty() {
            return Ok(());
        }
        let first = report
            .declared
            .iter()
            .find(|name| report.missing.contains(*name));
        let tool = first.map_or("<unknown>", String::as_str);
        Err(Error::validation(format!(
            "case '{}' declares unbound tool '{tool}' ({} of {} declared tools lack a stub)",
            case.case_id,
            report.missing.len(),
            report.declared.len()
        )))
    }
    /// The per-tool call counts of one run, delegating to [`ToolTrace::usage`].
    ///
    /// The binding is not consulted — a trace can mention any name — so this is an
    /// associated convenience that keeps tool accounting on the same value as
    /// binding and validation.
    #[allow(clippy::unused_self)] // `&self` keeps the call site uniform with `bind_case`/`validate_case`
    pub fn usage(&self, trace: &ToolTrace) -> BTreeMap<String, usize> {
        trace.usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::tool::{ToolCallRecord, ToolStub};
    use openjiuwen_foundation::tool::ToolInfo;
    use serde_json::json;

    fn stubs() -> ToolStubs {
        ToolStubs::new()
            .with_stub(ToolStub::constant("lookup", json!({"answer": "Paris"})).unwrap())
            .with_stub(ToolStub::constant("search", json!({"results": []})).unwrap())
    }

    fn case(tools: &[&str]) -> Case {
        let declarations: Vec<ToolInfo> = tools
            .iter()
            .map(|name| ToolInfo::new(*name, "does something", json!({})))
            .collect();
        Case::new(
            BTreeMap::from([("query".to_owned(), json!("capital of France"))]),
            BTreeMap::from([("output".to_owned(), json!("Paris"))]),
        )
        .unwrap()
        .with_case_id("c1")
        .with_tools(declarations)
    }

    #[test]
    fn bind_case_partitions_declared_tools() {
        let binding = ToolBinding::new(stubs());
        let report = binding
            .bind_case(&case(&["lookup", "weather", "search"]))
            .unwrap();
        assert_eq!(
            report.declared,
            vec![
                "lookup".to_owned(),
                "weather".to_owned(),
                "search".to_owned()
            ],
            "declaration order is preserved"
        );
        assert_eq!(
            report.bound,
            vec!["lookup".to_owned(), "search".to_owned()],
            "bound keeps declaration order"
        );
        assert_eq!(
            report.missing,
            vec!["weather".to_owned()],
            "missing is what is left"
        );
        assert!(!report.is_complete());
        assert_eq!(report.missing_count(), 1);
    }

    #[test]
    fn bind_case_sorts_missing_names() {
        let report = ToolBinding::new(stubs())
            .bind_case(&case(&["zebra", "lookup", "apple", "mango"]))
            .unwrap();
        assert_eq!(
            report.missing,
            vec!["apple".to_owned(), "mango".to_owned(), "zebra".to_owned()]
        );
        assert_eq!(report.bound, vec!["lookup".to_owned()]);
    }

    #[test]
    fn bind_case_accepts_a_tool_free_case() {
        let report = ToolBinding::new(stubs()).bind_case(&case(&[])).unwrap();
        assert_eq!(report, ToolBindingReport::default());
        assert!(report.is_complete());
        assert_eq!(report.missing_count(), 0);

        let report = ToolBinding::new(ToolStubs::new())
            .bind_case(&case(&[]))
            .unwrap();
        assert_eq!(report.declared, Vec::<String>::new());
    }

    #[test]
    fn bind_case_reports_every_tool_missing_without_stubs() {
        let report = ToolBinding::new(ToolStubs::new())
            .bind_case(&case(&["lookup", "search"]))
            .unwrap();
        assert_eq!(report.bound, Vec::<String>::new());
        assert_eq!(
            report.missing,
            vec!["lookup".to_owned(), "search".to_owned()]
        );
        assert!(!report.is_complete());
    }

    #[test]
    fn bind_case_deduplicates_a_repeated_declaration() {
        let report = ToolBinding::new(stubs())
            .bind_case(&case(&["lookup", "lookup"]))
            .unwrap();
        assert_eq!(report.declared, vec!["lookup".to_owned()]);
        assert_eq!(report.bound, vec!["lookup".to_owned()]);
        assert_eq!(report.missing, Vec::<String>::new());
    }

    #[test]
    fn bind_case_rejects_a_blank_declared_name() {
        let error = ToolBinding::new(stubs())
            .bind_case(&case(&["  "]))
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        let message = error.to_string();
        assert!(message.contains("case 'c1'"), "{message}");
        assert!(message.contains("blank tool name"), "{message}");
    }

    #[test]
    fn validate_case_is_strict_about_missing_tools() {
        let binding = ToolBinding::new(stubs());
        binding.validate_case(&case(&["lookup", "search"])).unwrap();
        binding.validate_case(&case(&[])).unwrap();

        let error = binding
            .validate_case(&case(&["weather", "lookup", "climate"]))
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        let message = error.to_string();
        assert!(message.contains("case 'c1'"), "{message}");
        assert!(
            message.contains("unbound tool 'weather'"),
            "the first declared missing tool is named: {message}"
        );
        assert!(message.contains("2 of 3"), "{message}");

        let error = ToolBinding::new(ToolStubs::new())
            .validate_case(&case(&["lookup"]))
            .unwrap_err();
        assert!(error.to_string().contains("1 of 1"));
    }

    #[test]
    fn declare_only_is_the_cases_declaration_list() {
        assert_eq!(
            ToolBinding::declare_only(&case(&["lookup", "search", "lookup"])),
            vec!["lookup".to_owned(), "search".to_owned()]
        );
        assert_eq!(ToolBinding::declare_only(&case(&[])), Vec::<String>::new());
    }

    #[test]
    fn usage_delegates_to_the_trace() {
        let binding = ToolBinding::new(stubs());
        let trace = ToolTrace::new([
            ToolCallRecord::success("lookup", json!({"q": "x"}), json!({"answer": "Paris"})),
            ToolCallRecord::success("lookup", json!({"q": "y"}), json!({"answer": "Rome"})),
            ToolCallRecord::failure("search", json!({}), "timeout"),
        ]);
        assert_eq!(
            binding.usage(&trace),
            BTreeMap::from([("lookup".to_owned(), 2), ("search".to_owned(), 1)])
        );
        assert!(binding.usage(&ToolTrace::default()).is_empty());
    }

    #[test]
    fn bound_tools_are_callable_through_the_table() {
        let binding = ToolBinding::new(stubs());
        let case = case(&["lookup"]);
        binding.validate_case(&case).unwrap();
        assert_eq!(binding.tool_stubs().names(), vec!["lookup", "search"]);
        assert_eq!(
            binding
                .tool_stubs()
                .call("lookup", &json!({"q": "capital of France"}))
                .unwrap(),
            json!({"answer": "Paris"})
        );
        let error = binding
            .tool_stubs()
            .call("weather", &json!({}))
            .unwrap_err();
        assert!(matches!(error, Error::NotFound(_)));
        assert!(error.to_string().contains("weather"));
    }

    #[test]
    fn tool_binding_report_round_trips_with_defaults() {
        let report = ToolBinding::new(stubs())
            .bind_case(&case(&["lookup", "weather"]))
            .unwrap();
        let encoded = serde_json::to_value(&report).unwrap();
        assert_eq!(encoded["declared"], json!(["lookup", "weather"]));
        assert_eq!(encoded["bound"], json!(["lookup"]));
        assert_eq!(encoded["missing"], json!(["weather"]));
        let decoded: ToolBindingReport = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, report);

        let decoded: ToolBindingReport =
            serde_json::from_value(json!({"bound": ["lookup"], "extra": true})).unwrap();
        assert_eq!(decoded.bound, vec!["lookup".to_owned()]);
        assert_eq!(decoded.declared, Vec::<String>::new());
        assert_eq!(decoded.missing, Vec::<String>::new());
        assert!(decoded.is_complete());
    }
}
