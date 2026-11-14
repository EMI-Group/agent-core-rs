//! Human-readable rendering of normalised LSP results.
//!
//! Ports `openjiuwen/harness/tools/lsp_tool/_formatter.py`. Every renderer is pure and
//! deterministic, so the tool payload text is stable across servers that return the same
//! locations. Positions are shown one-based (`line:column`), matching editors and the
//! model-facing convention.
use super::diagnostics::DiagnosticsFile;
use super::protocol::{CompletionItem, DocumentSymbol, Hover, Location, SymbolInformation};
use super::uri::uri_to_path;
use super::wire::{
    CallHierarchyIncomingCall, CallHierarchyItem, CallHierarchyOutgoingCall, Diagnostic,
    DiagnosticCode,
};

/// Maximum completion items included in one rendered result.
const MAX_COMPLETION_ITEMS: usize = 50;
/// Maximum diagnostics included in one rendered result.
const MAX_DIAGNOSTICS: usize = 50;
/// Severity assumed for a diagnostic (or an ordered list entry) without one.
const DEFAULT_SEVERITY: u32 = 3;

/// Returns the LSP `SymbolKind` name, or `"?"` for an unknown kind.
#[must_use]
pub fn symbol_kind_name(kind: u32) -> &'static str {
    match kind {
        1 => "File",
        2 => "Module",
        3 => "Namespace",
        4 => "Package",
        5 => "Class",
        6 => "Method",
        7 => "Property",
        8 => "Field",
        9 => "Constructor",
        10 => "Enum",
        11 => "Interface",
        12 => "Function",
        13 => "Variable",
        14 => "Constant",
        15 => "String",
        16 => "Number",
        17 => "Boolean",
        18 => "Array",
        19 => "Object",
        20 => "Key",
        21 => "Null",
        22 => "EnumMember",
        23 => "Event",
        24 => "Operator",
        25 => "TypeParameter",
        _ => "?",
    }
}

/// Returns the LSP `CompletionItemKind` name, or `"?"` for an unknown kind.
#[must_use]
pub fn completion_kind_name(kind: u32) -> &'static str {
    match kind {
        1 => "Text",
        2 => "Method",
        3 => "Function",
        4 => "Constructor",
        5 => "Field",
        6 => "Variable",
        7 => "Class",
        8 => "Interface",
        9 => "Module",
        10 => "Property",
        11 => "Unit",
        12 => "Value",
        13 => "Enum",
        14 => "Keyword",
        15 => "Snippet",
        16 => "Color",
        17 => "File",
        18 => "Reference",
        19 => "Folder",
        20 => "EnumMember",
        21 => "Constant",
        22 => "Struct",
        23 => "Event",
        24 => "Operator",
        25 => "TypeParameter",
        _ => "?",
    }
}

/// Converts a `file://` URI into a display path, leaving non-file URIs unchanged.
#[must_use]
pub fn display_uri(uri: &str) -> String {
    uri_to_path(uri).to_string_lossy().into_owned()
}

/// Formats one location as `path:line:column` (one-based).
#[must_use]
pub fn format_location(location: &Location) -> String {
    let (line, column) = location.range.start.to_one_based();
    format!("{}:{line}:{column}", display_uri(&location.uri))
}

/// Renders a hover result.
#[must_use]
pub fn render_hover(hover: Option<&Hover>) -> String {
    match hover {
        Some(hover) if !hover.contents.trim().is_empty() => hover.contents.trim().to_owned(),
        _ => "No hover information found.".to_owned(),
    }
}

/// Renders definition locations.
#[must_use]
pub fn render_definition(locations: &[Location]) -> String {
    match locations {
        [] => "No definition found.".to_owned(),
        [location] => format!("Defined in {}", format_location(location)),
        many => {
            let lines: Vec<String> = many
                .iter()
                .map(|location| format!("  {}", format_location(location)))
                .collect();
            format!("Found {} definitions:\n{}", many.len(), lines.join("\n"))
        }
    }
}

/// Renders reference locations grouped by file, one-based.
#[must_use]
pub fn render_references(locations: &[Location]) -> String {
    if locations.is_empty() {
        return "No references found.".to_owned();
    }
    let mut lines = Vec::new();
    for (uri, group) in group_by_uri(locations) {
        lines.push(format!("{}:", display_uri(uri)));
        for location in group {
            let (line, column) = location.range.start.to_one_based();
            lines.push(format!("  {line}:{column}"));
        }
    }
    lines.join("\n")
}

/// Renders completion items, capped at 50.
#[must_use]
pub fn render_completion(items: &[CompletionItem]) -> String {
    if items.is_empty() {
        return "No completion items found.".to_owned();
    }
    items
        .iter()
        .take(MAX_COMPLETION_ITEMS)
        .map(render_completion_item)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders one completion item as `label (Kind) - detail`.
fn render_completion_item(item: &CompletionItem) -> String {
    let mut line = match item.kind {
        Some(kind) => format!("{} ({})", item.label, completion_kind_name(kind)),
        None => item.label.clone(),
    };
    if let Some(detail) = item.detail.as_deref().filter(|detail| !detail.is_empty()) {
        line.push_str(" - ");
        line.push_str(detail);
    }
    line
}

/// Renders document symbols, detecting the hierarchical tree form versus the flat form.
#[must_use]
pub fn render_symbols(symbols: &[DocumentSymbol]) -> String {
    if symbols.is_empty() {
        return "No symbols found.".to_owned();
    }
    let is_tree = symbols.iter().all(|symbol| symbol.uri.is_none());
    if is_tree {
        let mut lines = Vec::new();
        for symbol in symbols {
            render_symbol_tree(symbol, 0, &mut lines);
        }
        lines.join("\n")
    } else {
        symbols
            .iter()
            .map(render_flat_symbol)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Recursively renders one symbol and its children.
fn render_symbol_tree(symbol: &DocumentSymbol, depth: usize, lines: &mut Vec<String>) {
    let indent = "  ".repeat(depth);
    let kind = symbol_kind_name(symbol.kind);
    let mut line = format!("{indent}{kind} {}", symbol.name);
    if let Some(detail) = symbol.detail.as_deref().filter(|detail| !detail.is_empty()) {
        line.push_str(" - ");
        line.push_str(detail);
    }
    lines.push(line);
    for child in &symbol.children {
        render_symbol_tree(child, depth + 1, lines);
    }
}

/// Renders one flat symbol as `path:line: Kind container.name`.
fn render_flat_symbol(symbol: &DocumentSymbol) -> String {
    let kind = symbol_kind_name(symbol.kind);
    let name = match symbol.container_name.as_deref() {
        Some(container) if !container.is_empty() => format!("{container}.{}", symbol.name),
        _ => symbol.name.clone(),
    };
    match (&symbol.uri, symbol.range) {
        (Some(uri), Some(range)) => {
            let (line, _) = range.start.to_one_based();
            format!("{}:{line}: {kind} {name}", display_uri(uri))
        }
        _ => format!("{kind} {name}"),
    }
}

/// Renders workspace symbols grouped by file, one-based.
#[must_use]
pub fn render_workspace_symbols(symbols: &[SymbolInformation]) -> String {
    if symbols.is_empty() {
        return "No symbols found.".to_owned();
    }
    let mut lines = Vec::new();
    let mut groups: Vec<(&str, Vec<&SymbolInformation>)> = Vec::new();
    for symbol in symbols {
        let uri = symbol.location.uri.as_str();
        match groups.iter_mut().find(|(group_uri, _)| *group_uri == uri) {
            Some((_, group)) => group.push(symbol),
            None => groups.push((uri, vec![symbol])),
        }
    }
    for (uri, group) in groups {
        lines.push(format!("{}:", display_uri(uri)));
        for symbol in group {
            let (line, _) = symbol.location.range.start.to_one_based();
            let kind = symbol_kind_name(symbol.kind);
            let name = match symbol.container_name.as_deref() {
                Some(container) if !container.is_empty() => {
                    format!("{container}.{}", symbol.name)
                }
                _ => symbol.name.clone(),
            };
            lines.push(format!("  {line}: {kind} {name}"));
        }
    }
    lines.join("\n")
}

/// Formats a numbered source excerpt around a location.
///
/// `first_line` is one-based; `lines` are rendered in order with a right-aligned line
/// number and a `|` separator, prefixed by a `-- path:first_line --` header.
#[must_use]
pub fn format_context_excerpt(path: &str, first_line: u32, lines: &[&str]) -> String {
    let mut rendered = vec![format!("-- {path}:{first_line} --")];
    for (offset, line) in lines.iter().enumerate() {
        let number = first_line + u32::try_from(offset).unwrap_or(u32::MAX);
        rendered.push(format!("{number:>5}| {line}"));
    }
    rendered.join("\n")
}

/// Groups locations by URI, preserving first-seen order.
fn group_by_uri(locations: &[Location]) -> Vec<(&str, Vec<&Location>)> {
    let mut groups: Vec<(&str, Vec<&Location>)> = Vec::new();
    for location in locations {
        match groups
            .iter_mut()
            .find(|(uri, _)| *uri == location.uri.as_str())
        {
            Some((_, group)) => group.push(location),
            None => groups.push((location.uri.as_str(), vec![location])),
        }
    }
    groups
}

/// Returns the LSP `DiagnosticSeverity` name; unknown values render as `Information`.
#[must_use]
pub fn diagnostic_severity_name(severity: u32) -> &'static str {
    match severity {
        1 => "Error",
        2 => "Warning",
        4 => "Hint",
        _ => "Information",
    }
}

/// Renders diagnostics one per line, severity first and capped at 50.
///
/// Each line is `line:column: Severity: message`, followed by ` [source(code)]` when the
/// diagnostic carries a source and/or a code. The [`Diagnostic`] type has no document URI, so
/// the file path is rendered by [`render_diagnostic_files`]; the tool payload carries
/// `file_path` separately. Sorting is stable, so equal severities keep their input order.
#[must_use]
pub fn render_diagnostics(diagnostics: &[Diagnostic]) -> String {
    if diagnostics.is_empty() {
        return "No diagnostics found.".to_owned();
    }
    let mut sorted: Vec<&Diagnostic> = diagnostics.iter().collect();
    sorted.sort_by_key(|diagnostic| severity_of(diagnostic));
    sorted
        .iter()
        .take(MAX_DIAGNOSTICS)
        .map(|diagnostic| render_diagnostic(diagnostic))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders buffered diagnostics grouped by document, one header per file.
#[must_use]
pub fn render_diagnostic_files(files: &[DiagnosticsFile]) -> String {
    if files.is_empty() {
        return "No diagnostics found.".to_owned();
    }
    let mut lines = Vec::new();
    for file in files {
        let path = if file.local_path.is_empty() {
            display_uri(&file.uri)
        } else {
            file.local_path.clone()
        };
        lines.push(format!("{path}:"));
        for diagnostic in file.diagnostics.iter().take(MAX_DIAGNOSTICS) {
            lines.push(format!("  {}", render_diagnostic(diagnostic)));
        }
    }
    lines.join("\n")
}

/// Renders one diagnostic line, appending `[source(code)]` when either is present.
fn render_diagnostic(diagnostic: &Diagnostic) -> String {
    let (line, column) = diagnostic.range.start.to_one_based();
    let mut text = format!(
        "{line}:{column}: {}: {}",
        diagnostic_severity_name(severity_of(diagnostic)),
        diagnostic.message
    );
    let source = diagnostic
        .source
        .as_deref()
        .filter(|source| !source.is_empty());
    let code = diagnostic.code.as_ref().map(diagnostic_code_text);
    let suffix = match (source, code) {
        (Some(source), Some(code)) => format!(" [{source}({code})]"),
        (Some(source), None) => format!(" [{source}]"),
        (None, Some(code)) => format!(" [{code}]"),
        (None, None) => String::new(),
    };
    text.push_str(&suffix);
    text
}

/// Renders a diagnostic code as text.
fn diagnostic_code_text(code: &DiagnosticCode) -> String {
    match code {
        DiagnosticCode::String(text) => text.clone(),
        DiagnosticCode::Number(number) => number.to_string(),
    }
}

/// The severity used for rendering and ordering, defaulting an absent value to `Information`.
fn severity_of(diagnostic: &Diagnostic) -> u32 {
    diagnostic.severity.unwrap_or(DEFAULT_SEVERITY)
}

/// Renders `prepareCallHierarchy` items, matching Python's singular/plural shape.
#[must_use]
pub fn render_call_hierarchy_items(items: &[CallHierarchyItem]) -> String {
    match items {
        [] => "No call hierarchy available.".to_owned(),
        [item] => render_call_hierarchy_item(item),
        many => {
            let lines: Vec<String> = many
                .iter()
                .map(|item| format!("  {}", render_call_hierarchy_item(item)))
                .collect();
            format!("{} call hierarchy items:\n{}", many.len(), lines.join("\n"))
        }
    }
}

/// Renders one call-hierarchy item as `path:line: name` (one-based line).
fn render_call_hierarchy_item(item: &CallHierarchyItem) -> String {
    let (line, _) = item.range.start.to_one_based();
    format!("{}:{line}: {}", display_uri(&item.uri), item.name)
}

/// Renders incoming calls grouped by caller, one line per call site.
///
/// Each group starts with `path:`; every `fromRanges` entry becomes
/// `  caller_path:caller_line -> call site line:column` (one-based).
#[must_use]
pub fn render_incoming_calls(calls: &[CallHierarchyIncomingCall]) -> String {
    if calls.is_empty() {
        return "No incoming calls found.".to_owned();
    }
    let mut groups: Vec<(&str, Vec<&CallHierarchyIncomingCall>)> = Vec::new();
    for call in calls {
        let uri = call.from.uri.as_str();
        match groups.iter_mut().find(|(group_uri, _)| *group_uri == uri) {
            Some((_, group)) => group.push(call),
            None => groups.push((uri, vec![call])),
        }
    }
    let mut lines = Vec::new();
    for (uri, group) in groups {
        lines.push(format!("{}:", display_uri(uri)));
        for call in group {
            let caller_path = display_uri(&call.from.uri);
            let (caller_line, _) = call.from.range.start.to_one_based();
            for from_range in &call.from_ranges {
                let (line, column) = from_range.start.to_one_based();
                lines.push(format!(
                    "  {caller_path}:{caller_line} -> call site {line}:{column}"
                ));
            }
        }
    }
    lines.join("\n")
}

/// Renders outgoing calls, one line per call site.
///
/// Every `fromRanges` entry becomes
/// `  call site line:column -> callee_name (callee_path:callee_line)` (one-based).
#[must_use]
pub fn render_outgoing_calls(calls: &[CallHierarchyOutgoingCall]) -> String {
    if calls.is_empty() {
        return "No outgoing calls found.".to_owned();
    }
    let mut lines = Vec::new();
    for call in calls {
        let callee_path = display_uri(&call.to.uri);
        let (callee_line, _) = call.to.range.start.to_one_based();
        for from_range in &call.from_ranges {
            let (line, column) = from_range.start.to_one_based();
            lines.push(format!(
                "  call site {line}:{column} -> {} ({callee_path}:{callee_line})",
                call.to.name
            ));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::protocol::{Position, Range};

    /// Builds a range starting at zero-based `(line, character)`.
    fn range_at(line: u32, character: u32) -> Range {
        Range {
            start: Position::new(line, character),
            end: Position::new(line, character + 1),
        }
    }

    /// Builds a diagnostic with an optional severity and the given message.
    fn diagnostic(message: &str, severity: Option<u32>) -> Diagnostic {
        Diagnostic {
            message: message.to_owned(),
            severity,
            range: range_at(1, 0),
            source: None,
            code: None,
            tags: Vec::new(),
            related_information: Vec::new(),
        }
    }

    /// Builds one call-hierarchy item.
    fn item(name: &str, uri: &str, line: u32) -> CallHierarchyItem {
        CallHierarchyItem {
            name: name.to_owned(),
            kind: 12,
            detail: None,
            uri: uri.to_owned(),
            range: range_at(line, 0),
            selection_range: range_at(line, 3),
            tags: Vec::new(),
            data: None,
        }
    }

    #[test]
    fn severity_names_cover_the_protocol_and_default_to_information() {
        assert_eq!(diagnostic_severity_name(1), "Error");
        assert_eq!(diagnostic_severity_name(2), "Warning");
        assert_eq!(diagnostic_severity_name(3), "Information");
        assert_eq!(diagnostic_severity_name(4), "Hint");
        assert_eq!(diagnostic_severity_name(0), "Information");
        assert_eq!(diagnostic_severity_name(99), "Information");
    }

    #[test]
    fn diagnostics_render_one_line_each_sorted_by_severity() {
        assert_eq!(render_diagnostics(&[]), "No diagnostics found.");
        let diagnostics = vec![
            diagnostic("last", Some(4)),
            diagnostic("first", Some(1)),
            diagnostic("middle", None),
        ];
        assert_eq!(
            render_diagnostics(&diagnostics),
            "2:1: Error: first\n2:1: Information: middle\n2:1: Hint: last"
        );
    }

    #[test]
    fn diagnostics_append_source_and_code_when_present() {
        let mut with_source = diagnostic("boom", Some(1));
        with_source.source = Some("rustc".to_owned());
        with_source.code = Some(DiagnosticCode::String("E1".to_owned()));
        assert_eq!(
            render_diagnostics(&[with_source]),
            "2:1: Error: boom [rustc(E1)]"
        );

        let mut code_only = diagnostic("boom", Some(1));
        code_only.code = Some(DiagnosticCode::Number(2304));
        assert_eq!(render_diagnostics(&[code_only]), "2:1: Error: boom [2304]");

        let mut source_only = diagnostic("boom", Some(2));
        source_only.source = Some("fake".to_owned());
        assert_eq!(
            render_diagnostics(&[source_only]),
            "2:1: Warning: boom [fake]"
        );
    }

    #[test]
    fn diagnostics_are_capped_at_fifty_entries() {
        let diagnostics: Vec<Diagnostic> = (0..60)
            .map(|index| diagnostic(&format!("d{index}"), Some(1)))
            .collect();
        assert_eq!(render_diagnostics(&diagnostics).lines().count(), 50);
    }

    #[test]
    fn diagnostic_files_group_under_one_header_each() {
        assert_eq!(render_diagnostic_files(&[]), "No diagnostics found.");
        let files = vec![
            DiagnosticsFile {
                uri: "file:///workspace/a.rs".to_owned(),
                version: Some(3),
                diagnostics: vec![diagnostic("first", Some(1))],
                local_path: "/workspace/a.rs".to_owned(),
            },
            DiagnosticsFile {
                uri: "file:///workspace/b.rs".to_owned(),
                version: None,
                diagnostics: vec![diagnostic("second", Some(2))],
                local_path: String::new(),
            },
        ];
        assert_eq!(
            render_diagnostic_files(&files),
            "/workspace/a.rs:\n  2:1: Error: first\n/workspace/b.rs:\n  2:1: Warning: second"
        );
    }

    #[test]
    fn call_hierarchy_items_render_single_and_plural_shapes() {
        assert_eq!(
            render_call_hierarchy_items(&[]),
            "No call hierarchy available."
        );
        let one = vec![item("caller", "file:///workspace/a.rs", 1)];
        assert_eq!(
            render_call_hierarchy_items(&one),
            "/workspace/a.rs:2: caller"
        );
        let two = vec![
            item("caller", "file:///workspace/a.rs", 1),
            item("callee", "file:///workspace/b.rs", 7),
        ];
        assert_eq!(
            render_call_hierarchy_items(&two),
            "2 call hierarchy items:\n  /workspace/a.rs:2: caller\n  /workspace/b.rs:8: callee"
        );
    }

    #[test]
    fn incoming_calls_group_by_caller_and_show_call_sites() {
        assert_eq!(render_incoming_calls(&[]), "No incoming calls found.");
        let calls = vec![
            CallHierarchyIncomingCall {
                from: item("caller", "file:///workspace/a.rs", 1),
                from_ranges: vec![range_at(2, 4), range_at(4, 0)],
            },
            CallHierarchyIncomingCall {
                from: item("other", "file:///workspace/b.rs", 3),
                from_ranges: vec![range_at(5, 1)],
            },
        ];
        assert_eq!(
            render_incoming_calls(&calls),
            "/workspace/a.rs:\n  /workspace/a.rs:2 -> call site 3:5\n  \
             /workspace/a.rs:2 -> call site 5:1\n/workspace/b.rs:\n  \
             /workspace/b.rs:4 -> call site 6:2"
        );
    }

    #[test]
    fn outgoing_calls_render_every_call_site() {
        assert_eq!(render_outgoing_calls(&[]), "No outgoing calls found.");
        let calls = vec![CallHierarchyOutgoingCall {
            to: item("callee", "file:///workspace/c.rs", 5),
            from_ranges: vec![range_at(3, 2)],
        }];
        assert_eq!(
            render_outgoing_calls(&calls),
            "  call site 4:3 -> callee (/workspace/c.rs:6)"
        );
    }
}
