//! Human-readable rendering of normalised LSP results.
//!
//! Ports `openjiuwen/harness/tools/lsp_tool/_formatter.py`. Every renderer is pure and
//! deterministic, so the tool payload text is stable across servers that return the same
//! locations. Positions are shown one-based (`line:column`), matching editors and the
//! model-facing convention.
use super::protocol::{CompletionItem, DocumentSymbol, Hover, Location, SymbolInformation};
use super::uri::uri_to_path;

/// Maximum completion items included in one rendered result.
const MAX_COMPLETION_ITEMS: usize = 50;

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
