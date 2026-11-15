//! Exposure, code assignment and rendering of one catalogue subtree.
//!
//! Ports the exposure half of Python
//! `openjiuwen.symphony.retrieval.search.runtime.render.disclosure` (v0.1.20):
//! [`build_exposed_fragment`] walks a [`RetrieverNode`]
//! depth-first, decides which nodes are *selectable* and which are only *categories*,
//! assigns one short code per selectable candidate, renders the prompt-facing candidate
//! block, and hashes the exposure into the fragment fingerprint.
//!
//! # Exposure order
//!
//! At each node its own [`RetrieverNode::items`] become selectable leaves first, then the
//! children in [`RetrieverNode::children`] order (both accessors are sorted, so the walk is
//! deterministic). That order is the order in which codes are assigned in tree mode.
//!
//! A non-root node with no items of its own and exactly one child is *spliced*: it never
//! becomes selectable, so a unary category chain costs no prompt line. A child becomes one
//! selectable branch (`is_terminal == false`) when the remaining exposure depth is
//! exhausted or when its subtree carries more than
//! [`DisclosureConfig::exposure_threshold`] items; otherwise it is expanded.
//!
//! # Codes
//!
//! - names mode ([`DisclosureConfig::compact_codes_enabled`] `== false`): a `PascalCase`
//!   boundary name derived from the terminal segment of the selectable canonical id,
//!   duplicates disambiguated by joining the branch path with `/` and finally by
//!   `name__{index}` (`index` is 1-based);
//! - compact mode with a code book: [`CompactCodebook::prefix`] positionally, so a book
//!   shorter than the exposure is an `Error::Validation`;
//! - compact mode without a code book: the numeric codes `"1"`, `"2"`, ….
//!
//! The reserved abstain token
//! [`ABSTAIN_TOKEN`](crate::retrieval::search::flow::ABSTAIN_TOKEN) is never assigned.
//!
//! # Flat mode
//!
//! [`DisclosureConfig::flatten_full_tree_in_prompt`] sorts the selectable entries
//! alphabetically by their selectable identity (ties by label) and emits them as one flat
//! block, so codes follow that order instead of the tree order. Compact flat mode renders a
//! pretty-printed JSON array whose objects carry the fixed field order of
//! [`FLAT_COMPACT_FIELD_ORDER`]; names flat mode renders an `- <name>: <description>` list.
//!
//! # Rust-only bounds
//!
//! Four [`DisclosureConfig`] knobs have no Python counterpart and are applied while the
//! candidate block is rendered:
//!
//! - [`DisclosureConfig::max_description_chars`] clamps a candidate description at a
//!   `。 . ； ;` boundary when one lies in the second half of the budget, else hard-cuts it
//!   with `...` (Python only clamps the flat compact JSON, always at 150 characters);
//! - [`DisclosureConfig::max_items_per_branch`] discloses at most that many of a node's
//!   items per prompt line; the withheld items stay in the exposed view, in
//!   [`ExposedFragment::code_to_resolution`] and in
//!   [`ExposedFragment::candidate_codes`], and a `tracing` record names the count, so
//!   nothing is lost silently;
//! - [`DisclosureConfig::max_chars`] truncates the rendered text with
//!   [`truncate_chars_with_ellipsis`] and reports the fragment as truncated;
//! - [`DisclosureConfig::code_width`] overrides the derived code width.
//!
//! Every truncation and every slice in this module is character-boundary safe, so a CJK
//! label is never split into a partial character.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::error::Result;
use crate::flow::model::sha256_short;
use crate::retrieval::build::schema::{RootCategory, FIXED_ROOT_CATEGORIES};
use crate::retrieval::search::codebooks::CompactCodebook;
use crate::retrieval::search::config::DisclosureConfig;
use crate::retrieval::search::flow::{ExposedFragment, ExposedNode, SelectableResolution};
use crate::retrieval::search::truncation::truncate_chars_with_ellipsis;
use crate::retrieval::types::{RetrieverItem, RetrieverNode};

/// The canonical-id prefix of an exposed branch (`node::<node_id>`).
pub const BRANCH_CANONICAL_PREFIX: &str = "node::";

/// The canonical-id prefix of an exposed item (`item::<payload>`).
pub const ITEM_CANONICAL_PREFIX: &str = "item::";

/// One indentation level of a rendered disclosure tree.
pub const TREE_INDENT: &str = "  ";

/// The fixed field order of a flat compact candidate object.
pub const FLAT_COMPACT_FIELD_ORDER: [&str; 5] =
    ["category", "name", "raw_name", "description", "id"];

/// The trailing remark a candidate description may carry and that is stripped.
pub const REPRESENTATIVE_DESCENDANTS: &str = "Representative descendants:";

/// The characters a clamped description prefers to end on.
const DESCRIPTION_BOUNDARIES: [char; 4] = ['。', '.', '；', ';'];

/// The divisor of the budget a boundary cut must reach to be accepted.
const DESCRIPTION_BOUNDARY_DIVISOR: usize = 2;

/// The absolute floor a boundary cut must reach to be accepted.
const DESCRIPTION_BOUNDARY_FLOOR: usize = 40;

/// The suffix of a hard-cut description.
const DESCRIPTION_ELLIPSIS: &str = "...";

/// One selectable candidate found while walking the tree.
///
/// Mirrors the private `_SelectableEntry` tuple of Python's disclosure module: the two
/// canonical ids differ because an *item's* exposure identity is `item::<payload>` while
/// its resolution points at the payload (or the item id) alone.
#[derive(Debug, Clone)]
struct SelectableEntry {
    /// The exposure identity (`node::<node_id>` / `item::<payload>`).
    canonical_id: String,
    /// The rendered label.
    label: String,
    /// The node or item description, before sanitizing.
    description: String,
    /// The identity a resolution carries: the node id, or the item payload.
    selectable_canonical_id: String,
    /// Whether the entry yields items instead of deeper branches.
    is_terminal: bool,
    /// The id chain that reached the entry, root id first.
    branch_path: Vec<String>,
    /// The branch node the entry stands for.
    node: Option<RetrieverNode>,
    /// The item the entry stands for.
    item: Option<RetrieverItem>,
}

impl SelectableEntry {
    /// Builds the resolution this entry exposes under `code`.
    fn resolution(&self, code: String, display_name: String) -> SelectableResolution {
        SelectableResolution {
            code,
            canonical_id: self.selectable_canonical_id.clone(),
            display_name,
            label: self.label.clone(),
            description: self.description.clone(),
            is_terminal: self.is_terminal,
            branch_path: self.branch_path.clone(),
            node_id: self.node.as_ref().map(|node| node.node_id.clone()),
            item: self.item.clone(),
        }
    }
}

/// The default subtree weight used by [`DisclosureConfig::exposure_threshold`].
///
/// Returns [`RetrieverNode::item_count`], so every descendant item counts once. A caller
/// that weights items differently (by score, say) passes its own counter to
/// [`build_exposed_fragment`].
#[must_use]
pub fn exposure_item_count(node: &RetrieverNode) -> usize {
    node.item_count()
}

/// The exposure fingerprint of `root` under `candidate_codes`.
///
/// The first 16 hex characters of the SHA-256 over the exposed-tree serialization plus the
/// joined codes; identical exposures produce identical fingerprints, and a changed label or
/// code changes it. The serialization mirrors Python's `_serialize_exposed_node` —
/// `canonical_id|label|is_selectable|children`, the children joined with `,` — followed by
/// `|<code>/<code>/…`.
#[must_use]
pub fn fragmentation_fingerprint(root: &ExposedNode, candidate_codes: &[String]) -> String {
    let payload = format!(
        "{}|{}",
        serialize_exposed_node(root),
        candidate_codes.join("/")
    );
    sha256_short(&payload, 16)
}

/// Builds the exposed fragment of `root`.
///
/// `branch_path` is the id chain that reached `root` (root id first, as
/// [`SearchCursor::branch_path`](crate::retrieval::search::flow::SearchCursor) carries it),
/// `config` supplies the exposure knobs, and `subtree_item_count` is the weight the
/// exposure threshold compares against (see [`exposure_item_count`]).
///
/// # Errors
///
/// [`Error::Validation`](crate::error::Error) naming `compact_codebook` when compact codes
/// are enabled and the configured code book is invalid or shorter than the number of
/// selectable candidates.
pub fn build_exposed_fragment(
    root: &RetrieverNode,
    branch_path: &[String],
    config: &DisclosureConfig,
    subtree_item_count: &dyn Fn(&RetrieverNode) -> usize,
) -> Result<ExposedFragment> {
    let mut entries: Vec<SelectableEntry> = Vec::new();
    let exposed_root = expand_node(
        root,
        branch_path,
        config.max_exposure_depth_per_call,
        true,
        config,
        subtree_item_count,
        &mut entries,
    );
    if config.flatten_full_tree_in_prompt {
        build_flat_fragment(exposed_root, &entries, config)
    } else {
        build_tree_fragment(exposed_root, &entries, config)
    }
}

/// Builds the nested-tree fragment: codes follow the tree order.
fn build_tree_fragment(
    root: ExposedNode,
    entries: &[SelectableEntry],
    config: &DisclosureConfig,
) -> Result<ExposedFragment> {
    let codes = build_codes(entries, config)?;
    let display_names = build_boundary_names(entries);
    let (code_to_resolution, canonical_to_code) = build_resolutions(entries, &codes, display_names);

    let rendered_tree = render_tree(&root, &canonical_to_code, &code_to_resolution, config, 0);
    let (rendered_tree, rendered_chars, truncated) = apply_char_budget(rendered_tree, config);
    let fragment_fingerprint = fragmentation_fingerprint(&root, &codes);
    let code_width = resolve_code_width(&codes, config.code_width);
    Ok(ExposedFragment {
        root,
        rendered_tree,
        code_to_resolution,
        candidate_codes: codes,
        fragment_fingerprint,
        code_width,
        compact_codes_enabled: config.compact_codes_enabled,
        flat_list_mode: false,
        rendered_chars,
        truncated,
    })
}

/// Builds the flat-list fragment: candidates are sorted alphabetically before coding.
fn build_flat_fragment(
    root: ExposedNode,
    entries: &[SelectableEntry],
    config: &DisclosureConfig,
) -> Result<ExposedFragment> {
    let mut sorted: Vec<&SelectableEntry> = entries.iter().collect();
    sorted.sort_by_key(|entry| flat_sort_key(entry));
    let codes = build_codes_borrowed(&sorted, config)?;
    let display_names = build_boundary_names_borrowed(&sorted);
    let (code_to_resolution, canonical_to_code) =
        build_resolutions_borrowed(&sorted, &codes, display_names);

    let mut selectable_nodes: Vec<ExposedNode> = Vec::new();
    for entry in &sorted {
        selectable_nodes.push(ExposedNode {
            canonical_id: entry.canonical_id.clone(),
            label: entry.label.clone(),
            description: entry.description.clone(),
            is_selectable: true,
            selectable_canonical_id: Some(entry.selectable_canonical_id.clone()),
            children: Vec::new(),
        });
    }
    let flat_root = ExposedNode {
        canonical_id: root.canonical_id,
        label: root.label,
        description: root.description,
        is_selectable: false,
        selectable_canonical_id: None,
        children: selectable_nodes.clone(),
    };
    let rendered_tree = render_flat_list(
        &selectable_nodes,
        &canonical_to_code,
        &code_to_resolution,
        config,
    );
    let (rendered_tree, rendered_chars, truncated) = apply_char_budget(rendered_tree, config);
    let fragment_fingerprint = fragmentation_fingerprint(&flat_root, &codes);
    let code_width = resolve_code_width(&codes, config.code_width);
    Ok(ExposedFragment {
        root: flat_root,
        rendered_tree,
        code_to_resolution,
        candidate_codes: codes,
        fragment_fingerprint,
        code_width,
        compact_codes_enabled: config.compact_codes_enabled,
        flat_list_mode: true,
        rendered_chars,
        truncated,
    })
}

/// Builds the code → resolution map plus the canonical-id → code index.
fn build_resolutions(
    entries: &[SelectableEntry],
    codes: &[String],
    display_names: Vec<String>,
) -> (
    BTreeMap<String, SelectableResolution>,
    BTreeMap<String, String>,
) {
    let borrowed: Vec<&SelectableEntry> = entries.iter().collect();
    build_resolutions_borrowed(&borrowed, codes, display_names)
}

/// [`build_resolutions`] over borrowed entries.
///
/// A duplicated canonical id keeps the *last* code, as Python's dict assignment does.
fn build_resolutions_borrowed(
    entries: &[&SelectableEntry],
    codes: &[String],
    display_names: Vec<String>,
) -> (
    BTreeMap<String, SelectableResolution>,
    BTreeMap<String, String>,
) {
    let mut by_code: BTreeMap<String, SelectableResolution> = BTreeMap::new();
    let mut canonical_to_code: BTreeMap<String, String> = BTreeMap::new();
    for ((entry, code), display_name) in entries.iter().zip(codes.iter()).zip(display_names) {
        canonical_to_code.insert(entry.canonical_id.clone(), code.clone());
        by_code.insert(code.clone(), entry.resolution(code.clone(), display_name));
    }
    (by_code, canonical_to_code)
}

/// Assigns one code per entry.
fn build_codes(entries: &[SelectableEntry], config: &DisclosureConfig) -> Result<Vec<String>> {
    let borrowed: Vec<&SelectableEntry> = entries.iter().collect();
    build_codes_borrowed(&borrowed, config)
}

/// [`build_codes`] over borrowed entries.
fn build_codes_borrowed(
    entries: &[&SelectableEntry],
    config: &DisclosureConfig,
) -> Result<Vec<String>> {
    let count = entries.len();
    if count == 0 {
        return Ok(Vec::new());
    }
    if !config.compact_codes_enabled {
        return Ok(build_boundary_names_borrowed(entries));
    }
    let book = CompactCodebook::new(config.compact_codebook.iter().cloned())?;
    if !book.is_empty() {
        return book.prefix(count);
    }
    Ok((1..=count).map(|value| value.to_string()).collect())
}

/// Derives one display name per entry (names mode).
fn build_boundary_names(entries: &[SelectableEntry]) -> Vec<String> {
    let borrowed: Vec<&SelectableEntry> = entries.iter().collect();
    build_boundary_names_borrowed(&borrowed)
}

/// [`build_boundary_names`] over borrowed entries.
fn build_boundary_names_borrowed(entries: &[&SelectableEntry]) -> Vec<String> {
    let bases: Vec<String> = entries
        .iter()
        .map(|entry| boundary_name_base(entry))
        .collect();
    let paths: Vec<&[String]> = entries
        .iter()
        .map(|entry| entry.branch_path.as_slice())
        .collect();
    disambiguate_names(&bases, &paths)
}

/// The undeduplicated boundary name of one entry.
///
/// Mirrors Python: the `PascalCase` terminal segment of the selectable canonical id, then
/// the `PascalCase` of the whole id with `.` replaced by `-`, then the raw id, then the
/// label — the first non-empty candidate wins.
fn boundary_name_base(entry: &SelectableEntry) -> String {
    let selectable = entry.selectable_canonical_id.trim();
    let terminal = selectable.rsplit('.').next().unwrap_or_default();
    let candidates = [
        pascal_case(terminal),
        pascal_case(&selectable.replace('.', "-")),
        selectable.to_owned(),
        entry.label.trim().to_owned(),
    ];
    candidates
        .into_iter()
        .find(|candidate| !candidate.is_empty())
        .unwrap_or_default()
}

/// Deduplicates boundary names against the branch paths that reached them.
///
/// A duplicated name joins its branch-path segments (root segment dropped, each segment
/// `PascalCase`d) with `/`; when the path holds nothing else, the name gets an `__{n}`
/// suffix instead, where `n` is the 1-based entry index. Both rules mirror Python's
/// `_build_boundary_names`.
fn disambiguate_names(bases: &[String], paths: &[&[String]]) -> Vec<String> {
    let mut identifiers = bases.to_vec();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for base in bases {
        *counts.entry(base.as_str()).or_insert(0) += 1;
    }
    for (index, base) in bases.iter().enumerate() {
        if counts.get(base.as_str()).copied().unwrap_or(0) <= 1 {
            continue;
        }
        let path = paths.get(index).copied().unwrap_or_default();
        let mut parts: Vec<String> = Vec::new();
        for part in path.iter().skip(1) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let folded = pascal_case(part);
            parts.push(if folded.is_empty() {
                part.to_owned()
            } else {
                folded
            });
        }
        if parts.is_empty() {
            identifiers[index] = format!("{base}__{}", index + 1);
        } else {
            parts.push(base.clone());
            identifiers[index] = parts.join("/");
        }
    }
    identifiers
}

/// Renders the nested candidate block.
fn render_tree(
    node: &ExposedNode,
    canonical_to_code: &BTreeMap<String, String>,
    resolutions: &BTreeMap<String, SelectableResolution>,
    config: &DisclosureConfig,
    depth: usize,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    let indent = TREE_INDENT.repeat(depth);
    let identifier = if node.is_selectable {
        selectable_identifier(node, canonical_to_code, resolutions, config)
    } else {
        format!("Category {}", node.label)
    };
    let description = if node.is_selectable {
        candidate_description(&node.description, config, false)
    } else {
        node.description.clone()
    };
    if description.is_empty() {
        lines.push(format!("{indent}- {identifier}"));
    } else {
        lines.push(format!("{indent}- {identifier}: {description}"));
    }

    let cap = config.max_items_per_branch;
    let mut disclosed_items = 0_usize;
    let mut withheld = 0_usize;
    for child in &node.children {
        if child.canonical_id.starts_with(ITEM_CANONICAL_PREFIX) {
            if cap > 0 && disclosed_items >= cap {
                withheld += 1;
                continue;
            }
            disclosed_items += 1;
        }
        lines.push(render_tree(
            child,
            canonical_to_code,
            resolutions,
            config,
            depth + 1,
        ));
    }
    if withheld > 0 {
        tracing::debug!(
            node = %node.canonical_id,
            withheld,
            "items withheld from the rendered disclosure by `max_items_per_branch`",
        );
    }
    lines.join("\n")
}

/// The `- Candidate …` / `- … [id: …]` identifier of a selectable node.
fn selectable_identifier(
    node: &ExposedNode,
    canonical_to_code: &BTreeMap<String, String>,
    resolutions: &BTreeMap<String, SelectableResolution>,
    config: &DisclosureConfig,
) -> String {
    let resolution = lookup_resolution(node, canonical_to_code, resolutions);
    let display_name = resolution.map_or(node.label.as_str(), |entry| entry.display_name.as_str());
    if !config.compact_codes_enabled {
        return format!("Candidate {display_name}");
    }
    let code = resolution.map_or("", |entry| entry.code.as_str());
    format!("{display_name} [id: {code}]")
}

/// Renders the flat candidate block.
fn render_flat_list(
    nodes: &[ExposedNode],
    canonical_to_code: &BTreeMap<String, String>,
    resolutions: &BTreeMap<String, SelectableResolution>,
    config: &DisclosureConfig,
) -> String {
    if config.compact_codes_enabled {
        return render_flat_compact_json(nodes, canonical_to_code, resolutions, config);
    }
    let mut lines: Vec<String> = Vec::new();
    for node in nodes {
        let Some(resolution) = lookup_resolution(node, canonical_to_code, resolutions) else {
            continue;
        };
        let description = candidate_description(&node.description, config, false);
        if description.is_empty() {
            lines.push(format!("- {}", resolution.display_name));
        } else {
            lines.push(format!("- {}: {description}", resolution.display_name));
        }
    }
    lines.join("\n")
}

/// Renders the flat compact candidate block as an indented JSON array.
fn render_flat_compact_json(
    nodes: &[ExposedNode],
    canonical_to_code: &BTreeMap<String, String>,
    resolutions: &BTreeMap<String, SelectableResolution>,
    config: &DisclosureConfig,
) -> String {
    let mut candidates: Vec<Value> = Vec::new();
    for node in nodes {
        let Some(resolution) = lookup_resolution(node, canonical_to_code, resolutions) else {
            continue;
        };
        let mut object = Map::new();
        object.insert(
            "category".to_owned(),
            Value::String(format_branch_category(&resolution.branch_path)),
        );
        object.insert(
            "name".to_owned(),
            Value::String(flat_compact_name(resolution)),
        );
        object.insert(
            "raw_name".to_owned(),
            Value::String(flat_compact_raw_name(resolution)),
        );
        object.insert(
            "description".to_owned(),
            Value::String(candidate_description(&node.description, config, true)),
        );
        object.insert("id".to_owned(), Value::String(resolution.code.clone()));
        candidates.push(Value::Object(object));
    }
    match serde_json::to_string_pretty(&Value::Array(candidates)) {
        Ok(rendered) => rendered,
        Err(error) => {
            // A `Value` of strings always serializes; this arm exists so the renderer never
            // panics if that ever stops being true.
            tracing::error!(%error, "flat compact candidate block could not be serialized");
            String::new()
        }
    }
}

/// Looks a selectable node's resolution up, warning when the exposure is inconsistent.
fn lookup_resolution<'a>(
    node: &ExposedNode,
    canonical_to_code: &BTreeMap<String, String>,
    resolutions: &'a BTreeMap<String, SelectableResolution>,
) -> Option<&'a SelectableResolution> {
    let resolution = canonical_to_code
        .get(&node.canonical_id)
        .and_then(|code| resolutions.get(code));
    if resolution.is_none() {
        tracing::warn!(
            node = %node.canonical_id,
            "selectable node has no resolution; the renderer falls back to its label",
        );
    }
    resolution
}

/// The flat compact `name` field: the item label, or the display name.
fn flat_compact_name(resolution: &SelectableResolution) -> String {
    if !resolution.is_terminal {
        return resolution.display_name.trim().to_owned();
    }
    let label = collapse_whitespace(resolution.label.trim());
    if label.is_empty() {
        resolution.display_name.trim().to_owned()
    } else {
        label
    }
}

/// The flat compact `raw_name` field.
///
/// Python reads an optional `worker_id` off the selected item and falls back to the display
/// name; [`RetrieverItem`] carries no `worker_id`, so the fallback is the only reachable
/// branch and this port keeps it unconditionally.
fn flat_compact_raw_name(resolution: &SelectableResolution) -> String {
    resolution.display_name.trim().to_owned()
}

/// The flat compact `category` field: the last three branch labels joined by ` > `.
fn format_branch_category(branch_path: &[String]) -> String {
    let labels_map = fixed_category_labels();
    let mut labels: Vec<String> = Vec::new();
    for raw in branch_path {
        let part = raw.trim();
        if part.is_empty() {
            continue;
        }
        let label = part.rsplit('.').next().unwrap_or_default();
        if label.eq_ignore_ascii_case("ROOT") {
            continue;
        }
        let localized = labels_map
            .get(label)
            .map_or(label, |name| name.as_str())
            .to_owned();
        if localized.is_empty() {
            continue;
        }
        if labels.last() != Some(&localized) {
            labels.push(localized);
        }
    }
    if labels.is_empty() {
        return String::new();
    }
    let start = labels.len().saturating_sub(3);
    labels[start..].join(" > ")
}

/// The localized category labels of the fixed root taxonomy.
///
/// Python imports `build.tree.schema.FIXED_ROOT_CATEGORIES` and maps both a category id and
/// its `PascalCase` form to the display name; this port walks the crate's own
/// [`FIXED_ROOT_CATEGORIES`] taxonomy the same way. The map is rebuilt per call — it holds a
/// few dozen entries and the renderers are not hot paths — so no global cache is involved.
fn fixed_category_labels() -> BTreeMap<String, String> {
    let mut labels: BTreeMap<String, String> = BTreeMap::new();
    for category in FIXED_ROOT_CATEGORIES {
        visit_category(category, &mut labels);
    }
    labels
}

/// Records one taxonomy node and its children.
fn visit_category(category: &'static RootCategory, labels: &mut BTreeMap<String, String>) {
    if !category.name.is_empty() {
        labels.insert(category.id.to_owned(), category.name.to_owned());
        let pascal = pascal_case(category.id);
        if !pascal.is_empty() {
            labels.insert(pascal, category.name.to_owned());
        }
    }
    for child in category.children {
        visit_category(child, labels);
    }
}

/// Sanitizes and clamps one candidate description.
fn candidate_description(
    description: &str,
    config: &DisclosureConfig,
    collapse_newlines: bool,
) -> String {
    let sanitized = strip_representative_descendants(description.trim());
    let text = if collapse_newlines {
        collapse_whitespace(&sanitized)
    } else {
        sanitized
    };
    clamp_description(&text, config.max_description_chars)
}

/// Clamps `text` to `max_chars` characters (`0` is unlimited).
///
/// Mirrors Python's `_maybe_shorten_flat_compact_description`: a `。 . ； ;` boundary in the
/// second half of the budget (and never before
/// [`DESCRIPTION_BOUNDARY_FLOOR`] characters) ends the text, otherwise it is hard-cut and
/// marked with `...`.
fn clamp_description(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return text.to_owned();
    }
    let cleaned = collapse_whitespace(text);
    let characters: Vec<char> = cleaned.chars().collect();
    if characters.len() <= max_chars {
        return cleaned;
    }
    let mut boundary: Option<usize> = None;
    for (index, character) in characters.iter().take(max_chars).enumerate() {
        if DESCRIPTION_BOUNDARIES.contains(character) {
            boundary = Some(index);
        }
    }
    if let Some(index) = boundary {
        let floor = DESCRIPTION_BOUNDARY_FLOOR.max(max_chars / DESCRIPTION_BOUNDARY_DIVISOR);
        if index >= floor {
            let head: String = characters[..=index].iter().collect();
            return head.trim().to_owned();
        }
    }
    let head: String = characters[..max_chars].iter().collect();
    let trimmed = head.trim_end_matches(|character: char| {
        matches!(character, ' ' | ',' | '，' | ';' | '；' | '。' | '.')
    });
    format!("{trimmed}{DESCRIPTION_ELLIPSIS}")
}

/// Removes a trailing `Representative descendants: …` remark.
///
/// Python's `_REPRESENTATIVE_DESCENDANTS_RE` is
/// `(?:\n\s*|\s+)Representative descendants:\s*.+$` with `IGNORECASE | DOTALL`, so the
/// remark is only removed when whitespace precedes it and at least one character follows
/// the colon. The scan below reproduces that: the first qualifying occurrence cuts the text
/// at the start of the whitespace run before it.
fn strip_representative_descendants(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let needle: Vec<char> = REPRESENTATIVE_DESCENDANTS.chars().collect();
    if needle.is_empty() || characters.len() <= needle.len() {
        return text.to_owned();
    }
    let last_start = characters.len() - needle.len();
    for start in 1..=last_start {
        let matches = characters[start..start + needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(left, right)| left.eq_ignore_ascii_case(right));
        if !matches || !characters[start - 1].is_whitespace() {
            continue;
        }
        let mut after = start + needle.len();
        while after < characters.len() && characters[after].is_whitespace() {
            after += 1;
        }
        if after >= characters.len() {
            // Nothing follows the colon, so the pattern cannot match here.
            continue;
        }
        let mut cut = start;
        while cut > 0 && characters[cut - 1].is_whitespace() {
            cut -= 1;
        }
        let head: String = characters[..cut].iter().collect();
        return head.trim_end().to_owned();
    }
    text.to_owned()
}

/// Collapses every whitespace run into one space and trims both ends.
fn collapse_whitespace(text: &str) -> String {
    let mut rendered = String::with_capacity(text.len());
    let mut pending_space = false;
    for character in text.chars() {
        if character.is_whitespace() {
            pending_space = !rendered.is_empty();
            continue;
        }
        if pending_space {
            rendered.push(' ');
            pending_space = false;
        }
        rendered.push(character);
    }
    rendered
}

/// Applies the `max_chars` budget and reports the rendered character count.
fn apply_char_budget(rendered: String, config: &DisclosureConfig) -> (String, usize, bool) {
    if config.max_chars == 0 {
        let rendered_chars = rendered.chars().count();
        return (rendered, rendered_chars, false);
    }
    let (text, truncated) = truncate_chars_with_ellipsis(&rendered, config.max_chars);
    let rendered_chars = text.chars().count();
    (text, rendered_chars, truncated)
}

/// Derives the code width: the common length, `0` when mixed, `1` when there is no code.
fn resolve_code_width(codes: &[String], override_width: usize) -> usize {
    if override_width > 0 {
        return override_width;
    }
    if codes.is_empty() {
        return 1;
    }
    let widths: BTreeSet<usize> = codes.iter().map(|code| code.chars().count()).collect();
    if widths.len() == 1 {
        return widths.iter().copied().next().unwrap_or(0);
    }
    0
}

/// The sort key of one flat-mode entry: the selectable identity, then the label.
fn flat_sort_key(entry: &SelectableEntry) -> (String, String) {
    let identity = if entry.selectable_canonical_id.is_empty() {
        &entry.canonical_id
    } else {
        &entry.selectable_canonical_id
    };
    let identity = if identity.is_empty() {
        &entry.label
    } else {
        identity
    };
    (identity.to_lowercase(), entry.label.to_lowercase())
}

/// The `canonical_id|label|is_selectable|children` serialization of an exposed node.
fn serialize_exposed_node(node: &ExposedNode) -> String {
    let children = node
        .children
        .iter()
        .map(serialize_exposed_node)
        .collect::<Vec<String>>()
        .join(",");
    format!(
        "{}|{}|{}|{}",
        node.canonical_id,
        node.label,
        u8::from(node.is_selectable),
        children
    )
}

/// `PascalCase`s a name the way Python's `shared.naming.to_pascal_case` does.
fn pascal_case(value: &str) -> String {
    let mut rendered = String::new();
    for part in split_name_parts(value) {
        let mut characters = part.chars();
        if let Some(first) = characters.next() {
            rendered.extend(first.to_uppercase());
            rendered.push_str(characters.as_str());
        }
    }
    rendered
}

/// Splits a name into its lowercase parts.
///
/// Mirrors `_split_name_parts`: a `-` is inserted at a camel boundary
/// (`(?<=[a-z0-9])(?=[A-Z])`), `_` and every non `[A-Za-z0-9]` character become `-`, the
/// runs collapse, and the remaining parts are lowercased. A name made only of non-ASCII
/// characters therefore has no part at all, exactly as in the reference.
fn split_name_parts(value: &str) -> Vec<String> {
    let mut staged = String::with_capacity(value.len());
    let mut previous: Option<char> = None;
    for character in value.trim().chars() {
        if previous.is_some_and(|last| {
            (last.is_ascii_lowercase() || last.is_ascii_digit()) && character.is_ascii_uppercase()
        }) {
            staged.push('-');
        }
        if character == '_' {
            staged.push('-');
        } else if character.is_ascii_alphanumeric() {
            staged.push(character);
        } else {
            staged.push('-');
        }
        previous = Some(character);
    }
    staged
        .split('-')
        .filter(|part| !part.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Walks `node`, pushing one selectable entry per item and per collapsed branch.
fn expand_node(
    node: &RetrieverNode,
    current_path: &[String],
    remaining_depth: usize,
    is_root: bool,
    config: &DisclosureConfig,
    subtree_item_count: &dyn Fn(&RetrieverNode) -> usize,
    entries: &mut Vec<SelectableEntry>,
) -> ExposedNode {
    let mut children_nodes: Vec<ExposedNode> = Vec::new();
    for item in node.items() {
        children_nodes.push(push_item_entry(item, current_path, entries));
    }

    let child_nodes = node.children();
    if should_auto_expand_single_child(is_root, remaining_depth, node, &child_nodes) {
        let only_child = child_nodes[0];
        let mut path = current_path.to_vec();
        path.push(only_child.node_id.clone());
        children_nodes.push(expand_node(
            only_child,
            &path,
            remaining_depth - 1,
            false,
            config,
            subtree_item_count,
            entries,
        ));
        return exposed_branch(node, children_nodes);
    }

    let threshold = config.exposure_threshold;
    for child in child_nodes {
        let mut child_path = current_path.to_vec();
        child_path.push(child.node_id.clone());
        if remaining_depth == 0 || subtree_item_count(child) > threshold {
            children_nodes.push(register_selectable_branch(child, &child_path, entries));
            continue;
        }
        children_nodes.push(expand_node(
            child,
            &child_path,
            remaining_depth - 1,
            false,
            config,
            subtree_item_count,
            entries,
        ));
    }
    exposed_branch(node, children_nodes)
}

/// Whether `node` is a unary, itemless, non-root chain link that is spliced away.
fn should_auto_expand_single_child(
    is_root: bool,
    remaining_depth: usize,
    node: &RetrieverNode,
    child_nodes: &[&RetrieverNode],
) -> bool {
    if is_root || !node.items().is_empty() {
        return false;
    }
    remaining_depth > 0 && child_nodes.len() == 1
}

/// The exposed view of a category node (never selectable).
fn exposed_branch(node: &RetrieverNode, children: Vec<ExposedNode>) -> ExposedNode {
    ExposedNode {
        canonical_id: format!("{BRANCH_CANONICAL_PREFIX}{}", node.node_id),
        label: non_blank(&node.label, &node.node_id),
        description: node.description.clone(),
        is_selectable: false,
        selectable_canonical_id: None,
        children,
    }
}

/// Registers one item as a selectable terminal leaf.
fn push_item_entry(
    item: &RetrieverItem,
    current_path: &[String],
    entries: &mut Vec<SelectableEntry>,
) -> ExposedNode {
    let identity = if item.payload.is_empty() {
        item.item_id.clone()
    } else {
        item.payload.clone()
    };
    let canonical_id = format!("{ITEM_CANONICAL_PREFIX}{identity}");
    let label = non_blank(&item.label, &item.item_id);
    entries.push(SelectableEntry {
        canonical_id: canonical_id.clone(),
        label: label.clone(),
        description: item.description.clone(),
        selectable_canonical_id: identity.clone(),
        is_terminal: true,
        branch_path: current_path.to_vec(),
        node: None,
        item: Some(item.clone()),
    });
    ExposedNode {
        canonical_id,
        label,
        description: item.description.clone(),
        is_selectable: true,
        selectable_canonical_id: Some(identity),
        children: Vec::new(),
    }
}

/// Registers one collapsed branch as a selectable non-terminal leaf.
fn register_selectable_branch(
    child: &RetrieverNode,
    child_path: &[String],
    entries: &mut Vec<SelectableEntry>,
) -> ExposedNode {
    let canonical_id = format!("{BRANCH_CANONICAL_PREFIX}{}", child.node_id);
    let label = non_blank(&child.label, &child.node_id);
    entries.push(SelectableEntry {
        canonical_id: canonical_id.clone(),
        label: label.clone(),
        description: child.description.clone(),
        selectable_canonical_id: child.node_id.clone(),
        is_terminal: false,
        branch_path: child_path.to_vec(),
        node: Some(child.clone()),
        item: None,
    });
    ExposedNode {
        canonical_id,
        label,
        description: child.description.clone(),
        is_selectable: true,
        selectable_canonical_id: Some(child.node_id.clone()),
        children: Vec::new(),
    }
}

/// `primary` when it is not blank, else `fallback`.
fn non_blank(primary: &str, fallback: &str) -> String {
    if primary.is_empty() {
        fallback.to_owned()
    } else {
        primary.to_owned()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::retrieval::types::ROOT_NODE_ID;

    fn item(item_id: &str, payload: &str, label: &str) -> RetrieverItem {
        RetrieverItem::new(item_id, payload)
            .with_label(label)
            .with_description(format!("{label} description"))
    }

    /// ROOT
    /// ├── Skills (Alpha: one item, Beta: one item)
    /// └── Tools  (one item)
    fn tree() -> RetrieverNode {
        RetrieverNode::new(ROOT_NODE_ID, "Root").with_children(vec![
            RetrieverNode::new("Skills", "Skills").with_children(vec![
                RetrieverNode::new("Skills.alpha", "Alpha").with_items(vec![item(
                    "alpha",
                    "custom.alpha",
                    "Alpha One",
                )]),
                RetrieverNode::new("Skills.beta", "Beta").with_items(vec![item(
                    "beta",
                    "custom.beta",
                    "Beta One",
                )]),
            ]),
            RetrieverNode::new("Tools", "Tools").with_items(vec![item(
                "tool",
                "custom.tool",
                "Tool One",
            )]),
        ])
    }

    fn fragment(config: &DisclosureConfig) -> ExposedFragment {
        build_exposed_fragment(&tree(), &[ROOT_NODE_ID.to_owned()], config, &|node| {
            node.item_count()
        })
        .unwrap()
    }

    #[test]
    fn exposure_orders_items_before_children_and_assigns_codes_in_tree_order() {
        let exposed = fragment(&DisclosureConfig::default());
        assert_eq!(
            exposed.candidate_codes,
            vec!["Alpha".to_owned(), "Beta".to_owned(), "Tool".to_owned()]
        );
        assert_eq!(
            exposed.rendered_tree,
            "- Category Root\n  \
             - Category Skills\n    \
             - Category Alpha\n      \
             - Candidate Alpha: Alpha One description\n    \
             - Category Beta\n      \
             - Candidate Beta: Beta One description\n  \
             - Category Tools\n    \
             - Candidate Tool: Tool One description"
        );
        let alpha = exposed.resolution("Alpha").unwrap();
        assert_eq!(alpha.canonical_id, "custom.alpha");
        assert!(alpha.is_terminal);
        assert_eq!(alpha.branch_path, vec!["ROOT", "Skills", "Skills.alpha"]);
        assert_eq!(
            alpha.item.as_ref().map(|item| item.item_id.as_str()),
            Some("alpha")
        );
        assert!(exposed.root.children[0].children[0].children[0].is_selectable);
        assert!(!exposed.root.is_selectable);
    }

    #[test]
    fn a_unary_itemless_chain_is_spliced_away() {
        let tree =
            RetrieverNode::new(ROOT_NODE_ID, "Root").with_children(vec![RetrieverNode::new(
                "Skills", "Skills",
            )
            .with_children(vec![RetrieverNode::new("Skills.only", "Only")
                .with_items(vec![item("x", "custom.x", "X")])])]);
        let exposed = build_exposed_fragment(
            &tree,
            &[ROOT_NODE_ID.to_owned()],
            &DisclosureConfig::default(),
            &exposure_item_count,
        )
        .unwrap();
        assert_eq!(exposed.candidate_codes, vec!["X".to_owned()]);
        assert_eq!(
            exposed.rendered_tree,
            "- Category Root\n  - Category Skills\n    - Category Only\n      - Candidate X: X description"
        );
        // The item leaf's path is the chain to its *parent*, so the spliced `Skills`
        // segment is still recorded but `Skills.only` is not repeated.
        assert_eq!(
            exposed.resolution("X").unwrap().branch_path,
            vec![
                "ROOT".to_owned(),
                "Skills".to_owned(),
                "Skills.only".to_owned()
            ]
        );
    }

    #[test]
    fn the_remaining_depth_collapses_children_into_selectable_branches() {
        let exposed = fragment(&DisclosureConfig::default().with_max_exposure_depth_per_call(1));
        assert_eq!(
            exposed.candidate_codes,
            vec!["Alpha".to_owned(), "Beta".to_owned(), "Tool".to_owned()]
        );
        let collapsed = exposed.resolution("Alpha").unwrap();
        assert!(!collapsed.is_terminal);
        assert_eq!(collapsed.node_id.as_deref(), Some("Skills.alpha"));
        assert_eq!(
            exposed.rendered_tree,
            "- Category Root\n  \
             - Category Skills\n    \
             - Candidate Alpha\n    \
             - Candidate Beta\n  \
             - Category Tools\n    \
             - Candidate Tool: Tool One description"
        );

        let flat_depth = fragment(&DisclosureConfig::default().with_max_exposure_depth_per_call(0));
        assert_eq!(
            flat_depth.candidate_codes,
            vec!["Skills".to_owned(), "Tools".to_owned()]
        );
        assert!(flat_depth
            .resolutions()
            .iter()
            .all(|resolution| !resolution.is_terminal));
    }

    #[test]
    fn the_exposure_threshold_collapses_a_large_subtree() {
        let exposed = fragment(&DisclosureConfig::default().with_exposure_threshold(1));
        assert_eq!(
            exposed.candidate_codes,
            vec!["Skills".to_owned(), "Tool".to_owned()]
        );
        let collapsed = exposed.resolution("Skills").unwrap();
        assert!(!collapsed.is_terminal);
        assert_eq!(collapsed.node_id.as_deref(), Some("Skills"));
        assert!(exposed.resolution("Tool").unwrap().is_terminal);
    }

    #[test]
    fn max_items_per_branch_hides_items_from_the_text_but_keeps_them_selectable() {
        let tree =
            RetrieverNode::new(ROOT_NODE_ID, "Root").with_children(vec![RetrieverNode::new(
                "Skills", "Skills",
            )
            .with_items(vec![
                item("a", "custom.a", "A"),
                item("b", "custom.b", "B"),
                item("c", "custom.c", "C"),
            ])]);
        let config = DisclosureConfig::default().with_max_items_per_branch(2);
        let exposed = build_exposed_fragment(
            &tree,
            &[ROOT_NODE_ID.to_owned()],
            &config,
            &exposure_item_count,
        )
        .unwrap();
        assert_eq!(exposed.rendered_tree.matches("Candidate").count(), 2);
        assert!(exposed.rendered_tree.contains("Candidate A"));
        assert!(exposed.rendered_tree.contains("Candidate B"));
        assert!(!exposed.rendered_tree.contains("Candidate C"));
        assert_eq!(exposed.candidate_codes.len(), 3);
        assert!(exposed.resolution("C").is_some());
        assert_eq!(exposed.root.children[0].children.len(), 3);
        assert_eq!(
            exposed.root.children[0].children.len(),
            exposed.candidate_codes.len()
        );
    }

    #[test]
    fn the_char_budget_truncates_and_reports_the_rendered_length() {
        let exposed = fragment(&DisclosureConfig::default().with_max_chars(20));
        assert!(exposed.truncated);
        assert!(exposed.rendered_chars <= 20, "{}", exposed.rendered_chars);
        assert_eq!(
            exposed.rendered_chars,
            exposed.rendered_tree.chars().count()
        );
        assert!(exposed.rendered_tree.ends_with("..."));

        let full = fragment(&DisclosureConfig::default());
        assert!(!full.truncated);
        assert_eq!(full.rendered_chars, full.rendered_tree.chars().count());
        assert!(full.rendered_chars > 20);
    }

    #[test]
    fn nothing_is_truncated_when_the_budget_is_zero() {
        let exposed = fragment(&DisclosureConfig::default().with_max_chars(0));
        assert!(!exposed.truncated);
        assert!(exposed.rendered_tree.ends_with("Tool One description"));
    }

    #[test]
    fn names_mode_codes_are_pascal_boundary_names_with_duplicates_disambiguated() {
        let tree =
            RetrieverNode::new(ROOT_NODE_ID, "Root").with_children(vec![RetrieverNode::new(
                "Skills", "Skills",
            )
            .with_children(vec![
                RetrieverNode::new("Skills.alpha", "Alpha").with_items(vec![item(
                    "shared",
                    "custom.alpha.shared",
                    "One",
                )]),
                RetrieverNode::new("Skills.beta", "Beta").with_items(vec![item(
                    "shared",
                    "custom.beta.shared",
                    "Two",
                )]),
            ])]);
        let exposed = build_exposed_fragment(
            &tree,
            &[ROOT_NODE_ID.to_owned()],
            &DisclosureConfig::default(),
            &exposure_item_count,
        )
        .unwrap();
        assert_eq!(
            exposed.candidate_codes,
            vec![
                "Skills/SkillsAlpha/Shared".to_owned(),
                "Skills/SkillsBeta/Shared".to_owned()
            ]
        );

        // Without a path segment the `__{index}` fallback applies.
        let tree = RetrieverNode::new(ROOT_NODE_ID, "Root").with_items(vec![
            item("shared", "custom.shared", "One"),
            item("shared2", "custom.shared", "Two"),
        ]);
        let exposed = build_exposed_fragment(
            &tree,
            &[ROOT_NODE_ID.to_owned()],
            &DisclosureConfig::default(),
            &exposure_item_count,
        )
        .unwrap();
        assert_eq!(
            exposed.candidate_codes,
            vec!["Shared__1".to_owned(), "Shared__2".to_owned()]
        );
    }

    #[test]
    fn a_short_codebook_is_a_validation_error_naming_the_shortfall() {
        let config = DisclosureConfig::default()
            .with_compact_codes_enabled(true)
            .with_compact_codebook(vec!["AA".to_owned()]);
        let error = build_exposed_fragment(
            &tree(),
            &[ROOT_NODE_ID.to_owned()],
            &config,
            &exposure_item_count,
        )
        .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(
            error
                .message()
                .contains("provides 1 codes, but 3 selectable"),
            "{}",
            error.message()
        );

        let valid = fragment(
            &DisclosureConfig::default()
                .with_compact_codes_enabled(true)
                .with_compact_codebook(vec!["AB".into(), "AC".into(), "AD".into()]),
        );
        assert_eq!(
            valid.candidate_codes,
            vec!["AB".to_owned(), "AC".to_owned(), "AD".to_owned()]
        );
        assert_eq!(valid.code_width, 2);
        assert!(valid
            .rendered_tree
            .contains("- Alpha [id: AB]: Alpha One description"));
    }

    #[test]
    fn compact_codes_without_a_codebook_are_numeric() {
        let exposed = fragment(&DisclosureConfig::default().with_compact_codes_enabled(true));
        assert_eq!(
            exposed.candidate_codes,
            vec!["1".to_owned(), "2".to_owned(), "3".to_owned()]
        );
        assert_eq!(exposed.code_width, 1);
        assert!(exposed
            .rendered_tree
            .contains("- Alpha [id: 1]: Alpha One description"));
    }

    #[test]
    fn code_width_is_common_mixed_or_overridden() {
        let uniform = fragment(&DisclosureConfig::default().with_compact_codes_enabled(true));
        assert_eq!(uniform.code_width, 1);

        let mixed = fragment(
            &DisclosureConfig::default()
                .with_compact_codes_enabled(true)
                .with_compact_codebook(vec!["A".into(), "BB".into(), "CCC".into()]),
        );
        assert_eq!(mixed.code_width, 0);

        let overridden = fragment(
            &DisclosureConfig::default()
                .with_compact_codes_enabled(true)
                .with_compact_codebook(vec!["A".into(), "BB".into(), "CCC".into()])
                .with_code_width(3),
        );
        assert_eq!(overridden.code_width, 3);

        let empty = build_exposed_fragment(
            &RetrieverNode::new(ROOT_NODE_ID, "Root"),
            &[ROOT_NODE_ID.to_owned()],
            &DisclosureConfig::default(),
            &exposure_item_count,
        )
        .unwrap();
        assert_eq!(empty.code_width, 1);
        assert!(empty.is_empty());
    }

    #[test]
    fn the_fingerprint_is_stable_and_sensitive_to_labels() {
        let first = fragment(&DisclosureConfig::default());
        let second = fragment(&DisclosureConfig::default());
        assert_eq!(first.fragment_fingerprint, second.fragment_fingerprint);
        assert_eq!(first.fragment_fingerprint.len(), 16);
        assert!(first
            .fragment_fingerprint
            .chars()
            .all(|character| character.is_ascii_hexdigit()));

        let mut relabelled = first.root.clone();
        relabelled.children[0].label = "Skills Renamed".to_owned();
        assert_ne!(
            fragmentation_fingerprint(&relabelled, &first.candidate_codes),
            first.fragment_fingerprint
        );
        assert_ne!(
            fragmentation_fingerprint(&first.root, &["Other".to_owned()]),
            first.fragment_fingerprint
        );
    }

    #[test]
    fn descriptions_are_stripped_and_clamped() {
        assert_eq!(
            strip_representative_descendants("Real text.\nRepresentative descendants: a, b"),
            "Real text."
        );
        assert_eq!(
            strip_representative_descendants("Real text. Representative descendants: a, b"),
            "Real text."
        );
        assert_eq!(
            strip_representative_descendants("Representative descendants: a"),
            "Representative descendants: a"
        );
        assert_eq!(
            strip_representative_descendants("Text without the remark"),
            "Text without the remark"
        );

        let bundled: String = "能力树索引构建".repeat(20);
        let clamp = clamp_description(&bundled, 10);
        assert!(clamp.ends_with("..."));
        assert_eq!(clamp.chars().count(), 13);

        let sentence = format!(
            "{}{}",
            "很长的描述".repeat(10),
            "。后面还有更多内容再来一段"
        );
        let clamped = clamp_description(&sentence, 60);
        assert_eq!(clamped, format!("{}。", "很长的描述".repeat(10)));

        assert_eq!(clamp_description("short", 150), "short");
        assert_eq!(clamp_description("short", 0), "short");

        let tree = RetrieverNode::new(ROOT_NODE_ID, "Root").with_items(vec![RetrieverItem::new(
            "a", "custom.a",
        )
        .with_label("A")
        .with_description("正文。 Representative descendants: x, y")]);
        let exposed = build_exposed_fragment(
            &tree,
            &[ROOT_NODE_ID.to_owned()],
            &DisclosureConfig::default().with_max_description_chars(40),
            &exposure_item_count,
        )
        .unwrap();
        assert_eq!(
            exposed.rendered_tree,
            "- Category Root\n  - Candidate A: 正文。"
        );
    }

    #[test]
    fn flat_mode_sorts_alphabetically_and_renders_fixed_json_fields() {
        let config = DisclosureConfig::default()
            .with_flatten_full_tree_in_prompt(true)
            .with_compact_codes_enabled(true)
            .with_compact_codebook(vec!["AA".into(), "AB".into(), "AC".into()]);
        let exposed = fragment(&config);
        assert!(exposed.flat_list_mode);
        assert_eq!(
            exposed.candidate_codes,
            vec!["AA".to_owned(), "AB".to_owned(), "AC".to_owned()]
        );
        let names: Vec<&str> = exposed
            .resolutions()
            .iter()
            .map(|resolution| resolution.canonical_id.as_str())
            .collect();
        assert_eq!(names, vec!["custom.alpha", "custom.beta", "custom.tool"]);

        let parsed: Value = serde_json::from_str(&exposed.rendered_tree).unwrap();
        let array = parsed.as_array().unwrap();
        assert_eq!(array.len(), 3);
        let first = array[0].as_object().unwrap();
        assert_eq!(
            first.keys().collect::<Vec<&String>>(),
            vec!["category", "name", "raw_name", "description", "id"]
        );
        // The category is the last three branch-path segments; the reference strips the
        // last `.` segment of each id, so the dotted cid yields its lowercase tail.
        assert_eq!(first["category"], "Skills > alpha");
        assert_eq!(first["name"], "Alpha One");
        assert_eq!(first["raw_name"], "Alpha");
        assert_eq!(first["description"], "Alpha One description");
        assert_eq!(first["id"], "AA");
        assert_eq!(exposed.root.children.len(), 3);
        assert!(exposed.root.children.iter().all(|node| node.is_selectable));

        let names_mode =
            fragment(&DisclosureConfig::default().with_flatten_full_tree_in_prompt(true));
        assert!(!names_mode.compact_codes_enabled);
        assert_eq!(
            names_mode.rendered_tree,
            "- Alpha: Alpha One description\n- Beta: Beta One description\n- Tool: Tool One description"
        );
        assert!(!names_mode.rendered_tree.contains("Category"));
    }

    #[test]
    fn the_category_join_keeps_the_last_three_localized_labels() {
        assert_eq!(
            format_branch_category(&["ROOT".to_owned(), "Skills".to_owned()]),
            "Skills"
        );
        assert_eq!(
            format_branch_category(&[
                "ROOT".to_owned(),
                "office-docs".to_owned(),
                "documents".to_owned()
            ]),
            "办公文档 > Documents"
        );
        assert_eq!(
            format_branch_category(&[
                "ROOT".to_owned(),
                "a".to_owned(),
                "b".to_owned(),
                "c".to_owned(),
                "d".to_owned()
            ]),
            "b > c > d"
        );
        assert_eq!(format_branch_category(&[]), "");
        assert_eq!(format_branch_category(&["ROOT".to_owned()]), "");
        assert_eq!(
            format_branch_category(&["root".to_owned(), "Skills.office-docs".to_owned()]),
            "办公文档"
        );
    }

    #[test]
    fn cjk_truncation_stays_on_character_boundaries() {
        let tree = RetrieverNode::new(ROOT_NODE_ID, "能力树").with_items(vec![RetrieverItem::new(
            "a", "custom.a",
        )
        .with_label("摘要")
        .with_description("把长文档总结成要点")]);
        let exposed = build_exposed_fragment(
            &tree,
            &[ROOT_NODE_ID.to_owned()],
            &DisclosureConfig::default().with_max_chars(16),
            &exposure_item_count,
        )
        .unwrap();
        assert!(exposed.truncated);
        // 13 characters of the head plus the three-character ellipsis: the cut lands
        // between `力` and `树`, so no half character can survive.
        assert_eq!(exposed.rendered_tree, "- Category 能力...");
        assert_eq!(exposed.rendered_chars, 16);
        assert_eq!(
            exposed.rendered_chars,
            exposed.rendered_tree.chars().count()
        );
        assert!(!exposed.rendered_tree.contains('\u{FFFD}'));

        let whole = build_exposed_fragment(
            &tree,
            &[ROOT_NODE_ID.to_owned()],
            &DisclosureConfig::default(),
            &exposure_item_count,
        )
        .unwrap();
        assert_eq!(
            whole.rendered_tree,
            "- Category 能力树\n  - Candidate A: 把长文档总结成要点"
        );
        assert!(!whole.truncated);
    }

    #[test]
    fn pascal_case_and_name_parts_mirror_the_reference() {
        assert_eq!(pascal_case("office-docs"), "OfficeDocs");
        assert_eq!(pascal_case("mySkill"), "MySkill");
        assert_eq!(pascal_case("ABCdef"), "Abcdef");
        assert_eq!(pascal_case("2 fast"), "2Fast");
        assert_eq!(pascal_case("***"), "");
        assert_eq!(pascal_case("能力树"), "");
        assert_eq!(split_name_parts("  a_b-C  "), vec!["a", "b", "c"]);
        assert_eq!(pascal_case(""), "");
    }
}
