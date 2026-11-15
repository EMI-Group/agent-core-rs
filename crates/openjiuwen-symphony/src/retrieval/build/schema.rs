//! The capability-tree schema: shape bounds, diagnostics and the fixed root taxonomy.
//!
//! Ports the tree-shape constants of Python `openjiuwen.symphony.retrieval.build.tree`
//! (v0.1.20) — `schema.py`'s `MAX_DEPTH` and `BRANCHING_FACTOR` — and the whole of
//! `root_categories.py` (`FIXED_ROOT_CATEGORIES`).
//!
//! # A Rust-only guard
//!
//! Python validates none of this: `schema.py` has no tree validator and
//! `build/io/tree.py` only parses, so whatever the model wrote reaches the search
//! runtime. `TreeSchema::validate` is the Rust guard instead. It reports every shape
//! problem it finds as a `TreeDiagnostic`, never failing as it goes, and
//! `TreeSchema::validate_or_error` turns the first one into an `Error::Validation`.
//! **Every diagnostic is an error** — there is no warning tier;
//! `TreeDiagnostic::is_error` exists so a caller can filter the list without matching
//! on codes.
//!
//! Two entry points check two shapes:
//!
//! - `TreeSchema::validate` checks a flat `tree_index.yaml` node list.
//! - `TreeSchema::validate_preset` checks a nested `TreePreset`
//!   and runs the flat checks over its flattened nodes. Nesting is the only way to
//!   express a cycle, so that is where the `cycle` diagnostic comes from.
//!
//! # The fixed root taxonomy
//!
//! `FIXED_ROOT_CATEGORIES` is Python's `FIXED_ROOT_CATEGORIES` with the same eight root
//! ids and the same four children each. The **ids are the behavioural contract** — they
//! become `cid` segments and preset paths — while `name` and `description` are display
//! text only: the Rust build derives and validates the ids and never sends them to a model.
//! Root names are Python's Chinese display names verbatim; child names are title-cased
//! forms of their ids.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{validation, Result};
use crate::retrieval::build::presets::TreePreset;
use crate::retrieval::types::{IndexNode, NODE_TYPE_BRANCH, NODE_TYPE_LEAF};

/// The deepest `cid` a tree may have when a build does not configure one.
///
/// Mirrors Python `MAX_DEPTH`.
pub const DEFAULT_MAX_DEPTH: usize = 6;

/// How many direct children one branch may have when a build does not configure it.
///
/// Mirrors Python `BRANCHING_FACTOR`.
pub const DEFAULT_MAX_CHILDREN: usize = 8;

/// How many nodes one tree may hold when a build does not configure it.
///
/// Python has no equivalent bound; it is a Rust-only ceiling so a runaway model cannot
/// exhaust memory.
pub const DEFAULT_MAX_NODES: usize = 100_000;

/// Diagnostic code for a node with a blank `cid`.
pub const DIAGNOSTIC_BLANK_CID: &str = "blank_cid";
/// Diagnostic code for a `cid` that appears more than once.
pub const DIAGNOSTIC_DUPLICATE_CID: &str = "duplicate_cid";
/// Diagnostic code for a branch with neither a description nor a worker id.
pub const DIAGNOSTIC_BLANK_LABEL: &str = "blank_label";
/// Diagnostic code for a `cid` deeper than [`TreeSchema::max_depth`].
pub const DIAGNOSTIC_EXCESSIVE_DEPTH: &str = "excessive_depth";
/// Diagnostic code for a `cid` whose parent is not in the node list.
pub const DIAGNOSTIC_MISSING_PARENT: &str = "missing_parent";
/// Diagnostic code for a parent with more than [`TreeSchema::max_children`] children.
pub const DIAGNOSTIC_EXCESSIVE_FAN_OUT: &str = "excessive_fan_out";
/// Diagnostic code for a node type other than `branch` or `leaf`.
pub const DIAGNOSTIC_INVALID_NODE_TYPE: &str = "invalid_node_type";
/// Diagnostic code for a leaf with a blank worker id.
pub const DIAGNOSTIC_MISSING_WORKER_ID: &str = "missing_worker_id";
/// Diagnostic code for a nested node whose `cid` repeats an ancestor's `cid`.
pub const DIAGNOSTIC_CYCLE: &str = "cycle";
/// Diagnostic code for a tree holding more than [`TreeSchema::max_nodes`] nodes.
pub const DIAGNOSTIC_TOO_MANY_NODES: &str = "too_many_nodes";

/// One shape problem found in a tree.
///
/// `code` is one of the `DIAGNOSTIC_*` constants so a caller can match on it without
/// parsing prose, `cid` names the offending node (blank when the problem has no single
/// node, such as [`DIAGNOSTIC_TOO_MANY_NODES`]), and `detail` is the actionable text.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TreeDiagnostic {
    /// The diagnostic code; one of the `DIAGNOSTIC_*` constants.
    pub code: String,
    /// The `cid` of the node the diagnostic is about.
    pub cid: String,
    /// The actionable description of the problem.
    pub detail: String,
}

impl TreeDiagnostic {
    /// Builds a diagnostic from its code, subject `cid` and detail.
    #[must_use]
    pub fn new(code: impl Into<String>, cid: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            cid: cid.into(),
            detail: detail.into(),
        }
    }

    /// Whether this diagnostic aborts a build.
    ///
    /// Always `true`: the tree guard has a single severity, so a caller that collects
    /// several diagnostics can still treat each one as fatal.
    #[must_use]
    pub fn is_error(&self) -> bool {
        true
    }
}

/// Builds a diagnostic whose detail is formatted.
fn diagnostic(code: &str, cid: &str, detail: impl fmt::Display) -> TreeDiagnostic {
    TreeDiagnostic::new(code, cid, detail.to_string())
}

/// Sorts diagnostics by `(cid, code)` and drops exact repeats.
fn sort_diagnostics(diagnostics: &mut Vec<TreeDiagnostic>) {
    diagnostics.sort_by(|left, right| {
        (left.cid.as_str(), left.code.as_str()).cmp(&(right.cid.as_str(), right.code.as_str()))
    });
    diagnostics.dedup();
}

/// The shape bounds a tree must satisfy.
///
/// Python keeps the same limits as module-level constants and never enforces them; here
/// they are carried by a value so a build can tighten them, and
/// [`TreeSchema::validate`] enforces them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TreeSchema {
    /// The deepest `cid` a node may have.
    pub max_depth: usize,
    /// How many direct children one parent may have.
    pub max_children: usize,
    /// How many nodes the whole tree may hold.
    pub max_nodes: usize,
}

impl Default for TreeSchema {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            max_children: DEFAULT_MAX_CHILDREN,
            max_nodes: DEFAULT_MAX_NODES,
        }
    }
}

impl TreeSchema {
    /// Builds the default schema: [`DEFAULT_MAX_DEPTH`], [`DEFAULT_MAX_CHILDREN`],
    /// [`DEFAULT_MAX_NODES`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the schema with a different depth bound.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Returns the schema with a different fan-out bound.
    #[must_use]
    pub fn with_max_children(mut self, max_children: usize) -> Self {
        self.max_children = max_children;
        self
    }

    /// Returns the schema with a different node-count bound.
    #[must_use]
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Checks a flat `tree_index.yaml` node list.
    ///
    /// Returns every diagnostic found, sorted by `(cid, code)`; it never fails, so a
    /// caller can report all problems at once.
    ///
    /// The checks are: a blank `cid`; a `cid` used more than once; a `cid` whose
    /// [`depth`](crate::retrieval::types::IndexNode::depth) exceeds
    /// [`max_depth`](Self::max_depth); a parent `cid` that is absent from the list; more
    /// than [`max_children`](Self::max_children) direct children under one parent; a node
    /// type other than `branch` or `leaf`; a branch with both a blank description and a
    /// blank worker id; a leaf with a blank worker id (a leaf with a blank description is
    /// fine, its item supplies the text); and more than
    /// [`max_nodes`](Self::max_nodes) nodes.
    #[must_use]
    pub fn validate(&self, nodes: &[IndexNode]) -> Vec<TreeDiagnostic> {
        let mut diagnostics: Vec<TreeDiagnostic> = Vec::new();
        let mut cids: BTreeSet<&str> = BTreeSet::new();
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut fan_out: BTreeMap<&str, usize> = BTreeMap::new();
        for node in nodes {
            let cid = node.cid.trim();
            if cid.is_empty() {
                diagnostics.push(diagnostic(
                    DIAGNOSTIC_BLANK_CID,
                    "",
                    "the node has a blank cid",
                ));
                continue;
            }
            cids.insert(cid);
            *counts.entry(cid).or_default() += 1;
            if let Some(position) = cid.rfind('.') {
                let parent = &cid[..position];
                if !parent.is_empty() {
                    *fan_out.entry(parent).or_default() += 1;
                }
            }
        }
        for (cid, count) in &counts {
            if *count > 1 {
                diagnostics.push(diagnostic(
                    DIAGNOSTIC_DUPLICATE_CID,
                    cid,
                    format_args!("the cid appears {count} times"),
                ));
            }
        }
        for (parent, count) in &fan_out {
            if *count > self.max_children {
                diagnostics.push(diagnostic(
                    DIAGNOSTIC_EXCESSIVE_FAN_OUT,
                    parent,
                    format_args!("{count} children exceed the limit of {}", self.max_children),
                ));
            }
        }
        self.check_nodes(nodes, &cids, &mut diagnostics);
        if nodes.len() > self.max_nodes {
            diagnostics.push(diagnostic(
                DIAGNOSTIC_TOO_MANY_NODES,
                "",
                format_args!(
                    "{} nodes exceed the limit of {}",
                    nodes.len(),
                    self.max_nodes
                ),
            ));
        }
        sort_diagnostics(&mut diagnostics);
        diagnostics
    }

    /// Checks the per-node shape rules into `diagnostics`.
    fn check_nodes(
        &self,
        nodes: &[IndexNode],
        cids: &BTreeSet<&str>,
        diagnostics: &mut Vec<TreeDiagnostic>,
    ) {
        for node in nodes {
            let cid = node.cid.trim();
            if cid.is_empty() {
                continue;
            }
            let depth = node.depth();
            if depth > self.max_depth {
                diagnostics.push(diagnostic(
                    DIAGNOSTIC_EXCESSIVE_DEPTH,
                    cid,
                    format_args!("depth {depth} exceeds the limit of {}", self.max_depth),
                ));
            }
            if let Some(position) = cid.rfind('.') {
                let parent = &cid[..position];
                if !cids.contains(parent) {
                    diagnostics.push(diagnostic(
                        DIAGNOSTIC_MISSING_PARENT,
                        cid,
                        format_args!("the parent cid `{parent}` is not in the node list"),
                    ));
                }
            }
            let node_type = node.node_type.trim();
            let type_valid = node_type.eq_ignore_ascii_case(NODE_TYPE_BRANCH)
                || node_type.eq_ignore_ascii_case(NODE_TYPE_LEAF);
            if type_valid {
                if node.is_branch() {
                    if node.description.trim().is_empty() && node.worker_id.trim().is_empty() {
                        diagnostics.push(diagnostic(
                            DIAGNOSTIC_BLANK_LABEL,
                            cid,
                            "the branch has neither a description nor a worker id",
                        ));
                    }
                } else if node.worker_id.trim().is_empty() {
                    diagnostics.push(diagnostic(
                        DIAGNOSTIC_MISSING_WORKER_ID,
                        cid,
                        "the leaf has no worker id",
                    ));
                }
            } else if node_type.is_empty() {
                diagnostics.push(diagnostic(
                    DIAGNOSTIC_INVALID_NODE_TYPE,
                    cid,
                    "the node type is blank",
                ));
            } else {
                diagnostics.push(diagnostic(
                    DIAGNOSTIC_INVALID_NODE_TYPE,
                    cid,
                    format_args!(
                        "the node type `{node_type}` is neither `{NODE_TYPE_BRANCH}` nor \
                         `{NODE_TYPE_LEAF}`"
                    ),
                ));
            }
        }
    }

    /// [`validate`](Self::validate) plus an `Error::Validation` naming the first problem.
    ///
    /// # Errors
    ///
    /// `Error::Validation` naming `tree_index.yaml`, the first diagnostic's code, its
    /// `cid` and its detail.
    pub fn validate_or_error(&self, nodes: &[IndexNode]) -> Result<()> {
        let diagnostics = self.validate(nodes);
        match diagnostics.first() {
            Some(first) => Err(validation(
                "tree_index.yaml",
                format_args!("{} (cid `{}`): {}", first.code, first.cid, first.detail),
            )),
            None => Ok(()),
        }
    }

    /// Checks a nested preset.
    ///
    /// Runs the flat checks over [`TreePreset::flatten`]'s nodes and adds the two checks
    /// only nesting can express: `cycle` when a node's derived `cid` equals an ancestor's
    /// `cid`, and `blank_cid` when a node has no `cid`, description or worker id, so no
    /// `cid` can be derived for it. Diagnostics are sorted by `(cid, code)`.
    #[must_use]
    pub fn validate_preset(&self, preset: &TreePreset) -> Vec<TreeDiagnostic> {
        let entries = preset.entries();
        let flat: Vec<IndexNode> = entries.iter().map(|entry| entry.node.clone()).collect();
        let mut diagnostics = self.validate(&flat);
        for entry in &entries {
            if !entry.derivable {
                diagnostics.push(diagnostic(
                    DIAGNOSTIC_BLANK_CID,
                    &entry.node.cid,
                    "the node has no cid, description or worker id to derive one from",
                ));
            }
            if entry.ancestors.contains(&entry.node.cid) {
                diagnostics.push(diagnostic(
                    DIAGNOSTIC_CYCLE,
                    &entry.node.cid,
                    "the cid repeats the cid of an ancestor",
                ));
            }
        }
        sort_diagnostics(&mut diagnostics);
        diagnostics
    }

    /// [`validate_preset`](Self::validate_preset) plus an `Error::Validation` naming the
    /// first problem.
    ///
    /// # Errors
    ///
    /// `Error::Validation` naming `tree_preset`, the first diagnostic's code, its `cid`
    /// and its detail.
    pub fn validate_preset_or_error(&self, preset: &TreePreset) -> Result<()> {
        let diagnostics = self.validate_preset(preset);
        match diagnostics.first() {
            Some(first) => Err(validation(
                "tree_preset",
                format_args!("{} (cid `{}`): {}", first.code, first.cid, first.detail),
            )),
            None => Ok(()),
        }
    }
}

/// One node of Python's `FIXED_ROOT_CATEGORIES` taxonomy.
///
/// The fields are `&'static` so the whole taxonomy is one `const` with no allocation and
/// no runtime initialisation. That is also why the type implements no `Serialize`,
/// `Deserialize` or `JsonSchema`: a `&'static [RootCategory]` cannot round-trip through a
/// document, and nothing reads the taxonomy from a file — Python's
/// `load_tree_root_categories` file loader is not ported, a caller hands its own taxonomy
/// to [`TreePreset`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RootCategory {
    /// The category id; the behavioural contract, used as a `cid` segment.
    pub id: &'static str,
    /// The display name; for the roots, Python's Chinese name verbatim.
    pub name: &'static str,
    /// One short English sentence describing what the category holds.
    pub description: &'static str,
    /// The category's children; empty for a leaf category.
    pub children: &'static [RootCategory],
}

impl RootCategory {
    /// Looks up a direct child by id.
    #[must_use]
    pub fn child(&self, id: &str) -> Option<&'static RootCategory> {
        self.children.iter().find(|child| child.id == id)
    }

    /// Whether the category has no children.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Builds one child category of the fixed taxonomy.
const fn child(id: &'static str, name: &'static str, description: &'static str) -> RootCategory {
    RootCategory {
        id,
        name,
        description,
        children: &[],
    }
}

/// The fixed root taxonomy: eight roots with four children each.
///
/// The ids are Python's `FIXED_ROOT_CATEGORIES` keys verbatim, in the same order, and the
/// root names are its Chinese display names. See the module docs for why names and
/// descriptions are display text only.
pub const FIXED_ROOT_CATEGORIES: &[RootCategory] = &[
    RootCategory {
        id: "office-docs",
        name: "办公文档",
        description: "Capabilities for office documents.",
        children: &[
            child(
                "documents",
                "Documents",
                "Capabilities for Word, PDF, text and Markdown document work.",
            ),
            child(
                "spreadsheets",
                "Spreadsheets",
                "Capabilities for spreadsheets, tabular data, formulas and charts.",
            ),
            child(
                "slides",
                "Slides",
                "Capabilities for slide decks, presentations and posters.",
            ),
            child(
                "mail-meeting",
                "Mail Meeting",
                "Capabilities for mail, calendars, meetings and office automation.",
            ),
        ],
    },
    RootCategory {
        id: "writing-content",
        name: "写作内容",
        description: "Capabilities for writing and content creation.",
        children: &[
            child(
                "general-writing",
                "General Writing",
                "Capabilities for articles, blogs, reports and scripts.",
            ),
            child(
                "marketing-copy",
                "Marketing Copy",
                "Capabilities for marketing copy, campaigns and conversion copy.",
            ),
            child(
                "story-creative",
                "Story Creative",
                "Capabilities for stories, fiction and creative writing.",
            ),
            child(
                "text-polish",
                "Text Polish",
                "Capabilities for rewriting, polishing and style cleanup.",
            ),
        ],
    },
    RootCategory {
        id: "search-research",
        name: "搜索研究",
        description: "Capabilities for search and research.",
        children: &[
            child(
                "web-search",
                "Web Search",
                "Capabilities for live web search and page lookup.",
            ),
            child(
                "deep-research",
                "Deep Research",
                "Capabilities for multi-step research and synthesis.",
            ),
            child(
                "news-papers",
                "News Papers",
                "Capabilities for news, papers and literature review.",
            ),
            child(
                "knowledge-notes",
                "Knowledge Notes",
                "Capabilities for knowledge bases, notes and bookmarks.",
            ),
        ],
    },
    RootCategory {
        id: "media-creative",
        name: "图片音视频",
        description: "Capabilities for image, video and audio work.",
        children: &[
            child(
                "image-understand",
                "Image Understand",
                "Capabilities for understanding and describing images.",
            ),
            child(
                "image-design",
                "Image Design",
                "Capabilities for image generation and visual design.",
            ),
            child(
                "video-animation",
                "Video Animation",
                "Capabilities for video, animation and motion work.",
            ),
            child(
                "audio-music",
                "Audio Music",
                "Capabilities for audio, speech and music work.",
            ),
        ],
    },
    RootCategory {
        id: "life-services",
        name: "生活服务",
        description: "Capabilities for everyday life services.",
        children: &[
            child(
                "travel",
                "Travel",
                "Capabilities for travel planning and booking support.",
            ),
            child(
                "maps-weather",
                "Maps Weather",
                "Capabilities for maps, routes and weather lookups.",
            ),
            child(
                "health-edu",
                "Health Edu",
                "Capabilities for health, fitness and education.",
            ),
            child(
                "local-deals",
                "Local Deals",
                "Capabilities for local services, deals and shopping.",
            ),
        ],
    },
    RootCategory {
        id: "product-dev",
        name: "开发产品",
        description: "Capabilities for product development.",
        children: &[
            child(
                "web-apps",
                "Web Apps",
                "Capabilities for web applications and frontend work.",
            ),
            child(
                "product-ux",
                "Product UX",
                "Capabilities for product design, UX and requirements.",
            ),
            child(
                "deploy-data",
                "Deploy Data",
                "Capabilities for deployment, operations and data stores.",
            ),
            child(
                "code-test",
                "Code Test",
                "Capabilities for coding, testing and code review.",
            ),
        ],
    },
    RootCategory {
        id: "finance-business",
        name: "金融商业",
        description: "Capabilities for finance and business work.",
        children: &[
            child(
                "market-data",
                "Market Data",
                "Capabilities for market data and quotes.",
            ),
            child(
                "investment",
                "Investment",
                "Capabilities for investment analysis and portfolios.",
            ),
            child(
                "business-data",
                "Business Data",
                "Capabilities for business analytics and reporting.",
            ),
            child(
                "trading-sim",
                "Trading Sim",
                "Capabilities for trading simulation and strategy.",
            ),
        ],
    },
    RootCategory {
        id: "system-tools",
        name: "系统工具",
        description: "Capabilities for system, device and agent tools.",
        children: &[
            child(
                "device-home",
                "Device Home",
                "Capabilities for devices, apps and smart home control.",
            ),
            child(
                "files-cloud",
                "Files Cloud",
                "Capabilities for local files and cloud storage.",
            ),
            child(
                "agents-tasks",
                "Agents Tasks",
                "Capabilities for skills, agents, personas and tasks.",
            ),
            child(
                "safety-channels",
                "Safety Channels",
                "Capabilities for safety checks and channel connections.",
            ),
        ],
    },
];

/// Looks up a top-level category of [`FIXED_ROOT_CATEGORIES`] by id.
#[must_use]
pub fn root_category(id: &str) -> Option<&'static RootCategory> {
    FIXED_ROOT_CATEGORIES
        .iter()
        .find(|category| category.id == id)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::retrieval::build::presets::{PresetNode, TreePreset};

    fn valid_tree() -> Vec<IndexNode> {
        vec![
            IndexNode::branch("Skills").with_description("root"),
            IndexNode::branch("Skills.alpha").with_description("alpha"),
            IndexNode::leaf("Skills.alpha.one", "worker-one"),
            IndexNode::leaf("Skills.beta", "worker-beta"),
        ]
    }

    fn preset_branch(cid: &str, description: &str, children: Vec<PresetNode>) -> PresetNode {
        PresetNode {
            cid: cid.to_string(),
            node_type: NODE_TYPE_BRANCH.to_string(),
            description: description.to_string(),
            children,
            ..PresetNode::default()
        }
    }

    fn codes(diagnostics: &[TreeDiagnostic]) -> Vec<&str> {
        diagnostics
            .iter()
            .map(|diagnostic| diagnostic.code.as_str())
            .collect()
    }

    #[test]
    fn a_well_formed_tree_has_no_diagnostics() {
        let schema = TreeSchema::new();
        assert_eq!(schema.validate(&valid_tree()), Vec::new());
        schema.validate_or_error(&valid_tree()).unwrap();
    }

    #[test]
    fn duplicate_cids_are_reported() {
        let mut nodes = valid_tree();
        nodes.push(IndexNode::branch("Skills.alpha").with_description("again"));
        let diagnostics = TreeSchema::new().validate(&nodes);
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_DUPLICATE_CID
                && diagnostic.cid == "Skills.alpha"
                && diagnostic.detail.contains("2 times")));
    }

    #[test]
    fn blank_cids_are_reported() {
        let nodes = vec![IndexNode::leaf("  ", "worker")];
        let diagnostics = TreeSchema::new().validate(&nodes);
        assert_eq!(codes(&diagnostics), vec![DIAGNOSTIC_BLANK_CID]);
        assert_eq!(diagnostics[0].cid, "");
        assert!(diagnostics[0].is_error());
    }

    #[test]
    fn blank_label_and_missing_worker_id_are_reported() {
        let nodes = vec![
            IndexNode::branch("Root"),
            IndexNode::leaf("Root.leaf", " "),
            IndexNode::leaf("Root.named", "worker").with_description(""),
            IndexNode::leaf("Root.labelled", "worker").with_description("text"),
        ];
        let diagnostics = TreeSchema::new().validate(&nodes);
        assert_eq!(
            codes(&diagnostics),
            vec![DIAGNOSTIC_BLANK_LABEL, DIAGNOSTIC_MISSING_WORKER_ID]
        );
        assert_eq!(diagnostics[0].cid, "Root");
        assert_eq!(diagnostics[1].cid, "Root.leaf");
    }

    #[test]
    fn excessive_depth_is_reported() {
        let schema = TreeSchema::new().with_max_depth(2);
        let diagnostics = schema.validate(&valid_tree());
        assert_eq!(codes(&diagnostics), vec![DIAGNOSTIC_EXCESSIVE_DEPTH]);
        assert_eq!(diagnostics[0].cid, "Skills.alpha.one");
        assert!(diagnostics[0].detail.contains("depth 3"));
    }

    #[test]
    fn missing_parent_is_reported() {
        let nodes = vec![IndexNode::branch("Skills.alpha.gamma").with_description("orphan")];
        let diagnostics = TreeSchema::new().validate(&nodes);
        assert_eq!(codes(&diagnostics), vec![DIAGNOSTIC_MISSING_PARENT]);
        assert_eq!(diagnostics[0].cid, "Skills.alpha.gamma");
        assert!(diagnostics[0].detail.contains("`Skills.alpha`"));
    }

    #[test]
    fn excessive_fan_out_is_reported() {
        let nodes = vec![
            IndexNode::branch("Root").with_description("root"),
            IndexNode::leaf("Root.a", "a"),
            IndexNode::leaf("Root.b", "b"),
        ];
        let schema = TreeSchema::new().with_max_children(1);
        let diagnostics = schema.validate(&nodes);
        assert_eq!(codes(&diagnostics), vec![DIAGNOSTIC_EXCESSIVE_FAN_OUT]);
        assert_eq!(diagnostics[0].cid, "Root");
        assert!(diagnostics[0].detail.contains("2 children"));
    }

    #[test]
    fn invalid_node_types_are_reported() {
        let nodes = vec![
            IndexNode::new("Root", "trunk").with_description("x"),
            IndexNode::new("Root.leaf", ""),
        ];
        let diagnostics = TreeSchema::new().validate(&nodes);
        assert_eq!(
            codes(&diagnostics),
            vec![DIAGNOSTIC_INVALID_NODE_TYPE, DIAGNOSTIC_INVALID_NODE_TYPE]
        );
        assert_eq!(diagnostics[0].cid, "Root");
        assert!(diagnostics[0].detail.contains("`trunk`"));
        assert_eq!(diagnostics[1].cid, "Root.leaf");
        assert!(diagnostics[1].detail.contains("blank"));
    }

    #[test]
    fn too_many_nodes_is_reported() {
        let schema = TreeSchema::new().with_max_nodes(2);
        let diagnostics = schema.validate(&valid_tree());
        assert_eq!(codes(&diagnostics), vec![DIAGNOSTIC_TOO_MANY_NODES]);
        assert_eq!(diagnostics[0].cid, "");
        assert!(diagnostics[0]
            .detail
            .contains("4 nodes exceed the limit of 2"));
    }

    #[test]
    fn validate_or_error_names_the_first_problem() {
        let mut nodes = valid_tree();
        nodes.push(IndexNode::branch("Skills").with_description("root"));
        nodes.push(IndexNode::branch("Skills.alpha.one.deep").with_description("deep"));
        let error = TreeSchema::new().validate_or_error(&nodes).unwrap_err();
        let message = error.message();
        assert!(message.starts_with("tree_index.yaml: "));
        assert!(message.contains(DIAGNOSTIC_DUPLICATE_CID));
        assert!(message.contains("cid `Skills`"));
    }

    #[test]
    fn validate_preset_reports_cycles_and_derivation_failures() {
        let preset = TreePreset {
            nodes: vec![
                preset_branch("A", "a", vec![preset_branch("A", "b", Vec::new())]),
                PresetNode::default(),
            ],
        };
        let diagnostics = TreeSchema::new().validate_preset(&preset);
        assert!(codes(&diagnostics).contains(&DIAGNOSTIC_CYCLE));
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_CYCLE && diagnostic.cid == "A"));
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == DIAGNOSTIC_BLANK_CID && diagnostic.detail.contains("no cid")
        }));
        assert_eq!(
            TreeSchema::new()
                .validate_preset_or_error(&preset)
                .unwrap_err()
                .message()
                .split(':')
                .next()
                .unwrap(),
            "tree_preset"
        );
    }

    #[test]
    fn validate_preset_accepts_a_curated_taxonomy() {
        let preset = TreePreset::from_yaml(
            "nodes:\n  - description: Research\n    children:\n      - description: Deep Research\n",
        )
        .unwrap();
        assert_eq!(TreeSchema::new().validate_preset(&preset), Vec::new());
        assert_eq!(preset.validate(&TreeSchema::new()), Vec::new());
    }

    #[test]
    fn the_taxonomy_has_eight_roots_with_four_children_each() {
        assert_eq!(FIXED_ROOT_CATEGORIES.len(), 8);
        let mut ids: BTreeSet<&str> = BTreeSet::new();
        for category in FIXED_ROOT_CATEGORIES {
            assert_eq!(category.children.len(), 4, "{}", category.id);
            assert!(ids.insert(category.id), "{}", category.id);
            assert!(!category.is_leaf());
            assert_ne!(category.name, "");
            assert!(category.description.starts_with("Capabilities for "));
            for child in category.children {
                assert!(ids.insert(child.id), "{}", child.id);
                assert!(child.is_leaf());
                assert!(category.child(child.id).is_some());
            }
        }
        assert_eq!(ids.len(), 40);
        let product = root_category("product-dev").unwrap();
        assert_eq!(product.name, "开发产品");
        assert_eq!(product.child("code-test").unwrap().name, "Code Test");
        assert!(product.child("absent").is_none());
        assert!(root_category("absent").is_none());
    }
}
