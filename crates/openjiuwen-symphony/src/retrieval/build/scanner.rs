//! Scanning of the items a build indexes.
//!
//! The scanning stage itself (front-matter parsing, worker ids, content hashes) is a
//! later build stage; this module currently declares only the value object the grouping
//! and tree stages consume, so those stages compile against the final shape.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::retrieval::types::RetrieverItem;

/// One scanned item, ready to be grouped and placed in the capability tree.
///
/// `worker_id` is the item's build identity (the skill directory name for a skill index)
/// and the key every later stage uses to find, replace or prune the item; `item` is the
/// retrievable pair the disclosure renders (`RetrieverItem::payload` holds the leaf `cid`
/// once the tree is built).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SourceItem {
    /// The item's build identity, unique within one build.
    pub worker_id: String,
    /// The retrievable item the tree leaf carries.
    pub item: RetrieverItem,
    /// The human-facing name, used for name-token grouping.
    pub name: String,
    /// The item's own description.
    pub description: String,
    /// The item body the catalog record compacts into its retrieval text.
    pub content: String,
    /// The on-disk path the item was scanned from; empty for a non-filesystem source.
    pub skill_path: String,
    /// The classification path the grouping stage derives branch segments from.
    pub classification: String,
    /// Free-form tags, used as the second grouping signal.
    pub tags: Vec<String>,
    /// The item type (`skill` by default).
    pub item_type: String,
    /// The SHA-256 of the item body, for incremental rebuilds.
    pub content_hash: String,
    /// Extra item metadata (routing hints, source attributes).
    pub metadata: BTreeMap<String, String>,
}
