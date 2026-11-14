//! The team-memory handle [`MemberHarnessContext`] carries into a member's harness.
//!
//! Python parity source: `openjiuwen/agent_teams/memory/toolkit.py`, where a member's
//! `MemberMemoryToolkit` is registered on its own agent. This crate keeps no tool logic of its own:
//! [`crate::memory::team_memory_tools`] already projects the engine toolkit onto the crate-wide
//! [`Tool`] seam, so this module only adapts that projection for the harness context and answers
//! the identity question the context's derived `Eq` needs.
//!
//! [`MemberHarnessContext`]: super::MemberHarnessContext
use std::sync::Arc;

use openjiuwen_foundation::tool::Tool;
use openjiuwen_memory::team::MemberMemoryToolkit;

/// The four model-facing team-memory tools of `toolkit`, in the toolkit's own order.
///
/// `toolkit` is borrowed so the context keeps the single owning `Arc`; each tool holds its own
/// clone of the handle.
pub(super) fn memory_tools(toolkit: &Arc<MemberMemoryToolkit>) -> Vec<Arc<dyn Tool>> {
    crate::memory::team_memory_tools(Arc::clone(toolkit))
}

/// Whether two optional toolkits denote the same attached handle.
///
/// `None` matches only `None`, and two attached toolkits match only when they are the same `Arc`
/// ([`Arc::ptr_eq`]); a toolkit is a live engine handle, so value equality is neither available nor
/// meaningful. This is the rule behind `MemberHarnessContext`'s hand-written `PartialEq`.
pub(super) fn same_toolkit(
    left: Option<&Arc<MemberMemoryToolkit>>,
    right: Option<&Arc<MemberMemoryToolkit>>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        _ => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_memory::team::{
        MemberMemoryContext, TeamMemory, TOOL_FORGET, TOOL_REMEMBER, TOOL_SEARCH, TOOL_STATS,
    };

    use super::*;

    /// One toolkit of `dev-1` over a fresh in-memory team memory.
    fn toolkit() -> Arc<MemberMemoryToolkit> {
        Arc::new(MemberMemoryToolkit::new(
            Arc::new(TeamMemory::in_memory()),
            MemberMemoryContext::new("team-1").with_member("dev-1"),
        ))
    }

    #[test]
    fn the_projection_mounts_the_four_engine_tools_in_order() {
        let toolkit = toolkit();
        let names: Vec<String> = memory_tools(&toolkit)
            .iter()
            .map(|tool| tool.card().name.clone())
            .collect();
        assert_eq!(
            names,
            vec![
                TOOL_REMEMBER.to_owned(),
                TOOL_SEARCH.to_owned(),
                TOOL_FORGET.to_owned(),
                TOOL_STATS.to_owned()
            ]
        );
        assert_eq!(names, toolkit.tool_names());
    }

    #[test]
    fn an_absent_toolkit_matches_only_an_absent_toolkit() {
        assert!(same_toolkit(None, None));
        let one = toolkit();
        assert!(!same_toolkit(Some(&one), None));
        assert!(!same_toolkit(None, Some(&one)));
    }

    #[test]
    fn a_toolkit_matches_itself_and_no_fresh_handle() {
        let toolkit = toolkit();
        let clone = Arc::clone(&toolkit);
        assert!(same_toolkit(Some(&toolkit), Some(&clone)));
        let fresh = self::toolkit();
        assert!(!same_toolkit(Some(&toolkit), Some(&fresh)));
    }
}
