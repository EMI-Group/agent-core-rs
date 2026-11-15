//! The `list` operation: the entries of one or more scopes in the requested view.

use crate::discovery::directory_toolkit::render::{list_row, ListView, Row};
use crate::discovery::directory_toolkit::validate::ValidatedRequest;
use crate::discovery::directory_toolkit::view::SkillDirectoryView;
use crate::error::Result;

use super::Execution;

/// Runs the `list` operation.
///
/// The tree view ignores `recursive` and walks with
/// [`SkillDirectoryView::tree_entries`]; every other view scopes with
/// [`SkillDirectoryView::entries`]. Rows are rendered by
/// [`list_row`].
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) for an unknown scope path.
pub fn execute_list(
    view: &SkillDirectoryView,
    paths: &[String],
    request: &ValidatedRequest,
) -> Result<Execution> {
    let entries = if request.view == ListView::Tree {
        view.tree_entries(paths, request.max_depth, request.directories_only)?
    } else {
        view.entries(
            paths,
            request.recursive,
            request.max_depth,
            request.directories_only,
            request.directory_entry,
        )?
    };
    let total_count = entries.len();
    let rows: Vec<Row> = entries
        .into_iter()
        .map(|entry| list_row(&entry, request.view, view))
        .collect();
    Ok(Execution {
        rows,
        total_count,
        complete: true,
    })
}
