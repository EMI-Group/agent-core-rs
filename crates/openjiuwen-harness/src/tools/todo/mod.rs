//! Port of `openjiuwen/harness/tools/todo.py`: the todo store and the four todo tools.
//!
//! [`TodoStore`] persists one JSON array per session at `<dir>/<session>/todo.json`. The tools
//! (`todo_create`, `todo_list`, `todo_get`, `todo_modify`) keep Python's names, argument shapes,
//! message text and payload envelopes. Every create/modify action holds the per-session
//! [`TodoStore::lock`] from load to save.
//!
//! This module is split by concern: `mod.rs`, `store`, `schema`, `shared`, one file per tool, `tests`.
//!
//! Deviations from Python: the session id comes from
//! [`ToolContext`](openjiuwen_foundation::tool::ToolContext) and falls back to `default`
//! (Python raises `Session ID is required`); the file lives under the workspace `todo` node;
//! expected failures (including IO errors and the validation errors Python raises) return a
//! `success=false` payload instead of an `Err`; and a save writes a temporary file that is
//! renamed over `todo.json`, so no reader sees a partial list.
//!
//! The cards do not use `support::schema_for` directly: it strips `$defs`, which would leave
//! the generated `$ref`s to [`TodoItem`](crate::schema::TodoItem),
//! [`TodoStatus`](crate::schema::TodoStatus) and [`TodoData`] dangling. They are built by
//! `card_schema`, which inlines those definitions and then applies Python's public shape
//! (`todo_modify`'s `action` enum, the per-tool item `required` lists, the `status` enum).

/// Generates the per-tool [`Tool`] glue: `$run` performs the action, `$render` renders it.
macro_rules! todo_tool {
    ($tool:ident, $name:literal, $run:ident, $render:path, $schema:path) => {
        /// A model-facing todo tool backed by a [`TodoStore`].
        pub struct $tool {
            store: Arc<TodoStore>,
            card: ToolCard,
        }

        impl $tool {
            /// Builds the tool over `store` with descriptions in `language`.
            #[must_use]
            pub fn new(store: Arc<TodoStore>, language: &str) -> Self {
                Self {
                    card: Self::card_for(language),
                    store,
                }
            }

            /// Builds the model-facing card in `language`; unknown languages fall back to `cn`.
            #[must_use]
            pub fn card_for(language: &str) -> ToolCard {
                tool_card(
                    $name,
                    descriptions::description($name, language),
                    $schema(),
                    true,
                )
            }
        }

        #[async_trait]
        impl Tool for $tool {
            fn card(&self) -> &ToolCard {
                &self.card
            }

            async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
                ctx.check_cancelled()?;
                Ok(self.$run(&inputs, ctx).await.to_json())
            }

            fn render_for_llm(&self, output: &Value) -> String {
                $render(output)
            }
        }
    };
}

mod create;
mod get;
mod list;
mod modify;
mod schema;
mod shared;
mod store;
#[cfg(test)]
mod tests;

pub use self::create::{TodoCreateArgs, TodoCreateTool};
pub use self::get::{TodoGetArgs, TodoGetTool};
pub use self::list::TodoListTool;
pub use self::modify::{TodoAction, TodoData, TodoModifyArgs, TodoModifyTool};
pub use self::store::{session_key, TodoStore};
