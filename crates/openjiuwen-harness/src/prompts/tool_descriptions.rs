//! Bilingual, model-facing descriptions of the tools the harness exposes.
//!
//! Ports the description half of `openjiuwen/harness/prompts/tools/` (Python keeps the
//! tool name, description and JSON schema in one table). Here the table only owns the
//! name and the two descriptions; the JSON schema stays with the tool implementation and
//! is attached by [`build_tool_card`], so a schema is never duplicated.
//!
//! The table is the single source of truth for tool names: [`tool_names`] is asserted
//! against [`TOOL_DESCRIPTIONS`] by [`validate_all`], so a name can never exist in one
//! and not the other.

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::ToolCard;

use super::builder::language_key;

/// One entry of the tool description table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolDescription {
    /// Tool name exactly as the model sees it.
    pub name: &'static str,
    /// Chinese description.
    pub description_cn: &'static str,
    /// English description.
    pub description_en: &'static str,
}

/// Every implemented tool, in the order they are presented to the model.
pub const TOOL_DESCRIPTIONS: &[ToolDescription] = &[
    ToolDescription {
        name: "bash",
        description_cn: "执行 Shell 命令并返回输出。工作目录在命令之间保持不变，但 Shell 状态（变量、函数、alias）不保留。搜索代码请使用 grep 工具，不要直接调用 grep/rg。",
        description_en: "Executes a given bash command and returns its output. The working directory persists between commands, but shell state (variables, functions, aliases) does not. Use the grep tool for search tasks instead of invoking grep/rg directly.",
    },
    ToolDescription {
        name: "read_file",
        description_cn: "从本地文件系统读取文件。file_path 必须是绝对路径。默认从文件开头最多读取 2000 行，结果带行号返回。只能读取文件，不能读取目录；大文件请用 offset/limit 读取指定部分。",
        description_en: "Read a file from the local filesystem. file_path must be an absolute path. By default reads up to 2000 lines from the beginning and returns results with line numbers. Reads files only, not directories; for large files use offset/limit.",
    },
    ToolDescription {
        name: "write_file",
        description_cn: "向本地文件系统写入文件。如果目标文件已存在，必须先使用 read_file 读取；修改已有文件优先使用 edit_file。除非用户明确要求，不要创建文档文件（*.md）或 README 文件。",
        description_en: "Writes a file to the local filesystem. If the file exists you MUST read it with read_file first; prefer edit_file for modifications. NEVER create documentation files (*.md) or README files unless explicitly requested.",
    },
    ToolDescription {
        name: "edit_file",
        description_cn: "对文件执行精确字符串替换。编辑已有文件前必须先完整读取该文件。old_string 必须唯一匹配，否则提供更多上下文或设置 replace_all=true。",
        description_en: "Performs exact string replacements in files. Existing files must be read before editing. old_string must match exactly once; otherwise provide more context or set replace_all=true.",
    },
    ToolDescription {
        name: "glob",
        description_cn: "按 glob 模式快速查找文件，支持 **/*.js、src/**/*.ts 等模式，返回按文件名排序的路径，默认最多 100 个结果。",
        description_en: "Fast file pattern matching for any codebase size. Supports patterns like **/*.js or src/**/*.ts and returns matching paths sorted by file name, up to 100 results by default.",
    },
    ToolDescription {
        name: "list_files",
        description_cn: "列出目录内容。",
        description_en: "List directory contents.",
    },
    ToolDescription {
        name: "grep",
        description_cn: "基于 ripgrep 的强大搜索工具。搜索任务必须使用 grep，不要在 bash 中直接调用 grep 或 rg。支持正则语法、glob/type 过滤和 content/files_with_matches/count 三种输出模式。",
        description_en: "A powerful search tool built on ripgrep. ALWAYS use grep for search tasks; NEVER invoke grep or rg as a bash command. Supports regex syntax, glob/type filters and content/files_with_matches/count output modes.",
    },
    ToolDescription {
        name: "todo_create",
        description_cn: "创建当前会话的待办事项列表，用于跟踪进度、组织复杂任务，帮助用户了解整体执行情况。",
        description_en: "Create a todo list for the current session to track progress, organise complex tasks and keep the user informed.",
    },
    ToolDescription {
        name: "todo_list",
        description_cn: "检索并显示当前会话的所有待办事项。",
        description_en: "Retrieve and display all todo items for the current session.",
    },
    ToolDescription {
        name: "todo_get",
        description_cn: "根据任务 ID 获取单个任务的完整详情。",
        description_en: "Get full details of a single task by its ID.",
    },
    ToolDescription {
        name: "todo_modify",
        description_cn: "修改当前会话的待办事项：更新、删除、取消、追加、在其后插入、在其前插入。",
        description_en: "Modify todo items for the current session: update, delete, cancel, append, insert_after, insert_before.",
    },
    ToolDescription {
        name: "ask_user",
        description_cn: "向用户提问以收集信息、澄清歧义或做出决策。支持 1-4 个问题，每个问题 2-4 个选项。不要添加“其他/自定义”之类的兜底选项。",
        description_en: "Ask the user questions to gather information, clarify ambiguity or make decisions. Supports 1-4 questions with 2-4 options each. Do not add catch-all \"Other/Custom\" options.",
    },
    ToolDescription {
        name: "task_tool",
        description_cn: "启动新的子代理，自主处理复杂、多步骤任务。每种子代理类型都有特定的能力和可用工具。",
        description_en: "Launch a new subagent to handle complex, multi-step tasks autonomously. Each subagent type has specific capabilities and tools.",
    },
    ToolDescription {
        name: "subagent_spawn",
        description_cn: "为范围清晰、可独立完成的子任务创建常驻子代理并下发首轮任务，立即返回 subagent_id 与 task_id；必须在同一 turn 内调用 subagent_wait。",
        description_en: "Create a persistent subagent for a bounded, self-contained subtask and dispatch its first task; returns subagent_id and task_id immediately, so you must call subagent_wait in the same turn.",
    },
    ToolDescription {
        name: "subagent_wait",
        description_cn: "阻塞等待一个或多个 subagent_id 达到终态，返回状态、结果与超时标记。",
        description_en: "Block until the listed subagent_ids reach a final status and return statuses, results and whether it timed out.",
    },
    ToolDescription {
        name: "subagent_list",
        description_cn: "列出当前父会话下的子代理与容量占用。",
        description_en: "List the subagents of the current parent session and their capacity usage.",
    },
    ToolDescription {
        name: "subagent_send_input",
        description_cn: "向已存在的 subagent 投递后续消息，立即返回新的 task_id。",
        description_en: "Send follow-up input to an existing subagent; returns a new task_id immediately.",
    },
    ToolDescription {
        name: "subagent_close",
        description_cn: "在子代理不再需要时关闭实例并释放常驻名额。",
        description_en: "Close a subagent when it is no longer needed and release its slot.",
    },
    ToolDescription {
        name: "subagent_resume",
        description_cn: "从检查点恢复已关闭的子代理并重新占用名额，不会自动投递任务。",
        description_en: "Restore a closed subagent from its checkpoint and reclaim a slot; does not dispatch work.",
    },
    ToolDescription {
        name: "memory_search",
        description_cn: "在长期记忆中检索过往信息（决策、偏好、人物、日期、TODO 等），返回相关片段与引用线索。",
        description_en: "Search long-term memory (prior decisions, preferences, people, dates, todos) and return relevant snippets and references.",
    },
    ToolDescription {
        name: "memory_get",
        description_cn: "按行号切片读取 memory/ 下的记忆 Markdown 文件（from_line + lines）。",
        description_en: "Read a slice of a memory markdown file under memory/ using from_line + lines.",
    },
    ToolDescription {
        name: "write_memory",
        description_cn: "把记忆内容写入 memory/ 下的 Markdown 文件，支持覆盖写或追加写。",
        description_en: "Write memory content to a markdown file under memory/; supports overwrite or append.",
    },
    ToolDescription {
        name: "read_memory",
        description_cn: "按 offset/limit 读取 memory/ 下记忆文件的部分内容，用于分页阅读。",
        description_en: "Read a portion of a memory file under memory/ with offset/limit for paging.",
    },
];

/// Tool names in presentation order.
const TOOL_NAMES: [&str; 23] = [
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "list_files",
    "grep",
    "todo_create",
    "todo_list",
    "todo_get",
    "todo_modify",
    "ask_user",
    "task_tool",
    "subagent_spawn",
    "subagent_wait",
    "subagent_list",
    "subagent_send_input",
    "subagent_close",
    "subagent_resume",
    "memory_search",
    "memory_get",
    "write_memory",
    "read_memory",
];

/// Returns every implemented tool name in presentation order.
#[must_use]
pub fn tool_names() -> &'static [&'static str] {
    &TOOL_NAMES
}

/// Returns the description of `name` in `language`.
///
/// `language` is trimmed and lower-cased: `cn`, `zh`, `zh-*` and anything unrecognised
/// select the Chinese description, `en` and `en-*` the English one. An unknown tool name
/// yields `None`.
#[must_use]
pub fn description(name: &str, language: &str) -> Option<&'static str> {
    let entry = TOOL_DESCRIPTIONS.iter().find(|entry| entry.name == name)?;
    if language_key(language) == "en" {
        Some(entry.description_en)
    } else {
        Some(entry.description_cn)
    }
}

/// Checks the table for empty names, duplicate names and blank descriptions.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the first offending entry, when the table and
/// [`tool_names`] disagree, or when an entry has an empty name, a duplicate name, or a
/// blank Chinese or English description.
pub fn validate_all() -> Result<()> {
    if TOOL_DESCRIPTIONS.len() != TOOL_NAMES.len() {
        return Err(Error::validation(format!(
            "tool description table has {} entries but {} names",
            TOOL_DESCRIPTIONS.len(),
            TOOL_NAMES.len()
        )));
    }
    for (index, entry) in TOOL_DESCRIPTIONS.iter().enumerate() {
        if entry.name.trim().is_empty() {
            return Err(Error::validation(format!(
                "tool description at index {index} has an empty name"
            )));
        }
        if TOOL_DESCRIPTIONS
            .iter()
            .take(index)
            .any(|seen| seen.name == entry.name)
        {
            return Err(Error::validation(format!(
                "tool description {:?} is duplicated",
                entry.name
            )));
        }
        if TOOL_NAMES.get(index).copied() != Some(entry.name) {
            return Err(Error::validation(format!(
                "tool name list is out of sync at index {index} ({:?})",
                entry.name
            )));
        }
        if entry.description_cn.trim().is_empty() {
            return Err(Error::validation(format!(
                "tool {:?} has an empty Chinese description",
                entry.name
            )));
        }
        if entry.description_en.trim().is_empty() {
            return Err(Error::validation(format!(
                "tool {:?} has an empty English description",
                entry.name
            )));
        }
    }
    Ok(())
}

/// Builds the [`ToolCard`] of `name` with a caller-supplied JSON schema.
///
/// `tool_id` identifies the card, `input_params` is the tool's JSON Schema, and every
/// policy flag keeps its [`ToolCard::default`] value.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `name` is not in [`TOOL_DESCRIPTIONS`].
pub fn build_tool_card(
    name: &str,
    tool_id: &str,
    language: &str,
    input_params: serde_json::Value,
) -> Result<ToolCard> {
    let Some(description) = description(name, language) else {
        return Err(Error::validation(format!(
            "unknown tool description: {name:?}"
        )));
    };
    Ok(ToolCard {
        id: tool_id.to_string(),
        name: name.to_string(),
        description: description.to_string(),
        input_params,
        ..ToolCard::default()
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeSet;

    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::{build_tool_card, description, tool_names, validate_all, TOOL_DESCRIPTIONS};

    #[test]
    fn there_are_exactly_23_unique_tools() {
        let names = tool_names();
        assert_eq!(names.len(), 23);
        let unique: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), 23);
        assert_eq!(TOOL_DESCRIPTIONS.len(), 23);
        assert_eq!(
            names,
            [
                "bash",
                "read_file",
                "write_file",
                "edit_file",
                "glob",
                "list_files",
                "grep",
                "todo_create",
                "todo_list",
                "todo_get",
                "todo_modify",
                "ask_user",
                "task_tool",
                "subagent_spawn",
                "subagent_wait",
                "subagent_list",
                "subagent_send_input",
                "subagent_close",
                "subagent_resume",
                "memory_search",
                "memory_get",
                "write_memory",
                "read_memory",
            ]
        );
    }

    #[test]
    fn every_tool_has_both_descriptions() {
        for name in tool_names() {
            let cn = description(name, "cn").expect("cn description");
            let en = description(name, "en").expect("en description");
            assert!(!cn.trim().is_empty(), "{name} cn");
            assert!(!en.trim().is_empty(), "{name} en");
            assert_ne!(cn, en, "{name} descriptions differ");
        }
    }

    #[test]
    fn language_resolution_maps_to_chinese_by_default() {
        assert_eq!(description("bash", "en"), description("bash", "EN"));
        assert_eq!(description("bash", "en-US"), description("bash", "en"));
        assert_eq!(description("bash", "cn"), description("bash", "zh"));
        assert_eq!(description("bash", "zh-Hant"), description("bash", "cn"));
        assert_eq!(description("bash", "klingon"), description("bash", "cn"));
        assert_eq!(description("bash", "  En  "), description("bash", "en"));
        assert!(description("bash", "cn").unwrap().contains("Shell"));
        assert!(description("bash", "en").unwrap().starts_with("Executes"));
    }

    #[test]
    fn unknown_tool_yields_none() {
        assert!(description("not_a_tool", "cn").is_none());
        assert!(description("", "en").is_none());
        assert!(description("Bash", "en").is_none());
    }

    #[test]
    fn validate_all_passes() {
        validate_all().expect("table is consistent");
    }

    #[test]
    fn build_tool_card_attaches_name_description_and_schema() {
        let schema = json!({"type": "object", "properties": {"command": {"type": "string"}}});
        let card = build_tool_card("bash", "tool-bash", "en", schema.clone()).expect("known tool");
        assert_eq!(card.id, "tool-bash");
        assert_eq!(card.name, "bash");
        assert_eq!(card.description, description("bash", "en").unwrap());
        assert_eq!(card.input_params, schema);

        let cn_card = build_tool_card("bash", "tool-bash", "cn", json!({})).expect("known tool");
        assert_eq!(cn_card.description, description("bash", "cn").unwrap());
    }

    #[test]
    fn build_tool_card_rejects_unknown_tools() {
        let error = build_tool_card("nope", "id", "en", json!({})).expect_err("unknown tool");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("nope"));
    }
}
