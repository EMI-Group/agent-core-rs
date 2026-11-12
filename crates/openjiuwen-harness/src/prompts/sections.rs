//! Factories for every standard system-prompt section.
//!
//! Ports the `sections/` package of `openjiuwen/harness/prompts/`: each factory returns
//! a [`PromptSection`] whose content is already bilingual (`cn` and `en`), so the
//! builder can render the whole prompt in either language without revisiting the data.
//!
//! The factories are pure: tool cards, skills, todos and workspace roots arrive as
//! values, and nothing here reads the environment, the clock or the filesystem. Text is
//! parity-critical — the model-facing wording (headings, tool names, promise tags) is
//! kept verbatim from the Python SDK.

use std::collections::BTreeMap;

use openjiuwen_foundation::ToolCard;

use crate::schema::{status_icon, AgentMode, TodoItem};
use crate::skills::Skill;
use crate::workspace::Workspace;

use super::builder::{PromptCategory, PromptSection, SectionName};

/// The completion promise used when the caller configures none.
pub const DEFAULT_COMPLETION_PROMISE: &str = "TASK_COMPLETE";

/// Wraps a completion promise in the marker tag the model emits to signal completion.
#[must_use]
pub fn promise_tag(promise: &str) -> String {
    format!("<promise>{promise}</promise>")
}

/// Builds a bilingual section with the given category.
fn section(
    name: &str,
    priority: i32,
    category: PromptCategory,
    cn: impl Into<String>,
    en: impl Into<String>,
) -> PromptSection {
    PromptSection::bilingual(name, cn, en, priority).with_category(category)
}

/// Builds a section whose content maps are empty (dropped by the builder).
fn empty_section(name: &str, priority: i32, category: PromptCategory) -> PromptSection {
    PromptSection::new(name, BTreeMap::new(), priority).with_category(category)
}

/// Renders `- **name**: description` bullets, or `fallback` for an empty list.
fn bullet_list<'a>(items: impl IntoIterator<Item = (&'a str, &'a str)>, fallback: &str) -> String {
    let bullets: Vec<String> = items
        .into_iter()
        .map(|(name, description)| format!("- **{name}**: {description}"))
        .collect();
    if bullets.is_empty() {
        fallback.to_string()
    } else {
        bullets.join("\n")
    }
}

const IDENTITY_CN: &str = concat!(
    "你是一个通用 AI 助手。请根据用户的需求，合理使用可用工具完成任务。\n",
    "在执行过程中保持目标聚焦，遇到问题时尝试不同策略。\n",
    "重要提示：如果某个工具返回无结果、空输出或错误，切勿使用相同参数重复调用同一工具。",
    "请根据对话上下文中已有的工具调用记录，尝试不同的工具或方法。",
);

const IDENTITY_EN: &str = concat!(
    "You are a general-purpose AI assistant. Use available tools to complete tasks ",
    "based on user needs.\n",
    "Stay focused on the goal during execution and try different strategies when ",
    "encountering problems.\n",
    "Important: if a tool returns no results, empty output, or an error, do NOT call the ",
    "same tool with identical arguments again. Refer to the tool-call history already ",
    "visible in the conversation context and try a different tool or approach.",
);

const SAFETY_CN: &str = concat!(
    "# 安全原则\n\n",
    "- 永远不要泄露隐私数据。\n",
    "- 修改或删除重要文件、执行影响系统的命令，以及涉及金钱、账号或敏感信息的操作前，",
    "先请示用户。\n",
    "- 违法、有害或侵犯他人权益的请求不予处理。\n",
    "- 不虚构工具结果、文件内容、执行状态或已经完成的操作。\n",
    "- 优先使用只读操作；破坏性命令执行前先向用户确认。\n",
    "- 不要把密钥、凭证或令牌写入文件、日志或输出中。\n",
    "- 不确定时先向用户询问，而不是猜测用户的意图。\n",
    "- 在授权范围内工作：只访问当前任务所需的文件和目录。",
);

const SAFETY_EN: &str = concat!(
    "# Safety\n\n",
    "- Never leak private data.\n",
    "- Ask the user before modifying or deleting important files, running commands that ",
    "affect the system, or performing operations involving money, accounts or sensitive ",
    "information.\n",
    "- Do not handle requests that are illegal, harmful or violate the rights of others.\n",
    "- Do not fabricate tool results, file contents, execution status or completed actions.\n",
    "- Prefer read-only operations; confirm with the user before running destructive ",
    "commands.\n",
    "- Never write secrets, credentials or tokens into files, logs or output.\n",
    "- When unsure, ask the user instead of guessing their intent.\n",
    "- Work within your authorisation: only access the files and directories the current ",
    "task needs.",
);

const TOOL_RULES_CN: &str = concat!(
    "使用原则：\n",
    "- 搜索代码或文件内容时使用 `grep` 工具，不要在 bash 中直接调用 `grep`/`rg`。\n",
    "- 修改已有文件前，先用 `read_file` 读取它。\n",
    "- 修改已有文件优先使用 `edit_file`；`write_file` 适合新建文件或整体重写。\n",
    "- 除非用户明确要求，不要创建文档文件（`*.md`）或 README 文件。\n\n",
    "可用工具：",
);

const TOOL_RULES_EN: &str = concat!(
    "Usage rules:\n",
    "- Use the `grep` tool for code and content search instead of invoking `grep`/`rg` ",
    "in bash.\n",
    "- Read a file with `read_file` before you modify it.\n",
    "- Prefer `edit_file` for existing files; `write_file` is for new files or full ",
    "rewrites.\n",
    "- Never create documentation files (`*.md`) or README files unless the user ",
    "explicitly asks.\n\n",
    "Available tools:",
);

const SKILL_RULES_CN: &str = concat!(
    "使用方式：先用 `read_file` 读取技能目录下的 `SKILL.md`，再严格按其中的说明执行。\n\n",
    "可用技能：",
);

const SKILL_RULES_EN: &str = concat!(
    "How to use a skill: read its `SKILL.md` with `read_file` first, then follow its ",
    "instructions exactly.\n\n",
    "Available skills:",
);

const MEMORY_GUIDE_CN: &str = concat!(
    "## 记忆\n\n",
    "- `MEMORY.md` 与 `memory/daily_memory/` 保存需要长期保留的笔记：决策、偏好、约定和",
    "未完成事项。\n",
    "- 用 `memory_search` 检索过往信息，用 `memory_get`/`read_memory` 读取片段，",
    "用 `write_memory` 写入。\n",
    "- 只记录有长期价值的信息；永远不要把密钥、凭证或隐私数据写入记忆。",
);

const MEMORY_GUIDE_EN: &str = concat!(
    "## Memory\n\n",
    "- `MEMORY.md` and `memory/daily_memory/` hold notes worth keeping: decisions, ",
    "preferences, conventions and unfinished work.\n",
    "- Use `memory_search` to look things up, `memory_get`/`read_memory` to read a slice, ",
    "and `write_memory` to record new notes.\n",
    "- Record only durable information; never write secrets, credentials or private data ",
    "into memory.",
);

const WORKSPACE_TABLE_CN: &str = concat!(
    "| 文件 | 用途 | 操作工具 |\n",
    "| --- | --- | --- |\n",
    "| `AGENT.md` | 智能体指令 | `read_file` |\n",
    "| `IDENTITY.md` | 智能体身份设定 | `read_file`, `edit_file` |\n",
    "| `USER.md` | 用户画像 | `read_file`, `edit_file` |\n",
    "| `memory/MEMORY.md` | 长期记忆 | `read_memory`, `write_memory` |\n",
    "| `memory/daily_memory/YYYY-MM-DD.md` | 每日记忆 | `write_memory` |",
);

const WORKSPACE_TABLE_EN: &str = concat!(
    "| File | Purpose | Tools |\n",
    "| --- | --- | --- |\n",
    "| `AGENT.md` | Agent instructions | `read_file` |\n",
    "| `IDENTITY.md` | Agent identity | `read_file`, `edit_file` |\n",
    "| `USER.md` | User profile | `read_file`, `edit_file` |\n",
    "| `memory/MEMORY.md` | Long-term memory | `read_memory`, `write_memory` |\n",
    "| `memory/daily_memory/YYYY-MM-DD.md` | Daily memory | `write_memory` |",
);

const TASK_LOOP_CN: &str = concat!(
    "# 子智能体使用规则\n\n",
    "## 应当使用\n",
    "- 任务包含多个步骤，或可以拆成互不依赖、可并行推进的子任务。\n",
    "- 子任务需要独立的上下文窗口、反复试错，或大量文件读写。\n\n",
    "## 不应使用\n",
    "- 一次查找或一次读取就能完成的简单问题。\n",
    "- 必须由你自己完成、需要与你当前上下文紧密衔接的工作。\n\n",
    "## 使用原则\n",
    "- 在 `task_description` 中写清目标、约束和期望的输出形式。\n",
    "- 依赖返回结果前先验证：检查结论是否完整、是否与代码实际一致。",
);

const TASK_LOOP_EN: &str = concat!(
    "# Subagent Usage Rules\n\n",
    "## When to Use\n",
    "- The task has multiple steps, or decomposes into independent sub-tasks that can run ",
    "in parallel.\n",
    "- The sub-task needs its own context window, repeated trial and error, or heavy file ",
    "reading.\n\n",
    "## When Not to Use\n",
    "- A simple question answered by one lookup or one file read.\n",
    "- Work you must do yourself because it has to stay tightly coupled to your current ",
    "context.\n\n",
    "## Usage Principles\n",
    "- State the goal, the constraints and the expected output shape in ",
    "`task_description`.\n",
    "- Verify the returned result before relying on it: check that it is complete and ",
    "matches the actual code.",
);

const SUBAGENT_CN: &str = concat!(
    "## 常驻子代理管理\n\n",
    "- `subagent_spawn` 会立即返回 `subagent_id` 与 `task_id`，因此必须在同一回合内调用 ",
    "`subagent_wait` 等待结果。\n",
    "- 用 `subagent_list` 查看当前会话的子代理与容量占用。\n",
    "- 需要补充信息时用 `subagent_send_input` 投递后续消息。\n",
    "- 不再需要时用 `subagent_close` 关闭并释放名额。\n",
    "- `subagent_resume` 可以从检查点恢复已关闭的子代理。",
);

const SUBAGENT_EN: &str = concat!(
    "## Persistent Subagents\n\n",
    "- `subagent_spawn` returns immediately with a `subagent_id` and a `task_id`, so you ",
    "must call `subagent_wait` in the same turn to collect the result.\n",
    "- Use `subagent_list` to see the subagents of this session and the capacity in use.\n",
    "- Send additional work with `subagent_send_input`.\n",
    "- Call `subagent_close` to free the slot once a subagent is no longer needed.\n",
    "- `subagent_resume` restores a closed subagent from its checkpoint.",
);

const PLAN_CN: &str = concat!(
    "## Plan 模式\n\n",
    "Plan 模式已激活。用户希望你先制定计划，不要求执行——你不得进行任何修改（plan 文件",
    "除外）、不得运行非只读工具、不得对系统做出任何更改。此约束优先于其他指令。\n\n",
    "### Phase 1: 初始理解\n",
    "- 阅读用户需求和相关文件，弄清目标、约束和验收标准。\n",
    "- 只做只读探索：`read_file`、`grep`、`glob`、`list_files`。\n\n",
    "### Phase 2: 设计\n",
    "- 提出实现方案，明确要改动的文件、模块边界和风险点。\n",
    "- 列出需要用户确认的开放问题。\n\n",
    "### Phase 3: 复核\n",
    "- 检查方案是否覆盖全部需求，是否有更简单或更稳妥的做法。\n",
    "- 指出未覆盖的边界情况和依赖关系。\n\n",
    "### Phase 4: 编写最终计划\n",
    "- 把计划写入 plan 文件：按顺序排列的步骤、涉及的文件、每一步的验证方式。\n",
    "- 计划必须可以端到端验证：写清要运行什么命令、检查什么输出。\n\n",
    "### Phase 5: 结束规划阶段\n",
    "- 用 `exit_plan_mode` 提交计划并等待用户批准；必要时先用 `ask_user` 澄清。\n\n",
    "### 回合结束规则\n",
    "- 一个回合只能通过 `ask_user` 或 `exit_plan_mode` 结束。\n",
    "- 需要用户批准计划时使用 `exit_plan_mode`。\n",
    "- 只允许用 `write_file` / `edit_file` 写入 plan 文件，不得写入其他文件。",
);

const PLAN_EN: &str = concat!(
    "## Plan Mode\n\n",
    "Plan mode is active. The user wants you to only plan. You must not make any ",
    "modifications (except to the plan file), must not run non-read-only tools, and must ",
    "not make any changes to the system. This constraint takes priority over any other ",
    "instructions you receive.\n\n",
    "### Phase 1: Initial Understanding\n",
    "- Read the user request and the relevant files; establish the goal, constraints and ",
    "acceptance criteria.\n",
    "- Explore read-only only: `read_file`, `grep`, `glob`, `list_files`.\n\n",
    "### Phase 2: Design\n",
    "- Propose an implementation approach naming the files to change, the module ",
    "boundaries and the risks.\n",
    "- List the open questions the user has to resolve.\n\n",
    "### Phase 3: Review\n",
    "- Check that the plan covers every requirement and that no simpler or safer ",
    "alternative exists.\n",
    "- Call out uncovered edge cases and dependencies.\n\n",
    "### Phase 4: Write Final Plan\n",
    "- Write the plan into the plan file: ordered steps, the files each step touches and ",
    "how each step is verified.\n",
    "- The plan must be verifiable end to end: name the commands to run and the output to ",
    "check.\n\n",
    "### Phase 5: End Planning Phase\n",
    "- Submit the plan with `exit_plan_mode` and wait for approval; use `ask_user` first ",
    "when something needs clarifying.\n\n",
    "### Turn-ending Rules\n",
    "- A turn may end only via `ask_user` or `exit_plan_mode`.\n",
    "- Ask for plan approval with `exit_plan_mode`.\n",
    "- Only the plan file may be written, using `write_file` / `edit_file`.",
);

const TODO_GUIDE_CN: &str = concat!(
    "## 任务计划\n\n",
    "使用 `todo_create`、`todo_modify`、`todo_list`、`todo_get` 拆解并跟踪多步骤工作：\n",
    "- 只对确实需要多步跟踪的任务建立列表，不要过度拆分。\n",
    "- 开始某项任务前把它标记为 `in_progress`，完成后标记为 `completed`，放弃时标记为 ",
    "`cancelled`。",
);

const TODO_GUIDE_EN: &str = concat!(
    "## Task Plan\n\n",
    "Use `todo_create`, `todo_modify`, `todo_list` and `todo_get` to break down and track ",
    "multi-step work:\n",
    "- Only build a list for work that really needs multi-step tracking; do not ",
    "over-decompose.\n",
    "- Mark an item `in_progress` before starting it, `completed` when done and ",
    "`cancelled` when abandoned.",
);

const COMPLETION_CN: &str = concat!(
    "\n\n## 完成信号\n",
    "任务完全完成后，在回复的最后一行输出 {tag}。\n",
    "在确认任务完成前，不要输出此标签。",
);

const COMPLETION_EN: &str = concat!(
    "\n\n## Completion Signal\n",
    "When the task is fully completed, output {tag} as the final line of your response.\n",
    "Do not output this tag until you are confident the task is complete.",
);

/// Agent identity and the general working rules.
#[must_use]
pub fn identity() -> PromptSection {
    section(
        SectionName::IDENTITY,
        10,
        PromptCategory::SystemPrompt,
        IDENTITY_CN,
        IDENTITY_EN,
    )
}

/// Safety and authorisation rules.
#[must_use]
pub fn safety() -> PromptSection {
    section(
        SectionName::SAFETY,
        20,
        PromptCategory::SystemPrompt,
        SAFETY_CN,
        SAFETY_EN,
    )
}

/// Tool usage rules followed by the model-facing tool list.
///
/// An empty `tools` list still keeps the usage rules; the list itself renders `(none)`.
#[must_use]
pub fn tools_section(tools: &[ToolCard]) -> PromptSection {
    let pairs = tools
        .iter()
        .map(|tool| (tool.name.as_str(), tool.description.as_str()));
    let cn = format!(
        "## 可用工具\n\n{TOOL_RULES_CN}\n{}",
        bullet_list(pairs.clone(), "(none)")
    );
    let en = format!(
        "## Available Tools\n\n{TOOL_RULES_EN}\n{}",
        bullet_list(pairs, "(none)")
    );
    section(SectionName::TOOLS, 30, PromptCategory::Tools, cn, en)
}

/// The skill list and the rule for loading a skill's `SKILL.md`.
#[must_use]
pub fn skills_section(skills: &[Skill]) -> PromptSection {
    let pairs = skills
        .iter()
        .map(|skill| (skill.name.as_str(), skill.description.as_str()));
    let cn = format!(
        "## 可用技能\n\n{SKILL_RULES_CN}\n{}",
        bullet_list(pairs.clone(), "当前没有可用技能。")
    );
    let en = format!(
        "## Available Skills\n\n{SKILL_RULES_EN}\n{}",
        bullet_list(pairs, "No skills are available.")
    );
    section(SectionName::SKILLS, 40, PromptCategory::Skills, cn, en)
}

/// Guidance on the long-term memory files and the memory tools.
#[must_use]
pub fn memory_section() -> PromptSection {
    section(
        SectionName::MEMORY,
        50,
        PromptCategory::Memory,
        MEMORY_GUIDE_CN,
        MEMORY_GUIDE_EN,
    )
}

/// The working directory and the well-known files inside it.
#[must_use]
pub fn workspace_section(workspace: &Workspace) -> PromptSection {
    let root = workspace.root_path.display();
    let cn = format!(
        "# 工作空间\n\n你的工作目录是：`{root}`\n\n## 工作目录下重要文件\n\n{WORKSPACE_TABLE_CN}"
    );
    let en = format!(
        "# Workspace\n\nYour working directory is: `{root}`\n\n## Important Files in Working Directory\n\n{WORKSPACE_TABLE_EN}"
    );
    section(SectionName::WORKSPACE, 70, PromptCategory::Memory, cn, en)
}

/// Project files already loaded into the prompt, as `(file name, file content)` pairs.
///
/// Entries whose content is blank are skipped; when every entry is skipped the section
/// is empty and the builder drops it.
#[must_use]
pub fn context_section(files: &[(String, String)]) -> PromptSection {
    let mut cn_blocks: Vec<String> = Vec::new();
    let mut en_blocks: Vec<String> = Vec::new();
    for (name, content) in files {
        if content.trim().is_empty() {
            continue;
        }
        let body = content.trim_end();
        cn_blocks.push(format!("## {name}\n\n{body}"));
        en_blocks.push(format!("## {name}\n\n{body}"));
    }
    if cn_blocks.is_empty() {
        return empty_section(SectionName::CONTEXT, 100, PromptCategory::Memory);
    }
    let cn = format!(
        "# 项目上下文\n\n以下文件已加载到上下文中，无需再次读取。\n\n{}",
        cn_blocks.join("\n\n")
    );
    let en = format!(
        "# Project Context\n\nThe following files are already loaded into context, so you do not need to read them again.\n\n{}",
        en_blocks.join("\n\n")
    );
    section(SectionName::CONTEXT, 100, PromptCategory::Memory, cn, en)
}

/// Rules for delegating work to ephemeral subagents via `task_tool`.
#[must_use]
pub fn task_loop_section() -> PromptSection {
    section(
        SectionName::TASK_LOOP,
        80,
        PromptCategory::SystemPrompt,
        TASK_LOOP_CN,
        TASK_LOOP_EN,
    )
}

/// Rules for managing persistent subagents.
#[must_use]
pub fn subagent_section() -> PromptSection {
    section(
        SectionName::SUBAGENT_TOOLS,
        81,
        PromptCategory::SystemPrompt,
        SUBAGENT_CN,
        SUBAGENT_EN,
    )
}

/// The completion-signal instructions, using [`DEFAULT_COMPLETION_PROMISE`].
#[must_use]
pub fn completion_signal() -> PromptSection {
    completion_signal_with(DEFAULT_COMPLETION_PROMISE)
}

/// The completion-signal instructions, using `promise` as the marker.
#[must_use]
pub fn completion_signal_with(promise: &str) -> PromptSection {
    let tag = promise_tag(promise);
    let cn = COMPLETION_CN.replace("{tag}", &tag);
    let en = COMPLETION_EN.replace("{tag}", &tag);
    section(
        SectionName::COMPLETION_SIGNAL,
        83,
        PromptCategory::SystemPrompt,
        cn,
        en,
    )
}

/// Mode-specific instructions.
///
/// [`AgentMode::Normal`] emits nothing (the section is empty and gets dropped);
/// [`AgentMode::Plan`] emits the planning workflow and its turn-ending rules.
#[must_use]
pub fn mode_instructions(mode: AgentMode) -> PromptSection {
    match mode {
        AgentMode::Normal => empty_section(
            SectionName::MODE_INSTRUCTIONS,
            84,
            PromptCategory::SystemPrompt,
        ),
        AgentMode::Plan => section(
            SectionName::MODE_INSTRUCTIONS,
            84,
            PromptCategory::SystemPrompt,
            PLAN_CN,
            PLAN_EN,
        ),
    }
}

/// The todo guidance plus, when non-empty, the current plan.
#[must_use]
pub fn todo_section(todos: &[TodoItem]) -> PromptSection {
    let mut cn = TODO_GUIDE_CN.to_string();
    let mut en = TODO_GUIDE_EN.to_string();
    if !todos.is_empty() {
        cn.push_str("\n\n### 当前计划\n\n");
        cn.push_str(&render_todos(todos));
        en.push_str("\n\n### Current Plan\n\n");
        en.push_str(&render_todos(todos));
    }
    section(SectionName::TODO, 92, PromptCategory::SystemPrompt, cn, en)
}

/// The runtime environment: working directory, model, git branch and date.
///
/// An absent or blank `git_branch` renders `N/A`.
#[must_use]
pub fn environment_section(
    cwd: &str,
    model: &str,
    date: &str,
    git_branch: Option<&str>,
) -> PromptSection {
    let branch = git_branch
        .filter(|branch| !branch.trim().is_empty())
        .unwrap_or("N/A");
    let cn = format!(
        "## 运行环境\n\n- 工作目录：{cwd}\n- 模型：{model}\n- Git 分支：{branch}\n- 日期：{date}"
    );
    let en = format!(
        "## Environment\n\n- CWD: {cwd}\n- Model: {model}\n- Git Branch: {branch}\n- Date: {date}"
    );
    section(
        SectionName::ENVIRONMENT,
        90,
        PromptCategory::SystemPrompt,
        cn,
        en,
    )
}

/// Renders one `- {icon} {id}: {content}` line per todo item.
fn render_todos(todos: &[TodoItem]) -> String {
    todos
        .iter()
        .map(|todo| {
            format!(
                "- {} {}: {}",
                status_icon(todo.status),
                todo.id,
                todo.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_foundation::ToolCard;

    use crate::schema::{AgentMode, TodoItem, TodoStatus};
    use crate::skills::Skill;
    use crate::workspace::Workspace;

    use super::{
        completion_signal, completion_signal_with, context_section, environment_section, identity,
        memory_section, mode_instructions, promise_tag, safety, skills_section, subagent_section,
        task_loop_section, todo_section, tools_section, workspace_section,
        DEFAULT_COMPLETION_PROMISE,
    };
    use crate::prompts::PromptCategory;

    fn card(name: &str, description: &str) -> ToolCard {
        ToolCard {
            name: name.to_string(),
            description: description.to_string(),
            ..ToolCard::default()
        }
    }

    fn todo(id: &str, content: &str, status: TodoStatus) -> TodoItem {
        TodoItem {
            id: id.to_string(),
            content: content.to_string(),
            status,
            ..TodoItem::default()
        }
    }

    #[test]
    fn identity_has_no_heading_and_both_languages() {
        let section = identity();
        assert_eq!(section.name, "identity");
        assert_eq!(section.priority, 10);
        assert_eq!(section.category, Some(PromptCategory::SystemPrompt));
        let cn = section.render("cn");
        let en = section.render("en");
        assert!(cn.starts_with("你是一个通用 AI 助手。"), "{cn}");
        assert!(
            en.starts_with("You are a general-purpose AI assistant."),
            "{en}"
        );
        assert!(cn.contains("切勿使用相同参数重复调用同一工具"));
        assert!(en.contains("do NOT call the same tool with identical arguments again"));
        assert!(!cn.contains('#'));
    }

    #[test]
    fn safety_has_eight_bullets_in_both_languages() {
        let section = safety();
        assert_eq!(section.priority, 20);
        for language in ["cn", "en"] {
            let text = section.render(language);
            let bullets = text.lines().filter(|line| line.starts_with("- ")).count();
            assert_eq!(bullets, 8, "{language}: {text}");
        }
        let cn = section.render("cn");
        assert!(cn.starts_with("# 安全原则\n"));
        assert!(cn.contains("- 永远不要泄露隐私数据。"));
        assert!(cn.contains("先请示用户。"));
        assert!(cn.contains("违法、有害或侵犯他人权益的请求不予处理。"));
        assert!(cn.contains("不虚构工具结果"));
        let en = section.render("en");
        assert!(en.starts_with("# Safety\n"));
        assert!(en.contains("Never leak private data."));
    }

    #[test]
    fn tools_section_lists_cards_under_rules() {
        let cards = [
            card("bash", "Run commands"),
            card("read_file", "Read a file"),
        ];
        let section = tools_section(&cards);
        assert_eq!(section.name, "tools");
        assert_eq!(section.priority, 30);
        assert_eq!(section.category, Some(PromptCategory::Tools));
        let cn = section.render("cn");
        assert!(cn.starts_with("## 可用工具\n\n使用原则："));
        assert!(cn.contains("不要在 bash 中直接调用 `grep`/`rg`"));
        assert!(cn.contains("\n- **bash**: Run commands"));
        assert!(cn.contains("\n- **read_file**: Read a file"));
        let en = section.render("en");
        assert!(en.starts_with("## Available Tools\n\nUsage rules:"));
        assert!(en.contains("\n- **bash**: Run commands"));
    }

    #[test]
    fn tools_section_without_cards_keeps_the_rules() {
        let section = tools_section(&[]);
        let cn = section.render("cn");
        let en = section.render("en");
        assert!(cn.contains("使用原则："));
        assert!(cn.ends_with("(none)"), "{cn}");
        assert!(en.contains("Usage rules:"));
        assert!(en.ends_with("(none)"), "{en}");
        assert!(!section.is_empty("cn"));
    }

    #[test]
    fn skills_section_lists_skills_and_handles_empty() {
        let skills = [Skill::new("deploy", "Deploy the service", "/skills/deploy")];
        let section = skills_section(&skills);
        assert_eq!(section.name, "skills");
        assert_eq!(section.priority, 40);
        assert_eq!(section.category, Some(PromptCategory::Skills));
        let cn = section.render("cn");
        assert!(cn.starts_with("## 可用技能"));
        assert!(cn.contains("`SKILL.md`"));
        assert!(cn.contains("- **deploy**: Deploy the service"));

        let empty = skills_section(&[]);
        assert!(empty.render("cn").ends_with("当前没有可用技能。"));
        assert!(empty.render("en").ends_with("No skills are available."));
    }

    #[test]
    fn memory_section_mentions_all_memory_tools() {
        let section = memory_section();
        assert_eq!(section.name, "memory");
        assert_eq!(section.priority, 50);
        assert_eq!(section.category, Some(PromptCategory::Memory));
        for tool in ["memory_search", "memory_get", "write_memory", "read_memory"] {
            assert!(section.render("cn").contains(tool), "cn mentions {tool}");
            assert!(section.render("en").contains(tool), "en mentions {tool}");
        }
        assert!(section.render("cn").contains("MEMORY.md"));
        assert!(section.render("cn").contains("memory/daily_memory/"));
        assert!(section.render("en").contains("never write secrets"));
    }

    #[test]
    fn workspace_section_renders_the_root_and_the_file_table() {
        let workspace = Workspace::new("/tmp/ws");
        let section = workspace_section(&workspace);
        assert_eq!(section.name, "workspace");
        assert_eq!(section.priority, 70);
        assert_eq!(section.category, Some(PromptCategory::Memory));

        let cn = section.render("cn");
        assert!(
            cn.starts_with("# 工作空间\n\n你的工作目录是：`/tmp/ws`"),
            "{cn}"
        );
        assert!(cn.contains("## 工作目录下重要文件"));
        assert!(cn.contains("| 文件 | 用途 | 操作工具 |"));
        for file in [
            "AGENT.md",
            "IDENTITY.md",
            "USER.md",
            "memory/MEMORY.md",
            "memory/daily_memory/YYYY-MM-DD.md",
        ] {
            assert!(cn.contains(file), "cn table lists {file}");
        }

        let en = section.render("en");
        assert!(
            en.starts_with("# Workspace\n\nYour working directory is: `/tmp/ws`"),
            "{en}"
        );
        assert!(en.contains("| File | Purpose | Tools |"));
        assert!(en.contains("| `AGENT.md` | Agent instructions | `read_file` |"));
    }

    #[test]
    fn context_section_lists_loaded_files_and_skips_blank_content() {
        let files = vec![
            ("AGENT.md".to_string(), "Do the thing.\n".to_string()),
            ("EMPTY.md".to_string(), "   \n".to_string()),
            ("README.md".to_string(), "readme body".to_string()),
        ];
        let section = context_section(&files);
        assert_eq!(section.name, "context");
        assert_eq!(section.priority, 100);
        assert_eq!(section.category, Some(PromptCategory::Memory));
        let cn = section.render("cn");
        assert!(cn.starts_with("# 项目上下文\n\n以下文件已加载到上下文中，无需再次读取。"));
        assert!(cn.contains("## AGENT.md\n\nDo the thing."));
        assert!(cn.contains("## README.md\n\nreadme body"));
        assert!(!cn.contains("EMPTY.md"));
        let en = section.render("en");
        assert!(en.starts_with("# Project Context"));
        assert!(en.contains("## AGENT.md"));

        let none = context_section(&[]);
        assert!(none.content.is_empty());
        assert!(none.is_empty("cn"));
        assert!(none.is_empty("en"));
    }

    #[test]
    fn task_loop_and_subagent_sections_carry_their_rules() {
        let task_loop = task_loop_section();
        assert_eq!(task_loop.name, "task_loop");
        assert_eq!(task_loop.priority, 80);
        let cn = task_loop.render("cn");
        assert!(cn.starts_with("# 子智能体使用规则"));
        assert!(cn.contains("## 应当使用"));
        assert!(cn.contains("## 不应使用"));
        assert!(cn.contains("## 使用原则"));
        assert!(cn.contains("task_description"));
        let en = task_loop.render("en");
        assert!(en.contains("## When to Use"));
        assert!(en.contains("## When Not to Use"));
        assert!(en.contains("## Usage Principles"));

        let subagent = subagent_section();
        assert_eq!(subagent.name, "subagent_tools");
        assert_eq!(subagent.priority, 81);
        for tool in [
            "subagent_spawn",
            "subagent_wait",
            "subagent_list",
            "subagent_send_input",
            "subagent_close",
            "subagent_resume",
        ] {
            assert!(subagent.render("cn").contains(tool), "cn mentions {tool}");
            assert!(subagent.render("en").contains(tool), "en mentions {tool}");
        }
        assert!(subagent.render("cn").contains("同一回合"));
    }

    #[test]
    fn completion_signal_uses_the_default_promise() {
        let section = completion_signal();
        assert_eq!(section.name, "completion_signal");
        assert_eq!(section.priority, 83);
        assert_eq!(DEFAULT_COMPLETION_PROMISE, "TASK_COMPLETE");
        assert_eq!(promise_tag("DONE"), "<promise>DONE</promise>");
        let cn = section.render("cn");
        assert!(cn.contains("## 完成信号"));
        assert!(cn.contains("<promise>TASK_COMPLETE</promise>"));
        let en = section.render("en");
        assert!(en.contains("## Completion Signal"));
        assert!(en.contains("<promise>TASK_COMPLETE</promise>"));
    }

    #[test]
    fn completion_signal_with_custom_promise() {
        let section = completion_signal_with("SHIPPED");
        assert!(section.render("cn").contains("<promise>SHIPPED</promise>"));
        assert!(section.render("en").contains("<promise>SHIPPED</promise>"));
        assert!(!section.render("en").contains("TASK_COMPLETE"));
    }

    #[test]
    fn mode_instructions_only_emit_in_plan_mode() {
        let normal = mode_instructions(AgentMode::Normal);
        assert_eq!(normal.name, "mode_instructions");
        assert_eq!(normal.priority, 84);
        assert!(normal.content.is_empty());
        assert!(normal.is_empty("cn"));

        let plan = mode_instructions(AgentMode::Plan);
        let cn = plan.render("cn");
        assert!(cn.starts_with("## Plan 模式"));
        assert!(cn.contains("不得进行任何修改"));
        for phase in [
            "Phase 1: 初始理解",
            "Phase 2: 设计",
            "Phase 3: 复核",
            "Phase 4: 编写最终计划",
            "Phase 5: 结束规划阶段",
        ] {
            assert!(cn.contains(phase), "cn has {phase}");
        }
        assert!(cn.contains("`ask_user`") && cn.contains("`exit_plan_mode`"));
        assert!(cn.contains("端到端验证"));
        let en = plan.render("en");
        assert!(en.starts_with("## Plan Mode"));
        for phase in [
            "Phase 1: Initial Understanding",
            "Phase 2: Design",
            "Phase 3: Review",
            "Phase 4: Write Final Plan",
            "Phase 5: End Planning Phase",
        ] {
            assert!(en.contains(phase), "en has {phase}");
        }
        assert!(en.contains("This constraint takes priority"));
    }

    #[test]
    fn todo_section_appends_the_current_plan_when_present() {
        let empty = todo_section(&[]);
        assert_eq!(empty.name, "todo");
        assert_eq!(empty.priority, 92);
        assert!(empty.render("cn").starts_with("## 任务计划"));
        assert!(!empty.render("cn").contains("当前计划"));
        assert!(!empty.render("en").contains("Current Plan"));

        let todos = [
            todo("setup", "Set up the crate", TodoStatus::Completed),
            todo("port", "Port the sections", TodoStatus::InProgress),
        ];
        let section = todo_section(&todos);
        let cn = section.render("cn");
        assert!(cn.contains("### 当前计划"));
        assert!(cn.contains("- [√] setup: Set up the crate"), "{cn}");
        assert!(cn.contains("- [→] port: Port the sections"), "{cn}");
        let en = section.render("en");
        assert!(en.contains("### Current Plan"));
        assert!(en.contains("- [√] setup: Set up the crate"));
    }

    #[test]
    fn environment_section_lists_runtime_facts() {
        let section = environment_section("/work", "gpt-5", "2026-11-12", Some("main"));
        assert_eq!(section.name, "environment");
        assert_eq!(section.priority, 90);
        let cn = section.render("cn");
        assert!(cn.starts_with("## 运行环境"));
        assert!(cn.contains("- 工作目录：/work"));
        assert!(cn.contains("- 模型：gpt-5"));
        assert!(cn.contains("- Git 分支：main"));
        assert!(cn.contains("- 日期：2026-11-12"));
        let en = section.render("en");
        assert!(en.contains("- CWD: /work"));
        assert!(en.contains("- Model: gpt-5"));
        assert!(en.contains("- Git Branch: main"));
        assert!(en.contains("- Date: 2026-11-12"));
    }

    #[test]
    fn environment_section_renders_na_without_a_branch() {
        assert!(environment_section("/w", "m", "d", None)
            .render("en")
            .contains("- Git Branch: N/A"));
        assert!(environment_section("/w", "m", "d", None)
            .render("cn")
            .contains("- Git 分支：N/A"));
        assert!(environment_section("/w", "m", "d", Some("   "))
            .render("en")
            .contains("- Git Branch: N/A"));
    }
}
