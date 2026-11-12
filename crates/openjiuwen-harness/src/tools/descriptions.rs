//! Model-facing description text of the twenty built-in tools, in English and Chinese.
//!
//! Ports the bilingual description constants of `openjiuwen/harness/prompts/tools/`
//! (`filesystem.py`, `bash.py`, `todo.py`, `ask_user.py`, `memory.py`, `coding_memory.py`).
//! The text is prompt material seen by the model, so it is kept verbatim from the Python SDK,
//! including blank lines; tools look it up through [`description`] by name and language.

/// The languages with a description table ("en", "cn").
pub const SUPPORTED_LANGUAGES: [&str; 2] = ["en", "cn"];

/// The description table: `(tool name, English description, Chinese description)`.
///
/// Rows are in registry order and hold the text verbatim from the Python SDK, so the table is
/// intentionally long; use [`description`] to look rows up instead of indexing it directly.
static DESCRIPTIONS: &[(&str, &str, &str)] = &[
    (
        "bash",
        concat!(
            "Executes a given bash command and returns its output.\n",
            "\n",
            "The working directory persists between commands, but shell state (variables, functions, aliases) does not. The shell environment is initialized from the user's profile (bash or zsh).\n",
            "\n",
            "Windows note: `cmd`/PowerShell `mkdir` **does not support `-p`**; do not use `mkdir -p` in cmd/PowerShell. POSIX `mkdir -p` is appropriate only when the runtime environment information shows Git Bash or a non-WSL-stub PATH bash is available and you are actually using bash/Git Bash. Otherwise, use PowerShell `New-Item ... -Force` or create each level with cmd `mkdir`.\n",
            "\n",
            "IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience for the user:\n",
            "\n",
            " - File search: Use glob tool (NOT find or ls)\n",
            " - Content search: Use grep tool (NOT grep or rg)\n",
            " - Read files: Use read_file tool (NOT cat/head/tail)\n",
            " - Edit files: Use edit_file tool (NOT sed/awk)\n",
            " - Write files: Use write_file tool (NOT echo >/cat <<EOF)\n",
            " - Communication: Output text directly (NOT echo/printf)\n",
            "While the bash tool can do similar things, it is better to use the built-in tools as they provide a better user experience and make it easier to review tool calls and give permission.\n",
            "\n",
            "# Instructions\n",
            " - If your command will create new directories or files, first use this tool to run `ls` to verify the parent directory exists and is the correct location.\n",
            " - Always quote file paths that contain spaces with double quotes in your command (e.g., cd \"path with spaces/file.txt\").\n",
            " - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the user explicitly requests it.\n",
            " - You may specify an optional timeout in seconds (up to 3600s / 60 minutes). By default, your command will timeout after 300s.\n",
            " - You can use the `run_in_background` parameter to run the command in the background. Only use this if you don't need the result immediately and are OK being notified when the command completes later. You do not need to use '&' at the end of the command when using this parameter.\n",
            " - When issuing multiple commands:\n",
            "   - If the commands are independent and can run in parallel, make multiple bash tool calls in a single message. Example: if you need to run \"git status\" and \"git diff\", send a single message with two bash tool calls in parallel.\n",
            "   - If the commands depend on each other and must run sequentially, use a single bash call with '&&' to chain them together.\n",
            "   - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail.\n",
            "   - DO NOT use newlines to separate commands (newlines are ok in quoted strings).\n",
            " - For git commands:\n",
            "   - Prefer to create a new commit rather than amending an existing commit.\n",
            "   - Before running destructive operations (e.g., git reset --hard, git push --force, git checkout --), consider whether there is a safer alternative that achieves the same goal. Only use destructive operations when they are truly the best approach.\n",
            "   - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign) unless the user has explicitly asked for it. If a hook fails, investigate and fix the underlying issue.\n",
            "   - When a user shares a repo URL and asks you to 'look at' or 'analyze' it, the natural first step is `git clone <url> <local_path>`; after cloning, use read_file/grep/glob on the working tree - it gives you far more than the rendered repository page would.\n",
            "   - When creating a worktree with `git worktree add`, the target path MUST live under `.worktrees/<name>` inside the current project directory, e.g. `git worktree add .worktrees/feature-x -b feature-x HEAD`. NEVER use `../` or any path outside the current project directory as the worktree target; the project's sibling directory is out of bounds and pollutes space outside the project. When the base branch is unspecified, use the current branch (`HEAD`); do not assume `main`.\n",
            " - Avoid unnecessary `sleep` commands:\n",
            "   - Do not sleep between commands that can run immediately -- just run them.\n",
            "   - If your command is long running and you would like to be notified when it finishes -- use `run_in_background: true`. No sleep needed.\n",
            "   - Do not retry failing commands in a sleep loop -- diagnose the root cause.\n",
            "   - If waiting for a background task you started with `run_in_background: true`, you will be notified when it completes -- do not poll.\n",
            "   - If you must poll an external process, use a check command (e.g. `gh run view`) rather than sleeping first.\n",
            "   - If you must sleep, keep the duration short (1-5 seconds) to avoid blocking the user.",
        ),
        concat!(
            "执行 Shell 命令并返回输出。\n",
            "\n",
            "工作目录在命令之间保持不变，但 Shell 状态（变量、函数、alias）不保留。Shell 环境从用户的 profile（bash 或 zsh）初始化。\n",
            "\n",
            "Windows 注意：`cmd`/PowerShell 自带 `mkdir` **不支持 `-p`**，不要在 cmd/PowerShell 中使用 `mkdir -p`。只有运行环境信息显示 Git Bash 或非 WSL stub 的 PATH bash 可用，并且实际使用 bash/Git Bash 时，POSIX `mkdir -p` 才适用。否则应使用 PowerShell `New-Item ... -Force` 或 cmd 逐级 `mkdir`。\n",
            "\n",
            "重要：避免使用本工具执行 `find`、`grep`、`cat`、`head`、`tail`、`sed`、`awk` 或 `echo` 命令，除非明确指示或确认专用工具无法完成任务。请使用对应的专用工具，以获得更好的体验：\n",
            "\n",
            " - 文件搜索：使用 glob 工具（不要用 find 或 ls）\n",
            " - 内容搜索：使用 grep 工具（不要用 grep 或 rg 命令）\n",
            " - 读取文件：使用 read_file 工具（不要用 cat/head/tail）\n",
            " - 编辑文件：使用 edit_file 工具（不要用 sed/awk）\n",
            " - 写入文件：使用 write_file 工具（不要用 echo > 或 cat <<EOF）\n",
            " - 输出文本：直接输出（不要用 echo/printf）\n",
            "虽然 bash 工具也能做到，但专用工具提供更好的用户体验，并且更方便审查和授权。\n",
            "\n",
            "# 使用说明\n",
            " - 创建文件或目录前，先用本工具执行 `ls` 确认父目录存在且位置正确\n",
            " - 路径包含空格时必须用双引号括起（例如 cd \"path with spaces/file.txt\"）\n",
            " - 尽量使用绝对路径维持当前工作目录，避免使用 `cd`；除非用户明确要求\n",
            " - 可通过 timeout 参数指定超时（秒），默认 300 秒，上限 3600 秒\n",
            " - 可将 run_in_background 设为 true 来后台运行命令。仅在不需要立即获取结果时使用，命令完成后会收到通知。使用该参数时无需在命令末尾加 `&`\n",
            " - 发出多条命令时：\n",
            "   - 独立命令：在同一消息中多次并行调用本工具。例如需要同时运行 \"git status\" 和 \"git diff\"，发送一条消息包含两次并行调用\n",
            "   - 依赖命令：在单次调用中用 `&&` 串联\n",
            "   - 仅在不关心前序命令是否失败时使用 `;`\n",
            "   - 不要用换行分隔命令（引号字符串内换行可以）\n",
            " - Git 命令规范：\n",
            "   - 优先创建新提交而非修改已有提交\n",
            "   - 执行破坏性操作（如 git reset --hard、git push --force、git checkout --）前，先考虑是否有更安全的替代方案，只在确实必要时使用\n",
            "   - 除非用户明确要求，不要跳过 hooks（--no-verify）或绕过签名（--no-gpg-sign）。hook 失败时应排查并修复根本原因\n",
            "   - 用户分享代码仓 URL 让你「看看这个仓库」或「分析一下」时，首选动作是 `git clone <url> <本地路径>`；克隆完再用 read_file/grep/glob 等专用工具读源码，比抓取仓库主页能拿到的信息完整得多\n",
            "   - 用 `git worktree add` 创建 worktree 时，目标路径必须位于当前项目目录下的 `.worktrees/<name>`，例如 `git worktree add .worktrees/feature-x -b feature-x HEAD`。严禁使用 `../` 或项目目录之外的路径作为 worktree 目标——项目同级目录属于越界，会污染项目外部空间。base 分支未指定时用当前分支（`HEAD`），不要臆测 `main`。\n",
            " - 避免不必要的 `sleep` 命令：\n",
            "   - 能立即执行的命令之间不要 sleep\n",
            "   - 长时间运行的命令使用 `run_in_background: true`，无需 sleep 等待\n",
            "   - 不要在 sleep 循环中重试失败命令——排查根本原因\n",
            "   - 等待后台任务完成时会自动通知——不要轮询\n",
            "   - 如必须轮询外部进程，使用检查命令（如 `gh run view`）而非先 sleep\n",
            "   - 如必须 sleep，保持短时间（1-5 秒）",
        ),
    ),
    (
        "read_file",
        "Read a file from the local filesystem. file_path must be an absolute path. By default, reads up to 2000 lines from the beginning and returns results with line numbers. Can read files only, not directories. For large files, use offset/limit to read a specific portion, or use grep to search for specific content.",
        "从本地文件系统读取文件。file_path 必须是绝对路径。默认从文件开头最多读取 2000 行，结果带行号返回。只能读取文件，不能读取目录。大文件请用 offset/limit 读取指定部分，或用 grep 搜索具体内容。",
    ),
    (
        "write_file",
        concat!(
            "Writes a file to the local filesystem.\n",
            "\n",
            "Usage:\n",
            "- This tool will overwrite the existing file if there is one at the provided path.\n",
            "- If this is an existing file, you MUST use read_file first to read the file's contents. This tool will fail if you did not read the file first.\n",
            "- Prefer edit_file for modifying existing files; it only sends the diff. Only use write_file to create new files or for complete rewrites.\n",
            "- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.",
        ),
        concat!(
            "向本地文件系统写入文件。\n",
            "\n",
            "用法：\n",
            "- 如果目标路径已有文件，本工具会覆盖该文件。\n",
            "- 如果目标文件已存在，必须先使用 read_file 读取文件内容；否则本工具会失败。\n",
            "- 修改已有文件时优先使用 edit_file；edit_file 只提交差异。write_file 主要用于创建新文件或完整重写文件。\n",
            "- 除非用户明确要求，不要创建文档文件（*.md）或 README 文件。",
        ),
    ),
    (
        "edit_file",
        "Performs exact string replacements in files. Existing files must be fully read with read_file before editing. old_string must match exactly once; if multiple matches exist, provide more context or set replace_all=true. Creates a new file when old_string is empty and the target file does not exist. Does not support editing .ipynb files.",
        "对文件执行精确字符串替换。编辑已有文件前必须先完整调用 read_file。old_string 必须唯一匹配；如果有多个匹配，请提供更多上下文或设置 replace_all=true。old_string 为空且目标文件不存在时，可创建新文件。不支持编辑 .ipynb 文件。",
    ),
    (
        "glob",
        "Fast file pattern matching tool that works with any codebase size. Supports glob patterns like **/*.js or src/**/*.ts and returns matching paths sorted by file name. Returns up to 100 results by default. Use this tool when you need to find files by name patterns.",
        "快速按 glob 模式查找文件，适用于任意规模代码库。支持 **/*.js、src/**/*.ts 等模式，返回按文件名排序的匹配路径。默认最多返回 100 个结果。需要按文件名模式找文件时使用本工具。",
    ),
    (
        "list_files",
        "List directory contents.",
        "列出目录内容。",
    ),
    (
        "grep",
        "A powerful search tool built on ripgrep. ALWAYS use grep for search tasks. NEVER invoke `grep` or `rg` as a bash command. The grep tool has been optimized for correct permissions and access. Supports full regex syntax, glob/type filters, and output modes: content, files_with_matches, count.",
        "基于 ripgrep 的强大搜索工具。搜索任务必须使用 grep，不要在 bash 命令中直接调用 `grep` 或 `rg`。grep 工具已针对权限和访问控制做过优化。支持完整正则语法、glob/type 文件过滤，以及输出模式：content、files_with_matches、count。",
    ),
    (
        "todo_create",
        concat!(
            "\n",
            "Create a todo list for the current session to track progress, organize complex tasks, and help the user understand overall execution status.\n",
            "\n",
            "## Usage\n",
            "\n",
            "Input is a JSON array (each task must include an id field):\n",
            "    {\"todos\": [{\"id\": \"translate_doc\", \"content\": \"Translate document\", \"activeForm\": \"Translating document\", \"description\": \"Translate the document into the target language\", \"selected_model_id\": \"fast\"}, {\"id\": \"analyze_arch\", \"content\": \"Analyze code architecture\", \"activeForm\": \"Analyzing code architecture\", \"description\": \"Map out module structure and dependencies\", \"selected_model_id\": \"smart\"}]}\n",
            "\n",
            "## Rules\n",
            "\n",
            "- **id is required**: You must provide a short, semantically meaningful unique string ID for each task (e.g. \"translate_doc\", \"analyze_code\"). Do NOT use random characters or UUIDs. IDs must be unique within a session. This ID is used by todo_modify to precisely locate tasks and is the sole key for cross-turn status updates\n",
            "- First task is automatically set to in_progress, others to pending\n",
            "- Only one task can be in_progress at a time\n",
            "- Task descriptions must be specific, actionable, and clear\n",
            "- Calling this tool replaces the current session's task list; use todo_modify to append tasks\n",
            "- When the currently available model information is not obtained, the selected_model_id field should not be added; otherwise, selected_model_id must be added to specify the model ID for executing the task.\n",
        ),
        concat!(
            "\n",
            "创建当前会话的待办事项列表，用于跟踪进度、组织复杂任务，帮助用户了解整体执行情况。\n",
            "\n",
            "## 使用方式\n",
            "\n",
            "入参为 JSON 数组（每个任务必须包含 id 字段）：\n",
            "    {\"todos\": [{\"id\": \"translate_doc\", \"content\": \"翻译文档\", \"activeForm\": \"正在翻译文档\", \"description\": \"将文档翻译为目标语言\", \"selected_model_id\": \"fast\"}, {\"id\": \"analyze_arch\", \"content\": \"分析代码架构\", \"activeForm\": \"正在分析代码架构\", \"description\": \"梳理代码模块结构与依赖关系\", \"selected_model_id\": \"smart\"}]}\n",
            "\n",
            "## 规则\n",
            "\n",
            "- **id 字段为必填**：必须为每个任务指定简短、语义清晰的唯一字符串 ID（如 \"translate_doc\"、\"analyze_code\"），禁止使用随机字符或 UUID。同一会话内 ID 不得重复。此 ID 在后续 todo_modify 中用于精准定位任务，是跨轮次更新状态的唯一依据\n",
            "- 第一个任务自动设为 in_progress，其余为 pending\n",
            "- 同一时间只能有一个 in_progress 任务\n",
            "- 任务描述必须具体、可执行、清晰明确\n",
            "- 调用本工具会覆盖当前会话的任务列表；若需追加任务，请使用 todo_modify\n",
            "- 当没有获取到当前可用模型信息时，不要添加 selected_model_id 字段；否则必须添加 selected_model_id 指定执行任务的模型 ID\n",
        ),
    ),
    (
        "todo_list",
        concat!(
            "\n",
            "Retrieve and display all todo items for the current session\n",
            "\n",
            "## When to Use todo_list (vs. todo_modify)\n",
            "\n",
            "Use todo_list when:\n",
            "- You need an overview of all tasks and their IDs before deciding how to update\n",
            "- You are unsure which tasks are currently in_progress or pending\n",
            "\n",
            "Use todo_modify directly (no need to call todo_list first) when:\n",
            "- You already know the task ID and want to update task information\n",
            "- A task just finished and you want to mark it completed immediately\n",
        ),
        concat!(
            "\n",
            "检索并显示当前会话的所有待办事项。\n",
            "\n",
            "## 何时使用 todo_list（而非 todo_modify）\n",
            "\n",
            "使用 todo_list 的场景：\n",
            "- 需要查看当前任务全貌和各任务ID，再决定如何更新\n",
            "- 不确定当前有哪些任务处于 in_progress 或 pending\n",
            "\n",
            "使用 todo_modify 的场景（不需要先调用 todo_list）：\n",
            "- 已知任务 ID，直接更新任务信息\n",
            "- 任务刚完成，立即标记为 completed\n",
        ),
    ),
    (
        "todo_get",
        concat!(
            "\n",
            "Get full details of a single task by its ID.\n",
            "\n",
            "Input: id (unique task identifier)\n",
            "\n",
            "Returns: complete task info including id, content (task summary), activeForm, description (detailed content), status, depends_on, result_summary, meta_data, selected_model_id.\n",
        ),
        concat!(
            "\n",
            "根据任务 ID 获取单个任务的完整详情。\n",
            "\n",
            "入参：id（任务唯一标识符）\n",
            "\n",
            "返回：完整的任务信息，包括 id、content（任务摘要）、activeForm、description（任务详细内容）、status、depends_on、result_summary、meta_data、selected_model_id。\n",
        ),
    ),
    (
        "todo_modify",
        concat!(
            "\n",
            "Modify todo items for the current session, including: update, delete, cancel, append, insert_after, and insert_before.\n",
            "\n",
            "Important notes:\n",
            "- To re-plan the entire task list, call todo_create instead\n",
            "- Batch structural changes that occur at the same time to avoid repeated calls. Status transitions should be updated immediately as execution progresses: a single update should normally change only the current task and the next task; do not change multiple pending tasks to completed in one call\n",
            "- **Use the semantic id you assigned in todo_create** (e.g. \"translate_doc\"); never use UUIDs\n",
            "- **Only send the fields that changed**: for update, each call only needs the id plus the fields that actually changed (usually just status) — do not resend unchanged fields like content, description, or activeForm\n",
            "\n",
            "Supported action types:\n",
            "\n",
            "update: Modify status or content of existing tasks (id cannot be changed; only send the fields that changed, omit the rest):\n",
            "    {\n",
            "        \"action\": \"update\",\n",
            "        \"todos\": [\n",
            "            {\"id\": \"translate_doc\", \"status\": \"completed\"},\n",
            "            {\"id\": \"analyze_code\", \"status\": \"in_progress\"}\n",
            "        ]\n",
            "    }\n",
            "\n",
            "Support modifying selected_model_id: If the task's selected_model_id is not empty and the execution result is of poor quality (inaccurate output, logical errors, or failure to meet expectations), the model ID should be updated according to the model description to a higher-quality model:\n",
            "    {\n",
            "        \"action\": \"update\",\n",
            "        \"todos\": [\n",
            "            {\"id\": \"translate_doc\", \"selected_model_id\": \"smart\", \"status\": \"pending\"}\n",
            "        ]\n",
            "    }\n",
            "\n",
            "cancel: Mark specified tasks as cancelled (tasks will be ignored and not executed):\n",
            "    {\n",
            "        \"action\": \"cancel\",\n",
            "        \"ids\": [\"translate_doc\", \"analyze_code\"]\n",
            "    }\n",
            "\n",
            "delete: Permanently remove specified tasks from the list:\n",
            "    {\n",
            "        \"action\": \"delete\",\n",
            "        \"ids\": [\"translate_doc\"]\n",
            "    }\n",
            "\n",
            "append: Add new tasks at the end of the list (id must be a short semantic string you choose, unique within the session):\n",
            "    {\n",
            "        \"action\": \"append\",\n",
            "        \"todos\": [\n",
            "            {\"id\": \"write_report\", \"content\": \"New task content\", \"activeForm\": \"Executing new task\", \"description\": \"Detailed description of the task\", \"status\": \"pending\"}\n",
            "        ]\n",
            "    }\n",
            "\n",
            "insert_after: Insert new tasks after the specified task (target must be in_progress or pending):\n",
            "    {\n",
            "        \"action\": \"insert_after\",\n",
            "        \"todo_data\": {\"target_id\": \"translate_doc\", \"items\": [{\"id\": \"review_translation\", \"content\": \"Inserted task\", \"activeForm\": \"Executing inserted task\", \"description\": \"Detailed description of the task\", \"status\": \"pending\", \"selected_model_id\": \"fast\"}]}\n",
            "    }\n",
            "\n",
            "insert_before: Insert new tasks before the specified task (target must be pending):\n",
            "    {\n",
            "        \"action\": \"insert_before\",\n",
            "        \"todo_data\": {\"target_id\": \"analyze_code\", \"items\": [{\"id\": \"setup_env\", \"content\": \"Inserted task\", \"activeForm\": \"Executing inserted task\", \"description\": \"Detailed description of the task\", \"status\": \"pending\"}]}\n",
            "    }\n",
            "\n",
            "Core rules:\n",
            "- update: the id field cannot be modified; all other fields support partial updates\n",
            "- Only one task can be in_progress at a time\n",
            "- insert_after: target task status must be in_progress or pending\n",
            "- insert_before: target task status must be pending\n",
            "- If the task's selected_model_id is empty, do not modify the selected_model_id field in any operation.\n",
        ),
        concat!(
            "\n",
            "修改当前会话的待办事项，包括：更新（update）、删除（delete）、取消（cancel）、追加（append）、在其后插入（insert_after）、在其前插入（insert_before）\n",
            "\n",
            "重要说明：\n",
            "- 若需重新规划整个任务列表，请调用 todo_create\n",
            "- 支持批量处理同一时点发生的结构变更，避免连续多次调用。状态流转应随执行进度即时更新：单次 update 的状态变更通常只涉及当前任务和下一任务，不要将多个 pending 任务批量改为 completed\n",
            "- **id 使用你在 todo_create 时自定义的语义 ID**（如 \"translate_doc\"），不要使用 UUID\n",
            "- **只传入发生变化的字段**：update 每次调用只需给出 id 加上实际改变的字段（通常只是 status），不要把 content、description、activeForm 等未变化的字段重新完整传一遍\n",
            "\n",
            "action 支持的操作类型：\n",
            "\n",
            "update：修改现有任务的状态或标题（id 不可修改，只需传入发生变化的字段，其余字段保持不传即可）：\n",
            "    {\n",
            "        \"action\": \"update\",\n",
            "        \"todos\": [\n",
            "            {\"id\": \"translate_doc\", \"status\": \"completed\"},\n",
            "            {\"id\": \"analyze_code\", \"status\": \"in_progress\"}\n",
            "        ]\n",
            "    }\n",
            "\n",
            "支持修改 selected_model_id：若任务 selected_model_id 不为空，且执行结果质量不佳（输出不准确、逻辑错误、未达预期），应根据模型描述更新质量更高的模型ID：\n",
            "    {\n",
            "        \"action\": \"update\",\n",
            "        \"todos\": [\n",
            "            {\"id\": \"translate_doc\", \"selected_model_id\": \"smart\", \"status\": \"pending\"}\n",
            "        ]\n",
            "    }\n",
            "\n",
            "cancel：将指定任务标记为 cancelled（任务将被忽略，不再执行）：\n",
            "    {\n",
            "        \"action\": \"cancel\",\n",
            "        \"ids\": [\"translate_doc\", \"analyze_code\"]\n",
            "    }\n",
            "\n",
            "delete：从列表中永久删除指定任务：\n",
            "    {\n",
            "        \"action\": \"delete\",\n",
            "        \"ids\": [\"translate_doc\"]\n",
            "    }\n",
            "\n",
            "append：在列表末尾追加新任务（id 由你指定，须简短语义且唯一）：\n",
            "    {\n",
            "        \"action\": \"append\",\n",
            "        \"todos\": [\n",
            "            {\"id\": \"write_report\", \"content\": \"新任务内容\", \"activeForm\": \"执行新任务\", \"description\": \"任务的详细描述\", \"status\": \"pending\"}\n",
            "        ]\n",
            "    }\n",
            "\n",
            "insert_after：在指定任务之后插入新任务（目标任务状态须为 in_progress 或 pending）：\n",
            "    {\n",
            "        \"action\": \"insert_after\",\n",
            "        \"todo_data\": {\"target_id\": \"translate_doc\", \"items\": [{\"id\": \"review_translation\", \"content\": \"插入的任务\", \"activeForm\": \"执行插入的任务\", \"description\": \"任务的详细描述\", \"status\": \"pending\", \"selected_model_id\": \"fast\"}]}\n",
            "    }\n",
            "\n",
            "insert_before：在指定任务之前插入新任务（目标任务状态须为 pending）：\n",
            "    {\n",
            "        \"action\": \"insert_before\",\n",
            "        \"todo_data\": {\"target_id\": \"analyze_code\", \"items\": [{\"id\": \"setup_env\", \"content\": \"插入的任务\", \"activeForm\": \"执行插入的任务\", \"description\": \"任务的详细描述\", \"status\": \"pending\"}]}\n",
            "    }\n",
            "\n",
            "核心规则：\n",
            "- update 操作：id 字段不可修改，其他字段支持部分更新\n",
            "- 同一时间只能有一个任务处于 in_progress\n",
            "- insert_after：目标任务状态必须为 in_progress 或 pending\n",
            "- insert_before：目标任务状态必须为 pending\n",
            "- 如果任务的 selected_model_id 为空时，任何操作都不要更改 selected_model_id 字段\n",
        ),
    ),
    (
        "ask_user",
        concat!(
            "Ask user questions to gather info, clarify ambiguity, or make decisions. Supports 1-4 questions, each with 2-4 options.\n",
            "\n",
            "When to use proactively: Ask when requirements are vague, multiple approaches exist, or user preferences matter. Don't assume.\n",
            "\n",
            "FORBIDDEN: Adding 'Other', 'Custom' etc. as options — system provides this automatically. RECOMMENDED: Place recommended option first, append '(Recommended)' to its label. Preview field is only for single-select questions with visual comparison needs.",
        ),
        concat!(
            "向用户提问以收集信息、澄清歧义或做出决策。支持1-4个问题，每个问题2-4个选项。\n",
            "\n",
            "何时主动使用：需求模糊、多种方案可选、涉及用户偏好时，应主动询问而非假设。\n",
            "\n",
            "【禁止】选项中添加'其他'、'自定义'等兜底选项，系统已自动提供。【推荐】将推荐选项放第一位，label末尾加'（推荐）'。preview字段仅用于单选问题的视觉比较场景。",
        ),
    ),
    (
        "memory_search",
        "Search long-term memory (prior decisions, preferences, people, dates, todos) and return relevant snippets and references.",
        "在长期记忆中检索过往信息（决策、偏好、人物、日期、TODO 等），返回相关片段与引用线索。",
    ),
    (
        "memory_get",
        "Read a slice of a memory markdown file under memory/ (from_line + lines).",
        "按行号切片读取 memory/ 下的记忆 Markdown 文件内容（from_line + lines）。",
    ),
    (
        "write_memory",
        "Write memory content to a markdown file under memory/; supports overwrite or append.",
        "写入记忆内容到 memory/ 下的 Markdown 文件；支持覆盖写或追加写（append）。",
    ),
    (
        "read_memory",
        "Read a portion of a memory file under memory/ using offset/limit (for paging).",
        "按 offset/limit 读取 memory/ 下记忆文件的部分内容（用于分页阅读）。",
    ),
    (
        "edit_memory",
        "Perform an exact string replacement inside a memory file (old_text → new_text).",
        "在 memory/ 下的记忆文件中做精确字符串替换（old_text → new_text）。",
    ),
    (
        "coding_memory_read",
        "Read a portion of a memory file under coding_memory/ using offset/limit (for paging).",
        "按 offset/limit 读取 coding_memory/ 下记忆文件的部分内容（用于分页阅读）。",
    ),
    (
        "coding_memory_write",
        "Write memory content to a markdown file under coding_memory/ (frontmatter required).",
        "写入记忆内容到 coding_memory/ 下的 Markdown 文件（要求 frontmatter）。",
    ),
    (
        "coding_memory_edit",
        "Perform an exact string replacement inside a coding memory file (old_text → new_text).",
        "在 coding_memory/ 下的记忆文件中做精确字符串替换（old_text → new_text）。",
    ),
];

/// Returns the description of `name` in `language`, or `""` for an unknown tool name.
///
/// Only the exact language string `"en"` selects English; every other value, including unknown
/// languages, falls back to Chinese, matching the Python SDK's `DESCRIPTIONS.get(lang, cn)`.
#[must_use]
pub fn description(name: &str, language: &str) -> &'static str {
    let column = usize::from(language != "en");
    DESCRIPTIONS
        .iter()
        .find(|(tool, ..)| *tool == name)
        .map_or("", |(_, english, chinese)| [english, chinese][column])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{description, DESCRIPTIONS, SUPPORTED_LANGUAGES};

    /// The twenty model-facing tool names, in registry order.
    const TOOL_NAMES: [&str; 20] = [
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
        "memory_search",
        "memory_get",
        "write_memory",
        "read_memory",
        "edit_memory",
        "coding_memory_read",
        "coding_memory_write",
        "coding_memory_edit",
    ];

    #[test]
    fn supported_languages_are_english_and_chinese() {
        assert_eq!(SUPPORTED_LANGUAGES, ["en", "cn"]);
    }

    #[test]
    fn every_tool_has_english_and_chinese_text() {
        for name in TOOL_NAMES {
            assert!(
                !description(name, "en").is_empty(),
                "empty English description for {name}"
            );
            assert!(
                !description(name, "cn").is_empty(),
                "empty Chinese description for {name}"
            );
        }
    }

    #[test]
    fn english_differs_from_chinese_for_every_tool() {
        for name in TOOL_NAMES {
            assert_ne!(
                description(name, "en"),
                description(name, "cn"),
                "English and Chinese descriptions are identical for {name}"
            );
        }
    }

    #[test]
    fn unknown_language_falls_back_to_chinese() {
        for name in TOOL_NAMES {
            assert_eq!(description(name, "fr"), description(name, "cn"));
            assert_eq!(description(name, "EN"), description(name, "cn"));
            assert_eq!(description(name, ""), description(name, "cn"));
        }
    }

    #[test]
    fn unknown_tool_name_is_empty() {
        assert_eq!(description("", "en"), "");
        assert_eq!(description("no_such_tool", "en"), "");
        assert_eq!(description("no_such_tool", "cn"), "");
        assert_eq!(description("Bash", "en"), "");
    }

    #[test]
    fn table_holds_exactly_the_twenty_tools_in_registry_order() {
        let names: Vec<&str> = DESCRIPTIONS.iter().map(|(name, ..)| *name).collect();
        assert_eq!(names, TOOL_NAMES);
    }

    #[test]
    fn descriptions_keep_their_leading_text() {
        assert!(description("bash", "en").starts_with("Executes a given bash command"));
        assert!(description("bash", "cn").starts_with("执行 Shell 命令并返回输出。"));
        assert!(
            description("read_file", "en").starts_with("Read a file from the local filesystem.")
        );
        assert!(
            description("write_file", "en").starts_with("Writes a file to the local filesystem.")
        );
        assert!(description("glob", "en").starts_with("Fast file pattern matching tool"));
        assert!(description("grep", "en").starts_with("A powerful search tool built on ripgrep"));
        assert!(description("ask_user", "en").starts_with("Ask user questions to gather info"));
        assert!(description("memory_search", "en").starts_with("Search long-term memory"));
        assert!(description("coding_memory_read", "en")
            .starts_with("Read a portion of a memory file under coding_memory/"));
        // Python stores the todo descriptions as triple-quoted strings, so the Chinese and
        // English text starts with a newline; the check is on the rendered first line.
        assert!(description("todo_create", "en")
            .trim_start()
            .starts_with("Create a todo list for the current session to track progress"));
        assert!(description("todo_list", "en")
            .trim_start()
            .starts_with("Retrieve and display all todo items for the current session"));
        assert!(description("todo_get", "en")
            .trim_start()
            .starts_with("Get full details of a single task by its ID."));
        assert!(description("todo_modify", "en")
            .trim_start()
            .starts_with("Modify todo items for the current session, including:"));
    }

    #[test]
    fn read_file_description_omits_the_deferred_readers() {
        let english = description("read_file", "en");
        assert_eq!(
            english,
            "Read a file from the local filesystem. file_path must be an absolute path. \
             By default, reads up to 2000 lines from the beginning and returns results with line \
             numbers. Can read files only, not directories. For large files, use offset/limit to \
             read a specific portion, or use grep to search for specific content."
        );
        assert!(!english.contains("PDF"));
        assert!(!description("read_file", "cn").contains("PDF"));
    }
}
