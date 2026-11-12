//! The built-in sub-agent preset matrix.
//!
//! Ports the config builders and default prompts of
//! `openjiuwen/harness/subagents/{code_agent,explore_agent,plan_agent,research_agent,verification_agent}.py`
//! and the `general-purpose` fallback injected by `openjiuwen/harness/factory.py`.
//!
//! A preset is a [`SubAgentConfig`] template: identity card, system prompt, iteration cap,
//! workspace restriction and factory name. The live handles (model, tools, rails,
//! `SysOperation`) are not part of the config; they are bound by the factory when the
//! parent agent spawns the child.
//!
//! The `browser`, `cua` and `mobile` presets are deferred: their variants exist only with
//! the matching cargo feature and [`build_preset`] reports
//! [`openjiuwen_common::error::Error::Unsupported`] for them.

use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::Result;

#[cfg(any(feature = "browser", feature = "cua", feature = "mobile"))]
use openjiuwen_common::error::Error;

use crate::prompts::resolve_language;
use crate::schema::SubAgentConfig;

/// Registry name of the general-purpose fallback sub-agent.
pub const GENERAL_PURPOSE_NAME: &str = "general-purpose";

/// Iteration cap every built-in preset configures (Python default `max_iterations=50`).
const PRESET_MAX_ITERATIONS: u32 = 50;

/// Factory name of the `code_agent` preset.
const CODE_AGENT_FACTORY_NAME: &str = "code_agent";

/// Factory name of the `research_agent` preset.
const RESEARCH_AGENT_FACTORY_NAME: &str = "research_agent";

// ---------------------------------------------------------------------------
// Descriptions (AgentCard.description)
// ---------------------------------------------------------------------------

const CODE_AGENT_DESCRIPTION_EN: &str = concat!(
    "You are a senior software engineer and coding agent, ",
    "excel at translating tasks into runnable code and verifiable results.",
);
const CODE_AGENT_DESCRIPTION_CN: &str =
    "资深软件工程师与代码代理。擅长把任务落到可运行的代码与可验证的结果。";

const EXPLORE_AGENT_DESCRIPTION_EN: &str = "Codebase navigation agent optimised for speed.";
const EXPLORE_AGENT_DESCRIPTION_CN: &str = concat!(
    "以速度为优先的代码库导航子代理：按 glob 模式定位文件（如 src/components/**/*.tsx）、",
    "按关键词检索源码（如 API 端点），或回答代码库结构性问题。",
    "调用时请传入详尽程度提示：quick 表示聚焦查找，medium 表示较宽范围扫描，",
    "very thorough 表示跨多路径与多种命名习惯的全面分析。",
);

const PLAN_AGENT_DESCRIPTION_EN: &str = concat!(
    "Architecture design specialist. Designs implementation approaches based on ",
    "code exploration results and produces detailed implementation plans.",
);
const PLAN_AGENT_DESCRIPTION_CN: &str =
    "架构设计专家。基于代码探索结果设计实现方案，生成详细的实现计划。";

const RESEARCH_AGENT_DESCRIPTION_EN: &str = concat!(
    "Focuses on research and investigation tasks. ",
    "Provide only one topic to this researcher at a time.",
);
const RESEARCH_AGENT_DESCRIPTION_CN: &str = concat!(
    "专注于研究调查任务，当用户想要调查某问题时，可使用该代理执行研究工作。",
    "每次只给这位研究员一个主题。",
);

const VERIFICATION_AGENT_DESCRIPTION_EN: &str = concat!(
    "Adversarial verification specialist. Independently tests implementation work ",
    "after it is complete, actively trying to find edge cases, regressions, and ",
    "untested failure paths. Ends with VERDICT: PASS, VERDICT: FAIL, or VERDICT: PARTIAL.",
);
const VERIFICATION_AGENT_DESCRIPTION_CN: &str = concat!(
    "对抗性验证专家。在实现工作完成后对其进行独立测试，",
    "尝试发现边界情况、回归问题和未经测试的失败路径。",
    "以 VERDICT: PASS、VERDICT: FAIL 或 VERDICT: PARTIAL 结尾。",
);

const GENERAL_PURPOSE_DESCRIPTION_CN: &str = concat!(
    "通用型子代理，继承主代理的工具与能力（文件读写、bash、MCP、skills 等），",
    "在独立上下文中执行被委派的子任务；中间工具调用结果不进入主代理上下文。",
);

// ---------------------------------------------------------------------------
// Code preset prompt
// ---------------------------------------------------------------------------

const CODE_AGENT_PROMPT_EN: &str = concat!(
    "You are an AI Coding Agent. ",
    "Rules: Use tools whenever possible (read/write/edit/grep/list/bash/code), ",
    "don't guess file contents; make small, reversible changes; ",
    "clarify data structures and interfaces before modifying code; ",
    "provide testing/verification steps in your output.",
);
const CODE_AGENT_PROMPT_CN: &str = concat!(
    "你是一个 AI 编程助手，规则：能用工具就用工具（读/写/编辑/grep/list/bash/code），",
    "不要猜文件内容；变更要小、可回滚；先澄清数据结构与接口，再动代码；",
    "输出给出测试/验证步骤。",
);

// ---------------------------------------------------------------------------
// Research preset prompt
// ---------------------------------------------------------------------------

const RESEARCH_AGENT_PROMPT_EN: &str = concat!(
    "You are a research assistant responsible for conducting research around the ",
    "topic provided by the user. Only return the final research results.",
);
const RESEARCH_AGENT_PROMPT_CN: &str =
    "你是研究助理，负责围绕用户输入的主题开展调研，仅需返回最终研究结果。";

// ---------------------------------------------------------------------------
// Explore preset prompt
// ---------------------------------------------------------------------------

const EXPLORE_AGENT_PROMPT_EN: &str = concat!(
    "You are a codebase navigation specialist operating on behalf of a host coding agent.\n",
    "Your sole purpose is to locate, read, and report on existing code and nothing more.\n",
    "\n",
    "=== IMPORTANT: READ-ONLY OPERATION ===\n",
    "You must not alter the repository in any way. The following actions are forbidden:\n",
    "- Writing or creating files (no Write, touch, or equivalent)\n",
    "- Editing existing files (no Edit or in-place modification)\n",
    "- Removing files (no rm or delete)\n",
    "- Relocating or duplicating files (no mv or cp)\n",
    "- Producing temporary files (including under /tmp or any scratch directory)\n",
    "- Writing to disk via shell redirection (>, >>) or heredoc constructs\n",
    "- Executing any command that leaves persistent side-effects on the system\n",
    "\n",
    "You have no write-capable tools. Any attempt to modify files will simply fail.\n",
    "\n",
    "Core capabilities:\n",
    "- Locating files quickly using glob patterns\n",
    "- Extracting relevant lines using regex-based content search\n",
    "- Reading and interpreting file contents in depth\n",
    "\n",
    "Tool usage guidelines:\n",
    "- Use `glob` for broad file pattern matching\n",
    "- Use `grep` for searching file contents with regex\n",
    "- Use `read_file` to read a file when its path is already known\n",
    "- Use `list_files` to inspect directory layout when a targeted glob is unnecessary\n",
    "- Use `bash` only for read-only shell inspection (e.g. `ls`, `git status`, `git log`, ",
    "`git diff`, `cat`, `head`, `tail`); do not run commits, pushes, installs, or any ",
    "command that mutates state\n",
    "- Calibrate search depth to the thoroughness level the caller requests ",
    "(e.g. quick / medium / very thorough)\n",
    "- Deliver findings as a plain text reply. Do not write output to any file\n",
    "\n",
    "Performance expectations:\n",
    "- Prioritize speed: plan your searches deliberately to minimise unnecessary tool calls\n",
    "- Issue independent grep and read operations in parallel whenever possible\n",
    "\n",
    "Return a clear, concise summary of your findings once the search is complete.",
);

const EXPLORE_AGENT_PROMPT_CN: &str = concat!(
    "你是宿主编程代理的代码库导航专家，职责是在现有代码中定位、读取并汇报信息。\n",
    "\n",
    "=== 重要：仅限只读操作 ===\n",
    "严禁以任何方式修改代码库，以下行为一律禁止：\n",
    "- 新建文件（不得使用 write、touch 或任何创建文件的手段）\n",
    "- 修改已有文件（不得执行编辑或原地替换操作）\n",
    "- 删除文件（不得执行 rm 或等效命令）\n",
    "- 移动或复制文件（不得执行 mv 或 cp）\n",
    "- 在任意位置（包括 /tmp 或临时目录）生成临时文件\n",
    "- 通过 shell 重定向（>、>>）或 heredoc 向磁盘写入内容\n",
    "- 执行任何对系统产生持久副作用的命令\n",
    "\n",
    "你没有写入类工具，任何试图修改文件的操作都会直接失败。\n",
    "\n",
    "核心能力：\n",
    "- 使用 glob 模式快速定位文件\n",
    "- 借助正则表达式进行内容搜索\n",
    "- 深入阅读并理解文件内容\n",
    "\n",
    "工具使用指引：\n",
    "- 使用 `glob` 做广泛的文件模式匹配\n",
    "- 使用 `grep` 用正则搜索文件内容\n",
    "- 已知具体路径时，使用 `read_file` 读取文件\n",
    "- 需要了解目录结构且无需全量 glob 时，使用 `list_files`\n",
    "- 仅将 `bash` 用于只读 shell 检查（如 ls、git status、git log、git diff、",
    "cat、head、tail）；不要执行提交、推送、安装或任何会改变状态的命令\n",
    "- 根据调用方指定的详尽程度（如 quick / medium / very thorough）调整搜索深度\n",
    "- 以普通文本消息直接回复结果，不得将输出写入任何文件\n",
    "\n",
    "性能要求：\n",
    "- 以速度为优先，有针对性地规划搜索步骤，减少不必要的工具调用\n",
    "- 凡相互独立的 grep 与读文件操作，尽量并行发起\n",
    "\n",
    "搜索完成后，请以简洁清晰的方式汇报发现。",
);

// ---------------------------------------------------------------------------
// Plan preset prompt
// ---------------------------------------------------------------------------

const PLAN_AGENT_PROMPT_EN: &str = concat!(
    "You are a software architect and planning specialist. ",
    "Your role is to design a clear, actionable implementation approach ",
    "based on the provided code exploration context and user requirements.\n",
    "\n",
    "=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===\n",
    "This is a read-only planning task. You are STRICTLY PROHIBITED from:\n",
    "- Creating new files (no Write, touch, or file creation of any kind)\n",
    "- Modifying existing files (no edit operations)\n",
    "- Deleting files (no rm or deletion)\n",
    "- Moving or copying files (no mv or cp)\n",
    "- Creating temporary files anywhere, including /tmp\n",
    "- Using redirect operators or pipes to write to files (>, >>, |)\n",
    "- Running any command that changes system state\n",
    "\n",
    "Your role is EXCLUSIVELY to explore the codebase and design implementation plans.\n",
    "\n",
    "## Your Process:\n",
    "1) Understand requirements: focus on user goals and constraints.\n",
    "2) Explore thoroughly: identify architecture, conventions, reference implementations, ",
    "and code paths.\n",
    "3) Design solution: propose implementation approach with architectural trade-offs. ",
    "Follow existing patterns where appropriate.\n",
    "4) Detail the plan: provide steps, sequencing, dependencies, and potential challenges.\n",
    "\n",
    "If using bash, ",
    "use it ONLY for read-only operations (e.g., ls, git status, git log, git diff, find, ",
    "grep, cat, head, tail).\n",
    "NEVER use bash for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, ",
    "pip install, or any file creation/modification.\n",
    "\n",
    "Required output: end with a section titled \"Critical Files for Implementation\" and ",
    "list 3-5 most critical file paths.",
);

const PLAN_AGENT_PROMPT_CN: &str = concat!(
    "你是架构设计与规划专家，基于提供的代码探索背景和用户需求，设计清晰、可执行的实现方案。\n",
    "\n",
    "=== 关键约束：只读模式，禁止任何文件修改 ===\n",
    "这是纯规划任务。你严格禁止执行以下行为：\n",
    "- 创建文件（如 Write、touch 或任何形式的新建文件）\n",
    "- 修改文件（任何编辑操作）\n",
    "- 删除文件（如 rm）\n",
    "- 移动/复制文件（如 mv、cp）\n",
    "- 在任意目录（含 /tmp）创建临时文件\n",
    "- 使用重定向或管道将内容写入文件（>, >>, |）\n",
    "- 执行任何会改变系统状态的命令\n",
    "\n",
    "你的职责仅限：探索代码库并设计可执行计划。\n",
    "\n",
    "## 工作流程：\n",
    "1) 理解需求：聚焦用户目标与约束。\n",
    "2) 充分探索：识别现有架构、相似实现、关键调用链与约定。\n",
    "3) 方案设计：给出实现路径，并说明关键取舍。适当遵循已有范式。\n",
    "4) 细化计划：拆分步骤、依赖关系、执行顺序与潜在风险。\n",
    "\n",
    "如需使用 bash，仅允许只读命令（例如 ls、git status、git log、git diff、find、grep、",
    "cat、head、tail）。\n",
    "严禁使用 bash 执行：mkdir、touch、rm、cp、mv、git add、git commit、npm install、",
    "pip install，或任何创建/修改文件的命令。\n",
    "\n",
    "输出要求：在回答末尾必须给出\"Critical Files for Implementation\"，",
    "列出 3-5 个最关键文件路径。",
);

// ---------------------------------------------------------------------------
// Verification preset prompt
// ---------------------------------------------------------------------------

const VERIFICATION_AGENT_PROMPT_EN: &str = concat!(
    "You are an adversarial verification specialist. Your job is NOT to confirm that ",
    "implementation work looks correct — it is to try to BREAK it. You are the last ",
    "line of defense before results are reported to the user.\n",
    "\n",
    "=== CRITICAL CONSTRAINTS ===\n",
    "- You CANNOT create, modify, or delete project files. /tmp is allowed for ephemeral ",
    "test scripts.\n",
    "- Every check MUST have a \"Command run\" block with actual terminal output copied ",
    "verbatim.\n",
    "- You MUST end your final response with exactly one of:\n",
    "    VERDICT: PASS\n",
    "    VERDICT: FAIL\n",
    "    VERDICT: PARTIAL\n",
    "  No markdown bold, no punctuation after the verdict word, no variation in format.\n",
    "\n",
    "=== TWO FAILURE MODES TO RESIST ===\n",
    "\n",
    "1. Verification avoidance — reading code, narrating what you *would* test, then ",
    "writing PASS without running anything. Reading is NOT verification. Every claim ",
    "requires a command and its output.\n",
    "   - \"The code looks correct based on my reading\" → Run it and show the output.\n",
    "   - \"I can see the logic handles this case\" → Prove it with a command.\n",
    "\n",
    "2. Seduced by the first 80% — seeing a passing test suite or clean output and ",
    "stopping without probing edge cases.\n",
    "\n",
    "=== REQUIRED BASELINE (no exceptions) ===\n",
    "1. Read AGENTS.md / README / pyproject.toml / Makefile for build and test commands.\n",
    "2. Run the build — a broken build is an automatic FAIL.\n",
    "3. Run the project test suite — failing tests are an automatic FAIL.\n",
    "4. Run linters and type-checkers (ruff, mypy, etc.).\n",
    "5. Check for regressions in code paths related to the changed files.\n",
    "\n",
    "Test suite results are context, not evidence. The implementer is also an LLM — ",
    "its tests may rely on mocks, circular assertions, or happy-path coverage that ",
    "proves nothing end-to-end.\n",
    "\n",
    "=== VERIFICATION STRATEGY BY CHANGE TYPE ===\n",
    "\n",
    "Backend / API changes:\n",
    "→ Start the server → call endpoints (curl / httpie) → verify response *shapes*, ",
    "not just status codes → test error paths (malformed input, missing fields, wrong ",
    "types) → test authentication and authorization boundaries.\n",
    "\n",
    "CLI / script changes:\n",
    "→ Run with representative inputs → verify stdout, stderr, and exit codes → test ",
    "edge inputs (no args, empty string, boundary values, malformed) → verify --help ",
    "output is accurate.\n",
    "\n",
    "Library / package changes:\n",
    "→ Build → run test suite → import from a fresh context → verify exported names and ",
    "signatures match documentation and examples.\n",
    "\n",
    "Bug fixes:\n",
    "→ Reproduce the original bug FIRST → apply fix → verify it no longer occurs → run ",
    "regression check → inspect related code paths for side effects.\n",
    "\n",
    "Refactoring:\n",
    "→ Existing test suite must pass unchanged → verify public API surface is identical ",
    "(no added or removed exports) → spot-check observable behavior is the same.\n",
    "\n",
    "Infrastructure / config changes:\n",
    "→ Validate syntax → dry-run where available → confirm env vars are actually ",
    "referenced, not just defined.\n",
    "\n",
    "Data / ML pipeline changes:\n",
    "→ Run with sample input → verify output shape, schema, and types → test empty ",
    "input, single row, null/NaN → confirm row counts in match row counts out (no ",
    "silent data loss).\n",
    "\n",
    "Database migrations:\n",
    "→ Run migration up → verify schema matches intent → run migration down ",
    "(reversibility check) → test against data that already existed, not just an ",
    "empty database.\n",
    "\n",
    "=== REQUIRED ADVERSARIAL PROBES ===\n",
    "Before issuing PASS, run at least one of:\n",
    "- Boundary values: 0, -1, empty string, very long strings, unicode, MAX_INT\n",
    "- Idempotency: same mutating call twice — duplicate created? correct no-op? ",
    "wrong error?\n",
    "- Orphan operations: reference or delete IDs / resources that do not exist\n",
    "- Concurrency (where applicable): parallel calls to create-if-not-exists paths\n",
    "\n",
    "A report with only \"exits 0\" or \"returns 200\" checks is happy-path confirmation, ",
    "not verification.\n",
    "\n",
    "=== BEFORE ISSUING FAIL ===\n",
    "Check first:\n",
    "- Is there defensive code elsewhere that already handles this case?\n",
    "- Is this intentional behavior documented in AGENTS.md, comments, or commit ",
    "messages?\n",
    "- Is this a real limitation that cannot be fixed without breaking an external ",
    "contract?\n",
    "  If so, note it as an observation rather than a FAIL — an unfixable bug is not ",
    "actionable.\n",
    "\n",
    "=== MANDATORY OUTPUT FORMAT ===\n",
    "Every check must use this exact structure:\n",
    "\n",
    "### Check: [what you are verifying]\n",
    "**Command run:**\n",
    "  [exact command executed]\n",
    "**Output observed:**\n",
    "  [verbatim terminal output — do not paraphrase]\n",
    "**Result: PASS**\n",
    "\n",
    "or\n",
    "\n",
    "**Result: FAIL**\n",
    "Expected: [what should have happened]\n",
    "Actual: [what actually happened]\n",
    "\n",
    "A check WITHOUT a \"Command run\" block is treated as a SKIP, not a PASS.\n",
    "\n",
    "BAD example (never do this):\n",
    "### Check: Input validation\n",
    "**Result: PASS**\n",
    "Evidence: Reviewed the handler. The logic correctly validates input before ",
    "processing.\n",
    "(No command run. Reading code is not verification.)\n",
    "\n",
    "=== FINAL VERDICT ===\n",
    "VERDICT: PASS    — all checks passed, adversarial probes survived\n",
    "VERDICT: FAIL    — include what failed, exact error output, and reproduction steps\n",
    "VERDICT: PARTIAL — environmental limitation only (tool unavailable, service cannot ",
    "start);\n",
    "                   NOT \"I am unsure whether this is a bug\"\n",
    "\n",
    "Use the literal string VERDICT: followed by exactly one of PASS, FAIL, PARTIAL.\n",
    "No markdown. No punctuation after the word. No variation.",
);

const VERIFICATION_AGENT_PROMPT_CN: &str = concat!(
    "你是一位对抗性验证专家。你的职责不是确认实现看起来正确——而是尝试将其破坏。",
    "你是在结果上报用户之前的最后一道防线。\n",
    "\n",
    "=== 关键约束 ===\n",
    "- 你不能创建、修改或删除项目文件。/tmp 可用于临时测试脚本。\n",
    "- 每项检查必须包含\"执行命令\"块，并逐字粘贴实际终端输出。\n",
    "- 你必须以以下之一结束最终回复：\n",
    "    VERDICT: PASS\n",
    "    VERDICT: FAIL\n",
    "    VERDICT: PARTIAL\n",
    "  不得加粗，不得在判决词后加标点，不得有任何格式变体。\n",
    "\n",
    "=== 必须抵制的两种失败模式 ===\n",
    "\n",
    "1. 验证规避——阅读代码、描述\"本应测试什么\"，然后在未实际运行任何内容的情况下",
    "写下 PASS。阅读代码不等于验证。每项断言都需要一条命令及其输出为证。\n",
    "\n",
    "2. 被前 80% 迷惑——看到测试通过或输出整洁就停下，而不深入探测边界情况。\n",
    "\n",
    "=== 必要基准步骤（不得省略）===\n",
    "1. 阅读 AGENTS.md / README / pyproject.toml / Makefile，获取构建和测试命令。\n",
    "2. 运行构建——构建失败即自动 FAIL。\n",
    "3. 运行项目测试套件——测试失败即自动 FAIL。\n",
    "4. 运行代码检查和类型检查（ruff、mypy 等）。\n",
    "5. 检查与已更改文件相关的代码路径是否存在回归。\n",
    "\n",
    "测试套件结果只是背景，不是证据。实现者也是 LLM——其测试可能依赖 mock、",
    "循环断言或仅覆盖正常路径，无法端到端证明任何问题。\n",
    "\n",
    "=== 按变更类型划分的验证策略 ===\n",
    "\n",
    "后端 / API 变更：\n",
    "→ 启动服务器 → 调用端点（curl / httpie）→ 验证响应*结构*（不只是状态码）",
    "→ 测试错误路径（格式错误、缺失字段、类型错误）→ 测试认证和授权边界。\n",
    "\n",
    "CLI / 脚本变更：\n",
    "→ 使用典型输入运行 → 验证 stdout、stderr 和退出码 → 测试边界输入",
    "（无参数、空字符串、边界值、格式错误）→ 确认 --help 输出准确。\n",
    "\n",
    "库 / 包变更：\n",
    "→ 构建 → 运行测试套件 → 在全新上下文中导入 → 验证导出名称和签名与文档及示例一致。\n",
    "\n",
    "缺陷修复：\n",
    "→ 先重现原始缺陷 → 应用修复 → 验证缺陷不再出现 → 运行回归检查 → ",
    "检查相关代码路径的副作用。\n",
    "\n",
    "重构：\n",
    "→ 现有测试套件必须原样通过 → 验证公开 API 表面完全一致（无新增或删除导出）",
    "→ 抽查可观测行为保持不变。\n",
    "\n",
    "基础设施 / 配置变更：\n",
    "→ 验证语法 → 在可用时进行试运行 → 确认环境变量被实际引用，而非只是定义。\n",
    "\n",
    "数据 / ML 流水线变更：\n",
    "→ 使用示例输入运行 → 验证输出的 shape、schema 和类型 → 测试空输入、单行数据、",
    "null/NaN→ 确认输入行数与输出行数匹配（无静默数据丢失）。\n",
    "\n",
    "数据库迁移：\n",
    "→ 运行向上迁移 → 验证 schema 符合意图 → 运行向下迁移（可逆性检查）→ ",
    "针对已存在的数据而非空数据库进行测试。\n",
    "\n",
    "=== 必要的对抗性探测 ===\n",
    "在发出 PASS 之前，至少运行以下之一：\n",
    "- 边界值：0、-1、空字符串、极长字符串、Unicode、MAX_INT\n",
    "- 幂等性：同一变更操作执行两次——是否创建了重复项？是否正确地无操作？是否报错？\n",
    "- 孤立操作：引用或删除不存在的 ID / 资源\n",
    "- 并发（如适用）：对\"不存在则创建\"路径发起并行调用\n",
    "\n",
    "仅包含\"退出码 0\"或\"返回 200\"的报告是正常路径确认，而非验证。\n",
    "\n",
    "=== 发出 FAIL 之前 ===\n",
    "先检查：\n",
    "- 是否有其他地方的防御性代码实际上已处理该情况？\n",
    "- 这是否是 AGENTS.md、注释或提交信息中记录的预期行为？\n",
    "- 这是否是真实限制，但在不破坏外部契约的情况下无法修复？\n",
    "  若是，将其作为观察结论而非 FAIL——无法修复的缺陷不具有可操作性。\n",
    "\n",
    "=== 强制输出格式 ===\n",
    "每项检查必须使用以下结构：\n",
    "\n",
    "### 检查：[正在验证的内容]\n",
    "**执行命令：**\n",
    "  [实际执行的确切命令]\n",
    "**观察到的输出：**\n",
    "  [逐字粘贴的终端输出——不得转述]\n",
    "**结果：PASS**\n",
    "\n",
    "或\n",
    "\n",
    "**结果：FAIL**\n",
    "预期：[应发生的情况]\n",
    "实际：[实际发生的情况]\n",
    "\n",
    "没有\"执行命令\"块的检查被视为跳过，而非 PASS。\n",
    "\n",
    "=== 最终判决 ===\n",
    "VERDICT: PASS    — 所有检查通过，对抗性探测均通过\n",
    "VERDICT: FAIL    — 包括失败内容、确切错误输出和复现步骤\n",
    "VERDICT: PARTIAL — 仅限环境限制（工具不可用、服务无法启动）；\n",
    "                   不适用于\"我不确定这是否是缺陷\"的情况\n",
    "\n",
    "使用字面字符串 VERDICT: 后接 PASS、FAIL 或 PARTIAL 之一。\n",
    "不加 Markdown 格式，判决词后不加标点，不得有任何格式变体。",
);

/// A built-in sub-agent preset.
///
/// Python parity: `build_*_agent_config` of `openjiuwen.harness.subagents`. Presets are
/// plain data; use [`build_preset`] to turn one into a [`SubAgentConfig`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum SubAgentPreset {
    /// General coding sub-agent backed by the `code_agent` factory.
    #[default]
    Code,
    /// Read-only codebase navigation sub-agent.
    Explore,
    /// Read-only architecture and implementation planning sub-agent.
    Plan,
    /// Research and investigation sub-agent.
    Research,
    /// Adversarial post-implementation verification sub-agent.
    Verification,
    /// Browser automation sub-agent; requires the `browser` cargo feature.
    #[cfg(feature = "browser")]
    Browser,
    /// Computer-use automation sub-agent; requires the `cua` cargo feature.
    #[cfg(feature = "cua")]
    Cua,
    /// Mobile GUI automation sub-agent; requires the `mobile` cargo feature.
    #[cfg(feature = "mobile")]
    Mobile,
}

/// The presets every build provides, in registry order.
const BUILTIN_PRESETS: &[SubAgentPreset] = &[
    SubAgentPreset::Code,
    SubAgentPreset::Explore,
    SubAgentPreset::Plan,
    SubAgentPreset::Research,
    SubAgentPreset::Verification,
];

impl SubAgentPreset {
    /// Returns the Python type name of this preset.
    ///
    /// The names are the `AgentCard.name` values Python uses, so they double as the
    /// registry keys of [`crate::subagents::SubAgentRegistry`].
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Code => "code_agent",
            Self::Explore => "explore_agent",
            Self::Plan => "plan_agent",
            Self::Research => "research_agent",
            Self::Verification => "verification_agent",
            #[cfg(feature = "browser")]
            Self::Browser => "browser_agent",
            #[cfg(feature = "cua")]
            Self::Cua => "cua_agent",
            #[cfg(feature = "mobile")]
            Self::Mobile => "mobile_gui_agent",
        }
    }

    /// Returns the presets that are always available, in Python registration order.
    ///
    /// Feature-gated presets (`browser`, `cua`, `mobile`) are not part of this list.
    #[must_use]
    pub const fn builtin() -> &'static [Self] {
        BUILTIN_PRESETS
    }
}

/// Selects the Chinese or English variant of a bilingual constant.
const fn bilingual(chinese: bool, cn: &'static str, en: &'static str) -> &'static str {
    if chinese {
        cn
    } else {
        en
    }
}

/// Builds the error of a preset whose cargo feature is enabled but whose agent is deferred.
#[cfg(any(feature = "browser", feature = "cua", feature = "mobile"))]
fn feature_error(preset: SubAgentPreset, feature: &str) -> Error {
    Error::unsupported(format!(
        "subagent preset '{}' requires the '{feature}' cargo feature",
        preset.as_str()
    ))
}

/// Builds the [`SubAgentConfig`] of `preset` for `language`.
///
/// Python parity: `build_code_agent_config`, `build_explore_agent_config`,
/// `build_plan_agent_config`, `build_research_agent_config` and
/// `build_verification_agent_config`. `language` is resolved with
/// [`resolve_language`], so an absent or unknown tag falls back to Chinese, exactly like
/// Python's `resolve_language`.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Unsupported`] for a feature-gated preset
/// whose cargo feature is enabled: the `browser`, `cua` and `mobile` agents are deferred.
pub fn build_preset(preset: SubAgentPreset, language: &str) -> Result<SubAgentConfig> {
    let resolved = resolve_language(Some(language));
    let chinese = resolved == "cn";

    // Python's `DEFAULT_*_DESCRIPTION` lookup falls back to Chinese, which is also what
    // `resolve_language` returns for an unknown tag.
    let (description, system_prompt) = match preset {
        SubAgentPreset::Code => (
            bilingual(
                chinese,
                CODE_AGENT_DESCRIPTION_CN,
                CODE_AGENT_DESCRIPTION_EN,
            ),
            bilingual(chinese, CODE_AGENT_PROMPT_CN, CODE_AGENT_PROMPT_EN),
        ),
        SubAgentPreset::Explore => (
            bilingual(
                chinese,
                EXPLORE_AGENT_DESCRIPTION_CN,
                EXPLORE_AGENT_DESCRIPTION_EN,
            ),
            bilingual(chinese, EXPLORE_AGENT_PROMPT_CN, EXPLORE_AGENT_PROMPT_EN),
        ),
        SubAgentPreset::Plan => (
            bilingual(
                chinese,
                PLAN_AGENT_DESCRIPTION_CN,
                PLAN_AGENT_DESCRIPTION_EN,
            ),
            bilingual(chinese, PLAN_AGENT_PROMPT_CN, PLAN_AGENT_PROMPT_EN),
        ),
        SubAgentPreset::Research => (
            bilingual(
                chinese,
                RESEARCH_AGENT_DESCRIPTION_CN,
                RESEARCH_AGENT_DESCRIPTION_EN,
            ),
            bilingual(chinese, RESEARCH_AGENT_PROMPT_CN, RESEARCH_AGENT_PROMPT_EN),
        ),
        SubAgentPreset::Verification => (
            bilingual(
                chinese,
                VERIFICATION_AGENT_DESCRIPTION_CN,
                VERIFICATION_AGENT_DESCRIPTION_EN,
            ),
            bilingual(
                chinese,
                VERIFICATION_AGENT_PROMPT_CN,
                VERIFICATION_AGENT_PROMPT_EN,
            ),
        ),
        #[cfg(feature = "browser")]
        SubAgentPreset::Browser => return Err(feature_error(preset, "browser")),
        #[cfg(feature = "cua")]
        SubAgentPreset::Cua => return Err(feature_error(preset, "cua")),
        #[cfg(feature = "mobile")]
        SubAgentPreset::Mobile => return Err(feature_error(preset, "mobile")),
    };

    let factory_name = match preset {
        SubAgentPreset::Code => Some(CODE_AGENT_FACTORY_NAME.to_owned()),
        SubAgentPreset::Research => Some(RESEARCH_AGENT_FACTORY_NAME.to_owned()),
        _ => None,
    };

    // Python only leaves Explore and Plan unrestricted.
    let restrict_to_work_dir = !matches!(preset, SubAgentPreset::Explore | SubAgentPreset::Plan);

    let card = AgentCard::new(preset.as_str()).with_description(description);
    Ok(SubAgentConfig::new(card)
        .with_system_prompt(Some(system_prompt.to_owned()))
        .with_language(Some(resolved))
        .with_max_iterations(Some(PRESET_MAX_ITERATIONS))
        .with_enable_task_loop(false)
        .with_enable_plan_mode(false)
        .with_parallel_tool_calls(true)
        .with_restrict_to_work_dir(restrict_to_work_dir)
        .with_factory_name(factory_name)
        .with_subagent_type(Some(preset.as_str().to_owned())))
}

/// Builds the configuration of the general-purpose fallback sub-agent.
///
/// Python parity: the `general-purpose` spec injected by
/// `_inject_general_purpose_subagent` in `openjiuwen/harness/factory.py`. The description
/// is the Python Chinese default, the system prompt is inherited from the parent and the
/// iteration cap is inherited too (`None`), so the agent runs with the parent's settings.
#[must_use]
pub fn general_purpose_config() -> SubAgentConfig {
    let card =
        AgentCard::new(GENERAL_PURPOSE_NAME).with_description(GENERAL_PURPOSE_DESCRIPTION_CN);
    SubAgentConfig::new(card)
        .with_restrict_to_work_dir(false)
        .with_subagent_type(Some(GENERAL_PURPOSE_NAME.to_owned()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use super::*;

    /// Builds the config of `preset` in English and asserts it succeeds.
    fn preset(preset: SubAgentPreset) -> SubAgentConfig {
        build_preset(preset, "en").expect("built-in preset must build")
    }

    #[test]
    fn as_str_uses_python_type_names() {
        assert_eq!(SubAgentPreset::Code.as_str(), "code_agent");
        assert_eq!(SubAgentPreset::Explore.as_str(), "explore_agent");
        assert_eq!(SubAgentPreset::Plan.as_str(), "plan_agent");
        assert_eq!(SubAgentPreset::Research.as_str(), "research_agent");
        assert_eq!(SubAgentPreset::Verification.as_str(), "verification_agent");
    }

    #[test]
    fn builtin_lists_the_five_always_available_presets_in_order() {
        assert_eq!(
            SubAgentPreset::builtin(),
            &[
                SubAgentPreset::Code,
                SubAgentPreset::Explore,
                SubAgentPreset::Plan,
                SubAgentPreset::Research,
                SubAgentPreset::Verification,
            ]
        );
        assert_eq!(SubAgentPreset::default(), SubAgentPreset::Code);
    }

    #[test]
    fn every_builtin_preset_shares_the_common_loop_settings() {
        for preset in SubAgentPreset::builtin() {
            let config = build_preset(*preset, "en").unwrap();
            assert_eq!(config.agent_card.name, preset.as_str());
            assert_eq!(config.subagent_type.as_deref(), Some(preset.as_str()));
            assert_eq!(config.max_iterations, Some(50));
            assert!(!config.enable_task_loop);
            assert!(!config.enable_plan_mode);
            assert!(config.parallel_tool_calls);
            assert_eq!(config.language.as_deref(), Some("en"));
            assert!(config
                .system_prompt
                .is_some_and(|prompt| !prompt.is_empty()));
            assert_eq!(config.tools, Vec::<String>::new());
            assert_eq!(config.skills, Vec::<String>::new());
            assert_eq!(config.factory_kwargs, BTreeMap::new());
            assert!(config.display_name.is_none());
            assert!(config.role.is_none());
        }
    }

    #[test]
    fn restrict_to_work_dir_is_off_only_for_explore_and_plan() {
        assert!(preset(SubAgentPreset::Code).restrict_to_work_dir);
        assert!(!preset(SubAgentPreset::Explore).restrict_to_work_dir);
        assert!(!preset(SubAgentPreset::Plan).restrict_to_work_dir);
        assert!(preset(SubAgentPreset::Research).restrict_to_work_dir);
        assert!(preset(SubAgentPreset::Verification).restrict_to_work_dir);
    }

    #[test]
    fn factory_name_is_set_only_for_code_and_research() {
        assert_eq!(
            preset(SubAgentPreset::Code).factory_name.as_deref(),
            Some("code_agent")
        );
        assert_eq!(
            preset(SubAgentPreset::Research).factory_name.as_deref(),
            Some("research_agent")
        );
        for other in [
            SubAgentPreset::Explore,
            SubAgentPreset::Plan,
            SubAgentPreset::Verification,
        ] {
            assert_eq!(preset(other).factory_name, None);
        }
    }

    #[test]
    fn language_selects_the_english_or_chinese_variant() {
        let english = preset(SubAgentPreset::Code);
        assert!(english
            .agent_card
            .description
            .contains("senior software engineer"));
        assert!(english
            .system_prompt
            .as_deref()
            .is_some_and(|prompt| prompt.contains("You are an AI Coding Agent.")));

        for tag in ["cn", "zh-CN", "klingon"] {
            let chinese = build_preset(SubAgentPreset::Code, tag).unwrap();
            assert_eq!(chinese.language.as_deref(), Some("cn"), "tag {tag}");
            assert!(
                chinese.agent_card.description.contains("资深软件工程师"),
                "tag {tag}"
            );
            assert!(chinese
                .system_prompt
                .as_deref()
                .is_some_and(|prompt| prompt.contains("你是一个 AI 编程助手")));
        }
    }

    #[test]
    fn english_and_chinese_descriptions_differ_per_preset() {
        for preset in SubAgentPreset::builtin() {
            let english = build_preset(*preset, "en").unwrap();
            let chinese = build_preset(*preset, "cn").unwrap();
            assert_ne!(
                english.agent_card.description,
                chinese.agent_card.description
            );
            assert_ne!(english.system_prompt, chinese.system_prompt);
        }
    }

    #[test]
    fn research_prompt_only_reports_the_final_results() {
        let config = preset(SubAgentPreset::Research);
        let prompt = config.system_prompt.unwrap();
        assert!(prompt.contains("You are a research assistant"));
        assert!(prompt.contains("Only return the final research results."));
    }

    #[test]
    fn explore_prompt_is_strictly_read_only() {
        let config = preset(SubAgentPreset::Explore);
        let prompt = config.system_prompt.unwrap();
        assert!(prompt.contains("=== IMPORTANT: READ-ONLY OPERATION ==="));
        assert!(prompt.contains("The following actions are forbidden"));
        assert!(prompt.contains("You have no write-capable tools"));
        for tool in ["glob", "grep", "read_file", "list_files", "bash"] {
            assert!(prompt.contains(tool), "missing tool guidance for {tool}");
        }
        assert!(prompt.contains("read-only shell inspection"));
        assert!(prompt.contains("plain text reply"));
    }

    #[test]
    fn plan_prompt_is_read_only_and_ends_with_critical_files() {
        let config = preset(SubAgentPreset::Plan);
        let prompt = config.system_prompt.unwrap();
        assert!(prompt.contains("=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ==="));
        assert!(prompt.contains("STRICTLY PROHIBITED"));
        assert!(prompt.contains("Critical Files for Implementation"));
        assert!(prompt.contains("3-5 most critical file paths"));

        let chinese = build_preset(SubAgentPreset::Plan, "cn").unwrap();
        let chinese_prompt = chinese.system_prompt.unwrap();
        assert!(chinese_prompt.contains("只读模式"));
        assert!(chinese_prompt.contains("Critical Files for Implementation"));
    }

    #[test]
    fn verification_prompt_forces_a_verdict_and_evidence() {
        let config = preset(SubAgentPreset::Verification);
        let prompt = config.system_prompt.unwrap();
        assert!(prompt.contains("adversarial verification specialist"));
        assert!(prompt.contains("Command run"));
        assert!(prompt.contains("VERDICT: PASS"));
        assert!(prompt.contains("VERDICT: FAIL"));
        assert!(prompt.contains("VERDICT: PARTIAL"));
        assert!(prompt.contains("/tmp is allowed"));

        let chinese = build_preset(SubAgentPreset::Verification, "cn").unwrap();
        let chinese_prompt = chinese.system_prompt.unwrap();
        assert!(chinese_prompt.contains("对抗性验证专家"));
        assert!(chinese_prompt.contains("执行命令"));
        assert!(chinese_prompt.contains("VERDICT: PASS"));
        assert!(chinese_prompt.contains("VERDICT: FAIL"));
        assert!(chinese_prompt.contains("VERDICT: PARTIAL"));
    }

    #[test]
    fn general_purpose_matches_the_python_fallback() {
        let config = general_purpose_config();
        assert_eq!(config.agent_card.name, GENERAL_PURPOSE_NAME);
        assert!(config.agent_card.description.contains("通用型子代理"));
        assert!(config.agent_card.description.contains("独立上下文"));
        assert!(config.system_prompt.is_none());
        assert!(!config.restrict_to_work_dir);
        assert_eq!(config.max_iterations, None);
        assert_eq!(config.subagent_type.as_deref(), Some(GENERAL_PURPOSE_NAME));
        assert!(config.factory_name.is_none());
    }

    #[test]
    fn presets_round_trip_through_json_like_the_python_config() {
        let config = preset(SubAgentPreset::Code);
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["agent_card"]["name"], "code_agent");
        assert_eq!(json["subagent_type"], "code_agent");
        assert_eq!(json["language"], "en");
        assert_eq!(json["restrict_to_work_dir"], true);
        let restored: SubAgentConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored, config);
    }

    #[cfg(feature = "browser")]
    #[test]
    fn browser_preset_reports_its_missing_feature() {
        let error = build_preset(SubAgentPreset::Browser, "en").unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Unsupported(_)
        ));
        assert!(error.to_string().contains("browser_agent"));
        assert!(error.to_string().contains("'browser' cargo feature"));
        assert_eq!(SubAgentPreset::Browser.as_str(), "browser_agent");
    }

    #[cfg(feature = "cua")]
    #[test]
    fn cua_preset_reports_its_missing_feature() {
        let error = build_preset(SubAgentPreset::Cua, "en").unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Unsupported(_)
        ));
        assert!(error.to_string().contains("cua_agent"));
        assert!(error.to_string().contains("'cua' cargo feature"));
        assert_eq!(SubAgentPreset::Cua.as_str(), "cua_agent");
    }

    #[cfg(feature = "mobile")]
    #[test]
    fn mobile_preset_reports_its_missing_feature() {
        let error = build_preset(SubAgentPreset::Mobile, "en").unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Unsupported(_)
        ));
        assert!(error.to_string().contains("mobile_gui_agent"));
        assert!(error.to_string().contains("'mobile' cargo feature"));
        assert_eq!(SubAgentPreset::Mobile.as_str(), "mobile_gui_agent");
    }
}
