//! The disclosure prompt bank and the chat messages built from one exposed fragment.
//!
//! Ports the prompt half of Python
//! `openjiuwen.symphony.retrieval.search.runtime.render.disclosure` (v0.1.20) together with
//! the Chinese prompt text of
//! `openjiuwen/symphony/retrieval/common/prompts/retrieval_disclosure.yaml` (the reference's
//! prompt text is Chinese by design and is kept verbatim here).
//!
//! # Variant selection
//!
//! The reference composes the system prompt from a shared preamble plus a structure block
//! and an output block. This port keeps exactly that structure:
//!
//! | `(compact, flat)` | structure block | output block |
//! | --- | --- | --- |
//! | `(true, false)` | [`STRUCTURE_CODES`] | [`OUTPUT_CODES`] |
//! | `(true, true)` | [`STRUCTURE_CODES_FLAT`] | [`COMPACT_FLAT_OUTPUT_BLOCK`] |
//! | `(false, true)` | [`STRUCTURE_NAMES_FLAT`] | [`OUTPUT_NAMES_FLAT`] |
//! | `(false, false)` | [`STRUCTURE_NAMES`] | [`OUTPUT_NAMES_TREE`] |
//!
//! The preamble is [`SHARED_TOP2`] when the resolved `top_k` is exactly `2` (the tuned
//! experiment default) and [`SHARED`] otherwise, with `{top_k}`, `{structure_block}`,
//! `{candidate_block}` and `{output_block}` substituted in one pass. Names mode with flat
//! list mode and any other `top_k` uses [`LEGACY_FLAT_NON_COMPACT_SYSTEM_PROMPT`] plus the
//! candidate block instead — the reference's legacy flat prompt, whose `top_k` is fixed at
//! 5 in the text itself.
//!
//! # Message shape
//!
//! [`build_disclosure_messages`] returns exactly two messages: a `system` message holding
//! the whole prompt bank plus the candidate block (`# 候选列表` followed by
//! `<CANDIDATE_TREE>…</CANDIDATE_TREE>`, or the `Available options:` block in flat mode), and
//! a `user` message holding the request marker and the query inside `<USER_REQUEST>…`.
//! [`build_disclosure_prompt_parts`] additionally returns the prefix-only messages a prefix
//! cache can reuse, the request suffix, and the cache identity.
//!
//! # Deviation: the request marker is deterministic
//!
//! Python prefixes every full prompt with `[req:<16 hex of a fresh uuid4>]` so two
//! identical-looking user messages never dedupe upstream. This crate forbids RNG- and
//! clock-dependent output, so the marker is the first 16 hex characters of the SHA-256 over
//! the query, the fragment fingerprint and `top_k`. Identical inputs therefore produce
//! identical prompts — which is what the tests here assert — while two requests that differ
//! in query, exposure or `top_k` still get distinct markers.
//!
//! # Other documented deviations
//!
//! - The reference memoizes the static prompt with an `lru_cache` and caches the prompt bank
//!   in a module-level function attribute; the constants here are compile-time text, so no
//!   cache and no filesystem read is involved.
//! - The reference passes an unused `{candidate_region}` format argument; this port omits
//!   it, because no prompt key interpolates it.
//! - [`OUTPUT_CODES_FLAT`] is ported for parity but is not selected by any variant (the
//!   reference's compact flat branch uses `compact_flat_output_block`); its doubled
//!   `{{`/`}}` format escapes are already expanded here.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::flow::model::{sha256_hex, sha256_short};
use crate::retrieval::search::flow::{ChatMessage, ExposedFragment};

/// The prompt-bank version tag of this disclosure renderer.
pub const DISCLOSURE_PROMPT_VERSION: &str = "symphony-disclosure-v1";

/// The prefix of the version string a fragment's cache id starts with.
pub const CACHE_ID_VERSION: &str = "progressive-disclosure-v1";

/// The header of the candidate block embedded in the system message.
pub const CANDIDATE_HEADER: &str = "# 候选列表";

/// The opening tag of the nested candidate block.
pub const CANDIDATE_TREE_OPEN: &str = "<CANDIDATE_TREE>";

/// The closing tag of the nested candidate block.
pub const CANDIDATE_TREE_CLOSE: &str = "</CANDIDATE_TREE>";

/// The label that opens the flat candidate block.
pub const FLAT_CANDIDATE_HEADER: &str = "Available options:";

/// The opening tag of the user request.
pub const USER_REQUEST_OPEN: &str = "<USER_REQUEST>";

/// The closing tag of the user request.
pub const USER_REQUEST_CLOSE: &str = "</USER_REQUEST>";

/// The user-message prefix of tree mode.
pub const USER_REQUEST_PREFIX: &str = "<USER_REQUEST>\n";

/// The user-message prefix of flat mode.
pub const FLAT_USER_PREFIX: &str = "User request:\n";

/// How many hex characters the per-call request marker carries.
pub const REQUEST_MARKER_HEX: usize = 16;

/// The `top_k` the legacy flat prompt text names.
pub const LEGACY_FLAT_TOP_K: usize = 5;

/// The shared preamble of the nested and compact prompt variants.
pub const SHARED: &str = r#"
/no_think

# 背景与目标：
- 系统的最终目标是解决用户的问题。
- 系统通过调用 Candidate skill 来执行任务。
- 你的职责是：组建一个最优 skill 小组，以端到端闭环用户的任务。

注意：
- 你只做“候选筛选”，不做规划、不做执行。


# 输出规则：

1. 总共输出至多 {top_k} 个 Candidate skill 节点 id

2. 该结果必须是：
   - 最优 Candidate skill 组合
   - 该组合能够完整闭环用户任务

3. 至多输出 {top_k} 行


# 选择规则：

## 1. 决策目标
- 组合应覆盖用户的核心目标、必要前置条件和关键辅助能力。
- 不要为了凑满 {top_k} 选择低相关或无关 skill。

## 2. 约束优先级
- 用户的显式约束优先级最高，包括时间、范围、条件、格式、工具类型、排除项等。
- 如果用户明确排除某类能力，不能选择对应 skill。
- 例如：用户明确不要 PPT，就不能选择 PPT、幻灯片、演示文稿相关 skill。

## 3. 意图理解
- 先识别用户的显式意图，再补全与任务闭环强相关的隐式意图。
- 当用户只表达状态、场景、计划或意向时，不要轻易返回空；应保守预测可帮助用户的潜在意图。
- 潜在意图必须与用户原话高度相关。

## 4. 组合补全
- 围绕用户目标，补全完成任务所需的关键上下游能力。
- 优先选择能形成闭环的互补 skill，而不是只选择表面最相似的 skill。
- 只补全与用户目标强相关、能够实际提升任务完成度的 skill。
- 不要为了覆盖更多方面而加入弱相关 skill。

## 5. Notes
- 务必识别真实隐含意图，避免被表面词误导。
- 例如：“生成一张地图”更可能需要图像生成，而不是地点查询。
- 用户要求根据时间线、历史路线、事件节点生成、绘制或标注地图/路线图时，应选择图像生成/视觉设计类 skill；不要选择地图查询、导航、定位或地图 JSAPI，除非用户明确要查询真实地点、规划路线或开发地图应用。
- 不要选择与用户任务无关的 skill。
- 必须严格遵守用户的显式约束（如时间、范围、条件等）
- 要选出**能够完整闭环用户意图的 skill 组合**，包括用户的显式意图和隐式意图。
  - 例如，用户有出行意图时，不仅考虑出行规划，也要选择查询天气相关 skill
  - 用户有饮食、购物等意图时，不仅直接选择对应的出行助手，也要选择美团红包优惠
  - 用户要搭建网站时，不仅直接选择 html 生成与前端开发，也可以考虑艺术创作
  - 务必确认用户的真实隐含意图，例如“生成一张地图”实质上对应于基于 AI 的图像生成，而非查询地图。
  - 不要附带任何无关的 skill，例如，用户不需要 PPT 时不要选择 PPT 相关技能。用户需要生成地图而非查询地点时，不要选择地图相关技能。
  - 用户明确不要某类 skill 时，不要选择对应的 skill。

{structure_block}

{candidate_block}

# 输出 id 说明：

{output_block}"#;

/// The shared preamble of the `top_k == 2` variant.
pub const SHARED_TOP2: &str = r#"
# 背景与目标：
- 系统的最终目标是解决用户的问题。
- 系统通过调用 Candidate skill 来执行任务。
- 你的职责是：从候选中选择 2 个 skill，组成能够闭环当前用户请求的执行小组。

注意：
- 你只做“候选筛选”，不做规划、不做执行。
- 默认必须输出 2 个候选；不要因为只找到一个最相似 skill 就停止。
- 少召回一个真正有用的 skill，比召回一个弱相关但仍可解释的辅助 skill 更糟。
- 第二个 skill 不要求和用户请求字面最相似，但必须能放进同一个任务链路中。
- 不要因为用户请求复杂，就选择用于创建、管理或调度其他 skill/agent 的候选；只有当这类候选本身就是用户明示目标时才选择。
- 候选在列表或树中的出现顺序没有相关性含义；不要因为候选靠前就选择它。
- 第一个候选不是默认项；如果不能用用户请求中的目标对象或动作解释它，禁止输出第一个候选。


# 输出规则：

1. 默认输出 2 个 Candidate skill 节点 id

2. 这 2 个 skill 应组成最优执行小组：
   - 第一个 skill 覆盖用户的主任务
   - 第二个 skill 覆盖同一任务链路中的前置、后置、输入理解、信息查询、内容生成、格式化、保存、发送、上下文补充或执行辅助能力
   - 两者共同提高完成用户请求的把握

3. 必须尽量输出 2 行，每行 1 个 id；只有当所有候选中完全不存在可放入同一任务链路的第二个 skill 时，才输出 1 行


# 选择规则：

## 1. 决策目标
- 目标不是找“最像 query 的单个 skill”，而是找“最可能完整完成任务的 2 个 skill 小组”。
- 先选择最能直接完成用户主任务的 skill。
- 再选择一个能补全任务链路的 skill。
- 如果用户请求包含多个动作、多个阶段、多个输入或多个输出格式，应优先用 2 个 skill 分别覆盖这些部分。

## 2. 第二个 skill 的召回规则
- 第二个 skill 可以是以下任一类型：
  - 获取或搜索完成任务所需的信息
  - 读取、识别、理解输入材料
  - 分析、计算、对比或总结内容
  - 生成用户要求的目标格式
  - 保存、上传、发送、同步或自动执行结果
  - 提供天气、地图、出行、优惠、健康数据等关键上下文
- 第二个 skill 只要能被用户请求的目标、动作、输入、输出或场景解释，就应该输出。
- 不要因为第二个 skill 不是字面最匹配而省略它。
- 不要输出两个功能完全重复的 skill；但如果两个候选分别覆盖同一任务的不同阶段，可以同时选择。

## 3. 无关项过滤
- 禁止选择与用户任务链路无关的 skill。
- 禁止用泛化 agent、skill 管理、内部治理、通用记忆、通用规划类候选填充第二个位置，除非用户明确要求这类能力。
- 如果某个候选只能通过非常牵强的联想解释，不能选择。
- 如果一个候选只是因为类别相近、名字常见或位置靠前而相关，不能选择。

## 4. 约束优先级
- 用户的显式约束优先级最高，包括时间、范围、条件、格式、工具类型、排除项等。
- 如果用户明确排除某类能力，不能选择对应 skill。
- 例如：用户明确不要 PPT，就不能选择 PPT、幻灯片、演示文稿相关 skill。

## 5. 意图理解
- 先识别用户的显式意图，再补全与任务闭环强相关的隐式意图。
- 当用户只表达状态、场景、计划或意向时，不要轻易只输出 1 个；应保守召回一个可帮助闭环的辅助 skill。
- 务必区分“查询已有信息”和“生成新内容”。
- 务必区分“读取/理解输入材料”和“创作/编辑输出材料”。
- 务必识别真实隐含意图，避免被表面词误导。

## 6. Notes
- 用户有出行或户外计划时，除旅行/路线 skill 外，天气通常是合理的第二候选。
- 用户有饮食、餐厅、门店、优惠相关意图时，美团优惠/券通常是合理的第二候选。
- 用户要生成网站、页面、展览展示、视觉作品时，网页/前端和图片/视觉创作可以形成互补。
- 用户要处理图片、截图、照片中的信息时，应优先考虑图像理解/识别；只有用户要求画、生成、编辑、美化图片时，才选择图像生成/设计。
- 用户要搜索、查询、对比真实产品或外部事实时，应选择搜索/调研类 skill；如果还要求表格、文档、邮件等输出，再选择对应输出 skill。
- 用户询问真实股票行情、走势、拐点、建仓或投资分析时，选择行情/金融数据/预测类 skill；只有用户明确要模拟交易、虚拟持仓或投资练习时，才选择股票模拟器。
- 用户要求根据时间线、历史路线、事件节点生成、绘制或标注地图/路线图时，应选择图像创作类 skill；不要选择地图查询、导航、定位或地图 JSAPI，除非用户明确要查询真实地点、规划路线或开发地图应用。
- 用户询问家中空间或智能家居物理设备的状态或控制时，应选择智能家居/家庭设备状态控制类 skill；例如灯光、空调、电视、路由器、门锁、传感器。不要用通用 APP 操控替代；也不要把运动健康、手机 App 数据或个人健康数据归到智能家居。
- 用户提到录音、语音、会议音频、语音转文字或从音频中找信息时，优先选择音频转写/听写类 skill；不要用视频剪辑、播客生成或图片识别替代。
- 用户要查询真实产品参数、价格、座位图、已有图片或外部事实时，信息获取 skill 是必要能力；不要用图像生成、表格、PPT 或文档输出 skill 替代搜索/调研/图片搜索。
- 用户要查看已有运动健康数据、心率、血压、睡眠或心理数据时，优先选择健康数据类 skill；不要用习惯打卡替代已有数据查询。
- 用户明确提到天气、气象、户外条件或气象风险时，天气通常是必要候选之一。
- 用户表达家中空间、睡眠环境或物理设备状态/控制时，优先考虑智能家居；设置闹钟只在用户要求提醒、叫醒或定时时选择。
- 不要附带与用户任务无关的 skill。

{structure_block}

{candidate_block}

# 输出 id 说明：

{output_block}"#;

/// The candidate-format block of compact tree mode.
pub const STRUCTURE_CODES: &str = r#"# 候选格式：
- <CANDIDATE_TREE> 中的 Category 表示分组，不可选择。
- <CANDIDATE_TREE> 中的 NAME [id: X] 表示候选 skill，可选择，其中 X 是候选 id。
- 缩进表示候选与分组之间的从属关系。"#;

/// The candidate-format block of names tree mode.
pub const STRUCTURE_NAMES: &str = r#"# 候选格式：
- <CANDIDATE_TREE> 中的 Category 表示分组，不可选择。
- <CANDIDATE_TREE> 中的 Candidate NAME 表示候选 skill，可选择，其中 NAME 是 Candidate skill 节点 id。
- 缩进表示候选与分组之间的从属关系。"#;

/// The candidate-format block of compact flat mode.
pub const STRUCTURE_CODES_FLAT: &str = r#"# 候选格式：
- Available options 是 JSON 数组，每个对象都是一个可选择的候选 skill。
- 每个对象格式为 {"category": "CATEGORY", "name": "NAME", "raw_name": "RAW_NAME", "description": "DESCRIPTION", "id": "ID"}。
- id 是两个大写字母候选 ID；category、name、raw_name 和 description 只用于判断相关性。
- 最终只能输出候选 JSON 对象 id 字段中的两个大写字母 ID。"#;

/// The candidate-format block of names flat mode (empty in the reference).
pub const STRUCTURE_NAMES_FLAT: &str = r#""#;

/// The output block of compact tree mode.
pub const OUTPUT_CODES: &str = r#"- 仅输出候选行 [id: ...] 方括号中的 id，不要附带任何其他内容。
- 每个非空行必须只包含一个候选 id；整行只能是 id 本身。
- 禁止输出横线、候选名称、[id: ...]、冒号、描述或整行候选文本。
- 任何以 - 开头的行都是错误输出。
- 如果想复制候选行，必须改为只输出该候选行 [id: ...] 中的 id。
- id 是无语义序号句柄，不是英文缩写、类别名或能力名，不参与语义判断。
- 不要根据 id 的字母含义、字母顺序或常见缩写含义判断相关性。
- 选择时只根据 NAME 和 DESCRIPTION 判断相关性；[id: X] 是无语义序号句柄，不是英文缩写、类别名或能力名，不参与语义判断。
- NAME 只能用于选择判断，不能出现在最终输出中。
- 最终只能输出方括号中的 id；不要输出 NAME、DESCRIPTION、[id:] 前缀或整行候选文本。
- 如果选择多个候选，每一行都必须是对应候选的 id；不要用候选名称替代 id。
- 绝对不要先输出 id，再继续输出候选名称。
- 如果想输出某个候选名称，必须改为输出该候选行 [id: ...] 中的 id。
- 输出前自检：每个非空行都必须完全匹配一个候选 id；如果某行不是候选 id，删除该行。
- 不要输出候选名称、描述、整行候选文本、[id:] 前缀、列表、编号、后缀或解释。
- 例如，候选行为 Summary [id: CC]: Summarize long documents 时，只能输出 CC。
- 错误输出包括：以 - 开头、包含候选名称、包含 [id:]、包含冒号、包含描述或复制整行候选文本。
- 正确输出示例：CC"#;

/// The output block the reference keeps for compact flat mode but selects nowhere.
pub const OUTPUT_CODES_FLAT: &str = r#"- 仅输出候选对象中的 id，不要附带任何其他内容。
- 每个非空行必须只包含一个候选 ID；整行只能是 ID 本身。
- 候选 ID 来自 JSON 对象的 "id" 字段。
- 禁止输出 JSON、方括号、花括号、引号、字段名、候选名称、冒号、描述或整个候选对象。
- 如果想复制候选对象，必须改为只输出该对象的 id。
- ID 是无语义序号句柄，不是英文缩写、类别名或能力名，不参与语义判断。
- 不要根据 ID 的字符含义、字符顺序或常见缩写含义判断相关性。
- 选择时只根据 NAME 和 DESCRIPTION 判断相关性；ID 是无语义序号句柄，不参与语义判断。
- NAME 只能用于选择判断，不能出现在最终输出中。
- 最终只能输出 ID；不要输出 NAME、DESCRIPTION、JSON 片段或整个候选对象。
- 如果选择多个候选，每一行都必须是对应候选的 ID；不要用候选名称替代 ID。
- 绝对不要先输出 ID，再继续输出候选名称。
- 输出前自检：每个非空行都必须完全匹配一个候选 ID；如果某行不是候选 ID，删除该行。
- 例如，候选对象为 {"id": "AB", "name": "Summary", "description": "Summarize long documents"} 时，只能输出 AB。
- 错误输出包括：包含候选名称、包含字段名、包含引号、包含冒号、包含描述或复制整个候选对象。
- 正确输出示例：AB"#;

/// The output block of names tree mode.
pub const OUTPUT_NAMES_TREE: &str = r#"- 仅输出驼峰格式的节点 id, 不要附带任何其他内容
- 只能输出 Candidate 行中的 NAME，不要输出 Category、描述、编号、解释或整行候选文本。"#;

/// The output block of names flat mode.
pub const OUTPUT_NAMES_FLAT: &str = r#"- 仅输出驼峰格式的节点 id, 不要附带任何其他内容"#;

/// The legacy flat, non-compact system prompt (used when `top_k != 2`).
pub const LEGACY_FLAT_NON_COMPACT_SYSTEM_PROMPT: &str = r#"
# 背景与目标：
- 系统的最终目标是解决用户的问题。
- 系统通过调用 Candidate skill 来执行任务。
- 你的职责是：组建一个最优 skill 小组，以端到端闭环用户的任务。

注意：
- 你只做“候选筛选”，不做规划、不做执行。


# 输出规则：

1. 总共输出至多 5 个 Candidate skill 节点 id

2. 该结果必须是：
   - 最优 Candidate skill 组合
   - 该组合能够完整闭环用户任务

3. 至多输出 5 行


# 选择规则：

- 必须严格遵守用户的显式约束（如时间、范围、条件等）
- 要选出**能够完整闭环用户意图的 skill 组合**，包括用户的显式意图和隐式意图。
  - 例如，用户有出行意图时，不仅考虑出行规划，也要选择查询天气相关skill
  - 用户有饮食、购物等意图时，不仅直接选择对应的出行助手，也要选择美团红包优惠
  - 用户要搭建网站时，不仅直接选择html生成与前端开发，也可以考虑艺术创作
  - 务必确认用户的真实隐含意图，例如“生成一张地图”实质上对应于基于AI的图像生成，而非查询地图。
  - 不要附带任何无关的skill，例如，用户不需要ppt时不要选择ppt相关技能。用户需要生成地图而非查询地点时，不要选择地图相关技能。
  - 用户明确不要某类skill时，不要选择对应的skill。




# 输出 id 说明：

- 仅输出驼峰格式的节点 id, 不要附带任何其他内容"#;

/// The output block of compact flat mode.
pub const COMPACT_FLAT_OUTPUT_BLOCK: &str = r#"- 只输出候选 JSON 对象中的两个大写字母 id，每行一个 id。
- 不要输出 JSON、字段名、候选名称、解释、编号、冒号、标点或整个候选对象。
- 如果想复制某个候选 JSON 对象，必须改为只输出该对象的两个大写字母 id。
- id 是无语义句柄，只能根据候选名称和能力说明判断相关性。"#;

/// Everything a prefix-cache-aware caller needs for one disclosure call.
///
/// Mirrors Python `DisclosurePromptParts`: the two full messages, the two prefix-only
/// messages, the request suffix (marker plus query) and the two cache digests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DisclosurePromptParts {
    /// The messages to send: `system` with the candidate block, then `user`.
    pub full_messages: Vec<ChatMessage>,
    /// The cacheable prefix: the same `system` message and a `user` message holding only
    /// the prefix.
    pub prefix_messages: Vec<ChatMessage>,
    /// The request suffix: the marker plus the query and the closing tag.
    pub suffix_text: String,
    /// The cache identity of this prompt.
    pub cache_id: String,
    /// The digest of the cacheable prefix.
    pub prefix_token_hash: String,
}

/// Builds the two disclosure messages of `fragment`.
///
/// Returns a `system` message holding the prompt bank plus the candidate block and a `user`
/// message holding the request marker and the query inside `<USER_REQUEST>…`. A blank
/// `query_messages` yields a user message holding only the marker (the warmup path).
#[must_use]
pub fn build_disclosure_messages(
    fragment: &ExposedFragment,
    query_messages: &[ChatMessage],
    top_k: Option<usize>,
) -> Vec<ChatMessage> {
    build_disclosure_prompt_parts(fragment, query_messages, top_k).full_messages
}

/// Builds the full and prefix-only disclosure messages plus their cache identity.
///
/// `top_k` is the advisory generation bound; `None` falls back to the number of candidates
/// (at least 1). See the module docs for the deterministic request marker.
#[must_use]
pub fn build_disclosure_prompt_parts(
    fragment: &ExposedFragment,
    query_messages: &[ChatMessage],
    top_k: Option<usize>,
) -> DisclosurePromptParts {
    let query = normalized_query(query_messages);
    let resolved_top_k = top_k.map(|value| value.max(1));
    let system_prompt = system_prompt_for(fragment, resolved_top_k);
    let user_prefix = user_prefix_for(fragment.flat_list_mode);
    let prefix_token_hash =
        prefix_token_hash(&system_prompt, user_prefix, fragment, resolved_top_k);
    let cache_id = cache_id(&prefix_token_hash, &fragment.candidate_codes);
    let marker = request_marker(&query, &fragment.fragment_fingerprint, resolved_top_k);
    let body = if fragment.flat_list_mode {
        query
    } else {
        format!("{query}\n{USER_REQUEST_CLOSE}")
            .trim_end()
            .to_owned()
    };
    let suffix_text = format!("{marker}{body}");
    let user_content = format!("{user_prefix}{suffix_text}").trim_end().to_owned();
    DisclosurePromptParts {
        full_messages: vec![
            ChatMessage::system(system_prompt.clone()),
            ChatMessage::user(user_content),
        ],
        prefix_messages: vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(user_prefix),
        ],
        suffix_text,
        cache_id,
        prefix_token_hash,
    }
}

/// The system prompt of `fragment`, with its candidate block embedded.
///
/// `top_k` defaults to the number of candidates (at least 1), as the reference's
/// `ExposedFragment::build_system_prompt` does.
#[must_use]
pub fn disclosure_system_prompt(fragment: &ExposedFragment, top_k: Option<usize>) -> String {
    system_prompt_for(fragment, top_k)
}

/// The user-message prefix of tree mode (`<USER_REQUEST>` plus a newline).
///
/// Flat mode uses [`FLAT_USER_PREFIX`] instead; the reference keeps both prefixes inside its
/// static prompt builder, while this port exposes the tree prefix here because it is the
/// one a caller composing its own messages needs.
#[must_use]
pub fn disclosure_user_prefix() -> String {
    USER_REQUEST_PREFIX.to_owned()
}

/// The deterministic per-call request marker of `query_text` under `fingerprint`.
///
/// The reference uses a fresh `uuid4`; see the module docs for why this port uses a digest
/// instead. [`build_disclosure_prompt_parts`] folds `top_k` into the digest as well, so
/// compare markers only when both calls pass the same `top_k` (or read the marker from
/// [`DisclosurePromptParts::suffix_text`]).
#[must_use]
pub fn disclosure_request_marker(query_text: &str, fingerprint: &str) -> String {
    request_marker(query_text, fingerprint, None)
}

/// The system prompt for one variant and candidate block.
fn build_system_prompt(
    compact_codes_enabled: bool,
    flat_list_mode: bool,
    top_k: usize,
    candidate_block: &str,
) -> String {
    let resolved_top_k = top_k.max(1);
    let candidate = candidate_block.trim_end();
    if !compact_codes_enabled && flat_list_mode && resolved_top_k != 2 {
        let mut parts: Vec<&str> = Vec::new();
        if !LEGACY_FLAT_NON_COMPACT_SYSTEM_PROMPT.is_empty() {
            parts.push(LEGACY_FLAT_NON_COMPACT_SYSTEM_PROMPT);
        }
        if !candidate.is_empty() {
            parts.push(candidate);
        }
        return parts.join("\n\n");
    }
    let (structure_block, output_block) = match (compact_codes_enabled, flat_list_mode) {
        (true, true) => (STRUCTURE_CODES_FLAT, COMPACT_FLAT_OUTPUT_BLOCK),
        (true, false) => (STRUCTURE_CODES, OUTPUT_CODES),
        (false, true) => (STRUCTURE_NAMES_FLAT, OUTPUT_NAMES_FLAT),
        (false, false) => (STRUCTURE_NAMES, OUTPUT_NAMES_TREE),
    };
    let template = if resolved_top_k == 2 { SHARED_TOP2 } else { SHARED };
    expand_template(
        template,
        &[
            ("{top_k}", resolved_top_k.to_string()),
            ("{structure_block}", structure_block.to_owned()),
            ("{candidate_block}", candidate.to_owned()),
            ("{output_block}", output_block.to_owned()),
        ],
    )
}

/// Substitutes the template placeholders in one pass.
///
/// A single pass keeps substituted text inert, so a candidate description containing `{`
/// can never be re-scanned — the same guarantee Python's `str.format` gives.
fn expand_template(template: &str, substitutions: &[(&str, String)]) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    'outer: while let Some(index) = rest.find('{') {
        rendered.push_str(&rest[..index]);
        let tail = &rest[index..];
        for (token, value) in substitutions {
            if let Some(after) = tail.strip_prefix(token) {
                rendered.push_str(value);
                rest = after;
                continue 'outer;
            }
        }
        rendered.push('{');
        rest = &tail[1..];
    }
    rendered.push_str(rest);
    rendered
}

/// The prompt for `fragment`, resolving a `None` `top_k` to the candidate count.
fn system_prompt_for(fragment: &ExposedFragment, top_k: Option<usize>) -> String {
    let resolved_top_k = top_k
        .map(|value| value.max(1))
        .unwrap_or_else(|| fragment.candidate_codes.len().max(1));
    build_system_prompt(
        fragment.compact_codes_enabled,
        fragment.flat_list_mode,
        resolved_top_k,
        &candidate_block(fragment),
    )
}

/// The candidate block embedded in the system message.
fn candidate_block(fragment: &ExposedFragment) -> String {
    let body = if fragment.flat_list_mode {
        format!("{FLAT_CANDIDATE_HEADER}\n{}", fragment.rendered_tree)
    } else {
        format!(
            "{CANDIDATE_TREE_OPEN}\n{}\n{CANDIDATE_TREE_CLOSE}",
            fragment.rendered_tree
        )
    };
    format!("{CANDIDATE_HEADER}\n{body}")
}

/// The user-message prefix of the given mode.
fn user_prefix_for(flat_list_mode: bool) -> &'static str {
    if flat_list_mode {
        FLAT_USER_PREFIX
    } else {
        USER_REQUEST_PREFIX
    }
}

/// The digest of the cacheable prefix.
fn prefix_token_hash(
    system_prompt: &str,
    user_prefix: &str,
    fragment: &ExposedFragment,
    top_k: Option<usize>,
) -> String {
    let payload = format!(
        "{system_prompt}\n{user_prefix}\n{}\n{}\n{}\n{}",
        fragment.fragment_fingerprint,
        top_k.map(|value| value.to_string()).unwrap_or_default(),
        u8::from(fragment.compact_codes_enabled),
        u8::from(fragment.flat_list_mode),
    );
    sha256_hex(&payload)
}

/// The cache identity of a prompt: the version, the prefix digest and the candidate codes.
fn cache_id(prefix_token_hash: &str, candidate_codes: &[String]) -> String {
    let payload = format!(
        "{CACHE_ID_VERSION}\n{prefix_token_hash}\n{}",
        candidate_codes.join("/")
    );
    sha256_short(&payload, 32)
}

/// The deterministic request marker.
fn request_marker(query_text: &str, fingerprint: &str, top_k: Option<usize>) -> String {
    let payload = format!(
        "{query_text}\n{fingerprint}\n{}",
        top_k.map(|value| value.to_string()).unwrap_or_default()
    );
    format!("[req:{}]", sha256_short(&payload, REQUEST_MARKER_HEX))
}

/// Joins the normalized contents of `query_messages`.
fn normalized_query(query_messages: &[ChatMessage]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for message in query_messages {
        if message.content.trim().is_empty() {
            continue;
        }
        parts.push(normalize_query_content(message.content.trim()));
    }
    parts.join("\n").trim().to_owned()
}

/// Drops `From <speaker>:` prefixes and blank lines from one query message.
fn normalize_query_content(text: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for raw in text.split('\n') {
        let cleaned = strip_from_prefix(raw.trim());
        if !cleaned.is_empty() {
            lines.push(cleaned);
        }
    }
    lines.join("\n")
}

/// Removes a leading `From <speaker>:` prefix, mirroring `^\s*From\s+[^:]+:\s*`.
fn strip_from_prefix(line: &str) -> &str {
    let Some(after_from) = strip_prefix_ignore_ascii_case(line.trim_start(), "From") else {
        return line;
    };
    let rest = after_from.trim_start();
    if rest.len() == after_from.len() {
        // `From` must be followed by at least one whitespace character.
        return line;
    }
    let Some(colon) = rest.find(':') else {
        return line;
    };
    if colon == 0 {
        // `[^:]+` requires at least one character before the colon.
        return line;
    }
    rest[colon + 1..].trim_start()
}

/// Strips `prefix` when the start of `text` matches it case-insensitively.
fn strip_prefix_ignore_ascii_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    if head.eq_ignore_ascii_case(prefix) {
        text.get(prefix.len()..)
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::retrieval::search::flow::{
        ExposedFragment, ExposedNode, SelectableResolution,
    };
    use std::collections::BTreeMap;

    fn fragment(compact: bool, flat: bool, codes: &[&str]) -> ExposedFragment {
        let candidate_codes: Vec<String> = codes.iter().map(|code| (*code).to_owned()).collect();
        let mut code_to_resolution: BTreeMap<String, SelectableResolution> = BTreeMap::new();
        for (index, code) in candidate_codes.iter().enumerate() {
            code_to_resolution.insert(
                code.clone(),
                SelectableResolution {
                    code: code.clone(),
                    canonical_id: format!("custom.item{index}"),
                    display_name: format!("Item{index}"),
                    label: format!("Item {index}"),
                    description: String::new(),
                    is_terminal: true,
                    branch_path: vec!["ROOT".to_owned()],
                    node_id: None,
                    item: None,
                },
            );
        }
        ExposedFragment {
            root: ExposedNode::default(),
            rendered_tree: "candidate-body".to_owned(),
            code_to_resolution,
            candidate_codes,
            fragment_fingerprint: "0123456789abcdef".to_owned(),
            code_width: 1,
            compact_codes_enabled: compact,
            flat_list_mode: flat,
            rendered_chars: 4,
            truncated: false,
        }
    }

    fn query(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage::user(text)]
    }

    #[test]
    fn the_prompt_variants_follow_the_compact_and_flat_pair() {
        let tree_codes = disclosure_system_prompt(&fragment(true, false, &["AB"]), Some(3));
        assert!(tree_codes.contains(STRUCTURE_CODES));
        assert!(tree_codes.contains(OUTPUT_CODES));
        assert!(tree_codes.contains("<CANDIDATE_TREE>\ncandidate-body\n</CANDIDATE_TREE>"));
        assert!(tree_codes.contains("# 候选列表"));
        assert!(!tree_codes.contains("Available options"));

        let flat_codes = disclosure_system_prompt(&fragment(true, true, &["AB"]), Some(3));
        assert!(flat_codes.contains(STRUCTURE_CODES_FLAT));
        assert!(flat_codes.contains(COMPACT_FLAT_OUTPUT_BLOCK));
        assert!(flat_codes.contains("Available options:\ncandidate-body"));

        let tree_names = disclosure_system_prompt(&fragment(false, false, &["Item"]), Some(3));
        assert!(tree_names.contains(STRUCTURE_NAMES));
        assert!(tree_names.contains(OUTPUT_NAMES_TREE));
        assert!(tree_names.contains("<CANDIDATE_TREE>\ncandidate-body\n</CANDIDATE_TREE>"));

        let flat_names = disclosure_system_prompt(&fragment(false, true, &["Item"]), Some(2));
        assert!(flat_names.contains(STRUCTURE_NAMES_FLAT));
        assert!(flat_names.contains(OUTPUT_NAMES_FLAT));
        assert!(flat_names.contains("Available options:\ncandidate-body"));

        let legacy = disclosure_system_prompt(&fragment(false, true, &["Item"]), Some(3));
        assert!(legacy.starts_with(LEGACY_FLAT_NON_COMPACT_SYSTEM_PROMPT));
        assert!(legacy.contains("总共输出至多 5 个 Candidate skill 节点 id"));
        assert!(legacy.contains("Available options:\ncandidate-body"));
        // `STRUCTURE_NAMES_FLAT` is the empty string, so only a rendered marker can
        // prove the legacy prompt carries no structure block.
        assert!(!legacy.contains("# 候选格式："));
    }

    #[test]
    fn top_k_selects_the_shared_preamble_and_is_interpolated() {
        let two = disclosure_system_prompt(&fragment(true, false, &["AB"]), Some(2));
        assert!(two.contains("默认输出 2 个 Candidate skill 节点 id"));
        assert!(!two.contains("{top_k}"));
        assert!(two.contains("只有当所有候选中完全不存在可放入同一任务链路的第二个 skill 时"));

        let five = disclosure_system_prompt(&fragment(true, false, &["AB"]), Some(5));
        assert!(five.contains("总共输出至多 5 个 Candidate skill 节点 id"));
        assert!(five.contains("至多输出 5 行"));
        assert!(!five.contains("默认输出 2 个 Candidate skill 节点 id"));

        let defaulted = disclosure_system_prompt(&fragment(true, false, &["AB"]), None);
        assert!(defaulted.contains("总共输出至多 1 个 Candidate skill 节点 id"));

        let zero = disclosure_system_prompt(&fragment(true, false, &["AB"]), Some(0));
        assert!(zero.contains("总共输出至多 1 个 Candidate skill 节点 id"));
    }

    #[test]
    fn the_candidate_block_appears_once_in_the_system_message() {
        let exposed = fragment(true, false, &["AB", "AC"]);
        let messages = build_disclosure_messages(&exposed, &query("帮我把这份文档总结成要点"), Some(2));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[0].content.matches("# 候选列表").count(), 1);
        // The structure block mentions `<CANDIDATE_TREE>` twice as a name; only the
        // candidate block opens the tag on its own line.
        assert_eq!(messages[0].content.matches("<CANDIDATE_TREE>\n").count(), 1);
        assert_eq!(messages[0].content.matches("candidate-body").count(), 1);
        assert!(!messages[1].content.contains("<CANDIDATE_TREE>"));
        assert!(messages[1]
            .content
            .starts_with("<USER_REQUEST>\n[req:"));
        assert!(messages[1].content.ends_with("</USER_REQUEST>"));
        assert!(messages[1].content.contains("帮我把这份文档总结成要点"));
    }

    #[test]
    fn prefix_messages_hold_the_prefix_without_the_query() {
        let exposed = fragment(true, false, &["AB"]);
        let parts = build_disclosure_prompt_parts(&exposed, &query("摘要"), Some(2));
        assert_eq!(parts.prefix_messages.len(), 2);
        assert_eq!(parts.prefix_messages[0], parts.full_messages[0]);
        assert_eq!(parts.prefix_messages[1].content, USER_REQUEST_PREFIX);
        assert!(!parts.prefix_messages[1].content.contains("摘要"));
        assert!(parts.suffix_text.starts_with("[req:"));
        assert!(parts.suffix_text.ends_with("摘要\n</USER_REQUEST>"));
        assert_eq!(parts.prefix_token_hash.len(), 64);
        assert_eq!(parts.cache_id.len(), 32);

        let flat = build_disclosure_prompt_parts(&fragment(true, true, &["AB"]), &query("摘要"), Some(2));
        assert_eq!(flat.prefix_messages[1].content, FLAT_USER_PREFIX);
        assert_eq!(flat.suffix_text, format!("{}摘要", flat.suffix_text.split_once(']').unwrap().0.to_owned() + "]"));
        assert!(flat.full_messages[1].content.starts_with("User request:\n[req:"));
    }

    #[test]
    fn the_request_marker_is_deterministic_and_query_sensitive() {
        let first = disclosure_request_marker("摘要", "abc");
        let second = disclosure_request_marker("摘要", "abc");
        assert_eq!(first, second);
        assert!(first.starts_with("[req:"));
        assert!(first.ends_with(']'));
        assert_eq!(first.len(), "[req:".len() + REQUEST_MARKER_HEX + 1);
        assert!(first["[req:".len()..first.len() - 1]
            .chars()
            .all(|character| character.is_ascii_hexdigit()));
        assert_ne!(first, disclosure_request_marker("摘要2", "abc"));
        assert_ne!(first, disclosure_request_marker("摘要", "abd"));

        let exposed = fragment(true, false, &["AB"]);
        let one = build_disclosure_prompt_parts(&exposed, &query("摘要"), Some(2));
        let two = build_disclosure_prompt_parts(&exposed, &query("摘要"), Some(2));
        assert_eq!(one.full_messages, two.full_messages);
        assert_eq!(one.cache_id, two.cache_id);
        assert_eq!(one.prefix_token_hash, two.prefix_token_hash);

        let other_top_k = build_disclosure_prompt_parts(&exposed, &query("摘要"), Some(3));
        assert_ne!(other_top_k.suffix_text, one.suffix_text);
    }

    #[test]
    fn query_normalization_strips_speaker_prefixes_and_blank_lines() {
        assert_eq!(normalize_query_content("   "), "");
        assert_eq!(
            normalize_query_content("From user: 帮我总结\n\nFrom assistant: 好的\n第二行"),
            "帮我总结\n好的\n第二行"
        );
        assert_eq!(normalize_query_content("From: no speaker"), "From: no speaker");
        assert_eq!(normalize_query_content("From me 总结"), "From me 总结");
        assert_eq!(normalize_query_content("总结 From user: 后面"), "总结 From user: 后面");

        let joined = normalized_query(&[
            ChatMessage::user("From user: 第一段"),
            ChatMessage::system("  "),
            ChatMessage::user("第二段"),
        ]);
        assert_eq!(joined, "第一段\n第二段");

        let messages = build_disclosure_messages(
            &fragment(true, false, &["AB"]),
            &[ChatMessage::user("From user: 帮我总结")],
            Some(2),
        );
        assert!(messages[1].content.ends_with("帮我总结\n</USER_REQUEST>"));
        assert!(!messages[1].content.contains("From user:"));
    }

    #[test]
    fn an_empty_query_yields_only_the_marker() {
        let exposed = fragment(true, false, &["AB"]);
        let parts = build_disclosure_prompt_parts(&exposed, &[], Some(2));
        assert_eq!(parts.full_messages.len(), 2);
        assert!(parts.suffix_text.starts_with("[req:"));
        assert!(parts.suffix_text.ends_with("\n</USER_REQUEST>"));
        assert_eq!(
            parts.full_messages[1].content,
            format!("{USER_REQUEST_PREFIX}{}", parts.suffix_text)
        );
        assert!(parts.full_messages[1]
            .content
            .starts_with("<USER_REQUEST>\n[req:"));
        assert!(!parts.suffix_text.contains("帮我"));
        // A query-less call is still deterministic.
        assert_eq!(
            build_disclosure_prompt_parts(&exposed, &[], Some(2)),
            parts
        );
    }
}
