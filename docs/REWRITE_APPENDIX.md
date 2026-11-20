# 重写统计附录

本文件是 [REPORT.md](../REPORT.md) 的统计口径与复现说明：历史任务账本、代码规模测法、测试计数、性能与验证命令。
所有数字都给出取数来源与计算方式；正文只引用结论，不重复推导。

## 1. 历史任务账本

### 1.1 数据来源

来自 Genesis 权威 task history 的两条有效任务记录与一条失败记录。
字段含义：`agent_count` 是该任务累计登记的 agent 实例数；`usage.total_cost` 是模型调用费用；token 计数由运行时记录。

| 字段 | `f47fe0e88cbfcc11` | `83005e9186866bb6` | 合计 |
|---|---|---:|---:|
| 类型 | genesis | evolve | — |
| 状态 | cancelled（用户手工中断，成果有效） | completed | — |
| `started_at` | 2026-11-11T17:46:03.137Z | 2026-11-19T16:09:10.767Z | — |
| `finished_at` | 2026-11-16T06:13:38.184Z | 2026-11-19T21:24:06.496Z | — |
| `usage.total_cost` | 144.5190509999999 | 5.505231999999999 | **150.0242829999999** |
| `agent_count` | 3,466 | 104 | **3,570** |
| `input_tokens` | 10,267,193,044 | 600,100,763 | **10,867,293,807** |
| `output_tokens` | 96,888,701 | 3,423,285 | **100,311,986** |
| `total_tokens` | 10,364,081,745 | 603,524,048 | **10,967,605,793** |
| `cached_tokens` | 10,055,394,944 | 588,868,992 | **10,644,263,936** |
| `base_sha` | —（首个任务） | `c39798a64a3efac8db2cd22737b3b9615772699e` | — |
| 结果提交 | `c39798a64a3efac8db2cd22737b3b9615772699e` | `e9e64d2256b91fd11b92e88ffef354cb8fe2d4fb` | — |

校验：`total_tokens = input_tokens + output_tokens` 在两个任务上都成立；合计列为逐字段相加。

### 1.2 墙钟时长

| 任务 | 秒 | 换算 |
|---|---:|---|
| `f47fe0e88cbfcc11` | 390,455.047 | 108h27m35.047s（4d12h27m35.047s） |
| `83005e9186866bb6` | 18,895.729 | 5h14m55.729s |
| **合计** | **409,350.776** | **113h42m30.776s（4d17h42m30.776s）** |

口径：

- 时长是**两次任务的墙钟时长之和**，含等待、排队、人工中断间隔；不是净计算时间，也不是人工工时。
- `f47fe0e88cbfcc11` 被用户中断，其 `finished_at` 是中断时刻；成果仍然有效，因此计入。
- 两个任务之间不是连续的：从第一个 `started_at` 到第二个 `finished_at` 的日历跨度为 **8d3h38m3.359s**，与墙钟之和不可混用。
- 复算代码：

```python
from datetime import datetime
def p(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))
a = p("2026-11-16T06:13:38.184Z") - p("2026-11-11T17:46:03.137Z")
b = p("2026-11-19T21:24:06.496Z") - p("2026-11-19T16:09:10.767Z")
print(a.total_seconds() + b.total_seconds())   # 409350.776
print(p("2026-11-19T21:24:06.496Z") - p("2026-11-11T17:46:03.137Z"))  # 日历跨度
```

### 1.3 没有计入的记录

| 任务 | 状态 | 窗口（UTC） | 墙钟 | usage / agent_count | 结果提交 |
|---|---|---|---:|---|---|
| `44c2113a77b2bb42` | failed（`Lease expired; owning instance no longer renewing`） | 2026-11-16T17:08:30.915Z → 2026-11-19T10:33:19.968Z | 235,489.053s（2d17h24m49s） | `null` / `null` | 无 |

`null` 表示未记录，**不等于 0**。因此：

- 上表两个有效任务的合计 **不是**该账户/环境的全部消耗，只是这两次任务的记录值。
- 失败任务既不并入费用，也不并入 agent 数与时长；它只作为透明度说明列出。
- 本次文档任务（README / REPORT / 附录的撰写）不在两条历史记录内，也不计入任何合计。

### 1.4 费用与 token 口径

- `usage.total_cost` 是 Genesis 记录的**模型调用费用**（USD），不含机器折旧、电力、网络、人工与任何未记录开销；不是账单核销金额。
- `input_tokens` **包含缓存命中的 token**（`cached_tokens` 约占 input 的 98%）；`total_tokens` 已含 input + output，不要再次叠加 `cached_tokens`。
- `agent_count` 是累计登记的 agent 实例，不是峰值并发数，也不代表人月/人数。

## 2. 代码规模

### 2.1 取数对象

统计对象是**当前提交中 `git ls-files '*.rs'` 的文件**，即 git 跟踪的 Rust 源码；两次任务快照的行数不能相加（会重复计算同一批最终代码）。
`target/`、`docs/bench/rust/target/` 等构建产物不在跟踪范围内，不计入。

### 2.2 分类规则

| 分类 | 规则 | 文件数 | 行数 |
|---|---|---:|---:|
| 生产代码 | `crates/*/src/**/*.rs`（不含路径中含 `/tests/` 的文件），减去每个文件第一个 `#[cfg(test)]` 到文件末尾的行 | 2,314 | 803,046 |
| 内嵌单元测试 | 上述被减去的行（`#[cfg(test)]` 模块） | — | 324,292 |
| 集成测试 | 路径中含 `/tests/` 的 `.rs`（含嵌套目录） | 593 | 295,499 |
| 示例 | `crates/openjiuwen-examples/examples/**/*.rs` | 27 | 11,930 |
| **合计** | `git ls-files '*.rs'` | **2,934** | **1,434,767** |

四类是互斥划分（先取 `examples/`，再取 `/tests/`，余下归 `src/`），文件数相加等于 2,934，行数相加等于 1,434,767。

两种口径的差异必须说明：

- “行数”是 `wc -l` 的物理行数，**包含空行与注释**。
- `tokei 14.0.0` 对同一批文件报 **1,098,785 行代码**（另有 16,582 行注释、107,382 行空行，以及 174k+ 行以 `///` 写成的 Markdown 文档注释被单独归类）。因此 `tokei` 的 Code 列不含空行/注释，也不含 doc-comment 行。
- 内嵌测试的切分是近似规则（取每个文件第一个 `#[cfg(test)]` 到末尾）；少数把测试模块写在文件中间的源码会低估测试行、高估生产行。它只用于给出量级，不用于逐行审计。

### 2.3 复现命令

```bash
# 跟踪的 Rust 文件与物理行
git ls-files '*.rs' | wc -l
git ls-files '*.rs' | xargs wc -l | awk '$2=="total"{s+=$1} END{print s}'   # 1434767

# 分类（示例：集成测试）
git ls-files 'crates/*/tests/*.rs' | wc -l
git ls-files 'crates/*/tests/*.rs' | xargs wc -l | awk '$2=="total"{s+=$1} END{print s}'

# 内嵌测试切分
python3 - <<'EOF'
import subprocess, pathlib
allf=subprocess.check_output(['git','ls-files','*.rs'],text=True).split()
ex=[f for f in allf if '/examples/' in f]
t=[f for f in allf if '/tests/' in f and f not in ex]
files=[f for f in allf if '/src/' in f and f not in ex and f not in t]
tot=inline=0
for f in files:
    lines=pathlib.Path(f).read_text(errors='replace').splitlines()
    tot+=len(lines)
    for i,l in enumerate(lines):
        if l.strip().startswith('#[cfg(test)]'):
            inline+=len(lines)-i
            break
print(len(files), tot, inline, tot-inline)   # 2314 1127338 324292 803046
EOF

# 代码行（排除空行/注释）
tokei crates/*/src crates/*/tests crates/openjiuwen-examples/examples
```

### 2.4 与 Python 参考的规模对照

`tokei` 对只读参考仓库 `/home/bill/Source/agent-core`（v0.1.20，HEAD `455e2a338`）的统计：Python **667 个文件 / 50,913 行**（43,655 行代码）。
两者不是逐行对应：Rust 侧包含大量公开文档注释、trait/类型样板与更完整的离线测试。
本附录不把“行数比”当作重写质量的度量。

## 3. 计数：crate、示例、测试

| 对象 | 值 | 取法 |
|---|---:|---|
| workspace crate | 30 | `ls -d crates/*/ \| wc -l`；`members = ["crates/*"]` |
| `default-members` | 27 | `Cargo.toml`；排除 `openjiuwen-e2e`、`openjiuwen-examples`、`openjiuwen-test-support` |
| 可运行示例 | 26 | `ls crates/openjiuwen-examples/examples/*.rs \| wc -l`（另有 `common/` 共享模块） |
| 测试函数 | 24,293 | `git grep -o '#\[\(tokio::\)\?test\]' -- '*.rs' \| wc -l` |
| `#[ignore]` 测试 | 5 | `git grep -n '#\[ignore =' -- '*.rs'`（3 个 stdio 协议子进程桩、1 个 live sandbox 探测、1 个 e2e 二进制探测） |
| CLI 子命令 | 8 | `openjiuwen --help`：`run`/`chat`/`sessions`/`lsp`/`cron`/`team`/`watch`/`auto-harness` |

## 4. 性能测量的位置

完整方法、环境、命令、原始 JSON 与限制位于 [BENCHMARKS.md](BENCHMARKS.md)：

- 测量脚本：`docs/bench/rust/`（独立 package，空 `[workspace]`，路径依赖工作区 crate）、`docs/bench/python/bench.py`、`docs/bench/python/measure_proc.py`。
- 原始结果：`docs/bench/results/process/*.json` 与 `docs/bench/results/{rust,python}/*_run{1,2,3}.json`。
- 重算：`python3 docs/bench/aggregate.py`。
- 环境：Intel Core Ultra 5 325（8 逻辑 CPU）、23.0 GiB、NixOS 26.11 / kernel 7.3.5、rustc 1.99.1（release，`lto="thin"`、`codegen-units=1`）、CPython 3.11.15。
- 边界：warm page cache；未测 LLM/网络、冷磁盘启动、并发扩展；Python 参考实现无脚本化模型，故 agent 回合只有 Rust 数据；进程启动行与导入行语义不同，不做倍数。

## 5. 本次文档任务的验证记录

只改文档与元数据（README、REPORT、附录、`docs/*` 勘误、`LICENSE`、`Cargo.toml` 的 `repository` 字段），未改产品语义。
实际执行的命令与结果：

| 命令 | 结果 |
|---|---|
| `cargo run -q -p openjiuwen-cli -- --version` | `openjiuwen 0.1.0`，exit 0 |
| `openjiuwen --help` / `run --help` | exit 0；8 个子命令与 README 表格一致 |
| `openjiuwen run "hello" --provider openai`（无凭据） | exit 1，`error: config: provider \`openai\` requires an API key; …`，未发 HTTP 请求 |
| `--example 01/02/04/06/08/16` | exit 0，离线输出与 README 引用一致 |
| `--example 11_a2a_server_client --features a2a` | exit 0，loopback A2A，`answer: echo: hello a2a` |
| `cargo build -p openjiuwen-examples --examples` | exit 0 |
| `cargo test -p openjiuwen-common --offline` | 193 单元 + 12+17+15+11+3+16+4 集成 + 26 doctest 全通过 |
| `cargo test -p openjiuwen-cli --offline` | 56 单元 + 96 集成（10 个测试二进制）全通过 |

未重跑 `cargo test --workspace`（24k+ 测试、且本任务未改代码）；workspace 全门禁的上一轮结果记录在 `docs/PARITY.md` §Verification，本次仅修正其中被源码证伪的语句。
README 中的链接、示例名、feature 名与 CLI 标志均按当前源码核对；`docs/PARITY.md` 的代码规模与状态表以其自身的 crate 级路由为准。
