# openJiuwen agent-core → Rust 重写总结

**范围**：本文总结两次有效历史任务（`f47fe0e88cbfcc11` 与 `83005e9186866bb6`）共同完成的 Rust 重写。
**基准**：openJiuwen agent-core（Python SDK v0.1.20，参考 commit `455e2a338`），原项目 <https://github.com/openJiuwen-ai/agent-core>。
**性质**：独立重写，不是上游官方发行物；当前为**可用但非完整**的子集，未实现能力以类型化错误显式返回。

## 1. 项目与实现范围

Rust workspace 由 30 个 crate 组成，按依赖方向分层：基础（`common` / `macros` / `harness-protocol`）、模型与工具（`foundation` / `retrieval` / `sysop`）、状态与治理（`session` / `memory` / `security`）、编排引擎（`graph` / `context` / `workflow`）、智能体（`agent` / `multiagent` / `runner`）、框架与扩展（`core` / `harness` / `harness-providers` / `extensions` / `dev-tools` / `symphony`）、应用（`agent-teams` / `rsi` / `agent-evolving` / `auto-harness`）、入口（`openjiuwen` umbrella、`openjiuwen-cli`）以及三个开发 crate（`test-support` / `e2e` / `examples`）。

| 交付面 | 状态 |
|---|---|
| kernel：错误/Card/配置/日志/HTTP、prompt、模型客户端、工具与 MCP、会话与检查点、图与工作流、上下文、记忆、检索、sysop、护栏 | 已实现或可用子集 |
| Agent 运行时：ReAct、rails、controller、工具策略、runner、资源注册、HITL、spawn | 可用子集（分布式模式未实现） |
| DeepAgent harness：任务循环、工作区、权限、20 个编码工具、subagent、skills、原生协议 | 可用子集 |
| 多智能体团队：进程内团队、`team-member` CLI、team MCP server、跨进程 attach | 可用子集（ZeroMQ/WS/Pulsar、PG/MySQL 未实现） |
| 扩展与工具链：A2A、OTel、prompt/skill/agent dev-tools、symphony、RSI/auto-harness、agent-evolving、`openjiuwen` CLI（8 个子命令） | 可用子集 |
| 可运行示例 | 26 个，默认离线、确定性 |

**兼容边界**：保留 OpenAI / Anthropic / Responses 线格式、`*Card` JSON 字段与默认值、`${node.field}` 引用、会话与检查点落盘格式、`OPENJIUWEN_*` 环境变量链与 CLI 子命令/退出码。
**未实现**：依赖 GPU/ML 栈、厂商 SDK、外部数据库/消息中间件、桌面与移动传输、分布式运行时的路径；全仓约 341 处生产代码 `Unsupported` 点，逐项原因见 `docs/PARITY.md`。
因此**不能声称 100% 等价**：覆盖面、性能与失败语义都按 Rust 实现重新定义，文档与测试是这一范围的权威。

## 2. 代码规模

统计对象是**当前提交（HEAD）中 git 跟踪的 `.rs` 文件**，不是两次任务快照相加，避免重复计算。

| 分类 | 文件数 | 行数 | 口径 |
|---|---:|---:|---|
| 生产代码 | 2,314 | 803,046 | `crates/*/src` 去掉每个文件第一个 `#[cfg(test)]` 之后的部分 |
| 内嵌单元测试 | — | 324,292 | 上述被去掉的部分 |
| 集成测试 | 593 | 295,499 | `crates/*/tests`（含嵌套目录），一个文件一个测试二进制 |
| 示例 | 27 | 11,930 | 26 个示例 + 共享 `common` |
| **合计** | **2,934** | **1,434,767** | `tokei` 计 1,098,785 行代码（排除空行/注释） |

测试代码约占全部 Rust 行的 **43%**：24,293 个 `#[test]`/`#[tokio::test]` 函数，30 个 crate，5 个 `#[ignore]`（均为被父测试重新执行的协议桩）。
Python 参考实现本身约 667 个文件 / 50,913 行（`tokei`，含测试）；两者不是逐行对应关系——Rust 侧包含大量文档注释、类型与 trait 样板、以及更完整的离线测试。
精确口径与复现命令见 [docs/REWRITE_APPENDIX.md](docs/REWRITE_APPENDIX.md)。

## 3. 两次历史任务的投入

| 任务 | 类型 / 状态 | 起止（UTC） | 墙钟时长 | Agent 实例 | 费用（USD） |
|---|---|---|---:|---:|---:|
| `f47fe0e88cbfcc11` | genesis / cancelled（用户中断，成果有效） | 11-11 17:46:03 → 11-16 06:13:38 | 108h27m35.047s | 3,466 | 144.519051 |
| `83005e9186866bb6` | evolve / completed | 11-19 16:09:10 → 11-19 21:24:06 | 5h14m55.729s | 104 | 5.505232 |
| **合计** | 两次有效任务 | 日历跨度 8d3h38m | **113h42m30.776s**（4d17h42m31s） | **3,570** | **150.024283** |

Tokens（合计）：input **10,867,293,807**（其中缓存 10,644,263,936，占 98%）、output **100,311,986**、total **10,967,605,793**。
口径说明：
- 费用是 **Genesis 记录的模型调用费用**，不含机器、人工、网络与未记录费用，不是账单核销金额。
- Agent 数是两次任务累计登记的 agent 实例，不是峰值并发，也不是人数。
- 时长是任务墙钟时长之和（含等待与排队），不是净计算工时，也不是人工工时。
- 另有任务 `44c2113a77b2bb42`（2026-11-16T17:08:30.915Z → 2026-11-19T10:33:19.968Z，因 `Lease expired` 失败，usage/agent_count 为 `null`，无结果提交）：**不计入**上表。未知用量不等于 0，因此不声称账户总消耗只有上述金额。
- 当前这次文档任务同样不计入。

## 4. 测试与验证

- 默认门禁：`make test` = `cargo test --workspace --offline --no-fail-fast`，30 个 crate 全部离线通过（无网络、无凭据、loopback socket、临时目录）。
- `make test-all`：7 组特性门禁（e2e `a2a`、extensions `a2a,otel`、harness 与 agent-teams `--no-default-features`、symphony `sqlite`、sysop `sandbox`、harness-providers `vendor`）。
- `make verify`：`fmt-check` + `clippy -D warnings` + `test` + `doc -D warnings` + `test-all` + 两组示例构建。
- 本次文档任务实际执行：`openjiuwen --version`、CLI `--help`/`run --help`、无凭据错误路径、示例 01/02/04/06/08/16/11（a2a）、`cargo build -p openjiuwen-examples --examples`、`cargo test -p openjiuwen-common`（193 单元 + 78 集成 + 26 doctest）、`cargo test -p openjiuwen-cli`（56 单元 + 96 集成）全部通过。
- 未重跑整个 workspace 套件（本任务只改文档）；`docs/PARITY.md` §Verification 记录上一轮全门禁结果，本次仅修正其中被源码证伪的表述。

## 5. 性能实测（摘要）

同机（8 逻辑核 / 23 GiB / NixOS / kernel 7.3.5，warm page cache，无网络），Rust release（`lto=thin`、`codegen-units=1`）与 CPython 3.11.15：

| 指标 | Rust | Python | 结论 |
|---|---|---|---|
| 进程启动、SDK 类型可达 | 1.20 ms / 3.60 MiB 峰值 RSS | — | Rust 专属下界，不做倍数 |
| 一次脚本化 Agent 回合（含进程启动） | 2.85 ms / 8.55 MiB | 未测（参考实现无脚本化模型） | 仅 Rust |
| `import openjiuwen` | — | 51.60 ms / 19.96 MiB | 仅并列展示 |
| prompt 模板渲染 | 0.356 µs/次 | 16.37 µs/次 | 同语义，约 46× |
| 2 节点图执行 | 7.23 µs/次 | 112.3 µs/次 | 同语义，约 15× |

每组 30 次重复（丢弃前 3 次）取中位数，另报 p95；只测 warm cache，未测 LLM 网络延迟、并发扩展与冷磁盘启动。
完整环境、命令、原始样本与限制见 [docs/BENCHMARKS.md](docs/BENCHMARKS.md)。

## 6. Rust 迁移的收益与限制

**收益**
- **内存与启动**：SDK 可达约 3.6 MiB、1.2 ms；完整 Agent 回合 8.6 MiB、2.9 ms。对 CLI、CI 与短生命周期工具，省掉解释器与 94 个模块的导入成本（Python 侧 20.0 MiB、51.6 ms）。
- **纯本地吞吐**：prompt 渲染约 46×、图执行约 15×；这是类型化、无 GIL、无解释器开销的直接结果。
- **架构**：trait 定义模型/工具/会话/runner 的缝，类型化 `Error`（18 个 kind）替代异常约定，`Send + Sync` 与 `CancellationToken` 把并发与取消写进类型，umbrella 只做再导出、公共类型下沉到最低公共祖先。
- **部署与安全**：单一静态二进制；自有代码 0 个 `unsafe`、0 个 `extern "C"`（`unsafe_code = "deny"`），凭据 `SecretString` + 日志脱敏，未实现能力显式失败而非静默降级。

**限制**
- **覆盖面**：仍是子集（约 341 个 `Unsupported` 点）；分布式模式、GPU/ML、厂商深度集成、浏览器/移动传输未实现。
- **性能结论的边界**：倍数只对被测的两个纯本地操作成立；warm cache、单机、无并发扩展，不能外推为"整体快 N 倍"。
- **安全不是自动保证**：Rust 消除的是内存安全类缺陷，不消除死锁、逻辑漏洞、资源耗尽与权限设计错误；FFI/子进程边界真实存在（`libsqlite3-sys`、`nix`/`libc`、`ring`、`inotify`、`/bin/sh`、`git`、LSP、MCP stdio、`docker`）。
- **工程代价**：1,434,767 行 Rust、24,293 个测试函数的维护面；编译时间与二进制体积（CLI 约 34 MB）高于脚本方案；生态与人才熟悉度弱于 Python；上游继续演进时需手工跟踪。
- **非官方**：无 CI、未发布 crates.io，兼容性以 v0.1.20 为基准。

## 7. 结论

两次任务把一个约 5 万行的 Python Agent SDK 迁移为 30 个 crate、143 万行（含约 40% 测试）的可编译 Rust workspace，保住了线格式、Card/会话/工作流契约与 CLI 形状，并在可比的纯本地操作上取得 15–46× 的吞吐提升、启动内存降到约 1/5。
它以 `Unsupported` 而不是空实现标注边界，因此范围是可审计的；同时必须承认它**不是完整等价物**：分布式、GPU/ML 与厂商集成仍在范围之外，性能数字只覆盖已测场景。
