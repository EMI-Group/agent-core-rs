# 从 Python 到 Rust：openJiuwen 智能体平台重写成果

> **执行摘要｜已形成覆盖核心运行链路的 Rust 底座，启动、内存与本地执行效率显著优于原实现，可以进入试点。**

本项目基于 [openJiuwen agent-core](https://github.com/openJiuwen-ai/agent-core)（Python SDK v0.1.20）独立重写，用 Rust 重新组织智能体基础设施，保持模型接口、数据格式与工作流契约不变。覆盖核心运行链路；少数依赖 GPU/ML 栈、厂商 SDK 或外部数据库的能力保留 API 形状并显式返回 Unsupported。

## 一、投入与产出

| **记录费用** | **累计 Agent 实例** | **任务墙钟合计** | **历史提交** |
|:---:|:---:|:---:|:---:|
| **$150.02** | **3,570** | **113 小时 42 分** | **3,765 次** |
| 精确 $150.024283 | 非人数、非峰值并发 | 含等待与排队 | 主体 3,625 + 演进 140 |

约 150 美元的模型费用、3,570 个 Agent 实例、不到 5 天的任务墙钟，换来一套可离线验证的完整底座。两次有效任务分别完成主体重写（3,625 次提交）与后续演进（140 次提交）。

| **交付结构** | **代码规模** | **测试资产** | **可体验成果** |
|:---:|:---:|:---:|:---:|
| **30 个 crate（模块包）** | **143.5 万物理行** | **24,293 个测试函数** | **26 个可运行示例** |
| 分层工作区 | 2,934 个 `.rs` 文件 | 593 个集成测试二进制 | 默认离线、无需模型凭据 |

其中生产代码 80.3 万行、内嵌单元测试 32.4 万行、集成测试 29.5 万行、示例 1.19 万行；tokei 计 109.9 万行代码，测试代码占 43%。

## 二、交付能力与架构

| 能力板块 | 已交付的主要能力 | 对应用的意义 |
|---|---|---|
| **模型与知识** | OpenAI / Anthropic / Responses 客户端，提示词、工具与 MCP，记忆与检索 | 复用模型与工具生态，减少重复接入 |
| **运行与编排** | Agent / ReAct、图与工作流、上下文管理、会话与检查点、护栏 | 支持多步任务、状态恢复与执行治理 |
| **编程与协作** | DeepAgent 任务循环、工作区与权限、20 个内置工具、子智能体与技能、进程内团队及跨进程接入 | 覆盖主要编程 Agent 与团队协作链路 |
| **应用与扩展** | CLI、A2A / OTel、开发工具、symphony、RSI / auto-harness、agent-evolving 核心 | 提供观测、评测与优化的可用子集 |

<p align="center"><img src="docs/assets/agent-run-flow.svg" alt="智能体运行链路：入口、Runner、Agent、模型与工具，以及共享状态和治理服务" width="560"></p>

入口统一接入，运行时调度智能体，智能体组合模型与工具；会话、上下文、记忆与治理作为共享服务复用。

### 架构总览：共享底座，而非单个演示程序

<p align="center"><img src="docs/assets/architecture-layers.svg" alt="30 个 crate 的分层架构" width="760"></p>

| 层 | 代表性 crate | 一句话职责 |
|---|---|---|
| **基础与模型工具** | `common`, `macros`, `harness-protocol`, `foundation`, `retrieval`, `sysop` | 错误、Card/Config、日志、密钥与 HTTP；提示词、模型客户端、工具/MCP、检索与系统操作 |
| **状态治理与编排** | `session`, `memory`, `security`, `graph`, `context`, `workflow` | 会话、检查点与长期记忆，护栏，Pregel 图引擎、上下文管理与工作流组件 |
| **智能体与框架扩展** | `agent`, `multiagent`, `runner`, `core`, `harness`, `harness-providers`, `extensions`, `dev-tools`, `symphony` | Agent/ReAct、团队调度与 Runner；kernel umbrella、DeepAgent 框架、provider 注册表与 A2A/OTel |
| **应用、入口与开发** | `agent-teams`, `rsi`, `agent-evolving`, `auto-harness`, `openjiuwen`, `cli`, `test-support`, `e2e`, `examples` | 团队产品、自动优化与评测；顶层 umbrella、CLI；假 LLM 服务器、跨 crate 场景与示例 |

两条结构性约定：**umbrella 只做 re-export**，`openjiuwen` 与 `openjiuwen-core` 不定义行为；**公共类型下沉到最低公共祖先**，避免横向依赖。最长依赖链为 13 条边、14 个 crate，架构深度可控。

## 三、性能、部署与安全

| 同机本地微基准（每次操作中位数） | Rust | Python | 倍数 |
|---|---:|---:|---:|
| 提示词模板渲染 | 0.356 µs | 16.37 µs | **约 46×** |
| 两节点图执行 | 7.23 µs | 112.3 µs | **约 15×** |

测量口径：同机、warm page cache、无网络，Rust release（lto=thin、codegen-units=1）对照 CPython 3.11.15。

**进程级对比**：Python `import openjiuwen` 为 51.60 ms / 19.96 MiB；导入后跑通第一个两节点图需 1.68 s / 235.7 MiB；Rust 一次完整脚本化 Agent 回合（含进程启动）为 2.85 ms / 8.55 MiB，最小 SDK 程序为 1.20 ms / 3.60 MiB。即：启动快数百倍，首个工作流从 1.68 秒降到 2.85 毫秒；峰值内存从 236 MiB 降到 8.55 MiB，约为 1/27。

- **部署更轻**：CLI 交付为单一 34 MB 二进制，不需要 Python 解释器或 virtualenv；部署侧第三方发行包从 173 个 PyPI 包（含 numpy、pandas、transformers、grpcio、aiohttp）降为 0，依赖由 Cargo.lock 锁定，默认离线构建。
- **内存更安全**：自有代码 143 万行 0 处 `unsafe`（workspace lint `unsafe_code = deny`），无 GC，所有权模型在编译期消除缓冲区溢出、释放后使用等内存安全类缺陷。
- **治理更显式**：统一类型化错误、异步取消与凭据脱敏；43% 的代码是测试，10 个 crate 提供假 LLM 服务器与跨 crate 场景，测试全程无网络、无凭据、无墙上时钟依赖。

## 四、成熟度与建议

**验证已通过**：全工作区 **24,293 个测试函数、593 个集成测试二进制**在离线环境全部通过，格式与静态检查、文档构建、7 组特性门禁、示例构建同步为绿，记录见 [docs/PARITY.md](docs/PARITY.md)。

**建议**：优先在编程助手、CLI 自动化与本地工作流三个方向做受控试点，以业务成功率、真实模型成本与部署资源作为验收口径；规模化推广前补齐 CI 与版本发布机制。分布式运行、GPU/ML 栈及部分厂商/外部系统适配不在当前范围，相关入口显式返回 Unsupported。

**详见**：[项目与架构](README.md) · [统计与历史口径](docs/REWRITE_APPENDIX.md) · [性能测量](docs/BENCHMARKS.md) · [覆盖范围与验证记录](docs/PARITY.md)
