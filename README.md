# 电商客服 Multi-Agent

一个面向**商品 / 订单 / 物流 / 政策 / 售后**五类请求的客服 Multi-Agent 系统。用户用自然语言提问，LLM Router/Planner 把它转成结构化计划，Supervisor 按计划把任务派发给领域 Agent（版本化 Agent 间消息协议），真实调用工具完成任务；售后这类高风险动作走"预览 → 用户确认 → 服务端复核 → 单事务提交"。

## 核心结果

dev 集 123 个唯一 case，单次运行；真实模型、关键词基线与 oracle 在同一套仓库演示数据上测量。

| 指标 | 本系统 | 关键词基线 | oracle |
|---|---:|---:|---:|
| 意图 Core Macro-F1（11 类） | **0.853** | 0.463 | 1.000 |
| exact handoff | **0.764** | 0.350 | — |
| 真实任务完成率 | **0.769** | 0.479 | 1.000 |
| 不必要澄清率（越低越好） | **0.057** | 0.390 | 0.000 |

`MULTI_INTENT` 是辅助标签，在 dev 集中的 gold support 为 0，因此主指标采用 11 类 Core Macro-F1；保留全部 12 类时，本系统兼容值为 0.782。指标定义、分母和证据见 [`docs/evaluation.md`](docs/evaluation.md) 与 [`docs/results.md`](docs/results.md)。

## 架构

```mermaid
flowchart TD
    U[用户消息] --> R[Router: 意图 / 实体候选 / 缺失信息]
    R --> P[Planner: DAG 节点 + 依赖 + typed binding]
    P --> V{计划校验器}
    V -->|合法| N[运行时归一化: 能力参数契约]
    N --> A[A2A 运行时: 幂等 / 依赖 / 有界重试 / 迟到隔离]
    A --> D[Product · Order · Logistics · Policy · AfterSales]
    D --> RES[canonical Result]
    RES --> OUT[回答 / 澄清 / 拒绝 / 人工升级 / 受控写]
```

几条关键设计：

- **单一权威计划**：能力集合与依赖从节点/边派生，不要求模型重复填写拓扑或能力汇总。
- **模型与规则分工**：结构化编号（订单号、运单号、SKU、手机号、售后类型、承运商）由确定性规则给候选，语义角色由模型判断；会话手机号与上游派生字段由运行时按契约补齐，不向用户追问。
- **受控写**：动作化确认令牌（申请/取消/修改分离）+ 状态 CAS + 幂等指纹 + 审计与 outbox，同一事务提交；单轮停在"等待确认"。
- **执行型评测**：模型生成的计划经同一个运行时真实执行，检查终态与业务结果；无效计划不执行、不计成功。

例如“查一下订单物流，并判断是否满足退款条件”会生成一张 DAG：先读取订单，再并行查询物流与售后资格。运行时从上游订单结果中取得运单号和承运商，并绑定到下游节点。

详见 [`docs/architecture.md`](docs/architecture.md)。

## 快速开始

```bash
pip install -r requirements-dev.txt

# 无 Key：生成隔离合成夹具并跑确定性的基线/上界
python scripts/r5_seed_synthetic_data.py --output-dir .tmp/r5-synthetic
python -m eval.r5_router_planner_eval --split dev --candidate keyword_router --data-dir .tmp/r5-synthetic
python -m eval.r5_router_planner_eval --split dev --candidate oracle        --data-dir .tmp/r5-synthetic
python -m pytest -q tests/r5
```

真实模型运行需要 `.env` 中的 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`：

```bash
python -m eval.r5_router_planner_eval --split dev --candidate real
```

完整命令见 [`docs/quickstart.md`](docs/quickstart.md)。

## 仓库结构

- `agent/`：Router/Planner、计划契约与参数绑定、A2A 运行时、五领域 repository/工具、安全写与确认令牌、Trace/存储。
- `data/knowledge_base/`：项目自建政策与商品知识语料。
- `eval/`：统一评测入口、真实执行链、真实模型 provider 与冻结数据集。
- `scripts/`：合成夹具、数据集构建、敏感扫描。
- `tests/r5/`：当前系统的单元与集成测试。
- `docs/`：架构、运行说明、结果与冻结证据。

## 文档

- [`docs/architecture.md`](docs/architecture.md)：系统架构与五领域职责。
- [`docs/a2a-protocol.md`](docs/a2a-protocol.md)：消息契约、调度与故障语义。
- [`docs/safe-write.md`](docs/safe-write.md)：售后写操作的确认和事务边界。
- [`docs/evaluation.md`](docs/evaluation.md)：数据、指标与对照设计。
- [`docs/results.md`](docs/results.md)：冻结结果和失败边界。

## 项目边界

- 数据为**项目自建合成样本**，非外部客户流量；来源分层（订单本地只读投影、物流订单派生快照、商品/售后自建、政策确定性夹具）。
- 写操作评测停在"等待用户确认"；"确认后提交完成"未测量。
- 未实现 replan，不主张动态规划增益。
- 任务完成率不评估最终回答文案质量。
- 不声明生产可用性或外部泛化。
