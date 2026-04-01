# LangGraph Ecommerce Customer Service Agent

这是一个基于 LangGraph 的电商客服 Agent 项目。当前版本已经具备稳定的主图运行骨架，以及 `ORDER / POLICY / AFTERSALES` 三条业务子链路，并在售后子图内接入了 `cache-first` 的快递物流查询能力。

## 项目结构

- `agent/`: 核心 Agent（主图、子图、工具、verifier）
- `app/`: Streamlit 本地演示界面
- `eval/`: 本地可复现评测（cases / evaluator / report）
- `tests/`: 单元测试（物流 pipeline、售后子图等）
- `docs/`: 设计与知识库（`docs/kb/`），以及静态资源（`docs/assets/`）
- `scripts/`: 常用脚本（例如一键跑 eval）

说明：

- `reports/`、`runtime/`、`dist/` 为运行生成物/缓存目录，默认不建议提交到 GitHub（已在 `.gitignore` 中忽略）。
- `*.db` 默认忽略；建议自行在本地生成/维护 `ecommerce.db`，并通过 `ECOMMERCE_DB_PATH` 指向它。

## 当前能力

- 主图保持固定骨架：
  `START -> ingest -> classify -> planner -> dispatch -> specialist/subgraph -> verifier -> (await_user | finalizer | handoff) -> END`
- 主图只按业务能力路由，不按数据源路由。
- `ORDER` 子图负责订单主数据查询。
- `POLICY` 子图负责规则检索和证据整理。
- `AFTERSALES` 子图已经收口为：
  `aftersales_slot_check -> aftersales_intent_split -> order_profile_lookup -> logistics_need_check -> logistics_slot_check -> logistics_snapshot_lookup -> eligibility_check -> create_or_query_aftersales -> result_interpret -> maybe_handoff`
- 物流查询采用 `cache -> kuaidi100/mock provider fallback`。
- verifier 已对常见物流错误码做最小收口：
  - `400 / 408 -> ask_user`
  - `500 / QUERY_TOO_FREQUENT / HTTP|NETWORK|UNKNOWN -> explain_limit`
  - `501 / 502 / 503 / 601 / config|parse error -> handoff`

## 运行前准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，至少补齐下面几项：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `ECOMMERCE_DB_PATH`
- `LOGISTICS_PROVIDER_MODE`
- `KUAIDI100_CUSTOMER`
- `KUAIDI100_KEY`

说明：

- `ECOMMERCE_DB_PATH` 指向本地 sqlite 数据库，ORDER 和 AFTERSALES 工具会读取这里的订单/售后数据。
- `LOGISTICS_PROVIDER_MODE=auto` 时，有快递100配置就走真实 API，没有就回退 mock。
- `LOGISTICS_PROVIDER_MODE=kuaidi100` 时，强制走真实快递100。

### 3. 启动界面

```bash
streamlit run app/streamlit_app.py
```

## 本地评测（Eval）

评测入口为 `eval/run_eval.py`，会输出：

- `reports/agent_v3_eval.json`
- `reports/agent_v3_eval.md`

推荐用脚本一键运行（PowerShell）：

```powershell
.\scripts\run_eval.ps1 -DbPath .\ecommerce.db -LogisticsMode stub -RunInspect
```

或直接运行（PowerShell）：

```powershell
$env:ECOMMERCE_DB_PATH="D:\Myproject\LLMproject1\ecommerce.db"
$env:AGENT_EVAL_LOGISTICS_MODE="stub"  # stub|real，默认 stub
python .\eval\run_eval.py
python .\eval\inspect.py
```

说明：

- `AGENT_EVAL_LOGISTICS_MODE=stub` 默认走稳定的物流 stub，保证评测可复现；`real` 仅作为 smoke，可选。
- 评测 case 已对齐当前订单范围（`20260320001~20260320020`），并重点覆盖“物流感知的售后子图”链路与业务结果。

## 本地 SQL 数据应长什么样（不包含真实数据）

本项目默认从 sqlite 读取订单与售后数据（由你本地提供，不建议上传到 GitHub）。
最小建议表结构如下（字段名以工具层读取为准）：

- `orders`
  - 必需：`order_id`、`phone_last4`、`product_name`、`amount`、`order_status`、`pay_status`、`created_at`、`can_apply_aftersales`
  - 可选（用于物流/售后链路）：`carrier_code`、`tracking_no`

- `aftersales_tickets`
  - 必需：`ticket_id`、`order_id`、`phone_last4`、`service_type`、`reason`、`ticket_status`、`created_at`、`updated_at`

你可以用自己的 seed 脚本初始化这些表，或直接导入已有业务数据，只要满足字段即可。

## 物流接入说明

- 订单主数据仍来自本地订单工具，不由快递100替代。
- 物流实时事实只在售后子图内部按需查询。
- 查询顺序为：
  `order_profile_lookup -> logistics_need_check -> logistics_slot_check -> logistics_snapshot_lookup`
- `logistics_snapshot` 是运行时消费的标准化结构。
- 本地 cache 只用于缓存、调试和限频保护，不作为主业务状态源。

## 测试

运行全部测试：

```bash
python -m unittest discover -s tests -v
```

重点测试覆盖：

- cache hit / miss
- 快递100签名与请求格式
- 物流错误结构化输出
- verifier 对物流错误码的收口
- 售后子图创建、查询、人工转接和恢复分支

## 下一步规划：Redis 等后端持续性服务

当前运行态主要依赖内存 checkpointer 与本地文件 cache，适合本地开发与评测。下一步如果要上更“持续”的后端服务，建议方向：

- **Redis 会话/状态存储**：把线程状态、对话上下文、关键 `trace_tags` 持久化，支持多实例与重启恢复。
- **Redis 缓存层**：将物流快照 cache、RAG 召回缓存等统一纳入 Redis，增强可观测与统一失效策略。
- **服务化拆分**：将订单/售后/物流 provider 抽象为独立服务或适配器层，便于接入真实 API 与权限控制。
- **CI/CD 与配置管理**：环境变量、密钥与配置按环境分层（dev/staging/prod），避免配置漂移。
