# Agent Eval 业务数据设计
## 1. 数据分组

`orders` 分为 A~F 组，分别覆盖：

- A组：订单查询成功 + 售后允许且可创建（无 active ticket）
- B组：订单可查但售后不允许（稳定触发 `AFTERSALES_NOT_ALLOWED`）
- C组：手机号错误/身份校验失败（稳定触发 `PHONE_MISMATCH`）
- D组：已有进行中售后（重复申请稳定触发 `AFTERSALES_ALREADY_EXISTS`；同时作为查询正样本）
- E组：有历史售后但当前无 active（允许再次创建）
- F组：售后查询混合（部分订单无工单稳定触发 `AFTERSALES_NOT_FOUND`）

`aftersales_tickets` 分为 T1~T18：

- T1~T6：D组 active 工单（状态覆盖待审核/审核通过/退款处理中/退货待寄回）
- T7~T12：E组 closed/history 工单（不属于 active 集合）
- T13~T15：F组 query 样本（混合场景正样本）
- T16~T18：同一订单多工单历史，用 `updated_at` 控制 `query_aftersales` 返回“最新一条”。

## 2. 时间戳与“最新工单”逻辑

`query_aftersales` 使用 `ORDER BY updated_at DESC, created_at DESC LIMIT 1`。

因此同一订单多条工单样本必须通过 `updated_at/created_at` 精确控制“最终返回哪条”。

## 3. 后续 eval cases 编写建议

- 订单查询类：优先覆盖 `first_route=order` 与 `PHONE_MISMATCH` / `ORDER_NOT_FOUND`
- 售后创建类：覆盖 `AFTERSALES_NOT_ALLOWED` / `AFTERSALES_ALREADY_EXISTS` / history-only 允许创建
- 售后查询类：覆盖 `AFTERSALES_NOT_FOUND` 以及“最新工单取值正确性”

# Agent Eval 业务数据设计（V2）

本设计文档对应 `seed_agent_eval_v2.sql`，目标是在**不修改 Agent 主流程**、不扩充 RAG 的前提下，仅通过增量补充 `ecommerce.db` 的两张业务表：

- `orders`
- `aftersales_tickets`

用于提升 `get_order_info` / `aftersales_service` 的本地业务评测数据质量与覆盖面。

## 1. 表结构与业务逻辑要点（与 tools.py 对齐）

- **身份校验**：`order_id` 存在后，校验 `phone_last4`（不匹配 -> `PHONE_MISMATCH`）。
- **售后创建限制**：
  - `orders.can_apply_aftersales != 1` -> `AFTERSALES_NOT_ALLOWED`
  - 若该 `order_id` 存在 **active** 工单（`待审核` / `审核通过` / `退款处理中` / `退货待寄回`）-> `AFTERSALES_ALREADY_EXISTS`
  - 允许的 `service_type`：`退款` / `退货` / `换货`
- **售后查询**：`query` 返回该订单最新工单（`ORDER BY updated_at DESC, created_at DESC LIMIT 1`）
  - 无工单 -> `AFTERSALES_NOT_FOUND`

## 2. orders 分组设计（A~F）

> 所有新增订单号统一落在 `20260301001`~`20260301036`，便于后续 eval case 维护；商品名/金额/状态/手机号后四位保持多样性，避免过度模板化。

### A组：正常订单查询 + 售后允许且可成功创建（6条）

- **用途**：
  - `get_order_info` happy path（`OK`）
  - `aftersales_service(create)` happy path（可创建成功 -> `OK`）
  - 混合对话：查单后追加“我要退款/退货/换货”
- **关键约束**：
  - `can_apply_aftersales = 1`
  - 不预置 active 工单（便于创建成功）
- **订单范围**：`20260301001` ~ `20260301006`

### B组：可查到订单，但明确不允许售后（6条）

- **用途**：
  - 稳定触发 `AFTERSALES_NOT_ALLOWED`
  - 评测 Agent 是否能解释“为什么不支持售后”并引导用户
- **关键约束**：
  - `can_apply_aftersales = 0`
  - 手机号正确时可查单成功，但创建售后必须失败
- **订单范围**：`20260301007` ~ `20260301012`

### C组：用于手机号错误 / 身份校验失败（6条）

- **用途**：
  - 稳定触发 `PHONE_MISMATCH`
  - 多轮补槽：用户反复提供不同后四位仍失败
- **关键约束**：
  - 订单本身正常（`can_apply_aftersales = 1`）
  - case 层故意输入错误 `phone_last4`
- **订单范围**：`20260301013` ~ `20260301018`

### D组：已有进行中售后，不允许重复申请（6条）

- **用途**：
  - 稳定触发 `AFTERSALES_ALREADY_EXISTS`
  - `aftersales_service(query)` 正样本（可查到进行中状态）
- **关键约束**：
  - `can_apply_aftersales = 1`
  - 每个订单至少 1 条 active 工单，状态覆盖 active 集合
- **订单范围**：`20260301019` ~ `20260301024`

### E组：有历史售后但当前无 active ticket（6条）

- **用途**：
  - 验证“有历史记录 ≠ 当前已有进行中售后”
  - `create` 应允许新建（不会命中 active 检查）
  - `query` 可返回最近历史单（用于对话追问）
- **关键约束**：
  - `can_apply_aftersales = 1`
  - 仅预置 closed/completed 状态（不在 active 集合）
- **订单范围**：`20260301025` ~ `20260301030`

### F组：售后查询 / 混合场景 / 边界测试（6条）

- **用途**：
  - `query` 有/无记录混合
  - 评测“查到订单后追问售后进度/查不到售后时如何引导”
- **关键约束**：
  - 其中 3 条有工单，3 条无工单（用于 `AFTERSALES_NOT_FOUND`）
- **订单范围**：`20260301031` ~ `20260301036`

## 3. aftersales_tickets 分组设计（T1~T18）

### T1~T6：D组 active tickets（6条）

- **用途**：
  - `create` 时稳定触发 `AFTERSALES_ALREADY_EXISTS`
  - `query` 时返回进行中进度（`OK`）
- **状态覆盖**：`待审核` / `审核通过` / `退款处理中` / `退货待寄回`

### T7~T12：E组 closed/history tickets（6条）

- **用途**：
  - 用 closed 状态构造“有历史但无 active”的订单
  - `create` 应允许新建；`query` 可返回历史单做解释
- **状态示例**：`已退款` / `已关闭` / `已完成`

### T13~T15：F组 query samples（3条）

- **用途**：
  - 售后查询正样本（返回 `OK`）
  - 混合对话：用户先问订单，再问售后进度
- **无工单订单（用于 AFTERSALES_NOT_FOUND）**：
  - `20260301032`
  - `20260301034`
  - `20260301036`

### T16~T18：latest ticket 取值验证（3条）

- **用途**：验证 `query` 确实按 `updated_at DESC, created_at DESC` 取最新工单
- **设计**：
  - `20260301031` 同时存在 T16（旧）与 T13（新）=> 应返回 **T13**
  - `20260301026` 同时存在 T17（旧）与 T8（新） => 应返回 **T8**
  - `20260301033` 同时存在 T18（旧）与 T14（新）=> 应返回 **T14**

## 4. 推荐如何基于这些数据编写后续 eval cases

### 4.1 最小覆盖（强烈建议先实现）

- **订单查询 OK**：A1 `20260301001 / 4812`
- **手机号错误**：对 A1 输入错误后四位（如 `0000`）触发 `PHONE_MISMATCH`
- **不允许售后**：B1 `20260301007 / 1935` 创建售后触发 `AFTERSALES_NOT_ALLOWED`
- **已有进行中售后**：D1 `20260301019 / 4402` 创建售后触发 `AFTERSALES_ALREADY_EXISTS`
- **查询无售后**：F2 `20260301032 / 7641` 查询售后触发 `AFTERSALES_NOT_FOUND`

### 4.2 排序与“最新工单”验证

- **同订单多工单**：查询 `20260301031 / 3185` 应返回 **T13**（而不是旧的 T16）

### 4.3 售后创建 happy path（新建成功）

- **无预置 active、且允许售后**：A组任一订单创建售后应返回 `OK`
- **有历史但无 active**：E1 `20260301025 / 3916` 创建售后应返回 `OK`（不会被误判为已有进行中）

## 5. 与旧 demo case 的兼容性

本 seed 为**增量插入**，不包含任何 DROP/DELETE。

- 不会删除或覆盖现有 demo 订单（例如 `20260226003` / `20260226004` / `20260226005` 仍可用）
- 采用 `INSERT OR IGNORE`：若主键已存在则跳过，便于重复导入与迭代扩充

