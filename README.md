# LangGraph-Native 电商客服 Agent（公开仓库骨架版）

这是一个基于 **LangGraph** 的“可观测 Agent Runtime / 编排系统”，而不是黑盒聊天机器人。
主流程通过显式的 **planner**、**specialists**、**verifier** 控制计划生成、工具调用、证据验证与收口。
同时提供 **RAG 规则检索**、**LangGraph `interrupt()` 补槽恢复**、以及 **本地评测 + LangSmith compare**。

## 核心能力

1. **StateGraph 主图 + 显式规划**
   - `graph_agent` 负责主图编排
   - planner 输出结构化计划，必要时可重规划（replan）
   - verifier 以规则/证据为先，决定是否追问、收口或转人工

2. **Specialists 分工**
   - Order：订单身份校验与订单查询
   - AfterSales：售后查询/创建（当前为工具层骨架，后续接真实业务）
   - Policy：规则检索与证据整理（RAG 证据交给 verifier 收口）

3. **RAG 规则检索与证据化输出**
   - 从 `docs/kb/*.md` 构建向量索引
   - query rewrite -> 检索 -> 结构化 evidence，最终由 verifier/finalizer 使用

4. **Streamlit 演示**
   - `app/streamlit_app.py` 可视化展示：当前节点、计划、interrupt 追问、工具与证据等

5. **Eval / Compare**
   - `eval/run_eval.py`：本地跑 curated cases 并输出 `reports/agent_v3_eval.json|md`
   - `eval/compare.py`：LangSmith pairwise comparative eval（planner prompt、query rewrite）

## 项目结构

```text
project_root/
├─ README.md
├─ .gitignore
├─ .env.example
├─ requirements.txt
├─ app/
│  └─ streamlit_app.py
├─ agent/
│  ├─ graph_agent.py
│  ├─ llm.py
│  ├─ planner.py
│  ├─ prompts.py
│  ├─ runtime.py
│  ├─ specialists.py
│  ├─ state.py
│  ├─ verifier.py
│  ├─ agent_tools.py
│  ├─ tools.py
│  ├─ tool_registry.py
│  ├─ rag_retriever.py
│  ├─ schemas.py
│  └─ langsmith_utils.py
├─ eval/
│  ├─ run_eval.py
│  ├─ compare.py
│  ├─ inspect.py
│  ├─ cases_demo.py
│  └─ cases_extended.py
└─ docs/
   ├─ agent_eval_data_design.md
   └─ kb/
      └─ *.md  （RAG 规则文档）
```

## 快速开始

### 1) 创建虚拟环境

```bash
python -m venv .venv
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

### 3) 配置环境变量

复制并编辑 `.env.example` 为 `.env`：

- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`
- `LANGSMITH_API_KEY`（如果要跑 LangSmith eval）

### 4) 启动 Streamlit

```bash
streamlit run app/streamlit_app.py
```

## 关于数据库与 seed SQL

当前公开仓库 **不包含** sqlite 数据库 `ecommerce.db`、也不包含任何 seed SQL / init 脚本。
因此：

- 工具层（`agent/tools.py`）仍保留 SQLite 调用逻辑，但需要你在运行时设置 `ECOMMERCE_DB_PATH`
- 或在后续将 order/aftersales provider 替换为真实 API / mock provider

## 运行本地评测

如果你本地准备了 `ecommerce.db`，请先设置：

```bash
set ECOMMERCE_DB_PATH=你的本地路径/ecommerce.db
```

然后运行：

```bash
python eval/run_eval.py
```

会生成：

- `reports/agent_v3_eval.json`
- `reports/agent_v3_eval.md`


