# 快速开始

## 1. 安装依赖

```bash
pip install -r requirements-dev.txt
```

## 2. 无 Key 的确定性路径（推荐先跑这个）

不加载 `.env`、不联网、不调用模型：

```bash
# 生成隔离的合成夹具：订单 / 商品 / 售后 / 物流 四个库 + manifest
python scripts/r5_seed_synthetic_data.py --output-dir .tmp/r5-synthetic

# 在该夹具上跑关键词基线与 oracle 上界
python -m eval.r5_router_planner_eval --split dev --candidate keyword_router --data-dir .tmp/r5-synthetic
python -m eval.r5_router_planner_eval --split dev --candidate oracle        --data-dir .tmp/r5-synthetic

# 测试
python -m pytest -q
```

## 3. 真实模型运行（需要 API Key）

复制 `.env.example` 为 `.env`，填写：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

然后运行：

```bash
# 有 Key 时直接跑（默认使用仓库演示数据）
python -m eval.r5_router_planner_eval --split dev --candidate real

# 或指定隔离夹具
python -m eval.r5_router_planner_eval --split dev --candidate real --data-dir .tmp/r5-synthetic
```

说明：

- 推理型模型建议输出预算 ≥ 8000 token，否则会在结构化输出前耗尽预算。
- 数据库（`*.db`）不入库，用上面的 seed 脚本生成；或在 `.env` 中用 `ECOMMERCE_DB_PATH` 指向本地库。

## 4. 常用命令

| 目的 | 命令 |
|---|---|
| 生成合成夹具 | `python scripts/r5_seed_synthetic_data.py --output-dir .tmp/r5-synthetic` |
| 构建数据集 | `python scripts/r5_build_datasets.py` |
| 关键词基线 | `python -m eval.r5_router_planner_eval --split dev --candidate keyword_router` |
| oracle 上界 | `python -m eval.r5_router_planner_eval --split dev --candidate oracle` |
| 真实模型 | `python -m eval.r5_router_planner_eval --split dev --candidate real` |
| 全量测试 | `python -m pytest -q` |
| 敏感扫描 | `python scripts/r5_secret_scan.py` |

## 5. 结果与文档

- 结果与限制：[`results.md`](results.md)
- 架构说明：[`architecture.md`](architecture.md)
- 冻结证据：[`evidence/`](evidence/)
