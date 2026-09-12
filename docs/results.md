# 评测结果

dev 集 123 个唯一 case，单次运行。真实模型、关键词基线与 oracle 使用同一套仓库演示数据。

## 1. 测的是什么

- **数据集**：`eval/datasets/r5/`，dev 123 个唯一语义 case（111 个 family、51 个多意图、18 个澄清），项目自建合成样本。
- **被测对象**：自然语言 → Router/Planner 生成结构化计划 → 运行时按能力参数契约归一化 → 经 A2A 运行时真实调用领域工具 → 检查终态与业务结果。
- **口径**：任务完成率 = 到达期望终态（回答 / 澄清 / 拒绝 / 待确认 / 人工）且必需读取成功、无未请求写。**不检查最终回答文字，也不代表写操作已提交**（写操作停在"等待用户确认"）。
- **分母**：123 例中 117 例可评估；6 例"数据冲突→问用户"因缺少冲突世界夹具被排除并记录。必需读取成功率分母 94；不必要澄清率分母 105。

## 2. 结果

| 指标 | 本系统 | 关键词基线 | oracle 上界 |
|---|---:|---:|---:|
| 意图 Core Macro-F1（11 类） | **0.853** | 0.463 | 1.000 |
| exact handoff | **0.764** | 0.350 | — |
| 真实任务完成率 | **0.769** | 0.479 | 1.000 |
| 不必要澄清率（越低越好） | **0.057** | 0.390 | 0.000 |

说明：

- **Core 11 类**排除了 `MULTI_INTENT`；该辅助标签在 dev 中的 gold support 为 0，放入宏平均会造成口径失真。保留全部 12 类时，本系统兼容值为 **0.782**。
- 任务完成率为 **90/117**；按全部 123 个 case 计算为 0.732。6 个“数据冲突后向用户澄清”的 case 因缺少冲突世界夹具而不进入主分母。
- 其他诊断指标：计划必需目标覆盖 0.780、计划符合性 0.772、必需读取成功率 0.777、schema 有效率 1.000、延迟 p50 约 4.2 秒。
- oracle 用 gold 构造计划，只用于验证评测链存在可达上界，不代表系统能力。其计划可能包含合法可选能力，因此不把 exact handoff 作为 oracle 上界。
- 相比关键词基线，本系统的意图 Core Macro-F1 提升 0.390，任务完成率提升 0.290，不必要澄清率降低 0.333。

## 3. 失败分析

117 个可评估 case 中有 27 个没有达到期望终态，主要表现为能力选择错误、多意图覆盖不全、售后动作遗漏写节点以及必要澄清判断错误。运行时会拒绝模型额外计划的未请求写操作，因此这类安全拒绝不会被计为任务完成。

## 4. 证据与溯源

- 逐例结果、关键词基线、oracle 上界、敏感扫描、数据集 manifest、汇总：`docs/evidence/`。
- 运行上下文（模型别名、provider 主机、prompt、预算和数据哈希）内嵌在 `docs/evidence/router_planner_dev_real_full.json` 的 `run_context`。
- 机器可读汇总：`docs/evidence/summary.json`。

## 5. 如何运行

见 [`quickstart.md`](quickstart.md)。无 Key 的确定性路径不加载 `.env`、不联网：

```bash
pip install -r requirements-dev.txt
python scripts/r5_seed_synthetic_data.py --output-dir .tmp/r5-synthetic
python -m eval.r5_router_planner_eval --split dev --candidate keyword_router --data-dir .tmp/r5-synthetic
python -m eval.r5_router_planner_eval --split dev --candidate oracle        --data-dir .tmp/r5-synthetic
python -m pytest -q tests/r5
```

## 6. 限制

- 数据为项目自建合成样本，非外部客户流量；来源分层（订单为本地只读投影、物流为订单派生快照、商品/售后自建、政策为确定性夹具）。
- 写操作停在待确认；"确认后提交完成"未测量。
- 未实现 replan，不主张动态规划增益。
- 任务完成率不含最终回答文案质量。
- 真实模型结果来自一次冻结运行；逐例结果与运行上下文已保留，不能据此声称跨模型或跨数据集泛化。
