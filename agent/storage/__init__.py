"""Storage package（M0 骨架）。

边界（module-guide-06 §6 / runbook §4.3）：
- M0 只交付 repository/dialect 接口、migration skeleton、backup/restore、preflight/postflight；
- 不建任何 v1.2 目标表（Run/PlanRevision/Task/... 属 M1）；
- 对源库 ecommerce.db 只读，任何写操作仅在隔离副本上执行。

注意：包内模块通过完整路径导入（如 `from agent.storage.backup import create_backup`），
包 `__init__` 不做 re-export，避免 `python -m agent.storage.<module>` 的双重导入告警。
"""

__all__: list = []
from .repositories import M1Repository

__all__ = ["M1Repository"]
