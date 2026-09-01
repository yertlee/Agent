"""Migration skeleton（M0，plan step 3）。

module-guide-06 §6 / runbook §4.3 任务 3。M0 只交付：
- 迁移记录格式（migration id、input/output schema、checksum、owner、date、evidence）；
- 空 skeleton（M0 无任何 schema 变更迁移；建表属 M1）；
- dry-run preflight：只对 ecommerce.db 临时副本执行，源库零写入。

禁止 INSERT OR IGNORE 或等效语句掩盖冲突；冲突/坏数据/计数不符必须失败
并保留诊断（06 §6 迁移协议）。
"""

__all__: list = []
