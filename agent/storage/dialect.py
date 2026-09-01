"""SQL dialect 边界（M0，plan step 2）。

module-guide-06 §6：SQLite→MySQL 需要 repository、SQL dialect、事务隔离、
类型映射与 migration adapter，不是纯配置切换。M0 只交付 SQLite dialect
与 MySQL 接缝占位（不实现）；占位符/类型映射边界在此集中，repository
不得散落 SQL 方言细节。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional, Protocol


class Dialect(Protocol):
    """dialect 接缝（06 §6）：M0 仅 SQLite 实现，MySQL 留待后续里程碑。"""

    paramstyle: str
    name: str

    def connect(self, db_path: str, readonly: bool = False) -> Any: ...


class SQLiteDialect:
    """SQLite 方言：连接工厂（row_factory、只读 URI）、占位符/类型映射边界。"""

    name = "sqlite"
    paramstyle = "qmark"          # 占位符边界：repository 一律使用 ? 占位符
    supports_returning = False    # 类型映射边界标记：SQLite 无 RETURNING 依赖

    def connect(self, db_path: str, readonly: bool = False) -> sqlite3.Connection:
        """创建连接。

        readonly=True 时以 file:...?mode=ro URI 打开（源库/校验场景零写入）；
        写连接默认开启 PRAGMA foreign_keys（06 §6 迁移协议前置）。
        """
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"database file not found: {path}")
        if readonly:
            uri = "file:" + path.resolve().as_posix() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        if not readonly:
            conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def placeholder(self) -> str:
        return "?"


class MySQLDialectStub:
    """MySQL dialect 接缝占位：显式未实现，防止误当成"纯配置切换"（06 §6）。"""

    name = "mysql"
    paramstyle = "format"

    def connect(self, db_path: str, readonly: bool = False) -> Any:
        raise NotImplementedError(
            "MySQL dialect is a seam only in M0 (module-guide-06 §6); "
            "implementation is out of M0 scope"
        )


def transaction(conn: sqlite3.Connection):
    """事务入口（context manager）：失败即回滚，成功才提交（06 §6 单事务语义）。"""
    return _Transaction(conn)


class _Transaction:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._committed = False

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self._conn.commit()
            self._committed = True
        else:
            self._conn.rollback()
        return False

    @property
    def committed(self) -> bool:
        return self._committed


SQLITE_DIALECT = SQLiteDialect()
