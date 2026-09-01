import sqlite3
import unittest

from agent.storage.migrations.m1 import apply_m1_schema


class M1MigrationAtomicityTests(unittest.TestCase):
    def test_injected_failure_rolls_back_every_m1_table(self):
        conn = sqlite3.connect(":memory:")
        with self.assertRaises(RuntimeError):
            apply_m1_schema(conn, fail_after=3)
        names = {r[0] for r in conn.execute("select name from sqlite_master where type in ('table','trigger')")}
        self.assertEqual(names, set())

    def test_reapply_is_idempotent(self):
        conn = sqlite3.connect(":memory:")
        apply_m1_schema(conn); apply_m1_schema(conn)
        self.assertIn("results", {r[0] for r in conn.execute("select name from sqlite_master where type='table'")})

    def test_all_m1_tables_have_lifecycle_timestamps(self):
        conn = sqlite3.connect(":memory:"); apply_m1_schema(conn)
        tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
        expected = {"sessions", "runs", "plan_revisions", "tasks", "task_attempts", "results", "checkpoints", "trace_outbox", "trace_manifests", "late_event_audit"}
        self.assertTrue(expected.issubset(tables))
        for table in expected:
            columns = {r[1] for r in conn.execute(f"pragma table_info({table})")}
            self.assertTrue({"created_at", "updated_at"}.issubset(columns), table)
