import sqlite3
import tempfile
import unittest

from agent.order_slice import run_order_read_slice
from agent.storage.repositories import M1Repository


class M1OrderSliceTests(unittest.TestCase):
    def test_read_slice_uses_new_chain_and_not_source_write(self):
        before = sqlite3.connect("ecommerce.db").execute("select count(*) from orders").fetchone()[0]
        with tempfile.TemporaryDirectory() as temp:
            runtime_db = temp + "/runtime.db"
            result = run_order_read_slice(source_db="ecommerce.db", runtime_db=runtime_db, order_id="does-not-exist", phone_last4="0000")
            reopened = M1Repository(runtime_db, initialize=False)
            persisted_run = reopened.conn.execute("select status,state_version from runs where run_id=?", (result["terminal_run"]["run_id"],)).fetchone()
            persisted_attempt = reopened.conn.execute("select status from task_attempts where attempt_id=?", (result["terminal_attempt"]["attempt_id"],)).fetchone()
            persisted_result = reopened.get_result(result["result"].result_id)
            self.assertEqual(tuple(persisted_run)[0], "FAILED")
            self.assertEqual(persisted_attempt[0], "FAILED")
            self.assertIsNotNone(persisted_result)
            self.assertEqual(reopened.conn.execute("select count(*) from trace_outbox where run_id=?", (result["terminal_run"]["run_id"],)).fetchone()[0], result["trace_event_count"])
            events = [__import__("json").loads(row[0]) for row in reopened.conn.execute("select envelope_json from trace_outbox where run_id=? order by seq_no", (result["terminal_run"]["run_id"],))]
            by_id = {e["trace_id"]: e for e in events}
            result_event = next(e for e in events if e["event_type"] == "RESULT_WRITTEN")
            self.assertEqual(by_id[result_event["parent_event_id"]]["event_type"], "TOOL_RETURNED")
            reopened.close()
        after = sqlite3.connect("ecommerce.db").execute("select count(*) from orders").fetchone()[0]
        self.assertEqual(result["business_code"], "ORDER_NOT_FOUND")
        self.assertGreaterEqual(result["trace_event_count"], 10)
        self.assertIn("TASK_STATE_CHANGED", result["trace_event_types"])
        self.assertIn("CHECKPOINT", result["trace_event_types"])
        self.assertIn("STATE_DELTA", result["trace_event_types"])
        self.assertIn("ERROR", result["trace_event_types"])
        self.assertEqual(before, after)

    def test_success_slice_has_result_written_without_error(self):
        conn = sqlite3.connect("ecommerce.db")
        order_id, phone = conn.execute("select order_id,phone_last4 from orders limit 1").fetchone()
        result = run_order_read_slice(source_db="ecommerce.db", order_id=order_id, phone_last4=phone)
        self.assertEqual(result["business_code"], "ORDER_FOUND")
        self.assertEqual(result["terminal_run"]["status"], "SUCCEEDED")
        self.assertEqual(result["terminal_task_status"], "SUCCEEDED")
        self.assertEqual(result["terminal_attempt"]["status"], "SUCCEEDED")
        self.assertEqual(result["result"].status.value, "SUCCEEDED")
        self.assertIn("RESULT_WRITTEN", result["trace_event_types"])
        self.assertNotIn("ERROR", result["trace_event_types"])
