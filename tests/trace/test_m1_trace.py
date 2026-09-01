import unittest
from agent.domain.objects import Run, sha256_json
from agent.storage.repositories import M1Repository
from agent.trace.events import TraceEvent, build_event, sensitive_surface_scan
from agent.trace.outbox import OutboxWriter


class M1TraceTests(unittest.TestCase):
    def setUp(self):
        self.repo = M1Repository(); self.repo.create_session("s1", "u1"); self.repo.create_run(Run(run_id="r1", session_id="s1", initial_world_hash=sha256_json({"w": 1}))); self.outbox = OutboxWriter(self.repo)
    def tearDown(self): self.repo.close()

    def test_seq_hash_parent_and_outbox_delivery(self):
        one = build_event(run_id="r1", session_id="s1", event_type="RUN_CREATED", seq_no=1, payload={"run_id": "r1"})
        two = build_event(run_id="r1", session_id="s1", event_type="ERROR", seq_no=2, parent_event_id=one.trace_id, payload={"code": "ORDER_NOT_FOUND", "layer": "business"})
        self.outbox.append(one); self.outbox.append(two)
        self.assertEqual([r[0] for r in self.repo.conn.execute("SELECT seq_no FROM trace_outbox ORDER BY seq_no")], [1, 2])
        self.assertEqual(two.parent_event_id, one.trace_id); self.assertEqual(two.payload_hash, sha256_json(two.payload))
        self.outbox.mark_delivered(one.trace_id); self.assertTrue(self.repo.conn.execute("SELECT delivered_at FROM trace_outbox WHERE event_id=?", (one.trace_id,)).fetchone()[0])
        with self.assertRaises(Exception): self.repo.conn.execute("UPDATE trace_outbox SET envelope_json='tampered' WHERE event_id=?", (one.trace_id,))
        with self.assertRaises(Exception): self.repo.conn.execute("UPDATE trace_outbox SET delivered_at=NULL WHERE event_id=?", (one.trace_id,))
        with self.assertRaises(Exception): self.repo.conn.execute("DELETE FROM trace_outbox WHERE event_id=?", (one.trace_id,))

    def test_invalid_seq_event_and_sensitive_payload_rejected(self):
        with self.assertRaises(ValueError): self.outbox.append(build_event(run_id="r1", session_id="s1", event_type="RUN_CREATED", seq_no=2))
        with self.assertRaises(ValueError): self.outbox.append(build_event(run_id="r1", session_id="s1", event_type="RUN_CREATED", seq_no=1, payload={"token": "raw"}))
        self.assertTrue(sensitive_surface_scan({"nested": {"payment_id": "x"}}))

    def test_explicit_payload_hash_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            TraceEvent(trace_id="e1", run_id="r1", session_id="s1", event_type="RUN_CREATED", seq_no=1, occurred_at="2026-01-01T00:00:00Z", actor="runtime", payload={"x": 1}, payload_hash="bad")

    def test_parent_must_be_same_run_and_earlier(self):
        one = build_event(run_id="r1", session_id="s1", event_type="RUN_CREATED", seq_no=1)
        self.outbox.append(one)
        wrong = build_event(run_id="r1", session_id="s1", event_type="ERROR", seq_no=2, parent_event_id="unknown")
        with self.assertRaises(ValueError): self.outbox.append(wrong)
