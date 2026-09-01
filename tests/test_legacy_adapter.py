import sqlite3
import unittest
import os
import tempfile
from unittest.mock import patch

from agent.legacy_adapter import LegacyOrderAdapter
from agent import tools


class LegacyAdapterTests(unittest.TestCase):
    def test_disabled_mode_preserves_legacy_read_path_and_enabled_mode_uses_m1(self):
        conn = sqlite3.connect("ecommerce.db")
        order_id, phone = conn.execute("select order_id,phone_last4 from orders limit 1").fetchone()
        legacy = LegacyOrderAdapter(source_db="ecommerce.db", enabled=False).query(order_id, phone)
        self.assertEqual(legacy["mode"], "legacy")
        m1 = LegacyOrderAdapter(source_db="ecommerce.db", enabled=True).query(order_id, phone)
        self.assertEqual(m1["mode"], "m1")
        self.assertEqual(m1["result"].business_code, "ORDER_FOUND")

    def test_get_order_info_feature_flag_integrates_adapter_and_default_disabled_path(self):
        conn = sqlite3.connect("ecommerce.db")
        order_id, phone = conn.execute("select order_id,phone_last4 from orders limit 1").fetchone()
        previous_db = tools.DB_PATH
        try:
            tools.DB_PATH = "ecommerce.db"
            with tempfile.TemporaryDirectory() as temp:
                with patch.dict(os.environ, {"M1_ORDER_SLICE_ENABLED": "1"}):
                    with patch.object(tools.LegacyOrderAdapter, "query", wraps=tools.LegacyOrderAdapter(source_db="ecommerce.db", runtime_db=temp + "/runtime.db", enabled=True).query) as spy:
                        # Tools constructs the adapter itself, so the spy proves
                        # actual get_order_info dispatch reached the M1 seam.
                        response = tools.get_order_info(order_id, phone)
                        self.assertEqual(response["success"], True)
                        self.assertEqual(response["code"], "OK")
                        self.assertTrue(spy.called)
                with patch.dict(os.environ, {"M1_ORDER_SLICE_ENABLED": "0"}):
                    response = tools.get_order_info(order_id, phone)
                    self.assertEqual(response["success"], True)
        finally:
            tools.DB_PATH = previous_db
