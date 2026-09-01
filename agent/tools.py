import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import ValidationError

from .logistics_runtime import query_logistics_snapshot_simulated
from .storage.repository import SQLiteOrderRepository
from .legacy_adapter import LegacyOrderAdapter
from .m2_legacy_adapter import M2OrderAdapter
from .schemas import (
    ToolResponse,
    OrderQueryInput,
    AfterSalesCreateInput,
    AfterSalesQueryInput,
    LogisticsQueryInput,
    HandoffInput,
)


# NOTE: public repo 不包含 sqlite 数据库文件。
# 运行时请通过环境变量 `ECOMMERCE_DB_PATH` 指向本地 ecommerce.db。
DB_PATH = os.getenv("ECOMMERCE_DB_PATH")

ALLOWED_SERVICE_TYPES = {"退款", "退货", "换货"}
ACTIVE_AFTERSALES_STATUSES = {"待审核", "审核通过", "退款处理中", "退货待寄回"}


def _model_to_dict(model_obj: Any) -> Dict[str, Any]:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


def _response(
    success: bool,
    code: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    user_hint: str = "",
) -> Dict[str, Any]:
    try:
        resp = ToolResponse(
            success=success,
            code=code,
            message=message,
            data=data,
            user_hint=user_hint,
        )
        return _model_to_dict(resp)
    except ValidationError:
        return {
            "success": bool(success),
            "code": str(code),
            "message": str(message),
            "data": data if isinstance(data, dict) or data is None else None,
            "user_hint": str(user_hint),
        }


def _validation_error_response(e: ValidationError) -> Dict[str, Any]:
    try:
        first_error = e.errors()[0]
        field_name = " -> ".join(str(x) for x in first_error.get("loc", []))
        error_msg = first_error.get("msg", "输入校验失败")
        detail = f"{field_name}: {error_msg}" if field_name else error_msg
    except Exception:
        detail = "输入校验失败"

    return _response(
        False,
        "INVALID_PARAMS",
        f"参数校验失败：{detail}",
        None,
        "输入参数不符合要求，请检查后重试。",
    )


def _get_conn() -> sqlite3.Connection:
    if not DB_PATH:
        raise FileNotFoundError(
            "缺少数据库配置：请设置 ECOMMERCE_DB_PATH（public 仓库未提供 sqlite 数据库文件）。"
        )
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库文件不存在: {os.path.abspath(DB_PATH)}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _generate_ticket_id() -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    rand_part = uuid.uuid4().hex[:6].upper()
    return f"AS{date_part}{rand_part}"


def _generate_handoff_id() -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    rand_part = uuid.uuid4().hex[:6].upper()
    return f"HF{date_part}{rand_part}"


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]) for row in cur.fetchall()}


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except Exception:
        return default


def _fetch_order(conn: sqlite3.Connection, order_id: str) -> Optional[sqlite3.Row]:
    columns = _table_columns(conn, "orders")
    select_fields = [
        "order_id",
        "phone_last4",
        "product_name",
        "amount",
        "order_status",
        "pay_status",
        "created_at",
        "can_apply_aftersales",
    ]
    for optional_field in ("carrier_code", "tracking_no"):
        if optional_field in columns:
            select_fields.append(optional_field)

    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT {", ".join(select_fields)}
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,),
    )
    return cur.fetchone()


def _fetch_latest_ticket_by_order(conn: sqlite3.Connection, order_id: str) -> Optional[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticket_id, order_id, phone_last4, service_type, reason, ticket_status, created_at, updated_at
        FROM aftersales_tickets
        WHERE order_id = ?
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 1
        """,
        (order_id,),
    )
    return cur.fetchone()


def _has_active_aftersales(conn: sqlite3.Connection, order_id: str) -> bool:
    statuses = sorted(ACTIVE_AFTERSALES_STATUSES)
    placeholders = ",".join(["?"] * len(statuses))
    sql = f"""
        SELECT 1
        FROM aftersales_tickets
        WHERE order_id = ?
          AND ticket_status IN ({placeholders})
        LIMIT 1
    """
    params = [order_id, *statuses]

    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchone() is not None


def get_order_info(order_id: str, phone_last4: str) -> Dict[str, Any]:
    try:
        payload = OrderQueryInput(order_id=order_id, phone_last4=phone_last4)
    except ValidationError as e:
        return _validation_error_response(e)

    # M2 feature-flag seam: default remains legacy; enabled calls traverse the
    # canonical Registry, trusted context and Executor and project back.
    if os.getenv("M2_EXECUTION_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            outcome = M2OrderAdapter(source_db=DB_PATH or "").query(payload.order_id, payload.phone_last4)
            return _response(bool(outcome.get("success")), "OK" if outcome.get("success") else str(outcome.get("code") or "TOOL_EXECUTION_FAILED"), str(outcome.get("message") or ""), outcome.get("data"), str(outcome.get("user_hint") or ""))
        except (FileNotFoundError, sqlite3.Error):
            return _response(False, "CONFIG_MISSING", "系统暂时无法访问订单数据", None, "系统暂时无法访问订单数据，请稍后再试。")

    # M1 feature-flag seam: default remains the legacy M0 repository path.  When
    # explicitly enabled, the actual tool call runs through the durable M1
    # object chain and is projected back to the legacy ToolResponse shape.
    if os.getenv("M1_ORDER_SLICE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            outcome = LegacyOrderAdapter(source_db=DB_PATH or "", enabled=True).query(payload.order_id, payload.phone_last4)
            result = outcome["result"]
            if result.status.value == "SUCCEEDED":
                return _response(True, "OK", "订单查询成功", dict(result.payload or {}), "已查询到订单信息。")
            return _response(False, "ORDER_NOT_FOUND", "未找到对应订单", None, "未查询到该订单，请确认订单号是否正确。")
        except (FileNotFoundError, sqlite3.Error):
            return _response(False, "DB_NOT_FOUND", "系统暂时无法访问订单数据", None, "系统暂时无法访问订单数据，请稍后再试。")

    try:
        # M0 read slice: repository applies ownership in SQL.  The unfiltered
        # metadata read only distinguishes an unknown order from a wrong
        # ownership proof so legacy response semantics remain stable.
        with SQLiteOrderRepository(DB_PATH or "") as repository:
            order_data = repository.get_order_for_owner(payload.order_id, payload.phone_last4)
            if order_data is None:
                exists = repository.get_order_by_id(payload.order_id)
                if exists is not None:
                    return _response(
                        False,
                        "PHONE_MISMATCH",
                        "手机号后四位校验失败",
                        None,
                        "您提供的手机号后四位与订单信息不一致，请重新确认。",
                    )
                return _response(
                    False,
                    "ORDER_NOT_FOUND",
                    "未找到对应订单",
                    None,
                    "未查询到该订单，请确认订单号是否正确。",
                )
    except FileNotFoundError as e:
        return _response(
            False,
            "DB_NOT_FOUND",
            str(e),
            None,
            "系统暂时无法访问订单数据，请稍后再试。",
        )
    except sqlite3.Error as e:
        return _response(
            False,
            "DB_ERROR",
            f"数据库查询失败: {e}",
            None,
            "系统繁忙，请稍后重试。",
        )

    data = {
        "order_id": order_data["order_id"],
        "product_name": order_data["product_name"],
        "amount": float(order_data["amount"]),
        "order_status": order_data["order_status"],
        "pay_status": order_data["pay_status"],
        "created_at": order_data["created_at"],
        "can_apply_aftersales": int(order_data["can_apply_aftersales"]),
        "carrier_code": str(order_data.get("carrier_code", "") or ""),
        "tracking_no": str(order_data.get("tracking_no", "") or ""),
        "phone_last4": order_data["phone_last4"],
        "source": "sql",
    }

    return _response(
        True,
        "OK",
        "订单查询成功",
        data,
        "已查询到订单信息。",
    )


def aftersales_service(
    action: str,
    order_id: str,
    phone_last4: str,
    service_type: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    if action == "create":
        try:
            payload = AfterSalesCreateInput(
                action=action,
                order_id=order_id,
                phone_last4=phone_last4,
                service_type=service_type,
                reason=reason,
            )
        except ValidationError as e:
            return _validation_error_response(e)

    elif action == "query":
        try:
            payload = AfterSalesQueryInput(
                action=action,
                order_id=order_id,
                phone_last4=phone_last4,
            )
        except ValidationError as e:
            return _validation_error_response(e)
    else:
        return _response(
            False,
            "INVALID_PARAMS",
            "action 必须是 create 或 query",
            None,
            "操作类型不正确，请稍后重试。",
        )

    try:
        conn = _get_conn()
        try:
            order = _fetch_order(conn, payload.order_id)

            if order is None:
                return _response(
                    False,
                    "ORDER_NOT_FOUND",
                    "未找到对应订单",
                    None,
                    "未查询到该订单，请确认订单号是否正确。",
                )

            if order["phone_last4"] != payload.phone_last4:
                return _response(
                    False,
                    "PHONE_MISMATCH",
                    "手机号后四位校验失败",
                    None,
                    "您提供的手机号后四位与订单信息不一致，请重新确认。",
                )

            if action == "query":
                latest_ticket = _fetch_latest_ticket_by_order(conn, payload.order_id)
                if latest_ticket is None:
                    return _response(
                        False,
                        "AFTERSALES_NOT_FOUND",
                        "未找到该订单对应的售后工单",
                        None,
                        "当前订单暂无售后记录。",
                    )

                data = {
                    "ticket_id": latest_ticket["ticket_id"],
                    "order_id": latest_ticket["order_id"],
                    "service_type": latest_ticket["service_type"],
                    "reason": latest_ticket["reason"],
                    "ticket_status": latest_ticket["ticket_status"],
                    "created_at": latest_ticket["created_at"],
                    "updated_at": latest_ticket["updated_at"],
                }
                return _response(
                    True,
                    "OK",
                    "售后进度查询成功",
                    data,
                    "已查询到售后进度。",
                )

            # create 分支
            if int(order["can_apply_aftersales"]) != 1:
                return _response(
                    False,
                    "AFTERSALES_NOT_ALLOWED",
                    "当前订单不满足售后条件",
                    None,
                    "当前订单暂不支持申请售后。",
                )

            if _has_active_aftersales(conn, payload.order_id):
                return _response(
                    False,
                    "AFTERSALES_ALREADY_EXISTS",
                    "该订单已有进行中的售后工单",
                    None,
                    "该订单已有进行中的售后申请，请先查看当前进度。",
                )

            ticket_id = _generate_ticket_id()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            initial_status = "待审核"

            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO aftersales_tickets
                (ticket_id, order_id, phone_last4, service_type, reason, ticket_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    payload.order_id,
                    payload.phone_last4,
                    payload.service_type,
                    payload.reason,
                    initial_status,
                    now_str,
                    now_str,
                ),
            )
            conn.commit()

            data = {
                "ticket_id": ticket_id,
                "order_id": payload.order_id,
                "service_type": payload.service_type,
                "reason": payload.reason,
                "ticket_status": initial_status,
                "created_at": now_str,
            }

            return _response(
                True,
                "OK",
                "售后申请创建成功",
                data,
                "您的售后申请已提交，当前状态为待审核。",
            )

        finally:
            conn.close()

    except FileNotFoundError as e:
        return _response(
            False,
            "DB_NOT_FOUND",
            str(e),
            None,
            "系统暂时无法访问售后数据，请稍后再试。",
        )
    except sqlite3.Error as e:
        return _response(
            False,
            "DB_ERROR",
            f"数据库操作失败: {e}",
            None,
            "系统繁忙，请稍后重试。",
        )


def query_logistics_snapshot(carrier_code: str, tracking_no: str, phone_last4: Optional[str] = None) -> Dict[str, Any]:
    try:
        payload = LogisticsQueryInput(
            carrier_code=carrier_code,
            tracking_no=tracking_no,
            phone_last4=phone_last4,
        )
    except ValidationError as e:
        return _validation_error_response(e)

    return query_logistics_snapshot_simulated(
        carrier_code=payload.carrier_code,
        tracking_no=payload.tracking_no,
        phone_last4=payload.phone_last4,
    )


def handoff_to_human(summary: str, reason: str) -> Dict[str, Any]:
    try:
        payload = HandoffInput(summary=summary, reason=reason)
    except ValidationError as e:
        return _validation_error_response(e)

    if payload.reason in {"用户投诉", "情绪激烈"}:
        priority = "high"
    elif payload.reason in {"身份无法确认", "系统无法处理"}:
        priority = "medium"
    else:
        priority = "low"

    handoff_id = _generate_handoff_id()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "handoff_id": handoff_id,
        "reason": payload.reason,
        "priority": priority,
        "status": "waiting_human",
        "summary": payload.summary,
        "created_at": now_str,
    }

    return _response(
        True,
        "HANDOFF_CREATED",
        "已创建转人工请求",
        data,
        "已为您转接人工客服，请稍候。",
    )


if __name__ == "__main__":
    print("当前数据库路径:", os.path.abspath(DB_PATH) if DB_PATH else "(未配置 ECOMMERCE_DB_PATH)")

    print("\n=== get_order_info 测试 ===")
    print(get_order_info("20260226003", "8820"))
    print(get_order_info("20260226003", "0000"))

    print("\n=== aftersales_service 查询测试 ===")
    print(aftersales_service("query", "20260226004", "1027"))

    print("\n=== aftersales_service 创建测试 ===")
    print(aftersales_service("create", "20260226003", "8820", service_type="退款", reason="不想要了"))

    print("\n=== handoff_to_human 测试 ===")
    print(handoff_to_human("用户多次催促且情绪激动，要求立即人工介入。", "情绪激烈"))
