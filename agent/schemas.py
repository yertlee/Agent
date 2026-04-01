from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ToolResponse(BaseModel):
    success: bool = Field(..., description="工具调用是否成功")
    code: str = Field(..., description="状态码或错误码")
    message: str = Field(..., description="系统级说明")
    data: Optional[Dict[str, Any]] = Field(default=None, description="业务数据")
    user_hint: str = Field(default="", description="给用户看的简短提示")


class OrderQueryInput(BaseModel):
    order_id: str = Field(..., description="订单号")
    phone_last4: str = Field(..., description="手机号后四位，4位数字")

    @field_validator("order_id")
    def validate_order_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.isdigit() or len(v) < 8:
            raise ValueError("order_id 必须是长度不少于 8 的纯数字字符串")
        return v

    @field_validator("phone_last4")
    def validate_phone_last4(cls, v: str) -> str:
        if not isinstance(v, str) or not v.isdigit() or len(v) != 4:
            raise ValueError("phone_last4 必须是 4 位数字字符串")
        return v


class AfterSalesCreateInput(BaseModel):
    action: Literal["create"] = Field(default="create", description="固定为 create")
    order_id: str = Field(..., description="订单号")
    phone_last4: str = Field(..., description="手机号后四位，4位数字")
    service_type: Literal["退款", "退货", "换货"] = Field(..., description="售后类型")
    reason: str = Field(..., description="售后原因")

    @field_validator("order_id")
    def validate_order_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.isdigit() or len(v) < 8:
            raise ValueError("order_id 必须是长度不少于 8 的纯数字字符串")
        return v

    @field_validator("phone_last4")
    def validate_phone_last4(cls, v: str) -> str:
        if not isinstance(v, str) or not v.isdigit() or len(v) != 4:
            raise ValueError("phone_last4 必须是 4 位数字字符串")
        return v

    @field_validator("reason")
    def validate_reason(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("reason 不能为空")
        return v.strip()


class AfterSalesQueryInput(BaseModel):
    action: Literal["query"] = Field(default="query", description="固定为 query")
    order_id: str = Field(..., description="订单号")
    phone_last4: str = Field(..., description="手机号后四位，4位数字")

    @field_validator("order_id")
    def validate_order_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.isdigit() or len(v) < 8:
            raise ValueError("order_id 必须是长度不少于 8 的纯数字字符串")
        return v

    @field_validator("phone_last4")
    def validate_phone_last4(cls, v: str) -> str:
        if not isinstance(v, str) or not v.isdigit() or len(v) != 4:
            raise ValueError("phone_last4 必须是 4 位数字字符串")
        return v


class LogisticsQueryInput(BaseModel):
    carrier_code: str = Field(..., description="快递公司编码")
    tracking_no: str = Field(..., description="快递单号")
    phone_last4: Optional[str] = Field(default=None, description="手机号后四位，可选")

    @field_validator("carrier_code")
    def validate_carrier_code(cls, v: str) -> str:
        value = (v or "").strip().lower()
        if not value:
            raise ValueError("carrier_code 不能为空")
        return value

    @field_validator("tracking_no")
    def validate_tracking_no(cls, v: str) -> str:
        value = (v or "").strip()
        if len(value) < 6 or len(value) > 32:
            raise ValueError("tracking_no 长度必须在 6 到 32 之间")
        return value

    @field_validator("phone_last4")
    def validate_phone_last4_optional(cls, v: Optional[str]) -> Optional[str]:
        if v in (None, ""):
            return None
        if not isinstance(v, str) or not v.isdigit() or len(v) != 4:
            raise ValueError("phone_last4 必须是 4 位数字字符串")
        return v


class HandoffInput(BaseModel):
    summary: str = Field(..., description="当前对话摘要")
    reason: str = Field(..., description="转人工原因")

    @field_validator("summary")
    def validate_summary(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("summary 不能为空")
        return v.strip()

    @field_validator("reason")
    def validate_reason(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("reason 不能为空")
        return v.strip()
