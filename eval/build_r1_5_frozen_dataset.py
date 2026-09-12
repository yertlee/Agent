"""Build the frozen R1.5 input/gold files owned by the primary auditor.

This is a dataset authoring utility, never imported by the runtime evaluator.
The generated evaluator consumes literal stored texts and never reconstructs
utterances from labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "datasets" / "r1_5"

DEV_PREFIXES = ("", "麻烦帮我处理：")
VAL_PREFIXES = ("想确认一件事，", "客服你好，")

DEV = {
    "ORDER_QUERY": [
        "查一下订单 {order}，手机号后四位 {phone}", "我买的那单现在是什么状态，订单号 {order} 后四位 {phone}",
        "只看订单信息，不要查物流：{order}，{phone}", "订单 {order} 付款成功了吗？验证尾号 {phone}",
        "帮我核对这笔购买记录 {order} / {phone}", "订单详情能调出来吗，编号 {order}，手机尾号 {phone}",
        "I need the order status for {order}, phone ending {phone}", "页面提到物流可能过期，但我现在只想查订单 {order}，尾号 {phone}",
    ],
    "LOGISTICS_QUERY": [
        "包裹到哪里了，运单 {tracking}，承运商 {carrier}，手机尾号 {phone}", "帮我看看快递进度 {carrier} {tracking}，{phone}",
        "这票货签收了吗 tracking={tracking} carrier={carrier} 后四位 {phone}", "只查运输轨迹，不需要订单详情，{carrier} {tracking} {phone}",
        "快件现在在哪个站点，单号 {tracking}，{carrier}，验证 {phone}", "物流跟踪号 {tracking} 有新动态吗，{carrier}，尾号 {phone}",
        "where is my parcel, carrier={carrier}, tracking={tracking}, phone {phone}", "不要查订单内容，我只关心包裹 {tracking} 的进度，{carrier}，{phone}",
    ],
    "ORDER_AND_LOGISTICS": [
        "查订单 {order} 并继续查它的配送进度，尾号 {phone}", "这单买了什么以及包裹到哪了：{order}，{phone}",
        "先核对订单归属，再查对应快递，订单 {order} 手机尾号 {phone}", "订单和运输状态都告诉我，{order} / {phone}",
        "看看这笔订单是否发货以及当前配送节点 {order} {phone}", "需要订单状态和关联运单进展，编号 {order}，尾号 {phone}",
        "check order {order} and its shipment, phone ending {phone}", "我怀疑页面有冲突，请实际查询订单和对应配送数据 {order} {phone}",
    ],
    "AFTERSALES_CREATE": [
        "我要给订单 {order} 申请退款，尾号 {phone}", "商品有问题，帮我发起退货，订单 {order}，{phone}",
        "这单想换货，编号 {order} 手机尾号 {phone}", "申请售后：订单 {order}，原因是收到破损商品，{phone}",
        "我不想要了，请创建退款申请 {order} {phone}", "能给这笔订单开一个退货工单吗 {order} / {phone}",
        "start a refund for order {order}, phone {phone}", "先别查进度，我现在要提交售后申请 {order}，{phone}",
    ],
    "AFTERSALES_STATUS": [
        "我的售后单 {ticket} 处理到哪了", "查询退款申请进度，工单号 {ticket}", "退货审核通过了吗 {ticket}",
        "售后状态帮我看一下，编号 {ticket}", "之前提交的换货请求有结果吗 {ticket}", "查一下服务单 {ticket} 当前节点",
        "what is the status of aftersales ticket {ticket}", "不要新建申请，只查询已有售后 {ticket}",
    ],
    "POLICY_QA": [
        "七天无理由的条件是什么", "拆封后还能退吗", "退款通常多久到账", "平台退换货规则发我看看",
        "生鲜商品支持无理由退货吗", "超过十五天还能申请售后吗", "what is the return policy", "我只是咨询规定，不需要查询我的订单",
    ],
    "PRODUCT_QA": [
        "商品 {sku} 有哪些颜色", "{sku} 的尺寸参数是什么", "这个型号支持快充吗，SKU {sku}", "查一下 {sku} 的库存规格",
        "商品 {sku} 是否防水", "{sku} 适配什么设备", "tell me the specifications of SKU {sku}", "只介绍商品 {sku}，不要查询订单",
    ],
    "COMPLAINT": [
        "客服一直不处理，我要投诉", "配送员态度很差，请记录投诉", "你们多次推诿，我需要人工处理", "商品描述严重不符，我要反馈",
        "退款拖了很久没人解决", "我要对这次服务提出正式投诉", "I want to file a complaint about the service", "先记录我的不满，不要自动退款",
    ],
    "CHITCHAT": [
        "你好", "谢谢你的帮助", "你今天怎么样", "讲个笑话", "晚安", "你叫什么名字", "hello there", "今天新加坡天气不错吧",
    ],
    "UNKNOWN": [
        "帮我订一张明天的机票", "写一段 Python 排序代码", "推荐附近的牙科诊所", "我的银行卡为什么被冻结",
        "播放一首音乐", "帮我修改电脑密码", "预测一下明天的股票", "这个问题和电商客服无关，请不要调用业务工具",
    ],
    "MULTI_INTENT": [
        "查订单 {order}，再告诉我退货政策，尾号 {phone}", "看看订单 {order} 的状态，同时解释七天无理由，{phone}",
        "我想核对订单 {order}，还想了解拆封退货规则，尾号 {phone}", "先查购买记录 {order}，然后回答退款多久到账，{phone}",
        "订单 {order} 现在怎样，以及超过十五天能否售后，{phone}", "check order {order} and explain the return policy, phone {phone}",
        "只读查询订单 {order}，另外咨询退换货规定，{phone}", "这单 {order} 是否付款成功？顺便说下无理由退货条件，{phone}",
    ],
}

VALIDATION = {
    "ORDER_QUERY": ["这笔购买记录目前处于哪一步 {order}，验证 {phone}", "不要看快递，我只核实交易单 {order}，{phone}", "订单页是什么状态 {order} / {phone}", "could you retrieve purchase {order}, ending {phone}"],
    "LOGISTICS_QUERY": ["快件 {tracking} 走到哪一站了，{carrier}，尾号 {phone}", "我只想追踪包裹 {carrier}/{tracking}/{phone}", "运输单 {tracking} 有没有签收，承运方 {carrier}，{phone}", "track parcel {tracking} via {carrier}, verification {phone}"],
    "ORDER_AND_LOGISTICS": ["核验交易 {order} 后继续看关联包裹，尾号 {phone}", "购买状态和寄送状态一起查 {order} {phone}", "从订单 {order} 找出运单并追踪，{phone}", "retrieve order {order} then follow its delivery, {phone}"],
    "AFTERSALES_CREATE": ["为 {order} 建立换货请求，尾号 {phone}", "收到坏件了，请提交退货 {order} {phone}", "我决定取消购买并申请退款 {order}/{phone}", "open an aftersales request for {order}, {phone}"],
    "AFTERSALES_STATUS": ["服务请求 {ticket} 有进展吗", "我之前的退款工单 {ticket} 到哪一步", "只看已有退货申请 {ticket}", "check service ticket {ticket}"],
    "POLICY_QA": ["哪些品类不能无理由退", "运费由谁承担", "价保规则是什么", "explain the refund rules"],
    "PRODUCT_QA": ["型号 {sku} 的重量是多少", "介绍一下 {sku} 的材质", "{sku} 能不能连接蓝牙", "show product details for {sku}"],
    "COMPLAINT": ["我要反馈商家服务问题", "这次配送体验太差了", "请转人工受理我的投诉", "I need to report poor service"],
    "CHITCHAT": ["早上好", "辛苦了", "陪我聊两句", "good afternoon"],
    "UNKNOWN": ["替我预约驾照考试", "分析这份医学报告", "打开我的邮箱", "convert dollars to yen"],
    "MULTI_INTENT": ["核对订单 {order}，并说明退款时效，尾号 {phone}", "查购买记录 {order}，另外问下运费退不退，{phone}", "retrieve {order} and explain exchange rules, {phone}", "订单 {order} 什么状态？商品拆封后还能退吗，{phone}"],
}


def values(index: int) -> dict[str, str]:
    return {
        "order": f"ORDR{index:06d}",
        "phone": f"{(3100 + index) % 10000:04d}",
        "tracking": f"TRKX{index:07d}",
        "carrier": ("sf", "jd", "yto", "zto")[index % 4],
        "sku": f"SKU-X{index:05d}",
        "ticket": f"AS-X{index:06d}",
    }


def expected_entities(intent: str, data: dict[str, str]) -> dict[str, str]:
    fields = {
        "ORDER_QUERY": ("order", "phone"), "LOGISTICS_QUERY": ("tracking", "carrier", "phone"),
        "ORDER_AND_LOGISTICS": ("order", "phone"), "AFTERSALES_CREATE": ("order", "phone"),
        "AFTERSALES_STATUS": ("ticket",), "PRODUCT_QA": ("sku",), "MULTI_INTENT": ("order", "phone"),
    }.get(intent, ())
    names = {"order": "order_id", "phone": "phone_last4", "tracking": "tracking_no", "carrier": "carrier_code", "sku": "product_sku", "ticket": "aftersales_ticket_id"}
    return {names[field]: data[field] for field in fields}


def build(split: str, pools: dict[str, list[str]], prefixes: tuple[str, str]) -> tuple[list[dict], list[dict]]:
    inputs, gold = [], []
    index = 1 if split == "dev" else 1001
    for intent, phrases in pools.items():
        for family_index, phrase in enumerate(phrases, 1):
            data = values(index)
            for variant, prefix in enumerate(prefixes, 1):
                case_id = f"r15-{split}-{intent.lower()}-{family_index:02d}-v{variant}"
                text = prefix + phrase.format(**data)
                inputs.append({"case_id": case_id, "split": split, "family_id": f"{split}-{intent}-{family_index:02d}", "text": text})
                gold.append({"case_id": case_id, "primary_intent": intent, "secondary_intents": ["ORDER_QUERY", "POLICY_QA"] if intent == "MULTI_INTENT" else [], "entities": expected_entities(intent, data)})
            index += 1
    return inputs, gold


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    all_meta = {}
    for split, pools, prefixes in (("dev", DEV, DEV_PREFIXES), ("validation", VALIDATION, VAL_PREFIXES)):
        inputs, gold = build(split, pools, prefixes)
        input_path = ROOT / f"{split}-inputs.jsonl"
        gold_path = ROOT / f"{split}-gold.jsonl"
        write_jsonl(input_path, inputs)
        write_jsonl(gold_path, gold)
        all_meta[split] = {
            "unique_case_N": len(inputs),
            "family_N": len({row["family_id"] for row in inputs}),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        }
    mutation_inputs = ROOT / "mutation-inputs.jsonl"
    mutation_gold = ROOT / "mutation-gold.jsonl"
    if mutation_inputs.is_file() and mutation_gold.is_file():
        mutation_rows = [json.loads(line) for line in mutation_inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
        all_meta["mutation"] = {
            "unique_case_N": len(mutation_rows),
            "family_N": len({row["family_id"] for row in mutation_rows}),
            "input_sha256": hashlib.sha256(mutation_inputs.read_bytes()).hexdigest(),
            "gold_sha256": hashlib.sha256(mutation_gold.read_bytes()).hexdigest(),
        }
    (ROOT / "dataset-manifest.json").write_text(json.dumps({"schema_version": "r1.5.dataset.v1", "splits": all_meta}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(all_meta, sort_keys=True))


if __name__ == "__main__":
    main()
