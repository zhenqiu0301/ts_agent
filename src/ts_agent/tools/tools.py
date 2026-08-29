import csv
import hashlib
import json
import os
import re
import threading
import uuid
from contextvars import ContextVar, Token
from datetime import datetime
from functools import lru_cache
from typing import Any

import httpx
import openai
import requests
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from ts_agent.rag.rag_service import RagSummarizeService
from ts_agent.rag.vector_store import VectorStoreService
from ts_agent.utils.config_handler import agent_conf
from ts_agent.utils.logger_handler import logger
from ts_agent.utils.path_tool import get_abs_path


@lru_cache(maxsize=1)
def _get_shared_vector_store() -> VectorStoreService:
    """进程内共享向量库：Chroma 与 DashScope embeddings 均为同步实现，跨事件循环安全。"""
    return VectorStoreService()


def get_rag_service() -> RagSummarizeService:
    # ChatOpenAI 的异步 httpx client 绑定创建时的事件循环，须按调用新建并在用后关闭，
    # 不能整体缓存服务实例。
    return RagSummarizeService(vector_store=_get_shared_vector_store())


def get_tavily_search() -> TavilySearch:
    return TavilySearch(max_results=5, topic="general")

DEFAULT_USER_ID = os.getenv("TS_DEMO_USER_ID", "1001")
_TOOL_RUNTIME_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "tool_runtime_context", default=None
)
external_data: dict[str, dict[str, dict[str, str]]] = {}
_BUSINESS_WRITE_LOCK = threading.Lock()


def set_tool_runtime_context(context: dict[str, Any]) -> Token:
    return _TOOL_RUNTIME_CONTEXT.set(context or {})


def reset_tool_runtime_context(token: Token) -> None:
    _TOOL_RUNTIME_CONTEXT.reset(token)


def _runtime_context() -> dict[str, Any]:
    context = _TOOL_RUNTIME_CONTEXT.get()
    return context if isinstance(context, dict) else {}


def _normalize_month(month: str) -> str:
    text = str(month or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return text
    return datetime.now().strftime("%Y-%m")


def _is_report_context_enabled() -> bool:
    return bool(_runtime_context().get("report", False))


def _validate_text(value: str, field: str, *, max_length: int = 500) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field} 不能为空")
    if len(cleaned) > max_length:
        raise ValueError(f"{field} 长度不能超过 {max_length} 个字符")
    return cleaned


def _validate_phone(value: str) -> str:
    cleaned = re.sub(r"[\s-]+", "", str(value or ""))
    if not re.fullmatch(r"\+?\d{7,20}", cleaned):
        raise ValueError("phone 格式不合法，请提供 7-20 位数字（可含+号）")
    return cleaned


def _idempotency_key(action: str, payload: dict[str, Any]) -> str:
    context = _runtime_context()
    identity = {
        "action": action,
        "user_id": str(context.get("user_id", "")).strip() or DEFAULT_USER_ID,
        "thread_id": str(context.get("thread_id", "")).strip(),
        "payload": payload,
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _append_business_record(path: str, record: dict[str, Any]) -> dict[str, Any] | None:
    """幂等写入业务记录；重试时返回已有记录而不重复创建。"""

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _BUSINESS_WRITE_LOCK:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as file:
                for line in file:
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if existing.get("idempotency_key") == record["idempotency_key"]:
                        return existing
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return None


@tool(parse_docstring=True)
async def web_search(query: str):
    """从互联网检索与商品、价格相关的信息。

    Args:
        query (str): 搜索关键词或完整查询语句。

    Returns:
        dict | list[dict]: Tavily 原始检索结果，供后续推理与比对使用。

    """
    return await get_tavily_search().ainvoke(query)


@tool(parse_docstring=True)
async def rag_summarize(query: str) -> str:
    """从向量存储中检索并总结参考资料，包括选购、维护、保养、排障等各种问题和知识。

    Args:
        query (str): 用于检索与总结的用户查询内容。

    Returns:
        str: 检索总结文本；若失败则返回可直接展示给用户的错误提示。

    """
    service = get_rag_service()
    try:
        return await service.rag_summarize(query)
    except (
        httpx.HTTPError,
        openai.APIError,
        requests.exceptions.RequestException,
    ) as e:
        # ChatOpenAI 走 openai/httpx 异常族，DashScope embeddings 走 requests 异常族
        logger.error(
            f"[rag_summarize]调用模型服务时发生网络异常: {str(e)}", exc_info=True
        )
        return "当前无法连接模型服务（网络异常）。请检查网络环境后重试。"
    except Exception as e:
        logger.error(f"[rag_summarize]执行失败: {str(e)}", exc_info=True)
        return "检索总结暂时不可用，请稍后重试。"
    finally:
        await service.close()


@tool(parse_docstring=True)
def get_user_context() -> dict[str, str]:
    """获取当前会话用户上下文（用户ID与当前月份）。

    Args:
        None.

    Returns:
        dict[str, str]: 包含 user_id 与 month 的上下文字段。

    """
    context = _runtime_context()
    user_id = str(context.get("user_id", "")).strip() or DEFAULT_USER_ID
    month = datetime.now().strftime("%Y-%m")
    return {"user_id": user_id, "month": month}


@tool(parse_docstring=True)
def create_after_sales_ticket(summary: str, symptoms: str, phone: str) -> str:
    """创建售后服务工单。

    Args:
        summary (str): 用户问题的简要摘要。
        symptoms (str): 症状或现象的简要描述。
        phone (str): 回访联系电话。

    Returns:
        str: 包含工单号、受理时间及关键信息的确认文本。

    """
    try:
        clean_summary = _validate_text(summary, "summary", max_length=200)
        clean_symptoms = _validate_text(symptoms, "symptoms", max_length=1000)
        clean_phone = _validate_phone(phone)
    except ValueError as exc:
        return f"工单创建失败：{exc}。"

    now = datetime.now()
    ticket_id = f"AS-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")

    context = _runtime_context()
    ticket_record = {
        "ticket_id": ticket_id,
        "created_at": created_at,
        "user_id": str(context.get("user_id", "")).strip() or DEFAULT_USER_ID,
        "thread_id": str(context.get("thread_id", "")).strip(),
        "route": str(context.get("route", "")).strip(),
        "summary": clean_summary,
        "symptoms": clean_symptoms,
        "phone": clean_phone,
        "status": "created",
    }
    ticket_record["idempotency_key"] = _idempotency_key(
        "after_sales_ticket",
        {"summary": clean_summary, "symptoms": clean_symptoms, "phone": clean_phone},
    )

    ticket_store_path = get_abs_path(
        os.getenv("TS_AFTER_SALES_TICKET_PATH", "data/db/after_sales_tickets.jsonl")
    )
    try:
        existing = _append_business_record(ticket_store_path, ticket_record)
    except OSError as e:
        logger.error(f"[create_after_sales_ticket]工单落盘失败: {e}", exc_info=True)
        return "工单创建失败：写入工单存储时发生异常，请稍后重试。"

    if existing:
        return f"重复请求已识别，未重复创建。已有工单号：{existing['ticket_id']}。"
    return (
        f"工单已创建。工单号：{ticket_id}；受理时间：{created_at}；"
        f"问题摘要：{clean_summary}；症状：{clean_symptoms}；回访电话：{clean_phone}。"
    )


@tool(parse_docstring=True)
def create_purchase_order(
    product_model: str,
    quantity: int,
    consignee: str,
    phone: str,
    address: str,
) -> str:
    """创建人工订单（确认购买后调用）。

    Args:
        product_model (str): 下单商品型号或名称。
        quantity (int): 购买数量，必须大于0。
        consignee (str): 收货人姓名。
        phone (str): 联系手机号。
        address (str): 收货地址。

    Returns:
        str: 订单创建结果与关键信息。

    """
    try:
        clean_model = _validate_text(product_model, "product_model", max_length=100)
        clean_consignee = _validate_text(consignee, "consignee", max_length=100)
        clean_phone = _validate_phone(phone)
        clean_address = _validate_text(address, "address", max_length=500)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or not 1 <= quantity <= 20:
            raise ValueError("quantity 必须是 1-20 的整数")
    except ValueError as exc:
        return f"订单创建失败：{exc}。"

    now = datetime.now()
    order_id = f"PO-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")
    context = _runtime_context()
    record = {
        "order_id": order_id,
        "created_at": created_at,
        "user_id": str(context.get("user_id", "")).strip() or DEFAULT_USER_ID,
        "thread_id": str(context.get("thread_id", "")).strip(),
        "route": str(context.get("route", "")).strip(),
        "product_model": clean_model,
        "quantity": quantity,
        "consignee": clean_consignee,
        "phone": clean_phone,
        "address": clean_address,
        "status": "created",
    }
    record["idempotency_key"] = _idempotency_key(
        "purchase_order",
        {
            "product_model": clean_model,
            "quantity": quantity,
            "consignee": clean_consignee,
            "phone": clean_phone,
            "address": clean_address,
        },
    )

    path = get_abs_path(
        os.getenv("TS_PURCHASE_ORDER_PATH", "data/db/purchase_orders.jsonl")
    )
    try:
        existing = _append_business_record(path, record)
    except OSError as e:
        logger.error(f"[create_purchase_order]订单落盘失败: {e}", exc_info=True)
        return "订单创建失败：写入订单存储时发生异常，请稍后重试。"

    if existing:
        return f"重复请求已识别，未重复下单。已有订单号：{existing['order_id']}。"
    return (
        f"订单已创建。订单号：{order_id}；下单时间：{created_at}；"
        f"型号：{clean_model}；数量：{quantity}；收货人：{clean_consignee}；"
        f"联系电话：{clean_phone}；收货地址：{clean_address}。"
    )


@tool(parse_docstring=True)
def create_manual_return_request(
    reason: str,
    product_model: str,
    phone: str,
    address: str,
) -> str:
    """创建人工退货申请单（售后人工处理）。

    Args:
        reason (str): 退货原因摘要。
        product_model (str): 产品型号或名称。
        phone (str): 联系手机号。
        address (str): 取件地址。

    Returns:
        str: 退货申请单创建结果与关键信息。

    """
    try:
        clean_reason = _validate_text(reason, "reason", max_length=500)
        clean_model = _validate_text(product_model, "product_model", max_length=100)
        clean_phone = _validate_phone(phone)
        clean_address = _validate_text(address, "address", max_length=500)
    except ValueError as exc:
        return f"退货申请创建失败：{exc}。"

    now = datetime.now()
    request_id = f"RT-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")
    context = _runtime_context()
    record = {
        "request_id": request_id,
        "created_at": created_at,
        "user_id": str(context.get("user_id", "")).strip() or DEFAULT_USER_ID,
        "thread_id": str(context.get("thread_id", "")).strip(),
        "route": str(context.get("route", "")).strip(),
        "reason": clean_reason,
        "product_model": clean_model,
        "phone": clean_phone,
        "address": clean_address,
        "status": "pending_manual_review",
    }
    record["idempotency_key"] = _idempotency_key(
        "return_request",
        {
            "reason": clean_reason,
            "product_model": clean_model,
            "phone": clean_phone,
            "address": clean_address,
        },
    )

    path = get_abs_path(
        os.getenv("TS_AFTER_SALES_RETURN_PATH", "data/db/after_sales_returns.jsonl")
    )
    try:
        existing = _append_business_record(path, record)
    except OSError as e:
        logger.error(
            f"[create_manual_return_request]退货申请落盘失败: {e}", exc_info=True
        )
        return "退货申请创建失败：写入退货申请存储时发生异常，请稍后重试。"

    if existing:
        return f"重复请求已识别，未重复申请。已有申请单号：{existing['request_id']}。"
    return (
        f"人工退货申请已创建。申请单号：{request_id}；受理时间：{created_at}；"
        f"型号：{clean_model}；原因：{clean_reason}；联系电话：{clean_phone}；"
        f"取件地址：{clean_address}。"
    )


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                user_id = str(row.get("用户ID", "")).strip()
                feature = str(row.get("特征", "")).strip()
                efficiency = str(row.get("清洁效率", "")).strip()
                consumables = str(row.get("耗材", "")).strip()
                comparison = str(row.get("对比", "")).strip()
                time = str(row.get("时间", "")).strip()
                if not user_id or not time:
                    continue

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(parse_docstring=True)
def fetch_external_data(user_id: str, month: str) -> str:
    """从外部数据存储中获取指定用户在指定月份的使用记录。

    Args:
        user_id (str): 目标用户 ID。
        month (str): 目标月份，格式为 YYYY-MM。

    Returns:
        str: 命中时返回该用户该月的使用记录；未命中时返回空字符串。

    """
    if not _is_report_context_enabled():
        logger.warning(
            "[fetch_external_data] 报告上下文未激活，拒绝查询。user_id=%s month=%s",
            user_id,
            month,
        )
        return "请先调用 fill_context_for_report 后再查询使用报告数据。"

    try:
        generate_external_data()
    except Exception as e:
        logger.error(f"[fetch_external_data]加载外部数据失败: {e}", exc_info=True)
        return ""
    normalized_user_id = str(user_id or "").strip()
    normalized_month = _normalize_month(month)

    try:
        data = external_data[normalized_user_id][normalized_month]
        return (
            f"用户ID: {normalized_user_id}\n"
            f"月份: {normalized_month}\n"
            f"特征: {data.get('特征', '')}\n"
            f"清洁效率: {data.get('效率', '')}\n"
            f"耗材状态: {data.get('耗材', '')}\n"
            f"同类对比: {data.get('对比', '')}"
        )
    except KeyError:
        logger.warning(
            f"[fetch_external_data]未能检索到用户：{normalized_user_id}在{normalized_month}的使用记录数据"
        )
        return ""


@tool(parse_docstring=True)
def fill_context_for_report() -> str:
    """标记当前运行上下文为报告生成场景。

    Args:
        None.

    Returns:
        str: 上下文标记调用成功的确认文本。

    """
    context = _runtime_context()
    context["report"] = True
    user_id = str(context.get("user_id", "")).strip() or DEFAULT_USER_ID
    return f"fill_context_for_report已调用，报告上下文已激活（user_id={user_id}）"


@tool(parse_docstring=True)
def get_usage_report_data(user_id: str, month: str) -> str:
    """按确定流程激活报告上下文并读取指定月份的使用数据。

    Args:
        user_id (str): 目标用户 ID。
        month (str): 目标月份，格式 YYYY-MM。

    Returns:
        str: 使用报告原始数据，未命中时返回明确提示。
    """

    fill_context_for_report.invoke({})
    result = fetch_external_data.invoke({"user_id": user_id, "month": month})
    return result or "未查询到该用户在指定月份的使用记录。"
