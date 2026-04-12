import json
import os
import re
import uuid
from contextvars import ContextVar, Token
from datetime import datetime

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from rag.rag_service import RagSummarizeService
from tools.mcp_tools import list_mcp_tools
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

import requests
from typing import Any

rag = RagSummarizeService()
tavily = TavilySearch(max_results=5, topic="general")

DEFAULT_USER_ID = os.getenv("TS_DEMO_USER_ID", "1001")
_TOOL_RUNTIME_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "tool_runtime_context", default={}
)
external_data: dict[str, dict[str, dict[str, str]]] = {}


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


@tool(parse_docstring=True)
def web_search(query: str):
    """从互联网检索与商品、价格相关的信息。

    Args:
        query (str): 搜索关键词或完整查询语句。

    Returns:
        dict | list[dict]: Tavily 原始检索结果，供后续推理与比对使用。

    """
    return tavily.invoke(query)


@tool(parse_docstring=True)
def rag_summarize(query: str) -> str:
    """从向量存储中检索并总结参考资料，包括选购、维护、保养、排障等各种问题和知识。

    Args:
        query (str): 用于检索与总结的用户查询内容。

    Returns:
        str: 检索总结文本；若失败则返回可直接展示给用户的错误提示。

    """
    try:
        return rag.rag_summarize(query)
        # return rag_state_graph_agent.invoke(query)
    except requests.exceptions.SSLError as e:
        logger.error(
            f"[rag_summarize]调用DashScope时发生SSL异常: {str(e)}", exc_info=True
        )
        return "当前与模型服务的安全连接失败（SSL）。请检查网络代理/VPN、系统证书或稍后重试。"
    except requests.exceptions.RequestException as e:
        logger.error(
            f"[rag_summarize]调用DashScope时发生网络异常: {str(e)}", exc_info=True
        )
        return "当前无法连接模型服务（网络异常）。请检查网络环境后重试。"
    except Exception as e:
        logger.error(f"[rag_summarize]执行失败: {str(e)}", exc_info=True)
        return "检索总结暂时不可用，请稍后重试。"


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
    clean_summary = str(summary or "").strip()
    clean_symptoms = str(symptoms or "").strip()
    clean_phone = re.sub(r"\s+", "", str(phone or ""))

    if not clean_summary or not clean_symptoms or not clean_phone:
        return "工单创建失败：summary、symptoms、phone 不能为空。"

    if not re.fullmatch(r"\+?\d{7,20}", clean_phone):
        return "工单创建失败：phone 格式不合法，请提供 7-20 位数字（可含+号）。"

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

    ticket_store_path = get_abs_path(
        os.getenv("TS_AFTER_SALES_TICKET_PATH", "data/db/after_sales_tickets.jsonl")
    )
    try:
        os.makedirs(os.path.dirname(ticket_store_path), exist_ok=True)
        with open(ticket_store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ticket_record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(f"[create_after_sales_ticket]工单落盘失败: {e}", exc_info=True)
        return "工单创建失败：写入工单存储时发生异常，请稍后重试。"

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
    clean_model = str(product_model or "").strip()
    clean_consignee = str(consignee or "").strip()
    clean_phone = re.sub(r"\s+", "", str(phone or ""))
    clean_address = str(address or "").strip()

    if not clean_model or not clean_consignee or not clean_phone or not clean_address:
        return "订单创建失败：product_model、consignee、phone、address 不能为空。"
    if not isinstance(quantity, int) or quantity <= 0:
        return "订单创建失败：quantity 必须为大于0的整数。"
    if not re.fullmatch(r"\+?\d{7,20}", clean_phone):
        return "订单创建失败：phone 格式不合法，请提供 7-20 位数字（可含+号）。"

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

    path = get_abs_path(
        os.getenv("TS_PURCHASE_ORDER_PATH", "data/db/purchase_orders.jsonl")
    )
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(f"[create_purchase_order]订单落盘失败: {e}", exc_info=True)
        return "订单创建失败：写入订单存储时发生异常，请稍后重试。"

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
    clean_reason = str(reason or "").strip()
    clean_model = str(product_model or "").strip()
    clean_phone = re.sub(r"\s+", "", str(phone or ""))
    clean_address = str(address or "").strip()

    if not clean_reason or not clean_model or not clean_phone or not clean_address:
        return "退货申请创建失败：reason、product_model、phone、address 不能为空。"
    if not re.fullmatch(r"\+?\d{7,20}", clean_phone):
        return "退货申请创建失败：phone 格式不合法，请提供 7-20 位数字（可含+号）。"

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

    path = get_abs_path(
        os.getenv("TS_AFTER_SALES_RETURN_PATH", "data/db/after_sales_returns.jsonl")
    )
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.error(
            f"[create_manual_return_request]退货申请落盘失败: {e}", exc_info=True
        )
        return "退货申请创建失败：写入退货申请存储时发生异常，请稍后重试。"

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

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")
                if len(arr) < 6:
                    continue

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

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
    user_id = str(context.get("user_id", "")).strip() or DEFAULT_USER_ID
    return f"fill_context_for_report已调用，报告上下文已激活（user_id={user_id}）"
