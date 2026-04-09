from __future__ import annotations

import uuid
from datetime import datetime
from langchain_core.tools import tool


@tool(description="创建售后工单，入参为问题摘要、症状列表和联系电话，返回工单编号与受理时间")
def create_after_sales_ticket(summary: str, symptoms: str, phone: str) -> str:
    ticket_id = f"AS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"工单已创建。工单号：{ticket_id}；受理时间：{created_at}；"
        f"问题摘要：{summary}；症状：{symptoms}；回访电话：{phone}。"
    )
