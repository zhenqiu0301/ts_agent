from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ts_agent.agents.sub_agents import after_sales_tools
from ts_agent.tools.mcp_tools import get_lazy_price_compare_tools, get_price_comparison_tool
from ts_agent.tools.tools import (
    create_after_sales_ticket,
    create_manual_return_request,
    create_purchase_order,
    external_data,
    fetch_external_data,
    fill_context_for_report,
    reset_tool_runtime_context,
    set_tool_runtime_context,
)


class ReportWorkflowTests(unittest.TestCase):
    def test_after_sales_agent_has_complete_report_toolchain(self) -> None:
        names = {tool.name for tool in after_sales_tools}
        self.assertTrue({"get_user_context", "get_usage_report_data"} <= names)
        self.assertNotIn("fetch_external_data", names)

    def test_report_data_requires_context_marker(self) -> None:
        external_data.clear()
        token = set_tool_runtime_context({"user_id": "1001", "report": False})
        try:
            blocked = fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})
            self.assertIn("请先调用", blocked)
            fill_context_for_report.invoke({})
            result = fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})
            self.assertIn("覆盖率:85%", result)
            self.assertIn("漏扫区域", result)
        finally:
            reset_tool_runtime_context(token)


class LazyMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_creation_does_not_connect_to_mcp(self) -> None:
        with patch("ts_agent.tools.mcp_tools.get_price_compare_mcp_tools", new=AsyncMock()) as load:
            tools = get_lazy_price_compare_tools()
            self.assertEqual([tool.name for tool in tools], ["jd.goods.query", "pdd.goods.search"])
            load.assert_not_awaited()

    async def test_combined_price_tool_enforces_platform_order(self) -> None:
        with patch(
            "ts_agent.tools.mcp_tools._invoke_price_tool",
            new=AsyncMock(side_effect=["jd-result", "pdd-result"]),
        ) as invoke:
            result = await get_price_comparison_tool().ainvoke({"keyword": "X1"})
            self.assertIn("jd-result", result)
            self.assertIn("pdd-result", result)
            self.assertEqual(
                [call.args[0] for call in invoke.await_args_list],
                ["jd.goods.query", "pdd.goods.search"],
            )


class BusinessActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.paths = {
            "TS_PURCHASE_ORDER_PATH": root / "orders.jsonl",
            "TS_AFTER_SALES_TICKET_PATH": root / "tickets.jsonl",
            "TS_AFTER_SALES_RETURN_PATH": root / "returns.jsonl",
        }
        self.previous = {key: os.environ.get(key) for key in self.paths}
        for key, path in self.paths.items():
            os.environ[key] = str(path)
        self.token = set_tool_runtime_context(
            {"user_id": "test-user", "thread_id": "test-thread", "route": "purchase"}
        )

    def tearDown(self) -> None:
        reset_tool_runtime_context(self.token)
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def test_purchase_order_is_validated_and_idempotent(self) -> None:
        payload = {
            "product_model": "X1",
            "quantity": 1,
            "consignee": "张三",
            "phone": "138-0000-0000",
            "address": "上海市测试路1号",
        }
        first = create_purchase_order.invoke(payload)
        second = create_purchase_order.invoke(payload)
        self.assertIn("订单已创建", first)
        self.assertIn("未重复下单", second)
        records = self._records("TS_PURCHASE_ORDER_PATH")
        self.assertEqual(len(records), 1)
        self.assertIn("idempotency_key", records[0])

        invalid = create_purchase_order.invoke({**payload, "quantity": 21})
        self.assertIn("1-20", invalid)

    def test_ticket_and_return_request_are_idempotent(self) -> None:
        ticket = {"summary": "无法开机", "symptoms": "按电源键无反应", "phone": "13800000000"}
        self.assertIn("工单已创建", create_after_sales_ticket.invoke(ticket))
        self.assertIn("未重复创建", create_after_sales_ticket.invoke(ticket))
        self.assertEqual(len(self._records("TS_AFTER_SALES_TICKET_PATH")), 1)

        request = {
            "reason": "不符合预期",
            "product_model": "X1",
            "phone": "13800000000",
            "address": "上海市测试路1号",
        }
        self.assertIn("申请已创建", create_manual_return_request.invoke(request))
        self.assertIn("未重复申请", create_manual_return_request.invoke(request))
        self.assertEqual(len(self._records("TS_AFTER_SALES_RETURN_PATH")), 1)

    def _records(self, env_name: str) -> list[dict]:
        path = Path(os.environ[env_name])
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
