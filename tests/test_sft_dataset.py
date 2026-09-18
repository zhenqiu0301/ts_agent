from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft.build_sft_dataset import REPO_ROOT, ContaminationGuard

CHAT_DATASET_PATH = REPO_ROOT / "sft" / "sft_chat.jsonl"
TRAJECTORY_PATH = REPO_ROOT / "sft" / "sft_trajectories.jsonl"
SFT_DATASET_PATH = REPO_ROOT / "sft" / "sft_dataset.jsonl"
BENCHMARK_PATH = REPO_ROOT / "evals" / "dataset.jsonl"

VALID_CATEGORIES = {
    "purchase_consult",
    "price_compare",
    "knowledge_qa",
    "troubleshoot",
    "usage_report",
    "ticket_create",
    "return_request",
    "unclear",
    "mixed_intent",
    "context_followup",
    "safety",
    "robustness",
}

# 每个分类在数据集中至少出现的组数（保证覆盖广度）
MIN_CASES_PER_CATEGORY = 5

# 生产中间件对工具结果的截断上限（TS_TOOL_RESULT_MAX_CHARS，默认 2000 + 截断标记）
TOOL_RESULT_MAX_LEN = 2100

# 同分类内两条用户 query 的相似度上限（difflib 比率），超过视为重复采样
MAX_USER_QUERY_SIMILARITY = 0.85

# 含工具调用的样本占总量的比例区间（对齐真实 Agent 行为分布：轨迹内 82% 含工具，
# 加上追问/模糊/安全等纯对话流量后约六成）
TOOL_SAMPLE_RATIO_BOUNDS = (0.45, 0.70)

# GLM 纯对话只允许覆盖真正零工具的场景；工具天然场景必须来自真实轨迹，
# 否则会教模型在不调工具的情况下"叙述正在查询"（虚构工具活动）。
CHAT_ALLOWED_CATEGORIES = VALID_CATEGORIES - {
    "price_compare",
    "knowledge_qa",
    "usage_report",
    "ticket_create",
    "return_request",
}

KNOWN_TOOL_NAMES = {
    "rag_summarize",
    "web_search",
    "get_user_context",
    "compare_prices",
    "get_usage_report_data",
    "create_after_sales_ticket",
    "fill_context_for_report",
    "fetch_external_data",
}


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_sft() -> list[dict]:
    return _load_jsonl(SFT_DATASET_PATH)


class SftDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SFT_DATASET_PATH.is_file():
            raise unittest.SkipTest(
                "sft/sft_dataset.jsonl 尚未生成，先运行 sft/build_sft_dataset.py"
            )
        cls.convs = load_sft()
        cls.benchmark_turns = [
            turn["content"]
            for line in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for turn in json.loads(line)["turns"]
        ]

    def test_dataset_has_conversations(self) -> None:
        self.assertGreaterEqual(len(self.convs), 200)
        self.assertGreaterEqual(
            len(self.convs) - len(_load_jsonl(CHAT_DATASET_PATH)),
            100,
            "工具轨迹数量不足",
        )

    def test_trajectory_files_merge_into_dataset(self) -> None:
        chat = _load_jsonl(CHAT_DATASET_PATH)
        traj = _load_jsonl(TRAJECTORY_PATH)
        self.assertGreaterEqual(len({c["id"] for c in chat} | {c["id"] for c in traj}), 200)
        self.assertEqual(len(self.convs), len(chat) + len(traj))
        sources = {c.get("source") for c in self.convs}
        self.assertIn("glm-chat", sources)
        self.assertIn("agent-trace", sources)

    def test_tool_trajectories_are_well_formed(self) -> None:
        checked = 0
        for conv in self.convs:
            if conv.get("source") != "agent-trace":
                continue
            messages = conv["messages"]
            tool_calls = [m for m in messages if m.get("tool_calls")]
            if not tool_calls:
                continue
            checked += 1
            # tools schema 必须存在且覆盖被调用的工具
            schema_names = {
                entry["function"]["name"] for entry in conv.get("tools", [])
            }
            self.assertTrue(schema_names, conv["id"])
            for message in tool_calls:
                self.assertEqual(message["role"], "assistant", conv["id"])
                for tool_call in message["tool_calls"]:
                    self.assertIn(tool_call["function"]["name"], schema_names, conv["id"])
                    self.assertTrue(tool_call.get("id"), conv["id"])
                    json.loads(tool_call["function"]["arguments"])  # arguments 必须是合法 JSON
            # 每个 tool_calls 之后必须紧跟对应的 tool 结果消息
            for index, message in enumerate(messages):
                for tool_call in message.get("tool_calls", []):
                    followers = messages[index + 1 :]
                    self.assertTrue(
                        any(
                            m["role"] == "tool" and m["tool_call_id"] == tool_call["id"]
                            for m in followers
                        ),
                        f"{conv['id']}: 工具调用缺少返回消息",
                    )
            # 被调用的工具必须在已知集合内
            for message in tool_calls:
                for tool_call in message["tool_calls"]:
                    self.assertIn(
                        tool_call["function"]["name"],
                        KNOWN_TOOL_NAMES,
                        conv["id"],
                    )
        self.assertGreaterEqual(checked, 80, "含工具调用的轨迹数量不足")

    def test_tool_required_categories_have_tool_calls(self) -> None:
        for conv in self.convs:
            if conv.get("source") != "agent-trace":
                continue
            if conv["category"] not in {"price_compare", "usage_report"}:
                continue
            has_tools = any(m.get("tool_calls") for m in conv["messages"])
            self.assertTrue(has_tools, f"{conv['id']}: 该场景必须包含工具调用")

    def test_ids_are_unique(self) -> None:
        ids = [conv["id"] for conv in self.convs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_categories_are_valid(self) -> None:
        for conv in self.convs:
            self.assertIn(conv["category"], VALID_CATEGORIES, conv["id"])

    def test_every_category_is_covered(self) -> None:
        import collections

        counter = collections.Counter(conv["category"] for conv in self.convs)
        for category in VALID_CATEGORIES:
            self.assertGreaterEqual(
                counter.get(category, 0),
                MIN_CASES_PER_CATEGORY,
                f"分类 {category} 覆盖不足",
            )

    def test_tool_sample_ratio_is_balanced(self) -> None:
        tool_samples = sum(
            1
            for conv in self.convs
            if conv.get("source") == "agent-trace" and conv["meta"]["has_tool_calls"]
        )
        ratio = tool_samples / len(self.convs)
        low, high = TOOL_SAMPLE_RATIO_BOUNDS
        self.assertGreaterEqual(ratio, low, f"工具样本占比过低：{ratio:.2f}")
        self.assertLessEqual(ratio, high, f"纯对话样本占比过高：{ratio:.2f}")

    def test_chat_samples_only_cover_tool_free_behaviors(self) -> None:
        for conv in _load_jsonl(CHAT_DATASET_PATH):
            self.assertIn(
                conv["category"],
                CHAT_ALLOWED_CATEGORIES,
                f"{conv['id']}: 工具天然场景不应有 GLM 纯对话版本"
                f"（会用无工具的回复教模型虚构工具活动）",
            )
            if conv["category"] in {"troubleshoot", "context_followup"}:
                continue
            for message in conv["messages"]:
                if message["role"] != "assistant":
                    continue
                for phrase in ("正在按", "已为您查询", "已为您比价", "统一查询"):
                    self.assertNotIn(
                        phrase,
                        message["content"],
                        f"{conv['id']}: 纯对话样本不得叙述未发生的工具调用",
                    )

    def test_tool_results_respect_production_cap(self) -> None:
        for conv in self.convs:
            if conv.get("source") != "agent-trace":
                continue
            for message in conv["messages"]:
                if message["role"] == "tool":
                    self.assertLessEqual(
                        len(message["content"]),
                        TOOL_RESULT_MAX_LEN,
                        f"{conv['id']}: 工具结果超过生产截断上限",
                    )

    def test_user_queries_have_no_near_duplicates_within_category(self) -> None:
        import difflib
        import unicodedata

        def norm(text: str) -> str:
            text = unicodedata.normalize("NFKC", text).lower()
            return re.sub(r"[\W_]+", "", text)

        by_category: dict[str, list[str]] = {}
        for conv in self.convs:
            first_turn = norm(conv["messages"][0]["content"])
            by_category.setdefault(conv["category"], []).append(first_turn)
        for category, turns in by_category.items():
            for i in range(len(turns)):
                for j in range(i + 1, len(turns)):
                    if not turns[i] or not turns[j]:
                        continue
                    ratio = difflib.SequenceMatcher(None, turns[i], turns[j]).ratio()
                    self.assertLessEqual(
                        ratio,
                        MAX_USER_QUERY_SIMILARITY,
                        f"分类 {category} 存在近似重复 query："
                        f"{turns[i][:24]!r} vs {turns[j][:24]!r}",
                    )

    def test_message_structure(self) -> None:
        for conv in self.convs:
            messages = conv["messages"]
            self.assertGreaterEqual(len(messages), 2, conv["id"])
            self.assertEqual(messages[0]["role"], "user", conv["id"])
            self.assertNotIn(
                "system",
                [m["role"] for m in messages],
                f"{conv['id']}: system 应放在独立字段，不占用 messages 轮次",
            )
            if conv.get("source") == "agent-trace":
                # 轨迹以 user 开头、无工具调用的 assistant 结尾
                self.assertNotIn(messages[-1]["role"], {"user", "tool"}, conv["id"])
                self.assertNotIn("tool_calls", messages[-1], conv["id"])
                continue
            for index in range(0, len(messages), 2):
                self.assertEqual(messages[index]["role"], "user", conv["id"])
            for index in range(1, len(messages), 2):
                self.assertEqual(messages[index]["role"], "assistant", conv["id"])

    def test_system_prompt_matches_online_prompts(self) -> None:
        online = {
            path.name: path.read_text(encoding="utf-8").strip()
            for path in (REPO_ROOT / "src/ts_agent/prompts").glob("*.txt")
        }
        known = set(online.values())
        for conv in self.convs:
            self.assertIn(
                conv["system"], known, f"{conv['id']}: system 不是线上提示词"
            )

    def test_replies_are_substantive_and_clean(self) -> None:
        for conv in self.convs:
            for message in conv["messages"]:
                if message["role"] != "assistant" or message.get("tool_calls"):
                    continue
                content = message["content"]
                self.assertGreaterEqual(len(content), 30, conv["id"])
                self.assertNotIn("采样说明", content, conv["id"])
                self.assertNotIn("【本轮用户消息】", content, conv["id"])
                self.assertNotIn("作为AI", content, conv["id"])

    def test_user_turns_do_not_overlap_benchmark(self) -> None:
        guard = ContaminationGuard(self.benchmark_turns)
        for conv in self.convs:
            for message in conv["messages"]:
                if message["role"] == "user":
                    self.assertFalse(
                        guard.is_contaminated(message["content"]),
                        f"{conv['id']}: 用户轮与 benchmark 重叠",
                    )


if __name__ == "__main__":
    unittest.main()
