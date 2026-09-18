from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.run_eval import CATEGORIES, DATASET_PATH, load_dataset, validate_dataset


class EvalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_dataset(DATASET_PATH)

    def test_dataset_structure_is_valid(self) -> None:
        self.assertEqual(validate_dataset(self.cases), [])

    def test_dataset_scale_reaches_target(self) -> None:
        self.assertGreaterEqual(len(self.cases), 90)

    def test_case_ids_are_unique(self) -> None:
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_category_is_covered(self) -> None:
        counter = Counter(case["category"] for case in self.cases)
        for category in CATEGORIES:
            self.assertGreaterEqual(
                counter.get(category, 0),
                5,
                f"分类 {category} 用例数不足",
            )

    def test_removed_order_and_return_tools_are_absent(self) -> None:
        text = DATASET_PATH.read_text(encoding="utf-8")
        self.assertNotIn("create_purchase_order", text)
        self.assertNotIn("create_manual_return_request", text)

    def test_sensitive_tool_must_calls_are_scoped(self) -> None:
        # 建单工具的"必须调用"断言只允许出现在多轮转人工可能触达的分类；
        # 其他分类只允许出现"禁止调用"断言（如路由、选购、排障、报告场景）。
        allowed = {"ticket", "return_request", "mixed_intent", "context"}
        for case in self.cases:
            expected = case.get("expected", {})
            requires = set(expected.get("tools_must", []))
            for group in expected.get("tools_must_any", []):
                requires |= set(group)
            if "create_after_sales_ticket" in requires:
                self.assertIn(
                    case["category"],
                    allowed,
                    f"{case['id']}: 建单工具的必调断言不应出现在 {case['category']} 分类",
                )


if __name__ == "__main__":
    unittest.main()
