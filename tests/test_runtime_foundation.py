"""Smoke tests for configuration, paths, and prompt loading."""

from __future__ import annotations

import unittest
from pathlib import Path

from ts_agent.utils.config_handler import (
    agent_conf,
    chroma_conf,
    load_yaml_config,
    prompts_conf,
    rag_conf,
    validate_config,
)
from ts_agent.utils.path_tool import get_abs_path, get_project_root
from ts_agent.utils.prompt_loader import (
    load_after_sales_prompts,
    load_purchase_prompts,
    load_rag_prompts,
    load_report_prompts,
    load_summary_prompts,
    load_system_prompts,
)


class PathToolTests(unittest.TestCase):
    def test_application_uses_src_package_layout(self) -> None:
        import ts_agent

        package_path = Path(ts_agent.__file__).resolve()
        self.assertIn("src/ts_agent", package_path.as_posix())

    def test_project_root_contains_project_metadata(self) -> None:
        root = Path(get_project_root())
        self.assertTrue(root.is_absolute())
        self.assertTrue((root / "pyproject.toml").is_file())

    def test_relative_paths_are_resolved_from_project_root(self) -> None:
        expected = Path(get_project_root()) / "config" / "rag.yml"
        self.assertEqual(Path(get_abs_path("config/rag.yml")), expected.resolve())


class ConfigTests(unittest.TestCase):
    def test_required_configuration_is_loaded(self) -> None:
        self.assertIn("chat_model_name", rag_conf)
        self.assertEqual(rag_conf["chat_model_name"], "deepseek-v4-flash")
        self.assertEqual(rag_conf["chat_base_url"], "https://api.deepseek.com")
        self.assertIn("embedding_model_name", rag_conf)
        self.assertIn("persist_directory", chroma_conf)
        self.assertIn("main_prompt_path", prompts_conf)
        self.assertIn("purchase_prompt_path", prompts_conf)
        self.assertIn("external_data_path", agent_conf)

    def test_configuration_schema_is_valid(self) -> None:
        validate_config()

    def test_non_mapping_yaml_is_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yml"
            path.write_text("- item\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "顶层必须是映射"):
                load_yaml_config(path)


class PromptLoaderTests(unittest.TestCase):
    def test_all_configured_prompts_are_non_empty(self) -> None:
        loaders = (
            load_system_prompts,
            load_rag_prompts,
            load_report_prompts,
            load_summary_prompts,
            load_after_sales_prompts,
            load_purchase_prompts,
        )
        for loader in loaders:
            with self.subTest(loader=loader.__name__):
                self.assertTrue(loader().strip())


if __name__ == "__main__":
    unittest.main()
