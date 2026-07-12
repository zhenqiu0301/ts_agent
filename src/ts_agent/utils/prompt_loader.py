"""Load prompt templates declared in ``config/prompts.yml``."""

from __future__ import annotations

from pathlib import Path

from ts_agent.utils.config_handler import prompts_conf
from ts_agent.utils.path_tool import get_abs_path


def _load_prompt(config_key: str) -> str:
    relative_path = prompts_conf.get(config_key)
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise KeyError(f"缺少有效的提示词配置项: {config_key}")

    prompt_path = Path(get_abs_path(relative_path))
    if not prompt_path.is_file():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_path}")

    content = prompt_path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"提示词文件为空: {prompt_path}")
    return content


def load_system_prompts() -> str:
    return _load_prompt("main_prompt_path")


def load_rag_prompts() -> str:
    return _load_prompt("rag_summarize_prompt_path")


def load_report_prompts() -> str:
    return _load_prompt("report_prompt_path")


def load_summary_prompts() -> str:
    return _load_prompt("summary_prompt_path")


def load_after_sales_prompts() -> str:
    return _load_prompt("after_sales_prompt_path")
