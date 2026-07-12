"""Load and validate the project's YAML configuration files."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from utils.path_tool import get_abs_path


def load_yaml_config(path: str | PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping from an absolute or project-relative path."""

    config_path = Path(get_abs_path(path))
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"配置文件顶层必须是映射: {config_path}")
    return dict(data)


def load_rag_config(path: str | PathLike[str] = "config/rag.yml") -> dict[str, Any]:
    return load_yaml_config(path)


def load_chroma_config(
    path: str | PathLike[str] = "config/chroma.yml",
) -> dict[str, Any]:
    return load_yaml_config(path)


def load_prompts_config(
    path: str | PathLike[str] = "config/prompts.yml",
) -> dict[str, Any]:
    return load_yaml_config(path)


def load_agent_config(
    path: str | PathLike[str] = "config/agent.yml",
) -> dict[str, Any]:
    return load_yaml_config(path)


def _require_keys(name: str, config: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if config.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{name} 缺少必填字段: {', '.join(missing)}")


def _require_positive_int(name: str, config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name}.{key} 必须是正整数")
    return value


def validate_config() -> None:
    """Validate cross-file settings early and report actionable startup errors."""

    _require_keys("rag", rag_conf, ("chat_model_name", "chat_base_url", "embedding_model_name"))
    parsed_url = urlparse(str(rag_conf["chat_base_url"]))
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("rag.chat_base_url 必须是有效的 HTTP(S) URL")

    _require_keys(
        "chroma",
        chroma_conf,
        (
            "collection_name",
            "persist_directory",
            "data_path",
            "manifest_store",
            "allow_knowledge_file_type",
            "chunk_size",
            "chunk_overlap",
            "k",
        ),
    )
    chunk_size = _require_positive_int("chroma", chroma_conf, "chunk_size")
    chunk_overlap = chroma_conf["chunk_overlap"]
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or not 0 <= chunk_overlap < chunk_size
    ):
        raise ValueError("chroma.chunk_overlap 必须是小于 chunk_size 的非负整数")
    _require_positive_int("chroma", chroma_conf, "k")
    file_types = chroma_conf["allow_knowledge_file_type"]
    if not isinstance(file_types, list) or not file_types or not all(
        isinstance(item, str) and item.strip() for item in file_types
    ):
        raise ValueError("chroma.allow_knowledge_file_type 必须是非空字符串列表")

    _require_keys(
        "prompts",
        prompts_conf,
        (
            "main_prompt_path",
            "rag_summarize_prompt_path",
            "report_prompt_path",
            "summary_prompt_path",
            "after_sales_prompt_path",
        ),
    )
    for key, value in prompts_conf.items():
        if key.endswith("_path") and not Path(get_abs_path(value)).is_file():
            raise ValueError(f"prompts.{key} 指向的文件不存在: {value}")

    _require_keys("agent", agent_conf, ("external_data_path",))


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

validate_config()
