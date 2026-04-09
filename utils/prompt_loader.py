import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(
            f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项"
        )
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


def load_summary_prompts():
    try:
        summary_prompt_path = get_abs_path(prompts_conf["summary_prompt_path"])
    except KeyError as e:
        logger.error(
            f"[load_summary_prompts]在yaml配置项中没有summary_prompt_path配置项"
        )
        raise e

    try:
        return open(summary_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_summary_prompts]解析摘要提示词出错，{str(e)}")
        raise e


def load_after_sales_prompts():
    try:
        after_sales_prompt_path = get_abs_path(prompts_conf["after_sales_prompt_path"])
    except KeyError as e:
        logger.error(
            "[load_after_sales_prompts]在yaml配置项中没有after_sales_prompt_path配置项"
        )
        raise e

    try:
        return open(after_sales_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_after_sales_prompts]解析售后提示词出错，{str(e)}")
        raise e


if __name__ == "__main__":
    print(load_report_prompts())
