"""Project path helpers.

All relative paths are resolved from the repository root so callers behave
consistently regardless of the current working directory.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> str:
    """Return the absolute repository root path."""

    return str(PROJECT_ROOT)


def get_abs_path(path: str | PathLike[str]) -> str:
    """Return an absolute path, resolving relative paths from the repo root."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate.resolve())
