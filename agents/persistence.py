from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


def build_persistent_backends(
    base_dir: str | Path = "data/long_memory",
    db_name: str | None = None,
    checkpoint_db_name: str = "checkpointer.sqlite",
    store_db_name: str = "store.sqlite",
) -> tuple[SqliteSaver, SqliteStore]:
    """Create persistent checkpoint/store backends under data/long_memory.

    checkpointer 和 store 使用独立 SQLite 连接与文件，避免共享同一连接。
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    if db_name:
        stem = Path(db_name).stem or "main_graph_memory"
        checkpoint_db_name = f"{stem}_checkpointer.sqlite"
        store_db_name = f"{stem}_store.sqlite"

    checkpoint_db_path = base_path / checkpoint_db_name
    store_db_path = base_path / store_db_name

    checkpoint_conn = sqlite3.connect(
        str(checkpoint_db_path),
        check_same_thread=False,
        isolation_level=None,
    )
    store_conn = sqlite3.connect(
        str(store_db_path),
        check_same_thread=False,
        isolation_level=None,
    )

    checkpointer = SqliteSaver(checkpoint_conn)
    checkpointer.setup()

    store = SqliteStore(store_conn)
    store.setup()

    return checkpointer, store
