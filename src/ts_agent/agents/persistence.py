"""Async SQLite-backed LangGraph persistence with explicit lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from ts_agent.utils.path_tool import get_abs_path


@dataclass
class PersistentBackends:
    checkpointer: AsyncSqliteSaver
    store: AsyncSqliteStore
    checkpoint_connection: aiosqlite.Connection
    store_connection: aiosqlite.Connection

    async def close(self) -> None:
        """Close both SQLite connections; safe to call more than once."""

        for connection in (self.checkpoint_connection, self.store_connection):
            try:
                await connection.close()
            except ValueError:
                pass


async def _connect(path: Path) -> aiosqlite.Connection:
    connection = await aiosqlite.connect(str(path), isolation_level=None, timeout=30)
    await connection.execute("PRAGMA journal_mode=WAL")
    await connection.execute("PRAGMA busy_timeout=30000")
    return connection


async def build_persistent_backends(
    base_dir: str | Path = "data/long_memory",
    db_name: str | None = None,
    checkpoint_db_name: str = "checkpointer.sqlite",
    store_db_name: str = "store.sqlite",
) -> PersistentBackends:
    """Create and initialize project-root-relative async persistence backends."""

    base_path = Path(get_abs_path(base_dir))
    base_path.mkdir(parents=True, exist_ok=True)

    if db_name:
        stem = Path(db_name).stem or "main_graph_memory"
        checkpoint_db_name = f"{stem}_checkpointer.sqlite"
        store_db_name = f"{stem}_store.sqlite"

    checkpoint_connection = await _connect(base_path / checkpoint_db_name)
    try:
        store_connection = await _connect(base_path / store_db_name)
    except BaseException:
        await checkpoint_connection.close()
        raise
    checkpointer = AsyncSqliteSaver(checkpoint_connection)
    store = AsyncSqliteStore(store_connection)
    try:
        await checkpointer.setup()
        await store.setup()
    except BaseException:
        # 半初始化失败时关闭已建立的连接，避免泄漏
        for connection in (checkpoint_connection, store_connection):
            try:
                await connection.close()
            except Exception:
                pass
        raise

    return PersistentBackends(
        checkpointer=checkpointer,
        store=store,
        checkpoint_connection=checkpoint_connection,
        store_connection=store_connection,
    )
