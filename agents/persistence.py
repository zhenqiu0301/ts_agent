"""SQLite-backed LangGraph persistence with explicit lifecycle management."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from utils.path_tool import get_abs_path


@dataclass
class PersistentBackends:
    checkpointer: SqliteSaver
    store: SqliteStore
    checkpoint_connection: sqlite3.Connection
    store_connection: sqlite3.Connection

    def close(self) -> None:
        """Close both SQLite connections; safe to call more than once."""

        for connection in (self.checkpoint_connection, self.store_connection):
            try:
                connection.close()
            except sqlite3.ProgrammingError:
                pass


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None,
        timeout=30,
    )
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def build_persistent_backends(
    base_dir: str | Path = "data/long_memory",
    db_name: str | None = None,
    checkpoint_db_name: str = "checkpointer.sqlite",
    store_db_name: str = "store.sqlite",
) -> PersistentBackends:
    """Create project-root-relative checkpoint and store databases."""

    base_path = Path(get_abs_path(base_dir))
    base_path.mkdir(parents=True, exist_ok=True)

    if db_name:
        stem = Path(db_name).stem or "main_graph_memory"
        checkpoint_db_name = f"{stem}_checkpointer.sqlite"
        store_db_name = f"{stem}_store.sqlite"

    checkpoint_connection = _connect(base_path / checkpoint_db_name)
    store_connection = _connect(base_path / store_db_name)

    checkpointer = SqliteSaver(checkpoint_connection)
    checkpointer.setup()
    store = SqliteStore(store_connection)
    store.setup()

    return PersistentBackends(
        checkpointer=checkpointer,
        store=store,
        checkpoint_connection=checkpoint_connection,
        store_connection=store_connection,
    )
