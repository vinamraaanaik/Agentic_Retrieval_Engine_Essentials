from __future__ import annotations

from typing import Any, Callable

from .orchestrator import ToolRegistry, ToolSpec


class SQLDatabaseTool:
    """Read-only SQLAlchemy tool."""

    def __init__(self, database_url: str, max_rows: int = 100) -> None:
        from sqlalchemy import create_engine
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.max_rows = max_rows

    def __call__(self, query: str) -> list[dict[str, Any]]:
        from sqlalchemy import text
        lowered = query.strip().lower()
        if lowered.startswith(("insert ", "update ", "delete ", "drop ", "alter ", "truncate ", "create ")):
            raise ValueError("Only read-only SQL is allowed.")
        statement = query.strip().rstrip(";")
        if " limit " not in statement.lower():
            statement = f"{statement} LIMIT {self.max_rows}"
        with self.engine.connect() as conn:
            rows = conn.execute(text(statement)).mappings().all()
        return [{"content": str(dict(r)), "source": "postgresql", "score": 0.8} for r in rows]


class QdrantVectorSearchTool:
    """Qdrant semantic retrieval tool using a caller-supplied embedding function."""

    def __init__(
        self,
        url: str,
        collection: str,
        embed: Callable[[str], list[float]],
        api_key: str | None = None,
        limit: int = 8,
    ) -> None:
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.embed = embed
        self.limit = limit

    def __call__(self, query: str) -> list[dict[str, Any]]:
        vector = self.embed(query)
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=self.limit,
            with_payload=True,
        ).points
        return [
            {
                "content": str((hit.payload or {}).get("content", (hit.payload or {}).get("text", hit.payload or {}))),
                "source": str((hit.payload or {}).get("source", self.collection)),
                "score": float(hit.score),
                "metadata": dict(hit.payload or {}),
            }
            for hit in hits
        ]


def build_registry(*specs: ToolSpec) -> ToolRegistry:
    return ToolRegistry(specs)
