import pytest
from agentic_rag.orchestrator import AgenticRetrievalOrchestrator, ToolRegistry, ToolSpec
from agentic_rag.citations import extract_citations


def test_empty_query_rejected():
    registry = ToolRegistry([ToolSpec("vector_search", "vector", lambda q: ["x"])])
    with AgenticRetrievalOrchestrator(registry) as rag:
        with pytest.raises(Exception, match="cannot be empty"):
            rag.run("")


def test_citation_extraction():
    assert extract_citations("Claim [SRC-abc1234567]") == {"[SRC-abc1234567]"}
