from agentic_rag.orchestrator import AgenticRetrievalOrchestrator, ConversationMemory, ToolRegistry, ToolSpec


def test_multitool_agentic_run():
    registry = ToolRegistry([
        ToolSpec("sql_db", "structured", lambda q: [{"content": "high churn", "score": 0.95}]),
        ToolSpec("vector_search", "unstructured", lambda q: [{"content": "unresolved ticket", "score": 0.90}]),
    ])
    with AgenticRetrievalOrchestrator(registry, min_evidence=2, max_iterations=2, max_retries=0) as rag:
        result = rag.run("Which customer has high churn and unresolved tickets?")
    assert result.confidence_achieved is True
    assert set(result.sources) == {"sql_db", "vector_search"}


def test_retry():
    state = {"n": 0}
    def flaky(_):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("temporary")
        return [{"content": "recovered", "score": 0.9}]

    registry = ToolRegistry([ToolSpec("vector_search", "vector", flaky)])
    with AgenticRetrievalOrchestrator(registry, min_evidence=1, max_iterations=1, max_retries=1) as rag:
        result = rag.run("find evidence")
    assert result.outcomes[0].attempts == 2
    assert result.outcomes[0].succeeded


def test_memory_bounded():
    m = ConversationMemory(max_turns=2)
    for x in range(3):
        m.add(str(x), str(x))
    assert len(m.snapshot()) == 2
