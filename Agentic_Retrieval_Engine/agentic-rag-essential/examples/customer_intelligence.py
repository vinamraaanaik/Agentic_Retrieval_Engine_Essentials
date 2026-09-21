from agentic_rag.orchestrator import AgenticRetrievalOrchestrator, ToolRegistry, ToolSpec


def sql_db(_):
    return [{"content": "Acme Corp: churn risk is high; revenue declined 18%.", "score": 0.94}]


def vector_search(_):
    return [{"content": "Acme Corp has two unresolved reliability escalations.", "score": 0.92}]


registry = ToolRegistry([
    ToolSpec("sql_db", "Structured customer/account data.", sql_db),
    ToolSpec("vector_search", "Support tickets and documents.", vector_search),
])

with AgenticRetrievalOrchestrator(registry, min_evidence=2, max_iterations=2) as rag:
    result = rag.run("Which customers have high churn risk and unresolved support escalations, and why?")
    print(result.answer)
    print("trace_id:", result.trace_id)
    print("sources:", result.sources)
