# Agentic RAG Platform

A production-oriented multi-tool Agentic RAG orchestration platform built around:

**Plan → Act → Observe → Reflect → Re-plan → Generate**

## What it demonstrates

- LLM-capable structured planning with deterministic fallback
- SQL + vector retrieval as independent tools
- Parallel tool execution
- Retries, exponential backoff and timeouts
- Evidence normalization, deduplication and reranking
- Reflective evidence sufficiency checks
- Iterative re-planning
- Grounded generation with citation validation
- Bounded conversation memory
- Execution budgets and trace metadata
- FastAPI API
- Tests, CI, Docker and evaluation scaffolding

## Architecture

![Architecture](diagrams/architecture.png)

See `docs/architecture.md` for the detailed flow.

## Project structure

```text
src/agentic_rag/
├── orchestrator.py   # core agentic control loop
├── agents.py         # optional OpenAI model adapter
├── tools.py          # SQL + Qdrant adapters
├── api.py            # FastAPI application
├── config.py         # environment configuration
├── citations.py      # source ID extraction
└── security.py       # input boundary helper

tests/                # unit + API tests
evals/                # small fixed evaluation set
docs/                 # architecture + evaluation
diagrams/             # architecture image
examples/             # runnable demo
```

## Local setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Optional LLM configuration:

```bash
copy .env.example .env
```

Set `OPENAI_API_KEY` to enable model-driven planning, reflection and generation. Without it, the project remains runnable using deterministic behavior and mock data.

## Run tests

```bash
pytest -q
```


### Query endpoint

`POST /api/v1/query`

```json
{
  "query": "Which customers have high churn risk and unresolved support tickets?"
}
```

The response contains the grounded answer, trace ID, iterations, confidence state, latency and sources.

## Agentic loop

```text
Question
   ↓
Plan
   ↓
Retrieve from independent tools
   ↓
Normalize + deduplicate + rerank evidence
   ↓
Reflect: is the evidence sufficient?
   ├── No → construct follow-up retrieval task → Plan again
   └── Yes → grounded generation
                    ↓
              citation validation
                    ↓
                 response
```

## Security / production notes

Retrieved text is treated as untrusted context. User input is bounded. The sample SQL tool is read-only at the statement boundary. Production deployments should add authentication, authorization, tenant isolation, rate limiting, stronger SQL allow-listing, secret management, PII controls and audit logging.

## Evaluation

`evals/dataset.json` provides a small fixed benchmark shape. Do not publish performance numbers unless they are generated from a reproducible evaluation run.

## License

MIT
