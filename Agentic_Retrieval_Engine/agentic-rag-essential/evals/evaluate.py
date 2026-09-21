"""Small evaluation harness; replace the demo runner with your real backend in CI."""
import json
from pathlib import Path


def score_result(result, required_sources):
    sources = set(getattr(result, "sources", ()))
    return {
        "source_coverage": all(s in sources for s in required_sources),
        "grounded": bool(getattr(result, "confidence_achieved", False)),
    }


if __name__ == "__main__":
    data = json.loads(Path(__file__).with_name("dataset.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} evaluation cases.")
