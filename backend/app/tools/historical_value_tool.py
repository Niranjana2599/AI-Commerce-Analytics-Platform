"""Historical monetary ranking used only when saved CLV prediction is unavailable."""

from typing import Any


def run(args: dict[str, Any]) -> dict[str, Any]:
    rows = args.get("customers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("customers must be a non-empty cohort list")
    ranked = [dict(row) for row in sorted(rows, key=lambda row: float(row.get("monetary", 0) or 0), reverse=True)]
    return {"customers":ranked[:50],"method":"sorted by observed historical monetary value","is_model_prediction":False}
