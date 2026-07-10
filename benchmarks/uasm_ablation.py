"""Print deterministic A0-A3 state-machine fault-injection ablations as JSON."""

from __future__ import annotations

import asyncio
import json

from aeloon_core.ablation import run_ablation, summarize_ablation


def main() -> None:
    rows = asyncio.run(run_ablation())
    report = {"results": rows, "summary": summarize_ablation(rows)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
