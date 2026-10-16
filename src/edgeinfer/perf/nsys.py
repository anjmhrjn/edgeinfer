"""Summaries of `nsys stats --format csv` output (C4 task 6: where the time goes).

The trace itself (.nsys-rep) is for looking at in the Nsight Systems GUI; these summaries
are what a report records: GPU kernel time by kernel, memory-copy time by direction,
and CUDA API time by call, all as totals over the profiled run.
"""

from __future__ import annotations

import csv
from io import StringIO
from typing import Any

# Column names in nsys 2026.5.1, checked on the T4 on 2026-10-16. Every summary has the
# total; the count and name columns differ by report.
TOTAL = "Total Time (ns)"
COLUMNS = {  # report -> (count column, name column)
    "cuda_gpu_kern_sum": ("Instances", "Name"),
    "cuda_gpu_mem_time_sum": ("Count", "Operation"),
    "cuda_api_sum": ("Num Calls", "Name"),
}


class NsysFormatError(ValueError):
    pass


def _rows(text: str, required: tuple[str, ...]) -> list[dict[str, str]]:
    reader = csv.DictReader(StringIO(text))
    missing = [c for c in required if c not in (reader.fieldnames or [])]
    if missing:
        raise NsysFormatError(f"nsys CSV lacks columns {missing}; header: {reader.fieldnames}")
    return list(reader)


def summarize_table(text: str, report: str, top: int = 10) -> dict[str, Any]:
    """Total ms of one `nsys stats` CSV report, and its `top` entries by total time with
    their share of it."""
    count_col, name_col = COLUMNS[report]
    rows = _rows(text, (TOTAL, count_col, name_col))
    total_ns = sum(float(r[TOTAL]) for r in rows)
    ranked = sorted(rows, key=lambda r: float(r[TOTAL]), reverse=True)
    return {
        "total_ms": total_ns / 1e6,
        "entries": len(rows),
        "top": [
            {
                "name": r[name_col],
                "total_ms": float(r[TOTAL]) / 1e6,
                "count": int(float(r[count_col])),
                "share": float(r[TOTAL]) / total_ns if total_ns else 0.0,
            }
            for r in ranked[:top]
        ],
    }
