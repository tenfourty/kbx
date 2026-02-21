"""Output formatting — table, JSON, JSONL, CSV with field selection and jq filtering."""

from __future__ import annotations

import csv
import io
import json
import sys
from typing import Any


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as JSON string."""
    return json.dumps(data, indent=indent, default=str, ensure_ascii=False)


def format_jsonl(items: list[dict[str, Any]]) -> str:
    """Format items as line-delimited JSON (one JSON object per line)."""
    lines = [json.dumps(item, default=str, ensure_ascii=False) for item in items]
    return "\n".join(lines)


def format_csv_str(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Format rows as CSV string."""
    if not rows:
        return ""
    cols = columns or list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in cols})
    return buf.getvalue()


def format_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Format rows as a pretty aligned table using rich, with plain-text fallback."""
    if not rows:
        return "No results."
    try:
        from rich.console import Console
        from rich.table import Table
    except (ImportError, ModuleNotFoundError):
        return _format_table_plain(rows, columns)

    cols = columns or list(rows[0].keys())
    table = Table(show_header=True, header_style="bold")
    for col in cols:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(row.get(c, "")) for c in cols])

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    console.print(table)
    return buf.getvalue()


def _format_table_plain(
    rows: list[dict[str, Any]], columns: list[str] | None = None, max_col_width: int = 60
) -> str:
    """Plain-text table fallback when rich is not installed."""
    cols = columns or list(rows[0].keys())

    def _trunc(val: str) -> str:
        return val[: max_col_width - 3] + "..." if len(val) > max_col_width else val

    # Compute column widths
    col_vals: dict[str, list[str]] = {c: [] for c in cols}
    for row in rows:
        for c in cols:
            col_vals[c].append(_trunc(str(row.get(c, ""))))

    widths = {c: max(len(c), max((len(v) for v in col_vals[c]), default=0)) for c in cols}

    # Build output
    lines: list[str] = []
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    lines.append(header)
    lines.append("  ".join("-" * widths[c] for c in cols))
    for i, _row in enumerate(rows):
        line = "  ".join(col_vals[c][i].ljust(widths[c]) for c in cols)
        lines.append(line)
    return "\n".join(lines) + "\n"


def select_fields(data: Any, fields: list[str]) -> Any:
    """Filter output to specified fields only."""
    if isinstance(data, dict):
        # If data has a 'results' key, filter within it
        if "results" in data:
            data = {**data, "results": [_pick(r, fields) for r in data["results"]]}
            return data
        return _pick(data, fields)
    if isinstance(data, list):
        return [_pick(item, fields) for item in data]
    return data


def _pick(d: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k in fields}


def apply_jq(data: Any, expr: str) -> Any:
    """Apply a jq expression to data. Returns the transformed result."""
    import jq

    return jq.first(expr, data)


def render(
    data: Any,
    fmt: str = "table",
    fields: list[str] | None = None,
    jq_expr: str | None = None,
    columns: list[str] | None = None,
) -> str:
    """Render data in the specified format.

    Args:
        data: The data to render (dict or list)
        fmt: Output format — table, json, jsonl, csv
        fields: Optional field selection (applied before formatting)
        jq_expr: Optional jq expression (applied after field selection)
        columns: Column order for table/csv output
    """
    if fields:
        data = select_fields(data, fields)

    if jq_expr:
        data = apply_jq(data, jq_expr)

    if fmt == "json":
        return format_json(data)
    elif fmt == "jsonl":
        items = data if isinstance(data, list) else data.get("results", [data])
        return format_jsonl(items)
    elif fmt == "csv":
        rows = data if isinstance(data, list) else data.get("results", [data])
        return format_csv_str(rows, columns)
    elif fmt == "table":
        rows = data if isinstance(data, list) else data.get("results", [data])
        return format_table(rows, columns)
    else:
        return format_json(data)


def output_error(
    error_type: str, message: str, suggestions: list[str] | None = None, exit_code: int = 1
) -> None:
    """Write a structured error to stderr and exit."""
    inner: dict[str, Any] = {"type": error_type, "message": message}
    if suggestions:
        inner["suggestions"] = suggestions
    err: dict[str, Any] = {"error": inner}
    print(json.dumps(err, ensure_ascii=False), file=sys.stderr)
    sys.exit(exit_code)
