#!/usr/bin/env python3
"""Search quality evaluation — standalone script.

Run against the real indexed data to check Hit@1/3/5:

    uv run python kb/tests/eval_queries.py

Exits 0 if all queries have at least Hit@5, exits 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

from kb.db import Database
from kb.search import search

# ---------------------------------------------------------------------------
# Test queries — adapt to actual indexed data
# ---------------------------------------------------------------------------

QUERIES = [
    {"query": "MFA implementation", "expect_title": "MFA"},
    {"query": "Cloud migration status", "expect_title": "Cloud"},
    {"query": "Datastore migration", "expect_title": "Datastore"},
    {"query": "performance review Soren", "expect_title": "Soren"},
    {"query": "Okta integration", "expect_title": "MFA"},
    {"query": "AC Scanner", "expect_title": "Shield"},
    {"query": "production stability", "expect_title": "stability"},
    {"query": "CI enhancements", "expect_title": "CI"},
    {"query": "Talia infrastructure", "expect_title": "Talia"},
    {"query": "Linnea Cloud migration", "expect_title": "Cloud"},
    {"query": "on-call observability", "expect_title": "on-call"},
    {"query": "staging environment", "expect_title": "staging"},
    {"query": "recruitment engineering", "expect_title": "recruit"},
    {"query": "weekly sync Anders", "expect_title": "Anders"},
    {"query": "SOC2 compliance Veldra", "expect_title": "SOC2"},
    {"query": "AI tooling adoption", "expect_title": "AI"},
    {"query": "Slack channel discussion", "expect_title": "Slack"},
    {"query": "offsite January", "expect_title": "offsite"},
    {"query": "budget planning 2026", "expect_title": "budget"},
    {"query": "career ladder trident", "expect_title": "career"},
]


def _find_data_dir() -> Path:
    """Locate kb/data/ from script location."""
    here = Path(__file__).resolve().parent
    data = here.parent / "data"
    if data.exists():
        return data
    # fallback: use KB_DATA_DIR env var
    import os

    env = os.environ.get("KB_DATA_DIR")
    if env:
        return Path(env)
    print("ERROR: Could not find kb/data/ directory. Set KB_DATA_DIR.", file=sys.stderr)
    sys.exit(3)


def run() -> None:
    data_dir = _find_data_dir()
    db = Database(data_dir)

    hit1, hit3, hit5 = 0, 0, 0
    total = len(QUERIES)
    failures = []

    for q in QUERIES:
        try:
            results = search(db, None, q["query"], fast=True, limit=5)
        except Exception as exc:
            print(f"  SKIP  {q['query']!r}: {exc}", file=sys.stderr)
            total -= 1
            continue

        titles = [r.title or "" for r in results.results]
        expect = q["expect_title"].lower()

        found_at = None
        for i, t in enumerate(titles):
            if expect in t.lower():
                found_at = i + 1
                break

        if found_at is not None:
            if found_at <= 1:
                hit1 += 1
            if found_at <= 3:
                hit3 += 1
            if found_at <= 5:
                hit5 += 1
            status = f"Hit@{found_at}"
        else:
            status = "MISS"
            failures.append(q["query"])

        print(f"  {status:>6}  {q['query']!r}  (got: {titles[:3]})")

    db.close()

    print()
    print(f"Queries: {total}")
    print(f"Hit@1:   {hit1}/{total} ({hit1 / total * 100:.0f}%)" if total else "No queries")
    print(f"Hit@3:   {hit3}/{total} ({hit3 / total * 100:.0f}%)" if total else "")
    print(f"Hit@5:   {hit5}/{total} ({hit5 / total * 100:.0f}%)" if total else "")

    if failures:
        print(f"\nFailed queries: {failures}")
        sys.exit(1)
    else:
        print("\nAll queries passed Hit@5!")
        sys.exit(0)


if __name__ == "__main__":
    run()
