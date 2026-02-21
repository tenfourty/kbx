"""Tests for kb.output — output formatting with graceful fallback."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch


class TestPydanticModelDump:
    """Verify output functions handle dicts from Pydantic .model_dump()."""

    def test_format_json_with_model_dump(self) -> None:
        from kb.output import format_json
        from kb.types import SearchResult

        r = SearchResult(
            chunk_id=1,
            document_id=1,
            title="T",
            path="p",
            date=None,
            doc_type="notes",
            score=0.5,
            section=None,
            snippet="s",
            entities=[],
        )
        result = format_json(r.model_dump())
        assert '"title": "T"' in result

    def test_render_json_with_search_response_dump(self) -> None:
        from kb.output import render
        from kb.types import SearchMeta, SearchResponse, SearchResult

        sr = SearchResult(
            chunk_id=1,
            document_id=1,
            title="Test Doc",
            path="test/path.md",
            date="2025-01-01",
            doc_type="notes",
            score=0.85,
            section="Intro",
            snippet="A snippet",
            entities=["Wren"],
        )
        meta = SearchMeta(query="test", total=1, limit=5, sort_by="score", execution_ms=10.0)
        resp = SearchResponse(results=[sr], meta=meta)
        result = render(resp.model_dump(), fmt="json")
        assert '"Test Doc"' in result
        assert '"results"' in result

    def test_render_table_with_glossary_entry_dump(self) -> None:
        from kb.output import render
        from kb.types import GlossaryEntry

        entry = GlossaryEntry(
            term="API", expansion="Application Programming Interface", section="Acronyms"
        )
        result = render(
            {"results": [entry.model_dump()], "meta": {"total": 1}},
            fmt="table",
            columns=["term", "expansion"],
        )
        assert "API" in result


class TestFormatTablePlainFallback:
    """When rich is not installed, format_table should still produce readable output."""

    def test_plain_fallback_has_headers(self):
        """Plain text table should include column headers."""
        rows = [{"name": "Wren", "role": "Engineer"}, {"name": "Soren", "role": "Manager"}]
        with patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None}):
            # Force reimport to pick up the patched modules
            import importlib

            import kb.output

            importlib.reload(kb.output)
            try:
                result = kb.output.format_table(rows)
                assert "name" in result
                assert "role" in result
            finally:
                importlib.reload(kb.output)

    def test_plain_fallback_has_data(self):
        """Plain text table should include data values."""
        rows = [{"name": "Wren", "role": "Engineer"}]
        with patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None}):
            import importlib

            import kb.output

            importlib.reload(kb.output)
            try:
                result = kb.output.format_table(rows)
                assert "Wren" in result
                assert "Engineer" in result
            finally:
                importlib.reload(kb.output)

    def test_plain_fallback_empty_rows(self):
        """Empty rows should return 'No results.' regardless of rich availability."""
        from kb.output import format_table

        assert format_table([]) == "No results."

    def test_plain_fallback_with_columns(self):
        """Column selection should work in plain fallback."""
        rows = [{"name": "Wren", "role": "Engineer", "age": "30"}]
        with patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None}):
            import importlib

            import kb.output

            importlib.reload(kb.output)
            try:
                result = kb.output.format_table(rows, columns=["name", "role"])
                assert "name" in result
                assert "role" in result
                assert "age" not in result
            finally:
                importlib.reload(kb.output)

    def test_plain_fallback_truncates_long_values(self):
        """Values longer than 60 chars should be truncated."""
        long_val = "x" * 100
        rows = [{"data": long_val}]
        with patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None}):
            import importlib

            import kb.output

            importlib.reload(kb.output)
            try:
                result = kb.output.format_table(rows)
                # The full 100-char value should not appear
                assert long_val not in result
                # But a truncated version should
                assert "x" * 57 in result
            finally:
                importlib.reload(kb.output)

    def test_rich_table_still_works(self):
        """When rich IS available, format_table should still work normally."""
        from kb.output import format_table

        rows = [{"name": "Wren", "role": "Engineer"}]
        result = format_table(rows)
        assert "Wren" in result
        assert "Engineer" in result


# ---------------------------------------------------------------------------
# Render formats
# ---------------------------------------------------------------------------


class TestRenderFormats:
    def test_render_json(self):
        from kb.output import render

        data = {"key": "value"}
        result = render(data, fmt="json")
        assert json.loads(result) == {"key": "value"}

    def test_render_jsonl(self):
        from kb.output import render

        data = {"results": [{"a": 1}, {"a": 2}]}
        result = render(data, fmt="jsonl")
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}

    def test_render_csv(self):
        from kb.output import render

        data = {"results": [{"name": "Wren", "age": "30"}]}
        result = render(data, fmt="csv")
        assert "name" in result
        assert "Wren" in result

    def test_render_unknown_format_falls_back_to_json(self):
        from kb.output import render

        data = {"key": "value"}
        result = render(data, fmt="xml")
        assert json.loads(result) == {"key": "value"}

    def test_render_with_jq_expression(self):
        from kb.output import render

        data = {"results": [{"name": "Wren"}]}
        result = render(data, fmt="json", jq_expr=".results[0].name")
        assert "Wren" in result

    def test_render_with_field_selection(self):
        from kb.output import render

        data = {"results": [{"name": "Wren", "age": 30, "city": "Paris"}]}
        result = render(data, fmt="json", fields=["name"])
        parsed = json.loads(result)
        assert "city" not in str(parsed["results"][0])


# ---------------------------------------------------------------------------
# select_fields
# ---------------------------------------------------------------------------


class TestSelectFields:
    def test_select_fields_on_list(self):
        from kb.output import select_fields

        data = [{"name": "A", "age": 1}, {"name": "B", "age": 2}]
        result = select_fields(data, ["name"])
        assert result == [{"name": "A"}, {"name": "B"}]

    def test_select_fields_on_dict_without_results(self):
        from kb.output import select_fields

        data = {"name": "A", "age": 1, "city": "Paris"}
        result = select_fields(data, ["name", "city"])
        assert result == {"name": "A", "city": "Paris"}


# ---------------------------------------------------------------------------
# output_error
# ---------------------------------------------------------------------------


class TestOutputError:
    def test_output_error_exits_with_code(self):
        import pytest

        from kb.output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("not_found", "Item not found", exit_code=1)
        assert exc_info.value.code == 1

    def test_output_error_with_suggestions(self, capsys):
        import pytest

        from kb.output import output_error

        with pytest.raises(SystemExit):
            output_error("not_found", "Not found", suggestions=["try X", "try Y"])
        captured = capsys.readouterr()
        assert "suggestions" in captured.err


# ---------------------------------------------------------------------------
# apply_jq
# ---------------------------------------------------------------------------


class TestApplyJq:
    def test_apply_jq_extracts_field(self):
        from kb.output import apply_jq

        data = {"results": [{"name": "Wren"}, {"name": "Soren"}]}
        result = apply_jq(data, ".results[0].name")
        assert result == "Wren"


# ---------------------------------------------------------------------------
# format_csv_str
# ---------------------------------------------------------------------------


class TestFormatCSV:
    def test_format_csv_empty(self):
        from kb.output import format_csv_str

        assert format_csv_str([]) == ""

    def test_format_csv_with_columns(self):
        from kb.output import format_csv_str

        rows = [{"name": "A", "age": "1", "city": "P"}]
        result = format_csv_str(rows, columns=["name", "age"])
        assert "name" in result
        assert "city" not in result


# ---------------------------------------------------------------------------
# format_jsonl
# ---------------------------------------------------------------------------


class TestFormatJSONL:
    def test_format_jsonl(self):
        from kb.output import format_jsonl

        items = [{"a": 1}, {"b": 2}]
        result = format_jsonl(items)
        lines = result.strip().split("\n")
        assert len(lines) == 2
