# Output Formatting

## Overview

`kb/output.py` provides consistent output formatting across all commands, following the `gh` CLI pattern.

## Formats

| Format | Flag | Use Case |
|--------|------|----------|
| `table` | `--format table` (default) | Human-readable aligned columns (via `rich`) |
| `json` | `--format json` / `--json` | Full structured output |
| `jsonl` | `--format jsonl` | Line-delimited JSON for streaming/piping |
| `csv` | `--format csv` | Spreadsheet export |

## Field Selection

`--fields title,date,score` — select specific fields. Applied before formatting. For results with a `results` key, filtering is applied within the results array.

## jq Filtering

`--jq '.results[0].title'` — built-in jq expression applied after field selection. Requires the `jq` Python package.

## Error Output

Structured JSON on stderr with `suggestion` and exit codes:
- `0` = success
- `1` = validation error
- `2` = ambiguous query
- `3` = system error

## Implementation

`render()` is the main entry point:
1. Apply field selection (`select_fields`)
2. Apply jq expression (`apply_jq`)
3. Format output (table/json/jsonl/csv)

Human messaging (progress, warnings) goes to stderr. Machine output to stdout.
