#!/usr/bin/env bash
# Guard: fail if uv.lock references a non-public package index.
# Public repo => resolve only against pypi.org / files.pythonhosted.org.
# Deliberately does NOT print the offending host (CI logs are public).
set -euo pipefail
lock="${1:-uv.lock}"
[ -f "$lock" ] || exit 0
bad="$(grep -oE 'https://[^"/]+' "$lock" | sort -u \
        | grep -vxE 'https://(pypi\.org|files\.pythonhosted\.org)' || true)"
if [ -n "$bad" ]; then
  echo "ERROR: $lock references a non-public package index." >&2
  echo "Public repos must resolve only against pypi.org / files.pythonhosted.org." >&2
  echo "Regenerate (UV_INDEX must be UNSET — UV_DEFAULT_INDEX alone is not enough):" >&2
  echo "  env -u UV_INDEX -u PIP_EXTRA_INDEX_URL UV_DEFAULT_INDEX=https://pypi.org/simple uv lock --refresh" >&2
  exit 1
fi
