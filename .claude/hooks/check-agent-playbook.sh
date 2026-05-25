#!/bin/bash
# PostToolUse hook: when src/kb/cli.py is edited, remind to verify _AGENT_PLAYBOOK
# Reads tool input from stdin (JSON), extracts file_path.
# kbx CLAUDE.md states `kb --help` is the LLM API contract — _AGENT_PLAYBOOK in
# cli.py is what powers that. Any CLI change must update it.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" || "$FILE_PATH" != *"src/kb/cli.py" ]]; then
  exit 0
fi

cat <<'EOF'
You edited kbx's CLI. Verify _AGENT_PLAYBOOK in cli.py reflects the change
(commands, flags, JSON shapes). kbx's CLAUDE.md states `kb --help` is the
LLM API contract — keep it accurate.
EOF

exit 0
