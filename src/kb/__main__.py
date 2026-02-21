"""Allow running kb as `python -m kb` — works on any system without venv shebangs."""

from kb.cli import cli

cli()
