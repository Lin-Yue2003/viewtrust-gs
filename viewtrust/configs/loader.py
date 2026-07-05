"""Small config loader for local-safe ViewTrust-GS checks.

This intentionally supports only the simple YAML subset used by the project
configs so local checks do not require PyYAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the indentation-based YAML subset used by ViewTrust configs."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected key/value pair")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if value:
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))

    return root
