"""Path helpers for portable ViewTrust-GS scripts."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a file or directory inside the project."""

    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "configs").is_dir() and (candidate / "viewtrust").is_dir():
            return candidate

    raise FileNotFoundError(f"could not find ViewTrust-GS project root from {current}")


def resolve_relative_path(project_root: Path, raw_path: str) -> Path:
    """Resolve a config path while rejecting absolute paths in portable configs."""

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"expected a relative path, got {raw_path}")
    return (project_root / path).resolve()


def ensure_child_path(parent: Path, child: Path) -> None:
    """Fail if child does not resolve under parent."""

    parent = parent.resolve()
    child = child.resolve()
    if parent != child and parent not in child.parents:
        raise ValueError(f"{child} is outside allowed parent {parent}")
