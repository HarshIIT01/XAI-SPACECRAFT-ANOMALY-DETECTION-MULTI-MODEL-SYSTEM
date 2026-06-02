"""Project-root resolution for dataset and checkpoint paths."""

from pathlib import Path


def project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p
