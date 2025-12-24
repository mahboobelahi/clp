# clp/clp/results/index.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .writer import read_json, write_json


def ensure_index(path: Path) -> None:
    if not path.exists():
        write_json([], path)
        return
    txt = path.read_text(encoding="utf-8").strip()
    if txt == "":
        write_json([], path)


def append_run(index_path: Path, entry: Dict[str, Any]) -> None:
    ensure_index(index_path)
    data: List[Dict[str, Any]] = read_json(index_path)
    data.append(entry)
    write_json(data, index_path)
