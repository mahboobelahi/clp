# clp/io/br_loader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class BRLoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class BRItemRow:
    length: int
    width: int   # JSON "Depth"
    height: int
    demand: int
    value: int

    c1_length: int
    c1_width: int   # JSON "C1_Depth"
    c1_height: int


@dataclass(frozen=True)
class BRInstance:
    instance_name: str  # "BR0", "BR1", ...
    case_id: int        # 1..100
    items: List[BRItemRow]
    source_path: Path


def _as_int(row: Dict[str, Any], key: str, path: Path) -> int:
    if key not in row:
        raise BRLoaderError(f"Missing '{key}' in {path}")
    v = row[key]
    if not isinstance(v, int):
        raise BRLoaderError(f"'{key}' must be int, got {type(v).__name__} in {path}")
    return v


def load_br_instance(dataset_root: str | Path, instance_name: str, case_id: int) -> BRInstance:
    """
    Loads: <dataset_root>/<instance_name>/<case_id>.json
    Example: .../clp/clp/datasets/br_original/BR0/1.json
    """
    dataset_root = Path(dataset_root)
    src = dataset_root / instance_name / f"{case_id}.json"
    if not src.exists():
        raise BRLoaderError(f"File not found: {src}")

    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:
        raise BRLoaderError(f"Invalid JSON: {src} ({e})") from e

    if not isinstance(data, dict) or "Items" not in data or not isinstance(data["Items"], list):
        raise BRLoaderError(f"Expected top-level dict with 'Items' list in {src}")

    out: List[BRItemRow] = []
    for i, row in enumerate(data["Items"]):
        if not isinstance(row, dict):
            raise BRLoaderError(f"Item #{i} is not an object in {src}")

        L = _as_int(row, "Length", src)
        W = _as_int(row, "Depth", src)   # Depth == Width
        H = _as_int(row, "Height", src)
        D = _as_int(row, "Demand", src)

        value = row.get("Value")
        if value is None:
            value = L * W * H
        if not isinstance(value, int):
            raise BRLoaderError(f"'Value' must be int if present (item #{i}) in {src}")

        c1_len = int(row.get("C1_Length", 1) or 0)
        c1_wid = int(row.get("C1_Depth", 1) or 0)
        c1_hgt = int(row.get("C1_Height", 1) or 0)

        if L <= 0 or W <= 0 or H <= 0:
            raise BRLoaderError(f"Non-positive dims in item #{i} in {src}")
        if D <= 0:
            raise BRLoaderError(f"Non-positive demand in item #{i} in {src}")

        out.append(
            BRItemRow(
                length=L, width=W, height=H, demand=D, value=value,
                c1_length=c1_len, c1_width=c1_wid, c1_height=c1_hgt
            )
        )

    return BRInstance(instance_name=instance_name, case_id=case_id, items=out, source_path=src)
