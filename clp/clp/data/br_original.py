# # clp\clp\data\br_original.py
# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional


# class BRLoaderError(RuntimeError):
#     pass


# @dataclass(frozen=True)
# class BRItemRow:
#     length: int
#     width: int   # JSON "Depth"
#     height: int
#     demand: int
#     value: int

#     c1_length: int
#     c1_width: int   # JSON "C1_Depth"
#     c1_height: int


# @dataclass(frozen=True)
# class BRInstance:
#     instance_name: str  # "BR0", "BR1", ...
#     case_id: int        # 1..100
#     items: List[BRItemRow]
#     source_path: Path


# def _as_int(row: Dict[str, Any], key: str, path: Path) -> int:
#     if key not in row:
#         raise BRLoaderError(f"Missing '{key}' in {path}")
#     v = row[key]
#     if not isinstance(v, int):
#         raise BRLoaderError(f"'{key}' must be int, got {type(v).__name__} in {path}")
#     return v


# def load_br_instance(dataset_root: str | Path, instance_name: str, case_id: int) -> BRInstance:
#     """
#     Loads: <dataset_root>/<instance_name>/<case_id>.json
#     Example: .../clp/clp/datasets/br_original/BR0/1.json
#     """
#     dataset_root = Path(dataset_root)
#     src = dataset_root / instance_name / f"{case_id}.json"
#     if not src.exists():
#         raise BRLoaderError(f"File not found: {src}")

#     try:
#         data = json.loads(src.read_text(encoding="utf-8"))
#     except Exception as e:
#         raise BRLoaderError(f"Invalid JSON: {src} ({e})") from e

#     if not isinstance(data, dict) or "Items" not in data or not isinstance(data["Items"], list):
#         raise BRLoaderError(f"Expected top-level dict with 'Items' list in {src}")

#     out: List[BRItemRow] = []
#     for i, row in enumerate(data["Items"]):
#         if not isinstance(row, dict):
#             raise BRLoaderError(f"Item #{i} is not an object in {src}")

#         L = _as_int(row, "Length", src)
#         W = _as_int(row, "Depth", src)   # Depth == Width
#         H = _as_int(row, "Height", src)
#         D = _as_int(row, "Demand", src)

#         value = row.get("Value")
#         if value is None:
#             value = L * W * H
#         if not isinstance(value, int):
#             raise BRLoaderError(f"'Value' must be int if present (item #{i}) in {src}")

#         c1_len = int(row.get("C1_Length", 1) or 0)
#         c1_wid = int(row.get("C1_Depth", 1) or 0)
#         c1_hgt = int(row.get("C1_Height", 1) or 0)

#         if L <= 0 or W <= 0 or H <= 0:
#             raise BRLoaderError(f"Non-positive dims in item #{i} in {src}")
#         if D <= 0:
#             raise BRLoaderError(f"Non-positive demand in item #{i} in {src}")

#         out.append(
#             BRItemRow(
#                 length=L, width=W, height=H, demand=D, value=value,
#                 c1_length=c1_len, c1_width=c1_wid, c1_height=c1_hgt
#             )
#         )

#     return BRInstance(instance_name=instance_name, case_id=case_id, items=out, source_path=src)


# clp\clp\data\br_original.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class BRLoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class BRItemRow:
    # --- original BR fields ---
    length: int
    width: int   # JSON "Depth"
    height: int
    demand: int
    value: int

    c1_length: int
    c1_width: int   # JSON "C1_Depth"
    c1_height: int

    # --- injected fields (BR-Modified) ---
    customer_id: Optional[int] = None
    priority: Optional[int] = None
    stackable: Optional[bool] = None
    alpha: Optional[int] = None
    beta: Optional[int] = None
    density: Optional[float] = None   # kg/m^3
    weight: Optional[float] = None    # kg per box instance


@dataclass(frozen=True)
class BRInstance:
    instance_name: str  # "BR0", "BR1", ...
    case_id: int        # 1..100
    items: List[BRItemRow]
    source_path: Path

    # Optional root metadata (present in BR-Modified)
    modification: Optional[Dict[str, Any]] = None


def _as_int(row: Dict[str, Any], key: str, path: Path) -> int:
    if key not in row:
        raise BRLoaderError(f"Missing '{key}' in {path}")
    v = row[key]
    if not isinstance(v, int):
        raise BRLoaderError(f"'{key}' must be int, got {type(v).__name__} in {path}")
    return v


def _as_opt_int(row: Dict[str, Any], key: str) -> Optional[int]:
    v = row.get(key, None)
    if v is None:
        return None
    if isinstance(v, bool):
        # avoid True/False being treated as ints
        return int(v)
    if isinstance(v, int):
        return v
    # allow numeric strings if you ever end up with them
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _as_opt_float(row: Dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key, None)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def _as_opt_bool01(row: Dict[str, Any], key: str) -> Optional[bool]:
    """
    Accepts:
      - 0/1
      - true/false
      - "0"/"1"
    """
    v = row.get(key, None)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes"):
            return True
        if s in ("0", "false", "no"):
            return False
    return None


def load_br_instance(dataset_root: str | Path, instance_name: str, case_id: int) -> BRInstance:
    """
    Loads: <dataset_root>/<instance_name>/<case_id>.json
    Works for BR-Original and BR-Modified (extra fields are optional).
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
        W = _as_int(row, "Depth", src)   # Depth == Width (Y)
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

        # --- optional modified fields ---
        customer_id = _as_opt_int(row, "CustomerId")
        priority = _as_opt_int(row, "Priority")
        stackable = _as_opt_bool01(row, "Stackable")
        alpha = _as_opt_int(row, "Alpha")
        beta = _as_opt_int(row, "Beta")
        density = _as_opt_float(row, "Density")
        weight = _as_opt_float(row, "Weight")

        out.append(
            BRItemRow(
                length=L, width=W, height=H, demand=D, value=value,
                c1_length=c1_len, c1_width=c1_wid, c1_height=c1_hgt,
                customer_id=customer_id,
                priority=priority,
                stackable=stackable,
                alpha=alpha,
                beta=beta,
                density=density,
                weight=weight,
            )
        )

    modification = data.get("Modification")
    if modification is not None and not isinstance(modification, dict):
        modification = None

    return BRInstance(
        instance_name=instance_name,
        case_id=case_id,
        items=out,
        source_path=src,
        modification=modification,
    )
