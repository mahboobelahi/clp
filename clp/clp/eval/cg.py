from dataclasses import dataclass
from typing import Callable, List, Optional

from clp.clp.models.geometry import AABB, Dims


@dataclass(frozen=True)
class CGMetrics:
    total_mass: float

    # Loaded cargo CG
    cg_x: float
    cg_y: float
    cg_z: float

    # Container geometric center
    container_cg_x: float
    container_cg_y: float
    container_cg_z: float

    # Absolute deviations (distance units)
    dev_x: float
    dev_y: float
    dev_z: float

    # NSGA-II objective: average deviation across {x,y,z}
    z3: float

    # Moment-based RD (%)
    rd_x_pct: float
    rd_y_pct: float
    rd_z_pct: float

    def __repr__(self) -> str:
        return (
            "CGMetrics("
            f"total_mass={self.total_mass:.4f}, "
            f"cg=({self.cg_x:.4f},{self.cg_y:.4f},{self.cg_z:.4f}), "
            f"container_cg=({self.container_cg_x:.4f},{self.container_cg_y:.4f},{self.container_cg_z:.4f}), "
            f"dev=({self.dev_x:.4f},{self.dev_y:.4f},{self.dev_z:.4f}), "
            f"z3={self.z3:.4f}, "
            f"rd_pct=({self.rd_x_pct:.4f},{self.rd_y_pct:.4f},{self.rd_z_pct:.4f})"
            ")"
        )

def compute_cg_metrics(
    placed: List[AABB],
    container: Dims,
    mass_fn: Optional[Callable[[AABB], float]] = None,
) -> CGMetrics:
    if mass_fn is None:
        mass_fn = lambda b: float(b.dims.L * b.dims.W * b.dims.H)

    # container geometric center
    xc = container.L / 2.0
    yc = container.W / 2.0
    zc = container.H / 2.0

    if not placed:
        return CGMetrics(
            total_mass=0.0,
            cg_x=0.0, cg_y=0.0, cg_z=0.0,
            container_cg_x=xc, container_cg_y=yc, container_cg_z=zc,
            dev_x=0.0, dev_y=0.0, dev_z=0.0,
            z3=0.0,
            rd_x_pct=0.0, rd_y_pct=0.0, rd_z_pct=0.0,
        )

    total_m = 0.0
    sum_mx = 0.0
    sum_my = 0.0
    sum_mz = 0.0

    moment_x = 0.0
    moment_y = 0.0
    moment_z = 0.0

    for b in placed:
        m = mass_fn(b)
        x = b.origin.x + b.dims.L / 2.0
        y = b.origin.y + b.dims.W / 2.0
        z = b.origin.z + b.dims.H / 2.0

        total_m += m
        sum_mx += m * x
        sum_my += m * y
        sum_mz += m * z

        moment_x += m * (x - xc)
        moment_y += m * (y - yc)
        moment_z += m * (z - zc)

    cg_x = sum_mx / total_m
    cg_y = sum_my / total_m
    cg_z = sum_mz / total_m

    # absolute deviations (distance units)
    dev_x = round(abs(cg_x - xc), 4)
    dev_y = round(abs(cg_y - yc), 4)
    dev_z = round(abs(cg_z - zc), 4)

    # Z3 objective per MIP: average of deviations across {x,y,z}
    z3 = round((dev_x + dev_y + dev_z) / 3.0, 4)

    # RD% (moment normalized)
    rd_x = round(
        (abs(moment_x) / (total_m * (container.L / 2.0))) * 100.0
        if container.L > 0 else 0.0,
        4
    )
    rd_y = round(
        (abs(moment_y) / (total_m * (container.W / 2.0))) * 100.0
        if container.W > 0 else 0.0,
        4
    )
    rd_z = round(
        (abs(moment_z) / (total_m * (container.H / 2.0))) * 100.0
        if container.H > 0 else 0.0,
        4
    )

    return CGMetrics(
        total_mass=round(total_m, 4),
        cg_x=round(cg_x, 4),
        cg_y=round(cg_y, 4),
        cg_z=round(cg_z, 4),
        container_cg_x=round(xc, 4),
        container_cg_y=round(yc, 4),
        container_cg_z=round(zc, 4),
        dev_x=dev_x,
        dev_y=dev_y,
        dev_z=dev_z,
        z3=z3,
        rd_x_pct=rd_x,
        rd_y_pct=rd_y,
        rd_z_pct=rd_z,
    )
