from vedo import Plotter, Sphere, Box as VedoBox, Text3D

from typing import List, Any, Iterable, List, Optional, Tuple


from clp.clp.models.geometry import AABB, Point, Dims


palette = [
    'darkgreen', 'tomato', 'yellow', 'darkblue', 'darkviolet', 'indianred', 'yellowgreen', 'mediumblue', 'cyan',
    'black', 'indigo', 'pink', 'lime', 'sienna', 'plum', 'deepskyblue', 'forestgreen', 'fuchsia', 'brown',
    'turquoise', 'aliceblue', 'blueviolet', 'rosybrown', 'powderblue', 'lightblue', 'skyblue', 'lightskyblue',
    'steelblue', 'dodgerblue', 'lightslategray', 'slategray', 'lightsteelblue',
    'cornflowerblue', 'royalblue', 'lavender', 'midnightblue', 'navy', 'blue',
    'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'darkorchid',
    'darkviolet', 'mediumorchid'
]


def _as_xyz(p: Any) -> Tuple[float, float, float]:
    """Accept Point, tuple, or object with position()."""
    if hasattr(p, "position") and callable(getattr(p, "position")):
        return tuple(p.position())
    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return (float(p.x), float(p.y), float(p.z))
    # assume tuple/list
    return (float(p[0]), float(p[1]), float(p[2]))


def _draw_container(vp: Plotter, L: float, W: float, H: float):
    vp += VedoBox(pos=(L / 2, W / 2, H / 2), length=L, width=W, height=H).wireframe().alpha(0.5).c("gray")


def _draw_aabb(vp: Plotter, aabb: Any, color: str, alpha: float = 0.75, label: Optional[str] = None):
    """Draw an AABB-like object with origin + dims."""
    x0, y0, z0 = _as_xyz(aabb.origin) if hasattr(aabb, "origin") else _as_xyz(aabb[0])
    dims = aabb.dims if hasattr(aabb, "dims") else aabb[1]
    L, W, H = float(dims.L), float(dims.W), float(dims.H)

    vp += VedoBox(pos=(x0 + L/2, y0 + W/2, z0 + H/2), length=L, width=W, height=H).alpha(alpha).c(color)

    if label is not None:
        vp += Text3D(str(label), pos=(x0 + L/2, y0 + W/2, z0 + H + 4), s=8, c="black")


def plot_container_debug(
    placed: List[Any],
    eps: Optional[List[Any]] = None,
    removed_eps: Optional[List[Any]] = None,
    container_dims: Tuple[float, float, float] = (0, 0, 0),
    title: str = "CLP Debug View",
    show_box_labels: bool = True,
    show_ep_labels: bool = False,
    ep_radius: float = 4,
):
    """
    Visualizes:
      - Container wireframe
      - Placed objects: supports either:
          A) new AABB list: obj has .origin (Point) and .dims (Dims)
          B) old Block-like objects: obj.position + obj.get_bounds() + obj.boxes
      - EPs and removed EPs (Point / tuple / object with position())
    """
    L, W, H = container_dims

    axes_opts = dict(
        xtitle='Length (X)', ytitle='Width (Y)', ztitle='Height (Z)',
        xrange=(0, L), yrange=(0, W), zrange=(0, H),
        number_of_divisions=15,
        axes_linewidth=1.5, grid_linewidth=1,
        xygrid=True, zxgrid=True, yzgrid=True,
        xygrid2=True, zxgrid2=True, yzgrid2=True,
        xtitle_offset=0.5, ytitle_offset=0.04, ztitle_offset=0.04,
        xtitle_justify='bottom-center', ytitle_justify='bottom-center', ztitle_justify='bottom-center',
        xyplane_color='lightgray', xygrid_color='gray', xyalpha=0.1,
        show_ticks=True, label_font="Roboto", text_scale=1.2,
    )

    vp = Plotter(title=title, axes=axes_opts, bg="white")
    _draw_container(vp, L, W, H)

    # ---- Draw placed objects ----
    for idx, obj in enumerate(placed):
        color = palette[obj.type_id]

        # Case A: AABB-like
        if hasattr(obj, "origin") and hasattr(obj, "dims"):
            label = f"B{idx}" if show_box_labels else None
            _draw_aabb(vp, obj, color=color, alpha=0.70, label=label)
            continue

        # Case B: old Block-like
        if hasattr(obj, "get_bounds") and hasattr(obj, "position"):
            lx, wy, zh = obj.get_bounds()
            x0, y0, z0 = obj.position

            # label customer if available
            if show_box_labels and hasattr(obj, "boxes") and obj.boxes:
                tag = getattr(obj.boxes[0], "customer_tag", None)
                if tag is not None:
                    vp += Text3D(str(tag), pos=(x0 + lx/2, y0 + wy/2, z0 + zh + 5), s=3.5, c="black")

            # draw each inner box if exists, else draw as one block
            if hasattr(obj, "boxes") and obj.boxes:
                for b in obj.boxes:
                    l, w, h = b.oriented_dimensions()
                    x, y, z = b.position
                    vp += VedoBox(pos=(x + l/2, y + w/2, z + h/2),
                                  length=l, width=w, height=h).alpha(0.8).c(color)
            else:
                vp += VedoBox(pos=(x0 + lx/2, y0 + wy/2, z0 + zh/2),
                              length=lx, width=wy, height=zh).alpha(0.8).c(color)
            continue

        # Fallback: ignore unknown object
        # (better than crashing mid-debug)
        continue

    # ---- Draw active EPs ----
    if eps:
        for i, p in enumerate(eps):
            pos = _as_xyz(p)
            vp += Sphere(pos=pos, r=ep_radius, c='red')
            if show_ep_labels:
                vp += Text3D(f"EP{i}", pos=(pos[0], pos[1], pos[2] + 3), s=2, c="black")

    # ---- Draw removed EPs ----
    if removed_eps:
        for i, p in enumerate(removed_eps):
            pos = _as_xyz(p)
            vp += Sphere(pos=pos, r=ep_radius + 1.0, c='lavender')
            if show_ep_labels:
                vp += Text3D(f"X{i}", pos=(pos[0], pos[1], pos[2] + 3), s=10, c="gray")

    vp.show(interactive=True)



# def draw_container(container: Dims):
#     # wireframe container
#     return Box(
#         pos=(container.L/2, container.W/2, container.H/2),
#         length=container.L,
#         width=container.W,
#         height=container.H,
#     ).wireframe().c("black")


# def draw_boxes(boxes: List[AABB], color="lightblue"):
#     actors = []
#     for i, b in enumerate(boxes):
#         x0, y0, z0 = b.origin.x, b.origin.y, b.origin.z
#         L, W, H = b.dims.L, b.dims.W, b.dims.H

#         box = Box(
#             pos=(x0 + L/2, y0 + W/2, z0 + H/2),
#             length=L,
#             width=W,
#             height=H,
#         ).alpha(0.6).c(color)

#         actors.append(box)
#     return actors


# def draw_eps(eps: List[Point], r=6, color="red"):
#     pts = [(p.x, p.y, p.z) for p in eps]
#     return Points(pts, r=r).c(color)


# def visualize(container: Dims, boxes: List[AABB], eps: List[Point], title="CLP Debug"):
#     actors = []
#     actors.append(draw_container(container))
#     actors.extend(draw_boxes(boxes))
#     actors.append(draw_eps(eps))

#     txt = Text2D(title, pos="top-center")

#     show(*actors, txt, axes=1, viewup="z",interactive=True)
