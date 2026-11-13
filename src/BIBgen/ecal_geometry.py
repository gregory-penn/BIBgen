"""
Cell geometry of the MAIA ECAL barrel, for snapping positions to detector cells.

All numbers come from the MAIA_v0 compact files in k4geo
(``MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml`` and ``ECalBarrel_o2_v01_02.xml``).

Layout
------
The barrel is 12 flat staves ("modules") around the beam axis. Module ``m`` faces
azimuth ``-m * 30 deg``: its outward unit normal is ``(cos a, sin a, 0)`` with
``a = -m * 30 deg``. Each module holds 50 layers stacked along its normal, each
5.35 mm thick, starting 1857 mm from the beam axis. Each layer contains one silicon
plane 0.5 mm thick whose middle lies 2.70 mm from the inner face of the layer.

The silicon is divided into 5.1 x 5.1 mm cells. Cell ``(x, y)`` of a layer is
centered at::

    depth * normal + x * 5.1 mm * axis_x + y * 5.1 mm * (0, 0, 1)

where ``depth`` is the distance of the silicon middle from the beam axis and
``axis_x = (-sin a, cos a, 0)`` runs across the module. Cell ``(0, 0)`` is on the
module's center line at z = 0.

Snapping rule
-------------
A position is assigned to the nearest cell center. Within one module the centers
form a rectangular grid, so the nearest center is found by rounding each of the
three coordinates (depth, across, along z) to the nearest grid value separately.
Neighboring modules overlap at their edges, so the two neighbors of the nearest
module are also tried, and the closest cell center among the three modules wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
UIntArray = npt.NDArray[np.uint64]

COLLECTION = "ECalBarrelCollection"
CELL_ID_ENCODING = (
    "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16"
)
SYSTEM_ID = 20  # DetID_ECal_Barrel; the side, stave and submodule fields are always 0

N_MODULES = 12
MODULE_ANGLE = 2 * np.pi / N_MODULES  # 30 deg between neighboring module normals
N_LAYERS = 50
INNER_RADIUS = 1857.0  # mm, distance of the first layer's inner face from the beam axis
LAYER_THICKNESS = 5.35  # mm, sum of the slice thicknesses of one layer
# mm, from a layer's inner face to the middle of its silicon:
# tungsten 2.20 + G10 0.15 + ground/HV 0.10 + half the silicon 0.25.
SILICON_OFFSET = 2.70
HALF_LENGTH = 2307.0  # mm, half-length of the barrel along z
CELL_SIZE = 5.1  # mm, in both x and y

# Largest |y| cell index: cells whose centers lie within half a cell of the silicon end.
MAX_Y_INDEX = int(np.floor(HALF_LENGTH / CELL_SIZE + 0.5))  # 452


class SnapStatus(IntEnum):
    """Outcome of snapping one position to a cell. Anything but INSIDE is rejected."""

    INSIDE = 0
    BELOW_FIRST_LAYER = 1
    BEYOND_LAST_LAYER = 2
    BEYOND_HALF_LENGTH = 3
    BEYOND_STAVE_EDGE = 4


@dataclass(frozen=True)
class SnapResult:
    """Cell indices, status and distance to the cell center for each snapped position.

    Indices and distance are only meaningful where ``status == SnapStatus.INSIDE``.
    """

    module: IntArray
    layer: IntArray
    x: IntArray
    y: IntArray
    status: IntArray
    distance: FloatArray

    @property
    def inside(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of positions that were snapped to a valid cell."""
        inside: npt.NDArray[np.bool_] = self.status == SnapStatus.INSIDE
        return inside


def module_axes(module: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return the unit vectors ``(normal, axis_x)`` of each module, each of shape (..., 3).

    ``normal`` points away from the beam axis and ``axis_x`` across the module, in
    the direction of increasing x cell index. The y cell index increases along z.
    """
    a = -np.asarray(module, dtype=np.float64) * MODULE_ANGLE
    zero = np.zeros_like(a)
    normal = np.stack([np.cos(a), np.sin(a), zero], axis=-1)
    axis_x = np.stack([-np.sin(a), np.cos(a), zero], axis=-1)
    return normal, axis_x


def silicon_depth(layer: npt.ArrayLike) -> FloatArray:
    """Distance in mm of the middle of each layer's silicon from the beam axis (along the normal)."""
    depth: FloatArray = INNER_RADIUS + np.asarray(layer, dtype=np.float64) * LAYER_THICKNESS + SILICON_OFFSET
    return depth


def max_x_index(layer: npt.ArrayLike) -> IntArray:
    """Largest |x| cell index in each layer.

    A module's cross-section is a trapezoid whose sides make 15 deg (half of 30 deg)
    with the normal, so the silicon's half-width is (distance of the layer's inner
    face from the axis) * tan(15 deg). The outermost cells are only partly silicon
    (their centers lie up to half a cell outside it); Geant4 records hits in them,
    so they count as valid.
    """
    inner_face = INNER_RADIUS + np.asarray(layer, dtype=np.float64) * LAYER_THICKNESS
    half_width = inner_face * np.tan(MODULE_ANGLE / 2)
    index: IntArray = np.floor(half_width / CELL_SIZE + 0.5).astype(np.int64)
    return index


def cell_center(
    module: npt.ArrayLike, layer: npt.ArrayLike, x: npt.ArrayLike, y: npt.ArrayLike
) -> FloatArray:
    """Global (x, y, z) in mm of the center of each given cell.

    The result has the shape of the inputs plus a trailing axis of length 3:
    (n, 3) for arrays of length n, (3,) for scalars.
    """
    normal, axis_x = module_axes(module)
    depth = silicon_depth(layer)[..., None]
    across = (np.asarray(x, dtype=np.float64) * CELL_SIZE)[..., None]
    along_z = np.asarray(y, dtype=np.float64) * CELL_SIZE
    center: FloatArray = depth * normal + across * axis_x
    center[..., 2] = along_z
    return center


def _snap_in_module(p: FloatArray, module: IntArray) -> SnapResult:
    """Snap each position in ``p`` (n, 3) to the nearest cell of the given module (n,)."""
    normal, axis_x = module_axes(module)
    depth = np.einsum("nk,nk->n", p, normal)
    across = np.einsum("nk,nk->n", p, axis_x)
    along_z = p[:, 2]

    # Nearest silicon plane, nearest cell across the module, nearest cell along z.
    layer = np.floor((depth - silicon_depth(0)) / LAYER_THICKNESS + 0.5).astype(np.int64)
    x = np.floor(across / CELL_SIZE + 0.5).astype(np.int64)
    y = np.floor(along_z / CELL_SIZE + 0.5).astype(np.int64)

    # A position more than half a layer inside the first silicon plane or outside the
    # last one is outside the ECAL. Out-of-range layers are clipped only so that the
    # following lines have a valid layer to compute with.
    layer_clipped = np.clip(layer, 0, N_LAYERS - 1)
    # np.select takes the first condition that is true.
    status = np.select(
        [
            layer < 0,
            layer >= N_LAYERS,
            np.abs(y) > MAX_Y_INDEX,
            np.abs(x) > max_x_index(layer_clipped),
        ],
        [
            SnapStatus.BELOW_FIRST_LAYER,
            SnapStatus.BEYOND_LAST_LAYER,
            SnapStatus.BEYOND_HALF_LENGTH,
            SnapStatus.BEYOND_STAVE_EDGE,
        ],
        default=SnapStatus.INSIDE,
    ).astype(np.int64)
    distance = np.linalg.norm(p - cell_center(module, layer_clipped, x, y), axis=1)
    return SnapResult(module=module, layer=layer_clipped, x=x, y=y, status=status, distance=distance)


def snap(positions: npt.ArrayLike) -> SnapResult:
    """Assign each global position (n, 3), in mm, to the nearest ECAL barrel cell center.

    The positions must be finite. See the module docstring for the rule.
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(p)

    # The module whose normal points closest to the position's azimuth, then its two neighbors.
    azimuth = np.arctan2(p[:, 1], p[:, 0])
    nearest = np.round(-azimuth / MODULE_ANGLE).astype(np.int64) % N_MODULES
    trials = [_snap_in_module(p, (nearest + step) % N_MODULES) for step in (0, -1, 1)]

    # Keep the closest cell center among the modules where the position is inside.
    # If it is inside none of them, argmin returns 0, so the nearest module's
    # status is reported.
    distance = np.stack([np.where(t.inside, t.distance, np.inf) for t in trials])
    pick = np.argmin(distance, axis=0)
    rows = np.arange(n)

    def choose(name: str) -> npt.NDArray[Any]:
        chosen: npt.NDArray[Any] = np.stack([getattr(t, name) for t in trials])[pick, rows]
        return chosen

    return SnapResult(
        module=choose("module"),
        layer=choose("layer"),
        x=choose("x"),
        y=choose("y"),
        status=choose("status"),
        distance=choose("distance"),
    )


def encode_cell_id(
    module: npt.ArrayLike, layer: npt.ArrayLike, x: npt.ArrayLike, y: npt.ArrayLike
) -> UIntArray:
    """Pack cell indices into 64-bit cell IDs following ``CELL_ID_ENCODING``.

    Bits 0-4 system, 7-14 module, 19-27 layer, 32-47 x and 48-63 y (both as 16-bit
    two's complement). The side, stave and submodule bits stay 0.
    """

    def bits(v: npt.ArrayLike) -> UIntArray:
        # int64 -> uint64 keeps the two's-complement bit pattern of negative values.
        return np.asarray(v, dtype=np.int64).astype(np.uint64)

    low16 = np.uint64(0xFFFF)
    cell_id: UIntArray = (
        np.uint64(SYSTEM_ID)
        | (bits(module) << np.uint64(7))
        | (bits(layer) << np.uint64(19))
        | ((bits(x) & low16) << np.uint64(32))
        | ((bits(y) & low16) << np.uint64(48))
    )
    return cell_id


def decode_cell_id(cell_id: npt.ArrayLike) -> dict[str, IntArray]:
    """Unpack 64-bit cell IDs into ``{"system", "module", "layer", "x", "y"}``."""
    c = np.asarray(cell_id, dtype=np.uint64)

    def field(offset: int, width: int) -> IntArray:
        return ((c >> np.uint64(offset)) & np.uint64((1 << width) - 1)).astype(np.int64)

    def signed16(v: IntArray) -> IntArray:
        return np.where(v >= 1 << 15, v - (1 << 16), v)

    return {
        "system": field(0, 5),
        "module": field(7, 8),
        "layer": field(19, 9),
        "x": signed16(field(32, 16)),
        "y": signed16(field(48, 16)),
    }
