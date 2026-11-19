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
A position is assigned to the cell whose center is nearest, where "center" is the
position ddsim writes for a hit in that cell. This reproduces the Geant4 cell IDs
of real hits from their stored positions.

Within one module the centers form a rectangular grid, so the nearest center is
found by rounding each of the three coordinates (depth, across, along z) to the
nearest grid value separately.

When the hit is near the boundary between two modules, the snapping is modified
to follow GEANT's behavior. GEANT assigns the hits to an index corresponding to 
the center of a cube. These cubes can overlap between modules, so special handling
is required to treat which hits belong to which cubes in his scenario. 
``crack_reach`` computes how far past a boundary a module's
cells can be nearest; ``CRACK_HALF_WIDTH`` is that distance rounded up.

"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

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

# mm, half-width of the crack window around each sector boundary. It must be at least
# crack_reach() (4.92 mm); tests check this.
CRACK_HALF_WIDTH = 5.0


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


def module_axes(module: IntArray) -> tuple[FloatArray, FloatArray]:
    """Return the unit vectors ``(normal, axis_x)`` for module numbers (n,), each of shape (n, 3).

    ``normal`` points away from the beam axis and ``axis_x`` across the module, in
    the direction of increasing x cell index. The y cell index increases along z.
    """
    a = -module * MODULE_ANGLE
    zero = np.zeros_like(a)
    normal = np.stack([np.cos(a), np.sin(a), zero], axis=1)
    axis_x = np.stack([-np.sin(a), np.cos(a), zero], axis=1)
    return normal, axis_x


def silicon_depth(layer: IntArray) -> FloatArray:
    """Distance in mm of the middle of the silicon of layers (n,) from the beam axis (along the normal)."""
    depth: FloatArray = INNER_RADIUS + layer * LAYER_THICKNESS + SILICON_OFFSET
    return depth


def max_x_index(layer: IntArray) -> IntArray:
    """Largest |x| cell index for layers (n,).

    A module's cross-section is a trapezoid whose sides make 15 deg (half of 30 deg)
    with the normal, so the silicon's half-width is (distance of the layer's inner
    face from the axis) * tan(15 deg). The outermost cells are only partly silicon
    (their centers lie up to half a cell outside it); Geant4 records hits in them,
    so they count as valid.
    """
    inner_face = INNER_RADIUS + layer * LAYER_THICKNESS
    half_width = inner_face * np.tan(MODULE_ANGLE / 2)
    index: IntArray = np.floor(half_width / CELL_SIZE + 0.5).astype(np.int64)
    return index


def cell_center(module: IntArray, layer: IntArray, x: IntArray, y: IntArray) -> FloatArray:
    """Global (x, y, z) in mm of the centers of cells given by four index arrays (n,); shape (n, 3)."""
    normal, axis_x = module_axes(module)
    depth = silicon_depth(layer)[:, None]
    across = (x * CELL_SIZE)[:, None]
    center: FloatArray = depth * normal + across * axis_x
    center[:, 2] = y * CELL_SIZE
    return center


def _snap_in_module(p: FloatArray, module: IntArray) -> SnapResult:
    """Snap each position in ``p`` (n, 3) to the nearest cell of the given module (n,)."""
    normal, axis_x = module_axes(module)
    depth = np.einsum("nk,nk->n", p, normal)
    across = np.einsum("nk,nk->n", p, axis_x)
    along_z = p[:, 2]

    # Nearest silicon plane, nearest cell across the module, nearest cell along z.
    layer = np.floor((depth - (INNER_RADIUS + SILICON_OFFSET)) / LAYER_THICKNESS + 0.5).astype(np.int64)
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


def snap(p: FloatArray) -> SnapResult:
    """Assign each global position (n, 3), in mm, to the nearest ECAL barrel cell center.

    The positions must be finite. See the module docstring for the rule.
    """

    # Sector module: module m faces azimuth -m * 30 deg, so -azimuth / 30 deg rounded
    # to a whole number is the module whose sector contains the position.
    steps = -np.arctan2(p[:, 1], p[:, 0]) / MODULE_ANGLE
    sector = np.round(steps)
    module = sector.astype(np.int64) % N_MODULES
    result = _snap_in_module(p, module)

    # Crack window. `offset` is the position's azimuth relative to its module's normal,
    # in units of 30 deg: -0.5 and +0.5 are the two sector boundaries.
    offset = steps - sector
    radius = np.hypot(p[:, 0], p[:, 1])
    distance_to_boundary = radius * np.sin((0.5 - np.abs(offset)) * MODULE_ANGLE)
    in_crack = np.flatnonzero(distance_to_boundary < CRACK_HALF_WIDTH)
    if len(in_crack) == 0:
        return result

    # The neighbor across the nearer boundary: the next module when offset > 0.
    neighbor = (module[in_crack] + np.where(offset[in_crack] > 0, 1, -1)) % N_MODULES
    other = _snap_in_module(p[in_crack], neighbor)
    current_ok = result.inside[in_crack]
    use_other = other.inside & (~current_ok | (other.distance < result.distance[in_crack]))

    values = {f.name: getattr(result, f.name).copy() for f in fields(SnapResult)}
    for name, array in values.items():
        array[in_crack[use_other]] = getattr(other, name)[use_other]
    return SnapResult(**values)


def crack_reach() -> float:
    """Largest distance in mm by which a module's cells can be nearest past its sector boundary.

    The points nearest to a cell's center lie within half a cell across and half a
    layer in depth of it. This returns how far the corners of those boxes, for the
    outermost cell of every layer, extend past the sector boundary. By symmetry,
    module 0 and its boundary at +15 deg cover every module and boundary.
    """
    layers = np.arange(N_LAYERS)
    module = np.zeros(N_LAYERS, dtype=np.int64)
    normal, axis_x = module_axes(module)
    boundary = MODULE_ANGLE / 2
    # Unit vector perpendicular to the boundary, pointing out of module 0's sector.
    outward = np.array([-np.sin(boundary), np.cos(boundary), 0.0])
    centers = cell_center(module, layers, max_x_index(layers), np.zeros(N_LAYERS, dtype=np.int64))
    reach = -np.inf
    for across in (-0.5, 0.5):
        for depth in (-0.5, 0.5):
            corner = centers + across * CELL_SIZE * axis_x + depth * LAYER_THICKNESS * normal
            reach = max(reach, float(np.max(corner @ outward)))
    return reach


def encode_cell_id(module: IntArray, layer: IntArray, x: IntArray, y: IntArray) -> UIntArray:
    """Pack cell indices into 64-bit cell IDs following ``CELL_ID_ENCODING``.

    Bits 0-4 system, 7-14 module, 19-27 layer, 32-47 x and 48-63 y (both as 16-bit
    two's complement). The side, stave and submodule bits stay 0.
    """

    def bits(v: IntArray) -> UIntArray:
        # int64 -> uint64 keeps the two's-complement bit pattern of negative values.
        return v.astype(np.uint64)

    low16 = np.uint64(0xFFFF)
    cell_id: UIntArray = (
        np.uint64(SYSTEM_ID)
        | (bits(module) << np.uint64(7))
        | (bits(layer) << np.uint64(19))
        | ((bits(x) & low16) << np.uint64(32))
        | ((bits(y) & low16) << np.uint64(48))
    )
    return cell_id


def decode_cell_id(c: UIntArray) -> dict[str, IntArray]:
    """Unpack 64-bit cell IDs (n,) into ``{"system", "module", "layer", "x", "y"}``, each (n,)."""

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
