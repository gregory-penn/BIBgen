"""
Cell geometry of the MAIA ECAL barrel, for snapping positions to detector cells.

The geometry is read from a JSON table written by
``examples/conversion/extract_ecal_geometry.py``, which takes it from DD4hep.
This module needs only numpy, so the snapping runs without DD4hep.

Geometry model
--------------
The barrel has 12 flat staves ("modules") of 50 layers each. Every layer holds one
silicon box segmented into square cells by a ``CartesianGridXY`` with zero offset.
For a (module, layer), cell ``(x, y)`` is centered at::

    origin + x * cell_size_x * axis_x + y * cell_size_y * axis_y

``axis_x`` runs across the stave, ``axis_y`` along the beam axis, and
``normal = axis_x x axis_y`` points outward. A position belongs to cell index
``floor(u / cell_size + 1/2)`` along each axis, which is DD4hep's own rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
UIntArray = npt.NDArray[np.uint64]

DEFAULT_GEOMETRY_TABLE = (
    Path(__file__).resolve().parents[2] / "config" / "maia_v0_ecal_barrel_geometry.json"
)


class SnapStatus(IntEnum):
    """Outcome of snapping one position to a cell. Anything but INSIDE is rejected."""

    INSIDE = 0
    NON_FINITE = 1
    BELOW_FIRST_LAYER = 2
    BEYOND_LAST_LAYER = 3
    BEYOND_HALF_LENGTH = 4
    BEYOND_STAVE_EDGE = 5


@dataclass(frozen=True)
class SnapResult:
    """Cell indices and status for each snapped position.

    Indices are only meaningful where ``status == SnapStatus.INSIDE``.
    ``layer`` and ``module`` are the cell-ID field values, not array positions.
    """

    module: IntArray
    layer: IntArray
    x: IntArray
    y: IntArray
    status: IntArray

    @property
    def inside(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of positions that were snapped to a valid cell."""
        inside: npt.NDArray[np.bool_] = self.status == SnapStatus.INSIDE
        return inside


def parse_bitfield(encoding: str) -> dict[str, tuple[int, int, bool]]:
    """Parse a DD4hep bitfield description.

    Parameters
    ----------
    encoding : str
        For example ``"system:0:5,side:5:-2,module:7:8"``. A field is either
        ``name:width`` (placed right after the previous field) or
        ``name:offset:width``. A negative width marks a signed field.

    Returns
    -------
    dict[str, tuple[int, int, bool]]
        ``{name: (offset, width, signed)}``.

    Examples
    --------
    >>> parse_bitfield("system:0:5,side:5:-2,x:32:-16")
    {'system': (0, 5, False), 'side': (5, 2, True), 'x': (32, 16, True)}
    """
    fields: dict[str, tuple[int, int, bool]] = {}
    offset = 0
    for item in encoding.split(","):
        parts = item.strip().split(":")
        if len(parts) == 3:
            offset = int(parts[1])
            width = int(parts[2])
        elif len(parts) == 2:
            width = int(parts[1])
        else:
            msg = f"Cannot parse bitfield entry {item!r}"
            raise ValueError(msg)
        fields[parts[0]] = (offset, abs(width), width < 0)
        offset += abs(width)
    return fields


class ECalBarrelGeometry:
    """Snap positions to MAIA ECAL barrel cells and compute cell IDs and centers.

    Parameters
    ----------
    table : dict
        Contents of the JSON geometry table (see ``from_json``).
    """

    def __init__(self, table: dict[str, object]) -> None:
        self.collection = str(table["collection"])
        self.cell_id_encoding = str(table["cell_id_encoding"])
        self.bitfield = parse_bitfield(self.cell_id_encoding)
        fixed = table["fixed_fields"]
        assert isinstance(fixed, dict)
        self.fixed_fields: dict[str, int] = {str(k): int(v) for k, v in fixed.items()}
        self.cell_size_x = float(table["cell_size_x"])  # type: ignore[arg-type]
        self.cell_size_y = float(table["cell_size_y"])  # type: ignore[arg-type]
        self.modules = np.asarray(table["modules"], dtype=np.int64)
        self.layers = np.asarray(table["layers"], dtype=np.int64)
        self.origin = np.asarray(table["origin"], dtype=np.float64)
        self.axis_x = np.asarray(table["axis_x"], dtype=np.float64)
        self.axis_y = np.asarray(table["axis_y"], dtype=np.float64)
        self.normal = np.asarray(table["normal"], dtype=np.float64)
        self.half_x = np.asarray(table["half_x"], dtype=np.float64)
        self.half_y = np.asarray(table["half_y"], dtype=np.float64)

        n_mod, n_lay = len(self.modules), len(self.layers)
        if self.origin.shape != (n_mod, n_lay, 3):
            msg = f"origin has shape {self.origin.shape}, expected {(n_mod, n_lay, 3)}"
            raise ValueError(msg)
        # The snapping assumes all layers of a stave are parallel, which holds for
        # the MAIA barrel. Check it rather than silently mis-assigning layers.
        if not np.allclose(self.normal, self.normal[:, :1, :], atol=1e-9):
            msg = "Layers within a module are not parallel; this geometry is not supported"
            raise ValueError(msg)
        # Distance of each layer's silicon mid-plane from the beam axis, along the
        # stave normal. Must increase with layer for the nearest-layer search.
        self.layer_depth = np.einsum("mlk,mlk->ml", self.origin, self.normal)
        if not np.all(np.diff(self.layer_depth, axis=1) > 0):
            msg = "Layer depths do not increase with layer index"
            raise ValueError(msg)
        pitch_first = self.layer_depth[:, 1] - self.layer_depth[:, 0]
        pitch_last = self.layer_depth[:, -1] - self.layer_depth[:, -2]
        # A position is assigned to the nearest layer. Positions more than half a
        # layer pitch inside the first layer or outside the last layer are rejected;
        # these limits coincide with the inner and outer faces of the stave.
        self.depth_min = self.layer_depth[:, 0] - pitch_first / 2
        self.depth_max = self.layer_depth[:, -1] + pitch_last / 2
        self.layer_boundaries = (self.layer_depth[:, 1:] + self.layer_depth[:, :-1]) / 2

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_GEOMETRY_TABLE) -> ECalBarrelGeometry:
        """Load the geometry from a JSON table written by ``extract_ecal_geometry.py``."""
        with Path(path).open() as f:
            table = json.load(f)
        return cls(table)

    def snap(self, positions: npt.ArrayLike) -> SnapResult:
        """Assign each global position to the ECAL barrel cell that contains it.

        The module is the stave whose outward normal is closest in direction to the
        position (in the transverse plane). The layer is the one whose silicon
        mid-plane is nearest along that normal. The x and y indices follow DD4hep's
        ``CartesianGridXY`` rule within that layer.

        Parameters
        ----------
        positions : array_like, shape (n, 3)
            Global (x, y, z) in mm.

        Returns
        -------
        SnapResult
            Cell-ID field values and a ``SnapStatus`` code per position.
        """
        p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        n = len(p)
        finite = np.all(np.isfinite(p), axis=1)
        p = np.where(finite[:, None], p, 0.0)

        # Candidate modules: the staves whose outward normals are closest in
        # direction to the position, in the transverse plane. The silicon of one
        # stave extends past the angular midpoint to its neighbor, so the nearest
        # normal alone does not identify the stave; the neighbors are tried too.
        normals_t = self.normal[:, 0, :].copy()
        normals_t[:, 2] = 0.0
        normals_t /= np.linalg.norm(normals_t, axis=1, keepdims=True)
        p_t = p.copy()
        p_t[:, 2] = 0.0
        n_candidates = min(self._N_CANDIDATE_MODULES, len(self.modules))
        candidates = np.argsort(-(p_t @ normals_t.T), axis=1)[:, :n_candidates]

        best = None
        best_dist = np.full(n, np.inf)
        for k in range(n_candidates):
            trial = self._snap_in_module(p, candidates[:, k])
            dist = np.where(trial[0] == SnapStatus.INSIDE, trial[5], np.inf)
            if best is None:
                # The nearest-normal stave supplies the status when no stave contains the point.
                best = list(trial)
                best_dist = dist
                continue
            better = dist < best_dist
            for i in range(len(best)):
                best[i] = np.where(better, trial[i], best[i])
            best_dist = np.where(better, dist, best_dist)
        assert best is not None
        status, m_idx, l_idx, ix, iy, _ = best
        final_status: IntArray = np.asarray(
            np.where(finite, status, SnapStatus.NON_FINITE), dtype=np.int64
        )
        return SnapResult(
            module=self.modules[m_idx],
            layer=self.layers[l_idx],
            x=np.asarray(ix, dtype=np.int64),
            y=np.asarray(iy, dtype=np.int64),
            status=final_status,
        )

    # Number of staves (nearest in direction first) tried for each position.
    _N_CANDIDATE_MODULES = 3

    def _snap_in_module(
        self, p: FloatArray, m_idx: IntArray
    ) -> tuple[IntArray, IntArray, IntArray, IntArray, IntArray, FloatArray]:
        """Snap positions to cells of the given modules (array positions in the table).

        Returns status, module position, layer position, x index, y index and the
        distance from each position to its cell center.
        """
        n = len(p)
        status = np.full(n, SnapStatus.INSIDE, dtype=np.int64)
        depth = np.einsum("nk,nk->n", p, self.normal[m_idx, 0, :])
        below = depth < self.depth_min[m_idx]
        beyond = depth > self.depth_max[m_idx]
        # Nearest layer: count the layer boundaries below this depth.
        l_idx = np.sum(depth[:, None] > self.layer_boundaries[m_idx], axis=1)

        rel = p - self.origin[m_idx, l_idx]
        u = np.einsum("nk,nk->n", rel, self.axis_x[m_idx, l_idx])
        v = np.einsum("nk,nk->n", rel, self.axis_y[m_idx, l_idx])
        ix = np.floor(u / self.cell_size_x + 0.5).astype(np.int64)
        iy = np.floor(v / self.cell_size_y + 0.5).astype(np.int64)
        # A cell is valid if any part of it overlaps the silicon. The outermost
        # cells along x are only partly silicon (their centers lie outside it), and
        # Geant4 still records hits in them.
        max_ix = np.floor(self.half_x[m_idx, l_idx] / self.cell_size_x + 0.5)
        max_iy = np.floor(self.half_y[m_idx, l_idx] / self.cell_size_y + 0.5)
        out_length = np.abs(iy) > max_iy
        out_edge = np.abs(ix) > max_ix

        ok = np.ones(n, dtype=bool)
        for mask, code in (
            (below, SnapStatus.BELOW_FIRST_LAYER),
            (beyond, SnapStatus.BEYOND_LAST_LAYER),
            (out_length, SnapStatus.BEYOND_HALF_LENGTH),
            (out_edge, SnapStatus.BEYOND_STAVE_EDGE),
        ):
            status[ok & mask] = code
            ok &= ~mask

        center = (
            self.origin[m_idx, l_idx]
            + (ix * self.cell_size_x)[:, None] * self.axis_x[m_idx, l_idx]
            + (iy * self.cell_size_y)[:, None] * self.axis_y[m_idx, l_idx]
        )
        dist = np.linalg.norm(p - center, axis=1)
        return status, m_idx, l_idx, ix, iy, dist

    def _indices(self, module: npt.ArrayLike, layer: npt.ArrayLike) -> tuple[IntArray, IntArray]:
        """Convert module/layer field values to array positions in the table."""
        m = np.searchsorted(self.modules, np.asarray(module, dtype=np.int64))
        lay = np.searchsorted(self.layers, np.asarray(layer, dtype=np.int64))
        return m, lay

    def cell_center(
        self, module: npt.ArrayLike, layer: npt.ArrayLike, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> FloatArray:
        """Global (x, y, z) in mm of the center of each given cell.

        The result has the broadcast shape of the inputs plus a trailing axis of
        length 3: (n, 3) for arrays of length n, (3,) for scalars.
        """
        m, lay = self._indices(module, layer)
        ix = np.asarray(x, dtype=np.float64)[..., None]
        iy = np.asarray(y, dtype=np.float64)[..., None]
        center: FloatArray = (
            self.origin[m, lay]
            + ix * self.cell_size_x * self.axis_x[m, lay]
            + iy * self.cell_size_y * self.axis_y[m, lay]
        )
        return center

    def encode_cell_id(
        self, module: npt.ArrayLike, layer: npt.ArrayLike, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> UIntArray:
        """Pack cell-ID fields into 64-bit cell IDs, with the fixed fields filled in."""
        values: dict[str, IntArray] = {
            **{k: np.asarray(v, dtype=np.int64) for k, v in self.fixed_fields.items()},
            "module": np.asarray(module, dtype=np.int64),
            "layer": np.asarray(layer, dtype=np.int64),
            "x": np.asarray(x, dtype=np.int64),
            "y": np.asarray(y, dtype=np.int64),
        }
        shape = np.broadcast(*values.values()).shape
        cell_id = np.zeros(shape, dtype=np.uint64)
        for name, value in values.items():
            offset, width, signed = self.bitfield[name]
            v = np.broadcast_to(value, shape)
            if signed:
                limit = 1 << (width - 1)
                if np.any((v < -limit) | (v >= limit)):
                    msg = f"Field {name!r} out of range for {width} signed bits"
                    raise ValueError(msg)
            elif np.any((v < 0) | (v >= (1 << width))):
                msg = f"Field {name!r} out of range for {width} unsigned bits"
                raise ValueError(msg)
            # Two's complement in `width` bits, then shift into place.
            packed = (v.astype(np.uint64) & np.uint64((1 << width) - 1)) << np.uint64(offset)
            cell_id |= packed
        return cell_id

    def decode_cell_id(self, cell_id: npt.ArrayLike) -> dict[str, IntArray]:
        """Unpack 64-bit cell IDs into ``{field: values}``."""
        c = np.asarray(cell_id, dtype=np.uint64)
        fields: dict[str, IntArray] = {}
        for name, (offset, width, signed) in self.bitfield.items():
            raw = ((c >> np.uint64(offset)) & np.uint64((1 << width) - 1)).astype(np.int64)
            if signed:
                raw = np.where(raw >= (1 << (width - 1)), raw - (1 << width), raw)
            fields[name] = raw
        return fields
