from __future__ import annotations

import numpy as np
import pytest

from BIBgen.ecal_geometry import (
    CELL_SIZE,
    CRACK_HALF_WIDTH,
    LAYER_THICKNESS,
    MODULE_ANGLE,
    N_MODULES,
    SnapStatus,
    _snap_in_module,
    cell_center,
    crack_reach,
    decode_cell_id,
    encode_cell_id,
    module_axes,
    snap,
)

# (cellID, cell-center position in mm) of real hits in a ddsim BIB file
# (BIB10TeV/sim_mp/BIB_sim_1000.edm4hep.root). The last six are cells at the stave
# edges (|x| >= 99), including cells whose stave is not the one with the nearest
# outward normal.
REAL_HITS = [
    (46443332509369108, (-1929.25, 45.9, 836.4)),
    (45035970514387732, (-1966.7, 30.6, 810.9)),
    (45035966216274708, (-1934.6, 35.7, 810.9)),
    (18418314748884419220, (-1907.679, -618.5032, -520.2)),
    (18420847924894761620, (-2003.3948, -538.3184, -474.3)),
    (18363990894438778260, (1562.176, 1537.9318, -1499.4)),
    (18415499900345124116, (1505.9565, 1537.3933, -571.2)),
    (18443085349996987156, (-1993.45, -535.5, -66.3)),
    (24207273202550036, (511.3188, 1895.43, 438.6)),
    (18339502532791894932, (-1927.8628, 530.04376, -1943.1)),
]


def test_real_cell_ids_are_reproduced():
    cell_ids = np.array([c for c, _ in REAL_HITS], dtype=np.uint64)
    positions = np.array([p for _, p in REAL_HITS])
    snapped = snap(positions)
    assert np.all(snapped.inside)
    np.testing.assert_array_equal(
        encode_cell_id(snapped.module, snapped.layer, snapped.x, snapped.y), cell_ids
    )
    centers = cell_center(snapped.module, snapped.layer, snapped.x, snapped.y)
    np.testing.assert_allclose(centers, positions, atol=1e-3)


def test_encode_decode_round_trip():
    module = np.array([0, 5, 11, 7])
    layer = np.array([0, 13, 49, 25])
    x = np.array([0, -108, 97, -1])
    y = np.array([0, 452, -452, -1])
    fields = decode_cell_id(encode_cell_id(module, layer, x, y))
    np.testing.assert_array_equal(fields["module"], module)
    np.testing.assert_array_equal(fields["layer"], layer)
    np.testing.assert_array_equal(fields["x"], x)
    np.testing.assert_array_equal(fields["y"], y)
    np.testing.assert_array_equal(fields["system"], 20)


def test_points_inside_a_cell_snap_to_it():
    # Cell (module 2, layer 10, x 3, y -7): move around within the cell, away from stave edges.
    center = cell_center(2, 10, 3, -7)
    normal, axis_x = module_axes(2)
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-0.49, 0.49, size=(1000, 3))
    points = (
        center
        + offsets[:, :1] * CELL_SIZE * axis_x
        + offsets[:, 1:2] * CELL_SIZE * np.array([0.0, 0.0, 1.0])
        + offsets[:, 2:] * LAYER_THICKNESS * normal
    )
    snapped = snap(points)
    assert np.all(snapped.inside)
    assert np.all(snapped.module == 2)
    assert np.all(snapped.layer == 10)
    assert np.all(snapped.x == 3)
    assert np.all(snapped.y == -7)


@pytest.mark.parametrize(
    ("position", "status"),
    [
        ((1800.0, 0.0, 0.0), SnapStatus.BELOW_FIRST_LAYER),
        ((2200.0, 0.0, 0.0), SnapStatus.BEYOND_LAST_LAYER),
        ((2000.0, 0.0, 2400.0), SnapStatus.BEYOND_HALF_LENGTH),
        # Between two staves (phi = 45 deg), outside the silicon of both.
        ((2120.0 * np.cos(np.pi / 4), 2120.0 * np.sin(np.pi / 4), 0.0), SnapStatus.BEYOND_STAVE_EDGE),
        ((2000.0, 0.0, 0.0), SnapStatus.INSIDE),
    ],
)
def test_snap_status(position, status):
    assert snap([position]).status[0] == status


def test_crack_window_covers_the_reach():
    assert crack_reach() == pytest.approx(4.92, abs=0.01)
    assert crack_reach() <= CRACK_HALF_WIDTH


def test_crack_window_matches_trying_all_neighbors():
    # Points within 8 mm of a sector boundary, over the full depth and length.
    rng = np.random.default_rng(1)
    n = 200_000
    radius = rng.uniform(1850, 2130, n)
    boundary = np.radians(15 + 30 * rng.integers(0, 12, n))
    azimuth = boundary + rng.uniform(-8, 8, n) / radius
    p = np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), rng.uniform(-2300, 2300, n)], axis=1)

    # Reference: try the sector module and both its neighbors; keep the nearest valid cell.
    sector = np.round(-azimuth / MODULE_ANGLE).astype(np.int64) % N_MODULES
    trials = [_snap_in_module(p, (sector + step) % N_MODULES) for step in (0, -1, 1)]
    distance = np.stack([np.where(t.inside, t.distance, np.inf) for t in trials])
    pick = np.argmin(distance, axis=0)
    rows = np.arange(n)

    result = snap(p)
    for name in ("module", "layer", "x", "y", "status"):
        expected = np.stack([getattr(t, name) for t in trials])[pick, rows]
        np.testing.assert_array_equal(getattr(result, name), expected, err_msg=name)
