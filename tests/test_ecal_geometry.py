from __future__ import annotations

import numpy as np
import pytest

from BIBgen.ecal_geometry import ECalBarrelGeometry, SnapStatus, parse_bitfield

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


@pytest.fixture(scope="module")
def geometry():
    return ECalBarrelGeometry.from_json()


def test_parse_bitfield():
    fields = parse_bitfield("system:0:5,side:5:-2,module:7:8,x:32:-16,y:-16")
    assert fields == {
        "system": (0, 5, False),
        "side": (5, 2, True),
        "module": (7, 8, False),
        "x": (32, 16, True),
        "y": (48, 16, True),
    }


def test_table_matches_maia(geometry):
    assert geometry.collection == "ECalBarrelCollection"
    assert geometry.cell_id_encoding == (
        "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16"
    )
    assert len(geometry.modules) == 12
    assert len(geometry.layers) == 50
    assert geometry.cell_size_x == pytest.approx(5.1)
    assert geometry.fixed_fields["system"] == 20


def test_real_cell_ids_are_reproduced(geometry):
    cell_ids = np.array([c for c, _ in REAL_HITS], dtype=np.uint64)
    positions = np.array([p for _, p in REAL_HITS])
    snap = geometry.snap(positions)
    assert np.all(snap.inside)
    np.testing.assert_array_equal(geometry.encode_cell_id(snap.module, snap.layer, snap.x, snap.y), cell_ids)
    centers = geometry.cell_center(snap.module, snap.layer, snap.x, snap.y)
    np.testing.assert_allclose(centers, positions, atol=1e-3)


def test_encode_decode_round_trip(geometry):
    module = np.array([0, 5, 11, 7])
    layer = np.array([0, 13, 49, 25])
    x = np.array([0, -108, 97, -1])
    y = np.array([0, 452, -452, -1])
    fields = geometry.decode_cell_id(geometry.encode_cell_id(module, layer, x, y))
    np.testing.assert_array_equal(fields["module"], module)
    np.testing.assert_array_equal(fields["layer"], layer)
    np.testing.assert_array_equal(fields["x"], x)
    np.testing.assert_array_equal(fields["y"], y)
    np.testing.assert_array_equal(fields["system"], 20)


def test_encode_rejects_out_of_range(geometry):
    with pytest.raises(ValueError, match="out of range"):
        geometry.encode_cell_id(0, 0, 40000, 0)


def test_points_inside_a_cell_snap_to_it(geometry):
    # Cell (module 2, layer 10, x 3, y -7): move around within the cell, away from stave edges.
    m, lay = 2, 10
    center = geometry.cell_center(m, lay, 3, -7)
    i = np.searchsorted(geometry.modules, m)
    j = np.searchsorted(geometry.layers, lay)
    pitch = geometry.layer_depth[i, j + 1] - geometry.layer_depth[i, j]
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-0.49, 0.49, size=(1000, 3))
    points = (
        center
        + offsets[:, :1] * geometry.cell_size_x * geometry.axis_x[i, j]
        + offsets[:, 1:2] * geometry.cell_size_y * geometry.axis_y[i, j]
        + offsets[:, 2:] * pitch * geometry.normal[i, j]
    )
    snap = geometry.snap(points)
    assert np.all(snap.inside)
    assert np.all(snap.module == m)
    assert np.all(snap.layer == lay)
    assert np.all(snap.x == 3)
    assert np.all(snap.y == -7)


@pytest.mark.parametrize(
    ("position", "status"),
    [
        ((1800.0, 0.0, 0.0), SnapStatus.BELOW_FIRST_LAYER),
        ((2200.0, 0.0, 0.0), SnapStatus.BEYOND_LAST_LAYER),
        ((2000.0, 0.0, 2400.0), SnapStatus.BEYOND_HALF_LENGTH),
        # Between two staves (phi = 45 deg), outside the silicon of both.
        ((2120.0 * np.cos(np.pi / 4), 2120.0 * np.sin(np.pi / 4), 0.0), SnapStatus.BEYOND_STAVE_EDGE),
        ((np.nan, 0.0, 0.0), SnapStatus.NON_FINITE),
        ((2000.0, 0.0, 0.0), SnapStatus.INSIDE),
    ],
)
def test_snap_status(geometry, position, status):
    assert geometry.snap([position]).status[0] == status
