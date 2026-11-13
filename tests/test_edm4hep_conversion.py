from __future__ import annotations

import numpy as np
import pytest

from BIBgen.ecal_geometry import cell_center
from BIBgen.edm4hep_conversion import (
    CUT_REASONS,
    cylindrical_to_cartesian,
    prepare_event,
)


def test_cylindrical_to_cartesian():
    p = cylindrical_to_cartesian([0.0, np.pi / 2], [2000.0, 1900.0], [10.0, -5.0])
    np.testing.assert_allclose(p, [[2000.0, 0.0, 10.0], [0.0, 1900.0, -5.0]], atol=1e-9)


def test_prepare_event_cuts_merges_and_counts():
    center = cell_center(0, 3, 10, -20)
    other = cell_center(4, 30, -50, 100)
    shift = np.array([0.0, 1.0, 1.0])
    energies = [1e-4, 2e-4, -5e-5, 3e-4, 1e-4, np.nan]
    positions = [
        center,                       # kept
        center + shift,               # kept, same cell as the first
        center,                       # cut: negative energy
        other,                        # kept, its own cell
        [1800.0, 0.0, 0.0],           # cut: below the first layer
        center,                       # cut: non-finite energy
    ]
    event = prepare_event(energies, positions)
    c = event.counts

    assert c["n_input"] == 6
    assert c["n_cut_non_positive_energy"] == 1
    assert c["energy_cut_non_positive_energy"] == pytest.approx(-5e-5)
    assert c["n_cut_below_first_layer"] == 1
    assert c["n_cut_non_finite"] == 1
    assert sum(c[f"n_cut_{r}"] for r in CUT_REASONS) == 3
    assert c["n_contributions"] == 3
    assert c["n_cells"] == 2
    assert c["n_cells_multiple_contributions"] == 1
    assert c["energy_written"] == pytest.approx(6e-4)

    # Cells are unique, hit energy is the sum of its contributions, position is the cell center.
    assert len(np.unique(event.cell_ids)) == 2
    np.testing.assert_allclose(
        np.bincount(event.contribution_cell, weights=event.contribution_energies), event.cell_energies
    )
    merged = np.argmax(np.bincount(event.contribution_cell))
    assert event.cell_energies[merged] == pytest.approx(3e-4)
    np.testing.assert_allclose(event.cell_positions[merged], center, atol=1e-9)
    # Contributions keep the unsnapped positions; times default to 0.
    assert any(np.allclose(p, center + shift) for p in event.contribution_positions)
    np.testing.assert_array_equal(event.contribution_times, 0.0)


def test_prepare_event_uses_given_times():
    p = cell_center([0, 1], [0, 0], [0, 0], [0, 0])
    event = prepare_event([1e-4, 2e-4], p, times=[3.5, 7.25])
    assert sorted(event.contribution_times.tolist()) == [3.5, 7.25]


def test_prepare_event_all_cut():
    event = prepare_event([-1.0, -2.0], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert event.counts["n_contributions"] == 0
    assert len(event.cell_ids) == 0
    assert event.counts["snap_distance_max"] == 0.0
