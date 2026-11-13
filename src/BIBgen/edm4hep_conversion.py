"""
Convert generated ECAL barrel hits into EDM4hep events usable as BIB overlay input.

The output mimics the ``ddsim`` BIB files (e.g. ``BIB10TeV/sim_mp/BIB_sim_*.edm4hep.root``)
as far as the generated information allows:

* Each generated hit becomes one ``CaloHitContribution``. Its energy and time are
  the generated energy and time; its ``stepPosition`` is the generated position
  (before snapping). ``PDG``, ``stepLength`` and the MCParticle link are left empty.
* Contributions are grouped by detector cell into ``SimCalorimeterHit``s, one per
  cell, as Geant4 does. A hit's energy is the sum of its contributions and its
  position is the cell center.
* Generated hits with non-positive energy, non-finite values, or positions outside
  the instrumented ECAL barrel are cut, and every cut is counted.

``prepare_event`` does the numerical work and needs only numpy. ``write_edm4hep_event``
writes the result and needs podio and edm4hep (available in the MuColl software image).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from BIBgen.ecal_geometry import ECalBarrelGeometry, SnapStatus

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
UIntArray = npt.NDArray[np.uint64]

# Collections in the ddsim BIB files other than the ECAL barrel. With
# ``placeholders=True`` they are written empty so the file has the same structure;
# the generated data contain no information about them.
PLACEHOLDER_CALO_COLLECTIONS = (
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
    "YokeBarrelCollection",
    "YokeEndcapCollection",
)
PLACEHOLDER_TRACKER_COLLECTIONS = (
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "InnerTrackerBarrelCollection",
    "InnerTrackerEndcapCollection",
    "OuterTrackerBarrelCollection",
    "OuterTrackerEndcapCollection",
)
MC_PARTICLE_COLLECTION = "MCParticles"
EVENT_HEADER_COLLECTION = "EventHeader"
CONTRIBUTION_SUFFIX = "Contributions"

# Reasons a generated hit is cut, in the order they are applied.
CUT_REASONS = (
    "non_finite",
    "non_positive_energy",
    "below_first_layer",
    "beyond_last_layer",
    "beyond_half_length",
    "beyond_stave_edge",
)
_STATUS_TO_REASON = {
    SnapStatus.BELOW_FIRST_LAYER: "below_first_layer",
    SnapStatus.BEYOND_LAST_LAYER: "beyond_last_layer",
    SnapStatus.BEYOND_HALF_LENGTH: "beyond_half_length",
    SnapStatus.BEYOND_STAVE_EDGE: "beyond_stave_edge",
}


@dataclass
class ConvertedEvent:
    """Generated hits after cuts and snapping, grouped by cell.

    Attributes
    ----------
    cell_ids : ndarray of uint64, shape (n_cells,)
        Cell ID of each ``SimCalorimeterHit``, unique within the event.
    cell_positions : ndarray, shape (n_cells, 3)
        Cell centers in mm.
    cell_energies : ndarray, shape (n_cells,)
        Sum of the contribution energies in each cell, in GeV.
    contribution_cell : ndarray of int64, shape (n_contributions,)
        Index into the cell arrays for each contribution, sorted ascending.
    contribution_energies, contribution_times : ndarray, shape (n_contributions,)
        Generated energy (GeV) and time (ns) of each kept hit.
    contribution_positions : ndarray, shape (n_contributions, 3)
        Generated (unsnapped) positions in mm, stored as ``stepPosition``.
    counts : dict
        Bookkeeping: input size, cuts by reason, what was written. See ``prepare_event``.
    """

    cell_ids: UIntArray
    cell_positions: FloatArray
    cell_energies: FloatArray
    contribution_cell: IntArray
    contribution_energies: FloatArray
    contribution_times: FloatArray
    contribution_positions: FloatArray
    counts: dict[str, float] = field(default_factory=dict)


def cylindrical_to_cartesian(phi: npt.ArrayLike, s: npt.ArrayLike, z: npt.ArrayLike) -> FloatArray:
    """Return (n, 3) Cartesian positions from azimuth ``phi`` (rad), transverse radius ``s`` and ``z``."""
    phi_a = np.asarray(phi, dtype=np.float64)
    s_a = np.asarray(s, dtype=np.float64)
    return np.stack([s_a * np.cos(phi_a), s_a * np.sin(phi_a), np.asarray(z, dtype=np.float64)], axis=-1)


def prepare_event(
    energies: npt.ArrayLike,
    positions: npt.ArrayLike,
    geometry: ECalBarrelGeometry,
    times: npt.ArrayLike | None = None,
) -> ConvertedEvent:
    """Cut, snap and group one event's generated hits.

    Parameters
    ----------
    energies : array_like, shape (n,)
        Hit energies in GeV.
    positions : array_like, shape (n, 3)
        Global (x, y, z) in mm.
    geometry : ECalBarrelGeometry
        Cell geometry used for snapping.
    times : array_like, shape (n,), optional
        Hit times in ns. If omitted, every time is 0.

    Returns
    -------
    ConvertedEvent
        ``counts`` holds, as floats:

        * ``n_input``: number of generated hits.
        * ``n_cut_<reason>`` and ``energy_cut_<reason>`` for each reason in
          ``CUT_REASONS`` (energy summed over the cut hits, NaN excluded).
        * ``n_contributions``, ``n_cells``, ``n_cells_multiple_contributions``,
          ``energy_written``.
        * ``snap_distance_mean``, ``snap_distance_max``: distance in mm from each
          kept generated position to its cell center.
    """
    e = np.asarray(energies, dtype=np.float64).reshape(-1)
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(e)
    if len(p) != n:
        msg = f"{n} energies but {len(p)} positions"
        raise ValueError(msg)
    t = np.zeros(n) if times is None else np.asarray(times, dtype=np.float64).reshape(-1)
    if len(t) != n:
        msg = f"{n} energies but {len(t)} times"
        raise ValueError(msg)

    counts: dict[str, float] = {"n_input": float(n)}
    reason = np.full(n, "", dtype=object)

    finite = np.isfinite(e) & np.all(np.isfinite(p), axis=1) & np.isfinite(t)
    reason[~finite] = "non_finite"
    nonpositive = finite & (e <= 0)
    reason[nonpositive] = "non_positive_energy"

    candidate = finite & ~nonpositive
    snap = geometry.snap(p[candidate])
    snap_reason = np.full(len(snap.status), "", dtype=object)
    for status, name in _STATUS_TO_REASON.items():
        snap_reason[snap.status == status] = name
    reason[candidate] = snap_reason

    for name in CUT_REASONS:
        mask = reason == name
        counts[f"n_cut_{name}"] = float(mask.sum())
        counts[f"energy_cut_{name}"] = float(np.nansum(e[mask]))

    keep_in_candidate = snap.inside
    keep = np.zeros(n, dtype=bool)
    keep[candidate] = keep_in_candidate

    cell_ids_all = geometry.encode_cell_id(
        snap.module[keep_in_candidate],
        snap.layer[keep_in_candidate],
        snap.x[keep_in_candidate],
        snap.y[keep_in_candidate],
    )
    cell_ids, contribution_cell = np.unique(cell_ids_all, return_inverse=True)
    contribution_cell = contribution_cell.reshape(-1).astype(np.int64)

    # Sort contributions by cell so each cell's contributions are contiguous.
    order = np.argsort(contribution_cell, kind="stable")
    contribution_cell = contribution_cell[order]
    kept_e = e[keep][order]
    kept_t = t[keep][order]
    kept_p = p[keep][order]

    fields = geometry.decode_cell_id(cell_ids)
    cell_positions = geometry.cell_center(fields["module"], fields["layer"], fields["x"], fields["y"])
    cell_energies = np.bincount(contribution_cell, weights=kept_e, minlength=len(cell_ids)).astype(np.float64)

    per_cell = np.bincount(contribution_cell, minlength=len(cell_ids))
    snap_distance = np.linalg.norm(kept_p - cell_positions[contribution_cell], axis=1)
    counts["n_contributions"] = float(len(kept_e))
    counts["n_cells"] = float(len(cell_ids))
    counts["n_cells_multiple_contributions"] = float(np.sum(per_cell > 1))
    counts["energy_written"] = float(kept_e.sum())
    counts["snap_distance_mean"] = float(snap_distance.mean()) if len(snap_distance) else 0.0
    counts["snap_distance_max"] = float(snap_distance.max()) if len(snap_distance) else 0.0

    return ConvertedEvent(
        cell_ids=cell_ids,
        cell_positions=cell_positions,
        cell_energies=cell_energies,
        contribution_cell=contribution_cell,
        contribution_energies=kept_e,
        contribution_times=kept_t,
        contribution_positions=kept_p,
        counts=counts,
    )


def write_edm4hep_event(
    path: str | Path,
    event: ConvertedEvent,
    geometry: ECalBarrelGeometry,
    event_number: int = 0,
    run_number: int = 0,
    placeholders: bool = True,
) -> None:
    """Write one converted event to its own EDM4hep ROOT file.

    One event per file is what ``OverlayTimingRandomMix`` expects: it reads only
    entry 0 of each background file.

    Parameters
    ----------
    path : str or Path
        Output ``.edm4hep.root`` file.
    event : ConvertedEvent
        Output of ``prepare_event``.
    geometry : ECalBarrelGeometry
        Supplies the collection name and cell-ID encoding.
    event_number, run_number : int
        Stored in the ``EventHeader``.
    placeholders : bool
        If True, also write empty collections for every other subdetector found in
        the ddsim BIB files. ``MCParticles`` (empty) and ``EventHeader`` are always written.
    """
    # podio and edm4hep exist only in the Key4hep/MuColl software environment, so they
    # are imported here rather than at module level: the rest of this module works without them.
    import edm4hep  # type: ignore[import-not-found]  # noqa: PLC0415
    import podio  # type: ignore[import-not-found]  # noqa: PLC0415
    from podio import root_io  # noqa: PLC0415

    frame = podio.Frame()

    header_coll = edm4hep.EventHeaderCollection()
    header = header_coll.create()
    header.setEventNumber(int(event_number))
    header.setRunNumber(int(run_number))
    frame.put(header_coll, EVENT_HEADER_COLLECTION)

    # No truth information is generated, so the MCParticle collection is empty and
    # contributions do not point to any particle.
    frame.put(edm4hep.MCParticleCollection(), MC_PARTICLE_COLLECTION)

    hits = edm4hep.SimCalorimeterHitCollection()
    contributions = edm4hep.CaloHitContributionCollection()
    contribution_handles = []
    for i in range(len(event.contribution_energies)):
        c = contributions.create()
        c.setPDG(0)
        c.setEnergy(float(event.contribution_energies[i]))
        c.setTime(float(event.contribution_times[i]))
        c.setStepLength(0.0)
        px, py, pz = event.contribution_positions[i]
        c.setStepPosition(edm4hep.Vector3f(float(px), float(py), float(pz)))
        contribution_handles.append(c)

    begin = np.searchsorted(event.contribution_cell, np.arange(len(event.cell_ids)), side="left")
    end = np.searchsorted(event.contribution_cell, np.arange(len(event.cell_ids)), side="right")
    for j in range(len(event.cell_ids)):
        h = hits.create()
        h.setCellID(int(event.cell_ids[j]))
        h.setEnergy(float(event.cell_energies[j]))
        x, y, z = event.cell_positions[j]
        h.setPosition(edm4hep.Vector3f(float(x), float(y), float(z)))
        for k in range(begin[j], end[j]):
            h.addToContributions(contribution_handles[k])

    frame.put(hits, geometry.collection)
    frame.put(contributions, geometry.collection + CONTRIBUTION_SUFFIX)

    if placeholders:
        for name in PLACEHOLDER_CALO_COLLECTIONS:
            frame.put(edm4hep.SimCalorimeterHitCollection(), name)
            frame.put(edm4hep.CaloHitContributionCollection(), name + CONTRIBUTION_SUFFIX)
        for name in PLACEHOLDER_TRACKER_COLLECTIONS:
            frame.put(edm4hep.SimTrackerHitCollection(), name)

    # The cell-ID encoding is stored as collection metadata, as ddsim does.
    metadata = podio.Frame()
    metadata.put_parameter(
        f"{geometry.collection}__{edm4hep.labels.CellIDEncoding}", geometry.cell_id_encoding
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = root_io.Writer(str(path))
    writer.write_frame(frame, "events")
    writer.write_frame(metadata, "metadata")
    # The ROOT writer finalizes the file when it is destroyed.
    del writer
