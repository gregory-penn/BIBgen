"""
Validate the HDF5 -> EDM4hep conversion on a real ddsim BIB file.

1. Read the ECAL barrel hits of a real file (positions are cell centers, cell IDs
   come from Geant4/DD4hep).
2. Write them to an HDF5 file in the generated layout: one evt_N dataset with
   columns (E, phi, s, z, t), where t is the earliest contribution time of the hit.
3. Run convert_hdf5_to_edm4hep.py on it (with --time-column 4).
4. Read the output back and require, hit by hit: identical cell IDs, equal
   energies and positions, one contribution per hit carrying the hit's energy,
   time and position, no cuts, and the same CellIDEncoding metadata.
5. Jitter test: move each real position to a random point inside its own cell
   (within the silicon plane and up to nearly half a layer pitch in depth) and
   report how often it snaps back to the original cell.

Run inside the MuColl software image (needs podio and edm4hep):

    apptainer exec -B /scratch -B /cvmfs -B /ospool \\
        /cvmfs/unpacked.cern.ch/ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.1-amd64 \\
        bash -c 'source /opt/setup_mucoll.sh && PYTHONPATH=src:$PYTHONPATH \\
            python examples/conversion/validate_conversion.py -i BIB_sim_1000.edm4hep.root'
"""
from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from podio.root_io import Reader

from BIBgen.ecal_geometry import DEFAULT_GEOMETRY_TABLE, ECalBarrelGeometry, SnapStatus
from BIBgen.edm4hep_conversion import CONTRIBUTION_SUFFIX, CUT_REASONS

CONVERTER = Path(__file__).resolve().parent / "convert_hdf5_to_edm4hep.py"


def read_ecal_barrel(path, collection):
    """Return cellID, energy, position (n, 3), earliest contribution time, and the encoding."""
    reader = Reader(str(path))
    frame = next(iter(reader.get("events")))
    hits = frame.get(collection)
    n = len(hits)
    cell_id = np.zeros(n, dtype=np.uint64)
    energy = np.zeros(n)
    position = np.zeros((n, 3))
    time = np.zeros(n)
    n_contrib = np.zeros(n, dtype=np.int64)
    for i, h in enumerate(hits):
        cell_id[i] = h.getCellID()
        energy[i] = h.getEnergy()
        p = h.getPosition()
        position[i] = (p.x, p.y, p.z)
        times = [c.getTime() for c in h.getContributions()]
        n_contrib[i] = len(times)
        time[i] = min(times) if times else np.nan
    encoding = None
    for meta in reader.get("metadata"):
        key = f"{collection}__CellIDEncoding"
        if key in meta.parameters:
            encoding = meta.get_parameter(key)
    return {
        "cell_id": cell_id,
        "energy": energy,
        "position": position,
        "time": time,
        "n_contrib": n_contrib,
        "encoding": encoding,
        "collections": sorted(str(c) for c in frame.getAvailableCollections()),
    }


def read_converted(path, collection):
    """Read hits and their contributions from a converted file."""
    reader = Reader(str(path))
    frame = next(iter(reader.get("events")))
    hits = frame.get(collection)
    # Cell IDs go straight into a uint64 array: building an array from a list of
    # Python ints above 2**63 can give float64 and lose the low bits.
    cell_id = np.zeros(len(hits), dtype=np.uint64)
    out = {
        "energy": [], "position": [], "n_contrib": [],
        "c_energy": [], "c_time": [], "c_step": [], "c_pdg": [], "c_has_particle": [],
    }
    for i, h in enumerate(hits):
        cell_id[i] = h.getCellID()
        out["energy"].append(h.getEnergy())
        p = h.getPosition()
        out["position"].append((p.x, p.y, p.z))
        contribs = list(h.getContributions())
        out["n_contrib"].append(len(contribs))
        c = contribs[0]
        out["c_energy"].append(c.getEnergy())
        out["c_time"].append(c.getTime())
        s = c.getStepPosition()
        out["c_step"].append((s.x, s.y, s.z))
        out["c_pdg"].append(c.getPDG())
        out["c_has_particle"].append(c.getParticle().isAvailable())
    result = {k: np.asarray(v) for k, v in out.items()}
    result["cell_id"] = cell_id
    encoding = None
    for meta in reader.get("metadata"):
        key = f"{collection}__CellIDEncoding"
        if key in meta.parameters:
            encoding = meta.get_parameter(key)
    result["encoding"] = encoding
    result["collections"] = sorted(str(c) for c in frame.getAvailableCollections())
    result["n_contributions_total"] = len(frame.get(collection + CONTRIBUTION_SUFFIX))
    return result


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return bool(ok)


def jitter_test(geometry, real, rng):
    """Move each real hit inside its cell and check it snaps back to the same cell."""
    fields = geometry.decode_cell_id(real["cell_id"])
    m = np.searchsorted(geometry.modules, fields["module"])
    lay = np.searchsorted(geometry.layers, fields["layer"])
    n = len(m)
    # Offsets in the cell's own frame, strictly inside the cell (99% of the half-size).
    du = rng.uniform(-0.495, 0.495, n) * geometry.cell_size_x
    dv = rng.uniform(-0.495, 0.495, n) * geometry.cell_size_y
    pitch = np.diff(geometry.layer_depth, axis=1).min()
    dn = rng.uniform(-0.495, 0.495, n) * pitch
    jittered = (
        real["position"]
        + du[:, None] * geometry.axis_x[m, lay]
        + dv[:, None] * geometry.axis_y[m, lay]
        + dn[:, None] * geometry.normal[m, lay]
    )
    snap = geometry.snap(jittered)
    back = geometry.encode_cell_id(snap.module, snap.layer, snap.x, snap.y)
    same = (back == real["cell_id"]) & snap.inside
    print(f"Jitter test: {same.sum()} of {n} jittered hits snap back to their own cell"
          f" ({100 * same.mean():.4f}%)")
    if (~same).any():
        statuses = {SnapStatus(s).name: int(c) for s, c in zip(*np.unique(snap.status[~same], return_counts=True), strict=True)}
        print("  statuses of the others:", statuses)
        print("  their x indices:", fields["x"][~same][:20])
    return same.mean()


def main(args):
    geometry = ECalBarrelGeometry.from_json(args.geometry)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading", args.input)
    real = read_ecal_barrel(args.input, geometry.collection)
    n = len(real["energy"])
    print(f"{n} ECAL barrel hits, {real['n_contrib'].sum()} contributions")

    # Write the real hits in the generated HDF5 layout: (E, phi, s, z, t).
    x, y, z = real["position"].T
    features = np.stack([real["energy"], np.arctan2(y, x), np.hypot(x, y), z, real["time"]], axis=1)
    h5_path = out_dir / "real_hits_as_generated.hdf5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(f"evt_{args.event_number}", data=features)
    print("Wrote", h5_path)

    convert_dir = out_dir / "converted"
    command = [
        sys.executable, str(CONVERTER), "-i", str(h5_path), "-o", str(convert_dir),
        "-g", str(args.geometry), "--time-column", "4", "--prefix", "BIB_validate",
    ]
    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    print(result.stdout[-2000:])
    if result.returncode != 0 or "Finished with exit code: 0" not in result.stdout:
        print(result.stderr[-4000:])
        print("Converter failed")
        return 1

    converted_path = convert_dir / f"BIB_validate_{args.event_number}.edm4hep.root"
    conv = read_converted(converted_path, geometry.collection)

    ok = True
    order_real = np.argsort(real["cell_id"])
    order_conv = np.argsort(conv["cell_id"])
    r_id = real["cell_id"][order_real]
    c_id = conv["cell_id"][order_conv]
    ok &= check("same number of hits", len(r_id) == len(c_id), f"{len(r_id)} real, {len(c_id)} converted")
    if len(r_id) != len(c_id):
        return 1
    ok &= check("identical cell IDs", np.array_equal(r_id, c_id),
                f"{np.sum(r_id != c_id)} differ")
    r_e = real["energy"][order_real]
    c_e = conv["energy"][order_conv]
    ok &= check("hit energies equal (float32 precision)", np.allclose(c_e, r_e, rtol=1e-6, atol=0),
                f"max relative difference {np.max(np.abs(c_e - r_e) / r_e):.2e}")
    d_pos = np.abs(conv["position"][order_conv] - real["position"][order_real]).max()
    ok &= check("hit positions equal the original cell centers", d_pos < 1e-3, f"max difference {d_pos:.2e} mm")
    ok &= check("one contribution per hit", np.all(conv["n_contrib"] == 1)
                and conv["n_contributions_total"] == n)
    ok &= check("contribution energy equals hit energy",
                np.allclose(conv["c_energy"], conv["energy"], rtol=1e-6, atol=0))
    r_t = real["time"][order_real]
    c_t = conv["c_time"][order_conv]
    ok &= check("contribution time equals input time", np.allclose(c_t, r_t, rtol=1e-6, atol=1e-6),
                f"max difference {np.max(np.abs(c_t - r_t)):.2e} ns")
    d_step = np.abs(conv["c_step"][order_conv] - real["position"][order_real]).max()
    ok &= check("stepPosition equals input position", d_step < 1e-3, f"max difference {d_step:.2e} mm")
    ok &= check("PDG is 0 and no MCParticle link", np.all(conv["c_pdg"] == 0)
                and not np.any(conv["c_has_particle"]))
    ok &= check("CellIDEncoding metadata matches", conv["encoding"] == real["encoding"],
                f"{conv['encoding']!r}")
    ok &= check("same collection names as the real file", conv["collections"] == real["collections"],
                f"missing {sorted(set(real['collections']) - set(conv['collections']))},"
                f" extra {sorted(set(conv['collections']) - set(real['collections']))}")

    with (convert_dir / "conversion_summary.csv").open() as f:
        header = f.readline().strip().split(",")
        values = f.readline().strip().split(",")
    counts = dict(zip(header, values, strict=True))
    n_cut = sum(float(counts[f"n_cut_{r}"]) for r in CUT_REASONS)
    ok &= check("no hits cut", n_cut == 0, f"{int(n_cut)} cut")

    jitter_fraction = jitter_test(geometry, real, np.random.default_rng(args.seed))

    print("\nRound-trip validation", "PASSED" if ok else "FAILED")
    print(f"Jitter test: {100 * jitter_fraction:.4f}% snap back (informational)")
    return 0 if ok else 1


# Example:
# python examples/conversion/validate_conversion.py \
#     -i /ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MAIA_v0/v9/BIB10TeV/sim_mp/BIB_sim_1000.edm4hep.root
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-i", "--input", required=True, help="Real ddsim EDM4hep BIB file")
    parser.add_argument("-o", "--output-dir", default="private/output/conversion_validation",
                        help="Directory for the intermediate HDF5 and converted file (default: %(default)s)")
    parser.add_argument("-g", "--geometry", default=str(DEFAULT_GEOMETRY_TABLE),
                        help="ECAL barrel geometry table (default: %(default)s)")
    parser.add_argument("--event-number", type=int, default=0, help="Event number used for the test")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the jitter test")
    print("\nFinished with exit code:", main(parser.parse_args()))
