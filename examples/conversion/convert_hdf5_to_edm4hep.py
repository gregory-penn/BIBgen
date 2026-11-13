"""
Convert ECAL barrel hits in an HDF5 file into EDM4hep files for BIB overlay.

Each event becomes its own file, ``<prefix>_<N>.edm4hep.root``, because the
overlay (``OverlayTimingRandomMix``) reads only the first event of each file.
Every generated hit becomes one ``CaloHitContribution``, snapped to a MAIA ECAL
barrel cell; see ``BIBgen.edm4hep_conversion`` for the details.

Input layout: events are ``evt_N`` entries holding an (n_hits, >=4) array whose
first four columns are (E [GeV], phi [rad], s [mm], z [mm]). They may sit at the
top level (generated ``<tag>_like.hdf5`` files) or inside a group, optionally as
a named dataset per event (raw files: ``--group test --dataset tau0``).

The values must be in physical units. Sphered (normalized) generated output must
be un-sphered first; this script does not do it.

Hits are cut, and counted per event in a CSV summary, if they have non-positive
energy, non-finite values, or a position outside the instrumented ECAL barrel.

Writing needs podio and edm4hep, so run inside the MuColl software image:

    apptainer exec -B /scratch -B /cvmfs \\
        /cvmfs/unpacked.cern.ch/ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.1-amd64 \\
        bash -c 'source /opt/setup_mucoll.sh && PYTHONPATH=src:$PYTHONPATH \\
            python examples/conversion/convert_hdf5_to_edm4hep.py -i in.hdf5 -o out_dir'
"""
from __future__ import annotations

import csv
import re
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np

from BIBgen.ecal_geometry import DEFAULT_GEOMETRY_TABLE, ECalBarrelGeometry
from BIBgen.edm4hep_conversion import (
    CUT_REASONS,
    cylindrical_to_cartesian,
    prepare_event,
    write_edm4hep_event,
)

# Column order of the hit features (see the data conventions in the README).
COL_E, COL_PHI, COL_S, COL_Z = 0, 1, 2, 3

EVENT_NAME = re.compile(r"^evt_(\d+)$")


def list_events(container, dataset):
    """Return [(event_number, h5py.Dataset)] sorted by event number."""
    events = []
    for name in container:
        match = EVENT_NAME.match(name)
        if match is None:
            continue
        obj = container[name]
        if dataset is not None:
            obj = obj[dataset]
        if not isinstance(obj, h5py.Dataset):
            msg = f"{name} is not a dataset; use --dataset to name the dataset inside each event group"
            raise TypeError(msg)
        events.append((int(match.group(1)), obj))
    return sorted(events, key=lambda item: item[0])


def summarize(rows):
    """Add a TOTAL row (sums; snap distances combined over all contributions)."""
    total = {"event": "TOTAL"}
    for key in rows[0]:
        if key in ("event", "file"):
            continue
        if key == "snap_distance_mean":
            n = sum(r["n_contributions"] for r in rows)
            total[key] = sum(r[key] * r["n_contributions"] for r in rows) / n if n else 0.0
        elif key == "snap_distance_max":
            total[key] = max(r[key] for r in rows)
        else:
            total[key] = sum(r[key] for r in rows)
    total["file"] = ""
    return total


def print_totals(total):
    n_input = total["n_input"]
    print(f"\nInput hits: {int(n_input)}")
    for reason in CUT_REASONS:
        n_cut = total[f"n_cut_{reason}"]
        frac = n_cut / n_input if n_input else 0.0
        print(f"  cut {reason:<22} {int(n_cut):>10}  ({100 * frac:.3f}%)  energy {total[f'energy_cut_{reason}']:.6g} GeV")
    print(f"Written contributions: {int(total['n_contributions'])}"
          f"  in {int(total['n_cells'])} cells"
          f"  ({int(total['n_cells_multiple_contributions'])} cells with >1 contribution)")
    print(f"Written energy: {total['energy_written']:.6g} GeV")
    print(f"Snap distance: mean {total['snap_distance_mean']:.4g} mm, max {total['snap_distance_max']:.4g} mm")


def main(args):
    geometry = ECalBarrelGeometry.from_json(args.geometry)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with h5py.File(args.input, "r") as f:
        container = f[args.group] if args.group else f
        events = list_events(container, args.dataset)
        if args.max_events is not None:
            events = events[: args.max_events]
        if not events:
            print("No evt_N entries found in", args.input)
            return 1
        print(f"Converting {len(events)} events from {args.input}")

        for event_number, dataset in events:
            data = np.asarray(dataset[...], dtype=np.float64)
            if data.ndim != 2 or data.shape[1] < 4:
                print(f"evt_{event_number} has shape {data.shape}; expected (n_hits, >=4)")
                return 1
            if args.time_column is not None and args.time_column >= data.shape[1]:
                print(f"--time-column {args.time_column} but evt_{event_number} has {data.shape[1]} columns")
                return 1
            positions = cylindrical_to_cartesian(data[:, COL_PHI], data[:, COL_S], data[:, COL_Z])
            times = data[:, args.time_column] if args.time_column is not None else None
            event = prepare_event(data[:, COL_E], positions, geometry, times=times)

            path = out_dir / f"{args.prefix}_{event_number}.edm4hep.root"
            write_edm4hep_event(
                path,
                event,
                geometry,
                event_number=event_number,
                run_number=args.run_number,
                placeholders=not args.no_placeholders,
            )
            rows.append({"event": event_number, **event.counts, "file": path.name})
            c = event.counts
            n_cut = sum(c[f"n_cut_{r}"] for r in CUT_REASONS)
            print(f"evt_{event_number}: {int(c['n_input'])} hits -> {int(c['n_contributions'])} contributions"
                  f" in {int(c['n_cells'])} cells ({int(n_cut)} cut) -> {path}")

    total = summarize(rows)
    summary = Path(args.summary) if args.summary else out_dir / "conversion_summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(total)
    print_totals(total)
    print("Wrote summary", summary)
    return 0


# Example:
# python examples/conversion/convert_hdf5_to_edm4hep.py -i generated.hdf5 -o edm4hep_out
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-i", "--input", required=True, help="Input HDF5 file")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory for the .edm4hep.root files")
    parser.add_argument("-g", "--geometry", default=str(DEFAULT_GEOMETRY_TABLE),
                        help="ECAL barrel geometry table (default: %(default)s)")
    parser.add_argument("--group", default=None,
                        help="HDF5 group holding the evt_N entries (default: top level)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name inside each evt_N group, e.g. tau0 (default: evt_N is the dataset)")
    parser.add_argument("--time-column", type=int, default=None,
                        help="Column holding hit time in ns (default: none; every time is 0)")
    parser.add_argument("--prefix", default="BIB_gen", help="Output file name prefix (default: %(default)s)")
    parser.add_argument("--run-number", type=int, default=0, help="Run number in the EventHeader")
    parser.add_argument("--max-events", type=int, default=None, help="Convert at most this many events")
    parser.add_argument("--no-placeholders", action="store_true",
                        help="Do not write empty collections for the other subdetectors")
    parser.add_argument("--summary", default=None,
                        help="Summary CSV (default: <output-dir>/conversion_summary.csv)")
    print("\nFinished with exit code:", main(parser.parse_args()))
