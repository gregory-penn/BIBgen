"""
Extract the MAIA ECAL barrel cell geometry from DD4hep into a JSON table.

The table is what ``BIBgen.ecal_geometry.ECalBarrelGeometry`` uses to snap
positions to cells, so the conversion to EDM4hep does not need DD4hep itself.

For every (module, layer) the table stores the global position of cell
(x=0, y=0) and the global directions of increasing x and y cell index. These
come from DD4hep's cellID -> position conversion, so they include the actual
placement and rotation of every silicon layer. The half-extents of the silicon
come from the placed sensitive volume (a box).

Must run inside the MuColl software image, where DD4hep and k4geo are available:

    apptainer exec -B /scratch -B /cvmfs \\
        /cvmfs/unpacked.cern.ch/ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.1-amd64 \\
        bash -c 'source /opt/setup_mucoll.sh && python examples/conversion/extract_ecal_geometry.py'
"""
from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import ROOT
from dd4hep import Detector

# DD4hep lengths are in cm; the table and EDM4hep use mm.
CM_TO_MM = 10.0

DEFAULT_COMPACT = "k4geo/MuColl/MAIA/compact/MAIA_v0/MAIA_v0.xml"


def parse_encoding(encoding):
    """Return {field: (offset, width, signed)} for a DD4hep bitfield string."""
    fields = {}
    offset = 0
    for item in encoding.split(","):
        parts = item.split(":")
        name = parts[0]
        if len(parts) == 3:
            offset = int(parts[1])
            width = int(parts[2])
        else:
            width = int(parts[1])
        fields[name] = (offset, abs(width), width < 0)
        offset += abs(width)
    return fields


def encode(fields, values):
    """Pack {field: value} into a cellID using the parsed bitfield."""
    cell_id = 0
    for name, value in values.items():
        offset, width, _ = fields[name]
        cell_id |= (int(value) & ((1 << width) - 1)) << offset
    return cell_id


def main(args):
    compact = args.compact
    if compact is None:
        compact = str(Path(os.environ["MUCOLL_GEO"]) / DEFAULT_COMPACT)
    print("Loading geometry from", compact)

    description = Detector.getInstance()
    description.fromXML(compact)
    detector = description.detector(args.detector)
    readout = description.readout(args.readout)
    encoding = readout.idSpec().fieldDescription()
    fields = parse_encoding(encoding)
    print("CellID encoding:", encoding)

    segmentation = readout.segmentation()
    seg_params = {p.name(): p.value() for p in segmentation.parameters()}
    print("Segmentation:", segmentation.type(), seg_params)
    if segmentation.type() != "CartesianGridXY":
        print("Expected a CartesianGridXY segmentation, got", segmentation.type())
        return 1
    cell_size_x = float(seg_params["grid_size_x"]) * CM_TO_MM
    cell_size_y = float(seg_params["grid_size_y"]) * CM_TO_MM
    if float(seg_params["offset_x"]) != 0.0 or float(seg_params["offset_y"]) != 0.0:
        print("Non-zero segmentation offsets are not supported")
        return 1

    system_id = detector.id()
    converter = ROOT.dd4hep.rec.CellIDPositionConverter(description)
    volume_manager = description.volumeManager()

    # The (module, layer) pairs that exist, read from the DetElement tree's volume IDs.
    module_layers = {}
    for _, stave in detector.children():
        stave_ids = dict(stave.placement().volIDs())
        module = int(stave_ids["module"])
        for _, layer in stave.children():
            layer_ids = dict(layer.placement().volIDs())
            module_layers.setdefault(module, []).append(int(layer_ids["layer"]))
    modules = sorted(module_layers)
    layers = sorted(module_layers[modules[0]])
    for module in modules:
        if sorted(module_layers[module]) != layers:
            print("Module", module, "has a different set of layers; not supported")
            return 1
    print(f"Found {len(modules)} modules x {len(layers)} layers")

    fixed_fields = {name: 0 for name in fields if name not in ("module", "layer", "x", "y")}
    fixed_fields["system"] = system_id

    def cell_id(module, layer, ix, iy):
        return encode(fields, {**fixed_fields, "module": module, "layer": layer, "x": ix, "y": iy})

    def position(cid):
        p = converter.position(cid)
        return np.array([p.x(), p.y(), p.z()]) * CM_TO_MM

    shape = (len(modules), len(layers))
    origin = np.zeros((*shape, 3))
    axis_x = np.zeros((*shape, 3))
    axis_y = np.zeros((*shape, 3))
    half_x = np.zeros(shape)
    half_y = np.zeros(shape)
    half_thickness = np.zeros(shape)

    # Use a step of 10 cells to find the axis directions, to reduce rounding error.
    step = 10
    for i, module in enumerate(modules):
        for j, layer in enumerate(layers):
            o = position(cell_id(module, layer, 0, 0))
            ex = (position(cell_id(module, layer, step, 0)) - o) / (step * cell_size_x)
            ey = (position(cell_id(module, layer, 0, step)) - o) / (step * cell_size_y)
            origin[i, j] = o
            axis_x[i, j] = ex / np.linalg.norm(ex)
            axis_y[i, j] = ey / np.linalg.norm(ey)

            placement = volume_manager.lookupVolumePlacement(cell_id(module, layer, 0, 0))
            solid = placement.volume().solid()
            if solid.IsA().GetName() != "TGeoBBox":
                print("Sensitive volume is not a box:", solid.IsA().GetName())
                return 1
            half_x[i, j] = solid.GetDX() * CM_TO_MM
            half_y[i, j] = solid.GetDY() * CM_TO_MM
            half_thickness[i, j] = solid.GetDZ() * CM_TO_MM

    normal = np.cross(axis_x, axis_y)

    # Sanity checks: the axes must be orthonormal and the grid origin must be the
    # center of the sensitive box (the segmentation is defined in the box frame).
    max_dot = np.abs(np.einsum("mlk,mlk->ml", axis_x, axis_y)).max()
    print(f"Largest |axis_x . axis_y| = {max_dot:.2e}")
    if max_dot > 1e-6:
        print("Cell axes are not orthogonal")
        return 1

    # Check the table against DD4hep on randomly chosen cells inside the silicon.
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(args.n_checks):
        i = rng.integers(len(modules))
        j = rng.integers(len(layers))
        ix = int(rng.integers(-int(half_x[i, j] / cell_size_x), int(half_x[i, j] / cell_size_x) + 1))
        iy = int(rng.integers(-int(half_y[i, j] / cell_size_y), int(half_y[i, j] / cell_size_y) + 1))
        predicted = origin[i, j] + ix * cell_size_x * axis_x[i, j] + iy * cell_size_y * axis_y[i, j]
        actual = position(cell_id(modules[i], layers[j], ix, iy))
        worst = max(worst, float(np.abs(predicted - actual).max()))
    print(f"Largest table-vs-DD4hep difference over {args.n_checks} random cells: {worst:.2e} mm")
    if worst > 1e-3:
        print("Table does not reproduce DD4hep cell positions")
        return 1

    table = {
        "description": (
            "MAIA ECAL barrel cell geometry extracted from DD4hep by "
            "examples/conversion/extract_ecal_geometry.py. Lengths in mm. Cell (module, layer, x, y) "
            "is centered at origin + x*cell_size_x*axis_x + y*cell_size_y*axis_y. half_x, half_y and "
            "half_thickness are the half-extents of the silicon box along axis_x, axis_y and normal."
        ),
        "compact_file": compact,
        "detector": args.detector,
        "collection": args.readout,
        "cell_id_encoding": encoding,
        "fixed_fields": fixed_fields,
        "cell_size_x": cell_size_x,
        "cell_size_y": cell_size_y,
        "modules": modules,
        "layers": layers,
        "origin": origin.tolist(),
        "axis_x": axis_x.tolist(),
        "axis_y": axis_y.tolist(),
        "normal": normal.tolist(),
        "half_x": half_x.tolist(),
        "half_y": half_y.tolist(),
        "half_thickness": half_thickness.tolist(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(table, f, indent=1)
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--compact", default=None,
                        help="DD4hep compact file (default: $MUCOLL_GEO/" + DEFAULT_COMPACT + ")")
    parser.add_argument("-o", "--output", default="config/maia_v0_ecal_barrel_geometry.json",
                        help="Output JSON table")
    parser.add_argument("--detector", default="ECalBarrel", help="DD4hep detector name")
    parser.add_argument("--readout", default="ECalBarrelCollection", help="DD4hep readout name")
    parser.add_argument("--n-checks", type=int, default=2000,
                        help="Number of random cells checked against DD4hep")
    print("\nFinished with exit code:", main(parser.parse_args()))
