from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import pymeshfix  # noqa: E402
from bench import disconnected_tetrahedra, grid, load_upstream, open_cylinder  # noqa: E402
from pymeshfix import _geometry, _lib  # noqa: E402


def timeit(fn, reps=5):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def show(label, fn, reps=5):
    print(f"{label:42s} {timeit(fn, reps)*1e3:9.3f} ms")


def loaded(module, points, faces):
    mesh = module.PyTMesh()
    mesh.set_quiet(1)
    mesh.load_array(points, faces)
    return mesh


def main():
    points, faces = grid(160)
    print(f"=== grid(160) n_faces={len(faces)} n_vertices={len(points)}")
    show("load_array (fix_connectivity)", lambda: _geometry.fix_connectivity(points, faces))
    mesh = loaded(pymeshfix, points, faces)
    v, f = mesh._vertices, mesh._faces
    print(f"    after load n_faces={len(f)} n_vertices={len(v)}")
    show("  boundary_edges", lambda: _lib.boundary_edges(f, len(v)))
    show("  face_components", lambda: _lib.face_components(f, len(v)))
    show("  orient_faces", lambda: _geometry.orient_faces(v, f))
    show("  n_boundaries (boundary_loops)", lambda: _geometry.boundary_loops(f, len(v)))

    points, faces = open_cylinder(256)
    print("=== cylinder 256 segments")

    def fill():
        m = loaded(pymeshfix, points, faces)
        m.fill_small_boundaries(0, False)
        return m.n_faces

    show("  fill_small_boundaries", fill)

    points, faces = disconnected_tetrahedra(250)
    print("=== 250 tetrahedra")

    def rmsc():
        m = loaded(pymeshfix, points, faces)
        m.remove_smallest_components()

    show("  remove_smallest_components", rmsc)

    points, faces = open_cylinder(32)
    print("=== cylinder 32, full repair")

    def repair():
        m = loaded(pymeshfix, points, faces)
        m.fill_small_boundaries(0, True)
        m.remove_smallest_components()
        m.clean()

    show("  full repair", repair)

    upstream = load_upstream()
    points, faces = grid(160)
    show("[upstream] load+n_boundaries", lambda: loaded(upstream, points, faces).n_boundaries, 3)
    points, faces = disconnected_tetrahedra(250)

    def up_rmsc():
        m = loaded(upstream, points, faces)
        m.remove_smallest_components()

    show("[upstream] remove_smallest_components", up_rmsc, 3)
    np.zeros(1)


if __name__ == "__main__":
    if "--prof" in sys.argv:
        points, faces = grid(160)
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(3):
            m = loaded(pymeshfix, points, faces)
            m.n_boundaries
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)
    else:
        main()
