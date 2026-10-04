from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import pymeshfix  # noqa: E402
from bench import disconnected_tetrahedra, grid, open_cylinder  # noqa: E402
from pymeshfix import _geometry, _lib  # noqa: E402


def t(fn, reps=7):
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3


def show(label, fn, reps=7):
    print(f"{label:44s} {t(fn, reps):9.3f} ms")


def loaded(points, faces):
    m = pymeshfix.PyTMesh()
    m.set_quiet(1)
    m.load_array(points, faces)
    return m


def main():
    points, faces = grid(160)
    print(f"=== grid(160) n_faces={len(faces)} n_v={len(points)}")
    show("checked_arrays", lambda: _geometry.checked_arrays(points, faces))
    v, f = _geometry.fix_connectivity(points, faces)
    print(f"    welded n_faces={len(f)} n_v={len(v)}")

    def weld():
        _, first, inv = np.unique(points, axis=0, return_index=True, return_inverse=True)
        np.argsort(first)

    show("  np.unique(vertices,axis=0)", weld)
    def dedup():
        c = np.sort(f, axis=1)
        np.unique(c, axis=0, return_index=True)

    show("  np.unique(faces,axis=0)", dedup)
    show("  orient_faces(python)", lambda: _geometry.orient_faces(v, f))
    show("  fix_connectivity(total)", lambda: _geometry.fix_connectivity(points, faces))
    show("  boundary_edges", lambda: _lib.boundary_edges(f, len(v)))
    show("  face_components", lambda: _lib.face_components(f, len(v)))
    show("  boundary_loops", lambda: _geometry.boundary_loops(f, len(v)))
    show("  full load", lambda: loaded(points, faces))

    points, faces = open_cylinder(256)
    print("=== cylinder 256")
    v, f = loaded(points, faces)._vertices, loaded(points, faces)._faces

    def fill():
        m = loaded(points, faces)
        m.fill_small_boundaries(0, False)

    show("  fill_small_boundaries", fill, 5)

    points, faces = disconnected_tetrahedra(250)
    print("=== 250 tets")

    def rmsc():
        m = loaded(points, faces)
        m.remove_smallest_components()

    show("  remove_smallest_components", rmsc, 5)

    points, faces = open_cylinder(32)
    print("=== cylinder 32 full repair")

    def repair():
        m = loaded(points, faces)
        m.fill_small_boundaries(0, True)
        m.remove_smallest_components()
        m.clean()

    show("  full repair", repair, 5)
    np.zeros(1)


if __name__ == "__main__":
    main()