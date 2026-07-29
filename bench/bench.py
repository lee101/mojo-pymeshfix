"""Benchmarks against the real pymeshfix 0.18.1 extension."""

from __future__ import annotations

import glob
import importlib.util
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import pymeshfix as mojo_meshfix  # noqa: E402


def load_upstream():
    pattern = os.path.join(
        sys.prefix, "lib", "python*", "site-packages", "pymeshfix", "_meshfix*.so"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise RuntimeError("real upstream pymeshfix is required for this benchmark")
    spec = importlib.util.spec_from_file_location("_meshfix", matches[0])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def timeit(function, repetitions=5):
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def grid(size: int):
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="xy")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(size * size)))
    a = np.arange((size - 1) * (size - 1), dtype=np.int32)
    row = a // (size - 1)
    col = a % (size - 1)
    v0 = row * size + col
    faces = np.vstack(
        (
            np.column_stack((v0, v0 + 1, v0 + size + 1)),
            np.column_stack((v0, v0 + size + 1, v0 + size)),
        )
    )
    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(
        faces, dtype=np.int32
    )


def open_cylinder(segments: int):
    angles = np.arange(segments) * (2 * np.pi / segments)
    bottom = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(segments)))
    top = bottom.copy()
    top[:, 2] = 1
    points = np.vstack((bottom, top, [[0, 0, 0]]))
    faces = []
    for i in range(segments):
        following = (i + 1) % segments
        faces.extend(
            (
                (i, following, following + segments),
                (i, following + segments, i + segments),
                (2 * segments, following, i),
            )
        )
    return points.astype(np.float64), np.asarray(faces, dtype=np.int32)


def disconnected_tetrahedra(count: int):
    base_points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    base_faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32
    )
    points = np.vstack(
        [base_points * (1.0 if i == 0 else 0.5) + [i * 3, 0, 0] for i in range(count)]
    )
    faces = np.vstack([base_faces + i * 4 for i in range(count)])
    return points, faces


def new_mesh(module, points, faces):
    mesh = module.PyTMesh()
    mesh.set_quiet(1)
    mesh.load_array(points, faces)
    return mesh


def cpu_name():
    try:
        for line in open("/proc/cpuinfo", encoding="utf8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    upstream = load_upstream()
    rows = []

    points, faces = grid(160)

    def mojo_boundary():
        return new_mesh(mojo_meshfix, points, faces).n_boundaries

    def upstream_boundary():
        return new_mesh(upstream, points, faces).n_boundaries

    mojo_time, mojo_result = timeit(mojo_boundary, 3)
    upstream_time, upstream_result = timeit(upstream_boundary, 3)
    assert mojo_result == upstream_result == 1
    rows.append(("load + boundaries, 50k faces", mojo_time, upstream_time))

    points, faces = open_cylinder(256)

    def mojo_fill():
        mesh = new_mesh(mojo_meshfix, points, faces)
        filled = mesh.fill_small_boundaries(0, False)
        return filled, mesh.n_faces, mesh.n_boundaries

    def upstream_fill():
        mesh = new_mesh(upstream, points, faces)
        filled = mesh.fill_small_boundaries(0, False)
        return filled, mesh.n_faces, mesh.n_boundaries

    mojo_time, mojo_result = timeit(mojo_fill, 3)
    upstream_time, upstream_result = timeit(upstream_fill, 3)
    assert mojo_result == upstream_result == (1, 1022, 0)
    rows.append(("fill 256-edge hole", mojo_time, upstream_time))

    pair = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0],
            [0, -0.5, -1],
            [0, -0.5, 1],
            [0, 1, 0.5],
        ],
        dtype=float,
    )
    points = np.vstack([pair + [4 * i, 0, 0] for i in range(100)])
    faces = np.arange(600, dtype=np.int32).reshape(-1, 3)
    mojo_mesh = new_mesh(mojo_meshfix, points, faces)
    upstream_mesh = new_mesh(upstream, points, faces)
    mojo_time, mojo_result = timeit(
        lambda: len(mojo_mesh.select_intersecting_triangles()), 5
    )
    upstream_time, upstream_result = timeit(
        lambda: len(upstream_mesh.select_intersecting_triangles()), 5
    )
    assert mojo_result == upstream_result
    assert mojo_result == 200
    rows.append(("select intersections, 200 tris", mojo_time, upstream_time))

    points, faces = disconnected_tetrahedra(250)

    def mojo_components():
        mesh = new_mesh(mojo_meshfix, points, faces)
        removed = mesh.remove_smallest_components()
        return removed, mesh.n_faces

    def upstream_components():
        mesh = new_mesh(upstream, points, faces)
        removed = mesh.remove_smallest_components()
        return removed, mesh.n_faces

    mojo_time, mojo_result = timeit(mojo_components, 3)
    upstream_time, upstream_result = timeit(upstream_components, 3)
    assert mojo_result == upstream_result == (249, 4)
    rows.append(("keep largest of 250 shells", mojo_time, upstream_time))

    points, faces = open_cylinder(32)

    def mojo_repair():
        mesh = new_mesh(mojo_meshfix, points, faces)
        mesh.fill_small_boundaries(0, True)
        mesh.remove_smallest_components()
        mesh.clean()
        vertices, triangles = mesh.return_arrays()
        volume = abs(
            np.einsum(
                "ij,ij->i",
                vertices[triangles[:, 0]],
                np.cross(
                    vertices[triangles[:, 1]], vertices[triangles[:, 2]]
                ),
            ).sum()
            / 6
        )
        return mesh.n_boundaries, volume

    def upstream_repair():
        mesh = new_mesh(upstream, points, faces)
        mesh.fill_small_boundaries(0, True)
        mesh.remove_smallest_components()
        mesh.clean()
        vertices, triangles = mesh.return_arrays()
        volume = abs(
            np.einsum(
                "ij,ij->i",
                vertices[triangles[:, 0]],
                np.cross(
                    vertices[triangles[:, 1]], vertices[triangles[:, 2]]
                ),
            ).sum()
            / 6
        )
        return mesh.n_boundaries, volume

    mojo_time, mojo_result = timeit(mojo_repair, 3)
    upstream_time, upstream_result = timeit(upstream_repair, 3)
    assert mojo_result[0] == upstream_result[0] == 0
    assert np.isclose(mojo_result[1], upstream_result[1], rtol=1e-6)
    rows.append(("full repair, 96-face cylinder", mojo_time, upstream_time))

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| Operation | Mojo port | pymeshfix 0.18.1 | Mojo speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_time, upstream_time in rows:
        speedup = upstream_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
