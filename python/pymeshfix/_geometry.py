from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from . import _lib


def checked_arrays(points, faces) -> tuple[np.ndarray, np.ndarray]:
    raw_vertices = np.asarray(points)
    raw_faces = np.asarray(faces)
    if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3:
        raise ValueError("Vertex array must have shape (n, 3)")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError("Face array must have shape (m, 3)")
    if len(raw_vertices) > np.iinfo(np.int32).max:
        raise OverflowError("Vertex count exceeds the int32 mesh index limit")
    if len(raw_faces) > np.iinfo(np.int32).max:
        raise OverflowError("Face count exceeds the int32 ABI limit")
    try:
        vertices = np.ascontiguousarray(raw_vertices, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Vertex array must contain numeric coordinates") from error
    if not np.isfinite(vertices).all():
        raise ValueError("Vertex array contains non-finite coordinates")
    if raw_faces.dtype.kind not in "iu":
        raise TypeError("Face array must contain integer indices")
    if len(raw_faces):
        minimum = int(raw_faces.min())
        maximum = int(raw_faces.max())
        if minimum < 0 or maximum >= len(vertices):
            raise ValueError("Face array contains an out-of-range vertex index")
        if maximum > np.iinfo(np.int32).max:
            raise OverflowError("Face index exceeds the int32 ABI limit")
    triangles = np.ascontiguousarray(raw_faces, dtype=np.int32)
    return vertices.copy(), triangles.copy()


def scale_epsilon(vertices: np.ndarray) -> float:
    if not len(vertices):
        return 1e-12
    scale = float(np.max(np.ptp(vertices, axis=0)))
    return max(scale * 1e-12, 1e-14)


def compact_vertices(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not len(faces):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)
    used = np.zeros(len(vertices), dtype=bool)
    used[faces] = True
    if used.all():
        return vertices, np.ascontiguousarray(faces, dtype=np.int32)
    valid = np.flatnonzero(used)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[valid] = np.arange(len(valid), dtype=np.int32)
    return (
        np.ascontiguousarray(vertices[valid]),
        np.ascontiguousarray(remap[faces], dtype=np.int32),
    )


def _float_key(column: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(column).view(np.int64)
    return bits ^ ((bits >> 63) & 0x7FFFFFFFFFFFFFFF)


def _run_starts(first: np.ndarray, second: np.ndarray, third: np.ndarray):
    if not len(first):
        return np.empty(0, dtype=np.int64)
    changed = np.empty(len(first), dtype=bool)
    changed[0] = True
    np.logical_or(
        first[1:] != first[:-1],
        np.logical_or(second[1:] != second[:-1], third[1:] != third[:-1]),
        out=changed[1:],
    )
    return np.flatnonzero(changed)


def weld_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _float_key(vertices[:, 0])
    y = _float_key(vertices[:, 1])
    z = _float_key(vertices[:, 2])
    order = np.lexsort((z, y, x))
    starts = _run_starts(x[order], y[order], z[order])
    group = np.zeros(len(vertices), dtype=np.int32)
    group[starts] = 1
    np.cumsum(group, out=group)
    group -= 1
    remap = np.empty(len(vertices), dtype=np.int32)
    remap[order] = group
    first = np.minimum.reduceat(order, starts)
    ranking = np.argsort(first, kind="stable")
    new_vertices = np.ascontiguousarray(vertices[first[ranking]])
    old_to_new = np.empty(len(starts), dtype=np.int32)
    old_to_new[ranking] = np.arange(len(starts), dtype=np.int32)
    return new_vertices, old_to_new[remap]


def _canonical_columns(faces: np.ndarray):
    low = faces.min(axis=1)
    high = faces.max(axis=1)
    return low, faces.sum(axis=1, dtype=np.int64) - low - high, high


def first_occurrence_indices(faces: np.ndarray) -> np.ndarray:
    low, mid, high = _canonical_columns(faces)
    order = np.lexsort((high, mid, low))
    starts = _run_starts(low[order], mid[order], high[order])
    return np.minimum.reduceat(order, starts)


def dedupe_faces(faces: np.ndarray) -> np.ndarray:
    keep = first_occurrence_indices(faces)
    keep.sort()
    return np.ascontiguousarray(faces[keep], dtype=np.int32)


def fix_connectivity(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not len(vertices):
        return vertices, faces
    new_vertices, remap = weld_vertices(vertices)
    welded = remap[faces] if len(faces) else np.empty((0, 3), dtype=np.int32)
    new_faces = dedupe_faces(welded) if len(welded) else welded
    return orient_faces(*compact_vertices(new_vertices, new_faces))


def orient_faces(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not len(faces):
        return vertices, faces
    return vertices, _lib.orient_faces(vertices, faces)


def boundary_loops(faces: np.ndarray, n_vertices: int) -> list[list[int]]:
    edges = _lib.boundary_edges(faces, n_vertices)
    if not len(edges):
        return []
    starts = edges[:, 0].tolist()
    ends = edges[:, 1].tolist()
    outgoing: dict[int, list[int]] = defaultdict(list)
    undirected: dict[int, list[int]] = defaultdict(list)
    for u, v in zip(starts, ends):
        outgoing[u].append(v)
        undirected[u].append(v)
        undirected[v].append(u)
    unused = set(zip(starts, ends))
    cursor: dict[int, int] = {}
    limit = len(edges)
    loops: list[list[int]] = []
    while unused:
        start, current = next(iter(unused))
        loop = [start]
        unused.remove((start, current))
        while current != start and len(loop) <= limit:
            loop.append(current)
            options = outgoing.get(current)
            nxt = None
            if options:
                index = cursor.get(current, 0)
                while index < len(options):
                    candidate = options[index]
                    if (current, candidate) in unused:
                        break
                    index += 1
                cursor[current] = index
                if index < len(options):
                    nxt = options[index]
                    unused.remove((current, nxt))
            if nxt is None:
                for candidate in undirected.get(current, ()):
                    if (current, candidate) in unused or (
                        candidate,
                        current,
                    ) in unused:
                        nxt = candidate
                        unused.discard((current, candidate))
                        unused.discard((candidate, current))
                        break
            if nxt is None:
                break
            current = nxt
        if current == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _point_in_triangle(point, a, b, c, orientation: float, tolerance: float) -> bool:
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return (
        orientation * cross(a, b, point) >= -tolerance
        and orientation * cross(b, c, point) >= -tolerance
        and orientation * cross(c, a, point) >= -tolerance
    )


def triangulate_loop(vertices: np.ndarray, loop: list[int]) -> list[tuple[int, int, int]]:
    polygon = list(reversed(loop))
    points3 = vertices[polygon]
    normal = np.sum(np.cross(points3, np.roll(points3, -1, axis=0)), axis=0)
    drop_axis = int(np.argmax(np.abs(normal)))
    points2 = np.delete(points3, drop_axis, axis=1)
    signed_area = float(
        np.sum(
            points2[:, 0] * np.roll(points2[:, 1], -1)
            - np.roll(points2[:, 0], -1) * points2[:, 1]
        )
    )
    orientation = 1.0 if signed_area >= 0 else -1.0
    tolerance = max(float(np.max(np.ptp(points2, axis=0))) ** 2 * 1e-14, 1e-15)
    previous = np.roll(points2, 1, axis=0)
    following = np.roll(points2, -1, axis=0)
    turns = (points2[:, 0] - previous[:, 0]) * (
        following[:, 1] - points2[:, 1]
    ) - (points2[:, 1] - previous[:, 1]) * (
        following[:, 0] - points2[:, 0]
    )
    if np.all(orientation * turns > tolerance):
        root = polygon[-1]
        return [
            (root, polygon[index], polygon[index + 1])
            for index in range(len(polygon) - 2)
        ]
    remaining = list(range(len(polygon)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        clipped = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = points2[previous], points2[current], points2[following]
            turn = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (
                c[0] - b[0]
            )
            if orientation * turn <= tolerance:
                continue
            if any(
                _point_in_triangle(points2[item], a, b, c, orientation, tolerance)
                for item in remaining
                if item not in (previous, current, following)
            ):
                continue
            triangles.append(
                (polygon[previous], polygon[current], polygon[following])
            )
            del remaining[position]
            clipped = True
            break
        if not clipped:
            root = remaining[0]
            triangles.extend(
                (polygon[root], polygon[remaining[i]], polygon[remaining[i + 1]])
                for i in range(1, len(remaining) - 1)
            )
            return triangles
    triangles.append(tuple(polygon[index] for index in remaining))
    return triangles
