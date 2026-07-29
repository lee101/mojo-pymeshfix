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
    used = np.unique(faces)
    valid = used[(used >= 0) & (used < len(vertices))]
    if (
        len(valid) == len(vertices)
        and valid[0] == 0
        and valid[-1] == len(vertices) - 1
    ):
        return vertices, np.ascontiguousarray(faces, dtype=np.int32)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[valid] = np.arange(len(valid), dtype=np.int32)
    return (
        np.ascontiguousarray(vertices[valid]),
        np.ascontiguousarray(remap[faces], dtype=np.int32),
    )


def fix_connectivity(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not len(vertices):
        return vertices, faces
    _, first_vertices, inverse = np.unique(
        vertices, axis=0, return_index=True, return_inverse=True
    )
    vertex_order = np.argsort(first_vertices)
    first_vertices = first_vertices[vertex_order]
    old_to_new = np.empty(len(vertex_order), dtype=np.int32)
    old_to_new[vertex_order] = np.arange(len(vertex_order), dtype=np.int32)
    remap = old_to_new[inverse]
    new_vertices = np.ascontiguousarray(vertices[first_vertices])
    welded = remap[faces] if len(faces) else faces.copy()
    if len(welded):
        canonical = np.sort(welded, axis=1)
        _, first_faces = np.unique(canonical, axis=0, return_index=True)
        first_faces.sort()
        new_faces = np.ascontiguousarray(welded[first_faces], dtype=np.int32)
    else:
        new_faces = np.empty((0, 3), dtype=np.int32)
    return orient_faces(*compact_vertices(new_vertices, new_faces))


def orient_faces(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not len(faces):
        return vertices, faces
    us = faces.reshape(-1)
    vs = faces[:, (1, 2, 0)].reshape(-1)
    owners = np.repeat(np.arange(len(faces), dtype=np.int32), 3)
    keys = (
        np.minimum(us, vs).astype(np.int64) * np.int64(len(vertices))
        + np.maximum(us, vs)
    )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_keys)])
    pair_starts = starts[counts == 2]
    first = order[pair_starts]
    second = order[pair_starts + 1]
    same_direction = (us[first] == us[second]) & (vs[first] == vs[second])
    oriented = faces.copy()
    if np.any(same_direction):
        graph: list[list[tuple[int, bool]]] = [[] for _ in faces]
        for left, right, toggle in zip(
            owners[first], owners[second], same_direction, strict=True
        ):
            graph[int(left)].append((int(right), bool(toggle)))
            graph[int(right)].append((int(left), bool(toggle)))
        flips = np.full(len(faces), -1, dtype=np.int8)
        for root in range(len(faces)):
            if flips[root] >= 0:
                continue
            flips[root] = 0
            queue = deque([root])
            while queue:
                current = queue.popleft()
                for neighbor, toggle in graph[current]:
                    wanted = int(flips[current]) ^ int(toggle)
                    if flips[neighbor] < 0:
                        flips[neighbor] = wanted
                        queue.append(neighbor)
        selected = np.flatnonzero(flips == 1)
        oriented[selected, 1], oriented[selected, 2] = (
            oriented[selected, 2].copy(),
            oriented[selected, 1].copy(),
        )

    labels = _lib.face_components(oriented, len(vertices))
    boundary_starts = starts[counts == 1]
    open_labels = (
        np.unique(labels[owners[order[boundary_starts]]])
        if len(boundary_starts)
        else np.empty(0, dtype=np.int32)
    )
    roots = np.unique(labels)
    closed_labels = roots[~np.isin(roots, open_labels)]
    if len(closed_labels) == 1 and len(roots) == 1:
        negative_labels = (
            closed_labels
            if _lib.signed_volume(vertices, oriented) < 0
            else np.empty(0, dtype=np.int32)
        )
    elif len(closed_labels):
        a = vertices[oriented[:, 0]]
        b = vertices[oriented[:, 1]]
        c = vertices[oriented[:, 2]]
        terms = np.einsum("ij,ij->i", a, np.cross(b, c))
        root_values, dense_labels = np.unique(labels, return_inverse=True)
        volumes = np.bincount(dense_labels, weights=terms)
        negative_labels = root_values[
            (volumes < 0) & np.isin(root_values, closed_labels)
        ]
    else:
        negative_labels = np.empty(0, dtype=np.int32)
    selected = np.flatnonzero(np.isin(labels, negative_labels))
    oriented[selected, 1], oriented[selected, 2] = (
        oriented[selected, 2].copy(),
        oriented[selected, 1].copy(),
    )
    return vertices, np.ascontiguousarray(oriented)


def boundary_loops(faces: np.ndarray, n_vertices: int) -> list[list[int]]:
    edges = _lib.boundary_edges(faces, n_vertices)
    if not len(edges):
        return []
    outgoing: dict[int, list[int]] = defaultdict(list)
    undirected: dict[int, list[int]] = defaultdict(list)
    for u, v, _ in edges:
        outgoing[int(u)].append(int(v))
        undirected[int(u)].append(int(v))
        undirected[int(v)].append(int(u))
    unused = {(int(u), int(v)) for u, v, _ in edges}
    loops: list[list[int]] = []
    while unused:
        start_edge = next(iter(unused))
        start, current = start_edge
        loop = [start]
        unused.remove(start_edge)
        while current != start and len(loop) <= len(edges):
            loop.append(current)
            candidates = [v for v in outgoing.get(current, ()) if (current, v) in unused]
            if candidates:
                nxt = candidates[0]
                unused.remove((current, nxt))
            else:
                candidates = [
                    v
                    for v in undirected.get(current, ())
                    if (current, v) in unused or (v, current) in unused
                ]
                if not candidates:
                    break
                nxt = candidates[0]
                unused.discard((current, nxt))
                unused.discard((nxt, current))
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
