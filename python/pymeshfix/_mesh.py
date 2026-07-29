from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _lib
from ._geometry import (
    boundary_loops,
    checked_arrays,
    compact_vertices,
    fix_connectivity as repair_connectivity,
    orient_faces,
    scale_epsilon,
    triangulate_loop,
)


class PyTMesh:
    def __init__(self) -> None:
        self._vertices = np.empty((0, 3), dtype=np.float64)
        self._faces = np.empty((0, 3), dtype=np.int32)
        self._quiet = True

    def load_array(self, points_arr, faces_arr) -> None:
        self._vertices, self._faces = checked_arrays(points_arr, faces_arr)
        self.fix_connectivity()

    def load_file(self, filename: str) -> None:
        raise NotImplementedError("File loading is outside the array-based covered subset")

    def save_file(self, filename: str, back_approx: bool = False) -> None:
        raise NotImplementedError("File saving is outside the array-based covered subset")

    def set_quiet(self, quiet: int) -> None:
        self._quiet = bool(quiet)

    @property
    def n_boundaries(self) -> int:
        return len(boundary_loops(self._faces, len(self._vertices)))

    def boundaries(self) -> int:
        return self.n_boundaries

    @property
    def n_faces(self) -> int:
        return len(self._faces)

    @property
    def n_points(self) -> int:
        return len(self._vertices)

    def return_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.return_points(), self.return_faces()

    def return_points(self) -> np.ndarray:
        return self._vertices.copy()

    def return_faces(self) -> np.ndarray:
        return self._faces.copy()

    def fix_connectivity(self) -> None:
        self._vertices, self._faces = repair_connectivity(
            self._vertices, self._faces
        )

    def fill_small_boundaries(self, nbe: int = 0, refine: bool = True) -> int:
        del refine
        self._vertices, self._faces = orient_faces(self._vertices, self._faces)
        loops = boundary_loops(self._faces, len(self._vertices))
        additions: list[tuple[int, int, int]] = []
        filled = 0
        for loop in loops:
            if nbe > 0 and len(loop) > nbe:
                continue
            additions.extend(triangulate_loop(self._vertices, loop))
            filled += 1
        if additions:
            self._faces = np.ascontiguousarray(
                np.vstack((self._faces, np.asarray(additions, dtype=np.int32)))
            )
            self._vertices, self._faces = orient_faces(
                self._vertices, self._faces
            )
        return filled

    def remove_smallest_components(self) -> int:
        if not len(self._faces):
            return 0
        labels = _lib.face_components(self._faces, len(self._vertices))
        roots, counts = np.unique(labels, return_counts=True)
        winner = roots[int(np.argmax(counts))]
        removed = len(roots) - 1
        self._faces = np.ascontiguousarray(self._faces[labels == winner])
        self._vertices, self._faces = compact_vertices(
            self._vertices, self._faces
        )
        return removed

    def select_intersecting_triangles(
        self, tris_per_cell: int = 50, justproper: bool = False
    ) -> np.ndarray:
        del tris_per_cell
        if not len(self._faces):
            return np.empty((0, 3), dtype=np.int32)
        mask = _lib.intersection_mask(
            self._vertices,
            self._faces,
            scale_epsilon(self._vertices),
            justproper,
        )
        return self._faces[mask].copy()

    def strong_degeneracy_removal(self, max_iter: int) -> bool:
        for _ in range(max(1, int(max_iter))):
            if not len(self._faces):
                return True
            epsilon = scale_epsilon(self._vertices)
            mask = _lib.degenerate_mask(
                self._vertices, self._faces, epsilon * epsilon
            )
            canonical = np.sort(self._faces, axis=1)
            _, first = np.unique(canonical, axis=0, return_index=True)
            duplicates = np.ones(len(self._faces), dtype=bool)
            duplicates[first] = False
            mask |= duplicates
            if not mask.any():
                return True
            self._faces = np.ascontiguousarray(self._faces[~mask])
            self._vertices, self._faces = compact_vertices(
                self._vertices, self._faces
            )
        return not _lib.degenerate_mask(
            self._vertices,
            self._faces,
            scale_epsilon(self._vertices) ** 2,
        ).any()

    def strong_intersection_removal(self, max_iter: int) -> bool:
        for _ in range(max(1, int(max_iter))):
            if not len(self._faces):
                return True
            mask = _lib.intersection_mask(
                self._vertices,
                self._faces,
                scale_epsilon(self._vertices),
            )
            if not mask.any():
                return True
            labels = _lib.face_components(self._faces, len(self._vertices))
            marked_roots = np.unique(labels[mask])
            if len(marked_roots) > 1:
                component_sizes = {
                    int(root): int(np.count_nonzero(labels == root))
                    for root in marked_roots
                }
                winner = max(component_sizes, key=component_sizes.get)
                discard_roots = marked_roots[marked_roots != winner]
                self._faces = np.ascontiguousarray(
                    self._faces[~np.isin(labels, discard_roots)]
                )
                self._vertices, self._faces = compact_vertices(
                    self._vertices, self._faces
                )
                continue
            keep = ~mask
            if not keep.any():
                return False
            self._faces = np.ascontiguousarray(self._faces[keep])
            self._vertices, self._faces = compact_vertices(
                self._vertices, self._faces
            )
            self.fill_small_boundaries(0, False)
            self.strong_degeneracy_removal(1)
        if not len(self._faces):
            return False
        return not _lib.intersection_mask(
            self._vertices,
            self._faces,
            scale_epsilon(self._vertices),
        ).any()

    def clean(self, max_iters: int = 10, inner_loops: int = 3) -> bool:
        for _ in range(max(1, int(max_iters))):
            degeneracies_ok = self.strong_degeneracy_removal(inner_loops)
            intersections_ok = self.strong_intersection_removal(inner_loops)
            if degeneracies_ok and intersections_ok:
                return True
        return False

    def join_closest_components(self) -> None:
        if not len(self._faces):
            return
        while True:
            labels = _lib.face_components(self._faces, len(self._vertices))
            roots = np.unique(labels)
            if len(roots) < 2:
                return
            component_loops: list[tuple[int, list[int]]] = []
            for root in roots:
                component_faces = np.ascontiguousarray(self._faces[labels == root])
                for loop in boundary_loops(component_faces, len(self._vertices)):
                    component_loops.append((int(root), loop))
            if len(component_loops) < 2:
                return
            best = None
            for i, (root_a, loop_a) in enumerate(component_loops):
                for root_b, loop_b in component_loops[i + 1 :]:
                    if root_a == root_b:
                        continue
                    pa = self._vertices[loop_a]
                    pb = self._vertices[loop_b]
                    distances = np.sum((pa[:, None, :] - pb[None, :, :]) ** 2, axis=2)
                    flat = int(np.argmin(distances))
                    ia, ib = np.unravel_index(flat, distances.shape)
                    candidate = (float(distances[ia, ib]), loop_a, loop_b, ia, ib)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            if best is None:
                break
            _, loop_a, loop_b, ia, ib = best
            a0, a1 = loop_a[ia], loop_a[(ia + 1) % len(loop_a)]
            b0, b1 = loop_b[ib], loop_b[(ib + 1) % len(loop_b)]
            bridge = np.asarray(((a1, a0, b0), (a1, b0, b1)), dtype=np.int32)
            self._faces = np.ascontiguousarray(np.vstack((self._faces, bridge)))
            self._vertices, self._faces = orient_faces(
                self._vertices, self._faces
            )


class MeshFix:
    def __init__(self, *args, verbose: bool = False):
        if len(args) != 2:
            raise TypeError("MeshFix requires a vertex array and a face array")
        self._mfix = PyTMesh()
        self._mfix.set_quiet(not verbose)
        self.load_arrays(args[0], args[1])

    def load_arrays(self, v, f) -> None:
        self._mfix.load_array(v, f)

    @property
    def points(self) -> np.ndarray:
        return self._mfix.return_points()

    @property
    def faces(self) -> np.ndarray:
        return self._mfix.return_faces()

    @property
    def v(self) -> np.ndarray:
        return self.points

    @property
    def f(self) -> np.ndarray:
        return self.faces

    @property
    def n_boundaries(self) -> int:
        return self._mfix.n_boundaries

    def fill_holes(self, n_edges: int = 0, refine: bool = True) -> int:
        return self._mfix.fill_small_boundaries(n_edges, refine)

    def remove_smallest_components(self) -> None:
        self._mfix.remove_smallest_components()

    def join_closest_components(self) -> None:
        self._mfix.join_closest_components()

    def degeneracy_removal(self, max_iter: int = 3) -> bool:
        return self._mfix.strong_degeneracy_removal(max_iter)

    def intersection_removal(self, max_iter: int = 3) -> bool:
        return self._mfix.strong_intersection_removal(max_iter)

    def clean(self, max_iters: int = 10, inner_loops: int = 3) -> bool:
        return self._mfix.clean(max_iters, inner_loops)

    def repair(
        self, joincomp: bool = False, remove_smallest_components: bool = True
    ) -> None:
        self.fill_holes(0, True)
        if joincomp:
            self.join_closest_components()
        if remove_smallest_components:
            self.remove_smallest_components()
        self.clean()

    @property
    def mesh(self):
        raise NotImplementedError("PyVista integration is outside the covered subset")

    def extract_holes(self):
        raise NotImplementedError("PyVista integration is outside the covered subset")

    def plot(self, show_holes: bool = True, **kwargs):
        raise NotImplementedError("Plotting is outside the covered subset")

    def save(self, filename: str | Path, binary=True):
        raise NotImplementedError("File saving is outside the covered subset")


def clean_from_arrays(
    v,
    f,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = MeshFix(v, f, verbose=verbose)
    mesh.repair(
        joincomp=joincomp,
        remove_smallest_components=remove_smallest_components,
    )
    return mesh.points, mesh.faces


def clean_from_file(
    infile: str,
    outfile: str,
    verbose: bool = False,
    joincomp: bool = False,
) -> None:
    raise NotImplementedError("File workflows are outside the array-based covered subset")
