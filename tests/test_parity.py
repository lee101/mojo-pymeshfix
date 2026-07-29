import inspect

import numpy as np
import pytest

import pymeshfix
from pymeshfix import _lib


TETRA_POINTS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
)
TETRA_FACES = np.array(
    [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32
)
CUBE_POINTS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)
CUBE_WITHOUT_TOP = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def canonical_faces(faces):
    rows = np.sort(np.asarray(faces), axis=1)
    return rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]


def upstream_mesh(upstream, points, faces):
    mesh = upstream.PyTMesh()
    mesh.set_quiet(1)
    mesh.load_array(points, faces)
    return mesh


def test_public_signatures_match_upstream_subset():
    assert str(inspect.signature(pymeshfix.clean_from_arrays)) == (
        "(v, f, verbose: 'bool' = False, joincomp: 'bool' = False, "
        "remove_smallest_components: 'bool' = True) -> 'tuple[np.ndarray, np.ndarray]'"
    )
    assert str(inspect.signature(pymeshfix.MeshFix.fill_holes)) == (
        "(self, n_edges: 'int' = 0, refine: 'bool' = True) -> 'int'"
    )
    assert str(inspect.signature(pymeshfix.MeshFix.clean)) == (
        "(self, max_iters: 'int' = 10, inner_loops: 'int' = 3) -> 'bool'"
    )


def test_closed_tetrahedron_matches_upstream(upstream):
    mojo = pymeshfix.PyTMesh()
    mojo.load_array(TETRA_POINTS, TETRA_FACES)
    reference = upstream_mesh(upstream, TETRA_POINTS, TETRA_FACES)
    assert (mojo.n_points, mojo.n_faces, mojo.n_boundaries) == (
        reference.n_points,
        reference.n_faces,
        reference.n_boundaries,
    )
    assert mojo.clean() is reference.clean()
    assert canonical_faces(mojo.return_faces()).tolist() == canonical_faces(
        reference.return_faces()
    ).tolist()
    assert abs(_lib.signed_volume(*mojo.return_arrays())) == pytest.approx(1 / 6)


def test_square_hole_fill_matches_upstream(upstream):
    mojo = pymeshfix.PyTMesh()
    mojo.load_array(CUBE_POINTS, CUBE_WITHOUT_TOP)
    reference = upstream_mesh(upstream, CUBE_POINTS, CUBE_WITHOUT_TOP)
    assert mojo.n_boundaries == reference.n_boundaries == 1
    assert mojo.fill_small_boundaries(0, False) == reference.fill_small_boundaries(
        0, False
    )
    assert (mojo.n_points, mojo.n_faces, mojo.n_boundaries) == (
        reference.n_points,
        reference.n_faces,
        reference.n_boundaries,
    )
    assert canonical_faces(mojo.return_faces()).tolist() == canonical_faces(
        reference.return_faces()
    ).tolist()
    assert _lib.signed_volume(*mojo.return_arrays()) == pytest.approx(8.0)


def test_hole_edge_limit_matches_upstream(upstream):
    mojo = pymeshfix.PyTMesh()
    mojo.load_array(CUBE_POINTS, CUBE_WITHOUT_TOP)
    reference = upstream_mesh(upstream, CUBE_POINTS, CUBE_WITHOUT_TOP)
    assert mojo.fill_small_boundaries(3) == reference.fill_small_boundaries(3) == 0
    assert mojo.n_boundaries == reference.n_boundaries == 1


def test_remove_smallest_components_matches_upstream(upstream):
    points = np.vstack((TETRA_POINTS, TETRA_POINTS * 0.5 + 3))
    faces = np.vstack((TETRA_FACES, TETRA_FACES + 4))
    mojo = pymeshfix.PyTMesh()
    mojo.load_array(points, faces)
    reference = upstream_mesh(upstream, points, faces)
    assert mojo.remove_smallest_components() == reference.remove_smallest_components()
    assert (mojo.n_points, mojo.n_faces) == (reference.n_points, reference.n_faces)
    assert np.ptp(mojo.return_points(), axis=0) == pytest.approx(
        np.ptp(reference.return_points(), axis=0)
    )


def test_intersection_selection_count_matches_upstream(upstream):
    points = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0],
            [0, -0.5, -1],
            [0, -0.5, 1],
            [0, 1, 0.5],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    mojo = pymeshfix.PyTMesh()
    mojo.load_array(points, faces)
    reference = upstream_mesh(upstream, points, faces)
    assert len(mojo.select_intersecting_triangles()) == len(
        reference.select_intersecting_triangles()
    ) == 2
    assert len(mojo.select_intersecting_triangles(justproper=True)) == len(
        reference.select_intersecting_triangles(50, True)
    ) == 2


def test_adjacent_faces_are_not_self_intersections(upstream):
    mojo = pymeshfix.PyTMesh()
    mojo.load_array(TETRA_POINTS, TETRA_FACES)
    reference = upstream_mesh(upstream, TETRA_POINTS, TETRA_FACES)
    assert len(mojo.select_intersecting_triangles()) == len(
        reference.select_intersecting_triangles()
    ) == 0


def test_intersection_removal_matches_upstream_on_intruding_component(upstream):
    points = np.vstack(
        (
            CUBE_POINTS,
            np.array([[0, -2, 0], [0, 2, 0], [0, 0, 2]], dtype=float),
        )
    )
    faces = np.vstack(
        (
            CUBE_WITHOUT_TOP,
            [[4, 5, 6], [4, 6, 7]],
            [[8, 9, 10]],
        )
    ).astype(np.int32)
    mojo = pymeshfix.PyTMesh()
    mojo.load_array(points, faces)
    reference = upstream_mesh(upstream, points, faces)
    assert len(mojo.select_intersecting_triangles()) == len(
        reference.select_intersecting_triangles()
    ) == 7
    assert mojo.strong_intersection_removal(3) == reference.strong_intersection_removal(
        3
    )
    assert (mojo.n_points, mojo.n_faces, mojo.n_boundaries) == (
        reference.n_points,
        reference.n_faces,
        reference.n_boundaries,
    )
    assert len(mojo.select_intersecting_triangles()) == 0
    assert _lib.signed_volume(*mojo.return_arrays()) == pytest.approx(8.0)


def test_degenerate_removal_matches_upstream_counts(upstream):
    points = np.vstack((TETRA_POINTS, [[0.5, 0, 0]]))
    faces = np.vstack((TETRA_FACES, [[0, 0, 1], [0, 1, 4]])).astype(np.int32)
    mojo = pymeshfix.PyTMesh()
    mojo.load_array(points, faces)
    reference = upstream_mesh(upstream, points, faces)
    assert mojo.strong_degeneracy_removal(3) == reference.strong_degeneracy_removal(3)
    assert (mojo.n_points, mojo.n_faces, mojo.n_boundaries) == (
        reference.n_points,
        reference.n_faces,
        reference.n_boundaries,
    )


def test_clean_from_arrays_matches_upstream_repair_invariants(upstream):
    got_points, got_faces = pymeshfix.clean_from_arrays(
        CUBE_POINTS, CUBE_WITHOUT_TOP
    )
    ref_points, ref_faces = upstream.clean_from_arrays(
        CUBE_POINTS, CUBE_WITHOUT_TOP, False, False, True
    )
    assert (got_points.shape, got_faces.shape) == (ref_points.shape, ref_faces.shape)
    assert np.min(got_points, axis=0) == pytest.approx(np.min(ref_points, axis=0))
    assert np.max(got_points, axis=0) == pytest.approx(np.max(ref_points, axis=0))
    assert abs(_lib.signed_volume(got_points, got_faces)) == pytest.approx(
        abs(_lib.signed_volume(ref_points, ref_faces))
    )


def test_concave_hole_uses_ear_clipping_and_becomes_watertight():
    polygon = np.array(
        [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float
    )
    points = np.vstack(
        (
            np.column_stack((polygon, np.zeros(6))),
            np.column_stack((polygon, np.ones(6))),
        )
    )
    bottom = np.array([[0, 2, 1], [0, 3, 2], [0, 5, 3], [3, 5, 4]], np.int32)
    sides = []
    for i in range(6):
        following = (i + 1) % 6
        sides.extend(((i, following, following + 6), (i, following + 6, i + 6)))
    mesh = pymeshfix.PyTMesh()
    mesh.load_array(points, np.vstack((bottom, np.asarray(sides, np.int32))))
    assert mesh.n_boundaries == 1
    before = mesh.n_faces
    assert mesh.fill_small_boundaries() == 1
    assert mesh.n_faces == before + 4
    assert mesh.n_boundaries == 0
    assert _lib.signed_volume(*mesh.return_arrays()) == pytest.approx(3.0)


def test_face_orientation_is_repaired_without_changing_topology():
    faces = TETRA_FACES.copy()
    faces[2, [1, 2]] = faces[2, [2, 1]]
    mesh = pymeshfix.PyTMesh()
    mesh.load_array(TETRA_POINTS, faces)
    assert mesh.n_boundaries == 0
    assert _lib.signed_volume(*mesh.return_arrays()) == pytest.approx(1 / 6)


def test_duplicate_vertices_and_faces_are_removed():
    points = np.vstack((TETRA_POINTS, TETRA_POINTS[[0]]))
    faces = np.vstack((TETRA_FACES, [[4, 2, 1]], [[1, 2, 0]]))
    mesh = pymeshfix.PyTMesh()
    mesh.load_array(points, faces)
    assert mesh.n_points == 4
    assert mesh.n_faces == 4
    assert mesh.n_boundaries == 0


def test_nonplanar_boundary_is_filled():
    points = CUBE_POINTS.copy()
    points[5, 2] += 0.2
    points[6, 2] -= 0.1
    mesh = pymeshfix.PyTMesh()
    mesh.load_array(points, CUBE_WITHOUT_TOP)
    assert mesh.fill_small_boundaries(0, False) == 1
    assert mesh.n_boundaries == 0
    assert mesh.n_faces == 12


def test_join_closest_components_joins_all_open_components():
    points = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [3, 0, 0], [4, 0, 0], [3, 1, 0],
            [6, 0, 0], [7, 0, 0], [6, 1, 0],
        ],
        dtype=np.float64,
    )
    faces = np.arange(9, dtype=np.int32).reshape(3, 3)
    mesh = pymeshfix.PyTMesh()
    mesh.load_array(points, faces)
    mesh.join_closest_components()
    labels = _lib.face_components(mesh.return_faces(), mesh.n_points)
    assert len(np.unique(labels)) == 1


def test_low_level_properties_and_copy_semantics():
    mesh = pymeshfix.MeshFix(TETRA_POINTS, TETRA_FACES)
    assert mesh.n_boundaries == 0
    assert mesh.v.dtype == np.float64
    assert mesh.f.dtype == np.int32
    points = mesh.points
    points[0] = 99
    assert not np.array_equal(points, mesh.points)


def test_face_components_simd_tail():
    faces = np.arange(15, dtype=np.int32).reshape(5, 3)
    n = len(faces) * 3
    keys = np.empty(n, dtype=np.int64)
    edge_faces = np.empty(n, dtype=np.int32)
    parent = np.empty(len(faces), dtype=np.int32)
    labels = np.empty(len(faces), dtype=np.int32)
    count = _lib.lib().mpf_face_components(
        _lib.addr(faces),
        len(faces),
        15,
        _lib.addr(keys),
        _lib.addr(edge_faces),
        _lib.addr(parent),
        _lib.addr(labels),
    )
    assert count == 5
    assert labels.tolist() == list(range(5))


@pytest.mark.parametrize("n_faces", [32767, 32768])
def test_topology_parallel_threshold(n_faces):
    faces = np.tile(np.array([[0, 1, 2]], dtype=np.int32), (n_faces, 1))
    assert _lib.boundary_edges(faces, 3).shape == (0, 3)
    labels = _lib.face_components(faces, 3)
    assert np.all(labels == labels[0])


@pytest.mark.parametrize(
    "points, faces, message",
    [
        (np.zeros((3, 2)), np.zeros((1, 3)), "Vertex"),
        (np.zeros((3, 3)), np.zeros((3,)), "Face"),
        (np.array([[np.nan, 0, 0]]), np.empty((0, 3)), "non-finite"),
        (np.zeros((3, 3)), np.array([[0, 1, 3]]), "out-of-range"),
    ],
)
def test_input_validation(points, faces, message):
    with pytest.raises(ValueError, match=message):
        pymeshfix.MeshFix(points, faces)


def test_large_indices_are_rejected_before_int32_conversion():
    wrapped = np.array([[0, 1, 2**32 + 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="out-of-range"):
        pymeshfix.MeshFix(np.zeros((3, 3)), wrapped)


def test_non_integer_face_indices_are_not_silently_narrowed():
    with pytest.raises(TypeError, match="integer"):
        pymeshfix.MeshFix(np.zeros((3, 3)), np.array([[0.0, 1.0, 2.0]]))


def test_empty_mesh_avoids_zero_length_ffi_calls():
    mesh = pymeshfix.PyTMesh()
    mesh.load_array(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
    assert mesh.n_boundaries == 0
    assert mesh.clean()
    assert mesh.return_arrays()[1].shape == (0, 3)


def test_low_level_ffi_rejects_wrong_dtype_and_stride():
    faces64 = TETRA_FACES.astype(np.int64)
    with pytest.raises(TypeError, match="int32"):
        _lib.boundary_edges(faces64, len(TETRA_POINTS))
    strided_vertices = np.zeros((8, 3), dtype=np.float64)[::2]
    with pytest.raises(TypeError, match="C-contiguous"):
        _lib.signed_volume(strided_vertices, TETRA_FACES)


def test_array_api_rejects_uncovered_workflows_explicitly(tmp_path):
    mesh = pymeshfix.PyTMesh()
    with pytest.raises(NotImplementedError, match="array-based"):
        mesh.load_file("mesh.obj")
    with pytest.raises(NotImplementedError, match="array-based"):
        mesh.save_file("mesh.obj")
    high_level = pymeshfix.MeshFix(TETRA_POINTS, TETRA_FACES)
    for operation in (
        lambda: high_level.mesh,
        high_level.extract_holes,
        high_level.plot,
        lambda: high_level.save(tmp_path / "mesh.obj"),
        lambda: pymeshfix.clean_from_file("in.obj", "out.obj"),
    ):
        with pytest.raises(NotImplementedError):
            operation()
