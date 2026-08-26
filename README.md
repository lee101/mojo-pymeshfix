# mojo-pymeshfix

`mojo-pymeshfix` is an independent Mojo implementation of the compute-heavy
parts of [pymeshfix](https://github.com/pyvista/pymeshfix) for triangular meshes.
It provides the upstream `MeshFix`, `PyTMesh`, and `clean_from_arrays` names for
the covered in-memory workflow. The goal is useful mesh repair, not a binding to
the upstream C++ library.

## Coverage

The covered subset is:

- validation, exact duplicate-vertex welding, duplicate-face removal, and
  consistent face orientation;
- boundary-loop discovery and `n_boundaries`;
- planar and non-planar hole filling with projected 3D ear clipping, including
  concave boundary loops and the upstream `n_edges` limit;
- edge-connected face components and removal of all but the largest shell;
- coplanar and non-coplanar triangle intersection selection;
- self-intersection repair by removing smaller intersecting components, or by
  excising locally intersecting faces and retriangulating their boundaries;
- degenerate triangle removal;
- the `clean` and `repair` pipelines; and
- array properties and methods on both `MeshFix` and `PyTMesh`.

The public signatures mirror pymeshfix 0.18.1 for this subset. `refine` is
accepted by the hole-filling methods, but this port does not add Steiner points
or perform MeshFix's filled-region refinement. Joining is limited to components
with open boundaries; it does not cut and join already-watertight shells.
PyVista objects, plotting, hole extraction as `PolyData`, and file load/save
workflows are not covered and raise `NotImplementedError`.

The algorithms use scale-aware floating-point predicates rather than
Shewchuk-style exact predicates. Extremely ill-conditioned or globally tangled
meshes may therefore need a full exact-predicate remesher. Output topology can
differ from upstream even when both outputs are watertight and geometrically
equivalent.

## Install

The Pixi environment includes the pinned Mojo nightly, NumPy, pytest, and the
real pymeshfix 0.18.1 package used by the parity suite.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-pymeshfix.so`.

## Usage

This example is also exercised by the tests. It closes the missing top of a
cube with two triangles.

```python
import numpy as np
from pymeshfix import MeshFix

points = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
], dtype=np.float64)

faces = np.array([
    [0, 2, 1], [0, 3, 2],
    [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6],
    [3, 0, 4], [3, 4, 7],
], dtype=np.int32)

mesh = MeshFix(points, faces)
assert mesh.n_boundaries == 1
assert mesh.fill_holes(refine=False) == 1
assert mesh.n_boundaries == 0
assert mesh.faces.shape == (12, 3)
```

For the default all-in-one workflow:

```python
from pymeshfix import clean_from_arrays

clean_points, clean_faces = clean_from_arrays(points, faces)
```

## Benchmarks

Measured with `pixi run bench` on this machine's Intel Xeon E5-2697 v4,
Linux x86-64. Times are the best of three or five runs. The comparison is the
real conda-forge pymeshfix 0.18.1 extension on the same arrays.

| Operation | Mojo port | pymeshfix 0.18.1 | Mojo speedup |
|---|---:|---:|---:|
| load + boundaries, 50k faces | 75.187 ms | 136.076 ms | 1.81x |
| fill 256-edge hole | 5.393 ms | 20.992 ms | 3.89x |
| select intersections, 200 tris | 0.109 ms | 0.498 ms | 4.55x |
| keep largest of 250 shells | 2.501 ms | 4.332 ms | 1.73x |
| full repair, 96-face cylinder | 2.636 ms | 2.891 ms | 1.10x |

The port is ahead on every measured operation. Connectivity welding,
orientation, and closed-component volume checks run in bulk instead of
face-by-face Python loops, and convex boundaries use a linear fan
triangulation fast path while concave boundaries retain ear clipping. These
are measured results, not projected performance.

GPU acceleration is intentionally not included. The profiled targets are
edge sorting, union-find, boundary traversal, and small branch-heavy polygon
work; they do not have enough arithmetic intensity to offset device transfers
and launch overhead.

## How it works

All kernels live in one Mojo compilation unit and are exported through a C ABI.
The Python layer loads the shared object with `ctypes`. NumPy buffers cross the
ABI as integer addresses, matching the pinned Mojo nightly's non-parametric
export requirements.

Vertices are C-contiguous row-major `float64` arrays with shape `(n, 3)`;
triangles are C-contiguous row-major `int32` arrays with shape `(m, 3)`.
Scratch arrays for edge keys, owners, union-find parents, and result flags are
allocated by NumPy and owned by Python. Mojo does not allocate or retain
buffers. Before each call, the bridge verifies shapes, exact dtypes,
C-contiguity, index ranges, ABI size limits, and non-null storage for non-empty
arrays. Python keeps every NumPy owner alive until the synchronous call returns;
empty work is handled without constructing Mojo pointers.

Boundary classification and connected components sort an index array over
packed undirected edge keys, avoiding multi-array swaps. Large independent
edge-preparation passes use up to eight CPU workers at a 65,536-face threshold,
while smaller meshes stay serial. Component setup and counting and signed-volume
reduction use native-width SIMD with scalar remainder handling. Intersection
selection precomputes face bounds, sweeps triangles in minimum-x order, and then
uses segment-triangle predicates or dominant-axis 2D predicates for coplanar
pairs. The Python topology layer walks oriented boundary loops and uses a convex
fan fast path before falling back to ear clipping.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The tests load upstream's native extension under a separate module name and
assert counts, topology, watertightness, volume, intersection behavior, and
return-value parity. Additional reference tests cover concave holes and input
validation.
