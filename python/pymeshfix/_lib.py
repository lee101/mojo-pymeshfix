from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYMESHFIX_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pymeshfix.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpf_boundary_edges": ([I, I, I, I, I, I, I], I),
    "mpf_face_components": ([I, I, I, I, I, I, I], I),
    "mpf_mark_degenerate": ([I, I, I, I, F, I], I),
    "mpf_mark_intersections": ([I, I, I, F, I, I], I),
    "mpf_signed_volume": ([I, I, I], F),
}


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    address = int(array.ctypes.data)
    if array.size and not address:
        raise RuntimeError("NumPy returned a null pointer for a non-empty buffer")
    return address


def _mesh_buffers(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if (
        vertices.dtype != np.float64
        or vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not vertices.flags.c_contiguous
    ):
        raise TypeError("vertices must be a C-contiguous float64 array of shape (n, 3)")
    if (
        faces.dtype != np.int32
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not faces.flags.c_contiguous
    ):
        raise TypeError("faces must be a C-contiguous int32 array of shape (m, 3)")
    if len(vertices) > np.iinfo(np.int32).max or len(faces) > np.iinfo(np.int32).max:
        raise OverflowError("mesh size exceeds the int32 ABI limit")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces contain an out-of-range vertex index")
    return vertices, faces


def _faces_buffer(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    if (
        faces.dtype != np.int32
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not faces.flags.c_contiguous
    ):
        raise TypeError("faces must be a C-contiguous int32 array of shape (m, 3)")
    if not 0 <= n_vertices <= np.iinfo(np.int32).max:
        raise OverflowError("vertex count exceeds the int32 mesh index limit")
    if len(faces) > np.iinfo(np.int32).max:
        raise OverflowError("face count exceeds the int32 ABI limit")
    if len(faces) and (faces.min() < 0 or faces.max() >= n_vertices):
        raise ValueError("faces contain an out-of-range vertex index")
    return faces


def boundary_edges(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    _faces_buffer(faces, n_vertices)
    if not len(faces):
        return np.empty((0, 3), dtype=np.int32)
    n = len(faces) * 3
    keys = np.empty(n, dtype=np.int64)
    us = np.empty(n, dtype=np.int32)
    vs = np.empty(n, dtype=np.int32)
    owners = np.empty(n, dtype=np.int32)
    count = lib().mpf_boundary_edges(
        addr(faces), len(faces), n_vertices, addr(keys), addr(us), addr(vs), addr(owners)
    )
    return np.column_stack((us[:count], vs[:count], owners[:count])).astype(
        np.int32, copy=False
    )


def face_components(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    _faces_buffer(faces, n_vertices)
    if not len(faces):
        return np.empty(0, dtype=np.int32)
    n = len(faces) * 3
    keys = np.empty(n, dtype=np.int64)
    edge_faces = np.empty(n, dtype=np.int32)
    parent = np.empty(len(faces), dtype=np.int32)
    labels = np.empty(len(faces), dtype=np.int32)
    lib().mpf_face_components(
        addr(faces),
        len(faces),
        n_vertices,
        addr(keys),
        addr(edge_faces),
        addr(parent),
        addr(labels),
    )
    return labels


def degenerate_mask(
    vertices: np.ndarray, faces: np.ndarray, area_epsilon: float
) -> np.ndarray:
    _mesh_buffers(vertices, faces)
    if not len(faces):
        return np.empty(0, dtype=bool)
    if not np.isfinite(area_epsilon) or area_epsilon < 0:
        raise ValueError("area_epsilon must be finite and non-negative")
    flags = np.empty(len(faces), dtype=np.int32)
    lib().mpf_mark_degenerate(
        addr(vertices),
        addr(faces),
        len(vertices),
        len(faces),
        area_epsilon,
        addr(flags),
    )
    return flags.astype(bool)


def intersection_mask(
    vertices: np.ndarray, faces: np.ndarray, epsilon: float, justproper: bool = False
) -> np.ndarray:
    _mesh_buffers(vertices, faces)
    if not len(faces):
        return np.empty(0, dtype=bool)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    flags = np.empty(len(faces), dtype=np.int32)
    lib().mpf_mark_intersections(
        addr(vertices),
        addr(faces),
        len(faces),
        epsilon,
        int(justproper),
        addr(flags),
    )
    return flags.astype(bool)


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    _mesh_buffers(vertices, faces)
    if not len(faces):
        return 0.0
    return float(lib().mpf_signed_volume(addr(vertices), addr(faces), len(faces)))
