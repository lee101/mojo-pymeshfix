"""Compute kernels for triangular mesh inspection and repair."""

from std.algorithm import parallelize
from std.math import abs, iota, sqrt
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def swap_key3(keys: I64Ptr, a: I32Ptr, b: I32Ptr, c: I32Ptr, i: Int, j: Int):
    var tk = keys[i]
    keys[i] = keys[j]
    keys[j] = tk
    var tv = a[i]
    a[i] = a[j]
    a[j] = tv
    tv = b[i]
    b[i] = b[j]
    b[j] = tv
    tv = c[i]
    c[i] = c[j]
    c[j] = tv


def sift_key3(
    keys: I64Ptr, a: I32Ptr, b: I32Ptr, c: I32Ptr, start: Int, end: Int
):
    var root = start
    while root * 2 + 1 <= end:
        var child = root * 2 + 1
        if child + 1 <= end and keys[child] < keys[child + 1]:
            child += 1
        if keys[root] >= keys[child]:
            return
        swap_key3(keys, a, b, c, root, child)
        root = child


def sort_key3(keys: I64Ptr, a: I32Ptr, b: I32Ptr, c: I32Ptr, n: Int):
    if n < 2:
        return
    var start = (n - 2) // 2
    while start >= 0:
        sift_key3(keys, a, b, c, start, n - 1)
        start -= 1
    var end = n - 1
    while end > 0:
        swap_key3(keys, a, b, c, 0, end)
        end -= 1
        sift_key3(keys, a, b, c, 0, end)


def swap_key1(keys: I64Ptr, values: I32Ptr, i: Int, j: Int):
    var tk = keys[i]
    keys[i] = keys[j]
    keys[j] = tk
    var tv = values[i]
    values[i] = values[j]
    values[j] = tv


def sift_key1(keys: I64Ptr, values: I32Ptr, start: Int, end: Int):
    var root = start
    while root * 2 + 1 <= end:
        var child = root * 2 + 1
        if child + 1 <= end and keys[child] < keys[child + 1]:
            child += 1
        if keys[root] >= keys[child]:
            return
        swap_key1(keys, values, root, child)
        root = child


def sort_key1(keys: I64Ptr, values: I32Ptr, n: Int):
    if n < 2:
        return
    var start = (n - 2) // 2
    while start >= 0:
        sift_key1(keys, values, start, n - 1)
        start -= 1
    var end = n - 1
    while end > 0:
        swap_key1(keys, values, 0, end)
        end -= 1
        sift_key1(keys, values, 0, end)


def edge_key(u: Int, v: Int, n_vertices: Int) -> Int64:
    var lo = u
    var hi = v
    if lo > hi:
        lo = v
        hi = u
    return Int64(lo) * Int64(n_vertices) + Int64(hi)


@export("mpf_boundary_edges")
def boundary_edges(
    faces_addr: Int,
    n_faces: Int,
    n_vertices: Int,
    keys_addr: Int,
    u_addr: Int,
    v_addr: Int,
    face_addr: Int,
) abi("C") -> Int:
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var keys = I64Ptr(unsafe_from_address=keys_addr)
    var us = I32Ptr(unsafe_from_address=u_addr)
    var vs = I32Ptr(unsafe_from_address=v_addr)
    var owners = I32Ptr(unsafe_from_address=face_addr)
    comptime chunk_size = 2048

    def prepare_chunk(
        chunk: Int,
    ) {
        imm faces,
        imm keys,
        imm us,
        imm vs,
        imm owners,
        imm n_faces,
        imm n_vertices,
    }:
        var start = chunk * chunk_size
        var end = min(start + chunk_size, n_faces)
        for f in range(start, end):
            var a = Int(faces[f * 3])
            var b = Int(faces[f * 3 + 1])
            var c = Int(faces[f * 3 + 2])
            var base = f * 3
            keys[base] = edge_key(a, b, n_vertices)
            keys[base + 1] = edge_key(b, c, n_vertices)
            keys[base + 2] = edge_key(c, a, n_vertices)
            us[base] = Int32(a)
            us[base + 1] = Int32(b)
            us[base + 2] = Int32(c)
            vs[base] = Int32(b)
            vs[base + 1] = Int32(c)
            vs[base + 2] = Int32(a)
            owners[base] = Int32(f)
            owners[base + 1] = Int32(f)
            owners[base + 2] = Int32(f)

    if n_faces >= 32768:
        parallelize(prepare_chunk, (n_faces + chunk_size - 1) // chunk_size, 8)
    else:
        for chunk in range((n_faces + chunk_size - 1) // chunk_size):
            prepare_chunk(chunk)
    var n = n_faces * 3
    sort_key3(keys, us, vs, owners, n)
    var write = 0
    var i = 0
    while i < n:
        var j = i + 1
        while j < n and keys[j] == keys[i]:
            j += 1
        if j - i == 1 and us[i] != vs[i]:
            keys[write] = keys[i]
            us[write] = us[i]
            vs[write] = vs[i]
            owners[write] = owners[i]
            write += 1
        i = j
    return write


def find_root(parent: I32Ptr, value: Int) -> Int:
    var root = value
    while Int(parent[root]) != root:
        root = Int(parent[root])
    var item = value
    while Int(parent[item]) != item:
        var nxt = Int(parent[item])
        parent[item] = Int32(root)
        item = nxt
    return root


def unite(parent: I32Ptr, a: Int, b: Int):
    var ra = find_root(parent, a)
    var rb = find_root(parent, b)
    if ra != rb:
        if ra < rb:
            parent[rb] = Int32(ra)
        else:
            parent[ra] = Int32(rb)


@export("mpf_face_components")
def face_components(
    faces_addr: Int,
    n_faces: Int,
    n_vertices: Int,
    keys_addr: Int,
    edge_faces_addr: Int,
    parent_addr: Int,
    labels_addr: Int,
) abi("C") -> Int:
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var keys = I64Ptr(unsafe_from_address=keys_addr)
    var edge_faces = I32Ptr(unsafe_from_address=edge_faces_addr)
    var parent = I32Ptr(unsafe_from_address=parent_addr)
    var labels = I32Ptr(unsafe_from_address=labels_addr)
    comptime W = simd_width_of[DType.int32]()
    var p = 0
    while p + W <= n_faces:
        parent.store(p, iota[DType.int32, W](Int32(p)))
        p += W
    while p < n_faces:
        parent[p] = Int32(p)
        p += 1
    comptime chunk_size = 2048

    def prepare_chunk(
        chunk: Int,
    ) {imm faces, imm keys, imm edge_faces, imm n_faces, imm n_vertices}:
        var start = chunk * chunk_size
        var end = min(start + chunk_size, n_faces)
        for f in range(start, end):
            var a = Int(faces[f * 3])
            var b = Int(faces[f * 3 + 1])
            var c = Int(faces[f * 3 + 2])
            var base = f * 3
            keys[base] = edge_key(a, b, n_vertices)
            keys[base + 1] = edge_key(b, c, n_vertices)
            keys[base + 2] = edge_key(c, a, n_vertices)
            edge_faces[base] = Int32(f)
            edge_faces[base + 1] = Int32(f)
            edge_faces[base + 2] = Int32(f)

    if n_faces >= 32768:
        parallelize(prepare_chunk, (n_faces + chunk_size - 1) // chunk_size, 8)
    else:
        for chunk in range((n_faces + chunk_size - 1) // chunk_size):
            prepare_chunk(chunk)
    var n = n_faces * 3
    sort_key1(keys, edge_faces, n)
    var i = 0
    while i < n:
        var j = i + 1
        while j < n and keys[j] == keys[i]:
            unite(parent, Int(edge_faces[i]), Int(edge_faces[j]))
            j += 1
        i = j
    for f in range(n_faces):
        labels[f] = Int32(find_root(parent, f))
    var count = 0
    p = 0
    while p + W <= n_faces:
        var roots = labels.load[width=W](p)
        var indices = iota[DType.int32, W](Int32(p))
        var matches = roots.eq(indices)
        count += Int(
            matches.select(
                SIMD[DType.int32, W](1), SIMD[DType.int32, W](0)
            ).reduce_add()
        )
        p += W
    while p < n_faces:
        if Int(labels[p]) == p:
            count += 1
        p += 1
    return count


def cross_norm2(
    ax: Float64,
    ay: Float64,
    az: Float64,
    bx: Float64,
    by: Float64,
    bz: Float64,
) -> Float64:
    var x = ay * bz - az * by
    var y = az * bx - ax * bz
    var z = ax * by - ay * bx
    return x * x + y * y + z * z


@export("mpf_mark_degenerate")
def mark_degenerate(
    vertices_addr: Int,
    faces_addr: Int,
    n_vertices: Int,
    n_faces: Int,
    area_epsilon: Float64,
    flags_addr: Int,
) abi("C") -> Int:
    var vertices = FPtr(unsafe_from_address=vertices_addr)
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var flags = I32Ptr(unsafe_from_address=flags_addr)
    var count = 0
    var limit = area_epsilon * area_epsilon * 4.0
    for f in range(n_faces):
        var ia = Int(faces[f * 3])
        var ib = Int(faces[f * 3 + 1])
        var ic = Int(faces[f * 3 + 2])
        var bad = ia < 0 or ib < 0 or ic < 0
        bad = bad or ia >= n_vertices or ib >= n_vertices or ic >= n_vertices
        bad = bad or ia == ib or ib == ic or ic == ia
        if not bad:
            var ax = vertices[ib * 3] - vertices[ia * 3]
            var ay = vertices[ib * 3 + 1] - vertices[ia * 3 + 1]
            var az = vertices[ib * 3 + 2] - vertices[ia * 3 + 2]
            var bx = vertices[ic * 3] - vertices[ia * 3]
            var by = vertices[ic * 3 + 1] - vertices[ia * 3 + 1]
            var bz = vertices[ic * 3 + 2] - vertices[ia * 3 + 2]
            bad = cross_norm2(ax, ay, az, bx, by, bz) <= limit
        flags[f] = 1 if bad else 0
        if bad:
            count += 1
    return count


def orient2d(
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    cx: Float64,
    cy: Float64,
) -> Float64:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def point_in_tri2d(
    px: Float64,
    py: Float64,
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    cx: Float64,
    cy: Float64,
    eps: Float64,
    proper: Bool,
) -> Bool:
    var o1 = orient2d(ax, ay, bx, by, px, py)
    var o2 = orient2d(bx, by, cx, cy, px, py)
    var o3 = orient2d(cx, cy, ax, ay, px, py)
    if proper:
        return (o1 > eps and o2 > eps and o3 > eps) or (
            o1 < -eps and o2 < -eps and o3 < -eps
        )
    var has_neg = o1 < -eps or o2 < -eps or o3 < -eps
    var has_pos = o1 > eps or o2 > eps or o3 > eps
    return not (has_neg and has_pos)


def on_segment2d(
    px: Float64,
    py: Float64,
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    eps: Float64,
) -> Bool:
    return (
        px >= min(ax, bx) - eps
        and px <= max(ax, bx) + eps
        and (py >= min(ay, by) - eps and py <= max(ay, by) + eps)
    )


def segment_cross2d_exact(
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    cx: Float64,
    cy: Float64,
    dx: Float64,
    dy: Float64,
    eps: Float64,
    proper: Bool,
) -> Bool:
    var o1 = orient2d(ax, ay, bx, by, cx, cy)
    var o2 = orient2d(ax, ay, bx, by, dx, dy)
    var o3 = orient2d(cx, cy, dx, dy, ax, ay)
    var o4 = orient2d(cx, cy, dx, dy, bx, by)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    if proper:
        return False
    if abs(o1) <= eps and on_segment2d(cx, cy, ax, ay, bx, by, eps):
        return True
    if abs(o2) <= eps and on_segment2d(dx, dy, ax, ay, bx, by, eps):
        return True
    if abs(o3) <= eps and on_segment2d(ax, ay, cx, cy, dx, dy, eps):
        return True
    if abs(o4) <= eps and on_segment2d(bx, by, cx, cy, dx, dy, eps):
        return True
    return False


def project_x(vertices: FPtr, index: Int, axis: Int) -> Float64:
    if axis == 0:
        return vertices[index * 3 + 1]
    return vertices[index * 3]


def project_y(vertices: FPtr, index: Int, axis: Int) -> Float64:
    if axis == 2:
        return vertices[index * 3 + 1]
    return vertices[index * 3 + 2]


def coplanar_overlap(
    vertices: FPtr,
    a0: Int,
    a1: Int,
    a2: Int,
    b0: Int,
    b1: Int,
    b2: Int,
    nx: Float64,
    ny: Float64,
    nz: Float64,
    eps: Float64,
    proper: Bool,
) -> Bool:
    var axis = 0
    if abs(ny) > abs(nx):
        axis = 1
    if abs(nz) > (abs(nx) if axis == 0 else abs(ny)):
        axis = 2
    var aa0x = project_x(vertices, a0, axis)
    var aa0y = project_y(vertices, a0, axis)
    var aa1x = project_x(vertices, a1, axis)
    var aa1y = project_y(vertices, a1, axis)
    var aa2x = project_x(vertices, a2, axis)
    var aa2y = project_y(vertices, a2, axis)
    var bb0x = project_x(vertices, b0, axis)
    var bb0y = project_y(vertices, b0, axis)
    var bb1x = project_x(vertices, b1, axis)
    var bb1y = project_y(vertices, b1, axis)
    var bb2x = project_x(vertices, b2, axis)
    var bb2y = project_y(vertices, b2, axis)
    if segment_cross2d_exact(
        aa0x, aa0y, aa1x, aa1y, bb0x, bb0y, bb1x, bb1y, eps, proper
    ):
        return True
    if segment_cross2d_exact(
        aa0x, aa0y, aa1x, aa1y, bb1x, bb1y, bb2x, bb2y, eps, proper
    ):
        return True
    if segment_cross2d_exact(
        aa0x, aa0y, aa1x, aa1y, bb2x, bb2y, bb0x, bb0y, eps, proper
    ):
        return True
    if segment_cross2d_exact(
        aa1x, aa1y, aa2x, aa2y, bb0x, bb0y, bb1x, bb1y, eps, proper
    ):
        return True
    if segment_cross2d_exact(
        aa1x, aa1y, aa2x, aa2y, bb1x, bb1y, bb2x, bb2y, eps, proper
    ):
        return True
    if segment_cross2d_exact(
        aa1x, aa1y, aa2x, aa2y, bb2x, bb2y, bb0x, bb0y, eps, proper
    ):
        return True
    if segment_cross2d_exact(
        aa2x, aa2y, aa0x, aa0y, bb0x, bb0y, bb1x, bb1y, eps, proper
    ):
        return True
    if segment_cross2d_exact(
        aa2x, aa2y, aa0x, aa0y, bb1x, bb1y, bb2x, bb2y, eps, proper
    ):
        return True
    if segment_cross2d_exact(
        aa2x, aa2y, aa0x, aa0y, bb2x, bb2y, bb0x, bb0y, eps, proper
    ):
        return True
    return point_in_tri2d(
        bb0x, bb0y, aa0x, aa0y, aa1x, aa1y, aa2x, aa2y, eps, proper
    ) or (
        point_in_tri2d(
            aa0x, aa0y, bb0x, bb0y, bb1x, bb1y, bb2x, bb2y, eps, proper
        )
    )


def segment_triangle(
    vertices: FPtr,
    p: Int,
    q: Int,
    a: Int,
    b: Int,
    c: Int,
    eps: Float64,
    proper: Bool,
) -> Bool:
    var dx = vertices[q * 3] - vertices[p * 3]
    var dy = vertices[q * 3 + 1] - vertices[p * 3 + 1]
    var dz = vertices[q * 3 + 2] - vertices[p * 3 + 2]
    var e1x = vertices[b * 3] - vertices[a * 3]
    var e1y = vertices[b * 3 + 1] - vertices[a * 3 + 1]
    var e1z = vertices[b * 3 + 2] - vertices[a * 3 + 2]
    var e2x = vertices[c * 3] - vertices[a * 3]
    var e2y = vertices[c * 3 + 1] - vertices[a * 3 + 1]
    var e2z = vertices[c * 3 + 2] - vertices[a * 3 + 2]
    var hx = dy * e2z - dz * e2y
    var hy = dz * e2x - dx * e2z
    var hz = dx * e2y - dy * e2x
    var det = e1x * hx + e1y * hy + e1z * hz
    if abs(det) <= eps:
        return False
    var inv = 1.0 / det
    var sx = vertices[p * 3] - vertices[a * 3]
    var sy = vertices[p * 3 + 1] - vertices[a * 3 + 1]
    var sz = vertices[p * 3 + 2] - vertices[a * 3 + 2]
    var u = (sx * hx + sy * hy + sz * hz) * inv
    var qx = sy * e1z - sz * e1y
    var qy = sz * e1x - sx * e1z
    var qz = sx * e1y - sy * e1x
    var v = (dx * qx + dy * qy + dz * qz) * inv
    var t = (e2x * qx + e2y * qy + e2z * qz) * inv
    if proper:
        return (
            u > eps
            and v > eps
            and u + v < 1.0 - eps
            and t > eps
            and t < 1.0 - eps
        )
    return (
        u >= -eps
        and v >= -eps
        and u + v <= 1.0 + eps
        and t >= -eps
        and t <= 1.0 + eps
    )


def triangles_intersect(
    vertices: FPtr,
    a0: Int,
    a1: Int,
    a2: Int,
    b0: Int,
    b1: Int,
    b2: Int,
    eps: Float64,
    proper: Bool,
) -> Bool:
    if (
        a0 == b0
        or a0 == b1
        or a0 == b2
        or a1 == b0
        or a1 == b1
        or a1 == b2
        or (a2 == b0 or a2 == b1 or a2 == b2)
    ):
        return False
    for axis in range(3):
        var amin = min(
            vertices[a0 * 3 + axis],
            min(vertices[a1 * 3 + axis], vertices[a2 * 3 + axis]),
        )
        var amax = max(
            vertices[a0 * 3 + axis],
            max(vertices[a1 * 3 + axis], vertices[a2 * 3 + axis]),
        )
        var bmin = min(
            vertices[b0 * 3 + axis],
            min(vertices[b1 * 3 + axis], vertices[b2 * 3 + axis]),
        )
        var bmax = max(
            vertices[b0 * 3 + axis],
            max(vertices[b1 * 3 + axis], vertices[b2 * 3 + axis]),
        )
        if amax < bmin - eps or bmax < amin - eps:
            return False
    var aex = vertices[a1 * 3] - vertices[a0 * 3]
    var aey = vertices[a1 * 3 + 1] - vertices[a0 * 3 + 1]
    var aez = vertices[a1 * 3 + 2] - vertices[a0 * 3 + 2]
    var afx = vertices[a2 * 3] - vertices[a0 * 3]
    var afy = vertices[a2 * 3 + 1] - vertices[a0 * 3 + 1]
    var afz = vertices[a2 * 3 + 2] - vertices[a0 * 3 + 2]
    var nx = aey * afz - aez * afy
    var ny = aez * afx - aex * afz
    var nz = aex * afy - aey * afx
    var norm = sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= eps:
        return False
    var d0 = (
        nx * (vertices[b0 * 3] - vertices[a0 * 3])
        + ny * (vertices[b0 * 3 + 1] - vertices[a0 * 3 + 1])
        + nz * (vertices[b0 * 3 + 2] - vertices[a0 * 3 + 2])
    )
    var d1 = (
        nx * (vertices[b1 * 3] - vertices[a0 * 3])
        + ny * (vertices[b1 * 3 + 1] - vertices[a0 * 3 + 1])
        + nz * (vertices[b1 * 3 + 2] - vertices[a0 * 3 + 2])
    )
    var d2 = (
        nx * (vertices[b2 * 3] - vertices[a0 * 3])
        + ny * (vertices[b2 * 3 + 1] - vertices[a0 * 3 + 1])
        + nz * (vertices[b2 * 3 + 2] - vertices[a0 * 3 + 2])
    )
    if (
        abs(d0) <= eps * norm
        and abs(d1) <= eps * norm
        and abs(d2) <= eps * norm
    ):
        return coplanar_overlap(
            vertices, a0, a1, a2, b0, b1, b2, nx, ny, nz, eps, proper
        )
    return (
        segment_triangle(vertices, a0, a1, b0, b1, b2, eps, proper)
        or (segment_triangle(vertices, a1, a2, b0, b1, b2, eps, proper))
        or segment_triangle(vertices, a2, a0, b0, b1, b2, eps, proper)
        or (segment_triangle(vertices, b0, b1, a0, a1, a2, eps, proper))
        or segment_triangle(vertices, b1, b2, a0, a1, a2, eps, proper)
        or (segment_triangle(vertices, b2, b0, a0, a1, a2, eps, proper))
    )


@export("mpf_mark_intersections")
def mark_intersections(
    vertices_addr: Int,
    faces_addr: Int,
    n_faces: Int,
    epsilon: Float64,
    justproper: Int,
    flags_addr: Int,
) abi("C") -> Int:
    var vertices = FPtr(unsafe_from_address=vertices_addr)
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var flags = I32Ptr(unsafe_from_address=flags_addr)
    for f in range(n_faces):
        flags[f] = 0
    for i in range(n_faces):
        var a0 = Int(faces[i * 3])
        var a1 = Int(faces[i * 3 + 1])
        var a2 = Int(faces[i * 3 + 2])
        for j in range(i + 1, n_faces):
            if triangles_intersect(
                vertices,
                a0,
                a1,
                a2,
                Int(faces[j * 3]),
                Int(faces[j * 3 + 1]),
                Int(faces[j * 3 + 2]),
                epsilon,
                justproper != 0,
            ):
                flags[i] = 1
                flags[j] = 1
    var count = 0
    for f in range(n_faces):
        count += Int(flags[f])
    return count


@export("mpf_signed_volume")
def signed_volume(
    vertices_addr: Int, faces_addr: Int, n_faces: Int
) abi("C") -> Float64:
    var vertices = FPtr(unsafe_from_address=vertices_addr)
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var total = 0.0
    for f in range(n_faces):
        var a = Int(faces[f * 3])
        var b = Int(faces[f * 3 + 1])
        var c = Int(faces[f * 3 + 2])
        var bx = vertices[b * 3]
        var by = vertices[b * 3 + 1]
        var bz = vertices[b * 3 + 2]
        var cx = vertices[c * 3]
        var cy = vertices[c * 3 + 1]
        var cz = vertices[c * 3 + 2]
        total += vertices[a * 3] * (by * cz - bz * cy)
        total += vertices[a * 3 + 1] * (bz * cx - bx * cz)
        total += vertices[a * 3 + 2] * (bx * cy - by * cx)
    return total / 6.0
