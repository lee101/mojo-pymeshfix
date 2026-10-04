"""Compute kernels for triangular mesh inspection and repair."""

from max.algorithm import parallelize
from std.math import abs, iota, sqrt
from std.sys import simd_width_of
from std.utils import StaticTuple

comptime RBITS: Int = 8
comptime RBUCKETS: Int = 1 << RBITS

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]



def even_passes(max_key: Int64) -> Int:
    var bits = RBITS
    var rest = Int64(RBITS)
    while (max_key >> rest) > 0:
        bits += RBITS
        rest += Int64(RBITS)
    return 2 * max((bits + 2 * RBITS - 1) // (2 * RBITS), 1)


def radix_sort(
    keys: I64Ptr,
    order: I32Ptr,
    keys2: I64Ptr,
    order2: I32Ptr,
    n: Int,
    passes: Int,
):
    if n < 2:
        return
    var hist = StaticTuple[Int32, RBUCKETS]()
    var src_k = keys
    var dst_k = keys2
    var src_o = order
    var dst_o = order2
    for p in range(passes):
        var d = 0
        while d < RBUCKETS:
            hist[d] = Int32(0)
            d += 1
        var shift = Int64(p * RBITS)
        var i = 0
        while i < n:
            var b = Int((src_k[i] >> shift) & Int64(RBUCKETS - 1))
            hist[b] = hist[b] + Int32(1)
            i += 1
        var running = Int32(0)
        d = 0
        while d < RBUCKETS:
            var c = hist[d]
            hist[d] = running
            running += c
            d += 1
        i = 0
        while i < n:
            var k = src_k[i]
            var b = Int((k >> shift) & Int64(RBUCKETS - 1))
            var pos = Int(hist[b])
            hist[b] = Int32(pos + 1)
            dst_k[pos] = k
            dst_o[pos] = src_o[i]
            i += 1
        var tk = src_k
        src_k = dst_k
        dst_k = tk
        var to = src_o
        src_o = dst_o
        dst_o = to


def sort_bound_order(bounds: FPtr, order: I32Ptr, n: Int):
    if n < 2:
        return
    var start = (n - 2) // 2
    while start >= 0:
        var root = start
        while root * 2 + 1 < n:
            var child = root * 2 + 1
            if (
                child + 1 < n
                and bounds[Int(order[child]) * 6]
                < bounds[Int(order[child + 1]) * 6]
            ):
                child += 1
            if bounds[Int(order[root]) * 6] >= bounds[Int(order[child]) * 6]:
                break
            var tmp = order[root]
            order[root] = order[child]
            order[child] = tmp
            root = child
        start -= 1
    var end = n - 1
    while end > 0:
        var tmp = order[0]
        order[0] = order[end]
        order[end] = tmp
        end -= 1
        var root = 0
        while root * 2 + 1 <= end:
            var child = root * 2 + 1
            if (
                child + 1 <= end
                and bounds[Int(order[child]) * 6]
                < bounds[Int(order[child + 1]) * 6]
            ):
                child += 1
            if bounds[Int(order[root]) * 6] >= bounds[Int(order[child]) * 6]:
                break
            tmp = order[root]
            order[root] = order[child]
            order[child] = tmp
            root = child


def edge_key(u: Int, v: Int, n_vertices: Int) -> Int64:
    var lo = u
    var hi = v
    if lo > hi:
        lo = v
        hi = u
    return Int64(lo) * Int64(n_vertices) + Int64(hi)


def max_edge_key(n_vertices: Int) -> Int64:
    if n_vertices <= 1:
        return Int64(1)
    var span = Int64(n_vertices)
    return span * span - Int64(1)


@export("mpf_boundary_edges")
def boundary_edges(
    faces_addr: Int,
    n_faces: Int,
    n_vertices: Int,
    keys_addr: Int,
    order_addr: Int,
    keys2_addr: Int,
    order2_addr: Int,
    edges_addr: Int,
) abi("C") -> Int:
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var keys = I64Ptr(unsafe_from_address=keys_addr)
    var order = I32Ptr(unsafe_from_address=order_addr)
    var keys2 = I64Ptr(unsafe_from_address=keys2_addr)
    var order2 = I32Ptr(unsafe_from_address=order2_addr)
    var edges = I32Ptr(unsafe_from_address=edges_addr)
    comptime chunk_size = 2048

    def prepare_chunk(
        chunk: Int,
    ) {imm faces, imm keys, imm order, imm n_faces, imm n_vertices,}:
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
            order[base] = Int32(base)
            order[base + 1] = Int32(base + 1)
            order[base + 2] = Int32(base + 2)

    var chunks = (n_faces + chunk_size - 1) // chunk_size
    if n_faces >= 65536:
        parallelize(prepare_chunk, chunks, min(chunks, 8))
    else:
        for chunk in range(chunks):
            prepare_chunk(chunk)
    var n = n_faces * 3
    radix_sort(keys, order, keys2, order2, n, even_passes(max_edge_key(n_vertices)))
    var write = 0
    var i = 0
    while i < n:
        var k = keys[i]
        var j = i + 1
        var edge = Int(order[i])
        while j < n and keys[j] == k:
            j += 1
        if j - i == 1:
            var owner = edge // 3
            var local = edge - owner * 3
            var u = faces[owner * 3 + local]
            var v = faces[owner * 3 + (local + 1) % 3]
            if u != v:
                edges[write * 3] = u
                edges[write * 3 + 1] = v
                edges[write * 3 + 2] = Int32(owner)
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
    keys2_addr: Int,
    order2_addr: Int,
    parent_addr: Int,
    labels_addr: Int,
) abi("C") -> Int:
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var keys = I64Ptr(unsafe_from_address=keys_addr)
    var edge_faces = I32Ptr(unsafe_from_address=edge_faces_addr)
    var keys2 = I64Ptr(unsafe_from_address=keys2_addr)
    var order2 = I32Ptr(unsafe_from_address=order2_addr)
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
            edge_faces[base] = Int32(base)
            edge_faces[base + 1] = Int32(base + 1)
            edge_faces[base + 2] = Int32(base + 2)

    var chunks = (n_faces + chunk_size - 1) // chunk_size
    if n_faces >= 65536:
        parallelize(prepare_chunk, chunks, min(chunks, 8))
    else:
        for chunk in range(chunks):
            prepare_chunk(chunk)
    var n = n_faces * 3
    radix_sort(
        keys, edge_faces, keys2, order2, n, even_passes(max_edge_key(n_vertices))
    )
    var i = 0
    while i < n:
        var k = keys[i]
        var j = i + 1
        var edge = Int(edge_faces[i])
        while j < n and keys[j] == k:
            unite(parent, edge // 3, Int(edge_faces[j]) // 3)
            j += 1
        i = j
    for f in range(n_faces):
        labels[f] = Int32(find_root(parent, f))
    var count = 0
    p = 0
    while p + W <= n_faces:
        var roots = labels.unsafe_load[width=W](p)
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
    bounds_addr: Int,
    order_addr: Int,
) abi("C") -> Int:
    var vertices = FPtr(unsafe_from_address=vertices_addr)
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var flags = I32Ptr(unsafe_from_address=flags_addr)
    var bounds = FPtr(unsafe_from_address=bounds_addr)
    var order = I32Ptr(unsafe_from_address=order_addr)
    for f in range(n_faces):
        flags[f] = 0
        order[f] = Int32(f)
        var a = Int(faces[f * 3])
        var b = Int(faces[f * 3 + 1])
        var c = Int(faces[f * 3 + 2])
        for axis in range(3):
            var av = vertices[a * 3 + axis]
            var bv = vertices[b * 3 + axis]
            var cv = vertices[c * 3 + axis]
            bounds[f * 6 + axis * 2] = min(av, min(bv, cv))
            bounds[f * 6 + axis * 2 + 1] = max(av, max(bv, cv))
    sort_bound_order(bounds, order, n_faces)
    for position in range(n_faces):
        var i = Int(order[position])
        var a0 = Int(faces[i * 3])
        var a1 = Int(faces[i * 3 + 1])
        var a2 = Int(faces[i * 3 + 2])
        for following in range(position + 1, n_faces):
            var j = Int(order[following])
            if bounds[j * 6] > bounds[i * 6 + 1] + epsilon:
                break
            if (
                bounds[i * 6 + 3] < bounds[j * 6 + 2] - epsilon
                or bounds[j * 6 + 3] < bounds[i * 6 + 2] - epsilon
                or bounds[i * 6 + 5] < bounds[j * 6 + 4] - epsilon
                or bounds[j * 6 + 5] < bounds[i * 6 + 4] - epsilon
            ):
                continue
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


def volume_total(vertices: FPtr, faces: I32Ptr, n_faces: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var vector_total = SIMD[DType.float64, W](0.0)
    var f = 0
    while f + W <= n_faces:
        var face_offsets = iota[DType.int32, W](Int32(f)) * 3
        var a = faces.gather(face_offsets) * 3
        var b = faces.gather(face_offsets + 1) * 3
        var c = faces.gather(face_offsets + 2) * 3
        var ax = vertices.gather(a)
        var ay = vertices.gather(a + 1)
        var az = vertices.gather(a + 2)
        var bx = vertices.gather(b)
        var by = vertices.gather(b + 1)
        var bz = vertices.gather(b + 2)
        var cx = vertices.gather(c)
        var cy = vertices.gather(c + 1)
        var cz = vertices.gather(c + 2)
        vector_total += ax * (by * cz - bz * cy)
        vector_total += ay * (bz * cx - bx * cz)
        vector_total += az * (bx * cy - by * cx)
        f += W
    var total = vector_total.reduce_add()
    while f < n_faces:
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
        f += 1
    return total / 6.0


@export("mpf_signed_volume")
def signed_volume(
    vertices_addr: Int, faces_addr: Int, n_faces: Int
) abi("C") -> Float64:
    var vertices = FPtr(unsafe_from_address=vertices_addr)
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    return volume_total(vertices, faces, n_faces)


@export("mpf_orient")
def orient(
    vertices_addr: Int,
    faces_addr: Int,
    out_addr: Int,
    n_faces: Int,
    n_vertices: Int,
    keys_addr: Int,
    order_addr: Int,
    keys2_addr: Int,
    order2_addr: Int,
    pair_addr: Int,
    parent_addr: Int,
    labels_addr: Int,
    flip_addr: Int,
    queue_addr: Int,
    vol_addr: Int,
) abi("C") -> Int:
    var vertices = FPtr(unsafe_from_address=vertices_addr)
    var faces = I32Ptr(unsafe_from_address=faces_addr)
    var out = I32Ptr(unsafe_from_address=out_addr)
    var keys = I64Ptr(unsafe_from_address=keys_addr)
    var order = I32Ptr(unsafe_from_address=order_addr)
    var keys2 = I64Ptr(unsafe_from_address=keys2_addr)
    var order2 = I32Ptr(unsafe_from_address=order2_addr)
    var pair = I32Ptr(unsafe_from_address=pair_addr)
    var parent = I32Ptr(unsafe_from_address=parent_addr)
    var labels = I32Ptr(unsafe_from_address=labels_addr)
    var flip = I32Ptr(unsafe_from_address=flip_addr)
    var queue = I32Ptr(unsafe_from_address=queue_addr)
    var vol = FPtr(unsafe_from_address=vol_addr)
    var n = n_faces * 3
    if n == 0:
        return 0
    comptime chunk_size = 2048

    def prepare_chunk(
        chunk: Int,
    ) {imm faces, imm keys, imm order, imm n_faces, imm n_vertices}:
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
            order[base] = Int32(base)
            order[base + 1] = Int32(base + 1)
            order[base + 2] = Int32(base + 2)

    var chunks = (n_faces + chunk_size - 1) // chunk_size
    if n_faces >= 65536:
        parallelize(prepare_chunk, chunks, min(chunks, 8))
    else:
        for chunk in range(chunks):
            prepare_chunk(chunk)

    var f = 0
    while f < n_faces:
        parent[f] = Int32(f)
        flip[f] = Int32(-1)
        f += 1

    radix_sort(keys, order, keys2, order2, n, even_passes(max_edge_key(n_vertices)))

    var i = 0
    while i < n:
        var k = keys[i]
        var j = i + 1
        while j < n and keys[j] == k:
            j += 1
        if j - i == 2:
            var ha = Int(order[i])
            var hb = Int(order[i + 1])
            var fa = ha // 3
            var la = ha - fa * 3
            var fb = hb // 3
            var lb = hb - fb * 3
            var togg = Int32(0)
            if faces[fa * 3 + la] == faces[fb * 3 + lb] and faces[
                fa * 3 + (la + 1) % 3
            ] == faces[fb * 3 + (lb + 1) % 3]:
                togg = Int32(1)
            pair[ha] = Int32(((i + 1) << 1) | Int(togg))
            pair[hb] = Int32((i << 1) | Int(togg))
            unite(parent, fa, fb)
        else:
            var t = i
            while t < j:
                pair[Int(order[t])] = Int32(-2) if j - i == 1 else Int32(-1)
                if t > i:
                    unite(parent, Int(order[t - 1]) // 3, Int(order[t]) // 3)
                t += 1
        i = j

    f = 0
    while f < n_faces:
        labels[f] = Int32(find_root(parent, f))
        f += 1

    for root in range(n_faces):
        if Int(flip[root]) >= 0:
            continue
        flip[root] = Int32(0)
        var head = 0
        var tail = 1
        queue[0] = Int32(root)
        while head < tail:
            var cur = Int(queue[head])
            head += 1
            var base = cur * 3
            for e in range(3):
                var code = Int(pair[base + e])
                if code < 0:
                    continue
                var other = Int(order[code >> 1]) // 3
                if Int(flip[other]) < 0:
                    flip[other] = flip[cur] ^ Int32(code & 1)
                    queue[tail] = Int32(other)
                    tail += 1

    f = 0
    while f < n_faces:
        queue[f] = Int32(0)
        f += 1
    f = 0
    while f < n:
        if Int(pair[f]) == -2:
            queue[Int(labels[f // 3])] = Int32(1)
        f += 1
    var n_roots = 0
    var n_open = 0
    f = 0
    while f < n_faces:
        var root = Int(labels[f])
        if root == f:
            n_roots += 1
            if Int(queue[f]) != 0:
                n_open += 1
        f += 1
    var closed = n_roots - n_open

    f = 0
    while f < n_faces:
        var b = faces[f * 3 + 1]
        var c = faces[f * 3 + 2]
        if Int(flip[f]) == 1:
            out[f * 3] = faces[f * 3]
            out[f * 3 + 1] = c
            out[f * 3 + 2] = b
        else:
            out[f * 3] = faces[f * 3]
            out[f * 3 + 1] = b
            out[f * 3 + 2] = c
        f += 1

    if closed == 1 and n_roots == 1:
        if volume_total(vertices, out, n_faces) < 0.0:
            for f in range(n_faces):
                var b = out[f * 3 + 1]
                out[f * 3 + 1] = out[f * 3 + 2]
                out[f * 3 + 2] = b
        return 0
    if closed > 0:
        f = 0
        while f < n_faces:
            vol[f] = 0.0
            f += 1
        for f in range(n_faces):
            var a = Int(out[f * 3])
            var b = Int(out[f * 3 + 1])
            var c = Int(out[f * 3 + 2])
            var bx = vertices[b * 3]
            var by = vertices[b * 3 + 1]
            var bz = vertices[b * 3 + 2]
            var cx = vertices[c * 3]
            var cy = vertices[c * 3 + 1]
            var cz = vertices[c * 3 + 2]
            vol[Int(labels[f])] += (
                vertices[a * 3] * (by * cz - bz * cy)
                + vertices[a * 3 + 1] * (bz * cx - bx * cz)
                + vertices[a * 3 + 2] * (bx * cy - by * cx)
            )
        for f in range(n_faces):
            var root = Int(labels[f])
            if vol[root] < 0.0 and Int(queue[root]) == 0:
                var b = out[f * 3 + 1]
                out[f * 3 + 1] = out[f * 3 + 2]
                out[f * 3 + 2] = b
    return 0
