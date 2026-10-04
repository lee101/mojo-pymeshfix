import os
import sys
import time

import numpy as np

ROOT = "/nvme0n1-disk/code/mojo-pymeshfix"
sys.path.insert(0, ROOT + "/python")
sys.path.insert(0, ROOT + "/bench")
import pymeshfix
from bench import grid
from collections import defaultdict
from pymeshfix import _geometry, _lib


def t(fn, reps=9):
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3


points, faces = grid(160)
m = pymeshfix.PyTMesh()
m.set_quiet(1)
m.load_array(points, faces)
v, f = m._vertices, m._faces
ed = _lib.boundary_edges(f, len(v))
print("boundary edge count", len(ed))
print("boundary_loops total      ", t(lambda: _geometry.boundary_loops(f, len(v))))
loops = _geometry.boundary_loops(f, len(v))
print("loops", [len(x) for x in loops])


def pyonly():
    edges = ed
    outgoing = defaultdict(list)
    undirected = defaultdict(list)
    for u, vv, _ in edges:
        outgoing[int(u)].append(int(vv))
        undirected[int(u)].append(int(vv))
        undirected[int(vv)].append(int(u))
    unused = {(int(u), int(vv)) for u, vv, _ in edges}
    loops = []
    while unused:
        se = next(iter(unused))
        start, current = se
        loop = [start]
        unused.remove(se)
        while current != start and len(loop) <= len(edges):
            loop.append(current)
            c = [x for x in outgoing.get(current, ()) if (current, x) in unused]
            if c:
                nxt = c[0]
                unused.remove((current, nxt))
            else:
                c = [
                    x
                    for x in undirected.get(current, ())
                    if (current, x) in unused or (x, current) in unused
                ]
                if not c:
                    break
                nxt = c[0]
                unused.discard((current, nxt))
                unused.discard((nxt, current))
            current = nxt
        if current == start and len(loop) >= 3:
            loops.append(loop)
    return loops


print("  standalone python walk  ", t(pyonly))
print("  int tuple build only    ", t(lambda: [(int(u), int(vv)) for u, vv, _ in ed]))
print("  boundary_edges          ", t(lambda: _lib.boundary_edges(f, len(v))))