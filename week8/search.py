# Week 8: BFS, DFS (from scratch) and IDDFS on 8-Puzzle + graph tracing
from collections import deque

# ---------------- 8-Puzzle ----------------
START = (1, 2, 3, 4, 0, 6, 7, 5, 8)
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES = {"Up": -3, "Down": 3, "Left": -1, "Right": 1}


def neighbors(s):
    i = s.index(0)
    for m, d in MOVES.items():
        j = i + d
        if (m == "Left" and i % 3 == 0) or (m == "Right" and i % 3 == 2) or not 0 <= j < 9:
            continue
        t = list(s); t[i], t[j] = t[j], t[i]
        yield m, tuple(t)


def show(s):
    return "\n".join(" ".join(map(str, s[r:r + 3])) for r in (0, 3, 6))


def bfs(start, goal):
    q, seen, n = deque([(start, [])]), {start}, 0
    while q:
        s, path = q.popleft(); n += 1
        if s == goal: return path, n
        for m, t in neighbors(s):
            if t not in seen:
                seen.add(t); q.append((t, path + [m]))


def dfs(start, goal, limit=30):
    # depth-limited so plain DFS terminates / returns a (non-optimal) path
    stack, seen, n = [(start, [])], set(), 0
    while stack:
        s, path = stack.pop()
        if s in seen: continue
        seen.add(s); n += 1
        if s == goal: return path, n
        if len(path) < limit:
            for m, t in neighbors(s):
                if t not in seen: stack.append((t, path + [m]))


def iddfs(start, goal, max_depth=30):
    def dls(s, path, depth, onpath):
        if s == goal: return path
        if depth == 0: return None
        for m, t in neighbors(s):
            if t not in onpath:
                r = dls(t, path + [m], depth - 1, onpath | {t})
                if r is not None: return r
    for d in range(max_depth + 1):
        r = dls(start, [], d, {start})
        if r is not None: return r, d


def replay(s, path):
    for m in path:
        s = dict(neighbors(s))[m]
    return s


print("Start:\n" + show(START) + "\n\nGoal:\n" + show(GOAL))
for name, fn in [("BFS", bfs), ("DFS", dfs), ("IDDFS", iddfs)]:
    path, extra = fn(START, GOAL)
    assert replay(START, path) == GOAL
    info = f"nodes expanded={extra}" if name != "IDDFS" else f"solution depth={extra}"
    print(f"\n{name}: {len(path)} moves, {info}\n  Moves: {' -> '.join(path)}")

# ---------------- Graph tracing (Lab 2) ----------------
G = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": ["G"], "E": ["G"], "F": ["G"], "G": []}


def bfs_graph(g, s, t):
    q, parent, order = deque([s]), {s: None}, []
    while q:
        u = q.popleft(); order.append(u)
        if u == t: break
        for v in g[u]:
            if v not in parent: parent[v] = u; q.append(v)
    p, u = [], t
    while u: p.append(u); u = parent[u]
    return p[::-1], order


def dfs_graph(g, s, t):
    stack, seen, order = [s], set(), []
    while stack:
        u = stack.pop()
        if u in seen: continue
        seen.add(u); order.append(u)
        if u == t: break
        stack.extend(reversed(g[u]))
    return order


def iddfs_graph(g, s, t):
    def dls(u, d, order):
        order.append(u)
        if u == t: return True
        return d > 0 and any(dls(v, d - 1, order) for v in g[u])
    for d in range(len(g)):
        order = []
        print(f"  depth {d}: ", end="")
        found = dls(s, d, order)
        print(" ".join(order))
        if found: return d


path, order = bfs_graph(G, "A", "G")
print("\n(a) Shortest path A->G:", " -> ".join(path))
print("(b) BFS expansion order:", " ".join(order))
print("(c) Nodes expanded by BFS (incl. G):", len(order), "| before reaching G:", len(order) - 1)
print("DFS order:", " ".join(dfs_graph(G, "A", "G")))
print("IDDFS trace:"); print("  found at depth", iddfs_graph(G, "A", "G"))
