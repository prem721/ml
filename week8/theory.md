# Theory answers

**5. Why IDDFS combines DFS space efficiency with BFS completeness/optimality**
IDDFS runs depth-limited DFS with limit 0, 1, 2, ... until the goal is found.
- Space: only the current path is stored, so O(b·d) like DFS (BFS needs O(b^d)).
- Completeness/optimality: it explores level by level, so the first goal found is at the shallowest depth (optimal for uniform step cost) and it never gets stuck in infinite branches.
- Trade-off: shallow levels are re-expanded on every iteration. Since most nodes sit at the deepest level, the overhead is small: total nodes ≈ b^d·(1 − 1/b)^-2, still O(b^d) time. You pay a constant-factor time cost to save exponential memory.

**6. Why graph search (visited set) beats tree search**
- Tree search can re-expand the same state through different paths and loops forever on cycles (e.g. 8-puzzle moving a tile back and forth).
- A visited/explored set prunes repeated states, so each state is expanded at most once. Worst-case time drops from exponential in path length to O(|V|+|E|), and search becomes complete on finite graphs with cycles.
- Cost: extra memory to store the visited set (and for A*, a consistent heuristic is needed to keep optimality).
