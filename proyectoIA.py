from collections import deque
import heapq

WALL = 1
START = 2
GOAL = 3

# -------------------------
# A) Lectura / utilidades
# -------------------------

def in_bounds(n, r, c):
    return 0 <= r < n and 0 <= c < n

def find_value_positions(grid, value):
    """Devuelve lista de (r,c) donde grid[r][c] == value."""
    n = len(grid)
    pos = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == value:
                pos.append((r, c))
    return pos

def get_start_goal(grid):
    """Encuentra exactamente un START=2 y un GOAL=3."""
    starts = find_value_positions(grid, START)
    goals = find_value_positions(grid, GOAL)

    if len(starts) != 1:
        raise ValueError(f"Se esperaba exactamente 1 salida (2), encontré {len(starts)}.")
    if len(goals) != 1:
        raise ValueError(f"Se esperaba exactamente 1 meta (3), encontré {len(goals)}.")

    return starts[0], goals[0]

# -------------------------
# B) Matriz -> Grafo
# -------------------------

def matrix_to_graph(grid):
    """
    Construye lista de adyacencia:
    graph[(r,c)] = [((nr,nc), cost), ...]
    Solo incluye celdas != WALL.
    """
    n = len(grid)
    graph = {}

    directions = [(1,0), (-1,0), (0,1), (0,-1)]  # 4-dir

    for r in range(n):
        for c in range(n):
            if grid[r][c] == WALL:
                continue

            node = (r, c)
            graph[node] = []

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if in_bounds(n, nr, nc) and grid[nr][nc] != WALL:
                    graph[node].append(((nr, nc), 1))  # costo 1 por paso

    return graph

# -------------------------
# D) Reconstrucción de ruta
# -------------------------

def reconstruct_path(parent, goal):
    """Reconstruye ruta desde goal usando parent dict, devuelve lista start->goal o None."""
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def print_path(path, title="Ruta"):
    if path is None:
        print(f"{title}: No hay ruta")
    else:
        print(f"{title} (len={len(path)}): {path}")

# -------------------------
# C1) BFS
# -------------------------

def bfs(graph, start, goal):
    cola = deque([start])
    visited = set([start])
    parent = {start: None}

    while cola:
        u = cola.popleft()
        if u == goal:
            return reconstruct_path(parent, goal)

        for v, cost in graph.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                cola.append(v)

    return None

# -------------------------
# C2) DFS
# -------------------------

def dfs(graph, start, goal):
    stack = [start]
    visited = set([start])
    parent = {start: None}

    while stack:
        u = stack.pop()
        if u == goal:
            return reconstruct_path(parent, goal)

        # Nota: el orden de vecinos afecta el camino que te sale en DFS.
        for v, cost in graph.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                stack.append(v)

    return None

# -------------------------
# C3) Heurística
# -------------------------

def heuristic_manhattan(node, goal):
    r, c = node
    rg, cg = goal
    return abs(r - rg) + abs(c - cg)

# -------------------------
# C4) A*
# -------------------------

def a_star(graph, start, goal, h_func=heuristic_manhattan):
    """
    A*: f(n)=g(n)+h(n)
    g: costo real desde start
    h: estimación al goal
    """
    # open set como heap: (f, g, node)
    open_heap = []
    heapq.heappush(open_heap, (h_func(start, goal), 0, start))

    parent = {start: None}
    g_score = {start: 0}

    closed = set()

    while open_heap:
        f, g, u = heapq.heappop(open_heap)

        if u in closed:
            continue
        closed.add(u)

        if u == goal:
            return reconstruct_path(parent, goal)

        for v, cost in graph.get(u, []):
            tentative_g = g_score[u] + cost

            if v in closed and tentative_g >= g_score.get(v, float("inf")):
                continue

            if tentative_g < g_score.get(v, float("inf")):
                parent[v] = u
                g_score[v] = tentative_g
                f_v = tentative_g + h_func(v, goal)
                heapq.heappush(open_heap, (f_v, tentative_g, v))

    return None

# -------------------------
# E) Main / Orquestación
# -------------------------

def solve(grid):
    start, goal = get_start_goal(grid)
    graph = matrix_to_graph(grid)

    path_bfs = bfs(graph, start, goal)
    path_dfs = dfs(graph, start, goal)
    path_astar = a_star(graph, start, goal)

    print_path(path_bfs, "BFS")
    print_path(path_dfs, "DFS")
    print_path(path_astar, "A*")



    

    return path_bfs, path_dfs, path_astar



def overlay_path_on_grid(grid, path):
    if path is None:
        return grid
    new_grid = [row[:] for row in grid]
    for (r, c) in path:
        if new_grid[r][c] == 0:
            new_grid[r][c] = '*'
    return new_grid

def print_grid(grid):
    for row in grid:
        for cell in row:
            print(f"{str(cell):>2}", end=" ")
        print()




# -------------------------
# Ejemplo de uso
# -------------------------
if __name__ == "__main__":
    Laberinto = [
        [2, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 3, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    ]

    path_bfs, path_dfs, path_astar = solve(Laberinto)

    print("\nLaberinto con ruta BFS:")
    print_grid(overlay_path_on_grid(Laberinto, path_bfs))

    print("\nLaberinto con ruta DFS:")
    print_grid(overlay_path_on_grid(Laberinto, path_dfs))

    print("\nLaberinto con ruta A*:")
    print_grid(overlay_path_on_grid(Laberinto, path_astar))
