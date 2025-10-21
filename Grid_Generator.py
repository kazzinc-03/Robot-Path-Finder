import random
import matplotlib.pyplot as plt
import numpy as np
import heapq

# ---------------- Grid Generation ----------------
def generate_grid(N=10, obstacle_density=0.3, seed=None):
    """
    Generates an N x N grid with randomly scattered obstacles.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            if random.random() < obstacle_density:
                grid[i, j] = 1

    free_cells = list(zip(*np.where(grid == 0)))
    if len(free_cells) < 2:
        raise ValueError("Too many obstacles! Try reducing obstacle_density.")

    start = tuple(random.choice(free_cells))
    free_cells.remove(start)
    goal = tuple(random.choice(free_cells))

    return grid, start, goal


# ---------------- Visualization ----------------
def plot_grid(grid, start, goal, figsize=(8, 8)):
    N = grid.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, cmap='gray_r', origin='upper', extent=[0, N, 0, N])
    sx, sy = start[1] + 0.5, N - start[0] - 0.5
    gx, gy = goal[1] + 0.5, N - goal[0] - 0.5
    ax.scatter(sx, sy, marker='o', s=200, facecolors='none', edgecolors='green', linewidths=2, label='Start')
    ax.scatter(gx, gy, marker='X', s=200, color='red', label='Goal')
    ax.grid(True, color='lightgray')
    ax.legend()
    plt.show()


# ---------------- A* Search ----------------
def astar(grid, start, goal, heuristic):
    N = grid.shape[0]
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    visited = set()
    nodes_expanded = 0

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            return g, nodes_expanded  # cost, nodes expanded

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < N and 0 <= ny < N and grid[nx, ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    new_g = g + 1
                    new_f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (new_f, new_g, neighbor))

    return float('inf'), nodes_expanded


# ---------------- Heuristics ----------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def vertical_bias(a, b):
    return 2 * abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------- Compare Heuristics ----------------
def compare_heuristics(grid, start, goal):
    heuristics = {
        'Manhattan': manhattan,
        'Euclidean': euclidean,
        'VerticalBias': vertical_bias
    }
    results = {}
    for name, h in heuristics.items():
        cost, nodes = astar(grid, start, goal, h)
        results[name] = {
            'cost': cost,
            'nodes_expanded': nodes
        }
    return results


# ---------------- Plot Comparison ----------------
def plot_comparison(results):
    labels = list(results.keys())
    costs = [results[h]['cost'] for h in labels]
    nodes = [results[h]['nodes_expanded'] for h in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - width/2, costs, width, label='Path Cost')
    ax.bar(x + width/2, nodes, width, label='Nodes Expanded')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Values")
    ax.set_title("Heuristic Performance on Random Grid")
    ax.legend()
    plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    grid, start, goal = generate_grid(N=20, obstacle_density=0.35, seed=None)
    print(f"Start: {start}, Goal: {goal}")

    plot_grid(grid, start, goal)

    results = compare_heuristics(grid, start, goal)
    print("\nResults:")
    for h, data in results.items():
        print(f"{h:15} -> Cost: {data['cost']}, Nodes: {data['nodes_expanded']}")

    plot_comparison(results)
