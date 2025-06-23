#!/usr/bin/env python3
import numpy as np
import heapq


class Graph(object):
    def __init__(self):
        self.grid = np.zeros((1, 1), dtype=bool)

    def add_node(self, i, j):
        self.grid[i][j] = True

    def remove_node(self, i, j):
        self.grid[i][j] = False

    def size_nodes(self):
        return np.sum(np.sum(self.grid))/2

    def node_exists(self, i, j):
        return self.grid[i][j]

    def get_all_nodes(self):
        return [(i, j) for i in range(self.grid.shape[0]) for j in range(self.grid.shape[1]) if self.grid[i, j] == 1]

    @staticmethod
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def a_star(self, start, goal):
        rows, cols = len(self.grid), len(self.grid[0])

        # Initialize distances with infinity
        g_dist = {(i,j) : float('inf') for i in range(rows) for j in range(cols)}
        g_dist[start] = 0

        # map_nodes to reconstruct the path
        map_nodes = {}

        # Closed list
        closed_list = []

        # Priority open_list for A* (estimated distance from goal, node)
        open_list = []
        heapq.heappush(open_list, (self.heuristic(start, goal), start))

        while open_list:
            current_dist, current_node = heapq.heappop(open_list)
            closed_list.append(current_node)

            # Stop if you have reached the goal
            if current_node == goal:
                path = []

                # Path reconstruction
                while current_node != start:
                    path.insert(0, current_node)
                    current_node = map_nodes[current_node]

                # Add the starting node
                path.insert(0, start)
                return path, open_list, closed_list

            # Examine all neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = current_node[0] + dx, current_node[1] + dy

                # Skip if neighbor is off the grid or inaccessible
                if not (0 <= nx < rows and 0 <= ny < cols) or self.grid[nx][ny] == 0:
                    continue
                
                neighbor = (nx, ny)

                # Avoid reviewing nodes in the closed list
                if neighbor in closed_list:
                    continue

                # Calculate the cost for the neighbor
                step_cost = 1.4 if dx != 0 and dy != 0 else 1
                new_g_dist = g_dist[current_node] + step_cost
                if new_g_dist >= g_dist[neighbor]:
                    continue
             
                # Update if the distance is smaller
                map_nodes[neighbor] = current_node
                g_dist[neighbor] = new_g_dist
                f_cost = new_g_dist + self.heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_cost, neighbor))

        # If there is no path, it returns an empty list
        return [], [], []

if __name__ == "__main__":
    g = Graph()
    g.grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]

    _start = (0, 0)
    _goal = (4, 4)

    _path = g.a_star(_start, _goal)
    if _path:
        print("Path found:", _path)
    else:
        print("No path found.")
