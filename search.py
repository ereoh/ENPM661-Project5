import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from map import Map  # Import user-provided map with obstacles

# Node represents a point in the search tree with a link to its parent
class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None

# Tree manages a list of nodes and supports nearest-neighbor queries
class Tree:
    def __init__(self, root):
        self.nodes = [root]
        self.kd_tree = KDTree([root.position])
        self.rebuild_threshold = 20  # Rebuild KDTree every N nodes

    # Add a new node and periodically rebuild the KDTree
    def add_node(self, node):
        self.nodes.append(node)
        if len(self.nodes) % self.rebuild_threshold == 0:
            self.kd_tree = KDTree([n.position for n in self.nodes])

    # Return the nearest node to a given position
    def nearest(self, position):
        _, index = self.kd_tree.query(position)
        return self.nodes[index]

# Compute Euclidean distance between two points
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Steer from one node toward a target point with a given step size
def steer(from_node, to_position, step_size):
    direction = np.array(to_position) - np.array(from_node.position)
    length = np.linalg.norm(direction)
    if length == 0:
        return from_node.position
    direction = direction / length
    new_position = np.array(from_node.position) + step_size * direction
    return new_position.tolist()

# Check if the straight path between two points is obstacle-free
def is_collision_free(p1, p2, world, num_samples=200):
    for i in range(num_samples + 1):
        t = i / num_samples
        x = p1[0] * (1 - t) + p2[0] * t
        y = p1[1] * (1 - t) + p2[1] * t
        if not world.is_valid_point(x, y):
            return False
    return True

# Attempt rewiring to a grandparent if it shortens the path
def triangular_rewiring(tree, new_node, world):
    if new_node.parent is None or new_node.parent.parent is None:
        return
    grandparent = new_node.parent.parent
    if is_collision_free(grandparent.position, new_node.position, world):
        if distance(grandparent.position, new_node.position) < \
           distance(grandparent.position, new_node.parent.position) + \
           distance(new_node.parent.position, new_node.position):
            new_node.parent = grandparent

# Try connecting a node to the other tree using steering and collision checks
def connect(tree, target_node, step_size, world, max_retries=5):
    current_node = tree.nearest(target_node.position)
    retries = 0
    while retries < max_retries:
        if distance(current_node.position, target_node.position) > 2 * step_size:
            return None
        new_position = steer(current_node, target_node.position, step_size)
        if not is_collision_free(current_node.position, new_position, world):
            retries += 1
            continue
        new_node = Node(new_position)
        new_node.parent = current_node
        tree.add_node(new_node)
        triangular_rewiring(tree, new_node, world)
        if is_collision_free(new_node.position, target_node.position, world) and \
           distance(new_node.position, target_node.position) < step_size:
            return new_node
        current_node = new_node
    return None

# Sample randomly from space, with a small bias toward the goal
def biased_sample(goal, bias_prob=0.05, world=None):
    if np.random.rand() < bias_prob:
        noise = np.random.normal(0, 5, size=2)
        candidate = (np.array(goal) + noise).tolist()
        if world.is_valid_point(candidate[0], candidate[1]):
            return candidate
    while True:
        sample = np.random.uniform(low=0, high=world.grid.shape[0], size=2).tolist()
        if world.is_valid_point(sample[0], sample[1]):
            return sample

# Try shortcutting random segments of the path to reduce total cost
def optimize_path(path, world, iterations=50):
    if not path:
        return path
    for _ in range(iterations):
        if len(path) < 3:
            break
        i, j = sorted(np.random.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if is_collision_free(path[i], path[j], world):
            path = path[:i+1] + path[j:]
    return path

# Main ARRT algorithm: grows two trees and tries to connect them
def arrt_anytime_connect(start, goal, step_size, max_iterations, world):
    # T_a.init(q_init), T_b.init(q_goal)
    start_node = Node(start)
    goal_node = Node(goal)
    tree_a = Tree(start_node)
    tree_b = Tree(goal_node)

    best_path = None
    best_cost = float("inf")
    stuck_counter = 0
    max_stuck_iterations = 100

    # for k = 1 to K do
    for i in range(max_iterations):
        rand_position = biased_sample(goal, world=world)
        for tree_from, tree_to in [(tree_a, tree_b), (tree_b, tree_a)]:
            # q_near = Nearest(T_a, q_rand)
            nearest_node = tree_from.nearest(rand_position)
            # q_new = Steer(q_near, q_rand)
            new_position = steer(nearest_node, rand_position, step_size)
            if not is_collision_free(nearest_node.position, new_position, world):
                continue
            new_node = Node(new_position)
            new_node.parent = nearest_node
            tree_from.add_node(new_node)
            triangular_rewiring(tree_from, new_node, world)

            # if Connect(T_b, q_new) == Reached then
            connect_node = connect(tree_to, new_node, step_size, world)
            if connect_node:
                path = []
                node = new_node
                while node:
                    path.append(node.position)
                    node = node.parent
                path = path[::-1]
                node = connect_node
                while node:
                    path.append(node.position)
                    node = node.parent
                # OptimizePath(path)
                path = optimize_path(path, world, iterations=20)
                cost = sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
                    print(f"Found better path with cost: {best_cost:.2f} at iteration {i}")

    return best_path, tree_a, tree_b

# Visualize full path, obstacles, trees, start and goal
def visualize_path(path, world, start, goal, tree_a, tree_b):
    plt.figure(figsize=(10, 6))
    plt.title("ARRT-Connect Path Planning")
    plt.xlim(0, world.grid.shape[0])
    plt.ylim(0, world.grid.shape[1])
    plt.gca().set_aspect('equal')

    obstacle_points = np.transpose(np.where(world.grid == False))
    if obstacle_points.size > 0:
        plt.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'ks', markersize=1)

    if path:
        px, py = zip(*path)
        plt.plot(px, py, 'r-', linewidth=2, label='Planned Path')

    plt.plot(start[0], start[1], 'go', markersize=8, label='Start')
    plt.plot(goal[0], goal[1], 'bo', markersize=8, label='Goal')

    for node in tree_a.nodes:
        plt.plot(node.position[0], node.position[1], 'g.')
    for node in tree_b.nodes:
        plt.plot(node.position[0], node.position[1], 'b.')

    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize only the path, obstacles, and start/goal markers
def visualize_path_only(path, start, goal, world):
    plt.figure(figsize=(10, 6))
    plt.title("ARRT-Connect Path Only")
    plt.xlim(0, world.grid.shape[0])
    plt.ylim(0, world.grid.shape[1])
    plt.gca().set_aspect('equal')

    # Plot obstacles
    obstacle_points = np.transpose(np.where(world.grid == False))
    if obstacle_points.size > 0:
        plt.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'ks', markersize=1)

    # Plot the path
    if path:
        px, py = zip(*path)
        plt.plot(px, py, 'r-', linewidth=2, label='Planned Path')

    # Plot start and goal points
    plt.plot(start[0], start[1], 'go', markersize=8, label='Start')
    plt.plot(goal[0], goal[1], 'bo', markersize=8, label='Goal')

    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution block: initialize world, plan path, and visualize
if __name__ == "__main__":
    r = 0
    world = Map(BUFFER=r)

    start = [50, 50]
    goal = [200, 175]
    step_size = 10
    max_iterations = 10_000

    path, tree_a, tree_b = arrt_anytime_connect(start, goal, step_size, max_iterations, world)
    if path:
        print("Path found:")
        for point in path:
            print(point)
    else:
        print("No path found.")

    visualize_path(path, world, start, goal, tree_a, tree_b)
    visualize_path_only(path, start, goal, world)
