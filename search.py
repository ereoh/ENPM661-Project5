import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from map import Map  # Import user-provided map with obstacles

# Algorithm 5 Line 1
# Swap(T_a, T_b)
def swap_if_needed(tree_a, tree_b, failure, threshold):
    # Algorithm 5 Line 1
    if len(tree_a.nodes) < len(tree_b.nodes):
        # Algorithm 5 Line 2
        return tree_b, tree_a, 0  # reset failure on swap
    else:
        # Algorithm 5 Line 4
        failure += 1
        # Algorithm 5 Line 5
        if failure >= threshold:
            # Algorithm 5 Line 6-7
            density_a = len(tree_a.nodes)
            density_b = len(tree_b.nodes)
            # Algorithm 5 Line 8-12
            if density_a > density_b:
                expand_target = 'b'
            else:
                expand_target = 'a'
            # Algorithm 5 Line 13
            return tree_b, tree_a, 0  # reset failure after swap
    return tree_a, tree_b, failure

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

    def add_node(self, node):
        self.nodes.append(node)
        if len(self.nodes) % self.rebuild_threshold == 0:
            self.kd_tree = KDTree([n.position for n in self.nodes])

    def nearest(self, position):
        _, index = self.kd_tree.query(position)
        return self.nodes[index]

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Algorithm 4 Line 2-3
# Extend(T, q_rand) — core of tree expansion logic
def steer(from_node, to_position, step_size):
    # Algorithm 4 Line 2
    direction = np.array(to_position) - np.array(from_node.position)
    length = np.linalg.norm(direction)
    if length == 0:
        return from_node.position
    direction = direction / length
    # Algorithm 4 Line 3
    new_position = np.array(from_node.position) + step_size * direction
    return new_position.tolist()  # q_new

def is_collision_free(p1, p2, world, num_samples=200):
    # Algorithm 4 Line 2 continued
    for i in range(num_samples + 1):
        t = i / num_samples
        x = p1[0] * (1 - t) + p2[0] * t
        y = p1[1] * (1 - t) + p2[1] * t
        if not world.is_valid_point(x, y):
            return False
    return True

# Algorithm 2 Line 1
# Connect(T, q)
def connect(tree, target_node, step_size, world, max_retries=5):
    current_node = tree.nearest(target_node.position)
    retries = 0
    while retries < max_retries:
        # Algorithm 2 Line 2
        if distance(current_node.position, target_node.position) > 2 * step_size:
            return None
        new_position = steer(current_node, target_node.position, step_size)
        if not is_collision_free(current_node.position, new_position, world):
            retries += 1
            continue  # Algorithm 2 Line 3

        new_node = Node(new_position)
        new_node.parent = current_node
        tree.add_node(new_node)

        # Algorithm 4 Line 6
        if is_collision_free(new_node.position, target_node.position, world) and \
           distance(new_node.position, target_node.position) < step_size:
            return new_node  # Algorithm 2 Line 4

        current_node = new_node
    return None

# Algorithm 3 Line 1
# Random_Config(R_T, q_goal, P_goal, P_outside)
def biased_sample(goal, bias_prob=0.05, world=None):
    if np.random.rand() < bias_prob:
        noise = np.random.normal(0, 5, size=2)  # Algorithm 3 Line 4
        candidate = (np.array(goal) + noise).tolist()
        if world.is_valid_point(candidate[0], candidate[1]):
            return candidate  # Algorithm 3 Line 6

    while True:
        sample = np.random.uniform(low=0, high=world.grid.shape[0], size=2).tolist()
        if world.is_valid_point(sample[0], sample[1]):
            return sample  # Algorithm 3 Line 26

# Algorithm 1 Line 1
# ARRT_Connect(q_init, q_goal)
def arrt_anytime_connect(start, goal, step_size, max_iterations, world):
    # Algorithm 1 Line 1-2
    start_node = Node(start)
    goal_node = Node(goal)
    tree_a = Tree(start_node)
    tree_b = Tree(goal_node)

    best_path = None
    best_cost = float("inf")
    failure = 0  # Algorithm 5 Line 4
    failure_threshold = 50

    # Algorithm 1 Line 3
    for i in range(max_iterations):
        # Algorithm 1 Line 4
        rand_position = biased_sample(goal, world=world)

        for tree_from, tree_to in [(tree_a, tree_b), (tree_b, tree_a)]:  # Algorithm 1 Line 10
            # Algorithm 1 Line 5
            nearest_node = tree_from.nearest(rand_position)
            new_position = steer(nearest_node, rand_position, step_size)
            if not is_collision_free(nearest_node.position, new_position, world):
                continue

            new_node = Node(new_position)
            new_node.parent = nearest_node
            tree_from.add_node(new_node)

            # Algorithm 1 Line 6
            connect_node = connect(tree_to, new_node, step_size, world)
            if connect_node:
                # Algorithm 1 Line 7
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
                cost = sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
                    print(f"Found better path with cost: {best_cost:.2f} at iteration {i}")

        # Algorithm 1 Line 10
        tree_a, tree_b, failure = swap_if_needed(tree_a, tree_b, failure, failure_threshold)

    # Algorithm 1 Line 12
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
    goal = [500, 50]
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
