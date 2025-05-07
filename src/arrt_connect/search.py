import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
r = 0  # Robot radius in meters

# Constants for map dimensions and scaling
CONVERT_TO_CM = 100  # Conversion factor to centimeters
WIDTH = int(5.4 * CONVERT_TO_CM)  # Map width in cm
HEIGHT = int(3 * CONVERT_TO_CM)  # Map height in cm
SCALE = 1  # Scaling factor for visualization

# Bounds for valid positions, accounting for robot radius
X_BOUNDS = (r, WIDTH - r)
Y_BOUNDS = (r, HEIGHT - r)

# Base class for obstacles
class Obstacle:
    def is_inside_obstacle(self, x, y):
        """
        Check if a point (x, y) is inside the obstacle.
        To be implemented by subclasses.
        """
        raise NotImplementedError

# Obstacle defined by lines connecting vertices
class LineDefinedObstacle(Obstacle):
    def __init__(self, vertices):
        """
        Initialize the obstacle with vertices and precompute line constraints.
        """
        self.lines = self.compute_line_constraints(vertices)

    def compute_line_constraints(self, vertices):
        """
        Compute line equations (ax + by + c = 0) for obstacle edges.
        """
        lines = []
        for i in range(len(vertices)):
            # Extract two consecutive vertices
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            # Compute line coefficients a, b, c
            a, b, c = y2 - y1, x1 - x2, -(y2 - y1) * x1 - (x1 - x2) * y1
            # Determine the side of the line that is inside the obstacle
            side = 1 if (a * vertices[(i + 2) % len(vertices)][0] + b * vertices[(i + 2) % len(vertices)][1] + c) > 0 else -1
            lines.append((a, b, c, side))
        return lines

    def is_inside_obstacle(self, x, y):
        """
        Check if a point (x, y) satisfies all line constraints.
        """
        return all((a * x + b * y + c) * side >= 0 for a, b, c, side in self.lines)

# Map class containing obstacles and grid representation
class Map:
    def __init__(self, BUFFER):
        """
        Initialize the map with obstacles and precompute the grid.
        """
        self.BUFFER = int(BUFFER)  # Buffer for obstacles in cm

        # Define obstacles as a list of LineDefinedObstacle instances
        self.OBSTACLES = [
            # First Wall
            LineDefinedObstacle([(1 * CONVERT_TO_CM - self.BUFFER - r, 0),
                                 (1 * CONVERT_TO_CM - self.BUFFER - r, 2 * CONVERT_TO_CM + self.BUFFER + r),
                                 (1.1 * CONVERT_TO_CM + self.BUFFER + r, 2 * CONVERT_TO_CM + self.BUFFER + r),
                                 (1.1 * CONVERT_TO_CM + self.BUFFER + r, 0)]),

            # Second Wall
            LineDefinedObstacle([(2.1 * CONVERT_TO_CM - self.BUFFER - r, 3 * CONVERT_TO_CM),
                                 (2.1 * CONVERT_TO_CM - self.BUFFER - r, 1 * CONVERT_TO_CM - self.BUFFER - r),
                                 (2.2 * CONVERT_TO_CM + self.BUFFER + r, 1 * CONVERT_TO_CM - self.BUFFER - r),
                                 (2.2 * CONVERT_TO_CM + self.BUFFER + r, 3 * CONVERT_TO_CM)]),

            # Third Wall Bottom
            LineDefinedObstacle([(3.2 * CONVERT_TO_CM - self.BUFFER - r, 0),
                                 (3.2 * CONVERT_TO_CM - self.BUFFER - r, 1 * CONVERT_TO_CM + self.BUFFER + r),
                                 (3.3 * CONVERT_TO_CM + self.BUFFER + r, 1 * CONVERT_TO_CM + self.BUFFER + r),
                                 (3.3 * CONVERT_TO_CM + self.BUFFER + r, 0)]),

            # Third Wall Top
            LineDefinedObstacle([(3.2 * CONVERT_TO_CM - self.BUFFER - r, 3 * CONVERT_TO_CM),
                                 (3.2 * CONVERT_TO_CM - self.BUFFER - r, 2 * CONVERT_TO_CM - self.BUFFER - r),
                                 (3.3 * CONVERT_TO_CM + self.BUFFER + r, 2 * CONVERT_TO_CM - self.BUFFER - r),
                                 (3.3 * CONVERT_TO_CM + self.BUFFER + r, 3 * CONVERT_TO_CM)]),

            # Fourth Wall
            LineDefinedObstacle([(4.3 * CONVERT_TO_CM - self.BUFFER - r, 0),
                                 (4.3 * CONVERT_TO_CM - self.BUFFER - r, 2 * CONVERT_TO_CM + self.BUFFER + r),
                                 (4.4 * CONVERT_TO_CM + self.BUFFER + r, 2 * CONVERT_TO_CM + self.BUFFER + r),
                                 (4.4 * CONVERT_TO_CM + self.BUFFER + r, 0)]),
        ]

        # Precompute the grid representation of the map
        self.grid = self.precompute_obstacle_grid()

    def is_valid_point(self, x, y):
        """
        Check whether a point (x, y) is in free space.

        Args:
            x (float): x position
            y (float): y position

        Returns:
            bool: True if the point is in free space, False otherwise.
        """
        # Ensure the point is within bounds
        if not ((X_BOUNDS[0] + self.BUFFER) <= x <= (X_BOUNDS[1] - self.BUFFER) and
                (Y_BOUNDS[0] + self.BUFFER) <= y <= (Y_BOUNDS[1] - self.BUFFER)):
            return False
        
        # Check if the point is inside any obstacle
        for obstacle in self.OBSTACLES:
            if obstacle.is_inside_obstacle(x, y):
                return False

        return True

    def precompute_obstacle_grid(self):
        """
        Precompute a grid representation of the map, marking free and occupied spaces.
        """
        grid = np.ones((WIDTH, HEIGHT), dtype=bool)  # Initialize grid as free space
        for x in range(WIDTH):
            for y in range(HEIGHT):
                # Mark grid cell as occupied if the point is not valid
                if not self.is_valid_point(x, y):
                    grid[x, y] = False
        return grid

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
def arrt_anytime_connect(start, goal, step_size, max_iterations, buffer):
    world = Map(BUFFER=buffer)

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
def visualize_search(path, start, goal, tree_a, tree_b, buffer):
    world = Map(BUFFER=buffer)
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
def visualize_path(path, start, goal, buffer):
    world = Map(BUFFER=buffer)
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
    buffer = 0
    world = Map(BUFFER=buffer)

    start = [50, 50]
    goal = [500, 50]
    step_size = 10
    max_iterations = 10_000

    path, tree_a, tree_b = arrt_anytime_connect(start, goal, step_size, max_iterations, buffer)
    if path:
        print("Path found:")
        for point in path:
            print(point)
    else:
        print("No path found.")

    visualize_search(path, start, goal, tree_a, tree_b, buffer)
    visualize_path(path, start, goal, world, buffer)
