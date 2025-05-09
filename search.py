import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
import time  # Import the time module

####################################################################
# Map and Obstacle Definitions
####################################################################

r = 22  # Robot radius in meters

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

####################################################################
# Path Planning Algorithms
####################################################################

map_instance = Map(BUFFER=10)  # Create an instance of the Map class

class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = [root]
        self.edges = []

    def add_node(self, node, parent=None):
        """
        Add a node to the tree only if it is in free space and has a valid parent.
        """
        if map_instance.is_valid_point(node[0], node[1]):  # Check if the node is in free space
            if parent is not None and parent in self.nodes:
                self.nodes.append(node)
                self.add_edge(parent, node)
            else:
                print(f"Node {node} not added because it has no valid parent.")
        else:
            print(f"Node {node} not added because it is inside an obstacle.")

    def add_edge(self, node1, node2):
        """
        Add an edge between two nodes.
        """
        self.edges.append((node1, node2))

############################################################
# Algorithm 3
############################################################

def random_config(tree, qgoal, pgoal, poutside):
    # Algorithm 3 line 1
    if all_dimensions_reach_boundary(tree):
        # Algorithm 3 line 2
        qrand = random_configuration_in_space()
    else:
        # Algorithm 3 line 4
        prand = random.random()
        # Algorithm 3 line 5
        if prand <= pgoal:
            # Algorithm 3 line 6
            qrand = qgoal
        else:
            # Algorithm 3 line 8
            if prand >= poutside:
                # Algorithm 3 line 9
                qrand = random_configuration_in_tree(tree)
            else:
                # Algorithm 3 line 11
                unexplored_intervals = calculate_unexplored_intervals(tree)
                dmax, ur_values = find_dimension_with_largest_unexplored(unexplored_intervals)
                # Algorithm 3 line 14
                pmax = ur_values[dmax] / sum(ur_values)
                # Algorithm 3 line 16
                if random.random() <= pmax:
                    dim = dmax
                else:
                    # Algorithm 3 line 18
                    dim = choose_other_dimension(dmax, ur_values)
                # Algorithm 3 line 21
                qrand = config_in_range(dim, unexplored_intervals)
        # Algorithm 3 line 24
        poutside = update_poutside(tree)
    # Algorithm 3 line 26
    return qrand, poutside

# Helper functions
def all_dimensions_reach_boundary(tree):
    """
    Check if all dimensions of the tree reach the boundary of the configuration space.
    """
    for node in tree.nodes:
        x, y = node
        if not (X_BOUNDS[0] <= x <= X_BOUNDS[1] and Y_BOUNDS[0] <= y <= Y_BOUNDS[1]):
            return False
    return True

def random_configuration_in_space():
    """
    Generate a random configuration in the entire space.
    """
    while True:
        x = random.uniform(X_BOUNDS[0], X_BOUNDS[1])
        y = random.uniform(Y_BOUNDS[0], Y_BOUNDS[1])
        if map_instance.is_valid_point(x, y):  # Use the Map class's is_valid_point method
            return (x, y)

def random_configuration_in_tree(tree):
    """
    Generate a random configuration within the region of the tree.
    """
    while True:
        node = random.choice(tree.nodes)
        x = node[0] + random.uniform(-r, r)  # Add a small random offset within the robot's radius
        y = node[1] + random.uniform(-r, r)
        if map_instance.is_valid_point(x, y):
            return (x, y)

def calculate_unexplored_intervals(tree):
    """
    Calculate unexplored intervals in each dimension.
    """
    explored_x = [node[0] for node in tree.nodes]
    explored_y = [node[1] for node in tree.nodes]

    unexplored_x = max(X_BOUNDS) - min(explored_x)
    unexplored_y = max(Y_BOUNDS) - min(explored_y)

    return [unexplored_x, unexplored_y]

def find_dimension_with_largest_unexplored(unexplored_intervals):
    """
    Find the dimension with the largest unexplored interval.
    """
    dmax = unexplored_intervals.index(max(unexplored_intervals))
    return dmax, unexplored_intervals

def choose_other_dimension(dmax, ur_values):
    """
    Choose one of the remaining dimensions.
    """
    remaining_dims = [i for i in range(len(ur_values)) if i != dmax]
    return random.choice(remaining_dims)

def config_in_range(dim, unexplored_intervals):
    """
    Generate a configuration in the range of the unexplored interval for the given dimension.
    """
    while True:
        if dim == 0:  # x-dimension
            x = random.uniform(X_BOUNDS[0], unexplored_intervals[dim])
            y = random.uniform(Y_BOUNDS[0], Y_BOUNDS[1])
        else:  # y-dimension
            x = random.uniform(X_BOUNDS[0], X_BOUNDS[1])
            y = random.uniform(Y_BOUNDS[0], unexplored_intervals[dim])
        if map_instance.is_valid_point(x, y):
            return (x, y)

def update_poutside(tree):
    """
    Update the value of Poutside based on the ratio of unexplored space to total space.
    """
    # Total space area
    total_space = (X_BOUNDS[1] - X_BOUNDS[0]) * (Y_BOUNDS[1] - Y_BOUNDS[0])

    # Calculate unexplored intervals
    unexplored_intervals = calculate_unexplored_intervals(tree)
    unexplored_space = unexplored_intervals[0] * unexplored_intervals[1]

    # Update Poutside as the ratio of unexplored space to total space
    poutside = unexplored_space / total_space

    # Optionally scale Poutside to avoid it becoming too small
    poutside = max(0.01, min(poutside, 1))  # Clamp between 0.01 and 1

    return poutside

##############################################################
# Algorithm 4
##############################################################

def extend(tree, qrand):
    """
    Extend the tree towards the sample point qrand.
    """
    # Algorithm 4 line 1
    qnear = nearest_neighbor(tree, qrand)

    # Algorithm 4 line 2
    if new_config(qnear, qrand):
        qnew = compute_new_config(qnear, qrand)  # Compute the new configuration
        if map_instance.is_valid_point(qnew[0], qnew[1]):
            # Algorithm 4 line 3
            tree.add_node(qnew, qnear)
            # Algorithm 4 line 4
            tree.add_edge(qnear, qnew)  # Add an edge between qnear and qnew
            # Algorithm 4 line 5
            if distance(qnew, qrand) <= delta():
                # Algorithm 4 line 6
                return "Reached", qnew
            else:
                # Algorithm 4 line 8
                return "Advanced", qnew
        else :
            # Algorithm 4 line 25
            return "Trapped", None
    else:
        # Algorithm 4 line 11
        s_points = local_sampling(qnear)
        # Algorithm 4 line 12
        s_obstacle, s_free = divide_based_on_free(s_points)
        # Algorithm 4 line 13
        aveobs = mean_point(s_obstacle)

        # Algorithm 4 line 14
        if aveobs is not None and in_obstacle(aveobs):
            # Algorithm 4 line 15
            qobs1, qobs2 = furthest_point_pair(s_obstacle)
            # Algorithm 4 line 16
            expand_along_line(tree, qnear, qobs1, qobs2)
        else:
            # Algorithm 4 line 18
            if aveobs is not None and distance(aveobs, qnear) >= delta():
                # Algorithm 4 line 19
                expand_towards(tree, qnear, aveobs)
            else:
                # Algorithm 4 line 21
                qfree1, qfree2 = furthest_point_pair(s_free)
                # Algorithm 4 line 22
                expand_along_line(tree, qnear, qfree1, qfree2)
        # Algorithm 4 line 25
        return "Trapped", None

# Helper functions
def nearest_neighbor(tree, qrand):
    """
    Find the nearest neighbor to qrand in the tree.
    """
    return min(tree.nodes, key=lambda node: distance(node, qrand))

def new_config(qnear, qrand):
    """
    Check if a new configuration can be created between qnear and qrand.
    Ensures the path between qnear and qrand is valid by sampling points.
    """
    seg_len   = distance(qnear, qrand)
    n_samples = max(1, int(seg_len/5))

    for i in range(n_samples + 1):
        t = i / n_samples
        x = qnear[0] + t * (qrand[0] - qnear[0])
        y = qnear[1] + t * (qrand[1] - qnear[1])
        if not map_instance.is_valid_point(x, y):
            return False
    return True

def compute_new_config(qnear, qrand):
    """
    Compute a new configuration between qnear and qrand.
    """
    step_size = delta()
    direction = np.array(qrand) - np.array(qnear)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return qnear
    direction = direction / norm
    qnew = np.array(qnear) + step_size * direction
    qnew = tuple(qnew)
    # Validate the new configuration
    if map_instance.is_valid_point(qnew[0], qnew[1]):
        return qnew
    else:
        return qnear

def local_sampling(qnear):
    """
    Perform local sampling around qnear.
    """
    samples = []
    for _ in range(10):  # Generate 10 random samples
        x = qnear[0] + random.uniform(-r, r)
        y = qnear[1] + random.uniform(-r, r)
        if map_instance.is_valid_point(x, y):
            samples.append((x, y))
    return samples

def divide_based_on_free(s_points):
    """
    Divide sampled points into free and obstacle points.
    """
    s_free = [p for p in s_points if map_instance.is_valid_point(p[0], p[1])]
    s_obstacle = [p for p in s_points if not map_instance.is_valid_point(p[0], p[1])]
    return s_obstacle, s_free

def mean_point(points):
    """
    Compute the mean point of a set of points.
    """
    if not points:
        return None
    x_mean = sum(p[0] for p in points) / len(points)
    y_mean = sum(p[1] for p in points) / len(points)
    return (x_mean, y_mean)

def in_obstacle(point):
    """
    Check if a point is inside an obstacle.
    """
    return not map_instance.is_valid_point(point[0], point[1])

def furthest_point_pair(points):
    """
    Find the pair of points with the maximum distance between them.
    """
    if not points or len(points) < 2:
        return None, None  # Return None if there are not enough points

    max_dist = 0
    pair = (None, None)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distance(points[i], points[j])
            if dist > max_dist:
                max_dist = dist
                pair = (points[i], points[j])
    return pair

def expand_along_line(tree, qnear, p1, p2):
    """
    Expand qnear along the line between p1 and p2.
    """
    if p1 is None or p2 is None:
        return

    direction = np.array(p2) - np.array(p1)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return
    direction = direction / norm
    step_size = delta()
    qnew = np.array(qnear) + step_size * direction
    if map_instance.is_valid_point(qnew[0], qnew[1]):
        tree.add_node(tuple(qnew), qnear)

def expand_towards(tree, qnear, target):
    """
    Expand qnear towards the target point.
    """
    direction = np.array(target) - np.array(qnear)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return
    direction = direction / norm
    step_size = delta()
    qnew = np.array(qnear) + step_size * direction
    if map_instance.is_valid_point(qnew[0], qnew[1]):
        tree.add_node(tuple(qnew), qnear)

def distance(p1, p2):
    """
    Compute the Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))

def delta():
    """
    Return the step size for expansion.
    """
    return 5  # Example step size

####################################################################
# Algorithm 2
####################################################################

def connect(tree, qnew):
    # Algorithm 2 line 1
    while True:
        # Algorithm 2 line 2
        extend_result, qnew = extend(tree, qnew)
        # Algorithm 2 line 3
        if extend_result != "Advanced":
            break
    # Algorithm 2 line 4
    return extend_result

##############################################################
# Algorithm 5
##############################################################

def swap_trees(tree_a, tree_b, failure, threshold):
    """
    Swap trees based on the number of nodes and density.
    """
    # Algorithm 5 line 1
    if len(tree_a.nodes) < len(tree_b.nodes):
        # Algorithm 5 line 2
        return tree_b, tree_a, failure  # Swap trees
    else:
        # Algorithm 5 line 3
        failure += 1

        # Algorithm 5 line 5
        if failure >= threshold:
            # Algorithm 5 line 6
            density_a = len(tree_a.nodes) / (X_BOUNDS[1] - X_BOUNDS[0]) * (Y_BOUNDS[1] - Y_BOUNDS[0])
            density_b = len(tree_b.nodes) / (X_BOUNDS[1] - X_BOUNDS[0]) * (Y_BOUNDS[1] - Y_BOUNDS[0])

            # Algorithm 5 line 8
            if density_a > density_b:
                # Algorithm 5 line 9
                expand_tree(tree_b)
            else:
                # Algorithm 5 line 11
                expand_tree(tree_a)

            # Algorithm 5 line 13
            tree_a, tree_b = tree_b, tree_a
            # Algorithm 5 line 14
            failure = 0

    return tree_a, tree_b, failure

def expand_tree(tree):
    """
    Expand the given tree by adding a random valid node.
    """
    qrand = random_configuration_in_space()
    extend(tree, qrand)

##############################################################
# Algorithm 1
##############################################################

def arrt_connect(qinit, qgoal, k_max, pgoal, poutside):
    # Start timing
    start_time = time.time()

    # Early exit if start and goal are the same
    if qinit == qgoal:
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        milliseconds = int((elapsed_time % 1) * 1000)
        print(f"Start and goal are the same. Path found immediately in {minutes}m {seconds}s {milliseconds}ms.")
        return [qinit], Tree(qinit), Tree(qgoal)

    # Algorithm 1 line 1
    tree_a = Tree(qinit)
    # Algorithm 1 line 2
    tree_b = Tree(qgoal)

    failure = 0  # Initialize failure counter
    threshold = 5  # Set a threshold for failure

    # Algorithm 1 line 3
    for k in range(1, k_max + 1):
        # Algorithm 1 line 4
        qrand, poutside = random_config(tree_a, qgoal, pgoal, poutside)
        extend_result, qnew = extend(tree_a, qrand)
        # Algorithm 1 line 5
        if extend_result != "Trapped":
            # Algorithm 1 line 6
            if connect(tree_b, qnew) == "Reached":
                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                milliseconds = int((elapsed_time % 1) * 1000)
                print(f"Path found in {minutes}m {seconds}s {milliseconds}ms with {len(tree_a.nodes)} nodes in Tree A and {len(tree_b.nodes)} nodes in Tree B.")
                return path(tree_a, tree_b), tree_a, tree_b

        # Algorithm 1 line 10
        tree_a, tree_b, failure = swap_trees(tree_a, tree_b, failure, threshold)

        if k % 100 == 0:  # Update every iteration
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            milliseconds = int((elapsed_time % 1) * 1000)
            print(f"Iteration {k}/{k_max}: Tree A nodes = {len(tree_a.nodes)}, Tree B nodes = {len(tree_b.nodes)}, Elapsed time = {minutes}m {seconds}s {milliseconds}ms")
            # visualize_path_and_trees(map_instance, tree_a, tree_b, [])

    # Algorithm 1 line 12
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int((elapsed_time % 1) * 1000)
    print(f"Search failed after {minutes}m {seconds}s {milliseconds}ms with {len(tree_a.nodes)} nodes in Tree A and {len(tree_b.nodes)} nodes in Tree B.")
    return "Fail", tree_a, tree_b

# Helper function to compute the path between two trees
def path(tree_a, tree_b):
    """
    Compute the path connecting tree_a and tree_b.
    """
    # Start from the root of tree_a
    path_a = []
    current = tree_a.nodes[-1]  # Last node added to tree_a
    while current != tree_a.root:
        path_a.append(current)
        # Find the parent of the current node
        for edge in tree_a.edges:
            if edge[1] == current:
                current = edge[0]
                break
    path_a.append(tree_a.root)  # Add the root of tree_a
    path_a.reverse()  # Reverse to get the path from root to the last node

    # Start from the root of tree_b
    path_b = []
    current = tree_b.nodes[-1]  # Last node added to tree_b
    while current != tree_b.root:
        path_b.append(current)
        # Find the parent of the current node
        for edge in tree_b.edges:
            if edge[1] == current:
                current = edge[0]
                break
    path_b.append(tree_b.root)  # Add the root of tree_b

    # Combine the two paths
    return path_a + path_b

####################################################################
# Visualization
####################################################################

def visualize_path_and_trees(tree_a, tree_b, path):
    """
    Visualize the map, trees, and path in three separate graphs.
    """
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Plot 1: Map with trees and path
    ax1 = axes[0]
    ax1.set_title("Map with Trees and Path")
    visualize_map(ax1, map_instance)
    visualize_tree(ax1, tree_a, color="blue")
    visualize_tree(ax1, tree_b, color="green")
    visualize_path(ax1, path, color="red")

    # Plot 2: Map with path only
    ax2 = axes[1]
    ax2.set_title("Map with Path Only")
    visualize_map(ax2, map_instance)
    visualize_path(ax2, path, color="red")

    # Plot 3: Trees only
    ax3 = axes[2]
    ax3.set_title("Map with Trees Only")
    visualize_map(ax3, map_instance)
    visualize_tree(ax3, tree_a, color="blue")
    visualize_tree(ax3, tree_b, color="green")
    ax3.set_xlim(0, WIDTH)
    ax3.set_ylim(0, HEIGHT)
    ax3.set_aspect("equal")

    # Show the plots
    plt.tight_layout()
    plt.show()

def visualize_map(ax, map_instance):
    """
    Visualize the map on the given axis using the precomputed grid.
    """
    # Use the precomputed grid from the map instance
    grid = ~map_instance.grid  # Invert the grid (True -> False, False -> True)

    # Display the grid as an image
    ax.imshow(grid.T, origin="lower", extent=(0, WIDTH, 0, HEIGHT), cmap="Greys", alpha=0.5)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")

def visualize_tree(ax, tree, color="blue"):
    """
    Visualize a tree on the given axis.
    """
    for edge in tree.edges:
        x1, y1 = edge[0]
        x2, y2 = edge[1]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.7)
    for node in tree.nodes:
        ax.scatter(node[0], node[1], color=color, s=10)

def visualize_path(ax, path, color="red"):
    """
    Visualize the path on the given axis.
    """
    x_coords = [point[0] for point in path]
    y_coords = [point[1] for point in path]
    ax.plot(x_coords, y_coords, color=color, linewidth=2)
    ax.scatter(x_coords, y_coords, color=color, s=20)

if __name__ == "__main__":
    # Example usage
    qinit = (50, 250)  # Starting point
    qgoal = (500, 50)  # Target point
    k_max = 25_000  # Maximum iterations
    pgoal = 0.2  # Probability of sampling qgoal
    poutside = 0.95  # Range for random sampling

    result, tree_a, tree_b = arrt_connect(qinit, qgoal, k_max, pgoal, poutside)
    if result != "Fail":
        print(f"Search completed. Path found with {len(tree_a.nodes)} nodes in Tree A and {len(tree_b.nodes)} nodes in Tree B.")
        visualize_path_and_trees(tree_a, tree_b, result)
    else:
        print("Search failed. No path found.")
        visualize_path_and_trees(tree_a, tree_b, result)