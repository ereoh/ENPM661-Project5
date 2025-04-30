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
        self.BUFFER = int(BUFFER / 10)  # Buffer for obstacles in cm

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


