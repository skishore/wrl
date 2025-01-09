from dataclasses import dataclass
import random
from typing import List

@dataclass(frozen=True)
class Point:
    x: int
    y: int


def generate_perlin_noise(width: int, height: int, scale: float = 10.0, octaves: int = 4) -> List[List[float]]:
    """Generate Perlin noise in range [0,1]"""

    def interpolate(a0: float, a1: float, w: float) -> float:
        # Smoothstep interpolation
        return (a1 - a0) * (3.0 - w * 2.0) * w * w + a0

    noise = [[0.0 for _ in range(height)] for _ in range(width)]

    for octave in range(octaves):
        period = scale * (2 ** octave)
        frequency = 1.0 / period

        # Generate grid of random values for this octave
        grid_width = int(width * frequency) + 2
        grid_height = int(height * frequency) + 2
        grid = [[random.random() for _ in range(grid_height)] for _ in range(grid_width)]

        for y in range(height):
            y0 = int(y * frequency)
            y1 = y0 + 1
            yfrac = (y * frequency) - y0

            for x in range(width):
                x0 = int(x * frequency)
                x1 = x0 + 1
                xfrac = (x * frequency) - x0

                # Use consistent grid values for interpolation
                v00 = grid[x0][y0]
                v10 = grid[x1][y0]
                v01 = grid[x0][y1]
                v11 = grid[x1][y1]

                # Interpolate
                x_interp1 = interpolate(v00, v10, xfrac)
                x_interp2 = interpolate(v01, v11, xfrac)
                value = interpolate(x_interp1, x_interp2, yfrac)

                noise[x][y] += value / (2 ** octave)

    # Normalize to [0,1]
    max_val = max(max(row) for row in noise)
    min_val = min(min(row) for row in noise)
    for x in range(width):
        for y in range(height):
            noise[x][y] = (noise[x][y] - min_val) / (max_val - min_val)

    return noise


class CaveMap:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cells = [[False for _ in range(height)] for _ in range(width)]

    def count_neighbors(self, x: int, y: int, distance: int = 1) -> int:
        count = 0
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    count += self.cells[nx][ny]
        return count

    def step(self, birth_limit: int, death_limit: int) -> None:
        """
        birth_limit: number of neighbors needed to make a dead cell alive
        death_limit: number of neighbors needed to keep a live cell alive
        """
        new_cells = [[False for _ in range(self.height)] for _ in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                if x in (0, self.width - 1) or y in (0, self.height - 1):
                    new_cells[x][y] = True
                    continue
                neighbors = self.count_neighbors(x, y)
                if self.cells[x][y]:
                    new_cells[x][y] = neighbors >= death_limit
                else:
                    new_cells[x][y] = neighbors >= birth_limit
        self.cells = new_cells

    def print(self):
        for y in range(self.height):
            chars = []
            for x in range(self.width):
                # Convert to wide chars for square display
                c = '#' if self.cells[x][y] else '.'
                chars.append(chr(ord(c) - 0x20 + 0xFF00))
            print(''.join(chars))


def generate_cave(width: int, height: int, wall_chance: float = 0.45,
                 steps: int = 3, birth_limit: int = 5, death_limit: int = 4,
                 noise_scale: float = 5.0, noise_weight: float = 0.5,
                 seed: int = None) -> CaveMap:
    if seed is not None:
        random.seed(seed)

    cave = CaveMap(width, height)

    # Generate noise map
    noise = generate_perlin_noise(width, height, scale=noise_scale)

    # Initialize with random walls, modified by noise
    for x in range(width):
        for y in range(height):
            cave.cells[x][y] = random.random() < wall_chance

    # Add border walls
    for x in range(width):
        cave.cells[x][0] = cave.cells[x][height-1] = True
    for y in range(height):
        cave.cells[0][y] = cave.cells[width-1][y] = True

    # Run automaton steps
    for _ in range(steps):
        cave.step(birth_limit, death_limit)

    return cave


@dataclass
class Room:
    x: int  # top-left corner
    y: int
    width: int
    height: int

    def overlaps(self, other: 'Room') -> bool:
        return (self.x < other.x + other.width and self.x + self.width > other.x and
                self.y < other.y + other.height and self.y + self.height > other.y)

    def touches(self, other: 'Room', gap: int = 0) -> bool:
        # Returns true if rooms are within gap distance of each other
        return (self.x - gap <= other.x + other.width and self.x + self.width + gap >= other.x and
                self.y - gap <= other.y + other.height and self.y + self.height + gap >= other.y)

def try_place_rooms(width: int, height: int, min_size: int = 20, max_size: int = 40, attempts: int = 1000, min_coverage: float = 0.5) -> List[Room]:
    rooms = []
    total_area = width * height
    room_area = 0

    # Place first room roughly in center
    first_w = random.randint(min_size, max_size + 1)
    first_h = random.randint(min_size, max_size + 1)
    first_x = (width - first_w) // 2
    first_y = (height - first_h) // 2
    rooms.append(Room(first_x, first_y, first_w, first_h))
    room_area = first_w * first_h

    for _ in range(attempts):
        w = random.randint(min_size, max_size + 1)
        h = random.randint(min_size, max_size + 1)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        new_room = Room(x, y, w, h)

        # Check if touches any existing room
        touches_existing = False
        for room in rooms:
            if new_room.touches(room):
                touches_existing = True
            if new_room.overlaps(room):
                touches_existing = False
                break

        if touches_existing:
            rooms.append(new_room)
            room_area += w * h

    # Return None if coverage is too low
    if room_area / total_area < min_coverage:
        return None

    return rooms


def fill_caves(cave: CaveMap, rooms: List[Room]):
    # First block everything outside rooms
    for x in range(cave.width):
        for y in range(cave.height):
            cave.cells[x][y] = True  # blocked by default

    # Run cellular automaton within each room
    for room in rooms:
        # Initialize room interior with random walls
        for x in range(room.x + 1, room.x + room.width - 1):
            for y in range(room.y + 1, room.y + room.height - 1):
                cave.cells[x][y] = random.random() < 0.45

        # Ensure room borders are walls
        for x in range(room.x, room.x + room.width):
            cave.cells[x][room.y] = True
            cave.cells[x][room.y + room.height - 1] = True
        for y in range(room.y, room.y + room.height):
            cave.cells[room.x][y] = True
            cave.cells[room.x + room.width - 1][y] = True

    # Run CA steps only within room interiors
    for _ in range(4):
        new_cells = [[cell for cell in row] for row in cave.cells]  # copy current state
        for room in rooms:
            for x in range(room.x + 1, room.x + room.width - 1):
                for y in range(room.y + 1, room.y + room.height - 1):
                    neighbors = cave.count_neighbors(x, y)
                    if cave.cells[x][y]:
                        new_cells[x][y] = neighbors >= 4
                    else:
                        new_cells[x][y] = neighbors >= 5
        cave.cells = new_cells


def find_cave_sections(cave: CaveMap) -> List[set[Point]]:
    """Find connected components of unblocked cells"""
    sections = []
    visited = set()

    for x in range(cave.width):
        for y in range(cave.height):
            if cave.cells[x][y] or Point(x, y) in visited:
                continue

            # New section - flood fill
            section = set()
            queue = [Point(x, y)]
            while queue:
                p = queue.pop(0)
                if p in visited:
                    continue
                visited.add(p)
                section.add(p)

                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    nx, ny = p.x + dx, p.y + dy
                    if (0 <= nx < cave.width and 0 <= ny < cave.height and
                        not cave.cells[nx][ny] and Point(nx, ny) not in visited):
                        queue.append(Point(nx, ny))

            sections.append(section)

    return sections

def connect_caves(cave: CaveMap, rooms: List[Room]):
    # First find all potential connections between nearby rooms
    connections = []
    for i, room1 in enumerate(rooms):
        for room2 in rooms[i+1:]:
            if room1.touches(room2, gap=4):
                connections.append((room1, room2))

    # Find initial cave sections
    sections = find_cave_sections(cave)

    # Find ALL potential connection points between ALL sections
    potential_connections = []
    for room1, room2 in connections:
        # Find which sections these rooms are part of
        section1 = next(s for s in sections
                       if any(Point(x, y) in s
                             for x in range(room1.x+1, room1.x+room1.width-1)
                             for y in range(room1.y+1, room1.y+room1.height-1)))
        section2 = next(s for s in sections
                       if any(Point(x, y) in s
                             for x in range(room2.x+1, room2.x+room2.width-1)
                             for y in range(room2.y+1, room2.y+room2.height-1)))

        # Find closest points between these sections
        min_dist = float('inf')
        best_points = None

        for p1 in section1:
            for p2 in section2:
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dist = dx*dx + dy*dy
                if dist < min_dist:
                    min_dist = dist
                    best_points = (p1, p2)

        if best_points:
            potential_connections.append((min_dist, best_points))

    # For each room pair, connect their cave sections
    for _, (p1, p2) in potential_connections:
        # Find which sections these rooms are part of
        section1 = next(s for s in sections
                       if any(Point(x, y) in s
                             for x in range(room1.x+1, room1.x+room1.width-1)
                             for y in range(room1.y+1, room1.y+room1.height-1)))
        section2 = next(s for s in sections
                       if any(Point(x, y) in s
                             for x in range(room2.x+1, room2.x+room2.width-1)
                             for y in range(room2.y+1, room2.y+room2.height-1)))

        if section1 == section2:
            continue

        # Find closest unblocked points between sections
        min_dist = float('inf')
        best_points = None

        for p1 in section1:
            for p2 in section2:
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dist = dx*dx + dy*dy
                if dist < min_dist:
                    min_dist = dist
                    best_points = (p1, p2)

        if best_points:
            p1, p2 = best_points
            # Draw wide diagonal corridor
            def plot_line(x0, y0, x1, y1):
                points = []
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                x, y = x0, y0
                sx = 1 if x1 > x0 else -1
                sy = 1 if y1 > y0 else -1

                if dx > dy:
                    err = dx / 2
                    while x != x1:
                        points.append((x, y))
                        err -= dy
                        if err < 0:
                            y += sy
                            err += dx
                        x += sx
                else:
                    err = dy / 2
                    while y != y1:
                        points.append((x, y))
                        err -= dx
                        if err < 0:
                            x += sx
                            err += dy
                        y += sy
                points.append((x, y))
                return points

            # Draw main corridor
            for x, y in plot_line(p1.x, p1.y, p2.x, p2.y):
                cave.cells[x][y] = False
                # Make corridor 3 wide
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < cave.width and 0 <= ny < cave.height:
                            cave.cells[nx][ny] = False

            # Merge sections
            sections.remove(section2)
            section1.update(section2)


def connect_rooms(cave: CaveMap, rooms: List[Room]):
    # Connect nearby rooms with corridors
    for i, room1 in enumerate(rooms):
        for room2 in rooms[i+1:]:
            if room1.touches(room2, gap=4):
                # Find closest points between rooms and draw corridor
                min_dist = float('inf')
                best_points = None

                for x1 in range(room1.x, room1.x + room1.width):
                    for y1 in range(room1.y, room1.y + room1.height):
                        for x2 in range(room2.x, room2.x + room2.width):
                            for y2 in range(room2.y, room2.y + room2.height):
                                dx = x2 - x1
                                dy = y2 - y1
                                dist = dx*dx + dy*dy
                                if dist < min_dist:
                                    min_dist = dist
                                    best_points = (x1, y1, x2, y2)

                if best_points:
                    x1, y1, x2, y2 = best_points
                    # Draw 3-wide corridor between points
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for t in range(101):
                                x = x1 + (x2-x1)*t//100 + dx
                                y = y1 + (y2-y1)*t//100 + dy
                                if 0 <= x < cave.width and 0 <= y < cave.height:
                                    cave.cells[x][y] = False

def generate_cave(width: int, height: int, seed: int = None) -> CaveMap:
    if seed is not None:
        random.seed(seed)

    cave = CaveMap(width, height)

    # Try to place rooms until we get good coverage
    while True:
        rooms = try_place_rooms(width, height)
        if rooms:
            break

    # Fill rooms with cave generation
    fill_caves(cave, rooms)

    # Connect nearby rooms
    connect_caves(cave, rooms)

    return cave


if __name__ == "__main__":
    cave = generate_cave(100, 100)
    cave.print()
