from dataclasses import dataclass
import math
import random

COLORS: dict[str, tuple[int, int, int]] = {
    "#": (0, 84, 0),
    ".": (168, 168, 168),
    '"': (84, 168, 0),
}

@dataclass(frozen=True)
class Point:
    x: int
    y: int

@dataclass
class MapgenConfig:
    # Overall size
    width: int = 100
    height: int = 100

    # Room placement
    min_room_size: int = 10
    max_room_size: int = 60
    room_attempts: int = 100
    min_coverage: float = 0.40
    start_with_center: bool = True

    # Room interior generation
    wall_chance: float = 0.45
    birth_limit: int = 5
    death_limit: int = 4
    cave_steps: int = 3

    # Connections
    corridor_width: int = 1
    max_connection_gap: int = 6

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
        return (self.x - gap <= other.x + other.width and self.x + self.width + gap >= other.x and
                self.y - gap <= other.y + other.height and self.y + self.height + gap >= other.y)

class CaveMap:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cells = [[True for _ in range(height)] for _ in range(width)]

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

    def print(self):
        for y in range(self.height):
            line = []
            last_color = None
            for x in range(self.width):
                c = self.cells[x][y]
                if c == ' ':
                    line.append('  ')
                    continue
                color = COLORS.get(c, (255, 255, 255))
                if color != last_color:
                    (r, g, b) = color
                    line.append(f"\x1b[38;2;{r};{g};{b}m")
                    last_color = color
                line.append(chr(ord(c) - 0x20 + 0xFF00))
            print(''.join(line))
        print("\x1b[0m")

def place_rooms(width: int, height: int, config: MapgenConfig) -> list[Room]:
    rooms = []
    total_area = width * height

    # Place first room roughly in center
    if config.start_with_center:
        first_w = random.randint(config.min_room_size, config.max_room_size + 1)
        first_h = random.randint(config.min_room_size, config.max_room_size + 1)
        first_x = (width - first_w) // 2
        first_y = (height - first_h) // 2
        rooms.append(Room(first_x, first_y, first_w, first_h))

    for _ in range(config.room_attempts):
        w = random.randint(config.min_room_size, config.max_room_size + 1)
        h = random.randint(config.min_room_size, config.max_room_size + 1)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        new_room = Room(x, y, w, h)

        # Check if touches any existing room
        touches_existing = not rooms
        for room in rooms:
            if new_room.touches(room):
                touches_existing = True
            if new_room.overlaps(room):
                touches_existing = False
                break

        if touches_existing:
            rooms.append(new_room)

    return rooms

def fill_cave(width: int, height: int, config: MapgenConfig) -> CaveMap:
    cave = CaveMap(width, height)

    # Initialize the room's interior with uniform random noise
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            cave.cells[x][y] = random.random() < config.wall_chance

    # Run CA steps only within the map's interior
    for _ in range(config.cave_steps):
        new_cells = [[cell for cell in row] for row in cave.cells]
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                neighbors = cave.count_neighbors(x, y)
                if cave.cells[x][y]:
                    new_cells[x][y] = neighbors >= config.death_limit
                else:
                    new_cells[x][y] = neighbors >= config.birth_limit
        cave.cells = new_cells

    return cave

def fill_caves(cave: CaveMap, rooms: list[Room], config: MapgenConfig):
    for room in rooms:
        while True:
            room_cave = fill_cave(room.width, room.height, config)
            if len(find_cave_sections(room_cave)) == 1:
                break
        for x in range(room.width):
            for y in range(room.height):
                cave.cells[room.x + x][room.y + y] = room_cave.cells[x][y]


def find_cave_sections(cave: CaveMap) -> list[set[Point]]:
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

                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nx, ny = p.x + dx, p.y + dy
                    if (0 <= nx < cave.width and 0 <= ny < cave.height and
                        not cave.cells[nx][ny] and Point(nx, ny) not in visited):
                        queue.append(Point(nx, ny))

            sections.append(section)

    return sections

def find_closest_points(section1: set[Point], section2: set[Point]) -> tuple[Point, Point]:
    """Find closest pair of points between two sections"""
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

    assert best_points is not None
    return best_points

def plot_line(start: Point, end: Point) -> list[tuple[int, int]]:
    """Bresenham's line algorithm"""
    points = []
    dx = abs(end.x - start.x)
    dy = abs(end.y - start.y)
    x, y = start.x, start.y
    sx = 1 if end.x > start.x else -1
    sy = 1 if end.y > start.y else -1

    if dx > dy:
        err = dx / 2
        while x != end.x:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2
        while y != end.y:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points


def room_contains_section(room: Room, section: set[Point]) -> bool:
    return any(Point(x, y) in section
               for x in range(room.x + 1, room.x + room.width)
               for y in range(room.y + 1, room.y + room.height))


def connect_caves(cave: CaveMap, rooms: list[Room], config: MapgenConfig):
    # Find all potential connections between nearby rooms
    connections = []
    gap = config.max_connection_gap
    for i, room1 in enumerate(rooms):
        for room2 in rooms[i+1:]:
            if room1.touches(room2, gap=config.max_connection_gap):
                connections.append((room1, room2))

    # For each connection, find the closest points between
    # their respective cave sections and connect them
    sections = find_cave_sections(cave)
    for room1, room2 in connections:
        # Find which sections these rooms belong to
        section1 = set()
        section2 = set()
        for s in sections:
            if room_contains_section(room1, s):
                section1 |= s
            if room_contains_section(room2, s):
                section2 |= s

        p1, p2 = find_closest_points(section1, section2)
        (dx, dy) = (p1.x - p2.x, p1.y - p2.y)
        if dx * dx + dy * dy > gap * gap:
            continue

        # Draw corridor
        for x, y in plot_line(p1, p2):
            cave.cells[x][y] = False
            # Make corridor wider
            for dx in range(-config.corridor_width//2, config.corridor_width//2 + 1):
                for dy in range(-config.corridor_width//2, config.corridor_width//2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < cave.width and 0 <= ny < cave.height:
                        cave.cells[nx][ny] = False

def generate_perlin_noise(width: int, height: int, scale: float = 10.0, octaves: int = 4, falloff = 0.5) -> list[list[float]]:
    """Generate Perlin noise in range [0,1]"""

    def interpolate(a0: float, a1: float, w: float) -> float:
        # Smoothstep interpolation
        return (a1 - a0) * (3.0 - w * 2.0) * w * w + a0

    noise = [[0.0 for _ in range(height)] for _ in range(width)]

    for octave in range(octaves):
        period = scale / (2 ** octave)
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

                noise[x][y] += value * falloff ** octave

    # Normalize to [0,1]
    max_val = max(max(row) for row in noise)
    min_val = min(min(row) for row in noise)
    for x in range(width):
        for y in range(height):
            noise[x][y] = (noise[x][y] - min_val) / (max_val - min_val)

    return noise

def generate_colored_noise(width: int, height: int) -> list[list[int]]:
    sines = {}
    tau = 2 * math.pi
    #frequencies = list(range(1, 31))
    frequencies = list(range(1, 50))
    for f in frequencies:
        x_phase = random.uniform(0, 1)
        y_phase = random.uniform(0, 1)
        sines[f] = [
            [
                math.sin(tau * f * (x / width + x_phase)) +
                math.sin(tau * f * (y / height + y_phase))
                for y in range(height)
            ]
            for x in range(width)
        ]

    amplitude = lambda f: f ** 1
    norm = sum(amplitude(f) for f in frequencies)
    noise = [[0.0 for _ in range(height)] for _ in range(width)]
    for x in range(width):
        for y in range(height):
            noise[x][y] = sum(amplitude(f) * sines[f][x][y] for f in frequencies) / norm

    return noise

def generate_blue_noise(width: int, height: int) -> list[list[int]]:
    noise = [[0.0 for _ in range(height)] for _ in range(width)]
    points = [Point(x, y) for x in range(width) for y in range(height)]
    random.shuffle(points)
    selected = set()

    min_l2_distance = 8
    d = math.ceil(math.sqrt(min_l2_distance // 2))
    for point in points:
        okay = True
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                if dx * dx + dy * dy >= min_l2_distance:
                    continue
                other = Point(point.x + dx, point.y + dy)
                if other in selected:
                    okay = False
        if okay:
            noise[point.x][point.y] = 1.0
            selected.add(point)

    return noise

def generate_bluish_noise(width: int, height: int, base: list[list[int]]) -> list[list[int]]:
    noise = [[0.0 for _ in range(height)] for _ in range(width)]
    points = [Point(x, y) for x in range(width) for y in range(height)]
    random.shuffle(points)
    selected = set()

    for point in points:
        okay = True
        here = base[point.x][point.y]
        min_l2_distance = math.pow(8.0 * here, 1.0)
        d = math.ceil(math.sqrt(min_l2_distance // 2))
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                if dx * dx + dy * dy >= min_l2_distance:
                    continue
                other = Point(point.x + dx, point.y + dy)
                if other in selected:
                    okay = False
        if okay:
            noise[point.x][point.y] = 1.0
            selected.add(point)

    return noise

def generate_cave(config: MapgenConfig, seed: int = None) -> CaveMap:
    (width, height) = (config.width, config.height)

    if seed is not None:
        random.seed(seed)

    cave = CaveMap(width, height)

    # Try to place rooms until we get good coverage
    while True:
        rooms = place_rooms(width, height, config)
        area = sum(room.width * room.height for room in rooms)
        if area / (width * height) >= config.min_coverage:
            break

    # Fill rooms with cave generation
    fill_caves(cave, rooms, config)

    # Detect each cave connected-component
    sections = find_cave_sections(cave)

    # Connect nearby rooms
    connect_caves(cave, rooms, config)

    noise = generate_perlin_noise(width, height, scale=4.0, octaves=2, falloff=0.65)
    #noise = generate_bluish_noise(width, height, noise)

    for x in range(width):
        for y in range(height):
            base = cave.cells[x][y]
            cave.cells[x][y] = '#' if base else '"' if noise[x][y] > 0.45 + 0.3 * random.random() else '.'

    return cave

def generate_room_cave(width: int, height: int, config: MapgenConfig) -> CaveMap:
    """Generate a single room's cave, ensuring it has exactly one connected component."""
    while True:
        cave = fill_cave(width, height, config)
        sections = find_cave_sections(cave)
        if len(sections) == 1:
            return cave

def convert_to_three_state(cave: CaveMap) -> CaveMap:
    """Convert a boolean cave (True=wall) to three-state (' ', '#', '.')"""
    result = CaveMap(cave.width, cave.height)
    # Start with undecided cells
    for x in range(cave.width):
        for y in range(cave.height):
            result.cells[x][y] = ' '

    # Mark floors and their adjacent walls
    for x in range(cave.width):
        for y in range(cave.height):
            if not cave.cells[x][y]:  # False = floor in input
                result.cells[x][y] = '.'
                # Mark adjacent cells as walls
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < cave.width and
                            0 <= ny < cave.height and
                            result.cells[nx][ny] == ' '):
                            result.cells[nx][ny] = '#'

    return result

def create_room_cave(width: int, height: int, config: MapgenConfig) -> CaveMap:
    """Generate a room cave and convert it to three-state format"""
    bool_cave = generate_room_cave(width, height, config)
    return convert_to_three_state(bool_cave)

def try_place_cave(map_cave: CaveMap, room_cave: CaveMap, config: MapgenConfig) -> bool:
   """Try to place room_cave onto map_cave such that:
   1. A wall from the new room touches a wall from the map
   2. No floor from either cave overlaps a wall from the other"""

   # Get walls and floors from both caves
   map_walls = {(x, y) for x in range(map_cave.width)
                for y in range(map_cave.height) if map_cave.cells[x][y] == '#'}
   room_walls = {(x, y) for x in range(room_cave.width)
                for y in range(room_cave.height) if room_cave.cells[x][y] == '#'}
   room_floors = {(x, y) for x in range(room_cave.width)
                 for y in range(room_cave.height) if room_cave.cells[x][y] == '.'}

   # Find all possible offsets where walls could align
   offsets = set()
   for mw in map_walls:
       for rw in room_walls:
           # Offset that would place room_wall at map_wall
           offset = (mw[0] - rw[0], mw[1] - rw[1])
           if (0 <= offset[0] < map_cave.width - room_cave.width and
               0 <= offset[1] < map_cave.height - room_cave.height):
               offsets.add(offset)

   # Try random offsets until we find one that works
   offsets = list(offsets)
   random.shuffle(offsets)

   for offset in offsets:
       # Check that no room floor overlaps a map wall
       valid = True
       for rf in room_floors:
           p = (offset[0] + rf[0], offset[1] + rf[1])
           if p in map_walls:
               valid = False
               break
       if not valid:
           continue

       # Check that no map floor overlaps a room wall
       for rw in room_walls:
           p = (offset[0] + rw[0], offset[1] + rw[1])
           if map_cave.cells[p[0]][p[1]] == '.':
               valid = False
               break
       if not valid:
           continue

       # Check that at least one wall touches
       touches = False
       for rw in room_walls:
           p = (offset[0] + rw[0], offset[1] + rw[1])
           if p in map_walls:
               touches = True
               break
       if not touches:
           continue

       # Place the cave
       for rx in range(room_cave.width):
           for ry in range(room_cave.height):
               if room_cave.cells[rx][ry] != ' ':
                   map_cave.cells[rx + offset[0]][ry + offset[1]] = room_cave.cells[rx][ry]
       return True

   return False

def generate_cave_map(config: MapgenConfig) -> CaveMap:
   cave = CaveMap(config.width, config.height)
   # Initialize with undecided cells
   for x in range(config.width):
       for y in range(config.height):
           cave.cells[x][y] = ' '

   # Place first room in center
   width = random.randint(config.min_room_size, config.max_room_size)
   height = random.randint(config.min_room_size, config.max_room_size)
   room = create_room_cave(width, height, config)

   # Center it
   x = (config.width - width) // 2
   y = (config.height - height) // 2
   for rx in range(width):
       for ry in range(height):
           if room.cells[rx][ry] != ' ':
               cave.cells[x + rx][y + ry] = room.cells[rx][ry]

   # Try to place more rooms
   attempts = config.room_attempts
   while attempts > 0:
       width = random.randint(config.min_room_size, config.max_room_size)
       height = random.randint(config.min_room_size, config.max_room_size)
       room = create_room_cave(width, height, config)

       if not try_place_cave(cave, room, config):
           attempts -= 1
           continue

       attempts = config.room_attempts  # Reset attempts on success

       # Check coverage
       floor_cells = sum(1 for x in range(config.width)
                        for y in range(config.height)
                        if cave.cells[x][y] == '.')
       if floor_cells / (config.width * config.height) >= config.min_coverage:
           break

   return cave

if __name__ == "__main__":
    config = MapgenConfig()
    cave = generate_cave_map(config)
    cave.print()
