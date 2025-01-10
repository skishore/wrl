from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Point:
    x: int
    y: int

@dataclass
class MapgenConfig:
    # Room placement
    min_room_size: int = 20
    max_room_size: int = 60
    room_attempts: int = 10000
    min_coverage: float = 0.75

    # Room interior generation
    wall_chance: float = 0.45
    birth_limit: int = 5
    death_limit: int = 4
    cave_steps: int = 2

    # Connections
    corridor_width: int = 1
    max_connection_gap: int = 20

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
            chars = []
            for x in range(self.width):
                c = '#' if self.cells[x][y] else '.'
                chars.append(chr(ord(c) - 0x20 + 0xFF00))
            print(''.join(chars))

def try_place_rooms(width: int, height: int, config: MapgenConfig) -> list[Room] | None:
    rooms = []
    total_area = width * height
    room_area = 0

    # Place first room roughly in center
    first_w = random.randint(config.min_room_size, config.max_room_size + 1)
    first_h = random.randint(config.min_room_size, config.max_room_size + 1)
    first_x = (width - first_w) // 2
    first_y = (height - first_h) // 2
    rooms.append(Room(first_x, first_y, first_w, first_h))
    room_area = first_w * first_h

    for _ in range(config.room_attempts):
        w = random.randint(config.min_room_size, config.max_room_size + 1)
        h = random.randint(config.min_room_size, config.max_room_size + 1)
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

    return rooms if room_area / total_area >= config.min_coverage else None


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

                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
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

def generate_cave(width: int, height: int, config: MapgenConfig = None,
                 seed: int = None) -> CaveMap:
    if seed is not None:
        random.seed(seed)
    if config is None:
        config = MapgenConfig()

    cave = CaveMap(width, height)

    # Try to place rooms until we get good coverage
    while True:
        rooms = try_place_rooms(width, height, config)
        if rooms is not None:
            break

    # Fill rooms with cave generation
    fill_caves(cave, rooms, config)

    # Connect nearby rooms
    connect_caves(cave, rooms, config)

    return cave

if __name__ == "__main__":
    cave = generate_cave(100, 100)
    cave.print()
