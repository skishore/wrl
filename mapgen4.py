from dataclasses import dataclass, field

import copy
import itertools
import math
import random

GLYPHS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "#": ('#', (32, 96, 0)),
    ".": ('.', (255, 255, 255)),
    ',': (',', (128, 192, 64)),
    '"': ('"', (64, 192, 0)),
    '~': ('~', (0, 128, 255)),
    'B': ('#', (192, 128, 0)),
    'F': ('=', (255, 128, 0)),
    'R': ('.', (255, 128, 0)),
    'S': ('.', (255, 255, 255)),
    'W': ('~', (0, 128, 255)),
}

@dataclass(frozen=True)
class Point:
    x: int
    y: int

@dataclass(eq=False)
class CaveRoom:
    points: set[Point]

@dataclass(frozen=True)
class RoomStep:
    min_size: int
    max_size: int
    attempts: int

@dataclass
class MapgenConfig:
    # Overall size
    width: int = 100
    height: int = 100

    # Room placement
    room_series: list[RoomStep] = field(default_factory=lambda: [
        #RoomStep(min_size=30, max_size=60, attempts=10),
        #RoomStep(min_size=25, max_size=50, attempts=15),
        #RoomStep(min_size=20, max_size=40, attempts=20),
        RoomStep(min_size=15, max_size=30, attempts=25),
        RoomStep(min_size=10, max_size=20, attempts=30),
    ])
    start_with_center: bool = True

    # Room interior generation
    wall_chance: float = 0.45
    birth_limit: int = 5
    death_limit: int = 4
    cave_steps: int = 3

    # Connections
    corridor_width: int = 2
    corridor_limit: float = 4.0
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
                (glyph, color) = GLYPHS.get(c, (c, (255, 255, 255)))
                if color != last_color:
                    (r, g, b) = color
                    line.append(f"\x1b[38;2;{r};{g};{b}m")
                    last_color = color
                line.append(chr(ord(glyph) - 0x20 + 0xFF00))
            print(''.join(line))
        print("\x1b[0m")


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


def find_cave_sections(cave: CaveMap, value: str) -> list[set[Point]]:
    """Find connected components of unblocked cells"""
    sections = []
    visited = set()

    for x in range(cave.width):
        for y in range(cave.height):
            if cave.cells[x][y] != value or Point(x, y) in visited:
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
                    if not (0 <= nx < cave.width and 0 <= ny < cave.height):
                        continue
                    if cave.cells[nx][ny] != value or Point(nx, ny) in visited:
                        continue
                    queue.append(Point(nx, ny))

            sections.append(section)

    return sections


def find_closest_pairs(section1: set[Point], section2: set[Point]) -> list[tuple[Point, Point]]:
    """Find closest pair of points between two sections"""
    best_pairs = []
    best_score = float('inf')

    for p1 in section1:
        for p2 in section2:
            score = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
            if score < best_score:
                best_pairs.clear()
                best_score = score
            if score == best_score:
                best_pairs.append((p1, p2))

    assert best_pairs
    return best_pairs


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


def generate_perlin_noise(width: int, height: int, scale: float = 10.0, octaves: int = 4, falloff = 0.5) -> list[list[float]]:
    """Generate Perlin noise in range [0,1]"""

    def interpolate(a0: float, a1: float, w: float) -> float:
        # Smoothstep interpolation
        return (a1 - a0) * (3.0 - w * 2.0) * w * w + a0

    noises = []
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

        noises.append(copy.deepcopy(noise))

    # Normalize each partial sum to [0,1]
    for noise in noises:
        max_val = max(max(row) for row in noise)
        min_val = min(min(row) for row in noise)
        for x in range(width):
            for y in range(height):
                noise[x][y] = (noise[x][y] - min_val) / (max_val - min_val)

    return noises


def generate_blue_noise(width: int, height: int, spacing: int) -> list[list[int]]:
    noise = [[0.0 for _ in range(height)] for _ in range(width)]
    points = [Point(x, y) for x in range(width) for y in range(height)]
    random.shuffle(points)
    selected = set()

    d = math.ceil(math.sqrt(spacing // 2))
    for point in points:
        okay = True
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                if dx * dx + dy * dy >= spacing:
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
    sections = []
    while len(sections) != 1:
        bool_cave = fill_cave(width, height, config)
        sections = find_cave_sections(bool_cave, False)
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
       valid = True

       # Check that no map floor overlaps a room wall
       for rw in room_walls:
           (x, y) = (offset[0] + rw[0], offset[1] + rw[1])
           if map_cave.cells[x][y] == '.':
               valid = False
               break
       if not valid:
           continue

       # Check that no room floor overlaps a map wall
       for rf in room_floors:
           (x, y) = (offset[0] + rf[0], offset[1] + rf[1])
           if map_cave.cells[x][y] == '#':
               valid = False
               break
       if not valid:
           continue

       # Place the cave
       for rx in range(room_cave.width):
           for ry in range(room_cave.height):
               if room_cave.cells[rx][ry] != ' ':
                   map_cave.cells[rx + offset[0]][ry + offset[1]] = room_cave.cells[rx][ry]
       return True

   return False

def build_lake(cave: CaveMap) -> None:
    #noise = noises[-1]
    #values = sorted([x for xs in noise for x in xs])
    #mid = values[int(0.5 * len(values))]

    #for x in range(width):
    #    for y in range(height):
    #        if noise[x][y] > mid + 0.0 * random.random():
    #            cave.cells[x][y] = '#'
    #        else:
    #            cave.cells[x][y] = '.'

    #noise = noises[0]
    #xs = list(range(int(0.25 * width), int(0.75 * width)))
    #ys = list(range(int(0.65 * height), int(0.85 * height)))
    #ps = [(x, y) for x in xs for y in ys]
    #root = min(ps, key=lambda p: noise[p[0]][p[1]])

    #visited = set()
    #frontier = [root]
    #dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    #bias = lambda p: math.exp(-(((p[0] - root[0]) / 64.0) ** 2 + ((p[1] - root[1]) / 16.0) ** 2))
    #while frontier and len(visited) < 100:
    #    c = 0.5 * len(visited) / 100
    #    p = min(frontier, key=lambda p: c * noise[p[0]][p[1]] - bias(p))
    #    frontier = [x for x in frontier if x != p]
    #    if p not in visited:
    #        cave.cells[p[0]][p[1]] = '~'
    #        visited.add(p)
    #        for d in dirs:
    #            (x, y) = (p[0] + d[0], p[1] + d[1])
    #            if 0 <= x < width and 0 <= y < height:
    #                frontier.append((x, y))

    #return cave
    pass

def find_tree_edges(vertices: list[any], edges: dict[tuple[any, any], float]) -> list[tuple[any, any]]:
    result = []
    groups = [[x] for x in vertices]

    while len(groups) > 1:
        best_edges = []
        best_score = float("inf")

        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                for edge in itertools.product(g1, g2):
                    score = edges[edge]
                    if score < best_score:
                        best_edges.clear()
                        best_score = score
                    if score == best_score:
                        best_edges.append(((g1, g2), edge))

        assert best_edges
        print(f"Selected edge of weight: {best_score}")
        (g1, g2), edge = random.choice(best_edges)
        groups = [x for x in groups if not (x is g1 or x is g2)] + [g1 + g2]
        result.append(edge)

    return result


def generate_cave_map(config: MapgenConfig) -> CaveMap:
    (width, height) = (config.width, config.height)
    cave = CaveMap(width, height)

    # Initialize with undecided cells
    for x in range(width):
        for y in range(height):
            cave.cells[x][y] = ' '

    # Place first room in center
    room_config = config.room_series[0]
    rw = random.randint(room_config.min_size, room_config.max_size)
    rh = random.randint(room_config.min_size, room_config.max_size)
    room = create_room_cave(rw, rh, config)

    # Center it
    x = (width - rw) // 2
    y = (height - rh) // 2
    for rx in range(rw):
        for ry in range(rh):
            if room.cells[rx][ry] != ' ':
                cave.cells[x + rx][y + ry] = room.cells[rx][ry]

    # Try to place more rooms
    for room_config in config.room_series:
        for _ in range(room_config.attempts):
            rw = random.randint(room_config.min_size, room_config.max_size)
            rh = random.randint(room_config.min_size, room_config.max_size)
            room = create_room_cave(rw, rh, config)
            try_place_cave(cave, room, config)

    # Connect up the rooms, with some cycles
    rooms = find_cave_sections(cave, '.')
    rooms = [CaveRoom(points=x) for x in rooms]
    edges: dict[tuple[CaveRoom, CaveRoom], float] = {}
    for i, r1 in enumerate(rooms):
        for r2 in rooms[i + 1:]:
            (pa, pb) = find_closest_pairs(r1.points, r2.points)[0]
            distance = math.sqrt((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2)
            edges[(r1, r2)] = distance
            edges[(r2, r1)] = distance

    tree = find_tree_edges(rooms, edges)

    loops = []
    for i, r1 in enumerate(rooms):
        for r2 in rooms[i + 1:]:
            e1, e2 = (r1, r2), (r2, r1)
            if e1 not in tree and e2 not in tree and edges[e1] < config.corridor_limit:
                loops.append(e1)

    for r1, r2 in tree + loops:
        corridor = config.corridor_width
        (l, r) = (-corridor // 2, -corridor // 2 + corridor)
        p1, p2 = random.choice(find_closest_pairs(r1.points, r2.points))
        for x, y in plot_line(p1, p2):
            cave.cells[x][y] = '.'
            for dx in range(l, r):
                for dy in range(l, r):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < cave.width and 0 <= ny < cave.height:
                        cave.cells[nx][ny] = '.'

    (lw, lh) = (24, 12)
    (ls, lt) = (config.width - lw, config.height - lh)
    lx = int((0.50 + 0.25 * random.random()) * ls)
    ly = int((0.75 + 0.25 * random.random()) * lt)
    while True:
        lake = create_room_cave(lw, lh, config)
        if len(find_cave_sections(lake, '#')) == 1:
            break
    mapping = {'#': 'S', '.': '~'}
    for x in range(lw):
        for y in range(lh):
            tile = mapping.get(lake.cells[x][y])
            if tile is not None:
                cave.cells[x + lx][y + ly] = tile

    inf = float('inf')
    noises = generate_perlin_noise(width, height, scale=4.0, octaves=2, falloff=0.65)
    berry_blue_noise = generate_blue_noise(width, height, spacing=10)
    trees_blue_noise = generate_blue_noise(width, height, spacing=5)
    noise = noises[-1]

    # Safe accessor for the map matrix
    def get_tile(p: tuple[int, int]) -> str | None:
        (x, y) = p
        if not (0 <= x < width and 0 <= y < height):
            return None
        return cave.cells[x][y]

    # Noise function used to build a river
    river_costs = {
        '#': 64.0,
        ' ': 64.0,
    }

    def river_score_one(p: tuple[int, int], center: bool) -> float:
        tile = get_tile(p)
        if tile is None:
            return inf if center else river_costs[' ']

        (x, y) = p
        noise_score = 8.0 * (1.0 - noise[x][y])
        tiles_score = river_costs.get(tile, 0)
        return noise_score + tiles_score

    def river_score(p: tuple[int, int]) -> float:
        (x, y) = p
        result = 0
        for dx in (-1, 0, 1, 2):
            for dy in (-1, 0, 1):
                center = dx in (0, 1) and dy == 0
                result += river_score_one((x + dx, y + dy), center=center)
        return result

    # Dijkstra to build a river...
    complete = {}
    frontier = {}
    for x in range(int(0.25 * width), int(0.75 * width)):
        frontier[(x, 0)] = (river_score((x, 0)), None)
    while frontier:
        (prev, result) = min(frontier.items(), key=lambda x: x[1][0])
        assert result[0] != inf
        assert prev not in complete
        complete[prev] = result
        del frontier[prev]

        if cave.cells[prev[0]][prev[1]] == '~':
            current = prev
            while current is not None:
                cave.cells[current[0]][current[1]] = 'W'
                cave.cells[current[0] + 1][current[1]] = 'W'
                current = complete[current][1]
            break

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                (x, y) = (prev[0] + dx, prev[1] + dy)
                if (x, y) in complete:
                    continue

                point = (x, y)
                score = result[0] + math.sqrt(dx ** 2 + dy ** 2) + river_score(point)
                if score == inf:
                    continue

                current = frontier.get(point)
                if current is None or current[0] > score:
                    frontier[point] = (score, prev)

    ordered = list(rooms)
    random.shuffle(ordered)
    l1 = int(round(0.2 * len(ordered)))
    l2 = int(round(0.4 * len(ordered)))
    groves = set(ordered[:l1])
    thickets = set(ordered[l1:l2])
    can_plant_grass = {'.', 'S'}
    can_plant_trees = {'.'}

    for room in rooms:
        values = {}
        for p in room.points:
            (x, y) = (p.x, p.y)
            if cave.cells[x][y] in can_plant_grass:
                values[p] = noise[x][y] + 0.3 * random.random()

        if not values:
            continue

        if room in groves:
            for p in values:
                (x, y) = (p.x, p.y)
                if cave.cells[x][y] in can_plant_trees:
                    if berry_blue_noise[x][y] > 0:
                        cave.cells[x][y] = 'B'
            grassiness = 0.0 + 0.2 * random.random()
        elif room in thickets:
            for p in values:
                (x, y) = (p.x, p.y)
                if cave.cells[x][y] in can_plant_trees:
                    if trees_blue_noise[x][y] > 0:
                        cave.cells[x][y] = '#'
            grassiness = 0.0 + 0.6 * random.random()
        else:
            grassiness = 0.2 + 0.2 * random.random()

        target = int(round(grassiness * len(values)))
        grass_count = min(max(0, target), len(values) - 1)

        ordered = sorted(values.values())
        grass_threshold = ordered[grass_count]

        for p in values:
            (x, y) = (p.x, p.y)
            if cave.cells[x][y] in can_plant_grass:
                if values[p] < grass_threshold:
                    cave.cells[x][y] = '"'

    # Noise function used to build a route
    route_costs = {
        '~': inf,
        '#': 64,
        ' ': 64,
        'B': 64,
        '"': 16,
        'W': 16,
        ',': 4,
        'S': 4,
    }

    def route_score_one(p: tuple[int, int], center=False) -> float:
        tile = get_tile(p)
        if tile is None:
            return inf if center else route_costs[' ']
        elif tile == 'W' and center:
            return inf

        (x, y) = p
        noise_score = 8.0 * (1.0 - noise[x][y])
        tiles_score = 100 * route_costs.get(tile, 0)
        return noise_score + tiles_score

    def route_score(p: tuple[int, int]) -> float:
        (x, y) = p
        result = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                center = dx == 0 and dy == 0
                result += route_score_one((x + dx, y + dy), center=center)
        return result

    def relax(source: tuple[int, int], target: tuple[int, int], score: float) -> None:
        if target in complete:
            return

        (dx, dy) = (target[0] - source[0], target[1] - source[1])
        target_score = score + math.sqrt(dx ** 2 + dy ** 2) + route_score(target)
        if target_score == inf:
            return

        current = frontier.get(target)
        if current is None or current[0] > target_score:
            frontier[target] = (target_score, source)

    # Dijkstra to build a route...
    complete = {}
    frontier = {}
    for y in range(int(0.25 * width), int(0.75 * width)):
        frontier[(0, y)] = (noise[0][y], None)

    while frontier:
        (source, (score, parent)) = min(frontier.items(), key=lambda x: x[1][0])
        assert score != inf
        assert source not in complete
        complete[source] = (score, parent)
        del frontier[source]

        if source[0] == width - 1:
            current = source
            while current is not None:
                cave.cells[current[0]][current[1]] = 'R'
                next = complete[current][1]
                if next is not None and current[0] - next[0] > 1:
                    for x in range(next[0] + 1, current[0]):
                        cave.cells[x][current[1]] = 'F'
                current = next
            break

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                target = (source[0] + dx, source[1] + dy)
                relax(source, target, score)

        (x, y) = source
        if get_tile((x + 1, y)) == 'W' and get_tile((x + 2, y)) == 'W':
            relax(source, (x + 3, y), score)

    for x in range(width):
        for y in range(height):
            if cave.cells[x][y] == ' ':
                cave.cells[x][y] = '#'

    return cave

if __name__ == "__main__":
    config = MapgenConfig()
    cave = generate_cave_map(config)
    cave.print()
