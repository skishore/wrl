from dataclasses import dataclass
from enum import IntEnum
import random
from typing import List, Set, Dict, Optional, Tuple

class Tile(IntEnum):
    DEFAULT = 0
    FREE = 1
    WALL = 2
    DOOR = 3
    FENCE = 4

@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return (self.x * self.x + self.y * self.y) ** 0.5

    def equals(self, other) -> bool:
        return self.x == other.x and self.y == other.y

@dataclass
class Rect:
    size: Point
    position: Point

class Room:
    def __init__(self):
        self.squares: List[Point] = []

    def get_random_square(self) -> Point:
        return random.choice(self.squares)

def in_bounds(point: Point, size: Point) -> bool:
    return 0 <= point.x < size.x and 0 <= point.y < size.y

def is_tile_blocked(tile: Tile) -> bool:
    return tile == Tile.DEFAULT or tile == Tile.WALL

def rect_to_rect_distance(rect1: Rect, rect2: Rect) -> float:
    distance = Point(
        max(rect1.position.x - rect2.position.x - rect2.size.x,
            rect2.position.x - rect1.position.x - rect1.size.x, 0),
        max(rect1.position.y - rect2.position.y - rect2.size.y,
            rect2.position.y - rect1.position.y - rect1.size.y, 0))
    return distance.length()

# Movement constants
BISHOP_MOVES = [Point(1, 1), Point(1, -1), Point(-1, 1), Point(-1, -1)]
KING_MOVES = [Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
              Point(-1, 0), Point(-1, -1), Point(0, -1), Point(1, -1)]
ROOK_MOVES = [Point(1, 0), Point(-1, 0), Point(0, 1), Point(0, -1)]

def add_door(square: Point, tiles: List[List[Tile]], diggable: List[List[bool]]) -> None:
    for step in KING_MOVES:
        neighbor = square + step
        if is_tile_blocked(tiles[neighbor.x][neighbor.y]):
            diggable[neighbor.x][neighbor.y] = False
    if random.random() < 0.5:
        tiles[square.x][square.y] = Tile.DOOR

class Level:
    def __init__(self, size: Point):
        self.size = size
        self.tiles = [[Tile.DEFAULT for _ in range(size.y)] for _ in range(size.x)]
        self.rids = [[0 for _ in range(size.y)] for _ in range(size.x)]
        self.diggable = [[True for _ in range(size.y)] for _ in range(size.x)]

    def add_walls(self):
        for x in range(self.size.x):
            for y in range(self.size.y):
                if is_tile_blocked(self.tiles[x][y]):
                    continue
                for move in KING_MOVES:
                    square = Point(x + move.x, y + move.y)
                    if (in_bounds(square, self.size) and
                        self.tiles[square.x][square.y] == Tile.DEFAULT):
                        self.tiles[square.x][square.y] = Tile.WALL

    def place_rectangular_room(self, rect: Rect, separation: int,
                             rects: List[Rect]) -> bool:
        for other in rects:
            if rect_to_rect_distance(rect, other) < separation:
                return False
        room_index = len(rects) + 1
        for x in range(rect.size.x):
            for y in range(rect.size.y):
                px, py = x + rect.position.x, y + rect.position.y
                self.tiles[px][py] = Tile.FREE
                self.rids[px][py] = room_index
        rects.append(rect)
        return True

    def dig_corridor(self, rooms: List[Room], index1: int, index2: int,
                    windiness: float) -> bool:
        """Digs a corridor between two rooms. Windiness (1.0-8.0) affects path length."""
        room1, room2 = rooms[index1], rooms[index2]
        source = room1.get_random_square()
        target = room2.get_random_square()

        assert in_bounds(source, self.size) and self.diggable[source.x][source.y]
        assert in_bounds(target, self.size) and self.diggable[target.x][target.y]

        # Run Dijkstra's
        visited = set()
        parents = {}
        distances = {source: 0}

        while distances and target not in visited:
            # Get closest unvisited node
            curr_point = min(distances.keys(), key=lambda p: distances[p])
            curr_dist = distances.pop(curr_point)
            visited.add(curr_point)

            # Check neighbors
            for step in ROOK_MOVES:
                child = curr_point + step
                if not in_bounds(child, self.size) or not self.diggable[child.x][child.y]:
                    continue
                if child in visited:
                    continue

                blocked = is_tile_blocked(self.tiles[child.x][child.y])
                distance = curr_dist + (2.0 if blocked else windiness)

                if child not in distances or distance < distances[child]:
                    distances[child] = distance
                    parents[child] = curr_point

        if target not in visited:
            return False

        # Build path
        node = target
        path = [node]
        while not node.equals(source):
            node = parents[node]
            path.append(node)

        # Truncate path to portions outside rooms
        truncated = []
        in_room2 = True
        for node in path:
            if in_room2 and self.rids[node.x][node.y] != index2 + 1:
                in_room2 = False
            if not in_room2:
                truncated.append(node)
                if self.rids[node.x][node.y] == index1 + 1:
                    break

        # Ensure exactly one connection point to each room
        def has_neighbor_in_room(point: Point, room_idx: int) -> Tuple[bool, Optional[Point]]:
            for step in ROOK_MOVES:
                neighbor = point + step
                if self.rids[neighbor.x][neighbor.y] == room_idx:
                    return True, neighbor
            return False, None

        while len(truncated) > 2:
            has_conn, neighbor = has_neighbor_in_room(truncated[2], index2 + 1)
            if not has_conn:
                break
            truncated = truncated[2:]
            truncated.insert(0, neighbor)

        while len(truncated) > 2:
            has_conn, neighbor = has_neighbor_in_room(
                truncated[-3], index1 + 1)
            if not has_conn:
                break
            truncated = truncated[:-2]
            truncated.append(neighbor)

        # Dig the corridor
        assert len(truncated) > 2
        for i in range(1, len(truncated)-1):
            node = truncated[i]
            if is_tile_blocked(self.tiles[node.x][node.y]):
                self.tiles[node.x][node.y] = Tile.FREE

        add_door(truncated[1], self.tiles, self.diggable)
        add_door(truncated[-2], self.tiles, self.diggable)
        return True

    def erode(self, islandness: int):
        def can_erode_square(rids: List[List[int]], square: Point) -> Tuple[bool, int]:
            room_index = 0
            has_free_orthogonal_neighbor = False
            min_unblocked_index = -1
            max_unblocked_index = -1
            gaps = 0

            for i in range(8):
                step = KING_MOVES[i]
                px, py = square.x + step.x, square.y + step.y
                adjacent = rids[px][py]
                if adjacent == 0:
                    continue
                room_index = adjacent
                has_free_orthogonal_neighbor |= (i % 2 == 0)
                if min_unblocked_index < 0:
                    min_unblocked_index = i
                    max_unblocked_index = i
                    continue
                if i > max_unblocked_index + 1:
                    gaps += 1
                max_unblocked_index = i

            if (min_unblocked_index >= 0 and
                not (min_unblocked_index == 0 and max_unblocked_index == 7)):
                gaps += 1
            return gaps <= 1 and has_free_orthogonal_neighbor, room_index

        new_rids = [[x for x in row] for row in self.rids]
        for x in range(1, self.size.x - 1):
            for y in range(1, self.size.y - 1):
                can_erode, room_index = can_erode_square(new_rids, Point(x, y))
                if not can_erode:
                    continue

                neighbors_blocked = 0
                for step in KING_MOVES:
                    if self.rids[x + step.x][y + step.y] == 0:
                        neighbors_blocked += 1

                blocked = self.rids[x][y] == 0
                matches = neighbors_blocked if blocked else 8 - neighbors_blocked
                inverse_blocked_to_free = 2
                inverse_free_to_blocked = 4
                cutoff = max(8 - matches, matches - 8 + islandness)

                changed = False
                if blocked:
                    changed = (random.randint(0, 8*inverse_blocked_to_free - 1) <
                             8 - matches)
                else:
                    changed = (random.randint(0, 8*inverse_free_to_blocked - 1) <
                             cutoff)

                if changed:
                    new_rids[x][y] = room_index if blocked else 0
                    self.tiles[x][y] = Tile.FREE if blocked else Tile.DEFAULT
        self.rids = new_rids

    def extract_final_rooms(self, n: int) -> List[Room]:
        result = [Room() for _ in range(n)]
        for x in range(self.size.x):
            for y in range(self.size.y):
                room_index = self.rids[x][y]
                assert (room_index == 0) == is_tile_blocked(self.tiles[x][y])
                if room_index == 0:
                    continue
                assert room_index - 1 < n
                result[room_index - 1].squares.append(Point(x, y))

                for step in BISHOP_MOVES:
                    neighbor = Point(x + step.x, y + step.y)
                    if is_tile_blocked(self.tiles[neighbor.x][neighbor.y]):
                        adjacent_to_room = False
                        for step_two in ROOK_MOVES:
                            neighbor_two = neighbor + step_two
                            if (in_bounds(neighbor_two, self.size) and
                                room_index == self.rids[neighbor_two.x][neighbor_two.y]):
                                adjacent_to_room = True
                        if not adjacent_to_room:
                            self.diggable[neighbor.x][neighbor.y] = False
        return result

    def to_debug_string(self, show_rooms: bool = False) -> str:
        chars = ' .#+'
        result = ''
        for y in range(self.size.y):
            row = '\n'
            for x in range(self.size.x):
                if show_rooms and self.rids[x][y] > 0:
                    row += str((self.rids[x][y] - 1) % 10)
                else:
                    tile = self.tiles[x][y]
                    row += chars[tile] if tile < len(chars) else '?'
            result += row
        return result

class RoomAndCorridorMap:
    def __init__(self, size: Point, verbose: bool = False):
        self.size = size
        while not self._try_build_map(verbose):
            pass

    def _try_build_map(self, verbose: bool) -> bool:
        level = Level(self.size)
        rects = []

        min_size, max_size = 6, 8
        separation = 3
        tries = self.size.x * self.size.y // (min_size * min_size)
        tries_left = tries

        while tries_left > 0:
            size = Point(random.randint(min_size, max_size),
                        random.randint(min_size, max_size))
            rect = Rect(
                size,
                Point(random.randint(1, self.size.x - size.x - 1),
                      random.randint(1, self.size.y - size.y - 1)))
            if not level.place_rectangular_room(rect, separation, rects):
                tries_left -= 1

        n = len(rects)
        assert n > 0
        if verbose:
            print(f"Placed {n} rectangular rooms after {tries} attempts.")

        islandness = random.randint(0, 2)
        for _ in range(3):
            level.erode(islandness)

        self.rooms = level.extract_final_rooms(n)
        level.add_walls()
        self.starting_square = self.rooms[0].get_random_square()

        for i in range(n-1):
            if not level.dig_corridor(self.rooms, i, i+1, 1.0):
                return False

        if verbose:
            print("Final map:", level.to_debug_string())

        return True

if __name__ == "__main__":
    size = Point(50, 50)
    dungeon = RoomAndCorridorMap(size, verbose=True)
