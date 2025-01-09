from dataclasses import dataclass
import random
from enum import Enum, auto
from typing import List, Set, Tuple, Dict

@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

class RoomType(Enum):
    CLEARING = auto()
    THICKET = auto()
    DENSE_FOREST = auto()
    LAKE = auto()

@dataclass
class Room:
    center: Point
    width: int
    height: int
    room_type: RoomType
    connections: Set[int]  # indices of connected rooms

    def contains(self, p: Point) -> bool:
        dx = abs(p.x - self.center.x)
        dy = abs(p.y - self.center.y)
        return dx <= self.width//2 and dy <= self.height//2

class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cells = [['.' for _ in range(width)] for _ in range(height)]
        self.rooms: List[Room] = []

    def get(self, p: Point) -> str:
        if 0 <= p.x < self.width and 0 <= p.y < self.height:
            return self.cells[p.y][p.x]
        return '#'

    def set(self, p: Point, tile: str):
        if 0 <= p.x < self.width and 0 <= p.y < self.height:
            self.cells[p.y][p.x] = tile

    def print(self):
        for row in self.cells:
            chars = [(ord(c) - 0x20 + 0xFF00) for c in row]
            print(''.join(chr(c) for c in chars))

def try_place_room(map: Map, min_size: int, max_size: int, rng: random.Random) -> bool:
    """Try to place a room without overlap"""
    width = rng.randint(min_size, max_size)
    height = rng.randint(min_size, max_size)

    # Try 50 times to place the room
    for _ in range(50):
        x = rng.randint(width//2, map.width - width//2 - 1)
        y = rng.randint(height//2, map.height - height//2 - 1)
        center = Point(x, y)

        # Check if room would overlap with existing rooms
        overlaps = False
        new_room = Room(center, width, height, RoomType.CLEARING, set())
        for room in map.rooms:
            # Add buffer space between rooms
            expanded = Room(room.center, room.width + 2, room.height + 2, room.room_type, set())

            # Check corners of new room against expanded existing room
            for dx in [-width//2, width//2]:
                for dy in [-height//2, height//2]:
                    p = Point(center.x + dx, center.y + dy)
                    if expanded.contains(p):
                        overlaps = True
                        break
                if overlaps:
                    break
            if overlaps:
                break

        if not overlaps:
            map.rooms.append(new_room)
            return True

    return False

def connect_rooms(map: Map) -> None:
    """Connect rooms using a simple MST + extra connections for loops"""
    def dist(r1: Room, r2: Room) -> float:
        dx = r1.center.x - r2.center.x
        dy = r1.center.y - r2.center.y
        return (dx * dx + dy * dy) ** 0.5

    # First create MST
    if not map.rooms:
        return

    connected = {0}
    while len(connected) < len(map.rooms):
        best_dist = float('inf')
        best_pair = None

        for i in connected:
            for j in range(len(map.rooms)):
                if j in connected:
                    continue
                d = dist(map.rooms[i], map.rooms[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        if best_pair:
            i, j = best_pair
            map.rooms[i].connections.add(j)
            map.rooms[j].connections.add(i)
            connected.add(j)

    # Add some extra connections for loops
    num_extra = len(map.rooms) // 3
    for _ in range(num_extra):
        i = random.randint(0, len(map.rooms)-1)
        candidates = []
        for j in range(len(map.rooms)):
            if i != j and j not in map.rooms[i].connections:
                candidates.append((dist(map.rooms[i], map.rooms[j]), j))
        if candidates:
            candidates.sort()
            # Take one of the closest unconnected rooms
            j = candidates[0][1]
            map.rooms[i].connections.add(j)
            map.rooms[j].connections.add(i)

def assign_room_types(map: Map, rng: random.Random) -> None:
    """Assign room types based on position and neighbors"""
    # First place a lake
    lake_idx = rng.randrange(len(map.rooms))
    map.rooms[lake_idx].room_type = RoomType.LAKE

    # Assign other rooms based on distance to water
    for i, room in enumerate(map.rooms):
        if i == lake_idx:
            continue

        dist_to_water = dist_between_rooms(room, map.rooms[lake_idx])

        # Clearings more likely near water
        if dist_to_water < 15 and rng.random() < 0.6:
            room.room_type = RoomType.CLEARING
        # Thickets more likely far from water
        elif dist_to_water > 25 and rng.random() < 0.6:
            room.room_type = RoomType.THICKET
        else:
            room.room_type = RoomType.DENSE_FOREST

def dist_between_rooms(r1: Room, r2: Room) -> float:
    dx = r1.center.x - r2.center.x
    dy = r1.center.y - r2.center.y
    return (dx * dx + dy * dy) ** 0.5

def visualize_rooms(map: Map) -> None:
    """Draw rooms and connections for visualization"""
    # Clear map
    map.cells = [['.' for _ in range(map.width)] for _ in range(map.height)]

    # Draw connections first
    for i, room in enumerate(map.rooms):
        for j in room.connections:
            other = map.rooms[j]
            # Draw simple line between centers
            x1, y1 = room.center.x, room.center.y
            x2, y2 = other.center.x, other.center.y

            # Bresenham's line algorithm
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            x, y = x1, y1
            while True:
                map.set(Point(x, y), '+')
                if x == x2 and y == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

    # Draw rooms
    for room in map.rooms:
        # Different character for each room type
        char = {
            RoomType.CLEARING: 'C',
            RoomType.THICKET: 'T',
            RoomType.DENSE_FOREST: 'F',
            RoomType.LAKE: 'L'
        }[room.room_type]

        for dx in range(-room.width//2, room.width//2 + 1):
            for dy in range(-room.height//2, room.height//2 + 1):
                p = Point(room.center.x + dx, room.center.y + dy)
                if dx in [-room.width//2, room.width//2] or dy in [-room.height//2, room.height//2]:
                    map.set(p, '#')
                else:
                    map.set(p, char)

def generate_room_layout(width: int, height: int, seed: int = None) -> Map:
    rng = random.Random(seed)
    map = Map(width, height)

    # Place rooms
    num_rooms = rng.randint(8, 12)
    for _ in range(num_rooms):
        try_place_room(map, 5, 10, rng)

    connect_rooms(map)
    assign_room_types(map, rng)
    visualize_rooms(map)

    return map

if __name__ == "__main__":
    m = generate_room_layout(50, 50, seed=42)
    m.print()
