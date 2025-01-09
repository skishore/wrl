import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cells = [['.' for _ in range(width)] for _ in range(height)]

    def get(self, p: Point) -> str:
        if 0 <= p.x < self.width and 0 <= p.y < self.height:
            return self.cells[p.y][p.x]
        return '#'  # Walls outside map

    def set(self, p: Point, tile: str):
        if 0 <= p.x < self.width and 0 <= p.y < self.height:
            self.cells[p.y][p.x] = tile

    def print(self):
        for row in self.cells:
            # Convert to wide chars by shifting into the Unicode FF00 block
            chars = [(ord(c) - 0x20 + 0xFF00) for c in row]
            print(''.join(chr(c) for c in chars))


def in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def cellular_automata(width: int, height: int, rng: random.Random, init_chance: int) -> List[List[bool]]:
    """Port of the Rust cellular automata for initial map generation"""
    # Initialize with random walls
    cells = [[False]*width for _ in range(height)]
    for y in range(height):
        cells[y][0] = cells[y][height-1] = True
    for x in range(width):
        cells[0][x] = cells[width-1][x] = True

    # Random initialization
    for y in range(height):
        for x in range(width):
            if rng.randint(0, 99) < init_chance:
                cells[y][x] = True

    # Run automata
    for _ in range(3):
        next_cells = [row[:] for row in cells]
        for y in range(1, height-1):
            for x in range(1, width-1):
                adj1 = adj2 = 0
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        if abs(dx) == 2 and abs(dy) == 2:
                            continue

                        nx, ny = x + dx, y + dy
                        if in_bounds(nx, ny, width, height) and cells[ny][nx]:
                            if max(abs(dx), abs(dy)) <= 1:
                                adj1 += 1
                            adj2 += 1

                next_cells[y][x] = adj1 >= 5 or (_ < 2 and adj2 <= 1)
        cells = next_cells

    return cells


def generate_map(width: int, height: int, seed: int = None) -> Map:
    rng = random.Random(seed)
    m = Map(width, height)

    # Generate forests and grass with cellular automata
    walls = cellular_automata(width, height, rng, 45)
    grass = cellular_automata(width, height, rng, 45)

    for y in range(height):
        for x in range(width):
            p = Point(x, y)
            if walls[y][x]:
                m.set(p, '#')
            elif grass[y][x]:
                m.set(p, '"')

    # Add river
    river = [Point(width//2, 0)]
    for y in range(1, height):
        last_x = river[-1].x
        next_x = last_x + rng.randint(-1, 1)
        river.append(Point(next_x, y))

    # Center river by computing offset
    first_p = river[0]
    last_p = river[-1]
    target = Point(first_p.x + last_p.x, first_p.y + last_p.y)
    offset = Point((width - target.x) // 2, 0)

    # Place river
    for p in river:
        m.set(p + offset, '~')

    # Add some flowers
    for _ in range(5):
        for _ in range(100):  # Try up to 100 times to place each flower
            x = rng.randint(0, width-1)
            y = rng.randint(0, height-1)
            p = Point(x, y)
            if m.get(p) == '.':
                m.set(p, '%')
                break

    return m


if __name__ == "__main__":
    m = generate_map(50, 50, seed=42)
    m.print()
