use std::cmp::Ordering;

use crate::base::{Matrix, Point};

//////////////////////////////////////////////////////////////////////////////

// Constants

// Partial transparency parameters. Selected so that we see a roughly-circular
// area if standing in a field of tall grass:
//   - Loss of 100 -> circle of radius 1
//   - Loss of 75  -> circle of radius 2
//   - Loss of 45  -> circle of radius 3
//   - Loss of 30  -> circle of radius 4
//   - Loss of 24  -> circle of radius 5
//   - Loss of 19  -> circle of radius 6
//   - Loss of 15  -> circle of radius 7
const INITIAL_VISIBILITY: i32 = 100;
const VISIBILITY_LOSSES: [i32; 7] = [100, 75, 45, 30, 24, 19, 15];

type Transform = [[i32; 2]; 2];

const TRANSFORMS: [Transform; 4] = [
    [[ 1,  0], [ 0,  1]],
    [[ 0,  1], [-1,  0]],
    [[-1,  0], [ 0, -1]],
    [[ 0, -1], [ 1,  0]],
];

const ROT_LEFT_: Transform = [[33, 56], [-56, 33]];
const ROT_RIGHT: Transform = [[33, -56], [56, 33]];

//////////////////////////////////////////////////////////////////////////////

// Rational slopes

// Invariant (enforced by new): den > 0
#[derive(Copy, Clone, Debug)]
struct Slope { num: i32, den: i32 }

impl Slope {
    fn new(num: i32, den: i32) -> Self {
        debug_assert!(den > 0);
        Self { num, den }
    }
}

impl Eq for Slope {}

impl Ord for Slope {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b < c/d  <=>  ad < bc  (valid since b, d > 0)
        (self.num * other.den).cmp(&(other.num * self.den))
    }
}

impl PartialOrd for Slope {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Slope {
    fn eq(&self, other: &Self) -> bool {
        // a/b == c/d <=> ad == bc (valid since b, d != 0)
        self.num * other.den == other.num * self.den
    }
}

//////////////////////////////////////////////////////////////////////////////

// State tracking

#[derive(Clone, Copy, Debug)]
struct SlopeRange {
    min: Slope,
    max: Slope,
    transform: &'static Transform,
    visibility: i32,
}

#[derive(Debug, Default)]
struct SlopeRanges {
    depth: i32,
    items: Vec<SlopeRange>,
}

//////////////////////////////////////////////////////////////////////////////

// Public API

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum TileVision { Full, Partial, Blocked }

pub struct Vision {
    dropoff: i32,
    radius: i32,
    offset: Point,
    points_seen: Vec<Point>,
    visibility: Matrix<i32>,

    // Allocations used in compute
    prev: SlopeRanges,
    next: SlopeRanges,
}

impl Vision {
    pub fn new(radius: i32, radius_obscured: i32) -> Self {
        let side = 2 * radius + 1;
        let size = Point(side, side);
        let dropoff = VISIBILITY_LOSSES[(radius_obscured - 1) as usize];
        Self {
            dropoff,
            radius,
            offset: Point::default(),
            points_seen: vec![],
            visibility: Matrix::new(size, -1),
            prev: SlopeRanges::default(),
            next: SlopeRanges::default(),
        }
    }

    pub fn get_points_seen(&self) -> &[Point] {
        &self.points_seen
    }

    pub fn get_visibility_at(&self, p: Point) -> i32 {
        self.visibility.get(p + self.offset)
    }

    pub fn clear(&mut self, pos: Point) {
        let center = Point(self.radius, self.radius);
        self.offset = center - pos;
        self.visibility.fill(-1);
        self.points_seen.clear();

        self.visibility.set(center, INITIAL_VISIBILITY);
        self.points_seen.push(pos);

        self.prev.depth = 1;
        self.next.depth = 2;
        self.prev.items.clear();
        self.next.items.clear();
    }

    pub fn compute<F: Fn(Point) -> TileVision>(&mut self, pos: Point, dir: Point, f: F) {
        self.clear(pos);

        let visibility = INITIAL_VISIBILITY;

        if dir == Point::default() {
            for transform in &TRANSFORMS {
                let (min, max) = (Slope::new(-1, 1), Slope::new(1, 1));
                self.prev.items.push(SlopeRange { min, max, transform, visibility });
            }
        } else {
            for transform in &TRANSFORMS {
                // Use the inverse to map dir into the right 90-degree quadrant.
                let [[a00, a01], [a10, a11]] = *transform;
                let [[b00, b01], [b10, b11]] = [[a00, -a01], [-a10, a11]];
                let (x, y) = (dir.0 * b00 + dir.1 * b10, dir.0 * b01 + dir.1 * b11);
                debug_assert!(x != 0 || y != 0);

                let [[l00, l01], [l10, l11]] = ROT_LEFT_;
                let [[r00, r01], [r10, r11]] = ROT_RIGHT;
                let (lx, ly) = (x * l00 + y * l10, x * l01 + y * l11);
                let (rx, ry) = (x * r00 + y * r10, x * r01 + y * r11);

                // Casework to figure out how the dir constrains slope ranges.
                // Here, we rely on the fact that the window is <= 180 degrees.
                let (mut min, mut max) = (Slope::new(-1, 1), Slope::new(1, 1));
                if x < 0 {
                    if y == 0 { continue; }
                    if y > 0 {
                        if rx <= 0 { continue; }
                        min = std::cmp::max(min, Slope::new(ry, rx));
                    } else {
                        if lx <= 0 { continue; }
                        max = std::cmp::min(max, Slope::new(ly, lx));
                    }
                } else {
                    if lx > 0 { max = std::cmp::min(max, Slope::new(ly, lx)); }
                    if rx > 0 { min = std::cmp::max(min, Slope::new(ry, rx)); }
                }

                if max <= min { continue; }
                self.prev.items.push(SlopeRange { min, max, transform, visibility });
            }
        }

        self.execute(pos, f);
    }

    fn execute<F: Fn(Point) -> TileVision>(&mut self, pos: Point, f: F) {
        let radius = self.radius;
        let center = Point(radius, radius);
        let r2 = radius * radius + radius;

        let push = |next: &mut SlopeRanges, s: SlopeRange| {
            if let Some(x) = next.items.last_mut() &&
                x.max == s.min && x.visibility == s.visibility &&
                x.transform.as_ptr() == s.transform.as_ptr() {
                    x.max = s.max;
            } else {
                next.items.push(s);
            }
        };

        while self.prev.depth <= radius && !self.prev.items.is_empty() {
            let depth = self.prev.depth;

            for range in &self.prev.items {
                let mut prev_visibility = -1;
                let [[a00, a01], [a10, a11]] = range.transform;
                let SlopeRange { mut min, max, transform, visibility } = *range;
                let start = (2 * min.num * depth + min.den).div_floor(2 * min.den);
                let limit = (2 * max.num * depth - max.den).div_ceil(2 * max.den);

                for width in start..=limit {
                    let (x, y) = (depth, width);
                    let nearby = x * x + y * y <= r2;
                    let point = Point(x * a00 + y * a10, x * a01 + y * a11);

                    let next_visibility = (|| {
                        if !nearby { return -1; }
                        let vision = f(point + pos);
                        if vision == TileVision::Full { return visibility; }
                        if vision == TileVision::Blocked { return 0; }
                        let r = 1.0 + (0.5 * y.abs() as f64) / (x as f64);
                        std::cmp::max(visibility - (r * self.dropoff as f64) as i32, 0)
                    })();

                    if next_visibility >= 0 {
                        let entry = self.visibility.entry_mut(point + center).unwrap();
                        if *entry < 0 { self.points_seen.push(point + pos); }
                        *entry = std::cmp::max(*entry, next_visibility);
                    }

                    if prev_visibility != next_visibility && prev_visibility >= 0 {
                        let slope = Slope::new(2 * width - 1, 2 * depth);
                        if prev_visibility > 0 {
                            let (max, visibility) = (slope, prev_visibility);
                            let range = SlopeRange { min, max, transform, visibility };
                            push(&mut self.next, range);
                        }
                        min = slope;
                    }
                    prev_visibility = next_visibility;
                }

                if prev_visibility > 0 {
                    let visibility = prev_visibility;
                    let range = SlopeRange { min, max, transform, visibility };
                    push(&mut self.next, range);
                }
            }

            std::mem::swap(&mut self.prev, &mut self.next);
            self.next.items.clear();
            self.next.depth += 2;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{Rng, SeedableRng};

    extern crate test;

    const DEBUG: bool = false;

    fn fov(eye: Point, dir: Point, map: &Matrix<char>, radius: i32) -> Matrix<bool> {
        // Wrapper around vision to make it easier to test.
        let mut vision = Vision::new(radius, 3);
        vision.compute(eye, dir, |p: Point| {
            let tile = if map.contains(p) { map.get(p) } else { '#' };
            if tile == '#' { return TileVision::Blocked; }
            if tile == ',' { return TileVision::Partial; }
            TileVision::Full
        });
        let mut result = Matrix::new(map.size, false);
        for y in 0..map.size.1 {
            for x in 0..map.size.0 {
                let p = Point(x, y);
                result.set(p, vision.get_visibility_at(p) >= 0);
            }
        }
        result
    }

    fn test_fov(input: &[&str], expected: &[&str]) {
        // Convert the input grid into a map.
        let height = input.len();
        let width = input[0].len();
        let mut map = Matrix::new(Point(width as i32, height as i32), '#');
        let mut eye = None;
        let mut target = None;

        for (y, row) in input.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                let point = Point(x as i32, y as i32);
                map.set(point, c);

                if c == '@' {
                    assert!(eye.is_none());
                    eye = Some(point);
                } else if c == 'X' {
                    assert!(target.is_none());
                    target = Some(point);
                }
            }
        }

        // Get the FOV result and compare it to the expected value.
        let eye = eye.unwrap();
        let dir = if let Some(x) = target { x - eye } else { Point::default() };
        let visible = fov(eye, dir, &map, map.size.0 + map.size.1);
        let result = show_fov(eye, &map, &visible);
        if expected != result {
            panic!("\nExpected:\n&{:#?}\n\nGot:\n&{:#?}", expected, result);
        }
    }

    fn show_fov(eye: Point, map: &Matrix<char>, visible: &Matrix<bool>) -> Vec<String> {
        let mut result = Vec::new();
        for y in 0..map.size.1 {
            let mut row = String::new();
            for x in 0..map.size.0 {
                let p = Point(x as i32, y as i32);
                let (is_eye, is_visible) = (p == eye, visible.get(p));
                let c = if is_eye { '@' } else if !is_visible { '%' } else { map.get(p) };
                row.push(c);
            }
            result.push(row);
        }
        result
    }

    #[test]
    fn test_empty() {
        test_fov(&[
            "@...",
            "....",
            "....",
        ], &[
            "@...",
            "....",
            "....",
        ]);
    }

    #[test]
    fn test_single_pillar() {
        test_fov(&[
            "@...",
            ".#..",
            "....",
        ], &[
            "@...",
            ".#..",
            "..%%",
        ]);
    }

    #[test]
    fn test_wall_with_gap() {
        test_fov(&[
            "@....",
            ".....",
            "..#..",
            ".....",
            "..#..",
        ], &[
            "@....",
            ".....",
            "..#..",
            "...%.",
            "..#.%",
        ]);
    }

    #[test]
    fn test_diagonal_wall() {
        test_fov(&[
            "@....",
            ".....",
            "..#..",
            "...#.",
            "....#",
        ], &[
            "@....",
            ".....",
            "..#..",
            "...%.",
            "....%",
        ]);
    }

    #[test]
    fn test_near_45() {
        test_fov(&[
            "@...",
            "....",
            "..#.",
            "....",
        ], &[
            "@...",
            "....",
            "..#.",
            "...%",
        ]);
    }

    #[test]
    fn test_gaps() {
        test_fov(&[
            "..........#",
            "..........#",
            "..........#",
            "......#...#",
            "..##..#...#",
            "..........#",
            "...@......#",
            "......#...#",
            "##....#...#",
            "..........#",
            "####..##..#",
        ], &[
            "%%%%%....%%",
            "%%%%....%%%",
            ".%%%...%%%%",
            "..%%..#%%.#",
            "..##..#...#",
            "..........#",
            "...@......#",
            "......#...#",
            "##....#%%%%",
            "%.......%%%",
            "####..##.%%",
        ]);
    }

    #[test]
    fn test_near_wall() {
        test_fov(&[
            "...............",
            ".#############.",
            ".#@..........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#############.",
            "...............",
        ], &[
            "%%%%%%%%%%%%%%%",
            "%#############%",
            "%#@..........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#...........#%",
            "%#############%",
            "%%%%%%%%%%%%%%%",
        ]);
    }


    #[test]
    fn test_large() {
        test_fov(&[
            "...............",
            ".#############.",
            ".#...........#.",
            ".#...........#.",
            ".#.......#...#.",
            ".#...........#.",
            ".#..#........#.",
            ".#.....@.....#.",
            ".#...........#.",
            ".#...........#.",
            ".#.......#...#.",
            ".#....#......#.",
            ".#...........#.",
            ".#...........#.",
            ".#...........#.",
            ".#############.",
            "...............",
        ], &[
            "%%%%%%%%%%%%%%%",
            "%##########%##%",
            "%#........%..#%",
            "%#...........#%",
            "%#.......#...#%",
            "%%%..........#%",
            "%#..#........#%",
            "%#.....@.....#%",
            "%#...........#%",
            "%#...........#%",
            "%#.......#...#%",
            "%#....#......#%",
            "%#........%..#%",
            "%#.........%.#%",
            "%#...%.....%%#%",
            "%####%######%%%",
            "%%%%%%%%%%%%%%%",
        ]);
    }

    #[test]
    fn test_field_of_grass() {
        test_fov(&[
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,@,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,",
        ], &[
            "%%%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%%%",
            "%%%%%%,,,%%%%%%",
            "%%%%%,,,,,%%%%%",
            "%%%%,,,,,,,%%%%",
            "%%%%,,,@,,,%%%%",
            "%%%%,,,,,,,%%%%",
            "%%%%%,,,,,%%%%%",
            "%%%%%%,,,%%%%%%",
            "%%%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%%%",
        ]);
    }

    #[test]
    fn test_semitransparent_walls() {
        test_fov(&[
            ".....,...,.,.,.",
            ".....,.@.,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
            ".....,...,.,.,.",
        ], &[
            ".....,...,.,.,%",
            ".....,.@.,.,.,%",
            ".....,...,.,.,%",
            ".....,...,.,%%%",
            ".....,...,.,%%%",
            ".....,...,.,%%%",
            ".....,...,.,%%%",
            "....%,...,%,%%%",
            "....%,...,%%%%%",
            "...%%,...,%%%%%",
            "..%%%,...,%%%%%",
            "..%%%,...,%%%%%",
            ".%%%%,...,%%%%%",
            "%%%%%,...,%%%%%",
            "%%%%%,...,%%%%%",
            "%%%%%,...,%%%%%",
            "%%%%%,...,%%%%%",
        ]);
    }

    #[test]
    fn test_directional_s() {
        test_fov(&[
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            "......@......",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            "......X......",
        ], &[
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%@%%%%%%",
            "%%%%.....%%%%",
            "%%.........%%",
            "%...........%",
            ".............",
            ".............",
            "......X......",
        ]);
    }

    #[test]
    fn test_directional_sw() {
        test_fov(&[
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            "......@......",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            "X............",
        ], &[
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            ".%%%%%%%%%%%%",
            ".....%%%%%%%%",
            "......@%%%%%%",
            ".......%%%%%%",
            "........%%%%%",
            "........%%%%%",
            "........%%%%%",
            "........%%%%%",
            "X........%%%%",
        ]);
    }

    #[test]
    fn test_directional_ssw() {
        test_fov(&[
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            "......@......",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            "...X.........",
        ], &[
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "%%%%%%%%%%%%%",
            "......@%%%%%%",
            "........%%%%%",
            "........%%%%%",
            ".........%%%%",
            "..........%%%",
            "..........%%%",
            "...X.......%%",
        ]);
    }


    #[bench]
    fn bench_fov_shadowcast(b: &mut test::Bencher) {
        let (eye, map) = generate_fov_input();
        b.iter(|| {
            let visible = fov(eye, Point::default(), &map, eye.0);
            debug_fov_output(eye, &map, &visible);
        });
    }

    #[bench]
    fn bench_fov_pc_vision(b: &mut test::Bencher) {
        run_vision_benchmark(b, true);
    }

    #[bench]
    fn bench_fov_npc_vision(b: &mut test::Bencher) {
        run_vision_benchmark(b, false);
    }

    fn run_vision_benchmark(b: &mut test::Bencher, player: bool) {
        let (eye, map) = generate_fov_input();
        let mapping: crate::base::HashMap<_, _> =
                [('.', '.'), ('#', '#'), (',', '"')].into_iter().collect();
        let mut tiles = Matrix::new(map.size, crate::game::Tile::get('#'));
        for x in 0..map.size.0 {
            for y in 0..map.size.1 {
                let p = Point(x, y);
                let c = *mapping.get(&map.get(p)).unwrap();
                tiles.set(p, crate::game::Tile::get(c));
            }
        }
        let lookup = |p: Point| {
            let tile = tiles.get(p);
            if tile.blocks_vision() { return TileVision::Blocked; }
            if tile.limits_vision() { return TileVision::Partial; }
            TileVision::Full
        };
        let mut vision = Vision::new(eye.0, 3);
        let dir = if player { Point::default() } else { Point(1, 1) };
        b.iter(|| { vision.compute(eye, dir, lookup) });
    }

    fn debug_fov_output(eye: Point, map: &Matrix<char>, visible: &Matrix<bool>) {
        if !DEBUG { return; }
        let count = visible.data.iter().filter(|&&x| x).count();
        println!("Visibility trie: {} cells visible!", count);
        for line in show_fov(eye, &map, &visible) { println!("{}", line); }
    }

    fn generate_fov_input() -> (Point, Matrix<char>) {
        let radius = 21;
        let side = 2 * radius + 1;
        let size = Point(side, side);
        let eye = Point(radius, radius);

        let mut rng = crate::base::RNG::seed_from_u64(17);
        let mut map = Matrix::new(size, '#');
        for x in 0..size.0 {
            for y in 0..size.1 {
                let sample = rng.gen_range(0..100);
                let c = if sample < 1 { '#' } else if sample < 5 { ',' } else { '.' };
                map.set(Point(x, y), c);
            }
        }
        (eye, map)
    }
}
