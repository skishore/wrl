use std::cmp::Ordering;

use crate::base::{Matrix, Point};

//////////////////////////////////////////////////////////////////////////////

// Can be replaced by (unstable) feature int_roundings: div_floor / div_ceil

fn div_floor(lhs: i32, rhs: i32) -> i32 {
    // Algorithm from [Daan Leijen. _Division and Modulus for Computer Scientists_,
    // December 2001](http://research.microsoft.com/pubs/151917/divmodnote-letter.pdf)
    let (d, r) = (lhs / rhs, lhs % rhs);
    if (r > 0 && rhs < 0) || (r < 0 && rhs > 0) { d - 1 } else { d }
}

fn div_ceil(lhs: i32, rhs: i32) -> i32 {
    let (d, r) = (lhs / rhs, lhs % rhs);
    if (r > 0 && rhs > 0) || (r < 0 && rhs < 0) { d + 1 } else { d }
}

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
pub const INITIAL_VISIBILITY: i32 = 100;
pub const VISIBILITY_LOSSES: [i32; 7] = [100, 75, 45, 30, 24, 19, 15];

#[derive(Clone, Copy, Debug)]
struct Transform([[i32; 2]; 2]);

const TRANSFORMS: [Transform; 4] = [
    Transform([[ 1,  0], [ 0,  1]]),
    Transform([[ 0,  1], [-1,  0]]),
    Transform([[-1,  0], [ 0, -1]]),
    Transform([[ 0, -1], [ 1,  0]]),
];

const ROT_LEFT_: Transform = Transform([[33, 56], [-56, 33]]);
const ROT_RIGHT: Transform = Transform([[33, -56], [56, 33]]);

impl std::ops::Mul<Point> for Transform {
    type Output = Point;
    fn mul(self, rhs: Point) -> Self::Output {
        let Transform([[a00, a01], [a10, a11]]) = self;
        Point(rhs.0 * a00 + rhs.1 * a10, rhs.0 * a01 + rhs.1 * a11)
    }
}

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
        (self.num as i64 * other.den as i64).cmp(&(other.num as i64 * self.den as i64))
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
        self.num as i64 * other.den as i64 == other.num as i64 * self.den as i64
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

pub struct VisionArgs<F: Fn(Point) -> i32> {
    pub pos: Point,
    pub dir: Point,
    pub opacity_lookup: F,
}

pub struct Vision {
    radius: i32,
    r2_limit: i64,
    initial_visibility: i32,
    offset: Point,
    points_seen: Vec<Point>,
    visibility: Matrix<i32>,

    // Allocations used in compute
    prev: SlopeRanges,
    next: SlopeRanges,
}

impl Vision {
    pub fn new(radius: i32) -> Self {
        Self::new_with_visibility(radius, INITIAL_VISIBILITY)
    }

    pub fn new_with_visibility(radius: i32, initial_visibility: i32) -> Self {
        let side = 2 * radius + 1;
        let size = Point(side, side);
        let r = radius as i64;
        Self {
            radius,
            r2_limit: r * r + r,
            initial_visibility,
            offset: Point::default(),
            points_seen: vec![],
            visibility: Matrix::new(size, -1),
            prev: SlopeRanges::default(),
            next: SlopeRanges::default(),
        }
    }

    pub fn can_see(&self, p: Point) -> bool {
        self.get_visibility_at(p) >= 0
    }

    pub fn get_points_seen(&self) -> &[Point] {
        &self.points_seen
    }

    pub fn get_visibility_at(&self, p: Point) -> i32 {
        self.visibility.get(p + self.offset)
    }

    pub fn clear(&mut self, pos: Point) {
        // Sparse clear optimization. The dense clear has much better constant
        // factors so we only switch over when it's sufficiently sparse.
        if self.visibility.data.len() < 16 * self.points_seen.len() {
            self.visibility.fill(-1);
        } else {
            for &point in &self.points_seen {
                debug_assert!(self.visibility.get(point + self.offset) >= 0);
                self.visibility.set(point + self.offset, -1);
            }
            debug_assert!(self.visibility.data.iter().all(|&x| x == -1));
        }

        let radius = self.radius;
        let center = Point(radius, radius);
        let visibility = self.initial_visibility;

        self.offset = center - pos;
        self.points_seen.clear();

        self.visibility.set(center, visibility);
        self.points_seen.push(pos);

        self.prev.depth = 1;
        self.next.depth = 2;
        self.prev.items.clear();
        self.next.items.clear();
    }

    pub fn check_point<F: Fn(Point) -> i32>(
            &mut self, args: &VisionArgs<F>, target: Point) -> bool {
        if args.pos == target { return true; }

        let delta = target - args.pos;
        if delta.len_l2_squared() > self.r2_limit { return false; }

        let Point(x, y) = delta;
        let limit = std::cmp::max(x.abs(), y.abs());

        self.clear(args.pos);
        self.seed_ranges(args.dir, Some(target - args.pos));
        self.execute(args.pos, limit, &args.opacity_lookup);
        self.can_see(target)
    }

    pub fn compute<F: Fn(Point) -> i32>(&mut self, args: &VisionArgs<F>) {
        self.clear(args.pos);
        self.seed_ranges(args.dir, None);
        self.execute(args.pos, self.radius, &args.opacity_lookup);
    }

    fn seed_ranges(&mut self, dir: Point, target: Option<Point>) {
        let visibility = self.initial_visibility;

        if dir == Point::default() {
            for transform in &TRANSFORMS {
                let (mut min, mut max) = (Slope::new(-1, 1), Slope::new(1, 1));

                // Skip this quadrant if the target outside it; else, filter.
                if let Some(target) = target {
                    let Transform([[a00, a01], [a10, a11]]) = *transform;
                    let inverse = Transform([[a00, -a01], [-a10, a11]]);

                    let Point(x, y) = inverse * target;
                    if x == 0 || x < y.abs() { continue; }
                    min = std::cmp::max(min, Slope::new(2 * y - 1, 2 * x));
                    max = std::cmp::min(max, Slope::new(2 * y + 1, 2 * x));
                }

                // If the range is still non-empty, scan it.
                if max <= min { continue; }
                self.prev.items.push(SlopeRange { min, max, transform, visibility });
            }
        } else {
            for transform in &TRANSFORMS {
                // Use the inverse to map dir into the right 90-degree quadrant.
                let Transform([[a00, a01], [a10, a11]]) = *transform;
                let inverse = Transform([[a00, -a01], [-a10, a11]]);
                let Point(x, y) = inverse * dir;
                let Point(lx, ly) = ROT_LEFT_ * Point(x, y);
                let Point(rx, ry) = ROT_RIGHT * Point(x, y);
                debug_assert!(x != 0 || y != 0);

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

                // Skip this quadrant if the target outside it; else, filter.
                if let Some(target) = target {
                    let Point(x, y) = inverse * target;
                    if x == 0 || x < y.abs() { continue; }
                    min = std::cmp::max(min, Slope::new(2 * y - 1, 2 * x));
                    max = std::cmp::min(max, Slope::new(2 * y + 1, 2 * x));
                }

                // If the range is still non-empty, scan it.
                if max <= min { continue; }
                self.prev.items.push(SlopeRange { min, max, transform, visibility });
            }
        }
    }

    fn execute<F: Fn(Point) -> i32>(
            &mut self, pos: Point, limit: i32, opacity_lookup: F) {
        let radius = self.radius;
        let center = Point(radius, radius);
        let r2_limit = self.r2_limit;

        let push = |next: &mut SlopeRanges, s: SlopeRange| {
            if let Some(x) = next.items.last_mut() &&
                x.max == s.min && x.visibility == s.visibility &&
                x.transform as *const Transform == s.transform as *const Transform {
                    x.max = s.max;
            } else {
                next.items.push(s);
            }
        };

        while self.prev.depth <= limit && !self.prev.items.is_empty() {
            let depth = self.prev.depth;

            for range in &self.prev.items {
                let mut prev_visibility = -1;
                let SlopeRange { mut min, max, transform, visibility } = *range;
                let start = div_floor(2 * min.num * depth + min.den, 2 * min.den);
                let limit = div_ceil(2 * max.num * depth - max.den, 2 * max.den);

                for width in start..=limit {
                    let source = Point(depth, width);
                    let nearby = source.len_l2_squared() <= r2_limit;
                    let point = *transform * source;

                    let next_visibility = (|| {
                        if !nearby { return -1; }
                        let opacity = opacity_lookup(point + pos);
                        if opacity == 0 { return visibility; }
                        if opacity >= visibility { return 0; }
                        let r = 1.0 + (0.5 * source.1.abs() as f64) / (source.0 as f64);
                        std::cmp::max(visibility - (r * opacity as f64) as i32, 0)
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

        self.points_seen.sort_by_key(|&x| (x - pos).len_l2_squared());
    }
}

//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{Rng, SeedableRng};

    extern crate test;

    const DEBUG: bool = false;
    const VISIBILITY_LOSS: i32 = VISIBILITY_LOSSES[2];

    fn run_fov(pos: Point, dir: Point, map: &Matrix<char>,
               radius: i32, check_point_lookups: bool) -> Matrix<bool> {
        // Wrapper around Vision to make it easier to test.
        let opacity_lookup = |p: Point| {
            let tile = if map.contains(p) { map.get(p) } else { '#' };
            if tile == '#' { return INITIAL_VISIBILITY; }
            if tile == ',' { return VISIBILITY_LOSS; }
            0
        };
        let args = VisionArgs { pos, dir, opacity_lookup };

        let mut vision = Vision::new(radius);
        vision.compute(&args);

        let mut result = Matrix::new(map.size, false);
        for y in 0..map.size.1 {
            for x in 0..map.size.0 {
                let p = Point(x, y);
                result.set(p, vision.can_see(p));
            }
        }

        // Check that point lookups are consistent with the full computation.
        if check_point_lookups {
            for y in 0..map.size.1 {
                for x in 0..map.size.0 {
                    let p = Point(x, y);
                    assert!(result.get(p) == vision.check_point(&args, p));
                }
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
        let visible = run_fov(eye, dir, &map, map.size.0 + map.size.1, true);
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
            let visible = run_fov(eye, Point::default(), &map, eye.0, false);
            debug_fov_output(eye, &map, &visible);
        });
    }

    #[bench]
    fn bench_fov_360_vision(b: &mut test::Bencher) {
        run_vision_benchmark(b, /*directional=*/false, /*point_lookups=*/false);
    }

    #[bench]
    fn bench_fov_directional_vision(b: &mut test::Bencher) {
        run_vision_benchmark(b, /*directional=*/true, /*point_lookups=*/false);
    }

    #[bench]
    fn bench_fov_360_point_lookups(b: &mut test::Bencher) {
        run_vision_benchmark(b, /*directional=*/false, /*point_lookups=*/true);
    }

    #[bench]
    fn bench_fov_directional_point_lookups(b: &mut test::Bencher) {
        run_vision_benchmark(b, /*directional=*/true, /*point_lookups=*/true);
    }

    fn run_vision_benchmark(b: &mut test::Bencher, directional: bool, point_lookups: bool) {
        let (pos, map) = generate_fov_input();
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
        let opacity_lookup = |p: Point| {
            let tile = tiles.get(p);
            if tile.blocks_vision() { return INITIAL_VISIBILITY; }
            if tile.limits_vision() { return VISIBILITY_LOSS; }
            0
        };
        let dir = if directional { Point(1, 1) } else { Point::default() };
        let args = VisionArgs { pos, dir, opacity_lookup };

        let mut vision = Vision::new(pos.0);
        if point_lookups {
            let mut rng = crate::base::RNG::seed_from_u64(17);
            b.iter(|| {
                let Point(x, y) = map.size;
                let target = Point(rng.gen_range(0..x), rng.gen_range(0..y));
                vision.check_point(&args, target);
            });
        } else {
            b.iter(|| { vision.compute(&args) });
        }
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
