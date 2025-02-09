use std::cmp::Ordering;

use crate::base::{Matrix, Point};

//////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug)]
struct Rational {
    num: i32,
    den: i32,  // Always positive
}

impl Rational {
    fn new(num: i32, den: i32) -> Self {
        debug_assert!(den > 0);
        Rational { num, den }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // a/b < c/d  <=>  ad < bc  (valid since b,d > 0)
        Some((self.num * other.den).cmp(&(other.num * self.den)))
    }
}

impl PartialEq for Rational {
    fn eq(&self, other: &Self) -> bool {
        self.num * other.den == other.num * self.den
    }
}

//////////////////////////////////////////////////////////////////////////////

type Transform = [[i32; 2]; 2];

const TRANSFORMS: [Transform; 8] = [
    [[ 1,  0], [ 0,  1]], [[ 0,  1], [ 1,  0]],
    [[ 0,  1], [-1,  0]], [[-1,  0], [ 0,  1]],
    [[-1,  0], [ 0, -1]], [[ 0, -1], [-1,  0]],
    [[ 0, -1], [ 1,  0]], [[ 1,  0], [ 0, -1]],
];

#[derive(Debug)]
struct SlopeRange {
    min: Rational,
    max: Rational,
    transform: &'static Transform,
}

#[derive(Debug)]
struct SlopeRanges {
    depth: i32,
    items: Vec<SlopeRange>,
}

fn fov(eye: Point, map: &Matrix<char>) -> Matrix<bool> {
    let mut result = Matrix::new(map.size, false);
    result.set(eye, true);

    let seeds = TRANSFORMS.iter().map(|x| {
        let (min, max) = (Rational::new(0, 1), Rational::new(1, 1));
        SlopeRange { min, max, transform: x }
    }).collect();
    let mut prev = SlopeRanges { depth: 1, items: seeds };
    let mut next = SlopeRanges { depth: 2, items: vec![] };

    while !prev.items.is_empty() {
        let depth = prev.depth;
        let start_depth = 2 * depth - 1;
        let limit_depth = 2 * depth + 1;

        for range in &prev.items {
            let mut prev_visible = true;
            let [[a00, a01], [a10, a11]] = range.transform;
            let SlopeRange { mut min, max, transform } = *range;
            let start = (min.den + min.num * start_depth - 0) / (min.den * 2);
            let limit = (max.den + max.num * limit_depth - 1) / (max.den * 2);

            for width in start..=limit {
                let (x, y) = (depth, width);
                let point = Point(x * a00 + y * a10, x * a01 + y * a11) + eye;
                result.set(point, true);

                let next_visible = map.get(point) == '.';
                if next_visible != prev_visible {
                    // Angle depends on if we're getting more visible or less.
                    let den = if prev_visible { limit_depth } else { start_depth };
                    let slope = Rational::new(2 * width - 1, den);
                    if prev_visible && min < slope {
                        next.items.push(SlopeRange { min, max: slope, transform });
                    }
                    min = slope;
                }

                if width == limit && next_visible && min < max {
                    next.items.push(SlopeRange { min, max, transform });
                }
                prev_visible = next_visible;
            }
        }

        std::mem::swap(&mut prev, &mut next);
        next.items.clear();
        next.depth += 2;
    }
    result
}

//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{Rng, SeedableRng};

    extern crate test;

    const DEBUG: bool = false;

    fn test_fov(input: &[&str], expected: &[&str]) {
        // Convert the input grid into a map.
        let height = input.len();
        let width = input[0].len();
        let mut map = Matrix::new(Point(width as i32, height as i32), '#');
        let mut eye = None;

        for (y, row) in input.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                let point = Point(x as i32, y as i32);
                map.set(point, c);
                if c != '@' { continue; }

                assert!(eye.is_none());
                eye = Some(point);
            }
        }

        // Get the FOV result and compare it to the expected value.
        let eye = eye.unwrap();
        let visible = fov(eye, &map);
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
            "......#...#",
            "..##..#...#",
            "..........#",
            "...@......#",
            "......#...#",
            "##....#...#",
            "..........#",
            "####..##..#",
        ], &[
            "%%%%...%%%%",
            ".%%%..#%%.#",
            "..##..#...#",
            "..........#",
            "...@......#",
            "......#...#",
            "##....#%%%%",
            "%......%%%%",
            "####..##%%%",
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
            "%##########%%#%",
            "%#........%%.#%",
            "%#........%..#%",
            "%#.......#...#%",
            "%%%..........#%",
            "%#..#........#%",
            "%#.....@.....#%",
            "%#...........#%",
            "%#...........#%",
            "%#.......#...#%",
            "%#....#...%..#%",
            "%#........%%.#%",
            "%#.........%%#%",
            "%#...%.....%%%%",
            "%####%######%%%",
            "%%%%%%%%%%%%%%%",
        ]);
    }

    #[bench]
    fn bench_fov_shadowcast(b: &mut test::Bencher) {
        let (eye, map) = generate_fov_input();
        b.iter(|| {
            let visible = fov(eye, &map);
            debug_fov_output(eye, &map, &visible);
        });
    }

    #[bench]
    fn bench_fov_visibility_trie(b: &mut test::Bencher) {
        let (eye, map) = generate_fov_input();
        let mut fov = crate::base::FOV::new(eye.0);
        b.iter(|| {
            let mut visible = Matrix::new(map.size, false);
            fov.apply(|x| {
                let p = x.next + eye;
                visible.set(p, true);
                x.next != Point::default() && map.get(p) == '#'
            });
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
        let mut tiles = Matrix::new(map.size, crate::game::Tile::get('#'));
        for x in 0..map.size.0 {
            for y in 0..map.size.1 {
                let p = Point(x, y);
                tiles.set(p, crate::game::Tile::get(map.get(p)));
            }
        }
        let mut vision = crate::knowledge::Vision::new(eye.0);
        let args = crate::knowledge::VisionArgs {
            player,
            pos: eye,
            dir: crate::base::dirs::S,
        };

        b.iter(|| {
            vision.compute(&args, |x| tiles.get(x));
            //debug_fov_output(eye, &map, &visible);
        });
    }

    fn debug_fov_output(eye: Point, map: &Matrix<char>, visible: &Matrix<bool>) {
        if !DEBUG { return; }
        let count = visible.data.iter().filter(|&&x| x).count();
        println!("Visibility trie: {} cells visible!", count);
        for line in show_fov(eye, &map, &visible) { println!("{}", line); }
    }

    fn generate_fov_input() -> (Point, Matrix<char>) {
        let radius = 40;
        let side = 2 * radius + 1;
        let size = Point(side, side);
        let eye = Point(radius, radius);

        let mut rng = crate::base::RNG::seed_from_u64(17);
        let mut map = Matrix::new(size, '#');
        for x in 0..size.0 {
            for y in 0..size.1 {
                let blocked = rng.gen_range(0..100) < 1;
                map.set(Point(x, y), if blocked { '#' } else { '.' });
            }
        }
        (eye, map)
    }
}
