use crate::base::{Matrix, Point};
use crate::shadowcast::{INITIAL_VISIBILITY, Vision, VisionArgs};

//////////////////////////////////////////////////////////////////////////////

const MAX_LIGHT_RADIUS: i32 = 12;
const MAX_LIGHT_DIAMETER: i32 = 2 * MAX_LIGHT_RADIUS + 1;
const N: usize = ((MAX_LIGHT_DIAMETER * MAX_LIGHT_DIAMETER + 63) / 64) as usize;

#[derive(Clone, Copy, Default)]
struct LightSourceBitset { words: [u64; N], }

struct OneBits(u64);

impl Iterator for OneBits {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 { return None; }
        let index = self.0.trailing_zeros();
        self.0 ^= 1 << index;
        Some(index)
    }
}

impl LightSourceBitset {
    fn sources(&self) -> impl Iterator<Item = Point> {
        self.words.iter().enumerate().flat_map(|(a, &x)| OneBits(x).map(move |b| {
            let i = (a * 64 + b as usize) as i32;
            let x = i % MAX_LIGHT_DIAMETER;
            let y = i / MAX_LIGHT_DIAMETER;
            Point(x - MAX_LIGHT_RADIUS, y - MAX_LIGHT_RADIUS)
        }))
    }

    fn toggle(&mut self, delta: Point) {
        let x = delta.0 + MAX_LIGHT_RADIUS;
        let y = delta.1 + MAX_LIGHT_RADIUS;
        let i = (x + y * MAX_LIGHT_DIAMETER) as usize;
        self.words[i / 64] ^= 1 << (i & 63);
    }
}

//////////////////////////////////////////////////////////////////////////////

struct Lighting {
    light_radius: Matrix<i32>,
    light_values: Matrix<i32>,
    opacity: Matrix<i32>,
    sources: Matrix<LightSourceBitset>,
    visions: Vec<Vision>,
}

impl Lighting {
    pub fn new(size: Point) -> Self {
        let mut result = Self {
            light_radius: Matrix::new(size, -1),
            light_values: Matrix::new(size, 0),
            opacity: Matrix::new(size, INITIAL_VISIBILITY),
            sources: Matrix::new(size, Default::default()),
            visions: (0..=MAX_LIGHT_RADIUS).map(|x| Vision::new(x)).collect(),
        };
        result.opacity.fill(0);
        result
    }

    pub fn get_light(&self, point: Point) -> i32 {
        self.light_values.get(point)
    }

    pub fn set_light(&mut self, point: Point, light: i32) {
        let light = std::cmp::min(light, MAX_LIGHT_RADIUS);
        let Some(entry) = self.light_radius.entry_mut(point) else { return };

        let value = std::mem::replace(entry, light);
        if value == light { return; }

        self.update_light(point, value, -1);
        self.update_light(point, light, 1);
    }

    pub fn set_opacity(&mut self, point: Point, value: i32) {
        let Some(index) = self.opacity.index(point) else { return };
        if self.opacity.data[index] == value { return; }

        let bitset = self.sources.data[index];

        for delta in bitset.sources() {
            let other = point + delta;
            let light = self.light_radius.get(other);
            self.update_light(other, light, -1);
        }

        self.opacity.data[index] = value;

        for delta in bitset.sources() {
            let other = point + delta;
            let light = self.light_radius.get(other);
            self.update_light(other, light, 1);
        }
    }

    fn update_light(&mut self, point: Point, light: i32, delta: i32) {
        if light < 0 { return; }

        let (pos, dir) = (point, Point::default());
        let opacity_lookup = |p: Point| { self.opacity.get(p) };
        let vision = &mut self.visions[light as usize];
        vision.compute(&VisionArgs { pos, dir, opacity_lookup });

        for &p in vision.get_points_seen() {
            let Some(index) = self.light_values.index(p) else { continue };
            let entry = &mut self.light_values.data[index];
            assert!(*entry + delta >= 0);
            *entry += delta;

            self.sources.data[index].toggle(point - p);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

#[allow(soft_unstable)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{Bound, RNG};
    use rand::{Rng, SeedableRng};
    extern crate test;

    const SEED: u64 = 17;
    const SIDE: i32 = 50;
    const ITERATIONS: i32 = 500;
    const MAX_LIGHT_RADIUS: i32 = 12;

    #[test]
    fn test_lighting() {
        let size = Point(SIDE, SIDE);
        let mut rng = RNG::seed_from_u64(SEED);
        let mut lighting = Lighting::new(size);
        check_lighting(&lighting);

        for _ in 0..ITERATIONS {
            let x = rng.random_range(0..size.0);
            let y = rng.random_range(0..size.0);
            let point = Point(x, y);

            if rng.random_bool(0.25) {
                let light = rng.random_range(-1..=6);
                lighting.set_light(point, light);
            } else {
                let opacity = lighting.opacity.get(point);
                lighting.set_opacity(point, INITIAL_VISIBILITY - opacity);
            }
            check_lighting(&lighting);
            debug_lighting(&lighting);
        }
    }

    #[bench]
    fn bench_lighting_light_change(b: &mut test::Bencher) {
        let size = Point(SIDE, SIDE);
        let mut rng = RNG::seed_from_u64(SEED);
        let mut lighting = Lighting::new(size);

        for x in 0..lighting.opacity.size.0 {
            for y in 0..lighting.opacity.size.1 {
                let point = Point(x, y);
                let light = rng.random_range(-1..=6);
                lighting.set_light(point, light);
            }
        }

        b.iter(|| {
            let x = rng.random_range(0..size.0);
            let y = rng.random_range(0..size.0);
            let point = Point(x, y);
            let light = rng.random_range(-1..=MAX_LIGHT_RADIUS);
            lighting.set_light(point, light);
        });
    }

    #[bench]
    fn bench_lighting_opacity_change(b: &mut test::Bencher) {
        let size = Point(SIDE, SIDE);
        let mut rng = RNG::seed_from_u64(SEED);
        let mut lighting = Lighting::new(size);

        for x in 0..lighting.opacity.size.0 {
            for y in 0..lighting.opacity.size.1 {
                let point = Point(x, y);
                let light = rng.random_range(-1..=6);
                lighting.set_light(point, light);
            }
        }

        b.iter(|| {
            let x = rng.random_range(0..size.0);
            let y = rng.random_range(0..size.0);
            let point = Point(x, y);
            let opacity = lighting.opacity.get(point);
            lighting.set_opacity(point, INITIAL_VISIBILITY - opacity);
        });
    }

    fn check_lighting(lighting: &Lighting) {
        for x in 0..lighting.opacity.size.0 {
            for y in 0..lighting.opacity.size.1 {
                check_lighting_at_cell(lighting, Point(x, y));
            }
        }
    }

    fn check_lighting_at_cell(lighting: &Lighting, point: Point) {
        let mut expected = 0;
        let mut vision = Vision::new(MAX_LIGHT_RADIUS);

        for x in 0..lighting.opacity.size.0 {
            for y in 0..lighting.opacity.size.1 {
                let other = Point(x, y);
                let light = lighting.light_radius.get(other);
                if !Bound::new(light).contains(other - point) { continue; }

                let dir = Point::default();
                let opacity_lookup = |x| lighting.opacity.get(x);
                let args = VisionArgs { pos: other, dir, opacity_lookup };
                if vision.check_point(&args, point) { expected += 1; }
            }
        }

        let actual = lighting.get_light(point);
        assert!(actual == expected, "Expected: light @ {:?} == {}; got: {}",
                point, expected, actual);
    }

    fn debug_lighting(lighting: &Lighting) {
        for y in 0..lighting.opacity.size.1 {
            let mut line = String::new();
            for x in 0..lighting.opacity.size.0 {
                let point = Point(x, y);
                let opacity = lighting.opacity.get(point);
                let light = lighting.light_radius.get(point);
                let value = lighting.light_values.get(point);
                let ch = if opacity > 0 {
                    '#'
                } else if light >= 0 {
                    if light < 10 {
                        ('0' as i32 + light) as u8 as char
                    } else {
                        ('A' as i32 + light - 10) as u8 as char
                    }
                } else if value > 0 {
                    'o'
                } else {
                    '.'
                };
                let ch = ch as u32 - 0x20 + 0xff00;
                line.push(char::from_u32(ch).unwrap());
            }
            println!("{}", line);
        }
        println!("");
    }
}
