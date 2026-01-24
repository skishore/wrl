use crate::base::{Bound, Matrix, Point};
use crate::shadowcast::{INITIAL_VISIBILITY, Vision, VisionArgs};

//////////////////////////////////////////////////////////////////////////////

struct Lighting {
    max_radius: i32,
    light_bounds: Matrix<Bound>,
    light_values: Matrix<i32>,
    opacity: Matrix<i32>,
    vision: Vision,
}

impl Lighting {
    pub fn new(size: Point, max_radius: i32) -> Self {
        let mut result = Self {
            max_radius,
            light_bounds: Matrix::new(size, Bound::new(-1)),
            light_values: Matrix::new(size, 0),
            opacity: Matrix::new(size, INITIAL_VISIBILITY),
            vision: Vision::new(max_radius),
        };
        result.opacity.fill(0);
        result
    }

    pub fn get_light(&self, point: Point) -> i32 {
        self.light_values.get(point)
    }

    pub fn set_light(&mut self, point: Point, light: Bound) {
        let Some(entry) = self.light_bounds.entry_mut(point) else { return };

        let bound = *entry;
        if bound.radius == light.radius { return; }

        *entry = light;
        self.update_light(point, bound, -1);
        self.update_light(point, light, 1);
    }

    pub fn set_opacity(&mut self, point: Point, value: i32) {
        let Some(index) = self.opacity.index(point) else { return };
        if self.opacity.data[index] == value { return; }

        let r = self.max_radius;

        let mut lights = vec![];
        for dx in -r..=r {
            for dy in -r..=r {
                let delta = Point(dx, dy);
                let other = point + delta;
                let light = self.light_bounds.get(other);
                if light.contains(delta) { lights.push((other, light)); }
            }
        }

        for &(other, light) in &lights { self.update_light(other, light, -1); }

        self.opacity.data[index] = value;

        for &(other, light) in &lights { self.update_light(other, light, 1); }
    }

    fn update_light(&mut self, point: Point, light: Bound, delta: i32) {
        if light.is_empty() { return; }

        let (pos, dir) = (point, Point::default());
        let opacity_lookup = |p: Point| {
            if !light.contains(p - point) { return INITIAL_VISIBILITY; }
            self.opacity.get(p)
        };
        self.vision.compute(&VisionArgs { pos, dir, opacity_lookup });

        for &p in self.vision.get_points_seen() {
            if !light.contains(p - point) { continue; }
            let Some(entry) = self.light_values.entry_mut(p) else { continue };
            assert!(*entry + delta >= 0);
            *entry += delta;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

#[allow(soft_unstable)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::RNG;
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
        let mut lighting = Lighting::new(size, MAX_LIGHT_RADIUS);
        check_lighting(&lighting);

        for _ in 0..ITERATIONS {
            let x = rng.random_range(0..size.0);
            let y = rng.random_range(0..size.0);
            let point = Point(x, y);

            if rng.random_bool(0.25) {
                let radius = rng.random_range(-1..=6);
                lighting.set_light(point, Bound::new(radius));
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
        let mut lighting = Lighting::new(size, MAX_LIGHT_RADIUS);

        for x in 0..lighting.opacity.size.0 {
            for y in 0..lighting.opacity.size.1 {
                let point = Point(x, y);
                let radius = rng.random_range(-1..=6);
                lighting.set_light(point, Bound::new(radius));
            }
        }

        b.iter(|| {
            let x = rng.random_range(0..size.0);
            let y = rng.random_range(0..size.0);
            let point = Point(x, y);
            let radius = rng.random_range(-1..=MAX_LIGHT_RADIUS);
            lighting.set_light(point, Bound::new(radius));
        });
    }

    #[bench]
    fn bench_lighting_opacity_change(b: &mut test::Bencher) {
        let size = Point(SIDE, SIDE);
        let mut rng = RNG::seed_from_u64(SEED);
        let mut lighting = Lighting::new(size, MAX_LIGHT_RADIUS);

        for x in 0..lighting.opacity.size.0 {
            for y in 0..lighting.opacity.size.1 {
                let point = Point(x, y);
                let radius = rng.random_range(-1..=6);
                lighting.set_light(point, Bound::new(radius));
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
                let light = lighting.light_bounds.get(other);
                if !light.contains(other - point) { continue; }

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
                let radius = lighting.light_bounds.get(point);
                let values = lighting.light_values.get(point);
                let ch = if opacity > 0 {
                    '#'
                } else if radius.radius >= 0 {
                    if radius.radius < 10 {
                        ('0' as i32 + radius.radius) as u8 as char
                    } else {
                        ('A' as i32 + radius.radius - 10) as u8 as char
                    }
                } else if values > 0 {
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
