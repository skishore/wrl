use std::cmp::max;
use std::fmt::Debug;

use crate::static_assert_size;

//////////////////////////////////////////////////////////////////////////////

// Bound:

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Bound {
    pub cutoff: i32,
    pub radius: i32,
}

impl Bound {
    pub const fn new(r: i32) -> Self {
        if r < 0 { return Self { cutoff: 0, radius: -1 }; }
        let cutoff = r * r + r + if r == 4 { 0 } else if r == 5 { -1 } else { 1 };
        Self { cutoff, radius: r }
    }

    pub fn contains(&self, delta: Delta) -> bool {
        delta.len_l2_squared() < self.cutoff as i64
    }

    pub fn is_empty(&self) -> bool {
        self.cutoff == 0
    }
}

//////////////////////////////////////////////////////////////////////////////

// Delta:

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Delta(pub i32, pub i32);
static_assert_size!(Delta, 8);

impl Delta {
    pub fn dot(&self, other: Delta) -> i64 {
        (self.0 as i64 * other.0 as i64) + (self.1 as i64 * other.1 as i64)
    }

    pub fn bound_radius(&self) -> i32 {
        let r = self.len_l2() as i32;
        if Bound::new(r).contains(*self) { r } else { r + 1 }
    }

    pub fn cross_product(&self, with: Self) -> i32 {
        let Delta(ax, ay) = self;
        let Delta(bx, by) = with;
        (ax * by - bx * ay).abs()
    }

    pub fn len_taxicab(&self) -> i32 {
        self.0.abs() + self.1.abs()
    }

    pub fn len_l1(&self) -> i32 {
        max(self.0.abs(), self.1.abs())
    }

    pub fn len_l2(&self) -> f64 {
        (self.len_l2_squared() as f64).sqrt()
    }

    pub fn len_l2_squared(&self) -> i64 {
        let (x, y) = (self.0 as i64, self.1 as i64);
        x * x + y * y
    }

    pub fn normalize(&self, length: f64) -> Delta {
        let factor = length / self.len_l2();
        let x = (self.0 as f64 * factor).round() as i32;
        let y = (self.1 as f64 * factor).round() as i32;
        Delta(x, y)
    }

    pub fn rotate_l(&self) -> Delta {
        let r = self.bound_radius();
        let (dx, dy) = (self.1.signum(), -self.0.signum());
        let shifts = [Delta(dx, 0), Delta(0, dy), Delta(dx, dy)];
        for x in shifts.into_iter().map(|x| *self + x)  {
            if x != *self && x.bound_radius() == r { return x; }
        }
        panic!();
    }

    pub fn rotate_r(&self) -> Delta {
        let r = self.bound_radius();
        let (dx, dy) = (-self.1.signum(), self.0.signum());
        let shifts = [Delta(dx, 0), Delta(0, dy), Delta(dx, dy)];
        for x in shifts.into_iter().map(|x| *self + x)  {
            if x != *self && x.bound_radius() == r { return x; }
        }
        panic!();
    }

    pub fn scale(&self, scale: i32) -> Delta {
        Delta(scale * self.0, scale * self.1)
    }
}

pub mod dirs {
    use super::Delta;

    pub const NONE: Delta = Delta( 0,  0);
    pub const N:    Delta = Delta( 0, -1);
    pub const S:    Delta = Delta( 0,  1);
    pub const E:    Delta = Delta( 1,  0);
    pub const W:    Delta = Delta(-1,  0);
    pub const NE:   Delta = Delta( 1, -1);
    pub const NW:   Delta = Delta(-1, -1);
    pub const SE:   Delta = Delta( 1,  1);
    pub const SW:   Delta = Delta(-1,  1);

    pub const ALL: [Delta; 8] = [N, S, E, W, NE, NW, SE, SW];
    pub const CARDINAL: [Delta; 4] = [N, S, E, W];
}

//////////////////////////////////////////////////////////////////////////////

// Point:

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Point(pub i32, pub i32);
static_assert_size!(Point, 8);

impl Point {
    pub const ORIGIN: Point = Point(0, 0);

    pub fn dir(self) -> Delta {
        Delta(self.0, self.1)
    }
}

impl std::ops::Add<Delta> for Delta {
    type Output = Delta;
    fn add(self, other: Delta) -> Delta {
        Delta(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Add<Point> for Delta {
    type Output = Point;
    fn add(self, other: Point) -> Point {
        Point(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Sub<Delta> for Delta {
    type Output = Delta;
    fn sub(self, other: Delta) -> Delta {
        Delta(self.0 - other.0, self.1 - other.1)
    }
}

impl std::ops::Add<Delta> for Point {
    type Output = Point;
    fn add(self, other: Delta) -> Point {
        Point(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Sub<Delta> for Point {
    type Output = Point;
    fn sub(self, other: Delta) -> Point {
        Point(self.0 - other.0, self.1 - other.1)
    }
}

impl std::ops::Sub<Point> for Point {
    type Output = Delta;
    fn sub(self, other: Point) -> Delta {
        Delta(self.0 - other.0, self.1 - other.1)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Matrix:

#[derive(Clone, Default)]
pub struct Matrix<T> {
    data: Box<[T]>,
    size: Point,
    default: T,
}

// SAFETY: Non-none index() results are always valid indices into data.
impl<T: Clone> Matrix<T> {
    pub fn new(size: Point, value: T) -> Self {
        assert!(0 <= size.0);
        assert!(0 <= size.1);
        let mut data = Vec::new();
        data.resize((size.0 * size.1) as usize, value.clone());
        Self { data: data.into_boxed_slice(), size, default: value }
    }

    // Trivial getters:

    pub fn mut_data(&mut self) -> &mut [T] { &mut *self.data }

    pub fn raw_data(&self) -> &[T] { &*self.data }

    pub fn default(&self) -> &T { &self.default }

    pub fn size(&self) -> Point { self.size }

    // Iterators:

    pub fn iter(&self) -> impl Iterator<Item = (Point, &T)> {
        std::iter::zip(self.iter_points(), self.data.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Point, &mut T)> {
        std::iter::zip(self.iter_points(), self.data.iter_mut())
    }

    pub fn iter_points(&self) -> impl Iterator<Item = Point> + use<T> {
        let Point(sx, sy) = self.size;
        (0..sy).flat_map(move |y| (0..sx).map(move |x| Point(x, y)))
    }

    // Non-trivial methods:

    pub fn get(&self, point: Point) -> T {
        let Some(x) = self.index(point) else { return self.default.clone(); };
        unsafe { self.data.get_unchecked(x).clone() }
    }

    pub fn set(&mut self, point: Point, value: T) {
        let Some(x) = self.index(point) else { return; };
        unsafe { *self.data.get_unchecked_mut(x) = value; }
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    pub fn entry_ref(&self, point: Point) -> &T {
        let Some(x) = self.index(point) else { return &self.default; };
        unsafe { self.data.get_unchecked(x) }
    }

    pub fn entry_mut(&mut self, point: Point) -> Option<&mut T> {
        let Some(x) = self.index(point) else { return None; };
        unsafe { Some(self.data.get_unchecked_mut(x)) }
    }

    #[inline(always)]
    pub fn contains(&self, point: Point) -> bool {
        let Point(px, py) = point;
        let Point(sx, sy) = self.size;
        0 <= px && px < sx && 0 <= py && py < sy
    }

    #[inline(always)]
    pub fn index(&self, point: Point) -> Option<usize> {
        if !self.contains(point) { return None; }
        Some((point.0 + point.1 * self.size.0) as usize)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Bresenham digital line:

#[allow(non_snake_case)]
pub fn DeltaLOS(x: Delta) -> Vec<Delta> {
    let origin = Point(0, 0);
    LOS(origin, origin + x).into_iter().map(|x| x - origin).collect()
}

#[allow(non_snake_case)]
pub fn LOS(a: Point, b: Point) -> Vec<Point> {
    let x_diff = (a.0 - b.0).abs();
    let y_diff = (a.1 - b.1).abs();
    let x_sign = if b.0 < a.0 { -1 } else { 1 };
    let y_sign = if b.1 < a.1 { -1 } else { 1 };

    let size = (max(x_diff, y_diff) + 1) as usize;
    let mut result = vec![];
    result.reserve_exact(size);
    result.push(a);

    let mut test = 0;
    let mut current = a;

    if x_diff >= y_diff {
        test = (x_diff + test) / 2;
        for _ in 0..x_diff {
            current.0 += x_sign;
            test -= y_diff;
            if test < 0 {
                current.1 += y_sign;
                test += x_diff;
            }
            result.push(current);
        }
    } else {
        test = (y_diff + test) / 2;
        for _ in 0..y_diff {
            current.1 += y_sign;
            test -= x_diff;
            if test < 0 {
                current.0 += x_sign;
                test += y_diff;
            }
            result.push(current);
        }
    }

    assert!(result.len() == size);
    result
}

//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;

    #[test]
    fn test_bound_radius() {
        let expected_bound_radius = |d: Delta| {
            for r in 0..=d.len_taxicab() {
                if Bound::new(r).contains(d) { return r; }
            }
            panic!("Failed to get true bound_radius: {:?}", d);
        };
        let radius = 10;
        for x in -radius..=radius {
            for y in -radius..=radius {
                let d = Delta(x, y);
                assert!(d.bound_radius() == expected_bound_radius(d));
            }
        }
    }

    #[test]
    fn test_rotate_left() {
        assert!(dirs::E.rotate_l()  == dirs::NE);
        assert!(dirs::NE.rotate_l() == dirs::N);
        assert!(dirs::N.rotate_l()  == dirs::NW);
        assert!(dirs::NW.rotate_l() == dirs::W);

        assert!(Delta(2, 0).rotate_r()  == Delta(2, 1));
        assert!(Delta(2, 1).rotate_r()  == Delta(1, 2));
        assert!(Delta(1, 2).rotate_r()  == Delta(0, 2));
        assert!(Delta(0, 2).rotate_r()  == Delta(-1, 2));
        assert!(Delta(-1, 2).rotate_r() == Delta(-2, 1));
        assert!(Delta(-2, 1).rotate_r() == Delta(-2, 0));
    }

    #[test]
    fn test_rotate_right() {
        assert!(dirs::E.rotate_r()  == dirs::SE);
        assert!(dirs::NE.rotate_r() == dirs::E);
        assert!(dirs::N.rotate_r()  == dirs::NE);
        assert!(dirs::NW.rotate_r() == dirs::N);

        assert!(Delta(2, 0).rotate_l()   == Delta(2, -1));
        assert!(Delta(2, -1).rotate_l()  == Delta(1, -2));
        assert!(Delta(1, -2).rotate_l()  == Delta(0, -2));
        assert!(Delta(0, -2).rotate_l()  == Delta(-1, -2));
        assert!(Delta(-1, -2).rotate_l() == Delta(-2, -1));
        assert!(Delta(-2, -1).rotate_l() == Delta(-2, 0));
    }
}
