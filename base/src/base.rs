use std::cmp::{max, min};
use std::fmt::Debug;

use rand::Rng;

//////////////////////////////////////////////////////////////////////////////

// Basics

#[macro_export]
macro_rules! gene {
    ($x:expr) => { $x }
}

#[macro_export]
macro_rules! static_assert_size {
    ($x:ty, $y:expr) => {
        const _: fn() = || { let _ = std::mem::transmute::<$x, [u8; $y]>; };
    }
}

pub type RNG = rand::rngs::StdRng;
pub type HashSet<K> = fxhash::FxHashSet<K>;
pub type HashMap<K, V> = fxhash::FxHashMap<K, V>;

pub fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min { min } else if x > max { max } else { x }
}

pub fn sample<'a, T>(xs: &'a [T], rng: &mut RNG) -> &'a T {
    assert!(!xs.is_empty());
    &xs[rng.random_range(0..xs.len())]
}

pub fn weighted<'a, T: Debug>(xs: &'a [(i32, T)], rng: &mut RNG) -> &'a T {
    let total = xs.iter().fold(0, |acc, x| acc + x.0);
    assert!(total > 0, "Total: {}; values: {:?}", total, xs);
    let mut value = rng.random_range(0..total);
    for (weight, choice) in xs {
        value -= weight;
        if value <= 0 { return choice; }
    }
    assert!(false);
    &xs[xs.len() - 1].1
}

//////////////////////////////////////////////////////////////////////////////

// Rendering helpers

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Char(pub u16);
static_assert_size!(Char, 2);

impl Char {
    pub fn is_wide(&self) -> bool { self.0 >= 0xff00 }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Color(pub u32);
static_assert_size!(Color, 4);

impl From<i32> for Color {
    fn from(v: i32) -> Self {
        let (r, g, b) = ((v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff);
        Color(((r as u32) << 16) | ((g as u32) << 8) | b as u32)
    }
}

impl Color {
    pub fn black() -> Self { Self::gray(0) }
    pub fn white() -> Self { Self::gray(0xff) }
    pub fn gray(n: u8) -> Self { Self(0x010101 * n as u32) }

    pub fn brighten(&self, alpha: f64) -> Self {
        self.interpolate(alpha, 0xffffff)
    }

    pub fn fade(&self, alpha: f64) -> Self {
        self.interpolate(1. - alpha, 0x000000)
    }

    pub fn interpolate(&self, alpha: f64, target: u32) -> Self {
        let (s, b) = (self.0, target);
        let (x, y) = (1. - alpha, alpha);
        let (sr, sg, sb) = (s >> 16, (s >> 8) & 0xff, s & 0xff);
        let (br, bg, bb) = (b >> 16, (b >> 8) & 0xff, b & 0xff);
        let r = (x * sr as f64 + y * br as f64) as i32;
        let g = (x * sg as f64 + y * bg as f64) as i32;
        let b = (x * sb as f64 + y * bb as f64) as i32;
        Color(((r << 16) | (g << 8) | b) as u32)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Glyph(u64);
static_assert_size!(Glyph, 8);

impl From<char> for Glyph {
    fn from(val: char) -> Self { Self::char(val) }
}

impl Glyph {
    // Constructors

    pub fn new(ch: Char, fg: Color, bg: Color) -> Self {
        Self((ch.0 as u64) | ((fg.0 as u64) << 16) | ((bg.0 as u64) << 40))
    }

    pub fn char(ch: char) -> Self {
        Self::new(Char(ch as u16), Color::white(), Color::black())
    }

    pub fn chfg<T: Into<Color>>(ch: char, fg: T) -> Self {
        Self::new(Char(ch as u16), fg.into(), Color::black())
    }

    pub fn wide(ch: char) -> Self {
        let ch = Char((ch as u16) + (0xff00 - 0x20));
        Self::new(ch, Color::white(), Color::black())
    }

    pub fn wdfg<T: Into<Color>>(ch: char, fg: T) -> Self {
        let ch = Char((ch as u16) + (0xff00 - 0x20));
        Self::new(ch, fg.into(), Color::black())
    }

    pub fn with_fg<T: Into<Color>>(&self, color: T) -> Self {
        Self((self.0 & 0xffffff000000ffff) | ((color.into().0 as u64) << 16))
    }

    pub fn with_bg<T: Into<Color>>(&self, color: T) -> Self {
        Self((self.0 & 0x000000ffffffffff) | ((color.into().0 as u64) << 40))
    }

    pub fn ray(delta: Point) -> char {
        let Point(x, y) = delta;
        let (ax, ay) = (x.abs(), y.abs());
        if ax > 2 * ay { return '-'; }
        if ay > 2 * ax { return '|'; }
        if (x > 0) == (y > 0) { '\\' } else { '/' }
    }

    // Field getters

    pub fn ch(&self) -> Char { Char(self.0 as u16) }

    pub fn fg(&self) -> Color { Color((self.0 >> 16) as u32 & 0xffffff) }

    pub fn bg(&self) -> Color { Color((self.0 >> 40) as u32 & 0xffffff) }
}

//////////////////////////////////////////////////////////////////////////////

// Point and Direction

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Point(pub i32, pub i32);
static_assert_size!(Point, 8);

impl Point {
    pub fn dot(&self, other: Point) -> i64 {
        (self.0 as i64 * other.0 as i64) + (self.1 as i64 * other.1 as i64)
    }

    pub fn in_l2_range(&self, range: i32) -> bool {
        self.len_l2() <= range as f64 - 0.5
    }

    pub fn len_nethack(&self) -> i32 {
        let (ax, ay) = (self.0.abs() as i64, self.1.abs() as i64);
        let (min, max) = (min(ax, ay), max(ax, ay));
        ((46 * min + 95 * max + 25) / 100) as i32
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

    pub fn normalize(&self, length: f64) -> Point {
        let factor = length / self.len_l2();
        let x = (self.0 as f64 * factor).round() as i32;
        let y = (self.1 as f64 * factor).round() as i32;
        Point(x, y)
    }
}

impl std::ops::Add for Point {
    type Output = Point;
    fn add(self, other: Point) -> Point {
        Point(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Sub for Point {
    type Output = Point;
    fn sub(self, other: Point) -> Point {
        Point(self.0 - other.0, self.1 - other.1)
    }
}

pub mod dirs {
    use crate::base::Point;

    pub const NONE: Point = Point( 0,  0);
    pub const N:    Point = Point( 0, -1);
    pub const S:    Point = Point( 0,  1);
    pub const E:    Point = Point( 1,  0);
    pub const W:    Point = Point(-1,  0);
    pub const NE:   Point = Point( 1, -1);
    pub const NW:   Point = Point(-1, -1);
    pub const SE:   Point = Point( 1,  1);
    pub const SW:   Point = Point(-1,  1);

    pub const ALL: [Point; 8] = [N, S, E, W, NE, NW, SE, SW];
    pub const CARDINAL: [Point; 4] = [N, S, E, W];
}

//////////////////////////////////////////////////////////////////////////////

// Matrix

#[derive(Clone, Default)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub size: Point,
    pub default: T,
}

// SAFETY: Non-none index() results are always valid indices into data.
impl<T: Clone> Matrix<T> {
    pub fn new(size: Point, value: T) -> Self {
        assert!(0 <= size.0);
        assert!(0 <= size.1);
        let mut data = Vec::new();
        data.resize((size.0 * size.1) as usize, value.clone());
        Self { data, size, default: value }
    }

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

pub type Buffer = Matrix<Glyph>;

#[derive(Clone, Copy, Default)]
pub struct Rect { pub root: Point, pub size: Point }

pub struct Slice<'a> {
    buffer: &'a mut Buffer,
    bounds: Rect,
    cursor: Point,
    fg: Option<Color>,
    bg: Option<Color>,
}

impl<'a> From<&'a mut Buffer> for Slice<'a> {
    fn from(buffer: &'a mut Buffer) -> Slice<'a> {
        let (root, size) = (Point::default(), buffer.size);
        Slice::new(buffer, Rect { root, size })
    }
}

impl<'a> Slice<'a> {
    pub fn new(buffer: &'a mut Buffer, bounds: Rect) -> Self {
        Self { buffer, bounds, cursor: Point::default(), fg: None, bg: None }
    }

    // Basic API

    pub fn get(&self, point: Point) -> Glyph {
        if !self.contains(point) { return self.buffer.default; }
        self.buffer.get(self.bounds.root + point)
    }

    pub fn set(&mut self, point: Point, glyph: Glyph) {
        if !self.contains(point) { return; }
        let glyph = self.fg.map(|x| glyph.with_fg(x)).unwrap_or(glyph);
        let glyph = self.bg.map(|x| glyph.with_bg(x)).unwrap_or(glyph);
        self.buffer.set(self.bounds.root + point, glyph);
    }

    pub fn fill(&mut self, glyph: Glyph) {
        for x in 0..self.bounds.size.0 {
            for y in 0..self.bounds.size.1 {
                self.buffer.set(self.bounds.root + Point(x, y), glyph);
            }
        }
    }

    pub fn contains(&self, point: Point) -> bool {
        let Point(px, py) = point;
        let Point(sx, sy) = self.bounds.size;
        0 <= px && px < sx && 0 <= py && py < sy
    }

    pub fn size(&self) -> Point { self.bounds.size }

    // Cursor API

    pub fn get_cursor(&self) -> Point { self.cursor }

    pub fn set_cursor(&mut self, p: Point) {
        self.cursor = p;
        self.set_fg(None).set_bg(None);
    }

    pub fn newline(&mut self) -> &mut Self {
        self.newlines(1)
    }

    pub fn newlines(&mut self, n: usize) -> &mut Self {
        self.cursor = Point(0, self.cursor.1 + n as i32);
        self.set_fg(None).set_bg(None)
    }

    pub fn space(&mut self) -> &mut Self {
        self.spaces(1)
    }

    pub fn spaces(&mut self, n: usize) -> &mut Self {
        self.cursor.0 += n as i32;
        self
    }

    pub fn write_chr<T: Into<Glyph>>(&mut self, t: T) -> &mut Self {
        let glyph = t.into();
        self.set(self.cursor, glyph);
        self.spaces(if glyph.ch().is_wide() { 2 } else { 1 })
    }

    pub fn write_str(&mut self, text: &str) -> &mut Self {
        text.chars().for_each(|x| { self.write_chr(x); });
        self
    }

    pub fn set_fg(&mut self, c: Option<Color>) -> &mut Self { self.fg = c; self }

    pub fn set_bg(&mut self, c: Option<Color>) -> &mut Self { self.bg = c; self }
}

//////////////////////////////////////////////////////////////////////////////

// Bresenham line-of-sight

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
