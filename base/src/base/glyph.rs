use std::fmt::Debug;

use crate::static_assert_size;
use super::point::{Matrix, Point};

//////////////////////////////////////////////////////////////////////////////

// Rendering helpers: Char, Color, Glyph

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
        self.apply_light(alpha, alpha, alpha)
    }

    pub fn apply_light(&self, r: f64, g: f64, b: f64) -> Self {
        let s = self.0;
        let (sr, sg, sb) = (s >> 16, (s >> 8) & 0xff, s & 0xff);
        let r = std::cmp::min((r * sr as f64) as i32, 0xff);
        let g = std::cmp::min((g * sg as f64) as i32, 0xff);
        let b = std::cmp::min((b * sb as f64) as i32, 0xff);
        Color(((r << 16) | (g << 8) | b) as u32)
    }

    fn interpolate(&self, alpha: f64, target: u32) -> Self {
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
    // Constructors:

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

// Buffer and Slice:

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
        let (root, size) = (Point::default(), buffer.size());
        Slice::new(buffer, Rect { root, size })
    }
}

impl<'a> Slice<'a> {
    pub fn new(buffer: &'a mut Buffer, bounds: Rect) -> Self {
        Self { buffer, bounds, cursor: Point::default(), fg: None, bg: None }
    }

    // Basic API

    pub fn get(&self, point: Point) -> Glyph {
        if !self.contains(point) { return *self.buffer.default(); }
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
