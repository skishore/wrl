use wrl_base::base::{HashMap, Point, RNG};
use wrl_base::mapgen::mapgen;

use lazy_static::lazy_static;
use rand::SeedableRng;

lazy_static! {
    static ref GLYPHS: HashMap<char, (char, (i32, i32, i32))> = [
        ('#', ('#', (32, 96, 0))),
        ('.', ('.', (255, 255, 255))),
        ('"', ('"', (64, 192, 0))),
        ('~', ('~', (0, 128, 255))),
        ('*', ('*', (192, 128, 0))),
        ('=', ('=', (255, 128, 0))),
        ('R', ('.', (255, 128, 0))),
    ].into_iter().collect();
}

fn main() {
    let mut rng = RNG::from_entropy();
    let map = mapgen(&mut rng);

    for y in 0..map.size.1 {
        let mut line = String::default();
        let mut last_color: Option<(i32, i32, i32)> = None;
        for x in 0..map.size.0 {
            let c = map.get(Point(x, y));
            if c == ' ' {
                line.push_str("  ");
                continue;
            }

            let &(glyph, color) = GLYPHS.get(&c).unwrap_or(&(c, (255, 255, 255)));
            if Some(color) != last_color {
                let (r, g, b) = color;
                line.push_str(&format!("\x1b[38;2;{};{};{}m", r, g, b));
                last_color = Some(color);
            }
            line.push(char::from_u32(glyph as u32 + (0xff00 - 0x20)).unwrap());
        }
        println!("{}\x1b[0m", line);
    }
}
