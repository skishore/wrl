use wrl_base::base::{Point, RNG};
use wrl_base::game::Tile;
use wrl_base::mapgen::mapgen;

use rand::SeedableRng;

fn main() {
    let mut rng = RNG::from_entropy();
    let map = mapgen(&mut rng);

    for y in 0..map.size.1 {
        let mut line = String::default();
        let mut last_color: Option<(u32, u32, u32)> = None;
        for x in 0..map.size.0 {
            let c = map.get(Point(x, y));
            if c == ' ' {
                line.push_str("  ");
                continue;
            }

            let (ch, color) = if let Some(x) = Tile::try_get(c) {
                let (ch, fg) = (x.glyph.ch().0, x.glyph.fg().0);
                (ch as u32, (fg >> 16, (fg >> 8) & 0xff, fg & 0xff))
            } else {
                let ch = c as u32 + (0xff00 - 0x20);
                (ch, (0xff as u32, 0xff as u32, 0xff as u32))
            };
            if Some(color) != last_color {
                let (r, g, b) = color;
                line.push_str(&format!("\x1b[38;2;{};{};{}m", r, g, b));
                last_color = Some(color);
            }
            line.push(char::from_u32(ch).unwrap());
        }
        println!("{}\x1b[0m", line);
    }
}
