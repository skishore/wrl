use wrl_base::base::RNG;
use wrl_base::game::Tile;
use wrl_base::mapgen::{legacy_mapgen, mapgen};

use rand::SeedableRng;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 || (args[1] != "old" && args[1] != "new") {
        panic!("Usage: mapgen-test (old|new)");
    }

    let f = if args[1] == "old" { legacy_mapgen } else { mapgen };
    let mut rng = RNG::from_os_rng();
    let map = f(&mut rng);

    let mut last_color: Option<u32> = None;
    let mut line = String::default();

    for (point, &ch) in map.iter() {
        if ch == ' ' {
            line.push_str("  ");
            continue;
        }

        let (ch, color) = if let Some(x) = Tile::try_get(ch) {
            (x.glyph.ch().0 as u32, x.glyph.fg().0)
        } else {
            (ch as u32 + (0xff00 - 0x20), 0xffffff)
        };
        if Some(color) != last_color {
            let (r, g, b) = ((color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff);
            line.push_str(&format!("\x1b[38;2;{};{};{}m", r, g, b));
            last_color = Some(color);
        }
        line.push(char::from_u32(ch).unwrap());

        if point.0 + 1 == map.size().0 {
            println!("{}\x1b[0m", line);
            last_color = None;
            line.clear();
        }
    }
}
