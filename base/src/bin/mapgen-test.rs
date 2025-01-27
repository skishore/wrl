use wrl_base::base::{Point, RNG};
use wrl_base::mapgen::{MapgenConfig, mapgen};

use rand::SeedableRng;

fn main() {
    let config = MapgenConfig::default();
    let mut rng = RNG::from_entropy();
    let map = mapgen(&config, &mut rng);

    for y in 0..map.size.1 {
        for x in 0..map.size.0 {
            let c = map.get(Point(x, y));
            print!("{}", char::from_u32(c as u32 + (0xff00 - 0x20)).unwrap());
        }
        print!("\n");
    }
}
