use lazy_static::lazy_static;

//////////////////////////////////////////////////////////////////////////////

const NORM_2D: f64    = 1.0 / 47.0;
const ROOT_THREE: f64 = 1.7320508075688772;
const SQUISH_2D: f64  = (ROOT_THREE - 1.) / 2.;
const STRETCH_2D: f64 = (1. / ROOT_THREE - 1.) / 2.;

const BASE_2D: [[[i32; 3]; 3]; 2] = [
    [[1, 1, 0], [1, 0, 1], [0, 0, 0]],
    [[1, 1, 0], [1, 0, 1], [2, 1, 1]],
];

const LOOKUP_PAIRS_2D: [[i32; 2]; 12] = [
    [0,  1], [1,  0], [4,  1], [17, 0], [20, 2], [21, 2],
    [22, 5], [23, 5], [26, 4], [39, 3], [42, 4], [43, 3],
];

const P2D: [[i32; 4]; 6] = [
    [0, 0, 1, -1], [0, 0, -1, 1], [0, 2, 1, 1],
    [1, 2, 2,  0], [1, 2,  0, 2], [1, 0, 0, 0],
];

#[derive(Default)]
struct Contribution {
    dx: f64,
    dy: f64,
    xsb: i32,
    ysb: i32,
}

struct Precomputation {
    contributions: [[Contribution; 4]; 6],
    lookup: [i8; 128],
}

lazy_static! {
    static ref PRECOMPUTATION: Precomputation = Default::default();
}

impl Default for Precomputation {
    fn default() -> Self {
        let contributions = Default::default();
        let mut result = Self { contributions, lookup: [0; 128] };

        let set = |contribution: &mut Contribution, multiplier: i32, xsb: i32, ysb: i32| {
            contribution.dx = -xsb as f64 - multiplier as f64 * SQUISH_2D;
            contribution.dy = -ysb as f64 - multiplier as f64 * SQUISH_2D;
            contribution.xsb = xsb;
            contribution.ysb = ysb;
        };

        let mut i = 0;
        for [base, multiplier, dx, dy] in P2D {
            let mut j = 0;
            let contribution = &mut result.contributions[i];
            i += 1;
            for [bm, bx, by] in BASE_2D[base as usize] {
                set(&mut contribution[j], bm, bx, by);
                j += 1;
            }
            set(&mut contribution[j], multiplier, dx, dy);
        };

        result.lookup.fill(-1);
        for [source, target] in LOOKUP_PAIRS_2D {
            result.lookup[source as usize] = target as i8;
        }
        result
    }
}

pub struct SimplexNoise2D {
    permutation: [u8; 256],
}

impl SimplexNoise2D {
    pub fn new(seed: u32) -> Self {
        let mut seed = Self::shuffle(Self::shuffle(Self::shuffle(seed)));

        let mut permutation = [0; 256];
        for i in 0..256 { permutation[i] = i as u8; }

        for i in (0..256).rev() {
            seed = Self::shuffle(seed);
            let j = (seed + 31) % (i + 1);
            permutation.swap(i as usize, j as usize);
        }
        Self { permutation }
    }

    pub fn query(&self, x: f64, y: f64) -> f64 {
        let stretch_offset = (x + y) * STRETCH_2D;

        let xs = x + stretch_offset;
        let ys = y + stretch_offset;

        let xsb = xs.floor();
        let ysb = ys.floor();

        let squish_offset = (xsb + ysb) * SQUISH_2D;

        let dx0 = x - (xsb + squish_offset);
        let dy0 = y - (ysb + squish_offset);

        let xins = xs - xsb;
        let yins = ys - ysb;

        let insum = xins + yins;
        let hash = ((xins - yins + 1.) as i32) << 0 |
                   ((insum) as i32)            << 1 |
                   ((insum + yins) as i32)     << 2 |
                   ((insum + xins) as i32)     << 4;

        let mut value = 0.0;
        let index = PRECOMPUTATION.lookup[hash as usize];

        for contribution in &PRECOMPUTATION.contributions[index as usize] {
            let dx = dx0 + contribution.dx;
            let dy = dy0 + contribution.dy;

            let attn = 2. - dx * dx - dy * dy;
            if attn <= 0. { continue; }

            let px = xsb as i32 + contribution.xsb;
            let py = ysb as i32 + contribution.ysb;

            let index = self.permutation[(px & 0xff) as usize];
            let index = self.permutation[((index as i32 + py) & 0xff) as usize];
            let index = index & 0xe;

            let abs_grad_x = if index & 2 != 0 { 2 } else { 5 };
            let abs_grad_y = 7 - abs_grad_x;
            let grad_x = if index & 4 != 0 { -abs_grad_x } else { abs_grad_x };
            let grad_y = if index & 8 != 0 { -abs_grad_y } else { abs_grad_y };

            let part = grad_x as f64 * dx + grad_y as f64 * dy;
            value += attn * attn * attn * attn * part;
        }
        value * NORM_2D
    }

    fn shuffle(seed: u32) -> u32 {
        return seed.wrapping_mul(1664525).wrapping_add(1013904223);
    }
}

//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_noise() {
        let base = 5;
        let side = 75;
        let offset = -30;
        let noise = SimplexNoise2D::new(17);
        for y in 0..side {
            let mut line = String::default();
            for x in 0..side {
                let dx = (x + offset) as f64 / (base as f64);
                let dy = (y + offset) as f64 / (base as f64);
                let value = noise.query(dx, dy);

                let strength = std::cmp::min((value.abs() * 256.) as i32, 0xff);
                let r = if value < 0.  { strength } else { 0 };
                let b = if value >= 0. { strength } else { 0 };
                let g = 0;
                line.push_str(&format!("\x1b[48;2;{};{};{}m  ", r, g, b));
            }
            println!("{}\x1b[0m", line);
        }
    }
}
