use std::cell::RefCell;

use rand_distr::num_traits::Pow;

use crate::base::{clamp, fmax};
use crate::wg_simplex::SimplexNoise2D;

//////////////////////////////////////////////////////////////////////////////

const ISLAND_RADIUS: i32 = 1024;
const WORLD_HEIGHT: i32 = 256;
const SEA_LEVEL: i32 = WORLD_HEIGHT / 4;

trait Noise = Fn(f64, f64) -> f64;

thread_local! {
    static SEED: RefCell<u32> = Default::default();
    static NOISES: RefCell<Noises> = Default::default();
}

fn seed() -> u32 {
    SEED.with_borrow_mut(|x| { *x += 1; *x - 1 })
}

fn MinetestNoise2D(offset: f64, scale: f64, spread: f64, octaves: i32,
                   persistence: f64, lacunarity: f64) -> impl Noise {
    let inverse_spread = 1.0 / spread;
    let noises: Vec<_> = (0..octaves).map(|_| SimplexNoise2D::new(seed())).collect();

    move |x: f64, y: f64| {
        let mut result = 0.0;
        let mut s = inverse_spread;
        let mut g = 1.0;

        for noise in &noises {
            result += g * noise.query(x * s, y * s);
            g *= persistence;
            s *= lacunarity;
        }
        scale * result + offset
    }
}

fn RidgeNoise(octaves: u32, persistence: f64, spread: f64) -> impl Noise {
    let inverse_spread = 1.0 / spread;
    let noises: Vec<_> = (0..octaves).map(|_| SimplexNoise2D::new(seed())).collect();

    move |x: f64, y: f64| {
        let mut result = 0.0;
        let mut s = inverse_spread;
        let mut g = 1.0;

        for noise in &noises {
            result += g * (1. - noise.query(x * s, y * s).abs());
            g *= persistence;
            s *= 2.0;
        }
        result
    }
}

struct Noises {
    mgv7_np_cliff_select:    Box<dyn Noise>,
    mgv7_np_mountain_select: Box<dyn Noise>,
    mgv7_np_terrain_ground:  Box<dyn Noise>,
    mgv7_np_terrain_cliff:   Box<dyn Noise>,
    mgv7_mountain_ridge: Box<dyn Noise>,
}

impl Default for Noises {
    fn default() -> Self {
        let mgv7_np_cliff_select    = Box::new(MinetestNoise2D(0., 1.,  512., 4, 0.7, 2.));
        let mgv7_np_mountain_select = Box::new(MinetestNoise2D(0., 1.,  512., 4, 0.7, 2.));
        let mgv7_np_terrain_ground  = Box::new(MinetestNoise2D(2., 8.,  512., 6, 0.6, 2.));
        let mgv7_np_terrain_cliff   = Box::new(MinetestNoise2D(8., 16., 512., 6, 0.6, 2.));
        let mgv7_mountain_ridge = Box::new(RidgeNoise(4, 0.5, 500.));

        Self {
            mgv7_np_cliff_select,
            mgv7_np_mountain_select,
            mgv7_np_terrain_ground,
            mgv7_np_terrain_cliff,
            mgv7_mountain_ridge,
        }
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum Block { Bedrock, Dirt, Grass, Sand, Snow, Stone }

pub struct HeightmapResult {
    pub block: Block,
    pub height: i32,
}

pub fn heightmap(x: i32, z: i32) -> HeightmapResult {
    let mut result = HeightmapResult { block: Block::Bedrock, height: 0 };

    let base = ((x * x + z * z) as f64).sqrt() / ISLAND_RADIUS as f64;
    let falloff = 16. * base * base;

    if falloff >= SEA_LEVEL as f64 { return result; }

    NOISES.with_borrow(|noises| {
        let scale = 4.;
        let (x, z) = (scale * x as f64, scale * z as f64);

        let cliff_select = (noises.mgv7_np_cliff_select)(x, z);
        let cliff_x = clamp(16. * cliff_select.abs() - 4., 0., 1.);

        let mountain_select = (noises.mgv7_np_mountain_select)(x, z);
        let mountain_x = fmax(8. * mountain_select, 0.).sqrt();

        let cliff = cliff_x - mountain_x;
        let mountain = -cliff;

        let height_ground = (noises.mgv7_np_terrain_ground)(x, z);
        let height_cliff = if cliff > 0. {
            (noises.mgv7_np_terrain_cliff)(x, z)
        } else {
            height_ground
        };
        let height_mountain = if mountain > 0. {
            height_ground + 64. * ((noises.mgv7_mountain_ridge)(x, z) - 1.25).pow(1.5)
        } else {
            height_ground
        };

        let height = if height_mountain > height_ground {
            height_mountain * mountain + height_ground * (1. - mountain)
        } else if height_cliff > height_ground {
            height_cliff * cliff + height_ground * (1. - cliff)
        } else {
            height_ground
        };

        let truncated = (height - falloff) as i32;
        let abs_height = truncated + SEA_LEVEL;
        let block = if truncated < -1 {
            Block::Dirt
        } else if height_mountain > height_ground {
            let base = height - (72. - 8. * mountain);
            if base > 0. { Block::Snow } else { Block::Stone }
        } else if height_cliff > height_ground {
            Block::Dirt
        } else if truncated < 1 {
            Block::Sand
        } else {
            Block::Grass
        };

        result.block = block;
        result.height = abs_height;
        result
    })
}
