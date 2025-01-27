use crate::base::{HashSet, Matrix, Point, RNG};

use rand::Rng;
use rand::seq::SliceRandom;

//////////////////////////////////////////////////////////////////////////////

pub struct RoomStep {
    pub min_size: i32,
    pub max_size: i32,
    pub attempts: i32,
}

pub struct MapgenConfig {
    pub size: Point,
    pub room_series: Vec<RoomStep>,
    pub wall_chance: f32,
    pub birth_limit: i32,
    pub death_limit: i32,
    pub cave_steps: i32,
}

impl Default for MapgenConfig {
    fn default() -> Self {
        let room_series = vec![
            // Temporarily stick to smaller rooms:
            //RoomStep { min_size: 30, max_size: 60, attempts: 10 },
            //RoomStep { min_size: 25, max_size: 50, attempts: 15 },
            //RoomStep { min_size: 20, max_size: 40, attempts: 20 },
            RoomStep { min_size: 15, max_size: 30, attempts: 25 },
            RoomStep { min_size: 10, max_size: 20, attempts: 30 },
        ];

        Self {
            size: Point(100, 100),
            room_series,
            wall_chance: 0.45,
            birth_limit: 5,
            death_limit: 4,
            cave_steps: 3,
        }
    }
}

fn count_neighbors(cave: &Matrix<char>, p: Point, v: char) -> i32 {
    let mut count = 0;
    for dx in -1..=1 {
        for dy in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            if cave.get(p + Point(dx, dy)) == v { count += 1; }
        }
    }
    count
}

fn build_cave(size: Point, config: &MapgenConfig, rng: &mut RNG) -> Matrix<char> {
    let Point(w, h) = size;
    let mut result = Matrix::new(size, ' ');

    // Initialize the interior with random values.
    for x in 1..(w - 1) {
        for y in 1..(h - 1) {
            let p = Point(x, y);
            let wall = rng.gen::<f32>() < config.wall_chance;
            result.set(p, if wall { ' ' } else { '.' });
        }
    }

    // Run cellular automaton steps.
    for _ in 0..config.cave_steps {
        let mut next = Matrix::new(size, ' ');
        for x in 1..(w - 1) {
            for y in 1..(h - 1) {
                let p = Point(x, y);
                let neighbors = count_neighbors(&result, p, ' ');
                let wall = if result.get(p) == ' ' {
                    neighbors >= config.death_limit
                } else {
                    neighbors >= config.birth_limit
                };
                next.set(p, if wall { ' ' } else { '.' });
            }
        }
        result = next;
    }

    // Mark cells adjacent to floor cells as walls.
    for x in 0..w {
        for y in 0..h {
            let p = Point(x, y);
            if result.get(p) != '.' { continue; }

            for dx in -1..=1 {
                for dy in -1..=1 {
                    let q = p + Point(dx, dy);
                    if result.get(q) == ' ' { result.set(q, '#'); }
                }
            }
        }
    }
    result
}

fn build_room_cave(size: Point, config: &MapgenConfig, rng: &mut RNG) -> Matrix<char> {
    loop {
        let result = build_cave(size, config, rng);
        if find_connected_components(&result, '.').len() == 1 { return result; }
    }
}

fn find_connected_components(map: &Matrix<char>, v: char) -> Vec<Vec<Point>> {
    let mut result = vec![];
    let mut visited = HashSet::default();

    for x in 0..map.size.0 {
        for y in 0..map.size.1 {
            let p = Point(x, y);
            if map.get(p) != v || visited.contains(&p) { continue; }

            let mut queue = vec![p];
            let mut component = vec![];
            while let Some(p) = queue.pop() {
                if !visited.insert(p) { continue; }

                for d in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let q = p + Point(d.0, d.1);
                    if map.get(q) == v && !visited.contains(&q) { queue.push(q); }
                }
                component.push(p);
            }
            result.push(component);
        }
    }
    result
}

fn try_place_room(map: &mut Matrix<char>, room: &Matrix<char>, rng: &mut RNG) -> bool {
    let mut map_walls = HashSet::default();
    for x in 0..map.size.0 {
        for y in 0..map.size.1 {
            let p = Point(x, y);
            if map.get(p) == '#' { map_walls.insert(p); }
        }
    }

    let mut room_walls = HashSet::default();
    let mut room_floor = HashSet::default();
    for x in 0..room.size.0 {
        for y in 0..room.size.1 {
            let p = Point(x, y);
            let tile = room.get(p);
            if tile == '#' { room_walls.insert(p); }
            if tile == '.' { room_floor.insert(p); }
        }
    }

    // Find all offsets at which at least one room wall and map wall align.
    let mut offsets = HashSet::default();
    let Point(lx, ly) = map.size - room.size;
    for &mw in &map_walls {
        for &rw in &room_walls {
            let p = mw - rw;
            if 0 <= p.0 && p.0 < lx && 0 <= p.1 && p.1 < ly { offsets.insert(p); }
        }
    }
    if offsets.is_empty() { return false; }


    // Permute the distinct offsets.
    let mut offsets: Vec<_> = offsets.into_iter().collect();
    offsets.shuffle(rng);
    let offsets = offsets;

    // Try each offset in turn, taking the first valid one. An offset is valid
    // iff all defined cells (with value != ' ') in the map and room match up.
    for &offset in &offsets {
        if room_walls.iter().any(|&x| map.get(x + offset) == '.') ||
           room_floor.iter().any(|&x| map.get(x + offset) == '#') {
            continue;
        }

        for x in 0..room.size.0 {
            for y in 0..room.size.1 {
                let p = Point(x, y);
                let c = room.get(p);
                if c != ' ' { map.set(p + offset, c); }
            }
        }
        return true;
    }
    false
}

pub fn mapgen(config: &MapgenConfig, rng: &mut RNG) -> Matrix<char> {
    let mut map = Matrix::new(config.size, ' ');

    // Place the first room in the center.
    let rc = &config.room_series[0];
    let (min, max) = (rc.min_size, rc.max_size);
    let size = Point(rng.gen_range(min..=max), rng.gen_range(min..=max));
    let room = build_room_cave(size, config, rng);

    let dx = (config.size - size).0 / 2;
    let dy = (config.size - size).1 / 2;
    for rx in 0..size.0 {
        for ry in 0..size.1 {
            let p = Point(rx, ry);
            let c = room.get(p);
            if c != ' ' { map.set(p + Point(dx, dy), c); }
        }
    }

    // Try to place more rooms, iterating through each room series.
    for rc in &config.room_series {
        for _ in 0..rc.attempts {
            let (min, max) = (rc.min_size, rc.max_size);
            let size = Point(rng.gen_range(min..=max), rng.gen_range(min..=max));
            let room = build_room_cave(size, config, rng);
            try_place_room(&mut map, &room, rng);
        }
    }

    // Convert the remaining undefined cells to walls.
    for x in 0..config.size.0 {
        for y in 0..config.size.1 {
            let p = Point(x, y);
            if map.get(p) == ' ' { map.set(p, '#'); }
        }
    }
    map
}

//////////////////////////////////////////////////////////////////////////////

#[allow(soft_unstable)]
#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;

    use std::time::Instant;
    use test::bench::black_box;

    use rand::SeedableRng;

    const BASE_SEED: u64 = 17;
    const NUM_SEEDS: u64 = 8;

    #[test]
    #[ignore]
    fn bench_mapgen() {
        let iterations = 3 * NUM_SEEDS;
        let start = Instant::now();

        for i in 0..iterations {
            black_box({
                let seed = (i % NUM_SEEDS) + BASE_SEED;
                let config = MapgenConfig::default();
                let mut rng = RNG::seed_from_u64(seed);
                mapgen(&config, &mut rng);
            });
        }

        let duration = start.elapsed();
        println!("Avg per iteration: {:?}", duration / iterations as u32);
    }
}
