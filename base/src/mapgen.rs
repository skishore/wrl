use crate::base::{HashMap, HashSet, LOS, Matrix, Point};
use crate::base::{RNG, sample};

use rand::Rng;
use rand::seq::SliceRandom;

//////////////////////////////////////////////////////////////////////////////

struct RoomStep {
    min_size: i32,
    max_size: i32,
    attempts: i32,
}

struct MapgenConfig {
    // Overall structure of the map.
    size: Point,
    room_series: Vec<RoomStep>,

    // Cellular automaton params. Move to RoomStep?
    wall_chance: f64,
    birth_limit: i32,
    death_limit: i32,
    cave_steps: i32,

    // Connections between each room.
    corridor_width: i32,
    corridor_limit: f64,
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
            corridor_width: 2,
            corridor_limit: 4.0,
        }
    }
}

impl MapgenConfig {
    fn with_size(size: Point) -> Self {
        let mut result = Self::default();
        result.size = size;
        result
    }
}

//////////////////////////////////////////////////////////////////////////////

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
            let wall = rng.gen::<f64>() < config.wall_chance;
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

//////////////////////////////////////////////////////////////////////////////

fn find_closest_pairs(r1: &[Point], r2: &[Point]) -> Vec<(Point, Point)> {
    let mut result = vec![];
    let mut best_score = std::i64::MAX;

    for &p1 in r1 {
        for &p2 in r2 {
            let score = (p1 - p2).len_l2_squared();
            if score > best_score { continue; }
            if score < best_score { result.clear(); }
            result.push((p1, p2));
            best_score = score;
        }
    }
    result
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

//////////////////////////////////////////////////////////////////////////////

fn generate_blue_noise(size: Point, spacing: i32, rng: &mut RNG) -> Matrix<f64> {
    let mut noise = Matrix::new(size, 0.0);
    let mut points: Vec<_> = (0..size.0).flat_map(
        |x| (0..size.1).map(move |y| Point(x, y))).collect();
    points.shuffle(rng);

    let d = ((spacing / 2) as f64).sqrt().ceil() as i32;

    for point in points {
        let okay = (|| {
            for dx in -d..=d {
                for dy in -d..=d {
                    if dx * dx + dy * dy >= spacing { continue; }
                    if noise.get(point + Point(dx, dy)) == 0.0 { continue; }
                    return false;
                }
            }
            true
        })();
        if okay { noise.set(point, 1.0); }
    }
    noise
}

fn generate_perlin_noise(
        size: Point, scale: f64, octaves: i32, falloff: f64, rng: &mut RNG) -> Matrix<f64> {
    fn interpolate(a0: f64, a1: f64, w: f64) -> f64 {
        // Smoothstep interpolation
        (a1 - a0) * (3.0 - w * 2.0) * w * w + a0
    }

    let mut noise = Matrix::new(size, 0.0);

    for octave in 0..octaves {
        let period = scale / (2.0 as f64).powi(octave);
        let frequency = 1.0 / period;

        // Generate a grid of random values for this octave.
        let gw = (size.0 as f64 * frequency).floor() as i32 + 2;
        let gh = (size.1 as f64 * frequency).floor() as i32 + 2;
        let mut grid = Matrix::new(Point(gw, gh), 0.0);
        for x in 0..gw {
            for y in 0..gh {
                grid.set(Point(x, y), rng.gen::<f64>());
            }
        }

        for y in 0..size.1 {
            let y0 = (y as f64 * frequency).floor() as i32;
            let y1 = y0 + 1;
            let yfrac = (y as f64 * frequency) - y0 as f64;

            for x in 0..size.0 {
                let x0 = (x as f64 * frequency).floor() as i32;
                let x1 = x0 + 1;
                let xfrac = (x as f64 * frequency) - x0 as f64;

                let v00 = grid.get(Point(x0, y0));
                let v10 = grid.get(Point(x1, y0));
                let v01 = grid.get(Point(x0, y1));
                let v11 = grid.get(Point(x1, y1));

                let x_interp1 = interpolate(v00, v10, xfrac);
                let x_interp2 = interpolate(v01, v11, xfrac);
                let value = interpolate(x_interp1, x_interp2, yfrac);

                let p = Point(x, y);
                noise.set(p, noise.get(p) + value * falloff.powi(octave));
            }
        }
    }

    // Normalize the output to [0, 1].
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for x in 0..size.0 {
        for y in 0..size.1 {
            let v = noise.get(Point(x, y));
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
    }

    let range = max_val - min_val;
    for x in 0..size.0 {
        for y in 0..size.1 {
            let p = Point(x, y);
            noise.set(p, (noise.get(p) - min_val) / range);
        }
    }
    noise
}

//////////////////////////////////////////////////////////////////////////////

fn f64_to_monotonic_u64(x: f64) -> u64 {
    let bits = x.to_bits();
    let sign = 1u64 << 63;
    if bits & sign == 0 { bits | sign } else { !bits }
}

fn dijkstra<F: Fn(Point) -> Vec<Point>, G: Fn(Point) -> f64, H: Fn(Point) -> bool>
        (sources: &[Point], edges: F, score: G, target: H) -> Option<Vec<Point>> {

    let mut complete = HashMap::default();
    let mut frontier = HashMap::default();

    for &source in sources {
        let distance = score(source);
        if distance != f64::INFINITY { frontier.insert(source, (distance, None)); }
    }

    while !frontier.is_empty() {
        let (&prev, &value) = frontier.iter().min_by_key(
            |&x| f64_to_monotonic_u64(x.1.0)).unwrap();
        assert!(value.0 != f64::INFINITY);
        let existing = complete.insert(prev, value);
        assert!(existing.is_none());
        frontier.remove(&prev);

        if target(prev) {
            let mut result = vec![];
            let mut current = Some(prev);
            while let Some(x) = current {
                result.push(x);
                current = complete.get(&x).unwrap().1;
            }
            result.reverse();
            return Some(result);
        }

        for next in edges(prev) {
            if complete.contains_key(&next) { continue; }

            let distance = value.0 + score(next) + (next - prev).len_l2();
            if distance == f64::INFINITY { continue; }

            let v = (distance, Some(prev));
            frontier.entry(next).and_modify(|x| { if distance < x.0 { *x = v; } }).or_insert(v);
        }
    }
    None
}

//////////////////////////////////////////////////////////////////////////////

fn mapgen_attempt(config: &MapgenConfig, rng: &mut RNG) -> Option<Matrix<char>> {
    let size = config.size;
    let mut map = Matrix::new(size, ' ');

    // Place the first room in the center.
    let rc = &config.room_series[0];
    let (min, max) = (rc.min_size, rc.max_size);
    let dims = Point(rng.gen_range(min..=max), rng.gen_range(min..=max));
    let room = build_room_cave(dims, config, rng);

    let dx = (size - dims).0 / 2;
    let dy = (size - dims).1 / 2;
    for rx in 0..dims.0 {
        for ry in 0..dims.1 {
            let p = Point(rx, ry);
            let c = room.get(p);
            if c != ' ' { map.set(p + Point(dx, dy), c); }
        }
    }

    // Try to place more rooms, iterating through each room series.
    for rc in &config.room_series {
        for _ in 0..rc.attempts {
            let (min, max) = (rc.min_size, rc.max_size);
            let dims = Point(rng.gen_range(min..=max), rng.gen_range(min..=max));
            let room = build_room_cave(dims, config, rng);
            try_place_room(&mut map, &room, rng);
        }
    }
    let mut rooms = find_connected_components(&map, '.');
    rooms.shuffle(rng);

    // Connect up the rooms. We could build a spanning tree here, but instead
    // we just connect all rooms that are sufficiently close. Doing so forms a
    // connected graph since try_place_room puts new rooms near existing ones.
    for (i, r1) in rooms.iter().enumerate() {
        for r2 in rooms.iter().skip(i) {
            let (p1, p2) = *sample(&find_closest_pairs(&r1, &r2), rng);
            let distance = (p1 - p2).len_l2();
            if distance > config.corridor_limit { continue; }

            let l = -config.corridor_width / 2;
            let r = l + config.corridor_width;
            for p in LOS(p1, p2) {
                map.set(p, '.');
                for dx in l..r {
                    for dy in l..r {
                        map.set(p + Point(dx, dy), '.');
                    }
                }
            }
        }
    }

    // Noises used to guide feature placement.
    let noise = generate_perlin_noise(size, 4.0, 2, 0.65, rng);
    let berry_blue_noise = generate_blue_noise(size, 10, rng);
    let trees_blue_noise = generate_blue_noise(size, 5, rng);

    // Build the lake...
    let ls = Point(rng.gen_range(18..=36), rng.gen_range(12..=24));
    let lz = config.size - ls;
    let lx = ((0.50 + 0.25 * rng.gen::<f64>()) * lz.0 as f64).round() as i32;
    let ly = ((0.75 + 0.25 * rng.gen::<f64>()) * lz.1 as f64).round() as i32;
    let lake = (|| loop {
        let result = build_room_cave(ls, config, rng);
        if find_connected_components(&result, '#').len() == 1 { return result; }
    })();

    // Then, place the lake.
    let mapping: HashMap<_, _> = [('#', 'S'), ('.', '~')].into_iter().collect();
    for x in 0..ls.0 {
        for y in 0..ls.1 {
            let Some(&c) = mapping.get(&lake.get(Point(x, y))) else { continue };
            map.set(Point(x + lx, y + ly), c);
        }
    }

    // Set up Dijkstra parameters for the river...
    let width = 2;
    let costs: HashMap<_, _> = [('#', 64.0), (' ', 64.0)].into_iter().collect();
    let score_one = |p: Point, center: bool| {
        if !map.contains(p) {
            return if center { f64::INFINITY } else { *costs.get(&' ').unwrap() };
        }
        let noise_score = 8.0 * (1.0 - noise.get(p));
        let tiles_score = *costs.get(&map.get(p)).unwrap_or(&0.0);
        noise_score + tiles_score
    };
    let score = |p: Point| {
        let mut result = 0.0;
        for dx in -1..=width {
            for dy in -1..=width {
                let center = 0 <= dx && dx < width && 0 <= dy && dy < width;
                result += score_one(p + Point(dx, dy), center);
            }
        }
        result
    };
    let edges = |p: Point| {
        (-1..=1).flat_map(|x| (-1..=1).map(move |y| p + Point(x, y))).collect()
    };
    let target = |p: Point| { map.get(p) == '~' };
    let sources: Vec<_> = (0..size.0).map(|x| Point(x, 0)).collect();

    // Then, build the river.
    let path = dijkstra(&sources, edges, score, target)?;
    for &p in path.iter().take(path.len() - 1) {
        for x in 0..width {
            for y in 0..width {
                map.set(p + Point(x, y), 'W');
            }
        }
    }

    // Plant grass and other features in each room.
    let l1 = (0.2 * (rooms.len() as f64)).round() as usize;
    let l2 = (0.4 * (rooms.len() as f64)).round() as usize;
    let can_plant_grass = ['.', 'S'];
    let can_plant_trees = ['.'];

    for (i, room) in rooms.iter().enumerate() {
        let mut values = HashMap::default();
        for &p in room {
            if can_plant_grass.contains(&map.get(p)) {
                values.insert(p, noise.get(p) + 0.3 * rng.gen::<f64>());
            }
        }
        if values.is_empty() { continue; }

        let mut grassiness = rng.gen::<f64>();
        if i < l1 {
            for &p in values.keys() {
                if !can_plant_trees.contains(&map.get(p)) { continue; }
                if berry_blue_noise.get(p) == 0.0 { continue; }
                map.set(p, 'B');
            }
            grassiness = 0.0 + 0.2 * grassiness;
        } else if i < l2 {
            for &p in values.keys() {
                if !can_plant_trees.contains(&map.get(p)) { continue; }
                if trees_blue_noise.get(p) == 0.0 { continue; }
                map.set(p, '#');
            }
            grassiness = 0.0 + 0.6 * grassiness;
        } else {
            grassiness = 0.2 + 0.2 * grassiness;
        };

        let target = (grassiness * values.len() as f64).round() as usize;
        let mut values: Vec<_> = values.into_iter().collect();
        values.sort_by_key(|&x| (f64_to_monotonic_u64(x.1), x.0.0, x.0.1));

        for (i, &(p, _)) in values.iter().enumerate() {
            if i >= target { break; }
            if !can_plant_grass.contains(&map.get(p)) { continue; }
            map.set(p, '"');
        }
    }

    // Set up Dijkstra parameters for the route...
    let costs: HashMap<_, _> = [
        ('~', f64::INFINITY),
        ('#', 64.0),
        (' ', 64.0),
        ('B', 64.0),
        ('"', 16.0),
        ('W', 16.0),
        (',', 4.0),
        ('S', 4.0),
    ].into_iter().collect();
    let score_one = |p: Point, center: bool| {
        if !map.contains(p) {
            return if center { f64::INFINITY } else { *costs.get(&' ').unwrap() };
        }
        let tile = map.get(p);
        if tile == 'W' && center { return f64::INFINITY; }

        let noise_score = 8.0 * (1.0 - noise.get(p));
        let tiles_score = *costs.get(&tile).unwrap_or(&0.0);
        noise_score + tiles_score
    };
    let score = |p: Point| {
        let mut result = 0.0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                let center = dx == 0 && dy == 0;
                result += score_one(p + Point(dx, dy), center);
            }
        }
        result
    };
    let edges = |p: Point| {
        let mut result: Vec<_> = (-1..=1).flat_map(
            |x| (-1..=1).map(move |y| p + Point(x, y))).collect();
        if map.get(p + Point(1, 0)) == 'W' && map.get(p + Point(2, 0)) == 'W' {
            result.push(p + Point(3, 0));
        }
        result
    };
    let target = |p: Point| { p.0 + 1 == size.0 };
    let sources: Vec<_> = (0..size.1).map(|y| Point(0, y)).collect();

    // Then, build the route.
    let path = dijkstra(&sources, edges, score, target)?;
    let mut prev: Option<Point> = None;
    for &p in &path {
        if let Some(q) = prev && p.0 - q.0 > 1 {
            for x in (q.0 + 1)..p.0 { map.set(Point(x, p.1), '='); }
        }
        map.set(p, 'R');
        prev = Some(p);
    }

    // Convert any undefined cells to walls, plus other similar replacements.
    let mapping: HashMap<_, _> =
        [(' ', '#'), ('B', '*'), ('S', '.'), ('W', '~')].into_iter().collect();
    for x in 0..size.0 {
        for y in 0..size.1 {
            let p = Point(x, y);
            if let Some(&c) = mapping.get(&map.get(p)) { map.set(p, c); }
        }
    }
    Some(map)
}

pub fn mapgen(rng: &mut RNG) -> Matrix<char> {
    let config = MapgenConfig::default();
    loop { if let Some(x) = mapgen_attempt(&config, rng) { return x; } }
}

pub fn mapgen_with_size(size: Point, rng: &mut RNG) -> Matrix<char> {
    let config = MapgenConfig::with_size(size);
    loop { if let Some(x) = mapgen_attempt(&config, rng) { return x; } }
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
                let mut rng = RNG::seed_from_u64(seed);
                mapgen(&mut rng);
            });
        }

        let duration = start.elapsed();
        println!("Avg per iteration: {:?}", duration / iterations as u32);
    }
}
