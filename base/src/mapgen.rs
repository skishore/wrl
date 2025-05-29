use std::cmp::{max, min};

use rand::Rng;
use rand::seq::SliceRandom;

use crate::base::{HashMap, HashSet, LOS, Matrix, Point, dirs};
use crate::base::{RNG, sample};

//////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct RoomStep {
    min_size: i32,
    max_size: i32,
    attempts: i32,
}

#[derive(Clone)]
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
            RoomStep { min_size: 30, max_size: 60, attempts: 10 },
            RoomStep { min_size: 25, max_size: 50, attempts: 15 },
            RoomStep { min_size: 20, max_size: 40, attempts: 20 },
            RoomStep { min_size: 15, max_size: 30, attempts: 25 },
            RoomStep { min_size: 10, max_size: 20, attempts: 30 },
        ];

        Self {
            size: Point(100, 100),
            room_series,
            wall_chance: 0.40,
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
        if find_cardinal_components(&result, '.').len() == 1 { return result; }
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

fn find_components(map: &Matrix<char>, v: char, dirs: &[Point]) -> Vec<Vec<Point>> {
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

                for &dir in dirs {
                    let q = p + dir;
                    if map.get(q) == v && !visited.contains(&q) { queue.push(q); }
                }
                component.push(p);
            }
            result.push(component);
        }
    }
    result
}

fn find_cardinal_components(map: &Matrix<char>, v: char) -> Vec<Vec<Point>> {
    find_components(map, v, &dirs::CARDINAL)
}

fn find_diagonal_components(map: &Matrix<char>, v: char) -> Vec<Vec<Point>> {
    find_components(map, v, &dirs::ALL)
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

struct DijkstraNode {
    point: Point,
    distance: f64,
}

impl Eq for DijkstraNode {}

impl PartialEq for DijkstraNode {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point && self.distance == other.distance
    }
}

impl std::cmp::Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let sd = f64_to_monotonic_u64(self.distance);
        let od = f64_to_monotonic_u64(other.distance);
        let (sp, op) = (self.point, other.point);
        od.cmp(&sd).then_with(|| (sp.0, sp.1).cmp(&(op.0, op.1)))
    }
}

impl std::cmp::PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn f64_to_monotonic_u64(x: f64) -> u64 {
    let bits = x.to_bits();
    let sign = 1u64 << 63;
    if bits & sign == 0 { bits | sign } else { !bits }
}

fn dijkstra<F: Fn(Point) -> Vec<Point>, G: Fn(Point) -> f64, H: Fn(Point) -> bool>
        (sources: &[Point], edges: F, score: G, target: H) -> Option<Vec<Point>> {

    let mut map = HashMap::default();
    let mut heap = std::collections::BinaryHeap::new();
    let sentinel = Point(std::i32::MAX, std::i32::MAX);

    // We assume sources are distinct. We can relax this assumption later.
    for &source in sources {
        let distance = score(source);
        if distance == f64::INFINITY { continue; }

        heap.push(DijkstraNode { point: source, distance });
        let existing = map.insert(source, (sentinel, distance, false));
        assert!(existing.is_none());
    }

    while let Some(node) = heap.pop() {
        let prev = node.point;
        assert!(node.distance != f64::INFINITY);
        let entry = map.get_mut(&prev).unwrap();
        if std::mem::replace(&mut entry.2, true) { continue; }

        if target(prev) {
            let mut result = vec![];
            let mut current = prev;
            while current != sentinel {
                result.push(current);
                current = map.get(&current).unwrap().0;
            }
            result.reverse();
            return Some(result);
        }

        for next in edges(prev) {
            let entry = map.get(&next);
            if let Some(x) = entry && x.2 { continue; }

            let distance = node.distance + score(next) + (next - prev).len_l2();
            if let Some(x) = entry && distance > x.1 { continue; }
            if distance == f64::INFINITY { continue; }

            heap.push(DijkstraNode { point: next, distance });
            map.insert(next, (prev, distance, false));
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
    let mut rooms = find_cardinal_components(&map, '.');
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
    let berry_blue_noise = generate_blue_noise(size, 24, rng);
    let trees_blue_noise = generate_blue_noise(size, 12, rng);

    // Build the lake...
    let lc = MapgenConfig { birth_limit: 6, ..config.clone() };
    let ls = Point(rng.gen_range(18..=36), rng.gen_range(12..=24));
    let lz = config.size - ls;
    let lx = ((0.50 + 0.25 * rng.gen::<f64>()) * lz.0 as f64).round() as i32;
    let ly = ((0.75 + 0.25 * rng.gen::<f64>()) * lz.1 as f64).round() as i32;
    let lake = (|| loop {
        let result = build_room_cave(ls, &lc, rng);
        if find_cardinal_components(&result, '#').len() == 1 { return result; }
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
    let l1 = std::cmp::max(2, (0.2 * (rooms.len() as f64)).round() as usize);
    let l2 = std::cmp::max(4, (0.4 * (rooms.len() as f64)).round() as usize);
    let can_plant_grass = ['.', 'S'];
    let can_plant_trees = ['.'];
    let mut has_thicket = false;
    let mut has_grove = false;

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
                has_grove = true;
                map.set(p, 'B');
            }
            grassiness = 0.0 + 0.1 * grassiness;
        } else if i < l2 {
            for &p in values.keys() {
                if !can_plant_trees.contains(&map.get(p)) { continue; }
                if trees_blue_noise.get(p) == 0.0 { continue; }
                has_thicket = true;
                map.set(p, '#');
            }
            grassiness = 0.2 + 0.1 * grassiness;
        } else {
            grassiness = 0.1 + 0.1 * grassiness;
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
    if !has_grove || !has_thicket { return None; }

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

    // Split the final map into components that are reachable by walking.
    let mapping: HashMap<_, _> =
        [('"', '.'), ('=', '.'), ('R', '.'), ('S', '.')].into_iter().collect();
    let mut copy = map.clone();
    for x in 0..size.0 {
        for y in 0..size.1 {
            let p = Point(x, y);
            if let Some(&c) = mapping.get(&copy.get(p)) { copy.set(p, c); }
        }
    }
    let mut components = find_diagonal_components(&copy, '.');
    if components.is_empty() { return None; }

    // Connect all other components to the biggest walkable component.
    components.sort_by_key(|x| x.len());
    let biggest: HashSet<_> = components.pop().unwrap().into_iter().collect();
    let target = |p: Point| { biggest.contains(&p) };
    for component in &components {
        let score = |p: Point| {
            let noise_score = 8.0 * (1.0 - noise.get(p));
            let tiles_score = *costs.get(&map.get(p)).unwrap_or(&0.0);
            noise_score + tiles_score
        };
        let edges = |p: Point| {
            let mut result: Vec<_> = (-1..=1).flat_map(
                |x| (-1..=1).map(move |y| p + Point(x, y))).collect();
            if map.get(p + Point(1, 0)) == 'W' && map.get(p + Point(2, 0)) == 'W' {
                result.push(p + Point(3, 0));
            }
            if map.get(p + Point(-1, 0)) == 'W' && map.get(p + Point(-2, 0)) == 'W' {
                result.push(p + Point(-3, 0));
            }
            result
        };
        let path = dijkstra(component, edges, score, target)?;
        let mut prev: Option<Point> = None;
        for &p in &path {
            if let Some(q) = prev && (p.0 - q.0).abs() > 1 {
                let (a, b) = (std::cmp::min(p.0, q.0), std::cmp::max(p.0, q.0));
                for x in (a + 1)..b { map.set(Point(x, p.1), '='); }
            }
            if copy.get(p) != '.' { map.set(p, '.'); }
            prev = Some(p);
        }
    }

    // Convert any undefined cells to walls, plus other similar replacements.
    let mapping: HashMap<_, _> =
        [(' ', '#'), ('S', '.'), ('W', '~')].into_iter().collect();
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

// Legacy code

pub fn legacy_mapgen(rng: &mut RNG) -> Matrix<char> {
    let config = MapgenConfig::default();
    legacy_mapgen_with_size(config.size, rng)
}

pub fn legacy_mapgen_with_size(size: Point, rng: &mut RNG) -> Matrix<char> {
    let mut map = Matrix::new(size, '.');

    let automata = |rng: &mut RNG, init: u32| -> Matrix<bool> {
        let mut result = Matrix::new(size, false);
        for x in 0..size.0 {
            result.set(Point(x, 0), true);
            result.set(Point(x, size.1 - 1), true);
        }
        for y in 0..size.1 {
            result.set(Point(0, y), true);
            result.set(Point(size.0 - 1, y), true);
        }

        for y in 0..size.1 {
            for x in 0..size.0 {
                let block = rng.gen_range(0..100) < init;
                if block { result.set(Point(x, y),  true); }
            }
        }

        for i in 0..3 {
            let mut next = result.clone();
            for y in 1..size.1 - 1 {
                for x in 1..size.0 - 1 {
                    let point = Point(x, y);
                    let (mut adj1, mut adj2) = (0, 0);
                    for dy in -2_i32..=2 {
                        for dx in -2_i32..=2 {
                            if dx == 0 && dy == 0 { continue; }
                            if min(dx.abs(), dy.abs()) == 2 { continue; }
                            let next = point + Point(dx, dy);
                            if !result.get(next) { continue; }
                            let distance = max(dx.abs(), dy.abs());
                            if distance <= 1 { adj1 += 1; }
                            if distance <= 2 { adj2 += 1; }
                        }
                    }
                    let blocked = adj1 >= 5 || (i < 2 && adj2 <= 1);
                    next.set(point, blocked);
                }
            }
            std::mem::swap(&mut result, &mut next);
        }
        result
    };

    let walls = automata(rng, 45);
    let grass = automata(rng, 45);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let point = Point(x, y);
            if walls.get(point) {
                map.set(point, '#');
            } else if grass.get(point) {
                map.set(point, '"');
            }
        }
    }

    let mut river = vec![Point::default()];
    for i in 1..size.1 {
        let last = river.last().unwrap().0;
        let next = last + rng.gen_range(-1..=1);
        river.push(Point(next, i));
    }
    let target = river[0] + *river.last().unwrap();
    let offset = Point((size - target).0 / 2, 0);
    for &x in &river { map.set(x + offset, '~'); }

    let mut free_point = |map: &Matrix<char>| {
        for _ in 0..100 {
            let p = Point(rng.gen_range(0..size.0), rng.gen_range(0..size.1));
            let c = map.get(p);
            if c == '.' || c == '"' { return Some(p); }
        }
        None
    };
    for _ in 0..5 {
        if let Some(p) = free_point(&map) { map.set(p, 'B'); }
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
                let mut rng = RNG::seed_from_u64(seed);
                mapgen(&mut rng);
            });
        }

        let duration = start.elapsed();
        println!("Avg per iteration: {:?}", duration / iterations as u32);
    }
}
