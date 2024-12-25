use std::cmp::{max, min};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::mem::{replace, swap};

use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::static_assert_size;
use crate::base::{Buffer, Color, Glyph, Rect, Slice};
use crate::base::{HashMap, HashSet, LOS, Matrix, Point, dirs};
use crate::base::{sample, weighted, RNG};
use crate::effect::{Effect, Event, Frame, FT, self};
use crate::entity::{EID, Entity, EntityArgs, EntityMap};
use crate::knowledge::{EntityKnowledge, Knowledge, Timestamp, Vision, VisionArgs};
use crate::pathing::{AStar, AStarLength, BFS, BFSResult, Status};
use crate::pathing::{DijkstraSearch, FastDijkstraLength, FastDijkstraMap};

//////////////////////////////////////////////////////////////////////////////

// Constants

const ASTAR_LIMIT_ATTACK: i32 = 32;
const ASTAR_LIMIT_SEARCH: i32 = 256;
const ASTAR_LIMIT_WANDER: i32 = 1024;
const BFS_LIMIT_ATTACK: i32 = 8;
const BFS_LIMIT_WANDER: i32 = 64;
const FLIGHT_MAP_LIMIT: i32 = 256;

const ASSESS_ANGLE: f64 = TAU / 18.;
const ASSESS_STEPS: i32 = 4;
const ASSESS_TURNS_EXPLORE: i32 = 8;
const ASSESS_TURNS_FLIGHT: i32 = 1;

const ATTACK_DAMAGE: i32 = 40;
const ATTACK_RANGE: i32 = 8;

const MAX_ASSESS: i32 = 32;
const MAX_HUNGER: i32 = 1024;
const MAX_THIRST: i32 = 256;

const MIN_RESTED: i32 = 2048;
const MAX_RESTED: i32 = 4096;

const MIN_FLIGHT_TURNS: i32 = 16;
const MAX_FLIGHT_TURNS: i32 = 64;
const MIN_SEARCH_TURNS: i32 = 16;
const MAX_SEARCH_TURNS: i32 = 32;
const TURN_TIMES_LIMIT: usize = 64;

const MOVE_TIMER: i32 = 960;
const TURN_TIMER: i32 = 120;
const SLOWED_TURNS: f64 = 2.;
const WANDER_TURNS: f64 = 2.;

const FOV_RADIUS_NPC: i32 = 12;
const FOV_RADIUS_PC_: i32 = 21;

const SPEED_PC_: f64 = 0.1;
const SPEED_NPC: f64 = 0.1;

const LIGHT: Light = Light::Sun(Point(0, 0));
const WEATHER: Weather = Weather::None;
const WORLD_SIZE: i32 = 50;

const UI_DAMAGE_FLASH: i32 = 6;
const UI_DAMAGE_TICKS: i32 = 6;

const UI_COLOR: i32 = 0x430;
const UI_MAP_SIZE_X: i32 = WORLD_SIZE;
const UI_MAP_SIZE_Y: i32 = WORLD_SIZE;

const UI_MOVE_ALPHA: f64 = 0.75;
const UI_MOVE_FRAMES: i32 = 12;

#[derive(Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char) }

//////////////////////////////////////////////////////////////////////////////

// Tile

const FLAG_BLOCKS_VISION: u32 = 1 << 0;
const FLAG_LIMITS_VISION: u32 = 1 << 1;
const FLAG_BLOCKS_MOVEMENT: u32 = 1 << 2;

const FLAGS_NONE: u32 = 0;
const FLAGS_BLOCKED: u32 = FLAG_BLOCKS_MOVEMENT | FLAG_BLOCKS_VISION;
const FLAGS_PARTLY_BLOCKED: u32 = FLAG_BLOCKS_MOVEMENT | FLAG_LIMITS_VISION;

pub struct Tile {
    pub flags: u32,
    pub glyph: Glyph,
    pub description: &'static str,
}
static_assert_size!(Tile, 32);

impl Tile {
    fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    pub fn casts_shadow(&self) -> bool { self.flags & FLAG_BLOCKS_MOVEMENT != 0 }
    pub fn blocks_vision(&self) -> bool { self.flags & FLAG_BLOCKS_VISION != 0 }
    pub fn limits_vision(&self) -> bool { self.flags & FLAG_LIMITS_VISION != 0 }
    pub fn blocks_movement(&self) -> bool { self.flags & FLAG_BLOCKS_MOVEMENT != 0 }
}

impl PartialEq for &'static Tile {
    fn eq(&self, next: &&'static Tile) -> bool {
        *self as *const Tile == *next as *const Tile
    }
}

impl Eq for &'static Tile {}

lazy_static! {
    static ref TILES: HashMap<char, Tile> = {
        let items = [
            ('.', (FLAGS_NONE,           Glyph::wdfg('.', 0x222), "grass")),
            ('"', (FLAG_LIMITS_VISION,   Glyph::wdfg('"', 0x120), "tall grass")),
            ('#', (FLAGS_BLOCKED,        Glyph::wdfg('#', 0x010), "a tree")),
            ('%', (FLAGS_NONE,           Glyph::wdfg('%', 0x200), "flowers")),
            ('~', (FLAGS_NONE,           Glyph::wdfg('~', 0x013), "water")),
        ];
        let mut result = HashMap::default();
        for (ch, (flags, glyph, description)) in items {
            result.insert(ch, Tile { flags, glyph, description });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Board

#[derive(Clone, Copy)]
pub struct Cell {
    pub eid: Option<EID>,
    pub shadow: i32,
    pub tile: &'static Tile,
}
static_assert_size!(Cell, 24);

pub enum Light { None, Sun(Point) }

enum Weather { None, Rain(Point, usize) }

struct Drop {
    frame: usize,
    point: Point,
}

struct Rain {
    ch: char,
    diff: Point,
    drops: VecDeque<Drop>,
    path: Vec<Point>,
    lightning: i32,
    thunder: i32,
}

pub struct Board {
    active_entity_index: usize,
    entity_order: Vec<EID>,
    entities: EntityMap,
    map: Matrix<Cell>,
    // Animation
    _effect: Effect,
    // Knowledge state
    known: Option<Box<Knowledge>>,
    npc_vision: Vision,
    _pc_vision: Vision,
    // Environmental effects
    light: Light,
    rain: Option<Rain>,
    shadow: Vec<Point>,
}

impl Board {
    fn new(size: Point, light: Light, weather: Weather) -> Self {
        let shadow = match light {
            Light::Sun(x) => LOS(Point::default(), x).into_iter().skip(1).collect(),
            Light::None => vec![],
        };
        let rain = match weather {
            Weather::Rain(x, y) => Some(Rain {
                ch: Glyph::ray(x),
                diff: x,
                drops: VecDeque::with_capacity(y),
                path: LOS(Point::default(), x),
                lightning: -1,
                thunder: 0,
            }),
            Weather::None => None,
        };
        let cell = Cell { eid: None, shadow: 0, tile: Tile::get('#') };

        let mut result = Self {
            active_entity_index: 0,
            entity_order: vec![],
            entities: EntityMap::default(),
            map: Matrix::new(size, cell),
            // Animation
            _effect: Effect::default(),
            // Knowledge state
            known: Some(Box::default()),
            npc_vision: Vision::new(FOV_RADIUS_NPC),
            _pc_vision: Vision::new(FOV_RADIUS_PC_),
            // Environmental effects
            light,
            rain,
            shadow,
        };
        result.update_edge_shadows();
        result
    }

    // Animation

    fn add_effect(&mut self, effect: Effect, rng: &mut RNG) {
        let mut existing = Effect::default();
        std::mem::swap(&mut self._effect, &mut existing);
        self._effect = existing.and(effect);
        self._execute_effect_callbacks(rng);
    }

    fn advance_effect(&mut self, rng: &mut RNG) -> bool {
        if self._effect.frames.is_empty() {
            assert!(self._effect.events.is_empty());
            return false;
        }
        self._effect.frames.remove(0);
        self._effect.events.iter_mut().for_each(|x| x.update_frame(|y| y - 1));
        self._execute_effect_callbacks(rng);
        true
    }

    fn _execute_effect_callbacks(&mut self, rng: &mut RNG) {
        while self._execute_one_effect_callback(rng) {}
    }

    fn _execute_one_effect_callback(&mut self, rng: &mut RNG) -> bool {
        if self._effect.events.is_empty() { return false; }
        let event = &self._effect.events[0];
        if !self._effect.frames.is_empty() && event.frame() > 0 { return false; }
        match self._effect.events.remove(0) {
            Event::Callback { callback, .. } => callback(self, rng),
            Event::Other { .. } => (),
        }
        true
    }

    // Getters

    pub fn get_active_entity(&self) -> EID {
        self.entity_order[self.active_entity_index]
    }

    pub fn get_cell(&self, p: Point) -> &Cell { self.map.entry_ref(p) }

    pub fn get_entity(&self, eid: EID) -> Option<&Entity> { self.entities.get(eid) }

    pub fn get_frame(&self) -> Option<&Frame> { self._effect.frames.iter().next() }

    pub fn get_light(&self) -> &Light { &self.light }

    pub fn get_size(&self) -> Point { self.map.size }

    pub fn get_status(&self, p: Point) -> Status {
        let Cell { eid, tile, .. } = self.get_cell(p);
        if eid.is_some() { return Status::Occupied; }
        if tile.blocks_movement() { Status::Blocked } else { Status::Free }
    }

    pub fn get_tile(&self, p: Point) -> &'static Tile { self.get_cell(p).tile }

    // Setters

    fn add_entity(&mut self, args: &EntityArgs, rng: &mut RNG) -> EID {
        let pos = args.pos;
        let eid = self.entities.add(args, rng);
        let cell = self.map.entry_mut(pos).unwrap();
        let prev = replace(&mut cell.eid, Some(eid));
        assert!(prev.is_none());
        self.entity_order.push(eid);
        self.update_known(eid);
        eid
    }

    fn advance_entity(&mut self) {
        let eid = self.get_active_entity();
        charge(&mut self.entities[eid]);
        self.active_entity_index += 1;
        if self.active_entity_index == self.entity_order.len() {
            self.active_entity_index = 0;
        }
    }

    fn move_entity(&mut self, eid: EID, target: Point) {
        let entity = &mut self.entities[eid];
        let source = replace(&mut entity.pos, target);
        let old = replace(&mut self.map.entry_mut(source).unwrap().eid, None);
        assert!(old == Some(eid));
        let new = replace(&mut self.map.entry_mut(target).unwrap().eid, old);
        assert!(new.is_none());
    }

    fn remove_entity(&mut self, eid: EID) {
        // The player entity is not removed, since it's the player's POV.
        let entity = &mut self.entities[eid];
        entity.cur_hp = 0;
        if entity.player { return; }

        // Remove the entity from the map.
        let existing = self.map.entry_mut(entity.pos).unwrap().eid.take();
        assert!(existing == Some(eid));

        // Remove the entity from the entity_order list.
        let index = self.entity_order.iter().position(|&x| x == eid).unwrap();
        self.entity_order.remove(index);
        if self.active_entity_index > index {
            self.active_entity_index -= 1;
        } else if self.active_entity_index == self.entity_order.len() {
            self.active_entity_index = 0;
        }
    }

    fn reset(&mut self, tile: &'static Tile) {
        self.map.fill(Cell { eid: None, shadow: 0, tile });
        self.update_edge_shadows();
        self.entity_order.clear();
        self.active_entity_index = 0;
    }

    fn set_tile(&mut self, point: Point, tile: &'static Tile) {
        let Some(cell) = self.map.entry_mut(point) else { return; };
        let old_shadow = if cell.tile.casts_shadow() { 1 } else { 0 };
        let new_shadow = if tile.casts_shadow() { 1 } else { 0 };
        cell.tile = tile;
        self.update_shadow(point, new_shadow - old_shadow);
    }

    // Knowledge

    fn update_known(&mut self, eid: EID) {
        let mut known = self.known.take().unwrap_or_default();
        swap(&mut known, &mut self.entities[eid].known);

        let me = &self.entities[eid];
        let player = me.player;
        let args = VisionArgs { player, pos: me.pos, dir: me.dir };
        let vision = if player { &mut self._pc_vision } else { &mut self.npc_vision };
        vision.compute(&args, |x| self.map.get(x).tile);
        let vision = if player { &self._pc_vision } else { &self.npc_vision };
        known.update(me, &self, vision);

        swap(&mut known, &mut self.entities[eid].known);
        self.known = Some(known);
    }

    fn update_known_entity(&mut self, eid: EID, oid: EID, heard: bool) {
        let mut known = self.known.take().unwrap_or_default();
        swap(&mut known, &mut self.entities[eid].known);

        let me = &self.entities[eid];
        let other = &self.entities[oid];
        known.update_entity(me, other, self, /*seen=*/false, /*heard=*/heard);

        swap(&mut known, &mut self.entities[eid].known);
        self.known = Some(known);
    }

    // Environmental effects

    fn update_env(&mut self, frame: usize, pos: Point, rng: &mut RNG) {
        let Some(rain) = &mut self.rain else { return; };

        while let Some(x) = rain.drops.front() && x.frame < frame {
            rain.drops.pop_front();
        }
        let total = rain.drops.capacity();
        let denom = max(rain.diff.1, 1);
        let delta = max(rain.diff.1, 1) as usize;
        let extra = (frame + 1) * total / delta - (frame * total) / delta;
        for _ in 0..min(extra, total - rain.drops.len()) {
            let x = rng.gen::<i32>().rem_euclid(denom);
            let y = rng.gen::<i32>().rem_euclid(denom);
            let target_frame = frame + rain.path.len() - 1;
            let target_point = Point(x - denom / 2, y - denom / 2) + pos;
            rain.drops.push_back(Drop { frame: target_frame, point: target_point });
        }

        assert!(rain.lightning >= -1);
        if rain.lightning == -1 {
            if rng.gen::<f32>() < 0.002 { rain.lightning = 10; }
        } else if rain.lightning > 0 {
            rain.lightning -= 1;
        }

        assert!(rain.thunder >= 0);
        if rain.thunder == 0 {
            if rain.lightning == 0 && rng.gen::<f32>() < 0.02 { rain.thunder = 16; }
        } else {
            rain.thunder -= 1;
            if rain.thunder == 0 { rain.lightning = -1; }
        }
    }

    fn update_shadow(&mut self, point: Point, delta: i32) {
        if delta == 0 { return; }
        for &x in &self.shadow {
            let Some(cell) = self.map.entry_mut(point + x) else { continue; };
            cell.shadow += delta;
            assert!(cell.shadow >= 0);
        }
    }

    fn update_edge_shadows(&mut self) {
        let delta = if self.map.default.tile.casts_shadow() { 1 } else { 0 };
        if delta == 0 || self.shadow.is_empty() { return; }

        for x in -1..(self.map.size.0 + 1) {
            self.update_shadow(Point(x, -1), delta);
            self.update_shadow(Point(x, self.map.size.1), delta);
        }
        for y in 0..self.map.size.1 {
            self.update_shadow(Point(-1, y), delta);
            self.update_shadow(Point(self.map.size.0, y), delta);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Map generation

fn mapgen(board: &mut Board, rng: &mut RNG) {
    let ft = Tile::get('.');
    let wt = Tile::get('#');
    let gt = Tile::get('"');
    let fl = Tile::get('%');
    let wa = Tile::get('~');

    board.reset(ft);
    let size = board.get_size();

    let automata = |rng: &mut RNG, init: u32| -> Matrix<bool> {
        let mut d100 = || rng.gen::<u32>() % 100;
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
                if d100() < init { result.set(Point(x, y),  true); }
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
                board.set_tile(point, wt);
            } else if grass.get(point) {
                board.set_tile(point, gt);
            }
        }
    }

    let die = |n: i32, rng: &mut RNG| rng.gen::<i32>().rem_euclid(n);
    let mut river = vec![Point::default()];
    for i in 1..size.1 {
        let last = river.last().unwrap().0;
        let next = last + die(3, rng) - 1;
        river.push(Point(next, i));
    }
    let target = river[0] + *river.last().unwrap();
    let offset = Point((size - target).0 / 2, 0);
    for &x in &river { board.set_tile(x + offset, wa); }

    let pos = |board: &Board, rng: &mut RNG| {
        for _ in 0..100 {
            let p = Point(die(size.0, rng), die(size.1, rng));
            if let Status::Free = board.get_status(p) { return Some(p); }
        }
        None
    };
    for _ in 0..5 {
        if let Some(p) = pos(board, rng) { board.set_tile(p, fl); }
    }
}

//////////////////////////////////////////////////////////////////////////////

// AI

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Goal { Assess, Chase, Drink, Eat, Explore, Flee, Rest }

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum StepKind { Drink, Eat, Move, Look, Rest }

#[derive(Clone, Copy, Debug)]
struct Step { kind: StepKind, target: Point }

type Hint = (Goal, &'static Tile);

#[derive(Clone, Debug)]
pub struct FightState {
    age: i32,
    bias: Point,
    target: Point,
    search_turns: i32,
}

#[derive(Clone, Debug)]
pub struct FlightState {
    done: bool,
    threats: Vec<Point>,
    since_seen: i32,
    till_assess: i32,
}

#[derive(Clone, Debug)]
pub struct AIState {
    goal: Goal,
    plan: Vec<Step>,
    time: Timestamp,
    hints: HashMap<Goal, Point>,
    fight: Option<FightState>,
    flight: Option<FlightState>,
    turn_times: VecDeque<Timestamp>,
    till_assess: i32,
    till_hunger: i32,
    till_thirst: i32,
    till_rested: i32,
    debug_targets: Vec<Point>,
    debug_utility: HashMap<Point, u8>,
}

impl AIState {
    pub fn new(rng: &mut RNG) -> Self {
        Self {
            goal: Goal::Explore,
            plan: vec![],
            time: Default::default(),
            hints: Default::default(),
            fight: None,
            flight: None,
            turn_times: Default::default(),
            till_assess: rng.gen::<i32>().rem_euclid(MAX_ASSESS),
            till_hunger: rng.gen::<i32>().rem_euclid(MAX_HUNGER),
            till_thirst: rng.gen::<i32>().rem_euclid(MAX_THIRST),
            till_rested: MAX_RESTED,
            debug_targets: vec![],
            debug_utility: Default::default(),
        }
    }

    fn age_at_turn(&self, turn: i32) -> i32 {
        if self.turn_times.is_empty() { return 0; }
        self.time - self.turn_times[min(self.turn_times.len() - 1, turn as usize)]
    }

    fn record_turn(&mut self, time: Timestamp) {
        if self.turn_times.len() == TURN_TIMES_LIMIT { self.turn_times.pop_back(); }
        self.turn_times.push_front(self.time);
        self.time = time;
    }
}

fn coerce(source: Point, path: &[Point]) -> BFSResult {
    if path.is_empty() {
        BFSResult { dirs: vec![dirs::NONE], targets: vec![source] }
    } else {
        BFSResult { dirs: vec![path[0] - source], targets: vec![path[path.len() - 1]] }
    }
}

fn safe_inv_l2(point: Point) -> f64 {
    if point == Point::default() { return 0. }
    (point.len_l2_squared() as f64).sqrt().recip()
}

fn sample_scored_points(entity: &Entity, scores: &Vec<(Point, f64)>,
                        rng: &mut RNG, utility: &mut HashMap<Point, u8>) -> Option<BFSResult> {
    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| known.get(p).status();

    let max = scores.iter().fold(
        0., |acc, x| if acc > x.1 { acc } else { x.1 });
    if max == 0. { return None; }

    let limit = (1 << 16) - 1;
    let inverse = (limit as f64) / max;
    let values: Vec<_> = scores.iter().filter_map(|&(p, score)| {
        let score = min((inverse * score).floor() as i32, limit);
        if score > 0 { Some((score, p)) } else { None }
    }).collect();
    if values.is_empty() { return None; }

    for &(score, point) in &values {
        utility.insert(point, (score >> 8) as u8);
    }

    let target = *weighted(&values, rng);
    let path = AStar(pos, target, 1024, check)?;
    Some(coerce(pos, &path))
}

fn explore(entity: &Entity, rng: &mut RNG,
           utility: &mut HashMap<Point, u8>) -> Option<BFSResult> {
    utility.clear();
    let (known, pos, dir) = (&*entity.known, entity.pos, entity.dir);
    let check = |p: Point| known.get(p).status();
    let map = FastDijkstraMap(pos, check, 1024, 64, 12);
    if map.is_empty() { return None; }

    let mut min_age = std::i32::MAX;
    for &point in map.keys() {
        let age = known.get(point).time_since_seen();
        if age > 0 { min_age = min(min_age, age); }
    }
    if min_age == std::i32::MAX { min_age = 1; }

    let inv_dir_l2 = safe_inv_l2(dir);

    let score = |p: Point, distance: i32| -> f64 {
        if distance == 0 { return 0.; }
        let age = known.get(p).time_since_seen();

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let unblocked_neighbors = dirs::ALL.iter().filter(
                |&&x| !known.get(p + x).blocked()).count();

        let bonus0 = 1. / 65536. * ((age as f64 / min_age as f64) + 1. / 16.);
        let bonus1 = unblocked_neighbors == dirs::ALL.len();
        let bonus2 = unblocked_neighbors > 0;

        let base = (if bonus0 > 1. { 1. } else { bonus0 }) *
                   (if bonus1 {  8.0 } else { 1.0 }) *
                   (if bonus2 { 64.0 } else { 1.0 });
        base * (cos + 1.).pow(4) / (distance as f64).pow(2)
    };

    let scores: Vec<_> = map.iter().map(
        |(&p, &distance)| (p, score(p, distance))).collect();
    sample_scored_points(entity, &scores, rng, utility)
}

fn search_around(entity: &Entity, source: Point, age: i32, bias: Point,
                 rng: &mut RNG, utility: &mut HashMap<Point, u8>) -> Option<BFSResult> {
    utility.clear();
    let (known, pos, dir) = (&*entity.known, entity.pos, entity.dir);
    let check = |p: Point| known.get(p).status();
    let map = FastDijkstraMap(pos, check, 1024, 64, 12);
    if map.is_empty() { return None; }

    let inv_dir_l2 = safe_inv_l2(dir);
    let inv_bias_l2 = safe_inv_l2(bias);

    let score = |p: Point, distance: i32| -> f64 {
        if distance == 0 { return 0.; }
        let cell = known.get(p);
        if cell.time_since_entity_visible() < age || cell.blocked() { return 0. }

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos0 = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let cos1 = delta.dot(bias) as f64 * inv_delta_l2 * inv_bias_l2;
        let angle = ((cos0 + 1.) * (cos1 + 1.)).pow(4);
        let bonus = if p == source && source != pos { 64. } else { 1. };

        angle * bonus / (((p - source).len_l2_squared() + 1) as f64).pow(2)
    };

    let scores: Vec<_> = map.iter().map(
        |(&p, &distance)| (p, score(p, distance))).collect();
    sample_scored_points(entity, &scores, rng, utility)
}

fn attack_target(entity: &Entity, target: Point, rng: &mut RNG) -> Action {
    let (known, source) = (&*entity.known, entity.pos);
    if source == target { return Action::Idle; }

    let range = ATTACK_RANGE;
    let valid = |x| has_line_of_sight(x, target, known, range);
    if move_ready(entity) && valid(source) {
        return Action::Attack(target);
    }
    path_to_target(entity, target, known, range, valid, rng)
}

fn assess_dirs(dirs: &[Point], turns: i32, ai: &mut AIState, rng: &mut RNG) {
    if dirs.is_empty() { return; }

    for i in 0..ASSESS_STEPS {
        let scale = 1000;
        let steps = rng.gen::<i32>().rem_euclid(turns) + 1;
        let angle = Normal::new(0., ASSESS_ANGLE).unwrap().sample(rng);
        let (sin, cos) = (angle.sin(), angle.cos());

        let Point(dx, dy) = dirs[i as usize % dirs.len()];
        let rx = (cos * (scale * dx) as f64) + (sin * (scale * dy) as f64);
        let ry = (cos * (scale * dy) as f64) - (sin * (scale * dx) as f64);
        let target = Point(rx as i32, ry as i32);
        for _ in 0..steps {
            ai.plan.push(Step { kind: StepKind::Look, target })
        }
    }
}

fn assess_nearby(entity: &Entity, ai: &mut AIState, rng: &mut RNG) {
    let mut base = |ai: &mut AIState| {
        assess_dirs(&[entity.dir], ASSESS_TURNS_EXPLORE, ai, rng);
    };

    let Some(flight) = &ai.flight else { return base(ai) };
    if flight.done || flight.threats.is_empty() { return base(ai); }

    let dirs: Vec<_> = flight.threats.iter().map(|&x| x - entity.pos).collect();
    assess_dirs(&dirs, ASSESS_TURNS_FLIGHT, ai, rng);
}

fn flee_from_threats(entity: &Entity, ai: &mut AIState, rng: &mut RNG) -> Option<BFSResult> {
    let Some(flight) = &ai.flight else { return None };
    if flight.done || flight.threats.is_empty() { return None; }
    if flight.since_seen >= max(flight.till_assess, 1) { return None; }

    // WARNING: FastDijkstraMap doesn't flag squares where we heard something,
    // but those squares tend to be where nearby enemies are!
    //
    // This reasoning applies to all uses of FastDijkstraMap, but it's worse
    // for fleeing entities because often the first step they try to make to
    // flee is to move into cell where they heard a target.

    ai.debug_utility.clear();
    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| known.get(p).status();
    let map = FastDijkstraMap(pos, check, 1024, 64, 12);
    if map.is_empty() { return None; }

    let threat_inv_l2s: Vec<_> = flight.threats.iter().map(
        |&x| (x, safe_inv_l2(pos - x))).collect();
    let scale = FastDijkstraLength(Point(1, 0)) as f64;

    let score = |p: Point, source_distance: i32| -> f64 {
        let mut inv_l2 = 0.;
        let mut threat = Point::default();
        let mut threat_distance = std::i32::MAX;
        for &(x, y) in &threat_inv_l2s {
            let z = FastDijkstraLength(x - p);
            if z < threat_distance { (threat, inv_l2, threat_distance) = (x, y, z); }
        }
        let blocked = {
            let los = LOS(p, threat);
            (1..los.len() - 1).any(|i| !known.get(los[i]).unblocked())
        };
        let frontier = dirs::ALL.iter().any(|&x| known.get(p + x).unknown());

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos = delta.dot(pos - threat) as f64 * inv_delta_l2 * inv_l2;

        // WARNING: This heuristic can cause a piece to be "checkmated" in a
        // corner, if the Dijkstra search below isn't wide enough for us to
        // find a cell which is further from the threat than the corner cell.
        let base = 1.5 * (threat_distance as f64) +
                   -1. * (source_distance as f64) +
                   16. * scale * (blocked as i32 as f64) +
                   16. * scale * (frontier as i32 as f64);
        (cos + 1.).pow(3) * base.pow(9)
    };

    let scores: Vec<_> = map.iter().map(
        |(&p, &distance)| (p, score(p, distance))).collect();
    sample_scored_points(entity, &scores, rng, &mut ai.debug_utility)
}

fn update_ai_state(entity: &Entity, hints: &[Hint], ai: &mut AIState) {
    ai.till_assess = max(0, ai.till_assess - 1);
    ai.till_hunger = max(0, ai.till_hunger - 1);
    ai.till_thirst = max(0, ai.till_thirst - 1);
    ai.till_rested = max(0, ai.till_rested - 1);

    let (known, pos) = (&*entity.known, entity.pos);
    let last_turn_age = known.time - ai.time;
    let mut seen = HashSet::default();
    for cell in &known.cells {
        if (ai.time - cell.last_seen) >= 0 { break; }
        for &(goal, tile) in hints {
            if cell.tile == tile && seen.insert(goal) {
                ai.hints.insert(goal, cell.point);
            }
        }
    }
    ai.record_turn(known.time);

    // "rival" means that we have a hostile relationship with that entity.
    // We'll end up with three states - Friendly, Neutral, or Rival - or more.
    // An entity is a "threat" if its a rival and our combat analysis against
    // it shows that we'd probably lose. These predicates can be generalized
    // to all entities.
    //
    // The threshold for a rival being a threat may also depend on some other
    // parameter like "aggressiveness". A maximally-aggressive entity will
    // stand and fight even in hopeless situations.

    // We're a predator, and we should chase and attack rivals.
    if entity.predator {
        let fight = std::mem::take(&mut ai.fight);
        let limit = ai.age_at_turn(MAX_SEARCH_TURNS);
        let mut targets: Vec<_> = known.entities.iter().filter(|x| x.rival).collect();
        if !targets.is_empty() {
            targets.sort_unstable_by_key(
                |x| (x.age, (x.pos - pos).len_l2_squared()));
            let EntityKnowledge { age, pos: target, .. } = *targets[0];
            let reset = age < last_turn_age;
            if age < limit {
                let (bias, search_turns) = if !reset && let Some(x) = fight {
                    (x.bias, x.search_turns + 1)
                } else {
                    (target - pos, 0)
                };
                ai.fight = Some(FightState { age, bias, search_turns, target });
            }
            if reset { ai.plan.clear(); }
        }
        return;
    }

    // We're prey, and we should treat rivals as threats.
    let limit = ai.age_at_turn(MAX_FLIGHT_TURNS);
    let reset = known.entities.iter().any(
        |x| x.age < last_turn_age && x.rival);
    let mut threats: Vec<_> = known.entities.iter().filter_map(
        |x| if x.age < limit && x.rival { Some(x.pos) } else { None }).collect();
    threats.sort_unstable_by_key(|x| (x.0, x.1));

    if let Some(x) = &mut ai.flight {
        if threats.is_empty() {
            ai.flight = None;
        } else {
            let assess = min(2 * x.till_assess, MAX_FLIGHT_TURNS);
            x.since_seen = if reset { 0 } else { x.since_seen + 1 };
            if reset || (!x.done && x.threats != threats) { ai.plan.clear(); }
            if reset && ai.goal == Goal::Assess { x.till_assess = assess;  }
            if reset { x.done = false; }
            x.threats = threats;
        }
    } else if !threats.is_empty() {
        let (done, since_seen, till_assess) = (false, 0, MIN_FLIGHT_TURNS);
        ai.flight = Some(FlightState { done, threats, since_seen, till_assess });
        ai.plan.clear();
    }
}

fn plan_cached(entity: &Entity, hints: &[Hint],
               ai: &mut AIState, rng: &mut RNG) -> Option<Action> {
    if ai.plan.is_empty() { return None; }

    // Check whether we can execute the next step in the plan.
    let (known, pos) = (&*entity.known, entity.pos);
    let next = *ai.plan.last().unwrap();
    let look = next.kind == StepKind::Look;
    let dir = next.target - pos;
    if !look && dir.len_l1() > 1 { return None; }

    // Check whether the plan's goal is still a top priority for us.
    let mut goals: Vec<Goal> = vec![];
    if let Some(x) = &ai.flight && !x.done {
        if x.since_seen > 0 { goals.push(Goal::Assess); }
        if x.since_seen < max(x.till_assess, 1) { goals.push(Goal::Flee); }
    } else if ai.fight.is_some() {
        goals.push(Goal::Chase);
    } else if ai.goal == Goal::Assess {
        goals.push(Goal::Assess);
    } else {
        if ai.till_hunger == 0 && ai.hints.contains_key(&Goal::Eat) {
            goals.push(Goal::Eat);
        }
        if ai.till_thirst == 0 && ai.hints.contains_key(&Goal::Drink) {
            goals.push(Goal::Drink);
        }
        if ai.till_rested < MIN_RESTED && ai.hints.contains_key(&Goal::Rest) {
            goals.push(Goal::Rest);
        } else if ai.goal == Goal::Rest {
            goals.push(Goal::Rest);
        }
    }
    if goals.is_empty() { goals.push(Goal::Explore); }
    if !goals.contains(&ai.goal) { return None; }

    // Check if we saw a shortcut that would also satisfy the goal.
    if let Some(&x) = ai.hints.get(&ai.goal) && known.get(x).visible() {
        let target = ai.plan.iter().find_map(
            |x| if x.kind == StepKind::Move { Some(x.target) } else { None });
        if let Some(y) = target && AStarLength(pos - x) < AStarLength(pos - y) {
            let los = LOS(pos, x);
            let check = |p: Point| known.get(p).status();
            let free = (1..los.len() - 1).all(|i| check(los[i]) == Status::Free);
            if free { return None; }
        }
    }

    // Check if we got specific information that invalidates the plan.
    let point_matches_goal = |goal: Goal, point: Point| {
        let tile = hints.iter().find_map(
            |x| if x.0 == goal { Some(x.1) } else { None });
        if tile.is_none() { return false; }
        known.get(point).tile() == tile
    };
    let step_valid = |Step { kind, target }| match kind {
        StepKind::Rest => point_matches_goal(Goal::Rest, target),
        StepKind::Drink => point_matches_goal(Goal::Drink, target),
        StepKind::Eat => point_matches_goal(Goal::Eat, target),
        StepKind::Look => true,
        StepKind::Move => match known.get(target).status() {
            Status::Occupied => target != next.target,
            Status::Blocked  => false,
            Status::Free     => true,
            Status::Unknown  => true,
        }
    };
    if !ai.plan.iter().all(|&x| step_valid(x)) { return None; }

    // The plan is good! Execute the next step.
    ai.plan.pop();
    let wait = Some(Action::Idle);
    match next.kind {
        StepKind::Rest => { ai.till_rested += 2; wait }
        StepKind::Drink => { ai.till_thirst = MAX_THIRST; wait }
        StepKind::Eat => { ai.till_hunger = MAX_HUNGER; wait }
        StepKind::Look => {
            if ai.plan.is_empty() && ai.goal == Goal::Assess {
                ai.till_assess = rng.gen::<i32>().rem_euclid(MAX_ASSESS);
                if let Some(x) = &mut ai.flight {
                    x.till_assess = MIN_FLIGHT_TURNS;
                    x.done = true;
                }
            }
            Some(Action::Look(next.target))
        }
        StepKind::Move => {
            let mut target = next.target;
            for next in ai.plan.iter().rev().take(8) {
                if next.kind == StepKind::Look { break; }
                if LOS(pos, next.target).iter().all(|&x| !known.get(x).blocked()) {
                    target = next.target;
                }
            }
            let turns = (||{
                let running = ai.goal == Goal::Flee || ai.goal == Goal::Chase;
                if !running { return WANDER_TURNS; }
                if !move_ready(entity) { return SLOWED_TURNS; }
                if ai.goal == Goal::Flee { return 1.; }
                let Some(fight) = &ai.fight else { return WANDER_TURNS; };
                if fight.search_turns < MIN_SEARCH_TURNS { 1. } else { WANDER_TURNS }
            })();
            let look = if target == pos { entity.dir } else { target - pos };
            Some(Action::Move(Move { look, step: dir, turns }))
        }
    }
}

fn plan_npc(entity: &Entity, ai: &mut AIState, rng: &mut RNG) -> Action {
    let hints = [
        (Goal::Drink, Tile::get('~')),
        (Goal::Eat,   Tile::get('%')),
        (Goal::Rest,  Tile::get('"')),
    ];
    update_ai_state(entity, &hints, ai);
    if let Some(x) = plan_cached(entity, &hints, ai, rng) { return x; }

    ai.plan.clear();
    ai.goal = Goal::Explore;
    ai.debug_targets.clear();

    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| known.get(p).status();

    let mut result = {
        let mut result = BFSResult::default();

        if let Some(x) = flee_from_threats(entity, ai, rng) {
            (ai.goal, result) = (Goal::Flee, x);
        } else if let Some(x) = &ai.flight && !x.done {
            (ai.goal, result) = (Goal::Assess, coerce(pos, &[]));
        } else if let Some(x) = &ai.fight {
            if x.age == 0 {
                ai.goal = Goal::Chase;
                return attack_target(entity, x.target, rng);
            }
            let utility = &mut ai.debug_utility;
            let search_nearby = x.search_turns > x.bias.len_l1();
            let source = if search_nearby { entity.pos } else { x.target };
            if let Some(y) = search_around(entity, source, x.age, x.bias, rng, utility) {
                (ai.goal, result) = (Goal::Chase, y);
            }
        }

        let mut add_candidates = |ai: &mut AIState, goal: Goal| {
            if ai.goal != Goal::Explore { return; }

            let option = ai.hints.get(&goal);
            let tile = hints.iter().find_map(
                |x| if x.0 == goal { Some(x.1) } else { None });
            if option.is_none() || tile.is_none() { return; }

            let (option, tile) = (*option.unwrap(), tile.unwrap());
            let target = |p: Point| known.get(p).tile() == Some(tile);
            if target(pos) {
                (ai.goal, result) = (goal, coerce(pos, &[]));
            } else if let Some(x) = DijkstraSearch(pos, target, ASTAR_LIMIT_WANDER, check) {
                (ai.goal, result) = (goal, coerce(pos, &x));
            } else if let Some(x) = AStar(pos, option, ASTAR_LIMIT_WANDER, check) {
                (ai.goal, result) = (goal, coerce(pos, &x));
            }
        };
        if ai.till_rested < MIN_RESTED { add_candidates(ai, Goal::Rest); }
        if ai.till_thirst == 0 { add_candidates(ai, Goal::Drink); }
        if ai.till_hunger == 0 { add_candidates(ai, Goal::Eat); }

        if result.dirs.is_empty() && ai.till_assess == 0 {
            (ai.goal, result) = (Goal::Assess, coerce(pos, &[]));
        } else if result.dirs.is_empty() &&
               let Some(x) = explore(entity, rng, &mut ai.debug_utility) {
            (ai.goal, result) = (Goal::Explore, x);
        }
        result
    };

    let fallback = |result: BFSResult, rng: &mut RNG| {
        let dirs = &result.dirs;
        let dirs = if dirs.is_empty() { dirs::ALL.as_slice() } else { dirs };
        wander(*sample(dirs, rng))
    };

    if result.targets.is_empty() { return fallback(result, rng); }

    ai.debug_targets = result.targets.clone();
    let target = *result.targets.select_nth_unstable_by_key(
        0, |&x| (x - pos).len_l2_squared()).1;

    if let Some(path) = AStar(pos, target, ASTAR_LIMIT_WANDER, check) {
        let kind = StepKind::Move;
        ai.plan = path.into_iter().map(|x| Step { kind, target: x }).collect();
        match ai.goal {
            Goal::Assess => assess_nearby(entity, ai, rng),
            Goal::Chase => {}
            Goal::Drink => ai.plan.push(Step { kind: StepKind::Drink, target }),
            Goal::Eat => ai.plan.push(Step { kind: StepKind::Eat, target }),
            Goal::Explore => {}
            Goal::Flee => {}
            Goal::Rest => {
                let delta = max(MAX_RESTED - ai.till_rested, 0);
                for _ in 0..delta { ai.plan.push(Step { kind: StepKind::Rest, target }) };
            }
        }
        ai.plan.reverse();
        if let Some(x) = plan_cached(entity, &hints, ai, rng) { return x; }
        ai.plan.clear();
    }
    fallback(result, rng)
}

fn has_line_of_sight(source: Point, target: Point, known: &Knowledge, range: i32) -> bool {
    if (source - target).len_nethack() > range { return false; }
    if !known.get(target).visible() { return false; }
    let los = LOS(source, target);
    let last = los.len() - 1;
    los.iter().enumerate().all(|(i, &p)| {
        if i == 0 || i == last { return true; }
        known.get(p).status() == Status::Free
    })
}

fn path_to_target<F: Fn(Point) -> bool>(
        entity: &Entity, target: Point, known: &Knowledge,
        range: i32, valid: F, rng: &mut RNG) -> Action {
    let source = entity.pos;
    let check = |p: Point| known.get(p).status();
    let step = |dir: Point| {
        let look = target - source - dir;
        Action::Move(Move { step: dir, look, turns: 1. })
    };

    // Given a non-empty list of "good" directions (each of which brings us
    // close to attacking the target), choose one closest to our attack range.
    let pick = |dirs: &Vec<Point>, rng: &mut RNG| {
        let cell = known.get(target);
        let obscured = cell.shade() || matches!(cell.tile(), Some(x) if x.limits_vision());
        let distance = if obscured { 1 } else { range };

        assert!(!dirs.is_empty());
        let scores: Vec<_> = dirs.iter().map(
            |&x| ((x + source - target).len_nethack() - distance).abs()).collect();
        let best = *scores.iter().reduce(|acc, x| min(acc, x)).unwrap();
        let opts: Vec<_> = (0..dirs.len()).filter(|&i| scores[i] == best).collect();
        step(dirs[*sample(&opts, rng)])
    };

    // If we could already attack the target, don't move out of view.
    if valid(source) {
        let mut dirs = vec![Point::default()];
        for &x in &dirs::ALL {
            if check(source + x) != Status::Free { continue; }
            if valid(source + x) { dirs.push(x); }
        }
        return pick(&dirs, rng);
    }

    // Else, pick a direction which brings us in view.
    let result = BFS(source, &valid, BFS_LIMIT_ATTACK, check);
    if let Some(x) = result && !x.dirs.is_empty() { return pick(&x.dirs, rng); }

    // Else, move towards the target.
    let path = AStar(source, target, ASTAR_LIMIT_ATTACK, check);
    let dir = path.and_then(|x| if x.is_empty() { None } else { Some(x[0] - source) });
    step(dir.unwrap_or_else(|| *sample(&dirs::ALL, rng)))
}

//////////////////////////////////////////////////////////////////////////////

// Action

struct Move { look: Point, step: Point, turns: f64 }

enum Action {
    Idle,
    WaitForInput,
    Look(Point),
    Move(Move),
    Attack(Point),
}

struct ActionResult {
    success: bool,
    moves: f64,
    turns: f64,
}

impl ActionResult {
    fn failure() -> Self { Self { success: false, moves: 0., turns: 1. } }
    fn success() -> Self { Self::success_turns(1.) }
    fn success_moves(moves: f64) -> Self { Self { success: true,  moves, turns: 1. } }
    fn success_turns(turns: f64) -> Self { Self { success: true,  moves: 0., turns } }
}

fn move_ready(entity: &Entity) -> bool { entity.move_timer <= 0 }

fn turn_ready(entity: &Entity) -> bool { entity.turn_timer <= 0 }

fn charge(entity: &mut Entity) {
    let charge = (TURN_TIMER as f64 * entity.speed).round() as i32;
    if entity.move_timer > 0 { entity.move_timer -= charge; }
    if entity.turn_timer > 0 { entity.turn_timer -= charge; }
}

fn drain(entity: &mut Entity, result: &ActionResult) {
    entity.move_timer += (MOVE_TIMER as f64 * result.moves).round() as i32;
    entity.turn_timer += (TURN_TIMER as f64 * result.turns).round() as i32;
}

fn step(dir: Point, turns: f64) -> Action {
    Action::Move(Move { look: dir, step: dir, turns })
}

fn wander(dir: Point) -> Action {
    step(dir, WANDER_TURNS)
}

fn plan(state: &mut State, eid: EID) -> Action {
    let player = eid == state.player;
    if player { return replace(&mut state.input, Action::WaitForInput); }

    let mut ai = state.ai.take().unwrap_or_else(
        || Box::new(AIState::new(&mut state.rng)));
    swap(&mut ai, &mut state.board.entities[eid].ai);

    let entity = &state.board.entities[eid];
    let action = plan_npc(entity, &mut ai, &mut state.rng);

    swap(&mut ai, &mut state.board.entities[eid].ai);
    state.ai = Some(ai);
    action
}

fn act(state: &mut State, eid: EID, action: Action) -> ActionResult {
    match action {
        Action::Idle => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::Look(dir) => {
            state.board.entities[eid].dir = dir;
            ActionResult::success()
        }
        Action::Move(Move { look, step, turns }) => {
            let entity = &mut state.board.entities[eid];
            if look != dirs::NONE { entity.dir = look; }
            if step == dirs::NONE { return ActionResult::success_turns(turns); }

            // Moving diagonally is slower. Moving quickly is noisier.
            let noisy = !(entity.player || turns > 1.);
            let turns = step.len_l2() * turns;
            let color = entity.glyph.fg();
            let source = entity.pos;
            let target = source + step;

            match state.board.get_status(target) {
                Status::Blocked | Status::Unknown => {
                    state.board.entities[eid].dir = step;
                    ActionResult::failure()
                }
                Status::Occupied => {
                    state.board.entities[eid].dir = step;
                    ActionResult::failure()
                }
                Status::Free => {
                    // Noise generation, for quick moves.
                    let mut updated = vec![];
                    let max = if noisy { 4 } else { 1 };
                    for &oid in &state.board.entity_order {
                        if oid == eid { continue; }
                        let other = &state.board.entities[oid];
                        let sr = (other.pos - source).len_nethack() <= max;
                        let tr = (other.pos - target).len_nethack() <= max;
                        if sr || tr { updated.push((oid, tr)); }
                    }
                    state.board.move_entity(eid, target);
                    for (oid, heard) in updated {
                        state.board.update_known_entity(oid, eid, heard);
                    }

                    // Move animations, only for the player.
                    if eid != state.player {
                        let known = &state.board.entities[state.player].known;
                        let source_seen = known.get(source).can_see_entity_at();
                        let target_seen = known.get(target).can_see_entity_at();
                        if source_seen || target_seen {
                            let limit = UI_MOVE_FRAMES;
                            if target_seen {
                                let m = MoveAnimation { color, frame: 0, limit };
                                state.moves.insert(source, m);
                            } else {
                                let a = limit / 2;
                                let m = MoveAnimation { color, frame: 0, limit };
                                state.moves.insert(source, m);
                                let m = MoveAnimation { color, frame: -a, limit };
                                state.moves.insert(target, m);
                            }
                        }
                    }
                    ActionResult::success_turns(turns)
                }
            }
        }
        Action::Attack(target) => {
            let entity = &mut state.board.entities[eid];
            let (known, source) = (&entity.known, entity.pos);
            if source == target {
                return ActionResult::failure();
            } else if !has_line_of_sight(source, target, known, ATTACK_RANGE) {
                return ActionResult::failure();
            }

            entity.dir = target - source;
            let oid = state.board.get_cell(target).eid;

            let cb = move |board: &mut Board, rng: &mut RNG| {
                let Some(oid) = oid else { return; };
                let cb = move |board: &mut Board, _: &mut RNG| {
                    let Some(other) = board.entities.get_mut(oid) else { return; };

                    // Player scores a critical hit if they're unseen.
                    let damage = if other.player {
                        1
                    } else if other.known.get(source).visible() {
                        0
                    } else {
                        other.cur_hp
                    };
                    let damage = 0;

                    if other.cur_hp > damage {
                        other.cur_hp -= damage;
                        board.update_known_entity(oid, eid, /*heard=*/true);
                    } else {
                        board.remove_entity(oid);
                    }
                };
                board.add_effect(apply_damage(board, target, Box::new(cb)), rng);
            };

            let rng = &mut state.rng;
            let effect = effect::HeadbuttEffect(&state.board, rng, source, target);
            state.add_effect(apply_effect(effect, FT::Hit, Box::new(cb)));
            ActionResult::success_moves(1.)
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Animation

type CB = Box<dyn Fn(&mut Board, &mut RNG)>;

fn apply_damage(board: &Board, target: Point, callback: CB) -> Effect {
    let eid = board.get_cell(target).eid;
    let Some(eid) = eid else { return Effect::default(); };

    let glyph = board.entities[eid].glyph;
    let flash = glyph.with_fg(Color::black()).with_bg(0x400);
    let particle = effect::Particle { glyph: flash, point: target };
    let mut effect = Effect::serial(vec![
        Effect::constant(particle, UI_DAMAGE_FLASH),
        Effect::pause(UI_DAMAGE_TICKS),
    ]);
    let frame = effect.frames.len() as i32;
    effect.add_event(Event::Callback { frame, callback });
    effect
}

fn apply_effect(mut effect: Effect, what: FT, callback: CB) -> Effect {
    let frame = effect.events.iter().find_map(
        |x| if x.what() == Some(what) { Some(x.frame()) } else { None });
    if let Some(frame) = frame {
        effect.add_event(Event::Callback { frame, callback });
    }
    effect
}

//////////////////////////////////////////////////////////////////////////////

// Update

fn get_direction(ch: char) -> Option<Point> {
    match ch {
        'h' => Some(dirs::W),
        'j' => Some(dirs::S),
        'k' => Some(dirs::N),
        'l' => Some(dirs::E),
        'y' => Some(dirs::NW),
        'u' => Some(dirs::NE),
        'b' => Some(dirs::SW),
        'n' => Some(dirs::SE),
        '.' => Some(dirs::NONE),
        _ => None,
    }
}

fn process_input(state: &mut State, input: Input) {
    if input == Input::Char('q') || input == Input::Char('w') {
        let board = &state.board;
        let i = board.entity_order.iter().position(
            |&x| Some(x) == state.pov).unwrap_or(0);
        let l = board.entity_order.len();
        let j = (i + if input == Input::Char('q') { l - 1 } else { 1 }) % l;
        state.pov = if j == 0 { None } else { Some(board.entity_order[j]) };
        return;
    }

    let Input::Char(ch) = input else { return; };
    let Some(dir) = get_direction(ch) else { return; };

    if dir == Point::default() {
        state.input = Action::Idle;
        return;
    }

    let player = state.get_player();
    let cell = player.known.get(player.pos + dir);
    if let Some(x) = cell.entity() && x.rival {
        state.input = Action::Attack(player.pos + dir);
    } else {
        state.input = step(dir, 1.);
    }
}

fn update_state(state: &mut State) {
    state.frame += 1;
    let pos = state.get_player().pos;
    state.board.update_env(state.frame, pos, &mut state.rng);

    for x in state.moves.values_mut() { x.frame += 1; }
    state.moves.retain(|_, v| v.frame < v.limit);

    if state.board.advance_effect(&mut state.rng) {
        state.board.update_known(state.player);
        return;
    }

    let game_loop_active = |state: &State| {
        state.get_player().cur_hp > 0 && state.board.get_frame().is_none()
    };

    let needs_input = |state: &State| {
        if !game_loop_active(state) { return false; }
        if !matches!(state.input, Action::WaitForInput) { return false; }
        state.board.get_active_entity() == state.player
    };

    while !state.inputs.is_empty() && needs_input(state) {
        let input = state.inputs.remove(0);
        process_input(state, input);
    }

    let mut update = false;

    while game_loop_active(state) {
        let eid = state.board.get_active_entity();
        let entity = &state.board.entities[eid];
        let player = entity.player;

        if !turn_ready(entity) {
            state.board.advance_entity();
            continue;
        } else if player && needs_input(state) {
            break;
        }

        state.board.update_known(eid);
        state.board.update_known(state.player);

        update = true;
        let action = plan(state, eid);
        let result = act(state, eid, action);
        if player && !result.success { break; }

        //state.board.update_known(eid);
        //state.board.update_known(state.player);

        if let Some(x) = state.board.entities.get_mut(eid) { drain(x, &result); }
    }

    if update {
        state.board.update_known(state.player);
        if let Some(x) = state.pov && state.board.entity_order.contains(&x) {
            state.board.update_known(x);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// State

#[derive(Copy, Clone)]
struct MoveAnimation {
    color: Color,
    frame: i32,
    limit: i32,
}

pub struct State {
    board: Board,
    frame: usize,
    input: Action,
    inputs: Vec<Input>,
    player: EID,
    pov: Option<EID>,
    rng: RNG,
    // Update fields
    ai: Option<Box<AIState>>,
    // Animations
    moves: HashMap<Point, MoveAnimation>,
}

impl State {
    pub fn new(seed: Option<u64>) -> Self {
        let size = Point(WORLD_SIZE, WORLD_SIZE);
        let pos = Point(size.0 / 2, size.1 / 2);
        let rng = seed.map(|x| RNG::seed_from_u64(x));
        let mut rng = rng.unwrap_or_else(|| RNG::from_entropy());
        let mut board = Board::new(size, LIGHT, WEATHER);

        loop {
            mapgen(&mut board, &mut rng);
            if !board.get_tile(pos).blocks_movement() { break; }
        }
        let input = Action::WaitForInput;
        let glyph = Glyph::wdfg('@', 0x222);
        let (player, speed) = (true, SPEED_PC_);
        let args = EntityArgs { glyph, player, predator: false, pos, speed };
        let player = board.add_entity(&args, &mut rng);

        let die = |n: i32, rng: &mut RNG| rng.gen::<i32>().rem_euclid(n);
        let pos = |board: &Board, rng: &mut RNG| {
            for _ in 0..100 {
                let p = Point(die(size.0, rng), die(size.1, rng));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        for i in 0..4 {
            if let Some(x) = pos(&board, &mut rng) {
                let predator = i % 10 == 0;
                let (player, speed) = (false, SPEED_NPC);
                let glyph = Glyph::wdfg(if predator { 'R' } else { 'P' }, 0x222);
                let args = EntityArgs { glyph, player, predator, pos: x, speed };
                board.add_entity(&args, &mut rng);
            }
        }
        board.entities[player].dir = Point::default();
        board.update_known(player);

        let moves = Default::default();
        let ai = Some(Box::new(AIState::new(&mut rng)));
        Self { board, frame: 0, input, inputs: vec![], player, pov: None, rng, ai, moves }
    }

    fn get_player(&self) -> &Entity { &self.board.entities[self.player] }

    fn mut_player(&mut self) -> &mut Entity { &mut self.board.entities[self.player] }

    pub fn add_effect(&mut self, x: Effect) { self.board.add_effect(x, &mut self.rng) }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Buffer, debug: &mut String) {
        if buffer.data.is_empty() {
            let size = Point(2 * UI_MAP_SIZE_X + 2, UI_MAP_SIZE_Y + 3);
            let _ = std::mem::replace(buffer, Matrix::new(size, ' '.into()));
        }

        let entity = self.pov.and_then(
            |x| self.board.get_entity(x)).unwrap_or(self.get_player());
        let offset = entity.pos - Point(UI_MAP_SIZE_X / 2, UI_MAP_SIZE_Y / 2);
        let offset = Point::default();

        let size = Point(2 * UI_MAP_SIZE_X, UI_MAP_SIZE_Y);
        let bound = Rect { root: Point(0, size.1 + 2), size: Point(size.0, 1) };
        let slice = &mut Slice::new(buffer, bound);
        slice.write_str(&format!("HP: {}/{}", entity.cur_hp, entity.max_hp));

        let bound = Rect { root: Point(1, 1), size };
        self.render_box(buffer, &bound);

        let frame = self.board.get_frame();
        let slice = &mut Slice::new(buffer, bound);
        self.render_map(entity, frame, offset, slice);
        let known = &*entity.known;

        if entity.eid != self.player && frame.is_none() {
            let mut ai = entity.ai.clone();
            while ai.turn_times.len() > 2 { ai.turn_times.pop_back(); }
            ai.debug_utility.clear();
            ai.plan.clear();
            *debug = format!("{:?}", ai);

            for step in &entity.ai.plan {
                if step.kind == StepKind::Look { continue; }
                let Point(x, y) = step.target - offset;
                let point = Point(2 * x, y);
                let mut glyph = slice.get(point);
                if glyph.ch() == Glyph::wide(' ').ch() { glyph = Glyph::wide('.'); }
                slice.set(point, glyph.with_fg(0x400));
            }
            for (&point, &score) in &entity.ai.debug_utility {
                let Point(x, y) = point - offset;
                let point = Point(2 * x, y);
                let glyph = slice.get(point);
                slice.set(point, glyph.with_bg(Color::dark(score)));
            }
            for &target in &entity.ai.debug_targets {
                let Point(x, y) = target - offset;
                let point = Point(2 * x, y);
                let glyph = slice.get(point);
                slice.set(point, glyph.with_fg(Color::black()).with_bg(0x400));
            }
            for &eid in &self.board.entity_order {
                let other = &self.board.entities[eid];
                let Point(x, y) = other.pos - offset;
                slice.set(Point(2 * x, y), other.glyph);
            }
            for other in &known.entities {
                let color = if other.age == 0 { 0x040 } else {
                    if other.moved { 0x400 } else { 0x440 }
                };
                let glyph = other.glyph.with_fg(Color::black()).with_bg(color);
                let Point(x, y) = other.pos - offset;
                slice.set(Point(2 * x, y), glyph);
            };
        }

        if let Some(rain) = &self.board.rain {
            let base = Tile::get('~').glyph.fg();
            for drop in &rain.drops {
                let index = drop.frame - self.frame;
                let Some(&delta) = rain.path.get(index) else { continue; };
                let point = drop.point - delta;
                let cell = if index == 0 { known.get(point) } else { known.default() };
                if index == 0 && !cell.visible() { continue; }

                let Point(x, y) = point - offset;
                let ch = if index == 0 { 'o' } else { rain.ch };
                let color = if cell.shade() { Color::gray() } else { base };
                let glyph = Glyph::wdfg(ch, color);
                slice.set(Point(2 * x, y), glyph);
            }

            if rain.lightning > 0 {
                let color = Color::from(0x111 * (rain.lightning / 2));
                for y in 0..UI_MAP_SIZE_Y {
                    for x in 0..UI_MAP_SIZE_X {
                        let point = Point(2 * x, y);
                        slice.set(point, slice.get(point).with_bg(color));
                    }
                }
            }

            if rain.thunder > 0 {
                let shift = (rain.thunder - 1) % 4;
                if shift % 2 == 0 {
                    let space = Glyph::char(' ');
                    let delta = Point(shift - 1, 0);
                    let limit = if delta.1 > 0 { -1 } else { 0 };
                    for y in 0..UI_MAP_SIZE_Y {
                        for x in 0..(UI_MAP_SIZE_X + limit) {
                            let point = Point(2 * x, y);
                            slice.set(point + delta, slice.get(point));
                        }
                        slice.set(Point(0, y), space);
                        slice.set(Point(2 * UI_MAP_SIZE_X - 1, y), space);
                    }
                }
            }
        }
    }

    fn render_box(&self, buffer: &mut Buffer, rect: &Rect) {
        let Point(w, h) = rect.size;
        let color: Color = UI_COLOR.into();
        buffer.set(rect.root + Point(-1, -1), Glyph::chfg('┌', color));
        buffer.set(rect.root + Point( w, -1), Glyph::chfg('┐', color));
        buffer.set(rect.root + Point(-1,  h), Glyph::chfg('└', color));
        buffer.set(rect.root + Point( w,  h), Glyph::chfg('┘', color));

        let tall = Glyph::chfg('│', color);
        let flat = Glyph::chfg('─', color);
        for x in 0..w {
            buffer.set(rect.root + Point(x, -1), flat);
            buffer.set(rect.root + Point(x,  h), flat);
        }
        for y in 0..h {
            buffer.set(rect.root + Point(-1, y), tall);
            buffer.set(rect.root + Point( w, y), tall);
        }
    }

    fn render_map(&self, entity: &Entity, frame: Option<&Frame>,
                  offset: Point, slice: &mut Slice) {
        // Render each tile's base glyph, if it's known.
        let (known, player) = (&*entity.known, entity.player);
        let unseen = Glyph::wide(' ');

        let lookup = |point: Point| -> Glyph {
            let cell = known.get(point);
            let Some(tile) = cell.tile() else { return unseen; };

            let other = cell.entity();
            let dead = other.is_some_and(|x| !x.alive);
            let obscured = tile.limits_vision();
            let shadowed = cell.shade();

            let glyph = other.map(|x| x.glyph).unwrap_or(tile.glyph);
            let color = {
                if !cell.visible() { 0x011.into() }
                else if dead { 0x400.into() }
                else if shadowed { Color::gray() }
                else if obscured { tile.glyph.fg() }
                else { glyph.fg() }
            };
            glyph.with_fg(color)
        };

        slice.fill(Glyph::wide(' '));
        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let glyph = lookup(Point(x, y) + offset);
                slice.set(Point(2 * x, y), glyph);
            }
        }
        for entity in &known.entities {
            if !entity.heard { continue; }
            let Point(x, y) = entity.pos - offset;
            slice.set(Point(2 * x, y), Glyph::wide('?'));
        }
        if entity.player {
            for (&k, v) in &self.moves {
                let Point(x, y) = k - offset;
                let p = Point(2 * x, y);
                if v.frame < 0 || !slice.contains(p) { continue; }

                let alpha = 1.0 - (v.frame as f64 / v.limit as f64);
                let color = v.color.fade(UI_MOVE_ALPHA * alpha);
                slice.set(p, slice.get(p).with_bg(color));
            }
        }

        // Render any animation that's currently running.
        if let Some(frame) = frame {
            for &effect::Particle { point, glyph } in frame {
                if player && !known.get(point).visible() { continue; }
                let Point(x, y) = point - offset;
                slice.set(Point(2 * x, y), glyph);
            }
        }

        // If we're still playing, animate arrows showing NPC facing.
        if entity.cur_hp == 0 { return; }

        let length = 3 as usize;
        let mut arrows = vec![];
        for other in &known.entities {
            if other.friend || other.age > 0 { continue; }

            let (pos, dir) = (other.pos, other.dir);
            let ch = Glyph::ray(dir);
            let diff = dir.normalize(length as f64);
            arrows.push((ch, LOS(pos, pos + diff)));
        }

        for (ch, arrow) in &arrows {
            let index = (self.frame / 2) % (8 * length);
            if let Some(x) = arrow.get(index + 1) {
                let point = Point(2 * (x.0 - offset.0), x.1 - offset.1);
                slice.set(point, Glyph::wide(*ch));
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

#[allow(soft_unstable)]
#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;
    use crate::pathing::DijkstraMap;

    const BFS_LIMIT: i32 = 32;
    const DIJKSTRA_LIMIT: i32 = 4096;

    #[bench]
    fn bench_bfs(b: &mut test::Bencher) {
        let map = generate_map(2 * BFS_LIMIT);
        b.iter(|| {
            let done = |_: Point| { false };
            let check = |p: Point| { map.get(&p).copied().unwrap_or(Status::Free) };
            BFS(Point::default(), done, BFS_LIMIT, check);
        });
    }

    #[bench]
    fn bench_dijkstra(b: &mut test::Bencher) {
        let map = generate_map(2 * BFS_LIMIT);
        b.iter(|| {
            let done = |_: Point| { false };
            let check = |p: Point| { map.get(&p).copied().unwrap_or(Status::Free) };
            DijkstraSearch(Point::default(), done, DIJKSTRA_LIMIT, check);
        });
    }

    #[bench]
    fn bench_dijkstra_map(b: &mut test::Bencher) {
        let map = generate_map(2 * BFS_LIMIT);
        b.iter(|| {
            let mut result = HashMap::default();
            result.insert(Point::default(), 0);
            let check = |p: Point| { map.get(&p).copied().unwrap_or(Status::Free) };
            DijkstraMap(check, DIJKSTRA_LIMIT, &mut result);
        });
    }

    #[bench]
    fn bench_fast_dijkstra_map(b: &mut test::Bencher) {
        let map = generate_map(2 * BFS_LIMIT);
        b.iter(|| {
            let mut result = HashMap::default();
            result.insert(Point::default(), 0);
            let check = |p: Point| { map.get(&p).copied().unwrap_or(Status::Free) };
            FastDijkstraMap(Point::default(), check, DIJKSTRA_LIMIT,
                            2 * BFS_LIMIT, FOV_RADIUS_NPC);
        });
    }

    #[bench]
    fn bench_state_update(b: &mut test::Bencher) {
        let mut state = State::new(Some(17));
        b.iter(|| {
            state.inputs.push(Input::Char('.'));
            state.update();
            while state.board.get_frame().is_some() { state.update(); }
        });
    }

    fn generate_map(n: i32) -> HashMap<Point, Status> {
        let mut result = HashMap::default();
        let mut rng = RNG::seed_from_u64(17);
        for x in -n..n + 1 {
            for y in -n..n + 1 {
                let f = rng.gen::<i32>().rem_euclid(100);
                let s = if f < 20 { Status::Blocked } else { Status::Free };
                result.insert(Point(x, y), s);
            }
        }
        result
    }
}
