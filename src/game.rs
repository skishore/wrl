use std::cmp::{max, min};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::mem::{replace, swap};
use std::num::NonZeroU64;
use std::ops::{Index, IndexMut};

use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use slotmap::{DefaultKey, Key, KeyData, SlotMap};

use crate::static_assert_size;
use crate::base::{Buffer, Color, Glyph};
use crate::base::{HashMap, HashSet, LOS, Matrix, Point, dirs};
use crate::base::{sample, RNG};
use crate::knowledge::{Knowledge, Timestamp, Vision, VisionArgs};
use crate::pathing::{AStar, AStarLength, BFS, BFSResult, Status};
use crate::pathing::DijkstraSearch;

//////////////////////////////////////////////////////////////////////////////

// Constants

const ASTAR_LIMIT_ATTACK: i32 = 32;
const ASTAR_LIMIT_SEARCH: i32 = 256;
const ASTAR_LIMIT_WANDER: i32 = 1024;
const BFS_LIMIT_ATTACK: i32 = 8;
const BFS_LIMIT_WANDER: i32 = 64;
const EXPLORE_FUZZ: i32 = 64;

const ASSESS_ANGLE: f64 = TAU / 18.;
const ASSESS_STEPS: i32 = 4;
const ASSESS_TURNS: i32 = 16;

const MAX_ASSESS: i32 = 32;
const MAX_HUNGER: i32 = 1024;
const MAX_THIRST: i32 = 256;

const MAX_FLIGHT_TURNS: i32 = 64;
const MAX_FOLLOW_TURNS: i32 = 64;
const TURN_TIMES_LIMIT: usize = 64;

const FOV_RADIUS_NPC: i32 = 12;
const FOV_RADIUS_PC_: i32 = 21;

const LIGHT: Light = Light::Sun(Point(4, 1));
const WEATHER: Weather = Weather::None;
const WORLD_SIZE: i32 = 100;

const UI_MAP_SIZE_X: i32 = 2 * FOV_RADIUS_PC_ + 1;
const UI_MAP_SIZE_Y: i32 = 2 * FOV_RADIUS_PC_ + 1;

#[derive(Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char) }

//////////////////////////////////////////////////////////////////////////////

// Tile

const FLAG_NONE: u32 = 0;
const FLAG_BLOCKED: u32 = 1 << 0;
const FLAG_OBSCURE: u32 = 1 << 1;

pub struct Tile {
    pub flags: u32,
    pub glyph: Glyph,
    pub description: &'static str,
}
static_assert_size!(Tile, 24);

impl Tile {
    fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    pub fn blocked(&self) -> bool { self.flags & FLAG_BLOCKED != 0 }
    pub fn obscure(&self) -> bool { self.flags & FLAG_OBSCURE != 0 }
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
            ('.', (FLAG_NONE,    Glyph::wdfg('.', 0x222), "grass")),
            ('"', (FLAG_OBSCURE, Glyph::wdfg('"', 0x120), "tall grass")),
            ('#', (FLAG_BLOCKED, Glyph::wdfg('#', 0x010), "a tree")),
            ('%', (FLAG_NONE,    Glyph::wdfg('%', 0x200), "flowers")),
            ('~', (FLAG_NONE,    Glyph::wdfg('~', 0x013), "water")),
        ];
        let mut result = HashMap::default();
        for (ch, (flags, glyph, description)) in items {
            result.insert(ch, Tile { flags, glyph, description });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Entity

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct EID(NonZeroU64);
static_assert_size!(Option<EID>, 8);

impl Default for EID {
    fn default() -> Self {
        to_eid(DefaultKey::null())
    }
}

fn to_key(eid: EID) -> DefaultKey {
    KeyData::from_ffi(eid.0.get()).into()
}

fn to_eid(key: DefaultKey) -> EID {
    EID(NonZeroU64::new(key.data().as_ffi()).unwrap())
}

#[derive(Default)]
struct EntityMap {
    map: SlotMap<DefaultKey, Entity>,
}

impl EntityMap {
    fn add(&mut self, args: &EntityArgs, rng: &mut RNG) -> EID {
        let dir = *sample(&dirs::ALL, rng);
        let key = self.map.insert_with_key(|x| Entity {
            eid: to_eid(x),
            glyph: args.glyph,
            known: Default::default(),
            ai: Box::new(AIState::new(rng)),
            player: args.player,
            pos: args.pos,
            dir,
        });
        to_eid(key)
    }

    fn get(&self, eid: EID) -> Option<&Entity> {
        self.map.get(to_key(eid))
    }

    fn get_mut(&mut self, eid: EID) -> Option<&mut Entity> {
        self.map.get_mut(to_key(eid))
    }
}

impl Index<EID> for EntityMap {
    type Output = Entity;
    fn index(&self, eid: EID) -> &Self::Output {
        self.get(eid).unwrap()
    }
}

impl IndexMut<EID> for EntityMap {
    fn index_mut(&mut self, eid: EID) -> &mut Self::Output {
        self.get_mut(eid).unwrap()
    }
}

pub struct Entity {
    pub eid: EID,
    pub glyph: Glyph,
    pub known: Box<Knowledge>,
    pub ai: Box<AIState>,
    pub player: bool,
    pub pos: Point,
    pub dir: Point,
}

struct EntityArgs {
    glyph: Glyph,
    player: bool,
    pos: Point,
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
            Light::Sun(x) =>
                LOS(Point::default(), x).into_iter().skip(1).collect(),
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

    // Getters

    fn get_active_entity(&self) -> EID {
        self.entity_order[self.active_entity_index]
    }

    pub fn get_cell(&self, p: Point) -> &Cell { self.map.entry_ref(p) }

    pub fn get_entity(&self, eid: EID) -> Option<&Entity> { self.entities.get(eid) }

    pub fn get_light(&self) -> &Light { &self.light }

    fn get_size(&self) -> Point { self.map.size }

    fn get_status(&self, p: Point) -> Status {
        let Cell { eid, tile, .. } = self.get_cell(p);
        if eid.is_some() { return Status::Occupied; }
        if tile.blocked() { Status::Blocked } else { Status::Free }
    }

    fn get_tile(&self, p: Point) -> &'static Tile { self.get_cell(p).tile }

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

    fn reset(&mut self, tile: &'static Tile) {
        self.map.fill(Cell { eid: None, shadow: 0, tile });
        self.update_edge_shadows();
        self.entity_order.clear();
        self.active_entity_index = 0;
    }

    fn set_tile(&mut self, point: Point, tile: &'static Tile) {
        let Some(cell) = self.map.entry_mut(point) else { return; };
        let old_shadow = if cell.tile.blocked() { 1 } else { 0 };
        let new_shadow = if tile.blocked() { 1 } else { 0 };
        cell.tile = tile;
        self.update_shadow(point, new_shadow - old_shadow);
    }

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
        for x in &self.shadow {
            let Some(cell) = self.map.entry_mut(point + *x) else { continue; };
            cell.shadow += delta;
            assert!(cell.shadow >= 0);
        }
    }

    fn update_edge_shadows(&mut self) {
        let delta = if self.map.default.tile.blocked() { 1 } else { 0 };
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

    let automata = |rng: &mut RNG| -> Matrix<bool> {
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
                if d100() < 45 { result.set(Point(x, y),  true); }
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

    let walls = automata(rng);
    let grass = automata(rng);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let point = Point(x, y);
            if walls.get(point) {
                //board.set_tile(point, wt);
            } else if grass.get(point) {
                //board.set_tile(point, gt);
            }
        }
    }

    let die = |n: i32, rng: &mut RNG| rng.gen::<i32>().rem_euclid(n);
    let mut river = vec![Point::default()];
    for i in 1..size.1 {
        let last = river.iter().last().unwrap().0;
        let next = last + die(3, rng) - 1;
        river.push(Point(next, i));
    }
    let target = river[0] + *river.iter().last().unwrap();
    let offset = Point((size - target).0 / 2, 0);
    for x in &river { board.set_tile(*x + offset, wa); }

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
enum Goal { Assess, Chase, Drink, Eat, Explore, Flee }

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum StepKind { Drink, Eat, Move, Look }

#[derive(Clone, Copy, Debug)]
struct Step { kind: StepKind, target: Point }

type Hint = (Goal, &'static Tile);

struct FightState {}

struct FlightState {
    turns_since_seen: i32,
}

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
    debug_targets: Vec<Point>,
}

impl AIState {
    fn new(rng: &mut RNG) -> Self {
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
            debug_targets: vec![],
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

    fn update(&mut self, entity: &Entity, hints: &[Hint]) {
        self.till_assess = max(0, self.till_assess - 1);
        self.till_hunger = max(0, self.till_hunger - 1);
        self.till_thirst = max(0, self.till_thirst - 1);

        let (known, pos) = (&*entity.known, entity.pos);
        let last_turn_age = known.time - self.time;
        let mut seen = HashSet::default();
        for cell in &known.cells {
            if (self.time - cell.time) >= 0 { break; }
            for (goal, tile) in hints {
                if cell.tile == tile && seen.insert(goal) {
                    self.hints.insert(*goal, cell.point);
                }
            }
        }
        self.record_turn(known.time);
    }
}

fn coerce(source: Point, path: &[Point]) -> BFSResult {
    if path.is_empty() {
        BFSResult { dirs: vec![Point::default()], targets: vec![source] }
    } else {
        BFSResult { dirs: vec![path[0] - source], targets: vec![path[path.len() - 1]] }
    }
}

fn explore(entity: &Entity) -> Option<BFSResult> {
    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| {
        if p == pos { return Status::Free; }
        known.get(p).status().unwrap_or(Status::Free)
    };
    let done1 = |p: Point| {
        if !known.get(p).unknown() { return false; }
        dirs::ALL.iter().any(|x| known.get(p + *x).unblocked())
    };
    let done0 = |p: Point| {
        done1(p) && dirs::ALL.iter().all(|x| !known.get(p + *x).blocked())
    };

    BFS(pos, done0, BFS_LIMIT_WANDER, check).or_else(||
    BFS(pos, done1, BFS_LIMIT_WANDER, check))
}

fn plan_cached(entity: &Entity, hints: &[Hint],
               ai: &mut AIState, rng: &mut RNG) -> Option<Action> {
    if ai.plan.is_empty() { return None; }

    // Check whether we can execute the next step in the plan.
    let (known, pos) = (&*entity.known, entity.pos);
    let next = *ai.plan.iter().last().unwrap();
    let look = next.kind == StepKind::Look;
    let dir = next.target - pos;
    if !look && dir.len_l1() > 1 { return None; }
    if look && let Some(x) = &ai.flight && x.turns_since_seen == 0 { return None; }

    // Check whether the plan's goal is still a top priority for us.
    let mut goals: Vec<Goal> = vec![];
    if ai.flight.is_some() {
        goals.push(Goal::Flee);
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
    }
    if goals.is_empty() { goals.push(Goal::Explore); }
    if !goals.contains(&ai.goal) { return None; }

    // Check if we saw a shortcut that would also satisfy the goal.
    if let Some(x) = ai.hints.get(&ai.goal) && known.get(*x).visible() {
        let target = ai.plan.iter().find_map(
            |x| if x.kind == StepKind::Move { Some(x.target) } else { None });
        if let Some(y) = target && AStarLength(pos - *x) < AStarLength(pos - y) {
            let los = LOS(pos, *x);
            let check = |p: Point| { known.get(p).status().unwrap_or(Status::Free) };
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
        StepKind::Drink => point_matches_goal(Goal::Drink, target),
        StepKind::Eat => point_matches_goal(Goal::Eat, target),
        StepKind::Look => true,
        StepKind::Move => match known.get(target).status().unwrap_or(Status::Free) {
            Status::Occupied => target != next.target,
            Status::Blocked  => false,
            Status::Free     => true,
        }
    };
    if !ai.plan.iter().all(|x| step_valid(*x)) { return None; }

    // The plan is good! Execute the next step.
    ai.plan.pop();
    let wait = Some(Action::Idle);
    match next.kind {
        StepKind::Drink => { ai.till_thirst = MAX_THIRST; wait }
        StepKind::Eat => { ai.till_hunger = MAX_HUNGER; wait }
        StepKind::Look => {
            if ai.plan.is_empty() && ai.goal == Goal::Assess {
                ai.till_assess = rng.gen::<i32>().rem_euclid(MAX_ASSESS);
            }
            wait
        }
        StepKind::Move => {
            let mut target = next.target;
            for next in ai.plan.iter().rev().take(8) {
                if next.kind == StepKind::Look { break; }
                if LOS(pos, next.target).iter().all(|x| !known.get(*x).blocked()) {
                    target = next.target;
                }
            }
            let look = if target == pos { entity.dir } else { target - pos };
            Some(Action::Move(Move { look, step: dir }))
        }
    }
}

fn plan_npc(entity: &Entity, ai: &mut AIState, rng: &mut RNG) -> Action {
    let hints = [
        (Goal::Drink, Tile::get('~')),
        (Goal::Eat, Tile::get('%')),
    ];
    ai.update(entity, &hints);
    if let Some(x) = plan_cached(entity, &hints, ai, rng) { return x; }

    ai.plan.clear();
    ai.goal = Goal::Explore;
    ai.debug_targets.clear();

    let (known, pos) = (&*entity.known, entity.pos);

    let check = |p: Point| {
        if p == pos { return Status::Free; }
        known.get(p).status().unwrap_or(Status::Free)
    };

    let mut result = {
        let mut result = BFSResult::default();
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
        if ai.till_thirst == 0 { add_candidates(ai, Goal::Drink); }
        if ai.till_hunger == 0 { add_candidates(ai, Goal::Eat); }

        if result.dirs.is_empty() && ai.till_assess == 0 {
            (ai.goal, result) = (Goal::Assess, coerce(pos, &[]));
        } else if result.dirs.is_empty() && let Some(x) = explore(entity) {
            (ai.goal, result) = (Goal::Explore, x);
        }
        result
    };

    let fallback = |result: BFSResult, rng: &mut RNG| {
        let dirs = &result.dirs;
        let dirs = if dirs.is_empty() { dirs::ALL.as_slice() } else { dirs };
        step(*sample(dirs, rng))
    };

    if result.targets.is_empty() { return fallback(result, rng); }

    ai.debug_targets = result.targets.clone();
    let mut target = *result.targets.select_nth_unstable_by_key(
        0, |x| (*x - pos).len_l2_squared()).1;
    if ai.goal == Goal::Explore {
        for _ in 0..EXPLORE_FUZZ {
            let candidate = target + *sample(&dirs::ALL, rng);
            if known.get(candidate).unknown() { target = candidate; }
        }
    }

    if let Some(path) = AStar(pos, target, ASTAR_LIMIT_WANDER, check) {
        let kind = StepKind::Move;
        ai.plan = path.into_iter().map(|x| Step { kind, target: x }).collect();
        match ai.goal {
            Goal::Assess => {
                for _ in 0..ASSESS_STEPS {
                    ai.plan.push(Step { kind: StepKind::Look, target });
                }
            }
            Goal::Chase => {}
            Goal::Drink => ai.plan.push(Step { kind: StepKind::Drink, target }),
            Goal::Eat => ai.plan.push(Step { kind: StepKind::Eat, target }),
            Goal::Explore => {}
            Goal::Flee => {}
        }
        ai.plan.reverse();
        if let Some(x) = plan_cached(entity, &hints, ai, rng) { return x; }
        ai.plan.clear();
    }
    fallback(result, rng)
}

//////////////////////////////////////////////////////////////////////////////

// Action

struct Move { look: Point, step: Point }

enum Action {
    Idle,
    Move(Move),
    WaitForInput,
}

struct ActionResult {
    success: bool,
}

impl ActionResult {
    fn failure() -> Self { Self { success: false } }
    fn success() -> Self { Self { success: true } }
}

fn step(dir: Point) -> Action {
    Action::Move(Move { look: dir, step: dir })
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
        Action::Move(Move { look, step }) => {
            let entity = &mut state.board.entities[eid];
            if look != Point::default() { entity.dir = look; }
            if step == Point::default() { return ActionResult::success(); }

            let target = entity.pos + step;
            match state.board.get_status(target) {
                Status::Blocked => {
                    state.board.entities[eid].dir = step;
                    ActionResult::failure()
                }
                Status::Occupied => {
                    state.board.entities[eid].dir = step;
                    ActionResult::failure()
                }
                Status::Free => {
                    state.board.move_entity(eid, target);
                    ActionResult::success()
                }
            }
        }
        Action::Idle => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
    }
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
    let dir = if let Input::Char(x) = input { get_direction(x) } else { None };
    //if let Some(x) = dir {
    //    let old = state.board.entities[state.player].dir;
    //    state.board.entities[state.player].dir = x + old;
    //    state.board.update_known(state.player);
    //};
    if let Some(x) = dir { state.input = step(x); }
}

fn update_state(state: &mut State) {
    state.frame += 1;
    let pos = state.get_player().pos;
    state.board.update_env(state.frame, pos, &mut state.rng);

    let needs_input = |state: &State| {
        if !matches!(state.input, Action::WaitForInput) { return false; }
        state.board.get_active_entity() == state.player
    };

    while !state.inputs.is_empty() && needs_input(state) {
        let input = state.inputs.remove(0);
        process_input(state, input);
    }

    let mut update = false;

    loop {
        let eid = state.board.get_active_entity();
        let player = eid == state.player;
        if player && needs_input(state) { break; }

        state.board.update_known(eid);

        update = true;
        let action = plan(state, eid);
        let result = act(state, eid, action);
        if player && !result.success { break; }

        state.board.advance_entity();
    }

    if update {
        state.board.update_known(state.player);
        state.board.update_known(state.board.entity_order[1]);
    }
}

//////////////////////////////////////////////////////////////////////////////

// State

pub struct State {
    board: Board,
    frame: usize,
    input: Action,
    inputs: Vec<Input>,
    player: EID,
    rng: RNG,
    // Update fields
    ai: Option<Box<AIState>>,
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
            if !board.get_tile(pos).blocked() { break; }
        }
        let input = Action::WaitForInput;
        let glyph = Glyph::wdfg('@', 0x222);
        let args = EntityArgs { glyph, player: true, pos };
        let player = board.add_entity(&args, &mut rng);

        let die = |n: i32, rng: &mut RNG| rng.gen::<i32>().rem_euclid(n);
        let pos = |board: &Board, rng: &mut RNG| {
            for _ in 0..100 {
                let p = Point(die(size.0, rng), die(size.1, rng));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        for _ in 0..20 {
            if let Some(x) = pos(&board, &mut rng) {
                let glyph = Glyph::wdfg('P', 0x222);
                let args = EntityArgs { glyph, player: false, pos: x };
                board.add_entity(&args, &mut rng);
            }
        }
        board.entities[player].dir = Point::default();
        board.update_known(player);

        let ai = Some(Box::new(AIState::new(&mut rng)));
        Self { board, frame: 0, input, inputs: vec![], player, rng, ai }
    }

    fn get_player(&self) -> &Entity { &self.board.entities[self.player] }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Buffer) {
        if buffer.data.is_empty() {
            let size = Point(2 * UI_MAP_SIZE_X, UI_MAP_SIZE_Y);
            let mut overwrite = Matrix::new(size, ' '.into());
            std::mem::swap(buffer, &mut overwrite);
        }

        let entity = self.get_player();
        //let entity = &self.board.entities[self.board.entity_order[1]];
        let offset = entity.pos - Point(UI_MAP_SIZE_X / 2, UI_MAP_SIZE_Y / 2);
        let unseen = Glyph::wide(' ');
        let known = &*entity.known;

        let lookup = |point: Point| -> Glyph {
            let cell = known.get(point);
            let Some(tile) = cell.tile() else { return unseen; };
            let glyph = cell.entity().map(|x| x.glyph).unwrap_or(tile.glyph);
            let shade = cell.shade() || !cell.visible();
            if shade { glyph.with_fg(Color::gray()) } else { glyph }
        };

        buffer.fill(Glyph::wide(' '));
        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let glyph = lookup(Point(x, y) + offset);
                buffer.set(Point(2 * x, y), glyph);
            }
        }

        let length = 3 as usize;
        let mut arrows = vec![];
        for other in &entity.known.entities {
            if other.friend || other.age > 0 { continue; }

            let (pos, dir) = (other.pos, other.dir);
            let ch = Glyph::ray(dir);
            let norm = length as f64 / dir.len_l2();
            let diff = Point((dir.0 as f64 * norm) as i32,
                             (dir.1 as f64 * norm) as i32);
            arrows.push((ch, LOS(pos, pos + diff)));
        }
        for (ch, arrow) in &arrows {
            let index = (self.frame / 2) % (8 * length);
            if let Some(x) = arrow.get(index + 1) {
                let point = Point(2 * (x.0 - offset.0), x.1 - offset.1);
                buffer.set(point, Glyph::wide(*ch));
            }
        }

        for step in &entity.ai.plan {
            if step.kind == StepKind::Look { continue; }
            let Point(x, y) = step.target - offset;
            let point = Point(2 * x, y);
            let mut glyph = buffer.get(point);
            if glyph.ch() == Glyph::wide(' ').ch() { glyph = Glyph::wide('.'); }
            buffer.set(point, glyph.with_fg(0x400));
        }

        for target in &entity.ai.debug_targets {
            let Point(x, y) = *target - offset;
            let point = Point(2 * x, y);
            let glyph = buffer.get(point);
            buffer.set(point, glyph.with_fg(Color::black()).with_bg(0x400));
        }

        if let Some(rain) = &self.board.rain {
            let base = Tile::get('~').glyph.fg();
            for drop in &rain.drops {
                let index = drop.frame - self.frame;
                let Some(delta) = rain.path.get(index) else { continue; };
                let point = drop.point - *delta;
                let cell = if index == 0 { known.get(point) } else { known.default() };
                if index == 0 && !cell.visible() { continue; }

                let Point(x, y) = point - offset;
                let ch = if index == 0 { 'o' } else { rain.ch };
                let color = if cell.shade() { Color::gray() } else { base };
                let glyph = Glyph::wdfg(ch, color);
                buffer.set(Point(2 * x, y), glyph);
            }

            if rain.lightning > 0 {
                let color = Color::from(0x111 * (rain.lightning / 2));
                for y in 0..UI_MAP_SIZE_Y {
                    for x in 0..UI_MAP_SIZE_X {
                        let point = Point(2 * x, y);
                        buffer.set(point, buffer.get(point).with_bg(color));
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
                            buffer.set(point + delta, buffer.get(point));
                        }
                        buffer.set(Point(0, y), space);
                        buffer.set(Point(2 * UI_MAP_SIZE_X - 1, y), space);
                    }
                }
            }
        }
    }
}
