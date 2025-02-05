use std::cmp::{max, min};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::mem::{replace, swap};

use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use thin_vec::{ThinVec, thin_vec};

use crate::static_assert_size;
use crate::ai::{AIEnv, AIState};
use crate::base::{Buffer, Color, Glyph, Rect, Slice};
use crate::base::{HashMap, LOS, Matrix, Point, RNG, dirs};
use crate::effect::{Effect, Event, Frame, FT, self};
use crate::entity::{EID, Entity, EntityArgs, EntityMap};
use crate::knowledge::{Knowledge, Timestamp, Vision, VisionArgs};
use crate::mapgen::mapgen_with_size;
use crate::pathing::Status;

//////////////////////////////////////////////////////////////////////////////

// Constants

const MOVE_TIMER: i32 = 960;
const TURN_TIMER: i32 = 120;

const FOV_RADIUS_NPC: i32 = 12;
const FOV_RADIUS_PC_: i32 = 21;

const SPEED_PC_: f64 = 0.1;
const SPEED_NPC: f64 = 0.1;

const LIGHT: Light = Light::Sun(Point(4, 1));
const WEATHER: Weather = Weather::None;
const WORLD_SIZE: i32 = 100;
const NUM_PREDATORS: i32 = 2;
const NUM_PREY: i32 = 18;

const FULL_VIEW: bool = true;
const UI_MAP_SIZE: i32 = if FULL_VIEW { WORLD_SIZE } else { 2 * FOV_RADIUS_PC_ + 1 };

const UI_DAMAGE_FLASH: i32 = 6;
const UI_DAMAGE_TICKS: i32 = 6;

const UI_COLOR: i32 = 0x430;
const UI_MAP_SIZE_X: i32 = UI_MAP_SIZE;
const UI_MAP_SIZE_Y: i32 = UI_MAP_SIZE;

const UI_MOVE_ALPHA: f64 = 0.75;
const UI_MOVE_FRAMES: i32 = 12;
const UI_MAP_MEMORY: usize = 32;

const UI_SHADE_FADE: f64 = 0.30;
const UI_REMEMBERED: f64 = 0.15;

pub const NOISY_RADIUS: i32 = 4;

#[derive(Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char) }

//////////////////////////////////////////////////////////////////////////////

// Tile

const FLAG_BLOCKS_VISION: u32 = 1 << 0;
const FLAG_LIMITS_VISION: u32 = 1 << 1;
const FLAG_BLOCKS_MOVEMENT: u32 = 1 << 2;
const FLAG_CAN_DRINK: u32 = 1 << 3;
const FLAG_CAN_EAT: u32 = 1 << 4;

const FLAGS_NONE: u32 = 0;
const FLAGS_BLOCKED: u32 = FLAG_BLOCKS_MOVEMENT | FLAG_BLOCKS_VISION;
const FLAGS_FRESH_WATER: u32 = FLAG_BLOCKS_MOVEMENT | FLAG_CAN_DRINK;
const FLAGS_BERRY_TREE: u32 = FLAG_BLOCKS_MOVEMENT | FLAG_LIMITS_VISION | FLAG_CAN_EAT;

pub struct Tile {
    pub flags: u32,
    pub glyph: Glyph,
    pub description: &'static str,
}
#[cfg(target_pointer_width = "32")]
static_assert_size!(Tile, 24);
#[cfg(target_pointer_width = "64")]
static_assert_size!(Tile, 32);

impl Tile {
    pub fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    pub fn try_get(ch: char) -> Option<&'static Tile> { TILES.get(&ch) }

    // Raw flags-based predicates.
    pub fn can_eat(&self) -> bool { self.flags & FLAG_CAN_EAT != 0 }
    pub fn can_drink(&self) -> bool { self.flags & FLAG_CAN_DRINK != 0 }
    pub fn blocks_vision(&self) -> bool { self.flags & FLAG_BLOCKS_VISION != 0 }
    pub fn limits_vision(&self) -> bool { self.flags & FLAG_LIMITS_VISION != 0 }
    pub fn blocks_movement(&self) -> bool { self.flags & FLAG_BLOCKS_MOVEMENT != 0 }

    // Derived predicates.
    pub fn casts_shadow(&self) -> bool { self.blocks_vision() }
}

impl Debug for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", (self.glyph.ch().0 - 0xff00 + 0x20) as u8 as char)
    }
}

impl Eq for &'static Tile {}

impl PartialEq for &'static Tile {
    fn eq(&self, next: &&'static Tile) -> bool {
        *self as *const Tile == *next as *const Tile
    }
}

lazy_static! {
    static ref TILES: HashMap<char, Tile> = {
        let items = [
            ('#', (FLAGS_BLOCKED,      Glyph::wdfg('#', (16, 96, 0)),     "a tree")),
            ('.', (FLAGS_NONE,         Glyph::wdfg('.', (255, 255, 255)), "grass")),
            ('"', (FLAG_LIMITS_VISION, Glyph::wdfg('"', (96, 192, 0)),    "tall grass")),
            ('~', (FLAGS_FRESH_WATER,  Glyph::wdfg('~', (0, 128, 255)),   "water")),
            ('B', (FLAGS_BERRY_TREE,   Glyph::wdfg('#', (192, 128, 0)),   "a berry tree")),
            ('=', (FLAGS_NONE,         Glyph::wdfg('=', (255, 128, 0)),   "a bridge")),
            ('R', (FLAGS_NONE,         Glyph::wdfg('.', (255, 128, 0)),   "a path")),
        ];
        let mut result = HashMap::default();
        for (ch, (flags, glyph, description)) in items {
            result.insert(ch, Tile { flags, glyph, description });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Item

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Item { Berry, Corpse }

fn show_item(item: &Item) -> Glyph {
    match item {
        Item::Berry => Glyph::wdfg('*', (192, 128, 0)),
        Item::Corpse => Glyph::wdfg('%', (255, 255, 255)),
    }
}

//////////////////////////////////////////////////////////////////////////////

// Environment

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

//////////////////////////////////////////////////////////////////////////////

// Board

#[derive(Clone)]
pub struct Cell {
    pub eid: Option<EID>,
    pub items: ThinVec<Item>,
    pub shadow: i32,
    pub tile: &'static Tile,
}
#[cfg(target_pointer_width = "32")]
static_assert_size!(Cell, 20);
#[cfg(target_pointer_width = "64")]
static_assert_size!(Cell, 32);

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
        let cell = Cell { eid: None, items: thin_vec![], shadow: 0, tile: Tile::get('#') };

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
        result.reset(Tile::get('.'));
        result
    }

    // Animation

    fn add_effect(&mut self, effect: Effect, rng: &mut RNG) {
        let mut existing = Effect::default();
        std::mem::swap(&mut self._effect, &mut existing);
        self._effect = existing.and(effect);
        self._execute_effect_callbacks(rng);
    }

    fn advance_effect(&mut self, pov: Option<EID>, rng: &mut RNG) -> bool {
        let mut visible = self._pov_sees_effect(pov);
        while self._advance_one_frame(rng) {
            visible = visible || self._pov_sees_effect(pov);
            if visible { return true; }
        }
        false
    }

    fn _advance_one_frame(&mut self, rng: &mut RNG) -> bool {
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

    fn _pov_sees_effect(&self, pov: Option<EID>) -> bool {
        if self._effect.frames.is_empty() { return false; }

        let eid = (|| {
            if let Some(x) = pov && self.entities.has(x) { return x; }
            let result = self.entity_order[0];
            assert!(self.entities[result].player);
            result
        })();

        let frame = &self._effect.frames[0];
        let known = &self.entities[eid].known;
        frame.iter().any(|y| known.get(y.point).visible())
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

    // Item setters

    fn add_item(&mut self, pos: Point, item: Item) {
        let Some(cell) = self.map.entry_mut(pos) else { return };
        cell.items.push(item);
    }

    fn remove_item(&mut self, pos: Point, item: Item) -> bool {
        let Some(cell) = self.map.entry_mut(pos) else { return false };
        let Some(index) = cell.items.iter().position(|&x| x == item) else { return false };
        cell.items.remove(index);
        true
    }

    // Entity setters

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

        // Remove the entity from the spatial map.
        let existing = self.map.entry_mut(entity.pos).unwrap().eid.take();
        assert!(existing == Some(eid));

        // Remove the entity from the entities SlotMap.
        let okay = self.entities.remove(eid).is_some();
        assert!(okay);

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
        self.map.fill(Cell { eid: None, items: thin_vec![], shadow: 0, tile });
        self.update_edge_shadows();
        self.entity_order.clear();
        self.active_entity_index = 0;
        self.entities.clear();
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
        let Entity { pos, dir, asleep, player, .. } = *me;
        let vision = if player { &mut self._pc_vision } else { &mut self.npc_vision };
        if asleep {
            vision.clear(pos);
        } else {
            let args = VisionArgs { player, pos, dir };
            vision.compute(&args, |x| self.map.get(x).tile);
        }
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
        known.update_entity(me, other, /*seen=*/false, /*heard=*/heard);

        swap(&mut known, &mut self.entities[eid].known);
        self.known = Some(known);
    }

    fn remove_known_entity(&mut self, eid: EID, oid: EID) {
        self.entities[eid].known.remove_entity(oid);
    }

    // Environmental effects

    fn update_env(&mut self, frame: usize, pos: Point, rng: &mut RNG) {
        let Some(rain) = &mut self.rain else { return; };

        while let Some(x) = rain.drops.front() && x.frame < frame {
            rain.drops.pop_front();
        }
        let total = rain.drops.capacity();
        let denom = max(rain.diff.1, 1);
        let delta = denom as usize;
        let extra = (frame + 1) * total / delta - (frame * total) / delta;
        for _ in 0..min(extra, total - rain.drops.len()) {
            let x = rng.gen_range(0..denom);
            let y = rng.gen_range(0..denom);
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

// Action

pub struct EatAction { pub point: Point, pub item: Option<Item> }

pub struct MoveAction { pub look: Point, pub step: Point, pub turns: f64 }

pub enum Action {
    Idle,
    Rest,
    WaitForInput,
    Look(Point),
    Move(MoveAction),
    Attack(Point),
    Drink(Point),
    Eat(EatAction),
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

pub fn move_ready(entity: &Entity) -> bool { entity.move_timer <= 0 }

pub fn turn_ready(entity: &Entity) -> bool { entity.turn_timer <= 0 }

fn charge(entity: &mut Entity) {
    let charge = (TURN_TIMER as f64 * entity.speed).round() as i32;
    if entity.move_timer > 0 { entity.move_timer -= charge; }
    if entity.turn_timer > 0 { entity.turn_timer -= charge; }
}

fn drain(entity: &mut Entity, result: &ActionResult) {
    entity.move_timer += (MOVE_TIMER as f64 * result.moves).round() as i32;
    entity.turn_timer += (TURN_TIMER as f64 * result.turns).round() as i32;
}

fn can_attack(board: &Board, entity: &Entity, target: Point, range: i32) -> bool {
    let (known, source) = (&entity.known, entity.pos);
    if (source - target).len_nethack() > range { return false; }
    if !known.get(target).visible() { return false; }
    if source == target { return false; }

    let los = LOS(source, target);
    let last = los.len() - 1;
    los.iter().enumerate().all(|(i, &p)| {
        if i == 0 || i == last { return true; }
        known.get(p).status() == Status::Free && board.get_status(p) == Status::Free
    })
}

fn plan(state: &mut State, eid: EID) -> Action {
    let player = eid == state.player;
    if player { return replace(&mut state.input, Action::WaitForInput); }

    let entity = &mut state.board.entities[eid];
    let mut debug = None;
    let mut ai = state.ai.take().unwrap_or_else(
        || Box::new(AIState::new(false, &mut state.rng)));
    swap(&mut debug, &mut entity.debug);
    swap(&mut ai, &mut entity.ai);

    let mut env = AIEnv { rng: &mut state.rng, debug };
    let entity = &state.board.entities[eid];
    let action = ai.plan(entity, &mut env);

    let entity = &mut state.board.entities[eid];
    swap(&mut env.debug, &mut entity.debug);
    swap(&mut ai, &mut entity.ai);
    state.ai = Some(ai);
    action
}

fn act(state: &mut State, eid: EID, action: Action) -> ActionResult {
    let entity = &mut state.board.entities[eid];
    entity.asleep = matches!(action, Action::Rest);

    match action {
        Action::Idle => ActionResult::success(),
        Action::Rest => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::Look(dir) => {
            state.board.entities[eid].dir = dir;
            ActionResult::success()
        }
        Action::Drink(point) => {
            let entity = &mut state.board.entities[eid];
            if (entity.pos - point).len_l1() > 1 { return ActionResult::failure(); }

            if entity.pos != point { entity.dir = point - entity.pos; }
            let okay = state.board.get_cell(point).tile.can_drink();
            if !okay { return ActionResult::failure(); }

            let board = &mut state.board;
            let cb = Box::new(|_: &mut Board, _: &mut RNG| {});
            board.add_effect(apply_flash(board, point, 0x004, cb), &mut state.rng);
            ActionResult::success()
        }
        Action::Eat(EatAction { point, item }) => {
            let entity = &mut state.board.entities[eid];
            if (entity.pos - point).len_l1() > 1 { return ActionResult::failure(); }

            if entity.pos != point { entity.dir = point - entity.pos; }
            let cell = state.board.get_cell(point);
            let okay = match item {
                Some(x) => cell.items.iter().find(|&&y| y == x).is_some(),
                None => cell.tile.can_eat(),
            };
            if !okay { return ActionResult::failure(); }

            let board = &mut state.board;
            let color = if item.is_some() { 0x400 } else { 0x440 };
            let cb = Box::new(move |board: &mut Board, _: &mut RNG| {
                let Some(item) = item else { return };
                board.remove_item(point, item);
            });
            board.add_effect(apply_flash(board, point, color, cb), &mut state.rng);
            ActionResult::success()
        }
        Action::Move(MoveAction { look, step, turns }) => {
            let entity = &mut state.board.entities[eid];
            if look != dirs::NONE { entity.dir = look; }
            if step == dirs::NONE { return ActionResult::success_turns(turns); }
            if step.len_l1() > 1 { return ActionResult::failure(); }

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
                    let max = if noisy { NOISY_RADIUS } else { 1 };
                    for &oid in &state.board.entity_order {
                        if oid == eid { continue; }
                        let other = &state.board.entities[oid];
                        if other.asleep && !noisy { continue; }
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
            let entity = &state.board.entities[eid];
            let (range, source) = (entity.range, entity.pos);
            if !can_attack(&state.board, entity, target, range) {
                return ActionResult::failure();
            }

            let oid = state.board.get_cell(target).eid;
            let entity = &mut state.board.entities[eid];
            entity.dir = target - source;

            let cb = move |board: &mut Board, rng: &mut RNG| {
                let Some(oid) = oid else { return; };
                let cb = move |board: &mut Board, _: &mut RNG| {
                    let Some(other) = board.entities.get_mut(oid) else { return; };

                    let damage = 1;
                    if other.cur_hp > damage {
                        other.cur_hp -= damage;
                        board.update_known_entity(oid, eid, /*heard=*/true);
                    } else {
                        let pos = other.pos;
                        board.remove_entity(oid);
                        board.add_item(pos, Item::Corpse);
                        board.remove_known_entity(eid, oid);
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
//
// TODO(skishore): All this code is WRONG! It may reveal things (e.g. hidden
// entities) to the player that they should not be able to see.
//
// To fix these bugs, we need to audit all usage of Board getters in Effects,
// and instead of directly overwriting cells, we need to apply updates (e.g.
// moving Entities) and use the player's FOV resolve what they can see.
//
// We also need to figure out if items are "hidden" the same way entities are.

type CB = Box<dyn Fn(&mut Board, &mut RNG)>;

fn apply_flash<T: Into<Color>>(board: &Board, target: Point, color: T, callback: CB) -> Effect {
    let cell = board.get_cell(target);
    let glyph = if let Some(x) = cell.eid {
        board.entities[x].glyph
    } else if let Some(x) = cell.items.last() {
        show_item(x)
    } else {
        cell.tile.glyph
    };

    let flash = glyph.with_fg(Color::black()).with_bg(color.into());
    let particle = effect::Particle { glyph: flash, point: target };
    let mut effect = Effect::constant(particle, UI_DAMAGE_FLASH);
    let frame = effect.frames.len() as i32;
    effect.add_event(Event::Callback { frame, callback });
    effect
}

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
        state.input = Action::Move(MoveAction { look: dir, step: dir, turns: 1. });
    }
}

fn update_pov_entities(state: &mut State) {
    state.board.update_known(state.player);
    if let Some(x) = state.pov && state.board.entities.has(x) {
        state.board.update_known(x);
    }
}

fn update_state(state: &mut State) {
    state.frame += 1;
    let pos = state.get_player().pos;
    state.board.update_env(state.frame, pos, &mut state.rng);

    for x in state.moves.values_mut() { x.frame += 1; }
    state.moves.retain(|_, v| v.frame < v.limit);

    if state.board.advance_effect(state.pov, &mut state.rng) {
        update_pov_entities(state);
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
        let pos = entity.pos;

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

        if player && state.get_player().pos != pos {
            let time = state.get_player().known.time;
            let times = &mut state.turn_times;
            if times.len() == UI_MAP_MEMORY { times.pop_back(); }
            times.push_front(time);
        }

        //state.board.update_known(eid);
        //state.board.update_known(state.player);

        if let Some(x) = state.board.entities.get_mut(eid) { drain(x, &result); }
    }

    if update { update_pov_entities(state); }
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
    turn_times: VecDeque<Timestamp>,
}

impl State {
    pub fn new(seed: Option<u64>) -> Self {
        let size = Point(WORLD_SIZE, WORLD_SIZE);
        let pos = Point(size.0 / 2, size.1 / 2);
        let rng = seed.map(|x| RNG::seed_from_u64(x));
        let mut rng = rng.unwrap_or_else(|| RNG::from_entropy());
        let mut board = Board::new(size, LIGHT, WEATHER);

        loop {
            let map = mapgen_with_size(size, &mut rng);
            for x in 0..size.0 {
                for y in 0..size.1 {
                    let p = Point(x, y);
                    board.set_tile(p, Tile::get(map.get(p)));
                }
            }
            if !board.get_tile(pos).blocks_movement() { break; }
        }

        let input = Action::WaitForInput;
        let glyph = Glyph::wdfg('@', 0x222);
        let (player, speed) = (true, SPEED_PC_);
        let args = EntityArgs { glyph, player, predator: false, pos, speed };
        let player = board.add_entity(&args, &mut rng);

        let pos = |board: &Board, rng: &mut RNG| {
            for _ in 0..100 {
                let p = Point(rng.gen_range(0..size.0), rng.gen_range(0..size.1));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        for i in 0..(NUM_PREDATORS + NUM_PREY) {
            if let Some(x) = pos(&board, &mut rng) {
                let predator = i < NUM_PREDATORS;
                let (player, speed) = (false, SPEED_NPC);
                let glyph = Glyph::wdfg(if predator { 'R' } else { 'P' }, 0x222);
                let args = EntityArgs { glyph, player, predator, pos: x, speed };
                board.add_entity(&args, &mut rng);
            }
        }
        board.entities[player].dir = Point::default();
        board.update_known(player);

        let inputs = vec![];
        let moves = Default::default();
        let ai = Some(Box::new(AIState::new(false, &mut rng)));
        let turn_times = [Timestamp::default()].into_iter().collect();
        Self { board, frame: 0, input, inputs, player, pov: None, rng, ai, moves, turn_times }
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
        let offset = if FULL_VIEW { Point::default() } else { offset };

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
            *debug = format!("{:?}", entity.ai);
            let debug = entity.debug.as_ref();
            let path = entity.ai.get_path();
            let target = path.first();
            let slice_point = |p: Point| Point(2 * (p.0 - offset.0), p.1 - offset.1);

            for &p in path.iter().skip(1) {
                let point = slice_point(p);
                let mut glyph = slice.get(point);
                if glyph.ch() == Glyph::wide(' ').ch() { glyph = Glyph::wide('.'); }
                slice.set(point, glyph.with_fg(0x400));
            }
            for &(p, score) in debug.map(|x| x.utility.as_slice()).unwrap_or(&[]) {
                let point = slice_point(p);
                let glyph = slice.get(point);
                slice.set(point, glyph.with_bg(Color::gray(score)));
            }
            if let Some(&p) = target {
                let point = slice_point(p);
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

                let (ground, point) = (index == 0, drop.point - delta);
                let cell = if ground { known.get(point) } else { known.default() };
                if ground && !cell.visible() { continue; }

                let Point(x, y) = point - offset;
                let ch = if index == 0 { 'o' } else { rain.ch };
                let shade = cell.shade();
                let color = if shade { base.fade(UI_SHADE_FADE) } else { base };
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

            let age_in_turns = (|| {
                if !player { return 0; }
                let age = cell.time_since_seen();
                for (turn, &turn_time) in self.turn_times.iter().enumerate() {
                    if age <= known.time - turn_time { return turn; }
                }
                return UI_MAP_MEMORY;
            })();
            if age_in_turns >= max(UI_MAP_MEMORY, 1) { return unseen; }

            let see_entity = cell.can_see_entity_at();
            let obscured = tile.limits_vision();
            let shadowed = cell.shade();

            let glyph = if see_entity && let Some(x) = cell.entity() {
                if obscured { x.glyph.with_fg(tile.glyph.fg()) } else { x.glyph }
            } else if let Some(x) = cell.items().last() {
                show_item(x)
            } else {
                tile.glyph
            };
            let mut color = glyph.fg();

            if !cell.visible() {
                let limit = max(UI_MAP_MEMORY, 1) as f64;
                let delta = (UI_MAP_MEMORY - age_in_turns) as f64;
                color = Color::white().fade(UI_REMEMBERED * delta / limit);
            } else if shadowed {
                color = color.fade(UI_SHADE_FADE);
            }
            glyph.with_fg(color)
        };

        // Render all currently-visible cells.
        slice.fill(Glyph::wide(' '));
        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let glyph = lookup(Point(x, y) + offset);
                slice.set(Point(2 * x, y), glyph);
            }
        }

        // Render ephemeral state: sounds we've heard and moves we've glimpsed.
        for entity in &known.entities {
            if !entity.heard { continue; }
            let Point(x, y) = entity.pos - offset;
            slice.set(Point(2 * x, y), Glyph::wide('?'));
        }
        if player {
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
                if !known.get(point).visible() { continue; }
                let Point(x, y) = point - offset;
                slice.set(Point(2 * x, y), glyph);
            }
        }

        // If we're still playing, render arrows showing NPC facing.
        if entity.cur_hp > 0 { self.render_arrows(known, offset, slice); }
    }

    fn render_arrows(&self, known: &Knowledge, offset: Point, slice: &mut Slice) {
        let arrow_length = 3;
        let sleep_length = 2;
        let mut arrows = vec![];
        for other in &known.entities {
            if other.friend || other.age > 0 { continue; }

            let (pos, dir) = (other.pos, other.dir);
            let mut ch = Glyph::ray(dir);
            let mut diff = dir.normalize(arrow_length as f64);
            if other.asleep { (ch, diff) = ('Z', Point(0, -sleep_length)); }
            arrows.push((ch, LOS(pos, pos + diff)));
        }

        for (ch, arrow) in &arrows {
            let speed = if *ch == 'Z' { 8 } else { 2 };
            let denom = if *ch == 'Z' { sleep_length } else { arrow_length };
            let index = (self.frame / speed) % (8 * denom as usize);
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

    const BASE_SEED: u64 = 17;
    const NUM_SEEDS: u64 = 8;

    #[bench]
    fn bench_state_update(b: &mut test::Bencher) {
        let mut index = 0;
        let mut states = vec![];
        for i in 0..NUM_SEEDS { states.push(State::new(Some(BASE_SEED + i))); }

        b.iter(|| {
            let i = index % states.len();
            let state = &mut states[i];
            index += 1;

            state.inputs.push(Input::Char('.'));
            state.update();
            while state.board.get_frame().is_some() { state.update(); }
        });
    }
}
