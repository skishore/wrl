use std::fmt::Debug;
use std::mem::{replace, swap};

use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use thin_vec::{ThinVec, thin_vec};

use crate::static_assert_size;
use crate::ai::{AIEnv, AIState};
use crate::base::{Buffer, Color, Glyph};
use crate::base::{HashMap, LOS, Matrix, Point, RNG, dirs};
use crate::effect::{Effect, Frame, FT, self};
use crate::entity::{EID, Entity, EntityArgs, EntityMap};
use crate::knowledge::{Knowledge, Scent, Timedelta, Timestamp};
use crate::knowledge::{EventData, MoveEvent, Sense, Event};
use crate::mapgen::mapgen_with_size as mapgen;
use crate::pathing::Status;
use crate::shadowcast::{INITIAL_VISIBILITY, VISIBILITY_LOSSES, Vision, VisionArgs};
use crate::ui::{UI, get_direction};

//////////////////////////////////////////////////////////////////////////////

// Constants

pub const MOVE_TIMER: i32 = 960;
pub const TURN_TIMER: i32 = 120;
pub const WORLD_SIZE: i32 = 100;

pub const FOV_RADIUS_NPC: i32 = 12;
pub const FOV_RADIUS_PC_: i32 = 21;

const FOV_RADIUS_IN_TALL_GRASS: usize = 4;
const VISIBILITY_LOSS: i32 = VISIBILITY_LOSSES[FOV_RADIUS_IN_TALL_GRASS - 1];

const SPEED_PC_: f64 = 1.;
const SPEED_NPC: f64 = 1.;

const LIGHT: Light = Light::Sun(Point(2, 0));
const WEATHER: Weather = Weather::None;
const NUM_PREDATORS: i32 = 5;
const NUM_PREY: i32 = 15;

const UI_DAMAGE_FLASH: i32 = 6;
const UI_DAMAGE_TICKS: i32 = 6;

pub const NOISY_RADIUS: i32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
const FLAGS_TALL_GRASS: u32 = FLAG_LIMITS_VISION;

pub struct Tile {
    pub flags: u32,
    pub glyph: Glyph,
    pub description: &'static str,
}
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

    pub fn opacity(&self) -> i32 {
        if self.blocks_vision() { return INITIAL_VISIBILITY; }
        if self.limits_vision() { return VISIBILITY_LOSS; }
        0
    }
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
            ('#', (FLAGS_BLOCKED,     Glyph::wdfg('#', 0x106000), "a tree")),
            ('.', (FLAGS_NONE,        Glyph::wdfg('.', 0xe0ffc0), "grass")),
            (',', (FLAGS_NONE,        Glyph::wdfg('`', 0x60c060), "weeds")),
            ('"', (FLAGS_TALL_GRASS,  Glyph::wdfg('"', 0x60c000), "tall grass")),
            ('|', (FLAGS_TALL_GRASS,  Glyph::wdfg('|', 0x60c000), "reeds")),
            ('+', (FLAGS_NONE,        Glyph::wdfg('+', 0xff6060), "a flower")),
            ('~', (FLAGS_FRESH_WATER, Glyph::wdfg('~', 0x0080ff), "water")),
            ('B', (FLAGS_BERRY_TREE,  Glyph::wdfg('#', 0xc08000), "a berry tree")),
            ('=', (FLAGS_NONE,        Glyph::wdfg('=', 0xff8000), "a bridge")),
            ('R', (FLAGS_NONE,        Glyph::wdfg('.', 0xff8000), "a path")),
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

pub fn show_item(item: &Item) -> Glyph {
    match item {
        Item::Berry  => Glyph::wdfg('*', 0xc08000),
        Item::Corpse => Glyph::wdfg('%', 0xffffff),
    }
}

//////////////////////////////////////////////////////////////////////////////

// Environment

pub enum Light { None, Sun(Point) }

enum Weather { None, Rain(Point, usize) }

//////////////////////////////////////////////////////////////////////////////

// FOV

struct FOV {
    npc_vision: Vision,
    _pc_vision: Vision,
}

impl Default for FOV {
    fn default() -> Self {
        Self {
            npc_vision: Vision::new(FOV_RADIUS_NPC),
            _pc_vision: Vision::new(FOV_RADIUS_PC_),
        }
    }
}

impl FOV {
    fn select_vision(&mut self, entity: &Entity) -> &mut Vision {
        if entity.player { &mut self._pc_vision } else { &mut self.npc_vision }
    }

    fn can_see(&mut self, board: &Board, entity: &Entity, point: Point) -> bool {
        let Entity { pos, dir, asleep, player, .. } = *entity;
        if asleep { return pos == point; }

        let vision = self.select_vision(entity);

        let map = &board.map;
        let dir = if player { Point::default() } else { dir };
        let opacity_lookup = |x| map.get(x).tile.opacity();
        vision.check_point(&VisionArgs { pos, dir, opacity_lookup }, point)
    }

    fn can_see_entity_at(&mut self, board: &Board, entity: &Entity, point: Point) -> bool {
        if !self.can_see(board, entity, point) { return false; }

        let nearby = (point - entity.pos).len_l1() <= 1;
        if nearby { return true; }

        let cell = board.get_cell(point);
        let dark = matches!(board.get_light(), Light::None);
        let shade = dark || cell.shadow > 0;
        nearby || !(shade || cell.tile.limits_vision())
    }

    fn compute(&mut self, board: &Board, entity: &Entity) -> &Vision {
        let vision = self.select_vision(entity);
        let Entity { pos, dir, asleep, player, .. } = *entity;
        if asleep {
            vision.clear(pos);
        } else {
            let map = &board.map;
            let dir = if player { Point::default() } else { dir };
            let opacity_lookup = |x| map.get(x).tile.opacity();
            vision.compute(&VisionArgs { pos, dir, opacity_lookup });
        }
        vision
    }
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
#[cfg(target_pointer_width = "64")]
static_assert_size!(Cell, 32);

pub struct Board {
    map: Matrix<Cell>,
    active_entity: Option<EID>,
    pub entities: EntityMap,
    pub time: Timestamp,
    _effect: Effect,

    // Environment:
    light: Light,
    shadow: Vec<Point>,
}

impl Board {
    fn new(size: Point, light: Light) -> Self {
        let shadow = match light {
            Light::Sun(x) => LOS(Point::default(), x).into_iter().skip(1).collect(),
            Light::None => vec![],
        };
        let tile = Tile::get('#');
        let cell = Cell { eid: None, items: thin_vec![], shadow: 0, tile };
        let time = Timedelta::from_seconds(365.75 * 86400.);

        let mut result = Self {
            map: Matrix::new(size, cell),
            active_entity: None,
            entities: EntityMap::default(),
            time: Timestamp::default().latch(time),
            _effect: Effect::default(),

            // Environment:
            light,
            shadow,
        };
        result.reset(Tile::get('.'));
        result
    }

    // Animation

    fn add_effect(&mut self, effect: Effect, env: &mut UpdateEnv) {
        let mut existing = Effect::default();
        swap(&mut self._effect, &mut existing);
        self._effect = existing.and(effect);
        self._execute_effect_callbacks(env);
    }

    fn advance_effect(&mut self, pov: EID, env: &mut UpdateEnv) -> bool {
        let mut visible = self.pov_sees_effect(pov);
        while self._advance_one_frame(env) {
            visible = visible || self.pov_sees_effect(pov);
            if visible { return true; }
        }
        false
    }

    fn pov_sees_effect(&self, pov: EID) -> bool {
        let Some(frame) = self._effect.frames.get(0) else { return false };
        let Some(entity) = self.entities.get(pov) else { return false };

        let known = &entity.known;
        frame.iter().any(|y| known.get(y.point).visible())
    }

    fn _advance_one_frame(&mut self, env: &mut UpdateEnv) -> bool {
        if self._effect.frames.is_empty() {
            assert!(self._effect.events.is_empty());
            return false;
        }
        self.time = self.time.bump();
        self._effect.frames.remove(0);
        self._effect.events.iter_mut().for_each(|x| x.update_frame(|y| y - 1));
        self._execute_effect_callbacks(env);
        true
    }

    fn _execute_effect_callbacks(&mut self, env: &mut UpdateEnv) {
        while self._execute_one_effect_callback(env) {}
    }

    fn _execute_one_effect_callback(&mut self, env: &mut UpdateEnv) -> bool {
        if self._effect.events.is_empty() { return false; }
        let event = &self._effect.events[0];
        if !self._effect.frames.is_empty() && event.frame() > 0 { return false; }
        match self._effect.events.remove(0) {
            effect::Event::Callback { callback, .. } => callback(self, env),
            effect::Event::Other { .. } => (),
        }
        true
    }

    // Getters

    pub fn get_cell(&self, p: Point) -> &Cell { self.map.entry_ref(p) }

    pub fn get_entity(&self, eid: EID) -> Option<&Entity> { self.entities.get(eid) }

    pub fn get_frame(&self) -> Option<&Frame> { self._effect.frames.iter().next() }

    pub fn get_light(&self) -> &Light { &self.light }

    pub fn get_size(&self) -> Point { self.map.size }

    pub fn get_status(&self, p: Point) -> Status {
        let Cell { eid, tile, .. } = self.get_cell(p);
        if tile.blocks_movement() { return Status::Blocked; }
        if eid.is_some() { Status::Occupied } else { Status::Free }
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

    fn add_entity(&mut self, args: &EntityArgs, env: &mut UpdateEnv) -> EID {
        let pos = args.pos;
        let eid = self.entities.add(args, &mut env.rng);
        let cell = self.map.entry_mut(pos).unwrap();
        let prev = replace(&mut cell.eid, Some(eid));
        assert!(prev.is_none());
        self.update_known(eid, env);
        eid
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

        // Mark the entity as inactive, if it was active.
        if self.active_entity == Some(eid) { self.active_entity = None; }
    }

    fn reset(&mut self, tile: &'static Tile) {
        self.map.fill(Cell { eid: None, items: thin_vec![], shadow: 0, tile });
        self.update_edge_shadows();
        self.active_entity = None;
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

    fn update_known(&mut self, eid: EID, env: &mut UpdateEnv) {
        let UpdateEnv { known, fov, rng } = env;
        swap(known, &mut self.entities[eid].known);

        let me = &self.entities[eid];
        let vision = fov.compute(&self, me);
        known.update(me, &self, vision, rng);

        swap(known, &mut self.entities[eid].known);
    }

    fn hear_entity(&mut self, eid: EID, oid: EID, env: &mut UpdateEnv) {
        let known = &mut env.known;
        swap(known, &mut self.entities[eid].known);

        let me = &self.entities[eid];
        let other = &self.entities[oid];
        known.update_entity(me, other, Sense::Sound, self.time);

        swap(known, &mut self.entities[eid].known);
    }

    fn observe_event(&mut self, eid: EID, event: &Event, env: &mut UpdateEnv) {
        let known = &mut env.known;
        swap(known, &mut self.entities[eid].known);
        known.observe_event(&mut self.entities[eid], event);
        swap(known, &mut self.entities[eid].known);
    }

    fn remove_known_entity(&mut self, eid: EID, oid: EID) {
        self.entities[eid].known.remove_entity(oid);
    }

    // Environmental effects

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

pub fn move_ready(entity: &Entity) -> bool { entity.move_timer <= 0 }

pub fn turn_ready(entity: &Entity) -> bool { entity.turn_timer <= 0 }

fn drain(entity: &mut Entity, result: &ActionResult) {
    entity.move_timer += (MOVE_TIMER as f64 * result.moves).round() as i32;
    entity.turn_timer += (TURN_TIMER as f64 * result.turns).round() as i32;
}

fn advance_turn(board: &mut Board) -> Option<EID> {
    if let Some(x) = board.active_entity { return Some(x); }

    let mut best = None;
    for (eid, entity) in &board.entities {
        assert!(entity.speed > 0.);
        let left = (entity.turn_timer as f64) * (1. / TURN_TIMER as f64);
        let time = Timedelta::from_seconds(left / entity.speed);
        if let Some((_, x)) = best && time >= x { continue; }
        best = Some((eid, time));
    }

    let (eid, time) = best?;
    let time = std::cmp::max(time, Timedelta::default());
    let charge = time.to_seconds() * TURN_TIMER as f64;

    for (_, entity) in &mut board.entities {
        let delta = (charge * entity.speed).round() as i32;
        if entity.move_timer > 0 { entity.move_timer -= delta; }
        if entity.turn_timer > 0 { entity.turn_timer -= delta; }
    }
    board.time = board.time.latch(time);
    board.active_entity = Some(eid);
    Some(eid)
}

//////////////////////////////////////////////////////////////////////////////

// Action

pub struct EatAction { pub target: Point, pub item: Option<Item> }

pub struct MoveAction { pub look: Point, pub step: Point, pub turns: f64 }

pub enum Action {
    Idle,
    Rest,
    SniffAround,
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

fn face_direction(entity: &mut Entity, dir: Point) {
    if dir != dirs::NONE { entity.dir = dir; }
}

fn plan(state: &mut State, eid: EID) -> Action {
    let player = eid == state.player;
    if player { return replace(&mut state.input, Action::WaitForInput); }

    let State { ai, board, env, ui: UI { debug, .. }, .. } = state;
    let vision = &mut env.fov.npc_vision;
    let entity = &mut board.entities[eid];
    swap(ai, &mut entity.ai);

    let debug = if state.pov == Some(eid) { Some(debug) } else { None };
    let mut env = AIEnv { debug, fov: vision, rng: &mut env.rng };
    let action = ai.plan(&entity, &mut env);

    swap(ai, &mut entity.ai);
    action
}

fn act(state: &mut State, eid: EID, action: Action) -> ActionResult {
    let entity = &mut state.board.entities[eid];
    entity.asleep = matches!(action, Action::Rest);

    match action {
        Action::Idle => ActionResult::success(),
        Action::Rest => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::SniffAround => {
            let (pos, color) = (entity.pos, 0xffff00);
            let board = &mut state.board;
            let cb = Box::new(|_: &mut Board, _: &mut UpdateEnv| {});
            board.add_effect(apply_flash(board, pos, color, cb), &mut state.env);
            ActionResult::success()
        }
        Action::Look(dir) => {
            face_direction(entity, dir);
            ActionResult::success()
        }
        Action::Drink(target) => {
            let dir = target - entity.pos;
            if dir.len_l1() > 1 { return ActionResult::failure(); }

            face_direction(entity, dir);
            let okay = state.board.get_cell(target).tile.can_drink();
            if !okay { return ActionResult::failure(); }

            let board = &mut state.board;
            let cb = Box::new(|_: &mut Board, _: &mut UpdateEnv| {});
            board.add_effect(apply_flash(board, target, 0x0000ff, cb), &mut state.env);
            ActionResult::success()
        }
        Action::Eat(EatAction { target, item }) => {
            let dir = target - entity.pos;
            if dir.len_l1() > 1 { return ActionResult::failure(); }

            face_direction(entity, dir);
            let cell = state.board.get_cell(target);
            let okay = match item {
                Some(x) => cell.items.iter().find(|&&y| y == x).is_some(),
                None => cell.tile.can_eat(),
            };
            if !okay { return ActionResult::failure(); }

            let board = &mut state.board;
            let color = if item.is_some() { 0xff0000 } else { 0xffff00 };
            let cb = Box::new(move |board: &mut Board, _: &mut UpdateEnv| {
                let Some(item) = item else { return };
                board.remove_item(target, item);
            });
            board.add_effect(apply_flash(board, target, color, cb), &mut state.env);
            ActionResult::success()
        }
        Action::Move(MoveAction { look, step, turns }) => {
            face_direction(entity, look);
            if step == dirs::NONE { return ActionResult::success_turns(turns); }
            if step.len_l1() > 1 { return ActionResult::failure(); }

            // Moving diagonally is slower. Moving quickly is noisier.
            let noisy = turns <= 1.;
            let turns = step.len_l2() * turns;
            let color = entity.glyph.fg();
            let player = entity.player;
            let source = entity.pos;
            let target = source + step;

            match state.board.get_status(target) {
                Status::Blocked | Status::Unknown => {
                    face_direction(&mut state.board.entities[eid], step);
                    ActionResult::failure()
                }
                Status::Occupied => {
                    face_direction(&mut state.board.entities[eid], step);
                    if player { state.ui.log.log_failure("There's something in the way!"); }
                    ActionResult::failure()
                }
                Status::Free => {
                    state.board.time = state.board.time.bump();

                    // Noise generation, for quick moves.
                    let mut updated = vec![];
                    let max = if noisy { NOISY_RADIUS } else { 1 };
                    for (oid, other) in &state.board.entities {
                        if oid == eid { continue; }
                        if other.asleep && !noisy { continue; }
                        let sr = (other.pos - source).len_nethack() <= max;
                        let tr = (other.pos - target).len_nethack() <= max;
                        if sr || tr { updated.push(oid); }
                    }
                    state.board.move_entity(eid, target);
                    for oid in updated {
                        if oid == state.player { continue; }
                        state.board.hear_entity(oid, eid, &mut state.env);
                    }

                    // Compute the list of entities that see or hear the move.
                    let board = &state.board;
                    let mut updated = vec![];
                    for (oid, other) in &board.entities {
                        if oid == eid { continue; }
                        if other.asleep && !noisy { continue; }

                        let fov = &mut state.env.fov;
                        let source_seen = fov.can_see_entity_at(board, other, source);
                        let target_seen = fov.can_see_entity_at(board, other, target);
                        let seen = source_seen || target_seen;

                        let source_heard = (other.pos - source).len_nethack() <= max;
                        let target_heard = (other.pos - target).len_nethack() <= max;
                        let heard = source_heard || target_heard;

                        if seen || heard { updated.push((oid, seen, target_seen)); }
                    }

                    // Deliver the MoveEvent to each entity in the list.
                    let (sense, time) = (Sense::Sight, state.board.time);
                    let data = EventData::Move(MoveEvent { from: source });
                    let mut event = Event { data, time, point: target, sense, turns: 0 };
                    for (oid, seen, target_seen) in updated {
                        event.sense = if seen { Sense::Sight } else { Sense::Sound };
                        state.board.observe_event(oid, &event, &mut state.env);
                        if oid != state.player { continue; }

                        let color = if seen { color } else { Color::white() };
                        state.ui.animate_move(color, 0, source);
                        if !target_seen { state.ui.animate_move(color, 1, target); }
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

            state.board.time = state.board.time.bump();

            let oid = state.board.get_cell(target).eid;
            let entity = &mut state.board.entities[eid];
            face_direction(entity, target - source);

            let cb = move |board: &mut Board, env: &mut UpdateEnv| {
                let Some(oid) = oid else { return; };
                let cb = move |board: &mut Board, env: &mut UpdateEnv| {
                    let Some(other) = board.entities.get_mut(oid) else { return; };

                    let damage = 1;
                    if other.cur_hp > damage {
                        other.cur_hp -= damage;
                        board.hear_entity(oid, eid, env);
                    } else {
                        let pos = other.pos;
                        board.remove_entity(oid);
                        board.add_item(pos, Item::Corpse);
                        board.remove_known_entity(eid, oid);
                    }
                };
                board.add_effect(apply_damage(board, target, Box::new(cb)), env);
            };

            let rng = &mut state.env.rng;
            let effect = effect::HeadbuttEffect(&state.board, rng, source, target);
            state.add_effect(apply_effect(effect, FT::Hit, Box::new(cb)));
            ActionResult::success_moves(1.)
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Animation
//
// TODO: All this code is WRONG! It may reveal things (e.g. hidden entities)
// to the player that they should not be able to see.
//
// To fix these bugs, we need to audit all usage of Board getters in Effects,
// and instead of directly overwriting cells, we need to apply updates (e.g.
// moving Entities) and use the player's FOV resolve what they can see.
//
// We also need to figure out if items are "hidden" the same way entities are.

type CB = Box<dyn Fn(&mut Board, &mut UpdateEnv)>;

fn apply_flash<T: Into<Color>>(board: &Board, target: Point, color: T, callback: CB) -> Effect {
    let cell = board.get_cell(target);
    let glyph = if let Some(x) = cell.eid {
        board.entities[x].glyph
    } else if let Some(x) = cell.items.last() {
        show_item(x)
    } else {
        cell.tile.glyph
    };

    let flash = glyph.with_fg(Color::black()).with_bg(color);
    let particle = effect::Particle { glyph: flash, point: target };
    let mut effect = Effect::constant(particle, UI_DAMAGE_FLASH);
    let frame = effect.frames.len() as i32;
    effect.add_event(effect::Event::Callback { frame, callback });
    effect
}

fn apply_damage(board: &Board, target: Point, callback: CB) -> Effect {
    let eid = board.get_cell(target).eid;
    let Some(eid) = eid else { return Effect::default(); };

    let glyph = board.entities[eid].glyph;
    let flash = glyph.with_fg(Color::black()).with_bg(0xff0000);
    let particle = effect::Particle { glyph: flash, point: target };
    let restored = effect::Particle { glyph, point: target };
    let mut effect = Effect::serial(vec![
        Effect::constant(particle, UI_DAMAGE_FLASH),
        Effect::constant(restored, UI_DAMAGE_TICKS),
    ]);
    let frame = effect.frames.len() as i32;
    effect.add_event(effect::Event::Callback { frame, callback });
    effect
}

fn apply_effect(mut effect: Effect, what: FT, callback: CB) -> Effect {
    let frame = effect.events.iter().find_map(
        |x| if x.what() == Some(what) { Some(x.frame()) } else { None });
    if let Some(frame) = frame {
        effect.add_event(effect::Event::Callback { frame, callback });
    }
    effect
}

//////////////////////////////////////////////////////////////////////////////

// Update

fn process_input(state: &mut State, input: Input) {
    if input == Input::Char('q') || input == Input::Char('w') {
        let board = &state.board;
        let eids: Vec<_> = board.entities.iter().map(|(eid, _)| eid).collect();
        let i = eids.iter().position(|&x| Some(x) == state.pov).unwrap_or(0);
        let l = eids.len();
        let j = (i + if input == Input::Char('q') { l - 1 } else { 1 }) % l;
        state.pov = if j == 0 { None } else { Some(eids[j]) };
        state.ui.debug = Default::default();
        return;
    }

    let player = &state.board.entities[state.player];
    if state.ui.process_input(player, input) { return; }

    let Input::Char(ch) = input else { return; };

    if ch == 'c' {
        let player = state.mut_player();
        player.sneaking = !player.sneaking;
        return;
    }

    let Some(dir) = get_direction(ch) else { return; };

    if dir == Point::default() {
        state.input = Action::Idle;
        return;
    }

    let player = state.get_player();
    let turns = if player.sneaking { 2. } else { 1. };
    state.input = Action::Move(MoveAction { look: dir, step: dir, turns });
}

fn update_pov_entities(state: &mut State) {
    state.board.update_known(state.player, &mut state.env);
    if let Some(x) = state.pov && state.board.entities.has(x) {
        state.board.update_known(x, &mut state.env);
    }
    let player = &state.board.entities[state.player];
    state.ui.update_focus(player);
}

fn update_state(state: &mut State) {
    let pos = state.get_player().pos;
    state.ui.update(pos, &mut state.env.rng);

    // If an Effect is active, run it, skipping frames the POV entity can't see.
    let pov = state.get_pov_entity().eid;
    if state.board.advance_effect(pov, &mut state.env) {
        update_pov_entities(state);
        return;
    }

    // The game loop is interrupted by animations, and if the player dies.
    let game_loop_active = |state: &State| {
        state.get_player().cur_hp > 0 && state.board.get_frame().is_none()
    };
    let needs_input = |state: &State| {
        game_loop_active(state) && matches!(state.input, Action::WaitForInput)
    };

    let mut update = false;
    while !state.inputs.is_empty() && needs_input(state) {
        let input = state.inputs.remove(0);
        process_input(state, input);
        update = true;
    }
    let player = &state.board.entities[state.player];
    if state.ui.update_target(player) { return; }

    while game_loop_active(state) {
        let Some(eid) = advance_turn(&mut state.board) else { break };

        let entity = &state.board.entities[eid];
        let player = entity.player;
        if player && needs_input(state) { break; }

        state.board.update_known(eid, &mut state.env);

        update = true;
        let action = plan(state, eid);
        let result = act(state, eid, action);
        if player && !result.success { break; }

        let Some(entity) = state.board.entities.get_mut(eid) else { continue };

        state.board.time = state.board.time.bump();

        entity.known.forget_events(player);

        let trail = &mut entity.trail;
        if trail.len() == trail.capacity() { trail.pop_back(); }
        trail.push_front(Scent { pos: entity.pos, time: state.board.time });

        state.board.active_entity = None;
        drain(entity, &result);
    }

    // Skip the prefix of Effect frames that the POV entity can't see.
    let pov = state.get_pov_entity().eid;
    if state.board.get_frame().is_some() && !state.board.pov_sees_effect(pov) {
        state.board.advance_effect(pov, &mut state.env);
    }

    if update { update_pov_entities(state); }
}

//////////////////////////////////////////////////////////////////////////////

// State

pub struct UpdateEnv {
    known: Box<Knowledge>,
    fov: FOV,
    rng: RNG,
}

pub struct State {
    board: Board,
    input: Action,
    inputs: Vec<Input>,
    player: EID,
    env: UpdateEnv,
    pov: Option<EID>,
    ai: Box<AIState>,
    ui: UI,
}

impl Default for State {
    fn default() -> Self {
        Self::new(/*seed=*/None, /*full=*/false)
    }
}

impl State {
    pub fn new(seed: Option<u64>, full: bool) -> Self {
        let size = Point(WORLD_SIZE, WORLD_SIZE);
        let rng = seed.map(|x| RNG::seed_from_u64(x));
        let rng = rng.unwrap_or_else(|| RNG::from_os_rng());
        let mut env = UpdateEnv {
            known: Default::default(),
            fov: Default::default(),
            rng,
        };
        let mut pos = Point(size.0 / 2, size.1 / 2);
        let mut board = Board::new(size, LIGHT);

        loop {
            let map = mapgen(size, &mut env.rng);
            for x in 0..size.0 {
                for y in 0..size.1 {
                    let p = Point(x, y);
                    board.set_tile(p, Tile::get(map.get(p)));
                }
            }
            for y in 0..size.1 {
                let p = Point(0, y);
                if map.get(p) == 'R' { pos = p; }
            }
            if !board.get_tile(pos).blocks_movement() { break; }
        }

        let input = Action::WaitForInput;
        let glyph = Glyph::wdfg('@', 0xffffff);
        let (player, speed) = (true, SPEED_PC_);
        let args = EntityArgs { glyph, player, predator: false, pos, speed };
        let player = board.add_entity(&args, &mut env);

        let pos = |board: &Board, rng: &mut RNG| {
            for _ in 0..100 {
                let p = Point(rng.gen_range(0..size.0), rng.gen_range(0..size.1));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        for i in 0..(NUM_PREDATORS + NUM_PREY) {
            if let Some(x) = pos(&board, &mut env.rng) {
                let predator = i < NUM_PREDATORS;
                let (player, speed) = (false, SPEED_NPC);
                let glyph = if predator {
                    Glyph::wdfg('R', 0xa060ff)
                } else {
                    Glyph::wdfg('P', 0xd0a070)
                };
                let args = EntityArgs { glyph, player, predator, pos: x, speed };
                board.add_entity(&args, &mut env);
            }
        }
        board.entities[player].dir = dirs::S;
        board.update_known(player, &mut env);

        let inputs = Default::default();
        let ai = Box::new(AIState::new(/*predator=*/false, &mut env.rng));
        let pov = None;

        let mut ui = UI::default();
        match WEATHER {
            Weather::Rain(angle, count) => ui.start_rain(angle, count),
            Weather::None => (),
        }
        ui.log.log("Welcome to WildsRL! Use vikeys (h/j/k/l/y/u/b/n) to move.");
        if full { ui.show_full_view(); }

        Self { board, input, inputs, player, env, pov, ai, ui }
    }

    fn get_pov_entity(&self) -> &Entity {
        self.pov.and_then(|x| self.board.get_entity(x)).unwrap_or(self.get_player())
    }

    fn get_player(&self) -> &Entity { &self.board.entities[self.player] }

    fn mut_player(&mut self) -> &mut Entity { &mut self.board.entities[self.player] }

    pub fn add_effect(&mut self, x: Effect) { self.board.add_effect(x, &mut self.env) }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Buffer) {
        let entity = self.get_pov_entity();
        self.ui.render(buffer, entity, &self.board);
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
        for i in 0..NUM_SEEDS {
            states.push(State::new(Some(BASE_SEED + i), /*full=*/false));
        }

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
