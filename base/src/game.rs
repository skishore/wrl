use core::ops::{Deref, DerefMut};
use std::fmt::Debug;
use std::mem::{replace, swap};

use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use thin_vec::{ThinVec, thin_vec};

use crate::static_assert_size;
use crate::ai::{AIEnv, AIState};
use crate::base::{Buffer, Color, Glyph};
use crate::base::{HashMap, LOS, Matrix, Point, RNG, dirs, sample, weighted};
use crate::dex::{Attack, Species};
use crate::effect::{CB, Effect, Frame, FT, self};
use crate::entity::{EID, Entity, EntityArgs, EntityMap};
use crate::knowledge::{Call, Knowledge, Scent, Sense, Timedelta, Timestamp};
use crate::knowledge::{AttackEvent, CallEvent, Event, EventData, MoveEvent};
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

const LIGHT: Light = Light::Sun(Point(2, 0));
const WEATHER: Weather = Weather::None;
const NUM_PREDATORS: i32 = 5;
const NUM_PREY: i32 = 15;

const UI_FLASH: i32 = 4;
const UI_DAMAGE_FLASH: i32 = 6;
const UI_DAMAGE_TICKS: i32 = 6;

const SLOWED_TURNS: f64 = 2.;

pub const ATTACK_VOLUME: i32 = FOV_RADIUS_NPC;
pub const CALL_VOLUME: i32 = FOV_RADIUS_NPC;
pub const MOVE_VOLUME: i32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char), Click(Point) }

//////////////////////////////////////////////////////////////////////////////

// Tile

const FLAG_BLOCKS_VISION: u32 = 1 << 0;
const FLAG_LIMITS_VISION: u32 = 1 << 1;
const FLAG_BLOCKS_MOVEMENT: u32 = 1 << 2;
const FLAG_CAN_DRINK: u32 = 1 << 3;
const FLAG_CAN_EAT: u32 = 1 << 4;
const FLAG_BERRY: u32 = 1 << 5;

const FLAGS_NONE: u32 = 0;
const FLAGS_BLOCKED: u32 = FLAG_BLOCKS_MOVEMENT | FLAG_BLOCKS_VISION;
const FLAGS_FRESH_WATER: u32 = FLAG_BLOCKS_MOVEMENT | FLAG_CAN_DRINK;
const FLAGS_BERRY_TREE: u32 = FLAG_BLOCKS_MOVEMENT | FLAG_LIMITS_VISION | FLAG_BERRY;
const FLAGS_TALL_GRASS: u32 = FLAG_LIMITS_VISION;

pub struct Tile {
    pub flags: u32,
    pub glyph: Glyph,
    pub description: &'static str,
}

impl Tile {
    pub fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    pub fn try_get(ch: char) -> Option<&'static Tile> { TILES.get(&ch) }

    // Raw flags-based predicates.
    pub fn can_eat(&self) -> bool { self.flags & FLAG_CAN_EAT != 0 }
    pub fn can_drink(&self) -> bool { self.flags & FLAG_CAN_DRINK != 0 }
    pub fn blocks_vision(&self) -> bool { self.flags & FLAG_BLOCKS_VISION != 0 }
    pub fn limits_vision(&self) -> bool { self.flags & FLAG_LIMITS_VISION != 0 }
    pub fn blocks_movement(&self) -> bool { self.flags & FLAG_BLOCKS_MOVEMENT != 0 }
    pub fn drops_berries(&self) -> bool { self.flags & FLAG_BERRY != 0 }

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
            ('#', FLAGS_BLOCKED,     ('#', 0x106000), "a tree"),
            ('.', FLAGS_NONE,        ('.', 0xe0ffc0), "grass"),
            (',', FLAGS_NONE,        ('`', 0x60c060), "weeds"),
            ('"', FLAGS_TALL_GRASS,  ('"', 0x60c000), "tall grass"),
            ('|', FLAGS_TALL_GRASS,  ('|', 0x60c000), "reeds"),
            ('+', FLAGS_NONE,        ('+', 0xff6060), "a flower"),
            ('~', FLAGS_FRESH_WATER, ('~', 0x0080ff), "water"),
            ('B', FLAGS_BERRY_TREE,  ('#', 0xc08000), "a berry tree"),
            ('=', FLAGS_NONE,        ('=', 0xff8000), "a bridge"),
            ('R', FLAGS_NONE,        ('.', 0xff8000), "a path"),
        ];
        let mut result = HashMap::default();
        for (ch, flags, glyph, description) in items {
            let glyph = Glyph::wdfg(glyph.0, glyph.1);
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
        Item::Berry  => Glyph::wdfg('o', 0xc08000),
        Item::Corpse => Glyph::wdfg('%', 0xffffff),
    }
}

pub fn capitalize(mut line: String) -> String {
    line.get_mut(..1).map(|x| x.make_ascii_uppercase()); line
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
        let mut visible = self._pov_sees_effect(pov, env);
        while self._advance_one_frame(env) {
            visible = visible || self._pov_sees_effect(pov, env);
            if visible { return true; }
        }
        false
    }

    fn start_effect(&mut self, pov: EID, env: &mut UpdateEnv) {
        if self.get_frame().is_none() { return; }
        if !self._pov_sees_effect(pov, env) { self.advance_effect(pov, env); }
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

    fn _pov_sees_effect(&self, pov: EID, env: &UpdateEnv) -> bool {
        if env.ui.full { return true; }

        let Some(frame) = self._effect.frames.get(0) else { return false };
        let Some(entity) = self.entities.get(pov) else { return false };

        let known = &entity.known;
        frame.iter().any(|y| known.get(y.point).visible())
    }

    // Getters

    pub fn get_cell(&self, p: Point) -> &Cell { self.map.entry_ref(p) }

    pub fn get_entity(&self, eid: EID) -> Option<&Entity> { self.entities.get(eid) }

    pub fn get_frame(&self) -> Option<&Frame> { self._effect.frames.first() }

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

    fn create_event(&self, eid: EID, data: EventData, point: Point) -> Event {
        let (eid, uid) = (Some(eid), None);
        Event { eid, uid, data, time: self.time, point, sense: Sense::Sight }
    }

    fn observe_event(&mut self, eid: EID, event: &Event, env: &mut UpdateEnv) {
        let known = &mut env.known;
        swap(known, &mut self.entities[eid].known);
        known.observe_event(&mut self.entities[eid], event);
        swap(known, &mut self.entities[eid].known);
    }

    fn remove_known_entity(&mut self, eid: EID, oid: EID) {
        self.entities[eid].known.remove_entity(oid, self.time);
    }

    fn update_known(&mut self, eid: EID, env: &mut UpdateEnv) {
        let UpdateEnv { known, fov, rng, .. } = env;
        swap(known, &mut self.entities[eid].known);

        let me = &self.entities[eid];
        let vision = fov.compute(&self, me);
        known.update(me, &self, vision, rng);

        swap(known, &mut self.entities[eid].known);
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

// Event delivery

struct Noise {
    cause: Option<EID>,
    point: Point,
    volume: i32,
}

struct Sighting {
    eid: EID,
    source_seen: bool,
    target_seen: bool,
}

type Senses = HashMap<EID, Sense>;

fn detect(board: &Board, noise: &Noise, env: &mut UpdateEnv) -> Senses {
    let Noise { cause, point, volume } = *noise;
    let mut result = Senses::default();

    for (eid, entity) in &board.entities {
        if cause == Some(eid) { continue; }
        if entity.asleep && volume == 1 && point != entity.pos { continue; }

        let seen = env.fov.can_see_entity_at(board, entity, point);
        let heard = !seen && (point - entity.pos).len_nethack() <= volume;
        if !seen && !heard { continue; }

        let sense = if seen { Sense::Sight } else { Sense::Sound };
        result.insert(eid, sense);
    }
    result
}

fn combine_views(board: &Board, saw_source: &Senses, saw_target: &Senses) -> Vec<Sighting> {
    let mut result = vec![];
    for (eid, _) in &board.entities {
        let ss = saw_source.get(&eid).copied();
        let st = saw_target.get(&eid).copied();
        if ss.is_none() && st.is_none() { continue; }

        let source_seen = ss == Some(Sense::Sight);
        let target_seen = st == Some(Sense::Sight);
        result.push(Sighting { eid, source_seen, target_seen });
    }
    result
}

fn get_sightings(board: &Board, noise: &Noise, env: &mut UpdateEnv) -> Vec<Sighting> {
    let seen = detect(board, noise, env);
    combine_views(board, &seen, &Default::default())
}

//////////////////////////////////////////////////////////////////////////////

// Attack effects

fn hit_tile(board: &mut Board, env: &mut UpdateEnv, target: Point) {
    if !board.get_tile(target).drops_berries() { return; }

    let options: Vec<_> = dirs::ALL.clone().into_iter().filter(
        |&x| board.get_status(target + x) != Status::Blocked).collect();
    if options.is_empty() { return; }

    let rng = &mut env.rng;
    let n = *weighted(&[(1, 0), (2, 1), (1, 2)], rng);
    for _ in 0..n { board.add_item(target + *sample(&options, rng), Item::Berry); }
}

fn hit_entity(board: &mut Board, env: &mut UpdateEnv, attack: &Attack, logged: bool, tid: EID) {
    let Some(target) = board.entities.get_mut(tid) else { return; };
    let (pos, desc) = (target.pos, target.desc());

    let critted = env.rng.gen_range(0..16) == 0;
    let factor = if critted { 1.5 } else { 1. } * env.rng.gen_range(0.85..=1.);
    let damage = (factor * attack.damage as f64).round() as i32;
    let fainted = target.cur_hp <= damage;

    if fainted {
        board.remove_entity(tid);
        board.add_item(pos, Item::Corpse);
    } else {
        target.cur_hp -= damage;
    }

    let volume = ATTACK_VOLUME;
    let noise = Noise { cause: Some(tid), point: pos, volume };
    let sightings = get_sightings(board, &noise, env);

    for Sighting { eid: oid, source_seen: seen, .. } in sightings {
        if fainted { board.remove_known_entity(oid, tid); }
        if !seen || !board.entities[oid].player { continue; }

        let log = &mut env.ui.log;
        if !logged { log.log(format!("Something attacked {}!", desc)); }
        if critted { log.log_append("A critical hit!"); }
        if fainted { log.log_append(capitalize(format!("{} fainted!", desc))); }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Turn-taking

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
    let charge = time.seconds() * TURN_TIMER as f64;

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

pub struct AttackAction { pub attack: &'static Attack, pub target: Point }

pub struct CallAction { pub call: Call, pub look: Point }

pub struct EatAction { pub target: Point, pub item: Option<Item> }

pub struct MoveAction { pub look: Point, pub step: Point, pub turns: f64 }

pub enum Action {
    Idle,
    Rest,
    SniffAround,
    WaitForInput,
    Look(Point),
    Call(CallAction),
    Move(MoveAction),
    Attack(AttackAction),
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

fn can_attack(board: &Board, entity: &Entity, action: &AttackAction) -> bool {
    let (known, source) = (&entity.known, entity.pos);
    let (range, target) = (action.attack.range, action.target);
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

    let State { ai, board, env, .. } = state;
    let vision = &mut env.fov.npc_vision;
    let entity = &mut board.entities[eid];
    swap(ai, &mut entity.ai);

    let debug = if state.pov == Some(eid) { Some(&mut env.ui.debug) } else { None };
    let mut env = AIEnv { debug, fov: vision, rng: &mut env.rng };
    let action = ai.plan(&entity, &mut env);

    swap(ai, &mut entity.ai);

    entity.known.events.clear();

    action
}

fn act(state: &mut State, eid: EID, action: Action) -> ActionResult {
    let entity = &mut state.board.entities[eid];
    entity.asleep = matches!(action, Action::Rest);
    let source = entity.pos;

    match action {
        Action::Idle => ActionResult::success(),
        Action::Rest => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::SniffAround => {
            let (pos, color) = (entity.pos, 0xffff00);
            let board = &mut state.board;
            let effect = apply_flash(board, pos, color, None);
            board.add_effect(effect, &mut state.env);
            ActionResult::success()
        }
        Action::Look(dir) => {
            entity.face_direction(dir);
            ActionResult::success()
        }
        Action::Drink(target) => {
            let (source, dir) = (entity.pos, target - entity.pos);
            if dir.len_l1() > 1 { return ActionResult::failure(); }

            entity.face_direction(dir);
            let okay = state.board.get_cell(target).tile.can_drink();
            if !okay { return ActionResult::failure(); }

            let color = 0x0080ff;
            let board = &mut state.board;
            let effect = Effect::serial(vec![
                apply_flash(board, target, color, None),
                apply_flash(board, source, color, None).delay(UI_FLASH / 2),
            ]);
            board.add_effect(effect, &mut state.env);
            ActionResult::success()
        }
        Action::Eat(EatAction { target, item }) => {
            let (source, dir) = (entity.pos, target - entity.pos);
            if dir.len_l1() > 1 { return ActionResult::failure(); }

            entity.face_direction(dir);
            let cell = state.board.get_cell(target);
            let okay = match item {
                Some(x) => cell.items.iter().find(|&&y| y == x).is_some(),
                None => cell.tile.can_eat(),
            };
            if !okay { return ActionResult::failure(); }

            let board = &mut state.board;
            let color = if item == Some(Item::Corpse) { 0xff0000 } else { 0xffff00 };
            let cb = Box::new(move |board: &mut Board, _: &mut UpdateEnv| {
                let Some(item) = item else { return };
                board.remove_item(target, item);
            });
            let effect = Effect::serial(vec![
                apply_flash(board, target, color, Some(cb)),
                apply_flash(board, source, color, None).delay(UI_FLASH / 2),
            ]);
            board.add_effect(effect, &mut state.env);
            ActionResult::success()
        }
        Action::Call(CallAction { call, look }) => {
            let noise = Noise { cause: Some(eid), point: source, volume: CALL_VOLUME };
            let sightings = get_sightings(&state.board, &noise, &mut state.env);

            // Deliver a CallEvent to each entity that heard the call.
            let data = EventData::Call(CallEvent { call });
            let mut event = state.board.create_event(eid, data, source);
            for Sighting { eid: oid, source_seen: seen, .. } in sightings {
                event.sense = if seen { Sense::Sight } else { Sense::Sound };
                state.board.observe_event(oid, &event, &mut state.env);
            }

            // Use a different color for different call types.
            let (color, wait) = match call {
                Call::Help    => (0x00ffff, true),
                Call::Warning => (0xff8000, false),
            };
            let board = &mut state.board;
            let mut effect = Effect::default();
            for _ in 0..3 {
                effect = effect.then(apply_flash(board, source, color, None));
                effect = effect.then(Effect::pause(UI_FLASH));
            }

            // For some call types, we look before calling; when calling for
            // help, we shout in the direction of our allies, then look.
            let cb = move |board: &mut Board, _: &mut UpdateEnv| {
                let Some(entity) = board.entities.get_mut(eid) else { return };
                entity.face_direction(look);
            };
            if wait {
                effect.sub_on_finished(Box::new(cb));
            } else {
                cb(board, &mut state.env);
            }

            board.add_effect(effect, &mut state.env);
            ActionResult::success()
        }
        Action::Move(MoveAction { look, step, turns }) => {
            entity.face_direction(look);
            let slowed = turns < SLOWED_TURNS && !move_ready(entity);
            let turns = if slowed { SLOWED_TURNS } else { turns };
            if step == dirs::NONE { return ActionResult::success_turns(turns); }
            if step.len_l1() > 1 { return ActionResult::failure(); }

            // Moving diagonally is slower. Moving quickly is noisier.
            let noisy = turns <= 1.;
            let turns = step.len_l2() * turns;
            let color = entity.species.glyph.fg();
            let player = entity.player;
            let target = source + step;

            match state.board.get_status(target) {
                Status::Blocked | Status::Unknown => {
                    state.board.entities[eid].face_direction(step);
                    ActionResult::failure()
                }
                Status::Occupied => {
                    state.board.entities[eid].face_direction(step);
                    if player { state.ui.log.log_failure("There's something in the way!"); }
                    ActionResult::failure()
                }
                Status::Free => {
                    state.board.time = state.board.time.bump();

                    let volume = if noisy { MOVE_VOLUME } else { 1 };
                    let noise = Noise { cause: Some(eid), point: source, volume };
                    let saw_source = detect(&state.board, &noise, &mut state.env);

                    state.board.move_entity(eid, target);

                    let noise = Noise { cause: Some(eid), point: target, volume };
                    let saw_target = detect(&state.board, &noise, &mut state.env);
                    let sightings = combine_views(&state.board, &saw_source, &saw_target);

                    // Deliver a MoveEvent to each entity that saw the move.
                    let data = EventData::Move(MoveEvent { from: source });
                    let mut event = state.board.create_event(eid, data, target);
                    for Sighting { eid: oid, source_seen, target_seen } in sightings {
                        let seen = source_seen || target_seen;
                        event.sense = if seen { Sense::Sight } else { Sense::Sound };
                        state.board.observe_event(oid, &event, &mut state.env);
                        if oid != state.player { continue; }

                        let color = if seen { color } else { Color::white() };
                        state.ui.animate_move(color, source);
                    }
                    ActionResult::success_turns(turns)
                }
            }
        }
        Action::Attack(action) => {
            let entity = &state.board.entities[eid];
            let AttackAction { attack, target } = action;
            if !can_attack(&state.board, entity, &action) {
                state.board.entities[eid].face_direction(target - source);
                return ActionResult::failure();
            }

            state.board.time = state.board.time.bump();

            let volume = ATTACK_VOLUME;
            let noise = Noise { cause: Some(eid), point: source, volume };
            let saw_source = detect(&state.board, &noise, &mut state.env);

            let tid = state.board.get_cell(target).eid;
            let entity = &mut state.board.entities[eid];
            entity.face_direction(target - source);

            let noise = Noise { cause: Some(eid), point: target, volume };
            let saw_target = detect(&state.board, &noise, &mut state.env);
            let sightings = combine_views(&state.board, &saw_source, &saw_target);

            // Deliver the AttackEvent to each entity in the list.
            let combat = tid.is_some();
            let data = EventData::Attack(AttackEvent { combat, target: None });
            let mut event = state.board.create_event(eid, data, source);
            let mut logged = false;
            for Sighting { eid: oid, source_seen, target_seen } in sightings {
                let target = if target_seen { tid } else { None };
                event.data = EventData::Attack(AttackEvent { combat, target });
                event.sense = if source_seen { Sense::Sight } else { Sense::Sound };
                state.board.observe_event(oid, &event, &mut state.env);
                if oid != state.player { continue; }

                let entities = &state.board.entities;
                let attacker = if source_seen { Some(&entities[eid]) } else { None };
                let attacked = if target_seen { tid.map(|x| &entities[x]) } else { None };
                logged = true;

                let line = if let Some(a) = attacker && let Some(b) = attacked {
                    format!("{} attacked {} with {}!", a.desc(), b.desc(), attack.name)
                } else if let Some(a) = attacker {
                    format!("{} used {}!", a.desc(), attack.name)
                } else if let Some(b) = attacked {
                    format!("Something attacked {}!", b.desc())
                } else {
                    "You hear fighting nearby!".into()
                };
                state.ui.log.log(capitalize(line));
            }

            let cb = move |board: &mut Board, env: &mut UpdateEnv| {
                hit_tile(board, env, target);

                let Some(tid) = tid else { return; };

                let cb = move |board: &mut Board, env: &mut UpdateEnv| {
                    hit_entity(board, env, attack, logged, tid);
                };
                board.add_effect(apply_damage(board, target, Box::new(cb)), env);
            };

            let rng = &mut state.env.rng;
            let effect = (attack.effect)(&state.board, rng, source, target);
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

fn apply_flash<T: Into<Color>>(board: &Board, target: Point, color: T, cb: Option<CB>) -> Effect {
    let cell = board.get_cell(target);
    let glyph = if let Some(x) = cell.eid {
        board.entities[x].species.glyph
    } else if let Some(x) = cell.items.last() {
        show_item(x)
    } else {
        cell.tile.glyph
    };

    let flash = glyph.with_fg(Color::black()).with_bg(color);
    let particle = effect::Particle { glyph: flash, point: target };
    let mut effect = Effect::constant(particle, UI_FLASH);
    if let Some(x) = cb { effect.sub_on_finished(x); }
    effect
}

fn apply_damage(board: &Board, target: Point, callback: CB) -> Effect {
    let eid = board.get_cell(target).eid;
    let Some(eid) = eid else { return Effect::default(); };

    let glyph = board.entities[eid].species.glyph;
    let flash = glyph.with_fg(Color::black()).with_bg(0xff0000);
    let particle = effect::Particle { glyph: flash, point: target };
    let restored = effect::Particle { glyph, point: target };
    let mut effect = Effect::serial(vec![
        Effect::constant(particle, UI_DAMAGE_FLASH),
        Effect::constant(restored, UI_DAMAGE_TICKS),
    ]);
    effect.sub_on_finished(callback);
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

    if state.ui.full && let Input::Click(pos) = input {
        let pov = state.get_pov_entity();
        let Some(pos) = state.ui.get_map_cell(pov, pos) else { return };
        let Some(eid) = state.board.get_cell(pos).eid else { return };
        let off = eid == state.player || state.pov == Some(eid);
        state.pov = if off { None } else { Some(eid) };
        return;
    }

    let player = &state.board.entities[state.player];
    if state.env.ui.process_input(player, input) { return; }

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
    state.env.ui.update_focus(player);
}

fn update_state(state: &mut State) {
    let pos = state.get_player().pos;
    state.env.ui.update(pos, &mut state.env.rng);

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
    if state.env.ui.update_target(player) { return; }

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

        if player { entity.known.forget_old_sources(); }

        let trail = &mut entity.trail;
        if trail.len() == trail.capacity() { trail.pop_back(); }
        trail.push_front(Scent { pos: entity.pos, time: state.board.time });

        state.board.active_entity = None;
        drain(entity, &result);
    }

    // Skip the prefix of Effect frames that the POV entity can't see.
    let pov = state.get_pov_entity().eid;
    state.board.start_effect(pov, &mut state.env);

    if update { update_pov_entities(state); }
}

//////////////////////////////////////////////////////////////////////////////

// State

pub struct UpdateEnv {
    known: Box<Knowledge>,
    fov: FOV,
    rng: RNG,
    ui: UI,
}

pub struct State {
    board: Board,
    input: Action,
    inputs: Vec<Input>,
    player: EID,
    env: UpdateEnv,
    pov: Option<EID>,
    ai: Box<AIState>,
}

impl Default for State {
    fn default() -> Self {
        Self::new(/*seed=*/None, /*full=*/false)
    }
}

impl Deref for State {
    type Target = UpdateEnv;
    fn deref(&self) -> &Self::Target {
        &self.env
    }
}

impl DerefMut for State {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.env
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
            ui: Default::default(),
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
            if full {
                let wt = board.map.default.tile;
                for &dir in &dirs::ALL { board.set_tile(pos + dir, wt); }
            }
            if !board.get_tile(pos).blocks_movement() { break; }
        }

        let input = Action::WaitForInput;
        let (player, species) = (true, Species::get("Human"));
        let args = EntityArgs { pos, player, predator: false, species };
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
                let species = Species::get(if predator { "Rattata" } else { "Pidgey" });
                let args = EntityArgs { pos: x, player: false, predator, species };
                board.add_entity(&args, &mut env);
            }
        }
        board.entities[player].dir = dirs::S;
        board.update_known(player, &mut env);

        let inputs = Default::default();
        let ai = Box::new(AIState::new(/*predator=*/false, &mut env.rng));
        let pov = None;

        let ui = &mut env.ui;
        match WEATHER {
            Weather::Rain(angle, count) => ui.start_rain(angle, count),
            Weather::None => (),
        }
        ui.log.log("Welcome to WildsRL! Use vikeys (h/j/k/l/y/u/b/n) to move.");
        if full { ui.show_full_view(); }

        Self { board, input, inputs, player, env, pov, ai }
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
    const NUM_STEPS: u64 = 4096;

    #[test]
    fn test_state_update() {
        let mut states = vec![];
        for i in 0..NUM_SEEDS {
            states.push(State::new(Some(BASE_SEED + i), /*full=*/true));
        }

        for index in 0..(NUM_SEEDS * NUM_STEPS) {
            let i = index as usize % states.len();
            let state = &mut states[i];

            state.inputs.push(Input::Char('.'));
            state.update();
            while state.board.get_frame().is_some() { state.update(); }
        }
    }

    #[bench]
    fn bench_state_update(b: &mut test::Bencher) {
        let mut index = 0;
        let mut states = vec![];
        for i in 0..NUM_SEEDS {
            states.push(State::new(Some(BASE_SEED + i), /*full=*/true));
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
