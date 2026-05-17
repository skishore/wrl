use std::fmt::Debug;
use std::mem::{replace, swap};
use std::sync::LazyLock;

use rand::{Rng, SeedableRng};
use thin_vec::ThinVec;

use crate::{flags, static_assert_size};
use crate::base::glyph::{Color, Glyph, Buffer};
use crate::base::lighting::Lighting;
use crate::base::pathing::Status;
use crate::base::point::{Bound, LOS, Matrix, Point, dirs};
use crate::base::util::{HashMap, HashSet, RNG, sample, weighted};
use crate::base::vision::{INITIAL_VISIBILITY, VISIBILITY_LOSSES, Vision, VisionArgs};

use super::ai::{AIEnv, AIState};
use super::dex::{Attack, Species};
use super::debug::DebugFile;
use super::effect::{CB, Effect, Frame, FT, Particle, ParticleData, RenderData, self};
use super::entity::{Command, Individual, Teammate};
use super::entity::{EID, Entity, EntityArgs, EntityMap};
use super::event::{Call, Location, Sense};
use super::event::{AttackEvent, CallEvent, Event, EventData, MoveEvent};
use super::knowledge::Knowledge;
use super::mapgen::mapgen_with_size as mapgen;
use super::time::{Timedelta, Timestamp};
use super::ui::UI;

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
const NUM_PREDATORS: i32 = 2;
const NUM_PREY: i32 = 18;

const UI_FLASH: i32 = 4;
const UI_NOISE: i32 = 8;
const UI_DAMAGE_FLASH: i32 = 6;
const UI_DAMAGE_TICKS: i32 = 6;

const SLOWED_TURNS: f64 = 1.5;

pub const ATTACK_VOLUME: Bound = Bound::new(FOV_RADIUS_NPC);
pub const CALL_VOLUME: Bound = Bound::new(FOV_RADIUS_NPC);
pub const MOVE_VOLUME: Bound = Bound::new(8);
pub const SNEAK_VOLUME: Bound = Bound::new(1);
pub const SNIFF_VOLUME: Bound = Bound::new(8);
pub const SHOUT_VOLUME: Bound = Bound::new(12);

pub const FOLLOW_RANGE: Bound = Bound::new(4);
pub const SUMMON_RANGE: Bound = Bound::new(12);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char), Click(Point) }

//////////////////////////////////////////////////////////////////////////////

// Tile

flags! {
    pub TileFlags(u32) {
        BlocksMovement,
        BlocksVision,
        LimitsVision,
        DropsBerries,
        CanDrink,
        CanEat,

        /// Deriving:
        Blocked = BlocksMovement | BlocksVision,
        FreshWater = BlocksMovement | CanDrink,
        BerryTree = BlocksMovement | LimitsVision | DropsBerries,
        TallGrass = LimitsVision,
    }
}

type TF = TileFlags;

pub struct Tile {
    pub flags: TileFlags,
    pub glyph: Glyph,
    pub description: &'static str,
}

impl Tile {
    pub fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    pub fn try_get(ch: char) -> Option<&'static Tile> { TILES.get(&ch) }

    // Raw flags-based predicates:

    pub fn can_eat(&self) -> bool { self.flags.any(TF::CanEat) }
    pub fn can_drink(&self) -> bool { self.flags.any(TF::CanDrink) }
    pub fn blocks_vision(&self) -> bool { self.flags.any(TF::BlocksVision) }
    pub fn limits_vision(&self) -> bool { self.flags.any(TF::LimitsVision) }
    pub fn blocks_movement(&self) -> bool { self.flags.any(TF::BlocksMovement) }
    pub fn drops_berries(&self) -> bool { self.flags.any(TF::DropsBerries) }

    // Derived predicates:

    pub fn casts_shadow(&self) -> bool { self.blocks_vision() }

    pub fn is_cover(&self) -> bool { self.limits_vision() && !self.blocks_movement() }

    pub fn opacity(&self) -> i32 {
        if self.blocks_vision() { return INITIAL_VISIBILITY; }
        if self.limits_vision() { return VISIBILITY_LOSS; }
        0
    }
}

impl Default for &'static Tile {
    fn default() -> Self { &DEFAULT_TILE }
}

impl Eq for &'static Tile {}

impl PartialEq for &'static Tile {
    fn eq(&self, next: &&'static Tile) -> bool {
        *self as *const Tile == *next as *const Tile
    }
}

static DEFAULT_TILE: LazyLock<&'static Tile> = LazyLock::new(|| TILES.get(&'#').unwrap());

static TILES: LazyLock<HashMap<char, Tile>> = LazyLock::new(|| {
    let items = [
        ('#', TF::Blocked,    ('#', 0x106000), "a tree"),
        ('.', TF::Empty,      ('.', 0xe0ffc0), "grass"),
        (',', TF::Empty,      ('`', 0x60c060), "weeds"),
        ('"', TF::TallGrass,  ('"', 0x60c000), "tall grass"),
        ('|', TF::TallGrass,  ('|', 0x60c000), "reeds"),
        ('+', TF::Empty,      ('+', 0xff6060), "a flower"),
        ('~', TF::FreshWater, ('~', 0x0080ff), "water"),
        ('B', TF::BerryTree,  ('#', 0xc08000), "a berry tree"),
        ('=', TF::Empty,      ('=', 0xff8000), "a bridge"),
        ('R', TF::Empty,      ('.', 0xff8000), "a path"),
    ];
    let mut result = HashMap::default();
    for (ch, flags, glyph, description) in items {
        let glyph = Glyph::wdfg(glyph.0, glyph.1);
        result.insert(ch, Tile { flags, glyph, description });
    }
    result
});

//////////////////////////////////////////////////////////////////////////////

// Item

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Item { Berry, Corpse }

pub fn show_item(item: &Item) -> Glyph {
    match item {
        Item::Berry  => Glyph::wdfg('o', 0xc08000),
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
    fn select_vision(&mut self, me: &Entity) -> &mut Vision {
        if me.player { &mut self._pc_vision } else { &mut self.npc_vision }
    }

    fn can_see(&mut self, board: &Board, me: &Entity, point: Point) -> bool {
        let Entity { pos, dir, asleep, player, .. } = *me;
        if asleep { return pos == point; }

        let vision = self.select_vision(me);

        let map = &board.map;
        let dir = if player { Point::default() } else { dir };
        let opacity_lookup = |x| map.get(x).tile.opacity();
        vision.check_point(&VisionArgs { pos, dir, opacity_lookup }, point)
    }

    fn can_see_entity(&mut self, board: &Board, me: &Entity, other: &Entity) -> bool {
        // If we can't see the cell at all, we can't see entities there.
        let point = other.pos;
        if !self.can_see(board, me, point) { return false; }

        // Entities can't hide if we're adjacent to them.
        let nearby = (point - me.pos).len_l1() <= 1;
        if nearby { return true; }

        // If they're hidden due to tall grass, then we can't see them.
        let cell = board.get_cell(point);
        if cell.tile.is_cover() && !other.too_big_to_hide() { return false; }

        // If they're hidden due to being in shadow, we can't see them.
        let unlit = matches!(board.get_light(), Light::None);
        let shade = unlit || cell.shadow > 0;
        !shade || board.is_cell_lit(point)
    }

    fn compute(&mut self, board: &Board, me: &Entity) -> &Vision {
        let vision = self.select_vision(me);
        let Entity { pos, dir, asleep, player, .. } = *me;
        if asleep {
            vision.clear();
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

#[derive(Clone, Default)]
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

    // Animation:
    _effect: Effect,
    _frame_mask: Vec<bool>,

    // Lighting:
    light: Light,
    lighting: Lighting,
    shadow: Vec<Point>,
}

impl Board {
    fn new(size: Point, light: Light) -> Self {
        let lighting = Lighting::new(size);
        let shadow = match light {
            Light::Sun(x) => LOS(Point::default(), x).into_iter().skip(1).collect(),
            Light::None => vec![],
        };
        let cell = Cell::default();
        let time = Timedelta::from_seconds(365.75 * 86400.);

        let mut result = Self {
            map: Matrix::new(size, cell),
            active_entity: None,
            entities: EntityMap::default(),
            time: Timestamp::default().latch(time),

            // Animation:
            _effect: Effect::default(),
            _frame_mask: vec![],

            // Lighting:
            light,
            lighting,
            shadow,
        };
        result.reset(Tile::get('.'));
        result
    }

    // Animation:

    fn add_effect(&mut self, effect: Effect) {
        self._effect = std::mem::take(&mut self._effect).and(effect);
        self._effect.events.retain(|x| matches!(x, effect::Event::Callback { .. }));
        self._frame_mask.clear();
    }

    fn pop_callback(&mut self) -> Option<CB> {
        let event = self._effect.events.first()?;
        if !self._effect.frames.is_empty() && event.frame() > 0 { return None };

        match self._effect.events.remove(0) {
            effect::Event::Callback { callback, .. } => Some(callback),
            effect::Event::Other { .. } => None
        }
    }

    fn pop_frame(&mut self) -> Option<Frame> {
        if self._effect.frames.is_empty() { return None; }

        let result = self._effect.frames.remove(0);
        for x in &mut self._effect.events { x.update_frame(|x| x - 1); }
        self._frame_mask.clear();
        Some(result)
    }

    fn pov_sees_effect(&mut self, pov: EID, fov: &mut FOV) -> bool {
        let Some(frame) = self.get_frame() else { return false };
        let Some(me) = self.entities.get(pov) else { return false };
        let count = frame.len();

        if self._frame_mask.is_empty() {
            self._frame_mask = self._compute_frame_mask(me, frame, fov);
        }
        assert!(self._frame_mask.len() == count);

        // Cheap check: are any of the particles directly visible?
        if self._frame_mask.iter().any(|&x| x) { return true; }

        // Expensive check: does lighting for any visible cell change?
        let seen = fov.select_vision(me).get_points_seen();
        let prev: Vec<_> = seen.iter().map(|&x| self.is_cell_lit(x)).collect();
        self.redo_effect_updates();
        let next: Vec<_> = seen.iter().map(|&x| self.is_cell_lit(x)).collect();
        self.undo_effect_updates();

        prev != next
    }

    fn redo_effect_updates(&mut self) {
        self._enter_effect_frame(true);
    }

    fn undo_effect_updates(&mut self) {
        self._enter_effect_frame(false);
    }

    fn _compute_frame_mask(&self, me: &Entity, frame: &Frame, fov: &mut FOV) -> Vec<bool> {
        assert!(self._frame_mask.is_empty());

        let vision = fov.select_vision(me);

        frame.iter().map(|x| match &x.data {
            ParticleData::Light(..) => false,
            ParticleData::Shift(..) => vision.can_see(x.point),
            ParticleData::Sight(..) => vision.can_see(x.point),
            ParticleData::Sound(volume, ..) => volume.contains(x.point - me.pos),
        }).collect()
    }

    fn _enter_effect_frame(&mut self, active: bool) {
        let Some(mut frame) = self.get_frame().cloned() else { return };

        if !active { frame.reverse(); }

        frame.iter().for_each(|x| match &x.data {
            &ParticleData::Light(light) => {
                let eid = self.get_cell(x.point).eid;
                let prev = eid.map(|x| self.entities[x].species.light.radius).unwrap_or(-1);
                let next = std::cmp::max(prev, light.radius);
                if prev == next { return; }

                let radius = if active { next } else { prev };
                self.lighting.set_light(x.point, radius);
            },
            &ParticleData::Shift(mut source) => {
                let mut target = x.point;
                if source == target { return; }

                if !active { swap(&mut source, &mut target); }

                let None = self.get_cell(target).eid else { return };
                let Some(eid) = self.get_cell(source).eid else { return };

                self.move_entity(eid, target);
            },
            ParticleData::Sight(..) => {},
            ParticleData::Sound(..) => {},
        });
    }

    // Getters:

    pub fn animation_running(&self) -> bool { self.get_frame().is_some() }

    pub fn get_cell(&self, p: Point) -> &Cell { self.map.entry_ref(p) }

    pub fn get_entity(&self, eid: EID) -> Option<&Entity> { self.entities.get(eid) }

    pub fn get_frame(&self) -> Option<&Frame> { self._effect.frames.first() }

    pub fn get_light(&self) -> &Light { &self.light }

    pub fn get_size(&self) -> Point { self.map.size() }

    pub fn get_status(&self, p: Point) -> Status {
        let Cell { eid, tile, .. } = self.get_cell(p);
        if tile.blocks_movement() { return Status::Blocked; }
        if eid.is_some() { Status::Occupied } else { Status::Free }
    }

    pub fn get_tile(&self, p: Point) -> &'static Tile { self.get_cell(p).tile }

    pub fn is_cell_lit(&self, p: Point) -> bool { self.lighting.get_light(p) > 0 }

    // Item setters:

    fn add_item(&mut self, eid: Option<EID>, pos: Point, item: Item) {
        let Some(cell) = self.map.entry_mut(pos) else { return };

        cell.items.push(item);

        let Some(eid) = eid else { return };
        let Some(entity) = self.entities.get_mut(eid) else { return };

        entity.known.update_items(pos, &item);
    }

    fn remove_item(&mut self, pos: Point, item: Item) -> bool {
        let Some(c) = self.map.entry_mut(pos) else { return false };
        let Some(i) = c.items.iter().position(|&x| x == item) else { return false };

        c.items.remove(i);
        true
    }

    // Entity setters:

    fn add_entity(&mut self, args: &EntityArgs, env: &mut Env) -> EID {
        let pos = args.pos;
        let eid = self.entities.add(args, &mut env.rng);
        let cell = self.map.entry_mut(pos).unwrap();
        let prev = replace(&mut cell.eid, Some(eid));
        assert!(prev.is_none());

        self.update_known(eid, env);
        let entity = &mut self.entities[eid];
        entity.known.mark_turn_boundary(entity.player, entity.speed, self.time);

        let light = args.species.light.radius;
        self.lighting.set_light(pos, light);

        eid
    }

    fn move_entity(&mut self, eid: EID, target: Point) {
        let entity = &mut self.entities[eid];
        let source = replace(&mut entity.pos, target);
        let light = entity.species.light.radius;

        let old = replace(&mut self.map.entry_mut(source).unwrap().eid, None);
        assert!(old == Some(eid));
        let new = replace(&mut self.map.entry_mut(target).unwrap().eid, old);
        assert!(new.is_none());

        self.lighting.set_light(source, -1);
        self.lighting.set_light(target, light);
    }

    fn swap_entities(&mut self, source: Point, target: Point) {
        assert!(source != target);

        let old = self.map.entry_mut(source).unwrap().eid.unwrap();
        let new = self.map.entry_mut(target).unwrap().eid.unwrap();

        let old_entity = &mut self.entities[old];
        let old_light = old_entity.species.light.radius;
        old_entity.pos = target;

        let new_entity = &mut self.entities[new];
        let new_light  = new_entity.species.light.radius;
        new_entity.pos = source;

        self.map.entry_mut(source).unwrap().eid = Some(new);
        self.map.entry_mut(target).unwrap().eid = Some(old);

        self.lighting.set_light(source, new_light);
        self.lighting.set_light(target, old_light);
    }

    fn remove_entity(&mut self, eid: EID) {
        // The player entity is not removed, since it's the player's POV.
        let entity = &mut self.entities[eid];
        let &mut Entity { pos, player, leader, .. } = entity;
        let team: Vec<_> = entity.team.iter().filter_map(
            |x| if let &Teammate::Out(oid) = x { Some(oid) } else { None }).collect();
        let ind = entity.to_individual();
        if player { return; }

        // Remove the entity's light source.
        self.lighting.set_light(pos, -1);

        // Remove the entity from the spatial map.
        let existing = self.map.entry_mut(pos).unwrap().eid.take();
        assert!(existing == Some(eid));

        // Remove the entity from the entities SlotMap.
        let okay = self.entities.remove(eid).is_some();
        assert!(okay);

        // Mark the entity as inactive, if it was active.
        if self.active_entity == Some(eid) { self.active_entity = None; }

        // Delete edges between this entity and its teammates.
        if let Some(x) = leader {
            let leader = &mut self.entities[x];
            for edge in &mut leader.team {
                if !matches!(edge, Teammate::Out(x) if *x == eid) { continue; }
                *edge = Teammate::In(ind);
                break;
            }
            leader.summons.retain(|&x| x != eid);
        }
        for x in team { self.entities[x].leader = None; }
    }

    fn reset(&mut self, tile: &'static Tile) {
        self.map.fill(Cell { tile, ..Default::default() });
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
        self.lighting.set_opacity(point, tile.opacity());
    }

    // Knowledge:

    fn create_event(&self, eid: EID, data: EventData, pos: Point) -> Event {
        let (eid, uid) = (Some(eid), None);
        let loc = Location { pos, time: self.time };
        Event { eid, uid, loc, data, sense: Sense::Sight }
    }

    fn observe_event(&mut self, eid: EID, s: &Senses, e: &mut Event, env: &mut Env) {
        let Some(entity) = self.entities.get_mut(eid) else { return };

        let (heard, seen) = (s.heard(), s.seen() && !entity.player);
        if !heard && !seen { return; }

        let known = &mut env.known;
        swap(known, &mut entity.known);

        e.sense = if seen { Sense::Sight } else { Sense::Sound };
        known.observe_event(&mut self.entities[eid], e);

        swap(known, &mut self.entities[eid].known);
    }

    fn remove_known_entity(&mut self, eid: EID, oid: EID) {
        let Some(entity) = self.entities.get_mut(eid) else { return };

        entity.known.remove_entity(oid, self.time);

        let command = &entity.command;
        if let Some(Command::Attack(_, x)) = command.get() && x.eid == Some(oid) {
            command.take();
        }
    }

    fn update_known(&mut self, eid: EID, env: &mut Env) {
        let Some(entity) = self.entities.get_mut(eid) else { return };

        let Env { known, fov, rng, .. } = env;
        swap(known, &mut entity.known);

        let me = &self.entities[eid];
        let vision = fov.compute(&self, me);
        known.update(me, &self, vision, rng);

        swap(known, &mut self.entities[eid].known);
    }

    // Lighting:

    fn update_shadow(&mut self, point: Point, delta: i32) {
        if delta == 0 { return; }
        for &x in &self.shadow {
            let Some(cell) = self.map.entry_mut(point + x) else { continue; };
            cell.shadow += delta;
            assert!(cell.shadow >= 0);
        }
    }

    fn update_edge_shadows(&mut self) {
        let delta = if self.map.default().tile.casts_shadow() { 1 } else { 0 };
        if delta == 0 || self.shadow.is_empty() { return; }

        let Point(sx, sy) = self.map.size();
        for x in -1..=sx {
            self.update_shadow(Point(x, -1), delta);
            self.update_shadow(Point(x, sy), delta);
        }
        for y in 0..sy {
            self.update_shadow(Point(-1, y), delta);
            self.update_shadow(Point(sx, y), delta);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Event delivery

type SenseMap = HashMap<EID, Senses>;

const SENSE_SEEN: u8 = 1 << 0;
const SENSE_HEARD: u8 = 1 << 1;

#[derive(Clone, Copy, Default)]
struct Senses(u8);

impl Senses {
    fn seen(&self) -> bool { self.0 & SENSE_SEEN != 0 }
    fn heard(&self) -> bool { self.0 & SENSE_HEARD != 0 }
}

impl std::ops::BitOr for Senses {
    type Output = Senses;
    fn bitor(self, other: Self) -> Self::Output {
        Self(self.0 | other.0)
    }
}

struct Noise {
    cause: Option<EID>,
    check: Option<EID>,
    point: Point,
    volume: Bound,
}

struct Sighting {
    eid: EID,
    merged: Senses,
    source: Senses,
    target: Senses,
}

impl Noise {
    fn from_eid(eid: EID, point: Point, volume: Bound) -> Self {
        Self { cause: Some(eid), check: Some(eid), point, volume }
    }

    fn from_entity(me: &Entity, volume: Bound) -> Self {
        Self::from_eid(me.eid, me.pos, volume)
    }

    fn for_target(me: &Entity, point: Point, volume: Bound, target: Option<EID>) -> Self {
        Self { cause: Some(me.eid), check: target, point, volume }
    }
}

fn detect(board: &Board, noise: &Noise, env: &mut Env) -> SenseMap {
    let Noise { cause, check, point, volume } = *noise;
    let mut result = SenseMap::default();

    let other = check.and_then(|x| board.entities.get(x));

    for (eid, me) in &board.entities {
        if cause == Some(eid) { continue; }
        if me.asleep && volume.radius == 1 && point != me.pos { continue; }

        let seen = if let Some(other) = other {
            env.fov.can_see_entity(board, me, other)
        } else {
            env.fov.can_see(board, me, point)
        };
        let heard = volume.contains(point - me.pos);
        if !seen && !heard { continue; }

        let seen = if seen { SENSE_SEEN } else { 0 };
        let heard = if heard { SENSE_HEARD } else { 0 };
        result.insert(eid, Senses(seen | heard));
    }
    result
}

fn merge_views(board: &Board, saw_source: &SenseMap, saw_target: &SenseMap) -> Vec<Sighting> {
    let mut result = vec![];

    for (eid, _) in &board.entities {
        let source = saw_source.get(&eid).copied();
        let target = saw_target.get(&eid).copied();
        if source.is_none() && target.is_none() { continue; }

        let source = source.unwrap_or_default();
        let target = target.unwrap_or_default();
        result.push(Sighting { eid, merged: source | target, source, target })
    }
    result
}

fn get_sightings(board: &Board, noise: &Noise, env: &mut Env) -> Vec<Sighting> {
    let seen = detect(board, noise, env);
    merge_views(board, &seen, &Default::default())
}

//////////////////////////////////////////////////////////////////////////////

// Attack effects

fn shout(state: &mut State, eid: EID, shout: &str, suffix: &str) {
    let board = &mut state.board;
    let Entity { pos: source, player, species, .. } = board.entities[eid];
    let noise = Noise::from_eid(eid, source, SHOUT_VOLUME);
    let sightings = get_sightings(board, &noise, &mut state.env);

    let log = &mut state.env.ui.log;
    if player && suffix.is_empty() {
        log.log_success(format!("You shout: \"{}\"", shout));
    } else if player {
        log.log_failure(format!("You shout: \"{}\"{}", shout, suffix));
    }

    // Create a call event that carries species info.
    let data = EventData::Call(CallEvent { call: Call::Command, species });
    let event = &mut board.create_event(eid, data, source);

    // Deliver the CallEvent to each other entity that heard the shout.
    for s in &sightings {
        board.observe_event(s.eid, &s.merged, event, &mut state.env);
        if s.eid != state.player { continue; }

        let log = &mut state.env.ui.log;
        let entity = &board.entities[eid];
        let source = if s.merged.seen() { entity.upper() } else { "Someone".into() };
        log.log_notable(format!("{} shouts: \"{}\"", source, shout));
    }
}

fn hit_tile(state: &mut State, eid: EID, target: Point) {
    let State { board, env, .. } = state;
    if !board.get_tile(target).drops_berries() { return; }

    let options: Vec<_> = dirs::ALL.clone().into_iter().filter(
        |&x| board.get_status(target + x) != Status::Blocked).collect();
    if options.is_empty() { return; }

    let rng = &mut env.rng;
    let n = *weighted(&[(1, 0), (2, 1), (1, 2)], rng);
    for _ in 0..n {
        let pos = target + *sample(&options, rng);
        board.add_item(Some(eid), pos, Item::Berry);
    }
}

fn hit_entity(state: &mut State, eid: EID, attack: &Attack, logged: bool, tid: EID) {
    let State { board, env, .. } = state;
    let Some(target) = board.entities.get_mut(tid) else { return; };

    let (pos, lower, upper) = (target.pos, target.lower(), target.upper());

    let critted = env.rng.random_range(0..16) == 0;
    let factor = if critted { 1.5 } else { 1. } * env.rng.random_range(0.85..=1.);
    let damage = (factor * attack.damage as f64).round() as i32;
    let damage = if target.species.name == "Human" { 1 } else { damage };

    target.cur_hp = std::cmp::max(target.cur_hp - damage, 0);
    let fainted = target.cur_hp == 0;

    let noise = Noise::from_entity(target, ATTACK_VOLUME);
    let sightings = get_sightings(board, &noise, env);

    if fainted {
        board.remove_entity(tid);
        board.add_item(Some(eid), pos, Item::Corpse);
    }

    for s in &sightings {
        let oid = s.eid;
        if fainted { board.remove_known_entity(oid, tid); }
        if !s.source.seen() || !board.entities[oid].player { continue; }

        let log = &mut env.ui.log;
        if !logged { log.log(format!("Something attacked {}!", lower)); }
        if critted { log.log_append("A critical hit!"); }
        if fainted { log.log_append(format!("{} fainted!", upper)); }
    }
}

fn try_recall_entity(state: &mut State, eid: EID, oid: EID) -> bool {
    let entity = &state.board.entities[eid];
    let summon = &state.board.entities[oid];
    if summon.leader != Some(eid) { return false; }

    let (source, target) = (entity.pos, summon.pos);
    if !can_summon(&state.board, entity, target) { return false; }

    let cb: CB = Box::new(move |x| x.board.remove_entity(oid));
    let effect = effect::WithdrawEffect(source, target);
    state.add_effect(apply_effect(effect, FT::Withdraw, cb));
    true
}

//////////////////////////////////////////////////////////////////////////////

// Turn-taking

pub fn move_ready(me: &Entity) -> bool { me.move_timer <= 0 }

pub fn turn_ready(me: &Entity) -> bool { me.turn_timer <= 0 }

fn drain(me: &mut Entity, result: &ActionResult) {
    me.move_timer += (MOVE_TIMER as f64 * result.moves).round() as i32;
    me.turn_timer += (TURN_TIMER as f64 * result.turns).round() as i32;
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

#[derive(Debug)]
pub enum Action {
    Idle,
    Rest,
    WaitForInput,
    SniffAround,
    Look { look: Point },
    Drink { target: Point },
    Eat { target: Point, item: Option<Item> },
    Call { look: Point, call: Call },
    Move { look: Point, step: Point, turns: f64 },
    Attack { target: Point, attack: &'static Attack },
    Recall { summon: usize },
    Summon { team: usize, target: Point },
    Shout { summon: usize, command: Command },
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

fn can_attack(board: &Board, me: &Entity, target: Point, range: Bound) -> bool {
    let (known, source) = (&me.known, me.pos);

    if source == target { return false; }
    if !known.get(target).visible() { return false; }
    if !range.contains(source - target) { return false; }

    let los = LOS(source, target);
    los.iter().skip(1).rev().skip(1).all(
        |&p| known.get(p).status() == Status::Free && board.get_status(p) == Status::Free)
}

fn can_summon(board: &Board, me: &Entity, target: Point) -> bool {
    let (known, range, source) = (&me.known, SUMMON_RANGE, me.pos);

    if source == target { return false; }
    if !range.contains(source - target) { return false; }
    if !known.get(target).can_see_entity_at() { return false; }

    let los = LOS(source, target);
    los[1..los.len() - 1].iter().all(|&p| {
        if known.get(p).status() != Status::Free { return false; }
        let status = board.get_status(p);
        status == Status::Free || status == Status::Occupied
    })
}

fn plan(state: &mut State, eid: EID, leader: Option<EID>) -> Action {
    let env = &mut state.env;
    state.board.update_known(eid, env);
    if let Some(x) = leader { state.board.update_known(x, env); }

    let player = eid == state.player;
    if player { return replace(&mut state.input, Action::WaitForInput); }

    let State { board, env, .. } = state;
    let debug = env.debug.as_deref_mut();
    let vision = &mut env.fov.npc_vision;
    let entity = &mut board.entities[eid];
    let ai = &mut env.ai;

    swap(ai, &mut entity.ai);

    let entity = &board.entities[eid];
    let leader = leader.map(|x| &board.entities[x]);
    let env = AIEnv { leader, debug, fov: vision, rng: &mut env.rng };
    let action = ai.plan(entity, env);

    let entity = &mut board.entities[eid];
    swap(ai, &mut entity.ai);

    entity.known.events.clear();

    action
}

fn act(state: &mut State, eid: EID, action: Action) -> ActionResult {
    let me = &mut state.board.entities[eid];
    me.asleep = matches!(action, Action::Rest);
    let source = me.pos;

    match action {
        Action::Idle => ActionResult::success(),
        Action::Rest => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::SniffAround => {
            let noise = Noise::from_entity(me, SNIFF_VOLUME);
            let board = &mut state.board;
            let sightings = get_sightings(board, &noise, &mut state.env);

            // Deliver a SniffEvent to each other entity that heard the sniff.
            let data = EventData::Sniff;
            let event = &mut board.create_event(eid, data, source);
            for s in &sightings {
                board.observe_event(s.eid, &s.merged, event, &mut state.env);
            }

            let effect = apply_noise(source, 0xffff00, "*sniff*", noise.volume);
            state.add_effect(effect);
            ActionResult::success()
        }
        Action::Look { look } => {
            me.face_direction(look);
            ActionResult::success()
        }
        Action::Drink { target } => {
            let (source, dir) = (me.pos, target - me.pos);
            if dir.len_l1() > 1 { return ActionResult::failure(); }

            me.face_direction(dir);
            let okay = state.board.get_cell(target).tile.can_drink();
            if !okay { return ActionResult::failure(); }

            let color = 0x0080ff;
            let effect = Effect::serial(vec![
                flash_tile(target, color, None),
                flash_entity(source, color).delay(UI_FLASH / 2),
            ]);
            state.add_effect(effect);
            ActionResult::success()
        }
        Action::Eat { target, item } => {
            let dir = target - source;
            if dir.len_l1() > 1 { return ActionResult::failure(); }

            me.face_direction(dir);
            let cell = state.board.get_cell(target);
            let okay = match item {
                Some(x) => cell.items.iter().find(|&&y| y == x).is_some(),
                None => cell.tile.can_eat(),
            };
            if !okay { return ActionResult::failure(); }

            let color = if item == Some(Item::Corpse) { 0xff0000 } else { 0xffff00 };
            let cb: CB = Box::new(move |state| {
                let Some(item) = item else { return };
                state.board.remove_item(target, item);
            });
            let effect = Effect::serial(vec![
                flash_tile(target, color, Some(cb)),
                flash_entity(source, color).delay(UI_FLASH / 2),
            ]);
            state.add_effect(effect);
            ActionResult::success()
        }
        Action::Call { call, look } => {
            let species = me.species;
            let noise = Noise::from_entity(me, CALL_VOLUME);
            let board = &mut state.board;
            let sightings = get_sightings(board, &noise, &mut state.env);

            // Deliver a CallEvent to each other entity that heard the call.
            let data = EventData::Call(CallEvent { call, species });
            let event = &mut board.create_event(eid, data, source);
            for s in &sightings {
                board.observe_event(s.eid, &s.merged, event, &mut state.env);
            }

            // Use a different color for different call types.
            let color = 0xff8000;
            let (text, wait) = match call {
                Call::Command => ("*shout*", false),
                Call::Help    => ("*chirp*", true),
                Call::Warning => ("*grrr*",  false),
            };

            // For some call types, we look before calling; when calling for
            // help, we shout in the direction of our allies, then look.
            let cb = move |state: &mut State| {
                let Some(me) = state.board.entities.get_mut(eid) else { return };
                me.face_direction(look);
            };

            let mut effect = apply_noise(source, color, text, CALL_VOLUME);
            if wait {
                effect.sub_on_finished(Box::new(cb));
            } else {
                cb(state);
            }
            state.add_effect(effect);
            ActionResult::success()
        }
        Action::Move { look, step, turns } => {
            me.face_direction(look);
            let slowed = turns < SLOWED_TURNS && !move_ready(me);
            let turns = if slowed { SLOWED_TURNS } else { turns };
            if step == dirs::NONE { return ActionResult::success_turns(turns); }
            if step.len_l1() > 1 { return ActionResult::failure(); }

            // Moving diagonally is slower. Moving quickly is noisier.
            let noisy = turns <= 1.;
            let turns = step.len_l2() * turns;
            let color = me.species.glyph.fg();
            let player = me.player;
            let target = source + step;
            let allied = player || me.leader == Some(state.player);

            let (board, log) = (&mut state.board, &mut state.env.ui.log);
            let cell = board.get_cell(target);
            let swap = cell.eid.is_some();

            if cell.tile.blocks_movement() {
                board.entities[eid].face_direction(step);
                return ActionResult::failure();
            }

            if let Some(x) = cell.eid && board.entities[x].leader != Some(eid) {
                board.entities[eid].face_direction(step);
                if player { log.log_failure("There's something in the way!"); }
                return ActionResult::failure();
            }

            board.time = board.time.bump();

            let volume = if noisy { MOVE_VOLUME } else { SNEAK_VOLUME };
            let noise = Noise::from_eid(eid, source, volume);
            let saw_source = detect(board, &noise, &mut state.env);

            if swap {
                board.swap_entities(source, target);
            } else {
                board.move_entity(eid, target);
            }

            let noise = Noise::from_eid(eid, target, volume);
            let saw_target = detect(board, &noise, &mut state.env);
            let sightings = merge_views(board, &saw_source, &saw_target);

            // Deliver a MoveEvent to each other entity that saw the move.
            let data = EventData::Move(MoveEvent { from: source });
            let event = &mut board.create_event(eid, data, target);
            for s in &sightings {
                state.board.observe_event(s.eid, &s.merged, event, &mut state.env);
                if allied || s.eid != state.player { continue; }

                let color = if s.merged.seen() { color } else { Color::white() };
                state.env.ui.animate_move(color, source, 0);
                state.env.ui.animate_move(color, target, 1);
            }
            ActionResult::success_turns(turns)
        }
        Action::Attack { attack, target } => {
            let board = &mut state.board;
            let me = &board.entities[eid];
            if !can_attack(board, me, target, attack.range) {
                board.entities[eid].face_direction(target - source);
                return ActionResult::failure();
            }

            board.time = board.time.bump();

            let volume = ATTACK_VOLUME;
            let noise = Noise::from_entity(me, volume);
            let saw_source = detect(board, &noise, &mut state.env);

            let dir = target - source;
            let tid = board.get_cell(target).eid;
            let me = &mut board.entities[eid];
            me.face_direction(dir);

            let noise = Noise::for_target(me, target, volume, tid);
            let saw_target = detect(board, &noise, &mut state.env);
            let sightings = merge_views(board, &saw_source, &saw_target);

            // Deliver an AttackEvent to each other entity that heard the attack.
            let combat = tid.is_some();
            let data = EventData::Attack(AttackEvent { combat, target: None });
            let event = &mut board.create_event(eid, data, source);
            let mut logged = false;
            for s in &sightings {
                let oid = s.eid;
                let target = if s.target.seen() { tid } else { None };
                event.data = EventData::Attack(AttackEvent { combat, target });
                state.board.observe_event(oid, &s.source, event, &mut state.env);
                if oid != state.player { continue; }

                let entities = &state.board.entities;
                let attacker = if s.source.seen() { Some(&entities[eid]) } else { None };
                let attacked = if s.target.seen() { tid.map(|x| &entities[x]) } else { None };
                logged = true;

                let line = if let Some(a) = attacker && let Some(b) = attacked {
                    format!("{} attacked {} with {}!", a.upper(), b.lower(), attack.name)
                } else if let Some(a) = attacker {
                    format!("{} used {}!", a.upper(), attack.name)
                } else if let Some(b) = attacked {
                    format!("Something attacked {}!", b.lower())
                } else {
                    "You hear fighting nearby!".into()
                };
                state.env.ui.log.log(line);
            }

            let cb: CB = Box::new(move |state| {
                hit_tile(state, eid, target);
                let Some(tid) = tid else { return; };

                let cb: CB = Box::new(move |x| { hit_entity(x, eid, attack, logged, tid); });
                state.add_effect(apply_damage(target, cb));
            });

            let rng = &mut state.env.rng;
            let shift = Particle::shift(source, source);
            let effect = (attack.effect)(rng, source, target);
            let effect = Effect::constant(shift, effect.frames.len() as i32).and(effect);
            state.add_effect(apply_effect(effect, FT::Hit, cb));
            ActionResult::success_moves(1.)
        }
        Action::Recall { summon } => {
            let summon = me.summons.get(summon);
            let Some(&oid) = summon else { return ActionResult::failure() };
            if !try_recall_entity(state, eid, oid) { return ActionResult::failure(); }
            ActionResult::success()
        }
        Action::Summon { team, target } => {
            let teammate = me.team.get(team);
            let Some(Teammate::In(x)) = teammate else { return ActionResult::failure() };

            let Individual { species, cur_hp } = *x;
            if cur_hp == 0 { return ActionResult::failure(); }

            let me = &state.board.entities[eid];
            if !can_summon(&state.board, me, target) { return ActionResult::failure(); }

            shout(state, eid, &format!("Go! {}!", species.name), "");

            let cb: CB = Box::new(move |state| {
                let State { board, env, .. } = state;
                let (name, leader, player) = (None, Some(eid), false);
                let args = EntityArgs { name, pos: target, player, leader, species };
                let oid = board.add_entity(&args, env);

                let other = &mut board.entities[oid];
                other.cur_hp = cur_hp;

                let me = &mut board.entities[eid];
                me.team[team] = Teammate::Out(oid);
                me.summons.push(oid);
            });

            let effect = effect::SummonEffect(source, target);
            state.add_effect(apply_effect(effect, FT::Summon, cb));
            ActionResult::success()
        }
        Action::Shout { summon, command } => {
            let summon = me.summons.get(summon);
            let Some(&oid) = summon else { return ActionResult::failure() };

            let summon = &state.board.entities[oid];
            let name = summon.species.name;

            let succeed = |state: &mut State, suffix: &str| {
                let command = match command {
                    Command::Attack(attack, target) => {
                        let foe = target.eid.and_then(|x| state.board.entities.get(x));
                        if let Some(foe) = foe.map(|x| x.species.name) {
                            format!("{}, attack {} with {}!", name, foe, attack.name)
                        } else {
                            format!("{}, use {}!", name, attack.name)
                        }
                    },
                    Command::Return => format!("{}, return!", name),
                };
                shout(state, eid, &command, suffix);
                ActionResult::success()
            };

            if !SHOUT_VOLUME.contains(summon.pos - source) {
                let suffix = format!(", but {} is too far away to hear.", name);
                return succeed(state, &suffix);
            }

            let done = matches!(command, Command::Return) && try_recall_entity(state, eid, oid);
            if done { return ActionResult::success(); }

            let summon = &mut state.board.entities[oid];
            summon.command.set(Some(command));

            succeed(state, "")
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Animation

fn flash_entity<T: Into<Color>>(target: Point, color: T) -> Effect {
    let frame = vec![
        Particle::shift(target, target),
        Particle::flash(target, color.into()),
    ];
    Effect::repeat(frame, UI_FLASH)
}

fn flash_tile<T: Into<Color>>(target: Point, color: T, cb: Option<CB>) -> Effect {
    let frame = vec![Particle::flash(target, color.into())];
    let mut effect = Effect::repeat(frame, UI_FLASH);
    if let Some(x) = cb { effect.sub_on_finished(x); }
    effect
}

fn apply_noise<T: Copy + Into<Color>>(
        target: Point, color: T, text: &'static str, volume: Bound) -> Effect {
    let frame = vec![Particle::noise(target, color.into(), volume)];
    let mut effect = Effect::repeat(frame, UI_NOISE);
    effect.frames[0].push(Particle::sound(target, text, volume));
    effect
}

fn apply_damage(target: Point, cb: CB) -> Effect {
    let dummy = vec![Particle::dummy(target)];
    let frame = vec![
        Particle::shift(target, target),
        Particle::flash(target, 0xff0000.into()),
    ];
    let mut effect = Effect::serial(vec![
        Effect::repeat(frame, UI_DAMAGE_FLASH),
        Effect::repeat(dummy, UI_DAMAGE_TICKS),
    ]);
    effect.sub_on_finished(cb);
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
    let player = &mut state.board.entities[state.player];
    state.env.ui.process_input(player, input);
    state.input = state.env.ui.action.take().unwrap_or(Action::WaitForInput);
}

fn update_player_knowledge(state: &mut State) {
    let eid = state.player;
    let State { board, env, .. } = state;

    board.update_known(eid, env);
    let player = &board.entities[eid];
    env.ui.update_focus(player);
    env.ui.update_moves(player);

    let Some(frame) = board.get_frame() else { return };

    let mut render_particle = |p: Point, r: &RenderData| {
        let RenderData::Text(t) = r else { return };
        env.ui.animate_text(p, t);
    };
    frame.iter().zip(&board._frame_mask).filter(|x| *x.1).for_each(|x| match &x.0.data {
        ParticleData::Light(..) => {},
        ParticleData::Shift(..) => {},
        ParticleData::Sight(r) => render_particle(x.0.point, r),
        ParticleData::Sound(_, r) => render_particle(x.0.point, r),
    });

    board.redo_effect_updates();

    swap(&mut env.known, &mut board.entities[eid].known);
    board.update_known(eid, env);
    swap(&mut env.known, &mut board.entities[eid].known);

    board.undo_effect_updates();
}

fn update_state(state: &mut State) {
    let pos = state.get_player().pos;
    state.env.ui.update(pos, &mut state.env.rng);

    // If an Effect is active, run it, skipping frames the player can't see.
    if state.advance_effect() {
        update_player_knowledge(state);
        return;
    }

    // The game loop is interrupted by animations, and if the player dies.
    let game_loop_active = |state: &State| {
        !state.board.animation_running() && state.get_player().cur_hp > 0
    };
    let stage_input = |state: &mut State| {
        if !game_loop_active(state) { return true; }
        if !matches!(state.input, Action::WaitForInput) { return true; }

        // Automatically stage the recall action.
        let player = state.get_player();
        for (i, &summon) in player.summons.iter().enumerate() {
            let summon = &state.board.entities[summon];
            if !matches!(summon.command.get(), Some(Command::Return)) { continue; }
            if !can_summon(&state.board, player, summon.pos) { continue; }

            state.input = Action::Recall { summon: i };
            summon.command.take();
            return true;
        }
        false
    };

    let mut update = false;
    while !state.inputs.is_empty() && !stage_input(state) {
        let input = state.inputs.remove(0);
        process_input(state, input);
        update = true;
    }
    let player = &state.board.entities[state.player];
    if state.env.ui.update_target(player) { return; }

    while game_loop_active(state) {
        let board = &mut state.board;
        let Some(eid) = advance_turn(board) else { break };

        let time = board.time;
        let entity = &board.entities[eid];
        let Entity { leader, player, .. } = *entity;
        if player && !stage_input(state) { break; }

        update = true;
        let action = plan(state, eid, leader);
        state.record_trace(&action, eid);
        let result = act(state, eid, action);
        if player && !result.success { break; }

        let board = &mut state.board;
        let Some(entity) = board.entities.get_mut(eid) else { continue };

        let Entity { pos, speed, .. } = *entity;
        entity.known.mark_turn_boundary(player, speed, time);

        board.time = board.time.bump();

        let trail = &mut entity.trail;
        if trail.len() == trail.capacity() { trail.pop_back(); }
        trail.push_front(Location { pos, time: board.time });

        board.active_entity = None;
        drain(entity, &result);

        state.start_effect();
    }
    if update { update_player_knowledge(state); }
}

//////////////////////////////////////////////////////////////////////////////

// State

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum GameMode { Debug, Gym, Play, Sim, Test }

pub struct Env {
    // Used for in-place Entity state updates:
    ai: Box<AIState>,
    debug: Option<Box<DebugFile>>,
    known: Box<Knowledge>,

    // Other update helpers:
    fov: FOV,
    rng: RNG,
    ui: UI,
}

pub struct State {
    board: Board,
    input: Action,
    inputs: Vec<Input>,
    player: EID,
    env: Env,
}

impl Default for State {
    fn default() -> Self {
        Self::new(/*seed=*/None, GameMode::Play)
    }
}

impl State {
    pub fn new(seed: Option<u64>, mode: GameMode) -> Self {
        let size = Point(WORLD_SIZE, WORLD_SIZE);
        let rng = seed.map(|x| RNG::seed_from_u64(x));
        let rng = rng.unwrap_or_else(|| RNG::from_os_rng());
        let debug = matches!(mode, GameMode::Debug | GameMode::Sim);

        let mut board = Board::new(size, LIGHT);
        let mut pos = Point(size.0 / 2, size.1 / 2);
        let mut rng = rng;

        loop {
            let map = mapgen(size, &mut rng);
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

        let mut env = Env {
            ai: Box::new(AIState::new(&mut rng)),
            debug: if debug { Some(Default::default()) } else { None },
            known: Default::default(),
            fov: Default::default(),
            ui: Default::default(),
            rng,
        };

        let input = Action::WaitForInput;
        let species = Species::get("Human");
        let (name, leader, player) = (Some("skishore".into()), None, true);
        let args = EntityArgs { name, pos, player, leader, species };
        let player = board.add_entity(&args, &mut env);

        if matches!(mode, GameMode::Gym | GameMode::Sim | GameMode::Test) {
            board.map.entry_mut(pos).unwrap().eid = None;
            let me = &mut board.entities[player];
            let Entity { player, speed, .. } = *me;
            me.known = Default::default();
            me.known.mark_turn_boundary(player, speed, board.time);
            me.pos = Point(-9999, -9999);
        }

        let pos = |board: &Board, rng: &mut RNG| {
            for _ in 0..100 {
                let Point(x, y) = size;
                let p = Point(rng.random_range(0..x), rng.random_range(0..y));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        for i in 0..(NUM_PREDATORS + NUM_PREY) {
            if let Some(x) = pos(&board, &mut env.rng) {
                let predator = i < NUM_PREDATORS;
                let species = match (predator, i % 2) {
                    (true, 0)  => "Rattata",
                    (true, _)  => "Charmander",
                    (false, _) => "Pidgey",
                };
                let species = Species::get(species);
                let (name, leader, player) = (None, None, false);
                let args = EntityArgs { name, pos: x, player, leader, species };
                board.add_entity(&args, &mut env);
            }
        }

        let teammate = |name: &str| {
            let species = Species::get(name);
            Teammate::In(Individual { species, cur_hp: species.hp })
        };
        let me = &mut board.entities[player];
        me.dir = dirs::S;
        me.team.push(teammate("Bulbasaur"));
        me.team.push(teammate("Charmander"));
        me.team.push(teammate("Squirtle"));
        me.team.push(teammate("Pikachu"));
        me.team.push(teammate("Eevee"));
        board.update_known(player, &mut env);

        let ui = &mut env.ui;
        let inputs = Default::default();
        std::mem::drop(Weather::Rain(Point(0, 64), 32));
        match WEATHER {
            Weather::Rain(angle, count) => ui.start_rain(angle, count),
            Weather::None => (),
        }
        ui.log.log("Welcome to WildsRL! Use vikeys (h/j/k/l/y/u/b/n) to move.");

        Self { board, input, inputs, player, env }
    }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Buffer) {
        let entity = self.get_player();
        let effect = self.board.get_frame().map(|frame| {
            let known = &*self.env.known;
            let mask = &self.board._frame_mask;

            let mut sources = HashSet::default();
            let mut targets = HashSet::default();

            for particle in frame {
                let target = particle.point;
                let &ParticleData::Shift(source) = &particle.data else { continue };

                targets.remove(&source);
                sources.insert(source);
                targets.insert(target);
            }
            super::ui::Effect { frame, known, mask, sources, targets }
        });
        self.env.ui.render(buffer, entity, effect.as_ref());
    }

    // Private helpers:

    fn get_player(&self) -> &Entity { &self.board.entities[self.player] }

    fn record_trace(&mut self, action: &Action, eid: EID) {
        if matches!(action, Action::WaitForInput) { return; }
        let Some(debug) = &mut self.env.debug else { return };

        let board = &self.board;
        let entity = &board.entities[eid];
        debug.record(action, board, entity);
    }

    // Animation:

    fn add_effect(&mut self, effect: Effect) {
        self.board.add_effect(effect);
        self._execute_effect_callbacks();
    }

    fn advance_effect(&mut self) -> bool {
        if !self.board.animation_running() { return false; }

        self._advance_one_frame();
        self.start_effect()
    }

    fn start_effect(&mut self) -> bool {
        if !self.board.animation_running() { return false; }

        let pov = self.player;
        self._arm_animation_fov(pov);

        while self.board.animation_running() {
            if self.board.pov_sees_effect(pov, &mut self.env.fov) { return true; }
            if self._advance_one_frame() { self._arm_animation_fov(pov); }
        }
        false
    }

    fn _advance_one_frame(&mut self) -> bool {
        let board = &mut self.board;
        let frame = board.pop_frame().unwrap();

        board.time = board.time.bump();
        self.env.debug.as_mut().map(|x| x.record_frame(board, &frame));
        self._execute_effect_callbacks()
    }

    fn _arm_animation_fov(&mut self, pov: EID) {
        let Some(me) = self.board.get_entity(pov) else { return };
        self.env.fov.compute(&self.board, me);
    }

    fn _execute_effect_callbacks(&mut self) -> bool {
        let mut result = false;
        while self._execute_one_effect_callback() { result = true; }
        result
    }

    fn _execute_one_effect_callback(&mut self) -> bool {
        self.board.pop_callback().map(|x| { x(self); true }).unwrap_or(false)
    }
}

//////////////////////////////////////////////////////////////////////////////

#[allow(soft_unstable)]
#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;

    const BASE_SEED: u64 = 17;
    const NUM_SEEDS: u64 = 16;
    const NUM_STEPS: u64 = 1024;

    #[test]
    fn test_state_update() {
        let mut states = vec![];
        for i in 0..NUM_SEEDS {
            states.push(State::new(Some(BASE_SEED + i), GameMode::Test));
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
            states.push(State::new(Some(BASE_SEED + i), GameMode::Test));
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
