use std::cmp::max;
use std::num::NonZeroU32;

use rand::Rng;

use thin_vec::{ThinVec, thin_vec};

use crate::static_assert_size;
use crate::base::{Glyph, HashMap, HashSet, Point, RNG, clamp};
use crate::entity::{EID, Entity};
use crate::game::{MOVE_TIMER, Board, Item, Light, Tile};
use crate::list::{Handle, List};
use crate::pathing::Status;
use crate::shadowcast::Vision;

//////////////////////////////////////////////////////////////////////////////

// Constants

const MAX_ENTITY_MEMORY: usize = 64;
const MAX_SOURCE_MEMORY: usize = 64;
const MAX_TILE_MEMORY: usize = 4096;

const SOURCE_TRACKING_LIMIT: Timedelta = Timedelta::from_seconds(24.);

fn trophic_level(x: &Entity) -> i32 {
    if x.player { 3 } else if !x.predator { 1 } else { 2 }
}

//////////////////////////////////////////////////////////////////////////////

// Timedelta

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Timedelta(i64);

impl Timedelta {
    const LATCH: i64 = 1 << 20;
    const SCALE: i64 = 1000;

    pub fn raw(self) -> i64 {
        self.0
    }

    pub const fn to_seconds(self) -> f64 {
        let factor = (Self::LATCH * Self::SCALE) as f64;
        (1. / factor) * self.0 as f64
    }

    pub const fn from_seconds(seconds: f64) -> Self {
        let factor = (Self::LATCH * Self::SCALE) as f64;
        Self((factor * seconds) as i64)
    }
}

impl std::ops::Add<Timedelta> for Timedelta {
    type Output = Timedelta;
    fn add(self, other: Timedelta) -> Self::Output {
        Self(self.0 + other.0)
    }
}

impl std::ops::Sub for Timedelta {
    type Output = Timedelta;
    fn sub(self, other: Timedelta) -> Self::Output {
        Self(self.0 - other.0)
    }
}

impl std::fmt::Debug for Timedelta {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let latch = Timedelta::LATCH;
        let scale = Timedelta::SCALE;
        let (left, tick) = (self.0 / latch, self.0 % latch);
        let (left, msec) = (left / scale, left % scale);
        write!(fmt, "{}.{:0>3}s (+{})", left, msec, tick)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Timestamp

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Timestamp(u64);

impl Timestamp {
    pub fn bump(self) -> Self {
        self.latch(Timedelta::default())
    }

    pub fn latch(self, other: Timedelta) -> Self {
        let other = std::cmp::max(other, Timedelta(1));
        let value = self.0 + (other.0 as u64);
        let latch = value & !(Timedelta::LATCH as u64 - 1);
        Self(if latch > self.0 { latch } else { value })
    }
}

impl std::ops::Sub for Timestamp {
    type Output = Timedelta;
    fn sub(self, other: Timestamp) -> Self::Output {
        Timedelta(self.0.wrapping_sub(other.0) as i64)
    }
}

impl std::fmt::Debug for Timestamp {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let latch = Timedelta::LATCH as u64;
        let scale = Timedelta::SCALE as u64;
        let (left, tick) = (self.0 / latch, self.0 % latch);
        let (left, msec) = (left / scale, left % scale);
        let (left, sec) = (left / 60, left % 60);
        let (left, min) = (left / 60, left % 60);
        let (left, hrs) = (left / 24, left % 24);
        write!(fmt, "{}d {:0>2}:{:0>2}:{:0>2}.{:0>3} (+{})",
               left, hrs, min, sec, msec, tick)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Events

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct UID(NonZeroU32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sense { Sight, Sound, Smell }

#[derive(Clone, Debug)]
pub struct AttackEvent { pub target: Option<EID> }

#[derive(Clone, Debug)]
pub struct MoveEvent { pub from: Point }

#[derive(Clone, Debug, Default)]
pub struct SpotEvent {}

#[derive(Clone, Debug)]
pub enum EventData {
    Attack(AttackEvent),
    Move(MoveEvent),
    Spot(SpotEvent),
}

#[derive(Clone, Debug)]
pub struct Event {
    pub eid: Option<EID>,
    pub uid: Option<UID>,
    pub data: EventData,
    pub time: Timestamp,
    pub point: Point,
    pub sense: Sense,
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

type CellHandle = Handle<CellKnowledge>;
type EntityHandle = Handle<EntityKnowledge>;
type SourceHandle = Handle<SourceKnowledge>;

pub struct CellKnowledge {
    pub last_see_entity_at: Timestamp,
    pub last_seen: Timestamp,
    pub items: ThinVec<Item>,
    pub point: Point,
    pub tile: &'static Tile,

    // Flags:
    pub shade: bool,
    pub visible: bool,
    pub see_entity_at: bool,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(CellKnowledge, 48);

pub struct EntityKnowledge {
    pub eid: EID,
    pub pos: Point,
    pub dir: Point,
    pub glyph: Glyph,
    pub sense: Sense,
    pub name: &'static str,
    pub time: Timestamp,

    // Stats:
    pub hp: f64,
    pub pp: f64,
    pub delta: i32,

    // Flags:
    pub asleep: bool,
    pub friend: bool,
    pub player: bool,
    pub sneaking: bool,
    pub visible: bool,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(EntityKnowledge, 88);

pub struct SourceKnowledge {
    eid: Option<EID>,
    pub uid: UID,
    pub pos: Point,
    pub sense: Sense,
    pub time: Timestamp,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(SourceKnowledge, 32);

#[derive(Default, Eq, PartialEq)]
pub struct PointKnowledge {
    cell: Option<CellHandle>,
    entity: Option<EntityHandle>,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(PointKnowledge, 8);

#[derive(Clone, Copy)]
pub struct Scent {
    pub pos: Point,
    pub time: Timestamp,
}

#[derive(Default)]
pub struct Knowledge {
    point_map: HashMap<Point, PointKnowledge>,
    entity_map: HashMap<EID, EntityHandle>,
    source_map: HashMap<EID, SourceHandle>,
    pub cells: List<CellKnowledge>,
    pub entities: List<EntityKnowledge>,
    pub sources: List<SourceKnowledge>,
    pub events: Vec<Event>,
    pub scents: Vec<Scent>,
    pub time: Timestamp,
}

impl CellKnowledge {
    fn new(point: Point, tile: &'static Tile) -> Self {
        Self {
            last_seen: Timestamp::default(),
            last_see_entity_at: Timestamp::default(),
            items: thin_vec![],
            point,
            tile,

            // Flags:
            shade: false,
            visible: false,
            see_entity_at: false,
        }
    }
}

impl EntityKnowledge {
    fn new(eid: EID) -> Self {
        Self {
            eid,
            pos: Default::default(),
            dir: Default::default(),
            glyph: Default::default(),
            sense: Sense::Sight,
            name: Default::default(),
            time: Default::default(),

            // Stats:
            hp: Default::default(),
            pp: Default::default(),
            delta: Default::default(),

            // Flags:
            asleep: false,
            friend: false,
            player: false,
            sneaking: false,
            visible: false,
        }
    }
}

impl SourceKnowledge {
    fn new(uid: UID) -> Self {
        let (pos, time) = (Default::default(), Default::default());
        Self { eid: None, uid, pos, sense: Sense::Sight, time }
    }

    fn from(uid: UID, event: &Event) -> Self {
        let mut result = Self::new(uid);
        result.update(event);
        result
    }

    fn update(&mut self, event: &Event) {
        self.pos = event.point;
        self.sense = event.sense;
        self.time = event.time;
    }
}

impl PointKnowledge {
    fn empty(&self) -> bool {
        *self == PointKnowledge::default()
    }
}

impl Knowledge {
    // Reads

    pub fn default(&self) -> PointLookup {
        PointLookup { root: self, spot: None }
    }

    pub fn entity(&self, eid: EID) -> Option<&EntityKnowledge> {
        self.entity_map.get(&eid).map(|&x| &self.entities[x])
    }

    pub fn get(&self, p: Point) -> PointLookup {
        PointLookup { root: self, spot: self.point_map.get(&p) }
    }

    // Writes

    pub fn update(&mut self, me: &Entity, board: &Board, vision: &Vision, rng: &mut RNG) {
        self.time = board.time;
        let (pos, time) = (me.pos, self.time);
        let dark = matches!(board.get_light(), Light::None);

        // Clear and recompute scents. Only prey gives off a scent.
        self.scents.clear();
        self.populate_scents(me, board, rng);

        // Clear visibility flags. Visible cells come first in the list so we
        // can stop when we see the first one that's not visible.
        //
        // Note that there may be cells with cell.last_seen == time that are
        // not visible, because of sub-turn visibility updates.
        for cell in &mut self.cells {
            if !cell.visible { break; }
            cell.see_entity_at = false;
            cell.visible = false;
        }
        for entity in &mut self.entities {
            entity.visible = false;
        }

        // Entities have exact knowledge about anything they can see.
        //
        // We want self.cells to be sorted by recency, and if there are ties,
        // by distance. Closer and more recently seen points come first.
        //
        // Within the loop here, we repeatedly move cells to the front of
        // self.cells. Because points_seen is sorted by distance, we iterate
        // over it in reverse order to get the ordering above.
        for &point in vision.get_points_seen().iter().rev() {
            let cell = board.get_cell(point);
            let (eid, items, tile) = (cell.eid, &cell.items, cell.tile);

            let nearby = (point - pos).len_l1() <= 1;
            if dark && !nearby { continue; }

            let visible = true;
            let shade = dark || cell.shadow > 0;
            let see_big_entities = nearby || !shade;
            let see_all_entities = nearby || !(shade || tile.limits_vision());

            let entity = (|| {
                if !see_big_entities { return None; }
                let other = board.get_entity(eid?)?;
                let big = other.player && !other.sneaking;
                if !big && !see_all_entities { return None; }
                Some(self.see_entity(me, other))
            })();

            // Update this point's cell, or create a new one.
            let entry = self.point_map.entry(point).or_default();
            let cell = entry.cell.unwrap_or_else(
                || self.cells.push_front(CellKnowledge::new(point, tile)));
            if entry.cell.is_some() { self.cells.move_to_front(cell); }
            if entry.cell.is_none() { entry.cell = Some(cell); }

            // Update basic information about the given cell.
            let cell = &mut self.cells[cell];
            cell.last_seen = time;
            cell.point = point;
            cell.tile = tile;

            // Update the cell's flags.
            cell.shade = shade;
            cell.visible = visible;
            cell.see_entity_at = see_all_entities;

            // Clone items, but reuse the existing allocation, if any.
            cell.items.clear();
            for &x in items { cell.items.push(x); }

            // Clear the cell's entity and event if we have full vision there.
            if see_all_entities {
                cell.last_see_entity_at = time;
            }
            if see_all_entities || entity.is_some() {
                entry.entity = entity;
            }
        }

        self.forget(me.player);
    }

    pub fn remove_entity(&mut self, eid: EID) {
        let Some(x) = self.entity_map.remove(&eid) else { return };
        let pos = self.entities.remove(x).pos;
        Self::move_from(&mut self.point_map, pos, x);
    }

    // Events helpers:

    pub fn forget_events_before(&mut self, time: Timestamp) {
        self.events.retain(|x| x.time > time);
    }

    pub fn observe_event(&mut self, me: &Entity, event: &Event) {
        if me.player { return; }

        let mut clone = event.clone();
        clone.eid = None;
        clone.uid = None;

        let Some(eid) = event.eid else {
            let uid = Self::get_next_uid(&self.sources);
            self.sources.push_front(SourceKnowledge::from(uid, event));
            clone.uid = Some(uid);
            self.events.push(clone);
            return;
        };

        let seen = match event.sense {
            Sense::Sight => true,
            Sense::Smell => false,
            Sense::Sound => match self.entity_map.get(&eid) {
                Some(&x) => event.time - self.entities[x].time < SOURCE_TRACKING_LIMIT,
                None => false,
            },
        };

        if seen && let Some(&handle) = self.entity_map.get(&eid) {
            self.entities.move_to_front(handle);

            let entity = &mut self.entities[handle];

            let (source, target) = (entity.pos, event.point);
            if source != target {
                Self::move_from(&mut self.point_map, source, handle);
            }
            if let Some(x) = self.point_map.get_mut(&target) {
                x.entity = Some(handle);
            }

            entity.pos = event.point;
            entity.sense = event.sense;
            entity.time = event.time;

            if let Some(x) = self.source_map.remove(&eid) {
                let source = &mut self.sources[x];
                if event.time - source.time >= SOURCE_TRACKING_LIMIT {
                    source.eid = None;
                } else {
                    clone.uid = Some(source.uid);
                    self.sources.remove(x);
                }
            }

            clone.eid = Some(eid);
            self.events.push(clone);
            return;
        }

        let create = |sources: &mut List<SourceKnowledge>| {
            let uid = Self::get_next_uid(sources);
            sources.push_front(SourceKnowledge::new(uid))
        };
        let handle = *self.source_map.entry(eid).and_modify(|x| {
            let source = &mut self.sources[*x];
            if event.time - source.time >= SOURCE_TRACKING_LIMIT {
                source.eid = None;
                *x = create(&mut self.sources);
            } else {
                self.sources.move_to_front(*x);
            }
        }).or_insert_with(|| create(&mut self.sources));

        let entry = &mut self.sources[handle];
        entry.eid = Some(eid);
        entry.update(event);

        clone.uid = Some(entry.uid);
        self.events.push(clone);
    }

    // Private helpers:

    fn forget(&mut self, player: bool) {
        if player {
            while let Some(x) = self.cells.back() && !x.visible {
                self.forget_last_cell();
            }
            return;
        }

        while self.point_map.len() > MAX_TILE_MEMORY && !self.cells.is_empty() {
            // We don't need to check age, here; we can only see a bounded
            // number of cells per turn, much less than MAX_TILE_MEMORY.
            self.forget_last_cell();
        }

        while self.entity_map.len() > MAX_ENTITY_MEMORY {
            let entity = self.entities.back().unwrap();
            if entity.visible { break; }

            // We clean up entities up to the first visible one.
            self.remove_entity(entity.eid);
        }

        while self.source_map.len() > MAX_SOURCE_MEMORY {
            let Some(source) = self.sources.pop_back() else { break };
            if let Some(eid) = source.eid { self.source_map.remove(&eid); }
        }
    }

    fn forget_last_cell(&mut self) {
        let CellKnowledge { point, .. } = self.cells.pop_back().unwrap();
        let Some(x) = self.point_map.get_mut(&point) else { return };
        x.cell = None;
        if x.empty() { self.point_map.remove(&point); }
    }

    fn populate_scents(&mut self, me: &Entity, board: &Board, rng: &mut RNG) {
        if me.asleep { return; }

        let level = trophic_level(me);

        for (_, other) in &board.entities {
            if trophic_level(other) >= level { continue; }
            let mut remainder = rng.gen::<f64>();
            for age in 0..other.trail.capacity() {
                remainder -= other.get_historical_scent_at(me.pos, age);
                if remainder >= 0. { continue; }
                self.scents.push(other.trail[age]);
                break;
            }
        }
    }

    fn see_entity(&mut self, me: &Entity, other: &Entity) -> EntityHandle {
        let handle = *self.entity_map.entry(other.eid).and_modify(|&mut x| {
            self.entities.move_to_front(x);
            let (source, target) = (self.entities[x].pos, other.pos);
            if source != target { Self::move_from(&mut self.point_map, source, x); }
        }).or_insert_with(|| {
            self.entities.push_front(EntityKnowledge::new(other.eid))
        });

        let entry = &mut self.entities[handle];

        entry.pos = other.pos;
        entry.dir = other.dir;
        entry.glyph = other.glyph;
        entry.sense = Sense::Sight;
        entry.name = other.name();
        entry.time = self.time;

        entry.hp = other.cur_hp as f64 / max(other.max_hp, 1) as f64;
        entry.pp = 1. - clamp(other.move_timer as f64 / MOVE_TIMER as f64, 0., 1.);
        entry.delta = trophic_level(other) - trophic_level(me);

        entry.asleep = other.asleep;
        entry.friend = me.eid == other.eid;
        entry.player = other.player;
        entry.sneaking = other.sneaking;
        entry.visible = true;

        handle
    }

    // These methods would take &mut self, but hit lifetime issues:

    fn get_next_uid(sources: &List<SourceKnowledge>) -> UID {
        let used: HashSet<_> = sources.iter().map(|x| x.uid).collect();
        let mut result = UID(1.try_into().unwrap());
        while used.contains(&result) { result.0 = result.0.checked_add(1).unwrap() }
        result
    }

    fn move_from(m: &mut HashMap<Point, PointKnowledge>, p: Point, h: EntityHandle) {
        let Some(x) = m.get_mut(&p) else { return };
        if x.entity == Some(h) { x.entity = None; }
        if x.empty() { m.remove(&p); }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Result of querying knowledge about a cell

pub struct PointLookup<'a> {
    root: &'a Knowledge,
    spot: Option<&'a PointKnowledge>,
}

impl<'a> PointLookup<'a> {
    // Field lookups

    pub fn time_since_seen(&self) -> Timedelta {
        let time = self.cell().map(|x| x.last_seen).unwrap_or_default();
        self.root.time - time
    }

    pub fn time_since_entity_visible(&self) -> Timedelta {
        let time = self.cell().map(|x| x.last_see_entity_at).unwrap_or_default();
        self.root.time - time
    }

    pub fn items(&self) -> &[Item] {
        self.cell().map(|x| x.items.as_slice()).unwrap_or(&[])
    }

    pub fn shade(&self) -> bool {
        self.cell().map(|x| x.shade).unwrap_or(false)
    }

    pub fn tile(&self) -> Option<&'static Tile> {
        self.cell().map(|x| x.tile)
    }

    // Derived fields

    pub fn cell(&self) -> Option<&CellKnowledge> {
        Some(&self.root.cells[self.spot?.cell?])
    }

    pub fn entity(&self) -> Option<&'a EntityKnowledge> {
        Some(&self.root.entities[self.spot?.entity?])
    }

    pub fn status(&self) -> Status {
        let Some(spot) = self.spot else { return Status::Unknown };
        let Some(cell) = spot.cell else { return Status::Unknown };
        let tile = self.root.cells[cell].tile;
        if tile.blocks_movement() { return Status::Blocked; }
        if spot.entity.is_some() { Status::Occupied } else { Status::Free }
    }

    // Predicates

    pub fn blocked(&self) -> bool {
        self.cell().map(|x| x.tile.blocks_movement()).unwrap_or(false)
    }

    pub fn unblocked(&self) -> bool {
        self.cell().map(|x| !x.tile.blocks_movement()).unwrap_or(false)
    }

    pub fn unknown(&self) -> bool {
        self.cell().is_none()
    }

    pub fn visible(&self) -> bool {
        self.cell().map(|x| x.visible).unwrap_or(false)
    }

    pub fn can_see_entity_at(&self) -> bool {
        self.cell().map(|x| x.see_entity_at).unwrap_or(false)
    }
}
