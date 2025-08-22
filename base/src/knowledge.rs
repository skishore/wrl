use std::cmp::max;
use std::num::NonZeroU64;

use rand::Rng;

use thin_vec::{ThinVec, thin_vec};

use crate::static_assert_size;
use crate::base::{Glyph, HashMap, Point, RNG, clamp};
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
    const MSEC_PER_SEC: i64 = 1_000;
    const NSEC_PER_SEC: i64 = 1_000_000_000;
    const NSEC_PER_MSEC: i64 = 1_000_000;

    pub const fn nsec(&self) -> i64 { self.0 }

    pub const fn from_nsec(nsec: i64) -> Self { Self(nsec) }

    pub const fn seconds(&self) -> f64 {
        let factor = Self::NSEC_PER_SEC as f64;
        (1. / factor) * self.nsec() as f64
    }

    pub const fn from_seconds(seconds: f64) -> Self {
        let factor = Self::NSEC_PER_SEC as f64;
        Self::from_nsec((factor * seconds) as i64)
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
        let nu = Timedelta::NSEC_PER_MSEC;
        let mu = Timedelta::MSEC_PER_SEC;
        let (left, nsec) = (self.nsec() / nu, self.nsec() % nu);
        let (left, msec) = (left / mu, left % mu);
        write!(fmt, "{}.{:0>3}s (+{})", left, msec, nsec)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Timestamp

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Timestamp(u64);

impl Timestamp {
    pub fn nsec(&self) -> u64 { self.0 }

    pub fn bump(&self) -> Self { self.latch(Timedelta::default()) }

    pub fn latch(&self, other: Timedelta) -> Self {
        let other = std::cmp::max(other, Timedelta(1));
        let value = self.0 + (other.0 as u64);
        let latch = value - (value % Timedelta::NSEC_PER_MSEC as u64);
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
        let nu = Timedelta::NSEC_PER_MSEC as u64;
        let mu = Timedelta::MSEC_PER_SEC as u64;
        let (left, nsec) = (self.nsec() / nu, self.nsec() % nu);
        let (left, msec) = (left / mu, left % mu);
        let (left, sec) = (left / 60, left % 60);
        let (left, min) = (left / 60, left % 60);
        let (left, hrs) = (left / 24, left % 24);
        write!(fmt, "{}d {:0>2}:{:0>2}:{:0>2}.{:0>3} (+{})",
               left, hrs, min, sec, msec, nsec)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Events

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct UID(NonZeroU64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sense { Sight, Sound, Smell }

#[derive(Clone, Debug)]
pub struct AttackEvent { pub target: Option<EID> }

#[derive(Clone, Debug, Default)]
pub struct ForgetEvent {}

#[derive(Clone, Debug)]
pub struct MoveEvent { pub from: Point }

#[derive(Clone, Debug, Default)]
pub struct SpotEvent {}

#[derive(Clone, Debug)]
pub enum EventData {
    Attack(AttackEvent),
    Forget(ForgetEvent),
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
// Scent

#[derive(Clone, Copy)]
pub struct Scent {
    pub pos: Point,
    pub time: Timestamp,
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

type CellHandle = Handle<CellKnowledge>;
type EntityHandle = Handle<EntityKnowledge>;
type SourceHandle = Handle<SourceKnowledge>;

// Detailed knowledge about a map cell. Only updated when we see it.
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

// Detailed knowledge about an entity. Only updated when we see it.
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

// Minimal knowledge about an entity that we're tracking by scent or sound.
// Used to tag events with a UID such that same UID => same event source.
struct SourceKnowledge {
    uid: UID,
    pos: Point,
    eid: Option<EID>,
    time: Timestamp,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(SourceKnowledge, 32);

#[derive(Default, Eq, PartialEq)]
struct EIDState {
    entity: Option<EntityHandle>,
    source: Option<SourceHandle>,
}

#[derive(Default, Eq, PartialEq)]
struct PointState {
    cell: Option<CellHandle>,
    entity: Option<EntityHandle>,
}

#[derive(Default)]
pub struct Knowledge {
    pub cells: List<CellKnowledge>,
    pub entities: List<EntityKnowledge>,
    pub events: Vec<Event>,
    pub scents: Vec<Scent>,
    pub time: Timestamp,

    eid_index: HashMap<EID, EIDState>,
    pos_index: HashMap<Point, PointState>,
    sources: List<SourceKnowledge>,
    last_uid: u64,
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
    fn new(uid: UID, event: &Event) -> Self {
        Self { uid, pos: event.point, eid: None, time: event.time }
    }
}

impl Knowledge {
    // Reads

    pub fn default(&self) -> PointLookup<'_> {
        PointLookup { root: self, spot: None }
    }

    pub fn entity(&self, eid: EID) -> Option<&EntityKnowledge> {
        Some(&self.entities[self.eid_index.get(&eid)?.entity?])
    }

    pub fn get(&self, p: Point) -> PointLookup<'_> {
        PointLookup { root: self, spot: self.pos_index.get(&p) }
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
            let entry = self.pos_index.entry(point).or_default();
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

        debug_assert!(self.check_invariants());
    }

    pub fn remove_entity(&mut self, eid: EID, time: Timestamp) {
        let Some(x) = self.eid_index.get_mut(&eid) else { return };
        let Some(h) = x.entity.take() else { return };

        if *x == Default::default() { self.eid_index.remove(&eid); }
        let EntityKnowledge { eid, pos, .. } = self.entities.remove(h);
        Self::move_from(&mut self.pos_index, pos, h);

        self.forget_event(Some(eid), None, pos, time);

        debug_assert!(self.check_invariants());
    }

    // Events helpers:

    pub fn observe_event(&mut self, me: &Entity, event: &Event) {
        if me.player { return; }

        let mut clone = event.clone();
        clone.eid = None;
        clone.uid = None;

        let Some(eid) = event.eid else {
            let uid = Self::get_next_uid(&mut self.last_uid);
            self.sources.push_front(SourceKnowledge::new(uid, event));
            clone.uid = Some(uid);
            self.events.push(clone);
            return;
        };

        let entry = self.eid_index.entry(eid).or_default();

        // Check if we can link this event to an existing source.
        if let Some(x) = entry.source {
            let source = &mut self.sources[x];
            if event.time - source.time < SOURCE_TRACKING_LIMIT {
                clone.uid = Some(source.uid);
            } else {
                entry.source = None;
            }
        }

        // Check if we can link this event to an existing entity.
        let link_to_entity = |x: &EntityKnowledge| match event.sense {
            Sense::Sight => true,
            Sense::Smell => false,
            Sense::Sound => event.time - x.time < SOURCE_TRACKING_LIMIT,
        };
        if let Some(h) = entry.entity && link_to_entity(&self.entities[h]) {
            let entity = &mut self.entities[h];
            let (source, target) = (entity.pos, event.point);
            if source != target {
                Self::move_from(&mut self.pos_index, source, h);
            }
            if let Some(x) = self.pos_index.get_mut(&target) {
                x.entity = Some(h);
            }

            entity.pos = event.point;
            entity.sense = event.sense;
            entity.time = event.time;

            self.entities.move_to_front(h);

            if let Some(x) = entry.source.take() { self.sources.remove(x); }

            clone.eid = Some(eid);
            self.events.push(clone);
            return;
        }

        let handle = entry.source.unwrap_or_else(|| {
            let uid = Self::get_next_uid(&mut self.last_uid);
            self.sources.push_front(SourceKnowledge::new(uid, &event))
        });
        if entry.source.is_some() { self.sources.move_to_front(handle); }
        if entry.source.is_none() { entry.source = Some(handle); }

        let source = &mut self.sources[handle];
        source.eid = Some(eid);
        source.pos = event.point;
        source.time = event.time;

        clone.uid = Some(source.uid);
        self.events.push(clone);

        debug_assert!(self.check_invariants());
    }

    // Private helpers:

    fn forget(&mut self, player: bool) {
        if player {
            while let Some(x) = self.cells.back() && !x.visible {
                self.forget_last_cell();
            }
            return;
        }

        // We don't need to check visible, here; we can only see a bounded
        // number of cells per turn, much less than MAX_TILE_MEMORY.
        while self.cells.len() > MAX_TILE_MEMORY {
            self.forget_last_cell();
        }

        // We clean up entities up to the first visible one.
        while self.entities.len() > MAX_ENTITY_MEMORY {
            let entity = self.entities.back().unwrap();
            if entity.visible { break; }
            self.remove_entity(entity.eid, self.time);
        }

        // We clean up all sources, both by count and age.
        while self.sources.len() > MAX_SOURCE_MEMORY {
            self.forget_last_source();
        }
        while let Some(x) = self.sources.back() &&
              self.time - x.time >= SOURCE_TRACKING_LIMIT {
            self.forget_last_source();
        }
    }

    fn forget_event(&mut self, eid: Option<EID>, uid: Option<UID>,
                    point: Point, time: Timestamp) {
        let data = EventData::Forget(ForgetEvent::default());
        let event = Event { eid, uid, data, time, point, sense: Sense::Sight };
        self.events.push(event);
    }

    fn forget_last_cell(&mut self) {
        let Some(x) = self.cells.pop_back() else { return };
        let Some(y) = self.pos_index.get_mut(&x.point) else { return };

        y.cell = None;
        if *y == Default::default() { self.pos_index.remove(&x.point); }
    }

    fn forget_last_source(&mut self) {
        let Some(h) = self.sources.back_handle() else { return };
        let Some(x) = self.sources.pop_back() else { return };

        if let Some(e) = x.eid { self.move_source(e, h); }

        self.forget_event(None, Some(x.uid), x.pos, self.time);
    }

    fn populate_scents(&mut self, me: &Entity, board: &Board, rng: &mut RNG) {
        if me.asleep { return; }

        let level = trophic_level(me);

        for (_, other) in &board.entities {
            if trophic_level(other) >= level { continue; }
            let mut remainder = rng.random::<f64>();

            for age in 0..other.trail.capacity() {
                remainder -= other.get_historical_scent_at(me.pos, age);
                if remainder >= 0. { continue; }
                self.scents.push(other.trail[age]);
                break;
            }
        }
    }

    fn see_entity(&mut self, me: &Entity, other: &Entity) -> EntityHandle {
        let (sense, time) = (Sense::Sight, self.time);
        let cached = self.eid_index.entry(other.eid).or_default();

        // Seeing this entity may let us identify an unknown event source.
        if let Some(x) = cached.source.take() {
            let SourceKnowledge { uid, time: last, .. } = self.sources.remove(x);
            if time - last < SOURCE_TRACKING_LIMIT {
                let (eid, uid) = (Some(other.eid), Some(uid));
                let data = EventData::Spot(SpotEvent::default());
                let event = Event { eid, uid, data, time, point: other.pos, sense };
                self.events.push(event);
            }
        }

        // Create a new EntityKnowledge instance or mark an old one as fresh.
        let handle = cached.entity.unwrap_or_else(
            || self.entities.push_front(EntityKnowledge::new(other.eid)));
        if cached.entity.is_some() {
            self.entities.move_to_front(handle);
            let (s, t) = (self.entities[handle].pos, other.pos);
            if s != t { Self::move_from(&mut self.pos_index, s, handle); }
        } else {
            cached.entity = Some(handle);
        };
        let entry = &mut self.entities[handle];

        // Update our knowledge with the entity's latest state.
        entry.pos = other.pos;
        entry.dir = other.dir;
        entry.glyph = other.glyph;
        entry.sense = sense;
        entry.name = other.name();
        entry.time = time;

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

    fn get_next_uid(last_uid: &mut u64) -> UID {
        *last_uid += 1;
        UID((*last_uid).try_into().unwrap())
    }

    fn move_from(m: &mut HashMap<Point, PointState>, p: Point, h: EntityHandle) {
        let Some(x) = m.get_mut(&p) else { return };
        if x.entity != Some(h) { return };

        x.entity = None;
        if *x == Default::default() { m.remove(&p); }
    }

    fn move_source(&mut self, e: EID, h: SourceHandle) {
        let Some(x) = self.eid_index.get_mut(&e) else { return };
        if x.source != Some(h) { return };

        x.source = None;
        if *x == Default::default() { self.eid_index.remove(&e); }
    }

    // Debug helpers:

    pub fn debug_noise_sources(&self) -> Vec<Point> {
        self.sources.iter().map(|x| x.pos).collect()
    }

    fn check_invariants(&self) -> bool {
        let check_sorted = |xs: Vec<Timestamp>| {
            let mut last = Timestamp::default();
            xs.into_iter().rev().for_each(|x| { assert!(x >= last); last = x; });
        };

        // Check that all lists are sorted in time order:
        check_sorted(self.cells.iter().map(|x| x.last_seen).collect());
        check_sorted(self.entities.iter().map(|x| x.time).collect());
        check_sorted(self.sources.iter().map(|x| x.time).collect());
        check_sorted(self.events.iter().rev().map(|x| x.time).collect());

        // Check that every cell and entity is indexed:
        for x in &self.cells {
            let entry = self.pos_index.get(&x.point);
            assert!(entry.and_then(|x| x.cell).is_some());
        }
        for x in &self.entities {
            let entry = self.eid_index.get(&x.eid);
            assert!(entry.and_then(|x| x.entity).is_some());
        }

        // Check that the indices are consistent and minimal:
        for (&pos, entry) in &self.pos_index {
            assert!(entry.cell.is_some() || entry.entity.is_some());
            if let Some(x) = entry.cell { assert!(self.cells[x].point == pos); }
            if let Some(x) = entry.entity { assert!(self.entities[x].pos == pos); }
        }
        for (&eid, entry) in &self.eid_index {
            assert!(entry.entity.is_some() || entry.source.is_some());
            if let Some(x) = entry.entity { assert!(self.entities[x].eid == eid); }
            if let Some(x) = entry.source { assert!(self.sources[x].eid == Some(eid)); }
        }
        true
    }
}

//////////////////////////////////////////////////////////////////////////////

// Result of querying knowledge about a cell

pub struct PointLookup<'a> {
    root: &'a Knowledge,
    spot: Option<&'a PointState>,
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
