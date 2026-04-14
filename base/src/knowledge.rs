use std::cmp::max;
use std::collections::VecDeque;
use std::num::NonZeroU64;
use std::rc::Rc;

use rand::Rng;

use thin_vec::ThinVec;

use crate::static_assert_size;
use crate::base::{HashMap, HashSet, Point, RNG, clamp};
use crate::dex::Species;
use crate::entity::{EID, Entity, Teammate};
use crate::game::{MOVE_TIMER, Board, Item, Light, Tile};
use crate::list::{Handle, List};
use crate::pathing::Status;
use crate::shadowcast::Vision;

//////////////////////////////////////////////////////////////////////////////

// Constants

const MAX_ENTITY_MEMORY: usize = 64;
const MAX_SOURCE_MEMORY: usize = 64;
const MAX_TILE_MEMORY: usize = 4096;
const MAX_TURN_MEMORY: usize = 256;

const SOURCE_LIMIT_PC_: i32 = 2;
const SOURCE_LIMIT_NPC: i32 = 72;
const SOURCE_TRACKING_LIMIT: i32 = 16;

const SCENT_TRACKING_LIMIT: i32 = 64;

fn trophic_level(x: &Entity) -> i32 {
    if x.species.human() { 0 } else if !x.species.predator() { 1 } else { 2 }
}

//////////////////////////////////////////////////////////////////////////////

impl std::ops::Deref for Event {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl std::ops::Deref for EntityKnowledge {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl std::ops::Deref for SourceKnowledge {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl std::ops::Deref for ScentKnowledge {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
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
        let mu = Timedelta::MSEC_PER_SEC;
        let ms = self.nsec() / Timedelta::NSEC_PER_MSEC;
        write!(fmt, "{}.{:0>3}s", ms / mu, ms % mu)
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
        let other = max(other, Timedelta(1));
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

impl std::ops::Sub<Timedelta> for Timestamp {
    type Output = Timestamp;
    fn sub(self, other: Timedelta) -> Self::Output {
        Timestamp((self.0 as i64 - other.0 as i64) as u64)
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

// Turn timing

#[derive(Default)]
pub struct TurnTimer {
    pub time: Timestamp,
    pub turn_times: VecDeque<Timestamp>,
}

impl TurnTimer {
    // Reads:

    pub fn debug_time(&self, time: Timestamp) -> String {
        if time == Timestamp::default() { return "<never>".into(); }

        let turns = self.time_to_turn(time);
        let (age, limit) = (self.time - time, MAX_TURN_MEMORY);
        if turns > limit as f64 { return format!("{:?} - >{} turns ago", age, limit); }

        let count = turns.ceil() as i32;
        let suffix = if count == 1 { "turn" } else { "turns" };
        let prefix = if turns == count as f64 { "" } else { "<" };
        format!("{:?} - {}{} {} ago", age, prefix, count, suffix)
    }

    pub fn time_at_turn(&self, turn: i32) -> Timestamp {
        if turn <= 0 { return self.time; }
        self.turn_times.get((turn - 1) as usize).map(|&x| x).unwrap_or_default()
    }

    pub fn time_to_turn(&self, time: Timestamp) -> f64 {
        let (mut prev, mut next) = (self.time, self.time);
        if time >= next { return 0.; }

        for (i, &n) in self.turn_times.iter().enumerate() {
            (prev, next) = (next, n);
            if time < next { continue; }

            let base = (i + 1) as f64;
            if time == next { return base; }

            let (a, b) = (prev - time, prev - next);
            return base - 1. + a.nsec() as f64 / max(b.nsec(), 1) as f64;
        }

        let base = self.turn_times.len() as f64;
        let (a, b) = (next - time, prev - next);
        base + a.nsec() as f64 / max(b.nsec(), 1) as f64
    }

    // Writes:

    fn end_turn(&mut self, speed: f64, time: Timestamp) {
        if self.turn_times.len() < MAX_TURN_MEMORY {
            let min = 1e-2;
            let speed = if speed < min { min } else { speed };
            let seconds_per_turn = 1. / speed;

            self.turn_times.reserve_exact(MAX_TURN_MEMORY);
            for i in 1..=MAX_TURN_MEMORY {
                let age = Timedelta::from_seconds(i as f64 * seconds_per_turn);
                self.turn_times.push_back(time - age);
            }
        }

        assert!(self.turn_times.len() == MAX_TURN_MEMORY);
        assert!(self.turn_times.capacity() == MAX_TURN_MEMORY);
        assert!(self.turn_times[0] < time);
        self.turn_times.pop_back();
        self.turn_times.push_front(time);
    }

    fn update(&mut self, time: Timestamp) {
        assert!(time >= self.time);
        self.time = time;
    }
}

//////////////////////////////////////////////////////////////////////////////

// Events

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct UID(NonZeroU64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sense { Sight, Sound, Smell }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Call { Command, Help, Warning }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sound { Attack, Call(Call), Move, Sniff }

#[derive(Clone, Debug)]
pub struct AttackEvent { pub combat: bool, pub target: Option<EID> }

#[derive(Clone, Debug)]
pub struct CallEvent { pub call: Call, pub species: &'static Species }

#[derive(Clone, Debug)]
pub struct MoveEvent { pub from: Point }

#[derive(Clone, Debug)]
pub enum EventData {
    Attack(AttackEvent),
    Call(CallEvent),
    Move(MoveEvent),
    Forget,
    Sniff,
    Spot,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub eid: Option<EID>,
    pub uid: Option<UID>,
    pub loc: Location,
    pub data: EventData,
    pub sense: Sense,
}

impl Event {
    fn sound(&self) -> Option<Sound> {
        if self.sense != Sense::Sound { return None; }
        match &self.data {
            EventData::Attack(_) => Some(Sound::Attack),
            EventData::Call(x)   => Some(Sound::Call(x.call)),
            EventData::Move(_)   => Some(Sound::Move),
            EventData::Sniff     => Some(Sound::Sniff),
            EventData::Forget    => None,
            EventData::Spot      => None,
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Scent

#[derive(Clone, Copy, Debug, Default)]
pub struct Location {
    pub pos: Point,
    pub time: Timestamp,
}

#[derive(Clone, Copy)]
pub struct ScentKnowledge {
    pub delta: i32,
    pub loc: Location,
    pub species: &'static Species,
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

type CellHandle   = Handle<CellKnowledge>;
type EntityHandle = Handle<EntityKnowledge>;
type SourceHandle = Handle<SourceKnowledge>;

type PointIndex = HashMap<Point, PointState>;

#[derive(Clone, Copy, Eq, PartialEq)]
enum OccupantHandle { Entity(EntityHandle), Source(SourceHandle) }

// Detailed knowledge about a map cell. Only updated when we see it.
#[derive(Default)]
pub struct CellKnowledge {
    pub last_see_entity_at: Timestamp,
    pub last_seen: Timestamp,
    pub items: ThinVec<Item>,
    pub point: Point,
    pub tile: &'static Tile,
    pub visibility: i32,

    // Flags:
    pub light: bool,
    pub shade: bool,
    pub visible: bool,
    pub see_entity_at: bool,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(CellKnowledge, 48);

// Detailed knowledge about an entity. Only updated when we see it.
pub struct EntityKnowledge {
    pub eid: EID,
    pub dir: Point,
    pub loc: Location,
    pub name: Option<Rc<str>>,
    pub sense: Sense,
    pub species: &'static Species,

    // Stats:
    pub hp: f64,
    pub pp: f64,
    pub delta: i32,

    // Team:
    pub team: ThinVec<bool>,

    // Flags:
    pub asleep: bool,
    pub friend: bool,
    pub rival: bool,
    pub sensed: bool,
    pub sneaking: bool,
    pub visible: bool,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(EntityKnowledge, 96);

// Minimal knowledge about entities that we're only tracking indirectly.
// We may have heard or glimpsed the entity, but we did not see it clearly.
//
// Used to tag events with a UID such that same UID => same event source.
pub struct SourceKnowledge {
    pub uid: UID,
    pub eid: Option<EID>,
    pub loc: Location,
    pub sound: Option<Sound>,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(SourceKnowledge, 40);

#[derive(Default, Eq, PartialEq)]
struct EIDState {
    entity: Option<EntityHandle>,
    source: Option<SourceHandle>,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(EIDState, 8);

#[derive(Default, Eq, PartialEq)]
struct PointState {
    cell: Option<CellHandle>,
    occupant: Option<OccupantHandle>,
}
#[cfg(target_pointer_width = "64")]
static_assert_size!(PointState, 12);

#[derive(Default)]
pub struct Knowledge {
    pub cells: List<CellKnowledge>,
    pub entities: List<EntityKnowledge>,
    pub sources: List<SourceKnowledge>,
    pub scents: Vec<ScentKnowledge>,
    pub events: Vec<Event>,

    timer: TurnTimer,
    eid_index: HashMap<EID, EIDState>,
    pos_index: HashMap<Point, PointState>,
    last_uid: u64,
}

impl EntityKnowledge {
    fn new(eid: EID, species: &'static Species) -> Self {
        Self {
            eid,
            dir: Default::default(),
            loc: Default::default(),
            name: Default::default(),
            sense: Sense::Sight,
            species,

            // Stats:
            hp: Default::default(),
            pp: Default::default(),
            delta: Default::default(),

            // Team:
            team: Default::default(),

            // Flags:
            asleep: false,
            friend: false,
            rival: false,
            sensed: false,
            sneaking: false,
            visible: false,
        }
    }

    pub fn too_big_to_hide(&self) -> bool {
        self.species.human() && !self.sneaking
    }
}

impl SourceKnowledge {
    fn new(uid: UID, event: &Event) -> Self {
        Self { uid, eid: None, loc: event.loc, sound: None }
    }

    pub fn freshness(&self, known: &Knowledge) -> f64 {
        let limit = max(SOURCE_LIMIT_PC_ - 1, 1);
        let turns = known.time_to_turn(self.time).floor() as i32;
        1. - clamp(turns, 0, limit) as f64 / limit as f64
    }
}

impl Knowledge {
    // Reads:

    pub fn debug_time(&self, time: Timestamp) -> String {
        self.timer.debug_time(time)
    }

    pub fn default(&self) -> PointLookup<'_> {
        PointLookup { root: self, spot: None }
    }

    pub fn entity(&self, eid: EID) -> Option<&EntityKnowledge> {
        Some(&self.entities[self.eid_index.get(&eid)?.entity?])
    }

    pub fn get(&self, p: Point) -> PointLookup<'_> {
        PointLookup { root: self, spot: self.pos_index.get(&p) }
    }

    pub fn time(&self) -> Timestamp {
        self.timer.time
    }

    pub fn time_at_turn(&self, turn: i32) -> Timestamp {
        self.timer.time_at_turn(turn)
    }

    pub fn time_to_turn(&self, time: Timestamp) -> f64 {
        self.timer.time_to_turn(time)
    }

    // Writes:

    pub fn mark_turn_boundary(&mut self, player: bool, speed: f64, time: Timestamp) {
        self.timer.end_turn(speed, time);

        self.forget_old_scents();
        self.forget_old_sources(player);

        debug_assert!(self.check_invariants());
    }

    pub fn observe_event(&mut self, me: &Entity, event: &Event) {
        self.timer.update(event.time);

        let mut clone = event.clone();
        clone.eid = None;
        clone.uid = None;

        let Some(eid) = event.eid else {
            let uid = Self::get_next_uid(&mut self.last_uid);
            let handle = self.sources.push_front(SourceKnowledge::new(uid, event));
            let occupant = OccupantHandle::Source(handle);
            Self::update_pos(&mut self.pos_index, occupant, None, Some(event.pos));
            clone.uid = Some(uid);
            self.events.push(clone);
            return;
        };

        let limit = self.time_at_turn(SOURCE_TRACKING_LIMIT);
        let entry = self.eid_index.entry(eid).or_default();

        // Check if we can link this event to an existing source.
        if let Some(x) = entry.source {
            let source = &mut self.sources[x];
            if source.time > limit {
                clone.uid = Some(source.uid);
            } else {
                entry.source = None;
            }
        }

        // Check if we can link this event to an existing entity.
        let link_to_entity = |x: &EntityKnowledge| match event.sense {
            Sense::Sight => true,
            Sense::Smell => false,
            Sense::Sound => x.time > limit,
        };
        if !me.player && let Some(h) = entry.entity && link_to_entity(&self.entities[h]) {
            let entity = &mut self.entities[h];
            let (s, t) = (entity.pos, event.pos);
            let occupant = OccupantHandle::Entity(h);
            Self::update_pos(&mut self.pos_index, occupant, Some(s), Some(t));

            if let Some(x) = entry.source.take() {
                let occupant = OccupantHandle::Source(x);
                let SourceKnowledge { loc, .. } = self.sources.remove(x);
                Self::update_pos(&mut self.pos_index, occupant, Some(loc.pos), None);
            }

            entity.loc = event.loc;
            entity.sense = event.sense;

            self.entities.move_to_front(h);

            clone.eid = Some(eid);
            self.events.push(clone);
            return;
        }

        let existing = entry.source;
        let handle = existing.unwrap_or_else(|| {
            let uid = Self::get_next_uid(&mut self.last_uid);
            self.sources.push_front(SourceKnowledge::new(uid, &event))
        });
        if entry.source.is_some() { self.sources.move_to_front(handle); }
        if entry.source.is_none() { entry.source = Some(handle); }

        let source = &mut self.sources[handle];
        let occupant = OccupantHandle::Source(handle);
        let (s, t) = (existing.map(|_| source.pos), event.pos);
        Self::update_pos(&mut self.pos_index, occupant, s, Some(t));

        source.eid = Some(eid);
        source.loc = event.loc;
        source.sound = event.sound();

        clone.uid = Some(source.uid);
        self.events.push(clone);

        debug_assert!(self.check_invariants());
    }

    pub fn remove_entity(&mut self, eid: EID, time: Timestamp) {
        self.timer.update(time);

        let Some(x) = self.eid_index.get_mut(&eid) else { return };
        let Some(h) = x.entity.take() else { return };

        if *x == Default::default() { self.eid_index.remove(&eid); }

        let occupant = OccupantHandle::Entity(h);
        let EntityKnowledge { eid, loc, .. } = self.entities.remove(h);
        Self::update_pos(&mut self.pos_index, occupant, Some(loc.pos), None);

        self.forget_event(Some(eid), None, loc.pos);

        debug_assert!(self.check_invariants());
    }

    pub fn update(&mut self, me: &Entity, board: &Board, vision: &Vision, rng: &mut RNG) {
        self.timer.update(board.time);

        let (pos, time) = (me.pos, board.time);
        let unlit = matches!(board.get_light(), Light::None);

        // Detect entities that were recently nearby by scent.
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
            entity.sensed = false;
            entity.visible = false;
        }

        // Entities know where their allies are, even if they can't see them.
        for &oid in me.leader.iter().chain(&me.summons) {
            let other = &board.entities[oid];
            let handle = self.update_entity(me, other, Sense::Sound);
            let cached = self.pos_index.entry(other.pos).or_default();
            cached.occupant = Some(OccupantHandle::Entity(handle));
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

            let light = board.is_cell_lit(point);
            let nearby = (point - pos).len_l1() <= 1;
            if unlit && !light && !nearby { continue; }

            let visible = true;
            let shade = unlit || cell.shadow > 0;
            let is_shadow_cover = shade && !light;
            let see_big_entities = nearby || !is_shadow_cover;
            let see_all_entities = nearby || !(is_shadow_cover || tile.is_cover());

            let entity = (|| {
                if !see_big_entities { return None; }
                let other = board.get_entity(eid?)?;
                if !see_all_entities && !other.too_big_to_hide() { return None; }
                Some(self.update_entity(me, other, Sense::Sight))
            })();

            // Update this point's cell, or create a new one.
            let entry = self.pos_index.entry(point).or_default();
            let cell = entry.cell.unwrap_or_else(
                || self.cells.push_front(Default::default()));
            if entry.cell.is_some() { self.cells.move_to_front(cell); }
            if entry.cell.is_none() { entry.cell = Some(cell); }

            // Update basic information about the given cell.
            let cell = &mut self.cells[cell];
            cell.last_seen = time;
            cell.point = point;
            cell.tile = tile;
            cell.visibility = vision.get_visibility_at(point);

            // Update the cell's flags.
            cell.light = light;
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
                entry.occupant = entity.map(|x| OccupantHandle::Entity(x));
            }
        }

        self.forget(me.player);

        debug_assert!(self.check_invariants());
    }

    // Private helpers:

    fn forget(&mut self, player: bool) {
        if player {
            while let Some(x) = self.cells.back() && !x.visible {
                self.forget_last_cell();
            }
            self.events.clear();
            return;
        }

        // We don't need to check visible, here; we can only see a bounded
        // number of cells per turn, much less than MAX_TILE_MEMORY.
        while self.cells.len() > MAX_TILE_MEMORY {
            self.forget_last_cell();
        }

        // We clean up entities up to the first one that we can still sense.
        while self.entities.len() > MAX_ENTITY_MEMORY {
            let entity = self.entities.back().unwrap();
            if entity.sensed { break; }
            self.remove_entity(entity.eid, self.timer.time);
        }

        // Clean up sources by count. On turn boundaries, we drop them by age.
        while self.sources.len() > MAX_SOURCE_MEMORY {
            self.forget_last_source();
        }
    }

    fn forget_event(&mut self, eid: Option<EID>, uid: Option<UID>, pos: Point) {
        let loc = Location { pos, time: self.timer.time };
        let event = Event { eid, uid, loc, data: EventData::Forget, sense: Sense::Sight };
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

        if let Some(e) = x.eid { self.unlink_source(e, h); }

        let occupant = OccupantHandle::Source(h);
        Self::update_pos(&mut self.pos_index, occupant, Some(x.pos), None);

        self.forget_event(None, Some(x.uid), x.pos);
    }

    fn forget_old_scents(&mut self) {
        let limit = self.time_at_turn(SCENT_TRACKING_LIMIT);

        while let Some(x) = self.scents.last() && x.time <= limit {
            self.scents.pop();
        }
    }

    fn forget_old_sources(&mut self, player: bool) {
        let turns = if player { SOURCE_LIMIT_PC_ } else { SOURCE_LIMIT_NPC };
        let limit = self.time_at_turn(turns);

        while let Some(x) = self.sources.back() && x.time <= limit {
            self.forget_last_source();
        }
    }

    fn populate_scents(&mut self, me: &Entity, board: &Board, rng: &mut RNG) {
        if me.asleep || me.player { return; }

        let initial = self.scents.len();

        for (_, other) in &board.entities {
            if other.eid == me.eid { continue; }

            let mut remainder = rng.random::<f64>();

            for (&loc, scent) in other.get_scent_trail(me.pos) {
                remainder -= scent * me.species.scent;
                if remainder >= 0. { continue; }

                let species = other.species;
                let delta = trophic_level(other) - trophic_level(me);
                self.scents.push(ScentKnowledge { delta, loc, species });
                break;
            }
        }

        if self.scents.len() == initial { return; }

        // Keep scents sorted by time and de-duplicated by species.
        let mut seen = HashSet::default();
        self.scents.sort_by_key(|x| self.timer.time - x.time);
        self.scents.retain(|x| seen.insert(x.species as *const Species));
    }

    fn update_entity(&mut self, me: &Entity, other: &Entity, sense: Sense) -> EntityHandle {
        let time = self.timer.time;
        let limit = self.time_at_turn(SOURCE_TRACKING_LIMIT);
        let cached = self.eid_index.entry(other.eid).or_default();

        // Seeing this entity may let us identify an unknown event source.
        if let Some(x) = cached.source.take() && self.sources[x].time > limit {
            let occupant = OccupantHandle::Source(x);
            let SourceKnowledge { uid, loc, .. } = self.sources.remove(x);
            Self::update_pos(&mut self.pos_index, occupant, Some(loc.pos), None);

            let loc = Location { pos: other.pos, time };
            let (eid, uid) = (Some(other.eid), Some(uid));
            let event = Event { eid, uid, loc, data: EventData::Spot, sense };
            self.events.push(event);
        }

        // Create a new EntityKnowledge instance or mark an old one as fresh.
        let handle = cached.entity.unwrap_or_else(
            || self.entities.push_front(EntityKnowledge::new(other.eid, other.species)));
        if cached.entity.is_some() {
            self.entities.move_to_front(handle);
            let occupant = OccupantHandle::Entity(handle);
            let (s, t) = (self.entities[handle].pos, other.pos);
            Self::update_pos(&mut self.pos_index, occupant, Some(s), Some(t));
        } else {
            cached.entity = Some(handle);
        };
        let entry = &mut self.entities[handle];

        // Entities are friends iff they're both tame and in the same party.
        let leader = |entity: &Entity| { entity.leader.unwrap_or(entity.eid) };

        // Entities are rivals iff one is tame and one is wild, or they're
        // both tame but as part of different parties.
        let trainer = |entity: &Entity| {
            if let Some(x) = entity.leader { return Some(x) };
            if entity.species.human() { return Some(entity.eid); }
            None
        };

        // Only a few species can lie about their identity. Update it anyway.
        entry.species = other.species;

        // Update our knowledge with the entity's latest state.
        entry.loc = Location { pos: other.pos, time };
        entry.dir = other.dir;
        entry.name = other.name.clone();
        entry.sense = sense;

        entry.hp = other.hp_fraction();
        entry.pp = 1. - clamp(other.move_timer as f64 / MOVE_TIMER as f64, 0., 1.);
        entry.delta = trophic_level(other) - trophic_level(me);

        // Clone the team, but reuse the existing allocation, if any.
        entry.team.clear();
        other.team.iter().for_each(|x| match x {
            Teammate::Out(_) => entry.team.push(true),
            Teammate::In(x) => entry.team.push(x.cur_hp > 0),
        });

        entry.asleep = other.asleep;
        entry.friend = leader(me) == leader(other);
        entry.rival = trainer(me) != trainer(other);
        entry.sensed = true;
        entry.sneaking = other.sneaking;
        entry.visible = sense == Sense::Sight;

        handle
    }

    // These methods would take &mut self, but hit lifetime issues:

    fn get_next_uid(last_uid: &mut u64) -> UID {
        *last_uid += 1;
        UID((*last_uid).try_into().unwrap())
    }

    fn unlink_source(&mut self, e: EID, h: SourceHandle) {
        let Some(x) = self.eid_index.get_mut(&e) else { return };
        if x.source != Some(h) { return };

        x.source = None;
        if *x == Default::default() { self.eid_index.remove(&e); }
    }

    fn update_pos(m: &mut PointIndex, h: OccupantHandle, from: Option<Point>, to: Option<Point>) {
        if from == to { return; }

        if let Some(p) = from && let Some(x) = m.get_mut(&p) && x.occupant == Some(h) {
            x.occupant = None;
            if *x == Default::default() { m.remove(&p); }
        }

        if let Some(p) = to {
            m.entry(p).or_default().occupant = Some(h);
        }
    }

    // Debug helpers:

    fn check_invariants(&self) -> bool {
        let check_sorted = |xs: Vec<Timestamp>| {
            let mut last = Timestamp::default();
            xs.into_iter().rev().for_each(|x| { assert!(x >= last); last = x; });
        };

        // Check that all lists are sorted in time order:
        check_sorted(self.cells.iter().map(|x| x.last_seen).collect());
        check_sorted(self.entities.iter().map(|x| x.time).collect());
        check_sorted(self.sources.iter().map(|x| x.time).collect());
        check_sorted(self.scents.iter().map(|x| x.time).collect());
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
        type OH = OccupantHandle;
        for (&pos, &PointState { cell, occupant }) in &self.pos_index {
            assert!(cell.is_some() || occupant.is_some());
            if let Some(x) = cell { assert!(self.cells[x].point == pos); }
            if let Some(OH::Entity(x)) = occupant { assert!(self.entities[x].pos == pos); }
            if let Some(OH::Source(x)) = occupant { assert!(self.sources[x].pos == pos); }
        }
        for (&eid, EIDState { entity, source }) in &self.eid_index {
            assert!(entity.is_some() || source.is_some());
            if let Some(x) = *entity { assert!(self.entities[x].eid == eid); }
            if let Some(x) = *source { assert!(self.sources[x].eid == Some(eid)); }
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
        self.root.time() - time
    }

    pub fn time_since_entity_visible(&self) -> Timedelta {
        let time = self.cell().map(|x| x.last_see_entity_at).unwrap_or_default();
        self.root.time() - time
    }

    pub fn items(&self) -> &[Item] {
        self.cell().map(|x| x.items.as_slice()).unwrap_or(&[])
    }

    pub fn light(&self) -> bool {
        self.cell().map(|x| x.light).unwrap_or(false)
    }

    pub fn shade(&self) -> bool {
        self.cell().map(|x| x.shade).unwrap_or(false)
    }

    pub fn tile(&self) -> Option<&'static Tile> {
        self.cell().map(|x| x.tile)
    }

    pub fn visibility(&self) -> i32 {
        self.cell().map(|x| x.visibility).unwrap_or(-1)
    }

    // Derived fields

    pub fn cell(&self) -> Option<&'a CellKnowledge> {
        Some(&self.root.cells[self.spot?.cell?])
    }

    pub fn entity(&self) -> Option<&'a EntityKnowledge> {
        let OccupantHandle::Entity(x) = self.spot?.occupant? else { return None };
        Some(&self.root.entities[x])
    }

    pub fn source(&self) -> Option<&'a SourceKnowledge> {
        let OccupantHandle::Source(x) = self.spot?.occupant? else { return None };
        Some(&self.root.sources[x])
    }

    pub fn status(&self) -> Status {
        let Some(spot) = self.spot else { return Status::Unknown };
        let Some(cell) = spot.cell else { return Status::Unknown };

        let tile = self.root.cells[cell].tile;
        if tile.blocks_movement() { return Status::Blocked; }
        if spot.occupant.is_some() { Status::Occupied } else { Status::Free }
    }

    // Predicates

    pub fn occupied(&self) -> bool {
        self.spot.map(|x| x.occupant.is_some()).unwrap_or(false)
    }

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

    pub fn is_shadow_cover(&self) -> bool {
        self.cell().map(|x| x.shade && !x.light).unwrap_or(false)
    }
}
