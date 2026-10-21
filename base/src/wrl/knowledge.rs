use std::cmp::max;
use std::rc::Rc;

use rand::Rng;
use thin_vec::ThinVec;

use crate::flags;
use crate::base::point::{Delta, Point, dirs};
use crate::base::util::{HashMap, HashSet, RNG, clamp};
use crate::base::pathing::Status;
use crate::base::vision::Vision;

use super::actor::{Actor, EID, Teammate};
use super::dex::Species;
use super::event::{Event, EventData, Location, Sense, Sound, UID};
use super::game::{MOVE_TIMER, Board, Cell, Item, Light, Tile, TileFlags};
use super::list::{Handle, List};
use super::time::{Timestamp, TurnTimer};

//////////////////////////////////////////////////////////////////////////////

// Constants

const MAX_ACTOR_MEMORY: usize = 64;
const MAX_NOISE_MEMORY: usize = 64;
const MAX_TILE_MEMORY: usize = 4096;
const MAX_TURN_MEMORY: usize = 256;

const SOURCE_LIMIT_PC_: i32 = 2;
const SOURCE_LIMIT_NPC: i32 = 72;
const SOURCE_TRACKING_LIMIT: i32 = 16;

const SCENT_TRACKING_LIMIT: i32 = 64;

fn trophic_level(x: &Actor) -> i32 {
    if x.species.human() { 0 } else if !x.species.predator() { 1 } else { 2 }
}

//////////////////////////////////////////////////////////////////////////////

// Per-cell knowledge:

flags! { pub CellFlags(u32) { Light, Shade, Visible, SeeActorAt } }

type CF = CellFlags;

#[derive(Default)]
pub struct CellKnowledge {
    pub flags: CellFlags,
    pub items: ThinVec<Item>,
    pub point: Point,
    pub tile: &'static Tile,
    pub visibility: i32,

    pub last_seen: Timestamp,
    pub last_see_actor_at: Timestamp,
}

impl CellKnowledge {
    fn new(point: Point) -> Self {
        Self { point, ..Default::default() }
    }

    // Raw flags-based predicates:

    pub fn light(&self) -> bool { self.flags.any(CF::Light) }
    pub fn shade(&self) -> bool { self.flags.any(CF::Shade) }
    pub fn visible(&self) -> bool { self.flags.any(CF::Visible) }
    pub fn see_actor_at(&self) -> bool { self.flags.any(CF::SeeActorAt) }

    // Updates:

    fn clear_visibility_flags(&mut self) {
        self.flags &= !(CF::Visible | CF::SeeActorAt);
    }

    fn update(&mut self, cell: &Cell, flags: CellFlags, time: Timestamp, visibility: i32) {
        self.flags = flags;
        self.tile = cell.tile;
        self.visibility = visibility;
        self.last_seen = time;

        if self.see_actor_at() {
            // Used in hunting: we know the cell was unoccupied at this time.
            self.last_see_actor_at = time;

            // Items have visibility of a small actor.
            //
            // Clone the items list, but reuse the existing allocation, if any.
            self.items.clear();
            for &x in &cell.items { self.items.push(x); }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Per-Actor knowledge:

flags! { pub ActorFlags(u32) { Asleep, Friend, Rival, Sensed, Sneaking, Visible } }

type AF = ActorFlags;

pub struct ActorKnowledge {
    pub eid: EID,
    pub uid: UID,
    pub dir: Delta,
    pub loc: Location,
    pub name: Option<Rc<str>>,
    pub flags: ActorFlags,
    pub sense: Sense,
    pub species: &'static Species,
    pub team: ThinVec<bool>,

    // Stats:
    pub hp: f64,
    pub pp: f64,
    pub delta: i32,

    // Internal index:
    uids: ThinVec<UID>,
}

impl std::ops::Deref for ActorKnowledge {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl ActorKnowledge {
    fn new(eid: EID, uid: UID, species: &'static Species) -> Self {
        Self {
            eid,
            uid,
            dir: dirs::NONE,
            loc: Default::default(),
            name: Default::default(),
            flags: AF::Empty,
            sense: Sense::Sight,
            species,
            team: Default::default(),

            // Stats:
            hp: 0.,
            pp: 0.,
            delta: 0,

            // Internal index:
            uids: Default::default(),
        }
    }

    // Raw flags-based predicates:

    pub fn asleep(&self) -> bool { self.flags.any(AF::Asleep) }
    pub fn friend(&self) -> bool { self.flags.any(AF::Friend) }
    pub fn rival(&self) -> bool { self.flags.any(AF::Rival) }
    pub fn sensed(&self) -> bool { self.flags.any(AF::Sensed) }
    pub fn sneaking(&self) -> bool { self.flags.any(AF::Sneaking) }
    pub fn visible(&self) -> bool { self.flags.any(AF::Visible) }

    pub fn hidden_by_cover(&self) -> bool { self.sneaking() || !self.species.human() }

    // Updates:

    fn clear_visibility_flags(&mut self) {
        self.flags &= !(AF::Sensed | AF::Visible);
    }

    fn update_for_event(&mut self, event: &Event) {
        self.loc = event.loc;
        self.sense = event.sense;
    }

    fn update(&mut self, me: &Actor, other: &Actor, sense: Sense, time: Timestamp) {
        // Actors are friends iff they're both tame and in the same party.
        let leader = |x: &Actor| { x.leader.unwrap_or(x.eid) };

        // Actors are rivals iff one is tame and one is wild, or they're both
        // tame but in different parties.
        let trainer = |x: &Actor| {
            if let Some(x) = x.leader { return Some(x) };
            if x.species.human() { Some(x.eid) } else { None }
        };

        self.dir = other.dir;
        self.loc = Location { pos: other.pos, time };
        self.name = other.name.clone();
        self.sense = sense;
        self.species = other.species;

        self.team.clear();
        other.team.iter().for_each(|x| match x {
            Teammate::In(x) => self.team.push(x.cur_hp > 0),
            Teammate::Out(_) => self.team.push(true),
        });

        self.hp = other.hp_fraction();
        self.pp = 1. - clamp(other.move_timer as f64 / MOVE_TIMER as f64, 0., 1.);
        self.delta = trophic_level(other) - trophic_level(me);

        self.flags = AF::Sensed;
        if other.asleep { self.flags |= AF::Asleep; }
        if other.sneaking { self.flags |= AF::Sneaking; }
        if sense == Sense::Sight { self.flags |= AF::Visible; }
        if leader(me) == leader(other) { self.flags |= AF::Friend; }
        if trainer(me) != trainer(other) { self.flags |= AF::Rival; }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Per-noise knowledge:

pub struct NoiseKnowledge {
    eid: Option<EID>,
    pub uid: UID,
    pub loc: Location,
    pub sense: Sense,
    pub sound: Option<Sound>,
}

impl std::ops::Deref for NoiseKnowledge {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl NoiseKnowledge {
    fn new(eid: Option<EID>, event: &Event) -> Self {
        let Event { uid, loc, sense, .. } = *event;
        Self { eid, uid, loc, sense, sound: event.sound() }
    }

    pub fn freshness(&self, known: &Knowledge) -> f64 {
        let limit = max(SOURCE_LIMIT_PC_ - 1, 1);
        let turns = known.time_to_turn(self.time).floor() as i32;
        1. - clamp(turns, 0, limit) as f64 / limit as f64
    }
}

//////////////////////////////////////////////////////////////////////////////

// Per-scent knowledge:

#[derive(Clone, Copy)]
pub struct ScentKnowledge {
    pub delta: i32,
    pub loc: Location,
    pub species: &'static Species,
}

impl std::ops::Deref for ScentKnowledge {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

//////////////////////////////////////////////////////////////////////////////

// A Trace is either an Actor or a Noise - i.e. something we can track.

pub enum Trace<'a> {
    Actor(&'a ActorKnowledge),
    Noise(&'a NoiseKnowledge),
}

impl<'a> std::ops::Deref for Trace<'a> {
    type Target = Location;
    fn deref(&self) -> &Self::Target {
        match self {
            Trace::Actor(x) => &x.loc,
            Trace::Noise(x) => &x.loc,
        }
    }
}

impl<'a> Trace<'a> {
    pub fn uid(&self) -> UID {
        match self {
            Trace::Actor(x) => x.uid,
            Trace::Noise(x) => x.uid,
        }
    }

    pub fn actor(&self) -> Option<&'a ActorKnowledge> {
        if let Trace::Actor(x) = self { Some(x) } else { None }
    }

    pub fn noise(&self) -> Option<&'a NoiseKnowledge> {
        if let Trace::Noise(x) = self { Some(x) } else { None }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Cell state cache:

#[derive(Eq, PartialEq)]
struct PointEntry {
    cell: Option<CellHandle>,
    trace: Option<TraceHandle>,
    status: Status,
}

impl Default for PointEntry {
    fn default() -> Self {
        Self { cell: None, trace: None, status: Status::Unknown }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Overall knowledge index:

type CellHandle = Handle<CellKnowledge>;
type ActorHandle = Handle<ActorKnowledge>;
type NoiseHandle = Handle<NoiseKnowledge>;

#[derive(Clone, Copy, Eq, PartialEq)]
enum TraceHandle { Actor(ActorHandle), Noise(NoiseHandle) }

#[derive(Default, Eq, PartialEq)]
struct EIDEntry {
    actor: Option<ActorHandle>,
    noise: Option<NoiseHandle>,
}

#[derive(Default)]
pub struct Knowledge {
    // Core memories. Recent memories come first.
    pub cells: List<CellKnowledge>,
    pub actors: List<ActorKnowledge>,
    pub noises: List<NoiseKnowledge>,

    // Scents. Recent scents come first.
    pub scents: Vec<ScentKnowledge>,

    // Events. Recent events come last, unlike the other lists.
    pub events: Vec<Event>,

    moves: TileFlags,
    timer: TurnTimer,
    eid_index: HashMap<EID, EIDEntry>,
    uid_index: HashMap<UID, TraceHandle>,
    pos_index: HashMap<Point, PointEntry>,
    last_uid: u64,
}

impl Knowledge {
    // Reads:

    pub fn debug_time(&self, time: Timestamp) -> String {
        self.timer.debug_time(time)
    }

    pub fn default(&self) -> PointLookup<'_> {
        PointLookup { root: self, spot: None }
    }

    pub fn actor(&self, eid: EID) -> Option<&ActorKnowledge> {
        Some(&self.actors[self.eid_index.get(&eid)?.actor?])
    }

    pub fn trace(&self, uid: UID) -> Option<Trace<'_>> {
        match *self.uid_index.get(&uid)? {
            TraceHandle::Actor(x) => Some(Trace::Actor(&self.actors[x])),
            TraceHandle::Noise(x) => Some(Trace::Noise(&self.noises[x])),
        }
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

    pub fn cleanup_old_events(&mut self, player: bool, time: Timestamp) {
        self.timer.update(time);

        self.forget_old_scents();
        self.forget_old_noises(player);

        debug_assert!(self.check_invariants());
    }

    pub fn mark_turn_boundary(&mut self, speed: f64, time: Timestamp) {
        self.timer.end_turn(MAX_TURN_MEMORY, speed, time);
    }

    pub fn observe_event(&mut self, me: &Actor, eid: Option<EID>, event: &Event) {
        self.timer.update(event.time);

        type TH = TraceHandle;
        let mut cloned = event.clone();
        match self.event_noise(me, eid, &mut cloned) {
            TH::Actor(x) => self.update_actor(x, |x| x.update_for_event(&cloned)),
            TH::Noise(x) => self.update_noise(x, |x| *x = NoiseKnowledge::new(eid, &cloned)),
        }
        self.events.push(cloned);

        debug_assert!(self.check_invariants());
    }

    pub fn remove_actor(&mut self, eid: EID, time: Timestamp) {
        self.timer.update(time);

        let Some(x) = self.eid_index.get_mut(&eid) else { return };
        let Some(h) = x.actor.take() else { return };

        if *x == Default::default() { self.eid_index.remove(&eid); }

        let x = self.actors.remove(h);
        x.uids.iter().for_each(|x| { self.uid_index.remove(x); });
        self.remove_trace(TraceHandle::Actor(h), x.pos);
        self.uid_index.remove(&x.uid);

        self.forget_event(x.uid, x.pos);

        debug_assert!(self.check_invariants());
    }

    pub fn update_items(&mut self, pos: Point, item: &Item) {
        let Some(x) = self.pos_index.get(&pos) else { return };
        let Some(h) = x.cell else { return };
        self.cells[h].items.push(*item);
    }

    pub fn update(&mut self, me: &Actor, board: &Board, vision: &Vision, rng: &mut RNG) {
        self.timer.update(board.time);
        self.moves = me.species.moves;

        // Detect actors that were recently nearby by scent.
        self.populate_scents(me, board, rng);

        // Clear visibility flags. Visible cells come first in the list so we
        // can stop when we see the first one that's not visible.
        //
        // We can't apply this optimization to actors, because we may sense
        // them via off-turn event updates.
        for cell in &mut self.cells {
            if !cell.visible() { break; }
            cell.clear_visibility_flags();
        }
        for actor in &mut self.actors {
            actor.clear_visibility_flags();
        }

        // Actors know where their allies are, even if they can't see them.
        for &oid in me.leader.iter().chain(&me.summons) {
            let other = &board.actors[oid];
            self.observe_actor(me, other, Sense::Sound);
        }

        // If we're asleep, just update knowledge about ourself.
        if me.asleep {
            assert!(vision.get_points_seen().is_empty());
            self.observe_actor(me, me, Sense::Sight);
        }

        // Compute a sorted list of points we've seen and their visibility.
        let pos = me.pos;
        let mut seen: Vec<_> = vision.get_points_seen().iter().map(
            |&x| (x, vision.get_visibility_at(x))).collect();
        if !me.player { seen.sort_by_key(|&(x, _)| (x - pos).len_l2_squared()); }

        // Actors have exact knowledge about anything they can see.
        //
        // We want self.cells to be sorted by recency, and if there are ties,
        // by distance. Closer and more recently seen points come first.
        //
        // Within the loop here, we repeatedly move cells to the front of
        // self.cells. Because `seen` is sorted by distance, we iterate over
        // it in reverse order to get the desired ordering.
        for &(point, visibility) in seen.iter().rev() {
            self.observe_cell(me, board, point, visibility);
        }
        self.forget(me.player);

        debug_assert!(self.check_invariants());
    }

    // Miscellaneous update helpers:

    fn observe_actor(&mut self, me: &Actor, other: &Actor, sense: Sense) -> ActorHandle {
        let time = self.timer.time;
        let actor = self.actor_for_sighting(other, sense);
        self.update_actor(actor, |x| x.update(me, other, sense, time));
        actor
    }

    fn observe_cell(&mut self, me: &Actor, board: &Board, point: Point, visibility: i32) {
        let cell = board.get_cell(point);
        let Cell { eid, tile, .. } = *cell;

        let light = board.is_cell_lit(point);
        let unlit = matches!(board.get_light(), Light::None);
        let nearby = (point - me.pos).len_l1() <= 1;
        if unlit && !light && !nearby { return; }

        let visible = true;
        let shade = unlit || cell.shadow > 0;
        let is_shadow_cover = shade && !light;
        let see_big_actors = nearby || !is_shadow_cover;
        let see_all_actors = nearby || !(is_shadow_cover || tile.is_cover());

        // Compute the cell's new flags.
        let mut flags = CF::Empty;
        if light { flags |= CF::Light; }
        if shade { flags |= CF::Shade; }
        if visible { flags |= CF::Visible; }
        if see_all_actors { flags |= CF::SeeActorAt; }

        // Check if we can see the actor at the given cell.
        let actor = (|| {
            if !see_big_actors { return None; }
            let other = board.get_actor(eid?)?;
            if !see_all_actors && other.hidden_by_cover() { return None; }
            Some(self.observe_actor(me, other, Sense::Sight))
        })();

        // Update this point's cell, or create a new one.
        let entry = self.pos_index.entry(point).or_default();
        let handle = entry.cell.unwrap_or_else(
            || self.cells.push_front(CellKnowledge::new(point)));
        if entry.cell.is_some() { self.cells.move_to_front(handle); }
        if entry.cell.is_none() { entry.cell = Some(handle); }

        // Update basic information about the given cell.
        let memory = &mut self.cells[handle];
        memory.update(cell, flags, self.timer.time, visibility);

        // Clear the cell's trace if it's definitely unoccupied.
        if see_all_actors && actor.is_none() { entry.trace = None; }

        // Update the cell's pathfinding status.
        entry.status = if !tile.can_move_to(self.moves) {
            Status::Blocked
        } else if entry.trace.is_some() {
            Status::Occupied
        } else {
            Status::Free
        };
    }

    fn populate_scents(&mut self, me: &Actor, board: &Board, rng: &mut RNG) {
        if me.asleep || me.player { return; }

        let mut update = false;
        let leader = |x: &Actor| { x.leader.unwrap_or(x.eid) };

        for (_, other) in &board.actors {
            if leader(me) == leader(other) { continue; }

            let mut remainder = rng.random::<f64>();

            for (&loc, scent) in other.get_scent_trail(me.pos) {
                remainder -= scent * me.species.scent;
                if remainder >= 0. { continue; }

                let species = other.species;
                let delta = trophic_level(other) - trophic_level(me);
                self.scents.push(ScentKnowledge { delta, loc, species });
                update = true;
                break;
            }
        }
        if !update { return; }

        // Keep scents sorted by time and de-duplicated by species.
        let mut seen = HashSet::default();
        self.scents.sort_by_key(|x| self.timer.time - x.time);
        self.scents.retain(|x| seen.insert(x.species as *const Species));
    }

    // Actor identification:

    fn actor_for_sighting(&mut self, other: &Actor, sense: Sense) -> ActorHandle {
        let eid = other.eid;
        let limit = self.timer.time_at_turn(SOURCE_TRACKING_LIMIT);
        let entry = self.eid_index.entry(eid).or_default();

        let actor = entry.actor.unwrap_or_else(|| {
            self.last_uid += 1;
            let uid = UID(self.last_uid.try_into().unwrap());
            let result = self.actors.push_front(ActorKnowledge::new(eid, uid, other.species));
            self.uid_index.insert(uid, TraceHandle::Actor(result));
            result
        });
        entry.actor = Some(actor);

        if let Some(x) = entry.noise.take() &&
           let Some(x) = self.identify_noise(actor, x, limit) {
            let (uid, old) = (self.actors[actor].uid, Some(x));
            let loc = Location { pos: other.pos, time: self.timer.time };
            let event = Event { uid, old, loc, data: EventData::Spot, sense };
            self.events.push(event);
        }
        actor
    }

    fn event_noise(&mut self, me: &Actor, eid: Option<EID>, event: &mut Event) -> TraceHandle {
        assert!(event.old.is_none());

        // Sounds from non-actors result in new, unknown noises.
        let Some(eid) = eid else {
            return TraceHandle::Noise(self.create_noise(eid, event));
        };

        let limit = self.timer.time_at_turn(SOURCE_TRACKING_LIMIT);
        let entry = self.eid_index.entry(eid).or_default();
        let sense = event.sense;

        // Try to link this event to an actor we've seen.
        let link = |x: &ActorKnowledge| match sense {
            Sense::Sight => true,
            Sense::Smell => false,
            Sense::Sound => x.time > limit,
        };
        if !me.player && let Some(x) = entry.actor && link(&self.actors[x]) {
            event.uid = self.actors[x].uid;
            if let Some(y) = entry.noise.take() {
               event.old = self.identify_noise(x, y, limit);
            }
            return TraceHandle::Actor(x);
        }

        // Try to link the event to an actor we've sensed indirectly.
        if let Some(x) = entry.noise && self.noises[x].time > limit {
            event.uid = self.noises[x].uid;
            return TraceHandle::Noise(x);
        };

        // No existing noise - create a new, unknown one.
        let result = self.create_noise(Some(eid), event);
        if let Some(x) = self.eid_index.get_mut(&eid) { x.noise = Some(result); }
        TraceHandle::Noise(result)
    }

    // Actor updates:

    fn create_noise(&mut self, eid: Option<EID>, event: &mut Event) -> NoiseHandle {
        self.last_uid += 1;
        event.uid = UID(self.last_uid.try_into().unwrap());
        let result = self.noises.push_front(NoiseKnowledge::new(eid, event));
        self.uid_index.insert(event.uid, TraceHandle::Noise(result));
        result
    }

    fn identify_noise(&mut self, h: ActorHandle, s: NoiseHandle, limit: Timestamp) -> Option<UID> {
        let noise = &self.noises[s];
        if noise.time <= limit { return None; }

        let NoiseKnowledge { uid, loc, .. } = *noise;
        self.remove_trace(TraceHandle::Noise(s), loc.pos);
        self.noises.remove(s);

        self.uid_index.insert(uid, TraceHandle::Actor(h));
        self.actors[h].uids.push(uid);
        Some(uid)
    }

    fn remove_trace(&mut self, h: TraceHandle, prev: Point) {
        if let Some(x) = self.pos_index.get_mut(&prev) && x.trace == Some(h) {
            x.trace = None;
            match x.cell {
                Some(_) => if x.status != Status::Blocked { x.status = Status::Free; }
                None => { self.pos_index.remove(&prev); }
            }
        }
    }

    fn update_position(&mut self, h: TraceHandle, prev: Point, next: Option<Point>) {
        if next != Some(prev) { self.remove_trace(h, prev); }

        if let Some(next) = next {
            let x = self.pos_index.entry(next).or_default();
            if x.status != Status::Blocked { x.status = Status::Occupied; }
            x.trace = Some(h);
        }
    }

    fn update_actor(&mut self, h: ActorHandle, f: impl FnOnce(&mut ActorKnowledge)) {
        self.actors.move_to_front(h);

        let actor = &mut self.actors[h];
        let prev = actor.pos;
        f(actor);
        let next = actor.pos;

        self.update_position(TraceHandle::Actor(h), prev, Some(next));
    }

    fn update_noise(&mut self, h: NoiseHandle, f: impl FnOnce(&mut NoiseKnowledge)) {
        self.noises.move_to_front(h);

        let noise = &mut self.noises[h];
        let prev = noise.pos;
        f(noise);
        let next = noise.pos;

        self.update_position(TraceHandle::Noise(h), prev, Some(next));
    }

    // Cleanup:

    fn forget(&mut self, player: bool) {
        if player {
            while let Some(x) = self.cells.back() && !x.visible() {
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

        // Clean up actors, but not any that we can currently sense.
        while self.actors.len() > MAX_ACTOR_MEMORY {
            let actor = self.actors.back().unwrap();
            if actor.sensed() { break; }
            self.remove_actor(actor.eid, self.timer.time);
        }

        // Clean up noises by count. On turn boundaries, we drop them by age.
        while self.noises.len() > MAX_NOISE_MEMORY {
            self.forget_last_noise();
        }
    }

    fn forget_event(&mut self, uid: UID, pos: Point) {
        let loc = Location { pos, time: self.timer.time };
        let event = Event { uid, old: None, loc, data: EventData::Forget, sense: Sense::Sight };
        self.events.push(event);
    }

    fn forget_last_cell(&mut self) {
        let Some(x) = self.cells.pop_back() else { return };
        let Some(y) = self.pos_index.get_mut(&x.point) else { return };

        y.cell = None;
        match y.trace {
            Some(_) => y.status = Status::Occupied,
            None => { self.pos_index.remove(&x.point); }
        }
    }

    fn forget_last_noise(&mut self) {
        let Some(h) = self.noises.back_handle() else { return };
        let Some(x) = self.noises.pop_back() else { return };

        if let Some(e) = x.eid { self.forget_noise_link(e, h); }

        self.remove_trace(TraceHandle::Noise(h), x.pos);
        self.uid_index.remove(&x.uid);

        self.forget_event(x.uid, x.pos);
    }

    fn forget_old_scents(&mut self) {
        let limit = self.timer.time_at_turn(SCENT_TRACKING_LIMIT);

        while let Some(x) = self.scents.last() && x.time <= limit {
            self.scents.pop();
        }
    }

    fn forget_old_noises(&mut self, player: bool) {
        let turns = if player { SOURCE_LIMIT_PC_ } else { SOURCE_LIMIT_NPC };
        let limit = self.timer.time_at_turn(turns);

        while let Some(x) = self.noises.back() && x.time <= limit {
            self.forget_last_noise();
        }
    }

    fn forget_noise_link(&mut self, e: EID, h: NoiseHandle) {
        let Some(x) = self.eid_index.get_mut(&e) else { return };
        if x.noise != Some(h) { return };

        x.noise = None;
        if *x == Default::default() { self.eid_index.remove(&e); }
    }

    // Debug helpers:

    fn check_invariants(&self) -> bool {
        let check_sorted = |xs: Vec<Timestamp>| {
            let mut last = Timestamp::default();
            xs.into_iter().rev().for_each(|x| { assert!(x >= last); last = x; });
        };

        // Check that all lists are sorted in time order:
        check_sorted(self.cells.iter().map(|x| x.last_seen).collect());
        check_sorted(self.actors.iter().map(|x| x.time).collect());
        check_sorted(self.noises.iter().map(|x| x.time).collect());
        check_sorted(self.scents.iter().map(|x| x.time).collect());
        check_sorted(self.events.iter().rev().map(|x| x.time).collect());

        // Check that every cell, noise, and actor is indexed:
        type TH = TraceHandle;
        for x in &self.cells {
            let entry = self.pos_index.get(&x.point);
            assert!(entry.and_then(|x| x.cell).is_some());
        }
        for x in &self.noises {
            let entry = self.uid_index.get(&x.uid);
            assert!(matches!(entry, Some(&TH::Noise(y)) if std::ptr::eq(x, &self.noises[y])));
        }
        for x in &self.actors {
            let entry = self.eid_index.get(&x.eid);
            assert!(entry.and_then(|x| x.actor).is_some());

            let entry = self.uid_index.get(&x.uid);
            assert!(matches!(entry, Some(&TH::Actor(y)) if std::ptr::eq(x, &self.actors[y])));
        }

        // Check that the indices are consistent and minimal:
        for (&pos, point) in &self.pos_index {
            let PointEntry { cell, trace, status } = *point;
            assert!(cell.is_some() || trace.is_some());

            if let Some(x) = cell { assert!(self.cells[x].point == pos); }
            if let Some(TH::Actor(x)) = trace { assert!(self.actors[x].pos == pos); }
            if let Some(TH::Noise(x)) = trace { assert!(self.noises[x].pos == pos); }

            let lookup = PointLookup { root: self, spot: Some(point) };
            assert!(status == lookup.status_for(self.moves));

            let actual = if let Some(x) = cell && !self.cells[x].tile.can_move_to(self.moves) {
                Status::Blocked
            } else if trace.is_some() {
                Status::Occupied
            } else {
                Status::Free
            };
            assert!(actual == status);
        }
        for (&eid, &EIDEntry { actor, noise }) in &self.eid_index {
            assert!(actor.is_some() || noise.is_some());
            if let Some(x) = actor { assert!(self.actors[x].eid == eid); }
            if let Some(x) = noise { assert!(self.noises[x].eid == Some(eid)); }
        }
        true
    }
}

//////////////////////////////////////////////////////////////////////////////

// Result of querying knowledge about a cell

pub struct PointLookup<'a> {
    root: &'a Knowledge,
    spot: Option<&'a PointEntry>,
}

impl<'a> PointLookup<'a> {
    // Field lookups:

    pub fn last_seen(&self) -> Timestamp {
        self.cell().map_or_default(|x| x.last_seen)
    }

    pub fn last_see_actor_at(&self) -> Timestamp {
        self.cell().map_or_default(|x| x.last_see_actor_at)
    }

    pub fn items(&self) -> &[Item] {
        self.cell().map_or(&[], |x| x.items.as_slice())
    }

    pub fn light(&self) -> bool {
        self.cell().map_or(false, |x| x.light())
    }

    pub fn shade(&self) -> bool {
        self.cell().map_or(false, |x| x.shade())
    }

    pub fn tile(&self) -> Option<&'static Tile> {
        self.cell().map(|x| x.tile)
    }

    pub fn visibility(&self) -> i32 {
        self.cell().map_or(-1, |x| x.visibility)
    }

    // Derived fields:

    pub fn cell(&self) -> Option<&'a CellKnowledge> {
        Some(&self.root.cells[self.spot?.cell?])
    }

    pub fn actor(&self) -> Option<&'a ActorKnowledge> {
        let TraceHandle::Actor(x) = self.spot?.trace? else { return None };
        Some(&self.root.actors[x])
    }

    pub fn noise(&self) -> Option<&'a NoiseKnowledge> {
        let TraceHandle::Noise(x) = self.spot?.trace? else { return None };
        Some(&self.root.noises[x])
    }

    pub fn trace(&self) -> Option<Trace<'a>> {
        match self.spot?.trace? {
            TraceHandle::Actor(x) => Some(Trace::Actor(&self.root.actors[x])),
            TraceHandle::Noise(x) => Some(Trace::Noise(&self.root.noises[x])),
        }
    }

    pub fn status(&self) -> Status {
        self.spot.map_or(Status::Unknown, |x| x.status)
    }

    pub fn status_for(&self, moves: TileFlags) -> Status {
        if let Some(x) = self.tile() && !x.can_move_to(moves) { return Status::Blocked; }
        if self.occupied() { return Status::Occupied; }
        if self.spot.is_some() { return Status::Free; }
        Status::Unknown
    }

    // Predicates:

    pub fn occupied(&self) -> bool {
        self.spot.map_or(false, |x| x.trace.is_some())
    }

    pub fn blocked(&self) -> bool {
        self.spot.map_or(false, |x| x.status == Status::Blocked)
    }

    pub fn unblocked(&self) -> bool {
        self.spot.map_or(false, |x| x.status != Status::Blocked)
    }

    pub fn unknown(&self) -> bool {
        self.cell().is_none()
    }

    pub fn visible(&self) -> bool {
        self.cell().map_or(false, |x| x.visible())
    }

    pub fn can_see_actor_at(&self) -> bool {
        self.cell().map_or(false, |x| x.see_actor_at())
    }

    pub fn is_cover(&self) -> bool {
        self.cell().map_or(false, |x| x.tile.is_cover())
    }

    pub fn is_shadow_cover(&self) -> bool {
        self.cell().map_or(false, |x| x.shade() && !x.light())
    }
}
