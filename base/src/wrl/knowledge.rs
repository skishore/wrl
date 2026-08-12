use std::cmp::max;
use std::rc::Rc;

use rand::Rng;
use thin_vec::ThinVec;

use crate::flags;
use crate::base::point::{Delta, Point, dirs};
use crate::base::util::{HashMap, HashSet, RNG, clamp};
use crate::base::pathing::Status;
use crate::base::vision::Vision;

use super::dex::Species;
use super::entity::{EID, Entity, Teammate};
use super::event::{Event, EventData, Location, Sense, Sound, UID};
use super::game::{MOVE_TIMER, Board, Cell, Item, Light, Tile, TileFlags};
use super::list::{Handle, List};
use super::time::{Timestamp, TurnTimer};

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

// Per-cell knowledge:

flags! { pub CellFlags(u32) { Light, Shade, Visible, SeeEntityAt } }

type CF = CellFlags;

#[derive(Default)]
pub struct CellKnowledge {
    pub flags: CellFlags,
    pub items: ThinVec<Item>,
    pub point: Point,
    pub tile: &'static Tile,
    pub visibility: i32,

    pub last_seen: Timestamp,
    pub last_see_entity_at: Timestamp,
}

impl CellKnowledge {
    fn new(point: Point) -> Self {
        Self { point, ..Default::default() }
    }

    // Raw flags-based predicates:

    pub fn light(&self) -> bool { self.flags.any(CF::Light) }
    pub fn shade(&self) -> bool { self.flags.any(CF::Shade) }
    pub fn visible(&self) -> bool { self.flags.any(CF::Visible) }
    pub fn see_entity_at(&self) -> bool { self.flags.any(CF::SeeEntityAt) }

    // Updates:

    fn clear_visibility_flags(&mut self) {
        self.flags &= !(CF::Visible | CF::SeeEntityAt);
    }

    fn update(&mut self, cell: &Cell, flags: CellFlags, time: Timestamp, visibility: i32) {
        self.flags = flags;
        self.tile = cell.tile;
        self.visibility = visibility;
        self.last_seen = time;

        if self.see_entity_at() {
            // Used in hunting: we know the cell was unoccupied at this time.
            self.last_see_entity_at = time;

            // Items have visibility of a small entity.
            //
            // Clone the items list, but reuse the existing allocation, if any.
            self.items.clear();
            for &x in &cell.items { self.items.push(x); }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Per-entity knowledge:

flags! { pub EntityFlags(u32) { Asleep, Friend, Rival, Sensed, Sneaking, Visible } }

type EF = EntityFlags;

pub struct EntityKnowledge {
    pub eid: EID,
    pub dir: Delta,
    pub loc: Location,
    pub name: Option<Rc<str>>,
    pub flags: EntityFlags,
    pub sense: Sense,
    pub species: &'static Species,
    pub team: ThinVec<bool>,

    // Stats:
    pub hp: f64,
    pub pp: f64,
    pub delta: i32,
}

impl std::ops::Deref for EntityKnowledge {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl EntityKnowledge {
    fn new(eid: EID, species: &'static Species) -> Self {
        Self {
            eid,
            dir: dirs::NONE,
            loc: Default::default(),
            name: Default::default(),
            flags: EF::Empty,
            sense: Sense::Sight,
            species,
            team: Default::default(),

            // Stats:
            hp: 0.,
            pp: 0.,
            delta: 0,
        }
    }

    // Raw flags-based predicates:

    pub fn asleep(&self) -> bool { self.flags.any(EF::Asleep) }
    pub fn friend(&self) -> bool { self.flags.any(EF::Friend) }
    pub fn rival(&self) -> bool { self.flags.any(EF::Rival) }
    pub fn sensed(&self) -> bool { self.flags.any(EF::Sensed) }
    pub fn sneaking(&self) -> bool { self.flags.any(EF::Sneaking) }
    pub fn visible(&self) -> bool { self.flags.any(EF::Visible) }

    pub fn too_big_to_hide(&self) -> bool { self.species.human() && !self.sneaking() }

    // Updates:

    fn clear_visibility_flags(&mut self) {
        self.flags &= !(EF::Sensed | EF::Visible);
    }

    fn update_for_event(&mut self, event: &Event) {
        self.loc = event.loc;
        self.sense = event.sense;
    }

    fn update(&mut self, me: &Entity, other: &Entity, sense: Sense, time: Timestamp) {
        // Entities are friends iff they're both tame and in the same party.
        let leader = |entity: &Entity| { entity.leader.unwrap_or(entity.eid) };

        // Entities are rivals iff one is tame and one is wild, or they're
        // both tame but in different parties.
        let trainer = |entity: &Entity| {
            if let Some(x) = entity.leader { return Some(x) };
            if entity.species.human() { return Some(entity.eid); }
            None
        };

        self.dir = other.dir;
        self.loc = Location { pos: other.pos, time };
        self.name = other.name.clone();
        self.sense = sense;
        self.species = other.species;

        self.team.clear();
        other.team.iter().for_each(|x| match x {
            Teammate::Out(_) => self.team.push(true),
            Teammate::In(x) => self.team.push(x.cur_hp > 0),
        });

        self.hp = other.hp_fraction();
        self.pp = 1. - clamp(other.move_timer as f64 / MOVE_TIMER as f64, 0., 1.);
        self.delta = trophic_level(other) - trophic_level(me);

        self.flags = EF::Sensed;
        if other.asleep { self.flags |= EF::Asleep; }
        if other.sneaking { self.flags |= EF::Sneaking; }
        if sense == Sense::Sight { self.flags |= EF::Visible; }
        if leader(me) == leader(other) { self.flags |= EF::Friend; }
        if trainer(me) != trainer(other) { self.flags |= EF::Rival; }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Per-source knowledge:

pub struct SourceKnowledge {
    eid: Option<EID>,
    pub uid: UID,
    pub loc: Location,
    pub sense: Sense,
    pub sound: Option<Sound>,
}

impl std::ops::Deref for SourceKnowledge {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl SourceKnowledge {
    fn new(uid: UID, event: &Event) -> Self {
        let Event { eid, loc, sense, .. } = *event;
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

// Cell state cache:

#[derive(Eq, PartialEq)]
struct PointEntry {
    cell: Option<CellHandle>,
    occupant: Option<OccupantHandle>,
    status: Status,
}

impl Default for PointEntry {
    fn default() -> Self {
        Self { cell: None, occupant: None, status: Status::Unknown }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Overall knowledge index:

type CellHandle = Handle<CellKnowledge>;
type EntityHandle = Handle<EntityKnowledge>;
type SourceHandle = Handle<SourceKnowledge>;

#[derive(Clone, Copy, Eq, PartialEq)]
enum OccupantHandle { Entity(EntityHandle), Source(SourceHandle) }

#[derive(Default, Eq, PartialEq)]
struct EIDEntry {
    entity: Option<EntityHandle>,
    source: Option<SourceHandle>,
}

#[derive(Default)]
pub struct Knowledge {
    // Core memories. Recent memories come first.
    pub cells: List<CellKnowledge>,
    pub entities: List<EntityKnowledge>,
    pub sources: List<SourceKnowledge>,

    // Scents. Recent scents come first.
    pub scents: Vec<ScentKnowledge>,

    // Events. Recent events come last, unlike the other lists.
    pub events: Vec<Event>,

    moves: TileFlags,
    timer: TurnTimer,
    eid_index: HashMap<EID, EIDEntry>,
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
        self.timer.end_turn(MAX_TURN_MEMORY, speed, time);

        self.forget_old_scents();
        self.forget_old_sources(player);

        debug_assert!(self.check_invariants());
    }

    pub fn observe_event(&mut self, me: &Entity, event: &Event) {
        self.timer.update(event.time);

        type OH = OccupantHandle;
        let mut cloned = event.clone();
        match self.source_for_event(me, &mut cloned) {
            OH::Entity(x) => self.update_entity(x, |x| x.update_for_event(event)),
            OH::Source(x) => self.update_source(x, |x| *x = SourceKnowledge::new(x.uid, event)),
        }
        self.events.push(cloned);

        debug_assert!(self.check_invariants());
    }

    pub fn remove_entity(&mut self, eid: EID, time: Timestamp) {
        self.timer.update(time);

        let Some(x) = self.eid_index.get_mut(&eid) else { return };
        let Some(h) = x.entity.take() else { return };

        if *x == Default::default() { self.eid_index.remove(&eid); }

        let pos = self.delete_entity(h);

        self.forget_event(Some(eid), None, pos);

        debug_assert!(self.check_invariants());
    }

    pub fn update_items(&mut self, pos: Point, item: &Item) {
        let Some(x) = self.pos_index.get(&pos) else { return };
        let Some(h) = x.cell else { return };

        self.cells[h].items.push(item.clone());
    }

    pub fn update(&mut self, me: &Entity, board: &Board, vision: &Vision, rng: &mut RNG) {
        self.timer.update(board.time);

        let (pos, time) = (me.pos, board.time);
        let unlit = matches!(board.get_light(), Light::None);
        let moves = me.species.moves;
        self.moves = moves;

        // Detect entities that were recently nearby by scent.
        self.populate_scents(me, board, rng);

        // Clear visibility flags. Visible cells come first in the list so we
        // can stop when we see the first one that's not visible.
        //
        // We can't apply the same optimization to entities, because we may
        // sense them via off-turn event updates.
        for cell in &mut self.cells {
            if !cell.visible() { break; }
            cell.clear_visibility_flags();
        }
        for entity in &mut self.entities {
            entity.clear_visibility_flags();
        }

        // Entities know where their allies are, even if they can't see them.
        for &oid in me.leader.iter().chain(&me.summons) {
            let other = &board.entities[oid];
            self.observe_entity(me, other, Sense::Sound);
        }

        // If we're asleep, just update knowledge about ourself.
        if me.asleep {
            assert!(vision.get_points_seen().is_empty());
            self.observe_entity(me, me, Sense::Sight);
        }

        // Compute a sorted list of points we've seen and their visibility.
        let mut seen: Vec<_> = vision.get_points_seen().iter().map(
            |&x| (x, vision.get_visibility_at(x))).collect();
        if !me.player {
            seen.sort_by_key(|&(x, _)| (x - pos).len_l2_squared());
        }

        // Entities have exact knowledge about anything they can see.
        //
        // We want self.cells to be sorted by recency, and if there are ties,
        // by distance. Closer and more recently seen points come first.
        //
        // Within the loop here, we repeatedly move cells to the front of
        // self.cells. Because `seen` is sorted by distance, we iterate over
        // it in reverse order to get the desired ordering.
        for &(point, visibility) in seen.iter().rev() {
            let cell = board.get_cell(point);
            let Cell { eid, tile, .. } = *cell;

            let light = board.is_cell_lit(point);
            let nearby = (point - pos).len_l1() <= 1;
            if unlit && !light && !nearby { continue; }

            let visible = true;
            let shade = unlit || cell.shadow > 0;
            let is_shadow_cover = shade && !light;
            let see_big_entities = nearby || !is_shadow_cover;
            let see_all_entities = nearby || !(is_shadow_cover || tile.is_cover());

            // Compute the cell's new flags.
            let mut flags = CF::Empty;
            if light { flags |= CF::Light; }
            if shade { flags |= CF::Shade; }
            if visible { flags |= CF::Visible; }
            if see_all_entities { flags |= CF::SeeEntityAt; }

            // Check if we can see the entity at the given cell.
            let entity = (|| {
                if !see_big_entities { return None; }
                let other = board.get_entity(eid?)?;
                if !see_all_entities && !other.too_big_to_hide() { return None; }
                Some(self.observe_entity(me, other, Sense::Sight))
            })();

            // Update this point's cell, or create a new one.
            let entry = self.pos_index.entry(point).or_default();
            let handle = entry.cell.unwrap_or_else(
                || self.cells.push_front(CellKnowledge::new(point)));
            if entry.cell.is_some() { self.cells.move_to_front(handle); }
            if entry.cell.is_none() { entry.cell = Some(handle); }

            // Update basic information about the given cell.
            let memory = &mut self.cells[handle];
            memory.update(cell, flags, time, visibility);

            // Clear the cell's entity if it's definitely unoccupied.
            if see_all_entities && entity.is_none() { entry.occupant = None; }

            // Update the cell's pathfinding status.
            entry.status = if !tile.can_move_to(moves) {
                Status::Blocked
            } else if entry.occupant.is_some() {
                Status::Occupied
            } else {
                Status::Free
            };
        }

        self.forget(me.player);

        debug_assert!(self.check_invariants());
    }

    // Miscellaneous update helpers:

    fn observe_entity(&mut self, me: &Entity, other: &Entity, sense: Sense) -> EntityHandle {
        let time = self.timer.time;
        let entity = self.entity_for_sighting(other, sense);
        self.update_entity(entity, |x| x.update(me, other, sense, time));
        entity
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

    // Entity identification:

    fn entity_for_sighting(&mut self, other: &Entity, sense: Sense) -> EntityHandle {
        let eid = other.eid;
        let limit = self.timer.time_at_turn(SOURCE_TRACKING_LIMIT);
        let entry = self.eid_index.entry(eid).or_default();

        let entity = entry.entity.unwrap_or_else(
            || self.entities.push_front(EntityKnowledge::new(eid, other.species)));
        entry.entity = Some(entity);

        if let Some(x) = entry.source.take() &&
           let Some(uid) = self.identify_source(x, limit) {
            let (eid, uid) = (Some(eid), Some(uid));
            let loc = Location { pos: other.pos, time: self.timer.time };
            let event = Event { eid, uid, loc, data: EventData::Spot, sense };
            self.events.push(event);
        }

        entity
    }

    fn source_for_event(&mut self, me: &Entity, event: &mut Event) -> OccupantHandle {
        assert!(event.uid.is_none());
        let eid = event.eid.take();

        let unknown = |event: &mut Event, last_uid: &mut u64, sources: &mut List<SourceKnowledge>| {
            *last_uid += 1;
            let uid = UID((*last_uid).try_into().unwrap());
            let handle = sources.push_front(SourceKnowledge::new(uid, event));
            event.uid = Some(uid);
            handle
        };

        // Sounds from non-entities result in new, unknown sources.
        let Some(eid) = eid else {
            return OccupantHandle::Source(unknown(event, &mut self.last_uid, &mut self.sources))
        };

        let limit = self.timer.time_at_turn(SOURCE_TRACKING_LIMIT);
        let entry = self.eid_index.entry(eid).or_default();
        let sense = event.sense;

        // Try to link this event to an entity we've seen.
        let link = |x: &EntityKnowledge| match sense {
            Sense::Sight => true,
            Sense::Smell => false,
            Sense::Sound => x.time > limit,
        };
        if !me.player && let Some(x) = entry.entity && link(&self.entities[x]) {
            event.eid = Some(eid);
            if let Some(y) = entry.source.take() {
               event.uid = self.identify_source(y, limit);
            }
            return OccupantHandle::Entity(x);
        }

        // Try to link the event to an entity we've sensed indirectly.
        if let Some(x) = entry.source && self.sources[x].time > limit {
            event.uid = Some(self.sources[x].uid);
            return OccupantHandle::Source(x);
        };

        // No existing source - create a new, unknown one.
        let result = unknown(event, &mut self.last_uid, &mut self.sources);
        entry.source = Some(result);
        OccupantHandle::Source(result)
    }

    // Entity updates:

    fn delete_entity(&mut self, h: EntityHandle) -> Point {
        let pos = self.entities.remove(h).loc.pos;
        self.update_pos(OccupantHandle::Entity(h), Some(pos), None);
        pos
    }

    fn delete_source(&mut self, h: SourceHandle) -> Point {
        let pos = self.sources.remove(h).loc.pos;
        self.update_pos(OccupantHandle::Source(h), Some(pos), None);
        pos
    }

    fn identify_source(&mut self, s: SourceHandle, limit: Timestamp) -> Option<UID> {
        let source = &self.sources[s];
        if source.time <= limit { return None; }

        let uid = source.uid;
        self.delete_source(s);
        Some(uid)
    }

    fn update_pos(&mut self, h: OccupantHandle, prev: Option<Point>, next: Option<Point>) {
        if prev != next && let Some(prev) = prev {
            if let Some(x) = self.pos_index.get_mut(&prev) && x.occupant == Some(h) {
                x.occupant = None;
                match x.cell {
                    Some(_) => if x.status != Status::Blocked { x.status = Status::Free; }
                    None => { self.pos_index.remove(&prev); }
                }
            }
        }

        if let Some(next) = next {
            let x = self.pos_index.entry(next).or_default();
            if x.status != Status::Blocked { x.status = Status::Occupied; }
            x.occupant = Some(h);
        }
    }

    fn update_entity(&mut self, h: EntityHandle, f: impl FnOnce(&mut EntityKnowledge)) {
        self.entities.move_to_front(h);

        let entity = &mut self.entities[h];
        let prev = entity.pos;
        f(entity);
        let next = entity.pos;

        self.update_pos(OccupantHandle::Entity(h), Some(prev), Some(next));
    }

    fn update_source(&mut self, h: SourceHandle, f: impl FnOnce(&mut SourceKnowledge)) {
        self.sources.move_to_front(h);

        let source = &mut self.sources[h];
        let prev = source.pos;
        f(source);
        let next = source.pos;

        self.update_pos(OccupantHandle::Source(h), Some(prev), Some(next));
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

        // Clean up entities, but not any that we can currently sense.
        while self.entities.len() > MAX_ENTITY_MEMORY {
            let entity = self.entities.back().unwrap();
            if entity.sensed() { break; }
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
        match y.occupant {
            Some(_) => y.status = Status::Occupied,
            None => { self.pos_index.remove(&x.point); }
        }
    }

    fn forget_last_source(&mut self) {
        let Some(h) = self.sources.back_handle() else { return };
        let Some(x) = self.sources.pop_back() else { return };

        if let Some(e) = x.eid { self.forget_source_link(e, h); }

        self.update_pos(OccupantHandle::Source(h), Some(x.pos), None);

        self.forget_event(None, Some(x.uid), x.pos);
    }

    fn forget_old_scents(&mut self) {
        let limit = self.timer.time_at_turn(SCENT_TRACKING_LIMIT);

        while let Some(x) = self.scents.last() && x.time <= limit {
            self.scents.pop();
        }
    }

    fn forget_old_sources(&mut self, player: bool) {
        let turns = if player { SOURCE_LIMIT_PC_ } else { SOURCE_LIMIT_NPC };
        let limit = self.timer.time_at_turn(turns);

        while let Some(x) = self.sources.back() && x.time <= limit {
            self.forget_last_source();
        }
    }

    fn forget_source_link(&mut self, e: EID, h: SourceHandle) {
        let Some(x) = self.eid_index.get_mut(&e) else { return };
        if x.source != Some(h) { return };

        x.source = None;
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
        for (&pos, point) in &self.pos_index {
            let PointEntry { cell, occupant, status } = *point;
            assert!(cell.is_some() || occupant.is_some());

            if let Some(x) = cell { assert!(self.cells[x].point == pos); }
            if let Some(OH::Entity(x)) = occupant { assert!(self.entities[x].pos == pos); }
            if let Some(OH::Source(x)) = occupant { assert!(self.sources[x].pos == pos); }

            let lookup = PointLookup { root: self, spot: Some(point) };
            assert!(status == lookup.status_for(self.moves));

            let actual = if let Some(x) = cell && !self.cells[x].tile.can_move_to(self.moves) {
                Status::Blocked
            } else if occupant.is_some() {
                Status::Occupied
            } else {
                Status::Free
            };
            assert!(actual == status);
        }
        for (&eid, &EIDEntry { entity, source }) in &self.eid_index {
            assert!(entity.is_some() || source.is_some());
            if let Some(x) = entity { assert!(self.entities[x].eid == eid); }
            if let Some(x) = source { assert!(self.sources[x].eid == Some(eid)); }
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

    pub fn last_see_entity_at(&self) -> Timestamp {
        self.cell().map_or_default(|x| x.last_see_entity_at)
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

    pub fn entity(&self) -> Option<&'a EntityKnowledge> {
        let OccupantHandle::Entity(x) = self.spot?.occupant? else { return None };
        Some(&self.root.entities[x])
    }

    pub fn source(&self) -> Option<&'a SourceKnowledge> {
        let OccupantHandle::Source(x) = self.spot?.occupant? else { return None };
        Some(&self.root.sources[x])
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
        self.spot.map_or(false, |x| x.occupant.is_some())
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

    pub fn can_see_entity_at(&self) -> bool {
        self.cell().map_or(false, |x| x.see_entity_at())
    }

    pub fn is_cover(&self) -> bool {
        self.cell().map_or(false, |x| x.tile.is_cover())
    }

    pub fn is_shadow_cover(&self) -> bool {
        self.cell().map_or(false, |x| x.shade() && !x.light())
    }
}
