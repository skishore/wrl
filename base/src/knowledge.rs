use std::cmp::max;

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
const MAX_TILE_MEMORY: usize = 4096;

//////////////////////////////////////////////////////////////////////////////

// New TD / TS (turn-free timedelta / timestamp)

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct TD(i32);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TS(u32);

impl TD {
    const SCALE: f64 = 1000.;

    pub fn max() -> Self {
        Self(i32::MAX)
    }

    pub fn from_seconds(seconds: f64) -> Self {
        Self((TD::SCALE * seconds).round() as i32)
    }

    pub fn to_seconds(&self) -> f64 {
        (1. / TD::SCALE) * self.0 as f64
    }
}

impl std::ops::Add<TD> for TD {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self(self.0 + o.0)
    }
}

impl std::ops::Sub for TD {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self(self.0 - o.0)
    }
}

impl std::ops::Add<TD> for TS {
    type Output = Self;
    fn add(self, o: TD) -> Self::Output {
        Self(self.0.wrapping_add(o.0 as u32))
    }
}

impl std::ops::Sub for TS {
    type Output = TD;
    fn sub(self, o: Self) -> Self::Output {
        TD(self.0.wrapping_sub(o.0) as i32)
    }
}

impl std::fmt::Display for TS {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let scale = TD::SCALE as u32;
        let (left, msec) = (self.0 / scale, self.0 % scale);
        let (left, sec) = (left / 60, left % 60);
        let (left, min) = (left / 60, left % 60);
        write!(fmt, "{:0>2}:{:0>2}:{:0>2}:{:0>3}", left, min, sec, msec)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Timestamp

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Timedelta { pub time: TD, pub ticks: i32, pub turns: i32 }

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Timestamp { time: TS, tick: u32, turn: u32 }

impl Timedelta {
    pub fn max() -> Self {
        Self { time: TD::max(), ticks: i32::MAX, turns: i32::MAX }
    }
}

impl std::ops::Add<Timedelta> for Timedelta {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self { time: self.time + o.time, ticks: self.ticks + o.ticks, turns: self.turns + o.turns }
    }
}

impl std::ops::Sub for Timedelta {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self { time: self.time - o.time, ticks: self.ticks - o.ticks, turns: self.turns - o.turns }
    }
}

impl std::ops::Add<Timedelta> for Timestamp {
    type Output = Self;
    fn add(self, o: Timedelta) -> Self::Output {
        let time = self.time + o.time;
        let tick = self.tick.wrapping_add(o.ticks as u32);
        let turn = self.turn.wrapping_add(o.turns as u32);
        Self::Output { time, tick, turn }
    }
}

impl std::ops::Sub for Timestamp {
    type Output = Timedelta;
    fn sub(self, o: Self) -> Self::Output {
        let time = self.time - o.time;
        let ticks = self.tick.wrapping_sub(o.tick) as i32;
        let turns = self.turn.wrapping_sub(o.turn) as i32;
        Self::Output { time, ticks, turns }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

type CellHandle = Handle<CellKnowledge>;
type EntityHandle = Handle<EntityKnowledge>;

#[derive(Clone)]
pub struct CellKnowledge {
    handle: Option<EntityHandle>,
    pub last_see_entity_at: Timestamp,
    pub last_seen: Timestamp,
    pub items: ThinVec<Item>,
    pub point: Point,
    pub shade: bool,
    pub tile: &'static Tile,
    visibility: i32,
}
#[cfg(target_pointer_width = "32")]
static_assert_size!(CellKnowledge, 40);
#[cfg(target_pointer_width = "64")]
static_assert_size!(CellKnowledge, 64);

pub struct EntityKnowledge {
    pub eid: EID,
    pub age: Timedelta,
    pub pos: Point,
    pub dir: Point,
    pub glyph: Glyph,

    // Public info about the entity's status.
    pub name: &'static str,
    pub hp: f64,
    pub pp: f64,

    // Small booleans, including some judgments.
    pub alive: bool,
    pub heard: bool,
    pub moved: bool,
    pub rival: bool,
    pub friend: bool,
    pub asleep: bool,
    pub player: bool,
    pub sneaking: bool,
}
#[cfg(target_pointer_width = "32")]
static_assert_size!(EntityKnowledge, 76);
#[cfg(target_pointer_width = "64")]
static_assert_size!(EntityKnowledge, 88);

#[derive(Default)]
pub struct Knowledge {
    cell_by_point: HashMap<Point, CellHandle>,
    entity_by_eid: HashMap<EID, EntityHandle>,
    pub cells: List<CellKnowledge>,
    pub entities: List<EntityKnowledge>,
    pub time: Timestamp,

    // Scent information
    pub picked_up_scent: bool,
    pub scent_steps: HashMap<Point, i32>,
}

impl CellKnowledge {
    fn new(point: Point, tile: &'static Tile) -> Self {
        Self {
            handle: None,
            items: thin_vec![],
            last_seen: Default::default(),
            last_see_entity_at: Default::default(),
            point,
            shade: false,
            tile,
            visibility: -1,
        }
    }
}

impl EntityKnowledge {
    pub fn visible(&self) -> bool {
        self.age.ticks == 0
    }
}

impl Knowledge {
    // Reads

    pub fn default(&self) -> CellResult {
        CellResult { root: self, cell: None }
    }

    pub fn entity(&self, eid: EID) -> Option<&EntityKnowledge> {
        self.entity_by_eid.get(&eid).map(|&x| &self.entities[x])
    }

    pub fn get(&self, p: Point) -> CellResult {
        let cell_handle = self.cell_by_point.get(&p);
        CellResult { root: self, cell: cell_handle.map(|&x| &self.cells[x]) }
    }

    // Writes

    pub fn advance_time(&mut self, delta: Timedelta, player: bool) {
        self.time = self.time + delta;
        for x in &mut self.entities { x.age = x.age + delta; }
        if !player { return; }

        while let Some(x) = self.cells.back() && self.time.tick != x.last_seen.tick {
            self.forget_last_cell();
        }
    }

    pub fn update(&mut self, me: &Entity, board: &Board, vision: &Vision, rng: &mut RNG) {
        let time = board.time - self.time.time;
        self.advance_time(Timedelta { time, ticks: 1, turns: 0 }, me.player);

        let (pos, time) = (me.pos, self.time);
        let dark = matches!(board.get_light(), Light::None);

        // Clear and recompute scents. Scent is a local sense. In the future,
        // we should deliver all scents when we're tracking and let the AI
        // code choose the best one - it also checks that we can move there.
        self.picked_up_scent = false;
        self.scent_steps.clear();
        if me.tracking > 0 {
            let radius = 2;
            for x in -radius..=radius {
                for y in -radius..=radius {
                    let step = Point(x, y);
                    self.scent_steps.insert(step, board.get_cell(me.pos + step).scent);
                }
            }
        } else {
            let scent = board.get_cell(me.pos).scent;
            if scent > 0 {
                let chance = 0.05 * ((scent as f64).log10() + 1.);
                self.picked_up_scent = rng.gen::<f64>() < chance;
            }
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

            let shade = dark || cell.shadow > 0;
            let see_big_entities = nearby || !shade;
            let see_all_entities = nearby || !(shade || tile.limits_vision());

            let handle = (|| {
                if !see_big_entities { return None; }
                let other = board.get_entity(eid?)?;
                let big = other.player && !other.sneaking;
                if !big && !see_all_entities { return None; }
                let (seen, heard) = (true, false);
                Some(self.update_entity(me, other, seen, heard))
            })();

            let cell_handle = *self.cell_by_point.entry(point).and_modify(|&mut x| {
                self.cells.move_to_front(x);
            }).or_insert_with(|| {
                self.cells.push_front(CellKnowledge::new(point, tile))
            });

            // Update basic information about the given cell.
            let cell = &mut self.cells[cell_handle];
            cell.last_seen = time;
            cell.point = point;
            cell.shade = shade;
            cell.tile = tile;

            // Clone items, but reuse the existing allocation, if any.
            cell.items.clear();
            for &x in items { cell.items.push(x); }

            // Only clear the cell's entity if we can see entities there.
            if see_all_entities {
                cell.last_see_entity_at = time;
            }
            if see_all_entities || handle.is_some() {
                let existing = std::mem::replace(&mut cell.handle, handle);
                if existing != handle && let Some(x) = existing {
                    self.mark_entity_moved(x, point);
                };
            }
        }

        self.forget(me.player);
    }

    pub fn update_entity(&mut self, entity: &Entity, other: &Entity,
                         seen: bool, heard: bool) -> EntityHandle {
        let handle = *self.entity_by_eid.entry(other.eid).and_modify(|&mut x| {
            self.entities.move_to_front(x);
            let existing = &mut self.entities[x];
            if !existing.moved && !(seen && existing.pos == other.pos) {
                let cell_handle = self.cell_by_point.get(&existing.pos);
                let cell = &mut self.cells[*cell_handle.unwrap()];
                assert!(cell.handle == Some(x));
                cell.handle = None;
            };
        }).or_insert_with(|| {
            self.entities.push_front(EntityKnowledge {
                eid: other.eid,
                age: Default::default(),
                pos: Default::default(),
                dir: Default::default(),
                name: Default::default(),
                hp: Default::default(),
                pp: Default::default(),
                alive: Default::default(),
                glyph: Default::default(),
                heard: Default::default(),
                moved: Default::default(),
                rival: Default::default(),
                friend: Default::default(),
                asleep: Default::default(),
                player: Default::default(),
                sneaking: Default::default(),
            })
        });

        let same = other.eid == entity.eid;
        let entry = &mut self.entities[handle];
        let aggressor = |x: &Entity| x.predator;
        let rival = !same && (aggressor(entity) != aggressor(other));

        entry.age = Default::default();
        entry.pos = other.pos;
        entry.dir = other.dir;
        entry.alive = other.cur_hp > 0;
        entry.glyph = other.glyph;

        entry.name = if other.player { "skishore" } else {
            if other.predator { "Rattata" } else { "Pidgey" }
        };
        entry.hp = other.cur_hp as f64 / max(other.max_hp, 1) as f64;
        entry.pp = 1. - clamp(other.move_timer as f64 / MOVE_TIMER as f64, 0., 1.);

        entry.heard = heard;
        entry.moved = !seen;
        entry.rival = rival;
        entry.friend = same;
        entry.asleep = other.asleep;
        entry.player = other.player;
        entry.sneaking = other.sneaking;

        handle
    }

    pub fn remove_entity(&mut self, oid: EID) {
        let Some(handle) = self.entity_by_eid.remove(&oid) else { return };
        let EntityKnowledge { moved, pos, .. } = self.entities.remove(handle);
        if !moved {
            let cell_handle = self.cell_by_point.get(&pos);
            let cell = &mut self.cells[*cell_handle.unwrap()];
            assert!(cell.handle == Some(handle));
            cell.handle = None;
        }
    }

    // Private helpers

    fn forget(&mut self, player: bool) {
        for entity in &mut self.entities {
            if !entity.heard { continue; }
            let Some(&h) = self.cell_by_point.get(&entity.pos) else { continue; };
            if self.cells[h].last_see_entity_at.tick == self.time.tick { entity.heard = false; }
        }
        if player { return; }

        while self.cell_by_point.len() > MAX_TILE_MEMORY {
            // We don't need to check age, here; we can only see a bounded
            // number of cells per turn, much less than MAX_TILE_MEMORY.
            self.forget_last_cell();
        }

        while self.entity_by_eid.len() > MAX_ENTITY_MEMORY {
            let entity = self.entities.back().unwrap();
            if entity.age.ticks == 0 { break; }

            let handle = self.entity_by_eid.remove(&entity.eid).unwrap();
            if !entity.moved {
                let cell_handle = self.cell_by_point.get(&entity.pos);
                let cell = &mut self.cells[*cell_handle.unwrap()];
                assert!(cell.handle == Some(handle));
                cell.handle = None;
            }
            self.entities.pop_back();
        }
    }

    fn forget_last_cell(&mut self) {
        let CellKnowledge { point, handle, .. } = self.cells.pop_back().unwrap();
        if let Some(x) = handle { self.mark_entity_moved(x, point); }
        self.cell_by_point.remove(&point);
    }

    fn mark_entity_moved(&mut self, handle: EntityHandle, pos: Point) {
        let entity = &mut self.entities[handle];
        assert!(entity.pos == pos);
        assert!(!entity.moved);
        entity.moved = true;
    }
}

//////////////////////////////////////////////////////////////////////////////

// Result of querying knowledge about a cell

pub struct CellResult<'a> {
    root: &'a Knowledge,
    cell: Option<&'a CellKnowledge>,
}

impl<'a> CellResult<'a> {
    // Field lookups

    pub fn get_cell(&self) -> Option<&CellKnowledge> { self.cell }

    pub fn time_since_seen(&self) -> Timedelta {
        let Some(x) = self.cell else { return Timedelta::max() };
        self.root.time - x.last_seen
    }

    pub fn time_since_entity_visible(&self) -> Timedelta {
        let Some(x) = self.cell else { return Timedelta::max() };
        self.root.time - x.last_see_entity_at
    }

    pub fn items(&self) -> &[Item] {
        self.cell.map(|x| x.items.as_slice()).unwrap_or(&[])
    }

    pub fn shade(&self) -> bool {
        self.cell.map(|x| x.shade).unwrap_or(false)
    }

    pub fn tile(&self) -> Option<&'static Tile> {
        self.cell.map(|x| x.tile)
    }

    pub fn visibility(&self) -> i32 {
        let Some(x) = self.cell else { return -1 };
        if x.last_seen.tick == self.root.time.tick { x.visibility } else { -1 }
    }

    // Derived fields

    pub fn entity(&self) -> Option<&'a EntityKnowledge> {
        self.cell.and_then(|x| x.handle.map(|y| &self.root.entities[y]))
    }

    pub fn status(&self) -> Status {
        let Some(x) = self.cell else { return Status::Unknown; };
        if x.handle.is_some() { return Status::Occupied; }
        if x.tile.blocks_movement() { Status::Blocked } else { Status::Free }
    }

    // Predicates

    pub fn blocked(&self) -> bool {
        self.cell.map(|x| x.tile.blocks_movement()).unwrap_or(false)
    }

    pub fn unblocked(&self) -> bool {
        self.cell.map(|x| !x.tile.blocks_movement()).unwrap_or(false)
    }

    pub fn unknown(&self) -> bool {
        self.cell.is_none()
    }

    pub fn visible(&self) -> bool {
        let Some(x) = self.cell else { return false };
        x.last_seen.tick == self.root.time.tick
    }

    pub fn can_see_entity_at(&self) -> bool {
        let Some(x) = self.cell else { return false };
        x.last_see_entity_at.tick == self.root.time.tick
    }
}
