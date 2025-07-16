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

// Knowledge

type CellHandle = Handle<CellKnowledge>;
type EntityHandle = Handle<EntityKnowledge>;

pub struct CellKnowledge {
    handle: Option<EntityHandle>,
    pub last_see_entity_at: Timestamp,
    pub last_seen: Timestamp,
    pub items: ThinVec<Item>,
    pub point: Point,
    pub tile: &'static Tile,
    visibility: i32,

    // Flags:
    pub shade: bool,
    pub visible: bool,
    pub see_entity_at: bool,
}
#[cfg(target_pointer_width = "32")]
static_assert_size!(CellKnowledge, 48);
#[cfg(target_pointer_width = "64")]
static_assert_size!(CellKnowledge, 56);

pub struct EntityKnowledge {
    pub eid: EID,
    pub pos: Point,
    pub dir: Point,
    pub glyph: Glyph,
    pub name: &'static str,
    pub time: Timestamp,

    // Stats:
    pub hp: f64,
    pub pp: f64,
    pub delta: i32,

    // Flags:
    pub alive: bool,
    pub heard: bool,
    pub moved: bool,
    pub asleep: bool,
    pub friend: bool,
    pub player: bool,
    pub visible: bool,
    pub sneaking: bool,
}
#[cfg(target_pointer_width = "32")]
static_assert_size!(EntityKnowledge, 80);
#[cfg(target_pointer_width = "64")]
static_assert_size!(EntityKnowledge, 88);

#[derive(Clone, Copy)]
pub struct Scent {
    pub pos: Point,
    pub time: Timestamp,
}

#[derive(Default)]
pub struct Knowledge {
    cell_by_point: HashMap<Point, CellHandle>,
    entity_by_eid: HashMap<EID, EntityHandle>,
    pub cells: List<CellKnowledge>,
    pub entities: List<EntityKnowledge>,
    pub scents: Vec<Scent>,
    pub time: Timestamp,
}

impl CellKnowledge {
    fn new(point: Point, tile: &'static Tile) -> Self {
        Self {
            handle: None,
            items: thin_vec![],
            last_seen: Default::default(),
            last_see_entity_at: Default::default(),
            point,
            tile,
            visibility: -1,

            // Flags:
            shade: false,
            visible: false,
            see_entity_at: false,
        }
    }
}

impl Knowledge {
    // Reads

    pub fn default(&self) -> CellResult<'_> {
        CellResult { root: self, cell: None }
    }

    pub fn entity(&self, eid: EID) -> Option<&EntityKnowledge> {
        self.entity_by_eid.get(&eid).map(|&x| &self.entities[x])
    }

    pub fn get(&self, p: Point) -> CellResult<'_> {
        let cell_handle = self.cell_by_point.get(&p);
        CellResult { root: self, cell: cell_handle.map(|&x| &self.cells[x]) }
    }

    // Writes

    pub fn update(&mut self, me: &Entity, board: &Board, vision: &Vision, rng: &mut RNG) {
        self.time = board.time;
        let (pos, time) = (me.pos, self.time);
        let dark = matches!(board.get_light(), Light::None);

        // Clear and recompute scents. Only prey gives off a scent.
        self.scents.clear();
        self.update_scents(me, board, rng);

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

            let handle = (|| {
                if !see_big_entities { return None; }
                let other = board.get_entity(eid?)?;
                let big = other.player && !other.sneaking;
                if !big && !see_all_entities { return None; }
                let (seen, heard) = (true, false);
                Some(self.update_entity(me, other, seen, heard, time))
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
            cell.tile = tile;

            // Update the cell's flags.
            cell.shade = shade;
            cell.visible = visible;
            cell.see_entity_at = see_all_entities;

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

    pub fn update_entity(&mut self, me: &Entity, other: &Entity,
                         seen: bool, heard: bool, time: Timestamp) -> EntityHandle {
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
                pos: Default::default(),
                dir: Default::default(),
                glyph: Default::default(),
                name: Default::default(),
                time: Default::default(),

                // Stats:
                hp: Default::default(),
                pp: Default::default(),
                delta: Default::default(),

                // Flags:
                alive: Default::default(),
                heard: Default::default(),
                moved: Default::default(),
                asleep: Default::default(),
                friend: Default::default(),
                player: Default::default(),
                visible: Default::default(),
                sneaking: Default::default(),
            })
        });

        let entry = &mut self.entities[handle];

        entry.time = time;
        entry.pos = other.pos;
        entry.dir = other.dir;
        entry.glyph = other.glyph;

        entry.name = if other.player { "skishore" } else {
            if other.predator { "Rattata" } else { "Pidgey" }
        };
        entry.hp = other.cur_hp as f64 / max(other.max_hp, 1) as f64;
        entry.pp = 1. - clamp(other.move_timer as f64 / MOVE_TIMER as f64, 0., 1.);
        entry.delta = trophic_level(other) - trophic_level(me);

        entry.alive = other.cur_hp > 0;
        entry.heard = heard;
        entry.moved = !seen;
        entry.asleep = other.asleep;
        entry.friend = me.eid == other.eid;
        entry.player = other.player;
        entry.visible = seen;
        entry.sneaking = other.sneaking;

        handle
    }

    pub fn remove_entity(&mut self, oid: EID) {
        let Some(handle) = self.entity_by_eid.remove(&oid) else { return };

        let EntityKnowledge { moved, pos, .. } = self.entities.remove(handle);
        if moved { return; }

        let cell_handle = self.cell_by_point.get(&pos);
        let cell = &mut self.cells[*cell_handle.unwrap()];
        assert!(cell.handle == Some(handle));
        cell.handle = None;
    }

    // Private helpers:

    fn forget(&mut self, player: bool) {
        for entity in &mut self.entities {
            if !entity.heard { continue; }
            let Some(&h) = self.cell_by_point.get(&entity.pos) else { continue; };
            if self.cells[h].see_entity_at { entity.heard = false; }
        }

        if player {
            while let Some(x) = self.cells.back() && !x.visible {
                self.forget_last_cell();
            }
            return;
        }

        while self.cell_by_point.len() > MAX_TILE_MEMORY {
            // We don't need to check age, here; we can only see a bounded
            // number of cells per turn, much less than MAX_TILE_MEMORY.
            self.forget_last_cell();
        }

        while self.entity_by_eid.len() > MAX_ENTITY_MEMORY {
            let entity = self.entities.back().unwrap();
            if entity.visible { break; }

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

    fn update_scents(&mut self, me: &Entity, board: &Board, rng: &mut RNG) {
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
        let time = self.cell.map(|x| x.last_seen).unwrap_or_default();
        self.root.time - time
    }

    pub fn time_since_entity_visible(&self) -> Timedelta {
        let time = self.cell.map(|x| x.last_see_entity_at).unwrap_or_default();
        self.root.time - time
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
        if x.visible { x.visibility } else { -1 }
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
        self.cell.map(|x| x.visible).unwrap_or(false)
    }

    pub fn can_see_entity_at(&self) -> bool {
        self.cell.map(|x| x.see_entity_at).unwrap_or(false)
    }
}
