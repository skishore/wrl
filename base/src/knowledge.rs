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

// Timestamp

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Timestamp(u32);

impl std::ops::Sub for Timestamp {
    type Output = i32;
    fn sub(self, other: Timestamp) -> Self::Output {
        self.0.wrapping_sub(other.0) as Self::Output
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
static_assert_size!(CellKnowledge, 36);
#[cfg(target_pointer_width = "64")]
static_assert_size!(CellKnowledge, 48);

pub struct EntityKnowledge {
    pub eid: EID,
    pub age: i32,
    pub pos: Point,
    pub dir: Point,
    pub glyph: Glyph,
    pub name: &'static str,

    // Stats:
    pub hp: f64,
    pub pp: f64,

    // Flags:
    pub alive: bool,
    pub heard: bool,
    pub moved: bool,
    pub rival: bool,
    pub friend: bool,
    pub asleep: bool,
    pub player: bool,
    pub visible: bool,
    pub sneaking: bool,
}
#[cfg(target_pointer_width = "32")]
static_assert_size!(EntityKnowledge, 72);
#[cfg(target_pointer_width = "64")]
static_assert_size!(EntityKnowledge, 80);

#[derive(Clone, Copy)]
pub struct Scent {
    pub age: i32,
    pub pos: Point,
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

    pub fn start_next_turn(&mut self, player: bool) {
        self.time.0 += 1;
        for x in &mut self.entities { x.age += 1; }
        if !player { return; }

        while let Some(x) = self.cells.back() && !x.visible {
            self.forget_last_cell();
        }
    }

    pub fn update(&mut self, me: &Entity, board: &Board, vision: &Vision, rng: &mut RNG) {
        let (pos, time) = (me.pos, self.time);
        let dark = matches!(board.get_light(), Light::None);

        // Clear and recompute scents. Only the player gives off a scent.
        self.scents.clear();
        for (oid, other) in &board.entities {
            if oid == me.eid || !other.player { continue; }
            let mut remainder = rng.gen::<f64>();
            for age in 0..other.history.capacity() {
                remainder -= other.get_historical_scent_at(me.pos, age);
                if remainder >= 0. { continue; }
                self.scents.push(Scent { age: age as i32, pos: other.history[age] });
                break;
            }
        }

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
                glyph: Default::default(),
                name: Default::default(),

                // Stats:
                hp: Default::default(),
                pp: Default::default(),

                // Flags:
                alive: Default::default(),
                heard: Default::default(),
                moved: Default::default(),
                rival: Default::default(),
                friend: Default::default(),
                asleep: Default::default(),
                player: Default::default(),
                visible: Default::default(),
                sneaking: Default::default(),
            })
        });

        let same = other.eid == entity.eid;
        let entry = &mut self.entities[handle];
        let aggressor = |x: &Entity| x.predator;
        let rival = !same && (aggressor(entity) != aggressor(other));

        entry.age = 0;
        entry.pos = other.pos;
        entry.dir = other.dir;
        entry.glyph = other.glyph;

        entry.name = if other.player { "skishore" } else {
            if other.predator { "Rattata" } else { "Pidgey" }
        };
        entry.hp = other.cur_hp as f64 / max(other.max_hp, 1) as f64;
        entry.pp = 1. - clamp(other.move_timer as f64 / MOVE_TIMER as f64, 0., 1.);

        entry.alive = other.cur_hp > 0;
        entry.heard = heard;
        entry.moved = !seen;
        entry.rival = rival;
        entry.friend = same;
        entry.asleep = other.asleep;
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
        if player { return; }

        while self.cell_by_point.len() > MAX_TILE_MEMORY {
            // We don't need to check age, here; we can only see a bounded
            // number of cells per turn, much less than MAX_TILE_MEMORY.
            self.forget_last_cell();
        }

        while self.entity_by_eid.len() > MAX_ENTITY_MEMORY {
            let entity = self.entities.back().unwrap();
            if entity.age == 0 { break; }

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

    pub fn turns_since_seen(&self) -> i32 {
        let Some(x) = self.cell else { return std::i32::MAX };
        self.root.time - x.last_seen
    }

    pub fn turns_since_entity_visible(&self) -> i32 {
        let Some(x) = self.cell else { return std::i32::MAX };
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
