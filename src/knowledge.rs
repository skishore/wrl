use std::cmp::max;
use std::f64::consts::TAU;

use crate::static_assert_size;
use crate::base::{FOV, FOVEndpoint, FOVNode, Glyph, HashMap, Matrix, Point};
use crate::entity::{EID, Entity};
use crate::game::{Board, Light, Tile};
use crate::list::{Handle, List};
use crate::pathing::Status;

//////////////////////////////////////////////////////////////////////////////

// Constants

const TURN_AGE: i32 = 2;

const MAX_ENTITY_MEMORY: usize = 64;
const MAX_TILE_MEMORY: usize = 4096;

// VISION_COSINE should be (0.5 * VISION_ANGLE).cos(), checked at runtime.
const VISION_ANGLE: f64 = TAU / 3.;
const VISION_COSINE: f64 = 0.5;
const _PC_VISION_RADIUS: i32 = 4;
const NPC_VISION_RADIUS: i32 = 3;

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

// Vision

pub struct VisionArgs {
    pub player: bool,
    pub pos: Point,
    pub dir: Point,
}

pub struct Vision {
    fov: FOV,
    center: Point,
    offset: Point,
    visibility: Matrix<i32>,
    points_seen: Vec<Point>,
}

impl Vision {
    pub fn new(radius: i32) -> Self {
        assert!((VISION_COSINE - (0.5 * VISION_ANGLE).cos()).abs() < 0.01);
        let vision_side = 2 * radius + 1;
        let vision_size = Point(vision_side, vision_side);
        Self {
            fov: FOV::new(radius),
            center: Point(radius, radius),
            offset: Point::default(),
            visibility: Matrix::new(vision_size, -1),
            points_seen: vec![],
        }
    }

    fn get_visibility_at(&self, p: Point) -> i32 {
        self.visibility.get(p + self.offset)
    }

    pub fn compute<F: Fn(Point) -> &'static Tile>(&mut self, args: &VisionArgs, f: F) {
        let VisionArgs { player, pos, dir } = *args;
        let inv_l2 = if !player && dir != Point::default() {
            1. / (dir.len_l2_squared() as f64).sqrt()
        } else {
            1.
        };
        let cone = FOVEndpoint { pos: dir, inv_l2 };
        let radius = if player { _PC_VISION_RADIUS } else { NPC_VISION_RADIUS };

        self.offset = self.center - pos;
        self.visibility.fill(-1);
        self.points_seen.clear();

        let blocked = |node: &FOVNode| {
            let p = node.next;
            let first = p == Point::default();
            if !player && !first && !Self::include_ray(&cone, &node.ends) { return true; }

            let lookup = p + self.center;
            let cached = self.visibility.get(lookup);

            let visibility = (|| {
                // These constant values come from Point.distanceNethack.
                // They are chosen such that, in a field of tall grass, we'll
                // only see cells at a distanceNethack <= kVisionRadius.
                if first { return 100 * (radius + 1) - 95 - 46 - 25; }

                let tile = f(p + pos);
                if tile.blocks_vision() { return 0; }

                let parent = node.prev;
                let obscure = tile.limits_vision();
                let diagonal = p.0 != parent.0 && p.1 != parent.1;
                let loss = if obscure { 95 + if diagonal { 46 } else { 0 } } else { 0 };
                let prev = self.visibility.get(parent + self.center);
                max(prev - loss, 0)
            })();

            if visibility > cached {
                self.visibility.set(lookup, visibility);
                if cached < 0 && 0 <= visibility {
                    self.points_seen.push(p + pos);
                }
            }
            visibility <= 0
        };
        self.fov.apply(blocked);
    }

    fn include_ray(cone: &FOVEndpoint, ends: &[FOVEndpoint]) -> bool {
        if cone.pos == Point::default() { return true; }
        for &FOVEndpoint { pos, inv_l2 } in ends {
            let cos = (cone.pos.dot(pos) as f64) * cone.inv_l2 * inv_l2;
            if cos >= VISION_COSINE { return true; }
        }
        false
    }
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

type CellHandle = Handle<CellKnowledge>;
type EntityHandle = Handle<EntityKnowledge>;

#[derive(Clone, Copy)]
pub struct CellKnowledge {
    handle: Option<EntityHandle>,
    pub last_see_entity_at: Timestamp,
    pub last_seen: Timestamp,
    pub point: Point,
    pub shade: bool,
    pub tile: &'static Tile,
    visibility: i32,
}
static_assert_size!(CellKnowledge, 40);

pub struct EntityKnowledge {
    pub eid: EID,
    pub age: i32,
    pub pos: Point,
    pub dir: Point,
    pub glyph: Glyph,
    pub alive: bool,
    pub heard: bool,
    pub moved: bool,
    pub rival: bool,
    pub friend: bool,
}
static_assert_size!(EntityKnowledge, 48);

#[derive(Default)]
pub struct Knowledge {
    cell_by_point: HashMap<Point, CellHandle>,
    entity_by_eid: HashMap<EID, EntityHandle>,
    pub cells: List<CellKnowledge>,
    pub entities: List<EntityKnowledge>,
    pub focus: Option<EID>,
    pub time: Timestamp,
}

pub struct CellResult<'a> {
    root: &'a Knowledge,
    cell: Option<&'a CellKnowledge>,
}

impl<'a> CellResult<'a> {
    // Field lookups

    pub fn time_since_seen(&self) -> i32 {
        let time = self.cell.map(|x| x.last_seen).unwrap_or_default();
        if time == Default::default() { std::i32::MAX } else { self.root.time - time }
    }

    pub fn time_since_entity_visible(&self) -> i32 {
        let time = self.cell.map(|x| x.last_see_entity_at).unwrap_or_default();
        if time == Default::default() { std::i32::MAX } else { self.root.time - time }
    }

    pub fn shade(&self) -> bool {
        self.cell.map(|x| x.shade).unwrap_or(false)
    }

    pub fn tile(&self) -> Option<&'static Tile> {
        self.cell.map(|x| x.tile)
    }

    pub fn visibility(&self) -> i32 {
        let Some(x) = self.cell else { return -1 };
        if x.last_seen == self.root.time { x.visibility } else { -1 }
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
        self.cell.map(|x| x.last_seen == self.root.time).unwrap_or(false)
    }

    pub fn can_see_entity_at(&self) -> bool {
        self.cell.map(|x| x.last_see_entity_at == self.root.time).unwrap_or(false)
    }
}

impl CellKnowledge {
    fn new(point: Point, tile: &'static Tile) -> Self {
        let (handle, shade, visibility) = (None, false, -1);
        let (last_seen, last_see_entity_at) = (Default::default(), Default::default());
        Self { handle, last_seen, last_see_entity_at, point, shade, tile, visibility }
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

    pub fn update(&mut self, me: &Entity, board: &Board, vision: &Vision) {
        self.age_out();

        let (pos, time) = (me.pos, self.time);
        let dark = matches!(board.get_light(), Light::None);

        // Entities have exact knowledge about anything they can see.
        for &point in &vision.points_seen {
            let visibility = vision.get_visibility_at(point);
            assert!(visibility >= 0);

            let cell = board.get_cell(point);
            let (eid, tile) = (cell.eid, cell.tile);

            let nearby = (point - pos).len_l1() <= 1;
            if dark && !nearby { continue; }

            let shade = dark || cell.shadow > 0;
            let see_entity_at = nearby || !(shade || tile.limits_vision());

            let handle = (|| {
                if !see_entity_at { return None; }
                let other = board.get_entity(eid?)?;
                let (seen, heard) = (true, false);
                Some(self.update_entity(me, other, board, seen, heard))
            })();

            let cell_handle = *self.cell_by_point.entry(point).and_modify(|&mut x| {
                self.cells.move_to_front(x);
            }).or_insert_with(|| {
                self.cells.push_front(CellKnowledge::new(point, tile))
            });

            let cell = &mut self.cells[cell_handle];
            let prev_handle = std::mem::replace(&mut cell.handle, handle);

            if see_entity_at { cell.last_see_entity_at = time; }

            cell.last_seen = time;
            cell.point = point;
            cell.shade = shade;
            cell.tile = tile;

            if prev_handle != handle && let Some(other) = prev_handle {
                self.mark_entity_moved(other, point);
            };
        }

        self.forget(me.player);
    }

    pub fn update_entity(&mut self, entity: &Entity, other: &Entity,
                         _: &Board, seen: bool, heard: bool) -> EntityHandle {
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
                alive: Default::default(),
                glyph: Default::default(),
                heard: Default::default(),
                moved: Default::default(),
                rival: Default::default(),
                friend: Default::default(),
            })
        });

        let same = other.eid == entity.eid;
        let entry = &mut self.entities[handle];
        let aggressor = |x: &Entity| x.player || x.predator;
        let rival = !same && (aggressor(entity) != aggressor(other));

        // It seems strange that we would assign a future age to an entity
        // that when we learn about without seeing it. The reason we do so is
        // because that update happens off-turn, so we must record it with an
        // age between our previous and next turns'.
        entry.age = if seen { 0 } else { -1 };
        entry.pos = other.pos;
        entry.dir = other.dir;
        entry.alive = other.cur_hp > 0;
        entry.glyph = other.glyph;
        entry.heard = heard;
        entry.moved = !seen;
        entry.rival = rival;
        entry.friend = same;

        handle
    }

    // Private helpers

    fn age_out(&mut self) {
        for x in &mut self.entities { x.age += TURN_AGE; }
        self.time.0 += TURN_AGE as u32;
    }

    fn forget(&mut self, player: bool) {
        for entity in &mut self.entities {
            if !entity.heard { continue; }
            let lookup = self.cell_by_point.get(&entity.pos);
            let Some(&h) = lookup else { continue; };
            if self.cells[h].last_see_entity_at != self.time { continue; }
            entity.heard = false;
        }

        if player {
            while let Some(x) = self.cells.back() && x.last_seen != self.time {
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
