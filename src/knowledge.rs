use std::cmp::max;
use std::f64::consts::TAU;

use crate::static_assert_size;
use crate::base::{FOV, FOVEndpoint, FOVNode, Glyph, HashMap, Matrix, Point};
use crate::game::{Board, EID, Entity, Light, Tile};
use crate::list::{Handle, List};
use crate::pathing::Status;

//////////////////////////////////////////////////////////////////////////////

// Constants

const MAX_ENTITY_MEMORY: usize = 64;
const MAX_TILE_MEMORY: usize = 4096;

const OBSCURED_VISION: i32 = 3;

// VISION_COSINE should be (0.5 * VISION_ANGLE).cos(), checked at runtime.
const VISION_ANGLE: f64 = TAU / 3.;
const VISION_COSINE: f64 = 0.5;
const VISION_RADIUS: i32 = 3;

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

    fn can_see_now(&self, p: Point) -> bool {
        self.get_visibility_at(p) >= 0
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
                if first { return 100 * (OBSCURED_VISION + 1) - 95 - 46 - 25; }

                let tile = f(p + pos);
                if tile.blocked() { return 0; }

                let parent = node.prev;
                let obscure = tile.obscure();
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
        for FOVEndpoint { pos, inv_l2 } in ends {
            let cos = (cone.pos.dot(*pos) as f64) * cone.inv_l2 * *inv_l2;
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
    pub point: Point,
    pub shade: bool,
    pub tile: &'static Tile,
    pub time: Timestamp,
    pub visibility: i32,
}
static_assert_size!(CellKnowledge, 32);

pub struct EntityKnowledge {
    pub eid: EID,
    pub age: i32,
    pub pos: Point,
    pub glyph: Glyph,
    pub moved: bool,
}

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

    pub fn age(&self) -> i32 {
        self.cell.map(|x| self.root.time - x.time).unwrap_or(std::i32::MAX)
    }

    pub fn shade(&self) -> bool {
        self.cell.map(|x| x.shade).unwrap_or(false)
    }

    pub fn tile(&self) -> Option<&'static Tile> {
        self.cell.map(|x| x.tile)
    }

    pub fn visibility(&self) -> i32 {
        self.cell.map(|x| x.visibility).unwrap_or(-1)
    }

    // Derived fields

    pub fn entity(&self) -> Option<&'a EntityKnowledge> {
        self.cell.and_then(|x| x.handle.map(|y| &self.root.entities[y]))
    }

    pub fn status(&self) -> Option<Status> {
        self.cell.map(|x| {
            if x.handle.is_some() { return Status::Occupied; }
            if x.tile.blocked() { Status::Blocked } else { Status::Free }
        })
    }

    // Predicates

    pub fn blocked(&self) -> bool {
        self.cell.map(|x| x.tile.blocked()).unwrap_or(false)
    }

    pub fn unblocked(&self) -> bool {
        self.cell.map(|x| !x.tile.blocked()).unwrap_or(false)
    }

    pub fn unknown(&self) -> bool {
        self.cell.is_none()
    }

    pub fn visible(&self) -> bool {
        self.cell.map(|x| self.root.time == x.time).unwrap_or(false)
    }
}

impl Knowledge {
    // Reads

    pub fn default(&self) -> CellResult {
        CellResult { root: self, cell: None }
    }

    pub fn entity(&self, eid: EID) -> Option<&EntityKnowledge> {
        self.entity_by_eid.get(&eid).map(|x| &self.entities[*x])
    }

    pub fn get(&self, p: Point) -> CellResult {
        let cell_handle = self.cell_by_point.get(&p);
        CellResult { root: self, cell: cell_handle.map(|x| &self.cells[*x]) }
    }

    // Writes

    pub fn update(&mut self, me: &Entity, board: &Board, vision: &Vision) {
        self.age_out();

        let (pos, time) = (me.pos, self.time);
        let (sun, dark) = match board.get_light() {
            Light::None => (Point::default(), true),
            Light::Sun(x) => (*x, false),
        };

        // Entities have exact knowledge about anything they can see.
        for point in &vision.points_seen {
            let point = *point;
            let visibility = vision.get_visibility_at(point);
            assert!(visibility >= 0);

            let cell = board.get_cell(point);
            let (eid, tile) = (cell.eid, cell.tile);

            let shadowed = cell.shadow > 0;
            let (delta, shade) = (point - pos, dark || shadowed);
            if (dark || (shadowed && !(tile.blocked() && sun.dot(delta) < 0))) &&
               delta.len_l1() > 1 {
                continue;
            }

            let handle = (|| {
                let other = board.get_entity(eid?)?;
                Some(self.update_entity(me, other, board, true))
            })();

            let mut prev_handle = None;
            let cell = CellKnowledge { handle, point, shade, tile, time, visibility };
            self.cell_by_point.entry(point).and_modify(|x| {
                self.cells.move_to_front(*x);
                prev_handle = std::mem::replace(&mut self.cells[*x], cell).handle;
            }).or_insert_with(|| {
                self.cells.push_front(cell)
            });

            if prev_handle != handle && let Some(other) = prev_handle {
                self.mark_entity_moved(other, point);
            };
        }

        self.forget(me.player);
    }

    pub fn update_entity(&mut self, _: &Entity, other: &Entity,
                         _: &Board, seen: bool) -> EntityHandle {
        let handle = *self.entity_by_eid.entry(other.eid).and_modify(|x| {
            self.entities.move_to_front(*x);
            let existing = &mut self.entities[*x];
            if !existing.moved && !(seen && existing.pos == other.pos) {
                let cell_handle = self.cell_by_point.get(&existing.pos);
                let cell = &mut self.cells[*cell_handle.unwrap()];
                assert!(cell.handle == Some(*x));
                cell.handle = None;
            };
        }).or_insert_with(|| {
            self.entities.push_front(EntityKnowledge {
                eid: other.eid,
                age: Default::default(),
                pos: Default::default(),
                moved: Default::default(),
                glyph: Default::default(),
            })
        });

        let entry = &mut self.entities[handle];

        entry.age = if seen { 0 } else { 1 };
        entry.pos = other.pos;
        entry.moved = !seen;
        entry.glyph = other.glyph;

        handle
    }

    // Private helpers

    fn age_out(&mut self) {
        for x in &mut self.entities { x.age += 1; }
        self.time.0 += 1;
    }

    fn forget(&mut self, player: bool) {
        if player {
            while let Some(x) = self.cells.back() && x.time != self.time {
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
