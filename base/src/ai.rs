use std::cmp::{max, min};
use std::f64::consts::TAU;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::{act, cb, pri, seq};
use crate::base::{Point, RNG, Slice, dirs, weighted};
use crate::bhv::{Bhv, Debug, Result};
use crate::entity::{Entity, EID};
use crate::game::{Action, AttackAction, CallForHelpAction, EatAction, MoveAction};
use crate::knowledge::{CellKnowledge, Knowledge, EntityKnowledge};
use crate::pathing::{AStar, BFS, DijkstraLength, DijkstraMap, Neighborhood, Status};
use crate::shadowcast::{INITIAL_VISIBILITY, Vision, VisionArgs};

//////////////////////////////////////////////////////////////////////////////

// Constants

const ASTAR_LIMIT_ATTACK: i32 = 256;
const ASTAR_LIMIT_WANDER: i32 = 1024;
const BFS_LIMIT_ATTACK: i32 = 8;
const HIDING_CELLS: i32 = 256;
const HIDING_LIMIT: i32 = 32;
const SEARCH_CELLS: i32 = 1024;
const SEARCH_LIMIT: i32 = 64;

const ASSESS_ANGLE: f64 = TAU / 18.;
const ASSESS_STEPS: i32 = 4;
const ASSESS_TURNS_EXPLORE: i32 = 8;
const ASSESS_TURNS_FLIGHT: i32 = 1;

const MAX_ASSESS: i32 = 32;
const MAX_RESTED: i32 = 4096;
const MAX_THIRST: i32 = 256;
const MAX_HUNGER_HERBIVORE: i32 = 1024;
const MAX_HUNGER_CARNIVORE: i32 = 2048;

const WANDER_TURNS: f64 = 2.;

//////////////////////////////////////////////////////////////////////////////

// Interface

#[derive(Default)]
pub struct AIDebug {
    pub targets: Vec<Point>,
    pub utility: Vec<(Point, u8)>,
}

pub struct AIEnv<'a> {
    pub debug: Option<&'a mut AIDebug>,
    pub fov: &'a mut Vision,
    pub rng: &'a mut RNG,
}

//////////////////////////////////////////////////////////////////////////////

// Blackboard

struct Blackboard {
    dirs: Vec<Point>,
    path: CachedPath,
    till_assess: i32,
}

impl Blackboard {
    fn new(predator: bool, rng: &mut RNG) -> Self {
        Self {
            dirs: Default::default(),
            path: Default::default(),
            till_assess: rng.random_range(0..MAX_ASSESS),
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Ctx

pub struct Ctx<'a, 'b> {
    // Derived from the entity.
    entity: &'a Entity,
    known: &'a Knowledge,
    pos: Point,
    dir: Point,
    // Computed by the executor during this turn.
    blackboard: &'a mut Blackboard,
    observations: Vec<&'a CellKnowledge>,
    neighborhood: Neighborhood,
    sneakable: Neighborhood,
    ran_vision: bool,
    // Mutable access to the RNG.
    env: &'a mut AIEnv<'b>,
    // The tree's output; written by Act.
    pub action: Option<Action>,
}

fn safe_inv_l2(point: Point) -> f64 {
    if point == Point::default() { return 0. }
    (point.len_l2_squared() as f64).sqrt().recip()
}

fn is_hiding_place(ctx: &Ctx, point: Point) -> bool {
    //if ctx.blackboard.threats.hostile.iter().any(
    //    |x| (x.pos - point).len_l1() <= 1) { return false; }
    let cell = ctx.known.get(point);
    cell.shade() || matches!(cell.tile(), Some(x) if x.limits_vision())
}

fn get_check<'a>(ctx: &'a Ctx) -> impl Fn(Point) -> Status + use<'a> {
    let (fov, known, pos) = (&ctx.env.fov, ctx.known, ctx.pos);
    move |p: Point| {
        let status = known.get(p).status();
        if status != Status::Unknown { return status; }
        if fov.can_see(p - pos) { Status::Free } else { Status::Unknown }
    }
}

fn ensure_neighborhood(ctx: &mut Ctx) {
    if !ctx.neighborhood.visited.is_empty() { return; }

    ensure_vision(ctx);
    let (pos, check) = (ctx.pos, get_check(ctx));
    ctx.neighborhood = DijkstraMap(pos, check, SEARCH_CELLS, SEARCH_LIMIT);
}

fn ensure_vision(ctx: &mut Ctx) {
    if ctx.ran_vision { return; }

    let Ctx { known, pos, .. } = *ctx;
    let opacity_lookup = |p: Point| {
        let blocked = known.get(p + pos).status() == Status::Blocked;
        if blocked { INITIAL_VISIBILITY } else { 0 }
    };
    let origin = Point::default();
    let args = VisionArgs { pos: origin, dir: origin, opacity_lookup, };

    ctx.env.fov.compute(&args);
    ctx.ran_vision = true;
}

//////////////////////////////////////////////////////////////////////////////

// Uncategorized helpers:

fn assess_directions(dirs: &[Point], turns: i32, rng: &mut RNG) -> Vec<Point> {
    if dirs.is_empty() { return vec![]; }

    let mut result = vec![];
    result.reserve((ASSESS_STEPS * turns) as usize);

    for i in 0..ASSESS_STEPS {
        let dir = dirs[i as usize % dirs.len()];
        if dir == Point::default() { continue; }

        let scale = 100. / dir.len_l2();
        let steps = rng.random_range(0..turns) + 1;
        let angle = Normal::new(0., ASSESS_ANGLE).unwrap().sample(rng);
        let (sin, cos) = (angle.sin(), angle.cos());

        let Point(dx, dy) = dir;
        let rx = (cos * scale * dx as f64) + (sin * scale * dy as f64);
        let ry = (cos * scale * dy as f64) - (sin * scale * dx as f64);
        let target = Point(rx as i32, ry as i32);
        for _ in 0..steps { result.push(target); }
    }

    result.reverse();
    result
}

fn select_target(scores: &[(Point, f64)], env: &mut AIEnv) -> Option<Point> {
    let max = scores.iter().fold(
        0., |acc, x| if acc > x.1 { acc } else { x.1 });
    if max == 0. { return None; }

    let limit = (1 << 16) - 1;
    let inverse = (limit as f64) / max;
    let values: Vec<_> = scores.iter().filter_map(|&(p, score)| {
        let score = min((inverse * score).floor() as i32, limit);
        if score > 0 { Some((score, p)) } else { None }
    }).collect();
    if values.is_empty() { return None; }

    if let Some(x) = env.debug.as_deref_mut() {
        x.utility.clear();
        for &(score, point) in &values {
            x.utility.push((point, (score >> 8) as u8));
        }
    }
    Some(*weighted(&values, env.rng))
}

//////////////////////////////////////////////////////////////////////////////

// Ticking counters

#[allow(non_snake_case)]
fn TickBasicNeeds(ctx: &mut Ctx) -> Result {
    let (bb, entity) = (&mut *ctx.blackboard, ctx.entity);
    if !entity.asleep {
        bb.till_assess = max(bb.till_assess - 1, 0);
    }
    Result::Unused
}

//////////////////////////////////////////////////////////////////////////////

// Assess

#[allow(non_snake_case)]
fn FollowDirs(ctx: &mut Ctx) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    let dir = bb.dirs.pop()?;

    if bb.dirs.is_empty() { bb.till_assess = rng.gen_range(0..MAX_ASSESS); }
    Some(Action::Look(dir))
}

#[allow(non_snake_case)]
fn Assess(ctx: &mut Ctx) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    if bb.till_assess > 0 { return None; }

    bb.dirs = assess_directions(&[ctx.dir], ASSESS_TURNS_EXPLORE, rng);
    FollowDirs(ctx)
}

//////////////////////////////////////////////////////////////////////////////

// Explore

#[allow(non_snake_case)]
fn Explore(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, dir, .. } = *ctx;
    let inv_dir_l2 = safe_inv_l2(dir);

    ensure_neighborhood(ctx);

    let score = |p: Point, distance: i32| -> f64 {
        if distance == 0 { return 0.; }

        let age = known.get(p).time_since_seen().seconds();
        let age_scale = 1. / (1 << 24) as f64;

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let unblocked_neighbors = dirs::ALL.iter().filter(
                |&&x| !known.get(p + x).blocked()).count();

        let bonus0 = age_scale * (age as f64 + 1. / 16.);
        let bonus1 = unblocked_neighbors == dirs::ALL.len();
        let bonus2 = unblocked_neighbors > 0;

        let base = (if bonus0 > 1. { 1. } else { bonus0 }) *
                   (if bonus1 {  8.0 } else { 1.0 }) *
                   (if bonus2 { 64.0 } else { 1.0 });
        base * (cos + 1.).pow(4) / (distance as f64).pow(2)
    };

    let scores: Vec<_> = ctx.neighborhood.visited.iter().map(
        |&(p, distance)| (p, score(p, distance))).collect();
    let target = select_target(&scores, ctx.env)?;
    FindPath(ctx, target)
}

//////////////////////////////////////////////////////////////////////////////

// Pathfinding

#[derive(Default)]
struct CachedPath {
    path: Vec<Point>,
    step: usize,
}

#[allow(non_snake_case)]
fn FindPath(ctx: &mut Ctx, target: Point) -> Option<Action> {
    ensure_vision(ctx);
    let (pos, check) = (ctx.pos, get_check(ctx));
    let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check)?;
    ctx.blackboard.path = CachedPath { path, step: 0 };
    ctx.blackboard.path.path.insert(0, pos);
    FollowPath(ctx)
}

#[allow(non_snake_case)]
fn FollowPath(ctx: &mut Ctx) -> Option<Action> {
    let (known, pos) = (ctx.known, ctx.pos);
    let path = std::mem::take(&mut ctx.blackboard.path);
    if path.path.is_empty() { return None; }

    let valid = (||{
        let (i, j) = (path.step, path.step + 1);
        let Some(&prev) = path.path.get(i) else { return false };
        let Some(&next) = path.path.get(j) else { return false };
        if prev != ctx.pos { return false };

        let valid = |p: Point| match known.get(p).status() {
            Status::Free | Status::Unknown => true,
            Status::Occupied => p != next,
            Status::Blocked => false,
        };
        path.path.iter().skip(j).all(|&x| valid(x))
    })();
    if !valid { return None; }

    let next = path.path[path.step + 1];
    let step = next - pos;
    let look = step;
    ctx.blackboard.path = path;
    ctx.blackboard.path.step += 1;
    Some(Action::Move(MoveAction { look, step, turns: WANDER_TURNS }))
}

//////////////////////////////////////////////////////////////////////////////

// Top-level tree

fn make_tree() -> Box<dyn Bhv> {
    Box::new(pri![
        "Root",
        cb!("TickBasicNeeds", TickBasicNeeds),
        act!("FollowDirs", FollowDirs),
        act!("FollowPath", FollowPath),
        act!("Assess", Assess),
        act!("Explore", Explore),
    ])
}

//////////////////////////////////////////////////////////////////////////////

// Entry point

pub struct AIState {
    blackboard: Blackboard,
    tree: Box<dyn Bhv>,
}

impl AIState {
    pub fn new(predator: bool, rng: &mut RNG) -> Self {
        let blackboard = Blackboard::new(predator, rng);
        Self { blackboard, tree: make_tree() }
    }

    pub fn get_path(&self) -> &[Point] {
        &self.blackboard.path.path
    }

    pub fn debug(&self, slice: &mut Slice) {
        let mut debug = Debug::default();
        self.tree.debug(&mut debug);
        for x in &debug.lines { slice.write_str(x).newline(); }
        slice.newline();
        let bb = &self.blackboard;
        slice.write_str(&format!("till_assess: {}", bb.till_assess)).newline();
        slice.write_str(&format!("dirs: {} items", bb.dirs.len())).newline();
    }

    pub fn plan(&mut self, entity: &Entity, env: &mut AIEnv) -> Action {
        // Step 0: update some initial, deterministic shared state.
        let known = &*entity.known;
        let observations = known.cells.iter().take_while(|x| x.visible).collect();
        let blackboard = &mut self.blackboard;

        let mut ctx = Ctx {
            entity,
            known,
            pos: entity.pos,
            dir: entity.dir,
            action: None,
            blackboard,
            observations,
            neighborhood: Default::default(),
            sneakable: Default::default(),
            ran_vision: false,
            env,
        };
        self.tree.tick(&mut ctx);
        ctx.action.take().unwrap_or(Action::Idle)
    }
}
