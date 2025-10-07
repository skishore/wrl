use std::cmp::{max, min};

use rand_distr::num_traits::Pow;

use crate::{composite, pri, seq};
use crate::base::{Point, RNG, Slice, dirs, weighted};
use crate::bhv::{Act, Bhv, Cond, Debug};
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

// SharedAIState

#[derive(Default)]
struct SharedAIState {}

//////////////////////////////////////////////////////////////////////////////

// Ctx

pub struct Ctx<'a, 'b> {
    // Derived from the entity.
    entity: &'a Entity,
    known: &'a Knowledge,
    pos: Point,
    dir: Point,
    // Computed by the executor during this turn.
    observations: Vec<&'a CellKnowledge>,
    neighborhood: Neighborhood,
    sneakable: Neighborhood,
    ran_vision: bool,
    shared: &'a mut SharedAIState,
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
    //if ctx.shared.threats.hostile.iter().any(
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

    // Copied from: ensure_neighborhood
    //
    // Need to port over CachedPath instead!

    ensure_vision(ctx);
    let (pos, check) = (ctx.pos, get_check(ctx));
    let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check)?;

    let step = *path.first()? - pos;
    Some(Action::Move(MoveAction { look: step, step, turns: WANDER_TURNS }))
}

//////////////////////////////////////////////////////////////////////////////

// Top-level tree

fn make_tree() -> Box<dyn Bhv> {
    Box::new(pri![
        "Root",
        Act::new("Explore", Explore),
    ])
}

//////////////////////////////////////////////////////////////////////////////

// Entry point

pub struct AIState {
    shared: SharedAIState,
    tree: Box<dyn Bhv>,
}

impl AIState {
    pub fn new(predator: bool, rng: &mut RNG) -> Self {
        Self { shared: SharedAIState::default(), tree: make_tree() }
    }

    pub fn get_path(&self) -> &[Point] { &[] }

    pub fn debug(&self, slice: &mut Slice) {
        let mut debug = Debug::default();
        self.tree.debug(&mut debug);
        for x in &debug.lines { slice.write_str(x).newline(); }
    }

    pub fn plan(&mut self, entity: &Entity, env: &mut AIEnv) -> Action {
        // Step 0: update some initial, deterministic shared state.
        let known = &*entity.known;
        let observations = known.cells.iter().take_while(|x| x.visible).collect();
        let shared = &mut self.shared;

        let mut ctx = Ctx {
            entity,
            known,
            pos: entity.pos,
            dir: entity.dir,
            action: None,
            observations,
            neighborhood: Default::default(),
            sneakable: Default::default(),
            ran_vision: false,
            shared,
            env,
        };
        self.tree.tick(&mut ctx);
        ctx.action.take().unwrap_or(Action::Idle)
    }
}
