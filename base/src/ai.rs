use std::cmp::{max, min};
use std::f64::consts::TAU;
use std::ops::RangeInclusive;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::{act, cb, cond, pri, seq, util};
use crate::base::{LOS, Point, RNG, Slice, dirs, sample, weighted};
use crate::bhv::{Bhv, Debug, Result};
use crate::entity::{Entity, EID};
use crate::game::{MOVE_VOLUME, Item, move_ready};
use crate::game::{Action, AttackAction, CallForHelpAction, EatAction, MoveAction};
use crate::knowledge::{CellKnowledge, Knowledge, EntityKnowledge, Timedelta, Timestamp};
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
const MAX_HUNGER: i32 = 512;
const MAX_THIRST: i32 = 128;
const MAX_WEARY_: i32 = 2048;

const ASSESS_GAIN: RangeInclusive<i32> = (MAX_ASSESS / 2)..=MAX_ASSESS;
const HUNGER_GAIN: RangeInclusive<i32> = (MAX_HUNGER / 4)..=(MAX_HUNGER / 2);
const THIRST_GAIN: RangeInclusive<i32> = (MAX_THIRST / 4)..=(MAX_THIRST / 2);
const RESTED_GAIN: RangeInclusive<i32> = 1..=2;

const MIN_SEARCH_TIME: Timedelta = Timedelta::from_seconds(24.);
const MAX_SEARCH_TIME: Timedelta = Timedelta::from_seconds(48.);

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
    target: Option<ChaseTarget>,

    prev_time: Timestamp,
    turn_time: Timestamp,

    // Basic needs:
    till_assess: i32,
    till_hunger: i32,
    till_thirst: i32,
    till_weary_: i32,
    finding_food_: bool,
    finding_water: bool,
    getting_rest_: bool,
    hunger: bool,
    thirst: bool,
    weary_: bool,
}

impl Blackboard {
    fn new(predator: bool, rng: &mut RNG) -> Self {
        Self {
            dirs: Default::default(),
            path: Default::default(),
            target: None,

            prev_time: Timestamp::default(),
            turn_time: Timestamp::default(),

            // Basic needs:
            till_assess: rng.random_range(0..MAX_ASSESS),
            till_hunger: rng.random_range(0..MAX_HUNGER),
            till_thirst: rng.random_range(0..MAX_THIRST),
            till_weary_: rng.random_range(0..MAX_WEARY_),
            finding_food_: false,
            finding_water: false,
            getting_rest_: false,
            hunger: false,
            thirst: false,
            weary_: false,
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
    move |p: Point| match known.get(p).status() {
        Status::Occupied if (p - pos).len_l1() == 1 => Status::Blocked,
        Status::Unknown if fov.can_see(p - pos) => Status::Free,
        x => x,
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

fn get_explore_target(ctx: &mut Ctx) -> Option<Point> {
    let Ctx { known, pos, dir, .. } = *ctx;
    let inv_dir_l2 = safe_inv_l2(dir);

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

    ensure_neighborhood(ctx);

    let scores: Vec<_> = ctx.neighborhood.visited.iter().map(
        |&(p, distance)| (p, score(p, distance))).collect();
    select_target(&scores, ctx.env)
}

fn get_chase_target(ctx: &mut Ctx, age: Timedelta, bias: Point, center: Point) -> Option<Point> {
    let Ctx { known, pos, dir, .. } = *ctx;
    let inv_dir_l2 = safe_inv_l2(dir);
    let inv_bias_l2 = safe_inv_l2(bias);

    let is_search_candidate = |p: Point| {
        let cell = known.get(p);
        !cell.blocked() && cell.time_since_entity_visible() > age
    };
    if center != pos && is_search_candidate(center) { return Some(center); }

    let score = |p: Point, distance: i32| -> f64 {
        if distance == 0 || !is_search_candidate(p) { return 0.; }

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos0 = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let cos1 = delta.dot(bias) as f64 * inv_delta_l2 * inv_bias_l2;
        let angle = ((cos0 + 1.) * (cos1 + 1.)).pow(4);

        angle / (((p - center).len_l2_squared() + 1) as f64).pow(2)
    };

    ensure_neighborhood(ctx);

    let scores: Vec<_> = ctx.neighborhood.visited.iter().map(
        |&(p, distance)| (p, score(p, distance))).collect();
    select_target(&scores, ctx.env)
}

//////////////////////////////////////////////////////////////////////////////

// Ticking counters

#[allow(non_snake_case)]
fn TickBasicNeeds(ctx: &mut Ctx) -> Result {
    let (bb, entity) = (&mut *ctx.blackboard, ctx.entity);
    bb.prev_time = bb.turn_time;
    bb.turn_time = ctx.entity.known.time;

    if !entity.asleep {
        bb.till_assess = max(bb.till_assess - 1, 0);
        bb.till_hunger = max(bb.till_hunger - 1, 0);
        bb.till_thirst = max(bb.till_thirst - 1, 0);
        bb.till_weary_ = max(bb.till_weary_ - 1, 0);
    }

    if bb.till_hunger < MAX_HUNGER / 8 { bb.hunger = true; }
    if bb.till_thirst < MAX_THIRST / 8 { bb.thirst = true; }
    if bb.till_weary_ < MAX_WEARY_ / 8 { bb.weary_ = true; }
    if bb.till_hunger >= 7 * MAX_HUNGER / 8 { bb.hunger = false; }
    if bb.till_thirst >= 7 * MAX_THIRST / 8 { bb.thirst = false; }
    if bb.till_weary_ >= 7 * MAX_WEARY_ / 8 { bb.weary_ = false; }

    Result::Failed
}

//////////////////////////////////////////////////////////////////////////////

// Assess

#[allow(non_snake_case)]
fn FollowDirs(ctx: &mut Ctx) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    let dir = bb.dirs.pop()?;

    if bb.dirs.is_empty() {
        let gain = rng.gen_range(ASSESS_GAIN);
        bb.till_assess = min(bb.till_assess + gain, MAX_ASSESS);
    }
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
    let kind = PathKind::Explore;
    let target = get_explore_target(ctx)?;
    if FindPath(ctx, target, kind) { FollowPath(ctx, kind) } else { None }
}

//////////////////////////////////////////////////////////////////////////////

// Pathfinding

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PathKind { Prey, Meat, Rest, Water, Berry, BerryTree, Explore, #[default] None }

#[derive(Default)]
struct CachedPath {
    kind: PathKind,
    path: Vec<Point>,
    skip: usize,
    step: usize,
}

#[allow(non_snake_case)]
fn FindPath(ctx: &mut Ctx, target: Point, kind: PathKind) -> bool {
    ensure_vision(ctx);
    let (pos, check) = (ctx.pos, get_check(ctx));
    let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check);
    let Some(path) = path else { return false };

    let skip = match kind {
        PathKind::Meat | PathKind::Water | PathKind::Berry | PathKind::BerryTree => 1,
        PathKind::Prey | PathKind::Rest | PathKind::Explore | PathKind::None => 0,
    };
    ctx.blackboard.path = CachedPath { kind, path, skip, step: 0 };
    ctx.blackboard.path.path.insert(0, pos);
    true
}

#[allow(non_snake_case)]
fn FollowPath(ctx: &mut Ctx, kind: PathKind) -> Option<Action> {
    if ctx.blackboard.path.kind != kind { return None; }

    let (known, pos) = (ctx.known, ctx.pos);
    let path = std::mem::take(&mut ctx.blackboard.path);
    if path.path.is_empty() { return None; }

    let (i, j) = (path.step, path.step + 1);
    let valid = (||{
        let Some(&prev) = path.path.get(i) else { return false };
        let Some(&next) = path.path.get(j) else { return false };
        if prev != ctx.pos { return false };

        let valid = |p: Point| match known.get(p).status() {
            Status::Free | Status::Unknown => true,
            Status::Occupied => p != next,
            Status::Blocked => false,
        };
        path.path.iter().skip(j).rev().skip(path.skip).rev().all(|&x| valid(x))
    })();
    if !valid { return None; }

    let next = path.path[j];
    let mut target = next;
    for &point in path.path.iter().skip(j).take(8) {
        let los = LOS(pos, point);
        if los.iter().all(|&x| !known.get(x).blocked()) { target = point; }
    }
    let (look, step) = (target - pos, next - pos);

    let mut turns = WANDER_TURNS;
    if kind == PathKind::Prey && let Some(x) = &ctx.blackboard.target {
        let age = ctx.known.time - x.time;
        if age < MIN_SEARCH_TIME { turns = 1. };
    }

    if path.step + 2 < path.path.len() {
        ctx.blackboard.path = path;
        ctx.blackboard.path.step += 1;
    }
    Some(Action::Move(MoveAction { look, step, turns }))
}

//////////////////////////////////////////////////////////////////////////////

#[allow(non_snake_case)]
fn AttackPathTarget(ctx: &mut Ctx, kind: PathKind) -> Option<Action> {
    if ctx.blackboard.path.kind != kind { return None; }
    let target = *ctx.blackboard.path.path.last()?;
    AttackTarget(ctx, target)
}

#[allow(non_snake_case)]
fn AttackTarget(ctx: &mut Ctx, target: Point) -> Option<Action> {
    let Ctx { entity, known, pos: source, .. } = *ctx;
    if !known.get(target).visible() { return None; }
    if source == target { return None; }

    let attacks = &ctx.entity.species.attacks;
    if attacks.is_empty() { return None; }

    let attack = sample(attacks, ctx.env.rng);
    let valid = |x| CanAttackTarget(x, target, known, attack.range);
    if move_ready(entity) && valid(source) {
        return Some(Action::Attack(AttackAction { attack, target }));
    }
    PathToTarget(ctx, target, attack.range, valid)
}

#[allow(non_snake_case)]
fn CanAttackTarget(source: Point, target: Point, known: &Knowledge, range: i32) -> bool {
    if (source - target).len_nethack() > range { return false; }
    if source == target { return false; }

    let los = LOS(source, target);
    los.iter().skip(1).rev().skip(1).all(|&p| known.get(p).status() == Status::Free)
}

#[allow(non_snake_case)]
fn PathToTarget<F: Fn(Point) -> bool>(
        ctx: &mut Ctx, target: Point, range: i32, valid: F) -> Option<Action> {
    let Ctx { known, pos: source, .. } = *ctx;
    let rng = &mut ctx.env.rng;
    let check = |p: Point| known.get(p).status();
    let step = |dir: Point| {
        let look = target - source - dir;
        Action::Move(MoveAction { step: dir, look, turns: 1. })
    };

    // Given a non-empty list of "good" directions (each of which brings us
    // close to attacking the target), choose one closest to our attack range.
    let pick = |dirs: &Vec<Point>, rng: &mut RNG| {
        let cell = known.get(target);
        let (shade, tile) = (cell.shade(), cell.tile());
        let obscured = shade || match tile {
            Some(x) => x.limits_vision() && !x.blocks_movement(),
            None => false,
        };
        let distance = if obscured { 1 } else { min(range, MOVE_VOLUME) };

        assert!(!dirs.is_empty());
        let scores: Vec<_> = dirs.iter().map(
            |&x| ((x + source - target).len_nethack() - distance).abs()).collect();
        let best = *scores.iter().reduce(|acc, x| min(acc, x)).unwrap();
        let opts: Vec<_> = (0..dirs.len()).filter(|&i| scores[i] == best).collect();
        step(dirs[*sample(&opts, rng)])
    };

    // If we could already attack the target, don't move out of view.
    if valid(source) {
        let mut dirs = vec![Point::default()];
        for &x in &dirs::ALL {
            if check(source + x) != Status::Free { continue; }
            if valid(source + x) { dirs.push(x); }
        }
        return Some(pick(&dirs, rng));
    }

    // Else, pick a direction which brings us in view.
    let result = BFS(source, &valid, BFS_LIMIT_ATTACK, check);
    if let Some(x) = result && !x.dirs.is_empty() { return Some(pick(&x.dirs, rng)); }

    // Else, move towards the target.
    let path = AStar(source, target, ASTAR_LIMIT_ATTACK, check)?;
    Some(step(*path.first()? - source))
}

//////////////////////////////////////////////////////////////////////////////

#[allow(non_snake_case)]
fn Hunger(x: &mut Ctx) -> i64 {
    if !x.blackboard.hunger { return -1; }
    if x.blackboard.finding_food_ { return 101; }
    let (limit, value) = (MAX_HUNGER, x.blackboard.till_hunger);
    (100 * (limit - value) / limit) as i64
}

#[allow(non_snake_case)]
fn Thirst(x: &mut Ctx) -> i64 {
    if !x.blackboard.thirst { return -1; }
    if x.blackboard.finding_water { return 101; }
    let (limit, value) = (MAX_THIRST, x.blackboard.till_thirst);
    (100 * (limit - value) / limit) as i64
}

#[allow(non_snake_case)]
fn Weariness(x: &mut Ctx) -> i64 {
    if !x.blackboard.weary_ { return -1; }
    if x.blackboard.getting_rest_ { return 101; }
    let (limit, value) = (MAX_WEARY_, x.blackboard.till_weary_);
    (100 * (limit - value) / limit) as i64
}

#[allow(non_snake_case)]
fn HasBerry(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map(|x| x.items.contains(&Item::Berry)).unwrap_or(false)
}

#[allow(non_snake_case)]
fn HasWater(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map(|x| x.tile.can_drink()).unwrap_or(false)
}

#[allow(non_snake_case)]
fn HasBerryTree(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map(|x| x.tile.drops_berries()).unwrap_or(false)
}

#[allow(non_snake_case)]
fn CanRestAt(ctx: &Ctx, point: Point) -> bool {
    if !is_hiding_place(ctx, point) { return false; }
    point == ctx.pos || ctx.known.get(point).status() == Status::Free
}

trait CellPredicate = Fn(&Ctx, Point) -> bool;

#[allow(non_snake_case)]
fn FindNeed<F: CellPredicate>(ctx: &mut Ctx, kind: PathKind, valid: F) -> bool {
    ensure_neighborhood(ctx);

    let n = &ctx.neighborhood;
    for &(point, _) in n.blocked.iter().chain(&n.visited) {
        if valid(ctx, point) { return FindPath(ctx, point, kind); }
    }
    false
}

#[allow(non_snake_case)]
fn CheckPathTarget<F: CellPredicate>(ctx: &mut Ctx, kind: PathKind, valid: F) -> bool {
    if ctx.blackboard.path.kind != kind { return false; }

    let okay = ctx.blackboard.path.path.last().map(|&x| valid(ctx, x)).unwrap_or(false);
    if !okay { ctx.blackboard.path = Default::default(); }
    okay
}

#[allow(non_snake_case)]
fn ChooseBestNeighbor<F: CellPredicate>(ctx: &Ctx, valid: F) -> Option<Point> {
    let mut best = (std::f64::NEG_INFINITY, None);
    let Ctx { pos, dir, .. } = *ctx;
    for &x in [dirs::NONE].iter().chain(&dirs::ALL) {
        if !valid(ctx, pos + x) { continue; }
        let score = (dir.dot(x) as f64).pow(2) / max(x.len_l2_squared(), 1) as f64;
        if score > best.0 { best = (score, Some(pos + x)); }
    }
    best.1
}

#[allow(non_snake_case)]
fn EatBerryNearby(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let target = ChooseBestNeighbor(ctx, HasBerry)?;
    if !known.get(target).visible() { return Some(Action::Look(target - pos)); }

    let gain = ctx.env.rng.gen_range(HUNGER_GAIN);
    ctx.blackboard.till_hunger = min(ctx.blackboard.till_hunger + gain, MAX_HUNGER);
    Some(Action::Eat(EatAction { target, item: Some(Item::Berry) }))
}

#[allow(non_snake_case)]
fn DrinkWaterNearby(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let target = ChooseBestNeighbor(ctx, HasWater)?;
    if !known.get(target).visible() { return Some(Action::Look(target - pos)); }

    let gain = ctx.env.rng.gen_range(THIRST_GAIN);
    ctx.blackboard.till_thirst = min(ctx.blackboard.till_thirst + gain, MAX_THIRST);
    Some(Action::Drink(target))
}

#[allow(non_snake_case)]
fn FindNearbyBerryTree(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let target = ChooseBestNeighbor(ctx, HasBerryTree)?;

    let path = LOS(pos, target);
    ctx.blackboard.path = CachedPath { kind: PathKind::BerryTree, path, skip: 1, step: 0 };

    if !known.get(target).visible() { Some(Action::Look(target - pos)) } else { None }
}

#[allow(non_snake_case)]
fn RestHere(ctx: &mut Ctx) -> Option<Action> {
    if !CanRestAt(ctx, ctx.pos) { return None; }

    let gain = ctx.env.rng.gen_range(RESTED_GAIN);
    ctx.blackboard.till_weary_ = min(ctx.blackboard.till_weary_ + gain, MAX_WEARY_);
    Some(Action::Rest)
}

//////////////////////////////////////////////////////////////////////////////

struct ChaseTarget {
    bias: Point,
    last: Point,
    time: Timestamp,
    steps: i32,
}

#[allow(non_snake_case)]
fn HungryForMeat(ctx: &Ctx) -> bool {
    ctx.entity.predator && ctx.blackboard.till_hunger < MAX_HUNGER / 2
}

#[allow(non_snake_case)]
fn HasMeat(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map(|x| x.items.contains(&Item::Corpse)).unwrap_or(false)
}

#[allow(non_snake_case)]
fn EatMeatNearby(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let target = ChooseBestNeighbor(ctx, HasMeat)?;
    if !known.get(target).visible() { return Some(Action::Look(target - pos)); }

    ctx.blackboard.till_hunger = MAX_HUNGER;
    Some(Action::Eat(EatAction { target, item: Some(Item::Corpse) }))
}

#[allow(non_snake_case)]
fn SelectTargetPrey(ctx: &mut Ctx) -> bool {
    let prev = ctx.blackboard.target.take();

    let Ctx { known, pos, .. } = *ctx;
    let mut targets: Vec<_> = ctx.known.entities.iter().filter_map(
        |x| if x.delta < 0 { Some((x.pos, x.time)) } else { None }).collect();
    let Some(x) = targets.first() else { return false };
    if (known.time - x.1) >= MAX_SEARCH_TIME { return false; }

    let target = *targets.select_nth_unstable_by_key(
        0, |x| (known.time - x.1, (pos - x.0).len_l2_squared())).1;
    let reset = target.1 > ctx.blackboard.prev_time;
    if reset && ctx.blackboard.path.kind == PathKind::Prey {
        ctx.blackboard.path = Default::default()
    }

    let (last, time) = target;
    let (bias, steps) = if !reset && let Some(x) = prev && x.last == last {
        (x.bias, x.steps + 1)
    } else {
        (last - pos, 0)
    };
    ctx.blackboard.target = Some(ChaseTarget { bias, last, time, steps });
    true
}

#[allow(non_snake_case)]
fn AttackPrey(ctx: &mut Ctx) -> Option<Action> {
    let &ChaseTarget { last, time, .. } = ctx.blackboard.target.as_ref()?;
    if time == ctx.known.time { AttackTarget(ctx, last) } else { None }
}

#[allow(non_snake_case)]
fn SearchForPrey(ctx: &mut Ctx) -> Option<Action> {
    let &ChaseTarget { bias, last, time, steps } = ctx.blackboard.target.as_ref()?;
    let Ctx { known, pos, .. } = *ctx;
    let age = known.time - time;

    let center = if steps > bias.len_l1() { pos } else { last };
    let target = get_chase_target(ctx, age, bias, center)?;

    // TODO: CachedPath should handle this case for us...
    if (target - pos).len_l1() == 1 { return Some(Action::Look(target - pos)); }

    let kind = PathKind::Prey;
    if FindPath(ctx, target, kind) { FollowPath(ctx, kind) } else { None }
}

//////////////////////////////////////////////////////////////////////////////

// Behavior tree configuration

// TODO list:
//
//  1. Last-seen cache for cells satisfying a need, to skip repeated searches.
//
//  2. Periodically re-plan a path to a need if there is a closer one?
//
//  3. Utility-based selection between different needs. But it seems like we'd
//     need to run searches for multiple ones to get it right. How to?
//
//  4. High-priority Survive subtree: Track, Chase, Flee, Hide, CallForHelp...

#[allow(non_snake_case)]
fn AttackOrFollowPath(kind: PathKind) -> impl Bhv {
    pri![
        "AttackOrFollowPath",
        act!("AttackPathTarget", move |x| AttackPathTarget(x, kind)),
        act!("FollowPath", move |x| FollowPath(x, kind)),
    ]
}

macro_rules! path {
    ($k:expr, $v:expr, $f:expr) => {
        crate::bhv::Node::new((), crate::bhv::Composite::new(
            crate::bhv::PriPolicy {},
            seq!["FollowExistingPath", cond!("CheckPath", |x| CheckPathTarget(x, $k, $v)), $f],
            seq!["FindNewPath", cond!("FindPath", |x| FindNeed(x, $k, $v)), $f],
        ))
    };
}

#[allow(non_snake_case)]
fn ForageForBerries() -> impl Bhv {
    const KIND: PathKind = PathKind::BerryTree;
    pri![
        "ForageForBerries",
        act!("FindNearbyBerryTree", FindNearbyBerryTree),
        path!(KIND, HasBerryTree, AttackOrFollowPath(KIND)),
    ]
}

#[allow(non_snake_case)]
fn EatBerries() -> impl Bhv {
    const KIND: PathKind = PathKind::Berry;
    pri![
        "EatBerries",
        act!("EatBerryNearby", EatBerryNearby),
        path!(KIND, HasBerry, act!("FollowPath", |x| FollowPath(x, KIND))),
    ]
}

#[allow(non_snake_case)]
fn EatFood() -> impl Bhv {
    pri!["EatFood", EatBerries(), ForageForBerries()]
        .on_enter(|x| x.blackboard.finding_food_ = true)
        .on_exit(|x| x.blackboard.finding_food_ = false)
}

#[allow(non_snake_case)]
fn DrinkWater() -> impl Bhv {
    const KIND: PathKind = PathKind::Water;
    pri![
        "DrinkWater",
        act!("DrinkWaterNearby", DrinkWaterNearby),
        path!(KIND, HasWater, act!("FollowPath", |x| FollowPath(x, KIND))),
    ]
    .on_enter(|x| x.blackboard.finding_water = true)
    .on_exit(|x| x.blackboard.finding_water = false)
}

#[allow(non_snake_case)]
fn GetRest() -> impl Bhv {
    const KIND: PathKind = PathKind::Rest;
    pri![
        "GetRest",
        act!("RestHere", RestHere),
        path!(KIND, CanRestAt, act!("FollowPath", |x| FollowPath(x, KIND))),
    ]
    .on_enter(|x| x.blackboard.getting_rest_ = true)
    .on_exit(|x| x.blackboard.getting_rest_ = false)
}

#[allow(non_snake_case)]
fn HuntForMeat() -> impl Bhv {
    const KIND: PathKind = PathKind::Meat;
    seq![
        "HuntForMeat",
        cond!("HungryForMeat", |x| HungryForMeat(x)),
        pri![
            "HuntForMeat",
            pri![
                "EatMeat",
                act!("EatMeatNearby", EatMeatNearby),
                path!(KIND, HasMeat, act!("FollowPath", |x| FollowPath(x, KIND))),
            ],
            seq![
                "HuntForPrey",
                cond!("SelectTargetPrey", SelectTargetPrey),
                pri![
                    "HuntDownTarget",
                    act!("AttackPrey", AttackPrey),
                    act!("Follow(Prey)", |x| FollowPath(x, PathKind::Prey)),
                    act!("Search(Prey)", SearchForPrey),
                ],
            ],
        ],
    ]
}

#[allow(non_snake_case)]
fn AddressBasicNeeds() -> impl Bhv {
    util![
        "AddressBasicNeeds",
        (Hunger, EatFood()),
        (Thirst, DrinkWater()),
        (Weariness, GetRest()),
    ]
}

#[allow(non_snake_case)]
fn Root() -> Box<dyn Bhv> {
    Box::new(pri![
        "Root",
        cb!("TickBasicNeeds", TickBasicNeeds),
        HuntForMeat(),
        act!("Follow(LookAround)", FollowDirs),
        AddressBasicNeeds(),
        act!("Follow(Explore)", |x| FollowPath(x, PathKind::Explore)),
        act!("LookAround", Assess),
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
        Self { blackboard, tree: Root() }
    }

    pub fn get_path(&self) -> &[Point] {
        &self.blackboard.path.path
    }

    pub fn debug(&self, slice: &mut Slice) {
        let mut debug = Debug { depth: 0, slice, verbose: true };
        self.tree.debug(&mut debug);
        slice.newline();
        let bb = &self.blackboard;
        slice.write_str(&format!("finding_food_: {}", bb.finding_food_)).newline();
        slice.write_str(&format!("finding_water: {}", bb.finding_water)).newline();
        slice.write_str(&format!(
                "till_assess: {} / {}", bb.till_assess, MAX_ASSESS)).newline();
        slice.write_str(&format!(
                "till_hunger: {} / {}{}", bb.till_hunger, MAX_HUNGER,
                if bb.hunger { " (hungry)" } else { "" })).newline();
        slice.write_str(&format!(
                "till_thirst: {} / {}{}", bb.till_thirst, MAX_THIRST,
                if bb.thirst { " (thirsty)" } else { "" })).newline();
        slice.write_str(&format!(
                "till_weary_: {} / {}{}", bb.till_weary_, MAX_WEARY_,
                if bb.weary_ { " (weary)" } else { "" })).newline();
        slice.write_str(&format!("dirs: {} items", bb.dirs.len())).newline();
        slice.write_str(&format!("path: {:?}", bb.path.kind)).newline();
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
