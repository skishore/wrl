use std::cmp::{max, min};
use std::f64::consts::TAU;
use std::ops::RangeInclusive;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::{act, cb, cond, pri, run, seq, util};
use crate::base::{LOS, Point, RNG, Slice, dirs, sample, weighted};
use crate::bhv::{Bhv, Debug, Result};
use crate::entity::Entity;
use crate::game::{FOV_RADIUS_NPC, CALL_VOLUME, MOVE_VOLUME, Item, move_ready};
use crate::game::{Action, AttackAction, CallAction, EatAction, MoveAction};
use crate::knowledge::{Call, Knowledge, Sense, Timedelta, Timestamp};
use crate::pathing::{AStar, AStarHeuristic, Status};
use crate::pathing::{BFS, DijkstraLength, DijkstraMap, Neighborhood};
use crate::shadowcast::{INITIAL_VISIBILITY, Vision, VisionArgs};
use crate::threats::{FightOrFlight, ThreatState};

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
const MAX_HUNGER: i32 = 1024;
const MAX_THIRST: i32 = 256;
const MAX_WEARY_: i32 = 4096;

const ASSESS_GAIN: RangeInclusive<i32> = (MAX_ASSESS / 2)..=MAX_ASSESS;
const HUNGER_GAIN: RangeInclusive<i32> = (MAX_HUNGER / 4)..=(MAX_HUNGER / 2);
const THIRST_GAIN: RangeInclusive<i32> = (MAX_THIRST / 4)..=(MAX_THIRST / 2);
const RESTED_GAIN: RangeInclusive<i32> = 1..=2;

const WARNING_LIMIT: Timedelta = Timedelta::from_seconds(16.);
const WARNING_RETRY: Timedelta = Timedelta::from_seconds(2.);

const MIN_SEARCH_TIME: Timedelta = Timedelta::from_seconds(24.);
const MAX_SEARCH_TIME: Timedelta = Timedelta::from_seconds(48.);

const MAX_TRACKING_TIME: Timedelta = Timedelta::from_seconds(64.);
const SCENT_AGE_PENALTY: Timedelta = Timedelta::from_seconds(1.);

const FLIGHT_PATH_TURNS: i32 = 8;
const MIN_FLIGHT_TURNS: i32 = 16;
const MAX_FLIGHT_TURNS: i32 = 64;

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
    dirs: CachedDirs,
    path: CachedPath,
    threats: ThreatState,
    flight: Option<FlightState>,

    // Per-tick chase state:
    target: Option<ChaseTarget>,
    options: Vec<Target>,
    had_target: bool,

    prev_time: Timestamp,
    turn_time: Timestamp,
    last_warning: Timestamp,

    // Basic needs:
    till_assess: i32,
    till_hunger: i32,
    till_thirst: i32,
    till_weary_: i32,

    chasing_enemy: bool,
    chasing_prey_: bool,
    finding_food_: bool,
    finding_water: bool,
    getting_rest_: bool,

    hunger: bool,
    thirst: bool,
    weary_: bool,
}

impl Blackboard {
    fn new(_predator: bool, rng: &mut RNG) -> Self {
        Self {
            dirs: Default::default(),
            path: Default::default(),
            threats: Default::default(),
            flight: None,

            // Per-tick chase state:
            target: None,
            options: vec![],
            had_target: false,

            prev_time: Timestamp::default(),
            turn_time: Timestamp::default(),
            last_warning: Timestamp::default(),

            // Basic needs:
            till_assess: rng.random_range(0..MAX_ASSESS),
            till_hunger: rng.random_range(0..MAX_HUNGER),
            till_thirst: rng.random_range(0..MAX_THIRST),
            till_weary_: rng.random_range(0..MAX_WEARY_),

            chasing_enemy: false,
            chasing_prey_: false,
            finding_food_: false,
            finding_water: false,
            getting_rest_: false,

            hunger: false,
            thirst: false,
            weary_: rng.gen_bool(0.5),
        }
    }

    fn debug(&self, slice: &mut Slice) {
        slice.write_str("Blackboard:").newline();
        slice.write_str(&format!("  finding_food_: {}", self.finding_food_)).newline();
        slice.write_str(&format!("  finding_water: {}", self.finding_water)).newline();
        slice.write_str(&format!(
                "  till_assess: {} / {}", self.till_assess, MAX_ASSESS)).newline();
        slice.write_str(&format!(
                "  till_hunger: {} / {}{}", self.till_hunger, MAX_HUNGER,
                if self.hunger { " (hungry)" } else { "" })).newline();
        slice.write_str(&format!(
                "  till_thirst: {} / {}{}", self.till_thirst, MAX_THIRST,
                if self.thirst { " (thirsty)" } else { "" })).newline();
        slice.write_str(&format!(
                "  till_weary_: {} / {}{}", self.till_weary_, MAX_WEARY_,
                if self.weary_ { " (weary)" } else { "" })).newline();
        slice.write_str(&format!(
                "  dirs: {:?} ({} items)", self.dirs.kind, self.dirs.dirs.len())).newline();
        slice.write_str(&format!("  path: {:?}", self.path.kind)).newline();
        slice.newline();

        if let Some(x) = &self.flight {
            slice.write_str("Flight:").newline();
            slice.write_str(&format!("  needs_path: {}", x.needs_path)).newline();
            slice.write_str(&format!("  since_seen: {}", x.since_seen)).newline();
            slice.write_str(&format!("  turn_limit: {}", x.turn_limit)).newline();
            slice.newline();
        }

        if let Some(x) = &self.target {
            slice.write_str("Target:").newline();
            slice.write_str(&format!("  bias: {:?}", x.bias)).newline();
            slice.write_str(&format!("  fresh: {}", x.fresh)).newline();
            slice.write_str(&format!("  steps: {}", x.steps)).newline();
            slice.write_str(&format!(
                    "  target.age: {:?}", self.turn_time - x.target.time)).newline();
            slice.write_str(&format!(
                    "  target.last: {:?}", x.target.last)).newline();
            slice.write_str(&format!(
                    "  target.sense: {:?}", x.target.sense)).newline();
            slice.newline();
        }

        self.threats.debug(slice, self.turn_time);
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

fn all_threats_asleep(ctx: &Ctx) -> bool {
    ctx.blackboard.threats.menacing.iter().all(|x| x.asleep)
}

fn is_hiding_place(ctx: &Ctx, point: Point) -> bool {
    if ctx.blackboard.threats.menacing.iter().any(
        |x| (x.pos - point).len_l1() <= 1) { return false; }
    let cell = ctx.known.get(point);
    cell.shade() || matches!(cell.tile(), Some(x) if x.limits_vision())
}

fn get_basic_check<'a>(ctx: &'a Ctx) -> impl Fn(Point) -> Status + use<'a> {
    let (fov, known, pos) = (&ctx.env.fov, ctx.known, ctx.pos);
    move |p: Point| match known.get(p).status() {
        Status::Occupied if (p - pos).len_l1() == 1 => Status::Blocked,
        Status::Unknown if fov.can_see(p - pos) => Status::Free,
        x => x,
    }
}

fn get_sneak_check<'a, 'b, 'c>(
        ctx: &'a Ctx<'b, 'c>) -> impl Fn(Point) -> Status + use<'a, 'b, 'c> {
    let (known, pos) = (ctx.known, ctx.pos);
    move |p: Point| {
        if !is_hiding_place(ctx, p) { return Status::Blocked; }
        match known.get(p).status() {
            Status::Occupied if (p - pos).len_l1() == 1 => Status::Blocked,
            x => x
        }
    }
}

fn ensure_neighborhood(ctx: &mut Ctx) {
    if !ctx.neighborhood.visited.is_empty() { return; }

    ensure_vision(ctx);
    let (pos, check) = (ctx.pos, get_basic_check(ctx));
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

fn select_target_softmax(scores: &[(Point, f64)], env: &mut AIEnv, temp: f64) -> Option<Point> {
    if scores.is_empty() { return None; }

    let max = scores.iter().fold(
        std::f64::NEG_INFINITY, |acc, x| if acc > x.1 { acc } else { x.1 });
    let scale = ((1 << 16) - 1) as f64;
    let inv_temp = 1. / temp;
    let values: Vec<_> = scores.iter().map(|&(p, score)| {
        let value = (scale * (inv_temp * (score - max)).exp()) as i32;
        assert!(0 <= value && value < (1 << 16));
        (value, p)
    }).collect();

    if let Some(x) = env.debug.as_deref_mut() {
        x.utility.clear();
        for &(score, point) in &values {
            x.utility.push((point, (score >> 8) as u8));
        }
    }
    Some(*weighted(&values, env.rng))
}

fn select_explore_target(ctx: &mut Ctx) -> Option<Point> {
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

fn select_chase_target(
        ctx: &mut Ctx, age: Timedelta, bias: Point, center: Point) -> Option<Point> {
    let Ctx { known, pos, dir, .. } = *ctx;
    let inv_dir_l2 = safe_inv_l2(dir);
    let inv_bias_l2 = safe_inv_l2(bias);

    let is_search_candidate = |p: Point| {
        if p == pos { return false; }
        let cell = known.get(p);
        !cell.blocked() && cell.time_since_entity_visible() >= age
    };
    if is_search_candidate(center) { return Some(center); }

    let score = |p: Point| -> f64 {
        if !is_search_candidate(p) { return 0.; }

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos0 = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let cos1 = delta.dot(bias) as f64 * inv_delta_l2 * inv_bias_l2;
        let angle = ((cos0 + 1.) * (cos1 + 1.)).pow(4);

        angle / (((p - center).len_l2_squared() + 1) as f64).pow(2)
    };

    ensure_neighborhood(ctx);

    let scores: Vec<_> = ctx.neighborhood.visited.iter().map(
        |&(p, _)| (p, score(p))).collect();
    select_target(&scores, ctx.env)
}

fn select_flight_target(ctx: &mut Ctx, hiding: bool) -> Option<Point> {
    let Ctx { known, pos, .. } = *ctx;

    let min_distance = DijkstraLength(Point(FOV_RADIUS_NPC, 0));
    let scale = 1. / DijkstraLength(Point(1, 0)) as f64;
    let threats = &ctx.blackboard.threats.menacing;

    let score = |p: Point, source_distance: i32| -> (f64, bool) {
        let mut threat = Point::default();
        let mut threat_distance = std::i32::MAX;
        for x in threats {
            let z = DijkstraLength(p - x.pos);
            if z < threat_distance { (threat, threat_distance) = (x.pos, z); }
        }

        let blocked = {
            let los = LOS(threat, p);
            (1..los.len() - 1).any(|i| known.get(los[i]).blocked())
        };
        let frontier = dirs::ALL.iter().any(|&x| known.get(p + x).unknown());
        let hidden = hiding || is_hiding_place(ctx, p);

        // This heuristic can cause a piece to be "checkmated" in a corner,
        // if we don't find a cell that's far enough away. But that's okay -
        // in that case, we'll switch to fighting back.
        let score = 2.5 * scale * threat_distance as f64 +
                    -1. * scale * source_distance as f64 +
                    16. * if blocked { 1. } else { 0. } +
                    16. * if frontier { 1. } else { 0. } +
                    64. * if hidden { 1. } else { 0. };
        let valid = hidden || blocked || threat_distance > min_distance;
        (score, valid)
    };

    let min_score = score(pos, 0).0;
    let map = if hiding { &ctx.sneakable.visited } else { &ctx.neighborhood.visited };
    let scores: Vec<_> = map.iter().filter_map(|&(p, distance)| {
        let (score, valid) = score(p, distance);
        if valid && score >= min_score { Some((p, score)) } else { None }
    }).collect();
    select_target_softmax(&scores, ctx.env, 0.1)
}

//////////////////////////////////////////////////////////////////////////////

// Basic state updates

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

#[allow(non_snake_case)]
fn RunCombatAnalysis(ctx: &mut Ctx) -> Result {
    let (bb, entity) = (&mut *ctx.blackboard, ctx.entity);
    bb.threats.update(entity, bb.prev_time);
    Result::Failed
}

#[allow(non_snake_case)]
fn ForceThreatState(ctx: &mut Ctx, state: FightOrFlight) {
    let threats = &mut ctx.blackboard.threats;
    if threats.state != FightOrFlight::Safe { threats.state = state; }
}

//////////////////////////////////////////////////////////////////////////////

// Wandering:

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DirsKind { Assess, Flight, Noises, Target, #[default] None }

#[derive(Default)]
struct CachedDirs {
    kind: DirsKind,
    dirs: Vec<Point>,
    used: bool,
}

#[allow(non_snake_case)]
fn CleanupDirs(ctx: &mut Ctx) {
    let dirs = &mut ctx.blackboard.dirs;
    if !dirs.used { *dirs = Default::default(); }
    dirs.used = false;
}

#[allow(non_snake_case)]
fn FollowDirs(ctx: &mut Ctx, kind: DirsKind) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    if bb.dirs.kind != kind { return None; }
    let dir = bb.dirs.dirs.pop()?;

    if bb.dirs.dirs.is_empty() {
        let gain = rng.gen_range(ASSESS_GAIN);
        bb.till_assess = min(bb.till_assess + gain, MAX_ASSESS);
    } else {
        bb.dirs.used = true;
    }
    Some(Action::Look(dir))
}

#[allow(non_snake_case)]
fn Assess(ctx: &mut Ctx) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    if bb.till_assess > 0 { return None; }

    let kind = DirsKind::Assess;
    let dirs = assess_directions(&[ctx.dir], ASSESS_TURNS_EXPLORE, rng);
    bb.dirs = CachedDirs { kind, dirs, used: false };
    FollowDirs(ctx, kind)
}

#[allow(non_snake_case)]
fn Explore(ctx: &mut Ctx) -> Option<Action> {
    let kind = PathKind::Explore;
    let target = select_explore_target(ctx)?;
    if FindPath(ctx, target, kind) { FollowPath(ctx, kind) } else { None }
}

#[allow(non_snake_case)]
fn HeardUnknownNoise(ctx: &mut Ctx) -> bool {
    let bb = &mut ctx.blackboard;
    if bb.threats.unknown.is_empty() { return false; }

    if bb.dirs.kind == DirsKind::Noises && bb.dirs.dirs.len() == 1 {
        for threat in &mut bb.threats.threats { threat.mark_scanned(); }
    }
    true
}

#[allow(non_snake_case)]
fn LookForLastTarget(ctx: &mut Ctx) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    if !bb.had_target { return None; }

    let kind = DirsKind::Target;
    let dirs = assess_directions(&[ctx.dir], ASSESS_TURNS_FLIGHT, rng);
    bb.dirs = CachedDirs { kind, dirs, used: false };
    FollowDirs(ctx, kind)
}

#[allow(non_snake_case)]
fn LookForNoises(ctx: &mut Ctx) -> Option<Action> {
    let threats = &ctx.blackboard.threats;
    let (pos, rng) = (ctx.pos, &mut ctx.env.rng);

    let dirs: Vec<_> = threats.unknown.iter().filter_map(
        |x| if x.pos != pos { Some(x.pos - pos) } else { None }).collect();
    let dirs = if dirs.is_empty() { &[ctx.dir] } else { dirs.as_slice() };

    let kind = DirsKind::Noises;
    let dirs = assess_directions(&dirs, ASSESS_TURNS_FLIGHT, rng);
    ctx.blackboard.dirs = CachedDirs { kind, dirs, used: false };
    FollowDirs(ctx, kind)
}

#[allow(non_snake_case)]
fn WarnOffThreats(ctx: &mut Ctx) -> Option<Action> {
    let (pos, time) = (ctx.pos, ctx.blackboard.turn_time);
    let (rng, threats) = (&mut *ctx.env.rng, &mut ctx.blackboard.threats);
    let stare = time - ctx.blackboard.last_warning < WARNING_RETRY;
    let mut result = None;

    for threat in &mut threats.threats {
        if time - threat.time >= WARNING_LIMIT { break; }

        if !threat.uncertain() { continue; }
        if (threat.pos - pos).len_nethack() > CALL_VOLUME { continue; }

        let warn = !stare && threat.time > ctx.blackboard.last_warning;
        if warn { threat.mark_warned(rng); }

        if result.is_some() { continue; }

        let (call, look) = (Call::Warning, threat.pos - pos);
        if warn {
            result = Some(Action::Call(CallAction { call, look }));
            ctx.blackboard.last_warning = time;
        } else if stare {
            result = Some(Action::Look(look));
        };
    }
    result
}

//////////////////////////////////////////////////////////////////////////////

// Pathfinding:

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PathKind {
    Enemy,
    Hide,
    Flee,
    Meat,
    Rest,
    Water,
    Berry,
    BerryTree,
    Explore,
    #[default] None,
}

#[derive(Default)]
struct CachedPath {
    kind: PathKind,
    path: Vec<Point>,
    skip: usize,
    step: usize,
}

#[allow(non_snake_case)]
pub fn AStarHelper(ctx: &mut Ctx, target: Point, hiding: bool) -> Option<Vec<Point>> {
    // Try using A* to find the best path:
    let source = ctx.pos;
    let result = if hiding {
        AStar(source, target, ASTAR_LIMIT_WANDER, get_sneak_check(ctx))
    } else {
        AStar(source, target, ASTAR_LIMIT_WANDER, get_basic_check(ctx))
    };
    if let Some(mut path) = result {
        path.insert(0, source);
        return Some(path);
    }

    // If that fails, recover a path from the Dijkstra neighborhood:
    let cells = if hiding { &mut ctx.sneakable } else { &mut ctx.neighborhood };
    if cells.visited.is_empty() { return None; }

    // Lazily construct a table of neighborhood's scores:
    let scores = &mut cells.scores;
    if scores.is_empty() { *scores = cells.visited.iter().map(|&x| x).collect(); }

    // Walk back from `target`, greedily moving to the closest neighbor to `source`.
    // Use the A* heuristic to break ties to favor that follows the LOS.
    let mut prev = target;
    let mut path = vec![target];
    let los = LOS(source, target);
    while prev != source {
        let (mut best_point, mut best_score) = (None, (std::i32::MAX, std::i32::MAX));
        for &dir in &dirs::ALL {
            let point = prev + dir;
            let Some(&score) = cells.scores.get(&point) else { continue };
            let score = (score, AStarHeuristic(point, &los));
            if score < best_score { (best_point, best_score) = (Some(point), score); }
        }
        let Some(next) = best_point else { return None };

        path.push(next);
        prev = next;
    }
    path.reverse();
    Some(path)
}

#[allow(non_snake_case)]
fn FindPath(ctx: &mut Ctx, target: Point, kind: PathKind) -> bool {
    ensure_vision(ctx);
    let path = AStarHelper(ctx, target, /*hiding=*/false);
    let Some(path) = path else { return false };

    type K = PathKind;
    let skip = match kind {
        K::Meat | K::Water | K::Berry | K::BerryTree => 1,
        K::Enemy | K::Hide | K::Flee | K::Rest | K::Explore | K::None => 0,
    };
    ctx.blackboard.path = CachedPath { kind, path, skip, step: 0 };
    true
}

#[allow(non_snake_case)]
fn FollowPath(ctx: &mut Ctx, kind: PathKind) -> Option<Action> {
    if ctx.blackboard.path.kind != kind { return None; }

    let (known, pos) = (ctx.known, ctx.pos);
    let path = std::mem::take(&mut ctx.blackboard.path);
    if path.path.is_empty() { return None; }

    // Check if every cell on the path is free. Other than the cell that we'll
    // move to next, we allow entities to temporarily move onto the path.
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

    // When sneaking, also check that all cells are valid hiding places.
    let seen = kind == PathKind::Hide && path.path.iter().skip(i).any(
        |&x| !is_hiding_place(ctx, x));
    if seen { return None; }

    // The path is good! Follow it. Look ahead as far as possible on the path.
    let next = path.path[j];
    let mut target = next;
    for &point in path.path.iter().skip(j).take(8) {
        let los = LOS(pos, point);
        if los.iter().all(|&x| known.get(x).unblocked()) { target = point; }
    }
    let (look, step) = (target - pos, next - pos);

    // Determine how fast to move on the path. Only move quickly (and noisily)
    // when fleeing from an enemy or chasing one down.
    let mut turns = WANDER_TURNS;
    if kind == PathKind::Enemy && let Some(x) = &ctx.blackboard.target {
        let age = ctx.known.time - x.target.time;
        if age < MIN_SEARCH_TIME && x.target.sense != Sense::Smell { turns = 1. };
    } else if kind == PathKind::Flee && !all_threats_asleep(ctx) {
        turns = 1.;
    }

    // Clear the path if this move takes us to the end.
    if path.step + 2 < path.path.len() {
        ctx.blackboard.path = path;
        ctx.blackboard.path.step += 1;
    }
    Some(Action::Move(MoveAction { look, step, turns }))
}

//////////////////////////////////////////////////////////////////////////////

// Attacking:

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

// Basic needs:

#[allow(non_snake_case)]
fn HungryForMeat(ctx: &Ctx) -> bool {
    ctx.entity.predator && ctx.blackboard.till_hunger < MAX_HUNGER / 2
}

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
fn HasMeat(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map(|x| x.items.contains(&Item::Corpse)).unwrap_or(false)
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
fn ChooseNeighbor<F: CellPredicate>(ctx: &mut Ctx, kind: PathKind, valid: F) -> Option<Point> {
    let Ctx { pos, dir, .. } = *ctx;
    let path = &ctx.blackboard.path;
    if path.kind == kind && let Some(&x) = path.path.last() &&
       (x - pos).len_l1() <= 1 && valid(ctx, x) && ctx.known.get(x).visible() {
        return Some(x);
    }

    let mut best = (std::f64::NEG_INFINITY, None);
    for &x in [dirs::NONE].iter().chain(&dirs::ALL) {
        if !valid(ctx, pos + x) { continue; }
        let score = (dir.dot(x) as f64).pow(2) / max(x.len_l2_squared(), 1) as f64;
        if score > best.0 { best = (score, Some(pos + x)); }
    }

    let result = best.1?;
    let path = LOS(pos, result);
    ctx.blackboard.path = CachedPath { kind, path, skip: 1, step: 0 };
    Some(result)
}

#[allow(non_snake_case)]
fn EatMeatNearby(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let target = ChooseNeighbor(ctx, PathKind::Meat, HasMeat)?;
    if !known.get(target).visible() { return Some(Action::Look(target - pos)); }

    ctx.blackboard.till_hunger = MAX_HUNGER;
    Some(Action::Eat(EatAction { target, item: Some(Item::Corpse) }))
}

#[allow(non_snake_case)]
fn EatBerryNearby(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let target = ChooseNeighbor(ctx, PathKind::Berry, HasBerry)?;
    if !known.get(target).visible() { return Some(Action::Look(target - pos)); }

    let gain = ctx.env.rng.gen_range(HUNGER_GAIN);
    ctx.blackboard.till_hunger = min(ctx.blackboard.till_hunger + gain, MAX_HUNGER);
    Some(Action::Eat(EatAction { target, item: Some(Item::Berry) }))
}

#[allow(non_snake_case)]
fn DrinkWaterNearby(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let target = ChooseNeighbor(ctx, PathKind::Water, HasWater)?;
    if !known.get(target).visible() { return Some(Action::Look(target - pos)); }

    let gain = ctx.env.rng.gen_range(THIRST_GAIN);
    ctx.blackboard.till_thirst = min(ctx.blackboard.till_thirst + gain, MAX_THIRST);
    Some(Action::Drink(target))
}

#[allow(non_snake_case)]
fn FindNearbyBerryTree(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let (kind, valid) = (PathKind::BerryTree, HasBerryTree);

    if CheckPathTarget(ctx, kind, valid) {
        let cur = ctx.blackboard.path.path.last().copied()?;
        if (cur - pos).len_l1() > 1 && known.get(cur).visible() { return None; }
    }

    let target = ChooseNeighbor(ctx, kind, valid)?;
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

// Hunting:

#[derive(Clone, Copy)]
struct Target {
    last: Point,
    time: Timestamp,
    sense: Sense,
}

struct ChaseTarget {
    bias: Point,
    fresh: bool,
    steps: i32,
    target: Target,
}

#[allow(non_snake_case)]
fn CleanupChaseState(ctx: &mut Ctx) {
    let bb = &mut ctx.blackboard;
    if bb.chasing_enemy || bb.chasing_prey_ { return; }

    let chasing = bb.path.kind == PathKind::Enemy;
    if chasing { bb.path = Default::default(); }
    bb.target = None;
}

#[allow(non_snake_case)]
fn CleanupTarget(ctx: &mut Ctx) {
    ctx.blackboard.had_target = false;
}

#[allow(non_snake_case)]
fn ClearTargets(ctx: &mut Ctx) {
    ctx.blackboard.options.clear();
}

#[allow(non_snake_case)]
fn MarkSafeIfLostView(ctx: &mut Ctx) -> bool {
    if !ctx.blackboard.options.is_empty() { return false; }
    ctx.blackboard.threats.state = FightOrFlight::Safe;
    true
}

macro_rules! check_time {
    ($ctx:ident, $time:expr, $limit:expr) => {{
        let &mut Blackboard { prev_time, turn_time, .. } = $ctx.blackboard;
        if prev_time - $time < $limit { $ctx.blackboard.had_target = true; }
        turn_time - $time < $limit
    }}
}

#[allow(non_snake_case)]
fn ListThreatsBySight(ctx: &mut Ctx) -> bool {
    let initial = ctx.blackboard.options.len();
    for other in &ctx.blackboard.threats.hostile {
        if !check_time!(ctx, other.time, MIN_SEARCH_TIME) { break; }
        let target = Target { last: other.pos, time: other.time, sense: other.sense };
        ctx.blackboard.options.push(target);
    }
    ctx.blackboard.options.len() > initial
}

#[allow(non_snake_case)]
fn ListPreyBySight(ctx: &mut Ctx) -> bool {
    let initial = ctx.blackboard.options.len();
    for other in &ctx.known.entities {
        if other.delta >= 0 { continue; }
        if !check_time!(ctx, other.time, MAX_SEARCH_TIME) { break; }
        let target = Target { last: other.pos, time: other.time, sense: other.sense };
        ctx.blackboard.options.push(target);
    }
    ctx.blackboard.options.len() > initial
}

#[allow(non_snake_case)]
fn ListPreyBySound(ctx: &mut Ctx) -> bool {
    let initial = ctx.blackboard.options.len();
    for other in &ctx.blackboard.threats.threats {
        if other.seen { continue; }
        if other.delta >= 0 { continue; }
        if !check_time!(ctx, other.time, MAX_SEARCH_TIME) { break; }
        let target = Target { last: other.pos, time: other.time, sense: other.sense };
        ctx.blackboard.options.push(target);
    }
    ctx.blackboard.options.len() > initial
}

#[allow(non_snake_case)]
fn ListPreyByScent(ctx: &mut Ctx) -> bool {
    let initial = ctx.blackboard.options.len();
    if let Some(x) = &ctx.blackboard.target && x.target.sense == Sense::Smell &&
       check_time!(ctx, x.target.time, MAX_TRACKING_TIME) {
        ctx.blackboard.options.push(x.target);
    }
    for &scent in &ctx.known.scents {
        if !check_time!(ctx, scent.time, MAX_TRACKING_TIME) { continue; }
        let target = Target { last: scent.pos, time: scent.time, sense: Sense::Smell };
        ctx.blackboard.options.push(target);
    }
    ctx.blackboard.options.len() > initial
}

#[allow(non_snake_case)]
fn SelectBestTarget(ctx: &mut Ctx) -> bool {
    let options = &mut ctx.blackboard.options;
    if options.is_empty() { return false; }

    let Ctx { known, pos, .. } = *ctx;
    let score = |x: &Target| {
        let age = known.time - x.time;
        let age = if x.sense == Sense::Smell { age + SCENT_AGE_PENALTY } else { age };
        (age, (pos - x.last).len_l2_squared())
    };
    let target = *options.select_nth_unstable_by_key(0, score).1;

    let prev = &ctx.blackboard.target;
    let recent = target.time > ctx.blackboard.prev_time;
    let change = if let Some(x) = prev { x.target.last != target.last } else { true };
    let fresh = change || (recent && target.sense != Sense::Smell);
    let reset = change || recent;

    if reset && ctx.blackboard.path.kind == PathKind::Enemy {
        ctx.blackboard.path = Default::default()
    }

    let (bias, steps) = if !reset && let Some(x) = prev {
        (x.bias, x.steps + 1)
    } else {
        (target.last - pos, 0)
    };
    ctx.blackboard.target = Some(ChaseTarget { bias, fresh, steps, target });
    true
}

#[allow(non_snake_case)]
fn AttackEnemy(ctx: &mut Ctx) -> Option<Action> {
    let state = ctx.blackboard.target.as_ref()?;
    if state.target.sense == Sense::Smell { return None; }
    if state.target.time != ctx.known.time { return None; }
    AttackTarget(ctx, state.target.last)
}

#[allow(non_snake_case)]
fn TrackEnemyByScent(ctx: &mut Ctx) -> Option<Action> {
    let state = ctx.blackboard.target.as_ref()?;
    if !state.fresh || state.target.sense != Sense::Smell { return None; }
    Some(Action::SniffAround)
}

#[allow(non_snake_case)]
fn SearchForEnemy(ctx: &mut Ctx) -> Option<Action> {
    let &ChaseTarget { bias, steps, target, .. } = ctx.blackboard.target.as_ref()?;
    let Ctx { known, pos, .. } = *ctx;
    let age = known.time - target.time;

    let target = if target.sense == Sense::Smell {
        select_chase_target(ctx, age, Point(0, 0), target.last)?
    } else {
        let center = if steps > bias.len_l1() { pos } else { target.last };
        select_chase_target(ctx, age, bias, center)?
    };

    if (target - pos).len_l1() == 1 {
        let look = matches!(known.get(target).status(), Status::Blocked | Status::Occupied);
        if look { return Some(Action::Look(target - pos)); }
    }

    let kind = PathKind::Enemy;
    if FindPath(ctx, target, kind) { FollowPath(ctx, kind) } else { None }
}

//////////////////////////////////////////////////////////////////////////////

// Fleeing:

#[derive(Default)]
struct FlightState {
    needs_path: bool,
    since_seen: i32,
    turn_limit: i32,
}

#[allow(non_snake_case)]
fn CheckFlightLimit(ctx: &mut Ctx) -> bool {
    let Some(x) = &ctx.blackboard.flight else { return false };
    x.since_seen >= x.turn_limit
}

#[allow(non_snake_case)]
fn ClearFlightPath(ctx: &mut Ctx) -> Result {
    let Some(x) = &mut ctx.blackboard.flight else { return Result::Failed };
    x.needs_path = false;
    Result::Failed
}

#[allow(non_snake_case)]
fn ClearFlightState(ctx: &mut Ctx) {
    let bb = &mut ctx.blackboard;
    let fleeing = bb.path.kind == PathKind::Hide || bb.path.kind == PathKind::Flee;
    let looking = bb.dirs.kind == DirsKind::Flight;

    if fleeing { bb.path = Default::default() };
    if looking { bb.dirs = Default::default() };
    bb.flight = None;
}

#[allow(non_snake_case)]
fn UpdateFlightState(ctx: &mut Ctx) -> bool {
    let bb = &mut ctx.blackboard;
    let prev = bb.flight.take();

    // State may be Safe even if we're aware of threats, if we tried to hunt
    // them down and lost sight for long enough. See: MarkSafeIfLostView.
    let threats = &bb.threats;
    if threats.state == FightOrFlight::Safe { return false; }
    let Some(threat) = threats.menacing.first() else { return false };

    let reset = threat.time > bb.prev_time;
    let fleeing = bb.path.kind == PathKind::Hide || bb.path.kind == PathKind::Flee;
    let looking = bb.dirs.kind == DirsKind::Flight;
    let turn = bb.path.step as i32;

    let prev = prev.unwrap_or_default();
    let mut flight = FlightState {
        needs_path: reset || prev.needs_path,
        since_seen: if reset { 0 } else { prev.since_seen + 1},
        turn_limit: max(prev.turn_limit, MIN_FLIGHT_TURNS),
    };

    if fleeing && flight.needs_path && turn > FLIGHT_PATH_TURNS {
        bb.path = Default::default();
        flight.needs_path = false;
    }

    if looking && reset {
        bb.dirs = Default::default();
        flight.turn_limit = min(2 * flight.turn_limit, MAX_FLIGHT_TURNS);
    }

    if looking && !reset && bb.dirs.dirs.len() == 1 {
        bb.threats.state = FightOrFlight::Safe;
    } else {
        bb.flight = Some(flight);
    }
    true
}

#[allow(non_snake_case)]
fn LookForThreats(ctx: &mut Ctx) -> Option<Action> {
    let threats = &ctx.blackboard.threats.menacing;
    let (pos, rng, time) = (ctx.pos, &mut *ctx.env.rng, ctx.known.time);

    let mut visible: Vec<_> = threats.iter().filter_map(
        |x| if x.time == time && x.pos != pos { Some(x.pos) } else { None }).collect();
    if !visible.is_empty() {
        let threat = *visible.select_nth_unstable_by_key(
            0, |&p| ((p - pos).len_l2_squared(), p.0, p.1)).1;
        return Some(Action::Look(threat - pos));
    }

    let dirs: Vec<_> = threats.iter().filter_map(
        |x| if x.pos != pos { Some(x.pos - pos) } else { None }).collect();
    let dirs = if dirs.is_empty() { &[ctx.dir] } else { dirs.as_slice() };

    let kind = DirsKind::Flight;
    let dirs = assess_directions(&dirs, ASSESS_TURNS_FLIGHT, rng);
    ctx.blackboard.dirs = CachedDirs { kind, dirs, used: false };
    FollowDirs(ctx, kind)
}

#[allow(non_snake_case)]
fn HideFromThreats(ctx: &mut Ctx) -> Result {
    let pos = ctx.pos;
    let okay = get_sneak_check(ctx);
    ctx.sneakable = DijkstraMap(pos, okay, HIDING_CELLS, HIDING_LIMIT);
    let target = select_flight_target(ctx, /*hiding=*/true);
    let Some(target) = target else { return Result::Failed };

    if target == pos { return Result::Success; }

    let kind = PathKind::Hide;
    let path = AStarHelper(ctx, target, /*hiding=*/true);
    let Some(path) = path else { return Result::Failed };

    ctx.blackboard.path = CachedPath { kind, path, skip: 0, step: 0 };
    let Some(action) = FollowPath(ctx, kind) else { return Result::Failed };

    ctx.action = Some(action);
    Result::Running
}

#[allow(non_snake_case)]
fn FleeFromThreats(ctx: &mut Ctx) -> Result {
    let pos = ctx.pos;
    ensure_neighborhood(ctx);
    let target = select_flight_target(ctx, /*hiding=*/false);
    let Some(target) = target else { return Result::Failed };

    if target == pos { return Result::Success; }

    let kind = PathKind::Flee;
    if !FindPath(ctx, target, kind) { return Result::Failed; }
    let Some(action) = FollowPath(ctx, kind) else { return Result::Failed };

    ctx.action = Some(action);
    Result::Running
}

//////////////////////////////////////////////////////////////////////////////

// Fight-or-flight:

#[allow(non_snake_case)]
fn CallStrength(ctx: &mut Ctx) -> i64 {
    if ctx.blackboard.threats.hostile.is_empty() { return -1; }

    if ctx.blackboard.threats.call_for_help { 2 } else { -1 }
}

#[allow(non_snake_case)]
fn FightStrength(ctx: &mut Ctx) -> i64 {
    if ctx.blackboard.threats.hostile.is_empty() { return -1; }

    match ctx.blackboard.threats.state {
        FightOrFlight::Safe   => -1,
        FightOrFlight::Fight  =>  1,
        FightOrFlight::Flight =>  0,
    }
}

#[allow(non_snake_case)]
fn FlightStrength(ctx: &mut Ctx) -> i64 {
    match ctx.blackboard.threats.state {
        FightOrFlight::Safe   => -1,
        FightOrFlight::Fight  =>  0,
        FightOrFlight::Flight =>  1,
    }
}

#[allow(non_snake_case)]
fn CallForHelp(ctx: &mut Ctx) -> Option<Action> {
    let threats = &mut ctx.blackboard.threats;
    threats.on_call_for_help(ctx.pos, ctx.known.time);

    let look = threats.hostile.first().map(|x| x.pos - ctx.pos).unwrap_or(ctx.dir);
    Some(Action::Call(CallAction { call: Call::Help, look }))
}

//////////////////////////////////////////////////////////////////////////////

// Behavior tree configuration

// TODO list:
//
//  - Last-seen cache for cells satisfying a need, to skip repeated searches.
//
//  - Periodically re-plan a path to a need if there is a closer one?
//
//  - Update CachedPath to do "look at the target for a path w/ skip = 1",
//    then get rid of the Look actions for basic needs and `SearchForPrey`.
//
//  - Fix bug in `select_chase_target`: if the target hasn't moved from where
//    we last saw it, then we may not choose that cell because of the check
//    on time_since_entity_visible() > age.
//
//  - Make the our-team-strength logic quadratic in team size.
//
//  - Add "growl" / "intimidate" subtrees.
//
//  - Drop "unknown" targets earlier if we're chasing down enemies.
//
//  - Only run InvestigateNoises for recent unknown sources.
//
//  - Push "if nowhere to look, look ahead" logic into assess_directions.

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
        .on_tick(|x| x.blackboard.finding_food_ = true)
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
    .on_tick(|x| x.blackboard.finding_water = true)
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
    .on_tick(|x| x.blackboard.getting_rest_ = true)
    .on_exit(|x| x.blackboard.getting_rest_ = false)
}

#[allow(non_snake_case)]
fn Wander() -> impl Bhv {
    pri![
        "Wander",
        act!("Follow(Assess)", |x| FollowDirs(x, DirsKind::Assess)),
        util![
            "AddressBasicNeeds",
            (Hunger, EatFood()),
            (Thirst, DrinkWater()),
            (Weariness, GetRest()),
        ],
        act!("Follow(Explore)", |x| FollowPath(x, PathKind::Explore)),
        act!("Search(Assess)", Assess),
        act!("Search(Explore)", Explore),
    ]
}

#[allow(non_snake_case)]
fn InvestigateNoises() -> impl Bhv {
    seq![
        "InvestigateNoises",
        cond!("HeardUnknownNoise", HeardUnknownNoise),
        pri![
            "LookForNoises",
            act!("Follow(Noises)", |x| FollowDirs(x, DirsKind::Noises)),
            act!("Search(Noises)", LookForNoises),
        ],
    ]
}

#[allow(non_snake_case)]
fn LookForTarget() -> impl Bhv {
    pri![
        "LookForTarget",
        act!("Follow(Target)", |x| FollowDirs(x, DirsKind::Target)),
        act!("Search(Target)", LookForLastTarget),
    ]
}

#[allow(non_snake_case)]
fn HuntSelectedTarget() -> impl Bhv {
    pri![
        "HuntSelectedTarget",
        act!("AttackEnemy", AttackEnemy),
        act!("TrackPreyByScent", TrackEnemyByScent),
        act!("Follow(Prey)", |x| FollowPath(x, PathKind::Enemy)),
        act!("Search(Prey)", SearchForEnemy),
    ]
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
                run![
                    "SelectPreyTarget",
                    cond!("ListPreyBySight", ListPreyBySight),
                    cond!("ListPreyBySound", ListPreyBySound),
                    cond!("ListPreyByScent", ListPreyByScent),
                    cond!("SelectBestTarget", SelectBestTarget),
                ],
                HuntSelectedTarget(),
            ]
            .on_tick(ClearTargets)
            .on_tick(|x| x.blackboard.chasing_prey_ = true)
            .on_exit(|x| x.blackboard.chasing_prey_ = false)
        ],
    ]
}

#[allow(non_snake_case)]
fn FightAgainstThreats() -> impl Bhv {
    seq![
        "FightAgainstThreats",
        run![
            "SelectThreatTarget",
            cond!("ListThreatsBySight", ListThreatsBySight),
            cond!("MarkSafeIfLostView", MarkSafeIfLostView),
            cond!("SelectBestTarget", SelectBestTarget),
        ],
        HuntSelectedTarget(),
    ]
    .on_tick(ClearTargets)
    .on_tick(|x| ForceThreatState(x, FightOrFlight::Fight))
    .on_tick(|x| x.blackboard.chasing_enemy = true)
    .on_exit(|x| x.blackboard.chasing_enemy = false)
}

#[allow(non_snake_case)]
fn EscapeFromThreats() -> impl Bhv {
    seq![
        "EscapeFromThreats",
        cond!("UpdateFlightState", UpdateFlightState),
        pri![
            "FlightSequence",
            act!("Follow(LookForThreats)", |x| FollowDirs(x, DirsKind::Flight)),
            seq![
                "CheckIfEscaped",
                cond!("CheckFlightLimit", CheckFlightLimit),
                act!("LookForThreats", LookForThreats),
            ],
            act!("Follow(Hide)", |x| FollowPath(x, PathKind::Hide)),
            act!("Follow(Flee)", |x| FollowPath(x, PathKind::Flee)),
            cb!("ClearFlightPath", ClearFlightPath),
            seq![
                "TryHiding",
                cond!("AnyThreatsAwake", |x| !all_threats_asleep(x)),
                cond!("CurrentlyHidden", |x| is_hiding_place(x, x.pos)),
                cb!("HideFromThreats", HideFromThreats),
                act!("LookForThreats", LookForThreats),
            ],
            seq![
                "TryFleeing",
                cb!("FleeFromThreats", FleeFromThreats),
                act!("LookForThreats", LookForThreats),
            ],
        ],
    ]
    .on_tick(|x| ForceThreatState(x, FightOrFlight::Flight))
    .on_exit(ClearFlightState)
}

#[allow(non_snake_case)]
fn FightOrFlight() -> impl Bhv {
    util![
        "FightOrFlight",
        (CallStrength, act!("CallForHelp", CallForHelp)),
        (FightStrength, FightAgainstThreats()),
        (FlightStrength, EscapeFromThreats()),
    ]
}

#[allow(non_snake_case)]
fn Root() -> impl Bhv {
    pri![
        "Root",
        cb!("TickBasicNeeds", TickBasicNeeds),
        cb!("RunCombatAnalysis", RunCombatAnalysis),
        FightOrFlight(),
        HuntForMeat(),
        LookForTarget(),
        act!("WarnOffThreats", WarnOffThreats),
        InvestigateNoises(),
        Wander(),
    ]
    .post_tick(CleanupChaseState)
    .post_tick(CleanupTarget)
    .post_tick(CleanupDirs)
}

//////////////////////////////////////////////////////////////////////////////

// Entry point:

pub struct AIState {
    blackboard: Blackboard,
    tree: Box<dyn Bhv>,
}

impl AIState {
    pub fn new(predator: bool, rng: &mut RNG) -> Self {
        Self { blackboard: Blackboard::new(predator, rng), tree: Box::new(Root()) }
    }

    pub fn get_path(&self) -> &[Point] {
        &self.blackboard.path.path
    }

    pub fn debug(&self, slice: &mut Slice) {
        let mut debug = Debug { depth: 0, slice, verbose: false };
        self.tree.debug(&mut debug);
        slice.newline();
        self.blackboard.debug(slice);
    }

    pub fn plan(&mut self, entity: &Entity, env: &mut AIEnv) -> Action {
        let known = &*entity.known;
        let blackboard = &mut self.blackboard;

        let mut ctx = Ctx {
            entity,
            known,
            pos: entity.pos,
            dir: entity.dir,
            action: None,
            blackboard,
            neighborhood: Default::default(),
            sneakable: Default::default(),
            ran_vision: false,
            env,
        };
        self.tree.tick(&mut ctx);
        ctx.action.take().unwrap_or(Action::Idle)
    }
}
