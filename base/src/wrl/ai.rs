#![allow(non_snake_case)]

use std::cmp::{max, min};
use std::f64::consts::TAU;
use std::ops::RangeInclusive;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::base::pathing::{AStar, AStarHeuristic, Status};
use crate::base::point::{Bound, Delta, LOS, Point, dirs};
use crate::base::pathing::{Dijkstra, DijkstraLength, DijkstraMap, Neighborhood};
use crate::base::util::{HashMap, HashSet, RNG, clamp, sample, sortable, weighted};
use crate::base::vision::{INITIAL_VISIBILITY, Vision, VisionArgs};

use crate::{act, cb, cond, pri, run, seq, util};
use super::bhv::{Bhv, Result};
use super::debug::{DebugFile, DebugLine, DebugLog};
use super::dex::{Attack, Species};
use super::entity::{AttackTarget, Command, Entity};
use super::event::{Call, Location, Sense};
use super::game::{Action, Item, TileFlags, move_ready};
use super::game::{FOV_RADIUS_NPC, CALL_VOLUME, FOLLOW_RANGE, SUMMON_RANGE};
use super::knowledge::{Knowledge, ScentKnowledge};
use super::threats::{FightOrFlight, ThreatState};
use super::time::Timestamp;

//////////////////////////////////////////////////////////////////////////////

// Constants:

const ASTAR_CELLS_ATTACK: i32 = 256;
const ASTAR_CELLS_WANDER: i32 = 1024;
const HIDING_CELLS: i32 = 256;
const HIDING_LIMIT: i32 = 32;
const SEARCH_CELLS: i32 = 1024;
const SEARCH_LIMIT: i32 = 64;

const ASSESS_ANGLE: f64 = TAU / 18.;
const ASSESS_TURNS_FLIGHT: (i32, i32) = (4, 1);
const ASSESS_TURNS_THREAT: (i32, i32) = (2, 1);
const ASSESS_TURNS_WANDER: (i32, i32) = (4, 2);

const MAX_ASSESS: i32 = 32;
const MAX_HUNGER: i32 = 512;
const MAX_THIRST: i32 = 128;
const MAX_WEARY_: i32 = 2048;

const HUNGRY_FOR_MEAT: i32 = MAX_HUNGER / 2;

const ASSESS_GAIN: RangeInclusive<i32> = (MAX_ASSESS / 2)..=MAX_ASSESS;
const HUNGER_GAIN: RangeInclusive<i32> = (MAX_HUNGER / 4)..=(MAX_HUNGER / 2);
const THIRST_GAIN: RangeInclusive<i32> = (MAX_THIRST / 4)..=(MAX_THIRST / 2);
const RESTED_GAIN: RangeInclusive<i32> = 1..=2;

const WARNING_LIMIT_TURNS: i32 = 8;
const WARNING_RETRY_TURNS: i32 = 2;

const MIN_SEARCH_TURNS: i32 = 16;
const MAX_SEARCH_TURNS: i32 = 32;
const MIN_TRACKING_TURNS: i32 = 24;
const MAX_TRACKING_TURNS: i32 = 48;

const FLIGHT_PATH_TURNS: i32 = 8;
const MIN_FLIGHT_TURNS: i32 = 16;
const MAX_FLIGHT_TURNS: i32 = 64;

const FOLLOW_TURNS: f64 = 0.5;
const WANDER_TURNS: f64 = 2.0;

//////////////////////////////////////////////////////////////////////////////

// Timer:

struct Timer {
    active: bool,
    cur: i32,
    max: i32,
}

impl Timer {
    fn new(rng: &mut RNG, max: i32) -> Self {
        Self { active: false, cur: rng.random_range(0..=max), max }
    }

    fn percent(&self) -> i64 {
        (100 * (self.max - self.cur) / max(self.max, 1)) as i64
    }

    fn update(&mut self, delta: i32) {
        self.cur = clamp(self.cur + delta, 0, self.max);
        if self.cur == self.max { self.active = false; }
        if self.cur == 0 { self.active = true; }
    }
}


//////////////////////////////////////////////////////////////////////////////

// Blackboard (i.e. persistent state):

struct Blackboard {
    dirs: CachedDirs,
    path: CachedPath,
    threats: ThreatState,
    flight: Option<FlightState>,
    target: Option<ChaseState>,

    prev_time: Timestamp,
    turn_time: Timestamp,
    last_scan: Timestamp,
    last_warning: Timestamp,

    assess: Timer,
    hunger: Timer,
    thirst: Timer,
    weariness: Timer,

    chasing_enemy: bool,
    finding_food_: bool,
    finding_water: bool,
    getting_rest_: bool,

    last_seen: HashMap<PathKind, Point>,
}

impl Blackboard {
    fn new(rng: &mut RNG) -> Self {
        let mut result = Self {
            dirs: Default::default(),
            path: Default::default(),
            threats: Default::default(),
            flight: None,
            target: None,

            prev_time: Timestamp::default(),
            turn_time: Timestamp::default(),
            last_scan: Timestamp::default(),
            last_warning: Timestamp::default(),

            assess: Timer::new(rng, MAX_ASSESS),
            hunger: Timer::new(rng, MAX_HUNGER),
            thirst: Timer::new(rng, MAX_THIRST),
            weariness: Timer::new(rng, MAX_WEARY_),

            chasing_enemy: false,
            finding_food_: false,
            finding_water: false,
            getting_rest_: false,

            last_seen: HashMap::default(),
        };
        result.weariness.active = rng.random_bool(0.5);
        result
    }

    fn debug(&self, debug: &mut DebugLog, known: &Knowledge) {
        let Blackboard { dirs, path, .. } = self;
        let dirs = if dirs.kind != DirsKind::None {
            format!(" (step {}/{})", dirs.step, dirs.dirs.len())
        } else {
            "".into()
        };
        let path = if path.kind != PathKind::None {
            format!(" (step {}/{})", path.step, path.path.len())
        } else {
            "".into()
        };

        debug.append("Blackboard:");
        debug.indent(1, |debug| {
            debug.append(format!("prev_turn: {}", known.debug_time(self.prev_time)));
            debug.append(format!("last_warning: {}", known.debug_time(self.last_warning)));
            debug.append(format!("dirs: {:?}{}", self.dirs.kind, dirs));
            debug.append(format!("path: {:?}{}", self.path.kind, path));
            debug.newline();
        });

        debug.append("Timers:");
        debug.indent(1, |debug| {
            let mut show = |prefix: &str, timer: &Timer| {
                let width = 20;
                let Timer { active, cur, max } = *timer;
                let count = (width as f64 * cur as f64 / max as f64).round() as usize;
                let eq = '='.to_string().repeat(count);
                let sp = ' '.to_string().repeat(width - count);
                let suffix = if active { " *" } else { "" };
                debug.append(format!("{}[{}{}] {} / {}{}", prefix, eq, sp, cur, max, suffix));
                if active && let Some(x) = debug.lines.last_mut() { x.color = 0x80c0ff.into() }
            };
            show("assess:    ", &self.assess);
            show("hunger:    ", &self.hunger);
            show("thirst:    ", &self.thirst);
            show("weariness: ", &self.weariness);
        });
        debug.newline();

        debug.append("Last seen:");
        debug.indent(1, |debug| {
            let mut items: Vec<_> = self.last_seen.iter().collect();
            items.sort_by_key(|x| *x.0 as i32);
            for (&k, &v) in items { debug.append(format!("{:?}: {:?}", k, v)); }
        });
        debug.newline();

        if let Some(x) = &self.flight {
            debug.append("Flight:");
            debug.indent(1, |debug| {
                debug.append(format!("needs_path: {}", x.needs_path));
                debug.append(format!("since_seen: {}", x.since_seen));
                debug.append(format!("turn_limit: {}", x.turn_limit));
            });
            debug.newline();
        }

        if let Some(x) = &self.target {
            debug.append("Target:");
            debug.indent(1, |debug| {
                let Target { loc, sense, .. } = x.target;
                debug.append(format!("age: {}", known.debug_time(loc.time)));
                debug.append(format!("pos: {:?}, by {:?}", loc.pos, sense));
                debug.append(format!("bias: {:?}", x.bias));
                debug.append(format!("fresh: {}", x.fresh));
                debug.append(format!("steps: {}", x.steps));
            });
            debug.newline();
        }

        self.threats.debug(debug, known);
    }
}

//////////////////////////////////////////////////////////////////////////////

// Ctx (i.e. temporary state):

#[derive(Default)]
struct ScoredNeighborhood {
    neighborhood: Neighborhood,
    scores: HashMap<Point, i32>,
}

impl std::ops::Deref for ScoredNeighborhood {
    type Target = Neighborhood;
    fn deref(&self) -> &Self::Target { &self.neighborhood }
}

#[derive(Default)]
struct PerTickState {
    attack_request: Option<AttackRequest>,
    path_request: Option<Point>,

    reachable: ScoredNeighborhood,
    sneakable: ScoredNeighborhood,
    ran_vision: bool,

    threats: Vec<Point>,
    targets: Vec<Target>,
    had_target: bool,
}

pub struct Ctx<'a> {
    // Derived from the entity:
    me: &'a Entity,
    known: &'a Knowledge,
    pos: Point,
    dir: Delta,

    // Mutable outputs:
    action: Option<Action>,
    blackboard: &'a mut Blackboard,
    env: &'a mut AIEnv<'a>,
    tmp: PerTickState,
}

impl<'a> Ctx<'a> {
    pub fn choose_action(&mut self, action: Action) -> Result {
        self.action = Some(action);
        Result::Running
    }
}

fn safe_inv_l2(dir: Delta) -> f64 {
    if dir == dirs::NONE { return 0. };
    (dir.len_l2_squared() as f64).sqrt().recip()
}

fn any_threat_awake(ctx: &Ctx) -> bool {
    ctx.blackboard.threats.menacing.iter().any(|x| !x.asleep)
}

fn choose_random_attack(ctx: &mut Ctx) -> Option<&'static Attack> {
    let attacks = &ctx.me.species.attacks;
    if attacks.is_empty() { None } else { Some(*sample(attacks, ctx.env.rng)) }
}

fn is_hiding_place(ctx: &Ctx, point: Point) -> bool {
    is_hidden_from(ctx, point, &ctx.tmp.threats)
}

fn is_hidden_from(ctx: &Ctx, point: Point, threats: &[Point]) -> bool {
    if threats.iter().any(|&x| (x - point).len_l1() <= 1) { return false; }

    let cell = ctx.known.get(point);
    let light = !ctx.me.species.light.is_empty();
    cell.is_cover() || !(light || !cell.is_shadow_cover())
}

fn get_reach_check<'a>(ctx: &'a Ctx) -> impl Fn(Point) -> Status + use<'a> {
    let (fov, known, pos) = (&ctx.env.fov, ctx.known, ctx.pos);
    move |p: Point| match known.get(p).status() {
        Status::Occupied if (p - pos).len_l1() == 1 => Status::Blocked,
        Status::Unknown if fov.can_see(p) => Status::Free,
        x => x,
    }
}

fn get_sneak_check<'a, 'b>(ctx: &'a Ctx<'b>) -> impl Fn(Point) -> Status + use<'a, 'b> {
    let (known, pos) = (ctx.known, ctx.pos);
    move |p: Point| {
        if !is_hiding_place(ctx, p) { return Status::Blocked; }
        match known.get(p).status() {
            Status::Occupied if (p - pos).len_l1() == 1 => Status::Blocked,
            x => x
        }
    }
}

fn ensure_reachable(ctx: &mut Ctx) {
    if !ctx.tmp.reachable.visited.is_empty() { return; }

    ensure_vision(ctx);
    let (pos, check) = (ctx.pos, get_reach_check(ctx));
    ctx.tmp.reachable.neighborhood = DijkstraMap(pos, check, SEARCH_CELLS, SEARCH_LIMIT);

    if let Some(x) = &mut ctx.env.debug { x.record_neighborhood(&ctx.tmp.reachable); }
}

fn ensure_sneakable(ctx: &mut Ctx) {
    if !ctx.tmp.sneakable.visited.is_empty() { return; }

    let (pos, check) = (ctx.pos, get_sneak_check(ctx));
    ctx.tmp.sneakable.neighborhood = DijkstraMap(pos, check, HIDING_CELLS, HIDING_LIMIT);
}

fn ensure_vision(ctx: &mut Ctx) {
    if ctx.tmp.ran_vision { return; }

    let Ctx { known, pos, .. } = *ctx;
    let opacity_lookup = |x| if known.get(x).blocked() { INITIAL_VISIBILITY } else { 0 };
    let args = VisionArgs { pos, dir: dirs::NONE, opacity_lookup, };

    ctx.env.fov.compute(&args);
    ctx.tmp.ran_vision = true;
}

fn assess_directions(dirs: &[Delta], turns: (i32, i32), rng: &mut RNG) -> Vec<Delta> {
    if dirs.is_empty() { return vec![]; }

    let mut result = vec![];
    let (steps, turns) = turns;
    result.reserve((steps * turns) as usize);

    for i in 0..steps {
        let dir = dirs[i as usize % dirs.len()];
        if dir == dirs::NONE { continue; }

        let scale = 100. / dir.len_l2();
        let steps = rng.random_range(0..turns) + 1;
        let angle = Normal::new(0., ASSESS_ANGLE).unwrap().sample(rng);
        let (sin, cos) = (angle.sin(), angle.cos());

        let Delta(dx, dy) = dir;
        let rx = (cos * scale * dx as f64) + (sin * scale * dy as f64);
        let ry = (cos * scale * dy as f64) - (sin * scale * dx as f64);
        let target = Delta(rx as i32, ry as i32);
        for _ in 0..steps { result.push(target); }
    }
    result
}

//////////////////////////////////////////////////////////////////////////////

// Path target selectors:

fn SelectChaseTarget(ctx: &mut Ctx) -> Option<Point> {
    select_chase_target(ctx)
}

fn SelectExploreTarget(ctx: &mut Ctx) -> Option<Point> {
    select_explore_target(ctx)
}

fn SelectFleeTarget(ctx: &mut Ctx) -> Option<Point> {
    ensure_reachable(ctx);
    select_flight_target(ctx, /*hiding=*/false)
}

fn SelectHideTarget(ctx: &mut Ctx) -> Option<Point> {
    ensure_sneakable(ctx);
    select_flight_target(ctx, /*hiding=*/true)
}

fn select_target_linear(scores: &[(Point, f64)], env: &mut AIEnv) -> Option<Point> {
    let max = scores.iter().fold(0f64, |acc, x| acc.max(x.1));
    if max == 0. { return None; }

    let limit = (1 << 16) - 1;
    let inverse = (limit as f64) / max;
    let values: Vec<_> = scores.iter().filter_map(|&(p, score)| {
        let score = min((inverse * score).floor() as i32, limit);
        if score > 0 { Some((score, p)) } else { None }
    }).collect();
    if values.is_empty() { return None; }

    if let Some(x) = &mut env.debug { x.record_utility(&values) };

    Some(*weighted(&values, env.rng))
}

fn select_target_softmax(scores: &[(Point, f64)], env: &mut AIEnv, temp: f64) -> Option<Point> {
    if scores.is_empty() { return None; }

    let max = scores.iter().fold(std::f64::NEG_INFINITY, |acc, x| acc.max(x.1));
    let scale = ((1 << 16) - 1) as f64;
    let inv_temp = 1. / temp;
    let values: Vec<_> = scores.iter().map(|&(p, score)| {
        let value = (scale * (inv_temp * (score - max)).exp()) as i32;
        assert!(0 <= value && value < (1 << 16));
        (value, p)
    }).collect();

    if let Some(x) = &mut env.debug { x.record_utility(&values) };

    Some(*weighted(&values, env.rng))
}

fn select_explore_target(ctx: &mut Ctx) -> Option<Point> {
    let Ctx { known, pos, dir, .. } = *ctx;
    let inv_dir_l2 = safe_inv_l2(dir);

    let score = |p: Point, distance: i32| -> f64 {
        if distance == 0 { return 0.; }

        let age = known.time() - known.get(p).last_seen();
        let age_scale = 1. / (1 << 24) as f64;

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let unblocked_neighbors = dirs::ALL.iter().filter(
            |&&x| !known.get(p + x).blocked()).count();

        let bonus0 = age_scale * (age.seconds() + 1. / 16.);
        let bonus1 = unblocked_neighbors == dirs::ALL.len();
        let bonus2 = unblocked_neighbors > 0;

        let base = bonus0.min(1.) *
                   (if bonus1 {  8.0 } else { 1.0 }) *
                   (if bonus2 { 64.0 } else { 1.0 });
        base * (cos + 1.).pow(4) / (distance as f64).pow(2)
    };

    ensure_reachable(ctx);

    let scores: Vec<_> = ctx.tmp.reachable.visited.iter().map(
        |&(p, distance)| (p, score(p, distance))).collect();
    select_target_linear(&scores, ctx.env)
}

fn select_chase_target(ctx: &mut Ctx) -> Option<Point> {
    let Ctx { known, pos, dir, .. } = *ctx;
    let state = ctx.blackboard.target.as_ref()?;
    let (bias, steps, target) = (state.bias, state.steps, &state.target);

    let Location { pos: center, time } = target.loc;
    let bias = if target.sense == Sense::Smell { dirs::NONE } else { bias };

    let inv_dir_l2 = safe_inv_l2(dir);
    let inv_bias_l2 = safe_inv_l2(bias);
    let scale = 1. / DijkstraLength(dirs::E) as f64;

    let k = 1.25 * MIN_SEARCH_TURNS as f64;
    let decay = k / (k + steps as f64);

    let is_search_candidate = |p: Point| {
        if p == pos { return false; }
        let cell = known.get(p);
        !cell.blocked() && cell.last_see_entity_at() <= time
    };
    if is_search_candidate(center) { return Some(center); }

    let score = |p: Point, distance: i32| -> Option<f64> {
        if !is_search_candidate(p) { return None; }

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos0 = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let cos1 = delta.dot(bias) as f64 * inv_delta_l2 * inv_bias_l2;

        let d0 = scale * distance as f64;
        let d1 = (p - center).len_l2();
        let n = if known.get(p).unknown() { 0 } else {
            dirs::ALL.iter().filter(|&&x| is_search_candidate(p + x)).count()
        };
        Some(-1.0 * d0 + -6.0 * d1 * decay + 12.0 * cos0 + 15.0 * cos1 + 4.0 * n as f64)
    };

    ensure_reachable(ctx);

    let n = &ctx.tmp.reachable.neighborhood;
    let scores: Vec<_> = n.blocked.iter().chain(&n.visited).filter_map(
        |&(p, distance)| Some((p, score(p, distance)?))).collect();
    select_target_softmax(&scores, ctx.env, 4.)
}

fn select_flight_target(ctx: &mut Ctx, hiding: bool) -> Option<Point> {
    let Ctx { known, pos, .. } = *ctx;

    let scale = 1. / DijkstraLength(dirs::E) as f64;
    let min_distance = DijkstraLength(dirs::E.scale(FOV_RADIUS_NPC));
    let threats = &ctx.blackboard.threats.menacing;
    let first = threats.iter().next()?.pos;

    let score = |p: Point, source_distance: i32| -> (f64, bool) {
        let mut threat = first;
        let mut threat_distance = std::i32::MAX;
        for x in threats {
            let z = DijkstraLength(p - x.pos);
            if z < threat_distance { (threat, threat_distance) = (x.pos, z); }
        }

        let blocked = (threat - p).len_l1() > 1 && {
            let los = LOS(threat, p);
            los[1..los.len() - 1].iter().any(|&x| known.get(x).blocked())
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
    let n = if hiding { &ctx.tmp.sneakable.visited } else { &ctx.tmp.reachable.visited };
    let scores: Vec<_> = n.iter().filter_map(|&(p, distance)| {
        let (score, valid) = score(p, distance);
        if valid && score >= min_score { Some((p, score)) } else { None }
    }).collect();
    select_target_softmax(&scores, ctx.env, 0.1)
}

//////////////////////////////////////////////////////////////////////////////

// Basic state updates:

fn TickBasicNeeds(ctx: &mut Ctx) -> Result {
    let (bb, me) = (&mut *ctx.blackboard, ctx.me);
    bb.prev_time = bb.turn_time;
    bb.turn_time = me.known.time();

    let delta = if me.asleep { 0 } else { -1 };
    bb.assess.update(delta);
    bb.hunger.update(delta);
    bb.thirst.update(delta);
    bb.weariness.update(delta);

    Result::Failed
}

fn RunCombatAnalysis(ctx: &mut Ctx) -> Result {
    let (bb, me) = (&mut *ctx.blackboard, ctx.me);
    bb.threats.update(me);

    let threats = &bb.threats.menacing;
    ctx.tmp.threats = threats.iter().map(|x| x.pos).collect();

    Result::Failed
}

fn ForceThreatState(ctx: &mut Ctx, state: FightOrFlight) {
    let threats = &mut ctx.blackboard.threats;
    if threats.state != FightOrFlight::Safe { threats.state = state; }
}

//////////////////////////////////////////////////////////////////////////////

// Last-seen cache:

fn CheckLastSeen(ctx: &mut Ctx, kind: PathKind) -> bool {
    if kind == PathKind::Leader { return true; }
    ctx.blackboard.last_seen.contains_key(&kind)
}

fn ClearLastSeen(ctx: &mut Ctx, kind: PathKind) {
    if kind == PathKind::Leader { return; }
    ctx.blackboard.last_seen.remove(&kind);
}

fn UpdateLastSeen(ctx: &mut Ctx, kind: PathKind, valid: CellPredicate) -> Result {
    for cell in &ctx.known.cells {
        if !cell.visible() { break; }
        if !valid(ctx, cell.point) { continue; }

        ctx.blackboard.last_seen.insert(kind, cell.point);

        // If we spot a path that's a clear improvement, switch to it.
        let Ctx { known, pos, .. } = *ctx;
        let path = &mut ctx.blackboard.path;
        if path.kind == kind && let Some(&target) = path.path.last() &&
           (cell.point - pos).len_l2_squared() < (target - pos).len_l2_squared() {
            let los = LOS(ctx.pos, cell.point);
            if PathIsFree(known, &los) { path.replace(los); }
        }
        return Result::Success;
    }

    let Some(&last) = ctx.blackboard.last_seen.get(&kind) else { return Result::Failed };

    if valid(ctx, last) { return Result::Running; }

    ctx.blackboard.last_seen.remove(&kind);
    Result::Failed
}

//////////////////////////////////////////////////////////////////////////////

// Wandering:

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DirsKind { Assess, Flight, Noises, Target, #[default] None }

#[derive(Default)]
struct CachedDirs {
    kind: DirsKind,
    dirs: Vec<Delta>,
    step: usize,
    used: bool,
}

impl CachedDirs {
    fn clear(&mut self) {
        *self = Default::default();
    }

    fn steps_left(&self) -> usize {
        self.dirs.len() - self.step
    }
}

fn CleanupDirs(ctx: &mut Ctx) {
    let dirs = &mut ctx.blackboard.dirs;
    if !dirs.used { dirs.clear(); }
    dirs.used = false;
}

fn FollowDirs(ctx: &mut Ctx, kind: DirsKind) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    if bb.dirs.kind != kind { return None; }

    let dirs = &mut bb.dirs;
    let dir = *dirs.dirs.get(dirs.step)?;
    dirs.step += 1;

    if dirs.steps_left() == 0 {
        bb.assess.update(rng.random_range(ASSESS_GAIN));
        bb.assess.active = false;
    } else {
        dirs.used = true;
    }
    Some(Action::Look { look: dir })
}

fn Assess(ctx: &mut Ctx) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    if !bb.assess.active { return None; }

    let kind = DirsKind::Assess;
    let dirs = assess_directions(&[ctx.dir], ASSESS_TURNS_WANDER, rng);
    bb.dirs = CachedDirs { kind, dirs, step: 0, used: false };
    FollowDirs(ctx, kind)
}

fn HeardUnknownNoise(ctx: &mut Ctx) -> bool {
    let bb = &mut ctx.blackboard;
    let (pos, threats) = (ctx.pos, &bb.threats);
    let scanning = bb.dirs.kind == DirsKind::Noises;
    let limit = if scanning { bb.last_scan } else { bb.prev_time };

    let result = threats.unknown.iter().any(
        |x| x.time > limit && x.pos != pos && CALL_VOLUME.contains(x.pos - pos));
    if !result { return false; }

    if bb.dirs.kind == DirsKind::Noises && bb.dirs.steps_left() == 1 {
        for threat in &mut bb.threats.threats { threat.mark_scanned(); }
    }
    true
}

fn LookForLastTarget(ctx: &mut Ctx) -> Option<Action> {
    let (bb, rng) = (&mut *ctx.blackboard, &mut *ctx.env.rng);
    if !ctx.tmp.had_target { return None; }

    let kind = DirsKind::Target;
    let dirs = assess_directions(&[ctx.dir], ASSESS_TURNS_FLIGHT, rng);
    bb.dirs = CachedDirs { kind, dirs, step: 0, used: false };
    FollowDirs(ctx, kind)
}

fn LookForNoises(ctx: &mut Ctx) -> Option<Action> {
    let threats = &ctx.blackboard.threats;
    let (pos, rng) = (ctx.pos, &mut ctx.env.rng);
    let limit = ctx.blackboard.prev_time;

    let dirs = threats.unknown.iter().filter(
        |x| x.time > limit && x.pos != pos && CALL_VOLUME.contains(x.pos - pos));
    let dirs = dirs.map(|x| x.pos - pos).collect::<Vec<_>>();
    let dirs = if dirs.is_empty() { &[ctx.dir] } else { dirs.as_slice() };

    let kind = DirsKind::Noises;
    let dirs = assess_directions(&dirs, ASSESS_TURNS_THREAT, rng);
    ctx.blackboard.dirs = CachedDirs { kind, dirs, step: 0, used: false };
    FollowDirs(ctx, kind)
}

fn WarnRecentThreats(ctx: &mut Ctx) -> Option<Action> {
    let bb = &mut ctx.blackboard;
    let Ctx { known, pos, .. } = *ctx;
    let stare = bb.last_warning > known.time_at_turn(WARNING_RETRY_TURNS);
    let limit = known.time_at_turn(WARNING_LIMIT_TURNS);
    let (mut call, mut scan) = (None, None);

    for threat in &mut bb.threats.threats {
        if threat.time <= limit { break; }

        if !threat.uncertain() { continue; }
        if !CALL_VOLUME.contains(threat.pos - pos) { continue; }

        let look = threat.pos - pos;
        let warn = !stare && (!threat.warned() || threat.time == known.time());

        if warn {
            call.get_or_insert(Action::Call { look, call: Call::Warning });
        } else if stare {
            scan.get_or_insert(Action::Look { look });
        }
        if warn { threat.mark_warned(ctx.me, &mut ctx.env.rng); }
        if warn { bb.last_warning = known.time(); }
    }
    call.or(scan)
}

//////////////////////////////////////////////////////////////////////////////

// Pathfinding:

type CellPredicate = fn(&Ctx, Point) -> bool;

type PathTargetSelector = fn(&mut Ctx) -> Option<Point>;

type AttackTargetSelector = fn(&mut Ctx) -> Option<AttackRequest>;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
enum PathKind {
    // High-priority actions:
    Leader,
    Hide,
    Flee,
    Chase,
    ChaseFallback,
    // Low-priority needs:
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

impl CachedPath {
    fn clear(&mut self) {
        *self = Default::default();
    }

    fn replace(&mut self, path: Vec<Point>) {
        self.path = path;
        self.step = 0;
    }
}

fn CleanupPath(ctx: &mut Ctx) {
    let path = &mut ctx.blackboard.path;
    let okay = path.path.get(path.step).cloned() == Some(ctx.pos);
    if !okay { path.clear(); }
}

fn AStarHelper(ctx: &mut Ctx, target: Point, kind: PathKind) -> Option<Vec<Point>> {
    // Try using A* to find the best path:
    let source = ctx.pos;
    let hiding = kind == PathKind::Hide;
    let result = if hiding {
        AStar(source, target, ASTAR_CELLS_WANDER, get_sneak_check(ctx))
    } else {
        AStar(source, target, ASTAR_CELLS_WANDER, get_reach_check(ctx))
    };
    if let Some(mut path) = result {
        path.insert(0, source);
        return Some(path);
    }

    // If that fails, recover a path from the Dijkstra neighborhood:
    let cells = if hiding { &mut ctx.tmp.sneakable } else { &mut ctx.tmp.reachable };
    if cells.visited.is_empty() { return None; }

    // Lazily construct a table of neighborhood's scores:
    let scores = &mut cells.scores;
    if scores.is_empty() { *scores = cells.neighborhood.visited.iter().map(|&x| x).collect(); }

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

// Helpers used to compute a new path:

fn ClearFinishedPath(ctx: &mut Ctx) -> Result {
    ctx.blackboard.path.clear();
    Result::Failed
}

fn ChoosePathTarget(ctx: &mut Ctx, target: PathTargetSelector) -> bool {
    ctx.tmp.path_request = target(ctx);
    ctx.tmp.path_request.is_some()
}

fn FindPathToTarget(ctx: &mut Ctx, kind: PathKind) -> bool {
    let Some(target) = ctx.tmp.path_request.take() else { return false };

    ensure_vision(ctx);
    let path = AStarHelper(ctx, target, kind);
    let Some(path) = path else { return false };

    type K = PathKind;
    let skip = match kind {
        // Move adjacent to the cell but not onto it.
        K::Leader | K::Meat | K::Water | K::Berry | K::BerryTree => 1,

        // High-priority search/flee pathing; move to the cell.
        K::Hide | K::Flee | K::Chase | K::ChaseFallback => 0,

        // Low-priority needs pathing; move to the cell.
        K::Rest | K::Explore | K::None => 0,
    };
    ctx.blackboard.path = CachedPath { kind, path, skip, step: 0 };
    true
}

// Path validity, broken down for debugging:

fn CheckPathKind(ctx: &Ctx, kind: PathKind) -> bool {
    ctx.blackboard.path.kind == kind
}

fn CheckPathTarget(ctx: &Ctx, valid: CellPredicate) -> bool {
    ctx.blackboard.path.path.last().map_or(false, |&x| valid(ctx, x))
}

fn CheckPathStepsFree(ctx: &Ctx) -> bool {
    // Check if every cell on the path is free. Other than the cell that we'll
    // move to next, we allow entities to temporarily move onto the path.
    let path = &ctx.blackboard.path;
    let steps = path.path.iter().skip(path.step + 1).rev().skip(path.skip).rev();
    let known = ctx.known;

    steps.enumerate().all(|(i, &x)| match known.get(x).status() {
        Status::Free | Status::Unknown => true,
        Status::Occupied => i != 0,
        Status::Blocked => false,
    })
}

fn CheckPathStepsHidden(ctx: &Ctx) -> bool {
    // If we're sneaking, check that all cells are valid hiding places.
    let path = &ctx.blackboard.path;
    if path.kind != PathKind::Hide { return true; }
    path.path.iter().skip(path.step).all(|&x| is_hiding_place(ctx, x))
}

// Follow a valid path. Succeeds if we've made it to the end of the path.

fn FollowPath(ctx: &mut Ctx) -> Result {
    let Ctx { known, pos, .. } = *ctx;
    let path = &ctx.blackboard.path;
    let kind = path.kind;

    let n = path.path.len();
    let (i, j) = (path.step, path.step + 1);
    let Some(&prev) = path.path.get(i) else { return Result::Success };
    let Some(&next) = path.path.get(j) else { return Result::Success };

    // To move adjacent to a cell, look instead of taking the last step.
    if j + path.skip >= n {
        if known.get(next).visible() { return Result::Success; }
        return ctx.choose_action(Action::Look { look: next - prev });
    }

    // If `next` is occupied, look at it and discard the rest of the path.
    let cell = known.get(next);
    if !cell.visible() && matches!(cell.status(), Status::Blocked | Status::Occupied) {
        ctx.blackboard.path.step += 1;
        return ctx.choose_action(Action::Look { look: next - prev });
    }

    // Else, follow the path, look as far ahead as posisble without seeing
    // any obstructions, and update our current step.
    //
    // Special case: if we're moving next to a cell (including when we're
    // chasing an enemy), try to combine the last Move and Look actions.
    let mut target = next;
    for &point in path.path.iter().skip(j).take(8) {
        let free = pos != point && {
            let los = LOS(pos, point);
            los[1..los.len() - 1].iter().all(|&x| known.get(x).unblocked())
        };
        if free { target = point; }
    }
    if (IsChasePathKind(kind) || path.skip > 0) && n == j + 2 {
        target = path.path[j + 1];
    }
    let (look, step) = (target - pos, next - pos);

    // Determine how fast to move on the path. Only move quickly (and noisily)
    // when fleeing from an enemy, chasing one down, or returning to a leader.
    let mut turns = WANDER_TURNS;
    if IsChasePathKind(kind) && let Some(x) = &ctx.blackboard.target {
        let limit = ctx.known.time_at_turn(MIN_SEARCH_TURNS);
        if x.target.time > limit && !x.target.slow { turns = 1. };
    } else if kind == PathKind::Flee && any_threat_awake(ctx) {
        turns = 1.;
    } else if kind == PathKind::Leader {
        turns = FOLLOW_TURNS;
    }

    ctx.blackboard.path.step += 1;
    ctx.choose_action(Action::Move { look, step, turns })
}

//////////////////////////////////////////////////////////////////////////////

// Attacking:

enum Choice { Attack(&'static Attack), Return }

struct AttackRequest { choice: Choice, target: Point }

// Target selection:

fn AttackEnemy(ctx: &mut Ctx) -> Option<AttackRequest> {
    let state = ctx.blackboard.target.as_ref()?;
    let target = state.target.pos;

    if !state.target.sure { return None; }
    if state.target.time != ctx.known.time() { return None; }
    if !ctx.known.get(target).visible() { return None; }

    let attack = state.attack.or_else(|| choose_random_attack(ctx))?;
    Some(AttackRequest { choice: Choice::Attack(attack), target })
}

fn AttackRival(ctx: &mut Ctx) -> Option<AttackRequest> {
    for other in &ctx.known.entities {
        if !other.visible() { break; }
        if !other.rival() { continue; }

        let target = other.pos;
        let attack = choose_random_attack(ctx)?;
        return Some(AttackRequest { choice: Choice::Attack(attack), target });
    }
    None
}

fn AttackPathTarget(ctx: &mut Ctx) -> Option<AttackRequest> {
    let target = *ctx.blackboard.path.path.last()?;
    if !ctx.known.get(target).visible() { return None; }

    let attack = choose_random_attack(ctx)?;
    Some(AttackRequest { choice: Choice::Attack(attack), target })
}

// Attack execution:

fn ChooseAttackTarget(ctx: &mut Ctx, target: AttackTargetSelector) -> bool {
    ctx.tmp.attack_request = target(ctx);
    ctx.tmp.attack_request.is_some()
}

fn AttackChosenTarget(ctx: &mut Ctx) -> Option<Action> {
    let AttackRequest { choice, target } = ctx.tmp.attack_request.take()?;
    match choice {
        Choice::Attack(attack) => AttackTarget(ctx, target, attack),
        Choice::Return => PathToTarget(ctx, target, SUMMON_RANGE, /*flip=*/true),
    }
}

fn AttackTarget(ctx: &mut Ctx, target: Point, attack: &'static Attack) -> Option<Action> {
    let Ctx { known, me, pos, .. } = *ctx;

    let range = attack.range;
    let ready = move_ready(me) && known.get(target).visible() &&
                CanAttackFrom(known, pos, target, range);

    if ready {
        Some(Action::Attack { target, attack })
    } else {
        PathToTarget(ctx, target, range, /*flip=*/false)
    }
}

fn CanAttackFrom(known: &Knowledge, source: Point, target: Point, range: Bound) -> bool {
    if source == target { return false; }
    if !range.contains(source - target) { return false; }
    PathIsFree(known, &LOS(source, target))
}

fn PathIsFree(known: &Knowledge, path: &[Point]) -> bool {
    path.iter().skip(1).rev().skip(1).all(|&p| known.get(p).status() == Status::Free)
}

fn PathToTarget(ctx: &mut Ctx, target: Point, range: Bound, flip: bool) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let update = ctx.blackboard.path.path.last().cloned() == Some(target);

    // Slight tweaks on get_reach_check, etc. to better handle small crowds.
    let step = |dir| {
        let look = target - pos - dir;
        Action::Move { step: dir, look, turns: 1. }
    };
    let check = |p| match known.get(p).status() {
        Status::Occupied if (p - pos).len_l1() == 1 => Status::Blocked,
        x => x
    };
    let valid = |p| {
        let cell = known.get(p);
        if p != pos && cell.status() != Status::Free { return false; }

        let (a, b) = if flip { (target, p) } else { (p, target) };
        if !CanAttackFrom(known, a, b, range) { return false; }

        !flip || !is_hidden_from(ctx, p, &[target])
    };

    // Given a non-empty list of "good" directions (each of which maintains
    // line-of-sight to the target), choose one closest to our attack range.
    let pick = |dirs: &[Delta], rng: &mut RNG| {
        let cell = known.get(target);
        let mut radius = range.radius;

        // Check for any of several reasons to move closer to a target entity.
        // These reasons don't apply to cells, because cells don't move.
        if cell.occupied() {
            let shade = cell.shade();
            let cover = cell.is_cover();
            let light = ctx.me.species.light.radius;

            radius = min(radius, FOLLOW_RANGE.radius);
            if shade { radius = min(radius, max(light - 1, 1)); }
            if cover { radius = min(radius, 1); }
        }
        let radius = radius;

        assert!(!dirs.is_empty());
        let scores: Vec<_> = dirs.iter().map(
            |&x| ((x + pos - target).bound_radius() - radius).abs()).collect();
        let best = *scores.iter().reduce(|acc, x| min(acc, x)).unwrap();
        let opts: Vec<_> = (0..dirs.len()).filter(|&i| scores[i] == best).collect();
        dirs[*sample(&opts, rng)]
    };

    // If we could already attack the target, don't move out of view.
    if valid(pos) {
        let dirs: Vec<_> = [dirs::NONE].iter().chain(
            dirs::ALL.iter().filter(|&&x| valid(pos + x))).copied().collect();
        let dir = pick(&dirs, ctx.env.rng);
        if update { ctx.blackboard.path.replace(LOS(pos + dir, target)); }
        return Some(step(dir))
    }

    // Find the closest `source` cell from which we could attack the target.
    let source = Dijkstra(pos, valid, ASTAR_CELLS_ATTACK, check, |_| 0);
    let source = source.and_then(|x| x.last().cloned()).unwrap_or(target);

    // Then, use A* to find a path to that cell.
    let mut path = AStar(pos, source, ASTAR_CELLS_ATTACK, check)?;
    let dir = *path.first()? - pos;

    if update {
        let (s, t) = (source, target);
        if s != t { path.extend(LOS(s, t).into_iter().skip(1)); }
        ctx.blackboard.path.replace(path);
    }
    Some(step(dir))
}

//////////////////////////////////////////////////////////////////////////////

// Basic needs:

fn HungryForMeat(ctx: &Ctx) -> bool {
    ctx.me.species.predator() && ctx.blackboard.hunger.cur < HUNGRY_FOR_MEAT
}

fn Hunger(ctx: &mut Ctx) -> i64 {
    if !ctx.blackboard.hunger.active { return -1; }
    if ctx.blackboard.finding_food_ { return 101; }
    ctx.blackboard.hunger.percent()
}

fn Thirst(ctx: &mut Ctx) -> i64 {
    if !ctx.blackboard.thirst.active { return -1; }
    if ctx.blackboard.finding_water { return 101; }
    ctx.blackboard.thirst.percent()
}

fn Weariness(ctx: &mut Ctx) -> i64 {
    if !ctx.blackboard.weariness.active { return -1; }
    if ctx.blackboard.getting_rest_ { return 101; }
    ctx.blackboard.weariness.percent()
}

fn IsLeader(ctx: &Ctx, point: Point) -> bool {
    ctx.env.leader.map_or(false, |x| x.pos == point)
}

fn HasMeat(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map_or(false, |x| x.items.contains(&Item::Corpse))
}

fn HasBerry(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map_or(false, |x| x.items.contains(&Item::Berry))
}

fn HasWater(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map_or(false, |x| x.tile.can_drink())
}

fn HasBerryTree(ctx: &Ctx, point: Point) -> bool {
    ctx.known.get(point).cell().map_or(false, |x| x.tile.drops_berries())
}

fn CanRestAt(ctx: &Ctx, point: Point) -> bool {
    if !is_hiding_place(ctx, point) { return false; }
    point == ctx.pos || ctx.known.get(point).status() == Status::Free
}

fn FindMatchingNeighbor(ctx: &mut Ctx, valid: CellPredicate) -> Option<Point> {
    let Ctx { pos, dir, .. } = *ctx;
    if valid(ctx, pos) { return Some(pos); }

    let mut best = (std::f64::NEG_INFINITY, None);
    for &x in &dirs::ALL {
        if !valid(ctx, pos + x) { continue; }
        let score = (dir.dot(x) as f64).pow(2) / x.len_taxicab() as f64;
        if score > best.0 { best = (score, Some(pos + x)); }
    }
    best.1
}

fn FindMatchingCell(ctx: &mut Ctx, kind: PathKind, valid: CellPredicate) -> bool {
    let success = |ctx: &mut Ctx, point: Point| {
        ctx.tmp.path_request = Some(point);
        true
    };

    if let Some(point) = FindMatchingNeighbor(ctx, valid) {
        return success(ctx, point);
    }

    ensure_reachable(ctx);
    let n = &ctx.tmp.reachable.neighborhood;
    for &(point, _) in n.blocked.iter().chain(&n.visited) {
        if valid(ctx, point) { return success(ctx, point); }
    }

    if let Some(point) = ctx.blackboard.last_seen.get(&kind).copied() {
        return success(ctx, point);
    }
    false
}

fn EatMeatNow(ctx: &mut Ctx) -> Option<Action> {
    let target = *ctx.blackboard.path.path.last()?;
    let dir = target - ctx.pos;

    ctx.blackboard.hunger.update(MAX_HUNGER);

    Some(Action::Eat { dir, item: Some(Item::Corpse) })
}

fn EatBerryNow(ctx: &mut Ctx) -> Option<Action> {
    let target = *ctx.blackboard.path.path.last()?;
    let dir = target - ctx.pos;

    let prev = ctx.blackboard.hunger.cur;
    let gain = ctx.env.rng.random_range(HUNGER_GAIN);
    ctx.blackboard.hunger.update(gain);

    if ctx.me.species.predator() && ctx.blackboard.hunger.cur > HUNGRY_FOR_MEAT {
        ctx.blackboard.hunger.cur = max(prev, HUNGRY_FOR_MEAT);
        ctx.blackboard.hunger.active = false;
    }

    Some(Action::Eat { dir, item: Some(Item::Berry) })
}

fn DrinkWaterNow(ctx: &mut Ctx) -> Option<Action> {
    let target = *ctx.blackboard.path.path.last()?;
    let dir = target - ctx.pos;

    let gain = ctx.env.rng.random_range(THIRST_GAIN);
    ctx.blackboard.thirst.update(gain);

    Some(Action::Drink { dir })
}

fn GetRestNow(ctx: &mut Ctx) -> Option<Action> {
    let gain = ctx.env.rng.random_range(RESTED_GAIN);
    ctx.blackboard.weariness.update(gain);

    Some(Action::Rest)
}

//////////////////////////////////////////////////////////////////////////////

// Hunting:

#[derive(Clone, Copy)]
struct Target {
    loc: Location,
    sense: Sense,
    slow: bool,
    sure: bool,
}

struct ChaseState {
    attack: Option<&'static Attack>,
    bias: Delta,
    fresh: bool,
    reset: bool,
    steps: i32,
    target: Target,
}

impl std::ops::Deref for Target {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

fn CleanupChaseState(ctx: &mut Ctx) {
    let bb = &mut ctx.blackboard;
    if std::mem::take(&mut bb.chasing_enemy) { return; }

    if IsChasePathKind(bb.path.kind) { bb.path.clear(); }
    bb.target = None;
}

fn ClearTargets(ctx: &mut Ctx) {
    ctx.tmp.targets.clear();
}

fn ChaseTargetUnchanged(ctx: &Ctx) -> bool {
    ctx.blackboard.target.as_ref().map_or(false, |x| !x.reset)
}

fn IsChasePathKind(kind: PathKind) -> bool {
    kind == PathKind::Chase || kind == PathKind::ChaseFallback
}

fn MarkSafeIfLostView(ctx: &mut Ctx) -> bool {
    if !ctx.tmp.targets.is_empty() { return false; }
    ctx.blackboard.threats.mark_safe(ctx.known.time());
    true
}

macro_rules! check_time {
    ($ctx:ident, $time:expr, $limit:expr) => {{
        let l0 = $ctx.known.time_at_turn($limit + 0);
        let l1 = $ctx.known.time_at_turn($limit + 1);
        if $time > l1 { $ctx.tmp.had_target = true; }
        $time > l0
    }}
}

fn List<T: Fn(&mut Ctx)>(t: T) -> impl Fn(&mut Ctx) -> bool {
    move |x| { let n = x.tmp.targets.len(); t(x); x.tmp.targets.len() > n }
}

// Motivation: self-defense => persistence: low

fn ListThreatsBySight(ctx: &mut Ctx) {
    for other in &ctx.blackboard.threats.hostile {
        if !check_time!(ctx, other.time, MIN_SEARCH_TURNS) { break; }

        let (loc, sense) = (other.loc, other.sense);
        let target = Target { loc, sense, slow: false, sure: other.hostile() };
        ctx.tmp.targets.push(target);
    }
}

fn ListThreatsByScent(ctx: &mut Ctx) {
    let hostile = &ctx.blackboard.threats.hostile;
    let threats: HashSet<_> = hostile.iter().filter_map(
        |x| x.species.map(|x| x as *const Species)).collect();
    let valid = |x: &ScentKnowledge| threats.contains(&(x.species as *const Species));
    ListTargetsByScent(ctx, valid, MIN_TRACKING_TURNS)
}

// Motivation: hunger => persistence: high

fn ListPreyBySight(ctx: &mut Ctx) {
    for other in &ctx.known.entities {
        if other.delta >= 0 { continue; }
        if !check_time!(ctx, other.time, MAX_SEARCH_TURNS) { break; }

        let (loc, sense) = (other.loc, other.sense);
        let target = Target { loc, sense, slow: false, sure: true };
        ctx.tmp.targets.push(target);
    }
}

fn ListPreyBySound(ctx: &mut Ctx) {
    for other in &ctx.known.sources {
        if !check_time!(ctx, other.time, MAX_SEARCH_TURNS) { break; }

        let (loc, sense) = (other.loc, other.sense);
        let target = Target { loc, sense, slow: false, sure: false };
        ctx.tmp.targets.push(target);
    }
}

fn ListPreyByScent(ctx: &mut Ctx) {
    ListTargetsByScent(ctx, |x| x.delta < 0, MAX_TRACKING_TURNS)
}

// Motivation: curiosity => persistence: low

fn ListHumansBySound(ctx: &mut Ctx) {
    let threats = &ctx.blackboard.threats;
    for other in threats.uncertain.iter().chain(threats.unknown.iter()) {
        if !check_time!(ctx, other.time, MIN_SEARCH_TURNS) { break; }

        let (loc, sense) = (other.loc, other.sense);
        let target = Target { loc, sense, slow: true, sure: false };
        ctx.tmp.targets.push(target);
    }
}

fn ListHumansByScent(ctx: &mut Ctx) {
    ListTargetsByScent(ctx, |x| x.species.human(), MIN_TRACKING_TURNS)
}

fn ListTargetsByScent<F: Fn(&ScentKnowledge) -> bool>(ctx: &mut Ctx, f: F, turns: i32) {
    for scent in &ctx.known.scents {
        if !f(scent) { continue; }
        if !check_time!(ctx, scent.time, turns) { break; }

        let (loc, sense) = (scent.loc, Sense::Smell);
        let target = Target { loc, sense, slow: true, sure: false };
        ctx.tmp.targets.push(target);
    }
}

fn SelectBestTarget(ctx: &mut Ctx) -> bool {
    let targets = &mut ctx.tmp.targets;
    if targets.is_empty() { return false; }

    let Ctx { known, pos, .. } = *ctx;
    let prev = ctx.blackboard.target.as_ref();
    let score = |target: &Target| {
        let age = known.time_to_turn(target.time);
        let bonus = target.sure as i32 as f64;
        let d0 = (target.pos - pos).len_l2();
        let d1 = prev.map_or(0.0, |x| (target.pos - x.target.pos).len_l2());
        1.0 * age - 2.0 * bonus + 0.5 * d0 + 0.25 * d1
    };

    let target = *targets.select_nth_unstable_by_key(0, |x| sortable(score(x))).1;
    SetChaseTarget(ctx, target, None);
    true
}

fn SetChaseTarget(ctx: &mut Ctx, target: Target, attack: Option<&'static Attack>) {
    let (pos, prev) = (ctx.pos, &ctx.blackboard.target);
    let recent = target.time > ctx.blackboard.prev_time;
    let change = if let Some(x) = prev { target.pos != x.target.pos } else { true };
    let fresh = change || (recent && target.sense != Sense::Smell);
    let reset = change || recent;

    let path = &mut ctx.blackboard.path;
    if reset && path.kind == PathKind::Chase { path.clear(); }

    let (bias, steps) = if !reset && let Some(x) = prev {
        (x.bias, x.steps + 1)
    } else {
        (target.pos - pos, 0)
    };
    ctx.blackboard.target = Some(ChaseState { attack, bias, fresh, reset, steps, target });
}

fn TrackEnemyByScent(ctx: &mut Ctx) -> Option<Action> {
    let state = ctx.blackboard.target.as_ref()?;
    if !state.fresh || state.target.sense != Sense::Smell { return None; }
    Some(Action::SniffAround)
}

//////////////////////////////////////////////////////////////////////////////

// Fleeing:

#[derive(Default)]
struct FlightState {
    needs_path: bool,
    since_seen: i32,
    turn_limit: i32,
}

fn CheckFlightLimit(ctx: &mut Ctx) -> bool {
    let Some(x) = &ctx.blackboard.flight else { return false };
    x.since_seen >= x.turn_limit
}

fn ClearFlightPath(ctx: &mut Ctx) -> Result {
    let Some(x) = &mut ctx.blackboard.flight else { return Result::Failed };
    x.needs_path = false;
    Result::Failed
}

fn ClearFlightState(ctx: &mut Ctx) {
    let bb = &mut ctx.blackboard;
    let fleeing = bb.path.kind == PathKind::Hide || bb.path.kind == PathKind::Flee;
    let looking = bb.dirs.kind == DirsKind::Flight;

    if fleeing { bb.path.clear(); }
    if looking { bb.dirs.clear(); }
    bb.flight = None;
}

fn UpdateFlightState(ctx: &mut Ctx) -> bool {
    let bb = &mut ctx.blackboard;
    let prev = bb.flight.take();

    // State may be Safe even if we're aware of threats, if we tried to hunt
    // them down and lost sight for long enough. See: MarkSafeIfLostView.
    let threats = &bb.threats;
    if threats.state == FightOrFlight::Safe { return false; }
    let Some(threat) = threats.menacing.first() else { return false };

    let reset = prev.is_none() || threat.time > bb.prev_time;
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
        flight.needs_path = false;
        bb.path.clear();
    }

    if looking && reset {
        flight.turn_limit = min(2 * flight.turn_limit, MAX_FLIGHT_TURNS);
        bb.dirs.clear();
    }

    if looking && !reset && bb.dirs.steps_left() == 1 {
        bb.threats.mark_safe(ctx.known.time());
    } else {
        bb.flight = Some(flight);
    }
    true
}

fn LookForThreats(ctx: &mut Ctx) -> Option<Action> {
    let threats = &ctx.blackboard.threats.menacing;
    let (pos, rng, time) = (ctx.pos, &mut *ctx.env.rng, ctx.known.time());

    let mut visible: Vec<_> = threats.iter().filter_map(
        |x| if x.time == time && x.pos != pos { Some(x.pos) } else { None }).collect();
    if !visible.is_empty() {
        let threat = *visible.select_nth_unstable_by_key(
            0, |&p| ((p - pos).len_l2_squared(), p.0, p.1)).1;
        return Some(Action::Look { look: threat - pos });
    }

    let dirs: Vec<_> = threats.iter().filter_map(
        |x| if x.pos != pos { Some(x.pos - pos) } else { None }).collect();
    let dirs = if dirs.is_empty() { &[ctx.dir] } else { dirs.as_slice() };

    let kind = DirsKind::Flight;
    let dirs = assess_directions(&dirs, ASSESS_TURNS_FLIGHT, rng);
    ctx.blackboard.dirs = CachedDirs { kind, dirs, step: 0, used: false };
    FollowDirs(ctx, kind)
}

//////////////////////////////////////////////////////////////////////////////

// Fight-or-flight:

fn CallStrength(ctx: &mut Ctx) -> i64 {
    if ctx.blackboard.threats.hostile.is_empty() { return -1; }

    if ctx.blackboard.threats.call_for_help { 2 } else { -1 }
}

fn FightStrength(ctx: &mut Ctx) -> i64 {
    if ctx.blackboard.threats.hostile.is_empty() { return -1; }

    match ctx.blackboard.threats.state {
        FightOrFlight::Safe   => -1,
        FightOrFlight::Fight  =>  1,
        FightOrFlight::Flight =>  0,
    }
}

fn FlightStrength(ctx: &mut Ctx) -> i64 {
    match ctx.blackboard.threats.state {
        FightOrFlight::Safe   => -1,
        FightOrFlight::Fight  =>  0,
        FightOrFlight::Flight =>  1,
    }
}

fn CallForHelp(ctx: &mut Ctx) -> Option<Action> {
    let threats = &mut ctx.blackboard.threats;
    threats.on_call_for_help(ctx.pos, ctx.known.time());

    let look = threats.hostile.first().map_or(ctx.dir, |x| x.pos - ctx.pos);
    Some(Action::Call { look, call: Call::Help })
}

//////////////////////////////////////////////////////////////////////////////

// Follower AI:

pub struct Follower { pub pos: Point, pub moves: TileFlags }

pub fn rivals(me: &Entity) -> Vec<Follower> {
    let mut result = HashMap::default();
    for other in &me.known.entities {
        if !other.visible() { break; }
        if other.friend() { continue; }
        *result.entry(other.pos).or_default() |= other.species.moves;
    }
    for sound in &me.known.sources {
        result.entry(sound.pos).or_insert(TileFlags::AllMoves);
    }
    let pos = me.pos;
    let mut result: Vec<_> = result.into_iter().map(
        |(pos, moves)| Follower { pos, moves }).collect();
    result.sort_by_cached_key(|x| ((x.pos - pos).len_l2_squared(), x.pos.0, x.pos.1));
    result
}

// Check if `point` is a valid cell for a follower of the `leader`.
pub fn CheckFollowerSquare(leader: &Entity, follower: &Follower, point: Point) -> bool {
    let delta = leader.pos - point;
    if !Bound::new(2).contains(delta) { return false; }

    let known = &*leader.known;
    let cell = known.get(point);
    let free = match cell.status_for(follower.moves) {
        Status::Free => true,
        Status::Occupied => point == follower.pos,
        Status::Blocked | Status::Unknown => false,
    };
    if !free { return false }

    if delta.len_l1() <= 1 { return true; }

    cell.can_see_entity_at() && cell.visibility() == known.get(leader.pos).visibility()
}

// Choose the best cell from which to defend `leader`, starting from `source`.
// This choice may fail, e.g. if there are no spots free near `leader`.
pub fn ChooseDefenseSquare(leader: &Entity, follower: &Follower) -> Option<Point> {
    let rivals = rivals(leader);
    if rivals.is_empty() { return None; }

    let (source, known) = (follower.pos, &*leader.known);

    // Score each point in a 5x5 cell centered on the leader based on how many
    // rivals' line-of-sights to the player we'd block from that cell.
    //
    // It's important that LOS starts at the rival's position, because the
    // digital line-of-sight is asymmetric. (Consider a rival positioned a
    // knight's move away from the leader.) We want to block the rival's LOS.
    let mut scores = HashMap::default();
    for rival in &rivals {
        if rival.pos == leader.pos { continue; }

        let los = LOS(rival.pos, leader.pos);
        let los = &los[1..los.len() - 1];

        let moves = rival.moves;
        let check = |&x| known.get(x).status_for(moves) == Status::Blocked;
        if los.iter().any(check) { continue; }

        let Delta(dx, dy) = rival.pos - leader.pos;
        let vertical = dy.abs() > dx.abs();
        let sign = if vertical { dy.signum() } else { dx.signum() };
        assert!(sign != 0);

        let knight = dx * dx + dy * dy == 5;
        let bonus = if knight { 4. } else { 0. };
        let shift = if vertical { dirs::E } else { dirs::S };
        let shifts: [Delta; 3] = [dirs::NONE, shift, dirs::NONE - shift];

        for shift in shifts {
            let mut defended = false;

            for &p in los {
                let point = p + shift;
                let delta = point - leader.pos;
                let penalty = if defended { 0.0625 } else { 1.0 };

                defended = defended || (
                    point != source && point != leader.pos &&
                    known.get(point).entity().map_or(false, |x| x.friend())
                );

                if !Bound::new(2).contains(delta) { continue; }

                let score = (||{
                    if shift == dirs::NONE { return 64.; }

                    let Delta(px, py) = p - leader.pos;
                    let (a, b) = (px * dy * sign, py * dx * sign);
                    if a == b { return 6. }

                    if vertical {
                        if (a < b) == (shift.0 > 0) { 8. + bonus } else { 6. }
                    } else {
                        if (a < b) == (shift.1 < 0) { 8. + bonus } else { 6. }
                    }
                })();
                *scores.entry(delta).or_insert(0.) += score * penalty;
            }
        }
    }

    let mut best = (f64::NEG_INFINITY, None);
    for x in -2..=2 {
        for y in -2..=2 {
            if x == 0 && y == 0 { continue; }

            let (d, p) = (Delta(x, y), Delta(x, y) + leader.pos);
            if !CheckFollowerSquare(leader, follower, p) { continue; }

            let mut score = scores.get(&d).cloned().unwrap_or(f64::NEG_INFINITY);
            if score == f64::NEG_INFINITY { continue; }

            score += if d.len_l1() > 1 { 4. } else { 0. };
            score += 0.0625 * d.len_l2_squared() as f64;
            score -= 0.015625 * (p - source).len_l2_squared() as f64;
            if score > best.0 { best = (score, Some(p)); }
        }
    }
    best.1
}

// TODO: Similar bug; if all cells near the player are defended from a given
// rival, we'll just hover near them with FollowLeader, but FollowLeader's
// move doesn't look in the direction of a rival. This prevents us from
// attacking an enemy even if it's attacking us. If there are rivals, we
// should always look towards them (and not just "in our last movement dir"
// or "away from the leader" - both conditions are wrong).
//
// TODO: Another reason "DefendLeader" may fail is because the player is in
// tall grass or shadow (so the only valid follower squares are 1 cell away)
// and all of those squares are blocked or occupied. We should instead have
// a dead-simple version that considers any cell within a 5x5 centered on
// the player and picks one if more sophisticated checks fail.
//
// TODO: Choose a defense square even if a rival is adjacent to the player.
//
// TODO: Perhaps an alternate claim: MaybeAttackRivals only attacks rivals
// that are currently visible; perhaps we should path to and attack any other
// rivals the player knows about instead.
//
// TODO: Just to corroborate the point above - if the leader is beset upon by
// two enemies that are diagonally-adjacent to it, and we're supposedely
// defending the leader but we're between them, we'll ignore them. See:
//
//  E@E     E - enemy       . - open space
//  .L.     L - leader      @ - us, obviously
//
// TODO: FollowSimpleCommand is weak. If we can't find a short-term path to
// the target within ASTAR_CELLS_ATTACK, we need to switch to CachedPath-based
// long-range pathing even for relatively nearby targets.
//
// Arguably, we only need the pathing for attack-point; attack-enemy already
// works this way, and return is backed by PathToLeader.
//
// TODO: If a defender is currently the only one defending against a particular
// rival, it should not move out of the way to defend against one other rival
// (even if that square is better, e.g. because it's further from the leader).
// This "stickiness" heuristic yields more predictable behavior.
fn SelectAttackTarget(ctx: &mut Ctx) -> bool {
    let me = ctx.me;
    let Some(command) = me.command.get() else { return false };
    let Command::Attack(attack, target) = command else { return false };
    let Some(eid) = target.eid else { return false };

    let other = ctx.known.entity(eid);
    let other = other.and_then(|x| if x.time < target.loc.time { None } else { Some(x) });

    let loc = other.map_or(target.loc, |x| x.loc);
    let sense = other.map_or(Sense::Sound, |x| x.sense);

    if !check_time!(ctx, loc.time, MIN_SEARCH_TURNS) {
        me.command.take();
        return false;
    }

    if target.seen && other.is_none() {
        me.command.take();
        return false;
    }

    if !target.seen && other.is_some() {
        let target = AttackTarget { seen: true, ..target };
        me.command.set(Some(Command::Attack(attack, target)));
    }

    let target = Target { loc, sense, slow: false, sure: other.is_some() };
    SetChaseTarget(ctx, target, Some(attack));
    true
}

fn ClearAttackCommand(ctx: &mut Ctx) {
    if matches!(ctx.action, Some(Action::Attack { .. })) { ctx.me.command.take(); }
}

fn FollowSimpleCommand(ctx: &mut Ctx) -> Option<AttackRequest> {
    match ctx.me.command.get()? {
        Command::Attack(attack, AttackTarget { eid, loc, .. }) => {
            if eid.is_some() { return None; }
            Some(AttackRequest { choice: Choice::Attack(attack), target: loc.pos })
        }
        Command::Return | Command::Switch(_) => {
            Some(AttackRequest { choice: Choice::Return, target: ctx.env.leader?.pos })
        }
    }
}

fn LeaderHasRivals(ctx: &Ctx) -> bool {
    ctx.env.leader.map_or(false, |x| !rivals(x).is_empty())
}

fn DefendLeader(ctx: &mut Ctx) -> Option<Action> {
    let source = ctx.pos;
    let leader = ctx.env.leader?;
    let follower = Follower { pos: ctx.pos, moves: ctx.me.species.moves };
    let target = ChooseDefenseSquare(leader, &follower)?;

    let known = &*ctx.known;
    let check = |p: Point| known.get(p).status();
    let path = AStar(source, target, ASTAR_CELLS_ATTACK, check)?;

    let Some(&next) = path.first() else {
        let (step, look) = (dirs::NONE, source - leader.pos);
        return Some(Action::Move { step, look, turns: FOLLOW_TURNS });
    };

    let (step, look) = (next - source, next - leader.pos);
    Some(Action::Move { step, look, turns: FOLLOW_TURNS })
}

fn FollowLeader(ctx: &mut Ctx) -> Option<Action> {
    let leader = ctx.env.leader?;
    let (source, target) = (ctx.pos, leader.pos);
    if !Bound::new(3).contains(source - target) { return None; }

    let moves = ctx.me.species.moves;
    let follower = Follower { pos: ctx.pos, moves };
    let valid = |p: Point| CheckFollowerSquare(leader, &follower, p);

    let mut moves: Vec<_> = dirs::ALL.iter().filter_map(
        |&x| if valid(source + x) { Some((1, x)) } else { None }).collect();
    if valid(source) { moves.push((16, dirs::NONE)); }
    if moves.is_empty() { return None; }

    let step = *weighted(&moves, ctx.env.rng);
    Some(Action::Move { look: step, step, turns: FOLLOW_TURNS })
}

fn MoveToLeader(ctx: &mut Ctx) -> Option<Action> {
    let leader = ctx.env.leader?;
    let (known, source, target) = (ctx.known, ctx.pos, leader.pos);

    let check = |p: Point| known.get(p).status();
    let path = AStar(source, target, ASTAR_CELLS_ATTACK, check)?;
    let step = *path.first()? - source;
    Some(Action::Move { look: step, step, turns: FOLLOW_TURNS })
}

//////////////////////////////////////////////////////////////////////////////

// Behavior tree configuration

// TODO list:
//
//  - Last-seen cache for cells satisfying a need, to skip repeated searches.
//    We have this cache, now, but we should add a "last failed" time too.
//
//  - If FindMatchingCell pulled from the cache, and we can't find a path
//    all the way to the cache target, path as close as possible and re-plan
//    when we're near it again. Or: generalize this fallback to all "path to
//    target" cases, and drop the first bullet above.
//
//  - Make the our-team-strength logic quadratic in team size.

macro_rules! path {
    ($n:expr, $k:expr, $v:expr, $f:expr) => {
        seq![$n, ComputePath($k, $v), cb!("FollowPath", FollowPath), $f]
    };
}

fn Attack(name: &'static str, target: AttackTargetSelector) -> impl Bhv {
    seq![
        name,
        cond!("ChooseAttackTarget", move |x| ChooseAttackTarget(x, target)),
        act!("AttackChosenTarget", move |x| AttackChosenTarget(x)),
    ]
}

fn Follow(name: &'static str, kind: PathKind) -> impl Bhv {
    seq![
        name,
        CheckPath(kind, |_, _| true),
        cb!("FollowPath", FollowPath),
        cb!("ClearFinishedPath", ClearFinishedPath),
    ]
}

fn Search(name: &'static str, kind: PathKind, target: PathTargetSelector) -> impl Bhv {
    seq![
        name,
        cond!("ChoosePathTarget", move |x| ChoosePathTarget(x, target)),
        cond!("FindPathToTarget", move |x| FindPathToTarget(x, kind)),
        cb!("FollowPath", FollowPath),
        cb!("ClearFinishedPath", ClearFinishedPath),
    ]
}

fn Flight(name: &'static str, kind: PathKind, target: PathTargetSelector) -> impl Bhv {
    seq![
        name,
        cond!("ChoosePathTarget", move |x| ChoosePathTarget(x, target)),
        cond!("FindPathToTarget", move |x| FindPathToTarget(x, kind)),
        cb!("FollowPath", FollowPath),
        act!("LookForThreats", LookForThreats),
    ]
}

fn CheckPath(kind: PathKind, valid: CellPredicate) -> impl Bhv {
    seq![
        "CheckPath",
        cond!("CheckPathKind", move |x| CheckPathKind(x, kind)),
        seq![
            "CheckPathValidity",
            cond!("CheckPathTarget", move |x| CheckPathTarget(x, valid)),
            cond!("CheckPathStepsFree", |x| CheckPathStepsFree(x)),
            cond!("CheckPathStepsHidden", |x| CheckPathStepsHidden(x)),
        ]
        .on_fail(move |x| x.blackboard.path.clear())
    ]
}

fn ComputePath(kind: PathKind, valid: CellPredicate) -> impl Bhv {
    seq![
        "ComputePath",
        cond!("CheckLastSeen", move |x| CheckLastSeen(x, kind)),
        pri![
            "TryOldOrNewPath",
            CheckPath(kind, valid),
            seq![
                "FindNewPath",
                cond!("FindMatchingCell", move |x| FindMatchingCell(x, kind, valid)),
                cond!("FindPathToTarget", move |x| FindPathToTarget(x, kind)),
            ],
        ]
        .on_fail(move |x| ClearLastSeen(x, kind)),
    ]
}

fn ForageForBerries() -> impl Bhv {
    const KIND: PathKind = PathKind::BerryTree;
    seq![
        "ForageForBerries",
        ComputePath(KIND, HasBerryTree),
        pri![
            "AttackBerryTree",
            Attack("AttackPathTarget", AttackPathTarget),
            cb!("FollowPath", FollowPath),
            cb!("ClearFinishedPath", ClearFinishedPath),
        ],
    ]
}

fn EatBerries() -> impl Bhv {
    const KIND: PathKind = PathKind::Berry;
    path!("EatBerries", KIND, HasBerry, act!("EatBerryNow", EatBerryNow))
}

fn EatFood() -> impl Bhv {
    pri!["EatFood", EatBerries(), ForageForBerries()]
        .on_tick(|x| x.blackboard.finding_food_ = true)
        .on_exit(|x| x.blackboard.finding_food_ = false)
}

fn DrinkWater() -> impl Bhv {
    const KIND: PathKind = PathKind::Water;
    path!("DrinkWater", KIND, HasWater, act!("DrinkWaterNow", DrinkWaterNow))
        .on_tick(|x| x.blackboard.finding_water = true)
        .on_exit(|x| x.blackboard.finding_water = false)
}

fn GetRest() -> impl Bhv {
    const KIND: PathKind = PathKind::Rest;
    path!("GetRest", KIND, CanRestAt, act!("GetRestNow", GetRestNow))
        .on_tick(|x| x.blackboard.getting_rest_ = true)
        .on_exit(|x| x.blackboard.getting_rest_ = false)
}

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
        Follow("Follow(Explore)", PathKind::Explore),
        act!("Search(Assess)", Assess),
        Search("Search(Explore)", PathKind::Explore, SelectExploreTarget),
    ]
}

fn InvestigateScents() -> impl Bhv {
    seq![
        "InvestigateScents",
        run![
            "SelectRecentScent",
            cond!("ListHumansBySound", List(ListHumansBySound)),
            cond!("ListHumansByScent", List(ListHumansByScent)),
            cond!("SelectBestTarget", SelectBestTarget),
        ],
        HuntSelectedTarget(),
    ]
    .on_tick(ClearTargets)
}

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

fn WarnOffThreats() -> impl Bhv {
    act!("WarnOffThreats", WarnRecentThreats)
}

fn LookForTarget() -> impl Bhv {
    pri![
        "LookForTarget",
        act!("Follow(Target)", |x| FollowDirs(x, DirsKind::Target)),
        act!("Search(Target)", LookForLastTarget),
    ]
}

fn ChaseDownTarget() -> impl Bhv {
    pri![
        "ChaseDownTarget",
        seq![
            "SkipRedundantSearch",
            cond!("ChaseTargetUnchanged", |x| ChaseTargetUnchanged(x)),
            Follow("Follow(ChaseFallback)", PathKind::ChaseFallback),
        ],
        Follow("Follow(Chase)", PathKind::Chase),
        Search("Search(Chase)", PathKind::Chase, SelectChaseTarget),
        Follow("Follow(ChaseFallback)", PathKind::ChaseFallback),
        Search("Search(ChaseFallback)", PathKind::ChaseFallback, SelectExploreTarget),
    ]
}

fn HuntSelectedTarget() -> impl Bhv {
    pri![
        "HuntSelectedTarget",
        Attack("AttackEnemy", AttackEnemy),
        act!("TrackEnemyByScent", TrackEnemyByScent),
        ChaseDownTarget(),
    ]
    .on_running(|x| x.blackboard.chasing_enemy = true)
}

fn HuntForMeat() -> impl Bhv {
    const KIND: PathKind = PathKind::Meat;
    seq![
        "HuntForMeat",
        cond!("HungryForMeat", |x| HungryForMeat(x)),
        pri![
            "HuntForMeat",
            path!("EatMeat", KIND, HasMeat, act!("EatMeatNow", EatMeatNow)),
            seq![
                "HuntForPrey",
                run![
                    "SelectPreyTarget",
                    cond!("ListPreyBySight", List(ListPreyBySight)),
                    cond!("ListPreyBySound", List(ListPreyBySound)),
                    cond!("ListPreyByScent", List(ListPreyByScent)),
                    cond!("SelectBestTarget", SelectBestTarget),
                ],
                HuntSelectedTarget(),
            ]
            .on_tick(ClearTargets)
        ],
    ]
}

fn FightAgainstThreats() -> impl Bhv {
    seq![
        "FightAgainstThreats",
        run![
            "SelectThreatTarget",
            cond!("ListThreatsBySight", List(ListThreatsBySight)),
            cond!("ListThreatsByScent", List(ListThreatsByScent)),
            cond!("MarkSafeIfLostView", MarkSafeIfLostView),
            cond!("SelectBestTarget", SelectBestTarget),
        ],
        HuntSelectedTarget(),
    ]
    .on_tick(ClearTargets)
    .on_tick(|x| ForceThreatState(x, FightOrFlight::Fight))
}

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
            Follow("Follow(Hide)", PathKind::Hide),
            Follow("Follow(Flee)", PathKind::Flee),
            cb!("ClearFlightPath", ClearFlightPath),
            seq![
                "TryHiding",
                cond!("AnyThreatsAwake", |x| any_threat_awake(x)),
                cond!("CurrentlyHidden", |x| is_hiding_place(x, x.pos)),
                cond!("NotPanicking", |x| !x.blackboard.threats.panicking),
                Flight("HideFromThreats", PathKind::Hide, SelectHideTarget),
            ],
            seq![
                "TryFleeing",
                Flight("FleeFromThreats", PathKind::Flee, SelectFleeTarget),
            ],
        ],
    ]
    .on_tick(|x| ForceThreatState(x, FightOrFlight::Flight))
    .on_exit(ClearFlightState)
}

fn FightOrFlight() -> impl Bhv {
    util![
        "FightOrFlight",
        (CallStrength, act!("CallForHelp", CallForHelp)),
        (FightStrength, FightAgainstThreats()),
        (FlightStrength, EscapeFromThreats()),
    ]
}

fn ReturnToLeader() -> impl Bhv {
    const KIND: PathKind = PathKind::Leader;
    path!("FindLeader", KIND, IsLeader, cb!("Fail", |_| Result::Failed))
}

fn SummonRoot() -> impl Bhv {
    seq![
        "SummonRoot",
        cond!("HasLeader", |x| x.env.leader.is_some()),
        pri![
            "SummonOptions",
            seq![
                "FollowCommands",
                cond!("HasCommand", |x| x.me.command.get().is_some()),
                pri![
                    "FollowCommand",
                    Attack("FollowSimpleCommand", FollowSimpleCommand),
                    seq![
                        "FollowAttackCommand",
                        cond!("SelectAttackTarget", SelectAttackTarget),
                        HuntSelectedTarget(),
                    ],
                ]
                .post_tick(ClearAttackCommand),
            ],
            seq![
                "AttackRivals",
                cond!("MoveReady", |x| move_ready(x.me)),
                Attack("AttackRival", AttackRival),
            ],
            seq![
                "MaybeDefendLeader",
                cond!("LeaderHasRivals", |x| LeaderHasRivals(x)),
                act!("DefendLeader", DefendLeader),
            ],
            act!("FollowLeader", FollowLeader),
            act!("MoveToLeader", MoveToLeader),
            ReturnToLeader(),
            act!("Idle", |_| Some(Action::Idle)),
        ],
    ]
}

fn Root() -> impl Bhv {
    pri![
        "Root",
        cb!("TickBasicNeeds", TickBasicNeeds),
        cb!("RunCombatAnalysis", RunCombatAnalysis),
        run![
            "UpdateLastSeen",
            cb!("SpotMeat", |x| UpdateLastSeen(x, PathKind::Meat, HasMeat)),
            cb!("SpotWater", |x| UpdateLastSeen(x, PathKind::Water, HasWater)),
            cb!("SpotBerry", |x| UpdateLastSeen(x, PathKind::Berry, HasBerry)),
            cb!("SpotBerryTree", |x| UpdateLastSeen(x, PathKind::BerryTree, HasBerryTree)),
            cb!("SpotRestArea", |x| UpdateLastSeen(x, PathKind::Rest, CanRestAt)),
            cb!("Fail", |_| Result::Failed),
        ],
        SummonRoot(),
        FightOrFlight(),
        HuntForMeat(),
        LookForTarget(),
        WarnOffThreats(),
        InvestigateNoises(),
        InvestigateScents(),
        Wander(),
    ]
    .on_tick(CleanupDirs)
    .on_tick(CleanupPath)
    .post_tick(CleanupChaseState)
}

//////////////////////////////////////////////////////////////////////////////

// Entry point:

pub struct AIEnv<'a> {
    pub leader: Option<&'a Entity>,
    pub debug: Option<&'a mut DebugFile>,
    pub fov: &'a mut Vision,
    pub rng: &'a mut RNG,
}

pub struct AIState {
    blackboard: Blackboard,
    tree: Box<dyn Bhv>,
}

impl AIState {
    pub fn new(rng: &mut RNG) -> Self {
        Self { blackboard: Blackboard::new(rng), tree: Box::new(Root()) }
    }

    pub fn get_path(&self) -> &[Point] {
        &self.blackboard.path.path
    }

    pub fn get_trace(&self, known: &Knowledge) -> Vec<DebugLine> {
        let mut debug = DebugLog { depth: 0, lines: vec![], verbose: false };
        self.tree.debug(&mut debug);
        debug.newline();
        self.blackboard.debug(&mut debug, known);
        debug.lines
    }

    pub fn notify_item(&mut self, pos: Point, item: &Item) {
        let map = &mut self.blackboard.last_seen;
        if item == &Item::Berry { map.entry(PathKind::Berry).or_insert(pos); }
        if item == &Item::Corpse { map.entry(PathKind::Meat).or_insert(pos); }
    }

    pub fn plan(&mut self, me: &Entity, env: AIEnv) -> Action {
        let known = &*me.known;
        let blackboard = &mut self.blackboard;
        let mut env = AIEnv { ..env };

        let mut ctx = Ctx {
            // Derived from the entity:
            me,
            known,
            pos: me.pos,
            dir: me.dir,

            // Mutable outputs:
            action: None,
            blackboard,
            env: &mut env,
            tmp: Default::default(),
        };
        self.tree.tick(&mut ctx);
        ctx.action.take().unwrap_or(Action::Idle)
    }
}
