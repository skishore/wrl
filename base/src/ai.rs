use std::cmp::{max, min};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::fmt::Debug;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::base::{LOS, Point, RNG, dirs, sample, weighted};
use crate::entity::Entity;
use crate::game::{Action, Move, Tile, move_ready};
use crate::knowledge::{CellKnowledge, Knowledge, Timestamp};
use crate::pathing::{AStar, BFS, DijkstraLength, DijkstraMap, Status};

//////////////////////////////////////////////////////////////////////////////

// Constants

const ASTAR_LIMIT_ATTACK: i32 = 32;
const ASTAR_LIMIT_WANDER: i32 = 1024;
const BFS_LIMIT_ATTACK: i32 = 8;
const HIDING_CELLS: i32 = 256;
const HIDING_LIMIT: i32 = 32;
const SEARCH_CELLS: i32 = 1024;
const SEARCH_LIMIT: i32 = 64;
const FOV_RADIUS: i32 = 12;

const ASSESS_ANGLE: f64 = TAU / 18.;
const ASSESS_STEPS: i32 = 4;
const ASSESS_TURNS_EXPLORE: i32 = 8;
const ASSESS_TURNS_FLIGHT: i32 = 1;

const MAX_ASSESS: i32 = 32;
const MAX_HUNGER: i32 = 1024;
const MAX_THIRST: i32 = 256;

const MIN_FLIGHT_TURNS: i32 = 16;
const MAX_FLIGHT_TURNS: i32 = 64;
const MIN_SEARCH_TURNS: i32 = 16;
const MAX_SEARCH_TURNS: i32 = 32;
const TURN_TIMES_LIMIT: usize = 64;

const SLOWED_TURNS: f64 = 2.;
const WANDER_TURNS: f64 = 2.;

//////////////////////////////////////////////////////////////////////////////

// Basic helpers

fn safe_inv_l2(point: Point) -> f64 {
    if point == Point::default() { return 0. }
    (point.len_l2_squared() as f64).sqrt().recip()
}

fn is_hiding_place(entity: &Entity, point: Point, threats: &[Point]) -> bool {
    if threats.iter().any(|&x| (x - point).len_l1() <= 1) { return false; }
    let cell = entity.known.get(point);
    cell.shade() || matches!(cell.tile(), Some(x) if x.limits_vision())
}

//////////////////////////////////////////////////////////////////////////////

// Interface

#[derive(Default)]
pub struct AIDebug {
    pub targets: Vec<Point>,
    pub utility: Vec<(Point, u8)>,
}

pub struct AIEnv<'a> {
    pub rng: &'a mut RNG,
    pub debug: Option<Box<AIDebug>>,
}

//////////////////////////////////////////////////////////////////////////////

// Path caching

#[derive(Default)]
struct CachedPath {
    steps: Vec<Point>,
    pos: Point,
}

impl Debug for CachedPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedPath").field("len", &self.steps.len()).finish()
    }
}

impl CachedPath {
    fn reset(&mut self) { self.steps.clear(); }

    fn check(&self, ctx: &Context) -> bool {
        let Some(&next) = self.steps.last() else { return false; };

        let Context { known, pos, .. } = *ctx;
        if pos != self.pos { return false; }

        let valid = |p: Point| match known.get(p).status() {
            Status::Occupied => p != next,
            Status::Blocked  => false,
            Status::Free     => true,
            Status::Unknown  => true,
        };
        self.steps.iter().all(|&x| valid(x))
    }

    fn sneak(&mut self, ctx: &Context, target: Point,
             turns: f64, threats: &[Point]) -> Option<Action> {
        let Context { entity, known, pos, .. } = *ctx;
        let check = |p: Point| {
            if !is_hiding_place(entity, p, threats) { return Status::Blocked; }
            known.get(p).status()
        };
        let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check);
        self.go(ctx, path?, turns)
    }

    fn start(&mut self, ctx: &Context, target: Point, turns: f64) -> Option<Action> {
        let Context { known, pos, .. } = *ctx;
        let check = |p: Point| known.get(p).status();
        let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check);
        self.go(ctx, path?, turns)
    }

    fn go(&mut self, ctx: &Context, path: Vec<Point>, turns: f64) -> Option<Action> {
        // TODO: AStar could be strict about neighbor statuses instead.
        // Alternatively, we could add this test to our check lambdas.
        let Context { known, pos, .. } = *ctx;
        let next = *path.first()?;
        let status = known.get(next).status();
        if status != Status::Free && status != Status::Unknown { return None; }

        self.pos = pos;
        self.steps = path;
        self.steps.reverse();
        self.follow(ctx, turns)
    }

    fn follow(&mut self, ctx: &Context, turns: f64) -> Option<Action> {
        let Context { known, pos, dir, .. } = *ctx;
        let next = self.steps.pop()?;
        self.pos = next;

        let mut target = next;
        for &point in self.steps.iter().rev().take(8) {
            if LOS(pos, point).iter().all(|&x| !known.get(x).blocked()) {
                target = point;
            }
        }
        let look = if target == pos { dir } else { target - pos };
        Some(Action::Move(Move { look, step: next - pos, turns }))
    }
}

//////////////////////////////////////////////////////////////////////////////

// Shared state

struct SharedAIState {
    till_assess: i32,
    till_hunger: i32,
    till_thirst: i32,
    time: Timestamp,
    turn_times: VecDeque<Timestamp>,
}

impl Debug for SharedAIState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedAIState")
         .field("till_assess", &self.till_assess)
         .field("till_hunger", &self.till_hunger)
         .field("till_thirst", &self.till_thirst)
         .field("time", &self.time)
         .finish()
    }
}

impl SharedAIState {
    pub fn new(rng: &mut RNG) -> Self {
        Self {
            till_assess: rng.gen_range(0..MAX_ASSESS),
            till_hunger: rng.gen_range(0..MAX_HUNGER),
            till_thirst: rng.gen_range(0..MAX_THIRST),
            time: Default::default(),
            turn_times: Default::default(),
        }
    }

    fn age_at_turn(&self, turn: i32) -> i32 {
        if self.turn_times.is_empty() { return 0; }
        self.time - self.turn_times[min(self.turn_times.len() - 1, turn as usize)]
    }

    fn update(&mut self, known: &Knowledge) {
        if self.turn_times.len() == TURN_TIMES_LIMIT { self.turn_times.pop_back(); }
        self.turn_times.push_front(self.time);
        self.time = known.time;
    }
}

//////////////////////////////////////////////////////////////////////////////

// Strategy-based planning

#[derive(Clone, Copy, Debug, Eq, Ord, Hash, PartialEq, PartialOrd)]
enum Priority { Survive, Hunt, SatisfyNeeds, Explore, Skip }

struct Context<'a, 'b> {
    // Derived from the entity.
    entity: &'a Entity,
    known: &'a Knowledge,
    pos: Point,
    dir: Point,
    // Computed by the executor during this turn.
    observations: Vec<&'a CellKnowledge>,
    neighborhood: Vec<(Point, i32)>,
    sneakable: Vec<(Point, i32)>,
    shared: &'a mut SharedAIState,
    // Mutable access to the RNG.
    env: &'a mut AIEnv<'b>,
}

trait Strategy : Debug {
    fn get_path(&self) -> &[Point];
    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i32);
    fn accept(&mut self, ctx: &mut Context) -> Option<Action>;
    fn reject(&mut self);
}

//////////////////////////////////////////////////////////////////////////////

// BasicNeedsStrategy

#[derive(Debug, Eq, PartialEq)]
enum BasicNeed { Eat, Drink }

#[derive(Debug)]
struct BasicNeedsStrategy {
    last: Option<Point>,
    need: BasicNeed,
    path: CachedPath,
    tile: &'static Tile,
}

impl BasicNeedsStrategy {
    fn new(need: BasicNeed) -> Self {
        let tile = match need {
            BasicNeed::Eat => Tile::get('%'),
            BasicNeed::Drink => Tile::get('~'),
        };
        Self { last: None, need, path: Default::default(), tile }
    }

    fn timeout(&self) -> i32 {
        match self.need {
            BasicNeed::Eat => MAX_HUNGER,
            BasicNeed::Drink => MAX_THIRST,
        }
    }

    fn turns_left<'a>(&self, ctx: &'a mut Context) -> &'a mut i32 {
        match self.need {
            BasicNeed::Eat => &mut ctx.shared.till_hunger,
            BasicNeed::Drink => &mut ctx.shared.till_thirst,
        }
    }
}

impl Strategy for BasicNeedsStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i32) {
        // Update our state, which stores up to one tile that meets this need.
        *self.turns_left(ctx) = max(*self.turns_left(ctx) - 1, 0);
        for cell in &ctx.observations {
            if cell.tile != self.tile { continue; }
            self.last = Some(cell.point);
            break;
        }

        // Clear the cached path, if it's no longer good.
        let known = ctx.known;
        let valid = |p: Point| known.get(p).tile() == Some(self.tile);
        if !(self.path.check(ctx) && valid(self.path.steps[0])) { self.path.reset(); }

        // Compute a priority for satisfying this need.
        if self.last.is_none() { return (Priority::Skip, 0); }
        let (turns_left, timeout) = (*self.turns_left(ctx), self.timeout());
        let cutoff = max(timeout / 2, 1);
        if turns_left >= cutoff { return (Priority::Skip, 0); }
        let tiebreaker = if last { -1 } else { 100 * turns_left / cutoff };
        (Priority::SatisfyNeeds, tiebreaker)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let turns = WANDER_TURNS;
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let Context { known, pos, .. } = *ctx;
        let valid = |p: Point| known.get(p).tile() == Some(self.tile);
        if valid(pos) {
            *self.turns_left(ctx) = self.timeout();
            return Some(Action::Idle);
        }

        ensure_neighborhood(ctx);

        for &(point, _) in &ctx.neighborhood {
            let cell = known.get(point);
            if cell.tile() != Some(self.tile) { continue; }
            if cell.status() != Status::Free { continue; }

            // Compute at most one path to a point in the neighborhood.
            let action = self.path.start(ctx, point, turns);
            if action.is_some() { return action; }
            break;
        }

        // Else, fall back to the last-seen satisfying cell.
        let action = self.path.start(ctx, self.last?, turns);
        if action.is_none() { self.last = None; }
        action
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

// AssessStrategy

#[derive(Default)]
struct AssessStrategy { dirs: Vec<Point> }

impl Debug for AssessStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssessStrategy")
         .field("len", &self.dirs.len())
         .finish()
    }
}

impl Strategy for AssessStrategy {
    fn get_path(&self) -> &[Point] { &[] }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i32) {
        ctx.shared.till_assess = max(ctx.shared.till_assess - 1, 0);

        if ctx.shared.till_assess > 0 { return (Priority::Skip, 0); }
        let priority = if last { Priority::SatisfyNeeds } else { Priority::Explore };
        let tiebreaker = if last { -1 } else { 0 };
        (priority, tiebreaker)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let rng = &mut ctx.env.rng;

        if self.dirs.is_empty() {
            self.dirs = assess_directions(&[ctx.dir], ASSESS_TURNS_EXPLORE, rng);
            self.dirs.reverse();
        }

        let target = self.dirs.pop()?;
        if self.dirs.is_empty() {
            ctx.shared.till_assess = rng.gen_range(0..MAX_ASSESS);
        }
        Some(Action::Look(target))
    }

    fn reject(&mut self) { self.dirs.clear(); }
}

//////////////////////////////////////////////////////////////////////////////

// ExploreStrategy

#[derive(Debug, Default)]
struct ExploreStrategy { path: CachedPath }

impl Strategy for ExploreStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i32) {
        if !self.path.check(ctx) { self.path.reset(); }

        let persist = last && !self.path.steps.is_empty();
        let tiebreaker = if persist { -1 } else { 1 };
        (Priority::Explore, tiebreaker)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let turns = WANDER_TURNS;
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let Context { known, pos, dir, .. } = *ctx;
        let inv_dir_l2 = safe_inv_l2(dir);

        ensure_neighborhood(ctx);

        let mut min_age = std::i32::MAX;
        for &(point, _) in &ctx.neighborhood {
            let age = known.get(point).time_since_seen();
            if age > 0 { min_age = min(min_age, age); }
        }
        if min_age == std::i32::MAX { min_age = 1; }

        let score = |p: Point, distance: i32| -> f64 {
            if distance == 0 { return 0.; }
            let age = known.get(p).time_since_seen();

            let delta = p - pos;
            let inv_delta_l2 = safe_inv_l2(delta);
            let cos = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
            let unblocked_neighbors = dirs::ALL.iter().filter(
                    |&&x| !known.get(p + x).blocked()).count();

            let bonus0 = 1. / 65536. * ((age as f64 / min_age as f64) + 1. / 16.);
            let bonus1 = unblocked_neighbors == dirs::ALL.len();
            let bonus2 = unblocked_neighbors > 0;

            let base = (if bonus0 > 1. { 1. } else { bonus0 }) *
                       (if bonus1 {  8.0 } else { 1.0 }) *
                       (if bonus2 { 64.0 } else { 1.0 });
            base * (cos + 1.).pow(4) / (distance as f64).pow(2)
        };

        let scores: Vec<_> = ctx.neighborhood.iter().map(
            |&(p, distance)| (p, score(p, distance))).collect();
        let target = select_target(&scores, ctx.env)?;
        self.path.start(ctx, target, turns)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct ChaseTarget {
    age: i32,
    bias: Point,
    last: Point,
    search_turns: i32,
}

#[derive(Debug, Default)]
struct ChaseStrategy {
    path: CachedPath,
    target: Option<ChaseTarget>,
}

impl Strategy for ChaseStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i32) {
        if !ctx.entity.predator { return (Priority::Skip, 0); }

        let prev = self.target.take();

        let Context { known, pos, .. } = *ctx;
        let last_turn_age = ctx.shared.age_at_turn(1);
        let limit = ctx.shared.age_at_turn(MAX_SEARCH_TURNS);

        let mut targets: Vec<_> = known.entities.iter().filter(
            |x| x.age < limit && x.rival).collect();
        if targets.is_empty() { return (Priority::Skip, 0); }

        let target = *targets.select_nth_unstable_by_key(
                0, |x| (x.age, (x.pos - pos).len_l2_squared())).1;
        let reset = target.age < last_turn_age;
        if reset || !self.path.check(ctx) { self.path.reset(); }

        let (age, last) = (target.age, target.pos);
        let (bias, search_turns) = if !reset && let Some(x) = prev {
            (x.bias, x.search_turns + 1)
        } else {
            (target.pos - pos, 0)
        };
        self.target = Some(ChaseTarget { age, bias, last, search_turns });
        (Priority::Hunt, 0)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let ChaseTarget { age, bias, last, search_turns } = *self.target.as_ref()?;
        if age == 0 { return Some(attack_target(ctx, last)); }

        let turns = {
            if !move_ready(ctx.entity) { SLOWED_TURNS }
            else if search_turns >= MIN_SEARCH_TURNS { WANDER_TURNS } else { 1. }
        };
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let search_nearby = search_turns > bias.len_l1();
        let center = if search_nearby { ctx.pos } else { last };
        search_around(ctx, &mut self.path, age, bias, center, turns)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum FlightStage { Flee, Hide, Done }

#[derive(Debug)]
struct FlightStrategy {
    dirs: Vec<Point>,
    path: CachedPath,
    stage: FlightStage,
    since_seen: i32,
    turn_limit: i32,
    threats: Vec<Point>,
}

impl FlightStrategy {
    fn done(&self) -> bool { self.stage == FlightStage::Done }

    fn look(&mut self, ctx: &mut Context) -> Option<Action> {
        let (pos, rng) = (ctx.pos, &mut ctx.env.rng);
        self.path.reset();

        if self.dirs.is_empty() {
            let dirs: Vec<_> = self.threats.iter().map(|&x| x - pos).collect();
            self.dirs = assess_directions(&dirs, ASSESS_TURNS_FLIGHT, rng);
            self.dirs.reverse();
        }

        let target = self.dirs.pop()?;
        if self.dirs.is_empty() {
            ctx.shared.till_assess = rng.gen_range(0..MAX_ASSESS);
            self.turn_limit = MIN_FLIGHT_TURNS;
            self.stage = FlightStage::Done;
        }
        Some(Action::Look(target))
    }
}

impl Default for FlightStrategy {
    fn default() -> Self {
        let (since_seen, turn_limit) = (0, MIN_FLIGHT_TURNS);
        let (path, stage) = (CachedPath::default(), FlightStage::Done);
        Self { dirs: vec![], path, stage, since_seen, turn_limit, threats: vec![] }
    }
}

impl Strategy for FlightStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i32) {
        if ctx.entity.predator { return (Priority::Skip, 0); }

        let Context { entity, known, pos, .. } = *ctx;
        let last_turn_age = ctx.shared.age_at_turn(1);
        let limit = ctx.shared.age_at_turn(MAX_FLIGHT_TURNS);

        let reset = known.entities.iter().any(|x| x.age < last_turn_age && x.rival);
        let mut threats: Vec<_> = known.entities.iter().filter_map(
            |x| if x.age < limit && x.rival { Some(x.pos) } else { None }).collect();
        threats.sort_unstable_by_key(|x| (x.0, x.1));

        if self.done() && !reset { return (Priority::Skip, 0); }

        let looking = !self.dirs.is_empty();
        let hiding = is_hiding_place(entity, pos, &threats);
        let stage = if hiding { FlightStage::Hide } else { FlightStage::Flee };

        if reset || threats != self.threats {
            self.reject();
        } else if !self.path.check(ctx) {
            self.path.reset();
        }

        self.stage = if reset { min(self.stage, stage) } else { self.stage };
        self.since_seen = if reset { 0 } else { self.since_seen + 1 };
        self.turn_limit = if reset && looking {
            min(2 * self.turn_limit, MAX_FLIGHT_TURNS)
        } else {
            self.turn_limit
        };
        if !threats.is_empty() { self.threats = threats; }

        (Priority::Survive, 0)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        if !self.dirs.is_empty() || self.since_seen >= self.turn_limit {
            self.path.reset();
            return self.look(ctx);
        }

        let turns = {
            if !move_ready(ctx.entity) { SLOWED_TURNS }
            else if self.stage == FlightStage::Hide { WANDER_TURNS } else { 1. }
        };
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let Context { entity, pos, .. } = *ctx;

        if is_hiding_place(entity, pos, &self.threats) {
            ensure_sneakable(ctx, &self.threats);

            if let Some(target) = flight_cell(ctx, &self.threats, true) {
                self.stage = FlightStage::Hide;
                if target == pos { return self.look(ctx); }
                return self.path.sneak(ctx, target, WANDER_TURNS, &self.threats);
            }
        }

        ensure_neighborhood(ctx);

        if let Some(target) = flight_cell(ctx, &self.threats, false) {
            self.stage = FlightStage::Flee;
            if target == pos { return self.look(ctx); }
            return self.path.start(ctx, target, 1.);
        }

        self.stage = FlightStage::Flee;
        self.look(ctx)
    }

    fn reject(&mut self) {
        self.dirs.clear();
        self.path.reset();
    }
}

//////////////////////////////////////////////////////////////////////////////

// Helpers for ChaseStrategy

fn attack_target(ctx: &mut Context, target: Point) -> Action {
    let Context { entity, known, pos: source, .. } = *ctx;
    if source == target { return Action::Idle; }

    let range = entity.range;
    let valid = |x| has_los(x, target, known, range);
    if move_ready(entity) && valid(source) { return Action::Attack(target); }

    path_to_target(ctx, target, range, valid)
}

fn has_los(source: Point, target: Point, known: &Knowledge, range: i32) -> bool {
    if (source - target).len_nethack() > range { return false; }
    if !known.get(target).visible() { return false; }
    let los = LOS(source, target);
    let last = los.len() - 1;
    los.iter().enumerate().all(|(i, &p)| {
        if i == 0 || i == last { return true; }
        known.get(p).status() == Status::Free
    })
}

fn path_to_target<F: Fn(Point) -> bool>(
        ctx: &mut Context, target: Point, range: i32, valid: F) -> Action {
    let Context { known, pos: source, .. } = *ctx;
    let rng = &mut ctx.env.rng;
    let check = |p: Point| known.get(p).status();
    let step = |dir: Point| {
        let look = target - source - dir;
        Action::Move(Move { step: dir, look, turns: 1. })
    };

    // Given a non-empty list of "good" directions (each of which brings us
    // close to attacking the target), choose one closest to our attack range.
    let pick = |dirs: &Vec<Point>, rng: &mut RNG| {
        let cell = known.get(target);
        let (shade, tile) = (cell.shade(), cell.tile());
        let obscured = shade || matches!(tile, Some(x) if x.limits_vision());
        let distance = if obscured { 1 } else { range };

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
        return pick(&dirs, rng);
    }

    // Else, pick a direction which brings us in view.
    let result = BFS(source, &valid, BFS_LIMIT_ATTACK, check);
    if let Some(x) = result && !x.dirs.is_empty() { return pick(&x.dirs, rng); }

    // Else, move towards the target.
    let path = AStar(source, target, ASTAR_LIMIT_ATTACK, check);
    let dir = path.and_then(
        |x| if x.is_empty() { None } else { Some(x[0] - source) });
    step(dir.unwrap_or_else(|| *sample(&dirs::ALL, rng)))
}

fn search_around(ctx: &mut Context, path: &mut CachedPath,
                 age: i32, bias: Point, center: Point, turns: f64) -> Option<Action> {
    let Context { known, pos, dir, .. } = *ctx;
    let inv_dir_l2 = safe_inv_l2(dir);
    let inv_bias_l2 = safe_inv_l2(bias);

    let score = |p: Point, distance: i32| -> f64 {
        if distance == 0 { return 0.; }
        let cell = known.get(p);
        if cell.time_since_entity_visible() < age || cell.blocked() { return 0. }

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos0 = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let cos1 = delta.dot(bias) as f64 * inv_delta_l2 * inv_bias_l2;
        let angle = ((cos0 + 1.) * (cos1 + 1.)).pow(4);
        let bonus = if p == center && center != pos { 64. } else { 1. };

        angle * bonus / (((p - center).len_l2_squared() + 1) as f64).pow(2)
    };

    ensure_neighborhood(ctx);

    let scores: Vec<_> = ctx.neighborhood.iter().map(
        |&(p, distance)| (p, score(p, distance))).collect();
    let target = select_target(&scores, ctx.env)?;
    path.start(ctx, target, turns)
}

//////////////////////////////////////////////////////////////////////////////

// Helpers for FlightStrategy

fn flight_cell(ctx: &mut Context, threats: &[Point], hiding: bool) -> Option<Point> {
    let Context { entity, known, pos, .. } = *ctx;

    let threat_inv_l2s: Vec<_> = threats.iter().map(
        |&x| (x, safe_inv_l2(pos - x))).collect();
    let scale = DijkstraLength(Point(1, 0)) as f64;

    let score = |p: Point, source_distance: i32| -> f64 {
        let mut inv_l2 = 0.;
        let mut threat = Point::default();
        let mut threat_distance = std::i32::MAX;
        for &(x, y) in &threat_inv_l2s {
            let z = DijkstraLength(x - p);
            if z < threat_distance { (threat, inv_l2, threat_distance) = (x, y, z); }
        }

        let blocked = {
            let los = LOS(p, threat);
            (1..los.len() - 1).any(|i| !known.get(los[i]).unblocked())
        };
        let frontier = dirs::ALL.iter().any(|&x| known.get(p + x).unknown());
        let hidden = !hiding && is_hiding_place(entity, p, &threats);

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos = delta.dot(pos - threat) as f64 * inv_delta_l2 * inv_l2;

        // WARNING: This heuristic can cause a piece to be "checkmated" in a
        // corner, if the Dijkstra search below isn't wide enough for us to
        // find a cell which is further from the threat than the corner cell.
        let base = 1.5 * (threat_distance as f64) +
                   -1. * (source_distance as f64) +
                   16. * scale * (blocked as i32 as f64) +
                   16. * scale * (frontier as i32 as f64) +
                   16. * scale * if hidden { 1. } else { 0. };
        (cos + 1.).pow(3) * base.pow(9)
    };

    let map = if hiding { &ctx.sneakable } else { &ctx.neighborhood };
    let scores: Vec<_> = map.iter().map(
        |&(p, distance)| (p, score(p, distance))).collect();
    select_target(&scores, ctx.env)
}

//////////////////////////////////////////////////////////////////////////////

// Helpers used to implement strategies in general

fn assess_directions(dirs: &[Point], turns: i32, rng: &mut RNG) -> Vec<Point> {
    if dirs.is_empty() { return vec![]; }

    let mut result = vec![];
    result.reserve((ASSESS_STEPS * turns) as usize);

    for i in 0..ASSESS_STEPS {
        let scale = 1000;
        let steps = rng.gen_range(0..turns) + 1;
        let angle = Normal::new(0., ASSESS_ANGLE).unwrap().sample(rng);
        let (sin, cos) = (angle.sin(), angle.cos());

        let Point(dx, dy) = dirs[i as usize % dirs.len()];
        let rx = (cos * (scale * dx) as f64) + (sin * (scale * dy) as f64);
        let ry = (cos * (scale * dy) as f64) - (sin * (scale * dx) as f64);
        let target = Point(rx as i32, ry as i32);
        for _ in 0..steps { result.push(target); }
    }
    result
}

fn ensure_neighborhood(ctx: &mut Context) {
    if !ctx.neighborhood.is_empty() { return; }
    let known = ctx.known;
    let check = |p: Point| known.get(p).status();
    ctx.neighborhood = DijkstraMap(ctx.pos, check, SEARCH_CELLS, SEARCH_LIMIT, FOV_RADIUS);
}

fn ensure_sneakable(ctx: &mut Context, threats: &[Point]) {
    if !ctx.sneakable.is_empty() { return; }
    let Context { entity, known, .. } = *ctx;
    let check = |p: Point| {
        if !is_hiding_place(entity, p, threats) { return Status::Blocked; }
        known.get(p).status()
    };
    ctx.sneakable = DijkstraMap(ctx.pos, check, HIDING_CELLS, HIDING_LIMIT, FOV_RADIUS);
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

    if let Some(x) = &mut env.debug {
        x.utility.clear();
        for &(score, point) in &values {
            x.utility.push((point, (score >> 8) as u8));
        }
    }
    Some(*weighted(&values, env.rng))
}

//////////////////////////////////////////////////////////////////////////////

// Execution

#[derive(Debug)]
pub struct AIState {
    bids: Vec<(Priority, i32, usize)>,
    strategies: Vec<Box<dyn Strategy>>,
    shared: SharedAIState,
    last: i32,
}

impl AIState {
    pub fn new(rng: &mut RNG) -> Self {
        let strategies: Vec<Box<dyn Strategy>> = vec![
            Box::new(ChaseStrategy::default()),
            Box::new(FlightStrategy::default()),
            Box::new(BasicNeedsStrategy::new(BasicNeed::Eat)),
            Box::new(BasicNeedsStrategy::new(BasicNeed::Drink)),
            Box::new(AssessStrategy::default()),
            Box::new(ExploreStrategy::default()),
        ];
        Self { bids: vec![], strategies, shared: SharedAIState::new(rng), last: -1 }
    }

    pub fn get_path(&self) -> &[Point] {
        if self.last == -1 { return &[]; }
        self.strategies[self.last as usize].get_path()
    }

    pub fn plan(&mut self, entity: &Entity, env: &mut AIEnv) -> Action {
        // Step 0: update some initial, deterministic shared state.
        let known = &*entity.known;
        let mut observations = vec![];
        for cell in &known.cells {
            if (self.shared.time - cell.last_seen) >= 0 { break; }
            observations.push(cell);
        }
        self.shared.update(known);

        // Step 1: build a context object with all strategies' mutable state.
        let shared = &mut self.shared;
        let mut ctx = Context {
            entity,
            known,
            pos: entity.pos,
            dir: entity.dir,
            observations,
            neighborhood: vec![],
            sneakable: vec![],
            shared,
            env,
        };

        // Step 2: update each strategy's state and compute priorities.
        self.bids.clear();
        for (i, strategy) in self.strategies.iter_mut().enumerate() {
            let last = i == self.last as usize;
            let (priority, tiebreaker) = strategy.bid(&mut ctx, last);
            if priority == Priority::Skip { continue; }
            self.bids.push((priority, tiebreaker, i));
        }
        self.bids.sort_unstable();

        // Step 3: execute strategies by priority order.
        self.last = -1;
        let mut action = None;
        for &(_, _, i) in &self.bids {
            action = self.strategies[i].accept(&mut ctx);
            if action.is_none() { continue; }
            self.last = i as i32;
            break;
        }

        // Step 4: clear any staged state in unused strategies.
        for (i, strategy) in self.strategies.iter_mut().enumerate() {
            let last = i == self.last as usize;
            if !last { strategy.reject(); }
        }
        action.unwrap_or(Action::Idle)
    }
}
