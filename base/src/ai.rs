use std::cmp::{max, min};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::fmt::Debug;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::base::{HashMap, HashSet, LOS, Point, RNG, dirs, sample, weighted};
use crate::entity::Entity;
use crate::game::{Action, Move, Tile, move_ready};
use crate::knowledge::{CellKnowledge, EntityKnowledge, Knowledge, Timestamp};
use crate::pathing::{AStar, BFS, BFSResult, Status};
use crate::pathing::{DijkstraSearch, DijkstraLength, DijkstraMap};

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

const MIN_RESTED: i32 = 2048;
const MAX_RESTED: i32 = 4096;

const MIN_FLIGHT_TURNS: i32 = 16;
const MAX_FLIGHT_TURNS: i32 = 64;
const MIN_SEARCH_TURNS: i32 = 16;
const MAX_SEARCH_TURNS: i32 = 32;
const TURN_TIMES_LIMIT: usize = 64;

const SLOWED_TURNS: f64 = 2.;
const WANDER_TURNS: f64 = 2.;

//////////////////////////////////////////////////////////////////////////////

// Basic actions

fn step(dir: Point, turns: f64) -> Action { Action::Move(Move { look: dir, step: dir, turns }) }

fn wander(dir: Point) -> Action { step(dir, WANDER_TURNS) }

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

    fn start(&mut self, ctx: &Context, target: Point) -> Option<Action> {
        let Context { known, pos, .. } = *ctx;
        let check = |p: Point| known.get(p).status();
        let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check);
        let Some(path) = path else { return None; };

        // TODO: AStar could be strict about neighbor statuses instead.
        let next = *path.first()?;
        let status = known.get(next).status();
        if status != Status::Free && status != Status::Unknown { return None; }

        self.pos = pos;
        self.steps = path;
        self.steps.reverse();
        self.follow(ctx)
    }

    fn follow(&mut self, ctx: &Context) -> Option<Action> {
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
        Some(Action::Move(Move { look, step: next - pos, turns: WANDER_TURNS }))
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
enum Priority { Survive, SatisfyNeeds, Hunt, Explore, Skip }

struct Context<'a, 'b> {
    // Derived from the entity.
    entity: &'a Entity,
    known: &'a Knowledge,
    pos: Point,
    dir: Point,
    // Computed by the executor during this turn.
    observations: Vec<&'a CellKnowledge>,
    neighborhood: Vec<(Point, i32)>,
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
        if let Some(x) = self.path.follow(ctx) { return Some(x); }

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
            let action = self.path.start(ctx, point);
            if action.is_some() { return action; }
            break;
        }

        // Else, fall back to the last-seen satisfying cell.
        let action = self.path.start(ctx, self.last?);
        if action.is_none() { self.last = None; }
        action
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

// AssessStrategy

#[derive(Default)]
struct AssessStrategy { steps: Vec<Point> }

impl Debug for AssessStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssessStrategy")
         .field("len", &self.steps.len())
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

        if self.steps.is_empty() {
            self.steps = assess_directions(&[ctx.dir], ASSESS_TURNS_EXPLORE, rng);
            self.steps.reverse();
        }

        let target = self.steps.pop();
        if self.steps.is_empty() {
            ctx.shared.till_assess = rng.gen_range(0..MAX_ASSESS);
        }
        target.map(|x| Action::Look(x))
    }

    fn reject(&mut self) { self.steps.clear(); }
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
        if let Some(x) = self.path.follow(ctx) { return Some(x); }

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
        self.path.start(ctx, target);
        self.path.follow(ctx)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

// Helpers used to implement strategies

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
pub struct NewAIState {
    bids: Vec<(Priority, i32, usize)>,
    strategies: Vec<Box<dyn Strategy>>,
    shared: SharedAIState,
    last: i32,
}

impl NewAIState {
    pub fn new(rng: &mut RNG) -> Self {
        let strategies: Vec<Box<dyn Strategy>> = vec![
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

//////////////////////////////////////////////////////////////////////////////

// AI

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum FlightStage { Flee, Hide, Done }

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Goal { Assess, Chase, Drink, Eat, Explore, Flee, Rest }

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum StepKind { Drink, Eat, Move, Look, Rest }

#[derive(Clone, Copy, Debug)]
struct Step { kind: StepKind, target: Point }

type Hint = (Goal, &'static Tile);

#[derive(Clone, Debug)]
struct FightState {
    age: i32,
    bias: Point,
    target: Point,
    search_turns: i32,
}

#[derive(Clone, Debug)]
struct FlightState {
    stage: FlightStage,
    threats: Vec<Point>,
    since_seen: i32,
    till_assess: i32,
}

impl FlightState {
    fn done(&self) -> bool {
        self.stage == FlightStage::Done
    }
}

#[derive(Clone, Debug)]
pub struct AIState {
    goal: Goal,
    plan: Vec<Step>,
    time: Timestamp,
    hints: HashMap<Goal, Point>,
    fight: Option<FightState>,
    flight: Option<FlightState>,
    turn_times: VecDeque<Timestamp>,
    till_assess: i32,
    till_hunger: i32,
    till_thirst: i32,
    till_rested: i32,
}

impl AIState {
    pub fn new(rng: &mut RNG) -> Self {
        Self {
            goal: Goal::Explore,
            plan: vec![],
            time: Default::default(),
            hints: Default::default(),
            fight: None,
            flight: None,
            turn_times: Default::default(),
            till_assess: rng.gen_range(0..MAX_ASSESS),
            till_hunger: rng.gen_range(0..MAX_HUNGER),
            till_thirst: rng.gen_range(0..MAX_THIRST),
            till_rested: MAX_RESTED,
        }
    }

    fn age_at_turn(&self, turn: i32) -> i32 {
        if self.turn_times.is_empty() { return 0; }
        self.time - self.turn_times[min(self.turn_times.len() - 1, turn as usize)]
    }

    fn record_turn(&mut self, time: Timestamp) {
        if self.turn_times.len() == TURN_TIMES_LIMIT { self.turn_times.pop_back(); }
        self.turn_times.push_front(self.time);
        self.time = time;
    }

    pub fn debug_plan(&self) -> Vec<Point> {
        self.plan.iter().filter_map(
            |x| if x.kind == StepKind::Look { None } else { Some(x.target) }).collect()
    }

    pub fn debug_string(&self) -> String {
        let mut copy = self.clone();
        while copy.turn_times.len() > 2 { copy.turn_times.pop_back(); }
        copy.plan.clear();
        format!("{:?}", copy)
    }
}

fn coerce(source: Point, path: &[Point]) -> BFSResult {
    if path.is_empty() {
        BFSResult { dirs: vec![dirs::NONE], targets: vec![source] }
    } else {
        BFSResult { dirs: vec![path[0] - source], targets: vec![path[path.len() - 1]] }
    }
}

fn safe_inv_l2(point: Point) -> f64 {
    if point == Point::default() { return 0. }
    (point.len_l2_squared() as f64).sqrt().recip()
}

fn sample_scored(entity: &Entity, scores: &[(Point, f64)], env: &mut AIEnv) -> Option<BFSResult> {
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
        for &(score, point) in &values {
            x.utility.push((point, (score >> 8) as u8));
        }
    }

    let target = *weighted(&values, env.rng);
    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| known.get(p).status();
    let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check)?;
    Some(coerce(pos, &path))
}

fn explore(entity: &Entity, env: &mut AIEnv) -> Option<BFSResult> {
    let (known, pos, dir) = (&*entity.known, entity.pos, entity.dir);
    let check = |p: Point| known.get(p).status();
    let map = DijkstraMap(pos, check, SEARCH_CELLS, SEARCH_LIMIT, FOV_RADIUS);
    if map.is_empty() { return None; }

    let mut min_age = std::i32::MAX;
    for &(point, _) in &map {
        let age = known.get(point).time_since_seen();
        if age > 0 { min_age = min(min_age, age); }
    }
    if min_age == std::i32::MAX { min_age = 1; }

    let inv_dir_l2 = safe_inv_l2(dir);

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

    let scores: Vec<_> = map.into_iter().map(
        |(p, distance)| (p, score(p, distance))).collect();
    sample_scored(entity, &scores, env)
}

fn search_around(entity: &Entity, source: Point, age: i32, bias: Point,
                 env: &mut AIEnv) -> Option<BFSResult> {
    let (known, pos, dir) = (&*entity.known, entity.pos, entity.dir);
    let check = |p: Point| known.get(p).status();
    let map = DijkstraMap(pos, check, SEARCH_CELLS, SEARCH_LIMIT, FOV_RADIUS);
    if map.is_empty() { return None; }

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
        let bonus = if p == source && source != pos { 64. } else { 1. };

        angle * bonus / (((p - source).len_l2_squared() + 1) as f64).pow(2)
    };

    let scores: Vec<_> = map.into_iter().map(
        |(p, distance)| (p, score(p, distance))).collect();
    sample_scored(entity, &scores, env)
}

fn has_line_of_sight(source: Point, target: Point, known: &Knowledge, range: i32) -> bool {
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
        entity: &Entity, target: Point, known: &Knowledge,
        range: i32, valid: F, rng: &mut RNG) -> Action {
    let source = entity.pos;
    let check = |p: Point| known.get(p).status();
    let step = |dir: Point| {
        let look = target - source - dir;
        Action::Move(Move { step: dir, look, turns: 1. })
    };

    // Given a non-empty list of "good" directions (each of which brings us
    // close to attacking the target), choose one closest to our attack range.
    let pick = |dirs: &Vec<Point>, rng: &mut RNG| {
        let cell = known.get(target);
        let obscured = cell.shade() || matches!(cell.tile(), Some(x) if x.limits_vision());
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
    let dir = path.and_then(|x| if x.is_empty() { None } else { Some(x[0] - source) });
    step(dir.unwrap_or_else(|| *sample(&dirs::ALL, rng)))
}

fn attack_target(entity: &Entity, target: Point, rng: &mut RNG) -> Action {
    let (known, source) = (&*entity.known, entity.pos);
    if source == target { return Action::Idle; }

    let range = entity.range;
    let valid = |x| has_line_of_sight(x, target, known, range);
    if move_ready(entity) && valid(source) {
        return Action::Attack(target);
    }
    path_to_target(entity, target, known, range, valid, rng)
}

fn assess_dirs(dirs: &[Point], turns: i32, ai: &mut AIState, rng: &mut RNG) {
    if dirs.is_empty() { return; }

    for i in 0..ASSESS_STEPS {
        let scale = 1000;
        let steps = rng.gen_range(0..turns) + 1;
        let angle = Normal::new(0., ASSESS_ANGLE).unwrap().sample(rng);
        let (sin, cos) = (angle.sin(), angle.cos());

        let Point(dx, dy) = dirs[i as usize % dirs.len()];
        let rx = (cos * (scale * dx) as f64) + (sin * (scale * dy) as f64);
        let ry = (cos * (scale * dy) as f64) - (sin * (scale * dx) as f64);
        let target = Point(rx as i32, ry as i32);
        for _ in 0..steps {
            ai.plan.push(Step { kind: StepKind::Look, target })
        }
    }
}

fn assess_nearby(entity: &Entity, ai: &mut AIState, rng: &mut RNG) {
    let mut base = |ai: &mut AIState| {
        assess_dirs(&[entity.dir], ASSESS_TURNS_EXPLORE, ai, rng);
    };

    let Some(flight) = &ai.flight else { return base(ai) };
    if flight.done() || flight.threats.is_empty() { return base(ai); }

    let hiding = flight.stage == FlightStage::Hide;
    let turns = if hiding { ASSESS_TURNS_EXPLORE } else { ASSESS_TURNS_FLIGHT };
    let dirs: Vec<_> = flight.threats.iter().map(|&x| x - entity.pos).collect();
    assess_dirs(&dirs, turns, ai, rng);
}

fn is_hiding_place(entity: &Entity, point: Point, threats: &Vec<Point>) -> bool {
    if threats.iter().any(|&x| (x - point).len_l1() <= 1) { return false; }
    let cell = entity.known.get(point);
    cell.shade() || matches!(cell.tile(), Some(x) if x.limits_vision())
}

fn run_away(entity: &Entity, map: Vec<(Point, i32)>,
            ai: &mut AIState, env: &mut AIEnv) -> Option<BFSResult> {
    let Some(flight) = &ai.flight else { return None };
    if flight.done() || flight.threats.is_empty() { return None; }

    if map.is_empty() { return None; }

    let (known, pos) = (&*entity.known, entity.pos);

    let threat_inv_l2s: Vec<_> = flight.threats.iter().map(
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
        let hiding = flight.stage == FlightStage::Hide;
        let hidden = !hiding && is_hiding_place(entity, p, &flight.threats);

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

    let scores: Vec<_> = map.into_iter().map(
        |(p, distance)| (p, score(p, distance))).collect();
    sample_scored(entity, &scores, env)
}

fn hide_from_threats(entity: &Entity, ai: &mut AIState, env: &mut AIEnv) -> Option<BFSResult> {
    let Some(flight) = &mut ai.flight else { return None };
    if flight.done() || flight.threats.is_empty() { return None; }
    if flight.since_seen >= max(flight.till_assess, 1) { return None; }

    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| {
        if !is_hiding_place(entity, p, &flight.threats) { return Status::Blocked; }
        known.get(p).status()
    };
    if check(entity.pos) == Status::Blocked { return None; }

    flight.stage = FlightStage::Hide;

    let map = DijkstraMap(pos, check, HIDING_CELLS, HIDING_LIMIT, FOV_RADIUS);
    run_away(entity, map, ai, env)
}

fn flee_from_threats(entity: &Entity, ai: &mut AIState, env: &mut AIEnv) -> Option<BFSResult> {
    let Some(flight) = &mut ai.flight else { return None };
    if flight.done() || flight.threats.is_empty() { return None; }
    if flight.since_seen >= max(flight.till_assess, 1) { return None; }

    flight.stage = FlightStage::Flee;

    // WARNING: DijkstraMap doesn't flag squares where we heard something,
    // but those squares tend to be where nearby enemies are!
    //
    // This reasoning applies to all uses of DijkstraMap, but it's worse
    // for fleeing entities because often the first step they try to make to
    // flee is to move into cell where they heard a target.

    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| known.get(p).status();
    let map = DijkstraMap(pos, check, SEARCH_CELLS, SEARCH_LIMIT, FOV_RADIUS);
    run_away(entity, map, ai, env)
}

fn update_ai_state(entity: &Entity, hints: &[Hint], ai: &mut AIState) {
    ai.till_assess = max(0, ai.till_assess - 1);
    ai.till_hunger = max(0, ai.till_hunger - 1);
    ai.till_thirst = max(0, ai.till_thirst - 1);
    ai.till_rested = max(0, ai.till_rested - 1);

    let (known, pos) = (&*entity.known, entity.pos);
    let last_turn_age = known.time - ai.time;
    let mut seen = HashSet::default();
    for cell in &known.cells {
        if (ai.time - cell.last_seen) >= 0 { break; }
        for &(goal, tile) in hints {
            if cell.tile == tile && seen.insert(goal) {
                ai.hints.insert(goal, cell.point);
            }
        }
    }
    ai.record_turn(known.time);

    // "rival" means that we have a hostile relationship with that entity.
    // We'll end up with three states - Friendly, Neutral, or Rival - or more.
    // An entity is a "threat" if its a rival and our combat analysis against
    // it shows that we'd probably lose. These predicates can be generalized
    // to all entities.
    //
    // The threshold for a rival being a threat may also depend on some other
    // parameter like "aggressiveness". A maximally-aggressive entity will
    // stand and fight even in hopeless situations.

    // We're a predator, and we should chase and attack rivals.
    if entity.predator {
        let fight = std::mem::take(&mut ai.fight);
        let limit = ai.age_at_turn(MAX_SEARCH_TURNS);
        let mut targets: Vec<_> = known.entities.iter().filter(|x| x.rival).collect();
        if !targets.is_empty() {
            targets.sort_unstable_by_key(
                |x| (x.age, (x.pos - pos).len_l2_squared()));
            let EntityKnowledge { age, pos: target, .. } = *targets[0];
            let reset = age < last_turn_age;
            if age < limit {
                let (bias, search_turns) = if !reset && let Some(x) = fight {
                    (x.bias, x.search_turns + 1)
                } else {
                    (target - pos, 0)
                };
                ai.fight = Some(FightState { age, bias, search_turns, target });
            }
            if reset { ai.plan.clear(); }
        }
        return;
    }

    // We're prey, and we should treat rivals as threats.
    let limit = ai.age_at_turn(MAX_FLIGHT_TURNS);
    let reset = known.entities.iter().any(
        |x| x.age < last_turn_age && x.rival);
    let mut threats: Vec<_> = known.entities.iter().filter_map(
        |x| if x.age < limit && x.rival { Some(x.pos) } else { None }).collect();
    threats.sort_unstable_by_key(|x| (x.0, x.1));

    let hidden = threats.is_empty() || is_hiding_place(entity, entity.pos, &threats);
    let stage = if hidden { FlightStage::Hide } else { FlightStage::Flee };

    if let Some(x) = &mut ai.flight {
        if threats.is_empty() {
            ai.flight = None;
        } else {
            let assess = min(2 * x.till_assess, MAX_FLIGHT_TURNS);
            x.since_seen = if reset { 0 } else { x.since_seen + 1 };
            if reset || (!x.done() && x.threats != threats) { ai.plan.clear(); }
            if reset && ai.goal == Goal::Assess { x.till_assess = assess;  }
            if reset { x.stage = min(x.stage, stage); }
            x.threats = threats;
        }
    } else if !threats.is_empty() {
        let (since_seen, till_assess) = (0, MIN_FLIGHT_TURNS);
        ai.flight = Some(FlightState { stage, threats, since_seen, till_assess });
        ai.plan.clear();
    }
}

fn plan_cached(entity: &Entity, hints: &[Hint],
               ai: &mut AIState, rng: &mut RNG) -> Option<Action> {
    if ai.plan.is_empty() { return None; }

    // Check whether we can execute the next step in the plan.
    let (known, pos) = (&*entity.known, entity.pos);
    let next = *ai.plan.last().unwrap();
    let look = next.kind == StepKind::Look;
    let dir = next.target - pos;
    if !look && dir.len_l1() > 1 { return None; }

    // Check whether the plan's goal is still a top priority for us.
    let mut goals: Vec<Goal> = vec![];
    if let Some(x) = &ai.flight && !x.done() {
        if x.since_seen > 0 { goals.push(Goal::Assess); }
        if x.since_seen < max(x.till_assess, 1) { goals.push(Goal::Flee); }
    } else if ai.fight.is_some() {
        goals.push(Goal::Chase);
    } else if ai.goal == Goal::Assess {
        goals.push(Goal::Assess);
    } else {
        if ai.till_hunger == 0 && ai.hints.contains_key(&Goal::Eat) {
            goals.push(Goal::Eat);
        }
        if ai.till_thirst == 0 && ai.hints.contains_key(&Goal::Drink) {
            goals.push(Goal::Drink);
        }
        if ai.till_rested < MIN_RESTED && ai.hints.contains_key(&Goal::Rest) {
            goals.push(Goal::Rest);
        } else if ai.goal == Goal::Rest {
            goals.push(Goal::Rest);
        }
    }
    if goals.is_empty() { goals.push(Goal::Explore); }
    if !goals.contains(&ai.goal) { return None; }

    // Check if we saw a shortcut that would also satisfy the goal.
    if let Some(&x) = ai.hints.get(&ai.goal) && known.get(x).visible() {
        let target = ai.plan.iter().find_map(
            |x| if x.kind == StepKind::Move { Some(x.target) } else { None });
        if let Some(y) = target && (pos - x).len_l2_squared() < (pos - y).len_l2_squared() {
            let los = LOS(pos, x);
            let check = |p: Point| known.get(p).status();
            let free = (1..los.len() - 1).all(|i| check(los[i]) == Status::Free);
            if free { return None; }
        }
    }

    // Check if we got specific information that invalidates the plan.
    let point_matches_goal = |goal: Goal, point: Point| {
        let tile = hints.iter().find_map(
            |x| if x.0 == goal { Some(x.1) } else { None });
        if tile.is_none() { return false; }
        known.get(point).tile() == tile
    };
    let step_valid = |Step { kind, target }| match kind {
        StepKind::Rest => point_matches_goal(Goal::Rest, target),
        StepKind::Drink => point_matches_goal(Goal::Drink, target),
        StepKind::Eat => point_matches_goal(Goal::Eat, target),
        StepKind::Look => true,
        StepKind::Move => match known.get(target).status() {
            Status::Occupied => target != next.target,
            Status::Blocked  => false,
            Status::Free     => true,
            Status::Unknown  => true,
        }
    };
    if !ai.plan.iter().all(|&x| step_valid(x)) { return None; }

    // The plan is good! Execute the next step.
    ai.plan.pop();
    let wait = Some(Action::Idle);
    match next.kind {
        StepKind::Rest => { ai.till_rested += 2; wait }
        StepKind::Drink => { ai.till_thirst = MAX_THIRST; wait }
        StepKind::Eat => { ai.till_hunger = MAX_HUNGER; wait }
        StepKind::Look => {
            if ai.plan.is_empty() && ai.goal == Goal::Assess {
                ai.till_assess = rng.gen_range(0..MAX_ASSESS);
                if let Some(x) = &mut ai.flight {
                    x.till_assess = MIN_FLIGHT_TURNS;
                    x.stage = FlightStage::Done;
                }
            }
            Some(Action::Look(next.target))
        }
        StepKind::Move => {
            let mut target = next.target;
            for next in ai.plan.iter().rev().take(8) {
                if next.kind == StepKind::Look { break; }
                if LOS(pos, next.target).iter().all(|&x| !known.get(x).blocked()) {
                    target = next.target;
                }
            }
            let turns = (|| {
                let running = ai.goal == Goal::Flee || ai.goal == Goal::Chase;
                if !running { return WANDER_TURNS; }
                if !move_ready(entity) { return SLOWED_TURNS; }
                if ai.goal == Goal::Flee {
                    let Some(x) = &ai.flight else { return WANDER_TURNS; };
                    if x.stage == FlightStage::Hide { WANDER_TURNS } else { 1. }
                } else {
                    let Some(x) = &ai.fight else { return WANDER_TURNS; };
                    if x.search_turns >= MIN_SEARCH_TURNS { WANDER_TURNS } else { 1. }
                }
            })();
            let look = if target == pos { entity.dir } else { target - pos };
            Some(Action::Move(Move { look, step: dir, turns }))
        }
    }
}

pub fn plan_npc(entity: &Entity, ai: &mut AIState, env: &mut AIEnv) -> Action {
    let hints = [
        (Goal::Drink, Tile::get('~')),
        (Goal::Eat,   Tile::get('%')),
        (Goal::Rest,  Tile::get('"')),
    ];
    update_ai_state(entity, &hints, ai);
    if let Some(x) = plan_cached(entity, &hints, ai, env.rng) { return x; }

    ai.plan.clear();
    ai.goal = Goal::Explore;

    if let Some(x) = &mut env.debug {
        x.targets.clear();
        x.utility.clear();
    }

    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| known.get(p).status();

    let mut result = {
        let mut result = BFSResult::default();

        if let Some(x) = hide_from_threats(entity, ai, env) {
            (ai.goal, result) = (Goal::Flee, x);
        } else if let Some(x) = flee_from_threats(entity, ai, env) {
            (ai.goal, result) = (Goal::Flee, x);
        } else if let Some(x) = &ai.flight && !x.done() {
            (ai.goal, result) = (Goal::Assess, coerce(pos, &[]));
        } else if let Some(x) = &ai.fight {
            if x.age == 0 {
                ai.goal = Goal::Chase;
                return attack_target(entity, x.target, env.rng);
            }
            let search_nearby = x.search_turns > x.bias.len_l1();
            let source = if search_nearby { entity.pos } else { x.target };
            if let Some(y) = search_around(entity, source, x.age, x.bias, env) {
                (ai.goal, result) = (Goal::Chase, y);
            }
        }

        let mut add_candidates = |ai: &mut AIState, goal: Goal| {
            if ai.goal != Goal::Explore { return; }

            let option = ai.hints.get(&goal);
            let tile = hints.iter().find_map(
                |x| if x.0 == goal { Some(x.1) } else { None });
            if option.is_none() || tile.is_none() { return; }

            let (option, tile) = (*option.unwrap(), tile.unwrap());
            let target = |p: Point| {
                let cell = known.get(p);
                cell.tile() == Some(tile) && cell.status() != Status::Occupied
            };
            if target(pos) {
                (ai.goal, result) = (goal, coerce(pos, &[]));
            } else if let Some(x) = DijkstraSearch(pos, target, ASTAR_LIMIT_WANDER, check) {
                (ai.goal, result) = (goal, coerce(pos, &x));
            } else if let Some(x) = AStar(pos, option, ASTAR_LIMIT_WANDER, check) {
                (ai.goal, result) = (goal, coerce(pos, &x));
            }
        };
        if ai.till_rested < MIN_RESTED { add_candidates(ai, Goal::Rest); }
        if ai.till_thirst == 0 { add_candidates(ai, Goal::Drink); }
        if ai.till_hunger == 0 { add_candidates(ai, Goal::Eat); }

        if result.dirs.is_empty() && ai.till_assess == 0 {
            (ai.goal, result) = (Goal::Assess, coerce(pos, &[]));
        } else if result.dirs.is_empty() && let Some(x) = explore(entity, env) {
            (ai.goal, result) = (Goal::Explore, x);
        }
        result
    };

    let fallback = |result: BFSResult, rng: &mut RNG| {
        let dirs = &result.dirs;
        let dirs = if dirs.is_empty() { dirs::ALL.as_slice() } else { dirs };
        wander(*sample(dirs, rng))
    };

    if result.targets.is_empty() { return fallback(result, env.rng); }

    if let Some(x) = &mut env.debug { x.targets = result.targets.clone(); }
    let target = *result.targets.select_nth_unstable_by_key(
        0, |&x| (x - pos).len_l2_squared()).1;
    if target == pos && ai.goal == Goal::Flee { ai.goal = Goal::Assess; }

    if let Some(path) = AStar(pos, target, ASTAR_LIMIT_WANDER, check) {
        let kind = StepKind::Move;
        ai.plan = path.into_iter().map(|x| Step { kind, target: x }).collect();
        match ai.goal {
            Goal::Assess => assess_nearby(entity, ai, env.rng),
            Goal::Chase => {}
            Goal::Drink => ai.plan.push(Step { kind: StepKind::Drink, target }),
            Goal::Eat => ai.plan.push(Step { kind: StepKind::Eat, target }),
            Goal::Explore => {}
            Goal::Flee => {}
            Goal::Rest => {
                let delta = max(MAX_RESTED - ai.till_rested, 0);
                for _ in 0..delta { ai.plan.push(Step { kind: StepKind::Rest, target }) };
            }
        }
        ai.plan.reverse();
        if let Some(x) = plan_cached(entity, &hints, ai, env.rng) { return x; }
        ai.plan.clear();
    }
    fallback(result, env.rng)
}
