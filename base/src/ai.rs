use std::cmp::{max, min};
use std::f64::consts::TAU;
use std::fmt::Debug;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::base::{LOS, Point, Slice, RNG, dirs, sample, weighted};
use crate::entity::Entity;
use crate::game::{NOISY_RADIUS, Item, move_ready};
use crate::game::{Action, EatAction, MoveAction};
use crate::knowledge::{CellKnowledge, Knowledge, Scent};
use crate::pathing::{AStar, BFS, DijkstraLength, DijkstraMap, Neighborhood, Status};
use crate::shadowcast::{INITIAL_VISIBILITY, Vision, VisionArgs};

//////////////////////////////////////////////////////////////////////////////

// Constants

const ASTAR_LIMIT_ATTACK: i32 = 32;
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

const MIN_FLIGHT_TURNS: i32 = 16;
const MAX_FLIGHT_TURNS: i32 = 64;
const MIN_SEARCH_TURNS: i32 = 16;
const MAX_SEARCH_TURNS: i32 = 32;

const SLOWED_TURNS: f64 = 2.;
const WANDER_TURNS: f64 = 2.;

//////////////////////////////////////////////////////////////////////////////

// Basic helpers

fn safe_inv_l2(point: Point) -> f64 {
    if point == Point::default() { return 0. }
    (point.len_l2_squared() as f64).sqrt().recip()
}

fn is_hiding_place(entity: &Entity, point: Point, threats: &[Threat]) -> bool {
    if threats.iter().any(|&x| (x.pos - point).len_l1() <= 1) { return false; }
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
    pub debug: Option<&'a mut AIDebug>,
    pub fov: &'a mut Vision,
    pub rng: &'a mut RNG,
}

//////////////////////////////////////////////////////////////////////////////

// Path caching

#[derive(Default)]
struct CachedPath {
    skipped: usize,
    steps: Vec<Point>,
    pos: Point,
}

impl CachedPath {
    fn reset(&mut self) { self.steps.clear(); }

    fn check(&self, ctx: &Context) -> bool {
        if self.steps.len() <= self.skipped { return false; }
        let Some(&next) = self.steps.last() else { return false; };

        let Context { known, pos, .. } = *ctx;
        if pos != self.pos { return false; }

        let valid = |p: Point| match known.get(p).status() {
            Status::Free | Status::Unknown => true,
            Status::Occupied => p != next,
            Status::Blocked  => false,
        };
        self.steps.iter().skip(self.skipped).all(|&x| valid(x))
    }

    fn sneak(&mut self, ctx: &Context, target: Point,
             turns: f64, threats: &[Threat]) -> Option<Action> {
        let Context { entity, known, pos, .. } = *ctx;

        let check = |p: Point| {
            if !is_hiding_place(entity, p, threats) { return Status::Blocked; }
            known.get(p).status()
        };
        let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check)?;
        self.go(ctx, path, turns)
    }

    fn start(&mut self, ctx: &mut Context, target: Point, turns: f64) -> Option<Action> {
        ensure_vision(ctx);

        let Context { known, pos, .. } = *ctx;
        let fov = &*ctx.env.fov;

        let check = |p: Point| {
            let status = known.get(p).status();
            if status != Status::Unknown { return status; }
            if fov.can_see(p - pos) { Status::Free } else { Status::Unknown }
        };
        let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check)?;
        self.go(ctx, path, turns)
    }

    fn go(&mut self, ctx: &Context, path: Vec<Point>, turns: f64) -> Option<Action> {
        self.pos = ctx.pos;
        self.steps = path;
        self.steps.reverse();

        if self.check(ctx)  { self.follow(ctx, turns) } else { None }
    }

    fn follow(&mut self, ctx: &Context, turns: f64) -> Option<Action> {
        let Context { known, pos, dir, .. } = *ctx;
        let next = self.steps.pop()?;
        self.pos = next;

        let mut target = next;
        for &point in self.steps.iter().rev().take(8) {
            let los = LOS(pos, point);
            if los.iter().all(|&x| !known.get(x).blocked()) { target = point; }
        }
        let look = if target == pos { dir } else { target - pos };
        Some(Action::Move(MoveAction { look, step: next - pos, turns }))
    }
}

//////////////////////////////////////////////////////////////////////////////

// Shared state

struct SharedAIState {
    till_assess: i32,
    till_hunger: i32,
    till_thirst: i32,
    till_rested: i32,
}

impl Debug for SharedAIState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedAIState")
         .field("till_assess", &self.till_assess)
         .field("till_hunger", &self.till_hunger)
         .field("till_rested", &self.till_rested)
         .field("till_thirst", &self.till_thirst)
         .finish()
    }
}

impl SharedAIState {
    pub fn new(predator: bool, rng: &mut RNG) -> Self {
        let max_hunger = if predator { MAX_HUNGER_CARNIVORE } else { MAX_HUNGER_HERBIVORE };
        Self {
            till_assess: rng.gen_range(0..MAX_ASSESS),
            till_hunger: rng.gen_range(0..max_hunger),
            till_rested: rng.gen_range(0..MAX_RESTED),
            till_thirst: rng.gen_range(0..MAX_THIRST),
        }
    }

    fn update(&mut self, entity: &Entity) {
        if !entity.asleep {
            self.till_assess = max(self.till_assess - 1, 0);
            self.till_hunger = max(self.till_hunger - 1, 0);
            self.till_thirst = max(self.till_thirst - 1, 0);
            self.till_rested = max(self.till_rested - 1, 0);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Strategy-based planning

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Priority { Survive, EatMeat, Hunt, SatisfyNeeds, Explore, Skip }

struct Context<'a, 'b> {
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
}

trait Strategy {
    fn get_path(&self) -> &[Point];
    fn debug(&self, slice: &mut Slice);
    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i32);
    fn accept(&mut self, ctx: &mut Context) -> Option<Action>;
    fn reject(&mut self);
}

//////////////////////////////////////////////////////////////////////////////

// BasicNeedsStrategy

#[derive(Debug, Eq, PartialEq)]
enum BasicNeed { EatMeat, EatPlants, Drink }

struct BasicNeedsStrategy {
    last: Option<Point>,
    need: BasicNeed,
    path: CachedPath,
}

impl BasicNeedsStrategy {
    fn new(need: BasicNeed) -> Self {
        let path = CachedPath { skipped: 1, ..Default::default() };
        Self { last: None, need, path }
    }

    fn action(&self, point: Point) -> Action {
        match self.need {
            BasicNeed::EatMeat => Action::Eat(EatAction { point, item: Some(Item::Corpse) }),
            BasicNeed::EatPlants => Action::Eat(EatAction { point, item: None }),
            BasicNeed::Drink => Action::Drink(point),
        }
    }

    fn priority(&self) -> Priority {
        match self.need {
            BasicNeed::EatMeat => Priority::EatMeat,
            BasicNeed::EatPlants => Priority::SatisfyNeeds,
            BasicNeed::Drink => Priority::SatisfyNeeds,
        }
    }

    fn satisfies_need(&self, cell: Option<&CellKnowledge>) -> bool {
        let Some(cell) = cell else { return false };
        match self.need {
            BasicNeed::EatMeat => cell.items.contains(&Item::Corpse),
            BasicNeed::EatPlants => cell.tile.can_eat(),
            BasicNeed::Drink => cell.tile.can_drink(),
        }
    }

    fn timeout(&self) -> i32 {
        match self.need {
            BasicNeed::EatMeat => MAX_HUNGER_CARNIVORE,
            BasicNeed::EatPlants => MAX_HUNGER_HERBIVORE,
            BasicNeed::Drink => MAX_THIRST,
        }
    }

    fn turns_left<'a>(&self, ctx: &'a mut Context) -> &'a mut i32 {
        match self.need {
            BasicNeed::EatMeat => &mut ctx.shared.till_hunger,
            BasicNeed::EatPlants => &mut ctx.shared.till_hunger,
            BasicNeed::Drink => &mut ctx.shared.till_thirst,
        }
    }

    fn update_last(&mut self, ctx: &Context) {
        // If we just saw a cell that satisfies our need, save the closest one.
        for cell in &ctx.observations {
            if !self.satisfies_need(Some(cell)) { continue; }
            self.last = Some(cell.point);
            break;
        }
        let Some(last) = self.last else { return };

        // If we can see `last` and it no longer satisfies our need, drop it.
        let Context { known, pos, .. } = *ctx;
        let cell = known.get(last);
        if cell.visible() && !self.satisfies_need(cell.get_cell()) {
            self.last = None;
            return;
        }

        // If `last` is a strict improvement over our current path, re-plan.
        if let Some(&target) = self.path.steps.first() &&
           (last - pos).len_l2_squared() < (target - pos).len_l2_squared() {
            let los = LOS(pos, last);
            let free = (1..los.len() - 1).all(
                |i| known.get(los[i]).status() == Status::Free);
            if free { self.path.reset(); }
        }
    }
}

impl Strategy for BasicNeedsStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice) {
        slice.write_str(&format!("BasicNeed: {:?}", self.need)).newline();
    }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i32) {
        // Keep track of the the last-seen cell that satisfies our need.
        self.update_last(ctx);

        // Clear the cached path, if it's no longer good.
        let known = ctx.known;
        let valid = |p: Point| self.satisfies_need(known.get(p).get_cell());
        if !(self.path.check(ctx) && valid(self.path.steps[0])) { self.path.reset(); }

        // Compute a priority for satisfying this need.
        if self.last.is_none() { return (Priority::Skip, 0); }
        let (turns_left, timeout) = (*self.turns_left(ctx), self.timeout());
        let cutoff = max(timeout / 2, 1);
        if turns_left >= cutoff { return (Priority::Skip, 0); }
        let tiebreaker = if last { -1 } else { 100 * turns_left / cutoff };
        (self.priority(), tiebreaker)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let turns = WANDER_TURNS;
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let Context { known, pos, .. } = *ctx;
        let valid = |p: Point| self.satisfies_need(known.get(p).get_cell());
        for &dir in [dirs::NONE].iter().chain(&dirs::ALL) {
            if valid(pos + dir) {
                *self.turns_left(ctx) = max(*self.turns_left(ctx), self.timeout());
                return Some(self.action(pos + dir));
            }
        }

        ensure_neighborhood(ctx);

        let n = &ctx.neighborhood;
        for &(point, _) in n.blocked.iter().chain(&n.visited) {
            if !self.satisfies_need(known.get(point).get_cell()) { continue; }

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

// RestStrategy

#[derive(Default)]
struct RestStrategy { path: CachedPath }

impl Strategy for RestStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice) {
        slice.write_str("Rest").newline();
    }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i32) {
        if !self.path.check(ctx) { self.path.reset(); }

        // Compute a priority for satisfying this need.
        let turns_left = ctx.shared.till_rested;
        let cutoff = max(MAX_RESTED / if last { 1 } else { 2 }, 1);
        if turns_left >= cutoff { return (Priority::Skip, 0); }
        let tiebreaker = if last { -1 } else { 100 * turns_left / cutoff };
        (Priority::SatisfyNeeds, tiebreaker)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let turns = WANDER_TURNS;
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let Context { entity, known, pos, .. } = *ctx;
        let valid = |p: Point| is_hiding_place(entity, p, &[]);
        if valid(pos) {
            ctx.shared.till_rested += 1;
            return Some(Action::Rest);
        }

        ensure_neighborhood(ctx);

        for &(point, _) in &ctx.neighborhood.visited {
            if !valid(point) { continue; }
            if known.get(point).status() != Status::Free { continue; }

            // Compute at most one path to a point in the neighborhood.
            let action = self.path.start(ctx, point, turns);
            if action.is_some() { return action; }
            break;
        }

        None
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

// AssessStrategy

#[derive(Default)]
struct AssessStrategy { dirs: Vec<Point> }

impl Strategy for AssessStrategy {
    fn get_path(&self) -> &[Point] { &[] }

    fn debug(&self, slice: &mut Slice) {
        slice.write_str("Assess").newline();
    }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i32) {
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

#[derive(Default)]
struct ExploreStrategy { path: CachedPath }

impl Strategy for ExploreStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice) {
        slice.write_str("Explore").newline();
    }

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

        let score = |p: Point, distance: i32| -> f64 {
            if distance == 0 { return 0.; }

            let age = known.get(p).turns_since_seen();
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
        self.path.start(ctx, target, turns)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

#[derive(Default)]
struct TrackStrategy {
    path: CachedPath,
    scent: Option<Scent>,
    fresh: bool,
}

impl Strategy for TrackStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice) {
        slice.write_str("Track").newline();
        if let Some(x) = &self.scent {
            slice.write_str(&format!("    scent.age: {}", x.age)).newline();
            slice.write_str(&format!("    scent.pos: {:?}", x.pos)).newline();
            slice.write_str(&format!("    fresh: {}", self.fresh)).newline();
        }
    }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i32) {
        if !self.path.check(ctx) { self.path.reset(); }

        if let Some(x) = &mut self.scent { x.age += 1; }
        self.fresh = false;

        for &scent in &ctx.known.scents {
            let (fresh, newer) = match &self.scent {
                Some(x) => (scent.pos != x.pos, scent.age < x.age),
                None => (true, true),
            };
            if !newer { continue; }

            if fresh { self.path.reset(); }
            self.scent = Some(scent);
            self.fresh = fresh;
        }

        if let Some(x) = &self.scent && x.age >= MAX_SEARCH_TURNS {
            self.scent = None;
            self.fresh = false;
        }

        let Some(x) = &self.scent else { return (Priority::Skip, 0); };

        (Priority::Hunt, 2 * x.age + 1)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let Some(x) = self.scent else { return None; };
        let Scent { age, pos } = x;

        if self.fresh { return Some(Action::SniffAround); }

        let turns = if !move_ready(ctx.entity) { SLOWED_TURNS } else { WANDER_TURNS };
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        search_around(ctx, &mut self.path, age, Point(0, 0), pos, turns)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct ChaseTarget {
    age: i32,
    bias: Point,
    last: Point,
}

#[derive(Default)]
struct ChaseStrategy {
    path: CachedPath,
    target: Option<ChaseTarget>,
}

impl Strategy for ChaseStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice) {
        slice.write_str("Chase").newline();
        if let Some(x) = &self.target {
            slice.write_str(&format!("    target.age: {}", x.age)).newline();
            slice.write_str(&format!("    target.bias: {:?}", x.bias)).newline();
            slice.write_str(&format!("    target.last: {:?}", x.last)).newline();
        }
    }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i32) {
        if !ctx.entity.predator { return (Priority::Skip, 0); }

        let prev = self.target.take();

        let turns_left = ctx.shared.till_hunger;
        let cutoff = max(MAX_HUNGER_CARNIVORE, 1);
        if turns_left >= cutoff { return (Priority::Skip, 0); }

        let Context { known, pos, .. } = *ctx;
        let mut targets: Vec<_> = known.entities.iter().filter(
            |x| x.age < MAX_SEARCH_TURNS && x.rival).collect();
        if targets.is_empty() { return (Priority::Skip, 0); }

        let target = *targets.select_nth_unstable_by_key(
                0, |x| (x.age, (x.pos - pos).len_l2_squared())).1;
        let reset = target.age == 0;
        if reset || !self.path.check(ctx) { self.path.reset(); }

        let (age, last) = (target.age, target.pos);
        let bias = if !reset && let Some(x) = prev {
            x.bias
        } else {
            target.pos - pos
        };
        self.target = Some(ChaseTarget { age, bias, last });
        (Priority::Hunt, 2 * age)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        // TODO: Don't attack the target if the last ping was because we heard
        // it but we don't currently see it - check that it's seen!
        let ChaseTarget { age, bias, last } = *self.target.as_ref()?;
        if age == 0 { return Some(attack_target(ctx, last)); }

        let turns = {
            if !move_ready(ctx.entity) { SLOWED_TURNS }
            else if age >= MIN_SEARCH_TURNS { WANDER_TURNS } else { 1. }
        };
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let search_nearby = age > bias.len_l1();
        let center = if search_nearby { ctx.pos } else { last };
        search_around(ctx, &mut self.path, age, bias, center, turns)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Threat {
    asleep: bool,
    pos: Point,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum FlightStage { Flee, Hide, Done }

struct FlightStrategy {
    dirs: Vec<Point>,
    path: CachedPath,
    stage: FlightStage,
    since_path: i32,
    since_seen: i32,
    turn_limit: i32,
    threats: Vec<Threat>,
    utility: Vec<String>,
}

impl FlightStrategy {
    fn done(&self) -> bool { self.stage == FlightStage::Done }

    fn look(&mut self, ctx: &mut Context) -> Option<Action> {
        let (pos, rng) = (ctx.pos, &mut ctx.env.rng);
        self.path.reset();

        if self.dirs.is_empty() {
            let dirs: Vec<_> = self.threats.iter().map(|&x| x.pos - pos).collect();
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
        Self {
            dirs: vec![],
            path: Default::default(),
            stage: FlightStage::Done,
            since_path: 0,
            since_seen: 0,
            turn_limit: MIN_FLIGHT_TURNS,
            threats: vec![],
            utility: vec![],
        }
    }
}

impl Strategy for FlightStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice) {
        slice.write_str("Flight").newline();
        slice.write_str(&format!("    stage: {:?}", self.stage)).newline();
        slice.write_str(&format!("    since_path: {}", self.since_path)).newline();
        slice.write_str(&format!("    since_seen: {}", self.since_seen)).newline();
        slice.write_str(&format!("    turn_limit: {}", self.turn_limit)).newline();
        slice.write_str("    utility:").newline();
        for line in &self.utility {
            slice.write_str("      ").write_str(line).newline();
        }
    }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i32) {
        if ctx.entity.predator { return (Priority::Skip, 0); }

        let Context { entity, known, pos, .. } = *ctx;
        let reset = known.entities.iter().any(|x| x.age == 0 && x.rival);
        let mut threats: Vec<_> = known.entities.iter().filter_map(|x| {
            if x.age >= MAX_FLIGHT_TURNS || !x.rival { return None; }
            Some(Threat { asleep: x.asleep, pos: x.pos })
        }).collect();
        threats.sort_unstable_by_key(|x| (x.pos.0, x.pos.1));

        if self.done() && !reset { return (Priority::Skip, 0); }

        let looking = !self.dirs.is_empty();
        let changed = !threats.is_empty() && threats != self.threats;
        let hiding = is_hiding_place(entity, pos, &threats);
        let stage = if hiding { FlightStage::Hide } else { FlightStage::Flee };

        if reset || changed {
            self.dirs.clear();
            if self.stage > stage || self.since_path >= 8 { self.path.reset(); }
        }
        if !self.path.check(ctx) {
            self.path.reset();
        }

        self.stage = if reset { min(self.stage, stage) } else { self.stage };
        self.since_seen = if reset { 0 } else { self.since_seen + 1 };
        self.turn_limit = if reset && looking {
            min(2 * self.turn_limit, MAX_FLIGHT_TURNS)
        } else {
            self.turn_limit
        };
        if changed { self.threats = threats; }

        (Priority::Survive, 0)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let last_since_path = self.since_path;
        self.since_path = 0;

        if !self.dirs.is_empty() || self.since_seen >= self.turn_limit {
            self.path.reset();
            return self.look(ctx);
        }

        let all_asleep = self.threats.iter().all(|x| x.asleep);
        let pick_turns = |stage: FlightStage| {
            if !move_ready(ctx.entity) { return SLOWED_TURNS }
            let sneak = all_asleep || stage == FlightStage::Hide;
            if sneak { WANDER_TURNS } else { 1. }
        };

        let turns = pick_turns(self.stage);
        if let Some(x) = self.path.follow(ctx, turns) {
            self.since_path = last_since_path + 1;
            return Some(x);
        }

        let Context { entity, pos, .. } = *ctx;

        if !all_asleep && is_hiding_place(entity, pos, &self.threats) {
            ensure_sneakable(ctx, &self.threats);

            if let Some(target) = flight_cell(ctx, &self.threats, &mut self.utility, true) {
                self.stage = FlightStage::Hide;
                if target == pos { return self.look(ctx); }
                return self.path.sneak(ctx, target, WANDER_TURNS, &self.threats);
            }
        }

        ensure_neighborhood(ctx);

        if let Some(target) = flight_cell(ctx, &self.threats, &mut self.utility, false) {
            self.stage = FlightStage::Flee;
            if target == pos { return self.look(ctx); }
            return self.path.start(ctx, target, pick_turns(self.stage));
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
        Action::Move(MoveAction { step: dir, look, turns: 1. })
    };

    // Given a non-empty list of "good" directions (each of which brings us
    // close to attacking the target), choose one closest to our attack range.
    let pick = |dirs: &Vec<Point>, rng: &mut RNG| {
        let cell = known.get(target);
        let (shade, tile) = (cell.shade(), cell.tile());
        let obscured = shade || matches!(tile, Some(x) if x.limits_vision());
        let distance = if obscured { 1 } else { min(range, NOISY_RADIUS) };

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
        assert!(!cell.blocked());
        if cell.turns_since_entity_visible() <= age { return 0. }

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos0 = delta.dot(dir) as f64 * inv_delta_l2 * inv_dir_l2;
        let cos1 = delta.dot(bias) as f64 * inv_delta_l2 * inv_bias_l2;
        let angle = ((cos0 + 1.) * (cos1 + 1.)).pow(4);
        let bonus = if p == center && center != pos { 64. } else { 1. };

        angle * bonus / (((p - center).len_l2_squared() + 1) as f64).pow(2)
    };

    ensure_neighborhood(ctx);

    let scores: Vec<_> = ctx.neighborhood.visited.iter().map(
        |&(p, distance)| (p, score(p, distance))).collect();
    let target = select_target(&scores, ctx.env)?;
    path.start(ctx, target, turns)
}

//////////////////////////////////////////////////////////////////////////////

// Helpers for FlightStrategy

fn flight_cell(ctx: &mut Context, threats: &[Threat],
               utility: &mut Vec<String>, hiding: bool) -> Option<Point> {
    let Context { entity, known, pos, .. } = *ctx;

    let threat_inv_l2s: Vec<_> = threats.iter().map(
        |&x| (x.pos, safe_inv_l2(pos - x.pos))).collect();
    let scale = 1. / DijkstraLength(Point(1, 0)) as f64;

    let score = |p: Point, source_distance: i32, debug: Option<&mut String>| -> f64 {
        let mut inv_l2 = 0.;
        let mut threat = Point::default();
        let mut threat_distance = std::i32::MAX;
        for &(x, y) in &threat_inv_l2s {
            let z = DijkstraLength(x - p);
            if z < threat_distance { (threat, inv_l2, threat_distance) = (x, y, z); }
        }

        let blocked = {
            let los = LOS(threat, p);
            (1..los.len() - 1).any(|i| known.get(los[i]).blocked())
        };
        let frontier = dirs::ALL.iter().any(|&x| known.get(p + x).unknown());
        let hidden = !hiding && is_hiding_place(entity, p, &threats);

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos = delta.dot(pos - threat) as f64 * inv_delta_l2 * inv_l2;

        if let Some(x) = debug {
            x.push_str(if blocked { "B" } else { "." });
            x.push_str(if frontier { "F" } else { "." });
            x.push_str(if hidden { "H" } else { "." });
            x.push_str(&format!(" cos: {:.2}; threat: {:?}", cos, threat));
        }

        // WARNING: This heuristic can cause a piece to be "checkmated" in a
        // corner, if the Dijkstra search below isn't wide enough for us to
        // find a cell which is further from the threat than the corner cell.
        let base = 1.5 * scale * (threat_distance as f64) +
                   -1. * scale * (source_distance as f64) +
                   16. * if blocked { 1. } else { 0. } +
                   16. * if frontier { 1. } else { 0. } +
                   16. * if hidden { 1. } else { 0. };
        (cos + 1.).pow(0) * base.pow(1)
    };

    let min_score = score(pos, 0, None);
    let map = if hiding { &ctx.sneakable.visited } else { &ctx.neighborhood.visited };
    let scores: Vec<_> = map.iter().filter_map(|&(p, distance)| {
        let delta = score(p, distance, None) - min_score;
        if delta >= 0. { Some((p, delta)) } else { None }
    }).collect();
    let result = select_target_softmax(&scores, ctx.env, 0.1);

    if let Some(p) = result {
        let mut threat_distance = std::i32::MAX;
        let mut cur_threat_distance = std::i32::MAX;
        for &(x, _) in &threat_inv_l2s {
            threat_distance = min(threat_distance, DijkstraLength(x - p));
            cur_threat_distance = min(cur_threat_distance, DijkstraLength(x - pos));
        }
        let sd = scale * DijkstraLength(pos - p) as f64;
        let td = scale * threat_distance as f64;
        let cur_td = scale * cur_threat_distance as f64;

        let mut distance = None;
        for &(q, d) in map { if p == q { distance = Some(d); } }

        let mut move_debug = String::new();
        let mut stay_debug = String::new();
        let move_score = score(p, distance.unwrap(), Some(&mut move_debug));
        let stay_score = score(pos, 0, Some(&mut stay_debug));

        utility.clear();
        utility.push(format!("move: sd = {:.2}; td = {:.2}; score = {:.2}",
                             sd, td, move_score));
        utility.push(format!("      {}", move_debug));
        utility.push(format!("stay: sd = {:.2}; td = {:.2}; score = {:.2}",
                             0., cur_td, stay_score));
        utility.push(format!("      {}", stay_debug));
        let prob = 2. / (1. + (-0.025 * move_score).exp()) - 1.;
        utility.push(format!("escape_probability: {:.2}", prob));
    }
    result
}

//////////////////////////////////////////////////////////////////////////////

// Helpers used to implement strategies in general

fn assess_directions(dirs: &[Point], turns: i32, rng: &mut RNG) -> Vec<Point> {
    if dirs.is_empty() { return vec![]; }

    let mut result = vec![];
    result.reserve((ASSESS_STEPS * turns) as usize);

    for i in 0..ASSESS_STEPS {
        let dir = dirs[i as usize % dirs.len()];
        let distance = dir.len_l2();
        let scale = 100. / if distance > 0. { dir.len_l2() } else { 1. };

        let steps = rng.gen_range(0..turns) + 1;
        let angle = Normal::new(0., ASSESS_ANGLE).unwrap().sample(rng);
        let (sin, cos) = (angle.sin(), angle.cos());

        let Point(dx, dy) = dir;
        let rx = (cos * scale * dx as f64) + (sin * scale * dy as f64);
        let ry = (cos * scale * dy as f64) - (sin * scale * dx as f64);
        let target = Point(rx as i32, ry as i32);
        for _ in 0..steps { result.push(target); }
    }
    result
}

fn ensure_neighborhood(ctx: &mut Context) {
    if !ctx.neighborhood.visited.is_empty() { return; }

    ensure_vision(ctx);

    let Context { known, pos, .. } = *ctx;
    let fov = &*ctx.env.fov;

    let check = |p: Point| {
        let status = known.get(p).status();
        if status != Status::Unknown { return status; }
        if fov.can_see(p - pos) { Status::Free } else { Status::Unknown }
    };
    ctx.neighborhood = DijkstraMap(ctx.pos, check, SEARCH_CELLS, SEARCH_LIMIT);
}

fn ensure_sneakable(ctx: &mut Context, threats: &[Threat]) {
    if !ctx.sneakable.visited.is_empty() { return; }

    let Context { entity, known, .. } = *ctx;

    let check = |p: Point| {
        if !is_hiding_place(entity, p, threats) { return Status::Blocked; }
        known.get(p).status()
    };
    ctx.sneakable = DijkstraMap(ctx.pos, check, HIDING_CELLS, HIDING_LIMIT);
}

fn ensure_vision(ctx: &mut Context) {
    if ctx.ran_vision { return; }

    let Context { known, pos, .. } = *ctx;

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

fn select_target_softmax(
        scores: &[(Point, f64)], env: &mut AIEnv, temperature: f64) -> Option<Point> {
    if scores.is_empty() { return None; }

    let max = scores.iter().fold(
        std::f64::NEG_INFINITY, |acc, x| if acc > x.1 { acc } else { x.1 });
    let scale = ((1 << 16) - 1) as f64;
    let inv_temperature = 1. / temperature;
    let values: Vec<_> = scores.iter().map(|&(p, score)| {
        let value = (scale * (inv_temperature * (score - max)).exp()) as i32;
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


//////////////////////////////////////////////////////////////////////////////

// Execution

pub struct AIState {
    bids: Vec<(Priority, i32, usize)>,
    strategies: Vec<Box<dyn Strategy>>,
    shared: SharedAIState,
    last: i32,
}

impl AIState {
    pub fn new(predator: bool, rng: &mut RNG) -> Self {
        let strategies: Vec<Box<dyn Strategy>> = vec![
            Box::new(TrackStrategy::default()),
            Box::new(ChaseStrategy::default()),
            Box::new(FlightStrategy::default()),
            //Box::new(RestStrategy::default()),
            Box::new(BasicNeedsStrategy::new(BasicNeed::EatMeat)),
            //Box::new(BasicNeedsStrategy::new(BasicNeed::EatPlants)),
            //Box::new(BasicNeedsStrategy::new(BasicNeed::Drink)),
            Box::new(AssessStrategy::default()),
            Box::new(ExploreStrategy::default()),
        ];
        let shared = SharedAIState::new(predator, rng);
        Self { bids: vec![], strategies, shared, last: -1 }
    }

    pub fn get_path(&self) -> &[Point] {
        if self.last == -1 { return &[]; }
        self.strategies[self.last as usize].get_path()
    }

    pub fn debug(&self, slice: &mut Slice) {
        for (i, strategy) in self.strategies.iter().enumerate() {
            slice.write_str(if i as i32 == self.last { "> " } else { "  " });
            strategy.debug(slice);
            slice.newline();
        }
    }

    pub fn plan(&mut self, entity: &Entity, env: &mut AIEnv) -> Action {
        // Step 0: update some initial, deterministic shared state.
        let known = &*entity.known;
        let mut observations = vec![];
        for cell in &known.cells {
            if known.time - cell.last_seen > 0 { break; }
            observations.push(cell);
        }
        self.shared.update(entity);

        // Step 1: build a context object with all strategies' mutable state.
        let shared = &mut self.shared;
        let mut ctx = Context {
            entity,
            known,
            pos: entity.pos,
            dir: entity.dir,
            observations,
            neighborhood: Default::default(),
            sneakable: Default::default(),
            ran_vision: false,
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
