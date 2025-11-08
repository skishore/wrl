use std::cmp::{max, min};
use std::f64::consts::TAU;
use std::ops::RangeInclusive;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::{act, cb, pri, seq, util};
use crate::base::{LOS, Point, RNG, Slice, dirs, sample, weighted};
use crate::bhv::{Bhv, Debug, Result};
use crate::entity::{Entity, EID};
use crate::game::{MOVE_VOLUME, Item, move_ready};
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
const MAX_HUNGER: i32 = 512;
const MAX_THIRST: i32 = 128;
const ASSESS_GAIN: RangeInclusive<i32> = (MAX_ASSESS / 2)..=MAX_ASSESS;
const HUNGER_GAIN: RangeInclusive<i32> = (MAX_HUNGER / 4)..=(MAX_HUNGER / 2);
const THIRST_GAIN: RangeInclusive<i32> = (MAX_THIRST / 4)..=(MAX_THIRST / 2);

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
    till_hunger: i32,
    till_thirst: i32,
    finding_food_: bool,
    finding_water: bool,
    hunger: bool,
    thirst: bool,
}

impl Blackboard {
    fn new(predator: bool, rng: &mut RNG) -> Self {
        Self {
            dirs: Default::default(),
            path: Default::default(),
            till_assess: rng.random_range(0..MAX_ASSESS),
            till_hunger: rng.random_range(0..MAX_HUNGER),
            till_thirst: rng.random_range(0..MAX_THIRST),
            finding_food_: false,
            finding_water: false,
            hunger: false,
            thirst: false,
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
        bb.till_hunger = max(bb.till_hunger - 1, 0);
        bb.till_thirst = max(bb.till_thirst - 1, 0);

        if bb.till_hunger < MAX_HUNGER / 8 { bb.hunger = true; }
        if bb.till_thirst < MAX_THIRST / 8 { bb.thirst = true; }
        if bb.till_hunger >= 7 * MAX_HUNGER / 8 { bb.hunger = false; }
        if bb.till_thirst >= 7 * MAX_THIRST / 8 { bb.thirst = false; }
    }
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
    FindPath(ctx, target, PathKind::Explore)
}

//////////////////////////////////////////////////////////////////////////////

// Pathfinding

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathKind { MoveToFood, MoveToWater, BerryTree, Explore }

#[derive(Default)]
struct CachedPath {
    kind: Option<PathKind>,
    path: Vec<Point>,
    skip: usize,
    step: usize,
}

#[allow(non_snake_case)]
fn FindPath(ctx: &mut Ctx, target: Point, kind: PathKind) -> Option<Action> {
    ensure_vision(ctx);
    let (pos, check) = (ctx.pos, get_check(ctx));
    let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check)?;
    let skip = match kind {
        PathKind::MoveToFood | PathKind::MoveToWater | PathKind::BerryTree => 1,
        PathKind::Explore => 0,
    };
    ctx.blackboard.path = CachedPath { kind: Some(kind), path, skip, step: 0 };
    ctx.blackboard.path.path.insert(0, pos);
    FollowPath(ctx, kind)
}

#[allow(non_snake_case)]
fn FollowPath(ctx: &mut Ctx, kind: PathKind) -> Option<Action> {
    if ctx.blackboard.path.kind != Some(kind) { return None; }

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

    if path.step + path.skip + 2 < path.path.len() {
        ctx.blackboard.path = path;
        ctx.blackboard.path.step += 1;
    }
    Some(Action::Move(MoveAction { look, step, turns: WANDER_TURNS }))
}

//////////////////////////////////////////////////////////////////////////////

#[allow(non_snake_case)]
fn AttackPathTarget(ctx: &mut Ctx, kind: PathKind) -> Option<Action> {
    if ctx.blackboard.path.kind != Some(kind) { return None; }
    let target = *ctx.blackboard.path.path.last()?;
    AttackTarget(ctx, target)
}

#[allow(non_snake_case)]
fn AttackTarget(ctx: &mut Ctx, target: Point) -> Option<Action> {
    let Ctx { entity, known, pos: source, .. } = *ctx;
    if source == target { return None; }

    let attacks = &ctx.entity.species.attacks;
    if attacks.is_empty() { return None; }

    let attack = sample(attacks, ctx.env.rng);
    let range = attack.range;
    let valid = |x| HasLOS(x, target, known, range);
    if move_ready(entity) && valid(source) {
        return Some(Action::Attack(AttackAction { attack, target }));
    }
    PathToTarget(ctx, target, range, valid)
}

#[allow(non_snake_case)]
fn HasLOS(source: Point, target: Point, known: &Knowledge, range: i32) -> bool {
    if (source - target).len_nethack() > range { return false; }
    if !known.get(target).visible() { return false; }
    let los = LOS(source, target);
    let last = los.len() - 1;
    los.iter().enumerate().all(|(i, &p)| {
        if i == 0 || i == last { return true; }
        known.get(p).status() == Status::Free
    })
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

#[allow(non_snake_case)]
fn HasBerryTree(x: &CellKnowledge) -> bool { x.tile.drops_berries() }

#[allow(non_snake_case)]
fn FindNearbyBerry(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let valid = |p: Point| known.get(p).cell().map(HasBerryTree).unwrap_or(false);
    for &dir in &dirs::ALL {
        let target = pos + dir;
        if !valid(target) { continue; }
        // Horrible hack because we use the path target as the attack target...
        let kind = Some(PathKind::BerryTree);
        ctx.blackboard.path = CachedPath { kind, path: vec![pos, target], skip: 1, step: 0 };
        return None;
    }
    None
}

#[allow(non_snake_case)]
fn ForageForBerries() -> impl Bhv {
    pri![
        "ForageForBerries",
        act!("FindNearbyBerry", FindNearbyBerry),
        act!("AttackBerryTree", |x| AttackPathTarget(x, PathKind::BerryTree)),
        act!("Follow(BerryTree)", |x| FollowPath(x, PathKind::BerryTree)),
        act!("MoveToBerryTree", |x| MoveToNeed(x, HasBerryTree, PathKind::BerryTree)),
    ]
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
fn HasFood(x: &CellKnowledge) -> bool { x.items.contains(&Item::Berry) }

#[allow(non_snake_case)]
fn HasWater(x: &CellKnowledge) -> bool { x.tile.can_drink() }

#[allow(non_snake_case)]
fn MoveToNeed<F: Fn(&CellKnowledge) -> bool>(
        ctx: &mut Ctx, f: F, kind: PathKind) -> Option<Action> {
    ensure_neighborhood(ctx);

    let n = &ctx.neighborhood;
    let Ctx { known, .. } = *ctx;
    for &(point, _) in n.blocked.iter().chain(&n.visited) {
        if !known.get(point).cell().map(&f).unwrap_or(false) { continue; }
        return FindPath(ctx, point, kind);
    }
    None
}

#[allow(non_snake_case)]
fn EatFoodNearby(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let valid = |p: Point| known.get(p).cell().map(HasFood).unwrap_or(false);
    for &dir in [dirs::NONE].iter().chain(&dirs::ALL) {
        let target = pos + dir;
        if !valid(target) { continue; }
        let gain = ctx.env.rng.gen_range(HUNGER_GAIN);
        ctx.blackboard.till_hunger = min(ctx.blackboard.till_hunger + gain, MAX_HUNGER);
        return Some(Action::Eat(EatAction { target, item: Some(Item::Berry) }));
    }
    None
}

#[allow(non_snake_case)]
fn DrinkWaterNearby(ctx: &mut Ctx) -> Option<Action> {
    let Ctx { known, pos, .. } = *ctx;
    let valid = |p: Point| known.get(p).cell().map(HasWater).unwrap_or(false);
    for &dir in [dirs::NONE].iter().chain(&dirs::ALL) {
        let target = pos + dir;
        if !valid(target) { continue; }
        let gain = ctx.env.rng.gen_range(THIRST_GAIN);
        ctx.blackboard.till_thirst = min(ctx.blackboard.till_thirst + gain, MAX_THIRST);
        return Some(Action::Drink(pos + dir));
    }
    None
}

#[allow(non_snake_case)]
fn Eat() -> impl Bhv {
    pri![
        "EatFood",
        act!("EatFoodNearby", EatFoodNearby),
        act!("Follow(MoveToFood)", |x| FollowPath(x, PathKind::MoveToFood)),
        act!("MoveToFood", |x| MoveToNeed(x, HasFood, PathKind::MoveToFood)),
        ForageForBerries(),
    ]
    .on_enter(|x| x.blackboard.finding_food_ = true)
    .on_exit(|x| x.blackboard.finding_food_ = false)
}

#[allow(non_snake_case)]
fn Drink() -> impl Bhv {
    pri![
        "DrinkWater",
        act!("DrinkWaterNearby", DrinkWaterNearby),
        act!("Follow(MoveToWater)", |x| FollowPath(x, PathKind::MoveToWater)),
        act!("MoveToWater", |x| MoveToNeed(x, HasWater, PathKind::MoveToWater)),
    ]
    .on_enter(|x| x.blackboard.finding_water = true)
    .on_exit(|x| x.blackboard.finding_water = false)
}

//////////////////////////////////////////////////////////////////////////////

// Top-level tree

#[allow(non_snake_case)]
fn Root() -> Box<dyn Bhv> {
    Box::new(pri![
        "Root",
        cb!("TickBasicNeeds", TickBasicNeeds),
        act!("Follow(LookAround)", FollowDirs),
        util!["BasicNeeds", (Hunger, Eat()), (Thirst, Drink())],
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
        let mut debug = Debug { depth: 0, slice };
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
