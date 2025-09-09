use std::cmp::{max, min};
use std::f64::consts::TAU;
use std::fmt::Debug;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_distr::num_traits::Pow;

use crate::base::{HashMap, LOS, Point, Slice, RNG, dirs, sample, weighted};
use crate::entity::{Entity, EID};
use crate::game::{MOVE_NOISE_RADIUS, Item, move_ready};
use crate::game::{Action, AttackAction, CallForHelpAction, EatAction, MoveAction};
use crate::knowledge::{CellKnowledge, Knowledge, EntityKnowledge};
use crate::knowledge::{Event, EventData, Scent, Sense, Timedelta, Timestamp, UID};
use crate::list::{Handle, List};
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

// TODO: Express all flight logic in terms of time, not turns. We can't make
// this change until we're back to a predator-prey environment.
//
// Introduce time-based logic for recomputing flight paths, too.
const FLIGHT_PATH_TURNS: i32 = 8;
const MIN_FLIGHT_TURNS: i32 = 16;
const MAX_FLIGHT_TURNS: i32 = 64;
const ACTIVE_THREAT_TIME: Timedelta = Timedelta::from_seconds(96.);

const CALL_FOR_HELP_LIMIT: Timedelta = Timedelta::from_seconds(4.);
const CALL_FOR_HELP_RETRY: Timedelta = Timedelta::from_seconds(24.);

const MIN_SEARCH_TIME: Timedelta = Timedelta::from_seconds(24.);
const MAX_SEARCH_TIME: Timedelta = Timedelta::from_seconds(48.);

const MAX_TRACKING_TIME: Timedelta = Timedelta::from_seconds(64.);
const SCENT_AGE_PENALTY: Timedelta = Timedelta::from_seconds(1.);

const WANDER_TURNS: f64 = 2.;

//////////////////////////////////////////////////////////////////////////////

// Basic helpers

fn safe_inv_l2(point: Point) -> f64 {
    if point == Point::default() { return 0. }
    (point.len_l2_squared() as f64).sqrt().recip()
}

fn is_hiding_place(ctx: &Context, point: Point) -> bool {
    if ctx.shared.threats.hostile.iter().any(
        |x| (x.pos - point).len_l1() <= 1) { return false; }
    let cell = ctx.known.get(point);
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

    fn sneak(&mut self, ctx: &Context, target: Point, turns: f64) -> Option<Action> {
        let Context { known, pos, .. } = *ctx;

        let check = |p: Point| {
            if !is_hiding_place(ctx, p) { return Status::Blocked; }
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

// Threat state

type ThreatHandle = Handle<Threat>;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
enum TID { EID(EID), UID(UID) }

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ThreatStatus { Hostile, Friendly, Neutral, Scanned, Unknown }

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
enum FightOrFlight { Fight, Flight, #[default] Safe }

#[derive(Clone)]
struct Threat {
    pos: Point,
    time: Timestamp,
    sense: Sense,
    status: ThreatStatus,
    combat: Timestamp,

    // Stats:
    hp: f64,
    delta: i32,

    // Flags:
    asleep: bool,
    seen: bool,
}

#[derive(Default)]
struct ThreatState {
    threats: List<Threat>,
    threat_index: HashMap<TID, ThreatHandle>,

    // Summaries used for flight pathing.
    hostile: Vec<Threat>,
    unknown: Vec<Threat>,

    // Fight-or-flight.
    reset: bool,
    state: FightOrFlight,
    our_strength: f64,
    call_strength: f64,
    their_strength: f64,
    win_probability: f64,

    // Calling for help.
    last_call: Timestamp,
    call_for_help: bool,
}

impl Threat {
    fn prior(me: &Entity) -> Self {
        Self {
            pos: Default::default(),
            time: Default::default(),
            sense: Sense::Sight,
            status: ThreatStatus::Unknown,
            combat: Default::default(),

            // Stats:
            hp: 1.,
            delta: if me.predator { -1 } else { 1 },

            // Flags:
            asleep: false,
            seen: false,
        }
    }
}

impl Threat {
    fn debug(&self, slice: &mut Slice, time: Timestamp) {
        let mut flags = vec![];
        if self.asleep { flags.push("Asleep"); }
        if self.seen { flags.push("Seen "); }
        if flags.is_empty() { flags.push("None"); }

        slice.write_str("    Threat:").newline();
        slice.write_str(&format!("      age: {:?}", time - self.time)).newline();
        slice.write_str(&format!("      pos: {:?}", self.pos)).newline();
        slice.write_str(&format!("      sense: {:?}", self.sense)).newline();
        slice.write_str(&format!("      status: {:?}", self.status)).newline();
        slice.write_str(&format!("      combat: {:?}", time - self.combat)).newline();
        slice.write_str(&format!("      hp: {:.2}", self.hp)).newline();
        slice.write_str(&format!("      delta: {}", self.delta)).newline();
        slice.write_str(&format!("      flags: {}", flags.join(" | "))).newline();
    }

    fn merge_from(&mut self, other: &Threat) {
        // No need to update any fields that we unconditionally update in
        // update_for_event, since we merge right before processing an event.
        self.sense = other.sense;
        self.seen |= other.seen;
        self.update_status(other.status);
    }

    fn update_status(&mut self, status: ThreatStatus) {
        self.status = min(self.status, status);
    }

    fn update_for_event(&mut self, me: &Entity, event: &Event) {
        self.pos = event.point;
        self.time = event.time;
        self.sense = event.sense;
        self.asleep = false;

        match &event.data {
            EventData::Attack(x) => {
                let attacked = x.target == Some(me.eid);
                if attacked { self.update_status(ThreatStatus::Hostile); }
                self.combat = event.time;
            },
            EventData::CallForHelp(_) => {
                if !me.predator { self.update_status(ThreatStatus::Friendly); }
                self.combat = event.time;
            },
            EventData::Forget(_) => {},
            EventData::Move(_) => {},
            EventData::Spot(_) => {},
        }
    }

    fn update_for_sighting(&mut self, me: &Entity, other: &EntityKnowledge) {
        self.pos = other.pos;
        self.time = other.time;
        self.sense = other.sense;

        self.hp = other.hp;
        self.delta = other.delta;

        self.asleep = other.asleep;
        self.seen = true;

        let status = if other.player {
            ThreatStatus::Neutral
        } else if other.delta > 0 {
            ThreatStatus::Hostile
        } else if !me.predator && other.delta == 0 {
            ThreatStatus::Friendly
        } else {
            ThreatStatus::Neutral
        };
        self.update_status(status);
    }
}

impl ThreatState {
    fn debug(&self, slice: &mut Slice, time: Timestamp) {
        slice.write_str("  ThreatState:").newline();
        slice.newline();
        slice.write_str(&format!("    reset: {}", self.reset)).newline();
        slice.write_str(&format!("    state: {:?}", self.state)).newline();
        slice.write_str(&format!("    our_strength: {:.2}", self.our_strength)).newline();
        slice.write_str(&format!("    call_strength: {:.2}", self.call_strength)).newline();
        slice.write_str(&format!("    their_strength: {:.2}", self.their_strength)).newline();
        slice.write_str(&format!("    win_probability: {:.2}", self.win_probability)).newline();
        slice.write_str(&format!("    call_for_help: {}", self.call_for_help)).newline();
        slice.write_str(&format!("    last_call: {:?}", time - self.last_call)).newline();
        slice.newline();
        for x in &self.threats { x.debug(slice, time) }
    }

    fn update(&mut self, me: &Entity, prev_time: Timestamp) {
        for event in &me.known.events {
            let Some(threat) = self.get_by_event(me, event) else { continue };
            threat.update_for_event(me, event);
        }
        for other in &me.known.entities {
            if !other.visible { break; }
            let Some(threat) = self.get_by_entity(me, other.eid) else { continue };
            threat.update_for_sighting(me, other);
        }

        self.hostile.clear();
        self.unknown.clear();
        self.reset = false;

        let was_active = self.state != FightOrFlight::Safe;
        let mut active = was_active;
        let mut hidden_hostile = 0;
        let mut seen_hostile = 0;

        for x in &self.threats {
            if me.known.time - x.time > ACTIVE_THREAT_TIME { break; }

            if x.status == ThreatStatus::Hostile { self.hostile.push(x.clone()); }
            if x.status == ThreatStatus::Unknown { self.unknown.push(x.clone()); }

            if x.status == ThreatStatus::Hostile && !x.seen { hidden_hostile += 1; }
            if x.status == ThreatStatus::Hostile && x.seen { seen_hostile += 1; }
        }
        if !self.hostile.is_empty() {
            self.hostile.extend_from_slice(&self.unknown);
            self.hostile.sort_by_key(|x| me.known.time - x.time);
        }
        if let Some(x) = self.hostile.first() && x.time > prev_time {
            active = true;
            self.reset = true;
        } else if self.hostile.is_empty() {
            active = false;
        }

        let strength = |x: &Threat| { 1.5f64.powi(x.delta.signum()) * x.hp };
        let mut hidden_count = max(hidden_hostile - seen_hostile, 0);
        let mut our_strength = me.hp_fraction();
        let mut call_strength = our_strength;
        let mut their_strength = 0.;

        for x in &self.threats {
            if me.known.time - x.time > ACTIVE_THREAT_TIME { break; }

            if x.status == ThreatStatus::Hostile {
                if !x.seen && hidden_count == 0 { continue; }
                if !x.seen { hidden_count -= 1; }
                their_strength += strength(x);
            } else if x.status == ThreatStatus::Friendly {
                let base = strength(x);
                let delay = me.known.time - x.combat;
                let ratio = delay.nsec() as f64 / ACTIVE_THREAT_TIME.nsec() as f64;
                let decay = if ratio > 1. { 0. } else { 1. - ratio };
                our_strength += base * decay;

                let can_call = me.known.time - x.time <= CALL_FOR_HELP_LIMIT;
                call_strength += if can_call { base } else { base * decay };
            }
        }

        self.our_strength = our_strength;
        self.call_strength = call_strength;
        self.their_strength = their_strength;
        self.win_probability = our_strength / (our_strength + their_strength);
        let p = self.win_probability;

        if active && !was_active {
            self.state = if p > 0.5 { FightOrFlight::Fight } else { FightOrFlight::Flight };
        } else if active {
            if p > 0.6 { self.state = FightOrFlight::Fight; }
            if p < 0.4 { self.state = FightOrFlight::Flight; }
        } else {
            self.state = FightOrFlight::Safe;
        }

        self.call_for_help = false;
        if self.state == FightOrFlight::Flight &&
           (me.known.time - self.last_call) > CALL_FOR_HELP_RETRY {
            let win_post_call = call_strength / (call_strength + their_strength);
            self.call_for_help = win_post_call > 0.6;
            if self.call_for_help { self.state = FightOrFlight::Fight; }
        }

        debug_assert!(self.check_invariants());
    }

    fn get_by_entity(&mut self, me: &Entity, eid: EID) -> Option<&mut Threat> {
        let handle = self.get_by_tid(me, TID::EID(eid))?;
        Some(&mut self.threats[handle])
    }

    fn get_by_event(&mut self, me: &Entity, event: &Event) -> Option<&mut Threat> {
        let tid = event.eid.map(|x| TID::EID(x)).or(event.uid.map(|x| TID::UID(x)))?;

        if let EventData::Forget(_) = &event.data {
            if let Some(x) = self.threat_index.remove(&tid) { self.threats.remove(x); }
            return None;
        }

        let handle = self.get_by_tid(me, tid)?;

        if event.eid.is_some() && let Some(x) = event.uid &&
           let Some(x) = self.threat_index.remove(&TID::UID(x)) {
            let existing = self.threats.remove(x);
            self.threats[handle].merge_from(&existing);
        }

        Some(&mut self.threats[handle])
    }

    fn get_by_tid(&mut self, me: &Entity, tid: TID) -> Option<ThreatHandle> {
        if self.known_good(me, tid) { return None; }

        Some(*self.threat_index.entry(tid).and_modify(|&mut x| {
            self.threats.move_to_front(x);
        }).or_insert_with(|| {
            self.threats.push_front(Threat::prior(me))
        }))
    }

    fn known_good(&self, me: &Entity, tid: TID) -> bool {
        let TID::EID(x) = tid else { return false };
        x == me.eid || me.known.entity(x).map(|x| x.friend).unwrap_or(false)
    }

    fn check_invariants(&self) -> bool {
        // Check that threats are sorted by time.
        let check_sorted = |xs: Vec<Timestamp>| {
            let mut last = Timestamp::default();
            xs.into_iter().rev().for_each(|x| { assert!(x >= last); last = x; });
        };
        check_sorted(self.threats.iter().map(|x| x.time).collect());

        // Check that every threat is indexed in the index.
        assert!(self.threats.len() == self.threat_index.len());
        let mut handles = HashMap::default();
        for (&tid, &handle) in self.threat_index.iter() {
            assert!(handles.insert(handle, tid).is_none());
            let _ = &self.threats[handle];
        }
        true
    }
}

//////////////////////////////////////////////////////////////////////////////

// Shared state

struct SharedAIState {
    till_assess: i32,
    till_hunger: i32,
    till_thirst: i32,
    till_rested: i32,
    turn_time: Timestamp,
    prev_time: Timestamp,
    threats: ThreatState,
}

impl SharedAIState {
    fn new(predator: bool, rng: &mut RNG) -> Self {
        let max_hunger = if predator { MAX_HUNGER_CARNIVORE } else { MAX_HUNGER_HERBIVORE };
        Self {
            till_assess: rng.random_range(0..MAX_ASSESS),
            till_hunger: rng.random_range(0..max_hunger),
            till_rested: rng.random_range(0..MAX_RESTED),
            till_thirst: rng.random_range(0..MAX_THIRST),
            prev_time: Timestamp::default(),
            turn_time: Timestamp::default(),
            threats: ThreatState::default(),
        }
    }

    fn debug(&self, slice: &mut Slice, time: Timestamp) {
        slice.write_str("  SharedAIState:").newline();
        slice.write_str(&format!("    till_assess: {}", self.till_assess)).newline();
        slice.write_str(&format!("    till_hunger: {}", self.till_hunger)).newline();
        slice.write_str(&format!("    till_thirst: {}", self.till_thirst)).newline();
        slice.write_str(&format!("    till_rested: {}", self.till_rested)).newline();
        slice.newline();
        self.threats.debug(slice, time);
    }


    fn update(&mut self, entity: &Entity) {
        if !entity.asleep {
            self.till_assess = max(self.till_assess - 1, 0);
            self.till_hunger = max(self.till_hunger - 1, 0);
            self.till_thirst = max(self.till_thirst - 1, 0);
            self.till_rested = max(self.till_rested - 1, 0);
        }
        self.prev_time = self.turn_time;
        self.turn_time = entity.known.time;
        self.threats.update(entity, self.prev_time);
    }
}

//////////////////////////////////////////////////////////////////////////////

// Strategy-based planning

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Priority { Survive, EatMeat, Hunt, SpotThreats, SatisfyNeeds, Explore, Skip }

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
    fn debug(&self, slice: &mut Slice, time: Timestamp);
    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i64);
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

    fn action(&self, target: Point) -> Action {
        match self.need {
            BasicNeed::EatMeat => Action::Eat(EatAction { target, item: Some(Item::Corpse) }),
            BasicNeed::EatPlants => Action::Eat(EatAction { target, item: None }),
            BasicNeed::Drink => Action::Drink(target),
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
        if cell.visible() && !self.satisfies_need(cell.cell()) {
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

    fn debug(&self, slice: &mut Slice, _: Timestamp) {
        slice.write_str(&format!("BasicNeed: {:?}", self.need)).newline();
    }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i64) {
        // Keep track of the the last-seen cell that satisfies our need.
        self.update_last(ctx);

        // Clear the cached path, if it's no longer good.
        let known = ctx.known;
        let valid = |p: Point| self.satisfies_need(known.get(p).cell());
        if !(self.path.check(ctx) && valid(self.path.steps[0])) { self.path.reset(); }

        // Compute a priority for satisfying this need.
        if self.last.is_none() { return (Priority::Skip, 0); }
        let (turns_left, timeout) = (*self.turns_left(ctx), self.timeout());
        let cutoff = max(timeout / 2, 1);
        if turns_left >= cutoff { return (Priority::Skip, 0); }
        let tiebreaker = if last { -1 } else { 100 * turns_left / cutoff };
        (self.priority(), tiebreaker as i64)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let turns = WANDER_TURNS;
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let Context { known, pos, .. } = *ctx;
        let valid = |p: Point| self.satisfies_need(known.get(p).cell());
        for &dir in [dirs::NONE].iter().chain(&dirs::ALL) {
            if valid(pos + dir) {
                *self.turns_left(ctx) = max(*self.turns_left(ctx), self.timeout());
                return Some(self.action(pos + dir));
            }
        }

        ensure_neighborhood(ctx);

        let n = &ctx.neighborhood;
        for &(point, _) in n.blocked.iter().chain(&n.visited) {
            if !self.satisfies_need(known.get(point).cell()) { continue; }

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

    fn debug(&self, slice: &mut Slice, _: Timestamp) {
        slice.write_str("Rest").newline();
    }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i64) {
        if !self.path.check(ctx) { self.path.reset(); }

        // Compute a priority for satisfying this need.
        let turns_left = ctx.shared.till_rested;
        let cutoff = max(MAX_RESTED / if last { 1 } else { 2 }, 1);
        if turns_left >= cutoff { return (Priority::Skip, 0); }
        let tiebreaker = if last { -1 } else { 100 * turns_left / cutoff };
        (Priority::SatisfyNeeds, tiebreaker as i64)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let turns = WANDER_TURNS;
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let Context { known, pos, .. } = *ctx;
        if is_hiding_place(ctx, pos) {
            ctx.shared.till_rested += 1;
            return Some(Action::Rest);
        }

        ensure_neighborhood(ctx);

        for &(point, _) in &ctx.neighborhood.visited {
            if !is_hiding_place(ctx, point) { continue; }
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

    fn debug(&self, slice: &mut Slice, _: Timestamp) {
        slice.write_str("Assess").newline();
    }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i64) {
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
            ctx.shared.till_assess = rng.random_range(0..MAX_ASSESS);
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

    fn debug(&self, slice: &mut Slice, _: Timestamp) {
        slice.write_str("Explore").newline();
    }

    fn bid(&mut self, ctx: &mut Context, last: bool) -> (Priority, i64) {
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
        self.path.start(ctx, target, turns)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

struct TrackTarget {
    fresh: bool,
    scent: Scent,
}

#[derive(Default)]
struct TrackStrategy {
    path: CachedPath,
    target: Option<TrackTarget>,
}

impl Strategy for TrackStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice, time: Timestamp) {
        slice.write_str("Track").newline();
        if let Some(x) = &self.target {
            slice.write_str(&format!("    age: {:?}", time - x.scent.time)).newline();
            slice.write_str(&format!("    pos: {:?}", x.scent.pos)).newline();
            slice.write_str(&format!("    fresh: {}", x.fresh)).newline();
        }
    }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i64) {
        let predator = ctx.entity.predator;
        let turns_left = ctx.shared.till_hunger;
        let cutoff = max(MAX_HUNGER_CARNIVORE / 2, 1);
        if predator && turns_left >= cutoff { return (Priority::Skip, 0); }

        if !self.path.check(ctx) { self.path.reset(); }

        if let Some(x) = &mut self.target { x.fresh = false; }

        for &scent in &ctx.known.scents {
            let (fresh, newer) = match &self.target {
                Some(x) => (scent.pos != x.scent.pos, scent.time > x.scent.time),
                None => (true, true),
            };
            if !newer { continue; }

            if fresh { self.path.reset(); }
            self.target = Some(TrackTarget { fresh, scent });
        }

        let Some(x) = &self.target else { return (Priority::Skip, 0) };

        let age = ctx.known.time - x.scent.time;
        if age > MAX_TRACKING_TIME { self.target = None; }

        let Some(_) = &self.target else { return (Priority::Skip, 0) };

        (Priority::Hunt, (age + SCENT_AGE_PENALTY).nsec())
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let Some(x) = &self.target else { return None; };

        if x.fresh { return Some(Action::SniffAround); }

        let turns = WANDER_TURNS;
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let age = ctx.known.time - x.scent.time;
        search_around(ctx, &mut self.path, age, Point(0, 0), x.scent.pos, turns)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

#[derive(Default)]
struct CallForHelpStrategy {}

impl Strategy for CallForHelpStrategy {
    fn get_path(&self) -> &[Point] { &[] }

    fn debug(&self, slice: &mut Slice, _: Timestamp) {
        slice.write_str("CallForHelp").newline();
    }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i64) {
        let threats = &ctx.shared.threats;
        if !threats.call_for_help { return (Priority::Skip, 0); }

        let fight = threats.state == FightOrFlight::Fight;
        (Priority::Survive, if fight { 0 } else { 1 })
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let time = ctx.shared.turn_time;
        for x in &mut ctx.shared.threats.threats {
            if time - x.time > CALL_FOR_HELP_LIMIT { break; }
            if x.status == ThreatStatus::Friendly { x.combat = time; }
        }
        ctx.shared.threats.last_call = time;
        Some(Action::CallForHelp(CallForHelpAction { targets: vec![] }))
    }

    fn reject(&mut self) {}
}

//////////////////////////////////////////////////////////////////////////////

// TODO(shaunak): If we're fleeing, spot a potential ally, call for help, and
// then want to fight, we may start the chase "running away" because of the
// bias towards our current direction.
//
// TODO(shaunak): If an entity is killed and we don't see it die, we may keep
// hunting for it. That's okay, but if we run out of chase time (48s), we may
// switch to fleeing from it (flight time goes up to 64 turns or 64-91s).
//
// Instead, if we're in FightOrFlight mode Fight and we time out, we should
// end with an assess (like Flight already does) and then mark targets safe.
#[derive(Debug)]
struct ChaseTarget {
    bias: Point,
    last: Point,
    time: Timestamp,
    steps: i32,
}

#[derive(Default)]
struct ChaseStrategy {
    path: CachedPath,
    target: Option<ChaseTarget>,
}

impl Strategy for ChaseStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice, time: Timestamp) {
        slice.write_str("Chase").newline();
        if let Some(x) = &self.target {
            slice.write_str(&format!("    age: {:?}", time - x.time)).newline();
            slice.write_str(&format!("    bias: {:?}", x.bias)).newline();
            slice.write_str(&format!("    last: {:?}", x.last)).newline();
            slice.write_str(&format!("    steps: {}", x.steps)).newline();
        }
    }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i64) {
        let prev = self.target.take();

        let predator = ctx.entity.predator;
        let turns_left = ctx.shared.till_hunger;
        let cutoff = max(MAX_HUNGER_CARNIVORE, 1); // / 2, 1);

        let state = ctx.shared.threats.state;
        let fight = state == FightOrFlight::Fight;
        let fight_for_food = predator && turns_left < cutoff;
        let fight_for_life = state != FightOrFlight::Safe;
        if !fight_for_food && !fight_for_life { return (Priority::Skip, 0); }

        let Context { known, pos, .. } = *ctx;
        let mut targets: Vec<(Point, Timestamp)> = vec![];
        if fight_for_life {
            targets = ctx.shared.threats.hostile.iter().map(
                |x| (x.pos, x.time)).collect();
        }
        if targets.is_empty() && fight_for_food {
            targets = known.entities.iter().filter_map(
                |x| if x.delta < 0 { Some((x.pos, x.time)) } else { None }).collect();
        }
        let Some(x) = targets.first() else { return (Priority::Skip, 0); };
        if (known.time - x.1) >= MAX_SEARCH_TIME { return (Priority::Skip, 0); }

        let target = *targets.select_nth_unstable_by_key(
            0, |x| (known.time - x.1, (pos - x.0).len_l2_squared())).1;
        let reset = target.1 > ctx.shared.prev_time;
        if reset || !self.path.check(ctx) { self.path.reset(); }

        let _age = known.time - target.1;
        let (last, time) = target;
        let (bias, steps) = if !reset && let Some(x) = prev {
            (x.bias, x.steps + 1)
        } else {
            (last - pos, 0)
        };
        self.target = Some(ChaseTarget { bias, last, time, steps });

        let priority = if fight_for_life { Priority::Survive } else { Priority::Hunt };
        let strength = if fight_for_life && fight { 1 } else { 2 };
        (priority, strength)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let ChaseTarget { bias, last, time, steps } = *self.target.as_ref()?;
        let age = ctx.known.time - time;
        let visible = age == Timedelta::default();
        if visible && let Some(x) = attack_target(ctx, last) { return Some(x); }

        let turns = if age >= MIN_SEARCH_TIME { WANDER_TURNS } else { 1. };
        if let Some(x) = self.path.follow(ctx, turns) { return Some(x); }

        let search_nearby = steps > bias.len_l1();
        let center = if search_nearby { ctx.pos } else { last };
        search_around(ctx, &mut self.path, age, bias, center, turns)
    }

    fn reject(&mut self) { self.path.reset(); }
}

//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum FlightStage { Flee, Hide, Done }

struct FlightStrategy {
    dirs: Vec<Point>,
    path: CachedPath,
    stage: FlightStage,
    needs_path: bool,
    since_path: i32,
    since_seen: i32,
    turn_limit: i32,
}

impl FlightStrategy {
    fn done(&self) -> bool { self.stage == FlightStage::Done }

    fn mark_flight_successful(&mut self, ctx: &mut Context) {
        ctx.shared.threats.state = FightOrFlight::Safe;
        self.turn_limit = MIN_FLIGHT_TURNS;
        self.stage = FlightStage::Done;
    }

    fn look(&mut self, ctx: &mut Context) -> Option<Action> {
        // TODO(shaunak): look should fail if we can see the enemy and if we're
        // not in a hidden cell, or if we are in a hidden cell but were still
        // attacked. (The latter should cause us to flag the enemy as "we can't
        // hide from this foe" - maybe they're psychic or can see in the dark?)
        let (pos, rng) = (ctx.pos, &mut ctx.env.rng);
        self.path.reset();

        let threats = &ctx.shared.threats.hostile;

        if self.dirs.is_empty() {
            let dirs: Vec<_> = threats.iter().map(|x| x.pos - pos).collect();
            self.dirs = assess_directions(&dirs, ASSESS_TURNS_FLIGHT, rng);
            self.dirs.reverse();
        }

        // Look in the next guess at a threat direction, unless there's a
        // visible threat, in which case we'll watch the closest one.
        let mut dir = self.dirs.pop()?;
        let mut visible_threats: Vec<_> = threats.iter().filter_map(|x| {
            let visible = x.time == ctx.shared.turn_time;
            if visible { Some(x.pos) } else { None }
        }).collect();
        if !visible_threats.is_empty() {
            let threat = *visible_threats.select_nth_unstable_by_key(
                0, |&p| ((p - pos).len_l2_squared(), p.0, p.1)).1;
            dir = threat - pos;
        }

        if self.dirs.is_empty() {
            ctx.shared.till_assess = rng.random_range(0..MAX_ASSESS);
            self.mark_flight_successful(ctx);
        }
        Some(Action::Look(dir))
    }
}

impl Default for FlightStrategy {
    fn default() -> Self {
        Self {
            dirs: vec![],
            path: Default::default(),
            stage: FlightStage::Done,
            needs_path: false,
            since_path: 0,
            since_seen: 0,
            turn_limit: MIN_FLIGHT_TURNS,
        }
    }
}

impl Strategy for FlightStrategy {
    fn get_path(&self) -> &[Point] { &self.path.steps }

    fn debug(&self, slice: &mut Slice, _: Timestamp) {
        slice.write_str("Flight").newline();
        if self.stage == FlightStage::Done { return; }

        slice.write_str(&format!("    stage: {:?}", self.stage)).newline();
        slice.write_str(&format!("    needs_path: {}", self.needs_path)).newline();
        slice.write_str(&format!("    since_path: {}", self.since_path)).newline();
        slice.write_str(&format!("    since_seen: {}", self.since_seen)).newline();
        slice.write_str(&format!("    turn_limit: {}", self.turn_limit)).newline();
    }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i64) {
        let reset = ctx.shared.threats.reset;
        let fight = ctx.shared.threats.state == FightOrFlight::Fight;
        if self.done() && !reset { return (Priority::Skip, 0); }

        // Flight can end because if the enemies were defeated in combat.
        //
        // (Should we also reset flight state if we turn back to fight them?)
        if !self.done() && ctx.shared.threats.hostile.is_empty() {
            self.mark_flight_successful(ctx);
            return (Priority::Skip, 0);
        }

        let looking = !self.dirs.is_empty();
        let hiding = is_hiding_place(ctx, ctx.pos);
        let stage = if hiding { FlightStage::Hide } else { FlightStage::Flee };

        if reset {
            self.dirs.clear();
            self.needs_path = true;
        }

        let path_stale = self.since_path >= FLIGHT_PATH_TURNS;
        let compute_path = self.needs_path && (self.stage > stage || path_stale);
        if compute_path || !self.path.check(ctx) ||
           (self.stage == FlightStage::Hide && self.path.steps.iter().any(
                |&x| !is_hiding_place(ctx, x))) {
            self.path.reset();
            self.needs_path = false;
        }

        self.stage = if reset { min(self.stage, stage) } else { self.stage };
        self.since_seen = if reset { 0 } else { self.since_seen + 1 };
        self.turn_limit = if reset && looking {
            min(2 * self.turn_limit, MAX_FLIGHT_TURNS)
        } else {
            self.turn_limit
        };
        (Priority::Survive, if fight { 2 } else { 0 })
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let last_since_path = self.since_path;
        self.since_path = 0;

        if !self.dirs.is_empty() || self.since_seen >= self.turn_limit {
            self.path.reset();
            self.needs_path = false;
            return self.look(ctx);
        }

        let all_asleep = ctx.shared.threats.hostile.iter().all(|x| x.asleep);
        let pick_turns = |stage: FlightStage| {
            let sneak = all_asleep || stage == FlightStage::Hide;
            if sneak { WANDER_TURNS } else { 1. }
        };

        let turns = pick_turns(self.stage);
        if let Some(x) = self.path.follow(ctx, turns) {
            self.since_path = last_since_path + 1;
            return Some(x);
        }

        if !all_asleep && is_hiding_place(ctx, ctx.pos) {
            ensure_sneakable(ctx);

            if let Some(target) = flight_cell(ctx, true) {
                self.stage = FlightStage::Hide;
                if target == ctx.pos { return self.look(ctx); }
                return self.path.sneak(ctx, target, WANDER_TURNS);
            }
        }

        ensure_neighborhood(ctx);

        if let Some(target) = flight_cell(ctx, false) {
            self.stage = FlightStage::Flee;
            if target == ctx.pos { return self.look(ctx); }
            return self.path.start(ctx, target, pick_turns(self.stage));
        }

        self.stage = FlightStage::Flee;
        self.look(ctx)
    }

    fn reject(&mut self) {
        self.dirs.clear();
        self.path.reset();
        self.needs_path = false;
    }
}

//////////////////////////////////////////////////////////////////////////////

#[derive(Default)]
struct LookForThreatsStrategy {
    dirs: Vec<Point>,
}

impl Strategy for LookForThreatsStrategy {
    fn get_path(&self) -> &[Point] { &[] }

    fn debug(&self, slice: &mut Slice, _: Timestamp) {
        slice.write_str("LookForThreats").newline();
    }

    fn bid(&mut self, ctx: &mut Context, _: bool) -> (Priority, i64) {
        let threats = &ctx.shared.threats.unknown;
        if threats.is_empty() { return (Priority::Skip, 0); }
        (Priority::SpotThreats, 0)
    }

    fn accept(&mut self, ctx: &mut Context) -> Option<Action> {
        let (pos, rng) = (ctx.pos, &mut ctx.env.rng);

        if self.dirs.is_empty() {
            let dirs: Vec<_> = ctx.shared.threats.unknown.iter().map(
                |x| x.pos - pos).collect();
            self.dirs = assess_directions(&dirs, ASSESS_TURNS_FLIGHT, rng);
            self.dirs.reverse();
        }

        let dir = self.dirs.pop()?;
        if self.dirs.is_empty() {
            for threat in &mut ctx.shared.threats.threats {
                threat.status = min(threat.status, ThreatStatus::Scanned);
            }
            ctx.shared.till_assess = rng.gen_range(0..MAX_ASSESS);
        }
        Some(Action::Look(dir))
    }

    fn reject(&mut self) { self.dirs.clear(); }
}

//////////////////////////////////////////////////////////////////////////////

// Helpers for ChaseStrategy

fn attack_target(ctx: &mut Context, target: Point) -> Option<Action> {
    let Context { entity, known, pos: source, .. } = *ctx;
    if source == target { return None; }

    let attacks = &ctx.entity.species.attacks;
    if attacks.is_empty() { return None; }

    let attack = sample(attacks, ctx.env.rng);
    let range = attack.range;
    let valid = |x| has_los(x, target, known, range);
    if move_ready(entity) && valid(source) {
        return Some(Action::Attack(AttackAction { attack, target }));
    }

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
        ctx: &mut Context, target: Point, range: i32, valid: F) -> Option<Action> {
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
        let distance = if obscured { 1 } else { min(range, MOVE_NOISE_RADIUS) };

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

fn search_around(ctx: &mut Context, path: &mut CachedPath, age: Timedelta,
                 bias: Point, center: Point, turns: f64) -> Option<Action> {
    let Context { known, pos, dir, .. } = *ctx;
    let inv_dir_l2 = safe_inv_l2(dir);
    let inv_bias_l2 = safe_inv_l2(bias);

    let is_search_candidate = |p: Point| {
        let cell = known.get(p);
        !cell.blocked() && cell.time_since_entity_visible() > age
    };

    if center != pos && is_search_candidate(center) {
        // HACK: CachedPath should handle this case for us.
        if (center - pos).len_l1() == 1 { return Some(Action::Look(center - pos)); }
        if let Some(x) = path.start(ctx, center, turns) { return Some(x); }
    }

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
    let target = select_target(&scores, ctx.env)?;
    path.start(ctx, target, turns)
}

//////////////////////////////////////////////////////////////////////////////

// Helpers for FlightStrategy

fn flight_cell(ctx: &mut Context, hiding: bool) -> Option<Point> {
    let Context { known, pos, .. } = *ctx;

    let threats = &ctx.shared.threats.hostile;
    let threat_inv_l2s: Vec<_> = threats.iter().map(
        |x| (x.pos, safe_inv_l2(pos - x.pos))).collect();
    let scale = 1. / DijkstraLength(Point(1, 0)) as f64;

    let score = |p: Point, source_distance: i32| -> f64 {
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
        let hidden = !hiding && is_hiding_place(ctx, p);

        let delta = p - pos;
        let inv_delta_l2 = safe_inv_l2(delta);
        let cos = delta.dot(pos - threat) as f64 * inv_delta_l2 * inv_l2;

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

    let min_score = score(pos, 0);
    let map = if hiding { &ctx.sneakable.visited } else { &ctx.neighborhood.visited };
    let scores: Vec<_> = map.iter().filter_map(|&(p, distance)| {
        let delta = score(p, distance) - min_score;
        if delta >= 0. { Some((p, delta)) } else { None }
    }).collect();
    select_target_softmax(&scores, ctx.env, 0.1)
}

//////////////////////////////////////////////////////////////////////////////

// Helpers used to implement strategies in general

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

fn ensure_sneakable(ctx: &mut Context) {
    if !ctx.sneakable.visited.is_empty() { return; }

    let known = ctx.known;

    let check = |p: Point| {
        if !is_hiding_place(ctx, p) { return Status::Blocked; }
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
    bids: Vec<(Priority, i64, usize)>,
    strategies: Vec<Box<dyn Strategy>>,
    shared: SharedAIState,
    last: i32,
}

impl AIState {
    pub fn new(predator: bool, rng: &mut RNG) -> Self {
        let strategies: Vec<Box<dyn Strategy>> = vec![
            //Box::new(TrackStrategy::default()),
            Box::new(ChaseStrategy::default()),
            Box::new(FlightStrategy::default()),
            Box::new(CallForHelpStrategy::default()),
            Box::new(LookForThreatsStrategy::default()),
            //Box::new(RestStrategy::default()),
            Box::new(BasicNeedsStrategy::new(BasicNeed::EatMeat)),
            Box::new(BasicNeedsStrategy::new(BasicNeed::EatPlants)),
            Box::new(BasicNeedsStrategy::new(BasicNeed::Drink)),
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
            strategy.debug(slice, self.shared.turn_time);
            slice.newline();
        }
        self.shared.debug(slice, self.shared.turn_time);
    }

    pub fn plan(&mut self, entity: &Entity, env: &mut AIEnv) -> Action {
        // Step 0: update some initial, deterministic shared state.
        let known = &*entity.known;
        let observations = known.cells.iter().take_while(|x| x.visible).collect();
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
