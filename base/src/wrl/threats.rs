use std::cmp::max;
use std::ops::Range;

use rand::Rng;

use crate::base::point::Point;
use crate::base::util::{HashMap, RNG};

use super::actor::Actor;
use super::debug::DebugLog;
use super::dex::Species;
use super::event::{Call, CallEvent, Event, EventData, Location, Sense, UID};
use super::knowledge::{ActorKnowledge, Knowledge, NoiseKnowledge};
use super::game::CALL_VOLUME;
use super::list::{Handle, List};
use super::time::Timestamp;

//////////////////////////////////////////////////////////////////////////////

pub const ACTIVE_THREAT_TURNS: i32 = 72;

pub const CALL_LIMIT_TURNS: i32 = 4;
pub const CALL_RETRY_TURNS: i32 = 16;

const PENALTY_DECAY: f64 = 0.5;
const PENALTY_RANGE: Range<f64> = 2.0..4.0;

fn timid(me: &Actor) -> bool { !me.species.predator() }

struct TimeDecay {
    current: Timestamp,
    timeout: Timestamp,
    turns: i32,
}

impl TimeDecay {
    fn new(me: &Actor, turns: i32) -> Self {
        let known = &*me.known;
        Self { current: known.time(), timeout: known.time_at_turn(turns), turns }
    }

    fn ratio(&self, time: Timestamp) -> f64 {
        let num = self.current - time;
        let den = self.current - self.timeout;
        num.nsec() as f64 / max(den.nsec(), 1) as f64
    }
}

//////////////////////////////////////////////////////////////////////////////

// Faction:

#[derive(Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
enum Confidence { #[default] Zero, Low, Mid, High }

#[derive(Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
enum Valence { Friendly, #[default] Neutral, Menacing, Hostile }

#[derive(Default)]
struct Faction {
    last_combat: Timestamp,
    penalty_score: f64,
    penalty_start: Timestamp,
    warning_count: i32,

    confidence: Confidence,
    valence: Valence,

    uids: Vec<UID>,
}

impl Faction {
    fn debug(&self, debug: &mut DebugLog, known: &Knowledge) {
        debug.append("Faction:");
        debug.indent(1, |debug| {
            let warnings = if self.warning_count == 0 {
                " ".into()
            } else if self.warning_count == 1 {
                " - 1 warning".into()
            } else {
                format!(" - {} warnings", self.warning_count)
            };
            let status = format!("{:?}:{:?}", self.confidence, self.valence);

            debug.append(format!("combat: {}", known.debug_time(self.last_combat)));
            debug.append(format!("status: {}{}", status, warnings));
            debug.append(format!("penalty_score: {}", self.penalty_score));
            debug.append(format!("penalty_start: {}", known.debug_time(self.penalty_start)));
        });
        debug.newline();
    }

    fn known_hostile(&self) -> bool {
        self.confidence == Confidence::High && self.valence == Valence::Hostile
    }

    fn friendly_call(me: &Actor, x: &CallEvent) -> bool {
       x.species == me.species
    }

    fn friendly_call_for_help(me: &Actor, x: &CallEvent) -> bool {
        x.call == Call::Help && Self::friendly_call(me, x)
    }

    fn mark_scanned(&mut self) {
        self.merge_status(Confidence::Low, self.valence);
    }

    fn mark_warned(&mut self, me: &Actor, rng: &mut RNG) {
        let warnings = self.warning_count;
        let sample = rng.random::<f32>() * 2f32.powi(warnings);

        if sample > 0.5 {
            self.merge_status(Confidence::Mid, Valence::Hostile);
        }
        if timid(me) {
            self.penalty_score = rng.random_range(PENALTY_RANGE);
            self.penalty_start = me.known.time();
        }
        self.warning_count += 1;
    }

    fn merge_from(&mut self, other: &Faction) {
        self.last_combat = max(self.last_combat, other.last_combat);
        self.warning_count = max(self.warning_count, other.warning_count);
        self.merge_status(other.confidence, other.valence);
    }

    fn merge_status(&mut self, confidence: Confidence, valence: Valence) {
        if confidence == self.confidence {
            self.valence = max(valence, self.valence);
        } else if confidence > self.confidence {
            self.confidence = confidence;
            self.valence = valence;
        }
    }

    fn update_for_actor(&mut self, me: &Actor, other: &ActorKnowledge) {
        use Confidence::*;
        use Valence::*;

        if other.species.human() {
            self.merge_status(Low, Neutral);
            return;
        }

        let predator = other.delta > 0;
        let active = self.last_combat > me.known.time_at_turn(ACTIVE_THREAT_TURNS);

        let valence = if other.friend() {
            Friendly
        } else if other.rival() || (predator && active) {
            Hostile
        } else if predator {
            Menacing
        } else if timid(me) && me.species == other.species {
            Friendly
        } else {
            Neutral
        };
        self.merge_status(High, valence);
    }

    fn update_for_event(&mut self, me: &Actor, event: &Event) {
        use super::event::Call::*;
        use Confidence::*;
        use EventData::*;
        use Valence::*;

        match &event.data {
            Attack(x) if x.combat => {
                let us = x.target == Some(me.eid);
                let sure = us || self.confidence == High && self.valence >= Menacing;
                let confidence = if sure { High } else { Mid };

                self.merge_status(confidence, Hostile);
                self.last_combat = event.time;
            },
            Call(x) => {
                if Self::friendly_call(me, x) {
                    self.merge_status(High, Friendly);
                } else if x.call == Warning {
                    let valence = if timid(me) { Menacing } else { Neutral };
                    self.merge_status(Mid, valence);
                }
                if x.call == Help { self.last_combat = event.time; }
            },
            Attack(_) | Move(_) | Forget | Sniff | Spot => {},
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// FactionMap:

type FactionHandle = Handle<Faction>;

#[derive(Default)]
struct FactionMap {
    factions: List<Faction>,
    lookup: HashMap<UID, FactionHandle>,
    guess: Option<Location>,
}

impl FactionMap {
    // Public API:

    fn get_faction(&self, uid: UID) -> Option<&Faction> {
        self.lookup.get(&uid).map(|&x| &self.factions[x])
    }

    fn mut_faction(&mut self, uid: UID) -> Option<&mut Faction> {
        self.lookup.get(&uid).map(|&x| &mut self.factions[x])
    }

    fn observe_event(&mut self, me: &Actor, event: &Event) {
        let Some(faction) = self.faction_for_event(event) else { return };
        faction.update_for_event(me, event);
        if faction.known_hostile() { self.guess = None; }
    }

    fn update(&mut self, me: &Actor) {
        for actor in &me.known.actors {
            if !actor.sensed() { break; }
            if actor.eid == me.eid { continue; }

            let faction = self.faction_for_actor(actor);
            faction.update_for_actor(me, actor);
            if faction.known_hostile() { self.guess = None; }
        }
    }

    // Private helpers:

    fn faction_for_actor(&mut self, actor: &ActorKnowledge) -> &mut Faction {
        let result = self.faction_for_uid(actor.uid);
        &mut self.factions[result]
    }

    fn faction_for_event(&mut self, event: &Event) -> Option<&mut Faction> {
        let uid = event.uid;
        if matches!(event.data, EventData::Forget) {
            self.forget_uid(uid);
            return None;
        }

        let handle = self.faction_for_uid(uid);
        if let Some(old) = event.old && let Some(&merged) = self.lookup.get(&old) {
            self.merge_factions(handle, merged);
            self.forget_uid(old);
        }
        Some(&mut self.factions[handle])
    }

    fn faction_for_uid(&mut self, uid: UID) -> FactionHandle {
        *self.lookup.entry(uid).or_insert_with(|| {
            let faction = Faction { uids: vec![uid], ..Default::default() };
            self.factions.push_back(faction)
        })
    }

    fn forget_uid(&mut self, uid: UID) {
        let Some(handle) = self.lookup.remove(&uid) else { return };

        let faction = &mut self.factions[handle];
        faction.uids.retain(|&x| x != uid);

        if faction.uids.is_empty() { self.factions.remove(handle); }
    }

    fn merge_factions(&mut self, a: FactionHandle, b: FactionHandle) {
        if a == b { return; }

        let removed = self.factions.remove(b);
        self.factions[a].merge_from(&removed);

        for uid in removed.uids { self.lookup.insert(uid, a); }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Threat:

#[derive(Clone)]
pub struct Threat {
    pub uid: Option<UID>,
    pub loc: Location,
    pub sense: Sense,
    pub species: Option<&'static Species>,

    pub asleep: bool,
    pub warned: bool,
    pub seen: bool,
}

impl std::ops::Deref for Threat {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl Threat {
    fn for_actor(actor: &ActorKnowledge, faction: &Faction) -> Self {
        let species = Some(actor.species);
        let warned = faction.warning_count > 0;
        let (asleep, seen) = (actor.asleep(), true);
        let ActorKnowledge { uid, loc, sense, .. } = *actor;
        Self { uid: Some(uid), loc, sense, species, asleep, warned, seen }
    }

    fn for_noise(noise: &NoiseKnowledge, faction: &Faction) -> Self {
        let species = None;
        let (asleep, seen) = (false, false);
        let warned = faction.warning_count > 0;
        let NoiseKnowledge { uid, loc, sense, .. } = *noise;
        Self { uid: Some(uid), loc, sense, species, asleep, warned, seen }
    }

    fn for_guess(loc: Location) -> Self {
        let species = None;
        let (asleep, warned, seen) = (false, false, false);
        Self { uid: None, loc, sense: Sense::Sound, species, asleep, warned, seen }
    }
}

//////////////////////////////////////////////////////////////////////////////

// CombatThreat:

#[derive(Clone)]
struct Scoring {
    hp: f64,
    delta: i32,
    rival: bool,
}

#[derive(Clone)]
struct ScoredThreat<'a> {
    base: Threat,
    faction: &'a Faction,
    scoring: Scoring,
}

impl<'a> std::ops::Deref for ScoredThreat<'a> {
    type Target = Threat;
    fn deref(&self) -> &Self::Target { &self.base }
}

impl<'a> ScoredThreat<'a> {
    // Accessors:

    fn unknown(&self)   -> bool { self.faction.confidence == Confidence::Zero }
    fn uncertain(&self) -> bool { self.faction.confidence == Confidence::Low }

    fn hostile(&self)  -> bool { self.faction.valence == Valence::Hostile }
    fn friendly(&self) -> bool { self.faction.valence == Valence::Friendly }
    fn menacing(&self) -> bool { self.faction.valence == Valence::Menacing }

    // Scoring:

    fn strength(&self, decay: &TimeDecay) -> f64 {
        self.base_strength() + self.penalty_strength(decay)
    }

    fn base_strength(&self) -> f64 {
        if let Some(x) = self.species && x.human() { return 0.; }
        let delta = 1.75f64.powi(self.scoring.delta.signum());
        let rival = if self.scoring.rival { 0.25 } else { 1. };
        delta * rival * self.scoring.hp
    }

    fn penalty_strength(&self, decay: &TimeDecay) -> f64 {
        if self.faction.penalty_score <= 0. { return 0.; }
        if self.faction.penalty_start <= decay.timeout { return 0.; }

        let ratio = decay.ratio(self.faction.penalty_start);
        let value = self.faction.penalty_score - PENALTY_DECAY * ratio * decay.turns as f64;
        value.max(0.)
    }

    // Constructor:

    fn for_actor(actor: &ActorKnowledge, faction: &'a Faction) -> Self {
        let base = Threat::for_actor(actor, faction);
        let ActorKnowledge { hp, delta, .. } = *actor;
        let scoring = Scoring { hp, delta, rival: actor.rival() };
        Self { base, faction, scoring }
    }

    fn for_noise(me: &Actor, noise: &NoiseKnowledge, faction: &'a Faction) -> Self {
        let base = Threat::for_noise(noise, faction);
        let delta = if timid(me) { 1 } else { -1 };
        let hp = if faction.last_combat != Default::default() { 1. } else { 0. };
        let scoring = Scoring { hp, delta, rival: false };
        Self { base, faction, scoring }
    }

    fn for_guess(loc: Location, faction: &'a Faction) -> Self {
        let base = Threat::for_guess(loc);
        let scoring = Scoring { hp: 0., delta: 0, rival: false };
        Self { base, faction, scoring }
    }
}

//////////////////////////////////////////////////////////////////////////////

// ThreatState:

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum FightOrFlight { Fight, Flight, #[default] Safe }

#[derive(Default)]
pub struct Threats {
    pub uncertain: Vec<Threat>,
    pub menacing: Vec<Threat>,
    pub hostile: Vec<Threat>,
    pub unknown: Vec<Threat>,
}

#[derive(Default)]
pub struct ThreatState {
    // Threat types (not disjoint!):
    threats: Threats,

    // Fight-or-flight:
    pub state: FightOrFlight,
    pub last_safe: Timestamp,
    pub panicking: bool,

    // Calling for help:
    pub call_for_help: bool,
    pub last_call: Timestamp,

    // Internal state:
    factions: FactionMap,
}

impl std::ops::Deref for ThreatState {
    type Target = Threats;
    fn deref(&self) -> &Self::Target { &self.threats }
}

impl ThreatState {
    // TODO list for player interactions:
    //
    //   - We can turn the "scan unknown noise" subtree into just the first
    //     watch period for the first warning. ("auto-warn" unknown threats.)
    //
    //   - We should only update our valence after a warning after the watch
    //     period is complete.
    //
    //   - Can we make call-for-help carry information about target EID(s)?
    //     Seems hard - we can call based on unknown threats.
    //
    //   - Make the our-team-strength logic quadratic in team size.
    //
    //   - We currently count friendlies that we haven't seen for, say, 18
    //     turns as an ally with weight 0.75 - that's too high.
    //
    //   - When an actor is in combat it should prioritize threats that can
    //     hurt it (right now a predator may attack us even when it's under
    //     attack by prey, or vice-versa).
    //
    //   - We probably shouldn't call for help against threats we're not sure
    //     about (e.g. if we just warned a threat). But note that after warning
    //     a rival, we attack them even though we only have Confidence::Mid.

    pub fn debug(&self, debug: &mut DebugLog, known: &Knowledge) {
        for faction in &self.factions.factions {
            faction.debug(debug, known);
        }
    }

    pub fn mark_safe(&mut self, time: Timestamp) {
        if self.state == FightOrFlight::Safe { return; }

        if !self.panicking { self.last_safe = time; }

        self.state = FightOrFlight::Safe;
        self.panicking = false;
    }

    pub fn mark_scanned(&mut self) {
        for faction in &mut self.factions.factions {
            faction.mark_scanned();
        }
    }

    pub fn mark_warned(&mut self, me: &Actor, uid: UID, rng: &mut RNG) {
        let Some(faction) = self.factions.mut_faction(uid) else { return };
        faction.mark_warned(me, rng);
    }

    pub fn observe_event(&mut self, me: &Actor, event: &Event) {
        self.factions.observe_event(me, event);

        if let EventData::Call(x) = &event.data && Faction::friendly_call_for_help(me, x) {
            self.on_call_for_help(me, event.pos, event.time);
            self.factions.guess = Some(event.loc);
        }
    }

    pub fn on_call_for_help(&mut self, me: &Actor, point: Point, time: Timestamp) {
        let mut check = |uid: UID, pos: Point| {
            if !CALL_VOLUME.contains(pos - point) { return; }
            let Some(faction) = self.factions.mut_faction(uid) else { return };
            if faction.valence == Valence::Friendly { faction.last_combat = time; };
        };
        for actor in &me.known.actors { check(actor.uid, actor.pos); }
        for noise in &me.known.noises { check(noise.uid, noise.pos); }
        self.last_call = time;
    }

    pub fn update(&mut self, me: &Actor) {
        self.factions.update(me);
        self.threats = Default::default();

        let time = me.known.time();
        let decay = TimeDecay::new(me, ACTIVE_THREAT_TURNS);
        let call_limit = me.known.time_at_turn(CALL_LIMIT_TURNS);
        let call_retry = me.known.time_at_turn(CALL_RETRY_TURNS);

        // List all threats:

        let mut threats = vec![];
        for actor in &me.known.actors {
            if actor.time < decay.timeout { break };
            let Some(faction) = self.factions.get_faction(actor.uid) else { continue };
            threats.push(ScoredThreat::for_actor(actor, faction));
        }
        for noise in &me.known.noises {
            if noise.time < decay.timeout { break };
            let Some(faction) = self.factions.get_faction(noise.uid) else { continue };
            threats.push(ScoredThreat::for_noise(me, noise, faction));
        }
        let mut hostile = Faction::default();
        hostile.merge_status(Confidence::High, Valence::Hostile);
        if let Some(x) = self.factions.guess && x.time >= decay.timeout {
            threats.push(ScoredThreat::for_guess(x, &hostile));
        }
        threats.sort_by_key(|x| time - x.time);

        // Filter to active ones:

        let mut hidden_hostile = 0;
        let mut seen_hostile = 0;
        let mut panic = false;

        // List known enemies ("hostile") and potential enemies ("unknown").
        //
        // Every time we successfully flee from or fight back against all
        // known threats, we end an "epoch" by updating `last_safe`.
        for x in &threats {
            if x.time <= decay.timeout { break; }
            if x.time <= self.last_safe { break; }

            let foe = x.hostile() || x.menacing();

            if foe { self.threats.menacing.push(x.base.clone()); }
            if x.hostile() { self.threats.hostile.push(x.base.clone()); }
            if x.unknown() { self.threats.unknown.push(x.base.clone()); }
            if x.uncertain() { self.threats.uncertain.push(x.base.clone()); }

            if foe && !x.seen { hidden_hostile += 1; }
            if foe && x.seen { seen_hostile += 1; }

            panic |= x.uncertain() && x.penalty_strength(&decay) > 0.;
        }

        // Special case: flee from targets if we've just been scared by them.
        if panic && self.threats.menacing.is_empty() {
            self.threats.menacing.extend_from_slice(&self.threats.uncertain);
            self.threats.menacing.extend_from_slice(&self.threats.unknown);
            self.threats.menacing.sort_by_key(|x| time - x.time);

            self.state = FightOrFlight::Flight;
            self.call_for_help = false;
            self.panicking = true;
            return;
        }

        // Start fight-or-flight if we have an active known enemy. Stop when
        // we no longer have any known enemies.
        //
        // We also end fight-for-flight with known enemies if we lose sight
        // of them for long enough, where "long enough" is defined one way
        // when we're chasing them and another when we're fleeing from them.
        // See: callers of mark_safe().
        let was_active = self.state != FightOrFlight::Safe;
        let active = !self.threats.menacing.is_empty();

        // While active, also attack / flee from potential enemies.
        if active && !self.threats.menacing.is_empty() {
            self.threats.menacing.extend_from_slice(&self.threats.uncertain);
            self.threats.menacing.extend_from_slice(&self.threats.unknown);
            self.threats.menacing.sort_by_key(|x| time - x.time);
        }

        let mut hidden_count = max(hidden_hostile - seen_hostile, 0);
        let mut team_strength = me.hp_fraction();
        let mut call_strength = team_strength;
        let mut foes_strength = 0.;

        for x in &threats {
            if (x.hostile() || x.menacing()) && x.time > self.last_safe {
                if !x.seen && hidden_count == 0 { continue; }
                if !x.seen { hidden_count -= 1; }
                foes_strength += x.strength(&decay);
            } else if x.friendly() {
                let value = x.strength(&decay);
                let decay = (1. - decay.ratio(x.faction.last_combat)).max(0.);
                team_strength += decay * value;

                let recent = x.time > call_limit;
                let nearby = CALL_VOLUME.contains(me.pos - x.pos);
                call_strength += if nearby && recent { value } else { decay * value };
            }
        }

        let p = team_strength / (team_strength + foes_strength);
        let q = call_strength / (call_strength + foes_strength);

        if active && !was_active {
            self.state = if p > 0.5 { FightOrFlight::Fight } else { FightOrFlight::Flight };
        } else if active {
            if p > 0.6 { self.state = FightOrFlight::Fight; }
            if p < 0.4 { self.state = FightOrFlight::Flight; }
        } else {
            self.mark_safe(time);
        }
        self.panicking = false;

        self.call_for_help = false;
        if self.state == FightOrFlight::Flight && q > 0.6 && self.last_call <= call_retry {
            self.state = FightOrFlight::Fight;
            self.call_for_help = true;
        }
    }
}
