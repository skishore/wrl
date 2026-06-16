use std::cmp::max;

use rand::Rng;

use crate::base::point::Point;
use crate::base::util::{HashMap, RNG};

use super::debug::DebugLog;
use super::dex::Species;
use super::entity::{Entity, EID};
use super::event::{AttackEvent, Call, CallEvent, Event, EventData, Location, Sense, UID};
use super::knowledge::{EntityKnowledge, Knowledge};
use super::game::CALL_VOLUME;
use super::list::{Handle, List};
use super::time::Timestamp;

//////////////////////////////////////////////////////////////////////////////

pub const ACTIVE_THREAT_TURNS: i32 = 72;

pub const CALL_LIMIT_TURNS: i32 = 4;
pub const CALL_RETRY_TURNS: i32 = 16;

fn timid(me: &Entity) -> bool { !me.species.predator() }

//////////////////////////////////////////////////////////////////////////////

// Threat

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Confidence { Zero, Low, Mid, High }

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Valence { Friendly, Neutral, Menacing, Hostile }

#[derive(Clone)]
pub struct Threat {
    pub loc: Location,
    pub sense: Sense,
    pub combat: Timestamp,
    pub species: Option<&'static Species>,

    // Stats:
    hp: f64,
    delta: i32,

    // Flags:
    pub asleep: bool,
    pub rival: bool,
    pub seen: bool,

    // Danger penalty:
    penalty_score: f64,
    penalty_start: Timestamp,

    // Warnings:
    warnings: i32,

    // See status accessors:
    confidence: Confidence,
    valence: Valence,
}

impl std::ops::Deref for Threat {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl Threat {
    fn prior(me: &Entity) -> Self {
        Self {
            loc: Default::default(),
            sense: Sense::Sight,
            combat: Default::default(),
            species: None,

            // Stats:
            hp: 0.,
            delta: if timid(me) { 1 } else { -1 },

            // Flags:
            asleep: false,
            rival: false,
            seen: false,

            // Danger penalty:
            penalty_score: Default::default(),
            penalty_start: Default::default(),

            // Warnings:
            warnings: 0,

            // See status accessors:
            confidence: Confidence::Zero,
            valence: Valence::Neutral,
        }
    }

    pub fn debug(&self, debug: &mut DebugLog, known: &Knowledge) {
        debug.append("Threat:");
        debug.indent(1, |debug| {
            let warnings = if self.warnings == 0 {
                " ".into()
            } else if self.warnings == 1 {
                " - 1 warning".into()
            } else {
                format!(" - {} warnings", self.warnings)
            };
            let status = format!("{:?}:{:?}", self.confidence, self.valence);

            debug.append(format!("age: {}", known.debug_time(self.time)));
            debug.append(format!("pos: {:?}, by {:?}", self.pos, self.sense));
            debug.append(format!("combat: {}", known.debug_time(self.combat)));
            debug.append(format!("status: {}{}", status, warnings));
            debug.append(format!("strength: {} @ {:.2} hp", self.delta, self.hp));
        });
        debug.newline();
    }

    // Status accessors:

    pub fn friendly(&self) -> bool {
        self.valence == Valence::Friendly
    }

    pub fn hostile(&self) -> bool {
        self.valence == Valence::Hostile
    }

    pub fn menacing(&self) -> bool {
        self.valence == Valence::Menacing
    }

    pub fn certain(&self) -> bool {
        self.confidence == Confidence::High
    }

    pub fn uncertain(&self) -> bool {
        self.confidence == Confidence::Low
    }

    pub fn unknown(&self) -> bool {
        self.confidence == Confidence::Zero
    }

    pub fn mark_warned(&mut self, me: &Entity, rng: &mut RNG) {
        if !self.uncertain() { return; }

        let warnings = self.warnings;
        let sample = rng.random::<f32>() * 2f32.powi(warnings);

        if sample > 0.5 {
            self.merge_status(Confidence::Mid, Valence::Hostile);

            if timid(me) {
                self.penalty_score = rng.random_range(2.0..4.0);
                self.penalty_start = me.known.time();
            }
        }
        self.warnings += 1;
    }

    pub fn mark_scanned(&mut self) {
        self.merge_status(Confidence::Low, self.valence);
    }

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
    //   - We currently count friendlies that we haven't seen for, say, 18
    //     turns as an ally with weight 0.75 - that's too high.
    //
    //   - When an entity is in combat it should prioritize threats that can
    //     hurt it (right now a predator may attack us even when it's under
    //     attack by prey, or vice-versa).

    // State updates:

    fn merge_status(&mut self, confidence: Confidence, valence: Valence) {
        if confidence == self.confidence {
            self.valence = max(valence, self.valence);
        } else if confidence > self.confidence {
            self.confidence = confidence;
            self.valence = valence;
        }
    }

    fn merge_from(&mut self, other: &Threat) {
        // No need to update any fields that we unconditionally update in
        // update_for_event, since we merge right before processing an event.
        self.seen |= other.seen;
        self.combat = max(self.combat, other.combat);
        self.warnings = max(self.warnings, other.warnings);
        self.merge_status(other.confidence, other.valence);
    }

    fn mark_combat(&mut self, time: Timestamp) {
        if !self.seen { self.hp = 1. };
        self.combat = time;
    }

    fn update_for_event(&mut self, me: &Entity, event: &Event) {
        self.loc = event.loc;
        self.sense = event.sense;
        self.asleep = false;

        match &event.data {
            EventData::Attack(x) if x.combat => {
                if x.target == Some(me.eid) || (self.certain() && self.menacing()) {
                    self.merge_status(Confidence::High, Valence::Hostile);
                } else {
                    self.merge_status(Confidence::Mid, Valence::Hostile);
                }
                self.mark_combat(event.time);
            },
            EventData::Call(x) => {
                if ThreatState::friendly_call(me, x) {
                    self.merge_status(Confidence::High, Valence::Friendly);
                } else if x.call == Call::Warning {
                    let valence = if timid(me) { Valence::Menacing } else { Valence::Neutral };
                    self.merge_status(Confidence::Mid, valence);
                }
                if x.call == Call::Help { self.mark_combat(event.time); }
            },
            EventData::Attack(_) => {},
            EventData::Move(_) => {},
            EventData::Forget => {},
            EventData::Sniff => {},
            EventData::Spot => {},
        }
    }

    fn update_for_sighting(&mut self, me: &Entity, other: &EntityKnowledge) {
        self.loc = other.loc;
        self.sense = other.sense;
        self.species = Some(other.species);

        self.hp = other.hp;
        self.delta = other.delta;

        self.asleep = other.asleep();
        self.rival = other.rival();
        self.seen = true;

        let (confidence, valence) =
        if other.friend() {
            (Confidence::High, Valence::Friendly)
        } else if other.species.human() {
            (Confidence::Low, Valence::Neutral)
        } else if other.rival() {
            (Confidence::High, Valence::Hostile)
        } else if other.delta > 0 {
            let combat = self.combat > me.known.time_at_turn(ACTIVE_THREAT_TURNS);
            let valence = if combat { Valence::Hostile } else { Valence::Menacing };
            (Confidence::High, valence)
        } else if timid(me) && me.species == other.species {
            (Confidence::High, Valence::Friendly)
        } else {
            (Confidence::High, Valence::Neutral)
        };
        self.merge_status(confidence, valence);
    }
}

//////////////////////////////////////////////////////////////////////////////

// ThreatState

pub type ThreatHandle = Handle<Threat>;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum TID { CID, EID(EID), UID(UID) }

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum FightOrFlight { Fight, Flight, #[default] Safe }

#[derive(Default)]
pub struct ThreatState {
    pub threats: List<Threat>,
    pub threat_index: HashMap<TID, ThreatHandle>,

    // Summaries used for flight pathing.
    pub menacing: Vec<Threat>,
    pub hostile: Vec<Threat>,
    pub unknown: Vec<Threat>,

    // Fight-or-flight.
    pub state: FightOrFlight,
    pub last_safe: Timestamp,

    // Calling for help.
    pub call_for_help: bool,
    pub last_call: Timestamp,
}

impl ThreatState {
    pub fn debug(&self, debug: &mut DebugLog, known: &Knowledge) {
        debug.append("ThreatState:");
        debug.indent(1, |debug| {
            debug.append(format!("state: {:?}", self.state));
            debug.append(format!("call_for_help: {}", self.call_for_help));
            debug.append(format!("last_call: {}", known.debug_time(self.last_call)));
            debug.append(format!("last_safe: {}", known.debug_time(self.last_safe)));
        });
        debug.newline();

        for threat in &self.threats { threat.debug(debug, known); }
    }

    pub fn mark_safe(&mut self, time: Timestamp) {
        if self.state == FightOrFlight::Safe { return; }
        self.state = FightOrFlight::Safe;
        self.last_safe = time;
    }

    pub fn on_call_for_help(&mut self, point: Point, time: Timestamp) {
        for threat in &mut self.threats {
            if !threat.friendly() { continue; }
            if !CALL_VOLUME.contains(threat.pos - point) { continue; }
            threat.combat = time;
        }
        self.last_call = time;
    }

    pub fn update(&mut self, me: &Entity) {
        let time = me.known.time();

        for event in &me.known.events {
            let Some(threat) = self.get_by_event(me, event) else { continue };
            threat.update_for_event(me, event);
            if threat.certain() && threat.hostile() { self.forget_tid(TID::CID); }

            if let EventData::Call(x) = &event.data && Self::friendly_call_for_help(me, x) {
                self.on_call_for_help(event.pos, event.time);
                self.guess_threat_location(me, event);
            }
        }
        for other in &me.known.entities {
            if !other.sensed() { break; }
            let Some(threat) = self.get_by_entity(me, other.eid) else { continue };
            threat.update_for_sighting(me, other);
            if threat.certain() && threat.hostile() { self.forget_tid(TID::CID); }
        }

        self.menacing.clear();
        self.hostile.clear();
        self.unknown.clear();

        let limit = me.known.time_at_turn(ACTIVE_THREAT_TURNS);
        let call_limit = me.known.time_at_turn(CALL_LIMIT_TURNS);
        let call_retry = me.known.time_at_turn(CALL_RETRY_TURNS);

        let mut uncertain = vec![];
        let mut hidden_hostile = 0;
        let mut seen_hostile = 0;

        // List known enemies ("hostile") and potential enemies ("unknown").
        //
        // Every time we successfully flee from or fight back against all
        // known threats, we end an "epoch" by updating `last_safe`.
        for x in &self.threats {
            if x.time <= limit { break; }
            if x.time <= self.last_safe { break; }

            let foe = x.hostile() || x.menacing();

            if foe { self.menacing.push(x.clone()); }
            if x.hostile() { self.hostile.push(x.clone()); }
            if x.unknown() { self.unknown.push(x.clone()); }
            if x.uncertain() { uncertain.push(x.clone()); }

            if foe && !x.seen { hidden_hostile += 1; }
            if foe && x.seen { seen_hostile += 1; }
        }

        // Start fight-or-flight if we have an active known enemy. Stop when
        // we no longer have any known enemies.
        //
        // We also end fight-for-flight with known enemies if we lose sight
        // of them for long enough, where "long enough" is defined one way
        // when we're chasing them and another when we're fleeing from them.
        // See: callers of mark_safe().
        let was_active = self.state != FightOrFlight::Safe;
        let active = !self.menacing.is_empty();

        // While active, also attack / flee from potential enemies.
        if active && !self.menacing.is_empty() {
            self.menacing.extend_from_slice(&uncertain);
            self.menacing.extend_from_slice(&self.unknown);
            self.menacing.sort_by_key(|x| time - x.time);
        }
        if active && !self.hostile.is_empty() {
            self.hostile.extend_from_slice(&self.unknown);
            self.hostile.sort_by_key(|x| time - x.time);
        }

        // Compute a strength. For some entities that start by responding to
        // threats by fleeing from them, we'll add an additive penalty.
        let base_strength = |x: &Threat| {
            if let Some(x) = x.species && x.human() { return 0.; }
            let factor = if x.rival { 0.25 } else { 1. };
            factor * 1.75f64.powi(x.delta.signum()) * x.hp
        };
        let strength = |x: &Threat| {
            let base = base_strength(x);
            if x.penalty_score <= 0. || x.penalty_start < limit { return base; }

            let denom = time - limit;
            let delay = time - x.penalty_start;
            let ratio = delay.nsec() as f64 / max(denom.nsec(), 1) as f64;
            let bonus = x.penalty_score - 0.1 * ratio * ACTIVE_THREAT_TURNS as f64;
            base + bonus.max(0.)
        };
        let mut hidden_count = max(hidden_hostile - seen_hostile, 0);
        let mut team_strength = me.hp_fraction();
        let mut call_strength = team_strength;
        let mut foes_strength = 0.;

        for x in &self.threats {
            if x.time <= limit { break; }

            if (x.hostile() || x.menacing()) && x.time > self.last_safe {
                if !x.seen && hidden_count == 0 { continue; }
                if !x.seen { hidden_count -= 1; }
                foes_strength += strength(x);
            } else if x.friendly() {
                let base = strength(x);
                let denom = time - limit;
                let delay = time - x.combat;
                let ratio = delay.nsec() as f64 / max(denom.nsec(), 1) as f64;
                let decay = 1. - ratio.min(1.);
                team_strength += base * decay;

                let recent = x.time > call_limit;
                let nearby = CALL_VOLUME.contains(me.pos - x.pos);
                call_strength += if nearby && recent { base } else { base * decay };
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

        self.call_for_help = false;
        if self.state == FightOrFlight::Flight && q > 0.6 && self.last_call <= call_retry {
            self.state = FightOrFlight::Fight;
            self.call_for_help = true;
        }

        debug_assert!(self.check_invariants());
    }

    fn forget_tid(&mut self, tid: TID) {
        let Some(handle) = self.threat_index.remove(&tid) else { return };
        self.threats.remove(handle);
    }

    fn get_by_entity(&mut self, me: &Entity, eid: EID) -> Option<&mut Threat> {
        let handle = self.get_by_tid(me, TID::EID(eid))?;
        Some(&mut self.threats[handle])
    }

    fn get_by_event(&mut self, me: &Entity, event: &Event) -> Option<&mut Threat> {
        let tid = event.eid.map(|x| TID::EID(x)).or(event.uid.map(|x| TID::UID(x)))?;

        if matches!(event.data, EventData::Forget) {
            self.forget_tid(tid);
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
        x == me.eid || me.known.entity(x).map(|x| x.friend()).unwrap_or(false)
    }

    fn friendly_call(me: &Entity, x: &CallEvent) -> bool {
       x.species == me.species
    }

    fn friendly_call_for_help(me: &Entity, x: &CallEvent) -> bool {
        x.call == Call::Help && Self::friendly_call(me, x)
    }

    fn guess_threat_location(&mut self, me: &Entity, event: &Event) {
        let Some(handle) = self.get_by_tid(me, TID::CID) else { return };

        let mut attack = event.clone();
        let data = AttackEvent { combat: true, target: Some(me.eid) };
        attack.data = EventData::Attack(data);

        let threat = &mut self.threats[handle];
        threat.update_for_event(me, &attack);
        threat.hp = 0.;
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
