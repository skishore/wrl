use std::cmp::max;

use rand::Rng;

use crate::base::{HashMap, Point, RNG};
use crate::debug::DebugLog;
use crate::entity::{Entity, EID};
use crate::knowledge::{AttackEvent, Call, CallEvent, Event, EventData, Sense};
use crate::knowledge::{EntityKnowledge, Knowledge, Timestamp, UID};
use crate::game::CALL_VOLUME;
use crate::list::{Handle, List};

//////////////////////////////////////////////////////////////////////////////

pub const ACTIVE_THREAT_TURNS: i32 = 72;

pub const CALL_LIMIT_TURNS: i32 = 4;
pub const CALL_RETRY_TURNS: i32 = 16;

//////////////////////////////////////////////////////////////////////////////

// Threat

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Confidence { Low, Mid, High }

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Valence { Friendly, Neutral, Menacing, Hostile }

#[derive(Clone)]
pub struct Threat {
    pub pos: Point,
    pub time: Timestamp,
    pub sense: Sense,
    pub combat: Timestamp,

    // Stats:
    pub hp: f64,
    pub delta: i32,

    // Flags:
    pub asleep: bool,
    pub human: bool,
    pub seen: bool,

    // Warnings:
    timid: bool,
    warnings: i32,

    // See status accessors:
    confidence: Confidence,
    valence: Valence,
}

impl Threat {
    fn prior(me: &Entity) -> Self {
        Self {
            pos: Default::default(),
            time: Default::default(),
            sense: Sense::Sight,
            combat: Default::default(),

            // Stats:
            hp: 0.,
            delta: if me.predator { -1 } else { 1 },

            // Flags:
            asleep: false,
            human: false,
            seen: false,

            // Warnings:
            timid: !me.predator,
            warnings: 0,

            // See status accessors:
            confidence: Confidence::Low,
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

    pub fn unknown(&self) -> bool {
        self.confidence == Confidence::Low
    }

    pub fn mark_warned(&mut self, rng: &mut RNG) {
        let warnings = self.warnings;
        let sample = rng.random::<f32>() * 2f32.powi(warnings);

        if sample < 0.25 {
            let valence = if self.timid { Valence::Menacing } else { Valence::Hostile };
            self.merge_status(Confidence::Mid, valence);
        } else if sample >= 0.75 {
            self.merge_status(Confidence::Mid, Valence::Hostile);
        }
        self.warnings += 1;
    }

    pub fn mark_scanned(&mut self) {
        self.merge_status(Confidence::Low, self.valence);
    }

    // TODO list for player interactions:
    //
    //   - Wilds are not appropriately hostile, except on this branch. They
    //     spend too much time warning or scanning unknown noises.
    //
    //   - We can turn the "scan unknown noise" subtree into just the first
    //     watch period for the first warning. ("auto-warn" unknown threats.)
    //
    //   - We should only update our valence after a warning after the watch
    //     period is complete.
    //
    //   - Wilds may attack player; however, if they spot a predator, flee
    //     from it, and then spot a player within ACTIVE_THREAT_TURNS, we'll
    //     incorrectly flee from the player because of the other threat.
    //
    //   - Can we make scent just another event type? Can we tie scent to an
    //     EID / TID and disable further updates once we identify it?
    //
    //   - Can we make call-for-help carry information about target EID(s)?
    //     Seems hard - we can call based on unknown threats.

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
        self.pos = event.point;
        self.time = event.time;
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
                if ThreatState::call_for_us(me, x) {
                    self.merge_status(Confidence::High, Valence::Friendly);
                } else if x.call == Call::Warning {
                    let valence = if self.timid { Valence::Menacing } else { Valence::Hostile };
                    self.merge_status(Confidence::Mid, valence);
                }
                if x.call == Call::Help { self.mark_combat(event.time); }
            },
            EventData::Attack(_) => {},
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
        self.human = other.species.human;
        self.seen = true;

        let (confidence, valence) = if self.human {
            (Confidence::Low, Valence::Neutral)
        } else if other.delta > 0 {
            let combat = self.combat > me.known.time_at_turn(ACTIVE_THREAT_TURNS);
            let valence = if combat { Valence::Hostile } else { Valence::Menacing };
            (Confidence::High, valence)
        } else if !me.predator && me.species == other.species {
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
        });
        debug.newline();

        for threat in &self.threats { threat.debug(debug, known); }
    }

    pub fn on_call_for_help(&mut self, point: Point, time: Timestamp) {
        for threat in &mut self.threats {
            if !threat.friendly() { continue; }
            if (threat.pos - point).len_nethack() > CALL_VOLUME { continue; }
            threat.combat = time;
        }
        self.last_call = time;
    }

    pub fn update(&mut self, me: &Entity, prev_time: Timestamp) {
        for event in &me.known.events {
            let Some(threat) = self.get_by_event(me, event) else { continue };
            threat.update_for_event(me, event);
            if threat.certain() && threat.hostile() { self.forget_tid(TID::CID); }

            if let EventData::Call(x) = &event.data && Self::call_for_us(me, x) {
                self.on_call_for_help(event.point, event.time);
                self.guess_threat_location(me, event);
            }
        }
        for other in &me.known.entities {
            if !other.visible { break; }
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

        let was_active = self.state != FightOrFlight::Safe;
        let mut active = was_active;
        let mut hidden_hostile = 0;
        let mut seen_hostile = 0;

        // List known enemies ("hostile") and potential enemies ("unknown").
        for x in &self.threats {
            if x.time <= limit { break; }

            let menacing = x.menacing();
            let hostile = x.hostile();
            let unknown = x.unknown();
            let foe = hostile || menacing;

            if foe { self.menacing.push(x.clone()); }
            if hostile { self.hostile.push(x.clone()); }
            if unknown { self.unknown.push(x.clone()); }

            if foe && !x.seen { hidden_hostile += 1; }
            if foe && x.seen { seen_hostile += 1; }
        }

        // Start fight-or-flight if we have an active known enemy. Stop when
        // we no longer have any known enemies. We also stop it with known
        // enemies if we escape from them (see: UpdateFlightState).
        //
        // FATAL flaw: if we decide a threat is menacing because of our own
        // action, we may fail the "x.time > prev_time" check here...
        if let Some(x) = self.menacing.first() && x.time > prev_time {
            active = true;
        } else if self.menacing.is_empty() {
            active = false;
        }

        // While active, also attack / flee from potential enemies.
        if active && !self.menacing.is_empty() {
            self.menacing.extend_from_slice(&self.unknown);
            self.menacing.sort_by_key(|x| me.known.time - x.time);
        }
        if active && !self.hostile.is_empty() {
            self.hostile.extend_from_slice(&self.unknown);
            self.hostile.sort_by_key(|x| me.known.time - x.time);
        }

        let strength = |x: &Threat| {
            if x.human { 0. } else { 1.75f64.powi(x.delta.signum()) * x.hp }
        };
        let mut hidden_count = max(hidden_hostile - seen_hostile, 0);
        let mut team_strength = me.hp_fraction();
        let mut call_strength = team_strength;
        let mut foes_strength = 0.;

        for x in &self.threats {
            if x.time <= limit { break; }

            if x.hostile() || x.menacing() {
                if !x.seen && hidden_count == 0 { continue; }
                if !x.seen { hidden_count -= 1; }
                foes_strength += strength(x);
            } else if x.friendly() {
                let base = strength(x);
                let denom = me.known.time - limit;
                let delay = me.known.time - x.combat;
                let ratio = delay.nsec() as f64 / max(denom.nsec(), 1) as f64;
                let decay = if ratio > 1. { 0. } else { 1. - ratio };
                team_strength += base * decay;

                let recent = x.time > call_limit;
                let nearby = (me.pos - x.pos).len_nethack() <= CALL_VOLUME;
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
            self.state = FightOrFlight::Safe;
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

        if let EventData::Forget(_) = &event.data {
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
        x == me.eid || me.known.entity(x).map(|x| x.friend).unwrap_or(false)
    }

    fn call_for_us(me: &Entity, x: &CallEvent) -> bool {
        x.call == Call::Help && x.species == me.species
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
