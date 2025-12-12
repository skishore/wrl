use std::cmp::{max, min};
use std::fmt::Debug;

use crate::base::{HashMap, Point, Slice};
use crate::entity::{Entity, EID};
use crate::knowledge::{AttackEvent, CallForHelpEvent, Event, EventData};
use crate::knowledge::{EntityKnowledge, Sense, Timedelta, Timestamp, UID};
use crate::game::CALL_VOLUME;
use crate::list::{Handle, List};

//////////////////////////////////////////////////////////////////////////////

pub const ACTIVE_THREAT_TIME: Timedelta = Timedelta::from_seconds(96.);

pub const CALL_FOR_HELP_LIMIT: Timedelta = Timedelta::from_seconds(4.);
pub const CALL_FOR_HELP_RETRY: Timedelta = Timedelta::from_seconds(24.);

//////////////////////////////////////////////////////////////////////////////

// Threat

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ThreatStatus { Hostile, Friendly, Neutral, Scanned, Unknown }

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
    pub seen: bool,

    // See status accessors:
    status: ThreatStatus,
}

impl Threat {
    fn prior(me: &Entity) -> Self {
        Self {
            pos: Default::default(),
            time: Default::default(),
            sense: Sense::Sight,
            combat: Default::default(),

            // Stats:
            hp: 1.,
            delta: if me.predator { -1 } else { 1 },

            // Flags:
            asleep: false,
            seen: false,

            // See status accessors:
            status: ThreatStatus::Unknown,
        }
    }

    // Status accessors:

    pub fn friendly(&self) -> bool {
        self.status == ThreatStatus::Friendly
    }

    pub fn hostile(&self) -> bool {
        self.status == ThreatStatus::Hostile
    }

    pub fn unknown(&self) -> bool {
        self.status == ThreatStatus::Unknown
    }

    pub fn mark_scanned(&mut self) {
        self.update_status(ThreatStatus::Scanned);
    }

    // State updates:

    fn update_status(&mut self, status: ThreatStatus) {
        self.status = min(self.status, status);
    }

    fn merge_from(&mut self, other: &Threat) {
        // No need to update any fields that we unconditionally update in
        // update_for_event, since we merge right before processing an event.
        self.sense = other.sense;
        self.seen |= other.seen;
        self.update_status(other.status);
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
                if x.combat { self.combat = event.time; }
            },
            EventData::CallForHelp(x) => {
                let for_us = ThreatState::call_for_us(me, x);
                if for_us { self.update_status(ThreatStatus::Friendly); }
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
    pub hostile: Vec<Threat>,
    pub unknown: Vec<Threat>,

    // Fight-or-flight.
    pub state: FightOrFlight,

    // Calling for help.
    pub called: Timestamp,
    pub call_for_help: bool,
}

impl ThreatState {
    pub fn debug(&self, slice: &mut Slice, time: Timestamp) {
        slice.write_str("ThreatState:").newline();
        slice.write_str(&format!("  state: {:?}", self.state)).newline();
        slice.write_str(&format!("  called: {:?}", time - self.called)).newline();
        slice.write_str(&format!("  call_for_help: {}", self.call_for_help)).newline();
    }

    pub fn on_call_for_help(&mut self, point: Point, time: Timestamp) {
        for threat in &mut self.threats {
            if !threat.friendly() { continue; }
            if (threat.pos - point).len_nethack() > CALL_VOLUME { continue; }
            threat.combat = time;
        }
        self.called = time;
    }

    pub fn update(&mut self, me: &Entity, prev_time: Timestamp) {
        for event in &me.known.events {
            let Some(threat) = self.get_by_event(me, event) else { continue };
            threat.update_for_event(me, event);
            if threat.hostile() { self.forget_tid(TID::CID); }

            if let EventData::CallForHelp(x) = &event.data && Self::call_for_us(me, x) {
                self.on_call_for_help(event.point, event.time);
                self.guess_threat_location(me, event);
            }
        }
        for other in &me.known.entities {
            if !other.visible { break; }
            let Some(threat) = self.get_by_entity(me, other.eid) else { continue };
            threat.update_for_sighting(me, other);
            if threat.hostile() { self.forget_tid(TID::CID); }
        }

        self.hostile.clear();
        self.unknown.clear();

        let was_active = self.state != FightOrFlight::Safe;
        let mut active = was_active;
        let mut hidden_hostile = 0;
        let mut seen_hostile = 0;

        // List known enemies ("hostile") and potential enemies ("unknown").
        for x in &self.threats {
            if me.known.time - x.time > ACTIVE_THREAT_TIME { break; }

            if x.hostile() { self.hostile.push(x.clone()); }
            if x.unknown() { self.unknown.push(x.clone()); }

            if x.hostile() && !x.seen { hidden_hostile += 1; }
            if x.hostile() && x.seen { seen_hostile += 1; }
        }

        // Start fight-or-flight if we have an active known enemy. Stop when
        // we no longer have any known enemies. We also stop it with known
        // enemies if we escape from them (see: UpdateFlightState).
        if let Some(x) = self.hostile.first() && x.time > prev_time {
            active = true;
        } else if self.hostile.is_empty() {
            active = false;
        }

        // While active, also attack / flee from potential enemies.
        if active && !self.hostile.is_empty() {
            self.hostile.extend_from_slice(&self.unknown);
            self.hostile.sort_by_key(|x| me.known.time - x.time);
        }

        let strength = |x: &Threat| { 1.5f64.powi(x.delta.signum()) * x.hp };
        let mut hidden_count = max(hidden_hostile - seen_hostile, 0);
        let mut team_strength = me.hp_fraction();
        let mut call_strength = team_strength;
        let mut foes_strength = 0.;

        for x in &self.threats {
            if me.known.time - x.time > ACTIVE_THREAT_TIME { break; }

            if x.hostile() {
                if !x.seen && hidden_count == 0 { continue; }
                if !x.seen { hidden_count -= 1; }
                foes_strength += strength(x);
            } else if x.friendly() {
                let base = strength(x);
                let delay = me.known.time - x.combat;
                let ratio = delay.nsec() as f64 / ACTIVE_THREAT_TIME.nsec() as f64;
                let decay = if ratio > 1. { 0. } else { 1. - ratio };
                team_strength += base * decay;

                let nearby = (me.pos - x.pos).len_nethack() <= CALL_VOLUME;
                let recent = me.known.time - x.time <= CALL_FOR_HELP_LIMIT;
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
        if self.state == FightOrFlight::Flight && q > 0.6 &&
           (me.known.time - self.called) > CALL_FOR_HELP_RETRY {
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

    fn call_for_us(me: &Entity, _: &CallForHelpEvent) -> bool {
        // TODO(shaunak): Implement a better "this call is for us" check.
        !me.predator
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
